//! Parity ports for the runner-method variants of `tests/memory/test_session.py` (`run_sync`
//! and `run_streamed`, which `py_memory_session.rs` ports only for `run`) and for the
//! remaining `tests/memory/test_openai_conversations_session.py` cases that are observable
//! through the public API.

#![cfg(feature = "sqlite")]

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::memory::{OpenAIConversationsSession, Session, SqliteSession};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text, serve};

fn fixtures() -> (Arc<FakeModel>, Agent<()>) {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    (model, agent)
}

// ---------------------------------------------------------------------------
// test_session.py: `test_session_memory_basic_functionality_parametrized` for run_sync and
// run_streamed (the `run` variant is in py_memory_session.rs).
// ---------------------------------------------------------------------------

/// `run_sync`: the synchronous entry point must also read and write the session.
#[test]
fn session_memory_round_trips_through_run_sync() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(SqliteSession::open("sync_session", dir.path().join("s.db")).unwrap());
    let (model, agent) = fixtures();

    model.add_turn(vec![assistant_message("San Francisco")]);
    let first = Runner::run_sync_with_options(
        &agent,
        "What city is the Golden Gate Bridge in?",
        RunOptions::new(()).session(session.clone()),
    )
    .unwrap();
    assert_eq!(first.final_output_text().as_deref(), Some("San Francisco"));

    model.add_turn(vec![assistant_message("California")]);
    let second = Runner::run_sync_with_options(
        &agent,
        "What state is it in?",
        RunOptions::new(()).session(session.clone()),
    )
    .unwrap();
    assert_eq!(second.final_output_text().as_deref(), Some("California"));

    let last = model.last_call().unwrap();
    assert!(last.input.len() > 1, "{:?}", last.input);
}

/// `run_streamed`: the stream must persist the turn and the next run must see it.
#[tokio::test]
async fn session_memory_round_trips_through_run_streamed() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        Arc::new(SqliteSession::open("streamed_session", dir.path().join("s.db")).unwrap());
    let (model, agent) = fixtures();

    model.add_turn(vec![assistant_message("San Francisco")]);
    let mut first = Runner::run_streamed_with_options(
        &agent,
        "What city is the Golden Gate Bridge in?",
        RunOptions::new(()).session(session.clone()),
    );
    {
        let mut events = std::pin::pin!(first.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    first.finish().await.unwrap();

    model.add_turn(vec![assistant_message("California")]);
    let second = Runner::run_with_options(
        &agent,
        "What state is it in?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(second.final_output_text().as_deref(), Some("California"));

    let last = model.last_call().unwrap();
    assert!(last.input.len() > 1, "{:?}", last.input);
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
}

// ---------------------------------------------------------------------------
// test_openai_conversations_session.py (remaining, observable cases)
// ---------------------------------------------------------------------------

/// Conversations mock: `POST /conversations` fails on the first call only; item writes succeed.
async fn flaky_creation_mock() -> (OpenAIClient, Arc<Mutex<Vec<Request>>>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let creations = Arc::new(Mutex::new(0usize));
    let base = serve(Arc::new(move |req: &Request| {
        cap.lock().unwrap().push(req.clone());
        if req.method == "POST" && req.path.ends_with("/conversations") {
            let mut n = creations.lock().unwrap();
            *n += 1;
            if *n == 1 {
                return Reply::status(500, "{\"error\": {\"message\": \"creation failed\"}}");
            }
            return Reply::json(json!({"id": "conv_recovered"}));
        }
        Reply::json(json!({"data": []}))
    }))
    .await;
    (
        OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")),
        captured,
    )
}

/// `test_concurrent_first_write_recovers_after_creation_failure`: a failed lazy creation leaves
/// no conversation id behind, so the next write creates one and succeeds.
#[tokio::test]
async fn a_failed_first_creation_is_retried_by_the_next_write() {
    let (client, captured) = flaky_creation_mock().await;
    let session = OpenAIConversationsSession::with_client(client);

    assert!(
        session
            .add_items(vec![InputItem::user("first")])
            .await
            .is_err()
    );
    assert!(session.try_session_id().is_err());

    session
        .add_items(vec![InputItem::user("second")])
        .await
        .unwrap();
    assert_eq!(session.try_session_id().unwrap(), "conv_recovered");

    let creates = captured
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.method == "POST" && r.path.ends_with("/conversations"))
        .count();
    assert_eq!(creates, 2);
}

/// `test_operations_wait_for_complete_append`: concurrent appends through one session are
/// serialised, so no write is lost or interleaved mid-batch.
#[tokio::test]
async fn concurrent_appends_through_one_session_lose_nothing() {
    let (client, captured) = flaky_creation_mock().await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_x", client,
    ));

    let mut tasks = Vec::new();
    for i in 0..8 {
        let s = session.clone();
        tasks.push(tokio::spawn(async move {
            s.add_items(vec![InputItem::user(format!("w{i}"))])
                .await
                .unwrap();
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    let mut delivered: Vec<String> = captured
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.method == "POST" && r.path.contains("/items"))
        .flat_map(|r| {
            r.body["items"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|i| i["content"].as_str().unwrap_or_default().to_string())
        })
        .collect();
    delivered.sort();
    let mut expected: Vec<String> = (0..8).map(|i| format!("w{i}")).collect();
    expected.sort();
    assert_eq!(delivered, expected);
}

/// `test_get_items_zero_limit_returns_empty_without_api_call` is ported in
/// `py_memory_conversations.rs`; the item-by-item text of the stored history is checked here
/// through a round trip of the mock's page.
#[tokio::test]
async fn conversation_history_page_is_returned_oldest_first() {
    let page = json!({
        "data": [
            {"id": "i2", "role": "assistant", "content": "second"},
            {"id": "i1", "role": "user", "content": "first"},
        ],
        "has_more": false,
    });
    let base = serve(Arc::new(move |_req: &Request| Reply::json(page.clone()))).await;
    let client = OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"));
    let session = OpenAIConversationsSession::from_conversation_id("conv_page", client);

    let items = session.get_items(Some(2)).await.unwrap();
    let texts: Vec<String> = items.iter().map(item_text).collect();
    assert_eq!(texts, vec!["first", "second"]);
    assert!(items.iter().all(|i| i.as_value().get("id").is_none()));
}
