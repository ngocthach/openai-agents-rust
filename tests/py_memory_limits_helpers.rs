//! Parity port of `tests/memory/test_session_limit.py`, `tests/memory/test_session_helpers.py`
//! and `tests/utils/test_simple_session.py`.
#![cfg(feature = "sqlite")]

use std::sync::Arc;

use openai_agents_sdk::memory::{
    Session, SessionSettings, SqliteSession, await_session_mutation, resolve_session_limit,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::{ListSession, item_text};

fn user(content: &str) -> InputItem {
    InputItem::user(content)
}

fn contents(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

async fn run_turn(
    agent: &Arc<Agent<()>>,
    model: &Arc<FakeModel>,
    session: &Arc<SqliteSession>,
    input: &str,
    reply: &str,
    settings: Option<SessionSettings>,
) {
    model.add_turn(vec![assistant_message(reply)]);
    let mut options = RunOptions::new(()).session(session.clone());
    if let Some(settings) = settings {
        options = options.run_config(RunConfig::<()>::default().with_session_settings(settings));
    }
    Runner::run_with_options(agent, input, options)
        .await
        .unwrap();
}

fn fixtures() -> (Arc<Agent<()>>, Arc<FakeModel>) {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    (agent, model)
}

// ---------------------------------------------------------------------------
// tests/memory/test_session_limit.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn session_limit_parameter_limits_the_history_the_model_sees() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(SqliteSession::open("limit_test", dir.path().join("l.db")).unwrap());
    let (agent, model) = fixtures();

    run_turn(&agent, &model, &session, "Message 1", "Reply 1", None).await;
    run_turn(&agent, &model, &session, "Message 2", "Reply 2", None).await;
    run_turn(&agent, &model, &session, "Message 3", "Reply 3", None).await;
    assert_eq!(session.get_items(None).await.unwrap().len(), 6);

    run_turn(
        &agent,
        &model,
        &session,
        "Message 4",
        "Reply 4",
        Some(SessionSettings::with_limit(2)),
    )
    .await;

    let seen = contents(&model.last_call().unwrap().input);
    assert_eq!(seen.len(), 3, "{seen:?}");
    assert_eq!(seen[0], "Message 3");
    assert_eq!(seen[1], "Reply 3");
    assert_eq!(seen[2], "Message 4");
}

#[tokio::test]
async fn session_limit_zero_sends_only_the_new_message() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        Arc::new(SqliteSession::open("limit_zero_test", dir.path().join("z.db")).unwrap());
    let (agent, model) = fixtures();

    run_turn(&agent, &model, &session, "Message 1", "Reply 1", None).await;
    run_turn(&agent, &model, &session, "Message 2", "Reply 2", None).await;
    run_turn(
        &agent,
        &model,
        &session,
        "Message 3",
        "Reply 3",
        Some(SessionSettings { limit: Some(0) }),
    )
    .await;

    let seen = contents(&model.last_call().unwrap().input);
    assert_eq!(seen, vec!["Message 3"]);
}

#[tokio::test]
async fn an_unset_session_limit_sends_the_whole_history() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        Arc::new(SqliteSession::open("limit_none_test", dir.path().join("n.db")).unwrap());
    let (agent, model) = fixtures();

    for i in 1..=5 {
        run_turn(
            &agent,
            &model,
            &session,
            &format!("Message {i}"),
            &format!("Reply {i}"),
            None,
        )
        .await;
    }
    assert_eq!(session.get_items(None).await.unwrap().len(), 10);

    run_turn(
        &agent,
        &model,
        &session,
        "Message 6",
        "Reply 6",
        Some(SessionSettings { limit: None }),
    )
    .await;

    let seen = contents(&model.last_call().unwrap().input);
    assert_eq!(seen.len(), 11, "{seen:?}");
    assert_eq!(seen[0], "Message 1");
    assert_eq!(seen[10], "Message 6");
}

#[tokio::test]
async fn a_limit_larger_than_the_history_returns_everything() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        Arc::new(SqliteSession::open("limit_large_test", dir.path().join("big.db")).unwrap());
    let (agent, model) = fixtures();

    run_turn(&agent, &model, &session, "Message 1", "Reply 1", None).await;
    run_turn(
        &agent,
        &model,
        &session,
        "Message 2",
        "Reply 2",
        Some(SessionSettings::with_limit(100)),
    )
    .await;

    let seen = contents(&model.last_call().unwrap().input);
    assert_eq!(seen, vec!["Message 1", "Reply 1", "Message 2"]);
}

/// Python's `drop_orphan_function_calls` takes `output_pruning_indexes` and removes a
/// `function_call_output` whose `function_call` the session limit cut off, so the orphaned output
/// never reaches the model.
#[tokio::test]
async fn a_limit_boundary_drops_an_unmatched_function_call_output() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        Arc::new(SqliteSession::open("limit_tool_pair", dir.path().join("tp.db")).unwrap());
    let history = vec![
        user("What is the weather?"),
        InputItem::from_value(json!({
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{}",
        })),
        InputItem::from_value(json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "sunny",
        })),
        InputItem::from_value(json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "It is sunny.", "annotations": []}],
        })),
    ];
    session.add_items(history.clone()).await.unwrap();

    // The store itself returns the raw last two items.
    let tail = session.get_items(Some(2)).await.unwrap();
    assert_eq!(tail.len(), 2);
    assert_eq!(
        tail[0].as_value().get("type").and_then(|t| t.as_str()),
        Some("function_call_output")
    );

    let (agent, model) = fixtures();
    model.add_turn(vec![assistant_message("Tomorrow is sunny too.")]);
    Runner::run_with_options(
        &agent,
        "What about tomorrow?",
        RunOptions::new(()).session(session.clone()).run_config(
            RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(2)),
        ),
    )
    .await
    .unwrap();

    // The orphaned function_call_output must not reach the model.
    let input = model.last_call().unwrap().input;
    let types: Vec<Option<String>> = input
        .iter()
        .map(|i| {
            i.as_value()
                .get("type")
                .and_then(|t| t.as_str())
                .map(str::to_string)
        })
        .collect();
    assert!(
        !types.contains(&Some("function_call_output".to_string())),
        "orphan tool output reached the model: {types:?}"
    );
    assert_eq!(
        contents(&input).last().map(String::as_str),
        Some("What about tomorrow?")
    );
}

// ---------------------------------------------------------------------------
// tests/memory/test_session_helpers.py
// ---------------------------------------------------------------------------

#[test]
fn settings_normalization_and_limit_precedence() {
    let settings = SessionSettings::with_limit(2);
    assert_eq!(resolve_session_limit(None, Some(&settings)), Some(2));
    assert_eq!(resolve_session_limit(Some(0), Some(&settings)), Some(0));
    assert_eq!(resolve_session_limit(Some(3), Some(&settings)), Some(3));
    assert_eq!(
        resolve_session_limit(None, Some(&SessionSettings::with_limit(4))),
        Some(4)
    );
    assert_eq!(resolve_session_limit(None, None), None);
    assert_eq!(
        resolve_session_limit(None, Some(&SessionSettings::default())),
        None
    );
}

#[tokio::test]
async fn session_mutations_return_their_result_and_propagate_failures() {
    let value = await_session_mutation(async { Ok(42_u32) }).await.unwrap();
    assert_eq!(value, 42);

    let failed = await_session_mutation(async {
        Err::<u32, _>(openai_agents_sdk::error::AgentsError::other("write failed"))
    })
    .await;
    assert!(failed.unwrap_err().to_string().contains("write failed"));
}

#[tokio::test]
async fn a_session_mutation_runs_to_completion_after_its_caller_is_dropped() {
    // Rust cannot await a cancellation the way `asyncio` does, so the parity claim is the
    // documented one: the mutation still commits once polled, even if the caller goes away.
    let store = Arc::new(ListSession::new("mutation"));
    let writer = store.clone();
    let mutation = await_session_mutation(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        writer.add_items(vec![user("committed")]).await
    });
    let handle = tokio::spawn(mutation);
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    handle.abort();
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert_eq!(contents(&store.items()), vec!["committed"]);
}

#[tokio::test]
async fn a_custom_session_honours_the_runner_and_storage_contract() {
    let (agent, model) = fixtures();
    let session = Arc::new(ListSession::new("first").with_settings(SessionSettings::with_limit(2)));
    let other = ListSession::new("other");

    model.add_turn(vec![assistant_message("Hello!")]);
    Runner::run_with_options(
        &agent,
        "Hello",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    model.add_turn(vec![assistant_message("You said hello.")]);
    Runner::run_with_options(
        &agent,
        "What did I say?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();

    assert_eq!(
        item_text(&model.last_call().unwrap().input[0]),
        "Hello",
        "the limit of 2 keeps the first turn's user message at the head"
    );
    assert_eq!(session.get_items(None).await.unwrap().len(), 2);
    assert_eq!(session.get_items(Some(4)).await.unwrap().len(), 4);
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());
    assert!(other.get_items(None).await.unwrap().is_empty());

    let last = session
        .get_items(None)
        .await
        .unwrap()
        .last()
        .cloned()
        .unwrap();
    assert_eq!(session.pop_item().await.unwrap(), Some(last));
    session.clear_session().await.unwrap();
    assert_eq!(session.pop_item().await.unwrap(), None);
    assert!(session.get_items(None).await.unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// tests/utils/test_simple_session.py — the list-backed session used by the ports
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_session_preserves_history_and_returns_copies() {
    let history = vec![
        InputItem::from_value(json!({"id": "msg1", "content": "hi", "role": "user"})),
        InputItem::from_value(json!({"id": "msg2", "content": "hello", "role": "assistant"})),
    ];
    let session = ListSession::with_history("test", history.clone());

    let items = session.get_items(None).await.unwrap();
    assert_eq!(items, history);
    assert_eq!(session.items(), history);
}

#[tokio::test]
async fn list_session_counts_pop_calls() {
    let session = ListSession::with_history(
        "test",
        vec![InputItem::from_value(
            json!({"id": "x", "content": "hi", "role": "user"}),
        )],
    );

    assert_eq!(session.pop_calls(), 0);
    session.pop_item().await.unwrap();
    assert_eq!(session.pop_calls(), 1);
    session.pop_item().await.unwrap();
    assert_eq!(session.pop_calls(), 2);
}

#[tokio::test]
async fn an_id_stripping_session_drops_ids_on_write() {
    let session = ListSession::new("test").stripping_ids();
    session
        .add_items(vec![
            InputItem::from_value(
                json!({"id": "keep-removed", "content": "hello", "role": "user"}),
            ),
            InputItem::from_value(json!({"content": "no-id", "role": "assistant"})),
        ])
        .await
        .unwrap();

    let stored = session.get_items(None).await.unwrap();
    assert!(stored.iter().all(|i| i.as_value().get("id").is_none()));

    session.pop_item().await.unwrap();
    assert_eq!(session.pop_calls(), 1);
}
