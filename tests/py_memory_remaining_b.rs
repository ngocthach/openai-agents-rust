//! Parity ports from `tests/extensions/memory/test_async_sqlite_session.py` (the cases not
//! already covered by `py_memory_session.rs`) and `tests/memory/test_sqlalchemy_compaction.py`
//! (mapped to `SqlxSession` on SQLite, since Rust has no SQLAlchemy backend).

#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{
    OpenAIResponsesCompactionSession, Session, SqliteSession, SqlxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text, serve};

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

// ---------------------------------------------------------------------------
// test_async_sqlite_session.py
// ---------------------------------------------------------------------------

/// `test_async_sqlite_session_add_empty_items_list`.
#[tokio::test]
async fn sqlite_add_of_an_empty_list_stores_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("empty_add", dir.path().join("e.db")).unwrap();
    session.add_items(Vec::new()).await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

/// `test_async_sqlite_session_get_items_same_timestamp_consistent_order`: rows written in one
/// burst (identical `created_at`) must still come back in insertion order.
#[tokio::test]
async fn sqlite_order_survives_writes_within_one_timestamp() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("same_ts", dir.path().join("t.db")).unwrap();
    let items: Vec<InputItem> = (0..50).map(|i| user(&format!("m{i}"))).collect();
    session.add_items(items).await.unwrap();

    let all = texts(&session.get_items(None).await.unwrap());
    let expected: Vec<String> = (0..50).map(|i| format!("m{i}")).collect();
    assert_eq!(all, expected);
    assert_eq!(
        texts(&session.get_items(Some(3)).await.unwrap()),
        vec!["m47", "m48", "m49"]
    );
}

/// `test_async_sqlite_session_pop_item_same_timestamp_returns_latest`.
#[tokio::test]
async fn sqlite_pop_returns_the_latest_of_a_same_timestamp_burst() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("same_ts_pop", dir.path().join("p.db")).unwrap();
    session
        .add_items(vec![user("first"), user("second"), user("third")])
        .await
        .unwrap();
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "third"
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "second"
    );
}

// ---------------------------------------------------------------------------
// test_sqlalchemy_compaction.py (SqlxSession on SQLite)
// ---------------------------------------------------------------------------

async fn sqlx_store(url: &str, id: &str) -> SqlxSession {
    SqlxSession::from_url(id, url, true).await.unwrap()
}

/// Run one mutation of the same session from a different connection, on a thread with its own
/// runtime, so it can be invoked from the synchronous mock server handler.
fn mutate_from_another_instance(url: String, id: String, op: &'static str, newer: Value) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let other = SqlxSession::from_url(&id, &url, false).await.unwrap();
            match op {
                "append" => other
                    .add_items(vec![InputItem::from_value(newer)])
                    .await
                    .unwrap(),
                "pop" => {
                    other.pop_item().await.unwrap();
                }
                _ => other.clear_session().await.unwrap(),
            }
        });
    })
    .join()
    .unwrap();
}

/// `test_other_instance_mutation_invalidates_suffix` (3 parametrized operations).
#[tokio::test]
async fn sqlx_compaction_is_abandoned_when_another_instance_mutates_the_suffix() {
    for operation in ["append", "pop", "clear"] {
        let dir = tempfile::tempdir().unwrap();
        let url = format!("sqlite://{}?mode=rwc", dir.path().join("h.db").display());
        let backend = Arc::new(sqlx_store(&url, "bounded").await);
        let original: Vec<InputItem> = (0..24).map(|i| user(&format!("item {i}"))).collect();
        backend.add_items(original.clone()).await.unwrap();

        let reply = assistant_message("done");
        let newer = json!({"role": "user", "content": "concurrent"});
        let (base, captured) = {
            let captured = Arc::new(Mutex::new(Vec::<Request>::new()));
            let cap = captured.clone();
            let url_for_mutation = url.clone();
            let op = operation;
            let base = serve(Arc::new(move |req: &Request| {
                cap.lock().unwrap().push(req.clone());
                mutate_from_another_instance(
                    url_for_mutation.clone(),
                    "bounded".into(),
                    op,
                    newer.clone(),
                );
                Reply::json(json!({"id": "cmp_1", "output": [], "usage": null}))
            }))
            .await;
            (base, captured)
        };
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![reply]);
        let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
        let session = Arc::new(
            OpenAIResponsesCompactionSession::new("bounded", backend.clone())
                .unwrap()
                .with_client(OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")))
                .with_should_trigger_compaction(|_| true),
        );
        Runner::run_with_options(&agent, "continue", RunOptions::new(()).session(session))
            .await
            .unwrap();

        assert_eq!(captured.lock().unwrap().len(), 1, "{operation}");
        // The run persisted `continue` and the reply (26 items) before compaction started. The
        // other instance then changed the suffix, so the replacement must not have applied.
        let stored = backend.get_items(Some(100)).await.unwrap();
        match operation {
            "append" => {
                assert_eq!(stored.len(), 27, "{operation}");
                assert_eq!(item_text(stored.last().unwrap()), "concurrent");
            }
            "pop" => assert_eq!(stored.len(), 25, "{operation}"),
            _ => assert!(stored.is_empty(), "{operation}"),
        }
        assert!(stored.iter().all(|i| i.item_type() != Some("compaction")));
    }
}

/// `test_partial_previous_response_mode_preserves_history` (suffix suite): with the
/// `previous_response_id` mode and no response id, the run must not compact and the history must
/// stay whole.
#[tokio::test]
async fn previous_response_mode_without_a_response_id_leaves_history_alone() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}?mode=rwc", dir.path().join("pr.db").display());
    let backend = Arc::new(sqlx_store(&url, "suffix").await);
    let original: Vec<InputItem> = (0..4).map(|i| user(&i.to_string())).collect();
    backend.add_items(original.clone()).await.unwrap();

    let (base, captured) = {
        let captured = Arc::new(Mutex::new(Vec::<Request>::new()));
        let cap = captured.clone();
        let base = serve(Arc::new(move |req: &Request| {
            cap.lock().unwrap().push(req.clone());
            Reply::json(json!({"id": "cmp_1", "output": [], "usage": null}))
        }))
        .await;
        (base, captured)
    };
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
    let session = Arc::new(
        OpenAIResponsesCompactionSession::new("suffix", backend.clone())
            .unwrap()
            .with_compaction_mode(openai_agents_sdk::memory::CompactionMode::PreviousResponseId)
            .with_client(OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")))
            .with_should_trigger_compaction(|_| true),
    );
    Runner::run_with_options(
        &agent,
        "continue",
        RunOptions::new(()).session(session).run_config(
            openai_agents_sdk::RunConfig::<()>::default()
                .with_session_settings(openai_agents_sdk::memory::SessionSettings::with_limit(1)),
        ),
    )
    .await
    .unwrap();

    assert!(captured.lock().unwrap().is_empty());
    assert_eq!(backend.get_items(Some(100)).await.unwrap().len(), 6);
}
