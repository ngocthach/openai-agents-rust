//! In-memory twin of `tests/py_memory_sqlx_compaction.rs`: the `memory` half of the Python
//! `backend` fixture in `tests/memory/test_sqlalchemy_compaction.py`, which parametrizes every
//! test over a file and an in-memory SQLite engine.
//!
//! An in-memory SQLite database lives per connection, so the pool handed to
//! `SqlxSession::with_pool` is limited to one connection; that makes every session and raw
//! statement below see the same database, as Python's static pool does.
#![cfg(all(feature = "sqlx", feature = "sqlite", feature = "encrypt"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::memory::SqlxDialect;
use openai_agents_sdk::memory::{
    EncryptedSession, OpenAIResponsesCompactionSession, Session, SessionSettings, SqlxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};
use sqlx::AnyPool;
use sqlx::any::AnyPoolOptions;

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text, mock_openai, serve};

const KEY: &str = "0123456789abcdef0123456789abcdef";
const OTHER_KEY: &str = "fedcba9876543210fedcba9876543210";
/// How long an envelope written with `TTL` stays valid, plus a margin.
const TTL: u64 = 2;
const PAST_TTL: Duration = Duration::from_millis(3200);

async fn memory_pool() -> AnyPool {
    sqlx::any::install_default_drivers();
    AnyPoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap()
}

fn summary() -> Value {
    json!({"type": "compaction", "id": "cmp", "encrypted_content": "synthetic-summary"})
}

fn compact_reply(output: Vec<Value>) -> Value {
    json!({
        "id": "cmp_1",
        "output": output,
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    })
}

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn alternating(count: usize, prefix: &str) -> Vec<InputItem> {
    (0..count)
        .map(|i| {
            let text = format!("{prefix} {i}");
            if i % 2 == 1 {
                InputItem::from_value(json!({"role": "assistant", "content": text}))
            } else {
                user(&text)
            }
        })
        .collect()
}

async fn limited_backend(pool: &AnyPool, id: &str) -> Arc<SqlxSession> {
    Arc::new(
        SqlxSession::with_pool(id, pool.clone(), SqlxDialect::Sqlite, true)
            .await
            .unwrap()
            .with_settings(SessionSettings::with_limit(20)),
    )
}

fn compaction_over(
    store: Arc<dyn Session>,
    base: String,
    trigger: Option<bool>,
) -> Arc<OpenAIResponsesCompactionSession> {
    let session = OpenAIResponsesCompactionSession::new("bounded", store)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));
    Arc::new(match trigger {
        Some(always) => session.with_should_trigger_compaction(move |_| always),
        None => session,
    })
}

fn worker(replies: &[&str]) -> (Agent<()>, Arc<FakeModel>) {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    for reply in replies {
        model.add_turn(vec![assistant_message(*reply)]);
    }
    (
        Agent::<()>::new("worker").with_model_instance(model.clone()),
        model,
    )
}

async fn run_turn(
    agent: &Agent<()>,
    input: &str,
    session: Arc<dyn Session>,
    streamed: bool,
) -> Result<()> {
    let options = RunOptions::new(()).session(session);
    if !streamed {
        Runner::run_with_options(agent, input, options).await?;
        return Ok(());
    }
    let mut stream = Runner::run_streamed_with_options(agent, input, options);
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            event?;
        }
    }
    stream.finish().await
}

/// Run SQL against the shared in-memory database.
async fn raw_sql(pool: &AnyPool, sql: &str) {
    sqlx::query(sqlx::AssertSqlSafe(sql.to_string()))
        .execute(pool)
        .await
        .unwrap();
}

fn content_texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

// ---------------------------------------------------------------------------
// test_limited_sqlalchemy_compacts_visible_suffix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_limited_store_compacts_only_the_visible_suffix() {
    for streamed in [false, true] {
        let pool = memory_pool().await;
        let backend = limited_backend(&pool, "bounded").await;
        let hidden =
            InputItem::from_value(json!({"role": "user", "content": "older hidden history"}));
        let mut history = vec![hidden.clone()];
        history.extend(alternating(20, "visible"));
        backend.add_items(history).await.unwrap();

        let (base, captured) = mock_openai(vec![compact_reply(vec![])]).await;
        let session = compaction_over(backend.clone(), base.to_string(), None);
        let (agent, _model) = worker(&["done"; 5]);
        for turn in 0..5 {
            run_turn(&agent, &format!("turn {turn}"), session.clone(), streamed)
                .await
                .unwrap();
        }

        let requests = captured.lock().unwrap().clone();
        assert_eq!(
            requests.len(),
            1,
            "streamed={streamed}: compaction ran once"
        );
        assert_eq!(requests[0].path, "/v1/responses/compact");
        let body = &requests[0].body;
        // Scoped to the suffix the run saw: 20 visible items, the new input and the reply.
        assert_eq!(body["input"].as_array().map(Vec::len), Some(22), "{body}");
        assert!(
            !body.to_string().contains("older hidden history"),
            "history beyond the limit must not be sent"
        );
        assert!(body.get("previous_response_id").is_none(), "{body}");

        let retained = backend.get_items(Some(100)).await.unwrap();
        assert_eq!(retained[0], hidden, "streamed={streamed}");
        assert_eq!(retained.len(), 9, "streamed={streamed}");
    }
}

// ---------------------------------------------------------------------------
// test_encrypted_prefix_cleanup_is_atomic
// ---------------------------------------------------------------------------

/// Encrypted storage under the compaction wrapper: expired envelopes are reclaimed in the same
/// transaction as the replacement, and a rejected summary insert rolls the whole thing back.
async fn encrypted_prefix_cleanup(reject_insert: bool) {
    let pool = memory_pool().await;
    let backend = limited_backend(&pool, "bounded").await;
    let encrypted: Arc<dyn Session> =
        Arc::new(EncryptedSession::new("bounded", backend.clone(), KEY, TTL).unwrap());

    let compacting = Arc::new(Mutex::new(false));
    let captured: Arc<Mutex<Vec<Request>>> = Arc::new(Mutex::new(Vec::new()));
    let flag = compacting.clone();
    let cap = captured.clone();
    let trigger_pool = pool.clone();
    let handle = tokio::runtime::Handle::current();
    let base = serve(Arc::new(move |req: &Request| {
        cap.lock().unwrap().push(req.clone());
        *flag.lock().unwrap() = true;
        if reject_insert {
            // Python rejects `INSERT INTO custom_messages` from an engine event while compacting.
            let p = trigger_pool.clone();
            let h = handle.clone();
            std::thread::spawn(move || {
                h.block_on(raw_sql(
                    &p,
                    "CREATE TRIGGER reject_summary BEFORE INSERT ON agent_messages \
                     BEGIN SELECT RAISE(ABORT, 'synthetic summary insertion failure'); END",
                ))
            })
            .join()
            .unwrap();
        }
        Reply::json(compact_reply(vec![summary()]))
    }))
    .await;
    let session = compaction_over(encrypted.clone(), format!("{base}/v1"), None);

    let expired: Vec<InputItem> = (0..40).map(|_| user("expired")).collect();
    session.add_items(expired).await.unwrap();
    tokio::time::sleep(PAST_TTL).await;
    session.add_items(alternating(20, "live")).await.unwrap();
    let stored_before = backend.get_items(Some(100)).await.unwrap();
    assert_eq!(stored_before.len(), 60);

    let (agent, _model) = worker(&["done"]);
    let result = run_turn(&agent, "continue", session.clone(), false).await;

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].body.to_string().contains("expired"),
        "expired history must not reach the compaction request"
    );
    if reject_insert {
        let error = result
            .expect_err("the rejected summary insert must surface")
            .to_string();
        assert!(
            error.contains("synthetic summary insertion failure"),
            "{error}"
        );
        let after = backend.get_items(Some(100)).await.unwrap();
        // Nothing was reclaimed or replaced: the original rows plus the turn's two items.
        assert_eq!(&after[..60], &stored_before[..]);
        assert_eq!(after.len(), 62);
    } else {
        result.unwrap();
        assert_eq!(backend.get_items(Some(100)).await.unwrap().len(), 1);
        let visible = session.get_items(None).await.unwrap();
        assert_eq!(visible.len(), 1);
        for field in ["type", "id", "encrypted_content"] {
            assert_eq!(visible[0].as_value()[field], summary()[field], "{field}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn encrypted_prefix_cleanup_commits_with_the_replacement() {
    encrypted_prefix_cleanup(false).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_summary_insert_rolls_back_the_prefix_cleanup() {
    encrypted_prefix_cleanup(true).await;
}

/// Python's `outer=True` variants wrap the compaction session in the encrypted one. Rust's
/// `EncryptedSession` does not forward `on_run_completed` to a wrapped compaction session
/// (`src/memory/encrypted_session.rs:119`), so the runner never triggers compaction.
#[tokio::test(flavor = "multi_thread")]
async fn encrypted_prefix_cleanup_with_the_encryption_outermost() {
    let pool = memory_pool().await;
    let backend = limited_backend(&pool, "bounded").await;
    let (base, captured) = mock_openai(vec![compact_reply(vec![summary()])]).await;
    let compaction = compaction_over(backend.clone(), base, None);
    let session: Arc<dyn Session> =
        Arc::new(EncryptedSession::new("bounded", compaction, KEY, TTL).unwrap());

    let expired: Vec<InputItem> = (0..40).map(|_| user("expired")).collect();
    session.add_items(expired).await.unwrap();
    tokio::time::sleep(PAST_TTL).await;
    session.add_items(alternating(20, "live")).await.unwrap();

    let (agent, _model) = worker(&["done"]);
    run_turn(&agent, "continue", session.clone(), false)
        .await
        .unwrap();

    assert_eq!(
        captured.lock().unwrap().len(),
        1,
        "compaction must run once"
    );
    assert_eq!(backend.get_items(Some(100)).await.unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// test_replacement_uses_timestamp_order_not_id_order
// ---------------------------------------------------------------------------

#[tokio::test]
async fn replacement_follows_timestamp_order_not_id_order() {
    let pool = memory_pool().await;
    let backend = limited_backend(&pool, "bounded").await;
    backend
        .add_items((0..20).map(|i| user(&format!("visible {i}"))).collect())
        .await
        .unwrap();
    let hidden =
        InputItem::from_value(json!({"role": "user", "content": "old timestamp, newer ID"}));
    backend.add_items(vec![hidden.clone()]).await.unwrap();
    raw_sql(
        &pool,
        "UPDATE agent_messages SET created_at = '2000-01-01 00:00:00' \
         WHERE id = (SELECT MAX(id) FROM agent_messages)",
    )
    .await;

    let (base, _captured) = mock_openai(vec![compact_reply(vec![summary()])]).await;
    let session = compaction_over(backend.clone(), base, Some(true));
    let (agent, _model) = worker(&["done"]);
    run_turn(&agent, "continue", session, false).await.unwrap();

    let stored = backend.get_items(Some(100)).await.unwrap();
    assert_eq!(stored.len(), 2, "{:?}", content_texts(&stored));
    assert_eq!(stored[0], hidden);
    assert_eq!(stored[1].as_value()["type"], json!("compaction"));
}

// ---------------------------------------------------------------------------
// test_encrypted_unknown_row_is_retained
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_row_the_key_cannot_verify_survives_prefix_cleanup() {
    let pool = memory_pool().await;
    let backend = limited_backend(&pool, "bounded").await;
    let encrypted: Arc<dyn Session> =
        Arc::new(EncryptedSession::new("bounded", backend.clone(), KEY, TTL).unwrap());
    let wrong_key = EncryptedSession::new("bounded", backend.clone(), OTHER_KEY, TTL).unwrap();

    let expired: Vec<InputItem> = (0..40).map(|_| user("expired")).collect();
    encrypted.add_items(expired).await.unwrap();
    wrong_key
        .add_items(vec![user("unverifiable")])
        .await
        .unwrap();
    let retained = backend.get_items(Some(100)).await.unwrap().pop().unwrap();
    tokio::time::sleep(PAST_TTL).await;
    encrypted.add_items(alternating(20, "live")).await.unwrap();

    let (base, captured) = mock_openai(vec![compact_reply(vec![])]).await;
    let session = compaction_over(encrypted, base, None);
    let (agent, _model) = worker(&["done"]);
    run_turn(&agent, "continue", session, false).await.unwrap();

    assert_eq!(captured.lock().unwrap().len(), 1);
    assert_eq!(backend.get_items(Some(100)).await.unwrap(), vec![retained]);
}

// ---------------------------------------------------------------------------
// test_parent_constraint_rejection_propagates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_rejected_session_row_propagates_and_leaves_the_store_usable() {
    let pool = memory_pool().await;
    let backend = limited_backend(&pool, "bounded").await;
    backend.get_items(None).await.unwrap();
    raw_sql(
        &pool,
        "CREATE TRIGGER reject_parent BEFORE INSERT ON agent_sessions \
         BEGIN SELECT RAISE(ABORT, 'synthetic parent constraint'); END",
    )
    .await;

    let error = backend
        .add_items(vec![user("rejected")])
        .await
        .expect_err("the parent-row insert is rejected")
        .to_string();
    assert!(error.contains("synthetic parent constraint"), "{error}");
    assert!(backend.get_items(None).await.unwrap().is_empty());

    raw_sql(&pool, "DROP TRIGGER reject_parent").await;
    backend.add_items(vec![user("accepted")]).await.unwrap();
    assert_eq!(
        content_texts(&backend.get_items(None).await.unwrap()),
        vec!["accepted"]
    );
}

// ---------------------------------------------------------------------------
// test_other_instance_mutation_invalidates_suffix
// ---------------------------------------------------------------------------

/// Another session instance mutates the same stored history while the compaction request is in
/// flight; the replacement must then not be applied and the other instance's change survives.
#[tokio::test(flavor = "multi_thread")]
async fn compaction_is_abandoned_when_another_instance_mutates_the_suffix() {
    for operation in ["append", "pop", "clear"] {
        let pool = memory_pool().await;
        let backend = limited_backend_unbounded(&pool, "bounded").await;
        let original: Vec<InputItem> = (0..24).map(|i| user(&format!("item {i}"))).collect();
        backend.add_items(original).await.unwrap();

        let newer = json!({"role": "user", "content": "concurrent"});
        let captured: Arc<Mutex<Vec<Request>>> = Arc::new(Mutex::new(Vec::new()));
        let cap = captured.clone();
        let other_pool = pool.clone();
        let handle = tokio::runtime::Handle::current();
        let base = serve(Arc::new(move |req: &Request| {
            cap.lock().unwrap().push(req.clone());
            let pool = other_pool.clone();
            let handle = handle.clone();
            let newer = newer.clone();
            std::thread::spawn(move || {
                handle.block_on(async move {
                    let other = SqlxSession::with_pool("bounded", pool, SqlxDialect::Sqlite, false)
                        .await
                        .unwrap();
                    match operation {
                        "append" => other
                            .add_items(vec![InputItem::from_value(newer)])
                            .await
                            .unwrap(),
                        "pop" => {
                            other.pop_item().await.unwrap();
                        }
                        _ => other.clear_session().await.unwrap(),
                    }
                })
            })
            .join()
            .unwrap();
            Reply::json(compact_reply(vec![]))
        }))
        .await;
        let session = compaction_over(backend.clone(), format!("{base}/v1"), Some(true));
        let (agent, _model) = worker(&["done"]);
        run_turn(&agent, "continue", session, false).await.unwrap();

        assert_eq!(captured.lock().unwrap().len(), 1, "{operation}");
        // The run persisted `continue` and the reply (26 items) before compaction started.
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

async fn limited_backend_unbounded(pool: &AnyPool, id: &str) -> Arc<SqlxSession> {
    Arc::new(
        SqlxSession::with_pool(id, pool.clone(), SqlxDialect::Sqlite, true)
            .await
            .unwrap(),
    )
}
