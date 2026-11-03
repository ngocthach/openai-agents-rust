//! In-memory twins of the `SQLAlchemySession` compaction cases of
//! `tests/memory/test_sqlalchemy_compaction.py`.
//!
//! Python parametrizes each case over a file engine and an in-memory engine (SQLAlchemy shares
//! one connection for `:memory:`). The file variants live in `py_memory_sqlx_compaction.rs` and
//! `py_memory_remaining_b.rs`; here the same cases run on a single-connection `sqlx` pool over
//! `sqlite::memory:`, which is how one in-memory database is shared in Rust.
#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::memory::{
    OpenAIResponsesCompactionSession, Session, SessionSettings, SqlxDialect, SqlxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};
use sqlx::AnyPool;
use sqlx::any::AnyPoolOptions;

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text, mock_openai, serve};

async fn memory_pool() -> AnyPool {
    sqlx::any::install_default_drivers();
    AnyPoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap()
}

async fn memory_backend(pool: &AnyPool, id: &str, limit: Option<usize>) -> Arc<SqlxSession> {
    let session = SqlxSession::with_pool(id, pool.clone(), SqlxDialect::Sqlite, true)
        .await
        .unwrap();
    Arc::new(match limit {
        Some(limit) => session.with_settings(SessionSettings::with_limit(limit)),
        None => session,
    })
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

fn compaction_over(
    store: Arc<dyn Session>,
    base: String,
    always: Option<bool>,
) -> Arc<OpenAIResponsesCompactionSession> {
    let session = OpenAIResponsesCompactionSession::new("bounded", store)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));
    Arc::new(match always {
        Some(always) => session.with_should_trigger_compaction(move |_| always),
        None => session,
    })
}

fn worker(replies: &[&str]) -> Agent<()> {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    for reply in replies {
        model.add_turn(vec![assistant_message(*reply)]);
    }
    Agent::<()>::new("worker").with_model_instance(model)
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

async fn raw_sql(pool: &AnyPool, sql: &'static str) {
    sqlx::query(sql).execute(pool).await.unwrap();
}

// test_limited_sqlalchemy_compacts_visible_suffix[memory, streamed x2]
#[tokio::test]
async fn a_limited_in_memory_store_compacts_only_the_visible_suffix() {
    for streamed in [false, true] {
        let pool = memory_pool().await;
        let backend = memory_backend(&pool, "bounded", Some(20)).await;
        let hidden =
            InputItem::from_value(json!({"role": "user", "content": "older hidden history"}));
        let mut history = vec![hidden.clone()];
        history.extend(alternating(20, "visible"));
        backend.add_items(history).await.unwrap();

        let (base, captured) = mock_openai(vec![compact_reply(vec![])]).await;
        let session = compaction_over(backend.clone(), base, None);
        let agent = worker(&["done"; 5]);
        for turn in 0..5 {
            run_turn(&agent, &format!("turn {turn}"), session.clone(), streamed)
                .await
                .unwrap();
        }

        let requests = captured.lock().unwrap().clone();
        assert_eq!(requests.len(), 1, "streamed={streamed}");
        let body = &requests[0].body;
        assert_eq!(body["input"].as_array().map(Vec::len), Some(22), "{body}");
        assert!(!body.to_string().contains("older hidden history"));
        assert!(body.get("previous_response_id").is_none(), "{body}");

        let retained = backend.get_items(Some(100)).await.unwrap();
        assert_eq!(retained[0], hidden, "streamed={streamed}");
        assert_eq!(retained.len(), 9, "streamed={streamed}");
    }
}

// test_replacement_uses_timestamp_order_not_id_order[memory]
#[tokio::test]
async fn in_memory_replacement_follows_timestamp_order_not_id_order() {
    let pool = memory_pool().await;
    let backend = memory_backend(&pool, "bounded", Some(20)).await;
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
    let agent = worker(&["done"]);
    run_turn(&agent, "continue", session, false).await.unwrap();

    let stored = backend.get_items(Some(100)).await.unwrap();
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[0], hidden);
    assert_eq!(stored[1].as_value()["type"], json!("compaction"));
}

// test_parent_constraint_rejection_propagates[memory]
#[tokio::test]
async fn an_in_memory_rejected_session_row_propagates_and_leaves_the_store_usable() {
    let pool = memory_pool().await;
    let backend = memory_backend(&pool, "bounded", Some(20)).await;
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
    let texts: Vec<String> = backend
        .get_items(None)
        .await
        .unwrap()
        .iter()
        .map(item_text)
        .collect();
    assert_eq!(texts, vec!["accepted"]);
}

/// A runtime that outlives every test, so a pooled connection acquired on it is returned to the
/// pool while its runtime is still alive.
fn mutation_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    })
}

/// Run one mutation of the same session through another instance sharing the pool, awaited from
/// the synchronous mock-server handler.
fn mutate_from_another_instance(pool: AnyPool, id: String, op: &'static str, newer: Value) {
    let (done, finished) = std::sync::mpsc::channel();
    mutation_runtime().spawn(async move {
        let other = SqlxSession::with_pool(&id, pool, SqlxDialect::Sqlite, false)
            .await
            .unwrap();
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
        done.send(()).unwrap();
    });
    finished.recv().unwrap();
}

// test_other_instance_mutation_invalidates_suffix[memory; append, pop, clear]
#[tokio::test]
async fn in_memory_compaction_is_abandoned_when_another_instance_mutates_the_suffix() {
    for operation in ["append", "pop", "clear"] {
        let pool = memory_pool().await;
        let backend = memory_backend(&pool, "bounded", None).await;
        let original: Vec<InputItem> = (0..24).map(|i| user(&format!("item {i}"))).collect();
        backend.add_items(original).await.unwrap();

        let newer = json!({"role": "user", "content": "concurrent"});
        let captured = Arc::new(Mutex::new(Vec::<Request>::new()));
        let cap = captured.clone();
        let shared = pool.clone();
        let base = serve(Arc::new(move |req: &Request| {
            cap.lock().unwrap().push(req.clone());
            mutate_from_another_instance(
                shared.clone(),
                "bounded".into(),
                operation,
                newer.clone(),
            );
            Reply::json(json!({"id": "cmp_1", "output": [], "usage": null}))
        }))
        .await;
        let session = compaction_over(backend.clone(), format!("{base}/v1"), Some(true));
        let agent = worker(&["done"]);
        run_turn(&agent, "continue", session, false).await.unwrap();

        assert_eq!(captured.lock().unwrap().len(), 1, "{operation}");
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
