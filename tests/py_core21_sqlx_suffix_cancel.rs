//! The `AsyncSQLiteSession` variant of `test_cancelled_suffix_transaction_settles_before_newer_append`
//! (`tests/memory/test_compaction_suffix.py`), run against `SqlxSession`.
//!
//! Python gates the insert inside the suffix transaction, cancels the run, and checks that the
//! replacement still commits (it is shielded) and that a newer append through the compaction
//! session lands after it. Here the gate is the pool's only connection: the test holds it, so the
//! replacement, which runs on its own task, waits for it; the caller is dropped meanwhile. A
//! SQLite write lock held by another connection cannot stand in, because sqlx's deferred
//! transaction fails at once with `database is locked` instead of waiting.
#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionSession, Session, SessionSettings, SqlxDialect,
    SqlxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunConfig, RunOptions, Runner};
use serde_json::json;
use sqlx::any::AnyPoolOptions;
use tokio::sync::Notify;

mod py_mcpmem_support;
use py_mcpmem_support::mock_openai_with;

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

/// A file-backed session whose pool has exactly one connection, so holding it gates every
/// statement of the session.
async fn single_connection_session() -> (tempfile::TempDir, sqlx::AnyPool, Arc<SqlxSession>) {
    sqlx::any::install_default_drivers();
    let dir = tempfile::tempdir().unwrap();
    let url = format!(
        "sqlite://{}?mode=rwc",
        dir.path().join("history.db").display()
    );
    let pool = AnyPoolOptions::new()
        .max_connections(1)
        .connect(&url)
        .await
        .unwrap();
    let session = SqlxSession::with_pool("suffix", pool.clone(), SqlxDialect::Sqlite, true)
        .await
        .unwrap();
    (dir, pool, Arc::new(session))
}

/// The replacement's caller is dropped while the replacement waits for its connection; the
/// replacement still commits once the connection is free, and an append that starts after the
/// drop waits for it and lands after the summary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dropped_replacement_commits_before_a_newer_append() {
    let (_dir, pool, backend) = single_connection_session().await;
    let original: Vec<InputItem> = (0..4).map(|i| user(&i.to_string())).collect();
    backend.add_items(original.clone()).await.unwrap();
    let snapshot = backend.compaction_snapshot(10).await.unwrap().unwrap();
    assert_eq!(snapshot.items, original);

    let held = pool.acquire().await.unwrap();
    let summary = InputItem::from_value(json!({"type": "compaction", "id": "cmp"}));
    let replace = backend.replace_compaction_suffix(&snapshot.handles[3..], vec![summary.clone()]);
    // The first poll takes the session's lock and starts the replacement; the connection is
    // held, so it cannot finish, and the caller's future is dropped.
    assert!(
        tokio::time::timeout(Duration::from_millis(100), replace)
            .await
            .is_err()
    );

    let newer = user("newer");
    let append = {
        let backend = backend.clone();
        let newer = newer.clone();
        tokio::spawn(async move { backend.add_items(vec![newer]).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !append.is_finished(),
        "the append waits for the replacement"
    );

    drop(held);
    tokio::time::timeout(Duration::from_secs(10), append)
        .await
        .expect("the append finishes once the replacement settles")
        .unwrap()
        .unwrap();
    let mut expected = original[..3].to_vec();
    expected.push(summary);
    expected.push(newer);
    assert_eq!(backend.get_items(Some(100)).await.unwrap(), expected);
}

/// The Python case end to end: a run whose compaction is dropped while its suffix replacement
/// waits still commits the replacement, and a newer append through the compaction session
/// lands after the summary (`original[:-1] + [summary, newer]`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dropped_run_settles_its_suffix_replacement_before_a_newer_append() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (_dir, pool, backend) = single_connection_session().await;
    let original: Vec<InputItem> = (0..4).map(|i| user(&i.to_string())).collect();
    backend.add_items(original.clone()).await.unwrap();

    // When the compaction request arrives, the test takes the only connection before the reply
    // is sent, so the replacement that follows waits for it.
    let (request_tx, request_rx) = mpsc::channel::<()>();
    let (reply_tx, reply_rx) = mpsc::channel::<()>();
    let summary = json!({"type": "compaction", "id": "cmp", "encrypted_content": "synthetic"});
    let requested = Arc::new(AtomicBool::new(false));
    let (reply, request_tx, reply_rx) = (
        summary.clone(),
        Mutex::new(request_tx),
        Mutex::new(reply_rx),
    );
    let (base, _) = mock_openai_with(move |_| {
        if !requested.swap(true, Ordering::SeqCst) {
            request_tx.lock().unwrap().send(()).unwrap();
            reply_rx.lock().unwrap().recv().unwrap();
        }
        json!({
            "id": "cmp",
            "output": [reply.clone()],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        })
    })
    .await;

    let session: Arc<OpenAIResponsesCompactionSession> = Arc::new(
        OpenAIResponsesCompactionSession::new("suffix", backend.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(base))
            .with_compaction_mode(CompactionMode::Input)
            .with_should_trigger_compaction(|_| true),
    );
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("worker").with_model_instance(model);
    let work = {
        let session: Arc<dyn Session> = session.clone();
        tokio::spawn(async move {
            let options = RunOptions::new(()).session(session).run_config(
                RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(1)),
            );
            Runner::run_with_options(&agent, "continue", options).await
        })
    };

    let received = Arc::new(Notify::new());
    let signal = received.clone();
    std::thread::spawn(move || {
        if request_rx.recv().is_ok() {
            signal.notify_one();
        }
    });
    tokio::time::timeout(Duration::from_secs(5), received.notified())
        .await
        .expect("the compaction request was sent");
    let held = pool.acquire().await.unwrap();
    reply_tx.send(()).unwrap();
    // Give the run time to read the reply and start the replacement, which then waits for the
    // held connection; then drop the run.
    tokio::time::sleep(Duration::from_millis(500)).await;
    work.abort();
    assert!(work.await.unwrap_err().is_cancelled());

    let newer = user("newer");
    let append = {
        let session = session.clone();
        let newer = newer.clone();
        tokio::spawn(async move { session.add_items(vec![newer]).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !append.is_finished(),
        "the newer append waits for the replacement"
    );

    drop(held);
    tokio::time::timeout(Duration::from_secs(10), append)
        .await
        .expect("the append finishes once the replacement settles")
        .unwrap()
        .unwrap();
    let mut expected = original[..3].to_vec();
    expected.push(InputItem::from_value(summary));
    expected.push(newer);
    assert_eq!(backend.get_items(Some(100)).await.unwrap(), expected);
}
