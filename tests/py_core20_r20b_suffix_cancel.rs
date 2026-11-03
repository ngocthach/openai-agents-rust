//! Port of `test_cancelled_suffix_transaction_settles_before_newer_append`
//! (`tests/memory/test_compaction_suffix.py`, `SQLiteSession` variant).
//!
//! Python gates the insert inside the suffix transaction, cancels the run, and checks that a
//! newer append through the compaction session lands after the replacement settles. Here the
//! suffix transaction is held at its start by another connection's write lock on the same
//! database file (the store waits out its busy timeout), the run is aborted, and the newer
//! append must wait for the transaction and land after the summary.
#![cfg(feature = "sqlite")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionSession, Session, SessionSettings, SqliteSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunConfig, RunOptions, Runner};
use serde_json::json;
use tokio::sync::Notify;

mod py_mcpmem_support;
use py_mcpmem_support::mock_openai_with;

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

async fn suffix_case(backend: Arc<dyn Session>, path: std::path::PathBuf) {
    let original: Vec<InputItem> = (0..4).map(|i| user(&i.to_string())).collect();
    backend.add_items(original.clone()).await.unwrap();

    // A second connection takes the write lock when the compaction request arrives, so the
    // suffix transaction that follows waits at its start until the test releases it.
    let (lock_tx, lock_rx) = mpsc::channel::<()>();
    let (ack_tx, ack_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let locker_path = path.clone();
    let locker = std::thread::spawn(move || {
        if lock_rx.recv().is_err() {
            return;
        }
        let conn = rusqlite::Connection::open(&locker_path).unwrap();
        conn.execute_batch("BEGIN IMMEDIATE").unwrap();
        ack_tx.send(()).unwrap();
        let _ = release_rx.recv();
        conn.execute_batch("COMMIT").unwrap();
    });

    let summary = json!({"type": "compaction", "id": "cmp", "encrypted_content": "synthetic"});
    let started = Arc::new(Notify::new());
    let requested = Arc::new(AtomicBool::new(false));
    let (signal, reply, lock_tx, ack_rx) = (
        started.clone(),
        summary.clone(),
        Mutex::new(lock_tx),
        Mutex::new(ack_rx),
    );
    let (base, _) = mock_openai_with(move |_| {
        if !requested.swap(true, Ordering::SeqCst) {
            lock_tx.lock().unwrap().send(()).unwrap();
            ack_rx.lock().unwrap().recv().unwrap();
            signal.notify_one();
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

    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("the compaction request was sent");
    // Give the run time to receive the reply and start the suffix transaction, which then
    // waits on the held write lock (well inside SQLite's 5 s busy timeout); then drop the run.
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
        "the newer append waits for the suffix transaction"
    );

    release_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(10), append)
        .await
        .expect("the append finishes once the transaction settles")
        .unwrap()
        .unwrap();
    locker.join().unwrap();

    let mut expected = original[..3].to_vec();
    expected.push(InputItem::from_value(summary));
    expected.push(newer);
    assert_eq!(backend.get_items(Some(100)).await.unwrap(), expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dropped_run_settles_its_suffix_transaction_before_a_newer_append() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    let backend = Arc::new(SqliteSession::open("suffix", &path).unwrap());
    suffix_case(backend, path).await;
}
