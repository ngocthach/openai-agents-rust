//! Python `tests/extensions/memory/test_async_sqlite_session.py`: the observable halves of the
//! cancellation and start-up cases. A mutation that is cancelled part way leaves neither a write
//! lock nor a partial batch behind (Python checks `in_transaction` and an independent writer;
//! here an aborted task takes the place of `task.cancel()`), and a start-up that meets a busy
//! database waits for it instead of failing.
#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use std::path::Path;
use std::time::Duration;

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{Session, SqlxSession};
use sqlx::Executor;

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn url(path: &Path) -> String {
    format!("sqlite://{}?mode=rwc", path.display())
}

async fn open(path: &Path) -> SqlxSession {
    SqlxSession::from_url("cancel", &url(path), true)
        .await
        .unwrap()
}

/// Whether a second connection can take the write lock right now.
fn write_lock_is_free(path: &Path) -> bool {
    let probe = rusqlite::Connection::open(path).unwrap();
    probe.busy_handler(None).unwrap();
    probe
        .execute_batch("CREATE TABLE IF NOT EXISTS probe_lock (x INTEGER)")
        .is_ok()
}

/// `test_cancelled_add_items_rolls_back_write_transaction` and
/// `test_post_commit_cancellation_propagates_after_known_mutation_outcome`: a mutation whose
/// task is aborted at an arbitrary point releases the write lock and is all-or-nothing.
#[tokio::test]
async fn an_aborted_mutation_releases_the_write_lock_and_is_atomic() {
    for delay_micros in [0u64, 200, 1_000, 5_000, 20_000] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("aborted.db");
        let session = std::sync::Arc::new(open(&path).await);
        let batch: Vec<InputItem> = (0..400).map(|i| user(&format!("item {i}"))).collect();

        let task = {
            let session = session.clone();
            let batch = batch.clone();
            tokio::spawn(async move { session.add_items(batch).await })
        };
        tokio::time::sleep(Duration::from_micros(delay_micros)).await;
        task.abort();
        let _ = task.await;
        // Let a rollback queued by the dropped transaction run.
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(write_lock_is_free(&path), "delay {delay_micros}us");
        let stored = session.get_items(None).await.unwrap().len();
        assert!(
            stored == 0 || stored == batch.len(),
            "delay {delay_micros}us stored a partial batch of {stored}"
        );
        // The session is still usable after the abort.
        session.add_items(vec![user("after")]).await.unwrap();
        assert_eq!(session.get_items(None).await.unwrap().len(), stored + 1);
    }
}

/// `test_initialization_retries_transient_wal_lock`: a database that another connection holds
/// locked for a moment is waited for, and the session then works.
#[tokio::test]
async fn start_up_waits_for_a_briefly_locked_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("startup.db");
    let holder = rusqlite::Connection::open(&path).unwrap();
    holder
        .execute_batch("CREATE TABLE seed (x INTEGER); BEGIN EXCLUSIVE;")
        .unwrap();
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        holder.execute_batch("COMMIT").unwrap();
    });

    let session = open(&path).await;
    release.join().unwrap();
    session.add_items(vec![user("hello")]).await.unwrap();
    assert_eq!(session.get_items(None).await.unwrap().len(), 1);
}

/// `test_initialization_releases_wal_cursor_before_schema_write`: another writer creating its
/// own table in the same file does not stop the session from starting.
#[tokio::test]
async fn start_up_tolerates_a_competing_writer_in_the_same_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("startup.db");
    let session = open(&path).await;
    let peer = rusqlite::Connection::open(&path).unwrap();
    peer.execute_batch("CREATE TABLE competing_writer (id INTEGER)")
        .unwrap();
    session.add_items(vec![user("hello")]).await.unwrap();
    assert_eq!(session.get_items(None).await.unwrap().len(), 1);
    session
        .pool()
        .execute("SELECT 1 FROM competing_writer")
        .await
        .unwrap();
}
