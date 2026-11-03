//! Ports of the rollback cases of `tests/extensions/memory/test_async_sqlite_session.py`
//! (`test_failed_clear_session_rolls_back`, `test_failed_pop_item_releases_write_lock`): a
//! mutation that fails part way leaves no open write transaction, so an independent writer is
//! not blocked and the session stays usable.
#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use std::path::Path;

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{Session, SqlxSession};
use sqlx::Executor;

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

async fn open(path: &Path) -> SqlxSession {
    SqlxSession::from_url(
        "rollback",
        &format!("sqlite://{}?mode=rwc", path.display()),
        true,
    )
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

#[tokio::test]
async fn a_failed_clear_session_releases_the_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clear_rollback.db");
    let session = open(&path).await;
    session.add_items(vec![user("kept")]).await.unwrap();

    session
        .pool()
        .execute("DROP TABLE agent_sessions")
        .await
        .unwrap();
    assert!(session.clear_session().await.is_err());

    assert!(write_lock_is_free(&path));
}

#[tokio::test]
async fn a_failed_pop_item_releases_the_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pop_rollback.db");
    let session = open(&path).await;
    session.add_items(vec![user("kept")]).await.unwrap();

    session
        .pool()
        .execute("DROP TABLE agent_messages")
        .await
        .unwrap();
    assert!(session.pop_item().await.is_err());

    assert!(write_lock_is_free(&path));
}
