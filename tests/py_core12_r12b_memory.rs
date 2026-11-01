//! Ports of session-store cases that were filed as Python-only: behaviour that the Python tests
//! reach through private connection bookkeeping but that a store user can observe.
#![cfg(feature = "sqlite")]

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{Session, SqliteSession};

fn user(content: &str) -> InputItem {
    InputItem::user(content)
}

fn contents(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .map(|i| i.0["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn drop_table(db: &std::path::Path, table: &str) {
    let conn = rusqlite::Connection::open(db).unwrap();
    // The bundled SQLite enforces foreign keys by default: without this, dropping the parent
    // table would cascade into the messages and hide what the failed clear did.
    conn.execute_batch(&format!("PRAGMA foreign_keys=OFF; DROP TABLE {table}"))
        .unwrap();
}

/// Python: `test_sqlite_session_failed_clear_session_rolls_back`. A clear that fails on the
/// sessions row restores the messages it already deleted and leaves the write lock free.
#[tokio::test]
async fn a_failed_clear_session_restores_the_messages_it_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clear_rollback.db");
    let session = SqliteSession::open("clear_rollback", &path).unwrap();
    session.add_items(vec![user("kept")]).await.unwrap();

    drop_table(&path, "agent_sessions");
    assert!(session.clear_session().await.is_err());

    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        ["kept"],
        "the failed clear must not delete the messages"
    );
    let probe = rusqlite::Connection::open(&path).unwrap();
    probe.busy_handler(None).unwrap();
    probe
        .execute_batch("CREATE TABLE IF NOT EXISTS probe_lock (x INTEGER)")
        .expect("the write lock must be free after the failed clear");
}
