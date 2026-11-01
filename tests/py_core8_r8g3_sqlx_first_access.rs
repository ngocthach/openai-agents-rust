//! Ports of the SQLite first-access tests of
//! `tests/extensions/memory/test_sqlalchemy_session.py` for `SqlxSession`:
//! `test_add_items_concurrent_first_access_with_create_tables`,
//! `test_add_items_concurrent_first_write_after_tables_exist` and
//! `test_add_items_waits_for_transient_sqlite_write_lock`.
#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use std::sync::Arc;
use std::time::Duration;

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{Session, SqlxSession};

fn url(dir: &tempfile::TempDir, name: &str) -> String {
    format!("sqlite://{}?mode=rwc", dir.path().join(name).display())
}

fn contents(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.as_value()["content"].as_str().unwrap().to_string())
        .collect()
}

async fn add_concurrently(session: Arc<SqlxSession>, submitted: &[String]) {
    let workers: Vec<_> = submitted
        .iter()
        .cloned()
        .map(|content| {
            let session = session.clone();
            tokio::spawn(async move { session.add_items(vec![InputItem::user(content)]).await })
        })
        .collect();
    for worker in workers {
        worker
            .await
            .unwrap()
            .expect("a concurrent first write failed");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_first_writes_with_create_tables_are_all_kept() {
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(
        SqlxSession::from_url("concurrent_first_access", &url(&dir, "first.db"), true)
            .await
            .unwrap(),
    );
    let submitted: Vec<String> = (0..25).map(|i| format!("msg-{i}")).collect();

    add_concurrently(session.clone(), &submitted).await;

    let mut stored = contents(&session.get_items(None).await.unwrap());
    stored.sort();
    let mut expected = submitted;
    expected.sort();
    assert_eq!(stored, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_first_writes_after_the_tables_exist_are_all_kept() {
    let dir = tempfile::tempdir().unwrap();
    let path = url(&dir, "first_write.db");
    let setup = SqlxSession::from_url("concurrent_first_write", &path, true)
        .await
        .unwrap();
    setup.get_items(None).await.unwrap();

    let session = Arc::new(
        SqlxSession::from_url("concurrent_first_write", &path, false)
            .await
            .unwrap(),
    );
    let submitted: Vec<String> = (0..25).map(|i| format!("msg-{i}")).collect();

    add_concurrently(session.clone(), &submitted).await;

    let mut stored = contents(&session.get_items(None).await.unwrap());
    stored.sort();
    let mut expected = submitted;
    expected.sort();
    assert_eq!(stored, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_waits_for_a_transient_sqlite_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sqlite_write_lock_retry.db");
    let session = Arc::new(
        SqlxSession::from_url(
            "sqlite_write_lock_retry",
            &format!("sqlite://{}?mode=rwc", path.display()),
            true,
        )
        .await
        .unwrap(),
    );
    session.get_items(None).await.unwrap();

    let blocker = rusqlite::Connection::open(&path).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
    let writer = session.clone();
    let blocked_write =
        tokio::spawn(async move { writer.add_items(vec![InputItem::user("after-lock")]).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !blocked_write.is_finished(),
        "the write must wait for the lock instead of failing"
    );
    blocker.execute_batch("ROLLBACK").unwrap();

    tokio::time::timeout(Duration::from_secs(5), blocked_write)
        .await
        .expect("the blocked write never finished")
        .unwrap()
        .unwrap();

    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec!["after-lock"]
    );
}
