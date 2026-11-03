//! Ports of the close and connection-lifecycle cases of `tests/memory/test_session.py`,
//! `tests/extensions/memory/test_async_sqlite_session.py` and
//! `tests/extensions/memory/test_advanced_sqlite_session.py`.
//!
//! Rust has one `SqliteSession` for Python's `SQLiteSession` and `AsyncSQLiteSession`, so the
//! closed-session error is `SQLiteSession is closed` where the async Python class says
//! `AsyncSQLiteSession is closed`. Python serves each worker thread its own connection and
//! asserts on that private registry; Rust shares one connection per session, so the ports
//! assert what a caller sees: history order across runtimes and threads, and that a closed
//! session releases the database file.
#![cfg(feature = "sqlite")]

use std::path::Path;
use std::sync::{Arc, Barrier};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{AdvancedSqliteSession, Session, SqliteSession};
use serde_json::json;

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

fn contents(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.as_value()["content"].as_str().unwrap().to_string())
        .collect()
}

fn assert_closed<T: std::fmt::Debug>(result: openai_agents_sdk::Result<T>, name: &str) {
    let error = result.expect_err("a closed session must refuse the operation");
    assert!(
        error.to_string().contains(&format!("{name} is closed")),
        "{error}"
    );
}

async fn assert_every_operation_fails(session: &dyn Session, name: &str) {
    assert_closed(session.get_items(None).await, name);
    assert_closed(session.add_items(vec![user("after close")]).await, name);
    assert_closed(session.add_items(Vec::new()).await, name);
    assert_closed(session.pop_item().await, name);
    assert_closed(session.clear_session().await, name);
}

fn wal_path(db: &Path) -> std::path::PathBuf {
    let mut name = db.as_os_str().to_owned();
    name.push("-wal");
    name.into()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// `test_session.py::test_sqlite_session_closed_rejects_empty_add_items`.
#[tokio::test]
async fn a_closed_sqlite_session_rejects_an_empty_add_items() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        SqliteSession::open("closed_empty_add_test", dir.path().join("closed.db")).unwrap();
    session.close();
    assert_closed(session.add_items(Vec::new()).await, "SQLiteSession");
}

/// `test_session.py::test_sqlite_session_close_closes_worker_thread_connections`: closing
/// releases the connection, so SQLite checkpoints and removes the write-ahead log, and the
/// clones sharing the connection are closed with it.
#[tokio::test]
async fn close_releases_the_connection_and_the_write_ahead_log() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test_worker_thread_close.db");
    let session = SqliteSession::open("worker_thread_close", &db).unwrap();
    let sibling = session.for_session("sibling");
    session.add_items(vec![user("Hello")]).await.unwrap();
    assert!(
        wal_path(&db).exists(),
        "an open WAL connection keeps its log"
    );
    assert!(!session.is_closed());

    session.close();

    assert!(session.is_closed());
    assert!(sibling.is_closed());
    assert!(
        !wal_path(&db).exists(),
        "closing the last connection checkpoints and removes the log"
    );
    assert_every_operation_fails(&session, "SQLiteSession").await;
    assert_every_operation_fails(&sibling, "SQLiteSession").await;

    // The history itself is intact for a new session.
    let reopened = SqliteSession::open("worker_thread_close", &db).unwrap();
    assert_eq!(
        contents(&reopened.get_items(None).await.unwrap()),
        ["Hello"]
    );
}

/// `test_session.py::test_sqlite_session_reaps_connections_from_exited_worker_threads`: each
/// turn runs on a runtime that is shut down afterwards, as each Python `asyncio.run` does.
#[test]
fn a_session_keeps_its_history_across_runtimes_that_exit() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        SqliteSession::open("exited_worker_threads", dir.path().join("exited.db")).unwrap();
    for turn in 0..5 {
        runtime()
            .block_on(session.add_items(vec![user(&format!("turn {turn}"))]))
            .unwrap();
    }
    let items = runtime().block_on(session.get_items(None)).unwrap();
    assert_eq!(
        contents(&items),
        (0..5)
            .map(|turn| format!("turn {turn}"))
            .collect::<Vec<_>>()
    );
    session.close();
}

/// `test_session.py::test_sqlite_session_closes_retired_connections_before_the_next_allocation`:
/// three threads, each on its own runtime, write together and retire; the next turn lands
/// after them.
#[test]
fn concurrent_writers_on_retired_threads_keep_every_turn() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("retired_workers", dir.path().join("retired.db")).unwrap();
    let registered = Arc::new(Barrier::new(3));
    let workers: Vec<_> = (0..3)
        .map(|index| {
            let session = session.clone();
            let registered = registered.clone();
            std::thread::spawn(move || {
                runtime().block_on(async {
                    session
                        .add_items(vec![user(&format!("worker {index}"))])
                        .await
                        .unwrap();
                });
                registered.wait();
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }

    runtime()
        .block_on(session.add_items(vec![user("after they retired")]))
        .unwrap();

    let all = contents(&runtime().block_on(session.get_items(None)).unwrap());
    let (workers, last) = all.split_at(3);
    let mut workers = workers.to_vec();
    workers.sort();
    assert_eq!(workers, ["worker 0", "worker 1", "worker 2"]);
    assert_eq!(last, ["after they retired"]);
    session.close();
}

/// `test_session.py::test_sqlite_session_keeps_connections_of_live_worker_threads`: a runtime
/// that stays alive on another thread and this test's runtime share one session.
#[tokio::test]
async fn a_session_serves_a_live_runtime_on_another_thread_and_this_one() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("live_worker_threads", dir.path().join("live.db")).unwrap();
    let other = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let remote = session.clone();
    other
        .spawn(async move { remote.add_items(vec![user("from the other loop")]).await })
        .await
        .unwrap()
        .unwrap();

    session
        .add_items(vec![user("from this loop")])
        .await
        .unwrap();

    // The other runtime is still alive and can still use the session.
    let remote = session.clone();
    let seen = other
        .spawn(async move { remote.get_items(None).await })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(contents(&seen), ["from the other loop", "from this loop"]);
    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        ["from the other loop", "from this loop"]
    );
    other.shutdown_background();
    session.close();
}

/// `test_async_sqlite_session.py::test_async_sqlite_session_closed_operations_raise_runtime_error`.
#[tokio::test]
async fn closed_operations_fail_instead_of_reopening_the_database() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        SqliteSession::open("closed_state_test", dir.path().join("closed_state.db")).unwrap();
    session.add_items(vec![user("before close")]).await.unwrap();
    session.close();
    assert_every_operation_fails(&session, "SQLiteSession").await;
}

/// `test_async_sqlite_session.py::test_async_sqlite_session_closed_rejects_empty_add_items`.
#[tokio::test]
async fn a_closed_session_rejects_an_empty_add_items_before_any_use() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open(
        "closed_empty_add_test",
        dir.path().join("closed_empty_add.db"),
    )
    .unwrap();
    session.close();
    assert_closed(session.add_items(Vec::new()).await, "SQLiteSession");
}

/// `test_async_sqlite_session.py::test_async_sqlite_session_close_before_use_is_terminal`.
#[tokio::test]
async fn close_before_first_use_is_terminal() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open(
        "close_before_use_test",
        dir.path().join("close_before_use.db"),
    )
    .unwrap();
    session.close();
    assert_closed(session.get_items(None).await, "SQLiteSession");
    let memory = SqliteSession::in_memory("close_before_use_memory").unwrap();
    memory.close();
    assert_closed(memory.get_items(None).await, "SQLiteSession");
}

/// `test_async_sqlite_session.py::test_async_sqlite_session_close_is_idempotent`: concurrent
/// and repeated closes are safe.
#[tokio::test]
async fn repeated_and_concurrent_closes_are_safe() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open(
        "close_idempotent_test",
        dir.path().join("close_idempotent.db"),
    )
    .unwrap();
    session.add_items(vec![user("before close")]).await.unwrap();

    let closers: Vec<_> = (0..2)
        .map(|_| {
            let session = session.clone();
            std::thread::spawn(move || session.close())
        })
        .collect();
    for closer in closers {
        closer.join().unwrap();
    }
    session.close();

    assert!(session.is_closed());
    assert_closed(session.get_items(None).await, "SQLiteSession");
}

/// `test_advanced_sqlite_session.py::test_advanced_sqlite_session_closed_rejects_empty_add_items`
/// (the advanced session inherits the base message).
#[tokio::test]
async fn a_closed_advanced_session_rejects_an_empty_add_items() {
    let dir = tempfile::tempdir().unwrap();
    let session = AdvancedSqliteSession::open_with(
        "advanced_closed_empty_add",
        dir.path().join("closed_empty_add.db"),
        true,
    )
    .unwrap();
    session.close();
    assert!(session.is_closed());
    assert_closed(session.add_items(Vec::new()).await, "SQLiteSession");
    assert_every_operation_fails(&session, "SQLiteSession").await;
    assert_closed(session.list_branches().await, "SQLiteSession");
}

/// `test_advanced_sqlite_session.py::test_structure_initialization_failure_invalidates_connection`:
/// a constructor that fails on a conflicting schema leaves no connection or lock behind, so a
/// writer that will not wait can still write.
#[tokio::test]
async fn a_failed_structure_initialization_leaves_the_database_unlocked() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("advanced_init_failure.db");
    let setup = AdvancedSqliteSession::open_with("advanced_init_failure", &db, true).unwrap();
    setup.add_items(vec![user("existing")]).await.unwrap();
    setup.close();

    {
        let conflict = rusqlite::Connection::open(&db).unwrap();
        conflict
            .execute_batch(
                "DROP TABLE branch_reservations;
                 DROP INDEX idx_structure_session_seq;
                 CREATE TABLE idx_structure_session_seq (value INTEGER);",
            )
            .unwrap();
    }

    let error = AdvancedSqliteSession::open_with("advanced_init_failure", &db, true)
        .expect_err("the conflicting index name must fail initialization");
    assert!(error.to_string().contains("already a table"), "{error}");

    let probe = rusqlite::Connection::open(&db).unwrap();
    probe.busy_timeout(std::time::Duration::ZERO).unwrap();
    probe
        .execute_batch("CREATE TABLE IF NOT EXISTS probe_lock (x INTEGER)")
        .unwrap();
}
