//! Cases of `tests/extensions/memory/test_async_sqlite_session.py` and
//! `test_sqlalchemy_session.py` that earlier audits carried over as python-only: a start-up
//! failure that is not a transient lock is reported at once instead of being retried, and two
//! sessions racing to pop the last row hand it out once.
#![cfg(feature = "sqlite")]

use std::time::{Duration, Instant};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{Session, SqliteSession};

#[test]
fn a_startup_failure_that_is_not_a_transient_lock_is_not_retried() {
    // `test_initialization_does_not_retry_unrecoverable_wal_errors`: only a busy or locked
    // answer is worth retrying; a file that is not a database fails the first attempt.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-database.db");
    std::fs::write(
        &path,
        "this is not a sqlite database, only text ".repeat(64),
    )
    .unwrap();

    let started = Instant::now();
    let result = SqliteSession::open("startup", &path);
    assert!(
        result.is_err(),
        "a text file must not open as a session store"
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the failure took {:?}, so it was retried",
        started.elapsed()
    );
}

#[tokio::test]
async fn only_one_of_two_sessions_pops_the_last_row() {
    // `test_sqlite_fallback_reserves_writer_before_tail_claim`: two sessions sharing one file
    // race to claim the only row; it is returned once.
    for round in 0..10 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fallback_pop.db");
        let writer = SqliteSession::open("fallback_pop", &path).unwrap();
        let other = SqliteSession::open("fallback_pop", &path).unwrap();
        writer
            .add_items(vec![InputItem::user("only")])
            .await
            .unwrap();

        let (a, b) = tokio::join!(writer.pop_item(), other.pop_item());
        let popped: Vec<_> = [a.unwrap(), b.unwrap()].into_iter().flatten().collect();
        assert_eq!(popped.len(), 1, "round {round}");
        assert_eq!(popped[0].as_value()["content"], "only");
    }
}
