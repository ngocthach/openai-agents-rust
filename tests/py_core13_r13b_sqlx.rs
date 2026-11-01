//! Case of `tests/extensions/memory/test_sqlalchemy_session.py` that earlier audits carried
//! over as python-only: two sessions racing to pop the last row of one database.
#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{Session, SqlxSession};

#[tokio::test]
async fn only_one_of_two_sessions_pops_the_last_row() {
    // `test_sqlite_fallback_reserves_writer_before_tail_claim`.
    for round in 0..10 {
        let dir = tempfile::tempdir().unwrap();
        let url = format!("sqlite://{}?mode=rwc", dir.path().join("pop.db").display());
        let writer = SqlxSession::from_url("fallback_pop", &url, true)
            .await
            .unwrap();
        let other = SqlxSession::from_url("fallback_pop", &url, false)
            .await
            .unwrap();
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
