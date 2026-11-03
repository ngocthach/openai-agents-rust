//! First connections to one SQLite file racing each other, a port of the intent of
//! `test_initialization_retries_transient_wal_lock` and
//! `test_initialization_releases_wal_cursor_before_schema_write` in
//! `tests/extensions/memory/test_async_sqlite_session.py`: concurrent first opens must all
//! succeed and every session must be usable afterwards.
#![cfg(feature = "sqlite")]

use std::sync::{Arc, Barrier};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{Session, SqliteSession};

fn content(item: &InputItem) -> String {
    item.as_value()["content"].as_str().unwrap().to_string()
}

#[test]
fn concurrent_first_opens_of_one_file_all_succeed() {
    for round in 0..5 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("startup.db");
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|index| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    SqliteSession::open(format!("startup-{index}"), &path)
                })
            })
            .collect();
        let sessions: Vec<SqliteSession> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Result<_, _>>()
            .unwrap_or_else(|error| panic!("round {round}: a first open failed: {error}"));

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            for (index, session) in sessions.iter().enumerate() {
                session
                    .add_items(vec![InputItem::user(format!("hello {index}"))])
                    .await
                    .unwrap();
            }
            for (index, session) in sessions.iter().enumerate() {
                let items = session.get_items(None).await.unwrap();
                assert_eq!(
                    items.iter().map(content).collect::<Vec<_>>(),
                    vec![format!("hello {index}")],
                    "round {round}"
                );
            }
        });
    }
}
