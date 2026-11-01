//! Two connections to one SQLite file racing each other, ports of the multi-process tests in
//! `tests/extensions/memory/test_advanced_sqlite_session.py`:
//! `test_pop_item_claim_is_unique_across_processes`,
//! `test_branch_allocation_is_serialized_across_processes` (4) and
//! `test_concurrent_structure_table_claims_leave_one_coherent_owner`.
//!
//! Python synchronizes real processes with events; threads each holding their own connection
//! race through a barrier here, and the invariants are the ones the Python tests assert. Each
//! race is repeated so a missing write reservation shows up as a flaky failure.
#![cfg(feature = "sqlite")]

use std::path::Path;
use std::sync::{Arc, Barrier};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{AdvancedSqliteSession, Session};
use rusqlite::Connection;

const ROUNDS: usize = 12;

fn content(item: &InputItem) -> String {
    item.as_value()["content"].as_str().unwrap().to_string()
}

fn contents(items: &[InputItem]) -> Vec<String> {
    items.iter().map(content).collect()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn a_destructive_read_is_claimed_by_exactly_one_connection() {
    for round in 0..ROUNDS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("advanced_pop_processes.db");
        runtime().block_on(async {
            let setup =
                AdvancedSqliteSession::open_with("advanced_pop_processes", &path, true).unwrap();
            setup
                .add_items(vec![InputItem::user("only")])
                .await
                .unwrap();
        });

        let barrier = Arc::new(Barrier::new(2));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let session =
                        AdvancedSqliteSession::open_with("advanced_pop_processes", &path, false)
                            .unwrap();
                    barrier.wait();
                    runtime().block_on(session.pop_item()).unwrap()
                })
            })
            .collect();
        let popped: Vec<Option<InputItem>> =
            workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(
            popped.iter().filter(|item| item.is_some()).count(),
            1,
            "round {round}: {popped:?}"
        );
        assert_eq!(
            popped.iter().filter(|item| item.is_none()).count(),
            1,
            "round {round}"
        );
        assert_eq!(
            content(popped.iter().flatten().next().unwrap()),
            "only",
            "round {round}"
        );
    }
}

fn branch_collision_items() -> Vec<InputItem> {
    vec![
        InputItem::user("Turn one question"),
        InputItem::from_value(
            serde_json::json!({"role": "assistant", "content": "Turn one answer"}),
        ),
        InputItem::user("Turn two question"),
        InputItem::from_value(
            serde_json::json!({"role": "assistant", "content": "Turn two answer"}),
        ),
        InputItem::user("Turn three question"),
        InputItem::from_value(
            serde_json::json!({"role": "assistant", "content": "Turn three answer"}),
        ),
    ]
}

fn branch_allocation_round(round: usize, branch_name: Option<&str>, turn_number: i64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("branch_allocation.db");
    let session_id = "branch_allocation";
    let items = branch_collision_items();
    runtime().block_on(async {
        let setup = AdvancedSqliteSession::open_with(session_id, &path, true).unwrap();
        setup.add_items(items.clone()).await.unwrap();
    });

    let barrier = Arc::new(Barrier::new(2));
    let workers: Vec<_> = ["first", "second"]
        .into_iter()
        .map(|worker| {
            let path = path.clone();
            let barrier = barrier.clone();
            let branch_name = branch_name.map(str::to_string);
            std::thread::spawn(move || {
                let session = AdvancedSqliteSession::open_with(session_id, &path, true).unwrap();
                barrier.wait();
                runtime().block_on(async {
                    let created = session
                        .create_branch_from_turn(turn_number, branch_name.as_deref())
                        .await;
                    match created {
                        Ok(branch_id) => {
                            session
                                .add_items(vec![InputItem::user(format!("{worker} branch item"))])
                                .await
                                .unwrap();
                            Ok((worker, branch_id))
                        }
                        Err(error) => Err(error.to_string()),
                    }
                })
            })
        })
        .collect();
    let results: Vec<Result<(&str, String), String>> =
        workers.into_iter().map(|w| w.join().unwrap()).collect();
    let case = format!("round {round}, name {branch_name:?}, turn {turn_number}: {results:?}");

    let successes: Vec<&(&str, String)> = results.iter().filter_map(|r| r.as_ref().ok()).collect();
    match branch_name {
        None => {
            assert_eq!(successes.len(), 2, "{case}");
            assert_ne!(successes[0].1, successes[1].1, "{case}");
        }
        Some(name) => {
            assert_eq!(successes.len(), 1, "{case}");
            assert_eq!(successes[0].1, name, "{case}");
            let errors: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
            assert_eq!(errors.len(), 1, "{case}");
            assert!(errors[0].contains("already been used"), "{case}");
        }
    }

    runtime().block_on(async {
        let verifier = AdvancedSqliteSession::open_with(session_id, &path, true).unwrap();
        let copied = &items[..2 * (turn_number as usize - 1)];
        for (worker, branch_id) in successes {
            let stored = verifier
                .get_items_for_branch(None, Some(branch_id))
                .await
                .unwrap();
            let mut expected = contents(copied);
            expected.push(format!("{worker} branch item"));
            assert_eq!(contents(&stored), expected, "{case}");
        }
        let main = verifier
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap();
        assert_eq!(contents(&main), contents(&items), "{case}");
    });
}

#[test]
fn branch_allocation_is_serialized_between_connections() {
    for round in 0..ROUNDS {
        for branch_name in [None, Some("shared_branch")] {
            for turn_number in [1, 3] {
                branch_allocation_round(round, branch_name, turn_number);
            }
        }
    }
}

fn owner_tables(path: &Path, table: &str) -> Vec<(String, String)> {
    let conn = Connection::open(path).unwrap();
    let mut statement = conn
        .prepare(&format!("PRAGMA foreign_key_list({table})"))
        .unwrap();
    let mut owners: Vec<(String, String)> = statement
        .query_map([], |row| Ok((row.get(3)?, row.get(2)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    owners.sort();
    owners
}

#[test]
fn concurrent_structure_table_claims_leave_one_coherent_owner() {
    let pairs = [("a_sessions", "a_messages"), ("b_sessions", "b_messages")];
    for round in 0..ROUNDS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("advanced_concurrent_claim.db");
        let barrier = Arc::new(Barrier::new(2));
        let workers: Vec<_> = pairs
            .into_iter()
            .map(|(sessions, messages)| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    AdvancedSqliteSession::open_with_tables(
                        "concurrent",
                        &path,
                        sessions,
                        messages,
                        true,
                    )
                    .map(|_| (sessions, messages))
                    .map_err(|error| error.to_string())
                })
            })
            .collect();
        let outcomes: Vec<Result<(&str, &str), String>> =
            workers.into_iter().map(|w| w.join().unwrap()).collect();
        let claimed: Vec<(&str, &str)> = outcomes
            .iter()
            .filter_map(|outcome| outcome.as_ref().ok().copied())
            .collect();
        assert_eq!(claimed.len(), 1, "round {round}: {outcomes:?}");

        let (winner_sessions, winner_messages) = claimed[0];
        let (loser_sessions, loser_messages) =
            pairs.into_iter().find(|pair| *pair != claimed[0]).unwrap();
        assert_eq!(
            owner_tables(&path, "message_structure"),
            vec![
                ("message_id".to_string(), winner_messages.to_string()),
                ("session_id".to_string(), winner_sessions.to_string()),
            ],
            "round {round}"
        );
        assert_eq!(
            owner_tables(&path, "turn_usage"),
            vec![("session_id".to_string(), winner_sessions.to_string())],
            "round {round}"
        );
        let conn = Connection::open(&path).unwrap();
        let leaked: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name IN (?, ?, ?)",
                [
                    loser_sessions.to_string(),
                    loser_messages.to_string(),
                    format!("idx_{loser_messages}_session_id"),
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(leaked, 0, "round {round}");
    }
}

/// `test_add_items_rolls_back_partial_structure_metadata_write`: the second structure row of a
/// batch fails after the first was written, and the message rows go with it.
#[test]
fn a_partial_structure_write_rolls_back_with_the_message_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial.db");
    runtime().block_on(async {
        let session =
            AdvancedSqliteSession::open_with("advanced_add_items_partial_metadata", &path, true)
                .unwrap();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_second_structure_row BEFORE INSERT ON message_structure
             WHEN (SELECT COUNT(*) FROM message_structure) >= 1
             BEGIN SELECT RAISE(ABORT, 'structure metadata failed after partial write'); END;",
        )
        .unwrap();

        let error = session
            .add_items(vec![
                InputItem::user("not saved"),
                InputItem::from_value(
                    serde_json::json!({"role": "assistant", "content": "also not saved"}),
                ),
            ])
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("structure metadata failed after partial write"),
            "{error}"
        );
        assert!(session.get_items(None).await.unwrap().is_empty());
        let count = |table: &str| -> i64 {
            conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count("agent_messages"), 0);
        assert_eq!(count("message_structure"), 0);
    });
}
