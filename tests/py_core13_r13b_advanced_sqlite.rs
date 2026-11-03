//! Cases of `tests/extensions/memory/test_advanced_sqlite_session.py` that earlier audits
//! carried over as python-only: logging under the model-data policy, a usage failure that never
//! breaks the caller, operations queued on one session instance (the branch and turn a call
//! selects stay its target), several sessions racing on one database file, and a usage write
//! that waits for the database lock without blocking the runtime.
//!
//! Python gates private worker functions to force an interleaving; Rust drives the same
//! interleavings with `join!`, whose first-poll order is deterministic.
#![cfg(feature = "sqlite")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex, OnceLock};

use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::memory::{AdvancedSqliteSession, Session};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{AgentsError, InputItem};
use rusqlite::Connection;
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::item_text;

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": text}))
}

fn usage() -> &'static Usage {
    static USAGE: OnceLock<Usage> = OnceLock::new();
    USAGE.get_or_init(|| Usage {
        requests: 1,
        input_tokens: 50,
        output_tokens: 30,
        total_tokens: 80,
        ..Default::default()
    })
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn file_session(dir: &tempfile::TempDir, name: &str) -> (AdvancedSqliteSession, PathBuf) {
    let path = dir.path().join(format!("{name}.db"));
    (AdvancedSqliteSession::open(name, &path).unwrap(), path)
}

fn count(path: &Path, table: &str) -> i64 {
    Connection::open(path)
        .unwrap()
        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

async fn turn_tokens(session: &AdvancedSqliteSession, branch: &str) -> Vec<(i64, u64)> {
    session
        .get_turn_usage(None, Some(branch))
        .await
        .unwrap()
        .iter()
        .map(|row| (row.user_turn_number, row.total_tokens))
        .collect()
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS
            .lock()
            .unwrap()
            .push(format!("{} {}", record.level(), record.args()));
    }
    fn flush(&self) {}
}

fn install_logger() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

fn take_records() -> String {
    std::mem::take(&mut *RECORDS.lock().unwrap()).join("\n")
}

#[tokio::test]
async fn branch_creation_log_follows_the_model_data_policy() {
    let _guard = LOCK.lock().await;
    install_logger();
    for redacted in [true, false] {
        set_dont_log_model_data(redacted);
        let session = AdvancedSqliteSession::in_memory("advanced_branch_logging").unwrap();
        session
            .add_items(vec![
                user("SECRET_BRANCH_TURN_CONTENT"),
                assistant("response"),
            ])
            .await
            .unwrap();
        take_records();
        session
            .create_branch_from_turn(1, Some("branch"))
            .await
            .unwrap();
        let logged = take_records();
        assert!(
            logged.contains("Created branch 'branch' from turn 1"),
            "{logged}"
        );
        assert_eq!(
            !logged.contains("SECRET_BRANCH_TURN_CONTENT"),
            redacted,
            "redacted={redacted}: {logged}"
        );
    }
    reset_dont_log_model_data();
}

#[tokio::test]
async fn a_failed_usage_write_is_logged_not_raised() {
    let _guard = LOCK.lock().await;
    install_logger();
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "usage_failure");
    session.add_items(vec![user("turn")]).await.unwrap();
    Connection::open(&path)
        .unwrap()
        .execute_batch(
            "CREATE TRIGGER fail_turn_usage BEFORE INSERT ON turn_usage
             BEGIN SELECT RAISE(ABORT, 'SECRET_USAGE_FAILURE'); END;",
        )
        .unwrap();

    for redacted in [true, false] {
        set_dont_log_model_data(redacted);
        take_records();
        session.store_run_usage(usage()).await.unwrap();
        let logged = take_records();
        assert!(logged.contains("Failed to store session usage"), "{logged}");
        assert_eq!(
            logged.contains("SECRET_USAGE_FAILURE"),
            !redacted,
            "redacted={redacted}: {logged}"
        );
    }
    reset_dont_log_model_data();
    assert!(session.get_turn_usage(None, None).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_failed_add_is_logged_under_the_policy_and_still_returned() {
    let _guard = LOCK.lock().await;
    install_logger();
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "add_failure");
    Connection::open(&path)
        .unwrap()
        .execute_batch(
            "CREATE TRIGGER fail_structure BEFORE INSERT ON message_structure
             BEGIN SELECT RAISE(ABORT, 'SECRET_STRUCTURE_FAILURE'); END;",
        )
        .unwrap();
    for redacted in [true, false] {
        set_dont_log_model_data(redacted);
        set_dont_log_tool_data(redacted);
        take_records();
        let error = session.add_items(vec![user("turn")]).await.unwrap_err();
        assert!(
            error.to_string().contains("SECRET_STRUCTURE_FAILURE"),
            "{error}"
        );
        let logged = take_records();
        assert!(logged.contains("Failed to add session items"), "{logged}");
        assert_eq!(
            logged.contains("SECRET_STRUCTURE_FAILURE"),
            !redacted,
            "{logged}"
        );
    }
    reset_dont_log_tool_data();
    reset_dont_log_model_data();
    assert_eq!(
        count(&path, "agent_messages"),
        0,
        "the failed add rolled back"
    );
}

// ---------------------------------------------------------------------------
// Operations queued on one instance
// ---------------------------------------------------------------------------

fn main_turns() -> Vec<InputItem> {
    vec![user("u1"), assistant("a1"), user("u2"), assistant("a2")]
}

#[tokio::test]
async fn a_pop_keeps_the_branch_selected_when_it_was_called() {
    // `test_pop_item_uses_branch_snapshot_when_branch_switches_concurrently`.
    let session = AdvancedSqliteSession::in_memory("pop_snapshot_test").unwrap();
    session.add_items(main_turns()).await.unwrap();
    session
        .create_branch_from_turn(2, Some("branch_a"))
        .await
        .unwrap();
    session.switch_to_branch("branch_a").await.unwrap();
    session.add_items(vec![user("branch-only")]).await.unwrap();

    let (popped, switched) = tokio::join!(session.pop_item(), session.switch_to_branch("main"));
    switched.unwrap();
    assert_eq!(item_text(&popped.unwrap().unwrap()), "branch-only");
    assert_eq!(
        texts(
            &session
                .get_items_for_branch(None, Some("main"))
                .await
                .unwrap()
        ),
        ["u1", "a1", "u2", "a2"]
    );
}

#[tokio::test]
async fn an_append_keeps_the_branch_selected_when_it_was_called() {
    // `test_queued_add_keeps_selected_branch`, switch variants.
    for branch_turn in [1, 2] {
        let session = AdvancedSqliteSession::in_memory("queued_add_branch").unwrap();
        let original = vec![user("main first"), user("main second")];
        session.add_items(original.clone()).await.unwrap();
        session
            .create_branch_from_turn(branch_turn, Some("selected"))
            .await
            .unwrap();

        let (stored, added, switched) = tokio::join!(
            session.store_run_usage(usage()),
            session.add_items(vec![user("queued branch item")]),
            session.switch_to_branch("main"),
        );
        stored.unwrap();
        added.unwrap();
        switched.unwrap();

        assert_eq!(session.current_branch(), "main");
        assert_eq!(
            texts(&session.get_items(None).await.unwrap()),
            ["main first", "main second"]
        );
        let mut expected: Vec<String> = original[..branch_turn as usize - 1]
            .iter()
            .map(item_text)
            .collect();
        expected.push("queued branch item".into());
        assert_eq!(
            texts(
                &session
                    .get_items_for_branch(None, Some("selected"))
                    .await
                    .unwrap()
            ),
            expected,
            "branch_turn={branch_turn}"
        );
    }
}

#[tokio::test]
async fn a_usage_write_keeps_the_branch_selected_when_it_was_called() {
    // `test_store_run_usage_keeps_target_while_capture_is_waiting`, switch variant.
    let session = AdvancedSqliteSession::in_memory("usage_capture_switch").unwrap();
    session.add_items(vec![user("main turn")]).await.unwrap();
    session
        .create_branch_from_turn(1, Some("branch_a"))
        .await
        .unwrap();
    session.add_items(vec![user("branch turn")]).await.unwrap();

    let (stored, switched) = tokio::join!(
        session.store_run_usage(usage()),
        session.switch_to_branch("main")
    );
    stored.unwrap();
    switched.unwrap();

    assert!(turn_tokens(&session, "main").await.is_empty());
    assert_eq!(turn_tokens(&session, "branch_a").await, [(1, 80)]);
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        ["main turn"]
    );
}

#[tokio::test]
async fn later_history_changes_do_not_move_an_earlier_usage_write() {
    // `test_store_run_usage_keeps_turn_when_history_changes_during_capture`.
    for replace in [false, true] {
        let session = AdvancedSqliteSession::in_memory("capture_turn").unwrap();
        session
            .add_items(vec![user("original turn")])
            .await
            .unwrap();
        let newer = Usage {
            requests: 1,
            input_tokens: 7,
            output_tokens: 4,
            total_tokens: 11,
            ..Default::default()
        };

        let history = async {
            if replace {
                session.pop_item().await.unwrap();
            }
            session.add_items(vec![user("newer turn")]).await.unwrap();
            session.store_run_usage(&newer).await.unwrap();
        };
        let (first, ()) = tokio::join!(session.store_run_usage(usage()), history);
        first.unwrap();

        if replace {
            assert_eq!(turn_tokens(&session, "main").await, [(1, 11)]);
            assert_eq!(
                texts(&session.get_items(None).await.unwrap()),
                ["newer turn"]
            );
        } else {
            assert_eq!(turn_tokens(&session, "main").await, [(1, 80), (2, 11)]);
        }
    }
}

#[tokio::test]
async fn a_queued_append_cannot_resurrect_a_deleted_branch() {
    // `test_queued_append_cannot_resurrect_deleted_branch`.
    for force in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (session, path) = file_session(&dir, "queued_append_delete");
        let main_items = vec![user("main first"), user("main second")];
        session.add_items(main_items.clone()).await.unwrap();
        session
            .create_branch_from_turn(2, Some("selected"))
            .await
            .unwrap();

        let (stored, appended, deleted) = tokio::join!(
            session.store_run_usage(usage()),
            session.add_items(vec![user("queued branch item")]),
            async {
                if !force {
                    session.switch_to_branch("main").await?;
                }
                session.delete_branch("selected", force).await
            },
        );
        stored.unwrap();
        appended.unwrap();
        deleted.unwrap_or_else(|error| panic!("force={force}: {error}"));

        assert_eq!(session.current_branch(), "main");
        assert_eq!(
            texts(&session.get_items(None).await.unwrap()),
            ["main first", "main second"]
        );
        assert!(
            session
                .get_items_for_branch(None, Some("selected"))
                .await
                .unwrap()
                .is_empty()
        );
        let branches: Vec<String> = session
            .list_branches()
            .await
            .unwrap()
            .into_iter()
            .map(|b| b.branch_id)
            .collect();
        assert_eq!(branches, ["main"], "force={force}");
        assert!(turn_tokens(&session, "selected").await.is_empty());
        assert_eq!(count(&path, "agent_messages"), main_items.len() as i64);
        assert_eq!(count(&path, "message_structure"), main_items.len() as i64);
        assert_eq!(count(&path, "turn_usage"), 0);
    }
}

#[tokio::test]
async fn a_failed_queued_append_releases_the_operations_behind_it() {
    // `test_queued_turn_mutation_releases_later_usage_capture`, failure variant.
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "queued_turn_failure");
    session
        .add_items(vec![user("original turn"), assistant("original response")])
        .await
        .unwrap();
    Connection::open(&path)
        .unwrap()
        .execute_batch(
            "CREATE TRIGGER fail_next BEFORE INSERT ON message_structure
             WHEN NEW.message_type = 'user'
             BEGIN SELECT RAISE(ABORT, 'structure metadata failed'); END;",
        )
        .unwrap();

    let (popped, appended, stored) = tokio::join!(
        session.pop_item(),
        session.add_items(vec![user("next turn")]),
        session.store_run_usage(usage()),
    );
    assert_eq!(item_text(&popped.unwrap().unwrap()), "original response");
    let error: AgentsError = appended.unwrap_err();
    assert!(
        error.to_string().contains("structure metadata failed"),
        "{error}"
    );
    stored.unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        ["original turn"]
    );
    assert_eq!(turn_tokens(&session, "main").await, [(1, 80)]);
}

#[tokio::test]
async fn a_dropped_queued_append_never_runs_and_releases_the_queue() {
    // The caller cancelling an append that is still waiting behind another operation: it has
    // no effect, and the operations behind it proceed.
    let session = AdvancedSqliteSession::in_memory("queued_turn_cancel").unwrap();
    session
        .add_items(vec![user("original turn")])
        .await
        .unwrap();

    let mut predecessor = Box::pin(session.pop_item());
    assert!(futures::poll!(predecessor.as_mut()).is_pending());
    let mut queued = Box::pin(session.add_items(vec![user("never added")]));
    assert!(futures::poll!(queued.as_mut()).is_pending());
    drop(queued);

    assert_eq!(
        item_text(&predecessor.await.unwrap().unwrap()),
        "original turn"
    );
    session.add_items(vec![user("after")]).await.unwrap();
    session.store_run_usage(usage()).await.unwrap();
    assert_eq!(texts(&session.get_items(None).await.unwrap()), ["after"]);
    assert_eq!(turn_tokens(&session, "main").await, [(1, 80)]);
}

#[test]
fn a_session_serializes_its_operations_on_every_runtime_it_is_used_from() {
    // `test_turn_ordering_survives_reuse_on_another_event_loop`: the ordering state is not tied
    // to the runtime that first used the session.
    let session = Arc::new(AdvancedSqliteSession::in_memory("usage_two_loops").unwrap());
    for original_turn in [1, 3] {
        let session = session.clone();
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    session
                        .add_items(vec![user(&format!("usage owner {original_turn}"))])
                        .await
                        .unwrap();
                    let (stored, appended) = tokio::join!(
                        session.store_run_usage(usage()),
                        session.add_items(vec![user(&format!("following {original_turn}"))]),
                    );
                    stored.unwrap();
                    appended.unwrap();
                });
        })
        .join()
        .unwrap();
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        assert_eq!(turn_tokens(&session, "main").await, [(1, 80), (3, 80)]);
    });
}

#[tokio::test]
async fn a_clear_that_wins_leaves_no_branch_reservation_behind() {
    // `test_clear_before_branch_transaction_prevents_stale_reservation`: a branch request that
    // reaches the database after a clear fails validation and reserves nothing.
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "clear_before_branch");
    session
        .add_items(vec![user("u1"), assistant("a1")])
        .await
        .unwrap();
    session.clear_session().await.unwrap();

    let error = session
        .create_branch_from_turn(1, Some("stale_branch"))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not contain a user message"),
        "{error}"
    );
    assert_eq!(session.current_branch(), "main");
    assert!(session.list_branches().await.unwrap().is_empty());
    let reserved: Vec<String> = {
        let conn = Connection::open(&path).unwrap();
        let mut stmt = conn
            .prepare("SELECT branch_id FROM branch_reservations WHERE session_id = ?")
            .unwrap();
        stmt.query_map(["clear_before_branch"], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    assert_eq!(reserved, ["main"]);
}

#[tokio::test]
async fn switching_and_clearing_at_once_never_leave_the_pointer_on_a_removed_branch() {
    // `test_stale_switch_after_clear_does_not_repoint_to_deleted_branch` and
    // `test_stale_create_branch_after_clear_does_not_repoint`: whichever lands first, a clear
    // wins over a switch or a branch creation that was already validated.
    for round in 0..40 {
        let session = AdvancedSqliteSession::in_memory("stale_pointer").unwrap();
        session.add_items(main_turns()).await.unwrap();
        session
            .create_branch_from_turn(2, Some("branch_a"))
            .await
            .unwrap();
        session.switch_to_branch("main").await.unwrap();
        let (_switch, cleared) = tokio::join!(
            session.switch_to_branch("branch_a"),
            session.clear_session()
        );
        cleared.unwrap();
        assert_eq!(session.current_branch(), "main", "switch round {round}");
        assert!(session.get_items(None).await.unwrap().is_empty());

        let session = AdvancedSqliteSession::in_memory("stale_create").unwrap();
        session.add_items(main_turns()).await.unwrap();
        let (_create, cleared) = tokio::join!(
            session.create_branch_from_turn(2, Some("branch_b")),
            session.clear_session()
        );
        cleared.unwrap();
        assert_eq!(session.current_branch(), "main", "create round {round}");
        assert!(session.get_items(None).await.unwrap().is_empty());
    }
}

// ---------------------------------------------------------------------------
// Several sessions on one database file
// ---------------------------------------------------------------------------

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
}

#[test]
fn concurrent_branch_creation_reserves_each_id_once() {
    // `test_branch_allocation_is_serialized_across_processes`: separate session instances on
    // one file (the Python test uses separate processes) race to create a branch.
    for named in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("branch_allocation.db");
        let setup = AdvancedSqliteSession::open("allocation", &path).unwrap();
        runtime()
            .block_on(setup.add_items(vec![user("u1"), assistant("a1"), user("u2")]))
            .unwrap();
        drop(setup);

        let barrier = Arc::new(Barrier::new(2));
        let workers: Vec<_> = ["first", "second"]
            .into_iter()
            .map(|worker| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let session = AdvancedSqliteSession::open("allocation", &path).unwrap();
                    barrier.wait();
                    runtime().block_on(async {
                        let created = session
                            .create_branch_from_turn(2, named.then_some("shared-name"))
                            .await;
                        if let Ok(branch) = &created {
                            session
                                .add_items(vec![user(&format!("{worker} branch item"))])
                                .await
                                .unwrap();
                            let items = session
                                .get_items_for_branch(None, Some(branch))
                                .await
                                .unwrap();
                            assert_eq!(
                                texts(&items),
                                ["u1", "a1", &format!("{worker} branch item")]
                            );
                        }
                        created.map_err(|error| error.to_string())
                    })
                })
            })
            .collect();
        let outcomes: Vec<Result<String, String>> =
            workers.into_iter().map(|w| w.join().unwrap()).collect();

        if named {
            let ok: Vec<_> = outcomes.iter().filter_map(|o| o.as_ref().ok()).collect();
            assert_eq!(ok, ["shared-name"], "{outcomes:?}");
            let errors: Vec<_> = outcomes.iter().filter_map(|o| o.as_ref().err()).collect();
            assert_eq!(errors.len(), 1);
            assert!(errors[0].contains("already been used"), "{errors:?}");
        } else {
            let mut ids: Vec<_> = outcomes
                .iter()
                .map(|o| o.clone().expect("generated ids never collide"))
                .collect();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), 2, "{outcomes:?}");
        }
        let main = AdvancedSqliteSession::open("allocation", &path).unwrap();
        assert_eq!(
            texts(&runtime().block_on(main.get_items(None)).unwrap()),
            ["u1", "a1", "u2"]
        );
    }
}

#[test]
fn a_pop_is_claimed_by_exactly_one_of_two_sessions_on_one_file() {
    // `test_pop_item_claim_is_unique_across_processes`.
    for round in 0..10 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pop_claim.db");
        let setup = AdvancedSqliteSession::open("pop_claim", &path).unwrap();
        runtime()
            .block_on(setup.add_items(vec![user("only")]))
            .unwrap();
        drop(setup);

        let barrier = Arc::new(Barrier::new(2));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let session = AdvancedSqliteSession::open("pop_claim", &path).unwrap();
                    barrier.wait();
                    runtime()
                        .block_on(session.pop_item())
                        .unwrap()
                        .map(|item| item_text(&item))
                })
            })
            .collect();
        let popped: Vec<Option<String>> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(
            popped
                .iter()
                .filter(|p| p.as_deref() == Some("only"))
                .count(),
            1,
            "round {round}: {popped:?}"
        );
        assert_eq!(
            popped.iter().filter(|p| p.is_none()).count(),
            1,
            "round {round}"
        );
    }
}

#[test]
fn concurrent_claims_of_a_fresh_file_leave_one_coherent_owner() {
    // `test_concurrent_structure_table_claims_leave_one_coherent_owner`.
    let pairs = [("a_sessions", "a_messages"), ("b_sessions", "b_messages")];
    for round in 0..10 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent_claim.db");
        let barrier = Arc::new(Barrier::new(2));
        let workers: Vec<_> = pairs
            .into_iter()
            .map(|(sessions, messages)| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    AdvancedSqliteSession::open_with_tables(
                        "claim", &path, sessions, messages, true,
                    )
                    .map(|_| (sessions, messages))
                    .map_err(|error| error.to_string())
                })
            })
            .collect();
        let outcomes: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        let claimed: Vec<_> = outcomes.iter().filter_map(|o| o.as_ref().ok()).collect();
        assert_eq!(claimed.len(), 1, "round {round}: {outcomes:?}");
        let (winner_sessions, winner_messages) = *claimed[0];
        let (loser_sessions, loser_messages) = *pairs.iter().find(|p| **p != *claimed[0]).unwrap();

        let conn = Connection::open(&path).unwrap();
        let owners = |table: &str| -> Vec<(String, String)> {
            let mut stmt = conn
                .prepare(&format!("PRAGMA foreign_key_list({table})"))
                .unwrap();
            let mut rows: Vec<(String, String)> = stmt
                .query_map([], |r| Ok((r.get::<_, String>(3)?, r.get::<_, String>(2)?)))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            rows.sort();
            rows
        };
        assert_eq!(
            owners("message_structure"),
            [
                ("message_id".to_string(), winner_messages.to_string()),
                ("session_id".to_string(), winner_sessions.to_string())
            ],
            "round {round}"
        );
        assert_eq!(
            owners("turn_usage"),
            [("session_id".to_string(), winner_sessions.to_string())]
        );
        let leftovers: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name IN (?, ?, ?)",
                [
                    loser_sessions,
                    loser_messages,
                    &format!("idx_{loser_messages}_session_id"),
                ],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(leftovers, 0, "round {round}");
    }
}

#[tokio::test]
async fn deleting_a_branch_with_many_branch_only_messages_removes_them_all() {
    // `test_orphan_cleanup_uses_set_based_delete_for_many_messages`: cleanup must not need one
    // statement parameter per orphaned row.
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "many_orphans");
    session.add_items(vec![user("main turn")]).await.unwrap();
    session
        .create_branch_from_turn(1, Some("big"))
        .await
        .unwrap();
    // Creating a branch switches to it.
    assert_eq!(session.current_branch(), "big");
    let many: Vec<InputItem> = (0..40_000).map(|i| user(&format!("orphan {i}"))).collect();
    session.add_items(many).await.unwrap();
    assert_eq!(count(&path, "agent_messages"), 40_001);

    session.switch_to_branch("main").await.unwrap();
    session.delete_branch("big", false).await.unwrap();
    assert_eq!(count(&path, "agent_messages"), 1);
    assert_eq!(count(&path, "message_structure"), 1);
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        ["main turn"]
    );
}

// ---------------------------------------------------------------------------
// The runtime stays responsive while the database is locked
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn a_usage_write_waits_for_the_database_lock_without_blocking_the_runtime() {
    // `test_store_run_usage_waits_for_database_lock_off_event_loop`.
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "responsive_usage");
    session.add_items(vec![user("first turn")]).await.unwrap();

    let locker = Connection::open(&path).unwrap();
    locker.execute_batch("BEGIN IMMEDIATE").unwrap();
    let release = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(400));
        locker.execute_batch("ROLLBACK").unwrap();
    });

    let ticks = Arc::new(Mutex::new(0u32));
    let ticker = {
        let ticks = ticks.clone();
        async move {
            for _ in 0..8 {
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                *ticks.lock().unwrap() += 1;
            }
        }
    };
    let (stored, ()) = tokio::join!(session.store_run_usage(usage()), ticker);
    stored.unwrap();
    release.join().unwrap();

    assert_eq!(
        *ticks.lock().unwrap(),
        8,
        "the runtime kept turning while the write waited"
    );
    assert_eq!(turn_tokens(&session, "main").await, [(1, 80)]);
}
