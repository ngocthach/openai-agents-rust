//! Parity port of the last unported tests of
//! `tests/extensions/memory/test_advanced_sqlite_session.py`: Runner-driven tool usage by
//! branch, branch-reservation rollback and legacy backfill, corrupt-row limits, concurrent
//! writers, usage bookkeeping after pop/reopen/branch deletion, and rollback after a failure
//! partway through a destructive operation.
//!
//! Python injects faults by monkeypatching private methods (`_reserve_branch_id`,
//! `_cleanup_orphaned_messages_sync`, a wrapped connection). Rust exposes no such seams, so the
//! same failures are produced with SQLite triggers that abort the statement that follows the
//! earlier writes; the assertions (nothing persisted, no open transaction, session still
//! usable) are Python's.
//!
//! Documented divergence: `store_run_usage` takes an explicit `&Usage` and captures the turn
//! and writes the row in one blocking call, so Python's gated "usage read the turn, then a pop
//! raced it" tests (`_gate_worker("_update_sync")`) have no Rust equivalent. The observable
//! halves (usage rows follow their turn, nothing is attributed to a removed turn) are ported.
#![cfg(feature = "sqlite")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::memory::{AdvancedSqliteSession, Session};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ShellCallOutcome, ShellCommandOutput, ShellResult, ShellTool,
};
use openai_agents_sdk::usage::{InputTokensDetails, OutputTokensDetails, Usage};
use openai_agents_sdk::{Agent, InputItem, RunOptions, Runner};
use rusqlite::Connection;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::item_text;

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": text}))
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn usage_data() -> Usage {
    Usage {
        requests: 1,
        input_tokens: 50,
        output_tokens: 30,
        total_tokens: 80,
        input_tokens_details: InputTokensDetails {
            cached_tokens: 10,
            cache_write_tokens: 0,
        },
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 5,
        },
        ..Default::default()
    }
}

/// Three user turns with assistant replies, Python's `_branch_collision_items`.
fn branch_collision_items() -> Vec<InputItem> {
    vec![
        user("Turn one question"),
        assistant("Turn one answer"),
        user("Turn two question"),
        assistant("Turn two answer"),
        user("Turn three question"),
        assistant("Turn three answer"),
    ]
}

fn db_path(dir: &tempfile::TempDir, name: &str) -> PathBuf {
    dir.path().join(format!("{name}.db"))
}

fn raw(path: &Path) -> Connection {
    Connection::open(path).unwrap()
}

fn count(path: &Path, table: &str, session_id: &str) -> i64 {
    raw(path)
        .query_row(
            &format!("SELECT COUNT(*) FROM {table} WHERE session_id = ?"),
            [session_id],
            |r| r.get(0),
        )
        .unwrap()
}

/// True when an independent writer can take the SQLite write lock at once, which proves the
/// session did not leave a transaction open.
fn write_lock_is_free(path: &Path) -> bool {
    let conn = raw(path);
    conn.busy_handler(None).unwrap();

    conn.execute_batch("BEGIN IMMEDIATE; ROLLBACK;").is_ok()
}

fn already_used(error: openai_agents_sdk::AgentsError) -> bool {
    error.to_string().contains("already been used")
}

// ---------------------------------------------------------------------------
// Runner-driven tool usage by branch
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RecordingEditor {
    operations: Mutex<Vec<ApplyPatchOperation>>,
}

/// A shareable handle, so the test can read the operations after the tool owns the editor.
#[derive(Clone, Default)]
struct SharedEditor(Arc<RecordingEditor>);

impl std::ops::Deref for SharedEditor {
    type Target = RecordingEditor;
    fn deref(&self) -> &RecordingEditor {
        &self.0
    }
}

#[async_trait]
impl ApplyPatchEditor for SharedEditor {
    async fn create_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        self.operations.lock().unwrap().push(operation.clone());
        Ok(ApplyPatchResult::from(format!(
            "Created {}",
            operation.path
        )))
    }
    async fn update_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        self.operations.lock().unwrap().push(operation.clone());
        Ok(ApplyPatchResult::completed(format!(
            "Updated {}",
            operation.path
        )))
    }
    async fn delete_file(&self, operation: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        self.operations.lock().unwrap().push(operation.clone());
        Ok(ApplyPatchResult::from(format!(
            "Deleted {}",
            operation.path
        )))
    }
}

fn shell_call(call_id: &str) -> Value {
    json!({
        "type": "shell_call",
        "id": call_id,
        "call_id": call_id,
        "status": "completed",
        "action": {"type": "exec", "commands": ["echo test"], "timeout_ms": 1000},
    })
}

fn apply_patch_call(call_id: &str) -> Value {
    json!({
        "type": "apply_patch_call",
        "call_id": call_id,
        "status": "completed",
        "operation": {"type": "update_file", "path": "test.md", "diff": "-a\n+b\n"},
    })
}

fn pairs(usage: Vec<openai_agents_sdk::memory::ToolUsage>) -> Vec<(String, u64, i64)> {
    usage
        .into_iter()
        .map(|u| (u.tool_name, u.count, u.user_turn_number))
        .collect()
}

#[tokio::test]
async fn tool_usage_tracks_runner_shell_and_patch_calls_by_branch() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        OutputItem::from_value(shell_call("shell-1")),
        OutputItem::from_value(apply_patch_call("patch-1")),
    ]);
    model.add_turn(vec![assistant_message("Updated.")]);
    model.add_turn(vec![OutputItem::from_value(shell_call("shell-2"))]);
    model.add_turn(vec![assistant_message("Checked.")]);

    let editor = SharedEditor::default();
    let shell_calls = Arc::new(Mutex::new(0usize));
    let counted = shell_calls.clone();
    let shell = ShellTool::<()>::new(move |_req| {
        let counted = counted.clone();
        async move {
            *counted.lock().unwrap() += 1;
            Ok::<_, openai_agents_sdk::AgentsError>(ShellResult {
                output: vec![ShellCommandOutput {
                    stdout: "test".into(),
                    stderr: String::new(),
                    outcome: ShellCallOutcome::Exit { exit_code: Some(0) },
                    command: Some("echo test".into()),
                    provider_data: None,
                }],
                ..Default::default()
            })
        }
    });
    let agent = Agent::<()>::new("coding")
        .with_model_instance(model.clone())
        .with_tool(shell)
        .with_tool(ApplyPatchTool::<()>::new(editor.clone()));
    let session = Arc::new(AdvancedSqliteSession::in_memory("runner-tools").unwrap());

    let result = Runner::run_with_options(
        &agent,
        "Update the file.",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Updated."));
    assert_eq!(*shell_calls.lock().unwrap(), 1);
    assert_eq!(editor.operations.lock().unwrap().len(), 1);

    let items = session.get_items(None).await.unwrap();
    let types: std::collections::BTreeSet<String> = items
        .iter()
        .filter_map(|i| i.item_type().map(str::to_string))
        .collect();
    for expected in [
        "shell_call",
        "shell_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
    ] {
        assert!(
            types.contains(expected),
            "{expected} missing from {types:?}"
        );
    }
    let expected_main = vec![
        ("apply_patch_call".to_string(), 1, 1),
        ("shell_call".to_string(), 1, 1),
    ];
    let mut main = pairs(session.get_tool_usage(None).await.unwrap());
    main.sort();
    assert_eq!(main, expected_main);

    let branch = session
        .create_branch_from_turn(1, Some("alternative"))
        .await
        .unwrap();
    Runner::run_with_options(
        &agent,
        "Check the file.",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(
        pairs(session.get_tool_usage(None).await.unwrap()),
        vec![("shell_call".to_string(), 1, 1)]
    );
    assert_eq!(
        pairs(session.get_tool_usage(Some(&branch)).await.unwrap()),
        vec![("shell_call".to_string(), 1, 1)]
    );

    session
        .add_items(vec![
            user("Check again."),
            InputItem::from_value(shell_call("shell-3")),
        ])
        .await
        .unwrap();
    assert_eq!(
        pairs(session.get_tool_usage(None).await.unwrap()),
        vec![
            ("shell_call".to_string(), 1, 1),
            ("shell_call".to_string(), 1, 2)
        ]
    );
    let mut main = pairs(session.get_tool_usage(Some("main")).await.unwrap());
    main.sort();
    assert_eq!(main, expected_main);
}

// ---------------------------------------------------------------------------
// Branch reservations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_failed_branch_reservation_rolls_back_and_allows_retry() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "failed_branch_reservation");
    let session = AdvancedSqliteSession::open("failed_branch_reservation", &path).unwrap();
    let items = branch_collision_items();
    session.add_items(items.clone()).await.unwrap();

    // The reservation insert runs first; the structure copy that follows it fails.
    let conn = raw(&path);
    conn.execute_batch(
        "CREATE TRIGGER fail_branch_copy BEFORE INSERT ON message_structure
         WHEN NEW.branch_id = 'retryable_branch'
         BEGIN SELECT RAISE(ABORT, 'failed after reservation'); END;",
    )
    .unwrap();

    let error = session
        .create_branch_from_turn(3, Some("retryable_branch"))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("failed after reservation"),
        "{error}"
    );

    let reserved: i64 = raw(&path)
        .query_row(
            "SELECT COUNT(*) FROM branch_reservations WHERE session_id = ? AND branch_id = ?",
            ["failed_branch_reservation", "retryable_branch"],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(reserved, 0, "the failed reservation must not burn the id");
    assert!(
        session
            .get_items_for_branch(None, Some("retryable_branch"))
            .await
            .unwrap()
            .is_empty()
    );
    assert!(write_lock_is_free(&path));
    assert_eq!(session.current_branch(), "main");

    conn.execute_batch("DROP TRIGGER fail_branch_copy").unwrap();
    assert_eq!(
        session
            .create_branch_from_turn(3, Some("retryable_branch"))
            .await
            .unwrap(),
        "retryable_branch"
    );
    assert_eq!(
        session
            .get_items_for_branch(None, Some("retryable_branch"))
            .await
            .unwrap(),
        items[..4].to_vec()
    );
}

/// Create a populated database with `legacy_branch`, then drop the reservation table the way a
/// database written before reservations existed would look.
async fn legacy_database(dir: &tempfile::TempDir, name: &str, empty_main: bool) -> PathBuf {
    let path = db_path(dir, name);
    let setup = AdvancedSqliteSession::open(name, &path).unwrap();
    setup.add_items(branch_collision_items()).await.unwrap();
    setup
        .create_branch_from_turn(3, Some("legacy_branch"))
        .await
        .unwrap();
    if empty_main {
        setup.switch_to_branch("main").await.unwrap();
        while setup.pop_item().await.unwrap().is_some() {}
    }
    drop(setup);
    raw(&path)
        .execute_batch("DROP TABLE branch_reservations")
        .unwrap();
    path
}

#[tokio::test]
async fn branch_reservations_migrate_existing_populated_branches() {
    let dir = tempfile::tempdir().unwrap();
    let name = "branch_reservation_migration";
    let path = legacy_database(&dir, name, false).await;
    // The legacy fixture used `legacy_branch`; Python's names are cosmetic here.
    let session = AdvancedSqliteSession::open_with(name, &path, false).unwrap();

    let error = session
        .create_branch_from_turn(2, Some("legacy_branch"))
        .await
        .unwrap_err();
    assert!(already_used(error));
    session
        .create_branch_from_turn(1, Some("empty_branch"))
        .await
        .unwrap();

    let conn = raw(&path);
    let mut stmt = conn
        .prepare(
            "SELECT branch_id FROM branch_reservations WHERE session_id = ? ORDER BY branch_id",
        )
        .unwrap();
    let reservations: Vec<String> = stmt
        .query_map([name], |r| r.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(reservations, vec!["empty_branch", "legacy_branch", "main"]);
}

async fn backfill_before_destructive(operation: &str) {
    let dir = tempfile::tempdir().unwrap();
    let name = format!("legacy_branch_{operation}");
    let path = legacy_database(&dir, &name, false).await;
    let session = AdvancedSqliteSession::open_with(&name, &path, false).unwrap();

    match operation {
        "clear" => {
            session.clear_session().await.unwrap();
            session.add_items(branch_collision_items()).await.unwrap();
        }
        "delete" => session.delete_branch("legacy_branch", false).await.unwrap(),
        _ => {
            session.switch_to_branch("legacy_branch").await.unwrap();
            while session.pop_item().await.unwrap().is_some() {}
            session.switch_to_branch("main").await.unwrap();
        }
    }

    let error = session
        .create_branch_from_turn(2, Some("legacy_branch"))
        .await
        .unwrap_err();
    assert!(already_used(error), "{operation}: id was not preserved");
}

#[tokio::test]
async fn legacy_branch_ids_are_backfilled_before_clear() {
    backfill_before_destructive("clear").await;
}

#[tokio::test]
async fn legacy_branch_ids_are_backfilled_before_delete() {
    backfill_before_destructive("delete").await;
}

#[tokio::test]
async fn legacy_branch_ids_are_backfilled_before_pop() {
    backfill_before_destructive("pop").await;
}

#[tokio::test]
async fn a_legacy_missing_delete_leaves_the_database_unlocked() {
    let dir = tempfile::tempdir().unwrap();
    let name = "legacy_noop_missing_delete";
    let path = legacy_database(&dir, name, false).await;
    let session = AdvancedSqliteSession::open_with(name, &path, false).unwrap();
    let contender =
        AdvancedSqliteSession::open_with(format!("{name}_contender"), &path, false).unwrap();

    let error = session
        .delete_branch("missing_branch", false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("does not exist"), "{error}");

    assert!(write_lock_is_free(&path));
    contender
        .add_items(vec![user("writer acquired")])
        .await
        .unwrap();
}

#[tokio::test]
async fn a_legacy_empty_pop_leaves_the_database_unlocked() {
    let dir = tempfile::tempdir().unwrap();
    let name = "legacy_noop_empty_pop";
    let path = legacy_database(&dir, name, true).await;
    let session = AdvancedSqliteSession::open_with(name, &path, false).unwrap();
    let contender =
        AdvancedSqliteSession::open_with(format!("{name}_contender"), &path, false).unwrap();

    assert!(session.pop_item().await.unwrap().is_none());

    assert!(write_lock_is_free(&path));
    contender
        .add_items(vec![user("writer acquired")])
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// Reads and writes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_items_limit_skips_corrupt_newest_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "limit_corrupt");
    let session = AdvancedSqliteSession::open("limit_corrupt_test", &path).unwrap();
    session
        .add_items(vec![user("valid 0"), assistant("valid 1"), user("valid 2")])
        .await
        .unwrap();

    // A corrupt newest row with the branch structure the join needs.
    let conn = raw(&path);
    conn.execute(
        "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
        ["limit_corrupt_test", "not valid json {{{"],
    )
    .unwrap();
    let message_id = conn.last_insert_rowid();
    let next_sequence: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM message_structure
             WHERE session_id = ? AND branch_id = ?",
            ["limit_corrupt_test", "main"],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "INSERT INTO message_structure
            (session_id, message_id, branch_id, sequence_number, message_type,
             user_turn_number, branch_turn_number)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            "limit_corrupt_test",
            message_id,
            "main",
            next_sequence,
            "user",
            1,
            1
        ],
    )
    .unwrap();

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
    assert_eq!(
        texts(
            &session
                .get_items_for_branch(Some(2), Some("main"))
                .await
                .unwrap()
        ),
        vec!["valid 1", "valid 2"]
    );
}

#[tokio::test]
async fn concurrent_add_items_keep_messages_and_structure_aligned_for_a_file_db() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "advanced_concurrent");
    let session = Arc::new(AdvancedSqliteSession::open("advanced_concurrent", &path).unwrap());

    let mut tasks = Vec::new();
    let mut expected: Vec<String> = Vec::new();
    for worker in 0..8 {
        let contents: Vec<String> = (0..10)
            .map(|i| format!("worker-{worker}-message-{i}"))
            .collect();
        expected.extend(contents.clone());
        let session = session.clone();
        tasks.push(tokio::spawn(async move {
            session
                .add_items(contents.iter().map(|c| user(c)).collect())
                .await
                .unwrap();
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    expected.sort();

    let mut retrieved = texts(&session.get_items(None).await.unwrap());
    assert_eq!(retrieved.len(), 80);
    retrieved.sort();
    assert_eq!(retrieved, expected);

    let conn = raw(&path);
    let mut stmt = conn
        .prepare(
            "SELECT m.message_data FROM agent_messages m
             JOIN message_structure s ON s.message_id = m.id
             WHERE m.session_id = ? ORDER BY s.sequence_number ASC",
        )
        .unwrap();
    let mut structured: Vec<String> = stmt
        .query_map(["advanced_concurrent"], |r| r.get::<_, String>(0))
        .unwrap()
        .map(|r| {
            let value: Value = serde_json::from_str(&r.unwrap()).unwrap();
            value["content"].as_str().unwrap().to_string()
        })
        .collect();
    structured.sort();
    assert_eq!(structured, expected);
}

#[tokio::test]
async fn a_default_read_does_not_initialize_the_clear_generation_table() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "legacy_generation_read");
    let session = AdvancedSqliteSession::open("legacy_generation_read", &path).unwrap();
    session
        .add_items(vec![user("legacy history")])
        .await
        .unwrap();
    raw(&path)
        .execute_batch("DROP TABLE session_clear_generations")
        .unwrap();

    assert_eq!(
        session.get_items(None).await.unwrap(),
        vec![user("legacy history")]
    );

    let exists: Option<i64> = raw(&path)
        .query_row(
            "SELECT 1 FROM sqlite_master
             WHERE type = 'table' AND name = 'session_clear_generations'",
            [],
            |r| r.get(0),
        )
        .ok();
    assert!(exists.is_none(), "a read must not recreate the table");
}

#[tokio::test]
async fn a_post_clear_switch_synchronizes_the_generation_before_the_next_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "post_clear_branch_switch");
    let owner = AdvancedSqliteSession::open("post_clear_branch_switch", &path).unwrap();
    let other = AdvancedSqliteSession::open_with("post_clear_branch_switch", &path, false).unwrap();

    owner.clear_session().await.unwrap();
    owner
        .add_items(vec![user("u1"), assistant("a1"), user("u2")])
        .await
        .unwrap();
    owner
        .create_branch_from_turn(2, Some("fresh"))
        .await
        .unwrap();

    other.switch_to_branch("fresh").await.unwrap();
    other.add_items(vec![assistant("on fresh")]).await.unwrap();

    assert_eq!(other.current_branch(), "fresh");
    assert_eq!(
        texts(&other.get_items(None).await.unwrap()),
        vec!["u1", "a1", "on fresh"]
    );
    assert_eq!(
        texts(
            &other
                .get_items_for_branch(None, Some("main"))
                .await
                .unwrap()
        ),
        vec!["u1", "a1", "u2"]
    );
}

// ---------------------------------------------------------------------------
// Usage bookkeeping
// ---------------------------------------------------------------------------

async fn store_usage_after_reopening_a_cleared_session(create_tables: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "reopened_usage");
    let items = vec![user("current turn")];
    {
        let original = AdvancedSqliteSession::open("reopened_usage", &path).unwrap();
        original.add_items(vec![user("old turn")]).await.unwrap();
        original.clear_session().await.unwrap();
        original.add_items(items.clone()).await.unwrap();
    }

    let reopened =
        AdvancedSqliteSession::open_with("reopened_usage", &path, create_tables).unwrap();
    // Store before any read can refresh the reopened session's local state.
    reopened.store_run_usage(&usage_data()).await.unwrap();
    let recorded = reopened.get_turn_usage(Some(1), None).await.unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].total_tokens, usage_data().total_tokens);
    assert_eq!(reopened.get_items(None).await.unwrap(), items);
}

#[tokio::test]
async fn store_run_usage_after_reopening_a_cleared_session_without_creating_tables() {
    store_usage_after_reopening_a_cleared_session(false).await;
}

#[tokio::test]
async fn store_run_usage_after_reopening_a_cleared_session_creating_tables() {
    store_usage_after_reopening_a_cleared_session(true).await;
}

#[tokio::test]
async fn usage_is_not_attributed_to_a_turn_removed_by_pop() {
    // Python's gated race (usage read turn 1, pop removed it before the write) cannot happen
    // here: the turn is read and the row written in one call. The sequential half is ported.
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "stale_usage");
    let session = AdvancedSqliteSession::open("stale_usage_test", &path).unwrap();
    session
        .add_items(vec![user("u1"), assistant("a1")])
        .await
        .unwrap();
    session.pop_item().await.unwrap();
    session.pop_item().await.unwrap();

    session.store_run_usage(&usage_data()).await.unwrap();

    assert_eq!(count(&path, "turn_usage", "stale_usage_test"), 0);
}

#[tokio::test]
async fn a_reused_turn_number_does_not_inherit_usage_recorded_for_the_removed_turn() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "stale_usage_aba");
    let session = AdvancedSqliteSession::open("stale_usage_aba_test", &path).unwrap();
    session
        .add_items(vec![user("u1"), assistant("a1")])
        .await
        .unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();
    assert_eq!(count(&path, "turn_usage", "stale_usage_aba_test"), 1);

    // Remove turn 1 entirely, then let a brand-new turn reuse the numeric id.
    session.pop_item().await.unwrap();
    session.pop_item().await.unwrap();
    session.add_items(vec![user("fresh turn")]).await.unwrap();

    assert_eq!(count(&path, "turn_usage", "stale_usage_aba_test"), 0);
    assert!(
        session
            .get_turn_usage(Some(1), None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn store_run_usage_survives_unrelated_branch_deletion() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "usage_scope");
    let session = AdvancedSqliteSession::open("usage_scope_test", &path).unwrap();
    session
        .add_items(vec![
            user("u1"),
            assistant("a1"),
            user("u2"),
            assistant("a2"),
        ])
        .await
        .unwrap();
    session
        .create_branch_from_turn(2, Some("side_branch"))
        .await
        .unwrap();
    session.switch_to_branch("main").await.unwrap();

    session.delete_branch("side_branch", false).await.unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();

    assert_eq!(count(&path, "turn_usage", "usage_scope_test"), 1);
    let turn_2 = session.get_turn_usage(Some(2), None).await.unwrap();
    assert_eq!(turn_2.len(), 1);
    assert_eq!(turn_2[0].total_tokens, usage_data().total_tokens);
}

#[tokio::test]
async fn store_run_usage_skips_a_branch_without_a_turn() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "usage_no_turn");
    let session = AdvancedSqliteSession::open("usage_no_turn_test", &path).unwrap();

    // A fresh session has no turn on the current branch.
    session.store_run_usage(&usage_data()).await.unwrap();
    assert!(session.get_session_usage(None).await.unwrap().is_none());
    assert!(session.get_turn_usage(None, None).await.unwrap().is_empty());

    // A branch whose only turn was popped away has no turn either.
    session
        .add_items(vec![user("u1"), assistant("a1")])
        .await
        .unwrap();
    session.pop_item().await.unwrap();
    session.pop_item().await.unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();
    assert!(session.get_session_usage(None).await.unwrap().is_none());
    assert_eq!(count(&path, "turn_usage", "usage_no_turn_test"), 0);

    // Once a real turn exists, usage is recorded against it.
    session.add_items(vec![user("u2")]).await.unwrap();
    let second = Usage {
        requests: 2,
        input_tokens: 20,
        output_tokens: 5,
        total_tokens: 25,
        ..Default::default()
    };
    session.store_run_usage(&second).await.unwrap();
    let usage = session.get_session_usage(None).await.unwrap().unwrap();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.total_turns, 1);
    let turns = session.get_turn_usage(None, None).await.unwrap();
    assert_eq!(
        turns.iter().map(|t| t.user_turn_number).collect::<Vec<_>>(),
        vec![1]
    );
}

// ---------------------------------------------------------------------------
// Rollback after an earlier delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pop_item_rolls_back_when_a_later_delete_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "pop_rollback");
    let session = AdvancedSqliteSession::open("pop_rollback_test", &path).unwrap();
    session
        .add_items(vec![user("Question"), assistant("Answer")])
        .await
        .unwrap();
    let messages_before = count(&path, "agent_messages", "pop_rollback_test");
    let structure_before = count(&path, "message_structure", "pop_rollback_test");
    let branch_before = session.current_branch();

    // The message_structure delete runs first; deleting the orphaned message then fails.
    let conn = raw(&path);
    conn.execute_batch(
        "CREATE TRIGGER fail_message_delete BEFORE DELETE ON agent_messages
         BEGIN SELECT RAISE(ABORT, 'Simulated failure after earlier delete'); END;",
    )
    .unwrap();
    let error = session.pop_item().await.unwrap_err();
    assert!(error.to_string().contains("Simulated failure"), "{error}");

    assert_eq!(
        count(&path, "agent_messages", "pop_rollback_test"),
        messages_before
    );
    assert_eq!(
        count(&path, "message_structure", "pop_rollback_test"),
        structure_before
    );
    assert_eq!(session.current_branch(), branch_before);
    assert!(write_lock_is_free(&path));

    // The connection is left clean for a legitimate pop.
    conn.execute_batch("DROP TRIGGER fail_message_delete")
        .unwrap();
    let popped = session.pop_item().await.unwrap().unwrap();
    assert_eq!(popped, assistant("Answer"));
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Question"]
    );
}

#[tokio::test]
async fn clear_session_rolls_back_when_a_later_delete_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir, "clear_rollback");
    let session = AdvancedSqliteSession::open("clear_rollback_test", &path).unwrap();
    session
        .add_items(vec![user("First question"), assistant("First answer")])
        .await
        .unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();

    let messages_before = count(&path, "agent_messages", "clear_rollback_test");
    let structure_before = count(&path, "message_structure", "clear_rollback_test");
    let usage_before = count(&path, "turn_usage", "clear_rollback_test");
    assert!(structure_before > 0 && usage_before > 0);
    let branch_before = session.current_branch();

    let conn = raw(&path);
    conn.execute_batch(
        "CREATE TRIGGER fail_usage_delete BEFORE DELETE ON turn_usage
         BEGIN SELECT RAISE(ABORT, 'Simulated failure after earlier deletes'); END;",
    )
    .unwrap();
    let error = session.clear_session().await.unwrap_err();
    assert!(error.to_string().contains("Simulated failure"), "{error}");

    // The earlier deletes (messages, sessions, structure) were rolled back with it.
    assert_eq!(
        count(&path, "agent_messages", "clear_rollback_test"),
        messages_before
    );
    assert_eq!(
        count(&path, "message_structure", "clear_rollback_test"),
        structure_before
    );
    assert_eq!(
        count(&path, "turn_usage", "clear_rollback_test"),
        usage_before
    );
    assert!(write_lock_is_free(&path));
    assert_eq!(session.current_branch(), branch_before);

    // The connection is left clean for a legitimate clear.
    conn.execute_batch("DROP TRIGGER fail_usage_delete")
        .unwrap();
    session.clear_session().await.unwrap();
    assert_eq!(count(&path, "message_structure", "clear_rollback_test"), 0);
    assert_eq!(count(&path, "turn_usage", "clear_rollback_test"), 0);
    assert!(session.get_items(None).await.unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Quoted custom session table
// ---------------------------------------------------------------------------

/// `SqliteSession::with_tables` (`src/memory/sqlite_session.rs`,
/// `valid_identifier`) accepts ASCII letters, digits and underscores only, so the quoted
/// identifier Python's released `SQLiteSession` accepts is rejected before
/// `AdvancedSqliteSession` can use it. Minimal fix: accept SQL-quoted identifiers there.
#[tokio::test]
async fn structure_tables_accept_a_quoted_custom_session_table() {
    for create_tables in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "advanced_quoted_session_table");
        let (sessions, messages) = ("\"quoted_sessions\"", "quoted_messages");
        if !create_tables {
            AdvancedSqliteSession::open_with_tables("shared", &path, sessions, messages, true)
                .unwrap();
        }
        let session = AdvancedSqliteSession::open_with_tables(
            "shared",
            &path,
            sessions,
            messages,
            create_tables,
        )
        .unwrap();
        session.add_items(vec![user("quoted")]).await.unwrap();
        assert_eq!(session.get_items(None).await.unwrap(), vec![user("quoted")]);
    }
}
