//! Parity port of `tests/extensions/memory/test_advanced_sqlite_session.py`, second half: usage
//! tracking, tool-name extraction, pop/clear metadata hygiene, external clears and structure
//! table ownership.
#![cfg(feature = "sqlite")]

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use openai_agents_sdk::memory::{AdvancedSqliteSession, Session};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
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

fn raw(value: Value) -> InputItem {
    InputItem::from_value(value)
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

fn plain_usage(requests: u64, input: u64, output: u64, total: u64) -> Usage {
    Usage {
        requests,
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        ..Default::default()
    }
}

fn file_session(dir: &tempfile::TempDir, id: &str) -> (AdvancedSqliteSession, std::path::PathBuf) {
    let path = dir.path().join(format!("{id}.db"));
    (AdvancedSqliteSession::open(id, &path).unwrap(), path)
}

fn count(path: &Path, table: &str) -> i64 {
    Connection::open(path)
        .unwrap()
        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

fn custom(
    path: &Path,
    sessions: &str,
    messages: &str,
    create: bool,
) -> openai_agents_sdk::error::Result<AdvancedSqliteSession> {
    AdvancedSqliteSession::open_with_tables("shared", path, sessions, messages, create)
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_is_stored_per_turn_and_aggregated() {
    let session = AdvancedSqliteSession::in_memory("usage_test").unwrap();
    session.add_items(vec![user("First turn")]).await.unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();

    let second = Usage {
        requests: 2,
        input_tokens: 75,
        output_tokens: 45,
        total_tokens: 120,
        input_tokens_details: InputTokensDetails {
            cached_tokens: 20,
            cache_write_tokens: 0,
        },
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 15,
        },
        ..Default::default()
    };
    session.add_items(vec![user("Second turn")]).await.unwrap();
    session.store_run_usage(&second).await.unwrap();

    let total = session.get_session_usage(None).await.unwrap().unwrap();
    assert_eq!(
        (
            total.requests,
            total.total_tokens,
            total.input_tokens,
            total.output_tokens,
            total.total_turns
        ),
        (3, 200, 125, 75, 2)
    );

    let turn_1 = session.get_turn_usage(Some(1), None).await.unwrap();
    assert_eq!(turn_1.len(), 1);
    assert_eq!(turn_1[0].requests, 1);
    assert_eq!(turn_1[0].total_tokens, 80);
    assert_eq!(turn_1[0].input_tokens_details.cached_tokens, 10);
    assert_eq!(turn_1[0].output_tokens_details.reasoning_tokens, 5);
    let turn_2 = session.get_turn_usage(Some(2), None).await.unwrap();
    assert_eq!(turn_2[0].requests, 2);
    assert_eq!(turn_2[0].total_tokens, 120);
    assert_eq!(turn_2[0].input_tokens_details.cached_tokens, 20);
    assert_eq!(turn_2[0].output_tokens_details.reasoning_tokens, 15);

    let all = session.get_turn_usage(None, None).await.unwrap();
    assert_eq!(
        all.iter().map(|u| u.user_turn_number).collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[tokio::test]
async fn usage_is_tracked_per_branch() {
    let session = AdvancedSqliteSession::in_memory("branch_usage_test").unwrap();
    session
        .add_items(vec![user("Main question")])
        .await
        .unwrap();
    session
        .store_run_usage(&plain_usage(1, 50, 30, 80))
        .await
        .unwrap();

    session
        .create_branch_from_turn(1, Some("usage_branch"))
        .await
        .unwrap();
    session
        .add_items(vec![user("Branch question")])
        .await
        .unwrap();
    session
        .store_run_usage(&plain_usage(2, 100, 60, 160))
        .await
        .unwrap();

    let main = session
        .get_session_usage(Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (main.requests, main.total_tokens, main.total_turns),
        (1, 80, 1)
    );
    let branch = session
        .get_session_usage(Some("usage_branch"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (branch.requests, branch.total_tokens, branch.total_turns),
        (2, 160, 1)
    );
    let total = session.get_session_usage(None).await.unwrap().unwrap();
    assert_eq!(
        (total.requests, total.total_tokens, total.total_turns),
        (3, 240, 2)
    );

    let per_turn = session
        .get_turn_usage(None, Some("usage_branch"))
        .await
        .unwrap();
    assert_eq!(per_turn.len(), 1);
    assert_eq!(per_turn[0].requests, 2);
}

#[tokio::test]
async fn branch_specific_reads_do_not_cross_branches() {
    let session = AdvancedSqliteSession::in_memory("branch_specific_test").unwrap();
    session
        .add_items(vec![
            user("Main branch question"),
            assistant("Main branch answer"),
        ])
        .await
        .unwrap();
    session
        .store_run_usage(&plain_usage(1, 50, 30, 80))
        .await
        .unwrap();

    session
        .create_branch_from_turn(1, Some("test_branch"))
        .await
        .unwrap();
    session
        .add_items(vec![user("Branch question"), assistant("Branch answer")])
        .await
        .unwrap();
    session
        .store_run_usage(&plain_usage(1, 40, 20, 60))
        .await
        .unwrap();

    let main_items = session
        .get_items_for_branch(None, Some("main"))
        .await
        .unwrap();
    assert_eq!(main_items.len(), 2);
    assert_eq!(item_text(&main_items[0]), "Main branch question");
    assert_eq!(session.get_items(None).await.unwrap().len(), 2);

    let main_turns = session.get_conversation_turns(Some("main")).await.unwrap();
    assert_eq!(main_turns.len(), 1);
    assert_eq!(main_turns[0].content, "Main branch question");
    assert_eq!(session.get_conversation_turns(None).await.unwrap().len(), 1);

    assert_eq!(
        session
            .get_session_usage(Some("main"))
            .await
            .unwrap()
            .unwrap()
            .total_turns,
        1
    );
    assert_eq!(
        session
            .get_session_usage(None)
            .await
            .unwrap()
            .unwrap()
            .total_turns,
        2
    );
}

#[tokio::test]
async fn usage_details_default_to_zero_when_absent() {
    let session = AdvancedSqliteSession::in_memory("json_test").unwrap();
    session.add_items(vec![user("First test")]).await.unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();
    session.add_items(vec![user("Second test")]).await.unwrap();
    session
        .store_run_usage(&plain_usage(1, 10, 5, 15))
        .await
        .unwrap();

    let turn_1 = &session.get_turn_usage(Some(1), None).await.unwrap()[0];
    assert_eq!(
        (turn_1.requests, turn_1.input_tokens_details.cached_tokens),
        (1, 10)
    );
    let turn_2 = &session.get_turn_usage(Some(2), None).await.unwrap()[0];
    assert_eq!(turn_2.requests, 1);
    assert_eq!(turn_2.input_tokens_details.cached_tokens, 0);
    assert_eq!(turn_2.output_tokens_details.reasoning_tokens, 0);
}

#[tokio::test]
async fn output_token_details_persist_without_input_details() {
    let session = AdvancedSqliteSession::in_memory("output_only_usage").unwrap();
    let usage = Usage {
        requests: 1,
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 42,
        },
        ..Default::default()
    };
    session.add_items(vec![user("hi")]).await.unwrap();
    session.store_run_usage(&usage).await.unwrap();

    let turn = &session.get_turn_usage(Some(1), None).await.unwrap()[0];
    assert_eq!(turn.output_tokens_details.reasoning_tokens, 42);
}

#[tokio::test]
async fn a_failed_usage_write_leaves_the_database_unlocked() {
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "usage_rollback");
    session.add_items(vec![user("turn")]).await.unwrap();
    let helper = Connection::open(&path).unwrap();
    helper
        .execute_batch(
            "CREATE TRIGGER fail_turn_usage BEFORE INSERT ON turn_usage
             BEGIN SELECT RAISE(ABORT, 'usage write failed'); END;",
        )
        .unwrap();

    // The failure is logged, not returned, and must not strand a transaction.
    session.store_run_usage(&usage_data()).await.unwrap();

    let probe = Connection::open(&path).unwrap();
    probe.busy_handler(None).unwrap();
    probe
        .execute_batch("CREATE TABLE usage_lock_probe (x INTEGER)")
        .expect("the write lock must be free after the failed usage write");

    helper
        .execute_batch("DROP TRIGGER fail_turn_usage")
        .unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();
    assert!(
        !session
            .get_turn_usage(Some(1), None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn usage_with_no_turn_is_ignored() {
    let session = AdvancedSqliteSession::in_memory("error_test").unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();
    assert!(session.get_session_usage(None).await.unwrap().is_none());
}

#[tokio::test]
async fn the_runner_records_usage_per_turn() {
    openai_agents_sdk::set_tracing_disabled(true);
    let session = Arc::new(AdvancedSqliteSession::in_memory("integration_test").unwrap());
    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    for (question, answer) in [
        ("What city is the Golden Gate Bridge in?", "San Francisco"),
        ("What state is it in?", "California"),
    ] {
        model.add_turn(vec![assistant_message(answer)]);
        let result = Runner::run_with_options(
            &agent,
            question,
            RunOptions::new(()).session(session.clone()),
        )
        .await
        .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some(answer));
        session.store_run_usage(&result.usage()).await.unwrap();
    }

    assert_eq!(session.get_turn_structure(None).await.unwrap().len(), 2);
    assert_eq!(
        session
            .get_session_usage(None)
            .await
            .unwrap()
            .unwrap()
            .total_turns,
        2
    );
}

#[tokio::test]
async fn tool_executions_through_the_runner_are_tracked() {
    openai_agents_sdk::set_tracing_disabled(true);
    let session = Arc::new(AdvancedSqliteSession::in_memory("tool_integration_test").unwrap());
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call(
        "test_tool",
        r#"{"query": "test query"}"#,
    )]);
    model.add_turn(vec![assistant_message("Tool executed successfully")]);
    let tool = openai_agents_sdk::tool::FunctionTool::<()>::new_raw(
        "test_tool",
        "A test tool",
        json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        |_ctx, args: String| async move {
            let query: Value = serde_json::from_str(&args)?;
            Ok::<_, openai_agents_sdk::error::AgentsError>(format!(
                "Tool result for: {}",
                query["query"].as_str().unwrap_or_default()
            ))
        },
    );
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool);

    Runner::run_with_options(
        &agent,
        "Please use the test tool",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();

    assert!(!session.get_tool_usage(None).await.unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Tool names
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_names_follow_each_call_type() {
    let session = AdvancedSqliteSession::in_memory("advanced_tool_names_test").unwrap();
    session
        .add_items(vec![
            user("Use various tools"),
            raw(json!({"type": "mcp_call", "server_label": "filesystem", "name": "read_file", "arguments": "{}"})),
            raw(json!({"type": "mcp_approval_request", "server_label": "database", "name": "execute_query", "arguments": "{}"})),
            raw(json!({"type": "computer_call", "arguments": "{}"})),
            raw(json!({"type": "file_search_call", "arguments": "{}"})),
            raw(json!({"type": "web_search_call", "arguments": "{}"})),
            raw(json!({"type": "code_interpreter_call", "arguments": "{}"})),
            raw(json!({"type": "function_call", "name": "calculator", "arguments": "{}"})),
            raw(json!({"type": "custom_tool_call", "name": "custom_tool", "arguments": "{}"})),
        ])
        .await
        .unwrap();

    let structure = session.get_turn_structure(None).await.unwrap();
    let names: BTreeSet<String> = structure[&1]
        .iter()
        .filter_map(|m| m.tool_name.clone())
        .collect();
    for expected in [
        "filesystem.read_file",
        "database.execute_query",
        "computer_call",
        "file_search_call",
        "web_search_call",
        "code_interpreter_call",
        "calculator",
        "custom_tool",
    ] {
        assert!(
            names.contains(expected),
            "{expected} missing from {names:?}"
        );
    }
}

#[tokio::test]
async fn named_function_calls_are_extracted() {
    let session = AdvancedSqliteSession::in_memory("tool_names_test").unwrap();
    session
        .add_items(vec![
            user("Use tools please"),
            raw(json!({"type": "function_call", "name": "search_web", "arguments": "{}"})),
            raw(json!({"type": "function_call_output", "tool_name": "search_web", "output": "result"})),
            raw(json!({"type": "function_call", "name": "calculator", "arguments": "{}"})),
        ])
        .await
        .unwrap();

    let structure = session.get_turn_structure(None).await.unwrap();
    let names: BTreeSet<String> = structure[&1]
        .iter()
        .filter_map(|m| m.tool_name.clone())
        .collect();
    assert!(names.contains("search_web") && names.contains("calculator"));
}

// ---------------------------------------------------------------------------
// Clear and pop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn clear_session_removes_structure_and_usage_and_resets_numbering() {
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "clear_metadata_test");
    session
        .add_items(vec![user("First question"), assistant("First answer")])
        .await
        .unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();
    assert!(count(&path, "message_structure") > 0);
    assert!(count(&path, "turn_usage") > 0);

    session.clear_session().await.unwrap();

    assert!(session.get_items(None).await.unwrap().is_empty());
    assert_eq!(count(&path, "message_structure"), 0);
    assert_eq!(count(&path, "turn_usage"), 0);

    session.add_items(vec![user("Fresh start")]).await.unwrap();
    let conn = Connection::open(&path).unwrap();
    let rows: Vec<(i64, i64)> = conn
        .prepare("SELECT sequence_number, user_turn_number FROM message_structure")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(rows, vec![(1, 1)]);
}

#[tokio::test]
async fn pop_item_removes_its_structure_row() {
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "pop_structure_test");
    session
        .add_items(vec![user("Question"), assistant("Answer")])
        .await
        .unwrap();

    let popped = session.pop_item().await.unwrap().unwrap();
    assert_eq!(popped, assistant("Answer"));

    assert_eq!(count(&path, "message_structure"), 1);
    assert_eq!(count(&path, "agent_messages"), 1);
    assert_eq!(
        session.get_items(None).await.unwrap(),
        vec![user("Question")]
    );
}

#[tokio::test]
async fn pop_item_drops_turn_usage_only_when_the_turn_empties() {
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "pop_turn_usage_test");
    session
        .add_items(vec![user("Question"), assistant("Answer")])
        .await
        .unwrap();
    session.store_run_usage(&usage_data()).await.unwrap();
    assert_eq!(count(&path, "turn_usage"), 1);

    session.pop_item().await.unwrap();
    assert_eq!(count(&path, "turn_usage"), 1);

    session.pop_item().await.unwrap();
    assert_eq!(count(&path, "turn_usage"), 0);
    assert!(
        session
            .get_turn_usage(Some(1), None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn pop_item_works_on_the_current_branch_and_keeps_shared_messages() {
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "pop_branch_test");
    let main_items = vec![
        user("Main first question"),
        assistant("Main first answer"),
        user("Main second question"),
        assistant("Main second answer"),
    ];
    session.add_items(main_items.clone()).await.unwrap();
    session
        .create_branch_from_turn(2, Some("branch_a"))
        .await
        .unwrap();
    session.switch_to_branch("branch_a").await.unwrap();
    session
        .add_items(vec![user("Branch-only question")])
        .await
        .unwrap();

    let popped = session.pop_item().await.unwrap().unwrap();
    assert_eq!(popped, user("Branch-only question"));
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap(),
        main_items
    );

    let conn = Connection::open(&path).unwrap();
    let orphans: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM message_structure
             WHERE message_id NOT IN (SELECT id FROM agent_messages)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(orphans, 0);
}

#[tokio::test]
async fn a_shared_copied_message_is_deleted_only_when_unreferenced() {
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "pop_shared_copy_test");
    let main_items = vec![user("u1"), assistant("a1"), user("u2"), assistant("a2")];
    session.add_items(main_items.clone()).await.unwrap();
    session
        .create_branch_from_turn(2, Some("branch_a"))
        .await
        .unwrap();
    session.switch_to_branch("branch_a").await.unwrap();
    session.add_items(vec![user("branch-only")]).await.unwrap();
    assert_eq!(count(&path, "agent_messages"), 5);

    assert_eq!(session.pop_item().await.unwrap(), Some(user("branch-only")));
    assert_eq!(count(&path, "agent_messages"), 4);

    assert_eq!(session.pop_item().await.unwrap(), Some(assistant("a1")));
    assert_eq!(session.pop_item().await.unwrap(), Some(user("u1")));
    assert_eq!(count(&path, "agent_messages"), 4);
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap(),
        main_items
    );
    assert!(
        session
            .get_items_for_branch(None, Some("branch_a"))
            .await
            .unwrap()
            .is_empty()
    );

    session.switch_to_branch("main").await.unwrap();
    for _ in 0..main_items.len() {
        session.pop_item().await.unwrap();
    }
    assert_eq!(count(&path, "agent_messages"), 0);
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert_eq!(count(&path, "message_structure"), 0);
}

#[tokio::test]
async fn clear_session_resets_the_current_branch_to_main_but_keeps_reservations() {
    let dir = tempfile::tempdir().unwrap();
    let (session, path) = file_session(&dir, "clear_branch_reset_test");
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
        .create_branch_from_turn(2, Some("branch_a"))
        .await
        .unwrap();
    session.switch_to_branch("branch_a").await.unwrap();
    assert_eq!(session.current_branch(), "branch_a");
    assert_eq!(count(&path, "branch_reservations"), 2);

    session.clear_session().await.unwrap();

    assert_eq!(session.current_branch(), "main");
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert_eq!(count(&path, "branch_reservations"), 2);
}

// ---------------------------------------------------------------------------
// External clears
// ---------------------------------------------------------------------------

async fn stale_and_clearer(
    dir: &tempfile::TempDir,
    id: &str,
) -> (AdvancedSqliteSession, AdvancedSqliteSession) {
    let path = dir.path().join(format!("{id}.db"));
    let stale = AdvancedSqliteSession::open(id, &path).unwrap();
    let clearer = AdvancedSqliteSession::open(id, &path).unwrap();
    stale
        .add_items(vec![user("u1"), assistant("a1"), user("u2")])
        .await
        .unwrap();
    stale
        .create_branch_from_turn(2, Some("stale"))
        .await
        .unwrap();
    assert_eq!(stale.current_branch(), "stale");
    (stale, clearer)
}

#[tokio::test]
async fn an_external_clear_resets_a_stale_branch_before_the_next_write() {
    let dir = tempfile::tempdir().unwrap();
    let (stale, clearer) = stale_and_clearer(&dir, "external_clear_generation").await;

    clearer.clear_session().await.unwrap();
    stale.add_items(vec![user("after clear")]).await.unwrap();

    assert_eq!(stale.current_branch(), "main");
    assert_eq!(
        stale
            .get_items(None)
            .await
            .unwrap()
            .iter()
            .map(item_text)
            .collect::<Vec<_>>(),
        vec!["after clear"]
    );
    assert!(
        stale
            .get_items_for_branch(None, Some("stale"))
            .await
            .unwrap()
            .is_empty()
    );
    let branches: BTreeSet<String> = stale
        .list_branches()
        .await
        .unwrap()
        .into_iter()
        .map(|b| b.branch_id)
        .collect();
    assert_eq!(branches, ["main".to_string()].into_iter().collect());
}

#[tokio::test]
async fn an_external_clear_resets_a_stale_branch_before_a_pop() {
    let dir = tempfile::tempdir().unwrap();
    let (stale, clearer) = stale_and_clearer(&dir, "external_clear_pop_generation").await;

    clearer.clear_session().await.unwrap();
    let item = user("after clear");
    clearer.add_items(vec![item.clone()]).await.unwrap();

    assert_eq!(stale.pop_item().await.unwrap(), Some(item));
    assert_eq!(stale.current_branch(), "main");
    assert!(clearer.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn an_external_clear_resets_a_stale_branch_before_default_reads() {
    for read_path in [
        "items",
        "turns",
        "search",
        "conversation",
        "tools",
        "usage",
        "branches",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (stale, clearer) =
            stale_and_clearer(&dir, &format!("external_clear_read_{read_path}")).await;

        clearer.clear_session().await.unwrap();
        let new_items = vec![
            user("new main question"),
            raw(json!({"type": "function_call", "name": "lookup",
                       "arguments": "{\"query\": \"new\"}", "call_id": "lookup-new-main"})),
            assistant("new main answer"),
        ];
        clearer.add_items(new_items.clone()).await.unwrap();
        clearer.store_run_usage(&usage_data()).await.unwrap();

        match read_path {
            "items" => assert_eq!(stale.get_items(None).await.unwrap(), new_items),
            "turns" => assert_eq!(
                stale
                    .get_conversation_turns(None)
                    .await
                    .unwrap()
                    .iter()
                    .map(|t| t.full_content.clone())
                    .collect::<Vec<_>>(),
                vec!["new main question"]
            ),
            "search" => assert_eq!(
                stale
                    .find_turns_by_content("new", None)
                    .await
                    .unwrap()
                    .iter()
                    .map(|t| t.full_content.clone())
                    .collect::<Vec<_>>(),
                vec!["new main question"]
            ),
            "conversation" => assert_eq!(
                stale
                    .get_turn_structure(None)
                    .await
                    .unwrap()
                    .keys()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![1]
            ),
            "tools" => {
                let usage = stale.get_tool_usage(None).await.unwrap();
                assert_eq!(usage.len(), 1);
                assert_eq!(
                    (
                        usage[0].tool_name.as_str(),
                        usage[0].count,
                        usage[0].user_turn_number
                    ),
                    ("lookup", 1, 1)
                );
            }
            "usage" => {
                let turn = &stale.get_turn_usage(Some(1), None).await.unwrap()[0];
                assert_eq!(
                    (
                        turn.requests,
                        turn.input_tokens,
                        turn.output_tokens,
                        turn.total_tokens
                    ),
                    (1, 50, 30, 80)
                );
                assert_eq!(turn.input_tokens_details.cached_tokens, 10);
                assert_eq!(turn.output_tokens_details.reasoning_tokens, 5);
            }
            _ => {
                let branches: Vec<(String, bool)> = stale
                    .list_branches()
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|b| (b.branch_id, b.is_current))
                    .collect();
                assert_eq!(branches, vec![("main".to_string(), true)]);
            }
        }
        assert_eq!(stale.current_branch(), "main", "{read_path}");
    }
}

// ---------------------------------------------------------------------------
// Structure table ownership
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_second_base_table_pair_is_rejected_without_creating_tables() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared_structure.db");
    let first = custom(&path, "first_sessions", "first_messages", true).unwrap();
    first.add_items(vec![user("first")]).await.unwrap();

    let err = custom(&path, "second_sessions", "second_messages", true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("first_sessions"), "{err}");

    let conn = Connection::open(&path).unwrap();
    let leaked: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name IN
             ('second_sessions', 'second_messages', 'idx_second_messages_session_id')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(leaked, 0);
    assert_eq!(first.get_items(None).await.unwrap(), vec![user("first")]);
}

#[tokio::test]
async fn a_changed_sessions_table_over_shared_messages_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared_messages.db");
    let first = custom(&path, "first_sessions", "shared_messages", true).unwrap();
    first.add_items(vec![user("first")]).await.unwrap();

    let err = custom(&path, "second_sessions", "shared_messages", true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("first_sessions"), "{err}");
    let conn = Connection::open(&path).unwrap();
    let created: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'second_sessions'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(created, 0);
}

#[tokio::test]
async fn a_changed_messages_table_over_shared_sessions_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared_sessions.db");
    let first = custom(&path, "shared_sessions", "first_messages", true).unwrap();
    first.add_items(vec![user("first")]).await.unwrap();

    let err = custom(&path, "shared_sessions", "second_messages", true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("first_messages"), "{err}");
    let conn = Connection::open(&path).unwrap();
    let created: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'second_messages'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(created, 0);
}

#[tokio::test]
async fn a_recased_table_pair_is_the_same_pair() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recased.db");
    {
        let first = custom(&path, "FooSessions", "FooMessages", true).unwrap();
        first.add_items(vec![user("first")]).await.unwrap();
    }
    let recased = custom(&path, "foosessions", "foomessages", true).unwrap();
    assert_eq!(recased.get_items(None).await.unwrap(), vec![user("first")]);
}

#[tokio::test]
async fn distinct_non_ascii_identifiers_are_distinct_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("non_ascii.db");
    let first = custom(&path, "ßsessions", "ßmessages", true).unwrap();
    first.add_items(vec![user("first")]).await.unwrap();
    let err = custom(&path, "sssessions", "ssmessages", true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("ßsessions"), "{err}");
}

#[tokio::test]
async fn a_no_create_session_rejects_a_database_without_an_owner() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unclaimed.db");

    let err = AdvancedSqliteSession::open_with("shared", &path, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("create_tables=True"), "{err}");
    let tables: i64 = Connection::open(&path)
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(tables, 0);

    let owner = AdvancedSqliteSession::open("shared", &path).unwrap();
    owner.add_items(vec![user("first")]).await.unwrap();
    let reader = AdvancedSqliteSession::open_with("shared", &path, false).unwrap();
    assert_eq!(reader.get_items(None).await.unwrap(), vec![user("first")]);
}

fn owner_bearing_tables(
    path: &Path,
    base_tables: bool,
    message_fks: &str,
    usage_fk: &str,
    session_col: &str,
    id_col: &str,
) {
    let conn = Connection::open(path).unwrap();
    if base_tables {
        conn.execute_batch(
            "CREATE TABLE agent_sessions (session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
             CREATE TABLE agent_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                message_data TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES agent_sessions (session_id) ON DELETE CASCADE);",
        )
        .unwrap();
    }
    let message_constraints = if message_fks.is_empty() {
        String::new()
    } else {
        format!(", {message_fks}")
    };
    let usage_constraint = if usage_fk.is_empty() {
        String::new()
    } else {
        format!(", {usage_fk}")
    };
    conn.execute_batch(&format!(
        "CREATE TABLE wrong_sessions (session_id TEXT PRIMARY KEY);
         CREATE TABLE message_structure (id INTEGER PRIMARY KEY AUTOINCREMENT,
            {session_col} TEXT NOT NULL, {id_col} INTEGER NOT NULL, branch_id TEXT NOT NULL DEFAULT 'main',
            message_type TEXT NOT NULL, sequence_number INTEGER NOT NULL, user_turn_number INTEGER,
            branch_turn_number INTEGER, tool_name TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            {message_constraints});
         CREATE TABLE turn_usage (id INTEGER PRIMARY KEY AUTOINCREMENT, {session_col} TEXT NOT NULL,
            branch_id TEXT NOT NULL DEFAULT 'main', user_turn_number INTEGER NOT NULL,
            requests INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0, input_tokens_details JSON, output_tokens_details JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(session_id, branch_id, user_turn_number)
            {usage_constraint});"
    ))
    .unwrap();
}

#[tokio::test]
async fn a_no_create_session_rejects_owner_metadata_without_base_tables() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing_base.db");
    owner_bearing_tables(
        &path,
        false,
        "FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id), FOREIGN KEY (message_id) REFERENCES agent_messages(id)",
        "FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id)",
        "session_id",
        "message_id",
    );

    let err = AdvancedSqliteSession::open_with("shared", &path, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("configured base tables"), "{err}");
}

#[tokio::test]
async fn an_ownerless_structure_layout_is_rejected() {
    for create_tables in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ownerless.db");
        owner_bearing_tables(&path, true, "", "", "session_id", "message_id");
        let err = AdvancedSqliteSession::open_with("shared", &path, create_tables)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exactly one owner foreign key"), "{err}");
    }
}

#[tokio::test]
async fn malformed_owner_layouts_are_rejected() {
    let cases = [
        (
            "FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id), FOREIGN KEY (session_id) REFERENCES wrong_sessions(session_id), FOREIGN KEY (message_id) REFERENCES agent_messages(id)",
            "FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id)",
            "exactly one owner foreign key",
        ),
        (
            "FOREIGN KEY (session_id) REFERENCES agent_sessions(wrong_id), FOREIGN KEY (message_id) REFERENCES agent_messages(id)",
            "FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id)",
            "already belongs",
        ),
        (
            "FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id), FOREIGN KEY (message_id) REFERENCES agent_messages(id)",
            "FOREIGN KEY (session_id) REFERENCES wrong_sessions(session_id)",
            "already belongs",
        ),
    ];
    for create_tables in [false, true] {
        for (message_fks, usage_fk, expected) in cases {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("malformed.db");
            owner_bearing_tables(
                &path,
                true,
                message_fks,
                usage_fk,
                "session_id",
                "message_id",
            );
            let err = AdvancedSqliteSession::open_with("shared", &path, create_tables)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(expected),
                "{err:?} should contain {expected:?}"
            );
        }
    }
}

#[tokio::test]
async fn child_column_casing_does_not_matter() {
    for create_tables in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recased_cols.db");
        owner_bearing_tables(
            &path,
            true,
            "FOREIGN KEY (SESSION_ID) REFERENCES agent_sessions(session_id), FOREIGN KEY (MESSAGE_ID) REFERENCES agent_messages(id)",
            "FOREIGN KEY (SESSION_ID) REFERENCES agent_sessions(session_id)",
            "SESSION_ID",
            "MESSAGE_ID",
        );
        AdvancedSqliteSession::open_with("shared", &path, create_tables)
            .unwrap_or_else(|e| panic!("create_tables={create_tables}: {e}"));
    }
}
