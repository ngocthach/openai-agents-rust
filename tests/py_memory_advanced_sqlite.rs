//! Parity port of `tests/extensions/memory/test_advanced_sqlite_session.py`: structure and tool
//! tracking, branching, turn browsing and content search. Usage, pop/clear and table-ownership
//! tests live in `py_memory_advanced_sqlite_usage.rs`.
#![cfg(feature = "sqlite")]

use std::collections::BTreeSet;

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{AdvancedSqliteSession, MAIN_BRANCH, Session, SessionSettings};
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

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn new_session(id: &str) -> AdvancedSqliteSession {
    AdvancedSqliteSession::in_memory(id).unwrap()
}

fn collision_items() -> Vec<InputItem> {
    vec![
        user("Turn one question"),
        assistant("Turn one answer"),
        user("Turn two question"),
        assistant("Turn two answer"),
        user("Turn three question"),
        assistant("Turn three answer"),
    ]
}

fn usage_by_tool(
    usage: &[openai_agents_sdk::memory::ToolUsage],
) -> std::collections::BTreeMap<String, u64> {
    usage
        .iter()
        .map(|u| (u.tool_name.clone(), u.count))
        .collect()
}

// ---------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn basic_functionality() {
    let session = new_session("advanced_test");
    session
        .add_items(vec![user("Hello"), assistant("Hi there!")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Hello", "Hi there!"]
    );
}

#[tokio::test]
async fn custom_table_names_are_used_consistently() {
    let base = openai_agents_sdk::memory::SqliteSession::with_tables(
        "advanced_custom_tables",
        tempfile::tempdir().unwrap().keep().join("c.db"),
        "custom_agent_sessions",
        "custom_agent_messages",
    )
    .unwrap();
    let session = AdvancedSqliteSession::from_session(base).unwrap();
    let items = vec![
        user("Hello"),
        assistant("Hi there!"),
        user("Let's do some math"),
        assistant("Sure"),
    ];
    session.add_items(items.clone()).await.unwrap();

    assert_eq!(session.get_items(None).await.unwrap(), items);
    let turns = session.get_conversation_turns(None).await.unwrap();
    assert_eq!(turns.iter().map(|t| t.turn).collect::<Vec<_>>(), vec![1, 2]);
    let matches = session.find_turns_by_content("math", None).await.unwrap();
    assert_eq!(matches.iter().map(|t| t.turn).collect::<Vec<_>>(), vec![2]);

    let name = session
        .create_branch_from_turn(2, Some("custom_branch"))
        .await
        .unwrap();
    assert_eq!(name, "custom_branch");
    assert_eq!(session.get_items(None).await.unwrap(), items[..2].to_vec());
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap(),
        items
    );
}

#[tokio::test]
async fn a_failed_batch_does_not_hide_or_roll_back_earlier_history() {
    // Python injects a failing structure-metadata writer; the observable contract is that a
    // batch that fails leaves earlier committed history intact and can be retried exactly once.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fail.db");
    let session =
        AdvancedSqliteSession::open("advanced_add_items_existing_history", &path).unwrap();
    session
        .add_items(vec![user("already saved")])
        .await
        .unwrap();

    // Make the next structure write fail after the message row is inserted.
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER fail_structure BEFORE INSERT ON message_structure
         BEGIN SELECT RAISE(ABORT, 'structure metadata failed'); END;",
    )
    .unwrap();
    let failed = session.add_items(vec![assistant("not saved")]).await;
    assert!(
        failed
            .unwrap_err()
            .to_string()
            .contains("structure metadata failed")
    );
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["already saved"]
    );

    let count = |sql: &str| -> i64 { conn.query_row(sql, [], |r| r.get(0)).unwrap() };
    assert_eq!(count("SELECT COUNT(*) FROM agent_messages"), 1);
    assert_eq!(count("SELECT COUNT(*) FROM message_structure"), 1);

    // After the fault clears, retrying persists the batch exactly once.
    conn.execute_batch("DROP TRIGGER fail_structure").unwrap();
    session
        .add_items(vec![assistant("saved once")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["already saved", "saved once"]
    );
    assert_eq!(count("SELECT COUNT(*) FROM agent_messages"), 2);
    assert_eq!(count("SELECT COUNT(*) FROM message_structure"), 2);
}

// ---------------------------------------------------------------------------
// Structure and tool tracking
// ---------------------------------------------------------------------------

#[tokio::test]
async fn message_structure_is_tracked_by_turn() {
    let session = new_session("structure_test");
    session
        .add_items(vec![
            user("What's 2+2?"),
            raw(json!({"type": "function_call", "name": "calculator", "arguments": "{\"expression\": \"2+2\"}"})),
            raw(json!({"type": "function_call_output", "output": "4"})),
            assistant("The answer is 4"),
            raw(json!({"type": "reasoning", "summary": [{"text": "Simple math", "type": "summary_text"}]})),
        ])
        .await
        .unwrap();

    let structure = session.get_turn_structure(None).await.unwrap();
    assert_eq!(structure.len(), 1);
    let types: Vec<&str> = structure[&1]
        .iter()
        .map(|m| m.message_type.as_str())
        .collect();
    assert_eq!(
        types,
        vec![
            "user",
            "function_call",
            "function_call_output",
            "assistant",
            "reasoning"
        ]
    );
}

#[tokio::test]
async fn conversation_structure_spans_multiple_turns() {
    let session = new_session("multi_turn_test");
    session
        .add_items(vec![user("Hello"), assistant("Hi!")])
        .await
        .unwrap();
    session
        .add_items(vec![
            user("How are you?"),
            raw(json!({"type": "function_call", "name": "mood_check", "arguments": "{}"})),
            raw(json!({"type": "function_call_output", "output": "I'm good"})),
            assistant("I'm doing well!"),
        ])
        .await
        .unwrap();
    session
        .add_items(vec![user("Goodbye"), assistant("See you later!")])
        .await
        .unwrap();

    let structure = session.get_turn_structure(None).await.unwrap();
    assert_eq!(structure.len(), 3);
    assert_eq!(structure[&1].len(), 2);
    assert_eq!(structure[&1][0].message_type, "user");
    assert_eq!(structure[&1][1].message_type, "assistant");
    assert_eq!(structure[&2].len(), 4);
    assert_eq!(structure[&3].len(), 2);
}

#[tokio::test]
async fn tool_usage_counts_each_named_tool() {
    let session = new_session("tools_test");
    session
        .add_items(vec![
            user("Search for cats"),
            raw(json!({"type": "function_call", "name": "web_search", "arguments": "{\"query\": \"cats\"}"})),
            raw(json!({"type": "function_call_output", "output": "Found cat information"})),
            raw(json!({"type": "function_call", "name": "calculator", "arguments": "{\"expression\": \"1+1\"}"})),
            raw(json!({"type": "function_call_output", "output": "2"})),
            assistant("I found information about cats and calculated 1+1=2"),
        ])
        .await
        .unwrap();

    let usage = session.get_tool_usage(None).await.unwrap();
    assert_eq!(usage.len(), 2);
    let names: BTreeSet<&str> = usage.iter().map(|u| u.tool_name.as_str()).collect();
    assert!(names.contains("web_search") && names.contains("calculator"));
}

fn shell_call(id: &str) -> Value {
    json!({
        "type": "shell_call",
        "call_id": id,
        "status": "completed",
        "action": {"commands": ["echo test"], "timeout_ms": null, "max_output_length": null},
    })
}

fn apply_patch_call(id: &str) -> Value {
    json!({
        "type": "apply_patch_call",
        "call_id": id,
        "status": "completed",
        "operation": {"type": "update_file", "path": "test.md", "diff": "-a\n+b\n"},
    })
}

#[tokio::test]
async fn tool_usage_tracks_shell_and_patch_calls_not_their_outputs() {
    let cases = [
        (
            shell_call("shell-1"),
            json!({"type": "shell_call_output", "call_id": "shell-1",
                   "output": [{"stdout": "test", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]}),
            "shell_call",
        ),
        (
            apply_patch_call("patch-1"),
            json!({"type": "apply_patch_call_output", "call_id": "patch-1",
                   "status": "completed", "output": "Updated test.md"}),
            "apply_patch_call",
        ),
    ];
    for (call, output, tool_name) in cases {
        let session = new_session("builtin-tools");
        let items = vec![user("Update the file."), raw(call), raw(output)];
        session.add_items(items.clone()).await.unwrap();

        assert_eq!(session.get_items(None).await.unwrap(), items);
        let usage = session.get_tool_usage(None).await.unwrap();
        assert_eq!(usage.len(), 1, "{usage:?}");
        assert_eq!(
            (
                usage[0].tool_name.as_str(),
                usage[0].count,
                usage[0].user_turn_number
            ),
            (tool_name, 1, 1)
        );
        let structure = session.get_turn_structure(None).await.unwrap();
        assert_eq!(structure[&1][1].tool_name.as_deref(), Some(tool_name));
    }
}

#[tokio::test]
async fn tool_usage_reads_legacy_null_names_without_rewriting_them() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy-tools.db");
    {
        let session = AdvancedSqliteSession::open("legacy-tools", &path).unwrap();
        session
            .add_items(vec![
                user("Update the file."),
                raw(shell_call("shell-1")),
                raw(apply_patch_call("patch-1")),
            ])
            .await
            .unwrap();
    }
    // Earlier SDK versions stored these calls with NULL tool names.
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "UPDATE message_structure SET tool_name = NULL
         WHERE message_type IN ('shell_call', 'apply_patch_call')",
    )
    .unwrap();

    let reopened = AdvancedSqliteSession::open("legacy-tools", &path).unwrap();
    let usage = reopened.get_tool_usage(None).await.unwrap();
    let counts = usage_by_tool(&usage);
    assert_eq!(counts.get("shell_call"), Some(&1));
    assert_eq!(counts.get("apply_patch_call"), Some(&1));
    let structure = reopened.get_turn_structure(None).await.unwrap();
    assert_eq!(
        structure[&1][1..]
            .iter()
            .map(|m| m.tool_name.clone())
            .collect::<Vec<_>>(),
        vec![None, None]
    );

    reopened
        .add_items(vec![
            raw(shell_call("shell-2")),
            raw(apply_patch_call("patch-2")),
        ])
        .await
        .unwrap();
    let counts = usage_by_tool(&reopened.get_tool_usage(None).await.unwrap());
    assert_eq!(counts.get("shell_call"), Some(&2));
    assert_eq!(counts.get("apply_patch_call"), Some(&2));
}

#[tokio::test]
async fn tool_usage_keeps_namespaces_and_counts_tool_search_once() {
    let session = new_session("tools_namespace_test");
    session
        .add_items(vec![
            user("Look up the same account in multiple systems"),
            raw(json!({"type": "function_call", "name": "lookup_account", "namespace": "crm",
                       "arguments": "{\"account_id\": \"acct_123\"}", "call_id": "crm-call"})),
            raw(json!({"type": "function_call", "name": "lookup_account", "namespace": "billing",
                       "arguments": "{\"account_id\": \"acct_123\"}", "call_id": "billing-call"})),
            raw(json!({"type": "tool_search_call", "id": "tsc_memory",
                       "arguments": {"paths": ["crm"], "query": "lookup_account"},
                       "execution": "server", "status": "completed"})),
            raw(json!({"type": "tool_search_output", "id": "tso_memory", "execution": "server",
                       "status": "completed", "tools": [{"type": "function", "name": "lookup_account",
                       "description": "Look up an account.", "parameters": {"type": "object"},
                       "defer_loading": true}]})),
        ])
        .await
        .unwrap();

    let counts = usage_by_tool(&session.get_tool_usage(None).await.unwrap());
    assert_eq!(counts["crm.lookup_account"], 1);
    assert_eq!(counts["billing.lookup_account"], 1);
    assert_eq!(counts["tool_search"], 1);
}

#[tokio::test]
async fn a_tool_search_output_without_a_call_still_counts_once() {
    let session = new_session("tools_tool_search_output_only_test");
    session
        .add_items(vec![
            user("Look up customer_42"),
            raw(json!({"type": "tool_search_output", "id": "tso_memory_only", "execution": "server",
                       "status": "completed", "tools": [{"type": "function", "name": "lookup_account",
                       "description": "Look up an account.", "parameters": {"type": "object"}}]})),
        ])
        .await
        .unwrap();

    let counts = usage_by_tool(&session.get_tool_usage(None).await.unwrap());
    assert_eq!(counts["tool_search"], 1);
}

#[tokio::test]
async fn deferred_top_level_calls_use_the_bare_tool_name() {
    let session = new_session("tools_deferred_top_level_test");
    session
        .add_items(vec![
            user("What is the weather?"),
            raw(json!({"type": "function_call", "name": "get_weather",
                       "arguments": "{\"city\": \"Tokyo\"}", "call_id": "weather-call"})),
            raw(
                json!({"type": "function_call", "name": "get_weather", "namespace": "get_weather",
                       "arguments": "{\"city\": \"Osaka\"}", "call_id": "weather-call-2"}),
            ),
        ])
        .await
        .unwrap();

    let counts = usage_by_tool(&session.get_tool_usage(None).await.unwrap());
    assert_eq!(counts["get_weather"], 2);
    assert!(!counts.contains_key("get_weather.get_weather"));
}

#[tokio::test]
async fn a_same_name_namespace_collapses_to_the_bare_name() {
    let session = new_session("tools_deferred_top_level_namespace_test");
    session
        .add_items(vec![
            user("What is the weather?"),
            raw(json!({"type": "function_call", "name": "lookup_account", "namespace": "lookup_account",
                       "arguments": "{\"account_id\": \"acct_123\"}", "call_id": "lookup-call"})),
        ])
        .await
        .unwrap();

    let counts = usage_by_tool(&session.get_tool_usage(None).await.unwrap());
    assert_eq!(counts["lookup_account"], 1);
    assert!(!counts.contains_key("lookup_account.lookup_account"));
}

// ---------------------------------------------------------------------------
// Branching
// ---------------------------------------------------------------------------

#[tokio::test]
async fn branching_creates_switches_lists_and_deletes() {
    let session = new_session("branching_test");
    for turn in ["First", "Second", "Third"] {
        session
            .add_items(vec![
                user(&format!("{turn} question")),
                assistant(&format!("{turn} answer")),
            ])
            .await
            .unwrap();
    }
    assert_eq!(session.get_items(None).await.unwrap().len(), 6);

    let name = session
        .create_branch_from_turn(2, Some("test_branch"))
        .await
        .unwrap();
    assert_eq!(name, "test_branch");
    assert_eq!(session.current_branch(), "test_branch");
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["First question", "First answer"]
    );

    session.switch_to_branch("main").await.unwrap();
    assert_eq!(session.current_branch(), "main");
    assert_eq!(session.get_items(None).await.unwrap().len(), 6);

    let ids: BTreeSet<String> = session
        .list_branches()
        .await
        .unwrap()
        .into_iter()
        .map(|b| b.branch_id)
        .collect();
    assert_eq!(
        ids,
        ["main".to_string(), "test_branch".to_string()]
            .into_iter()
            .collect()
    );

    session.delete_branch("test_branch", false).await.unwrap();
    let after = session.list_branches().await.unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].branch_id, "main");
}

#[tokio::test]
async fn blank_branch_names_are_rejected() {
    for blank in ["", "   ", "\t"] {
        for from_content in [false, true] {
            let session = new_session("blank_branch_name");
            let items = vec![
                user("First question"),
                assistant("First answer"),
                user("Second question"),
            ];
            session.add_items(items.clone()).await.unwrap();
            let before = session.list_branches().await.unwrap();

            let err = if from_content {
                session
                    .create_branch_from_content("Second question", Some(blank))
                    .await
            } else {
                session.create_branch_from_turn(2, Some(blank)).await
            }
            .unwrap_err()
            .to_string();

            assert!(err.contains("Branch name cannot be empty"), "{err}");
            assert_eq!(session.list_branches().await.unwrap(), before);
            assert_eq!(session.current_branch(), "main");
            assert_eq!(session.get_items(None).await.unwrap(), items);
        }
    }
}

#[tokio::test]
async fn a_populated_branch_id_cannot_be_reused() {
    for branch_id in ["main", "existing_branch"] {
        let session = new_session("branch_collision");
        let items = collision_items();
        session.add_items(items.clone()).await.unwrap();
        if branch_id != "main" {
            session
                .create_branch_from_turn(3, Some(branch_id))
                .await
                .unwrap();
            session.switch_to_branch("main").await.unwrap();
        }
        let before = session
            .get_items_for_branch(None, Some(branch_id))
            .await
            .unwrap();

        let err = session
            .create_branch_from_turn(2, Some(branch_id))
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("already been used"), "{err}");
        assert_eq!(session.current_branch(), "main");
        assert_eq!(
            session
                .get_items_for_branch(None, Some(branch_id))
                .await
                .unwrap(),
            before
        );
        assert_eq!(
            session
                .get_items_for_branch(None, Some("main"))
                .await
                .unwrap(),
            items
        );
    }
}

#[tokio::test]
async fn generated_branch_ids_do_not_merge_within_a_second() {
    let session = new_session("generated_branch_collision");
    let items = collision_items();
    session.add_items(items.clone()).await.unwrap();

    let first = session.create_branch_from_turn(3, None).await.unwrap();
    session.switch_to_branch("main").await.unwrap();
    let second = session.create_branch_from_turn(3, None).await.unwrap();

    assert_ne!(first, second);
    assert!(first.starts_with("branch_from_turn_3_"), "{first}");
    assert!(second.starts_with("branch_from_turn_3_"), "{second}");
    assert_eq!(
        session
            .get_items_for_branch(None, Some(&first))
            .await
            .unwrap(),
        items[..4].to_vec()
    );
    assert_eq!(
        session
            .get_items_for_branch(None, Some(&second))
            .await
            .unwrap(),
        items[..4].to_vec()
    );
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap(),
        items
    );
}

#[tokio::test]
async fn branch_ids_stay_reserved_after_delete_and_clear() {
    let session = new_session("reserved_after_delete");
    let items = collision_items();
    session.add_items(items.clone()).await.unwrap();
    session
        .create_branch_from_turn(3, Some("reserved"))
        .await
        .unwrap();
    session.switch_to_branch("main").await.unwrap();
    session.delete_branch("reserved", false).await.unwrap();

    let err = session
        .create_branch_from_turn(2, Some("reserved"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("already been used"), "{err}");
}

#[tokio::test]
async fn deleting_a_branch_removes_branch_only_messages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cleanup.db");
    let session = AdvancedSqliteSession::open("branch_delete_cleanup_test", &path).unwrap();
    let main_items = vec![
        user("First question"),
        assistant("First answer"),
        user("Second question"),
        assistant("Second answer"),
    ];
    session.add_items(main_items.clone()).await.unwrap();
    session
        .create_branch_from_turn(2, Some("cleanup_branch"))
        .await
        .unwrap();
    session
        .add_items(vec![
            user("Branch-only question"),
            assistant("Branch-only answer"),
        ])
        .await
        .unwrap();

    session.delete_branch("cleanup_branch", true).await.unwrap();

    let conn = rusqlite::Connection::open(&path).unwrap();
    let mut stmt = conn
        .prepare("SELECT message_data FROM agent_messages ORDER BY id")
        .unwrap();
    let contents: Vec<String> = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .map(|r| {
            serde_json::from_str::<Value>(&r.unwrap()).unwrap()["content"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(
        contents,
        vec![
            "First question",
            "First answer",
            "Second question",
            "Second answer"
        ]
    );
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap(),
        main_items
    );
}

#[tokio::test]
async fn deleting_a_branch_keeps_messages_another_branch_still_references() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.db");
    let session =
        AdvancedSqliteSession::open("branch_delete_shared_descendant_test", &path).unwrap();
    let main_items = vec![
        user("Main first question"),
        assistant("Main first answer"),
        user("Main second question"),
        assistant("Main second answer"),
    ];
    let shared = vec![
        user("Branch A shared question"),
        assistant("Branch A shared answer"),
    ];
    let only = vec![
        user("Branch A only question"),
        assistant("Branch A only answer"),
    ];

    session.add_items(main_items.clone()).await.unwrap();
    session
        .create_branch_from_turn(2, Some("branch_a"))
        .await
        .unwrap();
    session
        .add_items([shared.clone(), only].concat())
        .await
        .unwrap();
    session
        .create_branch_from_turn(3, Some("branch_b"))
        .await
        .unwrap();
    session.delete_branch("branch_a", false).await.unwrap();

    let conn = rusqlite::Connection::open(&path).unwrap();
    let mut stmt = conn
        .prepare("SELECT message_data FROM agent_messages")
        .unwrap();
    let all: String = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(all.contains("Branch A shared question") && all.contains("Branch A shared answer"));
    assert!(!all.contains("Branch A only question") && !all.contains("Branch A only answer"));
    assert_eq!(
        session
            .get_items_for_branch(None, Some("branch_b"))
            .await
            .unwrap(),
        [main_items[..2].to_vec(), shared].concat()
    );
}

#[tokio::test]
async fn deleting_a_branch_by_its_exact_id_leaves_a_similar_id_intact() {
    for force in [false, true] {
        let session = new_session("exact_branch_deletion");
        let main_items = vec![
            user("First question"),
            assistant("First answer"),
            user("Second question"),
        ];
        let survivor = vec![user("Keep this branch's history")];
        let target = vec![user("Delete this branch's history")];

        session.add_items(main_items.clone()).await.unwrap();
        session
            .create_branch_from_turn(2, Some("draft"))
            .await
            .unwrap();
        session.add_items(survivor.clone()).await.unwrap();
        session.switch_to_branch("main").await.unwrap();
        let target_id = session
            .create_branch_from_turn(2, Some(" draft "))
            .await
            .unwrap();
        session.add_items(target).await.unwrap();
        if !force {
            session.switch_to_branch("main").await.unwrap();
        }

        session.delete_branch(&target_id, force).await.unwrap();

        assert_eq!(
            session
                .get_items_for_branch(None, Some("draft"))
                .await
                .unwrap(),
            [main_items[..2].to_vec(), survivor].concat()
        );
        assert!(
            session
                .get_items_for_branch(None, Some(&target_id))
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(session.get_items(None).await.unwrap(), main_items);
        let ids: BTreeSet<String> = session
            .list_branches()
            .await
            .unwrap()
            .into_iter()
            .map(|b| b.branch_id)
            .collect();
        assert_eq!(
            ids,
            ["main".to_string(), "draft".to_string()]
                .into_iter()
                .collect()
        );
    }
}

#[tokio::test]
async fn deleting_the_current_branch_requires_force_and_returns_to_main() {
    let session = new_session("force_delete_test");
    session
        .add_items(vec![user("Main question")])
        .await
        .unwrap();
    session
        .add_items(vec![user("Second question")])
        .await
        .unwrap();
    session
        .create_branch_from_turn(2, Some("temp_branch"))
        .await
        .unwrap();
    assert_eq!(session.current_branch(), "temp_branch");
    session
        .add_items(vec![user("Branch question")])
        .await
        .unwrap();

    let err = session
        .delete_branch("temp_branch", false)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("Cannot delete current branch"), "{err}");

    session.delete_branch("temp_branch", true).await.unwrap();
    assert_eq!(session.current_branch(), "main");
    let after = session.list_branches().await.unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].branch_id, "main");
}

#[tokio::test]
async fn a_failed_force_delete_keeps_the_current_branch() {
    let session = new_session("failed_force_delete");
    let main_items = vec![user("Main question"), assistant("Main answer")];
    let branch_items = vec![user("Branch question")];
    session.add_items(main_items.clone()).await.unwrap();

    // Branching from turn 1 copies nothing, so the branch has no rows yet.
    let branch_id = session
        .create_branch_from_turn(1, Some("alternative_path"))
        .await
        .unwrap();
    assert_eq!(session.current_branch(), branch_id);

    let err = session
        .delete_branch(&branch_id, true)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Branch 'alternative_path' does not exist"),
        "{err}"
    );
    assert_eq!(session.current_branch(), branch_id);

    session.add_items(branch_items.clone()).await.unwrap();
    assert_eq!(session.get_items(None).await.unwrap(), branch_items);
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap(),
        main_items
    );
}

#[tokio::test]
async fn branch_operations_report_python_error_messages() {
    let session = new_session("branch_error_test");

    let cases: Vec<(String, &str)> = vec![
        (
            session
                .create_branch_from_turn(5, Some("error_branch"))
                .await
                .unwrap_err()
                .to_string(),
            "Turn 5 does not contain a user message",
        ),
        (
            session
                .switch_to_branch("nonexistent")
                .await
                .unwrap_err()
                .to_string(),
            "Branch 'nonexistent' does not exist",
        ),
        (
            session
                .delete_branch("nonexistent", false)
                .await
                .unwrap_err()
                .to_string(),
            "Branch 'nonexistent' does not exist",
        ),
        (
            session
                .delete_branch("main", false)
                .await
                .unwrap_err()
                .to_string(),
            "Cannot delete the 'main' branch",
        ),
        (
            session
                .delete_branch("", false)
                .await
                .unwrap_err()
                .to_string(),
            "Branch ID cannot be empty",
        ),
        (
            session
                .delete_branch("   ", false)
                .await
                .unwrap_err()
                .to_string(),
            "Branch ID cannot be empty",
        ),
    ];
    for (actual, expected) in cases {
        assert!(
            actual.contains(expected),
            "{actual:?} should contain {expected:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Turns and content search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn conversation_turns_describe_each_user_turn() {
    let session = new_session("conversation_turns_test");
    session
        .add_items(vec![user("Hello there"), assistant("Hi!")])
        .await
        .unwrap();
    session
        .add_items(vec![
            user("How are you doing today?"),
            assistant("I'm doing well, thanks!"),
        ])
        .await
        .unwrap();

    let turns = session.get_conversation_turns(None).await.unwrap();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].turn, 1);
    assert_eq!(turns[0].content, "Hello there");
    assert_eq!(turns[0].full_content, "Hello there");
    assert!(turns[0].can_branch);
    assert!(turns[0].timestamp.is_some());
    assert_eq!(turns[1].turn, 2);
    assert_eq!(turns[1].content, "How are you doing today?");
}

#[tokio::test]
async fn find_turns_by_content_returns_matching_turns() {
    let session = new_session("find_turns_test");
    session
        .add_items(vec![
            user("Tell me about cats"),
            assistant("Cats are great pets"),
        ])
        .await
        .unwrap();
    session
        .add_items(vec![
            user("What about dogs?"),
            assistant("Dogs are also great pets"),
        ])
        .await
        .unwrap();
    session
        .add_items(vec![
            user("Tell me about cats again"),
            assistant("Cats are wonderful companions"),
        ])
        .await
        .unwrap();

    let turns_of = |found: Vec<openai_agents_sdk::memory::ConversationTurn>| -> Vec<i64> {
        found.iter().map(|t| t.turn).collect()
    };
    assert_eq!(
        turns_of(session.find_turns_by_content("cats", None).await.unwrap()),
        vec![1, 3]
    );
    assert_eq!(
        turns_of(session.find_turns_by_content("dogs", None).await.unwrap()),
        vec![2]
    );
    assert!(
        session
            .find_turns_by_content("elephants", None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn find_turns_by_content_matches_literal_text() {
    let cases: [(&str, &str, &str); 14] = [
        ("Dogs", "Tell me about cats", "CATS"),
        ("Tokyo", "大阪の天気", "大阪"),
        ("Hello", "你好世界", "你好"),
        (r"Literal \u00e9", "café", "é"),
        ("café", r"Literal \u00e9", r"\u00e9"),
        ("CAFÉ", "café", "é"),
        (r"Literal \n", "First\nsecond", "\n"),
        ("First\nsecond", r"Literal \n", r"\n"),
        ("No quotes", "Say \"hello\"", "\"hello\""),
        ("Plain path", r"C:\notes", r"C:\notes"),
        ("100 dollars 東京", "100% 東京", "100% 東京"),
        ("axb 東京", "a_b 東京", "a_b 東京"),
        ("No percent", "50%", "%"),
        ("No underscore", "a_b", "_"),
    ];
    for (earlier, matching, term) in cases {
        let session = new_session("literal_search");
        session.add_items(vec![user(earlier)]).await.unwrap();
        session.add_items(vec![user(matching)]).await.unwrap();

        let found = session.find_turns_by_content(term, None).await.unwrap();
        assert_eq!(
            found.iter().map(|t| t.turn).collect::<Vec<_>>(),
            vec![2],
            "earlier={earlier:?} matching={matching:?} term={term:?}"
        );
        assert_eq!(found[0].full_content, matching);
        assert_eq!(found[0].content, matching);
    }
}

#[tokio::test]
async fn create_branch_from_content_uses_the_decoded_match() {
    for structured in [false, true] {
        let session = new_session("decoded_branch");
        let (earlier, matching) = if structured {
            (
                raw(
                    json!({"role": "user", "content": [{"type": "input_text", "text": r"Literal \u00e9"}]}),
                ),
                raw(json!({"role": "user", "content": [{"type": "input_text", "text": "café"}]})),
            )
        } else {
            (user(r"Literal \u00e9"), user("café"))
        };
        session.add_items(vec![earlier.clone()]).await.unwrap();
        session.add_items(vec![matching]).await.unwrap();

        let found = session.find_turns_by_content("é", None).await.unwrap();
        assert_eq!(found.iter().map(|t| t.turn).collect::<Vec<_>>(), vec![2]);
        assert_eq!(
            session
                .create_branch_from_content("é", Some("cafe_branch"))
                .await
                .unwrap(),
            "cafe_branch"
        );
        assert_eq!(session.get_items(None).await.unwrap(), vec![earlier]);
    }
}

#[tokio::test]
async fn escaped_text_parts_are_searched_after_decoding() {
    let session = new_session("escaped_text_parts");
    session
        .add_items(vec![raw(
            json!({"role": "user", "content": [{"type": "input_text", "text": r"Literal \n"}]}),
        )])
        .await
        .unwrap();
    session
        .add_items(vec![raw(json!({"role": "user", "content": [
            {"type": "input_text", "text": "東京 \"café\"\n50% a_b"},
            {"type": "input_image", "image_url": "https://example.com/image.png"},
        ]}))])
        .await
        .unwrap();

    for term in ["東京 \"café\"", "\n", "50% a_b"] {
        let found = session.find_turns_by_content(term, None).await.unwrap();
        assert_eq!(
            found.iter().map(|t| t.turn).collect::<Vec<_>>(),
            vec![2],
            "{term:?}"
        );
    }
    let literal = session.find_turns_by_content(r"\n", None).await.unwrap();
    assert_eq!(literal.iter().map(|t| t.turn).collect::<Vec<_>>(), vec![1]);
    assert!(
        session
            .find_turns_by_content("user", None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn find_turns_filters_by_session_and_branch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("content_search.db");
    let session = AdvancedSqliteSession::open("search", &path).unwrap();
    let other = AdvancedSqliteSession::open("other", &path).unwrap();

    other
        .add_items(vec![user("東京 other session")])
        .await
        .unwrap();
    session.add_items(vec![user("First turn")]).await.unwrap();
    session.add_items(vec![user("東京 main")]).await.unwrap();
    session
        .create_branch_from_turn(2, Some("alternate"))
        .await
        .unwrap();
    session
        .add_items(vec![user("東京 alternate")])
        .await
        .unwrap();

    let current = session.find_turns_by_content("東京", None).await.unwrap();
    assert_eq!(
        current
            .iter()
            .map(|t| t.full_content.clone())
            .collect::<Vec<_>>(),
        vec!["東京 alternate"]
    );
    let main = session
        .find_turns_by_content("東京", Some("main"))
        .await
        .unwrap();
    assert_eq!(
        main.iter()
            .map(|t| t.full_content.clone())
            .collect::<Vec<_>>(),
        vec!["東京 main"]
    );
}

#[tokio::test]
async fn list_content_turns_have_a_string_preview() {
    let session = new_session("conversation_turns_list_content_test");
    session
        .add_items(vec![raw(
            json!({"role": "user", "content": [{"type": "input_text", "text": "hello"}]}),
        )])
        .await
        .unwrap();
    let long_parts: Vec<Value> = (0..101)
        .map(|i| json!({"type": "input_text", "text": i.to_string()}))
        .collect();
    session
        .add_items(vec![raw(json!({"role": "user", "content": long_parts}))])
        .await
        .unwrap();

    let turns = session.get_conversation_turns(None).await.unwrap();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].content, "hello");
    assert!(turns[1].content.ends_with("..."), "{}", turns[1].content);
    assert!(turns[1].full_content.len() > turns[1].content.len());
}

#[tokio::test]
async fn find_turns_by_content_handles_list_content() {
    let session = new_session("find_turns_list_content_test");
    session
        .add_items(vec![raw(
            json!({"role": "user", "content": [{"type": "input_text", "text": "hello world"}]}),
        )])
        .await
        .unwrap();

    let found = session.find_turns_by_content("hello", None).await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].content, "hello world");
}

#[tokio::test]
async fn create_branch_from_content_branches_before_the_first_match() {
    let session = new_session("branch_from_content_test");
    session
        .add_items(vec![
            user("First question about math"),
            assistant("Math answer"),
        ])
        .await
        .unwrap();
    session
        .add_items(vec![
            user("Second question about science"),
            assistant("Science answer"),
        ])
        .await
        .unwrap();
    session
        .add_items(vec![
            user("Another math question"),
            assistant("Another math answer"),
        ])
        .await
        .unwrap();

    let name = session
        .create_branch_from_content("math", Some("math_branch"))
        .await
        .unwrap();
    assert_eq!(name, "math_branch");
    assert_eq!(session.current_branch(), "math_branch");
    assert!(session.get_items(None).await.unwrap().is_empty());

    let err = session
        .create_branch_from_content("nonexistent", Some("error_branch"))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("No user turns found containing 'nonexistent'"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_items_supports_limits_and_branch_ids() {
    let session = new_session("get_items_params_test");
    let items = vec![
        user("First question"),
        assistant("First answer"),
        user("Second question"),
        assistant("Second answer"),
    ];
    session.add_items(items).await.unwrap();

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["Second question", "Second answer"]
    );
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap()
            .len(),
        4
    );

    session
        .create_branch_from_turn(2, Some("test_branch"))
        .await
        .unwrap();
    session
        .add_items(vec![user("Branch question"), assistant("Branch answer")])
        .await
        .unwrap();
    assert_eq!(
        session
            .get_items_for_branch(None, Some("test_branch"))
            .await
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        session
            .get_items_for_branch(None, Some("main"))
            .await
            .unwrap()
            .len(),
        4
    );
}

#[tokio::test]
async fn sequence_order_is_preserved() {
    let session = new_session("sequence_test");
    session
        .add_items(vec![
            user("Message 1"),
            assistant("Response 1"),
            user("Message 2"),
            assistant("Response 2"),
        ])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Message 1", "Response 1", "Message 2", "Response 2"]
    );
}

#[tokio::test]
async fn an_empty_session_answers_every_query_empty() {
    let session = new_session("empty_test");
    assert!(session.get_items(None).await.unwrap().is_empty());
    assert!(session.get_turn_structure(None).await.unwrap().is_empty());
    assert!(session.get_tool_usage(None).await.unwrap().is_empty());
    assert!(session.get_session_usage(None).await.unwrap().is_none());
    assert!(
        session
            .get_conversation_turns(None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn sessions_with_different_ids_stay_isolated() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("iso.db");
    let one = AdvancedSqliteSession::open("session_1", &path).unwrap();
    let two = AdvancedSqliteSession::open("session_2", &path).unwrap();
    one.add_items(vec![user("Session 1 message")])
        .await
        .unwrap();
    two.add_items(vec![user("Session 2 message")])
        .await
        .unwrap();

    assert_eq!(
        texts(&one.get_items(None).await.unwrap()),
        vec!["Session 1 message"]
    );
    assert_eq!(
        texts(&two.get_items(None).await.unwrap()),
        vec!["Session 2 message"]
    );
    assert_eq!(one.get_turn_structure(None).await.unwrap().len(), 1);
    assert_eq!(two.get_turn_structure(None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn session_settings_limit_and_explicit_limit() {
    let session = new_session("settings").with_settings(SessionSettings::with_limit(3));
    session
        .add_items((0..5).map(|i| user(&format!("Message {i}"))).collect())
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Message 2", "Message 3", "Message 4"]
    );
    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["Message 3", "Message 4"]
    );
    assert_eq!(session.session_settings().unwrap().limit, Some(3));
}

#[tokio::test]
async fn a_limit_larger_than_history_returns_everything() {
    let session = new_session("limit_large");
    session
        .add_items(vec![user("only"), assistant("two")])
        .await
        .unwrap();
    assert_eq!(session.get_items(Some(10)).await.unwrap().len(), 2);
    let _ = MAIN_BRANCH;
}
