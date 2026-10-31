//! Parity ports from `tests/memory/test_session_persistence_sanitize.py` (through the
//! Conversations write path, where the sanitizer is observable) and
//! `tests/memory/test_compaction_suffix.py` (the SQLite-backed, non-TTL cases).

#![cfg(feature = "sqlite")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{
    CompactionMode, OpenAIConversationsSession, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, Session, SqliteSession,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, mock_openai, serve};

/// Conversations mock that records the `items` of every items-create request.
async fn conversation_writes() -> (OpenAIClient, Arc<Mutex<Vec<Value>>>) {
    let written = Arc::new(Mutex::new(Vec::new()));
    let sink = written.clone();
    let base = serve(Arc::new(move |req: &Request| {
        if req.method == "POST"
            && req.path.contains("/items")
            && let Some(items) = req.body.get("items").and_then(Value::as_array)
        {
            sink.lock().unwrap().extend(items.clone());
        }
        Reply::json(json!({"data": []}))
    }))
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"));
    (client, written)
}

/// The item as the Conversations store receives it (one write, then read back).
async fn sanitized(item: Value) -> Value {
    let (client, written) = conversation_writes().await;
    let session = OpenAIConversationsSession::from_conversation_id("conv", client);
    session
        .add_items(vec![InputItem::from_value(item)])
        .await
        .unwrap();
    let stored = written.lock().unwrap().clone();
    assert_eq!(stored.len(), 1);
    stored.into_iter().next().unwrap()
}

// ---------------------------------------------------------------------------
// test_session_persistence_sanitize.py
// ---------------------------------------------------------------------------

/// `test_sanitize_preserves_ids_required_by_openai_conversation_items` (13 parametrized cases).
#[tokio::test]
async fn sanitize_preserves_ids_required_by_conversation_item_types() {
    for ty in [
        "file_search_call",
        "web_search_call",
        "computer_call",
        "code_interpreter_call",
        "image_generation_call",
        "local_shell_call",
        "local_shell_call_output",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_call",
        "item_reference",
        "program",
        "program_output",
    ] {
        let out = sanitized(json!({
            "type": ty, "id": format!("{ty}_abc123"), "status": "completed",
        }))
        .await;
        assert_eq!(out["id"], json!(format!("{ty}_abc123")), "{ty}");
        assert_eq!(out["type"], json!(ty));
    }
}

/// `test_sanitize_preserves_file_search_call_payload_id`.
#[tokio::test]
async fn sanitize_preserves_file_search_call_payload() {
    let out = sanitized(json!({
        "type": "file_search_call",
        "id": "fs_call_abc",
        "queries": ["latest q3 revenue"],
        "status": "completed",
        "results": [{"file_id": "file_1", "filename": "q3.pdf", "score": 0.9, "text": "..."}],
    }))
    .await;
    assert_eq!(out["id"], json!("fs_call_abc"));
    assert_eq!(out["queries"], json!(["latest q3 revenue"]));
    assert_eq!(out["status"], json!("completed"));
}

/// `test_sanitize_preserves_program_payload_id`.
#[tokio::test]
async fn sanitize_preserves_program_payload() {
    let out = sanitized(json!({
        "type": "program",
        "id": "program_abc",
        "call_id": "call_program",
        "code": "lookup_inventory(sku=\"A-1\")",
        "fingerprint": "fingerprint",
    }))
    .await;
    assert_eq!(out["id"], json!("program_abc"));
    assert_eq!(out["call_id"], json!("call_program"));
    assert_eq!(out["code"], json!("lookup_inventory(sku=\"A-1\")"));
    assert_eq!(out["fingerprint"], json!("fingerprint"));
}

/// `test_sanitize_preserves_program_output_payload_id`.
#[tokio::test]
async fn sanitize_preserves_program_output_payload() {
    let out = sanitized(json!({
        "type": "program_output",
        "id": "program_output_abc",
        "call_id": "call_program",
        "result": "{\"available_units\":42}",
        "status": "completed",
    }))
    .await;
    assert_eq!(out["id"], json!("program_output_abc"));
    assert_eq!(out["call_id"], json!("call_program"));
    assert_eq!(out["result"], json!("{\"available_units\":42}"));
    assert_eq!(out["status"], json!("completed"));
}

/// `test_sanitize_strips_optional_or_policy_controlled_ids` (7 parametrized cases).
#[tokio::test]
async fn sanitize_strips_optional_or_policy_controlled_ids() {
    let cases = [
        json!({"type": "message", "id": "msg_abc", "role": "assistant",
               "content": [{"type": "output_text", "text": "hi"}]}),
        json!({"type": "function_call", "id": "fc_abc", "call_id": "call_abc",
               "name": "get_weather", "arguments": "{}"}),
        json!({"type": "function_call_output", "id": "out_abc", "call_id": "call_abc", "output": "{}"}),
        json!({"type": "computer_call_output", "id": "ccout_abc", "call_id": "call_abc", "output": {}}),
        json!({"type": "tool_search_call", "id": "ts_abc", "status": "completed"}),
        json!({"type": "shell_call", "id": "sh_abc", "call_id": "call_abc", "action": {}}),
        json!({"type": "function_call", "id": "fc_prog", "call_id": "call_abc",
               "name": "get_weather", "arguments": "{}",
               "caller": {"type": "program", "caller_id": "call_program"}}),
    ];
    for item in cases {
        let ty = item["type"].clone();
        let out = sanitized(item).await;
        assert!(out.get("id").is_none(), "{ty}: {out}");
        assert_eq!(out["type"], ty);
    }
}

/// `test_sanitize_preserves_reasoning_id_for_openai_conversations`.
#[tokio::test]
async fn sanitize_preserves_reasoning_id_and_drops_provider_data() {
    let out = sanitized(json!({
        "type": "reasoning", "id": "rs_abc", "summary": [], "content": [],
        "provider_data": {"server": "metadata"},
    }))
    .await;
    assert_eq!(out["id"], json!("rs_abc"));
    assert!(out.get("provider_data").is_none());
}

/// `test_sanitize_preserves_reasoning_encrypted_content`.
#[tokio::test]
async fn sanitize_preserves_reasoning_encrypted_content() {
    let out = sanitized(json!({
        "type": "reasoning", "summary": [], "content": [], "encrypted_content": "encrypted",
    }))
    .await;
    assert_eq!(out["encrypted_content"], json!("encrypted"));
}

/// `test_sanitize_always_strips_provider_data`.
#[tokio::test]
async fn sanitize_always_strips_provider_data() {
    let out = sanitized(json!({
        "type": "file_search_call", "id": "fs_keep", "status": "completed",
        "provider_data": {"model": "gpt-4.1-mini"},
    }))
    .await;
    assert_eq!(out["id"], json!("fs_keep"));
    assert!(out.get("provider_data").is_none());
}

// `test_sanitize_passes_through_non_dict_items` is PYTHON-ONLY: it asserts that an arbitrary
// Python object is returned by identity. Rust items are typed `InputItem` and cannot hold a
// non-object value.

// ---------------------------------------------------------------------------
// test_compaction_suffix.py (SQLite backend, no TTL clock)
// ---------------------------------------------------------------------------

fn compact_reply(output: Vec<Value>) -> Value {
    json!({
        "id": "cmp_1",
        "output": output,
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    })
}

fn history(n: usize) -> Vec<InputItem> {
    (0..n)
        .map(|i| InputItem::from_value(json!({"role": "user", "content": i.to_string()})))
        .collect()
}

/// `test_suffix_transaction_rolls_back_failed_insert`: a rejected summary insert must roll the
/// whole suffix replacement back, leaving the original rows and no compaction item.
#[tokio::test]
async fn suffix_transaction_rolls_back_a_failed_insert() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    let backend = Arc::new(SqliteSession::open("suffix", &path).unwrap());
    let original = history(4);
    backend.add_items(original.clone()).await.unwrap();
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER reject_summary BEFORE INSERT ON agent_messages \
             WHEN NEW.message_data LIKE '%compaction%' \
             BEGIN SELECT RAISE(ABORT, 'synthetic insert failure'); END",
        )
        .unwrap();
    }
    let (base, _) = mock_openai(vec![compact_reply(vec![json!({
        "type": "compaction", "id": "cmp", "encrypted_content": "synthetic-summary",
    })])])
    .await;
    let session = OpenAIResponsesCompactionSession::new("suffix", backend.clone())
        .unwrap()
        .with_compaction_mode(CompactionMode::Input)
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_should_trigger_compaction(|_| true);

    let err = session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("synthetic insert failure"), "{err}");
    let stored = backend.get_items(None).await.unwrap();
    assert_eq!(stored.len(), original.len());
    assert!(stored.iter().all(|i| i.item_type() != Some("compaction")));
}

/// `test_new_backend_write_invalidates_snapshot`: a write that lands while the compaction request
/// is in flight must not be overwritten by the replacement.
#[tokio::test]
async fn a_backend_write_during_compaction_invalidates_the_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    let backend = Arc::new(SqliteSession::open("suffix", &path).unwrap());
    let original = history(4);
    backend.add_items(original.clone()).await.unwrap();

    // The mock writes the concurrent row directly into the database while the request is open.
    let writer_path = path.clone();
    let base = serve(Arc::new(move |req: &Request| {
        if req.path.contains("/compact") {
            let conn = rusqlite::Connection::open(&writer_path).unwrap();
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
                rusqlite::params![
                    "suffix",
                    json!({"role": "user", "content": "concurrent backend append"}).to_string()
                ],
            )
            .unwrap();
        }
        Reply::json(compact_reply(vec![]))
    }))
    .await;

    let session = OpenAIResponsesCompactionSession::new("suffix", backend.clone())
        .unwrap()
        .with_compaction_mode(CompactionMode::Input)
        .with_client(OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")))
        .with_should_trigger_compaction(|_| true);
    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let stored = backend.get_items(None).await.unwrap();
    let mut expected = original.clone();
    expected.push(InputItem::from_value(
        json!({"role": "user", "content": "concurrent backend append"}),
    ));
    assert_eq!(stored, expected);
}
