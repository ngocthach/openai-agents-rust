//! Parity port of the compaction-output normalization and compaction-mode tests of
//! `tests/memory/test_openai_responses_compaction_session.py` (the part not covered by
//! `py_memory_compaction.rs`). Python's `MagicMock` client becomes a local mock
//! `/v1/responses/compact` endpoint whose captured request bodies are asserted.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{
    CompactionMode, DEFAULT_COMPACTION_THRESHOLD, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, Session,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{ListSession, Request, mock_openai};

fn compact_reply(output: Vec<Value>) -> Value {
    json!({
        "id": "cmp_1",
        "output": output,
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    })
}

fn message(role: &str, content: Value) -> InputItem {
    InputItem::from_value(json!({"type": "message", "role": role, "content": content}))
}

async fn session_over(
    underlying: Arc<dyn Session>,
    replies: Vec<Value>,
) -> (OpenAIResponsesCompactionSession, Arc<Mutex<Vec<Request>>>) {
    let (base, captured) = mock_openai(replies).await;
    let session = OpenAIResponsesCompactionSession::new("test", underlying)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));
    (session, captured)
}

fn force_input() -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        force: true,
        compaction_mode: Some(CompactionMode::Input),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Output normalization
// ---------------------------------------------------------------------------

async fn compacted_user_message(parts: Value) -> Vec<InputItem> {
    let underlying = Arc::new(ListSession::new("test"));
    let (session, _) = session_over(
        underlying.clone(),
        vec![compact_reply(vec![json!({
            "type": "message", "role": "user", "content": parts,
        })])],
    )
    .await;
    let session = session.with_compaction_mode(CompactionMode::Input);
    session.run_compaction(force_input()).await.unwrap();
    underlying.items()
}

#[tokio::test]
async fn compacted_user_image_messages_drop_null_sources() {
    let stored = compacted_user_message(json!([
        {"type": "input_text", "text": "analyze this input"},
        {"type": "input_image", "image_url": "https://example.com/image.png",
         "file_id": null, "detail": "auto"},
    ]))
    .await;
    assert_eq!(
        stored,
        vec![message(
            "user",
            json!([
                {"type": "input_text", "text": "analyze this input"},
                {"type": "input_image", "image_url": "https://example.com/image.png",
                 "detail": "auto"},
            ])
        )]
    );
}

#[tokio::test]
async fn compacted_user_file_messages_drop_null_sources() {
    let stored = compacted_user_message(json!([
        {"type": "input_text", "text": "analyze this input"},
        {"type": "input_file", "file_url": "https://example.com/report.pdf",
         "file_id": null, "filename": "report.pdf", "detail": "high"},
    ]))
    .await;
    assert_eq!(
        stored,
        vec![message(
            "user",
            json!([
                {"type": "input_text", "text": "analyze this input"},
                {"type": "input_file", "file_url": "https://example.com/report.pdf",
                 "filename": "report.pdf", "detail": "high"},
            ])
        )]
    );
}

#[tokio::test]
async fn compacted_file_id_inputs_keep_their_metadata() {
    let stored = compacted_user_message(json!([
        {"type": "input_text", "text": "analyze this input"},
        {"type": "input_file", "file_id": "file_123", "file_url": null,
         "filename": "report.pdf", "detail": "low"},
    ]))
    .await;
    assert_eq!(
        stored,
        vec![message(
            "user",
            json!([
                {"type": "input_text", "text": "analyze this input"},
                {"type": "input_file", "file_id": "file_123",
                 "filename": "report.pdf", "detail": "low"},
            ])
        )]
    );
}

#[tokio::test]
async fn a_normalization_failure_preserves_the_history() {
    let history = vec![
        message("user", json!([{"type": "input_text", "text": "hello"}])),
        InputItem::from_value(json!({
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "world"}],
        })),
    ];
    let underlying = Arc::new(ListSession::with_history("test", history.clone()));
    let (session, _) = session_over(
        underlying.clone(),
        vec![compact_reply(vec![json!({
            "type": "message", "role": "user",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_image", "detail": "auto"},
            ],
        })])],
    )
    .await;
    let session = session.with_compaction_mode(CompactionMode::Input);

    let err = session
        .run_compaction(force_input())
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("Compaction input_image item missing image_url or file_id."),
        "{err}"
    );
    assert_eq!(session.get_items(None).await.unwrap(), history);
    assert_eq!(underlying.items(), history);
}

// ---------------------------------------------------------------------------
// Compaction modes and request shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn input_mode_without_a_response_id_sends_the_model_and_the_input() {
    for model in [
        "gpt-4.1",
        "ft:gpt-4.1:my-org::id",
        "openai/gpt-4.1",
        "openai/openai/gpt-5.6-terra",
        "openai/o3",
        "openai/ft:gpt-4.1:org:proj:id",
    ] {
        let items = vec![
            message("user", json!("hello")),
            message("assistant", json!("world")),
        ];
        let underlying = Arc::new(ListSession::with_history("test", items.clone()));
        let (session, captured) = session_over(
            underlying,
            vec![compact_reply(vec![
                json!({"type": "message", "role": "assistant", "content": "compacted"}),
            ])],
        )
        .await;
        let session = session
            .with_model(model)
            .unwrap()
            .with_compaction_mode(CompactionMode::Input);

        session
            .run_compaction(OpenAIResponsesCompactionArgs {
                force: true,
                ..Default::default()
            })
            .await
            .unwrap();

        let body = captured.lock().unwrap()[0].body.clone();
        assert_eq!(body["model"], json!(model));
        assert!(
            body.get("previous_response_id").is_none(),
            "{model}: {body}"
        );
        assert_eq!(
            body["input"],
            json!(
                items
                    .iter()
                    .map(|i| i.as_value().clone())
                    .collect::<Vec<_>>()
            )
        );
    }
}

#[tokio::test]
async fn auto_mode_without_a_response_id_uses_input() {
    let items = vec![message("user", json!("hello"))];
    let underlying = Arc::new(ListSession::with_history("test", items.clone()));
    let (session, captured) = session_over(underlying, vec![compact_reply(vec![])]).await;

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert!(body.get("previous_response_id").is_none());
    assert_eq!(body["input"], json!([items[0].as_value().clone()]));
}

#[tokio::test]
async fn auto_mode_uses_input_when_the_response_was_not_stored() {
    let items = vec![
        message("user", json!("hello")),
        message("assistant", json!("world")),
    ];
    let underlying = Arc::new(ListSession::with_history("test", items.clone()));
    let (session, captured) = session_over(underlying, vec![compact_reply(vec![])]).await;
    let session = session.with_compaction_mode(CompactionMode::Auto);

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-auto".into()),
            store: Some(false),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert_eq!(body["model"], json!("gpt-4.1"));
    assert!(body.get("previous_response_id").is_none());
    assert_eq!(body["input"].as_array().map(Vec::len), Some(2));
}

#[tokio::test]
async fn an_unset_store_flag_defaults_to_a_stored_response() {
    let items = vec![
        message("user", json!("hello")),
        message("assistant", json!("world")),
    ];
    let underlying = Arc::new(ListSession::with_history("test", items));
    let (session, captured) = session_over(
        underlying,
        vec![compact_reply(vec![]), compact_reply(vec![])],
    )
    .await;
    let session = session.with_compaction_mode(CompactionMode::Auto);

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-auto".into()),
            store: Some(false),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-stored".into()),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].body.get("previous_response_id").is_none());
    assert_eq!(
        requests[1].body["previous_response_id"],
        json!("resp-stored")
    );
    assert!(requests[1].body.get("input").is_none());
}

#[tokio::test]
async fn auto_mode_stays_on_input_after_an_unstored_response() {
    let items = vec![
        message("user", json!("hello")),
        message("assistant", json!("world")),
    ];
    let underlying = Arc::new(ListSession::with_history("test", items));
    let compacted = json!({"type": "message", "role": "assistant", "content": "compacted"});
    let (session, captured) = session_over(
        underlying,
        vec![
            compact_reply(vec![compacted.clone()]),
            compact_reply(vec![]),
        ],
    )
    .await;
    let session = session.with_compaction_mode(CompactionMode::Auto);

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-unstored".into()),
            store: Some(false),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].body.get("previous_response_id").is_none());
    assert!(requests[1].body.get("previous_response_id").is_none());
    assert_eq!(requests[1].body["input"], json!([compacted]));
}

#[tokio::test]
async fn input_mode_strips_internal_tool_call_metadata() {
    // The session-only description/title keys must never reach the compact request.
    let call = InputItem::from_value(json!({
        "type": "function_call", "call_id": "call_123", "name": "lookup_account",
        "arguments": "{}",
        "_session_description": "Lookup customer records.",
        "_session_title": "Lookup Account",
    }));
    let output = InputItem::from_value(json!({
        "type": "function_call_output", "call_id": "call_123", "output": "ok",
    }));
    let underlying = Arc::new(ListSession::with_history("test", vec![call, output]));
    let (session, captured) = session_over(underlying, vec![compact_reply(vec![])]).await;
    let session = session.with_compaction_mode(CompactionMode::Input);

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    let first = &body["input"][0];
    assert_eq!(first["type"], json!("function_call"));
    let rendered = first.to_string();
    assert!(!rendered.contains("Lookup customer records."), "{rendered}");
    assert!(!rendered.contains("Lookup Account"), "{rendered}");
}

#[tokio::test]
async fn compaction_after_the_threshold_via_the_cached_candidates() {
    // `run_compaction` with the default hook triggers exactly at the threshold.
    let items: Vec<InputItem> = (0..DEFAULT_COMPACTION_THRESHOLD)
        .map(|i| message("assistant", json!(format!("msg{i}"))))
        .collect();
    let underlying = Arc::new(ListSession::with_history("test", items));
    let (session, captured) = session_over(underlying, vec![compact_reply(vec![])]).await;

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-1".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(captured.lock().unwrap().len(), 1);
}
