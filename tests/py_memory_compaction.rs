//! Parity port of the behavioral core of `tests/memory/test_openai_responses_compaction_session.py`.
//!
//! Python injects a `MagicMock` OpenAI client; Rust's client is real HTTP, so each test runs
//! against a local mock `/v1/responses/compact` endpoint and asserts on the captured requests.
//! Tests that need pause-the-commit threads or asyncio task cancellation internals are listed
//! as Python-only in the report.

use std::sync::Arc;

use openai_agents_sdk::memory::{
    CompactionMode, DEFAULT_COMPACTION_THRESHOLD, InMemorySession, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, Session, default_should_trigger_compaction,
    is_openai_model_name, select_compaction_candidate_items,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{ListSession, mock_openai};

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"type": "message", "role": "assistant", "content": text}))
}

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"type": "message", "role": "user", "content": text}))
}

fn compact_reply(output: Vec<Value>) -> Value {
    json!({
        "id": "cmp_1",
        "output": output,
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    })
}

fn threshold_history() -> Vec<InputItem> {
    (0..DEFAULT_COMPACTION_THRESHOLD)
        .map(|i| assistant(&format!("msg{i}")))
        .collect()
}

async fn session_over(
    underlying: Arc<dyn Session>,
    replies: Vec<Value>,
) -> (
    OpenAIResponsesCompactionSession,
    Arc<std::sync::Mutex<Vec<py_mcpmem_support::Request>>>,
) {
    let (base, captured) = mock_openai(replies).await;
    let session = OpenAIResponsesCompactionSession::new("test", underlying)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base));
    (session, captured)
}

// ---------------------------------------------------------------------------
// Model names and candidate selection
// ---------------------------------------------------------------------------

#[test]
fn gpt_models_are_openai_models() {
    for name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-4.1",
        "gpt-5",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5-nano",
    ] {
        assert!(is_openai_model_name(name), "{name}");
    }
}

#[test]
fn o_models_and_fine_tunes_are_openai_models() {
    for name in [
        "o1",
        "o1-preview",
        "o3",
        "ft:gpt-4o-mini:org:proj:suffix",
        "ft:gpt-4.1:my-org::id",
    ] {
        assert!(is_openai_model_name(name), "{name}");
    }
}

#[test]
fn other_names_are_not_openai_models() {
    for name in ["", "not-openai", "claude-3", "anthropic/claude-3"] {
        assert!(!is_openai_model_name(name), "{name}");
    }
}

#[test]
fn provider_prefixed_openai_names_are_accepted() {
    for name in [
        "openai/gpt-4.1",
        "openai/openai/gpt-5.6-terra",
        "openai/o3",
        "openai/ft:gpt-4.1:org:proj:id",
    ] {
        assert!(is_openai_model_name(name), "{name}");
    }
}

#[test]
fn candidate_selection_excludes_user_messages() {
    let result = select_compaction_candidate_items(&[user("hello"), assistant("hi")]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].role(), Some("assistant"));
}

#[test]
fn candidate_selection_excludes_compaction_items() {
    let items = vec![
        InputItem::from_value(json!({"type": "compaction", "summary": "..."})),
        assistant("hi"),
    ];
    let result = select_compaction_candidate_items(&items);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].item_type(), Some("message"));
}

#[test]
fn candidate_selection_excludes_easy_user_messages_without_a_type() {
    let items = vec![
        InputItem::from_value(json!({"content": "hi", "role": "user"})),
        assistant("hello"),
    ];
    let result = select_compaction_candidate_items(&items);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].role(), Some("assistant"));
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn init_rejects_a_non_openai_model() {
    for model in ["claude-3", "anthropic/claude-3"] {
        let session =
            OpenAIResponsesCompactionSession::new("test", Arc::new(InMemorySession::new("t")))
                .unwrap();
        let err = session.with_model(model).unwrap_err().to_string();
        assert!(err.contains("Unsupported model"), "{err}");
    }
}

#[test]
fn init_accepts_a_valid_model() {
    let session =
        OpenAIResponsesCompactionSession::new("test", Arc::new(InMemorySession::new("t"))).unwrap();
    assert!(session.with_model("gpt-4.1").is_ok());
}

#[test]
fn nonpositive_rollback_budgets_are_rejected() {
    let session =
        OpenAIResponsesCompactionSession::new("test", Arc::new(InMemorySession::new("t"))).unwrap();
    let err = session.with_max_rollback_items(0).unwrap_err().to_string();
    assert!(err.contains("max_rollback_items must be positive"), "{err}");
}

// ---------------------------------------------------------------------------
// Delegation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_get_pop_and_clear_delegate_to_the_underlying_session() {
    let underlying = Arc::new(ListSession::new("test"));
    let (session, captured) = session_over(underlying.clone(), vec![]).await;

    session.add_items(vec![assistant("test")]).await.unwrap();
    assert_eq!(underlying.items().len(), 1);
    assert_eq!(session.get_items(None).await.unwrap().len(), 1);
    assert_eq!(
        session.pop_item().await.unwrap().unwrap().role(),
        Some("assistant")
    );
    session.add_items(vec![assistant("again")]).await.unwrap();
    session.clear_session().await.unwrap();
    assert!(underlying.items().is_empty());
    assert!(captured.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// run_compaction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn previous_response_id_mode_requires_a_response_id() {
    let (session, _) = session_over(Arc::new(ListSession::new("test")), vec![]).await;
    let session = session.with_compaction_mode(CompactionMode::PreviousResponseId);

    let err = session
        .run_compaction(OpenAIResponsesCompactionArgs::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("previous_response_id compaction"), "{err}");
}

#[tokio::test]
async fn compaction_is_skipped_below_the_threshold() {
    let history: Vec<InputItem> = (0..DEFAULT_COMPACTION_THRESHOLD - 1)
        .map(|i| assistant(&format!("msg{i}")))
        .collect();
    let underlying = Arc::new(ListSession::with_history("test", history));
    let (session, captured) = session_over(underlying, vec![]).await;

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-123".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(captured.lock().unwrap().is_empty());
}

#[tokio::test]
async fn compaction_runs_when_the_threshold_is_met_and_replaces_history() {
    let underlying = Arc::new(ListSession::with_history("test", threshold_history()));
    let (session, captured) = session_over(
        underlying.clone(),
        vec![compact_reply(vec![
            json!({"type": "compaction", "id": "cmp", "encrypted_content": "compacted"}),
        ])],
    )
    .await;
    let session = session.with_model("gpt-4.1").unwrap();

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-123".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].path, "/v1/responses/compact");
    assert_eq!(requests[0].body["previous_response_id"], json!("resp-123"));
    assert_eq!(requests[0].body["model"], json!("gpt-4.1"));
    let stored = underlying.items();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].item_type(), Some("compaction"));
}

#[tokio::test]
async fn force_bypasses_the_threshold() {
    let (session, captured) = session_over(
        Arc::new(ListSession::new("test")),
        vec![compact_reply(vec![])],
    )
    .await;

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-123".into()),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(captured.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn the_decision_hook_can_decline_compaction() {
    let underlying = Arc::new(ListSession::with_history("test", threshold_history()));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = calls.clone();
    let (session, captured) = session_over(underlying, vec![]).await;
    let session = session
        .with_compaction_mode(CompactionMode::Input)
        .with_should_trigger_compaction(move |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            false
        });

    session
        .run_compaction(OpenAIResponsesCompactionArgs::default())
        .await
        .unwrap();

    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(captured.lock().unwrap().is_empty());
}

#[tokio::test]
async fn input_mode_sends_the_stored_history_without_a_response_id() {
    let history = vec![user("hello"), assistant("world")];
    let underlying = Arc::new(ListSession::with_history("test", history.clone()));
    let (session, captured) = session_over(underlying, vec![compact_reply(vec![])]).await;
    let session = session.with_compaction_mode(CompactionMode::Input);

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].body.get("previous_response_id").is_none());
    assert_eq!(requests[0].body["input"].as_array().map(Vec::len), Some(2));
}

#[tokio::test]
async fn auto_mode_uses_input_when_the_response_was_not_stored() {
    let underlying = Arc::new(ListSession::with_history("test", vec![assistant("old")]));
    let (session, captured) = session_over(underlying, vec![compact_reply(vec![])]).await;

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-1".into()),
            store: Some(false),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert!(body.get("previous_response_id").is_none(), "{body}");
    assert!(body.get("input").is_some(), "{body}");
}

#[tokio::test]
async fn auto_mode_uses_the_response_chain_when_the_response_was_stored() {
    let underlying = Arc::new(ListSession::with_history("test", vec![assistant("old")]));
    let (session, captured) = session_over(underlying, vec![compact_reply(vec![])]).await;

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-1".into()),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let body = captured.lock().unwrap()[0].body.clone();
    assert_eq!(body["previous_response_id"], json!("resp-1"));
    assert!(body.get("input").is_none(), "{body}");
}

#[tokio::test]
async fn a_failed_compact_request_keeps_the_history() {
    let history = threshold_history();
    let underlying = Arc::new(ListSession::with_history("test", history.clone()));
    // The endpoint answers every request with a server error.
    let base = py_mcpmem_support::serve(Arc::new(|_req: &py_mcpmem_support::Request| {
        py_mcpmem_support::Reply::status(500, "{\"error\": {\"message\": \"boom\"}}")
    }))
    .await;
    let session = OpenAIResponsesCompactionSession::new("test", underlying.clone())
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")));

    let result = session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-1".into()),
            force: true,
            ..Default::default()
        })
        .await;

    assert!(result.is_err());
    assert_eq!(underlying.items(), history);
}

#[tokio::test]
async fn the_rollback_budget_rejects_before_any_request_or_write() {
    let history: Vec<InputItem> = (0..64)
        .map(|i| user(&format!("{i}:{}", "x".repeat(64))))
        .collect();
    let underlying = Arc::new(ListSession::with_history("test", history.clone()));
    let (session, captured) = session_over(underlying.clone(), vec![compact_reply(vec![])]).await;
    let session = session.with_max_rollback_items(3).unwrap();

    let err = session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("exceeds max_rollback_items"), "{err}");
    assert!(captured.lock().unwrap().is_empty());
    assert_eq!(underlying.items(), history);
}

#[tokio::test]
async fn an_exact_rollback_budget_still_compacts() {
    let underlying = Arc::new(ListSession::with_history("test", vec![user("hello")]));
    let (session, captured) = session_over(underlying.clone(), vec![compact_reply(vec![])]).await;
    let session = session.with_max_rollback_items(1).unwrap();

    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(captured.lock().unwrap().len(), 1);
    assert!(underlying.items().is_empty());
}

#[test]
fn the_default_decision_counts_candidates_against_the_threshold() {
    use openai_agents_sdk::memory::{CompactionContext, ResolvedCompactionMode};
    let make = |n: usize| CompactionContext {
        response_id: None,
        compaction_mode: ResolvedCompactionMode::Input,
        compaction_candidate_items: (0..n).map(|i| assistant(&format!("{i}"))).collect(),
        session_items: Vec::new(),
    };
    assert!(!default_should_trigger_compaction(&make(
        DEFAULT_COMPACTION_THRESHOLD - 1
    )));
    assert!(default_should_trigger_compaction(&make(
        DEFAULT_COMPACTION_THRESHOLD
    )));
}
