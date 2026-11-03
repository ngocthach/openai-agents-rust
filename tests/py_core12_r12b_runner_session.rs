//! Ports of `test_agent_runner.py` session-persistence cases that Python drives through the
//! private `save_result_to_session` / `prepare_input_with_session` helpers: here the same
//! behaviour is observed through the public session and runner APIs.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{OpenAIConversationsSession, Session};
use openai_agents_sdk::run_config::{ReasoningItemIdPolicy, RunConfig};
use openai_agents_sdk::testing::FakeModel;
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, serve};

/// A mock Conversations endpoint that records the items posted to it.
async fn conversations_mock() -> (OpenAIClient, Arc<Mutex<Vec<Value>>>) {
    let created = Arc::new(Mutex::new(Vec::new()));
    let created_items = created.clone();
    let base = serve(Arc::new(move |req: &Request| {
        if req.method == "POST" && req.path.ends_with("/conversations") {
            return Reply::json(json!({"id": "conv_test"}));
        }
        if req.method == "POST" && req.path.contains("/items") {
            if let Some(items) = req.body.get("items").and_then(Value::as_array) {
                created_items.lock().unwrap().extend(items.clone());
            }
            return Reply::json(json!({"data": []}));
        }
        Reply::json(json!({"data": [], "has_more": false}))
    }))
    .await;
    (
        OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")),
        created,
    )
}

async fn stored_reasoning(item: Value) -> Vec<Value> {
    let (client, created) = conversations_mock().await;
    let session = OpenAIConversationsSession::from_conversation_id("conv_test", client);
    session
        .add_items(vec![InputItem::from_value(item)])
        .await
        .unwrap();
    created.lock().unwrap().clone()
}

/// Python: `test_save_result_to_openai_conversation_drops_unpersistable_reasoning_item`: a
/// reasoning item with neither a server id nor encrypted content cannot be replayed from the
/// conversation store, so it is dropped.
#[tokio::test]
async fn conversation_store_drops_a_reasoning_item_it_cannot_replay() {
    let stored = stored_reasoning(json!({"type": "reasoning", "summary": [], "content": []})).await;
    assert!(stored.is_empty(), "{stored:?}");
}

/// Python: `test_save_result_to_openai_conversation_keeps_reasoning_encrypted_content`.
#[tokio::test]
async fn conversation_store_keeps_reasoning_encrypted_content() {
    let stored = stored_reasoning(json!({
        "type": "reasoning", "summary": [], "content": [], "encrypted_content": "encrypted"
    }))
    .await;
    assert_eq!(stored.len(), 1, "{stored:?}");
    assert_eq!(stored[0]["encrypted_content"], "encrypted");
}

/// Python: `test_save_result_to_openai_conversation_drops_placeholder_id_reasoning_item`: the
/// SDK's placeholder id is not a server identity, so such an item is no more persistable than
/// one without an id.
#[tokio::test]
async fn conversation_store_drops_a_placeholder_id_reasoning_item() {
    let stored = stored_reasoning(json!({
        "type": "reasoning", "id": "__fake_id__",
        "summary": [{"type": "summary_text", "text": "thinking"}]
    }))
    .await;
    assert!(stored.is_empty(), "{stored:?}");
}

/// Python: `test_save_result_to_openai_conversation_strips_placeholder_reasoning_id`.
#[tokio::test]
async fn conversation_store_strips_a_placeholder_reasoning_id() {
    let stored = stored_reasoning(json!({
        "type": "reasoning", "id": "__fake_id__", "summary": [], "encrypted_content": "encrypted"
    }))
    .await;
    assert_eq!(stored.len(), 1, "{stored:?}");
    assert_eq!(stored[0]["encrypted_content"], "encrypted");
    assert!(stored[0].get("id").is_none(), "{}", stored[0]);
}

/// Python: `test_save_result_to_session_counts_sanitized_openai_items` and
/// `test_save_result_to_openai_conversation_preserves_reasoning_id_when_policy_is_omit`: the run
/// writes its turn to a conversation store without provider data, and an `omit` reasoning policy
/// does not strip the ids the store owns.
#[tokio::test]
async fn a_run_writes_sanitized_items_and_keeps_conversation_reasoning_ids() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (client, created) = conversations_mock().await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client,
    ));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        openai_agents_sdk::items::OutputItem::from_value(json!({
            "type": "reasoning", "id": "rs_openai_conversation",
            "summary": [{"type": "summary_text", "text": "thinking"}]
        })),
        openai_agents_sdk::items::OutputItem::from_value(json!({
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            "provider_data": {"model": "litellm/test"}
        })),
    ]);
    let agent = Agent::<()>::new("agent").with_model_instance(model);
    let config = RunConfig::default().with_reasoning_item_id_policy(ReasoningItemIdPolicy::Omit);
    Runner::run_with_options(
        &agent,
        "input",
        RunOptions::new(()).session(session).run_config(config),
    )
    .await
    .unwrap();
    let stored = created.lock().unwrap().clone();
    let reasoning = stored
        .iter()
        .find(|item| item["type"] == "reasoning")
        .unwrap_or_else(|| panic!("no reasoning written: {stored:?}"));
    assert_eq!(reasoning["id"], "rs_openai_conversation");
    let message = stored
        .iter()
        .find(|item| item["role"] == "assistant")
        .unwrap_or_else(|| panic!("no message written: {stored:?}"));
    assert!(message.get("provider_data").is_none(), "{message}");
}

/// Python: `test_stream_input_persistence_strips_ids_for_openai_conversation_session`. A
/// streamed run with a session input callback writes the caller's input items to a conversation
/// store without their ids or provider data.
#[tokio::test]
async fn streamed_input_persistence_strips_ids_for_a_conversation_store() {
    use futures::StreamExt;
    use openai_agents_sdk::testing::assistant_message;
    openai_agents_sdk::set_tracing_disabled(true);
    let (client, created) = conversations_mock().await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client,
    ));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("ok")]);
    let agent = Agent::<()>::new("test").with_model_instance(model);
    let config = RunConfig {
        session_input_callback: Some(Arc::new(|existing, new| {
            let mut out = existing;
            out.extend(new);
            Box::pin(async move { Ok(out) })
        })),
        ..RunConfig::default()
    };
    let input = vec![InputItem::from_value(json!({
        "id": "message-1", "type": "message", "role": "user", "content": "hello",
        "provider_data": {"model": "litellm/test"}
    }))];
    let mut result = Runner::run_streamed_with_options(
        &agent,
        input,
        RunOptions::new(()).session(session).run_config(config),
    );
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    let stored = created.lock().unwrap().clone();
    let user = stored
        .iter()
        .find(|item| item["role"] == "user")
        .unwrap_or_else(|| panic!("input not persisted: {stored:?}"));
    assert_eq!(user["content"], "hello");
    assert!(user.get("id").is_none(), "{user}");
    assert!(user.get("provider_data").is_none(), "{user}");
}

/// Python: `test_sanitize_passes_through_non_dict_items`: an item that is not an object is
/// written exactly as it came.
#[tokio::test]
async fn conversation_store_passes_non_object_items_through() {
    let (client, created) = conversations_mock().await;
    let session = OpenAIConversationsSession::from_conversation_id("conv_test", client);
    session
        .add_items(vec![InputItem(json!("not an object"))])
        .await
        .unwrap();
    assert_eq!(*created.lock().unwrap(), vec![json!("not an object")]);
}
