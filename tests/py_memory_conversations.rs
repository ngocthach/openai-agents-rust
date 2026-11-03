//! Parity port of `tests/memory/test_openai_conversations_session.py`.
//!
//! Python patches the OpenAI SDK's `conversations` namespace; Rust's client speaks real HTTP, so
//! these tests drive it against a local mock Conversations endpoint and assert on the requests.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{OpenAIConversationsSession, Session, SessionSettings};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text, serve};

/// A mock Conversations endpoint: lazily creates `conv_test`, serves `items` pages from `store`
/// and records every request.
async fn conversations_mock(
    pages: Vec<Value>,
) -> (
    OpenAIClient,
    Arc<Mutex<Vec<Request>>>,
    Arc<Mutex<Vec<Value>>>,
) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let queue = Arc::new(Mutex::new(pages));
    let created = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let created_items = created.clone();
    let base = serve(Arc::new(move |req: &Request| {
        cap.lock().unwrap().push(req.clone());
        if req.method == "POST" && req.path.ends_with("/conversations") {
            return Reply::json(json!({"id": "conv_test"}));
        }
        if req.method == "POST" && req.path.contains("/items") {
            if let Some(items) = req.body.get("items").and_then(Value::as_array) {
                created_items.lock().unwrap().extend(items.clone());
            }
            return Reply::json(json!({"data": []}));
        }
        if req.method == "DELETE" {
            return Reply::json(json!({"deleted": true}));
        }
        let mut queue = queue.lock().unwrap();
        if queue.is_empty() {
            Reply::json(json!({"data": [], "has_more": false}))
        } else {
            Reply::json(queue.remove(0))
        }
    }))
    .await;
    (
        OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")),
        captured,
        created,
    )
}

fn paths(requests: &[Request]) -> Vec<String> {
    requests
        .iter()
        .map(|r| format!("{} {}", r.method, r.path))
        .collect()
}

#[tokio::test]
async fn a_zero_limit_is_never_forwarded_to_the_items_endpoint() {
    let (client, captured, _) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::with_client(client);

    assert!(session.get_items(Some(0)).await.unwrap().is_empty());

    let requests = captured.lock().unwrap().clone();
    // The conversation is created, as in Python, but no items page is requested.
    assert_eq!(
        paths(&requests),
        vec!["POST /v1/conversations".to_string()],
        "{:?}",
        paths(&requests)
    );
    assert_eq!(session.try_session_id().unwrap(), "conv_test");
}

#[tokio::test]
async fn a_large_limit_is_applied_after_the_providers_pagination() {
    // Newest-first pages of 100 and 2 items; a limit of 101 must cut locally at 101 items.
    let newest_first: Vec<Value> = (0..102)
        .rev()
        .map(|i| json!({"id": format!("item_{i}"), "role": "user", "content": format!("{i}")}))
        .collect();
    let first_page = json!({
        "data": newest_first[..100],
        "has_more": true,
        "last_id": newest_first[99]["id"],
    });
    let second_page = json!({"data": newest_first[100..], "has_more": false});
    let (client, captured, _) = conversations_mock(vec![first_page, second_page]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    let items = session.get_items(Some(101)).await.unwrap();
    assert_eq!(items.len(), 101);
    // Oldest of the window first, newest last.
    assert_eq!(item_text(&items[0]), "1");
    assert_eq!(item_text(items.last().unwrap()), "101");

    let requests = captured.lock().unwrap().clone();
    assert!(
        requests
            .iter()
            .all(|r| !r.path.contains("limit=101") && !r.path.contains("limit=0")),
        "the session limit must not become the API page size: {:?}",
        paths(&requests)
    );
    assert!(requests.iter().all(|r| r.path.contains("order=desc")));
}

#[tokio::test]
async fn add_items_posts_one_request_for_a_short_list() {
    let (client, captured, created) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    session
        .add_items(vec![
            InputItem::user("Hello"),
            InputItem::from_value(json!({"role": "assistant", "content": "Hi there!"})),
        ])
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(
        paths(&requests),
        vec!["POST /v1/conversations/test_id/items".to_string()]
    );
    assert_eq!(created.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn add_items_creates_the_conversation_first() {
    let (client, captured, _) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::with_client(client);

    session
        .add_items(vec![InputItem::user("Hello")])
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(
        paths(&requests),
        vec![
            "POST /v1/conversations".to_string(),
            "POST /v1/conversations/conv_test/items".to_string(),
        ]
    );
    assert_eq!(requests[0].body, json!({"items": []}));
}

#[tokio::test]
async fn an_empty_add_items_creates_nothing() {
    let (client, captured, _) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::with_client(client);

    session.add_items(Vec::new()).await.unwrap();

    assert!(captured.lock().unwrap().is_empty());
    let err = session.try_session_id().unwrap_err().to_string();
    assert!(err.contains("add_items() with a non-empty list"), "{err}");
}

#[tokio::test]
async fn an_empty_add_items_keeps_an_existing_conversation_id() {
    let (client, captured, _) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    session.add_items(Vec::new()).await.unwrap();

    assert!(captured.lock().unwrap().is_empty());
    assert_eq!(session.try_session_id().unwrap(), "test_id");
}

#[tokio::test]
async fn add_items_splits_long_lists_into_requests_of_twenty_in_order() {
    let (client, captured, created) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    let items: Vec<InputItem> = (0..45).map(|i| InputItem::user(format!("m{i}"))).collect();
    session.add_items(items).await.unwrap();

    let requests = captured.lock().unwrap().clone();
    let sizes: Vec<usize> = requests
        .iter()
        .map(|r| r.body["items"].as_array().map(Vec::len).unwrap_or(0))
        .collect();
    assert_eq!(sizes, vec![20, 20, 5]);
    let stored = created.lock().unwrap().clone();
    assert_eq!(stored.len(), 45);
    assert_eq!(stored[0]["content"], json!("m0"));
    assert_eq!(stored[44]["content"], json!("m44"));
}

#[tokio::test]
async fn a_failed_chunk_keeps_the_prefix_and_stops() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let calls = Arc::new(Mutex::new(0usize));
    let base = serve(Arc::new(move |req: &Request| {
        cap.lock().unwrap().push(req.clone());
        if req.path.contains("/items") {
            let mut calls = calls.lock().unwrap();
            *calls += 1;
            if *calls == 2 {
                return Reply::status(500, "{\"error\": {\"message\": \"boom\"}}");
            }
        }
        Reply::json(json!({"data": []}))
    }))
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"));
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    let items: Vec<InputItem> = (0..45).map(|i| InputItem::user(format!("m{i}"))).collect();
    assert!(session.add_items(items).await.is_err());

    // The third chunk is never sent: the write stops at the failure.
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 2, "{:?}", paths(&requests));
}

#[tokio::test]
async fn pop_item_deletes_the_newest_item() {
    let latest = json!({
        "id": "item_123",
        "role": "assistant",
        "content": "Latest message",
        "status": "completed",
    });
    let (client, captured, _) =
        conversations_mock(vec![json!({"data": [latest], "has_more": false})]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    let popped = session.pop_item().await.unwrap().unwrap();
    assert_eq!(item_text(&popped), "Latest message");
    // Server-owned fields are stripped from what the caller gets back.
    assert!(popped.as_value().get("id").is_none());
    assert!(popped.as_value().get("status").is_none());

    let requests = captured.lock().unwrap().clone();
    assert!(
        paths(&requests).contains(&"DELETE /v1/conversations/test_id/items/item_123".to_string()),
        "{:?}",
        paths(&requests)
    );
}

#[tokio::test]
async fn pop_item_on_an_empty_conversation_deletes_nothing() {
    let (client, captured, _) =
        conversations_mock(vec![json!({"data": [], "has_more": false})]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    assert!(session.pop_item().await.unwrap().is_none());
    let requests = captured.lock().unwrap().clone();
    assert!(
        requests.iter().all(|r| r.method != "DELETE"),
        "{:?}",
        paths(&requests)
    );
}

#[tokio::test]
async fn clear_session_deletes_the_conversation_and_forgets_its_id() {
    let (client, captured, _) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    session.clear_session().await.unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(
        paths(&requests),
        vec!["DELETE /v1/conversations/test_id".to_string()]
    );
    assert!(session.try_session_id().is_err());
}

#[tokio::test]
async fn clear_session_on_an_uninitialized_session_calls_nothing() {
    let (client, captured, _) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::with_client(client);

    session.clear_session().await.unwrap();

    assert!(captured.lock().unwrap().is_empty());
    assert!(session.try_session_id().is_err());
}

#[tokio::test]
async fn a_failed_delete_retains_the_conversation_id_and_can_be_retried() {
    let calls = Arc::new(Mutex::new(0usize));
    let base = serve(Arc::new(move |req: &Request| {
        if req.method == "DELETE" {
            let mut calls = calls.lock().unwrap();
            *calls += 1;
            if *calls == 1 {
                return Reply::status(500, "{\"error\": {\"message\": \"boom\"}}");
            }
        }
        Reply::json(json!({"deleted": true}))
    }))
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"));
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    assert!(session.clear_session().await.is_err());
    assert_eq!(
        session.try_session_id().unwrap(),
        "test_id",
        "a failed delete must not forget the conversation"
    );

    session.clear_session().await.unwrap();
    assert!(session.try_session_id().is_err());
}

#[tokio::test]
async fn an_api_failure_while_creating_the_conversation_propagates() {
    let base = serve(Arc::new(|_req: &Request| {
        Reply::status(500, "{\"error\": {\"message\": \"creation failed\"}}")
    }))
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"));
    let session = OpenAIConversationsSession::with_client(client);

    assert!(
        session
            .add_items(vec![InputItem::user("hi")])
            .await
            .is_err()
    );
    assert!(session.try_session_id().is_err());
}

#[tokio::test]
async fn an_item_without_an_id_cannot_be_popped() {
    let (client, captured, _) = conversations_mock(vec![json!({
        "data": [{"role": "assistant", "content": "no id"}],
        "has_more": false,
    })])
    .await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);

    // Python raises `KeyError('id')`; the item stays stored and nothing is deleted.
    assert!(session.pop_item().await.is_err());
    let requests = captured.lock().unwrap().clone();
    assert!(requests.iter().all(|r| r.method != "DELETE"));
}

#[tokio::test]
async fn separate_sessions_use_separate_conversations() {
    let (client_a, captured_a, _) = conversations_mock(vec![]).await;
    let (client_b, captured_b, _) = conversations_mock(vec![]).await;
    let a = OpenAIConversationsSession::from_conversation_id("conv_a", client_a);
    let b = OpenAIConversationsSession::from_conversation_id("conv_b", client_b);

    a.add_items(vec![InputItem::user("a")]).await.unwrap();
    b.add_items(vec![InputItem::user("b")]).await.unwrap();

    assert_eq!(
        paths(&captured_a.lock().unwrap().clone()),
        vec!["POST /v1/conversations/conv_a/items".to_string()]
    );
    assert_eq!(
        paths(&captured_b.lock().unwrap().clone()),
        vec!["POST /v1/conversations/conv_b/items".to_string()]
    );
}

#[tokio::test]
async fn concurrent_first_writes_share_one_conversation() {
    let (client, captured, _) = conversations_mock(vec![]).await;
    let session = Arc::new(OpenAIConversationsSession::with_client(client));

    let one = session.clone();
    let two = session.clone();
    let (a, b) = tokio::join!(
        tokio::spawn(async move { one.add_items(vec![InputItem::user("one")]).await }),
        tokio::spawn(async move { two.add_items(vec![InputItem::user("two")]).await })
    );
    a.unwrap().unwrap();
    b.unwrap().unwrap();

    let requests = captured.lock().unwrap().clone();
    let creates = requests
        .iter()
        .filter(|r| r.method == "POST" && r.path.ends_with("/conversations"))
        .count();
    assert_eq!(creates, 1, "{:?}", paths(&requests));
    assert_eq!(session.try_session_id().unwrap(), "conv_test");
}

#[tokio::test]
async fn the_session_reports_that_it_stores_history_remotely() {
    let (client, _, _) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);
    assert!(session.manages_history_remotely());
}

#[tokio::test]
async fn session_settings_default_to_none_and_are_configurable() {
    let (client, _, _) = conversations_mock(vec![]).await;
    let plain = OpenAIConversationsSession::from_conversation_id("test_id", client.clone());
    assert!(plain.session_settings().is_none());

    let configured = OpenAIConversationsSession::from_conversation_id("test_id", client)
        .with_settings(SessionSettings::with_limit(5));
    assert_eq!(configured.session_settings().unwrap().limit, Some(5));
}

#[tokio::test]
async fn the_runner_stores_a_turn_in_the_conversation() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (client, captured, created) =
        conversations_mock(vec![json!({"data": [], "has_more": false})]).await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "test_id", client,
    ));

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("Hello!")]);
    let agent = Agent::<()>::new("Assistant").with_model_instance(model.clone());

    let result = Runner::run_with_options(
        &agent,
        "Say hello",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Hello!"));

    let stored = created.lock().unwrap().clone();
    assert!(
        stored
            .iter()
            .any(|i| i.get("content").and_then(Value::as_str) == Some("Say hello")),
        "{stored:?}"
    );
    let requests = captured.lock().unwrap().clone();
    assert!(
        requests
            .iter()
            .any(|r| r.method == "POST" && r.path.contains("/items")),
        "{:?}",
        paths(&requests)
    );
}

/// Port of `tests/memory/test_session_persistence_sanitize.py` through the public runner path:
/// the Conversations store must not receive the model's server-assigned message ids.
#[tokio::test]
async fn the_runner_strips_message_ids_before_writing_to_the_conversation() {
    use openai_agents_sdk::testing::assistant_message_with_id;
    openai_agents_sdk::set_tracing_disabled(true);
    let (client, _captured, created) =
        conversations_mock(vec![json!({"data": [], "has_more": false})]).await;
    let session = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "test_id", client,
    ));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message_with_id("Hello!", "msg_abc")]);
    let agent = Agent::<()>::new("Assistant").with_model_instance(model.clone());

    Runner::run_with_options(&agent, "hi", RunOptions::new(()).session(session.clone()))
        .await
        .unwrap();

    let stored = created.lock().unwrap().clone();
    let assistant = stored
        .iter()
        .find(|i| i.get("role").and_then(Value::as_str) == Some("assistant"))
        .expect("the assistant message is persisted");
    assert!(assistant.get("id").is_none(), "{assistant}");
    assert!(assistant.get("provider_data").is_none(), "{assistant}");
}

#[tokio::test]
async fn conversation_writes_keep_ids_only_where_the_schema_requires_them() {
    let (client, _captured, created) = conversations_mock(vec![]).await;
    let session = OpenAIConversationsSession::from_conversation_id("test_id", client);
    let required = [
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
    ];
    let mut items: Vec<InputItem> = required
        .iter()
        .map(|t| {
            InputItem::from_value(json!({
                "type": t, "id": format!("{t}_abc"), "status": "completed",
                "provider_data": {"model": "x"},
            }))
        })
        .collect();
    items.push(InputItem::from_value(
        json!({"type": "function_call", "id": "fc_abc", "call_id": "c", "name": "n", "arguments": "{}"}),
    ));
    items.push(InputItem::from_value(
        json!({"type": "reasoning", "id": "rs_abc", "summary": [], "provider_data": {"s": 1}}),
    ));
    items.push(InputItem::from_value(
        json!({"type": "mcp_call", "id": "__fake_id__"}),
    ));
    session.add_items(items).await.unwrap();

    let stored = created.lock().unwrap().clone();
    for (i, t) in required.iter().enumerate() {
        assert_eq!(stored[i]["id"], json!(format!("{t}_abc")));
        assert!(stored[i].get("provider_data").is_none());
    }
    assert!(stored[required.len()].get("id").is_none());
    assert_eq!(stored[required.len() + 1]["id"], json!("rs_abc"));
    assert!(stored[required.len() + 1].get("provider_data").is_none());
    assert!(stored[required.len() + 2].get("id").is_none());
}
