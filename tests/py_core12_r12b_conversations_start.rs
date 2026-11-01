//! Ports of `start_openai_conversations_session` cases of
//! `tests/memory/test_openai_conversations_session.py`.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::OpenAIClient;
use openai_agents_sdk::memory::start_openai_conversations_session;
use openai_agents_sdk::models::set_default_openai_client;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, serve};

async fn mock(conversation_id: &'static str) -> (String, Arc<Mutex<Vec<Request>>>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let seen = captured.clone();
    let base = serve(Arc::new(move |req: &Request| {
        seen.lock().unwrap().push(req.clone());
        Reply::json(json!({"id": conversation_id}))
    }))
    .await;
    (format!("{base}/v1"), captured)
}

fn assert_created_with_empty_items(requests: &[Request]) {
    assert_eq!(requests.len(), 1, "{requests:?}");
    assert_eq!(requests[0].method, "POST");
    assert!(
        requests[0].path.ends_with("/conversations"),
        "{}",
        requests[0].path
    );
    assert_eq!(requests[0].body.get("items"), Some(&Value::Array(vec![])));
}

/// Python: `test_start_with_provided_client`, `test_start_with_none_client` (the default client
/// is used) and `test_start_with_none_client_fallback` (without one, a client built from the
/// environment). One test, because the default client and the environment are process-wide.
#[tokio::test]
async fn start_creates_an_empty_conversation_with_the_right_client() {
    let (provided_url, provided) = mock("test_conversation_id").await;
    let client = OpenAIClient::new("sk-test").with_base_url(provided_url);
    let id = start_openai_conversations_session(Some(client))
        .await
        .unwrap();
    assert_eq!(id, "test_conversation_id");
    assert_created_with_empty_items(&provided.lock().unwrap());

    let (default_url, default_requests) = mock("default_client_id").await;
    set_default_openai_client(
        OpenAIClient::new("sk-test").with_base_url(default_url),
        false,
    );
    let id = start_openai_conversations_session(None).await.unwrap();
    assert_eq!(id, "default_client_id");
    assert_created_with_empty_items(&default_requests.lock().unwrap());
    assert_eq!(provided.lock().unwrap().len(), 1);

    openai_agents_sdk::models::clear_default_openai_client();
    let (env_url, env_requests) = mock("fallback_client_id").await;
    // SAFETY: this is the only test in the binary, so nothing reads the environment concurrently.
    unsafe { std::env::set_var("OPENAI_BASE_URL", &env_url) };
    let id = start_openai_conversations_session(None).await.unwrap();
    assert_eq!(id, "fallback_client_id");
    assert_created_with_empty_items(&env_requests.lock().unwrap());
}
