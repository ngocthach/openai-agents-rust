//! Parity ports from `tests/memory/test_openai_conversations_session.py`
//! (`TestStartOpenAIConversationsSession`, `TestOpenAIConversationsSessionConstructor`) and the
//! default-client lookup of `OpenAIResponsesCompactionSession`: a session built without a client
//! uses the process default client, and a conversation is created with an empty item list.
//!
//! The default client is process-global, so this file holds a single test that owns it.
#![cfg(all(feature = "sqlite", feature = "encrypt"))]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::OpenAIClient;
use openai_agents_sdk::memory::{
    OpenAIConversationsSession, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession,
};
use openai_agents_sdk::models::{clear_default_openai_client, set_default_openai_client};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{ListSession, Reply, Request, serve};

async fn endpoint() -> (OpenAIClient, Arc<Mutex<Vec<Request>>>) {
    let seen = Arc::new(Mutex::new(Vec::<Request>::new()));
    let record = seen.clone();
    let base = serve(Arc::new(move |req: &Request| {
        record.lock().unwrap().push(req.clone());
        if req.path.ends_with("/conversations") {
            Reply::json(json!({"id": "test_conversation_id"}))
        } else {
            Reply::json(json!({"id": "cmp", "output": [], "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}))
        }
    }))
    .await;
    (
        OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1")),
        seen,
    )
}

fn paths(seen: &Arc<Mutex<Vec<Request>>>) -> Vec<String> {
    seen.lock()
        .unwrap()
        .iter()
        .map(|r| format!("{} {}", r.method, r.path))
        .collect()
}

#[tokio::test]
async fn sessions_without_a_client_use_the_process_default_client() {
    // start_openai_conversations_session with a provided client: one create, `items: []`.
    let (explicit, seen) = endpoint().await;
    let session = OpenAIConversationsSession::with_client(explicit);
    assert_eq!(
        session.conversation_id().await.unwrap(),
        "test_conversation_id"
    );
    {
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].body, json!({"items": []}));
    }

    // Constructor with a conversation id and a client, and with a client only.
    let (client, _) = endpoint().await;
    let resumed = OpenAIConversationsSession::from_conversation_id("test_id", client.clone());
    assert_eq!(resumed.try_session_id().unwrap(), "test_id");
    let fresh = OpenAIConversationsSession::with_client(client);
    assert!(fresh.try_session_id().is_err(), "no id before first use");

    // start_openai_conversations_session(None) and the constructor without a client both go
    // through the default client.
    let (default_client, default_seen) = endpoint().await;
    set_default_openai_client(default_client, false);
    let started = OpenAIConversationsSession::new();
    assert_eq!(
        started.conversation_id().await.unwrap(),
        "test_conversation_id"
    );
    assert_eq!(paths(&default_seen), vec!["POST /v1/conversations"]);
    assert_eq!(default_seen.lock().unwrap()[0].body, json!({"items": []}));

    let resumed = OpenAIConversationsSession::new();
    assert!(resumed.try_session_id().is_err());

    // The compaction session resolves its client the same way.
    default_seen.lock().unwrap().clear();
    let underlying = Arc::new(ListSession::with_history(
        "default-client",
        vec![InputItem::from_value(
            json!({"role": "user", "content": "compact me"}),
        )],
    ));
    let compaction = OpenAIResponsesCompactionSession::new("default-client", underlying).unwrap();
    compaction
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(paths(&default_seen), vec!["POST /v1/responses/compact"]);
    let _: Value = default_seen.lock().unwrap()[0].body.clone();

    clear_default_openai_client();
}
