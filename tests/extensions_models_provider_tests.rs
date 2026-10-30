//! Wire-format tests for the non-OpenAI provider endpoints reached by
//! `extensions::models`: Azure OpenAI, AWS Bedrock (SigV4), Google Vertex AI (OAuth) and
//! IBM watsonx.ai (IAM). Everything runs against local mock servers; no provider is contacted.

#![cfg(feature = "provider-auth")]

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::extensions::models::{
    AnyLLMModel, LitellmModel, convert_gemini_extra_content_to_provider_specific_fields,
    list_provider_models, provider_endpoint,
};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// One captured request.
#[derive(Clone, Debug)]
struct Captured {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Captured {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

/// Start a server replying with each canned JSON body in order.
async fn mock_server(replies: Vec<Value>) -> (String, Arc<Mutex<Vec<Captured>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let request = loop {
                let Ok(n) = sock.read(&mut tmp).await else {
                    return;
                };
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                let Some(idx) = text.find("\r\n\r\n") else {
                    continue;
                };
                let head = &text[..idx];
                let mut lines = head.lines();
                let start = lines.next().unwrap_or_default().to_string();
                let headers: Vec<(String, String)> = lines
                    .filter_map(|l| l.split_once(':'))
                    .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                    .collect();
                let len: usize = headers
                    .iter()
                    .find(|(k, _)| k == "content-length")
                    .and_then(|(_, v)| v.parse().ok())
                    .unwrap_or(0);
                if buf.len() < idx + 4 + len {
                    continue;
                }
                let mut parts = start.split_whitespace();
                break Captured {
                    method: parts.next().unwrap_or_default().to_string(),
                    target: parts.next().unwrap_or_default().to_string(),
                    headers,
                    body: String::from_utf8_lossy(&buf[idx + 4..idx + 4 + len]).to_string(),
                };
            };
            cap.lock().unwrap().push(request);
            let body = replies.next().expect("unexpected request").to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 connection: close\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}"), captured)
}

/// Provider resolution reads process-wide environment variables, so the tests that set them
/// take this lock.
async fn env_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

fn set_env(pairs: &[(&str, &str)]) {
    for (k, v) in pairs {
        // SAFETY: every test touching the environment holds `env_lock`.
        unsafe { std::env::set_var(k, v) };
    }
}

fn clear_env(keys: &[&str]) {
    for k in keys {
        // SAFETY: every test touching the environment holds `env_lock`.
        unsafe { std::env::remove_var(k) };
    }
}

fn chat_reply(text: &str) -> Value {
    json!({
        "id": "chatcmpl_1",
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

fn request(text: &str) -> ModelRequest {
    ModelRequest {
        input: vec![InputItem(json!({"role": "user", "content": text}))],
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

#[tokio::test]
async fn azure_uses_the_deployment_path_api_key_header_and_api_version() {
    let (base, captured) = mock_server(vec![chat_reply("hi")]).await;
    // Held for the whole test: provider resolution reads process-wide variables.
    let _guard = env_lock().await;
    let model = {
        set_env(&[
            (
                "AZURE_OPENAI_BASE_URL",
                &format!("{base}/openai/deployments/{{model}}"),
            ),
            ("AZURE_OPENAI_API_KEY", "azure-secret"),
            ("AZURE_OPENAI_API_VERSION", "2099-01-01"),
        ]);
        clear_env(&["AZURE_OPENAI_AD_TOKEN", "AZURE_AD_TOKEN"]);
        LitellmModel::new("azure/my-deployment").unwrap()
    };
    model.get_response(request("hello")).await.unwrap();

    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].method, "POST");
    assert_eq!(
        calls[0].target,
        "/openai/deployments/my-deployment/chat/completions?api-version=2099-01-01"
    );
    assert_eq!(calls[0].header("api-key"), Some("azure-secret"));
    // Azure does not take a bearer token alongside the deployment key.
    assert_eq!(calls[0].header("authorization"), None);
    clear_env(&[
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ]);
}

#[tokio::test]
async fn azure_prefers_an_entra_id_token_over_the_deployment_key() {
    let (base, captured) = mock_server(vec![chat_reply("hi")]).await;
    // Held for the whole test: provider resolution reads process-wide variables.
    let _guard = env_lock().await;
    let model = {
        set_env(&[
            (
                "AZURE_OPENAI_BASE_URL",
                &format!("{base}/openai/deployments/{{model}}"),
            ),
            ("AZURE_OPENAI_API_KEY", "azure-secret"),
            ("AZURE_OPENAI_AD_TOKEN", "entra-token"),
        ]);
        LitellmModel::new("azure/dep").unwrap()
    };
    model.get_response(request("hello")).await.unwrap();
    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls[0].header("authorization"), Some("Bearer entra-token"));
    clear_env(&[
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_AD_TOKEN",
    ]);
}

#[tokio::test]
async fn bedrock_signs_with_sigv4() {
    let (base, captured) = mock_server(vec![chat_reply("hi")]).await;
    // Held for the whole test: provider resolution reads process-wide variables.
    let _guard = env_lock().await;
    let model = {
        set_env(&[
            ("AWS_BEDROCK_BASE_URL", &format!("{base}/openai/v1")),
            ("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_SESSION_TOKEN", "session"),
            ("AWS_REGION", "eu-west-1"),
        ]);
        clear_env(&["AWS_BEARER_TOKEN_BEDROCK"]);
        AnyLLMModel::new("bedrock/anthropic.claude-v2", None, None, None).unwrap()
    };
    model.get_response(request("hello")).await.unwrap();

    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls[0].target, "/openai/v1/chat/completions");
    let authorization = calls[0].header("authorization").unwrap();
    assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
    assert!(authorization.contains("/eu-west-1/bedrock/aws4_request"));
    assert!(
        authorization.contains("SignedHeaders=content-type;host;x-amz-date;x-amz-security-token")
    );
    assert!(calls[0].header("x-amz-date").is_some());
    assert_eq!(calls[0].header("x-amz-security-token"), Some("session"));
    assert_eq!(calls[0].json()["model"], "anthropic.claude-v2");
    clear_env(&[
        "AWS_BEDROCK_BASE_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
    ]);
}

#[tokio::test]
async fn bedrock_uses_a_long_lived_api_key_when_one_is_set() {
    let (base, captured) = mock_server(vec![chat_reply("hi")]).await;
    // Held for the whole test: provider resolution reads process-wide variables.
    let _guard = env_lock().await;
    let model = {
        set_env(&[
            ("AWS_BEDROCK_BASE_URL", &format!("{base}/openai/v1")),
            ("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key"),
        ]);
        AnyLLMModel::new("bedrock/model", None, None, None).unwrap()
    };
    model.get_response(request("hello")).await.unwrap();
    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls[0].header("authorization"), Some("Bearer bedrock-key"));
    clear_env(&["AWS_BEDROCK_BASE_URL", "AWS_BEARER_TOKEN_BEDROCK"]);
}

#[tokio::test]
async fn vertex_ai_exchanges_a_refresh_token_for_a_bearer() {
    let (token_base, token_calls) = mock_server(vec![
        json!({"access_token": "ya29.mock", "expires_in": 3600}),
    ])
    .await;
    let (base, captured) = mock_server(vec![chat_reply("hi"), chat_reply("again")]).await;
    let credentials = serde_json::to_string(&json!({
        "type": "authorized_user",
        "client_id": "cid",
        "client_secret": "csecret",
        "refresh_token": "rtoken",
    }))
    .unwrap();
    // Held for the whole test: provider resolution reads process-wide variables.
    let _guard = env_lock().await;
    let model = {
        set_env(&[
            ("VERTEXAI_BASE_URL", &base),
            ("GOOGLE_SERVICE_ACCOUNT_JSON", &credentials),
            ("GOOGLE_OAUTH_TOKEN_URI", &format!("{token_base}/token")),
        ]);
        clear_env(&["GOOGLE_OAUTH_ACCESS_TOKEN", "GOOGLE_ACCESS_TOKEN"]);
        AnyLLMModel::new("vertex_ai/gemini-2.5-pro", None, None, None).unwrap()
    };
    model.get_response(request("hello")).await.unwrap();
    // The second call reuses the cached token instead of exchanging again.
    model.get_response(request("again")).await.unwrap();

    let exchange = token_calls.lock().unwrap().clone();
    assert_eq!(exchange.len(), 1);
    assert!(exchange[0].body.contains("grant_type=refresh_token"));
    assert!(exchange[0].body.contains("refresh_token=rtoken"));
    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls.len(), 2);
    for call in &calls {
        assert_eq!(call.header("authorization"), Some("Bearer ya29.mock"));
    }
    clear_env(&[
        "VERTEXAI_BASE_URL",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_OAUTH_TOKEN_URI",
    ]);
}

#[tokio::test]
async fn watsonx_exchanges_an_iam_token_and_rewrites_the_body() {
    let (iam_base, iam_calls) = mock_server(vec![
        json!({"access_token": "iam-token", "expires_in": 3600}),
    ])
    .await;
    let (base, captured) = mock_server(vec![chat_reply("hi")]).await;
    // Held for the whole test: provider resolution reads process-wide variables.
    let _guard = env_lock().await;
    let model = {
        set_env(&[
            ("WATSONX_URL", &base),
            ("WATSONX_APIKEY", "ibm-key"),
            ("WATSONX_PROJECT_ID", "proj-1"),
            ("WATSONX_VERSION", "2030-01-01"),
            ("IBM_IAM_TOKEN_URL", &format!("{iam_base}/identity/token")),
        ]);
        AnyLLMModel::new("watsonx/ibm/granite-13b", None, None, None).unwrap()
    };
    model.get_response(request("hello")).await.unwrap();

    let exchange = iam_calls.lock().unwrap().clone();
    assert_eq!(exchange.len(), 1);
    assert!(exchange[0].body.contains("apikey=ibm-key"));
    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls[0].target, "/ml/v1/text/chat?version=2030-01-01");
    assert_eq!(calls[0].header("authorization"), Some("Bearer iam-token"));
    let body = calls[0].json();
    assert_eq!(body["model_id"], "ibm/granite-13b");
    assert_eq!(body["project_id"], "proj-1");
    assert!(body.get("model").is_none());
    clear_env(&[
        "WATSONX_URL",
        "WATSONX_APIKEY",
        "WATSONX_PROJECT_ID",
        "WATSONX_VERSION",
        "IBM_IAM_TOKEN_URL",
    ]);
}

#[tokio::test]
async fn sagemaker_invokes_the_endpoint_and_replays_a_stream() {
    use futures::StreamExt;

    let (base, captured) = mock_server(vec![chat_reply("from sagemaker")]).await;
    // Held for the whole test: provider resolution reads process-wide variables.
    let _guard = env_lock().await;
    let model = {
        set_env(&[
            (
                "AWS_SAGEMAKER_BASE_URL",
                &format!("{base}/endpoints/{{model}}"),
            ),
            ("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION", "us-west-2"),
        ]);
        clear_env(&["AWS_SESSION_TOKEN"]);
        AnyLLMModel::new("sagemaker/my-endpoint", None, None, None).unwrap()
    };
    // `/invocations` has no SSE route, so a streamed turn replays the finished response.
    let events: Vec<Value> = model
        .stream_response(request("hello"))
        .await
        .unwrap()
        .map(|e| e.unwrap())
        .collect()
        .await;
    assert_eq!(events.first().unwrap()["type"], "response.created");
    assert!(
        events
            .iter()
            .any(|e| e["type"] == "response.output_text.delta" && e["delta"] == "from sagemaker")
    );
    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "from sagemaker"
    );

    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls[0].target, "/endpoints/my-endpoint/invocations");
    assert!(
        calls[0]
            .header("authorization")
            .unwrap()
            .contains("/us-west-2/sagemaker/aws4_request")
    );
    clear_env(&[
        "AWS_SAGEMAKER_BASE_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    ]);
}

#[tokio::test]
async fn lists_the_models_an_endpoint_exposes() {
    let (base, captured) = mock_server(vec![json!({
        "object": "list",
        "data": [{"id": "llama-3.3-70b", "object": "model"}, {"id": "mixtral", "object": "model"}]
    })])
    .await;
    let models = list_provider_models("groq", Some(&format!("{base}/v1")), Some("k"))
        .await
        .unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0]["id"], "llama-3.3-70b");
    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls[0].method, "GET");
    assert_eq!(calls[0].target, "/v1/models");
}

#[tokio::test]
async fn the_anthropic_native_prefix_routes_to_the_messages_api() {
    let (base, captured) = mock_server(vec![json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })])
    .await;
    let model = LitellmModel::with_options(
        "anthropic_native/claude-sonnet-4-5",
        Some(&base),
        Some("sk-ant-test"),
        None,
    )
    .unwrap();
    model.get_response(request("hello")).await.unwrap();

    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls[0].method, "POST");
    assert_eq!(calls[0].target, "/messages");
    assert_eq!(calls[0].header("x-api-key"), Some("sk-ant-test"));
    assert_eq!(calls[0].header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(calls[0].header("authorization"), None);
    // The native protocol, not Chat Completions.
    let body = calls[0].json();
    assert_eq!(body["model"], "claude-sonnet-4-5");
    assert_eq!(body["messages"][0]["content"][0]["type"], "text");
    assert!(body.get("max_tokens").is_some());

    // The same prefix resolves for model listing, with the native auth headers.
    let (list_base, list_calls) =
        mock_server(vec![json!({"data": [{"id": "claude-sonnet-4-5"}]})]).await;
    let _guard = env_lock().await;
    set_env(&[
        ("ANTHROPIC_BASE_URL", &list_base),
        ("ANTHROPIC_API_KEY", "sk-ant-list"),
    ]);
    let models = list_provider_models("anthropic_messages", None, None)
        .await
        .unwrap();
    clear_env(&["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"]);
    assert_eq!(models[0]["id"], "claude-sonnet-4-5");
    let calls = list_calls.lock().unwrap().clone();
    assert_eq!(calls[0].header("x-api-key"), Some("sk-ant-list"));
    assert_eq!(calls[0].header("anthropic-version"), Some("2023-06-01"));
}

#[test]
fn knows_the_providers_any_llm_reaches() {
    for name in [
        "bedrock",
        "sagemaker",
        "vertex_ai",
        "azure",
        "azure_ai",
        "watsonx",
        "huggingface",
        "databricks",
        "nebius",
        "sambanova",
        "cohere",
        "deepinfra",
        "anthropic_native",
    ] {
        assert!(provider_endpoint(name).is_some(), "missing provider {name}");
    }
    // Aliases resolve to the canonical entry.
    assert_eq!(provider_endpoint("vertexai").unwrap().name, "vertex_ai");
    assert_eq!(provider_endpoint("AWS").unwrap().name, "bedrock");
    assert_eq!(provider_endpoint("hf").unwrap().name, "huggingface");
    assert_eq!(
        provider_endpoint("anthropic_messages").unwrap().name,
        "anthropic_native"
    );
}

#[test]
fn rewrites_gemini_thought_signatures_for_a_litellm_proxy() {
    let messages = vec![
        json!({"role": "assistant", "tool_calls": [
            {"id": "old", "extra_content": {"google": {"thought_signature": "before"}}}]}),
        json!({"role": "user", "content": "go"}),
        json!({"role": "assistant", "tool_calls": [
            {"id": "a", "extra_content": {"google": {"thought_signature": "sig"}}},
            {"id": "b"}]}),
    ];
    let out = convert_gemini_extra_content_to_provider_specific_fields(messages);
    // Only tool calls after the last user message are rewritten.
    assert_eq!(
        out[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        "before"
    );
    assert!(
        out[0]["tool_calls"][0]
            .get("provider_specific_fields")
            .is_none()
    );
    assert_eq!(
        out[2]["tool_calls"][0]["provider_specific_fields"]["thought_signature"],
        "sig"
    );
    assert!(out[2]["tool_calls"][0].get("extra_content").is_none());
    assert_eq!(
        out[2]["tool_calls"][1]["provider_specific_fields"]["thought_signature"],
        "skip_thought_signature_validator"
    );
}
