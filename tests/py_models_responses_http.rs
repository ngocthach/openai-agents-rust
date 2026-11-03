//! Port of the HTTP/SSE cluster of `tests/models/test_openai_responses.py`.
//!
//! Python drives `OpenAIResponsesModel` through `httpx2.MockTransport` or hand-written dummy
//! clients. Rust has one transport, so every test talks to a local TCP mock server that records
//! the request (target, headers, JSON body) and replays a scripted reply.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{ModelRequest, ModelTracing};
use openai_agents_sdk::models::{Model, OpenAIClient, OpenAIResponsesModel};
use openai_agents_sdk::retry::{ModelRetryAdviceRequest, ReplaySafety};
use openai_agents_sdk::tool::ImageGenerationTool;
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, set_tracing_disabled, trace,
};
use openai_agents_sdk::{Agent, AgentsError, RunConfig, RunOptions, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ------------------------------------------------------------------------------ harness

#[derive(Clone)]
enum Reply {
    /// 200 with a JSON body.
    Json(Value),
    /// 200 SSE: every event, then `[DONE]`.
    Sse(Vec<Value>),
    /// 200 SSE that promises more bytes than it sends, then the socket is reset: the transport
    /// fails while the events already delivered are intact.
    SseThenReset(Vec<Value>),
    /// An HTTP error with a JSON body and extra headers.
    Status(u16, Value, Vec<(&'static str, &'static str)>),
}

#[derive(Clone, Debug)]
struct Captured {
    target: String,
    headers: HashMap<String, String>,
    body: Value,
}

struct Server {
    base_url: String,
    captured: Arc<Mutex<Vec<Captured>>>,
}

impl Server {
    fn requests(&self) -> Vec<Captured> {
        self.captured.lock().unwrap().clone()
    }
    fn client(&self) -> OpenAIClient {
        OpenAIClient::new("sk-test").with_base_url(self.base_url.clone())
    }
    fn model(&self, name: &str) -> OpenAIResponsesModel {
        OpenAIResponsesModel::new(name, self.client())
    }
}

fn sse_body(events: &[Value], done: bool) -> String {
    let mut body = String::new();
    for e in events {
        body.push_str(&format!(
            "event: {}\ndata: {}\n\n",
            e["type"].as_str().unwrap_or("message"),
            e
        ));
    }
    if done {
        body.push_str("data: [DONE]\n\n");
    }
    body
}

/// Start a server that answers requests in order. `request_id` is sent as `x-request-id`.
async fn mock_server(replies: Vec<Reply>, request_id: Option<&'static str>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured: Arc<Mutex<Vec<Captured>>> = Arc::default();
    let cap = captured.clone();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        let mut last: Option<Reply> = None;
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let request = loop {
                let n = sock.read(&mut tmp).await.unwrap_or(0);
                if n == 0 {
                    break None;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let head = &text[..idx];
                    let mut headers = HashMap::new();
                    for line in head.lines().skip(1) {
                        if let Some((k, v)) = line.split_once(':') {
                            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
                        }
                    }
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        let target = head.split_whitespace().nth(1).unwrap().to_string();
                        let body = serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                        break Some(Captured {
                            target,
                            headers,
                            body,
                        });
                    }
                }
            };
            let Some(request) = request else { continue };
            cap.lock().unwrap().push(request);
            // Past the scripted replies, repeat the last one so an extra request is visible
            // to the test as a second capture rather than as a dropped socket.
            let Some(reply) = replies.next().or_else(|| last.clone()) else {
                continue;
            };
            last = Some(reply.clone());
            let rid = request_id
                .map(|r| format!("x-request-id: {r}\r\n"))
                .unwrap_or_default();
            let resp = match reply {
                Reply::Json(v) => {
                    let b = v.to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n{rid}content-length: {}\r\nconnection: close\r\n\r\n{b}",
                        b.len()
                    )
                }
                Reply::Sse(events) => {
                    let b = sse_body(&events, true);
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n{rid}content-length: {}\r\nconnection: close\r\n\r\n{b}",
                        b.len()
                    )
                }
                Reply::SseThenReset(events) => {
                    let b = sse_body(&events, false);
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n{rid}content-length: {}\r\nconnection: close\r\n\r\n{b}",
                        b.len() + 4096
                    )
                }
                Reply::Status(code, v, extra) => {
                    let b = v.to_string();
                    let extra: String =
                        extra.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
                    format!(
                        "HTTP/1.1 {code} Error\r\ncontent-type: application/json\r\n{rid}{extra}content-length: {}\r\nconnection: close\r\n\r\n{b}",
                        b.len()
                    )
                }
            };
            sock.write_all(resp.as_bytes()).await.ok();
            sock.shutdown().await.ok();
        }
    });
    Server {
        base_url: format!("http://{addr}/v1"),
        captured,
    }
}

fn response_json(id: &str, status: &str) -> Value {
    json!({"id": id, "object": "response", "created_at": 0, "model": "fake",
        "status": status, "output": [], "tool_choice": "none", "tools": [],
        "parallel_tool_calls": false, "top_p": null, "usage": null})
}

fn message_response(text: &str) -> Value {
    let mut r = response_json("resp-text", "completed");
    r["output"] = json!([{"type": "message", "id": "msg_1", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}]}]);
    r
}

/// The same response with the provider's `usage` left out.
fn no_usage(mut response: Value) -> Value {
    response["usage"] = Value::Null;
    response
}

fn completed_event(response: Value) -> Value {
    json!({"type": "response.completed", "sequence_number": 0, "response": response})
}

fn terminal_event(event_type: &str, status: &str) -> Value {
    let mut r = response_json("resp-terminal", status);
    if status == "incomplete" {
        r["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }
    json!({"type": event_type, "sequence_number": 0, "response": r})
}

fn error_event() -> Value {
    json!({"type": "error", "sequence_number": 0, "code": "server_error",
        "message": "synthetic provider failure", "param": null})
}

fn req(input: &str) -> ModelRequest {
    ModelRequest {
        input: vec![InputItem::user(input)],
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

fn quiet_options() -> RunOptions<()> {
    RunOptions::new(()).run_config(RunConfig::new().with_tracing_disabled(true))
}

/// Drive a stream to its end, returning the events seen and the error that ended it, if any.
async fn drain(
    model: &OpenAIResponsesModel,
    request: ModelRequest,
) -> (Vec<Value>, Option<AgentsError>) {
    let mut events = Vec::new();
    let mut stream = model.stream_response(request).await.unwrap();
    while let Some(item) = stream.next().await {
        match item {
            Ok(e) => events.push(e),
            Err(e) => return (events, Some(e)),
        }
    }
    (events, None)
}

// ---------------------------------------------------------- usage normalization

#[tokio::test]
async fn runner_normalizes_nullable_responses_usage() {
    type Counts = (Option<u64>, Option<u64>, Option<u64>);
    let cases: [(&str, Counts, (u64, u64, u64)); 6] = [
        ("null-input", (None, Some(5), Some(12)), (0, 5, 12)),
        ("null-output", (Some(7), None, Some(12)), (7, 0, 12)),
        ("null-total", (Some(7), Some(5), None), (7, 5, 0)),
        ("all-null", (None, None, None), (0, 0, 0)),
        ("zero", (Some(0), Some(0), Some(0)), (0, 0, 0)),
        ("valid", (Some(7), Some(5), Some(12)), (7, 5, 12)),
    ];
    for stream in [false, true] {
        for (name, counts, expected) in cases {
            let usage = json!({
                "input_tokens": counts.0, "output_tokens": counts.1, "total_tokens": counts.2,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
            });
            let mut payload = message_response("Hello");
            payload["usage"] = usage.clone();
            let reply = if stream {
                Reply::Sse(vec![completed_event(payload)])
            } else {
                Reply::Json(payload)
            };
            let server = mock_server(vec![reply], None).await;
            let agent = Agent::<()>::new("test")
                .with_model_instance(Arc::new(server.model("fake")))
                .with_model_settings(ModelSettings {
                    preserve_raw_usage: Some(true),
                    ..Default::default()
                });
            let result = if stream {
                let mut result = Runner::run_streamed_with_options(&agent, "hi", quiet_options());
                {
                    let mut events = std::pin::pin!(result.stream_events());
                    while let Some(e) = events.next().await {
                        e.unwrap();
                    }
                }
                result.into_result().expect("completed run")
            } else {
                Runner::run_with_options(&agent, "hi", quiet_options())
                    .await
                    .unwrap()
            };
            let label = format!("{name} stream={stream}");
            assert_eq!(
                result.final_output_text().as_deref(),
                Some("Hello"),
                "{label}"
            );
            let u = result.usage();
            assert_eq!(u.requests, 1, "{label}");
            assert_eq!(
                (u.input_tokens, u.output_tokens, u.total_tokens),
                expected,
                "{label}"
            );
            assert_eq!(u.input_tokens_details.cached_tokens, 3, "{label}");
            assert_eq!(u.output_tokens_details.reasoning_tokens, 2, "{label}");
            assert_eq!(result.raw_responses[0].raw_usage, Some(usage), "{label}");
            if expected == (0, 0, 0) {
                assert!(u.request_usage_entries.is_empty(), "{label}");
            } else {
                assert_eq!(u.request_usage_entries.len(), 1, "{label}");
                let entry = &u.request_usage_entries[0];
                assert_eq!(
                    (entry.input_tokens, entry.output_tokens, entry.total_tokens),
                    expected,
                    "{label}"
                );
                assert_eq!(entry.input_tokens_details.cached_tokens, 3, "{label}");
                assert_eq!(entry.output_tokens_details.reasoning_tokens, 2, "{label}");
            }
        }
    }
}

#[tokio::test]
async fn request_is_counted_when_the_provider_omits_usage() {
    let server = mock_server(
        vec![Reply::Json(response_json("resp-no-usage", "completed"))],
        None,
    )
    .await;
    let request = ModelRequest {
        model_settings: ModelSettings {
            preserve_raw_usage: Some(true),
            ..Default::default()
        },
        ..req("hi")
    };
    let response = server.model("gpt-4").get_response(request).await.unwrap();
    assert_eq!(response.usage.requests, 1);
    assert_eq!(response.usage.total_tokens, 0);
    assert!(response.raw_usage.is_none());
}

#[tokio::test]
async fn streamed_request_is_counted_when_the_provider_omits_usage() {
    let server = mock_server(
        vec![Reply::Sse(vec![completed_event(no_usage(
            message_response("ok"),
        ))])],
        Some("req-1"),
    )
    .await;
    let agent = Agent::<()>::new("test").with_model_instance(Arc::new(server.model("gpt-4")));
    let mut result = Runner::run_streamed_with_options(&agent, "hi", quiet_options());
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(e) = events.next().await {
            e.unwrap();
        }
    }
    let usage = result.into_result().expect("completed run").usage();
    assert_eq!(
        server.requests().len(),
        1,
        "the run must end after one model call"
    );
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.total_tokens, 0);
}

// ------------------------------------------------------------------- wire payloads

#[tokio::test]
async fn unpaired_function_output_is_preserved_in_the_request() {
    for null_call_id in [false, true] {
        for stream in [false, true] {
            let reply = if stream {
                Reply::Sse(vec![completed_event(response_json("resp-id", "completed"))])
            } else {
                Reply::Json(response_json("resp-id", "completed"))
            };
            let server = mock_server(vec![reply], None).await;
            let mut item = json!({
                "type": "function_call_output", "name": "notifications", "namespace": "slack",
                "output": [
                    {"type": "input_text", "text": "Alice mentioned you in #deployments."},
                    {"type": "input_image", "image_url": "https://example.com/image.png"}],
            });
            if null_call_id {
                item["call_id"] = Value::Null;
            }
            let request = ModelRequest {
                input: vec![InputItem::from_value(item.clone())],
                ..req("")
            };
            let model = server.model("gpt-4");
            if stream {
                drain(&model, request).await;
            } else {
                model.get_response(request).await.unwrap();
            }
            let bodies: Vec<Value> = server
                .requests()
                .into_iter()
                .map(|r| r.body["input"].clone())
                .collect();
            assert_eq!(
                bodies,
                vec![json!([item])],
                "null_call_id={null_call_id} stream={stream}"
            );
        }
    }
}

#[tokio::test]
async fn web_search_image_options_reach_the_request() {
    let cases = [
        (json!({}), Vec::<&str>::new()),
        (json!({"search_content_types": ["text"]}), vec![]),
        (
            json!({"search_content_types": ["image", "text"],
                "image_settings": {"max_results": 3, "caption": false}}),
            vec!["web_search_call.results"],
        ),
    ];
    for stream in [false, true] {
        for (options, include) in &cases {
            let reply = if stream {
                Reply::Sse(vec![completed_event(response_json("resp-id", "completed"))])
            } else {
                Reply::Json(response_json("resp-id", "completed"))
            };
            let server = mock_server(vec![reply], None).await;
            let mut tool: openai_agents_sdk::tool::WebSearchTool =
                serde_json::from_value(options.clone()).unwrap();
            tool.user_location = None;
            let request = ModelRequest {
                tools: vec![openai_agents_sdk::models::ToolSpec::Hosted {
                    param: tool.to_responses_param(),
                    includes: if include.is_empty() {
                        vec![]
                    } else {
                        vec!["web_search_call.results".to_string()]
                    },
                }],
                ..req("Find images of the Golden Gate Bridge.")
            };
            let model = server.model("gpt-5.6");
            if stream {
                drain(&model, request).await;
            } else {
                model.get_response(request).await.unwrap();
            }
            let body = server.requests().remove(0).body;
            let mut expected = json!({"type": "web_search", "search_context_size": "medium"});
            for (k, v) in options.as_object().unwrap() {
                expected[k] = v.clone();
            }
            // Python also sends explicit `filters: null` / `user_location: null`; Rust leaves the
            // unset fields out, which the API treats identically.
            assert_eq!(body["tools"], json!([expected]), "stream={stream}");
            assert_eq!(
                body.get("include").cloned().unwrap_or(json!([])),
                json!(include),
                "stream={stream}"
            );
        }
    }
}

#[tokio::test]
async fn image_generation_options_reach_the_request() {
    let configs = [
        json!({"type": "image_generation"}),
        json!({"type": "image_generation", "model": "gpt-image-1", "quality": "high",
            "action": "edit", "input_fidelity": "high", "input_image_mask": {"file_id": "file-mask"}}),
        json!({"type": "image_generation", "model": "gpt-image-2.5-sunburst", "quality": "max",
            "action": "generate", "size": "1536x864", "background": "transparent",
            "output_format": "webp", "output_compression": 0, "partial_images": 0,
            "moderation": "auto"}),
        json!({"type": "image_generation", "model": "gpt-image-2.5-flare", "quality": "xhigh",
            "action": "auto", "size": "auto", "output_format": "png", "partial_images": 2}),
    ];
    for stream in [false, true] {
        for config in &configs {
            let reply = if stream {
                Reply::Sse(vec![completed_event(message_response("done"))])
            } else {
                Reply::Json(message_response("done"))
            };
            let server = mock_server(vec![reply], None).await;
            let tool = ImageGenerationTool {
                tool_config: config.clone(),
            };
            let agent = Agent::<()>::new("Image generator")
                .with_model_instance(Arc::new(server.model("gpt-5.5")))
                .with_tool(tool.clone());
            let input = vec![InputItem::user_parts(vec![
                json!({"type": "input_text", "text": "Generate an image, or edit this image."}),
                json!({"type": "input_image", "file_id": "file-image", "detail": "auto"}),
            ])];
            if stream {
                let mut result = Runner::run_streamed_with_options(&agent, input, quiet_options());
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(e) = events.next().await {
                    e.unwrap();
                }
            } else {
                Runner::run_with_options(&agent, input, quiet_options())
                    .await
                    .unwrap();
            }
            let requests = server.requests();
            assert_eq!(requests.len(), 1);
            assert_eq!(
                requests[0].body["tools"],
                json!([config]),
                "stream={stream}"
            );
            assert_eq!(requests[0].body["model"], "gpt-5.5");
            assert_eq!(&tool.tool_config, config, "the tool config is not mutated");
        }
    }
}

#[tokio::test]
async fn user_agent_header_is_sent_and_overridable() {
    let server = mock_server(
        vec![
            Reply::Sse(vec![completed_event(response_json("r1", "completed"))]),
            Reply::Sse(vec![completed_event(response_json("r2", "completed"))]),
        ],
        None,
    )
    .await;
    let model = server.model("gpt-4");
    drain(&model, req("hi")).await;
    let overridden = ModelRequest {
        model_settings: ModelSettings {
            extra_headers: Some(HashMap::from([(
                "User-Agent".to_string(),
                "test_user_agent".to_string(),
            )])),
            ..Default::default()
        },
        ..req("hi")
    };
    drain(&model, overridden).await;
    let requests = server.requests();
    assert!(requests[0].headers["user-agent"].starts_with("Agents/Rust "));
    assert_eq!(requests[1].headers["user-agent"], "test_user_agent");
}

#[tokio::test]
async fn official_client_applies_transport_extra_query_and_body() {
    let server = mock_server(
        vec![
            Reply::Json(response_json("r1", "completed")),
            Reply::Json(response_json("r2", "completed")),
        ],
        None,
    )
    .await;
    let model = server.model("gpt-4");
    model.get_response(req("hi")).await.unwrap();
    let mut extra_body = serde_json::Map::new();
    extra_body.insert("extra_transport_field".into(), json!("enabled"));
    let settings = ModelSettings {
        extra_query: Some(HashMap::from([(
            "api-version".to_string(),
            "2026-01-01-preview".to_string(),
        )])),
        extra_body: Some(extra_body),
        ..Default::default()
    };
    model
        .get_response(ModelRequest {
            model_settings: settings,
            ..req("hi")
        })
        .await
        .unwrap();
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].target, "/v1/responses");
    assert_eq!(
        requests[1].target,
        "/v1/responses?api-version=2026-01-01-preview"
    );
    assert_eq!(requests[1].body["extra_transport_field"], "enabled");
}

#[tokio::test]
async fn custom_base_url_prompt_cache_key_uses_model_settings_only() {
    // A non-OpenAI endpoint never gets a generated `prompt_cache_key`, but an explicit one in
    // `extra_args` is forwarded.
    let server = mock_server(
        vec![
            Reply::Json(message_response("one")),
            Reply::Json(message_response("two")),
        ],
        None,
    )
    .await;
    assert!(!server.model("gpt-4").supports_default_prompt_cache_key());
    assert!(
        OpenAIResponsesModel::new("gpt-4", OpenAIClient::new("sk-test"))
            .supports_default_prompt_cache_key()
    );
    let agent = Agent::<()>::new("test").with_model_instance(Arc::new(server.model("gpt-4")));
    Runner::run_with_options(&agent, "hi", quiet_options())
        .await
        .unwrap();
    let mut extra_args = serde_json::Map::new();
    extra_args.insert("prompt_cache_key".into(), json!("cache-key"));
    let explicit = Agent::<()>::new("test")
        .with_model_instance(Arc::new(server.model("gpt-4")))
        .with_model_settings(ModelSettings {
            extra_args: Some(extra_args),
            ..Default::default()
        });
    Runner::run_with_options(&explicit, "hi", quiet_options())
        .await
        .unwrap();
    let requests = server.requests();
    assert!(requests[0].body.get("prompt_cache_key").is_none());
    assert_eq!(requests[1].body["prompt_cache_key"], "cache-key");
}

// ------------------------------------------------------------- response surface

#[tokio::test]
async fn get_response_exposes_the_request_id() {
    let mut payload = response_json("resp-request-id", "completed");
    payload["usage"] = json!({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
        "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}});
    let server = mock_server(vec![Reply::Json(payload)], Some("req_nonstream_123")).await;
    let request = ModelRequest {
        model_settings: ModelSettings {
            preserve_raw_usage: Some(true),
            ..Default::default()
        },
        ..req("hi")
    };
    let response = server.model("gpt-4").get_response(request).await.unwrap();
    assert_eq!(response.response_id.as_deref(), Some("resp-request-id"));
    assert_eq!(response.request_id.as_deref(), Some("req_nonstream_123"));
    let raw = response.raw_usage.expect("raw usage");
    assert_eq!(raw["input_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn get_response_rejects_failed_terminal_response_status() {
    for status in ["incomplete", "failed"] {
        let server = mock_server(
            vec![Reply::Json(response_json("resp-terminal", status))],
            None,
        )
        .await;
        let error = server
            .model("gpt-4")
            .get_response(req("hi"))
            .await
            .unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert!(
            error.to_string().contains(&format!("response.{status}")),
            "{error}"
        );
    }
}

#[tokio::test]
async fn stream_attaches_the_request_id_to_the_terminal_response() {
    let server = mock_server(
        vec![Reply::Sse(vec![completed_event(response_json(
            "resp-stream-request-id",
            "completed",
        ))])],
        Some("req_stream_123"),
    )
    .await;
    let (events, error) = drain(&server.model("gpt-4"), req("hi")).await;
    assert!(error.is_none());
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["response"]["_request_id"], "req_stream_123");
}

#[tokio::test]
async fn stream_without_a_request_id_still_returns_events() {
    let server = mock_server(
        vec![Reply::Sse(vec![completed_event(response_json(
            "resp-stream-request-id",
            "completed",
        ))])],
        None,
    )
    .await;
    let (events, error) = drain(&server.model("gpt-4"), req("hi")).await;
    assert!(error.is_none());
    assert_eq!(events.len(), 1);
    assert!(events[0]["response"].get("_request_id").is_none());
}

#[tokio::test]
async fn stream_http_error_surfaces_instead_of_events() {
    // Python's `parse()` failure path: the failure reaches the caller and no events are made up.
    let server = mock_server(
        vec![Reply::Status(
            500,
            json!({"error": {"message": "boom", "code": "server_error"}}),
            vec![],
        )],
        None,
    )
    .await;
    let result = server.model("gpt-4").stream_response(req("hi")).await;
    assert!(result.is_err());
}

// ------------------------------------------------- terminal events over the stream

#[tokio::test]
async fn stream_response_rejects_response_error_terminal_event() {
    // The model yields the `error` frame and then raises ModelBehaviorError, as Python does.
    let event = json!({"type": "error", "sequence_number": 0, "code": "invalid_request_error",
        "message": "bad request", "param": null});
    let server = mock_server(vec![Reply::Sse(vec![event])], None).await;
    let (events, error) = drain(&server.model("gpt-4"), req("hi")).await;
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["type"], "error");
    let error = error.expect("terminal failure");
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains("bad request"), "{error}");
}

#[tokio::test]
async fn terminal_failures_survive_a_transport_that_fails_afterwards() {
    for event_type in [
        Some("error"),
        Some("response.failed"),
        Some("response.incomplete"),
        None,
    ] {
        let events = match event_type {
            Some("error") => vec![error_event()],
            Some(t) => vec![terminal_event(t, t.trim_start_matches("response."))],
            None => vec![],
        };
        let server = mock_server(vec![Reply::SseThenReset(events)], None).await;
        let (seen, error) = drain(&server.model("gpt-4"), req("hi")).await;
        let error = error.unwrap_or_else(|| panic!("{event_type:?}: the stream must fail"));
        match event_type {
            Some("error") => {
                // The frame is yielded, then raised as ModelBehaviorError; the transport reset
                // that follows is never seen.
                assert!(
                    matches!(error, AgentsError::ModelBehaviorError { .. }),
                    "{error:?}"
                );
            }
            Some(t) => {
                assert!(
                    matches!(error, AgentsError::ModelBehaviorError { .. }),
                    "{error:?}"
                );
                assert!(
                    error
                        .to_string()
                        .contains(&format!("status={}", t.trim_start_matches("response."))),
                    "{error}"
                );
            }
            // No terminal event: the transport error itself is what the caller sees.
            None => assert!(
                !matches!(error, AgentsError::ModelBehaviorError { .. }),
                "{error:?}"
            ),
        }
        let want: Vec<&str> = match event_type {
            None => vec![],
            Some(_) => event_type.into_iter().collect(),
        };
        let got: Vec<&str> = seen.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(got, want, "{event_type:?}");
    }
}

#[tokio::test]
async fn transport_failure_after_a_terminal_event_is_ignored() {
    // `response.completed` is yielded, then the connection dies: a consumer that drains the
    // stream sees the event and a clean end, never the transport failure.
    let server = mock_server(
        vec![Reply::SseThenReset(vec![completed_event(response_json(
            "resp-1",
            "completed",
        ))])],
        Some("req_stream_123"),
    )
    .await;
    let (events, error) = drain(&server.model("gpt-4"), req("hi")).await;
    assert!(error.is_none(), "{error:?}");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["type"], "response.completed");
}

// --------------------------------------------------------------------------- spans

static TRACING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn response_spans(processor: &InMemoryTraceProcessor) -> Vec<Value> {
    processor
        .items()
        .into_iter()
        .filter(|i| i["object"] == "trace.span" && i["span_data"]["type"] == "response")
        .collect()
}

fn enabled_request(input: &str) -> ModelRequest {
    ModelRequest {
        tracing: ModelTracing::Enabled,
        ..req(input)
    }
}

#[tokio::test]
async fn get_response_span_exports_usage() {
    let _guard = TRACING.lock().await;
    set_tracing_disabled(false);
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    let mut payload = response_json("resp-usage", "completed");
    payload["usage"] = json!({"input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
        "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}});
    let server = mock_server(vec![Reply::Json(payload)], None).await;
    let model = server.model("gpt-4");
    trace("test")
        .run(async {
            model.get_response(enabled_request("hi")).await.unwrap();
        })
        .await;
    let spans = response_spans(&processor);
    assert_eq!(spans.len(), 1);
    let data = &spans[0]["span_data"];
    assert_eq!(data["response_id"], "resp-usage");
    assert_eq!(data["usage"]["requests"], 1);
    assert_eq!(data["usage"]["input_tokens"], 10);
    assert_eq!(data["usage"]["output_tokens"], 4);
    assert_eq!(data["usage"]["total_tokens"], 14);
    assert_eq!(data["usage"]["input_tokens_details"]["cached_tokens"], 0);
    assert_eq!(
        data["usage"]["output_tokens_details"]["reasoning_tokens"],
        0
    );
}

#[tokio::test]
async fn response_span_counts_a_request_without_usage() {
    let _guard = TRACING.lock().await;
    set_tracing_disabled(false);
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    let server = mock_server(
        vec![Reply::Json(response_json("resp-no-usage", "completed"))],
        None,
    )
    .await;
    let model = server.model("gpt-4");
    trace("test")
        .run(async {
            model.get_response(enabled_request("hi")).await.unwrap();
        })
        .await;
    let spans = response_spans(&processor);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0]["span_data"]["usage"]["requests"], 1);
    assert_eq!(spans[0]["span_data"]["usage"]["total_tokens"], 0);
}

#[tokio::test]
async fn streamed_span_records_usage_before_a_consumer_closes_at_the_terminal_event() {
    let _guard = TRACING.lock().await;
    set_tracing_disabled(false);
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    let mut payload = response_json("resp-stream-early-close", "completed");
    payload["usage"] = json!({"input_tokens": 10, "output_tokens": 4, "total_tokens": 14});
    let server = mock_server(vec![Reply::Sse(vec![completed_event(payload)])], None).await;
    let model = server.model("gpt-4");
    trace("test")
        .run(async {
            let mut stream = model
                .stream_response(enabled_request("the user prompt"))
                .await
                .unwrap();
            while let Some(event) = stream.next().await {
                if event.unwrap()["type"] == "response.completed" {
                    break;
                }
            }
            drop(stream);
        })
        .await;
    let spans = response_spans(&processor);
    assert_eq!(spans.len(), 1);
    let data = &spans[0]["span_data"];
    assert_eq!(data["response_id"], "resp-stream-early-close");
    assert_eq!(data["usage"]["total_tokens"], 14);
}

#[tokio::test]
async fn streamed_span_counts_a_request_without_usage_at_the_terminal_event() {
    let _guard = TRACING.lock().await;
    set_tracing_disabled(false);
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    let server = mock_server(
        vec![Reply::Sse(vec![completed_event(response_json(
            "resp-no-usage",
            "completed",
        ))])],
        Some("req-1"),
    )
    .await;
    let model = server.model("gpt-4");
    trace("test")
        .run(async {
            let mut stream = model.stream_response(enabled_request("hi")).await.unwrap();
            while let Some(event) = stream.next().await {
                if event.unwrap()["type"] == "response.completed" {
                    break;
                }
            }
            drop(stream);
        })
        .await;
    let spans = response_spans(&processor);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0]["span_data"]["usage"]["requests"], 1);
    assert_eq!(spans[0]["span_data"]["usage"]["total_tokens"], 0);
}

// ----------------------------------------------------------------- retry advice

fn api_error(status: u16, headers: &[(&str, &str)], request_id: Option<&str>) -> AgentsError {
    AgentsError::ApiError {
        status,
        message: "rate limited".into(),
        body: Some(Box::new(json!({"error": {"code": "rate_limit"}}))),
        request_id: request_id.map(str::to_string),
        headers: if headers.is_empty() {
            None
        } else {
            Some(
                headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            )
        },
    }
}

fn model_advice(
    error: &AgentsError,
    previous: Option<&str>,
) -> openai_agents_sdk::retry::ModelRetryAdvice {
    OpenAIResponsesModel::new("gpt-4", OpenAIClient::new("sk-test"))
        .get_retry_advice(&ModelRetryAdviceRequest {
            error,
            attempt: 1,
            stream: false,
            previous_response_id: previous,
            conversation_id: None,
        })
        .expect("advice")
}

#[test]
fn retry_advice_uses_openai_headers() {
    let error = api_error(
        429,
        &[("x-should-retry", "true"), ("retry-after-ms", "250")],
        Some("req_456"),
    );
    let advice = model_advice(&error, None);
    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.retry_after, Some(0.25));
    assert_eq!(advice.replay_safety, Some(ReplaySafety::Safe));
    let normalized = advice.normalized.expect("normalized");
    assert_eq!(normalized.error_code.as_deref(), Some("rate_limit"));
    assert_eq!(normalized.status_code, Some(429));
    assert_eq!(normalized.request_id.as_deref(), Some("req_456"));
}

#[test]
fn retry_advice_marks_stateful_http_failures_replay_safe() {
    let advice = model_advice(&api_error(429, &[], None), Some("resp_prev"));
    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.replay_safety, Some(ReplaySafety::Safe));
    assert_eq!(
        advice.normalized.expect("normalized").status_code,
        Some(429)
    );
}

/// A genuine connection failure from the HTTP stack (nothing listens on the port).
async fn connection_error() -> AgentsError {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    OpenAIResponsesModel::new(
        "gpt-4",
        OpenAIClient::new("sk-test").with_base_url(format!("http://{addr}/v1")),
    )
    .get_response(req("hi"))
    .await
    .unwrap_err()
}

#[tokio::test]
async fn retry_advice_keeps_transport_failures_retryable_and_ambiguous() {
    let error = connection_error().await;
    for previous in [None, Some("resp_prev")] {
        let advice = model_advice(&error, previous);
        assert_eq!(advice.suggested, Some(true), "{previous:?}");
        assert_eq!(advice.replay_safety, None, "{previous:?}");
        assert!(
            advice.normalized.expect("normalized").is_network_error,
            "{previous:?}"
        );
    }
}
