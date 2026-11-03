//! Parity port of the `LitellmModel` test files under `tests/models/` in the Python SDK:
//! `test_litellm_extra_body.py`, `test_litellm_logprobs.py`, `test_litellm_usage_requests.py`,
//! `test_litellm_user_agent.py`, `test_litellm_content_filter.py` and the LiteLLM half of
//! `test_kwargs_functionality.py`.
//!
//! Python patches `litellm.acompletion` and reads the keyword arguments it received. Rust posts
//! straight to the provider's OpenAI-compatible Chat Completions endpoint (see the module doc of
//! `src/extensions/models/mod.rs`), so the same expectations are asserted against the captured
//! HTTP request body of a local mock server.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::extensions::models::LitellmModel;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Local mock Chat Completions server
// ---------------------------------------------------------------------------

/// One captured request.
#[derive(Clone, Debug)]
struct Captured {
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
                let headers: Vec<(String, String)> = text[..idx]
                    .lines()
                    .skip(1)
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
                break Captured {
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

/// The mock stands in for `litellm.acompletion` returning `Message(role, content)`.
fn ok_reply(text: &str) -> Value {
    json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })
}

fn request_with(settings: ModelSettings) -> ModelRequest {
    ModelRequest {
        input: vec![InputItem(json!({"role": "user", "content": "test input"}))],
        model_settings: settings,
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

/// Add the function tool or handoff a parallel-tool-calls case calls for.
fn add_tools(request: &mut ModelRequest, source: &str) {
    use openai_agents_sdk::handoffs::handoff;
    use openai_agents_sdk::models::interface::ToolSpec;
    use openai_agents_sdk::tool::FunctionTool;

    match source {
        "function" => {
            let tool = FunctionTool::<()>::new("test_tool", "", |_c, _a: NoArgs| async move {
                Ok("ok".to_string())
            });
            request.tools.push(ToolSpec::Function(tool.to_spec()));
        }
        "handoff" => request
            .handoffs
            .push(handoff::<()>(Agent::<()>::new("handoff")).to_spec()),
        _ => {}
    }
}

fn model(base_url: &str) -> LitellmModel {
    LitellmModel::with_options("test-model", Some(base_url), Some("sk-test"), None).unwrap()
}

fn map(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

/// Drive one request through `LitellmModel` and return the body the provider received.
async fn capture_body(settings: ModelSettings) -> Value {
    let (url, captured) = mock_server(vec![ok_reply("ok")]).await;
    model(&url)
        .get_response(request_with(settings))
        .await
        .unwrap();
    let guard = captured.lock().unwrap();
    guard[0].json()
}

// ---------------------------------------------------------------------------
// test_litellm_extra_body.py
// ---------------------------------------------------------------------------

/// Provider-specific request fields reach the upstream request body.
///
/// Python keeps them nested under LiteLLM's `extra_body` kwarg so the library merges them into
/// the body itself; here the body *is* the request, so the same fields appear at its top level.
#[tokio::test]
async fn extra_body_is_forwarded() {
    let settings = ModelSettings {
        temperature: Some(0.1),
        extra_body: Some(map(&[
            ("cached_content", json!("some_cache")),
            ("foo", json!(123)),
        ])),
        ..Default::default()
    };

    let body = capture_body(settings).await;

    assert_eq!(body["cached_content"], "some_cache");
    assert_eq!(body["foo"], 123);
    assert_eq!(body["temperature"], 0.1);
}

#[tokio::test]
async fn extra_body_reasoning_effort_is_promoted() {
    let extra_body = map(&[
        ("reasoning_effort", json!("none")),
        ("cached_content", json!("some_cache")),
    ]);
    let settings = ModelSettings {
        extra_body: Some(extra_body.clone()),
        ..Default::default()
    };

    let body = capture_body(settings).await;

    assert_eq!(body["reasoning_effort"], "none");
    assert_eq!(body["cached_content"], "some_cache");
    // The caller's settings are untouched.
    assert_eq!(
        extra_body,
        map(&[
            ("reasoning_effort", json!("none")),
            ("cached_content", json!("some_cache")),
        ])
    );
}

/// An explicit `ModelSettings.reasoning` wins over an `extra_body` entry.
#[tokio::test]
async fn reasoning_effort_prefers_model_settings() {
    let extra_body = map(&[("reasoning_effort", json!("high"))]);
    let settings = ModelSettings {
        reasoning: Some(Reasoning::effort("low")),
        extra_body: Some(extra_body.clone()),
        ..Default::default()
    };

    let body = capture_body(settings).await;

    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(extra_body, map(&[("reasoning_effort", json!("high"))]));
}

#[tokio::test]
async fn extra_body_reasoning_effort_overrides_extra_args() {
    let extra_args = map(&[
        ("reasoning_effort", json!("low")),
        ("custom_param", json!("custom")),
    ]);
    let settings = ModelSettings {
        extra_body: Some(map(&[("reasoning_effort", json!("none"))])),
        extra_args: Some(extra_args.clone()),
        ..Default::default()
    };

    let body = capture_body(settings).await;

    assert_eq!(body["reasoning_effort"], "none");
    assert_eq!(body["custom_param"], "custom");
    assert_eq!(
        extra_args,
        map(&[
            ("reasoning_effort", json!("low")),
            ("custom_param", json!("custom")),
        ])
    );
}

/// `reasoning.summary` does not change the chat-completions argument shape: a scalar
/// `reasoning_effort` is still sent, and exactly one warning says the summary was dropped.
#[tokio::test]
async fn reasoning_summary_uses_scalar_effort_and_warns() {
    const NEEDLE: &str = "does not forward Reasoning.summary";

    #[derive(Default)]
    struct Collect(Mutex<Vec<String>>);
    impl log::Log for Collect {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            let line = record.args().to_string();
            if line.contains(NEEDLE) {
                self.0.lock().unwrap().push(line);
            }
        }
        fn flush(&self) {}
    }
    static SINK: OnceLock<Collect> = OnceLock::new();
    let sink: &'static Collect = SINK.get_or_init(Collect::default);
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);

    for name in [
        "openai/gpt-5-mini",
        "anthropic/claude-sonnet-4-5",
        "gemini/gemini-2.5-pro",
    ] {
        sink.0.lock().unwrap().clear();
        let (url, captured) = mock_server(vec![ok_reply("ok")]).await;
        let model = LitellmModel::with_options(name, Some(&url), Some("sk-test"), None).unwrap();
        let settings = ModelSettings {
            reasoning: Some(Reasoning {
                effort: Some("medium".into()),
                summary: Some("auto".into()),
                ..Default::default()
            }),
            ..Default::default()
        };

        model.get_response(request_with(settings)).await.unwrap();

        assert_eq!(
            captured.lock().unwrap()[0].json()["reasoning_effort"],
            "medium"
        );
        assert_eq!(sink.0.lock().unwrap().len(), 1, "model {name}");
    }
}

// ---------------------------------------------------------------------------
// test_litellm_logprobs.py
// ---------------------------------------------------------------------------

/// The Chat Completions API rejects `top_logprobs` unless `logprobs` is true.
#[tokio::test]
async fn top_logprobs_sets_logprobs_flag() {
    let body = capture_body(ModelSettings {
        top_logprobs: Some(2),
        ..Default::default()
    })
    .await;

    assert_eq!(body["top_logprobs"], 2);
    assert_eq!(body["logprobs"], true);
}

#[tokio::test]
async fn omits_logprobs_when_top_logprobs_unset() {
    let body = capture_body(ModelSettings::default()).await;

    assert!(body.get("logprobs").is_none());
}

/// A caller-supplied `logprobs` must not be duplicated by the `top_logprobs` shorthand.
#[tokio::test]
async fn top_logprobs_with_extra_args_logprobs_does_not_collide() {
    let body = capture_body(ModelSettings {
        top_logprobs: Some(2),
        extra_args: Some(map(&[("logprobs", json!(true))])),
        ..Default::default()
    })
    .await;

    assert_eq!(body["top_logprobs"], 2);
    assert_eq!(body["logprobs"], true);
}

#[tokio::test]
async fn get_response_preserves_returned_logprobs_in_output() {
    let reply = json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello"},
            "logprobs": {"content": [{
                "token": "Hello",
                "logprob": -0.25,
                "bytes": [72, 101, 108, 108, 111],
                "top_logprobs": [
                    {"token": "Hello", "logprob": -0.25, "bytes": [72, 101, 108, 108, 111]},
                    {"token": "Hi", "logprob": -1.5, "bytes": [72, 105]}
                ]
            }]}
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    });
    let (url, _captured) = mock_server(vec![reply]).await;

    let response = model(&url)
        .get_response(request_with(ModelSettings {
            top_logprobs: Some(2),
            ..Default::default()
        }))
        .await
        .unwrap();

    let texts: Vec<Value> = response
        .output
        .iter()
        .map(|item| item.to_value())
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().cloned().unwrap_or_default())
        .filter(|part| part["type"] == "output_text")
        .collect();
    assert!(!texts.is_empty(), "expected an output_text in the output");
    let logprobs = texts[0]["logprobs"].as_array().expect("logprobs");
    assert_eq!(logprobs.len(), 1);
    assert_eq!(logprobs[0]["token"], "Hello");
    assert_eq!(logprobs[0]["logprob"], -0.25);
    let tops: Vec<&str> = logprobs[0]["top_logprobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["token"].as_str().unwrap())
        .collect();
    assert_eq!(tops, ["Hello", "Hi"]);
}

// ---------------------------------------------------------------------------
// test_litellm_usage_requests.py
// ---------------------------------------------------------------------------

/// The call happened, so it counts, even though no token counts came back.
#[tokio::test]
async fn request_is_counted_when_the_provider_reports_no_usage() {
    let reply = json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}]
    });
    let (url, _captured) = mock_server(vec![reply]).await;

    let response = model(&url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    assert_eq!(response.usage.requests, 1);
    assert_eq!(response.usage.input_tokens, 0);
    assert_eq!(response.usage.output_tokens, 0);
    assert_eq!(response.usage.total_tokens, 0);
}

/// Null token counts from a provider are normalized to zero without losing the details or the
/// per-request entry.
#[tokio::test]
async fn runner_normalizes_nullable_provider_usage() {
    let cases: [([Option<u64>; 3], [u64; 3]); 6] = [
        ([None, Some(5), Some(12)], [0, 5, 12]),
        ([Some(7), None, Some(12)], [7, 0, 12]),
        ([Some(7), Some(5), None], [7, 5, 0]),
        ([None, None, None], [0, 0, 0]),
        ([Some(0), Some(0), Some(0)], [0, 0, 0]),
        ([Some(7), Some(5), Some(12)], [7, 5, 12]),
    ];

    for (counts, expected) in cases {
        let to_json = |v: Option<u64>| v.map(Value::from).unwrap_or(Value::Null);
        let reply = json!({
            "id": "chatcmpl_1",
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": to_json(counts[0]),
                "completion_tokens": to_json(counts[1]),
                "total_tokens": to_json(counts[2]),
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        });
        let (url, _captured) = mock_server(vec![reply]).await;
        let agent = Agent::<()>::new("test").with_model_instance(Arc::new(model(&url)));

        let result = Runner::run(&agent, "hi").await.unwrap();

        assert_eq!(result.final_output_text().unwrap_or_default(), "ok");
        let usage = result.usage();
        assert_eq!(usage.requests, 1, "counts {counts:?}");
        assert_eq!(
            [usage.input_tokens, usage.output_tokens, usage.total_tokens],
            expected,
            "counts {counts:?}"
        );
        assert_eq!(usage.input_tokens_details.cached_tokens, 3);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 2);
        if expected == [0, 0, 0] {
            assert!(usage.request_usage_entries.is_empty(), "counts {counts:?}");
        } else {
            assert_eq!(usage.request_usage_entries.len(), 1);
            let entry = &usage.request_usage_entries[0];
            assert_eq!(
                [entry.input_tokens, entry.output_tokens, entry.total_tokens],
                expected
            );
            assert_eq!(entry.input_tokens_details.cached_tokens, 3);
            assert_eq!(entry.output_tokens_details.reasoning_tokens, 2);
        }
    }
}

// ---------------------------------------------------------------------------
// test_litellm_user_agent.py
// ---------------------------------------------------------------------------

/// Every request carries the SDK's own user agent. Python additionally supports a per-call
/// override through the `HEADERS_OVERRIDE` contextvar, which has no Rust counterpart.
#[tokio::test]
async fn user_agent_header_is_sent() {
    let (url, captured) = mock_server(vec![ok_reply("Hello")]).await;

    model(&url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();

    let expected = format!("Agents/Rust {}", env!("CARGO_PKG_VERSION"));
    assert_eq!(
        captured.lock().unwrap()[0].header("user-agent"),
        Some(expected.as_str())
    );
}

// ---------------------------------------------------------------------------
// test_kwargs_functionality.py (the LiteLLM half)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn litellm_kwargs_forwarded() {
    let body = capture_body(ModelSettings {
        temperature: Some(0.5),
        extra_args: Some(map(&[
            ("custom_param", json!("custom_value")),
            ("seed", json!(42)),
            ("stop", json!(["END"])),
            ("logit_bias", json!({"123": -100})),
        ])),
        ..Default::default()
    })
    .await;

    assert_eq!(body["custom_param"], "custom_value");
    assert_eq!(body["seed"], 42);
    assert_eq!(body["stop"], json!(["END"]));
    assert_eq!(body["logit_bias"], json!({"123": -100}));
    assert_eq!(body["temperature"], 0.5);
}

/// `parallel_tool_calls` only travels with converted tools; Chat Completions rejects it alone.
#[tokio::test]
async fn litellm_only_forwards_parallel_tool_calls_with_converted_tools() {
    for parallel in [true, false] {
        for source in ["none", "function", "handoff"] {
            let mut request = request_with(ModelSettings {
                parallel_tool_calls: Some(parallel),
                ..Default::default()
            });
            add_tools(&mut request, source);

            let (url, captured) = mock_server(vec![ok_reply("test response")]).await;
            model(&url).get_response(request).await.unwrap();
            let body = captured.lock().unwrap()[0].json();

            if source == "none" {
                assert!(body.get("parallel_tool_calls").is_none(), "{source}");
                assert!(body.get("tools").is_none(), "{source}");
            } else {
                assert_eq!(body["parallel_tool_calls"], parallel, "{source}");
                assert!(
                    body["tools"].as_array().is_some_and(|t| !t.is_empty()),
                    "{source}"
                );
            }
        }
    }
}

#[tokio::test]
async fn litellm_normalizes_agent_model_settings() {
    let body = capture_body(ModelSettings {
        temperature: Some(0.0),
        reasoning: Some(Reasoning::effort("low")),
        ..Default::default()
    })
    .await;

    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["reasoning_effort"], "low");
}

/// Absent and empty `extra_args` are both handled without error.
#[tokio::test]
async fn empty_kwargs_handling() {
    let body = capture_body(ModelSettings {
        temperature: Some(0.5),
        extra_args: None,
        ..Default::default()
    })
    .await;
    assert_eq!(body["temperature"], 0.5);

    let body = capture_body(ModelSettings {
        temperature: Some(0.3),
        extra_args: Some(Map::new()),
        ..Default::default()
    })
    .await;
    assert_eq!(body["temperature"], 0.3);
}

/// `reasoning_effort` from `extra_args` is promoted when no reasoning settings are present.
#[tokio::test]
async fn reasoning_effort_falls_back_to_extra_args() {
    let extra_args = map(&[
        ("reasoning_effort", json!("none")),
        ("custom_param", json!("custom_value")),
    ]);
    let settings = ModelSettings {
        extra_args: Some(extra_args.clone()),
        ..Default::default()
    };

    let body = capture_body(settings).await;

    assert_eq!(body["reasoning_effort"], "none");
    assert_eq!(body["custom_param"], "custom_value");
    assert_eq!(
        extra_args,
        map(&[
            ("reasoning_effort", json!("none")),
            ("custom_param", json!("custom_value")),
        ])
    );
}

// ---------------------------------------------------------------------------
// test_litellm_content_filter.py
// ---------------------------------------------------------------------------

async fn content_filter_output(finish_reason: &str, content: &str) -> Vec<Value> {
    let reply = json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    });
    let (url, _captured) = mock_server(vec![reply]).await;
    let response = model(&url)
        .get_response(request_with(ModelSettings::default()))
        .await
        .unwrap();
    response
        .output
        .iter()
        .map(|item| item.to_value())
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().cloned().unwrap_or_default())
        .filter(|part| part["type"] == "refusal")
        .collect()
}

/// A safety block signalled only by `finish_reason == "content_filter"` with an empty message
/// must become an explicit refusal rather than zero output items; otherwise the turn is
/// indistinguishable from an empty response and drives agent loops into fruitless retries.
#[tokio::test]
async fn content_filter_finish_reason_surfaces_refusal() {
    let refusals = content_filter_output("content_filter", "").await;

    assert!(!refusals.is_empty(), "expected a refusal item");
    assert!(
        refusals[0]["refusal"]
            .as_str()
            .is_some_and(|r| !r.is_empty())
    );
}

/// A `content_filter` finish reason that still carries text is left alone.
#[tokio::test]
async fn content_filter_does_not_clobber_real_content() {
    let refusals = content_filter_output("content_filter", "here is the answer").await;

    assert!(
        refusals.is_empty(),
        "should not synthesize a refusal when content is present"
    );
}

/// A normal completion is unchanged — no spurious refusal.
#[tokio::test]
async fn normal_stop_is_unaffected() {
    let refusals = content_filter_output("stop", "all good").await;

    assert!(refusals.is_empty());
}

// ---------------------------------------------------------------------------
// test_kwargs_functionality.py (the OpenAI Chat Completions half and retry advice)
// ---------------------------------------------------------------------------

fn chat_model(base_url: &str) -> openai_agents_sdk::OpenAIChatCompletionsModel {
    openai_agents_sdk::OpenAIChatCompletionsModel::new(
        "gpt-4",
        openai_agents_sdk::OpenAIClient::new("sk-test").with_base_url(base_url),
    )
}

#[tokio::test]
async fn openai_only_forwards_parallel_tool_calls_with_converted_tools() {
    for parallel in [true, false] {
        for source in ["none", "function", "handoff"] {
            let mut request = request_with(ModelSettings {
                parallel_tool_calls: Some(parallel),
                ..Default::default()
            });
            add_tools(&mut request, source);

            let (url, captured) = mock_server(vec![ok_reply("test response")]).await;
            chat_model(&url).get_response(request).await.unwrap();
            let body = captured.lock().unwrap()[0].json();

            if source == "none" {
                assert!(body.get("parallel_tool_calls").is_none(), "{source}");
                assert!(body.get("tools").is_none(), "{source}");
            } else {
                assert_eq!(body["parallel_tool_calls"], parallel, "{source}");
                assert!(body["tools"].as_array().is_some_and(|t| !t.is_empty()));
            }
        }
    }
}

#[tokio::test]
async fn openai_chatcompletions_kwargs_forwarded() {
    let (url, captured) = mock_server(vec![ok_reply("test response")]).await;
    let mut request = request_with(ModelSettings {
        temperature: Some(0.7),
        extra_args: Some(map(&[
            ("seed", json!(123)),
            ("logit_bias", json!({"456": 10})),
            ("stop", json!(["STOP", "END"])),
            ("user", json!("test-user")),
        ])),
        ..Default::default()
    });
    request.system_instructions = Some("Test system".into());

    chat_model(&url).get_response(request).await.unwrap();

    let body = captured.lock().unwrap()[0].json();
    assert_eq!(body["seed"], 123);
    assert_eq!(body["logit_bias"], json!({"456": 10}));
    assert_eq!(body["stop"], json!(["STOP", "END"]));
    assert_eq!(body["user"], "test-user");
    assert_eq!(body["temperature"], 0.7);
}

/// A 429 that carries OpenAI-style retry headers yields retry advice with the header's delay.
#[test]
fn litellm_get_retry_advice_uses_response_headers() {
    use openai_agents_sdk::error::AgentsError;
    use openai_agents_sdk::retry::ModelRetryAdviceRequest;

    let error = AgentsError::ApiError {
        status: 429,
        message: "rate limited".into(),
        body: None,
        request_id: None,
        headers: Some(
            [
                ("x-should-retry".to_string(), "true".to_string()),
                ("retry-after-ms".to_string(), "250".to_string()),
            ]
            .into_iter()
            .collect(),
        ),
    };
    let model =
        LitellmModel::with_options("test-model", Some("http://127.0.0.1:1"), None, None).unwrap();

    let advice = model
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: None,
            conversation_id: None,
        })
        .expect("advice");

    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.retry_after, Some(0.25));
}

/// A connection failure on a stateful request is worth retrying, but whether the provider
/// already acted on it is unknown, so no replay-safety verdict is given.
#[tokio::test]
async fn litellm_get_retry_advice_keeps_stateful_transport_failures_ambiguous() {
    use openai_agents_sdk::error::AgentsError;
    use openai_agents_sdk::retry::ModelRetryAdviceRequest;

    // A port nothing listens on: the connection is refused locally.
    let closed = {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        listener.local_addr().unwrap()
    };
    let transport = reqwest::Client::new()
        .post(format!("http://{closed}/responses"))
        .send()
        .await
        .unwrap_err();
    let error = AgentsError::Http(transport);
    let model =
        LitellmModel::with_options("test-model", Some("http://127.0.0.1:1"), None, None).unwrap();

    let advice = model
        .get_retry_advice(&ModelRetryAdviceRequest {
            error: &error,
            attempt: 1,
            stream: false,
            previous_response_id: Some("resp_prev"),
            conversation_id: None,
        })
        .expect("advice");

    assert_eq!(advice.suggested, Some(true));
    assert_eq!(advice.replay_safety, None);
}
