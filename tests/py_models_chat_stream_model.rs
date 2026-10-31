//! Port of the model-level tests in `tests/models/test_openai_chatcompletions_stream.py`:
//! `OpenAIChatCompletionsModel::stream_response` against a local SSE mock server.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{FunctionToolSpec, Model, ModelRequest, ToolSpec};
use openai_agents_sdk::models::{ModelTracing, OpenAIChatCompletionsModel, OpenAIClient};
use openai_agents_sdk::{AgentsError, Prompt};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---- local mock server -----------------------------------------------------------------

type Captured = Arc<Mutex<Vec<Value>>>;

/// Serve one SSE body per request (the last repeats); `request_id` is sent as `x-request-id`.
async fn serve_sse(bodies: Vec<Vec<Value>>, request_id: Option<&str>) -> (String, Captured) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let queue = Arc::new(Mutex::new(VecDeque::from(bodies)));
    let request_id = request_id.map(str::to_string);
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let cap = cap.clone();
            let queue = queue.clone();
            let request_id = request_id.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                let body = loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(idx) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let headers: HashMap<String, String> = text[..idx]
                        .lines()
                        .filter_map(|l| l.split_once(':'))
                        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
                        .collect();
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                    }
                };
                cap.lock().unwrap().push(body);
                let chunks = {
                    let mut q = queue.lock().unwrap();
                    if q.len() > 1 {
                        q.pop_front().unwrap()
                    } else {
                        q.front().unwrap().clone()
                    }
                };
                let mut payload = String::new();
                for c in &chunks {
                    payload.push_str(&format!("data: {c}\n\n"));
                }
                payload.push_str("data: [DONE]\n\n");
                let mut head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n",
                    payload.len()
                );
                if let Some(id) = request_id {
                    head.push_str(&format!("x-request-id: {id}\r\n"));
                }
                head.push_str("\r\n");
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(payload.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/v1"), captured)
}

/// Model configuration under test; the mock server's URL is filled in per run.
#[derive(Clone)]
struct Cfg {
    model: &'static str,
    buffered: bool,
    strict: bool,
}

impl Cfg {
    fn with_strict_feature_validation(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    fn build(&self, base: &str) -> OpenAIChatCompletionsModel {
        OpenAIChatCompletionsModel::new(
            self.model,
            OpenAIClient::new("sk-test").with_base_url(base),
        )
        .with_buffer_streamed_tool_calls(self.buffered)
        .with_strict_feature_validation(self.strict)
    }
}

fn model_at(base: &str) -> OpenAIChatCompletionsModel {
    plain().build(base)
}

fn request() -> ModelRequest {
    ModelRequest {
        tracing: ModelTracing::Disabled,
        ..Default::default()
    }
}

/// Stream `chunks` through the configured model; the error, if any, comes back after the
/// events before it.
async fn run(cfg: &Cfg, chunks: Vec<Value>) -> (Vec<Value>, Option<AgentsError>) {
    run_request(cfg, chunks, request()).await
}

async fn run_request(
    cfg: &Cfg,
    chunks: Vec<Value>,
    req: ModelRequest,
) -> (Vec<Value>, Option<AgentsError>) {
    let (base, _) = serve_sse(vec![chunks], None).await;
    collect(&cfg.build(&base), req).await
}

async fn collect(
    model: &OpenAIChatCompletionsModel,
    req: ModelRequest,
) -> (Vec<Value>, Option<AgentsError>) {
    let mut stream = match model.stream_response(req).await {
        Ok(s) => s,
        Err(e) => return (vec![], Some(e)),
    };
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(e) => events.push(e),
            Err(e) => return (events, Some(e)),
        }
    }
    (events, None)
}

async fn events_of(cfg: &Cfg, chunks: Vec<Value>) -> Vec<Value> {
    let (events, err) = run(cfg, chunks).await;
    assert!(err.is_none(), "unexpected stream error: {err:?}");
    events
}

fn plain() -> Cfg {
    Cfg {
        model: "gpt-4",
        buffered: false,
        strict: false,
    }
}

fn buffered() -> Cfg {
    Cfg {
        buffered: true,
        ..plain()
    }
}

fn strict() -> Cfg {
    Cfg {
        strict: true,
        ..plain()
    }
}

// ---- chunk helpers --------------------------------------------------------------------------

fn chunk_with(choices: Vec<Value>, usage: Option<Value>) -> Value {
    let mut c = json!({"id": "chunk-id", "created": 1, "model": "fake",
                       "object": "chat.completion.chunk", "choices": choices});
    if let Some(u) = usage {
        c["usage"] = u;
    }
    c
}

fn chunk(delta: Value) -> Value {
    chunk_with(vec![json!({"index": 0, "delta": delta})], None)
}

fn chunk_usage(delta: Value, usage: Value) -> Value {
    chunk_with(vec![json!({"index": 0, "delta": delta})], Some(usage))
}

fn finish(reason: &str) -> Value {
    chunk_with(
        vec![json!({"index": 0, "delta": {}, "finish_reason": reason})],
        None,
    )
}

fn usage(completion: u64, prompt: u64, total: u64) -> Value {
    json!({"completion_tokens": completion, "prompt_tokens": prompt, "total_tokens": total})
}

fn tool_call(index: u64, id: Option<&str>, name: Option<&str>, arguments: &str) -> Value {
    let mut tc = json!({"index": index, "type": "function",
                        "function": {"name": name, "arguments": arguments}});
    if let Some(id) = id {
        tc["id"] = json!(id);
    }
    tc
}

fn types(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

fn of_type<'a>(events: &'a [Value], ty: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["type"] == ty).collect()
}

fn completed(events: &[Value]) -> &Value {
    events
        .iter()
        .find(|e| e["type"] == "response.completed")
        .expect("completed event")
}

fn of_item<'a>(events: &'a [Value], ty: &str, item_type: &str) -> Vec<&'a Value> {
    events
        .iter()
        .filter(|e| e["type"] == ty && e["item"]["type"] == item_type)
        .collect()
}

fn function_done(events: &[Value]) -> Vec<&Value> {
    of_item(events, "response.output_item.done", "function_call")
}

// ---- warning capture -----------------------------------------------------------------------

#[derive(Default)]
struct LogSink(Mutex<Vec<String>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let s = SINK.get_or_init(LogSink::default);
    log::set_logger(s).ok();
    log::set_max_level(log::LevelFilter::Trace);
    s
}

fn log_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

// =========================================================================================
// request construction and events
// =========================================================================================

#[test]
fn stream_response_forwards_agent_model_settings() {
    // Official endpoint: `include_usage=false` is explicit, so it is sent as given.
    let model = OpenAIChatCompletionsModel::new("gpt-5.4-mini", OpenAIClient::new("sk-test"));
    let settings = ModelSettings {
        reasoning: Some(openai_agents_sdk::model_settings::Reasoning::effort("low")),
        prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
        prompt_cache_retention: Some("24h".into()),
        verbosity: Some("low".into()),
        store: Some(false),
        temperature: Some(0.0),
        top_p: Some(1.0),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        max_tokens: Some(64),
        parallel_tool_calls: Some(false),
        include_usage: Some(false),
        ..Default::default()
    };
    let body = model
        .build_request_body(
            &ModelRequest {
                model_settings: settings,
                tools: vec![ToolSpec::Function(FunctionToolSpec {
                    name: "test_tool".into(),
                    description: "t".into(),
                    parameters: json!({"type": "object", "properties": {}}),
                    strict: true,
                    ..Default::default()
                })],
                ..request()
            },
            true,
        )
        .unwrap();
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(
        body["prompt_cache_options"],
        json!({"mode": "explicit", "ttl": "30m"})
    );
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["verbosity"], "low");
    assert_eq!(body["store"], false);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["top_p"], 1.0);
    assert_eq!(body["frequency_penalty"], 0.0);
    assert_eq!(body["presence_penalty"], 0.0);
    assert_eq!(body["max_tokens"], 64);
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"], json!({"include_usage": false}));
}

#[tokio::test]
async fn stream_response_yields_events_for_text_content() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"content": "He"})),
            chunk_usage(
                json!({"content": "llo"}),
                json!({"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12,
                       "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 4},
                       "completion_tokens_details": {"reasoning_tokens": 3}}),
            ),
            finish("stop"),
        ],
    )
    .await;
    assert_eq!(
        types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(events[3]["delta"], "He");
    assert_eq!(events[4]["delta"], "llo");
    let done = completed(&events);
    assert_eq!(done["response"]["output"][0]["type"], "message");
    assert_eq!(done["response"]["output"][0]["content"][0]["text"], "Hello");
    let u = &done["response"]["usage"];
    assert_eq!(u["input_tokens"], 7);
    assert_eq!(u["output_tokens"], 5);
    assert_eq!(u["total_tokens"], 12);
    assert_eq!(u["input_tokens_details"]["cached_tokens"], 2);
    assert_eq!(u["input_tokens_details"]["cache_write_tokens"], 4);
    assert_eq!(u["output_tokens_details"]["reasoning_tokens"], 3);
}

#[tokio::test]
async fn stream_response_passes_strict_validation_to_stream_handler() {
    let (_, err) = run(
        &strict(),
        vec![chunk_with(
            vec![json!({"index": 1, "delta": {"content": "ignored"}})],
            None,
        )],
    )
    .await;
    let err = err.expect("strict validation error").to_string();
    assert!(err.contains("multiple choices or nonzero"), "{err}");
}

#[tokio::test]
async fn stream_response_warns_and_ignores_server_managed_conversation_state_by_default() {
    let _guard = log_lock().lock().await;
    for (previous, conversation, expected) in [
        (Some("resp_123"), None, "previous_response_id"),
        (None, Some("conv_123"), "conversation_id"),
    ] {
        sink().0.lock().unwrap().clear();
        let (events, err) = run_request(
            &plain(),
            vec![finish("stop")],
            ModelRequest {
                previous_response_id: previous.map(String::from),
                conversation_id: conversation.map(String::from),
                ..request()
            },
        )
        .await;
        assert!(err.is_none());
        assert!(!events.is_empty());
        let text = sink().0.lock().unwrap().join("\n");
        assert!(text.contains(expected), "{text}");
        assert!(
            text.contains("Ignoring unsupported server-managed conversation state"),
            "{text}"
        );
    }
}

#[tokio::test]
async fn stream_response_warns_and_ignores_prompt_by_default() {
    let _guard = log_lock().lock().await;
    sink().0.lock().unwrap().clear();
    let (base, captured) = serve_sse(vec![vec![finish("stop")]], None).await;
    let (_, err) = collect(
        &model_at(&base),
        ModelRequest {
            prompt: Some(Prompt::new("pmpt_123")),
            ..request()
        },
    )
    .await;
    assert!(err.is_none());
    let text = sink().0.lock().unwrap().join("\n");
    assert!(
        text.contains("Reusable prompts are only supported by the Responses API"),
        "{text}"
    );
    assert!(text.contains("Ignoring `prompt`"), "{text}");
    assert!(captured.lock().unwrap()[0].get("prompt").is_none());
}

#[tokio::test]
async fn stream_response_rejects_server_managed_conversation_state_in_strict_mode() {
    for (previous, conversation, expected) in [
        (Some("resp_123"), None, "previous_response_id"),
        (None, Some("conv_123"), "conversation_id"),
    ] {
        let (base, captured) = serve_sse(vec![vec![finish("stop")]], None).await;
        let (_, err) = collect(
            &model_at(&base).with_strict_feature_validation(true),
            ModelRequest {
                previous_response_id: previous.map(String::from),
                conversation_id: conversation.map(String::from),
                ..request()
            },
        )
        .await;
        let err = err.expect("strict error").to_string();
        assert!(err.contains("server-managed conversation state"), "{err}");
        assert!(err.contains(expected), "{err}");
        assert!(captured.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn stream_response_rejects_prompt_in_strict_mode() {
    let (base, captured) = serve_sse(vec![vec![finish("stop")]], None).await;
    let (_, err) = collect(
        &model_at(&base).with_strict_feature_validation(true),
        ModelRequest {
            prompt: Some(Prompt::new("pmpt_123")),
            ..request()
        },
    )
    .await;
    assert!(
        err.expect("strict error")
            .to_string()
            .contains("Reusable prompts")
    );
    assert!(captured.lock().unwrap().is_empty());
}

fn logprob_chunk(delta: &str, token: &str, logprob: f64, usage: Option<Value>) -> Value {
    let mut c = chunk(json!({"content": delta}));
    c["choices"][0]["logprobs"] = json!({"content": [{"token": token, "logprob": logprob,
        "bytes": [1], "top_logprobs": [{"token": token, "logprob": logprob, "bytes": [1]}]}]});
    if let Some(u) = usage {
        c["usage"] = u;
    }
    c
}

#[tokio::test]
async fn stream_response_includes_logprobs() {
    let events = events_of(
        &plain(),
        vec![
            logprob_chunk("Hi", "Hi", -0.5, None),
            logprob_chunk(" there", " there", -0.25, Some(usage(5, 7, 12))),
            finish("stop"),
        ],
    )
    .await;
    let deltas = of_type(&events, "response.output_text.delta");
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0]["logprobs"][0]["token"], "Hi");
    assert_eq!(deltas[1]["logprobs"][0]["token"], " there");
    let part = &completed(&events)["response"]["output"][0]["content"][0];
    assert_eq!(part["text"], "Hi there");
    let tokens: Vec<&str> = part["logprobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["token"].as_str().unwrap())
        .collect();
    assert_eq!(tokens, ["Hi", " there"]);
}

#[tokio::test]
async fn stream_response_accumulates_logprobs_across_many_deltas() {
    let tokens = ["a", "b", "c", "d", "e"];
    let mut chunks: Vec<Value> = tokens
        .iter()
        .map(|t| logprob_chunk(t, t, -0.5, None))
        .collect();
    chunks.push(finish("stop"));
    let events = events_of(&plain(), chunks).await;
    let part = &completed(&events)["response"]["output"][0]["content"][0];
    assert_eq!(part["text"], "abcde");
    let got: Vec<&str> = part["logprobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["token"].as_str().unwrap())
        .collect();
    assert_eq!(got, tokens);
}

#[tokio::test]
async fn stream_response_yields_events_for_refusal_content() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"refusal": "No"})),
            chunk_usage(json!({"refusal": "Thanks"}), usage(2, 2, 4)),
            finish("stop"),
        ],
    )
    .await;
    assert_eq!(
        types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.refusal.delta",
            "response.refusal.delta",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(events[3]["delta"], "No");
    assert_eq!(events[4]["delta"], "Thanks");
    let part = &completed(&events)["response"]["output"][0]["content"][0];
    assert_eq!(part["type"], "refusal");
    assert_eq!(part["refusal"], "NoThanks");
}

#[tokio::test]
async fn stream_response_yields_events_for_tool_call() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"tool_calls": [tool_call(0, Some("tool-id"), Some("my_func"), "arg1")]})),
            chunk_usage(
                json!({"tool_calls": [tool_call(0, Some("tool-id"), None, "arg2")]}),
                usage(1, 1, 2),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(events[1]["type"], "response.output_item.added");
    assert_eq!(events[1]["item"]["type"], "function_call");
    assert_eq!(events[1]["item"]["name"], "my_func");
    assert_eq!(events[1]["item"]["arguments"], "");
    assert_eq!(events[2]["type"], "response.function_call_arguments.delta");
    assert_eq!(events[2]["delta"], "arg1");
    assert_eq!(events[3]["type"], "response.function_call_arguments.delta");
    assert_eq!(events[3]["delta"], "arg2");
    assert_eq!(events[4]["type"], "response.output_item.done");
    assert_eq!(events[5]["type"], "response.completed");
    assert_eq!(events[4]["item"]["name"], "my_func");
    assert_eq!(events[4]["item"]["arguments"], "arg1arg2");
}

// =========================================================================================
// tool-call buffering
// =========================================================================================

#[tokio::test]
async fn stream_response_buffers_tool_call_deltas_when_enabled() {
    let events = events_of(
        &buffered(),
        vec![
            chunk(json!({"tool_calls": [tool_call(0, Some("tool-id"), Some("my_func"), "arg1")]})),
            chunk_usage(
                json!({"tool_calls": [tool_call(0, None, None, "arg2")]}),
                usage(1, 1, 2),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    let deltas = of_type(&events, "response.function_call_arguments.delta");
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0]["delta"], "arg1arg2");
    let done = of_type(&events, "response.output_item.done")[0];
    assert_eq!(done["item"]["call_id"], "tool-id");
    assert_eq!(done["item"]["name"], "my_func");
    assert_eq!(done["item"]["arguments"], "arg1arg2");
    assert_eq!(completed(&events)["response"]["usage"]["total_tokens"], 2);
}

#[tokio::test]
async fn stream_response_buffers_tool_call_usage_chunk_without_replay() {
    let events = events_of(
        &buffered(),
        vec![
            chunk_usage(
                json!({"tool_calls": [tool_call(0, Some("tool-id"), Some("my_func"), "arg1")]}),
                usage(1, 1, 2),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    let deltas = of_type(&events, "response.function_call_arguments.delta");
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0]["delta"], "arg1");
    assert_eq!(function_done(&events).len(), 1);
    assert_eq!(completed(&events)["response"]["usage"]["total_tokens"], 2);
}

#[tokio::test]
async fn stream_response_buffers_tool_call_provider_fields() {
    let mut first = tool_call(0, Some("tool-id"), Some("my_func"), "");
    first["provider_specific_fields"] = json!({"thought_signature": "thought-sig"});
    let events = events_of(
        &Cfg {
            model: "gemini/gemini-3-pro",
            ..buffered()
        },
        vec![
            chunk(json!({"tool_calls": [first]})),
            chunk(json!({"tool_calls": [tool_call(0, None, None, "arg1")]})),
            finish("tool_calls"),
        ],
    )
    .await;
    let done = function_done(&events);
    assert_eq!(done.len(), 1);
    assert_eq!(
        done[0]["item"]["provider_data"]["thought_signature"],
        "thought-sig"
    );
}

#[tokio::test]
async fn stream_response_buffered_tool_calls_raise_for_missing_tool_call_delta() {
    let (_, err) = run(&buffered(), vec![finish("tool_calls")]).await;
    let err = err.expect("buffering error").to_string();
    assert!(err.contains("finish_reason='tool_calls'"), "{err}");
}

#[tokio::test]
async fn buffered_tool_calls_preserve_nonzero_choice_validation() {
    let (_, err) = run(
        &buffered().with_strict_feature_validation(true),
        vec![chunk_with(
            vec![json!({"index": 1, "delta": {"tool_calls": [
                tool_call(0, Some("tool-id"), Some("my_func"), "arg")]}})],
            None,
        )],
    )
    .await;
    let err = err.expect("strict error").to_string();
    assert!(
        err.contains("multiple choices or nonzero choice indexes"),
        "{err}"
    );
}

#[tokio::test]
async fn buffered_tool_calls_do_not_merge_nonzero_choice_tool_call_indexes() {
    let events = events_of(
        &buffered(),
        vec![
            chunk_with(
                vec![
                    json!({"index": 0, "delta": {"tool_calls": [
                        tool_call(0, Some("choice-zero-tool-id"), Some("choice_zero_func"),
                                  "choice-zero")]}}),
                    json!({"index": 1, "delta": {"tool_calls": [
                        tool_call(0, Some("choice-one-tool-id"), Some("choice_one_func"),
                                  "choice-one")]}}),
                ],
                None,
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    let done = function_done(&events);
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["item"]["call_id"], "choice-zero-tool-id");
    assert_eq!(done[0]["item"]["name"], "choice_zero_func");
    assert_eq!(done[0]["item"]["arguments"], "choice-zero");
}

fn custom_delta() -> Value {
    json!({"index": 0, "id": "tool-call-123", "type": "custom"})
}

fn custom_chunk() -> Value {
    chunk_with(
        vec![
            json!({"index": 0, "delta": {"tool_calls": [custom_delta()]},
                    "finish_reason": "tool_calls"}),
        ],
        None,
    )
}

#[tokio::test]
async fn buffered_tool_calls_preserve_custom_tool_call_strict_error() {
    let (_, err) = run(
        &buffered().with_strict_feature_validation(true),
        vec![custom_chunk()],
    )
    .await;
    let err = err.expect("strict error").to_string();
    assert!(err.contains("Custom tool calls are not supported"), "{err}");
}

#[tokio::test]
async fn buffered_tool_calls_ignore_custom_tool_call_by_default() {
    let events = events_of(&buffered(), vec![custom_chunk()]).await;
    assert_eq!(completed(&events)["response"]["output"], json!([]));
}

#[tokio::test]
async fn stream_response_with_custom_tool_call_raises_in_strict_mode() {
    let (_, err) = run(
        &strict(),
        vec![chunk(json!({"tool_calls": [custom_delta()]}))],
    )
    .await;
    let err = err.expect("strict error").to_string();
    assert!(err.contains("Custom tool calls are not supported"), "{err}");
}

#[tokio::test]
async fn stream_response_ignores_custom_tool_call_chunks_by_default() {
    let omitted_type = json!({"index": 0, "function": {"name": "custom_tool",
                                                       "arguments": "payload"}});
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"tool_calls": [custom_delta()]})),
            chunk(json!({"tool_calls": [omitted_type]})),
            chunk(json!({"content": "done"})),
            finish("tool_calls"),
        ],
    )
    .await;
    assert!(events.iter().all(|e| e["item"]["type"] != "function_call"));
    let output = completed(&events)["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"].as_array().unwrap().len(), 1);
    assert_eq!(output[0]["content"][0]["text"], "done");
}

// =========================================================================================
// output indexes
// =========================================================================================

#[tokio::test]
async fn stream_response_yields_real_time_function_call_arguments() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"tool_calls": [tool_call(0, Some("tool-call-123"), Some("write_file"), "")]})),
            chunk(json!({"tool_calls": [tool_call(0, None, None, "{\"filename\": \"")]})),
            chunk(json!({"tool_calls": [tool_call(0, None, None, "test.py\", \"content\": \"")]})),
            chunk_usage(
                json!({"tool_calls": [tool_call(0, None, None, "print(hello)\"}")]}),
                usage(1, 1, 2),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    assert_eq!(of_type(&events, "response.created").len(), 1);
    let added = of_type(&events, "response.output_item.added");
    assert_eq!(added.len(), 1);
    assert_eq!(added[0]["item"]["name"], "write_file");
    assert_eq!(added[0]["item"]["call_id"], "tool-call-123");
    assert_eq!(added[0]["item"]["arguments"], "");
    let deltas = of_type(&events, "response.function_call_arguments.delta");
    assert_eq!(deltas.len(), 3);
    let expected = [
        "{\"filename\": \"",
        "test.py\", \"content\": \"",
        "print(hello)\"}",
    ];
    for (delta, want) in deltas.iter().zip(expected) {
        assert_eq!(delta["delta"], want);
        assert_eq!(delta["item_id"], "__fake_id__");
        assert_eq!(delta["output_index"], 0);
    }
    let done = of_type(&events, "response.output_item.done");
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["item"]["name"], "write_file");
    assert_eq!(
        done[0]["item"]["arguments"],
        "{\"filename\": \"test.py\", \"content\": \"print(hello)\"}"
    );
    assert_eq!(of_type(&events, "response.completed").len(), 1);
    let first = &completed(&events)["response"]["output"][0];
    assert_eq!(first["type"], "function_call");
    assert_eq!(first["name"], "write_file");
}

fn indexes(events: &[Value], ty: &str) -> Vec<Value> {
    events
        .iter()
        .filter(|e| e["type"] == ty)
        .map(|e| e["output_index"].clone())
        .collect()
}

#[tokio::test]
async fn fallback_function_calls_have_unique_output_indexes() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"tool_calls": [tool_call(0, None, Some("first_tool"), "{\"a\": 1}")]})),
            chunk_usage(
                json!({"tool_calls": [tool_call(1, None, Some("second_tool"), "{\"b\": 2}")]}),
                usage(1, 1, 2),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    assert_eq!(
        indexes(&events, "response.output_item.added"),
        [json!(0), json!(1)]
    );
    assert_eq!(
        indexes(&events, "response.function_call_arguments.delta"),
        [json!(0), json!(1)]
    );
    assert_eq!(
        indexes(&events, "response.output_item.done"),
        [json!(0), json!(1)]
    );
}

#[tokio::test]
async fn fallback_function_call_keeps_index_before_streamed_call() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"tool_calls": [tool_call(0, None, Some("fallback_first"), "{\"a\": 1}")]})),
            chunk(json!({"tool_calls": [tool_call(1, Some("tool-call-2"), Some("streamed_second"), "")]})),
            chunk_usage(
                json!({"tool_calls": [tool_call(1, None, None, "{\"b\": 2}")]}),
                usage(1, 1, 2),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    let names: Vec<&str> = completed(&events)["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["type"] == "function_call")
        .map(|i| i["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["fallback_first", "streamed_second"]);
    let by_name = |ty: &str| -> HashMap<String, Value> {
        of_item(&events, ty, "function_call")
            .iter()
            .map(|e| {
                (
                    e["item"]["name"].as_str().unwrap().to_string(),
                    e["output_index"].clone(),
                )
            })
            .collect()
    };
    let expected: HashMap<String, Value> = [
        ("fallback_first".to_string(), json!(0)),
        ("streamed_second".to_string(), json!(1)),
    ]
    .into();
    assert_eq!(by_name("response.output_item.added"), expected);
    assert_eq!(
        indexes(&events, "response.function_call_arguments.delta"),
        [json!(1), json!(0)]
    );
    assert_eq!(by_name("response.output_item.done"), expected);
}

#[tokio::test]
async fn fallback_function_call_before_text_uses_final_output_index() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"tool_calls": [tool_call(0, None, Some("first_tool"), "{\"a\": 1}")]})),
            chunk_usage(json!({"content": "answer"}), usage(1, 1, 2)),
            finish("tool_calls"),
        ],
    )
    .await;
    let index_of = |ty: &str, item_type: &str| -> Value {
        of_item(&events, ty, item_type)[0]["output_index"].clone()
    };
    assert_eq!(index_of("response.output_item.added", "message"), 0);
    assert_eq!(index_of("response.output_item.added", "function_call"), 1);
    assert_eq!(
        indexes(&events, "response.function_call_arguments.delta"),
        [json!(1)]
    );
    assert_eq!(index_of("response.output_item.done", "message"), 0);
    assert_eq!(index_of("response.output_item.done", "function_call"), 1);
    let output = completed(&events)["response"]["output"].as_array().unwrap();
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[1]["type"], "function_call");
}

#[tokio::test]
async fn streamed_function_call_before_text_keeps_realtime_order() {
    let events = events_of(
        &plain(),
        vec![
            chunk(
                json!({"tool_calls": [tool_call(0, Some("tool-call-1"), Some("first_tool"), "")]}),
            ),
            chunk(json!({"tool_calls": [tool_call(0, None, None, "{\"a\": 1}")]})),
            chunk_usage(json!({"content": "answer"}), usage(1, 1, 2)),
            finish("tool_calls"),
        ],
    )
    .await;
    let index_of = |ty: &str, item_type: &str| -> Value {
        of_item(&events, ty, item_type)[0]["output_index"].clone()
    };
    assert_eq!(index_of("response.output_item.added", "function_call"), 0);
    assert_eq!(index_of("response.output_item.added", "message"), 1);
    assert_eq!(
        indexes(&events, "response.function_call_arguments.delta"),
        [json!(0)]
    );
    assert_eq!(index_of("response.output_item.done", "function_call"), 0);
    assert_eq!(index_of("response.output_item.done", "message"), 1);
    let output = completed(&events)["response"]["output"].as_array().unwrap();
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[1]["type"], "message");
}

#[tokio::test]
async fn mixed_function_calls_before_text_keep_tracked_order() {
    let events = events_of(
        &plain(),
        vec![
            chunk(json!({"tool_calls": [tool_call(0, None, Some("fallback_first"), "{\"a\": 1}")]})),
            chunk(json!({"tool_calls": [tool_call(1, Some("tool-call-2"), Some("streamed_second"), "")]})),
            chunk(json!({"tool_calls": [tool_call(1, None, None, "{\"b\": 2}")]})),
            chunk_usage(json!({"content": "answer"}), usage(1, 1, 2)),
            finish("tool_calls"),
        ],
    )
    .await;
    let tool_indexes: HashMap<String, Value> =
        of_item(&events, "response.output_item.added", "function_call")
            .iter()
            .map(|e| {
                (
                    e["item"]["name"].as_str().unwrap().to_string(),
                    e["output_index"].clone(),
                )
            })
            .collect();
    let expected: HashMap<String, Value> = [
        ("streamed_second".to_string(), json!(1)),
        ("fallback_first".to_string(), json!(0)),
    ]
    .into();
    assert_eq!(tool_indexes, expected);
    assert_eq!(
        of_item(&events, "response.output_item.added", "message")[0]["output_index"],
        2
    );
    let by_delta: HashMap<String, Value> =
        of_type(&events, "response.function_call_arguments.delta")
            .iter()
            .map(|e| {
                (
                    e["delta"].as_str().unwrap().to_string(),
                    e["output_index"].clone(),
                )
            })
            .collect();
    let want: HashMap<String, Value> = [
        ("{\"b\": 2}".to_string(), json!(1)),
        ("{\"a\": 1}".to_string(), json!(0)),
    ]
    .into();
    assert_eq!(by_delta, want);
    let output = completed(&events)["response"]["output"].as_array().unwrap();
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[2]["type"], "message");
}

// =========================================================================================
// buffered content filter and length
// =========================================================================================

#[tokio::test]
async fn buffered_stream_synthesizes_refusal_on_content_filter() {
    let events = events_of(
        &buffered(),
        vec![
            chunk(json!({"role": "assistant", "content": ""})),
            chunk_with(
                vec![json!({"index": 0, "delta": {}, "finish_reason": "content_filter"})],
                Some(usage(0, 7, 7)),
            ),
        ],
    )
    .await;
    let kinds = types(&events);
    assert!(kinds.contains(&"response.refusal.delta"));
    assert_eq!(*kinds.last().unwrap(), "response.completed");
    let refusal_deltas = of_type(&events, "response.refusal.delta");
    assert!(!refusal_deltas[0]["delta"].as_str().unwrap().is_empty());
    // The assistant message is announced once and every opened part is closed.
    assert_eq!(
        kinds
            .iter()
            .filter(|t| **t == "response.output_item.added")
            .count(),
        1
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|t| **t == "response.content_part.added")
            .count(),
        kinds
            .iter()
            .filter(|t| **t == "response.content_part.done")
            .count()
    );
    // The empty "" content delta must not open a text content part.
    assert!(!kinds.contains(&"response.output_text.delta"));
    let added = of_type(&events, "response.content_part.added");
    assert_eq!(added.len(), 1);
    assert_eq!(added[0]["part"]["type"], "refusal");
    let message = &completed(&events)["response"]["output"][0];
    assert_eq!(message["type"], "message");
    assert_eq!(message["content"].as_array().unwrap().len(), 1);
    assert_eq!(message["content"][0]["type"], "refusal");
    assert!(
        !message["content"][0]["refusal"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    assert_eq!(added[0]["content_index"], 0);
    assert_eq!(refusal_deltas[0]["content_index"], 0);
}

#[tokio::test]
async fn buffered_stream_content_filter_does_not_clobber_text() {
    let events = events_of(
        &buffered(),
        vec![
            chunk(json!({"content": "answer"})),
            chunk_with(
                vec![json!({"index": 0, "delta": {}, "finish_reason": "content_filter"})],
                Some(usage(1, 7, 8)),
            ),
        ],
    )
    .await;
    assert!(!types(&events).contains(&"response.refusal.delta"));
    let message = &completed(&events)["response"]["output"][0];
    assert_eq!(message["content"][0]["type"], "output_text");
    assert_eq!(message["content"][0]["text"], "answer");
}

#[tokio::test]
async fn buffered_stream_content_filter_refusal_after_reasoning() {
    let events = events_of(
        &buffered(),
        vec![
            chunk(
                json!({"role": "assistant", "content": null, "reasoning_content": "thinking..."}),
            ),
            chunk(json!({"content": ""})),
            chunk_with(
                vec![json!({"index": 0, "delta": {}, "finish_reason": "content_filter"})],
                Some(usage(0, 7, 7)),
            ),
        ],
    )
    .await;
    let output = completed(&events)["response"]["output"].as_array().unwrap();
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"].as_array().unwrap().len(), 1);
    assert_eq!(output[1]["content"][0]["type"], "refusal");
    let added: Vec<&Value> = of_type(&events, "response.content_part.added")
        .into_iter()
        .filter(|e| e["part"]["type"] == "refusal")
        .collect();
    let deltas = of_type(&events, "response.refusal.delta");
    assert_eq!(added.len(), 1);
    assert_eq!(added[0]["content_index"], 0);
    assert_eq!(added[0]["output_index"], 1);
    assert!(!deltas.is_empty());
    assert!(
        deltas
            .iter()
            .all(|d| d["content_index"] == 0 && d["output_index"] == 1)
    );
    assert!(!types(&events).contains(&"response.output_text.delta"));
}

#[tokio::test]
async fn buffered_stream_raises_on_length_truncation() {
    let (_, err) = run(
        &buffered(),
        vec![
            chunk(json!({"role": "assistant", "content": ""})),
            chunk_with(
                vec![json!({"index": 0, "delta": {}, "finish_reason": "length"})],
                Some(usage(0, 7, 7)),
            ),
        ],
    )
    .await;
    let err = err.expect("length error").to_string();
    assert!(err.contains("finish_reason='length'"), "{err}");
}

#[tokio::test]
async fn buffered_stream_length_does_not_clobber_text() {
    let events = events_of(
        &buffered(),
        vec![
            chunk(json!({"content": "answer"})),
            chunk_with(
                vec![json!({"index": 0, "delta": {}, "finish_reason": "length"})],
                Some(usage(1, 7, 8)),
            ),
        ],
    )
    .await;
    assert!(!types(&events).contains(&"response.refusal.delta"));
    let message = &completed(&events)["response"]["output"][0];
    assert_eq!(message["content"][0]["text"], "answer");
}

// =========================================================================================
// request id, gemini ids
// =========================================================================================

#[tokio::test]
async fn stream_response_propagates_request_id() {
    let (base, _) = serve_sse(
        vec![vec![chunk(json!({"content": "Hello"})), finish("stop")]],
        Some("req_streamed_456"),
    )
    .await;
    let (events, err) = collect(&model_at(&base), request()).await;
    assert!(err.is_none());
    assert_eq!(
        completed(&events)["response"]["_request_id"],
        "req_streamed_456"
    );
}

#[tokio::test]
async fn stream_response_without_http_response_has_no_request_id() {
    // No `x-request-id` header on the response: the terminal response carries no id.
    let (base, _) = serve_sse(
        vec![vec![chunk(json!({"content": "Hello"})), finish("stop")]],
        None,
    )
    .await;
    let (events, err) = collect(&model_at(&base), request()).await;
    assert!(err.is_none());
    assert!(completed(&events)["response"].get("_request_id").is_none());
}

#[tokio::test]
async fn clean_gemini_tool_call_id_removes_thought_suffix() {
    let model = Cfg {
        model: "gemini-2.5-pro",
        ..plain()
    };
    let events = events_of(
        &model,
        vec![
            chunk(
                json!({"tool_calls": [tool_call(0, Some("call_123__thought__signature"),
                                                   Some("my_func"), "{}")]}),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    assert_eq!(function_done(&events)[0]["item"]["call_id"], "call_123");
    // Other models keep the suffix.
    let events = events_of(
        &plain(),
        vec![
            chunk(
                json!({"tool_calls": [tool_call(0, Some("call_123__thought__signature"),
                                                   Some("my_func"), "{}")]}),
            ),
            finish("tool_calls"),
        ],
    )
    .await;
    assert_eq!(
        function_done(&events)[0]["item"]["call_id"],
        "call_123__thought__signature"
    );
}
