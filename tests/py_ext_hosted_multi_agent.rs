//! Port of `tests/extensions/experimental/hosted_multi_agent/test_model.py` (and `test_live.py`).
//!
//! Python drives the model through a dummy beta WebSocket connection that replays a queue of
//! events and records the frames it was sent. The mock server below is the same thing over a
//! real local WebSocket: each accepted connection serves one scripted event queue, sent when
//! the first frame arrives, and records every frame plus whether the client closed it.
#![cfg(feature = "websocket")]
#![allow(clippy::result_large_err)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::extensions::hosted_multi_agent::{
    HostedMultiAgentConfig, OpenAIHostedMultiAgentModel, get_hosted_agent_metadata,
    get_hosted_agent_metadata_for_tool,
};
use openai_agents_sdk::models::interface::{Model, ModelProvider, ModelRequest};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, ModelSettings, OpenAIClient, RunConfig, RunOptions, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

// ---------------------------------------------------------------------------------------------
// event builders (the helpers at the top of the Python file)
// ---------------------------------------------------------------------------------------------

fn response(id: &str, output: Vec<Value>) -> Value {
    json!({"id": id, "object": "response", "status": "completed", "output": output, "usage": null})
}

fn usage(input: u64, cached: u64, cache_write: u64, output: u64, reasoning: u64) -> Value {
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": cache_write},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input + output,
    })
}

fn created(id: &str) -> Value {
    json!({"type": "response.created", "sequence_number": 1, "response": response(id, vec![])})
}

fn done(item: &Value, sequence_number: u64, output_index: u64) -> Value {
    json!({
        "type": "response.output_item.done",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item": item,
        "agent": item.get("agent").cloned().unwrap_or(Value::Null),
    })
}

fn completed(id: &str, output: Vec<Value>, sequence_number: u64) -> Value {
    json!({
        "type": "response.completed",
        "sequence_number": sequence_number,
        "response": response(id, output),
    })
}

fn completed_with_usage(id: &str, output: Vec<Value>, sequence_number: u64, usage: Value) -> Value {
    let mut event = completed(id, output, sequence_number);
    event["response"]["usage"] = usage;
    event
}

fn root_final_message(text: &str) -> Value {
    json!({
        "id": "msg_root", "type": "message", "role": "assistant", "status": "completed",
        "agent": {"agent_name": "/root"}, "phase": "final_answer",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    })
}

fn subagent_message(text: &str) -> Value {
    json!({
        "id": "msg_subagent", "type": "message", "role": "assistant", "status": "completed",
        "agent": {"agent_name": "/root/researcher"}, "phase": "commentary",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    })
}

fn function_call_item(id: &str, call_id: &str, section: &str, agent: &str) -> Value {
    json!({
        "id": id, "type": "function_call", "call_id": call_id, "name": "lookup_document",
        "arguments": format!("{{\"section\":\"{section}\"}}"), "status": "completed",
        "agent": {"agent_name": agent},
    })
}

fn inject_created(response_id: &str, sequence_number: u64) -> Value {
    json!({
        "type": "response.inject.created",
        "sequence_number": sequence_number,
        "response_id": response_id,
    })
}

fn tool_flow_events(response_id: &str, first: &[Value], last: &[Value]) -> Vec<Value> {
    let mut events = vec![created(response_id)];
    let mut sequence = 1;
    for (index, item) in first.iter().enumerate() {
        sequence += 1;
        events.push(done(item, sequence, index as u64));
    }
    sequence += 1;
    events.push(inject_created(response_id, sequence));
    for (index, item) in last.iter().enumerate() {
        sequence += 1;
        events.push(done(item, sequence, (first.len() + index) as u64));
    }
    sequence += 1;
    let all: Vec<Value> = first.iter().chain(last.iter()).cloned().collect();
    events.push(completed(response_id, all, sequence));
    events
}

// ---------------------------------------------------------------------------------------------
// the mock beta WebSocket
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct ConnectionLog {
    sent: Mutex<Vec<Value>>,
    closed: AtomicBool,
}

/// One connection's scripted events and the log it records into.
type Script = (Vec<Value>, Arc<ConnectionLog>);

struct Mock {
    base_url: String,
    connections: Vec<Arc<ConnectionLog>>,
    connects: Arc<Mutex<Vec<String>>>,
}

impl Mock {
    fn sent(&self, index: usize) -> Vec<Value> {
        self.connections[index].sent.lock().unwrap().clone()
    }

    /// The mock records a frame when its server task reads it, which can trail the client
    /// finishing its run (the whole event queue is delivered on the first frame), so wait for
    /// at least `count` frames before inspecting them.
    async fn sent_at_least(&self, index: usize, count: usize) -> Vec<Value> {
        for _ in 0..200 {
            let frames = self.sent(index);
            if frames.len() >= count {
                return frames;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.sent(index)
    }

    /// The server notices a client close asynchronously, so poll for it briefly.
    async fn assert_closed(&self, index: usize) {
        for _ in 0..200 {
            if self.connections[index].closed.load(Ordering::SeqCst) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("connection {index} was not closed by the client");
    }

    fn assert_still_open(&self, index: usize) {
        assert!(!self.connections[index].closed.load(Ordering::SeqCst));
    }

    fn connect_count(&self) -> usize {
        self.connects.lock().unwrap().len()
    }
}

/// Serve one scripted event queue per accepted connection, in order.
async fn mock(batches: Vec<Vec<Value>>) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let connections: Vec<Arc<ConnectionLog>> = batches
        .iter()
        .map(|_| Arc::new(ConnectionLog::default()))
        .collect();
    let connects: Arc<Mutex<Vec<String>>> = Arc::default();
    let queue: Arc<Mutex<VecDeque<Script>>> = Arc::new(Mutex::new(
        batches
            .into_iter()
            .zip(connections.iter().cloned())
            .collect(),
    ));
    let recorded = connects.clone();
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            let Some((events, log)) = queue.lock().unwrap().pop_front() else {
                continue;
            };
            let recorded = recorded.clone();
            let callback = move |req: &Request, resp: Response| {
                let beta = req
                    .headers()
                    .get("openai-beta")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                recorded.lock().unwrap().push(beta);
                Ok(resp)
            };
            let Ok(mut ws) = tokio_tungstenite::accept_hdr_async(tcp, callback).await else {
                continue;
            };
            tokio::spawn(async move {
                let mut events = Some(events);
                while let Some(message) = ws.next().await {
                    match message {
                        Ok(Message::Text(text)) => {
                            log.sent
                                .lock()
                                .unwrap()
                                .push(serde_json::from_str(&text).unwrap());
                            // The dummy connection hands its whole queue to whoever reads first.
                            if let Some(events) = events.take() {
                                for event in events {
                                    if ws.send(Message::text(event.to_string())).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        Ok(Message::Close(_)) | Err(_) => break,
                        Ok(_) => {}
                    }
                }
                log.closed.store(true, Ordering::SeqCst);
            });
        }
    });
    Mock {
        base_url: format!("http://{addr}/v1"),
        connections,
        connects,
    }
}

fn model_for(mock: &Mock) -> OpenAIHostedMultiAgentModel {
    OpenAIHostedMultiAgentModel::new(
        "gpt-5.6-sol",
        OpenAIClient::new("sk-test").with_base_url(mock.base_url.clone()),
    )
}

fn offline_model() -> OpenAIHostedMultiAgentModel {
    OpenAIHostedMultiAgentModel::new(
        "gpt-5.6-sol",
        OpenAIClient::new("sk-test").with_base_url("http://127.0.0.1:9/v1"),
    )
}

fn request(input: &str, settings: ModelSettings) -> ModelRequest {
    ModelRequest {
        input: vec![openai_agents_sdk::items::InputItem(
            json!({"role": "user", "content": input}),
        )],
        model_settings: settings,
        ..Default::default()
    }
}

fn options() -> RunOptions<()> {
    let mut options = RunOptions::<()>::default();
    options.run_config = RunConfig::<()>::new().with_tracing_disabled(true);
    options
}

#[derive(Deserialize, JsonSchema)]
struct SectionArgs {
    section: String,
}

#[derive(Deserialize, JsonSchema)]
struct ProposalArgs {
    proposal: String,
}

/// `lookup_document`, recording the hosted agent that called it.
fn lookup_tool(callers: Arc<Mutex<Vec<String>>>) -> FunctionTool<()> {
    FunctionTool::new(
        "lookup_document",
        "Look up a document",
        move |ctx, a: SectionArgs| {
            let callers = callers.clone();
            async move {
                let metadata = get_hosted_agent_metadata_for_tool(&ctx).expect("hosted metadata");
                callers.lock().unwrap().push(metadata.agent_name);
                Ok(format!("document:{}", a.section))
            }
        },
    )
}

fn plain_lookup_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "lookup_document",
        "Look up a document",
        |_ctx, a: SectionArgs| async move { Ok(format!("document:{}", a.section)) },
    )
}

fn item_types(result: &openai_agents_sdk::RunResult<()>) -> Vec<&'static str> {
    result.new_items.iter().map(|i| i.item_type()).collect()
}

async fn with_timeout<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(20), future)
        .await
        .expect("timed out")
}

// ---------------------------------------------------------------------------------------------
// request building
// ---------------------------------------------------------------------------------------------

#[test]
fn build_request_enables_beta_and_preserves_context_management() {
    let model = offline_model()
        .with_config(HostedMultiAgentConfig {
            max_concurrent_subagents: Some(2),
        })
        .unwrap();
    let settings = ModelSettings {
        context_management: Some(vec![json!({"type": "compaction", "compact_threshold": 10})]),
        ..Default::default()
    };

    let frame = model.build_frame(&request("hello", settings)).unwrap();

    assert_eq!(
        frame["multi_agent"],
        json!({"enabled": true, "max_concurrent_subagents": 2})
    );
    assert_eq!(
        frame["context_management"],
        json!([{"type": "compaction", "compact_threshold": 10}])
    );
    // The beta is switched on by the `OpenAI-Beta` header of the connection, not by the frame.
    assert!(frame.get("betas").is_none());
    assert_eq!(frame["type"], json!("response.create"));
    assert!(frame.get("stream").is_none());
}

/// Python accepts a mapping for `config`; Rust takes the typed struct, so the mapping form is
/// the struct literal.
#[test]
fn the_model_accepts_a_config_value() {
    let config = HostedMultiAgentConfig {
        max_concurrent_subagents: Some(2),
    };
    let model = offline_model().with_config(config).unwrap();
    assert_eq!(model.config, config);
}

#[test]
fn the_model_can_be_built_without_connecting() {
    let model = offline_model()
        .with_config(HostedMultiAgentConfig {
            max_concurrent_subagents: Some(2),
        })
        .unwrap();
    assert_eq!(model.model_name().as_deref(), Some("gpt-5.6-sol"));
}

/// `True` cannot be passed in Rust (the field is a `u32`), so only zero is reachable.
#[test]
fn a_non_positive_concurrency_is_rejected() {
    let err = offline_model()
        .with_config(HostedMultiAgentConfig {
            max_concurrent_subagents: Some(0),
        })
        .unwrap_err();
    assert!(err.to_string().contains("positive integer"));
}

#[test]
fn reserved_settings_fail_before_transport() {
    let model = offline_model();

    let reserved = ModelSettings {
        extra_args: Some(
            json!({"multi_agent": {"enabled": true}})
                .as_object()
                .unwrap()
                .clone(),
        ),
        ..Default::default()
    };
    let err = model.build_frame(&request("hello", reserved)).unwrap_err();
    assert!(
        err.to_string()
            .contains("through OpenAIHostedMultiAgentModel")
    );

    let max_tool_calls = ModelSettings {
        extra_args: Some(json!({"max_tool_calls": 1}).as_object().unwrap().clone()),
        ..Default::default()
    };
    let err = model
        .build_frame(&request("hello", max_tool_calls))
        .unwrap_err();
    assert!(err.to_string().contains("max_tool_calls"));

    let summary = ModelSettings {
        reasoning: Some(openai_agents_sdk::Reasoning {
            summary: Some("auto".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let err = model.build_frame(&request("hello", summary)).unwrap_err();
    assert!(err.to_string().contains("reasoning.summary"));
}

#[test]
fn sdk_handoffs_are_rejected() {
    let model = offline_model();
    let mut req = request("hello", ModelSettings::default());
    req.handoffs = vec![openai_agents_sdk::models::interface::FunctionToolSpec {
        name: "transfer_to_target".into(),
        description: "Handoff".into(),
        parameters: json!({"type": "object", "properties": {}}),
        strict: true,
        ..Default::default()
    }];
    let err = model.build_frame(&req).unwrap_err();
    assert!(
        err.to_string()
            .contains("cannot be combined with SDK handoffs")
    );
}

#[test]
fn the_default_config_omits_the_service_default_concurrency() {
    let frame = offline_model()
        .build_frame(&request("hello", ModelSettings::default()))
        .unwrap();
    assert_eq!(frame["multi_agent"], json!({"enabled": true}));
    assert!(frame.get("stream").is_none());
}

/// Python rejects tools with SDK approval settings in `_build_response_create_kwargs`. The Rust
/// request carries tool *specs* (name, schema), not the tool objects, so the approval check
/// happens where the tool is still visible: when the run builds the model request.
#[tokio::test]
async fn tools_with_approval_settings_are_rejected_before_transport() {
    let mock = mock(vec![vec![]]).await;
    let model = Arc::new(model_for(&mock));

    let always = plain_lookup_tool().with_needs_approval(true);
    let agent = Agent::<()>::new("SDK root")
        .with_model_instance(model.clone())
        .with_tool(always);
    let err = with_timeout(Runner::run_with_options(&agent, "hello", options()))
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not support SDK tool approval interruptions"),
        "{err}"
    );
    assert_eq!(mock.connect_count(), 0);

    let dynamic =
        plain_lookup_tool().with_needs_approval_fn(|_ctx, _params, _call_id| async { false });
    let agent = Agent::<()>::new("SDK root")
        .with_model_instance(model)
        .with_tool(dynamic);
    let err = with_timeout(Runner::run_with_options(&agent, "hello", options()))
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not support SDK tool approval interruptions"),
        "{err}"
    );
    assert_eq!(mock.connect_count(), 0);
}

// ---------------------------------------------------------------------------------------------
// runner routing
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn runner_routes_a_subagent_tool_call_without_exposing_hosted_items() {
    let first_items = vec![
        json!({
            "id": "mac_1", "type": "multi_agent_call", "call_id": "call_spawn",
            "action": "spawn_agent", "arguments": "{}", "agent": {"agent_name": "/root"},
        }),
        function_call_item("fc_1", "call_lookup", "alpha", "/root/researcher"),
    ];
    let final_items = vec![
        subagent_message("working"),
        json!({
            "id": "maco_1", "type": "multi_agent_call_output", "call_id": "call_spawn",
            "action": "spawn_agent", "output": [], "agent": {"agent_name": "/root"},
        }),
        root_final_message("done"),
    ];
    let mock = mock(vec![tool_flow_events("resp_1", &first_items, &final_items)]).await;
    let callers = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::<()>::new("SDK root")
        .with_instructions("Delegate the lookup and synthesize the answer.")
        .with_model_instance(Arc::new(model_for(&mock)))
        .with_tool(lookup_tool(callers.clone()));

    let result = with_timeout(Runner::run_with_options(
        &agent,
        "Compare alpha.",
        options(),
    ))
    .await
    .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(*callers.lock().unwrap(), vec!["/root/researcher"]);
    assert_eq!(
        item_types(&result),
        [
            "tool_call_item",
            "tool_call_output_item",
            "message_output_item"
        ]
    );

    let sent = mock.sent_at_least(0, 2).await;
    assert_eq!(sent[0]["type"], json!("response.create"));
    assert!(sent[0].get("stream").is_none());
    assert!(sent[0].get("betas").is_none());
    assert_eq!(
        sent[1],
        json!({
            "type": "response.inject",
            "response_id": "resp_1",
            "input": [{
                "type": "function_call_output",
                "call_id": "call_lookup",
                "output": "document:alpha",
            }],
        })
    );
    assert_eq!(
        *mock.connects.lock().unwrap(),
        vec!["responses_multi_agent=v1"]
    );
    mock.assert_closed(0).await;
}

#[tokio::test]
async fn runner_injects_two_subagent_calls_into_one_active_response() {
    let spawn = json!({
        "id": "mac_two", "type": "multi_agent_call", "call_id": "call_spawn_two",
        "action": "spawn_agent", "arguments": "{}", "agent": {"agent_name": "/root"},
    });
    let proposal_call = |id: &str, call_id: &str, proposal: &str, agent: &str| {
        json!({
            "id": id, "type": "function_call", "call_id": call_id, "name": "get_proposal",
            "arguments": format!("{{\"proposal\":\"{proposal}\"}}"), "status": "completed",
            "agent": {"agent_name": agent},
        })
    };
    let alpha_call = proposal_call("fc_alpha", "call_alpha", "alpha", "/root/alpha");
    let beta_call = proposal_call("fc_beta", "call_beta", "beta", "/root/beta");
    let mut alpha_message = subagent_message("alpha complete");
    alpha_message["id"] = json!("msg_alpha");
    alpha_message["agent"] = json!({"agent_name": "/root/alpha"});
    let mut beta_message = subagent_message("beta complete");
    beta_message["id"] = json!("msg_beta");
    beta_message["agent"] = json!({"agent_name": "/root/beta"});
    let root_final = root_final_message("comparison complete");
    let output = vec![
        spawn.clone(),
        alpha_call.clone(),
        alpha_message.clone(),
        beta_call.clone(),
        beta_message.clone(),
        root_final.clone(),
    ];
    let events = vec![
        created("resp_two"),
        done(&spawn, 2, 0),
        done(&alpha_call, 3, 1),
        inject_created("resp_two", 4),
        done(&alpha_message, 5, 2),
        done(&beta_call, 6, 3),
        inject_created("resp_two", 7),
        done(&beta_message, 8, 4),
        done(&root_final, 9, 5),
        completed("resp_two", output, 10),
    ];
    let mock = mock(vec![events]).await;

    let callers = Arc::new(Mutex::new(Vec::new()));
    let recorded = callers.clone();
    let tool = FunctionTool::new(
        "get_proposal",
        "Get a proposal",
        move |ctx, a: ProposalArgs| {
            let recorded = recorded.clone();
            async move {
                let metadata = get_hosted_agent_metadata_for_tool(&ctx).expect("hosted metadata");
                recorded.lock().unwrap().push(metadata.agent_name);
                Ok(format!("proposal:{}", a.proposal))
            }
        },
    );
    let agent = Agent::<()>::new("SDK root")
        .with_model_instance(Arc::new(model_for(&mock)))
        .with_tool(tool);

    let result = with_timeout(Runner::run_with_options(
        &agent,
        "Compare both proposals.",
        options(),
    ))
    .await
    .unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("comparison complete")
    );
    assert_eq!(*callers.lock().unwrap(), vec!["/root/alpha", "/root/beta"]);
    let injected: Vec<String> = mock
        .sent_at_least(0, 3)
        .await
        .iter()
        .filter(|f| f["type"] == "response.inject")
        .map(|f| f["input"][0]["call_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(injected, ["call_alpha", "call_beta"]);
    assert_eq!(
        item_types(&result),
        [
            "tool_call_item",
            "tool_call_output_item",
            "tool_call_item",
            "tool_call_output_item",
            "message_output_item",
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// injection failure continuation
// ---------------------------------------------------------------------------------------------

fn injection_failed(response_id: &str, input: Vec<Value>, code: &str, message: &str) -> Value {
    json!({
        "type": "response.inject.failed",
        "sequence_number": 4,
        "response_id": response_id,
        "input": input,
        "error": {"code": code, "message": message},
    })
}

async fn run_streamed_to_end(
    agent: &Agent<()>,
    input: &str,
    options: RunOptions<()>,
) -> (
    openai_agents_sdk::result::RunResultStreaming<()>,
    Vec<Value>,
    Option<AgentsError>,
) {
    let mut result = Runner::run_streamed_with_options(agent, input, options);
    let mut raw = Vec::new();
    let mut error = None;
    {
        let mut stream = Box::pin(result.stream_events());
        while let Some(event) = with_timeout(stream.next()).await {
            match event {
                Ok(openai_agents_sdk::StreamEvent::RawResponse(value)) => raw.push(value),
                Ok(_) => {}
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
    }
    (result, raw, error)
}

#[tokio::test]
async fn an_injection_failure_after_completion_starts_a_continuation_response() {
    for streamed in [false, true] {
        for reports_usage in [false, true] {
            let function_call =
                function_call_item("fc_fallback", "call_fallback", "alpha", "/root/researcher");
            let function_output = json!({
                "type": "function_call_output", "call_id": "call_fallback", "output": "document:alpha",
            });
            let mut first_completed = completed("resp_old", vec![function_call.clone()], 3);
            if reports_usage {
                first_completed["response"]["usage"] = usage(12, 3, 1, 4, 2);
            }
            let first = vec![
                created("resp_old"),
                done(&function_call, 2, 0),
                first_completed,
                injection_failed(
                    "resp_old",
                    vec![function_output.clone()],
                    "response_already_completed",
                    "The response already completed.",
                ),
            ];
            let final_message = root_final_message("continued");
            let mut second_completed = completed("resp_new", vec![final_message.clone()], 3);
            if reports_usage {
                second_completed["response"]["usage"] = usage(7, 2, 4, 3, 1);
            }
            let second = vec![
                created("resp_new"),
                done(&final_message, 2, 0),
                second_completed,
            ];
            let mut events = first;
            events.extend(second);
            let mock = mock(vec![events]).await;

            let agent = Agent::<()>::new("SDK root")
                .with_model_instance(Arc::new(model_for(&mock)))
                .with_model_settings(ModelSettings {
                    store: Some(false),
                    ..Default::default()
                })
                .with_tool(plain_lookup_tool());

            let label = format!("streamed={streamed} reports_usage={reports_usage}");
            let (final_text, run_usage) = if streamed {
                let (result, _raw, error) =
                    run_streamed_to_end(&agent, "Inspect alpha.", options()).await;
                assert!(error.is_none(), "{label}: {error:?}");
                (
                    result.final_output_text(),
                    result.result().expect("completed run").usage(),
                )
            } else {
                let result = with_timeout(Runner::run_with_options(
                    &agent,
                    "Inspect alpha.",
                    options(),
                ))
                .await
                .unwrap();
                (result.final_output_text(), result.usage())
            };

            assert_eq!(final_text.as_deref(), Some("continued"), "{label}");
            // The streamed runner path parses each terminal event with `parse_response`, which
            // ignores the aggregated `usage.requests` / per-request entries (cross-area, see the
            // ignored `streamed_continuation_reports_aggregated_usage_entries`).
            if !streamed {
                if reports_usage {
                    assert_eq!(run_usage.input_tokens, 19, "{label}");
                    assert_eq!(run_usage.input_tokens_details.cached_tokens, 5, "{label}");
                    assert_eq!(
                        run_usage.input_tokens_details.cache_write_tokens, 5,
                        "{label}"
                    );
                    assert_eq!(run_usage.output_tokens, 7, "{label}");
                    assert_eq!(
                        run_usage.output_tokens_details.reasoning_tokens, 3,
                        "{label}"
                    );
                    assert_eq!(run_usage.total_tokens, 26, "{label}");
                    assert_eq!(run_usage.request_usage_entries.len(), 2, "{label}");
                    assert_eq!(
                        run_usage
                            .request_usage_entries
                            .iter()
                            .map(|e| e.total_tokens)
                            .collect::<Vec<_>>(),
                        [16, 10],
                        "{label}"
                    );
                } else {
                    assert_eq!(run_usage.input_tokens, 0, "{label}");
                    assert_eq!(run_usage.output_tokens, 0, "{label}");
                    assert_eq!(run_usage.total_tokens, 0, "{label}");
                    assert!(run_usage.request_usage_entries.is_empty(), "{label}");
                }
                assert_eq!(run_usage.requests, 2, "{label}");
            }

            assert_eq!(mock.connections.len(), 1);
            let continuation = mock.sent_at_least(0, 3).await[2].clone();
            assert_eq!(continuation["type"], json!("response.create"), "{label}");
            assert_eq!(
                continuation["previous_response_id"],
                json!("resp_old"),
                "{label}"
            );
            assert_eq!(continuation["input"], json!([function_output]), "{label}");
            mock.assert_closed(0).await;
        }
    }
}

#[tokio::test]
async fn an_injection_failure_continues_with_the_conversation_instead_of_a_previous_response() {
    let function_call = function_call_item(
        "fc_conversation",
        "call_conversation",
        "alpha",
        "/root/researcher",
    );
    let function_output = json!({
        "type": "function_call_output", "call_id": "call_conversation", "output": "document:alpha",
    });
    let final_message = root_final_message("continued in conversation");
    let mock = mock(vec![vec![
        created("resp_conversation_old"),
        done(&function_call, 2, 0),
        completed("resp_conversation_old", vec![function_call.clone()], 3),
        injection_failed(
            "resp_conversation_old",
            vec![function_output.clone()],
            "response_already_completed",
            "The response already completed.",
        ),
        created("resp_conversation_new"),
        done(&final_message, 2, 0),
        completed("resp_conversation_new", vec![final_message.clone()], 3),
    ]])
    .await;
    let agent = Agent::<()>::new("SDK root")
        .with_model_instance(Arc::new(model_for(&mock)))
        .with_tool(plain_lookup_tool());
    let mut opts = options();
    opts.conversation_id = Some("conv_123".into());

    let result = with_timeout(Runner::run_with_options(&agent, "Inspect alpha.", opts))
        .await
        .unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("continued in conversation")
    );
    let continuation = mock.sent_at_least(0, 3).await[2].clone();
    assert_eq!(continuation["conversation"], json!("conv_123"));
    assert!(continuation.get("previous_response_id").is_none());
    assert_eq!(continuation["input"], json!([function_output]));
    mock.assert_closed(0).await;
}

#[tokio::test]
async fn a_nonrecoverable_injection_failure_closes_the_active_response() {
    let function_call = function_call_item("fc_failed", "call_failed", "alpha", "/root/researcher");
    let mock = mock(vec![vec![
        created("resp_failed"),
        done(&function_call, 2, 0),
        json!({
            "type": "response.inject.failed", "sequence_number": 3,
            "response_id": "resp_failed", "input": [],
            "error": {"code": "response_not_found", "message": "Missing response."},
        }),
    ]])
    .await;
    let agent = Agent::<()>::new("SDK root")
        .with_model_instance(Arc::new(model_for(&mock)))
        .with_tool(plain_lookup_tool());

    let err = with_timeout(Runner::run_with_options(
        &agent,
        "Inspect alpha.",
        options(),
    ))
    .await
    .unwrap_err();

    assert!(err.to_string().contains("response_not_found"), "{err}");
    mock.assert_closed(0).await;
}

// ---------------------------------------------------------------------------------------------
// lifecycle: close and cleanup
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn closing_the_model_releases_a_paused_response() {
    let function_call = function_call_item("fc_close", "call_close", "alpha", "/root/researcher");
    let mock = mock(vec![vec![
        created("resp_close"),
        done(&function_call, 2, 0),
    ]])
    .await;
    let model = model_for(&mock);

    let response =
        with_timeout(model.get_response(request("Inspect alpha.", ModelSettings::default())))
            .await
            .unwrap();

    assert_eq!(response.response_id.as_deref(), Some("resp_close"));
    mock.assert_still_open(0);
    model.close().await;
    mock.assert_closed(0).await;
}

#[tokio::test]
async fn the_runner_closes_a_paused_response_when_max_turns_is_exceeded() {
    for streamed in [false, true] {
        let function_call = function_call_item(
            "fc_max_turns",
            "call_max_turns",
            "alpha",
            "/root/researcher",
        );
        let mock = mock(vec![vec![
            created("resp_max_turns"),
            done(&function_call, 2, 0),
        ]])
        .await;
        let agent = Agent::<()>::new("SDK root")
            .with_model_instance(Arc::new(model_for(&mock)))
            .with_tool(plain_lookup_tool());
        let mut opts = options();
        opts.max_turns = Some(1);

        let err = if streamed {
            let (_result, _raw, error) = run_streamed_to_end(&agent, "Inspect alpha.", opts).await;
            error.expect("max turns error")
        } else {
            with_timeout(Runner::run_with_options(&agent, "Inspect alpha.", opts))
                .await
                .unwrap_err()
        };

        assert!(err.is_max_turns_exceeded(), "streamed={streamed}: {err}");
        mock.assert_closed(0).await;
    }
}

/// A model resolved through a `ModelProvider` (by name) is cleaned up like one set on the agent.
#[derive(Debug)]
struct StaticProvider {
    model: Arc<OpenAIHostedMultiAgentModel>,
    requested: Mutex<Vec<Option<String>>>,
}

impl ModelProvider for StaticProvider {
    fn get_model(&self, model_name: Option<&str>) -> openai_agents_sdk::Result<Arc<dyn Model>> {
        self.requested
            .lock()
            .unwrap()
            .push(model_name.map(str::to_string));
        Ok(self.model.clone())
    }
}

#[tokio::test]
async fn the_runner_cleans_up_a_provider_resolved_paused_response() {
    for streamed in [false, true] {
        for model_source in ["run_config", "agent"] {
            let function_call =
                function_call_item("fc_provider", "call_provider", "alpha", "/root/researcher");
            let mock = mock(vec![vec![
                created("resp_provider"),
                done(&function_call, 2, 0),
            ]])
            .await;
            let provider = Arc::new(StaticProvider {
                model: Arc::new(model_for(&mock)),
                requested: Mutex::new(Vec::new()),
            });
            let agent = Agent::<()>::new("SDK root")
                .with_model(if model_source == "agent" {
                    "hosted"
                } else {
                    "unused"
                })
                .with_tool(plain_lookup_tool());
            let mut config = RunConfig::<()>::new()
                .with_model_provider(provider.clone())
                .with_tracing_disabled(true);
            if model_source == "run_config" {
                config = config.with_model("hosted");
            }
            let mut opts = RunOptions::<()>::default();
            opts.run_config = config;
            opts.max_turns = Some(1);

            let err = if streamed {
                let (_result, _raw, error) =
                    run_streamed_to_end(&agent, "Inspect alpha.", opts).await;
                error.expect("max turns error")
            } else {
                with_timeout(Runner::run_with_options(&agent, "Inspect alpha.", opts))
                    .await
                    .unwrap_err()
            };

            let label = format!("streamed={streamed} source={model_source}");
            assert!(err.is_max_turns_exceeded(), "{label}: {err}");
            assert_eq!(
                *provider.requested.lock().unwrap(),
                vec![Some("hosted".to_string())],
                "{label}"
            );
            mock.assert_closed(0).await;
        }
    }
}

#[tokio::test]
async fn the_runner_closes_a_paused_response_when_tool_execution_fails() {
    let function_call = function_call_item(
        "fc_tool_failure",
        "call_tool_failure",
        "alpha",
        "/root/researcher",
    );
    let recovered = root_final_message("recovered");
    let mock = mock(vec![
        vec![created("resp_tool_failure"), done(&function_call, 2, 0)],
        vec![
            created("resp_recovered"),
            done(&recovered, 2, 0),
            completed("resp_recovered", vec![recovered.clone()], 3),
        ],
    ])
    .await;
    let failing = FunctionTool::new(
        "lookup_document",
        "Look up a document",
        |_ctx, a: SectionArgs| async move {
            Err::<String, _>(AgentsError::user(format!("Unable to read {}", a.section)))
        },
    )
    .without_failure_error_function();
    let agent = Agent::<()>::new("SDK root")
        .with_model_instance(Arc::new(model_for(&mock)))
        .with_tool(failing);

    let err = with_timeout(Runner::run_with_options(
        &agent,
        "Inspect alpha.",
        options(),
    ))
    .await
    .unwrap_err();
    assert!(err.to_string().contains("Unable to read alpha"), "{err}");
    mock.assert_closed(0).await;

    // The failed run released the paused response, so the next run connects afresh.
    let result = with_timeout(Runner::run_with_options(&agent, "Try again.", options()))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("recovered"));
    assert_eq!(mock.connect_count(), 2);
    mock.assert_closed(1).await;
}

#[tokio::test]
async fn a_concurrent_run_does_not_consume_or_close_a_paused_response() {
    for streamed in [false, true] {
        let function_call = function_call_item(
            "fc_concurrent",
            "call_concurrent",
            "alpha",
            "/root/researcher",
        );
        let final_message = root_final_message("continued");
        let mock = mock(vec![tool_flow_events(
            "resp_concurrent",
            &[function_call],
            &[final_message],
        )])
        .await;
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let (tool_started, tool_release) = (started.clone(), release.clone());
        let tool = FunctionTool::new(
            "lookup_document",
            "Look up a document",
            move |_ctx, a: SectionArgs| {
                let (started, release) = (tool_started.clone(), tool_release.clone());
                async move {
                    started.notify_one();
                    release.notified().await;
                    Ok(format!("document:{}", a.section))
                }
            },
        );
        let agent = Arc::new(
            Agent::<()>::new("SDK root")
                .with_model_instance(Arc::new(model_for(&mock)))
                .with_tool(tool),
        );

        let first_agent = agent.clone();
        let first = tokio::spawn(async move {
            if streamed {
                let (result, _raw, error) =
                    run_streamed_to_end(&first_agent, "Inspect alpha.", options()).await;
                assert!(error.is_none(), "{error:?}");
                result.final_output_text()
            } else {
                Runner::run_with_options(&first_agent, "Inspect alpha.", options())
                    .await
                    .unwrap()
                    .final_output_text()
            }
        });
        with_timeout(started.notified()).await;

        let err = if streamed {
            let (_result, _raw, error) =
                run_streamed_to_end(&agent, "Inspect beta.", options()).await;
            error.expect("second run must be refused")
        } else {
            with_timeout(Runner::run_with_options(&agent, "Inspect beta.", options()))
                .await
                .unwrap_err()
        };
        assert!(
            err.to_string().contains("another agent run"),
            "streamed={streamed}: {err}"
        );
        assert_eq!(mock.connect_count(), 1);
        mock.assert_still_open(0);

        release.notify_one();
        let text = with_timeout(first).await.unwrap();
        assert_eq!(text.as_deref(), Some("continued"), "streamed={streamed}");
        mock.assert_closed(0).await;
    }
}

// ---------------------------------------------------------------------------------------------
// response normalization
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn non_root_and_unknown_items_are_filtered_from_the_model_response() {
    let output = vec![
        subagent_message("working"),
        json!({"type": "future_beta_item", "id": "future_1"}),
        root_final_message("done"),
    ];
    let mut events = vec![created("resp_1")];
    for (index, item) in output.iter().enumerate() {
        events.push(done(item, index as u64 + 2, index as u64));
    }
    events.push(completed("resp_1", output.clone(), 5));
    let mock = mock(vec![events]).await;
    let model = model_for(&mock);

    let response = with_timeout(model.get_response(request("hello", ModelSettings::default())))
        .await
        .unwrap();

    assert_eq!(response.output.len(), 1);
    let item = serde_json::to_value(&response.output[0]).unwrap();
    assert_eq!(item["type"], json!("message"));
    assert!(get_hosted_agent_metadata(&item).is_some());
}

#[tokio::test]
async fn beta_usage_details_are_converted_to_stable_usage_types() {
    let final_message = root_final_message("usage");
    let mock = mock(vec![vec![
        created("resp_usage"),
        done(&final_message, 2, 0),
        completed_with_usage(
            "resp_usage",
            vec![final_message.clone()],
            3,
            usage(12, 3, 1, 4, 2),
        ),
    ]])
    .await;
    let model = model_for(&mock);

    let response = with_timeout(model.get_response(request("hello", ModelSettings::default())))
        .await
        .unwrap();

    assert_eq!(response.usage.input_tokens, 12);
    assert_eq!(response.usage.input_tokens_details.cached_tokens, 3);
    assert_eq!(response.usage.output_tokens_details.reasoning_tokens, 2);
}

/// Python asserts the stream keeps the *raw* hosted subagent `output_item.done` event while the
/// typed done events and the completed response carry only the root message.
#[tokio::test]
async fn the_stream_preserves_hosted_raw_events_and_normalizes_the_root_output() {
    let output = vec![subagent_message("working"), root_final_message("done")];
    let mock = mock(vec![vec![
        created("resp_stream"),
        done(&output[0], 2, 0),
        done(&output[1], 3, 1),
        completed("resp_stream", output.clone(), 4),
    ]])
    .await;
    let model = model_for(&mock);

    let stream = with_timeout(model.stream_response(request("hello", ModelSettings::default())))
        .await
        .unwrap();
    let events: Vec<Value> = with_timeout(stream.collect::<Vec<_>>())
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();

    let done_events: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "response.output_item.done")
        .collect();
    let typed: Vec<&&Value> = done_events
        .iter()
        .filter(|e| e["item"]["type"] == "message" && e["item"]["phase"] == "final_answer")
        .collect();
    assert_eq!(typed.len(), 1);
    let raw_subagent = done_events
        .iter()
        .find(|e| e["item"]["agent"]["agent_name"] == "/root/researcher");
    let raw_subagent = raw_subagent.expect("the raw subagent done event must be preserved");
    assert!(get_hosted_agent_metadata(&raw_subagent["item"]).is_some());

    let completed_event = events
        .iter()
        .find(|e| e["type"] == "response.completed")
        .unwrap();
    assert_eq!(
        completed_event["response"]["output"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        completed_event["response"]["output"][0]["type"],
        json!("message")
    );
}

#[tokio::test]
async fn a_streamed_run_routes_tools_and_emits_raw_hosted_items() {
    let first_items = vec![
        json!({
            "id": "mac_stream", "type": "multi_agent_call", "call_id": "call_spawn_stream",
            "action": "spawn_agent", "arguments": "{}", "agent": {"agent_name": "/root"},
        }),
        function_call_item("fc_stream", "call_lookup_stream", "beta", "/root/reviewer"),
    ];
    let final_items = vec![
        subagent_message("review complete"),
        root_final_message("stream done"),
    ];
    let mock = mock(vec![tool_flow_events(
        "resp_stream",
        &first_items,
        &final_items,
    )])
    .await;
    let callers = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::<()>::new("SDK root")
        .with_instructions("Delegate and synthesize.")
        .with_model_instance(Arc::new(model_for(&mock)))
        .with_tool(lookup_tool(callers.clone()));

    let (result, raw, error) = run_streamed_to_end(&agent, "Compare beta.", options()).await;

    assert!(error.is_none(), "{error:?}");
    assert_eq!(result.final_output_text().as_deref(), Some("stream done"));
    assert_eq!(*callers.lock().unwrap(), vec!["/root/reviewer"]);
    assert!(
        raw.iter()
            .any(|e| e["type"] == "response.output_item.done"
                && e["item"]["type"] == "multi_agent_call"),
        "the raw hosted multi_agent_call item must reach the stream consumer"
    );
}

// ---------------------------------------------------------------------------------------------
// test_live.py
// ---------------------------------------------------------------------------------------------

/// `test_live.py` drives the beta endpoint on api.openai.com with a real `OPENAI_API_KEY`; it
/// cannot run against a mock, so it is ported as an ignored test that is skipped without a key.
#[tokio::test]
#[ignore = "needs a real OPENAI_API_KEY and network access to the hosted multi-agent beta"]
async fn live_hosted_multi_agent_round_trip() {
    let Ok(key) = std::env::var("OPENAI_API_KEY") else {
        return;
    };
    let model = OpenAIHostedMultiAgentModel::new("gpt-5.6-sol", OpenAIClient::new(key));
    let agent = Agent::<()>::new("SDK root").with_model_instance(Arc::new(model));
    let result = Runner::run_with_options(&agent, "Say hello.", options()).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn streamed_continuation_reports_aggregated_usage_entries() {
    let function_call =
        function_call_item("fc_fallback", "call_fallback", "alpha", "/root/researcher");
    let function_output = json!({
        "type": "function_call_output", "call_id": "call_fallback", "output": "document:alpha",
    });
    let mut first_completed = completed("resp_old", vec![function_call.clone()], 3);
    first_completed["response"]["usage"] = usage(12, 3, 1, 4, 2);
    let final_message = root_final_message("continued");
    let mut second_completed = completed("resp_new", vec![final_message.clone()], 3);
    second_completed["response"]["usage"] = usage(7, 2, 4, 3, 1);
    let mock = mock(vec![vec![
        created("resp_old"),
        done(&function_call, 2, 0),
        first_completed,
        injection_failed(
            "resp_old",
            vec![function_output],
            "response_already_completed",
            "The response already completed.",
        ),
        created("resp_new"),
        done(&final_message, 2, 0),
        second_completed,
    ]])
    .await;
    let agent = Agent::<()>::new("SDK root")
        .with_model_instance(Arc::new(model_for(&mock)))
        .with_tool(plain_lookup_tool());
    let (result, _raw, error) = run_streamed_to_end(&agent, "Inspect alpha.", options()).await;
    assert!(error.is_none(), "{error:?}");
    let usage = result.result().expect("completed run").usage();
    assert_eq!(usage.request_usage_entries.len(), 2);
    assert_eq!(usage.requests, 2);
}
