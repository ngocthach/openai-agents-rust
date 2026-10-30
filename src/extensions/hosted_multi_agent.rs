//! Experimental: OpenAI-hosted multi-agent orchestration (Responses API beta
//! `responses_multi_agent=v1`) over a WebSocket.
//!
//! Port of `agents.extensions.experimental.hosted_multi_agent`. OpenAI runs a tree of hosted
//! subagents; when any of them calls one of your function tools, the model returns a logical
//! "pause" response containing the call. The runner executes the tool and calls the model
//! again, and the output is injected into the still-running hosted response
//! (`response.inject`). Hosted orchestration items (`agent_message`, `multi_agent_call`, ...)
//! and non-final messages are filtered out of the SDK-visible output.
//!
//! Limitations (same as Python): SDK handoffs and tool approval interruptions are not
//! supported, and one model instance must not be shared by concurrent runs.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde_json::{Map, Value, json};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, mpsc};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::models::interface::{Model, ModelRequest, ResponseEventStream};
use crate::models::openai_client::OpenAIClient;
use crate::models::openai_responses::{OpenAIResponsesModel, parse_response};
use crate::models::openai_responses_ws::websocket_url;
use crate::models::response_terminal::terminal_failure_error;
use crate::tool_context::ToolContext;

const BETA_ID: &str = "responses_multi_agent=v1";
const ROOT_AGENT_NAME: &str = "/root";
const HOSTED_ITEM_TYPES: &[&str] = &[
    "agent_message",
    "multi_agent_call",
    "multi_agent_call_output",
];
const STABLE_OUTPUT_TYPES: &[&str] = &[
    "message",
    "file_search_call",
    "function_call",
    "web_search_call",
    "computer_call",
    "reasoning",
    "image_generation_call",
    "code_interpreter_call",
    "local_shell_call",
    "shell_call",
    "shell_call_output",
    "apply_patch_call",
    "apply_patch_call_output",
    "mcp_call",
    "mcp_list_tools",
    "mcp_approval_request",
    "custom_tool_call",
    "tool_search_call",
    "tool_search_output",
    "compaction",
];

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Configuration of the hosted multi-agent beta.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostedMultiAgentConfig {
    /// Maximum active subagents across the hosted tree, excluding the root agent.
    pub max_concurrent_subagents: Option<u32>,
}

/// Hosted-agent attribution attached to a beta response item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostedAgentMetadata {
    /// The hosted agent path (for example `/root` or `/root/researcher`).
    pub agent_name: String,
    /// The item phase (for example `final_answer`).
    pub phase: Option<String>,
}

/// Hosted-agent attribution of a raw item.
pub fn get_hosted_agent_metadata(item: &Value) -> Option<HostedAgentMetadata> {
    let agent_name = item.pointer("/agent/agent_name")?.as_str()?;
    if agent_name.is_empty() {
        return None;
    }
    Some(HostedAgentMetadata {
        agent_name: agent_name.to_string(),
        phase: item
            .get("phase")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// Hosted-agent attribution of the function call a tool is executing.
pub fn get_hosted_agent_metadata_for_tool<C>(ctx: &ToolContext<C>) -> Option<HostedAgentMetadata> {
    let call = ctx.tool_call.as_ref()?;
    get_hosted_agent_metadata(&Value::Object(call.extra.clone()))
}

fn is_root_final_message(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("message")
        && item.pointer("/agent/agent_name").and_then(Value::as_str) == Some(ROOT_AGENT_NAME)
        && item.get("phase").and_then(Value::as_str) == Some("final_answer")
}

fn item_key(item: &Value) -> Option<(String, String)> {
    let ty = item.get("type")?.as_str()?;
    ["id", "call_id"].iter().find_map(|f| {
        item.get(*f)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|id| (ty.to_string(), id.to_string()))
    })
}

/// Keep only SDK-visible output items.
fn normalize_output_item(item: &Value) -> Option<Value> {
    let ty = item.get("type")?.as_str()?;
    if HOSTED_ITEM_TYPES.contains(&ty) {
        return None;
    }
    if ty == "message" && !is_root_final_message(item) {
        return None;
    }
    STABLE_OUTPUT_TYPES.contains(&ty).then(|| item.clone())
}

fn normalize_output(items: &[Value]) -> Vec<Value> {
    items.iter().filter_map(normalize_output_item).collect()
}

fn add_details(a: &Value, b: &Value, key: &str) -> Value {
    let mut merged: Map<String, Value> = a.as_object().cloned().unwrap_or_default();
    if let Some(bm) = b.as_object() {
        for (k, v) in bm {
            merged.insert(k.clone(), v.clone());
        }
    }
    let sum = |field: &str| {
        a.get(field).and_then(Value::as_i64).unwrap_or(0)
            + b.get(field).and_then(Value::as_i64).unwrap_or(0)
    };
    if key == "input" {
        merged.insert("cached_tokens".into(), json!(sum("cached_tokens")));
        merged.insert(
            "cache_write_tokens".into(),
            json!(sum("cache_write_tokens")),
        );
    } else {
        merged.insert("reasoning_tokens".into(), json!(sum("reasoning_tokens")));
    }
    Value::Object(merged)
}

fn merge_usage(previous: Option<&Value>, current: &Value) -> Value {
    let Some(prev) = previous else {
        return current.clone();
    };
    let mut merged = current.clone();
    for f in ["input_tokens", "output_tokens", "total_tokens"] {
        let total = prev.get(f).and_then(Value::as_i64).unwrap_or(0)
            + current.get(f).and_then(Value::as_i64).unwrap_or(0);
        merged[f] = json!(total);
    }
    merged["input_tokens_details"] = add_details(
        prev.get("input_tokens_details").unwrap_or(&Value::Null),
        current.get("input_tokens_details").unwrap_or(&Value::Null),
        "input",
    );
    merged["output_tokens_details"] = add_details(
        prev.get("output_tokens_details").unwrap_or(&Value::Null),
        current.get("output_tokens_details").unwrap_or(&Value::Null),
        "output",
    );
    merged
}

struct PendingInjection {
    call_id: String,
    input_item: Value,
}

struct Active {
    /// The run that opened this paused response, so another run cannot resume it.
    owner: Option<crate::models::run_context::ModelRunOwner>,
    socket: Ws,
    response_id: Option<String>,
    response_template: Option<Value>,
    pending_call_ids: HashSet<String>,
    sent_call_ids: HashSet<String>,
    pending_injections: VecDeque<PendingInjection>,
    delivered_item_keys: HashSet<(String, String)>,
    completed_response: Option<Value>,
    fallback_input: Vec<Value>,
    accumulated_usage: Option<Value>,
    request_count: u64,
    last_sequence_number: i64,
}

/// A Responses model backed by OpenAI-hosted multi-agent orchestration.
#[derive(Clone)]
pub struct OpenAIHostedMultiAgentModel {
    inner: OpenAIResponsesModel,
    client: OpenAIClient,
    /// Beta configuration.
    pub config: HostedMultiAgentConfig,
    websocket_base_url: Option<String>,
    active: Arc<Mutex<Option<Active>>>,
}

impl std::fmt::Debug for OpenAIHostedMultiAgentModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIHostedMultiAgentModel")
            .field("model", &self.inner.model())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn ws_send_err(e: impl std::fmt::Display) -> AgentsError {
    AgentsError::other(format!("Hosted multi-agent websocket error: {e}"))
}

async fn send(socket: &mut Ws, event: &Value) -> Result<()> {
    socket
        .send(Message::text(event.to_string()))
        .await
        .map_err(ws_send_err)
}

async fn recv(socket: &mut Ws) -> Result<Value> {
    loop {
        match socket.next().await {
            None => {
                return Err(AgentsError::other(
                    "Hosted multi-agent websocket closed before a terminal response event.",
                ));
            }
            Some(Err(e)) => return Err(ws_send_err(e)),
            Some(Ok(Message::Text(t))) => return Ok(serde_json::from_str(&t)?),
            Some(Ok(Message::Binary(b))) => return Ok(serde_json::from_slice(&b)?),
            Some(Ok(Message::Close(_))) => {
                return Err(AgentsError::other(
                    "Hosted multi-agent websocket closed before a terminal response event.",
                ));
            }
            Some(Ok(_)) => continue,
        }
    }
}

impl OpenAIHostedMultiAgentModel {
    /// Create the model.
    pub fn new(model: impl Into<String>, client: OpenAIClient) -> Self {
        OpenAIHostedMultiAgentModel {
            inner: OpenAIResponsesModel::new(model, client.clone()),
            client,
            config: HostedMultiAgentConfig::default(),
            websocket_base_url: None,
            active: Arc::new(Mutex::new(None)),
        }
    }

    /// Set the beta configuration.
    pub fn with_config(mut self, config: HostedMultiAgentConfig) -> Result<Self> {
        if config.max_concurrent_subagents == Some(0) {
            return Err(AgentsError::user(
                "max_concurrent_subagents must be a positive integer or None.",
            ));
        }
        self.config = config;
        Ok(self)
    }

    /// Override the WebSocket base URL.
    pub fn with_websocket_base_url(mut self, url: impl Into<String>) -> Self {
        self.websocket_base_url = Some(url.into());
        self
    }

    fn validate(&self, request: &ModelRequest) -> Result<()> {
        if !request.handoffs.is_empty() {
            return Err(AgentsError::user(
                "OpenAI hosted multi-agent cannot be combined with SDK handoffs. Use local \
                 function tools or agents-as-tools instead.",
            ));
        }
        let s = &request.model_settings;
        let extra_args = s.extra_args.as_ref();
        let extra_body = s.extra_body.as_ref();
        for key in ["multi_agent", "betas"] {
            if extra_args.is_some_and(|m| m.contains_key(key))
                || extra_body.and_then(|b| b.get(key)).is_some()
            {
                return Err(AgentsError::user(format!(
                    "Configure '{key}' through OpenAIHostedMultiAgentModel, not ModelSettings."
                )));
            }
        }
        if extra_args.is_some_and(|m| m.contains_key("max_tool_calls"))
            || extra_body.and_then(|b| b.get("max_tool_calls")).is_some()
        {
            return Err(AgentsError::user(
                "max_tool_calls is not supported by the hosted multi-agent beta.",
            ));
        }
        if s.reasoning.as_ref().is_some_and(|r| r.summary.is_some()) {
            return Err(AgentsError::user(
                "reasoning.summary is not supported by the hosted multi-agent beta.",
            ));
        }
        Ok(())
    }

    /// Build the `response.create` frame.
    pub fn build_frame(&self, request: &ModelRequest) -> Result<Value> {
        self.validate(request)?;
        let mut frame = self.inner.build_request_body(request, false)?;
        if let Value::Object(m) = &mut frame {
            m.remove("stream");
            m.remove("betas");
        }
        let mut multi_agent = json!({"enabled": true});
        if let Some(n) = self.config.max_concurrent_subagents {
            multi_agent["max_concurrent_subagents"] = json!(n);
        }
        frame["multi_agent"] = multi_agent;
        frame["type"] = json!("response.create");
        Ok(frame)
    }

    async fn connect(&self, request: &ModelRequest) -> Result<Ws> {
        let base = self
            .websocket_base_url
            .clone()
            .unwrap_or_else(|| websocket_url(self.client.base_url()));
        let mut url = format!("{}/responses", base.trim_end_matches('/'));
        if let Some(q) = request
            .model_settings
            .extra_query
            .as_ref()
            .filter(|q| !q.is_empty())
        {
            let mut pairs: Vec<_> = q.iter().map(|(k, v)| format!("{k}={v}")).collect();
            pairs.sort();
            url = format!("{url}?{}", pairs.join("&"));
        }
        let mut extra: HashMap<String, String> = request
            .model_settings
            .extra_headers
            .clone()
            .unwrap_or_default()
            .into_iter()
            .filter(|(k, _)| !k.eq_ignore_ascii_case("openai-beta"))
            .collect();
        extra.insert("OpenAI-Beta".into(), BETA_ID.into());
        let mut req = url
            .into_client_request()
            .map_err(|e| AgentsError::user(format!("Invalid hosted multi-agent URL: {e}")))?;
        for (k, v) in self.client.headers(Some(&extra))?.iter() {
            req.headers_mut().insert(k.clone(), v.clone());
        }
        crate::util::ensure_crypto_provider();
        let (socket, _) = tokio_tungstenite::connect_async(req)
            .await
            .map_err(|e| AgentsError::other(format!("Hosted multi-agent connect failed: {e}")))?;
        Ok(socket)
    }

    /// Close any paused hosted response.
    pub async fn close_active(&self) {
        if let Some(mut a) = self.active.lock().await.take() {
            let _ = a.socket.close(None).await;
        }
    }

    async fn inject_outputs(active: &mut Active, request: &ModelRequest) -> Result<()> {
        let unsent: HashSet<String> = active
            .pending_call_ids
            .difference(&active.sent_call_ids)
            .cloned()
            .collect();
        if unsent.is_empty() {
            return Ok(());
        }
        let outputs: Vec<Value> = request
            .input
            .iter()
            .map(|i| i.0.clone())
            .filter(|v| {
                v.get("type").and_then(Value::as_str) == Some("function_call_output")
                    && v.get("call_id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| unsent.contains(id))
            })
            .collect();
        let have: HashSet<&str> = outputs
            .iter()
            .filter_map(|o| o.get("call_id").and_then(Value::as_str))
            .collect();
        let mut missing: Vec<&String> = unsent
            .iter()
            .filter(|id| !have.contains(id.as_str()))
            .collect();
        if !missing.is_empty() {
            missing.sort();
            return Err(AgentsError::user(format!(
                "OpenAIHostedMultiAgentModel has an active response waiting for function outputs, \
                 but the next model input did not contain outputs for: {}.",
                missing
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        for output in outputs {
            let call_id = output["call_id"].as_str().unwrap_or_default().to_string();
            send(
                &mut active.socket,
                &json!({"type": "response.inject", "response_id": active.response_id,
                        "input": [output]}),
            )
            .await?;
            active.sent_call_ids.insert(call_id.clone());
            active.pending_injections.push_back(PendingInjection {
                call_id,
                input_item: output,
            });
        }
        Ok(())
    }

    /// Run one logical model turn, sending Responses-style events to `tx`.
    async fn run_turn(&self, request: ModelRequest, tx: mpsc::Sender<Result<Value>>) -> Result<()> {
        let frame = self.build_frame(&request)?;
        let owner = crate::models::run_context::get_model_run_owner();
        let mut guard = self.active.clone().lock_owned().await;
        if let Some(active) = guard.as_ref()
            && active.owner.is_some()
            && active.owner != owner
        {
            return Err(AgentsError::user(
                "OpenAIHostedMultiAgentModel already has a paused response owned by another \
                 agent run. Use a separate model instance for concurrent runs.",
            ));
        }
        if guard.is_none() {
            let mut socket = self.connect(&request).await?;
            send(&mut socket, &frame).await?;
            *guard = Some(Active {
                owner,
                socket,
                response_id: None,
                response_template: None,
                pending_call_ids: HashSet::new(),
                sent_call_ids: HashSet::new(),
                pending_injections: VecDeque::new(),
                delivered_item_keys: HashSet::new(),
                completed_response: None,
                fallback_input: Vec::new(),
                accumulated_usage: None,
                request_count: 1,
                last_sequence_number: 0,
            });
        } else if let Err(e) = Self::inject_outputs(guard.as_mut().expect("active"), &request).await
        {
            if let Some(mut a) = guard.take() {
                let _ = a.socket.close(None).await;
            }
            return Err(e);
        }

        let result = Self::drive(guard.as_mut().expect("active"), &frame, &tx).await;
        match result {
            Ok(true) => Ok(()), // paused on a function call: keep the response open
            Ok(false) | Err(_) => {
                if let Some(mut a) = guard.take() {
                    let _ = a.socket.close(None).await;
                }
                result.map(|_| ())
            }
        }
    }

    /// Read events until a pause (returns true) or a terminal event (returns false).
    async fn drive(
        active: &mut Active,
        frame: &Value,
        tx: &mpsc::Sender<Result<Value>>,
    ) -> Result<bool> {
        let mut current_output: Vec<Value> = Vec::new();
        loop {
            if active.completed_response.is_some() && active.pending_injections.is_empty() {
                let completed = active.completed_response.take().expect("completed");
                let response = completed.get("response").cloned().unwrap_or(Value::Null);
                if !active.fallback_input.is_empty() {
                    // An injection raced with completion: continue from the completed response.
                    let id = response
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            AgentsError::user(
                                "Hosted multi-agent could not continue after a completed \
                                 response injection.",
                            )
                        })?
                        .to_string();
                    if let Some(u) = response.get("usage").filter(|u| !u.is_null()) {
                        active.accumulated_usage =
                            Some(merge_usage(active.accumulated_usage.as_ref(), u));
                    }
                    let mut next = frame.clone();
                    next["input"] = Value::Array(std::mem::take(&mut active.fallback_input));
                    if next.get("conversation").is_some_and(|c| !c.is_null()) {
                        if let Value::Object(m) = &mut next {
                            m.remove("previous_response_id");
                        }
                    } else {
                        next["previous_response_id"] = json!(id);
                    }
                    active.response_id = None;
                    active.response_template = None;
                    active.pending_call_ids.clear();
                    active.sent_call_ids.clear();
                    active.pending_injections.clear();
                    active.delivered_item_keys.clear();
                    active.request_count += 1;
                    active.last_sequence_number = 0;
                    current_output.clear();
                    send(&mut active.socket, &next).await?;
                    continue;
                }
                let mut normalized = response.clone();
                let mut output: Vec<Value> = response
                    .get("output")
                    .and_then(Value::as_array)
                    .cloned()
                    .ok_or_else(|| {
                        AgentsError::user(
                            "Hosted multi-agent response did not contain an output list.",
                        )
                    })?;
                if output.is_empty() && !current_output.is_empty() {
                    output = current_output.clone();
                }
                output.retain(|i| {
                    item_key(i).is_none_or(|k| !active.delivered_item_keys.contains(&k))
                });
                normalized["output"] = Value::Array(normalize_output(&output));
                let usage = match response.get("usage").filter(|u| !u.is_null()) {
                    Some(u) => Some(merge_usage(active.accumulated_usage.as_ref(), u)),
                    None => active.accumulated_usage.clone(),
                };
                if let Some(mut u) = usage {
                    if active.request_count > 1 {
                        u["requests"] = json!(active.request_count);
                    }
                    normalized["usage"] = u;
                }
                let mut event = completed.clone();
                event["response"] = normalized;
                let _ = tx.send(Ok(event)).await;
                return Ok(false);
            }

            let mut event = recv(&mut active.socket).await?;
            let ty = event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if let Some(n) = event.get("sequence_number").and_then(Value::as_i64) {
                active.last_sequence_number = n;
            }
            match ty.as_str() {
                "response.created" => {
                    let response = event.get("response").cloned().unwrap_or(Value::Null);
                    let id = response
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            AgentsError::user(
                                "Hosted multi-agent response.created did not contain a response ID.",
                            )
                        })?;
                    active.response_id = Some(id.to_string());
                    active.response_template = Some(response);
                }
                "response.inject.created" => {
                    let pending = active.pending_injections.pop_front().ok_or_else(|| {
                        AgentsError::user(
                            "Hosted multi-agent received response.inject.created without a \
                             pending injection.",
                        )
                    })?;
                    active.pending_call_ids.remove(&pending.call_id);
                    active.sent_call_ids.remove(&pending.call_id);
                }
                "response.inject.failed" => {
                    let pending = active.pending_injections.pop_front().ok_or_else(|| {
                        AgentsError::user(
                            "Hosted multi-agent received response.inject.failed without a \
                             pending injection.",
                        )
                    })?;
                    active.pending_call_ids.remove(&pending.call_id);
                    active.sent_call_ids.remove(&pending.call_id);
                    let code = event.pointer("/error/code").and_then(Value::as_str);
                    if code != Some("response_already_completed") {
                        return Err(AgentsError::user(match code {
                            Some(c) => format!(
                                "Hosted multi-agent function output injection failed with code '{c}'."
                            ),
                            None => "Hosted multi-agent function output injection failed.".into(),
                        }));
                    }
                    match event.get("input").and_then(Value::as_array) {
                        Some(items) => active.fallback_input.extend(items.iter().cloned()),
                        None => active.fallback_input.push(pending.input_item),
                    }
                }
                "response.completed" => {
                    active.completed_response = Some(event);
                    continue;
                }
                "response.failed" | "response.incomplete" | "error" | "response.error" => {
                    if let Some(r) = event.get("response").cloned() {
                        let mut r = r;
                        if let Some(out) = r.get("output").and_then(Value::as_array) {
                            r["output"] = Value::Array(normalize_output(out));
                        }
                        event["response"] = r;
                    }
                    return Err(terminal_failure_error(&ty, &event));
                }
                _ => {}
            }

            // Forward the event, dropping hosted-only items.
            let is_item_event = matches!(
                ty.as_str(),
                "response.output_item.added" | "response.output_item.done"
            );
            let visible =
                !is_item_event || event.get("item").and_then(normalize_output_item).is_some();
            if visible && tx.send(Ok(event.clone())).await.is_err() {
                return Err(AgentsError::other("Hosted multi-agent stream was dropped."));
            }
            if ty != "response.output_item.done" {
                continue;
            }
            let Some(item) = event.get("item").cloned() else {
                continue;
            };
            current_output.push(item.clone());
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                continue;
            }
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    AgentsError::user("Hosted multi-agent function call did not contain a call ID.")
                })?;
            active.pending_call_ids.insert(call_id.to_string());
            for o in &current_output {
                if let Some(k) = item_key(o) {
                    active.delivered_item_keys.insert(k);
                }
            }
            let (Some(template), Some(id)) = (&active.response_template, &active.response_id)
            else {
                return Err(AgentsError::user(
                    "Hosted multi-agent received a function call before response.created.",
                ));
            };
            let mut pause = template.clone();
            pause["id"] = json!(id);
            pause["status"] = json!("completed");
            pause["usage"] = Value::Null;
            pause["output"] = Value::Array(normalize_output(&current_output));
            let _ = tx
                .send(Ok(json!({
                    "type": "response.completed",
                    "sequence_number": active.last_sequence_number,
                    "response": pause,
                    "hosted_multi_agent_pause": true,
                })))
                .await;
            return Ok(true);
        }
    }
}

#[async_trait]
impl Model for OpenAIHostedMultiAgentModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        let preserve = request.model_settings.preserve_raw_usage.unwrap_or(false);
        let mut stream = self.stream_response(request).await?;
        let mut last: Option<Value> = None;
        while let Some(ev) = stream.next().await {
            let ev = ev?;
            if ev.get("type").and_then(Value::as_str) == Some("response.completed") {
                last = ev.get("response").cloned();
            }
        }
        let response = last.ok_or_else(|| {
            AgentsError::user(
                "Hosted multi-agent WebSocket turn ended without a logical or terminal response.",
            )
        })?;
        let mut parsed = parse_response(&response, None, preserve)?;
        if let Some(n) = response.pointer("/usage/requests").and_then(Value::as_u64) {
            parsed.usage.requests = n;
        }
        Ok(parsed)
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let (tx, rx) = mpsc::channel(64);
        let this = self.clone();
        tokio::spawn(async move {
            let err_tx = tx.clone();
            if let Err(e) = this.run_turn(request, tx).await {
                let _ = err_tx.send(Err(e)).await;
            }
        });
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }

    fn model_name(&self) -> Option<String> {
        Some(self.inner.model().to_string())
    }

    async fn cleanup_on_run_end(&self, owner: crate::models::run_context::ModelRunOwner) {
        let mut guard = self.active.lock().await;
        if guard.as_ref().is_some_and(|a| a.owner == Some(owner))
            && let Some(mut a) = guard.take()
        {
            let _ = a.socket.close(None).await;
        }
    }

    async fn close(&self) {
        self.close_active().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_hosted_items_and_merges_usage() {
        let items = vec![
            json!({"type": "agent_message", "id": "a"}),
            json!({"type": "message", "id": "m1", "agent": {"agent_name": "/root/sub"}}),
            json!({"type": "message", "id": "m2", "agent": {"agent_name": "/root"}, "phase": "final_answer"}),
            json!({"type": "function_call", "call_id": "c"}),
        ];
        let out = normalize_output(&items);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["id"], "m2");
        let meta = get_hosted_agent_metadata(&items[2]).unwrap();
        assert_eq!(meta.agent_name, "/root");
        assert_eq!(meta.phase.as_deref(), Some("final_answer"));
        let u = merge_usage(
            Some(
                &json!({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
                "input_tokens_details": {"cached_tokens": 1}, "output_tokens_details": {"reasoning_tokens": 1}}),
            ),
            &json!({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 3}}),
        );
        assert_eq!(u["total_tokens"], 33);
        assert_eq!(u["input_tokens_details"]["cached_tokens"], 3);
        assert_eq!(u["output_tokens_details"]["reasoning_tokens"], 4);
    }
}
