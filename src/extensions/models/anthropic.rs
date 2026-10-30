//! Anthropic's native Messages API (`POST /v1/messages`).
//!
//! any-llm talks to Anthropic through its own protocol rather than an OpenAI-compatible shim,
//! and so does this model. It is the transport to use when a run depends on anything the
//! OpenAI-compatible route cannot carry: signed interleaved-thinking blocks, `redacted_thinking`
//! blocks, prompt caching markers, or Anthropic-only stop reasons.
//!
//! [`crate::extensions::models::LitellmModel`] and [`crate::extensions::models::AnyLLMModel`]
//! keep using the OpenAI-compatible route for `anthropic/...`; pass
//! [`AnthropicMessagesModel`] directly (or use the `anthropic_native/...` prefix) to get this
//! one.
//!
//! ```no_run
//! use std::sync::Arc;
//! use openai_agents_sdk::Agent;
//! use openai_agents_sdk::extensions::models::anthropic::AnthropicMessagesModel;
//!
//! # fn demo() -> openai_agents_sdk::Result<()> {
//! // Uses ANTHROPIC_API_KEY.
//! let model = AnthropicMessagesModel::from_env("claude-sonnet-4-5")?;
//! let agent = Agent::<()>::new("Assistant").with_model_instance(Arc::new(model));
//! # Ok(()) }
//! ```

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::models::FAKE_RESPONSES_ID;
use crate::models::interface::{Model, ModelRequest, ResponseEventStream, ToolSpec};
use crate::models::openai_client::OpenAIClient;
use crate::models::openai_responses::parse_response;
use crate::tracing::{SpanData, generation_span, record_model_error_on_span};
use crate::usage::Usage;

/// Default Anthropic API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

/// The `anthropic-version` header every request carries.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic requires `max_tokens`; this is used when the agent did not set one.
pub const DEFAULT_MAX_TOKENS: u64 = 4096;

/// A model that calls Anthropic's native Messages API.
#[derive(Debug, Clone)]
pub struct AnthropicMessagesModel {
    model: String,
    client: OpenAIClient,
    default_max_tokens: u64,
}

impl AnthropicMessagesModel {
    /// Build a model from an API key.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let client = OpenAIClient::new(String::new())
            .with_base_url(
                std::env::var("ANTHROPIC_BASE_URL")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
                    .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            )
            .with_bearer_auth(false)
            .with_header("x-api-key", api_key)
            .with_header("anthropic-version", ANTHROPIC_VERSION);
        Self::with_client(model, client)
    }

    /// Build a model from `ANTHROPIC_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| {
                AgentsError::user(
                    "ANTHROPIC_API_KEY must be set to use the Anthropic Messages API.",
                )
            })?;
        Ok(Self::new(model, key))
    }

    /// Build a model on a pre-configured client (a proxy, custom headers, a signing hook).
    ///
    /// The client must already carry whatever authentication the endpoint needs; the base URL is
    /// used verbatim and `/messages` is appended.
    pub fn with_client(model: impl Into<String>, client: OpenAIClient) -> Self {
        AnthropicMessagesModel {
            model: model.into(),
            client,
            default_max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    /// `max_tokens` to send when the agent's settings do not specify one.
    pub fn with_default_max_tokens(mut self, max_tokens: u64) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// The model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The HTTP client this model uses.
    pub fn client(&self) -> &OpenAIClient {
        &self.client
    }

    /// Build the `/messages` request body for a request.
    pub fn build_request_body(&self, req: &ModelRequest, stream: bool) -> Result<Value> {
        let settings = &req.model_settings;
        let mut body = json!({
            "model": self.model,
            "max_tokens": settings.max_tokens.map(u64::from).unwrap_or(self.default_max_tokens),
            "messages": items_to_messages(&req.input)?,
        });
        if let Some(system) = &req.system_instructions {
            body["system"] = json!(system);
        }
        if stream {
            body["stream"] = json!(true);
        }
        for (key, value) in [
            ("temperature", settings.temperature.map(|v| json!(v))),
            ("top_p", settings.top_p.map(|v| json!(v))),
        ] {
            if let Some(v) = value {
                body[key] = v;
            }
        }
        // Anthropic takes a thinking budget, not an effort level.
        if let Some(reasoning) = &settings.reasoning
            && let Some(effort) = reasoning.effort.as_deref()
            && let Some(budget) = thinking_budget(
                effort,
                settings.max_tokens.map(u64::from),
                self.default_max_tokens,
            )
        {
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
        }
        let mut tools: Vec<Value> = Vec::new();
        for tool in &req.tools {
            match tool {
                ToolSpec::Function(f) => tools.push(json!({
                    "name": f.name,
                    "description": f.description,
                    "input_schema": f.parameters,
                })),
                ToolSpec::Hosted { param, .. } => {
                    return Err(AgentsError::user(format!(
                        "Hosted tools are not supported by the Anthropic Messages API. Got tool \
                         type: {}",
                        param
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown")
                    )));
                }
            }
        }
        for handoff in &req.handoffs {
            tools.push(json!({
                "name": handoff.name,
                "description": handoff.description,
                "input_schema": handoff.parameters,
            }));
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            if let Some(choice) = convert_tool_choice(settings.tool_choice.as_ref()) {
                body["tool_choice"] = choice;
            }
        }
        if let Some(extra) = &settings.extra_args {
            let Some(object) = body.as_object_mut() else {
                unreachable!("body is an object");
            };
            for (k, v) in extra {
                if object.contains_key(k) {
                    return Err(AgentsError::user(format!(
                        "messages.create() got multiple values for keyword argument '{k}'"
                    )));
                }
                object.insert(k.clone(), v.clone());
            }
        }
        Ok(body)
    }

    fn span(&self, req: &ModelRequest, body: &Value) -> crate::tracing::Span {
        let span = generation_span(
            req.tracing
                .include_data()
                .then(|| body.get("messages").cloned().unwrap_or(Value::Null)),
            Some(self.model.clone()),
            Some(crate::models::model_config_for_trace(
                &req.model_settings,
                req.tracing,
                Some(self.client.base_url()),
            )),
            req.tracing.is_disabled(),
        );
        span.start();
        span
    }
}

/// Map a reasoning effort onto Anthropic's thinking budget.
fn thinking_budget(effort: &str, max_tokens: Option<u64>, default_max: u64) -> Option<u64> {
    let max = max_tokens.unwrap_or(default_max);
    let fraction = match effort {
        "none" => return None,
        "minimal" => 0.1,
        "low" => 0.25,
        "medium" => 0.5,
        _ => 0.8,
    };
    // Anthropic requires `budget_tokens` to be at least 1024 and below `max_tokens`.
    let budget = ((max as f64) * fraction) as u64;
    Some(budget.clamp(1024, max.saturating_sub(1).max(1024)))
}

fn convert_tool_choice(choice: Option<&crate::model_settings::ToolChoice>) -> Option<Value> {
    let crate::model_settings::ToolChoice::Named(name) = choice? else {
        // A hosted MCP tool choice has no Anthropic equivalent.
        return None;
    };
    Some(match name.as_str() {
        "auto" => json!({"type": "auto"}),
        "required" => json!({"type": "any"}),
        "none" => json!({"type": "none"}),
        name => json!({"type": "tool", "name": name}),
    })
}

/// Convert one Responses content part into an Anthropic content block.
fn convert_part(part: &Value) -> Result<Value> {
    match part.get("type").and_then(Value::as_str).unwrap_or_default() {
        "input_text" | "text" | "output_text" => Ok(json!({
            "type": "text",
            "text": part.get("text").cloned().unwrap_or(Value::String(String::new())),
        })),
        "input_image" => {
            let Some(url) = part.get("image_url").and_then(Value::as_str) else {
                return Err(AgentsError::user(
                    "Only image URLs and data URLs are supported for input_image with the \
                     Anthropic Messages API",
                ));
            };
            // A data URL carries the bytes inline; anything else is fetched by Anthropic.
            if let Some(rest) = url.strip_prefix("data:")
                && let Some((media_type, data)) = rest.split_once(";base64,")
            {
                return Ok(json!({"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": data}}));
            }
            Ok(json!({"type": "image", "source": {"type": "url", "url": url}}))
        }
        "refusal" => Ok(json!({
            "type": "text",
            "text": part.get("refusal").cloned().unwrap_or(Value::String(String::new())),
        })),
        other => Err(AgentsError::user(format!(
            "Unknown content part for the Anthropic Messages API: {other}"
        ))),
    }
}

fn content_blocks(content: &Value) -> Result<Vec<Value>> {
    match content {
        Value::String(s) => Ok(vec![json!({"type": "text", "text": s})]),
        Value::Array(parts) => parts.iter().map(convert_part).collect(),
        Value::Null => Ok(Vec::new()),
        other => Ok(vec![json!({"type": "text", "text": other.to_string()})]),
    }
}

/// The thinking blocks a reasoning item should replay, native list first.
fn thinking_blocks_of(item: &Value) -> Vec<Value> {
    if let Some(blocks) = item
        .pointer("/provider_data/thinking_blocks")
        .and_then(Value::as_array)
        .filter(|b| !b.is_empty())
    {
        return blocks.clone();
    }
    let mut signatures: std::collections::VecDeque<&str> = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .map(|c| c.split('\n').collect())
        .unwrap_or_default();
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|p| p.get("type").and_then(Value::as_str) == Some("reasoning_text"))
        .map(|p| {
            let mut block = json!({
                "type": "thinking",
                "thinking": p.get("text").cloned().unwrap_or(Value::String(String::new())),
            });
            if let Some(sig) = signatures.pop_front() {
                block["signature"] = json!(sig);
            }
            block
        })
        .collect()
}

/// Convert Responses input items into Anthropic messages.
///
/// Anthropic alternates `user` and `assistant` turns and carries tool results as `tool_result`
/// blocks on a user turn, so consecutive items of the same role are merged.
pub fn items_to_messages(items: &[crate::items::InputItem]) -> Result<Vec<Value>> {
    let mut messages: Vec<Value> = Vec::new();
    let mut pending_thinking: Vec<Value> = Vec::new();

    fn push(messages: &mut Vec<Value>, role: &str, mut blocks: Vec<Value>) {
        if blocks.is_empty() {
            return;
        }
        if let Some(last) = messages.last_mut()
            && last.get("role").and_then(Value::as_str) == Some(role)
            && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
        {
            content.append(&mut blocks);
            return;
        }
        messages.push(json!({"role": role, "content": blocks}));
    }

    for item in items {
        let v = &item.0;
        let ty = v.get("type").and_then(Value::as_str);
        let role = v.get("role").and_then(Value::as_str);
        match (ty, role) {
            (None | Some("message"), Some("user")) => {
                pending_thinking.clear();
                push(
                    &mut messages,
                    "user",
                    content_blocks(v.get("content").unwrap_or(&Value::Null))?,
                );
            }
            (None | Some("message"), Some("system" | "developer")) => {
                // Anthropic takes system instructions out of band; fold a stray system message
                // into the conversation as a user turn so its content is not silently lost.
                pending_thinking.clear();
                push(
                    &mut messages,
                    "user",
                    content_blocks(v.get("content").unwrap_or(&Value::Null))?,
                );
            }
            (None | Some("message"), Some("assistant")) => {
                let mut blocks = std::mem::take(&mut pending_thinking);
                blocks.extend(content_blocks(v.get("content").unwrap_or(&Value::Null))?);
                push(&mut messages, "assistant", blocks);
            }
            (Some("reasoning"), _) => {
                pending_thinking = thinking_blocks_of(v);
            }
            (Some("function_call"), _) => {
                let arguments = v.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                let input: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
                let mut blocks = std::mem::take(&mut pending_thinking);
                blocks.push(json!({
                    "type": "tool_use",
                    "id": v.get("call_id").cloned().unwrap_or(Value::Null),
                    "name": v.get("name").cloned().unwrap_or(Value::Null),
                    "input": input,
                }));
                push(&mut messages, "assistant", blocks);
            }
            (Some("function_call_output"), _) => {
                pending_thinking.clear();
                let Some(call_id) = v.get("call_id").filter(|c| !c.is_null()).cloned() else {
                    return Err(AgentsError::user(
                        "Unpaired function outputs are supported by Responses but cannot be \
                         converted to Anthropic tool_result blocks. Use a Responses model to \
                         preserve this input.",
                    ));
                };
                let output = v.get("output").cloned().unwrap_or(Value::Null);
                let content = match &output {
                    Value::String(s) => json!(s),
                    other => json!(content_blocks(other)?),
                };
                push(
                    &mut messages,
                    "user",
                    vec![json!({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": content,
                    })],
                );
            }
            (Some(other), _) => {
                log::debug!(target: crate::logger::LOG_TARGET, "Skipping item type {other} not supported by Anthropic Messages");
            }
            (None, _) => {
                return Err(AgentsError::user(format!("Unhandled input item: {v}")));
            }
        }
    }
    Ok(messages)
}

fn usage_of(value: &Value) -> Usage {
    let field = |k: &str| value.get(k).and_then(Value::as_u64).unwrap_or(0);
    let input = field("input_tokens");
    let output = field("output_tokens");
    let cached = field("cache_read_input_tokens");
    let cache_write = field("cache_creation_input_tokens");
    Usage {
        requests: 1,
        input_tokens: input + cached + cache_write,
        output_tokens: output,
        total_tokens: input + cached + cache_write + output,
        input_tokens_details: crate::usage::InputTokensDetails {
            cached_tokens: cached,
            cache_write_tokens: cache_write,
        },
        ..Default::default()
    }
}

/// Convert an Anthropic message payload into the Responses `response` shape.
pub fn message_to_response(message: &Value, model: &str) -> Value {
    let mut output: Vec<Value> = Vec::new();
    let mut text = String::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for block in message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "text" => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or("")),
            "thinking" | "redacted_thinking" => thinking_blocks.push(block.clone()),
            "tool_use" => {
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "type": "function_call",
                    "id": FAKE_RESPONSES_ID,
                    "call_id": block.get("id").cloned().unwrap_or(Value::Null),
                    "name": block.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": input.to_string(),
                }));
            }
            _ => {}
        }
    }

    if !thinking_blocks.is_empty() {
        let mut thinking_text = String::new();
        let mut last_signature: Option<String> = None;
        for block in &thinking_blocks {
            if block.get("type").and_then(Value::as_str) != Some("thinking") {
                continue;
            }
            thinking_text.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
            if let Some(sig) = block.get("signature").and_then(Value::as_str) {
                last_signature = Some(sig.to_string());
            }
        }
        let mut item = json!({
            "type": "reasoning", "id": FAKE_RESPONSES_ID, "summary": [],
            "provider_data": {"model": model, "thinking_blocks": thinking_blocks},
        });
        if !thinking_text.is_empty() {
            item["content"] = json!([{"type": "reasoning_text", "text": thinking_text}]);
        }
        if let Some(sig) = last_signature {
            item["encrypted_content"] = json!(sig);
        }
        output.push(item);
    }
    if !text.is_empty() {
        output.push(json!({
            "type": "message", "id": FAKE_RESPONSES_ID, "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }));
    }
    output.extend(tool_calls);

    let usage = usage_of(message.get("usage").unwrap_or(&Value::Null));
    json!({
        "id": message.get("id").cloned().unwrap_or(json!(FAKE_RESPONSES_ID)),
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "input_tokens_details": {"cached_tokens": usage.input_tokens_details.cached_tokens},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })
}

/// Rebuilds a complete Anthropic message from the native stream events.
#[derive(Debug, Default)]
struct StreamAccumulator {
    message: Value,
    blocks: BTreeMap<u64, Value>,
    /// Partial `input_json_delta` text per tool-use block.
    partial_json: BTreeMap<u64, String>,
}

impl StreamAccumulator {
    fn handle(&mut self, event: &Value) {
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "message_start" => {
                self.message = event.get("message").cloned().unwrap_or_else(|| json!({}));
            }
            "content_block_start" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let block = event
                    .get("content_block")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                self.blocks.insert(index, block);
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let delta = event.get("delta").cloned().unwrap_or(Value::Null);
                let block = self.blocks.entry(index).or_insert_with(|| json!({}));
                match delta
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                {
                    "text_delta" => append(block, "text", delta.get("text")),
                    "thinking_delta" => append(block, "thinking", delta.get("thinking")),
                    "signature_delta" => {
                        if let Some(sig) = delta.get("signature") {
                            block["signature"] = sig.clone();
                        }
                    }
                    "input_json_delta" => {
                        self.partial_json.entry(index).or_default().push_str(
                            delta
                                .get("partial_json")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                        );
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                if let Some(raw) = self.partial_json.remove(&index)
                    && let Some(block) = self.blocks.get_mut(&index)
                {
                    block["input"] = serde_json::from_str(&raw).unwrap_or_else(|_| json!({}));
                }
            }
            "message_delta" => {
                if let Some(usage) = event.get("usage") {
                    merge_usage(&mut self.message, usage);
                }
                if let Some(delta) = event.get("delta").and_then(Value::as_object) {
                    for (k, v) in delta {
                        self.message[k.as_str()] = v.clone();
                    }
                }
            }
            _ => {}
        }
    }

    fn finish(mut self) -> Value {
        self.message["content"] = Value::Array(self.blocks.into_values().collect());
        self.message
    }
}

fn append(block: &mut Value, key: &str, value: Option<&Value>) {
    let text = value.and_then(Value::as_str).unwrap_or("");
    let existing = block
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    block[key] = json!(format!("{existing}{text}"));
}

fn merge_usage(message: &mut Value, usage: &Value) {
    let target = message
        .as_object_mut()
        .expect("message is an object")
        .entry("usage")
        .or_insert_with(|| json!({}));
    for (k, v) in usage.as_object().into_iter().flatten() {
        if let Some(n) = v.as_u64() {
            let current = target.get(k.as_str()).and_then(Value::as_u64).unwrap_or(0);
            // `message_delta` reports the running output total, not an increment.
            target[k.as_str()] = json!(if k == "output_tokens" { n } else { current + n });
        }
    }
}

#[async_trait]
impl Model for AnthropicMessagesModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = self.build_request_body(&request, false)?;
        let span = self.span(&request, &body);
        let include_data = request.tracing.include_data();
        let result = async {
            let (value, request_id) = self
                .client
                .post_json(
                    "messages",
                    &body,
                    request.model_settings.extra_headers.as_ref(),
                    request.model_settings.extra_query.as_ref(),
                )
                .await?;
            let response = message_to_response(&value, &self.model);
            parse_response(
                &response,
                request_id,
                request.model_settings.preserve_raw_usage.unwrap_or(false),
            )
        }
        .await;
        match &result {
            Ok(r) => finish_span(&span, r, include_data),
            Err(e) => {
                record_model_error_on_span(&span, "Error getting response", e, include_data);
                crate::logger::log_model_action_error("Error getting response", e);
            }
        }
        span.finish();
        result
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let body = self.build_request_body(&request, true)?;
        let span = self.span(&request, &body);
        let include_data = request.tracing.include_data();
        let (inner, _) = match self
            .client
            .post_sse(
                "messages",
                &body,
                request.model_settings.extra_headers.as_ref(),
                request.model_settings.extra_query.as_ref(),
            )
            .await
        {
            Ok(s) => s,
            Err(e) => {
                record_model_error_on_span(&span, "Error streaming response", &e, include_data);
                crate::logger::log_model_action_error("Error streaming response", &e);
                span.finish();
                return Err(e);
            }
        };

        let model = self.model.clone();
        let state = (
            inner,
            StreamAccumulator::default(),
            Some(span),
            false,
            0usize,
        );
        let stream = futures::stream::unfold(state, move |state| {
            let model = model.clone();
            async move {
                let (mut inner, mut acc, span, mut opened, mut index) = state;
                let span = span?;
                loop {
                    let Some(event) = inner.next().await else {
                        // The native stream ended: emit the assembled response.
                        let message = acc.finish();
                        let response = message_to_response(&message, &model);
                        if let Ok(parsed) = parse_response(&response, None, false) {
                            finish_span(&span, &parsed, include_data);
                        }
                        span.finish();
                        return Some((
                            Ok(json!({"type": "response.completed", "response": response})),
                            (inner, StreamAccumulator::default(), None, opened, index),
                        ));
                    };
                    let event = match event {
                        Ok(e) => e,
                        Err(e) => {
                            record_model_error_on_span(
                                &span,
                                "Error streaming response",
                                &e,
                                include_data,
                            );
                            crate::logger::log_model_action_error("Error streaming response", &e);
                            span.finish();
                            return Some((Err(e), (inner, acc, None, opened, index)));
                        }
                    };
                    acc.handle(&event);
                    // Forward the deltas a consumer can act on; everything else is folded into
                    // the terminal `response.completed` event.
                    let forwarded = match event.get("type").and_then(Value::as_str) {
                        Some("message_start") => Some(json!({
                            "type": "response.created",
                            "response": {"id": FAKE_RESPONSES_ID, "object": "response",
                                         "status": "in_progress", "output": [], "model": model},
                        })),
                        Some("content_block_start")
                            if event.pointer("/content_block/type").and_then(Value::as_str)
                                == Some("text") =>
                        {
                            index =
                                event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                            opened = true;
                            Some(json!({
                                "type": "response.output_item.added",
                                "output_index": index,
                                "item": {"type": "message", "id": FAKE_RESPONSES_ID,
                                         "role": "assistant", "status": "in_progress",
                                         "content": []},
                            }))
                        }
                        Some("content_block_delta") => {
                            let delta = event.get("delta").cloned().unwrap_or(Value::Null);
                            let output_index =
                                event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                            match delta.get("type").and_then(Value::as_str) {
                                Some("text_delta") if opened => Some(json!({
                                    "type": "response.output_text.delta",
                                    "item_id": FAKE_RESPONSES_ID,
                                    "output_index": output_index, "content_index": 0,
                                    "delta": delta.get("text").cloned().unwrap_or(Value::Null),
                                    "logprobs": [],
                                })),
                                Some("thinking_delta") => Some(json!({
                                    "type": "response.reasoning_text.delta",
                                    "item_id": FAKE_RESPONSES_ID,
                                    "output_index": output_index, "content_index": 0,
                                    "delta": delta.get("thinking").cloned().unwrap_or(Value::Null),
                                })),
                                Some("input_json_delta") => Some(json!({
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": FAKE_RESPONSES_ID,
                                    "output_index": output_index,
                                    "delta": delta.get("partial_json").cloned()
                                        .unwrap_or(Value::Null),
                                })),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    if let Some(forwarded) = forwarded {
                        return Some((Ok(forwarded), (inner, acc, Some(span), opened, index)));
                    }
                }
            }
        });
        Ok(stream.boxed())
    }

    fn model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn get_retry_advice(
        &self,
        request: &crate::retry::ModelRetryAdviceRequest<'_>,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        crate::retry::get_openai_retry_advice(request)
    }
}

fn finish_span(span: &crate::tracing::Span, resp: &ModelResponse, include_data: bool) {
    span.update_data(|d| {
        if let SpanData::Generation { output, usage, .. } = d {
            if include_data {
                *output = Some(json!(
                    resp.output.iter().map(|o| o.to_value()).collect::<Vec<_>>()
                ));
            }
            *usage = Some(resp.usage.to_span_usage());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::InputItem;

    #[test]
    fn converts_items_into_anthropic_messages() {
        let items = vec![
            InputItem(json!({"role": "user", "content": "weather?"})),
            InputItem(json!({"type": "reasoning", "id": "r",
                "provider_data": {"thinking_blocks": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"}]}})),
            InputItem(
                json!({"type": "function_call", "call_id": "c1", "name": "f",
                "arguments": "{\"city\":\"Hue\"}"}),
            ),
            InputItem(json!({"type": "function_call_output", "call_id": "c1",
                "output": "sunny"})),
        ];
        let messages = items_to_messages(&items).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"][0]["text"], "weather?");
        // The thinking block is replayed in front of the tool use, as Anthropic requires.
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "thinking");
        assert_eq!(messages[1]["content"][1]["type"], "tool_use");
        assert_eq!(messages[1]["content"][1]["input"]["city"], "Hue");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "c1");
        assert_eq!(messages[2]["content"][0]["content"], "sunny");
    }

    #[test]
    fn converts_a_message_into_responses_output() {
        let message = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
            "content": [
                {"type": "thinking", "thinking": "think", "signature": "sig"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "tu_1", "name": "f", "input": {"a": 1}}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 7, "cache_read_input_tokens": 2}
        });
        let response = message_to_response(&message, "claude-sonnet-4-5");
        let output = response["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(
            output[0]["provider_data"]["thinking_blocks"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(output[0]["encrypted_content"], "sig");
        assert_eq!(output[1]["content"][0]["text"], "hello");
        assert_eq!(output[2]["type"], "function_call");
        assert_eq!(output[2]["arguments"], "{\"a\":1}");
        assert_eq!(response["usage"]["input_tokens"], 7);
        assert_eq!(response["usage"]["total_tokens"], 14);
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            2
        );
    }

    #[test]
    fn accumulates_a_native_stream() {
        let mut acc = StreamAccumulator::default();
        for event in [
            json!({"type": "message_start", "message": {"id": "m", "role": "assistant",
                "usage": {"input_tokens": 4, "output_tokens": 0}}}),
            json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "tool_use", "id": "tu", "name": "f"}}),
            json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "1}"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 9}}),
            json!({"type": "message_stop"}),
        ] {
            acc.handle(&event);
        }
        let message = acc.finish();
        assert_eq!(message["content"][0]["text"], "Hello");
        assert_eq!(message["content"][1]["input"]["a"], 1);
        assert_eq!(message["stop_reason"], "tool_use");
        assert_eq!(message["usage"]["output_tokens"], 9);
        assert_eq!(message["usage"]["input_tokens"], 4);
    }

    #[test]
    fn builds_a_request_body() {
        let model = AnthropicMessagesModel::new("claude-sonnet-4-5", "sk-ant-test");
        let request = ModelRequest {
            system_instructions: Some("be brief".into()),
            input: vec![InputItem(json!({"role": "user", "content": "hi"}))],
            model_settings: crate::model_settings::ModelSettings {
                max_tokens: Some(2000),
                reasoning: Some(crate::model_settings::Reasoning {
                    effort: Some("medium".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let body = model.build_request_body(&request, true).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-5");
        assert_eq!(body["max_tokens"], 2000);
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["stream"], true);
        // 50% of 2000 is below Anthropic's 1024 floor, so the budget is clamped up to it.
        assert_eq!(body["thinking"]["budget_tokens"], 1024);
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
        assert!(body.get("tools").is_none());
    }
}
