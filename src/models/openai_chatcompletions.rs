//! Model backed by the OpenAI Chat Completions API. Mirrors
//! `agents.models.openai_chatcompletions` and `chatcmpl_converter`.
//!
//! Responses-format input items are converted to chat messages, and streamed chat chunks are
//! re-emitted as Responses API stream events so the runner handles both APIs the same way.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Map, Value, json};

use super::FAKE_RESPONSES_ID;
use super::interface::{Model, ModelRequest, ResponseEventStream, ToolSpec};
use super::openai_client::OpenAIClient;
use crate::error::{AgentsError, Result};
use crate::items::{
    FunctionCall, InputItem, MessageContent, MessageOutput, ModelResponse, OutputItem,
};
use crate::model_settings::ToolChoice;
use crate::tracing::{Span, SpanData, generation_span};
use crate::usage::Usage;

/// Provider marker stored on reasoning items that came from a `reasoning` (not
/// `reasoning_content`) field.
const REASONING_FIELD_KEY: &str = "_chat_completions_reasoning_field";

/// A reasoning item considered for replay into the next Chat Completions request.
#[derive(Debug, Clone)]
pub struct ReasoningContentReplayContext {
    /// The model that will receive the next request.
    pub model: String,
    /// The request base URL.
    pub base_url: Option<String>,
    /// The raw reasoning item.
    pub item: Value,
    /// The model that produced the reasoning item, if known.
    pub origin_model: Option<String>,
    /// Provider metadata captured on the reasoning item.
    pub provider_data: Map<String, Value>,
}

/// Decides whether a reasoning item is replayed as `reasoning_content`.
pub type ShouldReplayReasoningContent =
    Arc<dyn Fn(&ReasoningContentReplayContext) -> bool + Send + Sync>;

/// Default replay rule: only for DeepSeek models, and only reasoning that came from DeepSeek
/// (or predates provider tracking).
pub fn default_should_replay_reasoning_content(ctx: &ReasoningContentReplayContext) -> bool {
    if !ctx.model.to_lowercase().contains("deepseek") {
        return false;
    }
    let other_data = ctx.provider_data.keys().any(|k| k != "thinking_blocks");
    ctx.origin_model
        .as_ref()
        .is_some_and(|m| m.to_lowercase().contains("deepseek"))
        || !other_data
}

/// Options for converting Responses input items to Chat Completions messages.
#[derive(Clone, Default)]
pub struct ChatConversionOptions {
    /// The target model (enables reasoning replay).
    pub model: Option<String>,
    /// The target base URL.
    pub base_url: Option<String>,
    /// Custom reasoning replay rule.
    pub should_replay_reasoning_content: Option<ShouldReplayReasoningContent>,
}

impl std::fmt::Debug for ChatConversionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatConversionOptions")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

/// A model that calls `POST /chat/completions`.
#[derive(Clone)]
pub struct OpenAIChatCompletionsModel {
    model: String,
    client: OpenAIClient,
    should_replay_reasoning_content: Option<ShouldReplayReasoningContent>,
}

impl std::fmt::Debug for OpenAIChatCompletionsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIChatCompletionsModel")
            .field("model", &self.model)
            .field("client", &self.client)
            .finish_non_exhaustive()
    }
}

impl OpenAIChatCompletionsModel {
    /// Create a Chat Completions model for `model` using `client`.
    pub fn new(model: impl Into<String>, client: OpenAIClient) -> Self {
        OpenAIChatCompletionsModel {
            model: model.into(),
            client,
            should_replay_reasoning_content: None,
        }
    }

    /// Decide which reasoning items are replayed to the provider as `reasoning_content`.
    pub fn with_should_replay_reasoning_content<F>(mut self, f: F) -> Self
    where
        F: Fn(&ReasoningContentReplayContext) -> bool + Send + Sync + 'static,
    {
        self.should_replay_reasoning_content = Some(Arc::new(f));
        self
    }

    /// The model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Build the JSON request body.
    pub fn build_request_body(&self, req: &ModelRequest, stream: bool) -> Result<Value> {
        let s = &req.model_settings;
        let messages = items_to_messages_with(
            &req.input,
            req.system_instructions.as_deref(),
            &ChatConversionOptions {
                model: Some(self.model.clone()),
                base_url: Some(self.client.base_url().to_string()),
                should_replay_reasoning_content: self.should_replay_reasoning_content.clone(),
            },
        )?;
        let mut tools: Vec<Value> = Vec::new();
        for t in &req.tools {
            match t {
                ToolSpec::Function(f) => {
                    let mut features = Vec::new();
                    if f.namespace.is_some() {
                        features.push("tool_namespace()");
                    }
                    if f.defer_loading {
                        features.push("defer_loading");
                    }
                    if f.allowed_callers.is_some() {
                        features.push("allowed_callers");
                    }
                    if !features.is_empty() {
                        return Err(AgentsError::user(format!(
                            "The following function-tool features are only supported with \
                             OpenAI Responses models: {}. Tool `{}` cannot be used with \
                             OpenAIChatCompletionsModel.",
                            features.join(", "),
                            f.name
                        )));
                    }
                    tools.push(f.to_chat_param())
                }
                ToolSpec::Hosted { param, .. } => {
                    return Err(AgentsError::user(format!(
                        "Hosted tools are not supported with the ChatCompletions API. Got tool type: {}",
                        param
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown")
                    )));
                }
            }
        }
        for h in &req.handoffs {
            tools.push(h.to_chat_param());
        }
        let mut body = Map::new();
        body.insert("model".into(), Value::String(self.model.clone()));
        body.insert("messages".into(), Value::Array(messages));
        let has_tools = !tools.is_empty();
        if has_tools {
            body.insert("tools".into(), Value::Array(tools));
            if let Some(tc) = &s.tool_choice {
                body.insert("tool_choice".into(), convert_chat_tool_choice(tc));
            }
            if let Some(p) = s.parallel_tool_calls {
                body.insert("parallel_tool_calls".into(), Value::Bool(p));
            }
        }
        if let Some(schema) = &req.output_schema
            && !schema.is_plain_text()
        {
            body.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "final_output",
                        "strict": schema.is_strict_json_schema(),
                        "schema": schema.json_schema()?,
                    }
                }),
            );
        }
        let mut put = |k: &str, v: Option<Value>| {
            if let Some(v) = v {
                body.insert(k.to_string(), v);
            }
        };
        put("temperature", s.temperature.map(|v| json!(v)));
        put("top_p", s.top_p.map(|v| json!(v)));
        put("frequency_penalty", s.frequency_penalty.map(|v| json!(v)));
        put("presence_penalty", s.presence_penalty.map(|v| json!(v)));
        put("max_tokens", s.max_tokens.map(|v| json!(v)));
        put(
            "reasoning_effort",
            s.reasoning
                .as_ref()
                .and_then(|r| r.effort.clone())
                .map(Value::String),
        );
        put("verbosity", s.verbosity.clone().map(Value::String));
        put("metadata", s.metadata.as_ref().map(|m| json!(m)));
        put(
            "prompt_cache_retention",
            s.prompt_cache_retention.clone().map(Value::String),
        );
        put("prompt_cache_options", s.prompt_cache_options.clone());
        let store = s.store.or(if self.client.is_openai_api() {
            Some(true)
        } else {
            None
        });
        put("store", store.map(Value::Bool));
        if let Some(n) = s.top_logprobs {
            put("top_logprobs", Some(json!(n)));
            let wants_logprobs = s
                .extra_args
                .as_ref()
                .map(|e| !e.contains_key("logprobs"))
                .unwrap_or(true);
            if wants_logprobs {
                put("logprobs", Some(Value::Bool(true)));
            }
        }
        if stream {
            body.insert("stream".into(), Value::Bool(true));
            let include_usage = match s.include_usage {
                Some(v) => v,
                None => self.client.is_openai_api(),
            };
            if include_usage {
                body.insert("stream_options".into(), json!({"include_usage": true}));
            }
        }
        if let Some(extra) = &s.extra_args {
            for (k, v) in extra {
                body.insert(k.clone(), v.clone());
            }
        }
        if let Some(extra) = &s.extra_body {
            for (k, v) in extra {
                body.insert(k.clone(), v.clone());
            }
        }
        Ok(Value::Object(body))
    }

    fn start_span(&self, req: &ModelRequest, body: &Value) -> Span {
        let include = req.tracing.include_data();
        let span = generation_span(
            if include {
                body.get("messages").cloned()
            } else {
                None
            },
            Some(self.model.clone()),
            Some(req.model_settings.to_traceable_value()),
            req.tracing.is_disabled(),
        );
        span.start();
        span
    }
}

fn finish_generation_span(span: &Span, resp: &ModelResponse, include_data: bool) {
    span.update_data(|d| {
        if let SpanData::Generation { output, usage, .. } = d {
            if include_data {
                *output = Some(Value::Array(
                    resp.output.iter().map(OutputItem::to_value).collect(),
                ));
            }
            *usage = Some(resp.usage.to_span_usage());
        }
    });
}

/// Convert a `ToolChoice` to the Chat Completions format.
pub(crate) fn convert_chat_tool_choice(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Named(n) if matches!(n.as_str(), "auto" | "required" | "none") => {
            Value::String(n.clone())
        }
        ToolChoice::Named(n) => json!({"type": "function", "function": {"name": n}}),
        ToolChoice::Mcp(m) => json!({"type": "function", "function": {"name": m.name}}),
    }
}

fn convert_user_part(part: &Value) -> Result<Value> {
    let ty = part.get("type").and_then(Value::as_str).unwrap_or_default();
    match ty {
        "input_text" | "text" | "output_text" => Ok(json!({
            "type": "text",
            "text": part.get("text").cloned().unwrap_or(Value::String(String::new())),
        })),
        "input_image" => {
            let Some(url) = part.get("image_url").and_then(Value::as_str) else {
                return Err(AgentsError::user(
                    "Only image URLs are supported for input_image with the Chat Completions API",
                ));
            };
            let mut image = json!({"url": url});
            if let Some(d) = part.get("detail") {
                image["detail"] = d.clone();
            }
            Ok(json!({"type": "image_url", "image_url": image}))
        }
        "input_file" => {
            let mut file = Map::new();
            for k in ["file_data", "file_id", "filename"] {
                if let Some(v) = part.get(k) {
                    file.insert(k.into(), v.clone());
                }
            }
            Ok(json!({"type": "file", "file": file}))
        }
        "input_audio" => Ok(part.clone()),
        _ => Err(AgentsError::user(format!(
            "Unknown content part for Chat Completions: {part}"
        ))),
    }
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Convert Responses-format input items to Chat Completions messages.
pub fn items_to_messages(items: &[InputItem], system: Option<&str>) -> Result<Vec<Value>> {
    items_to_messages_with(items, system, &ChatConversionOptions::default())
}

/// [`items_to_messages`] with reasoning replay options.
pub fn items_to_messages_with(
    items: &[InputItem],
    system: Option<&str>,
    options: &ChatConversionOptions,
) -> Result<Vec<Value>> {
    let mut result: Vec<Value> = Vec::new();
    // Reasoning waiting to be attached to the next assistant message.
    let mut pending_reasoning_content: Option<String> = None;
    let mut pending_reasoning: Option<String> = None;
    if let Some(s) = system {
        result.push(json!({"role": "system", "content": s}));
    }
    let mut current: Option<Map<String, Value>> = None;
    fn flush(result: &mut Vec<Value>, current: &mut Option<Map<String, Value>>) {
        if let Some(mut a) = current.take() {
            if a.get("tool_calls")
                .and_then(Value::as_array)
                .map(Vec::is_empty)
                .unwrap_or(false)
            {
                a.remove("tool_calls");
            }
            result.push(Value::Object(a));
        }
    }
    let apply_reasoning =
        |asst: &mut Map<String, Value>, rc: &mut Option<String>, r: &mut Option<String>| {
            if let Some(text) = rc.take() {
                asst.insert("reasoning_content".into(), Value::String(text));
            }
            if let Some(text) = r.take() {
                asst.insert("reasoning".into(), Value::String(text));
            }
        };
    for item in items {
        let v = &item.0;
        let ty = v.get("type").and_then(Value::as_str);
        let role = v.get("role").and_then(Value::as_str);
        match (ty, role) {
            (None | Some("message"), Some(role @ ("user" | "system" | "developer"))) => {
                flush(&mut result, &mut current);
                pending_reasoning_content = None;
                pending_reasoning = None;
                let content = v.get("content").cloned().unwrap_or(Value::Null);
                let converted = match (&content, role) {
                    (Value::Array(parts), "user") => {
                        Value::Array(parts.iter().map(convert_user_part).collect::<Result<_>>()?)
                    }
                    (Value::Array(_), _) => Value::String(text_of(&content)),
                    _ => content,
                };
                result.push(json!({"role": role, "content": converted}));
            }
            (None | Some("message"), Some("assistant")) => {
                flush(&mut result, &mut current);
                let content = v.get("content").cloned().unwrap_or(Value::Null);
                let mut asst = Map::new();
                asst.insert("role".into(), Value::String("assistant".into()));
                match &content {
                    Value::String(s) => {
                        asst.insert("content".into(), Value::String(s.clone()));
                    }
                    Value::Array(parts) => {
                        let mut text = String::new();
                        let mut refusal = String::new();
                        for p in parts {
                            match p.get("type").and_then(Value::as_str) {
                                Some("refusal") => refusal.push_str(
                                    p.get("refusal").and_then(Value::as_str).unwrap_or(""),
                                ),
                                _ => text
                                    .push_str(p.get("text").and_then(Value::as_str).unwrap_or("")),
                            }
                        }
                        if !text.is_empty() {
                            asst.insert("content".into(), Value::String(text));
                        }
                        if !refusal.is_empty() {
                            asst.insert("refusal".into(), Value::String(refusal));
                        }
                    }
                    _ => {}
                }
                asst.insert("tool_calls".into(), Value::Array(Vec::new()));
                apply_reasoning(
                    &mut asst,
                    &mut pending_reasoning_content,
                    &mut pending_reasoning,
                );
                current = Some(asst);
            }
            (Some("reasoning"), _) => {
                flush(&mut result, &mut current);
                pending_reasoning_content = None;
                pending_reasoning = None;
                let provider_data = v
                    .get("provider_data")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let item_model = provider_data
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let texts = |key: &str, part_type: &str| -> Vec<String> {
                    v.get(key)
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter(|p| p.get("type").and_then(Value::as_str) == Some(part_type))
                        .filter_map(|p| p.get("text").and_then(Value::as_str))
                        .filter(|t| !t.is_empty())
                        .map(str::to_string)
                        .collect()
                };
                if let Some(model) = &options.model {
                    if provider_data
                        .get(REASONING_FIELD_KEY)
                        .and_then(Value::as_str)
                        == Some("reasoning")
                        && *model == item_model
                    {
                        let t = texts("content", "reasoning_text");
                        if !t.is_empty() {
                            pending_reasoning = Some(t.join("\n"));
                        }
                    }
                    let ctx = ReasoningContentReplayContext {
                        model: model.clone(),
                        base_url: options
                            .base_url
                            .as_ref()
                            .map(|b| b.trim_end_matches('/').to_string()),
                        item: v.clone(),
                        origin_model: (!item_model.is_empty()).then(|| item_model.clone()),
                        provider_data: provider_data.clone(),
                    };
                    let replay = match &options.should_replay_reasoning_content {
                        Some(f) => f(&ctx),
                        None => default_should_replay_reasoning_content(&ctx),
                    };
                    if replay {
                        let t = texts("summary", "summary_text");
                        if !t.is_empty() {
                            pending_reasoning_content = Some(t.join("\n"));
                        }
                    }
                }
            }
            (Some("function_call"), _) => {
                let asst = current.get_or_insert_with(|| {
                    let mut m = Map::new();
                    m.insert("role".into(), Value::String("assistant".into()));
                    m.insert("content".into(), Value::Null);
                    m.insert("tool_calls".into(), Value::Array(Vec::new()));
                    m
                });
                apply_reasoning(asst, &mut pending_reasoning_content, &mut pending_reasoning);
                let call = json!({
                    "id": v.get("call_id").cloned().unwrap_or(Value::Null),
                    "type": "function",
                    "function": {
                        "name": v.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": v.get("arguments").cloned().unwrap_or(Value::String("{}".into())),
                    }
                });
                if let Some(Value::Array(calls)) = asst.get_mut("tool_calls") {
                    calls.push(call);
                }
            }
            (Some("function_call_output"), _) => {
                flush(&mut result, &mut current);
                pending_reasoning_content = None;
                pending_reasoning = None;
                let output = v.get("output").cloned().unwrap_or(Value::Null);
                let content = match &output {
                    Value::String(s) => Value::String(s.clone()),
                    Value::Array(_) => Value::String(text_of(&output)),
                    other => Value::String(other.to_string()),
                };
                result.push(json!({
                    "role": "tool",
                    "tool_call_id": v.get("call_id").cloned().unwrap_or(Value::Null),
                    "content": content,
                }));
            }
            (Some("item_reference"), _) => {
                return Err(AgentsError::user(
                    "Encountered an item_reference, which is not supported by Chat Completions",
                ));
            }
            (Some(other), _) => {
                log::debug!("Skipping item type {other} not supported by Chat Completions");
            }
            (None, _) => {
                return Err(AgentsError::user(format!("Unhandled input item: {v}")));
            }
        }
    }
    flush(&mut result, &mut current);
    Ok(result)
}

/// Parse a Chat Completions response into a `ModelResponse`.
pub(crate) fn parse_chat_completion(
    v: &Value,
    request_id: Option<String>,
    preserve_raw_usage: bool,
) -> Result<ModelResponse> {
    let choice = v.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    if message.get("audio").map(|a| !a.is_null()).unwrap_or(false) {
        return Err(AgentsError::other("Audio is not currently supported"));
    }
    let mut output = Vec::new();
    let model = v.get("model").and_then(Value::as_str).unwrap_or_default();
    let reasoning_content = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .unwrap_or("");
    let reasoning = message
        .get("reasoning")
        .and_then(Value::as_str)
        .filter(|_| reasoning_content.is_empty())
        .unwrap_or("");
    if !reasoning_content.is_empty() || !reasoning.is_empty() {
        let mut provider_data = Map::new();
        if !model.is_empty() {
            provider_data.insert("model".into(), json!(model));
        }
        let mut item = json!({"type": "reasoning", "id": FAKE_RESPONSES_ID, "summary": []});
        if !reasoning_content.is_empty() {
            item["summary"] = json!([{"type": "summary_text", "text": reasoning_content}]);
        } else {
            item["content"] = json!([{"type": "reasoning_text", "text": reasoning}]);
            provider_data.insert(REASONING_FIELD_KEY.into(), json!("reasoning"));
        }
        if !provider_data.is_empty() {
            item["provider_data"] = Value::Object(provider_data);
        }
        output.push(OutputItem::from_value(item));
    }
    let text = message.get("content").and_then(Value::as_str).unwrap_or("");
    let refusal = message.get("refusal").and_then(Value::as_str).unwrap_or("");
    if !text.is_empty() || !refusal.is_empty() {
        let mut content = Vec::new();
        if !text.is_empty() {
            content.push(MessageContent::OutputText {
                text: text.to_string(),
                annotations: Vec::new(),
                logprobs: None,
            });
        }
        if !refusal.is_empty() {
            content.push(MessageContent::Refusal {
                refusal: refusal.to_string(),
            });
        }
        output.push(OutputItem::Message(MessageOutput {
            id: FAKE_RESPONSES_ID.into(),
            role: "assistant".into(),
            content,
            status: Some("completed".into()),
            extra: Map::new(),
        }));
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            output.push(OutputItem::FunctionCall(FunctionCall {
                id: Some(FAKE_RESPONSES_ID.into()),
                call_id: c
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: c
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                arguments: c
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}")
                    .to_string(),
                status: None,
                extra: Map::new(),
            }));
        }
    }
    if output.is_empty() && choice.get("finish_reason").and_then(Value::as_str) == Some("length") {
        return Err(AgentsError::model_behavior(
            "Chat Completions response ended with finish_reason='length' before producing output",
        ));
    }
    let usage_v = v.get("usage").filter(|u| !u.is_null());
    let usage = usage_v.map(Usage::from_chat_usage).unwrap_or(Usage {
        requests: 1,
        ..Default::default()
    });
    Ok(ModelResponse {
        output,
        usage,
        response_id: None,
        request_id,
        raw_usage: if preserve_raw_usage {
            usage_v.cloned()
        } else {
            None
        },
    })
}

struct StreamState {
    inner: BoxStream<'static, Result<Value>>,
    pending: VecDeque<Result<Value>>,
    text: String,
    refusal: String,
    message_started: bool,
    tool_calls: BTreeMap<u64, (String, String, String)>,
    /// Streamed reasoning: (summary text, reasoning text, summary part open).
    reasoning: Option<(String, String, bool)>,
    usage: Option<Value>,
    finished: bool,
    model: String,
    span: Span,
    include_data: bool,
}

impl StreamState {
    fn message_index(&self) -> usize {
        usize::from(self.reasoning.is_some())
    }

    fn reasoning_provider_data(&self, reasoning_field: bool) -> Value {
        let mut m = Map::new();
        m.insert("model".into(), json!(self.model));
        if reasoning_field {
            m.insert(REASONING_FIELD_KEY.into(), json!("reasoning"));
        }
        Value::Object(m)
    }

    fn start_reasoning(&mut self, reasoning_field: bool) {
        if self.reasoning.is_some() || self.message_started {
            return;
        }
        self.reasoning = Some((String::new(), String::new(), false));
        let mut item = json!({"type": "reasoning", "id": FAKE_RESPONSES_ID, "summary": [],
            "provider_data": self.reasoning_provider_data(reasoning_field)});
        if reasoning_field {
            item["content"] = json!([{"type": "reasoning_text", "text": ""}]);
        }
        self.pending.push_back(Ok(json!({
            "type": "response.output_item.added", "output_index": 0, "item": item,
        })));
    }

    fn close_summary_part(&mut self) {
        if let Some((summary, _, open)) = &mut self.reasoning
            && *open
        {
            *open = false;
            let text = summary.clone();
            self.pending.push_back(Ok(json!({
                "type": "response.reasoning_summary_text.done", "item_id": FAKE_RESPONSES_ID,
                "output_index": 0, "summary_index": 0, "text": text,
            })));
            self.pending.push_back(Ok(json!({
                "type": "response.reasoning_summary_part.done", "item_id": FAKE_RESPONSES_ID,
                "output_index": 0, "summary_index": 0,
                "part": {"type": "summary_text", "text": text},
            })));
        }
    }

    fn start_message(&mut self) {
        if self.message_started {
            return;
        }
        self.close_summary_part();
        self.message_started = true;
        let index = self.message_index();
        self.pending.push_back(Ok(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": {"type": "message", "id": FAKE_RESPONSES_ID, "role": "assistant", "status": "in_progress", "content": []}
        })));
        self.pending.push_back(Ok(json!({
            "type": "response.content_part.added",
            "item_id": FAKE_RESPONSES_ID,
            "output_index": index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
        })));
    }

    fn handle_chunk(&mut self, chunk: &Value) {
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(u.clone());
        }
        let Some(delta) = chunk.pointer("/choices/0/delta") else {
            return;
        };
        if let Some(rc) = delta.get("reasoning_content").and_then(Value::as_str)
            && !rc.is_empty()
        {
            self.start_reasoning(false);
            if let Some((summary, _, open)) = &mut self.reasoning
                && !self.message_started
            {
                if !*open {
                    *open = true;
                    self.pending.push_back(Ok(json!({
                        "type": "response.reasoning_summary_part.added",
                        "item_id": FAKE_RESPONSES_ID, "output_index": 0, "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    })));
                }
                summary.push_str(rc);
                self.pending.push_back(Ok(json!({
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": FAKE_RESPONSES_ID, "output_index": 0, "summary_index": 0,
                    "delta": rc,
                })));
            }
        }
        if let Some(r) = delta.get("reasoning").and_then(Value::as_str)
            && !r.is_empty()
        {
            self.start_reasoning(true);
            if let Some((_, text, _)) = &mut self.reasoning
                && !self.message_started
            {
                text.push_str(r);
                self.pending.push_back(Ok(json!({
                    "type": "response.reasoning_text.delta",
                    "item_id": FAKE_RESPONSES_ID, "output_index": 0, "content_index": 0,
                    "delta": r,
                })));
            }
        }
        if delta
            .get("tool_calls")
            .is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty()))
        {
            self.close_summary_part();
        }
        if let Some(t) = delta.get("content").and_then(Value::as_str)
            && !t.is_empty()
        {
            self.start_message();
            self.text.push_str(t);
            let index = self.message_index();
            self.pending.push_back(Ok(json!({
                "type": "response.output_text.delta",
                "item_id": FAKE_RESPONSES_ID,
                "output_index": index,
                "content_index": 0,
                "delta": t,
                "logprobs": [],
            })));
        }
        if let Some(r) = delta.get("refusal").and_then(Value::as_str)
            && !r.is_empty()
        {
            self.start_message();
            self.refusal.push_str(r);
            let index = self.message_index();
            self.pending.push_back(Ok(json!({
                "type": "response.refusal.delta",
                "item_id": FAKE_RESPONSES_ID,
                "output_index": index,
                "content_index": 0,
                "delta": r,
            })));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for c in calls {
                let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
                let entry = self.tool_calls.entry(idx).or_default();
                if let Some(id) = c.get("id").and_then(Value::as_str) {
                    entry.0.push_str(id);
                }
                if let Some(n) = c.pointer("/function/name").and_then(Value::as_str) {
                    entry.1.push_str(n);
                }
                if let Some(a) = c.pointer("/function/arguments").and_then(Value::as_str) {
                    entry.2.push_str(a);
                }
            }
        }
    }

    fn finish(&mut self) {
        let mut output: Vec<Value> = Vec::new();
        self.close_summary_part();
        if let Some((summary, text, _)) = self.reasoning.clone() {
            let reasoning_field = summary.is_empty() && !text.is_empty();
            let mut item = json!({"type": "reasoning", "id": FAKE_RESPONSES_ID,
                "summary": if summary.is_empty() { json!([]) }
                           else { json!([{"type": "summary_text", "text": summary}]) },
                "provider_data": self.reasoning_provider_data(reasoning_field)});
            if !text.is_empty() {
                item["content"] = json!([{"type": "reasoning_text", "text": text}]);
                self.pending.push_back(Ok(json!({
                    "type": "response.reasoning_text.done", "item_id": FAKE_RESPONSES_ID,
                    "output_index": 0, "content_index": 0, "text": text,
                })));
            }
            self.pending.push_back(Ok(json!({
                "type": "response.output_item.done", "output_index": 0, "item": item.clone(),
            })));
            output.push(item);
        }
        let mi = self.message_index();
        if self.message_started {
            let mut content = Vec::new();
            if !self.text.is_empty() {
                content.push(json!({"type": "output_text", "text": self.text, "annotations": []}));
                self.pending.push_back(Ok(json!({
                    "type": "response.output_text.done",
                    "item_id": FAKE_RESPONSES_ID, "output_index": mi, "content_index": 0,
                    "text": self.text,
                })));
                self.pending.push_back(Ok(json!({
                    "type": "response.content_part.done",
                    "item_id": FAKE_RESPONSES_ID, "output_index": mi, "content_index": 0,
                    "part": {"type": "output_text", "text": self.text, "annotations": []},
                })));
            }
            if !self.refusal.is_empty() {
                content.push(json!({"type": "refusal", "refusal": self.refusal}));
            }
            let item = json!({
                "type": "message", "id": FAKE_RESPONSES_ID, "role": "assistant",
                "status": "completed", "content": content,
            });
            self.pending.push_back(Ok(json!({
                "type": "response.output_item.done", "output_index": mi, "item": item.clone(),
            })));
            output.push(item);
        }
        for (id, name, args) in self.tool_calls.values() {
            let output_index = output.len();
            let args = if args.is_empty() {
                "{}".to_string()
            } else {
                args.clone()
            };
            let added = json!({
                "type": "function_call", "id": FAKE_RESPONSES_ID, "call_id": id,
                "name": name, "arguments": "",
            });
            self.pending.push_back(Ok(json!({
                "type": "response.output_item.added", "output_index": output_index, "item": added,
            })));
            self.pending.push_back(Ok(json!({
                "type": "response.function_call_arguments.delta",
                "item_id": FAKE_RESPONSES_ID, "output_index": output_index, "delta": args,
            })));
            self.pending.push_back(Ok(json!({
                "type": "response.function_call_arguments.done",
                "item_id": FAKE_RESPONSES_ID, "output_index": output_index, "arguments": args,
            })));
            let item = json!({
                "type": "function_call", "id": FAKE_RESPONSES_ID, "call_id": id,
                "name": name, "arguments": args,
            });
            self.pending.push_back(Ok(json!({
                "type": "response.output_item.done", "output_index": output_index, "item": item.clone(),
            })));
            output.push(item);
        }
        let usage = self.usage.as_ref().map(|u| {
            let parsed = Usage::from_chat_usage(u);
            json!({
                "input_tokens": parsed.input_tokens,
                "output_tokens": parsed.output_tokens,
                "total_tokens": parsed.total_tokens,
                "input_tokens_details": {"cached_tokens": parsed.input_tokens_details.cached_tokens},
                "output_tokens_details": {"reasoning_tokens": parsed.output_tokens_details.reasoning_tokens},
            })
        });
        let response = json!({
            "id": FAKE_RESPONSES_ID, "object": "response", "status": "completed",
            "model": self.model, "output": output, "usage": usage,
        });
        if let Ok(parsed) = super::openai_responses::parse_response(&response, None, false) {
            let mut parsed = parsed;
            parsed.response_id = None;
            finish_generation_span(&self.span, &parsed, self.include_data);
        }
        self.span.finish();
        self.pending.push_back(Ok(
            json!({"type": "response.completed", "response": response}),
        ));
    }
}

#[async_trait]
impl Model for OpenAIChatCompletionsModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = self.build_request_body(&request, false)?;
        let span = self.start_span(&request, &body);
        let result = async {
            let (value, request_id) = self
                .client
                .post_json(
                    "chat/completions",
                    &body,
                    request.model_settings.extra_headers.as_ref(),
                    request.model_settings.extra_query.as_ref(),
                )
                .await?;
            parse_chat_completion(
                &value,
                request_id,
                request.model_settings.preserve_raw_usage.unwrap_or(false),
            )
        }
        .await;
        match &result {
            Ok(r) => finish_generation_span(&span, r, request.tracing.include_data()),
            Err(e) => span.set_error(
                "Error getting response",
                Some(json!({"error": e.to_string()})),
            ),
        }
        span.finish();
        result
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let body = self.build_request_body(&request, true)?;
        let span = self.start_span(&request, &body);
        let (inner, _) = match self
            .client
            .post_sse(
                "chat/completions",
                &body,
                request.model_settings.extra_headers.as_ref(),
                request.model_settings.extra_query.as_ref(),
            )
            .await
        {
            Ok(s) => s,
            Err(e) => {
                span.set_error(
                    "Error streaming response",
                    Some(json!({"error": e.to_string()})),
                );
                span.finish();
                return Err(e);
            }
        };
        let mut state = StreamState {
            inner,
            pending: VecDeque::new(),
            text: String::new(),
            refusal: String::new(),
            message_started: false,
            tool_calls: BTreeMap::new(),
            reasoning: None,
            usage: None,
            finished: false,
            model: self.model.clone(),
            span,
            include_data: request.tracing.include_data(),
        };
        state.pending.push_back(Ok(json!({
            "type": "response.created",
            "response": {"id": FAKE_RESPONSES_ID, "object": "response", "status": "in_progress", "output": [], "model": self.model},
        })));
        let stream = futures::stream::unfold(state, |mut st| async move {
            loop {
                if let Some(ev) = st.pending.pop_front() {
                    return Some((ev, st));
                }
                if st.finished {
                    return None;
                }
                match st.inner.next().await {
                    Some(Ok(chunk)) => st.handle_chunk(&chunk),
                    Some(Err(e)) => {
                        st.finished = true;
                        st.span.set_error(
                            "Error streaming response",
                            Some(json!({"error": e.to_string()})),
                        );
                        st.span.finish();
                        return Some((Err(e), st));
                    }
                    None => {
                        st.finished = true;
                        st.finish();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_items_to_messages() {
        let items: Vec<InputItem> = vec![
            InputItem::user("hi"),
            json!({"type": "message", "role": "assistant", "id": "m", "content": [{"type": "output_text", "text": "calling"}]}).into(),
            json!({"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}).into(),
            json!({"type": "function_call_output", "call_id": "c1", "output": "42"}).into(),
            json!({"role": "user", "content": [{"type": "input_text", "text": "see"}, {"type": "input_image", "image_url": "http://x", "detail": "low"}]}).into(),
        ];
        let msgs = items_to_messages(&items, Some("sys")).unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "calling");
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "c1");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[4]["content"][1]["image_url"]["detail"], "low");
    }

    #[test]
    fn parses_chat_completion() {
        let v = json!({"choices": [{"message": {"content": "hi", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}
        ]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}});
        let r = parse_chat_completion(&v, None, false).unwrap();
        assert_eq!(r.output.len(), 2);
        assert!(matches!(&r.output[1], OutputItem::FunctionCall(f) if f.call_id == "call_1"));
        assert_eq!(r.usage.input_tokens, 5);
    }

    #[tokio::test]
    async fn synthesizes_responses_stream_events() {
        let chunks = vec![
            Ok(json!({"choices": [{"delta": {"content": "Hel"}}]})),
            Ok(json!({"choices": [{"delta": {"content": "lo"}}]})),
            Ok(
                json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "f", "arguments": "{\"x\""}}]}}]}),
            ),
            Ok(
                json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}}]}),
            ),
            Ok(
                json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
            ),
        ];
        let mut st = StreamState {
            inner: futures::stream::iter(chunks).boxed(),
            pending: VecDeque::new(),
            text: String::new(),
            refusal: String::new(),
            message_started: false,
            tool_calls: BTreeMap::new(),
            reasoning: None,
            usage: None,
            finished: false,
            model: "m".into(),
            span: crate::tracing::create_span(
                SpanData::Custom {
                    name: "t".into(),
                    data: Value::Null,
                },
                None,
                true,
            ),
            include_data: false,
        };
        while let Some(Ok(c)) = st.inner.next().await {
            st.handle_chunk(&c);
        }
        st.finish();
        let events: Vec<Value> = st.pending.into_iter().map(|e| e.unwrap()).collect();
        let completed = events.last().unwrap();
        assert_eq!(completed["type"], "response.completed");
        let out = &completed["response"]["output"];
        assert_eq!(out[0]["content"][0]["text"], "Hello");
        assert_eq!(out[1]["arguments"], "{\"x\":1}");
        assert_eq!(completed["response"]["usage"]["total_tokens"], 2);
    }

    #[tokio::test]
    async fn streams_and_replays_deepseek_reasoning() {
        let chunks = vec![
            Ok(json!({"choices": [{"delta": {"reasoning_content": "think "}}]})),
            Ok(json!({"choices": [{"delta": {"reasoning_content": "hard"}}]})),
            Ok(json!({"choices": [{"delta": {"content": "Answer"}}]})),
        ];
        let mut st = StreamState {
            inner: futures::stream::iter(chunks).boxed(),
            pending: VecDeque::new(),
            text: String::new(),
            refusal: String::new(),
            message_started: false,
            tool_calls: BTreeMap::new(),
            reasoning: None,
            usage: None,
            finished: false,
            model: "deepseek-reasoner".into(),
            span: crate::tracing::create_span(
                SpanData::Custom {
                    name: "t".into(),
                    data: Value::Null,
                },
                None,
                true,
            ),
            include_data: false,
        };
        while let Some(Ok(c)) = st.inner.next().await {
            st.handle_chunk(&c);
        }
        st.finish();
        let events: Vec<Value> = st.pending.into_iter().map(|e| e.unwrap()).collect();
        let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(
            &types[..5],
            [
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
            ]
        );
        let text_delta = events
            .iter()
            .find(|e| e["type"] == "response.output_text.delta")
            .unwrap();
        assert_eq!(text_delta["output_index"], 1);
        let out = &events.last().unwrap()["response"]["output"];
        assert_eq!(out[0]["summary"][0]["text"], "think hard");
        assert_eq!(out[0]["provider_data"]["model"], "deepseek-reasoner");
        assert_eq!(out[1]["content"][0]["text"], "Answer");

        // Replay into the next DeepSeek request, but not for other models.
        let items: Vec<InputItem> = vec![
            InputItem::user("q"),
            out[0].clone().into(),
            json!({"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"})
                .into(),
            json!({"type": "function_call_output", "call_id": "c1", "output": "ok"}).into(),
        ];
        let opts = |m: &str| ChatConversionOptions {
            model: Some(m.into()),
            ..Default::default()
        };
        let msgs = items_to_messages_with(&items, None, &opts("deepseek-reasoner")).unwrap();
        assert_eq!(msgs[1]["reasoning_content"], "think hard");
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "c1");
        let msgs = items_to_messages_with(&items, None, &opts("gpt-4o")).unwrap();
        assert!(msgs[1].get("reasoning_content").is_none());
    }

    #[test]
    fn parses_reasoning_fields() {
        let v = json!({"model": "deepseek-r1", "choices": [{"message": {"role": "assistant",
            "content": "hi", "reasoning_content": "because"}}]});
        let r = parse_chat_completion(&v, None, false).unwrap();
        assert!(matches!(&r.output[0], OutputItem::Reasoning(_)));
        let v = json!({"model": "or/x", "choices": [{"message": {"role": "assistant",
            "content": "hi", "reasoning": "steps"}}]});
        let r = parse_chat_completion(&v, None, false).unwrap();
        let item = r.output[0].to_value();
        assert_eq!(item["content"][0]["text"], "steps");
        assert_eq!(item["provider_data"][REASONING_FIELD_KEY], "reasoning");
    }
}
