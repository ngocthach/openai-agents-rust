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

/// Rewrites the converted chat messages just before they are sent.
pub type ChatMessageTransform = Arc<dyn Fn(Vec<Value>) -> Vec<Value> + Send + Sync>;

/// Rewrites the whole request body just before it is sent.
pub type ChatBodyTransform = Arc<dyn Fn(Value) -> Value + Send + Sync>;

/// The default Chat Completions route.
pub const DEFAULT_CHAT_PATH: &str = "chat/completions";

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

/// Replaces a tool output that Chat Completions cannot represent.
pub const OMITTED_TOOL_OUTPUT_PLACEHOLDER: &str = "[tool output omitted]";

/// Options for converting Responses input items to Chat Completions messages.
#[derive(Clone, Default)]
pub struct ChatConversionOptions {
    /// The target model (enables reasoning replay).
    pub model: Option<String>,
    /// The target base URL.
    pub base_url: Option<String>,
    /// Custom reasoning replay rule.
    pub should_replay_reasoning_content: Option<ShouldReplayReasoningContent>,
    /// Keep every content part of a multimodal tool output instead of reducing it to text.
    /// Providers reached through the adapters in [`crate::extensions::models`] accept this.
    pub preserve_tool_output_all_content: bool,
    /// Reject what Chat Completions cannot represent instead of warning and degrading.
    pub strict_feature_validation: bool,
    /// Replay Anthropic interleaved-thinking blocks on the assistant message that follows a
    /// reasoning item. Mirrors `Converter.items_to_messages(preserve_thinking_blocks=...)`,
    /// which LiteLLM turns on whenever `reasoning.effort` is set.
    pub preserve_thinking_blocks: bool,
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
    strict_feature_validation: bool,
    preserve_tool_output_all_content: bool,
    message_transform: Option<ChatMessageTransform>,
    preserve_thinking_blocks: bool,
    buffer_streamed_tool_calls: bool,
    body_transform: Option<ChatBodyTransform>,
    chat_path: String,
    /// Each unsupported-feature warning is emitted once per model, as in Python.
    warned_unsupported: Arc<[std::sync::atomic::AtomicBool; 2]>,
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
            strict_feature_validation: false,
            preserve_tool_output_all_content: false,
            message_transform: None,
            preserve_thinking_blocks: false,
            buffer_streamed_tool_calls: false,
            body_transform: None,
            chat_path: DEFAULT_CHAT_PATH.to_string(),
            warned_unsupported: Arc::new([
                std::sync::atomic::AtomicBool::new(false),
                std::sync::atomic::AtomicBool::new(false),
            ]),
        }
    }

    /// Keep every content part of a multimodal tool output instead of reducing it to text.
    pub fn with_preserve_tool_output_all_content(mut self, preserve: bool) -> Self {
        self.preserve_tool_output_all_content = preserve;
        self
    }

    /// Rewrite the converted chat messages just before they are sent. Used by the adapters in
    /// [`crate::extensions::models`] to satisfy provider-specific message ordering rules.
    pub fn with_message_transform<F>(mut self, f: F) -> Self
    where
        F: Fn(Vec<Value>) -> Vec<Value> + Send + Sync + 'static,
    {
        self.message_transform = Some(Arc::new(f));
        self
    }

    /// Emit each streamed function tool call once, complete, instead of forwarding the
    /// provider's argument fragments as `response.function_call_arguments.delta` events.
    /// Mirrors `buffer_streamed_tool_calls`; off by default, as in Python.
    pub fn with_buffer_streamed_tool_calls(mut self, buffer: bool) -> Self {
        self.buffer_streamed_tool_calls = buffer;
        self
    }

    /// Replay Anthropic interleaved-thinking blocks (see
    /// [`ChatConversionOptions::preserve_thinking_blocks`]).
    pub fn with_preserve_thinking_blocks(mut self, preserve: bool) -> Self {
        self.preserve_thinking_blocks = preserve;
        self
    }

    /// Rewrite the whole request body just before it is sent. Providers whose endpoint is
    /// OpenAI-shaped except for a couple of envelope keys (watsonx's `model_id`/`project_id`,
    /// for example) are reached this way instead of through a second model implementation.
    pub fn with_body_transform<F>(mut self, f: F) -> Self
    where
        F: Fn(Value) -> Value + Send + Sync + 'static,
    {
        self.body_transform = Some(Arc::new(f));
        self
    }

    /// Use a different route than `chat/completions` (relative to the client base URL).
    pub fn with_chat_path(mut self, path: impl Into<String>) -> Self {
        self.chat_path = path.into();
        self
    }

    /// Reject Responses-only features (`previous_response_id`, `conversation_id`, `prompt`)
    /// instead of warning and ignoring them.
    pub fn with_strict_feature_validation(mut self, strict: bool) -> Self {
        self.strict_feature_validation = strict;
        self
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

    /// Warn about, or reject, Responses-only features a Chat Completions model cannot honour.
    fn validate_responses_only_features(&self, req: &ModelRequest) -> Result<()> {
        if req.prompt.is_some() {
            let message = "Reusable prompts are only supported by the Responses API. \
                 OpenAIChatCompletionsModel does not support `prompt`; use a Responses model \
                 instead.";
            if self.strict_feature_validation {
                return Err(AgentsError::user(message));
            }
            if !self.warned_unsupported[0].swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!(
                    target: crate::logger::LOG_TARGET,
                    "{message} Ignoring `prompt`; enable strict feature validation to raise an \
                     error instead."
                );
            }
        }
        let mut unsupported = Vec::new();
        if req.previous_response_id.is_some() {
            unsupported.push("previous_response_id");
        }
        if req.conversation_id.is_some() {
            unsupported.push("conversation_id");
        }
        if !unsupported.is_empty() {
            let message = format!(
                "OpenAIChatCompletionsModel does not support server-managed conversation state \
                 ({}). Chat Completions requires callers to pass the full conversation history; \
                 use a Responses API model for previous_response_id or a conversation-capable \
                 model for conversation_id.",
                unsupported.join(", ")
            );
            if self.strict_feature_validation {
                return Err(AgentsError::user(message));
            }
            if !self.warned_unsupported[1].swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!(
                    target: crate::logger::LOG_TARGET,
                    "{message} Ignoring unsupported server-managed conversation state; enable \
                     strict feature validation to raise an error instead."
                );
            }
        }
        Ok(())
    }

    /// [`Self::build_request_body`] with the provider body transform applied.
    pub fn transformed_body(&self, req: &ModelRequest, stream: bool) -> Result<Value> {
        let body = self.build_request_body(req, stream)?;
        Ok(match &self.body_transform {
            Some(f) => f(body),
            None => body,
        })
    }

    /// Build the JSON request body.
    pub fn build_request_body(&self, req: &ModelRequest, stream: bool) -> Result<Value> {
        self.validate_responses_only_features(req)?;
        let s = &req.model_settings;
        let mut messages = items_to_messages_with(
            &req.input,
            req.system_instructions.as_deref(),
            &ChatConversionOptions {
                model: Some(self.model.clone()),
                base_url: Some(self.client.base_url().to_string()),
                should_replay_reasoning_content: self.should_replay_reasoning_content.clone(),
                preserve_tool_output_all_content: self.preserve_tool_output_all_content,
                strict_feature_validation: self.strict_feature_validation,
                // Claude's interleaved thinking needs the signed blocks replayed whenever the
                // caller asked for reasoning, exactly as LiteLLM does.
                preserve_thinking_blocks: self.preserve_thinking_blocks
                    && req
                        .model_settings
                        .reasoning
                        .as_ref()
                        .is_some_and(|r| r.effort.is_some()),
            },
        )?;
        if let Some(transform) = &self.message_transform {
            // The system message is inserted first, so the transform sees it too, as in Python
            // where it runs before the system message is prepended only for tool ordering.
            let system = (req.system_instructions.is_some()).then(|| messages.remove(0));
            messages = transform(messages);
            if let Some(system) = system {
                messages.insert(0, system);
            }
        }
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
            // Chat Completions rejects parallel_tool_calls unless tools are present.
            if let Some(p) = s.parallel_tool_calls {
                body.insert("parallel_tool_calls".into(), Value::Bool(p));
            }
        }
        // `tool_choice` is sent whenever it is set, so `tool_choice: "none"` without tools
        // still reaches the provider, as in the Python SDK.
        if let Some(tc) = &s.tool_choice {
            body.insert("tool_choice".into(), convert_chat_tool_choice(tc));
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
            let duplicates: Vec<&str> = {
                let mut keys: Vec<&str> = extra
                    .keys()
                    .filter(|k| body.contains_key(k.as_str()))
                    .map(String::as_str)
                    .collect();
                keys.sort_unstable();
                keys
            };
            if !duplicates.is_empty() {
                let keys = duplicates
                    .iter()
                    .map(|k| format!("'{k}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let plural = if duplicates.len() == 1 {
                    "argument"
                } else {
                    "arguments"
                };
                return Err(AgentsError::user(format!(
                    "chat.completions.create() got multiple values for keyword {plural} {keys}"
                )));
            }
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
            Some(super::trace::model_config_for_trace(
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

/// Convert Chat Completions `url_citation` annotations into Responses output-text annotations.
///
/// A provider that reports an incomplete citation is skipped rather than failing the turn.
pub fn convert_url_citations(raw: Option<&Value>) -> Vec<Value> {
    let Some(Value::Array(items)) = raw else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for annotation in items {
        if annotation.get("type").and_then(Value::as_str) != Some("url_citation") {
            continue;
        }
        let Some(citation) = annotation.get("url_citation") else {
            continue;
        };
        let (Some(start), Some(end), Some(url)) = (
            citation.get("start_index").and_then(Value::as_u64),
            citation.get("end_index").and_then(Value::as_u64),
            citation.get("url").and_then(Value::as_str),
        ) else {
            crate::logger::log_model_action_debug("Skipping malformed url citation", annotation);
            continue;
        };
        out.push(json!({
            "type": "url_citation",
            "start_index": start,
            "end_index": end,
            "url": url,
            "title": citation.get("title").and_then(Value::as_str).unwrap_or_default(),
        }));
    }
    out
}

fn chat_token_logprobs(logprobs: Option<&Value>) -> &[Value] {
    logprobs
        .and_then(|l| l.get("content"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

/// Convert Chat Completions token logprobs into the Responses `output_text` logprob shape.
pub fn convert_logprobs_for_output_text(logprobs: Option<&Value>) -> Option<Value> {
    let tokens = chat_token_logprobs(logprobs);
    if tokens.is_empty() {
        return None;
    }
    let bytes = |v: &Value| v.get("bytes").cloned().unwrap_or_else(|| json!([]));
    Some(Value::Array(
        tokens
            .iter()
            .map(|t| {
                json!({
                    "token": t.get("token").cloned().unwrap_or_default(),
                    "logprob": t.get("logprob").cloned().unwrap_or_else(|| json!(0.0)),
                    "bytes": bytes(t),
                    "top_logprobs": t.get("top_logprobs").and_then(Value::as_array)
                        .map(|tops| tops.iter().map(|top| json!({
                            "token": top.get("token").cloned().unwrap_or_default(),
                            "logprob": top.get("logprob").cloned().unwrap_or_else(|| json!(0.0)),
                            "bytes": bytes(top),
                        })).collect::<Vec<_>>())
                        .unwrap_or_default(),
                })
            })
            .collect(),
    ))
}

/// Convert Chat Completions token logprobs into the `response.output_text.delta` logprob shape.
pub fn convert_logprobs_for_text_delta(logprobs: Option<&Value>) -> Vec<Value> {
    chat_token_logprobs(logprobs)
        .iter()
        .map(|t| {
            let tops = t
                .get("top_logprobs")
                .and_then(Value::as_array)
                .map(|tops| {
                    tops.iter()
                        .map(|top| {
                            json!({
                                "token": top.get("token").cloned().unwrap_or_default(),
                                "logprob": top.get("logprob").cloned()
                                    .unwrap_or_else(|| json!(0.0)),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            json!({
                "token": t.get("token").cloned().unwrap_or_default(),
                "logprob": t.get("logprob").cloned().unwrap_or_else(|| json!(0.0)),
                "top_logprobs": if tops.is_empty() { Value::Null } else { json!(tops) },
            })
        })
        .collect()
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

/// Attach replayed Anthropic thinking blocks to an assistant message.
/// Port of `apply_pending_thinking_blocks`.
///
/// A native block list (LiteLLM's `thinking_blocks` field) keeps the complete signed sequence,
/// including empty and redacted blocks, and travels in its own key. Blocks reconstructed from a
/// stored reasoning item instead go in front of the message content, which is the shape the
/// older stored items were replayed as.
fn apply_pending_thinking_blocks(
    asst: &mut Map<String, Value>,
    pending: &mut Option<(Vec<Value>, bool)>,
) {
    let Some((blocks, native)) = pending.take() else {
        return;
    };
    if blocks.is_empty() {
        return;
    }
    if native {
        asst.insert("thinking_blocks".into(), Value::Array(blocks));
        return;
    }
    let mut parts = blocks;
    match asst.get("content") {
        Some(Value::String(text)) => parts.push(json!({"type": "text", "text": text})),
        Some(Value::Array(existing)) => parts.extend(existing.iter().cloned()),
        _ => {}
    }
    asst.insert("content".into(), Value::Array(parts));
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
    // Anthropic thinking blocks waiting to be replayed: (blocks, came from a native field).
    let mut pending_thinking: Option<(Vec<Value>, bool)> = None;
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
    let apply_reasoning = |asst: &mut Map<String, Value>,
                           rc: &mut Option<String>,
                           r: &mut Option<String>,
                           tb: &mut Option<(Vec<Value>, bool)>| {
        if let Some(text) = rc.take() {
            asst.insert("reasoning_content".into(), Value::String(text));
        }
        if let Some(text) = r.take() {
            asst.insert("reasoning".into(), Value::String(text));
        }
        apply_pending_thinking_blocks(asst, tb);
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
                pending_thinking = None;
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
                    &mut pending_thinking,
                );
                current = Some(asst);
            }
            (Some("reasoning"), _) => {
                flush(&mut result, &mut current);
                pending_reasoning_content = None;
                pending_reasoning = None;
                pending_thinking = None;
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
                    // Claude interleaved thinking: replay the signed blocks on the next
                    // assistant message. Items may come from another provider, so only an
                    // item whose model matches (or that carries no other provider metadata)
                    // counts as Claude's own.
                    let lower = model.to_lowercase();
                    let origin_unknown = provider_data.keys().all(|k| k == "thinking_blocks");
                    if options.preserve_thinking_blocks
                        && (lower.contains("claude") || lower.contains("anthropic"))
                        && (*model == item_model || origin_unknown)
                    {
                        let native = provider_data
                            .get("thinking_blocks")
                            .and_then(Value::as_array)
                            .filter(|b| !b.is_empty() && b.iter().all(Value::is_object));
                        if let Some(blocks) = native {
                            pending_reasoning = None;
                            pending_thinking = Some((blocks.clone(), true));
                        } else if provider_data
                            .get(REASONING_FIELD_KEY)
                            .and_then(Value::as_str)
                            != Some("reasoning")
                        {
                            let mut signatures: VecDeque<&str> = v
                                .get("encrypted_content")
                                .and_then(Value::as_str)
                                .map(|c| c.split('\n').collect())
                                .unwrap_or_default();
                            let blocks: Vec<Value> = v
                                .get("content")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .filter(|p| {
                                    p.get("type").and_then(Value::as_str) == Some("reasoning_text")
                                })
                                .map(|p| {
                                    let mut block = json!({
                                        "type": "thinking",
                                        "thinking": p.get("text").cloned()
                                            .unwrap_or_else(|| Value::String(String::new())),
                                    });
                                    if let Some(sig) = signatures.pop_front() {
                                        block["signature"] = Value::String(sig.to_string());
                                    }
                                    block
                                })
                                .collect();
                            if !blocks.is_empty() {
                                pending_thinking = Some((blocks, false));
                            }
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
                apply_reasoning(
                    asst,
                    &mut pending_reasoning_content,
                    &mut pending_reasoning,
                    &mut pending_thinking,
                );
                let mut call = json!({
                    "id": v.get("call_id").cloned().unwrap_or(Value::Null),
                    "type": "function",
                    "function": {
                        "name": v.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": v.get("arguments").cloned().unwrap_or(Value::String("{}".into())),
                    }
                });
                // Gemini rejects a replayed tool call whose thought signature was dropped.
                if options
                    .model
                    .as_deref()
                    .is_some_and(|m| m.to_lowercase().contains("gemini"))
                    && let Some(sig) = v
                        .pointer("/provider_data/thought_signature")
                        .and_then(Value::as_str)
                {
                    call["extra_content"] = json!({"google": {"thought_signature": sig}});
                }
                if let Some(Value::Array(calls)) = asst.get_mut("tool_calls") {
                    calls.push(call);
                }
            }
            (Some("function_call_output"), _) => {
                flush(&mut result, &mut current);
                pending_reasoning_content = None;
                pending_reasoning = None;
                pending_thinking = None;
                let Some(call_id) = v.get("call_id").filter(|c| !c.is_null()).cloned() else {
                    return Err(AgentsError::user(
                        "Unpaired function outputs are supported by Responses but cannot be \
                         converted to Chat Completions tool messages. Use a Responses model to \
                         preserve this input.",
                    ));
                };
                let output = v.get("output").cloned().unwrap_or(Value::Null);
                let content = match &output {
                    Value::String(s) => Value::String(s.clone()),
                    Value::Array(parts) => {
                        let converted: Vec<Value> =
                            parts.iter().map(convert_user_part).collect::<Result<_>>()?;
                        if options.preserve_tool_output_all_content {
                            Value::Array(converted)
                        } else {
                            let text_only: Vec<Value> = converted
                                .into_iter()
                                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                                .collect();
                            if text_only.is_empty() {
                                let message = "Chat Completions tool outputs cannot be empty or \
                                     contain only non-text content unless \
                                     preserve_tool_output_all_content is set.";
                                if options.strict_feature_validation {
                                    return Err(AgentsError::user(message));
                                }
                                log::warn!(
                                    target: crate::logger::LOG_TARGET,
                                    "{message} Replacing the tool output with a placeholder; \
                                     enable strict feature validation to raise an error instead."
                                );
                                Value::String(OMITTED_TOOL_OUTPUT_PLACEHOLDER.to_string())
                            } else {
                                Value::Array(text_only)
                            }
                        }
                    }
                    other => Value::String(other.to_string()),
                };
                result.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }));
            }
            (Some("item_reference"), _) => {
                return Err(AgentsError::user(
                    "Encountered an item_reference, which is not supported by Chat Completions",
                ));
            }
            (Some(other), _) => {
                log::debug!(target: crate::logger::LOG_TARGET, "Skipping item type {other} not supported by Chat Completions");
            }
            (None, _) => {
                return Err(AgentsError::user(format!("Unhandled input item: {v}")));
            }
        }
    }
    flush(&mut result, &mut current);
    Ok(result)
}

/// Carry a provider's per-tool-call metadata (today: Google's thought signature) on the
/// function call item, so it can be replayed on the next request.
/// Port of the `extra_content` -> `provider_data` conversion in `message_to_output_items`.
fn tool_call_provider_data(call: &Value) -> Map<String, Value> {
    let mut extra = Map::new();
    if let Some(sig) = call
        .pointer("/extra_content/google/thought_signature")
        .and_then(Value::as_str)
    {
        extra.insert("provider_data".into(), json!({"thought_signature": sig}));
    }
    extra
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
    let thinking_blocks = message
        .get("thinking_blocks")
        .and_then(Value::as_array)
        .map(|b| {
            b.iter()
                .filter(|v| v.is_object())
                .cloned()
                .collect::<Vec<_>>()
        })
        .filter(|b: &Vec<Value>| !b.is_empty());
    if !reasoning_content.is_empty() || !reasoning.is_empty() || thinking_blocks.is_some() {
        let mut provider_data = Map::new();
        if !model.is_empty() {
            provider_data.insert("model".into(), json!(model));
        }
        let mut item = json!({"type": "reasoning", "id": FAKE_RESPONSES_ID, "summary": []});
        if !reasoning_content.is_empty() {
            item["summary"] = json!([{"type": "summary_text", "text": reasoning_content}]);
        } else if !reasoning.is_empty() {
            item["content"] = json!([{"type": "reasoning_text", "text": reasoning}]);
            provider_data.insert(REASONING_FIELD_KEY.into(), json!("reasoning"));
        }
        // Claude's signed interleaved-thinking blocks must survive a round trip verbatim, so
        // keep the whole sequence and also the normalized text/signature view.
        if let Some(blocks) = thinking_blocks {
            let mut text = String::new();
            let mut last_signature: Option<String> = None;
            for block in &blocks {
                if block.get("type").and_then(Value::as_str) != Some("thinking") {
                    continue;
                }
                text.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
                if let Some(sig) = block.get("signature").and_then(Value::as_str) {
                    last_signature = Some(sig.to_string());
                }
            }
            provider_data.insert("thinking_blocks".into(), Value::Array(blocks));
            if !text.is_empty() {
                let content = item
                    .get_mut("content")
                    .and_then(Value::as_array_mut)
                    .map(std::mem::take)
                    .unwrap_or_default();
                let mut content = content;
                content.push(json!({"type": "reasoning_text", "text": text}));
                item["content"] = Value::Array(content);
            }
            if let Some(sig) = &last_signature {
                item["encrypted_content"] = json!(sig);
            }
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
                annotations: convert_url_citations(message.get("annotations")),
                logprobs: convert_logprobs_for_output_text(choice.get("logprobs")),
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
                extra: tool_call_provider_data(c),
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
    /// Accumulated url-citation annotations on the assistant text part.
    annotations: Vec<Value>,
    /// Accumulated Responses-shaped logprobs on the assistant text part.
    logprobs: Vec<Value>,
    refusal: String,
    message_started: bool,
    tool_calls: BTreeMap<u64, (String, String, String)>,
    /// Hold streamed tool-call deltas back and emit each call once, complete.
    buffer_tool_calls: bool,
    /// Output indices are handed out in announcement order, so a message that starts after a
    /// tool call cannot take the index that call already used.
    next_output_index: usize,
    reasoning_index: Option<usize>,
    message_output_index: Option<usize>,
    /// Chat Completions tool-call index -> Responses output index, once announced.
    call_indexes: BTreeMap<u64, usize>,
    /// Streamed reasoning: (summary text, reasoning text, summary part open).
    reasoning: Option<(String, String, bool)>,
    /// The ordered Anthropic thinking-block sequence, rebuilt from the streamed deltas.
    /// Text and signatures cannot be flattened: each block carries its own signature and a
    /// `redacted_thinking` block carries neither.
    thinking_blocks: Vec<Value>,
    usage: Option<Value>,
    finished: bool,
    model: String,
    span: Span,
    include_data: bool,
}

impl StreamState {
    fn message_index(&self) -> usize {
        self.message_output_index.unwrap_or(0)
    }

    fn take_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
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
        let index = self.take_output_index();
        self.reasoning_index = Some(index);
        let mut item = json!({"type": "reasoning", "id": FAKE_RESPONSES_ID, "summary": [],
            "provider_data": self.reasoning_provider_data(reasoning_field)});
        if reasoning_field {
            item["content"] = json!([{"type": "reasoning_text", "text": ""}]);
        }
        self.pending.push_back(Ok(json!({
            "type": "response.output_item.added", "output_index": index, "item": item,
        })));
    }

    fn close_summary_part(&mut self) {
        let reasoning_index = self.reasoning_index.unwrap_or(0);
        if let Some((summary, _, open)) = &mut self.reasoning
            && *open
        {
            *open = false;
            let text = summary.clone();
            self.pending.push_back(Ok(json!({
                "type": "response.reasoning_summary_text.done", "item_id": FAKE_RESPONSES_ID,
                "output_index": reasoning_index, "summary_index": 0, "text": text,
            })));
            self.pending.push_back(Ok(json!({
                "type": "response.reasoning_summary_part.done", "item_id": FAKE_RESPONSES_ID,
                "output_index": reasoning_index, "summary_index": 0,
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
        let index = self.take_output_index();
        self.message_output_index = Some(index);
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

    /// Fold one streamed thinking delta into the ordered block sequence.
    ///
    /// Anthropic streams a thinking block as a run of text deltas terminated by a single
    /// signature delta, so a signature both belongs to the block being accumulated and ends it.
    /// A `redacted_thinking` block arrives whole and is kept verbatim.
    /// Port of `StreamingState.accumulate_thinking_block`.
    fn accumulate_thinking_block(&mut self, block: &Value) {
        let Some(block) = block.as_object() else {
            return;
        };
        let block_type = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("thinking");
        if block_type == "redacted_thinking" {
            self.thinking_blocks.push(Value::Object(block.clone()));
            return;
        }
        if block_type != "thinking" {
            return;
        }
        let thinking = block.get("thinking").and_then(Value::as_str).unwrap_or("");
        let signature = block.get("signature").and_then(Value::as_str).unwrap_or("");
        let metadata: Vec<(&String, &Value)> = block
            .iter()
            .filter(|(k, _)| !matches!(k.as_str(), "type" | "thinking" | "signature"))
            .collect();
        if thinking.is_empty() && signature.is_empty() && metadata.is_empty() {
            return;
        }
        let reopen = self.thinking_blocks.last().is_some_and(|b| {
            b.get("type").and_then(Value::as_str) == Some("thinking")
                && b.get("signature").is_none()
        });
        if !reopen {
            self.thinking_blocks
                .push(json!({"type": "thinking", "thinking": ""}));
        }
        let current = self
            .thinking_blocks
            .last_mut()
            .and_then(Value::as_object_mut)
            .expect("open thinking block");
        for (k, v) in metadata {
            current.insert(k.clone(), v.clone());
        }
        if !thinking.is_empty() {
            let existing = current
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            current.insert("thinking".into(), json!(format!("{existing}{thinking}")));
        }
        if !signature.is_empty() {
            current.insert("signature".into(), json!(signature));
        }
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
            let reasoning_index = self.reasoning_index.unwrap_or(0);
            if let Some((summary, _, open)) = &mut self.reasoning
                && !self.message_started
            {
                if !*open {
                    *open = true;
                    self.pending.push_back(Ok(json!({
                        "type": "response.reasoning_summary_part.added",
                        "item_id": FAKE_RESPONSES_ID, "output_index": reasoning_index, "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    })));
                }
                summary.push_str(rc);
                self.pending.push_back(Ok(json!({
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": FAKE_RESPONSES_ID, "output_index": reasoning_index, "summary_index": 0,
                    "delta": rc,
                })));
            }
        }
        if let Some(blocks) = delta.get("thinking_blocks").and_then(Value::as_array)
            && !blocks.is_empty()
        {
            self.start_reasoning(false);
            for block in blocks {
                self.accumulate_thinking_block(block);
            }
        }
        if let Some(r) = delta.get("reasoning").and_then(Value::as_str)
            && !r.is_empty()
        {
            self.start_reasoning(true);
            let reasoning_index = self.reasoning_index.unwrap_or(0);
            if let Some((_, text, _)) = &mut self.reasoning
                && !self.message_started
            {
                text.push_str(r);
                self.pending.push_back(Ok(json!({
                    "type": "response.reasoning_text.delta",
                    "item_id": FAKE_RESPONSES_ID, "output_index": reasoning_index, "content_index": 0,
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
        let choice_logprobs = chunk.pointer("/choices/0/logprobs").cloned();
        if let Some(t) = delta.get("content").and_then(Value::as_str)
            && !t.is_empty()
        {
            self.start_message();
            self.text.push_str(t);
            let index = self.message_index();
            let delta_logprobs = convert_logprobs_for_text_delta(choice_logprobs.as_ref());
            if let Some(Value::Array(output_logprobs)) =
                convert_logprobs_for_output_text(choice_logprobs.as_ref())
            {
                self.logprobs.extend(output_logprobs);
            }
            self.pending.push_back(Ok(json!({
                "type": "response.output_text.delta",
                "item_id": FAKE_RESPONSES_ID,
                "output_index": index,
                "content_index": 0,
                "delta": t,
                "logprobs": delta_logprobs,
            })));
        }
        // Citations can arrive on the delta carrying the cited text or on a later one.
        if self.message_started {
            self.annotations
                .extend(convert_url_citations(delta.get("annotations")));
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
            for c in calls.clone() {
                let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
                let entry = self.tool_calls.entry(idx).or_default();
                // The id and the name are assigned, never appended: a provider that repeats
                // them on every chunk must not produce a duplicated `call_id` or `name`.
                if let Some(id) = c.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    entry.0.clear();
                    entry.0.push_str(id);
                }
                if let Some(n) = c.pointer("/function/name").and_then(Value::as_str)
                    && !n.is_empty()
                {
                    entry.1.clear();
                    entry.1.push_str(n);
                }
                let arguments = c
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                entry.2.push_str(&arguments);
                if self.buffer_tool_calls {
                    continue;
                }
                // Announce the call as soon as its id and name are known, then forward each
                // argument fragment, as Python does when tool-call buffering is off.
                let (id, name) = (entry.0.clone(), entry.1.clone());
                if !self.call_indexes.contains_key(&idx) {
                    if id.is_empty() || name.is_empty() {
                        continue;
                    }
                    let output_index = self.take_output_index();
                    self.call_indexes.insert(idx, output_index);
                    self.pending.push_back(Ok(json!({
                        "type": "response.output_item.added", "output_index": output_index,
                        "item": {"type": "function_call", "id": FAKE_RESPONSES_ID,
                                 "call_id": id, "name": name, "arguments": ""},
                    })));
                }
                if !arguments.is_empty() {
                    let output_index = self.call_indexes[&idx];
                    self.pending.push_back(Ok(json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": FAKE_RESPONSES_ID, "output_index": output_index,
                        "delta": arguments,
                    })));
                }
            }
        }
    }

    fn finish(&mut self) {
        let mut slots: BTreeMap<usize, Value> = BTreeMap::new();
        self.close_summary_part();
        let reasoning_index = self.reasoning_index.unwrap_or(0);
        if let Some((summary, mut text, _)) = self.reasoning.clone() {
            let reasoning_field = summary.is_empty() && !text.is_empty();
            let mut item = json!({"type": "reasoning", "id": FAKE_RESPONSES_ID,
                "summary": if summary.is_empty() { json!([]) }
                           else { json!([{"type": "summary_text", "text": summary}]) },
                "provider_data": self.reasoning_provider_data(reasoning_field)});
            if !self.thinking_blocks.is_empty() {
                let mut thinking_text = String::new();
                let mut last_signature: Option<String> = None;
                for block in &self.thinking_blocks {
                    if block.get("type").and_then(Value::as_str) != Some("thinking") {
                        continue;
                    }
                    thinking_text
                        .push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
                    if let Some(sig) = block.get("signature").and_then(Value::as_str) {
                        last_signature = Some(sig.to_string());
                    }
                }
                item["provider_data"]["thinking_blocks"] = json!(self.thinking_blocks);
                if let Some(sig) = &last_signature {
                    item["encrypted_content"] = json!(sig);
                }
                if text.is_empty() {
                    text = thinking_text;
                }
            }
            if !text.is_empty() {
                item["content"] = json!([{"type": "reasoning_text", "text": text}]);
                self.pending.push_back(Ok(json!({
                    "type": "response.reasoning_text.done", "item_id": FAKE_RESPONSES_ID,
                    "output_index": reasoning_index, "content_index": 0, "text": text,
                })));
            }
            self.pending.push_back(Ok(json!({
                "type": "response.output_item.done", "output_index": reasoning_index, "item": item.clone(),
            })));
            slots.insert(reasoning_index, item);
        }
        let mi = self.message_index();
        if self.message_started {
            let mut content = Vec::new();
            if !self.text.is_empty() {
                let mut part = json!({"type": "output_text", "text": self.text,
                    "annotations": self.annotations});
                if !self.logprobs.is_empty() {
                    part["logprobs"] = json!(self.logprobs);
                }
                content.push(part.clone());
                self.pending.push_back(Ok(json!({
                    "type": "response.output_text.done",
                    "item_id": FAKE_RESPONSES_ID, "output_index": mi, "content_index": 0,
                    "text": self.text, "logprobs": self.logprobs,
                })));
                self.pending.push_back(Ok(json!({
                    "type": "response.content_part.done",
                    "item_id": FAKE_RESPONSES_ID, "output_index": mi, "content_index": 0,
                    "part": part,
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
            slots.insert(mi, item);
        }
        let calls: Vec<(u64, (String, String, String))> = self
            .tool_calls
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        for (idx, (id, name, args)) in calls {
            let args = if args.is_empty() {
                "{}".to_string()
            } else {
                args
            };
            let announced = self.call_indexes.get(&idx).copied();
            let output_index = match announced {
                Some(i) => i,
                None => {
                    let output_index = self.take_output_index();
                    self.call_indexes.insert(idx, output_index);
                    self.pending.push_back(Ok(json!({
                        "type": "response.output_item.added", "output_index": output_index,
                        "item": {"type": "function_call", "id": FAKE_RESPONSES_ID,
                                 "call_id": id, "name": name, "arguments": ""},
                    })));
                    self.pending.push_back(Ok(json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": FAKE_RESPONSES_ID, "output_index": output_index,
                        "delta": args,
                    })));
                    output_index
                }
            };
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
            slots.insert(output_index, item);
        }
        let output: Vec<Value> = slots.into_values().collect();
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
    fn supports_default_prompt_cache_key(&self) -> bool {
        self.client.is_openai_api()
    }

    fn is_chat_completions(&self) -> bool {
        true
    }

    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = self.transformed_body(&request, false)?;
        let span = self.start_span(&request, &body);
        let result = async {
            let (value, request_id) = self
                .client
                .post_json(
                    &self.chat_path,
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
        let include_data = request.tracing.include_data();
        match &result {
            Ok(r) => finish_generation_span(&span, r, include_data),
            Err(e) => {
                crate::tracing::record_model_error_on_span(
                    &span,
                    "Error getting response",
                    e,
                    include_data,
                );
                crate::logger::log_model_action_error("Error getting response", e);
            }
        }
        span.finish();
        result
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let body = self.transformed_body(&request, true)?;
        let span = self.start_span(&request, &body);
        let (inner, _) = match self
            .client
            .post_sse(
                &self.chat_path,
                &body,
                request.model_settings.extra_headers.as_ref(),
                request.model_settings.extra_query.as_ref(),
            )
            .await
        {
            Ok(s) => s,
            Err(e) => {
                crate::tracing::record_model_error_on_span(
                    &span,
                    "Error streaming response",
                    &e,
                    request.tracing.include_data(),
                );
                crate::logger::log_model_action_error("Error streaming response", &e);
                span.finish();
                return Err(e);
            }
        };
        let mut state = StreamState {
            inner,
            pending: VecDeque::new(),
            text: String::new(),
            annotations: Vec::new(),
            logprobs: Vec::new(),
            refusal: String::new(),
            message_started: false,
            tool_calls: BTreeMap::new(),
            buffer_tool_calls: self.buffer_streamed_tool_calls,
            next_output_index: 0,
            reasoning_index: None,
            message_output_index: None,
            call_indexes: BTreeMap::new(),
            reasoning: None,
            thinking_blocks: Vec::new(),
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
                        crate::tracing::record_model_error_on_span(
                            &st.span,
                            "Error streaming response",
                            &e,
                            st.include_data,
                        );
                        crate::logger::log_model_action_error("Error streaming response", &e);
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
            annotations: Vec::new(),
            logprobs: Vec::new(),
            refusal: String::new(),
            message_started: false,
            tool_calls: BTreeMap::new(),
            buffer_tool_calls: false,
            next_output_index: 0,
            reasoning_index: None,
            message_output_index: None,
            call_indexes: BTreeMap::new(),
            reasoning: None,
            thinking_blocks: Vec::new(),
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
            annotations: Vec::new(),
            logprobs: Vec::new(),
            refusal: String::new(),
            message_started: false,
            tool_calls: BTreeMap::new(),
            buffer_tool_calls: false,
            next_output_index: 0,
            reasoning_index: None,
            message_output_index: None,
            call_indexes: BTreeMap::new(),
            reasoning: None,
            thinking_blocks: Vec::new(),
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

    fn stream_state(model: &str, chunks: Vec<Result<Value>>) -> StreamState {
        StreamState {
            inner: futures::stream::iter(chunks).boxed(),
            pending: VecDeque::new(),
            text: String::new(),
            annotations: Vec::new(),
            logprobs: Vec::new(),
            refusal: String::new(),
            message_started: false,
            tool_calls: BTreeMap::new(),
            buffer_tool_calls: false,
            next_output_index: 0,
            reasoning_index: None,
            message_output_index: None,
            call_indexes: BTreeMap::new(),
            reasoning: None,
            thinking_blocks: Vec::new(),
            usage: None,
            finished: false,
            model: model.into(),
            span: crate::tracing::create_span(
                SpanData::Custom {
                    name: "t".into(),
                    data: Value::Null,
                },
                None,
                true,
            ),
            include_data: false,
        }
    }

    #[tokio::test]
    async fn assigns_repeated_tool_call_id_and_name() {
        // Some providers repeat the id and the name on every chunk; they must be assigned,
        // not appended.
        let chunks = vec![
            Ok(
                json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": "{\"x\""}}]}}]}),
            ),
            Ok(
                json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": ":1}"}}]}}]}),
            ),
            Ok(
                json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "", "function": {"name": "", "arguments": ""}}]}}]}),
            ),
        ];
        let mut st = stream_state("m", chunks);
        while let Some(Ok(c)) = st.inner.next().await {
            st.handle_chunk(&c);
        }
        st.finish();
        let events: Vec<Value> = st.pending.into_iter().map(|e| e.unwrap()).collect();
        let out = &events.last().unwrap()["response"]["output"];
        assert_eq!(out[0]["call_id"], "call_1");
        assert_eq!(out[0]["name"], "f");
        assert_eq!(out[0]["arguments"], "{\"x\":1}");
    }

    #[tokio::test]
    async fn streams_logprobs_and_url_citations() {
        let chunks = vec![
            Ok(
                json!({"choices": [{"delta": {"content": "hi"}, "logprobs": {"content": [
                    {"token": "hi", "logprob": -0.5, "bytes": [104, 105],
                     "top_logprobs": [{"token": "hi", "logprob": -0.5, "bytes": [104, 105]}]}
                ]}}]}),
            ),
            Ok(json!({"choices": [{"delta": {"annotations": [
                {"type": "url_citation", "url_citation": {"start_index": 0, "end_index": 2,
                 "url": "https://example.com", "title": "Example"}}
            ]}}]})),
        ];
        let mut st = stream_state("m", chunks);
        while let Some(Ok(c)) = st.inner.next().await {
            st.handle_chunk(&c);
        }
        st.finish();
        let events: Vec<Value> = st.pending.into_iter().map(|e| e.unwrap()).collect();
        let delta = events
            .iter()
            .find(|e| e["type"] == "response.output_text.delta")
            .unwrap();
        assert_eq!(delta["logprobs"][0]["token"], "hi");
        assert!(delta["logprobs"][0].get("bytes").is_none());
        let part = &events.last().unwrap()["response"]["output"][0]["content"][0];
        assert_eq!(part["annotations"][0]["url"], "https://example.com");
        assert_eq!(part["logprobs"][0]["bytes"][0], 104);
    }

    #[test]
    fn parses_annotations_and_logprobs() {
        let v = json!({"choices": [{"message": {"content": "hi", "annotations": [
            {"type": "url_citation", "url_citation": {"start_index": 0, "end_index": 2,
             "url": "https://example.com", "title": "Example"}},
            {"type": "url_citation", "url_citation": {"url": "https://broken"}}
        ]}, "logprobs": {"content": [{"token": "hi", "logprob": -0.1, "bytes": [104, 105],
            "top_logprobs": []}]}}]});
        let r = parse_chat_completion(&v, None, false).unwrap();
        let item = r.output[0].to_value();
        assert_eq!(
            item["content"][0]["annotations"].as_array().unwrap().len(),
            1
        );
        assert_eq!(item["content"][0]["annotations"][0]["title"], "Example");
        assert_eq!(item["content"][0]["logprobs"][0]["token"], "hi");
    }

    #[test]
    fn sends_tool_choice_without_tools_and_rejects_duplicate_extra_args() {
        use crate::model_settings::ModelSettings;
        let model = OpenAIChatCompletionsModel::new("gpt-4o", OpenAIClient::new("k"));
        let req = ModelRequest {
            model_settings: ModelSettings::default().with_tool_choice("none"),
            ..Default::default()
        };
        let body = model.build_request_body(&req, false).unwrap();
        assert_eq!(body["tool_choice"], "none");
        assert!(body.get("tools").is_none());

        let req = ModelRequest {
            model_settings: ModelSettings {
                extra_args: Some(
                    [("model".to_string(), json!("other"))]
                        .into_iter()
                        .collect(),
                ),
                ..Default::default()
            },
            ..Default::default()
        };
        let err = model
            .build_request_body(&req, false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("multiple values for keyword argument 'model'"),
            "{err}"
        );
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
