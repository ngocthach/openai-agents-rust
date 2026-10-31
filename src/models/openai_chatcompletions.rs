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
    warned_unsupported: Arc<[std::sync::atomic::AtomicBool; 3]>,
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
        if let Some(reasoning) = req.model_settings.reasoning.as_ref() {
            let mut names = Vec::new();
            if reasoning.mode.is_some() {
                names.push("reasoning.mode");
            }
            if reasoning.context.is_some() {
                names.push("reasoning.context");
            }
            if !names.is_empty() {
                let message = format!(
                    "OpenAIChatCompletionsModel does not support {}. These reasoning settings \
                     require the Responses API; Chat Completions only uses reasoning.effort.",
                    names.join(", ")
                );
                if self.strict_feature_validation {
                    return Err(AgentsError::user(message));
                }
                if !self.warned_unsupported[2].swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::warn!(
                        target: crate::logger::LOG_TARGET,
                        "{message} Ignoring unsupported reasoning settings; enable strict \
                         feature validation to raise an error instead."
                    );
                }
            }
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
            if matches!(tc, ToolChoice::Named(n) if n == "programmatic_tool_calling") {
                return Err(AgentsError::user(
                    "tool_choice='programmatic_tool_calling' is only supported with OpenAI \
                     Responses models and cannot be used with Chat Completions-compatible models.",
                ));
            }
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
        let mut reasoning_effort = s.reasoning.as_ref().and_then(|r| r.effort.clone());
        // These models require non-reasoning mode for Chat Completions function tools.
        // Caller-provided effort overrides stay unchanged.
        if reasoning_effort.is_none()
            && matches!(self.model.as_str(), "gpt-6-sol" | "gpt-6-luna")
            && has_tools
            && !s
                .extra_args
                .as_ref()
                .is_some_and(|e| e.contains_key("reasoning_effort"))
        {
            reasoning_effort = Some("none".to_string());
        }
        put("reasoning_effort", reasoning_effort.map(Value::String));
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
            // An explicit setting is sent as given; only the official API defaults to true.
            let include_usage = s
                .include_usage
                .or_else(|| self.client.is_openai_api().then_some(true));
            if let Some(include_usage) = include_usage {
                body.insert(
                    "stream_options".into(),
                    json!({"include_usage": include_usage}),
                );
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
pub fn convert_chat_tool_choice(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Named(n) if matches!(n.as_str(), "auto" | "required" | "none") => {
            Value::String(n.clone())
        }
        ToolChoice::Named(n) => json!({"type": "function", "function": {"name": n}}),
        ToolChoice::Mcp(m) => json!({"type": "function", "function": {"name": m.name}}),
    }
}

/// Copy a `prompt_cache_breakpoint` from the source content part onto the converted part.
fn copy_prompt_cache_breakpoint(source: &Value, target: &mut Map<String, Value>) {
    if let Some(bp) = source
        .get("prompt_cache_breakpoint")
        .filter(|v| !v.is_null())
    {
        target.insert("prompt_cache_breakpoint".into(), bp.clone());
    }
}

/// Accept raw Chat Completions content parts by mapping them to the SDK canonical shapes.
/// Port of `Converter._normalize_input_content_part_alias`.
fn normalize_input_content_part_alias(part: &Value) -> Result<Value> {
    let Some(obj) = part.as_object() else {
        return Ok(part.clone());
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("text") => {
            let Some(text) = obj.get("text").and_then(Value::as_str) else {
                return Err(AgentsError::user(format!(
                    "Only text content is supported here, got: {part}"
                )));
            };
            let mut m = Map::new();
            m.insert("type".into(), json!("input_text"));
            m.insert("text".into(), json!(text));
            copy_prompt_cache_breakpoint(part, &mut m);
            Ok(Value::Object(m))
        }
        Some("image_url") => {
            let Some(payload) = obj.get("image_url").and_then(Value::as_object) else {
                return Err(AgentsError::user(format!(
                    "Only image URLs are supported for image_url {part}"
                )));
            };
            let Some(url) = payload
                .get("url")
                .and_then(Value::as_str)
                .filter(|u| !u.is_empty())
            else {
                return Err(AgentsError::user(format!(
                    "Only image URLs are supported for image_url {part}"
                )));
            };
            let mut m = Map::new();
            m.insert("type".into(), json!("input_image"));
            m.insert("image_url".into(), json!(url));
            if let Some(d) = payload.get("detail").filter(|v| !v.is_null()) {
                m.insert("detail".into(), d.clone());
            }
            copy_prompt_cache_breakpoint(part, &mut m);
            Ok(Value::Object(m))
        }
        _ => Ok(part.clone()),
    }
}

/// Convert Responses input content into Chat Completions content parts.
/// Port of `Converter.extract_all_content`.
pub fn extract_all_content(content: &Value) -> Result<Value> {
    if let Value::String(s) = content {
        return Ok(Value::String(s.clone()));
    }
    let Some(parts) = content.as_array() else {
        return Err(AgentsError::user(format!("Unknown content: {content}")));
    };
    let mut out: Vec<Value> = Vec::new();
    for raw in parts {
        let c = normalize_input_content_part_alias(raw)?;
        let ty = c.get("type").and_then(Value::as_str).unwrap_or_default();
        let mut m = Map::new();
        match ty {
            "input_text" => {
                m.insert("type".into(), json!("text"));
                m.insert(
                    "text".into(),
                    c.get("text").cloned().unwrap_or_else(|| json!("")),
                );
            }
            "input_image" => {
                let Some(url) = c
                    .get("image_url")
                    .and_then(Value::as_str)
                    .filter(|u| !u.is_empty())
                else {
                    return Err(AgentsError::user(format!(
                        "Only image URLs are supported for input_image {c}"
                    )));
                };
                let detail = c
                    .get("detail")
                    .and_then(Value::as_str)
                    .filter(|d| !d.is_empty())
                    .unwrap_or("auto");
                // Chat Completions only supports auto/low/high, so preserve the caller's
                // highest-fidelity intent with the closest available value.
                let detail = if detail == "original" { "high" } else { detail };
                m.insert("type".into(), json!("image_url"));
                m.insert("image_url".into(), json!({"url": url, "detail": detail}));
            }
            "video_url" => {
                let url = c
                    .get("video_url")
                    .and_then(Value::as_object)
                    .and_then(|p| p.get("url"))
                    .and_then(Value::as_str)
                    .filter(|u| !u.is_empty());
                let Some(url) = url else {
                    return Err(AgentsError::user(format!(
                        "Only video URLs are supported for video_url {c}"
                    )));
                };
                m.insert("type".into(), json!("video_url"));
                m.insert("video_url".into(), json!({"url": url}));
            }
            "input_audio" => {
                let payload = c.get("input_audio").filter(|v| !v.is_null());
                let Some(payload) = payload else {
                    return Err(AgentsError::user(format!(
                        "Only audio data is supported for input_audio {c}"
                    )));
                };
                let Some(payload) = payload.as_object() else {
                    return Err(AgentsError::user(format!(
                        "input_audio must provide audio data and format {c}"
                    )));
                };
                let data = payload
                    .get("data")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty());
                let format = payload
                    .get("format")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty());
                let (Some(data), Some(format)) = (data, format) else {
                    return Err(AgentsError::user(format!(
                        "input_audio requires both data and format {c}"
                    )));
                };
                m.insert("type".into(), json!("input_audio"));
                m.insert(
                    "input_audio".into(),
                    json!({"data": data, "format": format}),
                );
            }
            "input_file" => {
                // The Chat Completions file part accepts inline file_data or a reference to an
                // uploaded file_id. Inline data wins when both are present.
                let file_data = c
                    .get("file_data")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty());
                let file_id = c
                    .get("file_id")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty());
                let mut file = Map::new();
                if let Some(d) = file_data {
                    file.insert("file_data".into(), json!(d));
                } else if let Some(i) = file_id {
                    file.insert("file_id".into(), json!(i));
                } else {
                    return Err(AgentsError::user(format!(
                        "Only file_data or file_id is supported for input_file {c}"
                    )));
                }
                if let Some(name) = c
                    .get("filename")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                {
                    file.insert("filename".into(), json!(name));
                }
                m.insert("type".into(), json!("file"));
                m.insert("file".into(), Value::Object(file));
            }
            _ => {
                return Err(AgentsError::user(format!("Unknown content: {c}")));
            }
        }
        copy_prompt_cache_breakpoint(&c, &mut m);
        out.push(Value::Object(m));
    }
    Ok(Value::Array(out))
}

/// [`extract_all_content`] reduced to the textual parts.
/// Port of `Converter.extract_text_content`.
pub fn extract_text_content(content: &Value) -> Result<Value> {
    let all = extract_all_content(content)?;
    if all.is_string() {
        return Ok(all);
    }
    let mut out: Vec<Value> = Vec::new();
    for c in all.as_array().into_iter().flatten() {
        match c.get("type").and_then(Value::as_str) {
            Some("text") => out.push(c.clone()),
            Some("video_url") => {
                return Err(AgentsError::user(format!(
                    "Only text content is supported here, got: {c}"
                )));
            }
            _ => {}
        }
    }
    Ok(Value::Array(out))
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

/// True when the item is a Responses assistant output message.
/// Port of `Converter.maybe_response_output_message`.
fn is_output_message(v: &Value) -> bool {
    if v.get("type").and_then(Value::as_str) != Some("message")
        || v.get("role").and_then(Value::as_str) != Some("assistant")
    {
        return false;
    }
    let Some(content) = v.get("content") else {
        return false;
    };
    if v.get("id").is_some() {
        return true;
    }
    content.as_array().is_some_and(|parts| {
        !parts.is_empty()
            && parts.iter().all(|p| {
                p.is_object()
                    && matches!(
                        p.get("type").and_then(Value::as_str),
                        Some("output_text" | "refusal")
                    )
            })
    })
}

/// The role of an easy-input message, if the item is one.
/// Port of `Converter.maybe_easy_input_message`.
fn easy_input_role(v: &Value) -> Option<&str> {
    let obj = v.as_object()?;
    if !obj.contains_key("content") || !obj.contains_key("role") {
        return None;
    }
    if !obj
        .keys()
        .all(|k| matches!(k.as_str(), "content" | "role" | "type" | "phase"))
    {
        return None;
    }
    if obj.contains_key("type") && obj.get("type").and_then(Value::as_str) != Some("message") {
        return None;
    }
    match obj.get("phase") {
        None | Some(Value::Null) => {}
        Some(Value::String(p)) if p == "commentary" || p == "final_answer" => {}
        _ => return None,
    }
    let role = obj.get("role").and_then(Value::as_str)?;
    if !matches!(role, "user" | "assistant" | "system" | "developer") {
        return None;
    }
    if is_output_message(v) {
        return None;
    }
    Some(role)
}

/// The role of a typed input message, if the item is one.
/// Port of `Converter.maybe_input_message`.
fn input_message_role(v: &Value) -> Option<&str> {
    if v.get("type").and_then(Value::as_str) != Some("message") {
        return None;
    }
    v.get("role")
        .and_then(Value::as_str)
        .filter(|r| matches!(*r, "user" | "system" | "developer"))
}

/// Convert Responses-format input items to Chat Completions messages.
pub fn items_to_messages(items: &[InputItem], system: Option<&str>) -> Result<Vec<Value>> {
    items_to_messages_with(items, system, &ChatConversionOptions::default())
}

/// Pending reasoning state carried from a reasoning item to the assistant message that follows.
#[derive(Default)]
struct PendingReasoning {
    /// Replayed as `reasoning_content` (DeepSeek-style).
    content: Option<String>,
    /// Replayed as `reasoning` (third-party plaintext reasoning).
    text: Option<String>,
    /// Anthropic thinking blocks: (blocks, came from a native `thinking_blocks` field).
    thinking: Option<(Vec<Value>, bool)>,
}

impl PendingReasoning {
    fn clear(&mut self) {
        self.content = None;
        self.text = None;
        self.thinking = None;
    }

    /// Port of `apply_pending_thinking_blocks` + `apply_pending_reasoning_content`.
    fn apply(&mut self, asst: &mut Map<String, Value>) {
        apply_pending_thinking_blocks(asst, &mut self.thinking);
        if let Some(text) = self.content.take() {
            asst.insert("reasoning_content".into(), Value::String(text));
        }
        if let Some(text) = self.text.take() {
            asst.insert("reasoning".into(), Value::String(text));
        }
    }
}

/// [`items_to_messages`] with reasoning replay options.
pub fn items_to_messages_with(
    items: &[InputItem],
    system: Option<&str>,
    options: &ChatConversionOptions,
) -> Result<Vec<Value>> {
    let mut result: Vec<Value> = Vec::new();
    let mut pending = PendingReasoning::default();
    if let Some(s) = system {
        result.push(json!({"role": "system", "content": s}));
    }
    let mut current: Option<Map<String, Value>> = None;
    fn flush(
        result: &mut Vec<Value>,
        current: &mut Option<Map<String, Value>>,
        pending: &mut PendingReasoning,
        clear_pending: bool,
    ) {
        if let Some(mut a) = current.take() {
            if a.get("tool_calls")
                .and_then(Value::as_array)
                .map(Vec::is_empty)
                .unwrap_or(false)
            {
                a.remove("tool_calls");
                // Prevents stale reasoning from contaminating later turns.
                pending.content = None;
                pending.text = None;
            }
            result.push(Value::Object(a));
        }
        if clear_pending {
            pending.clear();
        }
    }
    for item in items {
        let v = &item.0;
        if let Some(role) = easy_input_role(v) {
            flush(&mut result, &mut current, &mut pending, true);
            let content = v.get("content").cloned().unwrap_or(Value::Null);
            let converted = if role == "user" {
                extract_all_content(&content)?
            } else {
                extract_text_content(&content)?
            };
            result.push(json!({"role": role, "content": converted}));
            continue;
        }
        if let Some(role) = input_message_role(v) {
            flush(&mut result, &mut current, &mut pending, true);
            let content = v.get("content").cloned().unwrap_or(Value::Null);
            let converted = if role == "user" {
                extract_all_content(&content)?
            } else {
                extract_text_content(&content)?
            };
            result.push(json!({"role": role, "content": converted}));
            continue;
        }
        if is_output_message(v) {
            let mut text_segments: Vec<String> = Vec::new();
            let mut refusal: Option<String> = None;
            for c in v
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                match c.get("type").and_then(Value::as_str) {
                    Some("output_text") => text_segments.push(
                        c.get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    ),
                    Some("refusal") => {
                        refusal = Some(
                            c.get("refusal")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        )
                    }
                    Some("output_audio") => {
                        return Err(AgentsError::user(format!(
                            "Only audio IDs are supported for chat completions, but got: {c}"
                        )));
                    }
                    _ => {
                        return Err(AgentsError::user(format!(
                            "Unknown content type in ResponseOutputMessage: {c}"
                        )));
                    }
                }
            }
            // A streamed turn can order its function calls before its text message, and the
            // pending assistant message then already carries this turn's tool calls, because a
            // function_call_output always flushes. Merge into it: flushing instead would emit an
            // assistant message with tool_calls directly followed by another assistant message,
            // a sequence the Chat Completions API rejects.
            let pending_content = current.as_ref().and_then(|a| a.get("content")).cloned();
            let mergeable = current.as_ref().is_some_and(|a| {
                a.get("tool_calls")
                    .and_then(Value::as_array)
                    .is_some_and(|c| !c.is_empty())
                    && !a.contains_key("refusal")
                    && matches!(
                        pending_content,
                        None | Some(Value::Null) | Some(Value::Array(_))
                    )
            });
            if mergeable {
                let merged = current.as_mut().expect("pending assistant message");
                if !text_segments.is_empty() {
                    let combined = text_segments.join("\n");
                    match &pending_content {
                        Some(Value::Array(existing)) => {
                            let mut parts = existing.clone();
                            parts.push(json!({"type": "text", "text": combined}));
                            merged.insert("content".into(), Value::Array(parts));
                        }
                        _ => {
                            merged.insert("content".into(), Value::String(combined));
                        }
                    }
                }
                if let Some(r) = refusal {
                    merged.insert("refusal".into(), Value::String(r));
                }
                pending.apply(merged);
            } else {
                // A reasoning item can be followed by an assistant message and then tool calls
                // in the same turn, so preserve pending reasoning across this flush.
                flush(&mut result, &mut current, &mut pending, false);
                let mut asst = Map::new();
                asst.insert("role".into(), Value::String("assistant".into()));
                if let Some(r) = refusal {
                    asst.insert("refusal".into(), Value::String(r));
                }
                if !text_segments.is_empty() {
                    asst.insert("content".into(), Value::String(text_segments.join("\n")));
                }
                apply_pending_thinking_blocks(&mut asst, &mut pending.thinking);
                asst.insert("tool_calls".into(), Value::Array(Vec::new()));
                if let Some(text) = pending.content.take() {
                    asst.insert("reasoning_content".into(), Value::String(text));
                }
                if let Some(text) = pending.text.take() {
                    asst.insert("reasoning".into(), Value::String(text));
                }
                current = Some(asst);
            }
            continue;
        }
        let ty = v.get("type").and_then(Value::as_str);
        match ty {
            Some("file_search_call") => {
                let asst = current.get_or_insert_with(new_assistant_message);
                pending.apply(asst);
                let call = json!({
                    "id": v.get("id").cloned().unwrap_or(Value::Null),
                    "type": "function",
                    "function": {
                        "name": "file_search_call",
                        "arguments": serde_json::to_string(&json!({
                            "queries": v.get("queries").cloned().unwrap_or_else(|| json!([])),
                            "status": v.get("status").cloned().unwrap_or(Value::Null),
                        })).unwrap_or_else(|_| "{}".into()),
                    }
                });
                if let Some(Value::Array(calls)) = asst.get_mut("tool_calls") {
                    calls.push(call);
                }
            }
            Some("function_call") => {
                let asst = current.get_or_insert_with(new_assistant_message);
                pending.apply(asst);
                let arguments = v
                    .get("arguments")
                    .and_then(Value::as_str)
                    .filter(|a| !a.is_empty())
                    .unwrap_or("{}");
                let mut call = json!({
                    "id": v.get("call_id").cloned().unwrap_or(Value::Null),
                    "type": "function",
                    "function": {
                        "name": v.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": arguments,
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
            Some("function_call_output") => {
                let Some(call_id) = v.get("call_id").filter(|c| !c.is_null()).cloned() else {
                    return Err(AgentsError::user(
                        "Unpaired function outputs are supported by Responses but cannot be \
                         converted to Chat Completions tool messages. Use a Responses model to \
                         preserve this input.",
                    ));
                };
                flush(&mut result, &mut current, &mut pending, true);
                let output = v.get("output").cloned().unwrap_or(Value::Null);
                let all = match &output {
                    Value::String(_) | Value::Array(_) => extract_all_content(&output)?,
                    other => Value::String(other.to_string()),
                };
                let content = if options.preserve_tool_output_all_content || all.is_string() {
                    all
                } else {
                    let text_only: Vec<Value> = all
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                        .cloned()
                        .collect();
                    if text_only.is_empty() {
                        let message = "Chat Completions tool outputs cannot be empty or contain \
                             only non-text content unless preserve_tool_output_all_content is set.";
                        if options.strict_feature_validation {
                            return Err(AgentsError::user(message));
                        }
                        log::warn!(
                            target: crate::logger::LOG_TARGET,
                            "{message} Replacing the tool output with a placeholder; enable \
                             strict feature validation to raise an error instead."
                        );
                        Value::String(OMITTED_TOOL_OUTPUT_PLACEHOLDER.to_string())
                    } else {
                        Value::Array(text_only)
                    }
                };
                result.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }));
            }
            Some("item_reference") => {
                return Err(AgentsError::user(format!(
                    "Encountered an item_reference, which is not supported: {v}"
                )));
            }
            Some("reasoning") => {
                pending.clear();
                convert_reasoning_item(v, options, &mut pending);
            }
            Some("compaction") => {
                return Err(AgentsError::user(
                    "Compaction items are not supported for chat completions. Please use the \
                     Responses API to handle compaction.",
                ));
            }
            // An item of a type Chat Completions has no equivalent for (a hosted tool call,
            // for example) is skipped; anything message-shaped that was not recognized, or
            // that carries no type at all, is an error.
            Some(other) if other != "message" => {
                log::debug!(
                    target: crate::logger::LOG_TARGET,
                    "Skipping item type {other} not supported by Chat Completions"
                );
            }
            _ => {
                return Err(AgentsError::user(format!(
                    "Unhandled item type or structure: {v}"
                )));
            }
        }
    }
    flush(&mut result, &mut current, &mut pending, true);
    Ok(result)
}

/// A fresh assistant message that tool calls attach to.
fn new_assistant_message() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("role".into(), Value::String("assistant".into()));
    m.insert("content".into(), Value::Null);
    m.insert("tool_calls".into(), Value::Array(Vec::new()));
    m
}

/// Decide what a stored reasoning item replays into the next assistant message.
fn convert_reasoning_item(
    v: &Value,
    options: &ChatConversionOptions,
    pending: &mut PendingReasoning,
) {
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
    let Some(model) = &options.model else {
        return;
    };
    if provider_data
        .get(REASONING_FIELD_KEY)
        .and_then(Value::as_str)
        == Some("reasoning")
        && *model == item_model
    {
        let t = texts("content", "reasoning_text");
        if !t.is_empty() {
            pending.text = Some(t.join("\n"));
        }
    }
    // Claude interleaved thinking: replay the signed blocks on the next assistant message.
    // Items may come from another provider, so only an item whose model matches (or that
    // carries no other provider metadata) counts as Claude's own.
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
            pending.text = None;
            pending.thinking = Some((blocks.clone(), true));
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
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("reasoning_text"))
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
                pending.thinking = Some((blocks, false));
            }
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
            pending.content = Some(t.join("\n"));
        }
    }
}

/// Carry a provider's per-tool-call metadata (today: Google's thought signature) on the
/// function call item, so it can be replayed on the next request.
/// Port of the `extra_content` -> `provider_data` conversion in `message_to_output_items`.
fn tool_call_provider_data(call: &Value, base: &Map<String, Value>) -> Map<String, Value> {
    let mut provider_data = base.clone();
    if let Some(sig) = call
        .pointer("/extra_content/google/thought_signature")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        provider_data.insert("thought_signature".into(), json!(sig));
    }
    let mut extra = Map::new();
    if !provider_data.is_empty() {
        extra.insert("provider_data".into(), Value::Object(provider_data));
    }
    extra
}

/// Normalize a Chat Completions `usage` payload. A null count is zero, and the request counts
/// even when the provider omits usage entirely.
fn chat_usage(usage: Option<&Value>) -> Usage {
    let Some(u) = usage.filter(|u| !u.is_null()) else {
        return Usage {
            requests: 1,
            ..Default::default()
        };
    };
    let get = |p: &str| u.pointer(p).and_then(Value::as_u64).unwrap_or(0);
    Usage {
        requests: 1,
        input_tokens: get("/prompt_tokens"),
        output_tokens: get("/completion_tokens"),
        total_tokens: get("/total_tokens"),
        input_tokens_details: crate::usage::InputTokensDetails {
            cached_tokens: get("/prompt_tokens_details/cached_tokens"),
            cache_write_tokens: get("/prompt_tokens_details/cache_write_tokens"),
        },
        output_tokens_details: crate::usage::OutputTokensDetails {
            reasoning_tokens: get("/completion_tokens_details/reasoning_tokens"),
        },
        request_usage_entries: Vec::new(),
    }
}

/// Parse a Chat Completions response into a `ModelResponse`.
pub fn parse_chat_completion(
    v: &Value,
    request_id: Option<String>,
    preserve_raw_usage: bool,
) -> Result<ModelResponse> {
    parse_chat_completion_with(v, request_id, preserve_raw_usage, false)
}

/// [`parse_chat_completion`] with strict feature validation.
pub fn parse_chat_completion_with(
    v: &Value,
    request_id: Option<String>,
    preserve_raw_usage: bool,
    strict_feature_validation: bool,
) -> Result<ModelResponse> {
    let model = v
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    parse_chat_completion_as(
        v,
        request_id,
        preserve_raw_usage,
        strict_feature_validation,
        &model,
    )
}

/// [`parse_chat_completion_with`] recording `provider_model` (the model the request was made
/// to) as the origin of every output item, as `get_response` does.
pub fn parse_chat_completion_as(
    v: &Value,
    request_id: Option<String>,
    preserve_raw_usage: bool,
    strict_feature_validation: bool,
    provider_model: &str,
) -> Result<ModelResponse> {
    if v.get("choices")
        .and_then(Value::as_array)
        .map(Vec::is_empty)
        .unwrap_or(true)
    {
        let details = match v.get("error").filter(|e| !e.is_null()) {
            Some(e) => format!(": {e}"),
            None => String::new(),
        };
        return Err(AgentsError::model_behavior(format!(
            "ChatCompletion response has no choices (possible provider error payload){details}"
        )));
    }
    let choice = v.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let mut message = choice.get("message").cloned().unwrap_or(Value::Null);
    if message.get("audio").map(|a| !a.is_null()).unwrap_or(false) {
        return Err(AgentsError::other("Audio is not currently supported"));
    }
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let is_empty_turn = |m: &Value| {
        m.get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .is_empty()
            && m.get("refusal")
                .and_then(Value::as_str)
                .unwrap_or("")
                .is_empty()
            && m.get("tool_calls")
                .and_then(Value::as_array)
                .map(Vec::is_empty)
                .unwrap_or(true)
    };
    // Some providers signal a filtered completion only through finish_reason="content_filter"
    // and an otherwise empty message. Preserve that terminal signal as a refusal.
    if finish_reason == Some("content_filter")
        && !message.is_null()
        && is_empty_turn(&message)
        && let Some(obj) = message.as_object_mut()
    {
        obj.insert(
            "refusal".into(),
            json!("Response withheld by the provider's content filter."),
        );
    }
    // A completion truncated before any visible token is budget exhaustion, not a refusal.
    if finish_reason == Some("length") && !message.is_null() && is_empty_turn(&message) {
        return Err(AgentsError::model_behavior(
            "Chat Completions response terminated with finish_reason='length' but produced no \
             assistant text, tool call, or refusal.",
        ));
    }
    let message = message;
    let mut output = Vec::new();
    // Metadata naming the source of every output item, for replay to another provider.
    let mut base_provider_data = Map::new();
    if !provider_model.is_empty() {
        base_provider_data.insert("model".into(), json!(provider_model));
    }
    if !message.is_null()
        && let Some(id) = v.get("id").and_then(Value::as_str)
    {
        base_provider_data.insert("response_id".into(), json!(id));
    }
    let reasoning_content = message
        .get("reasoning_content")
        .and_then(Value::as_str)
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
    // Prefer the structured/provider-native representations when a provider sends more than
    // one reasoning field on the same message.
    let reasoning = message
        .get("reasoning")
        .and_then(Value::as_str)
        .filter(|_| reasoning_content.is_empty() && thinking_blocks.is_none())
        .unwrap_or("");
    if !reasoning_content.is_empty() || !reasoning.is_empty() || thinking_blocks.is_some() {
        let mut provider_data = base_provider_data.clone();
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
            let mut signatures: Vec<String> = Vec::new();
            for block in &blocks {
                if block.get("type").and_then(Value::as_str) != Some("thinking") {
                    continue;
                }
                text.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
                if let Some(sig) = block
                    .get("signature")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    signatures.push(sig.to_string());
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
            // Every signature is kept, newline-joined; replay splits them back apart.
            if !signatures.is_empty() {
                item["encrypted_content"] = json!(signatures.join("\n"));
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
            extra: {
                let mut extra = Map::new();
                if !base_provider_data.is_empty() {
                    extra.insert(
                        "provider_data".into(),
                        Value::Object(base_provider_data.clone()),
                    );
                }
                extra
            },
        }));
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            if c.get("type").and_then(Value::as_str).unwrap_or("function") != "function" {
                if strict_feature_validation {
                    return Err(AgentsError::user(
                        "Custom tool calls are not supported by the Chat Completions converter",
                    ));
                }
                continue;
            }
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
                extra: tool_call_provider_data(c, &base_provider_data),
            }));
        }
    }
    let usage_v = v.get("usage").filter(|u| !u.is_null());
    let usage = chat_usage(usage_v);
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

/// Refusal text synthesized when a provider reports `content_filter` with no output.
const CONTENT_FILTER_REFUSAL: &str = "Response withheld by the provider's content filter.";

/// Python truthiness of a JSON value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// The Python type name of a JSON value, for error messages.
fn python_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Clean LiteLLM's `__thought__` suffix from Gemini tool call ids.
/// Port of `ChatCmplHelpers.clean_gemini_tool_call_id`.
fn clean_gemini_tool_call_id(id: &str, model: &str) -> String {
    if !model.is_empty() && model.to_lowercase().contains("gemini") && id.contains("__thought__") {
        return id.split("__thought__").next().unwrap_or(id).to_string();
    }
    id.to_string()
}

/// Whether a streamed delta carries any output the handler must see.
/// Port of `_delta_has_passthrough_output`.
fn delta_has_passthrough_output(delta: Option<&Value>) -> bool {
    let Some(d) = delta.filter(|d| !d.is_null()) else {
        return false;
    };
    if d.get("content").is_some_and(|c| !c.is_null()) || truthy(d.get("tool_calls")) {
        return true;
    }
    [
        "refusal",
        "reasoning_content",
        "reasoning",
        "thinking_blocks",
        "annotations",
        "audio",
    ]
    .iter()
    .any(|k| truthy(d.get(*k)))
}

/// Merge provider metadata without letting empty chunks erase earlier fields.
/// Port of `_merge_buffered_metadata`.
pub fn merge_buffered_metadata(
    current: Option<Map<String, Value>>,
    incoming: &Map<String, Value>,
) -> Option<Map<String, Value>> {
    if incoming.is_empty() {
        return current;
    }
    let Some(current) = current else {
        return Some(incoming.clone());
    };
    let mut merged = current;
    for (key, value) in incoming {
        match (
            merged.get(key).and_then(Value::as_object),
            value.as_object(),
        ) {
            (Some(existing), Some(inc)) => {
                let nested =
                    merge_buffered_metadata(Some(existing.clone()), inc).unwrap_or_default();
                merged.insert(key.clone(), Value::Object(nested));
            }
            (_, Some(inc)) if inc.is_empty() && merged.contains_key(key) => {}
            _ => {
                merged.insert(key.clone(), value.clone());
            }
        }
    }
    Some(merged)
}

/// A streamed function tool call being accumulated for buffering.
#[derive(Default)]
struct BufferedToolCall {
    index: u64,
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
    provider_specific_fields: Option<Map<String, Value>>,
    extra_content: Option<Map<String, Value>>,
}

/// Holds streamed function tool-call deltas back and re-emits each call once, complete.
/// Port of `ChatCmplStreamHandler.buffer_tool_call_stream`.
#[derive(Default)]
struct ToolCallBuffer {
    calls: BTreeMap<u64, BufferedToolCall>,
    passthrough_indexes: std::collections::HashSet<u64>,
    saw_passthrough: bool,
    last_chunk: Option<Value>,
}

impl ToolCallBuffer {
    fn accumulate(&mut self, delta: &Value) {
        let index = delta.get("index").and_then(Value::as_u64).unwrap_or(0);
        let call = self.calls.entry(index).or_insert_with(|| BufferedToolCall {
            index,
            ..Default::default()
        });
        if let Some(id) = delta
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            call.call_id = Some(id.to_string());
        }
        if let Some(f) = delta.get("function") {
            if let Some(n) = f
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                call.name = Some(n.to_string());
            }
            if let Some(a) = f.get("arguments").and_then(Value::as_str) {
                call.arguments.push_str(a);
            }
        }
        if let Some(m) = delta
            .get("provider_specific_fields")
            .and_then(Value::as_object)
        {
            call.provider_specific_fields =
                merge_buffered_metadata(call.provider_specific_fields.take(), m);
        }
        if let Some(m) = delta.get("extra_content").and_then(Value::as_object) {
            call.extra_content = merge_buffered_metadata(call.extra_content.take(), m);
        }
    }

    /// Process one provider chunk; returns the chunk to forward, if any.
    fn feed(&mut self, chunk: Value) -> Result<Option<Value>> {
        self.last_chunk = Some(chunk.clone());
        let choices = chunk
            .get("choices")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if choices.is_empty() {
            return Ok(Some(chunk));
        }
        let mut passthrough: Vec<Value> = Vec::new();
        for mut choice in choices {
            let index = choice.get("index").and_then(Value::as_i64).unwrap_or(0);
            if index != 0 {
                if truthy(choice.pointer("/delta/tool_calls")) {
                    self.saw_passthrough = true;
                }
                passthrough.push(choice);
                continue;
            }
            if let Some(deltas) = choice
                .pointer("/delta/tool_calls")
                .and_then(Value::as_array)
                .filter(|d| !d.is_empty())
                .cloned()
            {
                let mut remaining: Vec<Value> = Vec::new();
                for d in deltas {
                    let idx = d.get("index").and_then(Value::as_u64).unwrap_or(0);
                    let ty = d.get("type").and_then(Value::as_str);
                    if self.passthrough_indexes.contains(&idx) {
                        self.saw_passthrough = true;
                        remaining.push(d);
                    } else if matches!(ty, None | Some("function")) {
                        self.accumulate(&d);
                    } else {
                        self.passthrough_indexes.insert(idx);
                        self.saw_passthrough = true;
                        remaining.push(d);
                    }
                }
                choice["delta"]["tool_calls"] = if remaining.is_empty() {
                    Value::Null
                } else {
                    Value::Array(remaining)
                };
            }
            let has_passthrough = delta_has_passthrough_output(choice.get("delta"));
            let finish = choice.get("finish_reason").and_then(Value::as_str);
            if finish == Some("tool_calls")
                && self.calls.is_empty()
                && !self.saw_passthrough
                && !has_passthrough
            {
                return Err(AgentsError::model_behavior(
                    "Chat Completions stream finished with finish_reason='tool_calls' but did \
                     not include any streamed tool call deltas.",
                ));
            }
            if has_passthrough {
                passthrough.push(choice);
            } else if finish.is_some() {
                // Preserve real terminal evidence even when all output was buffered.
                choice["delta"] = json!({});
                passthrough.push(choice);
            }
        }
        if !passthrough.is_empty() || chunk.get("usage").is_some_and(|u| !u.is_null()) {
            let mut out = chunk;
            out["choices"] = Value::Array(passthrough);
            return Ok(Some(out));
        }
        Ok(None)
    }

    /// The chunk carrying every buffered tool call, once the stream has ended.
    fn flush(&mut self) -> Result<Option<Value>> {
        if self.calls.is_empty() {
            return Ok(None);
        }
        let Some(template) = self.last_chunk.clone() else {
            return Ok(None);
        };
        let mut deltas = Vec::new();
        for call in self.calls.values() {
            let Some(id) = call.call_id.as_deref().filter(|s| !s.is_empty()) else {
                return Err(AgentsError::model_behavior(
                    "Buffered Chat Completions tool call stream ended without a tool call id.",
                ));
            };
            let Some(name) = call.name.as_deref().filter(|s| !s.is_empty()) else {
                return Err(AgentsError::model_behavior(
                    "Buffered Chat Completions tool call stream ended without a function name.",
                ));
            };
            let mut d = json!({"index": call.index, "id": id, "type": "function",
                "function": {"name": name, "arguments": call.arguments}});
            if let Some(m) = &call.provider_specific_fields {
                d["provider_specific_fields"] = Value::Object(m.clone());
            }
            if let Some(m) = &call.extra_content {
                d["extra_content"] = Value::Object(m.clone());
            }
            deltas.push(d);
        }
        let mut out = template;
        // Assembling buffered deltas does not prove the provider finished the choice.
        out["choices"] =
            json!([{"index": 0, "delta": {"tool_calls": deltas}, "finish_reason": null}]);
        out["usage"] = Value::Null;
        Ok(Some(out))
    }
}

struct TextPart {
    index: usize,
    text: String,
    annotations: Vec<Value>,
    logprobs: Vec<Value>,
}

impl TextPart {
    fn to_value(&self) -> Value {
        let mut part = json!({"text": self.text, "type": "output_text",
            "annotations": self.annotations});
        if !self.logprobs.is_empty() {
            part["logprobs"] = Value::Array(self.logprobs.clone());
        }
        part
    }
}

struct StreamCall {
    index: u64,
    call_id: String,
    name: String,
    arguments: String,
    provider_data: Option<Map<String, Value>>,
    streaming: bool,
}

/// Behavior switches for [`ChatStreamHandler`].
struct StreamOptions {
    model: String,
    strict_feature_validation: bool,
    raise_on_length_truncation: bool,
    require_finish_reason: bool,
    /// The skeleton `response` object announced by `response.created`.
    base_response: Value,
}

/// Converts Chat Completions chunks into Responses API stream events.
/// Port of `ChatCmplStreamHandler.handle_stream`.
struct ChatStreamHandler {
    opts: StreamOptions,
    started: bool,
    sequence: u64,
    text: Option<TextPart>,
    refusal: Option<(usize, String)>,
    reasoning: Option<Value>,
    active_summary: Option<usize>,
    reasoning_done: bool,
    calls: Vec<StreamCall>,
    ignored_calls: std::collections::HashSet<u64>,
    thinking_blocks: Vec<Value>,
    provider_data: Map<String, Value>,
    warned_choice: bool,
    usage: Option<Value>,
    message_index: Option<usize>,
    call_indexes: BTreeMap<u64, usize>,
    saw_content_filter: bool,
    saw_length: bool,
    saw_finish_reason: bool,
}

impl ChatStreamHandler {
    fn new(opts: StreamOptions) -> Self {
        ChatStreamHandler {
            opts,
            started: false,
            sequence: 0,
            text: None,
            refusal: None,
            reasoning: None,
            active_summary: None,
            reasoning_done: false,
            calls: Vec::new(),
            ignored_calls: Default::default(),
            thinking_blocks: Vec::new(),
            provider_data: Map::new(),
            warned_choice: false,
            usage: None,
            message_index: None,
            call_indexes: BTreeMap::new(),
            saw_content_filter: false,
            saw_length: false,
            saw_finish_reason: false,
        }
    }

    fn emit(&mut self, out: &mut Vec<Value>, mut event: Value) {
        event["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        out.push(event);
    }

    fn reasoning_count(&self) -> usize {
        usize::from(self.reasoning.is_some())
    }

    /// Output index of the assistant message, fixed the first time it is needed.
    fn message_output_index(&mut self) -> usize {
        if let Some(i) = self.message_index {
            return i;
        }
        let mut i = self.reasoning_count();
        if !self.call_indexes.is_empty() {
            i += self.calls.len();
        }
        self.message_index = Some(i);
        i
    }

    /// Output index of a function call, fixed the first time it is needed.
    fn call_output_index(&mut self, call_index: u64) -> usize {
        if let Some(i) = self.call_indexes.get(&call_index) {
            return *i;
        }
        let offset = self
            .calls
            .iter()
            .position(|c| c.index == call_index)
            .expect("function call index has been tracked");
        let rc = self.reasoning_count();
        let index = match self.message_index {
            None => rc + offset,
            Some(m) => {
                if offset < m.saturating_sub(rc) {
                    rc + offset
                } else {
                    rc + offset + 1
                }
            }
        };
        self.call_indexes.insert(call_index, index);
        index
    }

    fn calls_before_message(&self) -> usize {
        match self.message_index {
            None => 0,
            Some(m) => m
                .saturating_sub(self.reasoning_count())
                .min(self.calls.len()),
        }
    }

    fn function_call_item(&self, call: &StreamCall, arguments: &str) -> Value {
        let mut item = json!({"id": FAKE_RESPONSES_ID, "call_id": call.call_id,
            "arguments": arguments, "name": call.name, "type": "function_call"});
        if !self.provider_data.is_empty() || call.provider_data.is_some() {
            let mut merged = self.provider_data.clone();
            for (k, v) in call.provider_data.iter().flatten() {
                merged.insert(k.clone(), v.clone());
            }
            if !merged.is_empty() {
                item["provider_data"] = Value::Object(merged);
            }
        }
        item
    }

    fn open_reasoning(&mut self, out: &mut Vec<Value>, plaintext: bool) {
        let mut item = json!({"id": FAKE_RESPONSES_ID, "summary": [], "type": "reasoning"});
        if plaintext {
            item["content"] = json!([{"text": "", "type": "reasoning_text"}]);
            let mut pd = self.provider_data.clone();
            pd.insert(REASONING_FIELD_KEY.into(), json!("reasoning"));
            item["provider_data"] = Value::Object(pd);
        } else if !self.provider_data.is_empty() {
            item["provider_data"] = Value::Object(self.provider_data.clone());
        }
        self.reasoning = Some(item.clone());
        self.emit(
            out,
            json!({"type": "response.output_item.added", "item": item, "output_index": 0}),
        );
    }

    /// Drop the plaintext-reasoning replay marker once structured reasoning arrives.
    fn discard_plaintext_marker(&mut self) {
        if let Some(pd) = self
            .reasoning
            .as_mut()
            .and_then(|r| r.get_mut("provider_data"))
            .and_then(Value::as_object_mut)
            && pd.get(REASONING_FIELD_KEY).and_then(Value::as_str) == Some("reasoning")
        {
            pd.remove(REASONING_FIELD_KEY);
        }
    }

    fn finish_summary_part(&mut self, out: &mut Vec<Value>) {
        let (Some(item), Some(index)) = (&self.reasoning, self.active_summary) else {
            return;
        };
        let part = item
            .get("summary")
            .and_then(Value::as_array)
            .and_then(|s| s.get(index))
            .cloned();
        self.active_summary = None;
        let Some(part) = part else { return };
        self.emit(
            out,
            json!({"type": "response.reasoning_summary_part.done", "item_id": FAKE_RESPONSES_ID,
                   "output_index": 0, "summary_index": index,
                   "part": {"text": part["text"], "type": "summary_text"}}),
        );
    }

    /// Fold one streamed thinking delta into the ordered block sequence.
    /// Port of `StreamingState.accumulate_thinking_block`.
    fn accumulate_thinking_block(&mut self, block: &Value) -> Result<bool> {
        let Some(obj) = block.as_object() else {
            return Ok(false);
        };
        let block_type = match obj.get("type") {
            None => Some("thinking"),
            Some(t) => t.as_str(),
        };
        if block_type == Some("redacted_thinking") {
            self.thinking_blocks.push(block.clone());
            return Ok(true);
        }
        if block_type != Some("thinking") {
            return Ok(false);
        }
        for field in ["thinking", "signature"] {
            if let Some(v) = obj.get(field)
                && !v.is_string()
            {
                return Err(AgentsError::model_behavior(format!(
                    "Expected streamed thinking block field '{field}' to be a string, got {}.",
                    python_type_name(v)
                )));
            }
        }
        let thinking = obj.get("thinking").and_then(Value::as_str).unwrap_or("");
        let signature = obj.get("signature").and_then(Value::as_str).unwrap_or("");
        let metadata: Vec<(&String, &Value)> = obj
            .iter()
            .filter(|(k, _)| !matches!(k.as_str(), "type" | "thinking" | "signature"))
            .collect();
        if thinking.is_empty() && signature.is_empty() && metadata.is_empty() {
            return Ok(false);
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
                .unwrap_or("");
            let joined = format!("{existing}{thinking}");
            current.insert("thinking".into(), json!(joined));
        }
        if !signature.is_empty() {
            current.insert("signature".into(), json!(signature));
        }
        Ok(true)
    }

    fn start_message(&mut self, out: &mut Vec<Value>) {
        let mut item = json!({"id": FAKE_RESPONSES_ID, "content": [], "role": "assistant",
            "type": "message", "status": "in_progress"});
        if !self.provider_data.is_empty() {
            item["provider_data"] = Value::Object(self.provider_data.clone());
        }
        let output_index = self.message_output_index();
        self.emit(
            out,
            json!({"type": "response.output_item.added", "item": item,
                   "output_index": output_index}),
        );
    }

    fn process_chunk(&mut self, chunk: &Value, out: &mut Vec<Value>) -> Result<()> {
        if !self.started {
            self.started = true;
            let created = json!({"type": "response.created",
                                 "response": self.opts.base_response.clone()});
            self.emit(out, created);
        }
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(u.clone());
        }
        let Some(choices) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .filter(|c| !c.is_empty())
        else {
            return Ok(());
        };
        let index_of = |c: &Value| c.get("index").and_then(Value::as_i64).unwrap_or(0);
        if choices.len() > 1 || choices.iter().any(|c| index_of(c) != 0) {
            let message = "Chat Completions streaming with multiple choices or nonzero choice \
                           indexes is not fully supported; only choice index 0 can be processed.";
            if self.opts.strict_feature_validation {
                return Err(AgentsError::user(message));
            }
            if !self.warned_choice {
                log::warn!(
                    target: crate::logger::LOG_TARGET,
                    "{message} Ignoring the other choices; enable strict feature validation to \
                     raise an error instead."
                );
                self.warned_choice = true;
            }
        }
        let Some(choice) = choices.iter().find(|c| index_of(c) == 0) else {
            return Ok(());
        };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.saw_finish_reason = true;
            if reason == "content_filter" {
                self.saw_content_filter = true;
            } else if reason == "length" {
                self.saw_length = true;
            }
        }
        let Some(delta) = choice.get("delta").filter(|d| !d.is_null()) else {
            return Ok(());
        };
        // Provider data for non-OpenAI Responses endpoints.
        if !self.opts.model.is_empty() {
            self.provider_data
                .insert("model".into(), json!(self.opts.model));
        } else if let Some(m) = chunk
            .get("model")
            .and_then(Value::as_str)
            .filter(|m| !m.is_empty())
        {
            self.provider_data.insert("model".into(), json!(m));
        }
        if let Some(id) = chunk
            .get("id")
            .and_then(Value::as_str)
            .filter(|i| !i.is_empty())
        {
            self.provider_data.insert("response_id".into(), json!(id));
        }
        if truthy(delta.get("audio")) {
            // The sync converter rejects audio output; a silent empty stream would diverge.
            return Err(AgentsError::other("Audio is not currently supported"));
        }

        // Anthropic thinking blocks (preserving signatures).
        if let Some(blocks) = delta.get("thinking_blocks").and_then(Value::as_array)
            && !blocks.is_empty()
        {
            let mut has_block = false;
            for block in blocks.iter().filter(|b| b.is_object()) {
                has_block |= self.accumulate_thinking_block(block)?;
            }
            if has_block && self.reasoning.is_none() {
                self.open_reasoning(out, false);
            }
            if has_block {
                self.discard_plaintext_marker();
            }
        }

        // Reasoning summaries.
        let reasoning_content = delta
            .get("reasoning_content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !reasoning_content.is_empty() {
            if self.reasoning.is_none() {
                self.open_reasoning(out, false);
            }
            self.discard_plaintext_marker();
            if self.active_summary.is_none() {
                let summary = self
                    .reasoning
                    .as_mut()
                    .and_then(|r| r.get_mut("summary"))
                    .and_then(Value::as_array_mut)
                    .expect("reasoning summary");
                let index = summary.len();
                summary.push(json!({"text": "", "type": "summary_text"}));
                self.active_summary = Some(index);
                self.emit(
                    out,
                    json!({"type": "response.reasoning_summary_part.added",
                           "item_id": FAKE_RESPONSES_ID, "output_index": 0,
                           "summary_index": index, "part": {"text": "", "type": "summary_text"}}),
                );
            }
            let index = self.active_summary.expect("active summary");
            self.emit(
                out,
                json!({"type": "response.reasoning_summary_text.delta",
                       "delta": reasoning_content, "item_id": FAKE_RESPONSES_ID,
                       "output_index": 0, "summary_index": index}),
            );
            let slot = self
                .reasoning
                .as_mut()
                .and_then(|r| r.pointer_mut(&format!("/summary/{index}/text")))
                .expect("summary text");
            let joined = format!("{}{reasoning_content}", slot.as_str().unwrap_or(""));
            *slot = json!(joined);
        }

        // Plaintext reasoning from third-party platforms.
        if let Some(Value::String(reasoning_text)) = delta.get("reasoning")
            && !reasoning_text.is_empty()
        {
            if self.reasoning.is_none() {
                self.open_reasoning(out, true);
            }
            self.emit(
                out,
                json!({"type": "response.reasoning_text.delta", "delta": reasoning_text,
                       "item_id": FAKE_RESPONSES_ID, "output_index": 0, "content_index": 0}),
            );
            let item = self.reasoning.as_mut().expect("reasoning item");
            let content_empty = item
                .get("content")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty);
            if content_empty {
                item["content"] = json!([{"text": "", "type": "reasoning_text"}]);
            }
            let slot = item.pointer_mut("/content/0/text").expect("reasoning text");
            let joined = format!("{}{reasoning_text}", slot.as_str().unwrap_or(""));
            *slot = json!(joined);
        }

        if self.reasoning.is_some()
            && self.active_summary.is_some()
            && !truthy(delta.get("reasoning_content"))
            && (delta.get("content").is_some_and(|c| !c.is_null())
                || truthy(delta.get("refusal"))
                || truthy(delta.get("tool_calls")))
        {
            self.finish_summary_part(out);
        }

        // Regular content.
        if let Some(content) = delta.get("content").and_then(Value::as_str)
            && !(self.text.is_none() && content.is_empty())
        {
            if self.text.is_none() {
                let content_index = usize::from(self.refusal.is_some());
                self.text = Some(TextPart {
                    index: content_index,
                    text: String::new(),
                    annotations: Vec::new(),
                    logprobs: Vec::new(),
                });
                // A refusal part already opened this assistant message, so only the new content
                // part is announced.
                if content_index == 0 {
                    self.start_message(out);
                }
                let output_index = self.message_output_index();
                self.emit(
                    out,
                    json!({"type": "response.content_part.added", "content_index": content_index,
                           "item_id": FAKE_RESPONSES_ID, "output_index": output_index,
                           "part": {"text": "", "type": "output_text", "annotations": [],
                                    "logprobs": []}}),
                );
            }
            let choice_logprobs = choice.get("logprobs");
            let delta_logprobs = convert_logprobs_for_text_delta(choice_logprobs);
            let output_logprobs = convert_logprobs_for_output_text(choice_logprobs);
            let output_index = self.message_output_index();
            let content_index = self.text.as_ref().expect("text part").index;
            self.emit(
                out,
                json!({"type": "response.output_text.delta", "content_index": content_index,
                       "delta": content, "item_id": FAKE_RESPONSES_ID,
                       "output_index": output_index, "logprobs": delta_logprobs}),
            );
            let part = self.text.as_mut().expect("text part");
            part.text.push_str(content);
            if let Some(Value::Array(lp)) = output_logprobs {
                part.logprobs.extend(lp);
            }
        }

        // Citations arrive on the delta carrying the cited text or on a later one.
        if let Some(part) = self.text.as_mut() {
            part.annotations
                .extend(convert_url_citations(delta.get("annotations")));
        }

        // Refusals.
        if let Some(refusal) = delta
            .get("refusal")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
        {
            if self.refusal.is_none() {
                let refusal_index = usize::from(self.text.is_some());
                self.refusal = Some((refusal_index, String::new()));
                if refusal_index == 0 {
                    self.start_message(out);
                }
                let output_index = self.message_output_index();
                self.emit(
                    out,
                    json!({"type": "response.content_part.added",
                           "content_index": refusal_index, "item_id": FAKE_RESPONSES_ID,
                           "output_index": output_index,
                           "part": {"refusal": "", "type": "refusal"}}),
                );
            }
            let output_index = self.message_output_index();
            let content_index = self.refusal.as_ref().expect("refusal part").0;
            self.emit(
                out,
                json!({"type": "response.refusal.delta", "content_index": content_index,
                       "delta": refusal, "item_id": FAKE_RESPONSES_ID,
                       "output_index": output_index}),
            );
            self.refusal
                .as_mut()
                .expect("refusal part")
                .1
                .push_str(refusal);
        }

        // Tool calls, streamed in real time.
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                self.process_tool_call_delta(tc, out)?;
            }
        }
        Ok(())
    }

    fn process_tool_call_delta(&mut self, tc: &Value, out: &mut Vec<Value>) -> Result<()> {
        let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
        if self.ignored_calls.contains(&index) {
            return Ok(());
        }
        if tc.get("type").and_then(Value::as_str) == Some("custom") {
            if self.opts.strict_feature_validation {
                return Err(AgentsError::user(
                    "Custom tool calls are not supported by the Chat Completions converter",
                ));
            }
            self.ignored_calls.insert(index);
            return Ok(());
        }
        let model = self.opts.model.clone();
        let pos = match self.calls.iter().position(|c| c.index == index) {
            Some(p) => p,
            None => {
                self.calls.push(StreamCall {
                    index,
                    call_id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                    provider_data: None,
                    streaming: false,
                });
                self.calls.len() - 1
            }
        };
        let function = tc.get("function");
        let arguments = function
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let state_provider_data = self.provider_data.clone();
        let call = &mut self.calls[pos];
        call.arguments.push_str(arguments);
        if let Some(name) = function
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
        {
            call.name = name.to_string();
        }
        if let Some(id) = tc
            .get("id")
            .and_then(Value::as_str)
            .filter(|i| !i.is_empty())
        {
            // Clean up LiteLLM's `__thought__` suffix on Gemini tool call ids.
            call.call_id = clean_gemini_tool_call_id(id, &model);
        }
        if call.provider_data.is_none() && !state_provider_data.is_empty() {
            call.provider_data = Some(state_provider_data.clone());
        }
        // Gemini thought signatures from LiteLLM's `provider_specific_fields`.
        if model.to_lowercase().contains("gemini")
            && let Some(sig) = tc
                .pointer("/provider_specific_fields/thought_signature")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        {
            let mut pd = state_provider_data.clone();
            pd.insert("thought_signature".into(), json!(sig));
            call.provider_data = Some(pd);
        }
        // Google's chat completions endpoint reports it under `extra_content`.
        if let Some(sig) = tc
            .pointer("/extra_content/google/thought_signature")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let mut pd = state_provider_data;
            pd.insert("thought_signature".into(), json!(sig));
            call.provider_data = Some(pd);
        }
        // Start streaming as soon as the name and call id are known.
        if !self.calls[pos].streaming
            && !self.calls[pos].name.is_empty()
            && !self.calls[pos].call_id.is_empty()
        {
            let output_index = self.call_output_index(index);
            self.calls[pos].streaming = true;
            let item = self.function_call_item(&self.calls[pos], "");
            self.emit(
                out,
                json!({"type": "response.output_item.added", "item": item,
                       "output_index": output_index}),
            );
        }
        if self.calls[pos].streaming && !arguments.is_empty() {
            let output_index = self.call_output_index(index);
            self.emit(
                out,
                json!({"type": "response.function_call_arguments.delta", "delta": arguments,
                       "item_id": FAKE_RESPONSES_ID, "output_index": output_index}),
            );
        }
        Ok(())
    }

    /// Usage in the Responses shape, or `null` when the provider reported none.
    fn response_usage(&self) -> Value {
        let Some(u) = &self.usage else {
            return Value::Null;
        };
        let usage = chat_usage(Some(u));
        json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "input_tokens_details": {
                "cached_tokens": usage.input_tokens_details.cached_tokens,
                "cache_write_tokens": usage.input_tokens_details.cache_write_tokens,
            },
            "output_tokens_details": {
                "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
            },
        })
    }

    fn finalize_thinking_blocks(&mut self) {
        if self.thinking_blocks.is_empty() {
            return;
        }
        let blocks = self.thinking_blocks.clone();
        let Some(item) = self.reasoning.as_mut() else {
            return;
        };
        let mut pd = item
            .get("provider_data")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        pd.insert("thinking_blocks".into(), Value::Array(blocks.clone()));
        item["provider_data"] = Value::Object(pd);
        // Retain the normalized projection for callers and legacy replay.
        let mut text = String::new();
        let mut signatures: Vec<String> = Vec::new();
        for block in &blocks {
            if block.get("type").and_then(Value::as_str) != Some("thinking") {
                continue;
            }
            text.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
            if let Some(sig) = block
                .get("signature")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                signatures.push(sig.to_string());
            }
        }
        if !text.is_empty() {
            if !item.get("content").is_some_and(|c| c.is_array()) {
                item["content"] = json!([]);
            }
            if let Some(c) = item.get_mut("content").and_then(Value::as_array_mut) {
                c.push(json!({"text": text, "type": "reasoning_text"}));
            }
        }
        if !signatures.is_empty() {
            item["encrypted_content"] = json!(signatures.join("\n"));
        }
    }

    /// Close the stream: terminal events, then `response.completed`.
    fn finish(&mut self, out: &mut Vec<Value>) -> Result<()> {
        // A content-filtered turn with no emitted output becomes an explicit refusal.
        if self.saw_content_filter
            && self.text.is_none()
            && self.refusal.is_none()
            && self.calls.is_empty()
        {
            self.refusal = Some((0, CONTENT_FILTER_REFUSAL.to_string()));
            self.start_message(out);
            let output_index = self.message_output_index();
            self.emit(
                out,
                json!({"type": "response.content_part.added", "content_index": 0,
                       "item_id": FAKE_RESPONSES_ID, "output_index": output_index,
                       "part": {"refusal": "", "type": "refusal"}}),
            );
            self.emit(
                out,
                json!({"type": "response.refusal.delta", "content_index": 0,
                       "delta": CONTENT_FILTER_REFUSAL, "item_id": FAKE_RESPONSES_ID,
                       "output_index": output_index}),
            );
        }
        if self.opts.require_finish_reason && !self.saw_finish_reason {
            return Err(AgentsError::model_behavior(
                "Chat Completions stream ended before receiving a finish_reason for choice 0.",
            ));
        }
        // A completion truncated before any visible token is budget exhaustion, not a refusal.
        if self.saw_length
            && self.opts.raise_on_length_truncation
            && self.text.is_none()
            && self.refusal.is_none()
            && self.calls.is_empty()
        {
            return Err(AgentsError::model_behavior(
                "Chat Completions stream terminated with finish_reason='length' but produced no \
                 assistant text, tool call, or refusal.",
            ));
        }

        self.finalize_thinking_blocks();
        if self.reasoning.is_some() && !self.reasoning_done {
            let has_summary = self
                .reasoning
                .as_ref()
                .and_then(|r| r.get("summary"))
                .and_then(Value::as_array)
                .is_some_and(|s| !s.is_empty());
            if has_summary {
                self.finish_summary_part(out);
            } else if let Some(text) = self
                .reasoning
                .as_ref()
                .and_then(|r| r.pointer("/content/0/text"))
                .cloned()
            {
                self.emit(
                    out,
                    json!({"type": "response.reasoning_text.done", "item_id": FAKE_RESPONSES_ID,
                           "output_index": 0, "content_index": 0, "text": text}),
                );
            }
            let item = self.reasoning.clone().expect("reasoning item");
            self.emit(
                out,
                json!({"type": "response.output_item.done", "item": item, "output_index": 0}),
            );
            self.reasoning_done = true;
        }

        let has_message = self.text.is_some() || self.refusal.is_some();
        if has_message {
            let output_index = self.message_output_index();
            if let Some(part) = &self.text {
                let event = json!({"type": "response.content_part.done",
                    "content_index": part.index, "item_id": FAKE_RESPONSES_ID,
                    "output_index": output_index, "part": part.to_value()});
                self.emit(out, event);
            }
            if let Some((index, refusal)) = self.refusal.clone() {
                self.emit(
                    out,
                    json!({"type": "response.content_part.done", "content_index": index,
                           "item_id": FAKE_RESPONSES_ID, "output_index": output_index,
                           "part": {"refusal": refusal, "type": "refusal"}}),
                );
            }
        }

        // Function call completions.
        for i in 0..self.calls.len() {
            let index = self.calls[i].index;
            let output_index = self.call_output_index(index);
            let arguments = self.calls[i].arguments.clone();
            let item = self.function_call_item(&self.calls[i], &arguments);
            if !self.calls[i].streaming {
                // The call never announced itself (no id or name), so send it all at once.
                self.emit(
                    out,
                    json!({"type": "response.output_item.added", "item": item.clone(),
                           "output_index": output_index}),
                );
                self.emit(
                    out,
                    json!({"type": "response.function_call_arguments.delta", "delta": arguments,
                           "item_id": FAKE_RESPONSES_ID, "output_index": output_index}),
                );
            }
            self.emit(
                out,
                json!({"type": "response.output_item.done", "item": item,
                       "output_index": output_index}),
            );
        }

        let mut outputs: Vec<Value> = Vec::new();
        if let Some(r) = &self.reasoning {
            outputs.push(r.clone());
        }
        let before = self.calls_before_message();
        let call_items: Vec<Value> = self
            .calls
            .iter()
            .map(|c| self.function_call_item(c, &c.arguments))
            .collect();
        outputs.extend(call_items[..before].iter().cloned());
        if has_message {
            let mut content: Vec<(usize, Value)> = Vec::new();
            if let Some(part) = &self.text {
                content.push((part.index, part.to_value()));
            }
            if let Some((index, refusal)) = &self.refusal {
                content.push((*index, json!({"refusal": refusal, "type": "refusal"})));
            }
            // Assemble the parts in the order of the content indexes already announced.
            content.sort_by_key(|(i, _)| *i);
            let mut message = json!({"id": FAKE_RESPONSES_ID,
                "content": content.into_iter().map(|(_, p)| p).collect::<Vec<_>>(),
                "role": "assistant", "type": "message", "status": "completed"});
            if !self.provider_data.is_empty() {
                message["provider_data"] = Value::Object(self.provider_data.clone());
            }
            let output_index = self.message_output_index();
            self.emit(
                out,
                json!({"type": "response.output_item.done", "item": message.clone(),
                       "output_index": output_index}),
            );
            outputs.push(message);
        }
        outputs.extend(call_items[before..].iter().cloned());

        let mut response = self.opts.base_response.clone();
        response["output"] = Value::Array(outputs);
        response["status"] = json!("completed");
        response["usage"] = self.response_usage();
        self.emit(
            out,
            json!({"type": "response.completed", "response": response}),
        );
        Ok(())
    }
}

/// Options for [`handle_chat_stream_chunks`].
#[derive(Debug, Clone, Default)]
pub struct ChatStreamOptions {
    /// The model generating the stream; falls back to each chunk's `model` when empty.
    pub model: Option<String>,
    /// Reject what Chat Completions streams cannot faithfully represent.
    pub strict_feature_validation: bool,
    /// Raise when the stream ends with `finish_reason="length"` and no output.
    pub raise_on_length_truncation: bool,
    /// Require a provider finish reason for choice zero (official OpenAI endpoint only).
    pub require_finish_reason: bool,
}

/// Convert Chat Completions stream chunks into Responses stream events, without any transport.
/// Port of `ChatCmplStreamHandler.handle_stream`.
pub fn handle_chat_stream_chunks(
    chunks: &[Value],
    options: &ChatStreamOptions,
) -> Result<Vec<Value>> {
    let mut handler = ChatStreamHandler::new(StreamOptions {
        model: options.model.clone().unwrap_or_default(),
        strict_feature_validation: options.strict_feature_validation,
        raise_on_length_truncation: options.raise_on_length_truncation,
        require_finish_reason: options.require_finish_reason,
        base_response: json!({"id": FAKE_RESPONSES_ID, "object": "response",
            "status": "in_progress", "model": "fake-model", "output": [], "tools": [],
            "tool_choice": "none", "parallel_tool_calls": false}),
    });
    let mut events = Vec::new();
    for chunk in chunks {
        handler.process_chunk(chunk, &mut events)?;
    }
    handler.finish(&mut events)?;
    Ok(events)
}

/// Hold streamed function tool-call deltas back and re-emit each call once, complete.
/// Port of `ChatCmplStreamHandler.buffer_tool_call_stream`.
pub fn buffer_tool_call_chunks(chunks: Vec<Value>) -> Result<Vec<Value>> {
    let mut buffer = ToolCallBuffer::default();
    let mut out = Vec::new();
    for chunk in chunks {
        if let Some(c) = buffer.feed(chunk)? {
            out.push(c);
        }
    }
    if let Some(c) = buffer.flush()? {
        out.push(c);
    }
    Ok(out)
}

/// Stream plumbing: the provider chunks, the optional tool-call buffer, and the handler.
struct ChatStream {
    inner: BoxStream<'static, Result<Value>>,
    buffer: Option<ToolCallBuffer>,
    handler: ChatStreamHandler,
    pending: VecDeque<Result<Value>>,
    finished: bool,
    request_id: Option<String>,
    span: Span,
    include_data: bool,
}

impl ChatStream {
    fn feed_chunk(&mut self, chunk: Value) -> Result<()> {
        let chunk = match self.buffer.as_mut() {
            Some(b) => match b.feed(chunk)? {
                Some(c) => c,
                None => return Ok(()),
            },
            None => chunk,
        };
        self.process(chunk)
    }

    fn process(&mut self, chunk: Value) -> Result<()> {
        let mut events = Vec::new();
        let result = self.handler.process_chunk(&chunk, &mut events);
        self.pending.extend(events.into_iter().map(Ok));
        result
    }

    fn end_of_stream(&mut self) -> Result<()> {
        if let Some(chunk) = self
            .buffer
            .as_mut()
            .map(ToolCallBuffer::flush)
            .transpose()?
            .flatten()
        {
            self.process(chunk)?;
        }
        let mut events = Vec::new();
        let result = self.handler.finish(&mut events);
        if result.is_ok() {
            self.complete_span(&events);
            if let Some(last) = events.last_mut() {
                super::openai_responses::attach_stream_request_id(last, self.request_id.as_deref());
            }
        }
        self.pending.extend(events.into_iter().map(Ok));
        result
    }

    /// Record the output and usage on the generation span before the terminal event is handed
    /// out, because a consumer that stops there never resumes this stream.
    fn complete_span(&self, events: &[Value]) {
        let usage = chat_usage(self.handler.usage.as_ref()).to_span_usage();
        let output = events
            .last()
            .and_then(|e| e.pointer("/response/output"))
            .cloned();
        let include_data = self.include_data;
        self.span.update_data(|d| {
            if let SpanData::Generation {
                output: out,
                usage: u,
                ..
            } = d
            {
                if include_data {
                    *out = output;
                }
                *u = Some(usage);
            }
        });
        self.span.finish();
    }

    fn fail(&mut self, error: AgentsError) -> AgentsError {
        self.finished = true;
        // The request counts, and any reported tokens are kept, even when the turn is rejected.
        let usage = chat_usage(self.handler.usage.as_ref()).to_span_usage();
        self.span.update_data(|d| {
            if let SpanData::Generation { usage: u, .. } = d {
                *u = Some(usage);
            }
        });
        crate::tracing::record_model_error_on_span(
            &self.span,
            "Error streaming response",
            &error,
            self.include_data,
        );
        crate::logger::log_model_action_error("Error streaming response", &error);
        self.span.finish();
        error
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
        let mut guard = crate::tracing::ModelSpanGuard::new(
            &span,
            "Error getting response",
            request.tracing.include_data(),
        );
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
            // The request counts, and any reported tokens are kept on the span, even when the
            // turn is rejected below (for example a truncated empty completion).
            span.update_data(|d| {
                if let SpanData::Generation { usage, .. } = d {
                    *usage = Some(chat_usage(value.get("usage")).to_span_usage());
                }
            });
            parse_chat_completion_as(
                &value,
                request_id,
                request.model_settings.preserve_raw_usage.unwrap_or(false),
                self.strict_feature_validation,
                &self.model,
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
        guard.disarm();
        span.finish();
        result
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let body = self.transformed_body(&request, true)?;
        let span = self.start_span(&request, &body);
        let (inner, request_id) = match self
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
        let settings = &request.model_settings;
        let has_tools = body.get("tools").is_some();
        let mut base_response = json!({
            "id": FAKE_RESPONSES_ID, "object": "response", "status": "in_progress",
            "model": self.model, "output": [], "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": has_tools && settings.parallel_tool_calls.unwrap_or(false),
        });
        if let Some(t) = settings.temperature {
            base_response["temperature"] = json!(t);
        }
        if let Some(t) = settings.top_p {
            base_response["top_p"] = json!(t);
        }
        let handler = ChatStreamHandler::new(StreamOptions {
            model: self.model.clone(),
            strict_feature_validation: self.strict_feature_validation,
            raise_on_length_truncation: true,
            require_finish_reason: self.client.is_openai_api(),
            base_response,
        });
        let state = ChatStream {
            inner,
            buffer: self
                .buffer_streamed_tool_calls
                .then(ToolCallBuffer::default),
            handler,
            pending: VecDeque::new(),
            finished: false,
            request_id,
            span,
            include_data: request.tracing.include_data(),
        };
        let stream = futures::stream::unfold(state, |mut st| async move {
            loop {
                if let Some(ev) = st.pending.pop_front() {
                    return Some((ev, st));
                }
                if st.finished {
                    return None;
                }
                let step = match st.inner.next().await {
                    Some(Ok(chunk)) => st.feed_chunk(chunk),
                    Some(Err(e)) => Err(e),
                    None => {
                        st.finished = true;
                        st.end_of_stream()
                    }
                };
                if let Err(e) = step {
                    let e = st.fail(e);
                    st.pending.push_back(Err(e));
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
