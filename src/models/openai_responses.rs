//! Model backed by the OpenAI Responses API. Mirrors `agents.models.openai_responses`.

use std::collections::BTreeSet;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Map, Value, json};

use super::FAKE_RESPONSES_ID;
use super::interface::{Model, ModelRequest, ResponseEventStream, ToolSpec};
use super::openai_client::OpenAIClient;
use crate::error::{AgentsError, Result};
use crate::items::{InputItem, ModelResponse, OutputItem};
use crate::model_settings::ToolChoice;
use crate::tracing::{Span, SpanData, record_model_error_on_span, response_span};
use crate::usage::Usage;

use super::response_terminal::{is_terminal_failure_event, terminal_failure_error};

/// A model that calls `POST /responses`.
#[derive(Debug, Clone)]
pub struct OpenAIResponsesModel {
    model: String,
    client: OpenAIClient,
    model_is_explicit: bool,
}

impl OpenAIResponsesModel {
    /// Create a Responses model for `model` using `client`.
    pub fn new(model: impl Into<String>, client: OpenAIClient) -> Self {
        OpenAIResponsesModel {
            model: model.into(),
            client,
            model_is_explicit: true,
        }
    }

    pub(crate) fn with_explicit(mut self, explicit: bool) -> Self {
        self.model_is_explicit = explicit;
        self
    }

    /// The model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Build the JSON request body. Exposed for testing and advanced use.
    pub fn build_request_body(&self, req: &ModelRequest, stream: bool) -> Result<Value> {
        let s = &req.model_settings;
        let mut body = Map::new();
        if req.prompt.is_none() || self.model_is_explicit {
            body.insert("model".into(), Value::String(self.model.clone()));
        }
        body.insert(
            "input".into(),
            Value::Array(clean_input_for_openai(&req.input)),
        );

        let request_model = body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string);
        let (mut tools, mut includes) =
            convert_tools(&req.tools, request_model.as_deref(), s.tool_choice.as_ref())?;
        for h in &req.handoffs {
            tools.push(h.to_responses_param());
        }
        if let Some(extra) = &s.response_include {
            includes.extend(extra.iter().cloned());
        }
        if s.top_logprobs.is_some() {
            includes.insert("message.output_text.logprobs".into());
        }
        let has_tools = !tools.is_empty();
        if has_tools {
            body.insert("tools".into(), Value::Array(tools));
        }
        if !includes.is_empty() {
            body.insert(
                "include".into(),
                json!(includes.into_iter().collect::<Vec<_>>()),
            );
        }
        if let Some(tc) = &s.tool_choice {
            let mut converted = convert_tool_choice(tc);
            if converted.get("type").and_then(Value::as_str) == Some("computer_use_preview")
                && !use_preview_computer_tool(request_model.as_deref(), Some(tc))
            {
                converted = json!({"type": "computer"});
            }
            let omit = req.prompt.is_some() && !has_tools && converted.is_object();
            if !omit {
                body.insert("tool_choice".into(), converted);
            }
        }
        if (has_tools || req.prompt.is_some())
            && let Some(p) = s.parallel_tool_calls
        {
            body.insert("parallel_tool_calls".into(), Value::Bool(p));
        }

        let mut text: Option<Map<String, Value>> = None;
        if let Some(schema) = &req.output_schema
            && !schema.is_plain_text()
        {
            let mut t = Map::new();
            t.insert(
                "format".into(),
                json!({
                    "type": "json_schema",
                    "name": "final_output",
                    "schema": schema.json_schema()?,
                    "strict": schema.is_strict_json_schema(),
                }),
            );
            text = Some(t);
        }
        if let Some(v) = &s.verbosity {
            text.get_or_insert_with(Map::new)
                .insert("verbosity".into(), Value::String(v.clone()));
        }
        if let Some(t) = text {
            body.insert("text".into(), Value::Object(t));
        }

        let mut put = |k: &str, v: Option<Value>| {
            if let Some(v) = v {
                body.insert(k.to_string(), v);
            }
        };
        put(
            "instructions",
            req.system_instructions.clone().map(Value::String),
        );
        put(
            "previous_response_id",
            req.previous_response_id.clone().map(Value::String),
        );
        put(
            "conversation",
            req.conversation_id.clone().map(Value::String),
        );
        put("prompt", req.prompt.as_ref().map(|p| json!(p)));
        put("temperature", s.temperature.map(|v| json!(v)));
        put("top_p", s.top_p.map(|v| json!(v)));
        put("truncation", s.truncation.clone().map(Value::String));
        put("max_output_tokens", s.max_tokens.map(|v| json!(v)));
        put("store", s.store.map(Value::Bool));
        put(
            "prompt_cache_retention",
            s.prompt_cache_retention.clone().map(Value::String),
        );
        put("prompt_cache_options", s.prompt_cache_options.clone());
        put("reasoning", s.reasoning.as_ref().map(|r| json!(r)));
        put("metadata", s.metadata.as_ref().map(|m| json!(m)));
        put(
            "context_management",
            s.context_management.as_ref().map(|c| json!(c)),
        );
        put("top_logprobs", s.top_logprobs.map(|v| json!(v)));
        if stream {
            body.insert("stream".into(), Value::Bool(true));
        }

        if let Some(extra) = &s.extra_args {
            let mut dups: Vec<&String> = extra.keys().filter(|k| body.contains_key(*k)).collect();
            dups.sort();
            if !dups.is_empty() {
                return Err(AgentsError::user(format!(
                    "responses.create() got multiple values for keyword argument(s): {}",
                    dups.iter()
                        .map(|d| format!("'{d}'"))
                        .collect::<Vec<_>>()
                        .join(", ")
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
}

fn use_preview_computer_tool(model: Option<&str>, tool_choice: Option<&ToolChoice>) -> bool {
    match model {
        Some(m) => m.starts_with("computer-use-preview"),
        // Prompt-managed calls omit the model: default to the preview payload unless the
        // caller explicitly selected the GA tool.
        None => !matches!(
            tool_choice,
            Some(ToolChoice::Named(n)) if n == "computer" || n == "computer_use"
        ),
    }
}

/// Convert tool specs to Responses tool params: namespaced function tools are grouped into
/// `namespace` tools, the computer tool gets its GA or preview shape, and deferred tools are
/// validated against the presence of a `tool_search` tool.
fn convert_tools(
    specs: &[ToolSpec],
    model: Option<&str>,
    tool_choice: Option<&ToolChoice>,
) -> Result<(Vec<Value>, BTreeSet<String>)> {
    let mut tools: Vec<Value> = Vec::new();
    let mut includes = BTreeSet::new();
    let mut namespaces: Vec<(String, String, usize)> = Vec::new();
    let mut namespace_members: std::collections::HashMap<String, Vec<Value>> =
        std::collections::HashMap::new();
    let mut tool_search_count = 0;
    let mut needs_tool_search = false;
    for t in specs {
        match t {
            ToolSpec::Function(f) => {
                if f.defer_loading {
                    needs_tool_search = true;
                }
                match &f.namespace {
                    Some(ns) => {
                        let desc = f.namespace_description.clone().unwrap_or_default();
                        match namespaces.iter().find(|(n, _, _)| n == ns) {
                            Some((_, d, _)) if *d != desc => {
                                return Err(AgentsError::user(format!(
                                    "All tools in namespace '{ns}' must share the same description."
                                )));
                            }
                            Some(_) => {}
                            None => {
                                namespaces.push((ns.clone(), desc, tools.len()));
                                tools.push(Value::Null);
                            }
                        }
                        namespace_members
                            .entry(ns.clone())
                            .or_default()
                            .push(f.to_responses_param());
                    }
                    None => tools.push(f.to_responses_param()),
                }
            }
            ToolSpec::Hosted {
                param,
                includes: inc,
            } => {
                let ty = param
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if ty == "tool_search" {
                    tool_search_count += 1;
                }
                if param.get("defer_loading").and_then(Value::as_bool) == Some(true) {
                    needs_tool_search = true;
                }
                if ty == "computer" {
                    if use_preview_computer_tool(model, tool_choice) {
                        let (Some(env), Some(w), Some(h)) = (
                            param.get("environment"),
                            param.get("display_width"),
                            param.get("display_height"),
                        ) else {
                            return Err(AgentsError::user(
                                "Preview computer tool payloads require `environment` and \
                                 `dimensions` on the Computer implementation.",
                            ));
                        };
                        tools.push(json!({"type": "computer_use_preview", "environment": env,
                            "display_width": w, "display_height": h}));
                    } else {
                        tools.push(json!({"type": "computer"}));
                    }
                } else {
                    tools.push(param.clone());
                }
                includes.extend(inc.iter().cloned());
            }
        }
    }
    if tool_search_count > 1 {
        return Err(AgentsError::user(
            "Only one ToolSearchTool() is allowed when using OpenAI Responses models.",
        ));
    }
    if needs_tool_search && tool_search_count == 0 {
        return Err(AgentsError::user(
            "Deferred-loading Responses tools require ToolSearchTool() when using OpenAI \
             Responses models.",
        ));
    }
    for (name, description, index) in namespaces {
        tools[index] = json!({
            "type": "namespace",
            "name": name,
            "description": description,
            "tools": namespace_members.remove(&name).unwrap_or_default(),
        });
    }
    Ok((tools, includes))
}

/// Convert a `ToolChoice` to the Responses API format.
pub(crate) fn convert_tool_choice(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Mcp(m) => {
            json!({"type": "mcp", "server_label": m.server_label, "name": m.name})
        }
        ToolChoice::Named(n) => match n.as_str() {
            "auto" | "required" | "none" => Value::String(n.clone()),
            "file_search"
            | "web_search"
            | "web_search_preview"
            | "image_generation"
            | "code_interpreter"
            | "mcp"
            | "tool_search"
            | "programmatic_tool_calling"
            | "computer_use_preview" => json!({"type": n}),
            "computer" | "computer_use" => json!({"type": "computer_use_preview"}),
            other => json!({"type": "function", "name": other}),
        },
    }
}

/// Remove placeholder ids and provider-specific fields before sending to OpenAI.
pub(crate) fn clean_input_for_openai(items: &[InputItem]) -> Vec<Value> {
    items
        .iter()
        .filter_map(|item| {
            let mut v = item.0.clone();
            if let Value::Object(m) = &mut v {
                if m.remove("provider_data").is_some()
                    && m.get("type").and_then(Value::as_str) == Some("reasoning")
                {
                    return None;
                }
                if m.get("id").and_then(Value::as_str) == Some(FAKE_RESPONSES_ID) {
                    m.remove("id");
                }
            }
            Some(v)
        })
        .collect()
}

/// Parse a Responses API `response` object.
pub(crate) fn parse_response(
    v: &Value,
    request_id: Option<String>,
    preserve_raw_usage: bool,
) -> Result<ModelResponse> {
    match v.get("status").and_then(Value::as_str) {
        Some("failed") => {
            let detail = v
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(AgentsError::model_behavior(format!(
                "Responses API returned terminal status failed: {detail}"
            )));
        }
        Some("incomplete") => {
            let reason = v
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            return Err(AgentsError::model_behavior(format!(
                "Responses API returned terminal status incomplete: {reason}"
            )));
        }
        _ => {}
    }
    let output = v
        .get("output")
        .and_then(Value::as_array)
        .map(|a| a.iter().cloned().map(OutputItem::from_value).collect())
        .unwrap_or_default();
    let usage_v = v.get("usage").filter(|u| !u.is_null());
    let usage = usage_v
        .map(Usage::from_responses_usage)
        .unwrap_or_else(|| Usage {
            requests: 1,
            ..Default::default()
        });
    Ok(ModelResponse {
        output,
        usage,
        response_id: v
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| *id != FAKE_RESPONSES_ID)
            .map(str::to_string),
        request_id,
        raw_usage: if preserve_raw_usage {
            usage_v.cloned()
        } else {
            None
        },
    })
}

pub(crate) fn finish_response_span(span: &Span, resp: &ModelResponse, include_data: bool) {
    span.update_data(|d| {
        if let SpanData::Response { response_id, usage } = d {
            *response_id = resp.response_id.clone();
            if include_data {
                *usage = Some(resp.usage.to_span_usage());
            }
        }
    });
}

/// Ensures a span is finished when a stream is dropped early.
pub(crate) struct SpanGuard(pub(crate) Span);

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.0.finish();
    }
}

#[async_trait]
impl Model for OpenAIResponsesModel {
    fn supports_default_prompt_cache_key(&self) -> bool {
        self.client.is_openai_api()
    }

    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = self.build_request_body(&request, false)?;
        let span = response_span(request.tracing.is_disabled());
        span.start();
        let result = async {
            let (value, request_id) = self
                .client
                .post_json(
                    "responses",
                    &body,
                    request.model_settings.extra_headers.as_ref(),
                    request.model_settings.extra_query.as_ref(),
                )
                .await?;
            parse_response(
                &value,
                request_id,
                request.model_settings.preserve_raw_usage.unwrap_or(false),
            )
        }
        .await;
        let include_data = request.tracing.include_data();
        match &result {
            Ok(resp) => finish_response_span(&span, resp, true),
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
        let span = response_span(request.tracing.is_disabled());
        span.start();
        let include_data = request.tracing.include_data();
        let (stream, _request_id) = match self
            .client
            .post_sse(
                "responses",
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
        let guard = SpanGuard(span);
        let mapped = stream.map(move |ev| {
            let span = &guard.0;
            match ev {
                Ok(event) => {
                    let ty = event
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if ty == "response.completed" {
                        if let Some(r) = event.get("response")
                            && let Ok(parsed) = parse_response(r, None, false)
                        {
                            finish_response_span(span, &parsed, true);
                        }
                        return Ok(event);
                    }
                    if is_terminal_failure_event(ty) {
                        let err = terminal_failure_error(ty, &event);
                        record_model_error_on_span(
                            span,
                            "Error streaming response",
                            &err,
                            include_data,
                        );
                        return Err(err);
                    }
                    Ok(event)
                }
                Err(e) => {
                    record_model_error_on_span(span, "Error streaming response", &e, include_data);
                    crate::logger::log_model_action_error("Error streaming response", &e);
                    Err(e)
                }
            }
        });
        Ok(mapped.boxed())
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
    use crate::agent_output::AgentOutputSchema;
    use crate::model_settings::ModelSettings;
    use crate::models::interface::FunctionToolSpec;
    use std::sync::Arc;

    fn spec(name: &str, ns: Option<&str>, defer: bool) -> ToolSpec {
        ToolSpec::Function(FunctionToolSpec {
            name: name.into(),
            description: "d".into(),
            parameters: json!({"type": "object", "properties": {}}),
            strict: true,
            defer_loading: defer,
            namespace: ns.map(str::to_string),
            namespace_description: ns.map(|_| "ns desc".to_string()),
            allowed_callers: None,
            output_json_schema: None,
        })
    }

    #[test]
    fn groups_namespaces_and_validates_tool_search() {
        let model = OpenAIResponsesModel::new("gpt-5.6-luna", OpenAIClient::new("sk-test"));
        let search = ToolSpec::Hosted {
            param: json!({"type": "tool_search"}),
            includes: vec![],
        };
        let req = ModelRequest {
            tools: vec![
                spec("a", Some("crm"), false),
                spec("plain", None, false),
                spec("b", Some("crm"), true),
                search.clone(),
            ],
            ..Default::default()
        };
        let body = model.build_request_body(&req, false).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[0]["type"], "namespace");
        assert_eq!(tools[0]["name"], "crm");
        assert_eq!(tools[0]["description"], "ns desc");
        assert_eq!(tools[0]["tools"][1]["name"], "b");
        assert_eq!(tools[0]["tools"][1]["defer_loading"], true);
        assert_eq!(tools[1]["name"], "plain");

        let missing = ModelRequest {
            tools: vec![spec("x", None, true)],
            ..Default::default()
        };
        assert!(model.build_request_body(&missing, false).is_err());
        let twice = ModelRequest {
            tools: vec![search.clone(), search],
            ..Default::default()
        };
        assert!(model.build_request_body(&twice, false).is_err());
    }

    #[test]
    fn computer_tool_shape_follows_model() {
        let computer = ToolSpec::Hosted {
            param: json!({"type": "computer", "environment": "browser",
                "display_width": 800, "display_height": 600}),
            includes: vec![],
        };
        let req = ModelRequest {
            tools: vec![computer],
            model_settings: ModelSettings {
                tool_choice: Some(ToolChoice::from("computer_use")),
                ..Default::default()
            },
            ..Default::default()
        };
        let ga = OpenAIResponsesModel::new("gpt-5.6-luna", OpenAIClient::new("sk-test"))
            .build_request_body(&req, false)
            .unwrap();
        assert_eq!(ga["tools"][0], json!({"type": "computer"}));
        assert_eq!(ga["tool_choice"], json!({"type": "computer"}));
        let preview =
            OpenAIResponsesModel::new("computer-use-preview", OpenAIClient::new("sk-test"))
                .build_request_body(&req, false)
                .unwrap();
        assert_eq!(preview["tools"][0]["type"], "computer_use_preview");
        assert_eq!(preview["tools"][0]["display_height"], 600);
        assert_eq!(
            preview["tool_choice"],
            json!({"type": "computer_use_preview"})
        );
    }

    #[test]
    fn builds_request_body() {
        let model = OpenAIResponsesModel::new("gpt-5.6-luna", OpenAIClient::new("sk-test"));
        let req = ModelRequest {
            system_instructions: Some("be nice".into()),
            input: vec![InputItem::user("hi"), json!({"type": "function_call_output", "call_id": "c", "output": "x", "id": FAKE_RESPONSES_ID}).into()],
            model_settings: ModelSettings {
                tool_choice: Some(ToolChoice::tool("get_weather")),
                verbosity: Some("low".into()),
                parallel_tool_calls: Some(false),
                ..Default::default()
            },
            tools: vec![ToolSpec::Function(FunctionToolSpec {
                name: "get_weather".into(),
                description: "d".into(),
                parameters: json!({"type": "object", "properties": {}}),
                strict: true,
                ..Default::default()
            })],
            output_schema: Some(Arc::new(AgentOutputSchema::new::<Vec<String>>().unwrap())),
            ..Default::default()
        };
        let body = model.build_request_body(&req, true).unwrap();
        assert_eq!(body["model"], "gpt-5.6-luna");
        assert_eq!(body["instructions"], "be nice");
        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "name": "get_weather"})
        );
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["text"]["format"]["name"], "final_output");
        assert_eq!(body["text"]["verbosity"], "low");
        assert_eq!(body["stream"], true);
        assert!(body["input"][1].get("id").is_none());
    }

    #[test]
    fn rejects_duplicate_extra_args() {
        let model = OpenAIResponsesModel::new("m", OpenAIClient::new("k"));
        let mut extra = Map::new();
        extra.insert("instructions".into(), json!("x"));
        let req = ModelRequest {
            system_instructions: Some("y".into()),
            model_settings: ModelSettings {
                extra_args: Some(extra),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.build_request_body(&req, false).is_err());
    }

    #[test]
    fn parses_responses_and_terminal_errors() {
        let ok = json!({"id": "resp_1", "status": "completed", "output": [
            {"type": "message", "id": "m", "role": "assistant", "content": [{"type": "output_text", "text": "hey", "annotations": []}]}
        ], "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}});
        let r = parse_response(&ok, Some("req_1".into()), true).unwrap();
        assert_eq!(r.response_id.as_deref(), Some("resp_1"));
        assert_eq!(r.usage.total_tokens, 4);
        assert!(r.raw_usage.is_some());
        let failed = json!({"status": "failed", "error": {"message": "nope"}});
        assert!(parse_response(&failed, None, false).is_err());
    }
}
