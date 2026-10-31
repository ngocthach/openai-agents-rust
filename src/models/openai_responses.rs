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

use super::response_terminal::{
    is_terminal_failure_event, response_terminal_failure_error, terminal_failure_error,
};

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

    pub(crate) fn with_explicit(self, explicit: bool) -> Self {
        self.with_explicit_model(explicit)
    }

    /// Whether the caller chose this model name explicitly. Port of `model_is_explicit`: a
    /// prompt-managed request from a non-explicit model omits `model` so the prompt's own
    /// model wins.
    pub fn with_explicit_model(mut self, explicit: bool) -> Self {
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
        let surfaces = ToolSurfaces::collect(&req.tools, &req.handoffs);
        // Python resolves the computer-tool model to `None` when no computer tool is present,
        // so the wire shape is never inferred from a model that cannot use it.
        let computer_model = if surfaces.has_computer_tool {
            request_model.clone()
        } else {
            None
        };
        // Python validates the tool choice before converting the tools, so a bad choice wins.
        let converted_tool_choice = match &s.tool_choice {
            Some(tc) => Some(convert_tool_choice_for_request(
                tc,
                &surfaces,
                computer_model.as_deref(),
            )?),
            None => None,
        };
        let (mut tools, mut includes) = convert_tools(
            &req.tools,
            computer_model.as_deref(),
            s.tool_choice.as_ref(),
            &surfaces,
            req.prompt.is_some(),
        )?;
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
        if let Some(converted) = converted_tool_choice {
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

/// The tool surfaces a tool choice is validated against. Port of the name sets that
/// `Converter._validate_named_function_tool_choice` and `validate_responses_*` build.
#[derive(Debug, Default)]
pub struct ToolSurfaces {
    top_level_function_names: BTreeSet<String>,
    all_local_function_names: BTreeSet<String>,
    deferred_only_function_names: BTreeSet<String>,
    namespaced_function_names: BTreeSet<String>,
    namespace_names: BTreeSet<String>,
    /// Any `ToolSearchTool()` in the tool list.
    has_hosted_tool_search: bool,
    has_computer_tool: bool,
    has_shell_tool: bool,
    has_apply_patch_tool: bool,
    has_programmatic_tool: bool,
    /// A surface that stops being reachable without `ToolSearchTool()`.
    has_required_tool_search_surface: bool,
    /// A surface `ToolSearchTool()` can actually search.
    has_tool_search_surface: bool,
    /// At least one tool advertises `programmatic` in `allowed_callers`.
    has_programmatic_caller: bool,
    /// Tools that allow only programmatic callers, by display name.
    programmatic_only_names: Vec<String>,
    computer_tool_count: usize,
    tool_search_count: usize,
    programmatic_tool_count: usize,
    no_tools: bool,
    no_handoffs: bool,
    /// Every function tool as (name, explicit namespace, defer_loading), in request order.
    functions: Vec<(String, Option<String>, bool)>,
}

/// Record a tool's `allowed_callers` against the programmatic-tool-calling contract.
fn note_allowed_callers(s: &mut ToolSurfaces, callers: Option<&Value>, display_name: &str) {
    let Some(list) = callers.and_then(Value::as_array) else {
        return;
    };
    let has = |c: &str| list.iter().any(|v| v.as_str() == Some(c));
    if !has("programmatic") {
        return;
    }
    s.has_programmatic_caller = true;
    if !has("direct") {
        s.programmatic_only_names.push(display_name.to_string());
    }
}

impl ToolSurfaces {
    /// Collect the surfaces a tool choice is validated against.
    pub fn collect(specs: &[ToolSpec], handoffs: &[super::interface::FunctionToolSpec]) -> Self {
        let mut s = ToolSurfaces {
            no_tools: specs.is_empty(),
            no_handoffs: handoffs.is_empty(),
            ..Default::default()
        };
        for h in handoffs {
            s.top_level_function_names.insert(h.name.clone());
            s.all_local_function_names.insert(h.name.clone());
        }
        for t in specs {
            match t {
                ToolSpec::Function(f) => {
                    s.functions
                        .push((f.name.clone(), f.namespace.clone(), f.defer_loading));
                    s.all_local_function_names.insert(f.name.clone());
                    match &f.namespace {
                        Some(ns) => {
                            s.namespaced_function_names.insert(f.name.clone());
                            s.namespace_names.insert(ns.clone());
                            s.has_tool_search_surface = true;
                        }
                        None if f.defer_loading => {
                            s.deferred_only_function_names.insert(f.name.clone());
                        }
                        None => {
                            s.top_level_function_names.insert(f.name.clone());
                        }
                    }
                    if f.defer_loading {
                        s.has_required_tool_search_surface = true;
                        s.has_tool_search_surface = true;
                    }
                    let qualified = match &f.namespace {
                        Some(ns) => format!("{ns}.{}", f.name),
                        None => f.name.clone(),
                    };
                    let callers = f.allowed_callers.as_ref().map(|c| json!(c));
                    note_allowed_callers(&mut s, callers.as_ref(), &qualified);
                }
                ToolSpec::Hosted { param, .. } => {
                    let ty = param
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let deferred =
                        param.get("defer_loading").and_then(Value::as_bool) == Some(true);
                    match ty {
                        "tool_search" => {
                            s.has_hosted_tool_search = true;
                            s.tool_search_count += 1;
                        }
                        "computer" => {
                            s.has_computer_tool = true;
                            s.computer_tool_count += 1;
                        }
                        "shell" => s.has_shell_tool = true,
                        "apply_patch" => s.has_apply_patch_tool = true,
                        "programmatic_tool_calling" => {
                            s.has_programmatic_tool = true;
                            s.programmatic_tool_count += 1;
                        }
                        "mcp" if deferred => {
                            s.has_required_tool_search_surface = true;
                            s.has_tool_search_surface = true;
                        }
                        _ => {}
                    }
                    if deferred && ty != "mcp" {
                        s.has_required_tool_search_surface = true;
                        s.has_tool_search_surface = true;
                    }
                    let display = if ty == "mcp" {
                        format!(
                            "hosted MCP server `{}`",
                            param
                                .get("server_label")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown")
                        )
                    } else {
                        param
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or(ty)
                            .to_string()
                    };
                    note_allowed_callers(&mut s, param.get("allowed_callers"), &display);
                }
            }
        }
        s
    }
}

/// `required` is only rejected when a deferred surface would leave no callable tool.
/// Port of `Converter._validate_required_tool_choice`.
fn validate_required_tool_choice(surfaces: &ToolSurfaces) -> Result<()> {
    if surfaces.no_tools || surfaces.has_hosted_tool_search {
        return Ok(());
    }
    if surfaces.has_required_tool_search_surface {
        return Err(AgentsError::user(
            "tool_choice='required' is not currently supported when deferred-loading Responses \
             tools are configured without ToolSearchTool() on the OpenAI Responses API. Add \
             ToolSearchTool() or use `auto`.",
        ));
    }
    Ok(())
}

/// Port of `Converter._validate_named_function_tool_choice`.
fn validate_named_function_tool_choice(choice: &str, surfaces: &ToolSurfaces) -> Result<()> {
    if surfaces.no_tools && surfaces.no_handoffs {
        return Ok(());
    }
    if choice == "tool_search" && !surfaces.all_local_function_names.contains("tool_search") {
        return Err(AgentsError::user(if surfaces.has_hosted_tool_search {
            "tool_choice='tool_search' is not supported for ToolSearchTool() on the OpenAI \
             Responses API. Use `auto` or `required`, or target a real top-level function tool \
             named `tool_search`."
        } else {
            "tool_choice='tool_search' requires ToolSearchTool() or a real top-level function \
             tool named `tool_search` on the OpenAI Responses API."
        }));
    }
    let top_level = surfaces.top_level_function_names.contains(choice);
    if !top_level
        && (surfaces.namespaced_function_names.contains(choice)
            || surfaces.namespace_names.contains(choice))
    {
        return Err(AgentsError::user(
            "Named tool_choice must target a callable tool, not a namespace wrapper or bare \
             inner name from tool_namespace(), on the OpenAI Responses API. Use `auto`, \
             `required`, `none`, or target a top-level or qualified namespaced function tool.",
        ));
    }
    if !top_level && surfaces.deferred_only_function_names.contains(choice) {
        return Err(AgentsError::user(
            "Named tool_choice is not currently supported for deferred-loading function tools \
             on the OpenAI Responses API. Use `auto`, `required`, `none`, or load the tool via \
             ToolSearchTool() first.",
        ));
    }
    Ok(())
}

/// Port of `Converter._convert_builtin_computer_tool_choice`.
fn convert_builtin_computer_tool_choice(choice: &str, model: Option<&str>) -> Value {
    let preview = model.is_some_and(|m| m.starts_with("computer-use-preview"))
        || use_preview_computer_tool_name(model, Some(choice));
    if preview {
        json!({"type": "computer_use_preview"})
    } else {
        json!({"type": "computer"})
    }
}

/// Convert a tool choice in the context of the request's tools and handoffs.
/// Port of `Converter.convert_tool_choice`.
pub fn convert_tool_choice_for_request(
    tc: &ToolChoice,
    surfaces: &ToolSurfaces,
    model: Option<&str>,
) -> Result<Value> {
    let name = match tc {
        ToolChoice::Mcp(m) => {
            return Ok(json!({"type": "mcp", "server_label": m.server_label, "name": m.name}));
        }
        ToolChoice::Named(n) => n.as_str(),
    };
    match name {
        "required" => {
            validate_required_tool_choice(surfaces)?;
            Ok(Value::String("required".into()))
        }
        "auto" | "none" => Ok(Value::String(name.to_string())),
        "programmatic_tool_calling"
        | "file_search"
        | "web_search"
        | "web_search_preview"
        | "image_generation"
        | "code_interpreter"
        | "mcp" => Ok(json!({"type": name})),
        "computer" | "computer_use" | "computer_use_preview" if surfaces.has_computer_tool => {
            Ok(convert_builtin_computer_tool_choice(name, model))
        }
        "computer_use_preview" => Ok(json!({"type": "computer_use_preview"})),
        "shell" if surfaces.has_shell_tool => Ok(json!({"type": "shell"})),
        "apply_patch" if surfaces.has_apply_patch_tool => Ok(json!({"type": "apply_patch"})),
        other => {
            validate_named_function_tool_choice(other, surfaces)?;
            Ok(json!({"type": "function", "name": other}))
        }
    }
}

fn use_preview_computer_tool_name(model: Option<&str>, tool_choice: Option<&str>) -> bool {
    match model {
        Some(m) => m.starts_with("computer-use-preview"),
        None => !matches!(tool_choice, Some("computer") | Some("computer_use")),
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
/// Validate a file-search tool's `max_num_results`: zero or an integer from 1 to 50. Zero and
/// `None` both leave the key out so the provider default applies, as in Python.
fn convert_file_search_param(param: &Value) -> Result<Value> {
    let mut param = param.clone();
    let Some(limit) = param.get("max_num_results").filter(|v| !v.is_null()) else {
        if let Some(map) = param.as_object_mut() {
            map.remove("max_num_results");
        }
        return Ok(param);
    };
    match limit.as_u64().filter(|n| *n <= 50) {
        None => {
            return Err(AgentsError::user(
                "FileSearchTool max_num_results must be zero, an integer between 1 and 50, or \
                 None.",
            ));
        }
        Some(0) => {
            if let Some(map) = param.as_object_mut() {
                map.remove("max_num_results");
            }
        }
        Some(_) => {}
    }
    Ok(param)
}

/// Reject function-tool combinations that are ambiguous on the Responses wire.
/// Port of `validate_function_tool_lookup_configuration`.
fn validate_function_tool_lookup_configuration(surfaces: &ToolSurfaces) -> Result<()> {
    let mut qualified_owners: std::collections::HashMap<String, Option<&str>> = Default::default();
    let mut deferred_top_level: BTreeSet<&str> = BTreeSet::new();
    for (name, namespace, defer_loading) in &surfaces.functions {
        // A tool in a namespace of its own name collides with the synthetic namespace
        // tool search reserves for deferred top-level function tools.
        if namespace.as_deref() == Some(name.as_str()) {
            return Err(AgentsError::user(format!(
                "Responses tool-search reserves the synthetic namespace `{name}.{name}` for \
                 deferred top-level function tools. Rename the namespace or tool name to avoid \
                 ambiguous dispatch."
            )));
        }
        if namespace.is_none() && *defer_loading && !deferred_top_level.insert(name) {
            return Err(AgentsError::user(format!(
                "Ambiguous function tool configuration: the deferred top-level tool name \
                 `{name}` is used by multiple tools. Rename one of the deferred-loading \
                 top-level function tools to avoid ambiguous dispatch."
            )));
        }
        let qualified = match namespace {
            Some(ns) => format!("{ns}.{name}"),
            None => name.clone(),
        };
        match qualified_owners.get(&qualified) {
            None => {
                qualified_owners.insert(qualified, namespace.as_deref());
            }
            // Repeated bare names are resolved by the name-collision policy elsewhere.
            Some(None) if namespace.is_none() => {}
            Some(_) => {
                return Err(AgentsError::user(format!(
                    "Ambiguous function tool configuration: the qualified name `{qualified}` is \
                     used by multiple tools. Rename the namespace-wrapped function or dotted \
                     top-level tool to avoid ambiguous dispatch."
                )));
            }
        }
    }
    Ok(())
}

/// Port of `validate_responses_tool_search_configuration`.
fn validate_tool_search_configuration(
    surfaces: &ToolSurfaces,
    allow_opaque_search_surface: bool,
) -> Result<()> {
    if surfaces.tool_search_count > 1 {
        return Err(AgentsError::user(
            "Only one ToolSearchTool() is allowed when using OpenAI Responses models.",
        ));
    }
    validate_function_tool_lookup_configuration(surfaces)?;
    if surfaces.has_required_tool_search_surface && !surfaces.has_hosted_tool_search {
        return Err(AgentsError::user(
            "Deferred-loading Responses tools require ToolSearchTool() when using OpenAI \
             Responses models.",
        ));
    }
    if surfaces.has_hosted_tool_search
        && !surfaces.has_tool_search_surface
        && !allow_opaque_search_surface
    {
        return Err(AgentsError::user(
            "ToolSearchTool() requires at least one searchable Responses surface: a \
             tool_namespace(...) function tool, a deferred-loading function tool \
             (`function_tool(..., defer_loading=True)`), or a deferred-loading hosted MCP \
             server (`HostedMCPTool(tool_config={..., 'defer_loading': True})`).",
        ));
    }
    Ok(())
}

/// Port of `validate_responses_programmatic_tool_calling_configuration`.
fn validate_programmatic_tool_calling_configuration(
    surfaces: &ToolSurfaces,
    tool_choice: Option<&ToolChoice>,
    allow_opaque_tool_search_surface: bool,
) -> Result<()> {
    if surfaces.programmatic_tool_count > 1 {
        return Err(AgentsError::user(
            "Only one ProgrammaticToolCallingTool() is allowed when using OpenAI Responses \
             models.",
        ));
    }
    let chose_programmatic = matches!(
        tool_choice,
        Some(ToolChoice::Named(n)) if n == "programmatic_tool_calling"
    );
    if chose_programmatic && !surfaces.has_programmatic_tool {
        return Err(AgentsError::user(
            "tool_choice='programmatic_tool_calling' requires ProgrammaticToolCallingTool() \
             when using OpenAI Responses models.",
        ));
    }
    if !surfaces.has_programmatic_tool
        && let Some(name) = surfaces.programmatic_only_names.first()
    {
        return Err(AgentsError::user(format!(
            "Tool `{name}` only allows programmatic callers and requires \
             ProgrammaticToolCallingTool() when using OpenAI Responses models."
        )));
    }
    if surfaces.has_programmatic_tool
        && !surfaces.has_programmatic_caller
        && !surfaces.has_hosted_tool_search
        && !allow_opaque_tool_search_surface
    {
        return Err(AgentsError::user(
            "ProgrammaticToolCallingTool() requires at least one tool whose allowed_callers \
             includes 'programmatic', a ToolSearchTool(), or an opaque prompt-managed tool \
             surface.",
        ));
    }
    Ok(())
}

fn convert_tools(
    specs: &[ToolSpec],
    model: Option<&str>,
    tool_choice: Option<&ToolChoice>,
    surfaces: &ToolSurfaces,
    allow_opaque_tool_search_surface: bool,
) -> Result<(Vec<Value>, BTreeSet<String>)> {
    validate_tool_search_configuration(surfaces, allow_opaque_tool_search_surface)?;
    validate_programmatic_tool_calling_configuration(
        surfaces,
        tool_choice,
        allow_opaque_tool_search_surface,
    )?;
    if surfaces.computer_tool_count > 1 {
        return Err(AgentsError::user(format!(
            "You can only provide one computer tool. Got {}",
            surfaces.computer_tool_count
        )));
    }
    let mut tools: Vec<Value> = Vec::new();
    let mut includes = BTreeSet::new();
    let mut namespaces: Vec<(String, String, usize)> = Vec::new();
    let mut namespace_members: std::collections::HashMap<String, Vec<Value>> =
        std::collections::HashMap::new();
    for t in specs {
        match t {
            ToolSpec::Function(f) => match &f.namespace {
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
            },
            ToolSpec::Hosted {
                param,
                includes: inc,
            } => {
                let ty = param
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
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
                } else if ty == "file_search" {
                    tools.push(convert_file_search_param(param)?);
                } else {
                    tools.push(param.clone());
                }
                includes.extend(inc.iter().cloned());
            }
        }
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

/// Remove placeholder ids and provider-specific fields before sending to OpenAI.
pub fn clean_input_for_openai(items: &[InputItem]) -> Vec<Value> {
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
pub fn parse_response(
    v: &Value,
    request_id: Option<String>,
    preserve_raw_usage: bool,
) -> Result<ModelResponse> {
    // Python raises `response_terminal_failure_error("response.{status}", response)` for a
    // non-streamed response whose status is terminal-but-unsuccessful.
    if let Some(status @ ("failed" | "incomplete")) = v.get("status").and_then(Value::as_str) {
        return Err(response_terminal_failure_error(
            &format!("response.{status}"),
            Some(v),
        ));
    }
    let output = v
        .get("output")
        .and_then(Value::as_array)
        .map(|a| a.iter().cloned().map(OutputItem::from_value).collect())
        .unwrap_or_default();
    let usage_v = v.get("usage").filter(|u| !u.is_null());
    // Python reads `total_tokens` as reported (a null count is 0) rather than deriving it, so a
    // provider that omits the total is not credited with input + output.
    let usage = usage_v
        .map(|u| {
            let mut usage = Usage::from_responses_usage(u);
            usage.total_tokens = u.get("total_tokens").and_then(Value::as_u64).unwrap_or(0);
            usage
        })
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
        // On the streaming path the HTTP response is long gone by the time the terminal event
        // arrives, so the id travels on the event itself (Python sets `response._request_id`).
        request_id: request_id.or_else(|| {
            v.get(STREAM_REQUEST_ID_KEY)
                .and_then(Value::as_str)
                .map(str::to_string)
        }),
        raw_usage: if preserve_raw_usage {
            usage_v.cloned()
        } else {
            None
        },
    })
}

/// Key carrying the HTTP request id on a streamed `response` object.
/// The Rust equivalent of Python's private `response._request_id` attribute.
pub const STREAM_REQUEST_ID_KEY: &str = "_request_id";

/// Attach the SSE response's `x-request-id` to a stream event's `response` object, so the
/// terminal event can carry it into [`ModelResponse::request_id`].
/// Port of `_ResponseStreamWithRequestId._attach_request_id`.
pub fn attach_stream_request_id(event: &mut Value, request_id: Option<&str>) {
    let Some(request_id) = request_id else { return };
    if let Some(response) = event.get_mut("response").and_then(Value::as_object_mut) {
        response
            .entry(STREAM_REQUEST_ID_KEY)
            .or_insert_with(|| Value::String(request_id.to_string()));
    }
}

/// Record a finished response on its span.
///
/// Mirrors Python: the usage is always recorded; the response and the request input only when
/// tracing includes data (neither is part of the exported payload).
pub(crate) fn finish_response_span(
    span: &Span,
    resp: &ModelResponse,
    include_data: bool,
    input: &[InputItem],
) {
    span.update_data(|d| {
        if let SpanData::Response {
            response_id,
            usage,
            response,
            input: span_input,
        } = d
        {
            *response_id = resp.response_id.clone();
            *usage = Some(resp.usage.to_span_usage());
            if include_data {
                *response = serde_json::to_value(resp).ok();
                *span_input = serde_json::to_value(input).ok();
            }
        }
    });
}

/// Expand each stream item into zero or more items, optionally ending the stream afterwards.
///
/// Python yields a terminal event to the consumer and then stops reading the stream: a
/// terminal failure is raised after the event, and any transport trouble after a terminal
/// event is never seen. A plain `map` cannot emit the event and the error from one source item,
/// and a `take_while` would still poll the source once more, so `f` returns the items to emit
/// plus whether this was the last source item, and the stream ends without touching the source
/// again.
pub(crate) fn deliver_terminal_failures(
    stream: ResponseEventStream,
    f: impl FnMut(Result<Value>) -> (Vec<Result<Value>>, bool) + Send + 'static,
) -> ResponseEventStream {
    let state = (stream, std::collections::VecDeque::new(), false, f);
    futures::stream::unfold(
        state,
        |(mut inner, mut queue, mut stopped, mut f)| async move {
            loop {
                if let Some(item) = queue.pop_front() {
                    return Some((item, (inner, queue, stopped, f)));
                }
                if stopped {
                    return None;
                }
                let item = inner.next().await?;
                let (items, stop) = f(item);
                queue.extend(items);
                stopped = stop;
            }
        },
    )
    .boxed()
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
            Ok(resp) => finish_response_span(&span, resp, include_data, &request.input),
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
        let (stream, request_id) = match self
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
        let span_input = request.input.clone();
        let guard = SpanGuard(span);
        let mapped = deliver_terminal_failures(stream, move |ev| {
            let span = &guard.0;
            match ev {
                Ok(mut event) => {
                    attach_stream_request_id(&mut event, request_id.as_deref());
                    let ty = event
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    if ty == "response.completed" {
                        if let Some(r) = event.get("response")
                            && let Ok(parsed) = parse_response(r, None, false)
                        {
                            finish_response_span(span, &parsed, include_data, &span_input);
                        }
                        return (vec![Ok(event)], true);
                    }
                    if is_terminal_failure_event(&ty) {
                        let err = terminal_failure_error(&ty, &event);
                        record_model_error_on_span(
                            span,
                            "Error streaming response",
                            &err,
                            include_data,
                        );
                        // The consumer sees the terminal event first; the error follows.
                        return (vec![Ok(event), Err(err)], true);
                    }
                    (vec![Ok(event)], false)
                }
                Err(e) => {
                    record_model_error_on_span(span, "Error streaming response", &e, include_data);
                    crate::logger::log_model_action_error("Error streaming response", &e);
                    (vec![Err(e)], false)
                }
            }
        });
        Ok(mapped)
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
    #[test]
    fn a_streamed_response_falls_back_to_the_event_request_id() {
        let mut event = serde_json::json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "output": []},
        });
        super::attach_stream_request_id(&mut event, Some("req_abc"));
        let parsed = super::parse_response(&event["response"], None, false).unwrap();
        assert_eq!(parsed.request_id.as_deref(), Some("req_abc"));
        // An explicit id (the non-streaming path) still wins.
        let parsed =
            super::parse_response(&event["response"], Some("req_http".into()), false).unwrap();
        assert_eq!(parsed.request_id.as_deref(), Some("req_http"));
        // Nothing is attached when the response carried no request id.
        let mut bare = serde_json::json!({"response": {"id": "r", "status": "completed"}});
        super::attach_stream_request_id(&mut bare, None);
        assert!(bare["response"].get(super::STREAM_REQUEST_ID_KEY).is_none());
    }

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
            needs_approval: false,
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
