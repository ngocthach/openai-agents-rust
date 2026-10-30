//! The `codex_tool` function tool. Mirrors
//! `agents.extensions.experimental.codex.codex_tool`.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;

use crate::error::{AgentsError, Result};
use crate::extensions::codex::client::Codex;
use crate::extensions::codex::events::{ThreadEvent, Usage};
use crate::extensions::codex::items::ThreadItem;
use crate::extensions::codex::options::CodexOptions;
use crate::extensions::codex::output_schema::OutputSchema;
use crate::extensions::codex::py_json;
use crate::extensions::codex::span_budget::{merge_span_data, truncate_span_value};
use crate::extensions::codex::thread::{Input, StreamedTurn, Thread, UserInput};
use crate::extensions::codex::thread_options::{SandboxMode, ThreadOptions};
use crate::extensions::codex::turn_options::TurnOptions;
use crate::items::FunctionCall;
use crate::tool::{FunctionTool, ToolOutput};
use crate::tool_context::ToolContext;
use crate::tracing::{Span, SpanData, custom_span};
use crate::util::BoxFuture;

/// Default name of the Codex tool.
pub const DEFAULT_CODEX_TOOL_NAME: &str = "codex";
/// Required prefix for custom Codex tool names.
pub const CODEX_TOOL_NAME_PREFIX: &str = "codex_";
/// Default description of the Codex tool.
pub const DEFAULT_CODEX_TOOL_DESCRIPTION: &str =
    "Executes an agentic Codex task against the current workspace.";
/// Default character budget for Codex command spans.
pub const DEFAULT_SPAN_DATA_MAX_CHARS: usize = 8192;

/// The result returned by the Codex tool, sent to the model as JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodexToolResult {
    /// Thread id of the Codex conversation, when known.
    pub thread_id: Option<String>,
    /// The final Codex response.
    pub response: String,
    /// Token usage reported by the turn.
    pub usage: Option<Usage>,
}

impl fmt::Display for CodexToolResult {
    /// The exact bytes Python's `json.dumps(result.as_dict())` produces: field order, the
    /// `", "` / `": "` separators and `ensure_ascii` escaping.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (item_sep, key_sep) = py_json::DEFAULT_SEPARATORS;
        let mut out = String::from("{");
        py_json::write_string(&mut out, "thread_id");
        out.push_str(key_sep);
        match &self.thread_id {
            Some(thread_id) => py_json::write_string(&mut out, thread_id),
            None => out.push_str("null"),
        }
        out.push_str(item_sep);
        py_json::write_string(&mut out, "response");
        out.push_str(key_sep);
        py_json::write_string(&mut out, &self.response);
        out.push_str(item_sep);
        py_json::write_string(&mut out, "usage");
        out.push_str(key_sep);
        match &self.usage {
            Some(usage) => {
                // `dataclasses.asdict` keeps the declaration order of `Usage`.
                out.push('{');
                for (index, (key, value)) in [
                    ("input_tokens", usage.input_tokens),
                    ("cached_input_tokens", usage.cached_input_tokens),
                    ("output_tokens", usage.output_tokens),
                ]
                .iter()
                .enumerate()
                {
                    if index > 0 {
                        out.push_str(item_sep);
                    }
                    py_json::write_string(&mut out, key);
                    out.push_str(key_sep);
                    out.push_str(&value.to_string());
                }
                out.push('}');
            }
            None => out.push_str("null"),
        }
        out.push('}');
        f.write_str(&out)
    }
}

/// One event forwarded to the `on_stream` callback.
#[derive(Debug, Clone)]
pub struct CodexToolStreamEvent {
    /// The Codex event.
    pub event: ThreadEvent,
    /// The thread the event belongs to.
    pub thread: Thread,
    /// The model tool call that started this Codex turn, when driven by a model.
    pub tool_call: Option<FunctionCall>,
}

type OnStreamFn = Arc<dyn Fn(CodexToolStreamEvent) -> BoxFuture<'static, ()> + Send + Sync>;

/// Reads the Codex thread id from the run context.
pub type ThreadIdGetFn<C> = Arc<dyn Fn(&C) -> Option<String> + Send + Sync>;
/// Writes the Codex thread id to the run context.
pub type ThreadIdSetFn<C> = Arc<dyn Fn(&C, &str) + Send + Sync>;

/// Reads and writes the Codex thread id on the run context, so turns of one run share a thread.
pub struct ThreadIdStore<C> {
    get: ThreadIdGetFn<C>,
    set: ThreadIdSetFn<C>,
}

impl<C> Clone for ThreadIdStore<C> {
    fn clone(&self) -> Self {
        ThreadIdStore {
            get: self.get.clone(),
            set: self.set.clone(),
        }
    }
}

impl<C> fmt::Debug for ThreadIdStore<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadIdStore").finish_non_exhaustive()
    }
}

/// Configuration of [`codex_tool`].
pub struct CodexToolOptions<C> {
    /// Tool name; must be `codex` or start with `codex_`.
    pub name: Option<String>,
    /// Tool description shown to the model.
    pub description: Option<String>,
    /// Replace the generated parameters schema.
    pub params_json_schema: Option<Value>,
    /// Structured output requested from Codex.
    pub output_schema: Option<OutputSchema>,
    /// An existing Codex client; otherwise one is built from `codex_options`.
    pub codex: Option<Codex>,
    /// Options of the Codex CLI process.
    pub codex_options: Option<CodexOptions>,
    /// Thread options applied to every turn.
    pub default_thread_options: Option<ThreadOptions>,
    /// Thread id resumed when the model does not pass one.
    pub thread_id: Option<String>,
    /// Shorthand for `default_thread_options.sandbox_mode`.
    pub sandbox_mode: Option<SandboxMode>,
    /// Shorthand for `default_thread_options.working_directory`.
    pub working_directory: Option<String>,
    /// Shorthand for `default_thread_options.skip_git_repo_check`.
    pub skip_git_repo_check: Option<bool>,
    /// Turn options applied to every turn.
    pub default_turn_options: Option<TurnOptions>,
    /// Character budget for the **whole** command span payload, measured as compact
    /// `ensure_ascii` JSON; `None` means no limit. Long keys share what is left over.
    pub span_data_max_chars: Option<usize>,
    /// Include commands and their output in spans. The run's
    /// `RunConfig::trace_include_sensitive_data` can only tighten this, never loosen it.
    pub include_sensitive_span_data: bool,
    /// Reuse a single Codex thread across tool calls.
    pub persist_session: bool,
    on_stream: Option<OnStreamFn>,
    thread_id_store: Option<ThreadIdStore<C>>,
}

impl<C> Default for CodexToolOptions<C> {
    fn default() -> Self {
        CodexToolOptions {
            name: None,
            description: None,
            params_json_schema: None,
            output_schema: None,
            codex: None,
            codex_options: None,
            default_thread_options: None,
            thread_id: None,
            sandbox_mode: None,
            working_directory: None,
            skip_git_repo_check: None,
            default_turn_options: None,
            span_data_max_chars: Some(DEFAULT_SPAN_DATA_MAX_CHARS),
            include_sensitive_span_data: true,
            persist_session: false,
            on_stream: None,
            thread_id_store: None,
        }
    }
}

impl<C> fmt::Debug for CodexToolOptions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodexToolOptions")
            .field("name", &self.name)
            .field("thread_id", &self.thread_id)
            .field("persist_session", &self.persist_session)
            .finish_non_exhaustive()
    }
}

impl<C> CodexToolOptions<C> {
    /// Default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the tool name (`codex` or `codex_*`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the tool description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Replace the parameters schema.
    pub fn with_params_json_schema(mut self, schema: Value) -> Self {
        self.params_json_schema = Some(schema);
        self
    }

    /// Request structured output from Codex.
    pub fn with_output_schema(mut self, schema: impl Into<OutputSchema>) -> Self {
        self.output_schema = Some(schema.into());
        self
    }

    /// Use an existing Codex client.
    pub fn with_codex(mut self, codex: Codex) -> Self {
        self.codex = Some(codex);
        self
    }

    /// Set the Codex CLI process options.
    pub fn with_codex_options(mut self, options: CodexOptions) -> Self {
        self.codex_options = Some(options);
        self
    }

    /// Set the thread options applied to every turn.
    pub fn with_default_thread_options(mut self, options: ThreadOptions) -> Self {
        self.default_thread_options = Some(options);
        self
    }

    /// Resume this thread when the model does not pass a thread id.
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Set the sandbox mode.
    pub fn with_sandbox_mode(mut self, mode: SandboxMode) -> Self {
        self.sandbox_mode = Some(mode);
        self
    }

    /// Set the working directory.
    pub fn with_working_directory(mut self, dir: impl Into<String>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    /// Allow running outside a Git repository.
    pub fn with_skip_git_repo_check(mut self, skip: bool) -> Self {
        self.skip_git_repo_check = Some(skip);
        self
    }

    /// Set the turn options applied to every turn.
    pub fn with_default_turn_options(mut self, options: TurnOptions) -> Self {
        self.default_turn_options = Some(options);
        self
    }

    /// Set the whole-payload character budget for command spans; `None` disables truncation.
    pub fn with_span_data_max_chars(mut self, max_chars: Option<usize>) -> Self {
        self.span_data_max_chars = max_chars;
        self
    }

    /// Keep commands and their output out of spans.
    pub fn with_include_sensitive_span_data(mut self, include: bool) -> Self {
        self.include_sensitive_span_data = include;
        self
    }

    /// Reuse one Codex thread across tool calls.
    pub fn with_persist_session(mut self, persist: bool) -> Self {
        self.persist_session = persist;
        self
    }

    /// Observe every Codex event of every turn.
    pub fn with_on_stream<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(CodexToolStreamEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_stream = Some(Arc::new(move |event| Box::pin(f(event))));
        self
    }

    /// Keep the Codex thread id on the run context: `get` is read before a turn and `set` is
    /// called with the id afterwards. The `thread_id` parameter is then hidden from the model.
    ///
    /// This replaces Python's `use_run_context_thread_id` / `run_context_thread_id_key`, which
    /// write the id onto an arbitrary context object through `setattr` or mapping access under a
    /// key derived from the tool name. `RunContext<C>` exposes only `&C`, so there is no generic
    /// field to write and no key namespace to derive; the closures do the same work explicitly.
    pub fn with_thread_id_store<G, S>(mut self, get: G, set: S) -> Self
    where
        G: Fn(&C) -> Option<String> + Send + Sync + 'static,
        S: Fn(&C, &str) + Send + Sync + 'static,
    {
        self.thread_id_store = Some(ThreadIdStore {
            get: Arc::new(get),
            set: Arc::new(set),
        });
        self
    }
}

/// The parameters schema of the Codex tool.
pub fn codex_tool_params_schema(include_thread_id: bool) -> Value {
    let mut properties = json!({
        "inputs": {
            "type": "array",
            "minItems": 1,
            "description":
                "Structured inputs appended to the Codex task. Provide at least one input item.",
            "items": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "type": {"type": "string", "enum": ["text", "local_image"]},
                    "text": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Text to append to the Codex task; required for text inputs."
                    },
                    "path": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Path of a local image; required for local_image inputs."
                    }
                },
                "required": ["type", "text", "path"]
            }
        }
    });
    let mut required = vec![Value::String("inputs".into())];
    if include_thread_id {
        properties.as_object_mut().expect("object").insert(
            "thread_id".into(),
            json!({
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description":
                    "Optional Codex thread ID to resume. If omitted, a new thread is started \
                     unless configured elsewhere."
            }),
        );
        required.push(Value::String("thread_id".into()));
    }
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
        "required": required,
    })
}

/// Build a function tool that runs an agentic Codex task.
///
/// ```no_run
/// use openai_agents_sdk::extensions::codex::{CodexToolOptions, codex_tool};
///
/// let tool = codex_tool(CodexToolOptions::<()>::new().with_persist_session(true))?;
/// assert_eq!(tool.name, "codex");
/// # Ok::<(), openai_agents_sdk::AgentsError>(())
/// ```
pub fn codex_tool<C: Send + Sync + 'static>(
    options: CodexToolOptions<C>,
) -> Result<FunctionTool<C>> {
    let name = resolve_tool_name(options.name.as_deref())?;
    let description = options
        .description
        .clone()
        .unwrap_or_else(|| DEFAULT_CODEX_TOOL_DESCRIPTION.to_string());
    let include_thread_id = options.thread_id_store.is_none();
    let custom_params_schema = options.params_json_schema.clone();
    let params_json_schema = custom_params_schema
        .clone()
        .unwrap_or_else(|| codex_tool_params_schema(include_thread_id));

    let codex = match options.codex {
        Some(codex) => codex,
        None => {
            let mut codex_options = options.codex_options.unwrap_or_default();
            if codex_options.api_key.is_none()
                && let Some(api_key) = codex_options.resolve_api_key()
            {
                codex_options.api_key = Some(api_key);
            }
            Codex::new(codex_options)
        }
    };

    let thread_options = resolve_thread_options(
        options.default_thread_options,
        options.sandbox_mode,
        options.working_directory,
        options.skip_git_repo_check,
    );
    let mut turn_options = options.default_turn_options.unwrap_or_default();
    if let Some(output_schema) = &options.output_schema {
        turn_options.output_schema = Some(output_schema.to_json_schema()?);
    }

    let config = Arc::new(ToolConfig {
        codex,
        thread_options,
        turn_options,
        configured_thread_id: options.thread_id,
        persist_session: options.persist_session,
        persisted_thread: Mutex::new(None),
        span_data_max_chars: options.span_data_max_chars,
        include_sensitive_span_data: options.include_sensitive_span_data,
        on_stream: options.on_stream,
        thread_id_store: options.thread_id_store,
        custom_params_schema,
    });

    let mut tool = FunctionTool::new_raw(
        name,
        description,
        params_json_schema,
        move |ctx: ToolContext<C>, arguments: String| {
            let config = config.clone();
            async move { invoke(config, ctx, arguments).await }
        },
    );
    // Marks the tool for `Agent`'s duplicate-Codex-tool check. Port of `_is_codex_tool`.
    tool.mark_codex_tool();
    Ok(tool)
}

struct ToolConfig<C> {
    codex: Codex,
    thread_options: Option<ThreadOptions>,
    turn_options: TurnOptions,
    configured_thread_id: Option<String>,
    persist_session: bool,
    persisted_thread: Mutex<Option<Thread>>,
    span_data_max_chars: Option<usize>,
    include_sensitive_span_data: bool,
    on_stream: Option<OnStreamFn>,
    thread_id_store: Option<ThreadIdStore<C>>,
    /// The caller-provided parameters schema, which arguments are validated against.
    custom_params_schema: Option<Value>,
}

fn resolve_tool_name(configured: Option<&str>) -> Result<String> {
    let Some(configured) = configured else {
        return Ok(DEFAULT_CODEX_TOOL_NAME.to_string());
    };
    let normalized = configured.trim();
    if normalized.is_empty() {
        return Err(AgentsError::user(
            "Codex tool name must be a non-empty string.",
        ));
    }
    if normalized != DEFAULT_CODEX_TOOL_NAME && !normalized.starts_with(CODEX_TOOL_NAME_PREFIX) {
        return Err(AgentsError::user(format!(
            "Codex tool name must be \"{DEFAULT_CODEX_TOOL_NAME}\" or start with \
             \"{CODEX_TOOL_NAME_PREFIX}\"."
        )));
    }
    Ok(normalized.to_string())
}

fn resolve_thread_options(
    defaults: Option<ThreadOptions>,
    sandbox_mode: Option<SandboxMode>,
    working_directory: Option<String>,
    skip_git_repo_check: Option<bool>,
) -> Option<ThreadOptions> {
    if defaults.is_none()
        && sandbox_mode.is_none()
        && working_directory.is_none()
        && skip_git_repo_check.is_none()
    {
        return None;
    }
    let mut options = defaults.unwrap_or_default();
    if let Some(mode) = sandbox_mode {
        options.sandbox_mode = Some(mode);
    }
    if let Some(dir) = working_directory {
        options.working_directory = Some(dir);
    }
    if let Some(skip) = skip_git_repo_check {
        options.skip_git_repo_check = Some(skip);
    }
    Some(options)
}

#[derive(Debug, Clone, Deserialize)]
struct RawToolInputItem {
    #[serde(rename = "type")]
    item_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawToolCallArgs {
    inputs: Vec<RawToolInputItem>,
    #[serde(default)]
    thread_id: Option<String>,
}

const ARGUMENT_KEYS: [&str; 2] = ["inputs", "thread_id"];
const INPUT_ITEM_KEYS: [&str; 3] = ["type", "text", "path"];

/// Reject fields the default parameters schema forbids (`additionalProperties: false`).
fn reject_unknown_fields(raw: &Value) -> Result<()> {
    let Some(map) = raw.as_object() else {
        return Ok(());
    };
    for key in map.keys() {
        if !ARGUMENT_KEYS.contains(&key.as_str()) {
            return Err(invalid_input(format!("unknown field \"{key}\"")));
        }
    }
    if let Some(Value::Array(items)) = map.get("inputs") {
        for item in items {
            let Some(item) = item.as_object() else {
                continue;
            };
            for key in item.keys() {
                if !INPUT_ITEM_KEYS.contains(&key.as_str()) {
                    return Err(invalid_input(format!("unknown input field \"{key}\"")));
                }
            }
        }
    }
    Ok(())
}

fn invalid_input(message: impl fmt::Display) -> AgentsError {
    AgentsError::model_behavior(format!("Invalid JSON input for codex tool: {message}"))
}

/// Validate the arguments against a caller-supplied parameters schema.
///
/// Python validates against the caller's own Pydantic model; this honours the parts of a JSON
/// schema that reject arguments outright: `required` and `additionalProperties: false`, at the
/// top level and on the items of `inputs`.
fn validate_against_schema(value: &Value, schema: &Value) -> Result<()> {
    check_object_against_schema(value, schema, "field")?;
    let item_schema = &schema["properties"]["inputs"]["items"];
    if item_schema.is_object()
        && let Some(items) = value.get("inputs").and_then(Value::as_array)
    {
        for item in items {
            check_object_against_schema(item, item_schema, "input field")?;
        }
    }
    Ok(())
}

fn check_object_against_schema(value: &Value, schema: &Value, label: &str) -> Result<()> {
    let Some(map) = value.as_object() else {
        return Ok(());
    };
    if schema["additionalProperties"] == Value::Bool(false) {
        let properties = schema["properties"].as_object();
        for key in map.keys() {
            if !properties.is_some_and(|props| props.contains_key(key)) {
                return Err(invalid_input(format!("unknown {label} \"{key}\"")));
            }
        }
    }
    if let Some(required) = schema["required"].as_array() {
        for name in required.iter().filter_map(Value::as_str) {
            if !map.contains_key(name) {
                return Err(invalid_input(format!("missing {label} \"{name}\"")));
            }
        }
    }
    Ok(())
}

/// Parse and validate the tool arguments into Codex inputs and an optional thread id.
///
/// `custom_schema` is the caller-supplied parameters schema, if any. Without one the default
/// schema's `additionalProperties: false` is enforced directly.
fn parse_tool_input(
    arguments: &str,
    custom_schema: Option<&Value>,
) -> Result<(Vec<UserInput>, Option<String>)> {
    let text = if arguments.trim().is_empty() {
        "{}"
    } else {
        arguments
    };
    let value: Value = serde_json::from_str(text).map_err(invalid_input)?;
    match custom_schema {
        Some(schema) => validate_against_schema(&value, schema)?,
        None => reject_unknown_fields(&value)?,
    }
    let raw: RawToolCallArgs = serde_json::from_value(value).map_err(invalid_input)?;

    if raw.inputs.is_empty() {
        return Err(invalid_input(
            "\"inputs\" must contain at least one input item.",
        ));
    }

    let mut inputs: Vec<UserInput> = Vec::with_capacity(raw.inputs.len());
    for item in raw.inputs {
        let text = item.text.unwrap_or_default().trim().to_string();
        let path = item.path.unwrap_or_default().trim().to_string();
        match item.item_type.as_str() {
            "text" => {
                if text.is_empty() {
                    return Err(invalid_input(
                        "Text inputs must include a non-empty \"text\" field.",
                    ));
                }
                if !path.is_empty() {
                    return Err(invalid_input(
                        "\"path\" is not allowed when type is \"text\".",
                    ));
                }
                inputs.push(UserInput::Text { text });
            }
            "local_image" => {
                if path.is_empty() {
                    return Err(invalid_input(
                        "Local image inputs must include a non-empty \"path\" field.",
                    ));
                }
                if !text.is_empty() {
                    return Err(invalid_input(
                        "\"text\" is not allowed when type is \"local_image\".",
                    ));
                }
                inputs.push(UserInput::LocalImage { path });
            }
            other => {
                return Err(invalid_input(format!(
                    "Unknown input type \"{other}\"; expected \"text\" or \"local_image\"."
                )));
            }
        }
    }

    let thread_id = raw
        .thread_id
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty());
    Ok((inputs, thread_id))
}

fn default_response(inputs: &[UserInput]) -> String {
    let summary = if inputs.is_empty() {
        "with no inputs."
    } else {
        "with inputs."
    };
    format!("Codex task completed {summary}")
}

async fn invoke<C: Send + Sync + 'static>(
    config: Arc<ToolConfig<C>>,
    ctx: ToolContext<C>,
    arguments: String,
) -> Result<ToolOutput> {
    let (inputs, call_thread_id) =
        parse_tool_input(&arguments, config.custom_params_schema.as_ref())?;
    let call_thread_id = call_thread_id
        .or_else(|| {
            config
                .thread_id_store
                .as_ref()
                .and_then(|store| (store.get)(ctx.context()))
                .map(|id| id.trim().to_string())
                .filter(|id| !id.is_empty())
        })
        .or_else(|| config.configured_thread_id.clone());

    let thread = resolve_thread(&config, call_thread_id.clone())?;
    let mut stream = thread
        .run_streamed(
            Input::Items(inputs.clone()),
            Some(config.turn_options.clone()),
        )
        .await?;

    let mut state = ConsumeState {
        final_response: String::new(),
        usage: None,
        thread_id: thread.id().or(call_thread_id),
    };
    let mut spans: HashMap<String, Span> = HashMap::new();
    let outcome = consume_events(&config, &ctx, &thread, &mut stream, &mut spans, &mut state).await;
    // Close spans even when the turn failed.
    for (_, span) in spans.drain() {
        span.finish();
    }

    if let (Some(store), Some(thread_id)) = (&config.thread_id_store, &state.thread_id) {
        (store.set)(ctx.context(), thread_id);
    }
    outcome?;

    if let Some(usage) = state.usage {
        ctx.add_usage(&usage.to_agent_usage());
    }
    if state.final_response.is_empty() {
        state.final_response = default_response(&inputs);
    }

    let result = CodexToolResult {
        thread_id: state.thread_id,
        response: state.final_response,
        usage: state.usage,
    };
    Ok(ToolOutput::Text(result.to_string()))
}

fn resolve_thread<C>(config: &ToolConfig<C>, call_thread_id: Option<String>) -> Result<Thread> {
    if !config.persist_session {
        return Ok(new_thread(config, call_thread_id));
    }
    let mut persisted = config
        .persisted_thread
        .lock()
        .expect("codex tool state poisoned");
    if let Some(existing) = persisted.as_ref() {
        if let (Some(requested), Some(current)) = (&call_thread_id, existing.id())
            && *requested != current
        {
            return Err(AgentsError::user(
                "Codex tool is configured with persist_session=true and already has an active \
                 thread.",
            ));
        }
        return Ok(existing.clone());
    }
    let thread = new_thread(config, call_thread_id);
    *persisted = Some(thread.clone());
    Ok(thread)
}

fn new_thread<C>(config: &ToolConfig<C>, call_thread_id: Option<String>) -> Thread {
    match call_thread_id {
        Some(thread_id) => config
            .codex
            .resume_thread(thread_id, config.thread_options.clone()),
        None => config.codex.start_thread(config.thread_options.clone()),
    }
}

struct ConsumeState {
    final_response: String,
    usage: Option<Usage>,
    thread_id: Option<String>,
}

async fn consume_events<C: Send + Sync + 'static>(
    config: &ToolConfig<C>,
    ctx: &ToolContext<C>,
    thread: &Thread,
    stream: &mut StreamedTurn,
    spans: &mut HashMap<String, Span>,
    state: &mut ConsumeState,
) -> Result<()> {
    let policy = TracingPolicy::resolve(config, ctx);
    // Buffer events so a slow `on_stream` callback cannot block the Codex stream loop.
    let dispatcher = config.on_stream.as_ref().map(|on_stream| {
        let (sender, mut receiver) = mpsc::unbounded_channel::<CodexToolStreamEvent>();
        let on_stream = on_stream.clone();
        let consumer = tokio::spawn(async move {
            while let Some(payload) = receiver.recv().await {
                on_stream(payload).await;
            }
        });
        (sender, consumer)
    });

    let outcome = process_events(
        config,
        ctx,
        &policy,
        thread,
        stream,
        spans,
        state,
        &dispatcher,
    )
    .await;

    if let Some((sender, consumer)) = dispatcher {
        // Closing the channel ends the consumer once it drained every queued event.
        drop(sender);
        let _ = consumer.await;
    }
    outcome
}

type Dispatcher = Option<(
    mpsc::UnboundedSender<CodexToolStreamEvent>,
    tokio::task::JoinHandle<()>,
)>;

#[allow(clippy::too_many_arguments)]
async fn process_events<C: Send + Sync + 'static>(
    config: &ToolConfig<C>,
    ctx: &ToolContext<C>,
    policy: &TracingPolicy,
    thread: &Thread,
    stream: &mut StreamedTurn,
    spans: &mut HashMap<String, Span>,
    state: &mut ConsumeState,
    dispatcher: &Dispatcher,
) -> Result<()> {
    while let Some(event) = stream.next_event().await {
        let event = event?;
        if let Some((sender, _)) = dispatcher {
            let _ = sender.send(CodexToolStreamEvent {
                event: event.clone(),
                thread: thread.clone(),
                tool_call: ctx.tool_call.clone(),
            });
        }
        match event {
            ThreadEvent::ThreadStarted { thread_id } => state.thread_id = Some(thread_id),
            ThreadEvent::ItemStarted { item } => start_item_span(config, policy, spans, &item),
            ThreadEvent::ItemUpdated { item } => update_item_span(config, policy, spans, &item),
            ThreadEvent::ItemCompleted { item } => {
                complete_item_span(config, policy, spans, &item);
                if let Some(text) = item.as_agent_message() {
                    state.final_response = text.to_string();
                }
            }
            ThreadEvent::TurnCompleted { usage } => state.usage = usage,
            ThreadEvent::TurnFailed { error } => {
                let suffix = if error.message.is_empty() {
                    String::new()
                } else {
                    format!(": {}", error.message)
                };
                return Err(AgentsError::user(format!("Codex turn failed{suffix}")));
            }
            ThreadEvent::Error { message } => {
                return Err(AgentsError::user(format!("Codex stream error: {message}")));
            }
            ThreadEvent::TurnStarted | ThreadEvent::Unknown { .. } => {}
        }
    }
    Ok(())
}

/// The span updates for a command execution item, or `None` for any other item.
/// Port of the `updates` dict shared by `_handle_item_started` and `_update_command_span`.
fn command_span_updates(
    policy: &TracingPolicy,
    max_chars: Option<usize>,
    item: &ThreadItem,
) -> Option<Map<String, Value>> {
    let ThreadItem::CommandExecution {
        command,
        status,
        aggregated_output,
        exit_code,
        ..
    } = item
    else {
        return None;
    };
    let mut updates = Map::new();
    updates.insert("status".into(), Value::String(status.clone()));
    updates.insert(
        "exit_code".into(),
        exit_code.map(Value::from).unwrap_or(Value::Null),
    );
    if policy.include_sensitive_data {
        updates.insert("command".into(), Value::String(command.clone()));
        if !aggregated_output.is_empty() {
            updates.insert(
                "output".into(),
                truncate_span_value(&Value::String(aggregated_output.clone()), max_chars),
            );
        }
    }
    Some(updates)
}

fn apply_span_updates(span: &Span, updates: &Map<String, Value>, max_chars: Option<usize>) {
    span.update_data(|current| {
        if let SpanData::Custom { data, .. } = current {
            let existing = data.as_object().cloned().unwrap_or_default();
            *data = Value::Object(merge_span_data(&existing, updates, max_chars));
        }
    });
}

fn start_item_span<C>(
    config: &ToolConfig<C>,
    policy: &TracingPolicy,
    spans: &mut HashMap<String, Span>,
    item: &ThreadItem,
) {
    if policy.tracing_disabled {
        return;
    }
    let Some(item_id) = item.id().filter(|id| !id.is_empty()) else {
        return;
    };
    let max_chars = config.span_data_max_chars;
    let Some(updates) = command_span_updates(policy, max_chars, item) else {
        return;
    };
    let data = merge_span_data(&Map::new(), &updates, max_chars);
    let span = custom_span("Codex command execution", Value::Object(data));
    span.start();
    spans.insert(item_id.to_string(), span);
}

fn update_item_span<C>(
    config: &ToolConfig<C>,
    policy: &TracingPolicy,
    spans: &mut HashMap<String, Span>,
    item: &ThreadItem,
) {
    let Some(item_id) = item.id() else {
        return;
    };
    let Some(span) = spans.get(item_id) else {
        return;
    };
    if let Some(updates) = command_span_updates(policy, config.span_data_max_chars, item) {
        apply_span_updates(span, &updates, config.span_data_max_chars);
    }
}

fn complete_item_span<C>(
    config: &ToolConfig<C>,
    policy: &TracingPolicy,
    spans: &mut HashMap<String, Span>,
    item: &ThreadItem,
) {
    let Some(item_id) = item.id() else {
        return;
    };
    let Some(span) = spans.remove(item_id) else {
        return;
    };
    if let Some(updates) = command_span_updates(policy, config.span_data_max_chars, item) {
        apply_span_updates(&span, &updates, config.span_data_max_chars);
    }
    if let ThreadItem::CommandExecution {
        status,
        aggregated_output,
        exit_code,
        ..
    } = item
        && status == "failed"
    {
        let mut error_data = json!({"exit_code": exit_code});
        if policy.include_sensitive_data && !aggregated_output.is_empty() {
            error_data.as_object_mut().expect("object").insert(
                "output".into(),
                truncate_span_value(
                    &Value::String(aggregated_output.clone()),
                    config.span_data_max_chars,
                ),
            );
        }
        span.set_error("Codex command execution failed.", Some(error_data));
    }
    span.finish();
}

/// The run's tracing policy, as `_consume_events` reads it off the `RunConfig`.
#[derive(Debug, Clone, Copy)]
struct TracingPolicy {
    tracing_disabled: bool,
    include_sensitive_data: bool,
}

impl TracingPolicy {
    fn resolve<C>(config: &ToolConfig<C>, ctx: &ToolContext<C>) -> Self {
        let run_config = ctx.run_config.as_ref();
        TracingPolicy {
            tracing_disabled: run_config.is_some_and(|rc| rc.tracing_disabled),
            include_sensitive_data: config.include_sensitive_span_data
                && run_config.is_none_or(|rc| rc.trace_include_sensitive_data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::stream::StreamExt;

    fn test_config(persist_session: bool) -> ToolConfig<()> {
        ToolConfig {
            codex: Codex::new(CodexOptions::default()),
            thread_options: None,
            turn_options: TurnOptions::default(),
            configured_thread_id: None,
            persist_session,
            persisted_thread: Mutex::new(None),
            span_data_max_chars: Some(DEFAULT_SPAN_DATA_MAX_CHARS),
            include_sensitive_span_data: true,
            on_stream: None,
            thread_id_store: None,
            custom_params_schema: None,
        }
    }

    const PERMISSIVE: TracingPolicy = TracingPolicy {
        tracing_disabled: false,
        include_sensitive_data: true,
    };

    #[test]
    fn resolves_tool_names() {
        assert_eq!(resolve_tool_name(None).unwrap(), "codex");
        assert_eq!(
            resolve_tool_name(Some(" codex_review ")).unwrap(),
            "codex_review"
        );
        assert!(resolve_tool_name(Some("review")).is_err());
        assert!(resolve_tool_name(Some("  ")).is_err());
    }

    #[test]
    fn builds_the_default_parameters_schema() {
        let tool: FunctionTool<()> = codex_tool(CodexToolOptions::new()).unwrap();
        assert_eq!(tool.name, "codex");
        assert_eq!(tool.description, DEFAULT_CODEX_TOOL_DESCRIPTION);
        let schema = &tool.params_json_schema;
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"], json!(["inputs", "thread_id"]));
        assert_eq!(schema["properties"]["inputs"]["minItems"], 1);
        let mut item_required: Vec<&str> = schema["properties"]["inputs"]["items"]["required"]
            .as_array()
            .expect("required")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        item_required.sort_unstable();
        assert_eq!(item_required, vec!["path", "text", "type"]);
    }

    #[test]
    fn thread_id_store_hides_the_thread_id_parameter() {
        let tool: FunctionTool<()> = codex_tool(
            CodexToolOptions::new().with_thread_id_store(|_: &()| None, |_: &(), _: &str| {}),
        )
        .unwrap();
        assert_eq!(tool.params_json_schema["required"], json!(["inputs"]));
        assert!(
            tool.params_json_schema["properties"]
                .get("thread_id")
                .is_none()
        );
    }

    #[test]
    fn parses_tool_input() {
        let (inputs, thread_id) = parse_tool_input(
            r#"{"inputs":[{"type":"text","text":" fix it ","path":""},
                          {"type":"local_image","path":" /a.png ","text":null}],
                "thread_id":" thread-1 "}"#,
            None,
        )
        .unwrap();
        assert_eq!(
            inputs,
            vec![UserInput::text("fix it"), UserInput::local_image("/a.png"),]
        );
        assert_eq!(thread_id.as_deref(), Some("thread-1"));
    }

    #[test]
    fn rejects_invalid_tool_input() {
        for arguments in [
            "{bad",
            r#"{"inputs":[]}"#,
            r#"{"inputs":[{"type":"text","text":"  "}]}"#,
            r#"{"inputs":[{"type":"text","text":"hi","path":"/a.png"}]}"#,
            r#"{"inputs":[{"type":"local_image"}]}"#,
            r#"{"inputs":[{"type":"local_image","path":"/a.png","text":"hi"}]}"#,
            r#"{"inputs":[{"type":"video","path":"/a.mp4"}]}"#,
            r#"{"inputs":[{"type":"text","text":"hi"}],"extra":1}"#,
        ] {
            let error = parse_tool_input(arguments, None).unwrap_err();
            assert!(
                matches!(error, AgentsError::ModelBehaviorError { .. }),
                "expected a model behavior error for {arguments}"
            );
            assert!(
                error
                    .to_string()
                    .starts_with("Invalid JSON input for codex tool")
            );
        }
    }

    #[test]
    fn empty_thread_id_falls_back_to_none() {
        let (_, thread_id) = parse_tool_input(
            r#"{"inputs":[{"type":"text","text":"hi"}],"thread_id":"  "}"#,
            None,
        )
        .unwrap();
        assert_eq!(thread_id, None);
    }

    #[test]
    fn a_custom_schema_validates_its_own_fields() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "type": {"type": "string"},
                            "text": {"type": "string"},
                            "note": {"type": "string"}
                        },
                        "required": ["type", "text"]
                    }
                }
            },
            "required": ["inputs"]
        });
        // Fields the custom schema declares are accepted, unlike with the default schema.
        let allowed = r#"{"inputs":[{"type":"text","text":"hi","note":"x"}]}"#;
        assert!(parse_tool_input(allowed, None).is_err());
        let (inputs, thread_id) = parse_tool_input(allowed, Some(&schema)).unwrap();
        assert_eq!(inputs, vec![UserInput::text("hi")]);
        assert_eq!(thread_id, None);

        // `additionalProperties: false` and `required` are still enforced.
        let extra_top_level = r#"{"inputs":[{"type":"text","text":"hi"}],"extra":1}"#;
        assert!(
            parse_tool_input(extra_top_level, Some(&schema))
                .unwrap_err()
                .to_string()
                .contains("unknown field \"extra\"")
        );
        let extra_item_field = r#"{"inputs":[{"type":"text","text":"hi","rogue":1}]}"#;
        assert!(
            parse_tool_input(extra_item_field, Some(&schema))
                .unwrap_err()
                .to_string()
                .contains("unknown input field \"rogue\"")
        );
        let missing_required = r#"{"inputs":[{"type":"text"}]}"#;
        assert!(
            parse_tool_input(missing_required, Some(&schema))
                .unwrap_err()
                .to_string()
                .contains("missing input field \"text\"")
        );
        assert!(
            parse_tool_input("{}", Some(&schema))
                .unwrap_err()
                .to_string()
                .contains("missing field \"inputs\"")
        );
    }

    #[test]
    fn an_open_custom_schema_ignores_extra_fields() {
        let schema = json!({"type": "object", "properties": {}});
        let (inputs, _) = parse_tool_input(
            r#"{"inputs":[{"type":"text","text":"hi","note":"x"}],"extra":1}"#,
            Some(&schema),
        )
        .unwrap();
        assert_eq!(inputs, vec![UserInput::text("hi")]);
    }

    #[test]
    fn formats_the_default_response() {
        assert_eq!(
            default_response(&[]),
            "Codex task completed with no inputs."
        );
        assert_eq!(
            default_response(&[UserInput::text("hi")]),
            "Codex task completed with inputs."
        );
    }

    #[test]
    fn serializes_the_tool_result_like_python_json_dumps() {
        let result = CodexToolResult {
            thread_id: Some("thread-1".into()),
            response: "ok".into(),
            usage: None,
        };
        assert_eq!(
            result.to_string(),
            r#"{"thread_id": "thread-1", "response": "ok", "usage": null}"#
        );

        let with_usage = CodexToolResult {
            thread_id: None,
            // `ensure_ascii` escapes every non-ASCII character, astral ones as a surrogate pair.
            response: "résumé ✅ 😀".into(),
            usage: Some(Usage {
                input_tokens: 10,
                cached_input_tokens: 1,
                output_tokens: 5,
            }),
        };
        assert_eq!(
            with_usage.to_string(),
            "{\"thread_id\": null, \"response\": \
             \"r\\u00e9sum\\u00e9 \\u2705 \\ud83d\\ude00\", \"usage\": {\"input_tokens\": 10, \
             \"cached_input_tokens\": 1, \"output_tokens\": 5}}"
        );
    }

    #[test]
    fn merges_command_span_data() {
        let started = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "pytest".into(),
            status: "in_progress".into(),
            aggregated_output: String::new(),
            exit_code: None,
        };
        let updates = command_span_updates(&PERMISSIVE, None, &started).unwrap();
        let data = merge_span_data(&Map::new(), &updates, None);
        assert_eq!(data["command"], "pytest");
        assert!(data.get("output").is_none());

        let completed = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "pytest".into(),
            status: "completed".into(),
            aggregated_output: "All good".into(),
            exit_code: Some(0),
        };
        let merged = merge_span_data(
            &data,
            &command_span_updates(&PERMISSIVE, None, &completed).unwrap(),
            None,
        );
        assert_eq!(merged["status"], "completed");
        assert_eq!(merged["output"], "All good");
        assert_eq!(merged["exit_code"], 0);
    }

    #[test]
    fn omits_sensitive_span_data_when_disabled() {
        let policy = TracingPolicy {
            tracing_disabled: false,
            include_sensitive_data: false,
        };
        let item = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "pytest".into(),
            status: "completed".into(),
            aggregated_output: "All good".into(),
            exit_code: Some(0),
        };
        let updates = command_span_updates(&policy, None, &item).unwrap();
        assert!(updates.get("command").is_none());
        assert!(updates.get("output").is_none());
    }

    #[test]
    fn the_run_config_decides_the_tracing_policy() {
        use crate::run_config::RunConfig;
        use crate::run_context::RunContext;

        let config = test_config(false);
        let bare = ToolContext::direct(RunContext::new(()), "agent", "codex", "{}");
        let policy = TracingPolicy::resolve(&config, &bare);
        assert!(!policy.tracing_disabled);
        assert!(policy.include_sensitive_data);

        let run_config = Arc::new(
            RunConfig::<()>::default()
                .with_tracing_disabled(true)
                .with_trace_include_sensitive_data(false),
        );
        let restricted = bare.clone().with_run_config(run_config);
        let policy = TracingPolicy::resolve(&config, &restricted);
        assert!(policy.tracing_disabled);
        assert!(!policy.include_sensitive_data);
    }

    #[test]
    fn a_disabled_run_creates_no_command_span() {
        let config = test_config(false);
        let item = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "pytest".into(),
            status: "in_progress".into(),
            aggregated_output: String::new(),
            exit_code: None,
        };
        let mut spans = HashMap::new();
        let disabled = TracingPolicy {
            tracing_disabled: true,
            include_sensitive_data: true,
        };
        start_item_span(&config, &disabled, &mut spans, &item);
        assert!(spans.is_empty());
        start_item_span(&config, &PERMISSIVE, &mut spans, &item);
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn the_span_budget_covers_the_whole_payload() {
        let mut config = test_config(false);
        config.span_data_max_chars = Some(120);
        let item = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "c".repeat(200),
            status: "completed".into(),
            aggregated_output: "o".repeat(400),
            exit_code: Some(0),
        };
        let updates = command_span_updates(&PERMISSIVE, config.span_data_max_chars, &item).unwrap();
        let data = merge_span_data(&Map::new(), &updates, config.span_data_max_chars);
        let size = py_json::char_size(&Value::Object(data.clone()));
        assert!(size <= 120, "payload of {size} chars exceeds the budget");
        assert_eq!(data["status"], "completed");
        assert!(data["command"].as_str().expect("command").len() < 200);
    }

    #[tokio::test]
    async fn on_stream_callbacks_do_not_block_the_codex_stream() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pulled = Arc::new(AtomicUsize::new(0));
        let handled = Arc::new(AtomicUsize::new(0));
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let release = Arc::new(Mutex::new(Some(release_rx)));

        let handled_in_callback = handled.clone();
        let mut config = test_config(false);
        config.on_stream = Some(Arc::new(move |_event| {
            let handled = handled_in_callback.clone();
            let release = release.clone();
            Box::pin(async move {
                // The first callback blocks until the test releases it.
                let waiter = release.lock().expect("poisoned").take();
                if let Some(waiter) = waiter {
                    let _ = waiter.await;
                }
                handled.fetch_add(1, Ordering::SeqCst);
            })
        }));

        let events: Vec<Result<ThreadEvent>> = (0..5)
            .map(|index| {
                Ok(ThreadEvent::ItemCompleted {
                    item: ThreadItem::AgentMessage {
                        id: format!("agent-{index}"),
                        text: format!("message {index}"),
                    },
                })
            })
            .collect();
        let pulled_in_stream = pulled.clone();
        let stream = futures::stream::iter(events)
            .inspect(move |_| {
                pulled_in_stream.fetch_add(1, Ordering::SeqCst);
            })
            .boxed();
        let mut streamed = StreamedTurn::from_events(stream);

        let codex = Codex::new(CodexOptions::default());
        let thread = codex.start_thread(None);
        let ctx = ToolContext::direct(crate::run_context::RunContext::new(()), "a", "codex", "{}");
        let mut spans = HashMap::new();
        let mut state = ConsumeState {
            final_response: String::new(),
            usage: None,
            thread_id: None,
        };

        let consume = tokio::spawn(async move {
            let _ = consume_events(
                &config,
                &ctx,
                &thread,
                &mut streamed,
                &mut spans,
                &mut state,
            )
            .await;
            state.final_response
        });

        // The producer drains the Codex stream while the first callback is still blocked.
        while pulled.load(Ordering::SeqCst) < 5 {
            tokio::task::yield_now().await;
        }
        assert_eq!(handled.load(Ordering::SeqCst), 0);

        release_tx.send(()).expect("release");
        let final_response = consume.await.expect("join");
        assert_eq!(final_response, "message 4");
        assert_eq!(handled.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn merges_thread_option_shorthands() {
        let options = resolve_thread_options(
            Some(ThreadOptions::new().with_model("gpt-5-codex")),
            Some(SandboxMode::ReadOnly),
            Some("/repo".into()),
            Some(true),
        )
        .unwrap();
        assert_eq!(options.model.as_deref(), Some("gpt-5-codex"));
        assert_eq!(options.sandbox_mode, Some(SandboxMode::ReadOnly));
        assert_eq!(options.working_directory.as_deref(), Some("/repo"));
        assert_eq!(options.skip_git_repo_check, Some(true));
        assert!(resolve_thread_options(None, None, None, None).is_none());
    }

    #[test]
    fn persisted_sessions_reuse_one_thread() {
        let config = test_config(true);
        let first = resolve_thread(&config, Some("thread-1".into())).unwrap();
        assert_eq!(first.id().as_deref(), Some("thread-1"));
        let second = resolve_thread(&config, None).unwrap();
        assert_eq!(second.id().as_deref(), Some("thread-1"));
        let error = resolve_thread(&config, Some("thread-2".into())).unwrap_err();
        assert!(error.to_string().contains("already has an active thread"));
    }

    #[test]
    fn non_persisted_sessions_start_a_fresh_thread_per_call() {
        let config = test_config(false);
        let first = resolve_thread(&config, Some("thread-1".into())).unwrap();
        assert_eq!(first.id().as_deref(), Some("thread-1"));
        let second = resolve_thread(&config, None).unwrap();
        assert_eq!(second.id(), None);
    }
}
