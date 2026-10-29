//! The `codex_tool` function tool. Mirrors
//! `agents.extensions.experimental.codex.codex_tool`.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{AgentsError, Result};
use crate::extensions::codex::client::Codex;
use crate::extensions::codex::events::{ThreadEvent, Usage};
use crate::extensions::codex::items::ThreadItem;
use crate::extensions::codex::options::CodexOptions;
use crate::extensions::codex::output_schema::OutputSchema;
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string(self) {
            Ok(json) => f.write_str(&json),
            Err(_) => f.write_str("{}"),
        }
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
    /// Character budget for command span data; `None` means no limit.
    pub span_data_max_chars: Option<usize>,
    /// Include commands and their output in spans.
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

    /// Set the character budget for command spans; `None` disables truncation.
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
    let custom_params_schema = options.params_json_schema.is_some();
    let params_json_schema = options
        .params_json_schema
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

    Ok(FunctionTool::new_raw(
        name,
        description,
        params_json_schema,
        move |ctx: ToolContext<C>, arguments: String| {
            let config = config.clone();
            async move { invoke(config, ctx, arguments).await }
        },
    ))
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
    /// A caller-provided schema may allow fields the default schema forbids.
    custom_params_schema: bool,
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

/// Parse and validate the tool arguments into Codex inputs and an optional thread id.
///
/// `reject_extra_fields` mirrors the `additionalProperties: false` of the default schema; a custom
/// parameters schema may carry extra fields, which are then ignored.
fn parse_tool_input(
    arguments: &str,
    reject_extra_fields: bool,
) -> Result<(Vec<UserInput>, Option<String>)> {
    let text = if arguments.trim().is_empty() {
        "{}"
    } else {
        arguments
    };
    let value: Value = serde_json::from_str(text).map_err(invalid_input)?;
    if reject_extra_fields {
        reject_unknown_fields(&value)?;
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
    let (inputs, call_thread_id) = parse_tool_input(&arguments, !config.custom_params_schema)?;
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
    while let Some(event) = stream.next_event().await {
        let event = event?;
        if let Some(on_stream) = &config.on_stream {
            on_stream(CodexToolStreamEvent {
                event: event.clone(),
                thread: thread.clone(),
                tool_call: ctx.tool_call.clone(),
            })
            .await;
        }
        match event {
            ThreadEvent::ThreadStarted { thread_id } => state.thread_id = Some(thread_id),
            ThreadEvent::ItemStarted { item } => start_item_span(config, spans, &item),
            ThreadEvent::ItemUpdated { item } => update_item_span(config, spans, &item),
            ThreadEvent::ItemCompleted { item } => {
                complete_item_span(config, spans, &item);
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

fn command_span_data<C>(config: &ToolConfig<C>, item: &ThreadItem) -> Option<Value> {
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
    let mut data = json!({"status": status, "exit_code": exit_code});
    if config.include_sensitive_span_data {
        let map = data.as_object_mut().expect("object");
        map.insert("command".into(), Value::String(command.clone()));
        if !aggregated_output.is_empty() {
            map.insert(
                "output".into(),
                Value::String(truncate_span_string(
                    aggregated_output,
                    config.span_data_max_chars,
                )),
            );
        }
    }
    Some(data)
}

fn start_item_span<C>(
    config: &ToolConfig<C>,
    spans: &mut HashMap<String, Span>,
    item: &ThreadItem,
) {
    let Some(item_id) = item.id().filter(|id| !id.is_empty()) else {
        return;
    };
    let Some(data) = command_span_data(config, item) else {
        return;
    };
    let span = custom_span("Codex command execution", data);
    span.start();
    spans.insert(item_id.to_string(), span);
}

fn update_item_span<C>(
    config: &ToolConfig<C>,
    spans: &mut HashMap<String, Span>,
    item: &ThreadItem,
) {
    let Some(item_id) = item.id() else {
        return;
    };
    let Some(span) = spans.get(item_id) else {
        return;
    };
    if let Some(data) = command_span_data(config, item) {
        span.update_data(|current| {
            if let SpanData::Custom { data: existing, .. } = current {
                merge_span_data(existing, &data);
            }
        });
    }
}

fn complete_item_span<C>(
    config: &ToolConfig<C>,
    spans: &mut HashMap<String, Span>,
    item: &ThreadItem,
) {
    let Some(item_id) = item.id() else {
        return;
    };
    let Some(span) = spans.remove(item_id) else {
        return;
    };
    if let Some(data) = command_span_data(config, item) {
        span.update_data(|current| {
            if let SpanData::Custom { data: existing, .. } = current {
                merge_span_data(existing, &data);
            }
        });
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
        if config.include_sensitive_span_data && !aggregated_output.is_empty() {
            error_data.as_object_mut().expect("object").insert(
                "output".into(),
                Value::String(truncate_span_string(
                    aggregated_output,
                    config.span_data_max_chars,
                )),
            );
        }
        span.set_error("Codex command execution failed.", Some(error_data));
    }
    span.finish();
}

/// Merge `updates` into `data`, keeping values the update does not carry.
fn merge_span_data(data: &mut Value, updates: &Value) {
    let (Some(target), Some(source)) = (data.as_object_mut(), updates.as_object()) else {
        *data = updates.clone();
        return;
    };
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

/// Truncate a span value to `max_chars` characters, mirroring the Python suffix format.
fn truncate_span_string(value: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return value.to_string();
    };
    if max_chars == 0 {
        return String::new();
    }
    let total = value.chars().count();
    if total <= max_chars {
        return value.to_string();
    }
    let suffix = format!("... [truncated, {total} chars]");
    let suffix_len = suffix.chars().count();
    if max_chars <= suffix_len {
        return value.chars().take(max_chars).collect();
    }
    let mut truncated: String = value.chars().take(max_chars - suffix_len).collect();
    truncated.push_str(&suffix);
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

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
            custom_params_schema: false,
        }
    }

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
            true,
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
            let error = parse_tool_input(arguments, true).unwrap_err();
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
            true,
        )
        .unwrap();
        assert_eq!(thread_id, None);
    }

    #[test]
    fn a_custom_schema_may_carry_extra_fields() {
        let arguments = r#"{"inputs":[{"type":"text","text":"hi","note":"x"}],"extra":1}"#;
        assert!(parse_tool_input(arguments, true).is_err());
        let (inputs, thread_id) = parse_tool_input(arguments, false).unwrap();
        assert_eq!(inputs, vec![UserInput::text("hi")]);
        assert_eq!(thread_id, None);
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
    fn serializes_the_tool_result_in_python_field_order() {
        let result = CodexToolResult {
            thread_id: Some("thread-1".into()),
            response: "ok".into(),
            usage: None,
        };
        assert_eq!(
            result.to_string(),
            r#"{"thread_id":"thread-1","response":"ok","usage":null}"#
        );
    }

    #[test]
    fn truncates_span_strings() {
        assert_eq!(truncate_span_string("hello", None), "hello");
        assert_eq!(truncate_span_string("hello", Some(0)), "");
        assert_eq!(truncate_span_string("hello", Some(5)), "hello");
        assert_eq!(truncate_span_string("hello", Some(3)), "hel");
        let value = "x".repeat(100);
        let truncated = truncate_span_string(&value, Some(40));
        assert_eq!(truncated.chars().count(), 40);
        assert!(truncated.ends_with("... [truncated, 100 chars]"));
    }

    #[test]
    fn merges_command_span_data() {
        let config = test_config(false);
        let started = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "pytest".into(),
            status: "in_progress".into(),
            aggregated_output: String::new(),
            exit_code: None,
        };
        let mut data = command_span_data(&config, &started).unwrap();
        assert_eq!(data["command"], "pytest");
        assert!(data.get("output").is_none());

        let completed = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "pytest".into(),
            status: "completed".into(),
            aggregated_output: "All good".into(),
            exit_code: Some(0),
        };
        merge_span_data(&mut data, &command_span_data(&config, &completed).unwrap());
        assert_eq!(data["status"], "completed");
        assert_eq!(data["output"], "All good");
        assert_eq!(data["exit_code"], 0);
    }

    #[test]
    fn omits_sensitive_span_data_when_disabled() {
        let mut config = test_config(false);
        config.include_sensitive_span_data = false;
        let item = ThreadItem::CommandExecution {
            id: "cmd-1".into(),
            command: "pytest".into(),
            status: "completed".into(),
            aggregated_output: "All good".into(),
            exit_code: Some(0),
        };
        let data = command_span_data(&config, &item).unwrap();
        assert!(data.get("command").is_none());
        assert!(data.get("output").is_none());
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
