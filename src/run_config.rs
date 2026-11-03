//! Per-run configuration. Mirrors `agents.run_config`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::agent::{Agent, AgentModel};
use crate::error::Result;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoffs::{HandoffInputData, HandoffInputFilter};
use crate::items::InputItem;
use crate::memory::SessionSettings;
use crate::model_settings::ModelSettings;
use crate::models::interface::ModelProvider;
use crate::models::multi_provider::MultiProvider;
use crate::run_context::RunContext;
use crate::util::BoxFuture;

/// The model input right before a model call.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInputData {
    pub input: Vec<InputItem>,
    pub instructions: Option<String>,
}

/// Data passed to `RunConfig::call_model_input_filter`.
pub struct CallModelData<C> {
    pub model_data: ModelInputData,
    pub agent: Arc<Agent<C>>,
    pub context: RunContext<C>,
}

impl<C> fmt::Debug for CallModelData<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallModelData")
            .field("model_data", &self.model_data)
            .field("agent", &self.agent.name)
            .finish()
    }
}

/// Category of a model-visible tool error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolErrorKind {
    ApprovalRejected,
    ToolNotFound,
    /// The tool raised and the default failure handler produced the model-visible message.
    ToolException,
}

/// Arguments for `RunConfig::tool_error_formatter`.
///
/// `error` is only set for [`ToolErrorKind::ToolException`] and can carry sensitive data;
/// return only application-approved details from a formatter.
pub struct ToolErrorFormatterArgs<'a, C> {
    pub kind: ToolErrorKind,
    /// Tool runtime (`function`, `mcp`, ...).
    pub tool_type: String,
    pub tool_name: String,
    pub call_id: String,
    pub default_message: String,
    /// The active run context.
    pub run_context: &'a RunContext<C>,
    /// The original error, for [`ToolErrorKind::ToolException`].
    pub error: Option<&'a crate::error::AgentsError>,
}

/// Formats the model-visible message for a tool error the SDK answers itself.
///
/// Port of Python's `ToolErrorFormatter`: it may await and it may fail. `Ok(None)` keeps the SDK
/// default message, and so does an `Err`, which is logged (the detail only when the tool-data
/// logging policy allows it) and never ends the run: Python catches whatever a formatter raises.
/// Build one with [`tool_error_formatter_fn`] (plain closure), [`try_tool_error_formatter_fn`]
/// (plain closure that may fail) or [`async_tool_error_formatter_fn`] (async).
pub type ToolErrorFormatter<C> = Arc<
    dyn for<'a> Fn(&'a ToolErrorFormatterArgs<'a, C>) -> BoxFuture<'a, Result<Option<String>>>
        + Send
        + Sync,
>;

/// Wrap a synchronous closure that returns the message (or `None`) as a [`ToolErrorFormatter`].
pub fn tool_error_formatter_fn<C, F>(f: F) -> ToolErrorFormatter<C>
where
    F: for<'a> Fn(&ToolErrorFormatterArgs<'a, C>) -> Option<String> + Send + Sync + 'static,
{
    async_tool_error_formatter_fn(move |args| Box::pin(std::future::ready(Ok(f(args)))))
}

/// Wrap a synchronous closure that may fail as a [`ToolErrorFormatter`]. An `Err` is logged and
/// the default message is used.
pub fn try_tool_error_formatter_fn<C, F>(f: F) -> ToolErrorFormatter<C>
where
    F: for<'a> Fn(&ToolErrorFormatterArgs<'a, C>) -> Result<Option<String>> + Send + Sync + 'static,
{
    async_tool_error_formatter_fn(move |args| Box::pin(std::future::ready(f(args))))
}

/// Wrap an async closure as a [`ToolErrorFormatter`]. The closure returns a boxed future, usually
/// `Box::pin(async move { ... })`.
pub fn async_tool_error_formatter_fn<C, F>(f: F) -> ToolErrorFormatter<C>
where
    F: for<'a> Fn(&'a ToolErrorFormatterArgs<'a, C>) -> BoxFuture<'a, Result<Option<String>>>
        + Send
        + Sync
        + 'static,
{
    Arc::new(f)
}

/// Ask `formatter` for the message of `args`, falling back to `None` (the caller's default) when
/// it fails. Port of the `try`/`except Exception` around Python's formatter call.
pub(crate) async fn apply_tool_error_formatter<C>(
    formatter: &ToolErrorFormatter<C>,
    args: &ToolErrorFormatterArgs<'_, C>,
) -> Option<String> {
    match formatter(args).await {
        Ok(message) => message,
        Err(error) => {
            match args.kind {
                ToolErrorKind::ToolException => {
                    crate::logger::log_tool_action_warning("Tool error formatter failed", &error)
                }
                ToolErrorKind::ApprovalRejected => {
                    crate::logger::log_tool_action_error("Tool error formatter failed", &error)
                }
                ToolErrorKind::ToolNotFound => crate::logger::log_tool_action_error(
                    "Tool error formatter failed for missing tool",
                    &error,
                ),
            }
            None
        }
    }
}

impl<C> fmt::Debug for ToolErrorFormatterArgs<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolErrorFormatterArgs")
            .field("kind", &self.kind)
            .field("tool_type", &self.tool_type)
            .field("tool_name", &self.tool_name)
            .field("call_id", &self.call_id)
            .field("default_message", &self.default_message)
            .finish_non_exhaustive()
    }
}

/// SDK-side execution settings for local function tools.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolExecutionConfig {
    /// Maximum number of function tool calls run concurrently (`None` = all at once).
    pub max_function_tool_concurrency: Option<usize>,
    /// Run tool input guardrails before emitting an approval interruption.
    pub pre_approval_tool_input_guardrails: bool,
}

/// What to do when the model calls a tool that does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolNotFoundBehavior {
    /// Raise `ModelBehaviorError` (default).
    #[default]
    RaiseError,
    /// Send an error output to the model and continue.
    ReturnErrorToModel,
}

/// What to do when a function tool and a handoff (or two tools) expose the same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolNameCollisionPolicy {
    /// Log a warning and expose only the dispatch winner (the last handoff, else the last
    /// tool) (default).
    #[default]
    Warn,
    /// Raise `UserError` before the model is called.
    Error,
}

/// Default model-visible placeholder for a terminal tool output blocked by an output guardrail.
pub const OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT: &str = "Output withheld by an output guardrail.";

/// Data passed to an output guardrail blocked-message formatter. It never includes the
/// rejected output or the guardrail's `output_info`.
pub struct OutputGuardrailBlockedMessageArgs<'a, C> {
    /// The SDK default placeholder.
    pub default_message: String,
    /// The guardrail that tripped.
    pub guardrail_name: String,
    /// The agent whose output was blocked.
    pub agent: &'a Arc<Agent<C>>,
    /// The active run context.
    pub run_context: &'a RunContext<C>,
}

impl<C> OutputGuardrailBlockedMessageArgs<'_, C> {
    /// Name of the agent whose output was blocked.
    pub fn agent_name(&self) -> &str {
        &self.agent.name
    }
}

impl<C> fmt::Debug for OutputGuardrailBlockedMessageArgs<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputGuardrailBlockedMessageArgs")
            .field("default_message", &self.default_message)
            .field("guardrail_name", &self.guardrail_name)
            .field("agent", &self.agent.name)
            .finish_non_exhaustive()
    }
}

/// Formats the blocked-output placeholder.
///
/// Keep it synchronous: it runs after a terminal tool output is rejected but before every
/// replay and persistence owner is rebuilt with the data-free replacement.
pub type BlockedMessageFormatter<C> =
    Arc<dyn for<'a> Fn(&OutputGuardrailBlockedMessageArgs<'a, C>) -> Option<String> + Send + Sync>;

/// The placeholder kept in history for a terminal tool output blocked by an output guardrail.
pub enum OutputGuardrailBlockedMessage<C> {
    /// A fixed non-empty message.
    Text(String),
    /// A formatter; returning `None` or an empty string uses the default.
    Formatter(BlockedMessageFormatter<C>),
}

impl<C> Clone for OutputGuardrailBlockedMessage<C> {
    fn clone(&self) -> Self {
        match self {
            OutputGuardrailBlockedMessage::Text(t) => {
                OutputGuardrailBlockedMessage::Text(t.clone())
            }
            OutputGuardrailBlockedMessage::Formatter(f) => {
                OutputGuardrailBlockedMessage::Formatter(f.clone())
            }
        }
    }
}

impl<C> fmt::Debug for OutputGuardrailBlockedMessage<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputGuardrailBlockedMessage::Text(t) => f.debug_tuple("Text").field(t).finish(),
            OutputGuardrailBlockedMessage::Formatter(_) => f.write_str("Formatter(..)"),
        }
    }
}

/// How reasoning item ids are handled when building next-turn input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningItemIdPolicy {
    #[default]
    Preserve,
    Omit,
}

type SessionInputCallback = Arc<
    dyn Fn(Vec<InputItem>, Vec<InputItem>) -> BoxFuture<'static, Result<Vec<InputItem>>>
        + Send
        + Sync,
>;
type CallModelInputFilter<C> =
    Arc<dyn Fn(CallModelData<C>) -> BoxFuture<'static, Result<ModelInputData>> + Send + Sync>;
/// Maps the normalized transcript to the input of the next agent when nesting handoff history.
pub type HandoffHistoryMapper = Arc<dyn Fn(Vec<InputItem>) -> Vec<InputItem> + Send + Sync>;

/// Settings for a single run, overriding agent-level configuration.
pub struct RunConfig<C> {
    /// Model used for every agent in the run.
    pub model: Option<AgentModel>,
    /// Function tool / handoff name collision handling.
    pub tool_name_collision_policy: ToolNameCollisionPolicy,
    /// Placeholder kept for a terminal tool output blocked by an output guardrail.
    pub output_guardrail_blocked_message: Option<OutputGuardrailBlockedMessage<C>>,
    /// Resolves model names (default: [`MultiProvider`]).
    pub model_provider: Arc<dyn ModelProvider>,
    /// Settings overlaid on each agent's settings.
    pub model_settings: Option<ModelSettings>,
    /// Filter applied to every handoff without its own filter.
    pub handoff_input_filter: Option<HandoffInputFilter>,
    /// Collapse prior history into a summary message on handoff (beta).
    pub nest_handoff_history: bool,
    /// Custom mapper for nested handoff history.
    pub handoff_history_mapper: Option<HandoffHistoryMapper>,
    /// Input guardrails applied to every run.
    pub input_guardrails: Vec<InputGuardrail<C>>,
    /// Output guardrails applied to every run.
    pub output_guardrails: Vec<OutputGuardrail<C>>,
    /// Disable tracing for this run.
    pub tracing_disabled: bool,
    /// API key used to export this run's traces.
    pub tracing_api_key: Option<String>,
    /// Include model and tool inputs/outputs in traces.
    pub trace_include_sensitive_data: bool,
    /// Emit task and turn spans.
    pub include_task_and_turn_spans: bool,
    /// Trace workflow name.
    pub workflow_name: String,
    /// Trace id.
    pub trace_id: Option<String>,
    /// Trace group id.
    pub group_id: Option<String>,
    /// Trace metadata.
    pub trace_metadata: Option<Map<String, Value>>,
    /// Merge session history with new input.
    pub session_input_callback: Option<SessionInputCallback>,
    /// Edit model input right before the call.
    pub call_model_input_filter: Option<CallModelInputFilter<C>>,
    /// Customize model-visible tool error messages.
    pub tool_error_formatter: Option<ToolErrorFormatter<C>>,
    /// Session retrieval settings.
    pub session_settings: Option<SessionSettings>,
    /// Reasoning item id handling.
    pub reasoning_item_id_policy: ReasoningItemIdPolicy,
    /// Local tool execution settings.
    pub tool_execution: ToolExecutionConfig,
    /// How long a tool that already ran keeps finalizing (output guardrails, custom data, end
    /// hooks) after a sibling tool failed, before it is abandoned. `None` waits 100 ms, like
    /// the Python SDK's post-invoke wait.
    pub tool_post_invoke_wait: Option<std::time::Duration>,
    /// Unknown tool handling.
    pub tool_not_found_behavior: ToolNotFoundBehavior,
}

impl<C> Clone for RunConfig<C> {
    fn clone(&self) -> Self {
        RunConfig {
            model: self.model.clone(),
            model_provider: self.model_provider.clone(),
            model_settings: self.model_settings.clone(),
            handoff_input_filter: self.handoff_input_filter.clone(),
            nest_handoff_history: self.nest_handoff_history,
            handoff_history_mapper: self.handoff_history_mapper.clone(),
            input_guardrails: self.input_guardrails.clone(),
            output_guardrails: self.output_guardrails.clone(),
            tracing_disabled: self.tracing_disabled,
            tracing_api_key: self.tracing_api_key.clone(),
            trace_include_sensitive_data: self.trace_include_sensitive_data,
            include_task_and_turn_spans: self.include_task_and_turn_spans,
            workflow_name: self.workflow_name.clone(),
            trace_id: self.trace_id.clone(),
            group_id: self.group_id.clone(),
            trace_metadata: self.trace_metadata.clone(),
            session_input_callback: self.session_input_callback.clone(),
            call_model_input_filter: self.call_model_input_filter.clone(),
            tool_error_formatter: self.tool_error_formatter.clone(),
            session_settings: self.session_settings.clone(),
            reasoning_item_id_policy: self.reasoning_item_id_policy,
            tool_execution: self.tool_execution.clone(),
            tool_post_invoke_wait: self.tool_post_invoke_wait,
            tool_not_found_behavior: self.tool_not_found_behavior,
            tool_name_collision_policy: self.tool_name_collision_policy,
            output_guardrail_blocked_message: self.output_guardrail_blocked_message.clone(),
        }
    }
}

impl<C> fmt::Debug for RunConfig<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunConfig")
            .field("model", &self.model)
            .field("model_settings", &self.model_settings)
            .field("workflow_name", &self.workflow_name)
            .field("tracing_disabled", &self.tracing_disabled)
            .finish_non_exhaustive()
    }
}

fn env_include_sensitive() -> bool {
    match std::env::var("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA") {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => true,
    }
}

impl<C> Default for RunConfig<C> {
    fn default() -> Self {
        RunConfig {
            model: None,
            model_provider: Arc::new(MultiProvider::default().into_run_owned()),
            model_settings: None,
            handoff_input_filter: None,
            nest_handoff_history: false,
            handoff_history_mapper: None,
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            tracing_disabled: false,
            tracing_api_key: None,
            trace_include_sensitive_data: env_include_sensitive(),
            include_task_and_turn_spans: true,
            workflow_name: "Agent workflow".to_string(),
            trace_id: None,
            group_id: None,
            trace_metadata: None,
            session_input_callback: None,
            call_model_input_filter: None,
            tool_error_formatter: None,
            session_settings: None,
            reasoning_item_id_policy: ReasoningItemIdPolicy::Preserve,
            tool_execution: ToolExecutionConfig::default(),
            tool_post_invoke_wait: None,
            tool_not_found_behavior: ToolNotFoundBehavior::RaiseError,
            tool_name_collision_policy: ToolNameCollisionPolicy::Warn,
            output_guardrail_blocked_message: None,
        }
    }
}

impl<C: Send + Sync + 'static> RunConfig<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Use this model for every agent.
    pub fn with_model(mut self, model: impl Into<AgentModel>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Use this model provider.
    pub fn with_model_provider(mut self, provider: Arc<dyn ModelProvider>) -> Self {
        self.model_provider = provider;
        self
    }

    /// Overlay these model settings.
    pub fn with_model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = Some(settings);
        self
    }

    /// Set the trace workflow name.
    pub fn with_workflow_name(mut self, name: impl Into<String>) -> Self {
        self.workflow_name = name.into();
        self
    }

    /// Set the trace group id.
    pub fn with_group_id(mut self, id: impl Into<String>) -> Self {
        self.group_id = Some(id.into());
        self
    }

    /// Disable tracing for this run.
    pub fn with_tracing_disabled(mut self, disabled: bool) -> Self {
        self.tracing_disabled = disabled;
        self
    }

    /// Set whether sensitive data is included in traces.
    pub fn with_trace_include_sensitive_data(mut self, include: bool) -> Self {
        self.trace_include_sensitive_data = include;
        self
    }

    /// Add a run-wide input guardrail.
    pub fn with_input_guardrail(mut self, g: InputGuardrail<C>) -> Self {
        self.input_guardrails.push(g);
        self
    }

    /// Add a run-wide output guardrail.
    pub fn with_output_guardrail(mut self, g: OutputGuardrail<C>) -> Self {
        self.output_guardrails.push(g);
        self
    }

    /// Set a global handoff input filter.
    pub fn with_handoff_input_filter<F>(mut self, f: F) -> Self
    where
        F: Fn(HandoffInputData) -> HandoffInputData + Send + Sync + 'static,
    {
        self.handoff_input_filter = Some(crate::handoffs::handoff_input_filter(f));
        self
    }

    /// Set a global [`HandoffInputFilter`], which may await and may fail (an `Err` ends the run).
    pub fn with_handoff_input_filter_handler(mut self, filter: HandoffInputFilter) -> Self {
        self.handoff_input_filter = Some(filter);
        self
    }

    /// Enable nested handoff history (beta).
    pub fn with_nest_handoff_history(mut self, nest: bool) -> Self {
        self.nest_handoff_history = nest;
        self
    }

    /// Customize how session history and new input are merged.
    pub fn with_session_input_callback<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Vec<InputItem>, Vec<InputItem>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<InputItem>>> + Send + 'static,
    {
        self.session_input_callback = Some(Arc::new(move |h, n| Box::pin(f(h, n))));
        self
    }

    /// Edit model input right before each call.
    pub fn with_call_model_input_filter<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(CallModelData<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ModelInputData>> + Send + 'static,
    {
        self.call_model_input_filter = Some(Arc::new(move |d| Box::pin(f(d))));
        self
    }

    /// Customize model-visible tool error messages.
    pub fn with_tool_error_formatter<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&ToolErrorFormatterArgs<'a, C>) -> Option<String> + Send + Sync + 'static,
    {
        self.tool_error_formatter = Some(tool_error_formatter_fn(f));
        self
    }

    /// Customize model-visible tool error messages with a [`ToolErrorFormatter`], which may
    /// await and may fail (a failure keeps the default message).
    pub fn with_tool_error_formatter_handler(mut self, formatter: ToolErrorFormatter<C>) -> Self {
        self.tool_error_formatter = Some(formatter);
        self
    }

    /// Set session retrieval settings.
    pub fn with_session_settings(mut self, settings: SessionSettings) -> Self {
        self.session_settings = Some(settings);
        self
    }

    /// Set local tool execution settings.
    pub fn with_tool_execution(mut self, config: ToolExecutionConfig) -> Self {
        self.tool_execution = config;
        self
    }

    /// Set how long finalizing tools may outlive a failed sibling tool.
    pub fn with_tool_post_invoke_wait(mut self, wait: std::time::Duration) -> Self {
        self.tool_post_invoke_wait = Some(wait);
        self
    }

    /// Set unknown tool handling.
    pub fn with_tool_not_found_behavior(mut self, behavior: ToolNotFoundBehavior) -> Self {
        self.tool_not_found_behavior = behavior;
        self
    }

    /// Set the tool name collision policy.
    pub fn with_tool_name_collision_policy(mut self, policy: ToolNameCollisionPolicy) -> Self {
        self.tool_name_collision_policy = policy;
        self
    }

    /// Set the placeholder kept for terminal tool outputs blocked by an output guardrail.
    pub fn with_output_guardrail_blocked_message(
        mut self,
        message: OutputGuardrailBlockedMessage<C>,
    ) -> Self {
        self.output_guardrail_blocked_message = Some(message);
        self
    }

    /// Set the reasoning item id policy.
    pub fn with_reasoning_item_id_policy(mut self, policy: ReasoningItemIdPolicy) -> Self {
        self.reasoning_item_id_policy = policy;
        self
    }
}
