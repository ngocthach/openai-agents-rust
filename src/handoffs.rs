//! Handoffs: delegating the conversation to another agent. Mirrors `agents.handoffs`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::items::{InputItem, RunItem, RunItemKind};
use crate::models::interface::FunctionToolSpec;
use crate::run_context::RunContext;
use crate::strict_schema::ensure_strict_json_schema;
use crate::tool::{IsEnabled, ToolErrorFn};
use crate::util::{BoxFuture, transform_string_function_style_with};

/// Data passed to a handoff input filter.
#[derive(Debug, Clone, PartialEq)]
pub struct HandoffInputData {
    /// Input history before `Runner::run` started.
    pub input_history: Vec<InputItem>,
    /// Items generated before the agent turn where the handoff was invoked.
    pub pre_handoff_items: Vec<RunItem>,
    /// Items generated during the current turn, including the handoff call and output.
    pub new_items: Vec<RunItem>,
    /// Optional items to forward to the next agent instead of `new_items`.
    pub input_items: Option<Vec<RunItem>>,
}

/// A function that rewrites the input the next agent receives.
pub type HandoffInputFilter = Arc<dyn Fn(HandoffInputData) -> HandoffInputData + Send + Sync>;

/// Maps the transcript of a handoff to the input of the next agent.
pub type HandoffHistoryMapper = crate::run_config::HandoffHistoryMapper;

pub use crate::run_internal::history::{
    DEFAULT_HISTORY_END, DEFAULT_HISTORY_START, default_handoff_history_mapper,
    get_conversation_history_wrappers, reset_conversation_history_wrappers,
    set_conversation_history_wrappers,
};

/// Summarize the previous transcript for the next agent, forwarding verbatim the items the
/// summary does not already cover. Port of `agents.handoffs.history.nest_handoff_history`.
///
/// The transcript is built from `input_history`, `pre_handoff_items` and `new_items`; any
/// existing `input_items` override is ignored. Tool arguments and outputs can remain in the
/// generated summary text, so sanitize the source fields (or the returned `input_history`)
/// when they may carry sensitive data.
pub fn nest_handoff_history(
    data: HandoffInputData,
    history_mapper: Option<&HandoffHistoryMapper>,
) -> HandoffInputData {
    let pre_count = data.pre_handoff_items.len();
    let mut items = data.pre_handoff_items.clone();
    items.extend(data.new_items.clone());
    let normalized = crate::run_internal::history::normalize_handoff_items(&items, pre_count);
    if history_mapper.is_none() {
        // Record which run items the summary forwarded verbatim, so the runner knows they now
        // live in the input and must not be replayed again. Items not forwarded are summarized.
        let forwarded = items
            .iter()
            .filter(|item| !matches!(item.kind, RunItemKind::ToolApproval { .. }))
            .zip(normalized.iter())
            .filter(|(_, (_, forward))| *forward)
            .filter_map(|(item, (plain, _))| Some((item.occurrence.0?, plain.0.clone())));
        NESTED_PROVENANCE.with(|p| p.borrow_mut().extend(forwarded));
    }
    // Earlier summaries are flattened back into the items they recorded, as Python does, so a
    // custom mapper receives the transcript itself rather than a summary of a summary.
    let flattened =
        crate::run_internal::history::flatten_nested_history_messages(&data.input_history);
    let input_history = match history_mapper {
        Some(m) => {
            let mut transcript = flattened;
            transcript.extend(normalized.into_iter().map(|(item, _)| item));
            m(transcript)
        }
        None => crate::run_internal::history::build_ordered_history(flattened, normalized),
    };
    HandoffInputData {
        input_history,
        pre_handoff_items: Vec::new(),
        // The mapped history is the exact model input; `new_items` stays unchanged for the
        // session history.
        new_items: data.new_items,
        input_items: Some(Vec::new()),
    }
}

thread_local! {
    /// Occurrence keys and forwarded inputs recorded by [`nest_handoff_history`] on this thread.
    /// The runner drains it right after a handoff input filter returns, which is what ties a
    /// filter's output to the run items it kept, as Python's provenance does.
    static NESTED_PROVENANCE: std::cell::RefCell<Vec<(u64, serde_json::Value)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Start a fresh provenance record for the handoff input filter about to run.
pub(crate) fn clear_nested_provenance() {
    NESTED_PROVENANCE.with(|p| p.borrow_mut().clear());
}

/// The provenance recorded since [`clear_nested_provenance`]: `(occurrence key, forwarded input)`.
pub(crate) fn take_nested_provenance() -> Vec<(u64, serde_json::Value)> {
    NESTED_PROVENANCE.with(|p| std::mem::take(&mut *p.borrow_mut()))
}

type InvokeFn<C> =
    Arc<dyn Fn(RunContext<C>, String) -> BoxFuture<'static, Result<Arc<Agent<C>>>> + Send + Sync>;

/// A handoff to another agent, exposed to the model as a `transfer_to_<agent>` tool.
pub struct Handoff<C> {
    /// Tool name shown to the model.
    pub tool_name: String,
    /// Tool description shown to the model.
    pub tool_description: String,
    /// JSON schema of the handoff arguments.
    pub input_json_schema: Value,
    /// Name of the target agent.
    pub agent_name: String,
    /// Filter applied to the input the target agent sees.
    pub input_filter: Option<HandoffInputFilter>,
    /// Per-handoff override of `RunConfig::nest_handoff_history`.
    pub nest_handoff_history: Option<bool>,
    /// Whether the input schema is strict.
    pub strict_json_schema: bool,
    /// Enablement check.
    pub is_enabled: IsEnabled<C>,
    /// The strict-schema error of the input type, if any. The runner raises it before the first
    /// model call, because the builder cannot return it.
    pub(crate) schema_error: Option<String>,
    /// Formats the model-visible output when the handoff invocation fails.
    ///
    /// `None` propagates the error and terminates the run. When it is set, a failed
    /// invocation is not a transfer: the run records a tool output carrying the formatted
    /// text and gives the calling agent another turn, so it can retry or answer without
    /// transferring. Transfer hooks and input filters run only after a successful
    /// invocation and are not covered by this handler.
    pub failure_error_function: Option<ToolErrorFn<C>>,
    on_invoke_handoff: InvokeFn<C>,
    pub(crate) agent: Option<Arc<Agent<C>>>,
}

impl<C> Clone for Handoff<C> {
    fn clone(&self) -> Self {
        Handoff {
            tool_name: self.tool_name.clone(),
            tool_description: self.tool_description.clone(),
            input_json_schema: self.input_json_schema.clone(),
            agent_name: self.agent_name.clone(),
            input_filter: self.input_filter.clone(),
            nest_handoff_history: self.nest_handoff_history,
            strict_json_schema: self.strict_json_schema,
            is_enabled: self.is_enabled.clone(),
            schema_error: self.schema_error.clone(),
            failure_error_function: self.failure_error_function.clone(),
            on_invoke_handoff: self.on_invoke_handoff.clone(),
            agent: self.agent.clone(),
        }
    }
}

impl<C> fmt::Debug for Handoff<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handoff")
            .field("tool_name", &self.tool_name)
            .field("agent_name", &self.agent_name)
            .finish_non_exhaustive()
    }
}

/// Default tool name for a handoff to `agent_name`.
pub fn default_tool_name(agent_name: &str) -> String {
    // Whitespace alone is a normal way to name an agent, so only other invalid characters warn.
    transform_string_function_style_with(&format!("transfer_to_{agent_name}"), false)
}

/// Default tool description for a handoff.
pub fn default_tool_description(agent_name: &str, handoff_description: Option<&str>) -> String {
    format!(
        "Handoff to the {agent_name} agent to handle the request. {}",
        handoff_description.unwrap_or("")
    )
}

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false})
}

impl<C: Send + Sync + 'static> Handoff<C> {
    /// A handoff to `agent` with default name and description.
    pub fn new(agent: Arc<Agent<C>>) -> Self {
        let target = agent.clone();
        Handoff {
            tool_name: default_tool_name(&agent.name),
            tool_description: default_tool_description(
                &agent.name,
                agent.handoff_description.as_deref(),
            ),
            input_json_schema: empty_schema(),
            agent_name: agent.name.clone(),
            input_filter: None,
            nest_handoff_history: None,
            strict_json_schema: true,
            is_enabled: IsEnabled::Static(true),
            schema_error: None,
            failure_error_function: None,
            on_invoke_handoff: Arc::new(move |_ctx, _args| {
                let a = target.clone();
                Box::pin(async move { Ok(a) })
            }),
            agent: Some(agent),
        }
    }

    /// A fully custom handoff whose target agent is chosen at invocation time.
    ///
    /// Useful for cyclic agent graphs: the closure can resolve agents lazily.
    pub fn custom<F, Fut>(
        tool_name: impl Into<String>,
        tool_description: impl Into<String>,
        agent_name: impl Into<String>,
        input_json_schema: Value,
        f: F,
    ) -> Self
    where
        F: Fn(RunContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Arc<Agent<C>>>> + Send + 'static,
    {
        Handoff {
            tool_name: tool_name.into(),
            tool_description: tool_description.into(),
            input_json_schema,
            agent_name: agent_name.into(),
            input_filter: None,
            nest_handoff_history: None,
            strict_json_schema: true,
            is_enabled: IsEnabled::Static(true),
            schema_error: None,
            failure_error_function: None,
            on_invoke_handoff: Arc::new(move |c, a| Box::pin(f(c, a))),
            agent: None,
        }
    }

    /// Override the tool name.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    /// Override the tool description.
    pub fn with_tool_description(mut self, description: impl Into<String>) -> Self {
        self.tool_description = description.into();
        self
    }

    /// Run a callback when the handoff is invoked (no model-provided input).
    pub fn with_on_handoff<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let target = self.agent.clone();
        let inner = self.on_invoke_handoff.clone();
        let f = Arc::new(f);
        self.on_invoke_handoff = Arc::new(move |ctx, args| {
            let f = f.clone();
            let inner = inner.clone();
            let target = target.clone();
            Box::pin(async move {
                f(ctx.clone()).await?;
                match target {
                    Some(a) => Ok(a),
                    None => inner(ctx, args).await,
                }
            })
        });
        self
    }

    /// Run a callback receiving model-provided input of type `T` when the handoff is invoked.
    /// The schema of `T` becomes the handoff tool's parameters.
    pub fn with_on_handoff_input<T, F, Fut>(mut self, f: F) -> Self
    where
        T: JsonSchema + DeserializeOwned + Send + 'static,
        F: Fn(RunContext<C>, T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let mut schema = schemars::schema_for!(T).to_value();
        if let Value::Object(m) = &mut schema {
            m.remove("$schema");
        }
        match ensure_strict_json_schema(&schema) {
            Ok(s) => {
                self.input_json_schema = s;
                self.strict_json_schema = true;
            }
            Err(error) => {
                self.schema_error = Some(match &error {
                    AgentsError::UserError { message, .. } => message.clone(),
                    other => other.to_string(),
                });
                self.input_json_schema = schema;
                self.strict_json_schema = false;
            }
        }
        let target = self.agent.clone();
        let inner = self.on_invoke_handoff.clone();
        let tool_name = self.tool_name.clone();
        let f = Arc::new(f);
        self.on_invoke_handoff = Arc::new(move |ctx, args| {
            let f = f.clone();
            let inner = inner.clone();
            let target = target.clone();
            let tool_name = tool_name.clone();
            Box::pin(async move {
                // Handoff arguments are model and tool data. Port of `validate_json(...,
                // contains_tool_data=True)`.
                let payload = if args.trim().is_empty() { "{}" } else { &args };
                let parsed: T = serde_json::from_str(payload).map_err(|e| {
                    AgentsError::invalid_json(payload, &format!("handoff {tool_name}"), &e, true)
                })?;
                f(ctx.clone(), parsed).await?;
                match target {
                    Some(a) => Ok(a),
                    None => inner(ctx, args).await,
                }
            })
        });
        self
    }

    /// Filter the input the next agent receives.
    pub fn with_input_filter<F>(mut self, f: F) -> Self
    where
        F: Fn(HandoffInputData) -> HandoffInputData + Send + Sync + 'static,
    {
        self.input_filter = Some(Arc::new(f));
        self
    }

    /// Enable or disable the handoff.
    pub fn with_is_enabled(mut self, is_enabled: impl Into<IsEnabled<C>>) -> Self {
        self.is_enabled = is_enabled.into();
        self
    }

    /// Enable the handoff based on a dynamic check.
    pub fn with_is_enabled_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.is_enabled = IsEnabled::Dynamic(Arc::new(move |c, a| Box::pin(f(c, a))));
        self
    }

    /// Recover from a failed handoff invocation by sending `f`'s text back to the calling
    /// agent as the handoff tool's output, instead of ending the run with the error.
    ///
    /// Pass [`crate::tool::default_tool_error_function`] for the generic SDK message. A custom
    /// formatter must return only application-approved error details. Recovery does not roll
    /// back side effects an `on_handoff` callback already performed.
    pub fn with_failure_error_function<F>(mut self, f: F) -> Self
    where
        F: Fn(&RunContext<C>, &AgentsError) -> String + Send + Sync + 'static,
    {
        self.failure_error_function = Some(crate::tool::tool_error_fn(f));
        self
    }

    /// Like [`Self::with_failure_error_function`] with a handler that may await or fail
    /// (see [`crate::tool::ToolErrorFn`]). An `Err` from the handler ends the run with that
    /// error, as a Python formatter that re-raises does.
    pub fn with_failure_error_handler(mut self, f: ToolErrorFn<C>) -> Self {
        self.failure_error_function = Some(f);
        self
    }

    /// Override nested handoff history for this handoff.
    pub fn with_nest_handoff_history(mut self, nest: bool) -> Self {
        self.nest_handoff_history = Some(nest);
        self
    }

    /// Invoke the handoff and get the target agent.
    pub async fn invoke(&self, ctx: RunContext<C>, arguments: String) -> Result<Arc<Agent<C>>> {
        (self.on_invoke_handoff)(ctx, arguments).await
    }

    /// The target agent, when statically known.
    pub fn agent(&self) -> Option<&Arc<Agent<C>>> {
        self.agent.as_ref()
    }

    /// The transfer message used as the handoff tool output.
    pub fn get_transfer_message(agent_name: &str) -> String {
        json!({"assistant": agent_name}).to_string()
    }

    /// The tool spec exposed to the model.
    pub fn to_spec(&self) -> FunctionToolSpec {
        FunctionToolSpec {
            name: self.tool_name.clone(),
            description: self.tool_description.clone(),
            parameters: self.input_json_schema.clone(),
            strict: self.strict_json_schema,
            ..Default::default()
        }
    }
}

impl<C: Send + Sync + 'static> From<Arc<Agent<C>>> for Handoff<C> {
    fn from(value: Arc<Agent<C>>) -> Self {
        Handoff::new(value)
    }
}

impl<C: Send + Sync + 'static> From<Agent<C>> for Handoff<C> {
    fn from(value: Agent<C>) -> Self {
        Handoff::new(Arc::new(value))
    }
}

impl<C: Send + Sync + 'static> From<&Agent<C>> for Handoff<C> {
    fn from(value: &Agent<C>) -> Self {
        Handoff::new(Arc::new(value.clone()))
    }
}

/// Create a handoff to `agent`. Shorthand for [`Handoff::new`].
pub fn handoff<C: Send + Sync + 'static>(agent: impl Into<Arc<Agent<C>>>) -> Handoff<C> {
    Handoff::new(agent.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_naming_and_invocation() {
        let billing =
            Arc::new(Agent::<()>::new("Billing agent").with_handoff_description("Billing"));
        let h = handoff(billing);
        assert_eq!(h.tool_name, "transfer_to_billing_agent");
        assert!(h.tool_description.contains("Billing agent agent"));
        let a = h.invoke(RunContext::new(()), "{}".into()).await.unwrap();
        assert_eq!(a.name, "Billing agent");
        assert_eq!(
            Handoff::<()>::get_transfer_message("x"),
            r#"{"assistant":"x"}"#
        );
    }

    #[derive(serde::Deserialize, JsonSchema)]
    struct Escalation {
        reason: String,
    }

    #[tokio::test]
    async fn handoff_with_input() {
        let seen = Arc::new(std::sync::Mutex::new(String::new()));
        let s2 = seen.clone();
        let h = handoff(Arc::new(Agent::<()>::new("Escalation"))).with_on_handoff_input(
            move |_ctx, data: Escalation| {
                let s2 = s2.clone();
                async move {
                    *s2.lock().unwrap() = data.reason;
                    Ok(())
                }
            },
        );
        assert_eq!(h.input_json_schema["required"], json!(["reason"]));
        h.invoke(RunContext::new(()), r#"{"reason":"angry"}"#.into())
            .await
            .unwrap();
        assert_eq!(*seen.lock().unwrap(), "angry");
        assert!(h.invoke(RunContext::new(()), "nope".into()).await.is_err());
    }
}
