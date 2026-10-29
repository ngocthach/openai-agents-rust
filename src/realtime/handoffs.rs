//! Handoffs between realtime agents. Mirrors `agents.realtime.handoffs`.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::error::{AgentsError, Result};
use crate::handoffs::{default_tool_description, default_tool_name};
use crate::realtime::agent::RealtimeAgent;
use crate::realtime::config::RealtimeFunctionToolSpec;
use crate::run_context::RunContext;
use crate::strict_schema::ensure_strict_json_schema;
use crate::tool::IsEnabled;
use crate::util::BoxFuture;

type InvokeFn<C> = Arc<
    dyn Fn(RunContext<C>, String) -> BoxFuture<'static, Result<Arc<RealtimeAgent<C>>>>
        + Send
        + Sync,
>;

/// A handoff to another [`RealtimeAgent`], exposed to the model as a `transfer_to_<agent>` tool.
///
/// Unlike [`crate::Handoff`], realtime handoffs target realtime agents and do not support input
/// filters: the realtime session keeps a single, server-side conversation.
pub struct RealtimeHandoff<C> {
    /// Tool name shown to the model.
    pub tool_name: String,
    /// Tool description shown to the model.
    pub tool_description: String,
    /// JSON schema of the handoff arguments.
    pub input_json_schema: Value,
    /// Name of the target agent.
    pub agent_name: String,
    /// Enablement check.
    pub is_enabled: IsEnabled<C>,
    on_invoke_handoff: InvokeFn<C>,
    agent: Option<Arc<RealtimeAgent<C>>>,
}

impl<C> Clone for RealtimeHandoff<C> {
    fn clone(&self) -> Self {
        RealtimeHandoff {
            tool_name: self.tool_name.clone(),
            tool_description: self.tool_description.clone(),
            input_json_schema: self.input_json_schema.clone(),
            agent_name: self.agent_name.clone(),
            is_enabled: self.is_enabled.clone(),
            on_invoke_handoff: self.on_invoke_handoff.clone(),
            agent: self.agent.clone(),
        }
    }
}

impl<C> fmt::Debug for RealtimeHandoff<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeHandoff")
            .field("tool_name", &self.tool_name)
            .field("agent_name", &self.agent_name)
            .finish_non_exhaustive()
    }
}

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false})
}

/// Create a handoff to `agent` with the default tool name and description.
pub fn realtime_handoff<C: Send + Sync + 'static>(
    agent: Arc<RealtimeAgent<C>>,
) -> RealtimeHandoff<C> {
    RealtimeHandoff::new(agent)
}

impl<C: Send + Sync + 'static> RealtimeHandoff<C> {
    /// A handoff to `agent` with the default tool name and description.
    pub fn new(agent: Arc<RealtimeAgent<C>>) -> Self {
        let target = agent.clone();
        RealtimeHandoff {
            tool_name: default_tool_name(&agent.name),
            tool_description: default_tool_description(
                &agent.name,
                agent.handoff_description.as_deref(),
            ),
            input_json_schema: empty_schema(),
            agent_name: agent.name.clone(),
            is_enabled: IsEnabled::Static(true),
            on_invoke_handoff: Arc::new(move |_ctx, _args| {
                let agent = target.clone();
                Box::pin(async move { Ok(agent) })
            }),
            agent: Some(agent),
        }
    }

    /// A handoff whose target is resolved when the model invokes it.
    pub fn custom<F, Fut>(
        tool_name: impl Into<String>,
        tool_description: impl Into<String>,
        agent_name: impl Into<String>,
        input_json_schema: Value,
        f: F,
    ) -> Self
    where
        F: Fn(RunContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Arc<RealtimeAgent<C>>>> + Send + 'static,
    {
        RealtimeHandoff {
            tool_name: tool_name.into(),
            tool_description: tool_description.into(),
            input_json_schema,
            agent_name: agent_name.into(),
            is_enabled: IsEnabled::Static(true),
            on_invoke_handoff: Arc::new(move |ctx, args| Box::pin(f(ctx, args))),
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

    /// Run a callback when the handoff is invoked, without model-provided input.
    pub fn with_on_handoff<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let Some(agent) = self.agent.clone() else {
            return self;
        };
        let f = Arc::new(f);
        self.on_invoke_handoff = Arc::new(move |ctx, _args| {
            let f = f.clone();
            let agent = agent.clone();
            Box::pin(async move {
                f(ctx).await?;
                Ok(agent)
            })
        });
        self
    }

    /// Run a callback with typed, model-provided input when the handoff is invoked.
    ///
    /// `T`'s JSON schema becomes the handoff's input schema.
    pub fn with_on_handoff_input<T, F, Fut>(mut self, f: F) -> Self
    where
        T: JsonSchema + DeserializeOwned + Send + 'static,
        F: Fn(RunContext<C>, T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let Some(agent) = self.agent.clone() else {
            return self;
        };
        let mut schema = schemars::schema_for!(T).to_value();
        if let Value::Object(map) = &mut schema {
            map.remove("$schema");
            map.remove("title");
        }
        self.input_json_schema = ensure_strict_json_schema(&schema).unwrap_or(schema);

        let f = Arc::new(f);
        let tool_name = self.tool_name.clone();
        self.on_invoke_handoff = Arc::new(move |ctx, args| {
            let f = f.clone();
            let agent = agent.clone();
            let tool_name = tool_name.clone();
            Box::pin(async move {
                let parsed: T = serde_json::from_str(if args.trim().is_empty() {
                    "{}"
                } else {
                    args.as_str()
                })
                .map_err(|e| {
                    AgentsError::model_behavior(format!(
                        "Invalid JSON input for handoff {tool_name}: {e}"
                    ))
                })?;
                f(ctx, parsed).await?;
                Ok(agent)
            })
        });
        self
    }

    /// Set a static or dynamic enablement check.
    pub fn with_is_enabled(mut self, is_enabled: impl Into<IsEnabled<C>>) -> Self {
        self.is_enabled = is_enabled.into();
        self
    }

    /// Set a dynamic enablement check.
    pub fn with_is_enabled_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.is_enabled = IsEnabled::Dynamic(Arc::new(move |ctx, agent| Box::pin(f(ctx, agent))));
        self
    }
}

impl<C> RealtimeHandoff<C> {
    /// Invoke the handoff and get the agent that takes over.
    pub async fn invoke(
        &self,
        context: RunContext<C>,
        arguments: String,
    ) -> Result<Arc<RealtimeAgent<C>>> {
        (self.on_invoke_handoff)(context, arguments).await
    }

    /// The target agent, when statically known.
    pub fn agent(&self) -> Option<&Arc<RealtimeAgent<C>>> {
        self.agent.as_ref()
    }

    /// The model-visible output sent when the handoff completes.
    pub fn transfer_message(agent_name: &str) -> String {
        json!({ "assistant": agent_name }).to_string()
    }

    /// The tool spec exposed to the realtime model.
    pub fn to_spec(&self) -> RealtimeFunctionToolSpec {
        RealtimeFunctionToolSpec {
            name: self.tool_name.clone(),
            description: self.tool_description.clone(),
            parameters: self.input_json_schema.clone(),
        }
    }
}

impl<C: Send + Sync + 'static> From<Arc<RealtimeAgent<C>>> for RealtimeHandoff<C> {
    fn from(value: Arc<RealtimeAgent<C>>) -> Self {
        RealtimeHandoff::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_naming_matches_core_handoffs() {
        let agent = RealtimeAgent::<()>::new("Billing Agent").into_arc();
        let handoff = realtime_handoff(agent);
        assert_eq!(handoff.tool_name, "transfer_to_Billing_Agent");
        assert_eq!(handoff.agent_name, "Billing Agent");
    }

    #[test]
    fn transfer_message_matches_core_format() {
        assert_eq!(
            RealtimeHandoff::<()>::transfer_message("x"),
            r#"{"assistant":"x"}"#
        );
    }
}
