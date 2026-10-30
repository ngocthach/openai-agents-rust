//! Context passed to function tools. Mirrors `agents.tool_context.ToolContext`.

use std::fmt;
use std::ops::Deref;

use std::sync::Arc;

use crate::agent::Agent;
use crate::items::FunctionCall;
use crate::run_config::RunConfig;
use crate::run_context::RunContext;

/// The run context plus metadata about the tool call being executed.
pub struct ToolContext<C> {
    run_context: RunContext<C>,
    /// Name of the tool being invoked.
    pub tool_name: String,
    /// Id of the tool call.
    pub tool_call_id: String,
    /// Raw JSON arguments.
    pub tool_arguments: String,
    /// The full function call item, when invoked by a model tool call.
    pub tool_call: Option<FunctionCall>,
    /// Name of the agent invoking the tool.
    pub agent_name: String,
    /// Namespace of the tool, when it has one.
    pub tool_namespace: Option<String>,
    /// The agent invoking the tool, when the caller supplied it.
    pub agent: Option<Arc<Agent<C>>>,
    /// The active run configuration, when the caller supplied it.
    pub run_config: Option<Arc<RunConfig<C>>>,
    /// SDK-only data the tool body attached to its output item.
    custom_data: Arc<std::sync::Mutex<Option<serde_json::Value>>>,
}

impl<C> Clone for ToolContext<C> {
    fn clone(&self) -> Self {
        ToolContext {
            run_context: self.run_context.clone(),
            tool_name: self.tool_name.clone(),
            tool_call_id: self.tool_call_id.clone(),
            tool_arguments: self.tool_arguments.clone(),
            tool_call: self.tool_call.clone(),
            agent_name: self.agent_name.clone(),
            tool_namespace: self.tool_namespace.clone(),
            agent: self.agent.clone(),
            run_config: self.run_config.clone(),
            custom_data: self.custom_data.clone(),
        }
    }
}

impl<C> fmt::Debug for ToolContext<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("tool_name", &self.tool_name)
            .field("tool_call_id", &self.tool_call_id)
            .field("agent_name", &self.agent_name)
            .field("tool_namespace", &self.tool_namespace)
            .finish_non_exhaustive()
    }
}

impl<C> Deref for ToolContext<C> {
    type Target = RunContext<C>;
    fn deref(&self) -> &Self::Target {
        &self.run_context
    }
}

impl<C> ToolContext<C> {
    /// Build a tool context from a run context and a function call.
    pub fn from_call(run_context: RunContext<C>, agent_name: &str, call: &FunctionCall) -> Self {
        ToolContext {
            run_context,
            tool_name: call.name.clone(),
            tool_call_id: call.call_id.clone(),
            tool_arguments: call.arguments.clone(),
            tool_call: Some(call.clone()),
            agent_name: agent_name.to_string(),
            tool_namespace: None,
            agent: None,
            run_config: None,
            custom_data: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Build a tool context for a direct (non model-driven) invocation.
    pub fn direct(
        run_context: RunContext<C>,
        agent_name: &str,
        tool_name: &str,
        arguments: &str,
    ) -> Self {
        ToolContext {
            run_context,
            tool_name: tool_name.to_string(),
            tool_call_id: format!("call_{}", crate::util::gen_hex24()),
            tool_arguments: arguments.to_string(),
            tool_call: None,
            agent_name: agent_name.to_string(),
            tool_namespace: None,
            agent: None,
            run_config: None,
            custom_data: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// The underlying run context.
    pub fn run_context(&self) -> &RunContext<C> {
        &self.run_context
    }

    /// Attach the tool namespace.
    pub fn with_namespace(mut self, namespace: Option<String>) -> Self {
        self.tool_namespace = namespace;
        self
    }

    /// Attach the invoking agent.
    pub fn with_agent(mut self, agent: Arc<Agent<C>>) -> Self {
        self.agent = Some(agent);
        self
    }

    /// Attach the active run configuration.
    pub fn with_run_config(mut self, run_config: Arc<RunConfig<C>>) -> Self {
        self.run_config = Some(run_config);
        self
    }

    /// Attach SDK-only data to this call's output item. Later calls merge into earlier ones.
    /// Mirrors `ToolContext._custom_data`.
    pub fn set_custom_data(&self, data: serde_json::Value) {
        let mut slot = self.custom_data.lock().expect("tool context poisoned");
        match (slot.as_mut(), data) {
            (Some(serde_json::Value::Object(existing)), serde_json::Value::Object(new)) => {
                existing.extend(new);
            }
            (_, data) => *slot = Some(data),
        }
    }

    /// The SDK-only data attached with [`set_custom_data`](Self::set_custom_data).
    pub fn custom_data(&self) -> Option<serde_json::Value> {
        self.custom_data
            .lock()
            .expect("tool context poisoned")
            .clone()
    }

    /// `namespace.name` when the tool is namespaced, otherwise the bare tool name.
    /// Port of `ToolContext.qualified_tool_name`.
    pub fn qualified_tool_name(&self) -> String {
        match &self.tool_namespace {
            Some(ns) => format!("{ns}.{}", self.tool_name),
            None => self.tool_name.clone(),
        }
    }
}
