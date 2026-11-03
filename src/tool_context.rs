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
    /// The `_meta` of the MCP call result, when this tool is an MCP tool. Read by the
    /// custom-data extractor as `result_meta`.
    result_meta: Arc<std::sync::Mutex<Option<serde_json::Value>>>,
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
            result_meta: self.result_meta.clone(),
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

/// The name a model tool call is traced and tracked under: `namespace.name`, or the bare name
/// when the call has no namespace or its namespace is the synthetic one of a deferred top-level
/// tool. Port of `get_tool_call_trace_name`.
pub(crate) fn call_trace_name(call: &FunctionCall) -> String {
    match call
        .extra
        .get("namespace")
        .and_then(serde_json::Value::as_str)
        .filter(|ns| !ns.is_empty())
    {
        Some(ns) if ns != call.name => format!("{ns}.{}", call.name),
        _ => call.name.clone(),
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
            // Python infers the namespace from the model's tool call.
            tool_namespace: call
                .extra
                .get("namespace")
                .and_then(serde_json::Value::as_str)
                .filter(|ns| !ns.is_empty())
                .map(str::to_string),
            agent: None,
            run_config: None,
            custom_data: Arc::new(std::sync::Mutex::new(None)),
            result_meta: Arc::new(std::sync::Mutex::new(None)),
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
            result_meta: Arc::new(std::sync::Mutex::new(None)),
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

    /// Record the `_meta` of the MCP call result for this invocation (shared by clones).
    #[cfg(feature = "mcp")]
    pub(crate) fn set_result_meta(&self, meta: Option<serde_json::Value>) {
        *self.result_meta.lock().expect("tool context poisoned") = meta;
    }

    /// The `_meta` of the MCP call result, or `None` for a tool that is not an MCP tool.
    /// Mirrors `FunctionToolCustomDataContext.result_meta`.
    pub fn result_meta(&self) -> Option<serde_json::Value> {
        self.result_meta
            .lock()
            .expect("tool context poisoned")
            .clone()
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
    /// A namespace equal to the tool name is the synthetic namespace of a deferred top-level
    /// tool and is not shown.
    pub fn qualified_tool_name(&self) -> String {
        match &self.tool_namespace {
            Some(ns) if ns != &self.tool_name => format!("{ns}.{}", self.tool_name),
            _ => self.tool_name.clone(),
        }
    }
}
