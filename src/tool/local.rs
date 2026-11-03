//! Tools executed on the local machine (shell, apply patch, computer, custom) and Responses-only
//! hosted helpers (tool search, programmatic tool calling).

use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::computer::Computer;
use crate::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchStatus};
use crate::error::{AgentsError, Result};
use crate::items::ToolApprovalItem;
use crate::run_context::RunContext;
use crate::tool_context::ToolContext;
use crate::util::BoxFuture;

// ---------------------------------------------------------------------------------------------
// Custom data extractors
// ---------------------------------------------------------------------------------------------

/// Context passed to a computer-tool custom data extractor.
/// Mirrors `agents.tool.ComputerToolCustomDataContext`.
pub struct ComputerToolCustomDataContext<'a, C> {
    /// The current run context.
    pub run_context: &'a RunContext<C>,
    /// The computer tool that was invoked.
    pub tool: &'a ComputerTool<C>,
    /// The raw computer tool call produced by the model.
    pub tool_call: &'a Value,
    /// The screenshot data URL returned to the model.
    pub output: &'a str,
    /// The raw computer call output item that will be replayed to the model.
    pub raw_item: &'a Value,
}

impl<C> fmt::Debug for ComputerToolCustomDataContext<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputerToolCustomDataContext")
            .field("raw_item", &self.raw_item)
            .finish_non_exhaustive()
    }
}

/// Produces SDK-only data to attach to a computer call output item.
/// Mirrors `ComputerToolCustomDataExtractor`.
pub type ComputerToolCustomDataExtractor<C> = Arc<
    dyn for<'a> Fn(&ComputerToolCustomDataContext<'a, C>) -> BoxFuture<'static, Option<Value>>
        + Send
        + Sync,
>;

/// Context passed to an apply-patch custom data extractor.
/// Mirrors `agents.tool.ApplyPatchToolCustomDataContext`.
pub struct ApplyPatchToolCustomDataContext<'a, C> {
    /// The current run context.
    pub run_context: &'a RunContext<C>,
    /// The `apply_patch` tool that was invoked.
    pub tool: &'a ApplyPatchTool<C>,
    /// The patch operations requested by the model.
    pub operations: &'a [ApplyPatchOperation],
    /// The model-visible `apply_patch` output.
    pub output: &'a str,
    /// The serialized `apply_patch` output status.
    pub status: ApplyPatchStatus,
    /// The raw `apply_patch` output item that will be replayed to the model.
    pub raw_item: &'a Value,
}

impl<C> fmt::Debug for ApplyPatchToolCustomDataContext<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApplyPatchToolCustomDataContext")
            .field("status", &self.status)
            .field("raw_item", &self.raw_item)
            .finish_non_exhaustive()
    }
}

/// Produces SDK-only data to attach to an `apply_patch` output item.
/// Mirrors `ApplyPatchToolCustomDataExtractor`.
pub type ApplyPatchToolCustomDataExtractor<C> = Arc<
    dyn for<'a> Fn(&ApplyPatchToolCustomDataContext<'a, C>) -> BoxFuture<'static, Option<Value>>
        + Send
        + Sync,
>;

/// Context passed to a custom-tool custom data extractor.
/// Mirrors `agents.tool.CustomToolCustomDataContext`.
pub struct CustomToolCustomDataContext<'a, C> {
    /// The tool invocation context.
    pub tool_context: &'a ToolContext<C>,
    /// The custom tool that was invoked.
    pub tool: &'a CustomTool<C>,
    /// The raw model-provided custom tool input.
    pub input: &'a str,
    /// The model-visible custom tool output.
    pub output: &'a str,
    /// The raw custom tool output item that will be replayed to the model.
    pub raw_item: &'a Value,
}

impl<C> fmt::Debug for CustomToolCustomDataContext<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomToolCustomDataContext")
            .field("tool", &self.tool.name)
            .field("raw_item", &self.raw_item)
            .finish_non_exhaustive()
    }
}

/// Produces SDK-only data to attach to a custom tool output item.
/// Mirrors `CustomToolCustomDataExtractor`.
pub type CustomToolCustomDataExtractor<C> = Arc<
    dyn for<'a> Fn(&CustomToolCustomDataContext<'a, C>) -> BoxFuture<'static, Option<Value>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------------------------
// Approvals
// ---------------------------------------------------------------------------------------------

/// Per-call approval rule: `(context, payload, call id) -> needs approval`.
pub type ApprovalPolicyFn<C, T> =
    Arc<dyn Fn(RunContext<C>, T, String) -> BoxFuture<'static, bool> + Send + Sync>;

/// [`ApprovalPolicyFn`] that also receives the name of the agent running the call:
/// `(context, agent name, payload, call id) -> needs approval`. A rule that consults the run's
/// recorded decisions (see [`RunContext::is_tool_approved`]) needs the agent they are filed under.
pub type ApprovalPolicyForAgentFn<C, T> =
    Arc<dyn Fn(RunContext<C>, String, T, String) -> BoxFuture<'static, bool> + Send + Sync>;

/// Whether a local tool call needs human approval. `T` is the payload the rule inspects
/// (the shell action, the patch operation or the custom tool input).
pub enum ApprovalPolicy<C, T> {
    /// Always (true) or never (false).
    Static(bool),
    /// Decided per call.
    Dynamic(ApprovalPolicyFn<C, T>),
    /// Decided per call by a rule that also learns which agent runs it.
    DynamicForAgent(ApprovalPolicyForAgentFn<C, T>),
}

impl<C, T> Clone for ApprovalPolicy<C, T> {
    fn clone(&self) -> Self {
        match self {
            ApprovalPolicy::Static(b) => ApprovalPolicy::Static(*b),
            ApprovalPolicy::Dynamic(f) => ApprovalPolicy::Dynamic(f.clone()),
            ApprovalPolicy::DynamicForAgent(f) => ApprovalPolicy::DynamicForAgent(f.clone()),
        }
    }
}

impl<C, T> fmt::Debug for ApprovalPolicy<C, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApprovalPolicy::Static(b) => write!(f, "ApprovalPolicy::Static({b})"),
            ApprovalPolicy::Dynamic(_) => write!(f, "ApprovalPolicy::Dynamic(..)"),
            ApprovalPolicy::DynamicForAgent(_) => write!(f, "ApprovalPolicy::DynamicForAgent(..)"),
        }
    }
}

impl<C, T> From<bool> for ApprovalPolicy<C, T> {
    fn from(value: bool) -> Self {
        ApprovalPolicy::Static(value)
    }
}

impl<C, T> ApprovalPolicy<C, T> {
    /// Evaluate the rule for one call. A [`DynamicForAgent`](Self::DynamicForAgent) rule sees an
    /// empty agent name here; use [`evaluate_for_agent`](Self::evaluate_for_agent) to name it.
    pub async fn evaluate(&self, ctx: &RunContext<C>, payload: T, call_id: &str) -> bool {
        self.evaluate_for_agent(ctx, "", payload, call_id).await
    }

    /// Evaluate the rule for one call run by the agent named `agent`.
    pub async fn evaluate_for_agent(
        &self,
        ctx: &RunContext<C>,
        agent: &str,
        payload: T,
        call_id: &str,
    ) -> bool {
        match self {
            ApprovalPolicy::Static(b) => *b,
            ApprovalPolicy::Dynamic(f) => f(ctx.clone(), payload, call_id.to_string()).await,
            ApprovalPolicy::DynamicForAgent(f) => {
                f(ctx.clone(), agent.to_string(), payload, call_id.to_string()).await
            }
        }
    }

    fn dynamic<F, Fut>(f: F) -> Self
    where
        F: Fn(RunContext<C>, T, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        ApprovalPolicy::Dynamic(Arc::new(move |c, t, id| Box::pin(f(c, t, id))))
    }
}

/// The decision returned by an `on_approval` handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalDecision {
    /// Approve (true) or reject (false) the call.
    pub approve: bool,
    /// Model-visible reason for a rejection.
    pub reason: Option<String>,
}

impl ApprovalDecision {
    /// Approve the call.
    pub fn approve() -> Self {
        ApprovalDecision {
            approve: true,
            reason: None,
        }
    }

    /// Reject the call with an optional reason.
    pub fn reject(reason: Option<String>) -> Self {
        ApprovalDecision {
            approve: false,
            reason,
        }
    }
}

/// Handler invoked when a call needs approval. Returning `None` leaves the call pending, which
/// interrupts the run like a tool without a handler.
pub type OnApprovalFn<C> = Arc<
    dyn Fn(RunContext<C>, ToolApprovalItem) -> BoxFuture<'static, Option<ApprovalDecision>>
        + Send
        + Sync,
>;

fn on_approval_fn<C, F, Fut>(f: F) -> OnApprovalFn<C>
where
    F: Fn(RunContext<C>, ToolApprovalItem) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Option<ApprovalDecision>> + Send + 'static,
{
    Arc::new(move |c, i| Box::pin(f(c, i)))
}

// ---------------------------------------------------------------------------------------------
// Shell tool
// ---------------------------------------------------------------------------------------------

/// The action of a `shell_call`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellActionRequest {
    /// Commands to run in order.
    pub commands: Vec<String>,
    /// Per-command timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Maximum combined output length the model wants back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_length: Option<usize>,
}

/// A normalized `shell_call`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellCallData {
    /// The call id.
    pub call_id: String,
    /// The requested action.
    pub action: ShellActionRequest,
    /// `in_progress` or `completed`, when reported.
    pub status: Option<String>,
    /// The raw call item.
    pub raw: Value,
}

impl ShellCallData {
    /// Normalize a raw `shell_call` item.
    pub fn from_raw(raw: &Value) -> Result<Self> {
        let call_id = raw
            .get("call_id")
            .or_else(|| raw.get("id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| AgentsError::model_behavior("Shell call is missing call_id."))?
            .to_string();
        let action = raw.get("action").filter(|a| !a.is_null()).ok_or_else(|| {
            AgentsError::model_behavior("Shell call is missing an action payload.")
        })?;
        let commands_value = action
            .get("commands")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                AgentsError::model_behavior(
                    "Shell call action commands must be a sequence of command strings.",
                )
            })?;
        let commands: Vec<String> = commands_value
            .iter()
            .filter(|v| !v.is_null())
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect();
        if commands.is_empty() {
            return Err(AgentsError::model_behavior(
                "Shell call action must include at least one command.",
            ));
        }
        // Zero has no portable meaning and falls through to the next alias, like None.
        let mut timeout_ms = None;
        for key in ["timeout_ms", "timeoutMs", "timeout"] {
            match action.get(key) {
                None | Some(Value::Null) => continue,
                Some(v) if v.as_f64() == Some(0.0) => continue,
                Some(v) => match v.as_u64() {
                    Some(n) => {
                        timeout_ms = Some(n);
                        break;
                    }
                    None => {
                        return Err(AgentsError::model_behavior(
                            "Shell call action timeout must be a positive integer in \
                             milliseconds, zero, or None.",
                        ));
                    }
                },
            }
        }
        let max_output_length = action
            .get("max_output_length")
            .or_else(|| action.get("maxOutputLength"))
            .and_then(Value::as_f64)
            .map(|n| n.max(0.0) as usize);
        let status = raw
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_lowercase)
            .filter(|s| s == "in_progress" || s == "completed");
        Ok(ShellCallData {
            call_id,
            action: ShellActionRequest {
                commands,
                timeout_ms,
                max_output_length,
            },
            status,
            raw: raw.clone(),
        })
    }
}

/// A request to execute a `shell_call`.
pub struct ShellCommandRequest<C> {
    /// The run context.
    pub ctx: RunContext<C>,
    /// The normalized call.
    pub data: ShellCallData,
}

impl<C> fmt::Debug for ShellCommandRequest<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShellCommandRequest")
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
}

/// How a shell command ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellCallOutcome {
    /// The command exited, with its exit code when known.
    Exit {
        /// Exit code.
        exit_code: Option<i32>,
    },
    /// The command timed out.
    Timeout,
}

impl Default for ShellCallOutcome {
    fn default() -> Self {
        ShellCallOutcome::Exit { exit_code: None }
    }
}

/// Structured output of one shell command.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShellCommandOutput {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// How the command ended.
    pub outcome: ShellCallOutcome,
    /// The command that was run.
    pub command: Option<String>,
    /// Provider-specific metadata.
    pub provider_data: Option<Value>,
}

impl ShellCommandOutput {
    /// The exit code, when the command exited.
    pub fn exit_code(&self) -> Option<i32> {
        match self.outcome {
            ShellCallOutcome::Exit { exit_code } => exit_code,
            ShellCallOutcome::Timeout => None,
        }
    }

    /// `completed` or `timeout`.
    pub fn status(&self) -> &'static str {
        match self.outcome {
            ShellCallOutcome::Timeout => "timeout",
            ShellCallOutcome::Exit { .. } => "completed",
        }
    }

    pub(crate) fn to_value(&self) -> Value {
        let mut payload = json!({
            "stdout": self.stdout,
            "stderr": self.stderr,
            "status": self.status(),
        });
        match self.outcome {
            ShellCallOutcome::Exit { exit_code } => {
                payload["outcome"] = json!({"type": "exit", "exit_code": exit_code});
                if let Some(code) = exit_code {
                    payload["exit_code"] = json!(code);
                }
            }
            ShellCallOutcome::Timeout => payload["outcome"] = json!({"type": "timeout"}),
        }
        if let Some(c) = &self.command {
            payload["command"] = json!(c);
        }
        if let Some(p) = self.provider_data.as_ref().filter(|p| !p.is_null()) {
            payload["provider_data"] = p.clone();
        }
        payload
    }
}

/// Structured result of a shell executor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShellResult {
    /// One entry per command.
    pub output: Vec<ShellCommandOutput>,
    /// Maximum output length applied by the executor.
    pub max_output_length: Option<usize>,
    /// Provider-specific metadata.
    pub provider_data: Option<Value>,
}

/// What a shell executor returns: plain text or structured per-command output.
#[derive(Debug, Clone, PartialEq)]
pub enum ShellExecutorOutput {
    /// Plain text output.
    Text(String),
    /// Structured output.
    Structured(ShellResult),
}

impl From<String> for ShellExecutorOutput {
    fn from(value: String) -> Self {
        ShellExecutorOutput::Text(value)
    }
}

impl From<&str> for ShellExecutorOutput {
    fn from(value: &str) -> Self {
        ShellExecutorOutput::Text(value.to_string())
    }
}

impl From<ShellResult> for ShellExecutorOutput {
    fn from(value: ShellResult) -> Self {
        ShellExecutorOutput::Structured(value)
    }
}

/// Executes a `shell_call`.
pub type ShellExecutorFn<C> = Arc<
    dyn Fn(ShellCommandRequest<C>) -> BoxFuture<'static, Result<ShellExecutorOutput>> + Send + Sync,
>;

/// The next-generation shell tool. With a local environment (the default) the SDK executes the
/// model's commands through your executor; with a hosted container environment OpenAI runs
/// them and the SDK only records the calls.
pub struct ShellTool<C> {
    /// Tool name.
    pub name: String,
    /// Executor for local environments.
    pub executor: Option<ShellExecutorFn<C>>,
    /// Approval rule (local environments only).
    pub needs_approval: ApprovalPolicy<C, ShellActionRequest>,
    /// Optional automatic approval handler.
    pub on_approval: Option<OnApprovalFn<C>>,
    /// Execution environment payload (`{"type": "local"}` by default).
    pub environment: Value,
    /// Callers allowed to invoke this tool (`["direct"]` when unset).
    pub allowed_callers: Option<Vec<String>>,
}

impl<C> Clone for ShellTool<C> {
    fn clone(&self) -> Self {
        ShellTool {
            name: self.name.clone(),
            executor: self.executor.clone(),
            needs_approval: self.needs_approval.clone(),
            on_approval: self.on_approval.clone(),
            environment: self.environment.clone(),
            allowed_callers: self.allowed_callers.clone(),
        }
    }
}

impl<C> fmt::Debug for ShellTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShellTool")
            .field("name", &self.name)
            .field("environment", &self.environment)
            .field("needs_approval", &self.needs_approval)
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> ShellTool<C> {
    /// A local shell tool backed by `executor`.
    pub fn new<F, Fut, O>(executor: F) -> Self
    where
        F: Fn(ShellCommandRequest<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O>> + Send + 'static,
        O: Into<ShellExecutorOutput>,
    {
        let f = Arc::new(executor);
        ShellTool {
            allowed_callers: None,
            name: "shell".into(),
            executor: Some(Arc::new(move |req| {
                let f = f.clone();
                Box::pin(async move { f(req).await.map(Into::into) })
            })),
            needs_approval: ApprovalPolicy::Static(false),
            on_approval: None,
            environment: json!({"type": "local"}),
        }
    }

    /// A local shell tool that runs commands with the system shell (`sh -c`, or `cmd /C` on
    /// Windows) in `working_dir`. This executes model-chosen commands on your machine, so pair
    /// it with [`with_needs_approval`](Self::with_needs_approval) unless you sandbox the host.
    pub fn with_system_shell(working_dir: Option<PathBuf>) -> Self {
        Self::new(move |req: ShellCommandRequest<C>| {
            let dir = working_dir.clone();
            async move { Ok::<_, AgentsError>(run_system_shell(&req.data.action, dir).await) }
        })
    }

    /// A shell tool running in a hosted OpenAI container, from the typed environment.
    /// Rejects [`ShellToolEnvironment::Local`], which needs an executor.
    pub fn hosted_typed(environment: ShellToolEnvironment) -> Result<Self> {
        Self::hosted(environment.to_value())
    }

    /// A shell tool running in a hosted OpenAI container (`container_auto` or
    /// `container_reference` environment). Hosted shells do not support approvals.
    ///
    /// [`hosted_typed`](Self::hosted_typed) takes the typed form; this one stays a [`Value`]
    /// so an unmodelled environment still works.
    pub fn hosted(environment: Value) -> Result<Self> {
        let ty = environment.get("type").and_then(Value::as_str);
        if matches!(ty, None | Some("local")) {
            return Err(AgentsError::user(
                "ShellTool with local environment requires an executor. Use ShellTool::new for \
                 local shells; ShellTool::hosted takes a container_auto or container_reference \
                 environment.",
            ));
        }
        Ok(ShellTool {
            allowed_callers: None,
            name: "shell".into(),
            executor: None,
            needs_approval: ApprovalPolicy::Static(false),
            on_approval: None,
            environment,
        })
    }
}

impl<C> ShellTool<C> {
    /// Set the tool name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Require approval for every call (or none).
    pub fn with_needs_approval(mut self, needs_approval: bool) -> Self {
        self.needs_approval = ApprovalPolicy::Static(needs_approval);
        self
    }

    /// Decide approval per call.
    pub fn with_needs_approval_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, ShellActionRequest, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.needs_approval = ApprovalPolicy::dynamic(f);
        self
    }

    /// Approve or reject calls automatically when approval is required.
    pub fn with_on_approval<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, ToolApprovalItem) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<ApprovalDecision>> + Send + 'static,
    {
        self.on_approval = Some(on_approval_fn(f));
        self
    }

    /// Set the local environment payload (for example to attach skills). The `type` defaults to
    /// `local`.
    pub fn with_local_environment(mut self, mut environment: Value) -> Self {
        if let Value::Object(m) = &mut environment {
            m.entry("type").or_insert_with(|| json!("local"));
        }
        self.environment = environment;
        self
    }

    /// True when commands run on the local machine.
    pub fn is_local(&self) -> bool {
        matches!(
            self.environment.get("type").and_then(Value::as_str),
            None | Some("local")
        )
    }

    /// Check that the executor, environment and approval settings agree. Python rejects these
    /// combinations when the tool is constructed; the Rust builders are infallible, so the agent
    /// runs this check when it resolves its tools.
    pub fn validate(&self) -> Result<()> {
        if self.is_local() {
            if self.executor.is_none() {
                return Err(AgentsError::user(
                    "ShellTool with local environment requires an executor.",
                ));
            }
            return Ok(());
        }
        if self.executor.is_some() {
            return Err(AgentsError::user(
                "ShellTool with hosted environment does not accept an executor.",
            ));
        }
        if !matches!(self.needs_approval, ApprovalPolicy::Static(false))
            || self.on_approval.is_some()
        {
            return Err(AgentsError::user(
                "ShellTool with hosted environment does not support needs_approval or on_approval.",
            ));
        }
        Ok(())
    }

    /// Restrict which callers may invoke the tool (`"direct"`, `"programmatic"`).
    pub fn with_allowed_callers(mut self, callers: Vec<String>) -> Self {
        self.allowed_callers = Some(callers);
        self
    }

    /// Set the execution environment from the typed form.
    pub fn with_typed_environment(mut self, environment: ShellToolEnvironment) -> Self {
        self.environment = environment.to_value();
        self
    }

    /// Responses API tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = json!({"type": "shell", "environment": self.environment});
        if let Some(callers) = &self.allowed_callers {
            v["allowed_callers"] = json!(callers);
        }
        v
    }
}

/// Run shell commands with the system shell, one after another, honoring the action timeout.
pub async fn run_system_shell(
    action: &ShellActionRequest,
    working_dir: Option<PathBuf>,
) -> ShellResult {
    let mut output = Vec::new();
    for command in &action.commands {
        let mut cmd = if cfg!(windows) {
            let mut c = tokio::process::Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(command);
            c
        };
        if let Some(dir) = &working_dir {
            cmd.current_dir(dir);
        }
        cmd.kill_on_drop(true);
        cmd.stdin(std::process::Stdio::null());
        let run = cmd.output();
        let result = match action.timeout_ms {
            Some(ms) => match tokio::time::timeout(Duration::from_millis(ms), run).await {
                Ok(r) => r,
                Err(_) => {
                    output.push(ShellCommandOutput {
                        outcome: ShellCallOutcome::Timeout,
                        command: Some(command.clone()),
                        ..Default::default()
                    });
                    continue;
                }
            },
            None => run.await,
        };
        output.push(match result {
            Ok(o) => ShellCommandOutput {
                stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
                outcome: ShellCallOutcome::Exit {
                    exit_code: o.status.code(),
                },
                command: Some(command.clone()),
                provider_data: None,
            },
            Err(e) => ShellCommandOutput {
                stderr: e.to_string(),
                outcome: ShellCallOutcome::Exit { exit_code: Some(1) },
                command: Some(command.clone()),
                ..Default::default()
            },
        });
    }
    ShellResult {
        output,
        max_output_length: action.max_output_length,
        provider_data: None,
    }
}

// ---------------------------------------------------------------------------------------------
// Local shell tool (legacy)
// ---------------------------------------------------------------------------------------------

/// A request to execute a legacy `local_shell_call`.
pub struct LocalShellCommandRequest<C> {
    /// The run context.
    pub ctx: RunContext<C>,
    /// The raw `local_shell_call` item (`action.command`, `action.env`, `action.timeout_ms`,
    /// `action.working_directory`, ...).
    pub data: Value,
}

impl<C> fmt::Debug for LocalShellCommandRequest<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalShellCommandRequest")
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
}

/// Executes a `local_shell_call` and returns its output text.
pub type LocalShellExecutorFn<C> =
    Arc<dyn Fn(LocalShellCommandRequest<C>) -> BoxFuture<'static, Result<String>> + Send + Sync>;

/// The legacy local shell tool (`local_shell`). Prefer [`ShellTool`].
pub struct LocalShellTool<C> {
    /// The executor.
    pub executor: LocalShellExecutorFn<C>,
}

impl<C> Clone for LocalShellTool<C> {
    fn clone(&self) -> Self {
        LocalShellTool {
            executor: self.executor.clone(),
        }
    }
}

impl<C> fmt::Debug for LocalShellTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LocalShellTool")
    }
}

impl<C: Send + Sync + 'static> LocalShellTool<C> {
    /// Create the tool from an executor.
    pub fn new<F, Fut>(executor: F) -> Self
    where
        F: Fn(LocalShellCommandRequest<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        LocalShellTool {
            executor: Arc::new(move |r| Box::pin(executor(r))),
        }
    }
}

impl<C> LocalShellTool<C> {
    /// The tool name.
    pub fn name(&self) -> &'static str {
        "local_shell"
    }

    /// Responses API tool param.
    pub fn to_responses_param(&self) -> Value {
        json!({"type": "local_shell"})
    }
}

// ---------------------------------------------------------------------------------------------
// Apply patch tool
// ---------------------------------------------------------------------------------------------

/// Lets the model create, update and delete files with V4A diffs through an
/// [`ApplyPatchEditor`].
pub struct ApplyPatchTool<C> {
    /// The editor applying the operations.
    pub editor: Arc<dyn ApplyPatchEditor>,
    /// Tool name.
    pub name: String,
    /// Approval rule, evaluated per operation.
    pub needs_approval: ApprovalPolicy<C, ApplyPatchOperation>,
    /// Optional automatic approval handler.
    pub on_approval: Option<OnApprovalFn<C>>,
    /// Callers allowed to invoke this tool (`["direct"]` when unset).
    pub allowed_callers: Option<Vec<String>>,
    /// Attaches SDK-only data to the `apply_patch` output item.
    pub custom_data_extractor: Option<ApplyPatchToolCustomDataExtractor<C>>,
}

impl<C> Clone for ApplyPatchTool<C> {
    fn clone(&self) -> Self {
        ApplyPatchTool {
            editor: self.editor.clone(),
            name: self.name.clone(),
            needs_approval: self.needs_approval.clone(),
            on_approval: self.on_approval.clone(),
            allowed_callers: self.allowed_callers.clone(),
            custom_data_extractor: self.custom_data_extractor.clone(),
        }
    }
}

impl<C> fmt::Debug for ApplyPatchTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApplyPatchTool")
            .field("name", &self.name)
            .field("needs_approval", &self.needs_approval)
            .finish_non_exhaustive()
    }
}

impl<C> ApplyPatchTool<C> {
    /// Create the tool around an editor.
    pub fn new(editor: impl ApplyPatchEditor + 'static) -> Self {
        Self::from_arc(Arc::new(editor))
    }

    /// Create the tool around a shared editor.
    pub fn from_arc(editor: Arc<dyn ApplyPatchEditor>) -> Self {
        ApplyPatchTool {
            allowed_callers: None,
            editor,
            name: "apply_patch".into(),
            needs_approval: ApprovalPolicy::Static(false),
            on_approval: None,
            custom_data_extractor: None,
        }
    }

    /// Set the tool name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Require approval for every call (or none).
    pub fn with_needs_approval(mut self, needs_approval: bool) -> Self {
        self.needs_approval = ApprovalPolicy::Static(needs_approval);
        self
    }

    /// Decide approval per operation.
    pub fn with_needs_approval_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, ApplyPatchOperation, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.needs_approval = ApprovalPolicy::dynamic(f);
        self
    }

    /// Approve or reject calls automatically when approval is required.
    pub fn with_on_approval<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, ToolApprovalItem) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<ApprovalDecision>> + Send + 'static,
    {
        self.on_approval = Some(on_approval_fn(f));
        self
    }

    /// Restrict which callers may invoke the tool (`"direct"`, `"programmatic"`).
    pub fn with_allowed_callers(mut self, callers: Vec<String>) -> Self {
        self.allowed_callers = Some(callers);
        self
    }

    /// Attach SDK-only data to the `apply_patch` output item.
    pub fn with_custom_data_extractor<F, Fut>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&ApplyPatchToolCustomDataContext<'a, C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<Value>> + Send + 'static,
    {
        self.custom_data_extractor = Some(Arc::new(move |ctx| Box::pin(f(ctx))));
        self
    }

    /// Responses API tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = json!({"type": "apply_patch"});
        if let Some(callers) = &self.allowed_callers {
            v["allowed_callers"] = json!(callers);
        }
        v
    }
}

/// Parse the operations of an `apply_patch_call` (`operations` list or single `operation`).
pub fn parse_apply_patch_operations(call: &Value) -> Result<Vec<ApplyPatchOperation>> {
    fn one(raw: &Value) -> Result<ApplyPatchOperation> {
        let ty = raw.get("type").and_then(Value::as_str).unwrap_or("None");
        let op_type = match ty {
            "create_file" => crate::editor::ApplyPatchOperationType::CreateFile,
            "update_file" => crate::editor::ApplyPatchOperationType::UpdateFile,
            "delete_file" => crate::editor::ApplyPatchOperationType::DeleteFile,
            other => {
                return Err(AgentsError::model_behavior(format!(
                    "Unknown apply_patch operation: {other}"
                )));
            }
        };
        let path = raw
            .get("path")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                AgentsError::model_behavior("Apply patch operation is missing a valid path.")
            })?;
        let diff = match op_type {
            crate::editor::ApplyPatchOperationType::DeleteFile => None,
            _ => Some(
                raw.get("diff")
                    .and_then(Value::as_str)
                    .filter(|d| !d.is_empty())
                    .ok_or_else(|| {
                        AgentsError::model_behavior(format!(
                            "Apply patch operation {ty} is missing the required diff payload."
                        ))
                    })?
                    .to_string(),
            ),
        };
        let move_to = match raw.get("move_to") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(_) => {
                return Err(AgentsError::model_behavior(
                    "Apply patch operation move_to must be a non-empty path.",
                ));
            }
        };
        Ok(ApplyPatchOperation {
            op_type,
            path: path.to_string(),
            diff,
            move_to,
        })
    }
    if let Some(list) = call.get("operations").and_then(Value::as_array) {
        if list.is_empty() {
            return Err(AgentsError::model_behavior(
                "Apply patch call includes no operations.",
            ));
        }
        return list.iter().map(one).collect();
    }
    match call.get("operation").filter(|v| !v.is_null()) {
        Some(op) => Ok(vec![one(op)?]),
        None => Err(AgentsError::model_behavior(
            "Apply patch call is missing an operation payload.",
        )),
    }
}

// ---------------------------------------------------------------------------------------------
// Computer tool
// ---------------------------------------------------------------------------------------------

/// Data passed to [`ComputerTool::on_safety_check`].
pub struct ComputerToolSafetyCheckData<C> {
    /// The run context.
    pub ctx: RunContext<C>,
    /// Name of the agent running the tool.
    pub agent_name: String,
    /// The raw `computer_call` item.
    pub tool_call: Value,
    /// The pending safety check (`id`, `code`, `message`).
    pub safety_check: Value,
}

impl<C> fmt::Debug for ComputerToolSafetyCheckData<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputerToolSafetyCheckData")
            .field("agent_name", &self.agent_name)
            .field("safety_check", &self.safety_check)
            .finish_non_exhaustive()
    }
}

/// Safety check callback: return true to acknowledge the check, false to abort the run.
pub type SafetyCheckFn<C> =
    Arc<dyn Fn(ComputerToolSafetyCheckData<C>) -> BoxFuture<'static, bool> + Send + Sync>;

type ComputerFactory = Arc<dyn Fn() -> BoxFuture<'static, Result<Arc<dyn Computer>>> + Send + Sync>;
/// What a computer's dispose hook returns: `()`, or a `Result` whose error is logged.
pub trait ComputerDisposeOutput: Send + 'static {
    /// The failure to log, if the dispose failed.
    fn into_dispose_error(self) -> Option<String>;
}

impl ComputerDisposeOutput for () {
    fn into_dispose_error(self) -> Option<String> {
        None
    }
}

impl<E: fmt::Display + Send + 'static> ComputerDisposeOutput for std::result::Result<(), E> {
    fn into_dispose_error(self) -> Option<String> {
        self.err().map(|e| e.to_string())
    }
}

/// Tears down a computer a factory created. Port of `ComputerProvider.dispose`. Returns the
/// failure message to log, if any.
pub type ComputerDisposeFn =
    Arc<dyn Fn(Arc<dyn Computer>) -> BoxFuture<'static, Option<String>> + Send + Sync>;

#[derive(Clone)]
enum ComputerSource {
    Ready(Arc<dyn Computer>),
    Lazy {
        factory: ComputerFactory,
        dispose: Option<ComputerDisposeFn>,
        /// Identifies this tool's computer within a run (see `RunContext`).
        id: u64,
    },
}

/// A computer a lazy computer tool created for one run, kept until the run ends.
pub(crate) struct ResolvedComputer {
    pub(crate) computer: Arc<dyn Computer>,
    pub(crate) dispose: Option<ComputerDisposeFn>,
}

static NEXT_COMPUTER_TOOL_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_computer_tool_id() -> u64 {
    NEXT_COMPUTER_TOOL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Lets the model control a [`Computer`] through the Responses API computer tool.
pub struct ComputerTool<C> {
    source: ComputerSource,
    /// Callback for pending safety checks. Without it, checks are left unacknowledged and the
    /// action proceeds with a warning.
    pub on_safety_check: Option<SafetyCheckFn<C>>,
    /// Attaches SDK-only data to the computer call output item.
    pub custom_data_extractor: Option<ComputerToolCustomDataExtractor<C>>,
}

impl<C> Clone for ComputerTool<C> {
    fn clone(&self) -> Self {
        ComputerTool {
            source: self.source.clone(),
            on_safety_check: self.on_safety_check.clone(),
            custom_data_extractor: self.custom_data_extractor.clone(),
        }
    }
}

impl<C> fmt::Debug for ComputerTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ComputerTool")
    }
}

impl<C> ComputerTool<C> {
    /// Wrap a computer.
    pub fn new(computer: impl Computer + 'static) -> Self {
        Self::from_arc(Arc::new(computer))
    }

    /// Wrap a shared computer.
    pub fn from_arc(computer: Arc<dyn Computer>) -> Self {
        ComputerTool {
            source: ComputerSource::Ready(computer),
            on_safety_check: None,
            custom_data_extractor: None,
        }
    }

    /// Create the computer lazily, the first time a run needs it. Each run gets its own computer.
    pub fn from_factory<F, Fut>(factory: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Arc<dyn Computer>>> + Send + 'static,
    {
        ComputerTool {
            source: ComputerSource::Lazy {
                factory: Arc::new(move || Box::pin(factory())),
                dispose: None,
                id: next_computer_tool_id(),
            },
            on_safety_check: None,
            custom_data_extractor: None,
        }
    }

    /// Create the computer lazily and tear it down when the run that created it finishes,
    /// so the next run starts from a fresh computer.
    /// Port of `ComputerProvider { create, dispose }`.
    pub fn from_provider<F, Fut, D, DFut, T>(factory: F, dispose: D) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Arc<dyn Computer>>> + Send + 'static,
        D: Fn(Arc<dyn Computer>) -> DFut + Send + Sync + 'static,
        DFut: Future<Output = T> + Send + 'static,
        T: ComputerDisposeOutput,
    {
        ComputerTool {
            source: ComputerSource::Lazy {
                factory: Arc::new(move || Box::pin(factory())),
                dispose: Some(Arc::new(move |c| {
                    let fut = dispose(c);
                    Box::pin(async move { fut.await.into_dispose_error() })
                })),
                id: next_computer_tool_id(),
            },
            on_safety_check: None,
            custom_data_extractor: None,
        }
    }

    /// Set the safety check callback.
    pub fn with_on_safety_check<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(ComputerToolSafetyCheckData<C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.on_safety_check = Some(Arc::new(move |d| Box::pin(f(d))));
        self
    }

    /// Attach SDK-only data to the computer call output item.
    pub fn with_custom_data_extractor<F, Fut>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&ComputerToolCustomDataContext<'a, C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<Value>> + Send + 'static,
    {
        self.custom_data_extractor = Some(Arc::new(move |ctx| Box::pin(f(ctx))));
        self
    }

    /// The runtime tool name (kept as the preview-era name for hooks and approvals).
    pub fn name(&self) -> &'static str {
        "computer_use_preview"
    }

    /// The computer for `ctx`'s run, creating it on first use. Lazy computers are per run: two runs
    /// of the same agent get separate computers (Python's per-run-context `resolve_computer`).
    pub async fn resolve(&self, ctx: &RunContext<C>) -> Result<Arc<dyn Computer>> {
        match &self.source {
            ComputerSource::Ready(c) => Ok(c.clone()),
            ComputerSource::Lazy {
                factory,
                dispose,
                id,
            } => {
                if let Some(computer) = ctx.run_computer(*id) {
                    return Ok(computer);
                }
                let computer = factory().await?;
                Ok(ctx.store_run_computer(
                    *id,
                    ResolvedComputer {
                        computer,
                        dispose: dispose.clone(),
                    },
                ))
            }
        }
    }

    /// The tool param handed to models. The Responses model picks the GA `computer` or preview
    /// `computer_use_preview` wire shape from the model name.
    pub async fn to_responses_param(&self, ctx: &RunContext<C>) -> Result<Value> {
        let computer = self.resolve(ctx).await?;
        let mut param = json!({"type": "computer"});
        if let Some(env) = computer.environment() {
            param["environment"] = json!(env.as_str());
        }
        if let Some((w, h)) = computer.dimensions() {
            param["display_width"] = json!(w);
            param["display_height"] = json!(h);
        }
        Ok(param)
    }
}

// ---------------------------------------------------------------------------------------------
// Custom tool
// ---------------------------------------------------------------------------------------------

/// Executes a custom tool call with its raw string input.
pub type CustomToolFn<C> =
    Arc<dyn Fn(ToolContext<C>, String) -> BoxFuture<'static, Result<String>> + Send + Sync>;

/// A Responses custom tool: the model sends one raw string (optionally constrained by a grammar
/// `format`) instead of JSON arguments.
pub struct CustomTool<C> {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Optional input format (`{"type": "grammar", ...}` or `{"type": "text"}`).
    pub format: Option<Value>,
    /// Approval rule; the payload is the raw input.
    pub needs_approval: ApprovalPolicy<C, String>,
    /// Optional automatic approval handler.
    pub on_approval: Option<OnApprovalFn<C>>,
    /// Hide the tool until loaded through tool search.
    pub defer_loading: bool,
    /// Callers allowed to invoke this tool (`["direct"]` when unset).
    pub allowed_callers: Option<Vec<String>>,
    /// Attaches SDK-only data to the custom tool output item.
    pub custom_data_extractor: Option<CustomToolCustomDataExtractor<C>>,
    on_invoke: CustomToolFn<C>,
}

impl<C> Clone for CustomTool<C> {
    fn clone(&self) -> Self {
        CustomTool {
            name: self.name.clone(),
            description: self.description.clone(),
            format: self.format.clone(),
            needs_approval: self.needs_approval.clone(),
            on_approval: self.on_approval.clone(),
            defer_loading: self.defer_loading,
            allowed_callers: self.allowed_callers.clone(),
            custom_data_extractor: self.custom_data_extractor.clone(),
            on_invoke: self.on_invoke.clone(),
        }
    }
}

impl<C> fmt::Debug for CustomTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> CustomTool<C> {
    /// Create a custom tool.
    pub fn new<F, Fut>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
    where
        F: Fn(ToolContext<C>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        CustomTool {
            allowed_callers: None,
            name: name.into(),
            description: description.into(),
            format: None,
            needs_approval: ApprovalPolicy::Static(false),
            on_approval: None,
            defer_loading: false,
            custom_data_extractor: None,
            on_invoke: Arc::new(move |c, i| Box::pin(f(c, i))),
        }
    }
}

impl<C> CustomTool<C> {
    /// Constrain the input format.
    pub fn with_format(mut self, format: Value) -> Self {
        self.format = Some(format);
        self
    }

    /// Attach SDK-only data to the custom tool output item.
    pub fn with_custom_data_extractor<F, Fut>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&CustomToolCustomDataContext<'a, C>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<Value>> + Send + 'static,
    {
        self.custom_data_extractor = Some(Arc::new(move |ctx| Box::pin(f(ctx))));
        self
    }

    /// Require approval for every call (or none).
    pub fn with_needs_approval(mut self, needs_approval: bool) -> Self {
        self.needs_approval = ApprovalPolicy::Static(needs_approval);
        self
    }

    /// Decide approval per call from the raw input.
    pub fn with_needs_approval_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, String, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.needs_approval = ApprovalPolicy::dynamic(f);
        self
    }

    /// Approve or reject calls automatically when approval is required.
    pub fn with_on_approval<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, ToolApprovalItem) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<ApprovalDecision>> + Send + 'static,
    {
        self.on_approval = Some(on_approval_fn(f));
        self
    }

    /// Defer loading until the model searches for the tool.
    pub fn with_defer_loading(mut self, defer: bool) -> Self {
        self.defer_loading = defer;
        self
    }

    /// Invoke the tool directly.
    pub async fn invoke(&self, ctx: ToolContext<C>, input: String) -> Result<String> {
        (self.on_invoke)(ctx, input).await
    }

    /// Responses API tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = Map::new();
        v.insert("type".into(), json!("custom"));
        v.insert("name".into(), json!(self.name));
        v.insert("description".into(), json!(self.description));
        if let Some(f) = &self.format {
            v.insert("format".into(), f.clone());
        }
        if self.defer_loading {
            v.insert("defer_loading".into(), json!(true));
        }
        if let Some(callers) = &self.allowed_callers {
            v.insert("allowed_callers".into(), json!(callers));
        }
        Value::Object(v)
    }

    /// Restrict which callers may invoke the tool (`"direct"`, `"programmatic"`).
    pub fn with_allowed_callers(mut self, callers: Vec<String>) -> Self {
        self.allowed_callers = Some(callers);
        self
    }
}

// ---------------------------------------------------------------------------------------------
// Hosted helpers
// ---------------------------------------------------------------------------------------------

/// Hosted tool search: lets the model load deferred tools (`defer_loading`) and namespaces on
/// demand. Responses API only.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolSearchTool {
    /// Optional description.
    pub description: Option<String>,
    /// `server` (default) or `client`. The runner does not execute client-side searches.
    pub execution: Option<String>,
    /// Optional parameters schema for client execution.
    pub parameters: Option<Value>,
}

impl ToolSearchTool {
    /// Create a tool search tool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Responses API tool param.
    pub fn to_responses_param(&self) -> Value {
        let mut v = json!({"type": "tool_search"});
        if let Some(d) = &self.description {
            v["description"] = json!(d);
        }
        if let Some(e) = &self.execution {
            v["execution"] = json!(e);
        }
        if let Some(p) = &self.parameters {
            v["parameters"] = p.clone();
        }
        v
    }
}

/// Hosted programmatic tool calling: lets model-generated code orchestrate other tools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgrammaticToolCallingTool;

impl ProgrammaticToolCallingTool {
    /// Responses API tool param.
    pub fn to_responses_param(&self) -> Value {
        json!({"type": "programmatic_tool_calling"})
    }
}

// ---------------------------------------------------------------------------------------------
// Shell output helpers
// ---------------------------------------------------------------------------------------------

fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Truncate shell outputs to a maximum combined length.
pub(crate) fn truncate_shell_outputs(
    outputs: &[ShellCommandOutput],
    max_length: usize,
) -> Vec<ShellCommandOutput> {
    let mut remaining = max_length;
    outputs
        .iter()
        .map(|o| {
            let mut stdout = String::new();
            let mut stderr = String::new();
            if remaining > 0 && !o.stdout.is_empty() {
                stdout = truncate_chars(&o.stdout, remaining);
                remaining -= stdout.chars().count();
            }
            if remaining > 0 && !o.stderr.is_empty() {
                stderr = truncate_chars(&o.stderr, remaining);
                remaining -= stderr.chars().count();
            }
            ShellCommandOutput {
                stdout,
                stderr,
                ..o.clone()
            }
        })
        .collect()
}

/// Render shell outputs as model-readable text.
pub(crate) fn render_shell_outputs(outputs: &[ShellCommandOutput]) -> String {
    if outputs.is_empty() {
        return "(no output)".into();
    }
    outputs
        .iter()
        .map(|r| {
            let mut lines: Vec<String> = Vec::new();
            if let Some(c) = &r.command {
                lines.push(format!("$ {c}"));
            }
            let stdout = r.stdout.trim_end_matches('\n');
            let stderr = r.stderr.trim_end_matches('\n');
            if !stdout.is_empty() {
                lines.push(stdout.to_string());
            }
            if !stderr.is_empty() {
                if !stdout.is_empty() {
                    lines.push(String::new());
                }
                lines.push("stderr:".into());
                lines.push(stderr.to_string());
            }
            if let Some(code) = r.exit_code().filter(|c| *c != 0) {
                lines.push(format!("exit code: {code}"));
            }
            if r.status() == "timeout" {
                lines.push("status: timeout".into());
            }
            let chunk = lines.join("\n").trim().to_string();
            if chunk.is_empty() {
                "(no output)".to_string()
            } else {
                chunk
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Convert serialized shell output entries into the model-facing `output` payload.
pub(crate) fn normalize_shell_output_entries(entries: &[Value]) -> Vec<Value> {
    entries
        .iter()
        .map(|entry| {
            let mut m = entry.as_object().cloned().unwrap_or_default();
            let status = m.remove("status");
            m.remove("provider_data");
            let raw_exit = m.remove("exit_code");
            m.remove("command");
            let resolve_exit = |status_str: Option<&str>| -> Value {
                if let Some(code) = raw_exit.as_ref().and_then(Value::as_i64) {
                    return json!(code);
                }
                match status_str.map(str::to_lowercase).as_deref() {
                    Some("failure") => json!(1),
                    _ => json!(0),
                }
            };
            match m.get("outcome").cloned() {
                Some(Value::String(outcome)) => {
                    let timeout = status.as_ref().and_then(Value::as_str) == Some("timeout");
                    let payload = if timeout {
                        json!({"type": "timeout"})
                    } else {
                        json!({"type": "exit", "exit_code": resolve_exit(Some(&outcome))})
                    };
                    m.insert("outcome".into(), payload);
                }
                Some(Value::Object(mut o)) => {
                    let st = o.remove("status");
                    // Python's `setdefault` keeps an explicit `exit_code: null` as it is; only a
                    // missing key is resolved from the status.
                    if o.get("type").and_then(Value::as_str) != Some("timeout")
                        && !o.contains_key("exit_code")
                    {
                        o.insert(
                            "exit_code".into(),
                            resolve_exit(st.as_ref().and_then(Value::as_str)),
                        );
                    }
                    m.insert("outcome".into(), Value::Object(o));
                }
                _ => {}
            }
            Value::Object(m)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coerces_shell_call() {
        let raw = json!({"type": "shell_call", "call_id": "c1",
            "action": {"commands": ["ls", null], "timeout_ms": 0, "timeout": 500, "max_output_length": 10}});
        let data = ShellCallData::from_raw(&raw).unwrap();
        assert_eq!(data.action.commands, vec!["ls"]);
        assert_eq!(data.action.timeout_ms, Some(500));
        assert_eq!(data.action.max_output_length, Some(10));
        assert!(
            ShellCallData::from_raw(&json!({"call_id": "x", "action": {"commands": []}})).is_err()
        );
    }

    #[test]
    fn renders_and_truncates() {
        let outs = vec![ShellCommandOutput {
            stdout: "hello\n".into(),
            stderr: "warn".into(),
            outcome: ShellCallOutcome::Exit { exit_code: Some(2) },
            command: Some("echo hello".into()),
            provider_data: None,
        }];
        assert_eq!(
            render_shell_outputs(&outs),
            "$ echo hello\nhello\n\nstderr:\nwarn\nexit code: 2"
        );
        let t = truncate_shell_outputs(&outs, 8);
        assert_eq!(t[0].stdout, "hello\n");
        assert_eq!(t[0].stderr, "wa");
        let entries = normalize_shell_output_entries(&[outs[0].to_value()]);
        assert_eq!(
            entries[0],
            json!({"stdout": "hello\n", "stderr": "warn", "outcome": {"type": "exit", "exit_code": 2}})
        );
    }

    #[test]
    fn parses_patch_operations() {
        let ops = parse_apply_patch_operations(&json!({"operation": {
            "type": "update_file", "path": "a.txt", "diff": "@@\n-a\n+b\n"}}))
        .unwrap();
        assert_eq!(ops.len(), 1);
        assert!(
            parse_apply_patch_operations(
                &json!({"operation": {"type": "create_file", "path": "a"}})
            )
            .is_err()
        );
        assert!(parse_apply_patch_operations(&json!({"operations": []})).is_err());
    }

    #[tokio::test]
    async fn system_shell_runs_commands() {
        let action = ShellActionRequest {
            commands: vec!["echo hi".into(), "exit 3".into()],
            timeout_ms: Some(10_000),
            max_output_length: None,
        };
        let r = run_system_shell(&action, None).await;
        assert_eq!(r.output[0].stdout.trim(), "hi");
        assert_eq!(r.output[1].exit_code(), Some(3));
    }
}

// ---------------------------------------------------------------------------------------------
// Typed shell tool configuration
// ---------------------------------------------------------------------------------------------

/// Skill metadata for a local shell environment.
/// Mirrors `agents.tool.ShellToolLocalSkill`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellToolLocalSkill {
    pub description: String,
    pub name: String,
    pub path: String,
}

/// Inline skill payload. Mirrors `ShellToolInlineSkillSource`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellToolInlineSkillSource {
    /// Base64 zip bundle.
    pub data: String,
    /// Always `application/zip`.
    pub media_type: String,
    /// Always `base64`.
    #[serde(rename = "type")]
    pub source_type: String,
}

impl ShellToolInlineSkillSource {
    /// A base64 zip bundle.
    pub fn zip(data: impl Into<String>) -> Self {
        ShellToolInlineSkillSource {
            data: data.into(),
            media_type: "application/zip".into(),
            source_type: "base64".into(),
        }
    }
}

/// A skill a hosted container may load. Mirrors `ShellToolContainerSkill`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShellToolContainerSkill {
    /// A hosted skill by id. Mirrors `ShellToolSkillReference`.
    SkillReference {
        skill_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<String>,
    },
    /// An inline skill bundle. Mirrors `ShellToolInlineSkill`.
    Inline {
        description: String,
        name: String,
        source: ShellToolInlineSkillSource,
    },
}

/// A secret bound to one domain in allowlist mode.
/// Mirrors `ShellToolContainerNetworkPolicyDomainSecret`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellToolContainerNetworkPolicyDomainSecret {
    pub domain: String,
    pub name: String,
    pub value: String,
}

/// Network policy of a hosted shell container.
/// Mirrors `ShellToolContainerNetworkPolicy`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShellToolContainerNetworkPolicy {
    Allowlist {
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        domain_secrets: Option<Vec<ShellToolContainerNetworkPolicyDomainSecret>>,
    },
    Disabled,
}

/// A shell execution environment. Mirrors `ShellToolEnvironment`.
///
/// [`ShellTool::environment`] stays a [`Value`] so an unmodelled payload still works; use
/// [`ShellTool::with_typed_environment`] for the typed form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShellToolEnvironment {
    /// The SDK runs the commands through the tool's executor.
    Local {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        skills: Option<Vec<ShellToolLocalSkill>>,
    },
    /// OpenAI provisions a container and runs the commands.
    ContainerAuto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_ids: Option<Vec<String>>,
        /// `1g`, `4g`, `16g` or `64g`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        memory_limit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        network_policy: Option<ShellToolContainerNetworkPolicy>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        skills: Option<Vec<ShellToolContainerSkill>>,
    },
    /// An existing hosted container.
    ContainerReference { container_id: String },
}

impl ShellToolEnvironment {
    /// The wire payload for [`ShellTool::environment`].
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({"type": "local"}))
    }

    /// True for the two environments OpenAI runs the commands in. The Rust equivalent of
    /// Python's `ShellToolHostedEnvironment` union.
    pub fn is_hosted(&self) -> bool {
        !matches!(self, ShellToolEnvironment::Local { .. })
    }
}

/// Image result settings for a web search. Mirrors `WebSearchToolImageSettings`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchToolImageSettings {
    /// Number of image results to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
    /// Include a short caption with each image when one is available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<bool>,
}

impl WebSearchToolImageSettings {
    /// The wire payload for `WebSearchTool::image_settings`.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({}))
    }
}

/// Responses image generation settings. Mirrors `ImageGenerationToolConfig`.
///
/// `ImageGenerationTool::tool_config` stays a [`Value`], so a field this struct does not model
/// can still be sent; use [`ImageGenerationToolConfig::to_value`] for the typed form.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageGenerationToolConfig {
    /// `generate`, `edit` or `auto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// `transparent`, `opaque` or `auto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<String>,
    /// `high` or `low`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fidelity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_image_mask: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// `auto` or `low`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moderation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u32>,
    /// `png`, `webp` or `jpeg`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u32>,
    /// `low`, `medium`, `high`, `xhigh`, `max` or `auto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
}

impl ImageGenerationToolConfig {
    /// The wire payload for `ImageGenerationTool::tool_config`, including `type`.
    pub fn to_value(&self) -> Value {
        let mut value = serde_json::to_value(self).unwrap_or_else(|_| json!({}));
        value["type"] = json!("image_generation");
        value
    }
}

#[cfg(test)]
mod typed_config_tests {
    use super::*;

    #[test]
    fn shell_environments_match_the_wire_shape() {
        assert_eq!(
            ShellToolEnvironment::Local { skills: None }.to_value(),
            json!({"type": "local"})
        );
        assert_eq!(
            ShellToolEnvironment::ContainerReference {
                container_id: "c_1".into()
            }
            .to_value(),
            json!({"type": "container_reference", "container_id": "c_1"})
        );
        let auto = ShellToolEnvironment::ContainerAuto {
            file_ids: Some(vec!["f1".into()]),
            memory_limit: Some("4g".into()),
            network_policy: Some(ShellToolContainerNetworkPolicy::Allowlist {
                allowed_domains: vec!["example.com".into()],
                domain_secrets: None,
            }),
            skills: Some(vec![ShellToolContainerSkill::SkillReference {
                skill_id: "s_1".into(),
                version: None,
            }]),
        };
        assert_eq!(
            auto.to_value(),
            json!({
                "type": "container_auto",
                "file_ids": ["f1"],
                "memory_limit": "4g",
                "network_policy": {"type": "allowlist", "allowed_domains": ["example.com"]},
                "skills": [{"type": "skill_reference", "skill_id": "s_1"}],
            })
        );
    }

    #[test]
    fn image_generation_config_keeps_its_type() {
        let config = ImageGenerationToolConfig {
            quality: Some("high".into()),
            ..Default::default()
        };
        assert_eq!(
            config.to_value(),
            json!({"type": "image_generation", "quality": "high"})
        );
    }
}
