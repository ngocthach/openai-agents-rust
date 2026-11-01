//! Execution of local runtime tool calls: computer, shell, local shell, apply patch and custom
//! tools. Each call produces a Responses `*_output` item or an approval interruption.

use std::sync::Arc;

use serde_json::{Map, Value, json};

use super::tools::{
    Signal, ToolBatchFailure, ToolEnv, ToolExecResult, execute_function_tools,
    format_tool_error_for, post_invoke_wait,
};
use crate::editor::{ApplyPatchOperationType, ApplyPatchStatus};
use crate::error::{AgentsError, Result};
use crate::items::{FunctionCall, ToolApprovalItem};
use crate::run_config::ToolErrorKind;
use crate::tool::FunctionTool;
use crate::tool::{
    ApplyPatchTool, ApplyPatchToolCustomDataContext, ComputerTool, ComputerToolCustomDataContext,
    ComputerToolSafetyCheckData, CustomTool, CustomToolCustomDataContext,
    DEFAULT_APPROVAL_REJECTION_MESSAGE, LocalShellCommandRequest, LocalShellTool, OnApprovalFn,
    ShellCallData, ShellCommandRequest, ShellExecutorOutput, ShellTool, Tool, ToolOutput,
    normalize_custom_data, normalize_shell_output_entries, parse_apply_patch_operations,
    render_shell_outputs, truncate_shell_outputs,
};
use crate::tool_context::ToolContext;
use crate::tracing::{Span, SpanData, create_span, scope};

/// Log a tool failure, keeping the error text out unless tool data logging is enabled.
fn log_tool_error(label: &str, msg: &str) {
    if crate::debug::dont_log_tool_data() {
        log::error!(target: crate::logger::LOG_TARGET, "{label}");
    } else {
        log::error!(target: crate::logger::LOG_TARGET, "{label}: {msg}");
    }
}

/// Record a failed local tool on `span` with the tool name, as Python's `SpanError` for
/// shell, apply_patch and custom tools carries it. The error text is redacted unless sensitive
/// trace data is enabled.
fn record_local_tool_error(span: &Span, tool_name: &str, message: &str, include_sensitive: bool) {
    let text = if include_sensitive {
        message.to_string()
    } else {
        crate::tracing::REDACTED_TOOL_ERROR_MESSAGE.to_string()
    };
    span.set_error(
        "Error running tool",
        Some(json!({"tool_name": tool_name, "error": text})),
    );
}

/// A local tool call found in a model response.
pub(crate) enum LocalCall<C> {
    Computer(Value, Arc<ComputerTool<C>>),
    Shell(Value, Arc<ShellTool<C>>),
    LocalShell(Value, Arc<LocalShellTool<C>>),
    ApplyPatch(Value, Arc<ApplyPatchTool<C>>),
    Custom(Value, Arc<CustomTool<C>>),
}

impl<C> LocalCall<C> {
    /// The call item as recorded and executed (sanitized, or normalized for a fallback).
    pub(crate) fn raw(&self) -> &Value {
        match self {
            LocalCall::Computer(v, _)
            | LocalCall::Shell(v, _)
            | LocalCall::LocalShell(v, _)
            | LocalCall::ApplyPatch(v, _)
            | LocalCall::Custom(v, _) => v,
        }
    }

    pub(crate) fn tool_name(&self) -> String {
        match self {
            LocalCall::Computer(_, t) => t.name().to_string(),
            LocalCall::Shell(_, t) => t.name.clone(),
            LocalCall::LocalShell(_, t) => t.name().to_string(),
            LocalCall::ApplyPatch(_, t) => t.name.clone(),
            LocalCall::Custom(_, t) => t.name.clone(),
        }
    }
}

/// Outcome of one local tool call.
pub(crate) enum LocalOutcome {
    Output {
        raw_item: Value,
        output: Value,
        /// SDK-only data a tool's `custom_data_extractor` attached to the output item.
        custom_data: Option<Value>,
    },
    Interruption(ToolApprovalItem),
    /// The call was stopped by a failed sibling: before it ran (nothing to record), or while it
    /// finalized (its committed output, if any, is in the call's [`NativeCtl`]).
    Stopped,
    /// The tool ran, but a step after its side effect failed. The output is committed before
    /// the error propagates, so a retry does not repeat the side effect.
    OutputThenError {
        raw_item: Value,
        output: Value,
        error: AgentsError,
    },
}

pub(crate) fn call_id_of(raw: &Value) -> String {
    raw.get("call_id")
        .or_else(|| raw.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Represent a non-function tool call as a [`FunctionCall`] for approvals and run state: the
/// original item fields are kept in `extra`, so `type` stays e.g. `shell_call`.
pub(crate) fn approval_call(raw: &Value, tool_name: &str, arguments: String) -> FunctionCall {
    let mut extra: Map<String, Value> = raw.as_object().cloned().unwrap_or_default();
    let id = extra
        .remove("id")
        .and_then(|v| v.as_str().map(str::to_string));
    let call_id = extra
        .remove("call_id")
        .and_then(|v| v.as_str().map(str::to_string))
        .or_else(|| id.clone())
        .unwrap_or_default();
    let status = extra
        .remove("status")
        .and_then(|v| v.as_str().map(str::to_string));
    extra.remove("name");
    extra.remove("arguments");
    FunctionCall {
        id,
        call_id,
        name: tool_name.to_string(),
        arguments,
        status,
        extra,
    }
}

fn copy_caller(call: &Value, output: &mut Value) {
    if let Some(caller) = call.get("caller").filter(|c| !c.is_null()) {
        output["caller"] = caller.clone();
    }
}

enum Gate {
    Proceed,
    Pending(Box<ToolApprovalItem>),
    Rejected(String),
}

/// Resolve the approval state of a call: existing decisions win, then the `needs_approval` rule,
/// then the `on_approval` handler.
async fn approval_gate<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    tool_type: &str,
    tool_name: &str,
    call: FunctionCall,
    needs_approval: impl std::future::Future<Output = bool>,
    on_approval: Option<&OnApprovalFn<C>>,
) -> Result<Gate> {
    let agent = &env.agent_key.to_string();
    let call_id = call.call_id.clone();
    // Sticky decisions are scoped by how the tool was invoked, so a native `apply_patch_call`
    // approval never approves a custom tool of the same name.
    let tool_key = crate::run_context::approval_tool_key(tool_name, &call);
    let mut status = super::tools::bound_approval_status(
        env.ctx,
        env.unbound_approvals,
        agent,
        &tool_key,
        &call_id,
    );
    if status.is_none() {
        let needs = needs_approval.await;
        // A decision recorded while the rule ran wins over the rule, but only one recorded
        // for this very call: a decision about changed content under the same call id is a
        // reused call id. Port of the binding check in `_tool_invocation_status`.
        let current = crate::run_context::approval_binding(&ToolApprovalItem {
            agent: agent.clone(),
            raw_item: call.clone(),
            tool_name: tool_name.to_string(),
            tool_origin: None,
        });
        if let (Some((_, current)), Some(bound)) = (current, env.ctx.bound_invocation(&call_id))
            && (bound.invocation_type != current.invocation_type
                || bound.approval_scope != current.approval_scope
                || bound.fingerprint != current.fingerprint)
        {
            return Err(AgentsError::model_behavior(
                "Model reused a tool call ID for a different invocation. \
                 Use a unique call ID for each tool invocation.",
            ));
        }
        status = super::tools::bound_approval_status(
            env.ctx,
            env.unbound_approvals,
            agent,
            &tool_key,
            &call_id,
        );
        if status.is_none() && !needs {
            return Ok(Gate::Proceed);
        }
    }
    if status.is_none() {
        let item = ToolApprovalItem {
            agent: agent.clone(),
            raw_item: call,
            tool_name: tool_name.to_string(),
            tool_origin: None,
        };
        let mut decided = false;
        if let Some(handler) = on_approval
            && let Some(decision) = handler(env.ctx.clone(), item.clone()).await
        {
            decided = true;
            if decision.approve {
                env.ctx.approve_tool(&item, false);
            } else {
                env.ctx
                    .reject_tool(&item, false, decision.reason.filter(|r| !r.is_empty()));
            }
        }
        // A decision the handler just made is bound to this very call, so it is read without
        // the restored-binding mask. Without a fresh decision the call stays undecided, which
        // is what the mask means. Mirrors the `discard(call_id)` that ends Python's
        // `_apply_approval_decision`.
        status = decided
            .then(|| env.ctx.is_tool_approved(agent, &tool_key, &call_id))
            .flatten();
        if status.is_none() {
            return Ok(Gate::Pending(Box::new(item)));
        }
    }
    match status {
        Some(false) => {
            let message = match env.ctx.rejection_message(agent, &tool_key, &call_id) {
                Some(message) => message,
                None => {
                    format_tool_error_for(
                        env.config,
                        env.ctx,
                        ToolErrorKind::ApprovalRejected,
                        tool_type,
                        tool_name,
                        &call_id,
                        DEFAULT_APPROVAL_REJECTION_MESSAGE.to_string(),
                        None,
                    )
                    .await
                }
            };
            Ok(Gate::Rejected(message))
        }
        _ => Ok(Gate::Proceed),
    }
}

fn tool_span(env: &ToolEnv<'_, impl Sized>, name: &str, input: Option<String>) -> Span {
    let include = env.config.trace_include_sensitive_data;
    create_span(
        SpanData::Function {
            name: name.to_string(),
            input: input.filter(|_| include),
            output: None,
            mcp_data: None,
        },
        None,
        false,
    )
}

fn finish_span(env: &ToolEnv<'_, impl Sized>, span: &Span, output: &str) {
    if env.config.trace_include_sensitive_data {
        let text = output.to_string();
        span.update_data(|d| {
            if let SpanData::Function { output, .. } = d {
                *output = Some(text);
            }
        });
    }
}

async fn start_hooks<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    ctx: &ToolContext<C>,
    tool: &Tool<C>,
) -> Result<()> {
    // From here on user code runs: a retry must not run it again.
    env.ctx.mark_invocation_executed(&ctx.tool_call_id);
    let run_hook = async {
        match env.hooks {
            Some(h) => h.try_on_tool_start(ctx, env.agent, tool).await,
            None => Ok(()),
        }
    };
    let agent_hook = async {
        match &env.agent.hooks {
            Some(h) => h.try_on_tool_start(ctx, env.agent, tool).await,
            None => Ok(()),
        }
    };
    let (run_hook, agent_hook) = futures::join!(run_hook, agent_hook);
    run_hook.and(agent_hook)
}

async fn end_hooks<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    ctx: &ToolContext<C>,
    tool: &Tool<C>,
    output: &str,
) -> Result<()> {
    let out = ToolOutput::Text(output.to_string());
    let run_hook = async {
        match env.hooks {
            Some(h) => h.try_on_tool_end(ctx, env.agent, tool, &out).await,
            None => Ok(()),
        }
    };
    let agent_hook = async {
        match &env.agent.hooks {
            Some(h) => h.try_on_tool_end(ctx, env.agent, tool, &out).await,
            None => Ok(()),
        }
    };
    let (run_hook, agent_hook) = futures::join!(run_hook, agent_hook);
    run_hook.and(agent_hook)
}

/// Execute one local tool call.
pub(crate) async fn execute_local_call<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    call: LocalCall<C>,
    ctl: &NativeCtl<'_>,
) -> Result<LocalOutcome> {
    let mut run = Box::pin(async {
        match call {
            LocalCall::Computer(raw, tool) => Box::pin(run_computer(env, ctl, raw, tool)).await,
            LocalCall::Shell(raw, tool) => Box::pin(run_shell(env, ctl, raw, tool)).await,
            LocalCall::LocalShell(raw, tool) => {
                Box::pin(run_local_shell(env, ctl, raw, tool)).await
            }
            LocalCall::ApplyPatch(raw, tool) => {
                Box::pin(run_apply_patch(env, ctl, raw, tool)).await
            }
            LocalCall::Custom(raw, tool) => Box::pin(run_custom(env, ctl, raw, tool)).await,
        }
    });
    let Some(stop) = ctl.stop else {
        return run.await;
    };
    // A failed sibling cancels a call that is still invoking; one that already produced its
    // output finalizes under its own bounded wait (`NativeCtl::post_invoke`).
    tokio::select! {
        biased;
        r = &mut run => r,
        _ = async {
            stop.wait().await;
            if ctl.in_post_invoke() {
                std::future::pending::<()>().await;
            }
        } => Ok(LocalOutcome::Stopped),
    }
}

/// Per-call state of a native tool call: the sibling failure signal it observes and the output
/// it committed before finalizing.
pub(crate) struct NativeCtl<'a> {
    stop: Option<&'a Signal>,
    wait: std::time::Duration,
    in_post: std::sync::atomic::AtomicBool,
    committed: std::sync::Mutex<Option<(Value, Value)>>,
}

impl<'a> NativeCtl<'a> {
    pub(crate) fn new(stop: Option<&'a Signal>, wait: std::time::Duration) -> Self {
        NativeCtl {
            stop,
            wait,
            in_post: std::sync::atomic::AtomicBool::new(false),
            committed: std::sync::Mutex::new(None),
        }
    }

    fn in_post_invoke(&self) -> bool {
        self.in_post.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Record the accepted output before its finalization (custom data, end hooks) runs.
    fn commit(&self, raw_item: &Value, output: &Value) {
        *self.committed.lock().expect("commit slot") = Some((raw_item.clone(), output.clone()));
    }

    /// The output committed by a call that was stopped while finalizing.
    pub(crate) fn take_committed(&self) -> Option<(Value, Value)> {
        self.committed.lock().expect("commit slot").take()
    }

    /// Run the finalization of a call that already produced its output. When a sibling fails
    /// meanwhile it gets a bounded wait; `None` means it was abandoned.
    async fn post_invoke<T>(&self, post: impl std::future::Future<Output = T>) -> Option<T> {
        self.in_post
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let Some(stop) = self.stop else {
            return Some(post.await);
        };
        let mut post = std::pin::pin!(post);
        tokio::select! {
            biased;
            r = &mut post => Some(r),
            _ = stop.wait() => tokio::time::timeout(self.wait, &mut post).await.ok(),
        }
    }
}

fn int(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0) as i64
}

fn modifier_keys(action: &Value) -> Option<Vec<String>> {
    action.get("keys").and_then(Value::as_array).map(|a| {
        a.iter()
            .filter_map(|k| k.as_str().map(str::to_string))
            .collect()
    })
}

async fn run_computer_actions(
    computer: &dyn crate::computer::Computer,
    raw: &Value,
) -> Result<String> {
    let actions: Vec<Value> = match raw.get("actions").and_then(Value::as_array) {
        Some(a) if !a.is_empty() => a.clone(),
        _ => raw
            .get("action")
            .filter(|a| !a.is_null())
            .cloned()
            .into_iter()
            .collect(),
    };
    let mut last_screenshot: Option<String> = None;
    for action in &actions {
        last_screenshot = None;
        let keys = modifier_keys(action);
        let keys = keys.as_deref();
        match action
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "click" => {
                let button = action
                    .get("button")
                    .and_then(Value::as_str)
                    .unwrap_or("left");
                computer
                    .click(int(action, "x"), int(action, "y"), button, keys)
                    .await?
            }
            "double_click" => {
                computer
                    .double_click(int(action, "x"), int(action, "y"), keys)
                    .await?
            }
            "drag" => {
                let path: Vec<(i64, i64)> = action
                    .get("path")
                    .and_then(Value::as_array)
                    .map(|p| p.iter().map(|pt| (int(pt, "x"), int(pt, "y"))).collect())
                    .unwrap_or_default();
                computer.drag(&path, keys).await?
            }
            "keypress" => computer.keypress(keys.unwrap_or_default()).await?,
            "move" => {
                computer
                    .move_mouse(int(action, "x"), int(action, "y"), keys)
                    .await?
            }
            "screenshot" => last_screenshot = Some(computer.screenshot().await?),
            "scroll" => {
                computer
                    .scroll(
                        int(action, "x"),
                        int(action, "y"),
                        int(action, "scroll_x"),
                        int(action, "scroll_y"),
                        keys,
                    )
                    .await?
            }
            "type" => {
                let text = action.get("text").and_then(Value::as_str).unwrap_or("");
                computer.type_text(text).await?
            }
            "wait" => computer.wait().await?,
            other => {
                return Err(AgentsError::model_behavior(format!(
                    "Computer tool returned unknown action type '{other}'"
                )));
            }
        }
    }
    match last_screenshot {
        Some(s) => Ok(s),
        None => computer.screenshot().await,
    }
}

async fn run_computer<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    ctl: &NativeCtl<'_>,
    raw: Value,
    tool: Arc<ComputerTool<C>>,
) -> Result<LocalOutcome> {
    let call_id = call_id_of(&raw);
    let pending: Vec<Value> = raw
        .get("pending_safety_checks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut acknowledged: Option<Vec<Value>> = None;
    if !pending.is_empty() {
        match &tool.on_safety_check {
            Some(cb) => {
                let mut acks = Vec::new();
                for check in &pending {
                    let ok = cb(ComputerToolSafetyCheckData {
                        ctx: env.ctx.clone(),
                        agent_name: env.agent.name.clone(),
                        tool_call: raw.clone(),
                        safety_check: check.clone(),
                    })
                    .await;
                    if !ok {
                        return Err(AgentsError::user(
                            "Computer tool safety check was not acknowledged",
                        ));
                    }
                    acks.push(json!({
                        "id": check.get("id"),
                        "code": check.get("code"),
                        "message": check.get("message"),
                    }));
                }
                acknowledged = Some(acks);
            }
            None => log::warn!(target: crate::logger::LOG_TARGET,
                "Computer call has pending safety checks, but no on_safety_check handler is \
                 configured. The action will proceed without acknowledging the checks. \
                 Configure ComputerTool.on_safety_check to review or reject them."
            ),
        }
    }

    let input = raw
        .get("actions")
        .or_else(|| raw.get("action"))
        .map(Value::to_string);
    let span = tool_span(env, "computer", input);
    span.start();
    let tool_enum = Tool::Computer(tool.clone());
    let fc = approval_call(&raw, tool.name(), String::new());
    let tool_ctx = ToolContext::from_call(env.ctx.clone(), &env.agent.name, &fc)
        .with_agent(env.agent.clone())
        .with_run_config(env.config.clone());
    let invoke = async {
        let computer = tool.resolve(env.ctx).await?;
        start_hooks(env, &tool_ctx, &tool_enum).await?;
        run_computer_actions(computer.as_ref(), &raw).await
    };
    let shot = match scope::with_span(span.clone(), Box::pin(invoke)).await {
        Ok(s) => s,
        Err(e) => {
            let msg = e.to_string();
            record_local_tool_error(
                &span,
                "computer",
                &msg,
                env.config.trace_include_sensitive_data,
            );
            log_tool_error("Failed to execute computer action", &msg);
            span.finish();
            return Err(e);
        }
    };
    let image_url = if shot.is_empty() {
        String::new()
    } else {
        format!("data:image/png;base64,{shot}")
    };
    let mut raw_item = json!({
        "type": "computer_call_output",
        "call_id": call_id,
        "output": {"type": "computer_screenshot", "image_url": image_url},
    });
    if let Some(acks) = acknowledged {
        raw_item["acknowledged_safety_checks"] = Value::Array(acks);
    }
    ctl.commit(&raw_item, &Value::String(image_url.clone()));
    let post = async {
        let custom_data = match &tool.custom_data_extractor {
            Some(extractor) => {
                let ctx = ComputerToolCustomDataContext {
                    run_context: env.ctx,
                    tool: &tool,
                    tool_call: &raw,
                    output: &image_url,
                    raw_item: &raw_item,
                };
                normalize_custom_data(extractor(&ctx).await)
            }
            None => Ok(None),
        };
        if custom_data.is_ok()
            && let Err(e) = end_hooks(env, &tool_ctx, &tool_enum, &shot).await
        {
            return Err(e);
        }
        custom_data
    };
    let Some(custom_data) = scope::with_span(span.clone(), Box::pin(ctl.post_invoke(post))).await
    else {
        span.finish();
        return Ok(LocalOutcome::Stopped);
    };
    finish_span(env, &span, &image_url);
    span.finish();
    match custom_data {
        Ok(custom_data) => Ok(LocalOutcome::Output {
            raw_item,
            output: Value::String(image_url),
            custom_data,
        }),
        Err(error) => Ok(LocalOutcome::OutputThenError {
            raw_item,
            output: Value::String(image_url),
            error,
        }),
    }
}

fn shell_rejection(call: &Value, call_id: &str, message: String) -> LocalOutcome {
    let mut raw_item = json!({
        "type": "shell_call_output",
        "call_id": call_id,
        "output": [{"stdout": "", "stderr": message, "outcome": {"type": "exit", "exit_code": 1}}],
    });
    copy_caller(call, &mut raw_item);
    LocalOutcome::Output {
        raw_item,
        output: Value::String(message),
        custom_data: None,
    }
}

async fn run_shell<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    ctl: &NativeCtl<'_>,
    raw: Value,
    tool: Arc<ShellTool<C>>,
) -> Result<LocalOutcome> {
    let data = ShellCallData::from_raw(&raw)?;
    let call_id = data.call_id.clone();
    let action_json = serde_json::to_string(&data.action)?;
    let fc = approval_call(&raw, &tool.name, action_json.clone());
    let gate = approval_gate(
        env,
        "shell",
        &tool.name,
        fc.clone(),
        tool.needs_approval
            .evaluate(env.ctx, data.action.clone(), &call_id),
        tool.on_approval.as_ref(),
    )
    .await?;
    match gate {
        Gate::Pending(item) => return Ok(LocalOutcome::Interruption(*item)),
        Gate::Rejected(message) => return Ok(shell_rejection(&raw, &call_id, message)),
        Gate::Proceed => {}
    }

    let span = tool_span(env, &tool.name, Some(action_json));
    span.start();
    let tool_enum = Tool::Shell(tool.clone());
    let tool_ctx = ToolContext::from_call(env.ctx.clone(), &env.agent.name, &fc)
        .with_agent(env.agent.clone())
        .with_run_config(env.config.clone());
    let requested_max = data.action.max_output_length;
    let run = async {
        start_hooks(env, &tool_ctx, &tool_enum).await?;
        let mut status = "completed";
        let mut max_output_length: Option<usize> = None;
        let mut structured: Option<Vec<Value>> = None;
        let mut provider_meta: Option<Value> = None;
        let result = match &tool.executor {
            Some(exec) => {
                exec(ShellCommandRequest {
                    ctx: env.ctx.clone(),
                    data: data.clone(),
                })
                .await
            }
            None => Err(AgentsError::model_behavior(
                "Shell tool has no local executor configured.",
            )),
        };
        let output_text = match result {
            Ok(ShellExecutorOutput::Structured(r)) => {
                max_output_length = match (r.max_output_length, requested_max) {
                    (None, req) => req,
                    (Some(res), None) => Some(res),
                    (Some(res), Some(req)) => Some(res.min(req)),
                };
                let mut entries = r.output.clone();
                if let Some(max) = max_output_length {
                    entries = truncate_shell_outputs(&entries, max);
                }
                let mut text = render_shell_outputs(&entries);
                if let Some(max) = max_output_length {
                    text = text.chars().take(max).collect();
                }
                structured = Some(entries.iter().map(|e| e.to_value()).collect());
                provider_meta = r.provider_data.filter(|p| !p.is_null());
                text
            }
            Ok(ShellExecutorOutput::Text(t)) => match requested_max {
                Some(max) => {
                    max_output_length = Some(max);
                    t.chars().take(max).collect()
                }
                None => t,
            },
            Err(e) => {
                status = "failed";
                let msg = e.to_string();
                if let Some(s) = scope::current_span() {
                    record_local_tool_error(
                        &s,
                        &tool.name,
                        &msg,
                        env.config.trace_include_sensitive_data,
                    );
                }
                log_tool_error("Shell executor failed", &msg);
                match requested_max {
                    Some(max) => {
                        max_output_length = Some(max);
                        msg.chars().take(max).collect()
                    }
                    None => msg,
                }
            }
        };
        let raw_entries: Option<Vec<Value>> = structured.or_else(|| {
            (!output_text.is_empty()).then(|| {
                vec![json!({
                    "stdout": output_text,
                    "stderr": "",
                    "status": status,
                    "outcome": if status == "completed" { "success" } else { "failure" },
                })]
            })
        });
        let structured_output = raw_entries
            .as_deref()
            .map(normalize_shell_output_entries)
            .unwrap_or_default();
        let mut raw_item = json!({
            "type": "shell_call_output",
            "call_id": call_id,
            "output": structured_output,
            "status": status,
        });
        copy_caller(&raw, &mut raw_item);
        if let Some(max) = max_output_length {
            raw_item["max_output_length"] = json!(max);
        }
        if let Some(entries) = raw_entries.filter(|e| !e.is_empty()) {
            raw_item["shell_output"] = Value::Array(entries);
        }
        if let Some(p) = provider_meta {
            raw_item["provider_data"] = p;
        }
        Ok::<_, AgentsError>((raw_item, output_text))
    };
    let (raw_item, output_text) = match scope::with_span(span.clone(), Box::pin(run)).await {
        Ok(done) => done,
        Err(e) => {
            span.finish();
            return Err(e);
        }
    };
    ctl.commit(&raw_item, &Value::String(output_text.clone()));
    let post = end_hooks(env, &tool_ctx, &tool_enum, &output_text);
    match scope::with_span(span.clone(), Box::pin(ctl.post_invoke(post))).await {
        None => {
            span.finish();
            return Ok(LocalOutcome::Stopped);
        }
        Some(Err(e)) => {
            span.finish();
            return Err(e);
        }
        Some(Ok(())) => {}
    }
    finish_span(env, &span, &output_text);
    span.finish();
    Ok(LocalOutcome::Output {
        raw_item,
        output: Value::String(output_text),
        custom_data: None,
    })
}

async fn run_local_shell<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    ctl: &NativeCtl<'_>,
    raw: Value,
    tool: Arc<LocalShellTool<C>>,
) -> Result<LocalOutcome> {
    let call_id = call_id_of(&raw);
    let fc = approval_call(
        &raw,
        tool.name(),
        raw.get("action").map(Value::to_string).unwrap_or_default(),
    );
    let tool_ctx = ToolContext::from_call(env.ctx.clone(), &env.agent.name, &fc)
        .with_agent(env.agent.clone())
        .with_run_config(env.config.clone());
    let tool_enum = Tool::LocalShell(tool.clone());
    start_hooks(env, &tool_ctx, &tool_enum).await?;
    let result = (tool.executor)(LocalShellCommandRequest {
        ctx: env.ctx.clone(),
        data: raw.clone(),
    })
    .await?;
    let raw_item = json!({"type": "local_shell_call_output", "call_id": call_id, "output": result});
    ctl.commit(&raw_item, &Value::String(result.clone()));
    match ctl
        .post_invoke(end_hooks(env, &tool_ctx, &tool_enum, &result))
        .await
    {
        None => return Ok(LocalOutcome::Stopped),
        Some(Err(e)) => return Err(e),
        Some(Ok(())) => {}
    }
    Ok(LocalOutcome::Output {
        raw_item,
        output: Value::String(result),
        custom_data: None,
    })
}

async fn run_apply_patch<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    ctl: &NativeCtl<'_>,
    raw: Value,
    tool: Arc<ApplyPatchTool<C>>,
) -> Result<LocalOutcome> {
    let call_id = call_id_of(&raw);
    if call_id.is_empty() {
        return Err(AgentsError::model_behavior(
            "Apply patch call is missing call_id.",
        ));
    }
    let operations = parse_apply_patch_operations(&raw)?;
    let ops_json = serde_json::to_string(&operations)?;
    let fc = approval_call(&raw, &tool.name, ops_json.clone());
    // Once a decision lands for the call, the remaining operations are not checked: the
    // decision governs the whole call.
    let tool_key = crate::run_context::approval_tool_key(&tool.name, &fc);
    let needs = async {
        for op in &operations {
            if tool
                .needs_approval
                .evaluate(env.ctx, op.clone(), &call_id)
                .await
            {
                return true;
            }
            if super::tools::bound_approval_status(
                env.ctx,
                env.unbound_approvals,
                &env.agent.name,
                &tool_key,
                &call_id,
            )
            .is_some()
            {
                return false;
            }
        }
        false
    };
    let gate = approval_gate(
        env,
        "apply_patch",
        &tool.name,
        fc.clone(),
        needs,
        tool.on_approval.as_ref(),
    )
    .await?;
    match gate {
        Gate::Pending(item) => return Ok(LocalOutcome::Interruption(*item)),
        Gate::Rejected(message) => {
            let mut raw_item = json!({
                "type": "apply_patch_call_output",
                "call_id": call_id,
                "output": message,
                "status": "failed",
            });
            copy_caller(&raw, &mut raw_item);
            return Ok(LocalOutcome::Output {
                raw_item,
                output: Value::String(message),
                custom_data: None,
            });
        }
        Gate::Proceed => {}
    }

    let span = tool_span(env, &tool.name, Some(ops_json));
    span.start();
    let tool_enum = Tool::ApplyPatch(tool.clone());
    let tool_ctx = ToolContext::from_call(env.ctx.clone(), &env.agent.name, &fc)
        .with_agent(env.agent.clone())
        .with_run_config(env.config.clone());
    let run = async {
        start_hooks(env, &tool_ctx, &tool_enum).await?;
        let mut status = ApplyPatchStatus::Completed;
        let applied = async {
            let mut outputs = Vec::new();
            for op in &operations {
                let r = match op.op_type {
                    ApplyPatchOperationType::CreateFile => tool.editor.create_file(op).await?,
                    ApplyPatchOperationType::UpdateFile => tool.editor.update_file(op).await?,
                    ApplyPatchOperationType::DeleteFile => tool.editor.delete_file(op).await?,
                };
                if r.status == Some(ApplyPatchStatus::Failed) {
                    status = ApplyPatchStatus::Failed;
                }
                if let Some(o) = r.output.filter(|o| !o.is_empty()) {
                    outputs.push(o);
                }
            }
            Ok::<_, AgentsError>(outputs.join("\n"))
        }
        .await;
        let output_text = match applied {
            Ok(t) => t,
            Err(e) => {
                status = ApplyPatchStatus::Failed;
                let msg = e.to_string();
                if let Some(s) = scope::current_span() {
                    record_local_tool_error(
                        &s,
                        &tool.name,
                        &msg,
                        env.config.trace_include_sensitive_data,
                    );
                }
                log_tool_error("Apply patch editor failed", &msg);
                msg
            }
        };
        let mut raw_item = json!({
            "type": "apply_patch_call_output",
            "call_id": call_id,
            "status": if status == ApplyPatchStatus::Failed { "failed" } else { "completed" },
        });
        copy_caller(&raw, &mut raw_item);
        if !output_text.is_empty() {
            raw_item["output"] = json!(output_text);
        }
        Ok::<_, AgentsError>((raw_item, output_text, status))
    };
    let (raw_item, output_text, status) = match scope::with_span(span.clone(), Box::pin(run)).await
    {
        Ok(done) => done,
        Err(e) => {
            span.finish();
            return Err(e);
        }
    };
    ctl.commit(&raw_item, &Value::String(output_text.clone()));
    let post = async {
        let custom_data = match &tool.custom_data_extractor {
            Some(extractor) => {
                let ctx = ApplyPatchToolCustomDataContext {
                    run_context: env.ctx,
                    tool: &tool,
                    operations: &operations,
                    output: &output_text,
                    status,
                    raw_item: &raw_item,
                };
                normalize_custom_data(extractor(&ctx).await)
            }
            None => Ok(None),
        };
        if custom_data.is_ok()
            && let Err(e) = end_hooks(env, &tool_ctx, &tool_enum, &output_text).await
        {
            return Err(e);
        }
        custom_data
    };
    let Some(custom_data) = scope::with_span(span.clone(), Box::pin(ctl.post_invoke(post))).await
    else {
        span.finish();
        return Ok(LocalOutcome::Stopped);
    };
    finish_span(env, &span, &output_text);
    span.finish();
    // The patch is applied: a failed custom data extraction keeps its output committed.
    match custom_data {
        Ok(custom_data) => Ok(LocalOutcome::Output {
            raw_item,
            output: Value::String(output_text),
            custom_data,
        }),
        Err(error) => Ok(LocalOutcome::OutputThenError {
            raw_item,
            output: Value::String(output_text),
            error,
        }),
    }
}

async fn run_custom<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    ctl: &NativeCtl<'_>,
    raw: Value,
    tool: Arc<CustomTool<C>>,
) -> Result<LocalOutcome> {
    let call_id = raw
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            AgentsError::model_behavior("Custom tool call is missing a non-empty call_id.")
        })?
        .to_string();
    let input = raw
        .get("input")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentsError::model_behavior("Custom tool call is missing input."))?
        .to_string();
    let fc = approval_call(&raw, &tool.name, input.clone());
    let output_item = |text: String| {
        let mut raw_item =
            json!({"type": "custom_tool_call_output", "call_id": call_id, "output": text});
        copy_caller(&raw, &mut raw_item);
        LocalOutcome::Output {
            raw_item,
            output: Value::String(text),
            custom_data: None,
        }
    };
    let gate = approval_gate(
        env,
        "custom",
        &tool.name,
        fc.clone(),
        tool.needs_approval
            .evaluate(env.ctx, input.clone(), &call_id),
        tool.on_approval.as_ref(),
    )
    .await?;
    match gate {
        Gate::Pending(item) => return Ok(LocalOutcome::Interruption(*item)),
        Gate::Rejected(message) => return Ok(output_item(message)),
        Gate::Proceed => {}
    }

    let span = tool_span(env, &tool.name, Some(input.clone()));
    span.start();
    let tool_enum = Tool::Custom(tool.clone());
    let tool_ctx = ToolContext::from_call(env.ctx.clone(), &env.agent.name, &fc)
        .with_agent(env.agent.clone())
        .with_run_config(env.config.clone());
    let run = async {
        start_hooks(env, &tool_ctx, &tool_enum).await?;
        Ok::<_, AgentsError>(match tool.invoke(tool_ctx.clone(), input.clone()).await {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                if let Some(s) = scope::current_span() {
                    record_local_tool_error(
                        &s,
                        &tool.name,
                        &msg,
                        env.config.trace_include_sensitive_data,
                    );
                }
                log_tool_error("Custom tool failed", &msg);
                msg
            }
        })
    };
    let text = match scope::with_span(span.clone(), Box::pin(run)).await {
        Ok(text) => text,
        Err(e) => {
            span.finish();
            return Err(e);
        }
    };
    let mut outcome = output_item(text.clone());
    let LocalOutcome::Output {
        raw_item,
        custom_data,
        ..
    } = &mut outcome
    else {
        unreachable!("output_item builds an output")
    };
    ctl.commit(raw_item, &Value::String(text.clone()));
    let post = async {
        let extracted = match &tool.custom_data_extractor {
            Some(extractor) => {
                let ctx = CustomToolCustomDataContext {
                    tool_context: &tool_ctx,
                    tool: &tool,
                    input: &input,
                    output: &text,
                    raw_item,
                };
                normalize_custom_data(extractor(&ctx).await)
            }
            None => Ok(None),
        };
        if extracted.is_ok()
            && let Err(e) = end_hooks(env, &tool_ctx, &tool_enum, &text).await
        {
            return Err(e);
        }
        extracted
    };
    let Some(extracted) = scope::with_span(span.clone(), Box::pin(ctl.post_invoke(post))).await
    else {
        span.finish();
        return Ok(LocalOutcome::Stopped);
    };
    finish_span(env, &span, &text);
    span.finish();
    match extracted {
        Ok(data) => *custom_data = data,
        Err(error) => {
            // The tool ran: a failed custom data extraction keeps its output committed.
            let LocalOutcome::Output {
                raw_item, output, ..
            } = outcome
            else {
                unreachable!("matched above")
            };
            return Ok(LocalOutcome::OutputThenError {
                raw_item,
                output,
                error,
            });
        }
    }
    Ok(outcome)
}

/// Which tool category selected the batch's failure first.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FailureSource {
    Function,
    Native,
}

/// The settled result of one turn's tool batch.
pub(crate) struct ToolBatchOutcome {
    pub functions: std::result::Result<Vec<ToolExecResult>, ToolBatchFailure>,
    /// Native outputs and approvals as `(call index, outcome)`, in call order. A call stopped
    /// while it finalized contributes the output it had committed.
    pub locals: Vec<(usize, LocalOutcome)>,
    /// The failure of a native call (a failed finalization keeps its output in `locals`).
    pub native_error: Option<AgentsError>,
    /// The native failure was selected before the function batch's own.
    native_first: bool,
}

impl ToolBatchOutcome {
    /// The error that fails the turn, if any: the category that failed first wins.
    pub(crate) fn take_error(&mut self) -> Option<AgentsError> {
        let function_error = match &mut self.functions {
            Err(failure) if !failure.sibling_stopped => {
                Some(std::mem::replace(&mut failure.error, AgentsError::user("")))
            }
            _ => None,
        };
        match (function_error, self.native_error.take()) {
            (Some(f), Some(n)) => Some(if self.native_first { n } else { f }),
            (f, n) => f.or(n),
        }
    }

    /// Whether any call of the batch committed an output.
    pub(crate) fn has_outputs(&self) -> bool {
        let function_outputs = match &self.functions {
            Ok(done) => !done.is_empty(),
            Err(failure) => !failure.completed.is_empty(),
        };
        function_outputs
            || self
                .locals
                .iter()
                .any(|(_, o)| matches!(o, LocalOutcome::Output { .. }))
    }
}

fn category_of<C>(call: &LocalCall<C>) -> usize {
    match call {
        LocalCall::Computer(..) => 0,
        LocalCall::Custom(..) => 1,
        LocalCall::Shell(..) => 2,
        LocalCall::ApplyPatch(..) => 3,
        LocalCall::LocalShell(..) => 4,
    }
}

/// Run a turn's function tools and native tools (computer, shell, apply patch, custom, local
/// shell) concurrently. Each native category runs its calls serially, the categories and the
/// function batch run side by side. Port of `_execute_tool_plan`.
///
/// When one category fails, the others are wound down: calls still invoking are cancelled,
/// queued native calls never start, and calls that already produced their output get a bounded
/// wait to finalize it. Outputs committed so far are always returned.
pub(crate) async fn run_tool_batch<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    functions: Vec<(FunctionCall, Arc<FunctionTool<C>>)>,
    locals: Vec<LocalCall<C>>,
    // Set as soon as any category selects its failure.
    error_selected: &Signal,
) -> ToolBatchOutcome {
    // Stops the function batch (a native category failed).
    let function_stop = Signal::new();
    // Stops queued and in-flight native calls (any category failed).
    let native_stop = Signal::new();
    let first: std::sync::Mutex<Option<FailureSource>> = std::sync::Mutex::new(None);
    let wait = post_invoke_wait(env.config);
    let record_first = |source: FailureSource| {
        first.lock().expect("first failure").get_or_insert(source);
        error_selected.set();
    };
    let on_function_selected = || {
        record_first(FailureSource::Function);
        native_stop.set();
    };

    let mut categories: [Vec<(usize, LocalCall<C>)>; 5] = Default::default();
    for (index, call) in locals.into_iter().enumerate() {
        categories[category_of(&call)].push((index, call));
    }
    let native_error: std::sync::Mutex<Option<AgentsError>> = std::sync::Mutex::new(None);
    let run_category = |calls: Vec<(usize, LocalCall<C>)>| {
        let (native_stop, function_stop) = (&native_stop, &function_stop);
        let (native_error, record_first) = (&native_error, &record_first);
        async move {
            let mut done: Vec<(usize, LocalOutcome)> = Vec::new();
            for (index, call) in calls {
                // A failure elsewhere is selected: queued calls never start.
                if native_stop.is_set() {
                    break;
                }
                let ctl = NativeCtl::new(Some(native_stop), wait);
                let result = Box::pin(execute_local_call(env, call, &ctl)).await;
                let fail = |error: AgentsError| {
                    record_first(FailureSource::Native);
                    native_error
                        .lock()
                        .expect("native error")
                        .get_or_insert(error);
                    native_stop.set();
                    function_stop.set();
                };
                match result {
                    Ok(LocalOutcome::Stopped) => {
                        if let Some((raw_item, output)) = ctl.take_committed() {
                            done.push((
                                index,
                                LocalOutcome::Output {
                                    raw_item,
                                    output,
                                    custom_data: None,
                                },
                            ));
                        }
                    }
                    Ok(LocalOutcome::OutputThenError {
                        raw_item,
                        output,
                        error,
                    }) => {
                        done.push((
                            index,
                            LocalOutcome::Output {
                                raw_item,
                                output,
                                custom_data: None,
                            },
                        ));
                        fail(error);
                    }
                    Ok(outcome) => done.push((index, outcome)),
                    Err(error) => fail(error),
                }
            }
            done
        }
    };
    let native = futures::future::join_all(categories.into_iter().map(run_category));
    let function_batch =
        execute_function_tools(env, functions, Some(&function_stop), &on_function_selected);
    let (functions, categories) = futures::future::join(function_batch, native).await;
    let mut locals: Vec<(usize, LocalOutcome)> = categories.into_iter().flatten().collect();
    locals.sort_by_key(|(index, _)| *index);
    let native_first = *first.lock().expect("first failure") == Some(FailureSource::Native);
    ToolBatchOutcome {
        functions,
        locals,
        native_error: native_error.into_inner().expect("native error"),
        native_first,
    }
}
