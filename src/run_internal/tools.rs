//! Function tool execution: approvals, tool guardrails, hooks, timeouts and error handling.

use std::collections::BTreeSet;
use std::sync::Arc;

use futures::StreamExt;
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::items::{FunctionCall, ItemHelpers, ToolApprovalItem};
use crate::lifecycle::RunHooks;
use crate::run_config::{RunConfig, ToolErrorFormatterArgs, ToolErrorKind};
use crate::run_context::RunContext;
use crate::tool::{
    DEFAULT_APPROVAL_REJECTION_MESSAGE, FunctionTool, Tool, ToolOrigin, ToolOutput,
    ToolTimeoutBehavior,
};
use crate::tool_context::ToolContext;
use crate::tool_guardrails::{
    ToolGuardrailBehavior, ToolInputGuardrailData, ToolInputGuardrailResult,
    ToolOutputGuardrailData, ToolOutputGuardrailResult,
};
use crate::tracing::{SpanData, create_span, scope};

/// Shared inputs for executing tools in one turn.
pub(crate) struct ToolEnv<'a, C> {
    pub ctx: &'a RunContext<C>,
    pub agent: &'a Arc<Agent<C>>,
    pub hooks: Option<&'a Arc<dyn RunHooks<C>>>,
    pub config: &'a Arc<RunConfig<C>>,
    /// Call ids whose restored approval decision authorizes nothing; see
    /// [`bound_approval_status`].
    pub unbound_approvals: &'a BTreeSet<String>,
    /// Call ids pending in the resumed snapshot; see `resumed_pending_approvals`.
    pub pending_approvals: &'a BTreeSet<String>,
}

/// The approval decision in force for `call_id`, honouring the restored binding rule.
///
/// Port of the `_restored_unbound_approval_call_ids` branch of
/// `RunContextWrapper._tool_invocation_status`: a decision restored from a snapshot that holds
/// no ledger entry binding it to a concrete invocation authorizes nothing, so the call reads as
/// undecided and goes back to the user for a fresh decision.
pub(crate) fn bound_approval_status<C>(
    ctx: &RunContext<C>,
    unbound: &BTreeSet<String>,
    agent: &str,
    tool_name: &str,
    call_id: &str,
) -> Option<bool> {
    if unbound.contains(call_id) {
        return None;
    }
    ctx.is_tool_approved(agent, tool_name, call_id)
}

/// The outcome of one function tool call.
pub(crate) enum ToolOutcome {
    /// The tool produced output (or a model-visible error message).
    Output { output: ToolOutput, item: Value },
    /// The call needs approval (possibly several, from a nested agent run).
    Interruptions(Vec<ToolApprovalItem>),
}

/// A tool call that failed, possibly after its output was already committed.
///
/// The output of a tool whose body succeeded is committed before the post-output step (the
/// custom-data merge) runs, so a failure there still hands back the output: a retry skips the
/// completed invocation instead of running the tool again.
pub(crate) struct ToolFailure {
    pub error: AgentsError,
    pub committed: Option<ToolExecResult>,
}

impl From<AgentsError> for ToolFailure {
    fn from(error: AgentsError) -> Self {
        ToolFailure {
            error,
            committed: None,
        }
    }
}

/// A batch of tool calls that failed, with the outputs that completed before the failure in
/// model call order.
pub(crate) struct ToolBatchFailure {
    pub error: AgentsError,
    pub completed: Vec<ToolExecResult>,
    /// Whether the selected failure's own output was committed before it failed (a
    /// post-output step), rather than the call failing without output.
    pub failure_committed: bool,
}

/// Result of executing one tool call.
pub(crate) struct ToolExecResult {
    pub call: FunctionCall,
    pub tool_name: String,
    pub qualified_tool_name: String,
    pub custom_data: Option<Value>,
    /// Where the tool came from, attached to the output and approval items it produces.
    pub tool_origin: ToolOrigin,
    pub outcome: ToolOutcome,
    pub input_guardrail_results: Vec<ToolInputGuardrailResult>,
    pub output_guardrail_results: Vec<ToolOutputGuardrailResult>,
}

fn format_tool_error<C>(
    config: &RunConfig<C>,
    ctx: &RunContext<C>,
    kind: ToolErrorKind,
    tool_name: &str,
    call_id: &str,
    default_message: String,
) -> String {
    format_tool_error_for(
        config,
        ctx,
        kind,
        "function",
        tool_name,
        call_id,
        default_message,
        None,
    )
}

/// Apply the run's tool error formatter for a tool of the given runtime type.
#[allow(clippy::too_many_arguments)]
pub(crate) fn format_tool_error_for<C>(
    config: &RunConfig<C>,
    ctx: &RunContext<C>,
    kind: ToolErrorKind,
    tool_type: &str,
    tool_name: &str,
    call_id: &str,
    default_message: String,
    error: Option<&AgentsError>,
) -> String {
    if let Some(f) = &config.tool_error_formatter {
        let args = ToolErrorFormatterArgs {
            kind,
            tool_type: tool_type.into(),
            tool_name: tool_name.to_string(),
            call_id: call_id.to_string(),
            default_message: default_message.clone(),
            run_context: ctx,
            error,
        };
        if let Some(m) = f(&args) {
            return m;
        }
    }
    default_message
}

/// Model-visible output for a tool the model called but the agent does not have.
pub(crate) fn tool_not_found_message<C>(
    config: &RunConfig<C>,
    ctx: &RunContext<C>,
    name: &str,
    call_id: &str,
) -> String {
    format_tool_error(
        config,
        ctx,
        ToolErrorKind::ToolNotFound,
        name,
        call_id,
        format!("Tool '{name}' not found."),
    )
}

fn output_outcome(call: &FunctionCall, output: ToolOutput) -> ToolOutcome {
    let item = ItemHelpers::tool_call_output_item(call, &output);
    ToolOutcome::Output { output, item }
}

/// An SDK-generated error for a call (a rejection, a guardrail verdict or a default failure
/// message). A schema-backed tool that a program called answers with `{"error": text}`, so the
/// program can parse the result; the run item keeps the plain text. Port of
/// `function_tool_error_output`.
fn error_outcome(
    output_json_schema: Option<&Value>,
    call: &FunctionCall,
    text: String,
) -> ToolOutcome {
    let called_by_program = call
        .extra
        .get("caller")
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
        == Some("program");
    let provider_text = if output_json_schema.is_some() && called_by_program {
        json!({"error": text}).to_string()
    } else {
        text.clone()
    };
    ToolOutcome::Output {
        output: ToolOutput::Text(text),
        item: ItemHelpers::tool_call_output_item(call, &ToolOutput::Text(provider_text)),
    }
}

enum InputGuardrailVerdict {
    Allow,
    Reject(String),
}

async fn run_input_guardrails<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    tool: &FunctionTool<C>,
    tool_ctx: &ToolContext<C>,
    results: &mut Vec<ToolInputGuardrailResult>,
) -> Result<InputGuardrailVerdict> {
    for g in &tool.tool_input_guardrails {
        let out = g
            .run(ToolInputGuardrailData {
                context: tool_ctx.clone(),
                agent_name: env.agent.name.clone(),
            })
            .await?;
        results.push(ToolInputGuardrailResult {
            guardrail_name: g.name.clone(),
            tool_name: tool.name.clone(),
            output: out.clone(),
        });
        match &out.behavior {
            ToolGuardrailBehavior::Allow => {}
            ToolGuardrailBehavior::RejectContent { message } => {
                return Ok(InputGuardrailVerdict::Reject(message.clone()));
            }
            ToolGuardrailBehavior::RaiseException => {
                return Err(AgentsError::ToolInputGuardrailTripwireTriggered {
                    guardrail_name: g.name.clone(),
                    output: Box::new(out),
                    run_data: None,
                });
            }
        }
    }
    Ok(InputGuardrailVerdict::Allow)
}

/// Execute one function tool call.
pub(crate) async fn execute_function_tool<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    call: FunctionCall,
    tool: Arc<FunctionTool<C>>,
) -> std::result::Result<ToolExecResult, ToolFailure> {
    let mut input_results = Vec::new();
    let mut output_results = Vec::new();
    let tool_ctx = ToolContext::from_call(env.ctx.clone(), &env.agent.name, &call)
        .with_namespace(tool.namespace.clone().or_else(|| {
            // A deferred top-level tool loaded by tool search is called under its synthetic
            // `name` namespace, which the tool context reports as Python does.
            call.extra
                .get("namespace")
                .and_then(serde_json::Value::as_str)
                .filter(|ns| tool.defer_loading && *ns == tool.name)
                .map(str::to_string)
        }))
        .with_agent(env.agent.clone())
        .with_run_config(env.config.clone());
    let finish = |outcome: ToolOutcome, i: Vec<_>, o: Vec<_>| ToolExecResult {
        call: call.clone(),
        tool_name: tool.name.clone(),
        qualified_tool_name: tool.qualified_name(),
        custom_data: None,
        tool_origin: tool.tool_origin(),
        outcome,
        input_guardrail_results: i,
        output_guardrail_results: o,
    };

    // Approvals are keyed by the qualified name so namespaced tools never collide.
    let approval_name = tool.qualified_name();
    // Decisions are recorded under the invocation-scoped key (`apply_patch:function_call` for a
    // function tool named `apply_patch`), so they are read back under the same key.
    let approval_key = crate::run_context::approval_tool_key(&approval_name, &call);
    // A restored decision that no ledger entry binds authorizes nothing; see
    // [`bound_approval_status`].
    let status_now = || {
        if env.unbound_approvals.contains(&call.call_id) {
            return None;
        }
        env.ctx
            .function_tool_status(&env.agent.name, &approval_key, &call.call_id)
            .approved
    };
    // A recorded decision wins: the policy runs only for an undecided call, so a resumed call
    // never re-runs it and a rejection holds whatever the policy would say. A decision recorded
    // while the policy ran is read again afterwards. Mirrors `_maybe_execute_tool_approval`.
    let mut status = status_now();
    let mut gated = status.is_some();
    if status.is_none() {
        // A call still pending from the resumed snapshot requires approval as it did.
        let needs_approval = env.pending_approvals.contains(&call.call_id)
            || tool
                .evaluate_needs_approval(env.ctx, &call.arguments, &call.call_id)
                .await;
        status = status_now();
        gated = needs_approval || status.is_some();
    }
    if gated {
        // The guardrails vet a call before it is offered for approval. Once an approval
        // decision exists they run with the invocation instead, so a resumed run does not
        // run them a second time. A decision recorded while they ran wins over their verdict.
        if status.is_none() && env.config.tool_execution.pre_approval_tool_input_guardrails {
            let verdict = run_input_guardrails(env, &tool, &tool_ctx, &mut input_results).await?;
            status = status_now();
            if let (None, InputGuardrailVerdict::Reject(msg)) = (status, verdict) {
                return Ok(finish(
                    error_outcome(tool.output_json_schema.as_ref(), &call, msg),
                    input_results,
                    output_results,
                ));
            }
        }
        match status {
            None => {
                let item = ToolApprovalItem {
                    agent: env.agent.name.clone(),
                    raw_item: call.clone(),
                    tool_name: approval_name.clone(),
                    tool_origin: Some(tool.tool_origin()),
                };
                return Ok(finish(
                    ToolOutcome::Interruptions(vec![item]),
                    input_results,
                    output_results,
                ));
            }
            Some(false) => {
                let message = env
                    .ctx
                    .function_tool_status(&env.agent.name, &approval_key, &call.call_id)
                    .rejection_message
                    .unwrap_or_else(|| {
                        format_tool_error(
                            env.config,
                            env.ctx,
                            ToolErrorKind::ApprovalRejected,
                            &tool.name,
                            &call.call_id,
                            DEFAULT_APPROVAL_REJECTION_MESSAGE.to_string(),
                        )
                    });
                let span = create_span(
                    SpanData::Function {
                        name: tool.qualified_name(),
                        input: None,
                        output: None,
                        mcp_data: None,
                    },
                    None,
                    false,
                );
                span.start();
                // The rejection message is app-supplied text, so it reaches the span under the
                // sensitive-data gate. Port of `_execute_single_tool`'s rejection span.
                let include = env.config.trace_include_sensitive_data;
                if include {
                    span.update_data(|d| {
                        if let SpanData::Function { output, .. } = d {
                            *output = Some(message.clone());
                        }
                    });
                }
                span.set_error(
                    if include {
                        message.clone()
                    } else {
                        "Tool execution rejected".to_string()
                    },
                    Some(serde_json::json!({
                        "tool_name": tool.name,
                        "error": format!(
                            "Tool execution for {} was manually rejected by user.",
                            call.call_id
                        ),
                    })),
                );
                span.finish();
                return Ok(finish(
                    error_outcome(tool.output_json_schema.as_ref(), &call, message),
                    input_results,
                    output_results,
                ));
            }
            Some(true) => {}
        }
    }

    let include = env.config.trace_include_sensitive_data;
    let span = create_span(
        SpanData::Function {
            name: tool.qualified_name(),
            input: if include {
                Some(call.arguments.clone())
            } else {
                None
            },
            output: None,
            mcp_data: tool.mcp_server_name().map(|s| json!({"server": s})),
        },
        None,
        false,
    );
    span.start();
    // Resuming a nested agent run re-enters only that run: the outer call's input guardrails
    // and start hooks already ran when it first started.
    let continues_nested_run =
        tool.is_agent_tool && env.ctx.nested_states().contains_key(&call.call_id);
    let run = async {
        if !continues_nested_run
            && let InputGuardrailVerdict::Reject(msg) =
                run_input_guardrails(env, &tool, &tool_ctx, &mut input_results).await?
        {
            return Ok(error_outcome(tool.output_json_schema.as_ref(), &call, msg));
        }
        let tool_enum = Tool::Function(tool.clone());
        // From here on user code runs: a retry must not run it again.
        env.ctx.mark_invocation_executed(&call.call_id);
        if !continues_nested_run {
            if let Some(h) = env.hooks {
                h.on_tool_start(&tool_ctx, env.agent, &tool_enum).await;
            }
            if let Some(h) = &env.agent.hooks {
                h.on_tool_start(&tool_ctx, env.agent, &tool_enum).await;
            }
        }

        // A failure or timeout that the tool's error function turned into model text is not the
        // tool's own output, so the declared output type does not apply to it.
        let mut formatted_failure = false;
        let mut output_rejected = false;
        // The text is the SDK's own (a default failure or timeout message), not the tool's.
        let mut sdk_generated_text = false;
        let invocation = tool.invoke_body(tool_ctx.clone(), call.arguments.clone());
        let result = match tool.timeout {
            Some(d) => match tokio::time::timeout(d, invocation).await {
                Ok(r) => r,
                Err(_) => {
                    let err = AgentsError::ToolTimeoutError {
                        tool_name: tool.name.clone(),
                        timeout_seconds: d.as_secs_f64(),
                        run_data: None,
                    };
                    match tool.timeout_behavior {
                        ToolTimeoutBehavior::RaiseException => return Err(err),
                        ToolTimeoutBehavior::ErrorAsResult => {
                            formatted_failure = true;
                            let msg = match &tool.timeout_error_function {
                                Some(f) => f(env.ctx, &err),
                                None => {
                                    sdk_generated_text = true;
                                    format!(
                                        "Tool '{}' timed out after {} seconds.",
                                        tool.name,
                                        d.as_secs_f64()
                                    )
                                }
                            };
                            Ok(ToolOutput::Text(msg))
                        }
                    }
                }
            },
            None => invocation.await,
        };

        if let Some(nested) = env.ctx.take_nested_interruptions(&call.call_id) {
            // A nested run parked on an approval resumes later; the call has not finished.
            env.ctx.mark_invocation_completed(&call.call_id);
            return Ok(ToolOutcome::Interruptions(nested));
        }

        let mut output = match result {
            Ok(o) => o,
            Err(e) => {
                formatted_failure = true;
                let propagate = matches!(
                    e,
                    AgentsError::InputGuardrailTripwireTriggered { .. }
                        | AgentsError::OutputGuardrailTripwireTriggered { .. }
                        | AgentsError::ToolInputGuardrailTripwireTriggered { .. }
                        | AgentsError::ToolOutputGuardrailTripwireTriggered { .. }
                );
                // A hosted program needs the declared structured output, so the default
                // message must not stand in for it. Port of
                // `resolve_function_tool_failure_error_function`.
                let propagate = propagate
                    || (tool.uses_default_failure_error_function
                        && tool.output_json_schema.is_some()
                        && !matches!(e, AgentsError::ToolTimeoutError { .. })
                        && call
                            .extra
                            .get("caller")
                            .and_then(|c| c.get("type"))
                            .and_then(Value::as_str)
                            == Some("program"));
                match (&tool.failure_error_function, propagate) {
                    (Some(f), false) => {
                        let mut msg = f(env.ctx, &e);
                        if tool.uses_default_failure_error_function {
                            // Keep the default failure policy identity: approving model
                            // feedback must not opt the original error into handled traces.
                            msg = format_tool_error_for(
                                env.config,
                                env.ctx,
                                ToolErrorKind::ToolException,
                                "function",
                                &tool.qualified_name(),
                                &call.call_id,
                                msg,
                                Some(&e),
                            );
                            sdk_generated_text = true;
                        }
                        // The default policy keeps the error out of traces even when sensitive
                        // data is traced; only an application handler opts the detail in.
                        if let Some(s) = scope::current_span() {
                            let include = env.config.trace_include_sensitive_data
                                && !tool.default_failure_error_function;
                            s.set_error(
                                "Error running tool (non-fatal)",
                                Some(serde_json::json!({
                                    "tool_name": tool.name,
                                    "error": crate::tracing::get_trace_tool_error(
                                        include,
                                        &e.to_string()
                                    ),
                                })),
                            );
                        }
                        // Port of `_on_handled_error`'s log: the detail only reaches the log
                        // when the tool data policy allows it.
                        if crate::debug::dont_log_tool_data() {
                            log::debug!(target: crate::logger::LOG_TARGET, "Tool {} failed", tool.name);
                        } else {
                            log::error!(
                                target: crate::logger::LOG_TARGET,
                                "Tool {} failed: {} {e}",
                                tool.name,
                                call.arguments
                            );
                        }
                        ToolOutput::Text(msg)
                    }
                    _ => return Err(e),
                }
            }
        };

        // A declared output type is a contract on the tool, not model feedback: a mismatch
        // fails the run rather than reaching `failure_error_function`.
        // The default failure text is model feedback, not a structured result. An application
        // handler's output is the tool's result, so it must match the declared type too.
        if !formatted_failure {
            tool.validate_output(&tool_ctx.tool_name, &output)?;
        } else if !tool.uses_default_failure_error_function
            && let ToolOutput::Text(text) = &output
        {
            tool.validate_failure_text(&tool_ctx.tool_name, text)?;
        }

        for g in &tool.tool_output_guardrails {
            let out = g
                .run(ToolOutputGuardrailData {
                    context: tool_ctx.clone(),
                    agent_name: env.agent.name.clone(),
                    output: output.clone(),
                })
                .await?;
            output_results.push(ToolOutputGuardrailResult {
                guardrail_name: g.name.clone(),
                tool_name: tool.name.clone(),
                output: out.clone(),
            });
            match &out.behavior {
                ToolGuardrailBehavior::Allow => {}
                ToolGuardrailBehavior::RejectContent { message } => {
                    output = ToolOutput::Text(message.clone());
                    output_rejected = true;
                    break;
                }
                ToolGuardrailBehavior::RaiseException => {
                    return Err(AgentsError::ToolOutputGuardrailTripwireTriggered {
                        guardrail_name: g.name.clone(),
                        output: Box::new(out),
                        run_data: None,
                    });
                }
            }
        }

        if let Some(h) = env.hooks {
            h.on_tool_end(&tool_ctx, env.agent, &tool_enum, &output)
                .await;
        }
        if let Some(h) = &env.agent.hooks {
            h.on_tool_end(&tool_ctx, env.agent, &tool_enum, &output)
                .await;
        }
        // The default failure text and a guardrail's rejection are SDK-generated errors, which a
        // schema-backed tool called by a program answers as a JSON error object.
        if let ToolOutput::Text(text) = &output
            && (sdk_generated_text || output_rejected)
        {
            return Ok(error_outcome(
                tool.output_json_schema.as_ref(),
                &call,
                text.clone(),
            ));
        }
        Ok(output_outcome(&call, output))
    };
    let outcome = scope::with_span(span.clone(), Box::pin(run)).await;
    match &outcome {
        Ok(ToolOutcome::Output { output, .. }) if include => {
            let text = output.to_model_string();
            span.update_data(|d| {
                if let SpanData::Function { output, .. } = d {
                    *output = Some(text);
                }
            });
        }
        Err(e) => span.set_error(
            "Error running tool",
            Some(serde_json::json!({
                "tool_name": tool.name,
                "error": crate::tracing::get_trace_tool_error(
                    env.config.trace_include_sensitive_data,
                    &e.to_string()
                ),
            })),
        ),
        _ => {}
    }
    span.finish();
    // A failure that is not an SDK error surfaces as a `UserError` naming the tool, like
    // Python's `UserError(f"Error running tool {name}: {e}")`.
    let outcome = outcome.map_err(|e| match e {
        AgentsError::Other(_) | AgentsError::Json(_) | AgentsError::Io(_) => {
            AgentsError::user(format!("Error running tool {}: {e}", tool.name))
        }
        other => other,
    })?;
    // SDK-only data the tool body attached plus whatever the extractor produces; the
    // extractor wins on conflicting keys. Port of `maybe_extract_custom_data` +
    // `merge_custom_data`.
    let extracted = match (&tool.custom_data_extractor, &outcome) {
        (Some(extractor), ToolOutcome::Output { output, item }) => {
            let ctx = crate::tool::FunctionToolCustomDataContext {
                tool_context: &tool_ctx,
                tool: &tool,
                output,
                raw_item: item,
                result_meta: tool_ctx.result_meta(),
            };
            extractor(&ctx).await
        }
        _ => None,
    };
    let custom_data = match crate::tool::merge_custom_data(&[tool_ctx.custom_data(), extracted]) {
        Ok(data) => data,
        Err(error) => {
            // The output is committed; only the SDK-only data failed to attach.
            return Err(ToolFailure {
                error,
                committed: Some(finish(outcome, input_results, output_results)),
            });
        }
    };
    let mut result = finish(outcome, input_results, output_results);
    result.custom_data = custom_data;
    Ok(result)
}

/// Execute tool calls concurrently (bounded by `max_function_tool_concurrency`), preserving
/// the call order in the results.
pub(crate) async fn execute_function_tools<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    runs: Vec<(FunctionCall, Arc<FunctionTool<C>>)>,
) -> std::result::Result<Vec<ToolExecResult>, ToolBatchFailure> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }
    let limit = env
        .config
        .tool_execution
        .max_function_tool_concurrency
        .unwrap_or(runs.len())
        .max(1);
    // Up to `limit` calls run at once, and a finished call frees its slot immediately, so the
    // queue refills in call order however the in-flight calls complete. Port of the
    // `_FunctionToolBatchExecutor` FIFO refill. On the first failure every sibling still in
    // flight is cancelled (the stream is dropped); among the failures already finished the one
    // with the lowest tool-call order wins (`_select_function_tool_failure` +
    // `_cancel_function_tool_tasks`). Siblings that had already committed their output are
    // kept, in call order, so a retry does not run them again.
    // Enablement is re-checked right before execution: a configured tool that became disabled
    // since the model saw it fails the batch before any call starts. Port of
    // `_FunctionToolBatchExecutor.execute`'s `resolve_enabled_function_tools` pass.
    let mut checked: Vec<Arc<FunctionTool<C>>> = Vec::new();
    for (_, tool) in &runs {
        if checked.iter().any(|t| Arc::ptr_eq(t, tool)) {
            continue;
        }
        checked.push(tool.clone());
        let configured = env
            .agent
            .tools
            .iter()
            .any(|t| t.as_function().is_some_and(|f| Arc::ptr_eq(f, tool)));
        if configured && !tool.is_enabled.evaluate(env.ctx, &env.agent.name).await {
            return Err(ToolBatchFailure {
                error: AgentsError::model_behavior(format!(
                    "Tool {} is currently disabled for agent {}.",
                    tool.name, env.agent.name
                )),
                completed: Vec::new(),
                failure_committed: false,
            });
        }
    }
    // The in-flight set is refilled by hand so that, once a call fails, draining the
    // already-finished siblings never starts a queued call.
    let mut queue = runs.into_iter().enumerate();
    let start = |(order, (call, tool)): (usize, (FunctionCall, Arc<FunctionTool<C>>))| {
        Box::pin(async move { (order, execute_function_tool(env, call, tool).await) })
    };
    let mut in_flight = futures::stream::FuturesUnordered::new();
    for run in queue.by_ref().take(limit) {
        in_flight.push(start(run));
    }
    let mut results: Vec<(usize, ToolExecResult)> = Vec::new();
    let into_ordered = |mut results: Vec<(usize, ToolExecResult)>| {
        results.sort_by_key(|(order, _)| *order);
        results.into_iter().map(|(_, r)| r).collect::<Vec<_>>()
    };
    while let Some((order, result)) = in_flight.next().await {
        match result {
            Ok(result) => {
                results.push((order, result));
                if let Some(run) = queue.next() {
                    in_flight.push(start(run));
                }
            }
            Err(failure) => {
                let mut chosen = (order, failure.error, failure.committed.is_some());
                results.extend(failure.committed.into_iter().map(|c| (order, c)));
                while let Some(Some((other_order, sibling))) =
                    futures::FutureExt::now_or_never(in_flight.next())
                {
                    match sibling {
                        Ok(result) => results.push((other_order, result)),
                        Err(other) => {
                            let committed = other.committed.is_some();
                            results.extend(other.committed.into_iter().map(|c| (other_order, c)));
                            if other_order < chosen.0 {
                                chosen = (other_order, other.error, committed);
                            }
                        }
                    }
                }
                return Err(ToolBatchFailure {
                    error: chosen.1,
                    completed: into_ordered(results),
                    failure_committed: chosen.2,
                });
            }
        }
    }
    Ok(into_ordered(results))
}
