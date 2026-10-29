//! Function tool execution: approvals, tool guardrails, hooks, timeouts and error handling.

use std::sync::Arc;

use futures::StreamExt;
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::items::{FunctionCall, ItemHelpers, ToolApprovalItem};
use crate::lifecycle::RunHooks;
use crate::run_config::{RunConfig, ToolErrorFormatterArgs, ToolErrorKind};
use crate::run_context::RunContext;
use crate::tool::{DEFAULT_APPROVAL_REJECTION_MESSAGE, FunctionTool, Tool, ToolOutput, ToolTimeoutBehavior};
use crate::tool_context::ToolContext;
use crate::tool_guardrails::{
    ToolGuardrailBehavior, ToolInputGuardrailData, ToolInputGuardrailResult, ToolOutputGuardrailData,
    ToolOutputGuardrailResult,
};
use crate::tracing::{SpanData, create_span, scope};

/// Shared inputs for executing tools in one turn.
pub(crate) struct ToolEnv<'a, C> {
    pub ctx: &'a RunContext<C>,
    pub agent: &'a Arc<Agent<C>>,
    pub hooks: Option<&'a Arc<dyn RunHooks<C>>>,
    pub config: &'a RunConfig<C>,
}

/// The outcome of one function tool call.
pub(crate) enum ToolOutcome {
    /// The tool produced output (or a model-visible error message).
    Output { output: ToolOutput, item: Value },
    /// The call needs approval (possibly several, from a nested agent run).
    Interruptions(Vec<ToolApprovalItem>),
}

/// Result of executing one tool call.
pub(crate) struct ToolExecResult {
    pub call: FunctionCall,
    pub tool_name: String,
    pub outcome: ToolOutcome,
    pub input_guardrail_results: Vec<ToolInputGuardrailResult>,
    pub output_guardrail_results: Vec<ToolOutputGuardrailResult>,
}

fn format_tool_error<C>(
    config: &RunConfig<C>,
    kind: ToolErrorKind,
    tool_name: &str,
    call_id: &str,
    default_message: String,
) -> String {
    if let Some(f) = &config.tool_error_formatter {
        let args = ToolErrorFormatterArgs {
            kind,
            tool_type: "function".into(),
            tool_name: tool_name.to_string(),
            call_id: call_id.to_string(),
            default_message: default_message.clone(),
        };
        if let Some(m) = f(&args) {
            return m;
        }
    }
    default_message
}

/// Model-visible output for a tool the model called but the agent does not have.
pub(crate) fn tool_not_found_message<C>(config: &RunConfig<C>, name: &str, call_id: &str) -> String {
    format_tool_error(
        config,
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
                    output: out,
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
) -> Result<ToolExecResult> {
    let mut input_results = Vec::new();
    let mut output_results = Vec::new();
    let tool_ctx = ToolContext::from_call(env.ctx.clone(), &env.agent.name, &call);
    let finish = |outcome: ToolOutcome, i: Vec<_>, o: Vec<_>| ToolExecResult {
        call: call.clone(),
        tool_name: tool.name.clone(),
        outcome,
        input_guardrail_results: i,
        output_guardrail_results: o,
    };

    // Approvals.
    let needs_approval = tool
        .needs_approval
        .evaluate(env.ctx, &call.arguments, &call.call_id)
        .await;
    if needs_approval {
        if env.config.tool_execution.pre_approval_tool_input_guardrails
            && let InputGuardrailVerdict::Reject(msg) =
                run_input_guardrails(env, &tool, &tool_ctx, &mut input_results).await?
        {
            return Ok(finish(output_outcome(&call, ToolOutput::Text(msg)), input_results, output_results));
        }
        match env.ctx.is_tool_approved(&env.agent.name, &tool.name, &call.call_id) {
            None => {
                let item = ToolApprovalItem {
                    agent: env.agent.name.clone(),
                    raw_item: call.clone(),
                    tool_name: tool.name.clone(),
                };
                return Ok(finish(ToolOutcome::Interruptions(vec![item]), input_results, output_results));
            }
            Some(false) => {
                let message = env
                    .ctx
                    .rejection_message(&env.agent.name, &tool.name, &call.call_id)
                    .unwrap_or_else(|| {
                        format_tool_error(
                            env.config,
                            ToolErrorKind::ApprovalRejected,
                            &tool.name,
                            &call.call_id,
                            DEFAULT_APPROVAL_REJECTION_MESSAGE.to_string(),
                        )
                    });
                let span = create_span(
                    SpanData::Function {
                        name: tool.name.clone(),
                        input: None,
                        output: None,
                        mcp_data: None,
                    },
                    None,
                    false,
                );
                span.start();
                span.set_error("Tool execution rejected", None);
                span.finish();
                return Ok(finish(
                    output_outcome(&call, ToolOutput::Text(message)),
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
            name: tool.name.clone(),
            input: if include { Some(call.arguments.clone()) } else { None },
            output: None,
            mcp_data: tool
                .mcp_server_name()
                .map(|s| json!({"server": s})),
        },
        None,
        false,
    );
    span.start();
    let run = async {
        if let InputGuardrailVerdict::Reject(msg) =
            run_input_guardrails(env, &tool, &tool_ctx, &mut input_results).await?
        {
            return Ok(output_outcome(&call, ToolOutput::Text(msg)));
        }
        let tool_enum = Tool::Function(tool.clone());
        if let Some(h) = env.hooks {
            h.on_tool_start(&tool_ctx, env.agent, &tool_enum).await;
        }
        if let Some(h) = &env.agent.hooks {
            h.on_tool_start(&tool_ctx, env.agent, &tool_enum).await;
        }

        let invocation = tool.invoke(tool_ctx.clone(), call.arguments.clone());
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
                            let msg = match &tool.timeout_error_function {
                                Some(f) => f(env.ctx, &err),
                                None => format!(
                                    "Tool '{}' timed out after {} seconds.",
                                    tool.name,
                                    d.as_secs_f64()
                                ),
                            };
                            Ok(ToolOutput::Text(msg))
                        }
                    }
                }
            },
            None => invocation.await,
        };

        if let Some(nested) = env.ctx.take_nested_interruptions(&call.call_id) {
            return Ok(ToolOutcome::Interruptions(nested));
        }

        let mut output = match result {
            Ok(o) => o,
            Err(e) => {
                let propagate = matches!(
                    e,
                    AgentsError::InputGuardrailTripwireTriggered { .. }
                        | AgentsError::OutputGuardrailTripwireTriggered { .. }
                        | AgentsError::ToolInputGuardrailTripwireTriggered { .. }
                        | AgentsError::ToolOutputGuardrailTripwireTriggered { .. }
                );
                match (&tool.failure_error_function, propagate) {
                    (Some(f), false) => {
                        let msg = f(env.ctx, &e);
                        scope::current_span().map(|s| {
                            s.set_error(
                                "Error running tool (non-fatal)",
                                Some(json!({"tool_name": tool.name, "error": e.to_string()})),
                            )
                        });
                        ToolOutput::Text(msg)
                    }
                    _ => return Err(e),
                }
            }
        };

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
                    break;
                }
                ToolGuardrailBehavior::RaiseException => {
                    return Err(AgentsError::ToolOutputGuardrailTripwireTriggered {
                        guardrail_name: g.name.clone(),
                        output: out,
                        run_data: None,
                    });
                }
            }
        }

        if let Some(h) = env.hooks {
            h.on_tool_end(&tool_ctx, env.agent, &tool_enum, &output).await;
        }
        if let Some(h) = &env.agent.hooks {
            h.on_tool_end(&tool_ctx, env.agent, &tool_enum, &output).await;
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
        Err(e) => span.set_error("Error running tool", Some(json!({"error": e.to_string()}))),
        _ => {}
    }
    span.finish();
    Ok(finish(outcome?, input_results, output_results))
}

/// Execute tool calls concurrently (bounded by `max_function_tool_concurrency`), preserving
/// the call order in the results.
pub(crate) async fn execute_function_tools<C: Send + Sync + 'static>(
    env: &ToolEnv<'_, C>,
    runs: Vec<(FunctionCall, Arc<FunctionTool<C>>)>,
) -> Result<Vec<ToolExecResult>> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }
    let limit = env
        .config
        .tool_execution
        .max_function_tool_concurrency
        .unwrap_or(runs.len())
        .max(1);
    let results: Vec<Result<ToolExecResult>> = futures::stream::iter(runs)
        .map(|(call, tool)| Box::pin(execute_function_tool(env, call, tool)))
        .buffered(limit)
        .collect()
        .await;
    results.into_iter().collect()
}
