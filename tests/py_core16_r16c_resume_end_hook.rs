//! Port of the `end_hook` variants of `tests/test_run_impl_resume_paths.py`
//! (`test_callback_retry_preserves_tool_output_in_result_history` and
//! `test_redacted_callback_preserves_private_failed_stream_checkpoint`): an `on_tool_end` hook
//! fails after an approved tool already ran. The failed run leaves a checkpoint that holds the
//! committed output and its guardrail results, and resuming it neither runs the tool nor the
//! hook again. A Rust hook fails through its fallible `try_` twin.

mod py_core6_resume_support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData, ToolOutputGuardrail,
    ToolOutputGuardrailData,
};
use openai_agents_sdk::{Agent, AgentsError, Result, RunState};
use py_core6_resume_support::*;
use serde_json::{Value, json};

struct FailingEndHook {
    calls: AtomicUsize,
}

#[async_trait]
impl RunHooks<()> for FailingEndHook {
    async fn try_on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &Tool<()>,
        _result: &ToolOutput,
    ) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(AgentsError::other("end hook failed"))
    }
}

fn tool_output_texts(state: &RunState) -> Vec<String> {
    state
        .generated_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => {
                Some(output.as_str().unwrap_or_default().to_string())
            }
            _ => None,
        })
        .collect()
}

fn infos<T>(results: &[T], info: impl Fn(&T) -> Value) -> Vec<Value> {
    results.iter().map(info).collect()
}

#[tokio::test]
async fn a_failing_end_hook_leaves_a_checkpoint_that_resumes_without_rerunning_anything() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for with_output_guardrail in [false, true] {
            let label = format!("streamed={streamed} output_guardrail={with_output_guardrail}");
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let recorder = effects.clone();
            let mut charge = FunctionTool::new_no_args("charge", "", move |_c| {
                let recorder = recorder.clone();
                async move {
                    recorder.lock().unwrap().push("charged".into());
                    Ok("receipt-7")
                }
            })
            .with_needs_approval(true)
            .with_input_guardrail(ToolInputGuardrail::new(
                "record_input",
                |_d: ToolInputGuardrailData<()>| async move {
                    Ok(ToolGuardrailFunctionOutput::allow()
                        .with_output_info(json!("approval input checked")))
                },
            ));
            if with_output_guardrail {
                charge = charge.with_output_guardrail(ToolOutputGuardrail::new(
                    "record_output",
                    |d: ToolOutputGuardrailData<()>| async move {
                        Ok(ToolGuardrailFunctionOutput::allow()
                            .with_output_info(d.output.to_json_value()))
                    },
                ));
            }
            let model = scripted(vec![vec![call("charge", json!({}), "charge-1")]]);
            let agent = Agent::<()>::new("agent")
                .with_model_instance(model.clone())
                .with_tool(charge)
                .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
                    stop_at_tool_names: vec!["charge".to_string()],
                }));
            let hook = Arc::new(FailingEndHook {
                calls: AtomicUsize::new(0),
            });

            let slot = RunCheckpoint::new();
            let paused = drive(&agent, "charge once", None, streamed, &slot)
                .await
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let mut state = paused.state.expect("state");
            let first = state.get_interruptions()[0].clone();
            state.approve(&first, false);

            let failed_slot = RunCheckpoint::new();
            let hooks: Arc<dyn RunHooks<()>> = hook.clone();
            let failed = drive_with(&agent, state, None, streamed, &failed_slot, |o| {
                o.hooks(hooks)
            })
            .await;
            let error = failed.err().unwrap_or_else(|| panic!("{label}: must fail"));
            assert!(
                error.to_string().contains("end hook failed"),
                "{label}: {error}"
            );
            let checkpoint = failed_slot.take().expect("checkpoint");
            assert_eq!(tool_output_texts(&checkpoint), vec!["receipt-7"], "{label}");
            assert_eq!(
                infos(&checkpoint.tool_input_guardrail_results, |r| r
                    .output
                    .output_info
                    .clone()),
                vec![json!("approval input checked")],
                "{label}"
            );

            let hooks: Arc<dyn RunHooks<()>> = hook.clone();
            let retry_slot = RunCheckpoint::new();
            let retried = drive_with(
                &agent,
                round_trip(&checkpoint),
                None,
                streamed,
                &retry_slot,
                |o| o.hooks(hooks),
            )
            .await
            .unwrap_or_else(|e| panic!("{label}: retry failed: {e}"));
            assert_eq!(retried.final_output, Some(json!("receipt-7")), "{label}");
            assert_eq!(*effects.lock().unwrap(), vec!["charged"], "{label}");
            assert_eq!(hook.calls.load(Ordering::SeqCst), 1, "{label}");
            assert_eq!(model.calls().len(), 1, "{label}");
            assert_eq!(
                call_pair(&retried.input_list, "charge-1"),
                pair(),
                "{label}"
            );
            let final_state = retried.state.expect("final state");
            assert_eq!(
                infos(&final_state.tool_output_guardrail_results, |r| r
                    .output
                    .output_info
                    .clone()),
                if with_output_guardrail {
                    vec![json!("receipt-7")]
                } else {
                    vec![]
                },
                "{label}"
            );
        }
    }
}
