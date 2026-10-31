//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py`: snapshots written under
//! schema 1.16 predate current-response ownership, so a callback failure on their resumed turn
//! must not publish unchecked output, and a later-turn retry that cannot prove which saved items
//! belong to the failed response is refused.
//!
//! Python fails the post-tool step with a raising extractor or tool-use behavior; a Rust
//! extractor returns a non-object instead (see `py_core6_resume_callback_retry.rs`) and a
//! custom tool-use behavior returns an error.

mod py_core6_resume_support;

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, RunState, Runner,
};
use py_core6_resume_support::*;
use serde_json::{Value, json};

const EXTRACTOR_FAILURE: &str = "custom_data_extractor must return a JSON object";

/// Relabel a current payload as schema 1.16 the way Python's tests do: the label predates
/// agent-scoped approvals, held Session writes and current-response ownership.
fn relabel_as_1_16(payload: &mut Value) {
    if let Some(entries) = payload["context"]
        .as_object_mut()
        .and_then(|c| c.remove("function_tool_approvals"))
        .and_then(|e| e.as_array().cloned())
    {
        for entry in entries {
            payload["context"]["approvals"][entry["tool_key"].as_str().unwrap()] =
                entry["decision"].clone();
        }
    }
    payload["$schemaVersion"] = json!("1.16");
    let map = payload.as_object_mut().unwrap();
    for key in [
        "generated_session_item_indexes",
        "current_response_generated_item_ownership",
        "pending_session_write",
    ] {
        map.remove(key);
    }
}

fn output_events(events: &[(RunItemStreamEventName, RunItemKindOutput)]) -> Vec<String> {
    events.iter().map(|(_, o)| o.0.clone()).collect()
}

struct RunItemKindOutput(String);

struct StreamRun {
    events: Vec<(RunItemStreamEventName, RunItemKindOutput)>,
    failure: Option<AgentsError>,
    final_output: Option<Value>,
    state: Option<RunState>,
}

async fn stream(agent: &Agent<()>, state: RunState) -> StreamRun {
    let mut run = Runner::run_streamed_with_options(agent, state, RunOptions::new(()));
    let mut out = StreamRun {
        events: Vec::new(),
        failure: None,
        final_output: None,
        state: None,
    };
    {
        let mut events = std::pin::pin!(run.stream_events());
        while let Some(event) = events.next().await {
            match event {
                Ok(StreamEvent::RunItem {
                    name: name @ RunItemStreamEventName::ToolOutput,
                    item,
                }) => {
                    if let RunItemKind::ToolCallOutput { output, .. } = &item.kind {
                        out.events.push((
                            name,
                            RunItemKindOutput(output.as_str().unwrap_or_default().to_string()),
                        ));
                    }
                }
                Ok(_) => {}
                Err(e) => out.failure = Some(e),
            }
        }
    }
    out.final_output = run.final_output().cloned();
    out.state = run.to_state();
    out
}

#[tokio::test]
async fn legacy_first_turn_callback_failure_withholds_normalized_output() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for guardrail_blocks in [false, true] {
        for published_sibling in [false, true] {
            let label = format!("blocks={guardrail_blocks} sibling={published_sibling}");
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let checked = Arc::new(Mutex::new(Vec::<String>::new()));
            let e = effects.clone();
            let charge = FunctionTool::new_no_args("charge", "", move |_c| {
                let e = e.clone();
                async move {
                    e.lock().unwrap().push("charged".into());
                    Ok("receipt-7")
                }
            })
            .with_needs_approval(true)
            .with_custom_data_extractor(|_ctx| async { Some(json!("post-tool callback failed")) });
            let prelude =
                FunctionTool::new_no_args(
                    "prelude",
                    "",
                    |_c| async move { Ok("published-receipt") },
                )
                .with_needs_approval(true);
            let seen = checked.clone();
            let terminal = OutputGuardrail::<()>::new("terminal", move |_c, _a, output: Value| {
                let seen = seen.clone();
                async move {
                    seen.lock()
                        .unwrap()
                        .push(output.as_str().unwrap_or_default().to_string());
                    Ok(GuardrailFunctionOutput::new(Value::Null, guardrail_blocks))
                }
            });
            let mut calls = vec![call("charge", json!({}), "charge-1")];
            if published_sibling {
                calls.insert(0, call("prelude", json!({}), "prelude-1"));
            }
            let model = scripted(vec![calls]);
            let agent = Agent::<()>::new("legacy-retry")
                .with_model_instance(model.clone())
                .with_tool(prelude)
                .with_tool(charge)
                .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
                    stop_at_tool_names: vec!["charge".into()],
                }))
                .with_output_guardrail(terminal);
            let paused = drive(&agent, "charge once", None, false, &RunCheckpoint::new())
                .await
                .unwrap();
            let mut state = paused.state.unwrap();
            if published_sibling {
                let first = state.get_interruptions()[0].clone();
                state.approve(&first, false);
                state = drive(&agent, state, None, false, &RunCheckpoint::new())
                    .await
                    .unwrap()
                    .state
                    .unwrap();
            }
            // Schema 1.16 predates current-response ownership and held Session writes.
            let mut payload = state.to_json().unwrap();
            relabel_as_1_16(&mut payload);
            let mut state = RunState::from_json(payload).unwrap();
            let first = state.get_interruptions()[0].clone();
            state.approve(&first, false);

            let failed = stream(&agent, state).await;
            let err = failed.failure.expect("the callback fails");
            assert!(
                err.to_string().contains(EXTRACTOR_FAILURE),
                "{label}: {err}"
            );
            assert_eq!(*effects.lock().unwrap(), vec!["charged"], "{label}");
            assert!(checked.lock().unwrap().is_empty(), "{label}");
            // Unchecked output is never announced.
            assert!(failed.events.is_empty(), "{label}");

            let ck = failed.state.expect("checkpoint");
            let retry = stream(&agent, ck).await;
            if guardrail_blocks {
                assert!(
                    matches!(
                        retry.failure,
                        Some(AgentsError::OutputGuardrailTripwireTriggered { .. })
                    ),
                    "{label}: {:?}",
                    retry.failure
                );
            } else {
                assert!(retry.failure.is_none(), "{label}: {:?}", retry.failure);
                assert_eq!(retry.final_output, Some(json!("receipt-7")), "{label}");
            }
            assert_eq!(*checked.lock().unwrap(), vec!["receipt-7"], "{label}");
            assert_eq!(*effects.lock().unwrap(), vec!["charged"], "{label}");
            let expected: Vec<String> = if guardrail_blocks {
                vec![]
            } else {
                vec!["receipt-7".into()]
            };
            // Only the output the failed attempt withheld is announced; the sibling output an
            // earlier resume already published is not announced again.
            assert_eq!(output_events(&retry.events), expected, "{label}");
        }
    }
}

#[tokio::test]
async fn legacy_later_turn_callback_failure_rejects_incomplete_retry() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for failure in ["extractor", "tool_use_behavior"] {
            let label = format!("streamed={streamed} failure={failure}");
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let charged_decisions = Arc::new(Mutex::new(Vec::<String>::new()));
            let before =
                FunctionTool::new_no_args("before", "", |_c| async move { Ok("before-ok") });
            let e = effects.clone();
            let mut charge = FunctionTool::new_no_args("charge", "", move |_c| {
                let e = e.clone();
                async move {
                    e.lock().unwrap().push("charged".into());
                    Ok("receipt-7")
                }
            })
            .with_needs_approval(true);
            if failure == "extractor" {
                charge = charge.with_custom_data_extractor(|_ctx| async {
                    Some(json!("post-tool callback failed"))
                });
            }
            let decisions = charged_decisions.clone();
            let behavior = if failure == "tool_use_behavior" {
                ToolUseBehavior::custom(move |_ctx, results| {
                    let decisions = decisions.clone();
                    async move {
                        if results.iter().any(|r| r.tool_name == "charge") {
                            let mut seen = decisions.lock().unwrap();
                            seen.push("charge".into());
                            if seen.len() == 1 {
                                return Err(AgentsError::other("post-tool callback failed"));
                            }
                        }
                        Ok(ToolsToFinalOutputResult {
                            is_final_output: false,
                            final_output: None,
                        })
                    }
                })
            } else {
                ToolUseBehavior::RunLlmAgain
            };
            let model = scripted(vec![
                vec![call("before", json!({}), "before-1")],
                vec![call("charge", json!({}), "charge-1")],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("legacy-later-turn")
                .with_model_instance(model.clone())
                .with_tool(before)
                .with_tool(charge)
                .with_tool_use_behavior(behavior);
            let paused = drive(&agent, "charge once", None, false, &RunCheckpoint::new())
                .await
                .unwrap();
            // Schema 1.16 cannot prove which saved items belong to a later model response.
            let mut payload = paused.state.unwrap().to_json().unwrap();
            relabel_as_1_16(&mut payload);
            let mut state = RunState::from_json(payload).unwrap();
            let first = state.get_interruptions()[0].clone();
            state.approve(&first, false);

            let slot = RunCheckpoint::new();
            let err = drive(&agent, state, None, streamed, &slot)
                .await
                .err()
                .expect("the callback fails");
            let expected = if failure == "extractor" {
                EXTRACTOR_FAILURE
            } else {
                "post-tool callback failed"
            };
            assert!(err.to_string().contains(expected), "{label}: {err}");
            assert_eq!(*effects.lock().unwrap(), vec!["charged"], "{label}");
            assert_eq!(model.calls().len(), 2, "{label}");

            let restored = round_trip(&slot.take().expect("checkpoint"));
            let err = drive(&agent, restored, None, !streamed, &RunCheckpoint::new())
                .await
                .err()
                .expect("the incomplete retry is refused");
            assert!(
                matches!(err, AgentsError::UserError { .. }),
                "{label}: {err:?}"
            );
            assert!(
                err.to_string().contains("cannot be resumed"),
                "{label}: {err}"
            );
            assert_eq!(*effects.lock().unwrap(), vec!["charged"], "{label}");
            let decided = if failure == "tool_use_behavior" {
                vec!["charge"]
            } else {
                vec![]
            };
            assert_eq!(*charged_decisions.lock().unwrap(), decided, "{label}");
            assert_eq!(model.calls().len(), 2, "{label}");
        }
    }
}
