//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py`: a callback that fails
//! after an approved tool already ran must not run the tool again. The retry keeps the committed
//! output, its guardrail results and its position among the current response's outputs.
//!
//! Python fails the post-tool step with a raising `custom_data_extractor` or `on_tool_end` hook.
//! A Rust extractor returns `Option<Value>` and a hook returns nothing, so neither can raise; the
//! equivalent failure at the same boundary is an extractor returning a non-object, which the
//! runner rejects with "custom_data_extractor must return a JSON object or None". Python's
//! `end_hook` variants have no Rust analogue and are classified Python-only in the report.

mod py_core6_resume_support;

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::items::{RunInput, RunItemKind};
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::run_config::{RunConfig, ToolExecutionConfig};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::tool::{FunctionTool, ShellTool, Tool};
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData, ToolOutputGuardrail,
    ToolOutputGuardrailData,
};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, RunState, Runner,
};
use py_core6_resume_support::*;
use serde_json::{Value, json};

const EXTRACTOR_FAILURE: &str = "custom_data_extractor must return a JSON object";

/// What a driven run produced, including the tool outputs its stream published.
struct Driven {
    result: std::result::Result<Outcome, AgentsError>,
    published: Vec<String>,
    state: Option<RunState>,
}

/// Run to completion or failure, recording every tool-output event of a streamed run and the
/// checkpoint a failure leaves.
async fn run_collecting(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Option<Arc<FailingSession>>,
    streamed: bool,
    configure: impl FnOnce(RunOptions<()>) -> RunOptions<()>,
) -> Driven {
    let slot = RunCheckpoint::new();
    let mut options = configure(RunOptions::new(()).checkpoint(&slot));
    if let Some(session) = session {
        options = options.session(session);
    }
    let mut published = Vec::new();
    if streamed {
        let mut run = Runner::run_streamed_with_options(agent, input, options);
        let mut failure = None;
        {
            let mut events = std::pin::pin!(run.stream_events());
            while let Some(event) = events.next().await {
                match event {
                    Ok(StreamEvent::RunItem {
                        name: RunItemStreamEventName::ToolOutput,
                        item,
                    }) => {
                        if let RunItemKind::ToolCallOutput { output, .. } = &item.kind {
                            published.push(output.as_str().unwrap_or_default().to_string());
                        }
                    }
                    Ok(_) => {}
                    Err(e) => failure = Some(e),
                }
            }
        }
        let state = run.to_state();
        match failure {
            Some(e) => Driven {
                result: Err(e),
                published,
                state,
            },
            None => Driven {
                result: Ok(Outcome {
                    final_output: run.final_output().cloned(),
                    interruptions: run.interruptions(),
                    state: state.clone(),
                    input_list: run.result().map(|r| r.to_input_list()).unwrap_or_default(),
                }),
                published,
                state,
            },
        }
    } else {
        match Runner::run_with_options(agent, input, options).await {
            Ok(r) => {
                let state = Some(r.to_state());
                Driven {
                    result: Ok(Outcome {
                        final_output: r.final_output.clone(),
                        interruptions: r.interruptions.clone(),
                        state: state.clone(),
                        input_list: r.to_input_list(),
                    }),
                    published,
                    state,
                }
            }
            Err(e) => Driven {
                result: Err(e),
                published,
                state: slot.take(),
            },
        }
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

fn input_infos(state: &RunState) -> Vec<Value> {
    state
        .tool_input_guardrail_results
        .iter()
        .map(|r| r.output.output_info.clone())
        .collect()
}

fn output_infos(state: &RunState) -> Vec<Value> {
    state
        .tool_output_guardrail_results
        .iter()
        .map(|r| r.output.output_info.clone())
        .collect()
}

fn function_call_outputs(items: &[openai_agents_sdk::InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.0.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .map(|i| {
            i.0.get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

fn allow_input(info: &'static str) -> ToolInputGuardrail<()> {
    ToolInputGuardrail::new(
        "check_input",
        move |_d: ToolInputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!(info)))
        },
    )
}

fn input_info_is_call_id() -> ToolInputGuardrail<()> {
    ToolInputGuardrail::new("check_input", |d: ToolInputGuardrailData<()>| async move {
        Ok(ToolGuardrailFunctionOutput::allow()
            .with_output_info(json!(d.context.tool_call_id.clone())))
    })
}

fn allow_output(info: &'static str) -> ToolOutputGuardrail<()> {
    ToolOutputGuardrail::new(
        "check_output",
        move |_d: ToolOutputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!(info)))
        },
    )
}

fn output_info_is_output() -> ToolOutputGuardrail<()> {
    ToolOutputGuardrail::new(
        "check_output",
        |d: ToolOutputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(d.output.to_json_value()))
        },
    )
}

fn stop_at(names: &[&str]) -> ToolUseBehavior<()> {
    ToolUseBehavior::StopAtTools(StopAtTools {
        stop_at_tool_names: names.iter().map(|n| n.to_string()).collect(),
    })
}

fn approved(state: &RunState) -> RunState {
    let mut state = state.clone();
    let first = state.get_interruptions()[0].clone();
    state.approve(&first, false);
    state
}

#[tokio::test]
async fn callback_retry_preserves_tool_output_in_result_history() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for json_trip in [false, true] {
            let label = format!("streamed={streamed} json={json_trip}");
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let callbacks = Arc::new(Mutex::new(Vec::<String>::new()));
            let (e, c) = (effects.clone(), callbacks.clone());
            let charge = FunctionTool::new_no_args("charge", "", move |_c| {
                let e = e.clone();
                async move {
                    e.lock().unwrap().push("charged".into());
                    Ok("receipt-7")
                }
            })
            .with_needs_approval(true)
            .with_custom_data_extractor(move |_ctx| {
                let c = c.clone();
                async move {
                    c.lock().unwrap().push("extractor".into());
                    Some(json!("post-tool callback failed"))
                }
            })
            .with_input_guardrail(allow_input("input checked"))
            .with_output_guardrail(allow_output("output checked"));
            let model = scripted(vec![
                vec![call("charge", json!({}), "charge-1")],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("agent")
                .with_model_instance(model.clone())
                .with_tool(charge)
                .with_tool_use_behavior(stop_at(&["charge"]));
            let paused = run_collecting(&agent, "charge once", None, streamed, |o| o).await;
            let state = approved(paused.state.as_ref().unwrap());

            let failed = run_collecting(&agent, state, None, streamed, |o| o).await;
            let err = failed.result.err().expect("the callback fails");
            assert!(
                err.to_string().contains(EXTRACTOR_FAILURE),
                "{label}: {err}"
            );
            let mut state = failed.state.expect("checkpoint");
            assert_eq!(tool_output_texts(&state), vec!["receipt-7"], "{label}");
            assert_eq!(input_infos(&state), vec![json!("input checked")], "{label}");
            assert_eq!(
                output_infos(&state),
                vec![json!("output checked")],
                "{label}"
            );
            let mut events = failed.published;

            if json_trip {
                state = round_trip(&state);
            }
            let retried = run_collecting(&agent, state, None, streamed, |o| o).await;
            events.extend(retried.published);
            let result = retried
                .result
                .unwrap_or_else(|e| panic!("{label}: retry failed: {e}"));
            assert_eq!(*effects.lock().unwrap(), vec!["charged"], "{label}");
            assert_eq!(*callbacks.lock().unwrap(), vec!["extractor"], "{label}");
            let final_state = result.state.unwrap();
            assert_eq!(
                input_infos(&final_state),
                vec![json!("input checked")],
                "{label}"
            );
            assert_eq!(
                output_infos(&final_state),
                vec![json!("output checked")],
                "{label}"
            );
            if streamed {
                assert_eq!(events, vec!["receipt-7"], "{label}");
            }
            assert_eq!(result.final_output, Some(json!("receipt-7")), "{label}");
            assert_eq!(
                tool_output_texts(&final_state),
                vec!["receipt-7"],
                "{label}"
            );
            assert_eq!(call_pair(&result.input_list, "charge-1"), pair(), "{label}");
            let restored = round_trip(&final_state);
            assert_eq!(tool_output_texts(&restored), vec!["receipt-7"], "{label}");
        }
    }
}

#[tokio::test]
async fn callback_retry_keeps_current_response_tool_order() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for previously_published in [false, true] {
            let label = format!("streamed={streamed} published={previously_published}");
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let make = |name: &'static str, result: &'static str, fail: bool| {
                let effects = effects.clone();
                let mut tool = FunctionTool::new_no_args(name, "", move |_c| {
                    let effects = effects.clone();
                    async move {
                        effects.lock().unwrap().push(name.to_string());
                        Ok(result)
                    }
                })
                .with_needs_approval(true)
                .with_input_guardrail(input_info_is_call_id())
                .with_output_guardrail(output_info_is_output());
                if fail {
                    tool = tool.with_custom_data_extractor(|_ctx| async {
                        Some(json!("tool callback failed"))
                    });
                }
                tool
            };
            let session = FailingSession::new("parallel");
            let model = scripted(vec![
                vec![
                    call("left", json!({}), "left-1"),
                    call("right", json!({}), "right-1"),
                ],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("parallel")
                .with_model_instance(model.clone())
                .with_tool(make("left", "left-result", previously_published))
                .with_tool(make("right", "right-result", !previously_published));
            let paused =
                run_collecting(&agent, "run both", Some(session.clone()), streamed, |o| o).await;
            let mut state = paused.state.unwrap();
            let interruptions = state.get_interruptions();
            let left_approval = interruptions
                .iter()
                .find(|i| i.call_id() == "left-1")
                .unwrap()
                .clone();
            let right_approval = interruptions
                .iter()
                .find(|i| i.call_id() == "right-1")
                .unwrap()
                .clone();
            state.approve(&right_approval, false);
            if previously_published {
                let partial =
                    run_collecting(&agent, state, Some(session.clone()), streamed, |o| o).await;
                let partial_state = partial.result.unwrap().state.unwrap();
                assert_eq!(
                    output_infos(&partial_state),
                    vec![json!("right-result")],
                    "{label}"
                );
                assert_eq!(
                    function_call_outputs(&session.items()),
                    vec!["right-1"],
                    "{label}"
                );
                state = partial_state;
                let next = state.get_interruptions()[0].clone();
                state.approve(&next, false);
            }
            let failed =
                run_collecting(&agent, state, Some(session.clone()), streamed, |o| o).await;
            let err = failed.result.err().expect("the callback fails");
            assert!(
                err.to_string().contains(EXTRACTOR_FAILURE),
                "{label}: {err}"
            );
            let failed_state = failed.state.expect("checkpoint");
            let expected_failed: Vec<Value> = if previously_published {
                vec![json!("right-result")]
            } else {
                vec![]
            };
            // The public failure data leaves out the output whose callback failed; only the
            // checkpoint keeps it.
            let public: Vec<Value> = err
                .run_data()
                .expect("a failed run reports its data")
                .tool_output_guardrail_results
                .iter()
                .map(|r| r.output.output_info.clone())
                .collect();
            assert_eq!(public, expected_failed, "{label}");

            let mut state = round_trip(&failed_state);
            state.approve(&left_approval, false);
            let retried =
                run_collecting(&agent, state, Some(session.clone()), streamed, |o| o).await;
            let result = retried
                .result
                .unwrap_or_else(|e| panic!("{label}: retry failed: {e}"));
            assert_eq!(*effects.lock().unwrap(), vec!["right", "left"], "{label}");
            assert_eq!(result.final_output, Some(json!("done")), "{label}");
            let (expected_ids, expected_outputs) = if previously_published {
                (
                    vec!["right-1", "left-1"],
                    vec!["right-result", "left-result"],
                )
            } else {
                (
                    vec!["left-1", "right-1"],
                    vec!["left-result", "right-result"],
                )
            };
            let final_state = result.state.unwrap();
            assert_eq!(tool_output_texts(&final_state), expected_outputs, "{label}");
            assert_eq!(
                function_call_outputs(&result.input_list),
                expected_ids,
                "{label}"
            );
            let last = model.calls().pop().unwrap();
            assert_eq!(
                function_call_outputs(&last.request.input),
                expected_ids,
                "{label}"
            );
            let restored = round_trip(&final_state);
            assert_eq!(tool_output_texts(&restored), expected_outputs, "{label}");
            assert_eq!(
                function_call_outputs(&session.items()),
                expected_ids,
                "{label}"
            );
            assert_eq!(
                input_infos(&final_state),
                vec![json!("right-1"), json!("left-1")],
                "{label}"
            );
            assert_eq!(
                output_infos(&final_state),
                vec![json!("right-result"), json!("left-result")],
                "{label}"
            );
            if streamed {
                let expected_events: Vec<String> = if previously_published {
                    vec!["left-result".into()]
                } else {
                    expected_outputs.iter().map(|s| s.to_string()).collect()
                };
                assert_eq!(retried.published, expected_events, "{label}");
            }
        }
    }
}

#[tokio::test]
async fn callback_retry_preserves_hosted_output_before_local_result() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for json_trip in [false, true] {
            let label = format!("streamed={streamed} json={json_trip}");
            let effects = Arc::new(Mutex::new(Vec::<String>::new()));
            let prelude =
                FunctionTool::new_no_args("prelude", "", |_c| async move { Ok("prefix") });
            let e = effects.clone();
            let charge = FunctionTool::new_no_args("charge", "", move |_c| {
                let e = e.clone();
                async move {
                    e.lock().unwrap().push("charged".into());
                    Ok("receipt-7")
                }
            })
            .with_needs_approval(true)
            .with_custom_data_extractor(|_ctx| async { Some(json!("charge callback failed")) });
            let allowed = OutputGuardrail::<()>::new("allowed", |_c, _a, _o: Value| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, false))
            });
            let model = scripted(vec![
                vec![call("prelude", json!({}), "prelude")],
                vec![
                    call("charge", json!({}), "charge"),
                    OutputItem::Other(json!({
                        "type": "shell_call",
                        "id": "sh-hosted",
                        "call_id": "shell-1",
                        "status": "completed",
                        "action": {"commands": ["echo hi"], "timeout_ms": null, "max_output_length": null},
                        "environment": {"type": "container_auto"},
                    })),
                    OutputItem::Other(json!({
                        "type": "shell_call_output",
                        "id": "sh-out-hosted",
                        "call_id": "shell-1",
                        "status": "completed",
                        "output": [{"stdout": "hi", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}],
                    })),
                ],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("hosted-shell")
                .with_model_instance(model.clone())
                .with_tool(prelude)
                .with_tool(charge)
                .with_tool(Tool::Shell(Arc::new(
                    ShellTool::hosted(json!({"type": "container_auto"})).unwrap(),
                )))
                .with_output_guardrail(allowed);
            // The scripted model does not stream mapping-shaped hosted output items.
            let paused = run_collecting(&agent, "prepare and charge", None, false, |o| o).await;
            let state = approved(paused.state.as_ref().unwrap());
            let failed = run_collecting(&agent, state, None, streamed, |o| o).await;
            let err = failed.result.err().expect("the callback fails");
            assert!(
                err.to_string().contains(EXTRACTOR_FAILURE),
                "{label}: {err}"
            );
            let mut state = failed.state.expect("checkpoint");
            if json_trip {
                state = round_trip(&state);
            }
            let retried = run_collecting(&agent, state, None, streamed, |o| o).await;
            let result = retried
                .result
                .unwrap_or_else(|e| panic!("{label}: retry failed: {e}"));
            assert_eq!(result.final_output, Some(json!("done")), "{label}");
            assert_eq!(*effects.lock().unwrap(), vec!["charged"], "{label}");
            let ids: Vec<String> = result
                .input_list
                .iter()
                .filter(|i| {
                    matches!(
                        i.0.get("type").and_then(Value::as_str),
                        Some("function_call_output" | "shell_call_output")
                    )
                })
                .map(|i| {
                    i.0.get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                })
                .collect();
            assert_eq!(ids, vec!["prelude", "shell-1", "charge"], "{label}");
        }
    }
}

#[tokio::test]
async fn streamed_preapproval_checkpoint_preserves_input_guardrail_results() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let charge = FunctionTool::new_no_args("charge", "", |_c| async move {
        panic!("An unapproved tool must not run");
        #[allow(unreachable_code)]
        Ok("never")
    })
    .with_needs_approval(true)
    .with_input_guardrail(allow_input("input checked before approval"));
    let model = scripted(vec![vec![call("charge", json!({}), "charge-1")]]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model)
        .with_tool(charge);
    let config = RunConfig::<()>::default().with_tool_execution(ToolExecutionConfig {
        pre_approval_tool_input_guardrails: true,
        ..Default::default()
    });
    let run = run_collecting(&agent, "charge once", None, true, |o| o.run_config(config)).await;
    let outcome = run.result.unwrap();
    assert_eq!(outcome.interruptions.len(), 1);
    let restored = round_trip(outcome.state.as_ref().unwrap());
    assert_eq!(
        input_infos(&restored),
        vec![json!("input checked before approval")]
    );
}

/// Python's `_approved_session_state` with a failing post-tool step: the tool has an extractor
/// that fails the same way for every call.
async fn failing_callback_session_state(streamed: bool) -> (Paused, Arc<Mutex<Vec<String>>>) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::new()));
    let callbacks = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = callbacks.clone();
    let tool = charge_tool(&effects).with_custom_data_extractor(move |_ctx| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push("charge".into());
            Some(json!("post-tool callback failed"))
        }
    });
    let model = scripted(vec![
        vec![call("charge", json!({"amount": 7}), "charge-1")],
        vec![assistant_message("done")],
        vec![assistant_message("fresh")],
    ]);
    let agent = Agent::<()>::new("payment")
        .with_model_instance(model.clone())
        .with_tool(tool);
    let session = FailingSession::new("resume");
    let paused = drive(
        &agent,
        "charge 7",
        Some(session.clone()),
        streamed,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    let state = approved(paused.state.as_ref().unwrap());
    (
        Paused {
            agent,
            model,
            session,
            state,
            effects,
        },
        callbacks,
    )
}

#[tokio::test]
async fn callback_failure_retry_delivers_output_to_session() {
    // (failing streamed, retry streamed, round trip, retry append)
    let cases = [
        (false, false, false, "success"),
        (true, true, true, "success"),
        (false, true, true, "lost-ack"),
        (true, false, false, "lost-ack"),
    ];
    for (failing_streamed, retry_streamed, json_trip, retry_append) in cases {
        let label =
            format!("fs={failing_streamed} rs={retry_streamed} json={json_trip} {retry_append}");
        let (p, callbacks) = failing_callback_session_state(failing_streamed).await;
        let slot = RunCheckpoint::new();
        let err = drive(
            &p.agent,
            p.state.clone(),
            Some(p.session.clone()),
            failing_streamed,
            &slot,
        )
        .await
        .err()
        .expect("the callback fails");
        assert!(
            err.to_string().contains(EXTRACTOR_FAILURE),
            "{label}: {err}"
        );
        let mut state = slot.take().unwrap();
        assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
        assert_eq!(*callbacks.lock().unwrap(), vec!["charge"], "{label}");
        assert_eq!(p.model.calls().len(), 1, "{label}");
        assert_eq!(
            charge_pair(&p.session.items()),
            vec!["function_call"],
            "{label}"
        );
        if json_trip {
            state = round_trip(&state);
        }
        if retry_append == "lost-ack" {
            p.session.fail_next("after");
            let slot = RunCheckpoint::new();
            let err = drive(
                &p.agent,
                state,
                Some(p.session.clone()),
                retry_streamed,
                &slot,
            )
            .await
            .err()
            .expect("the lost acknowledgement surfaces");
            assert!(
                err.to_string().contains("session append failed"),
                "{label}: {err}"
            );
            assert_eq!(p.model.calls().len(), 1, "{label}");
            assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
            assert_eq!(*callbacks.lock().unwrap(), vec!["charge"], "{label}");
            assert_eq!(charge_pair(&p.session.items()), pair(), "{label}");
            state = slot.take().unwrap();
            // Also serialize while the Session acknowledgement is outstanding.
            if json_trip {
                state = round_trip(&state);
            }
        }
        let result = drive(
            &p.agent,
            state,
            Some(p.session.clone()),
            retry_streamed,
            &RunCheckpoint::new(),
        )
        .await
        .unwrap_or_else(|e| panic!("{label}: retry failed: {e}"));
        assert_eq!(result.final_output, Some(json!("done")), "{label}");
        assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
        assert_eq!(*callbacks.lock().unwrap(), vec!["charge"], "{label}");
        assert_eq!(charge_pair(&result.input_list), pair(), "{label}");
        assert_eq!(charge_pair(&p.session.items()), pair(), "{label}");
        let last = p.model.calls().pop().unwrap();
        assert_eq!(charge_pair(&last.request.input), pair(), "{label}");

        drive(
            &p.agent,
            "What was the receipt?",
            Some(p.session.clone()),
            retry_streamed,
            &RunCheckpoint::new(),
        )
        .await
        .unwrap();
        let last = p.model.calls().pop().unwrap();
        assert_eq!(charge_pair(&last.request.input), pair(), "{label}");
        assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
    }
}
