//! Port of `TestRunStateResumption` and the module-level resume cases of the Python SDK's
//! `tests/test_run_state.py`: resuming through the public `Runner` from a `RunState`, with and
//! without streaming, context overrides, server-managed conversation ids, approvals on the
//! parked turn, and nested `Agent::as_tool` runs.
//!
//! Python's `RunState` is mutable and shared with the result that produced it; Rust's is a
//! value the runner consumes, so "does not mutate the state it resumed from" is checked on the
//! clone that was not handed to the runner.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunContext, RunOptions, RunResult, Runner};
use serde_json::{Value, json};

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn agent(model: &Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new("TestAgent").with_model_instance(model.clone())
}

/// Drain a streamed run and return it once it is complete.
async fn drain<C: Send + Sync + 'static>(
    mut stream: openai_agents_sdk::RunResultStreaming<C>,
) -> openai_agents_sdk::RunResultStreaming<C> {
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    stream
}

type Log = Arc<Mutex<Vec<String>>>;

fn recording_tool(name: &str, log: &Log, needs_approval: bool) -> FunctionTool<()> {
    let log = log.clone();
    FunctionTool::<()>::new(name, "Records its value", move |_c, a: Value| {
        let log = log.clone();
        async move {
            let value = a["value"].as_str().unwrap_or("").to_string();
            log.lock().unwrap().push(value.clone());
            Ok(format!("approved:{value}"))
        }
    })
    .with_needs_approval(needs_approval)
}

// ---------------------------------------------------------------------------
// TestRunStateResumption
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resume_executes_only_exact_override_result() {
    for streamed in [false, true] {
        for sticky_approved in [true, false] {
            let log: Log = Arc::default();
            let model = fake();
            model.add_turns(vec![
                vec![
                    function_call_with_id("approval_tool", r#"{"value": "sticky"}"#, "sticky-call"),
                    function_call_with_id(
                        "approval_tool",
                        r#"{"value": "exception"}"#,
                        "exception-call",
                    ),
                ],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("MixedApprovalAgent")
                .with_tool(recording_tool("approval_tool", &log, true))
                .with_model_instance(model.clone());

            let initial = Runner::run(&agent, "start").await.unwrap();
            let mut state = initial.to_state();
            let by_id = |id: &str| {
                state
                    .get_interruptions()
                    .into_iter()
                    .find(|i| i.call_id() == id)
                    .unwrap()
            };
            let (sticky, exception) = (by_id("sticky-call"), by_id("exception-call"));
            if sticky_approved {
                state.approve(&sticky, true);
                state.reject(&exception, false, Some("denied exactly".into()));
            } else {
                state.reject(&sticky, true, Some("denied by default".into()));
                state.approve(&exception, false);
            }

            let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
            let (final_output, interruptions) = if streamed {
                let resumed = drain(Runner::run_streamed(&agent, restored)).await;
                (resumed.final_output_text(), resumed.interruptions().len())
            } else {
                let resumed = Runner::run(&agent, restored).await.unwrap();
                (resumed.final_output_text(), resumed.interruptions.len())
            };
            assert_eq!(final_output.as_deref(), Some("done"));
            assert_eq!(interruptions, 0);
            let expected = if sticky_approved {
                "sticky"
            } else {
                "exception"
            };
            assert_eq!(
                *log.lock().unwrap(),
                vec![expected.to_string()],
                "streamed={streamed} sticky_approved={sticky_approved}"
            );
        }
    }
}

#[tokio::test]
async fn resume_from_run_state() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
    ]);
    let agent = agent(&model);
    let result1 = Runner::run(&agent, "First input").await.unwrap();
    // Python resumes the finished checkpoint as is and runs another model turn.
    let state = result1.to_state();
    let result2 = Runner::run(&agent, state).await.unwrap();
    assert_eq!(
        result2.final_output_text().as_deref(),
        Some("Second response")
    );
}

/// One finished turn, then a resume of its checkpoint with a follow-up input.
async fn first_and_resumed(model: &Arc<FakeModel>) -> (RunResult<()>, RunResult<()>) {
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
    ]);
    let agent = agent(model);
    let result1 = Runner::run(&agent, "First input").await.unwrap();
    let state = result1.to_state();
    let result2 = Runner::run(&agent, state).await.unwrap();
    (result1, result2)
}

#[tokio::test]
async fn resume_from_run_state_does_not_mutate_source_result() {
    let (result1, result2) = first_and_resumed(&fake()).await;
    assert_eq!(result2.raw_responses.len(), 2);
    assert_eq!(result1.raw_responses.len(), 1);
}

#[tokio::test]
async fn resume_from_run_state_does_not_mutate_source_result_usage() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
    ]);
    let agent = agent(&model);
    let result1 = Runner::run(&agent, "First input").await.unwrap();
    let requests_after_first_run = result1.usage().requests;
    let state = result1.to_state();
    let result2 = Runner::run(&agent, state).await.unwrap();
    assert!(result2.usage().requests > requests_after_first_run);
    assert_eq!(result1.usage().requests, requests_after_first_run);
}

#[tokio::test]
async fn two_checkpoints_from_one_result_do_not_share_usage() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
        vec![assistant_message("Third response")],
    ]);
    let agent = agent(&model);
    let result = Runner::run(&agent, "First input").await.unwrap();
    let first = result.to_state();
    let second = result.to_state();
    let first_resume = Runner::run(&agent, first).await.unwrap();
    let second_resume = Runner::run(&agent, second).await.unwrap();
    assert_eq!(first_resume.usage().requests, 2);
    assert_eq!(second_resume.usage().requests, 2);
}

#[tokio::test]
async fn resume_does_not_append_to_the_state_it_resumed_from() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
        vec![assistant_message("Third response")],
    ]);
    let agent = agent(&model);
    let state = Runner::run(&agent, "First input").await.unwrap().to_state();
    let serialized_before = state.to_json().unwrap()["model_responses"].clone();

    let result2 = Runner::run(&agent, state.clone()).await.unwrap();
    assert_eq!(result2.raw_responses.len(), 2);
    assert_eq!(state.model_responses.len(), 1);
    assert_eq!(
        state.to_json().unwrap()["model_responses"],
        serialized_before
    );

    // Re-running the same checkpoint replays only its own history.
    let result3 = Runner::run(&agent, state).await.unwrap();
    assert_eq!(result3.raw_responses.len(), 2);
}

#[tokio::test]
async fn streamed_resume_does_not_append_to_the_state_it_resumed_from() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
        vec![assistant_message("Third response")],
    ]);
    let agent = agent(&model);
    let state = Runner::run(&agent, "First input").await.unwrap().to_state();
    let serialized_before = state.to_json().unwrap()["session_items"].clone();

    let result2 = drain(Runner::run_streamed(&agent, state.clone())).await;
    let new_items = &result2.result().unwrap().new_items;
    // Rust records the admitted follow-up input as an item of the resumed turn.
    let messages = new_items
        .iter()
        .filter(|i| i.item_type() == "message_output_item")
        .count();
    assert_eq!(messages, 2);
    assert_eq!(state.generated_items.len(), 1);
    assert_eq!(state.to_json().unwrap()["session_items"], serialized_before);

    let result3 = drain(Runner::run_streamed(&agent, state)).await;
    let result3 = result3.result().unwrap();
    let messages = result3
        .new_items
        .iter()
        .filter(|i| i.item_type() == "message_output_item")
        .count();
    assert_eq!(messages, 2);
    // The abandoned attempt's reply does not leak into the replayed history.
    let history = result3.to_input_list();
    assert!(
        !history
            .iter()
            .any(|i| i.text_content().as_deref() == Some("Second response"))
    );
}

#[tokio::test]
async fn resumed_max_turns_handler_does_not_append_to_state_items() {
    let model = fake();
    model.add_turn(vec![assistant_message("First response")]);
    let agent = agent(&model);
    let mut state = Runner::run_with_options(
        &agent,
        "First input",
        RunOptions::new(()).max_turns(Some(1)),
    )
    .await
    .unwrap()
    .to_state();
    let serialized_before = state.to_json().unwrap()["generated_items"].clone();

    let handlers = RunErrorHandlers::<()>::new()
        .on_max_turns(|_input| async { Ok(Some(RunErrorHandlerResult::new("fallback"))) });
    // The checkpoint has no turns left; staging input is refused, so resume it as is with a
    // pending tool-less turn that trips the limit.
    assert!(state.add_input("more").is_err());
    let result2 = Runner::run_with_options(
        &agent,
        state.clone(),
        RunOptions::new(()).error_handlers(handlers),
    )
    .await
    .unwrap();
    assert_eq!(result2.final_output_text().as_deref(), Some("fallback"));
    assert_eq!(state.generated_items.len(), 1);
    assert_eq!(
        state.to_json().unwrap()["generated_items"],
        serialized_before
    );
}

#[tokio::test]
async fn fresh_runs_still_report_their_own_history() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Streamed response")],
    ]);
    let agent = agent(&model);
    let result1 = Runner::run(&agent, "First input").await.unwrap();
    assert_eq!(result1.raw_responses.len(), 1);
    assert_eq!(result1.new_items.len(), 1);
    let result2 = drain(Runner::run_streamed(&agent, "Second input")).await;
    let result2 = result2.result().unwrap();
    assert_eq!(result2.raw_responses.len(), 1);
    assert_eq!(result2.new_items.len(), 1);
}

#[tokio::test]
async fn resume_from_run_state_with_context() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
    ]);
    let agent = Agent::<Value>::new("TestAgent").with_model_instance(model.clone());
    let result1 = Runner::run_with_options(
        &agent,
        "First input",
        RunOptions::new(json!({"key": "value1"})),
    )
    .await
    .unwrap();
    let state = result1.to_state();
    let result2 =
        Runner::run_with_options(&agent, state, RunOptions::new(json!({"key": "value2"})))
            .await
            .unwrap();
    assert_eq!(
        result2.final_output_text().as_deref(),
        Some("Second response")
    );
    assert_eq!(*result2.context_wrapper.context(), json!({"key": "value2"}));
}

#[tokio::test]
async fn resume_from_run_state_with_conversation_id() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
    ]);
    let agent = agent(&model);
    let result1 = Runner::run_with_options(
        &agent,
        "First input",
        RunOptions::new(()).conversation_id("conv123"),
    )
    .await
    .unwrap();
    let state = result1.to_state();
    assert_eq!(state.conversation_id.as_deref(), Some("conv123"));
    let result2 = Runner::run_with_options(
        &agent,
        state,
        RunOptions::new(()).conversation_id("conv123"),
    )
    .await
    .unwrap();
    assert_eq!(
        result2.final_output_text().as_deref(),
        Some("Second response")
    );
}

#[tokio::test]
async fn resume_from_run_state_with_previous_response_id() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
    ]);
    let agent = agent(&model);
    let result1 = Runner::run_with_options(
        &agent,
        "First input",
        RunOptions::new(()).previous_response_id("resp123"),
    )
    .await
    .unwrap();
    let state = result1.to_state();
    let result2 = Runner::run_with_options(
        &agent,
        state,
        RunOptions::new(()).previous_response_id("resp123"),
    )
    .await
    .unwrap();
    assert_eq!(
        result2.final_output_text().as_deref(),
        Some("Second response")
    );
}

#[tokio::test]
async fn resume_from_run_state_with_interruption() {
    let log: Log = Arc::default();
    let model = fake();
    model.add_turns(vec![
        vec![function_call_with_id("test_tool", "{}", "call-1")],
        vec![assistant_message("Second response")],
    ]);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(recording_tool("test_tool", &log, false))
        .with_model_instance(model.clone());
    let result1 = Runner::run(&agent, "First input").await.unwrap();
    let mut state = result1.to_state();
    if let Some(item) = state.get_interruptions().first().cloned() {
        state.approve(&item, false);
    }
    // The tool needs no approval, so the first run already finished with the scripted reply.
    assert!(state.interruptions.is_empty());
    assert_eq!(
        result1.final_output_text().as_deref(),
        Some("Second response")
    );
    assert_eq!(*log.lock().unwrap(), vec![String::new()]);
}

#[tokio::test]
async fn result_to_state_detaches_tool_decision_ledgers() {
    for streamed in [false, true] {
        let log: Log = Arc::default();
        let model = fake();
        model.add_turn(vec![function_call_with_id(
            "approval_tool",
            r#"{"value":"x"}"#,
            "call-1",
        )]);
        let agent = Agent::<()>::new("TestAgent")
            .with_tool(recording_tool("approval_tool", &log, true))
            .with_model_instance(model.clone());
        let (mut decided, untouched) = if streamed {
            let result = drain(Runner::run_streamed(&agent, "First input")).await;
            (result.to_state().unwrap(), result.to_state().unwrap())
        } else {
            let result = Runner::run(&agent, "First input").await.unwrap();
            (result.to_state(), result.to_state())
        };
        let before = untouched.to_json().unwrap()["context"]["function_tool_approvals"].clone();
        let item = decided.get_interruptions()[0].clone();
        decided.approve(&item, false);
        assert_eq!(
            untouched.to_json().unwrap()["context"]["function_tool_approvals"],
            before
        );

        model.add_turn(vec![assistant_message("unused")]);
        let interruptions = if streamed {
            drain(Runner::run_streamed(&agent, untouched))
                .await
                .interruptions()
                .len()
        } else {
            Runner::run(&agent, untouched)
                .await
                .unwrap()
                .interruptions
                .len()
        };
        assert_eq!(interruptions, 1, "streamed={streamed}");
        assert!(log.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn resume_from_run_state_streamed() {
    let model = fake();
    model.add_turns(vec![
        vec![assistant_message("First response")],
        vec![assistant_message("Second response")],
    ]);
    let agent = agent(&model);
    let state = Runner::run(&agent, "First input").await.unwrap().to_state();
    let result2 = drain(Runner::run_streamed(&agent, state)).await;
    assert_eq!(
        result2.final_output_text().as_deref(),
        Some("Second response")
    );
}

fn hand_built_state() -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "TestAgent".into(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(1),
        context: Some(json!({"key": "value"})),
        ..Default::default()
    }
}

#[tokio::test]
async fn resume_from_run_state_streamed_uses_context_from_state() {
    let model = fake();
    model.add_turn(vec![assistant_message("done")]);
    let agent = agent(&model);
    let result = drain(Runner::run_streamed(&agent, hand_built_state())).await;
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

#[tokio::test]
async fn resume_from_run_state_streamed_with_context_override() {
    let model = fake();
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<Value>::new("TestAgent").with_model_instance(model.clone());
    let result = drain(Runner::run_streamed_with_options(
        &agent,
        hand_built_state(),
        RunOptions::new(json!({"key": "value2"})),
    ))
    .await;
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        *result.result().unwrap().context_wrapper.context(),
        json!({"key": "value2"})
    );
}

#[tokio::test]
async fn run_result_streaming_to_state_with_interruptions() {
    let log: Log = Arc::default();
    let model = fake();
    model.add_turns(vec![
        vec![function_call_with_id("test_tool", "{}", "call-1")],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("TestAgent")
        .with_tool(recording_tool("test_tool", &log, true))
        .with_model_instance(model.clone());
    let result = drain(Runner::run_streamed(&agent, "test")).await;
    let interruptions = result.interruptions();
    assert!(!interruptions.is_empty());
    let state = result.to_state().unwrap();
    assert!(state.pending_tool_processing);
    assert_eq!(state.interruptions.len(), interruptions.len());
    let wire = state.to_json().unwrap();
    assert_eq!(
        wire["current_step"]["type"],
        json!("next_step_interruption")
    );
}

// ---------------------------------------------------------------------------
// Module-level runner guardrail case
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_guardrail_models_survive_state_serialization() {
    use openai_agents_sdk::{GuardrailFunctionOutput, InputGuardrail, OutputGuardrail};
    let model = fake();
    model.add_turn(vec![assistant_message(r#"{"text":"hello"}"#)]);
    let verdict = json!({"allowed": true, "reason": "approved"});
    let input_verdict = verdict.clone();
    let output_verdict = verdict.clone();
    let agent = Agent::<()>::new("Audit")
        .with_input_guardrail(InputGuardrail::new("check", move |_c, _a, _i| {
            let v = input_verdict.clone();
            async move {
                Ok(GuardrailFunctionOutput {
                    output_info: v,
                    tripwire_triggered: false,
                })
            }
        }))
        .with_output_guardrail(OutputGuardrail::new("check", move |_c, _a, _o| {
            let v = output_verdict.clone();
            async move {
                Ok(GuardrailFunctionOutput {
                    output_info: v,
                    tripwire_triggered: false,
                })
            }
        }))
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "hello").await.unwrap();
    let restored = RunState::from_json_str(&result.to_state().to_json_string().unwrap()).unwrap();
    assert_eq!(
        restored.input_guardrail_results[0].output.output_info,
        verdict
    );
    assert_eq!(
        restored.output_guardrail_results[0].output.output_info,
        verdict
    );
    assert_eq!(
        restored.output_guardrail_results[0].agent_output,
        json!(r#"{"text":"hello"}"#)
    );
}

// ---------------------------------------------------------------------------
// Nested `Agent::as_tool` decisions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Timing {
    Live,
    BeforeRestore,
    AfterRestore,
}

/// Build `nesting_edges` agent-tool layers around a leaf agent whose `protected` tool needs
/// approval, scripted so every layer finishes after its nested tool returns.
fn nested_agents(nesting_edges: usize, calls: &Arc<AtomicUsize>) -> Agent<()> {
    let leaf_model = fake();
    leaf_model.add_turns(vec![
        vec![function_call_with_id(
            "protected",
            r#"{"value":"one"}"#,
            "inner-1",
        )],
        vec![assistant_message("inner-done")],
    ]);
    let calls = calls.clone();
    let protected = FunctionTool::<()>::new("protected", "Protected tool", move |_c, _a: Value| {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok("approved:one".to_string())
        }
    })
    .with_needs_approval(true);
    let mut outer = Agent::<()>::new("inner")
        .with_tool(protected)
        .with_model_instance(leaf_model);
    for edge in 0..nesting_edges {
        let tool_name = format!("agent_tool_{edge}");
        let model = fake();
        model.add_turns(vec![
            vec![function_call_with_id(
                &tool_name,
                r#"{"input":"go"}"#,
                format!("agent-call-{edge}"),
            )],
            vec![assistant_message(format!("done-{edge}"))],
        ]);
        outer = Agent::<()>::new(format!("agent-{edge}"))
            .with_tool(outer.as_tool(tool_name, "Run nested agent"))
            .with_model_instance(model);
    }
    outer
}

#[tokio::test]
async fn resume_recursively_nested_agent_as_tool_decision() {
    for nesting_edges in [2usize, 3] {
        for timing in [Timing::Live, Timing::BeforeRestore, Timing::AfterRestore] {
            for approve in [true, false] {
                for streamed in [false, true] {
                    let calls = Arc::new(AtomicUsize::new(0));
                    let outer = nested_agents(nesting_edges, &calls);
                    let mut state = if streamed {
                        drain(Runner::run_streamed(&outer, "start"))
                            .await
                            .to_state()
                            .unwrap()
                    } else {
                        Runner::run(&outer, "start").await.unwrap().to_state()
                    };
                    assert_eq!(state.get_interruptions().len(), 1);
                    let decide = |state: &mut RunState| {
                        let item = state.get_interruptions()[0].clone();
                        if approve {
                            state.approve(&item, false);
                        } else {
                            state.reject(&item, false, None);
                        }
                    };
                    let reload = |s: &RunState| RunState::from_json(s.to_json().unwrap()).unwrap();
                    match timing {
                        Timing::BeforeRestore => {
                            decide(&mut state);
                            state = reload(&state);
                        }
                        Timing::AfterRestore => {
                            state = reload(&state);
                            decide(&mut state);
                        }
                        Timing::Live => decide(&mut state),
                    }
                    let (final_output, interruptions) = if streamed {
                        let resumed = drain(Runner::run_streamed(&outer, state)).await;
                        (resumed.final_output_text(), resumed.interruptions().len())
                    } else {
                        let resumed = Runner::run(&outer, state).await.unwrap();
                        (resumed.final_output_text(), resumed.interruptions.len())
                    };
                    let label = format!(
                        "edges={nesting_edges} timing={timing:?} approve={approve} \
                         streamed={streamed}"
                    );
                    assert_eq!(
                        final_output.as_deref(),
                        Some(format!("done-{}", nesting_edges - 1).as_str()),
                        "{label}"
                    );
                    assert_eq!(interruptions, 0, "{label}");
                    assert_eq!(
                        calls.load(Ordering::SeqCst),
                        usize::from(approve),
                        "{label}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn resume_nested_agent_as_tool_with_context_override() {
    #[derive(Default, Clone)]
    struct AppContext {
        user: String,
    }
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let tool = {
        let seen = seen.clone();
        FunctionTool::<AppContext>::new("needs_ok", "Sensitive", move |ctx, _a: Value| {
            let seen = seen.clone();
            let user = ctx.context().user.clone();
            async move {
                seen.lock().unwrap().push(user);
                Ok("one".to_string())
            }
        })
        .with_needs_approval(true)
    };
    let nested_model = fake();
    nested_model.add_turns(vec![
        vec![function_call_with_id(
            "needs_ok",
            r#"{"text":"one"}"#,
            "inner-1",
        )],
        vec![assistant_message("nested-done")],
    ]);
    let nested = Agent::<AppContext>::new("nested")
        .with_tool(tool)
        .with_model_instance(nested_model);
    let outer_model = fake();
    outer_model.add_turns(vec![
        vec![function_call_with_id(
            "nested_agent",
            r#"{"input":"hi"}"#,
            "outer-1",
        )],
        vec![assistant_message("done")],
    ]);
    let outer = Agent::<AppContext>::new("outer")
        .with_tool(nested.as_tool("nested_agent", "Run nested agent"))
        .with_model_instance(outer_model);

    let first = Runner::run_with_options(
        &outer,
        "hi",
        RunOptions::with_run_context(RunContext::new(AppContext {
            user: "original".into(),
        })),
    )
    .await
    .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    assert_eq!(first.interruptions[0].tool_name, "needs_ok");
    let mut state = first.to_state();
    let item = first.interruptions[0].clone();
    state.approve(&item, false);
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    assert!(!restored.nested_agent_tool_states.is_empty());

    let resumed = Runner::run_with_options(
        &outer,
        restored,
        RunOptions::new(AppContext {
            user: "reviewer".into(),
        }),
    )
    .await
    .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(resumed.interruptions.is_empty());
    assert_eq!(*seen.lock().unwrap(), vec!["reviewer".to_string()]);
    assert_eq!(resumed.context_wrapper.context().user, "reviewer");
}

/// A model that asks for `tool_calls` (named, with `{prefix}-{tool}` call ids) until a tool
/// output is in its input, then answers `final_text`. Port of
/// `make_tool_calls_then_final_model`.
fn tool_calls_then_final(
    tool_calls: &[(&str, &str)],
    final_text: &str,
    prefix: &str,
) -> Arc<FakeModel> {
    let model = fake();
    for _ in 0..2 {
        let calls: Vec<(String, String)> = tool_calls
            .iter()
            .map(|(n, a)| (n.to_string(), a.to_string()))
            .collect();
        let final_text = final_text.to_string();
        let prefix = prefix.to_string();
        model.add_responder(move |request| {
            let has_output = request
                .input
                .iter()
                .any(|i| i.item_type() == Some("function_call_output"));
            Ok(if has_output {
                vec![assistant_message(final_text.clone())]
            } else {
                calls
                    .iter()
                    .map(|(name, args)| {
                        function_call_with_id(name, args, format!("{prefix}-{name}"))
                    })
                    .collect()
            })
        });
    }
    model
}

fn sensitive_tool(label: &str, log: &Log) -> FunctionTool<()> {
    let label = label.to_string();
    let log = log.clone();
    FunctionTool::<()>::new("inner_sensitive_tool", "Sensitive", move |_c, a: Value| {
        let label = label.clone();
        let log = log.clone();
        async move {
            let text = a["text"].as_str().unwrap_or("").to_string();
            log.lock().unwrap().push(format!("{label}:{text}"));
            Ok(format!("approved:{text}"))
        }
    })
    .with_needs_approval(true)
}

fn worker(name: &str, label: &str, log: &Log) -> Agent<()> {
    Agent::<()>::new(name)
        .with_tool(sensitive_tool(label, log))
        .with_model_instance(tool_calls_then_final(
            &[("inner_sensitive_tool", r#"{"text":"hello"}"#)],
            &format!("{label}-complete"),
            label,
        ))
}

#[tokio::test]
async fn nested_agent_tool_hitl_resume_with_duplicate_agent_names() {
    for graph in [
        "inner_shares_outer_name",
        "sibling_agents_share_name",
        "cloned_sibling_agents",
    ] {
        for approve in [true, false] {
            let log: Log = Arc::default();
            let (outer_name, target, sibling) = match graph {
                "inner_shares_outer_name" => {
                    ("Assistant", worker("Assistant", "target", &log), None)
                }
                "sibling_agents_share_name" => (
                    "OuterAgent",
                    worker("Worker", "target", &log),
                    Some(worker("Worker", "sibling", &log)),
                ),
                _ => {
                    let base = Agent::<()>::new("Worker");
                    let target = base
                        .clone()
                        .with_tool(sensitive_tool("target", &log))
                        .with_model_instance(tool_calls_then_final(
                            &[("inner_sensitive_tool", r#"{"text":"hello"}"#)],
                            "target-complete",
                            "target",
                        ));
                    let sibling = base
                        .clone()
                        .with_tool(sensitive_tool("sibling", &log))
                        .with_model_instance(tool_calls_then_final(
                            &[("inner_sensitive_tool", r#"{"text":"hello"}"#)],
                            "sibling-complete",
                            "sibling",
                        ));
                    ("OuterAgent", target, Some(sibling))
                }
            };
            // The sibling is registered first, so the target needs a suffixed identity.
            let mut outer = Agent::<()>::new(outer_name);
            if let Some(sibling) = &sibling {
                outer =
                    outer.with_tool(sibling.as_tool("sibling_agent_tool", "Sibling agent tool"));
            }
            let outer = outer
                .with_tool(target.as_tool("target_agent_tool", "Target agent tool"))
                .with_model_instance(tool_calls_then_final(
                    &[("target_agent_tool", r#"{"input":"hello"}"#)],
                    "outer-complete",
                    "outer",
                ));

            let first = Runner::run(&outer, "start").await.unwrap();
            assert_eq!(first.final_output_text(), None);
            assert_eq!(first.interruptions.len(), 1);
            let wire: Value =
                serde_json::from_str(&first.to_state().to_json_string().unwrap()).unwrap();
            drop(first);
            let mut restored = RunState::from_json(wire).unwrap();
            let interruptions = restored.get_interruptions();
            assert_eq!(interruptions.len(), 1);
            assert_eq!(interruptions[0].agent, target.name);
            if approve {
                restored.approve(&interruptions[0], false);
            } else {
                restored.reject(&interruptions[0], false, None);
            }
            let resumed = Runner::run(&outer, restored).await.unwrap();
            let label = format!("graph={graph} approve={approve}");
            assert_eq!(
                resumed.final_output_text().as_deref(),
                Some("outer-complete"),
                "{label}"
            );
            assert!(resumed.interruptions.is_empty(), "{label}");
            let expected: Vec<String> = if approve {
                vec!["target:hello".into()]
            } else {
                Vec::new()
            };
            assert_eq!(*log.lock().unwrap(), expected, "{label}");
        }
    }
}

#[tokio::test]
async fn multiple_nested_agent_tool_hitl_resumes_with_duplicate_agent_names() {
    let log: Log = Arc::default();
    let first_worker = worker("Worker", "first", &log);
    let second_worker = worker("Worker", "second", &log);
    let outer = Agent::<()>::new("OuterAgent")
        .with_tool(first_worker.as_tool("first_agent_tool", "First agent tool"))
        .with_tool(second_worker.as_tool("second_agent_tool", "Second agent tool"))
        .with_model_instance(tool_calls_then_final(
            &[
                ("first_agent_tool", r#"{"input":"hello"}"#),
                ("second_agent_tool", r#"{"input":"hello"}"#),
            ],
            "outer-complete",
            "outer",
        ));

    let first = Runner::run(&outer, "start").await.unwrap();
    assert_eq!(first.final_output_text(), None);
    assert_eq!(first.interruptions.len(), 2);
    let wire: Value = serde_json::from_str(&first.to_state().to_json_string().unwrap()).unwrap();
    let mut restored = RunState::from_json(wire).unwrap();
    for interruption in restored.get_interruptions() {
        if interruption.call_id().starts_with("second") {
            restored.approve(&interruption, false);
        } else {
            restored.reject(&interruption, false, None);
        }
    }
    let resumed = Runner::run(&outer, restored).await.unwrap();
    assert_eq!(
        resumed.final_output_text().as_deref(),
        Some("outer-complete")
    );
    assert!(resumed.interruptions.is_empty());
    assert_eq!(*log.lock().unwrap(), vec!["second:hello".to_string()]);
}
