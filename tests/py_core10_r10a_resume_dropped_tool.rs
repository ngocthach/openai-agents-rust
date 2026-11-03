//! Ports of the Python SDK's resume cases where a parked tool call lost its tool (disabled or
//! removed) between the snapshot and the resume (`tests/test_run_state.py`, nested agent tool
//! state surviving dropped functions, and the reconciliation in `resolve_interrupted_turn`).
//!
//! Python restores the parked response without a function whose tool is gone and carries on:
//! a call nothing waits on is dropped, an undecided approval stays pending, a rejected one still
//! answers the model, and only an approved call fails with "Tool X not found in agent Y".

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::{InputItem, RunInput};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

#[derive(Clone, Copy, Debug)]
enum Loss {
    Removed,
    Disabled,
}

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn tool(name: &'static str, approval: bool, ran: Arc<Mutex<Vec<String>>>) -> FunctionTool<()> {
    let tool = FunctionTool::new(name, "a tool", move |_ctx, _a: NoArgs| {
        let ran = ran.clone();
        async move {
            ran.lock().unwrap().push(name.to_string());
            Ok(format!("{name} ran"))
        }
    });
    if approval {
        tool.with_needs_approval(true)
    } else {
        tool
    }
}

async fn run_mode(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    streamed: bool,
) -> Result<RunResult<()>, AgentsError> {
    let options = RunOptions::new(());
    if !streamed {
        return Runner::run_with_options(agent.clone(), input, options).await;
    }
    let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream.into_result().expect("finished")),
    }
}

fn outputs(input: &[InputItem]) -> Vec<String> {
    input
        .iter()
        .filter(|i| i.0["type"] == "function_call_output")
        .map(|i| format!("{}={}", i.0["call_id"], i.0["output"]))
        .collect()
}

/// Park two approval calls (`keep` and `lost`) behind one model response, then resume with a
/// graph where `lost` is gone or disabled. `decide` records the decisions on the restored state.
struct Resumed {
    result: Result<RunResult<()>, AgentsError>,
    model: Arc<FakeModel>,
    ran: Arc<Mutex<Vec<String>>>,
}

async fn resume_without_tool(
    loss: Loss,
    streamed: bool,
    decide_keep: Option<bool>,
    decide_lost: Option<bool>,
) -> Resumed {
    let ran = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("lost_tool", "{}", "lost-call"),
        function_call_with_id("keep_tool", "{}", "keep-call"),
    ]);
    let first = Arc::new(
        Agent::<()>::new("agent")
            .with_model_instance(model.clone())
            .with_tool(tool("lost_tool", true, ran.clone()))
            .with_tool(tool("keep_tool", true, ran.clone())),
    );
    let interrupted = run_mode(&first, "start", streamed).await.unwrap();
    assert_eq!(interrupted.interruptions.len(), 2);
    let mut state = interrupted.to_state();
    for item in &interrupted.interruptions {
        let decision = match item.call_id() {
            "keep-call" => decide_keep,
            _ => decide_lost,
        };
        match decision {
            Some(true) => state.approve(item, false),
            Some(false) => state.reject(item, false, None),
            None => {}
        }
    }
    let wire = state.to_json().unwrap();

    let lost = tool("lost_tool", true, ran.clone());
    let lost = match loss {
        Loss::Removed => None,
        Loss::Disabled => Some(lost.with_is_enabled(false)),
    };
    let mut second = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(tool("keep_tool", true, ran.clone()));
    if let Some(lost) = lost {
        second = second.with_tool(lost);
    }
    model.add_turn(vec![assistant_message("done")]);
    let second = Arc::new(second);
    let restored = RunState::from_json(wire).unwrap();
    let result = run_mode(&second, restored, streamed).await;
    Resumed { result, model, ran }
}

#[tokio::test]
async fn an_undecided_call_whose_tool_is_gone_stays_pending_while_the_rest_resumes() {
    for streamed in [false, true] {
        for loss in [Loss::Removed, Loss::Disabled] {
            let label = format!("streamed={streamed} loss={loss:?}");
            let resumed = resume_without_tool(loss, streamed, Some(true), None).await;
            let result = resumed.result.unwrap_or_else(|e| panic!("{label}: {e}"));
            // The approved retained call ran; the undecided lost one is asked again.
            assert_eq!(*resumed.ran.lock().unwrap(), vec!["keep_tool"], "{label}");
            assert_eq!(result.interruptions.len(), 1, "{label}");
            assert_eq!(result.interruptions[0].call_id(), "lost-call", "{label}");
        }
    }
}

#[tokio::test]
async fn a_rejected_call_whose_tool_is_gone_still_answers_the_model() {
    for streamed in [false, true] {
        for loss in [Loss::Removed, Loss::Disabled] {
            let label = format!("streamed={streamed} loss={loss:?}");
            let resumed = resume_without_tool(loss, streamed, Some(true), Some(false)).await;
            let result = resumed.result.unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(*resumed.ran.lock().unwrap(), vec!["keep_tool"], "{label}");
            assert_eq!(
                result.final_output_text().as_deref(),
                Some("done"),
                "{label}"
            );
            let sent = outputs(&resumed.model.last_call().unwrap().input);
            assert!(
                sent.iter().any(|o| o.starts_with("\"keep-call\"=")),
                "{label}: {sent:?}"
            );
            assert!(
                sent.iter().any(|o| o.starts_with("\"lost-call\"=")),
                "{label}: the rejection answers the lost call: {sent:?}"
            );
        }
    }
}

#[tokio::test]
async fn an_approved_call_whose_tool_is_gone_is_refused() {
    for streamed in [false, true] {
        for loss in [Loss::Removed, Loss::Disabled] {
            let label = format!("streamed={streamed} loss={loss:?}");
            let resumed = resume_without_tool(loss, streamed, Some(true), Some(true)).await;
            let message = resumed.result.err().map(|e| e.to_string());
            assert_eq!(
                message.as_deref(),
                Some("Tool lost_tool not found in agent agent"),
                "{label}"
            );
            assert!(resumed.ran.lock().unwrap().is_empty(), "{label}");
        }
    }
}

/// A call that already ran has nothing left to resume: losing its tool afterwards neither
/// refuses the resume nor runs it again.
#[tokio::test]
async fn a_completed_call_whose_tool_is_gone_does_not_stop_the_resume() {
    for streamed in [false, true] {
        for loss in [Loss::Removed, Loss::Disabled] {
            let label = format!("streamed={streamed} loss={loss:?}");
            let ran = Arc::new(Mutex::new(Vec::new()));
            let model = setup();
            model.add_turn(vec![
                function_call_with_id("early_tool", "{}", "early-call"),
                function_call_with_id("approval_tool", "{}", "approval-call"),
            ]);
            let first = Arc::new(
                Agent::<()>::new("agent")
                    .with_model_instance(model.clone())
                    .with_tool(tool("early_tool", false, ran.clone()))
                    .with_tool(tool("approval_tool", true, ran.clone())),
            );
            let interrupted = run_mode(&first, "start", streamed).await.unwrap();
            assert_eq!(interrupted.interruptions.len(), 1, "{label}");
            let mut state = interrupted.to_state();
            state.approve(&interrupted.interruptions[0], false);
            let wire = state.to_json().unwrap();
            assert_eq!(*ran.lock().unwrap(), vec!["early_tool"], "{label}");

            let mut second = Agent::<()>::new("agent")
                .with_model_instance(model.clone())
                .with_tool(tool("approval_tool", true, ran.clone()));
            if let Loss::Disabled = loss {
                second =
                    second.with_tool(tool("early_tool", false, ran.clone()).with_is_enabled(false));
            }
            model.add_turn(vec![assistant_message("done")]);
            let result = run_mode(
                &Arc::new(second),
                RunState::from_json(wire).unwrap(),
                streamed,
            )
            .await
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                result.final_output_text().as_deref(),
                Some("done"),
                "{label}"
            );
            assert_eq!(
                *ran.lock().unwrap(),
                vec!["early_tool", "approval_tool"],
                "{label}"
            );
        }
    }
}

/// Python: `test_nested_agent_tool_state_survives_when_earlier_function_is_dropped` (disabled and
/// removed) and `test_dropped_nested_agent_tool_state_is_not_moved_to_later_function`: the paused
/// nested run of a retained agent tool resumes with its own state when an earlier function of
/// the same parked response lost its tool.
#[tokio::test]
async fn a_retained_agent_tool_keeps_its_nested_state_when_an_earlier_function_is_dropped() {
    for streamed in [false, true] {
        for loss in [Loss::Removed, Loss::Disabled] {
            let label = format!("streamed={streamed} loss={loss:?}");
            let ran = Arc::new(Mutex::new(Vec::new()));
            let inner_model = setup();
            inner_model.add_turn(vec![function_call_with_id("confirm", "{}", "inner-call")]);
            inner_model.add_turn(vec![assistant_message("inner done")]);
            let build_inner = || {
                Agent::<()>::new("inner")
                    .with_model_instance(inner_model.clone())
                    .with_tool(tool("confirm", true, ran.clone()))
            };
            let outer_model = setup();
            outer_model.add_turn(vec![
                function_call_with_id("conditional_tool", "{}", "conditional-call"),
                function_call_with_id("delegate", r#"{"input":"go"}"#, "nested-call"),
            ]);
            let first = Arc::new(
                Agent::<()>::new("outer")
                    .with_model_instance(outer_model.clone())
                    .with_tool(tool("conditional_tool", false, ran.clone()))
                    .with_tool(build_inner().as_tool("delegate", "Delegate")),
            );
            let interrupted = run_mode(&first, "start", streamed).await.unwrap();
            assert_eq!(interrupted.interruptions.len(), 1, "{label}");
            let mut state = interrupted.to_state();
            state.approve(&interrupted.interruptions[0], false);
            let wire = state.to_json().unwrap();
            ran.lock().unwrap().clear();

            let mut second = Agent::<()>::new("outer")
                .with_model_instance(outer_model.clone())
                .with_tool(build_inner().as_tool("delegate", "Delegate"));
            if let Loss::Disabled = loss {
                second = second
                    .with_tool(tool("conditional_tool", false, ran.clone()).with_is_enabled(false));
            }
            outer_model.add_turn(vec![assistant_message("outer done")]);
            let result = run_mode(
                &Arc::new(second),
                RunState::from_json(wire).unwrap(),
                streamed,
            )
            .await
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                result.final_output_text().as_deref(),
                Some("outer done"),
                "{label}"
            );
            assert_eq!(*ran.lock().unwrap(), vec!["confirm"], "{label}");
        }
    }
}
