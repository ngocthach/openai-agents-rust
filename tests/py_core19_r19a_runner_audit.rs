//! Runner cases from `tests/test_agent_runner.py` and `tests/test_agent_runner_streamed.py`
//! that earlier rounds left in the Python-only bucket and that are observable through the
//! public API.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, Runner};
use serde_json::Value;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn tool_outputs(model: &FakeModel) -> Vec<Value> {
    model
        .calls()
        .last()
        .expect("a model call")
        .input
        .iter()
        .map(|item| item.0.clone())
        .filter(|item| item["type"] == "function_call_output")
        .collect()
}

/// Python cancels the tool from inside its body (`raise asyncio.CancelledError`); a Rust tool
/// body cannot cancel itself, so the analog is a body that fails: the default failure text
/// reaches the model and the run still finishes.
fn failing_tool() -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args("cancel_tool", "c", |_c| async {
        Err::<String, _>(AgentsError::other("tool-cancelled"))
    })
}

async fn run_streamed(agent: Agent<()>) -> (Option<String>, usize) {
    let mut streamed = Runner::run_streamed(agent, "user_message");
    {
        let events = streamed.stream_events();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            event.expect("stream event");
        }
    }
    let result = streamed.result().expect("the run completed");
    (result.final_output_text(), result.raw_responses.len())
}

const DEFAULT_FAILURE: &str = "An error occurred while running the tool. Please try again.";

/// Python: `test_streamed_parallel_tool_call_with_cancelled_sibling_reaches_final_output`.
#[tokio::test]
async fn a_streamed_run_with_a_failing_sibling_reaches_the_final_output() {
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("ok_tool", "{}", "call_ok"),
        function_call_with_id("cancel_tool", "{}", "call_cancel"),
    ]);
    model.add_turn(vec![assistant_message("final answer")]);
    let ok = FunctionTool::<()>::new_no_args("ok_tool", "ok", |_c| async { Ok("ok".to_string()) });
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(ok)
        .with_tool(failing_tool());

    let (output, responses) = run_streamed(agent).await;

    assert_eq!(output.as_deref(), Some("final answer"));
    assert_eq!(responses, 2);
    let outputs = tool_outputs(&model);
    assert_eq!(outputs.len(), 2, "{outputs:?}");
    assert_eq!(outputs[0]["call_id"], "call_ok");
    assert_eq!(outputs[0]["output"], "ok");
    assert_eq!(outputs[1]["call_id"], "call_cancel");
    assert!(
        outputs[1]["output"]
            .as_str()
            .unwrap()
            .starts_with(DEFAULT_FAILURE),
        "{outputs:?}"
    );
}

/// Python: `test_streamed_single_tool_call_with_cancelled_tool_reaches_final_output`.
#[tokio::test]
async fn a_streamed_run_with_a_single_failing_tool_reaches_the_final_output() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "cancel_tool",
        "{}",
        "call_cancel",
    )]);
    model.add_turn(vec![assistant_message("final answer")]);
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(failing_tool());

    let (output, responses) = run_streamed(agent).await;

    assert_eq!(output.as_deref(), Some("final answer"));
    assert_eq!(responses, 2);
    let outputs = tool_outputs(&model);
    assert_eq!(outputs.len(), 1, "{outputs:?}");
    assert_eq!(outputs[0]["call_id"], "call_cancel");
    assert!(
        outputs[0]["output"]
            .as_str()
            .unwrap()
            .starts_with(DEFAULT_FAILURE),
        "{outputs:?}"
    );
}

/// Python: `test_execute_approved_tools_rejects_changed_pending_invocation`. A decision made
/// for one payload does not authorize a pending call that shares its call id but carries other
/// arguments: the tool never runs.
#[tokio::test]
async fn an_approval_for_another_payload_does_not_run_the_changed_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let tool = FunctionTool::<()>::new("test_tool", "t", move |_c, args: Value| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(args["value"].as_str().unwrap_or_default().to_string())
        }
    })
    .with_needs_approval(true);
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "test_tool",
        r#"{"value":"changed"}"#,
        "call-shared",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool);

    let parked = Runner::run(&agent, "go").await.expect("the run parks");
    assert_eq!(parked.interruptions.len(), 1);
    let mut state = parked.to_state();
    let mut approved_call = FunctionCall::new("test_tool", r#"{"value":"safe"}"#);
    approved_call.call_id = "call-shared".to_string();
    let approved = ToolApprovalItem {
        agent: "test".to_string(),
        raw_item: approved_call,
        tool_name: "test_tool".to_string(),
        tool_origin: None,
    };
    state.approve(&approved, false);

    let outcome = Runner::run(&agent, state).await;

    assert_eq!(calls.load(Ordering::SeqCst), 0, "{outcome:?}");
    let error = outcome.expect_err("the changed call is refused");
    assert!(error.to_string().contains("unique call ID"), "{error}");
}
