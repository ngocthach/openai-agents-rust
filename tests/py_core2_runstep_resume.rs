//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py` (the tests whose
//! behaviour is observable through `Runner` and `RunState`; the remainder monkeypatches
//! private run-loop functions and is classified in `plans/reports/py-test-parity-core2-runstep.md`).

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, InMemorySession, RunConfig, RunInput, RunOptions, RunState, Runner, Session,
    ToolNotFoundBehavior,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

fn call(name: &str, args: Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

fn needs_ok(prefix: &'static str) -> FunctionTool<()> {
    #[derive(Deserialize, JsonSchema)]
    struct Args {
        text: String,
    }
    FunctionTool::new("needs_ok", "", move |_c, a: Args| async move {
        Ok(format!("{prefix}{}", a.text))
    })
    .with_needs_approval(true)
}

/// Run under a server-managed continuation mode, streamed or not, returning the snapshot and
/// the final output.
async fn run_server_managed(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    use_conversation_id: bool,
    streaming: bool,
) -> (RunState, Option<Value>) {
    let config =
        RunConfig::default().with_tool_not_found_behavior(ToolNotFoundBehavior::ReturnErrorToModel);
    let mut options = RunOptions::new(()).run_config(config);
    options = if use_conversation_id {
        options.conversation_id("conv-resume")
    } else {
        options.auto_previous_response_id(true)
    };
    if streaming {
        let mut streamed = Runner::run_streamed_with_options(agent.clone(), input, options);
        {
            let mut events = std::pin::pin!(streamed.stream_events());
            while let Some(ev) = events.next().await {
                ev.unwrap();
            }
        }
        (
            streamed.to_state().unwrap(),
            streamed.final_output().cloned(),
        )
    } else {
        let result = Runner::run_with_options(agent.clone(), input, options)
            .await
            .unwrap();
        (result.to_state(), result.final_output.clone())
    }
}

/// The tool outputs the model received, from call `first` onward. Mirrors `_sent_tool_outputs`.
fn sent_tool_outputs(model: &ScriptedModel, first: usize) -> Vec<(String, String)> {
    model.calls()[first..]
        .iter()
        .flat_map(|c| c.request.input.clone())
        .filter(|i| i.0["type"] == "function_call_output")
        .map(|i| {
            (
                i.0["call_id"].as_str().unwrap_or_default().to_string(),
                i.0["output"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

async fn tool_not_found_case(streaming: bool, serialize_state: bool, use_conversation_id: bool) {
    disable_tracing();
    let model = scripted(vec![
        vec![
            call("needs_ok", json!({"text": "one"}), "call-approval"),
            call("missing_tool", json!({}), "call-missing"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(needs_ok("")),
    );
    let (mut state, _) = run_server_managed(
        &agent,
        "Use needs_ok and missing_tool",
        use_conversation_id,
        streaming,
    )
    .await;
    if serialize_state {
        state = RunState::from_json(state.to_json().unwrap()).unwrap();
    }
    let interruptions = state.get_interruptions();
    assert_eq!(
        interruptions
            .iter()
            .map(|i| i.call_id().to_string())
            .collect::<Vec<_>>(),
        vec!["call-approval"]
    );
    state.approve(&interruptions[0], false);
    let (_, output) = run_server_managed(&agent, state, use_conversation_id, streaming).await;
    assert_eq!(output, Some(json!("done")));
    let delivered = sent_tool_outputs(&model, 1);
    let mut ids: Vec<&str> = delivered.iter().map(|(id, _)| id.as_str()).collect();
    ids.sort();
    assert_eq!(ids, vec!["call-approval", "call-missing"]);
    let missing = &delivered
        .iter()
        .find(|(id, _)| id == "call-missing")
        .unwrap()
        .1;
    assert!(missing.contains("missing_tool"), "{missing}");
}

#[tokio::test]
async fn resumed_server_managed_run_sends_tool_not_found_output_matrix() {
    for streaming in [false, true] {
        for serialize_state in [false, true] {
            for use_conversation_id in [false, true] {
                tool_not_found_case(streaming, serialize_state, use_conversation_id).await;
            }
        }
    }
}

#[tokio::test]
async fn resumed_server_managed_run_sends_each_tool_output_once() {
    disable_tracing();
    let model = scripted(vec![
        vec![
            call("needs_ok", json!({"text": "a"}), "call-a"),
            call("needs_ok", json!({"text": "b"}), "call-b"),
            call("missing_tool", json!({}), "call-missing"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(needs_ok("ok:")),
    );
    let (mut state, mut output) =
        run_server_managed(&agent, "Use needs_ok twice and missing_tool", false, false).await;
    for expected_model_calls in [1, 2] {
        state = RunState::from_json(state.to_json().unwrap()).unwrap();
        let interruptions = state.get_interruptions();
        assert!(!interruptions.is_empty());
        state.approve(&interruptions[0], false);
        (state, output) = run_server_managed(&agent, state, false, false).await;
        // Approving only the first gated call resumes without asking the model again.
        assert_eq!(model.calls().len(), expected_model_calls);
    }
    assert_eq!(output, Some(json!("done")));
    let mut ids: Vec<String> = sent_tool_outputs(&model, 1)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["call-a", "call-b", "call-missing"]);
}

#[tokio::test]
async fn resumed_approval_does_not_duplicate_session_items() {
    disable_tracing();
    let tool = FunctionTool::new_no_args("test_tool", "", |_c| async move { Ok("tool_result") })
        .with_needs_approval(true);
    let model = scripted(vec![
        vec![call("test_tool", json!({}), "call-resume")],
        vec![assistant_message("done")],
    ]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(tool),
    );
    let session = Arc::new(InMemorySession::new("resume"));
    let first = Runner::run_with_options(
        agent.clone(),
        "Use test_tool",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert!(!first.interruptions.is_empty());
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let resumed = Runner::run_with_options(
        agent.clone(),
        state,
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(resumed.final_output, Some(json!("done")));
    let saved = session.get_items(None).await.unwrap();
    let count = |ty: &str| {
        saved
            .iter()
            .filter(|i| i.0["type"] == ty && i.0["call_id"] == "call-resume")
            .count()
    };
    assert_eq!(count("function_call"), 1);
    assert_eq!(count("function_call_output"), 1);
}
