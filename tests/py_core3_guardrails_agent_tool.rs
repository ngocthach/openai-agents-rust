//! Ported from the Python SDK's `tests/test_agent_as_tool_output_guardrails.py`: output
//! guardrails on an agent used as a tool, and the output the parent run receives from it.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::{RunItemKind, ToolApprovalItem};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call, refusal_message,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, OutputGuardrail, RunConfig, RunOptions, RunState, Runner,
};
use serde_json::{Value, json};

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

type Checked = Arc<Mutex<Vec<(String, Value)>>>;

/// `check_output` from the Python test: records `(agent name, output)` and trips on "internal".
fn check_output(checked: Checked) -> OutputGuardrail<()> {
    OutputGuardrail::new("check_output", move |_c, agent, output: Value| {
        let checked = checked.clone();
        async move {
            let trips = output == json!("internal");
            checked.lock().unwrap().push((agent.name.clone(), output));
            Ok(GuardrailFunctionOutput::new(Value::Null, trips))
        }
    })
}

/// Ignores the nested run's stream events; the streamed branch only needs a subscriber.
fn noop_stream_event(
    _event: openai_agents_sdk::agent::AgentToolStreamEvent<()>,
) -> futures::future::BoxFuture<'static, ()> {
    Box::pin(async {})
}

fn lookup_tool(returns_internal: bool) -> FunctionTool<()> {
    FunctionTool::new("lookup", "", move |_c, _a: Value| async move {
        Ok(if returns_internal { "internal" } else { "" }.to_string())
    })
}

fn tool_outputs(items: &[openai_agents_sdk::items::RunItem]) -> Vec<Value> {
    items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect()
}

/// The `output` of every `function_call_output` item in a model request's input.
fn forwarded_outputs(input: &[openai_agents_sdk::items::InputItem]) -> Vec<String> {
    input
        .iter()
        .filter(|item| item.0.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .map(|item| item.0["output"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// One parametrised case of the Python test.
async fn run_case(
    streamed: bool,
    guardrail_owner: &str,
    fallback_source: &str,
    completion: &str,
    custom: bool,
) {
    disable_tracing();
    let checked: Checked = Arc::new(Mutex::new(Vec::new()));

    let mut first_turn = vec![function_call("lookup", "{}")];
    if fallback_source == "message" {
        first_turn.insert(0, assistant_message("internal"));
    }
    let final_item = match completion {
        "nonempty" => assistant_message("done"),
        "none" => assistant_message(r#"{"response":null}"#),
        "refusal" => refusal_message("Cannot answer"),
        _ => assistant_message(""),
    };
    let worker_model = Arc::new(ScriptedModel::new());
    worker_model
        .enqueue(ScriptedStep::output(first_turn))
        .unwrap();
    worker_model
        .enqueue(ScriptedStep::output(vec![final_item]))
        .unwrap();

    let mut worker = Agent::<()>::new("worker")
        .with_model_instance(worker_model.clone())
        .with_tool(lookup_tool(fallback_source == "tool"));
    if completion == "none" {
        worker = worker.with_output_type::<()>();
    }
    if matches!(guardrail_owner, "agent" | "handoff") {
        worker = worker.with_output_guardrail(check_output(checked.clone()));
    }
    let worker = Arc::new(worker);

    let nested = if guardrail_owner == "handoff" {
        let delegator_model = Arc::new(ScriptedModel::new());
        delegator_model
            .enqueue(ScriptedStep::output(vec![function_call(
                "transfer_to_worker",
                "{}",
            )]))
            .unwrap();
        Arc::new(
            Agent::<()>::new("delegator")
                .with_model_instance(delegator_model)
                .with_handoff(worker.clone()),
        )
    } else {
        worker.clone()
    };

    let mut options = AgentAsToolOptions::<()> {
        tool_name: Some("delegate".into()),
        tool_description: Some("Delegate to a worker".into()),
        failure_error_function: None,
        ..Default::default()
    };
    if streamed {
        options.on_stream = Some(Arc::new(noop_stream_event));
    }
    if guardrail_owner == "config" {
        options.run_config =
            Some(RunConfig::default().with_output_guardrail(check_output(checked.clone())));
    }
    if custom {
        options.custom_output_extractor = Some(Arc::new(|_result| {
            Box::pin(async { Ok("custom output".to_string()) })
        }));
    }
    // Python passes `failure_error_function=None`, which re-raises nested failures.
    let nested_tool = nested
        .as_tool_with(options)
        .without_failure_error_function();

    let parent_model = Arc::new(ScriptedModel::new());
    parent_model
        .enqueue(ScriptedStep::output(vec![function_call(
            "delegate",
            r#"{"input":"work"}"#,
        )]))
        .unwrap();
    parent_model
        .enqueue(ScriptedStep::output(vec![assistant_message("parent done")]))
        .unwrap();
    let parent = Agent::<()>::new("parent")
        .with_model_instance(parent_model.clone())
        .with_tool(nested_tool);

    let outcome = if streamed {
        let mut result = Runner::run_streamed_with_options(&parent, "go", RunOptions::new(()));
        let error = {
            let mut events = std::pin::pin!(result.stream_events());
            let mut error = None;
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                    break;
                }
            }
            error
        };
        match error {
            Some(e) => Err(e),
            None => Ok(result.into_result().expect("completed result")),
        }
    } else {
        Runner::run(&parent, "go").await
    };

    if completion == "refusal" {
        let err = outcome.expect_err("refusal must propagate");
        assert!(
            matches!(err, AgentsError::ModelRefusalError { .. }),
            "{err:?}"
        );
        assert!(checked.lock().unwrap().is_empty());
        assert_eq!(parent_model.calls().len(), 1);
        return;
    }
    let result = outcome.unwrap();

    let expected_final: Value = match completion {
        "nonempty" => json!("done"),
        "none" => Value::Null,
        _ => json!(""),
    };
    let expected_export: Value = if custom {
        json!("custom output")
    } else if guardrail_owner == "none" {
        json!("internal")
    } else {
        expected_final.clone()
    };
    let expected_checked: Vec<(String, Value)> = if guardrail_owner == "none" {
        Vec::new()
    } else {
        vec![("worker".to_string(), expected_final)]
    };
    assert_eq!(*checked.lock().unwrap(), expected_checked);
    // Python exports a `None` output as `None`. Tool outputs are text in Rust, so a null
    // output is exported as its JSON text, `null`.
    let expected_export = match expected_export {
        Value::Null => json!("null"),
        other => other,
    };
    assert_eq!(
        tool_outputs(&result.new_items),
        vec![expected_export.clone()]
    );

    let parent_calls = parent_model.calls();
    let forwarded = forwarded_outputs(&parent_calls[1].request.input);
    let expected_text = expected_export.as_str().unwrap().to_string();
    assert_eq!(forwarded, vec![expected_text]);
}

macro_rules! agent_tool_output_guardrail_case {
    ($name:ident, $streamed:expr, $owner:expr, $fallback:expr, $completion:expr, $custom:expr) => {
        #[tokio::test]
        async fn $name() {
            run_case($streamed, $owner, $fallback, $completion, $custom).await;
        }
    };
}

agent_tool_output_guardrail_case!(
    agent_guard_empty_output_is_exported_non_streamed,
    false,
    "agent",
    "tool",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    agent_guard_empty_output_is_exported_streamed,
    true,
    "agent",
    "tool",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    config_guard_empty_output_is_exported_non_streamed,
    false,
    "config",
    "message",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    config_guard_empty_output_is_exported_streamed,
    true,
    "config",
    "message",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    handoff_guard_empty_output_is_exported_non_streamed,
    false,
    "handoff",
    "tool",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    handoff_guard_empty_output_is_exported_streamed,
    true,
    "handoff",
    "tool",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    agent_guard_none_output_is_exported_non_streamed,
    false,
    "agent",
    "tool",
    "none",
    false
);
agent_tool_output_guardrail_case!(
    agent_guard_none_output_is_exported_streamed,
    true,
    "agent",
    "tool",
    "none",
    false
);
agent_tool_output_guardrail_case!(
    agent_guard_nonempty_output_is_exported_non_streamed,
    false,
    "agent",
    "tool",
    "nonempty",
    false
);
agent_tool_output_guardrail_case!(
    agent_guard_nonempty_output_is_exported_streamed,
    true,
    "agent",
    "tool",
    "nonempty",
    false
);
agent_tool_output_guardrail_case!(
    unguarded_tool_fallback_is_exported_non_streamed,
    false,
    "none",
    "tool",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    unguarded_tool_fallback_is_exported_streamed,
    true,
    "none",
    "tool",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    unguarded_message_fallback_is_exported_non_streamed,
    false,
    "none",
    "message",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    unguarded_message_fallback_is_exported_streamed,
    true,
    "none",
    "message",
    "empty",
    false
);
agent_tool_output_guardrail_case!(
    agent_guard_refusal_propagates_non_streamed,
    false,
    "agent",
    "tool",
    "refusal",
    false
);
agent_tool_output_guardrail_case!(
    agent_guard_refusal_propagates_streamed,
    true,
    "agent",
    "tool",
    "refusal",
    false
);
agent_tool_output_guardrail_case!(
    custom_extractor_wins_over_guarded_empty_output_non_streamed,
    false,
    "agent",
    "tool",
    "empty",
    true
);
agent_tool_output_guardrail_case!(
    custom_extractor_wins_over_guarded_empty_output_streamed,
    true,
    "agent",
    "tool",
    "empty",
    true
);

/// Approval-gated worker tool, paused then resumed: the guarded empty output is checked once,
/// after the approval, and is exported as the tool result.
async fn approval_resume_case(streamed: bool) {
    disable_tracing();
    let checked: Checked = Arc::new(Mutex::new(Vec::new()));
    let lookup = FunctionTool::new("lookup", "", |_c, _a: Value| async move {
        Ok("internal".to_string())
    })
    .with_needs_approval(true);
    let worker_model = Arc::new(ScriptedModel::new());
    worker_model
        .enqueue(ScriptedStep::output(vec![function_call("lookup", "{}")]))
        .unwrap();
    worker_model
        .enqueue(ScriptedStep::output(vec![assistant_message("")]))
        .unwrap();
    let worker = Arc::new(
        Agent::<()>::new("worker")
            .with_model_instance(worker_model)
            .with_tool(lookup)
            .with_output_guardrail(check_output(checked.clone())),
    );
    let nested_tool = worker
        .as_tool_with(AgentAsToolOptions {
            tool_name: Some("delegate".into()),
            tool_description: Some("Delegate".into()),
            on_stream: streamed.then(|| Arc::new(noop_stream_event) as _),
            ..Default::default()
        })
        .without_failure_error_function();
    let parent_model = Arc::new(ScriptedModel::new());
    parent_model
        .enqueue(ScriptedStep::output(vec![function_call(
            "delegate",
            r#"{"input":"work"}"#,
        )]))
        .unwrap();
    parent_model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let parent = Agent::<()>::new("parent")
        .with_model_instance(parent_model.clone())
        .with_tool(nested_tool);

    let paused = Runner::run(&parent, "go").await.unwrap();
    assert_eq!(paused.interruptions.len(), 1);
    assert!(checked.lock().unwrap().is_empty());
    assert!(tool_outputs(&paused.new_items).is_empty());

    let mut state: RunState = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
    let approval: ToolApprovalItem = state.get_interruptions()[0].clone();
    state.approve(&approval, true);
    let resumed = Runner::run(&parent, state).await.unwrap();

    assert_eq!(
        *checked.lock().unwrap(),
        vec![("worker".to_string(), json!(""))]
    );
    assert_eq!(tool_outputs(&resumed.new_items), vec![json!("")]);
    let forwarded = forwarded_outputs(&parent_model.calls()[1].request.input);
    assert_eq!(forwarded, vec![String::new()]);
}

#[tokio::test]
async fn guarded_empty_agent_tool_output_after_approval_resume_non_streamed() {
    approval_resume_case(false).await;
}

#[tokio::test]
async fn guarded_empty_agent_tool_output_after_approval_resume_streamed() {
    approval_resume_case(true).await;
}
