//! Ports of cases of `tests/test_agent_runner_streamed.py` that the earlier streamed ports left
//! as Python-only: the usage entries of a streamed run whose model call is retried after a
//! `conversation_locked` error, and the stream events of a tool call that needs approval.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::model_settings::{
    ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
};
use openai_agents_sdk::retry::retry_policies;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, AgentsError, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn conversation_locked_error() -> AgentsError {
    AgentsError::ApiError {
        status: 400,
        message: "locked".into(),
        body: Some(Box::new(
            json!({"error": {"code": "conversation_locked", "message": "locked"}}),
        )),
        request_id: None,
        headers: None,
    }
}

/// Python: `test_streamed_run_preserves_request_usage_entries_after_conversation_locked_retry`.
#[tokio::test]
async fn a_streamed_conversation_locked_retry_keeps_one_usage_entry_per_request() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::raise_error(conversation_locked_error()),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    model.set_default_usage(Some(Usage {
        requests: 1,
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        ..Default::default()
    }));
    let settings = ModelSettings {
        retry: Some(
            ModelRetrySettings {
                max_retries: Some(1),
                backoff: Some(ModelRetryBackoffSettings {
                    initial_delay: Some(0.0),
                    max_delay: Some(0.0),
                    multiplier: Some(1.0),
                    jitter: Some(false),
                }),
                ..Default::default()
            }
            .with_policy(retry_policies::network_error()),
        ),
        ..Default::default()
    };
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_model_settings(settings),
    );

    let mut stream = Runner::run_streamed(agent, "test");
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            event.expect("the retried stream completes");
        }
    }

    let usage = stream.result().unwrap().usage();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.request_usage_entries.len(), 2);
    assert_eq!(usage.request_usage_entries[0].total_tokens, 0);
    assert_eq!(usage.request_usage_entries[1].input_tokens, 10);
    assert_eq!(usage.request_usage_entries[1].output_tokens, 5);
    assert_eq!(usage.request_usage_entries[1].total_tokens, 15);
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

/// Python: `test_stream_step_items_to_queue_handles_tool_approval_item`: a tool call that needs
/// approval streams its call item, and the pending approval itself is never an event.
#[tokio::test]
async fn a_pending_approval_is_not_streamed_as_a_run_item() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![function_call_with_id(
            "test_tool",
            "{}",
            "call_1",
        )])])
        .unwrap(),
    );
    let tool = FunctionTool::<()>::new("test_tool", "a tool", |_ctx, _a: NoArgs| async move {
        Ok("tool_result".to_string())
    })
    .with_needs_approval(true);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(tool),
    );

    let mut stream = Runner::run_streamed(agent, "go");
    let mut names = Vec::new();
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem { name, .. } = event.expect("the run pauses cleanly") {
                names.push(format!("{name:?}"));
            }
        }
    }

    assert_eq!(stream.interruptions().len(), 1);
    assert!(
        !names
            .iter()
            .any(|name| name.to_lowercase().contains("approval")),
        "{names:?}"
    );
    assert!(
        names
            .iter()
            .any(|name| name.to_lowercase().contains("tool")),
        "{names:?}"
    );
}

fn plain_tool() -> FunctionTool<()> {
    FunctionTool::<()>::new("test_tool", "a tool", |_ctx, _a: NoArgs| async move {
        Ok("tool_result".to_string())
    })
}

/// Python: `test_streaming_max_turns_emits_pending_tool_output_events`: a run that exceeds its
/// turn budget on a tool turn still streams that turn's tool call and tool output items.
#[tokio::test]
async fn a_max_turns_failure_streams_the_pending_tool_output_events() {
    use openai_agents_sdk::stream_events::RunItemStreamEventName::{ToolCalled, ToolOutput};
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![function_call_with_id("test_tool", "{}", "call_1")]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(plain_tool()),
    );

    let mut stream = Runner::run_streamed_with_options(
        agent,
        "Use test_tool",
        openai_agents_sdk::RunOptions::new(()).max_turns(Some(1)),
    );
    let mut names = Vec::new();
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            match event {
                Ok(StreamEvent::RunItem { name, .. }) => names.push(name),
                Ok(_) => {}
                Err(e) => error = Some(e),
            }
        }
    }

    assert!(
        matches!(error, Some(AgentsError::MaxTurnsExceeded { .. })),
        "{error:?}"
    );
    assert!(names.contains(&ToolCalled), "{names:?}");
    assert!(names.contains(&ToolOutput), "{names:?}");
}

/// Python: `test_streamed_tool_guardrail_results_survive_handoff`: the guardrail results of a
/// turn that runs a guarded tool and hands off reach the streamed result.
#[tokio::test]
async fn streamed_tool_guardrail_results_survive_a_handoff() {
    use openai_agents_sdk::tool_guardrails::{
        ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
    };
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![
                function_call_with_id("test_tool", "{}", "call_1"),
                function_call_with_id("transfer_to_target", "{}", "call_handoff"),
            ]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
    let guarded = plain_tool()
        .with_input_guardrail(ToolInputGuardrail::new("in", |_d| async {
            Ok(ToolGuardrailFunctionOutput::allow())
        }))
        .with_output_guardrail(ToolOutputGuardrail::new("out", |_d| async {
            Ok(ToolGuardrailFunctionOutput::allow())
        }));
    let agent = Arc::new(
        Agent::<()>::new("source")
            .with_model_instance(model)
            .with_tool(guarded)
            .with_handoff(openai_agents_sdk::handoff(target)),
    );

    let mut stream = Runner::run_streamed(agent, "hello");
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            event.expect("the run completes");
        }
    }

    let result = stream.result().unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(result.tool_input_guardrail_results.len(), 1);
    assert_eq!(result.tool_output_guardrail_results.len(), 1);
}
