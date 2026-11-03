//! Leftovers of `test_tool_guardrails.py` (the SSN output guardrail, the tripwire exceptions) and
//! `test_usage.py` (`test_runner_run_carries_request_usage_entries`).

use std::sync::Arc;

use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::ToolOutput;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailBehavior, ToolGuardrailFunctionOutput, ToolOutputGuardrail,
    ToolOutputGuardrailData,
};
use openai_agents_sdk::usage::{InputTokensDetails, OutputTokensDetails, RequestUsage, Usage};
use openai_agents_sdk::{Agent, AgentsError, RunContext, Runner};
use serde_json::json;

fn output_data(output: &str) -> ToolOutputGuardrailData<()> {
    ToolOutputGuardrailData {
        context: ToolContext::direct(RunContext::new(()), "test", "test_tool", "{}"),
        agent_name: "test".into(),
        output: ToolOutput::text(output),
    }
}

#[tokio::test]
async fn ssn_blocking_output_guardrail() {
    let guardrail = ToolOutputGuardrail::new(
        "check_for_ssn",
        |d: ToolOutputGuardrailData<()>| async move {
            let text = d.output.to_model_string().to_lowercase();
            if text.contains("ssn") || text.contains("123-45-6789") {
                Ok(ToolGuardrailFunctionOutput::raise_exception()
                    .with_output_info(json!({"blocked_pattern": "SSN"})))
            } else {
                Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("safe_output")))
            }
        },
    );

    let result = guardrail
        .run(output_data("User SSN is 123-45-6789"))
        .await
        .unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::RaiseException);
    assert_eq!(result.output_info["blocked_pattern"], json!("SSN"));

    let result = guardrail
        .run(output_data("User name is John Doe"))
        .await
        .unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::Allow);
    assert_eq!(result.output_info, json!("safe_output"));
}

/// `test_tool_input_guardrail_exception` / `test_tool_output_guardrail_exception`. Python's
/// message names the guardrail's class (`ToolInputGuardrail`); Rust's names the guardrail itself,
/// which is what identifies it.
#[test]
fn tool_guardrail_tripwire_errors_carry_guardrail_and_output() {
    let output = ToolGuardrailFunctionOutput::raise_exception().with_output_info(json!("test"));

    let input = AgentsError::ToolInputGuardrailTripwireTriggered {
        guardrail_name: "test_guardrail".into(),
        output: Box::new(output.clone()),
        run_data: None,
    };
    assert_eq!(
        input.to_string(),
        "Tool input guardrail test_guardrail triggered tripwire"
    );
    let AgentsError::ToolInputGuardrailTripwireTriggered {
        guardrail_name,
        output: carried,
        ..
    } = &input
    else {
        unreachable!()
    };
    assert_eq!(guardrail_name, "test_guardrail");
    assert_eq!(**carried, output);

    let output_error = AgentsError::ToolOutputGuardrailTripwireTriggered {
        guardrail_name: "test_guardrail".into(),
        output: Box::new(output.clone()),
        run_data: None,
    };
    assert_eq!(
        output_error.to_string(),
        "Tool output guardrail test_guardrail triggered tripwire"
    );
}

#[tokio::test]
async fn runner_run_carries_request_usage_entries() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let usage = Usage {
        requests: 1,
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        request_usage_entries: vec![RequestUsage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            input_tokens_details: InputTokensDetails {
                cached_tokens: 0,
                cache_write_tokens: 0,
            },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 0,
            },
        }],
        ..Default::default()
    };
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    model.set_default_usage(Some(usage));
    let agent = Agent::<()>::new("usage-agent").with_model_instance(model);

    let result = Runner::run(&agent, "hi").await.unwrap();

    let propagated = result.context_wrapper.usage();
    assert_eq!(propagated.requests, 1);
    assert_eq!(propagated.total_tokens, 15);
    assert_eq!(propagated.request_usage_entries.len(), 1);
    let entry = &propagated.request_usage_entries[0];
    assert_eq!(entry.input_tokens, 10);
    assert_eq!(entry.output_tokens, 5);
    assert_eq!(entry.total_tokens, 15);
}
