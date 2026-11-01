//! Parity ports of `tests/utils/test_pretty_print_and_items.py`'s run-error summary: the
//! summary reports input and output guardrail results alongside the other counts.

use openai_agents_sdk::error::RunErrorDetails;
use openai_agents_sdk::guardrail::{
    GuardrailFunctionOutput, InputGuardrailResult, OutputGuardrailResult,
};
use serde_json::Value;

fn details(inputs: usize, outputs: usize) -> RunErrorDetails {
    RunErrorDetails {
        last_agent: "test".into(),
        input_guardrail_results: (0..inputs)
            .map(|_| InputGuardrailResult {
                guardrail_name: "input".into(),
                output: GuardrailFunctionOutput::pass(),
            })
            .collect(),
        output_guardrail_results: (0..outputs)
            .map(|_| OutputGuardrailResult {
                guardrail_name: "output".into(),
                agent_name: "test".into(),
                agent_output: Value::Null,
                output: GuardrailFunctionOutput::pass(),
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn pretty_print_run_error_details_includes_output_guardrail_count() {
    let text = details(1, 2).pretty_print();
    assert!(text.contains("1 input guardrail result(s)"), "{text}");
    assert!(text.contains("2 output guardrail result(s)"), "{text}");
}

#[test]
fn pretty_print_run_error_details_zero_output_guardrails() {
    let text = details(0, 0).pretty_print();
    assert!(text.contains("0 output guardrail result(s)"), "{text}");
}
