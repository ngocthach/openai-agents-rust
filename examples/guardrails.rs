//! An input guardrail that uses another agent to block math homework.
//! Run: cargo run --example guardrails
use std::sync::Arc;

use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, InputItem, RunOptions, Runner,
    flush_traces_async,
};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct MathHomeworkOutput {
    is_math_homework: bool,
    reasoning: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let guardrail_agent = Arc::new(
        Agent::<()>::new("Guardrail check")
            .with_instructions("Check if the user is asking you to do their math homework.")
            .with_output_type::<MathHomeworkOutput>(),
    );
    let math_guardrail = InputGuardrail::new(
        "math_homework",
        move |ctx, _agent, input: Vec<InputItem>| {
            let guardrail_agent = guardrail_agent.clone();
            async move {
                let result = Runner::run_with_options(
                    guardrail_agent,
                    input,
                    RunOptions::with_run_context(ctx),
                )
                .await?;
                let out: MathHomeworkOutput = result.final_output_as()?;
                Ok(GuardrailFunctionOutput::new(
                    serde_json::json!({"reasoning": out.reasoning}),
                    out.is_math_homework,
                ))
            }
        },
    );
    let agent = Agent::<()>::new("Customer support agent")
        .with_instructions(
            "You are a customer support agent. You help customers with their questions.",
        )
        .with_input_guardrail(math_guardrail);

    match Runner::run(&agent, "Hello, can you help me solve for x: 2x + 3 = 11?").await {
        Err(AgentsError::InputGuardrailTripwireTriggered {
            guardrail_result, ..
        }) => {
            println!(
                "Math homework guardrail tripped: {:?}",
                guardrail_result.output.output_info
            )
        }
        Ok(r) => println!(
            "Guardrail didn't trip: {}",
            r.final_output_text().unwrap_or_default()
        ),
        Err(e) => return Err(e.into()),
    }
    flush_traces_async().await;
    Ok(())
}
