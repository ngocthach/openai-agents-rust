//! Routing between specialist agents with handoffs.
//! Run: cargo run --example handoffs
use std::sync::Arc;

use openai_agents_sdk::extensions::handoff_prompt::prompt_with_handoff_instructions;
use openai_agents_sdk::{Agent, Runner, flush_traces_async, handoff};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let history_tutor = Arc::new(
        Agent::<()>::new("History Tutor")
            .with_handoff_description("Specialist agent for historical questions")
            .with_instructions("You answer history questions clearly and concisely."),
    );
    let math_tutor = Arc::new(
        Agent::<()>::new("Math Tutor")
            .with_handoff_description("Specialist agent for math questions")
            .with_instructions("You explain math step by step and include worked examples."),
    );
    let triage = Agent::<()>::new("Triage Agent")
        .with_instructions(prompt_with_handoff_instructions(
            "Route each homework question to the right specialist.",
        ))
        .with_handoffs([handoff(history_tutor), handoff(math_tutor)]);

    let result = Runner::run(&triage, "Who was the first president of the United States?").await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    println!("Answered by: {}", result.last_agent.name);
    flush_traces_async().await;
    Ok(())
}
