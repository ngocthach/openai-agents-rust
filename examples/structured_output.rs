//! Structured output with a strict JSON schema derived from a Rust type.
//! Run: cargo run --example structured_output
use openai_agents_sdk::{Agent, Runner, flush_traces_async};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct CalendarEvent {
    name: String,
    date: String,
    participants: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::<()>::new("Calendar extractor")
        .with_instructions("Extract calendar events from text")
        .with_output_type::<CalendarEvent>();
    let result = Runner::run(
        &agent,
        "Alice and Bob are going to a science fair on Friday.",
    )
    .await?;
    let event: CalendarEvent = result.final_output_as()?;
    println!(
        "{} on {} with {}",
        event.name,
        event.date,
        event.participants.join(", ")
    );
    flush_traces_async().await;
    Ok(())
}
