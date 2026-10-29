//! The smallest agent. Requires `OPENAI_API_KEY`.
//! Run: cargo run --example hello_world
use openai_agents_sdk::{Agent, Runner, flush_traces_async};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::<()>::new("Assistant").with_instructions("You only respond in haikus.");
    let result = Runner::run(&agent, "Tell me about recursion in programming.").await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    flush_traces_async().await;
    Ok(())
}
