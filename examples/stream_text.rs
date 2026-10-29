//! Stream text deltas as the model produces them.
//! Run: cargo run --example stream_text
use futures::StreamExt;
use openai_agents_sdk::{Agent, Runner, flush_traces_async};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::<()>::new("Joker").with_instructions("You are a helpful assistant.");
    let mut result = Runner::run_streamed(&agent, "Please tell me 5 jokes.");
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            if let Some(delta) = event?.text_delta() {
                print!("{delta}");
                std::io::stdout().flush()?;
            }
        }
    }
    println!();
    flush_traces_async().await;
    Ok(())
}
