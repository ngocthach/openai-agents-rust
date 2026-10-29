//! Run an agent over the Responses API WebSocket transport, reusing one connection.
//! Run: cargo run --example responses_websocket --features websocket
use openai_agents_sdk::models::openai_responses_ws::ResponsesWebSocketSession;
use openai_agents_sdk::{Agent, RunOptions, Runner, flush_traces_async};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ws = ResponsesWebSocketSession::from_env();
    let agent = Agent::<()>::new("Assistant").with_instructions("Answer in one sentence.");
    for question in ["What is Rust?", "And what is Tokio?"] {
        let opts = RunOptions::new(()).run_config(ws.run_config());
        let result = Runner::run_with_options(&agent, question, opts).await?;
        println!(
            "{question} -> {}",
            result.final_output_text().unwrap_or_default()
        );
    }
    ws.close().await;
    flush_traces_async().await;
    Ok(())
}
