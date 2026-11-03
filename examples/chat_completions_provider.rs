//! Use any OpenAI-compatible Chat Completions endpoint.
//! Run: EXAMPLE_BASE_URL=... EXAMPLE_API_KEY=... EXAMPLE_MODEL_NAME=... cargo run --example chat_completions_provider
use std::sync::Arc;

use openai_agents_sdk::{
    Agent, OpenAIChatCompletionsModel, OpenAIClient, Runner, set_tracing_disabled,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("EXAMPLE_BASE_URL")?;
    let api_key = std::env::var("EXAMPLE_API_KEY")?;
    let model_name = std::env::var("EXAMPLE_MODEL_NAME")?;
    // Traces go to OpenAI; disable them when you do not have an OpenAI key.
    set_tracing_disabled(true);
    let client = OpenAIClient::new(api_key).with_base_url(base_url);
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("You only respond in haikus.")
        .with_model_instance(Arc::new(OpenAIChatCompletionsModel::new(
            model_name, client,
        )));
    let result = Runner::run(&agent, "What's the weather in Tokyo?").await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    Ok(())
}
