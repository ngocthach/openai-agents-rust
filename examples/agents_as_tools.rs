//! An orchestrator calling specialist agents as tools.
//! Run: cargo run --example agents_as_tools
use openai_agents_sdk::{Agent, Runner, flush_traces_async, trace};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spanish = Agent::<()>::new("spanish_agent")
        .with_instructions("You translate the user's message to Spanish");
    let french = Agent::<()>::new("french_agent")
        .with_instructions("You translate the user's message to French");
    let orchestrator = Agent::<()>::new("orchestrator_agent")
        .with_instructions(
            "You are a translation agent. You use the tools given to you to translate. \
             If asked for multiple translations, you call the relevant tools.",
        )
        .with_tool(spanish.as_tool(
            "translate_to_spanish",
            "Translate the user's message to Spanish",
        ))
        .with_tool(french.as_tool(
            "translate_to_french",
            "Translate the user's message to French",
        ));

    let result = trace("Orchestrator evaluator")
        .run(Runner::run(
            &orchestrator,
            "Say 'Hello, how are you?' in Spanish and French.",
        ))
        .await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    flush_traces_async().await;
    Ok(())
}
