//! Pause for approval, persist the run state, approve, and resume.
//! Run: cargo run --example human_in_the_loop
use openai_agents_sdk::{Agent, FunctionTool, RunState, Runner, flush_traces_async};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct CityArgs {
    city: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let get_temperature =
        FunctionTool::new(
            "get_temperature",
            "Get the temperature",
            |_ctx, a: CityArgs| async move {
                Ok(format!("The temperature in {} is 20° Celsius", a.city))
            },
        )
        .with_needs_approval_fn(|_ctx, args, _call_id| async move {
            args["city"]
                .as_str()
                .unwrap_or_default()
                .contains("Oakland")
        });
    let agent = Agent::<()>::new("Weather assistant")
        .with_instructions("Answer weather questions with the provided tools.")
        .with_tool(get_temperature);

    let mut result = Runner::run(&agent, "What is the temperature in Oakland?").await?;
    while !result.interruptions.is_empty() {
        // Persist the paused state (e.g. in a database), then load it later.
        let stored = result.to_state().to_json_string()?;
        let mut state = RunState::from_json_str(&stored)?;
        for item in state.get_interruptions() {
            println!("Approving {} with {}", item.name(), item.arguments());
            state.approve(&item, false);
        }
        result = Runner::run(&agent, state).await?;
    }
    println!("{}", result.final_output_text().unwrap_or_default());
    flush_traces_async().await;
    Ok(())
}
