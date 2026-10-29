//! Function tools with typed arguments.
//! Run: cargo run --example tools
use openai_agents_sdk::{Agent, FunctionTool, Runner, flush_traces_async};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// The city to get the weather for.
    city: String,
}

#[derive(Serialize)]
struct Weather {
    city: String,
    temperature_range: String,
    conditions: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let get_weather = FunctionTool::new(
        "get_weather",
        "Get the current weather information for a specified city.",
        |_ctx, args: WeatherArgs| async move {
            let w = Weather {
                city: args.city,
                temperature_range: "14-20C".into(),
                conditions: "Sunny with wind.".into(),
            };
            Ok(serde_json::to_value(w)?)
        },
    );
    let agent = Agent::<()>::new("Hello world")
        .with_instructions("You are a helpful agent.")
        .with_tool(get_weather);
    let result = Runner::run(&agent, "What's the weather in Tokyo?").await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    flush_traces_async().await;
    Ok(())
}
