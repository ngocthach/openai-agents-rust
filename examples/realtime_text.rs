//! A text-only realtime session: `OPENAI_API_KEY=... cargo run --features realtime --example
//! realtime_text`.
//!
//! The agent answers in text instead of audio, which keeps the example readable in a terminal.

#[cfg(not(feature = "realtime"))]
fn main() {
    eprintln!(
        "This example needs the `realtime` feature: cargo run --features realtime --example realtime_text"
    );
}

#[cfg(feature = "realtime")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use openai_agents_sdk::realtime::{
        RealtimeAgent, RealtimeModelEvent, RealtimeRunner, RealtimeSessionEvent,
        RealtimeSessionModelSettings,
    };
    use openai_agents_sdk::tool::FunctionTool;
    use schemars::JsonSchema;
    use serde::Deserialize;

    env_logger::init();

    #[derive(Deserialize, JsonSchema)]
    struct CityArgs {
        /// The city to look up.
        city: String,
    }

    let weather = FunctionTool::new(
        "get_weather",
        "Get the current weather for a city",
        |_ctx, args: CityArgs| async move { Ok(format!("The weather in {} is sunny", args.city)) },
    );

    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_instructions("You are a concise assistant. Answer in one short sentence.")
        .with_tool(weather)
        // Text-only output, and no server-side turn detection: this example never sends audio.
        .with_model_settings(
            RealtimeSessionModelSettings::new()
                .with_output_modalities(["text"])
                .without_turn_detection(),
        )
        .into_arc();

    let session = RealtimeRunner::new(agent).run(()).await?;
    session
        .send_message("What is the weather in Oakland?")
        .await?;

    // A tool call and the answer that follows it are two separate model responses, so a turn that
    // ran a tool is not the end of the exchange.
    let mut turn_called_a_tool = false;
    while let Some(event) = session.recv().await {
        match event {
            RealtimeSessionEvent::ToolStart { tool, .. } => {
                println!("\n[tool] calling {}", tool.name);
            }
            RealtimeSessionEvent::ToolEnd { tool, output, .. } => {
                println!("[tool] {} -> {}", tool.name, output.to_model_string());
                turn_called_a_tool = true;
            }
            // Text output arrives as deltas; the history only carries the finished item.
            RealtimeSessionEvent::RawModelEvent {
                data: RealtimeModelEvent::OutputTextDelta { delta, .. },
                ..
            } => {
                print!("{delta}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            RealtimeSessionEvent::Error { error, .. } => {
                eprintln!("\n[error] {error}");
            }
            RealtimeSessionEvent::AgentEnd { .. } => {
                if turn_called_a_tool {
                    turn_called_a_tool = false;
                    continue;
                }
                println!();
                break;
            }
            _ => {}
        }
    }

    session.close().await?;
    Ok(())
}
