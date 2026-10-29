//! Stream higher-level run item events (tool calls, outputs, messages).
//! Run: cargo run --example stream_items
use futures::StreamExt;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::{Agent, FunctionTool, ItemHelpers, Runner, flush_traces_async};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let how_many_jokes = FunctionTool::new_no_args(
        "how_many_jokes",
        "Return a number of jokes.",
        |_ctx| async { Ok(3) },
    );
    let agent = Agent::<()>::new("Joker")
        .with_instructions("First call the `how_many_jokes` tool, then tell that many jokes.")
        .with_tool(how_many_jokes);
    let mut result = Runner::run_streamed(&agent, "Hello");
    println!("=== Run starting ===");
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            match event? {
                StreamEvent::RawResponse(_) => {}
                StreamEvent::AgentUpdated { new_agent } => {
                    println!("Agent updated: {}", new_agent.name)
                }
                StreamEvent::RunItem { name, item } => match name {
                    RunItemStreamEventName::ToolCalled => println!("-- Tool was called"),
                    RunItemStreamEventName::ToolOutput => {
                        println!(
                            "-- Tool output: {}",
                            item.tool_output().cloned().unwrap_or_default()
                        )
                    }
                    RunItemStreamEventName::MessageOutputCreated => {
                        println!(
                            "-- Message output:\n {}",
                            ItemHelpers::text_message_output(&item)
                        )
                    }
                    _ => {}
                },
            }
        }
    }
    println!("=== Run complete ===");
    flush_traces_async().await;
    Ok(())
}
