//! Send traces to a custom processor (here: stdout) instead of OpenAI.
//! Run: cargo run --example custom_trace_processor
use std::sync::Arc;

use openai_agents_sdk::tracing::{
    BatchTraceProcessor, ConsoleSpanExporter, TracingProcessor, set_trace_processors,
};
use openai_agents_sdk::{Agent, Runner, flush_traces};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = Arc::new(BatchTraceProcessor::new(Arc::new(ConsoleSpanExporter)));
    set_trace_processors(vec![processor as Arc<dyn TracingProcessor>]);
    let agent = Agent::<()>::new("Assistant").with_instructions("Be brief.");
    let result = Runner::run(&agent, "Say hi").await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    tokio::task::spawn_blocking(flush_traces).await?;
    Ok(())
}
