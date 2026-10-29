//! A simple interactive loop for trying agents in a terminal. Mirrors `agents.repl`.

use std::io::Write;
use std::sync::Arc;

use futures::StreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::agent::Agent;
use crate::error::Result;
use crate::items::{InputItem, RunInput};
use crate::run::{RunOptions, Runner};
use crate::stream_events::{RunItemStreamEventName, StreamEvent};

/// Run a read-eval-print loop: read a line from stdin, run the agent, print the answer.
/// Type `exit` or `quit` (or send EOF) to stop. Conversation history is kept between turns.
pub async fn run_demo_loop<C: Default + Send + Sync + 'static>(
    agent: Arc<Agent<C>>,
    stream: bool,
) -> Result<()> {
    let mut current = agent;
    let mut history: Vec<InputItem> = Vec::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        print!(" > ");
        std::io::stdout().flush().ok();
        let Some(line) = lines.next_line().await? else {
            break;
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if matches!(line.to_lowercase().as_str(), "exit" | "quit") {
            break;
        }
        history.push(InputItem::user(line));
        let input = RunInput::Items(history.clone());
        if stream {
            let mut result =
                Runner::run_streamed_with_options(current.clone(), input, RunOptions::default());
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(ev) = events.next().await {
                    match ev? {
                        ev @ StreamEvent::RawResponse(_) => {
                            if let Some(d) = ev.text_delta() {
                                print!("{d}");
                                std::io::stdout().flush().ok();
                            }
                        }
                        StreamEvent::RunItem { name, item } => match name {
                            RunItemStreamEventName::ToolCalled => println!("\n[tool called]"),
                            RunItemStreamEventName::ToolOutput => {
                                println!(
                                    "\n[tool output: {}]",
                                    item.tool_output().cloned().unwrap_or_default()
                                )
                            }
                            _ => {}
                        },
                        StreamEvent::AgentUpdated { new_agent } => {
                            println!("\n[Agent updated: {}]", new_agent.name)
                        }
                    }
                }
            }
            println!();
            if let Some(r) = result.into_result() {
                history = r.to_input_list();
                current = r.last_agent.clone();
            }
        } else {
            let r = Runner::run_with_options(current.clone(), input, RunOptions::default()).await?;
            println!("{}", r.final_output_text().unwrap_or_default());
            history = r.to_input_list();
            current = r.last_agent.clone();
        }
    }
    Ok(())
}
