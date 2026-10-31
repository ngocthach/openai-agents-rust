//! A simple interactive loop for trying agents in a terminal. Mirrors `agents.repl`.

use std::io::Write;
use std::sync::Arc;

use futures::StreamExt;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};

use crate::agent::Agent;
use crate::error::Result;
use crate::items::{InputItem, RunInput};
use crate::run::{DEFAULT_MAX_TURNS, RunOptions, Runner};
use crate::run_context::RunContext;
use crate::stream_events::{RunItemStreamEventName, StreamEvent};

/// Run a read-eval-print loop: read a line from stdin, run the agent, print the answer.
/// Type `exit` or `quit` (or send EOF) to stop. Conversation history is kept between turns.
pub async fn run_demo_loop<C: Default + Send + Sync + 'static>(
    agent: Arc<Agent<C>>,
    stream: bool,
) -> Result<()> {
    run_demo_loop_with(agent, stream, C::default(), Some(DEFAULT_MAX_TURNS)).await
}

/// [`run_demo_loop`] with an app context and a turn limit. `max_turns` of `None` disables the
/// limit. Mirrors `agents.repl.run_demo_loop(context=..., max_turns=...)`.
pub async fn run_demo_loop_with<C: Send + Sync + 'static>(
    agent: Arc<Agent<C>>,
    stream: bool,
    context: C,
    max_turns: Option<usize>,
) -> Result<()> {
    run_demo_loop_io(
        agent,
        stream,
        context,
        max_turns,
        BufReader::new(tokio::io::stdin()),
        std::io::stdout(),
    )
    .await
}

/// [`run_demo_loop_with`] reading lines from `input` and writing prompts and answers to
/// `output` instead of stdin and stdout.
pub async fn run_demo_loop_io<C, R, W>(
    agent: Arc<Agent<C>>,
    stream: bool,
    context: C,
    max_turns: Option<usize>,
    input: R,
    mut output: W,
) -> Result<()>
where
    C: Send + Sync + 'static,
    R: AsyncBufRead + Unpin,
    W: Write,
{
    // One run context for the whole loop, so usage and approvals accumulate across turns.
    let run_context = RunContext::new(context);
    let options = || RunOptions::with_run_context(run_context.clone()).max_turns(max_turns);
    let mut current = agent;
    let mut history: Vec<InputItem> = Vec::new();
    let mut lines = input.lines();
    loop {
        write!(output, " > ").ok();
        output.flush().ok();
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
            let mut result = Runner::run_streamed_with_options(current.clone(), input, options());
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(ev) = events.next().await {
                    match ev? {
                        ev @ StreamEvent::RawResponse(_) => {
                            if let Some(d) = ev.text_delta() {
                                write!(output, "{d}").ok();
                                output.flush().ok();
                            }
                        }
                        StreamEvent::RunItem { name, item } => match name {
                            RunItemStreamEventName::ToolCalled => {
                                writeln!(output, "\n[tool called]").ok();
                            }
                            RunItemStreamEventName::ToolOutput => {
                                // Python prints the output itself, not its JSON encoding.
                                let text = match item.tool_output() {
                                    Some(serde_json::Value::String(s)) => s.clone(),
                                    Some(other) => other.to_string(),
                                    None => String::new(),
                                };
                                writeln!(output, "\n[tool output: {text}]").ok();
                            }
                            _ => {}
                        },
                        StreamEvent::AgentUpdated { new_agent } => {
                            writeln!(output, "\n[Agent updated: {}]", new_agent.name).ok();
                        }
                    }
                }
            }
            writeln!(output).ok();
            if let Some(r) = result.into_result() {
                history = r.to_input_list();
                current = r.last_agent.clone();
            }
        } else {
            let r = Runner::run_with_options(current.clone(), input, options()).await?;
            writeln!(output, "{}", r.final_output_text().unwrap_or_default()).ok();
            history = r.to_input_list();
            current = r.last_agent.clone();
        }
    }
    Ok(())
}
