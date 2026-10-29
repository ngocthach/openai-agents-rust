//! Conversation memory persisted in SQLite (feature `sqlite`).
//! Run: cargo run --example sessions_sqlite --features sqlite
use std::sync::Arc;

use openai_agents_sdk::memory::SqliteSession;
use openai_agents_sdk::{Agent, RunOptions, Runner, Session, flush_traces_async};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::<()>::new("Assistant").with_instructions("Reply very concisely.");
    let session: Arc<dyn Session> = Arc::new(SqliteSession::open(
        "conversation_123",
        "conversation_history.db",
    )?);
    for question in [
        "What city is the Golden Gate Bridge in?",
        "What state is it in?",
        "What's the population of that state?",
    ] {
        println!("User: {question}");
        let result = Runner::run_with_options(
            &agent,
            question,
            RunOptions::new(()).session(session.clone()),
        )
        .await?;
        println!(
            "Assistant: {}\n",
            result.final_output_text().unwrap_or_default()
        );
    }
    flush_traces_async().await;
    Ok(())
}
