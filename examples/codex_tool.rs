//! Give an agent the experimental Codex tool.
//!
//! Requires the `codex` CLI on `PATH` (or `CODEX_PATH`) and `OPENAI_API_KEY`.
//!
//! ```sh
//! cargo run --example codex_tool
//! ```

use openai_agents_sdk::extensions::codex::{
    CodexToolOptions, SandboxMode, ThreadOptions, codex_tool,
};
use openai_agents_sdk::{Agent, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let workspace = std::env::current_dir()?.to_string_lossy().into_owned();
    let tool = codex_tool(
        CodexToolOptions::<()>::new()
            // One Codex thread for the whole run, so follow-up calls keep the context.
            .with_persist_session(true)
            .with_working_directory(workspace)
            .with_sandbox_mode(SandboxMode::ReadOnly)
            .with_default_thread_options(ThreadOptions::new().with_skip_git_repo_check(true))
            .with_on_stream(|event| async move {
                println!("[codex] {}", event.event.event_type());
            }),
    )?;

    let agent = Agent::<()>::new("Dev assistant")
        .with_instructions(
            "You delegate code questions about this repository to the codex tool, then summarize \
             what it found.",
        )
        .with_tool(tool);

    let result = Runner::run(
        &agent,
        "What does this crate expose at its root? Keep it short.",
    )
    .await?;
    println!("\n{}", result.final_output_text().unwrap_or_default());

    openai_agents_sdk::tracing::flush_traces();
    Ok(())
}
