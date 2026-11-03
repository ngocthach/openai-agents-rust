//! A coding agent with a local shell (every command needs approval) and file editing through
//! V4A patches, confined to a temporary workspace.
//! Run: cargo run --example shell_and_apply_patch
use openai_agents_sdk::editor::WorkspaceEditor;
use openai_agents_sdk::tool::{ApplyPatchTool, ShellTool};
use openai_agents_sdk::{Agent, Runner, flush_traces_async};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace = std::env::temp_dir().join("agents-rust-workspace");
    std::fs::create_dir_all(&workspace)?;

    let agent = Agent::<()>::new("Coder")
        .with_instructions(
            "You edit files with apply_patch and inspect the workspace with the shell tool.",
        )
        .with_tool(
            ShellTool::<()>::with_system_shell(Some(workspace.clone())).with_needs_approval(true),
        )
        .with_tool(ApplyPatchTool::<()>::new(WorkspaceEditor::new(&workspace)));

    let mut result = Runner::run(
        &agent,
        "Create hello.txt containing a haiku, then show it with cat.",
    )
    .await?;
    while !result.interruptions.is_empty() {
        let mut state = result.to_state();
        for item in state.get_interruptions() {
            println!("Approving {}: {}", item.name(), item.arguments());
            state.approve(&item, false);
        }
        result = Runner::run(&agent, state).await?;
    }
    println!("{}", result.final_output_text().unwrap_or_default());
    flush_traces_async().await;
    Ok(())
}
