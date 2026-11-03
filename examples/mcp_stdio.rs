//! Use tools from a local MCP server (feature `mcp`). Requires Node.js for `npx`.
//! Run: cargo run --example mcp_stdio --features mcp
use std::sync::Arc;

use openai_agents_sdk::mcp::{MCPServer, MCPServerStdio, MCPServerStdioParams};
use openai_agents_sdk::{Agent, Runner, flush_traces_async};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::current_dir()?.display().to_string();
    let server: Arc<dyn MCPServer> = Arc::new(
        MCPServerStdio::new(
            "Filesystem Server via npx",
            MCPServerStdioParams {
                command: "npx".into(),
                args: vec![
                    "-y".into(),
                    "@modelcontextprotocol/server-filesystem".into(),
                    dir,
                ],
                ..Default::default()
            },
        )
        .with_client_session_timeout(Some(std::time::Duration::from_secs(30))),
    );
    server.connect().await?;
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Use the tools to read the filesystem and answer questions.")
        .with_mcp_server(server.clone());
    let result = Runner::run(&agent, "List the files in the current directory.").await?;
    println!("{}", result.final_output_text().unwrap_or_default());
    server.cleanup().await?;
    flush_traces_async().await;
    Ok(())
}
