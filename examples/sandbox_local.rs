//! A sandbox agent working in a Unix-local workspace.
//!
//! The manifest stages a tiny repository, the default capabilities give the model a shell and
//! `apply_patch`, and `SandboxRunner` creates the workspace, runs the agent and tears the
//! workspace down again.
//!
//! ```bash
//! OPENAI_API_KEY=... cargo run --features sandbox --example sandbox_local
//! ```
//!
//! Commands run on the host with the permissions of this process: the Unix-local backend adds no
//! OS-level confinement. Use it for trusted local development only.

#[cfg(feature = "sandbox")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use openai_agents_sdk::sandbox::entries::Entry;
    use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
    use openai_agents_sdk::sandbox::{Manifest, SandboxAgent, SandboxRunConfig, SandboxRunner};

    env_logger::init();

    let manifest = Manifest::new().with_entry(
        "repo",
        Entry::dir()
            .with_description("staged repository")
            .with_child(
                "task.md",
                Entry::text_file(
                    "# Task\n\n\
                     `greet.sh` prints the wrong greeting. Fix it so it prints `Hello, world!`, \
                     then run it to verify.\n",
                ),
            )
            .with_child(
                "greet.sh",
                Entry::text_file("#!/bin/sh\necho \"Goodbye, world!\"\n"),
            ),
    );

    let agent = SandboxAgent::<()>::new("Sandbox engineer")
        .with_model("gpt-5.5")
        .with_instructions(
            "Read `repo/task.md` before editing files. Stay grounded in the repository, preserve \
             existing behavior, and mention the exact verification command you ran. Paths passed \
             to apply_patch are relative to the sandbox workspace root.",
        )
        .with_default_manifest(manifest);

    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));
    let result = SandboxRunner::run(
        &agent,
        "Open `repo/task.md`, fix the issue, verify it, and summarize the change.",
        &config,
    )
    .await?;

    println!("{}", result.final_output_text().unwrap_or_default());
    Ok(())
}

#[cfg(not(feature = "sandbox"))]
fn main() {
    eprintln!(
        "This example requires the `sandbox` feature: cargo run --features sandbox --example sandbox_local"
    );
}
