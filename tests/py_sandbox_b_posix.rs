//! Port of `tests/sandbox/test_posix_tool_paths.py::test_memory_layout_probes_existing_files_with_posix_paths`.
//!
//! The existence probe must name the sandbox's POSIX paths, so an existing memory file is not
//! mistaken for a missing one and overwritten with an empty file on every layout check.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::memory::{MemoryLayoutConfig, SandboxMemoryStorage};
use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};
use openai_agents_sdk::sandbox::types::ExecResult;

#[tokio::test]
async fn memory_layout_probes_existing_files_with_posix_paths() {
    let ok = ExecResult {
        stdout: Vec::new(),
        stderr: Vec::new(),
        exit_code: 0,
    };
    let mut steps: Vec<SandboxStep> = (0..5)
        .map(|_| SandboxStep::new(SandboxMethod::Mkdir))
        .collect();
    steps.push(SandboxStep::new(SandboxMethod::Exec).returning_exec(ok.clone()));
    steps.push(SandboxStep::new(SandboxMethod::Exec).returning_exec(ok));
    let session =
        scripted_sandbox_session(steps).with_manifest(Manifest::new().with_root("/workspace"));
    let storage = SandboxMemoryStorage::new(MemoryLayoutConfig::new());

    storage.ensure_layout(&session).await.expect("layout");

    let probes: Vec<String> = session
        .calls()
        .into_iter()
        .filter(|call| call.method == SandboxMethod::Exec)
        .map(|call| call.arguments[2].clone())
        .collect();
    assert_eq!(
        probes,
        vec![
            "/workspace/memories/MEMORY.md".to_string(),
            "/workspace/memories/memory_summary.md".to_string(),
        ]
    );
    session.assert_complete();
}
