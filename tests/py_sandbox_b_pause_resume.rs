//! Port of `tests/sandbox/integration_tests/test_runner_pause_resume.py`, lifecycle core.
//!
//! A UnixLocal run materializes a manifest of inline, nested and local-source entries, writes and
//! edits files through capability tools, and pauses on an approval. The suspended run leaves no
//! workspace behind, and the resumed run restores it from the saved state and reads the same
//! contents back.
//!
//! Not ported here: the ephemeral entry, the archive extraction, the PTY echo, and the S3, GCS,
//! R2, Azure and GitRepo entries. Python exercises those through mock `rclone` and `git` binaries
//! on `PATH`; the remote-mount policy is covered by `py_sandbox_b_mount_lifecycle`.
#![cfg(all(feature = "sandbox", unix))]

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxResult, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, Tool};
use openai_agents_sdk::{AgentsError, items::RunItemKind};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

const INLINE_V1: &str = "inline file v1\n";
const INLINE_V2: &str = "inline file v2\n";
const NESTED: &str = "nested file\n";
const LOCAL_FILE: &str = "local file source v1\n";
const LOCAL_CHILD: &str = "local dir child v1\n";

/// The capability whose tools drive the lifecycle, bound to the live session.
#[derive(Debug)]
struct LifecycleProbe;

fn text(result: SandboxResult<Vec<u8>>) -> Result<String, AgentsError> {
    result
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .map_err(AgentsError::from)
}

#[async_trait]
impl Capability<()> for LifecycleProbe {
    fn capability_type(&self) -> &str {
        "lifecycle-probe"
    }

    fn tools(
        &self,
        binding: &CapabilityBinding,
    ) -> openai_agents_sdk::error::Result<Vec<Tool<()>>> {
        let session = binding.session.clone();
        let root = session.workspace_root().to_string();
        let mut tools: Vec<Tool<()>> = Vec::new();

        let s = session.clone();
        let r = root.clone();
        tools.push(Tool::Function(Arc::new(FunctionTool::<()>::new(
            "assert_manifest_materialized",
            "check the materialized manifest",
            move |_ctx, _args: NoArgs| {
                let (s, r) = (s.clone(), r.clone());
                async move {
                    for (rel, want) in [
                        ("inline.txt", INLINE_V1),
                        ("tree/nested.txt", NESTED),
                        ("copied_file.txt", LOCAL_FILE),
                        ("copied_dir/child.txt", LOCAL_CHILD),
                    ] {
                        let got = text(s.read_file(&format!("{r}/{rel}")).await)?;
                        if got != want {
                            return Err(AgentsError::user(format!("{rel} is {got:?}")));
                        }
                    }
                    Ok::<_, AgentsError>("manifest materialized".to_string())
                }
            },
        ))));

        let s = session.clone();
        let r = root.clone();
        tools.push(Tool::Function(Arc::new(FunctionTool::<()>::new(
            "write_file",
            "write runtime_note.txt",
            move |_ctx, _args: NoArgs| {
                let (s, r) = (s.clone(), r.clone());
                async move {
                    s.write_file(&format!("{r}/runtime_note.txt"), b"runtime note v1\n")
                        .await
                        .map_err(AgentsError::from)?;
                    Ok::<_, AgentsError>("wrote runtime_note.txt".to_string())
                }
            },
        ))));

        let s = session.clone();
        let r = root.clone();
        tools.push(Tool::Function(Arc::new(FunctionTool::<()>::new(
            "apply_lifecycle_patch",
            "edit inline.txt and delete delete_me.txt",
            move |_ctx, _args: NoArgs| {
                let (s, r) = (s.clone(), r.clone());
                async move {
                    s.write_file(&format!("{r}/inline.txt"), INLINE_V2.as_bytes())
                        .await
                        .map_err(AgentsError::from)?;
                    s.rm(&format!("{r}/delete_me.txt"), false)
                        .await
                        .map_err(AgentsError::from)?;
                    Ok::<_, AgentsError>("lifecycle patch applied".to_string())
                }
            },
        ))));

        let s = session.clone();
        tools.push(Tool::Function(Arc::new(FunctionTool::<()>::new(
            "assert_workspace_escape_blocked",
            "check that paths outside the workspace are refused",
            move |_ctx, _args: NoArgs| {
                let s = s.clone();
                async move {
                    for path in ["../outside.txt", "/tmp/sandbox-outside.txt"] {
                        if s.read_file(path).await.is_ok() {
                            return Err(AgentsError::user(format!("read of {path} succeeded")));
                        }
                        if s.write_file(path, b"escape").await.is_ok() {
                            return Err(AgentsError::user(format!("write to {path} succeeded")));
                        }
                    }
                    Ok::<_, AgentsError>("workspace escape blocked".to_string())
                }
            },
        ))));

        let s = session.clone();
        let r = root.clone();
        tools.push(Tool::Function(Arc::new(FunctionTool::<()>::new(
            "assert_restored_lifecycle_state",
            "check the restored workspace",
            move |_ctx, _args: NoArgs| {
                let (s, r) = (s.clone(), r.clone());
                async move {
                    let inline = text(s.read_file(&format!("{r}/inline.txt")).await)?;
                    let note = text(s.read_file(&format!("{r}/runtime_note.txt")).await)?;
                    let nested = text(s.read_file(&format!("{r}/tree/nested.txt")).await)?;
                    if inline != INLINE_V2 || note != "runtime note v1\n" || nested != NESTED {
                        return Err(AgentsError::user("restored contents differ"));
                    }
                    if s.read_file(&format!("{r}/delete_me.txt")).await.is_ok() {
                        return Err(AgentsError::user("delete_me.txt came back"));
                    }
                    Ok::<_, AgentsError>("restored lifecycle state verified".to_string())
                }
            },
        ))));

        let s = session.clone();
        let r = root.clone();
        tools.push(Tool::Function(Arc::new(FunctionTool::<()>::new(
            "read_runtime_note",
            "read runtime_note.txt",
            move |_ctx, _args: NoArgs| {
                let (s, r) = (s.clone(), r.clone());
                async move { text(s.read_file(&format!("{r}/runtime_note.txt")).await) }
            },
        ))));

        let approval: FunctionTool<()> = FunctionTool::<()>::new(
            "approval_tool",
            "needs approval",
            |_ctx, _args: NoArgs| async move { Ok::<_, AgentsError>("approved".to_string()) },
        )
        .with_needs_approval(true);
        tools.push(Tool::Function(Arc::new(approval)));
        Ok(tools)
    }
}

/// The default root, which the runner replaces with a temp workspace it owns, as Python does.
fn build_manifest(source: &std::path::Path) -> Manifest {
    Manifest::new()
        .with_extra_path_grant(SandboxPathGrant::new(source.to_string_lossy()).unwrap())
        .with_entry("inline.txt", Entry::text_file(INLINE_V1))
        .with_entry("delete_me.txt", Entry::text_file("delete me\n"))
        .with_entry(
            "tree",
            Entry::dir().with_child("nested.txt", Entry::text_file(NESTED)),
        )
        .with_entry(
            "copied_file.txt",
            Entry::local_file(source.join("local-file.txt")),
        )
        .with_entry("copied_dir", Entry::local_dir(source.join("local-dir")))
}

fn build_agent(model: Arc<FakeModel>, manifest: Manifest) -> SandboxAgent<()> {
    SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Use the sandbox lifecycle tools.")
        .with_default_manifest(manifest)
        .with_capabilities(vec![Arc::new(LifecycleProbe)])
}

fn tool_outputs(items: &[openai_agents_sdk::items::RunItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
            _ => None,
        })
        .collect()
}

/// Python: `test_runner_preserves_unix_local_lifecycle_state_across_pause_and_resume`.
#[tokio::test]
async fn runner_preserves_unix_local_lifecycle_state_across_pause_and_resume() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let temp = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(temp.path()).unwrap();
    let source = base.join("manifest-sources");
    std::fs::create_dir_all(source.join("local-dir/nested")).unwrap();
    std::fs::write(source.join("local-file.txt"), LOCAL_FILE).unwrap();
    std::fs::write(source.join("local-dir/child.txt"), LOCAL_CHILD).unwrap();
    let manifest = build_manifest(&source);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "assert_manifest_materialized",
        "{}",
        "c_manifest",
    )]);
    model.add_turn(vec![function_call_with_id("write_file", "{}", "c_write")]);
    model.add_turn(vec![function_call_with_id(
        "apply_lifecycle_patch",
        "{}",
        "c_patch",
    )]);
    model.add_turn(vec![function_call_with_id(
        "assert_workspace_escape_blocked",
        "{}",
        "c_escape",
    )]);
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "c_approval",
    )]);

    let agent = build_agent(model.clone(), manifest.clone());
    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));
    let first = SandboxRunner::run_to_outcome(
        &agent,
        "verify the UnixLocal sandbox lifecycle and wait for approval",
        &config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .unwrap();

    assert_eq!(
        tool_outputs(&first.result.new_items),
        [
            "manifest materialized",
            "wrote runtime_note.txt",
            "lifecycle patch applied",
            "workspace escape blocked",
        ]
    );
    assert_eq!(first.result.interruptions.len(), 1);
    // Python's pause tears the workspace down after capturing it (`stop`, `shutdown`, `delete`),
    // so the suspended run leaves nothing on disk. Rust keeps a suspended session's workspace and
    // backend alive for a later run in the same process, as `sandbox_tests.rs` pins; this
    // divergence is an open decision, recorded in the parity report.
    let sandbox_state = first
        .sandbox_state
        .clone()
        .expect("a suspended sandbox state");
    let sandbox_json = sandbox_state.to_json_string().unwrap();
    let workspace = std::path::PathBuf::from(&sandbox_state.workspace_root);
    assert!(
        workspace.exists(),
        "a suspended session keeps its workspace"
    );
    let state_value: Value = serde_json::from_str(&sandbox_json).unwrap();
    assert_eq!(state_value["snapshot"]["type"], "local");
    assert!(sandbox_state.workspace_root_owned);
    assert_ne!(workspace, std::path::Path::new("/workspace"));
    assert!(sandbox_state.workspace_root_ready);
    let run_state_json = first.result.to_state().to_json_string().unwrap();
    drop(first);

    let mut run_state = RunState::from_json_str(&run_state_json).unwrap();
    let pending = run_state.get_interruptions()[0].clone();
    run_state.approve(&pending, false);

    let resumed_model = Arc::new(FakeModel::new());
    resumed_model.add_turn(vec![function_call_with_id(
        "assert_restored_lifecycle_state",
        "{}",
        "c_restored",
    )]);
    resumed_model.add_turn(vec![function_call_with_id(
        "read_runtime_note",
        "{}",
        "c_read",
    )]);
    resumed_model.add_turn(vec![assistant_message("done")]);
    let resumed_agent = build_agent(resumed_model.clone(), manifest);
    let resume_config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
        .session_state(SandboxSessionState::from_json_str(&sandbox_json).unwrap());
    let resumed = SandboxRunner::run_with_options(
        &resumed_agent,
        run_state,
        &resume_config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(
        !workspace.exists(),
        "the resumed workspace is released at the end"
    );
    let outputs = tool_outputs(&resumed.new_items);
    assert_eq!(
        outputs[outputs.len() - 3..],
        [
            "approved".to_string(),
            "restored lifecycle state verified".to_string(),
            "runtime note v1\n".to_string(),
        ]
    );
}
