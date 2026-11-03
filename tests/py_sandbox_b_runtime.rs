//! Ports of the rows of `tests/sandbox/test_runtime_agent_preparation.py` that the sibling
//! `py_sandbox_a_runtime_*` files do not already cover.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::sandbox::SandboxAgent;
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, Compaction, Filesystem, Memory, Shell,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::memory::MemoryReadConfig;
use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};
use openai_agents_sdk::sandbox::workspace_paths::SandboxWorkspaceScope;
use openai_agents_sdk::tool::Tool;
use serde_json::{Map, Value, json};

/// Python: `_Capability`. Records what the preparation hands it.
#[derive(Debug)]
struct Recording {
    kind: &'static str,
    fragment: Option<&'static str>,
    manifests: Mutex<Vec<Manifest>>,
    sampling_calls: Mutex<Vec<Map<String, Value>>>,
}

impl Recording {
    fn new(kind: &'static str, fragment: Option<&'static str>) -> Arc<Self> {
        Arc::new(Recording {
            kind,
            fragment,
            manifests: Mutex::new(Vec::new()),
            sampling_calls: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl Capability<()> for Recording {
    fn capability_type(&self) -> &str {
        self.kind
    }

    fn tools(&self, _binding: &CapabilityBinding) -> Result<Vec<Tool<()>>> {
        Ok(Vec::new())
    }

    fn sampling_params(&self, sampling_params: &Map<String, Value>) -> Map<String, Value> {
        self.sampling_calls
            .lock()
            .unwrap()
            .push(sampling_params.clone());
        Map::new()
    }

    async fn instructions(&self, manifest: &Manifest) -> Result<Option<String>> {
        self.manifests.lock().unwrap().push(manifest.clone());
        Ok(self.fragment.map(str::to_string))
    }
}

fn binding(manifest: Manifest) -> CapabilityBinding {
    CapabilityBinding::new(Arc::new(
        scripted_sandbox_session([]).with_manifest(manifest),
    ))
}

async fn instructions_of(agent: &openai_agents_sdk::Agent<()>) -> Option<String> {
    use openai_agents_sdk::RunContext;
    Arc::new(agent.clone())
        .get_system_prompt(&RunContext::new(()))
        .await
        .expect("instructions resolve")
}

/// Python: `test_prepare_sandbox_agent_passes_session_manifest_to_capability_instructions`
/// (also covers `..._wraps_capabilities_without_agent_instructions` and
/// `..._uses_default_sandbox_instructions_when_base_missing`, whose only difference is which
/// leading blocks are present).
#[tokio::test]
async fn preparation_passes_the_session_manifest_to_capability_instructions() {
    let manifest = Manifest::new().with_root("/workspace");
    let capability = Recording::new("test", Some("capability fragment"));
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_base_instructions("base instructions")
        .with_instructions("additional instructions")
        .with_capabilities(vec![capability.clone()]);

    let prepared = agent.prepare(&binding(manifest.clone())).await.unwrap();

    let result = instructions_of(&prepared).await.expect("instructions");
    assert!(result.starts_with("base instructions\n\n# Agent instructions\n\nadditional instructions\n\n# Sandbox capability instructions\n\ncapability fragment\n\n# Filesystem\n"), "{result}");
    let seen = capability.manifests.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].root(), manifest.root());

    // Without agent instructions the middle block is absent.
    let capability = Recording::new("test", Some("capability fragment"));
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_base_instructions("base instructions")
        .with_capabilities(vec![capability.clone()]);
    let prepared = agent.prepare(&binding(manifest.clone())).await.unwrap();
    let result = instructions_of(&prepared).await.expect("instructions");
    assert!(result.starts_with(
        "base instructions\n\n# Sandbox capability instructions\n\ncapability fragment\n\n# Filesystem\n"
    ), "{result}");
    assert!(!result.contains("# Agent instructions"), "{result}");
}

/// Python: `test_prepare_sandbox_agent_prepares_default_compaction_policy`.
#[tokio::test]
async fn preparation_prepares_the_default_compaction_policy() {
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_instructions("base instructions")
        .with_capabilities(vec![Arc::new(Compaction::new())]);

    let prepared = agent
        .prepare(&binding(Manifest::new().with_root("/workspace")))
        .await
        .unwrap();

    let extra_args = prepared
        .model_settings()
        .extra_args
        .as_ref()
        .expect("extra args");
    assert!(extra_args.contains_key("context_management"));
    assert!(!extra_args.contains_key("model"));
}

/// Python: `test_prepare_sandbox_agent_passes_default_model_to_capability_sampling_params`.
#[tokio::test]
async fn preparation_passes_the_model_to_capability_sampling_params() {
    let capability = Recording::new("test", None);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_instructions("base instructions")
        .with_model("gpt-5.5")
        .with_capabilities(vec![capability.clone()]);

    agent
        .prepare(&binding(Manifest::new().with_root("/workspace")))
        .await
        .unwrap();

    let calls = capability.sampling_calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].get("model"), Some(&json!("gpt-5.5")));
}

/// Python: `test_filesystem_instructions_tell_model_to_ls_when_manifest_tree_is_truncated`.
#[tokio::test]
async fn the_filesystem_block_tells_the_model_to_ls_when_the_tree_is_truncated() {
    let mut manifest = Manifest::new().with_root("/workspace");
    for index in 0..200 {
        manifest = manifest.with_entry(
            format!("file_{index:03}.txt"),
            Entry::file(Vec::new()).with_description("x".repeat(40)),
        );
    }
    let agent = SandboxAgent::<()>::new("sandbox").with_instructions("base");

    let prepared = agent.prepare(&binding(manifest)).await.unwrap();

    let result = instructions_of(&prepared).await.expect("instructions");
    assert!(result.contains("... (truncated "), "{result}");
    assert!(
        result.contains(
            "The filesystem layout above was truncated. Use `ls` to explore specific directories before relying on omitted paths."
        ),
        "{result}"
    );
}

/// Python: `test_filesystem_instructions_describe_run_working_directory`.
#[tokio::test]
async fn the_filesystem_block_describes_the_run_working_directory() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("tasks/a", Entry::file(Vec::new()));
    let binding = binding(manifest)
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/a")).expect("scope"));
    let agent = SandboxAgent::<()>::new("sandbox").with_instructions("base");

    let prepared = agent.prepare(&binding).await.unwrap();

    let result = instructions_of(&prepared).await.expect("instructions");
    for line in [
        "For this run, the working directory is `/workspace/tasks/a`.",
        "Relative paths passed to the built-in `exec_command`, `view_image`, and `apply_patch` tools resolve from this directory.",
        "Other sandbox tools follow their own path contract.",
        "The session workspace root remains `/workspace`.",
        "The working directory changes path resolution; it does not isolate this run from the rest of the session workspace.",
        "Files outside the working directory may be visible to or shared with other runs.",
    ] {
        assert!(result.contains(line), "missing {line:?} in {result}");
    }
}

/// Python: `test_prepare_sandbox_agent_validates_required_capabilities`.
#[tokio::test]
async fn preparation_validates_required_capabilities() {
    let manifest = Manifest::new().with_root("/workspace");

    let error = SandboxAgent::<()>::new("sandbox")
        .with_instructions("base instructions")
        .with_capabilities(vec![Arc::new(Memory::new())])
        .prepare(&binding(manifest.clone()))
        .await
        .expect_err("Memory needs filesystem and shell");
    assert!(
        error
            .to_string()
            .contains("Memory requires missing capabilities: filesystem, shell"),
        "{error}"
    );

    let error = SandboxAgent::<()>::new("sandbox")
        .with_instructions("base instructions")
        .with_capabilities(vec![Arc::new(
            Memory::new()
                .with_read(Some(MemoryReadConfig { live_update: false }))
                .with_generate(None),
        )])
        .prepare(&binding(manifest.clone()))
        .await
        .expect_err("Memory needs a shell");
    assert!(
        error
            .to_string()
            .contains("Memory requires missing capabilities: shell"),
        "{error}"
    );

    let prepared = SandboxAgent::<()>::new("sandbox")
        .with_instructions("base instructions")
        .with_capabilities(vec![
            Arc::new(Memory::new()),
            Arc::new(Filesystem::new()),
            Arc::new(Shell::new()),
        ])
        .prepare(&CapabilityBinding::new(Arc::new(
            scripted_sandbox_session([SandboxStep::new(SandboxMethod::ReadFile).failing(
                SandboxError::new(ErrorCode::WorkspaceReadNotFound, "no memory summary yet"),
            )])
            .with_manifest(manifest),
        )))
        .await
        .expect("the requirements are met");
    assert_eq!(prepared.name, "sandbox");
}
