//! Ports of `tests/sandbox/test_runtime.py` (part 1): run preparation — instruction merging,
//! the default sandbox prompt, `run_as` provisioning, the remote mount policy block and the
//! session lifecycle a runner-owned sandbox goes through.
#![cfg(feature = "sandbox")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, Compaction, CompactionPolicy,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::{ExecResult, Group, Permissions, User, file_mode};
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    DEFAULT_SANDBOX_INSTRUCTIONS, ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig,
    SandboxRunner,
};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{FunctionTool, Tool};
use serde_json::json;

// ----------------------------------------------------------------------------------------------
// Harness — mirrors `_FakeSession`, `_FakeClient` and `_RecordingCapability` in the Python file.
// ----------------------------------------------------------------------------------------------

/// A session that counts the lifecycle calls made against it and refuses every I/O call, so a
/// test that only cares about preparation cannot pass against accidental workspace traffic.
#[derive(Debug)]
pub struct CountingSession {
    id: String,
    manifest: Manifest,
    running: std::sync::Mutex<bool>,
    pub start_calls: AtomicUsize,
    pub stop_calls: AtomicUsize,
    pub shutdown_calls: AtomicUsize,
    pub running_calls: AtomicUsize,
    pub exec_commands: std::sync::Mutex<Vec<Vec<String>>>,
    pub files: std::sync::Mutex<BTreeMap<String, Vec<u8>>>,
}

impl CountingSession {
    pub fn new(manifest: Manifest) -> Self {
        CountingSession {
            id: "fake-session".to_string(),
            manifest,
            running: std::sync::Mutex::new(false),
            start_calls: AtomicUsize::new(0),
            stop_calls: AtomicUsize::new(0),
            shutdown_calls: AtomicUsize::new(0),
            running_calls: AtomicUsize::new(0),
            exec_commands: std::sync::Mutex::new(Vec::new()),
            files: std::sync::Mutex::new(BTreeMap::new()),
        }
    }

    pub fn starts(&self) -> usize {
        self.start_calls.load(Ordering::SeqCst)
    }

    pub fn stops(&self) -> usize {
        self.stop_calls.load(Ordering::SeqCst)
    }

    pub fn shutdowns(&self) -> usize {
        self.shutdown_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl SandboxSession for CountingSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "fake"
    }

    async fn start(&self) -> Result<(), SandboxError> {
        self.start_calls.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = true;
        Ok(())
    }

    async fn stop(&self) -> Result<(), SandboxError> {
        self.stop_calls.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = false;
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn running(&self) -> Result<bool, SandboxError> {
        self.running_calls.fetch_add(1, Ordering::SeqCst);
        Ok(*self.running.lock().unwrap())
    }

    async fn exec(&self, request: ExecRequest) -> Result<ExecResult, SandboxError> {
        self.exec_commands
            .lock()
            .unwrap()
            .push(request.resolved_argv());
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        self.files
            .lock()
            .unwrap()
            .get(path)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::WorkspaceReadNotFound,
                    format!("no file at {path}"),
                )
            })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> Result<(), SandboxError> {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), contents.to_vec());
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// Mirrors Python's `_FakeClient`: one pre-built session, with the create/resume/delete calls
/// recorded so a test can assert on the manifest a run created the session with.
#[derive(Debug)]
pub struct FakeClient {
    session: Arc<CountingSession>,
    pub created_manifests: std::sync::Mutex<Vec<Option<Manifest>>>,
    pub resume_states: std::sync::Mutex<Vec<SandboxSessionState>>,
    pub delete_calls: AtomicUsize,
}

impl FakeClient {
    pub fn new(session: Arc<CountingSession>) -> Self {
        FakeClient {
            session,
            created_manifests: std::sync::Mutex::new(Vec::new()),
            resume_states: std::sync::Mutex::new(Vec::new()),
            delete_calls: AtomicUsize::new(0),
        }
    }

    /// A client over a session created from `manifest`.
    pub fn over(manifest: Manifest) -> (Arc<CountingSession>, Arc<FakeClient>) {
        let session = Arc::new(CountingSession::new(manifest));
        (session.clone(), Arc::new(FakeClient::new(session)))
    }

    pub fn created_manifest(&self) -> Manifest {
        self.created_manifests
            .lock()
            .unwrap()
            .first()
            .cloned()
            .expect("the run created a session")
            .expect("the run passed a manifest")
    }

    pub fn deletes(&self) -> usize {
        self.delete_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl SandboxClient for FakeClient {
    fn backend_id(&self) -> &str {
        "fake"
    }

    async fn create(
        &self,
        manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.created_manifests.lock().unwrap().push(manifest);
        Ok(self.session.clone())
    }

    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        self.delete_calls.fetch_add(1, Ordering::SeqCst);
        // The real clients stop and shut the session down before releasing the backend
        // resource; Python's `_SandboxSessionResources.cleanup` asserts that ordering.
        self.session.stop().await?;
        self.session.shutdown().await?;
        Ok(())
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        _manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        self.resume_states.lock().unwrap().push(state.clone());
        Ok(self.session.clone())
    }
}

/// Mirrors Python's `_RecordingCapability`: contributes a fixed instruction fragment and a fixed
/// set of tools, and records the session it was bound to.
#[derive(Debug)]
pub struct RecordingCapability {
    instruction_text: Option<String>,
    tool_names: Vec<String>,
    pub bound_sessions: std::sync::Mutex<Vec<String>>,
}

impl RecordingCapability {
    pub fn new(instruction_text: Option<&str>) -> Self {
        RecordingCapability {
            instruction_text: instruction_text.map(str::to_string),
            tool_names: Vec::new(),
            bound_sessions: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn with_tool(mut self, name: &str) -> Self {
        self.tool_names.push(name.to_string());
        self
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for RecordingCapability {
    fn capability_type(&self) -> &str {
        "recording"
    }

    fn tools(&self, binding: &CapabilityBinding) -> openai_agents_sdk::error::Result<Vec<Tool<C>>> {
        self.bound_sessions
            .lock()
            .unwrap()
            .push(binding.session.id().to_string());
        Ok(self
            .tool_names
            .iter()
            .map(|name| Tool::from(function_tool::<C>(name, "ok")))
            .collect())
    }

    async fn instructions(
        &self,
        _manifest: &Manifest,
    ) -> openai_agents_sdk::error::Result<Option<String>> {
        Ok(self.instruction_text.clone())
    }
}

/// Mirrors `tests/test_responses.py::get_function_tool`.
fn function_tool<C: Send + Sync + 'static>(name: &str, result: &str) -> FunctionTool<C> {
    let result = result.to_string();
    FunctionTool::new(name, "", move |_ctx, _a: serde_json::Value| {
        let result = result.clone();
        async move { Ok(result) }
    })
}

fn scripted(text: &str) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message(text)])])
            .expect("scripted model"),
    )
}

/// The names of the function tools a recorded model call carried.
fn tool_names(call: &openai_agents_sdk::testing::ModelCall) -> Vec<String> {
    call.request
        .tools
        .iter()
        .filter_map(|tool| match tool {
            openai_agents_sdk::models::interface::ToolSpec::Function(spec) => {
                Some(spec.name.clone())
            }
            openai_agents_sdk::models::interface::ToolSpec::Hosted { .. } => None,
        })
        .collect()
}

/// The `# Filesystem` block the Rust preparation appends, mirroring Python's
/// `runtime_agent_preparation._filesystem_instructions`.
fn filesystem_instructions(manifest: &Manifest) -> String {
    format!(
        "# Filesystem\nYou have access to a container with a filesystem. The filesystem layout \
         is:\n\n{}",
        manifest.describe().trim_end()
    )
}

// ----------------------------------------------------------------------------------------------
// Preparation: instructions and tools
// ----------------------------------------------------------------------------------------------

/// Python: `test_sandbox_package_exports_permission_types`.
#[test]
fn sandbox_package_exports_permission_types() {
    assert_eq!(User::new("sandbox-user").name, "sandbox-user");
    assert!(Group::new("sandbox-group").users.is_empty());
    assert_eq!(Permissions::default().owner, file_mode::ALL);
}

/// Python: `test_runner_merges_sandbox_instructions_and_tools`.
///
/// The prepared agent's system prompt is the default sandbox prompt, then the agent's own
/// instructions, then the capability fragment, then the filesystem layout — and the capability's
/// tools are the only tools the model sees.
#[tokio::test]
async fn runner_merges_sandbox_instructions_and_tools() {
    let model = scripted("done");
    let manifest = Manifest::new().with_entry(
        "README.md",
        Entry::file(b"Follow the repo contract.".to_vec()),
    );
    let (session, client) = FakeClient::over(manifest.clone());
    let capability = Arc::new(
        RecordingCapability::new(Some("Capability instructions.")).with_tool("capability_tool"),
    );

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Additional instructions.")
        .with_default_manifest(manifest.clone())
        .with_capabilities(vec![capability.clone()]);

    let config = SandboxRunConfig::with_client(client.clone());
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(session.starts(), 1);
    assert_eq!(session.stops(), 1);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(client.deletes(), 1);

    // The created session got the agent's default manifest, not the agent's own value.
    assert_eq!(client.created_manifest().entries().len(), 1);

    let call = model.first_call().expect("the model was called");
    let expected = format!(
        "{}\n\n# Agent instructions\n\nAdditional instructions.\n\n# Sandbox capability \
         instructions\n\nCapability instructions.\n\n{}",
        DEFAULT_SANDBOX_INSTRUCTIONS.trim(),
        filesystem_instructions(&manifest),
    );
    assert_eq!(
        call.request.system_instructions.as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(tool_names(&call), vec!["capability_tool".to_string()]);
}

/// Python: `test_runner_uses_default_sandbox_prompt_when_instructions_missing`.
#[tokio::test]
async fn runner_uses_default_sandbox_prompt_when_instructions_missing() {
    let model = scripted("done");
    let (_session, client) = FakeClient::over(Manifest::new());
    let capability = Arc::new(RecordingCapability::new(Some("Capability instructions.")));

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![capability]);

    let config = SandboxRunConfig::with_client(client);
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let call = model.first_call().expect("the model was called");
    let expected = format!(
        "{}\n\n# Sandbox capability instructions\n\nCapability instructions.\n\n{}",
        DEFAULT_SANDBOX_INSTRUCTIONS.trim(),
        filesystem_instructions(&Manifest::new()),
    );
    assert_eq!(
        call.request.system_instructions.as_deref(),
        Some(expected.as_str())
    );
}

/// Python: `test_filesystem_instructions_omit_extra_path_grants`.
///
/// Extra path grants widen what the tools may touch; they are deliberately not advertised in the
/// filesystem layout block.
#[test]
fn filesystem_instructions_omit_extra_path_grants() {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(
            SandboxPathGrant::new("/tmp")
                .expect("grant path")
                .with_description("temporary files"),
        )
        .with_extra_path_grant(
            SandboxPathGrant::new("/opt/toolchain")
                .expect("grant path")
                .read_only(true)
                .with_description("compiler runtime"),
        );

    let rendered = filesystem_instructions(&manifest);
    assert!(!rendered.contains("/tmp"), "{rendered}");
    assert!(!rendered.contains("/opt/toolchain"), "{rendered}");
    assert!(rendered.starts_with(
        "# Filesystem\nYou have access to a container with a filesystem. The filesystem layout \
         is:\n\n/workspace"
    ));
}

/// Python: `test_runner_adds_run_as_user_to_created_manifest_without_default_manifest`.
#[tokio::test]
async fn runner_adds_run_as_user_to_created_manifest_without_default_manifest() {
    let model = scripted("done");
    let (_session, client) = FakeClient::over(Manifest::new());
    let run_as = User::new("sandbox-user");

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_run_as(run_as.clone());

    let config = SandboxRunConfig::with_client(client.clone());
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let created = client.created_manifest();
    assert_eq!(
        created
            .users()
            .iter()
            .map(|user| user.name.clone())
            .collect::<Vec<_>>(),
        vec!["sandbox-user".to_string()]
    );
}

/// Python: `test_runner_requires_sandbox_config_for_sandbox_agent`.
#[tokio::test]
async fn runner_requires_sandbox_config_for_sandbox_agent() {
    let agent = SandboxAgent::<()>::new("sandbox").with_model_instance(scripted("done"));
    let config = SandboxRunConfig::default();
    let error = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect_err("a sandbox agent needs a client or a session");
    assert!(error.to_string().contains("client"), "{error}");
}

/// Python: `test_runner_dynamic_instructions_do_not_override_default_sandbox_prompt`.
///
/// Dynamic agent instructions that resolve to an empty string contribute no `# Agent
/// instructions` section, and never displace the default sandbox prompt.
#[tokio::test]
async fn dynamic_instructions_do_not_override_the_default_sandbox_prompt() {
    let model = scripted("done");
    let (_session, client) = FakeClient::over(Manifest::new());
    let capability = Arc::new(RecordingCapability::new(Some("Capability instructions.")));

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![capability])
        .map_agent(|agent| {
            agent.with_dynamic_instructions(|_ctx, _agent| async move { Ok(String::new()) })
        });

    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let call = model.first_call().expect("the model was called");
    let expected = format!(
        "{}\n\n# Sandbox capability instructions\n\nCapability instructions.\n\n{}",
        DEFAULT_SANDBOX_INSTRUCTIONS.trim(),
        filesystem_instructions(&Manifest::new()),
    );
    assert_eq!(
        call.request.system_instructions.as_deref(),
        Some(expected.as_str())
    );
}

/// Python: `test_runner_base_instructions_override_default_sandbox_prompt`.
#[tokio::test]
async fn base_instructions_override_the_default_sandbox_prompt() {
    let model = scripted("done");
    let (_session, client) = FakeClient::over(Manifest::new());
    let capability = Arc::new(RecordingCapability::new(Some("Capability instructions.")));

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_base_instructions("Custom base instructions.")
        .with_instructions("Additional instructions.")
        .with_capabilities(vec![capability]);

    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let call = model.first_call().expect("the model was called");
    let expected = format!(
        "Custom base instructions.\n\n# Agent instructions\n\nAdditional instructions.\n\n# \
         Sandbox capability instructions\n\nCapability instructions.\n\n{}",
        filesystem_instructions(&Manifest::new()),
    );
    assert_eq!(
        call.request.system_instructions.as_deref(),
        Some(expected.as_str())
    );
}

/// A read-only S3 mount placed at `remote`, attached in-container through `mount-s3`.
fn s3_mount_manifest(read_only: bool, ephemeral: bool) -> Manifest {
    let mount = ConfiguredMountStrategy::mountpoint(S3Mount::new("bucket"))
        .mount()
        .with_read_only(read_only)
        .with_ephemeral(ephemeral);
    Manifest::new().with_entry("remote", Entry::mount(mount))
}

/// Python: `test_runner_adds_remote_mount_policy_instructions`.
///
/// The policy block names the mount path and its mode, and sits between the capability section
/// and the filesystem layout.
#[tokio::test]
async fn runner_adds_remote_mount_policy_instructions() {
    let model = scripted("done");
    let manifest = s3_mount_manifest(true, true);
    let (_session, client) = FakeClient::over(manifest.clone());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_default_manifest(manifest.clone())
        .with_capabilities(vec![Arc::new(RecordingCapability::new(Some(
            "Capability text.",
        )))]);

    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let call = model.first_call().expect("the model was called");
    let instructions = call
        .request
        .system_instructions
        .clone()
        .expect("the prepared agent has instructions");
    assert!(
        instructions.contains("- /workspace/remote (mounted in read-only mode)"),
        "{instructions}"
    );
    assert!(
        instructions
            .contains("Do not edit paths marked read-only in place, including with `apply_patch`"),
        "{instructions}"
    );

    let agent_index = instructions
        .find("# Agent instructions")
        .expect("agent block");
    let capability_index = instructions
        .find("# Sandbox capability instructions")
        .expect("capability block");
    let policy_index = instructions
        .find("# Sandbox remote mount policy")
        .expect("policy block");
    let filesystem_index = instructions.find("# Filesystem").expect("filesystem block");
    assert!(agent_index < capability_index);
    assert!(capability_index < policy_index);
    assert!(policy_index < filesystem_index);
}

/// Python: `test_runner_adds_remote_mount_policy_for_non_ephemeral_mounts`.
#[tokio::test]
async fn runner_adds_remote_mount_policy_for_non_ephemeral_mounts() {
    let model = scripted("done");
    let manifest = s3_mount_manifest(true, false);
    let (_session, client) = FakeClient::over(manifest.clone());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_default_manifest(manifest);

    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let instructions = model
        .first_call()
        .expect("the model was called")
        .request
        .system_instructions
        .clone()
        .expect("instructions");
    assert!(
        instructions.contains("- /workspace/remote (mounted in read-only mode)"),
        "{instructions}"
    );
}

/// Python: `test_runner_marks_writable_remote_mounts_in_policy`.
#[tokio::test]
async fn runner_marks_writable_remote_mounts_in_policy() {
    let model = scripted("done");
    let manifest = s3_mount_manifest(false, true);
    let (_session, client) = FakeClient::over(manifest.clone());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_default_manifest(manifest);

    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let instructions = model
        .first_call()
        .expect("the model was called")
        .request
        .system_instructions
        .clone()
        .expect("instructions");
    assert!(
        instructions.contains("- /workspace/remote (mounted in read+write mode)"),
        "{instructions}"
    );
    assert!(
        instructions.contains("Use `apply_patch` directly for text edits on read+write mounts."),
        "{instructions}"
    );
    assert!(
        instructions.contains(
            "For shell-based edits on read+write mounts, first `cp` the mounted file to a normal \
             local workspace path, edit the local copy there, then copy it back."
        ),
        "{instructions}"
    );
}

/// Python: `test_runner_uses_manifest_remote_mount_command_allowlist_override`.
#[tokio::test]
async fn runner_uses_the_manifest_remote_mount_command_allowlist_override() {
    let model = scripted("done");
    let manifest = s3_mount_manifest(true, true).with_remote_mount_command_allowlist(["ls", "cp"]);
    let (_session, client) = FakeClient::over(manifest.clone());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_default_manifest(manifest);

    let config = SandboxRunConfig::with_client(client);
    SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    let instructions = model
        .first_call()
        .expect("the model was called")
        .request
        .system_instructions
        .clone()
        .expect("instructions");
    assert!(
        instructions.contains("Only use these commands on remote mounts:"),
        "{instructions}"
    );
    assert!(instructions.contains("`ls`, `cp`"), "{instructions}");
}

/// Python: `test_runner_applies_compaction_capability_to_input_and_model_settings`.
///
/// The compaction capability drops everything before the last `compaction` item and adds the
/// Responses `context_management` request parameter.
#[tokio::test]
async fn compaction_capability_reaches_the_input_and_the_model_settings() {
    let model = scripted("done");
    let (_session, client) = FakeClient::over(Manifest::new());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_default_manifest(Manifest::new())
        .with_capabilities(vec![Arc::new(
            Compaction::new().with_policy(CompactionPolicy::Static { threshold: 123 }),
        )]);

    let input: Vec<InputItem> = vec![
        InputItem::from_value(json!({"type": "message", "role": "user", "content": "old-user"})),
        InputItem::from_value(json!({"type": "compaction", "summary": "compacted-up-to-here"})),
        InputItem::from_value(
            json!({"type": "message", "role": "assistant", "content": "recent-assistant"}),
        ),
        InputItem::from_value(json!({"type": "message", "role": "user", "content": "new-user"})),
    ];

    let config = SandboxRunConfig::with_client(client);
    let result = SandboxRunner::run(&agent, input.clone(), &config)
        .await
        .expect("the run completes");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let call = model.first_call().expect("the model was called");
    assert_eq!(call.request.input, input[1..].to_vec());
    assert_eq!(
        call.request
            .model_settings
            .extra_args
            .clone()
            .map(serde_json::Value::Object),
        Some(json!({
            "context_management": [{"type": "compaction", "compact_threshold": 123}]
        }))
    );
}

/// Python: `test_runner_does_not_close_injected_sandbox_session`.
///
/// A caller-owned session is started if needed but never stopped, shut down or deleted, and the
/// run keeps the manifest the session was started with rather than the agent's default or the
/// run config override.
#[tokio::test]
async fn runner_does_not_close_an_injected_sandbox_session() {
    let model = scripted("done");
    let default_manifest = Manifest::new().with_entry("default.txt", Entry::text_file("default"));
    let session_manifest = Manifest::new().with_entry("session.txt", Entry::text_file("session"));
    let injected = Arc::new(CountingSession::new(session_manifest.clone()));

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_default_manifest(default_manifest);

    let config = SandboxRunConfig::with_session(injected.clone())
        .manifest(Manifest::new().with_entry("override.txt", Entry::text_file("override")));
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(injected.starts(), 1);
    assert_eq!(injected.stops(), 0);
    assert_eq!(injected.shutdowns(), 0);
    assert_eq!(
        injected.manifest().entries().keys().collect::<Vec<_>>(),
        session_manifest.entries().keys().collect::<Vec<_>>()
    );
}

/// Python: `test_runner_does_not_restart_running_injected_sandbox_session`.
#[tokio::test]
async fn runner_does_not_restart_a_running_injected_session() {
    let model = scripted("done");
    let injected = Arc::new(CountingSession::new(
        Manifest::new().with_entry("session.txt", Entry::text_file("session")),
    ));
    injected.start().await.expect("pre-start the session");
    injected.start_calls.store(0, Ordering::SeqCst);

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.");

    let config = SandboxRunConfig::with_session(injected.clone());
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(injected.starts(), 0);
    assert_eq!(injected.stops(), 0);
    assert_eq!(injected.shutdowns(), 0);
}

/// Python: `test_runner_streamed_cleans_runner_owned_session`.
#[tokio::test]
async fn streamed_runs_clean_up_the_runner_owned_session() {
    use futures::StreamExt;

    let model = scripted("done");
    let (session, client) = FakeClient::over(Manifest::new());
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.");

    let config = SandboxRunConfig::with_client(client.clone());
    let mut run = SandboxRunner::run_streamed(&agent, "hello", &config)
        .await
        .expect("the stream starts");
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        let mut seen = 0usize;
        while stream.next().await.is_some() {
            seen += 1;
        }
        assert!(seen > 0, "the stream produced events");
    }

    assert_eq!(run.streaming.final_output_text().as_deref(), Some("done"));
    assert_eq!(session.starts(), 1);
    assert_eq!(session.stops(), 1);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(client.deletes(), 1);
    // A completed run releases the sandbox rather than leaving resume state behind.
    assert!(run.sandbox_state().is_none());
    assert!(run.take_guard().is_none());
}

/// Python: `test_runner_streamed_cleans_sandbox_once_after_stream_completion` — draining the
/// stream twice must not tear the session down twice.
#[tokio::test]
async fn a_streamed_sandbox_is_cleaned_up_exactly_once() {
    use futures::StreamExt;

    let model = scripted("done");
    let (session, client) = FakeClient::over(Manifest::new());
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.");

    let config = SandboxRunConfig::with_client(client.clone());
    let mut run = SandboxRunner::run_streamed(&agent, "hello", &config)
        .await
        .expect("the stream starts");
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {}
    }
    run.finish().await.expect("a second finish is a no-op");

    assert_eq!(session.stops(), 1);
    assert_eq!(session.shutdowns(), 1);
    assert_eq!(client.deletes(), 1);
}

/// Python: `test_runner_allows_fresh_sessions_for_clients_with_default_options`.
#[tokio::test]
async fn a_client_with_default_options_gets_one_fresh_session() {
    let model = scripted("done");
    let (_session, client) = FakeClient::over(Manifest::new());
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_default_manifest(Manifest::new());

    let config = SandboxRunConfig::with_client(client.clone());
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(client.created_manifests.lock().unwrap().len(), 1);
}

/// Python: `test_runner_reuses_prepared_sandbox_agent_across_turns_for_tool_choice_reset`.
///
/// `tool_choice="required"` applies to the first turn only; the prepared agent is reused across
/// turns, so the reset survives into the final call.
#[tokio::test]
async fn tool_choice_required_is_reset_after_the_first_tool_call() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![openai_agents_sdk::testing::function_call(
                "capability_tool",
                "{}",
            )]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .expect("scripted model"),
    );
    let (_session, client) = FakeClient::over(Manifest::new());

    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_tool(function_tool::<()>("capability_tool", "ok"))
        .map_agent(|agent| {
            let mut settings = agent.model_settings().clone();
            settings.tool_choice = Some(openai_agents_sdk::model_settings::ToolChoice::required());
            agent.with_model_settings(settings)
        });

    let config = SandboxRunConfig::with_client(client);
    let result = SandboxRunner::run(&agent, "hello", &config)
        .await
        .expect("the run completes");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let calls = model.calls();
    assert_eq!(
        calls[0].request.model_settings.tool_choice,
        Some(openai_agents_sdk::model_settings::ToolChoice::required())
    );
    assert_eq!(
        calls.last().unwrap().request.model_settings.tool_choice,
        None
    );
}
