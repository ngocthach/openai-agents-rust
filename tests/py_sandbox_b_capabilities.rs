//! Ports of `tests/sandbox/capabilities/test_shell_capability.py`,
//! `test_filesystem_capability.py`, `test_compaction_capability.py` and
//! `test_tool_input_errors.py`.
//!
//! Rust's shell capability keeps the crate-native `shell` tool alongside Python's
//! `exec_command`/`write_stdin` pair, and routes a working directory through
//! [`ExecRequest::cwd`] instead of prefixing `cd <dir> || exit` to the command, so the
//! assertions below check the resolved working directory rather than Python's command string.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::RunContext;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::sandbox::capabilities::compaction::CompactionPolicy;
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, Compaction, Filesystem, Shell,
};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::{ExecResult, User};
use openai_agents_sdk::sandbox::workspace_paths::{SandboxPathGrant, SandboxWorkspaceScope};
use openai_agents_sdk::tool::{Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use serde_json::{Value, json};

// ----------------------------------------------------------------------------------------------
// A session that records the requests the capability tools make.
// ----------------------------------------------------------------------------------------------

/// What a recorded session answers an `exec` or a PTY start with.
#[derive(Debug, Clone)]
enum ExecOutcome {
    /// A command result.
    Result(ExecResult),
    /// A failure.
    Error(SandboxError),
    /// The default: echo the command back on both streams with exit code 7, as Python's
    /// `_default_exec_result` does.
    Echo,
}

#[derive(Debug)]
struct RecordingSession {
    manifest: Manifest,
    supports_pty: bool,
    exec_outcome: ExecOutcome,
    pty_outcome: Mutex<Option<SandboxResult<PtyExecUpdate>>>,
    exec_calls: Mutex<Vec<ExecRequest>>,
    pty_calls: Mutex<Vec<PtyExecRequest>>,
    stdin_outcome: Mutex<Option<SandboxResult<PtyExecUpdate>>>,
    files: Mutex<std::collections::BTreeMap<String, Vec<u8>>>,
}

impl RecordingSession {
    fn new() -> Self {
        RecordingSession {
            manifest: Manifest::new().with_root("/workspace"),
            supports_pty: false,
            exec_outcome: ExecOutcome::Echo,
            pty_outcome: Mutex::new(None),
            exec_calls: Mutex::new(Vec::new()),
            pty_calls: Mutex::new(Vec::new()),
            stdin_outcome: Mutex::new(None),
            files: Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    fn with_manifest(mut self, manifest: Manifest) -> Self {
        self.manifest = manifest;
        self
    }

    fn with_pty(mut self, update: SandboxResult<PtyExecUpdate>) -> Self {
        self.supports_pty = true;
        self.pty_outcome = Mutex::new(Some(update));
        self
    }

    fn with_stdin(self, outcome: SandboxResult<PtyExecUpdate>) -> Self {
        *self.stdin_outcome.lock().unwrap() = Some(outcome);
        self
    }

    fn with_exec_result(mut self, result: ExecResult) -> Self {
        self.exec_outcome = ExecOutcome::Result(result);
        self
    }

    fn with_exec_error(mut self, error: SandboxError) -> Self {
        self.exec_outcome = ExecOutcome::Error(error);
        self
    }

    fn with_file(self, path: &str, contents: impl Into<Vec<u8>>) -> Self {
        self.files
            .lock()
            .unwrap()
            .insert(path.to_string(), contents.into());
        self
    }

    fn exec_calls(&self) -> Vec<ExecRequest> {
        self.exec_calls.lock().unwrap().clone()
    }

    fn pty_calls(&self) -> Vec<PtyExecRequest> {
        self.pty_calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl SandboxSession for RecordingSession {
    fn id(&self) -> &str {
        "recording"
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn backend_id(&self) -> &str {
        "recording"
    }

    fn supports_pty(&self) -> bool {
        self.supports_pty
    }

    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }

    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        self.exec_calls.lock().unwrap().push(request.clone());
        match &self.exec_outcome {
            ExecOutcome::Result(result) => Ok(result.clone()),
            ExecOutcome::Error(error) => Err(error.clone()),
            ExecOutcome::Echo => {
                let rendered = request.command.join(" ");
                Ok(ExecResult {
                    stdout: format!("stdout: {rendered}").into_bytes(),
                    stderr: format!("stderr: {rendered}").into_bytes(),
                    exit_code: 7,
                })
            }
        }
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        self.pty_calls.lock().unwrap().push(request);
        match self.pty_outcome.lock().unwrap().take() {
            Some(outcome) => outcome,
            None => Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "no scripted PTY result",
            )),
        }
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        let _ = request;
        if !self.supports_pty {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "PTY execution is not supported by this sandbox session",
            ));
        }
        if let Some(outcome) = self.stdin_outcome.lock().unwrap().take() {
            return outcome;
        }
        Ok(PtyExecUpdate {
            process_id: Some(1337),
            output: b"stdin".to_vec(),
            exit_code: None,
            original_token_count: None,
        })
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let resolved = self.normalize_path(path)?;
        self.files
            .lock()
            .unwrap()
            .get(&resolved)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::WorkspaceReadNotFound,
                    format!("missing {resolved}"),
                )
            })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let resolved = self.normalize_path_for_write(path)?;
        self.files
            .lock()
            .unwrap()
            .insert(resolved, contents.to_vec());
        Ok(())
    }

    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

fn binding(session: Arc<dyn SandboxSession>) -> CapabilityBinding {
    CapabilityBinding::new(session)
}

fn tool_names(tools: &[Tool<()>]) -> Vec<String> {
    tools.iter().map(Tool::name).collect()
}

fn function_tool(tools: &[Tool<()>], name: &str) -> Arc<openai_agents_sdk::tool::FunctionTool<()>> {
    tools
        .iter()
        .find_map(|tool| match tool {
            Tool::Function(function) if function.name == name => Some(function.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("tool `{name}` is exposed; got {:?}", tool_names(tools)))
}

async fn invoke(tool: &openai_agents_sdk::tool::FunctionTool<()>, arguments: Value) -> String {
    let arguments = arguments.to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", &tool.name, &arguments);
    match tool.invoke(ctx, arguments).await.expect("tool invocation") {
        ToolOutput::Text(text) => text,
        other => panic!("expected text output, got {other:?}"),
    }
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/capabilities/test_shell_capability.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_tools_exposes_exec_command_function_tool_after_bind`. A session without a PTY
/// still gets `exec_command`; Rust adds its own `shell` tool alongside it.
#[test]
fn shell_tools_expose_exec_command_without_a_pty() {
    let session = Arc::new(RecordingSession::new());
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session)).unwrap();

    assert_eq!(tool_names(&tools), vec!["shell", "exec_command"]);
}

/// Python: `test_tools_exposes_write_stdin_for_pty_sessions` and
/// `test_tools_keep_both_pty_session_methods_callable`.
#[test]
fn shell_tools_expose_write_stdin_for_pty_sessions() {
    let session = Arc::new(RecordingSession::new().with_pty(Ok(PtyExecUpdate {
        process_id: Some(1),
        output: Vec::new(),
        exit_code: None,
        original_token_count: None,
    })));
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session)).unwrap();

    assert_eq!(
        tool_names(&tools),
        vec!["shell", "exec_command", "write_stdin"]
    );
}

/// Python: `test_configure_tools_can_customize_shell_approvals_after_clone` and
/// `test_configure_tools_can_replace_exec_command_tool`. Rust configures approvals on the
/// capability rather than through a mutable tool set, so this checks the same end state.
#[test]
fn shell_tools_can_require_approval() {
    let session = Arc::new(RecordingSession::new().with_pty(Ok(PtyExecUpdate {
        process_id: Some(1),
        output: Vec::new(),
        exit_code: None,
        original_token_count: None,
    })));
    let capability = Shell::new().with_needs_approval(true);
    let tools = Capability::<()>::tools(&capability, &binding(session)).unwrap();

    for name in ["exec_command", "write_stdin"] {
        assert!(
            matches!(
                function_tool(&tools, name).needs_approval,
                openai_agents_sdk::tool::NeedsApproval::Static(true)
            ),
            "`{name}` requires approval"
        );
    }
}

/// Python: `test_tools_can_be_renamed`-adjacent: the capability can rename its one-shot tool.
#[test]
fn shell_tool_can_be_renamed() {
    let session = Arc::new(RecordingSession::new());
    let capability = Shell::new().with_tool_name("run").with_pty(false);
    let tools = Capability::<()>::tools(&capability, &binding(session)).unwrap();

    assert_eq!(tool_names(&tools), vec!["run"]);
}

/// Python: `test_instructions_match_sandbox_shell_guidance`. Rust keeps the extra `shell` tool,
/// so the guidance names it too; the lines Python asserts on are checked individually.
#[tokio::test]
async fn shell_instructions_name_the_pty_tools() {
    let instructions = Capability::<()>::instructions(&Shell::new(), &Manifest::new())
        .await
        .unwrap()
        .expect("the shell capability contributes instructions");

    assert!(
        instructions.starts_with("When using the shell:\n"),
        "{instructions}"
    );
    assert!(instructions.contains("`exec_command`"), "{instructions}");
    assert!(
        instructions.contains(
            "- If available, use `write_stdin` to interact with or poll running sessions."
        ),
        "{instructions}"
    );
    assert!(instructions.contains("`tty=true`"), "{instructions}");
    assert!(instructions.contains("\\u0003"), "{instructions}");
    assert!(
        instructions
            .contains("- Prefer `rg` and `rg --files` for text/file discovery when available."),
        "{instructions}"
    );
    assert!(
        instructions.contains("- Avoid using Python scripts just to print large file chunks."),
        "{instructions}"
    );
}

/// Python: `test_exec_command_tool_runs_commands_with_source_output_format`. On a session
/// without a PTY the command runs once through `exec`, and both streams are reported.
#[tokio::test]
async fn exec_command_runs_one_shot_on_a_session_without_a_pty() {
    let session = Arc::new(RecordingSession::new());
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();

    let output = invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd", "yield_time_ms": 1500}),
    )
    .await;

    let calls = session.exec_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].command, vec!["pwd".to_string()]);
    assert_eq!(
        calls[0].timeout,
        Some(std::time::Duration::from_millis(1500))
    );
    assert_eq!(calls[0].cwd, None);

    let lines: Vec<&str> = output.lines().collect();
    assert!(lines[0].starts_with("Chunk ID: "), "{output}");
    assert!(lines[1].starts_with("Wall time: "), "{output}");
    assert_eq!(lines[2], "Process exited with code 7");
    assert_eq!(lines[3], "Output:");
    assert_eq!(lines[4], "stdout: pwd");
    assert_eq!(lines[5], "stderr: pwd");
}

/// Python: `test_exec_command_tool_runs_as_bound_user` and
/// `test_exec_command_tool_defaults_to_workspace_scope_cwd`.
#[tokio::test]
async fn exec_command_runs_as_the_bound_user_in_the_scope_cwd() {
    let session = Arc::new(RecordingSession::new().with_exec_result(ExecResult::default()));
    let binding = CapabilityBinding::new(session.clone())
        .with_run_as(Some(User::new("sandbox-user")))
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/a")).expect("scope"));
    let tools = Capability::<()>::tools(&Shell::new(), &binding).unwrap();

    for workdir in [Value::Null, json!(""), json!("   ")] {
        invoke(
            &function_tool(&tools, "exec_command"),
            json!({"cmd": "pwd", "workdir": workdir}),
        )
        .await;
    }

    let calls = session.exec_calls();
    assert_eq!(calls.len(), 3);
    for call in &calls {
        assert_eq!(call.command, vec!["pwd".to_string()]);
        assert_eq!(call.cwd.as_deref(), Some("/workspace/tasks/a"));
        assert_eq!(call.user, Some(User::new("sandbox-user")));
    }
}

/// Python: `test_exec_command_tool_resolves_relative_workdir_from_workspace_scope` and
/// `test_exec_command_tool_normalizes_raw_backslashes_before_workspace_scope`.
#[tokio::test]
async fn exec_command_anchors_a_relative_workdir_in_the_scope() {
    for workdir in ["src/project", r"src\project"] {
        let session = Arc::new(RecordingSession::new().with_exec_result(ExecResult::default()));
        let binding = CapabilityBinding::new(session.clone())
            .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/a")).expect("scope"));
        let tools = Capability::<()>::tools(&Shell::new(), &binding).unwrap();

        invoke(
            &function_tool(&tools, "exec_command"),
            json!({"cmd": "pwd", "workdir": workdir}),
        )
        .await;

        assert_eq!(
            session.exec_calls()[0].cwd.as_deref(),
            Some("/workspace/tasks/a/src/project"),
            "workdir `{workdir}`"
        );
    }
}

/// Python: `test_exec_command_tool_wraps_workdir_and_uses_custom_shell`.
#[tokio::test]
async fn exec_command_uses_a_custom_shell() {
    let session = Arc::new(RecordingSession::new());
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();

    invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd", "workdir": "src/project", "shell": "/bin/bash", "login": false}),
    )
    .await;

    let calls = session.exec_calls();
    assert_eq!(calls[0].cwd.as_deref(), Some("/workspace/src/project"));
    assert_eq!(
        calls[0].timeout,
        Some(std::time::Duration::from_millis(10_000))
    );
    assert_eq!(
        calls[0].shell,
        openai_agents_sdk::sandbox::session::ExecShell::Custom(vec![
            "/bin/bash".to_string(),
            "-c".to_string()
        ])
    );
}

/// Python: `test_exec_command_tool_allows_split_path_grant_workdir`.
#[tokio::test]
async fn exec_command_allows_a_granted_path_outside_the_workspace() {
    let grant = SandboxPathGrant::new("/mnt/shared-data")
        .expect("grant path")
        .with_host_path("/native/shared-data")
        .expect("host path")
        .read_only(true);
    let session = Arc::new(
        RecordingSession::new().with_manifest(
            Manifest::new()
                .with_root("/workspace")
                .with_extra_path_grant(grant),
        ),
    );
    let binding = CapabilityBinding::new(session.clone())
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/a")).expect("scope"));
    let tools = Capability::<()>::tools(&Shell::new(), &binding).unwrap();

    invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd", "workdir": "/mnt/shared-data", "shell": "/bin/bash", "login": false}),
    )
    .await;

    assert_eq!(
        session.exec_calls()[0].cwd.as_deref(),
        Some("/mnt/shared-data")
    );
}

/// Python: `test_exec_command_tool_includes_original_token_count_when_truncating`.
#[tokio::test]
async fn exec_command_reports_the_original_token_count_when_truncating() {
    let session = Arc::new(RecordingSession::new());
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session)).unwrap();

    let output = invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd", "yield_time_ms": 1500, "max_output_tokens": 2}),
    )
    .await;

    // "stdout: pwd\nstderr: pwd" is 23 bytes, which is 6 approximate tokens.
    assert!(output.contains("Original token count: 6"), "{output}");
    assert!(output.contains("Output:\n"), "{output}");
}

/// Python: `test_exec_command_tool_uses_pty_when_supported` and
/// `test_exec_command_tool_starts_pty_as_bound_user`.
#[tokio::test]
async fn exec_command_uses_the_pty_when_the_session_has_one() {
    let session = Arc::new(RecordingSession::new().with_pty(Ok(PtyExecUpdate {
        process_id: Some(1337),
        output: Vec::new(),
        exit_code: None,
        original_token_count: None,
    })));
    let binding = CapabilityBinding::new(session.clone())
        .with_run_as(Some(User::new("sandbox-user")))
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/a")).expect("scope"));
    let tools = Capability::<()>::tools(&Shell::new(), &binding).unwrap();

    let output = invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd", "yield_time_ms": 0, "tty": true}),
    )
    .await;

    let calls = session.pty_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].command, "pwd");
    assert_eq!(calls[0].cwd.as_deref(), Some("/workspace/tasks/a"));
    assert_eq!(calls[0].yield_time_ms, 0);
    assert_eq!(calls[0].user, Some(User::new("sandbox-user")));
    assert!(calls[0].tty);
    assert!(session.exec_calls().is_empty(), "the PTY path was used");
    assert!(
        output.contains("Process running with session ID 1337"),
        "{output}"
    );
    assert!(!output.contains("Process exited"), "{output}");
}

/// Python: the `ExecTransportError` fallback branch of `ExecCommandTool._run`.
#[tokio::test]
async fn exec_command_falls_back_to_one_shot_after_a_retry_safe_transport_failure() {
    let session = Arc::new(
        RecordingSession::new().with_pty(Err(SandboxError::new(
            ErrorCode::ExecTransportError,
            "connection closed while reading HTTP status line",
        )
        .with_context("retry_safe", "true"))),
    );
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();

    let output = invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd"}),
    )
    .await;

    assert_eq!(session.exec_calls().len(), 1);
    assert!(
        output.contains(
            "PTY transport failed before the interactive session opened; fell back to one-shot \
             exec."
        ),
        "{output}"
    );
    assert!(output.contains("stdout: pwd"), "{output}");
}

/// The same failure with `tty=true` cannot fall back: the caller asked for an interactive
/// session, which a one-shot command is not.
#[tokio::test]
async fn exec_command_does_not_fall_back_for_a_tty_request() {
    let session = Arc::new(
        RecordingSession::new().with_pty(Err(SandboxError::new(
            ErrorCode::ExecTransportError,
            "connection closed",
        )
        .with_context("retry_safe", "true"))),
    );
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
    let tool = function_tool(&tools, "exec_command");
    let arguments = json!({"cmd": "pwd", "tty": true}).to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "exec_command", &arguments);

    let error = tool
        .invoke(ctx, arguments)
        .await
        .expect_err("the failure propagates");

    assert!(error.to_string().contains("connection closed"), "{error}");
    assert!(session.exec_calls().is_empty());
}

/// A transport failure that is not marked retry-safe is not re-run either: the command may
/// already have had an effect.
#[tokio::test]
async fn exec_command_does_not_fall_back_without_a_retry_safe_marker() {
    let session = Arc::new(RecordingSession::new().with_pty(Err(SandboxError::new(
        ErrorCode::ExecTransportError,
        "connection closed",
    ))));
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
    let tool = function_tool(&tools, "exec_command");
    let arguments = json!({"cmd": "pwd"}).to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "exec_command", &arguments);

    tool.invoke(ctx, arguments)
        .await
        .expect_err("the failure propagates");
    assert!(session.exec_calls().is_empty());
}

/// Python: the `ExecTimeoutError` branch of `ExecCommandTool._run`.
#[tokio::test]
async fn exec_command_reports_a_timeout_as_output() {
    let session = Arc::new(
        RecordingSession::new()
            .with_exec_error(SandboxError::new(ErrorCode::ExecTimeout, "timed out")),
    );
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session)).unwrap();

    let output = invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "sleep 60", "yield_time_ms": 1500}),
    )
    .await;

    assert!(
        output.contains("Command timed out after 1.500 seconds."),
        "{output}"
    );
    assert!(!output.contains("Process exited"), "{output}");
}

/// Python: `test_invalid_model_input_returns_safe_feedback_without_backend_calls`, the
/// `ExecCommandTool` workdir row, and `test_direct_exec_run_still_raises_for_invalid_workdir`.
#[tokio::test]
async fn exec_command_refuses_a_workdir_outside_the_workspace() {
    let session = Arc::new(RecordingSession::new());
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();

    let output = invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd", "workdir": "/outside/private-sentinel"}),
    )
    .await;

    assert!(output.starts_with("Invalid workdir."), "{output}");
    assert!(!output.contains("private-sentinel"), "{output}");
    assert!(session.exec_calls().is_empty(), "no backend call was made");
}

/// Python: `test_invalid_model_input_returns_safe_feedback_without_backend_calls`, the
/// `WriteStdinTool` row.
#[tokio::test]
async fn write_stdin_requires_a_session_id() {
    let session = Arc::new(RecordingSession::new().with_pty(Ok(PtyExecUpdate {
        process_id: Some(1),
        output: Vec::new(),
        exit_code: None,
        original_token_count: None,
    })));
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
    let tool = function_tool(&tools, "write_stdin");
    let arguments = json!({"chars": "private-sentinel"}).to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "write_stdin", &arguments);

    let output = tool
        .invoke(ctx, arguments)
        .await
        .expect("a missing session id is answered, not raised");

    let ToolOutput::Text(text) = output else {
        panic!("text feedback");
    };
    assert!(
        text.starts_with("Invalid tool arguments for session_id."),
        "{text}"
    );
    assert!(!text.contains("private-sentinel"), "{text}");
    assert!(session.exec_calls().is_empty(), "no backend call was made");
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/capabilities/test_filesystem_capability.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_tools_exposes_view_image_and_apply_patch_after_bind`.
#[test]
fn filesystem_tools_expose_apply_patch_and_view_image() {
    let session = Arc::new(RecordingSession::new());
    let tools = Capability::<()>::tools(&Filesystem::new(), &binding(session)).unwrap();

    // Python's order and flavour: `view_image` first, then the freeform custom `apply_patch`.
    assert_eq!(tool_names(&tools), vec!["view_image", "apply_patch"]);
    assert!(matches!(tools[0], Tool::Function(_)));
    assert!(matches!(tools[1], Tool::Custom(_)));

    // The native `apply_patch_call` tool stays available as an opt-in.
    let session = Arc::new(RecordingSession::new());
    let native = Capability::<()>::tools(
        &Filesystem::new().with_native_apply_patch(),
        &binding(session),
    )
    .unwrap();
    assert!(matches!(native[1], Tool::ApplyPatch(_)));
}

/// Python: `test_configure_tools_can_customize_approvals_after_clone` and
/// `test_configure_tools_can_replace_tool_instances`, reduced to the end state Rust configures
/// on the capability.
#[test]
fn filesystem_apply_patch_can_require_approval_and_view_image_can_be_dropped() {
    let session = Arc::new(RecordingSession::new());
    let capability = Filesystem::new()
        .with_apply_patch_needs_approval(true)
        .with_view_image(false);
    let tools = Capability::<()>::tools(&capability, &binding(session)).unwrap();

    assert_eq!(tool_names(&tools), vec!["apply_patch"]);
}

/// Python: `test_configure_tools_can_replace_tool_instances`: `view_image` can be made to require
/// approval (Python replaces the tool with `needs_approval=True`); it needs none by default.
#[test]
fn filesystem_view_image_can_require_approval() {
    use openai_agents_sdk::tool::NeedsApproval;
    let session = Arc::new(RecordingSession::new());
    let default_tools =
        Capability::<()>::tools(&Filesystem::new(), &binding(session.clone())).unwrap();
    assert!(matches!(
        function_tool(&default_tools, "view_image").needs_approval,
        NeedsApproval::Static(false)
    ));

    let capability = Filesystem::new().with_view_image_needs_approval(true);
    let tools = Capability::<()>::tools(&capability, &binding(session)).unwrap();
    assert!(matches!(
        function_tool(&tools, "view_image").needs_approval,
        NeedsApproval::Static(true)
    ));
    assert_eq!(tool_names(&tools), vec!["view_image", "apply_patch"]);
}

/// Python: `test_tools_and_configurator_receive_bound_workspace_scope` and
/// `test_tools_passes_bound_run_as_to_file_tools`: both tools act inside the bound scope.
#[tokio::test]
async fn filesystem_tools_act_inside_the_bound_workspace_scope() {
    let png = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a84QAAAAASUVORK5CYII=",
            )
            .unwrap()
    };
    let session = Arc::new(
        RecordingSession::new().with_file("/workspace/tasks/a/images/plot.png", png.clone()),
    );
    let binding = CapabilityBinding::new(session.clone())
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/a")).expect("scope"));
    let tools = Capability::<()>::tools(&Filesystem::new(), &binding).unwrap();

    let tool = function_tool(&tools, "view_image");
    let arguments = json!({"path": r"images\plot.png"}).to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "view_image", &arguments);
    let output = tool.invoke(ctx, arguments).await.expect("view_image");

    match output {
        ToolOutput::Parts(parts) => assert_eq!(parts.len(), 1),
        other => panic!("expected an image part, got {other:?}"),
    }
}

/// Python: `test_instructions_default_to_none`.
#[tokio::test]
async fn filesystem_instructions_default_to_none() {
    let instructions = Capability::<()>::instructions(
        &Filesystem::new(),
        &Manifest::new().with_root("/workspace"),
    )
    .await
    .unwrap();

    assert_eq!(instructions, None);
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/capabilities/test_compaction_capability.py
// ----------------------------------------------------------------------------------------------

fn sampling_params(capability: &Compaction, model: Option<&str>) -> Value {
    let mut params = serde_json::Map::new();
    if let Some(model) = model {
        params.insert("model".to_string(), Value::String(model.to_string()));
    }
    Value::Object(Capability::<()>::sampling_params(capability, &params))
}

/// Python: `test_sampling_params_uses_static_threshold`.
#[test]
fn compaction_sampling_params_use_a_static_threshold() {
    let capability = Compaction::new().with_policy(CompactionPolicy::Static { threshold: 123 });

    assert_eq!(
        sampling_params(&capability, None),
        json!({"context_management": [{"type": "compaction", "compact_threshold": 123}]})
    );
}

/// Python: `test_sampling_params_infers_hyphenated_model_threshold`.
#[test]
fn compaction_sampling_params_infer_a_hyphenated_model_threshold() {
    assert_eq!(
        sampling_params(&Compaction::new(), Some("gpt-5-2")),
        json!({"context_management": [{"type": "compaction", "compact_threshold": 360_000}]})
    );
}

/// Python: `test_sampling_params_infers_gpt_5_6_sol_dynamic_threshold`.
#[test]
fn compaction_sampling_params_infer_the_gpt_5_6_sol_threshold() {
    assert_eq!(
        sampling_params(&Compaction::new(), Some("gpt-5.6-sol")),
        json!({"context_management": [{"type": "compaction", "compact_threshold": 942_818}]})
    );
}

/// Python: `test_sampling_params_falls_back_for_unknown_model`.
#[test]
fn compaction_sampling_params_fall_back_for_an_unknown_model() {
    assert_eq!(
        sampling_params(&Compaction::new(), Some("azure-prod-deployment")),
        json!({"context_management": [{"type": "compaction", "compact_threshold": 240_000}]})
    );
}

/// Python: `test_process_context_keeps_items_from_last_compaction`.
#[test]
fn compaction_process_context_keeps_items_from_the_last_compaction() {
    let context: Vec<InputItem> = [
        json!({"type": "message", "role": "user", "content": "old-1"}),
        json!({"type": "compaction", "summary": "first"}),
        json!({"type": "message", "role": "assistant", "content": "between"}),
        json!({"type": "compaction", "summary": "second"}),
        json!({"type": "message", "role": "assistant", "content": "latest"}),
    ]
    .into_iter()
    .map(InputItem::from_value)
    .collect();

    let processed = Capability::<()>::process_context(&Compaction::new(), context.clone());

    assert_eq!(processed, context[3..].to_vec());
}

/// Python: `test_process_context_returns_original_when_no_compaction`.
#[test]
fn compaction_process_context_returns_the_original_without_a_compaction_item() {
    let context: Vec<InputItem> = [
        json!({"type": "message", "role": "user", "content": "hello"}),
        json!({"type": "message", "role": "assistant", "content": "world"}),
    ]
    .into_iter()
    .map(InputItem::from_value)
    .collect();

    let processed = Capability::<()>::process_context(&Compaction::new(), context.clone());

    assert_eq!(processed, context);
}

// ----------------------------------------------------------------------------------------------
// Remaining test_shell_capability.py rows
// ----------------------------------------------------------------------------------------------

fn pty_session() -> RecordingSession {
    RecordingSession::new().with_pty(Ok(PtyExecUpdate {
        process_id: Some(1),
        output: Vec::new(),
        exit_code: None,
        original_token_count: None,
    }))
}

fn exec_tool(session: &Arc<RecordingSession>) -> Arc<openai_agents_sdk::tool::FunctionTool<()>> {
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
    function_tool(&tools, "exec_command")
}

fn stdin_tool(session: &Arc<RecordingSession>) -> Arc<openai_agents_sdk::tool::FunctionTool<()>> {
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
    function_tool(&tools, "write_stdin")
}

/// The sections of a tool response after the random chunk id and the wall time.
fn response_body(output: &str) -> Vec<&str> {
    let mut lines = output.split('\n');
    assert!(lines.next().unwrap().starts_with("Chunk ID: "), "{output}");
    assert!(lines.next().unwrap().starts_with("Wall time: "), "{output}");
    lines.collect()
}

/// Python: `test_resolve_shell_uses_plain_sh_when_login_is_false`, observed through the request
/// the tool builds.
#[tokio::test]
async fn exec_command_uses_plain_sh_when_login_is_false() {
    let session = Arc::new(RecordingSession::new());
    invoke(&exec_tool(&session), json!({"cmd": "pwd", "login": false})).await;

    assert_eq!(
        session.exec_calls()[0].shell,
        openai_agents_sdk::sandbox::session::ExecShell::Custom(vec!["sh".into(), "-c".into()])
    );
}

/// Python: `test_exec_command_tool_runs_commands_with_source_output_format`, the default shell:
/// `login` defaults to true, which is the login shell form.
#[tokio::test]
async fn exec_command_defaults_to_the_login_shell() {
    let session = Arc::new(RecordingSession::new());
    invoke(&exec_tool(&session), json!({"cmd": "pwd"})).await;

    assert_eq!(
        session.exec_calls()[0].shell,
        openai_agents_sdk::sandbox::session::ExecShell::Login
    );
}

/// Python: `test_exec_command_tool_uses_stdout_only_when_stderr_is_empty`.
#[tokio::test]
async fn exec_command_uses_stdout_only_when_stderr_is_empty() {
    let session = Arc::new(RecordingSession::new().with_exec_result(ExecResult {
        stdout: b"stdout only\n".to_vec(),
        stderr: Vec::new(),
        exit_code: 7,
    }));

    let output = invoke(&exec_tool(&session), json!({"cmd": "pwd"})).await;

    assert_eq!(
        response_body(&output),
        vec!["Process exited with code 7", "Output:", "stdout only", ""]
    );
}

/// Python: `test_exec_command_tool_uses_stderr_only_when_stdout_is_empty`.
#[tokio::test]
async fn exec_command_uses_stderr_only_when_stdout_is_empty() {
    let session = Arc::new(RecordingSession::new().with_exec_result(ExecResult {
        stdout: Vec::new(),
        stderr: b"stderr only\n".to_vec(),
        exit_code: 7,
    }));

    let output = invoke(&exec_tool(&session), json!({"cmd": "pwd"})).await;

    assert_eq!(
        response_body(&output),
        vec!["Process exited with code 7", "Output:", "stderr only", ""]
    );
}

/// Python: `test_exec_command_tool_does_not_insert_extra_newline_when_stdout_already_has_one`.
#[tokio::test]
async fn exec_command_does_not_insert_an_extra_newline_when_stdout_already_has_one() {
    let session = Arc::new(RecordingSession::new().with_exec_result(ExecResult {
        stdout: b"stdout line\n".to_vec(),
        stderr: b"stderr line\n".to_vec(),
        exit_code: 7,
    }));

    let output = invoke(&exec_tool(&session), json!({"cmd": "pwd"})).await;

    assert_eq!(
        response_body(&output),
        vec![
            "Process exited with code 7",
            "Output:",
            "stdout line",
            "stderr line",
            ""
        ]
    );
}

/// Python: `test_exec_command_tool_formats_timeout_without_exit_code`, with the exact body.
#[tokio::test]
async fn exec_command_formats_a_timeout_without_an_exit_code() {
    let session = Arc::new(
        RecordingSession::new()
            .with_exec_error(SandboxError::new(ErrorCode::ExecTimeout, "timed out")),
    );

    let output = invoke(
        &exec_tool(&session),
        json!({"cmd": "sleep 30", "yield_time_ms": 5}),
    )
    .await;

    assert_eq!(
        response_body(&output),
        vec!["Output:", "Command timed out after 0.005 seconds."]
    );
}

/// Python: `test_exec_command_tool_falls_back_to_one_shot_exec_after_startup_transport_error`
/// with the scope's working directory.
#[tokio::test]
async fn exec_command_fallback_keeps_the_scope_cwd() {
    let session = Arc::new(
        RecordingSession::new()
            .with_pty(Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "connection closed while reading HTTP status line",
            )
            .with_context("stage", "open_pipe")
            .with_context("retry_safe", "true")))
            .with_exec_result(ExecResult {
                stdout: b"fallback ok".to_vec(),
                stderr: Vec::new(),
                exit_code: 0,
            }),
    );
    let binding = CapabilityBinding::new(session.clone())
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/a")).expect("scope"));
    let tools = Capability::<()>::tools(&Shell::new(), &binding).unwrap();

    let output = invoke(
        &function_tool(&tools, "exec_command"),
        json!({"cmd": "pwd"}),
    )
    .await;

    assert!(
        output.contains("PTY transport failed before the interactive session opened"),
        "{output}"
    );
    assert!(output.contains("Process exited with code 0"), "{output}");
    assert!(
        !output.contains("Process running with session ID"),
        "{output}"
    );
    assert!(output.contains("fallback ok"), "{output}");
    assert_eq!(
        session.pty_calls()[0].cwd.as_deref(),
        Some("/workspace/tasks/a")
    );
    assert_eq!(
        session.exec_calls()[0].cwd.as_deref(),
        Some("/workspace/tasks/a")
    );
}

/// Python: `test_write_stdin_tool_writes_and_finishes_session`.
#[tokio::test]
async fn write_stdin_writes_and_finishes_the_session() {
    let session = Arc::new(pty_session().with_stdin(Ok(PtyExecUpdate {
        process_id: None,
        output: b"hello".to_vec(),
        exit_code: Some(0),
        original_token_count: None,
    })));

    let output = invoke(
        &stdin_tool(&session),
        json!({"session_id": 1337, "chars": "hello"}),
    )
    .await;

    assert_eq!(
        response_body(&output),
        vec!["Process exited with code 0", "Output:", "hello"]
    );
}

/// Python: `test_write_stdin_tool_rejects_non_pty_sessions`. The tool is not exposed without a
/// PTY, so the rejection is the capability not offering it; a direct call fails too.
#[tokio::test]
async fn write_stdin_is_unavailable_on_non_pty_sessions() {
    let session = Arc::new(RecordingSession::new());
    let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
    assert!(
        !tool_names(&tools).contains(&"write_stdin".to_string()),
        "{:?}",
        tool_names(&tools)
    );

    let direct =
        openai_agents_sdk::sandbox::capabilities::pty_tools::write_stdin_tool::<()>(session, false);
    let Tool::Function(direct) = direct else {
        panic!("a function tool")
    };
    let arguments = json!({"session_id": 1337}).to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "write_stdin", &arguments);
    direct
        .invoke(ctx, arguments)
        .await
        .expect_err("a non-PTY session cannot write stdin");
}

/// Python: `test_write_stdin_tool_formats_unknown_session_error`.
#[tokio::test]
async fn write_stdin_formats_an_unknown_session_error() {
    let session = Arc::new(pty_session().with_stdin(Err(SandboxError::new(
        ErrorCode::PtySessionNotFound,
        "PTY session not found: 9999",
    ))));

    let output = invoke(&stdin_tool(&session), json!({"session_id": 9999})).await;

    assert_eq!(
        response_body(&output),
        vec![
            "Process exited with code 1",
            "Output:",
            "write_stdin failed: PTY session not found: 9999"
        ]
    );
}

/// Python: `test_write_stdin_tool_formats_missing_stdin_error`.
#[tokio::test]
async fn write_stdin_formats_a_missing_stdin_error() {
    let session = Arc::new(pty_session().with_stdin(Err(SandboxError::new(
        ErrorCode::PtyStdinUnavailable,
        "stdin is not available for this process",
    ))));

    let output = invoke(&stdin_tool(&session), json!({"session_id": 1337})).await;

    assert_eq!(
        response_body(&output),
        vec![
            "Process exited with code 1",
            "Output:",
            "stdin is not available for this process. Start the command with `tty=true` in \
             `exec_command` before using `write_stdin`."
        ]
    );
}

/// Python: `test_write_stdin_tool_reraises_unexpected_runtime_error`.
#[tokio::test]
async fn write_stdin_reraises_an_unexpected_failure() {
    let session = Arc::new(pty_session().with_stdin(Err(SandboxError::new(
        ErrorCode::ExecTransportError,
        "unexpected stdin failure",
    ))));
    let tool = stdin_tool(&session);
    let arguments = json!({"session_id": 1337}).to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "write_stdin", &arguments);

    let error = tool.invoke(ctx, arguments).await.expect_err("propagates");

    assert!(
        error.to_string().contains("unexpected stdin failure"),
        "{error}"
    );
}

/// Python: `test_exec_command_workdir_applies_to_entire_command_list` (6 parameter sets).
///
/// Python prefixes `cd <workdir> || exit` so a command list runs wholly inside the directory or
/// not at all. Rust hands the directory to the backend, which starts the process there, so the
/// same guarantees are checked against a real local session.
#[cfg(unix)]
#[tokio::test]
async fn exec_command_workdir_applies_to_the_entire_command_list() {
    use openai_agents_sdk::sandbox::SandboxClient;
    use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;

    let commands = [
        "true; printf expected > marker",
        "true\nprintf expected > marker",
        "true & wait; printf expected > marker",
    ];
    for directory_exists in [false, true] {
        for command in commands {
            let temp = tempfile::tempdir().unwrap();
            let root = std::fs::canonicalize(temp.path()).unwrap();
            let workdir = root.join("project with spaces");
            if directory_exists {
                std::fs::create_dir(&workdir).unwrap();
            }
            let client = UnixLocalSandboxClient::new();
            let session = client
                .create(Some(Manifest::new().with_root(root.to_string_lossy())))
                .await
                .expect("create session");
            session.start().await.expect("start session");
            let tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
            let tool = function_tool(&tools, "exec_command");
            let arguments = json!({
                "cmd": command, "workdir": "project with spaces", "login": false
            })
            .to_string();
            let ctx = ToolContext::direct(RunContext::new(()), "agent", "exec_command", &arguments);

            let result = tool.invoke(ctx, arguments).await;

            assert!(
                !root.join("marker").exists(),
                "marker leaked outside the workdir (exists={directory_exists}, {command:?}, result={result:?})"
            );
            if directory_exists {
                assert_eq!(
                    std::fs::read_to_string(workdir.join("marker")).unwrap(),
                    "expected"
                );
                let ToolOutput::Text(output) = result.expect("exec_command") else {
                    panic!("text output")
                };
                assert!(output.contains("Process exited with code 0"), "{output}");
            } else {
                // A missing directory fails the call, or at least never reports success.
                if let Ok(ToolOutput::Text(output)) = &result {
                    assert!(!output.contains("Process exited with code 0"), "{output}");
                }
                assert!(!workdir.join("marker").exists());
            }
            client.delete(session).await.ok();
        }
    }
}

// ----------------------------------------------------------------------------------------------
// test_tool_input_errors.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_invalid_model_input_returns_safe_feedback_without_backend_calls` (7 cases x 2
/// run modes). Rust surfaces a rejected argument as a tool failure that the runner turns into a
/// recoverable model message rather than formatting a per-field hint, so the shared guarantees
/// are asserted: the run recovers, the model-visible output never echoes the sentinel, and no
/// backend call is made. The streamed mode shares the tool path and is covered by the same case.
#[tokio::test]
async fn invalid_model_input_returns_safe_feedback_without_backend_calls() {
    use openai_agents_sdk::items::{OutputItem, RunItemKind};
    use openai_agents_sdk::testing::{FakeModel, assistant_message};
    use openai_agents_sdk::{Agent, Runner};

    let cases: Vec<(&str, Value)> = vec![
        (
            "exec_command",
            json!({"cmd": "pwd", "workdir": "/outside/private-sentinel"}),
        ),
        (
            "exec_command",
            json!({"cmd": 5, "max_output_tokens": "private-sentinel"}),
        ),
        ("view_image", json!({"path": "/outside/private-sentinel"})),
        ("view_image", json!({"path": {"private-sentinel": "value"}})),
        ("write_stdin", json!({"chars": "private-sentinel"})),
    ];
    for (name, arguments) in cases {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let session = Arc::new(pty_session());
        let mut tools = Capability::<()>::tools(&Shell::new(), &binding(session.clone())).unwrap();
        tools.extend(
            Capability::<()>::tools(&Filesystem::new(), &binding(session.clone())).unwrap(),
        );
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![OutputItem::from_value(json!({
            "type": "function_call", "call_id": "bad-input", "name": name,
            "arguments": arguments.to_string()
        }))]);
        model.add_turn(vec![assistant_message("recovered")]);
        let mut agent = Agent::<()>::new("test").with_model_instance(model);
        for tool in tools {
            agent = agent.with_tool(tool);
        }

        let result = Runner::run(&agent, "go").await.expect("the run recovers");

        assert_eq!(
            result.final_output_text().as_deref(),
            Some("recovered"),
            "{name}"
        );
        let outputs: Vec<String> = result
            .new_items
            .iter()
            .filter_map(|item| match &item.kind {
                RunItemKind::ToolCallOutput { output, .. } => Some(output.to_string()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 1, "{name}: {outputs:?}");
        assert!(
            !outputs[0].contains("private-sentinel"),
            "{name}: {}",
            outputs[0]
        );
        assert!(session.exec_calls().is_empty(), "{name}: no backend call");
        assert!(session.pty_calls().is_empty(), "{name}: no PTY call");
    }
}

/// Python: `test_exec_backend_failures_are_not_converted_to_argument_feedback`, the `runtime`
/// and `path` rows: a backend failure is not dressed up as a bad-argument hint.
#[tokio::test]
async fn exec_backend_failures_are_not_converted_to_argument_feedback() {
    for failure in [
        SandboxError::new(ErrorCode::ExecTransportError, "backend failure"),
        SandboxError::new(ErrorCode::InvalidManifestPath, "backend-private"),
    ] {
        let message = failure.message.clone();
        let session = Arc::new(RecordingSession::new().with_exec_error(failure));
        let tool = exec_tool(&session);
        let arguments = json!({"cmd": "pwd"}).to_string();
        let ctx = ToolContext::direct(RunContext::new(()), "agent", "exec_command", &arguments);

        let error = tool
            .invoke(ctx, arguments)
            .await
            .expect_err("the failure propagates");

        assert!(error.to_string().contains(&message), "{error}");
        assert_eq!(session.exec_calls().len(), 1);
    }
}

/// Python: `test_direct_exec_run_still_raises_for_invalid_workdir`. A direct call to the
/// tool body with an outside workdir makes no backend call.
#[tokio::test]
async fn direct_exec_with_an_invalid_workdir_makes_no_backend_call() {
    let session = Arc::new(RecordingSession::new());

    let output = invoke(
        &exec_tool(&session),
        json!({"cmd": "pwd", "workdir": "/outside"}),
    )
    .await;

    assert!(output.starts_with("Invalid workdir."), "{output}");
    assert!(session.exec_calls().is_empty());
}
