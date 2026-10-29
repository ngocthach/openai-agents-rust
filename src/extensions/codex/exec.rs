//! Driving the `codex exec --experimental-json` subprocess. Mirrors
//! `agents.extensions.experimental.codex.exec`.

use std::collections::BTreeMap;
use std::process::Stdio;
use std::time::Duration;

use futures::stream::{BoxStream, StreamExt};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdout, Command};
use tokio::task::JoinHandle;

use crate::error::{AgentsError, Result};
use crate::extensions::codex::thread_options::{
    ApprovalMode, ModelReasoningEffort, SandboxMode, WebSearchMode,
};

/// Environment variable the Codex CLI reads to identify its caller.
const INTERNAL_ORIGINATOR_ENV: &str = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE";
const SDK_ORIGINATOR: &str = "codex_sdk_ts";

/// A stream of raw JSONL lines emitted by the Codex CLI.
pub type CodexLineStream = BoxStream<'static, Result<String>>;

/// Everything needed to run one `codex exec` invocation.
#[derive(Debug, Clone, Default)]
pub struct CodexExecArgs {
    /// Prompt written to the Codex CLI on stdin.
    pub input: String,
    /// Override the OpenAI base URL (`OPENAI_BASE_URL`).
    pub base_url: Option<String>,
    /// API key passed as `CODEX_API_KEY`.
    pub api_key: Option<String>,
    /// Thread id to resume.
    pub thread_id: Option<String>,
    /// Local image paths attached to the turn.
    pub images: Vec<String>,
    /// Model identifier (`--model`).
    pub model: Option<String>,
    /// Sandbox mode (`--sandbox`).
    pub sandbox_mode: Option<SandboxMode>,
    /// Working directory (`--cd`).
    pub working_directory: Option<String>,
    /// Extra filesystem roots (`--add-dir`).
    pub additional_directories: Vec<String>,
    /// Allow running outside a Git repository (`--skip-git-repo-check`).
    pub skip_git_repo_check: Option<bool>,
    /// Path of the structured output schema file (`--output-schema`).
    pub output_schema_file: Option<String>,
    /// Reasoning effort config.
    pub model_reasoning_effort: Option<ModelReasoningEffort>,
    /// Abort the turn when no event arrives within this duration.
    pub idle_timeout: Option<Duration>,
    /// Sandboxed network access config.
    pub network_access_enabled: Option<bool>,
    /// Web search config.
    pub web_search_mode: Option<WebSearchMode>,
    /// Legacy web search toggle, used only when `web_search_mode` is unset.
    pub web_search_enabled: Option<bool>,
    /// Approval policy config.
    pub approval_policy: Option<ApprovalMode>,
}

fn push(command_args: &mut Vec<String>, flag: &str, value: String) {
    command_args.push(flag.to_string());
    command_args.push(value);
}

/// Build the `codex` argument vector for one turn.
pub fn build_command_args(args: &CodexExecArgs) -> Vec<String> {
    let mut command_args: Vec<String> = vec!["exec".into(), "--experimental-json".into()];

    if let Some(model) = args.model.as_ref().filter(|v| !v.is_empty()) {
        push(&mut command_args, "--model", model.clone());
    }
    if let Some(sandbox) = args.sandbox_mode {
        push(&mut command_args, "--sandbox", sandbox.as_str().to_string());
    }
    if let Some(dir) = args.working_directory.as_ref().filter(|v| !v.is_empty()) {
        push(&mut command_args, "--cd", dir.clone());
    }
    for directory in &args.additional_directories {
        push(&mut command_args, "--add-dir", directory.clone());
    }
    if args.skip_git_repo_check == Some(true) {
        command_args.push("--skip-git-repo-check".into());
    }
    if let Some(schema) = args.output_schema_file.as_ref().filter(|v| !v.is_empty()) {
        push(&mut command_args, "--output-schema", schema.clone());
    }
    if let Some(effort) = args.model_reasoning_effort {
        push(
            &mut command_args,
            "--config",
            format!("model_reasoning_effort=\"{}\"", effort.as_str()),
        );
    }
    if let Some(enabled) = args.network_access_enabled {
        push(
            &mut command_args,
            "--config",
            format!("sandbox_workspace_write.network_access={enabled}"),
        );
    }
    match (args.web_search_mode, args.web_search_enabled) {
        (Some(mode), _) => push(
            &mut command_args,
            "--config",
            format!("web_search=\"{}\"", mode.as_str()),
        ),
        (None, Some(true)) => push(&mut command_args, "--config", "web_search=\"live\"".into()),
        (None, Some(false)) => push(
            &mut command_args,
            "--config",
            "web_search=\"disabled\"".into(),
        ),
        (None, None) => {}
    }
    if let Some(policy) = args.approval_policy {
        push(
            &mut command_args,
            "--config",
            format!("approval_policy=\"{}\"", policy.as_str()),
        );
    }

    let thread_id = args.thread_id.as_ref().filter(|v| !v.is_empty());
    if thread_id.is_some() {
        command_args.push("resume".into());
    }
    for image in &args.images {
        push(&mut command_args, "--image", image.clone());
    }
    if let Some(thread_id) = thread_id {
        command_args.push("--".into());
        command_args.push(thread_id.clone());
    }
    // The Codex CLI expects a prompt argument; "-" tells it to read from stdin.
    command_args.push("-".into());
    command_args
}

/// Resolve the `codex` executable: `CODEX_PATH`, then a `PATH` lookup, then the bare name.
pub fn find_codex_path() -> String {
    if let Ok(path) = std::env::var("CODEX_PATH")
        && !path.is_empty()
    {
        return path;
    }
    let binary = if cfg!(windows) { "codex.exe" } else { "codex" };
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join(binary);
            if candidate.is_file() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    binary.to_string()
}

/// Runs the Codex CLI and streams its JSONL output.
#[derive(Debug, Clone)]
pub struct CodexExec {
    executable_path: String,
    env_override: Option<BTreeMap<String, String>>,
}

impl CodexExec {
    /// Create a runner for the given executable (resolved when `None`) and environment.
    pub fn new(
        executable_path: Option<String>,
        env_override: Option<BTreeMap<String, String>>,
    ) -> Self {
        CodexExec {
            executable_path: executable_path
                .filter(|p| !p.is_empty())
                .unwrap_or_else(find_codex_path),
            env_override,
        }
    }

    /// The resolved path of the `codex` binary.
    pub fn executable_path(&self) -> &str {
        &self.executable_path
    }

    /// The environment of the Codex CLI process for these arguments.
    pub fn build_env(&self, args: &CodexExecArgs) -> BTreeMap<String, String> {
        let mut env: BTreeMap<String, String> = match &self.env_override {
            Some(override_env) => override_env.clone(),
            None => std::env::vars().collect(),
        };
        env.entry(INTERNAL_ORIGINATOR_ENV.to_string())
            .or_insert_with(|| SDK_ORIGINATOR.to_string());
        if let Some(base_url) = args.base_url.as_ref().filter(|v| !v.is_empty()) {
            env.insert("OPENAI_BASE_URL".into(), base_url.clone());
        }
        if let Some(api_key) = args.api_key.as_ref().filter(|v| !v.is_empty()) {
            env.insert("CODEX_API_KEY".into(), api_key.clone());
        }
        env
    }

    /// Spawn `codex exec --experimental-json` and stream one JSONL event per line.
    ///
    /// The last stream item is an error when the process exits with a non-zero status. Dropping
    /// the stream kills the process.
    pub async fn run(&self, args: CodexExecArgs) -> Result<CodexLineStream> {
        let command_args = build_command_args(&args);
        let env = self.build_env(&args);

        let mut child = Command::new(&self.executable_path)
            .args(&command_args)
            .env_clear()
            .envs(&env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                AgentsError::other(format!(
                    "Failed to start the Codex CLI at `{}`: {e}",
                    self.executable_path
                ))
            })?;

        // Drain stderr before writing the prompt, so a process that logs while reading stdin
        // cannot fill its stderr pipe and block.
        let stderr_task = child.stderr.take().map(|mut stderr| {
            tokio::spawn(async move {
                let mut buffer = Vec::new();
                let _ = stderr.read_to_end(&mut buffer).await;
                buffer
            })
        });

        let Some(mut stdin) = child.stdin.take() else {
            let _ = child.start_kill();
            return Err(AgentsError::other("Codex subprocess has no stdin"));
        };
        // A process that exits before reading the prompt closes the pipe; the exit status below
        // is the useful error in that case.
        if let Err(e) = stdin.write_all(args.input.as_bytes()).await
            && e.kind() != std::io::ErrorKind::BrokenPipe
        {
            let _ = child.start_kill();
            return Err(AgentsError::Io(e));
        }
        let _ = stdin.shutdown().await;
        drop(stdin);

        let Some(stdout) = child.stdout.take() else {
            let _ = child.start_kill();
            return Err(AgentsError::other("Codex subprocess has no stdout"));
        };

        let state = StreamState {
            child,
            lines: BufReader::new(stdout).lines(),
            stderr_task,
            idle_timeout: args.idle_timeout,
            done: false,
        };
        Ok(futures::stream::unfold(state, next_line).boxed())
    }
}

struct StreamState {
    child: Child,
    lines: Lines<BufReader<ChildStdout>>,
    stderr_task: Option<JoinHandle<Vec<u8>>>,
    idle_timeout: Option<Duration>,
    done: bool,
}

async fn next_line(mut state: StreamState) -> Option<(Result<String>, StreamState)> {
    if state.done {
        return None;
    }
    let line = match state.idle_timeout {
        Some(timeout) => match tokio::time::timeout(timeout, state.lines.next_line()).await {
            Ok(result) => result,
            Err(_) => {
                state.done = true;
                let _ = state.child.start_kill();
                let message = format!("Codex stream idle for {} seconds.", timeout.as_secs_f64());
                return Some((Err(AgentsError::other(message)), state));
            }
        },
        None => state.lines.next_line().await,
    };
    match line {
        Ok(Some(line)) => Some((Ok(line), state)),
        Ok(None) => {
            state.done = true;
            exit_error(&mut state)
                .await
                .map(|error| (Err(error), state))
        }
        Err(e) => {
            state.done = true;
            Some((Err(AgentsError::Io(e)), state))
        }
    }
}

async fn exit_error(state: &mut StreamState) -> Option<AgentsError> {
    let status = match state.child.wait().await {
        Ok(status) => status,
        Err(e) => return Some(AgentsError::Io(e)),
    };
    if status.success() {
        return None;
    }
    let stderr = match state.stderr_task.take() {
        Some(task) => task.await.unwrap_or_default(),
        None => Vec::new(),
    };
    let stderr_text = String::from_utf8_lossy(&stderr);
    Some(AgentsError::other(format!(
        "Codex exec exited with code {}: {stderr_text}",
        status.code().unwrap_or(-1)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_minimal_args() {
        let args = CodexExecArgs {
            input: "hello".into(),
            ..Default::default()
        };
        assert_eq!(
            build_command_args(&args),
            vec!["exec", "--experimental-json", "-"]
        );
    }

    #[test]
    fn builds_full_args_in_cli_order() {
        let args = CodexExecArgs {
            input: "hello".into(),
            model: Some("gpt-5-codex".into()),
            sandbox_mode: Some(SandboxMode::WorkspaceWrite),
            working_directory: Some("/repo".into()),
            additional_directories: vec!["/extra".into(), "/more".into()],
            skip_git_repo_check: Some(true),
            output_schema_file: Some("/tmp/schema.json".into()),
            model_reasoning_effort: Some(ModelReasoningEffort::High),
            network_access_enabled: Some(false),
            web_search_mode: Some(WebSearchMode::Cached),
            web_search_enabled: Some(true),
            approval_policy: Some(ApprovalMode::OnFailure),
            ..Default::default()
        };
        assert_eq!(
            build_command_args(&args),
            vec![
                "exec",
                "--experimental-json",
                "--model",
                "gpt-5-codex",
                "--sandbox",
                "workspace-write",
                "--cd",
                "/repo",
                "--add-dir",
                "/extra",
                "--add-dir",
                "/more",
                "--skip-git-repo-check",
                "--output-schema",
                "/tmp/schema.json",
                "--config",
                "model_reasoning_effort=\"high\"",
                "--config",
                "sandbox_workspace_write.network_access=false",
                "--config",
                "web_search=\"cached\"",
                "--config",
                "approval_policy=\"on-failure\"",
                "-",
            ]
        );
    }

    #[test]
    fn resume_places_thread_id_after_images() {
        let args = CodexExecArgs {
            input: "hello".into(),
            thread_id: Some("thread-1".into()),
            images: vec!["/a.png".into(), "/b.png".into()],
            ..Default::default()
        };
        assert_eq!(
            build_command_args(&args),
            vec![
                "exec",
                "--experimental-json",
                "resume",
                "--image",
                "/a.png",
                "--image",
                "/b.png",
                "--",
                "thread-1",
                "-",
            ]
        );
    }

    #[test]
    fn legacy_web_search_toggle_maps_to_config() {
        let enabled = CodexExecArgs {
            web_search_enabled: Some(true),
            ..Default::default()
        };
        assert!(build_command_args(&enabled).contains(&"web_search=\"live\"".to_string()));
        let disabled = CodexExecArgs {
            web_search_enabled: Some(false),
            ..Default::default()
        };
        assert!(build_command_args(&disabled).contains(&"web_search=\"disabled\"".to_string()));
    }

    #[test]
    fn env_override_replaces_the_inherited_environment() {
        let exec = CodexExec::new(
            Some("/bin/true".into()),
            Some(BTreeMap::from([("HOME".to_string(), "/tmp".to_string())])),
        );
        let env = exec.build_env(&CodexExecArgs {
            base_url: Some("https://example.test/v1".into()),
            api_key: Some("sk-test".into()),
            ..Default::default()
        });
        assert_eq!(env.get("HOME").map(String::as_str), Some("/tmp"));
        assert_eq!(
            env.get("OPENAI_BASE_URL").map(String::as_str),
            Some("https://example.test/v1")
        );
        assert_eq!(
            env.get("CODEX_API_KEY").map(String::as_str),
            Some("sk-test")
        );
        assert_eq!(
            env.get(INTERNAL_ORIGINATOR_ENV).map(String::as_str),
            Some(SDK_ORIGINATOR)
        );
        assert!(!env.contains_key("PATH"));
    }

    #[test]
    fn env_override_keeps_a_custom_originator() {
        let exec = CodexExec::new(
            None,
            Some(BTreeMap::from([(
                INTERNAL_ORIGINATOR_ENV.to_string(),
                "custom".to_string(),
            )])),
        );
        let env = exec.build_env(&CodexExecArgs::default());
        assert_eq!(
            env.get(INTERNAL_ORIGINATOR_ENV).map(String::as_str),
            Some("custom")
        );
    }
}
