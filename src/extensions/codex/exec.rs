//! Driving the `codex exec --experimental-json` subprocess. Mirrors
//! `agents.extensions.experimental.codex.exec`.

use std::collections::BTreeMap;
use std::process::Stdio;
use std::time::Duration;

use futures::stream::{BoxStream, StreamExt};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdout, Command};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::error::{AgentsError, Result};
use crate::extensions::codex::thread_options::{
    ApprovalMode, ModelReasoningEffort, SandboxMode, WebSearchMode,
};

/// Environment variable the Codex CLI reads to identify its caller.
const INTERNAL_ORIGINATOR_ENV: &str = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE";
const SDK_ORIGINATOR: &str = "codex_sdk_ts";

/// Environment variable overriding the per-line read limit of the Codex subprocess.
pub const SUBPROCESS_STREAM_LIMIT_ENV: &str = "OPENAI_AGENTS_CODEX_SUBPROCESS_STREAM_LIMIT_BYTES";
/// Default per-line read limit of the Codex subprocess (8 MiB).
pub const DEFAULT_SUBPROCESS_STREAM_LIMIT_BYTES: usize = 8 * 1024 * 1024;
/// Smallest accepted per-line read limit (64 KiB).
pub const MIN_SUBPROCESS_STREAM_LIMIT_BYTES: usize = 64 * 1024;
/// Largest accepted per-line read limit (64 MiB).
pub const MAX_SUBPROCESS_STREAM_LIMIT_BYTES: usize = 64 * 1024 * 1024;

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
    /// Kill the Codex CLI when this token is cancelled. Port of `CodexExecArgs.signal`.
    pub cancel: Option<CancellationToken>,
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

/// `true` when the path is an executable file, like `shutil.which`'s `os.access(…, X_OK)` check.
fn is_executable_file(path: &std::path::Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .map(|meta| meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Resolve the `codex` executable: `CODEX_PATH`, then a `PATH` lookup, then the bare name.
///
/// Python falls back to a binary vendored in the wheel
/// (`vendor/<target-triple>/codex/codex`); this crate ships no vendored binaries, so the bare
/// name is returned instead and the spawn error names the missing executable.
pub fn find_codex_path() -> String {
    resolve_codex_path(
        std::env::var("CODEX_PATH").ok().as_deref(),
        std::env::var_os("PATH").as_deref(),
    )
}

/// [`find_codex_path`] with the environment passed in, so it can be tested in parallel.
fn resolve_codex_path(codex_path: Option<&str>, search_path: Option<&std::ffi::OsStr>) -> String {
    if let Some(path) = codex_path.filter(|path| !path.is_empty()) {
        return path.to_string();
    }
    let binary = if cfg!(windows) { "codex.exe" } else { "codex" };
    if let Some(paths) = search_path {
        for dir in std::env::split_paths(paths) {
            let candidate = dir.join(binary);
            if is_executable_file(&candidate) {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    binary.to_string()
}

/// Validate a per-line read limit. Port of `_validate_subprocess_stream_limit_bytes`.
fn validate_subprocess_stream_limit_bytes(value: usize) -> Result<usize> {
    if !(MIN_SUBPROCESS_STREAM_LIMIT_BYTES..=MAX_SUBPROCESS_STREAM_LIMIT_BYTES).contains(&value) {
        return Err(AgentsError::user(format!(
            "codex_subprocess_stream_limit_bytes must be between \
             {MIN_SUBPROCESS_STREAM_LIMIT_BYTES} and {MAX_SUBPROCESS_STREAM_LIMIT_BYTES} bytes."
        )));
    }
    Ok(value)
}

/// Resolve the per-line read limit from the explicit value, then `env_value`, then the default.
/// Port of `_resolve_subprocess_stream_limit_bytes`.
pub(crate) fn resolve_subprocess_stream_limit_bytes(
    explicit: Option<usize>,
    env_value: Option<&str>,
) -> Result<usize> {
    if let Some(value) = explicit {
        return validate_subprocess_stream_limit_bytes(value);
    }
    let Some(env_value) = env_value else {
        return Ok(DEFAULT_SUBPROCESS_STREAM_LIMIT_BYTES);
    };
    let parsed: i64 = env_value.trim().parse().map_err(|_| {
        AgentsError::user(format!(
            "{SUBPROCESS_STREAM_LIMIT_ENV} must be an integer number of bytes."
        ))
    })?;
    // A negative value parses in Python and is rejected by the range check.
    validate_subprocess_stream_limit_bytes(usize::try_from(parsed).unwrap_or(0))
}

/// Runs the Codex CLI and streams its JSONL output.
#[derive(Debug, Clone)]
pub struct CodexExec {
    executable_path: String,
    env_override: Option<BTreeMap<String, String>>,
    subprocess_stream_limit_bytes: Option<usize>,
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
            subprocess_stream_limit_bytes: None,
        }
    }

    /// Cap how many bytes one JSONL line may occupy; `None` uses
    /// [`SUBPROCESS_STREAM_LIMIT_ENV`] and then
    /// [`DEFAULT_SUBPROCESS_STREAM_LIMIT_BYTES`]. The value is validated when a turn starts.
    pub fn with_subprocess_stream_limit_bytes(mut self, limit: Option<usize>) -> Self {
        self.subprocess_stream_limit_bytes = limit;
        self
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
        let stream_limit = resolve_subprocess_stream_limit_bytes(
            self.subprocess_stream_limit_bytes,
            std::env::var(SUBPROCESS_STREAM_LIMIT_ENV).ok().as_deref(),
        )?;

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
            reader: LineReader {
                reader: BufReader::new(stdout),
                limit: stream_limit,
            },
            stderr_task,
            cancel: args.cancel.clone(),
            idle_timeout: args.idle_timeout,
            done: false,
        };
        Ok(futures::stream::unfold(state, next_line).boxed())
    }
}

/// Reads one JSONL line at a time, refusing lines longer than `limit` bytes the way asyncio's
/// `StreamReader.readline` refuses a line longer than its buffer limit.
struct LineReader<R> {
    reader: BufReader<R>,
    limit: usize,
}

fn stream_limit_error(reason: &str, limit: usize) -> AgentsError {
    AgentsError::other(format!(
        "Codex stdout exceeded the {limit} byte subprocess stream limit: {reason}"
    ))
}

enum Chunk {
    Eof,
    Separator(usize),
    More(usize),
}

impl<R: tokio::io::AsyncRead + Unpin> LineReader<R> {
    /// The next line without its trailing newline, or `None` at end of stream.
    async fn next_line(&mut self) -> Result<Option<String>> {
        let mut buffer: Vec<u8> = Vec::new();
        loop {
            let chunk = {
                let available = self.reader.fill_buf().await.map_err(AgentsError::Io)?;
                if available.is_empty() {
                    Chunk::Eof
                } else if let Some(index) = available.iter().position(|byte| *byte == b'\n') {
                    buffer.extend_from_slice(&available[..index]);
                    Chunk::Separator(index + 1)
                } else {
                    buffer.extend_from_slice(available);
                    Chunk::More(available.len())
                }
            };
            match chunk {
                Chunk::Eof => {
                    if buffer.is_empty() {
                        return Ok(None);
                    }
                    break;
                }
                Chunk::Separator(consumed) => {
                    self.reader.consume(consumed);
                    if buffer.len() > self.limit {
                        return Err(stream_limit_error(
                            "Separator is found, but chunk is longer than limit",
                            self.limit,
                        ));
                    }
                    break;
                }
                Chunk::More(consumed) => {
                    self.reader.consume(consumed);
                    if buffer.len() > self.limit {
                        return Err(stream_limit_error(
                            "Separator is not found, and chunk exceed the limit",
                            self.limit,
                        ));
                    }
                }
            }
        }
        // Python decodes then `rstrip("\n")`, so a trailing carriage return is kept.
        Ok(Some(String::from_utf8_lossy(&buffer).into_owned()))
    }
}

struct StreamState {
    child: Child,
    reader: LineReader<ChildStdout>,
    stderr_task: Option<JoinHandle<Vec<u8>>>,
    cancel: Option<CancellationToken>,
    idle_timeout: Option<Duration>,
    done: bool,
}

enum ReadOutcome {
    Line(String),
    Eof,
    Error(AgentsError),
    Cancelled,
    IdleTimeout,
}

fn to_outcome(result: Result<Option<String>>) -> ReadOutcome {
    match result {
        Ok(Some(line)) => ReadOutcome::Line(line),
        Ok(None) => ReadOutcome::Eof,
        Err(e) => ReadOutcome::Error(e),
    }
}

async fn read_outcome(
    reader: &mut LineReader<ChildStdout>,
    cancel: Option<&CancellationToken>,
    idle_timeout: Option<Duration>,
) -> ReadOutcome {
    let read = reader.next_line();
    tokio::pin!(read);
    match (cancel, idle_timeout) {
        (Some(token), Some(timeout)) => tokio::select! {
            biased;
            _ = token.cancelled() => ReadOutcome::Cancelled,
            result = &mut read => to_outcome(result),
            _ = tokio::time::sleep(timeout) => ReadOutcome::IdleTimeout,
        },
        (Some(token), None) => tokio::select! {
            biased;
            _ = token.cancelled() => ReadOutcome::Cancelled,
            result = &mut read => to_outcome(result),
        },
        (None, Some(timeout)) => tokio::select! {
            biased;
            result = &mut read => to_outcome(result),
            _ = tokio::time::sleep(timeout) => ReadOutcome::IdleTimeout,
        },
        (None, None) => to_outcome(read.await),
    }
}

async fn next_line(mut state: StreamState) -> Option<(Result<String>, StreamState)> {
    if state.done {
        return None;
    }
    let outcome = read_outcome(&mut state.reader, state.cancel.as_ref(), state.idle_timeout).await;
    match outcome {
        ReadOutcome::Line(line) => Some((Ok(line), state)),
        ReadOutcome::Error(e) => {
            state.done = true;
            Some((Err(e), state))
        }
        ReadOutcome::Eof => {
            state.done = true;
            exit_error(&mut state)
                .await
                .map(|error| (Err(error), state))
        }
        ReadOutcome::Cancelled => {
            // Python terminates the subprocess, so the turn ends with its non-zero exit status.
            state.done = true;
            let _ = state.child.start_kill();
            let error = exit_error(&mut state)
                .await
                .unwrap_or_else(|| AgentsError::other("Codex exec was cancelled."));
            Some((Err(error), state))
        }
        ReadOutcome::IdleTimeout => {
            state.done = true;
            if let Some(token) = &state.cancel {
                token.cancel();
            }
            let _ = state.child.start_kill();
            let timeout = state.idle_timeout.unwrap_or_default();
            let message = format!("Codex stream idle for {} seconds.", timeout.as_secs_f64());
            Some((Err(AgentsError::other(message)), state))
        }
    }
}

/// The exit code of a finished process, negated for a signal like Python's `returncode`.
fn exit_code(status: &std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return -signal;
        }
    }
    -1
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
        exit_code(&status)
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
    fn resolves_the_codex_path_like_shutil_which() {
        let dir = tempfile::tempdir().expect("temp dir");
        let not_executable = dir.path().join("skipped");
        let executable = dir.path().join("found");
        std::fs::create_dir_all(&not_executable).expect("dir");
        std::fs::create_dir_all(&executable).expect("dir");
        std::fs::write(not_executable.join("codex"), "#!/bin/sh\n").expect("write");
        std::fs::write(executable.join("codex"), "#!/bin/sh\n").expect("write");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                not_executable.join("codex"),
                std::fs::Permissions::from_mode(0o644),
            )
            .expect("chmod");
            std::fs::set_permissions(
                executable.join("codex"),
                std::fs::Permissions::from_mode(0o755),
            )
            .expect("chmod");
        }
        let search_path =
            std::env::join_paths([not_executable.as_path(), executable.as_path()]).expect("join");

        // `CODEX_PATH` wins over the search path.
        assert_eq!(
            resolve_codex_path(Some("/custom/codex"), Some(search_path.as_os_str())),
            "/custom/codex"
        );
        // An empty override is ignored; a non-executable candidate is skipped on unix.
        let resolved = resolve_codex_path(Some(""), Some(search_path.as_os_str()));
        #[cfg(unix)]
        assert_eq!(resolved, executable.join("codex").to_string_lossy());
        #[cfg(not(unix))]
        assert_eq!(resolved, not_executable.join("codex").to_string_lossy());
        // No vendored binary to fall back to, so the bare name is returned.
        assert_eq!(resolve_codex_path(None, None), "codex");
    }

    #[test]
    fn resolves_the_subprocess_stream_limit() {
        assert_eq!(
            resolve_subprocess_stream_limit_bytes(None, None).unwrap(),
            DEFAULT_SUBPROCESS_STREAM_LIMIT_BYTES
        );
        assert_eq!(
            resolve_subprocess_stream_limit_bytes(Some(65536), Some("131072")).unwrap(),
            65536
        );
        assert_eq!(
            resolve_subprocess_stream_limit_bytes(None, Some(" 131072 ")).unwrap(),
            131072
        );
        let not_an_int = resolve_subprocess_stream_limit_bytes(None, Some("many")).unwrap_err();
        assert_eq!(
            not_an_int.to_string(),
            format!("{SUBPROCESS_STREAM_LIMIT_ENV} must be an integer number of bytes.")
        );
        for out_of_range in [Some(1024), Some(MAX_SUBPROCESS_STREAM_LIMIT_BYTES + 1)] {
            let error = resolve_subprocess_stream_limit_bytes(out_of_range, None).unwrap_err();
            assert_eq!(
                error.to_string(),
                "codex_subprocess_stream_limit_bytes must be between 65536 and 67108864 bytes."
            );
        }
        let negative = resolve_subprocess_stream_limit_bytes(None, Some("-1")).unwrap_err();
        assert!(error_mentions_range(&negative));
    }

    fn error_mentions_range(error: &AgentsError) -> bool {
        error.to_string().contains("must be between")
    }

    fn line_reader(input: &'static str, limit: usize) -> LineReader<&'static [u8]> {
        LineReader {
            reader: BufReader::with_capacity(16, input.as_bytes()),
            limit,
        }
    }

    #[tokio::test]
    async fn the_line_reader_splits_on_newlines_and_keeps_carriage_returns() {
        let mut reader = line_reader("first\r\nsecond\nno trailing newline", 1024);
        assert_eq!(
            reader.next_line().await.unwrap().as_deref(),
            Some("first\r")
        );
        assert_eq!(reader.next_line().await.unwrap().as_deref(), Some("second"));
        assert_eq!(
            reader.next_line().await.unwrap().as_deref(),
            Some("no trailing newline")
        );
        assert_eq!(reader.next_line().await.unwrap(), None);
    }

    #[tokio::test]
    async fn the_line_reader_rejects_lines_over_the_limit() {
        let mut reader = line_reader(
            "short\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
            32,
        );
        assert_eq!(reader.next_line().await.unwrap().as_deref(), Some("short"));
        let error = reader.next_line().await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Codex stdout exceeded the 32 byte subprocess stream limit: Separator is not found, \
             and chunk exceed the limit"
        );
    }

    #[tokio::test]
    async fn the_line_reader_rejects_a_long_line_it_read_whole() {
        let input = format!("{}\n", "y".repeat(40));
        let mut reader = LineReader {
            reader: BufReader::with_capacity(256, input.as_bytes()),
            limit: 32,
        };
        let error = reader.next_line().await.unwrap_err();
        assert!(
            error
                .to_string()
                .ends_with("Separator is found, but chunk is longer than limit"),
            "unexpected error: {error}"
        );
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
