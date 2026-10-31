//! The PTY-backed `exec_command` and `write_stdin` tools. Mirrors
//! `agents.sandbox.capabilities.tools.shell_tool`.

use crate::sandbox::capabilities::tool_input_errors::{FieldKind, FieldSpec, tool_with_safe_input};
use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;

use std::time::Duration;

use crate::error::AgentsError;
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::pty::{
    DEFAULT_EXEC_YIELD_TIME_MS, DEFAULT_WRITE_STDIN_YIELD_TIME_MS, PtyExecRequest, PtyExecUpdate,
    PtyWriteRequest,
};
use crate::sandbox::session::{ExecRequest, ExecShell, SandboxSession};
use crate::sandbox::types::User;
use crate::sandbox::util::formatted_truncate_text_with_token_count;
use crate::sandbox::workspace_paths::SandboxWorkspaceScope;
use crate::tool::Tool;

/// The name of the PTY command tool.
pub const EXEC_COMMAND_TOOL_NAME: &str = "exec_command";
/// The name of the PTY stdin tool.
pub const WRITE_STDIN_TOOL_NAME: &str = "write_stdin";

const EXEC_COMMAND_DESCRIPTION: &str =
    "Runs a command in a PTY, returning output or a session ID for ongoing interaction.";
const WRITE_STDIN_DESCRIPTION: &str =
    "Writes characters to an existing unified exec session and returns recent output.";
const TOOL_OUTPUT_HEADER: &str = "Output:";

fn default_exec_yield_time_ms() -> u64 {
    DEFAULT_EXEC_YIELD_TIME_MS
}

fn default_write_stdin_yield_time_ms() -> u64 {
    DEFAULT_WRITE_STDIN_YIELD_TIME_MS
}

fn default_login() -> bool {
    true
}

/// Arguments of the `exec_command` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ExecCommandArgs {
    /// Shell command to execute.
    pub cmd: String,
    /// Optional working directory to run the command in; defaults to the turn cwd.
    #[serde(default)]
    pub workdir: Option<String>,
    /// Shell binary to launch. Defaults to the user's default shell.
    #[serde(default)]
    pub shell: Option<String>,
    /// Whether to run the shell with `-l`/`-i` semantics. Defaults to true.
    #[serde(default = "default_login")]
    pub login: bool,
    /// Whether to allocate a TTY for the command. Defaults to false (plain pipes); set to true
    /// to open a PTY and access the TTY process.
    #[serde(default)]
    pub tty: bool,
    /// How long to wait (in milliseconds) for output before yielding.
    #[serde(default = "default_exec_yield_time_ms")]
    pub yield_time_ms: u64,
    /// Maximum number of tokens to return. Excess output will be truncated.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
}

/// Arguments of the `write_stdin` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct WriteStdinArgs {
    /// Identifier of the running unified exec session.
    pub session_id: u32,
    /// Bytes to write to stdin (may be empty to poll).
    #[serde(default)]
    pub chars: String,
    /// How long to wait (in milliseconds) for output before yielding.
    #[serde(default = "default_write_stdin_yield_time_ms")]
    pub yield_time_ms: u64,
    /// Maximum number of tokens to return. Excess output will be truncated.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
}

const EXEC_COMMAND_FIELDS: &[FieldSpec] = &[
    ("cmd", FieldKind::Str { non_empty: true }),
    ("workdir", FieldKind::OptStr),
    ("shell", FieldKind::OptStr),
    ("login", FieldKind::Bool),
    ("tty", FieldKind::Bool),
    ("yield_time_ms", FieldKind::DefaultedInt { min: 0 }),
    ("max_output_tokens", FieldKind::OptInt { min: 1 }),
];

const WRITE_STDIN_FIELDS: &[FieldSpec] = &[
    ("session_id", FieldKind::Int { min: 0 }),
    ("chars", FieldKind::DefaultedStr),
    ("yield_time_ms", FieldKind::DefaultedInt { min: 0 }),
    ("max_output_tokens", FieldKind::OptInt { min: 1 }),
];

/// Resolve the shell prefix the way Python's `_resolve_shell` does.
fn resolve_shell(shell: Option<&str>, login: bool) -> ExecShell {
    match shell {
        None if login => ExecShell::Login,
        None => ExecShell::Custom(vec!["sh".into(), "-c".into()]),
        Some(shell) => ExecShell::Custom(vec![
            shell.to_string(),
            if login { "-lc" } else { "-c" }.to_string(),
        ]),
    }
}

/// Join a command's two streams the way Python's `_normalize_output` does.
///
/// A command that wrote to both streams gets them concatenated, with a newline inserted only
/// when stdout did not already end in one, so interleaved output does not run two lines
/// together. A command that wrote to one stream gets that stream unchanged.
fn normalize_output(stdout: &str, stderr: &str) -> String {
    if !stdout.is_empty() && !stderr.is_empty() {
        let joiner = if stdout.ends_with('\n') { "" } else { "\n" };
        return format!("{stdout}{joiner}{stderr}");
    }
    if stdout.is_empty() {
        stderr.to_string()
    } else {
        stdout.to_string()
    }
}

/// Put `notice` above `output`, or use it alone when there was no output.
fn prepend_notice(output: &str, notice: &str) -> String {
    if output.is_empty() {
        notice.to_string()
    } else {
        format!("{notice}\n{output}")
    }
}

/// Whether a PTY transport failure may be retried as a plain one-shot command.
///
/// Mirrors Python's `_supports_transport_fallback`: only a transport that says the command
/// never started can be re-run, because re-running one that did would repeat its side effects.
fn supports_transport_fallback(error: &SandboxError) -> bool {
    error.context.get("retry_safe").map(String::as_str) == Some("true")
}

/// The result of one command run outside a PTY: output, exit code and pre-truncation tokens.
async fn run_one_shot_exec(
    session: &dyn SandboxSession,
    command: String,
    timeout: Option<Duration>,
    shell: ExecShell,
    cwd: Option<String>,
    user: Option<User>,
    max_output_tokens: Option<usize>,
) -> SandboxResult<(String, Option<i32>, Option<usize>)> {
    let request = ExecRequest::shell(command)
        .with_shell(shell)
        .with_timeout(timeout)
        .with_cwd(cwd)
        .with_user(user);
    let result = session.exec(request).await?;
    let output = normalize_output(&result.stdout_text(), &result.stderr_text());
    let (output, original_token_count) =
        formatted_truncate_text_with_token_count(&output, max_output_tokens);
    Ok((output, Some(result.exit_code), original_token_count))
}

/// Format a PTY result the way the Python tool does.
fn format_response(
    output: &str,
    wall_time_seconds: f64,
    exit_code: Option<i32>,
    process_id: Option<u32>,
    original_token_count: Option<usize>,
) -> String {
    let mut sections = vec![
        format!(
            "Chunk ID: {}",
            &uuid::Uuid::new_v4().simple().to_string()[..6]
        ),
        format!("Wall time: {wall_time_seconds:.4} seconds"),
    ];
    if let Some(exit_code) = exit_code {
        sections.push(format!("Process exited with code {exit_code}"));
    }
    if let Some(process_id) = process_id {
        sections.push(format!("Process running with session ID {process_id}"));
    }
    if let Some(original_token_count) = original_token_count {
        sections.push(format!("Original token count: {original_token_count}"));
    }
    sections.push(TOOL_OUTPUT_HEADER.to_string());
    sections.push(output.to_string());
    sections.join("\n")
}

fn response_from_update(update: &PtyExecUpdate, elapsed: f64) -> String {
    format_response(
        &update.output_text(),
        elapsed,
        update.exit_code,
        update.process_id,
        update.original_token_count,
    )
}

/// Build the `exec_command` tool bound to a session.
pub fn exec_command_tool<C: Send + Sync + 'static>(
    session: Arc<dyn SandboxSession>,
    run_as: Option<User>,
    scope: SandboxWorkspaceScope,
    needs_approval: bool,
) -> Tool<C> {
    let tool = tool_with_safe_input::<C, ExecCommandArgs, _, _, _>(
        EXEC_COMMAND_TOOL_NAME,
        EXEC_COMMAND_DESCRIPTION,
        EXEC_COMMAND_FIELDS,
        move |_ctx, args: ExecCommandArgs| {
            let session = session.clone();
            let run_as = run_as.clone();
            let scope = scope.clone();
            async move {
                // A snapshot detaches the workspace's mounts; a process started now would see
                // empty directories where mounted storage should be.
                session.ensure_mounts_settled()?;
                let started = std::time::Instant::now();
                let requested_workdir = match args.workdir.as_deref().map(str::trim) {
                    None | Some("") => scope.cwd().map(str::to_string),
                    Some(workdir) => Some(scope.anchor(workdir)),
                };
                // The backend receives the resolved absolute sandbox path, not the relative
                // spelling the model used: a backend that resolves a relative directory against
                // its own process cwd would run the command somewhere else entirely.
                let workdir = match &requested_workdir {
                    Some(workdir) => match session.normalize_path(workdir) {
                        Ok(resolved) => Some(resolved),
                        Err(_) => {
                            return Ok::<_, AgentsError>(
                                "Invalid workdir. Use a path within the workspace or an \
                                 explicitly granted path."
                                    .to_string(),
                            );
                        }
                    },
                    None => None,
                };
                let shell = resolve_shell(args.shell.as_deref(), args.login);
                let timeout = Duration::from_millis(args.yield_time_ms);
                let one_shot = async |notice: Option<&str>| -> Result<String, SandboxError> {
                    let (output, exit_code, original_token_count) = run_one_shot_exec(
                        session.as_ref(),
                        args.cmd.clone(),
                        Some(timeout),
                        shell.clone(),
                        workdir.clone(),
                        run_as.clone(),
                        args.max_output_tokens,
                    )
                    .await?;
                    let output = match notice {
                        Some(notice) => prepend_notice(&output, notice),
                        None => output,
                    };
                    Ok(format_response(
                        &output,
                        started.elapsed().as_secs_f64(),
                        exit_code,
                        None,
                        original_token_count,
                    ))
                };
                let timed_out = |elapsed: f64| {
                    format_response(
                        &format!(
                            "Command timed out after {:.3} seconds.",
                            args.yield_time_ms as f64 / 1000.0
                        ),
                        elapsed,
                        None,
                        None,
                        None,
                    )
                };

                // A session without a PTY runs the command once and reports its exit code,
                // which is what Python's `exec_command` does on the same backend.
                if !session.supports_pty() {
                    return match one_shot(None).await {
                        Ok(response) => Ok(response),
                        Err(error) if error.is_timeout() => {
                            Ok(timed_out(started.elapsed().as_secs_f64()))
                        }
                        Err(error) => Err(AgentsError::from(error)),
                    };
                }

                let request = PtyExecRequest::new(args.cmd.clone())
                    .with_shell(shell.clone())
                    .with_tty(args.tty)
                    .with_user(run_as.clone())
                    .with_cwd(workdir.clone())
                    .with_yield_time_ms(args.yield_time_ms)
                    .with_max_output_tokens(args.max_output_tokens);
                match session.pty_exec_start(request).await {
                    Ok(update) => Ok(response_from_update(
                        &update,
                        started.elapsed().as_secs_f64(),
                    )),
                    Err(error) if error.is_timeout() => {
                        Ok(timed_out(started.elapsed().as_secs_f64()))
                    }
                    // A transport that dropped before the session opened never ran the command,
                    // so running it once more cannot repeat a side effect. A `tty=true` call
                    // asked for an interactive session, which one-shot exec cannot give it.
                    Err(error)
                        if !args.tty
                            && error.code == ErrorCode::ExecTransportError
                            && supports_transport_fallback(&error) =>
                    {
                        match one_shot(Some(
                            "PTY transport failed before the interactive session opened; fell \
                             back to one-shot exec.",
                        ))
                        .await
                        {
                            Ok(response) => Ok(response),
                            Err(error) if error.is_timeout() => {
                                Ok(timed_out(started.elapsed().as_secs_f64()))
                            }
                            Err(error) => Err(AgentsError::from(error)),
                        }
                    }
                    Err(error) => Err(AgentsError::from(error)),
                }
            }
        },
    )
    .with_strict_json_schema(false)
    .with_needs_approval(needs_approval);
    Tool::Function(Arc::new(tool))
}

/// Build the `write_stdin` tool bound to a session.
pub fn write_stdin_tool<C: Send + Sync + 'static>(
    session: Arc<dyn SandboxSession>,
    needs_approval: bool,
) -> Tool<C> {
    let tool = tool_with_safe_input::<C, WriteStdinArgs, _, _, _>(
        WRITE_STDIN_TOOL_NAME,
        WRITE_STDIN_DESCRIPTION,
        WRITE_STDIN_FIELDS,
        move |_ctx, args: WriteStdinArgs| {
            let session = session.clone();
            async move {
                session.ensure_mounts_settled()?;
                let started = std::time::Instant::now();
                let request = PtyWriteRequest::new(args.session_id, args.chars)
                    .with_yield_time_ms(args.yield_time_ms)
                    .with_max_output_tokens(args.max_output_tokens);
                match session.pty_write_stdin(request).await {
                    Ok(update) => Ok::<_, AgentsError>(response_from_update(
                        &update,
                        started.elapsed().as_secs_f64(),
                    )),
                    Err(error) if error.code == ErrorCode::PtySessionNotFound => {
                        Ok(format_response(
                            &format!("write_stdin failed: {}", error.message),
                            started.elapsed().as_secs_f64(),
                            Some(1),
                            None,
                            None,
                        ))
                    }
                    Err(error) if error.code == ErrorCode::PtyStdinUnavailable => {
                        Ok(format_response(
                            "stdin is not available for this process. Start the command with \
                             `tty=true` in `exec_command` before using `write_stdin`.",
                            started.elapsed().as_secs_f64(),
                            Some(1),
                            None,
                            None,
                        ))
                    }
                    Err(error) => Err(AgentsError::from(error)),
                }
            }
        },
    )
    .with_strict_json_schema(false)
    .with_needs_approval(needs_approval);
    Tool::Function(Arc::new(tool))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_shell_prefixes() {
        assert_eq!(resolve_shell(None, true), ExecShell::Login);
        assert_eq!(
            resolve_shell(None, false),
            ExecShell::Custom(vec!["sh".into(), "-c".into()])
        );
        assert_eq!(
            resolve_shell(Some("bash"), true),
            ExecShell::Custom(vec!["bash".into(), "-lc".into()])
        );
        assert_eq!(
            resolve_shell(Some("bash"), false),
            ExecShell::Custom(vec!["bash".into(), "-c".into()])
        );
    }

    #[test]
    fn formats_the_python_response_layout() {
        let response = format_response("hello\n", 1.5, Some(0), None, Some(42));
        let lines: Vec<&str> = response.lines().collect();
        assert!(lines[0].starts_with("Chunk ID: "));
        assert_eq!(lines[0].len(), "Chunk ID: ".len() + 6);
        assert_eq!(lines[1], "Wall time: 1.5000 seconds");
        assert_eq!(lines[2], "Process exited with code 0");
        assert_eq!(lines[3], "Original token count: 42");
        assert_eq!(lines[4], "Output:");
        assert_eq!(lines[5], "hello");

        let running = format_response("", 0.25, None, Some(1234), None);
        assert!(running.contains("Process running with session ID 1234"));
        assert!(!running.contains("Process exited"));
    }
}
