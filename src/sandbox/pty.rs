//! PTY-backed interactive sessions. Mirrors `agents.sandbox.session.pty_types`,
//! `agents.sandbox.session.pty_output` and the backend `pty_exec_start` implementations.
//!
//! A PTY session is started with [`SandboxSession::pty_exec_start`] and then polled or driven
//! with [`SandboxSession::pty_write_stdin`]. Both return a [`PtyExecUpdate`] holding the output
//! collected within the call's yield window. Processes live until they exit, the session stops,
//! or [`PtyManager`] prunes them.
//!
//! [`SandboxSession::pty_exec_start`]: crate::sandbox::SandboxSession::pty_exec_start
//! [`SandboxSession::pty_write_stdin`]: crate::sandbox::SandboxSession::pty_write_stdin

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use portable_pty::{ChildKiller, CommandBuilder, PtySize, native_pty_system};
use tokio::sync::Notify;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::session::ExecShell;
use crate::sandbox::types::User;
use crate::sandbox::util::truncate_text_by_tokens;

/// Default yield window of `exec_command`, in milliseconds.
pub const DEFAULT_EXEC_YIELD_TIME_MS: u64 = 10_000;
/// Default yield window of `write_stdin`, in milliseconds.
pub const DEFAULT_WRITE_STDIN_YIELD_TIME_MS: u64 = 250;
/// Smallest yield window honored by a PTY call.
pub const PTY_YIELD_TIME_MS_MIN: u64 = 250;
/// Smallest yield window honored by a `write_stdin` call that sends nothing.
pub const PTY_EMPTY_YIELD_TIME_MS_MIN: u64 = 5_000;
/// Largest yield window honored by a PTY call.
pub const PTY_YIELD_TIME_MS_MAX: u64 = 30_000;
/// Maximum number of live PTY processes per session.
pub const PTY_PROCESSES_MAX: usize = 64;
/// Number of live processes at which the session logs a warning.
pub const PTY_PROCESSES_WARNING: usize = 60;
/// Number of most-recently-used processes that pruning never touches.
pub const PTY_PROCESSES_PROTECTED_RECENT: usize = 8;
/// Lowest PTY process id handed to the model.
pub const PTY_PROCESS_ID_MIN: u32 = 1_000;
/// Exclusive upper bound of PTY process ids.
pub const PTY_PROCESS_ID_MAX_EXCLUSIVE: u32 = 100_000;

/// Clamp a model-provided yield window into the supported range.
pub fn clamp_pty_yield_time_ms(yield_time_ms: u64) -> u64 {
    yield_time_ms.clamp(PTY_YIELD_TIME_MS_MIN, PTY_YIELD_TIME_MS_MAX)
}

/// The yield window of a `write_stdin` call; a call that sends nothing waits longer, because it
/// only exists to collect output.
pub fn resolve_pty_write_yield_time_ms(yield_time_ms: u64, input_empty: bool) -> u64 {
    let clamped = clamp_pty_yield_time_ms(yield_time_ms);
    if input_empty {
        clamped.max(PTY_EMPTY_YIELD_TIME_MS_MIN.min(PTY_YIELD_TIME_MS_MAX))
    } else {
        clamped
    }
}

/// Output and status collected from a PTY session within one yield window.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtyExecUpdate {
    /// The session id to pass to `write_stdin`, while the process is still running.
    pub process_id: Option<u32>,
    /// Raw output bytes collected in this window.
    pub output: Vec<u8>,
    /// The exit code, once the process has finished.
    pub exit_code: Option<i32>,
    /// Approximate token count of the output before truncation.
    pub original_token_count: Option<usize>,
}

impl PtyExecUpdate {
    /// The output decoded as lossy UTF-8.
    pub fn output_text(&self) -> String {
        String::from_utf8_lossy(&self.output).into_owned()
    }
}

/// A request to start a command in a PTY.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtyExecRequest {
    /// The command line to run.
    pub command: String,
    /// How the command reaches the shell.
    pub shell: ExecShell,
    /// Whether the model may write to the process (`write_stdin`).
    pub tty: bool,
    /// Sandbox user to run as.
    pub user: Option<User>,
    /// Workspace-relative or absolute working directory.
    pub cwd: Option<String>,
    /// How long to wait for output before returning.
    pub yield_time_ms: u64,
    /// Maximum number of tokens of output to return.
    pub max_output_tokens: Option<usize>,
}

impl PtyExecRequest {
    /// A PTY command with the default yield window.
    pub fn new(command: impl Into<String>) -> Self {
        PtyExecRequest {
            command: command.into(),
            shell: ExecShell::Login,
            tty: false,
            user: None,
            cwd: None,
            yield_time_ms: DEFAULT_EXEC_YIELD_TIME_MS,
            max_output_tokens: None,
        }
    }

    /// Allow the model to write to the process.
    pub fn with_tty(mut self, tty: bool) -> Self {
        self.tty = tty;
        self
    }

    /// Set the shell handling.
    pub fn with_shell(mut self, shell: ExecShell) -> Self {
        self.shell = shell;
        self
    }

    /// Set the sandbox user.
    pub fn with_user(mut self, user: Option<User>) -> Self {
        self.user = user;
        self
    }

    /// Set the working directory.
    pub fn with_cwd(mut self, cwd: Option<String>) -> Self {
        self.cwd = cwd;
        self
    }

    /// Set the yield window.
    pub fn with_yield_time_ms(mut self, yield_time_ms: u64) -> Self {
        self.yield_time_ms = yield_time_ms;
        self
    }

    /// Set the output token budget.
    pub fn with_max_output_tokens(mut self, max_output_tokens: Option<usize>) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }

    /// The argument vector handed to the PTY, after shell and `sudo` wrapping.
    pub fn resolved_argv(&self) -> Vec<String> {
        let mut argv = match self.shell.prefix() {
            None => vec![self.command.clone()],
            Some(mut prefix) => {
                prefix.push(self.command.clone());
                prefix
            }
        };
        if let Some(user) = &self.user {
            let mut sudo = vec![
                "sudo".to_string(),
                "-u".to_string(),
                user.name.clone(),
                "--".to_string(),
            ];
            sudo.append(&mut argv);
            argv = sudo;
        }
        argv
    }
}

/// A request to write to a running PTY session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtyWriteRequest {
    /// The session id returned by a previous `exec_command`.
    pub session_id: u32,
    /// Bytes to write; empty polls for output.
    pub chars: String,
    /// How long to wait for output before returning.
    pub yield_time_ms: u64,
    /// Maximum number of tokens of output to return.
    pub max_output_tokens: Option<usize>,
}

impl PtyWriteRequest {
    /// A write (or poll, with empty `chars`) against `session_id`.
    pub fn new(session_id: u32, chars: impl Into<String>) -> Self {
        PtyWriteRequest {
            session_id,
            chars: chars.into(),
            yield_time_ms: DEFAULT_WRITE_STDIN_YIELD_TIME_MS,
            max_output_tokens: None,
        }
    }

    /// Set the yield window.
    pub fn with_yield_time_ms(mut self, yield_time_ms: u64) -> Self {
        self.yield_time_ms = yield_time_ms;
        self
    }

    /// Set the output token budget.
    pub fn with_max_output_tokens(mut self, max_output_tokens: Option<usize>) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }
}

/// How a backend wants a PTY process started.
pub struct PtySpawn {
    /// The argument vector to run in the PTY.
    pub argv: Vec<String>,
    /// Host working directory, when the backend runs the command locally.
    pub cwd: Option<std::path::PathBuf>,
    /// Environment variables to export.
    pub env: Vec<(String, String)>,
    /// Whether to clear the inherited host environment first.
    pub clear_env: bool,
    /// Whether the model may write to the process.
    pub tty: bool,
    /// A command run inside the sandbox when the process is terminated, used by backends whose
    /// PTY process is only a proxy for a process running elsewhere.
    pub remote_kill_command: Option<String>,
}

impl fmt::Debug for PtySpawn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PtySpawn")
            .field("argv", &self.argv)
            .field("cwd", &self.cwd)
            .field("tty", &self.tty)
            .field("clear_env", &self.clear_env)
            .finish_non_exhaustive()
    }
}

struct PtyProcess {
    id: u32,
    tty: bool,
    buffer: Mutex<Vec<u8>>,
    /// The child process has exited.
    closed: Arc<AtomicBool>,
    /// The pty reader reached EOF, so no further output can arrive.
    output_closed: Arc<AtomicBool>,
    exit_code: Arc<Mutex<Option<i32>>>,
    notify: Arc<Notify>,
    writer: Mutex<Option<Box<dyn Write + Send>>>,
    killer: Mutex<Box<dyn ChildKiller + Send + Sync>>,
    last_used: Mutex<Instant>,
    remote_kill_command: Option<String>,
}

impl fmt::Debug for PtyProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PtyProcess")
            .field("id", &self.id)
            .field("tty", &self.tty)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl PtyProcess {
    fn touch(&self) {
        *self.last_used.lock().expect("poisoned") = Instant::now();
    }

    /// Whether the process is finished *and* its output has been fully drained from the pty.
    ///
    /// Waiting for EOF as well as exit matters: the child can write its last bytes and exit
    /// before the pump thread has read them, and treating exit alone as "done" loses that tail.
    fn is_done(&self) -> bool {
        self.closed.load(Ordering::SeqCst) && self.output_closed.load(Ordering::SeqCst)
    }

    fn take_output(&self, carry_incomplete: bool) -> Vec<u8> {
        let mut buffer = self.buffer.lock().expect("poisoned");
        if buffer.is_empty() {
            return Vec::new();
        }
        let mut out = std::mem::take(&mut *buffer);
        if carry_incomplete {
            let carry = incomplete_utf8_suffix_length(&out);
            if carry > 0 {
                let tail = out.split_off(out.len() - carry);
                *buffer = tail;
            }
        }
        out
    }

    fn exit_code(&self) -> Option<i32> {
        *self.exit_code.lock().expect("poisoned")
    }
}

/// Return the trailing byte count that can still become one valid UTF-8 scalar.
///
/// A PTY read can split a multi-byte character, so those bytes are carried into the next window
/// instead of being decoded as replacement characters.
pub fn incomplete_utf8_suffix_length(data: &[u8]) -> usize {
    let mut continuation_count = 0usize;
    for byte in data.iter().rev().take(3) {
        if (0x80..=0xBF).contains(byte) {
            continuation_count += 1;
        } else {
            break;
        }
    }
    if continuation_count + 1 > data.len() {
        return 0;
    }
    let lead_index = data.len() - continuation_count - 1;
    let lead = data[lead_index];
    let expected_length = match lead {
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return 0,
    };
    let suffix_length = continuation_count + 1;
    if suffix_length >= expected_length {
        return 0;
    }
    if continuation_count > 0 {
        let second = data[lead_index + 1];
        if (lead == 0xE0 && second < 0xA0)
            || (lead == 0xED && second > 0x9F)
            || (lead == 0xF0 && second < 0x90)
            || (lead == 0xF4 && second > 0x8F)
        {
            return 0;
        }
    }
    suffix_length
}

/// Tracks the live PTY processes of one sandbox session.
///
/// Shared by every backend: a backend only has to describe how to spawn the process
/// ([`PtySpawn`]), and the manager owns the pty, the output pump, process ids and pruning.
#[derive(Debug, Default)]
pub struct PtyManager {
    processes: Mutex<HashMap<u32, Arc<PtyProcess>>>,
}

impl PtyManager {
    /// An empty manager.
    pub fn new() -> Self {
        PtyManager::default()
    }

    /// Number of live processes.
    pub fn len(&self) -> usize {
        self.processes.lock().expect("poisoned").len()
    }

    /// Whether no process is live.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn allocate_id(&self, processes: &HashMap<u32, Arc<PtyProcess>>) -> SandboxResult<u32> {
        for _ in 0..1024 {
            let span = PTY_PROCESS_ID_MAX_EXCLUSIVE - PTY_PROCESS_ID_MIN;
            let candidate =
                PTY_PROCESS_ID_MIN + (uuid::Uuid::new_v4().as_u128() % span as u128) as u32;
            if !processes.contains_key(&candidate) {
                return Ok(candidate);
            }
        }
        Err(SandboxError::new(
            ErrorCode::ExecTransportError,
            "could not allocate a free PTY session id",
        ))
    }

    /// Drop finished processes, then the least recently used ones, keeping the newest
    /// [`PTY_PROCESSES_PROTECTED_RECENT`] alive.
    ///
    /// Returns the sandbox-side cleanup commands of the processes that were killed, so a backend
    /// whose PTY process only proxies work running elsewhere can finish the job.
    fn prune(processes: &mut HashMap<u32, Arc<PtyProcess>>) -> Vec<String> {
        let mut cleanup = Vec::new();
        processes.retain(|_, process| {
            if process.is_done() {
                if let Some(command) = &process.remote_kill_command {
                    cleanup.push(command.clone());
                }
                false
            } else {
                true
            }
        });
        if processes.len() < PTY_PROCESSES_MAX {
            return cleanup;
        }
        let mut by_age: Vec<(u32, Instant)> = processes
            .iter()
            .map(|(id, p)| (*id, *p.last_used.lock().expect("poisoned")))
            .collect();
        by_age.sort_by_key(|(_, last_used)| *last_used);
        let prunable = by_age.len().saturating_sub(PTY_PROCESSES_PROTECTED_RECENT);
        for (id, _) in by_age.into_iter().take(prunable) {
            if let Some(process) = processes.remove(&id) {
                if let Some(command) = &process.remote_kill_command {
                    cleanup.push(command.clone());
                }
                let _ = process.killer.lock().expect("poisoned").kill();
            }
            if processes.len() < PTY_PROCESSES_MAX {
                break;
            }
        }
        cleanup
    }

    /// Start a process in a PTY and collect its first output window.
    ///
    /// Returns the update plus the sandbox-side cleanup commands of any process that finished or
    /// was pruned, which a backend whose PTY process only proxies remote work must run.
    pub async fn start(
        &self,
        spawn: PtySpawn,
        yield_time_ms: u64,
        max_output_tokens: Option<usize>,
    ) -> SandboxResult<(PtyExecUpdate, Vec<String>)> {
        let (process, mut cleanup) = self.spawn(spawn)?;
        let update = collect(&process, yield_time_ms, max_output_tokens).await;
        if process.is_done()
            && let Some(finished) = self.processes.lock().expect("poisoned").remove(&process.id)
            && let Some(command) = &finished.remote_kill_command
        {
            cleanup.push(command.clone());
        }
        Ok((update, cleanup))
    }

    fn spawn(&self, spawn: PtySpawn) -> SandboxResult<(Arc<PtyProcess>, Vec<String>)> {
        let PtySpawn {
            argv,
            cwd,
            env,
            clear_env,
            tty,
            remote_kill_command,
        } = spawn;
        let Some((program, args)) = argv.split_first() else {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "a PTY command needs at least one word",
            ));
        };

        let pair = native_pty_system()
            .openpty(PtySize::default())
            .map_err(|e| transport_error(format!("failed to open a pty: {e}")))?;
        let mut command = CommandBuilder::new(program);
        command.args(args);
        if clear_env {
            command.env_clear();
            for key in ["PATH", "HOME", "LANG", "TERM"] {
                if let Ok(value) = std::env::var(key) {
                    command.env(key, value);
                }
            }
        }
        for (key, value) in env {
            command.env(key, value);
        }
        if let Some(cwd) = &cwd {
            command.cwd(cwd);
        }

        let mut child = pair
            .slave
            .spawn_command(command)
            .map_err(|e| transport_error(format!("failed to start the pty command: {e}")))?;
        // The reader only sees EOF once every slave handle is closed.
        drop(pair.slave);

        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| transport_error(format!("failed to read from the pty: {e}")))?;
        let writer = if tty {
            Some(
                pair.master
                    .take_writer()
                    .map_err(|e| transport_error(format!("failed to write to the pty: {e}")))?,
            )
        } else {
            None
        };
        let killer = child.clone_killer();

        let mut processes = self.processes.lock().expect("poisoned");
        let cleanup = Self::prune(&mut processes);
        if processes.len() >= PTY_PROCESSES_MAX {
            let _ = killer_kill(killer);
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                format!("the sandbox already has {PTY_PROCESSES_MAX} live PTY sessions"),
            ));
        }
        if processes.len() >= PTY_PROCESSES_WARNING {
            log::warn!(
                "sandbox session has {} live PTY sessions (limit {PTY_PROCESSES_MAX})",
                processes.len()
            );
        }
        let id = self.allocate_id(&processes)?;

        let process = Arc::new(PtyProcess {
            id,
            tty,
            buffer: Mutex::new(Vec::new()),
            closed: Arc::new(AtomicBool::new(false)),
            output_closed: Arc::new(AtomicBool::new(false)),
            exit_code: Arc::new(Mutex::new(None)),
            notify: Arc::new(Notify::new()),
            writer: Mutex::new(writer),
            killer: Mutex::new(killer),
            last_used: Mutex::new(Instant::now()),
            remote_kill_command,
        });
        processes.insert(id, process.clone());
        drop(processes);

        // Pump output on a blocking thread; `master` is kept alive by the closure.
        {
            let buffer_owner = process.clone();
            let notify = process.notify.clone();
            let output_closed = process.output_closed.clone();
            let master = pair.master;
            std::thread::spawn(move || {
                let mut reader = reader;
                let mut chunk = [0u8; 8192];
                loop {
                    match std::io::Read::read(&mut reader, &mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            buffer_owner
                                .buffer
                                .lock()
                                .expect("poisoned")
                                .extend_from_slice(&chunk[..n]);
                            notify.notify_waiters();
                        }
                    }
                }
                drop(master);
                output_closed.store(true, Ordering::SeqCst);
                notify.notify_waiters();
            });
        }

        // Reap the child on another blocking thread so exit codes land without polling.
        {
            let closed = process.closed.clone();
            let exit_code = process.exit_code.clone();
            let notify = process.notify.clone();
            std::thread::spawn(move || {
                let status = child.wait();
                if let Ok(status) = status {
                    *exit_code.lock().expect("poisoned") = Some(status.exit_code() as i32);
                }
                closed.store(true, Ordering::SeqCst);
                notify.notify_waiters();
            });
        }

        Ok((process, cleanup))
    }

    /// Write to (or poll) a running PTY session and collect its next output window.
    ///
    /// Returns the update plus the sandbox-side cleanup commands of any process that finished.
    pub async fn write_stdin(
        &self,
        request: &PtyWriteRequest,
    ) -> SandboxResult<(PtyExecUpdate, Vec<String>)> {
        let process = self
            .processes
            .lock()
            .expect("poisoned")
            .get(&request.session_id)
            .cloned()
            .ok_or_else(|| {
                SandboxError::new(
                    ErrorCode::PtySessionNotFound,
                    format!("PTY session `{}` is not running", request.session_id),
                )
            })?;
        if !request.chars.is_empty() {
            if !process.tty {
                return Err(SandboxError::new(
                    ErrorCode::PtyStdinUnavailable,
                    "stdin is not available for this process",
                ));
            }
            let mut guard = process.writer.lock().expect("poisoned");
            let Some(writer) = guard.as_mut() else {
                return Err(SandboxError::new(
                    ErrorCode::PtyStdinUnavailable,
                    "stdin is not available for this process",
                ));
            };
            writer
                .write_all(request.chars.as_bytes())
                .and_then(|()| writer.flush())
                .map_err(|e| transport_error(format!("failed to write to the pty: {e}")))?;
        }
        let yield_time_ms =
            resolve_pty_write_yield_time_ms(request.yield_time_ms, request.chars.is_empty());
        let update = collect(&process, yield_time_ms, request.max_output_tokens).await;
        let mut cleanup = Vec::new();
        if process.is_done()
            && let Some(finished) = self
                .processes
                .lock()
                .expect("poisoned")
                .remove(&request.session_id)
            && let Some(command) = &finished.remote_kill_command
        {
            cleanup.push(command.clone());
        }
        Ok((update, cleanup))
    }

    /// Kill every live process and return the sandbox-side cleanup commands of the processes
    /// that proxy work running elsewhere.
    pub fn terminate_all(&self) -> Vec<String> {
        let mut processes = self.processes.lock().expect("poisoned");
        let mut cleanup = Vec::new();
        for (_, process) in processes.drain() {
            if let Some(command) = &process.remote_kill_command {
                cleanup.push(command.clone());
            }
            let _ = process.killer.lock().expect("poisoned").kill();
            process.notify.notify_waiters();
        }
        cleanup
    }
}

fn killer_kill(mut killer: Box<dyn ChildKiller + Send + Sync>) -> std::io::Result<()> {
    killer.kill()
}

fn transport_error(message: String) -> SandboxError {
    SandboxError::new(ErrorCode::ExecTransportError, message)
}

/// Collect output until the yield window elapses or the process finishes.
async fn collect(
    process: &Arc<PtyProcess>,
    yield_time_ms: u64,
    max_output_tokens: Option<usize>,
) -> PtyExecUpdate {
    process.touch();
    let deadline = Instant::now() + Duration::from_millis(clamp_pty_yield_time_ms(yield_time_ms));
    let mut output = Vec::new();
    let mut closed = false;
    loop {
        output.extend(process.take_output(false));
        if process.is_done() {
            closed = true;
            // Give the pump a moment to flush whatever the process wrote before exiting.
            tokio::time::sleep(Duration::from_millis(20)).await;
            output.extend(process.take_output(false));
            break;
        }
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let notified = process.notify.notified();
        if tokio::time::timeout(deadline - now, notified)
            .await
            .is_err()
        {
            break;
        }
    }
    if !closed {
        // An unfinished process may have been cut mid-character; carry those bytes over.
        output.extend(process.take_output(true));
    } else {
        output.extend(process.take_output(false));
    }
    process.touch();

    let text = String::from_utf8_lossy(&output).into_owned();
    let (text, original_token_count) = truncate_text_by_tokens(&text, max_output_tokens);
    PtyExecUpdate {
        process_id: (!closed).then_some(process.id),
        output: text.into_bytes(),
        exit_code: process.exit_code(),
        original_token_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_yield_windows() {
        assert_eq!(clamp_pty_yield_time_ms(0), PTY_YIELD_TIME_MS_MIN);
        assert_eq!(clamp_pty_yield_time_ms(1_000), 1_000);
        assert_eq!(clamp_pty_yield_time_ms(60_000), PTY_YIELD_TIME_MS_MAX);
        assert_eq!(resolve_pty_write_yield_time_ms(250, false), 250);
        assert_eq!(
            resolve_pty_write_yield_time_ms(250, true),
            PTY_EMPTY_YIELD_TIME_MS_MIN
        );
    }

    #[test]
    fn detects_incomplete_utf8_suffixes() {
        assert_eq!(incomplete_utf8_suffix_length(b"hello"), 0);
        assert_eq!(incomplete_utf8_suffix_length("é".as_bytes()), 0);
        assert_eq!(incomplete_utf8_suffix_length(&[0xC3]), 1);
        assert_eq!(incomplete_utf8_suffix_length(&[0xE2, 0x82]), 2);
        assert_eq!(incomplete_utf8_suffix_length("€".as_bytes()), 0);
        assert_eq!(incomplete_utf8_suffix_length(&[0xF0, 0x9F, 0x92]), 3);
        // An over-long or invalid lead byte carries nothing.
        assert_eq!(incomplete_utf8_suffix_length(&[0xFF, 0x80]), 0);
    }

    #[test]
    fn builds_pty_argv() {
        let request = PtyExecRequest::new("echo hi");
        assert_eq!(request.resolved_argv(), vec!["sh", "-lc", "echo hi"]);
        let as_user = PtyExecRequest::new("id").with_user(Some(User::new("agent")));
        assert_eq!(
            as_user.resolved_argv(),
            vec!["sudo", "-u", "agent", "--", "sh", "-lc", "id"]
        );
    }

    #[tokio::test]
    async fn runs_a_command_to_completion() {
        let manager = PtyManager::new();
        let (update, cleanup) = manager
            .start(
                PtySpawn {
                    argv: vec!["sh".into(), "-c".into(), "echo hello-pty".into()],
                    cwd: None,
                    env: Vec::new(),
                    clear_env: false,
                    tty: false,
                    remote_kill_command: None,
                },
                2_000,
                None,
            )
            .await
            .unwrap();
        assert!(cleanup.is_empty());
        assert!(update.output_text().contains("hello-pty"), "{update:?}");
        assert_eq!(update.exit_code, Some(0));
        assert_eq!(update.process_id, None);
        assert!(manager.is_empty());
    }

    #[tokio::test]
    async fn keeps_an_interactive_session_and_writes_to_it() {
        let manager = PtyManager::new();
        let (update, _) = manager
            .start(
                PtySpawn {
                    argv: vec!["sh".into(), "-c".into(), "read line; echo got:$line".into()],
                    cwd: None,
                    env: Vec::new(),
                    clear_env: false,
                    tty: true,
                    remote_kill_command: None,
                },
                400,
                None,
            )
            .await
            .unwrap();
        let session_id = update.process_id.expect("session should still be running");

        let mut seen = String::new();
        let mut exit_code = None;
        for _ in 0..10 {
            let (update, _) = manager
                .write_stdin(&PtyWriteRequest::new(session_id, "ping\n").with_yield_time_ms(500))
                .await
                .unwrap_or_default();
            seen.push_str(&update.output_text());
            exit_code = exit_code.or(update.exit_code);
            if seen.contains("got:ping") {
                break;
            }
        }
        assert!(seen.contains("got:ping"), "{seen:?}");
        assert!(manager.terminate_all().is_empty());
    }

    #[tokio::test]
    async fn rejects_stdin_without_a_tty() {
        let manager = PtyManager::new();
        let (update, _) = manager
            .start(
                PtySpawn {
                    argv: vec!["sh".into(), "-c".into(), "sleep 3".into()],
                    cwd: None,
                    env: Vec::new(),
                    clear_env: false,
                    tty: false,
                    remote_kill_command: None,
                },
                300,
                None,
            )
            .await
            .unwrap();
        let session_id = update.process_id.expect("sleep should still be running");
        let err = manager
            .write_stdin(&PtyWriteRequest::new(session_id, "x"))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::PtyStdinUnavailable);
        // Polling with no input still works.
        assert!(
            manager
                .write_stdin(&PtyWriteRequest::new(session_id, "").with_yield_time_ms(250))
                .await
                .is_ok()
        );
        manager.terminate_all();
    }

    #[tokio::test]
    async fn reports_unknown_sessions() {
        let manager = PtyManager::new();
        let err = manager
            .write_stdin(&PtyWriteRequest::new(4242, ""))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::PtySessionNotFound);
    }
}
