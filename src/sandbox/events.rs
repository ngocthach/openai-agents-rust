//! An audit trail of what a sandbox session did. Mirrors `agents.sandbox.session.events` and
//! `session.sinks`.
//!
//! A sandbox runs code an agent wrote, so "what did it actually do" is a question that gets asked
//! after the fact — during an incident, a compliance review, or a confusing run. The session's
//! own return values answer it only while the run is in memory. An [`EventSink`] records each
//! operation as it happens: which command, against which path, how long it took, and whether it
//! worked.
//!
//! What is *not* recorded by default is as deliberate: command output and file contents are the
//! agent's data, often the user's, and copying them into a log is a new place for a secret to
//! live. [`EventPayloadPolicy`] turns output on explicitly and bounds it when it is on.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxOp, SandboxResult};

/// The current time in milliseconds since the Unix epoch.
fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or_default()
}

/// Which half of an operation an event describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventPhase {
    /// The operation is about to run.
    Start,
    /// The operation finished, successfully or not.
    Finish,
}

/// How much of an operation's data reaches the sink. Mirrors Python's `EventPayloadPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventPayloadPolicy {
    /// Whether command output is recorded at all. Off by default: it is the agent's data.
    pub include_exec_output: bool,
    /// The most standard output recorded, in characters.
    pub max_stdout_chars: usize,
    /// The most standard error recorded, in characters.
    pub max_stderr_chars: usize,
    /// Whether a write records how many bytes it wrote. The bytes themselves never are.
    pub include_write_len: bool,
}

impl Default for EventPayloadPolicy {
    fn default() -> Self {
        EventPayloadPolicy {
            include_exec_output: false,
            max_stdout_chars: 8_000,
            max_stderr_chars: 8_000,
            include_write_len: true,
        }
    }
}

impl EventPayloadPolicy {
    /// The default policy: no command output.
    pub fn new() -> Self {
        EventPayloadPolicy::default()
    }

    /// Record command output, bounded by this policy's limits.
    pub fn with_exec_output(mut self, include: bool) -> Self {
        self.include_exec_output = include;
        self
    }

    /// Bound how much output is recorded.
    pub fn with_output_limits(mut self, stdout: usize, stderr: usize) -> Self {
        self.max_stdout_chars = stdout;
        self.max_stderr_chars = stderr;
        self
    }

    /// Apply the policy to one stream.
    fn apply(&self, raw: &[u8], limit: usize) -> Option<String> {
        if !self.include_exec_output {
            return None;
        }
        let text = String::from_utf8_lossy(raw);
        if text.chars().count() <= limit {
            return Some(text.into_owned());
        }
        Some(text.chars().take(limit).collect())
    }
}

/// One recorded sandbox operation. Mirrors Python's `SandboxSessionEvent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxSessionEvent {
    /// Payload format version.
    pub version: u32,
    /// When the event was recorded, in milliseconds since the Unix epoch.
    ///
    /// An audit record with no time on it cannot be lined up with anything else that happened.
    pub ts: u64,
    /// The session the operation ran against.
    pub session_id: String,
    /// The backend that owns the session.
    pub backend: String,
    /// Monotonic position in this session's event stream.
    pub seq: u64,
    /// Correlates the start and finish of one operation.
    pub span_id: String,
    /// The operation.
    pub op: String,
    /// Which half of the operation this is.
    pub phase: EventPhase,
    /// Operation metadata: paths, argument vectors, sizes. Never file contents.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<(String, String)>,
    /// Whether the operation succeeded, on a finish event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
    /// How long the operation took, on a finish event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// The failure's stable code, on a failed finish event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// The failure's message, on a failed finish event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Whether retrying is expected to help, when the failure says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_retryable: Option<bool>,
    /// Command output, only when [`EventPayloadPolicy::include_exec_output`] is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    /// Command error output, only when [`EventPayloadPolicy::include_exec_output`] is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
}

impl SandboxSessionEvent {
    /// A start event.
    pub fn start(
        session_id: impl Into<String>,
        backend: impl Into<String>,
        seq: u64,
        span_id: impl Into<String>,
        op: SandboxOp,
    ) -> Self {
        SandboxSessionEvent {
            version: 1,
            ts: unix_millis(),
            session_id: session_id.into(),
            backend: backend.into(),
            seq,
            span_id: span_id.into(),
            op: op.as_str().to_string(),
            phase: EventPhase::Start,
            data: Vec::new(),
            ok: None,
            duration_ms: None,
            error_code: None,
            error_message: None,
            error_retryable: None,
            stdout: None,
            stderr: None,
        }
    }

    /// The matching finish event.
    pub fn finish(&self, ok: bool, duration_ms: u64) -> Self {
        SandboxSessionEvent {
            phase: EventPhase::Finish,
            ts: unix_millis(),
            ok: Some(ok),
            duration_ms: Some(duration_ms),
            data: Vec::new(),
            ..self.clone()
        }
    }

    /// Attach one piece of metadata.
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.push((key.into(), value.into()));
        self
    }

    /// Record the failure that ended the operation.
    pub fn with_error(mut self, error: &SandboxError) -> Self {
        self.ok = Some(false);
        self.error_code = Some(error.code.as_str().to_string());
        self.error_message = Some(error.message.clone());
        self.error_retryable = error.retryable;
        self
    }

    /// Record command output under `policy`.
    pub fn with_output(mut self, policy: EventPayloadPolicy, stdout: &[u8], stderr: &[u8]) -> Self {
        self.stdout = policy.apply(stdout, policy.max_stdout_chars);
        self.stderr = policy.apply(stderr, policy.max_stderr_chars);
        self
    }

    /// The event as one JSON line.
    pub fn to_json_line(&self) -> SandboxResult<String> {
        let mut line = serde_json::to_string(self).map_err(|error| {
            SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                format!("failed to serialize a sandbox session event: {error}"),
            )
        })?;
        line.push('\n');
        Ok(line)
    }
}

/// Somewhere sandbox session events are recorded. Mirrors Python's `EventSink`.
///
/// A sink must not fail the operation it is describing: an audit trail that can break a run is
/// worse than no audit trail, so implementors swallow their own transport errors.
#[async_trait]
pub trait EventSink: Send + Sync + fmt::Debug {
    /// Record one event.
    async fn emit(&self, event: &SandboxSessionEvent);

    /// Flush anything buffered. The default does nothing.
    async fn flush(&self) {}
}

/// Hands each event to a closure. Mirrors Python's `CallbackSink`.
pub struct CallbackSink {
    callback: Arc<dyn Fn(&SandboxSessionEvent) + Send + Sync>,
}

impl CallbackSink {
    /// A sink calling `callback` for every event.
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(&SandboxSessionEvent) + Send + Sync + 'static,
    {
        CallbackSink {
            callback: Arc::new(callback),
        }
    }
}

impl fmt::Debug for CallbackSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CallbackSink")
    }
}

#[async_trait]
impl EventSink for CallbackSink {
    async fn emit(&self, event: &SandboxSessionEvent) {
        (self.callback)(event);
    }
}

/// Appends each event to a JSON Lines file on the machine running the SDK. Mirrors Python's
/// `JsonlOutboxSink`.
#[derive(Debug)]
pub struct JsonlFileSink {
    path: PathBuf,
    handle: Mutex<Option<tokio::fs::File>>,
}

impl JsonlFileSink {
    /// A sink appending to `path`, which is created on the first event.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        JsonlFileSink {
            path: path.into(),
            handle: Mutex::new(None),
        }
    }

    /// The file being written.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

#[async_trait]
impl EventSink for JsonlFileSink {
    async fn emit(&self, event: &SandboxSessionEvent) {
        use tokio::io::AsyncWriteExt as _;

        let Ok(line) = event.to_json_line() else {
            return;
        };
        let mut handle = self.handle.lock().await;
        if handle.is_none() {
            match tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .await
            {
                Ok(file) => *handle = Some(file),
                // A broken audit file must not break the run it is describing.
                Err(error) => {
                    log::warn!(
                        "failed to open the sandbox event log `{}`: {error}",
                        self.path.display()
                    );
                    return;
                }
            }
        }
        if let Some(file) = handle.as_mut()
            && let Err(error) = file.write_all(line.as_bytes()).await
        {
            log::warn!(
                "failed to append to the sandbox event log `{}`: {error}",
                self.path.display()
            );
        }
    }

    async fn flush(&self) {
        use tokio::io::AsyncWriteExt as _;

        let mut handle = self.handle.lock().await;
        if let Some(file) = handle.as_mut() {
            let _ = file.flush().await;
        }
    }
}

/// Sends every event to several sinks. Mirrors Python's `ChainedSink`.
#[derive(Debug, Default)]
pub struct ChainedSink {
    sinks: Vec<Arc<dyn EventSink>>,
}

impl ChainedSink {
    /// A chain with no sinks.
    pub fn new() -> Self {
        ChainedSink::default()
    }

    /// Add a sink to the chain.
    pub fn with_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.sinks.push(sink);
        self
    }
}

#[async_trait]
impl EventSink for ChainedSink {
    async fn emit(&self, event: &SandboxSessionEvent) {
        for sink in &self.sinks {
            sink.emit(event).await;
        }
    }

    async fn flush(&self) {
        for sink in &self.sinks {
            sink.flush().await;
        }
    }
}

/// Issues the start and finish events for a session's operations. Mirrors Python's
/// `Instrumentation`.
#[derive(Debug)]
pub struct Instrumentation {
    sink: Arc<dyn EventSink>,
    policy: EventPayloadPolicy,
    session_id: String,
    backend: String,
    seq: AtomicU64,
}

impl Instrumentation {
    /// Instrumentation for one session, writing to `sink`.
    pub fn new(
        sink: Arc<dyn EventSink>,
        session_id: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        Instrumentation {
            sink,
            policy: EventPayloadPolicy::new(),
            session_id: session_id.into(),
            backend: backend.into(),
            seq: AtomicU64::new(0),
        }
    }

    /// Record command output under `policy`.
    pub fn with_policy(mut self, policy: EventPayloadPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The payload policy in force.
    pub fn policy(&self) -> EventPayloadPolicy {
        self.policy
    }

    /// Emit the start event for one operation and return it, for the matching finish.
    pub async fn start(&self, op: SandboxOp, data: Vec<(String, String)>) -> SandboxSessionEvent {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let mut event = SandboxSessionEvent::start(
            self.session_id.clone(),
            self.backend.clone(),
            seq,
            format!("{}-{seq}", self.session_id),
            op,
        );
        event.data = data;
        self.sink.emit(&event).await;
        event
    }

    /// Emit the finish event for an operation started with [`start`](Self::start).
    pub async fn finish(
        &self,
        started: &SandboxSessionEvent,
        duration_ms: u64,
        outcome: Result<(&[u8], &[u8]), &SandboxError>,
    ) {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let mut event = match outcome {
            Ok((stdout, stderr)) => {
                started
                    .finish(true, duration_ms)
                    .with_output(self.policy, stdout, stderr)
            }
            Err(error) => started.finish(false, duration_ms).with_error(error),
        };
        event.seq = seq;
        self.sink.emit(&event).await;
    }

    /// Flush the sink.
    pub async fn flush(&self) {
        self.sink.flush().await;
    }
}

/// Wraps a session so every operation it performs is recorded.
///
/// Pass this to [`SandboxRunConfig::with_session`](crate::sandbox::SandboxRunConfig::with_session)
/// in place of the real session; it behaves identically and writes an audit trail as it goes.
///
/// ```no_run
/// use std::sync::Arc;
///
/// use openai_agents_sdk::sandbox::events::{InstrumentedSession, JsonlFileSink};
/// # fn example(session: Arc<dyn openai_agents_sdk::sandbox::SandboxSession>) {
/// let audited = Arc::new(InstrumentedSession::new(
///     session,
///     Arc::new(JsonlFileSink::new("/var/log/sandbox-events.jsonl")),
/// ));
/// # let _ = audited;
/// # }
/// ```
#[derive(Debug)]
pub struct InstrumentedSession {
    inner: Arc<dyn crate::sandbox::session::SandboxSession>,
    instrumentation: Instrumentation,
}

impl InstrumentedSession {
    /// Record `inner`'s operations to `sink`.
    pub fn new(
        inner: Arc<dyn crate::sandbox::session::SandboxSession>,
        sink: Arc<dyn EventSink>,
    ) -> Self {
        let instrumentation = Instrumentation::new(sink, inner.id(), inner.backend_id());
        InstrumentedSession {
            inner,
            instrumentation,
        }
    }

    /// Record command output under `policy`.
    pub fn with_policy(mut self, policy: EventPayloadPolicy) -> Self {
        self.instrumentation = self.instrumentation.with_policy(policy);
        self
    }

    /// The session being recorded.
    pub fn inner(&self) -> &Arc<dyn crate::sandbox::session::SandboxSession> {
        &self.inner
    }

    /// Run one operation, recording its start and finish.
    async fn record<T, F>(
        &self,
        op: SandboxOp,
        data: Vec<(String, String)>,
        operation: F,
    ) -> SandboxResult<T>
    where
        F: std::future::Future<Output = SandboxResult<T>>,
    {
        let started = self.instrumentation.start(op, data).await;
        let clock = std::time::Instant::now();
        let outcome = operation.await;
        let elapsed = clock.elapsed().as_millis() as u64;
        match &outcome {
            Ok(_) => {
                self.instrumentation
                    .finish(&started, elapsed, Ok((&[], &[])))
                    .await
            }
            Err(error) => {
                self.instrumentation
                    .finish(&started, elapsed, Err(error))
                    .await
            }
        }
        outcome
    }
}

#[async_trait]
impl crate::sandbox::session::SandboxSession for InstrumentedSession {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn manifest(&self) -> &crate::sandbox::manifest::Manifest {
        self.inner.manifest()
    }

    fn workspace_root(&self) -> &str {
        self.inner.workspace_root()
    }

    fn backend_id(&self) -> &str {
        self.inner.backend_id()
    }

    fn supports_pty(&self) -> bool {
        self.inner.supports_pty()
    }

    fn attaches_mounts_at_create(&self) -> bool {
        self.inner.attaches_mounts_at_create()
    }

    fn resolves_environment(&self) -> bool {
        self.inner.resolves_environment()
    }

    fn mount_transition(&self) -> Option<&crate::sandbox::mounts::MountTransition> {
        self.inner.mount_transition()
    }

    fn authorize_mount_activation(
        &self,
        mount: &crate::sandbox::mounts::Mount,
        mount_path: &str,
    ) -> SandboxResult<()> {
        self.inner.authorize_mount_activation(mount, mount_path)
    }

    fn normalize_path(&self, path: &str) -> SandboxResult<String> {
        self.inner.normalize_path(path)
    }

    fn resolve_internal_path(&self, path: &str) -> SandboxResult<String> {
        self.inner.resolve_internal_path(path)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        self.inner.as_any()
    }

    fn describe(&self) -> String {
        self.inner.describe()
    }

    fn session_state(&self) -> crate::sandbox::state::SandboxSessionState {
        self.inner.session_state()
    }

    fn snapshot(&self) -> Arc<dyn crate::sandbox::snapshot::Snapshot> {
        self.inner.snapshot()
    }

    async fn start(&self) -> SandboxResult<()> {
        self.record(SandboxOp::Start, Vec::new(), self.inner.start())
            .await
    }

    async fn stop(&self) -> SandboxResult<()> {
        self.record(SandboxOp::Stop, Vec::new(), self.inner.stop())
            .await
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        let result = self
            .record(SandboxOp::Shutdown, Vec::new(), self.inner.shutdown())
            .await;
        // Nothing else will run against this session, so anything buffered is written now.
        self.instrumentation.flush().await;
        result
    }

    async fn running(&self) -> SandboxResult<bool> {
        self.inner.running().await
    }

    async fn exec(
        &self,
        request: crate::sandbox::session::ExecRequest,
    ) -> SandboxResult<crate::sandbox::types::ExecResult> {
        let argv = request.resolved_argv().join(" ");
        let started = self
            .instrumentation
            .start(SandboxOp::Exec, vec![("argv".to_string(), argv)])
            .await;
        let clock = std::time::Instant::now();
        let outcome = self.inner.exec(request).await;
        let elapsed = clock.elapsed().as_millis() as u64;
        match &outcome {
            Ok(result) => {
                self.instrumentation
                    .finish(&started, elapsed, Ok((&result.stdout, &result.stderr)))
                    .await
            }
            Err(error) => {
                self.instrumentation
                    .finish(&started, elapsed, Err(error))
                    .await
            }
        }
        outcome
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.record(
            SandboxOp::Read,
            vec![("path".to_string(), path.to_string())],
            self.inner.read_file(path),
        )
        .await
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let mut data = vec![("path".to_string(), path.to_string())];
        if self.instrumentation.policy().include_write_len {
            // The size, never the bytes: the contents are the agent's data.
            data.push(("bytes".to_string(), contents.len().to_string()));
        }
        self.record(
            SandboxOp::Write,
            data,
            self.inner.write_file(path, contents),
        )
        .await
    }

    async fn ls(&self, path: &str) -> SandboxResult<Vec<crate::sandbox::types::FileEntry>> {
        self.record(
            SandboxOp::Read,
            vec![("ls".to_string(), path.to_string())],
            self.inner.ls(path),
        )
        .await
    }

    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        self.record(
            SandboxOp::Write,
            vec![("mkdir".to_string(), path.to_string())],
            self.inner.mkdir(path, parents),
        )
        .await
    }

    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        self.record(
            SandboxOp::Write,
            vec![("rm".to_string(), path.to_string())],
            self.inner.rm(path, recursive),
        )
        .await
    }

    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        self.record(
            SandboxOp::Write,
            vec![
                ("mv".to_string(), source.to_string()),
                ("to".to_string(), destination.to_string()),
            ],
            self.inner.mv(source, destination),
        )
        .await
    }

    async fn pty_exec_start(
        &self,
        request: crate::sandbox::pty::PtyExecRequest,
    ) -> SandboxResult<crate::sandbox::pty::PtyExecUpdate> {
        let argv = request.resolved_argv().join(" ");
        self.record(
            SandboxOp::Exec,
            vec![("pty_argv".to_string(), argv)],
            self.inner.pty_exec_start(request),
        )
        .await
    }

    async fn pty_write_stdin(
        &self,
        request: crate::sandbox::pty::PtyWriteRequest,
    ) -> SandboxResult<crate::sandbox::pty::PtyExecUpdate> {
        let session = request.session_id.to_string();
        self.record(
            SandboxOp::Exec,
            vec![("pty_session".to_string(), session)],
            self.inner.pty_write_stdin(request),
        )
        .await
    }

    async fn pty_terminate_all(&self) -> SandboxResult<()> {
        self.inner.pty_terminate_all().await
    }

    async fn read_file_as(
        &self,
        path: &str,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<Vec<u8>> {
        // Delegated rather than inherited: the trait default reads through `exec`, which this
        // wrapper records in full — and for a write that would put the file's own bytes in the
        // audit log, which is exactly what the payload policy promises never to do. Delegating
        // also keeps any native implementation a backend provides.
        self.record(
            SandboxOp::Read,
            vec![("path".to_string(), path.to_string())],
            self.inner.read_file_as(path, user),
        )
        .await
    }

    async fn write_file_as(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        let mut data = vec![("path".to_string(), path.to_string())];
        if self.instrumentation.policy().include_write_len {
            data.push(("bytes".to_string(), contents.len().to_string()));
        }
        self.record(
            SandboxOp::Write,
            data,
            self.inner.write_file_as(path, contents, user),
        )
        .await
    }

    async fn mkdir_as(
        &self,
        path: &str,
        parents: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        self.record(
            SandboxOp::Write,
            vec![("mkdir".to_string(), path.to_string())],
            self.inner.mkdir_as(path, parents, user),
        )
        .await
    }

    async fn rm_as(
        &self,
        path: &str,
        recursive: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        self.record(
            SandboxOp::Write,
            vec![("rm".to_string(), path.to_string())],
            self.inner.rm_as(path, recursive, user),
        )
        .await
    }

    async fn mv_as(
        &self,
        source: &str,
        destination: &str,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        self.record(
            SandboxOp::Write,
            vec![
                ("mv".to_string(), source.to_string()),
                ("to".to_string(), destination.to_string()),
            ],
            self.inner.mv_as(source, destination, user),
        )
        .await
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        self.record(
            SandboxOp::PersistWorkspace,
            Vec::new(),
            self.inner.persist_workspace(),
        )
        .await
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.record(
            SandboxOp::HydrateWorkspace,
            vec![("bytes".to_string(), archive.len().to_string())],
            self.inner.hydrate_workspace(archive),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct Recorder {
        events: std::sync::Mutex<Vec<SandboxSessionEvent>>,
    }

    #[async_trait]
    impl EventSink for Recorder {
        async fn emit(&self, event: &SandboxSessionEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    #[tokio::test]
    async fn records_a_start_and_finish_pair_per_operation() {
        let recorder = Arc::new(Recorder::default());
        let instrumentation = Instrumentation::new(recorder.clone(), "sess", "unix_local");

        let started = instrumentation
            .start(
                SandboxOp::Exec,
                vec![("argv".to_string(), "ls -la".to_string())],
            )
            .await;
        instrumentation
            .finish(&started, 12, Ok((b"out", b"")))
            .await;

        let events = recorder.events.lock().unwrap().clone();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].phase, EventPhase::Start);
        assert_eq!(events[0].op, "exec");
        assert_eq!(events[0].data[0].1, "ls -la");
        assert_eq!(events[1].phase, EventPhase::Finish);
        assert_eq!(events[1].ok, Some(true));
        assert_eq!(events[1].duration_ms, Some(12));
        // Start and finish share a span id, so they can be paired after the fact.
        assert_eq!(events[0].span_id, events[1].span_id);
    }

    #[tokio::test]
    async fn command_output_is_withheld_unless_the_policy_asks_for_it() {
        let recorder = Arc::new(Recorder::default());
        let instrumentation = Instrumentation::new(recorder.clone(), "sess", "unix_local");
        let started = instrumentation.start(SandboxOp::Exec, Vec::new()).await;
        instrumentation
            .finish(&started, 1, Ok((b"SECRET", b"also secret")))
            .await;
        let events = recorder.events.lock().unwrap().clone();
        assert_eq!(events[1].stdout, None);
        assert_eq!(events[1].stderr, None);

        // Turned on, it is still bounded.
        let recorder = Arc::new(Recorder::default());
        let instrumentation = Instrumentation::new(recorder.clone(), "sess", "unix_local")
            .with_policy(
                EventPayloadPolicy::new()
                    .with_exec_output(true)
                    .with_output_limits(3, 2),
            );
        let started = instrumentation.start(SandboxOp::Exec, Vec::new()).await;
        instrumentation
            .finish(&started, 1, Ok((b"abcdef", b"xyz")))
            .await;
        let events = recorder.events.lock().unwrap().clone();
        assert_eq!(events[1].stdout.as_deref(), Some("abc"));
        assert_eq!(events[1].stderr.as_deref(), Some("xy"));
    }

    #[tokio::test]
    async fn a_failure_records_its_code_and_retry_classification() {
        let recorder = Arc::new(Recorder::default());
        let instrumentation = Instrumentation::new(recorder.clone(), "sess", "docker");
        let started = instrumentation.start(SandboxOp::Write, Vec::new()).await;
        let error = SandboxError::new(ErrorCode::ExecTransportError, "reset").with_retryable(true);
        instrumentation.finish(&started, 5, Err(&error)).await;
        let events = recorder.events.lock().unwrap().clone();
        assert_eq!(events[1].ok, Some(false));
        assert_eq!(
            events[1].error_code.as_deref(),
            Some("exec_transport_error")
        );
        assert_eq!(events[1].error_retryable, Some(true));
    }

    #[tokio::test]
    async fn a_chain_reaches_every_sink_and_jsonl_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let recorder = Arc::new(Recorder::default());
        let chain = ChainedSink::new()
            .with_sink(recorder.clone())
            .with_sink(Arc::new(JsonlFileSink::new(&path)));

        let event = SandboxSessionEvent::start("sess", "unix_local", 0, "span", SandboxOp::Read)
            .with_data("path", "/workspace/a.txt");
        chain.emit(&event).await;
        chain.flush().await;

        assert_eq!(recorder.events.lock().unwrap().len(), 1);
        let written = tokio::fs::read_to_string(&path).await.unwrap();
        let parsed: SandboxSessionEvent = serde_json::from_str(written.trim()).unwrap();
        assert_eq!(parsed, event);
    }
}
