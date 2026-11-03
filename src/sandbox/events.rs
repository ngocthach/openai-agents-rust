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

    /// Record how many bytes a write carried (default: true). The bytes themselves never are.
    pub fn with_write_len(mut self, include: bool) -> Self {
        self.include_write_len = include;
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

/// The id of the trace this task is running inside, when tracing is active.
///
/// Mirrors how Python stamps `trace_id` onto every sandbox event: a sandbox event stream is only
/// useful next to the run that produced it, and the trace id is the key that joins them.
pub fn current_trace_id() -> Option<String> {
    // A disabled parent span means this operation is not part of any exported trace, so there is
    // no id that would join it to one.
    if crate::tracing::scope::current_span().is_some_and(|span| span.is_disabled()) {
        return None;
    }
    crate::tracing::current_trace()
        .filter(|trace| !trace.is_disabled())
        .map(|trace| trace.trace_id().to_string())
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
    /// The SDK trace this operation belongs to, when the run is traced.
    ///
    /// Mirrors Python's `SandboxSessionEvent.trace_id`: it is what lines a sandbox event stream
    /// up with the model calls and tool spans of the run that produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
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
    /// The raw standard output of the command, while the event is on its way to the sinks.
    ///
    /// Each sink may carry its own payload policy, so the bytes travel with the event until the
    /// last decision is made. They are never serialized and never reach a sink: delivery clears
    /// them after applying that sink's policy.
    #[serde(skip)]
    pub stdout_bytes: Option<Vec<u8>>,
    /// The raw standard error of the command. See [`stdout_bytes`](Self::stdout_bytes).
    #[serde(skip)]
    pub stderr_bytes: Option<Vec<u8>>,
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
            trace_id: current_trace_id(),
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
            stdout_bytes: None,
            stderr_bytes: None,
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

    /// Attach the trace this operation belongs to.
    pub fn with_trace_id(mut self, trace_id: Option<String>) -> Self {
        self.trace_id = trace_id;
        self
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

    /// Carry the raw command output with the event, for per-sink payload policies.
    pub fn with_raw_output(mut self, stdout: &[u8], stderr: &[u8]) -> Self {
        self.stdout_bytes = Some(stdout.to_vec());
        self.stderr_bytes = Some(stderr.to_vec());
        self
    }

    /// Decide what this event shows of the command output under `policy`, then drop the raw
    /// bytes. What the policy withholds is not recoverable from the event afterwards.
    pub fn apply_payload_policy(mut self, policy: EventPayloadPolicy) -> Self {
        if self.stdout_bytes.is_some() || self.stderr_bytes.is_some() {
            let stdout = self.stdout_bytes.take().unwrap_or_default();
            let stderr = self.stderr_bytes.take().unwrap_or_default();
            self.stdout = policy.apply(&stdout, policy.max_stdout_chars);
            self.stderr = policy.apply(&stderr, policy.max_stderr_chars);
        }
        self.stdout_bytes = None;
        self.stderr_bytes = None;
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

    /// Hand the sink the session whose events it receives. Mirrors Python's
    /// `EventSink.bind`; [`InstrumentedSession::new`] calls it once. The default ignores it.
    fn bind(&self, _session: Arc<dyn crate::sandbox::session::SandboxSession>) {}

    /// The payload policy this sink wants, overriding the instrumentation's own.
    ///
    /// Mirrors Python's per-sink `payload_policy`: a sink that ships events off the machine may
    /// want no command output while a local debugging sink wants all of it. `None` defers to the
    /// per-operation policy and then to the default.
    fn payload_policy(&self) -> Option<EventPayloadPolicy> {
        None
    }

    /// Whether this sink hands events to others that each apply their own payload policy. A
    /// chain does; the instrumentation then leaves the raw output on the event for it.
    fn delivers_own_policies(&self) -> bool {
        false
    }

    /// Whether a failed delivery fails the operation the event describes. Mirrors Python's
    /// `on_error="raise"`. The default is false: a sink that only logs cannot break a run.
    fn raises_on_error(&self) -> bool {
        false
    }

    /// Deliver one event, reporting a failure. Only consulted when
    /// [`raises_on_error`](Self::raises_on_error) is true.
    async fn try_emit(&self, event: &SandboxSessionEvent) -> SandboxResult<()> {
        self.emit(event).await;
        Ok(())
    }
}

/// The closure a [`CallbackSink`] runs for each event, with the bound session when there is one.
type EventCallback = dyn Fn(
        &SandboxSessionEvent,
        Option<&Arc<dyn crate::sandbox::session::SandboxSession>>,
    ) -> SandboxResult<()>
    + Send
    + Sync;

/// Hands each event to a closure. Mirrors Python's `CallbackSink`.
pub struct CallbackSink {
    callback: Arc<EventCallback>,
    raises: bool,
    policy: Option<EventPayloadPolicy>,
    session: std::sync::Mutex<Option<Arc<dyn crate::sandbox::session::SandboxSession>>>,
}

impl CallbackSink {
    fn build(callback: Arc<EventCallback>, raises: bool) -> Self {
        CallbackSink {
            callback,
            raises,
            policy: None,
            session: std::sync::Mutex::new(None),
        }
    }

    /// A sink calling `callback` for every event. A failure in `callback` cannot fail the
    /// operation; use [`raising`](Self::raising) for that.
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(&SandboxSessionEvent) + Send + Sync + 'static,
    {
        Self::build(
            Arc::new(move |event, _| {
                callback(event);
                Ok(())
            }),
            false,
        )
    }

    /// A sink whose callback may fail. A failure fails the operation that produced the event,
    /// as Python's `CallbackSink(..., on_error="raise")` does.
    pub fn raising<F>(callback: F) -> Self
    where
        F: Fn(&SandboxSessionEvent) -> SandboxResult<()> + Send + Sync + 'static,
    {
        Self::build(Arc::new(move |event, _| callback(event)), true)
    }

    /// Like [`new`](Self::new), for a callback that also receives the session the event came
    /// from, as Python's `CallbackSink` callback does. The session is bound when the sink is
    /// handed to an [`InstrumentedSession`]; an event delivered before that fails with
    /// `CallbackSink requires a bound session`, which a non-raising sink only logs.
    pub fn with_session<F>(callback: F) -> Self
    where
        F: Fn(&SandboxSessionEvent, &Arc<dyn crate::sandbox::session::SandboxSession>)
            + Send
            + Sync
            + 'static,
    {
        Self::build(
            Arc::new(move |event, session| {
                callback(event, Self::require_session(session)?);
                Ok(())
            }),
            false,
        )
    }

    /// [`raising`](Self::raising) for a callback that also receives the bound session. See
    /// [`with_session`](Self::with_session).
    pub fn raising_with_session<F>(callback: F) -> Self
    where
        F: Fn(
                &SandboxSessionEvent,
                &Arc<dyn crate::sandbox::session::SandboxSession>,
            ) -> SandboxResult<()>
            + Send
            + Sync
            + 'static,
    {
        Self::build(
            Arc::new(move |event, session| callback(event, Self::require_session(session)?)),
            true,
        )
    }

    fn require_session(
        session: Option<&Arc<dyn crate::sandbox::session::SandboxSession>>,
    ) -> SandboxResult<&Arc<dyn crate::sandbox::session::SandboxSession>> {
        session.ok_or_else(|| {
            SandboxError::new(
                ErrorCode::EventSinkFailed,
                "CallbackSink requires a bound session; wrap the session in an \
                 InstrumentedSession (or call bind)",
            )
        })
    }

    fn call(&self, event: &SandboxSessionEvent) -> SandboxResult<()> {
        let session = self.session.lock().expect("poisoned").clone();
        (self.callback)(event, session.as_ref())
    }

    /// Give this sink its own payload policy.
    pub fn with_payload_policy(mut self, policy: EventPayloadPolicy) -> Self {
        self.policy = Some(policy);
        self
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
        if let Err(error) = self.call(event) {
            // The failure's own text may repeat what the event described, so it is logged only
            // when tool data may be logged; the sink and the event stay out of the message.
            crate::logger::log_tool_action_error("Sandbox event sink failed (ignored)", &error);
        }
    }

    async fn try_emit(&self, event: &SandboxSessionEvent) -> SandboxResult<()> {
        self.call(event)
    }

    fn bind(&self, session: Arc<dyn crate::sandbox::session::SandboxSession>) {
        *self.session.lock().expect("poisoned") = Some(session);
    }

    fn raises_on_error(&self) -> bool {
        self.raises
    }

    fn payload_policy(&self) -> Option<EventPayloadPolicy> {
        self.policy
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
                    log::warn!(target: crate::logger::LOG_TARGET,
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
            log::warn!(target: crate::logger::LOG_TARGET,
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

/// POSTs every event as JSON to an endpoint. Mirrors Python's `HttpProxySink`.
///
/// A log that lives on the machine running the sandbox is lost with it. This is the sink for the
/// case where the audit trail has to outlive the host: a local collector daemon or a remote
/// service. A failed POST falls back to `spool_path`, so a collector that is down loses nothing
/// more than the ordering.
#[derive(Debug)]
pub struct HttpProxySink {
    endpoint: String,
    headers: Vec<(String, String)>,
    timeout: std::time::Duration,
    spool: Option<Arc<JsonlFileSink>>,
    client: reqwest::Client,
}

impl HttpProxySink {
    /// A sink posting to `endpoint`.
    pub fn new(endpoint: impl Into<String>) -> Self {
        HttpProxySink {
            endpoint: endpoint.into(),
            headers: Vec::new(),
            timeout: std::time::Duration::from_secs(5),
            spool: None,
            client: reqwest::Client::new(),
        }
    }

    /// Send `value` as `name` on every request.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// How long one POST may take. The default is five seconds, as in Python.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Append an event to `path` when its POST fails.
    pub fn with_spool_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.spool = Some(Arc::new(JsonlFileSink::new(path)));
        self
    }

    /// Reuse an existing HTTP client instead of building one.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

#[async_trait]
impl EventSink for HttpProxySink {
    async fn emit(&self, event: &SandboxSessionEvent) {
        let Ok(body) = serde_json::to_vec(event) else {
            return;
        };
        let mut request = self
            .client
            .post(&self.endpoint)
            .header("content-type", "application/json")
            .timeout(self.timeout)
            .body(body);
        for (name, value) in &self.headers {
            request = request.header(name, value);
        }
        let delivered =
            matches!(request.send().await, Ok(response) if response.status().is_success());
        if !delivered && let Some(spool) = &self.spool {
            spool.emit(event).await;
        }
    }

    async fn flush(&self) {
        if let Some(spool) = &self.spool {
            spool.flush().await;
        }
    }
}

/// Appends events to a JSON Lines file **inside** the sandbox workspace. Mirrors Python's
/// `WorkspaceJsonlSink`.
///
/// The agent can read, rewrite and delete this file, so it is a convenience for the workload —
/// a tool that wants to see what it has done — and never an authoritative audit record. Use
/// [`JsonlFileSink`] or [`HttpProxySink`] when the log has to be trusted. The sink buffers and
/// rewrites the whole file, so `max_bytes` bounds both the buffer and the file it reads back;
/// exceeding either stops the sink permanently rather than truncating what is already there.
#[derive(Debug)]
pub struct WorkspaceJsonlSink {
    session: Arc<dyn crate::sandbox::session::SandboxSession>,
    workspace_relpath: String,
    flush_every: usize,
    max_bytes: Option<usize>,
    state: Mutex<WorkspaceSinkState>,
}

#[derive(Debug, Default)]
struct WorkspaceSinkState {
    buffer: String,
    seen: usize,
    disabled: bool,
}

impl WorkspaceJsonlSink {
    /// A sink appending to `workspace_relpath` inside `session`'s workspace.
    ///
    /// `{session_id}` in the path is expanded on construction, as Python expands it on `bind`.
    pub fn new(
        session: Arc<dyn crate::sandbox::session::SandboxSession>,
        workspace_relpath: impl Into<String>,
    ) -> Self {
        let workspace_relpath = workspace_relpath
            .into()
            .replace("{session_id_hex}", &session.id().replace('-', ""))
            .replace("{session_id}", session.id());
        WorkspaceJsonlSink {
            session,
            workspace_relpath,
            flush_every: 1,
            max_bytes: None,
            state: Mutex::new(WorkspaceSinkState::default()),
        }
    }

    /// Write to the workspace only every `events` events.
    pub fn with_flush_every(mut self, events: usize) -> Self {
        self.flush_every = events.max(1);
        self
    }

    /// Stop the sink once the outbox or the pending buffer would exceed `max_bytes`.
    pub fn with_max_bytes(mut self, max_bytes: Option<usize>) -> SandboxResult<Self> {
        if max_bytes == Some(0) {
            return Err(SandboxError::new(
                ErrorCode::SandboxConfigInvalid,
                "WorkspaceJsonlSink max_bytes must be positive",
            ));
        }
        self.max_bytes = max_bytes;
        Ok(self)
    }

    /// The workspace-relative path being written.
    pub fn workspace_relpath(&self) -> &str {
        &self.workspace_relpath
    }

    /// Keep the outbox out of workspace archives, so it is a runtime artifact rather than
    /// durable workspace content (default: false).
    ///
    /// Mirrors Python's `ephemeral`. The path is registered with the session, which refuses one
    /// that overlaps a mount.
    pub fn with_ephemeral(self, ephemeral: bool) -> SandboxResult<Self> {
        if ephemeral {
            self.session
                .register_persist_workspace_skip_path(&self.workspace_relpath)?;
        }
        Ok(self)
    }

    /// Whether an event should trigger a write.
    fn should_flush(&self, event: &SandboxSessionEvent, seen: usize) -> bool {
        if seen.is_multiple_of(self.flush_every) {
            return true;
        }
        // These are the last chance to get the buffer out: the workspace is about to be
        // archived, stopped or removed.
        matches!(
            (event.op.as_str(), event.phase),
            ("persist_workspace", EventPhase::Start)
                | ("stop", _)
                | ("shutdown", EventPhase::Start)
        )
    }

    /// The error that ends delivery for good. Nothing is truncated or rotated: a bounded sink
    /// that cannot hold what is there stops rather than discarding the workload's own data.
    fn stopped_at_limit(&self, state: &mut WorkspaceSinkState) -> SandboxError {
        state.disabled = true;
        state.buffer.clear();
        SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            format!(
                "WorkspaceJsonlSink exceeded max_bytes={}; delivery stopped. Use a new outbox and \
                 sink, or JsonlFileSink/HttpProxySink.",
                self.max_bytes
                    .map(|max| max.to_string())
                    .unwrap_or_else(|| "unbounded".to_string())
            ),
        )
        .with_retryable(false)
    }

    /// The outbox as it is now, read no further than the budget allows.
    ///
    /// The outbox is workload-modifiable, so a bounded sink bounds the read as well as the
    /// write. One byte over the budget proves it was exceeded without pulling in the rest. A
    /// missing outbox is empty; a read that fails for any other reason fails the delivery and
    /// leaves the pending events where they are, for the next attempt.
    async fn read_existing_outbox(&self, state: &mut WorkspaceSinkState) -> SandboxResult<Vec<u8>> {
        let read = match self.max_bytes {
            None => self.session.read_file(&self.workspace_relpath).await,
            Some(max_bytes) => {
                self.session
                    .read_bounded(&self.workspace_relpath, max_bytes + 1)
                    .await
            }
        };
        match read {
            Ok(bytes) => Ok(bytes),
            Err(error) if error.code == ErrorCode::WorkspaceReadNotFound => Ok(Vec::new()),
            Err(error)
                if error.context.get("reason").map(String::as_str)
                    == Some("bounded_read_wire_limit") =>
            {
                Err(self.stopped_at_limit(state))
            }
            Err(error) => Err(error),
        }
    }

    async fn flush_buffer(&self, state: &mut WorkspaceSinkState) -> SandboxResult<()> {
        if state.buffer.is_empty() {
            return Ok(());
        }
        let existing = self.read_existing_outbox(state).await?;
        if let Some(max_bytes) = self.max_bytes
            && existing.len() + state.buffer.len() > max_bytes
        {
            return Err(self.stopped_at_limit(state));
        }
        let mut payload = existing;
        payload.extend_from_slice(state.buffer.as_bytes());
        self.session
            .write_file(&self.workspace_relpath, &payload)
            .await?;
        state.buffer.clear();
        Ok(())
    }

    /// Record one event, reporting why delivery failed.
    ///
    /// [`EventSink::emit`] calls this and logs the failure, because a sink must not fail the
    /// operation it describes. A caller that wants Python's `on_error="raise"` behaviour calls
    /// this directly. A failed delivery keeps the pending events for the next attempt, except
    /// when the budget was exceeded: that clears them and stops the sink for good.
    pub async fn try_emit(&self, event: &SandboxSessionEvent) -> SandboxResult<()> {
        let line = event.to_json_line()?;
        let mut state = self.state.lock().await;
        if state.disabled {
            return Ok(());
        }
        if let Some(max_bytes) = self.max_bytes
            && state.buffer.len() + line.len() > max_bytes
        {
            return Err(self.stopped_at_limit(&mut state));
        }
        state.buffer.push_str(&line);
        state.seen += 1;
        if !self.should_flush(event, state.seen) {
            return Ok(());
        }
        // `start` emits before the sandbox is fully running, so writes may still fail during
        // early startup or late teardown: the buffer waits for a later event.
        if !matches!(self.session.running().await, Ok(true)) {
            return Ok(());
        }
        self.flush_buffer(&mut state).await
    }

    /// The events waiting to be written.
    pub async fn pending_bytes(&self) -> usize {
        self.state.lock().await.buffer.len()
    }
}

#[async_trait]
impl EventSink for WorkspaceJsonlSink {
    async fn emit(&self, event: &SandboxSessionEvent) {
        if let Err(error) = self.try_emit(event).await {
            // Only the code and the message: the context can hold the workload's own bytes.
            log::warn!(target: crate::logger::LOG_TARGET,
                "WorkspaceJsonlSink delivery failed ({}): {}", error.code, error.message
            );
        }
    }

    async fn flush(&self) {
        let mut state = self.state.lock().await;
        if state.disabled {
            return;
        }
        if let Err(error) = self.flush_buffer(&mut state).await {
            log::warn!(target: crate::logger::LOG_TARGET,
                "WorkspaceJsonlSink flush failed ({}): {}", error.code, error.message
            );
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
        // In order, each completing before the next starts, so a later sink can rely on an
        // earlier one having finished with the event.
        for sink in &self.sinks {
            let view = match sink.payload_policy() {
                Some(policy) => event.clone().apply_payload_policy(policy),
                None => {
                    let mut stripped = event.clone();
                    stripped.stdout_bytes = None;
                    stripped.stderr_bytes = None;
                    stripped
                }
            };
            sink.emit(&view).await;
        }
    }

    fn delivers_own_policies(&self) -> bool {
        true
    }

    async fn flush(&self) {
        for sink in &self.sinks {
            sink.flush().await;
        }
    }

    fn bind(&self, session: Arc<dyn crate::sandbox::session::SandboxSession>) {
        for sink in &self.sinks {
            sink.bind(session.clone());
        }
    }
}

/// Issues the start and finish events for a session's operations. Mirrors Python's
/// `Instrumentation`.
#[derive(Debug)]
pub struct Instrumentation {
    sink: Arc<dyn EventSink>,
    policy: EventPayloadPolicy,
    policy_by_op: Vec<(SandboxOp, EventPayloadPolicy)>,
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
            policy_by_op: Vec::new(),
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

    /// Use `policy` for `op` instead of the default.
    ///
    /// Mirrors Python's `payload_policy_by_op`. The mapping is taken by value, so changing the
    /// caller's copy afterwards does not change what is recorded.
    pub fn with_policy_for_op(mut self, op: SandboxOp, policy: EventPayloadPolicy) -> Self {
        self.policy_by_op.retain(|(existing, _)| *existing != op);
        self.policy_by_op.push((op, policy));
        self
    }

    /// The payload policy in force.
    pub fn policy(&self) -> EventPayloadPolicy {
        self.policy
    }

    /// The policy that applies to events of `op`: its own, else the default.
    pub fn policy_for_op(&self, op: &str) -> EventPayloadPolicy {
        self.policy_by_op
            .iter()
            .find(|(candidate, _)| candidate.as_str() == op)
            .map(|(_, policy)| *policy)
            .unwrap_or(self.policy)
    }

    /// Deliver one event to the sinks.
    ///
    /// The policy decides what the event shows of the command output: the sink's own when it
    /// has one, else the operation's, else the default. A chain applies its members' own
    /// policies as it delivers to each. The raw bytes the event carried never reach a sink.
    pub async fn emit(&self, event: SandboxSessionEvent) {
        if let Err(error) = self.deliver(event).await {
            // Neither the sink nor the event is named: both can describe the workload. The
            // sink's own failure text appears only where tool data may be logged.
            match error.context.get("cause") {
                Some(cause) => crate::logger::log_tool_action_error(
                    "Sandbox event sink failed (ignored)",
                    cause,
                ),
                None => crate::logger::log_tool_action_error(
                    "Sandbox event sink failed (ignored)",
                    &error.code,
                ),
            }
        }
    }

    /// Deliver one event, reporting a failure from a sink that raises. See
    /// [`EventSink::raises_on_error`].
    pub async fn deliver(&self, event: SandboxSessionEvent) -> SandboxResult<()> {
        let operation_policy = self.policy_for_op(&event.op);
        if self.sink.delivers_own_policies() {
            // Members that have no policy of their own see what the operation's policy shows;
            // the raw bytes stay on the event only until each member has taken its view.
            let mut shown = event.clone().apply_payload_policy(operation_policy);
            shown.stdout_bytes = event.stdout_bytes;
            shown.stderr_bytes = event.stderr_bytes;
            return self.send(&shown).await;
        }
        let policy = self.sink.payload_policy().unwrap_or(operation_policy);
        self.send(&event.apply_payload_policy(policy)).await
    }

    async fn send(&self, event: &SandboxSessionEvent) -> SandboxResult<()> {
        if !self.sink.raises_on_error() {
            self.sink.emit(event).await;
            return Ok(());
        }
        self.sink.try_emit(event).await.map_err(|cause| {
            let error = SandboxError::new(ErrorCode::EventSinkFailed, "sandbox event sink failed");
            // The sink's text can repeat what the event described, so it travels with the
            // error only where tool data may be logged.
            if crate::debug::dont_log_tool_data() {
                error
            } else {
                error.with_context("cause", cause.to_string())
            }
        })
    }

    /// Emit the start event for one operation and return it, for the matching finish.
    pub async fn start(&self, op: SandboxOp, data: Vec<(String, String)>) -> SandboxSessionEvent {
        let event = self.start_event(op, data);
        self.emit(event.clone()).await;
        event
    }

    /// Like [`start`](Self::start), but a sink that raises fails the operation before it runs.
    pub async fn try_start(
        &self,
        op: SandboxOp,
        data: Vec<(String, String)>,
    ) -> SandboxResult<SandboxSessionEvent> {
        let event = self.start_event(op, data);
        self.deliver(event.clone()).await?;
        Ok(event)
    }

    fn start_event(&self, op: SandboxOp, data: Vec<(String, String)>) -> SandboxSessionEvent {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let mut event = SandboxSessionEvent::start(
            self.session_id.clone(),
            self.backend.clone(),
            seq,
            format!("{}-{seq}", self.session_id),
            op,
        );
        event.data = data;
        event
    }

    /// Emit the finish event for an operation started with [`start`](Self::start).
    pub async fn finish(
        &self,
        started: &SandboxSessionEvent,
        duration_ms: u64,
        outcome: Result<(&[u8], &[u8]), &SandboxError>,
    ) {
        let event = self.finish_event(started, duration_ms, outcome);
        self.emit(event).await;
    }

    /// Like [`finish`](Self::finish), but a sink that raises reports its failure.
    pub async fn try_finish(
        &self,
        started: &SandboxSessionEvent,
        duration_ms: u64,
        outcome: Result<(&[u8], &[u8]), &SandboxError>,
    ) -> SandboxResult<()> {
        let event = self.finish_event(started, duration_ms, outcome);
        self.deliver(event).await
    }

    fn finish_event(
        &self,
        started: &SandboxSessionEvent,
        duration_ms: u64,
        outcome: Result<(&[u8], &[u8]), &SandboxError>,
    ) -> SandboxSessionEvent {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let mut event = match outcome {
            Ok((stdout, stderr)) => started
                .finish(true, duration_ms)
                .with_raw_output(stdout, stderr),
            Err(error) => started.finish(false, duration_ms).with_error(error),
        };
        event.seq = seq;
        event
    }

    /// Flush the sink.
    pub async fn flush(&self) {
        self.sink.flush().await;
    }
}

/// The span name for an operation, `sandbox.<op>` as Python names them.
fn span_name(op: SandboxOp) -> &'static str {
    match op {
        SandboxOp::Start => "sandbox.start",
        SandboxOp::Stop => "sandbox.stop",
        SandboxOp::Exec => "sandbox.exec",
        SandboxOp::Read => "sandbox.read",
        SandboxOp::Write => "sandbox.write",
        SandboxOp::Shutdown => "sandbox.shutdown",
        SandboxOp::Running => "sandbox.running",
        SandboxOp::PersistWorkspace => "sandbox.persist_workspace",
        SandboxOp::HydrateWorkspace => "sandbox.hydrate_workspace",
        SandboxOp::ResolveExposedPort => "sandbox.resolve_exposed_port",
        SandboxOp::Materialize => "sandbox.materialize",
        SandboxOp::SnapshotPersist => "sandbox.snapshot_persist",
        SandboxOp::SnapshotRestore => "sandbox.snapshot_restore",
        SandboxOp::ApplyPatch => "sandbox.apply_patch",
    }
}

/// The span data for an operation's recorded metadata.
fn span_data(data: &[(String, String)]) -> serde_json::Value {
    serde_json::Value::Object(
        data.iter()
            .map(|(key, value)| (key.clone(), serde_json::Value::String(value.clone())))
            .collect(),
    )
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
        sink.bind(inner.clone());
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
        let started = self.instrumentation.try_start(op, data.clone()).await?;
        let clock = std::time::Instant::now();
        crate::sandbox::runtime::sandbox_span_result(span_name(op), span_data(&data), async {
            let outcome = operation.await;
            self.finish_recorded(&started, clock, &outcome)
                .await
                .and(outcome)
        })
        .await
    }

    /// Record the finish of an operation. A sink that raises reports its failure here, and that
    /// failure replaces the operation's outcome, as in Python.
    async fn finish_recorded<T>(
        &self,
        started: &SandboxSessionEvent,
        clock: std::time::Instant,
        outcome: &SandboxResult<T>,
    ) -> SandboxResult<()> {
        let elapsed = clock.elapsed().as_millis() as u64;
        match outcome {
            Ok(_) => {
                self.instrumentation
                    .try_finish(started, elapsed, Ok((&[], &[])))
                    .await
            }
            Err(error) => {
                self.instrumentation
                    .try_finish(started, elapsed, Err(error))
                    .await
            }
        }
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

    fn live_grants(&self) -> Option<&crate::sandbox::session::LiveGrants> {
        self.inner.live_grants()
    }

    fn shutdown_after_failed_close(&self) -> bool {
        self.inner.shutdown_after_failed_close()
    }

    fn extra_path_grants(&self) -> Vec<crate::sandbox::workspace_paths::SandboxPathGrant> {
        self.inner.extra_path_grants()
    }

    fn update_extra_path_grants(
        &self,
        grants: &[crate::sandbox::workspace_paths::SandboxPathGrant],
    ) {
        self.inner.update_extra_path_grants(grants)
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
        let data = vec![("argv".to_string(), argv)];
        let started = self
            .instrumentation
            .start(SandboxOp::Exec, data.clone())
            .await;
        let clock = std::time::Instant::now();
        let outcome = crate::sandbox::runtime::sandbox_span_result(
            span_name(SandboxOp::Exec),
            span_data(&data),
            self.inner.exec(request),
        )
        .await;
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
            data.push(("bytes".to_string(), contents.len().to_string()));
        }
        self.record(
            SandboxOp::Write,
            data,
            self.inner.write_file(path, contents),
        )
        .await
    }

    async fn pty_exec_start(
        &self,
        request: crate::sandbox::pty::PtyExecRequest,
    ) -> SandboxResult<crate::sandbox::pty::PtyExecUpdate> {
        self.inner.pty_exec_start(request).await
    }

    async fn pty_write_stdin(
        &self,
        request: crate::sandbox::pty::PtyWriteRequest,
    ) -> SandboxResult<crate::sandbox::pty::PtyExecUpdate> {
        self.inner.pty_write_stdin(request).await
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

    async fn read_file_expecting(
        &self,
        path: &str,
        user: Option<&crate::sandbox::types::User>,
        expected: &[ErrorCode],
    ) -> SandboxResult<Vec<u8>> {
        let data = vec![("path".to_string(), path.to_string())];
        let started = self
            .instrumentation
            .try_start(SandboxOp::Read, data.clone())
            .await?;
        let clock = std::time::Instant::now();
        // The finish is recorded inside the span, so a sink failure marks the span even when the
        // read's own error was expected.
        crate::sandbox::runtime::sandbox_span_result_expecting(
            span_name(SandboxOp::Read),
            span_data(&data),
            expected,
            async {
                let outcome = self.inner.read_file_as(path, user).await;
                self.finish_recorded(&started, clock, &outcome)
                    .await
                    .and(outcome)
            },
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
        // Before any archive work, as Python's session wrapper does: an unsafe mount is refused
        // without a byte of the workspace being captured.
        self.inner
            .manifest()
            .validate_mount_credential_boundaries()?;
        self.record(
            SandboxOp::PersistWorkspace,
            Vec::new(),
            self.inner.persist_workspace(),
        )
        .await
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner
            .manifest()
            .validate_mount_credential_boundaries()?;
        self.record(
            SandboxOp::HydrateWorkspace,
            vec![("bytes".to_string(), archive.len().to_string())],
            self.inner.hydrate_workspace(archive),
        )
        .await
    }

    fn validate_restorable_archive(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.validate_restorable_archive(archive)
    }

    fn archive_limits(&self) -> Option<crate::sandbox::types::SandboxArchiveLimits> {
        self.inner.archive_limits()
    }

    fn set_archive_limits(&self, limits: Option<crate::sandbox::types::SandboxArchiveLimits>) {
        self.inner.set_archive_limits(limits);
    }

    fn set_concurrency_limits(&self, limits: crate::sandbox::manifest::SandboxConcurrencyLimits) {
        self.inner.set_concurrency_limits(limits);
    }

    async fn ls(&self, path: &str) -> SandboxResult<Vec<crate::sandbox::types::FileEntry>> {
        self.inner.ls(path).await
    }

    async fn ls_as(
        &self,
        path: &str,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<Vec<crate::sandbox::types::FileEntry>> {
        self.inner.ls_as(path, user).await
    }

    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        self.inner.mkdir(path, parents).await
    }

    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        self.inner.rm(path, recursive).await
    }

    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        self.inner.mv(source, destination).await
    }

    async fn check_mkdir_access(
        &self,
        path: &str,
        parents: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        self.inner.check_mkdir_access(path, parents, user).await
    }

    async fn check_read_access(
        &self,
        path: &str,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<String> {
        self.inner.check_read_access(path, user).await
    }

    async fn check_rm_access(
        &self,
        path: &str,
        recursive: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        self.inner.check_rm_access(path, recursive, user).await
    }

    async fn check_write_access(
        &self,
        path: &str,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        self.inner.check_write_access(path, user).await
    }

    fn computes_snapshot_fingerprint(&self) -> bool {
        self.inner.computes_snapshot_fingerprint()
    }

    fn dependencies(&self) -> Option<&Arc<crate::sandbox::dependencies::Dependencies>> {
        self.inner.dependencies()
    }

    fn ensure_mounts_settled(&self) -> SandboxResult<()> {
        self.inner.ensure_mounts_settled()
    }

    async fn ensure_runtime_helper_installed(
        &self,
        helper: &crate::sandbox::runtime_helpers::RuntimeHelperScript,
    ) -> SandboxResult<String> {
        self.inner.ensure_runtime_helper_installed(helper).await
    }

    async fn extract(
        &self,
        path: &str,
        data: &[u8],
        compression_scheme: Option<crate::sandbox::archive::CompressionScheme>,
    ) -> SandboxResult<()> {
        self.inner.extract(path, data, compression_scheme).await
    }

    async fn extract_with_limits(
        &self,
        path: &str,
        data: &[u8],
        compression_scheme: Option<crate::sandbox::archive::CompressionScheme>,
        limits: Option<crate::sandbox::types::SandboxArchiveLimits>,
    ) -> SandboxResult<()> {
        self.inner
            .extract_with_limits(path, data, compression_scheme, limits)
            .await
    }

    fn has_read_only_grant(&self) -> bool {
        self.inner.has_read_only_grant()
    }

    fn resolves_paths_remotely(&self) -> bool {
        self.inner.resolves_paths_remotely()
    }

    async fn validate_manifest_application(
        &self,
        manifest: &crate::sandbox::Manifest,
        session_running: Option<bool>,
    ) -> SandboxResult<()> {
        self.inner
            .validate_manifest_application(manifest, session_running)
            .await
    }

    fn normalize_path_for_write(&self, path: &str) -> SandboxResult<String> {
        self.inner.normalize_path_for_write(path)
    }

    async fn persist_workspace_once(&self) -> SandboxResult<Vec<u8>> {
        self.inner.persist_workspace_once().await
    }

    fn persist_workspace_skip_paths(&self) -> Vec<String> {
        self.inner.persist_workspace_skip_paths()
    }

    async fn probe_read_path(&self, path: &str) -> SandboxResult<bool> {
        self.inner.probe_read_path(path).await
    }

    async fn read_error_from_exec(
        &self,
        path: &str,
        result: &crate::sandbox::types::ExecResult,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxError {
        self.inner.read_error_from_exec(path, result, user).await
    }

    fn record_snapshot_fingerprint(
        &self,
        record: Option<crate::sandbox::state::SnapshotFingerprint>,
    ) {
        self.inner.record_snapshot_fingerprint(record);
    }

    fn resume_snapshot_fingerprint(&self) -> Option<crate::sandbox::state::SnapshotFingerprint> {
        self.inner.resume_snapshot_fingerprint()
    }

    async fn run_access_check(
        &self,
        script: &str,
        label: &str,
        args: &[&str],
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<()> {
        self.inner.run_access_check(script, label, args, user).await
    }

    fn system_state_preserved_on_start(&self) -> bool {
        self.inner.system_state_preserved_on_start()
    }

    async fn validate_remote_path_access(
        &self,
        path: &str,
        for_write: bool,
        recursive_remove: bool,
    ) -> SandboxResult<String> {
        self.inner
            .validate_remote_path_access(path, for_write, recursive_remove)
            .await
    }

    fn workspace_path_policy(
        &self,
    ) -> SandboxResult<crate::sandbox::workspace_paths::WorkspacePathPolicy> {
        self.inner.workspace_path_policy()
    }

    // Everything below is delegated for the same reason `read_file_as` is: the trait defaults
    // would run through this wrapper's `exec`, losing the backend's native implementation (an
    // exclusive create, an inode comparison, a bounded fetch) and, for a write, recording the
    // file's own bytes in the audit log.

    fn exposed_ports(&self) -> Vec<u16> {
        self.inner.exposed_ports()
    }

    async fn resolve_exposed_port(
        &self,
        port: u16,
    ) -> SandboxResult<crate::sandbox::types::ExposedPortEndpoint> {
        self.inner.resolve_exposed_port(port).await
    }

    fn persist_skip_registry(
        &self,
    ) -> Option<&std::sync::Mutex<std::collections::BTreeSet<String>>> {
        self.inner.persist_skip_registry()
    }

    fn register_persist_workspace_skip_path(&self, path: &str) -> SandboxResult<String> {
        self.inner.register_persist_workspace_skip_path(path)
    }

    async fn read_bounded(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        self.record(
            SandboxOp::Read,
            vec![("path".to_string(), path.to_string())],
            self.inner.read_bounded(path, max_bytes),
        )
        .await
    }

    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        self.inner.read_bounded_unchecked(path, max_bytes).await
    }

    async fn write_new_file(
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
            self.inner.write_new_file(path, contents, user),
        )
        .await
    }

    async fn same_file(
        &self,
        left: &str,
        right: &str,
        follow_symlinks: bool,
        user: Option<&crate::sandbox::types::User>,
    ) -> SandboxResult<bool> {
        self.inner
            .same_file(left, right, follow_symlinks, user)
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
