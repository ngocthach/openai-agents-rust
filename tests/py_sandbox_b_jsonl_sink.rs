//! Port of `tests/sandbox/test_workspace_jsonl_sink.py`.
//!
//! Python drives the sink with `on_error="raise"`; here [`WorkspaceJsonlSink::try_emit`] is the
//! raising form and `emit` is the logging one. The fake session wraps a real local workspace and
//! can record, delay or fail the file operations the sink makes.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::SandboxOp;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::events::{
    EventSink, InstrumentedSession, SandboxSessionEvent, WorkspaceJsonlSink,
};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;

/// How the next bounded read behaves.
#[derive(Debug, Clone)]
enum ReadFault {
    Fail(SandboxError),
}

/// Python: `_BoundedReadSession`, with injectable faults.
#[derive(Debug)]
struct FsSession {
    inner: Arc<dyn SandboxSession>,
    bounded_requests: Mutex<Vec<(String, usize)>>,
    whole_reads: Mutex<usize>,
    writes: Mutex<usize>,
    read_fault: Mutex<Option<ReadFault>>,
    write_fault: Mutex<Option<SandboxError>>,
    /// A backend that offers no bounded fetch, like a custom backend with only `read`.
    legacy: bool,
    running: Mutex<bool>,
    /// A manifest reported in place of the workspace's own, for entries the fake never mounts.
    overlay: Option<Manifest>,
}

impl FsSession {
    async fn new(legacy: bool) -> (UnixLocalSandboxClient, Arc<FsSession>) {
        FsSession::build(legacy, |_| None).await
    }

    /// A session whose manifest also holds the entries `extra` adds, under the workspace root.
    async fn with_entries(
        extra: impl FnOnce(Manifest) -> Manifest,
    ) -> (UnixLocalSandboxClient, Arc<FsSession>) {
        FsSession::build(false, |root| Some(extra(Manifest::new().with_root(root)))).await
    }

    async fn build(
        legacy: bool,
        overlay: impl FnOnce(&str) -> Option<Manifest>,
    ) -> (UnixLocalSandboxClient, Arc<FsSession>) {
        let client = UnixLocalSandboxClient::new();
        let inner = client.create(None).await.unwrap();
        inner.start().await.unwrap();
        let overlay = overlay(inner.workspace_root());
        (
            client,
            Arc::new(FsSession {
                overlay,
                inner,
                bounded_requests: Mutex::new(Vec::new()),
                whole_reads: Mutex::new(0),
                writes: Mutex::new(0),
                read_fault: Mutex::new(None),
                write_fault: Mutex::new(None),
                legacy,
                running: Mutex::new(true),
            }),
        )
    }

    fn root_file(&self, relpath: &str) -> std::path::PathBuf {
        std::path::Path::new(self.inner.workspace_root()).join(relpath)
    }

    fn fail_reads(&self, error: SandboxError) {
        *self.read_fault.lock().unwrap() = Some(ReadFault::Fail(error));
    }

    fn fail_writes(&self, error: SandboxError) {
        *self.write_fault.lock().unwrap() = Some(error);
    }

    fn bounded_requests(&self) -> Vec<(String, usize)> {
        self.bounded_requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl SandboxSession for FsSession {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn manifest(&self) -> &Manifest {
        self.overlay
            .as_ref()
            .unwrap_or_else(|| self.inner.manifest())
    }
    fn backend_id(&self) -> &str {
        "fs-session"
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
        Ok(*self.running.lock().unwrap())
    }
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        self.inner.exec(request).await
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        *self.whole_reads.lock().unwrap() += 1;
        if let Some(ReadFault::Fail(error)) = self.read_fault.lock().unwrap().clone() {
            return Err(error);
        }
        self.inner.read_file(path).await
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        *self.writes.lock().unwrap() += 1;
        if let Some(error) = self.write_fault.lock().unwrap().take() {
            return Err(error);
        }
        self.inner.write_file(path, contents).await
    }
    async fn read_bounded_unchecked(&self, path: &str, max_bytes: usize) -> SandboxResult<Vec<u8>> {
        if self.legacy {
            // The trait default: read it whole, then cut.
            let mut contents = self.read_file(path).await?;
            contents.truncate(max_bytes);
            return Ok(contents);
        }
        self.bounded_requests
            .lock()
            .unwrap()
            .push((path.to_string(), max_bytes));
        if let Some(ReadFault::Fail(error)) = self.read_fault.lock().unwrap().clone() {
            return Err(error);
        }
        let mut contents = self.inner.read_file(path).await?;
        contents.truncate(max_bytes);
        Ok(contents)
    }
    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        self.inner.mkdir(path, parents).await
    }
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.hydrate_workspace(archive).await
    }
}

fn event(session: &FsSession, op: SandboxOp) -> SandboxSessionEvent {
    SandboxSessionEvent::start(session.id(), "fs-session", 1, "test-span", op)
}

fn line_len(event: &SandboxSessionEvent) -> usize {
    event.to_json_line().unwrap().len()
}

fn sink(session: &Arc<FsSession>, relpath: &str) -> WorkspaceJsonlSink {
    WorkspaceJsonlSink::new(session.clone(), relpath)
}

fn lines(path: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

// ----------------------------------------------------------------------------------------------
// Budgets
// ----------------------------------------------------------------------------------------------

/// Python: `test_workspace_jsonl_sink_replacement_budget` (3 parameter sets).
#[tokio::test]
async fn the_replacement_budget_decides_whether_an_event_is_delivered() {
    for slack in [-1i64, 0, 1] {
        let (client, session) = FsSession::new(false).await;
        let ev = event(&session, SandboxOp::Write);
        let old = b"{\"old\":true}\n".to_vec();
        session.write_file("out.jsonl", &old).await.unwrap();
        let budget = (old.len() + line_len(&ev)) as i64 + slack;
        let sink = sink(&session, "out.jsonl")
            .with_max_bytes(Some(budget as usize))
            .unwrap();
        // A whole-file read would be the "unbounded download" Python's test forbids.
        *session.whole_reads.lock().unwrap() = 0;

        let result = sink.try_emit(&ev).await;

        let content = std::fs::read(session.root_file("out.jsonl")).unwrap();
        assert!(content.starts_with(&old));
        if slack < 0 {
            let error = result.expect_err("over budget");
            assert!(
                error.message.contains("delivery stopped"),
                "{}",
                error.message
            );
            assert_eq!(content, old);
        } else {
            result.unwrap();
            let delivered: serde_json::Value =
                serde_json::from_slice(&content[old.len()..]).unwrap();
            assert_eq!(delivered["seq"], 1);
        }
        assert_eq!(sink.pending_bytes().await, 0, "slack {slack}");
        assert_eq!(*session.whole_reads.lock().unwrap(), 0, "slack {slack}");
        client.delete(session.inner.clone()).await.unwrap();
    }
}

/// Python: `test_workspace_jsonl_sink_exhaustion_stops_buffering` (the raise and log/ignore
/// outcomes: one error once, then nothing, and the file is never touched).
#[tokio::test]
async fn an_exhausted_sink_stops_buffering() {
    for raises in [true, false] {
        let (client, session) = FsSession::new(false).await;
        let old = b"synthetic-history\n".repeat(100);
        session.write_file("out.jsonl", &old).await.unwrap();
        let sink = sink(&session, "out.jsonl")
            .with_max_bytes(Some(1024))
            .unwrap();
        let ev = event(&session, SandboxOp::Write);

        let first = if raises {
            sink.try_emit(&ev).await
        } else {
            sink.emit(&ev).await;
            Ok(())
        };
        assert_eq!(first.is_err(), raises);
        for _ in 0..20 {
            sink.try_emit(&ev)
                .await
                .expect("a stopped sink ignores events");
        }

        assert_eq!(session.bounded_requests().len(), 1);
        assert_eq!(sink.pending_bytes().await, 0);
        assert_eq!(std::fs::read(session.root_file("out.jsonl")).unwrap(), old);
        client.delete(session.inner.clone()).await.unwrap();
    }
}

/// Python: `test_workspace_jsonl_sink_pending_budget_before_flush`.
#[tokio::test]
async fn the_pending_buffer_is_budgeted_before_a_flush() {
    let (client, session) = FsSession::new(false).await;
    let ev = event(&session, SandboxOp::Write);
    let sink = sink(&session, "out.jsonl")
        .with_flush_every(100)
        .with_max_bytes(Some(line_len(&ev)))
        .unwrap();

    sink.try_emit(&ev).await.unwrap();
    let error = sink
        .try_emit(&ev)
        .await
        .expect_err("the second event overflows");
    assert!(error.message.contains("max_bytes"), "{}", error.message);
    sink.try_emit(&ev).await.unwrap();

    assert!(session.bounded_requests().is_empty());
    assert_eq!(sink.pending_bytes().await, 0);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_bounds_read_before_transfer`.
#[tokio::test]
async fn the_read_is_bounded_before_the_transfer() {
    let (client, session) = FsSession::new(false).await;
    let old = b"ordinary fixture\n".repeat(1024);
    session.write_file("out.jsonl", &old).await.unwrap();
    let sink = sink(&session, "out.jsonl")
        .with_max_bytes(Some(1024))
        .unwrap();

    let error = sink
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .expect_err("the outbox is over budget");

    assert!(error.message.contains("delivery stopped"));
    // One byte past the budget is all that was asked for.
    assert_eq!(
        session.bounded_requests(),
        vec![("out.jsonl".to_string(), 1025)]
    );
    assert_eq!(std::fs::read(session.root_file("out.jsonl")).unwrap(), old);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_wire_budget_stops_delivery`.
#[tokio::test]
async fn a_transport_limit_stops_delivery() {
    let (client, session) = FsSession::new(false).await;
    let sink = sink(&session, "out.jsonl")
        .with_max_bytes(Some(8 * 1024 * 1024))
        .unwrap();
    session.fail_reads(
        SandboxError::new(ErrorCode::WorkspaceArchiveReadError, "wire limit")
            .with_context("reason", "bounded_read_wire_limit"),
    );

    let error = sink
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .expect_err("the transport refused the read");
    sink.try_emit(&event(&session, SandboxOp::Write))
        .await
        .unwrap();

    assert!(error.message.contains("delivery stopped"));
    assert_eq!(session.bounded_requests().len(), 1);
    assert_eq!(*session.writes.lock().unwrap(), 0);
    assert_eq!(sink.pending_bytes().await, 0);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_requires_positive_budget`.
#[tokio::test]
async fn a_zero_budget_is_refused() {
    let (client, session) = FsSession::new(false).await;

    let error = sink(&session, "out.jsonl")
        .with_max_bytes(Some(0))
        .expect_err("zero is not a budget");

    assert!(error.message.contains("max_bytes must be positive"));
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_default_keeps_delivering_after_eight_mib` (2 sets).
#[tokio::test]
async fn the_default_keeps_delivering_after_eight_mib() {
    for explicit_none in [false, true] {
        let (client, session) = FsSession::new(false).await;
        let mut old = b"{\"old\":\"".to_vec();
        old.extend(std::iter::repeat_n(b'x', 8 * 1024 * 1024));
        old.extend_from_slice(b"\"}\n");
        session.write_file("out.jsonl", &old).await.unwrap();
        let sink = if explicit_none {
            sink(&session, "out.jsonl").with_max_bytes(None).unwrap()
        } else {
            sink(&session, "out.jsonl")
        };

        sink.try_emit(&event(&session, SandboxOp::Write))
            .await
            .unwrap();
        sink.try_emit(&event(&session, SandboxOp::Stop))
            .await
            .unwrap();

        let content = std::fs::read(session.root_file("out.jsonl")).unwrap();
        assert!(content.starts_with(&old));
        let ops: Vec<String> = String::from_utf8(content[old.len()..].to_vec())
            .unwrap()
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["op"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(ops, ["write", "stop"]);
        assert_eq!(sink.pending_bytes().await, 0);
        client.delete(session.inner.clone()).await.unwrap();
    }
}

// ----------------------------------------------------------------------------------------------
// Failure handling
// ----------------------------------------------------------------------------------------------

/// Python: `test_workspace_jsonl_sink_failed_read_without_writing` and
/// `test_workspace_sink_waits_for_backend_read_cleanup`: a failed read never causes a
/// replacement write, keeps the event, and delivers it once the backend recovers.
#[tokio::test]
async fn a_failed_read_keeps_the_event_without_writing() {
    let (client, session) = FsSession::new(false).await;
    let sink = sink(&session, "out.jsonl")
        .with_max_bytes(Some(8 * 1024 * 1024))
        .unwrap();
    session.fail_reads(
        SandboxError::new(ErrorCode::WorkspaceArchiveReadError, "read failed").with_retryable(true),
    );

    let error = sink
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .expect_err("the read fails");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(*session.writes.lock().unwrap(), 0);
    assert!(sink.pending_bytes().await > 0);

    // The backend recovers: the retained event is delivered with the one that follows it.
    *session.read_fault.lock().unwrap() = None;
    sink.try_emit(&event(&session, SandboxOp::Stop))
        .await
        .unwrap();
    let delivered = lines(&session.root_file("out.jsonl"));
    let ops: Vec<&str> = delivered
        .iter()
        .map(|line| line["op"].as_str().unwrap())
        .collect();
    assert_eq!(ops, ["write", "stop"]);
    assert_eq!(sink.pending_bytes().await, 0);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_retries_delivery_and_flushes_lifecycle` (the retry half:
/// a failed write keeps the buffer, and the lifecycle event flushes everything in order).
#[tokio::test]
async fn a_failed_write_is_retried_on_the_next_lifecycle_event() {
    let (client, session) = FsSession::new(false).await;
    let sink = sink(&session, "out.jsonl").with_flush_every(100);
    sink.try_emit(&event(&session, SandboxOp::Write))
        .await
        .unwrap();
    session.fail_writes(SandboxError::new(
        ErrorCode::WorkspaceArchiveWriteError,
        "temporary failure",
    ));

    sink.try_emit(&event(&session, SandboxOp::PersistWorkspace))
        .await
        .expect_err("the flush at persist fails");
    assert!(sink.pending_bytes().await > 0);
    sink.try_emit(&event(&session, SandboxOp::Stop))
        .await
        .unwrap();

    let delivered = lines(&session.root_file("out.jsonl"));
    let ops: Vec<&str> = delivered
        .iter()
        .map(|line| line["op"].as_str().unwrap())
        .collect();
    assert_eq!(ops, ["write", "persist_workspace", "stop"]);
    assert_eq!(sink.pending_bytes().await, 0);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_preserves_bytes_and_reports_failure`.
#[tokio::test]
async fn existing_bytes_are_preserved_and_a_read_failure_is_reported() {
    let (client, session) = FsSession::new(false).await;
    let sink_a = sink(&session, "out.jsonl");
    sink_a
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .unwrap();
    let mut old = b"\x00\xff\n".to_vec();
    old.extend("日本語\n".as_bytes());
    old.extend(std::iter::repeat_n(b'x', 20_000));
    old.push(b'\n');
    session.write_file("out.jsonl", &old).await.unwrap();

    sink_a
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .unwrap();

    let content = std::fs::read(session.root_file("out.jsonl")).unwrap();
    assert!(content.starts_with(&old));
    let delivered: serde_json::Value = serde_json::from_slice(&content[old.len()..]).unwrap();
    assert_eq!(delivered["seq"], 1);

    // Read failures must not cause a replacement write.
    session.inner.mkdir("directory", false).await.unwrap();
    let on_a_directory = sink(&session, "directory");
    let error = on_a_directory
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .expect_err("a directory cannot be read as a file");
    assert_ne!(error.code, ErrorCode::WorkspaceReadNotFound);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_errors_have_no_pending_payload` (the sink half): a
/// failure carries the error's code and message, never the pending event data.
#[tokio::test]
async fn delivery_errors_do_not_carry_the_pending_payload() {
    for timeout in [false, true] {
        let (client, session) = FsSession::new(false).await;
        let sink = sink(&session, "out.jsonl")
            .with_max_bytes(Some(8 * 1024 * 1024))
            .unwrap();
        let code = if timeout {
            ErrorCode::ExecTimeout
        } else {
            ErrorCode::ExecTransportError
        };
        session.fail_reads(SandboxError::new(code, "provider read failed"));
        let ev = event(&session, SandboxOp::Write).with_data("secret", "synthetic-private-value");

        let error = sink.try_emit(&ev).await.expect_err("the read fails");

        assert!(!format!("{error:?}").contains("synthetic-private-value"));
        assert!(sink.pending_bytes().await > 0, "the event is kept");
        client.delete(session.inner.clone()).await.unwrap();
    }
}

// ----------------------------------------------------------------------------------------------
// Delivery
// ----------------------------------------------------------------------------------------------

/// Python: `test_workspace_jsonl_sink_legacy_backend_delivers` (the `missing`, `binary` and
/// `short` histories): a backend with only a whole-file read still preserves what is there.
#[tokio::test]
async fn a_legacy_backend_delivers_after_any_history() {
    let histories: [(&str, Vec<u8>); 3] = [
        ("missing", Vec::new()),
        ("binary", b"\x00\xff\n".to_vec()),
        ("text", "日本語\n".as_bytes().to_vec()),
    ];
    for (kind, old) in histories {
        let (client, session) = FsSession::new(true).await;
        let sink = sink(&session, "out.jsonl")
            .with_max_bytes(Some(8 * 1024 * 1024))
            .unwrap();
        if kind != "missing" {
            session.write_file("out.jsonl", &old).await.unwrap();
        }

        sink.try_emit(&event(&session, SandboxOp::Write))
            .await
            .unwrap();

        let content = std::fs::read(session.root_file("out.jsonl")).unwrap();
        assert!(content.starts_with(&old), "{kind}");
        let delivered: serde_json::Value = serde_json::from_slice(&content[old.len()..]).unwrap();
        assert_eq!(delivered["seq"], 1);
        assert_eq!(sink.pending_bytes().await, 0);
        client.delete(session.inner.clone()).await.unwrap();
    }
}

/// Python: `test_workspace_jsonl_sink_legacy_backend_stops_at_limit` (2 sets).
#[tokio::test]
async fn a_legacy_backend_stops_at_the_limit() {
    let (client, session) = FsSession::new(true).await;
    let old = "日本語\n".repeat(1024);
    session
        .write_file("out.jsonl", old.as_bytes())
        .await
        .unwrap();
    let sink = sink(&session, "out.jsonl")
        .with_max_bytes(Some(1024))
        .unwrap();

    let error = sink
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .expect_err("the outbox is over budget");

    assert!(error.message.contains("delivery stopped"));
    assert_eq!(
        std::fs::read(session.root_file("out.jsonl")).unwrap(),
        old.as_bytes()
    );
    assert_eq!(sink.pending_bytes().await, 0);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_preserves_preexisting_outbox_contents`.
#[tokio::test]
async fn preexisting_outbox_contents_are_preserved() {
    let (client, session) = FsSession::new(false).await;
    let relpath = format!("logs/events-{}.jsonl", session.id());
    session
        .write_file(&relpath, b"{\"old\":true}\n")
        .await
        .unwrap();
    let sink = sink(&session, "logs/events-{session_id}.jsonl");
    let start = event(&session, SandboxOp::Write);
    let mut finish = start.finish(true, 0);
    finish.seq = 2;

    sink.try_emit(&start).await.unwrap();
    sink.try_emit(&finish).await.unwrap();

    let delivered = lines(&session.root_file(&relpath));
    assert_eq!(delivered.len(), 3);
    assert_eq!(delivered[0], serde_json::json!({"old": true}));
    assert_eq!(delivered[1]["seq"], 1);
    assert_eq!(delivered[2]["seq"], 2);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_does_not_duplicate_lines_across_flushes` and
/// `..._clears_flushed_buffer`.
#[tokio::test]
async fn flushes_do_not_duplicate_lines_and_clear_the_buffer() {
    let (client, session) = FsSession::new(false).await;
    let sink = sink(&session, "logs/events.jsonl").with_flush_every(1);
    for seq in [1u64, 2, 3] {
        let mut ev = event(&session, SandboxOp::Write);
        ev.seq = seq;
        sink.try_emit(&ev).await.unwrap();
        assert_eq!(sink.pending_bytes().await, 0);
    }

    let delivered = lines(&session.root_file("logs/events.jsonl"));
    let seqs: Vec<u64> = delivered
        .iter()
        .map(|line| line["seq"].as_u64().unwrap())
        .collect();
    assert_eq!(seqs, [1, 2, 3]);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_flushes_on_stop_when_flush_every_gt_one`.
#[tokio::test]
async fn a_stop_event_flushes_when_flush_every_is_greater_than_one() {
    let (client, session) = FsSession::new(false).await;
    let sink = sink(&session, "logs/events.jsonl").with_flush_every(10);

    sink.try_emit(&event(&session, SandboxOp::Write))
        .await
        .unwrap();
    assert!(!session.root_file("logs/events.jsonl").exists());
    sink.try_emit(&event(&session, SandboxOp::Stop))
        .await
        .unwrap();

    assert_eq!(lines(&session.root_file("logs/events.jsonl")).len(), 2);
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_supports_session_id_template`, plus the `session_id_hex`
/// form the Python docstring documents.
#[tokio::test]
async fn the_relpath_supports_session_id_templates() {
    let (client, session) = FsSession::new(false).await;
    let sink_a = sink(&session, "logs/events-{session_id}.jsonl");
    let sink_b = sink(&session, "logs/events-{session_id_hex}.jsonl");

    assert_eq!(
        sink_a.workspace_relpath(),
        format!("logs/events-{}.jsonl", session.id())
    );
    assert_eq!(
        sink_b.workspace_relpath(),
        format!("logs/events-{}.jsonl", session.id().replace('-', ""))
    );
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_workspace_jsonl_sink_writes_into_workspace_and_persists` and
/// `..._ephemeral_excludes_runtime_outbox_with_existing_parent`: an event stream recorded by an
/// instrumented session lands in the workspace, and an ephemeral outbox is left out of the
/// workspace archive while a durable one is kept.
#[tokio::test]
async fn the_outbox_is_archived_unless_it_is_ephemeral() {
    for ephemeral in [false, true] {
        let client = UnixLocalSandboxClient::new();
        let inner = client.create(None).await.unwrap();
        inner.start().await.unwrap();
        inner.write_file("logs/keep.txt", b"keep").await.unwrap();
        let relpath = format!("logs/events-{}.jsonl", inner.id());
        let sink = Arc::new(
            WorkspaceJsonlSink::new(inner.clone(), "logs/events-{session_id}.jsonl")
                .with_ephemeral(ephemeral)
                .unwrap(),
        );
        let session = InstrumentedSession::new(inner.clone(), sink.clone());

        session.exec(ExecRequest::shell("echo hi")).await.unwrap();
        session.stop().await.unwrap();

        let outbox = std::path::Path::new(inner.workspace_root()).join(&relpath);
        assert!(!std::fs::read_to_string(&outbox).unwrap().is_empty());
        let archive = inner.persist_workspace().await.unwrap();
        let mut names = Vec::new();
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(archive.as_slice()));
        for entry in tar.entries().unwrap() {
            names.push(
                entry
                    .unwrap()
                    .path()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        assert!(
            names.iter().any(|name| name.ends_with("logs/keep.txt")),
            "{names:?}"
        );
        assert_eq!(
            names
                .iter()
                .any(|name| name.contains(&format!("events-{}.jsonl", inner.id()))),
            !ephemeral,
            "ephemeral={ephemeral}: {names:?}"
        );
        client.delete(inner).await.unwrap();
    }
}

/// Python: `test_workspace_jsonl_sink_preserves_mounted_write_and_rebind` is the same sink
/// rebinding to a session whose outbox sits on a mount; the Rust sink binds at construction, so
/// two sinks over one outbox append rather than replace each other.
#[tokio::test]
async fn two_sinks_over_one_outbox_append() {
    let (client, session) = FsSession::new(false).await;

    sink(&session, "logs/out.jsonl")
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .unwrap();
    sink(&session, "logs/out.jsonl")
        .try_emit(&event(&session, SandboxOp::Write))
        .await
        .unwrap();

    let delivered = lines(&session.root_file("logs/out.jsonl"));
    assert_eq!(delivered.len(), 2);
    assert!(delivered.iter().all(|line| line["seq"] == 1));
    client.delete(session.inner.clone()).await.unwrap();
}

/// Python: `test_bounded_read_preserves_retryability_without_provider_diagnostics` (3 rows). A
/// failed bounded read reaches the caller, and the finish event, with only the path and a
/// reason; the provider's context, response and cause are dropped, and retryability survives.
#[tokio::test]
async fn bounded_read_preserves_retryability_without_provider_diagnostics() {
    use openai_agents_sdk::sandbox::events::{CallbackSink, EventPhase};

    let rows: [(Option<bool>, &str, &str); 3] = [
        (Some(true), "provider_failure", "bounded_read_failed"),
        (
            Some(false),
            "bounded_read_wire_limit",
            "bounded_read_wire_limit",
        ),
        (None, "provider_failure", "bounded_read_failed"),
    ];
    for (retryable, reason, expected_reason) in rows {
        let (client, inner) = FsSession::new(false).await;
        let mut failure = SandboxError::new(
            ErrorCode::WorkspaceArchiveReadError,
            "synthetic-private-payload",
        )
        .with_context("reason", reason)
        .with_context("response", "synthetic-private-payload")
        .with_context("path", "out.jsonl");
        failure.retryable = retryable;
        inner.fail_reads(failure);
        let events: Arc<Mutex<Vec<SandboxSessionEvent>>> = Arc::default();
        let seen = events.clone();
        let session = InstrumentedSession::new(
            inner.clone(),
            Arc::new(CallbackSink::new(move |event| {
                seen.lock().unwrap().push(event.clone())
            })),
        );

        let error = session.read_bounded("out.jsonl", 100).await.unwrap_err();

        assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
        assert_eq!(error.retryable, retryable, "{reason}");
        let expected: std::collections::BTreeMap<String, String> = [
            ("path".to_string(), "out.jsonl".to_string()),
            ("reason".to_string(), expected_reason.to_string()),
        ]
        .into();
        assert_eq!(error.context, expected);
        assert!(!error.message.contains("synthetic-private-payload"));
        {
            let events = events.lock().unwrap();
            let finish = events
                .iter()
                .find(|event| event.op == "read" && event.phase == EventPhase::Finish)
                .expect("a read finish event");
            assert_eq!(finish.error_retryable, retryable);
            assert!(
                !serde_json::to_string(finish)
                    .unwrap()
                    .contains("synthetic-private-payload")
            );
        }
        client.delete(inner.inner.clone()).await.ok();
    }
}

/// Python: `test_bounded_read_preserves_classified_transport_retryability`: a classified
/// transport failure keeps its retryability and leaks nothing, called on the session itself.
#[tokio::test]
async fn bounded_read_preserves_classified_transport_retryability() {
    let (client, inner) = FsSession::new(false).await;
    inner.fail_reads(
        SandboxError::new(ErrorCode::ExecTransportError, "synthetic-private-payload")
            .with_retryable(true),
    );

    let error = inner.read_bounded("out.jsonl", 100).await.unwrap_err();

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(error.retryable, Some(true));
    assert!(!error.message.contains("synthetic-private-payload"));
    assert!(
        !format!("{error:?}").contains("synthetic-private-payload"),
        "{error:?}"
    );
    client.delete(inner.inner.clone()).await.ok();
}

/// Python: `test_workspace_jsonl_sink_preserves_mounted_write_and_rebind` (2 rows). An outbox under
/// a mount, or under a mount at the workspace root, is written through, binding the sink leaves the
/// session's persist exclusions alone (a mount is already excluded), and a sink bound afterwards
/// appends to the same file instead of replacing it.
#[tokio::test]
async fn a_mounted_outbox_is_written_and_rebound_without_changing_exclusions() {
    use openai_agents_sdk::sandbox::entries::Entry;
    use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};

    for root_mount in [false, true] {
        let (client, session) = FsSession::with_entries(|manifest| {
            let mount_path = if root_mount {
                manifest.root().to_string()
            } else {
                "logs".to_string()
            };
            manifest.with_entry(
                "storage",
                Entry::mount(
                    ConfiguredMountStrategy::rclone(S3Mount::new("test-bucket"))
                        .mount()
                        .with_mount_path(mount_path),
                ),
            )
        })
        .await;
        let exclusions = |session: &FsSession| {
            session
                .inner
                .persist_skip_registry()
                .map(|registry| registry.lock().unwrap().clone())
                .unwrap_or_default()
        };
        let original = exclusions(&session);

        let first = sink(&session, "logs/out.jsonl");
        assert_eq!(exclusions(&session), original, "root_mount={root_mount}");
        first
            .try_emit(&event(&session, SandboxOp::Write))
            .await
            .unwrap();
        let second = sink(&session, "logs/out.jsonl");
        assert_eq!(exclusions(&session), original, "root_mount={root_mount}");
        second
            .try_emit(&event(&session, SandboxOp::Write))
            .await
            .unwrap();

        let delivered = lines(&session.root_file("logs/out.jsonl"));
        assert_eq!(delivered.len(), 2, "root_mount={root_mount}");
        assert!(delivered.iter().all(|line| line["seq"] == 1));
        client.delete(session.inner.clone()).await.unwrap();
    }
}
