//! What a failed sandbox event sink writes to the log, from `tests/sandbox/test_session_manager.py`
//! (`test_logged_sink_failure_conditionally_includes_sink_type`) and
//! `tests/sandbox/test_workspace_jsonl_sink.py` (`test_workspace_jsonl_sink_errors_have_no_pending_payload`).
//!
//! The `log` backend and the redaction flags are process-wide, so everything runs in one test.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::debug::{reset_dont_log_tool_data, set_dont_log_tool_data};
use openai_agents_sdk::sandbox::SandboxOp;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::events::{
    CallbackSink, EventSink, Instrumentation, SandboxSessionEvent, WorkspaceJsonlSink,
};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;

struct LogSink(Mutex<Vec<String>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn log_sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let sink = SINK.get_or_init(|| LogSink(Mutex::new(Vec::new())));
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

/// Both tests flip the process-wide tool-data flag, so they run one at a time.
fn flag_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn take_records() -> Vec<String> {
    std::mem::take(&mut *log_sink().0.lock().unwrap())
}

/// A backend with only a whole-file read, whose read can be made to fail.
#[derive(Debug)]
struct FailingReadSession {
    inner: Arc<dyn SandboxSession>,
    read_error: Mutex<Option<SandboxError>>,
}

#[async_trait]
impl SandboxSession for FailingReadSession {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn manifest(&self) -> &Manifest {
        self.inner.manifest()
    }
    fn backend_id(&self) -> &str {
        "failing-read-session"
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
        self.inner.exec(request).await
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        if let Some(error) = self.read_error.lock().unwrap().clone() {
            return Err(error);
        }
        self.inner.read_file(path).await
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.inner.write_file(path, contents).await
    }
    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        self.inner.mkdir(path, parents).await
    }
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.hydrate_workspace(archive).await
    }
}

/// Python: `test_logged_sink_failure_conditionally_includes_sink_type` (`redacted` and
/// `diagnostic`). A sink that fails and is told to ignore it logs one error. Under the redacted
/// policy the record is the fixed sentence; under the diagnostic one it also carries the
/// failure's text. The sink's type is never part of the Rust message.
#[tokio::test]
async fn a_logged_sink_failure_follows_the_tool_data_policy() {
    let _guard = flag_lock().lock().await;
    let _ = log_sink();

    for redacted in [true, false] {
        set_dont_log_tool_data(redacted);
        take_records();
        let sink = CallbackSink::raising(|_| {
            Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "SECRET_SINK_ERROR",
            ))
        });
        let instrumentation = Instrumentation::new(Arc::new(sink), "session", "unix_local");
        let event =
            SandboxSessionEvent::start("session", "unix_local", 1, "span_running", SandboxOp::Exec);

        instrumentation.emit(event).await;

        let records = take_records();
        let record = records
            .iter()
            .find(|record| record.contains("Sandbox event sink failed"))
            .unwrap_or_else(|| panic!("a failure record in {records:?}"));
        assert!(
            record.starts_with("Sandbox event sink failed (ignored)"),
            "{record}"
        );
        assert!(!record.contains("CallbackSink"), "{record}");
        if redacted {
            assert_eq!(record, "Sandbox event sink failed (ignored)");
            assert!(!record.contains("SECRET_SINK_ERROR"), "{record}");
        } else {
            assert!(record.contains("SECRET_SINK_ERROR"), "{record}");
        }
    }

    reset_dont_log_tool_data();
}

/// Python: `test_workspace_jsonl_sink_errors_have_no_pending_payload` (all 8 parameter sets:
/// timeout or transport failure, `raise` or `log` policy, redacted or diagnostic logging). The
/// pending event holds a private value; neither the returned error nor any log record may repeat
/// it, and the event stays pending.
#[tokio::test]
async fn delivery_failures_never_repeat_the_pending_payload() {
    let _guard = flag_lock().lock().await;
    let _ = log_sink();
    const SECRET: &str = "synthetic-private-value";

    for timeout in [false, true] {
        for raise in [true, false] {
            for redact in [false, true] {
                set_dont_log_tool_data(redact);
                take_records();
                let client = UnixLocalSandboxClient::new();
                let inner = client.create(None).await.unwrap();
                inner.start().await.unwrap();
                let code = if timeout {
                    ErrorCode::ExecTimeout
                } else {
                    ErrorCode::ExecTransportError
                };
                let session = Arc::new(FailingReadSession {
                    inner: inner.clone(),
                    read_error: Mutex::new(Some(SandboxError::new(code, "provider read failed"))),
                });
                let sink = WorkspaceJsonlSink::new(session.clone(), "out.jsonl")
                    .with_max_bytes(Some(8 * 1024 * 1024))
                    .unwrap();
                let event = SandboxSessionEvent::start(
                    session.id(),
                    "failing-read-session",
                    1,
                    "test-span",
                    SandboxOp::Write,
                )
                .with_data("secret", SECRET);
                let label = format!("timeout={timeout} raise={raise} redact={redact}");

                if raise {
                    let error = sink.try_emit(&event).await.expect_err(&label);
                    assert!(!format!("{error:?}").contains(SECRET), "{label}");
                } else {
                    sink.emit(&event).await;
                }

                for record in take_records() {
                    assert!(!record.contains(SECRET), "{label}: {record}");
                }
                assert!(sink.pending_bytes().await > 0, "{label}: the event is kept");
                client.delete(inner).await.unwrap();
            }
        }
    }

    reset_dont_log_tool_data();
}
