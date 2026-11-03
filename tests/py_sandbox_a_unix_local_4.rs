//! The `sandbox.stop` and `sandbox.shutdown` spans a Unix-local session emits.
//!
//! Python wraps `SandboxSession.stop`/`shutdown` in `instrumented_op`, which opens a
//! `sandbox.<op>` custom span (backend, operation and session id attributes) whenever the current
//! trace exports spans; `test_sandbox_runtime_emits_high_level_sdk_spans` asserts both names. In
//! Rust the backend's own `stop`/`shutdown` open them, and `SandboxClient::delete` performs both.
//!
//! The trace provider is process-global, so this binary holds only tests that share it, run one
//! at a time.

#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient};
use openai_agents_sdk::tracing::{
    Span, SpanData, Trace, TracingProcessor, set_trace_processors, set_tracing_disabled, trace,
};
use serde_json::Value;

/// Every custom span that ended, in end order, with its attributes.
#[derive(Debug, Default)]
struct SpanRecorder {
    spans: Mutex<Vec<(String, Value)>>,
    /// The error each custom span ended with, by span name.
    errors: Mutex<Vec<(String, Option<openai_agents_sdk::tracing::SpanError>)>>,
}

impl TracingProcessor for SpanRecorder {
    fn on_trace_start(&self, _trace: &Trace) {}
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let SpanData::Custom { name, data } = span.data() {
            self.errors
                .lock()
                .unwrap()
                .push((name.clone(), span.error()));
            self.spans.lock().unwrap().push((name, data));
        }
    }
}

async fn serial() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// Create, start and release a Unix-local session, returning its id.
async fn run_a_session(client: &UnixLocalSandboxClient) -> String {
    let session = client.create(Some(Manifest::new())).await.expect("create");
    session.start().await.expect("start");
    let id = session.id().to_string();
    client.delete(session).await.expect("delete");
    id
}

fn recorder() -> Arc<SpanRecorder> {
    set_tracing_disabled(false);
    let recorder = Arc::new(SpanRecorder::default());
    set_trace_processors(vec![recorder.clone()]);
    recorder
}

fn sandbox_spans(recorder: &SpanRecorder) -> Vec<(String, Value)> {
    recorder
        .spans
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| name.starts_with("sandbox."))
        .cloned()
        .collect()
}

/// Releasing a session inside a trace records `sandbox.stop` and then `sandbox.shutdown`, each
/// carrying the backend, the operation and the session id.
#[tokio::test(flavor = "current_thread")]
async fn delete_emits_stop_and_shutdown_spans() {
    let _serial = serial().await;
    let recorder = recorder();
    let client = UnixLocalSandboxClient::new();

    let id = trace("unix_local_spans")
        .run(async { run_a_session(&client).await })
        .await;

    let spans = sandbox_spans(&recorder);
    let names: Vec<&str> = spans.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["sandbox.stop", "sandbox.shutdown"], "{spans:?}");
    for ((_, data), operation) in spans.iter().zip(["stop", "shutdown"]) {
        assert_eq!(data["sandbox.backend"], "unix_local");
        assert_eq!(data["sandbox.operation"], operation);
        assert_eq!(data["sandbox.session.id"], id.as_str());
        assert_eq!(data["session_id"], id.as_str());
    }

    set_trace_processors(Vec::new());
}

/// With no active trace there is nothing to record into, and releasing still works.
#[tokio::test(flavor = "current_thread")]
async fn no_spans_are_emitted_outside_a_trace() {
    let _serial = serial().await;
    let recorder = recorder();

    run_a_session(&UnixLocalSandboxClient::new()).await;

    assert!(sandbox_spans(&recorder).is_empty());
    set_trace_processors(Vec::new());
}

/// A disabled trace, which is how the runner masks cleanup when the run has tracing turned off,
/// records no `sandbox.*` spans.
#[tokio::test(flavor = "current_thread")]
async fn no_spans_are_emitted_when_tracing_is_disabled_for_the_trace() {
    let _serial = serial().await;
    let recorder = recorder();
    let client = UnixLocalSandboxClient::new();

    trace("unix_local_spans_disabled")
        .disabled(true)
        .build()
        .run(async { run_a_session(&client).await })
        .await;

    assert!(sandbox_spans(&recorder).is_empty());
    set_trace_processors(Vec::new());
}

/// A mount whose detach always fails, so stopping the session fails.
#[derive(Debug)]
struct BusyMountPattern;

#[async_trait::async_trait]
impl openai_agents_sdk::sandbox::mounts::MountPattern for BusyMountPattern {
    fn pattern_type(&self) -> &str {
        "busy"
    }

    fn exposes_credentials(&self) -> bool {
        false
    }

    async fn apply(
        &self,
        _session: &dyn openai_agents_sdk::sandbox::SandboxSession,
        _mount_path: &str,
        _config: &openai_agents_sdk::sandbox::mounts::MountPatternConfig,
    ) -> Result<(), openai_agents_sdk::sandbox::SandboxError> {
        Ok(())
    }

    async fn unapply(
        &self,
        _session: &dyn openai_agents_sdk::sandbox::SandboxSession,
        _mount_path: &str,
        _config: &openai_agents_sdk::sandbox::mounts::MountPatternConfig,
    ) -> Result<(), openai_agents_sdk::sandbox::SandboxError> {
        Err(openai_agents_sdk::sandbox::SandboxError::new(
            openai_agents_sdk::sandbox::ErrorCode::MountFailed,
            "device is busy",
        ))
    }
}

/// Python: `_apply_trace_finish_data`. A `stop` that fails records the failure on its
/// `sandbox.stop` span: `error.type`, `error_code` and the operation.
#[tokio::test(flavor = "current_thread")]
async fn a_failing_stop_records_error_attributes_on_its_span() {
    use openai_agents_sdk::sandbox::ErrorCode;
    use openai_agents_sdk::sandbox::entries::Entry;
    use openai_agents_sdk::sandbox::mounts::{InContainerMountStrategy, Mount, MountPatternConfig};

    let _serial = serial().await;
    let recorder = recorder();
    let client = UnixLocalSandboxClient::new();
    let probe = client.create(None).await.expect("probe");
    let root = probe.workspace_root().to_string();
    client.delete(probe).await.expect("delete probe");

    let mount = Mount::new(Arc::new(InContainerMountStrategy::new(
        Arc::new(BusyMountPattern),
        MountPatternConfig::new(),
    )));
    let manifest = Manifest::new()
        .with_root(root.clone())
        .with_entry("store", Entry::mount(mount));
    let session = client.create(Some(manifest)).await.expect("create");
    session.start().await.expect("start");

    let error = trace("stop_error")
        .run(async { session.stop().await })
        .await
        .expect_err("the busy mount fails the stop");
    assert_eq!(error.code, ErrorCode::MountFailed);

    let spans = sandbox_spans(&recorder);
    let (_, data) = spans
        .iter()
        .find(|(name, _)| name == "sandbox.stop")
        .expect("stop span");
    assert_eq!(data["error.type"], "SandboxError");
    assert_eq!(data["error_code"], ErrorCode::MountFailed.as_str());
    let recorded = recorder
        .errors
        .lock()
        .unwrap()
        .iter()
        .find(|(name, _)| name == "sandbox.stop")
        .and_then(|(_, error)| error.clone())
        .expect("the span records an error");
    assert_eq!(recorded.data.expect("data")["operation"], "stop");

    std::fs::remove_dir_all(&root).ok();
    set_trace_processors(Vec::new());
}
