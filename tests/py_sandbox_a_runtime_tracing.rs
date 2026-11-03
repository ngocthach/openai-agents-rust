//! Ports of `tests/sandbox/test_runtime.py` (lines 1-4000), tracing part: the high-level
//! `sandbox.*` spans a runner-owned session emits. The trace provider is process-global, so this
//! binary holds only tests that share it, run one at a time.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, SandboxAgent, SandboxError, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tracing::{
    Span, SpanData, Trace, TracingProcessor, set_trace_processors, set_tracing_disabled, trace,
};

#[derive(Debug, Default)]
struct SpanNames {
    names: Mutex<Vec<String>>,
    /// Every custom span that ended: name, attributes and the error it recorded.
    ended: Mutex<
        Vec<(
            String,
            serde_json::Value,
            Option<openai_agents_sdk::tracing::SpanError>,
        )>,
    >,
}

impl TracingProcessor for SpanNames {
    fn on_trace_start(&self, _trace: &Trace) {}
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let SpanData::Custom { name, data } = span.data() {
            self.names.lock().unwrap().push(name.clone());
            self.ended.lock().unwrap().push((name, data, span.error()));
        }
    }
}

#[derive(Debug)]
struct IdleSession {
    manifest: Manifest,
}

#[async_trait]
impl SandboxSession for IdleSession {
    fn id(&self) -> &str {
        "idle"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "fake"
    }
    async fn start(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn stop(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn shutdown(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn running(&self) -> Result<bool, SandboxError> {
        Ok(false)
    }
    async fn exec(&self, _request: ExecRequest) -> Result<ExecResult, SandboxError> {
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
        Err(SandboxError::new(
            ErrorCode::WorkspaceReadNotFound,
            format!("no file at {path}"),
        ))
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
        Ok(())
    }
}

#[derive(Debug)]
struct IdleClient;

#[async_trait]
impl SandboxClient for IdleClient {
    fn backend_id(&self) -> &str {
        "fake"
    }
    async fn create(
        &self,
        manifest: Option<Manifest>,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(Arc::new(IdleSession {
            manifest: manifest.unwrap_or_default(),
        }))
    }
    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn resume(
        &self,
        _state: &SandboxSessionState,
        manifest: Manifest,
    ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
        Ok(Arc::new(IdleSession { manifest }))
    }
}

async fn serial() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// Python: `test_streamed_sandbox_cleanup_honors_tracing_opt_out` (the tracing-enabled halves for
/// the `success` outcome) and the sandbox half of `test_sandbox_runtime_emits_high_level_sdk_spans`.
///
/// A runner-owned session emits `sandbox.prepare_agent`, `sandbox.create_session`,
/// `sandbox.start`, and a `sandbox.cleanup` span that contains `sandbox.cleanup_sessions`, all
/// inside the caller's trace.
///
/// DEFERRED: Python also emits `sandbox.stop` and `sandbox.shutdown` spans, because it wraps the
/// session lifecycle itself. Rust hands both to `SandboxClient::delete`, so the runner has no seam
/// to span them; a backend's `delete` wraps its own calls in `runtime::sandbox_span`.
#[tokio::test]
async fn a_runner_owned_session_emits_high_level_sandbox_spans() {
    let _serial = serial().await;
    set_tracing_disabled(false);
    let recorder = Arc::new(SpanNames::default());
    set_trace_processors(vec![recorder.clone()]);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_capabilities(Vec::new());
    let config = SandboxRunConfig::with_client(Arc::new(IdleClient));

    trace("sandbox_runtime_test")
        .run(async {
            SandboxRunner::run(&agent, "hello", &config)
                .await
                .expect("the run completes");
        })
        .await;

    let names = recorder.names.lock().unwrap().clone();
    for expected in [
        "sandbox.prepare_agent",
        "sandbox.create_session",
        "sandbox.start",
        "sandbox.cleanup",
        "sandbox.cleanup_sessions",
    ] {
        assert!(
            names.iter().any(|name| name == expected),
            "{expected} in {names:?}"
        );
    }
    // `sandbox.cleanup_sessions` ends before the `sandbox.cleanup` that contains it.
    let cleanup_sessions = names
        .iter()
        .position(|n| n == "sandbox.cleanup_sessions")
        .unwrap();
    let cleanup = names.iter().position(|n| n == "sandbox.cleanup").unwrap();
    assert!(cleanup_sessions < cleanup, "{names:?}");

    set_trace_processors(Vec::new());
    set_tracing_disabled(true);
}

/// Python: `test_streamed_sandbox_cleanup_honors_tracing_opt_out` (`disabled=True`): with
/// `RunConfig::tracing_disabled` the sandbox records no `sandbox.*` spans, preparation and cleanup
/// included, even inside an enabled caller trace.
#[tokio::test]
async fn tracing_disabled_keeps_sandbox_spans_out_of_the_trace() {
    let _serial = serial().await;
    set_tracing_disabled(false);
    let recorder = Arc::new(SpanNames::default());
    set_trace_processors(vec![recorder.clone()]);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_capabilities(Vec::new());
    let config = SandboxRunConfig::with_client(Arc::new(IdleClient));
    let mut options = openai_agents_sdk::RunOptions::<()>::default();
    options.run_config.tracing_disabled = true;

    trace("caller")
        .run(async {
            SandboxRunner::run_with_options(&agent, "hello", &config, options)
                .await
                .expect("the run completes");
        })
        .await;

    let names = recorder.names.lock().unwrap().clone();
    assert!(
        names.iter().all(|name| !name.starts_with("sandbox.")),
        "{names:?}"
    );

    set_trace_processors(Vec::new());
    set_tracing_disabled(true);
}

/// Python: `_apply_trace_finish_data`. A failing sandbox operation records `error.type`,
/// `error_code` and the operation on its span, and a success records no error.
#[tokio::test(flavor = "current_thread")]
async fn a_failing_sandbox_operation_records_error_attributes_on_its_span() {
    use openai_agents_sdk::sandbox::runtime::sandbox_span_result;

    let _serial = serial().await;
    set_tracing_disabled(false);
    let recorder = Arc::new(SpanNames::default());
    set_trace_processors(vec![recorder.clone()]);

    trace("span_errors")
        .run(async {
            let ok: Result<(), SandboxError> = sandbox_span_result(
                "sandbox.stop",
                serde_json::json!({"sandbox.operation": "stop"}),
                async { Ok(()) },
            )
            .await;
            assert!(ok.is_ok());
            let failed: Result<(), SandboxError> = sandbox_span_result(
                "sandbox.shutdown",
                serde_json::json!({"sandbox.operation": "shutdown"}),
                async {
                    Err(
                        SandboxError::new(ErrorCode::WorkspaceStopError, "synthetic")
                            .with_retryable(true),
                    )
                },
            )
            .await;
            assert!(failed.is_err());
        })
        .await;

    let ended = recorder.ended.lock().unwrap().clone();
    let (_, ok_data, ok_error) = ended
        .iter()
        .find(|(name, ..)| name == "sandbox.stop")
        .expect("stop span");
    assert!(ok_error.is_none());
    assert!(ok_data.get("error.type").is_none());

    let (_, data, error) = ended
        .iter()
        .find(|(name, ..)| name == "sandbox.shutdown")
        .expect("shutdown span");
    assert_eq!(data["error.type"], "SandboxError");
    assert_eq!(data["error_type"], "SandboxError");
    assert_eq!(data["error_code"], ErrorCode::WorkspaceStopError.as_str());
    assert_eq!(data["error_retryable"], true);
    let error = error.clone().expect("the span records an error");
    assert_eq!(error.message, "SandboxError");
    let details = error.data.expect("error data");
    assert_eq!(details["operation"], "shutdown");
    assert_eq!(
        details["error_code"],
        ErrorCode::WorkspaceStopError.as_str()
    );

    set_trace_processors(Vec::new());
    set_tracing_disabled(true);
}
