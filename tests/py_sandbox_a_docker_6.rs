//! The `sandbox.stop` and `sandbox.shutdown` spans a Docker session emits.
//!
//! Python wraps `SandboxSession.stop`/`shutdown` in `instrumented_op`, which opens a
//! `sandbox.<op>` custom span (backend, operation and session id attributes) whenever the current
//! trace exports spans. In Rust the Docker session's own `stop`/`shutdown` open them, and
//! `SandboxClient::delete` performs both. Docker is reached through a fake `DockerRunner`.
//!
//! The trace provider is process-global, so this binary holds only tests that share it, run one
//! at a time.

#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::sandboxes::docker::{DockerRunner, DockerSandboxClient};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxResult};
use openai_agents_sdk::tracing::{
    Span, SpanData, Trace, TracingProcessor, set_trace_processors, set_tracing_disabled, trace,
};
use serde_json::Value;

#[derive(Debug, Default)]
struct SpanRecorder {
    spans: Mutex<Vec<(String, Value)>>,
}

impl TracingProcessor for SpanRecorder {
    fn on_trace_start(&self, _trace: &Trace) {}
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let SpanData::Custom { name, data } = span.data() {
            self.spans.lock().unwrap().push((name, data));
        }
    }
}

/// Answers every `docker` call: `create` names a container, the rest succeed silently.
#[derive(Debug)]
struct QuietDocker;

#[async_trait]
impl DockerRunner for QuietDocker {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        let stdout = if argv.first().map(String::as_str) == Some("create") {
            "container-1\n"
        } else {
            ""
        };
        Ok(ExecResult {
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
        })
    }
}

async fn serial() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// Create and release a Docker session, returning the session id its spans carry.
async fn release_a_session() -> String {
    let client = DockerSandboxClient::new().with_runner(Arc::new(QuietDocker));
    let session = client.create(Some(Manifest::new())).await.expect("create");
    let id = session
        .session_state()
        .provider_metadata("docker_session_id")
        .and_then(Value::as_str)
        .expect("session id")
        .to_string();
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

    let id = trace("docker_spans")
        .run(async { release_a_session().await })
        .await;

    let spans = sandbox_spans(&recorder);
    let names: Vec<&str> = spans.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["sandbox.stop", "sandbox.shutdown"], "{spans:?}");
    for ((_, data), operation) in spans.iter().zip(["stop", "shutdown"]) {
        assert_eq!(data["sandbox.backend"], "docker");
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

    release_a_session().await;

    assert!(sandbox_spans(&recorder).is_empty());
    set_trace_processors(Vec::new());
}

/// A disabled trace, which is how the runner masks cleanup when the run has tracing turned off,
/// records no `sandbox.*` spans.
#[tokio::test(flavor = "current_thread")]
async fn no_spans_are_emitted_when_tracing_is_disabled_for_the_trace() {
    let _serial = serial().await;
    let recorder = recorder();

    trace("docker_spans_disabled")
        .disabled(true)
        .build()
        .run(async { release_a_session().await })
        .await;

    assert!(sandbox_spans(&recorder).is_empty());
    set_trace_processors(Vec::new());
}
