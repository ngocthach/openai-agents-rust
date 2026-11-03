//! Port of `test_expected_read_span_error_is_call_scoped_and_preserves_audit_failures` from
//! `tests/sandbox/test_session_sinks.py`. Trace processors are process-global, so this is the only
//! test in its binary.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::events::{
    CallbackSink, EventPhase, InstrumentedSession, SandboxSessionEvent,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::{ErrorCode, Manifest};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, trace,
};
use serde_json::Value;

fn read_spans(processor: &InMemoryTraceProcessor) -> Vec<Value> {
    processor
        .items()
        .into_iter()
        .filter(|item| {
            item["object"] == "trace.span" && item["span_data"]["name"] == "sandbox.read"
        })
        .collect()
}

#[tokio::test]
async fn expected_read_span_error_is_call_scoped_and_preserves_audit_failures() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = UnixLocalSandboxClient::new()
        .create(Some(Manifest::new().with_root(root.to_string_lossy())))
        .await
        .unwrap();
    let events: Arc<Mutex<Vec<SandboxSessionEvent>>> = Arc::default();
    let seen = events.clone();
    let sink = CallbackSink::new(move |event| seen.lock().unwrap().push(event.clone()));
    let session = InstrumentedSession::new(inner, Arc::new(sink));

    let expected_path = "expected-missing.txt";
    let ordinary_path = "ordinary-missing.txt";
    let (expected, ordinary) = trace("sandbox_expected_read_error_test")
        .run(async {
            tokio::join!(
                session.read_file_expecting(
                    expected_path,
                    None,
                    &[ErrorCode::WorkspaceReadNotFound]
                ),
                session.read_file(ordinary_path),
            )
        })
        .await;

    assert_eq!(expected.unwrap_err().code, ErrorCode::WorkspaceReadNotFound);
    assert_eq!(ordinary.unwrap_err().code, ErrorCode::WorkspaceReadNotFound);

    let spans = read_spans(&processor);
    let error_for = |path: &str| {
        spans
            .iter()
            .find(|span| span["span_data"]["data"]["path"] == path)
            .unwrap_or_else(|| panic!("a read span for {path}"))["error"]
            .clone()
    };
    assert!(
        error_for(expected_path).is_null(),
        "an expected error leaves the span clean"
    );
    assert!(
        error_for(ordinary_path).is_object(),
        "an unexpected error marks the span"
    );

    let finishes: Vec<SandboxSessionEvent> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.op == "read" && event.phase == EventPhase::Finish)
        .cloned()
        .collect();
    assert_eq!(finishes.len(), 2);
    for event in &finishes {
        assert_eq!(event.ok, Some(false));
        assert_eq!(
            event.error_code.as_deref(),
            Some("workspace_read_not_found")
        );
        assert_eq!(event.error_retryable, Some(false));
    }
}
