//! Port of `test_expected_read_span_records_finish_sink_failure` from
//! `tests/sandbox/test_session_sinks.py`: a raising sink that fails on the read's finish fails
//! the read, and the failure is recorded on the read span even though the read's own error was
//! expected. Trace processors are process-global, so this is the only test in its binary.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::sandbox::events::{CallbackSink, EventPhase, InstrumentedSession};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::{ErrorCode, Manifest, SandboxError};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, trace,
};

#[tokio::test]
async fn expected_read_span_records_finish_sink_failure() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = UnixLocalSandboxClient::new()
        .create(Some(Manifest::new().with_root(root.to_string_lossy())))
        .await
        .unwrap();
    let sink = CallbackSink::raising(|event| {
        if event.op == "read" && event.phase == EventPhase::Finish {
            return Err(SandboxError::new(
                ErrorCode::ExecTransportError,
                "simulated sink failure",
            ));
        }
        Ok(())
    });
    let session = InstrumentedSession::new(inner, Arc::new(sink));

    let error = trace("sandbox_expected_read_sink_failure_test")
        .run(async {
            session
                .read_file_expecting(
                    "expected-missing.txt",
                    None,
                    &[ErrorCode::WorkspaceReadNotFound],
                )
                .await
        })
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::EventSinkFailed);
    assert_eq!(error.message, "sandbox event sink failed");

    let span = processor
        .items()
        .into_iter()
        .find(|item| item["object"] == "trace.span" && item["span_data"]["name"] == "sandbox.read")
        .expect("a read span");
    assert!(
        span["error"].is_object(),
        "the sink failure marks the read span"
    );
    assert_eq!(span["error"]["data"]["error_code"], "event_sink_failed");
    assert_eq!(span["span_data"]["data"]["error_code"], "event_sink_failed");
}
