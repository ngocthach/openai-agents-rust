//! The trace-dependent rows of `tests/sandbox/test_session_sinks.py`. Trace processors are
//! process-global, so every scenario runs in one test.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::SandboxOp;
use openai_agents_sdk::sandbox::events::{
    CallbackSink, EventPhase, InstrumentedSession, SandboxSessionEvent,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, SpanData, TracingProcessor, custom_span, set_trace_processors, span,
    trace,
};
use serde_json::Value;

type Collected = Arc<Mutex<Vec<SandboxSessionEvent>>>;

fn collecting() -> (CallbackSink, Collected) {
    let events: Collected = Arc::default();
    let seen = events.clone();
    (
        CallbackSink::new(move |event| seen.lock().unwrap().push(event.clone())),
        events,
    )
}

fn spans_named(processor: &InMemoryTraceProcessor, name: &str) -> Vec<Value> {
    processor
        .items()
        .into_iter()
        .filter(|item| item["object"] == "trace.span" && item["span_data"]["name"] == name)
        .collect()
}

#[tokio::test]
async fn sandbox_operations_are_traced_and_events_carry_trace_ids() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    let client = UnixLocalSandboxClient::new().with_exposed_ports([8765]);
    let inner = client.create(None).await.unwrap();
    inner.start().await.unwrap();

    // Python: `test_sandbox_session_error_events_and_traces_include_retryability` (span half) and
    // `test_sandbox_session_ops_nest_under_sdk_trace_and_events_carry_trace_ids`.
    let (sink, events) = collecting();
    let session = InstrumentedSession::new(inner.clone(), Arc::new(sink));
    let parent_id = Arc::new(Mutex::new(String::new()));
    let recorded_parent = parent_id.clone();
    trace("sandbox_test")
        .run(async {
            let parent = custom_span("sandbox_parent", serde_json::json!({}));
            *recorded_parent.lock().unwrap() = parent.span_id().to_string();
            parent
                .run(async {
                    session.write_file("notes.txt", b"hello").await.unwrap();
                    assert_eq!(session.read_file("notes.txt").await.unwrap(), b"hello");
                    session.read_file("missing.txt").await.unwrap_err();
                    let ok = session.exec(ExecRequest::shell("echo hi")).await.unwrap();
                    assert!(ok.ok());
                    let failed = session
                        .exec(ExecRequest::shell("echo failing >&2; exit 7"))
                        .await
                        .unwrap();
                    assert_eq!(failed.exit_code, 7);
                })
                .await;
        })
        .await;

    let parent_id = parent_id.lock().unwrap().clone();
    for name in ["sandbox.write", "sandbox.read", "sandbox.exec"] {
        let found = spans_named(&processor, name);
        assert!(!found.is_empty(), "{name} spans exist");
        assert!(
            found.iter().all(|s| s["parent_id"] == parent_id),
            "{name} nests under the parent span"
        );
    }
    let reads = spans_named(&processor, "sandbox.read");
    let failed_read = reads
        .iter()
        .find(|span| span["error"].is_object())
        .expect("the missing read's span carries an error");
    assert_eq!(failed_read["span_data"]["data"]["error_retryable"], false);
    assert_eq!(failed_read["error"]["data"]["error_retryable"], false);
    assert!(
        reads.iter().any(|span| span["error"].is_null()),
        "the successful read's span has no error"
    );

    let recorded = events.lock().unwrap().clone();
    let read_finish = recorded
        .iter()
        .find(|e| e.op == "read" && e.phase == EventPhase::Finish && e.ok == Some(false))
        .expect("a failed read finish event");
    assert_eq!(read_finish.error_retryable, Some(false));
    assert!(
        recorded.iter().all(|event| event.trace_id.is_some()),
        "every event inside the trace carries its id"
    );

    // Python: `test_sandbox_session_events_fallback_to_audit_ids_under_disabled_parent_span`.
    let (sink, events) = collecting();
    let session = InstrumentedSession::new(inner.clone(), Arc::new(sink));
    trace("sandbox_disabled_parent_test")
        .run(async {
            span(SpanData::Custom {
                name: "disabled_parent".into(),
                data: serde_json::json!({}),
            })
            .disabled(true)
            .build()
            .run(async {
                let result = session.exec(ExecRequest::shell("echo hi")).await.unwrap();
                assert!(result.ok());
            })
            .await;
        })
        .await;
    let recorded = events.lock().unwrap().clone();
    let exec_events: Vec<_> = recorded.iter().filter(|e| e.op == "exec").collect();
    assert_eq!(exec_events.len(), 2);
    assert_eq!(exec_events[0].trace_id, None);
    assert_eq!(exec_events[1].trace_id, None);
    assert_eq!(exec_events[0].span_id, exec_events[1].span_id);
    assert!(!exec_events[0].span_id.is_empty());
    let _ = SandboxOp::Exec;
    client.delete(inner).await.unwrap();
}
