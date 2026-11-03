//! Ports of the portable rows of `tests/sandbox/test_session_manager.py` (instrumentation
//! payload policies, chained sinks) and `tests/sandbox/test_session_sinks.py` (sink behaviour).
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::SandboxOp;
use openai_agents_sdk::sandbox::events::{
    CallbackSink, ChainedSink, EventPayloadPolicy, EventPhase, EventSink, HttpProxySink,
    Instrumentation, InstrumentedSession, JsonlFileSink, SandboxSessionEvent,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use serde_json::Value;

type Collected = Arc<Mutex<Vec<SandboxSessionEvent>>>;

fn collecting_sink() -> (CallbackSink, Collected) {
    let events: Collected = Arc::default();
    let seen = events.clone();
    (
        CallbackSink::new(move |event| seen.lock().unwrap().push(event.clone())),
        events,
    )
}

fn exec_finish(stdout: &[u8], stderr: &[u8]) -> SandboxSessionEvent {
    SandboxSessionEvent::start("session", "unix_local", 1, "span_exec", SandboxOp::Exec)
        .finish(true, 0)
        .with_raw_output(stdout, stderr)
}

// ----------------------------------------------------------------------------------------------
// test_session_manager.py: payload policies
// ----------------------------------------------------------------------------------------------

/// Python: `test_instrumentation_per_op_policy_overrides_default`.
#[tokio::test]
async fn a_per_operation_policy_overrides_the_default() {
    let (sink, events) = collecting_sink();
    let instrumentation = Instrumentation::new(Arc::new(sink), "session", "unix_local")
        .with_policy(EventPayloadPolicy::new().with_exec_output(false))
        .with_policy_for_op(
            SandboxOp::Exec,
            EventPayloadPolicy::new().with_exec_output(true),
        );

    instrumentation.emit(exec_finish(b"hello", b"")).await;

    assert_eq!(events.lock().unwrap()[0].stdout.as_deref(), Some("hello"));
}

/// Python: `test_instrumentation_snapshots_per_op_policy_mapping`. The mapping is taken by
/// value, so the caller cannot change it afterwards; the operation's policy still wins.
#[tokio::test]
async fn the_per_operation_policy_is_a_snapshot() {
    let (sink, events) = collecting_sink();
    let exec_policy = EventPayloadPolicy::new().with_exec_output(false);
    let instrumentation = Instrumentation::new(Arc::new(sink), "session", "unix_local")
        .with_policy(EventPayloadPolicy::new().with_exec_output(true))
        .with_policy_for_op(SandboxOp::Exec, exec_policy);

    instrumentation.emit(exec_finish(b"secret", b"")).await;

    let event = events.lock().unwrap()[0].clone();
    assert_eq!(event.stdout, None);
    assert_eq!(event.stdout_bytes, None);
}

/// Python: `test_instrumentation_per_sink_policy_overrides_per_op`.
#[tokio::test]
async fn a_per_sink_policy_overrides_the_per_operation_policy() {
    let (sink_a, first) = collecting_sink();
    let (sink_b, second) = collecting_sink();
    let sink_b = sink_b.with_payload_policy(EventPayloadPolicy::new().with_exec_output(true));
    let chain = ChainedSink::new()
        .with_sink(Arc::new(sink_a))
        .with_sink(Arc::new(sink_b));
    let instrumentation = Instrumentation::new(Arc::new(chain), "session", "unix_local")
        .with_policy(EventPayloadPolicy::new().with_exec_output(false))
        .with_policy_for_op(
            SandboxOp::Exec,
            EventPayloadPolicy::new().with_exec_output(false),
        );

    instrumentation.emit(exec_finish(b"hello", b"")).await;

    assert_eq!(first.lock().unwrap()[0].stdout, None);
    assert_eq!(second.lock().unwrap()[0].stdout.as_deref(), Some("hello"));
}

/// Python: `test_instrumentation_redacts_raw_exec_bytes_when_output_disabled`.
#[tokio::test]
async fn raw_exec_bytes_are_redacted_when_output_is_disabled() {
    let (sink, events) = collecting_sink();
    let instrumentation = Instrumentation::new(Arc::new(sink), "session", "unix_local")
        .with_policy(EventPayloadPolicy::new().with_exec_output(false));

    instrumentation
        .emit(exec_finish(b"secret", b"secret2"))
        .await;

    let event = events.lock().unwrap()[0].clone();
    assert_eq!(event.stdout_bytes, None);
    assert_eq!(event.stderr_bytes, None);
    assert_eq!(event.stdout, None);
    assert_eq!(event.stderr, None);
}

/// Python: `test_chained_sink_preserves_completion_order_across_modes`. Rust sinks have no
/// delivery modes: every sink is awaited, in order, so the ordering guarantee is the one the
/// chain gives directly.
#[tokio::test]
async fn a_chained_sink_completes_each_member_before_the_next() {
    #[derive(Debug)]
    struct Order {
        log: Arc<Mutex<Vec<&'static str>>>,
        name: &'static str,
        yield_first: bool,
    }
    #[async_trait::async_trait]
    impl EventSink for Order {
        async fn emit(&self, _event: &SandboxSessionEvent) {
            if self.yield_first {
                tokio::task::yield_now().await;
            }
            self.log.lock().unwrap().push(self.name);
        }
    }
    let log = Arc::new(Mutex::new(Vec::new()));
    let chain = ChainedSink::new()
        .with_sink(Arc::new(Order {
            log: log.clone(),
            name: "slow",
            yield_first: true,
        }))
        .with_sink(Arc::new(Order {
            log: log.clone(),
            name: "after",
            yield_first: false,
        }));
    let instrumentation = Instrumentation::new(Arc::new(chain), "session", "unix_local");

    instrumentation.emit(exec_finish(b"", b"")).await;

    assert_eq!(*log.lock().unwrap(), vec!["slow", "after"]);
}

// ----------------------------------------------------------------------------------------------
// test_session_sinks.py
// ----------------------------------------------------------------------------------------------

async fn local_session() -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(None).await.unwrap();
    session.start().await.unwrap();
    (client, session)
}

/// Python: `test_sandbox_session_write_does_not_include_bytes_when_disabled`.
#[tokio::test]
async fn a_write_event_omits_the_byte_count_when_disabled() {
    let (sink, events) = collecting_sink();
    let (client, inner) = local_session().await;
    let session = InstrumentedSession::new(inner.clone(), Arc::new(sink))
        .with_policy(EventPayloadPolicy::new().with_write_len(false));

    session.write_file("x.txt", b"hello").await.unwrap();

    let events = events.lock().unwrap().clone();
    let start = events
        .iter()
        .find(|event| event.op == "write" && event.phase == EventPhase::Start)
        .expect("a write start event");
    assert!(start.data.iter().all(|(key, _)| key != "bytes"));
    client.delete(inner).await.unwrap();
}

/// Python: `test_sandbox_session_exec_emits_stdout_when_enabled`.
#[tokio::test]
async fn an_exec_finish_event_carries_stdout_when_enabled() {
    let (sink, events) = collecting_sink();
    let (client, inner) = local_session().await;
    let session = InstrumentedSession::new(inner.clone(), Arc::new(sink))
        .with_policy(EventPayloadPolicy::new().with_exec_output(true));

    let result = session
        .exec(openai_agents_sdk::sandbox::ExecRequest::shell("echo hi"))
        .await
        .unwrap();
    assert!(result.ok());

    let events = events.lock().unwrap().clone();
    let finish = events
        .iter()
        .find(|event| event.op == "exec" && event.phase == EventPhase::Finish)
        .expect("an exec finish event");
    assert!(finish.stdout.as_deref().unwrap_or_default().contains("hi"));
    assert_eq!(finish.trace_id, None);
    assert_eq!(finish.stdout_bytes, None);
    client.delete(inner).await.unwrap();
}

/// Python: `test_jsonl_outbox_sink_appends_one_line_per_event`.
#[tokio::test]
async fn a_jsonl_sink_appends_one_line_per_event() {
    let temp = tempfile::tempdir().unwrap();
    let outbox = temp.path().join("events.jsonl");
    let sink = JsonlFileSink::new(&outbox);
    let start =
        SandboxSessionEvent::start("session", "unix_local", 1, "span_write", SandboxOp::Write);
    let finish = start.finish(true, 0);

    sink.emit(&start).await;
    sink.emit(&finish).await;
    sink.flush().await;

    let text = std::fs::read_to_string(&outbox).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        serde_json::from_str::<Value>(lines[0]).unwrap()["phase"],
        "start"
    );
    assert_eq!(
        serde_json::from_str::<Value>(lines[1]).unwrap()["phase"],
        "finish"
    );
}

/// Python: `test_chained_sink_runs_in_order`.
#[tokio::test]
async fn a_chained_sink_runs_in_order() {
    let temp = tempfile::tempdir().unwrap();
    let outbox = temp.path().join("events.jsonl");
    let seen: Arc<Mutex<Vec<usize>>> = Arc::default();
    let observer = {
        let outbox = outbox.clone();
        let seen = seen.clone();
        CallbackSink::new(move |_event| {
            let lines = std::fs::read_to_string(&outbox)
                .map(|text| text.lines().count())
                .unwrap_or(0);
            seen.lock().unwrap().push(lines);
        })
    };
    let chain = ChainedSink::new()
        .with_sink(Arc::new(JsonlFileSink::new(&outbox)))
        .with_sink(Arc::new(observer));
    let instrumentation = Instrumentation::new(Arc::new(chain), "session", "unix_local");
    let start =
        SandboxSessionEvent::start("session", "unix_local", 1, "span_write", SandboxOp::Write);
    let finish = start.finish(true, 0);

    instrumentation.emit(start).await;
    instrumentation.emit(finish).await;

    assert_eq!(*seen.lock().unwrap(), vec![1, 2]);
}

/// Python: `test_http_proxy_sink_spools_direct_timeout`. The Python test patches `urlopen` to
/// time out; here the endpoint is a closed port, which fails the same way.
#[tokio::test]
async fn an_http_proxy_sink_spools_a_failed_post() {
    let temp = tempfile::tempdir().unwrap();
    let spool = temp.path().join("events.jsonl");
    let sink = HttpProxySink::new("http://127.0.0.1:9/events").with_spool_path(&spool);
    let event = SandboxSessionEvent::start("session", "unix_local", 1, "span", SandboxOp::Write);

    sink.emit(&event).await;
    sink.flush().await;

    let text = std::fs::read_to_string(&spool).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1);
    assert_eq!(serde_json::from_str::<Value>(lines[0]).unwrap()["seq"], 1);
}

/// Python: `test_sandbox_session_error_events_and_traces_include_retryability`, the event half.
/// The trace-span half has no Rust counterpart: sandbox operations do not open spans.
#[tokio::test]
async fn an_error_event_carries_the_retryability() {
    let (sink, events) = collecting_sink();
    let (client, inner) = local_session().await;
    let session = InstrumentedSession::new(inner.clone(), Arc::new(sink));

    let error = session.read_file("missing.txt").await.unwrap_err();
    assert_eq!(error.retryable, Some(false));

    let events = events.lock().unwrap().clone();
    let finish = events
        .iter()
        .find(|event| event.op == "read" && event.phase == EventPhase::Finish)
        .expect("a read finish event");
    assert_eq!(finish.error_retryable, Some(false));
    assert_eq!(finish.ok, Some(false));
    client.delete(inner).await.unwrap();
}
