//! Port of `test_callback_sink_receives_bound_inner_session` in
//! `tests/sandbox/test_session_sinks.py`: a session-aware `CallbackSink` is handed the session
//! it was bound to, and one that was never bound reports it.
#![cfg(all(feature = "sandbox", unix))]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::ExecRequest;
use openai_agents_sdk::sandbox::events::{
    CallbackSink, ChainedSink, Instrumentation, InstrumentedSession,
};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::{ErrorCode, SandboxOp};

async fn local_session(root: &std::path::Path) -> Arc<dyn SandboxSession> {
    let manifest = Manifest::new().with_root(root.to_string_lossy());
    let session = UnixLocalSandboxClient::new()
        .create(Some(manifest))
        .await
        .expect("create");
    session.start().await.expect("start");
    session
}

fn same_session(a: &Arc<dyn SandboxSession>, b: &Arc<dyn SandboxSession>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

#[tokio::test]
async fn callback_sink_receives_the_bound_inner_session() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = local_session(&root).await;
    let seen: Arc<Mutex<Vec<(String, bool)>>> = Arc::default();
    let record = seen.clone();
    let expected = inner.clone();
    let sink = CallbackSink::with_session(move |event, session| {
        record
            .lock()
            .unwrap()
            .push((event.op.clone(), same_session(session, &expected)));
    });
    let wrapped = InstrumentedSession::new(inner, Arc::new(sink));

    wrapped
        .exec(ExecRequest::argv(["echo", "hi"]))
        .await
        .expect("exec");

    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|(_, is_inner)| *is_inner), "{seen:?}");
}

#[tokio::test]
async fn a_chained_session_aware_sink_is_bound_too() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = local_session(&root).await;
    let count = Arc::new(Mutex::new(0usize));
    let counter = count.clone();
    let chain = ChainedSink::new().with_sink(Arc::new(CallbackSink::with_session(
        move |_event, _session| *counter.lock().unwrap() += 1,
    )));
    let wrapped = InstrumentedSession::new(inner, Arc::new(chain));

    wrapped
        .exec(ExecRequest::argv(["true"]))
        .await
        .expect("exec");

    assert!(*count.lock().unwrap() >= 2);
}

#[tokio::test]
async fn an_unbound_session_aware_sink_reports_it() {
    let sink = CallbackSink::raising_with_session(|_event, _session| Ok(()));
    let instrumentation = Instrumentation::new(Arc::new(sink), "session", "unix_local");

    let error = instrumentation
        .try_start(SandboxOp::Exec, Vec::new())
        .await
        .expect_err("no bound session");

    assert_eq!(error.code, ErrorCode::EventSinkFailed);
    // The sink's own text ("requires a bound session") travels only where tool data may be
    // logged, so the default policy shows the generic failure.
    assert_eq!(error.message, "sandbox event sink failed");
}
