//! Python `tests/sandbox/test_session_sinks.py::test_callback_sink_receives_bound_inner_session`:
//! a callback that also takes the session is handed the wrapped session, not the wrapper.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::events::{CallbackSink, InstrumentedSession};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};

fn address(session: &Arc<dyn SandboxSession>) -> usize {
    Arc::as_ptr(session) as *const () as usize
}

#[tokio::test]
async fn a_callback_sink_receives_the_bound_inner_session() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = UnixLocalSandboxClient::new()
        .create(Some(Manifest::new().with_root(root.to_string_lossy())))
        .await
        .unwrap();
    let inner_address = address(&inner);

    let seen: Arc<Mutex<Vec<(String, usize)>>> = Arc::default();
    let log = seen.clone();
    let sink = CallbackSink::with_session(move |event, session| {
        log.lock()
            .unwrap()
            .push((event.op.to_string(), address(session)));
    });
    let wrapped = InstrumentedSession::new(inner, Arc::new(sink));

    wrapped.start().await.unwrap();
    wrapped.exec(ExecRequest::shell("echo hi")).await.unwrap();
    wrapped.stop().await.unwrap();

    let seen = seen.lock().unwrap().clone();
    assert!(seen.iter().any(|(op, _)| op == "exec"), "{seen:?}");
    assert!(
        seen.iter().all(|(_, session)| *session == inner_address),
        "{seen:?}"
    );
}
