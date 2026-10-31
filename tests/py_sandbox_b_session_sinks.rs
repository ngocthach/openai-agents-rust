//! Port of `test_sandbox_session_apply_manifest_preserves_write_instrumentation` from
//! `tests/sandbox/test_session_sinks.py`: writes made while a manifest is applied through an
//! instrumented session are recorded as write start and finish events.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::events::{
    CallbackSink, EventPhase, InstrumentedSession, SandboxSessionEvent,
};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};

#[tokio::test]
async fn apply_manifest_preserves_write_instrumentation() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let manifest = Manifest::new()
        .with_root(root.to_string_lossy())
        .with_entry("materialized.txt", Entry::text_file("hello"));
    let inner = UnixLocalSandboxClient::new()
        .create(Some(manifest.clone()))
        .await
        .unwrap();

    let events: Arc<Mutex<Vec<SandboxSessionEvent>>> = Arc::default();
    let seen = events.clone();
    let sink = CallbackSink::new(move |event| seen.lock().unwrap().push(event.clone()));
    let session = InstrumentedSession::new(inner, Arc::new(sink));

    manifest.apply(&session).await.unwrap();

    let phases: Vec<EventPhase> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.op == "write")
        .map(|event| event.phase)
        .collect();
    assert_eq!(phases, vec![EventPhase::Start, EventPhase::Finish]);
    assert_eq!(
        std::fs::read(root.join("materialized.txt")).unwrap(),
        b"hello"
    );
}

/// The wrapper forwards `ls`, `mkdir`, `rm` and `mv` to the inner session, the way Python's
/// wrapper leaves its helper operations to the session it wraps. Their effects reach the workspace
/// and no event is recorded for them: the wrapper's own instrumented operations are the `_as`
/// variants, read, write, exec and the lifecycle calls.
#[tokio::test]
async fn filesystem_helpers_are_forwarded_without_events() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = UnixLocalSandboxClient::new()
        .create(Some(Manifest::new().with_root(root.to_string_lossy())))
        .await
        .unwrap();
    inner.start().await.unwrap();

    let events: Arc<Mutex<Vec<SandboxSessionEvent>>> = Arc::default();
    let seen = events.clone();
    let sink = CallbackSink::new(move |event| seen.lock().unwrap().push(event.clone()));
    let session = InstrumentedSession::new(inner, Arc::new(sink));

    session.mkdir("dir", false).await.unwrap();
    std::fs::write(root.join("dir").join("a.txt"), b"x").unwrap();
    session.mv("dir/a.txt", "dir/b.txt").await.unwrap();
    let listing = session.ls("dir").await.unwrap();
    assert!(
        listing.iter().any(|entry| entry.name == "b.txt"),
        "ls sees the moved file"
    );
    session.rm("dir/b.txt", false).await.unwrap();
    assert!(!root.join("dir").join("b.txt").exists());

    let recorded = events.lock().unwrap().clone();
    assert!(
        recorded.is_empty(),
        "forwarded helpers record no event: {:?}",
        recorded
            .iter()
            .map(|event| (event.op.clone(), event.phase))
            .collect::<Vec<_>>()
    );
}
