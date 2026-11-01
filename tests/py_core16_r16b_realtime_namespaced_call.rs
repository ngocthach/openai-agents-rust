//! Port of `tests/realtime/test_session.py::test_namespaced_realtime_call_rebinds_active_identity_after_resolution`:
//! a call to a namespaced tool that is still running keeps its identity once routing resolves it,
//! so an identical delivery is ignored and a delivery with different arguments under the same call
//! id fails without running the tool again.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModelEvent, RealtimeModelToolCallEvent,
    RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::tool::{FunctionTool, tool_namespace};
use serde_json::json;
use tokio::sync::Notify;

type Event = RealtimeSessionEvent<()>;

const WAIT: Duration = Duration::from_secs(5);

async fn recv(session: &RealtimeSession<()>) -> Event {
    tokio::time::timeout(WAIT, session.recv())
        .await
        .expect("timed out waiting for a session event")
        .expect("session stream ended")
}

async fn until(session: &RealtimeSession<()>, predicate: impl Fn(&Event) -> bool) -> Vec<Event> {
    let mut seen = Vec::new();
    loop {
        let event = recv(session).await;
        let done = predicate(&event);
        seen.push(event);
        if done {
            return seen;
        }
    }
}

fn call(arguments: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent::new(
        "lookup_account",
        "call_namespaced_active",
        arguments,
    ))
}

#[tokio::test]
async fn a_running_namespaced_call_keeps_its_identity_after_resolution() {
    let (started, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    let runs = Arc::new(AtomicUsize::new(0));
    let (s, r, n) = (started.clone(), release.clone(), runs.clone());
    let gated = FunctionTool::<()>::new_raw(
        "lookup_account",
        "Look up an account.",
        json!({"type": "object", "properties": {}}),
        move |_ctx, _args| {
            let (s, r, n) = (s.clone(), r.clone(), n.clone());
            async move {
                n.fetch_add(1, Ordering::SeqCst);
                s.notify_one();
                r.notified().await;
                Ok("ok".to_string())
            }
        },
    );
    let namespaced = tool_namespace("crm", "CRM tools", vec![gated])
        .unwrap()
        .remove(0);
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(namespaced)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = RealtimeSession::connect(
        model.clone(),
        agent,
        (),
        None,
        Some(RealtimeRunConfig::<()>::default()),
    )
    .await
    .unwrap();

    model.emit(call("{}"));
    tokio::time::timeout(WAIT, started.notified())
        .await
        .unwrap();
    // The same delivery again is ignored; a changed one under the same call id is an error.
    model.emit(call("{}"));
    model.emit(call(r#"{"changed":true}"#));
    let events = until(&session, |e| {
        matches!(e, RealtimeSessionEvent::Error { .. })
    })
    .await;
    let RealtimeSessionEvent::Error { error, .. } = events.last().unwrap() else {
        unreachable!()
    };
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("unique call ID"),
        "{error}"
    );

    release.notify_one();
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let outputs: Vec<String> = model
        .sent_events()
        .iter()
        .filter_map(|event| {
            event
                .as_tool_output()
                .map(|(_, output, _)| output.to_string())
        })
        .collect();
    assert_eq!(outputs, ["ok"]);
}
