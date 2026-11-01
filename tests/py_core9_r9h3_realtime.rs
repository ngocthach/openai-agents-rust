//! Ports of the realtime approval cases `tests/realtime/test_session_approvals.py` left without a
//! Rust counterpart: what a rejected call sends to the model.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModelEvent, RealtimeModelToolCallEvent,
    RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

type Event = RealtimeSessionEvent<()>;

const WAIT: Duration = Duration::from_secs(5);

fn guarded_tool(runs: Arc<AtomicUsize>) -> FunctionTool<()> {
    FunctionTool::new_raw(
        "test_function",
        "test tool",
        json!({"type": "object", "properties": {}}),
        move |_ctx, _args| {
            runs.fetch_add(1, Ordering::SeqCst);
            async { Ok("ok".to_string()) }
        },
    )
    .with_needs_approval(true)
}

fn call(call_id: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent::new(
        "test_function",
        call_id,
        "{}",
    ))
}

async fn until(session: &RealtimeSession<()>, predicate: impl Fn(&Event) -> bool) -> Vec<Event> {
    let mut seen = Vec::new();
    loop {
        let event = tokio::time::timeout(WAIT, session.recv())
            .await
            .expect("timed out waiting for a session event")
            .expect("session stream ended");
        let done = predicate(&event);
        seen.push(event);
        if done {
            return seen;
        }
    }
}

fn outputs(model: &FakeRealtimeModel) -> Vec<(String, String, bool)> {
    model
        .sent_events()
        .iter()
        .filter_map(|event| {
            event
                .as_tool_output()
                .map(|(call, output, start)| (call.call_id.clone(), output.to_string(), start))
        })
        .collect()
}

async fn pending_session(
    model: &Arc<FakeRealtimeModel>,
    run: RealtimeRunConfig<()>,
    runs: Arc<AtomicUsize>,
    call_id: &str,
) -> RealtimeSession<()> {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded_tool(runs))
        .into_arc();
    let session = RealtimeSession::connect(model.clone(), agent, (), None, Some(run))
        .await
        .expect("session connects");
    model.emit(call(call_id));
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    session
}

fn tool_end_output(events: &[Event]) -> String {
    match events.last().unwrap() {
        RealtimeSessionEvent::ToolEnd { output, .. } => output.to_model_string(),
        other => panic!("expected tool_end, got {}", other.event_type()),
    }
}

/// Python: `test_reject_pending_tool_call_sends_rejection_output`. A rejection without a message
/// tells the model the SDK's default text and asks it to respond; the tool never runs, and the
/// model delivering the same call again neither runs it nor produces a second output.
#[tokio::test]
async fn a_rejection_sends_the_default_message_once() {
    let model = Arc::new(FakeRealtimeModel::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let session = pending_session(&model, Default::default(), runs.clone(), "call_reject").await;

    session
        .reject_tool_call("call_reject", false, None)
        .await
        .unwrap();
    let events = until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    assert_eq!(tool_end_output(&events), "Tool execution was not approved.");

    model.emit(call("call_reject"));
    model.emit(RealtimeModelEvent::Other {
        data: json!("barrier"),
    });
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::RawModelEvent { data: RealtimeModelEvent::Other { data }, .. } if data == "barrier")
    })
    .await;

    assert_eq!(runs.load(Ordering::SeqCst), 0);
    let sent = outputs(&model);
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0].1, "Tool execution was not approved.");
    assert!(sent[0].2, "the model is asked to respond");
}

/// Python: `test_reject_pending_tool_call_prefers_explicit_message`. An explicit rejection message
/// wins over the run-level tool error formatter.
#[tokio::test]
async fn an_explicit_rejection_message_wins_over_the_run_level_formatter() {
    let model = Arc::new(FakeRealtimeModel::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let run = RealtimeRunConfig {
        tool_error_formatter: Some(openai_agents_sdk::tool_error_formatter_fn(|args| {
            Some(format!(
                "run-level {} denied ({})",
                args.tool_name, args.call_id
            ))
        })),
        ..Default::default()
    };
    let session = pending_session(&model, run, runs.clone(), "call_reject_explicit").await;

    session
        .reject_tool_call(
            "call_reject_explicit",
            false,
            Some("explicit rejection message".into()),
        )
        .await
        .unwrap();

    let sent = outputs(&model);
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0].1, "explicit rejection message");
    assert!(sent[0].2);
    let events = until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    assert_eq!(tool_end_output(&events), "explicit rejection message");
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}
