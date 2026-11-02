//! Ports from the Python `tests/realtime/test_session.py` python-only bucket.
//!
//! - `test_on_guardrail_task_done_emits_error_event`: a guardrail task that fails outside the
//!   guardrail function (here the interrupt send after a tripwire) is reported as
//!   `Guardrail task failed: <error>` and the session keeps running.
//! - `test_realtime_tool_contexts_share_session_tool_state`: tool input guardrails and the tool
//!   body see the session's approval state, not a private copy.

#![cfg(feature = "realtime")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeGuardrailsSettings, RealtimeModel,
    RealtimeModelConfig, RealtimeModelEvent, RealtimeModelEvents, RealtimeModelSendEvent,
    RealtimeModelToolCallEvent, RealtimeRunConfig, RealtimeSendCondition, RealtimeSession,
    RealtimeSessionEvent,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{AgentsError, GuardrailFunctionOutput, OutputGuardrail};
use serde_json::json;

type Event = RealtimeSessionEvent<()>;
/// Who looked up the approval, and what it saw.
type Observed = Arc<Mutex<Vec<(&'static str, Option<bool>)>>>;

const WAIT: Duration = Duration::from_secs(5);

/// A transport whose interrupt sends fail; everything else goes to the fake model.
#[derive(Debug)]
struct FailingInterruptModel {
    inner: Arc<FakeRealtimeModel>,
}

#[async_trait]
impl RealtimeModel for FailingInterruptModel {
    async fn connect(
        &self,
        config: RealtimeModelConfig,
    ) -> openai_agents_sdk::Result<RealtimeModelEvents> {
        self.inner.connect(config).await
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> openai_agents_sdk::Result<()> {
        if matches!(event, RealtimeModelSendEvent::Interrupt(_)) {
            return Err(AgentsError::other("interrupt transport down"));
        }
        self.inner.send_event(event).await
    }

    async fn send_event_if(
        &self,
        event: RealtimeModelSendEvent,
        condition: RealtimeSendCondition,
    ) -> openai_agents_sdk::Result<bool> {
        self.inner.send_event_if(event, condition).await
    }

    async fn retire_response_audio(&self, response_id: &str) -> openai_agents_sdk::Result<()> {
        self.inner.retire_response_audio(response_id).await
    }

    async fn close(&self) -> openai_agents_sdk::Result<()> {
        self.inner.close().await
    }
}

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

fn error_message(event: &Event) -> Option<String> {
    match event {
        RealtimeSessionEvent::Error { error, .. } => error["message"].as_str().map(str::to_string),
        _ => None,
    }
}

fn is_marker(event: &Event) -> bool {
    matches!(
        event,
        RealtimeSessionEvent::RawModelEvent {
            data: RealtimeModelEvent::Other { data },
            ..
        } if data == "barrier"
    )
}

#[tokio::test]
async fn a_failed_guardrail_interrupt_is_reported_as_a_guardrail_task_failure() {
    let tripping = OutputGuardrail::<()>::new("tripping", |_ctx, _agent, _output| async move {
        Ok(GuardrailFunctionOutput::new(json!({}), true))
    });
    let inner = Arc::new(FakeRealtimeModel::new());
    let model = Arc::new(FailingInterruptModel {
        inner: inner.clone(),
    });
    let config = RealtimeRunConfig::<()> {
        output_guardrails: vec![tripping],
        guardrails_settings: RealtimeGuardrailsSettings {
            debounce_text_length: 5,
        },
        ..Default::default()
    };
    let session = RealtimeSession::connect(
        model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        (),
        None,
        Some(config),
    )
    .await
    .expect("session connects");

    inner.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "item_1".into(),
        delta: "model text".into(),
        response_id: "response_1".into(),
    });
    let events = until(&session, |event| error_message(event).is_some()).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::GuardrailTripped { .. })),
        "the tripwire is reported before the failure"
    );
    let message = error_message(events.last().unwrap()).unwrap();
    assert_eq!(message, "Guardrail task failed: interrupt transport down");

    // The failure neither ends the session nor sends the guardrail feedback.
    assert!(inner.emit(RealtimeModelEvent::Other {
        data: json!("barrier"),
    }));
    until(&session, is_marker).await;
    assert!(session.take_error().is_none());
    assert!(
        inner
            .sent_events()
            .iter()
            .all(|event| !matches!(event, RealtimeModelSendEvent::UserInput { .. })),
        "no feedback follows a failed interrupt"
    );
    session.close().await.unwrap();
}

#[tokio::test]
async fn tool_guardrails_and_tools_see_the_session_approval_state() {
    let observed: Observed = Arc::default();

    let seen_by_guardrail = observed.clone();
    let guardrail = ToolInputGuardrail::<()>::new("capture", move |data| {
        let seen = seen_by_guardrail.clone();
        async move {
            let approved =
                data.context
                    .run_context()
                    .is_tool_approved("agent", "approval_tool", "call_1");
            seen.lock().unwrap().push(("guardrail", approved));
            Ok(ToolGuardrailFunctionOutput::allow())
        }
    });
    let seen_by_tool = observed.clone();
    let tool = FunctionTool::<()>::new_raw(
        "approval_tool",
        "guarded",
        json!({"type": "object", "properties": {}}),
        move |ctx, _args| {
            let approved = ctx
                .run_context()
                .is_tool_approved("agent", "approval_tool", "call_1");
            seen_by_tool.lock().unwrap().push(("tool", approved));
            async move { Ok("ok".to_string()) }
        },
    )
    .with_needs_approval(true)
    .with_input_guardrail(guardrail);

    let model = Arc::new(FakeRealtimeModel::new());
    let session = RealtimeSession::connect(
        model.clone(),
        RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc(),
        (),
        None,
        Some(RealtimeRunConfig {
            async_tool_calls: false,
            ..Default::default()
        }),
    )
    .await
    .expect("session connects");

    model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("approval_tool", "call_1", "{}"),
    ));
    until(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    assert!(observed.lock().unwrap().is_empty());

    // The decision is recorded through the session; the guardrail and the tool read it back
    // through their own tool contexts.
    session.approve_tool_call("call_1", false).await.unwrap();
    assert_eq!(
        *observed.lock().unwrap(),
        [("guardrail", Some(true)), ("tool", Some(true))]
    );
    assert_eq!(
        session
            .context()
            .is_tool_approved("agent", "approval_tool", "call_1"),
        Some(true)
    );
    assert_eq!(
        model.tool_outputs(),
        [("call_1".to_string(), "ok".to_string())]
    );
    session.close().await.unwrap();
}
