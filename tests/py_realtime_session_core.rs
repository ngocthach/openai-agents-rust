//! Ports of the Python `tests/realtime/test_session.py` lifecycle, event forwarding, tool
//! dispatch and model-settings tests.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeHandoff, RealtimeModel, RealtimeModelConfig,
    RealtimeModelEvent, RealtimeModelSendEvent, RealtimeModelToolCallEvent, RealtimeRunConfig,
    RealtimeSession, RealtimeSessionEvent, RealtimeSessionModelSettings, RealtimeStep,
    RealtimeVoice, ScriptedRealtimeModel, realtime_handoff,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::usage::Usage;
use serde_json::json;
use tokio::sync::Notify;

type Event = RealtimeSessionEvent<()>;

const WAIT: Duration = Duration::from_secs(5);

fn raw_tool(name: &str, output: &str) -> FunctionTool<()> {
    let output = output.to_string();
    FunctionTool::new_raw(
        name,
        "test tool",
        json!({"type": "object", "properties": {}}),
        move |_ctx, _args| {
            let output = output.clone();
            async move { Ok(output) }
        },
    )
}

/// A tool that signals `started`, then waits for `release` before answering `output`.
fn gated_tool(
    name: &str,
    output: &str,
    started: Arc<Notify>,
    release: Arc<Notify>,
    runs: Arc<AtomicUsize>,
) -> FunctionTool<()> {
    let output = output.to_string();
    FunctionTool::new_raw(
        name,
        "gated tool",
        json!({"type": "object", "properties": {}}),
        move |_ctx, _args| {
            let (output, started, release, runs) = (
                output.clone(),
                started.clone(),
                release.clone(),
                runs.clone(),
            );
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                started.notify_one();
                release.notified().await;
                Ok(output)
            }
        },
    )
}

fn sync_config() -> RealtimeRunConfig<()> {
    RealtimeRunConfig {
        async_tool_calls: false,
        ..Default::default()
    }
}

fn call(name: &str, call_id: &str, arguments: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent::new(name, call_id, arguments))
}

async fn start(
    model: &Arc<FakeRealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
    run: RealtimeRunConfig<()>,
    model_config: Option<RealtimeModelConfig>,
) -> RealtimeSession<()> {
    RealtimeSession::connect(model.clone(), agent, (), model_config, Some(run))
        .await
        .expect("session connects")
}

async fn recv(session: &RealtimeSession<()>) -> Event {
    tokio::time::timeout(WAIT, session.recv())
        .await
        .expect("timed out waiting for a session event")
        .expect("session stream ended")
}

/// Collect events up to and including the first one matching `predicate`.
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

fn is_marker(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::RawModelEvent { data: RealtimeModelEvent::Other { data }, .. } if data == "barrier")
}

/// Everything the session emits for the events fed so far. The marker is raw-only, so once it
/// comes back every earlier event (handled in order by the session) has been delivered.
async fn drain(model: &FakeRealtimeModel, session: &RealtimeSession<()>) -> Vec<Event> {
    assert!(model.emit(RealtimeModelEvent::Other {
        data: json!("barrier"),
    }));
    let mut events = until(session, is_marker).await;
    events.pop();
    events
}

fn types(events: &[Event]) -> Vec<&'static str> {
    events.iter().map(Event::event_type).collect()
}

fn tool_outputs(model: &FakeRealtimeModel) -> Vec<(String, String, bool)> {
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

fn outputs_only(model: &FakeRealtimeModel) -> Vec<String> {
    tool_outputs(model).into_iter().map(|(_, o, _)| o).collect()
}

fn is_error(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::Error { .. })
}

fn error_message(event: &Event) -> String {
    match event {
        RealtimeSessionEvent::Error { error, .. } => error["message"].as_str().unwrap().to_string(),
        other => panic!("expected an error event, got {}", other.event_type()),
    }
}

fn dyn_model(model: &Arc<FakeRealtimeModel>) -> Arc<dyn RealtimeModel> {
    model.clone()
}

// ---------------------------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn send_helpers_reach_the_model() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        Default::default(),
        None,
    )
    .await;
    assert!(Arc::ptr_eq(session.model(), &dyn_model(&model)));

    session.send_message("hi").await.unwrap();
    session.send_audio(b"abc".to_vec(), true).await.unwrap();
    session.interrupt().await.unwrap();

    let sent = model.sent_events();
    assert!(
        sent.iter()
            .any(|e| matches!(e, RealtimeModelSendEvent::UserInput { .. }))
    );
    assert!(sent.iter().any(
        |e| matches!(e, RealtimeModelSendEvent::Audio { audio, commit } if audio == b"abc" && *commit)
    ));
    assert!(
        sent.iter()
            .any(|e| e.as_interrupt().is_some_and(|i| i.force_response_cancel))
    );
    session.close().await.unwrap();
}

#[tokio::test]
async fn waiting_receivers_end_when_the_session_closes() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        Default::default(),
        None,
    )
    .await;
    // The connect-time history_updated event is the first thing a reader sees.
    assert_eq!(recv(&session).await.event_type(), "history_updated");

    let first = tokio::spawn({
        let session = session.clone();
        async move { session.recv().await.is_none() }
    });
    let second = tokio::spawn({
        let session = session.clone();
        async move { session.recv().await.is_none() }
    });
    tokio::task::yield_now().await;

    session.close().await.unwrap();
    assert!(tokio::time::timeout(WAIT, first).await.unwrap().unwrap());
    assert!(tokio::time::timeout(WAIT, second).await.unwrap().unwrap());
}

/// A transport whose first `close` fails; later closes succeed.
#[derive(Debug)]
struct FlakyCloseModel {
    inner: FakeRealtimeModel,
    close_calls: AtomicUsize,
}

#[async_trait::async_trait]
impl RealtimeModel for FlakyCloseModel {
    async fn connect(
        &self,
        config: RealtimeModelConfig,
    ) -> openai_agents_sdk::error::Result<openai_agents_sdk::realtime::RealtimeModelEvents> {
        self.inner.connect(config).await
    }
    async fn send_event(
        &self,
        event: RealtimeModelSendEvent,
    ) -> openai_agents_sdk::error::Result<()> {
        self.inner.send_event(event).await
    }
    async fn close(&self) -> openai_agents_sdk::error::Result<()> {
        if self.close_calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(AgentsError::other("close failed"));
        }
        self.inner.close().await
    }
}

#[tokio::test]
async fn a_failed_model_close_is_retried_by_the_next_close() {
    let model = Arc::new(FlakyCloseModel {
        inner: FakeRealtimeModel::new(),
        close_calls: AtomicUsize::new(0),
    });
    let session = RealtimeSession::connect(
        model.clone(),
        RealtimeAgent::<()>::new("agent").into_arc(),
        (),
        None,
        None,
    )
    .await
    .unwrap();

    let error = session.close().await.expect_err("the first close fails");
    assert!(error.to_string().contains("close failed"));
    assert!(session.is_closed());
    assert_eq!(model.close_calls.load(Ordering::SeqCst), 1);

    session.close().await.expect("the retry succeeds");
    assert_eq!(model.close_calls.load(Ordering::SeqCst), 2);
    // Once closed, further calls do nothing.
    session.close().await.unwrap();
    assert_eq!(model.close_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn agent_end_names_the_agent_whose_turn_ended_after_handoff() {
    let agent_b = RealtimeAgent::<()>::new("b").into_arc();
    let agent_a = RealtimeAgent::<()>::new("a")
        .with_handoff(agent_b.clone())
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent_a, sync_config(), None).await;

    model.emit(RealtimeModelEvent::TurnStarted {
        response_id: Some("response_1".into()),
    });
    model.emit(call("transfer_to_b", "call_1", "{}"));
    model.emit(RealtimeModelEvent::TurnEnded {
        response_id: Some("response_1".into()),
    });
    let events = drain(&model, &session).await;

    let names = |wanted: &str| -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match (wanted, event) {
                ("start", RealtimeSessionEvent::AgentStart { agent, .. }) => {
                    Some(agent.name.clone())
                }
                ("end", RealtimeSessionEvent::AgentEnd { agent, .. }) => Some(agent.name.clone()),
                (
                    "handoff",
                    RealtimeSessionEvent::Handoff {
                        from_agent,
                        to_agent,
                        ..
                    },
                ) => Some(format!("{}->{}", from_agent.name, to_agent.name)),
                _ => None,
            })
            .collect()
    };
    assert_eq!(names("start"), ["a"]);
    assert_eq!(names("handoff"), ["a->b"]);
    // The turn that ended was produced by agent a; the handoff only changes who speaks next.
    assert_eq!(names("end"), ["a"]);
    assert_eq!(session.current_agent_name(), "b");
}

#[tokio::test]
async fn a_tool_can_close_the_session_it_runs_in() {
    let model = Arc::new(FakeRealtimeModel::new());
    let slot: Arc<std::sync::Mutex<Option<RealtimeSession<()>>>> = Arc::default();
    let done = Arc::new(Notify::new());
    let tool = {
        let (slot, done) = (slot.clone(), done.clone());
        FunctionTool::new_raw(
            "closer",
            "closes the session",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                let (slot, done) = (slot.clone(), done.clone());
                async move {
                    let session = slot.lock().unwrap().clone().expect("session stored");
                    session.close().await?;
                    done.notify_one();
                    Ok("closed".to_string())
                }
            },
        )
    };
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, Default::default(), None).await;
    *slot.lock().unwrap() = Some(session.clone());

    model.emit(call("closer", "call_close", "{}"));
    tokio::time::timeout(WAIT, done.notified())
        .await
        .expect("the tool finished closing the session");
    assert!(session.is_closed());
    assert_eq!(model.close_calls(), 1);
}

#[tokio::test]
async fn a_late_tool_completion_cannot_send_after_close() {
    let model = Arc::new(FakeRealtimeModel::new());
    let (started, release, runs) = (
        Arc::new(Notify::new()),
        Arc::new(Notify::new()),
        Arc::new(AtomicUsize::new(0)),
    );
    let tool = gated_tool(
        "late",
        "late output",
        started.clone(),
        release.clone(),
        runs,
    );
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, Default::default(), None).await;

    model.emit(call("late", "late-call", "{}"));
    tokio::time::timeout(WAIT, started.notified())
        .await
        .unwrap();

    session.close().await.unwrap();
    release.notify_waiters();
    release.notify_one();
    tokio::task::yield_now().await;

    assert!(session.is_closed());
    assert!(tool_outputs(&model).is_empty());
    assert!(session.take_error().is_none());
}

#[tokio::test]
async fn model_events_after_close_change_nothing() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        Default::default(),
        None,
    )
    .await;
    session.close().await.unwrap();

    // The transport is gone, so nothing late can be fed; the session shows no events or history.
    assert!(
        !model.emit(RealtimeModelEvent::InputAudioTranscriptionCompleted {
            item_id: "late-item".into(),
            transcript: "late transcript".into(),
        })
    );
    assert!(session.history().is_empty());
    // Whatever was queued before close is dropped and the stream ends.
    while let Some(event) = tokio::time::timeout(WAIT, session.recv()).await.unwrap() {
        assert_eq!(event.event_type(), "history_updated");
    }
}

#[tokio::test]
async fn ambiguous_tool_and_handoff_names_fail_at_connect() {
    let target = RealtimeAgent::<()>::new("billing").into_arc();
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(raw_tool("transfer_to_billing", "ok"))
        .with_handoff(target)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let error = RealtimeSession::connect(model.clone(), agent, (), None, None)
        .await
        .expect_err("duplicate names are rejected");
    assert!(error.to_string().contains("function tool and handoff"));
    // Validation happens before the transport is touched.
    assert!(model.connect_calls().is_empty());
}

#[tokio::test]
async fn a_failed_connect_surfaces_and_records_the_attempt() {
    let model = Arc::new(ScriptedRealtimeModel::new([]).with_connect_error("connect failed"));
    let error = RealtimeSession::connect(
        model.clone(),
        RealtimeAgent::<()>::new("agent").into_arc(),
        (),
        None,
        None,
    )
    .await
    .expect_err("connect fails");
    assert!(error.to_string().contains("connect failed"));
    assert_eq!(model.connect_calls().len(), 1);
}

#[tokio::test]
async fn events_are_not_delivered_after_the_session_closed() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        Default::default(),
        None,
    )
    .await;
    session.close().await.unwrap();
    // A closed session has no listener: every send boundary reports nothing was delivered.
    assert!(!model.emit(RealtimeModelEvent::Error {
        error: json!({"message": "late error"}),
    }));
    let mut leftover = Vec::new();
    while let Some(event) = session.recv().await {
        leftover.push(event.event_type());
    }
    assert!(!leftover.contains(&"error"));
}

#[tokio::test]
async fn a_failing_tool_reports_a_tool_error_and_ends_the_session() {
    let model = Arc::new(FakeRealtimeModel::new());
    let tool = FunctionTool::new_raw(
        "boom",
        "fails",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async move { Err::<String, _>(AgentsError::other("tool failed")) },
    )
    .without_failure_error_function();
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, Default::default(), None).await;

    model.emit(call("boom", "call_boom", "{}"));
    let events = until(&session, is_error).await;
    assert!(types(&events).contains(&"tool_start"));
    // The event carries only Python's fixed message; the detail stays with `take_error`.
    assert_eq!(
        error_message(events.last().unwrap()),
        "Tool call task failed"
    );
    // No tool_end and no output: the call never completed.
    assert!(!types(&events).contains(&"tool_end"));
    assert!(tool_outputs(&model).is_empty());
    while session.recv().await.is_some() {}
    assert!(
        session
            .take_error()
            .unwrap()
            .to_string()
            .contains("tool failed")
    );
}

#[tokio::test]
async fn a_failed_output_send_emits_the_retry_error_event() {
    let model = Arc::new(ScriptedRealtimeModel::new([RealtimeStep::new(
        "tool_output",
    )
    .failing("send failed")]));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(raw_tool("lookup", "output"))
        .into_arc();
    let session = RealtimeSession::connect(model.clone(), agent, (), None, Some(sync_config()))
        .await
        .unwrap();
    model.emit(call("lookup", "call-1", "{}"));
    let events = until(&session, is_error).await;
    assert_eq!(
        error_message(events.last().unwrap()),
        "Tool output send failed; cached output will be retried"
    );
    assert!(!types(&events).contains(&"tool_end"));
}

// ---------------------------------------------------------------------------------------------
// Enablement and model settings built at connect
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn async_handoff_is_enabled_is_evaluated_for_every_handoff() {
    let target = RealtimeAgent::<()>::new("target").into_arc();
    let other = RealtimeAgent::<()>::new("other").into_arc();
    let direct = realtime_handoff(target).with_is_enabled_fn(|_ctx, _agent| async { true });
    let agent = RealtimeAgent::<()>::new("a")
        .with_handoff(direct)
        .with_handoff(other)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let _session = start(&model, agent, Default::default(), None).await;
    let tools = model.last_session_settings().unwrap().tools.unwrap();
    let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["transfer_to_target", "transfer_to_other"]);
}

#[tokio::test]
async fn a_disabled_handoff_does_not_conflict_with_a_same_named_tool() {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(raw_tool("transfer_to_billing", "ok"))
        .with_handoff(
            realtime_handoff(RealtimeAgent::<()>::new("billing").into_arc()).with_is_enabled(false),
        )
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let _session = start(&model, agent, Default::default(), None).await;
    let tools = model.last_session_settings().unwrap().tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "transfer_to_billing");
}

#[tokio::test]
async fn an_agent_handoff_is_not_reevaluated_when_settings_are_overridden() {
    let calls = Arc::new(AtomicUsize::new(0));
    let handoff = {
        let calls = calls.clone();
        realtime_handoff(RealtimeAgent::<()>::new("billing").into_arc()).with_is_enabled_fn(
            move |_ctx, _agent| {
                let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
                async move { first }
            },
        )
    };
    let agent = RealtimeAgent::<()>::new("agent")
        .with_handoff(handoff)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let config = RealtimeModelConfig {
        initial_model_settings: Some(
            RealtimeSessionModelSettings::new().with_voice(RealtimeVoice::Name("verse".into())),
        ),
        ..Default::default()
    };
    let _session = start(&model, agent, Default::default(), Some(config)).await;
    let tools = model.last_session_settings().unwrap().tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dispatch_rechecks_dynamic_function_tool_enablement() {
    let enabled = Arc::new(AtomicBool::new(true));
    let runs = Arc::new(AtomicUsize::new(0));
    let tool = {
        let (enabled, runs) = (enabled.clone(), runs.clone());
        FunctionTool::new_raw(
            "dynamic_tool",
            "dynamic",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                runs.fetch_add(1, Ordering::SeqCst);
                async move { Ok("should not run".to_string()) }
            },
        )
        .with_is_enabled_fn(move |_ctx, _agent| {
            let enabled = enabled.load(Ordering::SeqCst);
            async move { enabled }
        })
    };
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config(), None).await;
    enabled.store(false, Ordering::SeqCst);

    model.emit(call("dynamic_tool", "call_dynamic_tool_disabled", "{}"));
    drain(&model, &session).await;
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(outputs_only(&model), ["Tool dynamic_tool not found"]);
}

#[tokio::test]
async fn dispatch_rechecks_dynamic_handoff_enablement() {
    let enabled = Arc::new(AtomicBool::new(true));
    let target = RealtimeAgent::<()>::new("target").into_arc();
    let handoff = {
        let enabled = enabled.clone();
        realtime_handoff(target).with_is_enabled_fn(move |_ctx, _agent| {
            let enabled = enabled.load(Ordering::SeqCst);
            async move { enabled }
        })
    };
    let agent = RealtimeAgent::<()>::new("agent")
        .with_handoff(handoff)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config(), None).await;
    enabled.store(false, Ordering::SeqCst);

    model.emit(call(
        "transfer_to_target",
        "call_dynamic_handoff_disabled",
        "{}",
    ));
    drain(&model, &session).await;
    assert_eq!(session.current_agent_name(), "agent");
    assert_eq!(outputs_only(&model), ["Tool transfer_to_target not found"]);
}

#[tokio::test]
async fn connect_options_carry_the_agent_settings() {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_instructions("Agent instructions")
        .with_tool(raw_tool("agent_tool", "ok"))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let config = RealtimeModelConfig {
        initial_model_settings: Some(
            RealtimeSessionModelSettings::new()
                .with_voice(RealtimeVoice::Name("nova".into()))
                .with_model_name("gpt-4o-realtime"),
        ),
        ..Default::default()
    };
    let _session = start(&model, agent, Default::default(), Some(config)).await;
    let settings = model.connect_calls()[0]
        .initial_model_settings
        .clone()
        .unwrap();
    assert_eq!(settings.instructions.as_deref(), Some("Agent instructions"));
    assert_eq!(settings.tools.as_ref().unwrap()[0].name, "agent_tool");
    assert_eq!(settings.voice, Some(RealtimeVoice::Name("nova".into())));
    assert_eq!(settings.model_name.as_deref(), Some("gpt-4o-realtime"));
}

#[tokio::test]
async fn handoffs_are_included_in_model_settings() {
    let specialist = RealtimeAgent::<()>::new("specialist").into_arc();
    let handoff: RealtimeHandoff<()> = realtime_handoff(specialist);
    let tool_name = handoff.tool_name.clone();
    let agent = RealtimeAgent::<()>::new("agent")
        .with_instructions("Agent with handoffs")
        .with_handoff(handoff)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let _session = start(&model, agent, Default::default(), None).await;
    let settings = model.last_session_settings().unwrap();
    let names: Vec<_> = settings
        .tools
        .unwrap()
        .into_iter()
        .map(|spec| spec.name)
        .collect();
    assert_eq!(names, [tool_name]);
}

#[tokio::test]
async fn model_settings_follow_run_config_then_agent_then_model_config() {
    let agent = RealtimeAgent::<()>::new("test_agent")
        .with_instructions("agent_instructions")
        .into_arc();
    let run = RealtimeRunConfig {
        model_settings: Some(RealtimeSessionModelSettings {
            voice: Some(RealtimeVoice::Name("run_config_voice".into())),
            modalities: Some(vec!["text".into()]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let config = RealtimeModelConfig {
        initial_model_settings: Some(RealtimeSessionModelSettings {
            voice: Some(RealtimeVoice::Name("model_config_voice".into())),
            tool_choice: Some("auto".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let model = Arc::new(FakeRealtimeModel::new());
    let _session = start(&model, agent, run, Some(config)).await;
    let settings = model.last_session_settings().unwrap();
    assert_eq!(settings.instructions.as_deref(), Some("agent_instructions"));
    assert_eq!(settings.tools, Some(vec![]));
    // model_config wins over run_config, run_config survives where not overridden.
    assert_eq!(
        settings.voice,
        Some(RealtimeVoice::Name("model_config_voice".into()))
    );
    assert_eq!(settings.modalities, Some(vec!["text".to_string()]));
    assert_eq!(settings.tool_choice.as_deref(), Some("auto"));
}

#[tokio::test]
async fn model_config_only_settings_are_preserved() {
    let agent = RealtimeAgent::<()>::new("test_agent")
        .with_instructions("test_prompt")
        .into_arc();
    let config = RealtimeModelConfig {
        initial_model_settings: Some(RealtimeSessionModelSettings {
            voice: Some(RealtimeVoice::Name("model_config_only_voice".into())),
            tool_choice: Some("required".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let model = Arc::new(FakeRealtimeModel::new());
    let _session = start(&model, agent, RealtimeRunConfig::default(), Some(config)).await;
    let settings = model.last_session_settings().unwrap();
    assert_eq!(settings.instructions.as_deref(), Some("test_prompt"));
    assert_eq!(settings.tools, Some(vec![]));
    assert_eq!(
        settings.voice,
        Some(RealtimeVoice::Name("model_config_only_voice".into()))
    );
    assert_eq!(settings.tool_choice.as_deref(), Some("required"));
}

#[tokio::test]
async fn initial_settings_persist_when_the_agent_is_updated() {
    let first = RealtimeAgent::<()>::new("first")
        .with_instructions("first")
        .into_arc();
    let second = RealtimeAgent::<()>::new("second")
        .with_instructions("second")
        .into_arc();
    let config = RealtimeModelConfig {
        initial_model_settings: Some(RealtimeSessionModelSettings {
            voice: Some(RealtimeVoice::Name("initial_voice".into())),
            ..Default::default()
        }),
        ..Default::default()
    };
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, first, Default::default(), Some(config)).await;
    session.update_agent(second).await.unwrap();
    let update = model.session_updates().pop().expect("a session update");
    assert_eq!(
        update.voice,
        Some(RealtimeVoice::Name("initial_voice".into()))
    );
    assert_eq!(update.instructions.as_deref(), Some("second"));
}

#[tokio::test]
async fn update_agent_sends_a_session_update_and_switches_agent() {
    let first = RealtimeAgent::<()>::new("first")
        .with_instructions("first")
        .into_arc();
    let second = RealtimeAgent::<()>::new("second")
        .with_instructions("second")
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, first, Default::default(), None).await;
    session.update_agent(second).await.unwrap();

    let sent = model.sent_events();
    let update = sent[0].as_session_update().expect("session update first");
    assert_eq!(update.instructions.as_deref(), Some("second"));
    assert_eq!(session.current_agent_name(), "second");
}

fn ambiguous_agent(name: &str) -> Arc<RealtimeAgent<()>> {
    RealtimeAgent::<()>::new(name)
        .with_tool(raw_tool("transfer_to_billing", "ok"))
        .with_handoff(realtime_handoff(
            RealtimeAgent::<()>::new("billing").into_arc(),
        ))
        .into_arc()
}

#[tokio::test]
async fn update_agent_validation_failure_keeps_the_current_agent() {
    let first = RealtimeAgent::<()>::new("first").into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, first, Default::default(), None).await;
    let error = session
        .update_agent(ambiguous_agent("invalid_agent"))
        .await
        .expect_err("duplicate names");
    assert!(error.to_string().contains("Duplicate"));
    assert_eq!(session.current_agent_name(), "first");
    assert!(model.sent_events().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Event forwarding
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn error_events_are_forwarded_after_the_raw_event() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        sync_config(),
        None,
    )
    .await;
    model.emit(RealtimeModelEvent::Error {
        error: json!("Test error"),
    });
    let events = drain(&model, &session).await;
    // connect-time history, raw model event, then the transformed error.
    assert_eq!(
        types(&events),
        ["history_updated", "raw_model_event", "error"]
    );
    match &events[1] {
        RealtimeSessionEvent::RawModelEvent { data, .. } => {
            assert!(matches!(data, RealtimeModelEvent::Error { error } if error == "Test error"));
        }
        _ => unreachable!(),
    }
    match &events[2] {
        RealtimeSessionEvent::Error { error, .. } => assert_eq!(error, "Test error"),
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn turn_events_become_agent_start_and_end() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("agent").into_arc();
    let session = start(&model, agent.clone(), sync_config(), None).await;
    model.emit(RealtimeModelEvent::TurnStarted { response_id: None });
    model.emit(RealtimeModelEvent::TurnEnded { response_id: None });
    let events = drain(&model, &session).await;
    assert_eq!(
        types(&events),
        [
            "history_updated",
            "raw_model_event",
            "agent_start",
            "raw_model_event",
            "agent_end"
        ]
    );
    for event in &events {
        if let RealtimeSessionEvent::AgentStart { agent: a, .. }
        | RealtimeSessionEvent::AgentEnd { agent: a, .. } = event
        {
            assert!(Arc::ptr_eq(a, &agent));
        }
    }
}

#[tokio::test]
async fn usage_events_accumulate_in_the_session_context() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        sync_config(),
        None,
    )
    .await;
    for (input, output) in [(10, 4), (7, 3)] {
        model.emit(RealtimeModelEvent::Usage {
            usage: Box::new(Usage::single(input, output)),
            input_tokens_details: None,
            output_tokens_details: None,
        });
    }
    let events = drain(&model, &session).await;
    // history_updated + two raw usage events and nothing else.
    assert_eq!(
        types(&events),
        ["history_updated", "raw_model_event", "raw_model_event"]
    );
    let usage = events[2].info().context.usage();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.input_tokens, 17);
    assert_eq!(usage.output_tokens, 7);
    assert_eq!(usage.total_tokens, 24);
    assert_eq!(usage.request_usage_entries.len(), 2);
}

#[tokio::test]
async fn ignored_events_only_generate_raw_events() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        Default::default(),
        None,
    )
    .await;
    model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "item_1".into(),
        delta: "hello".into(),
        response_id: "resp_1".into(),
    });
    model.emit(RealtimeModelEvent::ConnectionStatus {
        status: openai_agents_sdk::realtime::RealtimeConnectionStatus::Connected,
    });
    model.emit(RealtimeModelEvent::Other {
        data: json!({"custom": "data"}),
    });
    let events = drain(&model, &session).await;
    assert_eq!(
        types(&events),
        [
            "history_updated",
            "raw_model_event",
            "raw_model_event",
            "raw_model_event"
        ]
    );
}

#[tokio::test]
async fn a_transient_disconnect_does_not_end_iteration() {
    use openai_agents_sdk::realtime::RealtimeConnectionStatus as Status;
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        Default::default(),
        None,
    )
    .await;
    recv(&session).await; // connect-time history

    model.emit(RealtimeModelEvent::ConnectionStatus {
        status: Status::Disconnected,
    });
    assert_eq!(recv(&session).await.event_type(), "raw_model_event");
    // Nothing ends the stream: the next receive is still pending.
    assert!(
        tokio::time::timeout(Duration::from_millis(100), session.recv())
            .await
            .is_err()
    );
    model.emit(RealtimeModelEvent::ConnectionStatus {
        status: Status::Connected,
    });
    match recv(&session).await {
        RealtimeSessionEvent::RawModelEvent { data, .. } => assert_eq!(
            data,
            RealtimeModelEvent::ConnectionStatus {
                status: Status::Connected
            }
        ),
        other => panic!("unexpected {}", other.event_type()),
    }
}

#[tokio::test]
async fn end_of_stream_drains_events_then_ends_iteration() {
    use openai_agents_sdk::realtime::RealtimeConnectionStatus as Status;
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        &model,
        RealtimeAgent::<()>::new("agent").into_arc(),
        Default::default(),
        None,
    )
    .await;
    model.emit(RealtimeModelEvent::ConnectionStatus {
        status: Status::Disconnected,
    });
    model.emit(RealtimeModelEvent::EndOfStream);

    let mut seen = Vec::new();
    while let Some(event) = tokio::time::timeout(WAIT, session.recv()).await.unwrap() {
        seen.push(event);
    }
    assert_eq!(
        types(&seen),
        ["history_updated", "raw_model_event", "raw_model_event"]
    );
    assert!(matches!(
        &seen[2],
        RealtimeSessionEvent::RawModelEvent {
            data: RealtimeModelEvent::EndOfStream,
            ..
        }
    ));
}
