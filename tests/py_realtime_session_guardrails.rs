//! Ports of the Python realtime session output-guardrail and tool-output-guardrail tests
//! (`test_session_guardrails.py`, `test_session_tool_output_guardrails.py`).
//!
//! Python drives `session.on_event` and reads private queues; here events travel through the
//! transport (`FakeRealtimeModel::emit`) and are observed through the public event stream and the
//! recorded outbound events. The runtime is single-threaded, so yielding repeatedly lets every
//! spawned guardrail task run to completion without sleeping.

#![cfg(feature = "realtime")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeGuardrailsSettings, RealtimeInterrupt, RealtimeModel,
    RealtimeModelAudioEvent, RealtimeModelConfig, RealtimeModelEvent, RealtimeModelEvents,
    RealtimeModelSendEvent, RealtimeModelToolCallEvent, RealtimeRunConfig, RealtimeSendCondition,
    RealtimeSession, RealtimeSessionEvent, RealtimeUserInput,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolOutputGuardrail};
use openai_agents_sdk::{AgentsError, GuardrailFunctionOutput, OutputGuardrail};
use serde_json::{Value, json};
use tokio::sync::{Notify, Semaphore};

const EVENT_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// Let every runnable task make progress. Deterministic on the single-threaded test runtime.
async fn settle() {
    settle_for(300).await;
}

async fn settle_for(rounds: usize) {
    for _ in 0..rounds {
        tokio::task::yield_now().await;
    }
}

/// How the test transport reacts after the session sends an interrupt.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AfterInterrupt {
    Nothing,
    /// The source turn ends.
    EndTurn,
    /// The source turn ends, a newer id-less turn starts and also ends.
    EndStartEnd,
}

/// A transport that records order and can misbehave in the ways the Python tests need.
struct TestModel {
    inner: Arc<FakeRealtimeModel>,
    ops: Mutex<Vec<String>>,
    /// `false` mimics a transport that does not override the atomic conditional send.
    atomic_send: bool,
    fail_retire: bool,
    fail_first_tool_output: bool,
    after_interrupt: AfterInterrupt,
    boundary_started: Option<Arc<Notify>>,
    boundary_gate: Option<Arc<Semaphore>>,
    tool_output_attempts: Mutex<Vec<String>>,
}

impl std::fmt::Debug for TestModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestModel").finish_non_exhaustive()
    }
}

impl TestModel {
    fn new() -> Self {
        TestModel {
            inner: Arc::new(FakeRealtimeModel::new()),
            ops: Mutex::new(Vec::new()),
            atomic_send: true,
            fail_retire: false,
            fail_first_tool_output: false,
            after_interrupt: AfterInterrupt::Nothing,
            boundary_started: None,
            boundary_gate: None,
            tool_output_attempts: Mutex::new(Vec::new()),
        }
    }

    fn ops(&self) -> Vec<String> {
        self.ops.lock().unwrap().clone()
    }
}

#[async_trait]
impl RealtimeModel for TestModel {
    async fn connect(
        &self,
        config: RealtimeModelConfig,
    ) -> openai_agents_sdk::Result<RealtimeModelEvents> {
        self.inner.connect(config).await
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> openai_agents_sdk::Result<()> {
        if let RealtimeModelSendEvent::ToolOutput { output, .. } = &event {
            let attempt = {
                let mut attempts = self.tool_output_attempts.lock().unwrap();
                attempts.push(output.clone());
                attempts.len()
            };
            if self.fail_first_tool_output && attempt == 1 {
                return Err(AgentsError::other("transport unavailable"));
            }
        }
        let is_interrupt = matches!(event, RealtimeModelSendEvent::Interrupt(_));
        if is_interrupt {
            self.ops.lock().unwrap().push("interrupt".to_string());
        }
        self.inner.send_event(event).await?;
        if is_interrupt {
            match self.after_interrupt {
                AfterInterrupt::Nothing => {}
                AfterInterrupt::EndTurn => {
                    self.inner
                        .emit(RealtimeModelEvent::TurnEnded { response_id: None });
                    settle_for(40).await;
                }
                AfterInterrupt::EndStartEnd => {
                    self.inner
                        .emit(RealtimeModelEvent::TurnEnded { response_id: None });
                    self.inner
                        .emit(RealtimeModelEvent::TurnStarted { response_id: None });
                    self.inner
                        .emit(RealtimeModelEvent::TurnEnded { response_id: None });
                    settle_for(40).await;
                }
            }
        }
        Ok(())
    }

    async fn send_event_if(
        &self,
        event: RealtimeModelSendEvent,
        condition: RealtimeSendCondition,
    ) -> openai_agents_sdk::Result<bool> {
        if !self.atomic_send {
            return Ok(false);
        }
        if let Some(started) = &self.boundary_started {
            started.notify_one();
        }
        if let Some(gate) = &self.boundary_gate {
            let _permit = gate.acquire().await.expect("gate open");
        }
        self.inner.send_event_if(event, condition).await
    }

    async fn retire_response_audio(&self, response_id: &str) -> openai_agents_sdk::Result<()> {
        self.ops.lock().unwrap().push("retire".to_string());
        if self.fail_retire {
            return Err(AgentsError::other(format!(
                "failed to retire {response_id}"
            )));
        }
        self.inner.retire_response_audio(response_id).await
    }

    async fn close(&self) -> openai_agents_sdk::Result<()> {
        self.inner.close().await
    }
}

fn run_config(debounce: usize, guardrails: Vec<OutputGuardrail<()>>) -> RealtimeRunConfig<()> {
    RealtimeRunConfig::<()> {
        output_guardrails: guardrails,
        guardrails_settings: RealtimeGuardrailsSettings {
            debounce_text_length: debounce,
        },
        ..Default::default()
    }
}

async fn start(
    model: Arc<dyn RealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
    config: RealtimeRunConfig<()>,
) -> RealtimeSession<()> {
    RealtimeSession::connect(model, agent, (), None, Some(config))
        .await
        .expect("session connects")
}

fn plain_agent() -> Arc<RealtimeAgent<()>> {
    RealtimeAgent::<()>::new("test_agent").into_arc()
}

fn tripping(name: &str) -> OutputGuardrail<()> {
    OutputGuardrail::new(name, |_ctx, _agent, _output| async move {
        Ok(GuardrailFunctionOutput::new(
            json!({"reason": "test trigger"}),
            true,
        ))
    })
}

/// A guardrail that trips once released; `started` is signalled when it begins evaluating.
fn delayed_tripping(
    name: &str,
    started: Arc<Notify>,
    release: Arc<Semaphore>,
    runs: Arc<AtomicUsize>,
) -> OutputGuardrail<()> {
    OutputGuardrail::new(name, move |_ctx, _agent, _output| {
        let started = started.clone();
        let release = release.clone();
        let runs = runs.clone();
        async move {
            runs.fetch_add(1, Ordering::SeqCst);
            started.notify_one();
            let _permit = release.acquire().await.expect("release open");
            Ok(GuardrailFunctionOutput::new(json!({}), true))
        }
    })
}

fn turn_started(id: Option<&str>) -> RealtimeModelEvent {
    RealtimeModelEvent::TurnStarted {
        response_id: id.map(str::to_string),
    }
}

fn turn_ended(id: Option<&str>) -> RealtimeModelEvent {
    RealtimeModelEvent::TurnEnded {
        response_id: id.map(str::to_string),
    }
}

fn transcript(item: &str, delta: &str, response: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::TranscriptDelta {
        item_id: item.to_string(),
        delta: delta.to_string(),
        response_id: response.to_string(),
    }
}

fn text_delta(item: &str, delta: &str, response: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::OutputTextDelta {
        item_id: item.to_string(),
        delta: delta.to_string(),
        response_id: response.to_string(),
    }
}

fn audio(response: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::Audio(Box::new(RealtimeModelAudioEvent {
        data: b"audio".to_vec(),
        response_id: response.to_string(),
        item_id: "item_1".to_string(),
        content_index: 0,
    }))
}

fn interrupts(model: &FakeRealtimeModel) -> Vec<RealtimeInterrupt> {
    model.interrupts()
}

fn messages(model: &FakeRealtimeModel) -> Vec<String> {
    model
        .sent_events()
        .into_iter()
        .filter_map(|event| match event {
            RealtimeModelSendEvent::UserInput {
                user_input: RealtimeUserInput::Text(text),
            } => Some(text),
            _ => None,
        })
        .collect()
}

/// Everything the session has queued so far, in order. A marker event bounds the read.
async fn drain(
    session: &RealtimeSession<()>,
    model: &FakeRealtimeModel,
) -> Vec<RealtimeSessionEvent<()>> {
    settle().await;
    assert!(model.emit(RealtimeModelEvent::Other {
        data: json!({"marker": true}),
    }));
    let mut seen = Vec::new();
    tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            let event = session.recv().await.expect("event stream open");
            let done = matches!(
                &event,
                RealtimeSessionEvent::RawModelEvent {
                    data: RealtimeModelEvent::Other { data },
                    ..
                } if data == &json!({"marker": true})
            );
            if done {
                return;
            }
            seen.push(event);
        }
    })
    .await
    .expect("marker event arrives");
    seen
}

fn tripped(
    events: &[RealtimeSessionEvent<()>],
) -> Vec<(&Vec<openai_agents_sdk::OutputGuardrailResult>, &String)> {
    events
        .iter()
        .filter_map(|event| match event {
            RealtimeSessionEvent::GuardrailTripped {
                guardrail_results,
                message,
                ..
            } => Some((guardrail_results, message)),
            _ => None,
        })
        .collect()
}

async fn wait_for_event<F>(session: &RealtimeSession<()>, predicate: F) -> RealtimeSessionEvent<()>
where
    F: Fn(&RealtimeSessionEvent<()>) -> bool,
{
    tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            let event = session.recv().await.expect("event stream open");
            if predicate(&event) {
                return event;
            }
        }
    })
    .await
    .expect("expected event arrives")
}

fn release_all(semaphore: &Semaphore) {
    semaphore.add_permits(1024);
}

// ---------------------------------------------------------------------------------------------
// test_session_guardrails.py
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn output_guardrail_failure_is_skipped_without_tripping() {
    let failing = OutputGuardrail::<()>::new("failing", |_c, _a, _o| async move {
        Err(AgentsError::other("SECRET_REALTIME_GUARDRAIL_ERROR"))
    });
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), plain_agent(), run_config(5, vec![failing])).await;

    model.emit(transcript("item_1", "model text", "response-id"));
    let events = drain(&session, &model).await;

    assert!(tripped(&events).is_empty());
    assert!(interrupts(&model).is_empty());
    assert!(messages(&model).is_empty());
}

#[tokio::test]
async fn output_text_delta_triggers_response_scoped_guardrail() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    model.emit(turn_started(None));
    model.emit(text_delta("item_1", "hello", "response_1"));
    settle().await;

    let sent = interrupts(&model);
    assert_eq!(sent.len(), 1);
    assert!(sent[0].force_response_cancel);
    assert_eq!(sent[0].response_id.as_deref(), Some("response_1"));
    assert!(sent[0].cancel_response_only);
    assert_eq!(
        messages(&model),
        vec!["guardrail triggered: triggered_guardrail"]
    );
    drop(session);
}

#[tokio::test]
async fn stale_output_text_guardrail_does_not_affect_newer_response() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let runs = Arc::new(AtomicUsize::new(0));
    let agent = RealtimeAgent::<()>::new("source")
        .with_output_guardrail(delayed_tripping(
            "delayed_guardrail",
            started.clone(),
            release.clone(),
            runs,
        ))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, run_config(1, vec![])).await;

    model.emit(turn_started(Some("response_1")));
    model.emit(text_delta("item_1", "blocked", "response_1"));
    started.notified().await;

    model.emit(turn_started(Some("response_2")));
    settle().await;
    release_all(&release);

    let events = drain(&session, &model).await;
    assert!(interrupts(&model).is_empty());
    assert!(messages(&model).is_empty());
    assert_eq!(tripped(&events).len(), 1);
}

#[tokio::test]
async fn stale_audio_guardrail_interrupts_only_source_playback() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let agent = RealtimeAgent::<()>::new("source")
        .with_output_guardrail(delayed_tripping(
            "delayed_guardrail",
            started.clone(),
            release.clone(),
            Arc::new(AtomicUsize::new(0)),
        ))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, run_config(1, vec![])).await;

    model.emit(turn_started(Some("response_1")));
    model.emit(transcript("item_1", "blocked", "response_1"));
    started.notified().await;
    model.emit(turn_ended(Some("response_1")));
    model.emit(turn_started(Some("response_2")));
    settle().await;

    assert!(model.retired_responses().is_empty());
    release_all(&release);
    settle().await;

    let sent = interrupts(&model);
    assert_eq!(sent, vec![RealtimeInterrupt::playback_only("response_1")]);
    assert!(messages(&model).is_empty());
    assert_eq!(model.retired_responses(), vec!["response_1"]);

    // Suppression for the retired response is released: its audio flows again.
    model.emit(audio("response_1"));
    let events = drain(&session, &model).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::Audio { .. }))
    );
}

#[tokio::test]
async fn response_audio_cleanup_waits_for_delayed_guardrail() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let test_model = Arc::new(TestModel::new());
    let session = start(
        test_model.clone(),
        plain_agent(),
        run_config(
            1,
            vec![delayed_tripping(
                "delayed_guardrail",
                started.clone(),
                release.clone(),
                Arc::new(AtomicUsize::new(0)),
            )],
        ),
    )
    .await;

    test_model.inner.emit(turn_started(None));
    test_model
        .inner
        .emit(transcript("item_1", "blocked", "response_1"));
    started.notified().await;
    test_model.inner.emit(turn_ended(None));
    settle().await;

    assert!(test_model.ops().is_empty());
    release_all(&release);
    settle().await;

    assert_eq!(test_model.ops(), vec!["interrupt", "retire"]);
    assert_eq!(test_model.inner.retired_responses(), vec!["response_1"]);
    drop(session);
}

#[tokio::test]
async fn response_audio_cleanup_runs_immediately_without_guardrail_tasks() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), plain_agent(), RealtimeRunConfig::default()).await;

    model.emit(turn_ended(Some("response_1")));
    settle().await;

    assert_eq!(model.retired_responses(), vec!["response_1"]);
    drop(session);
}

#[tokio::test]
async fn stale_explicit_turn_end_preserves_active_response_guardrail_state() {
    let model = Arc::new(FakeRealtimeModel::new());
    // "still active" is 12 characters; the next delta crosses the 15 character threshold only if
    // the stale turn end left the accumulated transcript alone.
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(15, vec![tripping("triggered_guardrail")]),
    )
    .await;

    model.emit(turn_started(Some("new_response")));
    model.emit(transcript("new_item", "still active", "new_response"));
    model.emit(turn_ended(Some("old_response")));
    model.emit(transcript("new_item", " more", "new_response"));
    let events = drain(&session, &model).await;

    assert_eq!(model.retired_responses(), vec!["old_response"]);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::AgentEnd { .. }))
    );
    let trips = tripped(&events);
    assert_eq!(trips.len(), 1);
    assert_eq!(trips[0].1, "still active more");
    // The still-active response is cancelled outright.
    assert_eq!(
        interrupts(&model),
        vec![RealtimeInterrupt::forced_for_response("new_response", true)]
    );
}

#[tokio::test]
async fn interrupted_response_audio_delta_is_not_forwarded() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    // Tripping the guardrail marks response_1 as interrupted.
    model.emit(turn_started(Some("response_1")));
    model.emit(transcript("item_1", "trigger", "response_1"));
    settle().await;
    model.emit(audio("response_1"));
    let events = drain(&session, &model).await;

    assert_eq!(tripped(&events).len(), 1);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::Audio { .. }))
    );
}

#[tokio::test]
async fn response_audio_cleanup_error_releases_session_suppression() {
    let mut test_model = TestModel::new();
    test_model.fail_retire = true;
    let test_model = Arc::new(test_model);
    let session = start(
        test_model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    test_model.inner.emit(turn_started(Some("response_1")));
    test_model
        .inner
        .emit(transcript("item_1", "trigger", "response_1"));
    settle().await;
    test_model.inner.emit(turn_ended(Some("response_1")));
    settle().await;
    test_model.inner.emit(audio("response_1"));
    let events = drain(&session, &test_model.inner).await;

    let errors: Vec<&Value> = events
        .iter()
        .filter_map(|event| match event {
            RealtimeSessionEvent::Error { error, .. } => Some(error),
            _ => None,
        })
        .collect();
    assert_eq!(
        errors,
        vec![&json!({"message": "Response audio cleanup failed: failed to retire response_1"})]
    );
    // The failed cleanup still released the suppression, so the audio is forwarded.
    assert!(
        events
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::Audio { .. }))
    );
}

#[tokio::test]
async fn output_text_guardrail_sends_feedback_after_source_turn_ends() {
    let mut test_model = TestModel::new();
    test_model.after_interrupt = AfterInterrupt::EndTurn;
    let test_model = Arc::new(test_model);
    let session = start(
        test_model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    test_model.inner.emit(turn_started(Some("response_1")));
    test_model
        .inner
        .emit(text_delta("item_1", "hello", "response_1"));
    settle().await;

    assert_eq!(
        messages(&test_model.inner),
        vec!["guardrail triggered: triggered_guardrail"]
    );
    drop(session);
}

#[tokio::test]
async fn output_text_guardrail_skips_feedback_for_completed_idless_newer_turn() {
    let mut test_model = TestModel::new();
    test_model.after_interrupt = AfterInterrupt::EndStartEnd;
    let test_model = Arc::new(test_model);
    let session = start(
        test_model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    test_model.inner.emit(turn_started(Some("response_1")));
    test_model
        .inner
        .emit(text_delta("item_1", "hello", "response_1"));
    settle().await;

    assert_eq!(interrupts(&test_model.inner).len(), 1);
    assert!(messages(&test_model.inner).is_empty());
    drop(session);
}

#[tokio::test]
async fn output_text_guardrail_rechecks_generation_at_feedback_send_boundary() {
    let started = Arc::new(Notify::new());
    let gate = Arc::new(Semaphore::new(0));
    let mut test_model = TestModel::new();
    test_model.boundary_started = Some(started.clone());
    test_model.boundary_gate = Some(gate.clone());
    let test_model = Arc::new(test_model);
    let session = start(
        test_model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    test_model.inner.emit(turn_started(Some("response_1")));
    test_model
        .inner
        .emit(text_delta("item_1", "hello", "response_1"));
    started.notified().await;

    test_model.inner.emit(turn_started(Some("response_2")));
    settle().await;
    release_all(&gate);
    settle().await;

    assert!(messages(&test_model.inner).is_empty());
    assert_eq!(test_model.inner.skipped_conditional_sends(), 1);
    drop(session);
}

#[tokio::test]
async fn output_text_guardrail_skips_feedback_without_atomic_model_send() {
    let mut test_model = TestModel::new();
    test_model.atomic_send = false;
    let test_model = Arc::new(test_model);
    let session = start(
        test_model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    test_model.inner.emit(turn_started(Some("response_1")));
    test_model
        .inner
        .emit(text_delta("item_1", "hello", "response_1"));
    settle().await;

    assert_eq!(interrupts(&test_model.inner).len(), 1);
    assert!(messages(&test_model.inner).is_empty());
    drop(session);
}

/// A guardrail that records the agent name it was shown and trips.
fn recording_tripping(name: &str, seen: Arc<Mutex<Vec<String>>>) -> OutputGuardrail<()> {
    OutputGuardrail::new(name, move |_ctx, agent, _output| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push(agent.name.clone());
            Ok(GuardrailFunctionOutput::new(json!({}), true))
        }
    })
}

#[tokio::test]
async fn output_text_guardrail_uses_agent_from_turn_start() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let replacement_called = Arc::new(AtomicUsize::new(0));
    let source = RealtimeAgent::<()>::new("source")
        .with_output_guardrail(recording_tripping("source_guardrail", observed.clone()))
        .into_arc();
    let counter = replacement_called.clone();
    let replacement = RealtimeAgent::<()>::new("replacement")
        .with_output_guardrail(OutputGuardrail::new(
            "replacement_guardrail",
            move |_c, _a, _o| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(GuardrailFunctionOutput::new(json!({}), false))
                }
            },
        ))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), source, run_config(5, vec![])).await;

    model.emit(turn_started(Some("response_1")));
    settle().await;
    session.update_agent(replacement).await.unwrap();
    model.emit(text_delta("item_1", "hello", "response_1"));
    settle().await;

    assert_eq!(*observed.lock().unwrap(), vec!["source"]);
    assert_eq!(replacement_called.load(Ordering::SeqCst), 0);
    assert_eq!(
        messages(&model),
        vec!["guardrail triggered: source_guardrail"]
    );
}

#[tokio::test]
async fn output_text_guardrail_retains_agent_for_matching_late_turn_start() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let source = RealtimeAgent::<()>::new("source")
        .with_output_guardrail(recording_tripping("source_guardrail", observed.clone()))
        .into_arc();
    let replacement = RealtimeAgent::<()>::new("replacement").into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), source, run_config(5, vec![])).await;

    model.emit(text_delta("item_1", "he", "response_1"));
    settle().await;
    session.update_agent(replacement).await.unwrap();
    model.emit(turn_started(Some("response_1")));
    model.emit(text_delta("item_1", "llo", "response_1"));
    settle().await;

    assert_eq!(*observed.lock().unwrap(), vec!["source"]);
    assert_eq!(
        messages(&model),
        vec!["guardrail triggered: source_guardrail"]
    );
}

#[tokio::test]
async fn matching_late_turn_start_retains_pending_guardrail_generation() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let source = RealtimeAgent::<()>::new("source")
        .with_output_guardrail(delayed_tripping(
            "source_guardrail",
            started.clone(),
            release.clone(),
            Arc::new(AtomicUsize::new(0)),
        ))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), source, run_config(2, vec![])).await;

    model.emit(text_delta("item_1", "he", "response_1"));
    started.notified().await;
    model.emit(turn_started(Some("response_1")));
    settle().await;
    release_all(&release);
    settle().await;

    let sent = interrupts(&model);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].response_id.as_deref(), Some("response_1"));
    assert_eq!(
        messages(&model),
        vec!["guardrail triggered: source_guardrail"]
    );
    drop(session);
}

#[tokio::test]
async fn output_text_guardrail_sends_feedback_if_source_ends_during_evaluation() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let agent = RealtimeAgent::<()>::new("source")
        .with_output_guardrail(delayed_tripping(
            "delayed_guardrail",
            started.clone(),
            release.clone(),
            Arc::new(AtomicUsize::new(0)),
        ))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, run_config(1, vec![])).await;

    model.emit(turn_started(Some("response_1")));
    model.emit(text_delta("item_1", "blocked", "response_1"));
    started.notified().await;

    model.emit(turn_ended(None));
    settle().await;
    release_all(&release);
    settle().await;

    assert!(interrupts(&model).is_empty());
    assert_eq!(
        messages(&model),
        vec!["guardrail triggered: delayed_guardrail"]
    );
    drop(session);
}

#[tokio::test]
async fn agent_and_run_config_guardrails_not_run_twice() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let shared = OutputGuardrail::<()>::new("shared_guardrail", move |_c, _a, _o| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(json!({}), false))
        }
    });
    let agent = RealtimeAgent::<()>::new("agent")
        .with_output_guardrail(shared.clone())
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, run_config(5, vec![shared])).await;

    model.emit(transcript("item_1", "hello", "resp_1"));
    settle().await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(session);
}

#[tokio::test]
async fn transcript_delta_multiple_thresholds_same_item() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("triggered_guardrail")]),
    )
    .await;

    model.emit(transcript("item_1", "12345", "resp_1"));
    model.emit(transcript("item_1", "67890", "resp_1"));
    settle().await;

    assert_eq!(interrupts(&model).len(), 1);
    assert_eq!(messages(&model).len(), 1);
    drop(session);
}

/// A non-tripping guardrail that records every text it was asked to check.
fn recording_safe(texts: Arc<Mutex<Vec<String>>>) -> OutputGuardrail<()> {
    OutputGuardrail::new("recording", move |_c, _a, output| {
        let texts = texts.clone();
        async move {
            texts
                .lock()
                .unwrap()
                .push(output.as_str().unwrap_or_default().to_string());
            Ok(GuardrailFunctionOutput::new(json!({}), false))
        }
    })
}

#[tokio::test]
async fn large_transcript_delta_advances_past_each_crossed_threshold() {
    let texts = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(5, vec![recording_safe(texts.clone())]),
    )
    .await;

    model.emit(transcript("item_1", "123456789012", "resp_1"));
    settle().await;
    assert_eq!(texts.lock().unwrap().len(), 1);

    model.emit(transcript("item_1", "3", "resp_1"));
    settle().await;
    assert_eq!(texts.lock().unwrap().len(), 1);
    drop(session);
}

#[tokio::test]
async fn transcript_delta_different_items_tracked_separately() {
    let texts = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(10, vec![recording_safe(texts.clone())]),
    )
    .await;

    model.emit(transcript("item_1", "12345678", "resp_1"));
    model.emit(transcript("item_2", "abcdefgh", "resp_2"));
    settle().await;
    assert!(texts.lock().unwrap().is_empty());
    assert!(interrupts(&model).is_empty());

    model.emit(transcript("item_1", "90ab", "resp_1"));
    settle().await;
    // Only item_1 crossed its threshold; item_2 is still below it.
    assert_eq!(*texts.lock().unwrap(), vec!["1234567890ab"]);
    drop(session);
}

#[tokio::test]
async fn turn_ended_clears_guardrail_state() {
    let texts = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(5, vec![recording_safe(texts.clone())]),
    )
    .await;

    model.emit(transcript("item_1", "trigger", "resp_1"));
    settle().await;
    assert_eq!(*texts.lock().unwrap(), vec!["trigger"]);

    model.emit(turn_ended(None));
    settle().await;
    // The accumulated transcript and run count were cleared: the same item starts over.
    model.emit(transcript("item_1", "fresh", "resp_2"));
    settle().await;
    assert_eq!(*texts.lock().unwrap(), vec!["trigger", "fresh"]);
    drop(session);
}

#[tokio::test]
async fn multiple_guardrails_all_triggered() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(5, vec![tripping("guardrail_1"), tripping("guardrail_2")]),
    )
    .await;

    model.emit(transcript("item_1", "trigger", "resp_1"));
    let events = drain(&session, &model).await;

    assert_eq!(interrupts(&model).len(), 1);
    let sent = messages(&model);
    assert_eq!(sent.len(), 1);
    assert!(sent[0].contains("guardrail_1") && sent[0].contains("guardrail_2"));
    let trips = tripped(&events);
    assert_eq!(trips.len(), 1);
    assert_eq!(trips[0].0.len(), 2);
}

#[tokio::test]
async fn concurrent_guardrail_tasks_interrupt_once_per_response() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let runs = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(
        model.clone(),
        plain_agent(),
        run_config(
            5,
            vec![delayed_tripping(
                "concurrent_trigger",
                started.clone(),
                release.clone(),
                runs.clone(),
            )],
        ),
    )
    .await;

    model.emit(transcript("item_1", "12345", "resp_same"));
    model.emit(transcript("item_1", "67890", "resp_same"));
    // Both guardrail runs are in flight before either is released.
    while runs.load(Ordering::SeqCst) < 2 {
        tokio::task::yield_now().await;
    }
    release_all(&release);
    settle().await;

    assert_eq!(interrupts(&model).len(), 1);
    assert_eq!(messages(&model).len(), 1);
    drop(session);
}

// ---------------------------------------------------------------------------------------------
// test_session_tool_output_guardrails.py
// ---------------------------------------------------------------------------------------------

fn lookup_call() -> RealtimeModelToolCallEvent {
    RealtimeModelToolCallEvent::new("lookup", "call_lookup", "{}")
}

fn tool_outputs(model: &FakeRealtimeModel) -> Vec<String> {
    model
        .tool_outputs()
        .into_iter()
        .map(|(_call, output)| output)
        .collect()
}

fn tool_config(async_tool_calls: bool) -> RealtimeRunConfig<()> {
    RealtimeRunConfig::<()> {
        async_tool_calls,
        ..Default::default()
    }
}

type SeenOutputs = Arc<Mutex<Vec<(String, String, String, String, String)>>>;

/// A policy that records `(label, call_id, arguments, agent_name, output)` and returns `verdict`.
fn recording_policy(
    label: &'static str,
    seen: SeenOutputs,
    verdict: ToolGuardrailFunctionOutput,
) -> ToolOutputGuardrail<()> {
    ToolOutputGuardrail::new(label, move |data| {
        let seen = seen.clone();
        let verdict = verdict.clone();
        async move {
            seen.lock().unwrap().push((
                label.to_string(),
                data.context.tool_call_id.clone(),
                data.context.tool_arguments.clone(),
                data.agent_name.clone(),
                data.output.to_model_string(),
            ));
            Ok(verdict)
        }
    })
}

async fn tool_end_output(session: &RealtimeSession<()>) -> (String, String) {
    match wait_for_event(session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await
    {
        RealtimeSessionEvent::ToolEnd { agent, output, .. } => {
            (agent.name.clone(), output.to_model_string())
        }
        _ => unreachable!(),
    }
}

async fn tool_output_policies_check_original_result_in_order(reject: bool) {
    let original = json!({"value": "synthetic tool result"});
    let seen: SeenOutputs = Arc::default();
    let invocations = Arc::new(AtomicUsize::new(0));
    let counter = invocations.clone();
    let value = original.clone();
    let second_verdict = if reject {
        ToolGuardrailFunctionOutput::reject_content("replacement output")
    } else {
        ToolGuardrailFunctionOutput::allow()
    };
    let tool = FunctionTool::<()>::new_no_args("lookup", "lookup", move |_ctx| {
        let counter = counter.clone();
        let value = value.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(value)
        }
    })
    .with_output_guardrail(recording_policy(
        "first",
        seen.clone(),
        ToolGuardrailFunctionOutput::allow(),
    ))
    .with_output_guardrail(recording_policy("second", seen.clone(), second_verdict))
    .with_output_guardrail(recording_policy(
        "third",
        seen.clone(),
        ToolGuardrailFunctionOutput::allow(),
    ));
    let agent = RealtimeAgent::<()>::new("source")
        .with_tool(tool)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, tool_config(false)).await;

    model.emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    let (end_agent, end_output) = tool_end_output(&session).await;

    let original_text = original.to_string();
    let expected = if reject {
        "replacement output".to_string()
    } else {
        original_text.clone()
    };
    assert_eq!(end_agent, "source");
    assert_eq!(end_output, expected);
    assert_eq!(tool_outputs(&model), vec![expected]);
    let labels: Vec<String> = seen
        .lock()
        .unwrap()
        .iter()
        .map(|row| row.0.clone())
        .collect();
    assert_eq!(
        labels,
        if reject {
            vec!["first", "second"]
        } else {
            vec!["first", "second", "third"]
        }
    );
    for (_label, call_id, arguments, agent_name, output) in seen.lock().unwrap().iter() {
        // Every policy sees the original, unreplaced result and the dispatch context.
        assert_eq!(output, &original_text);
        assert_eq!(agent_name, "source");
        assert_eq!(call_id, "call_lookup");
        assert_eq!(arguments, "{}");
    }

    // The redelivered call neither reruns the tool nor sends a second output.
    model.emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    settle().await;
    assert_eq!(invocations.load(Ordering::SeqCst), 1);
    assert_eq!(tool_outputs(&model).len(), 1);
}

#[tokio::test]
async fn tool_output_policies_check_original_result_in_order_allow() {
    tool_output_policies_check_original_result_in_order(false).await;
}

#[tokio::test]
async fn tool_output_policies_check_original_result_in_order_reject() {
    tool_output_policies_check_original_result_in_order(true).await;
}

async fn output_policy_failure_never_publishes_or_reexecutes(async_tools: bool, raise: bool) {
    let calls = Arc::new(AtomicUsize::new(0));
    let checks = Arc::new(AtomicUsize::new(0));
    let call_counter = calls.clone();
    let check_counter = checks.clone();
    let policy = ToolOutputGuardrail::<()>::new("policy", move |data| {
        let check_counter = check_counter.clone();
        async move {
            check_counter.fetch_add(1, Ordering::SeqCst);
            assert_eq!(data.output.to_model_string(), "unchecked result");
            if raise {
                Ok(ToolGuardrailFunctionOutput::raise_exception())
            } else {
                Err(AgentsError::other("policy unavailable"))
            }
        }
    });
    let tool = FunctionTool::<()>::new_no_args("lookup", "lookup", move |_ctx| {
        let call_counter = call_counter.clone();
        async move {
            call_counter.fetch_add(1, Ordering::SeqCst);
            Ok("unchecked result".to_string())
        }
    })
    .with_output_guardrail(policy);
    let agent = RealtimeAgent::<()>::new("source")
        .with_tool(tool)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, tool_config(async_tools)).await;

    model.emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    // The failure ends the session: events drain, then the stream closes.
    let mut events = Vec::new();
    tokio::time::timeout(EVENT_TIMEOUT, async {
        while let Some(event) = session.recv().await {
            events.push(event);
        }
    })
    .await
    .expect("session ends after the policy failure");
    let error = session.take_error().expect("the policy failure is stored");
    if raise {
        assert!(matches!(
            error,
            AgentsError::ToolOutputGuardrailTripwireTriggered { .. }
        ));
    } else {
        assert!(error.to_string().contains("policy unavailable"));
    }
    // A redelivery after the failure must not run anything again.
    model.emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    settle().await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(checks.load(Ordering::SeqCst), 1);
    assert!(tool_outputs(&model).is_empty());
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::ToolEnd { .. }))
    );
}

#[tokio::test]
async fn output_policy_failure_sync_tripwire() {
    output_policy_failure_never_publishes_or_reexecutes(false, true).await;
}

#[tokio::test]
async fn output_policy_failure_sync_error() {
    output_policy_failure_never_publishes_or_reexecutes(false, false).await;
}

#[tokio::test]
async fn output_policy_failure_async_tripwire() {
    output_policy_failure_never_publishes_or_reexecutes(true, true).await;
}

#[tokio::test]
async fn output_policy_failure_async_error() {
    output_policy_failure_never_publishes_or_reexecutes(true, false).await;
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn mcp_output_policy_is_enforced_after_conversion() {
    use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool, MCPToolPolicy};

    #[derive(Debug)]
    struct LookupServer {
        config: MCPServerConfig,
        calls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl MCPServer for LookupServer {
        fn name(&self) -> &str {
            "fake"
        }
        fn config(&self) -> &MCPServerConfig {
            &self.config
        }
        async fn connect(&self) -> openai_agents_sdk::Result<()> {
            Ok(())
        }
        async fn cleanup(&self) -> openai_agents_sdk::Result<()> {
            Ok(())
        }
        async fn list_tools(&self) -> openai_agents_sdk::Result<Vec<MCPTool>> {
            Ok(vec![MCPTool {
                meta: None,
                name: "lookup".to_string(),
                title: None,
                description: Some("lookup".to_string()),
                input_schema: json!({"type": "object", "properties": {}}),
                annotations: None,
            }])
        }
        async fn call_tool(
            &self,
            tool_name: &str,
            _arguments: Option<serde_json::Map<String, Value>>,
        ) -> openai_agents_sdk::Result<Value> {
            self.calls.lock().unwrap().push(tool_name.to_string());
            Ok(json!({"content": [{"type": "text", "text": "mcp raw result"}]}))
        }
        async fn list_prompts(&self) -> openai_agents_sdk::Result<Vec<Value>> {
            Ok(Vec::new())
        }
        async fn get_prompt(
            &self,
            _name: &str,
            _arguments: Option<std::collections::BTreeMap<String, String>>,
        ) -> openai_agents_sdk::Result<Value> {
            Ok(Value::Null)
        }
    }

    let seen: SeenOutputs = Arc::default();
    let policy = recording_policy(
        "policy",
        seen.clone(),
        ToolGuardrailFunctionOutput::reject_content("MCP replacement"),
    );
    let server = Arc::new(LookupServer {
        config: MCPServerConfig::default()
            .with_tool_policy(MCPToolPolicy::<()>::new().with_tool_output_guardrails(vec![policy])),
        calls: Mutex::new(Vec::new()),
    });
    let agent = RealtimeAgent::<()>::new("source")
        .with_mcp_server(server.clone())
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, tool_config(false)).await;

    model.emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    let (_agent, output) = tool_end_output(&session).await;

    assert_eq!(output, "MCP replacement");
    assert_eq!(tool_outputs(&model), vec!["MCP replacement"]);
    assert_eq!(*server.calls.lock().unwrap(), vec!["lookup"]);
    // The policy ran on the converted MCP result, not on the raw protocol payload.
    let rows = seen.lock().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].4, "mcp raw result");
}

#[tokio::test]
async fn approved_async_tool_output_policy_keeps_dispatch_agent() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let checks: Arc<Mutex<Vec<String>>> = Arc::default();
    let (started_in, release_in, checks_in) = (started.clone(), release.clone(), checks.clone());
    let policy = ToolOutputGuardrail::<()>::new("policy", move |data| {
        let started = started_in.clone();
        let release = release_in.clone();
        let checks = checks_in.clone();
        async move {
            checks.lock().unwrap().push(data.agent_name.clone());
            started.notify_one();
            let _permit = release.acquire().await.expect("release open");
            Ok(ToolGuardrailFunctionOutput::reject_content(
                "approved replacement",
            ))
        }
    });
    let tool = FunctionTool::<()>::new_no_args("lookup", "lookup", |_ctx| async move {
        Ok("unchecked result".to_string())
    })
    .with_needs_approval(true)
    .with_output_guardrail(policy);
    let agent = RealtimeAgent::<()>::new("source")
        .with_tool(tool)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, tool_config(true)).await;

    model.emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    wait_for_event(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    assert!(checks.lock().unwrap().is_empty());

    // Approval runs the tool on its own task so the session can be updated meanwhile.
    let approver = session.clone();
    let approval =
        tokio::spawn(async move { approver.approve_tool_call("call_lookup", false).await });
    tokio::time::timeout(EVENT_TIMEOUT, started.notified())
        .await
        .expect("policy starts");
    session
        .update_agent(RealtimeAgent::<()>::new("replacement").into_arc())
        .await
        .unwrap();
    assert!(tool_outputs(&model).is_empty());
    release_all(&release);

    let (end_agent, end_output) = tool_end_output(&session).await;
    approval.await.unwrap().unwrap();
    assert_eq!(end_agent, "source");
    assert_eq!(end_output, "approved replacement");
    assert_eq!(*checks.lock().unwrap(), vec!["source"]);
    assert_eq!(tool_outputs(&model), vec!["approved replacement"]);
}

#[tokio::test]
async fn transport_retry_reuses_only_checked_output() {
    let calls = Arc::new(AtomicUsize::new(0));
    let checks = Arc::new(AtomicUsize::new(0));
    let (call_counter, check_counter) = (calls.clone(), checks.clone());
    let policy = ToolOutputGuardrail::<()>::new("policy", move |_data| {
        let check_counter = check_counter.clone();
        async move {
            check_counter.fetch_add(1, Ordering::SeqCst);
            Ok(ToolGuardrailFunctionOutput::reject_content(
                "retry replacement",
            ))
        }
    });
    let tool = FunctionTool::<()>::new_no_args("lookup", "lookup", move |_ctx| {
        let call_counter = call_counter.clone();
        async move {
            call_counter.fetch_add(1, Ordering::SeqCst);
            Ok("unchecked result".to_string())
        }
    })
    .with_output_guardrail(policy);
    let agent = RealtimeAgent::<()>::new("source")
        .with_tool(tool)
        .into_arc();
    let mut test_model = TestModel::new();
    test_model.fail_first_tool_output = true;
    let test_model = Arc::new(test_model);
    let session = start(test_model.clone(), agent, tool_config(true)).await;

    test_model
        .inner
        .emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    wait_for_event(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { .. })
    })
    .await;
    assert!(tool_outputs(&test_model.inner).is_empty());

    test_model
        .inner
        .emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    let (_agent, output) = tool_end_output(&session).await;

    assert_eq!(output, "retry replacement");
    assert_eq!(tool_outputs(&test_model.inner), vec!["retry replacement"]);
    assert_eq!(
        *test_model.tool_output_attempts.lock().unwrap(),
        vec!["retry replacement", "retry replacement"]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(checks.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn output_policy_shutdown_prevents_late_output() {
    struct CancelFlag(Arc<AtomicUsize>);
    impl Drop for CancelFlag {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let (started_in, cancelled_in) = (started.clone(), cancelled.clone());
    let policy = ToolOutputGuardrail::<()>::new("policy", move |_data| {
        let started = started_in.clone();
        let cancelled = cancelled_in.clone();
        async move {
            let _flag = CancelFlag(cancelled);
            started.notify_one();
            std::future::pending::<()>().await;
            Ok(ToolGuardrailFunctionOutput::allow())
        }
    });
    let call_counter = calls.clone();
    let tool = FunctionTool::<()>::new_no_args("lookup", "lookup", move |_ctx| {
        let call_counter = call_counter.clone();
        async move {
            call_counter.fetch_add(1, Ordering::SeqCst);
            Ok("unchecked result".to_string())
        }
    })
    .with_output_guardrail(policy);
    let agent = RealtimeAgent::<()>::new("source")
        .with_tool(tool)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone(), agent, tool_config(true)).await;

    model.emit(RealtimeModelEvent::FunctionCall(lookup_call()));
    tokio::time::timeout(EVENT_TIMEOUT, started.notified())
        .await
        .expect("policy starts");
    session.close().await.unwrap();
    settle().await;

    assert_eq!(cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(tool_outputs(&model).is_empty());
    // The closed session delivers nothing further.
    let mut late = Vec::new();
    while let Some(event) = session.recv().await {
        late.push(event);
    }
    assert!(
        !late
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::ToolEnd { .. }))
    );
}
