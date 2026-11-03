//! Ports of the Python SDK's `tests/realtime/test_testing.py` and `test_runner.py`.
//!
//! The Rust scripted transport hands the session a single event channel instead of fanning events
//! out to listener callbacks, so the Python tests that exercise listener delivery order, delivery
//! workers or task cancellation have no Rust counterpart; the ledger records each of them.

#![cfg(feature = "realtime")]

use std::sync::Arc;

use openai_agents_sdk::realtime::{
    RealtimeAgent, RealtimeConnectCall, RealtimeFunctionToolSpec, RealtimeModel,
    RealtimeModelConfig, RealtimeModelEvent, RealtimeModelSendEvent, RealtimePlaybackTracker,
    RealtimeRunConfig, RealtimeRunner, RealtimeSendMatcher, RealtimeSessionModelSettings,
    RealtimeStep, RealtimeUserInput, RealtimeVoice, ScriptedRealtimeModel,
};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

fn delta(item: &str, text: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::OutputTextDelta {
        item_id: item.to_string(),
        delta: text.to_string(),
        response_id: format!("response_{item}"),
    }
}

fn user_input(text: &str) -> RealtimeModelSendEvent {
    RealtimeModelSendEvent::UserInput {
        user_input: RealtimeUserInput::Text(text.to_string()),
    }
}

fn drain(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<RealtimeModelEvent>,
) -> Vec<RealtimeModelEvent> {
    let mut out = Vec::new();
    while let Ok(event) = rx.try_recv() {
        out.push(event);
    }
    out
}

fn config(call_id: &str) -> RealtimeModelConfig {
    RealtimeModelConfig::new().with_call_id(call_id)
}

fn modalities(values: &[&str]) -> RealtimeSessionModelSettings {
    RealtimeSessionModelSettings {
        modalities: Some(values.iter().map(|v| v.to_string()).collect()),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_testing.py
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn scripted_realtime_model_records_sends_and_emits_events() {
    let emitted = delta("item_1", "hello");
    let model = ScriptedRealtimeModel::new([
        RealtimeStep::new(user_input("hi")).emitting([emitted.clone()])
    ]);
    let mut rx = model.connect(RealtimeModelConfig::new()).await.unwrap();
    model.send_event(user_input("hi")).await.unwrap();

    assert_eq!(drain(&mut rx), vec![emitted]);
    assert_eq!(model.sent_events(), vec![user_input("hi")]);
    model.assert_complete().unwrap();
}

#[tokio::test]
async fn scripted_realtime_model_records_sanitized_connect_snapshot() {
    let settings = modalities(&["text"]);
    let options = RealtimeModelConfig::new()
        .with_api_key("secret")
        .with_header("Authorization", "Bearer secret")
        .with_url("wss://user:password@example.test:8443/realtime?token=secret#fragment")
        .with_initial_model_settings(settings.clone())
        .with_call_id("call_1");
    let model = ScriptedRealtimeModel::new([]);
    model.connect(options).await.unwrap();

    assert_eq!(
        model.connect_calls(),
        vec![RealtimeConnectCall {
            api_key_provided: true,
            headers_provided: true,
            url: Some("wss://example.test:8443/realtime".to_string()),
            initial_model_settings: Some(settings),
            call_id: Some("call_1".to_string()),
            playback_tracker_provided: false,
        }]
    );
    let rendered = format!("{:?}", model.connect_calls());
    assert!(!rendered.contains("password"));
    assert!(!rendered.contains("secret"));
}

#[tokio::test]
async fn scripted_realtime_model_rejects_duplicate_connection_before_side_effects() {
    let connected = delta("item_1", "connected");
    let model = ScriptedRealtimeModel::new([]).with_connect_events([connected.clone()]);
    let mut rx = model.connect(config("first")).await.unwrap();

    let error = model.connect(config("second")).await.unwrap_err();
    assert!(error.to_string().contains("Already connected"), "{error}");

    let calls = model.connect_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id.as_deref(), Some("first"));
    assert!(!calls[0].api_key_provided);
    assert!(!calls[0].headers_provided);
    assert_eq!(drain(&mut rx), vec![connected]);
}

#[tokio::test]
async fn scripted_realtime_model_exposes_detached_read_only_histories() {
    let tracker = Arc::new(RealtimePlaybackTracker::new());
    let mut settings = modalities(&["text"]);
    settings.tools = Some(vec![RealtimeFunctionToolSpec::new(
        "lookup",
        "tool result",
        json!({"type": "object", "properties": {}}),
    )]);
    let event = user_input("hello");
    let update = RealtimeModelSendEvent::session_update(settings.clone());
    let model = ScriptedRealtimeModel::new([
        RealtimeStep::new(event.clone()),
        RealtimeStep::new("session_update"),
    ]);
    model
        .connect(
            RealtimeModelConfig::new()
                .with_initial_model_settings(settings.clone())
                .with_playback_tracker(tracker),
        )
        .await
        .unwrap();
    model.send_event(event).await.unwrap();
    model.send_event(update).await.unwrap();

    // Every accessor returns an owned copy, so editing it cannot change the recorded history.
    let mut connect_history = model.connect_calls();
    assert!(connect_history[0].playback_tracker_provided);
    connect_history[0]
        .initial_model_settings
        .as_mut()
        .unwrap()
        .modalities
        .as_mut()
        .unwrap()
        .push("audio".to_string());
    connect_history[0]
        .initial_model_settings
        .as_mut()
        .unwrap()
        .tools
        .as_mut()
        .unwrap()
        .clear();
    let mut send_history = model.sent_events();
    send_history[0] = user_input("changed through accessor");
    send_history.clear();

    let retained = model.connect_calls()[0]
        .initial_model_settings
        .clone()
        .unwrap();
    assert_eq!(retained.modalities, Some(vec!["text".to_string()]));
    assert_eq!(retained.tools, settings.tools);
    let sent = model.sent_events();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0], user_input("hello"));
    let retained_update = sent[1].as_session_update().unwrap();
    assert_eq!(retained_update.modalities, Some(vec!["text".to_string()]));
    assert_eq!(retained_update.tools, settings.tools);
}

#[tokio::test]
async fn realtime_step_freezes_emit_and_rejects_emit_with_error() {
    let event = delta("item_1", "hello");
    let mut source = vec![event.clone()];
    let step = RealtimeStep::new("interrupt").emitting(source.clone());
    source.clear();
    assert_eq!(step.emit, vec![event.clone()]);

    // Rust validates the step when the transport connects; Python raises at construction.
    let model = ScriptedRealtimeModel::new([RealtimeStep::new("interrupt")
        .emitting([event])
        .failing("failed")]);
    let error = model.connect(RealtimeModelConfig::new()).await.unwrap_err();
    assert!(
        error.to_string().contains("both emit events and an error"),
        "{error}"
    );
}

#[tokio::test]
async fn scripted_realtime_model_rejects_connect_events_with_error_before_steps() {
    let model = ScriptedRealtimeModel::new([RealtimeStep::new("interrupt")])
        .with_connect_events([delta("item_1", "hello")])
        .with_connect_error("failed");
    let error = model.connect(RealtimeModelConfig::new()).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("both connect events and a connect error"),
        "{error}"
    );
    // The invalid configuration is refused before the connection is recorded.
    assert!(model.connect_calls().is_empty());
    assert!(!model.is_connected());
}

#[tokio::test]
async fn scripted_realtime_model_preserves_error_only_connection() {
    let model = ScriptedRealtimeModel::new([]).with_connect_error("failed");
    let error = model.connect(config("call_1")).await.unwrap_err();
    assert!(error.to_string().contains("failed"), "{error}");

    let calls = model.connect_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id.as_deref(), Some("call_1"));
    assert!(!calls[0].api_key_provided);
    assert!(!calls[0].headers_provided);
}

#[tokio::test]
async fn scripted_realtime_model_preserves_matcher_and_error_identity() {
    // A predicate matcher paired with a failing step surfaces the configured failure.
    let matcher = RealtimeSendMatcher::predicate(|event| event.as_interrupt().is_some());
    let model = ScriptedRealtimeModel::new([RealtimeStep::new(matcher).failing("failed")]);
    model.connect(RealtimeModelConfig::new()).await.unwrap();
    let error = model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("failed"), "{error}");

    // A rejecting matcher reports an unexpected send that names the predicate it expected.
    let mismatch =
        ScriptedRealtimeModel::new([RealtimeStep::new(RealtimeSendMatcher::predicate(|_| false))]);
    mismatch.connect(RealtimeModelConfig::new()).await.unwrap();
    let error = mismatch
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("predicate"), "{error}");
    assert!(error.to_string().contains("interrupt"), "{error}");
}

#[tokio::test]
async fn scripted_realtime_model_isolates_rejected_matcher_mutations() {
    // Matchers receive `&RealtimeModelSendEvent`, so they cannot mutate what the model records;
    // the Python variant whose matcher raises has no Rust counterpart.
    let event = user_input("before");
    let model =
        ScriptedRealtimeModel::new([RealtimeStep::new(RealtimeSendMatcher::predicate(|_| false))]);
    model.connect(RealtimeModelConfig::new()).await.unwrap();

    let error = model.send_event(event.clone()).await.unwrap_err();
    assert!(error.to_string().contains("expected"), "{error}");
    assert_eq!(event, user_input("before"));
    assert_eq!(model.remaining_steps(), 1);
    assert!(model.sent_events().is_empty());
}

#[tokio::test]
async fn scripted_realtime_model_preserves_emitted_exception_identity() {
    let error_event = RealtimeModelEvent::Error {
        error: json!({"message": "failed", "context": "connect"}),
    };
    let model = ScriptedRealtimeModel::new([]).with_connect_events([error_event.clone()]);
    let mut rx = model.connect(RealtimeModelConfig::new()).await.unwrap();
    assert_eq!(drain(&mut rx), vec![error_event]);
}

#[tokio::test]
async fn scripted_realtime_model_serializes_connect_and_reentrant_send_events() {
    let connect_event = delta("item_1", "connect");
    let reply_event = delta("item_2", "reply");
    let model = ScriptedRealtimeModel::new([
        RealtimeStep::new(user_input("hello")).emitting([reply_event.clone()])
    ])
    .with_connect_events([connect_event.clone()]);
    let mut rx = model.connect(RealtimeModelConfig::new()).await.unwrap();

    // Python sends from inside the connect-event listener; the channel transport lets the
    // consumer react to the connect event by sending, and the reply lands after it.
    assert_eq!(rx.recv().await, Some(connect_event.clone()));
    model.send_event(user_input("hello")).await.unwrap();

    assert_eq!(drain(&mut rx), vec![reply_event]);
    model.assert_complete().unwrap();
}

#[tokio::test]
async fn scripted_realtime_model_delivers_concurrent_sends_in_commit_order() {
    let first = user_input("first");
    let second = user_input("second");
    let first_event = delta("item_1", "first");
    let second_event = delta("item_2", "second");
    let model = ScriptedRealtimeModel::new([
        RealtimeStep::new(first.clone()).emitting([first_event.clone()]),
        RealtimeStep::new(second.clone()).emitting([second_event.clone()]),
    ]);
    let mut rx = model.connect(RealtimeModelConfig::new()).await.unwrap();

    let (a, b) = tokio::join!(
        model.send_event(first.clone()),
        model.send_event(second.clone())
    );
    a.unwrap();
    b.unwrap();

    assert_eq!(drain(&mut rx), vec![first_event, second_event]);
    assert_eq!(model.sent_events(), vec![first, second]);
    model.assert_complete().unwrap();
}

#[tokio::test]
async fn scripted_realtime_model_conditionally_commits_under_send_lock() {
    let model = ScriptedRealtimeModel::new([RealtimeStep::new("interrupt")]);
    model.connect(RealtimeModelConfig::new()).await.unwrap();

    let skipped = model
        .send_event_if(RealtimeModelSendEvent::interrupt(), Box::new(|| false))
        .await
        .unwrap();
    let sent = model
        .send_event_if(RealtimeModelSendEvent::interrupt(), Box::new(|| true))
        .await
        .unwrap();

    assert!(!skipped);
    assert!(sent);
    assert_eq!(
        model.sent_events(),
        vec![RealtimeModelSendEvent::interrupt()]
    );
    model.assert_complete().unwrap();
}

#[tokio::test]
async fn scripted_realtime_model_rejects_unexpected_send() {
    let expected = user_input("expected secret");
    let actual = user_input("actual secret");
    let model = ScriptedRealtimeModel::new([RealtimeStep::new(expected.clone())]);
    model.connect(RealtimeModelConfig::new()).await.unwrap();

    let error = model.send_event(actual).await.unwrap_err();
    let message = error.to_string();
    assert!(message.contains("expected"), "{message}");
    assert!(!message.contains("actual secret"), "{message}");
    assert!(!message.contains("expected secret"), "{message}");
    assert_eq!(model.remaining_steps(), 1);
    assert!(model.sent_events().is_empty());
    model.send_event(expected).await.unwrap();
    model.assert_complete().unwrap();
}

#[tokio::test]
async fn scripted_realtime_model_reports_exhausted_send_attributes() {
    let model = ScriptedRealtimeModel::new([]);
    model.connect(RealtimeModelConfig::new()).await.unwrap();

    let error = model
        .send_event(user_input("exhausted secret"))
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("no scripted steps"), "{message}");
    assert!(!message.contains("exhausted secret"), "{message}");
    assert!(model.sent_events().is_empty());
}

#[tokio::test]
async fn scripted_realtime_model_conditional_mismatch_preserves_step() {
    let model = ScriptedRealtimeModel::new([RealtimeStep::new("interrupt")]);
    model.connect(RealtimeModelConfig::new()).await.unwrap();

    let error = model
        .send_event_if(user_input("wrong"), Box::new(|| true))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("expected"), "{error}");
    assert_eq!(model.remaining_steps(), 1);
    assert!(model.sent_events().is_empty());
}

#[tokio::test]
async fn scripted_realtime_model_can_record_unscripted_sends_explicitly() {
    let model = ScriptedRealtimeModel::new([]).lenient();
    model.connect(RealtimeModelConfig::new()).await.unwrap();
    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();
    assert_eq!(
        model.sent_events(),
        vec![RealtimeModelSendEvent::interrupt()]
    );
}

async fn non_strict_mismatch_preserves_pending_step(conditional: bool) {
    let expected = RealtimeModelSendEvent::interrupt();
    let unrelated = user_input("unrelated");
    let model = ScriptedRealtimeModel::new([RealtimeStep::new(expected.clone())]).lenient();
    model.connect(RealtimeModelConfig::new()).await.unwrap();

    if conditional {
        assert!(
            model
                .send_event_if(unrelated.clone(), Box::new(|| true))
                .await
                .unwrap()
        );
    } else {
        model.send_event(unrelated.clone()).await.unwrap();
    }
    assert_eq!(model.sent_events(), vec![unrelated.clone()]);
    assert_eq!(model.remaining_steps(), 1);

    model.send_event(expected.clone()).await.unwrap();
    assert_eq!(model.sent_events(), vec![unrelated, expected]);
    model.assert_complete().unwrap();
}

#[tokio::test]
async fn scripted_realtime_model_non_strict_mismatch_preserves_pending_step() {
    non_strict_mismatch_preserves_pending_step(false).await;
    non_strict_mismatch_preserves_pending_step(true).await;
}

#[tokio::test]
async fn scripted_realtime_model_closes_idempotently() {
    let model = ScriptedRealtimeModel::new([]);
    model.connect(RealtimeModelConfig::new()).await.unwrap();

    model.close().await.unwrap();
    model.close().await.unwrap();

    assert!(model.is_closed());
    assert!(!model.is_connected());
    assert_eq!(model.close_calls(), 2);

    // A closed transport refuses further sends and drops late emissions.
    let error = model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("disconnected"), "{error}");
    assert!(!model.emit(delta("item_1", "late")));
}

#[test]
fn scripted_realtime_model_reports_unconsumed_steps() {
    let model = ScriptedRealtimeModel::new([RealtimeStep::new("interrupt")]);
    let error = model.assert_complete().unwrap_err();
    assert!(
        error.to_string().contains("1 scripted Realtime step"),
        "{error}"
    );
    assert_eq!(model.remaining_steps(), 1);
}

// ---------------------------------------------------------------------------------------------
// tests/realtime/test_runner.py
//
// Python patches `RealtimeSession` and inspects its constructor arguments. The Rust runner
// connects before returning, so these tests observe the same plumbing through the transport's
// connect call instead.
// ---------------------------------------------------------------------------------------------

fn lenient_model() -> Arc<ScriptedRealtimeModel> {
    Arc::new(ScriptedRealtimeModel::new([]).lenient())
}

fn runner_agent() -> Arc<RealtimeAgent<()>> {
    RealtimeAgent::<()>::new("test_agent")
        .with_instructions("Test instructions")
        .into_arc()
}

#[tokio::test]
async fn run_preserves_custom_model() {
    let model = lenient_model();
    let session = RealtimeRunner::new(runner_agent())
        .with_model(model.clone())
        .run(())
        .await
        .unwrap();
    let expected: Arc<dyn RealtimeModel> = model;
    assert!(Arc::ptr_eq(session.model(), &expected));
}

#[tokio::test]
async fn run_creates_session_with_no_settings() {
    let model = lenient_model();
    let session = RealtimeRunner::new(runner_agent())
        .with_model(model.clone())
        .run(())
        .await
        .unwrap();

    let calls = model.connect_calls();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].api_key_provided);
    assert!(!calls[0].headers_provided);
    assert_eq!(calls[0].call_id, None);
    assert_eq!(calls[0].url, None);
    // With no settings supplied, only the agent contributes: its instructions and no tools.
    let settings = calls[0].initial_model_settings.as_ref().unwrap();
    assert_eq!(settings.instructions.as_deref(), Some("Test instructions"));
    assert_eq!(settings.voice, None);
    assert_eq!(settings.tools, Some(Vec::new()));
    assert_eq!(session.current_agent_name(), "test_agent");
}

#[tokio::test]
async fn run_creates_session_with_settings_only_in_init() {
    let model = lenient_model();
    let config = RealtimeRunConfig::<()> {
        model_settings: Some(
            RealtimeSessionModelSettings::new()
                .with_model_name("gpt-4o-realtime")
                .with_voice(RealtimeVoice::Name("nova".to_string())),
        ),
        ..Default::default()
    };
    RealtimeRunner::new(runner_agent())
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    let calls = model.connect_calls();
    let settings = calls[0].initial_model_settings.as_ref().unwrap();
    assert_eq!(settings.model_name.as_deref(), Some("gpt-4o-realtime"));
    assert_eq!(
        settings.voice,
        Some(RealtimeVoice::Name("nova".to_string()))
    );
}

#[tokio::test]
async fn run_creates_session_with_settings_in_both_init_and_run_overrides() {
    let model = lenient_model();
    let config = RealtimeRunConfig::<()> {
        model_settings: Some(
            RealtimeSessionModelSettings::new()
                .with_model_name("gpt-4o-realtime")
                .with_voice(RealtimeVoice::Name("nova".to_string())),
        ),
        ..Default::default()
    };
    let run_settings =
        RealtimeSessionModelSettings::new().with_voice(RealtimeVoice::Name("alloy".to_string()));
    RealtimeRunner::new(runner_agent())
        .with_model(model.clone())
        .with_config(config)
        .with_model_config(RealtimeModelConfig::new().with_initial_model_settings(run_settings))
        .run(())
        .await
        .unwrap();

    let calls = model.connect_calls();
    let settings = calls[0].initial_model_settings.as_ref().unwrap();
    // The run-level voice wins; the init-level model name is still applied.
    assert_eq!(
        settings.voice,
        Some(RealtimeVoice::Name("alloy".to_string()))
    );
    assert_eq!(settings.model_name.as_deref(), Some("gpt-4o-realtime"));
}

#[tokio::test]
async fn run_creates_session_with_settings_only_in_run() {
    let model = lenient_model();
    let run_settings = RealtimeSessionModelSettings::new()
        .with_model_name("gpt-4o-realtime-preview")
        .with_voice(RealtimeVoice::Name("shimmer".to_string()))
        .with_output_modalities(["text", "audio"]);
    RealtimeRunner::new(runner_agent())
        .with_model(model.clone())
        .with_model_config(RealtimeModelConfig::new().with_initial_model_settings(run_settings))
        .run(())
        .await
        .unwrap();

    let calls = model.connect_calls();
    let settings = calls[0].initial_model_settings.as_ref().unwrap();
    assert_eq!(
        settings.model_name.as_deref(),
        Some("gpt-4o-realtime-preview")
    );
    assert_eq!(
        settings.voice,
        Some(RealtimeVoice::Name("shimmer".to_string()))
    );
    assert_eq!(
        settings.output_modalities,
        Some(vec!["text".to_string(), "audio".to_string()])
    );
}

#[tokio::test]
async fn run_with_context_parameter() {
    let model = lenient_model();
    let agent = RealtimeAgent::<String>::new("ctx_agent").into_arc();
    let session = RealtimeRunner::new(agent)
        .with_model(model)
        .run("user_id=test123".to_string())
        .await
        .unwrap();
    assert_eq!(session.context().context(), "user_id=test123");
}

#[tokio::test]
async fn run_with_agent_without_instructions_or_tools_does_not_crash() {
    let model = lenient_model();
    let agent = RealtimeAgent::<()>::new("bare").into_arc();
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .run(())
        .await
        .unwrap();
    assert!(!session.is_closed());
    assert_eq!(model.connect_calls().len(), 1);
}

#[tokio::test]
async fn tool_and_handoffs_are_correct() {
    #[derive(schemars::JsonSchema, serde::Deserialize)]
    struct NoArgs {}

    let model = lenient_model();
    let agent_1 = RealtimeAgent::<()>::new("one")
        .with_instructions("instr_one")
        .into_arc();
    let agent_2 = RealtimeAgent::<()>::new("two")
        .with_instructions("instr_two")
        .with_tool(FunctionTool::new(
            "tool_one",
            "tool one",
            |_ctx, _args: NoArgs| async move { Ok("result_one".to_string()) },
        ))
        .with_handoff(agent_1)
        .into_arc();

    let session = RealtimeRunner::new(agent_2)
        .with_model(model.clone())
        .run(())
        .await
        .unwrap();
    session.close().await.unwrap();

    let calls = model.connect_calls();
    let settings = calls
        .last()
        .unwrap()
        .initial_model_settings
        .as_ref()
        .unwrap();
    assert_eq!(settings.instructions.as_deref(), Some("instr_two"));
    let specs = settings.tools.as_ref().unwrap();
    // One function tool plus one handoff, which the model sees as a tool of its own.
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].name, "tool_one");
    assert_eq!(specs[1].name, "transfer_to_one");
}
