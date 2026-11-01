//! End-to-end realtime session behavior against a fake transport, plus the OpenAI WebSocket
//! transport against a local mock server.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, OpenAIRealtimeSIPModel, OpenAIRealtimeWebSocketModel, RealtimeAgent,
    RealtimeAudioConfig, RealtimeAudioFormat, RealtimeAudioInputConfig, RealtimeAudioOutputConfig,
    RealtimeAudioStorage, RealtimeContent, RealtimeInputAudioTranscriptionConfig,
    RealtimeInterrupt, RealtimeMessageItem, RealtimeMessageRole, RealtimeModel,
    RealtimeModelConfig, RealtimeModelEvent, RealtimeModelInputTokensDetails,
    RealtimeModelOutputTokensDetails, RealtimeModelSendEvent, RealtimeModelToolCallEvent,
    RealtimeNoiseReductionConfig, RealtimePlaybackTracker, RealtimeRunConfig, RealtimeRunner,
    RealtimeScriptError, RealtimeSendMatcher, RealtimeSession, RealtimeSessionEvent,
    RealtimeSessionModelSettings, RealtimeStep, RealtimeToolExecutionConfig,
    RealtimeTurnDetectionConfig, RealtimeUserInput, RealtimeVoice, ScriptedRealtimeModel,
    realtime_handoff,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{GuardrailFunctionOutput, OutputGuardrail};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::protocol::Message;

/// How long a test waits for an expected session event before failing.
const EVENT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Deserialize, JsonSchema)]
struct CityArgs {
    city: String,
}

fn weather_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "get_weather",
        "Get the weather for a city",
        |_ctx, args: CityArgs| async move { Ok(format!("The weather in {} is sunny", args.city)) },
    )
}

/// Wait for the first event matching `predicate`, collecting everything seen on the way.
async fn wait_for<F>(
    session: &RealtimeSession<()>,
    predicate: F,
) -> (RealtimeSessionEvent<()>, Vec<RealtimeSessionEvent<()>>)
where
    F: Fn(&RealtimeSessionEvent<()>) -> bool,
{
    let mut seen = Vec::new();
    let found = tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            let event = session
                .recv()
                .await
                .expect("session event stream closed before the expected event arrived");
            if predicate(&event) {
                return event;
            }
            seen.push(event);
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "timed out; saw {:?}",
            seen.iter()
                .map(RealtimeSessionEvent::event_type)
                .collect::<Vec<_>>()
        )
    });
    (found, seen)
}

async fn start_session(
    model: Arc<FakeRealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
) -> RealtimeSession<()> {
    RealtimeRunner::new(agent)
        .with_model(model)
        .run(())
        .await
        .expect("session connects")
}

#[tokio::test]
async fn tool_call_runs_end_to_end_and_sends_function_call_output() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_instructions("Be brief")
        .with_tool(weather_tool())
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    // The session advertises the tool on connect.
    let settings = model.last_session_settings().expect("initial settings");
    let tools = settings.tools.clone().unwrap_or_default();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(settings.instructions.as_deref(), Some("Be brief"));

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent {
            name: "get_weather".to_string(),
            call_id: "call_1".to_string(),
            arguments: r#"{"city":"Oakland"}"#.to_string(),
            id: Some("fc_1".to_string()),
            previous_item_id: None,
        }
    )));

    let (tool_end, seen) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;

    // tool_start precedes tool_end.
    assert!(
        seen.iter().any(|event| match event {
            RealtimeSessionEvent::ToolStart {
                tool, arguments, ..
            } => tool.name == "get_weather" && arguments == r#"{"city":"Oakland"}"#,
            _ => false,
        }),
        "expected tool_start, saw {:?}",
        seen.iter()
            .map(RealtimeSessionEvent::event_type)
            .collect::<Vec<_>>()
    );

    match tool_end {
        RealtimeSessionEvent::ToolEnd {
            agent,
            tool,
            output,
            ..
        } => {
            assert_eq!(agent.name, "Assistant");
            assert_eq!(tool.name, "get_weather");
            assert_eq!(output.to_model_string(), "The weather in Oakland is sunny");
        }
        other => panic!("expected tool_end, got {other:?}"),
    }

    // The output was already sent to the model when tool_end was emitted.
    assert_eq!(
        model.tool_outputs(),
        vec![(
            "call_1".to_string(),
            "The weather in Oakland is sunny".to_string()
        )]
    );
    let sent = model.sent_events();
    let (call, _, start_response) = sent
        .iter()
        .find_map(|event| event.as_tool_output())
        .expect("tool output event");
    assert_eq!(call.call_id, "call_1");
    assert!(start_response, "the tool output starts a new response");

    // The call is recorded in the history.
    let history = session.history();
    let recorded = history
        .iter()
        .find_map(|item| item.as_function_call())
        .expect("history holds the tool call");
    assert_eq!(recorded.name, "get_weather");
    assert_eq!(recorded.status, "completed");
    assert_eq!(
        recorded.output.as_deref(),
        Some("The weather in Oakland is sunny")
    );

    session.close().await.unwrap();
    assert_eq!(model.close_calls(), 1);
}

#[tokio::test]
async fn handoff_switches_agent_and_sends_session_update() {
    let model = Arc::new(FakeRealtimeModel::new());
    let billing = RealtimeAgent::<()>::new("Billing")
        .with_instructions("Talk about billing")
        .with_tool(weather_tool())
        .into_arc();
    let triage = RealtimeAgent::<()>::new("Triage")
        .with_instructions("Route the user")
        .with_handoff(realtime_handoff(billing.clone()))
        .into_arc();
    let session = start_session(model.clone(), triage).await;

    // The handoff is exposed as a tool.
    let settings = model.last_session_settings().expect("initial settings");
    let tools = settings.tools.clone().unwrap_or_default();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "transfer_to_billing");
    assert_eq!(session.current_agent_name(), "Triage");

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("transfer_to_billing", "call_h", "{}")
    )));

    let (handoff, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Handoff { .. })
    })
    .await;
    match handoff {
        RealtimeSessionEvent::Handoff {
            from_agent,
            to_agent,
            ..
        } => {
            assert_eq!(from_agent.name, "Triage");
            assert_eq!(to_agent.name, "Billing");
        }
        other => panic!("expected handoff, got {other:?}"),
    }
    assert_eq!(session.current_agent_name(), "Billing");

    // The session update carrying the new agent's settings precedes the tool output.
    let sent = model.sent_events();
    let update_index = sent
        .iter()
        .position(|event| event.as_session_update().is_some())
        .expect("session.update was sent");
    let output_index = sent
        .iter()
        .position(|event| event.as_tool_output().is_some())
        .expect("tool output was sent");
    assert!(
        update_index < output_index,
        "session.update must reach the model before the tool output starts a response"
    );

    let update = sent[update_index].as_session_update().unwrap();
    assert_eq!(update.instructions.as_deref(), Some("Talk about billing"));
    let names: Vec<&str> = update
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|spec| spec.name.as_str())
        .collect();
    assert_eq!(names, vec!["get_weather"]);

    assert_eq!(
        model.tool_outputs(),
        vec![(
            "call_h".to_string(),
            r#"{"assistant":"Billing"}"#.to_string()
        )]
    );

    session.close().await.unwrap();
}

#[tokio::test]
async fn tool_approval_is_requested_then_approved_and_rejected() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(weather_tool().with_needs_approval(true))
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Paris"}"#)
    )));

    let (approval, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    match &approval {
        RealtimeSessionEvent::ToolApprovalRequired {
            tool,
            call_id,
            arguments,
            ..
        } => {
            assert_eq!(tool.name, "get_weather");
            assert_eq!(call_id, "call_1");
            assert_eq!(arguments, r#"{"city":"Paris"}"#);
        }
        other => panic!("expected tool_approval_required, got {other:?}"),
    }
    // Nothing was sent to the model while the call waits for a decision.
    assert!(model.tool_outputs().is_empty());

    session.approve_tool_call("call_1", false).await.unwrap();
    let (tool_end, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    match tool_end {
        RealtimeSessionEvent::ToolEnd { output, .. } => {
            assert_eq!(output.to_model_string(), "The weather in Paris is sunny");
        }
        other => panic!("expected tool_end, got {other:?}"),
    }
    assert_eq!(
        model.tool_outputs(),
        vec![(
            "call_1".to_string(),
            "The weather in Paris is sunny".to_string()
        )]
    );
    // Approving an unknown call is a no-op.
    session.approve_tool_call("call_1", false).await.unwrap();
    assert_eq!(model.tool_outputs().len(), 1);

    // A second call can be rejected with a custom message.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_weather", "call_2", r#"{"city":"Rome"}"#)
    )));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;

    session
        .reject_tool_call("call_2", false, Some("Not allowed right now".to_string()))
        .await
        .unwrap();
    let (tool_end, seen) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    assert!(
        !seen
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::ToolStart { .. })),
        "a rejected call must not start the tool"
    );
    match tool_end {
        RealtimeSessionEvent::ToolEnd { output, .. } => {
            assert_eq!(output.to_model_string(), "Not allowed right now");
        }
        other => panic!("expected tool_end, got {other:?}"),
    }
    assert_eq!(
        model.tool_outputs().last().cloned(),
        Some(("call_2".to_string(), "Not allowed right now".to_string()))
    );

    session.close().await.unwrap();
}

#[tokio::test]
async fn transcript_deltas_build_history_and_trip_output_guardrails() {
    let model = Arc::new(FakeRealtimeModel::new());
    let guardrail = OutputGuardrail::<()>::new("no-secrets", |_ctx, _agent, output| async move {
        let text = output.as_str().unwrap_or_default().to_string();
        Ok(GuardrailFunctionOutput::new(
            Value::Null,
            text.contains("secret"),
        ))
    });
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_output_guardrail(guardrail)
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        guardrails_settings: openai_agents_sdk::realtime::RealtimeGuardrailsSettings {
            debounce_text_length: 6,
        },
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    assert!(model.emit(RealtimeModelEvent::TurnStarted {
        response_id: Some("resp_1".to_string()),
    }));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::AgentStart { .. })
    })
    .await;

    // Below the debounce threshold: no guardrail run.
    assert!(model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "item_1".to_string(),
        delta: "hi".to_string(),
        response_id: "resp_1".to_string(),
    }));
    // Crossing the threshold with offending text trips the guardrail.
    assert!(model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "item_1".to_string(),
        delta: " the secret is out".to_string(),
        response_id: "resp_1".to_string(),
    }));

    let (tripped, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::GuardrailTripped { .. })
    })
    .await;
    match tripped {
        RealtimeSessionEvent::GuardrailTripped {
            guardrail_results,
            message,
            ..
        } => {
            assert_eq!(guardrail_results.len(), 1);
            assert_eq!(guardrail_results[0].guardrail_name, "no-secrets");
            assert_eq!(message, "hi the secret is out");
        }
        other => panic!("expected guardrail_tripped, got {other:?}"),
    }

    // The response is still the active one, so it is cancelled outright and its audio stopped.
    let sent = model.sent_events();
    let interrupts = model.interrupts();
    assert_eq!(
        interrupts,
        vec![RealtimeInterrupt::forced_for_response("resp_1", true)],
        "expected a forced, response-scoped interrupt, saw {:?}",
        sent.iter()
            .map(RealtimeModelSendEvent::event_type)
            .collect::<Vec<_>>()
    );
    assert!(sent.iter().any(|event| matches!(
        event,
        RealtimeModelSendEvent::UserInput {
            user_input: RealtimeUserInput::Text(text),
        } if text == "guardrail triggered: no-secrets"
    )));

    // The transcript is reflected in the history.
    let history = session.history();
    let message = history
        .iter()
        .find_map(|item| item.as_message())
        .expect("assistant message in history");
    assert_eq!(message.text(), "hi the secret is out");

    session.close().await.unwrap();
}

#[tokio::test]
async fn history_and_audio_events_are_forwarded() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
    let session = start_session(model.clone(), agent).await;

    // The session emits an initial, empty history snapshot.
    let (first, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::HistoryUpdated { .. })
    })
    .await;
    match first {
        RealtimeSessionEvent::HistoryUpdated { history, .. } => assert!(history.is_empty()),
        other => panic!("expected history_updated, got {other:?}"),
    }

    assert!(
        model.emit(RealtimeModelEvent::item_updated(RealtimeMessageItem::new(
            "item_1",
            RealtimeMessageRole::User,
            vec![RealtimeContent::input_text("hello there")],
        )))
    );
    let (added, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::HistoryAdded { .. })
    })
    .await;
    match added {
        RealtimeSessionEvent::HistoryAdded { item, .. } => {
            assert_eq!(item.item_id(), "item_1");
            assert_eq!(item.as_message().unwrap().text(), "hello there");
        }
        other => panic!("expected history_added, got {other:?}"),
    }

    assert!(model.emit(RealtimeModelEvent::audio(
        openai_agents_sdk::realtime::RealtimeModelAudioEvent {
            data: vec![1, 2, 3, 4],
            response_id: "resp_1".to_string(),
            item_id: "item_2".to_string(),
            content_index: 0,
        }
    )));
    let (audio, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Audio { .. })
    })
    .await;
    match audio {
        RealtimeSessionEvent::Audio { audio, item_id, .. } => {
            assert_eq!(audio.data, vec![1, 2, 3, 4]);
            assert_eq!(item_id, "item_2");
        }
        other => panic!("expected audio, got {other:?}"),
    }

    assert!(model.emit(RealtimeModelEvent::AudioDone {
        item_id: "item_2".to_string(),
        content_index: 0,
    }));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::AudioEnd { .. })
    })
    .await;

    assert!(model.emit(RealtimeModelEvent::InputAudioTimeoutTriggered {
        item_id: "item_3".to_string(),
        audio_start_ms: 0,
        audio_end_ms: 500,
    }));
    wait_for(&session, |event| {
        matches!(
            event,
            RealtimeSessionEvent::InputAudioTimeoutTriggered { .. }
        )
    })
    .await;

    // Closing drains the stream and ends it.
    session.close().await.unwrap();
    assert!(session.is_closed());
    while session.recv().await.is_some() {}
}

#[tokio::test]
async fn update_agent_pushes_new_settings() {
    let model = Arc::new(FakeRealtimeModel::new());
    let first = RealtimeAgent::<()>::new("First")
        .with_instructions("first")
        .into_arc();
    let second = RealtimeAgent::<()>::new("Second")
        .with_instructions("second")
        .with_tool(weather_tool())
        .into_arc();
    let session = start_session(model.clone(), first).await;

    session.update_agent(second).await.unwrap();
    assert_eq!(session.current_agent_name(), "Second");

    let updates = model.session_updates();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].instructions.as_deref(), Some("second"));
    assert_eq!(updates[0].tools.as_deref().unwrap_or_default().len(), 1);

    session.close().await.unwrap();
}

#[tokio::test]
async fn dynamic_instructions_and_run_config_settings_reach_the_model() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_instructions_fn(|_ctx, agent| async move { Ok(format!("You are {}", agent.name)) })
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        model_settings: Some(
            RealtimeSessionModelSettings::new()
                .with_voice("marin")
                .with_output_modalities(["text"]),
        ),
        ..Default::default()
    };
    let tracker = RealtimePlaybackTracker::shared();
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .with_model_config(RealtimeModelConfig::new().with_playback_tracker(tracker.clone()))
        .run(())
        .await
        .unwrap();

    // The runner's model config, including the playback tracker, reaches the transport.
    assert!(
        model
            .playback_tracker()
            .is_some_and(|forwarded| Arc::ptr_eq(&forwarded, &tracker)),
        "the session must forward the playback tracker to the transport"
    );

    let settings = model.last_session_settings().unwrap();
    assert_eq!(settings.instructions.as_deref(), Some("You are Assistant"));
    assert_eq!(settings.voice, Some(RealtimeVoice::from("marin")));
    assert_eq!(
        settings.output_modalities.as_deref(),
        Some(["text".to_string()].as_slice())
    );

    session.close().await.unwrap();
}

#[tokio::test]
async fn unknown_tool_reports_an_error_to_the_model() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("nope", "call_x", "{}")
    )));

    let (error, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { .. })
    })
    .await;
    match error {
        RealtimeSessionEvent::Error { error, .. } => {
            assert_eq!(error["message"], "Tool nope not found");
        }
        other => panic!("expected error, got {other:?}"),
    }

    let sent = model.sent_events();
    let (_, output, start_response) = sent
        .iter()
        .find_map(|event| event.as_tool_output())
        .expect("tool output event");
    assert_eq!(output, "Tool nope not found");
    assert!(!start_response, "a missing tool must not start a response");

    session.close().await.unwrap();
}

#[tokio::test]
async fn pre_approval_input_guardrails_block_a_call_before_a_human_sees_it() {
    let blocked = ToolInputGuardrail::<()>::new("no-paris", |data| async move {
        Ok(if data.context.tool_arguments.contains("Paris") {
            ToolGuardrailFunctionOutput::reject_content("That city is off limits")
        } else {
            ToolGuardrailFunctionOutput::allow()
        })
    });
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(
            weather_tool()
                .with_needs_approval(true)
                .with_input_guardrail(blocked),
        )
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        tool_execution: RealtimeToolExecutionConfig {
            pre_approval_tool_input_guardrails: true,
        },
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Paris"}"#)
    )));

    // The guardrail answers the model directly; no approval is ever requested.
    let (tool_end, seen) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    assert!(
        !seen.iter().any(|event| matches!(
            event,
            RealtimeSessionEvent::ToolApprovalRequired { .. }
                | RealtimeSessionEvent::ToolStart { .. }
        )),
        "a pre-approval rejection must not ask for approval or run the tool"
    );
    match tool_end {
        RealtimeSessionEvent::ToolEnd { output, .. } => {
            assert_eq!(output.to_model_string(), "That city is off limits");
        }
        other => panic!("expected tool_end, got {other:?}"),
    }
    assert_eq!(
        model.tool_outputs(),
        vec![("call_1".to_string(), "That city is off limits".to_string())]
    );

    // A call the guardrail allows still goes to a human.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_weather", "call_2", r#"{"city":"Rome"}"#)
    )));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;

    session.close().await.unwrap();
}

#[tokio::test]
async fn a_guardrail_on_text_output_cancels_the_response_without_touching_audio() {
    let guardrail = OutputGuardrail::<()>::new("no-secrets", |_ctx, _agent, output| async move {
        let text = output.as_str().unwrap_or_default().to_string();
        Ok(GuardrailFunctionOutput::new(
            Value::Null,
            text.contains("secret"),
        ))
    });
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_output_guardrail(guardrail)
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        guardrails_settings: openai_agents_sdk::realtime::RealtimeGuardrailsSettings {
            debounce_text_length: 6,
        },
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    // Text deltas open a generation on their own, without a `turn_started`.
    assert!(model.emit(RealtimeModelEvent::OutputTextDelta {
        item_id: "item_1".to_string(),
        delta: "the secret is out".to_string(),
        response_id: "resp_1".to_string(),
    }));

    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::GuardrailTripped { .. })
    })
    .await;

    let deadline = std::time::Instant::now() + EVENT_TIMEOUT;
    let interrupts = loop {
        let interrupts = model.interrupts();
        if !interrupts.is_empty() || std::time::Instant::now() > deadline {
            break interrupts;
        }
        tokio::task::yield_now().await;
    };
    // There is no audio to stop, so only the response is cancelled.
    assert_eq!(
        interrupts,
        vec![RealtimeInterrupt::forced_for_response("resp_1", false)]
    );
    assert!(interrupts[0].cancel_response_only);

    session.close().await.unwrap();
}

#[tokio::test]
async fn a_guardrail_on_a_superseded_response_only_stops_that_response_audio() {
    // The guardrail blocks until the test releases it, so a newer response can start first.
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let guardrail_gate = gate.clone();
    let guardrail = OutputGuardrail::<()>::new("no-secrets", move |_ctx, _agent, output| {
        let gate = guardrail_gate.clone();
        async move {
            let permit = gate.acquire().await.expect("gate stays open");
            permit.forget();
            let text = output.as_str().unwrap_or_default().to_string();
            Ok(GuardrailFunctionOutput::new(
                Value::Null,
                text.contains("secret"),
            ))
        }
    });
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_output_guardrail(guardrail)
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        guardrails_settings: openai_agents_sdk::realtime::RealtimeGuardrailsSettings {
            debounce_text_length: 6,
        },
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    // Generation 1 produces offending text, but the guardrail has not finished yet.
    assert!(model.emit_all([
        RealtimeModelEvent::TurnStarted {
            response_id: Some("resp_1".to_string()),
        },
        RealtimeModelEvent::TranscriptDelta {
            item_id: "item_1".to_string(),
            delta: "the secret is out".to_string(),
            response_id: "resp_1".to_string(),
        },
        RealtimeModelEvent::TurnEnded {
            response_id: Some("resp_1".to_string()),
        },
        // Generation 2 starts, so generation 1 is now superseded.
        RealtimeModelEvent::TurnStarted {
            response_id: Some("resp_2".to_string()),
        },
    ]));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::AgentEnd { .. })
    })
    .await;

    // The first response's audio is still owed cleanup while its guardrail runs.
    assert!(model.retired_responses().is_empty());

    gate.add_permits(1);
    let (tripped, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::GuardrailTripped { .. })
    })
    .await;
    match tripped {
        RealtimeSessionEvent::GuardrailTripped { message, .. } => {
            assert_eq!(message, "the secret is out");
        }
        other => panic!("expected guardrail_tripped, got {other:?}"),
    }

    // Only the superseded response's playback is stopped; the live response keeps going.
    let deadline = std::time::Instant::now() + EVENT_TIMEOUT;
    let interrupts = loop {
        let interrupts = model.interrupts();
        if !interrupts.is_empty() || std::time::Instant::now() > deadline {
            break interrupts;
        }
        tokio::task::yield_now().await;
    };
    assert_eq!(
        interrupts,
        vec![RealtimeInterrupt::playback_only("resp_1")],
        "a superseded response must not be cancelled, only silenced"
    );
    assert!(
        !model.sent_events().iter().any(|event| matches!(
            event,
            RealtimeModelSendEvent::UserInput {
                user_input: RealtimeUserInput::Text(text),
            } if text.starts_with("guardrail triggered")
        )),
        "the newer response must not be told about an older response's guardrail"
    );

    // Once the guardrail settles, the transport may release that response's audio.
    let deadline = std::time::Instant::now() + EVENT_TIMEOUT;
    while model.retired_responses().is_empty() && std::time::Instant::now() < deadline {
        tokio::task::yield_now().await;
    }
    assert_eq!(model.retired_responses(), vec!["resp_1".to_string()]);

    session.close().await.unwrap();
}

#[tokio::test]
async fn a_response_with_no_guardrails_releases_its_audio_immediately() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit_all([
        RealtimeModelEvent::TurnStarted {
            response_id: Some("resp_1".to_string()),
        },
        RealtimeModelEvent::TurnEnded {
            response_id: Some("resp_1".to_string()),
        },
    ]));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::AgentEnd { .. })
    })
    .await;

    assert_eq!(model.retired_responses(), vec!["resp_1".to_string()]);
    session.close().await.unwrap();
}

#[tokio::test]
async fn the_sip_model_builds_an_accept_payload_from_the_agent() {
    let agent = RealtimeAgent::<()>::new("Phone")
        .with_instructions("Answer politely")
        .with_tool(weather_tool())
        .into_arc();

    let payload =
        OpenAIRealtimeSIPModel::build_initial_session_payload(&agent, (), None, None, None)
            .await
            .unwrap();
    assert_eq!(payload["type"], "realtime");
    assert_eq!(payload["model"], "gpt-realtime-2.1");
    assert_eq!(payload["instructions"], "Answer politely");
    assert_eq!(payload["tools"][0]["name"], "get_weather");
    assert_eq!(
        payload["audio"]["input"]["turn_detection"]["type"],
        "semantic_vad"
    );

    // Overrides are applied after the agent and the configs.
    let payload = OpenAIRealtimeSIPModel::build_initial_session_payload(
        &agent,
        (),
        None,
        None,
        Some(&RealtimeSessionModelSettings::new().with_voice("marin")),
    )
    .await
    .unwrap();
    assert_eq!(payload["audio"]["output"]["voice"], "marin");

    // The run config supplies the base settings, and tracing_disabled suppresses tracing.
    let run_config = RealtimeRunConfig::<()> {
        model_settings: Some(RealtimeSessionModelSettings::new().with_output_modalities(["text"])),
        tracing_disabled: true,
        ..Default::default()
    };
    let payload = OpenAIRealtimeSIPModel::build_initial_session_payload(
        &agent,
        (),
        None,
        Some(&run_config),
        None,
    )
    .await
    .unwrap();
    assert_eq!(payload["output_modalities"], json!(["text"]));
}

#[tokio::test]
async fn the_sip_model_requires_a_call_id() {
    let model = OpenAIRealtimeSIPModel::new();
    let error = model
        .connect(RealtimeModelConfig::new().with_api_key("sk-test"))
        .await
        .expect_err("a SIP attach without a call id is rejected");
    assert!(
        error.to_string().contains("requires `call_id`"),
        "unexpected error: {error}"
    );
    assert_eq!(model.model_name(), "gpt-realtime-2.1");
}

#[tokio::test]
async fn the_sip_model_attaches_to_a_call_and_forwards_events() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (client_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                let value: Value = serde_json::from_str(text.as_str()).unwrap();
                let first = value["type"] == "session.update";
                let _ = client_tx.send(value);
                if first {
                    break;
                }
            }
        }
        ws.send(Message::text(
            json!({
                "type": "conversation.item.deleted", "event_id": "evt",
                "item_id": "item_gone",
            })
            .to_string(),
        ))
        .await
        .unwrap();
        while ws.next().await.is_some() {}
    });

    let model = OpenAIRealtimeSIPModel::new();
    let mut events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("sk-test")
                .with_call_id("call_789")
                .with_url(format!(
                    "ws://127.0.0.1:{port}/v1/realtime?call_id=call_789"
                )),
        )
        .await
        .expect("attaches to the call");

    let update = tokio::time::timeout(EVENT_TIMEOUT, client_rx.recv())
        .await
        .expect("session.update received")
        .unwrap();
    assert_eq!(update["type"], "session.update");
    // A call already negotiated its audio formats, so the attach must not override them.
    assert!(update["session"]["audio"]["input"].get("format").is_none());
    assert!(update["session"]["audio"]["output"].get("format").is_none());

    let deleted = tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            if let RealtimeModelEvent::ItemDeleted { item_id } =
                events.recv().await.expect("stream stays open")
            {
                return item_id;
            }
        }
    })
    .await
    .expect("item_deleted mapped");
    assert_eq!(deleted, "item_gone");

    model.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn the_playback_tracker_decides_where_audio_is_truncated() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (client_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                let value: Value = serde_json::from_str(text.as_str()).unwrap();
                let first = value["type"] == "session.update";
                let _ = client_tx.send(value);
                if first {
                    break;
                }
            }
        }
        // Negotiate pcm16, then send 100 ms of audio (24 kHz, 2 bytes per sample).
        ws.send(Message::text(
            json!({
                "type": "session.created", "event_id": "evt",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}},
                },
            })
            .to_string(),
        ))
        .await
        .unwrap();
        let silence = base64_encode(&vec![0u8; 4800]);
        ws.send(Message::text(
            json!({
                "type": "response.output_audio.delta", "event_id": "evt", "output_index": 0,
                "item_id": "item_1",
                "content_index": 0,
                "response_id": "resp_1",
                "delta": silence,
            })
            .to_string(),
        ))
        .await
        .unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message
                && let Ok(value) = serde_json::from_str::<Value>(text.as_str())
            {
                let _ = client_tx.send(value);
            }
        }
    });

    let tracker = RealtimePlaybackTracker::shared();
    let model = OpenAIRealtimeWebSocketModel::new();
    let mut events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("sk-test")
                .with_url(format!("ws://127.0.0.1:{port}/v1/realtime"))
                .with_playback_tracker(tracker.clone()),
        )
        .await
        .expect("connects to the mock server");

    // Wait for the audio to arrive before reporting playback.
    let audio = tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            if let RealtimeModelEvent::Audio(audio) =
                events.recv().await.expect("stream stays open")
            {
                return audio;
            }
        }
    })
    .await
    .expect("audio delta mapped");
    assert_eq!(audio.data.len(), 4800);
    assert_eq!(audio.item_id, "item_1");

    // The transport told the tracker which format the session negotiated.
    // The GA `audio/pcm` name is normalized to `pcm16`, as Python does.
    assert_eq!(tracker.audio_format().as_deref(), Some("pcm16"));

    // The user heard only 30 ms of the 100 ms the model sent.
    tracker.on_play_ms("item_1", 0, 30.0);
    assert_eq!(tracker.state().elapsed_ms, Some(30.0));

    model
        .send_event(RealtimeModelSendEvent::interrupt())
        .await
        .unwrap();

    let truncate = tokio::time::timeout(EVENT_TIMEOUT, async {
        while let Some(value) = client_rx.recv().await {
            if value["type"] == "conversation.item.truncate" {
                return value;
            }
        }
        panic!("conversation.item.truncate never arrived");
    })
    .await
    .expect("truncate received");
    assert_eq!(truncate["item_id"], "item_1");
    assert_eq!(truncate["content_index"], 0);
    assert_eq!(
        truncate["audio_end_ms"], 30,
        "the truncate point comes from the tracker, not from wall-clock time"
    );

    // Interrupting resets the tracker so later playback queries do not reference stale audio.
    assert_eq!(tracker.state().current_item_id, None);

    // The transport also emitted the interruption to the application.
    let interrupted = tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            if let RealtimeModelEvent::AudioInterrupted { item_id, .. } =
                events.recv().await.expect("stream stays open")
            {
                return item_id;
            }
        }
    })
    .await
    .expect("audio_interrupted emitted");
    assert_eq!(interrupted, "item_1");

    model.close().await.unwrap();
    server.abort();
}

/// Minimal base64 encoder, so the mock server does not need the `base64` crate.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[tokio::test]
async fn a_duplicate_tool_call_id_runs_the_tool_only_once() {
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = runs.clone();
    let tool = FunctionTool::new(
        "get_weather",
        "Get the weather for a city",
        move |_ctx, args: CityArgs| {
            let counted = counted.clone();
            async move {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(format!("The weather in {} is sunny", args.city))
            }
        },
    );
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(tool)
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    let call = RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Oslo"}"#);
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call.clone())));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;

    // The transport redelivering the same call must not run the tool again.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call)));
    assert!(model.emit(RealtimeModelEvent::AudioDone {
        item_id: "item_1".to_string(),
        content_index: 0,
    }));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::AudioEnd { .. })
    })
    .await;

    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(model.tool_outputs().len(), 1);

    session.close().await.unwrap();
}

#[tokio::test]
async fn reusing_a_call_id_for_a_different_invocation_is_an_error() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(weather_tool())
        .with_tool(FunctionTool::new(
            "get_time",
            "Get the time",
            |_ctx, _args: CityArgs| async move { Ok("noon".to_string()) },
        ))
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Oslo"}"#)
    )));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;

    // Same call id, different tool: the outputs would collide on the server.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_time", "call_1", r#"{"city":"Oslo"}"#)
    )));
    let (error, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { .. })
    })
    .await;
    match error {
        RealtimeSessionEvent::Error { error, .. } => {
            assert!(
                error["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("reused a Realtime tool call ID"),
                "unexpected error: {error}"
            );
        }
        other => panic!("expected error, got {other:?}"),
    }
    // Only the first call answered the model.
    assert_eq!(model.tool_outputs().len(), 1);

    session.close().await.unwrap();
}

#[tokio::test]
async fn an_undelivered_tool_output_is_cached_and_resent() {
    // The connect-time settings travel with `connect`, so the first send is the tool output.
    let model = Arc::new(ScriptedRealtimeModel::new([
        RealtimeStep::new("tool_output").failing("transport write failed"),
        RealtimeStep::new("tool_output"),
    ]));
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(weather_tool())
        .into_arc();
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .run(())
        .await
        .unwrap();

    let call = RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Oslo"}"#);
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call.clone())));

    // The failed send surfaces as an error event, not a lost call.
    let (error, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { .. })
    })
    .await;
    match error {
        RealtimeSessionEvent::Error { error, .. } => {
            assert_eq!(
                error["message"],
                "Tool output send failed; cached output will be retried"
            );
        }
        other => panic!("expected error, got {other:?}"),
    }

    // Retrying the same call resends the cached output instead of running the tool again.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call)));
    let (tool_end, seen) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    assert!(
        !seen
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::ToolStart { .. })),
        "a cached output must not start the tool a second time"
    );
    match tool_end {
        RealtimeSessionEvent::ToolEnd { output, .. } => {
            assert_eq!(output.to_model_string(), "The weather in Oslo is sunny");
        }
        other => panic!("expected tool_end, got {other:?}"),
    }

    model.assert_complete().unwrap();
    session.close().await.unwrap();
}

#[tokio::test]
async fn the_scripted_model_enforces_the_expected_exchange() {
    let model = Arc::new(ScriptedRealtimeModel::new([RealtimeStep::new(
        RealtimeSendMatcher::predicate(|event| {
            matches!(
                event,
                RealtimeModelSendEvent::UserInput {
                    user_input: RealtimeUserInput::Text(text),
                } if text == "Hello"
            )
        }),
    )
    .emitting([RealtimeModelEvent::TurnStarted {
        response_id: Some("resp_1".to_string()),
    }])]));
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_instructions("Be brief")
        .into_arc();
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_model_config(RealtimeModelConfig::new().with_api_key("sk-test"))
        .run(())
        .await
        .unwrap();

    // The connect snapshot records that a key was supplied, never the key.
    let calls = model.connect_calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].api_key_provided);
    assert!(!calls[0].headers_provided);
    assert!(!calls[0].playback_tracker_provided);
    assert_eq!(calls[0].url, None);
    // Connect-time settings reach the transport through `connect`, not through a send.
    assert_eq!(
        calls[0]
            .initial_model_settings
            .as_ref()
            .and_then(|settings| settings.instructions.as_deref()),
        Some("Be brief")
    );

    assert_eq!(model.remaining_steps(), 1);
    session.send_message("Hello").await.unwrap();
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::AgentStart { .. })
    })
    .await;
    model.assert_complete().unwrap();

    // An unscripted send is refused.
    let error = session
        .send_message("Again")
        .await
        .expect_err("the script is exhausted");
    assert!(
        error.to_string().contains("no scripted steps remain"),
        "unexpected error: {error}"
    );

    session.close().await.unwrap();
    assert_eq!(model.close_calls(), 1);
}

#[tokio::test]
async fn the_scripted_model_reports_unconsumed_steps() {
    let model = Arc::new(ScriptedRealtimeModel::new([
        RealtimeStep::new("audio"),
        RealtimeStep::new("interrupt"),
    ]));
    let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .run(())
        .await
        .unwrap();

    let error = model
        .assert_complete()
        .expect_err("both steps are unconsumed");
    assert_eq!(error, RealtimeScriptError::UnconsumedSteps { remaining: 2 });
    assert!(error.to_string().contains("not consumed"));

    // A send of the wrong kind names what was expected.
    let error = session
        .send_message("Hello")
        .await
        .expect_err("user_input does not match the audio step");
    assert!(
        error.to_string().contains("expected an event of kind"),
        "unexpected error: {error}"
    );

    session.close().await.unwrap();
}

#[tokio::test]
async fn discarding_history_audio_keeps_transcripts() {
    for (storage, expect_audio) in [
        (RealtimeAudioStorage::Keep, true),
        (RealtimeAudioStorage::DiscardAudio, false),
    ] {
        let model = Arc::new(FakeRealtimeModel::new());
        let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
        let config = RealtimeRunConfig::<()> {
            history_audio_storage: storage,
            ..Default::default()
        };
        let session = RealtimeRunner::new(agent)
            .with_model(model.clone())
            .with_config(config)
            .run(())
            .await
            .unwrap();

        assert!(
            model.emit(RealtimeModelEvent::item_updated(RealtimeMessageItem::new(
                "item_1",
                RealtimeMessageRole::Assistant,
                vec![RealtimeContent::Audio {
                    audio: Some("AAAA".to_string()),
                    transcript: Some("hello".to_string()),
                    extra: Default::default(),
                }],
            )))
        );
        wait_for(&session, |event| {
            matches!(event, RealtimeSessionEvent::HistoryAdded { .. })
        })
        .await;

        let history = session.history();
        let message = history
            .iter()
            .find_map(|item| item.as_message())
            .expect("assistant message in history");
        // The transcript survives either way; only the bytes are optional.
        assert_eq!(message.text(), "hello");
        match &message.content[0] {
            RealtimeContent::Audio { audio, .. } => assert_eq!(
                audio.is_some(),
                expect_audio,
                "unexpected audio retention for {storage:?}"
            ),
            other => panic!("expected audio content, got {other:?}"),
        }

        session.close().await.unwrap();
    }
}

#[test]
fn typed_reasoning_config_keeps_the_wire_payload() {
    use openai_agents_sdk::realtime::RealtimeReasoningConfig;

    let transport = OpenAIRealtimeWebSocketModel::new();

    let payload = transport
        .session_payload_for_settings(&RealtimeSessionModelSettings::new().with_reasoning("low"));
    assert_eq!(payload["reasoning"], json!({"effort": "low"}));

    // Keys the struct does not model still reach the wire verbatim.
    let with_extra = RealtimeReasoningConfig::new("high").with_extra("summary", "concise");
    let payload = transport.session_payload_for_settings(
        &RealtimeSessionModelSettings::new().with_reasoning(with_extra.clone()),
    );
    assert_eq!(
        payload["reasoning"],
        json!({"effort": "high", "summary": "concise"})
    );
    assert_eq!(
        serde_json::from_value::<RealtimeReasoningConfig>(payload["reasoning"].clone()).unwrap(),
        with_extra,
        "the payload round-trips back into the typed config"
    );

    // Unset reasoning sends no key at all.
    let payload = transport.session_payload_for_settings(&RealtimeSessionModelSettings::new());
    assert!(payload.get("reasoning").is_none());

    // The override wins when settings are merged, like every other field.
    let merged = RealtimeSessionModelSettings::new()
        .with_reasoning("low")
        .merged_with(&RealtimeSessionModelSettings::new().with_reasoning("xhigh"));
    assert_eq!(
        merged.reasoning,
        Some(RealtimeReasoningConfig::new("xhigh"))
    );
}

#[test]
fn log_redaction_overrides_beat_the_environment() {
    use openai_agents_sdk::realtime::{
        dont_log_model_data, reset_dont_log_model_data, set_dont_log_model_data,
    };

    // Redaction is on unless the environment turns it off, so derive the expectation rather than
    // assuming an unset variable.
    let from_env = !matches!(
        std::env::var("OPENAI_AGENTS_DONT_LOG_MODEL_DATA")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no"
    );
    assert_eq!(dont_log_model_data(), from_env);

    set_dont_log_model_data(false);
    assert!(!dont_log_model_data());
    set_dont_log_model_data(true);
    assert!(dont_log_model_data());
    reset_dont_log_model_data();
    assert_eq!(dont_log_model_data(), from_env);
}

#[tokio::test]
async fn a_call_redelivered_while_its_approved_tool_runs_is_ignored() {
    // The tool blocks until the test releases it, so the redelivery lands mid-run.
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tool_gate = gate.clone();
    let counted = runs.clone();
    let tool = FunctionTool::new(
        "get_weather",
        "Get the weather for a city",
        move |_ctx, args: CityArgs| {
            let gate = tool_gate.clone();
            let counted = counted.clone();
            async move {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let permit = gate.acquire().await.expect("gate stays open");
                permit.forget();
                Ok(format!("The weather in {} is sunny", args.city))
            }
        },
    )
    .with_needs_approval(true);

    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(tool)
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    let call = RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Oslo"}"#);
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call.clone())));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;

    // Approving runs the tool, which blocks, so the approval future stays pending.
    let approving = {
        let session = session.clone();
        tokio::spawn(async move { session.approve_tool_call("call_1", false).await })
    };
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolStart { .. })
    })
    .await;

    // The model redelivers the call while the approved tool is still running.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call)));
    assert!(model.emit(RealtimeModelEvent::AudioDone {
        item_id: "item_1".to_string(),
        content_index: 0,
    }));
    let (_, seen) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::AudioEnd { .. })
    })
    .await;
    assert!(
        !seen
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::Error { .. })),
        "a redelivery during an approved run must be ignored, not reported as an error"
    );

    gate.add_permits(1);
    approving.await.unwrap().unwrap();
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(model.tool_outputs().len(), 1);

    session.close().await.unwrap();
}

#[tokio::test]
async fn reusing_a_rejected_call_id_for_another_tool_is_an_error() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(weather_tool().with_needs_approval(true))
        .with_tool(FunctionTool::new(
            "get_time",
            "Get the time",
            |_ctx, _args: CityArgs| async move { Ok("noon".to_string()) },
        ))
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Oslo"}"#)
    )));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    session
        .reject_tool_call("call_1", false, None)
        .await
        .unwrap();
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;

    // The rejected call never ran a tool, but its id is still taken.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_time", "call_1", r#"{"city":"Oslo"}"#)
    )));
    let (error, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { .. })
    })
    .await;
    match error {
        RealtimeSessionEvent::Error { error, .. } => assert!(
            error["message"]
                .as_str()
                .unwrap_or_default()
                .contains("reused a Realtime tool call ID"),
            "unexpected error: {error}"
        ),
        other => panic!("expected error, got {other:?}"),
    }
    assert_eq!(model.tool_outputs().len(), 1);

    session.close().await.unwrap();
}

#[tokio::test]
async fn an_undelivered_handoff_output_resends_its_session_update() {
    let model = Arc::new(ScriptedRealtimeModel::new([
        // The handoff's session.update is the first send, and it fails.
        RealtimeStep::new("session_update").failing("transport write failed"),
        RealtimeStep::new("session_update"),
        RealtimeStep::new("tool_output"),
    ]));
    let billing = RealtimeAgent::<()>::new("Billing")
        .with_instructions("Talk about billing")
        .into_arc();
    let agent = RealtimeAgent::<()>::new("Triage")
        .with_handoff(realtime_handoff(billing))
        .into_arc();
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .run(())
        .await
        .unwrap();

    let call = RealtimeModelToolCallEvent::new("transfer_to_billing", "call_h", "{}");
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call.clone())));
    let (error, _) = wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { .. })
    })
    .await;
    match error {
        RealtimeSessionEvent::Error { error, .. } => assert_eq!(
            error["message"],
            "Tool output send failed; cached output will be retried"
        ),
        other => panic!("expected error, got {other:?}"),
    }
    // The agent switched even though the output never landed.
    assert_eq!(session.current_agent_name(), "Billing");

    // The retry resends the session update before the output, in that order.
    assert!(model.emit(RealtimeModelEvent::FunctionCall(call)));
    let deadline = std::time::Instant::now() + EVENT_TIMEOUT;
    while model.remaining_steps() > 0 && std::time::Instant::now() < deadline {
        tokio::task::yield_now().await;
    }
    model.assert_complete().unwrap();

    let sent = model.sent_events();
    let update_index = sent
        .iter()
        .rposition(|event| event.as_session_update().is_some())
        .expect("session.update resent");
    let output_index = sent
        .iter()
        .position(|event| event.as_tool_output().is_some())
        .expect("tool output sent");
    assert!(
        update_index < output_index,
        "the cached session.update must precede the output on a retry"
    );
    let update = sent[update_index].as_session_update().unwrap();
    assert_eq!(update.instructions.as_deref(), Some("Talk about billing"));

    session.close().await.unwrap();
}

#[tokio::test]
async fn the_nested_audio_config_wins_over_the_flat_keys() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
    let settings = RealtimeSessionModelSettings::new()
        // Flat legacy keys, which the nested tree below must override.
        .with_voice("ash")
        .with_audio(
            RealtimeAudioConfig::new()
                .with_input(RealtimeAudioInputConfig {
                    format: Some(RealtimeAudioFormat::from("g711_ulaw")),
                    transcription: Some(RealtimeInputAudioTranscriptionConfig {
                        model: Some("gpt-transcribe".to_string()),
                        keywords: Some(vec!["Oslo".to_string()]),
                        languages: Some(vec!["en".to_string()]),
                        delay: Some("low".to_string()),
                        ..Default::default()
                    }),
                    turn_detection: Some(Some(RealtimeTurnDetectionConfig {
                        kind: Some("server_vad".to_string()),
                        model_version: Some("v2".to_string()),
                        ..Default::default()
                    })),
                    noise_reduction: Some(Some(RealtimeNoiseReductionConfig {
                        kind: Some("far_field".to_string()),
                    })),
                })
                .with_output(RealtimeAudioOutputConfig {
                    format: Some(RealtimeAudioFormat::from("g711_alaw")),
                    voice: Some(RealtimeVoice::Custom {
                        id: "voice_123".to_string(),
                    }),
                    speed: Some(1.25),
                }),
        );
    let mut settings = settings;
    settings.input_audio_format = Some(RealtimeAudioFormat::from("pcm16"));
    settings.output_audio_format = Some(RealtimeAudioFormat::from("pcm16"));
    settings.speed = Some(0.5);
    // `modalities` is the legacy name, used only when `output_modalities` is unset.
    settings.modalities = Some(vec!["text".to_string()]);

    let config = RealtimeRunConfig::<()> {
        model_settings: Some(settings),
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    let payload = session_payload_for(&model);
    assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcmu");
    assert_eq!(payload["audio"]["output"]["format"]["type"], "audio/pcma");
    assert_eq!(payload["audio"]["output"]["voice"]["id"], "voice_123");
    assert_eq!(payload["audio"]["output"]["speed"], 1.25);
    assert_eq!(
        payload["audio"]["input"]["transcription"]["model"],
        "gpt-transcribe"
    );
    assert_eq!(
        payload["audio"]["input"]["transcription"]["keywords"][0],
        "Oslo"
    );
    assert_eq!(
        payload["audio"]["input"]["transcription"]["languages"][0],
        "en"
    );
    assert_eq!(payload["audio"]["input"]["transcription"]["delay"], "low");
    assert_eq!(
        payload["audio"]["input"]["turn_detection"]["type"],
        "server_vad"
    );
    assert_eq!(
        payload["audio"]["input"]["turn_detection"]["model_version"],
        "v2"
    );
    assert_eq!(
        payload["audio"]["input"]["noise_reduction"]["type"],
        "far_field"
    );
    assert_eq!(payload["output_modalities"], json!(["text"]));

    session.close().await.unwrap();
}

/// Render the session payload the transport would send for the session's current settings.
fn session_payload_for(model: &Arc<FakeRealtimeModel>) -> Value {
    let settings = model.last_session_settings().expect("initial settings");
    let transport = OpenAIRealtimeWebSocketModel::new();
    transport.session_payload_for_settings(&settings)
}

#[tokio::test]
async fn mcp_tools_reach_the_realtime_session() {
    #[cfg(feature = "mcp")]
    {
        use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};

        #[derive(Debug)]
        struct StaticMcpServer;

        #[async_trait::async_trait]
        impl MCPServer for StaticMcpServer {
            fn name(&self) -> &str {
                "docs"
            }
            fn config(&self) -> &MCPServerConfig {
                static CONFIG: std::sync::OnceLock<MCPServerConfig> = std::sync::OnceLock::new();
                CONFIG.get_or_init(MCPServerConfig::default)
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
                    name: "search_docs".to_string(),
                    title: None,
                    description: Some("Search the docs".to_string()),
                    input_schema: json!({"type": "object", "properties": {}}),
                    annotations: None,
                }])
            }
            async fn call_tool(
                &self,
                _tool_name: &str,
                _arguments: Option<serde_json::Map<String, Value>>,
            ) -> openai_agents_sdk::Result<Value> {
                Ok(json!({"content": [{"type": "text", "text": "a doc"}]}))
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

        let model = Arc::new(FakeRealtimeModel::new());
        let agent = RealtimeAgent::<()>::new("Assistant")
            .with_tool(weather_tool())
            .with_mcp_server(Arc::new(StaticMcpServer))
            .into_arc();
        let session = start_session(model.clone(), agent).await;

        // The MCP server's tool is advertised to the model alongside the agent's own.
        let settings = model.last_session_settings().expect("initial settings");
        let names: Vec<&str> = settings
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|spec| spec.name.as_str())
            .collect();
        assert!(
            names.contains(&"search_docs"),
            "expected the MCP tool, saw {names:?}"
        );
        assert!(names.contains(&"get_weather"));

        // And the session can run it.
        assert!(model.emit(RealtimeModelEvent::FunctionCall(
            RealtimeModelToolCallEvent::new("search_docs", "call_1", "{}")
        )));
        let (tool_end, _) = wait_for(&session, |event| {
            matches!(event, RealtimeSessionEvent::ToolEnd { .. })
        })
        .await;
        match tool_end {
            RealtimeSessionEvent::ToolEnd { tool, output, .. } => {
                assert_eq!(tool.name, "search_docs");
                assert_eq!(output.to_model_string(), "a doc");
            }
            other => panic!("expected tool_end, got {other:?}"),
        }

        session.close().await.unwrap();
    }
}

#[tokio::test]
async fn the_transport_re_requests_an_item_the_server_rewrote() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (client_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                let value: Value = serde_json::from_str(text.as_str()).unwrap();
                let first = value["type"] == "session.update";
                let _ = client_tx.send(value);
                if first {
                    break;
                }
            }
        }
        // Audio for item_1 makes it the current item, then the server rewrites it.
        let silence = base64_encode(&[0u8; 48]);
        for event in [
            json!({
                "type": "response.output_audio.delta", "event_id": "evt", "output_index": 0,
                "item_id": "item_1",
                "content_index": 0,
                "response_id": "resp_1",
                "delta": silence,
            }),
            json!({
                "type": "conversation.item.input_audio_transcription.completed", "content_index": 0, "event_id": "evt",
                "item_id": "user_1",
                "transcript": "hello",
                "usage": {"type": "tokens", "total_tokens": 1},
            }),
            json!({"type": "conversation.item.truncated", "audio_end_ms": 0, "content_index": 0, "event_id": "evt", "item_id": "item_1"}),
        ] {
            ws.send(Message::text(event.to_string())).await.unwrap();
        }
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message
                && let Ok(value) = serde_json::from_str::<Value>(text.as_str())
            {
                let _ = client_tx.send(value);
            }
        }
    });

    let model = OpenAIRealtimeWebSocketModel::new();
    let mut events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("sk-test")
                .with_url(format!("ws://127.0.0.1:{port}/v1/realtime")),
        )
        .await
        .expect("connects to the mock server");

    // The transcription still reaches the session.
    tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            if let RealtimeModelEvent::InputAudioTranscriptionCompleted { transcript, .. } =
                events.recv().await.expect("stream stays open")
            {
                return transcript;
            }
        }
    })
    .await
    .expect("transcription mapped");

    // Both rewrites prompt a re-read of the item the model is speaking.
    let mut retrieves = 0;
    tokio::time::timeout(EVENT_TIMEOUT, async {
        while let Some(value) = client_rx.recv().await {
            if value["type"] == "conversation.item.retrieve" {
                assert_eq!(value["item_id"], "item_1");
                retrieves += 1;
                if retrieves == 2 {
                    return;
                }
            }
        }
        panic!("conversation.item.retrieve never arrived");
    })
    .await
    .expect("retrieve requests received");

    model.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn usage_events_carry_the_modality_breakdown() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(RealtimeModelEvent::Usage {
        usage: Box::new(openai_agents_sdk::Usage {
            requests: 1,
            input_tokens: 10,
            output_tokens: 4,
            total_tokens: 14,
            ..Default::default()
        }),
        input_tokens_details: Some(RealtimeModelInputTokensDetails {
            text_tokens: Some(6),
            audio_tokens: Some(4),
            ..Default::default()
        }),
        output_tokens_details: Some(RealtimeModelOutputTokensDetails {
            text_tokens: Some(1),
            audio_tokens: Some(3),
        }),
    }));

    let (raw, _) = wait_for(&session, |event| {
        matches!(
            event,
            RealtimeSessionEvent::RawModelEvent {
                data: RealtimeModelEvent::Usage { .. },
                ..
            }
        )
    })
    .await;
    match raw {
        RealtimeSessionEvent::RawModelEvent {
            data:
                RealtimeModelEvent::Usage {
                    input_tokens_details,
                    output_tokens_details,
                    ..
                },
            ..
        } => {
            assert_eq!(input_tokens_details.unwrap().audio_tokens, Some(4));
            assert_eq!(output_tokens_details.unwrap().audio_tokens, Some(3));
        }
        other => panic!("expected a usage raw event, got {other:?}"),
    }

    // The totals still reach the shared usage accounting.
    assert_eq!(session.context().usage().total_tokens, 14);

    session.close().await.unwrap();
}

#[tokio::test]
async fn hooks_see_the_same_lifecycle_as_the_events() {
    #[derive(Debug, Default)]
    struct RecordingHooks {
        calls: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl openai_agents_sdk::realtime::RealtimeHooks<()> for RecordingHooks {
        async fn on_agent_start(
            &self,
            _context: &openai_agents_sdk::RunContext<()>,
            agent: &Arc<RealtimeAgent<()>>,
        ) {
            self.calls
                .lock()
                .unwrap()
                .push(format!("start:{}", agent.name));
        }
        async fn on_tool_start(
            &self,
            _context: &openai_agents_sdk::RunContext<()>,
            _agent: &Arc<RealtimeAgent<()>>,
            tool: &Arc<openai_agents_sdk::tool::FunctionTool<()>>,
        ) {
            self.calls
                .lock()
                .unwrap()
                .push(format!("tool_start:{}", tool.name));
        }
        async fn on_tool_end(
            &self,
            _context: &openai_agents_sdk::RunContext<()>,
            _agent: &Arc<RealtimeAgent<()>>,
            tool: &Arc<openai_agents_sdk::tool::FunctionTool<()>>,
            _output: &openai_agents_sdk::tool::ToolOutput,
        ) {
            self.calls
                .lock()
                .unwrap()
                .push(format!("tool_end:{}", tool.name));
        }
    }

    let hooks = Arc::new(RecordingHooks::default());
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(weather_tool())
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        hooks: Some(hooks.clone()),
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    assert!(model.emit(RealtimeModelEvent::TurnStarted {
        response_id: Some("resp_1".to_string()),
    }));
    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("get_weather", "call_1", r#"{"city":"Oslo"}"#)
    )));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;

    let calls = hooks.calls.lock().unwrap().clone();
    assert_eq!(
        calls,
        vec![
            "start:Assistant".to_string(),
            "tool_start:get_weather".to_string(),
            "tool_end:get_weather".to_string(),
        ]
    );

    session.close().await.unwrap();
}

/// What the mock server captured from the client handshake.
#[derive(Debug)]
struct Handshake {
    uri: String,
    authorization: Option<String>,
    api_key: Option<String>,
}

#[tokio::test]
async fn websocket_model_speaks_the_ga_protocol_against_a_mock_server() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    let (handshake_tx, handshake_rx) = tokio::sync::oneshot::channel::<Handshake>();
    let (client_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut handshake_tx = Some(handshake_tx);
        #[allow(clippy::result_large_err)]
        let callback = |request: &Request, response: Response| {
            let captured = Handshake {
                uri: request.uri().to_string(),
                authorization: request
                    .headers()
                    .get("authorization")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string),
                api_key: request
                    .headers()
                    .get("api-key")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string),
            };
            if let Some(tx) = handshake_tx.take() {
                let _ = tx.send(captured);
            }
            Ok(response)
        };
        let mut ws = tokio_tungstenite::accept_hdr_async(stream, callback)
            .await
            .unwrap();

        // The client sends its initial session.update right after connecting.
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                let value: Value = serde_json::from_str(text.as_str()).unwrap();
                let is_first_update = value["type"] == "session.update";
                let _ = client_tx.send(value);
                if is_first_update {
                    break;
                }
            }
        }

        // Acknowledge the session, then push one server event the client must map.
        ws.send(Message::text(
            json!({
                "type": "session.created", "event_id": "evt",
                "session": {"type": "realtime", "model": "gpt-realtime-2.1"},
            })
            .to_string(),
        ))
        .await
        .unwrap();
        ws.send(Message::text(
            json!({
                "type": "response.output_audio_transcript.delta", "content_index": 0, "event_id": "evt", "output_index": 0,
                "item_id": "item_1",
                "response_id": "resp_1",
                "delta": "hello from the server",
            })
            .to_string(),
        ))
        .await
        .unwrap();

        // Keep reading so the client's tracing session.update is captured too.
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message
                && let Ok(value) = serde_json::from_str::<Value>(text.as_str())
            {
                let _ = client_tx.send(value);
            }
        }
    });

    let model = OpenAIRealtimeWebSocketModel::new();
    let mut events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("sk-test")
                .with_url(format!(
                    "ws://127.0.0.1:{port}/v1/realtime?model=test-model"
                ))
                .with_initial_model_settings(
                    RealtimeSessionModelSettings::new()
                        .with_instructions("Be brief")
                        .with_output_modalities(["text"]),
                ),
        )
        .await
        .expect("connects to the mock server");

    let handshake = tokio::time::timeout(EVENT_TIMEOUT, handshake_rx)
        .await
        .expect("handshake captured")
        .unwrap();
    assert_eq!(handshake.uri, "/v1/realtime?model=test-model");
    assert_eq!(
        handshake.authorization.as_deref(),
        Some("Bearer sk-test"),
        "the transport authenticates with a bearer token"
    );
    assert!(handshake.api_key.is_none());

    // The first client message is the session.update built from the settings.
    let update = tokio::time::timeout(EVENT_TIMEOUT, client_rx.recv())
        .await
        .expect("session.update received")
        .unwrap();
    assert_eq!(update["type"], "session.update");
    let session = &update["session"];
    assert_eq!(session["type"], "realtime");
    assert_eq!(session["model"], "gpt-realtime-2.1");
    assert_eq!(session["instructions"], "Be brief");
    assert_eq!(session["output_modalities"], json!(["text"]));
    assert_eq!(session["audio"]["output"]["voice"], "ash");
    assert_eq!(session["audio"]["input"]["format"]["type"], "audio/pcm");
    assert_eq!(
        session["audio"]["input"]["transcription"]["model"],
        "gpt-4o-mini-transcribe"
    );
    assert_eq!(
        session["audio"]["input"]["turn_detection"]["type"],
        "semantic_vad"
    );
    assert_eq!(session["tools"], json!([]));

    // The raw server event is forwarded, then mapped to a transcript delta.
    let mut saw_raw = false;
    let mapped = tokio::time::timeout(EVENT_TIMEOUT, async {
        loop {
            match events.recv().await.expect("event stream stays open") {
                RealtimeModelEvent::RawServerEvent { .. } => saw_raw = true,
                RealtimeModelEvent::TranscriptDelta {
                    item_id,
                    delta,
                    response_id,
                } => return (item_id, delta, response_id),
                _ => {}
            }
        }
    })
    .await
    .expect("transcript delta mapped");
    assert!(saw_raw, "raw server events are forwarded");
    assert_eq!(mapped.0, "item_1");
    assert_eq!(mapped.1, "hello from the server");
    assert_eq!(mapped.2, "resp_1");

    // `session.created` makes the transport push its tracing configuration.
    let tracing_update = tokio::time::timeout(EVENT_TIMEOUT, async {
        while let Some(value) = client_rx.recv().await {
            if value["type"] == "session.update" && value["session"]["tracing"] == "auto" {
                return value;
            }
        }
        panic!("tracing session.update never arrived");
    })
    .await
    .expect("tracing session.update received");
    assert_eq!(tracing_update["session"]["model"], "gpt-realtime-2.1");

    model.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn custom_headers_replace_the_bearer_token() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (handshake_tx, handshake_rx) = tokio::sync::oneshot::channel::<Handshake>();

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut handshake_tx = Some(handshake_tx);
        #[allow(clippy::result_large_err)]
        let callback = |request: &Request, response: Response| {
            let captured = Handshake {
                uri: request.uri().to_string(),
                authorization: request
                    .headers()
                    .get("authorization")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string),
                api_key: request
                    .headers()
                    .get("api-key")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string),
            };
            if let Some(tx) = handshake_tx.take() {
                let _ = tx.send(captured);
            }
            Ok(response)
        };
        let mut ws = tokio_tungstenite::accept_hdr_async(stream, callback)
            .await
            .unwrap();
        while ws.next().await.is_some() {}
    });

    let model = OpenAIRealtimeWebSocketModel::new();
    model
        .connect(
            RealtimeModelConfig::new()
                .with_url(format!("ws://127.0.0.1:{port}/v1/realtime"))
                .with_header("api-key", "from-header"),
        )
        .await
        .expect("connects with custom headers");

    let handshake = tokio::time::timeout(EVENT_TIMEOUT, handshake_rx)
        .await
        .expect("handshake captured")
        .unwrap();
    assert_eq!(handshake.api_key.as_deref(), Some("from-header"));
    assert!(
        handshake.authorization.is_none(),
        "custom headers suppress the default Authorization header"
    );

    model.close().await.unwrap();
    server.abort();
}

/// A tool that blocks until the test releases it, so the event loop can be observed while it runs.
fn gated_tool(gate: Arc<tokio::sync::Semaphore>) -> FunctionTool<()> {
    FunctionTool::new(
        "slow_tool",
        "Blocks until released",
        move |_ctx, _args: CityArgs| {
            let gate = gate.clone();
            async move {
                let permit = gate.acquire().await.expect("gate open");
                permit.forget();
                Ok("done".to_string())
            }
        },
    )
}

/// Whether `event` forwards the transcript delta the `async_tool_calls` tests emit.
fn is_transcript_delta(event: &RealtimeSessionEvent<()>) -> bool {
    matches!(
        event,
        RealtimeSessionEvent::RawModelEvent {
            data: RealtimeModelEvent::TranscriptDelta { delta, .. },
            ..
        } if delta == "hi"
    )
}

fn slow_tool_call() -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent {
        name: "slow_tool".to_string(),
        call_id: "call_slow".to_string(),
        arguments: r#"{"city":"Oslo"}"#.to_string(),
        id: Some("fc_slow".to_string()),
        previous_item_id: None,
    })
}

#[tokio::test]
async fn a_tool_call_runs_off_the_event_loop_by_default() {
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(gated_tool(gate.clone()))
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(slow_tool_call()));
    assert!(model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "item_1".to_string(),
        delta: "hi".to_string(),
        response_id: "resp_1".to_string(),
    }));

    // The transcript arrives while the tool is still blocked, which is what Python's
    // `async_tool_calls` default buys.
    let (_, seen) = wait_for(&session, is_transcript_delta).await;
    assert!(
        !seen
            .iter()
            .any(|event| matches!(event, RealtimeSessionEvent::ToolEnd { .. })),
        "the blocked tool must not have finished yet"
    );

    gate.add_permits(1);
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::ToolEnd { .. })
    })
    .await;
    session.close().await.unwrap();
}

#[tokio::test]
async fn async_tool_calls_off_serializes_a_call_with_the_event_stream() {
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(gated_tool(gate.clone()))
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        async_tool_calls: false,
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    assert!(model.emit(slow_tool_call()));
    assert!(model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "item_1".to_string(),
        delta: "hi".to_string(),
        response_id: "resp_1".to_string(),
    }));
    gate.add_permits(1);

    // With the flag off the transcript waits behind the tool call.
    let (_, seen) = wait_for(&session, is_transcript_delta).await;
    assert!(
        seen.iter()
            .any(|event| matches!(event, RealtimeSessionEvent::ToolEnd { .. })),
        "the tool ran before the transcript was processed"
    );
    session.close().await.unwrap();
}

#[tokio::test]
async fn a_transport_exception_ends_the_session() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(RealtimeModelEvent::Exception {
        message: "socket closed".to_string(),
        context: Some("read loop".to_string()),
    }));

    // The error reaches the caller first...
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { error, .. }
            if error["message"] == "read loop: socket closed")
    })
    .await;
    // ...then the stream ends, where Python raises the stored exception.
    let ended = tokio::time::timeout(EVENT_TIMEOUT, async {
        while let Some(event) = session.recv().await {
            let _ = event;
        }
    })
    .await;
    assert!(ended.is_ok(), "the session stream ended");
    assert!(session.is_closed());
    let error = session.take_error().expect("the failure is kept");
    assert!(error.to_string().contains("socket closed"));
    assert!(session.take_error().is_none(), "the failure is taken once");
}

#[tokio::test]
async fn a_hosted_tool_on_a_realtime_agent_is_rejected() {
    use openai_agents_sdk::tool::WebSearchTool;

    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(WebSearchTool::new())
        .into_arc();
    let error = RealtimeRunner::new(agent)
        .with_model(model)
        .run(())
        .await
        .expect_err("realtime models accept function tools only");
    assert!(
        error.to_string().contains("Must be a function tool"),
        "got {error}"
    );
}

#[test]
fn unknown_content_part_fields_survive_a_round_trip() {
    let wire = json!({
        "type": "message",
        "item_id": "item_1",
        "role": "assistant",
        "content": [{"type": "audio", "transcript": "hi", "logprobs": [1, 2]}],
        "vendor_field": "kept",
    });
    let item: openai_agents_sdk::realtime::RealtimeItem =
        serde_json::from_value(wire.clone()).expect("parses");
    assert_eq!(serde_json::to_value(&item).expect("serializes"), wire);
}

#[tokio::test]
async fn the_tool_error_formatter_is_the_core_one() {
    use openai_agents_sdk::{ToolErrorFormatterArgs, ToolErrorKind};

    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(weather_tool())
        .into_arc();
    let config = RealtimeRunConfig::<()> {
        tool_error_formatter: Some(openai_agents_sdk::tool_error_formatter_fn(
            |args: &ToolErrorFormatterArgs<'_, ()>| {
                assert!(matches!(args.kind, ToolErrorKind::ToolNotFound));
                assert!(args.error.is_none());
                Some(format!("no tool called {}", args.tool_name))
            },
        )),
        ..Default::default()
    };
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent {
            name: "missing".to_string(),
            call_id: "call_x".to_string(),
            arguments: "{}".to_string(),
            id: None,
            previous_item_id: None,
        }
    )));
    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { .. })
    })
    .await;
    assert_eq!(
        model.tool_outputs(),
        vec![("call_x".to_string(), "no tool called missing".to_string())]
    );
    session.close().await.unwrap();
}

#[tokio::test]
async fn a_tool_guardrail_tripwire_ends_the_session() {
    let tripwire = ToolInputGuardrail::<()>::new("halt", |_data| async move {
        Ok(ToolGuardrailFunctionOutput::raise_exception())
    });
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("Assistant")
        .with_tool(weather_tool().with_input_guardrail(tripwire))
        .into_arc();
    let session = start_session(model.clone(), agent).await;

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent {
            name: "get_weather".to_string(),
            call_id: "call_1".to_string(),
            arguments: r#"{"city":"Oslo"}"#.to_string(),
            id: Some("fc_1".to_string()),
            previous_item_id: None,
        }
    )));

    wait_for(&session, |event| {
        matches!(event, RealtimeSessionEvent::Error { error, .. }
            if error["message"] == "Tool call task failed")
    })
    .await;

    let drained = tokio::time::timeout(EVENT_TIMEOUT, async {
        while session.recv().await.is_some() {}
    })
    .await;
    assert!(drained.is_ok(), "the failed tool task ended the stream");
    assert!(session.is_closed());
    assert!(
        session
            .take_error()
            .is_some_and(|error| error.to_string().contains("halt")),
        "the tripwire is kept for the caller"
    );
    assert!(model.tool_outputs().is_empty(), "the tool never ran");
}
