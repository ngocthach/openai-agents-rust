//! End-to-end realtime session behavior against a fake transport, plus the OpenAI WebSocket
//! transport against a local mock server.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, OpenAIRealtimeWebSocketModel, RealtimeAgent, RealtimeContent,
    RealtimeMessageItem, RealtimeMessageRole, RealtimeModel, RealtimeModelConfig,
    RealtimeModelEvent, RealtimeModelToolCallEvent, RealtimeRunConfig, RealtimeRunner,
    RealtimeSession, RealtimeSessionEvent, RealtimeSessionModelSettings,
    RealtimeToolExecutionConfig, realtime_handoff,
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
    assert_eq!(tools[0].name, "transfer_to_Billing");
    assert_eq!(session.current_agent_name(), "Triage");

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("transfer_to_Billing", "call_h", "{}")
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

    // The session interrupted the model and told it which guardrail tripped.
    let sent = model.sent_events();
    assert!(
        sent.iter().any(|event| matches!(
            event,
            openai_agents_sdk::realtime::RealtimeModelSendEvent::Interrupt {
                force_response_cancel: true
            }
        )),
        "expected an interrupt, saw {:?}",
        sent.iter()
            .map(openai_agents_sdk::realtime::RealtimeModelSendEvent::event_type)
            .collect::<Vec<_>>()
    );
    assert!(sent.iter().any(|event| matches!(
        event,
        openai_agents_sdk::realtime::RealtimeModelSendEvent::UserInput {
            user_input: openai_agents_sdk::realtime::RealtimeUserInput::Text(text),
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
    let session = RealtimeRunner::new(agent)
        .with_model(model.clone())
        .with_config(config)
        .run(())
        .await
        .unwrap();

    let settings = model.last_session_settings().unwrap();
    assert_eq!(settings.instructions.as_deref(), Some("You are Assistant"));
    assert_eq!(settings.voice.as_deref(), Some("marin"));
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
                "type": "session.created",
                "session": {"type": "realtime", "model": "gpt-realtime-2.1"},
            })
            .to_string(),
        ))
        .await
        .unwrap();
        ws.send(Message::text(
            json!({
                "type": "response.output_audio_transcript.delta",
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
