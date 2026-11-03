//! Port of Python `tests/realtime/test_realtime_model_settings.py`: how the agent's tools,
//! handoffs and instructions are merged into the model settings, the SIP accept payload and the
//! call-id `session.update`.
//!
//! Python's settings carry live `FunctionTool` / `Handoff` objects, so overriding them through
//! `initial_model_settings` or `overrides` has no Rust analog (the settings hold wire specs). The
//! enablement filtering those tests exercise is covered here through the agent's own tools and
//! handoffs.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::prompts::Prompt;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, OpenAIRealtimeSIPModel, RealtimeAgent, RealtimeAudioConfig,
    RealtimeAudioFormat, RealtimeAudioInputConfig, RealtimeHandoff, RealtimeModel,
    RealtimeModelConfig, RealtimeModelTracing, RealtimeRunConfig, RealtimeRunner,
    RealtimeSessionModelSettings, RealtimeVoice,
};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::protocol::Message;

fn billing_tool(enabled: bool) -> FunctionTool<()> {
    FunctionTool::new_no_args("transfer_to_billing", "tool", |_ctx| async { Ok("ok") })
        .with_is_enabled(enabled)
}

fn billing_handoff(enabled: bool) -> RealtimeHandoff<()> {
    RealtimeHandoff::new(RealtimeAgent::<()>::new("billing").into_arc())
        .with_tool_name("transfer_to_billing")
        .with_tool_description("Transfer to billing")
        .with_is_enabled(enabled)
}

fn counting_handoff(count: Arc<AtomicUsize>) -> RealtimeHandoff<()> {
    RealtimeHandoff::new(RealtimeAgent::<()>::new("billing").into_arc()).with_is_enabled_fn(
        move |_ctx, _agent| {
            let call = count.fetch_add(1, Ordering::SeqCst) + 1;
            async move { call == 1 }
        },
    )
}

async fn connect_names(
    agent: Arc<RealtimeAgent<()>>,
    settings: Option<RealtimeSessionModelSettings>,
) -> (Vec<String>, RealtimeSessionModelSettings) {
    let model = Arc::new(FakeRealtimeModel::new());
    let mut runner = RealtimeRunner::new(agent).with_model(model.clone());
    if let Some(settings) = settings {
        runner = runner
            .with_model_config(RealtimeModelConfig::new().with_initial_model_settings(settings));
    }
    let session = runner.run(()).await.expect("session connects");
    let settings = model.last_session_settings().expect("connect settings");
    let names = settings
        .tools
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    session.close().await.unwrap();
    (names, settings)
}

fn payload_tool_names(payload: &Value) -> Vec<String> {
    payload["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|tool| tool["name"].as_str().map(str::to_string))
        .collect()
}

#[tokio::test]
async fn collect_enabled_handoffs_filters_disabled() {
    let child_enabled = RealtimeAgent::<()>::new("child_enabled").into_arc();
    let disabled = RealtimeHandoff::new(RealtimeAgent::<()>::new("child_disabled").into_arc())
        .with_is_enabled_fn(|_ctx, _agent| async { false });
    let parent = RealtimeAgent::<()>::new("parent")
        .with_handoff(disabled)
        .with_handoff(child_enabled)
        .into_arc();

    let (names, _) = connect_names(parent, None).await;

    assert_eq!(names, vec!["transfer_to_child_enabled"]);
}

#[tokio::test]
async fn build_model_settings_from_agent_merges_agent_fields() {
    let helper =
        FunctionTool::<()>::new_no_args("helper", "Helper tool for testing.", |_ctx| async {
            Ok("ok")
        });
    let agent = RealtimeAgent::<()>::new("root")
        .with_prompt(Prompt::new("prompt-id"))
        .with_instructions("sys")
        .with_tool(helper)
        .with_handoff(RealtimeAgent::<()>::new("handoff-child").into_arc())
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let config = RealtimeRunConfig {
        model_settings: Some(
            RealtimeSessionModelSettings::new().with_model_name("gpt-realtime-2.1"),
        ),
        tracing_disabled: true,
        ..Default::default()
    };
    let session =
        RealtimeRunner::new(agent)
            .with_model(model.clone())
            .with_config(config)
            .with_model_config(RealtimeModelConfig::new().with_initial_model_settings(
                RealtimeSessionModelSettings::new().with_voice("verse"),
            ))
            .run(())
            .await
            .unwrap();

    let merged = model.last_session_settings().unwrap();
    assert_eq!(
        merged.prompt.as_ref().map(|p| p.id.as_str()),
        Some("prompt-id")
    );
    assert_eq!(merged.instructions.as_deref(), Some("sys"));
    let names: Vec<_> = merged.tools.unwrap().into_iter().map(|t| t.name).collect();
    assert_eq!(names, vec!["helper", "transfer_to_handoff_child"]);
    assert_eq!(merged.voice, Some(RealtimeVoice::from("verse")));
    assert_eq!(merged.model_name.as_deref(), Some("gpt-realtime-2.1"));
    // Python reports a disabled tracing config as `None`.
    assert_eq!(merged.tracing, Some(RealtimeModelTracing::Disabled));
    session.close().await.unwrap();
}

#[tokio::test]
async fn build_model_settings_does_not_reevaluate_agent_handoff_without_override() {
    let count = Arc::new(AtomicUsize::new(0));
    let agent = RealtimeAgent::<()>::new("parent")
        .with_handoff(counting_handoff(count.clone()))
        .into_arc();

    let (names, _) = connect_names(
        agent,
        Some(RealtimeSessionModelSettings::new().with_voice("verse")),
    )
    .await;

    assert_eq!(names, vec!["transfer_to_billing"]);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn sip_model_build_initial_session_payload() {
    let child = RealtimeAgent::<()>::new("child").into_arc();
    let ping = FunctionTool::<()>::new_no_args(
        "ping",
        "Ping tool used for session payload building.",
        |_ctx| async { Ok("pong") },
    );
    let agent = RealtimeAgent::<()>::new("parent")
        .with_prompt(Prompt::new("prompt-99"))
        .with_instructions("parent-system")
        .with_tool(ping)
        .with_handoff(child)
        .into_arc();
    let model_config = RealtimeModelConfig::new().with_initial_model_settings(
        RealtimeSessionModelSettings::new()
            .with_model_name("gpt-realtime-mini")
            .with_voice("verse"),
    );
    let run_config = RealtimeRunConfig {
        model_settings: Some(RealtimeSessionModelSettings::new().with_output_modalities(["text"])),
        tracing_disabled: true,
        ..Default::default()
    };
    let mut overrides = RealtimeSessionModelSettings::new().with_audio(
        RealtimeAudioConfig::new().with_input(RealtimeAudioInputConfig {
            format: Some(RealtimeAudioFormat::from(json!({"type": "audio/pcmu"}))),
            ..Default::default()
        }),
    );
    overrides.output_audio_format = Some("g711_ulaw".into());

    let payload = OpenAIRealtimeSIPModel::build_initial_session_payload(
        &agent,
        (),
        Some(&model_config),
        Some(&run_config),
        Some(&overrides),
    )
    .await
    .unwrap();

    assert_eq!(payload["model"], "gpt-realtime-mini");
    assert_eq!(payload["output_modalities"], json!(["text"]));
    assert_eq!(payload["audio"]["input"]["format"]["type"], "audio/pcmu");
    assert_eq!(payload["audio"]["output"]["format"]["type"], "audio/pcmu");
    assert_eq!(payload["audio"]["output"]["voice"], "verse");
    assert_eq!(payload["instructions"], "parent-system");
    assert_eq!(payload["prompt"]["id"], "prompt-99");
    let names = payload_tool_names(&payload);
    assert!(names.contains(&"ping".to_string()));
    assert!(names.contains(&"transfer_to_child".to_string()));
}

// Adapted: Python filters a disabled handoff passed through settings; Rust filters the agent's own.
#[tokio::test]
async fn sip_payload_filters_disabled_handoff_name_conflict() {
    let agent = RealtimeAgent::<()>::new("parent")
        .with_tool(billing_tool(true))
        .with_handoff(billing_handoff(false))
        .into_arc();

    let payload =
        OpenAIRealtimeSIPModel::build_initial_session_payload(&agent, (), None, None, None)
            .await
            .unwrap();

    let names = payload_tool_names(&payload);
    assert_eq!(
        names.iter().filter(|n| *n == "transfer_to_billing").count(),
        1
    );
}

#[tokio::test]
async fn sip_payload_filters_disabled_tool_name_conflict() {
    let agent = RealtimeAgent::<()>::new("parent")
        .with_tool(billing_tool(false))
        .with_handoff(billing_handoff(true))
        .into_arc();

    let payload =
        OpenAIRealtimeSIPModel::build_initial_session_payload(&agent, (), None, None, None)
            .await
            .unwrap();

    assert_eq!(payload_tool_names(&payload), vec!["transfer_to_billing"]);
}

#[tokio::test]
async fn sip_initial_session_payload_does_not_reevaluate_agent_handoff_without_override() {
    let count = Arc::new(AtomicUsize::new(0));
    let agent = RealtimeAgent::<()>::new("parent")
        .with_handoff(counting_handoff(count.clone()))
        .into_arc();
    let overrides = RealtimeSessionModelSettings::new().with_voice("verse");

    let payload = OpenAIRealtimeSIPModel::build_initial_session_payload(
        &agent,
        (),
        None,
        None,
        Some(&overrides),
    )
    .await
    .unwrap();

    assert!(payload_tool_names(&payload).contains(&"transfer_to_billing".to_string()));
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn call_id_session_update_includes_explicit_audio_formats() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if let Message::Text(text) = message {
                let value: Value = serde_json::from_str(text.as_str()).unwrap();
                let _ = tx.send(value);
            }
        }
        let _ = ws.close(None).await;
    });

    let mut settings = RealtimeSessionModelSettings::new();
    settings.input_audio_format = Some("g711_ulaw".into());
    settings.output_audio_format = Some("g711_ulaw".into());
    let model = OpenAIRealtimeSIPModel::new();
    let _events = model
        .connect(
            RealtimeModelConfig::new()
                .with_api_key("sk-test")
                .with_call_id("call_123")
                .with_initial_model_settings(settings)
                .with_url(format!(
                    "ws://127.0.0.1:{port}/v1/realtime?call_id=call_123"
                )),
        )
        .await
        .expect("attaches to the call");

    let update = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("session.update received")
        .unwrap();
    assert_eq!(update["type"], "session.update");
    assert_eq!(
        update["session"]["audio"]["input"]["format"]["type"],
        "audio/pcmu"
    );
    assert_eq!(
        update["session"]["audio"]["output"]["format"]["type"],
        "audio/pcmu"
    );

    model.close().await.unwrap();
    server.abort();
}
