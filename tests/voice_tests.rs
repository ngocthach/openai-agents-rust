//! Voice pipeline tests: fake STT/TTS/workflow pipelines, the single-agent workflow, the
//! sentence splitter, and the OpenAI STT/TTS models against local mock HTTP and websocket
//! servers. No network and no real API keys are used.
#![cfg(feature = "voice")]

use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, trace,
};
use openai_agents_sdk::voice::testing::{
    FakeSTTModel, FakeTTSModel, FakeTranscriptionSession, FakeVoiceWorkflow, VoiceScriptError,
    operation, pcm16_bytes, pcm16_samples,
};
use openai_agents_sdk::voice::{
    AudioData, AudioInput, OpenAISTTModel, OpenAITTSModel, STTModel, STTModelSettings,
    SingleAgentVoiceWorkflow, StreamedAudioInput, StreamedAudioResult,
    StreamedTranscriptionSession, TTSModel, TTSModelSettings, TextStream, VoicePipeline,
    VoicePipelineConfig, VoiceStreamEvent, VoiceWebSocketOptions, VoiceWorkflowBase,
    get_sentence_based_splitter,
};
use openai_agents_sdk::{Agent, OpenAIClient};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Record traces in memory. This replaces the OpenAI exporter, so no test can reach the
/// network, while spans stay observable for the tracing assertions.
fn trace_processor() -> Arc<InMemoryTraceProcessor> {
    static PROCESSOR: OnceLock<Arc<InMemoryTraceProcessor>> = OnceLock::new();
    PROCESSOR
        .get_or_init(|| {
            let processor = Arc::new(InMemoryTraceProcessor::new());
            set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
            processor
        })
        .clone()
}

/// The spans recorded under `trace_id`, in the order they ended.
fn recorded_spans(processor: &InMemoryTraceProcessor, trace_id: &str) -> Vec<Value> {
    processor
        .items()
        .into_iter()
        .filter(|item| item["object"] == "trace.span" && item["trace_id"] == trace_id)
        .collect()
}

/// The first recorded span of this `span_data.type`.
fn span_of_type<'a>(spans: &'a [Value], span_type: &str) -> &'a Value {
    spans
        .iter()
        .find(|span| span["span_data"]["type"] == span_type)
        .unwrap_or_else(|| panic!("no {span_type} span in {spans:#?}"))
}

/// Collapse pipeline events to labels plus the audio samples, as Python's `extract_events` does.
async fn extract_events(result: &mut StreamedAudioResult) -> (Vec<String>, Vec<i16>) {
    let mut labels = Vec::new();
    let mut samples = Vec::new();
    let mut stream = std::pin::pin!(result.stream());
    while let Some(event) = stream.next().await {
        match event {
            VoiceStreamEvent::Audio(audio) => {
                samples.extend(audio.to_int16());
                labels.push("audio".to_string());
            }
            VoiceStreamEvent::Lifecycle(event) => labels.push(event.as_str().to_string()),
            VoiceStreamEvent::Error(error) => labels.push(format!("error: {error}")),
        }
    }
    (labels, samples)
}

#[tokio::test]
async fn static_audio_produces_audio_then_session_ended() {
    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_transcription("what is the weather"));
    let tts =
        Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[1, 2]), pcm16_bytes(&[3])]));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Sunny today.".to_string()]));

    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(stt.clone())
        .with_tts_model(tts.clone())
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let audio = AudioInput::new(vec![0; 10]);
    let mut result = pipeline.run(audio.clone()).await.unwrap();
    let (labels, samples) = extract_events(&mut result).await;

    assert_eq!(
        labels,
        ["turn_started", "audio", "turn_ended", "session_ended"]
    );
    assert_eq!(samples, vec![1, 2, 3]);
    assert_eq!(result.total_output_text(), "Sunny today.");
    assert!(result.is_complete());

    assert_eq!(workflow.transcriptions(), ["what is the weather"]);
    let calls = stt.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].input, audio);
    assert!(calls[0].trace_include_sensitive_data);
    assert_eq!(
        tts.calls()
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>(),
        ["Sunny today."]
    );
}

#[tokio::test]
async fn flushed_sentences_are_spoken_in_order() {
    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_transcription("hi"));
    let tts = Arc::new(
        FakeTTSModel::new()
            .with_result(vec![pcm16_bytes(&[1])])
            .with_result(vec![pcm16_bytes(&[2])]),
    );
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec![
        "Long enough to flush this one. Tail here.".to_string(),
    ]));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt)
        .with_tts_model(tts.clone())
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
    let (labels, samples) = extract_events(&mut result).await;

    assert_eq!(
        labels,
        [
            "turn_started",
            "audio",
            "audio",
            "turn_ended",
            "session_ended"
        ]
    );
    assert_eq!(samples, vec![1, 2]);
    assert_eq!(
        tts.calls()
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>(),
        ["Long enough to flush this one.", "Tail here."]
    );
}

#[tokio::test]
async fn streamed_audio_runs_a_turn_per_transcription() {
    trace_processor();
    let stt =
        Arc::new(FakeSTTModel::new().with_session(vec!["first".to_string(), "second".to_string()]));
    let tts = Arc::new(
        FakeTTSModel::new()
            .with_result(vec![pcm16_bytes(&[1])])
            .with_result(vec![pcm16_bytes(&[2])]),
    );
    let workflow = Arc::new(
        FakeVoiceWorkflow::new()
            .with_turn(vec!["One.".to_string()])
            .with_turn(vec!["Two.".to_string()]),
    );

    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let input = StreamedAudioInput::new();
    input.add_audio(vec![0; 4]).unwrap();
    input.end().unwrap();

    let mut result = pipeline.run(input).await.unwrap();
    let (labels, samples) = extract_events(&mut result).await;

    assert_eq!(
        labels,
        [
            "turn_started",
            "audio",
            "turn_ended",
            "turn_started",
            "audio",
            "turn_ended",
            "session_ended"
        ]
    );
    assert_eq!(samples, vec![1, 2]);
    assert_eq!(workflow.transcriptions(), ["first", "second"]);
}

#[tokio::test]
async fn a_failing_workflow_ends_the_stream_with_an_error() {
    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_transcription("hi"));
    let tts = Arc::new(FakeTTSModel::new());
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn_error("workflow exploded"));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
    let (labels, _) = extract_events(&mut result).await;

    assert_eq!(labels, ["error: workflow exploded"]);
}

#[tokio::test]
async fn single_agent_workflow_streams_text_deltas_and_keeps_history() {
    trace_processor();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("Hello from the agent")]);
    model.add_turn(vec![assistant_message("And again")]);
    let agent = Agent::<()>::new("Assistant").with_model_instance(model.clone());
    let workflow = SingleAgentVoiceWorkflow::new(agent);

    let text: Vec<String> = workflow
        .run("hello".to_string())
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;
    assert_eq!(text, ["Hello from the agent"]);

    // The transcription and the reply are carried into the next turn.
    assert_eq!(workflow.input_history().len(), 2);
    let text: Vec<String> = workflow
        .run("again".to_string())
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;
    assert_eq!(text, ["And again"]);
    assert_eq!(workflow.input_history().len(), 4);

    let last_call = model.last_call().unwrap();
    assert_eq!(last_call.input.len(), 3);
    assert_eq!(last_call.input[0].text_content().as_deref(), Some("hello"));
    assert_eq!(last_call.input[2].text_content().as_deref(), Some("again"));
}

#[tokio::test]
async fn single_agent_workflow_drives_a_pipeline() {
    trace_processor();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("All good.")]);
    let agent = Agent::<()>::new("Assistant").with_model_instance(model);
    let stt = Arc::new(FakeSTTModel::new().with_transcription("how are you"));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[7])]));

    let pipeline = VoicePipeline::new(Arc::new(SingleAgentVoiceWorkflow::new(agent)))
        .with_stt_model(stt)
        .with_tts_model(tts.clone())
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
    let (labels, samples) = extract_events(&mut result).await;

    assert_eq!(
        labels,
        ["turn_started", "audio", "turn_ended", "session_ended"]
    );
    assert_eq!(samples, vec![7]);
    assert_eq!(
        tts.calls()
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>(),
        ["All good."]
    );
}

#[test]
fn the_sentence_splitter_holds_back_the_last_sentence() {
    let split = get_sentence_based_splitter(20);
    assert_eq!(
        split("This sentence is long enough to flush. He "),
        (
            "This sentence is long enough to flush.".to_string(),
            "He ".to_string()
        )
    );
    assert_eq!(
        split("Too short. "),
        (String::new(), "Too short. ".to_string())
    );
    assert_eq!(split(""), (String::new(), String::new()));
}

#[tokio::test]
async fn transform_data_reshapes_emitted_audio() {
    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_transcription("hi"));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[4, 5])]));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Done.".to_string()]));
    let settings = TTSModelSettings::default().with_transform_data(Arc::new(|audio: AudioData| {
        AudioData::Int16(audio.to_int16().into_iter().map(|s| s * 2).collect())
    }));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(
            VoicePipelineConfig::default()
                .with_tracing_disabled(true)
                .with_tts_settings(settings),
        );

    let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
    let (_, samples) = extract_events(&mut result).await;
    assert_eq!(samples, vec![8, 10]);
}

// ===== OpenAI HTTP models against a local mock server =====

/// A canned HTTP response.
enum Reply {
    /// A JSON body served with 200.
    Json(Value),
    /// A JSON body served with this status.
    Status(u16, Value),
    /// Raw bytes with a content type, served with 200.
    Bytes(&'static str, Vec<u8>),
}

/// One captured request: the request line target and the raw body.
#[derive(Clone)]
struct Captured {
    path: String,
    body: Vec<u8>,
    content_type: String,
}

/// Serve `replies` in order; returns the base URL and the captured requests.
async fn mock_server(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<Captured>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let request = loop {
                let Ok(n) = socket.read(&mut tmp).await else {
                    return;
                };
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                let Some(index) = find_headers_end(&buf) else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&buf[..index]).to_string();
                let length = header_value(&headers, "content-length")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);
                if buf.len() >= index + 4 + length {
                    break Captured {
                        path: headers.split_whitespace().nth(1).unwrap().to_string(),
                        body: buf[index + 4..index + 4 + length].to_vec(),
                        content_type: header_value(&headers, "content-type").unwrap_or_default(),
                    };
                }
            };
            sink.lock().unwrap().push(request);
            let (status, content_type, body) = match replies.next().expect("unexpected request") {
                Reply::Json(value) => (
                    200,
                    "application/json".to_string(),
                    value.to_string().into_bytes(),
                ),
                Reply::Status(status, value) => (
                    status,
                    "application/json".to_string(),
                    value.to_string().into_bytes(),
                ),
                Reply::Bytes(content_type, bytes) => (200, content_type.to_string(), bytes),
            };
            let head = format!(
                "HTTP/1.1 {status} STATUS\r\ncontent-type: {content_type}\r\nx-request-id: req_voice\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
            socket.shutdown().await.ok();
        }
    });
    (format!("http://{addr}/v1"), captured)
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn header_value(headers: &str, name: &str) -> Option<String> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim().eq_ignore_ascii_case(name)).then(|| value.trim().to_string())
    })
}

#[tokio::test]
async fn openai_stt_posts_the_audio_as_multipart() {
    trace_processor();
    let (base, captured) = mock_server(vec![Reply::Json(json!({"text": "hello there"}))]).await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("gpt-4o-mini-transcribe", client);
    let settings = STTModelSettings::default()
        .with_prompt("be precise")
        .with_language("en")
        .with_temperature(0.5);

    let text = model
        .transcribe(&AudioInput::new(vec![1, 2, 3]), &settings, true, true)
        .await
        .unwrap();
    assert_eq!(text, "hello there");

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].path, "/v1/audio/transcriptions");
    assert!(requests[0].content_type.starts_with("multipart/form-data"));
    let body = String::from_utf8_lossy(&requests[0].body).to_string();
    assert!(body.contains("name=\"model\""));
    assert!(body.contains("gpt-4o-mini-transcribe"));
    assert!(body.contains("name=\"file\""));
    assert!(body.contains("filename=\"audio.wav\""));
    assert!(body.contains("audio/wav"));
    assert!(body.contains("RIFF"));
    assert!(body.contains("name=\"prompt\""));
    assert!(body.contains("be precise"));
    assert!(body.contains("name=\"language\""));
    assert!(body.contains("name=\"temperature\""));
}

#[tokio::test]
async fn openai_stt_sends_a_language_list_for_multi_language_models() {
    trace_processor();
    let (base, captured) = mock_server(vec![Reply::Json(json!({"text": "bonjour"}))]).await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("gpt-transcribe", client);
    let settings = STTModelSettings::default().with_language("fr");

    model
        .transcribe(&AudioInput::new(vec![0, 0]), &settings, false, false)
        .await
        .unwrap();

    let body = String::from_utf8_lossy(&captured.lock().unwrap()[0].body).to_string();
    assert!(body.contains("name=\"languages[]\""));
    assert!(!body.contains("name=\"language\""));
}

#[tokio::test]
async fn openai_stt_rejects_a_response_without_text() {
    trace_processor();
    let (base, _) = mock_server(vec![Reply::Json(json!({"text": 42}))]).await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("whisper-1", client);
    let error = model
        .transcribe(
            &AudioInput::new(vec![0, 0]),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("text"), "{error}");
}

#[tokio::test]
async fn openai_stt_reports_api_errors() {
    trace_processor();
    let (base, _) = mock_server(vec![Reply::Status(
        400,
        json!({"error": {"message": "audio too short"}}),
    )])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("whisper-1", client);
    let error = model
        .transcribe(
            &AudioInput::new(vec![0, 0]),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap_err();
    match error {
        openai_agents_sdk::AgentsError::ApiError {
            status,
            message,
            request_id,
            ..
        } => {
            assert_eq!(status, 400);
            assert_eq!(message, "audio too short");
            assert_eq!(request_id.as_deref(), Some("req_voice"));
        }
        other => panic!("expected an API error, got {other}"),
    }
}

#[tokio::test]
async fn openai_tts_streams_pcm_chunks() {
    trace_processor();
    let pcm: Vec<u8> = vec![1, 0, 2, 0, 3, 0];
    let (base, captured) =
        mock_server(vec![Reply::Bytes("application/octet-stream", pcm.clone())]).await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAITTSModel::new("gpt-4o-mini-tts", client);
    let settings = TTSModelSettings::default().with_speed(1.25);

    let mut stream = model.run("hello world", &settings).await.unwrap();
    let mut audio = Vec::new();
    while let Some(chunk) = stream.next().await {
        audio.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(audio, pcm);

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests[0].path, "/v1/audio/speech");
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "gpt-4o-mini-tts");
    assert_eq!(body["voice"], "ash");
    assert_eq!(body["input"], "hello world");
    assert_eq!(body["response_format"], "pcm");
    assert_eq!(body["speed"], 1.25);
    assert_eq!(body["instructions"], settings.instructions);
}

// ===== The streamed transcription session against a local mock websocket server =====

/// Serve one realtime transcription handshake, then one transcript for the streamed audio.
async fn mock_transcription_server() -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
        let mut received = Vec::new();
        ws.send(tokio_tungstenite::tungstenite::Message::text(
            json!({"type": "session.created"}).to_string(),
        ))
        .await
        .unwrap();
        while let Some(Ok(message)) = ws.next().await {
            let Ok(event) = serde_json::from_str::<Value>(message.to_text().unwrap_or_default())
            else {
                continue;
            };
            let event_type = event["type"].as_str().unwrap_or_default().to_string();
            received.push(event);
            let replies: Vec<Value> = match event_type.as_str() {
                "session.update" => vec![json!({"type": "session.updated"})],
                "input_audio_buffer.append" => vec![
                    json!({"type": "input_audio_buffer.committed", "item_id": "item_1"}),
                    json!({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "item_1",
                        "transcript": "hello from the socket",
                    }),
                ],
                "input_audio_buffer.clear" => vec![json!({"type": "input_audio_buffer.cleared"})],
                _ => vec![],
            };
            for reply in replies {
                ws.send(tokio_tungstenite::tungstenite::Message::text(
                    reply.to_string(),
                ))
                .await
                .unwrap();
            }
        }
        received
    });
    (format!("http://{addr}/v1"), server)
}

#[tokio::test]
async fn streamed_transcription_session_yields_turns_over_the_websocket() {
    trace_processor();
    let (base, server) = mock_transcription_server().await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("gpt-transcribe", client);

    let input = StreamedAudioInput::new();
    input.add_audio(vec![1, 2, 3]).unwrap();
    input.end().unwrap();

    let settings = STTModelSettings::default().with_language("en");
    let mut session = model
        .create_session(&input, &settings, false, false)
        .await
        .unwrap();

    let mut turns = Vec::new();
    while let Some(turn) = session.next_turn().await.unwrap() {
        turns.push(turn);
    }
    session.close().await.unwrap();
    assert_eq!(turns, ["hello from the socket"]);

    let received = server.await.unwrap();
    let update = received
        .iter()
        .find(|event| event["type"] == "session.update")
        .expect("session.update");
    let audio = &update["session"]["audio"]["input"];
    assert_eq!(audio["format"], json!({"type": "audio/pcm", "rate": 24000}));
    assert_eq!(audio["transcription"]["model"], "gpt-transcribe");
    assert_eq!(audio["transcription"]["languages"], json!(["en"]));
    assert_eq!(audio["turn_detection"], json!({"type": "semantic_vad"}));

    let append = received
        .iter()
        .find(|event| event["type"] == "input_audio_buffer.append")
        .expect("input_audio_buffer.append");
    assert_eq!(append["audio"], "AQACAAMA");
    assert!(
        received
            .iter()
            .any(|event| event["type"] == "input_audio_buffer.commit")
    );
}

#[tokio::test]
async fn a_streamed_audio_input_can_only_back_one_session() {
    trace_processor();
    let (base, _server) = mock_transcription_server().await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("gpt-transcribe", client);
    let input = StreamedAudioInput::new();
    input.end().unwrap();

    let mut session = model
        .create_session(&input, &STTModelSettings::default(), false, false)
        .await
        .unwrap();
    let error = model
        .create_session(&input, &STTModelSettings::default(), false, false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("already consumed"), "{error}");
    session.close().await.unwrap();
}

#[tokio::test]
async fn the_voice_pipeline_streams_turns_from_the_websocket() {
    trace_processor();
    let (base, _server) = mock_transcription_server().await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let stt = Arc::new(OpenAISTTModel::new("gpt-transcribe", client));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[9])]));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Heard you.".to_string()]));

    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let input = StreamedAudioInput::new();
    input.add_audio(vec![1, 2, 3]).unwrap();
    input.end().unwrap();

    let mut result = pipeline.run(input).await.unwrap();
    let (labels, samples) = extract_events(&mut result).await;

    assert_eq!(
        labels,
        ["turn_started", "audio", "turn_ended", "session_ended"]
    );
    assert_eq!(samples, vec![9]);
    assert_eq!(workflow.transcriptions(), ["hello from the socket"]);
}

#[tokio::test]
async fn a_failing_tts_model_ends_the_stream_with_an_error() {
    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_transcription("hi"));
    let tts = Arc::new(FakeTTSModel::new().with_error("speech service down"));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Done.".to_string()]));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
    let (labels, samples) = extract_events(&mut result).await;

    assert_eq!(labels, ["turn_started", "error: speech service down"]);
    assert!(samples.is_empty());
}

#[tokio::test]
async fn a_failing_tts_model_stops_a_streamed_session() {
    trace_processor();
    let stt =
        Arc::new(FakeSTTModel::new().with_session(vec!["first".to_string(), "second".to_string()]));
    let tts = Arc::new(FakeTTSModel::new().with_error("speech service down"));
    let workflow = Arc::new(
        FakeVoiceWorkflow::new()
            .with_turn(vec!["One.".to_string()])
            .with_turn(vec!["Two.".to_string()]),
    );

    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let input = StreamedAudioInput::new();
    input.add_audio(vec![0; 4]).unwrap();
    input.end().unwrap();

    let mut result = pipeline.run(input).await.unwrap();
    let (labels, _) = extract_events(&mut result).await;
    assert_eq!(labels, ["turn_started", "error: speech service down"]);

    // The second turn must not run: its audio could never reach the consumer.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(workflow.transcriptions(), ["first"]);
}

#[tokio::test]
async fn dropping_the_result_closes_the_transcription_websocket() {
    trace_processor();
    let (base, server) = mock_transcription_server().await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let stt = Arc::new(OpenAISTTModel::new("gpt-transcribe", client));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[9])]));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Heard you.".to_string()]));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    // The input stays open, so only dropping the result can end the session.
    let input = StreamedAudioInput::new();
    input.add_audio(vec![1, 2, 3]).unwrap();

    let mut result = pipeline.run(input).await.unwrap();
    {
        let mut events = std::pin::pin!(result.stream());
        while let Some(event) = events.next().await {
            if matches!(event, VoiceStreamEvent::Audio(_)) {
                break;
            }
        }
    }
    drop(result);

    // The server task returns once the socket is gone.
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("the websocket was left open")
        .unwrap();
}

#[tokio::test]
async fn speech_spans_are_grouped_per_turn_and_honor_the_privacy_flags() {
    let processor = trace_processor();
    for include_sensitive in [true, false] {
        let stt = Arc::new(FakeSTTModel::new().with_transcription("hi"));
        let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[1, 2])]));
        let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Spoken.".to_string()]));
        let pipeline = VoicePipeline::new(workflow)
            .with_stt_model(stt)
            .with_tts_model(tts)
            .with_config(
                VoicePipelineConfig::default()
                    .with_trace_include_sensitive_data(include_sensitive)
                    .with_trace_include_sensitive_audio_data(include_sensitive),
            );

        let voice_trace = trace("voice span test").build();
        let trace_id = voice_trace.trace_id().to_string();
        let labels = voice_trace
            .run(async {
                let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
                extract_events(&mut result).await.0
            })
            .await;
        assert_eq!(
            labels,
            ["turn_started", "audio", "turn_ended", "session_ended"]
        );

        let spans = recorded_spans(&processor, &trace_id);
        let speech = span_of_type(&spans, "speech");
        let group = span_of_type(&spans, "speech_group");
        assert_eq!(speech["parent_id"], group["id"]);
        assert_eq!(speech["span_data"]["output"]["format"], "pcm");
        assert_eq!(speech["span_data"]["model"], "fake-tts");

        let data = &speech["span_data"];
        if include_sensitive {
            assert_eq!(data["input"], "Spoken.");
            assert_eq!(group["span_data"]["input"], "Spoken.");
            assert!(data["model_config"]["instructions"].is_string());
            assert_ne!(data["output"]["data"], "");
        } else {
            assert_eq!(data["input"], "");
            assert_eq!(group["span_data"]["input"], "");
            assert!(data["model_config"]["instructions"].is_null());
            assert_eq!(data["output"]["data"], "");
        }
    }
}

#[tokio::test]
async fn session_creation_and_tts_settings_are_recorded() {
    use openai_agents_sdk::voice::TTSVoice;

    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_session(vec!["only turn".to_string()]));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[3])]));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Spoken.".to_string()]));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt.clone())
        .with_tts_model(tts.clone())
        .with_config(
            VoicePipelineConfig::default()
                .with_tracing_disabled(true)
                .with_trace_include_sensitive_data(false)
                .with_trace_include_sensitive_audio_data(true)
                .with_tts_settings(TTSModelSettings {
                    voice: Some(TTSVoice::Echo),
                    instructions: "Be brisk.".to_string(),
                    ..Default::default()
                }),
        );

    let input = StreamedAudioInput::new();
    input.add_audio(vec![0; 4]).unwrap();
    input.end().unwrap();
    let mut result = pipeline.run(input).await.unwrap();
    let (_labels, samples) = extract_events(&mut result).await;
    assert_eq!(samples, vec![3]);

    // The settings and privacy flags of `create_session` are observable.
    let session_calls = stt.session_calls();
    assert_eq!(session_calls.len(), 1);
    assert!(!session_calls[0].trace_include_sensitive_data);
    assert!(session_calls[0].trace_include_sensitive_audio_data);
    assert!(stt.calls().is_empty(), "no static transcription happened");

    // The created session is still observable although the pipeline owns it.
    let sessions = stt.created_sessions();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].close_calls(), 1);
    assert_eq!(sessions[0].remaining_turns(), 0);

    // TTS calls carry the settings the pipeline resolved, not just the text.
    let tts_calls = tts.calls();
    assert_eq!(tts_calls[0].text, "Spoken.");
    assert_eq!(tts_calls[0].settings.voice, Some(TTSVoice::Echo));
    assert_eq!(tts_calls[0].settings.instructions, "Be brisk.");
}

// ===== The script-error hierarchy and `assert_complete` =====

#[tokio::test]
async fn an_exhausted_script_fails_with_pythons_messages() {
    let stt = FakeSTTModel::new();
    let error = stt
        .transcribe(
            &AudioInput::new(vec![0; 2]),
            &Default::default(),
            true,
            true,
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Unexpected static transcription call: no scripted transcriptions remain."
    );

    let input = StreamedAudioInput::new();
    input.end().unwrap();
    let error = stt
        .create_session(&input, &Default::default(), true, true)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Unexpected streamed transcription session: no scripted sessions remain."
    );

    let tts = FakeTTSModel::new();
    let error = tts
        .run("hi", &Default::default())
        .await
        .err()
        .expect("no scripted result");
    assert_eq!(
        error.to_string(),
        "Unexpected TTS call: no scripted results remain."
    );

    let workflow = FakeVoiceWorkflow::new();
    let turn = std::pin::pin!(workflow.run("hi".to_string()))
        .next()
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(
        turn.to_string(),
        "Unexpected workflow turn: no scripted turns remain."
    );
}

#[tokio::test]
async fn assert_complete_reports_every_unconsumed_step() {
    let stt = FakeSTTModel::new()
        .with_transcription("a")
        .with_session(vec!["b".to_string()]);
    assert_eq!(
        stt.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "2 scripted STT step(s) were not consumed.".to_string(),
            remaining_steps: 2,
        })
    );

    let tts = FakeTTSModel::new().with_result(vec![pcm16_samples(&[1])]);
    assert_eq!(
        tts.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "1 scripted TTS result(s) were not consumed.".to_string(),
            remaining_steps: 1,
        })
    );

    let workflow = FakeVoiceWorkflow::new().with_turn(vec!["x".to_string()]);
    assert_eq!(
        workflow.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "1 scripted workflow turn(s) were not consumed.".to_string(),
            remaining_steps: 1,
        })
    );

    // A created session with turns left fails the model's own `assert_complete`.
    let session = FakeTranscriptionSession::new(vec!["one".to_string(), "two".to_string()]);
    let stt = FakeSTTModel::new().with_session_instance(session.clone());
    let input = StreamedAudioInput::new();
    input.end().unwrap();
    let mut created = stt
        .create_session(&input, &Default::default(), false, false)
        .await
        .unwrap();
    assert_eq!(created.next_turn().await.unwrap().as_deref(), Some("one"));
    assert_eq!(
        stt.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "1 scripted transcription turn(s) were not consumed.".to_string(),
            remaining_steps: 1,
        })
    );
    assert_eq!(session.assert_complete(), stt.assert_complete());
}

#[tokio::test]
async fn a_configured_workflow_start_is_consumed_exactly_once() {
    // An unconfigured start yields nothing and never fails, however often it is called.
    let workflow = FakeVoiceWorkflow::new();
    for _ in 0..2 {
        let fragments: Vec<_> = std::pin::pin!(workflow.on_start()).collect().await;
        assert!(fragments.is_empty());
    }
    assert_eq!(workflow.assert_complete(), Ok(()));

    let workflow = FakeVoiceWorkflow::new().with_start(vec!["Hello.".to_string()]);
    let first: Vec<String> = std::pin::pin!(workflow.on_start())
        .map(|f| f.unwrap())
        .collect()
        .await;
    assert_eq!(first, ["Hello."]);
    assert_eq!(workflow.assert_complete(), Ok(()));

    let error = std::pin::pin!(workflow.on_start())
        .next()
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Unexpected workflow startup call: no scripted startup step remains."
    );
    assert_eq!(
        VoiceScriptError::UnexpectedVoiceCall {
            message: error.to_string(),
            operation: operation::WORKFLOW_START,
        }
        .to_string(),
        error.to_string()
    );
}

#[tokio::test]
async fn a_scripted_session_close_failure_reaches_the_pipeline_and_close_is_idempotent() {
    trace_processor();
    let session = FakeTranscriptionSession::new(vec!["first".to_string()])
        .with_close_error("could not close the session");
    let stt = Arc::new(FakeSTTModel::new().with_session_instance(session.clone()));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[5])]));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Ok.".to_string()]));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt.clone())
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let input = StreamedAudioInput::new();
    input.end().unwrap();
    let mut result = pipeline.run(input).await.unwrap();
    let (labels, _) = extract_events(&mut result).await;

    assert_eq!(
        labels,
        [
            "turn_started",
            "audio",
            "turn_ended",
            "error: could not close the session"
        ]
    );
    assert_eq!(session.close_calls(), 1);
    assert!(session.is_closed());

    // A second close is a no-op: it counts, and it does not fail again.
    let mut again = session.clone();
    again.close().await.unwrap();
    assert_eq!(session.close_calls(), 2);
    assert_eq!(stt.assert_complete(), Ok(()));
}

#[tokio::test]
async fn a_closed_session_stops_yielding_its_remaining_turns() {
    let mut session = FakeTranscriptionSession::new(vec!["one".to_string(), "two".to_string()]);
    assert_eq!(session.next_turn().await.unwrap().as_deref(), Some("one"));
    session.close().await.unwrap();
    assert_eq!(session.next_turn().await.unwrap(), None);
    assert_eq!(
        session.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "1 scripted transcription turn(s) were not consumed.".to_string(),
            remaining_steps: 1,
        })
    );
}

#[tokio::test]
async fn a_failing_turn_ends_a_streamed_session() {
    trace_processor();
    let session =
        FakeTranscriptionSession::new(vec!["first".to_string()]).with_turn_error("socket died");
    let stt = Arc::new(FakeSTTModel::new().with_session_instance(session));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[5])]));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["Ok.".to_string()]));

    let pipeline = VoicePipeline::new(workflow)
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let input = StreamedAudioInput::new();
    input.end().unwrap();
    let mut result = pipeline.run(input).await.unwrap();
    let (labels, _) = extract_events(&mut result).await;
    assert_eq!(
        labels,
        ["turn_started", "audio", "turn_ended", "error: socket died"]
    );
}

// ===== Cancellation and panic bookkeeping =====

/// A workflow that panics instead of producing text.
#[derive(Debug)]
struct PanickingWorkflow;

impl VoiceWorkflowBase for PanickingWorkflow {
    fn run(&self, _transcription: String) -> TextStream<'_> {
        panic!("the workflow exploded");
    }
}

#[tokio::test]
async fn a_panicking_workflow_still_ends_the_stream() {
    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_transcription("hi"));
    let tts = Arc::new(FakeTTSModel::new());
    let pipeline = VoicePipeline::new(Arc::new(PanickingWorkflow))
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
    let (labels, samples) = extract_events(&mut result).await;
    assert_eq!(labels, ["error: The voice pipeline panicked"]);
    assert!(samples.is_empty());
    assert!(result.is_complete());
}

#[tokio::test]
async fn dropping_the_result_stops_the_pipeline() {
    trace_processor();
    let stt = Arc::new(FakeSTTModel::new().with_session(vec![
        "first".to_string(),
        "second".to_string(),
        "third".to_string(),
    ]));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![pcm16_bytes(&[1])]));
    let workflow = Arc::new(
        FakeVoiceWorkflow::new()
            .with_turn(vec!["One.".to_string()])
            .with_turn(vec!["Two.".to_string()])
            .with_turn(vec!["Three.".to_string()]),
    );

    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(stt.clone())
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let input = StreamedAudioInput::new();
    input.end().unwrap();
    let mut result = pipeline.run(input).await.unwrap();
    {
        let mut stream = std::pin::pin!(result.stream());
        assert!(stream.next().await.is_some());
    }
    drop(result);
    // The aborted producer never reaches the later turns.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(workflow.transcriptions().len() < 3);
}

// ===== Trace configuration =====

/// A workflow recording the tracing API key of the trace it runs in.
#[derive(Debug, Default)]
struct TraceKeyWorkflow {
    key: Mutex<Option<String>>,
}

impl VoiceWorkflowBase for TraceKeyWorkflow {
    fn run(&self, _transcription: String) -> TextStream<'_> {
        *self.key.lock().unwrap() = openai_agents_sdk::tracing::current_trace()
            .and_then(|trace| trace.tracing_api_key().map(str::to_string));
        futures::stream::iter(vec![Ok("Done.".to_string())]).boxed()
    }
}

#[tokio::test]
async fn the_pipeline_trace_carries_the_configured_tracing_api_key() {
    trace_processor();
    let workflow = Arc::new(TraceKeyWorkflow::default());
    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
        .with_tts_model(Arc::new(
            FakeTTSModel::new().with_result(vec![pcm16_bytes(&[1])]),
        ))
        .with_config(VoicePipelineConfig::default().with_tracing_api_key("sk-trace-test"));

    let mut result = pipeline.run(AudioInput::new(vec![0; 4])).await.unwrap();
    extract_events(&mut result).await;
    assert_eq!(
        workflow.key.lock().unwrap().as_deref(),
        Some("sk-trace-test")
    );
}

// ===== Websocket keepalive and frame-size settings =====

#[tokio::test]
async fn the_websocket_max_size_rejects_an_oversized_event() {
    trace_processor();
    let (base, _server) = mock_transcription_server().await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("gpt-transcribe", client)
        .with_websocket_options(VoiceWebSocketOptions::default().with_max_size(Some(4)));
    assert_eq!(model.websocket_options().max_size, Some(4));

    let input = StreamedAudioInput::new();
    input.end().unwrap();
    let error = model
        .create_session(&input, &STTModelSettings::default(), false, false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("STT websocket"), "{error}");
}

/// Complete the handshake, then stop reading so no pong can be sent.
async fn mock_silent_transcription_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
        ws.send(tokio_tungstenite::tungstenite::Message::text(
            json!({"type": "session.created"}).to_string(),
        ))
        .await
        .unwrap();
        while let Some(Ok(message)) = ws.next().await {
            if message
                .to_text()
                .is_ok_and(|text| text.contains("session.update"))
            {
                ws.send(tokio_tungstenite::tungstenite::Message::text(
                    json!({"type": "session.updated"}).to_string(),
                ))
                .await
                .unwrap();
                break;
            }
        }
        // Hold the socket open without reading: keepalive pings go unanswered.
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    format!("http://{addr}/v1")
}

#[tokio::test]
async fn an_unanswered_keepalive_ping_fails_the_session() {
    trace_processor();
    let base = mock_silent_transcription_server().await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model = OpenAISTTModel::new("gpt-transcribe", client).with_websocket_options(
        VoiceWebSocketOptions::default()
            .with_ping_interval(Some(0.05))
            .with_ping_timeout(Some(0.05)),
    );

    let input = StreamedAudioInput::new();
    input.add_audio(vec![1, 2, 3]).unwrap();
    let mut session = model
        .create_session(&input, &STTModelSettings::default(), false, false)
        .await
        .unwrap();
    let error = session.next_turn().await.unwrap_err();
    assert!(error.to_string().contains("keepalive pong"), "{error}");
    session.close().await.unwrap();
}
