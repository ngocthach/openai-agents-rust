//! Port of the Python SDK's `tests/voice/test_pipeline_tracing.py`.
//!
//! The OpenAI models run against local mock HTTP and websocket servers; no network and no real
//! API keys are used.
#![cfg(feature = "voice")]

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures::{SinkExt, StreamExt};
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, current_trace, custom_span, set_trace_processors,
    trace,
};
use openai_agents_sdk::voice::testing::{FakeSTTModel, FakeTTSModel, FakeVoiceWorkflow};
use openai_agents_sdk::voice::{
    AudioInput, OpenAISTTModel, OpenAITTSModel, STTModelSettings, StreamedAudioInput,
    StreamedAudioResult, TTSModelSettings, TextStream, VoicePipeline, VoicePipelineConfig,
    VoiceStreamEvent, VoiceWorkflowBase,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

// ===== Shared test plumbing =====

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

/// Every exported span of `trace_id`.
fn spans_of(processor: &InMemoryTraceProcessor, trace_id: &str) -> Vec<Value> {
    processor
        .items()
        .into_iter()
        .filter(|item| item["object"] == "trace.span" && item["trace_id"] == trace_id)
        .collect()
}

fn spans_of_type<'a>(spans: &'a [Value], span_type: &str) -> Vec<&'a Value> {
    spans
        .iter()
        .filter(|span| span["span_data"]["type"] == span_type)
        .collect()
}

/// Collapse pipeline events to labels plus the audio bytes, as Python's `extract_events` does.
async fn extract_events(result: &mut StreamedAudioResult) -> (Vec<String>, Vec<u8>) {
    let mut labels = Vec::new();
    let mut audio = Vec::new();
    let mut stream = std::pin::pin!(result.stream());
    while let Some(event) = stream.next().await {
        match event {
            VoiceStreamEvent::Audio(data) => {
                audio.extend(data.to_bytes());
                labels.push("audio".to_string());
            }
            VoiceStreamEvent::Lifecycle(event) => labels.push(event.as_str().to_string()),
            VoiceStreamEvent::Error(error) => labels.push(format!("error: {error}")),
        }
    }
    (labels, audio)
}

fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

// ===== A mock HTTP server =====

/// A canned HTTP response.
#[allow(dead_code)] // copy of the shared mock; this file only uses some variants
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
#[allow(dead_code)]
struct Captured {
    path: String,
    body: Vec<u8>,
}

impl Captured {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }

    /// The text of a multipart form field.
    fn field(&self, name: &str) -> Option<String> {
        let body = String::from_utf8_lossy(&self.body).to_string();
        let marker = format!("name=\"{name}\"\r\n\r\n");
        let start = body.find(&marker)? + marker.len();
        let end = body[start..].find("\r\n")?;
        Some(body[start..start + end].to_string())
    }
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
                let Some(index) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&buf[..index]).to_string();
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if buf.len() >= index + 4 + length {
                    break Captured {
                        path: headers.split_whitespace().nth(1).unwrap().to_string(),
                        body: buf[index + 4..index + 4 + length].to_vec(),
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

// ===== A scripted realtime-transcription websocket server =====

/// One step of the server's script.
#[derive(Clone)]
#[allow(dead_code)]
enum Step {
    /// Send this event.
    Send(Value),
    /// Send this raw text frame.
    SendText(&'static str),
    /// Wait for the next client event and require this type.
    Recv(&'static str),
    /// Send the `input_audio_buffer_commit_empty` error for the last received commit.
    CommitEmptyError,
    /// Wait until the test releases the server.
    Wait(Arc<Notify>),
    /// Tell the test the script reached this point.
    Signal(Arc<Notify>),
    /// Sleep for this many milliseconds.
    Sleep(u64),
    /// Close the socket from the server side.
    Close,
    /// Read until the client closes the socket.
    WaitClosed,
}

/// What the client's handshake looked like.
#[derive(Clone, Default)]
#[allow(dead_code)]
struct Handshake {
    target: String,
    headers: Vec<(String, String)>,
}

#[allow(dead_code)]
impl Handshake {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

#[allow(dead_code)]
struct WsServer {
    base: String,
    received: Arc<Mutex<Vec<Value>>>,
    handshake: Arc<Mutex<Handshake>>,
    task: JoinHandle<()>,
}

impl WsServer {
    fn client(&self) -> OpenAIClient {
        OpenAIClient::new("test-key").with_base_url(self.base.clone())
    }

    fn received(&self) -> Vec<Value> {
        self.received.lock().unwrap().clone()
    }

    #[allow(dead_code)]
    fn handshake(&self) -> Handshake {
        self.handshake.lock().unwrap().clone()
    }

    fn of_type(&self, event_type: &str) -> Vec<Value> {
        self.received()
            .into_iter()
            .filter(|event| event["type"] == event_type)
            .collect()
    }

    /// The `transcription` block of the `session.update` the client sent.
    fn transcription_config(&self) -> Value {
        self.of_type("session.update")[0]["session"]["audio"]["input"]["transcription"].clone()
    }

    /// Wait for the script to finish, which needs the client to have closed the socket.
    async fn finished(self) {
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .expect("the websocket was left open")
            .unwrap();
    }
}

/// The handshake events of a session, with the given event names.
fn setup_with(created: &str, updated: &str) -> Vec<Step> {
    vec![
        Step::Send(json!({"type": created})),
        Step::Recv("session.update"),
        Step::Send(json!({"type": updated})),
    ]
}

fn setup() -> Vec<Step> {
    setup_with("session.created", "session.updated")
}

fn completed(item_id: &str, transcript: &str) -> Step {
    Step::Send(json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "transcript": transcript,
    }))
}

/// The commit, clear and transcript steps a client's `end()` triggers.
fn eof_flow(transcript: &str) -> Vec<Step> {
    vec![
        Step::Recv("input_audio_buffer.append"),
        Step::Recv("input_audio_buffer.commit"),
        Step::Send(json!({"type": "input_audio_buffer.committed", "item_id": "first"})),
        Step::Recv("input_audio_buffer.clear"),
        Step::Send(json!({"type": "input_audio_buffer.cleared"})),
        completed("first", transcript),
        Step::WaitClosed,
    ]
}

async fn ws_server(steps: Vec<Step>) -> WsServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let handshake = Arc::new(Mutex::new(Handshake::default()));
    let (sink, shake) = (received.clone(), handshake.clone());
    let task = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        #[allow(clippy::result_large_err)]
        let mut ws =
            tokio_tungstenite::accept_hdr_async(socket, |req: &Request, resp: Response| {
                let headers = req
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or_default().to_string()))
                    .collect();
                *shake.lock().unwrap() = Handshake {
                    target: req.uri().to_string(),
                    headers,
                };
                Ok(resp)
            })
            .await
            .unwrap();
        let mut last_commit_id = String::new();
        for step in steps {
            match step {
                Step::Send(event) => {
                    if ws.send(Message::text(event.to_string())).await.is_err() {
                        return;
                    }
                }
                Step::SendText(text) => {
                    if ws.send(Message::text(text)).await.is_err() {
                        return;
                    }
                }
                Step::CommitEmptyError => {
                    let event = json!({
                        "type": "error",
                        "error": {
                            "code": "input_audio_buffer_commit_empty",
                            "event_id": last_commit_id,
                            "message": "Input buffer is empty",
                        },
                    });
                    if ws.send(Message::text(event.to_string())).await.is_err() {
                        return;
                    }
                }
                Step::Recv(expected) => loop {
                    let Some(Ok(message)) = ws.next().await else {
                        return;
                    };
                    let Ok(text) = message.to_text() else {
                        continue;
                    };
                    let Ok(event) = serde_json::from_str::<Value>(text) else {
                        continue;
                    };
                    if event["type"] == "input_audio_buffer.commit" {
                        last_commit_id = event["event_id"].as_str().unwrap_or_default().into();
                    }
                    sink.lock().unwrap().push(event.clone());
                    assert_eq!(event["type"], expected, "unexpected client event {event}");
                    break;
                },
                Step::Wait(notify) => notify.notified().await,
                Step::Signal(notify) => notify.notify_one(),
                Step::Sleep(ms) => tokio::time::sleep(Duration::from_millis(ms)).await,
                Step::Close => {
                    let _ = ws.close(None).await;
                }
                Step::WaitClosed => {
                    while let Some(Ok(message)) = ws.next().await {
                        if let Ok(text) = message.to_text()
                            && let Ok(event) = serde_json::from_str::<Value>(text)
                        {
                            sink.lock().unwrap().push(event);
                        }
                    }
                }
            }
        }
    });
    WsServer {
        base: format!("http://{addr}/v1"),
        received,
        handshake,
        task,
    }
}

// ===== tests/voice/test_pipeline_tracing.py =====

const PCM: [i16; 4] = [1001, -1002, 1003, -1004];
const TRANSCRIPT: &str = "synthetic transcript sentinel";
const TEXT: &str = "synthetic speech sentinel.";
const PROMPT: &str = "synthetic STT prompt sentinel";
const INSTRUCTIONS: &str = "synthetic TTS instructions sentinel";

/// Yields the speech sentinel for the transcript sentinel.
#[derive(Debug)]
struct SentinelWorkflow;

impl VoiceWorkflowBase for SentinelWorkflow {
    fn run(&self, transcription: String) -> TextStream<'_> {
        assert_eq!(transcription, TRANSCRIPT);
        futures::stream::iter(vec![Ok(TEXT.to_string())]).boxed()
    }
}

/// The mock backends of one traced pipeline run.
struct Backends {
    stt_client: OpenAIClient,
    tts_client: OpenAIClient,
    http: Arc<Mutex<Vec<Captured>>>,
    ws: Option<WsServer>,
}

/// Mock the speech API (and, for static input, the transcription API too).
async fn backends(streamed: bool) -> Backends {
    let speech = Reply::Bytes("application/octet-stream", pcm_bytes(&PCM));
    if streamed {
        let (base, http) = mock_server(vec![speech]).await;
        let mut steps = setup();
        steps.extend(eof_flow(TRANSCRIPT));
        // `eof_flow` starts with the append the client sends first.
        let ws = ws_server(steps).await;
        Backends {
            stt_client: ws.client(),
            tts_client: OpenAIClient::new("synthetic-placeholder").with_base_url(base),
            http,
            ws: Some(ws),
        }
    } else {
        let (base, http) =
            mock_server(vec![Reply::Json(json!({"text": TRANSCRIPT})), speech]).await;
        let client = OpenAIClient::new("synthetic-placeholder").with_base_url(base);
        Backends {
            stt_client: client.clone(),
            tts_client: client,
            http,
            ws: None,
        }
    }
}

fn voice_input(streamed: bool) -> openai_agents_sdk::voice::VoiceInput {
    if streamed {
        let input = StreamedAudioInput::new();
        input.add_audio(PCM.to_vec()).unwrap();
        input.end().unwrap();
        input.into()
    } else {
        AudioInput::new(PCM.to_vec()).into()
    }
}

fn traced_pipeline(backends: &Backends, config: VoicePipelineConfig) -> VoicePipeline {
    VoicePipeline::new(Arc::new(SentinelWorkflow))
        .with_stt_model(Arc::new(OpenAISTTModel::new(
            "gpt-4o-transcribe",
            backends.stt_client.clone(),
        )))
        .with_tts_model(Arc::new(OpenAITTSModel::new(
            "gpt-4o-mini-tts",
            backends.tts_client.clone(),
        )))
        .with_config(config)
}

#[tokio::test]
async fn released_privacy_controls_exported_voice_payloads() {
    let processor = trace_processor();
    let encoded = BASE64.encode(pcm_bytes(&PCM));
    for streamed in [false, true] {
        for include_audio in [false, true] {
            let backends = backends(streamed).await;
            let pipeline = traced_pipeline(
                &backends,
                VoicePipelineConfig::default()
                    .with_trace_include_sensitive_data(false)
                    .with_trace_include_sensitive_audio_data(include_audio),
            );
            let caller = trace("voice privacy").build();
            let trace_id = caller.trace_id().to_string();
            let (labels, audio) = caller
                .run(async {
                    let mut result = pipeline.run(voice_input(streamed)).await.unwrap();
                    let events = extract_events(&mut result).await;
                    assert_eq!(result.total_output_text(), TEXT);
                    events
                })
                .await;
            assert_eq!(labels.last().unwrap(), "session_ended", "{labels:?}");
            assert_eq!(audio, pcm_bytes(&PCM));
            let speech_request = backends.http.lock().unwrap().last().unwrap().json();
            assert_eq!(speech_request["input"], TEXT);

            let spans = spans_of(&processor, &trace_id);
            let voice: Vec<Value> = spans
                .iter()
                .filter(|span| {
                    matches!(
                        span["span_data"]["type"].as_str(),
                        Some("transcription" | "speech")
                    )
                })
                .cloned()
                .collect();
            for kind in ["transcription", "speech"] {
                let payload = serde_json::to_string(&spans_of_type(&voice, kind)).unwrap();
                assert!(
                    !spans_of_type(&voice, kind).is_empty(),
                    "{kind} span missing"
                );
                assert_eq!(
                    payload.contains(&encoded),
                    include_audio,
                    "{kind} {streamed}"
                );
                assert!(!payload.contains(TRANSCRIPT), "{kind}");
                assert!(!payload.contains(TEXT), "{kind}");
            }
            if let Some(ws) = backends.ws {
                ws.finished().await;
            }
        }
    }
}

#[tokio::test]
async fn voice_tracing_controls_under_caller_span() {
    let processor = trace_processor();
    let encoded = BASE64.encode(pcm_bytes(&PCM));
    for streamed in [false, true] {
        for (sensitive, audio_flag, disabled) in [
            (true, true, false),
            (false, true, false),
            (true, false, false),
            (false, false, false),
            (true, true, true),
        ] {
            let backends = backends(streamed).await;
            let pipeline = traced_pipeline(
                &backends,
                VoicePipelineConfig::default()
                    .with_trace_include_sensitive_data(sensitive)
                    .with_trace_include_sensitive_audio_data(audio_flag)
                    .with_tracing_disabled(disabled)
                    .with_stt_settings(STTModelSettings::default().with_prompt(PROMPT))
                    .with_tts_settings(
                        TTSModelSettings::default()
                            .with_instructions(INSTRUCTIONS)
                            .with_buffer_size(1),
                    ),
            );
            let caller = trace("voice caller").build();
            let trace_id = caller.trace_id().to_string();
            let parent_slot = Mutex::new(String::new());
            let (labels, audio) = caller
                .run(async {
                    let parent = custom_span("parent", json!({}));
                    *parent_slot.lock().unwrap() = parent.span_id().to_string();
                    parent
                        .run(async {
                            let mut result = pipeline.run(voice_input(streamed)).await.unwrap();
                            let events = extract_events(&mut result).await;
                            assert_eq!(result.total_output_text(), TEXT);
                            // The caller's trace and span are still current.
                            assert_eq!(current_trace().unwrap().trace_id(), trace_id);
                            custom_span("sibling", json!({})).run(async {}).await;
                            events
                        })
                        .await
                })
                .await;
            assert_eq!(labels.last().unwrap(), "session_ended", "{labels:?}");
            assert_eq!(audio, pcm_bytes(&PCM));
            let speech_request = backends.http.lock().unwrap().last().unwrap().json();
            assert_eq!(speech_request["input"], TEXT);
            assert_eq!(speech_request["instructions"], INSTRUCTIONS);
            if let Some(ws) = &backends.ws {
                assert_eq!(ws.transcription_config()["prompt"], PROMPT);
            } else {
                let transcription = backends.http.lock().unwrap()[0].clone();
                assert_eq!(transcription.field("prompt").as_deref(), Some(PROMPT));
            }

            let spans = spans_of(&processor, &trace_id);
            let sibling = spans
                .iter()
                .find(|span| span["span_data"]["name"] == "sibling")
                .expect("sibling span");
            let parent_id = parent_slot.lock().unwrap().clone();
            assert_eq!(sibling["parent_id"], json!(parent_id));
            let voice: Vec<Value> = spans
                .iter()
                .filter(|span| span["span_data"]["type"] != "custom")
                .cloned()
                .collect();
            if disabled {
                assert!(voice.is_empty(), "{voice:#?}");
                let payload = serde_json::to_string(&spans).unwrap();
                for value in [TRANSCRIPT, TEXT, encoded.as_str()] {
                    assert!(!payload.contains(value), "{value}");
                }
            } else {
                let kinds: std::collections::BTreeSet<&str> = voice
                    .iter()
                    .filter_map(|span| span["span_data"]["type"].as_str())
                    .collect();
                assert_eq!(
                    kinds,
                    ["speech", "speech_group", "transcription"]
                        .into_iter()
                        .collect()
                );
                let payload = serde_json::to_string(&voice).unwrap();
                for sentinel in [TRANSCRIPT, TEXT, PROMPT, INSTRUCTIONS] {
                    assert_eq!(payload.contains(sentinel), sensitive, "{sentinel}");
                }
                assert_eq!(payload.contains(&encoded), audio_flag);
            }
            if let Some(ws) = backends.ws {
                ws.finished().await;
            }
        }
    }
}

#[tokio::test]
async fn disabled_voice_pipeline_preserves_caller_lifecycle() {
    let processor = trace_processor();
    for streamed in [false, true] {
        for outcome in ["success", "error"] {
            let stt = if streamed {
                FakeSTTModel::new().with_session(vec!["hello".to_string()])
            } else {
                FakeSTTModel::new().with_transcription("hello")
            };
            let workflow = if outcome == "error" {
                FakeVoiceWorkflow::new().with_turn_error("synthetic workflow failure")
            } else {
                FakeVoiceWorkflow::new().with_turn(vec!["Done.".to_string()])
            };
            let tts = FakeTTSModel::new().with_result(vec![pcm_bytes(&[1, 2]).into()]);
            let pipeline = VoicePipeline::new(Arc::new(workflow))
                .with_stt_model(Arc::new(stt))
                .with_tts_model(Arc::new(tts))
                .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

            let caller = trace("disabled caller").build();
            let trace_id = caller.trace_id().to_string();
            let parent_slot = Mutex::new(String::new());
            let labels = caller
                .run(async {
                    let parent = custom_span("parent", json!({}));
                    *parent_slot.lock().unwrap() = parent.span_id().to_string();
                    parent
                        .run(async {
                            let mut result = pipeline.run(voice_input(streamed)).await.unwrap();
                            custom_span("concurrent sibling", json!({}))
                                .run(async {})
                                .await;
                            let (labels, _) = extract_events(&mut result).await;
                            // The pipeline neither ended nor replaced the caller's trace.
                            assert_eq!(current_trace().unwrap().trace_id(), trace_id);
                            custom_span("after sibling", json!({})).run(async {}).await;
                            labels
                        })
                        .await
                })
                .await;
            if outcome == "error" {
                assert_eq!(labels.last().unwrap(), "error: synthetic workflow failure");
            } else {
                assert_eq!(labels.last().unwrap(), "session_ended");
            }

            let spans = spans_of(&processor, &trace_id);
            let names: std::collections::BTreeSet<&str> = spans
                .iter()
                .filter_map(|span| span["span_data"]["name"].as_str())
                .collect();
            assert_eq!(
                names,
                ["after sibling", "concurrent sibling", "parent"]
                    .into_iter()
                    .collect(),
                "{outcome} {streamed}"
            );
            assert!(spans_of_type(&spans, "custom").len() == 3);
            assert_eq!(spans.len(), 3, "no voice span may be recorded: {spans:#?}");
            assert!(spans.iter().all(|span| span["trace_id"] == json!(trace_id)));
            // The caller's parent span never received the workflow's failure.
            let parent_id = parent_slot.lock().unwrap().clone();
            let parent_span = spans.iter().find(|s| s["id"] == json!(parent_id)).unwrap();
            assert!(parent_span["error"].is_null(), "{parent_span}");
        }
    }
}
