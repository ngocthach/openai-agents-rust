//! Port of the Python SDK's `tests/voice/test_pipeline.py`, `test_pipeline_config.py` and
//! `test_workflow.py`.
//!
//! No network and no real API keys are used.
#![cfg(feature = "voice")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::span_data::SpanData;
use openai_agents_sdk::tracing::spans::{Span, Trace};
use openai_agents_sdk::tracing::{TracingProcessor, set_trace_processors};
use openai_agents_sdk::voice::testing::{FakeSTTModel, FakeVoiceWorkflow, pcm16_bytes};
use openai_agents_sdk::voice::{
    AudioByteStream, AudioFormat, AudioInput, STTModel, STTModelSettings, SingleAgentVoiceWorkflow,
    StreamedAudioInput, StreamedAudioResult, StreamedTranscriptionSession, TTSModel,
    TTSModelSettings, TextStream, VoicePipeline, VoicePipelineConfig, VoiceStreamEvent,
    VoiceWorkflowBase,
};
use openai_agents_sdk::{Agent, Result};

// ===== Tracing: a processor that records trace start/end per trace name =====

#[derive(Debug, Default)]
struct Recorder {
    traces: Mutex<Vec<(String, String)>>,
    speech_errors: Mutex<Vec<(String, String, String)>>,
}

impl Recorder {
    /// The lifecycle events recorded for traces called `name`.
    fn events_for(&self, name: &str) -> Vec<String> {
        self.traces
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, trace_name)| trace_name == name)
            .map(|(event, _)| event.clone())
            .collect()
    }

    /// The `(message, data.text)` pairs of the failed `speech` spans of one TTS model.
    fn speech_errors_for(&self, model: &str) -> Vec<(String, String)> {
        self.speech_errors
            .lock()
            .unwrap()
            .iter()
            .filter(|(span_model, _, _)| span_model == model)
            .map(|(_, message, text)| (message.clone(), text.clone()))
            .collect()
    }
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, trace: &Trace) {
        self.traces
            .lock()
            .unwrap()
            .push(("trace_start".to_string(), trace.name().to_string()));
    }

    fn on_trace_end(&self, trace: &Trace) {
        self.traces
            .lock()
            .unwrap()
            .push(("trace_end".to_string(), trace.name().to_string()));
    }

    fn on_span_start(&self, _span: &Span) {}

    fn on_span_end(&self, span: &Span) {
        let Some(error) = span.error() else {
            return;
        };
        let SpanData::Speech { model, .. } = span.data() else {
            return;
        };
        let text = error
            .data
            .as_ref()
            .and_then(|data| data.get("text"))
            .and_then(|text| text.as_str())
            .unwrap_or_default()
            .to_string();
        self.speech_errors.lock().unwrap().push((
            model.unwrap_or_default(),
            error.message.clone(),
            text,
        ));
    }
}

/// Replace the OpenAI exporter with an in-process recorder, so no test reaches the network.
fn trace_processor() -> Arc<Recorder> {
    static PROCESSOR: OnceLock<Arc<Recorder>> = OnceLock::new();
    PROCESSOR
        .get_or_init(|| {
            let processor = Arc::new(Recorder::default());
            set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
            processor
        })
        .clone()
}

/// Collapse pipeline events to labels plus the audio chunks, as Python's `extract_events` does.
async fn extract_events(result: &mut StreamedAudioResult) -> (Vec<String>, Vec<Vec<u8>>) {
    let mut labels = Vec::new();
    let mut chunks = Vec::new();
    let mut stream = std::pin::pin!(result.stream());
    while let Some(event) = stream.next().await {
        match event {
            VoiceStreamEvent::Audio(audio) => {
                chunks.push(audio.to_bytes());
                labels.push("audio".to_string());
            }
            VoiceStreamEvent::Lifecycle(event) => labels.push(event.as_str().to_string()),
            VoiceStreamEvent::Error(error) => labels.push(format!("error: {error}")),
        }
    }
    (labels, chunks)
}

/// Python's `_never_complete`: hold every character back until the turn ends.
fn never_complete() -> openai_agents_sdk::voice::TextSplitter {
    Arc::new(|text: &str| (String::new(), text.to_string()))
}

/// Python's `split_immediately`: speak the whole buffer at once.
fn split_immediately() -> openai_agents_sdk::voice::TextSplitter {
    Arc::new(|text: &str| (text.to_string(), String::new()))
}

/// Python's `discard_text`: never speak anything.
fn discard_text() -> openai_agents_sdk::voice::TextSplitter {
    Arc::new(|_text: &str| (String::new(), String::new()))
}

// ===== Test doubles =====

/// Python's `ZeroPcmTTSModel`: two zero samples per call, or per word in `split_words` mode.
#[derive(Debug, Default)]
struct ZeroPcmTts {
    split_words: bool,
    texts: Mutex<Vec<String>>,
    error: Option<String>,
    model_name: Option<String>,
}

impl ZeroPcmTts {
    fn new() -> Self {
        Self::default()
    }

    fn split_words() -> Self {
        ZeroPcmTts {
            split_words: true,
            ..Default::default()
        }
    }

    fn failing(message: &str) -> Self {
        ZeroPcmTts {
            error: Some(message.to_string()),
            ..Default::default()
        }
    }

    fn named(mut self, name: &str) -> Self {
        self.model_name = Some(name.to_string());
        self
    }

    fn texts(&self) -> Vec<String> {
        self.texts.lock().unwrap().clone()
    }
}

/// Python's `np.zeros(2, dtype=np.int16).tobytes()`.
fn zero_pcm() -> Bytes {
    pcm16_bytes(&[0, 0])
}

#[async_trait]
impl TTSModel for ZeroPcmTts {
    fn model_name(&self) -> &str {
        self.model_name.as_deref().unwrap_or("zero-pcm-tts")
    }

    async fn run(&self, text: &str, _settings: &TTSModelSettings) -> Result<AudioByteStream> {
        self.texts.lock().unwrap().push(text.to_string());
        if let Some(message) = &self.error {
            return Err(openai_agents_sdk::AgentsError::other(message.clone()));
        }
        let chunks: Vec<Bytes> = if self.split_words {
            text.split_whitespace().map(|_| zero_pcm()).collect()
        } else {
            vec![zero_pcm()]
        };
        Ok(futures::stream::iter(chunks.into_iter().map(Ok)).boxed())
    }
}

/// A TTS model that yields one byte, so a sample straddles two chunks.
#[derive(Debug)]
struct SingleBytePcmTts;

#[async_trait]
impl TTSModel for SingleBytePcmTts {
    fn model_name(&self) -> &str {
        "single-byte-tts"
    }

    async fn run(&self, _text: &str, _settings: &TTSModelSettings) -> Result<AudioByteStream> {
        Ok(futures::stream::iter(vec![Ok(Bytes::from_static(&[1]))]).boxed())
    }
}

/// A transcription session whose turns, close behaviour and gating a test controls.
#[derive(Debug, Clone, Default)]
struct ControlledSession {
    turns: Arc<Mutex<Vec<String>>>,
    closed: Arc<AtomicBool>,
    gate: Option<Arc<tokio::sync::Notify>>,
    gate_passed: Arc<AtomicBool>,
    close_started: Option<Arc<tokio::sync::Notify>>,
    release_close: Option<Arc<tokio::sync::Notify>>,
    close_error: Option<String>,
    hang_after_turns: bool,
}

impl ControlledSession {
    fn new(turns: &[&str]) -> Self {
        ControlledSession {
            turns: Arc::new(Mutex::new(
                turns.iter().rev().map(|t| (*t).to_string()).collect(),
            )),
            ..Default::default()
        }
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl StreamedTranscriptionSession for ControlledSession {
    async fn next_turn(&mut self) -> Result<Option<String>> {
        if let Some(gate) = &self.gate
            && !self.gate_passed.swap(true, Ordering::SeqCst)
        {
            gate.notified().await;
        }
        let turn = self.turns.lock().unwrap().pop();
        match turn {
            Some(turn) => Ok(Some(turn)),
            None if self.hang_after_turns => {
                std::future::pending::<()>().await;
                unreachable!()
            }
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(started) = &self.close_started {
            started.notify_waiters();
        }
        if let Some(release) = &self.release_close {
            release.notified().await;
        }
        self.closed.store(true, Ordering::SeqCst);
        match &self.close_error {
            Some(message) => Err(openai_agents_sdk::AgentsError::other(message.clone())),
            None => Ok(()),
        }
    }
}

/// An STT model handing out one pre-built [`ControlledSession`].
#[derive(Debug)]
struct ControlledSTT {
    session: ControlledSession,
}

#[async_trait]
impl STTModel for ControlledSTT {
    fn model_name(&self) -> &str {
        "controlled-stt"
    }

    async fn transcribe(
        &self,
        _input: &AudioInput,
        _settings: &STTModelSettings,
        _trace_include_sensitive_data: bool,
        _trace_include_sensitive_audio_data: bool,
    ) -> Result<String> {
        unreachable!("the controlled STT model is only used for streamed input")
    }

    async fn create_session(
        &self,
        _input: &StreamedAudioInput,
        _settings: &STTModelSettings,
        _trace_include_sensitive_data: bool,
        _trace_include_sensitive_audio_data: bool,
    ) -> Result<Box<dyn StreamedTranscriptionSession>> {
        Ok(Box::new(self.session.clone()))
    }
}

/// A workflow whose turn output, startup output and failures a test controls.
#[derive(Debug, Default)]
struct ControlledWorkflow {
    turns: Mutex<Vec<Vec<String>>>,
    start: Option<Vec<String>>,
    start_error: Option<String>,
    turn_error: Option<String>,
    partial: Option<String>,
    gate: Option<Arc<tokio::sync::Notify>>,
    buffered: Option<Arc<tokio::sync::Notify>>,
    hang_after_partial: bool,
    transcriptions: Mutex<Vec<String>>,
}

impl ControlledWorkflow {
    fn with_turns(turns: &[&[&str]]) -> Self {
        ControlledWorkflow {
            turns: Mutex::new(
                turns
                    .iter()
                    .rev()
                    .map(|turn| turn.iter().map(|t| (*t).to_string()).collect())
                    .collect(),
            ),
            ..Default::default()
        }
    }

    fn transcriptions(&self) -> Vec<String> {
        self.transcriptions.lock().unwrap().clone()
    }
}

/// One step of a [`ControlledWorkflow`] turn.
#[derive(Debug, Clone)]
enum TurnStep {
    /// Wait until the test releases the turn.
    Gate(Arc<tokio::sync::Notify>),
    /// Yield one text fragment.
    Text(String),
    /// Tell the test the partial text has been buffered.
    Notify(Arc<tokio::sync::Notify>),
    /// Never finish the turn.
    Hang,
    /// Fail the turn.
    Fail(String),
}

/// Drive a list of [`TurnStep`]s as a text stream.
fn turn_stream(steps: Vec<TurnStep>) -> TextStream<'static> {
    Box::pin(futures::stream::unfold(
        std::collections::VecDeque::from(steps),
        |mut steps| async move {
            loop {
                match steps.pop_front()? {
                    TurnStep::Gate(gate) => gate.notified().await,
                    TurnStep::Notify(notify) => notify.notify_waiters(),
                    TurnStep::Hang => std::future::pending::<()>().await,
                    TurnStep::Text(text) => return Some((Ok(text), steps)),
                    TurnStep::Fail(message) => {
                        return Some((
                            Err(openai_agents_sdk::AgentsError::other(message)),
                            std::collections::VecDeque::new(),
                        ));
                    }
                }
            }
        },
    ))
}

impl VoiceWorkflowBase for ControlledWorkflow {
    fn run(&self, transcription: String) -> TextStream<'_> {
        self.transcriptions.lock().unwrap().push(transcription);
        let mut steps = Vec::new();
        if let Some(gate) = &self.gate {
            steps.push(TurnStep::Gate(gate.clone()));
        }
        if let Some(partial) = &self.partial {
            steps.push(TurnStep::Text(partial.clone()));
            if let Some(buffered) = &self.buffered {
                steps.push(TurnStep::Notify(buffered.clone()));
            }
            if self.hang_after_partial {
                steps.push(TurnStep::Hang);
            }
        }
        if let Some(error) = &self.turn_error {
            steps.push(TurnStep::Fail(error.clone()));
        } else {
            steps.extend(
                self.turns
                    .lock()
                    .unwrap()
                    .pop()
                    .unwrap_or_default()
                    .into_iter()
                    .map(TurnStep::Text),
            );
        }
        turn_stream(steps)
    }

    fn on_start(&self) -> TextStream<'_> {
        let mut steps: Vec<TurnStep> = self
            .start
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(TurnStep::Text)
            .collect();
        if let Some(error) = &self.start_error {
            steps.push(TurnStep::Fail(error.clone()));
        }
        turn_stream(steps)
    }
}

/// A streamed input holding `count` zero buffers, Python's `StreamedAudioInputFactory`.
fn streamed_input(count: usize) -> StreamedAudioInput {
    let input = StreamedAudioInput::new();
    for _ in 0..count {
        input.add_audio(vec![0i16, 0]).unwrap();
    }
    input
}

fn buffered_config(splitter: openai_agents_sdk::voice::TextSplitter) -> VoicePipelineConfig {
    VoicePipelineConfig::default()
        .with_tracing_disabled(true)
        .with_tts_settings(
            TTSModelSettings::default()
                .with_buffer_size(1)
                .with_text_splitter(splitter),
        )
}

// ===== tests/voice/test_pipeline.py: StreamedAudioResult =====

#[tokio::test]
async fn streamed_audio_result_closes_gracefully_after_session_end() {
    trace_processor();
    let pipeline = VoicePipeline::new(Arc::new(
        FakeVoiceWorkflow::new().with_turn(vec!["Done.".to_string()]),
    ))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
    .with_tts_model(Arc::new(ZeroPcmTts::new()))
    .with_config(buffered_config(split_immediately()));

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let (labels, _) = extract_events(&mut result).await;
    assert_eq!(labels.last().unwrap(), "session_ended");
    assert!(result.is_complete());

    // The stream is finished: it never yields again and the result drops cleanly.
    let mut stream = std::pin::pin!(result.stream());
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn streamed_audio_result_odd_length_buffer_float32() {
    trace_processor();
    let pipeline = VoicePipeline::new(Arc::new(
        FakeVoiceWorkflow::new().with_turn(vec!["ok".to_string()]),
    ))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
    .with_tts_model(Arc::new(SingleBytePcmTts))
    .with_config(
        VoicePipelineConfig::default()
            .with_tracing_disabled(true)
            .with_tts_settings(
                TTSModelSettings::default()
                    .with_buffer_size(1)
                    .with_dtype(AudioFormat::Float32)
                    .with_text_splitter(split_immediately()),
            ),
    );

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let mut samples = Vec::new();
    {
        let mut stream = std::pin::pin!(result.stream());
        while let Some(event) = stream.next().await {
            if let VoiceStreamEvent::Audio(openai_agents_sdk::voice::AudioData::Float32(data)) =
                event
            {
                samples.extend(data);
            }
        }
    }
    assert_eq!(samples.len(), 1);
    assert!((samples[0] - 1.0 / 32767.0).abs() < 1e-9, "{samples:?}");
}

#[tokio::test]
async fn streamed_audio_error_respects_sensitive_data_setting() {
    use openai_agents_sdk::tracing::REDACTED_TRACE_ERROR_MESSAGE;

    let processor = trace_processor();
    // Each run uses its own TTS model name so the assertions survive the shared processor.
    for (include_sensitive, model, expected) in [
        (
            false,
            "sensitive-probe-redacted",
            REDACTED_TRACE_ERROR_MESSAGE,
        ),
        (true, "sensitive-probe-raw", "sensitive-tts-error"),
    ] {
        let pipeline = VoicePipeline::new(Arc::new(
            FakeVoiceWorkflow::new().with_turn(vec!["sensitive input".to_string()]),
        ))
        .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
        .with_tts_model(Arc::new(
            ZeroPcmTts::failing("sensitive-tts-error").named(model),
        ))
        .with_config(
            VoicePipelineConfig::default()
                .with_trace_include_sensitive_data(include_sensitive)
                .with_tts_settings(
                    TTSModelSettings::default()
                        .with_buffer_size(1)
                        .with_text_splitter(split_immediately()),
                ),
        );

        let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
        let (labels, _) = extract_events(&mut result).await;
        assert_eq!(labels.last().unwrap(), "error: sensitive-tts-error");
        drop(result);

        let recorded = processor.speech_errors_for(model);
        assert_eq!(recorded.len(), 1, "{recorded:?}");
        assert_eq!(recorded[0].0, expected);
        assert_eq!(
            recorded[0].1,
            if include_sensitive {
                "sensitive input"
            } else {
                ""
            }
        );
    }
}

#[tokio::test]
async fn streamed_audio_dispatcher_handles_stream_failure() {
    trace_processor();
    let pipeline = VoicePipeline::new(Arc::new(FakeVoiceWorkflow::new().with_turn(vec![
        "This is the first sentence. This is the second one.".to_string(),
    ])))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
    .with_tts_model(Arc::new(ZeroPcmTts::failing("tts-failure")))
    .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = tokio::time::timeout(
        Duration::from_secs(5),
        pipeline.run(AudioInput::new(vec![0i16; 2])),
    )
    .await
    .expect("the pipeline must not block")
    .unwrap();
    let (labels, _) = tokio::time::timeout(Duration::from_secs(5), extract_events(&mut result))
        .await
        .expect("the dispatcher must not block on the dead segment");

    // Exactly one terminal event: the failure, never a second `session_ended` after it.
    assert_eq!(labels.last().unwrap(), "error: tts-failure");
    assert_eq!(
        labels.iter().filter(|l| *l == "session_ended").count(),
        0,
        "{labels:?}"
    );
}

#[tokio::test]
async fn streamed_audio_result_synthesizes_short_custom_splitter_chunk() {
    trace_processor();
    let tts = Arc::new(ZeroPcmTts::new());
    let pipeline = VoicePipeline::new(Arc::new(
        FakeVoiceWorkflow::new().with_turn(vec!["ok".to_string()]),
    ))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(split_immediately()));

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let (labels, chunks) = extract_events(&mut result).await;

    assert_eq!(tts.texts(), ["ok"]);
    assert_eq!(
        labels,
        ["turn_started", "audio", "turn_ended", "session_ended"]
    );
    assert_eq!(chunks, vec![zero_pcm().to_vec()]);
}

#[tokio::test]
async fn streamed_audio_result_ignores_empty_custom_splitter_chunk() {
    trace_processor();
    let tts = Arc::new(ZeroPcmTts::new());
    let pipeline = VoicePipeline::new(Arc::new(
        FakeVoiceWorkflow::new().with_turn(vec!["ok".to_string()]),
    ))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(discard_text()));

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let (labels, chunks) = extract_events(&mut result).await;

    assert!(tts.texts().is_empty());
    assert_eq!(labels, ["turn_started", "turn_ended", "session_ended"]);
    assert!(chunks.is_empty());
}

// ===== tests/voice/test_pipeline.py: VoicePipeline =====

#[tokio::test]
async fn voicepipeline_streamed_audio_input_without_turns() {
    trace_processor();
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow::default()))
        .with_stt_model(Arc::new(ControlledSTT {
            session: ControlledSession::new(&[]),
        }))
        .with_tts_model(Arc::new(ZeroPcmTts::new()))
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(streamed_input(0)).await.unwrap();
    // The timeout bounds the failure mode under test: a stream that never terminates.
    let (labels, chunks) =
        tokio::time::timeout(Duration::from_secs(5), extract_events(&mut result))
            .await
            .expect("the session must end");
    assert_eq!(labels, ["session_ended"]);
    assert!(chunks.is_empty());
}

#[tokio::test]
async fn voicepipeline_delivers_on_start_output_during_startup() {
    trace_processor();
    // The session only yields once the greeting has been fully delivered: if the intro turn
    // were held open until the session ends, this would deadlock.
    let gate = Arc::new(tokio::sync::Notify::new());
    let session = ControlledSession {
        gate: Some(gate.clone()),
        ..ControlledSession::new(&[])
    };
    let tts = Arc::new(ZeroPcmTts::new());
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        start: Some(vec!["Hello there".to_string()]),
        ..Default::default()
    }))
    .with_stt_model(Arc::new(ControlledSTT { session }))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(never_complete()));

    let mut result = pipeline.run(streamed_input(0)).await.unwrap();
    let mut labels = Vec::new();
    let consume = async {
        let mut stream = std::pin::pin!(result.stream());
        while let Some(event) = stream.next().await {
            match event {
                VoiceStreamEvent::Lifecycle(event) => {
                    labels.push(event.as_str().to_string());
                    if event.as_str() == "turn_ended" {
                        gate.notify_waiters();
                    }
                }
                VoiceStreamEvent::Audio(_) => labels.push("audio".to_string()),
                VoiceStreamEvent::Error(error) => panic!("unexpected error: {error}"),
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(5), consume)
        .await
        .expect("the greeting must be delivered during startup");

    assert_eq!(
        labels,
        ["turn_started", "audio", "turn_ended", "session_ended"]
    );
    assert_eq!(tts.texts(), ["Hello there"]);
}

#[tokio::test]
async fn voicepipeline_on_start_output_is_its_own_turn() {
    trace_processor();
    let tts = Arc::new(ZeroPcmTts::new());
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        start: Some(vec!["Hello there".to_string()]),
        ..ControlledWorkflow::with_turns(&[&["the reply"]])
    }))
    .with_stt_model(Arc::new(ControlledSTT {
        session: ControlledSession::new(&["hello"]),
    }))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(never_complete()));

    let mut result = pipeline.run(streamed_input(1)).await.unwrap();
    let (labels, _) = tokio::time::timeout(Duration::from_secs(5), extract_events(&mut result))
        .await
        .expect("the session must end");

    assert_eq!(
        labels,
        [
            "turn_started",
            "audio",
            "turn_ended",
            "turn_started",
            "audio",
            "turn_ended",
            "session_ended",
        ]
    );
    assert_eq!(tts.texts(), ["Hello there", "the reply"]);
}

#[tokio::test]
async fn voicepipeline_failed_turn_closes_the_session_without_further_tts() {
    trace_processor();
    // A failing turn must still close the transcription session, and must not speak the text
    // it had buffered: the consumer stops at the error, so that audio is unobservable.
    let session = ControlledSession::new(&["hello"]);
    let tts = Arc::new(ZeroPcmTts::new());
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        partial: Some("partial".to_string()),
        turn_error: Some("workflow blew up".to_string()),
        ..Default::default()
    }))
    .with_stt_model(Arc::new(ControlledSTT {
        session: session.clone(),
    }))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(never_complete()));

    let mut result = pipeline.run(streamed_input(1)).await.unwrap();
    let (labels, _) = tokio::time::timeout(Duration::from_secs(5), extract_events(&mut result))
        .await
        .expect("the session must end");

    assert_eq!(labels.last().unwrap(), "error: workflow blew up");
    assert!(session.is_closed());
    assert!(tts.texts().is_empty());
}

#[tokio::test]
async fn voicepipeline_error_waits_for_the_session_close_before_cleanup() {
    trace_processor();
    // An error is terminal, but the producer still closes the transcription session before the
    // consumer sees it, so the session is never torn down mid-close.
    let close_started = Arc::new(tokio::sync::Notify::new());
    let release_close = Arc::new(tokio::sync::Notify::new());
    let session = ControlledSession {
        close_started: Some(close_started.clone()),
        release_close: Some(release_close.clone()),
        ..ControlledSession::new(&["hello"])
    };
    let tts = Arc::new(ZeroPcmTts::new());
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        partial: Some("partial".to_string()),
        turn_error: Some("workflow blew up".to_string()),
        ..Default::default()
    }))
    .with_stt_model(Arc::new(ControlledSTT {
        session: session.clone(),
    }))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(never_complete()));

    let mut result = pipeline.run(streamed_input(1)).await.unwrap();
    let started = close_started.notified();
    let mut started = std::pin::pin!(started);
    started.as_mut().enable();

    let consumer = async {
        tokio::time::timeout(Duration::from_secs(5), started)
            .await
            .expect("close must start");
        // The close is still running: no terminal event can have been delivered yet.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!session.is_closed());
        release_close.notify_waiters();
    };
    let (events, ()) = tokio::join!(extract_events(&mut result), consumer);

    assert_eq!(events.0.last().unwrap(), "error: workflow blew up");
    assert!(session.is_closed());
    assert!(tts.texts().is_empty());
}

#[tokio::test]
async fn voicepipeline_failing_close_does_not_replace_the_turn_error() {
    trace_processor();
    let session = ControlledSession {
        close_error: Some("close blew up".to_string()),
        ..ControlledSession::new(&["hello"])
    };
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        turn_error: Some("workflow blew up".to_string()),
        ..Default::default()
    }))
    .with_stt_model(Arc::new(ControlledSTT { session }))
    .with_tts_model(Arc::new(ZeroPcmTts::new()))
    .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(streamed_input(1)).await.unwrap();
    let (labels, _) = tokio::time::timeout(Duration::from_secs(5), extract_events(&mut result))
        .await
        .expect("the session must end");
    assert_eq!(labels.last().unwrap(), "error: workflow blew up");
}

#[tokio::test]
async fn voicepipeline_dropped_consumer_closes_the_session_without_further_tts() {
    trace_processor();
    // Dropping the result tears down the producer. The turn it had buffered must never be
    // sent to TTS on the way out.
    let buffered = Arc::new(tokio::sync::Notify::new());
    let session = ControlledSession {
        hang_after_turns: true,
        ..ControlledSession::new(&["hello"])
    };
    let tts = Arc::new(ZeroPcmTts::new());
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        partial: Some("partial".to_string()),
        buffered: Some(buffered.clone()),
        hang_after_partial: true,
        ..Default::default()
    }))
    .with_stt_model(Arc::new(ControlledSTT { session }))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(never_complete()));

    let result = pipeline.run(streamed_input(1)).await.unwrap();
    let wait = buffered.notified();
    let mut wait = std::pin::pin!(wait);
    wait.as_mut().enable();
    tokio::time::timeout(Duration::from_secs(5), wait)
        .await
        .expect("the workflow must buffer its partial text");
    drop(result);

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(tts.texts().is_empty());
}

#[tokio::test]
async fn voicepipeline_run_single_turn_split_words() {
    trace_processor();
    let tts = Arc::new(ZeroPcmTts::split_words());
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow::with_turns(&[&[
        "foo bar baz",
    ]])))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("first")))
    .with_tts_model(tts.clone())
    .with_config(buffered_config(split_immediately()));

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let (labels, chunks) = extract_events(&mut result).await;
    assert_eq!(
        labels,
        [
            "turn_started",
            "audio",
            "audio",
            "audio",
            "turn_ended",
            "session_ended",
        ]
    );
    assert_eq!(chunks, vec![zero_pcm().to_vec(); 3]);
}

#[tokio::test]
async fn voicepipeline_run_multi_turn_split_words() {
    trace_processor();
    let tts = Arc::new(ZeroPcmTts::split_words());
    let workflow = Arc::new(ControlledWorkflow::with_turns(&[
        &["foo bar baz"],
        &["foo2 bar2 baz2"],
    ]));
    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(Arc::new(ControlledSTT {
            session: ControlledSession::new(&["first", "second"]),
        }))
        .with_tts_model(tts.clone())
        .with_config(buffered_config(split_immediately()));

    let mut result = pipeline.run(streamed_input(6)).await.unwrap();
    let (labels, chunks) = extract_events(&mut result).await;
    assert_eq!(
        labels,
        [
            "turn_started",
            "audio",
            "audio",
            "audio",
            "turn_ended",
            "turn_started",
            "audio",
            "audio",
            "audio",
            "turn_ended",
            "session_ended",
        ]
    );
    assert_eq!(chunks.len(), 6);
    assert_eq!(chunks, vec![zero_pcm().to_vec(); 6]);
    assert_eq!(workflow.transcriptions(), ["first", "second"]);
}

#[tokio::test]
async fn voicepipeline_float32() {
    trace_processor();
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow::with_turns(&[&["out_1"]])))
        .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("first")))
        .with_tts_model(Arc::new(ZeroPcmTts::new()))
        .with_config(
            VoicePipelineConfig::default()
                .with_tracing_disabled(true)
                .with_tts_settings(
                    TTSModelSettings::default()
                        .with_buffer_size(1)
                        .with_dtype(AudioFormat::Float32)
                        .with_text_splitter(split_immediately()),
                ),
        );

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let mut labels = Vec::new();
    let mut audio = Vec::new();
    {
        let mut stream = std::pin::pin!(result.stream());
        while let Some(event) = stream.next().await {
            match event {
                VoiceStreamEvent::Audio(data) => {
                    assert_eq!(data.format(), AudioFormat::Float32);
                    audio.push(data);
                    labels.push("audio".to_string());
                }
                VoiceStreamEvent::Lifecycle(event) => labels.push(event.as_str().to_string()),
                VoiceStreamEvent::Error(error) => panic!("unexpected error: {error}"),
            }
        }
    }
    assert_eq!(
        labels,
        ["turn_started", "audio", "turn_ended", "session_ended"]
    );
    assert_eq!(
        audio[0],
        openai_agents_sdk::voice::AudioData::Float32(vec![0.0, 0.0])
    );
}

// ===== tests/voice/test_pipeline.py: tracing lifecycle =====

#[tokio::test]
async fn voicepipeline_trace_not_finished_before_single_turn_completes() {
    let processor = trace_processor();
    let name = "voice trace single turn";
    let gate = Arc::new(tokio::sync::Notify::new());
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        gate: Some(gate.clone()),
        ..ControlledWorkflow::with_turns(&[&["out_1"]])
    }))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("first")))
    .with_tts_model(Arc::new(ZeroPcmTts::new()))
    .with_config(
        VoicePipelineConfig::default()
            .with_workflow_name(name)
            .with_tts_settings(
                TTSModelSettings::default()
                    .with_buffer_size(1)
                    .with_text_splitter(split_immediately()),
            ),
    );

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let before = processor.events_for(name);
    assert!(before.contains(&"trace_start".to_string()), "{before:?}");
    assert!(!before.contains(&"trace_end".to_string()), "{before:?}");

    gate.notify_waiters();
    extract_events(&mut result).await;
    assert_eq!(processor.events_for(name).last().unwrap(), "trace_end");
}

#[tokio::test]
async fn voicepipeline_trace_finishes_after_multi_turn_processing() {
    let processor = trace_processor();
    let name = "voice trace multi turn";
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow::with_turns(&[
        &["out_1"],
        &["out_2"],
    ])))
    .with_stt_model(Arc::new(ControlledSTT {
        session: ControlledSession::new(&["first", "second"]),
    }))
    .with_tts_model(Arc::new(ZeroPcmTts::new()))
    .with_config(VoicePipelineConfig::default().with_workflow_name(name));

    let mut result = pipeline.run(streamed_input(2)).await.unwrap();
    extract_events(&mut result).await;
    assert_eq!(processor.events_for(name).last().unwrap(), "trace_end");
}

#[tokio::test]
async fn voicepipeline_multi_turn_on_start_exception_does_not_abort() {
    trace_processor();
    let pipeline = VoicePipeline::new(Arc::new(ControlledWorkflow {
        start: Some(vec!["intro".to_string()]),
        start_error: Some("boom".to_string()),
        ..ControlledWorkflow::with_turns(&[&["out_1"]])
    }))
    .with_stt_model(Arc::new(ControlledSTT {
        session: ControlledSession::new(&["first"]),
    }))
    .with_tts_model(Arc::new(ZeroPcmTts::new()))
    .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(streamed_input(1)).await.unwrap();
    let (labels, _) = tokio::time::timeout(Duration::from_secs(5), extract_events(&mut result))
        .await
        .expect("the session must end");

    assert_eq!(labels.last().unwrap(), "session_ended");
    assert!(
        !labels.iter().any(|label| label.starts_with("error")),
        "{labels:?}"
    );
}

// ===== tests/voice/test_pipeline_config.py =====

#[test]
fn voice_pipeline_config_defaults() {
    // Python's legacy-positional test; only the defaults half is representable in Rust.
    let config = VoicePipelineConfig::default();
    assert!(!config.tracing_disabled);
    assert!(config.tracing_api_key.is_none());
    assert!(config.trace_include_sensitive_data);
    assert!(config.trace_include_sensitive_audio_data);
    assert_eq!(config.workflow_name, "Voice Agent");
    assert!(config.group_id.starts_with("group_"), "{}", config.group_id);
    assert!(config.trace_metadata.is_none());
    assert_eq!(config.stt_settings, STTModelSettings::default());
    assert_eq!(
        config.tts_settings.buffer_size,
        TTSModelSettings::default().buffer_size
    );
    assert!(config.tts_settings.voice.is_none());

    let configured = VoicePipelineConfig::default()
        .with_tracing_disabled(true)
        .with_trace_include_sensitive_data(true)
        .with_trace_include_sensitive_audio_data(false)
        .with_stt_settings(STTModelSettings::default().with_language("en"))
        .with_tts_settings(
            TTSModelSettings::default().with_voice(openai_agents_sdk::voice::TTSVoice::Alloy),
        );
    assert!(configured.tracing_disabled);
    assert!(configured.trace_include_sensitive_data);
    assert!(!configured.trace_include_sensitive_audio_data);
    assert_eq!(configured.stt_settings.language.as_deref(), Some("en"));
    assert_eq!(
        configured.tts_settings.voice,
        Some(openai_agents_sdk::voice::TTSVoice::Alloy)
    );
}

// ===== tests/voice/test_workflow.py =====

/// Mirrors `tests/test_responses.py::get_function_tool`.
fn get_function_tool<C: Send + Sync + 'static>(name: &str, result: &str) -> FunctionTool<C> {
    let result = result.to_string();
    FunctionTool::new(name, "", move |_ctx, _a: serde_json::Value| {
        let result = result.clone();
        async move { Ok(result) }
    })
}

#[tokio::test]
async fn single_agent_workflow() {
    trace_processor();
    let model = Arc::new(ScriptedModel::new());
    model
        .extend(vec![
            // First turn: a tool call and a message.
            ScriptedStep::output(vec![
                function_call_full(
                    "some_function",
                    "{\"a\": \"b\"}",
                    "2",
                    Some("1".into()),
                    None,
                ),
                assistant_message("a_message"),
            ]),
            // Second turn: a text message.
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap();

    let agent = Agent::<()>::new("initial_agent")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("some_function", "tool_result"));
    let workflow = SingleAgentVoiceWorkflow::new(agent);

    let output: Vec<String> = std::pin::pin!(workflow.run("transcription_1".to_string()))
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;
    assert_eq!(output, ["a_message", "done"]);

    // The transcription, both model turns and the tool output are carried forward.
    let history = workflow.input_history();
    assert_eq!(history.len(), 5, "{history:#?}");
    assert_eq!(
        history[0].text_content().as_deref(),
        Some("transcription_1")
    );
    assert_eq!(workflow.current_agent().name, "initial_agent");

    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done_2")]))
        .unwrap();

    let output: Vec<String> = std::pin::pin!(workflow.run("transcription_2".to_string()))
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;
    assert_eq!(output, ["done_2"]);

    let history = workflow.input_history();
    assert_eq!(history.len(), 7, "{history:#?}");
    assert_eq!(
        history[5].text_content().as_deref(),
        Some("transcription_2")
    );
    assert_eq!(workflow.current_agent().name, "initial_agent");
}

#[tokio::test]
async fn single_agent_workflow_forwards_context_on_every_turn() {
    trace_processor();
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = seen.clone();
    let read_user_id = FunctionTool::new(
        "read_user_id",
        "Return the current user ID.",
        move |ctx: openai_agents_sdk::tool_context::ToolContext<String>, _a: serde_json::Value| {
            let sink = sink.clone();
            async move {
                let user_id = ctx.run_context().context().clone();
                sink.lock().unwrap().push(user_id.clone());
                Ok(user_id)
            }
        },
    );

    let model = Arc::new(ScriptedModel::new());
    model
        .extend(vec![
            ScriptedStep::output(vec![function_call_full(
                "read_user_id",
                "{}",
                "context_call_1",
                None,
                None,
            )]),
            ScriptedStep::output(vec![assistant_message("first turn done")]),
            ScriptedStep::output(vec![function_call_full(
                "read_user_id",
                "{}",
                "context_call_2",
                None,
                None,
            )]),
            ScriptedStep::output(vec![assistant_message("second turn done")]),
        ])
        .unwrap();

    let agent = Agent::<String>::new("context_agent")
        .with_model_instance(model)
        .with_tool(read_user_id);
    let workflow = SingleAgentVoiceWorkflow::with_context(agent, "user-123".to_string());

    let first: Vec<String> = std::pin::pin!(workflow.run("first transcription".to_string()))
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;
    let second: Vec<String> = std::pin::pin!(workflow.run("second transcription".to_string()))
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;

    assert_eq!(first, ["first turn done"]);
    assert_eq!(second, ["second turn done"]);
    assert_eq!(*seen.lock().unwrap(), ["user-123", "user-123"]);
}
