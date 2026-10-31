//! Port of the Python SDK's `tests/voice/test_input.py`, `test_utils.py`,
//! `test_tts_voice_types.py` and `test_testing.py`.
//!
//! No network and no real API keys are used.
#![cfg(feature = "voice")]

use std::sync::Arc;
use std::sync::OnceLock;

use futures::StreamExt;
use openai_agents_sdk::tracing::{InMemoryTraceProcessor, TracingProcessor, set_trace_processors};
use openai_agents_sdk::voice::testing::{
    FakeSTTModel, FakeTTSModel, FakeTranscriptionSession, FakeVoiceWorkflow, VoiceScriptError,
    operation, pcm16_samples,
};
use openai_agents_sdk::voice::{
    AudioInput, DEFAULT_SAMPLE_RATE, STTModel, STTModelSettings, StreamedAudioInput,
    StreamedTranscriptionSession, TTSModel, TTSModelSettings, TTSVoice, VoicePipeline,
    VoicePipelineConfig, VoiceStreamEvent, VoiceWorkflowBase, f32_to_i16,
};
use serde_json::json;

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

// ===== WAV decoding, the Rust stand-in for Python's `wave` module =====

/// The fields Python's `wave.Wave_read` exposes: channels, sample width, frame rate and the
/// raw frame bytes.
struct Wav {
    channels: u16,
    sample_width: u16,
    frame_rate: u32,
    frames: Vec<u8>,
}

impl Wav {
    fn n_frames(&self) -> usize {
        self.frames.len() / usize::from(self.sample_width * self.channels)
    }
}

/// Parse the canonical 44-byte WAV file the voice input encoder writes.
fn read_wav(bytes: &[u8]) -> Wav {
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[12..16], b"fmt ");
    let channels = u16::from_le_bytes(bytes[22..24].try_into().unwrap());
    let frame_rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let bits = u16::from_le_bytes(bytes[34..36].try_into().unwrap());
    assert_eq!(&bytes[36..40], b"data");
    let data_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
    Wav {
        channels,
        sample_width: bits / 8,
        frame_rate,
        frames: bytes[44..44 + data_len].to_vec(),
    }
}

/// The frames decoded as little-endian 16-bit samples.
fn wav_i16(wav: &Wav) -> Vec<i16> {
    wav.frames
        .chunks_exact(2)
        .map(|f| i16::from_le_bytes([f[0], f[1]]))
        .collect()
}

/// The frames decoded as little-endian 32-bit samples.
fn wav_i32(wav: &Wav) -> Vec<i32> {
    wav.frames
        .chunks_exact(4)
        .map(|f| i32::from_le_bytes([f[0], f[1], f[2], f[3]]))
        .collect()
}

/// A 440 Hz sine of `len` float samples, Python's test buffer.
fn sine_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / (len - 1) as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect()
}

// ===== tests/voice/test_input.py =====

#[test]
fn buffer_to_audio_file_int16() {
    let buffer: Vec<i16> = sine_f32(DEFAULT_SAMPLE_RATE as usize)
        .iter()
        .map(|s| (s * 32767.0) as i16)
        .collect();
    let input = AudioInput::new(buffer.clone());
    let (filename, bytes, content_type) = input.to_audio_file().unwrap();

    assert_eq!(filename, "audio.wav");
    assert_eq!(content_type, "audio/wav");
    let wav = read_wav(&bytes);
    assert_eq!(wav.channels, 1);
    assert_eq!(wav.sample_width, 2);
    assert_eq!(wav.frame_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(wav.n_frames(), buffer.len());
}

#[test]
fn buffer_to_audio_file_float32() {
    let buffer = sine_f32(DEFAULT_SAMPLE_RATE as usize);
    let input = AudioInput::from_f32(&buffer);
    let (filename, bytes, content_type) = input.to_audio_file().unwrap();

    assert_eq!(filename, "audio.wav");
    assert_eq!(content_type, "audio/wav");
    let wav = read_wav(&bytes);
    assert_eq!(wav.channels, 1);
    assert_eq!(wav.sample_width, 2);
    assert_eq!(wav.frame_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(wav.n_frames(), buffer.len());
}

#[test]
fn buffer_to_audio_file_honors_sample_width() {
    let int16 = vec![-1000i16, 0, 1000, 2000];
    let from_float = f32_to_i16(&[-1000.0, 0.0, 1000.0, 2000.0]);
    for buffer in [int16, from_float] {
        for sample_width in [1u16, 2, 3, 4] {
            let bytes = AudioInput::new(buffer.clone())
                .with_sample_width(sample_width)
                .to_wav_bytes()
                .unwrap();
            let wav = read_wav(&bytes);
            assert_eq!(wav.sample_width, sample_width);
            assert_eq!(wav.n_frames(), buffer.len());
            assert_eq!(wav.frames.len(), buffer.len() * usize::from(sample_width));
        }
    }
}

#[test]
fn buffer_to_audio_file_preserves_int16_amplitude_across_sample_widths() {
    let buffer = vec![-32768i16, -1, 0, 1, 32767];

    let eight = AudioInput::new(buffer.clone())
        .with_sample_width(1)
        .to_wav_bytes()
        .unwrap();
    assert_eq!(read_wav(&eight).frames, vec![0u8, 127, 128, 128, 255]);

    let thirty_two = AudioInput::new(buffer.clone())
        .with_sample_width(4)
        .to_wav_bytes()
        .unwrap();
    let decoded = wav_i32(&read_wav(&thirty_two));
    let expected: Vec<i32> = buffer.iter().map(|s| i32::from(*s) << 16).collect();
    assert_eq!(decoded, expected);
}

#[test]
fn buffer_to_audio_file_keeps_default_float32_quantization() {
    let buffer = [-0.606_067_06f32, -0.5, 0.0, 0.5, 0.896_542_25];
    let expected: Vec<i16> = buffer
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    let bytes = AudioInput::from_f32(&buffer).to_wav_bytes().unwrap();
    assert_eq!(wav_i16(&read_wav(&bytes)), expected);
}

#[test]
fn buffer_to_audio_file_rejects_unsupported_sample_width() {
    let error = AudioInput::new(vec![0i16; 4])
        .with_sample_width(5)
        .to_audio_file()
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Sample width must be between 1 and 4 bytes"),
        "{error}"
    );
}

#[test]
fn audio_input_validates_multichannel_frame_alignment() {
    let complete = AudioInput::new(vec![1i16, 2, 3, 4]).with_channels(2);
    let (_, bytes, _) = complete.to_audio_file().unwrap();
    let wav = read_wav(&bytes);
    assert_eq!(wav.channels, 2);
    assert_eq!(wav.n_frames(), 2);

    let error = AudioInput::new(vec![1i16, 2, 3])
        .with_channels(2)
        .to_audio_file()
        .unwrap_err();
    assert!(
        error.to_string().contains("complete channel frames"),
        "{error}"
    );
}

#[test]
fn audio_input_rejects_non_positive_channels() {
    // Python also parametrizes -1; a negative channel count cannot be built in Rust (`u16`).
    let error = AudioInput::new(vec![0i16; 4])
        .with_channels(0)
        .to_audio_file()
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Channels must be greater than zero"),
        "{error}"
    );
}

#[test]
fn audio_input_rejects_non_positive_frame_rate() {
    // Python also parametrizes -8000; a negative frame rate cannot be built in Rust (`u32`).
    let error = AudioInput::new(vec![0i16; 4])
        .with_frame_rate(0)
        .to_audio_file()
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Frame rate must be greater than zero"),
        "{error}"
    );
}

#[test]
fn audio_input_default_params() {
    let buffer = sine_f32(DEFAULT_SAMPLE_RATE as usize);
    let input = AudioInput::from_f32(&buffer);

    assert_eq!(input.frame_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(input.sample_width, 2);
    assert_eq!(input.channels, 1);
    assert_eq!(input.buffer, f32_to_i16(&buffer));
}

#[test]
fn audio_input_custom_params() {
    let buffer = sine_f32(48000);
    let input = AudioInput::from_f32(&buffer)
        .with_frame_rate(48000)
        .with_sample_width(4)
        .with_channels(2);

    assert_eq!(input.frame_rate, 48000);
    assert_eq!(input.sample_width, 4);
    assert_eq!(input.channels, 2);
    assert_eq!(input.buffer, f32_to_i16(&buffer));
}

#[test]
fn audio_input_to_audio_file() {
    let buffer = sine_f32(DEFAULT_SAMPLE_RATE as usize);
    let input = AudioInput::from_f32(&buffer);
    let (filename, bytes, content_type) = input.to_audio_file().unwrap();

    assert_eq!(filename, "audio.wav");
    assert_eq!(content_type, "audio/wav");
    let wav = read_wav(&bytes);
    assert_eq!(wav.channels, 1);
    assert_eq!(wav.sample_width, 2);
    assert_eq!(wav.frame_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(wav.n_frames(), buffer.len());
}

#[test]
fn audio_input_to_base64_does_not_mutate_the_buffer() {
    // Python's regression: `to_base64()` rebound `self.buffer` to int16, corrupting a
    // caller-held float32 array. Rust stores 16-bit PCM, so only the invariant is portable:
    // encoding must not touch the buffer and must be repeatable.
    let buffer = sine_f32(100);
    let input = AudioInput::from_f32(&buffer);
    let original = input.buffer.clone();

    let first = input.to_base64();
    assert_eq!(input.buffer, original);
    assert_eq!(input.to_base64(), first);
}

// ===== tests/voice/test_tts_voice_types.py =====

#[test]
fn tts_voice_type_includes_current_openai_builtin_voices() {
    let builtin: Vec<&str> = [
        TTSVoice::Alloy,
        TTSVoice::Ash,
        TTSVoice::Ballad,
        TTSVoice::Coral,
        TTSVoice::Echo,
        TTSVoice::Fable,
        TTSVoice::Onyx,
        TTSVoice::Nova,
        TTSVoice::Sage,
        TTSVoice::Shimmer,
        TTSVoice::Verse,
        TTSVoice::Marin,
        TTSVoice::Cedar,
    ]
    .iter()
    .map(|voice| voice.builtin_name().expect("builtin"))
    .collect();
    for voice in ["ballad", "verse", "marin", "cedar"] {
        assert!(
            builtin.contains(&voice),
            "{voice} is missing from {builtin:?}"
        );
    }
}

#[test]
fn tts_voice_type_accepts_custom_voice_ids() {
    let custom = TTSVoice::custom("voice_1234");
    let settings = TTSModelSettings::default().with_voice(custom.clone());

    assert!(custom.builtin_name().is_none());
    assert_eq!(custom.to_json(), json!({"id": "voice_1234"}));
    assert_eq!(settings.voice, Some(TTSVoice::Custom("voice_1234".into())));
}

// ===== tests/voice/test_testing.py =====

fn static_pipeline(
    stt: Arc<FakeSTTModel>,
    tts: Arc<FakeTTSModel>,
    workflow: Arc<FakeVoiceWorkflow>,
    buffer_size: usize,
) -> VoicePipeline {
    VoicePipeline::new(workflow)
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(
            VoicePipelineConfig::default()
                .with_tracing_disabled(true)
                .with_tts_settings(TTSModelSettings::default().with_buffer_size(buffer_size)),
        )
}

#[tokio::test]
async fn scripted_voice_components_run_static_pipeline() {
    trace_processor();
    let audio = pcm16_samples(&[0, 100, -100, 0]);
    let stt = Arc::new(FakeSTTModel::new().with_transcription("hello"));
    let workflow = Arc::new(FakeVoiceWorkflow::new().with_turn(vec!["hi.".to_string()]));
    let tts = Arc::new(FakeTTSModel::new().with_result(vec![audio.clone()]));
    let pipeline = static_pipeline(stt.clone(), tts.clone(), workflow.clone(), 1);

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let mut lifecycle = Vec::new();
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    {
        let mut stream = std::pin::pin!(result.stream());
        while let Some(event) = stream.next().await {
            match event {
                VoiceStreamEvent::Lifecycle(event) => lifecycle.push(event.as_str().to_string()),
                VoiceStreamEvent::Audio(data) => chunks.push(data.to_bytes()),
                VoiceStreamEvent::Error(error) => panic!("unexpected error: {error}"),
            }
        }
    }

    assert_eq!(lifecycle, ["turn_started", "turn_ended", "session_ended"]);
    assert_eq!(chunks, vec![audio.to_vec()]);
    assert_eq!(workflow.transcriptions(), ["hello"]);
    assert_eq!(
        tts.calls()
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>(),
        ["hi."]
    );
    assert_eq!(stt.assert_complete(), Ok(()));
    assert_eq!(workflow.assert_complete(), Ok(()));
    assert_eq!(tts.assert_complete(), Ok(()));
}

#[tokio::test]
async fn scripted_stt_treats_a_single_transcription_as_one_result() {
    let stt = FakeSTTModel::new().with_transcription("hello");

    let transcription = stt
        .transcribe(
            &AudioInput::new(vec![0i16; 2]),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap();

    assert_eq!(transcription, "hello");
    assert_eq!(stt.assert_complete(), Ok(()));
}

#[tokio::test]
async fn scripted_stt_creates_closable_streamed_session() {
    let session = FakeTranscriptionSession::new(vec!["first".to_string(), "second".to_string()]);
    let stt = FakeSTTModel::new().with_session_instance(session.clone());

    let mut created = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap();
    let mut turns = Vec::new();
    while let Some(turn) = created.next_turn().await.unwrap() {
        turns.push(turn);
    }
    created.close().await.unwrap();
    created.close().await.unwrap();

    assert_eq!(turns, ["first", "second"]);
    assert!(session.is_closed());
    assert_eq!(session.close_calls(), 2);
    assert_eq!(stt.assert_complete(), Ok(()));
}

#[tokio::test]
async fn scripted_stt_yields_one_turn_per_configured_transcript() {
    let stt = FakeSTTModel::new().with_session(vec!["hello".to_string()]);

    let mut created = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap();
    let mut turns = Vec::new();
    while let Some(turn) = created.next_turn().await.unwrap() {
        turns.push(turn);
    }

    assert_eq!(turns, ["hello"]);
    assert_eq!(stt.assert_complete(), Ok(()));
}

#[tokio::test]
async fn scripted_transcription_session_yields_its_configured_turn() {
    let mut session = FakeTranscriptionSession::new(vec!["hello".to_string()]);
    let mut turns = Vec::new();
    while let Some(turn) = session.next_turn().await.unwrap() {
        turns.push(turn);
    }
    assert_eq!(turns, ["hello"]);
    assert_eq!(session.assert_complete(), Ok(()));
}

#[tokio::test]
async fn scripted_transcription_session_does_not_start_after_close() {
    let mut session = FakeTranscriptionSession::new(vec!["unconsumed".to_string()]);
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
async fn scripted_voice_components_surface_configured_errors() {
    let stt = FakeSTTModel::new().with_transcription_error("stt failed");

    let error = stt
        .transcribe(
            &AudioInput::new(vec![0i16; 2]),
            &STTModelSettings::default(),
            true,
            true,
        )
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "stt failed");
}

#[tokio::test]
async fn scripted_voice_components_snapshot_recorded_settings() {
    let stt_settings = STTModelSettings::default()
        .with_language("en")
        .with_turn_detection(json!({"type": "server_vad", "threshold": 0.5}));
    let session_settings = STTModelSettings::default()
        .with_language("ja")
        .with_turn_detection(json!({"type": "semantic_vad", "eagerness": "auto"}));
    let tts_settings = TTSModelSettings::default()
        .with_voice(TTSVoice::Alloy)
        .with_buffer_size(20);
    let stt = FakeSTTModel::new()
        .with_transcription("hello")
        .with_session(vec!["こんにちは".to_string()]);
    let tts = FakeTTSModel::new().with_result(Vec::new());

    stt.transcribe(&AudioInput::new(vec![0i16; 2]), &stt_settings, false, false)
        .await
        .unwrap();
    stt.create_session(&StreamedAudioInput::new(), &session_settings, false, false)
        .await
        .unwrap();
    let mut audio = tts.run("hello", &tts_settings).await.unwrap();
    while audio.next().await.is_some() {}

    assert_eq!(stt.calls()[0].settings.language.as_deref(), Some("en"));
    assert_eq!(
        stt.calls()[0].settings.turn_detection,
        Some(json!({"type": "server_vad", "threshold": 0.5}))
    );
    assert_eq!(
        stt.session_calls()[0].settings.language.as_deref(),
        Some("ja")
    );
    assert_eq!(
        stt.session_calls()[0].settings.turn_detection,
        Some(json!({"type": "semantic_vad", "eagerness": "auto"}))
    );
    assert_eq!(tts.calls()[0].settings.voice, Some(TTSVoice::Alloy));
    assert_eq!(tts.calls()[0].settings.buffer_size, 20);
}

#[tokio::test]
async fn scripted_voice_components_expose_detached_histories() {
    let audio_input = AudioInput::new(vec![1i16, 2]);
    let session = FakeTranscriptionSession::default();
    let stt = FakeSTTModel::new()
        .with_transcription("hello")
        .with_session_instance(session.clone());
    let tts = FakeTTSModel::new().with_result(Vec::new());
    let workflow = FakeVoiceWorkflow::new().with_turn(Vec::new());

    stt.transcribe(
        &audio_input,
        &STTModelSettings::default().with_language("en"),
        false,
        false,
    )
    .await
    .unwrap();
    let mut created = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default().with_language("ja"),
            false,
            false,
        )
        .await
        .unwrap();
    let mut audio = tts
        .run(
            "hello",
            &TTSModelSettings::default().with_voice(TTSVoice::Alloy),
        )
        .await
        .unwrap();
    while audio.next().await.is_some() {}
    let mut fragments = std::pin::pin!(workflow.run("transcript".to_string()));
    while fragments.next().await.is_some() {}

    // The accessors hand out detached copies: mutating them cannot reach the model.
    let mut stt_calls = stt.calls();
    stt_calls[0].input.buffer[1] = 9;
    stt_calls[0].settings.language = Some("changed".to_string());
    let mut session_calls = stt.session_calls();
    session_calls[0].settings.language = Some("changed".to_string());
    let mut tts_calls = tts.calls();
    tts_calls[0].settings.voice = Some(TTSVoice::Nova);

    assert_eq!(stt.calls()[0].input.buffer, vec![1i16, 2]);
    assert_eq!(stt.calls()[0].settings.language.as_deref(), Some("en"));
    assert_eq!(
        stt.session_calls()[0].settings.language.as_deref(),
        Some("ja")
    );
    assert_eq!(tts.calls()[0].settings.voice, Some(TTSVoice::Alloy));
    assert_eq!(workflow.transcriptions(), ["transcript"]);

    // The created session is the configured one: both halves share the same state.
    created.close().await.unwrap();
    assert_eq!(stt.created_sessions().len(), 1);
    assert!(session.is_closed());
}

#[tokio::test]
async fn scripted_voice_errors_identify_exhausted_operations() {
    let stt = FakeSTTModel::new();
    let workflow = FakeVoiceWorkflow::new();
    let tts = FakeTTSModel::new();

    let static_error = stt
        .transcribe(
            &AudioInput::new(vec![0i16; 1]),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap_err();
    let session_error = stt
        .create_session(
            &StreamedAudioInput::new(),
            &STTModelSettings::default(),
            false,
            false,
        )
        .await
        .unwrap_err();
    let workflow_error = std::pin::pin!(workflow.run("hello".to_string()))
        .next()
        .await
        .unwrap()
        .unwrap_err();
    let tts_error = tts
        .run("hello", &TTSModelSettings::default())
        .await
        .err()
        .expect("no scripted result");

    // The operation ids Python exposes on `UnexpectedVoiceCall.operation`.
    assert_eq!(operation::STATIC_TRANSCRIPTION, "static_transcription");
    assert_eq!(operation::STREAMED_SESSION, "streamed_session");
    assert_eq!(operation::WORKFLOW_TURN, "workflow_turn");
    assert_eq!(operation::TTS, "tts");
    for (error, message, op) in [
        (
            static_error,
            "Unexpected static transcription call: no scripted transcriptions remain.",
            operation::STATIC_TRANSCRIPTION,
        ),
        (
            session_error,
            "Unexpected streamed transcription session: no scripted sessions remain.",
            operation::STREAMED_SESSION,
        ),
        (
            workflow_error,
            "Unexpected workflow turn: no scripted turns remain.",
            operation::WORKFLOW_TURN,
        ),
        (
            tts_error,
            "Unexpected TTS call: no scripted results remain.",
            operation::TTS,
        ),
    ] {
        assert_eq!(error.to_string(), message);
        assert_eq!(
            VoiceScriptError::UnexpectedVoiceCall {
                message: message.to_string(),
                operation: op,
            }
            .to_string(),
            error.to_string()
        );
    }
}

#[tokio::test]
async fn scripted_tts_rejects_unexpected_call() {
    let tts = FakeTTSModel::new();
    let error = tts
        .run("hello", &TTSModelSettings::default())
        .await
        .err()
        .expect("no scripted result");
    assert!(
        error.to_string().contains("no scripted results remain"),
        "{error}"
    );
}

#[tokio::test]
async fn scripted_workflow_yields_its_configured_start_fragment() {
    let workflow = FakeVoiceWorkflow::new().with_start(vec!["hello".to_string()]);
    let fragments: Vec<String> = std::pin::pin!(workflow.on_start())
        .map(|f| f.unwrap())
        .collect()
        .await;
    assert_eq!(fragments, ["hello"]);
    assert_eq!(workflow.assert_complete(), Ok(()));
}

#[tokio::test]
async fn scripted_workflow_tracks_explicit_empty_start_step() {
    let workflow = FakeVoiceWorkflow::new().with_start(Vec::new());

    assert_eq!(
        workflow.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "1 scripted workflow startup step was not consumed.".to_string(),
            remaining_steps: 1,
        })
    );

    let fragments: Vec<_> = std::pin::pin!(workflow.on_start()).collect().await;
    assert!(fragments.is_empty());
    assert_eq!(workflow.assert_complete(), Ok(()));

    let repeated = std::pin::pin!(workflow.on_start())
        .next()
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(
        repeated.to_string(),
        "Unexpected workflow startup call: no scripted startup step remains."
    );
}

#[tokio::test]
async fn scripted_workflow_yields_its_configured_turn_fragment() {
    let workflow = FakeVoiceWorkflow::new().with_turn(vec!["hello".to_string()]);
    let fragments: Vec<String> = std::pin::pin!(workflow.run("transcript".to_string()))
        .map(|f| f.unwrap())
        .collect()
        .await;
    assert_eq!(fragments, ["hello"]);
    assert_eq!(workflow.assert_complete(), Ok(()));
}

#[test]
fn scripted_workflow_reports_all_unconsumed_steps() {
    let workflow = FakeVoiceWorkflow::new()
        .with_turn(vec!["unused".to_string()])
        .with_start(vec!["unused".to_string()]);
    assert_eq!(
        workflow.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "2 scripted workflow step(s) were not consumed.".to_string(),
            remaining_steps: 2,
        })
    );
}

#[tokio::test]
async fn static_pipeline_leaves_configured_workflow_start_unconsumed() {
    trace_processor();
    let workflow = Arc::new(
        FakeVoiceWorkflow::new()
            .with_turn(Vec::new())
            .with_start(vec!["streamed-only greeting".to_string()]),
    );
    let pipeline = VoicePipeline::new(workflow.clone())
        .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hello")))
        .with_tts_model(Arc::new(FakeTTSModel::new()))
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    {
        let mut stream = std::pin::pin!(result.stream());
        while stream.next().await.is_some() {}
    }

    assert_eq!(
        workflow.assert_complete(),
        Err(VoiceScriptError::UnconsumedVoiceSteps {
            message: "1 scripted workflow startup step was not consumed.".to_string(),
            remaining_steps: 1,
        })
    );
}
