//! Voice pipeline failure logging under the model and tool data policies: ports of
//! `test_voice_workflow_errors_apply_model_and_tool_logging_policies` and
//! `test_voice_on_start_errors_apply_model_and_tool_logging_policies` from
//! `tests/voice/test_pipeline.py`.
//!
//! A workflow failure can carry model and tool data, so the log line keeps the error text only
//! when neither `dont_log_model_data` nor `dont_log_tool_data` is set.
#![cfg(feature = "voice")]

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use openai_agents_sdk::Result;
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::voice::testing::{FakeSTTModel, FakeVoiceWorkflow, pcm16_bytes};
use openai_agents_sdk::voice::{
    AudioByteStream, AudioInput, StreamedAudioInput, StreamedAudioResult, TTSModel,
    TTSModelSettings, VoicePipeline, VoicePipelineConfig, VoiceStreamEvent,
};

struct LogSink(Mutex<Vec<(log::Level, String)>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0
                .lock()
                .unwrap()
                .push((record.level(), record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let sink = SINK.get_or_init(|| LogSink(Mutex::new(Vec::new())));
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

/// The flags and the log sink are process-global, so the tests take turns.
fn serial() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Restores the redaction flags to the environment defaults when a test ends.
struct PolicyGuard;

impl PolicyGuard {
    fn set(model_redacted: bool, tool_redacted: bool) -> Self {
        set_dont_log_model_data(model_redacted);
        set_dont_log_tool_data(tool_redacted);
        PolicyGuard
    }
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

#[derive(Debug)]
struct ZeroPcmTts;

#[async_trait]
impl TTSModel for ZeroPcmTts {
    fn model_name(&self) -> &str {
        "zero-pcm-tts"
    }

    async fn run(&self, _text: &str, _settings: &TTSModelSettings) -> Result<AudioByteStream> {
        let chunk: Bytes = pcm16_bytes(&[0, 0]);
        Ok(futures::stream::iter(vec![Ok(chunk)]).boxed())
    }
}

async fn extract_labels(result: &mut StreamedAudioResult) -> Vec<String> {
    let mut labels = Vec::new();
    let mut stream = std::pin::pin!(result.stream());
    while let Some(event) = stream.next().await {
        match event {
            VoiceStreamEvent::Audio(_) => labels.push("audio".to_string()),
            VoiceStreamEvent::Lifecycle(event) => labels.push(event.as_str().to_string()),
            VoiceStreamEvent::Error(error) => labels.push(format!("error: {error}")),
        }
    }
    labels
}

fn streamed_input(count: usize) -> StreamedAudioInput {
    let input = StreamedAudioInput::new();
    for _ in 0..count {
        input.add_audio(vec![0i16, 0]).unwrap();
    }
    input
}

const POLICIES: [(bool, bool, bool); 3] = [
    (true, false, true),
    (false, true, true),
    (false, false, false),
];

#[tokio::test]
async fn voice_workflow_errors_apply_model_and_tool_logging_policies() {
    const SECRET: &str = "SECRET_VOICE_TOOL_PAYLOAD";
    let _turn = serial().lock().await;
    let sink = sink();

    for streamed in [false, true] {
        for (model_redacted, tool_redacted, redacted) in POLICIES {
            let _policy = PolicyGuard::set(model_redacted, tool_redacted);
            sink.0.lock().unwrap().clear();
            let pipeline =
                VoicePipeline::new(Arc::new(FakeVoiceWorkflow::new().with_turn_error(SECRET)))
                    .with_stt_model(Arc::new(
                        FakeSTTModel::new()
                            .with_transcription("first")
                            .with_session(vec!["first".to_string()]),
                    ))
                    .with_tts_model(Arc::new(ZeroPcmTts))
                    .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

            let mut result = if streamed {
                pipeline.run(streamed_input(1)).await.unwrap()
            } else {
                pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap()
            };
            let labels = tokio::time::timeout(Duration::from_secs(5), extract_labels(&mut result))
                .await
                .expect("the session must end");
            assert!(labels.last().unwrap().starts_with("error: "), "{labels:?}");
            assert!(labels.last().unwrap().contains(SECRET), "{labels:?}");

            let pipeline_message = if streamed {
                "Error processing voice turns"
            } else {
                "Error processing single voice turn"
            };
            let records: Vec<String> = sink
                .0
                .lock()
                .unwrap()
                .iter()
                .filter(|(level, message)| {
                    *level == log::Level::Error
                        && (message.starts_with(pipeline_message)
                            || message.starts_with("Error processing voice output"))
                })
                .map(|(_, message)| message.clone())
                .collect();
            let case = format!("streamed={streamed} model={model_redacted} tool={tool_redacted}");
            assert_eq!(records.len(), 2, "{case}: {records:?}");
            for record in &records {
                if redacted {
                    assert!(
                        record == pipeline_message || record == "Error processing voice output",
                        "{case}: {record}"
                    );
                    assert!(!record.contains(SECRET), "{case}: {record}");
                } else {
                    assert!(record.contains(&format!(": {SECRET}")), "{case}: {record}");
                }
            }
        }
    }
}

#[tokio::test]
async fn voice_on_start_errors_apply_model_and_tool_logging_policies() {
    const SECRET: &str = "SECRET_VOICE_ON_START_TOOL_PAYLOAD";
    let _turn = serial().lock().await;
    let sink = sink();

    for (model_redacted, tool_redacted, redacted) in POLICIES {
        let _policy = PolicyGuard::set(model_redacted, tool_redacted);
        sink.0.lock().unwrap().clear();
        let pipeline = VoicePipeline::new(Arc::new(
            FakeVoiceWorkflow::new()
                .with_start_error(SECRET)
                .with_turn(vec!["out_1".to_string()]),
        ))
        .with_stt_model(Arc::new(
            FakeSTTModel::new().with_session(vec!["first".to_string()]),
        ))
        .with_tts_model(Arc::new(ZeroPcmTts))
        .with_config(VoicePipelineConfig::default().with_tracing_disabled(true));

        let mut result = pipeline.run(streamed_input(1)).await.unwrap();
        let labels = tokio::time::timeout(Duration::from_secs(5), extract_labels(&mut result))
            .await
            .expect("the session must end");
        assert_eq!(labels.last().unwrap(), "session_ended", "{labels:?}");

        let records: Vec<(log::Level, String)> = sink
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, message)| message.starts_with("Voice workflow on_start failed"))
            .cloned()
            .collect();
        let case = format!("model={model_redacted} tool={tool_redacted}");
        assert_eq!(records.len(), 1, "{case}: {records:?}");
        let (level, record) = &records[0];
        assert_eq!(*level, log::Level::Warn, "{case}");
        if redacted {
            assert_eq!(record, "Voice workflow on_start failed", "{case}");
        } else {
            assert!(record.contains(&format!(": {SECRET}")), "{case}: {record}");
        }
    }
}
