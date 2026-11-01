//! Port of `test_voice_pipeline_awaits_task_cleanup_after_tts_failure`
//! (`tests/voice/test_pipeline.py`): when one TTS segment fails, a sibling segment that is still
//! synthesizing must be stopped before the public stream reports the failure.
#![cfg(feature = "voice")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::voice::testing::{FakeSTTModel, FakeVoiceWorkflow};
use openai_agents_sdk::voice::{
    AudioByteStream, AudioInput, TTSModel, TTSModelSettings, VoicePipeline, VoicePipelineConfig,
    VoiceStreamEvent,
};
use openai_agents_sdk::{AgentsError, Result};
use tokio::sync::Notify;

struct StopFlag(Arc<AtomicBool>);

impl Drop for StopFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct FailingFirstSegmentTts {
    second_started: Arc<Notify>,
    second_stopped: Arc<AtomicBool>,
}

#[async_trait]
impl TTSModel for FailingFirstSegmentTts {
    fn model_name(&self) -> &str {
        "failing-first-segment-tts"
    }

    async fn run(&self, text: &str, _settings: &TTSModelSettings) -> Result<AudioByteStream> {
        if text == "first" {
            self.second_started.notified().await;
            return Err(AgentsError::other("tts-failure"));
        }
        let _stop = StopFlag(self.second_stopped.clone());
        self.second_started.notify_one();
        futures::future::pending::<()>().await;
        unreachable!()
    }
}

#[tokio::test]
async fn voice_pipeline_awaits_task_cleanup_after_tts_failure() {
    let second_stopped = Arc::new(AtomicBool::new(false));
    let tts = Arc::new(FailingFirstSegmentTts {
        second_started: Arc::new(Notify::new()),
        second_stopped: second_stopped.clone(),
    });
    let config = VoicePipelineConfig::default()
        .with_tracing_disabled(true)
        .with_tts_settings(
            TTSModelSettings::default()
                .with_text_splitter(Arc::new(|text: &str| (text.to_string(), String::new()))),
        );
    let pipeline = VoicePipeline::new(Arc::new(
        FakeVoiceWorkflow::new().with_turn(vec!["first".to_string(), "second".to_string()]),
    ))
    .with_stt_model(Arc::new(
        FakeSTTModel::new().with_transcription("user input"),
    ))
    .with_tts_model(tts)
    .with_config(config);

    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let mut error = None;
    {
        let mut stream = std::pin::pin!(result.stream());
        let drain = async {
            while let Some(event) = stream.next().await {
                if let VoiceStreamEvent::Error(e) = event {
                    error = Some(e.to_string());
                }
            }
        };
        tokio::time::timeout(Duration::from_secs(5), drain)
            .await
            .expect("the stream must end");
    }
    assert!(
        error
            .expect("the failure is reported")
            .contains("tts-failure")
    );
    assert!(
        second_stopped.load(Ordering::SeqCst),
        "the sibling segment was still running when the stream ended"
    );
}
