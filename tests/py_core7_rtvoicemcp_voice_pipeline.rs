//! Ports of the int16 and cross-chunk `StreamedAudioResult` tests of
//! `tests/voice/test_pipeline.py`: a sample whose two bytes arrive in different TTS chunks is
//! reassembled, and a trailing odd byte is padded rather than dropped. The float32 twin lives in
//! `py_voice_pipeline.rs`.
#![cfg(feature = "voice")]

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use openai_agents_sdk::Result;
use openai_agents_sdk::voice::testing::{FakeSTTModel, FakeVoiceWorkflow};
use openai_agents_sdk::voice::{
    AudioByteStream, AudioData, AudioFormat, AudioInput, TTSModel, TTSModelSettings, VoicePipeline,
    VoicePipelineConfig, VoiceStreamEvent,
};

/// A TTS model that yields the given chunks, byte for byte, for every call.
#[derive(Debug)]
struct ChunkedPcmTts(Vec<Vec<u8>>);

#[async_trait]
impl TTSModel for ChunkedPcmTts {
    fn model_name(&self) -> &str {
        "chunked-pcm-tts"
    }

    async fn run(&self, _text: &str, _settings: &TTSModelSettings) -> Result<AudioByteStream> {
        let chunks: Vec<Result<Bytes>> =
            self.0.iter().map(|c| Ok(Bytes::from(c.clone()))).collect();
        Ok(futures::stream::iter(chunks).boxed())
    }
}

/// Run one turn through a pipeline whose TTS yields `chunks`, and collect the audio events.
async fn audio_events(chunks: Vec<Vec<u8>>, dtype: AudioFormat) -> Vec<AudioData> {
    let pipeline = VoicePipeline::new(Arc::new(
        FakeVoiceWorkflow::new().with_turn(vec!["hello".to_string()]),
    ))
    .with_stt_model(Arc::new(FakeSTTModel::new().with_transcription("hi")))
    .with_tts_model(Arc::new(ChunkedPcmTts(chunks)))
    .with_config(
        VoicePipelineConfig::default()
            .with_tracing_disabled(true)
            .with_tts_settings(
                TTSModelSettings::default()
                    .with_buffer_size(1)
                    .with_dtype(dtype)
                    .with_text_splitter(Arc::new(|text: &str| (text.to_string(), String::new()))),
            ),
    );
    let mut result = pipeline.run(AudioInput::new(vec![0i16; 2])).await.unwrap();
    let mut audio = Vec::new();
    let mut stream = std::pin::pin!(result.stream());
    while let Some(event) = stream.next().await {
        if let VoiceStreamEvent::Audio(data) = event {
            audio.push(data);
        }
    }
    audio
}

fn int16_samples(audio: &[AudioData]) -> Vec<i16> {
    audio
        .iter()
        .flat_map(|data| match data {
            AudioData::Int16(samples) => samples.clone(),
            other => panic!("expected int16 audio, got {other:?}"),
        })
        .collect()
}

/// `test_streamed_audio_result_odd_length_buffer_int16`
#[tokio::test]
async fn odd_length_buffer_is_padded_for_int16() {
    let audio = audio_events(vec![vec![0x01]], AudioFormat::Int16).await;
    assert_eq!(int16_samples(&audio), [1]);
}

/// `test_streamed_audio_result_preserves_cross_chunk_sample_boundaries`: two single-byte chunks
/// make exactly one sample, not two padded ones.
#[tokio::test]
async fn cross_chunk_sample_boundaries_are_preserved() {
    let audio = audio_events(vec![vec![0x01], vec![0x00]], AudioFormat::Int16).await;
    assert_eq!(int16_samples(&audio), [1]);
    let bytes: Vec<u8> = int16_samples(&audio)
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    assert_eq!(bytes, [1, 0]);
}
