//! Audio inputs for a [`VoicePipeline`](crate::voice::VoicePipeline). Mirrors
//! `agents.voice.input`.
//!
//! Python uses NumPy arrays of `int16` or `float32`; this port stores 16-bit PCM in
//! `Vec<i16>` and offers [`f32_to_i16`] / [`i16_to_f32`] for float buffers.

use std::fmt;
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use tokio::sync::{Mutex, mpsc};

use crate::error::{AgentsError, Result};

/// Default sample rate of voice audio, in Hz.
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

/// The sample format of the audio emitted by a pipeline.
///
/// Replaces Python's `npt.DTypeLike`, which only supports `int16` and `float32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioFormat {
    /// Signed 16-bit PCM samples.
    #[default]
    Int16,
    /// Floating point samples in `[-1.0, 1.0]`.
    Float32,
}

/// A buffer of audio samples in one of the supported [`AudioFormat`]s.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioData {
    /// Signed 16-bit PCM samples.
    Int16(Vec<i16>),
    /// Floating point samples in `[-1.0, 1.0]`.
    Float32(Vec<f32>),
}

impl AudioData {
    /// The number of samples.
    pub fn len(&self) -> usize {
        match self {
            AudioData::Int16(v) => v.len(),
            AudioData::Float32(v) => v.len(),
        }
    }

    /// Whether the buffer holds no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sample format of this buffer.
    pub fn format(&self) -> AudioFormat {
        match self {
            AudioData::Int16(_) => AudioFormat::Int16,
            AudioData::Float32(_) => AudioFormat::Float32,
        }
    }

    /// The samples as 16-bit PCM, converting floats if needed.
    pub fn to_int16(&self) -> Vec<i16> {
        match self {
            AudioData::Int16(v) => v.clone(),
            AudioData::Float32(v) => f32_to_i16(v),
        }
    }

    /// The samples as little-endian bytes, as Python's `ndarray.tobytes()` returns them.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            AudioData::Int16(v) => v.iter().flat_map(|s| s.to_le_bytes()).collect(),
            AudioData::Float32(v) => v.iter().flat_map(|s| s.to_le_bytes()).collect(),
        }
    }
}

/// Convert float samples in `[-1.0, 1.0]` to 16-bit PCM, clipping out-of-range values.
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

/// Convert 16-bit PCM samples to floats in `[-1.0, 1.0]`.
pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|s| f32::from(*s) / 32767.0).collect()
}

/// Static audio used as the input of a [`VoicePipeline`](crate::voice::VoicePipeline).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioInput {
    /// 16-bit PCM samples, interleaved across [`AudioInput::channels`].
    pub buffer: Vec<i16>,
    /// The sample rate of the audio data. Defaults to [`DEFAULT_SAMPLE_RATE`].
    pub frame_rate: u32,
    /// The sample width, in bytes, used when encoding the buffer as a WAV file (1 to 4).
    pub sample_width: u16,
    /// The number of channels in the audio data. Defaults to 1.
    pub channels: u16,
}

impl Default for AudioInput {
    fn default() -> Self {
        AudioInput {
            buffer: Vec::new(),
            frame_rate: DEFAULT_SAMPLE_RATE,
            sample_width: 2,
            channels: 1,
        }
    }
}

impl AudioInput {
    /// Mono 16-bit PCM audio at [`DEFAULT_SAMPLE_RATE`].
    pub fn new(buffer: Vec<i16>) -> Self {
        AudioInput {
            buffer,
            ..Default::default()
        }
    }

    /// Mono audio converted from float samples in `[-1.0, 1.0]`.
    pub fn from_f32(samples: &[f32]) -> Self {
        Self::new(f32_to_i16(samples))
    }

    /// Set the sample rate.
    pub fn with_frame_rate(mut self, frame_rate: u32) -> Self {
        self.frame_rate = frame_rate;
        self
    }

    /// Set the WAV sample width in bytes (1 to 4).
    pub fn with_sample_width(mut self, sample_width: u16) -> Self {
        self.sample_width = sample_width;
        self
    }

    /// Set the number of interleaved channels.
    pub fn with_channels(mut self, channels: u16) -> Self {
        self.channels = channels;
        self
    }

    /// The audio data as a base64 encoded string of little-endian 16-bit PCM.
    pub fn to_base64(&self) -> String {
        BASE64.encode(pcm16_to_bytes(&self.buffer))
    }

    /// The audio data as a WAV file: `(filename, bytes, content_type)`.
    ///
    /// Port of Python's `_buffer_to_audio_file`.
    pub fn to_audio_file(&self) -> Result<(String, Vec<u8>, String)> {
        Ok((
            "audio.wav".to_string(),
            self.to_wav_bytes()?,
            "audio/wav".to_string(),
        ))
    }

    /// Encode the buffer as a WAV file.
    pub fn to_wav_bytes(&self) -> Result<Vec<u8>> {
        if !(1..=4).contains(&self.sample_width) {
            return Err(AgentsError::user(
                "Sample width must be between 1 and 4 bytes",
            ));
        }
        if self.frame_rate == 0 {
            return Err(AgentsError::user("Frame rate must be greater than zero"));
        }
        if self.channels == 0 {
            return Err(AgentsError::user("Channels must be greater than zero"));
        }
        if !self.buffer.len().is_multiple_of(usize::from(self.channels)) {
            return Err(AgentsError::user(
                "Buffer must contain complete channel frames",
            ));
        }

        let width = usize::from(self.sample_width);
        let mut audio = Vec::with_capacity(self.buffer.len() * width);
        for sample in &self.buffer {
            if width == 1 {
                // 8-bit WAV samples are unsigned.
                audio.push((((i32::from(*sample)) >> 8) + 128) as u8);
            } else {
                let scaled = i32::from(*sample) << (8 * (width - 2));
                audio.extend_from_slice(&scaled.to_le_bytes()[..width]);
            }
        }
        Ok(wav_file(
            &audio,
            self.channels,
            self.sample_width,
            self.frame_rate,
        ))
    }
}

/// Little-endian bytes of a 16-bit PCM buffer.
pub(crate) fn pcm16_to_bytes(buffer: &[i16]) -> Vec<u8> {
    buffer.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// Wrap raw PCM frames in a canonical 44-byte WAV header.
fn wav_file(audio: &[u8], channels: u16, sample_width: u16, frame_rate: u32) -> Vec<u8> {
    let block_align = channels * sample_width;
    let byte_rate = frame_rate * u32::from(block_align);
    let data_len = audio.len() as u32;
    let mut out = Vec::with_capacity(audio.len() + 44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&frame_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&(sample_width * 8).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(audio);
    out
}

/// The queue a [`StreamedAudioInput`] pushes audio into; `None` marks the end of the stream.
pub type AudioQueueReceiver = mpsc::UnboundedReceiver<Option<Vec<i16>>>;

/// Audio input represented as a stream of buffers.
///
/// Pass it to [`VoicePipeline::run`](crate::voice::VoicePipeline::run) and push audio with
/// [`StreamedAudioInput::add_audio`]; call [`StreamedAudioInput::end`] to close the stream.
/// Clones share the same queue.
#[derive(Clone)]
pub struct StreamedAudioInput {
    tx: mpsc::UnboundedSender<Option<Vec<i16>>>,
    rx: Arc<Mutex<Option<AudioQueueReceiver>>>,
}

impl fmt::Debug for StreamedAudioInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamedAudioInput").finish_non_exhaustive()
    }
}

impl Default for StreamedAudioInput {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamedAudioInput {
    /// An empty audio stream.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        StreamedAudioInput {
            tx,
            rx: Arc::new(Mutex::new(Some(rx))),
        }
    }

    /// Add 16-bit PCM audio to the stream.
    pub fn add_audio(&self, audio: Vec<i16>) -> Result<()> {
        self.send(Some(audio))
    }

    /// Add float audio in `[-1.0, 1.0]` to the stream.
    pub fn add_audio_f32(&self, audio: &[f32]) -> Result<()> {
        self.send(Some(f32_to_i16(audio)))
    }

    /// Signal the end of the stream (Python's `add_audio(None)`).
    pub fn end(&self) -> Result<()> {
        self.send(None)
    }

    fn send(&self, audio: Option<Vec<i16>>) -> Result<()> {
        self.tx
            .send(audio)
            .map_err(|_| AgentsError::user("The streamed audio input is no longer being read"))
    }

    /// Take the receiving half of the queue, so a transcription session can consume it.
    ///
    /// Returns `None` once it has been taken; `None` items mark the end of the stream.
    pub async fn take_receiver(&self) -> Option<AudioQueueReceiver> {
        self.rx.lock().await.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_a_wav_file_with_a_canonical_header() {
        let input = AudioInput::new(vec![1, -1, 300]);
        let wav = input.to_wav_bytes().unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(&wav[44..], &[1, 0, 255, 255, 44, 1]);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24000);
    }

    #[test]
    fn rejects_invalid_audio_geometry() {
        assert!(
            AudioInput::new(vec![1])
                .with_sample_width(5)
                .to_wav_bytes()
                .is_err()
        );
        assert!(
            AudioInput::new(vec![1])
                .with_frame_rate(0)
                .to_wav_bytes()
                .is_err()
        );
        assert!(
            AudioInput::new(vec![1, 2, 3])
                .with_channels(2)
                .to_wav_bytes()
                .is_err()
        );
    }

    #[test]
    fn converts_between_float_and_pcm16() {
        assert_eq!(
            f32_to_i16(&[0.0, 1.0, -1.0, 2.0]),
            vec![0, 32767, -32767, 32767]
        );
        assert_eq!(i16_to_f32(&[0, 32767]), vec![0.0, 1.0]);
    }

    #[test]
    fn base64_encodes_little_endian_samples() {
        assert_eq!(
            AudioInput::new(vec![1]).to_base64(),
            BASE64.encode([1u8, 0])
        );
    }

    #[tokio::test]
    async fn streams_audio_buffers_in_order() {
        let input = StreamedAudioInput::new();
        input.add_audio(vec![1, 2]).unwrap();
        input.end().unwrap();
        let mut rx = input.take_receiver().await.unwrap();
        assert_eq!(rx.recv().await, Some(Some(vec![1, 2])));
        assert_eq!(rx.recv().await, Some(None));
        assert!(input.take_receiver().await.is_none());
    }
}
