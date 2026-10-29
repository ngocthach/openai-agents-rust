//! Deterministic voice components for tests. Mirrors `agents.voice.testing`.
//!
//! ```
//! use openai_agents_sdk::voice::testing::{FakeSTTModel, FakeTTSModel, pcm16_bytes};
//! use openai_agents_sdk::voice::{STTModelSettings, STTModel};
//!
//! # tokio_test_block(async {
//! let stt = FakeSTTModel::new().with_transcription("hello");
//! let text = stt
//!     .transcribe(&Default::default(), &STTModelSettings::default(), false, false)
//!     .await
//!     .unwrap();
//! assert_eq!(text, "hello");
//! let _tts = FakeTTSModel::new().with_result(vec![pcm16_bytes(&[1, 2])]);
//! # });
//! # fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)
//! # }
//! ```

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;

use crate::error::{AgentsError, Result};
use crate::voice::input::{AudioInput, StreamedAudioInput};
use crate::voice::model::{
    AudioByteStream, STTModel, STTModelSettings, StreamedTranscriptionSession, TTSModel,
    TTSModelSettings,
};
use crate::voice::workflow::{TextStream, VoiceWorkflowBase};

/// Encode 16-bit PCM samples as little-endian bytes.
pub fn pcm16_bytes(samples: &[i16]) -> Bytes {
    Bytes::from(crate::voice::input::pcm16_to_bytes(samples))
}

/// One scripted step: a value or a failure.
#[derive(Debug, Clone)]
enum Step<T> {
    Value(T),
    Error(String),
}

impl<T> Step<T> {
    fn take(self) -> Result<T> {
        match self {
            Step::Value(value) => Ok(value),
            Step::Error(message) => Err(AgentsError::other(message)),
        }
    }
}

/// A recorded static transcription call.
#[derive(Debug, Clone, PartialEq)]
pub struct STTCall {
    /// The audio that was transcribed.
    pub input: AudioInput,
    /// The settings the pipeline passed.
    pub settings: STTModelSettings,
    /// Whether sensitive data tracing was enabled.
    pub trace_include_sensitive_data: bool,
    /// Whether sensitive audio tracing was enabled.
    pub trace_include_sensitive_audio_data: bool,
}

/// A recorded text-to-speech call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TTSCall {
    /// The text that was spoken.
    pub text: String,
}

/// A speech-to-text model returning scripted transcriptions and sessions.
#[derive(Debug, Default)]
pub struct FakeSTTModel {
    transcriptions: Mutex<VecDeque<Step<String>>>,
    sessions: Mutex<VecDeque<Step<Vec<String>>>>,
    calls: Mutex<Vec<STTCall>>,
    model_name: String,
}

impl FakeSTTModel {
    /// A model with no scripted steps.
    pub fn new() -> Self {
        FakeSTTModel {
            model_name: "fake-stt".to_string(),
            ..Default::default()
        }
    }

    /// Queue a transcription for [`STTModel::transcribe`].
    pub fn with_transcription(self, text: impl Into<String>) -> Self {
        self.transcriptions
            .lock()
            .expect("poisoned")
            .push_back(Step::Value(text.into()));
        self
    }

    /// Queue a transcription failure.
    pub fn with_transcription_error(self, message: impl Into<String>) -> Self {
        self.transcriptions
            .lock()
            .expect("poisoned")
            .push_back(Step::Error(message.into()));
        self
    }

    /// Queue a session for [`STTModel::create_session`], yielding one turn per entry.
    pub fn with_session(self, turns: Vec<String>) -> Self {
        self.sessions
            .lock()
            .expect("poisoned")
            .push_back(Step::Value(turns));
        self
    }

    /// Queue a session creation failure.
    pub fn with_session_error(self, message: impl Into<String>) -> Self {
        self.sessions
            .lock()
            .expect("poisoned")
            .push_back(Step::Error(message.into()));
        self
    }

    /// Set the reported model name.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// The recorded static transcription calls.
    pub fn calls(&self) -> Vec<STTCall> {
        self.calls.lock().expect("poisoned").clone()
    }
}

#[async_trait]
impl STTModel for FakeSTTModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn transcribe(
        &self,
        input: &AudioInput,
        settings: &STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> Result<String> {
        self.calls.lock().expect("poisoned").push(STTCall {
            input: input.clone(),
            settings: settings.clone(),
            trace_include_sensitive_data,
            trace_include_sensitive_audio_data,
        });
        self.transcriptions
            .lock()
            .expect("poisoned")
            .pop_front()
            .ok_or_else(|| {
                AgentsError::other(
                    "Unexpected static transcription call: no scripted transcriptions remain",
                )
            })?
            .take()
    }

    async fn create_session(
        &self,
        _input: &StreamedAudioInput,
        _settings: &STTModelSettings,
        _trace_include_sensitive_data: bool,
        _trace_include_sensitive_audio_data: bool,
    ) -> Result<Box<dyn StreamedTranscriptionSession>> {
        let turns = self
            .sessions
            .lock()
            .expect("poisoned")
            .pop_front()
            .ok_or_else(|| {
                AgentsError::other(
                    "Unexpected streamed transcription session: no scripted sessions remain",
                )
            })?
            .take()?;
        Ok(Box::new(FakeTranscriptionSession {
            turns: turns.into(),
            close_calls: 0,
        }))
    }
}

/// A transcription session yielding scripted turns.
#[derive(Debug)]
pub struct FakeTranscriptionSession {
    turns: VecDeque<String>,
    close_calls: usize,
}

impl FakeTranscriptionSession {
    /// How often the session was closed.
    pub fn close_calls(&self) -> usize {
        self.close_calls
    }
}

#[async_trait]
impl StreamedTranscriptionSession for FakeTranscriptionSession {
    async fn next_turn(&mut self) -> Result<Option<String>> {
        Ok(self.turns.pop_front())
    }

    async fn close(&mut self) -> Result<()> {
        self.close_calls += 1;
        Ok(())
    }
}

/// A text-to-speech model yielding scripted PCM chunks.
#[derive(Debug, Default)]
pub struct FakeTTSModel {
    results: Mutex<VecDeque<Step<Vec<Bytes>>>>,
    calls: Mutex<Vec<TTSCall>>,
    model_name: String,
}

impl FakeTTSModel {
    /// A model with no scripted results.
    pub fn new() -> Self {
        FakeTTSModel {
            model_name: "fake-tts".to_string(),
            ..Default::default()
        }
    }

    /// Queue the chunks returned by the next call.
    pub fn with_result(self, chunks: Vec<Bytes>) -> Self {
        self.results
            .lock()
            .expect("poisoned")
            .push_back(Step::Value(chunks));
        self
    }

    /// Queue a failure for the next call.
    pub fn with_error(self, message: impl Into<String>) -> Self {
        self.results
            .lock()
            .expect("poisoned")
            .push_back(Step::Error(message.into()));
        self
    }

    /// Set the reported model name.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// The recorded calls.
    pub fn calls(&self) -> Vec<TTSCall> {
        self.calls.lock().expect("poisoned").clone()
    }
}

#[async_trait]
impl TTSModel for FakeTTSModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn run(&self, text: &str, _settings: &TTSModelSettings) -> Result<AudioByteStream> {
        self.calls.lock().expect("poisoned").push(TTSCall {
            text: text.to_string(),
        });
        let chunks = self
            .results
            .lock()
            .expect("poisoned")
            .pop_front()
            .ok_or_else(|| AgentsError::other("Unexpected TTS call: no scripted results remain"))?
            .take()?;
        Ok(futures::stream::iter(chunks.into_iter().map(Ok)).boxed())
    }
}

/// A workflow yielding scripted text fragments, one entry per turn.
#[derive(Debug, Default)]
pub struct FakeVoiceWorkflow {
    turns: Mutex<VecDeque<Step<Vec<String>>>>,
    start: Mutex<Vec<String>>,
    transcriptions: Mutex<Vec<String>>,
}

impl FakeVoiceWorkflow {
    /// A workflow with no scripted turns.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue the fragments yielded for the next transcription.
    pub fn with_turn(self, fragments: Vec<String>) -> Self {
        self.turns
            .lock()
            .expect("poisoned")
            .push_back(Step::Value(fragments));
        self
    }

    /// Queue a failure for the next turn.
    pub fn with_turn_error(self, message: impl Into<String>) -> Self {
        self.turns
            .lock()
            .expect("poisoned")
            .push_back(Step::Error(message.into()));
        self
    }

    /// Set the fragments yielded before any user input.
    pub fn with_start(self, fragments: Vec<String>) -> Self {
        *self.start.lock().expect("poisoned") = fragments;
        self
    }

    /// The transcriptions the workflow was run with.
    pub fn transcriptions(&self) -> Vec<String> {
        self.transcriptions.lock().expect("poisoned").clone()
    }
}

impl VoiceWorkflowBase for FakeVoiceWorkflow {
    fn run(&self, transcription: String) -> TextStream<'_> {
        self.transcriptions
            .lock()
            .expect("poisoned")
            .push(transcription);
        let step = self.turns.lock().expect("poisoned").pop_front();
        let fragments = match step {
            Some(step) => step.take(),
            None => Err(AgentsError::other(
                "Unexpected workflow turn: no scripted turns remain",
            )),
        };
        match fragments {
            Ok(fragments) => futures::stream::iter(fragments.into_iter().map(Ok)).boxed(),
            Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
        }
    }

    fn on_start(&self) -> TextStream<'_> {
        let fragments = self.start.lock().expect("poisoned").clone();
        futures::stream::iter(fragments.into_iter().map(Ok)).boxed()
    }
}
