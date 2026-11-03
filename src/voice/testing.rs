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
//! stt.assert_complete().unwrap();
//! let _tts = FakeTTSModel::new().with_result(vec![pcm16_bytes(&[1, 2])]);
//! # });
//! # fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)
//! # }
//! ```

use std::collections::VecDeque;
use std::fmt;
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

/// Encode 16-bit PCM samples as little-endian bytes. Python's `pcm16_samples`.
pub fn pcm16_samples(samples: &[i16]) -> Bytes {
    pcm16_bytes(samples)
}

/// The scripted operation a [`VoiceScriptError::UnexpectedVoiceCall`] refers to.
pub mod operation {
    /// [`super::FakeSTTModel::transcribe`](crate::voice::STTModel::transcribe).
    pub const STATIC_TRANSCRIPTION: &str = "static_transcription";
    /// [`super::FakeSTTModel::create_session`](crate::voice::STTModel::create_session).
    pub const STREAMED_SESSION: &str = "streamed_session";
    /// [`super::FakeTTSModel::run`](crate::voice::TTSModel::run).
    pub const TTS: &str = "tts";
    /// [`super::FakeVoiceWorkflow::on_start`](crate::voice::VoiceWorkflowBase::on_start).
    pub const WORKFLOW_START: &str = "workflow_start";
    /// [`super::FakeVoiceWorkflow::run`](crate::voice::VoiceWorkflowBase::run).
    pub const WORKFLOW_TURN: &str = "workflow_turn";
}

/// An invalid or incompletely consumed voice script.
///
/// Mirrors Python's `VoiceScriptError` hierarchy, and matches the shape of
/// [`ModelScriptError`](crate::testing::ModelScriptError).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceScriptError {
    /// A voice component was called after its script was exhausted.
    UnexpectedVoiceCall {
        /// The message Python raises.
        message: String,
        /// Which scripted operation found no step left (see [`operation`]).
        operation: &'static str,
    },
    /// A test finished before consuming every configured step.
    UnconsumedVoiceSteps {
        /// The message Python raises.
        message: String,
        /// Steps still queued.
        remaining_steps: usize,
    },
}

impl VoiceScriptError {
    fn unexpected(message: &str, operation: &'static str) -> Self {
        VoiceScriptError::UnexpectedVoiceCall {
            message: message.to_string(),
            operation,
        }
    }

    fn unconsumed(message: String, remaining_steps: usize) -> Self {
        VoiceScriptError::UnconsumedVoiceSteps {
            message,
            remaining_steps,
        }
    }
}

impl fmt::Display for VoiceScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoiceScriptError::UnexpectedVoiceCall { message, .. }
            | VoiceScriptError::UnconsumedVoiceSteps { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for VoiceScriptError {}

impl From<VoiceScriptError> for AgentsError {
    fn from(error: VoiceScriptError) -> Self {
        AgentsError::other(error.to_string())
    }
}

/// The result of an `assert_complete` call.
pub type ScriptResult = std::result::Result<(), VoiceScriptError>;

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

/// A recorded streamed transcription-session creation call.
///
/// The streamed input itself is a live queue that cannot be snapshotted, so only the settings
/// and the two tracing flags are recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct STTSessionCall {
    /// The settings the pipeline passed.
    pub settings: STTModelSettings,
    /// Whether sensitive data tracing was enabled.
    pub trace_include_sensitive_data: bool,
    /// Whether sensitive audio tracing was enabled.
    pub trace_include_sensitive_audio_data: bool,
}

/// A recorded text-to-speech call.
#[derive(Debug, Clone)]
pub struct TTSCall {
    /// The text that was spoken.
    pub text: String,
    /// The settings the pipeline passed.
    pub settings: TTSModelSettings,
}

/// The PCM byte chunks one scripted text-to-speech call returns.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TTSResult {
    /// The chunks yielded, in order.
    pub chunks: Vec<Bytes>,
}

impl From<Vec<Bytes>> for TTSResult {
    fn from(chunks: Vec<Bytes>) -> Self {
        TTSResult { chunks }
    }
}

/// A speech-to-text model returning scripted transcriptions and sessions.
#[derive(Debug, Default)]
pub struct FakeSTTModel {
    transcriptions: Mutex<VecDeque<Step<String>>>,
    sessions: Mutex<VecDeque<Step<FakeTranscriptionSession>>>,
    calls: Mutex<Vec<STTCall>>,
    session_calls: Mutex<Vec<STTSessionCall>>,
    created_sessions: Mutex<Vec<FakeTranscriptionSession>>,
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
        self.with_session_instance(FakeTranscriptionSession::new(turns))
    }

    /// Queue a pre-built session, so a test can configure its turns and its close failure.
    pub fn with_session_instance(self, session: FakeTranscriptionSession) -> Self {
        self.sessions
            .lock()
            .expect("poisoned")
            .push_back(Step::Value(session));
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

    /// The recorded streamed-session creation calls.
    pub fn session_calls(&self) -> Vec<STTSessionCall> {
        self.session_calls.lock().expect("poisoned").clone()
    }

    /// The sessions this model created, sharing state with the ones the pipeline holds.
    pub fn created_sessions(&self) -> Vec<FakeTranscriptionSession> {
        self.created_sessions.lock().expect("poisoned").clone()
    }

    /// Fail when static transcriptions, sessions, or any created session's turns remain.
    pub fn assert_complete(&self) -> ScriptResult {
        let remaining = self.transcriptions.lock().expect("poisoned").len()
            + self.sessions.lock().expect("poisoned").len();
        if remaining > 0 {
            return Err(VoiceScriptError::unconsumed(
                format!("{remaining} scripted STT step(s) were not consumed."),
                remaining,
            ));
        }
        for session in self.created_sessions() {
            session.assert_complete()?;
        }
        Ok(())
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
                VoiceScriptError::unexpected(
                    "Unexpected static transcription call: no scripted transcriptions remain.",
                    operation::STATIC_TRANSCRIPTION,
                )
            })?
            .take()
    }

    async fn create_session(
        &self,
        _input: &StreamedAudioInput,
        settings: &STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> Result<Box<dyn StreamedTranscriptionSession>> {
        self.session_calls
            .lock()
            .expect("poisoned")
            .push(STTSessionCall {
                settings: settings.clone(),
                trace_include_sensitive_data,
                trace_include_sensitive_audio_data,
            });
        let session = self
            .sessions
            .lock()
            .expect("poisoned")
            .pop_front()
            .ok_or_else(|| {
                VoiceScriptError::unexpected(
                    "Unexpected streamed transcription session: no scripted sessions remain.",
                    operation::STREAMED_SESSION,
                )
            })?
            .take()?;
        self.created_sessions
            .lock()
            .expect("poisoned")
            .push(session.clone());
        Ok(Box::new(session))
    }
}

/// Turns left to yield, the close failure to inject, and how often the session was closed.
#[derive(Debug, Default)]
struct TranscriptionSessionState {
    turns: VecDeque<Step<String>>,
    closed: bool,
    close_calls: usize,
    close_error: Option<String>,
}

/// A transcription session yielding scripted turns.
///
/// Clones share one state, so a test can keep observing a session the pipeline took ownership
/// of (see [`FakeSTTModel::created_sessions`]).
#[derive(Debug, Clone, Default)]
pub struct FakeTranscriptionSession {
    state: std::sync::Arc<Mutex<TranscriptionSessionState>>,
}

impl FakeTranscriptionSession {
    /// A session yielding `turns`, one per call to `next_turn`.
    pub fn new(turns: Vec<String>) -> Self {
        FakeTranscriptionSession {
            state: std::sync::Arc::new(Mutex::new(TranscriptionSessionState {
                turns: turns.into_iter().map(Step::Value).collect(),
                ..Default::default()
            })),
        }
    }

    /// Queue a failure for the next turn.
    pub fn with_turn_error(self, message: impl Into<String>) -> Self {
        self.state
            .lock()
            .expect("poisoned")
            .turns
            .push_back(Step::Error(message.into()));
        self
    }

    /// Fail the first [`StreamedTranscriptionSession::close`] with this message.
    pub fn with_close_error(self, message: impl Into<String>) -> Self {
        self.state.lock().expect("poisoned").close_error = Some(message.into());
        self
    }

    /// How often the session was closed, including the closes that were already no-ops.
    pub fn close_calls(&self) -> usize {
        self.state.lock().expect("poisoned").close_calls
    }

    /// Whether the session has been closed.
    pub fn is_closed(&self) -> bool {
        self.state.lock().expect("poisoned").closed
    }

    /// Turns the session has not yielded yet.
    pub fn remaining_turns(&self) -> usize {
        self.state.lock().expect("poisoned").turns.len()
    }

    /// Fail when configured transcript turns remain unconsumed.
    pub fn assert_complete(&self) -> ScriptResult {
        let remaining = self.remaining_turns();
        if remaining > 0 {
            return Err(VoiceScriptError::unconsumed(
                format!("{remaining} scripted transcription turn(s) were not consumed."),
                remaining,
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl StreamedTranscriptionSession for FakeTranscriptionSession {
    async fn next_turn(&mut self) -> Result<Option<String>> {
        // Python's `transcribe_turns` loop stops at `close()`, even with turns left.
        let step = {
            let mut state = self.state.lock().expect("poisoned");
            if state.closed {
                return Ok(None);
            }
            state.turns.pop_front()
        };
        step.map(Step::take).transpose()
    }

    async fn close(&mut self) -> Result<()> {
        let error = {
            let mut state = self.state.lock().expect("poisoned");
            state.close_calls += 1;
            if state.closed {
                // A second close is a no-op and does not raise again.
                return Ok(());
            }
            state.closed = true;
            state.close_error.clone()
        };
        match error {
            Some(message) => Err(AgentsError::other(message)),
            None => Ok(()),
        }
    }
}

/// A text-to-speech model yielding scripted PCM chunks.
#[derive(Debug, Default)]
pub struct FakeTTSModel {
    results: Mutex<VecDeque<Step<TTSResult>>>,
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
    pub fn with_result(self, chunks: impl Into<TTSResult>) -> Self {
        self.results
            .lock()
            .expect("poisoned")
            .push_back(Step::Value(chunks.into()));
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

    /// Fail when configured results remain unconsumed.
    pub fn assert_complete(&self) -> ScriptResult {
        let remaining = self.results.lock().expect("poisoned").len();
        if remaining > 0 {
            return Err(VoiceScriptError::unconsumed(
                format!("{remaining} scripted TTS result(s) were not consumed."),
                remaining,
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl TTSModel for FakeTTSModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn run(&self, text: &str, settings: &TTSModelSettings) -> Result<AudioByteStream> {
        self.calls.lock().expect("poisoned").push(TTSCall {
            text: text.to_string(),
            settings: settings.clone(),
        });
        let result = self
            .results
            .lock()
            .expect("poisoned")
            .pop_front()
            .ok_or_else(|| {
                VoiceScriptError::unexpected(
                    "Unexpected TTS call: no scripted results remain.",
                    operation::TTS,
                )
            })?
            .take()?;
        Ok(futures::stream::iter(result.chunks.into_iter().map(Ok)).boxed())
    }
}

/// The startup step of a [`FakeVoiceWorkflow`].
///
/// `configured` is tracked apart from `pending`: an unconfigured startup yields nothing and
/// never fails, while a configured one may only be consumed once.
#[derive(Debug, Default)]
struct StartState {
    configured: bool,
    pending: bool,
    step: Option<Step<Vec<String>>>,
}

/// A workflow yielding scripted text fragments, one entry per turn.
#[derive(Debug, Default)]
pub struct FakeVoiceWorkflow {
    turns: Mutex<VecDeque<Step<Vec<String>>>>,
    start: Mutex<StartState>,
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
        self.set_start(Step::Value(fragments))
    }

    /// Fail the startup step.
    pub fn with_start_error(self, message: impl Into<String>) -> Self {
        self.set_start(Step::Error(message.into()))
    }

    fn set_start(self, step: Step<Vec<String>>) -> Self {
        *self.start.lock().expect("poisoned") = StartState {
            configured: true,
            pending: true,
            step: Some(step),
        };
        self
    }

    /// The transcriptions the workflow was run with.
    pub fn transcriptions(&self) -> Vec<String> {
        self.transcriptions.lock().expect("poisoned").clone()
    }

    /// Fail when the startup step or configured turns remain unconsumed.
    pub fn assert_complete(&self) -> ScriptResult {
        let pending = self.start.lock().expect("poisoned").pending;
        let turns = self.turns.lock().expect("poisoned").len();
        let remaining = usize::from(pending) + turns;
        if remaining == 0 {
            return Ok(());
        }
        if pending && turns == 0 {
            return Err(VoiceScriptError::unconsumed(
                "1 scripted workflow startup step was not consumed.".to_string(),
                1,
            ));
        }
        if !pending {
            return Err(VoiceScriptError::unconsumed(
                format!("{turns} scripted workflow turn(s) were not consumed."),
                turns,
            ));
        }
        Err(VoiceScriptError::unconsumed(
            format!("{remaining} scripted workflow step(s) were not consumed."),
            remaining,
        ))
    }
}

/// A stream yielding the fragments of one step, or its failure.
fn fragment_stream(fragments: Result<Vec<String>>) -> TextStream<'static> {
    match fragments {
        Ok(fragments) => futures::stream::iter(fragments.into_iter().map(Ok)).boxed(),
        Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
    }
}

impl VoiceWorkflowBase for FakeVoiceWorkflow {
    fn run(&self, transcription: String) -> TextStream<'_> {
        self.transcriptions
            .lock()
            .expect("poisoned")
            .push(transcription);
        let step = self.turns.lock().expect("poisoned").pop_front();
        fragment_stream(match step {
            Some(step) => step.take(),
            None => Err(VoiceScriptError::unexpected(
                "Unexpected workflow turn: no scripted turns remain.",
                operation::WORKFLOW_TURN,
            )
            .into()),
        })
    }

    fn on_start(&self) -> TextStream<'_> {
        let step = {
            let mut start = self.start.lock().expect("poisoned");
            if start.configured && !start.pending {
                return fragment_stream(Err(VoiceScriptError::unexpected(
                    "Unexpected workflow startup call: no scripted startup step remains.",
                    operation::WORKFLOW_START,
                )
                .into()));
            }
            start.pending = false;
            start.step.clone()
        };
        fragment_stream(match step {
            Some(step) => step.take(),
            None => Ok(Vec::new()),
        })
    }
}

/// Python's name for [`FakeSTTModel`].
pub type ScriptedSTTModel = FakeSTTModel;
/// Python's name for [`FakeTranscriptionSession`].
pub type ScriptedTranscriptionSession = FakeTranscriptionSession;
/// Python's name for [`FakeTTSModel`].
pub type ScriptedTTSModel = FakeTTSModel;
/// Python's name for [`FakeVoiceWorkflow`].
pub type ScriptedVoiceWorkflow = FakeVoiceWorkflow;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unconfigured_start_is_never_pending() {
        let workflow = FakeVoiceWorkflow::new();
        assert_eq!(workflow.assert_complete(), Ok(()));
    }

    #[test]
    fn a_configured_start_must_be_consumed() {
        let workflow = FakeVoiceWorkflow::new().with_start(vec!["hi".into()]);
        assert_eq!(
            workflow.assert_complete(),
            Err(VoiceScriptError::UnconsumedVoiceSteps {
                message: "1 scripted workflow startup step was not consumed.".to_string(),
                remaining_steps: 1,
            })
        );
    }

    #[test]
    fn unconsumed_turns_and_a_pending_start_report_steps() {
        let workflow = FakeVoiceWorkflow::new()
            .with_start(vec!["hi".into()])
            .with_turn(vec!["a".into()])
            .with_turn(vec!["b".into()]);
        assert_eq!(
            workflow.assert_complete(),
            Err(VoiceScriptError::UnconsumedVoiceSteps {
                message: "3 scripted workflow step(s) were not consumed.".to_string(),
                remaining_steps: 3,
            })
        );
    }

    #[test]
    fn the_python_names_resolve_to_the_same_types() {
        let _: ScriptedSTTModel = FakeSTTModel::new();
        let _: ScriptedTTSModel = FakeTTSModel::new();
        let _: ScriptedTranscriptionSession = FakeTranscriptionSession::default();
        let _: ScriptedVoiceWorkflow = FakeVoiceWorkflow::new();
        assert_eq!(pcm16_samples(&[1, -1]), pcm16_bytes(&[1, -1]));
    }

    #[test]
    fn a_script_error_converts_to_an_agents_error_with_the_same_message() {
        let error = VoiceScriptError::unexpected("boom.", operation::TTS);
        let converted: AgentsError = error.clone().into();
        assert_eq!(converted.to_string(), error.to_string());
        assert!(std::error::Error::source(&error).is_none());
    }
}
