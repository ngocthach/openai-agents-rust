//! A deterministic in-memory transport for tests. Port of `agents.realtime.testing`.

use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::{AgentsError, Result};
use crate::realtime::config::RealtimeSessionModelSettings;
use crate::realtime::model::{
    RealtimeModel, RealtimeModelConfig, RealtimeModelEvents, RealtimePlaybackTracker,
    RealtimeSendCondition,
};
use crate::realtime::model_events::RealtimeModelEvent;
use crate::realtime::model_inputs::RealtimeModelSendEvent;

#[derive(Default)]
struct FakeState {
    connect_calls: Vec<RealtimeModelConfig>,
    sent_events: Vec<RealtimeModelSendEvent>,
    sender: Option<mpsc::UnboundedSender<RealtimeModelEvent>>,
    connected: bool,
    close_calls: usize,
    skipped_conditional_sends: usize,
    retired_responses: Vec<String>,
}

/// A [`RealtimeModel`] that records what the session sends and emits exactly what a test asks for.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use openai_agents_sdk::realtime::{FakeRealtimeModel, RealtimeAgent, RealtimeRunner};
/// # use openai_agents_sdk::realtime::RealtimeModelEvent;
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let model = Arc::new(FakeRealtimeModel::new());
/// let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
/// let session = RealtimeRunner::new(agent).with_model(model.clone()).run(()).await?;
/// model.emit(RealtimeModelEvent::TurnStarted { response_id: Some("resp_1".into()) });
/// assert_eq!(session.recv().await.unwrap().event_type(), "history_updated");
/// # Ok(())
/// # }
/// ```
pub struct FakeRealtimeModel {
    state: Mutex<FakeState>,
}

impl Default for FakeRealtimeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for FakeRealtimeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("FakeRealtimeModel")
            .field("connected", &state.connected)
            .field("sent_events", &state.sent_events.len())
            .finish_non_exhaustive()
    }
}

impl FakeRealtimeModel {
    /// A disconnected fake transport.
    pub fn new() -> Self {
        FakeRealtimeModel {
            state: Mutex::new(FakeState::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FakeState> {
        self.state.lock().expect("fake realtime model poisoned")
    }

    /// Deliver a model event to the connected session.
    ///
    /// Returns `false` when nothing is connected, or the session has stopped reading.
    pub fn emit(&self, event: RealtimeModelEvent) -> bool {
        let sender = { self.lock().sender.clone() };
        match sender {
            Some(sender) => sender.send(event).is_ok(),
            None => false,
        }
    }

    /// Deliver several model events in order.
    pub fn emit_all<I: IntoIterator<Item = RealtimeModelEvent>>(&self, events: I) -> bool {
        events.into_iter().all(|event| self.emit(event))
    }

    /// Every event the session has sent, in order.
    pub fn sent_events(&self) -> Vec<RealtimeModelSendEvent> {
        self.lock().sent_events.clone()
    }

    /// The options of each connection attempt.
    pub fn connect_calls(&self) -> Vec<RealtimeModelConfig> {
        self.lock().connect_calls.clone()
    }

    /// The settings of the most recent `session.update`, including the one sent on connect.
    pub fn last_session_settings(&self) -> Option<RealtimeSessionModelSettings> {
        let state = self.lock();
        state
            .sent_events
            .iter()
            .rev()
            .find_map(RealtimeModelSendEvent::as_session_update)
            .cloned()
            .or_else(|| {
                state
                    .connect_calls
                    .last()
                    .and_then(|call| call.initial_model_settings.clone())
            })
    }

    /// The settings applied by each `session.update` the session sent after connecting.
    pub fn session_updates(&self) -> Vec<RealtimeSessionModelSettings> {
        self.lock()
            .sent_events
            .iter()
            .filter_map(RealtimeModelSendEvent::as_session_update)
            .cloned()
            .collect()
    }

    /// The interruptions the session sent, in order.
    pub fn interrupts(&self) -> Vec<crate::realtime::model_inputs::RealtimeInterrupt> {
        self.lock()
            .sent_events
            .iter()
            .filter_map(|event| event.as_interrupt().cloned())
            .collect()
    }

    /// Responses whose transport audio the session asked to release, in order.
    pub fn retired_responses(&self) -> Vec<String> {
        self.lock().retired_responses.clone()
    }

    /// How many conditional sends were dropped because their condition no longer held.
    pub fn skipped_conditional_sends(&self) -> usize {
        self.lock().skipped_conditional_sends
    }

    /// The playback tracker the session connected with, if any.
    pub fn playback_tracker(&self) -> Option<Arc<RealtimePlaybackTracker>> {
        self.lock()
            .connect_calls
            .last()
            .and_then(|call| call.playback_tracker.clone())
    }

    /// The outputs the session sent back for tool calls, as `(call_id, output)` pairs.
    pub fn tool_outputs(&self) -> Vec<(String, String)> {
        self.lock()
            .sent_events
            .iter()
            .filter_map(|event| {
                event
                    .as_tool_output()
                    .map(|(call, output, _)| (call.call_id.clone(), output.to_string()))
            })
            .collect()
    }

    /// Whether the transport is connected.
    pub fn is_connected(&self) -> bool {
        self.lock().connected
    }

    /// How many times [`RealtimeModel::close`] was called.
    pub fn close_calls(&self) -> usize {
        self.lock().close_calls
    }

    /// Forget every recorded connection and event.
    pub fn clear(&self) {
        let mut state = self.lock();
        state.connect_calls.clear();
        state.sent_events.clear();
        state.retired_responses.clear();
        state.skipped_conditional_sends = 0;
    }
}

#[async_trait]
impl RealtimeModel for FakeRealtimeModel {
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        let mut state = self.lock();
        if state.connected {
            return Err(AgentsError::user("The fake realtime model is connected."));
        }
        state.connect_calls.push(config);
        let (tx, rx) = mpsc::unbounded_channel();
        state.sender = Some(tx);
        state.connected = true;
        Ok(rx)
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()> {
        let mut state = self.lock();
        if !state.connected {
            return Err(AgentsError::user(
                "The fake realtime model is not connected.",
            ));
        }
        state.sent_events.push(event);
        Ok(())
    }

    async fn send_event_if(
        &self,
        event: RealtimeModelSendEvent,
        condition: RealtimeSendCondition,
    ) -> Result<bool> {
        let mut state = self.lock();
        if !state.connected {
            return Err(AgentsError::user(
                "The fake realtime model is not connected.",
            ));
        }
        // The fake transport has no write boundary to race with, so the check is exact.
        if !condition() {
            state.skipped_conditional_sends += 1;
            return Ok(false);
        }
        state.sent_events.push(event);
        Ok(true)
    }

    async fn retire_response_audio(&self, response_id: &str) -> Result<()> {
        self.lock().retired_responses.push(response_id.to_string());
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        let mut state = self.lock();
        state.close_calls += 1;
        state.connected = false;
        state.sender = None;
        Ok(())
    }
}

/// Convenience constructor for a shared fake transport.
pub fn fake_realtime_model() -> Arc<FakeRealtimeModel> {
    Arc::new(FakeRealtimeModel::new())
}

/// How a [`RealtimeStep`] recognizes the outbound event it expects.
pub enum RealtimeSendMatcher {
    /// The event must equal this one.
    Event(Box<RealtimeModelSendEvent>),
    /// The event's kind must equal this name, as [`RealtimeModelSendEvent::event_type`] reports
    /// it: `raw_message`, `user_input`, `audio`, `tool_output`, `interrupt` or `session_update`.
    Kind(&'static str),
    /// The event must satisfy this predicate.
    Predicate(Arc<dyn Fn(&RealtimeModelSendEvent) -> bool + Send + Sync>),
}

impl Clone for RealtimeSendMatcher {
    fn clone(&self) -> Self {
        match self {
            RealtimeSendMatcher::Event(event) => RealtimeSendMatcher::Event(event.clone()),
            RealtimeSendMatcher::Kind(kind) => RealtimeSendMatcher::Kind(kind),
            RealtimeSendMatcher::Predicate(f) => RealtimeSendMatcher::Predicate(f.clone()),
        }
    }
}

impl fmt::Debug for RealtimeSendMatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RealtimeSendMatcher::Event(event) => f.debug_tuple("Event").field(event).finish(),
            RealtimeSendMatcher::Kind(kind) => f.debug_tuple("Kind").field(kind).finish(),
            RealtimeSendMatcher::Predicate(_) => write!(f, "Predicate(..)"),
        }
    }
}

impl RealtimeSendMatcher {
    /// Match any event of this kind.
    pub fn kind(kind: &'static str) -> Self {
        RealtimeSendMatcher::Kind(kind)
    }

    /// Match events satisfying `f`.
    pub fn predicate<F>(f: F) -> Self
    where
        F: Fn(&RealtimeModelSendEvent) -> bool + Send + Sync + 'static,
    {
        RealtimeSendMatcher::Predicate(Arc::new(f))
    }

    /// Whether `event` is what this matcher expects.
    pub fn matches(&self, event: &RealtimeModelSendEvent) -> bool {
        match self {
            RealtimeSendMatcher::Event(expected) => **expected == *event,
            RealtimeSendMatcher::Kind(kind) => event.event_type() == *kind,
            RealtimeSendMatcher::Predicate(f) => f(event),
        }
    }

    /// A short description for assertion messages.
    fn describe(&self) -> String {
        match self {
            // The event itself is left out: scripted exchanges can carry user data.
            RealtimeSendMatcher::Event(_) => "the scripted event".to_string(),
            RealtimeSendMatcher::Kind(kind) => format!("an event of kind {kind:?}"),
            RealtimeSendMatcher::Predicate(_) => "an event matching a predicate".to_string(),
        }
    }
}

impl From<RealtimeModelSendEvent> for RealtimeSendMatcher {
    fn from(value: RealtimeModelSendEvent) -> Self {
        RealtimeSendMatcher::Event(Box::new(value))
    }
}

impl From<&'static str> for RealtimeSendMatcher {
    fn from(value: &'static str) -> Self {
        RealtimeSendMatcher::Kind(value)
    }
}

/// One expected outbound event and what the transport does in response.
#[derive(Debug, Clone)]
pub struct RealtimeStep {
    /// The event this step expects next.
    pub expect: RealtimeSendMatcher,
    /// Events delivered to the session once the expected event arrives.
    pub emit: Vec<RealtimeModelEvent>,
    /// An error returned instead of delivering `emit`.
    pub error: Option<String>,
}

impl RealtimeStep {
    /// A step that expects `expect` and emits nothing.
    pub fn new(expect: impl Into<RealtimeSendMatcher>) -> Self {
        RealtimeStep {
            expect: expect.into(),
            emit: Vec::new(),
            error: None,
        }
    }

    /// Emit these events when the expected event arrives.
    pub fn emitting<I: IntoIterator<Item = RealtimeModelEvent>>(mut self, events: I) -> Self {
        self.emit = events.into_iter().collect();
        self
    }

    /// Fail the send with this message instead of emitting anything.
    pub fn failing(mut self, message: impl Into<String>) -> Self {
        self.error = Some(message.into());
        self
    }

    /// Reject a step that both emits and fails, which the transport cannot do.
    fn validate(&self) -> Result<()> {
        if self.error.is_some() && !self.emit.is_empty() {
            return Err(AgentsError::user(
                "A RealtimeStep cannot define both emit events and an error.",
            ));
        }
        Ok(())
    }
}

/// A scripted transport saw something the script did not allow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealtimeScriptError {
    /// An outbound event did not match the next scripted step.
    UnexpectedSend {
        /// What the session sent, as [`RealtimeModelSendEvent::event_type`] names it.
        actual: String,
        /// What the script expected, or `None` when the script was exhausted.
        expected: Option<String>,
    },
    /// The test finished with scripted steps left over.
    UnconsumedSteps {
        /// How many steps were never reached.
        remaining: usize,
    },
}

impl fmt::Display for RealtimeScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RealtimeScriptError::UnexpectedSend {
                actual,
                expected: Some(expected),
            } => write!(
                f,
                "Unexpected realtime send: expected {expected}, got {actual}."
            ),
            RealtimeScriptError::UnexpectedSend {
                actual,
                expected: None,
            } => write!(
                f,
                "Unexpected realtime send {actual}: no scripted steps remain."
            ),
            RealtimeScriptError::UnconsumedSteps { remaining } => write!(
                f,
                "{remaining} scripted Realtime step(s) were not consumed."
            ),
        }
    }
}

impl std::error::Error for RealtimeScriptError {}

impl From<RealtimeScriptError> for AgentsError {
    fn from(value: RealtimeScriptError) -> Self {
        AgentsError::other(value.to_string())
    }
}

/// A credential-free snapshot of one connection attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeConnectCall {
    /// Whether an API key was supplied, never the key itself.
    pub api_key_provided: bool,
    /// Whether connection headers were supplied, never their values.
    pub headers_provided: bool,
    /// The connection URL with its query string and any userinfo removed.
    pub url: Option<String>,
    /// The settings applied on connect.
    pub initial_model_settings: Option<RealtimeSessionModelSettings>,
    /// The call this connection attached to.
    pub call_id: Option<String>,
    /// Whether a playback tracker was supplied.
    pub playback_tracker_provided: bool,
}

impl RealtimeConnectCall {
    fn from_config(config: &RealtimeModelConfig) -> Self {
        RealtimeConnectCall {
            api_key_provided: config.api_key.is_some(),
            headers_provided: config.headers.is_some(),
            url: config.url.as_deref().map(sanitize_url),
            initial_model_settings: config.initial_model_settings.clone(),
            call_id: config.call_id.clone(),
            playback_tracker_provided: config.playback_tracker.is_some(),
        }
    }
}

/// Strip the query string and any userinfo, which can carry credentials.
fn sanitize_url(url: &str) -> String {
    let without_query = url.split(['?', '#']).next().unwrap_or_default();
    let Some((scheme, rest)) = without_query.split_once("://") else {
        return without_query.to_string();
    };
    let (authority, path) = match rest.split_once('/') {
        Some((authority, path)) => (authority, format!("/{path}")),
        None => (rest, String::new()),
    };
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    format!("{scheme}://{host}{path}")
}

#[derive(Default)]
struct ScriptedState {
    steps: std::collections::VecDeque<RealtimeStep>,
    connect_calls: Vec<RealtimeConnectCall>,
    sent_events: Vec<RealtimeModelSendEvent>,
    sender: Option<mpsc::UnboundedSender<RealtimeModelEvent>>,
    connected: bool,
    closed: bool,
    close_calls: usize,
}

/// A transport that accepts outbound events only in a scripted order.
///
/// Where [`FakeRealtimeModel`] records whatever the session sends and emits whatever a test asks
/// for, this one asserts the exchange: each send must match the next [`RealtimeStep`], which then
/// decides what the transport delivers back, or how it fails. Port of Python's
/// `ScriptedRealtimeModel`.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use openai_agents_sdk::realtime::{
/// #     RealtimeAgent, RealtimeModelEvent, RealtimeRunner, RealtimeStep, ScriptedRealtimeModel,
/// # };
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let model = Arc::new(ScriptedRealtimeModel::new([
///     RealtimeStep::new("user_input").emitting([RealtimeModelEvent::TurnStarted {
///         response_id: Some("resp_1".into()),
///     }]),
/// ]));
/// let agent = RealtimeAgent::<()>::new("Assistant").into_arc();
/// let session = RealtimeRunner::new(agent)
///     .with_model(model.clone())
///     .run(())
///     .await?;
/// session.send_message("Hello").await?;
/// model.assert_complete()?;
/// # Ok(())
/// # }
/// ```
pub struct ScriptedRealtimeModel {
    state: Mutex<ScriptedState>,
    connect_events: Vec<RealtimeModelEvent>,
    connect_error: Option<String>,
    close_error: Option<String>,
    strict: bool,
}

impl fmt::Debug for ScriptedRealtimeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("ScriptedRealtimeModel")
            .field("remaining_steps", &state.steps.len())
            .field("sent_events", &state.sent_events.len())
            .field("strict", &self.strict)
            .finish_non_exhaustive()
    }
}

impl ScriptedRealtimeModel {
    /// A transport expecting `steps`, in order.
    pub fn new<I: IntoIterator<Item = RealtimeStep>>(steps: I) -> Self {
        ScriptedRealtimeModel {
            state: Mutex::new(ScriptedState {
                steps: steps.into_iter().collect(),
                ..Default::default()
            }),
            connect_events: Vec::new(),
            connect_error: None,
            close_error: None,
            strict: true,
        }
    }

    /// Deliver these events as soon as the session connects.
    pub fn with_connect_events<I: IntoIterator<Item = RealtimeModelEvent>>(
        mut self,
        events: I,
    ) -> Self {
        self.connect_events = events.into_iter().collect();
        self
    }

    /// Fail the connection with this message.
    pub fn with_connect_error(mut self, message: impl Into<String>) -> Self {
        self.connect_error = Some(message.into());
        self
    }

    /// Fail `close` with this message.
    pub fn with_close_error(mut self, message: impl Into<String>) -> Self {
        self.close_error = Some(message.into());
        self
    }

    /// Accept sends the script does not cover instead of failing them.
    pub fn lenient(mut self) -> Self {
        self.strict = false;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ScriptedState> {
        self.state.lock().expect("scripted realtime model poisoned")
    }

    /// How many expected sends remain.
    pub fn remaining_steps(&self) -> usize {
        self.lock().steps.len()
    }

    /// Every event the session sent, in order.
    pub fn sent_events(&self) -> Vec<RealtimeModelSendEvent> {
        self.lock().sent_events.clone()
    }

    /// Credential-free snapshots of each connection attempt.
    pub fn connect_calls(&self) -> Vec<RealtimeConnectCall> {
        self.lock().connect_calls.clone()
    }

    /// Whether the model is currently connected.
    pub fn is_connected(&self) -> bool {
        self.lock().connected
    }

    /// Whether the model has been closed.
    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    /// How many times [`RealtimeModel::close`] was called.
    pub fn close_calls(&self) -> usize {
        self.lock().close_calls
    }

    /// Deliver events to the session outside the script.
    pub fn emit(&self, event: RealtimeModelEvent) -> bool {
        let sender = { self.lock().sender.clone() };
        sender.is_some_and(|sender| sender.send(event).is_ok())
    }

    /// Fail when expected sends remain unconsumed.
    pub fn assert_complete(&self) -> std::result::Result<(), RealtimeScriptError> {
        let remaining = self.remaining_steps();
        if remaining > 0 {
            return Err(RealtimeScriptError::UnconsumedSteps { remaining });
        }
        Ok(())
    }

    /// Take the step matching `event`, or report why it is unexpected.
    fn take_step(
        &self,
        event: &RealtimeModelSendEvent,
    ) -> std::result::Result<Option<RealtimeStep>, RealtimeScriptError> {
        let mut state = self.lock();
        let Some(next) = state.steps.front() else {
            if !self.strict {
                return Ok(None);
            }
            return Err(RealtimeScriptError::UnexpectedSend {
                actual: event.event_type().to_string(),
                expected: None,
            });
        };
        if !next.expect.matches(event) {
            if !self.strict {
                return Ok(None);
            }
            return Err(RealtimeScriptError::UnexpectedSend {
                actual: event.event_type().to_string(),
                expected: Some(next.expect.describe()),
            });
        }
        Ok(state.steps.pop_front())
    }
}

#[async_trait]
impl RealtimeModel for ScriptedRealtimeModel {
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        let receiver = {
            let mut state = self.lock();
            if state.connected {
                return Err(AgentsError::user("Already connected"));
            }
            if self.connect_error.is_some() && !self.connect_events.is_empty() {
                return Err(AgentsError::user(
                    "A ScriptedRealtimeModel cannot define both connect events and a connect error.",
                ));
            }
            state
                .connect_calls
                .push(RealtimeConnectCall::from_config(&config));
            if let Some(message) = &self.connect_error {
                return Err(AgentsError::other(message.clone()));
            }
            for step in state.steps.iter() {
                step.validate()?;
            }
            let (sender, receiver) = mpsc::unbounded_channel();
            for event in &self.connect_events {
                let _ = sender.send(event.clone());
            }
            state.sender = Some(sender);
            state.connected = true;
            state.closed = false;
            receiver
        };
        Ok(receiver)
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()> {
        {
            let state = self.lock();
            if !state.connected || state.closed {
                return Err(AgentsError::user(
                    "Cannot send an event while the scripted realtime model is disconnected.",
                ));
            }
        }
        let step = self.take_step(&event)?;
        let sender = {
            let mut state = self.lock();
            state.sent_events.push(event);
            state.sender.clone()
        };
        let Some(step) = step else {
            return Ok(());
        };
        if let Some(message) = step.error {
            return Err(AgentsError::other(message));
        }
        if let Some(sender) = sender {
            for emitted in step.emit {
                let _ = sender.send(emitted);
            }
        }
        Ok(())
    }

    async fn send_event_if(
        &self,
        event: RealtimeModelSendEvent,
        condition: RealtimeSendCondition,
    ) -> Result<bool> {
        if !condition() {
            return Ok(false);
        }
        self.send_event(event).await.map(|()| true)
    }

    async fn close(&self) -> Result<()> {
        {
            let mut state = self.lock();
            state.close_calls += 1;
            if state.closed {
                return Ok(());
            }
            state.connected = false;
            state.closed = true;
            state.sender = None;
        }
        match &self.close_error {
            Some(message) => Err(AgentsError::other(message.clone())),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_url_drops_query_and_userinfo() {
        assert_eq!(
            sanitize_url("wss://user:secret@api.openai.com/v1/realtime?model=gpt-realtime-2.1"),
            "wss://api.openai.com/v1/realtime"
        );
        assert_eq!(sanitize_url("ws://127.0.0.1:8080"), "ws://127.0.0.1:8080");
        assert_eq!(sanitize_url("not a url"), "not a url");
    }

    #[test]
    fn matchers_recognize_events_by_value_kind_and_predicate() {
        let event = RealtimeModelSendEvent::interrupt();
        assert!(RealtimeSendMatcher::from(event.clone()).matches(&event));
        assert!(RealtimeSendMatcher::kind("interrupt").matches(&event));
        assert!(!RealtimeSendMatcher::kind("audio").matches(&event));
        assert!(
            RealtimeSendMatcher::predicate(|event| event
                .as_interrupt()
                .is_some_and(|interrupt| interrupt.force_response_cancel))
            .matches(&event)
        );
    }

    #[test]
    fn a_step_cannot_both_emit_and_fail() {
        let step = RealtimeStep::new("audio")
            .emitting([RealtimeModelEvent::EndOfStream])
            .failing("boom");
        assert!(step.validate().is_err());
        assert!(
            RealtimeStep::new("audio")
                .failing("boom")
                .validate()
                .is_ok()
        );
    }
}
