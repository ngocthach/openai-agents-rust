//! A deterministic in-memory transport for tests. Port of `agents.realtime.testing`.

use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::{AgentsError, Result};
use crate::realtime::config::RealtimeSessionModelSettings;
use crate::realtime::model::{RealtimeModel, RealtimeModelConfig, RealtimeModelEvents};
use crate::realtime::model_events::RealtimeModelEvent;
use crate::realtime::model_inputs::RealtimeModelSendEvent;

#[derive(Default)]
struct FakeState {
    connect_calls: Vec<RealtimeModelConfig>,
    sent_events: Vec<RealtimeModelSendEvent>,
    sender: Option<mpsc::UnboundedSender<RealtimeModelEvent>>,
    connected: bool,
    close_calls: usize,
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
