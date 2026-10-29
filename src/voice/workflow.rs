//! Voice workflows: the code between transcription and speech. Mirrors
//! `agents.voice.workflow`.

use std::fmt;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use futures::stream::BoxStream;

use crate::agent::Agent;
use crate::error::Result;
use crate::items::InputItem;
use crate::result::RunResultStreaming;
use crate::run::{RunOptions, Runner};

/// A stream of text fragments to be spoken.
pub type TextStream<'a> = BoxStream<'a, Result<String>>;

/// A voice workflow: it receives a transcription and yields the text to speak.
///
/// In most cases the implementation creates [`Agent`]s and runs them with
/// [`Runner::run_streamed`], forwarding the text deltas with
/// [`VoiceWorkflowHelper::stream_text_from`]. For a single starting agent and no custom logic,
/// use [`SingleAgentVoiceWorkflow`].
pub trait VoiceWorkflowBase: Send + Sync + fmt::Debug {
    /// Run one turn of the workflow for `transcription`.
    fn run(&self, transcription: String) -> TextStream<'_>;

    /// Text spoken before any user input, e.g. a greeting. Defaults to nothing.
    fn on_start(&self) -> TextStream<'_> {
        Box::pin(futures::stream::empty())
    }
}

/// Helpers for extracting spoken text from a streamed run.
#[derive(Debug, Clone, Copy)]
pub struct VoiceWorkflowHelper;

impl VoiceWorkflowHelper {
    /// Yield the text deltas of a streamed run.
    ///
    /// The run's result is dropped when the stream ends; [`SingleAgentVoiceWorkflow`] keeps it
    /// instead so it can carry the history to the next turn.
    pub fn stream_text_from<C: Send + Sync + 'static>(
        result: RunResultStreaming<C>,
    ) -> TextStream<'static> {
        Box::pin(futures::stream::unfold(Some(result), |state| async move {
            let mut result = state?;
            match next_text_delta(&mut result).await {
                Some(Ok(text)) => Some((Ok(text), Some(result))),
                Some(Err(error)) => Some((Err(error), None)),
                None => None,
            }
        }))
    }
}

/// The next text delta of a streamed run, skipping every other event.
async fn next_text_delta<C: Send + Sync + 'static>(
    result: &mut RunResultStreaming<C>,
) -> Option<Result<String>> {
    loop {
        let event = {
            let mut events = Box::pin(result.stream_events());
            events.next().await
        };
        match event {
            Some(Ok(event)) => {
                if let Some(delta) = event.text_delta() {
                    return Some(Ok(delta.to_string()));
                }
            }
            Some(Err(error)) => return Some(Err(error)),
            None => return None,
        }
    }
}

/// Called when a [`SingleAgentVoiceWorkflow`] runs.
pub trait SingleAgentWorkflowCallbacks: Send + Sync + fmt::Debug {
    /// Called with the transcription at the start of every turn.
    fn on_run(&self, transcription: &str);
}

/// The agent and history carried from turn to turn.
struct WorkflowState<C> {
    input_history: Vec<InputItem>,
    current_agent: Arc<Agent<C>>,
}

/// A workflow that runs a single agent, adding every transcription and result to the history.
///
/// For more complex workflows (several runs, custom history, custom configuration) implement
/// [`VoiceWorkflowBase`] directly.
pub struct SingleAgentVoiceWorkflow<C> {
    state: Arc<Mutex<WorkflowState<C>>>,
    callbacks: Option<Arc<dyn SingleAgentWorkflowCallbacks>>,
    context: C,
}

impl<C> fmt::Debug for SingleAgentVoiceWorkflow<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.lock().expect("poisoned");
        f.debug_struct("SingleAgentVoiceWorkflow")
            .field("current_agent", &state.current_agent.name)
            .field("history", &state.input_history.len())
            .finish_non_exhaustive()
    }
}

impl<C: Clone> Clone for SingleAgentVoiceWorkflow<C> {
    fn clone(&self) -> Self {
        SingleAgentVoiceWorkflow {
            state: self.state.clone(),
            callbacks: self.callbacks.clone(),
            context: self.context.clone(),
        }
    }
}

impl<C: Default + Clone + Send + Sync + 'static> SingleAgentVoiceWorkflow<C> {
    /// A workflow running `agent` with a default app context.
    pub fn new(agent: impl crate::run::IntoAgentArc<C>) -> Self {
        Self::with_context(agent, C::default())
    }
}

impl<C: Clone + Send + Sync + 'static> SingleAgentVoiceWorkflow<C> {
    /// A workflow running `agent` with an explicit app context, cloned into every run.
    pub fn with_context(agent: impl crate::run::IntoAgentArc<C>, context: C) -> Self {
        SingleAgentVoiceWorkflow {
            state: Arc::new(Mutex::new(WorkflowState {
                input_history: Vec::new(),
                current_agent: agent.into_agent_arc(),
            })),
            callbacks: None,
            context,
        }
    }

    /// Set the callbacks invoked on every turn.
    pub fn with_callbacks(mut self, callbacks: Arc<dyn SingleAgentWorkflowCallbacks>) -> Self {
        self.callbacks = Some(callbacks);
        self
    }

    /// The agent that will handle the next turn.
    pub fn current_agent(&self) -> Arc<Agent<C>> {
        self.state.lock().expect("poisoned").current_agent.clone()
    }

    /// The input history carried to the next turn.
    pub fn input_history(&self) -> Vec<InputItem> {
        self.state.lock().expect("poisoned").input_history.clone()
    }
}

impl<C: Clone + Send + Sync + 'static> VoiceWorkflowBase for SingleAgentVoiceWorkflow<C> {
    fn run(&self, transcription: String) -> TextStream<'_> {
        if let Some(callbacks) = &self.callbacks {
            callbacks.on_run(&transcription);
        }
        let (agent, input) = {
            let mut state = self.state.lock().expect("poisoned");
            state.input_history.push(InputItem::user(transcription));
            (state.current_agent.clone(), state.input_history.clone())
        };
        let result =
            Runner::run_streamed_with_options(agent, input, RunOptions::new(self.context.clone()));
        let state = self.state.clone();
        Box::pin(futures::stream::unfold(
            Some(Box::new(result)),
            move |pending| {
                let state = state.clone();
                async move {
                    let mut result = pending?;
                    match next_text_delta(&mut result).await {
                        Some(Ok(text)) => Some((Ok(text), Some(result))),
                        Some(Err(error)) => Some((Err(error), None)),
                        None => {
                            // The run finished: carry its history and last agent to the next turn.
                            if let Some(run) = result.result() {
                                let mut state = state.lock().expect("poisoned");
                                state.input_history = run.to_input_list();
                                state.current_agent = run.last_agent.clone();
                            }
                            None
                        }
                    }
                }
            },
        ))
    }
}
