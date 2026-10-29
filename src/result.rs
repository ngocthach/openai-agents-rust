//! Run results. Mirrors `agents.result`.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::Stream;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{InputItem, ModelResponse, RunItem, RunItemKind, ToolApprovalItem};
use crate::run_context::RunContext;
use crate::run_state::RunState;
use crate::stream_events::StreamEvent;
use crate::tool_guardrails::{ToolInputGuardrailResult, ToolOutputGuardrailResult};
use crate::usage::Usage;

/// The result of a completed (or paused) run.
pub struct RunResult<C> {
    /// The input of the run (after any handoff input filter rewrites).
    pub input: Vec<InputItem>,
    /// Items generated during the run.
    pub new_items: Vec<RunItem>,
    /// Raw model responses.
    pub raw_responses: Vec<ModelResponse>,
    /// The final output: a string for text agents, the parsed JSON for structured output, or
    /// `None` when the run paused for approvals.
    pub final_output: Option<Value>,
    /// The last agent that ran.
    pub last_agent: Arc<Agent<C>>,
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
    pub tool_input_guardrail_results: Vec<ToolInputGuardrailResult>,
    pub tool_output_guardrail_results: Vec<ToolOutputGuardrailResult>,
    /// The run context (app context, usage, approvals).
    pub context_wrapper: RunContext<C>,
    /// Pending approvals.
    pub interruptions: Vec<ToolApprovalItem>,
    pub(crate) state: RunState,
}

impl<C> fmt::Debug for RunResult<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunResult")
            .field("final_output", &self.final_output)
            .field("last_agent", &self.last_agent.name)
            .field("new_items", &self.new_items.len())
            .field("interruptions", &self.interruptions)
            .finish_non_exhaustive()
    }
}

impl<C> RunResult<C> {
    /// The final output as text: strings as-is, other JSON values serialized.
    pub fn final_output_text(&self) -> Option<String> {
        self.final_output.as_ref().map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    }

    /// Deserialize the final output into `T` (for structured output agents).
    pub fn final_output_as<T: DeserializeOwned>(&self) -> Result<T> {
        let v = self
            .final_output
            .clone()
            .ok_or_else(|| AgentsError::user("The run has no final output"))?;
        Ok(serde_json::from_value(v)?)
    }

    /// Input items for the next turn: the run input plus generated items.
    pub fn to_input_list(&self) -> Vec<InputItem> {
        let mut out = self.input.clone();
        out.extend(
            self.new_items
                .iter()
                .filter(|i| !matches!(i.kind, RunItemKind::ToolApproval { .. }))
                .map(RunItem::to_input_item),
        );
        out
    }

    /// The id of the last model response (for `previous_response_id` chaining).
    pub fn last_response_id(&self) -> Option<String> {
        self.raw_responses
            .last()
            .and_then(|r| r.response_id.clone())
    }

    /// Usage accumulated by the run.
    pub fn usage(&self) -> Usage {
        self.context_wrapper.usage()
    }

    /// A resumable snapshot with an independent copy of the usage so far.
    pub fn to_state(&self) -> RunState {
        let mut s = self.state.clone();
        s.usage = self.context_wrapper.usage();
        s.approvals = self.context_wrapper.export_approvals();
        s.nested_agent_tool_states = self.context_wrapper.nested_states();
        s
    }
}

/// Messages sent from the background run task to a streamed result.
// Events are the common case; boxing each one would cost an allocation per event.
#[allow(clippy::large_enum_variant)]
pub(crate) enum StreamMsg<C> {
    Event(StreamEvent<C>),
    Done(Box<Result<RunResult<C>>>),
}

/// How to cancel a streamed run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelMode {
    /// Stop immediately.
    Immediate,
    /// Let the current turn finish, then stop.
    AfterTurn,
}

/// A streamed run. Consume [`RunResultStreaming::stream_events`] until it ends; afterwards the
/// result fields are available.
pub struct RunResultStreaming<C> {
    rx: mpsc::UnboundedReceiver<StreamMsg<C>>,
    result: Option<RunResult<C>>,
    current_agent: Arc<Agent<C>>,
    cancel_after_turn: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    complete: bool,
}

impl<C> fmt::Debug for RunResultStreaming<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunResultStreaming")
            .field("current_agent", &self.current_agent.name)
            .field("is_complete", &self.complete)
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> RunResultStreaming<C> {
    pub(crate) fn new(
        rx: mpsc::UnboundedReceiver<StreamMsg<C>>,
        current_agent: Arc<Agent<C>>,
        cancel_after_turn: Arc<AtomicBool>,
        handle: JoinHandle<()>,
    ) -> Self {
        RunResultStreaming {
            rx,
            result: None,
            current_agent,
            cancel_after_turn,
            handle: Some(handle),
            complete: false,
        }
    }

    /// Stream events until the run completes. A run error is yielded as the last item.
    pub fn stream_events(&mut self) -> impl Stream<Item = Result<StreamEvent<C>>> + Send + '_ {
        futures::stream::unfold(self, |this| async move {
            if this.complete {
                return None;
            }
            match this.rx.recv().await {
                Some(StreamMsg::Event(ev)) => {
                    if let StreamEvent::AgentUpdated { new_agent } = &ev {
                        this.current_agent = new_agent.clone();
                    }
                    Some((Ok(ev), this))
                }
                Some(StreamMsg::Done(res)) => {
                    this.complete = true;
                    match *res {
                        Ok(r) => {
                            this.current_agent = r.last_agent.clone();
                            this.result = Some(r);
                            None
                        }
                        Err(e) => Some((Err(e), this)),
                    }
                }
                None => {
                    this.complete = true;
                    None
                }
            }
        })
    }

    /// Cancel the run.
    pub fn cancel(&mut self, mode: CancelMode) {
        match mode {
            CancelMode::AfterTurn => self.cancel_after_turn.store(true, Ordering::SeqCst),
            CancelMode::Immediate => {
                if let Some(h) = self.handle.take() {
                    h.abort();
                }
                self.complete = true;
            }
        }
    }

    /// Whether the run has finished.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// The agent currently running (updates as handoffs happen).
    pub fn current_agent(&self) -> &Arc<Agent<C>> {
        &self.current_agent
    }

    /// The completed result, once the stream has been fully consumed.
    pub fn result(&self) -> Option<&RunResult<C>> {
        self.result.as_ref()
    }

    /// Take the completed result.
    pub fn into_result(self) -> Option<RunResult<C>> {
        self.result
    }

    /// The final output, once complete.
    pub fn final_output(&self) -> Option<&Value> {
        self.result.as_ref().and_then(|r| r.final_output.as_ref())
    }

    /// The final output as text, once complete.
    pub fn final_output_text(&self) -> Option<String> {
        self.result.as_ref().and_then(RunResult::final_output_text)
    }

    /// Pending approvals, once complete.
    pub fn interruptions(&self) -> Vec<ToolApprovalItem> {
        self.result
            .as_ref()
            .map(|r| r.interruptions.clone())
            .unwrap_or_default()
    }

    /// A resumable snapshot, once complete.
    pub fn to_state(&self) -> Option<RunState> {
        self.result.as_ref().map(RunResult::to_state)
    }
}
