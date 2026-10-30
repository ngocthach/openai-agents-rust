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
use crate::util::BoxFuture;

/// How [`RunResult::to_input_list_with`] builds the replay list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToInputListMode {
    /// Keep the full converted history of `new_items` (the historical behavior).
    #[default]
    PreserveAll,
    /// Prefer the canonical continuation input when a handoff filter or nested history
    /// rewrote model history; identical to [`ToInputListMode::PreserveAll`] otherwise.
    Normalized,
}

/// Immutable metadata about a nested `Agent::as_tool` invocation.
/// Mirrors `agents.result.AgentToolInvocation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentToolInvocation {
    /// The nested tool name exposed to the model.
    pub tool_name: String,
    /// The tool call id of the nested invocation.
    pub tool_call_id: String,
    /// The raw JSON arguments of the nested invocation.
    pub tool_arguments: String,
}

/// Escape C0, DEL and C1 controls, keeping tabs and line breaks.
/// Port of `_escape_terminal_controls`.
pub(crate) fn escape_terminal_controls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let code = c as u32;
        let escape =
            (code <= 0x08) || (0x0b..=0x1f).contains(&code) || (0x7f..=0x9f).contains(&code);
        if escape {
            out.push_str(&format!("\\x{code:02x}"));
        } else {
            out.push(c);
        }
    }
    out
}

fn indent(text: &str, level: usize) -> String {
    let normalized = escape_terminal_controls(&text.replace("\r\n", "\n"));
    let pad = "  ".repeat(level);
    normalized
        .lines()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn final_output_str(final_output: Option<&Value>) -> String {
    match final_output {
        None => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

fn final_output_kind(final_output: Option<&Value>) -> &'static str {
    match final_output {
        None => "NoneType",
        Some(Value::String(_)) => "str",
        Some(Value::Bool(_)) => "bool",
        Some(Value::Number(_)) => "number",
        Some(Value::Array(_)) => "list",
        Some(Value::Object(_)) => "dict",
        Some(Value::Null) => "NoneType",
    }
}

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
    /// Metadata of the nested `Agent::as_tool` invocation that produced this result, when it
    /// was produced by one.
    pub agent_tool_invocation: Option<AgentToolInvocation>,
    /// Items as the model saw them, which differ from `new_items` once a handoff input filter
    /// or nested handoff history rewrote model history.
    pub(crate) model_input_items: Vec<RunItem>,
    /// Whether model history diverged from the public history during this run.
    pub(crate) replay_from_model_input_items: bool,
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
        self.to_input_list_with(ToInputListMode::PreserveAll)
    }

    /// Input items for the next turn, choosing which history view to replay.
    /// Mirrors `RunResultBase.to_input_list(mode=...)`.
    pub fn to_input_list_with(&self, mode: ToInputListMode) -> Vec<InputItem> {
        let source = match mode {
            ToInputListMode::Normalized if self.replay_from_model_input_items => {
                &self.model_input_items
            }
            _ => &self.new_items,
        };
        let mut out = self.input.clone();
        out.extend(
            source
                .iter()
                .filter(|i| !matches!(i.kind, RunItemKind::ToolApproval { .. }))
                .map(RunItem::to_input_item),
        );
        out
    }

    /// A human-readable summary. Port of `pretty_print_result`.
    pub fn pretty_print(&self) -> String {
        let mut out = String::from("RunResult:");
        out.push_str(&format!(
            "\n- Last agent: Agent(name=\"{}\", ...)",
            self.last_agent.name
        ));
        out.push_str(&format!(
            "\n- Final output ({}):\n{}",
            final_output_kind(self.final_output.as_ref()),
            indent(&final_output_str(self.final_output.as_ref()), 2)
        ));
        out.push_str(&format!("\n- {} new item(s)", self.new_items.len()));
        out.push_str(&format!("\n- {} raw response(s)", self.raw_responses.len()));
        out.push_str(&format!(
            "\n- {} input guardrail result(s)",
            self.input_guardrail_results.len()
        ));
        out.push_str(&format!(
            "\n- {} output guardrail result(s)",
            self.output_guardrail_results.len()
        ));
        out.push_str("\n(See `RunResult` for more details)");
        escape_terminal_controls(&out)
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

/// How a streamed run ended, handed to an
/// [`on_complete`](RunResultStreaming::with_on_complete) hook.
///
/// Carries an owned snapshot of the run rather than the [`RunResult`] itself, so the hook can
/// outlive the stream and record what happened without borrowing it.
#[derive(Debug, Clone, Default)]
pub struct StreamCompletion {
    /// Whether the run produced a result rather than an error.
    pub succeeded: bool,
    /// Number of approval interruptions the run paused on.
    pub interruptions: usize,
    /// The input the run started from.
    pub input: Vec<InputItem>,
    /// Items generated during the run.
    pub new_items: Vec<RunItem>,
    /// The final output, when the run produced one.
    pub final_output: Option<Value>,
}

impl StreamCompletion {
    /// Whether the run paused waiting for approvals.
    pub fn is_suspended(&self) -> bool {
        self.interruptions > 0
    }
}

/// An async callback run once a streamed run finishes.
///
/// Used to release resources a streamed run borrowed for its whole lifetime, such as a sandbox
/// session, which cannot be dropped synchronously.
pub type StreamCompletionHook =
    Arc<dyn Fn(StreamCompletion) -> BoxFuture<'static, ()> + Send + Sync>;

/// A streamed run. Consume [`RunResultStreaming::stream_events`] until it ends; afterwards the
/// result fields are available.
pub struct RunResultStreaming<C> {
    rx: mpsc::UnboundedReceiver<StreamMsg<C>>,
    result: Option<RunResult<C>>,
    current_agent: Arc<Agent<C>>,
    cancel_after_turn: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    complete: bool,
    on_complete: Option<StreamCompletionHook>,
    max_turns: Option<usize>,
    run_loop_error: Option<String>,
}

impl<C> fmt::Debug for RunResultStreaming<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunResultStreaming")
            .field("current_agent", &self.current_agent.name)
            .field("is_complete", &self.complete)
            .field("has_on_complete", &self.on_complete.is_some())
            .finish_non_exhaustive()
    }
}

impl<C> fmt::Display for RunResult<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pretty_print())
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
            on_complete: None,
            max_turns: None,
            run_loop_error: None,
        }
    }

    /// Record the turn limit of the run, so `max_turns()` can report it before completion.
    pub(crate) fn set_max_turns(&mut self, max_turns: Option<usize>) {
        self.max_turns = max_turns;
    }

    /// Run `hook` once, when the stream finishes.
    ///
    /// The hook fires from [`stream_events`](Self::stream_events) when the run ends, and from
    /// [`finish`](Self::finish). A run abandoned without draining its stream never fires it, so
    /// a caller that cancels should still call `finish`.
    pub fn with_on_complete(mut self, hook: StreamCompletionHook) -> Self {
        self.on_complete = Some(hook);
        self
    }

    /// Set the completion hook in place. See [`with_on_complete`](Self::with_on_complete).
    pub fn set_on_complete(&mut self, hook: StreamCompletionHook) {
        self.on_complete = Some(hook);
    }

    /// Drain any remaining events and run the completion hook.
    ///
    /// Returns the run error when the run failed. Calling this after the stream was already
    /// drained only runs the hook, and the hook runs at most once.
    pub async fn finish(&mut self) -> Result<()> {
        let mut error = None;
        if !self.complete {
            let mut stream = std::pin::pin!(self.stream_events());
            while let Some(event) = futures::StreamExt::next(&mut stream).await {
                if let Err(e) = event {
                    error = Some(e);
                }
            }
        } else {
            self.run_completion_hook().await;
        }
        match error {
            None => Ok(()),
            Some(error) => Err(error),
        }
    }

    /// Run the completion hook, at most once.
    async fn run_completion_hook(&mut self) {
        if let Some(hook) = self.on_complete.take() {
            let completion = match self.result.as_ref() {
                None => StreamCompletion::default(),
                Some(result) => StreamCompletion {
                    succeeded: true,
                    interruptions: result.interruptions.len(),
                    input: result.input.clone(),
                    new_items: result.new_items.clone(),
                    final_output: result.final_output.clone(),
                },
            };
            hook(completion).await;
        }
    }

    /// Stream events until the run completes. A run error is yielded as the last item.
    ///
    /// When the run ends, any hook installed with
    /// [`with_on_complete`](Self::with_on_complete) runs before the stream terminates.
    pub fn stream_events(&mut self) -> impl Stream<Item = Result<StreamEvent<C>>> + Send + '_ {
        futures::stream::unfold(self, |this| async move {
            if this.complete {
                this.run_completion_hook().await;
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
                            this.run_completion_hook().await;
                            None
                        }
                        Err(e) => {
                            this.run_loop_error = Some(e.to_string());
                            // Run the hook before yielding the error: nothing obliges a consumer
                            // to poll again after one, and `while let Some(ev) = next().await`
                            // with `ev?` returns immediately. The hook is a no-op if the
                            // consumer does poll again.
                            this.run_completion_hook().await;
                            Some((Err(e), this))
                        }
                    }
                }
                None => {
                    this.complete = true;
                    this.run_completion_hook().await;
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

    /// Model turns completed so far.
    pub fn current_turn(&self) -> usize {
        self.result
            .as_ref()
            .map(|r| r.state.current_turn)
            .unwrap_or(0)
    }

    /// The turn limit of the run, when one is set.
    pub fn max_turns(&self) -> Option<usize> {
        self.max_turns
    }

    /// The message of the error the run loop ended with, when it failed.
    pub fn run_loop_error(&self) -> Option<&str> {
        self.run_loop_error.as_deref()
    }

    /// A human-readable summary. Port of `pretty_print_run_result_streaming`.
    pub fn pretty_print(&self) -> String {
        let final_output = self.final_output();
        let mut out = String::from("RunResultStreaming:");
        out.push_str(&format!(
            "\n- Current agent: Agent(name=\"{}\", ...)",
            self.current_agent.name
        ));
        out.push_str(&format!("\n- Current turn: {}", self.current_turn()));
        out.push_str(&format!(
            "\n- Max turns: {}",
            match self.max_turns {
                Some(m) => m.to_string(),
                None => "None".to_string(),
            }
        ));
        out.push_str(&format!("\n- Is complete: {}", self.complete));
        out.push_str(&format!(
            "\n- Final output ({}):\n{}",
            final_output_kind(final_output),
            indent(&final_output_str(final_output), 2)
        ));
        let (items, responses, inputs, outputs) = match self.result.as_ref() {
            Some(r) => (
                r.new_items.len(),
                r.raw_responses.len(),
                r.input_guardrail_results.len(),
                r.output_guardrail_results.len(),
            ),
            None => (0, 0, 0, 0),
        };
        out.push_str(&format!("\n- {items} new item(s)"));
        out.push_str(&format!("\n- {responses} raw response(s)"));
        out.push_str(&format!("\n- {inputs} input guardrail result(s)"));
        out.push_str(&format!("\n- {outputs} output guardrail result(s)"));
        out.push_str("\n(See `RunResultStreaming` for more details)");
        escape_terminal_controls(&out)
    }
}

impl<C: Send + Sync + 'static> fmt::Display for RunResultStreaming<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pretty_print())
    }
}
