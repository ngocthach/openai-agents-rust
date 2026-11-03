//! Test helpers: a scripted model that returns queued responses. Mirrors `agents.testing`.
//!
//! ```
//! use openai_agents_sdk::testing::{FakeModel, assistant_message};
//! use openai_agents_sdk::{Agent, Runner};
//! use std::sync::Arc;
//!
//! # tokio_test_block(async {
//! let model = Arc::new(FakeModel::new());
//! model.add_turn(vec![assistant_message("Hello!")]);
//! let agent = Agent::<()>::new("Assistant").with_model_instance(model.clone());
//! let result = Runner::run(&agent, "Hi").await.unwrap();
//! assert_eq!(result.final_output_text().as_deref(), Some("Hello!"));
//! # });
//! # fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)
//! # }
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::error::{AgentsError, Result};
use crate::items::{FunctionCall, MessageOutput, ModelResponse, OutputItem};
use crate::models::interface::{Model, ModelRequest, ResponseEventStream};
use crate::usage::Usage;

type Responder = Arc<dyn Fn(&ModelRequest) -> Result<Vec<OutputItem>> + Send + Sync>;

/// One scripted model call.
#[derive(Clone)]
pub enum ModelStep {
    /// Return these output items.
    Output(Vec<OutputItem>),
    /// Fail with this error message (as `AgentsError::Other`).
    Error(String),
    /// Fail with an API error of this HTTP status.
    ApiError(u16),
    /// Compute the output from the request.
    Respond(Responder),
}

impl fmt::Debug for ModelStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelStep::Output(o) => f.debug_tuple("Output").field(o).finish(),
            ModelStep::Error(e) => f.debug_tuple("Error").field(e).finish(),
            ModelStep::ApiError(s) => f.debug_tuple("ApiError").field(s).finish(),
            ModelStep::Respond(_) => write!(f, "Respond(..)"),
        }
    }
}

/// A model returning scripted outputs, recording every request it receives.
#[derive(Debug, Default)]
pub struct FakeModel {
    steps: Mutex<VecDeque<ModelStep>>,
    calls: Mutex<Vec<ModelRequest>>,
    usage: Mutex<Option<Usage>>,
    counter: Mutex<u64>,
}

impl FakeModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a step.
    pub fn enqueue(&self, step: ModelStep) {
        self.steps.lock().expect("poisoned").push_back(step);
    }

    /// Queue a turn returning these output items.
    pub fn add_turn(&self, output: Vec<OutputItem>) {
        self.enqueue(ModelStep::Output(output));
    }

    /// Queue several turns.
    pub fn add_turns(&self, turns: Vec<Vec<OutputItem>>) {
        for t in turns {
            self.add_turn(t);
        }
    }

    /// Queue an error.
    pub fn add_error(&self, message: impl Into<String>) {
        self.enqueue(ModelStep::Error(message.into()));
    }

    /// Queue a turn computed from the request.
    pub fn add_responder<F>(&self, f: F)
    where
        F: Fn(&ModelRequest) -> Result<Vec<OutputItem>> + Send + Sync + 'static,
    {
        self.enqueue(ModelStep::Respond(Arc::new(f)));
    }

    /// Usage reported for every response (default: 1 request, 10 input, 5 output tokens).
    pub fn set_usage(&self, usage: Usage) {
        *self.usage.lock().expect("poisoned") = Some(usage);
    }

    /// Requests received so far.
    pub fn calls(&self) -> Vec<ModelRequest> {
        self.calls.lock().expect("poisoned").clone()
    }

    /// The last request received.
    pub fn last_call(&self) -> Option<ModelRequest> {
        self.calls.lock().expect("poisoned").last().cloned()
    }

    /// Steps not consumed yet.
    pub fn remaining_steps(&self) -> usize {
        self.steps.lock().expect("poisoned").len()
    }

    fn next(&self, request: &ModelRequest) -> Result<ModelResponse> {
        self.calls.lock().expect("poisoned").push(request.clone());
        let step = self
            .steps
            .lock()
            .expect("poisoned")
            .pop_front()
            .ok_or_else(|| {
                AgentsError::other(
                    "FakeModel received an unexpected call: no scripted steps remain",
                )
            })?;
        let output = match step {
            ModelStep::Output(o) => o,
            ModelStep::Error(e) => return Err(AgentsError::Other(e)),
            ModelStep::ApiError(status) => {
                return Err(AgentsError::ApiError {
                    status,
                    message: format!("scripted API error {status}"),
                    body: None,
                    request_id: None,
                    headers: None,
                });
            }
            ModelStep::Respond(f) => f(request)?,
        };
        let n = {
            let mut c = self.counter.lock().expect("poisoned");
            *c += 1;
            *c
        };
        let usage = self
            .usage
            .lock()
            .expect("poisoned")
            .clone()
            .unwrap_or_else(|| Usage::single(10, 5));
        Ok(ModelResponse {
            output,
            usage,
            response_id: Some(format!("resp_fake_{n}")),
            request_id: None,
            raw_usage: None,
        })
    }
}

/// Build Responses-style stream events for a response.
pub fn response_to_stream_events(response: &ModelResponse) -> Vec<Value> {
    let mut events = vec![
        json!({"type": "response.created", "response": {"id": response.response_id, "status": "in_progress", "output": []}}),
    ];
    for (i, item) in response.output.iter().enumerate() {
        let v = item.to_value();
        events.push(json!({"type": "response.output_item.added", "output_index": i, "item": v}));
        if let OutputItem::Message(m) = item {
            let text = m.text_content();
            if !text.is_empty() {
                events.push(json!({
                    "type": "response.output_text.delta",
                    "item_id": m.id, "output_index": i, "content_index": 0, "delta": text,
                }));
            }
        }
        events.push(json!({"type": "response.output_item.done", "output_index": i, "item": v}));
    }
    events.push(json!({
        "type": "response.completed",
        "response": {
            "id": response.response_id,
            "status": "completed",
            "output": response.output.iter().map(OutputItem::to_value).collect::<Vec<_>>(),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                // The streamed usage carries the same token details as the non-streamed response.
                "input_tokens_details": {
                    "cached_tokens": response.usage.input_tokens_details.cached_tokens,
                    "cache_write_tokens": response.usage.input_tokens_details.cache_write_tokens,
                },
                "output_tokens_details": {
                    "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
                },
            }
        }
    }));
    events
}

#[async_trait]
impl Model for FakeModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.next(&request)
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let response = self.next(&request)?;
        let events = response_to_stream_events(&response);
        Ok(futures::stream::iter(events.into_iter().map(Ok)).boxed())
    }
}

/// An assistant message output item.
pub fn assistant_message(text: impl Into<String>) -> OutputItem {
    OutputItem::Message(MessageOutput::text(text))
}

/// A refusal message output item.
pub fn refusal_message(refusal: impl Into<String>) -> OutputItem {
    let mut m = MessageOutput::text("");
    m.content = vec![crate::items::MessageContent::Refusal {
        refusal: refusal.into(),
    }];
    OutputItem::Message(m)
}

/// A function call output item with a generated call id.
pub fn function_call(name: impl Into<String>, arguments: impl Into<String>) -> OutputItem {
    OutputItem::FunctionCall(FunctionCall::new(name, arguments))
}

/// A function call output item with an explicit call id.
pub fn function_call_with_id(
    name: impl Into<String>,
    arguments: impl Into<String>,
    call_id: impl Into<String>,
) -> OutputItem {
    let mut fc = FunctionCall::new(name, arguments);
    fc.call_id = call_id.into();
    OutputItem::FunctionCall(fc)
}

/// The item id [`assistant_message_with_id`] and the scripted helpers use by default.
pub const DEFAULT_SCRIPTED_ITEM_ID: &str = "scripted-message";

/// An assistant message output item with an explicit item id.
pub fn assistant_message_with_id(
    text: impl Into<String>,
    item_id: impl Into<String>,
) -> OutputItem {
    let mut m = MessageOutput::text(text);
    m.id = item_id.into();
    m.status = Some("completed".to_string());
    OutputItem::Message(m)
}

/// A function call output item with an explicit call id, item id and tool namespace.
///
/// `item_id` defaults to `call_id`, as the Responses API does for a call the model emitted in
/// one piece. `namespace` is written to the item only when given.
pub fn function_call_full(
    name: impl Into<String>,
    arguments: impl Into<String>,
    call_id: impl Into<String>,
    item_id: Option<String>,
    namespace: Option<String>,
) -> OutputItem {
    let call_id = call_id.into();
    let mut fc = FunctionCall::new(name, arguments);
    fc.id = Some(item_id.unwrap_or_else(|| call_id.clone()));
    fc.call_id = call_id;
    if let Some(namespace) = namespace {
        fc.extra.insert("namespace".into(), json!(namespace));
    }
    OutputItem::FunctionCall(fc)
}

// ---------------------------------------------------------------------------
// ScriptedModel: a strictly-matched provider-neutral model. Mirrors `agents.testing.model`.
// ---------------------------------------------------------------------------

/// Why a scripted model step was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStepReason {
    /// A field had a value the step cannot carry.
    InvalidInput,
    /// An unknown field was supplied.
    UnsupportedField,
    /// The error field was not usable.
    InvalidError,
    /// The responder was not usable.
    InvalidResponder,
    /// The supplied stream events were not usable.
    InvalidStreamEvents,
    /// More than one of error, responder and stream events was set.
    ConflictingOutcomes,
    /// Retry advice was given without an error to attach it to.
    InvalidRetryAdvice,
}

impl fmt::Display for ModelStepReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            ModelStepReason::InvalidInput => "invalid_input",
            ModelStepReason::UnsupportedField => "unsupported_field",
            ModelStepReason::InvalidError => "invalid_error",
            ModelStepReason::InvalidResponder => "invalid_responder",
            ModelStepReason::InvalidStreamEvents => "invalid_stream_events",
            ModelStepReason::ConflictingOutcomes => "conflicting_outcomes",
            ModelStepReason::InvalidRetryAdvice => "invalid_retry_advice",
        };
        f.write_str(text)
    }
}

/// An invalid or incompletely consumed model script.
#[derive(Debug, Clone)]
pub enum ModelScriptError {
    /// A step was rejected before it entered the script queue.
    InvalidModelStep {
        /// What was wrong with the step.
        reason: ModelStepReason,
        /// Position of the step in the batch it was added with.
        input_index: usize,
        /// Human-readable detail.
        detail: String,
    },
    /// The model was called after every configured step had been consumed.
    UnexpectedModelCall {
        /// The call that found no step left.
        call: Box<ModelCall>,
        /// How many calls the model had already served.
        call_index: usize,
    },
    /// A test finished before consuming every configured step.
    UnconsumedModelSteps {
        /// Steps still queued.
        remaining_steps: usize,
    },
}

impl fmt::Display for ModelScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Python's wording: the step number is 1-based, the index 0-based.
            ModelScriptError::InvalidModelStep {
                input_index,
                detail,
                ..
            } => write!(f, "Scripted model step #{} {detail}.", input_index + 1),
            ModelScriptError::UnexpectedModelCall { call, call_index } => write!(
                f,
                "Unexpected {} model call #{}: no scripted steps remain.",
                if call.streamed {
                    "streaming"
                } else {
                    "non-streaming"
                },
                call_index + 1
            ),
            ModelScriptError::UnconsumedModelSteps { remaining_steps } => write!(
                f,
                "{remaining_steps} scripted model step(s) were not consumed."
            ),
        }
    }
}

impl std::error::Error for ModelScriptError {}

impl From<ModelScriptError> for AgentsError {
    fn from(error: ModelScriptError) -> Self {
        AgentsError::other(error.to_string())
    }
}

/// A recorded call at the provider-neutral [`Model`] boundary.
#[derive(Debug, Clone)]
pub struct ModelCall {
    /// Everything the runner passed to the model.
    pub request: ModelRequest,
    /// Whether the call came through [`Model::stream_response`].
    pub streamed: bool,
}

/// Derives a step's result from the call that triggered it.
pub type ScriptedResponder =
    Arc<dyn Fn(&ModelCall) -> std::result::Result<ScriptedStep, AgentsError> + Send + Sync>;

/// Derives a step's result asynchronously from the call that triggered it.
pub type ScriptedAsyncResponder = Arc<
    dyn Fn(
            ModelCall,
        )
            -> futures::future::BoxFuture<'static, std::result::Result<ScriptedStep, AgentsError>>
        + Send
        + Sync,
>;

/// Builds the exact event stream of a streamed call, one event (or error) at a time.
pub type ScriptedStreamFactory =
    Arc<dyn Fn(&ModelCall) -> futures::stream::BoxStream<'static, Result<Value>> + Send + Sync>;

/// The response id [`ScriptedStep`] uses when none is given.
pub const DEFAULT_SCRIPTED_RESPONSE_ID: &str = "resp-789";

/// One deterministic model call result.
///
/// `output` is the normalized SDK output-item boundary. Set `error` to fail the call,
/// `responder` to derive the result from the recorded call, or `stream_events` to supply an
/// exact event stream for streaming tests. At most one of the three may be set.
pub struct ScriptedStep {
    /// The output items the call returns.
    pub output: Vec<OutputItem>,
    /// Usage reported for this call; `None` falls back to the model's default usage.
    pub usage: Option<Usage>,
    /// The response id; `None` reports no response id at all.
    pub response_id: Option<String>,
    /// The transport request id.
    pub request_id: Option<String>,
    /// The provider usage payload, surfaced when `preserve_raw_usage` is on.
    pub raw_usage: Option<Value>,
    /// Fail the call with this error.
    pub error: Option<AgentsError>,
    /// Derive the result from the recorded call.
    pub responder: Option<ScriptedResponder>,
    /// Derive the result from the recorded call, awaiting it first.
    pub async_responder: Option<ScriptedAsyncResponder>,
    /// Emit exactly these stream events instead of synthesizing them.
    pub stream_events: Option<Vec<Value>>,
    /// Produce the exact stream events lazily, so a stream can fail or block part-way.
    pub stream_factory: Option<ScriptedStreamFactory>,
    /// Retry guidance reported for `error` through [`Model::get_retry_advice`].
    pub retry_advice: Option<crate::retry::ModelRetryAdvice>,
}

impl Default for ScriptedStep {
    fn default() -> Self {
        ScriptedStep {
            output: Vec::new(),
            usage: None,
            response_id: Some(DEFAULT_SCRIPTED_RESPONSE_ID.to_string()),
            request_id: None,
            raw_usage: None,
            error: None,
            responder: None,
            async_responder: None,
            stream_events: None,
            stream_factory: None,
            retry_advice: None,
        }
    }
}

impl fmt::Debug for ScriptedStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScriptedStep")
            .field("output", &self.output)
            .field("usage", &self.usage)
            .field("response_id", &self.response_id)
            .field("request_id", &self.request_id)
            .field("raw_usage", &self.raw_usage)
            .field("error", &self.error)
            .field("responder", &self.responder.as_ref().map(|_| "Fn(..)"))
            .field(
                "async_responder",
                &self.async_responder.as_ref().map(|_| "Fn(..)"),
            )
            .field("stream_events", &self.stream_events)
            .field(
                "stream_factory",
                &self.stream_factory.as_ref().map(|_| "Fn(..)"),
            )
            .field("retry_advice", &self.retry_advice)
            .finish()
    }
}

impl ScriptedStep {
    /// A step returning `output`.
    pub fn output(output: Vec<OutputItem>) -> Self {
        ScriptedStep {
            output,
            ..Default::default()
        }
    }

    /// A step that fails with `error`.
    pub fn raise_error(error: AgentsError) -> Self {
        ScriptedStep {
            error: Some(error),
            ..Default::default()
        }
    }

    /// A step whose result is derived from the recorded call.
    pub fn respond<F>(responder: F) -> Self
    where
        F: Fn(&ModelCall) -> std::result::Result<ScriptedStep, AgentsError> + Send + Sync + 'static,
    {
        ScriptedStep {
            responder: Some(Arc::new(responder)),
            ..Default::default()
        }
    }

    /// A step whose result is derived asynchronously from the recorded call.
    pub fn respond_async<F, Fut>(responder: F) -> Self
    where
        F: Fn(ModelCall) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = std::result::Result<ScriptedStep, AgentsError>>
            + Send
            + 'static,
    {
        ScriptedStep {
            async_responder: Some(Arc::new(move |call| Box::pin(responder(call)))),
            ..Default::default()
        }
    }

    /// A step emitting an exact event sequence; `output` is what the non-streamed path returns.
    pub fn stream(events: Vec<Value>) -> Self {
        ScriptedStep {
            stream_events: Some(events),
            ..Default::default()
        }
    }

    /// A step whose exact event stream is built from the recorded call and polled lazily.
    ///
    /// An `Err` item fails the call part-way through; dropping the stream drops whatever the
    /// factory returned.
    pub fn stream_with<F>(factory: F) -> Self
    where
        F: Fn(&ModelCall) -> futures::stream::BoxStream<'static, Result<Value>>
            + Send
            + Sync
            + 'static,
    {
        ScriptedStep {
            stream_factory: Some(Arc::new(factory)),
            ..Default::default()
        }
    }

    /// Attach provider retry guidance. Only valid together with an error.
    pub fn with_retry_advice(mut self, advice: crate::retry::ModelRetryAdvice) -> Self {
        self.retry_advice = Some(advice);
        self
    }

    /// Report this usage instead of the model's default.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Report this response id (`None` reports none).
    pub fn with_response_id(mut self, response_id: Option<String>) -> Self {
        self.response_id = response_id;
        self
    }

    fn validate(&self, input_index: usize) -> std::result::Result<(), ModelScriptError> {
        let selected = u8::from(self.error.is_some())
            + u8::from(self.responder.is_some() || self.async_responder.is_some())
            + u8::from(self.stream_events.is_some() || self.stream_factory.is_some());
        if selected > 1
            || (self.responder.is_some() && self.async_responder.is_some())
            || (self.stream_events.is_some() && self.stream_factory.is_some())
        {
            return Err(ModelScriptError::InvalidModelStep {
                reason: ModelStepReason::ConflictingOutcomes,
                input_index,
                detail: "cannot combine error, responder, and stream_events outcomes".into(),
            });
        }
        if self.retry_advice.is_some() && self.error.is_none() {
            return Err(ModelScriptError::InvalidModelStep {
                reason: ModelStepReason::InvalidRetryAdvice,
                input_index,
                detail: "requires an error and a ModelRetryAdvice value".into(),
            });
        }
        Ok(())
    }
}

/// A deterministic, provider-neutral model for testing agent workflows.
///
/// Every call consumes the next configured step in order. A call with no step left fails with
/// [`ModelScriptError::UnexpectedModelCall`], and [`ScriptedModel::assert_complete`] reports
/// steps a test never reached — the strict matching `FakeModel` does not do.
#[derive(Debug, Default)]
pub struct ScriptedModel {
    steps: Mutex<VecDeque<ScriptedStep>>,
    calls: Mutex<Vec<ModelCall>>,
    default_usage: Mutex<Option<Usage>>,
    emit_traces: bool,
    /// Retry advice of the most recently raised scripted error, keyed by its message.
    last_retry_advice: Mutex<Option<(String, crate::retry::ModelRetryAdvice)>>,
}

impl ScriptedModel {
    /// A model with no configured steps.
    pub fn new() -> Self {
        Self::default()
    }

    /// A model running `steps` in order.
    pub fn with_steps(steps: Vec<ScriptedStep>) -> std::result::Result<Self, ModelScriptError> {
        let model = Self::new();
        model.extend(steps)?;
        Ok(model)
    }

    /// Emit a `generation` span per call, as a real model does.
    pub fn with_emit_traces(mut self, emit_traces: bool) -> Self {
        self.emit_traces = emit_traces;
        self
    }

    /// Append one step.
    pub fn enqueue(&self, step: ScriptedStep) -> std::result::Result<(), ModelScriptError> {
        step.validate(0)?;
        self.steps.lock().expect("poisoned").push_back(step);
        Ok(())
    }

    /// Append several steps; none is queued when one of them is invalid.
    pub fn extend(&self, steps: Vec<ScriptedStep>) -> std::result::Result<(), ModelScriptError> {
        for (input_index, step) in steps.iter().enumerate() {
            step.validate(input_index)?;
        }
        self.steps.lock().expect("poisoned").extend(steps);
        Ok(())
    }

    /// Usage reported for steps that carry none.
    pub fn set_default_usage(&self, usage: Option<Usage>) {
        *self.default_usage.lock().expect("poisoned") = usage;
    }

    /// Every recorded call, in order.
    pub fn calls(&self) -> Vec<ModelCall> {
        self.calls.lock().expect("poisoned").clone()
    }

    /// The first recorded call.
    pub fn first_call(&self) -> Option<ModelCall> {
        self.calls.lock().expect("poisoned").first().cloned()
    }

    /// The most recent recorded call.
    pub fn last_call(&self) -> Option<ModelCall> {
        self.calls.lock().expect("poisoned").last().cloned()
    }

    /// Configured calls that have not run yet.
    pub fn remaining_steps(&self) -> usize {
        self.steps.lock().expect("poisoned").len()
    }

    /// Fail when configured steps remain unconsumed.
    pub fn assert_complete(&self) -> std::result::Result<(), ModelScriptError> {
        let remaining = self.remaining_steps();
        if remaining > 0 {
            return Err(ModelScriptError::UnconsumedModelSteps {
                remaining_steps: remaining,
            });
        }
        Ok(())
    }

    /// Record the call and resolve the step it consumes, following responders.
    ///
    /// A responder's step never entered the queue, so it is validated here instead, and a
    /// responder may itself return another responder, which is followed in turn. A responder that
    /// fails forgets the retry advice remembered for an error with the same message, so advice
    /// attached to an earlier scripted error cannot leak onto a later failure.
    async fn next_step(&self, request: &ModelRequest, streamed: bool) -> Result<ScriptedStep> {
        let call = ModelCall {
            request: request.clone(),
            streamed,
        };
        let call_index = {
            let mut calls = self.calls.lock().expect("poisoned");
            calls.push(call.clone());
            calls.len() - 1
        };
        let popped = self.steps.lock().expect("poisoned").pop_front();
        let mut step = popped.ok_or_else(|| {
            AgentsError::from(ModelScriptError::UnexpectedModelCall {
                call: Box::new(call.clone()),
                call_index,
            })
        })?;
        loop {
            let next = if let Some(responder) = step.responder.take() {
                responder(&call)
            } else if let Some(responder) = step.async_responder.take() {
                responder(call.clone()).await
            } else {
                break;
            };
            step = match next {
                Ok(next) => next,
                Err(error) => {
                    self.forget_retry_advice(&error);
                    return Err(error);
                }
            };
            step.validate(0)?;
        }
        Ok(step)
    }

    fn model_response(&self, step: &ScriptedStep, request: &ModelRequest) -> Result<ModelResponse> {
        // A step reporting no usage (or an all-zero one) falls back to the model's default usage,
        // and that to a single counted request.
        let usage = step
            .usage
            .clone()
            .filter(|usage| *usage != Usage::default())
            .or_else(|| self.default_usage.lock().expect("poisoned").clone())
            .unwrap_or_else(|| Usage {
                requests: 1,
                ..Default::default()
            });
        Ok(ModelResponse {
            output: convert_output_items(&step.output)?,
            usage,
            response_id: step.response_id.clone(),
            request_id: step.request_id.clone(),
            raw_usage: preserve_raw_usage(request)
                .then(|| step.raw_usage.clone())
                .flatten(),
        })
    }

    /// Raise a step's error, remembering any retry advice attached to it.
    fn raise(&self, step: &mut ScriptedStep) -> Option<AgentsError> {
        let error = step.error.take()?;
        *self.last_retry_advice.lock().expect("poisoned") = step
            .retry_advice
            .clone()
            .map(|advice| (error.to_string(), advice));
        Some(error)
    }

    /// Drop the remembered advice when it belongs to an error with this message.
    fn forget_retry_advice(&self, error: &AgentsError) {
        let mut guard = self.last_retry_advice.lock().expect("poisoned");
        if guard
            .as_ref()
            .is_some_and(|(message, _)| *message == error.to_string())
        {
            *guard = None;
        }
    }

    fn span_for(&self) -> crate::tracing::Span {
        crate::tracing::generation_span(None, None, None, !self.emit_traces)
    }

    fn record_span_error(span: &crate::tracing::Span, error: &AgentsError, request: &ModelRequest) {
        let message = if request.tracing.include_data() {
            error.to_string()
        } else {
            crate::tracing::REDACTED_TRACE_ERROR_MESSAGE.to_string()
        };
        // Python records the exception class name next to the message; the Rust error kind plays
        // that role.
        let name = match error {
            AgentsError::UserError { .. } => "UserError",
            AgentsError::ModelBehaviorError { .. } => "ModelBehaviorError",
            AgentsError::ModelRefusalError { .. } => "ModelRefusalError",
            AgentsError::ModelTimeoutError { .. } => "ModelTimeoutError",
            AgentsError::MaxTurnsExceeded { .. } => "MaxTurnsExceeded",
            AgentsError::ApiError { .. } => "ApiError",
            _ => "AgentsError",
        };
        span.set_error("Error", Some(json!({"name": name, "message": message})));
    }
}

/// Whether the request asked for the provider's raw usage payload to be preserved.
fn preserve_raw_usage(request: &ModelRequest) -> bool {
    request.model_settings.preserve_raw_usage == Some(true)
}

/// Normalize scripted output items the way Python's `_convert_output_items` does: an
/// `apply_patch_call` must carry a non-empty call id, an item id that is non-empty when given
/// (defaulting to the call id), and a status of `in_progress` or `completed` (default
/// `completed`).
fn convert_output_items(output: &[OutputItem]) -> Result<Vec<OutputItem>> {
    output
        .iter()
        .map(|item| match item {
            OutputItem::Other(value)
                if value.get("type").and_then(Value::as_str) == Some("apply_patch_call") =>
            {
                convert_apply_patch_call(value)
            }
            other => Ok(other.clone()),
        })
        .collect()
}

fn convert_apply_patch_call(value: &Value) -> Result<OutputItem> {
    let call_id = match value.get("call_id").and_then(Value::as_str) {
        Some(call_id) if !call_id.is_empty() => call_id,
        _ => {
            return Err(AgentsError::model_behavior(
                "Tool invocations require a non-empty string call ID before execution.",
            ));
        }
    };
    let item_id = match value.get("id") {
        Some(Value::String(id)) if !id.is_empty() => id.as_str(),
        Some(_) => {
            return Err(AgentsError::model_behavior(
                "Apply-patch tool calls require a non-empty string item ID when provided.",
            ));
        }
        None => call_id,
    };
    let status = match value.get("status") {
        None => "completed",
        Some(Value::String(status)) if matches!(status.as_str(), "in_progress" | "completed") => {
            status.as_str()
        }
        Some(other) => {
            return Err(AgentsError::user(format!(
                "Invalid apply_patch_call status {other}: expected \"in_progress\" or \"completed\""
            )));
        }
    };
    let mut converted = json!({
        "type": "apply_patch_call",
        "id": item_id,
        "call_id": call_id,
        "status": status,
        "operation": value.get("operation").cloned().unwrap_or(Value::Null),
    });
    if let Some(caller) = value.get("caller") {
        converted["caller"] = caller.clone();
    }
    Ok(OutputItem::Other(converted))
}

/// Item kinds whose real streams carry events the automatic stream cannot reproduce. Python
/// rejects them with a pointer to the exact-stream step; a stream carrying them has to be scripted
/// event by event. Other raw items (shell, program, tool search, web search, ...) pass through as
/// a bare added/done pair.
const UNSUPPORTED_AUTOMATIC_STREAM_ITEMS: &[&str] = &[
    "custom_tool_call",
    "code_interpreter_call",
    "mcp_call",
    "image_generation_call",
];

/// Reject output the automatic stream cannot express.
fn check_automatic_stream_support(output: &[OutputItem]) -> Result<()> {
    for item in output {
        if let OutputItem::Other(value) = item {
            let kind = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            if UNSUPPORTED_AUTOMATIC_STREAM_ITEMS.contains(&kind) {
                return Err(AgentsError::model_behavior(format!(
                    "Automatic streaming does not support {kind}. \
                     Use ScriptedStep::stream(...) to provide exact normalized stream events."
                )));
            }
        }
    }
    Ok(())
}

#[async_trait]
impl Model for ScriptedModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        let span = self.span_for();
        span.start();
        // A cancelled call (a timeout drops this future) still finishes its span, and records the
        // timeout when the attempt that owns the call timed out.
        let mut guard =
            crate::tracing::ModelSpanGuard::new(&span, "Error", request.tracing.include_data());
        let result = async {
            let mut step = self.next_step(&request, false).await?;
            if let Some(error) = self.raise(&mut step) {
                return Err(error);
            }
            self.model_response(&step, &request)
        }
        .await;
        if let Err(error) = &result {
            Self::record_span_error(&span, error, &request);
        }
        guard.disarm();
        span.finish();
        result
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        let span = self.span_for();
        span.start();
        let mut guard =
            crate::tracing::ModelSpanGuard::new(&span, "Error", request.tracing.include_data());
        let prepared = async {
            let mut step = self.next_step(&request, true).await?;
            if let Some(error) = self.raise(&mut step) {
                return Err(error);
            }
            if let Some(factory) = &step.stream_factory {
                let call = ModelCall {
                    request: request.clone(),
                    streamed: true,
                };
                return Ok(factory(&call));
            }
            let events = match &step.stream_events {
                Some(events) => events.clone(),
                None => {
                    check_automatic_stream_support(&step.output)?;
                    let response = self.model_response(&step, &request)?;
                    let converted = ScriptedStep {
                        output: response.output.clone(),
                        ..Default::default()
                    };
                    scripted_stream_events(&converted, response)
                }
            };
            Ok(futures::stream::iter(events.into_iter().map(Ok)).boxed())
        }
        .await;
        match prepared {
            Ok(stream) => {
                // The span stays open while the consumer drains the stream: a failure part-way is
                // recorded on it, and dropping the stream (an early exit or a cancellation)
                // finishes it.
                let request_for_errors = request;
                Ok(stream
                    .map(move |event| {
                        if let Err(error) = &event {
                            Self::record_span_error(guard.span(), error, &request_for_errors);
                        }
                        event
                    })
                    .boxed())
            }
            Err(error) => {
                Self::record_span_error(&span, &error, &request);
                guard.disarm();
                span.finish();
                Err(error)
            }
        }
    }

    fn get_retry_advice(
        &self,
        request: &crate::retry::ModelRetryAdviceRequest<'_>,
    ) -> Option<crate::retry::ModelRetryAdvice> {
        let guard = self.last_retry_advice.lock().expect("poisoned");
        let (message, advice) = guard.as_ref()?;
        (*message == request.error.to_string()).then(|| advice.clone())
    }
}

/// Build the Responses stream events a scripted step turns into.
///
/// Mirrors `_stream_events_for_step`: `response.created`, `response.in_progress`, then per
/// output item an `output_item.added`, its per-part events and an `output_item.done`, and
/// finally `response.completed`. Every event carries a monotonic `sequence_number`.
pub fn scripted_stream_events(step: &ScriptedStep, response: ModelResponse) -> Vec<Value> {
    let mut events = Vec::new();
    let mut sequence = 0u64;
    let mut push = |events: &mut Vec<Value>, mut event: Value| {
        event["sequence_number"] = json!(sequence);
        sequence += 1;
        events.push(event);
    };

    let completed = completed_response_value(&response);
    let in_progress = json!({
        "id": response.response_id,
        "status": "in_progress",
        "output": [],
    });
    push(
        &mut events,
        json!({"type": "response.created", "response": in_progress}),
    );
    push(
        &mut events,
        json!({"type": "response.in_progress", "response": in_progress}),
    );

    for (output_index, item) in step.output.iter().enumerate() {
        let done_item = item.to_value();
        push(
            &mut events,
            json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": in_progress_output_item(item),
            }),
        );
        match item {
            OutputItem::Reasoning(r) => {
                for (summary_index, summary) in r.summary.iter().enumerate() {
                    let text = summary.get("text").and_then(Value::as_str).unwrap_or("");
                    let part_type = summary
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("summary_text");
                    for event in [
                        json!({"type": "response.reasoning_summary_part.added",
                               "part": {"type": part_type, "text": ""}}),
                        json!({"type": "response.reasoning_summary_text.delta", "delta": text}),
                        json!({"type": "response.reasoning_summary_text.done", "text": text}),
                        json!({"type": "response.reasoning_summary_part.done",
                               "part": {"type": part_type, "text": text}}),
                    ] {
                        let mut event = event;
                        event["item_id"] = json!(r.id);
                        event["output_index"] = json!(output_index);
                        event["summary_index"] = json!(summary_index);
                        push(&mut events, event);
                    }
                }
                for (content_index, content) in r.content.iter().flatten().enumerate() {
                    let text = content.get("text").and_then(Value::as_str).unwrap_or("");
                    for event in [
                        json!({"type": "response.reasoning_text.delta", "delta": text}),
                        json!({"type": "response.reasoning_text.done", "text": text}),
                    ] {
                        let mut event = event;
                        event["item_id"] = json!(r.id);
                        event["output_index"] = json!(output_index);
                        event["content_index"] = json!(content_index);
                        push(&mut events, event);
                    }
                }
            }
            OutputItem::FunctionCall(call) => {
                let item_id = call.id.clone().unwrap_or_else(|| call.call_id.clone());
                for event in [
                    json!({"type": "response.function_call_arguments.delta",
                           "delta": call.arguments}),
                    json!({"type": "response.function_call_arguments.done",
                           "arguments": call.arguments, "name": call.name}),
                ] {
                    let mut event = event;
                    event["item_id"] = json!(item_id);
                    event["output_index"] = json!(output_index);
                    push(&mut events, event);
                }
            }
            OutputItem::Message(message) => {
                for (content_index, part) in message.content.iter().enumerate() {
                    for event in message_part_events(part) {
                        let mut event = event;
                        event["item_id"] = json!(message.id);
                        event["output_index"] = json!(output_index);
                        event["content_index"] = json!(content_index);
                        push(&mut events, event);
                    }
                }
            }
            OutputItem::Other(_) => {}
        }
        push(
            &mut events,
            json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": done_item,
            }),
        );
    }

    push(
        &mut events,
        json!({"type": "response.completed", "response": completed}),
    );
    events
}

/// The events one message content part expands into, without the shared item/index fields.
fn message_part_events(part: &crate::items::MessageContent) -> Vec<Value> {
    use crate::items::MessageContent;
    match part {
        MessageContent::OutputText {
            text,
            annotations,
            logprobs,
        } => {
            // The delta, the done event and the finished part carry the scripted log probabilities;
            // the part the stream opens with is still empty.
            let logprobs = logprobs.clone().unwrap_or_else(|| json!([]));
            let mut out = vec![
                json!({"type": "response.content_part.added",
                       "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
                json!({"type": "response.output_text.delta", "delta": text, "logprobs": logprobs}),
            ];
            for (annotation_index, annotation) in annotations.iter().enumerate() {
                out.push(json!({
                    "type": "response.output_text.annotation.added",
                    "annotation_index": annotation_index,
                    "annotation": annotation,
                }));
            }
            out.push(
                json!({"type": "response.output_text.done", "text": text, "logprobs": logprobs}),
            );
            out.push(json!({"type": "response.content_part.done",
                            "part": {"type": "output_text", "text": text,
                                     "annotations": annotations, "logprobs": logprobs}}));
            out
        }
        MessageContent::Refusal { refusal } => vec![
            json!({"type": "response.content_part.added",
                   "part": {"type": "refusal", "refusal": ""}}),
            json!({"type": "response.refusal.delta", "delta": refusal}),
            json!({"type": "response.refusal.done", "refusal": refusal}),
            json!({"type": "response.content_part.done",
                   "part": {"type": "refusal", "refusal": refusal}}),
        ],
        MessageContent::Other(value) => vec![
            json!({"type": "response.content_part.added", "part": value}),
            json!({"type": "response.content_part.done", "part": value}),
        ],
    }
}

/// The item as `response.output_item.added` reports it: present but not yet filled in.
fn in_progress_output_item(item: &OutputItem) -> Value {
    let mut value = item.to_value();
    match item {
        OutputItem::Message(_) => {
            value["content"] = json!([]);
            value["status"] = json!("in_progress");
        }
        OutputItem::Reasoning(_) => {
            value["summary"] = json!([]);
            if value.get("content").is_some() {
                value["content"] = json!([]);
            }
            value["status"] = json!("in_progress");
            if let Some(object) = value.as_object_mut() {
                object.remove("encrypted_content");
            }
        }
        OutputItem::FunctionCall(_) => {
            value["arguments"] = json!("");
            value["status"] = json!("in_progress");
        }
        OutputItem::Other(inner)
            if inner.get("type").and_then(Value::as_str) == Some("apply_patch_call") =>
        {
            value["status"] = json!("in_progress");
        }
        OutputItem::Other(_) => {}
    }
    value
}

/// The `response` payload of the final `response.completed` event.
fn completed_response_value(response: &ModelResponse) -> Value {
    // A scripted raw usage payload carries the provider's extra fields, but the normalized
    // counts still have to survive it: the consumer reads this one object both to normalize
    // usage and, with `preserve_raw_usage`, to keep the raw snapshot.
    let mut usage = match &response.raw_usage {
        Some(Value::Object(raw)) => Value::Object(raw.clone()),
        _ => json!({}),
    };
    usage["input_tokens"] = json!(response.usage.input_tokens);
    usage["output_tokens"] = json!(response.usage.output_tokens);
    usage["total_tokens"] = json!(response.usage.total_tokens);
    // The request count and per-request breakdown ride on the usage object so a streamed run
    // reports the same usage as a non-streamed one (Python's `_agents_sdk_request_count`).
    usage["requests"] = json!(response.usage.requests);
    if !response.usage.request_usage_entries.is_empty() {
        usage["request_usage_entries"] =
            serde_json::to_value(&response.usage.request_usage_entries).unwrap_or(Value::Null);
    }
    // A provider payload that already carries token details keeps them verbatim.
    if usage.get("input_tokens_details").is_none() {
        usage["input_tokens_details"] = json!({
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
            "cache_write_tokens": response.usage.input_tokens_details.cache_write_tokens,
        });
    }
    if usage.get("output_tokens_details").is_none() {
        usage["output_tokens_details"] =
            json!({"reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens});
    }
    let mut completed = json!({
        "id": response.response_id,
        "status": "completed",
        "output": response.output.iter().map(OutputItem::to_value).collect::<Vec<_>>(),
        "usage": usage,
    });
    // A streamed response carries its transport request id on the terminal event, as a real
    // Responses stream does (Python's `response._request_id`).
    if let Some(request_id) = &response.request_id {
        completed[crate::models::openai_responses::STREAM_REQUEST_ID_KEY] = json!(request_id);
    }
    completed
}
