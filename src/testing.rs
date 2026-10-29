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
        let step = self.steps.lock().expect("poisoned").pop_front().ok_or_else(|| {
            AgentsError::other("FakeModel received an unexpected call: no scripted steps remain")
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
    let mut events = vec![json!({"type": "response.created", "response": {"id": response.response_id, "status": "in_progress", "output": []}})];
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
