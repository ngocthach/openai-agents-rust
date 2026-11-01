//! Scripted model and runner helpers shared by the `py_models_retry_*` parity tests.
//!
//! Python's `test_model_retry.py` calls `get_response_with_retry` / `stream_response_with_retry`
//! directly with a `get_response` / `get_stream` callable. The Rust equivalents are private to
//! the runner, so the tests drive them through `Runner` with the scripted [`Scripted`] model:
//! every attempt pops one [`Step`]. Python patches `asyncio.sleep` to record delays; the tests
//! here run on a paused tokio clock, so a delay is the virtual time between two attempts.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::model_settings::{
    ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::models::{
    should_disable_provider_managed_retries, should_disable_websocket_pre_event_retries,
};
use openai_agents_sdk::retry::{ModelRetryAdvice, ModelRetryAdviceRequest, RetryPolicy};
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, AgentsError, RunOptions, RunResult, Runner, StreamEvent};
use serde_json::{Value, json};
use tokio::time::Instant;

/// One element of a scripted stream attempt.
pub enum Item {
    Event(Value),
    Err(AgentsError),
    /// Never produce another element, so the attempt timeout fires.
    Hang,
}

/// One scripted model attempt.
pub enum Step {
    /// A successful non-streamed response with this id and usage.
    Ok(&'static str, Usage),
    /// Fail with this error (before any stream is returned for a streamed attempt).
    Err(AgentsError),
    /// Never finish.
    Hang,
    /// A streamed attempt. Unless it ends in an error or a hang, a terminal
    /// `response.completed` event is appended, as the runner requires one.
    Stream(Vec<Item>),
}

pub fn event(ty: &str) -> Item {
    Item::Event(json!({"type": ty}))
}

type AdviceFn = Box<dyn Fn(&ModelRetryAdviceRequest<'_>) -> Option<ModelRetryAdvice> + Send + Sync>;

pub struct Scripted {
    steps: Mutex<VecDeque<Step>>,
    started: Mutex<Vec<Instant>>,
    flags: Mutex<Vec<(bool, bool)>>,
    advice: Option<AdviceFn>,
}

impl std::fmt::Debug for Scripted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Scripted")
    }
}

impl Scripted {
    pub fn new(steps: Vec<Step>) -> Arc<Self> {
        Arc::new(Scripted {
            steps: Mutex::new(steps.into()),
            started: Mutex::new(Vec::new()),
            flags: Mutex::new(Vec::new()),
            advice: None,
        })
    }

    pub fn with_advice(
        steps: Vec<Step>,
        advice: impl Fn(&ModelRetryAdviceRequest<'_>) -> Option<ModelRetryAdvice>
        + Send
        + Sync
        + 'static,
    ) -> Arc<Self> {
        Arc::new(Scripted {
            steps: Mutex::new(steps.into()),
            started: Mutex::new(Vec::new()),
            flags: Mutex::new(Vec::new()),
            advice: Some(Box::new(advice)),
        })
    }

    pub fn calls(&self) -> usize {
        self.started.lock().unwrap().len()
    }

    /// Virtual seconds between consecutive attempts: the delays the runner slept.
    pub fn delays(&self) -> Vec<f64> {
        let started = self.started.lock().unwrap();
        started
            .windows(2)
            .map(|w| ((w[1] - w[0]).as_secs_f64() * 1000.0).round() / 1000.0)
            .collect()
    }

    /// `should_disable_provider_managed_retries()` as each attempt saw it.
    pub fn provider_retry_flags(&self) -> Vec<bool> {
        self.flags.lock().unwrap().iter().map(|f| f.0).collect()
    }

    /// `should_disable_websocket_pre_event_retries()` as each attempt saw it.
    pub fn websocket_flags(&self) -> Vec<bool> {
        self.flags.lock().unwrap().iter().map(|f| f.1).collect()
    }

    fn begin(&self) -> Step {
        self.started.lock().unwrap().push(Instant::now());
        self.flags.lock().unwrap().push((
            should_disable_provider_managed_retries(),
            should_disable_websocket_pre_event_retries(),
        ));
        self.steps
            .lock()
            .unwrap()
            .pop_front()
            .expect("the scripted model ran out of steps")
    }
}

fn response(id: &str, usage: &Usage) -> ModelResponse {
    ModelResponse {
        output: vec![assistant_message("ok")],
        usage: usage.clone(),
        response_id: Some(id.to_string()),
        request_id: None,
        raw_usage: None,
    }
}

#[async_trait]
impl Model for Scripted {
    async fn get_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        match self.begin() {
            Step::Ok(id, usage) => Ok(response(id, &usage)),
            Step::Err(e) => Err(e),
            Step::Hang => {
                std::future::pending::<()>().await;
                unreachable!()
            }
            Step::Stream(_) => panic!("a streamed step was scripted for a non-streamed call"),
        }
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        match self.begin() {
            Step::Err(e) => Err(e),
            Step::Hang => {
                std::future::pending::<()>().await;
                unreachable!()
            }
            Step::Ok(id, usage) => {
                Ok(futures::stream::iter(vec![Ok(completed(id, &usage))]).boxed())
            }
            Step::Stream(items) => {
                let ends_early = matches!(items.last(), Some(Item::Err(_) | Item::Hang));
                let mut stream_items: VecDeque<Item> = items.into();
                if !ends_early {
                    stream_items
                        .push_back(Item::Event(completed("resp_stream", &Usage::default())));
                }
                Ok(futures::stream::unfold(stream_items, |mut q| async move {
                    match q.pop_front()? {
                        Item::Event(v) => Some((Ok(v), q)),
                        Item::Err(e) => Some((Err(e), VecDeque::new())),
                        Item::Hang => {
                            std::future::pending::<()>().await;
                            None
                        }
                    }
                })
                .boxed())
            }
        }
    }

    fn get_retry_advice(&self, request: &ModelRetryAdviceRequest<'_>) -> Option<ModelRetryAdvice> {
        self.advice.as_ref().and_then(|f| f(request))
    }
}

pub fn completed(id: &str, usage: &Usage) -> Value {
    json!({
        "type": "response.completed",
        "response": {
            "id": id, "object": "response", "status": "completed",
            "output": [{
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }],
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    })
}

pub fn usage_requests_1() -> Usage {
    Usage {
        requests: 1,
        ..Default::default()
    }
}

/// A connection-level failure. reqwest errors cannot be built by hand, so provoke a real refused
/// connection on a port nothing listens on.
pub async fn connection_error() -> AgentsError {
    let err = reqwest::Client::new()
        .post("http://127.0.0.1:1/")
        .send()
        .await
        .expect_err("nothing listens on port 1");
    assert!(err.is_connect());
    AgentsError::Http(err)
}

pub fn status_error(status: u16, code: &str) -> AgentsError {
    AgentsError::ApiError {
        status,
        message: code.to_string(),
        body: Some(Box::new(json!({"error": {"code": code, "message": code}}))),
        request_id: None,
        headers: None,
    }
}

pub fn conversation_locked_error() -> AgentsError {
    status_error_with_code(400, "conversation_locked", "locked")
}

fn status_error_with_code(status: u16, code: &str, message: &str) -> AgentsError {
    AgentsError::ApiError {
        status,
        message: message.to_string(),
        body: Some(Box::new(
            json!({"error": {"code": code, "message": message}}),
        )),
        request_id: None,
        headers: None,
    }
}

pub fn retry(
    max_retries: u32,
    initial_delay: f64,
    policy: Option<RetryPolicy>,
) -> ModelRetrySettings {
    ModelRetrySettings {
        max_retries: Some(max_retries),
        backoff: Some(ModelRetryBackoffSettings {
            initial_delay: Some(initial_delay),
            jitter: Some(false),
            ..Default::default()
        }),
        policy,
        ..Default::default()
    }
}

#[derive(Default, Clone)]
pub struct Opts {
    pub retry: Option<ModelRetrySettings>,
    pub previous_response_id: Option<&'static str>,
    pub conversation_id: Option<&'static str>,
    pub timeout: Option<f64>,
    /// Add a `ProgrammaticToolCalling` tool, which makes the request replay-unsafe in Python.
    pub programmatic_tool_calling: bool,
}

fn agent_and_options(model: &Arc<Scripted>, opts: Opts) -> (Agent<()>, RunOptions<()>) {
    let mut agent = Agent::<()>::new("retry").with_model_instance(model.clone());
    if opts.programmatic_tool_calling {
        agent = agent.with_tool(openai_agents_sdk::tool::ProgrammaticToolCallingTool);
    }
    let agent = agent.with_model_settings(ModelSettings {
        retry: opts.retry,
        timeout: opts.timeout,
        ..Default::default()
    });
    let mut options = RunOptions::<()>::default();
    options.run_config = openai_agents_sdk::RunConfig::<()>::new().with_tracing_disabled(true);
    options.previous_response_id = opts.previous_response_id.map(str::to_string);
    options.conversation_id = opts.conversation_id.map(str::to_string);
    (agent, options)
}

pub async fn run(model: &Arc<Scripted>, opts: Opts) -> Result<RunResult<()>, AgentsError> {
    let (agent, options) = agent_and_options(model, opts);
    Runner::run_with_options(&agent, "hi", options).await
}

/// The outcome of a streamed run: the raw events the consumer saw and the final result or error.
pub struct Streamed {
    pub raw: Vec<Value>,
    pub error: Option<AgentsError>,
    pub result: Option<RunResult<()>>,
}

pub async fn run_streamed(model: &Arc<Scripted>, opts: Opts) -> Streamed {
    let (agent, options) = agent_and_options(model, opts);
    let mut result = Runner::run_streamed_with_options(&agent, "hi", options);
    let mut raw = Vec::new();
    let mut error = None;
    {
        let mut stream = Box::pin(result.stream_events());
        while let Some(ev) = stream.next().await {
            match ev {
                Ok(StreamEvent::RawResponse(v)) => raw.push(v),
                Ok(_) => {}
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
    }
    Streamed {
        raw,
        error,
        result: result.into_result(),
    }
}

impl Streamed {
    /// The `type` of every raw event, in order.
    pub fn types(&self) -> Vec<String> {
        self.raw
            .iter()
            .map(|e| e["type"].as_str().unwrap_or_default().to_string())
            .collect()
    }
}

pub fn response_id(result: &RunResult<()>) -> Option<String> {
    result
        .raw_responses
        .last()
        .and_then(|r| r.response_id.clone())
}

pub fn requests(result: &RunResult<()>) -> u64 {
    result.usage().requests
}

pub fn is_connect_error(err: &AgentsError) -> bool {
    matches!(err, AgentsError::Http(e) if e.is_connect())
}

pub fn is_status(err: &AgentsError, expected: u16) -> bool {
    matches!(err, AgentsError::ApiError { status, .. } if *status == expected)
}
