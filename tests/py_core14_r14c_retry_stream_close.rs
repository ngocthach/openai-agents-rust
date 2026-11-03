//! Ports of the stream-closing cases of `tests/models/test_model_retry.py`:
//! `test_stream_response_with_retry_closes_abandoned_stream_before_retry`,
//! `test_stream_response_with_retry_closes_current_stream_when_consumer_stops_early` and
//! `test_timed_stream_closes_provider_iterator_once`. Python counts `aclose()` calls on the
//! provider's async iterator; a Rust provider stream is closed by dropping it, so a drop guard
//! travelling inside the stream counts the closes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::model_settings::{
    ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::retry::retry_policies;
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, AgentsError, RunConfig, RunOptions, Runner};
use serde_json::{Value, json};

/// What one provider attempt does.
enum Attempt {
    /// The stream fails with a connection error before it yields anything.
    ErrorBeforeYield,
    /// The stream yields these events, then ends normally.
    Events(Vec<Value>),
    /// The stream yields these events, then stalls until it is dropped.
    EventsThenStall(Vec<Value>),
}

struct CloseGuard(Arc<AtomicUsize>);

impl Drop for CloseGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for Tracked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracked").finish_non_exhaustive()
    }
}

struct Tracked {
    attempts: Mutex<VecDeque<Attempt>>,
    closes: Arc<AtomicUsize>,
    /// The close count observed when each attempt began.
    closes_at_attempt_start: Mutex<Vec<usize>>,
}

impl Tracked {
    fn new(attempts: Vec<Attempt>) -> Arc<Self> {
        Arc::new(Tracked {
            attempts: Mutex::new(attempts.into()),
            closes: Arc::new(AtomicUsize::new(0)),
            closes_at_attempt_start: Mutex::new(Vec::new()),
        })
    }
    fn closes(&self) -> usize {
        self.closes.load(Ordering::SeqCst)
    }
}

fn created() -> Value {
    json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}})
}

fn completed() -> Value {
    json!({"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}})
}

async fn connection_error() -> AgentsError {
    let err = reqwest::Client::new()
        .post("http://127.0.0.1:1/")
        .send()
        .await
        .expect_err("nothing listens on port 1");
    AgentsError::Http(err)
}

#[async_trait]
impl Model for Tracked {
    async fn get_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        unreachable!("these tests only stream")
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        self.closes_at_attempt_start
            .lock()
            .unwrap()
            .push(self.closes());
        let attempt = self
            .attempts
            .lock()
            .unwrap()
            .pop_front()
            .expect("the model ran out of attempts");
        let guard = CloseGuard(self.closes.clone());
        let (events, error, stall) = match attempt {
            Attempt::ErrorBeforeYield => (VecDeque::new(), Some(connection_error().await), false),
            Attempt::Events(events) => (events.into(), None, false),
            Attempt::EventsThenStall(events) => (events.into(), None, true),
        };
        Ok(futures::stream::unfold(
            (guard, events, error, stall),
            |(guard, mut events, mut error, stall)| async move {
                if let Some(event) = events.pop_front() {
                    return Some((Ok(event), (guard, events, error, stall)));
                }
                if let Some(error) = error.take() {
                    return Some((Err(error), (guard, events, None, stall)));
                }
                if stall {
                    std::future::pending::<()>().await;
                }
                None
            },
        )
        .boxed())
    }
}

fn agent(
    model: &Arc<Tracked>,
    retry: Option<ModelRetrySettings>,
    timeout: Option<f64>,
) -> Agent<()> {
    Agent::<()>::new("retry")
        .with_model_instance(model.clone())
        .with_model_settings(ModelSettings {
            retry,
            timeout,
            ..Default::default()
        })
}

fn options() -> RunOptions<()> {
    let mut options = RunOptions::<()>::default();
    options.run_config = RunConfig::<()>::new().with_tracing_disabled(true);
    options
}

async fn consume(agent: &Agent<()>) -> Result<(), AgentsError> {
    let mut result = Runner::run_streamed_with_options(agent, "hi", options());
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        event?;
    }
    Ok(())
}

async fn wait_for_closes(model: &Tracked, expected: usize) {
    for _ in 0..200 {
        if model.closes() == expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        model.closes(),
        expected,
        "the stream was not dropped in time"
    );
}

/// Python: `test_stream_response_with_retry_closes_abandoned_stream_before_retry`.
#[tokio::test]
async fn an_abandoned_stream_is_closed_before_the_retry_starts() {
    let model = Tracked::new(vec![
        Attempt::ErrorBeforeYield,
        Attempt::Events(vec![created(), completed()]),
    ]);
    let retry = ModelRetrySettings {
        max_retries: Some(1),
        backoff: Some(ModelRetryBackoffSettings {
            initial_delay: Some(0.0),
            jitter: Some(false),
            ..Default::default()
        }),
        policy: Some(retry_policies::network_error()),
        ..Default::default()
    };
    consume(&agent(&model, Some(retry), None)).await.unwrap();

    // The first stream was already closed when the second attempt began.
    assert_eq!(*model.closes_at_attempt_start.lock().unwrap(), vec![0, 1]);
    wait_for_closes(&model, 2).await;
}

/// Python: `test_stream_response_with_retry_closes_current_stream_when_consumer_stops_early`.
/// Python closes the retry generator; a Rust consumer that stops cancels the run (merely dropping
/// the result leaves the run going, as in Python), and the provider stream is dropped with it.
#[tokio::test]
async fn cancelling_after_the_first_event_closes_the_current_stream() {
    let model = Tracked::new(vec![Attempt::EventsThenStall(vec![created(), created()])]);
    let retry = ModelRetrySettings {
        max_retries: Some(1),
        policy: Some(retry_policies::network_error()),
        ..Default::default()
    };
    let agent = agent(&model, Some(retry), None);
    let mut result = Runner::run_streamed_with_options(&agent, "hi", options());
    let cancel = result.cancel_handle();
    {
        let mut events = std::pin::pin!(result.stream_events());
        loop {
            let event = events.next().await.expect("an event").expect("no error");
            if matches!(event, openai_agents_sdk::StreamEvent::RawResponse(_)) {
                break;
            }
        }
        // The consumer stops here: cancelling the run stops the provider stream.
        cancel.cancel(openai_agents_sdk::result::CancelMode::Immediate);
        while events.next().await.is_some() {}
    }
    drop(result);

    wait_for_closes(&model, 1).await;
    // No retry attempt was started by the early exit.
    assert_eq!(model.closes_at_attempt_start.lock().unwrap().len(), 1);
}

/// Python: `test_timed_stream_closes_provider_iterator_once`.
#[tokio::test]
async fn a_timed_stream_closes_the_provider_stream_exactly_once() {
    let model = Tracked::new(vec![Attempt::Events(vec![created(), completed()])]);
    consume(&agent(&model, None, Some(1.0))).await.unwrap();
    wait_for_closes(&model, 1).await;
    // Give a stray second close the chance to show up.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(model.closes(), 1);
    let _ = Usage::default();
}
