//! Python `tests/models/test_model_retry.py` timeout cases whose observable half is the
//! surfaced error and the released provider work: a stream that stalls after delivering an
//! event, and a non-streaming call that never completes. Python also asserts the traceback and
//! exception graph carry no provider payload; a Rust error holds no source at all.
//! A provider stream or future is closed by dropping it, so a drop guard counts the closes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::{Agent, AgentsError, RunConfig, RunOptions, Runner, StreamEvent};
use serde_json::{Value, json};

struct CloseGuard(Arc<AtomicUsize>);

impl Drop for CloseGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Yields `events`, then stalls until dropped; a non-streaming call stalls at once.
struct Stalling {
    events: Mutex<VecDeque<Value>>,
    closes: Arc<AtomicUsize>,
}

impl std::fmt::Debug for Stalling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stalling").finish_non_exhaustive()
    }
}

#[async_trait]
impl Model for Stalling {
    async fn get_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        let _guard = CloseGuard(self.closes.clone());
        std::future::pending::<()>().await;
        unreachable!("the call is cancelled by the timeout")
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        let guard = CloseGuard(self.closes.clone());
        let events: VecDeque<Value> = std::mem::take(&mut *self.events.lock().unwrap());
        Ok(
            futures::stream::unfold((guard, events), |(guard, mut events)| async move {
                if let Some(event) = events.pop_front() {
                    return Some((Ok(event), (guard, events)));
                }
                std::future::pending::<()>().await;
                None
            })
            .boxed(),
        )
    }
}

fn model(events: Vec<Value>) -> Arc<Stalling> {
    Arc::new(Stalling {
        events: Mutex::new(events.into()),
        closes: Arc::new(AtomicUsize::new(0)),
    })
}

fn agent(model: &Arc<Stalling>, timeout: f64) -> Agent<()> {
    Agent::<()>::new("timeout")
        .with_model_instance(model.clone())
        .with_model_settings(ModelSettings {
            timeout: Some(timeout),
            ..Default::default()
        })
}

fn options() -> RunOptions<()> {
    let mut options = RunOptions::<()>::default();
    options.run_config = RunConfig::<()>::new().with_tracing_disabled(true);
    options
}

async fn wait_for_closes(model: &Stalling, expected: usize) {
    for _ in 0..200 {
        if model.closes.load(Ordering::SeqCst) == expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        model.closes.load(Ordering::SeqCst),
        expected,
        "not dropped in time"
    );
}

/// `test_stream_timeout_discards_previous_event_from_retry_traceback` (text and tool-argument
/// deltas): the event is delivered, the stalled stream times out with the configured timeout,
/// the provider stream is released and the error carries no source.
#[tokio::test]
async fn a_stalled_stream_delivers_its_event_then_times_out_and_is_released() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let payload = "synthetic-sensitive-stream-payload";
    let events = [
        json!({"type": "response.output_text.delta", "delta": payload, "content_index": 0,
               "output_index": 0, "item_id": "synthetic-item", "sequence_number": 0,
               "logprobs": []}),
        json!({"type": "response.function_call_arguments.delta", "delta": payload,
               "output_index": 0, "item_id": "synthetic-item", "sequence_number": 0}),
    ];
    for event in events {
        let model = model(vec![event.clone()]);
        let agent = agent(&model, 0.05);
        let mut result = Runner::run_streamed_with_options(&agent, "synthetic-input", options());
        let mut delivered = Vec::new();
        let mut failure = None;
        {
            let mut stream = std::pin::pin!(result.stream_events());
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamEvent::RawResponse(data)) => delivered.push(data),
                    Ok(_) => {}
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
        }
        let error = failure.expect("the stalled stream times out");
        assert_eq!(delivered.len(), 1, "{event}");
        assert!(format!("{:?}", delivered[0]).contains(payload));
        match &error {
            AgentsError::ModelTimeoutError {
                timeout_seconds, ..
            } => assert_eq!(*timeout_seconds, 0.05),
            other => panic!("expected a timeout, got {other}"),
        }
        assert!(std::error::Error::source(&error).is_none(), "{event}");
        wait_for_closes(&model, 1).await;
    }
}

/// `test_model_timeout_discards_cleanup_exception_graph` and
/// `test_model_timeout_cancels_blocked_cleanup_after_deadline`: a non-streaming call that never
/// completes is abandoned at the deadline (its future is dropped, so nothing can block the
/// timeout) and the error holds neither a cause nor provider text.
#[tokio::test]
async fn a_call_that_never_completes_times_out_and_is_dropped() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = model(Vec::new());
    let agent = agent(&model, 0.01);
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        Runner::run_with_options(&agent, "hi", options()),
    )
    .await
    .expect("the timeout must not wait for the provider");
    let error = outcome.expect_err("the call times out");
    match &error {
        AgentsError::ModelTimeoutError {
            timeout_seconds, ..
        } => assert_eq!(*timeout_seconds, 0.01),
        other => panic!("expected a timeout, got {other}"),
    }
    assert!(std::error::Error::source(&error).is_none());
    wait_for_closes(&model, 1).await;
}
