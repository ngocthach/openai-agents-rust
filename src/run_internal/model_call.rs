//! Model invocation with runner-managed retries, per-attempt timeouts and streaming.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::Value;

use super::EventSink;
use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::models::interface::{Model, ModelRequest};
use crate::models::openai_responses::parse_response;
use crate::stream_events::StreamEvent;

fn is_conversation_locked(e: &AgentsError) -> bool {
    match e {
        AgentsError::ApiError { body: Some(b), .. } => {
            b.pointer("/error/code").and_then(Value::as_str) == Some("conversation_locked")
        }
        _ => false,
    }
}

fn is_retryable(e: &AgentsError, statuses: &[u16], stateful: bool) -> bool {
    match e {
        AgentsError::ApiError { status, .. } if *status != 0 => {
            statuses.contains(status) && (!stateful || *status == 429)
        }
        AgentsError::Http(h) => h.is_connect() || (!stateful && (h.is_timeout() || h.is_request())),
        AgentsError::ModelTimeoutError { .. } => !stateful,
        _ => false,
    }
}

async fn with_timeout<T>(
    timeout: Option<f64>,
    fut: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match timeout {
        Some(secs) if secs > 0.0 => tokio::time::timeout(Duration::from_secs_f64(secs), fut)
            .await
            .map_err(|_| AgentsError::ModelTimeoutError {
                timeout_seconds: secs,
                run_data: None,
            })?,
        _ => fut.await,
    }
}

async fn consume_stream<C: Send + Sync + 'static>(
    model: &Arc<dyn Model>,
    request: ModelRequest,
    sink: &EventSink<C>,
    started: &mut bool,
) -> Result<ModelResponse> {
    let preserve = request.model_settings.preserve_raw_usage.unwrap_or(false);
    let mut stream = model.stream_response(request).await?;
    let mut final_response: Option<ModelResponse> = None;
    while let Some(ev) = stream.next().await {
        let ev = ev?;
        *started = true;
        if ev.get("type").and_then(Value::as_str) == Some("response.completed")
            && let Some(r) = ev.get("response")
        {
            final_response = Some(parse_response(r, None, preserve)?);
        }
        sink.send(StreamEvent::RawResponse(ev));
    }
    final_response.ok_or_else(|| {
        AgentsError::model_behavior("Model stream ended without a response.completed event")
    })
}

/// Call the model once (with retries), streaming raw events to `sink` when present.
pub(crate) async fn call_model<C: Send + Sync + 'static>(
    model: &Arc<dyn Model>,
    request: ModelRequest,
    sink: Option<&EventSink<C>>,
) -> Result<ModelResponse> {
    let settings = request.model_settings.clone();
    let stateful = request.previous_response_id.is_some() || request.conversation_id.is_some();
    let mut attempt: u32 = 0;
    let mut locked_attempts: u32 = 0;
    loop {
        let mut started = false;
        let result = match sink {
            Some(s) => {
                with_timeout(
                    settings.timeout,
                    consume_stream(model, request.clone(), s, &mut started),
                )
                .await
            }
            None => with_timeout(settings.timeout, model.get_response(request.clone())).await,
        };
        let err = match result {
            Ok(r) => return Ok(r),
            Err(e) => e,
        };
        if started {
            return Err(err);
        }
        if is_conversation_locked(&err) && locked_attempts < 3 {
            locked_attempts += 1;
            tokio::time::sleep(Duration::from_secs(1 << (locked_attempts - 1))).await;
            continue;
        }
        let Some(retry) = &settings.retry else {
            return Err(err);
        };
        if attempt >= retry.max_retries || !is_retryable(&err, &retry.retry_statuses, stateful) {
            return Err(err);
        }
        attempt += 1;
        log::warn!("Retrying model call (attempt {attempt}) after error: {err}");
        tokio::time::sleep(retry.delay_for_attempt(attempt)).await;
    }
}
