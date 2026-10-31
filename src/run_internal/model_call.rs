//! Model invocation with runner-managed retries, per-attempt timeouts and streaming.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::Value;

use super::EventSink;
use crate::error::{AgentsError, Result};
use crate::items::ModelResponse;
use crate::model_settings::ModelRetrySettings;
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
        Some(secs) if secs > 0.0 => {
            // The attempt's future owns the model span. It is only dropped after this function
            // returns, so the timeout is recorded in the slot first and the span can record it
            // when it is dropped. Python cancels the model task and marks it the same way.
            let slot = crate::tracing::ModelTimeoutSlot::default();
            let mut fut = std::pin::pin!(crate::tracing::scope_model_timeout(slot.clone(), fut));
            tokio::select! {
                biased;
                out = &mut fut => out,
                _ = tokio::time::sleep(Duration::from_secs_f64(secs)) => {
                    slot.mark_timed_out(secs);
                    Err(AgentsError::ModelTimeoutError {
                        timeout_seconds: secs,
                        run_data: None,
                    })
                }
            }
        }
        _ => fut.await,
    }
}

/// A response that aggregates several physical requests (a hosted continuation) reports how many
/// in `usage.requests` and each one in `usage.request_usage_entries`; keep both on the usage.
fn apply_aggregated_usage(parsed: &mut ModelResponse, response: &Value) {
    if let Some(n) = response.pointer("/usage/requests").and_then(Value::as_u64) {
        parsed.usage.requests = n;
    }
    if let Some(entries) = response
        .pointer("/usage/request_usage_entries")
        .and_then(Value::as_array)
    {
        parsed.usage.request_usage_entries = entries
            .iter()
            .map(|e| {
                let u = crate::usage::Usage::from_responses_usage(e);
                crate::usage::RequestUsage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    total_tokens: u.total_tokens,
                    input_tokens_details: u.input_tokens_details,
                    output_tokens_details: u.output_tokens_details,
                }
            })
            .collect();
    }
}

/// Stream events that do not block a retry (no user-visible output yet).
const RETRY_SAFE_STREAM_EVENTS: &[&str] = &["response.created", "response.in_progress"];

async fn consume_stream<C: Send + Sync + 'static>(
    model: &Arc<dyn Model>,
    request: ModelRequest,
    sink: &EventSink<C>,
    started: &mut bool,
) -> Result<ModelResponse> {
    let preserve = request.model_settings.preserve_raw_usage.unwrap_or(false);
    let mut stream = model.stream_response(request).await?;
    let mut final_response: Option<ModelResponse> = None;
    // Items the stream finished one by one. Some backends leave the terminal response's output
    // empty and only report items through `response.output_item.done`.
    let mut streamed_output: Vec<Value> = Vec::new();
    while let Some(ev) = stream.next().await {
        let ev = ev?;
        let ty = ev.get("type").and_then(Value::as_str).unwrap_or_default();
        if !RETRY_SAFE_STREAM_EVENTS.contains(&ty) {
            *started = true;
        }
        if ev.get("type").and_then(Value::as_str) == Some("response.completed")
            && let Some(r) = ev.get("response")
        {
            let backfilled;
            let r = if !streamed_output.is_empty()
                && r.get("output")
                    .and_then(Value::as_array)
                    .is_none_or(|output| output.is_empty())
            {
                let mut patched = r.clone();
                patched["output"] = Value::Array(streamed_output.clone());
                backfilled = patched;
                &backfilled
            } else {
                r
            };
            let mut parsed = parse_response(r, None, preserve)?;
            if ev.get("hosted_multi_agent_pause").is_some() {
                // A logical pause (a hosted response parked on a function call) carries no
                // usage of its own: the request is counted once, on the response that ends it.
                parsed.usage.requests = 0;
            }
            apply_aggregated_usage(&mut parsed, r);
            final_response = Some(parsed);
        }
        // A failed terminal event ends the run with the event still delivered as a raw
        // response. Port of the terminal-event branch of `run_single_turn_streamed`.
        let failure = match ty {
            "response.incomplete" | "response.failed" => Some(
                crate::models::response_terminal::response_terminal_failure_error(
                    ty,
                    ev.get("response"),
                ),
            ),
            "error" | "response.error" => {
                Some(crate::models::response_terminal::response_error_event_failure_error(ty, &ev))
            }
            _ => None,
        };
        if ty == "response.output_item.done"
            && let Some(item) = ev.get("item")
        {
            streamed_output.push(item.clone());
        }
        sink.send(StreamEvent::RawResponse(ev));
        if let Some(error) = failure {
            return Err(error);
        }
    }
    final_response.ok_or_else(|| {
        AgentsError::model_behavior("Model stream ended without a response.completed event")
    })
}

/// Whether provider-managed (hidden SDK) retries are off for this attempt. Port of
/// `_should_disable_provider_managed_retries`: they stay on for the first stateless attempt, and
/// are off wherever the runner owns the replay decision.
fn should_disable_provider_retries(
    retry: Option<&ModelRetrySettings>,
    attempt: u32,
    stateful: bool,
    replay_unsafe_request: bool,
) -> bool {
    if replay_unsafe_request {
        return true;
    }
    if let Some(retry) = retry
        && retry.max_retries == 0
    {
        // An explicit no-retry budget also disables hidden provider retries.
        return true;
    }
    if attempt > 1 {
        if stateful {
            return true;
        }
        return match retry {
            Some(retry) if retry.policy.is_some() => retry.max_retries > 0,
            _ => false,
        };
    }
    match retry {
        Some(retry) if stateful => retry.max_retries > 0 && retry.policy.is_some(),
        _ => false,
    }
}

/// Whether the WebSocket pre-event reconnect-and-resend is off. Port of
/// `_should_disable_websocket_pre_event_retry`.
fn should_disable_websocket_retry(
    retry: Option<&ModelRetrySettings>,
    replay_unsafe_request: bool,
) -> bool {
    if replay_unsafe_request {
        return true;
    }
    let Some(retry) = retry else {
        return false;
    };
    if retry.max_retries == 0 {
        return true;
    }
    retry
        .policy
        .as_ref()
        .is_some_and(|p| p.retries_safe_transport_errors)
}

/// The legacy `conversation_locked` retry stays unless the caller opts out with
/// `max_retries=0`. Port of `_should_preserve_conversation_locked_compatibility`.
fn preserves_conversation_locked_compatibility(retry: Option<&ModelRetrySettings>) -> bool {
    retry.is_none_or(|r| r.max_retries > 0)
}

async fn scope_retry_flags<T>(
    disable_provider: bool,
    disable_ws: bool,
    fut: impl std::future::Future<Output = T>,
) -> T {
    let provider_differs =
        crate::models::should_disable_provider_managed_retries() != disable_provider;
    let ws_differs = crate::models::should_disable_websocket_pre_event_retries() != disable_ws;
    match (provider_differs, ws_differs) {
        (false, false) => fut.await,
        (true, false) => {
            crate::models::provider_managed_retries_disabled(disable_provider, fut).await
        }
        (false, true) => crate::models::websocket_pre_event_retries_disabled(disable_ws, fut).await,
        (true, true) => {
            crate::models::provider_managed_retries_disabled(
                disable_provider,
                crate::models::websocket_pre_event_retries_disabled(disable_ws, fut),
            )
            .await
        }
    }
}

/// Call the model once (with retries), streaming raw events to `sink` when present.
///
/// `replay_unsafe_request` is set when the request carries application-local side effects that
/// may already have run (a Programmatic Tool Calling tool): hidden provider retries are off and
/// the runner only retries with the provider's own approval.
pub(crate) async fn call_model<C: Send + Sync + 'static>(
    model: &Arc<dyn Model>,
    request: ModelRequest,
    sink: Option<&EventSink<C>>,
    replay_unsafe_request: bool,
) -> Result<ModelResponse> {
    let settings = request.model_settings.clone();
    let stateful = request.previous_response_id.is_some() || request.conversation_id.is_some();
    let mut attempt: u32 = 0;
    let mut request_attempt: u32 = 1;
    let mut locked_attempts: u32 = 0;
    // Every attempt that failed and was retried still consumed a model request.
    let mut failed_attempts: u64 = 0;
    let disable_ws = should_disable_websocket_retry(settings.retry.as_ref(), replay_unsafe_request);
    loop {
        let mut started = false;
        let disable_provider = should_disable_provider_retries(
            settings.retry.as_ref(),
            request_attempt,
            stateful,
            replay_unsafe_request,
        );
        let attempt_future = async {
            match sink {
                Some(s) => {
                    with_timeout(
                        settings.timeout,
                        consume_stream(model, request.clone(), s, &mut started),
                    )
                    .await
                }
                None => with_timeout(settings.timeout, model.get_response(request.clone())).await,
            }
        };
        // A task-local scope costs a large poll frame on every call, so the flags are only
        // scoped when they differ from what the surrounding task already sees (an unscoped
        // flag reads `false`).
        let result = Box::pin(scope_retry_flags(
            disable_provider,
            disable_ws,
            attempt_future,
        ))
        .await;
        let err = match result {
            Ok(mut r) => {
                r.usage.apply_retry_attempt_usage(failed_attempts);
                return Ok(r);
            }
            Err(e) => e,
        };
        if started {
            return Err(err);
        }
        if !replay_unsafe_request
            && is_conversation_locked(&err)
            && preserves_conversation_locked_compatibility(settings.retry.as_ref())
            && locked_attempts < 3
        {
            locked_attempts += 1;
            failed_attempts += 1;
            request_attempt += 1;
            tokio::time::sleep(Duration::from_secs(1 << (locked_attempts - 1))).await;
            continue;
        }
        let Some(retry) = &settings.retry else {
            return Err(err);
        };
        if let Some(policy) = &retry.policy {
            let advice = model.get_retry_advice(&crate::retry::ModelRetryAdviceRequest {
                error: &err,
                attempt: attempt + 1,
                stream: sink.is_some(),
                previous_response_id: request.previous_response_id.as_deref(),
                conversation_id: request.conversation_id.as_deref(),
            });
            let decision = crate::retry::evaluate_retry_for(
                &err,
                attempt + 1,
                retry.max_retries,
                policy,
                sink.is_some(),
                replay_unsafe_request,
                started,
                advice.as_ref(),
                request.previous_response_id.as_deref(),
                request.conversation_id.as_deref(),
            );
            if !decision.retry {
                return Err(err);
            }
            attempt += 1;
            failed_attempts += 1;
            request_attempt += 1;
            let delay = decision
                .delay
                .map(|d| Duration::from_secs_f64(d.max(0.0)))
                .unwrap_or_else(|| retry.delay_for_attempt(attempt));
            crate::logger::log_model_action_warning(
                &format!("Retrying model call (attempt {attempt}) after error"),
                &err,
            );
            tokio::time::sleep(delay).await;
            continue;
        }
        if replay_unsafe_request
            || attempt >= retry.max_retries
            || !is_retryable(&err, &retry.retry_statuses, stateful)
        {
            return Err(err);
        }
        attempt += 1;
        failed_attempts += 1;
        request_attempt += 1;
        crate::logger::log_model_action_warning(
            &format!("Retrying model call (attempt {attempt}) after error"),
            &err,
        );
        tokio::time::sleep(retry.delay_for_attempt(attempt)).await;
    }
}
