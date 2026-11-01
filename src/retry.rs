//! Runner-managed model retry policies. Mirrors `agents.retry`.
//!
//! A [`RetryPolicy`] decides whether a failed model call is retried. Policies see a normalized
//! view of the error plus the model's own [`ModelRetryAdvice`], and the runner applies replay
//! safety vetoes on top: requests that depend on server-side state (`previous_response_id`,
//! `conversation_id`) and failures the provider marks replay-unsafe are only retried with an
//! explicit approval.
//!
//! ```
//! use openai_agents_sdk::model_settings::ModelRetrySettings;
//! use openai_agents_sdk::retry::retry_policies;
//!
//! let retry = ModelRetrySettings::default().with_policy(retry_policies::any(vec![
//!     retry_policies::provider_suggested(),
//!     retry_policies::http_status([429, 503]),
//! ]));
//! ```

use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use crate::error::AgentsError;

/// Replay classification reported by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplaySafety {
    /// Replaying the request cannot duplicate provider-side work.
    Safe,
    /// The provider may already have acted on the request.
    Unsafe,
    /// No information.
    Unknown,
}

/// Normalized error facts exposed to retry policies.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelRetryNormalizedError {
    /// HTTP status code.
    pub status_code: Option<u16>,
    /// Provider error code (`error.code`).
    pub error_code: Option<String>,
    /// Error message.
    pub message: Option<String>,
    /// Provider request id.
    pub request_id: Option<String>,
    /// Server-suggested delay in seconds.
    pub retry_after: Option<f64>,
    /// The call was cancelled.
    pub is_abort: bool,
    /// A connection-level failure.
    pub is_network_error: bool,
    /// A timeout.
    pub is_timeout: bool,
    /// Fields a provider set on purpose, by name (`"retry_after"`, `"is_network_error"`, ...).
    ///
    /// Mirrors Python's `_explicit_fields`. When this value is used as a provider override, a
    /// listed field replaces the inferred one even if it is `None`/`false` (so a provider can
    /// clear a `retry_after` or deny a network error); an unlisted field overrides only when it
    /// carries a value.
    pub explicit_fields: Vec<&'static str>,
}

impl ModelRetryNormalizedError {
    /// Mark `fields` as explicitly set, so they override inferred values even when empty.
    pub fn with_explicit<I: IntoIterator<Item = &'static str>>(mut self, fields: I) -> Self {
        for f in fields {
            if !self.explicit_fields.contains(&f) {
                self.explicit_fields.push(f);
            }
        }
        self
    }

    fn is_explicit(&self, field: &str) -> bool {
        self.explicit_fields.contains(&field)
    }

    /// Normalize an SDK error.
    pub fn from_error(error: &AgentsError) -> Self {
        let mut n = ModelRetryNormalizedError {
            message: Some(error.to_string()),
            ..Default::default()
        };
        match error {
            AgentsError::ApiError {
                status,
                body,
                request_id,
                headers,
                ..
            } => {
                n.status_code = (*status != 0).then_some(*status);
                n.request_id = request_id.clone();
                // A nested `error.code` wins over a top-level `code`, as in `get_error_code`.
                n.error_code = body
                    .as_ref()
                    .and_then(|b| b.pointer("/error/code").or_else(|| b.get("code")))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                n.retry_after = headers.as_ref().and_then(parse_retry_after);
            }
            AgentsError::Http(h) => {
                n.is_timeout = h.is_timeout();
                n.is_network_error = h.is_connect() || h.is_request() || h.is_body();
                n.status_code = h.status().map(|s| s.as_u16());
            }
            AgentsError::ModelTimeoutError { .. } => n.is_timeout = true,
            _ => {}
        }
        n
    }
}

/// A response header by name, matched case-insensitively as HTTP header names are.
fn header_value<'a>(
    headers: &'a std::collections::BTreeMap<String, String>,
    name: &str,
) -> Option<&'a String> {
    headers.get(name).or_else(|| {
        headers
            .iter()
            .find_map(|(k, v)| k.eq_ignore_ascii_case(name).then_some(v))
    })
}

/// Parse `retry-after-ms` / `retry-after` (seconds or an HTTP date) into seconds.
///
/// Mirrors `parse_retry_after_ms` / `parse_retry_after_value`: the millisecond header wins, a
/// negative or unparsable value is ignored, and an HTTP date in the past is a zero delay. There
/// is no upper bound here; callers that must not wait long (the trace exporter, the runner's
/// own backoff ceiling) clamp the result themselves.
pub(crate) fn parse_retry_after(
    headers: &std::collections::BTreeMap<String, String>,
) -> Option<f64> {
    if let Some(ms) = header_value(headers, "retry-after-ms")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|ms| ms / 1000.0)
        .filter(|d| *d >= 0.0)
    {
        return Some(ms);
    }
    let value = header_value(headers, "retry-after")?.trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds >= 0.0).then_some(seconds);
    }
    chrono::DateTime::parse_from_rfc2822(value).ok().map(|t| {
        let delta =
            (t.with_timezone(&chrono::Utc) - chrono::Utc::now()).num_milliseconds() as f64 / 1000.0;
        delta.max(0.0)
    })
}

/// Provider-specific retry guidance returned by [`crate::models::Model::get_retry_advice`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelRetryAdvice {
    /// The provider suggests retrying (`Some(true)`), not retrying, or has no opinion.
    pub suggested: Option<bool>,
    /// Suggested delay in seconds.
    pub retry_after: Option<f64>,
    /// Replay classification.
    pub replay_safety: Option<ReplaySafety>,
    /// Human-readable reason.
    pub reason: Option<String>,
    /// Overrides for the normalized error facts.
    pub normalized: Option<ModelRetryNormalizedError>,
    /// The provider had begun emitting the response when the failure occurred.
    pub response_started: bool,
}

/// Context passed to a model when deriving retry advice.
#[derive(Debug, Clone, Copy)]
pub struct ModelRetryAdviceRequest<'a> {
    /// The error.
    pub error: &'a AgentsError,
    /// The attempt that failed (1-based).
    pub attempt: u32,
    /// Whether the call was streamed.
    pub stream: bool,
    /// `previous_response_id` of the request.
    pub previous_response_id: Option<&'a str>,
    /// `conversation_id` of the request.
    pub conversation_id: Option<&'a str>,
}

/// A retry decision returned by a policy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RetryDecision {
    /// Retry the call.
    pub retry: bool,
    /// Delay in seconds (`None` uses `retry_after` or the backoff settings).
    pub delay: Option<f64>,
    /// Reason, for logs.
    pub reason: Option<String>,
    /// Explicitly approve replaying a request the provider marked replay-unsafe. An ordinary
    /// `retry: true` never bypasses replay protection.
    pub approve_unsafe_replay: bool,
    hard_veto: bool,
    delegable_replay_veto: bool,
    approves_replay: bool,
}

impl RetryDecision {
    /// Retry with the default delay.
    pub fn retry() -> Self {
        RetryDecision {
            retry: true,
            ..Default::default()
        }
    }

    /// Do not retry.
    pub fn stop() -> Self {
        RetryDecision::default()
    }

    /// Set the delay in seconds.
    pub fn with_delay(mut self, delay: f64) -> Self {
        self.delay = Some(delay);
        self
    }

    /// Set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Approve replaying a provider-marked replay-unsafe request.
    pub fn with_unsafe_replay_approved(mut self) -> Self {
        self.approve_unsafe_replay = true;
        self
    }

    fn hard_vetoed(mut self) -> Self {
        self.hard_veto = true;
        self
    }

    fn replay_vetoed(mut self) -> Self {
        self.hard_veto = true;
        self.delegable_replay_veto = true;
        self
    }

    fn replay_approved(mut self) -> Self {
        self.approves_replay = true;
        self
    }
}

impl From<bool> for RetryDecision {
    fn from(retry: bool) -> Self {
        RetryDecision {
            retry,
            ..Default::default()
        }
    }
}

/// Context passed to retry policies.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicyContext<'a> {
    /// The error.
    pub error: &'a AgentsError,
    /// The retry being considered (1-based).
    pub attempt: u32,
    /// The retry budget.
    pub max_retries: u32,
    /// Whether the call was streamed.
    pub stream: bool,
    /// Normalized error facts.
    pub normalized: &'a ModelRetryNormalizedError,
    /// The model's advice.
    pub provider_advice: Option<&'a ModelRetryAdvice>,
    /// `previous_response_id` of the request.
    pub previous_response_id: Option<&'a str>,
    /// `conversation_id` of the request.
    pub conversation_id: Option<&'a str>,
}

impl RetryPolicyContext<'_> {
    /// Whether the provider had begun emitting the response.
    pub fn response_started(&self) -> bool {
        self.provider_advice.is_some_and(|a| a.response_started)
    }

    /// The provider replay classification.
    pub fn replay_safety(&self) -> ReplaySafety {
        self.provider_advice
            .and_then(|a| a.replay_safety)
            .unwrap_or(ReplaySafety::Unknown)
    }

    /// Whether the request depends on server-side conversation state.
    pub fn stateful_request(&self) -> bool {
        self.previous_response_id.is_some() || self.conversation_id.is_some()
    }
}

type PolicyFn = Arc<dyn Fn(&RetryPolicyContext<'_>) -> RetryDecision + Send + Sync>;

/// A retry policy. Build one with [`retry_policies`] or [`RetryPolicy::new`].
#[derive(Clone)]
pub struct RetryPolicy {
    f: PolicyFn,
    pub(crate) retries_safe_transport_errors: bool,
    pub(crate) retries_all_transient_errors: bool,
}

impl fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RetryPolicy(..)")
    }
}

impl PartialEq for RetryPolicy {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.f, &other.f)
    }
}

impl RetryPolicy {
    /// A custom policy returning a bool or a [`RetryDecision`].
    pub fn new<F, D>(f: F) -> Self
    where
        F: Fn(&RetryPolicyContext<'_>) -> D + Send + Sync + 'static,
        D: Into<RetryDecision>,
    {
        RetryPolicy {
            f: Arc::new(move |c| f(c).into()),
            retries_safe_transport_errors: false,
            retries_all_transient_errors: false,
        }
    }

    fn marked(self, safe_transport: bool, all_transient: bool) -> Self {
        RetryPolicy {
            retries_safe_transport_errors: safe_transport,
            retries_all_transient_errors: all_transient,
            ..self
        }
    }

    /// Evaluate the policy.
    pub fn evaluate(&self, ctx: &RetryPolicyContext<'_>) -> RetryDecision {
        (self.f)(ctx)
    }
}

fn merge_positive(existing: RetryDecision, incoming: &RetryDecision) -> RetryDecision {
    let mut merged = RetryDecision {
        retry: true,
        delay: incoming.delay.or(existing.delay),
        reason: incoming.reason.clone().or(existing.reason),
        approve_unsafe_replay: existing.approve_unsafe_replay || incoming.approve_unsafe_replay,
        ..Default::default()
    };
    merged.approves_replay = existing.approves_replay || incoming.approves_replay;
    merged
}

fn resolve_replay_veto(veto: RetryDecision, approving: RetryDecision) -> RetryDecision {
    if !approving.retry || !approving.approve_unsafe_replay {
        return veto;
    }
    RetryDecision {
        retry: true,
        delay: approving.delay,
        reason: approving.reason.or(veto.reason),
        approve_unsafe_replay: true,
        approves_replay: approving.approves_replay,
        ..Default::default()
    }
}

/// Built-in retry policies. Mirrors `agents.retry_policies`.
pub mod retry_policies {
    use super::*;

    /// Never retry.
    pub fn never() -> RetryPolicy {
        RetryPolicy::new(|_| false).marked(false, false)
    }

    /// Follow the provider's advice (including its replay-safety classification).
    pub fn provider_suggested() -> RetryPolicy {
        RetryPolicy::new(|c: &RetryPolicyContext<'_>| {
            let Some(advice) = c.provider_advice else {
                return RetryDecision::stop();
            };
            let reason = advice.reason.clone();
            match advice.suggested {
                None => RetryDecision::stop(),
                Some(false) => {
                    let d = RetryDecision {
                        reason,
                        ..Default::default()
                    };
                    if c.replay_safety() == ReplaySafety::Unsafe {
                        d.replay_vetoed()
                    } else {
                        d.hard_vetoed()
                    }
                }
                Some(true) => {
                    let d = RetryDecision {
                        retry: true,
                        delay: advice.retry_after,
                        reason,
                        ..Default::default()
                    };
                    if c.replay_safety() == ReplaySafety::Safe {
                        d.replay_approved()
                    } else {
                        d
                    }
                }
            }
        })
        .marked(true, false)
    }

    /// Retry network errors and timeouts.
    pub fn network_error() -> RetryPolicy {
        RetryPolicy::new(|c: &RetryPolicyContext<'_>| {
            c.normalized.is_network_error || c.normalized.is_timeout
        })
        .marked(true, false)
    }

    /// Retry when the server suggested a delay.
    pub fn retry_after() -> RetryPolicy {
        RetryPolicy::new(|c: &RetryPolicyContext<'_>| {
            let delay = c
                .normalized
                .retry_after
                .or_else(|| c.provider_advice.and_then(|a| a.retry_after));
            match delay {
                Some(d) => RetryDecision::retry().with_delay(d),
                None => RetryDecision::stop(),
            }
        })
        .marked(false, false)
    }

    /// Retry the given HTTP statuses.
    pub fn http_status(statuses: impl IntoIterator<Item = u16>) -> RetryPolicy {
        let allowed: std::collections::HashSet<u16> = statuses.into_iter().collect();
        RetryPolicy::new(move |c: &RetryPolicyContext<'_>| {
            c.normalized
                .status_code
                .is_some_and(|s| allowed.contains(&s))
        })
        .marked(false, false)
    }

    /// Retry only when every policy agrees.
    pub fn all(policies: Vec<RetryPolicy>) -> RetryPolicy {
        if policies.is_empty() {
            return never();
        }
        let safe = policies.iter().all(|p| p.retries_safe_transport_errors);
        let transient = policies.iter().all(|p| p.retries_all_transient_errors);
        RetryPolicy::new(move |c: &RetryPolicyContext<'_>| {
            let mut merged = RetryDecision::retry();
            let mut replay_veto: Option<RetryDecision> = None;
            for p in &policies {
                let d = p.evaluate(c);
                if d.hard_veto {
                    if d.delegable_replay_veto {
                        replay_veto.get_or_insert(d);
                        continue;
                    }
                    return d;
                }
                if !d.retry {
                    return d;
                }
                if d.delay.is_some() {
                    merged.delay = d.delay;
                }
                if d.reason.is_some() {
                    merged.reason = d.reason.clone();
                }
                merged.approve_unsafe_replay |= d.approve_unsafe_replay;
                merged.approves_replay |= d.approves_replay;
            }
            match replay_veto {
                Some(v) => resolve_replay_veto(v, merged),
                None => merged,
            }
        })
        .marked(safe, transient)
    }

    /// Retry when any policy agrees (hard vetoes still win).
    pub fn any(policies: Vec<RetryPolicy>) -> RetryPolicy {
        if policies.is_empty() {
            return never();
        }
        let safe = policies.iter().any(|p| p.retries_safe_transport_errors);
        let transient = policies.iter().any(|p| p.retries_all_transient_errors);
        RetryPolicy::new(move |c: &RetryPolicyContext<'_>| {
            let mut first_positive: Option<RetryDecision> = None;
            let mut last_negative: Option<RetryDecision> = None;
            let mut replay_veto: Option<RetryDecision> = None;
            for p in &policies {
                let d = p.evaluate(c);
                if d.hard_veto {
                    if d.delegable_replay_veto {
                        replay_veto.get_or_insert(d);
                        continue;
                    }
                    return d;
                }
                if d.retry {
                    first_positive = Some(match first_positive {
                        None => d,
                        Some(existing) => merge_positive(existing, &d),
                    });
                    continue;
                }
                last_negative = Some(d);
            }
            if let Some(v) = replay_veto {
                return match first_positive {
                    None => v,
                    Some(p) => resolve_replay_veto(v, p),
                };
            }
            first_positive
                .or(last_negative)
                .unwrap_or_else(RetryDecision::stop)
        })
        .marked(safe, transient)
    }
}

/// A WebSocket request that was sent but never acknowledged: the server may have accepted it,
/// so replaying it could duplicate the response.
pub(crate) const AMBIGUOUS_WEBSOCKET_REPLAY_MESSAGE: &str =
    "Responses websocket connection closed before a terminal response event.";

/// The sentence Python's websocket adapter puts in an ambiguous-replay error (lowercase).
const PYTHON_AMBIGUOUS_WEBSOCKET_REPLAY_MESSAGE: &str = "the request may have been accepted, so \
     the sdk will not automatically retry this websocket request.";

/// Retry advice for OpenAI HTTP models (mirrors the OpenAI SDK's own retry rules).
pub fn get_openai_retry_advice(request: &ModelRetryAdviceRequest<'_>) -> Option<ModelRetryAdvice> {
    let error = request.error;
    let normalized = ModelRetryNormalizedError::from_error(error);
    let reason = Some(error.to_string());
    // The request may have been accepted, so the SDK will not automatically retry it.
    if normalized.message.as_deref().is_some_and(|m| {
        m.starts_with(AMBIGUOUS_WEBSOCKET_REPLAY_MESSAGE)
            || m.to_lowercase()
                .contains(PYTHON_AMBIGUOUS_WEBSOCKET_REPLAY_MESSAGE)
    }) {
        return Some(ModelRetryAdvice {
            suggested: Some(false),
            replay_safety: Some(ReplaySafety::Unsafe),
            reason,
            normalized: Some(normalized),
            ..Default::default()
        });
    }
    let stateful = request.previous_response_id.is_some() || request.conversation_id.is_some();
    if let AgentsError::ApiError {
        headers: Some(h), ..
    } = error
        && let Some(h) = header_value(h, "x-should-retry").and_then(|v| {
            match v.trim().to_ascii_lowercase().as_str() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }
        })
    {
        return Some(ModelRetryAdvice {
            suggested: Some(h),
            retry_after: normalized.retry_after,
            replay_safety: h.then_some(ReplaySafety::Safe),
            reason,
            normalized: Some(normalized),
            response_started: false,
        });
    }
    if normalized.is_network_error || normalized.is_timeout {
        return Some(ModelRetryAdvice {
            suggested: Some(true),
            retry_after: normalized.retry_after,
            reason,
            normalized: Some(normalized),
            ..Default::default()
        });
    }
    if let Some(status) = normalized.status_code
        && (matches!(status, 408 | 409 | 429) || status >= 500)
    {
        return Some(ModelRetryAdvice {
            suggested: Some(true),
            retry_after: normalized.retry_after,
            replay_safety: stateful.then_some(ReplaySafety::Safe),
            reason,
            normalized: Some(normalized),
            response_started: false,
        });
    }
    normalized.retry_after.map(|d| ModelRetryAdvice {
        retry_after: Some(d),
        reason,
        normalized: Some(normalized),
        ..Default::default()
    })
}

/// [`evaluate_retry_for`] for a request without a request-level replay veto.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_retry(
    error: &AgentsError,
    attempt: u32,
    max_retries: u32,
    policy: &RetryPolicy,
    stream: bool,
    emitted_retry_unsafe_event: bool,
    provider_advice: Option<&ModelRetryAdvice>,
    previous_response_id: Option<&str>,
    conversation_id: Option<&str>,
) -> RetryDecision {
    evaluate_retry_for(
        error,
        attempt,
        max_retries,
        policy,
        stream,
        false,
        emitted_retry_unsafe_event,
        provider_advice,
        previous_response_id,
        conversation_id,
    )
}

/// Apply the runner's replay-safety rules around a policy decision.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_retry_for(
    error: &AgentsError,
    attempt: u32,
    max_retries: u32,
    policy: &RetryPolicy,
    stream: bool,
    replay_unsafe_request: bool,
    emitted_retry_unsafe_event: bool,
    provider_advice: Option<&ModelRetryAdvice>,
    previous_response_id: Option<&str>,
    conversation_id: Option<&str>,
) -> RetryDecision {
    if attempt > max_retries {
        return RetryDecision::stop();
    }
    let mut normalized = ModelRetryNormalizedError::from_error(error);
    if let Some(a) = provider_advice {
        if a.retry_after.is_some() {
            normalized.retry_after = a.retry_after;
        }
        if let Some(o) = &a.normalized {
            if o.status_code.is_some() || o.is_explicit("status_code") {
                normalized.status_code = o.status_code;
            }
            if o.error_code.is_some() || o.is_explicit("error_code") {
                normalized.error_code = o.error_code.clone();
            }
            if o.message.is_some() || o.is_explicit("message") {
                normalized.message = o.message.clone();
            }
            if o.request_id.is_some() || o.is_explicit("request_id") {
                normalized.request_id = o.request_id.clone();
            }
            if o.retry_after.is_some() || o.is_explicit("retry_after") {
                normalized.retry_after = o.retry_after;
            }
            if o.is_network_error || o.is_explicit("is_network_error") {
                normalized.is_network_error = o.is_network_error;
            }
            if o.is_timeout || o.is_explicit("is_timeout") {
                normalized.is_timeout = o.is_timeout;
            }
            // Provider normalization may add abort evidence but cannot clear an abort inferred
            // from the raw error.
            normalized.is_abort |= o.is_abort;
        }
    }
    let ctx = RetryPolicyContext {
        error,
        attempt,
        max_retries,
        stream,
        normalized: &normalized,
        provider_advice,
        previous_response_id,
        conversation_id,
    };
    let advice_reason = || provider_advice.and_then(|a| a.reason.clone());
    let unsafe_replay = ctx.replay_safety() == ReplaySafety::Unsafe;
    let safe_replay = ctx.replay_safety() == ReplaySafety::Safe;
    // Aborts and failures that already emitted user-visible streamed output are absolute vetoes.
    // A provider-unsafe streamed failure, or one on a request with a separate local-side-effect
    // veto, stays blocked before the policy runs.
    if normalized.is_abort
        || emitted_retry_unsafe_event
        || (unsafe_replay && (stream || replay_unsafe_request))
    {
        return RetryDecision {
            reason: advice_reason(),
            ..Default::default()
        };
    }
    let decision = policy.evaluate(&ctx);
    if !decision.retry {
        return decision;
    }
    let veto = |d: &RetryDecision| RetryDecision {
        reason: d.reason.clone().or_else(advice_reason),
        ..Default::default()
    };
    // A request-level replay veto covers application-local side effects, which only the
    // provider's own approval lifts.
    if replay_unsafe_request && !(decision.approves_replay || safe_replay) {
        return veto(&decision);
    }
    // Stateful requests fail closed unless the replay is approved.
    if ctx.stateful_request()
        && !(decision.approves_replay
            || safe_replay
            || (decision.approve_unsafe_replay && unsafe_replay))
    {
        return veto(&decision);
    }
    if unsafe_replay && !(decision.approves_replay || decision.approve_unsafe_replay) {
        return veto(&decision);
    }
    RetryDecision {
        retry: true,
        delay: decision.delay.or(normalized.retry_after),
        reason: decision.reason.clone().or_else(advice_reason),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api(status: u16) -> AgentsError {
        AgentsError::ApiError {
            status,
            message: "x".into(),
            body: None,
            request_id: None,
            headers: None,
        }
    }

    #[test]
    fn reads_retry_headers() {
        let mut h = std::collections::BTreeMap::new();
        h.insert("retry-after".to_string(), "2".to_string());
        h.insert("x-should-retry".to_string(), "true".to_string());
        let err = AgentsError::ApiError {
            status: 400,
            message: "x".into(),
            body: None,
            request_id: None,
            headers: Some(h),
        };
        assert_eq!(
            ModelRetryNormalizedError::from_error(&err).retry_after,
            Some(2.0)
        );
        let advice = get_openai_retry_advice(&ModelRetryAdviceRequest {
            error: &err,
            attempt: 1,
            stream: false,
            previous_response_id: None,
            conversation_id: None,
        })
        .unwrap();
        assert_eq!(advice.suggested, Some(true));
        assert_eq!(advice.replay_safety, Some(ReplaySafety::Safe));
        let d = evaluate_retry(
            &err,
            1,
            1,
            &retry_policies::retry_after(),
            false,
            false,
            Some(&advice),
            None,
            None,
        );
        assert_eq!(d.delay, Some(2.0));
    }

    #[test]
    fn composes_policies_with_replay_rules() {
        let policy = retry_policies::any(vec![
            retry_policies::provider_suggested(),
            retry_policies::http_status([418]),
        ]);
        let err = api(503);
        let advice = get_openai_retry_advice(&ModelRetryAdviceRequest {
            error: &err,
            attempt: 1,
            stream: false,
            previous_response_id: None,
            conversation_id: None,
        });
        let d = evaluate_retry(
            &err,
            1,
            2,
            &policy,
            false,
            false,
            advice.as_ref(),
            None,
            None,
        );
        assert!(d.retry);
        assert!(
            !evaluate_retry(
                &err,
                3,
                2,
                &policy,
                false,
                false,
                advice.as_ref(),
                None,
                None
            )
            .retry
        );

        // Stateful requests need provider-approved replay: a 418 with no advice is vetoed.
        let teapot = api(418);
        let d = evaluate_retry(
            &teapot,
            1,
            2,
            &policy,
            false,
            false,
            None,
            Some("resp_1"),
            None,
        );
        assert!(!d.retry);
        // A stateful 503 is replay-safe per the OpenAI advice.
        let advice = get_openai_retry_advice(&ModelRetryAdviceRequest {
            error: &err,
            attempt: 1,
            stream: false,
            previous_response_id: Some("resp_1"),
            conversation_id: None,
        });
        let d = evaluate_retry(
            &err,
            1,
            2,
            &policy,
            false,
            false,
            advice.as_ref(),
            Some("resp_1"),
            None,
        );
        assert!(d.retry);

        // Provider says no: hard veto even though http_status would retry.
        let no = ModelRetryAdvice {
            suggested: Some(false),
            ..Default::default()
        };
        let d = evaluate_retry(&teapot, 1, 2, &policy, false, false, Some(&no), None, None);
        assert!(!d.retry);
        assert!(
            !retry_policies::all(vec![])
                .evaluate(&RetryPolicyContext {
                    error: &err,
                    attempt: 1,
                    max_retries: 1,
                    stream: false,
                    normalized: &ModelRetryNormalizedError::default(),
                    provider_advice: None,
                    previous_response_id: None,
                    conversation_id: None,
                })
                .retry
        );
    }
}
