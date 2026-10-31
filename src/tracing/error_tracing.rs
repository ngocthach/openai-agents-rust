//! Trace-safe rendering of errors recorded on spans. Mirrors `agents.util._error_tracing`
//! and `agents.util._tool_errors`.
//!
//! A span's `message` always says what failed; only the `error` detail is redacted, and only
//! when the run's `trace_include_sensitive_data` is off (`ModelTracing::EnabledWithoutData`).

use std::fmt::Display;

use serde_json::json;

use super::spans::Span;

/// Replaces an error detail that must not be exported.
pub const REDACTED_TRACE_ERROR_MESSAGE: &str = "Error details are redacted.";

/// Replaces a tool error detail that must not be exported.
pub const REDACTED_TOOL_ERROR_MESSAGE: &str = "Tool execution failed. Error details are redacted.";

/// Return a trace-safe error string, substituting [`REDACTED_TRACE_ERROR_MESSAGE`] when
/// sensitive data is excluded.
pub fn get_trace_error(trace_include_sensitive_data: bool, error_message: &str) -> String {
    if trace_include_sensitive_data {
        error_message.to_string()
    } else {
        REDACTED_TRACE_ERROR_MESSAGE.to_string()
    }
}

/// Return a trace-safe tool error string, substituting [`REDACTED_TOOL_ERROR_MESSAGE`] when
/// sensitive data is excluded.
pub fn get_trace_tool_error(trace_include_sensitive_data: bool, error_message: &str) -> String {
    if trace_include_sensitive_data {
        error_message.to_string()
    } else {
        REDACTED_TOOL_ERROR_MESSAGE.to_string()
    }
}

/// Render the span text for a failed model call.
///
/// The error is only stringified when its text will actually be exported, so a provider error
/// with an expensive or side-effecting `Display` is not invoked just to be thrown away. A
/// `Display` that fails is recorded as unrenderable instead of panicking, so the span still
/// carries the failure and the caller keeps its own error.
pub fn model_error_text(error: &dyn Display, trace_include_sensitive_data: bool) -> String {
    if !trace_include_sensitive_data {
        return REDACTED_TRACE_ERROR_MESSAGE.to_string();
    }
    let mut text = String::new();
    match std::fmt::Write::write_fmt(&mut text, format_args!("{error}")) {
        Ok(()) => text,
        Err(_) => "Unrenderable error".to_string(),
    }
}

/// Record a known model failure on its span, redacting the detail when required.
///
/// Streaming providers learn about a terminal failure before they raise it, and a consumer that
/// stops at that terminal event may never see the raise; recording at the point of knowledge
/// keeps the span accurate.
pub fn record_model_error_on_span(
    span: &Span,
    message: &str,
    error: &dyn Display,
    trace_include_sensitive_data: bool,
) {
    span.set_error(
        message,
        Some(json!({"error": model_error_text(error, trace_include_sensitive_data)})),
    );
}

/// Record a known tool failure on its span, redacting the detail when required.
pub fn record_tool_error_on_span(
    span: &Span,
    message: &str,
    error: &dyn Display,
    trace_include_sensitive_data: bool,
) {
    let text = if trace_include_sensitive_data {
        error.to_string()
    } else {
        REDACTED_TOOL_ERROR_MESSAGE.to_string()
    };
    span.set_error(message, Some(json!({"error": text})));
}

/// Mark the agent span of a failed run with the generic `Error in agent run` error.
///
/// Port of `attach_generic_agent_error`. Failures that a dedicated span reports (model behavior,
/// guardrail tripwires) are left alone, and so is a span that already carries a more specific
/// error. The error text is only rendered when sensitive data may be traced.
pub fn attach_generic_agent_error(
    span: Option<&Span>,
    error: &crate::error::AgentsError,
    trace_include_sensitive_data: bool,
) {
    use crate::error::AgentsError;
    let Some(span) = span else { return };
    if matches!(
        error,
        AgentsError::ModelBehaviorError { .. }
            | AgentsError::InputGuardrailTripwireTriggered { .. }
            | AgentsError::OutputGuardrailTripwireTriggered { .. }
    ) || span.error().is_some()
    {
        return;
    }
    let detail = if trace_include_sensitive_data {
        error.to_string()
    } else {
        REDACTED_TRACE_ERROR_MESSAGE.to_string()
    };
    span.set_error("Error in agent run", Some(json!({"error": detail})));
}

/// Record an error on the currently active span, if there is one.
///
/// Mirrors `attach_error_to_current_span`: without a span the failure is logged, and the log
/// itself is redacted when either data policy is on.
pub fn attach_error_to_current_span(message: &str, data: Option<serde_json::Value>) {
    match super::scope::current_span() {
        Some(span) => span.set_error(message, data),
        None => crate::logger::log_model_and_tool_data_warning(
            "No active span; trace error was not attached",
            || format!("No span to add error {message} to"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::SpanData;

    #[test]
    fn redacts_only_the_detail() {
        assert_eq!(get_trace_error(true, "boom"), "boom");
        assert_eq!(get_trace_error(false, "boom"), REDACTED_TRACE_ERROR_MESSAGE);
        assert_eq!(get_trace_tool_error(true, "boom"), "boom");
        assert_eq!(
            get_trace_tool_error(false, "boom"),
            REDACTED_TOOL_ERROR_MESSAGE
        );
    }

    #[test]
    fn records_a_redacted_error_on_a_span() {
        // An enabled span under a live trace; a span with no trace is a no-op.
        let parent = crate::tracing::trace("t").build();
        let span = crate::tracing::span(SpanData::Custom {
            name: "t".into(),
            data: serde_json::Value::Null,
        })
        .parent_trace(parent)
        .build();
        record_model_error_on_span(&span, "Error getting response", &"secret body", false);
        let err = span.error().expect("error recorded");
        assert_eq!(err.message, "Error getting response");
        assert_eq!(err.data.unwrap()["error"], REDACTED_TRACE_ERROR_MESSAGE);
    }
}

/// Where a model call's timeout is recorded before the call's future is dropped. The runner
/// creates one per attempt and scopes the attempt's future with it (see
/// [`scope_model_timeout`]); a timed-out attempt sets it, so the model span that the dropped
/// future owns can record the timeout. Port of `mark_model_timeout_task`.
#[derive(Clone, Debug, Default)]
pub struct ModelTimeoutSlot(std::sync::Arc<std::sync::Mutex<Option<f64>>>);

impl ModelTimeoutSlot {
    /// Record that the attempt timed out after `seconds`.
    pub fn mark_timed_out(&self, seconds: f64) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(seconds);
    }

    /// The timeout recorded so far, if the attempt timed out.
    pub fn timed_out_after(&self) -> Option<f64> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

tokio::task_local! {
    static MODEL_TIMEOUT_SLOT: ModelTimeoutSlot;
}

/// Run `fut` with `slot` as the timeout slot its model spans read.
pub fn scope_model_timeout<F: std::future::Future>(
    slot: ModelTimeoutSlot,
    fut: F,
) -> impl std::future::Future<Output = F::Output> {
    MODEL_TIMEOUT_SLOT.scope(slot, fut)
}

/// Finishes a model span when its call ends, and records a timeout on it if the attempt that owns
/// the call timed out. Dropping an armed guard means the call was cancelled, so the span must still
/// be finished, or it is never exported. Call [`ModelSpanGuard::disarm`] on the normal path, after
/// the outcome is recorded.
#[derive(Debug)]
pub struct ModelSpanGuard {
    span: Span,
    message: &'static str,
    include_data: bool,
    timeout: Option<ModelTimeoutSlot>,
    armed: bool,
}

impl ModelSpanGuard {
    /// Guard `span`, recording `message` if the call times out.
    pub fn new(span: &Span, message: &'static str, include_data: bool) -> Self {
        ModelSpanGuard {
            span: span.clone(),
            message,
            include_data,
            timeout: MODEL_TIMEOUT_SLOT.try_with(|s| s.clone()).ok(),
            armed: true,
        }
    }

    /// The guarded span.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// The call ended on its own path; the caller finishes the span.
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ModelSpanGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Some(seconds) = self
            .timeout
            .as_ref()
            .and_then(ModelTimeoutSlot::timed_out_after)
        {
            let error = crate::error::AgentsError::ModelTimeoutError {
                timeout_seconds: seconds,
                run_data: None,
            };
            record_model_error_on_span(&self.span, self.message, &error, self.include_data);
        }
        self.span.finish();
    }
}
