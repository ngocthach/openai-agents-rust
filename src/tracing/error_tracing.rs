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
/// with an expensive or side-effecting `Display` is not invoked just to be thrown away.
pub fn model_error_text(error: &dyn Display, trace_include_sensitive_data: bool) -> String {
    if trace_include_sensitive_data {
        error.to_string()
    } else {
        REDACTED_TRACE_ERROR_MESSAGE.to_string()
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
    use crate::tracing::{SpanData, create_span};

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
        let span = create_span(
            SpanData::Custom {
                name: "t".into(),
                data: serde_json::Value::Null,
            },
            None,
            true,
        );
        record_model_error_on_span(&span, "Error getting response", &"secret body", false);
        let err = span.error().expect("error recorded");
        assert_eq!(err.message, "Error getting response");
        assert_eq!(err.data.unwrap()["error"], REDACTED_TRACE_ERROR_MESSAGE);
    }
}
