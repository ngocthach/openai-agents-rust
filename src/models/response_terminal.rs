//! Terminal Responses-stream events rendered as errors. Mirrors
//! `agents.models._response_terminal`.

use serde_json::Value;

use crate::error::AgentsError;

fn detail(out: &mut Vec<String>, key: &str, value: Option<&Value>) {
    let Some(value) = value else { return };
    match value {
        Value::Null => {}
        Value::String(s) if s.is_empty() => {}
        Value::String(s) => out.push(format!("{key}={s}")),
        other => out.push(format!("{key}={other}")),
    }
}

/// Describe a `response.failed` / `response.incomplete` event.
pub fn format_response_terminal_failure(event_type: &str, response: Option<&Value>) -> String {
    let message = format!("{TERMINAL_PREFIX}{event_type}`.");
    let Some(response) = response.filter(|r| !r.is_null()) else {
        return message;
    };
    let mut details = Vec::new();
    detail(&mut details, "status", response.get("status"));
    detail(&mut details, "error", response.get("error"));
    detail(
        &mut details,
        "incomplete_details",
        response.get("incomplete_details"),
    );
    if details.is_empty() {
        message
    } else {
        format!("{message} {}.", details.join("; "))
    }
}

/// Describe an `error` / `response.error` event.
pub fn format_response_error_event(event_type: &str, event: &Value) -> String {
    let message = format!("{TERMINAL_PREFIX}{event_type}`.");
    // The OpenAI stream sends the fields at the top level; some gateways nest them under
    // `error`, so fall back to that shape.
    let source = event
        .get("error")
        .filter(|e| e.is_object())
        .unwrap_or(event);
    let mut details = Vec::new();
    detail(&mut details, "code", source.get("code"));
    detail(&mut details, "message", source.get("message"));
    detail(&mut details, "param", source.get("param"));
    if details.is_empty() {
        message
    } else {
        format!("{message} {}.", details.join("; "))
    }
}

/// The error raised for a `response.failed` / `response.incomplete` stream event.
pub fn response_terminal_failure_error(event_type: &str, response: Option<&Value>) -> AgentsError {
    AgentsError::model_behavior(format_response_terminal_failure(event_type, response))
}

/// The error raised for an `error` / `response.error` stream event.
pub fn response_error_event_failure_error(event_type: &str, event: &Value) -> AgentsError {
    AgentsError::model_behavior(format_response_error_event(event_type, event))
}

/// Prefix shared by every terminal-failure message this module builds.
const TERMINAL_PREFIX: &str = "Responses stream ended with terminal event `";

/// Whether the already-queued stream events should be delivered to the consumer before this
/// error is raised. Port of `_should_drain_stream_events_before_raising`.
///
/// Python marks the exception object itself; Rust's `AgentsError` carries no attributes, so the
/// marker is the message shape every terminal-failure error here has. A streaming run result
/// should drain its event queue before surfacing such an error, so the caller still sees the
/// output the model produced before the stream failed.
pub fn should_drain_stream_events_before_raising(error: &AgentsError) -> bool {
    matches!(error, AgentsError::ModelBehaviorError { message, .. } if message.starts_with(TERMINAL_PREFIX))
}

/// Whether `event_type` ends a Responses stream in failure.
pub fn is_terminal_failure_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "response.failed" | "response.incomplete" | "error" | "response.error"
    )
}

/// Build the error for any terminal failure event type.
pub fn terminal_failure_error(event_type: &str, event: &Value) -> AgentsError {
    match event_type {
        "response.failed" | "response.incomplete" => {
            response_terminal_failure_error(event_type, event.get("response"))
        }
        _ => response_error_event_failure_error(event_type, event),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn formats_a_terminal_response() {
        let event = json!({"type": "response.incomplete", "response": {
            "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}});
        assert_eq!(
            format_response_terminal_failure("response.incomplete", event.get("response")),
            "Responses stream ended with terminal event `response.incomplete`. \
             status=incomplete; incomplete_details={\"reason\":\"max_output_tokens\"}."
        );
        assert_eq!(
            format_response_terminal_failure("response.failed", None),
            "Responses stream ended with terminal event `response.failed`."
        );
    }

    #[test]
    fn formats_an_error_event() {
        let event =
            json!({"type": "error", "code": "server_error", "message": "boom", "param": null});
        assert_eq!(
            format_response_error_event("error", &event),
            "Responses stream ended with terminal event `error`. code=server_error; message=boom."
        );
        let nested = json!({"type": "response.error", "error": {"message": "nope"}});
        assert_eq!(
            format_response_error_event("response.error", &nested),
            "Responses stream ended with terminal event `response.error`. message=nope."
        );
    }
}
