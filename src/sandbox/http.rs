//! Mapping HTTP transports onto [`SandboxError`].
//!
//! Every hosted sandbox backend talks to its provider over HTTP, and they all have to answer the
//! same three questions: did the request reach the provider, did it answer with a success status,
//! and did the body parse. These helpers give one answer to each, so a caller only supplies the
//! short description of what it was doing (`"create an E2B sandbox"`) and the
//! [`ErrorCode`] that fits the operation.

use serde_json::Value;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};

/// Map a `reqwest` failure to a transport error.
///
/// A timeout becomes [`ErrorCode::ExecTimeout`] so a caller can tell a slow provider from an
/// unreachable one.
pub fn transport(what: &str, error: reqwest::Error) -> SandboxError {
    let code = if error.is_timeout() {
        ErrorCode::ExecTimeout
    } else {
        ErrorCode::ExecTransportError
    };
    SandboxError::new(code, format!("Failed to {what}: {error}"))
}

/// Map a non-success HTTP response to a sandbox error.
///
/// Providers report failures as `{"message": ...}` or `{"error": {"message": ...}}`; either is
/// unwrapped, and anything else is truncated so a stray HTML error page cannot flood a log.
pub fn http_error(
    code: ErrorCode,
    what: &str,
    status: reqwest::StatusCode,
    body: &str,
) -> SandboxError {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("message")
                .or_else(|| value.get("error").and_then(|error| error.get("message")))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.trim().chars().take(400).collect());
    SandboxError::new(
        code,
        format!("Failed to {what}: HTTP {} {detail}", status.as_u16()),
    )
}

/// Decode a JSON body, turning a non-success status into an error.
pub async fn parse_json<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
    what: &str,
) -> SandboxResult<T> {
    let status = response.status();
    let body = response.text().await.map_err(|e| transport(what, e))?;
    if !status.is_success() {
        return Err(http_error(
            ErrorCode::ExecTransportError,
            what,
            status,
            &body,
        ));
    }
    serde_json::from_str(&body).map_err(|e| {
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("Failed to {what}: unexpected response body ({e})"),
        )
    })
}

/// Require a success status, discarding the body.
pub async fn expect_success(
    response: reqwest::Response,
    what: &str,
    code: ErrorCode,
) -> SandboxResult<()> {
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let body = response.text().await.unwrap_or_default();
    Err(http_error(code, what, status, &body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwraps_provider_error_messages() {
        let flat = http_error(
            ErrorCode::ExecTransportError,
            "create a sandbox",
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"code":400,"message":"template not found"}"#,
        );
        assert!(
            flat.message.contains("HTTP 400 template not found"),
            "{flat}"
        );

        let nested = http_error(
            ErrorCode::ExecTransportError,
            "create a sandbox",
            reqwest::StatusCode::FORBIDDEN,
            r#"{"error":{"message":"not entitled"}}"#,
        );
        assert!(nested.message.contains("HTTP 403 not entitled"), "{nested}");
    }

    #[test]
    fn truncates_a_body_that_is_not_json() {
        let error = http_error(
            ErrorCode::WorkspaceStartError,
            "create a sandbox",
            reqwest::StatusCode::BAD_GATEWAY,
            &"x".repeat(1000),
        );
        assert_eq!(error.code, ErrorCode::WorkspaceStartError);
        // The detail is bounded, so one HTML error page cannot flood a log line.
        assert!(error.message.len() < 500, "{}", error.message.len());
    }
}
