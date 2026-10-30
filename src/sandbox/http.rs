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
    // A request that never reached the provider, or timed out on the way, is the canonical
    // transient failure: the same request can succeed on the next attempt.
    SandboxError::new(code, format!("Failed to {what}: {error}")).with_retryable(true)
}

/// Whether a provider is likely to answer the same request differently on a retry.
///
/// A 5xx or a 429 is the provider saying "not now"; every other 4xx is the provider saying the
/// request itself is wrong, and repeating it only costs quota. Mirrors Python's
/// `TRANSIENT_HTTP_STATUS_CODES` plus the rate-limit case the providers retry on.
pub fn status_is_retryable(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || crate::sandbox::retry::is_transient_http_status(status.as_u16())
        || (status.is_server_error() && !matches!(status, reqwest::StatusCode::NOT_IMPLEMENTED))
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
    // Every exec-driven retry in the provider backends reads this flag, so a 4xx has to say
    // "no" here or a misconfigured request is sent three times before it fails.
    SandboxError::new(
        code,
        format!("Failed to {what}: HTTP {} {detail}", status.as_u16()),
    )
    .with_retryable(status_is_retryable(status))
    .with_context("http_status", status.as_u16().to_string())
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
        // A success status with a body the SDK cannot parse is a contract mismatch, not a
        // hiccup; retrying would decode the same bytes again.
        SandboxError::new(
            ErrorCode::ExecTransportError,
            format!("Failed to {what}: unexpected response body ({e})"),
        )
        .with_retryable(false)
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
    fn a_client_error_is_not_retryable_but_a_server_error_is() {
        for (status, retryable) in [
            (reqwest::StatusCode::BAD_REQUEST, false),
            (reqwest::StatusCode::UNAUTHORIZED, false),
            (reqwest::StatusCode::NOT_FOUND, false),
            (reqwest::StatusCode::TOO_MANY_REQUESTS, true),
            (reqwest::StatusCode::INTERNAL_SERVER_ERROR, true),
            (reqwest::StatusCode::NOT_IMPLEMENTED, false),
            (reqwest::StatusCode::BAD_GATEWAY, true),
            (reqwest::StatusCode::SERVICE_UNAVAILABLE, true),
            (reqwest::StatusCode::GATEWAY_TIMEOUT, true),
        ] {
            let error = http_error(ErrorCode::ExecTransportError, "do a thing", status, "{}");
            assert_eq!(error.retryable, Some(retryable), "{status}");
            assert_eq!(
                error.context.get("http_status").map(String::as_str),
                Some(status.as_u16().to_string().as_str()),
            );
        }
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
