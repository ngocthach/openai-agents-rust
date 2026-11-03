//! Port of `tests/models/test_openai_retry_helpers.py`: header parsing, status and error-code
//! extraction, and the `get_openai_retry_advice` branches the behavioral retry suite does not
//! reach.
//!
//! Python reads these facts off arbitrary exception objects (plain attributes, `__cause__`
//! chains, three header container types). A Rust error is the closed [`AgentsError`] enum whose
//! `ApiError` carries `status`, `body`, `request_id` and lowercase `headers`, so the extraction
//! is exercised through `ModelRetryNormalizedError::from_error`.

use std::collections::BTreeMap;

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::retry::{
    ModelRetryAdviceRequest, ModelRetryNormalizedError, ReplaySafety, get_openai_retry_advice,
};
use serde_json::{Value, json};

fn api_error(
    status: u16,
    body: Option<Value>,
    request_id: Option<&str>,
    headers: &[(&str, &str)],
) -> AgentsError {
    AgentsError::ApiError {
        status,
        message: "boom".into(),
        body: body.map(Box::new),
        request_id: request_id.map(str::to_string),
        headers: (!headers.is_empty()).then(|| {
            headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>()
        }),
    }
}

fn with_headers(headers: &[(&str, &str)]) -> AgentsError {
    api_error(400, None, None, headers)
}

fn retry_after(error: &AgentsError) -> Option<f64> {
    ModelRetryNormalizedError::from_error(error).retry_after
}

fn advice_for(error: &AgentsError) -> Option<openai_agents_sdk::retry::ModelRetryAdvice> {
    get_openai_retry_advice(&ModelRetryAdviceRequest {
        error,
        attempt: 1,
        stream: false,
        previous_response_id: None,
        conversation_id: None,
    })
}

// ---------------------------------------------------------------------------------------------
// retry-after parsing
// ---------------------------------------------------------------------------------------------

#[test]
fn an_invalid_retry_after_ms_is_ignored() {
    assert_eq!(retry_after(&with_headers(&[])), None);
    assert_eq!(
        retry_after(&with_headers(&[("retry-after-ms", "not-a-number")])),
        None
    );
    assert_eq!(
        retry_after(&with_headers(&[("retry-after-ms", "-100")])),
        None
    );
    assert_eq!(
        retry_after(&with_headers(&[("retry-after-ms", "1500")])),
        Some(1.5)
    );
}

#[test]
fn retry_after_accepts_seconds_and_http_dates() {
    assert_eq!(
        retry_after(&with_headers(&[("retry-after", "2")])),
        Some(2.0)
    );
    assert_eq!(retry_after(&with_headers(&[("retry-after", "-1")])), None);

    let future = chrono::Utc::now() + chrono::Duration::seconds(120);
    let parsed = retry_after(&with_headers(&[("retry-after", &future.to_rfc2822())]));
    // Not clamped: a date two minutes out is a wait of about two minutes.
    assert!(parsed.is_some_and(|d| d > 60.0), "{parsed:?}");

    let past = chrono::Utc::now() - chrono::Duration::seconds(120);
    assert_eq!(
        retry_after(&with_headers(&[("retry-after", &past.to_rfc2822())])),
        Some(0.0)
    );

    assert_eq!(
        retry_after(&with_headers(&[("retry-after", "definitely not a date")])),
        None
    );
}

#[test]
fn retry_after_ms_wins_over_retry_after() {
    let error = with_headers(&[("retry-after-ms", "1500"), ("retry-after", "9")]);
    assert_eq!(retry_after(&error), Some(1.5));
}

// ---------------------------------------------------------------------------------------------
// status and error code
// ---------------------------------------------------------------------------------------------

#[test]
fn the_status_code_comes_from_an_api_error_only() {
    let n = ModelRetryNormalizedError::from_error(&api_error(503, None, None, &[]));
    assert_eq!(n.status_code, Some(503));
    let n = ModelRetryNormalizedError::from_error(&AgentsError::other("none"));
    assert_eq!(n.status_code, None);
}

#[test]
fn the_error_code_comes_from_the_nested_or_top_level_body() {
    let nested = api_error(
        429,
        Some(json!({"error": {"code": "rate_limit_exceeded"}})),
        None,
        &[],
    );
    let top_level = api_error(500, Some(json!({"code": "server_error"})), None, &[]);
    assert_eq!(
        ModelRetryNormalizedError::from_error(&nested)
            .error_code
            .as_deref(),
        Some("rate_limit_exceeded")
    );
    assert_eq!(
        ModelRetryNormalizedError::from_error(&top_level)
            .error_code
            .as_deref(),
        Some("server_error")
    );
    assert_eq!(
        ModelRetryNormalizedError::from_error(&AgentsError::other("none")).error_code,
        None
    );
    // A nested code wins over a top-level one.
    let both = api_error(
        400,
        Some(json!({"code": "outer", "error": {"code": "inner"}})),
        None,
        &[],
    );
    assert_eq!(
        ModelRetryNormalizedError::from_error(&both)
            .error_code
            .as_deref(),
        Some("inner")
    );
}

#[test]
fn provider_advice_and_runner_normalization_share_metadata() {
    let error = api_error(
        429,
        Some(json!({"error": {"code": "rate_limit_exceeded"}})),
        Some("req_test"),
        &[("retry-after-ms", "1500")],
    );
    let advice = advice_for(&error).expect("advice");
    let from_advice = advice.normalized.expect("normalized");
    let runner = ModelRetryNormalizedError::from_error(&error);

    assert_eq!(from_advice.status_code, runner.status_code);
    assert_eq!(from_advice.error_code, runner.error_code);
    assert_eq!(from_advice.request_id, runner.request_id);
    assert_eq!(from_advice.retry_after, runner.retry_after);
    assert_eq!(runner.retry_after, Some(1.5));
}

#[tokio::test]
async fn a_refused_connection_is_a_network_error() {
    // Python checks the legacy and native httpx families; reqwest is the one Rust transport.
    let err = reqwest::Client::new()
        .get("http://127.0.0.1:1/")
        .send()
        .await
        .expect_err("nothing listens on port 1");
    let normalized = ModelRetryNormalizedError::from_error(&AgentsError::Http(err));
    assert!(normalized.is_network_error);
}

// ---------------------------------------------------------------------------------------------
// get_openai_retry_advice branches
// ---------------------------------------------------------------------------------------------

#[test]
fn an_ambiguous_websocket_request_is_unsafe_to_replay() {
    for message in [
        "The request may have been accepted, so the SDK will not automatically retry this websocket request.",
        "Responses websocket connection closed before a terminal response event.",
    ] {
        let advice = advice_for(&AgentsError::other(message)).expect("advice");
        assert_eq!(advice.suggested, Some(false), "{message}");
        assert_eq!(
            advice.replay_safety,
            Some(ReplaySafety::Unsafe),
            "{message}"
        );
    }
}

#[test]
fn advice_respects_x_should_retry_false() {
    let advice = advice_for(&with_headers(&[("x-should-retry", "false")])).expect("advice");
    assert_eq!(advice.suggested, Some(false));
}

#[test]
fn advice_conveys_only_the_delay_when_there_is_no_other_signal() {
    // A 400 with no `x-should-retry` and no network/timeout signal would not normally retry, but
    // a `retry-after` header still yields advice carrying the delay.
    let advice = advice_for(&with_headers(&[("retry-after", "2")])).expect("advice");
    assert_eq!(advice.retry_after, Some(2.0));
    // It does not assert a retry decision, so `suggested` keeps its unset default.
    assert_eq!(advice.suggested, None);
}
