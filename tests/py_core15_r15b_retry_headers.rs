//! Port of the header-lookup half of the Python SDK's `tests/models/test_openai_retry_helpers.py`:
//! response header names match case-insensitively, so a mixed-case `Retry-After` still feeds the
//! normalized retry delay.

use std::collections::BTreeMap;

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::retry::ModelRetryNormalizedError;

fn error_with_headers(headers: &[(&str, &str)]) -> AgentsError {
    AgentsError::ApiError {
        status: 400,
        message: "boom".into(),
        body: None,
        request_id: None,
        headers: Some(
            headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
        ),
    }
}

#[test]
fn header_names_match_case_insensitively() {
    for (name, value) in [
        ("Retry-After", "5"),
        ("RETRY-AFTER", "5"),
        ("retry-after", "5"),
    ] {
        let error = error_with_headers(&[(name, value), ("X-Other", "ignored")]);
        assert_eq!(
            ModelRetryNormalizedError::from_error(&error).retry_after,
            Some(5.0),
            "{name}"
        );
    }
    let error = error_with_headers(&[("Retry-After-Ms", "1500")]);
    assert_eq!(
        ModelRetryNormalizedError::from_error(&error).retry_after,
        Some(1.5)
    );
    let error = error_with_headers(&[("X-Other", "ignored")]);
    assert_eq!(
        ModelRetryNormalizedError::from_error(&error).retry_after,
        None
    );
}
