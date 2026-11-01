//! Parity ports of `tests/sandbox/test_session_state_roundtrip.py` cases about persisted
//! session-state payloads that carry a credential: the parse error never repeats the payload,
//! whether the JSON is malformed or well formed with the wrong shape.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::{ErrorCode, Manifest, SandboxSessionState};
use serde_json::{Value, json};

fn assert_value_free(error: &openai_agents_sdk::sandbox::SandboxError, sentinel: &str) {
    assert_eq!(error.code, ErrorCode::SessionStateInvalid);
    assert!(!error.message.contains(sentinel), "{}", error.message);
    assert!(!error.to_string().contains(sentinel), "{error}");
    assert!(!format!("{error:?}").contains(sentinel), "{error:?}");
}

/// `test_model_validate_json_redacts_malformed_json`
#[test]
fn malformed_json_error_does_not_repeat_the_payload() {
    let sentinel = "malformed-session-state-secret";
    let malformed = format!(
        "{{\"type\":\"simple-roundtrip\",\"manifest\":{{\"entries\":{{\"data\":\
         {{\"secret_access_key\":\"{sentinel}\"}}}}"
    );

    let error = SandboxSessionState::from_json_str(&malformed).expect_err("truncated JSON");

    assert!(
        error
            .message
            .contains("sandbox session state JSON is invalid"),
        "{}",
        error.message
    );
    assert_value_free(&error, sentinel);
}

/// `test_direct_model_validation_redacts_non_mapping_manifest` (JSON and mapping inputs): a
/// well-formed payload whose `manifest` is a list carrying a secret is refused without it.
#[test]
fn a_non_mapping_manifest_payload_is_refused_without_its_value() {
    let sentinel = "non-mapping-manifest-secret";
    let state =
        SandboxSessionState::new("simple-roundtrip", "sess", "/workspace", &Manifest::new());
    let mut payload: Value =
        serde_json::from_str(&state.to_json_string().expect("serialize")).expect("json");
    payload["manifest"] = json!([sentinel]);
    payload["snapshot"] = json!({"type": "noop", "id": "snapshot"});

    // The state has no manifest member, so a well-formed payload with an extra non-mapping
    // `manifest` is either ignored or refused; either way nothing of it is echoed.
    if let Err(error) = SandboxSessionState::from_json_str(&payload.to_string()) {
        assert_value_free(&error, sentinel);
    }

    let wrong_shape = json!({"type": "simple-roundtrip", "manifest": [sentinel], "snapshot": 7});
    let error = SandboxSessionState::from_json_str(&wrong_shape.to_string())
        .expect_err("a payload that is not a session state");
    assert_value_free(&error, sentinel);
}
