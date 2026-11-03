//! Ports of the portable cases the first corpus port skipped in
//! `tests/test_run_state_compatibility_corpus.py`: the legacy mount-credential fixture, the
//! deeply nested parser failure and the value-free restoration errors.

use serde_json::{Value, json};

use openai_agents_sdk::RunState;
use openai_agents_sdk::run_state::ContextRestoreOptions;

fn corpus(path: &str) -> Value {
    let full = format!(
        "{}/tests/fixtures/py_run_state_corpus/{path}",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(full).unwrap()).unwrap()
}

fn err_text<T>(result: Result<T, openai_agents_sdk::AgentsError>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

/// `test_v0194_mount_credentials_are_sanitized_and_require_trusted_rebind`: a snapshot written
/// by the 1.13 writer carries mount credentials in the manifest; reading and re-writing it
/// removes all three, marks the session state as redacted, and the rewrite is stable.
#[test]
fn legacy_mount_credentials_are_sanitized() {
    let sources = corpus("sources.json");
    let entry = &sources["security"];
    assert_eq!(entry["provenance"], "historical_writer");
    assert_eq!(entry["version"], "1.13");
    let sentinels: Vec<&str> = entry["sentinels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    assert_eq!(sentinels.len(), 3);
    let fixture = entry["fixture"].as_str().unwrap();
    let payload = corpus(fixture);
    let fixture_text = payload.to_string();
    assert!(sentinels.iter().all(|s| fixture_text.contains(s)));

    let first = RunState::from_json(payload).unwrap().to_json().unwrap();
    let text = first.to_string();
    for sentinel in &sentinels {
        assert!(!text.contains(sentinel), "{sentinel} leaked: {text}");
    }
    assert_eq!(
        first["sandbox"]["session_state"]["__openai_agents_redacted_mount_authority"],
        json!(true)
    );
    // The opaque driver options that carried the credentials are gone too.
    assert!(!text.contains("vfs-cache-mode"), "{text}");

    let second = RunState::from_json(first.clone())
        .unwrap()
        .to_json()
        .unwrap();
    assert_eq!(first, second, "the sanitized rewrite is not idempotent");
}

/// `test_run_state_from_string_deep_json_failure_discards_parser_state`: a hundred thousand
/// nested arrays fail cleanly, without a stack overflow and without echoing the payload.
#[test]
fn deeply_nested_json_fails_without_disclosing_the_payload() {
    let sentinel = "RUNSTATE_DEEP_JSON_PARSER_SENTINEL_42";
    let text = format!(
        "{}\"{sentinel}\"{}",
        "[".repeat(100_000),
        "]".repeat(100_000)
    );
    let message = err_text(RunState::from_json_str(&text));
    assert!(
        message.contains("Failed to parse run state JSON")
            || message.contains("Run state JSON must be an object"),
        "{message}"
    );
    assert!(!message.contains(sentinel), "{message}");
}

/// `test_run_state_from_string_deep_json_failure...` for a nesting the parser accepts: the
/// top level is not an object, so the snapshot is refused.
#[test]
fn shallow_non_object_json_is_refused() {
    let message = err_text(RunState::from_json_str("[\"RUNSTATE_ARRAY_SENTINEL_42\"]"));
    assert!(
        message.contains("Failed to parse run state JSON")
            || message.contains("Run state JSON must be an object")
            || message.to_lowercase().contains("object"),
        "{message}"
    );
    assert!(!message.contains("RUNSTATE_ARRAY_SENTINEL_42"), "{message}");
}

fn restore(
    payload: &Value,
    strict: bool,
    from_string: bool,
) -> Result<RunState, openai_agents_sdk::AgentsError> {
    let options = ContextRestoreOptions {
        context_override: None,
        strict_context: strict,
    };
    if from_string {
        RunState::from_json_str_with(&payload.to_string(), options)
    } else {
        RunState::from_json_with(payload.clone(), options)
    }
}

/// `test_run_state_payload_derived_restoration_errors_are_value_free` (context type and
/// completed call id carriers, `from_json` and `from_string`).
#[test]
fn payload_derived_restoration_errors_are_value_free() {
    for from_string in [false, true] {
        // context_type: strict restoration of a context that needs a deserializer.
        let sentinel = "RUNSTATE_CONTEXT_TYPE_SECRET_42";
        let mut payload = corpus("minimal/v1_15.json");
        payload["context"]["context_meta"] =
            json!({"original_type": sentinel, "requires_deserializer": true});
        let message = err_text(restore(&payload, true, from_string));
        assert!(
            message.contains("requires explicit restoration"),
            "{message}"
        );
        assert!(!message.contains(sentinel), "{message}");

        // completed_call_id: a malformed lifecycle entry keyed by a secret call id.
        let sentinel = "RUNSTATE_COMPLETED_CALL_ID_SECRET_42";
        let mut payload = corpus("features/v1_15_canonical_invocation_identity.json");
        let invocations = payload["context"]["tool_invocations"]
            .as_object_mut()
            .unwrap();
        let (_, mut invocation) = invocations
            .iter()
            .next()
            .map(|(k, v)| (k.clone(), v.clone()))
            .unwrap();
        invocation["completed"] = json!(true);
        payload["context"]["tool_invocations"] = json!({ sentinel: invocation });
        let message = err_text(restore(&payload, false, from_string));
        assert!(message.contains("invalid lifecycle data"), "{message}");
        assert!(!message.contains(sentinel), "{message}");
    }
}

/// `test_sdk_owned_run_state_validation_preserves_sanitized_user_error`: a `pending_input` that
/// is not a list is refused with the SDK's own message, not the payload.
#[test]
fn invalid_pending_input_is_refused_with_a_value_free_message() {
    for from_string in [false, true] {
        let sentinel = "RUNSTATE_INVALID_PENDING_INPUT_SECRET_42";
        let mut payload = corpus("minimal/v1_15.json");
        payload["pending_input"] = json!(sentinel);
        let message = err_text(restore(&payload, false, from_string));
        assert!(
            message.contains("pending_input must be a list"),
            "{message}"
        );
        assert!(!message.contains(sentinel), "{message}");
    }
}
