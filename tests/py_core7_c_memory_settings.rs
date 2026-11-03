//! Ports of `tests/memory/test_session.py::test_session_settings_constructor_normalizes_dictionary`,
//! `test_session_settings_rejects_unknown_dictionary_fields` and
//! `tests/memory/test_session_helpers.py::test_settings_normalization_and_limit_precedence`
//! (dictionary half): settings arrive as JSON objects, an unknown field is refused and a limit
//! that is not a number is rejected.

use openai_agents_sdk::memory::{SessionSettings, resolve_session_limit};
use serde_json::json;

#[test]
fn a_settings_object_normalizes_into_typed_settings() {
    let settings: SessionSettings = serde_json::from_value(json!({"limit": 0})).unwrap();
    assert_eq!(settings.limit, Some(0));
    assert_eq!(
        settings
            .resolve(Some(&SessionSettings::with_limit(4)))
            .limit,
        Some(4)
    );
    let from_dict: SessionSettings = serde_json::from_value(json!({"limit": 4})).unwrap();
    assert_eq!(resolve_session_limit(None, Some(&from_dict)), Some(4));
    let empty: SessionSettings = serde_json::from_value(json!({})).unwrap();
    assert_eq!(empty, SessionSettings::default());
}

#[test]
fn an_unknown_settings_field_is_refused() {
    let error = serde_json::from_value::<SessionSettings>(json!({"limitt": 1}))
        .expect_err("unknown field")
        .to_string();
    assert!(error.contains("unknown field `limitt`"), "{error}");
    let error = serde_json::from_value::<SessionSettings>(json!({"unknown": 1}))
        .expect_err("unknown field")
        .to_string();
    assert!(error.contains("unknown field `unknown`"), "{error}");
}

#[test]
fn a_limit_that_is_not_a_number_is_rejected() {
    assert!(serde_json::from_value::<SessionSettings>(json!({"limit": "not-a-number"})).is_err());
}
