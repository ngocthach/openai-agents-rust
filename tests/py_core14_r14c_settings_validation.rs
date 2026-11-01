//! Ports of the settings validation cases of `tests/models/test_model_retry.py`: a negative
//! backoff value and a non-positive or non-finite model-call timeout are rejected when settings
//! are deserialized (Python raises `ValidationError`).

use openai_agents_sdk::model_settings::{
    ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
};
use serde_json::json;

#[test]
fn a_negative_backoff_value_is_rejected() {
    for field in ["initial_delay", "max_delay", "multiplier"] {
        let err = serde_json::from_value::<ModelRetryBackoffSettings>(json!({ field: -0.1 }))
            .expect_err("negative backoff value");
        assert!(
            err.to_string().contains("greater than or equal to 0"),
            "{field}: {err}"
        );
    }
}

#[test]
fn a_negative_backoff_nested_in_retry_settings_is_rejected() {
    let err = serde_json::from_value::<ModelRetrySettings>(
        json!({ "backoff": { "initial_delay": -0.1 } }),
    )
    .expect_err("negative nested backoff");
    assert!(
        err.to_string().contains("greater than or equal to 0"),
        "{err}"
    );
}

#[test]
fn zero_backoff_values_deserialize() {
    let backoff: ModelRetryBackoffSettings =
        serde_json::from_value(json!({ "initial_delay": 0, "max_delay": 0, "multiplier": 0 }))
            .unwrap();
    assert_eq!(backoff.initial_delay, Some(0.0));
    assert_eq!(backoff.max_delay, Some(0.0));
    assert_eq!(backoff.multiplier, Some(0.0));
}

#[test]
fn an_invalid_model_call_timeout_is_rejected() {
    for timeout in [json!(0), json!(-0.1), json!(-5)] {
        let result = serde_json::from_value::<ModelSettings>(json!({ "timeout": timeout }));
        assert!(result.is_err(), "timeout {timeout} must be rejected");
    }
    // JSON has no NaN or infinity literal; a non-finite float arrives as a string or null.
    let text = r#"{"timeout": 1e999}"#;
    assert!(serde_json::from_str::<ModelSettings>(text).is_err());
}

#[test]
fn a_finite_model_call_timeout_round_trips() {
    let settings: ModelSettings = serde_json::from_value(json!({ "timeout": 1.25 })).unwrap();
    assert_eq!(settings.timeout, Some(1.25));
    assert_eq!(settings.to_traceable_value()["timeout"], json!(1.25));
    let none: ModelSettings = serde_json::from_value(json!({ "timeout": null })).unwrap();
    assert_eq!(none.timeout, None);
}
