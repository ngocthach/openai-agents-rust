//! Ports of the Python `ModelSettings.resolve` retry merge cases
//! (`tests/model_settings/test_serialization.py`).

use openai_agents_sdk::ModelSettings;
use openai_agents_sdk::model_settings::{ModelRetryBackoffSettings, ModelRetrySettings};
use openai_agents_sdk::retry::retry_policies;
use serde_json::json;

/// Python: `test_model_settings_dictionary_override_merges_retry_settings`.
#[test]
fn a_partial_retry_override_keeps_the_inherited_fields() {
    let settings = ModelSettings {
        retry: Some(ModelRetrySettings {
            max_retries: Some(2),
            backoff: Some(ModelRetryBackoffSettings {
                initial_delay: Some(0.1),
                jitter: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let over = ModelSettings::from_value(json!({
        "retry": {"max_retries": 0, "backoff": {"jitter": false}}
    }))
    .unwrap();

    let resolved = settings.resolve(Some(&over));

    let retry = resolved.retry.unwrap();
    assert_eq!(retry.max_retries, Some(0));
    let backoff = retry.backoff.unwrap();
    assert_eq!(backoff.initial_delay, Some(0.1));
    assert_eq!(backoff.jitter, Some(false));
}

/// Python: `test_retry_resolve_deep_merges_backoff`.
#[test]
fn resolve_deep_merges_backoff_and_takes_the_override_policy() {
    let base = ModelSettings {
        retry: Some(ModelRetrySettings {
            max_retries: Some(1),
            backoff: Some(ModelRetryBackoffSettings {
                initial_delay: Some(0.1),
                max_delay: Some(1.0),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let over = ModelSettings {
        retry: Some(
            ModelRetrySettings {
                backoff: Some(ModelRetryBackoffSettings {
                    multiplier: Some(3.0),
                    jitter: Some(false),
                    ..Default::default()
                }),
                ..Default::default()
            }
            .with_policy(retry_policies::never()),
        ),
        ..Default::default()
    };

    let resolved = base.resolve(Some(&over));

    let retry = resolved.retry.unwrap();
    assert_eq!(retry.max_retries, Some(1));
    assert!(retry.policy.is_some());
    assert_eq!(
        retry.backoff,
        Some(ModelRetryBackoffSettings {
            initial_delay: Some(0.1),
            max_delay: Some(1.0),
            multiplier: Some(3.0),
            jitter: Some(false),
        })
    );
}

/// Either side may be missing: the other one is used as is.
#[test]
fn resolve_uses_the_only_side_that_sets_retry() {
    let with_retry = ModelSettings {
        retry: Some(ModelRetrySettings::default().with_max_retries(4)),
        ..Default::default()
    };
    let empty = ModelSettings::default();

    assert_eq!(empty.resolve(Some(&with_retry)).retry, with_retry.retry);
    assert_eq!(with_retry.resolve(Some(&empty)).retry, with_retry.retry);
}
