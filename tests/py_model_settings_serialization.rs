//! Port of `tests/model_settings/test_serialization.py`.
//!
//! Python serializes `ModelSettings` twice: `to_json_dict()` (everything that is JSON safe) and
//! `to_traceable_dict()` (the subset safe to record in a trace). In Rust those are the derived
//! `Serialize` impl and [`ModelSettings::to_traceable_value`].

use std::collections::HashMap;

use openai_agents_sdk::model_settings::{
    MCPToolChoice, ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings, Reasoning,
    ToolChoice,
};
use serde_json::{Value, json};

/// Every settings value must survive a round trip through JSON.
fn verify_serialization(settings: &ModelSettings) -> Value {
    let value = serde_json::to_value(settings).expect("ModelSettings must serialize");
    let text = serde_json::to_string(&value).expect("the JSON value must serialize");
    assert!(!text.is_empty());
    value
}

fn string_map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn object(value: Value) -> serde_json::Map<String, Value> {
    value.as_object().unwrap().clone()
}

// ---------------------------------------------------------------------------------------------
// basic serialization
// ---------------------------------------------------------------------------------------------

#[test]
fn basic_serialization() {
    let settings = ModelSettings {
        temperature: Some(0.5),
        top_p: Some(0.9),
        max_tokens: Some(100),
        ..Default::default()
    };
    verify_serialization(&settings);
}

#[test]
fn mcp_tool_choice_serialization() {
    let settings = ModelSettings {
        temperature: Some(0.5),
        tool_choice: Some(ToolChoice::Mcp(MCPToolChoice {
            server_label: "mcp".into(),
            name: "mcp_tool".into(),
        })),
        ..Default::default()
    };
    let value = verify_serialization(&settings);
    assert_eq!(
        value["tool_choice"],
        json!({"server_label": "mcp", "name": "mcp_tool"})
    );
}

/// Every field of `ModelSettings` is set, and the result still serializes.
fn all_fields() -> ModelSettings {
    ModelSettings {
        temperature: Some(0.5),
        top_p: Some(0.9),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        tool_choice: Some(ToolChoice::auto()),
        parallel_tool_calls: Some(true),
        truncation: Some("auto".into()),
        max_tokens: Some(100),
        reasoning: Some(Reasoning::default()),
        verbosity: Some("low".into()),
        metadata: Some(string_map(&[("foo", "bar")])),
        store: Some(false),
        prompt_cache_retention: Some("24h".into()),
        include_usage: Some(false),
        response_include: Some(vec!["reasoning.encrypted_content".into()]),
        top_logprobs: Some(1),
        extra_query: Some(string_map(&[("foo", "bar")])),
        extra_body: Some(object(json!({"foo": "bar"}))),
        extra_headers: Some(string_map(&[("foo", "bar")])),
        extra_args: Some(object(
            json!({"custom_param": "value", "another_param": 42}),
        )),
        retry: Some(ModelRetrySettings {
            max_retries: 2,
            backoff: ModelRetryBackoffSettings {
                initial_delay: 0.1,
                max_delay: 1.0,
                multiplier: 2.0,
                jitter: false,
            },
            ..Default::default()
        }),
        context_management: Some(vec![
            json!({"type": "compaction", "compact_threshold": 200000}),
        ]),
        prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
        preserve_raw_usage: Some(true),
        timeout: Some(1.25),
    }
}

#[test]
fn all_fields_serialization() {
    let settings = all_fields();
    let value = verify_serialization(&settings);
    // Python asserts no dataclass field was left at `None`; the Rust equivalent is that every
    // optional field made it into the serialized object.
    for field in [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "tool_choice",
        "parallel_tool_calls",
        "truncation",
        "max_tokens",
        "reasoning",
        "verbosity",
        "metadata",
        "store",
        "prompt_cache_retention",
        "include_usage",
        "response_include",
        "top_logprobs",
        "extra_query",
        "extra_body",
        "extra_headers",
        "extra_args",
        "retry",
        "context_management",
        "prompt_cache_options",
        "preserve_raw_usage",
        "timeout",
    ] {
        assert!(value.get(field).is_some(), "{field} must be serialized");
    }
}

#[test]
fn reasoning_and_prompt_cache_serialization() {
    let settings = ModelSettings {
        reasoning: Some(Reasoning {
            mode: Some("pro".into()),
            effort: Some("max".into()),
            context: Some("all_turns".into()),
            summary: None,
            ..Default::default()
        }),
        prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
        ..Default::default()
    };

    let value = serde_json::to_value(&settings).unwrap();
    assert_eq!(value["reasoning"]["context"], json!("all_turns"));
    assert_eq!(value["reasoning"]["effort"], json!("max"));
    assert_eq!(value["reasoning"]["mode"], json!("pro"));
    assert_eq!(
        settings.to_traceable_value()["prompt_cache_options"],
        json!({"mode": "explicit", "ttl": "30m"})
    );
}

/// The last four settings are the ones appended after the public ones, in this order.
#[test]
fn usage_preservation_is_appended_to_the_public_field_order() {
    let value = serde_json::to_value(all_fields()).unwrap();
    let keys: Vec<&String> = value.as_object().unwrap().keys().collect();
    assert_eq!(
        keys[keys.len() - 4..]
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>(),
        vec![
            "context_management",
            "prompt_cache_options",
            "preserve_raw_usage",
            "timeout"
        ]
    );
}

#[test]
fn extra_args_serialization() {
    let settings = ModelSettings {
        temperature: Some(0.5),
        extra_args: Some(object(json!({
            "custom_param": "value", "another_param": 42, "nested": {"key": "value"}
        }))),
        ..Default::default()
    };
    let value = verify_serialization(&settings);
    assert_eq!(
        value["extra_args"],
        json!({"custom_param": "value", "another_param": 42, "nested": {"key": "value"}})
    );
}

// ---------------------------------------------------------------------------------------------
// traceable serialization
// ---------------------------------------------------------------------------------------------

#[test]
fn traceable_serialization_omits_request_extras() {
    let settings = ModelSettings {
        temperature: Some(0.5),
        extra_headers: Some(string_map(&[("Authorization", "Bearer provider-token")])),
        extra_query: Some(string_map(&[("api-key", "query-token")])),
        extra_body: Some(object(json!({"secret": "body-token"}))),
        extra_args: Some(object(json!({"api_key": "arg-token"}))),
        preserve_raw_usage: Some(true),
        ..Default::default()
    };

    let value = serde_json::to_value(&settings).unwrap();
    assert_eq!(
        value["extra_headers"],
        json!({"Authorization": "Bearer provider-token"})
    );
    assert_eq!(value["extra_query"], json!({"api-key": "query-token"}));
    assert_eq!(value["extra_body"], json!({"secret": "body-token"}));
    assert_eq!(value["extra_args"], json!({"api_key": "arg-token"}));

    let traceable = settings.to_traceable_value();
    assert_eq!(traceable["temperature"], json!(0.5));
    for omitted in [
        "extra_headers",
        "extra_query",
        "extra_body",
        "extra_args",
        "preserve_raw_usage",
    ] {
        assert!(
            traceable.get(omitted).is_none(),
            "{omitted} must not be traced"
        );
    }
    // A traceable payload is always JSON serializable.
    assert!(serde_json::to_string(&traceable).is_ok());
}

// ---------------------------------------------------------------------------------------------
// resolve
// ---------------------------------------------------------------------------------------------

#[test]
fn an_override_preserves_omitted_values() {
    let settings = ModelSettings {
        temperature: Some(0.5),
        reasoning: Some(Reasoning {
            context: Some("all_turns".into()),
            ..Default::default()
        }),
        retry: Some(ModelRetrySettings::default().with_max_retries(2)),
        ..Default::default()
    };

    let resolved = settings.resolve(Some(&ModelSettings {
        temperature: Some(0.0),
        ..Default::default()
    }));

    assert_eq!(resolved.temperature, Some(0.0));
    assert_eq!(resolved.reasoning, settings.reasoning);
    assert_eq!(resolved.retry, settings.retry);
}

#[test]
fn extra_args_are_merged_by_resolve() {
    let base = ModelSettings {
        temperature: Some(0.5),
        extra_args: Some(object(
            json!({"param1": "base_value", "param2": "base_only"}),
        )),
        ..Default::default()
    };
    let over = ModelSettings {
        top_p: Some(0.9),
        extra_args: Some(object(
            json!({"param1": "override_value", "param3": "override_only"}),
        )),
        ..Default::default()
    };

    let resolved = base.resolve(Some(&over));

    assert_eq!(resolved.temperature, Some(0.5));
    assert_eq!(resolved.top_p, Some(0.9));
    assert_eq!(
        serde_json::to_value(resolved.extra_args.unwrap()).unwrap(),
        json!({
            "param1": "override_value",
            "param2": "base_only",
            "param3": "override_only",
        })
    );
}

#[test]
fn extra_args_resolve_when_one_side_is_unset() {
    let base = ModelSettings {
        extra_args: Some(object(json!({"param1": "value1"}))),
        ..Default::default()
    };
    let over = ModelSettings {
        temperature: Some(0.8),
        ..Default::default()
    };
    let resolved = base.resolve(Some(&over));
    assert_eq!(
        serde_json::to_value(resolved.extra_args.unwrap()).unwrap(),
        json!({"param1": "value1"})
    );
    assert_eq!(resolved.temperature, Some(0.8));

    let base = ModelSettings {
        temperature: Some(0.5),
        ..Default::default()
    };
    let over = ModelSettings {
        extra_args: Some(object(json!({"param2": "value2"}))),
        ..Default::default()
    };
    let resolved = base.resolve(Some(&over));
    assert_eq!(
        serde_json::to_value(resolved.extra_args.unwrap()).unwrap(),
        json!({"param2": "value2"})
    );
    assert_eq!(resolved.temperature, Some(0.5));
}

#[test]
fn extra_args_resolve_when_both_sides_are_unset() {
    let base = ModelSettings {
        temperature: Some(0.5),
        ..Default::default()
    };
    let over = ModelSettings {
        top_p: Some(0.9),
        ..Default::default()
    };
    let resolved = base.resolve(Some(&over));
    assert_eq!(resolved.extra_args, None);
    assert_eq!(resolved.temperature, Some(0.5));
    assert_eq!(resolved.top_p, Some(0.9));
}

// ---------------------------------------------------------------------------------------------
// round trip
// ---------------------------------------------------------------------------------------------

#[test]
fn settings_survive_a_serde_round_trip() {
    let settings = ModelSettings {
        temperature: Some(0.5),
        top_p: Some(0.9),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        tool_choice: Some(ToolChoice::auto()),
        parallel_tool_calls: Some(true),
        truncation: Some("auto".into()),
        max_tokens: Some(100),
        reasoning: Some(Reasoning::default()),
        metadata: Some(string_map(&[("foo", "bar")])),
        store: Some(false),
        include_usage: Some(false),
        top_logprobs: Some(1),
        extra_query: Some(string_map(&[("foo", "bar")])),
        extra_body: Some(object(json!({"foo": "bar"}))),
        extra_headers: Some(string_map(&[("foo", "bar")])),
        extra_args: Some(object(
            json!({"custom_param": "value", "another_param": 42}),
        )),
        ..Default::default()
    };

    let text = serde_json::to_string(&settings).unwrap();
    let back: ModelSettings = serde_json::from_str(&text).unwrap();
    assert_eq!(settings, back);
}

// ---------------------------------------------------------------------------------------------
// retry settings
// ---------------------------------------------------------------------------------------------

#[test]
fn the_retry_policy_is_excluded_from_serialization() {
    let settings = ModelSettings {
        retry: Some(
            ModelRetrySettings {
                max_retries: 1,
                backoff: ModelRetryBackoffSettings {
                    initial_delay: 0.1,
                    ..Default::default()
                },
                ..Default::default()
            }
            .with_policy(openai_agents_sdk::retry::retry_policies::http_status(vec![
                429,
            ])),
        ),
        ..Default::default()
    };

    let value = verify_serialization(&settings);
    let retry = &value["retry"];
    assert_eq!(retry["max_retries"], json!(1));
    assert_eq!(retry["backoff"]["initial_delay"], json!(0.1));
    // The policy is a runtime closure and never reaches the wire.
    assert!(retry.get("policy").is_none());
}

#[test]
fn the_retry_policy_is_dropped_by_a_round_trip() {
    let settings = ModelSettings {
        retry: Some(
            ModelRetrySettings {
                max_retries: 2,
                backoff: ModelRetryBackoffSettings {
                    initial_delay: 0.5,
                    ..Default::default()
                },
                ..Default::default()
            }
            .with_policy(openai_agents_sdk::retry::retry_policies::http_status(vec![
                429,
            ])),
        ),
        ..Default::default()
    };

    let text = serde_json::to_string(&settings).unwrap();
    let back: ModelSettings = serde_json::from_str(&text).unwrap();
    let retry = back.retry.unwrap();
    assert_eq!(retry.max_retries, 2);
    assert_eq!(retry.backoff.initial_delay, 0.5);
    assert!(retry.policy.is_none());
}

#[test]
fn nested_retry_input_is_deserialized_into_settings() {
    let settings: ModelSettings = serde_json::from_value(json!({
        "retry": {
            "max_retries": 3,
            "backoff": {
                "initial_delay": 0.25,
                "max_delay": 2.0,
                "multiplier": 3.0,
                "jitter": false,
            },
        }
    }))
    .unwrap();

    let retry = settings.retry.unwrap();
    assert_eq!(retry.max_retries, 3);
    assert_eq!(
        retry.backoff,
        ModelRetryBackoffSettings {
            initial_delay: 0.25,
            max_delay: 2.0,
            multiplier: 3.0,
            jitter: false,
        }
    );
}

/// A falsey backoff value survives deserialization and serialization.
#[test]
fn falsey_retry_backoff_values_survive_a_round_trip() {
    let retry: ModelRetrySettings = serde_json::from_value(json!({
        "max_retries": 1,
        "backoff": {"initial_delay": 0.5, "max_delay": 5.0, "multiplier": 2.0, "jitter": false},
    }))
    .unwrap();

    assert!(!retry.backoff.jitter);
    let value = serde_json::to_value(&retry).unwrap();
    assert_eq!(value["backoff"]["jitter"], json!(false));
}

/// An omitted `backoff` falls back to the documented defaults rather than failing.
#[test]
fn an_omitted_backoff_uses_the_defaults() {
    let retry: ModelRetrySettings = serde_json::from_value(json!({"max_retries": 1})).unwrap();
    assert_eq!(retry.backoff, ModelRetryBackoffSettings::default());
    assert_eq!(retry.max_retries, 1);
}
