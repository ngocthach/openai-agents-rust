//! Ports of the custom environment value type tests of `tests/sandbox/test_manifest.py`:
//! a registered type persists as its tagged payload without the resolved secret, restores to the
//! same type, and registering a type name twice is refused.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::environment::{EnvEntry, EnvValue};
use openai_agents_sdk::sandbox::errors::SandboxResult;
use openai_agents_sdk::sandbox::persisted_registry::register_env_value_type;
use openai_agents_sdk::sandbox::{ErrorCode, Manifest};
use serde_json::{Value, json};

#[derive(Debug)]
struct SecretReferenceEnvValue {
    key: String,
}

#[async_trait]
impl EnvValue for SecretReferenceEnvValue {
    fn env_value_type(&self) -> &str {
        "test.secret_reference"
    }

    fn to_payload(&self) -> Option<Value> {
        Some(json!({"type": "test.secret_reference", "key": self.key}))
    }

    async fn resolve(&self) -> SandboxResult<String> {
        Ok(format!("resolved-secret-for-{}", self.key))
    }
}

/// The serializer writes only the key; the internal value never leaves the process.
#[derive(Debug)]
struct CustomSerializedEnvValue {
    key: String,
    internal_value: String,
}

#[async_trait]
impl EnvValue for CustomSerializedEnvValue {
    fn env_value_type(&self) -> &str {
        "test.custom_serializer"
    }

    fn to_payload(&self) -> Option<Value> {
        Some(json!({"type": "test.custom_serializer", "key": self.key}))
    }

    async fn resolve(&self) -> SandboxResult<String> {
        Ok(self.internal_value.clone())
    }
}

fn register_all() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        register_env_value_type(
            "test.secret_reference",
            Arc::new(|payload| {
                let key = payload["key"].as_str().unwrap_or_default().to_string();
                Ok(Arc::new(SecretReferenceEnvValue { key }) as Arc<dyn EnvValue>)
            }),
        )
        .unwrap();
        register_env_value_type(
            "test.custom_serializer",
            Arc::new(|payload| {
                let key = payload["key"].as_str().unwrap_or_default().to_string();
                Ok(Arc::new(CustomSerializedEnvValue {
                    key,
                    internal_value: String::new(),
                }) as Arc<dyn EnvValue>)
            }),
        )
        .unwrap();
    });
}

/// Python: `test_manifest_round_trips_tagged_env_values_without_resolved_secrets`.
#[tokio::test]
async fn tagged_env_values_round_trip_without_resolved_secrets() {
    register_all();
    let manifest = Manifest::new()
        .with_env_entry(
            "DIRECT",
            EnvEntry::new(Arc::new(SecretReferenceEnvValue {
                key: "direct".into(),
            })),
        )
        .with_env_entry(
            "ENTRY",
            EnvEntry::new(Arc::new(SecretReferenceEnvValue {
                key: "entry".into(),
            }))
            .with_description("secret reference")
            .ephemeral(true),
        );

    let payload = manifest.to_value().unwrap();
    assert_eq!(
        payload["environment"],
        json!({
            "value": {
                "DIRECT": {
                    "description": null,
                    "ephemeral": false,
                    "value": {"type": "test.secret_reference", "key": "direct"},
                },
                "ENTRY": {
                    "description": "secret reference",
                    "ephemeral": true,
                    "value": {"type": "test.secret_reference", "key": "entry"},
                },
            }
        })
    );
    assert!(!payload.to_string().contains("resolved-secret"));

    let restored = Manifest::from_value(&payload).unwrap();
    let entries = restored.resolved_environment().entries();
    assert_eq!(
        entries["DIRECT"].value.env_value_type(),
        "test.secret_reference"
    );
    assert_eq!(
        entries["ENTRY"].value.env_value_type(),
        "test.secret_reference"
    );
    assert!(entries["ENTRY"].ephemeral);
    let resolved = restored.resolve_environment().await.unwrap();
    assert_eq!(resolved["DIRECT"], "resolved-secret-for-direct");
    assert_eq!(resolved["ENTRY"], "resolved-secret-for-entry");
    assert_eq!(restored.to_value().unwrap(), payload);
}

/// Python: `test_manifest_preserves_type_from_env_value_custom_serializer`.
#[tokio::test]
async fn custom_serializer_keeps_the_type_and_hides_internal_values() {
    register_all();
    let manifest = Manifest::new().with_env_entry(
        "DIRECT",
        EnvEntry::new(Arc::new(CustomSerializedEnvValue {
            key: "direct".into(),
            internal_value: "direct-secret".into(),
        })),
    );

    let payload = manifest.to_value().unwrap();

    assert_eq!(
        payload["environment"]["value"]["DIRECT"]["value"],
        json!({"type": "test.custom_serializer", "key": "direct"})
    );
    assert!(!payload.to_string().contains("direct-secret"));
    let restored = Manifest::from_value(&payload).unwrap();
    let entries = restored.resolved_environment().entries();
    assert_eq!(
        entries["DIRECT"].value.env_value_type(),
        "test.custom_serializer"
    );
}

/// Python: `test_duplicate_env_value_type_registration_raises`.
#[test]
fn registering_a_type_twice_is_refused() {
    register_all();
    let error = register_env_value_type(
        "test.secret_reference",
        Arc::new(
            |_| Ok(Arc::new(SecretReferenceEnvValue { key: "x".into() }) as Arc<dyn EnvValue>),
        ),
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(error.to_string().contains("already registered"), "{error}");
    // The built-in literal kind cannot be replaced either.
    assert!(register_env_value_type("str", Arc::new(|_| unreachable!())).is_err());
}

/// A value whose payload names another type than it declares is refused instead of written.
#[test]
fn a_payload_that_disagrees_with_the_declared_type_is_unwritable() {
    register_all();

    #[derive(Debug)]
    struct Liar;
    #[async_trait]
    impl EnvValue for Liar {
        fn env_value_type(&self) -> &str {
            "test.secret_reference"
        }
        fn to_payload(&self) -> Option<Value> {
            Some(json!({"type": "something.else"}))
        }
        async fn resolve(&self) -> SandboxResult<String> {
            Ok(String::new())
        }
    }
    let manifest = Manifest::new().with_env_entry("X", EnvEntry::new(Arc::new(Liar)));
    assert!(manifest.to_value().is_err());
}
