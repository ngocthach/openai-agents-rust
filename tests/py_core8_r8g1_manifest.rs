//! Ports of the Python sandbox suite's manifest payload tests: the serialization half of
//! `tests/sandbox/test_manifest.py` and the dictionary half of
//! `tests/sandbox/test_manifest_config.py`.
//!
//! Python's `Manifest` is a pydantic model; Rust reads and writes the same payload shape with
//! `Manifest::from_value`, `Manifest::from_untrusted_value` and `Manifest::to_value`. Two things
//! differ and are asserted as the Rust shape: a literal variable added with `with_env` is a
//! plain string and a start-up lookup is always written in the long `{description, ephemeral,
//! value}` form, where pydantic keeps whichever of `str`, bare `EnvValue` or `EnvEntry` the
//! caller built; and only the literal `str` env value kind is registered, so a custom `EnvValue`
//! can be resolved but is refused when written.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::environment::{EnvEntry, EnvValue};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::{Entry, EntryKind, Manifest};
use serde_json::{Value, json};

/// A start-up lookup that never declared a serializable type, as Python's `_UntaggedEnvValue`.
#[derive(Debug)]
struct UntaggedEnvValue {
    key: String,
}

#[async_trait]
impl EnvValue for UntaggedEnvValue {
    fn env_value_type(&self) -> &str {
        ""
    }

    async fn resolve(&self) -> SandboxResult<String> {
        Ok(format!("resolved-secret-for-{}", self.key))
    }
}

/// A subclass of the literal kind that inherits its `str` tag, as Python's
/// `_LabeledStrEnvValue(StrEnvValue)`.
#[derive(Debug)]
struct LabeledStrEnvValue {
    value: String,
}

#[async_trait]
impl EnvValue for LabeledStrEnvValue {
    fn env_value_type(&self) -> &str {
        "str"
    }

    async fn resolve(&self) -> SandboxResult<String> {
        Ok(self.value.clone())
    }
}

fn config_error(
    result: SandboxResult<Manifest>,
) -> openai_agents_sdk::sandbox::errors::SandboxError {
    match result {
        Ok(_) => panic!("the payload should have been rejected"),
        Err(error) => error,
    }
}

// ---------------------------------------------------------------------------
// tests/sandbox/test_manifest.py: environment payloads
// ---------------------------------------------------------------------------

/// Python: `test_manifest_round_trips_str_env_value`.
#[tokio::test]
async fn manifest_round_trips_str_env_value() {
    let manifest = Manifest::new()
        .with_env("PLAIN", "plain")
        .with_env_entry("TYPED", EnvEntry::literal("typed"));

    let payload = manifest.to_value().unwrap();
    assert_eq!(
        payload["environment"],
        json!({
            "value": {
                "PLAIN": "plain",
                "TYPED": {
                    "description": null,
                    "ephemeral": false,
                    "value": {"type": "str", "value": "typed"},
                },
            }
        })
    );

    let restored = Manifest::from_value(&payload).unwrap();
    assert_eq!(
        restored.environment().get("PLAIN").map(String::as_str),
        Some("plain")
    );
    let typed = &restored.resolved_environment().entries()["TYPED"];
    assert_eq!(typed.value.env_value_type(), "str");
    assert_eq!(typed.value.resolve().await.unwrap(), "typed");
    assert_eq!(restored.to_value().unwrap(), payload);
}

/// Python: `test_manifest_reads_legacy_discriminator_free_str_env_values`.
#[tokio::test]
async fn manifest_reads_legacy_discriminator_free_str_env_values() {
    let payload = json!({
        "environment": {
            "value": {
                "DIRECT": {"value": "direct-value"},
                "ENTRY": {
                    "description": "typed entry",
                    "ephemeral": true,
                    "value": {"value": "entry-value"},
                },
            }
        }
    });

    let restored = Manifest::from_value(&payload).unwrap();

    let entries = restored.resolved_environment().entries();
    let direct = &entries["DIRECT"];
    assert_eq!(direct.value.env_value_type(), "str");
    assert_eq!(direct.value.resolve().await.unwrap(), "direct-value");
    assert_eq!(direct.description, None);
    assert!(!direct.ephemeral);
    let entry = &entries["ENTRY"];
    assert_eq!(entry.value.env_value_type(), "str");
    assert_eq!(entry.value.resolve().await.unwrap(), "entry-value");
    assert_eq!(entry.description.as_deref(), Some("typed entry"));
    assert!(entry.ephemeral);
    assert_eq!(
        restored.resolve_environment().await.unwrap()["ENTRY"],
        "entry-value"
    );
}

/// Python: `test_manifest_rejects_ambiguous_discriminator_free_env_values`.
#[test]
fn manifest_rejects_ambiguous_discriminator_free_env_values() {
    let payload = json!({
        "environment": {
            "value": {
                "AMBIGUOUS": {"value": "plain", "description": "not a legacy StrEnvValue"},
            }
        }
    });

    let error = config_error(Manifest::from_value(&payload));

    assert!(
        error
            .to_string()
            .contains("must include a string `type` field"),
        "{error}"
    );
}

/// Python: `test_manifest_rejects_unknown_env_value_type`.
#[test]
fn manifest_rejects_unknown_env_value_type() {
    let payload = json!({"environment": {"value": {"TOKEN": {"type": "unknown.env.value"}}}});

    let error = config_error(Manifest::from_value(&payload));

    assert!(
        error
            .to_string()
            .contains("Unknown env value type `unknown.env.value`"),
        "{error}"
    );
}

/// Python: `test_untagged_env_value_imports_and_resolves_but_does_not_serialize`.
#[tokio::test]
async fn untagged_env_value_resolves_but_does_not_serialize() {
    let value = UntaggedEnvValue {
        key: "token".to_string(),
    };

    assert_eq!(value.resolve().await.unwrap(), "resolved-secret-for-token");
    let manifest = Manifest::new().with_env_entry("TOKEN", EnvEntry::new(Arc::new(value)));
    assert_eq!(
        manifest.resolve_environment().await.unwrap()["TOKEN"],
        "resolved-secret-for-token"
    );
    let error = manifest.to_value().unwrap_err();
    assert!(error.to_string().contains("environment value"), "{error}");
    assert!(!error.to_string().contains("resolved-secret"));
}

/// Python: `test_inherited_env_value_tag_imports_and_resolves_but_does_not_serialize`.
#[tokio::test]
async fn inherited_env_value_tag_resolves_but_does_not_serialize() {
    let value = LabeledStrEnvValue {
        value: "plain".to_string(),
    };

    assert_eq!(value.resolve().await.unwrap(), "plain");
    let manifest = Manifest::new().with_env_entry("VALUE", EnvEntry::new(Arc::new(value)));
    let error = manifest.to_value().unwrap_err();
    assert!(error.to_string().contains("environment value"), "{error}");
}

// ---------------------------------------------------------------------------
// the rest of the manifest payload
// ---------------------------------------------------------------------------

#[test]
fn manifest_payload_round_trips_every_field_and_keeps_entry_order() {
    let manifest = Manifest::new()
        .with_root("/workspace/project")
        .with_entry("zeta.txt", Entry::file(b"z".to_vec()))
        .with_entry(
            "alpha",
            Entry::dir().with_child("inner.txt", Entry::file(b"inner".to_vec())),
        )
        .with_user("alice")
        .with_group(openai_agents_sdk::sandbox::types::Group::new("staff"))
        .with_remote_mount_command_allowlist(["ls", "cat"]);

    let payload = manifest.to_value().unwrap();

    assert_eq!(payload["version"], 1);
    assert_eq!(payload["root"], "/workspace/project");
    assert_eq!(payload["users"], json!([{"name": "alice"}]));
    assert_eq!(
        payload["remote_mount_command_allowlist"],
        json!(["ls", "cat"])
    );
    let written: Vec<&str> = payload["entries"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(written, ["zeta.txt", "alpha"]);

    let restored = Manifest::from_value(&payload).unwrap();
    assert_eq!(restored.root(), "/workspace/project");
    assert_eq!(restored.users().len(), 1);
    assert_eq!(restored.groups()[0].name, "staff");
    assert_eq!(restored.remote_mount_command_allowlist(), ["ls", "cat"]);
    let order: Vec<&str> = restored
        .entries_in_order()
        .map(|(path, _)| path.as_str())
        .collect();
    assert_eq!(order, ["zeta.txt", "alpha"]);
    assert_eq!(restored.to_value().unwrap(), payload);
}

#[test]
fn manifest_payload_never_carries_the_credential_acknowledgements() {
    let manifest = Manifest::new()
        .with_in_container_mount_credential_exposure_acknowledged(["/workspace/data"]);

    let payload = manifest.to_value().unwrap();

    let text = payload.to_string();
    assert!(!text.contains("acknowledged"), "{text}");
    assert!(!text.contains("/workspace/data"), "{text}");
    for key in [
        "in_container_mount_credential_exposure_acknowledged_paths",
        "_mount_credential_exposure_policy",
        "inContainerMountBroadCredentialExposureAcknowledgedPaths",
    ] {
        let error = config_error(Manifest::from_value(&json!({ key: ["/workspace/data"] })));
        assert!(
            error
                .to_string()
                .contains("must be configured on a trusted Manifest instance"),
            "{key}: {error}"
        );
    }
}

#[test]
fn manifest_payload_rejects_a_version_other_than_one() {
    let error = config_error(Manifest::from_value(&json!({"version": 2})));
    assert!(error.to_string().contains("version"), "{error}");
}

#[test]
fn manifest_payload_round_trips_path_grants() {
    let manifest = Manifest::new().with_extra_path_grant(
        openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant::new("/opt/data")
            .unwrap()
            .read_only(true)
            .with_description("reference data"),
    );

    let payload = manifest.to_value().unwrap();

    assert_eq!(
        payload["extra_path_grants"],
        json!([{"path": "/opt/data", "read_only": true, "description": "reference data"}])
    );
    let restored = Manifest::from_value(&payload).unwrap();
    assert_eq!(restored.extra_path_grants(), manifest.extra_path_grants());
}

// ---------------------------------------------------------------------------
// tests/sandbox/test_manifest_config.py
// ---------------------------------------------------------------------------

/// Python: `test_dictionary_manifest_rejects_host_sources` (`entry_type` x `nested`), through
/// the one Rust configuration boundary.
#[test]
fn dictionary_manifest_rejects_host_sources() {
    for entry_type in ["local_file", "local_dir"] {
        for nested in [false, true] {
            let mut entry = json!({"type": entry_type, "src": ".env"});
            if nested {
                entry = json!({"type": "dir", "children": {"copied": entry}});
            }

            let error = config_error(Manifest::from_untrusted_value(
                &json!({"entries": {"output": entry}}),
                "sandbox.manifest",
            ));

            assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
            assert!(
                error
                    .to_string()
                    .contains("must be configured on a trusted Manifest"),
                "{entry_type} nested={nested}: {error}"
            );
            assert!(error.to_string().contains("sandbox.manifest"), "{error}");
        }
    }
}

#[test]
fn dictionary_manifest_rejects_non_empty_path_grants_but_allows_an_empty_list() {
    let error = config_error(Manifest::from_untrusted_value(
        &json!({"extra_path_grants": [{"path": "/opt/data"}]}),
        "sandbox.manifest",
    ));
    assert!(
        error
            .to_string()
            .contains("sandbox.manifest.extra_path_grants must be configured on a trusted"),
        "{error}"
    );

    let manifest =
        Manifest::from_untrusted_value(&json!({"extra_path_grants": []}), "sandbox.manifest")
            .unwrap();
    assert!(manifest.extra_path_grants().is_empty());
}

/// Python: `test_configuration_preserves_trusted_manifest_host_sources`. A trusted manifest
/// value keeps its host sources, and its own payload round-trips through the trusted reader.
#[test]
fn trusted_manifest_keeps_host_sources_through_the_trusted_reader() {
    let manifest = Manifest::new()
        .with_entry("file", Entry::local_file("input.txt"))
        .with_entry(
            "nested",
            Entry::dir().with_child("directory", Entry::local_dir("data")),
        );

    let restored = Manifest::from_value(&manifest.to_value().unwrap()).unwrap();

    let EntryKind::LocalFile { src } = &restored.entries()["file"].kind else {
        panic!("expected a local file");
    };
    assert_eq!(src, std::path::Path::new("input.txt"));
    let EntryKind::Dir { children } = &restored.entries()["nested"].kind else {
        panic!("expected a directory");
    };
    assert!(matches!(
        &children["directory"].kind,
        EntryKind::LocalDir { src: Some(_) }
    ));
}

/// Python: `test_dictionary_manifest_preserves_entries_without_host_sources`.
#[test]
fn dictionary_manifest_preserves_entries_without_host_sources() {
    let payload: Value = json!({
        "entries": {
            "nested": {
                "type": "dir",
                "children": {"hello.txt": {"type": "file", "content": "hello"}},
            },
            "empty": {"type": "local_dir"},
        }
    });

    let manifest = Manifest::from_untrusted_value(&payload, "sandbox.manifest").unwrap();

    let EntryKind::Dir { children } = &manifest.entries()["nested"].kind else {
        panic!("expected a directory");
    };
    let EntryKind::File { content } = &children["hello.txt"].kind else {
        panic!("expected a file");
    };
    assert_eq!(content, b"hello");
    assert!(matches!(
        &manifest.entries()["empty"].kind,
        EntryKind::LocalDir { src: None }
    ));
}
