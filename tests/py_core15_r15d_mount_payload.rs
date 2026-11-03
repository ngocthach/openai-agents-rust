//! Ports of the manifest-payload discriminator cases of `tests/sandbox/test_mount_security.py`:
//! an unknown `type` in a mount strategy, pattern or entry is refused without repeating the
//! value, and a credential file smuggled in under an unknown entry type is refused too.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::{
    BuiltinMountPattern, ConfiguredMountStrategy, Entry, Manifest, RcloneMountPattern, S3Mount,
};
use serde_json::{Value, json};

fn mount_manifest() -> Value {
    let mount = ConfiguredMountStrategy::in_container(
        S3Mount::new("example-bucket"),
        BuiltinMountPattern::Rclone(RcloneMountPattern::new()),
    )
    .mount();
    Manifest::new()
        .with_entry("data", Entry::mount(mount))
        .to_value()
        .expect("write manifest")
}

/// Python: `test_deserialization_rejects_unknown_string_discriminators_without_values` and
/// `test_raw_state_sanitization_rejects_unknown_pattern_discriminator`.
#[test]
fn unknown_strategy_and_pattern_types_are_refused_without_the_value() {
    for location in ["strategy", "pattern"] {
        let sentinel = format!("unknown-{location}-secret");
        let mut payload = mount_manifest();
        let strategy = &mut payload["entries"]["data"]["mount_strategy"];
        if location == "strategy" {
            strategy["type"] = json!(sentinel);
        } else {
            strategy["pattern"]["type"] = json!(sentinel);
        }
        for result in [
            Manifest::from_value(&payload),
            Manifest::from_untrusted_value(&payload, "manifest"),
        ] {
            let error = result.expect_err("an unknown discriminator is refused");
            assert!(!error.to_string().contains(&sentinel), "{error}");
        }
    }
}

/// Python: `test_deserialization_rejects_malformed_entry_container_without_values`.
#[test]
fn a_malformed_entry_container_is_refused_without_the_value() {
    let sentinel = "malformed-entry-container-secret";
    let mut payload = mount_manifest();
    payload["entries"] = json!(sentinel);
    let error = Manifest::from_value(&payload).expect_err("refused");
    assert!(!error.to_string().contains(sentinel), "{error}");
}

/// Python: `test_raw_state_sanitization_rejects_credential_content_without_file_discriminator`.
#[test]
fn credential_content_under_an_unknown_entry_type_is_refused_without_the_value() {
    let sentinel = "credential-file-secret";
    let mut payload = mount_manifest();
    payload["entries"]["credentials.json"] = json!({"type": "unknown_file", "content": sentinel});
    let error = Manifest::from_value(&payload).expect_err("refused");
    assert!(!error.to_string().contains(sentinel), "{error}");
}
