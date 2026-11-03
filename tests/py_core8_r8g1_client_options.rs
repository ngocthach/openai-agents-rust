//! Ports of `tests/sandbox/test_client_options.py`.
//!
//! Python's `BaseSandboxClientOptions.parse` is a registry of pydantic classes; Rust reads the
//! same `type`-discriminated payload into `SandboxClientOptionsRecord`. The three rows about
//! pydantic itself (an existing instance passed through `parse`, `model_dump(exclude_unset=True)`,
//! and registering a second class under an existing `type` or none at all) have no Rust form.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::SandboxClientOptionsRecord;
use openai_agents_sdk::extensions::sandbox::cloudflare::CloudflareSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::daytona::DaytonaSandboxClientOptions;
use openai_agents_sdk::extensions::sandbox::e2b::E2BSandboxClientOptions;
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DEFAULT_PYTHON_SANDBOX_IMAGE, DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClientOptions;
use serde_json::json;

/// Python: `test_sandbox_client_options_parse_uses_registered_builtin_type`: a payload that
/// names only the fields it sets reads as the backend its `type` selects, with the rest default.
#[test]
fn parse_uses_the_registered_builtin_type() {
    let parsed: SandboxClientOptionsRecord = serde_json::from_value(json!({
        "type": "docker",
        "image": DEFAULT_PYTHON_SANDBOX_IMAGE,
        "exposed_ports": [8080],
    }))
    .unwrap();

    assert_eq!(
        parsed,
        SandboxClientOptionsRecord::Docker(
            DockerSandboxClientOptions::new(DEFAULT_PYTHON_SANDBOX_IMAGE).with_exposed_port(8080)
        )
    );
}

/// Python: `test_docker_client_options_roundtrip_preserves_labels`.
#[test]
fn docker_client_options_roundtrip_preserves_labels() {
    let options = DockerSandboxClientOptions::new(DEFAULT_PYTHON_SANDBOX_IMAGE)
        .with_label("com.example.owner", "worker-123");
    let record = SandboxClientOptionsRecord::Docker(options.clone());

    let payload = serde_json::to_value(&record).unwrap();
    let restored: SandboxClientOptionsRecord = serde_json::from_value(payload).unwrap();

    assert_eq!(restored, record);
    let SandboxClientOptionsRecord::Docker(restored) = restored else {
        panic!("the payload named the docker backend");
    };
    assert_eq!(
        restored.labels.get("com.example.owner").map(String::as_str),
        Some("worker-123")
    );
}

/// Python: `test_sandbox_client_options_roundtrip_preserves_concrete_type` (one row per option).
#[test]
fn roundtrip_preserves_the_concrete_type() {
    let daytona = DaytonaSandboxClientOptions {
        image: Some(DEFAULT_PYTHON_SANDBOX_IMAGE.to_string()),
        ..DaytonaSandboxClientOptions::default()
    };
    let options = [
        SandboxClientOptionsRecord::Docker(
            DockerSandboxClientOptions::new(DEFAULT_PYTHON_SANDBOX_IMAGE).with_exposed_port(8080),
        ),
        SandboxClientOptionsRecord::UnixLocal(UnixLocalSandboxClientOptions {
            exposed_ports: vec![8080],
        }),
        SandboxClientOptionsRecord::E2b(E2BSandboxClientOptions::new().with_template("base")),
        SandboxClientOptionsRecord::Daytona(daytona),
        SandboxClientOptionsRecord::Cloudflare(CloudflareSandboxClientOptions::new(
            "https://example.com",
        )),
    ];

    for record in options {
        let payload = serde_json::to_value(&record).unwrap();
        assert_eq!(payload["type"], record.type_name());

        let restored: SandboxClientOptionsRecord = serde_json::from_value(payload).unwrap();

        assert_eq!(restored, record, "{}", record.type_name());
        assert_eq!(restored.type_name(), record.type_name());
    }
}

/// Python: `test_sandbox_client_options_parse_rejects_unknown_type`: the rejection names the
/// type it did not recognise.
#[test]
fn parse_rejects_an_unknown_type_and_names_it() {
    let error = serde_json::from_value::<SandboxClientOptionsRecord>(json!({"type": "unknown"}))
        .unwrap_err();

    assert!(error.to_string().contains("`unknown`"), "{error}");
}

/// Python: `test_sandbox_client_options_parse_rejects_invalid_payload`: a payload that is not
/// an object is refused.
#[test]
fn parse_rejects_a_payload_that_is_not_an_object() {
    for payload in [json!("docker"), json!(7), json!(null)] {
        assert!(
            serde_json::from_value::<SandboxClientOptionsRecord>(payload.clone()).is_err(),
            "{payload}"
        );
    }
}

/// A payload that names only `type` and the fields it must set reads as the default options for
/// that backend, so a payload written by hand does not have to spell out every setting.
#[test]
fn a_payload_naming_only_required_fields_reads_the_defaults() {
    use openai_agents_sdk::extensions::sandbox::blaxel::BlaxelSandboxClientOptions;
    use openai_agents_sdk::extensions::sandbox::modal::ModalSandboxClientOptions;
    use openai_agents_sdk::extensions::sandbox::vercel::VercelSandboxClientOptions;

    let cases = [
        (
            json!({"type": "unix_local"}),
            SandboxClientOptionsRecord::UnixLocal(UnixLocalSandboxClientOptions::default()),
        ),
        (
            json!({"type": "docker"}),
            SandboxClientOptionsRecord::Docker(DockerSandboxClientOptions::default()),
        ),
        (
            json!({"type": "e2b"}),
            SandboxClientOptionsRecord::E2b(E2BSandboxClientOptions::new()),
        ),
        (
            json!({"type": "modal", "app_name": "sandbox-tests"}),
            SandboxClientOptionsRecord::Modal(ModalSandboxClientOptions::new("sandbox-tests")),
        ),
        (
            json!({"type": "cloudflare", "worker_url": "https://example.com"}),
            SandboxClientOptionsRecord::Cloudflare(CloudflareSandboxClientOptions::new(
                "https://example.com",
            )),
        ),
        (
            json!({"type": "daytona"}),
            SandboxClientOptionsRecord::Daytona(DaytonaSandboxClientOptions::default()),
        ),
        (
            json!({"type": "vercel"}),
            SandboxClientOptionsRecord::Vercel(VercelSandboxClientOptions::default()),
        ),
        (
            json!({"type": "blaxel"}),
            SandboxClientOptionsRecord::Blaxel(BlaxelSandboxClientOptions::default()),
        ),
    ];

    for (payload, expected) in cases {
        let parsed: SandboxClientOptionsRecord = serde_json::from_value(payload.clone())
            .unwrap_or_else(|error| panic!("{payload}: {error}"));
        assert_eq!(parsed, expected, "{payload}");
    }
}
