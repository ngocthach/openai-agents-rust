//! Port of the sandbox-payload, nested-history, hosted-MCP, item-metadata and programmatic
//! resume cases of the Python SDK's `tests/test_run_state.py`.
//!
//! Where Rust deliberately differs from Python, the test name says so
//! (`rust_refuses_...`); see the parity report for the reasons.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::items::{FunctionCall, InputItem, RunItem, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_state() -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "TestAgent".into(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(5),
        context: Some(json!({})),
        ..Default::default()
    }
}

fn err_text<T>(result: openai_agents_sdk::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOG_LOCK: Mutex<()> = Mutex::new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.level() <= log::Level::Warn {
            RECORDS.lock().unwrap().push(format!("{}", record.args()));
        }
    }
    fn flush(&self) {}
}

/// Serialize the log-capturing tests and clear what earlier ones recorded.
fn capture_logs() -> std::sync::MutexGuard<'static, ()> {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Warn);
    });
    let guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    RECORDS.lock().unwrap().clear();
    guard
}

fn logged() -> String {
    RECORDS.lock().unwrap().join("\n")
}

// ---------------------------------------------------------------------------
// Sandbox resume payload
// ---------------------------------------------------------------------------

fn sandbox_error(sandbox: &Value, from_json: bool) -> String {
    if from_json {
        let mut wire = make_state().to_json().unwrap();
        wire["sandbox"] = sandbox.clone();
        err_text(RunState::from_json(wire))
    } else {
        let mut state = make_state();
        state.sandbox = Some(sandbox.clone());
        err_text(state.to_json())
    }
}

#[test]
fn run_state_round_trip_preserves_serialized_sandbox_session_snapshot_fields() {
    let session_state = json!({
        "type": "unix_local",
        "manifest": {"version": 1, "root": "/workspace", "entries": {}, "environment": {"value": {}}},
        "snapshot": {"type": "local", "id": "local-snapshot", "base_path": "/tmp/snapshots"},
    });
    let sandbox = json!({
        "backend_id": "unix_local",
        "current_agent_key": "TestAgent",
        "current_agent_name": "TestAgent",
        "session_state": session_state,
        "sessions_by_agent": {
            "TestAgent": {"agent_name": "TestAgent", "session_state": session_state},
        },
    });
    let mut state = make_state();
    state.sandbox = Some(sandbox.clone());
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    // Nothing in this payload is mount authority, so it round-trips exactly.
    assert_eq!(restored.sandbox, Some(sandbox));
    assert_eq!(
        restored.sandbox.unwrap()["session_state"]["snapshot"],
        json!({"type": "local", "id": "local-snapshot", "base_path": "/tmp/snapshots"})
    );
}

#[test]
fn run_state_round_trip_preserves_vercel_execution_identity() {
    for sandbox_name in [None, Some("agents-synthetic-name")] {
        let mut payload = json!({
            "type": "vercel",
            "sandbox_id": "synthetic-execution-id",
            "manifest": {"version": 1, "root": "/workspace", "entries": {}, "environment": {"value": {}}},
            "snapshot": {"type": "noop", "id": "synthetic"},
        });
        if let Some(name) = sandbox_name {
            payload["sandbox_name"] = json!(name);
        }
        let mut state = make_state();
        state.sandbox = Some(json!({
            "backend_id": "vercel",
            "current_agent_key": "TestAgent",
            "current_agent_name": "TestAgent",
            "session_state": payload,
            "sessions_by_agent": {"TestAgent": {"agent_name": "TestAgent", "session_state": payload}},
        }));
        let mut serialized = state.to_json().unwrap();
        if sandbox_name.is_none() {
            // Released saved state predates `sandbox_name`.
            serialized["$schemaVersion"] = json!("1.17");
            serialized["context"]["function_tool_approvals"] = json!([]);
        }
        let restored = RunState::from_json(serialized).unwrap();
        let session = &restored.sandbox.unwrap()["session_state"];
        assert_eq!(session["sandbox_id"], json!("synthetic-execution-id"));
        assert_eq!(
            session.get("sandbox_name").and_then(Value::as_str),
            sandbox_name
        );
    }
}

fn raw_credential_session_state() -> Value {
    json!({
        "type": "unimported-provider",
        "manifest": {
            "version": 1,
            "root": "/workspace",
            "entries": {
                "malformed-parent": {
                    "type": "unknown-parent",
                    "children": {
                        "data": {
                            "type": "s3_mount",
                            "access_key_id": "raw-access-key",
                            "secret_access_key": "raw-secret-key",
                            "mount_strategy": {
                                "type": {"invalid": "raw-strategy-discriminator-secret"},
                                "driver": "rclone",
                                "driver_options": {
                                    "vfs-cache-mode": "off",
                                    "s3-secret-access-key": "raw-driver-secret",
                                },
                                "pattern": {
                                    "type": {"invalid": "pattern-discriminator"},
                                    "config_file_path": "/workspace/raw-pattern-secret",
                                    "extra_args": ["--header", "Authorization: raw-header-secret"],
                                    "options": {
                                        "endpoint_url": {"credential": "raw-endpoint-secret"},
                                        "extra_options": {"password": "raw-option-secret"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "environment": {"value": {}},
        },
    })
}

#[test]
fn run_state_sanitizes_raw_mount_credentials_without_provider_imports() {
    let raw = raw_credential_session_state();
    let sandbox = json!({
        "backend_id": "unimported-provider",
        "session_state": raw,
        "sessions_by_agent": {"TestAgent": {"agent_name": "TestAgent", "session_state": raw}},
    });
    let mut state = make_state();
    state.sandbox = Some(sandbox);
    let mut serialized = state.to_json().unwrap();
    let text = serialized.to_string();
    for secret in [
        "raw-access-key",
        "raw-secret-key",
        "raw-driver-secret",
        "raw-pattern-secret",
        "raw-header-secret",
        "raw-endpoint-secret",
        "raw-option-secret",
        "raw-strategy-discriminator-secret",
        "vfs-cache-mode",
    ] {
        assert!(!text.contains(secret), "{secret} leaked: {text}");
    }
    assert_eq!(
        serialized["sandbox"]["session_state"]["__openai_agents_redacted_mount_authority"],
        json!(true)
    );

    // A snapshot edited to carry the raw payload again is sanitized on the way in as well.
    serialized["sandbox"]["session_state"] = raw_credential_session_state();
    let restored = RunState::from_json(serialized).unwrap();
    let text = format!("{:?}", restored.sandbox);
    assert!(!text.contains("raw-secret-key"));
    assert!(!text.contains("raw-strategy-discriminator-secret"));
}

fn manifest_sandbox(manifest: Value) -> Value {
    json!({
        "backend_id": "unimported-provider",
        "session_state": {"type": "unimported-provider", "manifest": manifest},
    })
}

#[test]
fn run_state_rejects_non_string_mount_entry_type_without_values() {
    let sentinel = "malformed-mount-entry-type-secret";
    let sandbox = manifest_sandbox(json!({
        "version": 1, "root": "/workspace",
        "entries": {"data": {
            "type": {"invalid": "discriminator"},
            "secret_access_key": sentinel,
            "mount_strategy": {"type": "in_container"},
        }},
        "environment": {"value": {}},
    }));
    for from_json in [false, true] {
        let message = sandbox_error(&sandbox, from_json);
        assert!(message.contains("invalid manifest"), "{message}");
        assert!(!message.contains(sentinel), "{message}");
    }
}

#[test]
fn run_state_rejects_malformed_manifest_entry_containers_without_values() {
    let sentinel = "malformed-entry-container-secret";
    let sandbox = manifest_sandbox(json!({
        "version": 1, "root": "/workspace", "entries": [sentinel],
        "environment": {"value": {}},
    }));
    for from_json in [false, true] {
        let message = sandbox_error(&sandbox, from_json);
        assert!(message.contains("invalid manifest"), "{message}");
        assert!(!message.contains(sentinel), "{message}");
    }
}

#[test]
fn run_state_rejects_non_mapping_session_manifest() {
    let sentinel = "non-mapping-manifest-secret";
    let sandbox = manifest_sandbox(json!([{"secret_access_key": sentinel}]));
    for from_json in [false, true] {
        let message = sandbox_error(&sandbox, from_json);
        assert!(message.contains("invalid manifest"), "{message}");
        assert!(!message.contains(sentinel), "{message}");
    }
}

#[test]
fn run_state_rejects_unknown_mount_discriminators_without_values() {
    for location in ["strategy", "pattern"] {
        let sentinel = format!("unknown-{location}-discriminator-secret");
        let mut strategy = json!({"type": "in_container", "pattern": {"type": "rclone"}});
        if location == "strategy" {
            strategy["type"] = json!(sentinel);
        } else {
            strategy["pattern"]["type"] = json!(sentinel);
        }
        let sandbox = manifest_sandbox(json!({
            "version": 1, "root": "/workspace",
            "entries": {"data": {"type": "s3_mount", "bucket": "bucket", "mount_strategy": strategy}},
            "environment": {"value": {}},
        }));
        for from_json in [false, true] {
            let message = sandbox_error(&sandbox, from_json);
            assert!(message.contains("invalid manifest"), "{message}");
            assert!(!message.contains(&sentinel), "{message}");
        }
    }
}

#[test]
fn run_state_redacts_unknown_mount_strategy_configuration() {
    let mut state = make_state();
    state.sandbox = Some(manifest_sandbox(json!({
        "version": 1, "root": "/workspace",
        "entries": {"data": {
            "type": "s3_mount", "bucket": "bucket",
            "mount_strategy": {
                "type": "in_container",
                "api_token": "custom-strategy-secret",
                "pattern": {
                    "type": "rclone",
                    "api_token": "nested-pattern-secret",
                    "options": {"authorization": "nested-options-secret"},
                },
            },
        }},
        "environment": {"value": {}},
    })));
    let serialized = state.to_json().unwrap();
    let strategy =
        &serialized["sandbox"]["session_state"]["manifest"]["entries"]["data"]["mount_strategy"];
    assert_eq!(strategy["type"], json!("in_container"));
    assert_eq!(strategy["pattern"]["type"], json!("rclone"));
    assert!(strategy.get("api_token").is_none());
    assert!(strategy["pattern"].get("api_token").is_none());
    assert!(strategy["pattern"].get("options").is_none());
    let text = serialized.to_string();
    for secret in [
        "custom-strategy-secret",
        "nested-pattern-secret",
        "nested-options-secret",
    ] {
        assert!(!text.contains(secret), "{secret}");
    }
}

#[test]
fn run_state_rejects_malformed_sandbox_session_envelopes_without_values() {
    let sentinel = "malformed-sandbox-secret";
    let cases = [
        json!(sentinel),
        json!({"session_state": [sentinel]}),
        json!({"sessions_by_agent": {"TestAgent": {
            "agent_name": "TestAgent", "session_state": [sentinel]}}}),
    ];
    for malformed in cases {
        for from_json in [false, true] {
            let message = sandbox_error(&malformed, from_json);
            assert!(message.contains("invalid envelope"), "{message}");
            assert!(!message.contains(sentinel), "{message}");
        }
    }
}

#[test]
fn run_state_preserves_blaxel_drive_mount() {
    // Python runs this with the provider entry registered and not; the extension mount is
    // classified by name here, so one payload covers both.
    let drive = json!({
        "type": "blaxel_drive_mount",
        "drive_name": "shared-drive",
        "drive_mount_path": "/data",
        "drive_path": "/",
        "drive_read_only": true,
        "mount_strategy": {"type": "blaxel_drive"},
    });
    let mut state = make_state();
    state.sandbox = Some(manifest_sandbox(json!({
        "version": 1, "root": "/workspace",
        "entries": {"drive": drive}, "environment": {"value": {}},
    })));
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let sandbox = restored.sandbox.unwrap();
    assert_eq!(
        sandbox["session_state"]["manifest"]["entries"]["drive"],
        drive
    );
}

#[test]
fn run_state_strips_credential_file_sources_and_inline_authority() {
    // The pieces of `_sanitize_raw_mount` the Python tests reach only through provider mounts:
    // a credential-file path blanks the inline file that materializes it, an endpoint with
    // userinfo is dropped, and an unrelated mount survives untouched.
    let mut state = make_state();
    state.sandbox = Some(manifest_sandbox(json!({
        "version": 1, "root": "/workspace",
        "entries": {
            "creds.json": {"type": "file", "content": "{\"private_key\": \"inline-key\"}"},
            "data": {
                "type": "gcs_mount", "bucket": "b",
                "service_account_file": "creds.json",
                "endpoint_url": "https://user:pw@example.test",
                "region": "us",
                "mount_strategy": {"type": "in_container", "pattern": {"type": "mountpoint"}},
            },
        },
        "environment": {"value": {}},
    })));
    let serialized = state.to_json().unwrap();
    let entries = &serialized["sandbox"]["session_state"]["manifest"]["entries"];
    assert_eq!(entries["creds.json"]["content"], json!(""));
    assert_eq!(entries["data"]["service_account_file"], Value::Null);
    assert_eq!(entries["data"]["endpoint_url"], Value::Null);
    assert_eq!(entries["data"]["region"], json!("us"));
    assert!(!serialized.to_string().contains("inline-key"));
}

#[test]
fn rust_refuses_an_unknown_mount_like_entry_in_the_sandbox_payload() {
    let sentinel = "unknown-mount-like-secret";
    let sandbox = manifest_sandbox(json!({
        "version": 1, "root": "/workspace",
        "entries": {"data": {"type": "future_mount", "secret": sentinel,
                             "mount_strategy": {"type": "in_container"}}},
        "environment": {"value": {}},
    }));
    let message = sandbox_error(&sandbox, true);
    assert!(message.contains("invalid manifest"), "{message}");
    assert!(!message.contains(sentinel));
}

// ---------------------------------------------------------------------------
// Nested-history ownership and the generated/session index map
// ---------------------------------------------------------------------------

const OWNED_DIGEST: &str = "8359850d0a341251272d79a50a45b89e915b2978946e4f3d4811edbaa36d16b3";
const NORMALIZED_DIGEST: &str = "a23714afb49491764c282bc5aed961ca7acdfb18f17f8495cf7e79597cbe377f";
const UNICODE_DIGEST: &str = "f3e137d47ed1cdc6a529c6514d71950ffa0427682b4c66ba62cf06fd6210826e";

fn owned_message() -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "owned", "annotations": []}],
    })
}

fn message_item(raw: Value) -> Value {
    json!({"type": "message_output_item", "agent": {"name": "TestAgent"}, "raw_item": raw})
}

/// A current-schema envelope whose single generated/session item is `raw`, with `original_input`
/// and `refs` as given.
fn ownership_wire(raw: Value, original_input: Value, refs: Value) -> Value {
    let mut wire = make_state().to_json().unwrap();
    wire["original_input"] = original_input;
    wire["generated_items"] = json!([message_item(raw.clone())]);
    wire["session_items"] = json!([message_item(raw)]);
    wire["generated_session_item_indexes"] = json!([0]);
    wire["nested_history_owned_session_item_refs"] = refs;
    wire
}

#[test]
fn nested_history_ownership_round_trips_and_defaults_for_schema_1_12() {
    let wire = ownership_wire(
        owned_message(),
        json!([owned_message()]),
        json!([{"index": 0, "digest": OWNED_DIGEST, "input_index": 0}]),
    );
    let restored = RunState::from_json(wire.clone()).unwrap();
    assert_eq!(restored.nested_history_owned_items, vec![0]);

    // Rust writes the same sidecar back, with Python's digest.
    let written = restored.to_json().unwrap();
    assert_eq!(
        written["nested_history_owned_session_item_refs"],
        json!([{"index": 0, "digest": OWNED_DIGEST, "input_index": 0}])
    );
    assert_eq!(written["generated_session_item_indexes"], json!([0]));

    // A released 1.12 snapshot has neither key and owns nothing.
    let mut legacy = wire;
    legacy["$schemaVersion"] = json!("1.12");
    legacy["context"]["function_tool_approvals"] = json!([]);
    let object = legacy.as_object_mut().unwrap();
    object.remove("nested_history_owned_session_item_refs");
    object.remove("generated_session_item_indexes");
    let restored = RunState::from_json(legacy).unwrap();
    assert!(restored.nested_history_owned_items.is_empty());
}

#[test]
fn nested_history_ownership_normalizes_raw_assistant_input_digest() {
    // The digest of an assistant input item does not depend on whether its content is a bare
    // string or output-text parts: Python hashes both to the same value.
    let session_raw = json!({
        "id": "msg_raw", "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "owned"}],
    });
    let wire = ownership_wire(
        session_raw,
        json!([{"id": "msg_raw", "type": "message", "role": "assistant", "content": "owned"}]),
        json!([{"index": 0, "digest": NORMALIZED_DIGEST, "input_index": 0}]),
    );
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(restored.nested_history_owned_items, vec![0]);
}

#[test]
fn nested_history_digest_matches_python_for_non_ascii_text() {
    // Python's `json.dumps` escapes non-ASCII as `\uXXXX` before hashing.
    let raw = json!({
        "id": "msg_u", "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "caf\u{e9} \u{1F600}", "annotations": []}],
    });
    let wire = ownership_wire(
        raw.clone(),
        json!([raw]),
        json!([{"index": 0, "digest": UNICODE_DIGEST, "input_index": 0}]),
    );
    assert_eq!(
        RunState::from_json(wire)
            .unwrap()
            .nested_history_owned_items,
        vec![0]
    );
}

#[test]
fn invalid_nested_history_ownership_sidecars_are_rejected() {
    let a64 = "a".repeat(64);
    let cases = [
        (json!({}), "must be a list of objects"),
        (json!(["not-an-object"]), "non-negative integer index"),
        (
            json!([{"index": -1, "digest": a64, "input_index": 0}]),
            "non-negative integer index",
        ),
        (
            json!([{"index": 0, "digest": "short", "input_index": 0}]),
            "non-negative integer index",
        ),
        (
            json!([{"index": 0, "digest": a64, "input_index": -1}]),
            "non-negative integer index",
        ),
        (
            json!([{"index": 9, "digest": a64, "input_index": 0}]),
            "missing session item",
        ),
        (
            json!([{"index": 0, "digest": a64, "input_index": 9}]),
            "missing input item",
        ),
        (
            json!([{"index": 0, "digest": a64, "input_index": 0}]),
            "session digest does not match",
        ),
    ];
    for (sidecar, expected) in cases {
        let wire = ownership_wire(owned_message(), json!([owned_message()]), sidecar.clone());
        let message = err_text(RunState::from_json(wire));
        assert!(message.contains(expected), "{sidecar}: {message}");
    }
}

#[test]
fn nested_history_ownership_with_changed_input_digest_is_rejected() {
    let mut changed = owned_message();
    changed["content"][0]["text"] = json!("changed");
    let wire = ownership_wire(
        owned_message(),
        json!([changed]),
        json!([{"index": 0, "digest": OWNED_DIGEST, "input_index": 0}]),
    );
    let message = err_text(RunState::from_json(wire));
    assert!(message.contains("input digest does not match"), "{message}");
}

#[test]
fn repeated_generated_item_identity_maps_to_distinct_session_occurrences() {
    let mut state = make_state();
    let item = RunItem::new(
        "TestAgent",
        RunItemKind::MessageOutput {
            raw_item: serde_json::from_value(owned_message()).unwrap(),
        },
    );
    state.generated_items = vec![item.clone(), item];
    let written = state.to_json().unwrap();
    assert_eq!(written["generated_session_item_indexes"], json!([0, 1]));
    assert_eq!(
        RunState::from_json(written).unwrap().generated_items.len(),
        2
    );
}

#[test]
fn invalid_generated_session_item_indexes_are_ignored() {
    let mapping_cases = [
        json!("not-a-list"),
        json!([0]),
        json!([-1, null]),
        json!([2, null]),
        json!([true, null]),
        json!([0, 0]),
    ];
    for invalid in mapping_cases {
        let _guard = capture_logs();
        let first = message_item(owned_message());
        let mut wire = make_state().to_json().unwrap();
        wire["generated_items"] = json!([first.clone(), first.clone()]);
        wire["session_items"] = json!([first.clone(), first]);
        wire["generated_session_item_indexes"] = invalid.clone();
        let restored = RunState::from_json(wire).unwrap();
        assert_eq!(restored.generated_items.len(), 2, "{invalid}");
        assert!(
            logged().contains("Ignoring invalid generated_session_item_indexes"),
            "{invalid}: {}",
            logged()
        );
    }
}

#[test]
fn mismatched_generated_session_occurrence_is_not_aliased() {
    let _guard = capture_logs();
    let mut changed = owned_message();
    changed["content"][0]["text"] = json!("changed");
    let mut wire = ownership_wire(owned_message(), json!([]), json!([]));
    wire["session_items"] = json!([message_item(changed)]);
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(restored.generated_items.len(), 1);
    assert!(
        logged().contains("Ignoring mismatched generated/session occurrence"),
        "{}",
        logged()
    );
}

// ---------------------------------------------------------------------------
// Fallible approval decisions
// ---------------------------------------------------------------------------

fn approval_with(extra: Value, name: &str, call_id: &str, arguments: &str) -> ToolApprovalItem {
    let mut call = FunctionCall::new(name, arguments);
    call.call_id = call_id.into();
    for (key, value) in extra.as_object().cloned().unwrap_or_default() {
        call.extra.insert(key, value);
    }
    ToolApprovalItem {
        agent: "TestAgent".into(),
        raw_item: call,
        tool_name: name.into(),
        tool_origin: None,
    }
}

fn hosted_request(request_id: &str, server_label: Value) -> ToolApprovalItem {
    approval_with(
        json!({"type": "mcp_approval_request", "server_label": server_label}),
        "lookup_account",
        request_id,
        "{}",
    )
}

#[test]
fn approval_decision_requires_canonical_invocation() {
    let cases = [
        approval_with(json!({"type": "unknown_tool_call"}), "tool1", "cid1", "{}"),
        hosted_request("request-empty-name", json!("server-a")).tap_name(""),
        hosted_request("request-null-server", Value::Null),
        approval_with(
            json!({"type": "hosted_tool_call",
                   "provider_data": {"type": "mcp_approval_request", "name": "",
                                     "server_label": "server-a", "arguments": "{}"}}),
            "",
            "request-wrapped-empty-name",
            "{}",
        ),
    ];
    for (index, item) in cases.into_iter().enumerate() {
        let mut state = make_state();
        let result = state.try_approve(&item, false);
        assert!(result.is_err(), "case {index} was accepted: {item:?}");
        let message = err_text(result);
        assert!(
            message.contains("canonical invocation identity")
                || message.contains("non-empty request id"),
            "{message}"
        );
        assert!(state.approvals.0.is_empty());
    }
}

trait TapName {
    fn tap_name(self, name: &str) -> Self;
}

impl TapName for ToolApprovalItem {
    fn tap_name(mut self, name: &str) -> Self {
        self.raw_item.name = name.into();
        self.tool_name = name.into();
        self
    }
}

#[test]
fn current_schema_missing_call_id_cannot_create_sticky_approval() {
    let mut state = make_state();
    let item = approval_with(json!({}), "tool1", "", r#"{"value":"safe"}"#);
    let message = err_text(state.try_approve(&item, false));
    assert!(message.contains("non-empty call ID"), "{message}");
    let message = err_text(state.try_approve(&item, true));
    assert!(message.contains("non-empty call ID"), "{message}");
    assert!(state.approvals.0.is_empty());
    // A fresh, well-formed call is not covered by the refused decision.
    assert_eq!(
        state.approvals.status("TestAgent", "tool1", "cid-fresh"),
        None
    );
}

#[test]
fn incomplete_hosted_mcp_query_cannot_create_approval_authority() {
    let mut state = make_state();
    let item = approval_with(
        json!({"type": "hosted_tool_call",
               "provider_data": {"type": "mcp_approval_request", "id": "request-a"}}),
        "lookup_account",
        "request-a",
        "{}",
    );
    let message = err_text(state.try_reject(&item, false, Some("exact denial".into())));
    assert!(
        message.contains("canonical invocation identity"),
        "{message}"
    );
    assert!(state.approvals.0.is_empty());
}

#[test]
fn persistent_hosted_mcp_decision_requires_server_label_and_tool_name() {
    let mut state = make_state();
    let item = hosted_request("request-a", Value::Null);
    let message = err_text(state.try_approve(&item, true));
    assert!(message.contains("server_label"), "{message}");
    let item = hosted_request("", json!("server-a"));
    let message = err_text(state.try_approve(&item, false));
    assert!(message.contains("non-empty request id"), "{message}");
    assert!(state.approvals.0.is_empty());
}

#[test]
fn a_well_formed_decision_is_recorded_by_the_fallible_api() {
    let mut state = make_state();
    let item = approval_with(json!({}), "tool1", "cid1", r#"{"value":"safe"}"#);
    state.try_approve(&item, false).unwrap();
    assert_eq!(
        state.approvals.status("TestAgent", "tool1", "cid1"),
        Some(true)
    );
    let hosted = hosted_request("request-a", json!("server-a"));
    state
        .try_reject(&hosted, false, Some("denied".into()))
        .unwrap();
    assert_eq!(
        state
            .approvals
            .status_for_item(&hosted, "request-a")
            .approved,
        Some(false)
    );
}

#[test]
fn hosted_mcp_approval_round_trip_uses_typed_identity_records() {
    let mut state = make_state();
    let item = hosted_request("request-a", json!("server-a"));
    state.approve(&item, true);
    let serialized = state.to_json().unwrap();
    assert_eq!(serialized["context"]["approvals"], json!({}));
    let hosted = serialized["context"]["hosted_mcp_approvals"]
        .as_array()
        .unwrap();
    let identities: Vec<&Value> = hosted.iter().map(|e| &e["identity"]).collect();
    assert_eq!(
        identities,
        vec![
            &json!({"type": "server_tool", "server_label": "server-a", "tool_name": "lookup_account"}),
            &json!({"type": "query", "tool_name": "lookup_account", "request_id": "request-a"}),
        ]
    );
    assert_eq!(hosted[0]["decision"]["approved"], json!(true));
    assert_eq!(hosted[0]["decision"]["rejected"], json!([]));
    assert_eq!(hosted[1]["decision"]["approved"], json!(["request-a"]));
    assert!(hosted[1]["decision"].get("invocations").is_none());
    let restored = RunState::from_json(serialized).unwrap();
    assert_eq!(
        restored
            .approvals
            .status_for_item(&item, "request-a")
            .approved,
        Some(true)
    );
    // The sticky server/tool record also covers a later request to the same server.
    let next = hosted_request("request-next", json!("server-a"));
    assert_eq!(
        restored
            .approvals
            .status_for_item(&next, "request-next")
            .approved,
        Some(true)
    );
}

#[test]
fn hosted_mcp_rejection_query_round_trip_does_not_cross_servers() {
    let mut state = make_state();
    let server_a = hosted_request("shared-request", json!("server-a"));
    let server_b = hosted_request("shared-request", json!("server-b"));
    state.reject(&server_a, false, Some("server-a denied".into()));
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let decision = restored
        .approvals
        .status_for_item(&server_a, "shared-request");
    assert_eq!(decision.approved, Some(false));
    assert_eq!(
        decision.rejection_message.as_deref(),
        Some("server-a denied")
    );
    // The same request id on another server is undecided.
    let other = restored
        .approvals
        .status_for_item(&server_b, "shared-request");
    assert_eq!(other.approved, None);
    assert_eq!(other.rejection_message, None);
}

#[test]
fn schema_1_13_ignores_typed_hosted_mcp_approval_records() {
    let mut state = make_state();
    state.approve(&hosted_request("request-a", json!("server-a")), true);
    let mut wire = state.to_json().unwrap();
    wire["$schemaVersion"] = json!("1.13");
    wire["context"]["function_tool_approvals"] = json!([]);
    let restored = RunState::from_json(wire).unwrap();
    let next = hosted_request("request-next", json!("server-a"));
    assert_eq!(
        restored
            .approvals
            .status_for_item(&next, "request-next")
            .approved,
        None
    );
}

// ---------------------------------------------------------------------------
// Item metadata that survives a read
// ---------------------------------------------------------------------------

fn approval_item_wire() -> Value {
    json!({
        "type": "tool_approval_item",
        "agent": {"name": "TestAgent"},
        "tool_name": "get_weather",
        "raw_item": {"type": "function_call", "name": "get_weather", "call_id": "call123",
                     "status": "completed", "arguments": "{}", "namespace": "get_weather"},
        "tool_namespace": "get_weather",
        "tool_lookup_key": {"kind": "deferred_top_level", "name": "get_weather"},
        "allow_bare_name_alias": true,
    })
}

#[test]
fn round_trip_serialization_preserves_allow_bare_name_alias_and_tool_lookup_key() {
    let mut wire = make_state().to_json().unwrap();
    wire["generated_items"] = json!([approval_item_wire()]);
    let restored = RunState::from_json(wire).unwrap();
    let item = &restored.generated_items[0];
    assert!(item.allow_bare_name_alias);
    assert_eq!(
        item.tool_lookup_key,
        Some(json!({"kind": "deferred_top_level", "name": "get_weather"}))
    );
    let written = restored.to_json().unwrap();
    let item = &written["generated_items"][0];
    assert_eq!(item["allow_bare_name_alias"], json!(true));
    assert_eq!(
        item["tool_lookup_key"],
        json!({"kind": "deferred_top_level", "name": "get_weather"})
    );
}

#[test]
fn malformed_item_metadata_is_dropped_like_python() {
    let mut item = approval_item_wire();
    item["allow_bare_name_alias"] = json!("yes");
    item["tool_lookup_key"] = json!("not-a-mapping");
    let mut wire = make_state().to_json().unwrap();
    wire["generated_items"] = json!([item]);
    let restored = RunState::from_json(wire).unwrap();
    assert!(!restored.generated_items[0].allow_bare_name_alias);
    assert_eq!(restored.generated_items[0].tool_lookup_key, None);
}

#[test]
fn tool_input_survives_serialization_round_trip() {
    let mut state = make_state().with_context(json!({"foo": "bar"}));
    state.tool_input = Some(json!({"text": "hola", "target": "en"}));
    let wire = state.to_json().unwrap();
    assert_eq!(
        wire["context"]["tool_input"],
        json!({"text": "hola", "target": "en"})
    );
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(
        restored.tool_input,
        Some(json!({"text": "hola", "target": "en"}))
    );
    assert!(
        make_state().to_json().unwrap()["context"]
            .get("tool_input")
            .is_none()
    );
}

fn items_wire(items: Value) -> Value {
    let mut wire = make_state().to_json().unwrap();
    wire["generated_items"] = items;
    wire
}

// ---------------------------------------------------------------------------
// Programmatic tool calling on resume
// ---------------------------------------------------------------------------

fn program_state(function_call_caller: Value, with_program: bool, completed: bool) -> RunState {
    let program = json!({
        "id": "program_item", "call_id": "call_program", "code": "saved_lookup()",
        "fingerprint": "fingerprint", "type": "program",
    });
    let function_call = json!({
        "id": "function_item", "call_id": "call_function", "name": "saved_lookup",
        "arguments": "{}", "caller": function_call_caller, "type": "function_call",
    });
    let mut output = Vec::new();
    if with_program {
        output.push(program);
    }
    output.push(function_call.clone());
    if completed {
        output.push(json!({
            "id": "program_output_item", "call_id": "call_program", "result": "done",
            "status": "completed", "type": "program_output",
        }));
    }
    let mut state = make_state();
    state.model_responses = vec![openai_agents_sdk::items::ModelResponse {
        output: output
            .into_iter()
            .map(openai_agents_sdk::items::OutputItem::from_value)
            .collect(),
        usage: Default::default(),
        response_id: Some("response_1".into()),
        request_id: None,
        raw_usage: None,
    }];
    state.pending_tool_processing = true;
    state
}

fn saved_lookup(allowed_callers: Option<Vec<&str>>) -> FunctionTool<()> {
    let tool = FunctionTool::<()>::new("saved_lookup", "Lookup", |_c, _a: Value| async {
        Ok("saved".to_string())
    });
    match allowed_callers {
        Some(callers) => {
            tool.with_allowed_callers(callers.into_iter().map(str::to_string).collect())
        }
        None => tool,
    }
}

async fn resume_program(state: RunState, tool: FunctionTool<()>) -> openai_agents_sdk::Result<()> {
    resume_program_with(state, tool, true).await
}

async fn resume_program_with(
    state: RunState,
    tool: FunctionTool<()>,
    with_program_tool: bool,
) -> openai_agents_sdk::Result<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let mut agent = Agent::<()>::new("TestAgent").with_model_instance(model);
    if with_program_tool {
        agent = agent.with_tool(openai_agents_sdk::tool::ProgrammaticToolCallingTool);
    }
    agent = agent.with_tool(tool);
    Runner::run(&agent, state).await.map(|_| ())
}

fn program_caller() -> Value {
    json!({"type": "program", "caller_id": "call_program"})
}

#[tokio::test]
async fn programmatic_tool_calling_rechecks_allowed_callers_on_resume() {
    // Saved with a tool that allowed programmatic callers; resumed with one that does not.
    let state = program_state(program_caller(), true, false);
    let result = resume_program(state, saved_lookup(None)).await;
    let message = err_text(result);
    assert!(message.contains("saved_lookup"), "{message}");
}

#[tokio::test]
async fn programmatic_tool_calling_requires_configured_tool_on_resume() {
    let state = program_state(program_caller(), true, false);
    let result = resume_program_with(state, saved_lookup(Some(vec!["programmatic"])), false).await;
    let message = err_text(result);
    assert!(message.contains("programmatic_tool_calling"), "{message}");
}

#[tokio::test]
async fn programmatic_tool_calling_rejects_missing_parent_on_resume() {
    let state = program_state(
        json!({"type": "program", "caller_id": "missing_program"}),
        false,
        false,
    );
    let result = resume_program(state, saved_lookup(Some(vec!["programmatic"]))).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn programmatic_tool_calling_rejects_completed_parent_on_resume() {
    let state = program_state(program_caller(), true, true);
    let result = resume_program(state, saved_lookup(Some(vec!["programmatic"]))).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn programmatic_tool_calling_accepts_a_live_parent_on_resume() {
    // The positive control: an unfinished program that allows programmatic callers resumes.
    let state = program_state(program_caller(), true, false);
    let result = resume_program(state, saved_lookup(Some(vec!["programmatic"]))).await;
    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
async fn programmatic_mcp_approval_rechecks_allowed_callers_on_resume() {
    let program = json!({
        "id": "program_item", "call_id": "call_program", "code": "tools.docs_server.lookup()",
        "fingerprint": "fingerprint", "type": "program",
    });
    let request = json!({
        "id": "approval_item", "arguments": "{}", "name": "lookup", "server_label": "docs_server",
        "type": "mcp_approval_request",
        "caller": {"type": "program", "caller_id": "call_program"},
    });
    let build_state = || {
        let mut state = make_state();
        state.model_responses = vec![openai_agents_sdk::items::ModelResponse {
            output: [&program, &request]
                .into_iter()
                .map(|v| openai_agents_sdk::items::OutputItem::from_value(v.clone()))
                .collect(),
            usage: Default::default(),
            response_id: Some("response_1".into()),
            request_id: None,
            raw_usage: None,
        }];
        state.pending_tool_processing = true;
        state
    };
    let run = |allowed: &str| {
        let state = build_state();
        let allowed = allowed.to_string();
        async move {
            openai_agents_sdk::tracing::set_tracing_disabled(true);
            let model = Arc::new(FakeModel::new());
            model.add_turn(vec![assistant_message("done")]);
            let tool = openai_agents_sdk::tool::HostedMCPTool::<()>::new(json!({
                "type": "mcp", "server_label": "docs_server",
                "server_url": "https://example.com/mcp", "allowed_callers": [allowed],
            }));
            let agent = Agent::<()>::new("TestAgent")
                .with_tool(openai_agents_sdk::tool::ProgrammaticToolCallingTool)
                .with_tool(tool)
                .with_model_instance(model);
            Runner::run(&agent, state).await.map(|_| ())
        }
    };
    // Saved with a server that allowed programs; resumed with one that only allows direct calls.
    let message = err_text(run("direct").await);
    assert!(
        message.contains("docs_server") || message.contains("hosted_mcp"),
        "{message}"
    );
    // The same resume against a server that allows programmatic callers is accepted.
    let result = run("programmatic").await;
    assert!(result.is_ok() || !err_text(result).contains("caller"));
}

#[test]
fn deserialize_rejects_invalid_local_shell_call_output() {
    let cases = [
        json!({"type": "local_shell_call_output", "call_id": "call123"}),
        json!({"type": "local_shell_call_output", "id": "shell123", "output": "result"}),
        json!({"type": "local_shell_call_output", "call_id": 123, "output": "result"}),
        json!({"type": "local_shell_call_output", "call_id": [99, 97], "output": "result"}),
        json!({"type": "local_shell_call_output", "call_id": "", "output": "result"}),
        json!({"type": "local_shell_call_output", "call_id": "call123", "output": 123}),
        json!({"type": "local_shell_call_output", "call_id": "call123", "output": [114, 101]}),
    ];
    for raw in cases {
        let message = err_text(RunState::from_json(items_wire(json!([{
            "type": "tool_call_output_item", "agent": {"name": "TestAgent"}, "raw_item": raw,
        }]))));
        assert!(
            message.contains("local_shell_call_output"),
            "{raw}: {message}"
        );
    }
    // The valid shape is kept verbatim.
    let raw = json!({"type": "local_shell_call_output", "id": "shell123", "call_id": "call123",
                     "output": "result"});
    let restored = RunState::from_json(items_wire(json!([{
        "type": "tool_call_output_item", "agent": {"name": "TestAgent"}, "raw_item": raw,
    }])))
    .unwrap();
    match &restored.generated_items[0].kind {
        RunItemKind::ToolCallOutput { raw_item, .. } => assert_eq!(raw_item, &raw),
        other => panic!("{other:?}"),
    }
}

#[test]
fn approve_tool_rejects_uncanonical_hosted_call_dict() {
    // A generic hosted call cannot create approval authority from its item id.
    let mut state = make_state();
    let mut call = FunctionCall::new("hosted_tool", "{}");
    call.call_id = "hosted_call_123".into();
    call.extra.insert("type".into(), json!("hosted_tool_call"));
    let item = ToolApprovalItem {
        agent: "TestAgent".into(),
        raw_item: call,
        tool_name: "hosted_tool".into(),
        tool_origin: None,
    };
    let message = err_text(state.try_approve(&item, false));
    assert!(
        message.contains("canonical invocation identity"),
        "{message}"
    );
    assert_eq!(
        state
            .approvals
            .status("TestAgent", "hosted_tool", "hosted_call_123"),
        None
    );
}

#[test]
fn tool_approval_item_with_an_unrecognized_raw_item_is_kept() {
    // Python falls back to the raw dict when the approval payload is not a function call.
    let items = RunState::from_json(items_wire(json!([{
        "type": "tool_approval_item", "agent": {"name": "TestAgent"},
        "raw_item": {"type": "invalid"},
    }])))
    .unwrap()
    .generated_items;
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].item_type(), "tool_approval_item");
}

#[test]
fn hosted_mcp_exact_rejection_overrides_sticky_approval_after_round_trip() {
    let mut state = make_state();
    state.approve(&hosted_request("sticky-request", json!("server-a")), true);
    let exception = hosted_request("exception-request", json!("server-a"));
    state.reject(&exception, false, Some("denied exactly".into()));
    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let decision = restored
        .approvals
        .status_for_item(&exception, "exception-request");
    assert_eq!(decision.approved, Some(false));
    assert_eq!(
        decision.rejection_message.as_deref(),
        Some("denied exactly")
    );
    let other = hosted_request("other-request", json!("server-a"));
    assert_eq!(
        restored
            .approvals
            .status_for_item(&other, "other-request")
            .approved,
        Some(true)
    );
}
