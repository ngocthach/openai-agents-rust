//! Port of the Python SDK's `tests/test_shell_call_serialization.py`.
//!
//! Shell call normalization is `ShellCallData::from_raw`. Output normalization and the
//! canonical outcome payload are observed through a `Runner` run, which records the
//! `shell_call_output` item the SDK sends back to the model.

use std::sync::Arc;

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{
    ShellCallData, ShellCallOutcome, ShellCommandOutput, ShellResult, ShellTool,
};
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

/// Runs one shell call through `tool` and returns the `shell_call_output` item the model sees on
/// its second request.
async fn shell_output_sent_back(tool: ShellTool<()>, shell_call: Value) -> Value {
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(shell_call)]);
    model.add_turn(vec![assistant_message("done")]);
    Runner::run(&agent, "run").await.unwrap();
    let second: Vec<Value> = model.calls()[1].input.iter().map(|i| i.0.clone()).collect();
    second
        .into_iter()
        .find(|i| i["type"] == "shell_call_output")
        .expect("shell_call_output is replayed to the model")
}

fn normalize(raw: Value) -> Result<ShellCallData, AgentsError> {
    ShellCallData::from_raw(&raw)
}

#[test]
fn coerce_shell_call_reads_max_output_length() {
    let tool_call = json!({
        "call_id": "shell-1",
        "action": {"commands": ["ls"], "maxOutputLength": 512},
        "status": "in_progress",
    });
    let result = normalize(tool_call).unwrap();
    assert_eq!(result.action.max_output_length, Some(512));
}

#[test]
fn coerce_shell_call_treats_zero_timeout_as_unspecified() {
    // Python parametrizes three timeout keys and two zero spellings (int and float).
    for timeout_key in ["timeout_ms", "timeoutMs", "timeout"] {
        for timeout_value in [json!(0), json!(0.0)] {
            let mut action = json!({"commands": ["ls"]});
            action[timeout_key] = timeout_value.clone();
            let tool_call = json!({"call_id": "shell-zero-timeout", "action": action});

            let result = normalize(tool_call).unwrap();

            assert_eq!(
                result.action.timeout_ms, None,
                "{timeout_key}={timeout_value} should be unspecified"
            );
        }
    }
}

#[test]
fn coerce_shell_call_preserves_positive_timeout() {
    for timeout_key in ["timeout_ms", "timeoutMs", "timeout"] {
        let mut action = json!({"commands": ["ls"]});
        action[timeout_key] = json!(250);
        let tool_call = json!({"call_id": "shell-positive-timeout", "action": action});

        let result = normalize(tool_call).unwrap();

        assert_eq!(result.action.timeout_ms, Some(250), "key {timeout_key}");
    }
}

#[test]
fn coerce_shell_call_rejects_unsupported_timeout_values() {
    for timeout_value in [json!(-1), json!(0.5), json!(false), json!("250")] {
        let tool_call = json!({
            "call_id": "shell-invalid-timeout",
            "action": {"commands": ["ls"], "timeout_ms": timeout_value},
        });

        let err = normalize(tool_call).expect_err("unsupported timeout must be rejected");

        assert!(
            err.to_string()
                .contains("Shell call action timeout must be a positive integer"),
            "unexpected message for {timeout_value}: {err}"
        );
    }
}

#[test]
fn coerce_shell_call_requires_commands() {
    let tool_call = json!({"call_id": "shell-2", "action": {"commands": []}});
    assert!(normalize(tool_call).is_err());
}

#[test]
fn coerce_shell_call_rejects_string_commands() {
    // Python also rejects `bytes` and `bytearray` commands; those have no JSON form, so only the
    // string-typed value can reach the Rust normalizer.
    let tool_call = json!({"call_id": "shell-3", "action": {"commands": "echo hi"}});
    let err = normalize(tool_call).expect_err("string commands must be rejected");
    assert!(
        err.to_string()
            .contains("Shell call action commands must be a sequence of command strings."),
        "unexpected message: {err}"
    );
}

#[tokio::test]
async fn normalize_shell_output_handles_timeout() {
    // A timed-out command keeps its provider data and reports the `timeout` status.
    let tool = ShellTool::<()>::new(|_req| async {
        Ok::<_, AgentsError>(ShellResult {
            output: vec![ShellCommandOutput {
                outcome: ShellCallOutcome::Timeout,
                provider_data: Some(json!({"truncated": true})),
                ..Default::default()
            }],
            ..Default::default()
        })
    });
    let call = json!({
        "type": "shell_call",
        "id": "sh_timeout",
        "call_id": "call-timeout",
        "status": "completed",
        "action": {"commands": ["sleep 10"]},
    });

    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(call)]);
    model.add_turn(vec![assistant_message("done")]);
    let result = Runner::run(&agent, "run").await.unwrap();

    // The model-facing entry reports the timeout outcome; the run item keeps the normalized
    // command output (status and provider data) that Python exposes on `ShellCommandOutput`.
    let raw = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("shell output item");
    assert_eq!(raw["output"][0]["outcome"]["type"], "timeout");
    assert!(raw["output"][0]["outcome"].get("exit_code").is_none());
    let entry = &raw["shell_output"][0];
    assert_eq!(entry["status"], "timeout");
    assert_eq!(entry["provider_data"], json!({"truncated": true}));
}

#[tokio::test]
async fn serialize_shell_output_emits_canonical_outcome() {
    let tool = ShellTool::<()>::new(|_req| async {
        Ok::<_, AgentsError>(ShellResult {
            output: vec![ShellCommandOutput {
                stdout: "hello".into(),
                outcome: ShellCallOutcome::Exit { exit_code: Some(0) },
                ..Default::default()
            }],
            ..Default::default()
        })
    });
    let call = json!({
        "type": "shell_call", "id": "sh_canon", "call_id": "call-canon",
        "status": "completed", "action": {"commands": ["echo hello"]},
    });

    let sent = shell_output_sent_back(tool, call).await;

    let outcome = &sent["output"][0]["outcome"];
    assert_eq!(outcome["type"], "exit");
    assert_eq!(outcome["exit_code"], 0);
    assert!(outcome.get("exitCode").is_none());
}

#[tokio::test]
async fn shell_rejection_payload_preserves_missing_exit_code() {
    let tool = ShellTool::<()>::new(|_req| async {
        Ok::<_, AgentsError>(ShellResult {
            output: vec![ShellCommandOutput {
                stderr: "rejected".into(),
                outcome: ShellCallOutcome::Exit { exit_code: None },
                ..Default::default()
            }],
            ..Default::default()
        })
    });
    let call = json!({
        "type": "shell_call", "id": "sh_rej", "call_id": "call-1",
        "status": "completed", "action": {"commands": ["false"]},
    });

    let sent = shell_output_sent_back(tool, call).await;

    let outcome = &sent["output"][0]["outcome"];
    assert!(outcome["exit_code"].is_null());
    assert!(outcome.get("exitCode").is_none());
}

#[tokio::test]
async fn shell_output_preserves_zero_exit_code() {
    let tool = ShellTool::<()>::new(|_req| async {
        Ok::<_, AgentsError>(ShellResult {
            output: vec![ShellCommandOutput {
                stdout: "ok".into(),
                outcome: ShellCallOutcome::Exit { exit_code: Some(0) },
                ..Default::default()
            }],
            ..Default::default()
        })
    });
    let call = json!({
        "type": "shell_call", "id": "sh_zero", "call_id": "call-2",
        "status": "completed", "action": {"commands": ["true"]},
    });

    let sent = shell_output_sent_back(tool, call).await;

    let outcome = &sent["output"][0]["outcome"];
    assert_eq!(outcome["exit_code"], 0);
    assert!(outcome.get("exitCode").is_none());
}
