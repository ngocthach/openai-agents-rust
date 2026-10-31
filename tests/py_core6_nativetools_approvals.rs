//! Ports the two schema-1.7 `approve` / `always_approve` cases of
//! `test_run_impl_resume_paths.py::test_resolve_interrupted_turn_only_uses_name_fallback_for_legacy_approval_agents`.
//!
//! Two agents share the name `duplicate`; the parked approval belongs to the second and its
//! snapshot lost the agent identity. From schema 1.7 the owner then resolves to the first
//! duplicate, so a decision on it must not run the call for the second: the run asks again.
//! (The other 22 cases of the test live in `py_core6_resume_name_fallback.rs`.)

// The shared support module (owned by the resume tests) has a collapsible `if`.
#![allow(clippy::collapsible_if)]

mod py_core6_resume_support;

use std::sync::{Arc, Mutex};

use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunState};
use py_core6_resume_support::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, JsonSchema)]
struct Text {
    text: String,
}

#[tokio::test]
async fn identity_less_approval_is_refused_from_schema_1_7() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let mut failures: Vec<String> = Vec::new();
    for schema_version in ["1.7"] {
        for decision in ["approve", "always_approve"] {
            let label = format!("{schema_version} {decision}");
            let calls = Arc::new(Mutex::new(Vec::<String>::new()));
            let seen = calls.clone();
            let needs_ok = || {
                let seen = seen.clone();
                FunctionTool::new("needs_ok", "", move |_c, a: Text| {
                    let seen = seen.clone();
                    async move {
                        seen.lock().unwrap().push(a.text.clone());
                        Ok(a.text)
                    }
                })
                .with_needs_approval(true)
            };
            let model = scripted(vec![
                vec![call("transfer_to_zeta", json!({}), "handoff-1")],
                vec![call("needs_ok", json!({"text": "one"}), "legacy-call")],
                vec![assistant_message("done")],
            ]);
            let alpha = Arc::new(
                Agent::<()>::new("duplicate")
                    .with_instructions("alpha")
                    .with_model_instance(model.clone())
                    .with_tool(needs_ok()),
            );
            let zeta = Arc::new(
                Agent::<()>::new("duplicate")
                    .with_instructions("zeta")
                    .with_model_instance(model.clone())
                    .with_tool(needs_ok()),
            );
            let root = Agent::<()>::new("triage")
                .with_model_instance(model.clone())
                .with_handoff(handoff(alpha).with_tool_name("transfer_to_alpha"))
                .with_handoff(handoff(zeta).with_tool_name("transfer_to_zeta"));
            let paused = match drive(&root, "input", None, false, &RunCheckpoint::new()).await {
                Ok(p) => p,
                Err(e) => {
                    failures.push(format!("{label}: setup failed: {e}"));
                    continue;
                }
            };
            let state = paused.state.unwrap();
            let mut payload = state.to_json().unwrap();
            // The interruption's agent loses its identity, as a pre-identity writer left it.
            if let Some(agent) =
                payload["current_step"]["data"]["interruptions"][0]["agent"].as_object_mut()
            {
                agent.remove("identity");
            }
            if let Some(entries) = payload["context"]
                .as_object_mut()
                .and_then(|c| c.remove("function_tool_approvals"))
                .and_then(|e| e.as_array().cloned())
            {
                for entry in entries {
                    payload["context"]["approvals"][entry["tool_key"].as_str().unwrap()] =
                        entry["decision"].clone();
                }
            }
            payload["$schemaVersion"] = json!(schema_version);
            if decision.starts_with("legacy_") {
                payload["context"]["approvals"]["needs_ok"] = json!({
                    "approved": decision == "legacy_approve",
                    "rejected": decision == "legacy_reject",
                    "sticky_rejection_message": "Old unowned rejection",
                });
            }
            let mut restored = match RunState::from_json(payload) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("{label}: restore failed: {e}"));
                    continue;
                }
            };
            let item = restored.get_interruptions()[0].clone();
            match decision {
                "approve" | "always_approve" => {
                    restored.approve(&item, decision == "always_approve")
                }
                "reject" | "always_reject" => restored.reject(
                    &item,
                    decision == "always_reject",
                    Some("Legacy exact rejection".into()),
                ),
                _ => {}
            }
            let result = drive(&root, restored, None, false, &RunCheckpoint::new()).await;
            let (outputs, interrupted): (Vec<String>, bool) = match &result {
                Ok(o) => (
                    o.state
                        .as_ref()
                        .map(|s| {
                            s.generated_items
                                .iter()
                                .filter_map(|i| match &i.kind {
                                    RunItemKind::ToolCallOutput { output, .. } => {
                                        Some(output.as_str().unwrap_or_default().to_string())
                                    }
                                    _ => None,
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    !o.interruptions.is_empty(),
                ),
                Err(e) => {
                    failures.push(format!("{label}: resume failed: {e}"));
                    continue;
                }
            };
            let executed = calls.lock().unwrap().clone();
            let ok = executed.is_empty() && !outputs.iter().any(|o| o == "one");
            if !ok {
                failures.push(format!(
                    "{label}: executed={executed:?} outputs={outputs:?} interrupted={interrupted}"
                ));
            }
            let _: Option<Value> = None;
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
