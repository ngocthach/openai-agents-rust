//! Sticky approvals are scoped by how a tool was invoked: a native `apply_patch_call` and a
//! custom tool are both called `apply_patch`, and a permanent decision about one must not decide
//! the other. Ports the native-sticky half of
//! `test_multi_operation_checker_ignores_native_sticky_approval`
//! (`tests/sandbox/capabilities/test_apply_patch_tool.py`) and the key-scoping rules of
//! `RunContextWrapper._approval_keys_for_scope`.
//!
//! Python's scope is a hash of the invocation payload; Rust scopes by the invocation type
//! (`apply_patch_call`, `custom_tool_call`, `function_call`), which separates the same tools.

use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::run_context::{
    ApprovalRecord, ApprovalsState, approval_tool_key, scoped_tool_key,
};
use openai_agents_sdk::{RunContext, RunState};
use serde_json::json;

fn call_of(kind: &str, call_id: &str) -> FunctionCall {
    let mut call = FunctionCall::new("apply_patch", "{}");
    call.call_id = call_id.to_string();
    if kind != "function_call" {
        call.extra.insert("type".into(), json!(kind));
    }
    call
}

fn approval(agent: &str, kind: &str, call_id: &str) -> ToolApprovalItem {
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: call_of(kind, call_id),
        tool_name: "apply_patch".into(),
        tool_origin: None,
    }
}

#[test]
fn only_apply_patch_is_scoped_by_invocation_type() {
    assert_eq!(
        scoped_tool_key("apply_patch", Some("apply_patch_call")),
        "apply_patch:apply_patch_call"
    );
    assert_eq!(
        scoped_tool_key("apply_patch", Some("custom_tool_call")),
        "apply_patch:custom_tool_call"
    );
    assert_eq!(
        scoped_tool_key("apply_patch", None),
        "apply_patch:function_call"
    );
    assert_eq!(scoped_tool_key("shell", Some("shell_call")), "shell");
    assert_eq!(
        approval_tool_key("apply_patch", &call_of("custom_tool_call", "c")),
        "apply_patch:custom_tool_call"
    );
}

#[test]
fn a_native_sticky_approval_does_not_approve_the_custom_tool() {
    let ctx = RunContext::new(());
    ctx.approve_tool(&approval("patcher", "apply_patch_call", "native"), true);

    let native = scoped_tool_key("apply_patch", Some("apply_patch_call"));
    let custom = scoped_tool_key("apply_patch", Some("custom_tool_call"));
    assert_eq!(
        ctx.is_tool_approved("patcher", &native, "later"),
        Some(true)
    );
    assert_eq!(ctx.is_tool_approved("patcher", &custom, "later"), None);
}

#[test]
fn a_custom_sticky_rejection_does_not_reject_the_native_tool() {
    let ctx = RunContext::new(());
    ctx.reject_tool(
        &approval("patcher", "custom_tool_call", "custom"),
        true,
        Some("custom denied".into()),
    );

    let native = scoped_tool_key("apply_patch", Some("apply_patch_call"));
    let custom = scoped_tool_key("apply_patch", Some("custom_tool_call"));
    assert_eq!(
        ctx.is_tool_approved("patcher", &custom, "later"),
        Some(false)
    );
    assert_eq!(
        ctx.rejection_message("patcher", &custom, "later")
            .as_deref(),
        Some("custom denied")
    );
    assert_eq!(ctx.is_tool_approved("patcher", &native, "later"), None);
    assert_eq!(ctx.rejection_message("patcher", &native, "later"), None);
}

#[test]
fn a_decision_recorded_under_the_bare_name_still_applies_to_both() {
    // Snapshots written before scoping existed carry the bare name.
    let mut legacy = ApprovalsState::default();
    legacy.0.insert(
        "patcher\u{1}apply_patch".to_string(),
        ApprovalRecord {
            approved_always: true,
            ..Default::default()
        },
    );
    let ctx = RunContext::new(());
    ctx.import_approvals(&legacy);

    for kind in ["apply_patch_call", "custom_tool_call"] {
        let key = scoped_tool_key("apply_patch", Some(kind));
        assert_eq!(
            ctx.is_tool_approved("patcher", &key, "later"),
            Some(true),
            "{kind}"
        );
    }
}

#[test]
fn the_scope_survives_a_serialized_round_trip() {
    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "patcher".into(),
        ..Default::default()
    };
    state.approve(&approval("patcher", "apply_patch_call", "native"), true);

    let restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    let ctx = RunContext::new(());
    ctx.import_approvals(&restored.approvals);

    let native = scoped_tool_key("apply_patch", Some("apply_patch_call"));
    let custom = scoped_tool_key("apply_patch", Some("custom_tool_call"));
    assert_eq!(
        ctx.is_tool_approved("patcher", &native, "later"),
        Some(true)
    );
    assert_eq!(ctx.is_tool_approved("patcher", &custom, "later"), None);
}

#[cfg(feature = "sandbox")]
mod py_sandbox_b_support;

#[cfg(feature = "sandbox")]
mod sandbox_half {
    //! The end-to-end half: the custom sandbox `apply_patch` tool runs through the runner in a
    //! context that already holds a sticky decision about the native tool.

    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use super::py_sandbox_b_support::{ApplyPatchSession, Store};
    use openai_agents_sdk::editor::ApplyPatchOperationType;
    use openai_agents_sdk::items::{OutputItem, RunItemKind};
    use openai_agents_sdk::run::RunOptions;
    use openai_agents_sdk::sandbox::capabilities::SandboxApplyPatchTool;
    use openai_agents_sdk::sandbox::editor::SandboxEditor;
    use openai_agents_sdk::testing::{FakeModel, assistant_message};
    use openai_agents_sdk::tool::Tool;
    use openai_agents_sdk::{Agent, RunContext, RunState, Runner};
    use serde_json::{Value, json};

    use super::approval;

    const RAW: &str = "*** Begin Patch\n*** Add File: harmless.txt\n+harmless\n\
                       *** Delete File: protected.txt\n*** End Patch\n";

    fn round_trip(state: &RunState) -> RunState {
        RunState::from_json_str(&state.to_json_string().unwrap()).unwrap()
    }

    /// Python: `test_multi_operation_checker_ignores_native_sticky_approval`
    /// (`native_approved` x `sandbox_approved` x `sandbox_always` x `round_trip`).
    async fn scenario(
        native_approved: bool,
        sandbox_approved: bool,
        sandbox_always: bool,
        serialize: bool,
    ) {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let session = Arc::new(ApplyPatchSession::new(Store::Sensitive));
        session.put("/workspace/protected.txt", b"protected");
        let checked = Arc::new(Mutex::new(Vec::new()));
        let seen = checked.clone();
        let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session.clone()))
            .with_needs_approval_fn(move |_ctx, operation, _call_id| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(operation.path.clone());
                    operation.op_type == ApplyPatchOperationType::DeleteFile
                }
            });
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![OutputItem::from_value(json!({
            "type": "custom_tool_call", "call_id": "call_apply",
            "name": "apply_patch", "input": RAW
        }))]);
        let agent = Agent::<()>::new("patcher")
            .with_model_instance(model.clone())
            .with_tool(Tool::Custom(Arc::new(tool.into_custom_tool())));

        // A sticky decision about the *native* tool is already in the shared context.
        let context = RunContext::new(());
        let native = approval("patcher", "apply_patch_call", "native_apply");
        if native_approved {
            context.approve_tool(&native, true);
        } else {
            context.reject_tool(&native, true, Some("Native patch denied".into()));
        }

        let first = Runner::run_with_options(&agent, "go", RunOptions::with_run_context(context))
            .await
            .unwrap();

        // The native decision does not decide the custom call: it still asks.
        assert_eq!(first.interruptions.len(), 1);
        assert_eq!(*checked.lock().unwrap(), ["harmless.txt", "protected.txt"]);
        assert_eq!(
            session.files(),
            BTreeMap::from([(
                "/workspace/protected.txt".to_string(),
                b"protected".to_vec()
            )])
        );

        let mut state = first.to_state();
        if serialize {
            state = round_trip(&state);
        }
        let pending = state.get_interruptions()[0].clone();
        if sandbox_approved {
            state.approve(&pending, sandbox_always);
        } else {
            state.reject(
                &pending,
                sandbox_always,
                Some("Sandbox patch denied".into()),
            );
        }
        if serialize {
            state = round_trip(&state);
        }
        model.add_turn(vec![assistant_message("done")]);

        let resumed = Runner::run(&agent, state).await.unwrap();

        let output = resumed.new_items.iter().find_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => raw_item
                .get("output")
                .and_then(Value::as_str)
                .map(str::to_string),
            _ => None,
        });
        if sandbox_approved {
            assert_eq!(
                session.files(),
                BTreeMap::from([("/workspace/harmless.txt".to_string(), b"harmless".to_vec())])
            );
        } else {
            assert_eq!(output.as_deref(), Some("Sandbox patch denied"));
            assert_eq!(
                session.files(),
                BTreeMap::from([(
                    "/workspace/protected.txt".to_string(),
                    b"protected".to_vec()
                )])
            );
        }
    }

    #[tokio::test]
    async fn the_custom_tool_ignores_a_native_sticky_approval() {
        for native_approved in [true, false] {
            for sandbox_approved in [true, false] {
                for sandbox_always in [false, true] {
                    for serialize in [false, true] {
                        scenario(native_approved, sandbox_approved, sandbox_always, serialize)
                            .await;
                    }
                }
            }
        }
    }
}
