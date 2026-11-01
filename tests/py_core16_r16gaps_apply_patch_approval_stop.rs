//! Port of `test_multi_operation_checker_stops_when_approval_resolves`
//! (`tests/sandbox/capabilities/test_apply_patch_tool.py`): once a decision for the call lands
//! while the approval checker evaluates one operation, the remaining operations are not
//! checked and the decision governs the whole call. The decision lands from inside the checker,
//! the Rust counterpart of Python's concurrent approval while an awaited callback runs.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_b_support/mod.rs"]
mod py_sandbox_b_support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{FunctionCall, OutputItem, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::sandbox::capabilities::SandboxApplyPatchTool;
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::Tool;
use openai_agents_sdk::{Agent, RunContext, Runner};
use py_sandbox_b_support::{ApplyPatchSession, Store};
use serde_json::json;

const RAW: &str = "*** Begin Patch\n*** Add File: first.txt\n+first\n\
                   *** Add File: second.txt\n+second\n*** End Patch\n";

fn custom_call_item() -> ToolApprovalItem {
    let mut call = FunctionCall::new("apply_patch", RAW);
    call.call_id = "call_apply".to_string();
    call.extra.insert("type".into(), json!("custom_tool_call"));
    ToolApprovalItem {
        agent: "patcher".into(),
        raw_item: call,
        tool_name: "apply_patch".into(),
        tool_origin: None,
    }
}

async fn scenario(approved: bool) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let session = Arc::new(ApplyPatchSession::new(Store::Sensitive));
    let checked: Arc<Mutex<Vec<String>>> = Arc::default();
    let seen = checked.clone();
    let tool = SandboxApplyPatchTool::<()>::new(SandboxEditor::new(session.clone()))
        .with_needs_approval_fn(move |ctx, operation, _call_id| {
            let seen = seen.clone();
            async move {
                let mut paths = seen.lock().unwrap();
                paths.push(operation.path.clone());
                assert!(
                    paths.len() == 1,
                    "a resolved approval must stop later callbacks"
                );
                // The decision lands while the first operation is being checked.
                if approved {
                    ctx.approve_tool(&custom_call_item(), false);
                } else {
                    ctx.reject_tool(&custom_call_item(), false, None);
                }
                false
            }
        });
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![OutputItem::from_value(json!({
        "type": "custom_tool_call", "call_id": "call_apply",
        "name": "apply_patch", "input": RAW
    }))]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("patcher")
        .with_model_instance(model)
        .with_tool(Tool::Custom(Arc::new(tool.into_custom_tool())));

    let result = Runner::run_with_options(
        &agent,
        "go",
        RunOptions::with_run_context(RunContext::new(())),
    )
    .await
    .unwrap();

    assert_eq!(*checked.lock().unwrap(), ["first.txt"]);
    assert!(
        result
            .new_items
            .iter()
            .any(|item| matches!(item.kind, RunItemKind::ToolCallOutput { .. }))
    );
    if approved {
        assert_eq!(
            session.files(),
            BTreeMap::from([
                ("/workspace/first.txt".to_string(), b"first".to_vec()),
                ("/workspace/second.txt".to_string(), b"second".to_vec()),
            ])
        );
    } else {
        assert!(session.files().is_empty());
    }
}

#[tokio::test]
async fn the_checker_stops_when_an_approval_lands() {
    scenario(true).await;
}

#[tokio::test]
async fn the_checker_stops_when_a_rejection_lands() {
    scenario(false).await;
}
