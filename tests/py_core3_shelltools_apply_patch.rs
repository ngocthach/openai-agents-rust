//! Port of the Python SDK's `tests/test_apply_patch_tool.py` (agent and tool layer only).
//!
//! Each case runs through `Runner::run`: the scripted model emits the `apply_patch_call`, and
//! the assertions read the recorded `apply_patch_call_output` and the input replayed to the
//! model. Span assertions live in `py_core3_shelltools_tool_spans.rs`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::editor::{
    ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType, ApplyPatchResult,
    ApplyPatchStatus,
};
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ApprovalDecision, DEFAULT_APPROVAL_REJECTION_MESSAGE,
};
use openai_agents_sdk::{Agent, AgentsError, Result, RunResult, Runner};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

/// Records every operation it is asked to apply (Python's `RecordingEditor`).
#[derive(Clone, Default)]
struct RecordingEditor {
    operations: Arc<Mutex<Vec<(ApplyPatchOperationType, String)>>>,
    fail_update: Option<String>,
    failed_update_output: Option<String>,
}

impl RecordingEditor {
    fn record(&self, op: &ApplyPatchOperation) {
        self.operations
            .lock()
            .unwrap()
            .push((op.op_type, op.path.clone()));
    }

    fn operations(&self) -> Vec<(ApplyPatchOperationType, String)> {
        self.operations.lock().unwrap().clone()
    }
}

#[async_trait]
impl ApplyPatchEditor for RecordingEditor {
    async fn create_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        self.record(op);
        Ok(ApplyPatchResult {
            status: None,
            output: Some(format!("Created {}", op.path)),
        })
    }

    async fn update_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        self.record(op);
        if let Some(message) = &self.fail_update {
            return Err(AgentsError::user(message.clone()));
        }
        if let Some(output) = &self.failed_update_output {
            return Ok(ApplyPatchResult {
                status: Some(ApplyPatchStatus::Failed),
                output: Some(output.clone()),
            });
        }
        Ok(ApplyPatchResult {
            status: Some(ApplyPatchStatus::Completed),
            output: Some(format!("Updated {}", op.path)),
        })
    }

    async fn delete_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        self.record(op);
        Ok(ApplyPatchResult {
            status: None,
            output: Some(format!("Deleted {}", op.path)),
        })
    }
}

fn update_op(path: &str) -> Value {
    json!({"type": "update_file", "path": path, "diff": "-a\n+b\n"})
}

/// The `apply_patch_call` payload the model emits (Python's `_call`).
fn apply_patch_call(call_id: &str, operation: Value) -> Value {
    json!({"type": "apply_patch_call", "call_id": call_id, "operation": operation})
}

fn patch_outputs(r: &RunResult<()>) -> Vec<Value> {
    r.new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

fn tool_output_text(r: &RunResult<()>) -> Option<String> {
    r.new_items.iter().find_map(|i| match &i.kind {
        RunItemKind::ToolCallOutput { output, .. } => output.as_str().map(str::to_string),
        _ => None,
    })
}

fn apply_patch_output(r: &RunResult<()>) -> Value {
    patch_outputs(r)
        .into_iter()
        .find(|o| o["type"] == "apply_patch_call_output")
        .expect("apply_patch_call_output recorded")
}

/// Runs one `apply_patch_call` through `tool` and returns the run result and the model.
async fn run_patch(tool: ApplyPatchTool<()>, call: Value) -> (RunResult<()>, Arc<FakeModel>) {
    let model = setup();
    let agent = Agent::<()>::new("patcher")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(call)]);
    model.add_turn(vec![assistant_message("done")]);
    let r = Runner::run(&agent, "run").await.unwrap();
    (r, model)
}

#[tokio::test]
async fn apply_patch_tool_success() {
    let editor = RecordingEditor::default();
    let tool = ApplyPatchTool::new(editor.clone());

    let (r, model) = run_patch(tool, apply_patch_call("call_apply", update_op("tasks.md"))).await;

    assert!(tool_output_text(&r).unwrap().contains("Updated tasks.md"));
    let raw = apply_patch_output(&r);
    assert_eq!(raw["status"], "completed");
    assert_eq!(raw["call_id"], "call_apply");
    assert!(
        raw["output"]
            .as_str()
            .unwrap()
            .starts_with("Updated tasks.md")
    );
    assert_eq!(
        editor.operations()[0].0,
        ApplyPatchOperationType::UpdateFile
    );

    // The replayed item keeps the type and status the model needs.
    let replayed: Vec<Value> = model.calls()[1].input.iter().map(|i| i.0.clone()).collect();
    let payload = replayed
        .iter()
        .find(|i| i["type"] == "apply_patch_call_output")
        .expect("replayed output");
    assert_eq!(payload["status"], "completed");
}

#[tokio::test]
async fn apply_patch_tool_failure() {
    let editor = RecordingEditor {
        fail_update: Some("boom".into()),
        ..Default::default()
    };
    let tool = ApplyPatchTool::new(editor);

    let (r, model) = run_patch(
        tool,
        apply_patch_call("call_apply_fail", update_op("tasks.md")),
    )
    .await;

    assert!(tool_output_text(&r).unwrap().contains("boom"));
    let raw = apply_patch_output(&r);
    assert_eq!(raw["status"], "failed");
    assert!(raw["output"].is_string());
    let replayed: Vec<Value> = model.calls()[1].input.iter().map(|i| i.0.clone()).collect();
    let payload = replayed
        .iter()
        .find(|i| i["type"] == "apply_patch_call_output")
        .unwrap();
    assert_eq!(payload["status"], "failed");
}

#[tokio::test]
async fn apply_patch_tool_accepts_mapping_call() {
    // Python distinguishes a dataclass call from a mapping call; the wire shape is the same
    // JSON object in Rust, so only the create-file mapping case is asserted here.
    let editor = RecordingEditor::default();
    let tool = ApplyPatchTool::new(editor.clone());
    let call = apply_patch_call(
        "call_mapping",
        json!({"type": "create_file", "path": "notes.md", "diff": "+hello\n"}),
    );

    let (r, _model) = run_patch(tool, call).await;

    let raw = apply_patch_output(&r);
    assert_eq!(raw["call_id"], "call_mapping");
    assert_eq!(editor.operations()[0].1, "notes.md");
}

#[tokio::test]
async fn apply_patch_tool_needs_approval_returns_approval_item() {
    let editor = RecordingEditor::default();
    let tool = ApplyPatchTool::new(editor.clone()).with_needs_approval(true);
    let model = setup();
    let agent = Agent::<()>::new("patcher")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(apply_patch_call(
        "call_apply",
        update_op("tasks.md"),
    ))]);

    let r = Runner::run(&agent, "run").await.unwrap();

    assert_eq!(r.interruptions.len(), 1);
    assert_eq!(r.interruptions[0].name(), "apply_patch");
    assert!(editor.operations().is_empty());
}

/// Runs to an approval interruption, then returns the paused state for the caller to decide.
async fn paused_patch_run(
    tool: ApplyPatchTool<()>,
) -> (Arc<FakeModel>, Agent<()>, openai_agents_sdk::RunState) {
    let model = setup();
    let agent = Agent::<()>::new("patcher")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(apply_patch_call(
        "call_apply",
        update_op("tasks.md"),
    ))]);
    let first = Runner::run(&agent, "run").await.unwrap();
    (model, agent, first.to_state())
}

#[tokio::test]
async fn apply_patch_tool_needs_approval_rejected_returns_rejection() {
    let editor = RecordingEditor::default();
    let tool = ApplyPatchTool::new(editor.clone()).with_needs_approval(true);
    let (model, agent, mut state) = paused_patch_run(tool).await;
    let pending = state.get_interruptions()[0].clone();
    state.reject(&pending, false, None);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, state).await.unwrap();

    assert_eq!(
        tool_output_text(&r).as_deref(),
        Some(DEFAULT_APPROVAL_REJECTION_MESSAGE)
    );
    let raw = apply_patch_output(&r);
    assert_eq!(raw["type"], "apply_patch_call_output");
    assert_eq!(raw["status"], "failed");
    assert_eq!(raw["output"], DEFAULT_APPROVAL_REJECTION_MESSAGE);
    assert!(editor.operations().is_empty());
}

#[tokio::test]
async fn apply_patch_rejection_uses_run_level_formatter() {
    let editor = RecordingEditor::default();
    let tool = ApplyPatchTool::new(editor.clone()).with_needs_approval(true);
    let (model, agent, mut state) = paused_patch_run(tool).await;
    let pending = state.get_interruptions()[0].clone();
    state.reject(&pending, false, None);
    model.add_turn(vec![assistant_message("done")]);
    let config = RunConfig::<()>::default().with_tool_error_formatter(|args| {
        Some(format!("{} denied ({})", args.tool_name, args.call_id))
    });

    let r = Runner::run_with_options(&agent, state, RunOptions::<()>::new(()).run_config(config))
        .await
        .unwrap();

    assert_eq!(
        tool_output_text(&r).as_deref(),
        Some("apply_patch denied (call_apply)")
    );
    let raw = apply_patch_output(&r);
    assert_eq!(raw["output"], "apply_patch denied (call_apply)");
}

#[tokio::test]
async fn apply_patch_tool_on_approval_callback_auto_approves() {
    let editor = RecordingEditor::default();
    let tool = ApplyPatchTool::new(editor.clone())
        .with_needs_approval(true)
        .with_on_approval(|_ctx, _item| async { Some(ApprovalDecision::approve()) });
    let model = setup();
    let agent = Agent::<()>::new("patcher")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(apply_patch_call(
        "call_apply",
        update_op("tasks.md"),
    ))]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "run").await.unwrap();

    assert!(tool_output_text(&r).unwrap().contains("Updated tasks.md"));
    assert_eq!(editor.operations().len(), 1);
}

#[tokio::test]
async fn apply_patch_tool_on_approval_callback_auto_rejects() {
    let editor = RecordingEditor::default();
    let tool = ApplyPatchTool::new(editor.clone())
        .with_needs_approval(true)
        .with_on_approval(|_ctx, _item| async {
            Some(ApprovalDecision::reject(Some("Not allowed".into())))
        });
    let model = setup();
    let agent = Agent::<()>::new("patcher")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(apply_patch_call(
        "call_apply",
        update_op("tasks.md"),
    ))]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "run").await.unwrap();

    assert_eq!(tool_output_text(&r).as_deref(), Some("Not allowed"));
    assert_eq!(apply_patch_output(&r)["output"], "Not allowed");
    assert!(
        editor.operations().is_empty(),
        "rejected patch must not execute"
    );
}

#[tokio::test]
async fn apply_patch_failed_status_not_overwritten_by_later_completed_op() {
    // The first operation reports `failed`; the second succeeds. The overall status stays
    // `failed` and both outputs are kept.
    let editor = RecordingEditor {
        failed_update_output: Some("Failed a.md".into()),
        ..Default::default()
    };
    let tool = ApplyPatchTool::new(editor.clone());
    let call = json!({
        "type": "apply_patch_call",
        "call_id": "call_multi",
        "operations": [
            {"type": "update_file", "path": "a.md", "diff": "-x\n+y\n"},
            {"type": "create_file", "path": "b.md", "diff": "+hi\n"},
        ],
    });

    let (r, _model) = run_patch(tool, call).await;

    let raw = apply_patch_output(&r);
    assert_eq!(raw["status"], "failed");
    let text = tool_output_text(&r).unwrap();
    assert!(text.contains("Failed a.md"), "{text}");
    assert!(text.contains("Created b.md"), "{text}");
    assert_eq!(editor.operations().len(), 2);
}
