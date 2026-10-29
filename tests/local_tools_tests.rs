//! Local runtime tools (shell, apply patch, computer, custom) and tool namespaces driven by a
//! scripted model.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::editor::WorkspaceEditor;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::models::interface::ToolSpec;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ApprovalDecision, ComputerTool, CustomTool, FunctionTool, ShellCallOutcome,
    ShellCommandOutput, ShellResult, ShellTool, ToolSearchTool, tool_namespace,
};
use openai_agents_sdk::{Agent, RunResult, RunState, Runner};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

fn outputs(r: &RunResult<()>) -> Vec<Value> {
    r.new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

fn shell_call(call_id: &str, commands: &[&str]) -> OutputItem {
    item(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": commands, "max_output_length": 100}
    }))
}

#[tokio::test]
async fn shell_tool_runs_executor_and_sends_structured_output() {
    let model = setup();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    let shell = ShellTool::<()>::new(move |req| {
        let s2 = s2.clone();
        async move {
            s2.lock().unwrap().extend(req.data.action.commands.clone());
            Ok::<_, openai_agents_sdk::AgentsError>(ShellResult {
                output: vec![ShellCommandOutput {
                    stdout: "file.txt\n".into(),
                    stderr: String::new(),
                    outcome: ShellCallOutcome::Exit { exit_code: Some(0) },
                    command: Some("ls".into()),
                    provider_data: None,
                }],
                ..Default::default()
            })
        }
    });
    let agent = Agent::<()>::new("Dev")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![shell_call("call_1", &["ls"])]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "list files").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("done"));
    assert_eq!(*seen.lock().unwrap(), vec!["ls".to_string()]);
    let out = &outputs(&r)[0];
    assert_eq!(out["type"], "shell_call_output");
    assert_eq!(out["call_id"], "call_1");
    assert_eq!(out["status"], "completed");
    assert_eq!(out["max_output_length"], 100);
    assert_eq!(
        out["output"],
        json!([{"stdout": "file.txt\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}])
    );
    // The tool param and the output reach the next model call.
    let calls = model.calls();
    assert!(matches!(&calls[0].tools[0], ToolSpec::Hosted { param, .. }
        if *param == json!({"type": "shell", "environment": {"type": "local"}})));
    let second_input: Vec<Value> = calls[1].input.iter().map(|i| i.0.clone()).collect();
    assert!(
        second_input
            .iter()
            .any(|i| i["type"] == "shell_call_output" && i["call_id"] == "call_1")
    );
}

#[tokio::test]
async fn shell_approval_interrupts_then_rejects_with_message() {
    let model = setup();
    let ran = Arc::new(Mutex::new(0));
    let r2c = ran.clone();
    let shell = ShellTool::<()>::new(move |_req| {
        let r2c = r2c.clone();
        async move {
            *r2c.lock().unwrap() += 1;
            Ok::<_, openai_agents_sdk::AgentsError>("ok")
        }
    })
    .with_needs_approval_fn(|_ctx, action, _id| async move {
        action.commands.iter().any(|c| c.starts_with("rm"))
    });
    let agent = Agent::<()>::new("Dev")
        .with_tool(shell)
        .with_model_instance(model.clone());
    model.add_turn(vec![shell_call("call_rm", &["rm -rf /tmp/x"])]);
    let r = Runner::run(&agent, "clean").await.unwrap();
    assert_eq!(r.interruptions.len(), 1);
    assert_eq!(r.interruptions[0].name(), "shell");
    assert_eq!(*ran.lock().unwrap(), 0);

    let json = r.to_state().to_json_string().unwrap();
    let mut state = RunState::from_json_str(&json).unwrap();
    let pending = state.get_interruptions()[0].clone();
    state.reject(&pending, false, Some("not allowed".into()));
    model.add_turn(vec![assistant_message("ok, skipped")]);
    let r2 = Runner::run(&agent, state).await.unwrap();
    assert_eq!(r2.final_output_text().as_deref(), Some("ok, skipped"));
    assert_eq!(*ran.lock().unwrap(), 0);
    let out = outputs(&r2)
        .into_iter()
        .find(|o| o["type"] == "shell_call_output")
        .unwrap();
    assert_eq!(out["output"][0]["stderr"], "not allowed");
    assert_eq!(out["output"][0]["outcome"]["exit_code"], 1);

    // An approved call runs.
    model.add_turn(vec![shell_call("call_rm2", &["rm a"])]);
    let r3 = Runner::run(&agent, "clean again").await.unwrap();
    let mut state = r3.to_state();
    let pending = state.get_interruptions()[0].clone();
    state.approve(&pending, false);
    model.add_turn(vec![assistant_message("cleaned")]);
    let r4 = Runner::run(&agent, state).await.unwrap();
    assert_eq!(r4.final_output_text().as_deref(), Some("cleaned"));
    assert_eq!(*ran.lock().unwrap(), 1);
}

#[tokio::test]
async fn apply_patch_tool_edits_workspace() {
    let model = setup();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("main.txt"), "alpha\nbeta\n").unwrap();
    let agent = Agent::<()>::new("Editor")
        .with_tool(ApplyPatchTool::<()>::new(WorkspaceEditor::new(dir.path())))
        .with_model_instance(model.clone());
    model.add_turn(vec![item(json!({
        "type": "apply_patch_call", "call_id": "call_p", "status": "completed",
        "operation": {"type": "update_file", "path": "main.txt", "diff": "@@\n alpha\n-beta\n+gamma\n"}
    }))]);
    model.add_turn(vec![item(json!({
        "type": "apply_patch_call", "call_id": "call_bad", "status": "completed",
        "operation": {"type": "update_file", "path": "missing.txt", "diff": "@@\n-x\n+y\n"}
    }))]);
    model.add_turn(vec![assistant_message("patched")]);
    let r = Runner::run(&agent, "rename beta").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("patched"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("main.txt")).unwrap(),
        "alpha\ngamma\n"
    );
    let outs = outputs(&r);
    assert_eq!(
        outs[0],
        json!({"type": "apply_patch_call_output", "call_id": "call_p", "status": "completed",
               "output": "Updated main.txt"})
    );
    assert_eq!(outs[1]["status"], "failed");
}

#[derive(Default)]
struct FakeComputer {
    log: Mutex<Vec<String>>,
}

#[async_trait]
impl Computer for FakeComputer {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Browser)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((1024, 768))
    }
    async fn screenshot(&self) -> openai_agents_sdk::Result<String> {
        self.log.lock().unwrap().push("screenshot".into());
        Ok("aW1n".into())
    }
    async fn click(
        &self,
        x: i64,
        y: i64,
        button: &str,
        _keys: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("click {x} {y} {button}"));
        Ok(())
    }
    async fn double_click(
        &self,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn scroll(
        &self,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn type_text(&self, text: &str) -> openai_agents_sdk::Result<()> {
        self.log.lock().unwrap().push(format!("type {text}"));
        Ok(())
    }
    async fn wait(&self) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn move_mouse(
        &self,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn keypress(&self, _: &[String]) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn drag(&self, _: &[(i64, i64)], _: Option<&[String]>) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn computer_tool_executes_batched_actions_and_acknowledges_checks() {
    let model = setup();
    let computer = Arc::new(FakeComputer::default());
    let tool =
        ComputerTool::<()>::from_arc(computer.clone()).with_on_safety_check(|data| async move {
            data.safety_check["code"] == "malicious_instructions"
        });
    let agent = Agent::<()>::new("Browser")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(json!({
        "type": "computer_call", "id": "cu_1", "call_id": "call_c", "status": "completed",
        "actions": [{"type": "click", "x": 10, "y": 20, "button": "left"}, {"type": "type", "text": "hi"}],
        "pending_safety_checks": [{"id": "sc_1", "code": "malicious_instructions", "message": "careful"}]
    }))]);
    model.add_turn(vec![assistant_message("clicked")]);
    let r = Runner::run(&agent, "click it").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("clicked"));
    assert_eq!(
        *computer.log.lock().unwrap(),
        vec!["click 10 20 left", "type hi", "screenshot"]
    );
    let out = &outputs(&r)[0];
    assert_eq!(out["type"], "computer_call_output");
    assert_eq!(out["output"]["image_url"], "data:image/png;base64,aW1n");
    assert_eq!(out["acknowledged_safety_checks"][0]["id"], "sc_1");
    let first = &model.calls()[0];
    assert!(matches!(&first.tools[0], ToolSpec::Hosted { param, .. }
        if param["type"] == "computer" && param["display_width"] == 1024));

    // A rejected safety check aborts the run.
    model.add_turn(vec![item(json!({
        "type": "computer_call", "call_id": "call_d", "action": {"type": "wait"},
        "pending_safety_checks": [{"id": "sc_2", "code": "other", "message": "no"}]
    }))]);
    let err = Runner::run(&agent, "again").await.unwrap_err();
    assert!(err.to_string().contains("safety check"));
}

#[tokio::test]
async fn custom_tool_with_auto_approval() {
    let model = setup();
    let tool = CustomTool::<()>::new("sql", "Run SQL", |_ctx, input| async move {
        Ok(format!("rows for: {input}"))
    })
    .with_format(json!({"type": "text"}))
    .with_needs_approval(true)
    .with_on_approval(|_ctx, item| async move {
        Some(if item.raw_item.arguments.starts_with("SELECT") {
            ApprovalDecision::approve()
        } else {
            ApprovalDecision::reject(Some("read only".into()))
        })
    });
    let agent = Agent::<()>::new("DB")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![
        item(json!({"type": "custom_tool_call", "call_id": "c1", "name": "sql", "input": "SELECT 1"})),
        item(json!({"type": "custom_tool_call", "call_id": "c2", "name": "sql", "input": "DROP t"})),
    ]);
    model.add_turn(vec![assistant_message("ok")]);
    let r = Runner::run(&agent, "query").await.unwrap();
    assert!(r.interruptions.is_empty());
    let outs = outputs(&r);
    assert_eq!(
        outs[0],
        json!({"type": "custom_tool_call_output", "call_id": "c1", "output": "rows for: SELECT 1"})
    );
    assert_eq!(outs[1]["output"], "read only");
    assert!(
        matches!(&model.calls()[0].tools[0], ToolSpec::Hosted { param, .. }
        if *param == json!({"type": "custom", "name": "sql", "description": "Run SQL", "format": {"type": "text"}}))
    );
}

#[tokio::test]
async fn namespaced_function_tools_dispatch_by_qualified_name() {
    let model = setup();
    let crm = tool_namespace(
        "crm",
        "CRM tools",
        vec![FunctionTool::<()>::new_no_args(
            "lookup",
            "Look up",
            |_c| async { Ok("crm hit") },
        )],
    )
    .unwrap();
    let billing = tool_namespace(
        "billing",
        "Billing tools",
        vec![FunctionTool::<()>::new_no_args(
            "lookup",
            "Look up",
            |_c| async { Ok("billing hit") },
        )],
    )
    .unwrap();
    let agent = Agent::<()>::new("Ops")
        .with_tools(crm.into_iter().chain(billing))
        .with_tool(ToolSearchTool::new())
        .with_model_instance(model.clone());
    model.add_turn(vec![item(json!({
        "type": "function_call", "call_id": "f1", "name": "lookup", "namespace": "billing", "arguments": "{}"
    }))]);
    model.add_turn(vec![assistant_message("found")]);
    let r = Runner::run(&agent, "find").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("found"));
    assert_eq!(outputs(&r)[0]["output"], "billing hit");
    assert!(tool_namespace::<()>("", "d", vec![]).is_err());
}
