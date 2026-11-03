//! Per-tool `custom_data_extractor` on the computer, apply-patch and custom tools, and the
//! typed shell environment.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::editor::{ApplyPatchStatus, WorkspaceEditor};
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ComputerTool, CustomTool, ShellTool, ShellToolContainerNetworkPolicy,
    ShellToolEnvironment,
};
use openai_agents_sdk::{Agent, RunResult, Runner};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

/// The custom data attached to each tool-output item of a run.
fn custom_data(r: &RunResult<()>) -> Vec<Option<Value>> {
    r.new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { custom_data, .. } => Some(custom_data.clone()),
            _ => None,
        })
        .collect()
}

#[derive(Default)]
struct StubComputer;

#[async_trait]
impl Computer for StubComputer {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Browser)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((800, 600))
    }
    async fn screenshot(&self) -> openai_agents_sdk::Result<String> {
        Ok("aW1n".into())
    }
    async fn wait(&self) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn click(
        &self,
        _x: i64,
        _y: i64,
        _button: &str,
        _keys: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn double_click(
        &self,
        _x: i64,
        _y: i64,
        _keys: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn scroll(
        &self,
        _x: i64,
        _y: i64,
        _scroll_x: i64,
        _scroll_y: i64,
        _keys: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn type_text(&self, _text: &str) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn move_mouse(
        &self,
        _x: i64,
        _y: i64,
        _keys: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn keypress(&self, _keys: &[String]) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn drag(
        &self,
        _path: &[(i64, i64)],
        _keys: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn computer_tool_custom_data_reaches_the_output_item() {
    let model = setup();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    let tool = ComputerTool::<()>::new(StubComputer).with_custom_data_extractor(move |ctx| {
        // The extractor sees the model's call, the screenshot URL and the raw output item.
        s2.lock().unwrap().push((
            ctx.tool_call["call_id"].clone(),
            ctx.output.to_string(),
            ctx.raw_item["type"].clone(),
        ));
        async move { Some(json!({"source": "computer"})) }
    });
    let agent = Agent::<()>::new("Browser")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(json!({
        "type": "computer_call", "call_id": "call_c", "action": {"type": "wait"}
    }))]);
    model.add_turn(vec![assistant_message("waited")]);

    let r = Runner::run(&agent, "wait").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("waited"));
    assert_eq!(custom_data(&r), vec![Some(json!({"source": "computer"}))]);
    let recorded = seen.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].0, json!("call_c"));
    assert_eq!(recorded[0].1, "data:image/png;base64,aW1n");
    assert_eq!(recorded[0].2, json!("computer_call_output"));
}

#[tokio::test]
async fn apply_patch_custom_data_carries_the_operations_and_status() {
    let model = setup();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("main.txt"), "alpha\nbeta\n").unwrap();
    let tool = ApplyPatchTool::<()>::new(WorkspaceEditor::new(dir.path()))
        .with_custom_data_extractor(|ctx| {
            let paths: Vec<String> = ctx.operations.iter().map(|o| o.path.clone()).collect();
            let failed = ctx.status == ApplyPatchStatus::Failed;
            async move { Some(json!({"paths": paths, "failed": failed})) }
        });
    let agent = Agent::<()>::new("Editor")
        .with_tool(tool)
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

    let r = Runner::run(&agent, "patch").await.unwrap();
    assert_eq!(
        custom_data(&r),
        vec![
            Some(json!({"paths": ["main.txt"], "failed": false})),
            Some(json!({"paths": ["missing.txt"], "failed": true})),
        ]
    );
}

#[tokio::test]
async fn custom_tool_custom_data_sees_the_raw_input_and_output() {
    let model = setup();
    let tool = CustomTool::<()>::new("sql", "Run SQL", |_ctx, input| async move {
        Ok(format!("rows for: {input}"))
    })
    .with_custom_data_extractor(|ctx| {
        let input = ctx.input.to_string();
        let output = ctx.output.to_string();
        let tool_name = ctx.tool_context.tool_name.clone();
        async move { Some(json!({"input": input, "output": output, "tool": tool_name})) }
    });
    let agent = Agent::<()>::new("Analyst")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(
        json!({"type": "custom_tool_call", "call_id": "c1", "name": "sql", "input": "SELECT 1"}),
    )]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "query").await.unwrap();
    assert_eq!(
        custom_data(&r),
        vec![Some(json!({
            "input": "SELECT 1",
            "output": "rows for: SELECT 1",
            "tool": "sql",
        }))]
    );
}

#[tokio::test]
async fn an_extractor_returning_none_leaves_the_item_without_custom_data() {
    let model = setup();
    let tool = CustomTool::<()>::new("noop", "No data", |_ctx, _input| async move {
        Ok("ok".to_string())
    })
    // An empty object normalizes to no custom data, exactly like `None`.
    .with_custom_data_extractor(|_ctx| async move { Some(json!({})) });
    let agent = Agent::<()>::new("Analyst")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(
        json!({"type": "custom_tool_call", "call_id": "c1", "name": "noop", "input": "x"}),
    )]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "go").await.unwrap();
    assert_eq!(custom_data(&r), vec![None]);
}

#[tokio::test]
async fn a_non_object_extractor_result_is_a_user_error() {
    let model = setup();
    let tool = CustomTool::<()>::new(
        "bad",
        "Bad",
        |_ctx, _input| async move { Ok("ok".to_string()) },
    )
    .with_custom_data_extractor(|_ctx| async move { Some(json!("not an object")) });
    let agent = Agent::<()>::new("Analyst")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(
        json!({"type": "custom_tool_call", "call_id": "c1", "name": "bad", "input": "x"}),
    )]);

    let err = Runner::run(&agent, "go").await.unwrap_err();
    assert!(err.to_string().contains("custom_data_extractor"), "{err}");
}

#[test]
fn the_typed_shell_environment_builds_a_hosted_tool() {
    let local = ShellToolEnvironment::Local { skills: None };
    assert!(!local.is_hosted());
    // A local environment has no executor, so it cannot back a hosted tool.
    assert!(ShellTool::<()>::hosted_typed(local).is_err());

    let hosted = ShellToolEnvironment::ContainerAuto {
        file_ids: None,
        memory_limit: Some("4g".into()),
        network_policy: Some(ShellToolContainerNetworkPolicy::Disabled),
        skills: None,
    };
    assert!(hosted.is_hosted());
    let tool = ShellTool::<()>::hosted_typed(hosted).unwrap();
    assert!(!tool.is_local());
    assert_eq!(
        tool.environment,
        json!({
            "type": "container_auto",
            "memory_limit": "4g",
            "network_policy": {"type": "disabled"},
        })
    );
}
