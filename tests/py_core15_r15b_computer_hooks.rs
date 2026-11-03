//! Port of `test_runner_preserves_concrete_computer_tool_identity_for_hooks` from the Python SDK's
//! `tests/test_computer_tool_lifecycle.py`: lifecycle hooks see a computer call's own tool (the
//! computer tool), not a stand-in.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::computer::Computer;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{ComputerTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, Result, Runner};
use serde_json::json;

#[derive(Clone)]
struct Fake;

#[async_trait]
impl Computer for Fake {
    async fn screenshot(&self) -> Result<String> {
        Ok("aW1n".into())
    }
    async fn click(&self, _: i64, _: i64, _: &str, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn double_click(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn scroll(&self, _: i64, _: i64, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn type_text(&self, _: &str) -> Result<()> {
        Ok(())
    }
    async fn wait(&self) -> Result<()> {
        Ok(())
    }
    async fn move_mouse(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn keypress(&self, _: &[String]) -> Result<()> {
        Ok(())
    }
    async fn drag(&self, _: &[(i64, i64)], _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct IdentityHooks {
    started: Mutex<Vec<(bool, String)>>,
    ended: Mutex<Vec<(bool, String)>>,
}

#[async_trait]
impl RunHooks<()> for IdentityHooks {
    async fn on_tool_start(&self, _c: &ToolContext<()>, _a: &Agent<()>, tool: &Tool<()>) {
        self.started
            .lock()
            .unwrap()
            .push((matches!(tool, Tool::Computer(_)), tool.name().to_string()));
    }

    async fn on_tool_end(
        &self,
        _c: &ToolContext<()>,
        _a: &Agent<()>,
        tool: &Tool<()>,
        _result: &ToolOutput,
    ) {
        self.ended
            .lock()
            .unwrap()
            .push((matches!(tool, Tool::Computer(_)), tool.name().to_string()));
    }
}

#[tokio::test]
async fn hooks_see_the_concrete_computer_tool() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![OutputItem::from_value(json!({
        "id": "computer-call", "type": "computer_call", "action": {"type": "screenshot"},
        "call_id": "computer-call", "status": "completed", "pending_safety_checks": [],
    }))]);
    model.add_turn(vec![assistant_message("done")]);
    let tool = ComputerTool::new(Fake);
    let expected_name = Tool::from(tool.clone()).name().to_string();
    let agent = Agent::<()>::new("ComputerAgent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let hooks = Arc::new(IdentityHooks::default());

    let result =
        Runner::run_with_options(&agent, "hello", RunOptions::new(()).hooks(hooks.clone()))
            .await
            .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(model.calls().len(), 2);
    assert_eq!(
        *hooks.started.lock().unwrap(),
        vec![(true, expected_name.clone())]
    );
    assert_eq!(*hooks.ended.lock().unwrap(), vec![(true, expected_name)]);
}
