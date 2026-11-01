//! Port of the prompt-failure half of `test_start_hook_replaces_computer_before_initialization`
//! (`tests/test_computer_tool_lifecycle.py`): a computer resolved for the model request is still
//! released when the agent's instructions fail before the model is called.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::ComputerTool;
use openai_agents_sdk::{Agent, AgentsError, Result, Runner};

struct Quiet;

#[async_trait]
impl Computer for Quiet {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Mac)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((1, 1))
    }
    async fn screenshot(&self) -> Result<String> {
        Ok("img".into())
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

async fn run(fail_prompt: bool) -> (Result<String>, Vec<&'static str>, usize) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let events: Arc<Mutex<Vec<&'static str>>> = Arc::default();
    let (created, disposed) = (events.clone(), events.clone());
    let tool = ComputerTool::<()>::from_provider(
        move || {
            let created = created.clone();
            async move {
                created.lock().unwrap().push("create");
                let computer: Arc<dyn Computer> = Arc::new(Quiet);
                Ok(computer)
            }
        },
        move |_computer| {
            let disposed = disposed.clone();
            async move {
                disposed.lock().unwrap().push("dispose");
            }
        },
    );
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("computer")
        .with_tool(tool)
        .with_model_instance(model.clone())
        .with_dynamic_instructions(move |_ctx, _agent| async move {
            if fail_prompt {
                Err(AgentsError::other("prompt failed"))
            } else {
                Ok("Use the computer.".to_string())
            }
        });

    let outcome = Runner::run(&agent, "go")
        .await
        .map(|result| result.final_output_text().unwrap_or_default());
    let seen = events.lock().unwrap().clone();
    (outcome, seen, model.calls().len())
}

#[tokio::test]
async fn a_failing_prompt_leaves_no_computer_unreleased() {
    let (ok, events, calls) = run(false).await;
    assert_eq!(ok.unwrap(), "done");
    assert_eq!(events, vec!["create", "dispose"]);
    assert_eq!(calls, 1);

    let (failed, events, calls) = run(true).await;
    assert!(failed.unwrap_err().to_string().contains("prompt failed"));
    assert_eq!(calls, 0, "the model is never called");
    assert_eq!(
        events.iter().filter(|e| **e == "create").count(),
        events.iter().filter(|e| **e == "dispose").count(),
        "every created computer is disposed: {events:?}"
    );
}
