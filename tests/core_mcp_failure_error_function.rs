//! `MCPConfig::failure_error_function`: how an MCP tool failure reaches the model.
#![cfg(feature = "mcp")]

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, AgentsError, MCPConfig, Result, RunOptions, Runner};
use serde_json::{Map, Value, json};

/// A server whose only tool always fails.
#[derive(Debug)]
struct FailingServer {
    config: MCPServerConfig,
}

impl FailingServer {
    fn new() -> Self {
        FailingServer {
            config: MCPServerConfig::default(),
        }
    }
}

#[async_trait]
impl MCPServer for FailingServer {
    fn name(&self) -> &str {
        "failing"
    }

    fn config(&self) -> &MCPServerConfig {
        &self.config
    }

    async fn connect(&self) -> Result<()> {
        Ok(())
    }

    async fn cleanup(&self) -> Result<()> {
        Ok(())
    }

    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Ok(vec![MCPTool {
            meta: None,
            name: "explode".into(),
            title: None,
            description: Some("Always fails".into()),
            input_schema: json!({"type": "object", "properties": {}}),
            annotations: None,
        }])
    }

    async fn call_tool(
        &self,
        _tool_name: &str,
        _arguments: Option<Map<String, Value>>,
    ) -> Result<Value> {
        Err(AgentsError::other("upstream exploded"))
    }

    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(Vec::new())
    }

    async fn get_prompt(
        &self,
        _name: &str,
        _arguments: Option<std::collections::BTreeMap<String, String>>,
    ) -> Result<Value> {
        Err(AgentsError::user("no prompts"))
    }
}

fn agent(config: MCPConfig<()>, model: Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new("Ops")
        .with_mcp_server(Arc::new(FailingServer::new()))
        .with_mcp_config(config)
        .with_model_instance(model)
}

#[tokio::test]
async fn the_default_handler_reports_the_failure_to_the_model() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("explode", "{}")]);
    model.add_turn(vec![assistant_message("recovered")]);

    let result = Runner::run(&agent(MCPConfig::default(), model.clone()), "go")
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("recovered"));
    let output = model.last_call().unwrap().input.last().unwrap().clone();
    assert_eq!(
        output.as_value()["output"],
        json!("An error occurred while running the tool. Please try again."),
        "the default message never exposes the upstream error"
    );
}

#[tokio::test]
async fn a_custom_handler_replaces_the_message() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("explode", "{}")]);
    model.add_turn(vec![assistant_message("ok")]);

    let config =
        MCPConfig::<()>::default().with_failure_error_function(|_ctx, _e| "mcp is down".into());
    Runner::run(&agent(config, model.clone()), "go")
        .await
        .unwrap();
    let output = model.last_call().unwrap().input.last().unwrap().clone();
    assert_eq!(output.as_value()["output"], json!("mcp is down"));
}

#[tokio::test]
async fn without_a_handler_the_failure_fails_the_run() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("explode", "{}")]);

    let config = MCPConfig::<()>::default().without_failure_error_function();
    let err = Runner::run_with_options(&agent(config, model), "go", RunOptions::new(()))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("explode"),
        "the error propagates: {err}"
    );
}
