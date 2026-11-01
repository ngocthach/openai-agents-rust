#![cfg(feature = "mcp")]
//! Port of the `MCPToolCancellationError` cases of `tests/mcp/test_mcp_util.py`: a tool call the
//! MCP client cancelled itself becomes `AgentsError::MCPToolCancellationError` naming the tool
//! and the credential-free server, the default failure policy reports it to the model, and the
//! run's `tool_error_formatter` is never consulted for it.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::error::Result;
use openai_agents_sdk::mcp::util::{invoke_mcp_tool, to_function_tool};
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::{Agent, AgentsError, RunConfig, RunItemKind, Runner};
use serde_json::{Map, Value, json};

const CREDENTIALED: &str =
    "sse: https://user:s3cr3t_pw@mcp.example.com/sse?api_key=SECRET_QS_KEY#SECRET_FRAGMENT";

#[derive(Debug)]
struct CancelledServer {
    name: String,
    config: MCPServerConfig,
}

impl CancelledServer {
    fn named(name: &str) -> Arc<dyn MCPServer> {
        Arc::new(Self {
            name: name.into(),
            config: MCPServerConfig::default(),
        })
    }
}

#[async_trait::async_trait]
impl MCPServer for CancelledServer {
    fn name(&self) -> &str {
        &self.name
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
        Ok(vec![tool()])
    }
    async fn call_tool(&self, _name: &str, _args: Option<Map<String, Value>>) -> Result<Value> {
        // The client cancelled the call itself; the message is rebuilt by the invoker.
        Err(AgentsError::mcp_tool_cancelled("synthetic mcp cancel"))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(Vec::new())
    }
    async fn get_prompt(
        &self,
        _name: &str,
        _arguments: Option<std::collections::BTreeMap<String, String>>,
    ) -> Result<Value> {
        Ok(Value::Null)
    }
}

fn tool() -> MCPTool {
    MCPTool {
        meta: None,
        name: "cancel_tool".into(),
        title: None,
        description: None,
        input_schema: json!({"type": "object", "properties": {}}),
        annotations: None,
    }
}

#[tokio::test]
async fn inner_cancellation_is_a_named_error_and_a_default_tool_message() {
    let server = CancelledServer::named("cancelled");
    let error = invoke_mcp_tool(&server, &tool(), "{}", None, None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, AgentsError::MCPToolCancellationError { .. }),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "Failed to call tool 'cancel_tool' on MCP server 'cancelled': tool execution was cancelled."
    );

    // Through a run, the default failure policy turns it into the generic tool message and the
    // run's formatter is not asked about it.
    let asked = Arc::new(Mutex::new(0usize));
    let seen = asked.clone();
    let config = RunConfig::<()>::default().with_tool_error_formatter(move |_args| {
        *seen.lock().unwrap() += 1;
        Some("formatter".to_string())
    });
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![function_call_full(
                "cancel_tool",
                "{}",
                "call-1",
                None,
                None,
            )]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("test")
        .with_mcp_server(server)
        .with_model_instance(model);
    let result = Runner::run_with_options(
        agent,
        "go",
        openai_agents_sdk::run::RunOptions::new(()).run_config(config),
    )
    .await
    .unwrap();
    let output = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .expect("a tool output item");
    assert_eq!(
        output,
        json!("An error occurred while running the tool. Please try again.")
    );
    assert_eq!(*asked.lock().unwrap(), 0);

    let _ = to_function_tool::<()>;
}

#[tokio::test]
async fn inner_cancellation_names_the_server_without_credentials() {
    let server = CancelledServer::named(CREDENTIALED);
    let error = invoke_mcp_tool(&server, &tool(), "{}", None, None)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        AgentsError::MCPToolCancellationError { .. }
    ));
    let message = error.to_string();
    assert!(message.contains("cancel_tool"), "{message}");
    assert!(message.contains("mcp.example.com/sse"), "{message}");
    for secret in ["s3cr3t_pw", "SECRET_QS_KEY", "SECRET_FRAGMENT"] {
        assert!(!message.contains(secret), "{message}");
    }
}
