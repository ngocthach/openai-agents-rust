#![cfg(feature = "mcp")]
//! Port of `test_mcp_custom_data_extractor_maps_result_meta_to_tool_output_item` in
//! `tests/test_tool_custom_data.py`: the MCP call result's `_meta` reaches the custom-data
//! extractor as `result_meta`.

use std::sync::Arc;

use openai_agents_sdk::error::Result;
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool, MCPToolPolicy};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::{Agent, RunItemKind, Runner};
use serde_json::{Map, Value, json};

#[derive(Debug)]
struct MetaServer {
    name: String,
    config: MCPServerConfig,
    tools: Vec<MCPTool>,
}

#[async_trait::async_trait]
impl MCPServer for MetaServer {
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
        Ok(self.tools.clone())
    }
    async fn call_tool(&self, name: &str, _args: Option<Map<String, Value>>) -> Result<Value> {
        Ok(json!({
            "content": [{"type": "text", "text": format!("result_{name}")}],
            "_meta": {"chart": {"type": "line"}},
        }))
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

#[tokio::test]
async fn mcp_custom_data_extractor_maps_result_meta_to_tool_output_item() {
    let mut policy = MCPToolPolicy::<()>::new();
    policy.custom_data_extractor = Some(Arc::new(|ctx| {
        let meta = ctx.result_meta.clone().unwrap_or_else(|| json!({}));
        Box::pin(async move { Some(json!({"mcp_response_meta": meta})) })
    }));
    let config = MCPServerConfig {
        tool_policy: Some(Arc::new(policy)),
        ..Default::default()
    };
    let server: Arc<dyn MCPServer> = Arc::new(MetaServer {
        name: "meta".into(),
        config,
        tools: vec![MCPTool {
            meta: None,
            name: "meta_tool".into(),
            title: None,
            description: Some("Returns metadata.".into()),
            input_schema: json!({"type": "object", "properties": {}}),
            annotations: None,
        }],
    });

    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![function_call_full(
                "meta_tool",
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
    let result = Runner::run(agent, "user").await.unwrap();

    let output = result
        .new_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .expect("a tool output item");
    assert_eq!(
        output.custom_data(),
        Some(&json!({"mcp_response_meta": {"chart": {"type": "line"}}}))
    );
}
