//! `test_stream_step_items_to_queue_emits_helper_events_and_skips_approvals` through the public
//! streamed run: hosted MCP list-tools, reasoning and approval items each get their own event
//! name, and a pending function-tool approval is not announced as an item event.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, MCPToolApprovalResult};
use openai_agents_sdk::{Agent, HostedMCPTool, Runner};
use serde_json::{Value, json};

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

async fn item_event_names(agent: &Agent<()>) -> Vec<String> {
    let mut stream = Runner::run_streamed(agent, "go");
    let mut names = Vec::new();
    let mut events = std::pin::pin!(stream.stream_events());
    while let Some(event) = events.next().await {
        if let StreamEvent::RunItem { name, .. } = event.unwrap() {
            names.push(name.as_str().to_string());
        }
    }
    names
}

#[tokio::test]
async fn mcp_reasoning_and_approval_items_have_their_own_event_names() {
    let hosted = HostedMCPTool::<()>::new(json!({
        "type": "mcp", "server_label": "test-mcp-server", "server_url": "https://example.com",
        "require_approval": "always"
    }))
    .with_on_approval_request(|_req| async move {
        MCPToolApprovalResult {
            approve: true,
            reason: None,
        }
    });
    let list_tools = OutputItem::from_value(json!({
        "id": "list_test-mcp-server", "type": "mcp_list_tools",
        "server_label": "test-mcp-server",
        "tools": [{"name": "search_docs", "input_schema": {},
                   "description": "Search the docs.", "annotations": {"title": "Search Docs"}}],
    }));
    let reasoning = OutputItem::from_value(json!({
        "id": "rid", "type": "reasoning",
        "summary": [{"text": "thinking", "type": "summary_text"}],
    }));
    let request = OutputItem::from_value(json!({
        "id": "mcp-approval-1", "type": "mcp_approval_request",
        "server_label": "test-mcp-server", "arguments": "{}", "name": "search_docs",
    }));
    let agent = Agent::<()>::new("StreamHelper")
        .with_model_instance(scripted(vec![
            vec![list_tools, reasoning, request],
            vec![assistant_message("done")],
        ]))
        .with_tool(hosted);

    let names = item_event_names(&agent).await;
    assert_eq!(
        names,
        [
            "mcp_list_tools",
            "reasoning_item_created",
            "mcp_approval_requested",
            "mcp_approval_response",
            "message_output_created",
        ]
    );
}

#[tokio::test]
async fn a_pending_function_tool_approval_is_not_an_item_event() {
    let tool = FunctionTool::new("tool", "", |_c, _a: Value| async move {
        Ok("never".to_string())
    })
    .with_needs_approval(true);
    let agent = Agent::<()>::new("StreamHelper")
        .with_model_instance(scripted(vec![vec![function_call_full(
            "tool", "{}", "call-1", None, None,
        )]]))
        .with_tool(tool);

    let names = item_event_names(&agent).await;
    assert_eq!(names, ["tool_called"]);
}
