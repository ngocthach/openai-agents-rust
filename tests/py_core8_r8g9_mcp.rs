//! Parity ports from the round-8 audit of `tests/mcp/test_mcp_server_manager.py` and
//! `tests/mcp/test_mcp_util.py`.
//!
//! Python reaches "a cleanup that failed after an interrupted connect is never retried" by
//! cancelling a worker task; Rust has no worker tasks, so the connect timeout is the
//! interruption that leaves a server half-opened.
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::util::get_all_function_tools;
use openai_agents_sdk::mcp::{
    MCPServer, MCPServerConfig, MCPServerManager, MCPServerSse, MCPServerSseParams, MCPServerStdio,
    MCPServerStdioParams, MCPServerStreamableHttp, MCPServerStreamableHttpParams, MCPTool,
    MCPToolPolicy,
};
use openai_agents_sdk::tool::Tool;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
};
use openai_agents_sdk::{MCPConfig, RunContext};
use serde_json::{Map, Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::FakeMcpServer;

/// Port of `EmergencyCleanupFailingServer`: its connect never finishes, and its cleanup either
/// raises or never finishes.
#[derive(Debug)]
struct HangingConnectServer {
    config: MCPServerConfig,
    hang_in_cleanup: bool,
    connect_calls: AtomicUsize,
    cleanup_calls: AtomicUsize,
}

#[async_trait::async_trait]
impl MCPServer for HangingConnectServer {
    fn name(&self) -> &str {
        "emergency"
    }
    fn config(&self) -> &MCPServerConfig {
        &self.config
    }
    async fn connect(&self) -> Result<()> {
        self.connect_calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        if self.hang_in_cleanup {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok(())
        } else {
            Err(AgentsError::other("emergency cleanup failed"))
        }
    }
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Err(AgentsError::other("not implemented"))
    }
    async fn call_tool(&self, _: &str, _: Option<Map<String, Value>>) -> Result<Value> {
        Err(AgentsError::other("not implemented"))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Err(AgentsError::other("not implemented"))
    }
    async fn get_prompt(&self, _: &str, _: Option<BTreeMap<String, String>>) -> Result<Value> {
        Err(AgentsError::other("not implemented"))
    }
}

// test_manager_reconnect_does_not_retry_after_failed_emergency_cleanup[raise, timeout]
#[tokio::test]
async fn reconnect_does_not_retry_after_a_failed_cleanup_of_an_interrupted_connect() {
    for hang_in_cleanup in [false, true] {
        let server = Arc::new(HangingConnectServer {
            config: MCPServerConfig::default(),
            hang_in_cleanup,
            connect_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
        });
        let mut manager = MCPServerManager::new(vec![server.clone() as Arc<dyn MCPServer>])
            .with_connect_in_parallel(true)
            .with_connect_timeout(Some(Duration::from_millis(100)))
            .with_cleanup_timeout(Some(Duration::from_millis(100)));

        manager.connect_all().await.unwrap();
        assert_eq!(server.connect_calls.load(Ordering::SeqCst), 1);
        assert_eq!(manager.failed_servers().len(), 1);

        manager.reconnect(true).await.unwrap();

        // The cleanup ran once and failed, so the half-opened server is never connected a
        // second time and the failure is what `errors` reports.
        assert_eq!(server.cleanup_calls.load(Ordering::SeqCst), 1);
        assert_eq!(server.connect_calls.load(Ordering::SeqCst), 1);
        assert!(manager.active_servers().is_empty());
        assert_eq!(manager.failed_servers().len(), 1);
        let errors = manager.errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].0, "emergency");
        if hang_in_cleanup {
            assert_eq!(errors[0].1, "Timed out cleaning up MCP server 'emergency'");
        } else {
            assert_eq!(errors[0].1, "emergency cleanup failed");
        }
    }
}

fn allow_input() -> ToolInputGuardrail<()> {
    ToolInputGuardrail::<()>::new("allow", |_data| async {
        Ok(ToolGuardrailFunctionOutput::allow())
    })
}

fn allow_output() -> ToolOutputGuardrail<()> {
    ToolOutputGuardrail::<()>::new("allow", |_data| async {
        Ok(ToolGuardrailFunctionOutput::allow())
    })
}

fn policy_with_guardrails() -> MCPToolPolicy<()> {
    MCPToolPolicy::<()> {
        tool_input_guardrails: vec![allow_input()],
        tool_output_guardrails: vec![allow_output()],
        ..Default::default()
    }
}

// test_mcp_server_guardrails_apply_to_every_converted_tool_with_isolated_lists
#[tokio::test]
async fn server_guardrails_apply_to_every_converted_tool_with_isolated_lists() {
    let server = Arc::new(
        FakeMcpServer::new("guarded")
            .configure(|config| {
                *config = std::mem::take(config).with_tool_policy(policy_with_guardrails());
            })
            .add_tool("first", json!({}))
            .add_tool("second", json!({})),
    );
    let servers: Vec<Arc<dyn MCPServer>> = vec![server.clone()];
    let tools = get_all_function_tools(
        &servers,
        &MCPConfig::<()>::default(),
        &RunContext::new(()),
        "test_agent",
    )
    .await
    .unwrap();

    let names: Vec<String> = tools.iter().map(Tool::name).collect();
    assert_eq!(names, ["first", "second"]);
    for tool in &tools {
        let Tool::Function(function) = tool else {
            panic!("expected a function tool");
        };
        assert_eq!(function.tool_input_guardrails.len(), 1);
        assert_eq!(function.tool_output_guardrails.len(), 1);
    }

    // The converted tools are immutable `Arc`s, so one tool's lists cannot be cleared under
    // the server's or a sibling's; the server's own policy is unchanged by the conversion.
    let policy = server.config().tool_policy::<()>().expect("the policy");
    assert_eq!(policy.tool_input_guardrails.len(), 1);
    assert_eq!(policy.tool_output_guardrails.len(), 1);
}

// test_public_mcp_server_constructors_forward_guardrail_configuration
#[test]
fn public_server_constructors_forward_guardrail_configuration() {
    let stdio = MCPServerStdio::new(
        "stdio",
        MCPServerStdioParams {
            command: "test".into(),
            ..Default::default()
        },
    )
    .with_tool_policy(policy_with_guardrails());
    let sse = MCPServerSse::new(
        "sse",
        MCPServerSseParams {
            url: "https://example.test/sse".into(),
            ..Default::default()
        },
    )
    .with_tool_policy(policy_with_guardrails());
    let http = MCPServerStreamableHttp::new(
        "streamable_http",
        MCPServerStreamableHttpParams {
            url: "https://example.test/mcp".into(),
            ..Default::default()
        },
    )
    .with_tool_policy(policy_with_guardrails());

    let configs = [stdio.config(), sse.config(), http.config()];
    for config in configs {
        let policy = config.tool_policy::<()>().expect("the policy");
        assert_eq!(policy.tool_input_guardrails.len(), 1);
        assert_eq!(policy.tool_output_guardrails.len(), 1);
    }
}
