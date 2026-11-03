//! Parity ports of the parallel-connect variants of `tests/mcp/test_mcp_server_manager.py` that
//! the earlier port ran only sequentially: the lifecycle timeouts in force at call time, and a
//! strict connect failure cleaning up every server that connected, including the failed one.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPServerManager, MCPTool};
use serde_json::{Map, Value};

/// Python's `TaskBoundServer`, optionally failing its connect after opening, or slow to connect.
#[derive(Debug)]
struct Server {
    name: &'static str,
    config: MCPServerConfig,
    fail_connect: bool,
    connect_delay: Duration,
    cleaned: AtomicBool,
}

impl Server {
    fn new(name: &'static str, fail_connect: bool, connect_delay: Duration) -> Arc<Self> {
        Arc::new(Server {
            name,
            config: MCPServerConfig::default(),
            fail_connect,
            connect_delay,
            cleaned: AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl MCPServer for Server {
    fn name(&self) -> &str {
        self.name
    }
    fn config(&self) -> &MCPServerConfig {
        &self.config
    }
    async fn connect(&self) -> Result<()> {
        self.cleaned.store(false, Ordering::SeqCst);
        tokio::time::sleep(self.connect_delay).await;
        if self.fail_connect {
            return Err(AgentsError::other("connect failed"));
        }
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        self.cleaned.store(true, Ordering::SeqCst);
        Ok(())
    }
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Err(AgentsError::other("not implemented"))
    }
    async fn call_tool(
        &self,
        _tool_name: &str,
        _arguments: Option<Map<String, Value>>,
    ) -> Result<Value> {
        Err(AgentsError::other("not implemented"))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Err(AgentsError::other("not implemented"))
    }
    async fn get_prompt(
        &self,
        _name: &str,
        _arguments: Option<BTreeMap<String, String>>,
    ) -> Result<Value> {
        Err(AgentsError::other("not implemented"))
    }
}

fn dynamic(server: &Arc<Server>) -> Arc<dyn MCPServer> {
    server.clone()
}

// test_manager_uses_current_lifecycle_timeouts (connect_in_parallel x 2)
#[tokio::test]
async fn manager_uses_current_lifecycle_timeouts() {
    for parallel in [false, true] {
        let server = Server::new("slow", false, Duration::from_millis(200));
        let mut manager = MCPServerManager::new(vec![dynamic(&server)])
            .with_connect_timeout(None)
            .with_cleanup_timeout(None)
            .with_connect_in_parallel(parallel);
        manager.set_connect_timeout(Some(Duration::from_millis(20)));
        manager.connect_all().await.unwrap();
        assert!(manager.active_servers().is_empty(), "parallel={parallel}");
        assert!(
            manager.errors()[0].1.contains("Timed out connecting"),
            "parallel={parallel}"
        );

        manager.set_connect_timeout(Some(Duration::from_secs(5)));
        manager.reconnect(true).await.unwrap();
        assert_eq!(manager.active_servers().len(), 1, "parallel={parallel}");

        manager.set_cleanup_timeout(Some(Duration::from_secs(2)));
        manager.cleanup_all().await;
        assert!(server.cleaned.load(Ordering::SeqCst), "parallel={parallel}");
    }
}

// test_manager_strict_connect_parallel_cleans_up_failed_server
#[tokio::test]
async fn strict_parallel_connect_cleans_up_the_failed_server() {
    let failing = Server::new("failing-task-bound", true, Duration::ZERO);
    let mut manager = MCPServerManager::new(vec![dynamic(&failing)])
        .strict(true)
        .with_connect_in_parallel(true);
    let error = manager.connect_all().await.unwrap_err();
    assert!(error.to_string().contains("connect failed"), "{error}");
    assert!(failing.cleaned.load(Ordering::SeqCst));
}

// test_manager_strict_connect_parallel_cleans_up_workers
#[tokio::test]
async fn strict_parallel_connect_cleans_up_connected_and_failed_servers() {
    let connected = Server::new("task-bound", false, Duration::ZERO);
    let failing = Server::new("failing-task-bound", true, Duration::from_millis(30));
    let mut manager = MCPServerManager::new(vec![dynamic(&connected), dynamic(&failing)])
        .strict(true)
        .with_connect_in_parallel(true);
    let error = manager.connect_all().await.unwrap_err();
    assert!(error.to_string().contains("connect failed"), "{error}");
    assert!(connected.cleaned.load(Ordering::SeqCst));
    assert!(failing.cleaned.load(Ordering::SeqCst));
    assert!(manager.active_servers().is_empty());
}
