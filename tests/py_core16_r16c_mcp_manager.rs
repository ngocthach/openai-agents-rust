//! Ports of `tests/mcp/test_mcp_server_manager.py` cases that the worker-task suite hid: the
//! restored active list after a strict failed connect with `drop_failed_servers=False`, and a
//! cleanup that hangs past the cleanup timeout (the timeout twin of
//! `test_manager_reconnect_does_not_retry_after_failed_emergency_cleanup`).
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPServerManager, MCPTool};
use serde_json::{Map, Value};

#[derive(Debug)]
struct Server {
    name: &'static str,
    config: MCPServerConfig,
    fail_connect: bool,
    hang_cleanup: bool,
    connect_calls: AtomicUsize,
    cleanup_calls: AtomicUsize,
}

impl Server {
    fn new(name: &'static str, fail_connect: bool, hang_cleanup: bool) -> Arc<Self> {
        Arc::new(Server {
            name,
            config: MCPServerConfig::default(),
            fail_connect,
            hang_cleanup,
            connect_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
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
        self.connect_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_connect {
            return Err(AgentsError::other("connect failed"));
        }
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        if self.hang_cleanup {
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
        Ok(())
    }
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Err(AgentsError::other("not implemented"))
    }
    async fn call_tool(&self, _n: &str, _a: Option<Map<String, Value>>) -> Result<Value> {
        Err(AgentsError::other("not implemented"))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Err(AgentsError::other("not implemented"))
    }
    async fn get_prompt(&self, _n: &str, _a: Option<BTreeMap<String, String>>) -> Result<Value> {
        Err(AgentsError::other("not implemented"))
    }
}

// test_manager_restores_one_shot_iterable_servers_after_a_failed_connect
#[tokio::test]
async fn strict_failed_connect_keeps_every_server_active_when_failed_servers_are_kept() {
    let server = Server::new("flaky", true, false);
    let mut manager = MCPServerManager::new(vec![server.clone() as Arc<dyn MCPServer>])
        .strict(true)
        .with_drop_failed_servers(false);
    assert!(manager.connect_all().await.is_err());
    let active = manager.active_servers();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].name(), "flaky");
}

// test_manager_reconnect_does_not_retry_after_failed_emergency_cleanup[timeout]
#[tokio::test]
async fn reconnect_does_not_retry_a_server_whose_cleanup_timed_out() {
    let server = Server::new("stuck", true, true);
    let mut manager = MCPServerManager::new(vec![server.clone() as Arc<dyn MCPServer>])
        .with_cleanup_timeout(Some(Duration::from_millis(100)));
    manager.connect_all().await.unwrap();
    manager.reconnect(true).await.unwrap();

    assert_eq!(server.connect_calls.load(Ordering::SeqCst), 1);
    assert_eq!(server.cleanup_calls.load(Ordering::SeqCst), 1);
    assert!(manager.active_servers().is_empty());
    assert_eq!(manager.failed_servers().len(), 1);
    let errors = manager.errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].1.contains("Timed out cleaning up"), "{errors:?}");
}
