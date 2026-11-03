//! Port of `test_cleanup_all_removes_cleaned_servers_from_active_servers` in
//! `tests/mcp/test_mcp_server_manager_cleanup_state.py`: once `cleanup_all` has run, no server is
//! active or connected, `reconnect` has nothing to retry and does not connect again, and
//! `connect_all` connects the server a second time.
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPServerManager, MCPTool};
use serde_json::{Map, Value};

#[derive(Debug)]
struct CountingServer {
    config: MCPServerConfig,
    connects: AtomicUsize,
    cleanups: AtomicUsize,
}

#[async_trait::async_trait]
impl MCPServer for CountingServer {
    fn name(&self) -> &str {
        "counting"
    }
    fn config(&self) -> &MCPServerConfig {
        &self.config
    }
    async fn connect(&self) -> Result<()> {
        self.connects.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        self.cleanups.fetch_add(1, Ordering::SeqCst);
        Ok(())
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

#[tokio::test]
async fn cleanup_all_removes_cleaned_servers_from_the_active_servers() {
    let server = Arc::new(CountingServer {
        config: MCPServerConfig::default(),
        connects: AtomicUsize::new(0),
        cleanups: AtomicUsize::new(0),
    });
    let mut manager = MCPServerManager::new(vec![server.clone() as Arc<dyn MCPServer>]);
    manager.connect_all().await.unwrap();
    assert_eq!(manager.active_servers().len(), 1);

    manager.cleanup_all().await;

    assert!(manager.active_servers().is_empty());
    assert_eq!(server.cleanups.load(Ordering::SeqCst), 1);

    manager.reconnect(true).await.unwrap();
    assert!(manager.active_servers().is_empty());
    assert_eq!(server.connects.load(Ordering::SeqCst), 1);

    manager.connect_all().await.unwrap();
    assert_eq!(manager.active_servers().len(), 1);
    assert_eq!(server.connects.load(Ordering::SeqCst), 2);
}
