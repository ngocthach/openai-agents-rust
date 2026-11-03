//! Cancellation of `MCPServerManager::connect_all` (`tests/mcp/test_mcp_server_manager.py`,
//! `test_manager_cleanup_runs_on_cancelled_error_during_connect` and
//! `test_manager_async_with_cleans_cancelled_server_when_unsuppressed`): a caller that gives up
//! on a connect (a dropped future, here a `timeout`) must still be able to close every server
//! it was opening, and the manager must stay usable afterwards.
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPServerManager, MCPTool};
use serde_json::{Map, Value};

/// A server that counts its lifecycle calls and holds a resource while it connects.
#[derive(Debug)]
struct Server {
    name: &'static str,
    config: MCPServerConfig,
    connect_delay: Duration,
    cleanup_delay: Duration,
    connect_calls: AtomicUsize,
    cleanup_calls: AtomicUsize,
    resource_open: std::sync::atomic::AtomicBool,
}

impl Server {
    fn new(name: &'static str, connect_delay: Duration) -> Arc<Self> {
        Self::with_cleanup_delay(name, connect_delay, Duration::ZERO)
    }

    fn with_cleanup_delay(
        name: &'static str,
        connect_delay: Duration,
        cleanup_delay: Duration,
    ) -> Arc<Self> {
        Arc::new(Server {
            name,
            config: MCPServerConfig::default(),
            connect_delay,
            cleanup_delay,
            connect_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
            resource_open: std::sync::atomic::AtomicBool::new(false),
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
        self.resource_open.store(true, Ordering::SeqCst);
        tokio::time::sleep(self.connect_delay).await;
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.cleanup_delay).await;
        self.resource_open.store(false, Ordering::SeqCst);
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

#[tokio::test]
async fn a_dropped_connect_all_leaves_every_server_cleanable_and_the_manager_reusable() {
    for parallel in [false, true] {
        let fast = Server::new("fast", Duration::ZERO);
        let slow = Server::new("slow", Duration::from_secs(30));
        let mut manager = MCPServerManager::new(vec![fast.clone(), slow.clone()])
            .with_connect_timeout(None)
            .with_connect_in_parallel(parallel);

        let outcome = tokio::time::timeout(Duration::from_millis(150), manager.connect_all()).await;
        assert!(
            outcome.is_err(),
            "parallel={parallel}: connect should still be waiting"
        );
        assert!(
            slow.resource_open.load(Ordering::SeqCst),
            "parallel={parallel}"
        );

        // Nothing can run on drop, so the caller's cleanup reaches every server, including the
        // one that was cancelled half-way through opening its resource.
        manager.cleanup_all().await;
        for server in [&fast, &slow] {
            assert_eq!(
                server.cleanup_calls.load(Ordering::SeqCst),
                1,
                "parallel={parallel}: {}",
                server.name
            );
            assert!(!server.resource_open.load(Ordering::SeqCst));
        }

        // The manager connects again afterwards.
        let mut again =
            MCPServerManager::new(vec![fast.clone()]).with_connect_in_parallel(parallel);
        again.connect_all().await.unwrap();
        assert_eq!(again.active_servers().len(), 1);
        again.cleanup_all().await;
    }
}

#[tokio::test]
async fn a_cleanup_that_times_out_is_recorded_and_the_server_is_not_connected_a_second_time() {
    for parallel in [false, true] {
        let server = Server::with_cleanup_delay("hanging", Duration::ZERO, Duration::from_secs(30));
        let mut manager = MCPServerManager::new(vec![server.clone()])
            .with_cleanup_timeout(Some(Duration::from_millis(100)))
            .with_connect_in_parallel(parallel);
        manager.connect_all().await.unwrap();
        assert_eq!(manager.active_servers().len(), 1);

        manager.cleanup_all().await;
        assert_eq!(
            server.cleanup_calls.load(Ordering::SeqCst),
            1,
            "parallel={parallel}"
        );
        let errors = manager.errors();
        assert_eq!(errors.len(), 1, "parallel={parallel}");
        assert!(errors[0].1.contains("Timed out cleaning up"), "{errors:?}");

        // The server that could not be closed is never opened over its old connection.
        manager.reconnect(true).await.unwrap();
        assert_eq!(
            server.connect_calls.load(Ordering::SeqCst),
            1,
            "parallel={parallel}"
        );
        assert!(manager.active_servers().is_empty(), "parallel={parallel}");
        assert!(manager.errors()[0].1.contains("Timed out cleaning up"));
    }
}
