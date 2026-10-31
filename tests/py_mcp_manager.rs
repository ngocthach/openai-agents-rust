//! Parity port of `tests/mcp/test_mcp_server_manager.py` and
//! `tests/mcp/test_mcp_server_manager_cleanup_state.py`.
//!
//! Python binds each server to a worker task because anyio requires connect and cleanup to run
//! in the same task; Rust has no such requirement, so the `_ServerWorker` tests have no
//! counterpart. The lifecycle contract the manager exposes — which servers are active, which
//! failed, what is retried and what is cleaned up first — is ported in full.
#![allow(clippy::await_holding_lock)] // the lock serializes the process-wide logger for the whole test
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::{
    MCPServer, MCPServerConfig, MCPServerManager, MCPTool, get_mcp_server_log_name,
};
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Fake servers (ports of the Python test doubles in the same file)
// ---------------------------------------------------------------------------

/// Emit the whole `MCPServer` impl for a fake that only implements the lifecycle.
///
/// The impl has to be generated in one piece: `#[async_trait]` rewrites the methods it can see,
/// so methods produced by a macro inside the impl would stay un-rewritten.
macro_rules! lifecycle_server {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl MCPServer for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn config(&self) -> &MCPServerConfig {
                &self.config
            }
            async fn connect(&self) -> Result<()> {
                self.do_connect().await
            }
            async fn cleanup(&self) -> Result<()> {
                self.do_cleanup().await
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
    };
}

/// Port of Python's `TaskBoundServer`: records that it was cleaned up.
#[derive(Debug)]
struct PlainServer {
    name: String,
    config: MCPServerConfig,
    cleaned: AtomicBool,
    connect_calls: AtomicUsize,
    cleanup_calls: AtomicUsize,
}

impl PlainServer {
    fn new(name: &str) -> Arc<Self> {
        Arc::new(PlainServer {
            name: name.to_string(),
            config: MCPServerConfig::default(),
            cleaned: AtomicBool::new(false),
            connect_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
        })
    }

    fn cleaned(&self) -> bool {
        self.cleaned.load(Ordering::SeqCst)
    }
}

impl PlainServer {
    async fn do_connect(&self) -> Result<()> {
        self.connect_calls.fetch_add(1, Ordering::SeqCst);
        self.cleaned.store(false, Ordering::SeqCst);
        Ok(())
    }
    async fn do_cleanup(&self) -> Result<()> {
        self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        self.cleaned.store(true, Ordering::SeqCst);
        Ok(())
    }
}

lifecycle_server!(PlainServer);

/// Port of Python's `FlakyServer`: fails its first `failures` connects, then succeeds.
#[derive(Debug)]
struct FlakyServer {
    name: String,
    config: MCPServerConfig,
    failures_remaining: AtomicUsize,
    connect_calls: AtomicUsize,
    cleanup_calls: AtomicUsize,
    cleaned: AtomicBool,
}

impl FlakyServer {
    fn new(name: &str, failures: usize) -> Arc<Self> {
        Arc::new(FlakyServer {
            name: name.to_string(),
            config: MCPServerConfig::default(),
            failures_remaining: AtomicUsize::new(failures),
            connect_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
            cleaned: AtomicBool::new(false),
        })
    }

    fn connect_calls(&self) -> usize {
        self.connect_calls.load(Ordering::SeqCst)
    }
}

impl FlakyServer {
    async fn do_connect(&self) -> Result<()> {
        self.connect_calls.fetch_add(1, Ordering::SeqCst);
        let remaining = self.failures_remaining.load(Ordering::SeqCst);
        if remaining > 0 {
            self.failures_remaining
                .store(remaining - 1, Ordering::SeqCst);
            return Err(AgentsError::other("connect failed"));
        }
        Ok(())
    }
    async fn do_cleanup(&self) -> Result<()> {
        self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        self.cleaned.store(true, Ordering::SeqCst);
        Ok(())
    }
}

lifecycle_server!(FlakyServer);

/// Port of Python's `PartialFailureServer`: the first connect opens a resource and then fails,
/// so a retry that did not clean up first is rejected.
#[derive(Debug)]
struct PartialFailureServer {
    name: String,
    config: MCPServerConfig,
    fail_cleanup: bool,
    connect_calls: AtomicUsize,
    cleanup_calls: AtomicUsize,
    resource_open: AtomicBool,
}

impl PartialFailureServer {
    fn new(fail_cleanup: bool) -> Arc<Self> {
        Arc::new(PartialFailureServer {
            name: "partial-failure".to_string(),
            config: MCPServerConfig::default(),
            fail_cleanup,
            connect_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
            resource_open: AtomicBool::new(false),
        })
    }
}

impl PartialFailureServer {
    async fn do_connect(&self) -> Result<()> {
        let calls = self.connect_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.resource_open.load(Ordering::SeqCst) {
            return Err(AgentsError::other("connect called without cleanup"));
        }
        self.resource_open.store(true, Ordering::SeqCst);
        if calls == 1 {
            return Err(AgentsError::other("connect failed after opening resource"));
        }
        Ok(())
    }
    async fn do_cleanup(&self) -> Result<()> {
        self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_cleanup {
            return Err(AgentsError::other("cleanup failed"));
        }
        self.resource_open.store(false, Ordering::SeqCst);
        Ok(())
    }
}

lifecycle_server!(PartialFailureServer);

/// Port of Python's `CleanupAwareServer`: connecting twice without a cleanup in between is an
/// error, so an idempotent `connect_all` is observable.
#[derive(Debug)]
struct CleanupAwareServer {
    name: String,
    config: MCPServerConfig,
    connect_calls: AtomicUsize,
    cleanup_calls: AtomicUsize,
}

impl CleanupAwareServer {
    fn new() -> Arc<Self> {
        Arc::new(CleanupAwareServer {
            name: "cleanup-aware".to_string(),
            config: MCPServerConfig::default(),
            connect_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
        })
    }
}

impl CleanupAwareServer {
    async fn do_connect(&self) -> Result<()> {
        if self.connect_calls.load(Ordering::SeqCst) > self.cleanup_calls.load(Ordering::SeqCst) {
            return Err(AgentsError::other("connect called without cleanup"));
        }
        self.connect_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn do_cleanup(&self) -> Result<()> {
        self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

lifecycle_server!(CleanupAwareServer);

/// Port of Python's `CleanupFailingServer`.
#[derive(Debug)]
struct CleanupFailingServer {
    name: String,
    config: MCPServerConfig,
}

impl CleanupFailingServer {
    fn new(name: &str) -> Arc<Self> {
        Arc::new(CleanupFailingServer {
            name: name.to_string(),
            config: MCPServerConfig::default(),
        })
    }
}

impl CleanupFailingServer {
    async fn do_connect(&self) -> Result<()> {
        Ok(())
    }
    async fn do_cleanup(&self) -> Result<()> {
        Err(AgentsError::other("cleanup failed"))
    }
}

lifecycle_server!(CleanupFailingServer);

/// A server whose connect blocks past the manager's connect timeout.
#[derive(Debug)]
struct SlowConnectServer {
    name: String,
    config: MCPServerConfig,
    delay: Duration,
}

impl SlowConnectServer {
    fn new(delay: Duration) -> Arc<Self> {
        Arc::new(SlowConnectServer {
            name: "slow".to_string(),
            config: MCPServerConfig::default(),
            delay,
        })
    }
}

impl SlowConnectServer {
    async fn do_connect(&self) -> Result<()> {
        tokio::time::sleep(self.delay).await;
        Ok(())
    }
    async fn do_cleanup(&self) -> Result<()> {
        Ok(())
    }
}

lifecycle_server!(SlowConnectServer);

/// Port of Python's `SensitiveNamedServer`: a URL-derived name whose connect error carries a
/// secret.
#[derive(Debug)]
struct SensitiveNamedServer {
    name: String,
    config: MCPServerConfig,
}

impl SensitiveNamedServer {
    fn new(name: &str) -> Arc<Self> {
        Arc::new(SensitiveNamedServer {
            name: name.to_string(),
            config: MCPServerConfig::default(),
        })
    }
}

impl SensitiveNamedServer {
    async fn do_connect(&self) -> Result<()> {
        Err(AgentsError::other("SECRET_MCP_CONNECT_ERROR"))
    }
    async fn do_cleanup(&self) -> Result<()> {
        Ok(())
    }
}

lifecycle_server!(SensitiveNamedServer);

fn as_server<T: MCPServer + 'static>(server: Arc<T>) -> Arc<dyn MCPServer> {
    server
}

fn names(servers: &[Arc<dyn MCPServer>]) -> Vec<String> {
    servers.iter().map(|s| s.name().to_string()).collect()
}

// ---------------------------------------------------------------------------
// test_get_mcp_server_log_name
// ---------------------------------------------------------------------------

#[test]
fn get_mcp_server_log_name_cases() {
    let cases = [
        ("ordinary-server", "ordinary-server"),
        (
            "sse: https://user:password@example.test/events?token=secret#fragment",
            "sse: https://example.test/events",
        ),
        (
            "streamable_http: https://example.test/mcp?token=secret",
            "streamable_http: https://example.test/mcp",
        ),
        (
            "streamable_http: https://user:password@example.test:8443/mcp?token=secret",
            "streamable_http: https://example.test:8443/mcp",
        ),
        (
            "streamable_http: https://[::1]:8000/mcp",
            "streamable_http: https://[::1]:8000/mcp",
        ),
        (
            "streamable-http: https://example.test/mcp#secret",
            "streamable-http: https://example.test/mcp",
        ),
        (
            "streamable_http: https://user:password@[invalid/mcp?token=secret",
            "streamable_http: <invalid-url>",
        ),
        (
            "streamable_http: https://user:password/mcp?token=secret",
            "streamable_http: <invalid-url>",
        ),
        (
            "https://user:password@example.test/mcp?token=secret",
            "https://example.test/mcp",
        ),
        (
            "https://user:password@[invalid/mcp?token=secret",
            "<invalid-url>",
        ),
        (
            "stdio: python server.py?token=secret",
            "stdio: python server.py?token=secret",
        ),
    ];
    for (name, expected) in cases {
        assert_eq!(get_mcp_server_log_name(name), expected, "name: {name}");
    }
}

// ---------------------------------------------------------------------------
// Lifecycle timeouts
// ---------------------------------------------------------------------------

#[test]
fn manager_defaults_to_finite_lifecycle_timeouts() {
    let manager = MCPServerManager::new(Vec::new());
    assert_eq!(manager.connect_timeout(), Some(Duration::from_secs(10)));
    assert_eq!(manager.cleanup_timeout(), Some(Duration::from_secs(10)));
}

#[tokio::test]
async fn manager_uses_current_lifecycle_timeouts() {
    // Python reassigns `connect_timeout_seconds` between calls; the timeout in force at call
    // time is the one applied.
    let server = as_server(SlowConnectServer::new(Duration::from_millis(200)));
    let mut manager = MCPServerManager::new(vec![server]).with_connect_timeout(None);
    manager.set_connect_timeout(Some(Duration::from_millis(20)));
    manager.connect_all().await.unwrap();

    assert!(manager.active_servers().is_empty());
    assert_eq!(manager.failed_servers().len(), 1);
    assert!(manager.errors()[0].1.contains("Timed out connecting"));

    manager.set_connect_timeout(Some(Duration::from_secs(5)));
    manager.reconnect(true).await.unwrap();
    assert_eq!(names(&manager.active_servers()), vec!["slow"]);
}

// ---------------------------------------------------------------------------
// connect / reconnect / cleanup
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manager_connects_and_cleans_up() {
    let server = PlainServer::new("task-bound");
    let mut manager = MCPServerManager::new(vec![as_server(server.clone())]);
    manager.connect_all().await.unwrap();
    assert_eq!(names(&manager.active_servers()), vec!["task-bound"]);
    manager.cleanup_all().await;
    assert!(server.cleaned());
}

#[tokio::test]
async fn manager_reconnect_failed_only() {
    let server = FlakyServer::new("flaky", 1);
    let mut manager = MCPServerManager::new(vec![as_server(server.clone())]);
    manager.connect_all().await.unwrap();
    assert!(manager.active_servers().is_empty());
    assert_eq!(
        names(&[manager.failed_servers()[0].0.clone()]),
        vec!["flaky"]
    );

    manager.reconnect(true).await.unwrap();
    assert_eq!(names(&manager.active_servers()), vec!["flaky"]);
    assert!(manager.failed_servers().is_empty());
}

#[tokio::test]
async fn failed_and_error_snapshots_are_copies() {
    // Python's `failed_servers` / `errors` return snapshots; mutating them changes nothing.
    let server = FlakyServer::new("flaky", 1);
    let mut manager = MCPServerManager::new(vec![as_server(server.clone())]);
    manager.connect_all().await.unwrap();

    let mut errors = manager.errors();
    let original = errors.clone();
    errors.clear();
    errors.push(("replacement".into(), "replacement error".into()));
    assert_eq!(manager.errors(), original);

    let mut active = manager.active_servers();
    active.clear();
    assert_eq!(manager.active_servers().len(), 0);

    manager.reconnect(true).await.unwrap();
    assert_eq!(server.connect_calls(), 2);
    assert_eq!(names(&manager.active_servers()), vec!["flaky"]);
    assert!(manager.failed_servers().is_empty());
    assert!(manager.errors().is_empty());
}

#[tokio::test]
async fn manager_reconnect_cleans_partial_failure_before_retry() {
    for parallel in [false, true] {
        let healthy = CleanupAwareServer::new();
        let failed = PartialFailureServer::new(false);
        let mut manager =
            MCPServerManager::new(vec![as_server(healthy.clone()), as_server(failed.clone())])
                .with_connect_in_parallel(parallel);

        manager.connect_all().await.unwrap();
        assert_eq!(names(&manager.active_servers()), vec!["cleanup-aware"]);
        assert_eq!(manager.failed_servers().len(), 1);

        manager.reconnect(true).await.unwrap();

        assert_eq!(
            names(&manager.active_servers()),
            vec!["cleanup-aware", "partial-failure"]
        );
        assert!(manager.failed_servers().is_empty());
        assert!(manager.errors().is_empty());
        assert_eq!(failed.connect_calls.load(Ordering::SeqCst), 2);
        assert_eq!(failed.cleanup_calls.load(Ordering::SeqCst), 1);
        assert!(failed.resource_open.load(Ordering::SeqCst));
        assert_eq!(healthy.connect_calls.load(Ordering::SeqCst), 1);
        assert_eq!(healthy.cleanup_calls.load(Ordering::SeqCst), 0);

        manager.cleanup_all().await;
    }
}

#[tokio::test]
async fn manager_reconnect_does_not_retry_after_cleanup_failure() {
    for parallel in [false, true] {
        let server = PartialFailureServer::new(true);
        let mut manager = MCPServerManager::new(vec![as_server(server.clone())])
            .with_connect_in_parallel(parallel);

        manager.connect_all().await.unwrap();
        manager.reconnect(true).await.unwrap();

        assert!(manager.active_servers().is_empty());
        assert_eq!(manager.failed_servers().len(), 1);
        assert_eq!(server.connect_calls.load(Ordering::SeqCst), 1);
        assert_eq!(server.cleanup_calls.load(Ordering::SeqCst), 1);
        assert!(server.resource_open.load(Ordering::SeqCst));
        assert_eq!(
            manager.errors(),
            vec![("partial-failure".into(), "cleanup failed".into())]
        );
    }
}

#[tokio::test]
async fn manager_reconnect_deduplicates_failures() {
    let server = FlakyServer::new("flaky", 2);
    let mut manager =
        MCPServerManager::new(vec![as_server(server.clone())]).with_connect_in_parallel(true);

    manager.connect_all().await.unwrap();
    assert!(manager.active_servers().is_empty());
    assert_eq!(manager.failed_servers().len(), 1);
    assert_eq!(server.connect_calls(), 1);

    manager.reconnect(true).await.unwrap();
    assert!(manager.active_servers().is_empty());
    assert_eq!(manager.failed_servers().len(), 1);
    assert_eq!(server.connect_calls(), 2);

    manager.reconnect(true).await.unwrap();
    assert_eq!(names(&manager.active_servers()), vec!["flaky"]);
    assert!(manager.failed_servers().is_empty());
    assert_eq!(server.connect_calls(), 3);
}

#[tokio::test]
async fn manager_connect_all_retries_all_servers() {
    let server = FlakyServer::new("flaky", 1);
    let mut manager = MCPServerManager::new(vec![as_server(server.clone())]);

    manager.connect_all().await.unwrap();
    assert!(manager.active_servers().is_empty());
    assert_eq!(manager.failed_servers().len(), 1);
    assert_eq!(server.connect_calls(), 1);

    manager.connect_all().await.unwrap();
    assert_eq!(names(&manager.active_servers()), vec!["flaky"]);
    assert!(manager.failed_servers().is_empty());
    assert_eq!(server.connect_calls(), 2);
    manager.cleanup_all().await;
}

#[tokio::test]
async fn manager_connect_all_is_idempotent() {
    let server = CleanupAwareServer::new();
    let mut manager = MCPServerManager::new(vec![as_server(server.clone())]);
    manager.connect_all().await.unwrap();
    assert_eq!(server.connect_calls.load(Ordering::SeqCst), 1);
    // A second connect_all must not open a second connection.
    manager.connect_all().await.unwrap();
    assert_eq!(server.connect_calls.load(Ordering::SeqCst), 1);
    manager.cleanup_all().await;
}

#[tokio::test]
async fn manager_reconnect_all_avoids_duplicate_connections() {
    let server = CleanupAwareServer::new();
    let mut manager = MCPServerManager::new(vec![as_server(server.clone())]);
    manager.connect_all().await.unwrap();
    assert_eq!(server.connect_calls.load(Ordering::SeqCst), 1);
    manager.reconnect(false).await.unwrap();
    assert_eq!(server.connect_calls.load(Ordering::SeqCst), 2);
    assert_eq!(server.cleanup_calls.load(Ordering::SeqCst), 1);
    assert_eq!(names(&manager.active_servers()), vec!["cleanup-aware"]);
}

#[tokio::test]
async fn manager_strict_reconnect_refreshes_active_servers() {
    let a = FlakyServer::new("a", 1);
    let b = FlakyServer::new("b", 2);
    let mut manager = MCPServerManager::new(vec![as_server(a.clone()), as_server(b.clone())]);
    manager.connect_all().await.unwrap();
    assert!(manager.active_servers().is_empty());

    manager.set_strict(true);
    let error = manager.reconnect(true).await.unwrap_err();
    assert!(error.to_string().contains("connect failed"), "{error}");

    assert_eq!(names(&manager.active_servers()), vec!["a"]);
    assert_eq!(manager.failed_servers().len(), 1);
    assert_eq!(manager.failed_servers()[0].0.name(), "b");
}

#[tokio::test]
async fn manager_strict_connect_preserves_existing_active_servers() {
    let connected = PlainServer::new("connected");
    let failing = FlakyServer::new("failing", 2);
    let mut manager = MCPServerManager::new(vec![
        as_server(connected.clone()),
        as_server(failing.clone()),
    ]);
    manager.connect_all().await.unwrap();
    assert_eq!(names(&manager.active_servers()), vec!["connected"]);
    assert_eq!(manager.failed_servers().len(), 1);

    manager.set_strict(true);
    let error = manager.connect_all().await.unwrap_err();
    assert!(error.to_string().contains("connect failed"), "{error}");

    assert_eq!(names(&manager.active_servers()), vec!["connected"]);
    assert_eq!(manager.failed_servers()[0].0.name(), "failing");
    manager.cleanup_all().await;
}

#[tokio::test]
async fn manager_strict_connect_cleans_up_connected_servers() {
    for parallel in [false, true] {
        let connected = PlainServer::new("connected");
        let failing = FlakyServer::new("failing", 1);
        let mut manager = MCPServerManager::new(vec![
            as_server(connected.clone()),
            as_server(failing.clone()),
        ])
        .strict(true)
        .with_connect_in_parallel(parallel);

        let error = manager.connect_all().await.unwrap_err();
        assert!(error.to_string().contains("connect failed"), "{error}");
        assert!(connected.cleaned(), "parallel={parallel}");
        assert!(manager.active_servers().is_empty());
    }
}

#[tokio::test]
async fn manager_strict_connect_cleans_up_failed_server() {
    for parallel in [false, true] {
        let failing = FlakyServer::new("failing-task-bound", 1);
        let mut manager = MCPServerManager::new(vec![as_server(failing.clone())])
            .strict(true)
            .with_connect_in_parallel(parallel);

        let error = manager.connect_all().await.unwrap_err();
        assert!(error.to_string().contains("connect failed"), "{error}");
        assert_eq!(failing.cleanup_calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn manager_cleanup_failure_is_recorded() {
    // Python keeps the worker's cleanup error in `manager.errors`.
    let server = CleanupFailingServer::new("cleanup-failing");
    let mut manager = MCPServerManager::new(vec![as_server(server.clone())]);
    manager.connect_all().await.unwrap();
    manager.cleanup_all().await;

    assert_eq!(
        manager.errors(),
        vec![("cleanup-failing".to_string(), "cleanup failed".to_string())]
    );
    assert!(manager.active_servers().is_empty());
}

#[tokio::test]
async fn manager_keeps_duplicate_servers_once_and_reports_all_servers() {
    let server = as_server(PlainServer::new("dup"));
    let manager = MCPServerManager::new(vec![server.clone(), server.clone()]);
    assert_eq!(names(&manager.all_servers()), vec!["dup"]);
    // Python exposes every managed server before a connection attempt.
    assert_eq!(names(&manager.active_servers()), vec!["dup"]);
}

#[tokio::test]
async fn manager_without_dropping_failed_servers_keeps_them_active() {
    let server = FlakyServer::new("flaky", 1);
    let mut manager =
        MCPServerManager::new(vec![as_server(server.clone())]).with_drop_failed_servers(false);
    manager.connect_all().await.unwrap();

    assert_eq!(names(&manager.active_servers()), vec!["flaky"]);
    assert_eq!(manager.failed_servers().len(), 1);
}

#[tokio::test]
async fn manager_cleans_up_in_reverse_order() {
    // Python documents "cleanup all servers in reverse order".
    let order = Arc::new(Mutex::new(Vec::new()));

    #[derive(Debug)]
    struct Recording {
        name: String,
        config: MCPServerConfig,
        order: Arc<Mutex<Vec<String>>>,
    }

    impl Recording {
        async fn do_connect(&self) -> Result<()> {
            Ok(())
        }
        async fn do_cleanup(&self) -> Result<()> {
            self.order.lock().unwrap().push(self.name.clone());
            Ok(())
        }
    }

    lifecycle_server!(Recording);

    let make = |name: &str| {
        as_server(Arc::new(Recording {
            name: name.to_string(),
            config: MCPServerConfig::default(),
            order: order.clone(),
        }))
    };
    let mut manager = MCPServerManager::new(vec![make("first"), make("second")]);
    manager.connect_all().await.unwrap();
    manager.cleanup_all().await;

    assert_eq!(*order.lock().unwrap(), vec!["second", "first"]);
}

// ---------------------------------------------------------------------------
// Log sanitization (test_manager_sanitizes_url_derived_server_names_in_failure_logs)
// ---------------------------------------------------------------------------

static CAPTURE: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();
static LOG_LOCK: Mutex<()> = Mutex::new(());

struct CaptureLogger;

impl log::Log for CaptureLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == "openai.agents" {
            capture().lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn capture() -> Arc<Mutex<Vec<String>>> {
    CAPTURE
        .get_or_init(|| Arc::new(Mutex::new(Vec::new())))
        .clone()
}

fn install_logger() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let _ = log::set_boxed_logger(Box::new(CaptureLogger));
        log::set_max_level(log::LevelFilter::Trace);
    });
}

async fn connect_failure_log(server_name: &str, redacted: bool) -> String {
    let _guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    install_logger();
    openai_agents_sdk::debug::set_dont_log_tool_data(redacted);
    capture().lock().unwrap().clear();

    let server = as_server(SensitiveNamedServer::new(server_name));
    let mut manager = MCPServerManager::new(vec![server]);
    manager.connect_all().await.unwrap();

    openai_agents_sdk::debug::reset_dont_log_tool_data();

    capture().lock().unwrap().join("\n")
}

#[tokio::test]
async fn manager_sanitizes_url_derived_server_names_in_failure_logs() {
    let url_name = "streamable_http: https://SECRET_CREDENTIAL@example.test/\
                    SECRET_MCP_PATH?token=SECRET_MCP_QUERY#SECRET_MCP_FRAGMENT";
    for redacted in [true, false] {
        let text = connect_failure_log(url_name, redacted).await;
        assert!(text.contains("Failed to connect MCP server"), "{text}");
        assert_eq!(text.contains("SECRET_MCP_PATH"), !redacted, "{text}");
        assert_eq!(
            text.contains("SECRET_MCP_CONNECT_ERROR"),
            !redacted,
            "{text}"
        );
        for hidden in [
            "SECRET_CREDENTIAL",
            "SECRET_MCP_QUERY",
            "SECRET_MCP_FRAGMENT",
        ] {
            assert!(!text.contains(hidden), "{hidden} leaked: {text}");
        }
    }
}

#[tokio::test]
async fn manager_hides_custom_server_names_in_redacted_mode() {
    for redacted in [true, false] {
        let text = connect_failure_log("SECRET_CUSTOM_MCP_SERVER_NAME", redacted).await;
        assert_eq!(
            text.contains("SECRET_CUSTOM_MCP_SERVER_NAME"),
            !redacted,
            "{text}"
        );
        assert_eq!(
            text.contains("SECRET_MCP_CONNECT_ERROR"),
            !redacted,
            "{text}"
        );
    }
}
