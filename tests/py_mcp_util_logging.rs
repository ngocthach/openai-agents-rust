//! Parity port of the `caplog` tests of `tests/mcp/test_mcp_util.py`: what `invoke_mcp_tool`
//! writes to the log under `DONT_LOG_TOOL_DATA`.
//!
//! The logging policy and the `log` backend are process-global, so every test holds one lock and
//! reads only the records produced while it holds it.
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::debug::{reset_dont_log_tool_data, set_dont_log_tool_data};
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::mcp::util::invoke_mcp_tool;
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};
use serde_json::{Map, Value, json};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push(format!("{}", record.args()));
    }
    fn flush(&self) {}
}

fn install_logger() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

#[derive(Debug)]
struct Crashing {
    name: String,
    config: MCPServerConfig,
    message: String,
}

#[async_trait::async_trait]
impl MCPServer for Crashing {
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
        Ok(vec![])
    }
    async fn call_tool(&self, _n: &str, _a: Option<Map<String, Value>>) -> Result<Value> {
        Err(AgentsError::other(self.message.clone()))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(vec![])
    }
    async fn get_prompt(&self, _n: &str, _a: Option<BTreeMap<String, String>>) -> Result<Value> {
        Ok(Value::Null)
    }
}

fn server(name: &str, message: &str) -> Arc<dyn MCPServer> {
    Arc::new(Crashing {
        name: name.into(),
        config: MCPServerConfig::default(),
        message: message.into(),
    })
}

fn tool(name: &str) -> MCPTool {
    MCPTool {
        meta: None,
        name: name.into(),
        title: None,
        description: None,
        input_schema: json!({}),
        annotations: None,
    }
}

/// Run `f` with the tool-data policy set to `redacted`, returning everything it logged.
async fn with_policy<F, Fut>(redacted: bool, f: F) -> (String, Fut::Output)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future,
{
    install_logger();
    RECORDS.lock().unwrap().clear();
    set_dont_log_tool_data(redacted);
    let out = f().await;
    reset_dont_log_tool_data();
    let text = RECORDS.lock().unwrap().join("\n");
    (text, out)
}

#[tokio::test]
async fn bad_json_redacts_the_payload_when_tool_data_logging_is_off() {
    let _g = LOCK.lock().await;
    let s = server("fake_mcp_server", "unused");
    let bad = r#"{"secret":"SECRET_TOKEN_123""#;
    let (log_text, error) = with_policy(true, || async {
        invoke_mcp_tool(&s, &tool("test_tool_1"), bad, None, None)
            .await
            .unwrap_err()
    })
    .await;
    assert!(matches!(error, AgentsError::ModelBehaviorError { .. }));
    assert_eq!(error.to_string(), "Invalid JSON input for tool test_tool_1");
    assert!(!error.to_string().contains("SECRET_TOKEN_123"));
    assert!(!log_text.contains("SECRET_TOKEN_123"), "{log_text}");
}

#[tokio::test]
async fn bad_json_includes_the_payload_when_tool_data_logging_is_on() {
    let _g = LOCK.lock().await;
    let s = server("fake_mcp_server", "unused");
    let bad = r#"{"secret":"SECRET_TOKEN_123""#;
    let (log_text, error) = with_policy(false, || async {
        invoke_mcp_tool(&s, &tool("test_tool_1"), bad, None, None)
            .await
            .unwrap_err()
    })
    .await;
    assert_eq!(
        error.to_string(),
        format!("Invalid JSON input for tool test_tool_1: {bad}")
    );
    assert!(
        log_text.contains("Invalid JSON input for MCP tool")
            || log_text.contains("SECRET_TOKEN_123"),
        "{log_text}"
    );
}

#[tokio::test]
async fn a_crash_logs_the_failure_and_redacts_it_when_tool_data_logging_is_off() {
    let _g = LOCK.lock().await;
    let s = server("SECRET_CUSTOM_MCP_SERVER", "crash with SECRET_CRASH_123");
    let (log_text, error) = with_policy(true, || async {
        invoke_mcp_tool(&s, &tool("SECRET_MCP_TOOL_NAME"), "", None, None)
            .await
            .unwrap_err()
    })
    .await;
    assert!(error.to_string().contains("Error invoking MCP tool"));
    assert!(log_text.contains("Error invoking MCP tool"), "{log_text}");
    assert!(!log_text.contains("SECRET_CUSTOM_MCP_SERVER"), "{log_text}");
    assert!(!log_text.contains("SECRET_MCP_TOOL_NAME"), "{log_text}");
    assert!(!log_text.contains("SECRET_CRASH_123"), "{log_text}");
}

#[tokio::test]
async fn a_crash_logs_the_error_but_never_url_credentials_when_tool_data_logging_is_on() {
    let _g = LOCK.lock().await;
    let s = server(
        "streamable_http: https://SECRET_CREDENTIAL@example.test/SECRET_MCP_PATH?token=SECRET_MCP_QUERY",
        "crash with SECRET_CRASH_123",
    );
    let (log_text, _) = with_policy(false, || async {
        invoke_mcp_tool(&s, &tool("test_tool_1"), "", None, None)
            .await
            .unwrap_err()
    })
    .await;
    assert!(log_text.contains("SECRET_CRASH_123"), "{log_text}");
    assert!(log_text.contains("SECRET_MCP_PATH"), "{log_text}");
    assert!(!log_text.contains("SECRET_CREDENTIAL"), "{log_text}");
    assert!(!log_text.contains("SECRET_MCP_QUERY"), "{log_text}");
}
