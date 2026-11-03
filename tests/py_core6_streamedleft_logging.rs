//! `test_error_logging_redaction.py` cases the earlier ports left open: Responses model request
//! failures (`get_response` / `stream_response`) follow the model data policy, and an
//! `Agent::as_tool` argument validation error follows the tool data policy.
//!
//! The logging policy flags and the `log` backend are process-global, so every test holds one
//! lock and resets the flags it touches.

use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::models::interface::{Model, ModelRequest};
use openai_agents_sdk::models::{OpenAIClient, OpenAIResponsesModel};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunContext};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const SECRET: &str = "SECRET_MODEL_BODY_123";
const TOOL_ARGUMENT_SECRET: &str = "SECRET_TOOL_ARGUMENT_123";

static RECORDS: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS
            .lock()
            .unwrap()
            .push((record.level(), format!("{}", record.args())));
    }
    fn flush(&self) {}
}

fn install() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

struct Flags;

impl Drop for Flags {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

fn flags(model: bool, tool: bool) -> Flags {
    install();
    RECORDS.lock().unwrap().clear();
    set_dont_log_model_data(model);
    set_dont_log_tool_data(tool);
    Flags
}

fn error_records() -> Vec<String> {
    RECORDS
        .lock()
        .unwrap()
        .iter()
        .filter(|(level, _)| *level == log::Level::Error)
        .map(|(_, message)| message.clone())
        .collect()
}

/// A server that answers every request with a 400 whose body carries `SECRET`.
async fn failing_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(idx) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let len: usize = text[..idx]
                        .lines()
                        .filter_map(|l| l.split_once(':'))
                        .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
                        .and_then(|(_, v)| v.trim().parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break;
                    }
                }
                let body = format!(r#"{{"error": {{"message": "{SECRET}"}}}}"#);
                let reply = format!(
                    "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(reply.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}/v1")
}

fn model_at(base: &str) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new(
        "test-model",
        OpenAIClient::new("sk-test").with_base_url(base),
    )
}

async fn failing_request(streamed: bool, base: &str) -> AgentsError {
    let model = model_at(base);
    if streamed {
        match model.stream_response(ModelRequest::default()).await {
            Err(error) => error,
            Ok(mut stream) => {
                let mut failure = None;
                while let Some(item) = stream.next().await {
                    if let Err(error) = item {
                        failure = Some(error);
                    }
                }
                failure.expect("the stream fails")
            }
        }
    } else {
        model
            .get_response(ModelRequest::default())
            .await
            .expect_err("the request fails")
    }
}

/// `test_get_response_error_redacts_exception_from_logs` and
/// `test_stream_response_error_redacts_exception_from_logs`.
#[tokio::test]
async fn model_request_failures_redact_the_error_from_logs() {
    let _lock = LOCK.lock().await;
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    let base = failing_server().await;
    for streamed in [false, true] {
        let _f = flags(true, false);
        failing_request(streamed, &base).await;
        let logged = error_records();
        assert_eq!(logged.len(), 1, "streamed={streamed}: {logged:?}");
        let expected = if streamed {
            "Error streaming response"
        } else {
            "Error getting response"
        };
        assert!(logged[0].contains(expected), "{logged:?}");
        assert!(
            !logged[0].contains(SECRET),
            "streamed={streamed}: {logged:?}"
        );
    }
}

/// `test_get_response_error_logs_exception_when_model_data_enabled`.
#[tokio::test]
async fn model_request_failure_logs_the_error_when_model_data_is_enabled() {
    let _lock = LOCK.lock().await;
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    let base = failing_server().await;
    let _f = flags(false, false);
    failing_request(false, &base).await;
    let logged = error_records();
    assert_eq!(logged.len(), 1, "{logged:?}");
    assert!(logged[0].contains(SECRET), "{logged:?}");
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AgentToolParameters {
    value: i64,
}

fn worker_tool() -> openai_agents_sdk::tool::FunctionTool<()> {
    Agent::<()>::new("worker")
        .as_tool_with(
            AgentAsToolOptions::new()
                .tool_name("worker_tool")
                .tool_description("Runs the worker agent.")
                .parameters::<AgentToolParameters>(),
        )
        .without_failure_error_function()
}

async fn invalid_agent_tool_call() -> AgentsError {
    let tool = worker_tool();
    let payload = format!(r#"{{"value": "{TOOL_ARGUMENT_SECRET}"}}"#);
    let ctx = ToolContext::direct(RunContext::new(()), "1", &tool.name, &payload);
    tool.invoke(ctx, payload).await.unwrap_err()
}

#[tokio::test]
async fn agent_tool_validation_error_redacts_payload_when_tool_data_disabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, true);
    let error = invalid_agent_tool_call().await;
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "Invalid JSON input for tool worker_tool");
    assert!(!error.to_string().contains(TOOL_ARGUMENT_SECRET));
}

#[tokio::test]
async fn agent_tool_validation_error_preserves_diagnostics_when_tool_data_enabled() {
    let _lock = LOCK.lock().await;
    let _f = flags(true, false);
    let error = invalid_agent_tool_call().await;
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains(TOOL_ARGUMENT_SECRET), "{error}");
}

#[allow(dead_code)]
fn _unused(_: Arc<()>) {}
