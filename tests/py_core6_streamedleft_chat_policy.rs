//! Output guardrails over a Chat Completions model that is handed server-managed conversation
//! state (`test_agent_runner_streamed.py::test_chat_completions_output_guardrails_use_adapter_conversation_policy`,
//! 8 cases: mode x strict x agent-model/run-config-model). The model owns the policy: default
//! mode warns and drops `previous_response_id`, strict mode refuses before any request.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::models::{OpenAIChatCompletionsModel, OpenAIClient};
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::FakeModel;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, Runner,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Value,
}

#[derive(Clone)]
struct Resp {
    status: u16,
    content_type: &'static str,
    headers: Vec<(String, String)>,
    body: String,
}

fn json_resp(v: Value) -> Resp {
    Resp {
        status: 200,
        content_type: "application/json",
        headers: vec![("x-request-id".into(), "req_default".into())],
        body: v.to_string(),
    }
}

fn sse_resp(chunks: &[Value]) -> Resp {
    let mut body = String::new();
    for c in chunks {
        body.push_str(&format!("data: {c}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    Resp {
        status: 200,
        content_type: "text/event-stream",
        headers: vec![("x-request-id".into(), "req_default".into())],
        body,
    }
}

type Captured = Arc<Mutex<Vec<Req>>>;

/// Serve `handler` on a local port; returns the `/v1` base URL and every captured request.
async fn serve<F>(handler: F) -> (String, Captured)
where
    F: Fn(&Req) -> Resp + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let cap = cap.clone();
            let handler = handler.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                let req = loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(idx) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let head = &text[..idx];
                    let mut lines = head.lines();
                    let mut first = lines.next().unwrap().split_whitespace();
                    let method = first.next().unwrap().to_string();
                    let path = first.next().unwrap().to_string();
                    let headers: HashMap<String, String> = lines
                        .filter_map(|l| l.split_once(':'))
                        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
                        .collect();
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        let body = serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                        break Req {
                            method,
                            path,
                            headers,
                            body,
                        };
                    }
                };
                cap.lock().unwrap().push(req.clone());
                let r = handler(&req);
                let mut head = format!(
                    "HTTP/1.1 {} X\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n",
                    r.status,
                    r.content_type,
                    r.body.len()
                );
                for (k, v) in &r.headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(r.body.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/v1"), captured)
}

/// Answer requests in order; the last reply repeats once the queue is exhausted.
async fn serve_queue(replies: Vec<Resp>) -> (String, Captured) {
    let queue = Mutex::new(VecDeque::from(replies));
    serve(move |_| {
        let mut q = queue.lock().unwrap();
        if q.len() > 1 {
            q.pop_front().unwrap()
        } else {
            q.front().unwrap().clone()
        }
    })
    .await
}

fn model_at(base: &str, strict: bool) -> Arc<OpenAIChatCompletionsModel> {
    Arc::new(
        OpenAIChatCompletionsModel::new("test", OpenAIClient::new("sk-test").with_base_url(base))
            .with_strict_feature_validation(strict),
    )
}

// ---- warning capture -----------------------------------------------------------------

#[derive(Default)]
struct LogSink(Mutex<Vec<String>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let s = SINK.get_or_init(LogSink::default);
    log::set_logger(s).ok();
    log::set_max_level(log::LevelFilter::Trace);
    s
}

/// Serializes tests that assert on captured log lines.
fn log_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn log_text() -> String {
    sink().0.lock().unwrap().join("\n")
}

fn guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("pass", |_ctx, _agent, _output| async move {
        Ok(GuardrailFunctionOutput {
            output_info: Value::Null,
            tripwire_triggered: false,
        })
    })
}

#[tokio::test]
async fn chat_completions_output_guardrails_use_adapter_conversation_policy() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let _guard = log_lock().lock().await;
    for streamed in [false, true] {
        for strict in [false, true] {
            for use_run_config_model in [false, true] {
                let label =
                    format!("streamed={streamed} strict={strict} run_model={use_run_config_model}");
                sink().0.lock().unwrap().clear();
                let reply = if streamed {
                    sse_resp(&[
                        json!({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "t",
                               "choices": [{"index": 0, "delta": {"role": "assistant", "content": "accepted-output"}, "finish_reason": null}]}),
                        json!({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "t",
                               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                    ])
                } else {
                    json_resp(
                        json!({"id": "c", "created": 0, "model": "t", "object": "chat.completion",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "accepted-output"}}]}),
                    )
                };
                let (base, captured) = serve_queue(vec![reply]).await;
                let chat = model_at(&base, strict);
                let unused = Arc::new(FakeModel::new());
                let mut agent = Agent::<()>::new("test").with_output_guardrail(guardrail());
                let mut config = RunConfig::default();
                if use_run_config_model {
                    agent = agent.with_model_instance(unused.clone());
                    config.model = Some(chat.clone().into());
                } else {
                    agent = agent.with_model_instance(chat.clone());
                }
                let agent = Arc::new(agent);
                let options = RunOptions::new(())
                    .previous_response_id("response-id")
                    .run_config(config);
                let outcome: Result<String, AgentsError> = if streamed {
                    let mut stream =
                        Runner::run_streamed_with_options(agent.clone(), "hello", options);
                    let mut error = None;
                    {
                        let mut events = std::pin::pin!(stream.stream_events());
                        while let Some(ev) = events.next().await {
                            if let Err(e) = ev {
                                error = Some(e);
                            }
                        }
                    }
                    match error {
                        Some(e) => Err(e),
                        None => Ok(stream
                            .into_result()
                            .expect("finished")
                            .final_output_text()
                            .unwrap_or_default()),
                    }
                } else {
                    Runner::run_with_options(agent.clone(), "hello", options)
                        .await
                        .map(|r| r.final_output_text().unwrap_or_default())
                };
                assert!(unused.calls().is_empty(), "{label}");
                if strict {
                    let err = outcome.unwrap_err().to_string();
                    assert!(
                        err.contains("OpenAIChatCompletionsModel does not support"),
                        "{label}: {err}"
                    );
                    assert!(captured.lock().unwrap().is_empty(), "{label}");
                } else {
                    assert_eq!(outcome.unwrap(), "accepted-output", "{label}");
                    assert!(
                        log_text()
                            .contains("Ignoring unsupported server-managed conversation state"),
                        "{label}: {}",
                        log_text()
                    );
                    let requests = captured.lock().unwrap();
                    assert_eq!(requests.len(), 1, "{label}");
                    assert!(
                        requests[0].body.get("previous_response_id").is_none(),
                        "{label}"
                    );
                }
            }
        }
    }
}
