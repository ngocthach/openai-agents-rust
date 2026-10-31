//! Port of the Python SDK's `tests/test_local_shell_tool.py`.
//!
//! The legacy `local_shell_call` runs through `LocalShellTool`. The resume cases replay the
//! recorded items to a local HTTP mock of the Responses API and read the request body, so the
//! replayed wire shape is checked and not only the in-memory state.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{InputItem, OutputItem, RunItemKind};
use openai_agents_sdk::models::openai_client::OpenAIClient;
use openai_agents_sdk::models::openai_responses::OpenAIResponsesModel;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{LocalShellCommandRequest, LocalShellTool};
use openai_agents_sdk::{Agent, RunState, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

fn local_shell_call(call_id: &str, id: &str) -> Value {
    json!({
        "type": "local_shell_call",
        "id": id,
        "call_id": call_id,
        "status": "completed",
        "action": {
            "type": "exec",
            "command": ["bash", "-c", "echo shell"],
            "env": {},
            "timeout_ms": 1000,
            "working_directory": "/tmp",
        },
    })
}

/// Records every request the executor receives and answers with `output`.
#[derive(Clone, Default)]
struct RecordingExecutor {
    output: String,
    calls: Arc<Mutex<Vec<Value>>>,
}

impl RecordingExecutor {
    fn new(output: &str) -> Self {
        RecordingExecutor {
            output: output.into(),
            calls: Arc::default(),
        }
    }

    fn tool(&self) -> LocalShellTool<()> {
        let this = self.clone();
        LocalShellTool::new(move |req: LocalShellCommandRequest<()>| {
            let this = this.clone();
            async move {
                this.calls.lock().unwrap().push(req.data.clone());
                Ok(this.output.clone())
            }
        })
    }

    fn calls(&self) -> Vec<Value> {
        self.calls.lock().unwrap().clone()
    }
}

/// Runs an agent that calls `local_shell` once and returns the serialized final state.
async fn serialized_local_shell_state(executor: &RecordingExecutor) -> Value {
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(executor.tool())
        .with_model_instance(model.clone());
    model.add_turn(vec![
        item(
            json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "running shell", "annotations": []}]}),
        ),
        item(local_shell_call("call_local_shell", "lsh_test")),
    ]);
    model.add_turn(vec![assistant_message("shell complete")]);
    let result = Runner::run(&agent, "please run shell").await.unwrap();
    let text = result.to_state().to_json_string().unwrap();
    serde_json::from_str(&text).unwrap()
}

/// A local Responses API mock: answers every request with one completed message and records the
/// request bodies.
async fn serve_responses_mock(reply_text: &str) -> (String, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let cap = bodies.clone();
    let reply = json!({
        "id": "resp_mock",
        "object": "response",
        "created_at": 0,
        "model": "codex-mini-latest",
        "status": "completed",
        "parallel_tool_calls": false,
        "tool_choice": "auto",
        "tools": [],
        "output": [{
            "type": "message",
            "id": "msg_mock",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": reply_text, "annotations": []}],
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    })
    .to_string();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let cap = cap.clone();
            let reply = reply.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 8192];
                let body = loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(idx) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let headers: HashMap<String, String> = text[..idx]
                        .lines()
                        .filter_map(|l| l.split_once(':'))
                        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
                        .collect();
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                    }
                };
                cap.lock().unwrap().push(body);
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    reply.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(reply.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/v1"), bodies)
}

fn responses_model(base_url: &str) -> Arc<OpenAIResponsesModel> {
    Arc::new(OpenAIResponsesModel::new(
        "codex-mini-latest",
        OpenAIClient::new("sk-test").with_base_url(base_url),
    ))
}

/// The two schema versions the Python tests parametrize: current, and the v0.19.4 layout.
const SCHEMA_VERSIONS: [Option<&str>; 2] = [None, Some("1.13")];

fn set_schema_version(serialized: &mut Value, version: Option<&str>) {
    if let Some(version) = version {
        serialized["$schemaVersion"] = json!(version);
    }
}

#[tokio::test]
async fn local_shell_action_execute_invokes_executor() {
    let executor = RecordingExecutor::new("test output");
    let model = setup();
    let agent = Agent::<()>::new("test_agent")
        .with_tool(executor.tool())
        .with_model_instance(model.clone());
    model.add_turn(vec![item(local_shell_call("call_456", "lsh_123"))]);
    model.add_turn(vec![assistant_message("done")]);

    let r = Runner::run(&agent, "run").await.unwrap();

    let calls = executor.calls();
    assert_eq!(calls.len(), 1);
    let request = &calls[0];
    assert_eq!(
        request["action"]["command"],
        json!(["bash", "-c", "echo shell"])
    );
    assert_eq!(request["action"]["env"], json!({}));
    assert_eq!(request["action"]["timeout_ms"], 1000);
    assert_eq!(request["action"]["working_directory"], "/tmp");
    assert_eq!(request["call_id"], "call_456");
    assert_eq!(request["id"], "lsh_123");

    let output = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput {
                raw_item, output, ..
            } => Some((raw_item.clone(), output.clone())),
            _ => None,
        })
        .expect("local shell output");
    assert_eq!(output.1, "test output");
    assert_eq!(output.0["type"], "local_shell_call_output");
    assert_eq!(output.0["call_id"], "call_456");
    assert_eq!(output.0["output"], "test output");
}

#[tokio::test]
async fn runner_executes_local_shell_calls() {
    let executor = RecordingExecutor::new("shell result");
    let model = setup();
    let agent = Agent::<()>::new("shell-agent")
        .with_tool(executor.tool())
        .with_model_instance(model.clone());
    let call = local_shell_call("call_local_shell", "lsh_test");
    model.add_turn(vec![
        item(
            json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "running shell", "annotations": []}]}),
        ),
        item(call.clone()),
    ]);
    model.add_turn(vec![item(
        json!({"type": "message", "id": "msg_2", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "shell complete", "annotations": []}]}),
    )]);

    let r = Runner::run(&agent, "please run shell").await.unwrap();

    let calls = executor.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["call_id"], "call_local_shell");

    let items = &r.new_items;
    assert_eq!(items.len(), 4, "{items:?}");
    assert!(matches!(items[0].kind, RunItemKind::MessageOutput { .. }));
    let tool_call_item = match &items[1].kind {
        RunItemKind::ToolCall { raw_item, .. } => raw_item.clone(),
        other => panic!("expected tool call item, got {other:?}"),
    };
    assert_eq!(tool_call_item, call);
    match &items[2].kind {
        RunItemKind::ToolCallOutput {
            raw_item, output, ..
        } => {
            assert_eq!(raw_item["type"], "local_shell_call_output");
            assert_eq!(output.as_str(), Some("shell result"));
        }
        other => panic!("expected tool output item, got {other:?}"),
    }
    assert!(matches!(items[3].kind, RunItemKind::MessageOutput { .. }));

    assert_eq!(r.final_output_text().as_deref(), Some("shell complete"));
    assert_eq!(r.raw_responses.len(), 2);
}

#[tokio::test]
async fn local_shell_output_survives_run_state_resume() {
    for version in SCHEMA_VERSIONS {
        let executor = RecordingExecutor::new("shell result");
        let mut serialized = serialized_local_shell_state(&executor).await;
        set_schema_version(&mut serialized, version);

        let state = RunState::from_json(serialized).unwrap();
        let shell_outputs: Vec<Value> = state
            .generated_items
            .iter()
            .filter_map(|item| match &item.kind {
                RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
                _ => None,
            })
            .filter(|raw| raw["type"] == "local_shell_call_output")
            .collect();
        assert_eq!(
            shell_outputs,
            vec![json!({
                "type": "local_shell_call_output",
                "call_id": "call_local_shell",
                "output": "shell result",
            })],
            "schema {version:?}"
        );

        let (base, bodies) = serve_responses_mock("resumed").await;
        let model = responses_model(&base);
        let agent = Agent::<()>::new("shell-agent")
            .with_tool(executor.tool())
            .with_model_instance(model);
        Runner::run(&agent, state).await.unwrap();

        let requests = bodies.lock().unwrap().clone();
        assert_eq!(requests.len(), 1, "schema {version:?}");
        let replayed: Vec<Value> = requests[0]["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|i| i.is_object())
            .cloned()
            .collect();
        let call = replayed
            .iter()
            .find(|i| i["type"] == "local_shell_call")
            .unwrap_or_else(|| panic!("no replayed local_shell_call in {replayed:?}"));
        let output = replayed
            .iter()
            .find(|i| i["type"] == "local_shell_call_output")
            .unwrap_or_else(|| panic!("no replayed local_shell_call_output in {replayed:?}"));
        assert_eq!(call["call_id"], "call_local_shell");
        assert_eq!(output["call_id"], "call_local_shell");
        assert!(
            output.get("id").is_none(),
            "output must not carry an id: {output}"
        );
    }
}

// An id-only `local_shell_call_output` is dropped by the reader, as Python's strict model does:
// the current schema then fails the completed-invocation check ("... does not match a restored
// tool call and output"), and the 1.13 schema resumes without replaying the pair.
#[tokio::test]
async fn run_state_rejects_id_only_local_shell_output() {
    for version in SCHEMA_VERSIONS {
        let executor = RecordingExecutor::new("shell result");
        let mut serialized = serialized_local_shell_state(&executor).await;
        let invalid_output = json!({
            "type": "local_shell_call_output",
            "id": "legacy-only",
            "output": "shell result",
        });
        for group in ["generated_items", "session_items"] {
            let outputs: Vec<&mut Value> = serialized[group]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .filter(|i| i["raw_item"]["type"] == "local_shell_call_output")
                .collect();
            assert_eq!(outputs.len(), 1, "{group}");
            outputs.into_iter().next().unwrap()["raw_item"] = invalid_output.clone();
        }
        set_schema_version(&mut serialized, version);

        match version {
            None => {
                let err = RunState::from_json(serialized).unwrap_err();
                assert!(
                    err.to_string().contains(
                        "completed tool invocation does not match a restored tool call and output"
                    ),
                    "{err}"
                );
                assert!(!err.to_string().contains("call_local_shell"), "{err}");
            }
            Some(_) => {
                let state = RunState::from_json(serialized).unwrap();
                let (base, bodies) = serve_responses_mock("resumed").await;
                let agent = Agent::<()>::new("shell-agent")
                    .with_tool(executor.tool())
                    .with_model_instance(responses_model(&base));
                Runner::run(&agent, state).await.unwrap();

                let requests = bodies.lock().unwrap().clone();
                assert_eq!(requests.len(), 1);
                let replayed_types: Vec<Value> = requests[0]["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|i| i.get("type").cloned())
                    .collect();
                assert!(!replayed_types.contains(&json!("local_shell_call")));
                assert!(!replayed_types.contains(&json!("local_shell_call_output")));
            }
        }
    }
}

#[tokio::test]
async fn run_state_preserves_official_local_shell_original_input() {
    for version in SCHEMA_VERSIONS {
        let original_input = json!({
            "type": "local_shell_call_output",
            "id": "lsh_output_123",
            "output": "shell result",
        });
        let model = setup();
        model.add_turn(vec![assistant_message("complete")]);
        let agent = Agent::<()>::new("shell-agent").with_model_instance(model.clone());

        let result = Runner::run(&agent, vec![InputItem(original_input.clone())])
            .await
            .unwrap();
        let mut serialized: Value =
            serde_json::from_str(&result.to_state().to_json_string().unwrap()).unwrap();
        set_schema_version(&mut serialized, version);

        let restored = RunState::from_json(serialized).unwrap();
        assert_eq!(
            restored.to_json().unwrap()["original_input"],
            json!([original_input]),
            "schema {version:?}"
        );
    }
}
