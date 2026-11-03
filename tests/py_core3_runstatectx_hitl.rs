//! Port of `tests/test_hitl_session_scenario.py`: three approve/reject turns over one session, in
//! memory and against a Conversations endpoint, each turn resuming from the state of the last.
//!
//! Python's scripted model answers from `tool_choice`; here each turn queues its own function call,
//! which is the same script. The Conversations endpoint is a local TCP mock that stores the
//! appended items and returns them in the requested order.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::ToolApprovalItem;
use openai_agents_sdk::memory::{InMemorySession, OpenAIConversationsSession, Session};
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::{DEFAULT_APPROVAL_REJECTION_MESSAGE, FunctionTool};
use openai_agents_sdk::{Agent, OpenAIClient, RunInput, Runner, ToolUseBehavior};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const TOOL_ECHO: &str = "approved_echo";
const TOOL_NOTE: &str = "approved_note";
const USER_MESSAGES: [&str; 3] = [
    "Fetch profile for customer 104.",
    "Update note for customer 104.",
    "Delete note for customer 104.",
];

type Counts = Arc<Mutex<HashMap<String, usize>>>;

fn approval_tools(counts: &Counts) -> Vec<FunctionTool<()>> {
    let echo_counts = counts.clone();
    let echo = FunctionTool::<()>::new(
        TOOL_ECHO,
        "Echoes back the provided query after approval.",
        move |_c, a: Value| {
            let counts = echo_counts.clone();
            async move {
                *counts.lock().unwrap().entry(TOOL_ECHO.into()).or_default() += 1;
                Ok(format!(
                    "approved:{}",
                    a["query"].as_str().unwrap_or_default()
                ))
            }
        },
    )
    .with_needs_approval(true);
    let note_counts = counts.clone();
    let note = FunctionTool::<()>::new(
        TOOL_NOTE,
        "Records the provided query after approval.",
        move |_c, a: Value| {
            let counts = note_counts.clone();
            async move {
                *counts.lock().unwrap().entry(TOOL_NOTE.into()).or_default() += 1;
                Ok(format!(
                    "approved_note:{}",
                    a["query"].as_str().unwrap_or_default()
                ))
            }
        },
    )
    .with_needs_approval(true);
    vec![echo, note]
}

#[derive(Clone, Copy)]
struct Step {
    tool: &'static str,
    message: &'static str,
    approve: bool,
    expected: &'static str,
}

fn steps() -> [Step; 3] {
    [
        Step {
            tool: TOOL_ECHO,
            message: USER_MESSAGES[0],
            approve: true,
            expected: "approved:Fetch profile for customer 104.",
        },
        Step {
            tool: TOOL_NOTE,
            message: USER_MESSAGES[1],
            approve: true,
            expected: "approved_note:Update note for customer 104.",
        },
        Step {
            tool: TOOL_ECHO,
            message: USER_MESSAGES[2],
            approve: false,
            expected: DEFAULT_APPROVAL_REJECTION_MESSAGE,
        },
    ]
}

fn options(session: &Arc<dyn Session>) -> openai_agents_sdk::RunOptions<()> {
    let mut options = openai_agents_sdk::RunOptions::new(())
        .run_config(openai_agents_sdk::run_config::RunConfig::new());
    options.session = Some(session.clone());
    options
}

/// One approve or reject turn: the model asks for the tool, the decision is applied to the state
/// that run left, and the resumed run stops on the tool output. Returns the approval item.
async fn run_step(
    session: Arc<dyn Session>,
    counts: &Counts,
    call_id: &str,
    step: Step,
) -> ToolApprovalItem {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        step.tool,
        json!({"query": step.message}).to_string(),
        call_id,
    )]);
    let mut agent = Agent::<()>::new("Scenario")
        .with_instructions(format!("Always call {} before responding.", step.tool))
        .with_model_instance(model)
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
    agent = agent.with_tools(approval_tools(counts));
    let agent = Arc::new(agent);

    let first = Runner::run_with_options(&agent, step.message, options(&session))
        .await
        .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let approval = first.interruptions[0].clone();
    let mut state = first.to_state();
    if step.approve {
        state.approve(&approval, false);
    } else {
        state.reject(&approval, false, None);
    }
    let resumed = Runner::run_with_options(&agent, RunInput::from(state), options(&session))
        .await
        .unwrap();
    assert!(resumed.interruptions.is_empty());
    assert_eq!(resumed.final_output_text().as_deref(), Some(step.expected));
    approval
}

/// The `function_call` and `function_call_output` items the turn appended, matched on call id.
fn assert_turn_items(items: &[Value], step: Step, approval: &ToolApprovalItem) {
    let users: Vec<&Value> = items.iter().filter(|i| i["role"] == "user").collect();
    let calls: Vec<&Value> = items
        .iter()
        .filter(|i| i["type"] == "function_call")
        .collect();
    let outputs: Vec<&Value> = items
        .iter()
        .filter(|i| i["type"] == "function_call_output")
        .collect();
    assert_eq!(users.len(), 1);
    assert_eq!(calls.len(), 1);
    assert_eq!(outputs.len(), 1);
    assert_eq!(users[0]["content"], step.message);
    assert_eq!(calls[0]["name"], step.tool);
    assert_eq!(calls[0]["call_id"], approval.call_id());
    assert_eq!(outputs[0]["call_id"], approval.call_id());
    assert_eq!(outputs[0]["output"], step.expected);
}

#[tokio::test]
async fn memory_session_hitl_scenario() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let counts: Counts = Arc::default();
    // One session carries the three turns; the Python test rehydrates a new session from the
    // first turn's items, which this shared handle reproduces.
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("memory"));
    for (index, step) in steps().into_iter().enumerate() {
        let approval = run_step(
            session.clone(),
            &counts,
            &format!("call_{}", index + 1),
            step,
        )
        .await;
        let items = session.get_items(None).await.unwrap();
        let items: Vec<Value> = items.into_iter().map(|i| i.0).collect();
        let count = |pred: fn(&Value) -> bool| items.iter().filter(|i| pred(i)).count();
        assert_eq!(count(|i| i["role"] == "user"), index + 1);
        assert_eq!(count(|i| i["type"] == "function_call"), index + 1);
        assert_eq!(count(|i| i["type"] == "function_call_output"), index + 1);
        let last_call = items
            .iter()
            .rev()
            .find(|i| i["type"] == "function_call")
            .unwrap();
        assert_eq!(last_call["call_id"], approval.call_id());
        assert_eq!(last_call["name"], step.tool);
    }
    let counts = counts.lock().unwrap();
    assert_eq!(counts.get(TOOL_ECHO), Some(&1));
    assert_eq!(counts.get(TOOL_NOTE), Some(&1));
}

// ---------------------------------------------------------------------------
// Conversations endpoint: a local mock that stores appended items.
// ---------------------------------------------------------------------------

type Store = Arc<Mutex<Vec<Value>>>;

async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<(String, String, Value)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        let text = String::from_utf8_lossy(&buf).to_string();
        let Some(idx) = text.find("\r\n\r\n") else {
            continue;
        };
        let head = &text[..idx];
        let len: usize = head
            .lines()
            .filter_map(|l| l.split_once(':'))
            .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.trim().parse().ok())
            .unwrap_or(0);
        if buf.len() < idx + 4 + len {
            continue;
        }
        let mut parts = head.split_whitespace();
        let method = parts.next()?.to_string();
        let path = parts.next()?.to_string();
        let body = serde_json::from_slice(&buf[idx + 4..idx + 4 + len]).unwrap_or(Value::Null);
        return Some((method, path, body));
    }
}

async fn serve(store: Store) -> OpenAIClient {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let store = store.clone();
            tokio::spawn(async move {
                let Some((method, path, body)) = read_request(&mut sock).await else {
                    return;
                };
                let reply = if method == "POST" && path.ends_with("/items") {
                    let items = body["items"].as_array().cloned().unwrap_or_default();
                    store.lock().unwrap().extend(items.clone());
                    json!({"object": "list", "data": items, "has_more": false})
                } else if method == "GET" && path.contains("/items") {
                    let desc = path.contains("order=desc");
                    let mut data = store.lock().unwrap().clone();
                    if desc {
                        data.reverse();
                    }
                    json!({"object": "list", "data": data, "has_more": false})
                } else {
                    json!({"id": "conv_test", "object": "conversation"})
                };
                let body = reply.to_string();
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(body.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    OpenAIClient::new("sk-test").with_base_url(format!("http://{addr}/v1"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn openai_conversations_session_hitl_scenario() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let store: Store = Arc::default();
    let client = serve(store.clone()).await;
    let counts: Counts = Arc::default();
    let session: Arc<dyn Session> = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        client.clone(),
    ));
    let mut offset = 0;
    for (index, step) in steps().into_iter().enumerate() {
        let approval = run_step(
            session.clone(),
            &counts,
            &format!("call_{}", index + 1),
            step,
        )
        .await;
        let stored = store.lock().unwrap().clone();
        let turn: Vec<Value> = stored[offset..].to_vec();
        offset = stored.len();
        assert_turn_items(&turn, step, &approval);
    }
    let counts = counts.lock().unwrap();
    assert_eq!(counts.get(TOOL_ECHO), Some(&1));
    assert_eq!(counts.get(TOOL_NOTE), Some(&1));
}
