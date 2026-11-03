//! Ports of the cancellation cases of `test_agent_runner.py` / `test_agent_runner_streamed.py`
//! that were filed as Python-only because they cancel an asyncio task. The Rust counterparts of
//! that cancellation are dropping the run future (aborting its task) and `CancelMode::Immediate`
//! on a streamed run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::error::Result as AgentsResult;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{ReasoningItemIdPolicy, RunConfig};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputItem, OutputGuardrail, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

/// A list session whose appends can be made to block once a flag is set.
#[derive(Debug)]
struct GatedSession {
    inner: InMemorySession,
    block_adds: AtomicBool,
    add_started: Notify,
    blocked_adds: Mutex<usize>,
}

impl GatedSession {
    fn new() -> Arc<Self> {
        Arc::new(GatedSession {
            inner: InMemorySession::new("gated"),
            block_adds: AtomicBool::new(false),
            add_started: Notify::new(),
            blocked_adds: Mutex::new(0),
        })
    }

    async fn kinds(&self) -> Vec<String> {
        self.inner
            .get_items(None)
            .await
            .unwrap()
            .into_iter()
            .map(|i| {
                i.0["type"]
                    .as_str()
                    .or_else(|| i.0["role"].as_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect()
    }
}

#[async_trait]
impl Session for GatedSession {
    fn session_id(&self) -> &str {
        "gated"
    }
    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        if self.block_adds.load(Ordering::SeqCst) {
            *self.blocked_adds.lock().unwrap() += 1;
            self.add_started.notify_one();
            std::future::pending::<()>().await;
        }
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> AgentsResult<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> AgentsResult<()> {
        self.inner.clear_session().await
    }
}

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn parked_guardrail(entered: Arc<Notify>) -> OutputGuardrail<()> {
    OutputGuardrail::new("parked", move |_c, _a, _o: Value| {
        let entered = entered.clone();
        async move {
            entered.notify_one();
            std::future::pending::<()>().await;
            Ok(GuardrailFunctionOutput::new(Value::Null, false))
        }
    })
}

async fn drain(result: &mut openai_agents_sdk::result::RunResultStreaming<()>) {
    let mut events = std::pin::pin!(result.stream_events());
    while events.next().await.is_some() {}
}

/// Python: `test_output_guardrail_cancellation_preserves_final_output_in_session`. A run whose
/// future is dropped while its output guardrail is running keeps the final message in the
/// session: the guardrail never finished, so nothing rejected it.
#[tokio::test]
async fn a_cancelled_output_guardrail_keeps_the_final_message_in_the_session() {
    let model = setup();
    model.add_turn(vec![assistant_message(
        "preserved_on_guardrail_cancellation",
    )]);
    let entered = Arc::new(Notify::new());
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_output_guardrail(parked_guardrail(entered.clone())),
    );
    let session = GatedSession::new();
    let task = {
        let session = session.clone();
        tokio::spawn(async move {
            let _ = Runner::run_with_options(
                agent,
                "user_message",
                RunOptions::new(()).session(session as Arc<dyn Session>),
            )
            .await;
        })
    };
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .expect("the guardrail never started");
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    // The dropped run hands its write to a detached task: wait for it to land.
    for _ in 0..200 {
        if session.kinds().await == ["user", "message"] {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(session.kinds().await, ["user", "message"]);
}

/// Python: `test_streamed_cancel_during_output_guardrail_starts_no_final_turn_write`. An
/// immediate cancel while the output guardrail is parked ends the stream without starting a
/// final-turn write; the tool ran once and only the user input is saved.
#[tokio::test]
async fn an_immediate_cancel_during_an_output_guardrail_starts_no_final_turn_write() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("assistant-preamble"),
        function_call_with_id("commit_tool", "{}", "call-cancel"),
    ]);
    let runs = Arc::new(Mutex::new(0usize));
    let tool = {
        let runs = runs.clone();
        FunctionTool::<()>::new("commit_tool", "a tool", move |_ctx, _a: NoArgs| {
            let runs = runs.clone();
            async move {
                *runs.lock().unwrap() += 1;
                Ok("committed-result".to_string())
            }
        })
    };
    let entered = Arc::new(Notify::new());
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_tool(tool)
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .with_output_guardrail(parked_guardrail(entered.clone())),
    );
    let session = GatedSession::new();
    let mut result = Runner::run_streamed_with_options(
        agent,
        "Use commit_tool",
        RunOptions::new(()).session(session.clone() as Arc<dyn Session>),
    );
    let handle = result.cancel_handle();
    let consumer = tokio::spawn(async move { drain(&mut result).await });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .expect("the guardrail never started");
    // A final-turn write after the cancel would block here and hang the stream.
    session.block_adds.store(true, Ordering::SeqCst);
    handle.cancel(CancelMode::Immediate);
    tokio::time::timeout(Duration::from_secs(5), consumer)
        .await
        .expect("the cancelled stream never ended")
        .unwrap();
    assert_eq!(*session.blocked_adds.lock().unwrap(), 0);
    assert_eq!(*runs.lock().unwrap(), 1);
    assert_eq!(session.kinds().await, ["user"]);
}

/// Python: `test_public_immediate_cancel_during_guardrail_recovery_save_stays_prompt`. After a
/// failing output guardrail the recovery save is in flight; a public immediate cancel still ends
/// the stream promptly without surfacing an error, and only the user input is saved.
#[tokio::test]
async fn an_immediate_cancel_during_the_guardrail_recovery_save_stays_prompt() {
    let model = setup();
    model.add_turn(vec![assistant_message("assistant-preamble")]);
    let session = GatedSession::new();
    let failing = {
        let session = session.clone();
        OutputGuardrail::new("failing", move |_c, _a, _o: Value| {
            let session = session.clone();
            async move {
                session.block_adds.store(true, Ordering::SeqCst);
                Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail failed"))
            }
        })
    };
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_output_guardrail(failing),
    );
    let mut result = Runner::run_streamed_with_options(
        agent,
        "Hello",
        RunOptions::new(()).session(session.clone() as Arc<dyn Session>),
    );
    let handle = result.cancel_handle();
    let consumer = tokio::spawn(async move { drain(&mut result).await });
    tokio::time::timeout(Duration::from_secs(5), session.add_started.notified())
        .await
        .expect("the recovery save never started");
    handle.cancel(CancelMode::Immediate);
    tokio::time::timeout(Duration::from_secs(5), consumer)
        .await
        .expect("the cancelled stream never ended")
        .unwrap();
    assert_eq!(session.kinds().await, ["user"]);
}

fn reasoning(id: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "reasoning", "id": id,
        "summary": [{"type": "summary_text", "text": "Thinking..."}]
    }))
}

fn find_reasoning(items: &[Value]) -> Option<&Value> {
    items.iter().find(|item| item["type"] == "reasoning")
}

/// Python: `test_streamed_reasoning_item_id_policy_omits_follow_up_reasoning_ids`.
#[tokio::test]
async fn a_streamed_run_with_the_omit_policy_strips_follow_up_reasoning_ids() {
    let model = setup();
    model.add_turn(vec![
        reasoning("rs_stream"),
        function_call_with_id("foo", r#"{"a":"b"}"#, "call_stream"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let tool = FunctionTool::<()>::new("foo", "a tool", |_ctx, _a: Value| async {
        Ok("tool_result".to_string())
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool),
    );
    let config = RunConfig::default().with_reasoning_item_id_policy(ReasoningItemIdPolicy::Omit);
    let mut result =
        Runner::run_streamed_with_options(agent, "hello", RunOptions::new(()).run_config(config));
    drain(&mut result).await;
    let result = result.into_result().expect("the stream finished");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let sent: Vec<Value> = model
        .calls()
        .last()
        .unwrap()
        .input
        .iter()
        .map(|i| i.0.clone())
        .collect();
    let second_request_reasoning = find_reasoning(&sent).expect("reasoning replayed");
    assert!(second_request_reasoning.get("id").is_none());
    let history: Vec<Value> = result.to_input_list().into_iter().map(|i| i.0).collect();
    assert!(
        find_reasoning(&history)
            .expect("reasoning in history")
            .get("id")
            .is_none()
    );
}
