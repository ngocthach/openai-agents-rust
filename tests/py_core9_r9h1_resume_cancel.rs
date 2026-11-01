//! Ports of the cancellation cases of the Python SDK's `tests/test_run_impl_resume_paths.py`.
//!
//! Python cancels an asyncio task while a Session append or a compaction replacement is in
//! flight; the Rust counterpart of that cancellation is dropping the run future (aborting its
//! task) or cancelling a streamed run with `CancelMode::Immediate`. A held Session write is
//! controlled through a public `Session` wrapper instead of monkeypatching SQLite's worker.

mod py_mcpmem_support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::{InputItem, OutputItem, RunInput, RunItemKind};
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionSession, Session, SessionSettings,
};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, OpenAIClient, RunOptions, RunState, Runner};
use py_mcpmem_support::{Request, mock_openai_with};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;

fn call(name: &str, args: Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

fn round_trip(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

/// A store whose append of one chosen batch commits, then holds its acknowledgement until
/// released. That is the point Python's SQLite worker is paused at: the rows are written, the
/// caller has not been told.
#[derive(Debug)]
struct HeldStore {
    id: String,
    items: Mutex<Vec<InputItem>>,
    hold_batch_containing: Mutex<Option<&'static str>>,
    held: Notify,
    release: Notify,
    batches: Mutex<Vec<Vec<InputItem>>>,
}

impl HeldStore {
    fn new(id: &str) -> Arc<Self> {
        Arc::new(HeldStore {
            id: id.to_string(),
            items: Mutex::new(Vec::new()),
            hold_batch_containing: Mutex::new(None),
            held: Notify::new(),
            release: Notify::new(),
            batches: Mutex::new(Vec::new()),
        })
    }

    fn stored(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }
}

#[async_trait]
impl Session for HeldStore {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let items = self.stored();
        Ok(match limit {
            Some(n) if items.len() > n => items[items.len() - n..].to_vec(),
            _ => items,
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        self.batches.lock().unwrap().push(items.clone());
        let hold = {
            let needle = *self.hold_batch_containing.lock().unwrap();
            needle.is_some_and(|n| items.iter().any(|i| i.0.to_string().contains(n)))
        };
        self.items.lock().unwrap().extend(items);
        if hold {
            *self.hold_batch_containing.lock().unwrap() = None;
            self.held.notify_one();
            self.release.notified().await;
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self.items.lock().unwrap().pop())
    }

    async fn clear_session(&self) -> Result<()> {
        self.items.lock().unwrap().clear();
        Ok(())
    }
}

type Captured = Arc<Mutex<Vec<Request>>>;

async fn compaction_over(
    backend: Arc<dyn Session>,
    output: Option<Vec<Value>>,
    trigger: impl Fn(&openai_agents_sdk::memory::CompactionContext) -> bool + Send + Sync + 'static,
) -> (Arc<OpenAIResponsesCompactionSession>, Captured) {
    let (base, captured) = mock_openai_with(move |req: &Request| {
        let out = match &output {
            Some(fixed) => fixed.clone(),
            None => req.body["input"].as_array().cloned().unwrap_or_default(),
        };
        json!({"id": "cmp_1", "output": out, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})
    })
    .await;
    let session = OpenAIResponsesCompactionSession::new(backend.session_id().to_string(), backend)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_compaction_mode(CompactionMode::Input)
        .with_should_trigger_compaction(trigger);
    (Arc::new(session), captured)
}

fn compact_inputs(captured: &Captured) -> Vec<Vec<Value>> {
    captured
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.path.ends_with("/responses/compact"))
        .map(|r| r.body["input"].as_array().cloned().unwrap_or_default())
        .collect()
}

fn dyn_session(s: &Arc<OpenAIResponsesCompactionSession>) -> Arc<dyn Session> {
    s.clone() as Arc<dyn Session>
}

fn response_step(output: Vec<OutputItem>, id: &str) -> ScriptedStep {
    ScriptedStep::output(output).with_response_id(Some(id.to_string()))
}

async fn drain(streamed: &mut openai_agents_sdk::result::RunResultStreaming<()>) {
    let mut events = std::pin::pin!(streamed.stream_events());
    while let Some(event) = events.next().await {
        let _ = event;
    }
}

async fn run_to_end(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Arc<dyn Session>,
    streamed: bool,
) -> (Option<Value>, Option<RunState>) {
    let options = RunOptions::new(()).session(session);
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        (result.final_output().cloned(), result.to_state())
    } else {
        let result = Runner::run_with_options(agent, input, options)
            .await
            .unwrap();
        (result.final_output.clone(), Some(result.to_state()))
    }
}

/// Python: `test_handoff_resume_after_cancelled_compaction_commits`. The compaction
/// replacement is committed but not acknowledged when the streamed run is cancelled; the
/// cancelled run leaves no pending write, and resuming it finishes the handoff once without
/// compacting again.
#[tokio::test]
async fn handoff_resume_after_cancelled_compaction_commits() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for round_trip_state in [false, true] {
        for retry_streamed in [false, true] {
            let label = format!("json={round_trip_state} streamed={retry_streamed}");
            let backend = HeldStore::new("cancelled-compaction");
            *backend.hold_batch_containing.lock().unwrap() = Some("compacted history");
            let compacted =
                json!({"type": "message", "role": "assistant", "content": "compacted history"});
            let (session, captured) =
                compaction_over(backend.clone(), Some(vec![compacted.clone()]), |ctx| {
                    ctx.response_id.as_deref() == Some("resp-handoff")
                })
                .await;

            let model = Arc::new(ScriptedModel::new());
            model
                .enqueue(response_step(
                    vec![
                        assistant_message("delegating"),
                        call("transfer_to_delegate", json!({}), "handoff-1"),
                    ],
                    "resp-handoff",
                ))
                .unwrap();
            model
                .enqueue(response_step(
                    vec![assistant_message("done")],
                    "resp-delegate",
                ))
                .unwrap();
            let handoff_calls = Arc::new(Mutex::new(Vec::<String>::new()));
            let seen = handoff_calls.clone();
            let delegate =
                Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
            let route = handoff(delegate)
                .with_on_handoff(move |_c| {
                    let seen = seen.clone();
                    async move {
                        seen.lock().unwrap().push("handoff".into());
                        Ok(())
                    }
                })
                .with_input_filter(|mut data| {
                    data.new_items
                        .retain(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }));
                    data
                });
            let triage = Agent::<()>::new("triage")
                .with_model_instance(model.clone())
                .with_handoff(route);

            let options = RunOptions::new(()).session(dyn_session(&session));
            let mut streamed = Runner::run_streamed_with_options(&triage, "hello", options);
            let handle = streamed.cancel_handle();
            let consumer = tokio::spawn(async move {
                drain(&mut streamed).await;
                streamed
            });
            tokio::time::timeout(Duration::from_secs(10), backend.held.notified())
                .await
                .unwrap_or_else(|_| panic!("{label}: the replacement never started"));
            handle.cancel(CancelMode::Immediate);
            backend.release.notify_waiters();
            backend.release.notify_one();
            let streamed = tokio::time::timeout(Duration::from_secs(10), consumer)
                .await
                .unwrap_or_else(|_| panic!("{label}: the cancelled stream never ended"))
                .unwrap();

            assert_eq!(
                backend.stored(),
                vec![InputItem(compacted.clone())],
                "{label}"
            );
            assert_eq!(model.calls().len(), 1, "{label}");
            let mut state = streamed
                .to_state()
                .expect("a cancelled run keeps its state");
            assert!(state.pending_session_write.is_none(), "{label}");
            if round_trip_state {
                state = round_trip(&state);
            }
            let (final_output, resumed_state) =
                run_to_end(&triage, state, dyn_session(&session), retry_streamed).await;
            assert_eq!(final_output, Some(json!("done")), "{label}");
            assert_eq!(model.calls().len(), 2, "{label}");
            assert_eq!(*handoff_calls.lock().unwrap(), vec!["handoff"], "{label}");
            assert_eq!(compact_inputs(&captured).len(), 1, "{label}");
            let stored = backend.stored();
            assert_eq!(stored.len(), 2, "{label}: {stored:?}");
            assert_eq!(stored[0].0, compacted, "{label}");
            assert_eq!(stored[1].0["role"], json!("assistant"), "{label}");
            assert!(
                resumed_state
                    .map(|s| s.pending_session_write.is_none())
                    .unwrap_or(true),
                "{label}"
            );
        }
    }
}

/// Python: `test_cancelled_compaction_append_preserves_committed_and_surviving_writes`. An
/// append cancelled after the backend committed it keeps its write, a writer queued behind it
/// still lands, and the forced compaction sees both in order.
#[tokio::test]
async fn cancelled_compaction_append_preserves_committed_and_surviving_writes() {
    let backend = HeldStore::new("cancelled-append");
    let (session, captured) = compaction_over(backend.clone(), None, |_| false).await;
    session
        .run_compaction(openai_agents_sdk::memory::OpenAIResponsesCompactionArgs::default())
        .await
        .unwrap();
    let first_item = InputItem(json!({"role": "user", "content": "committed before cancellation"}));
    let newer_item = InputItem(json!({"role": "user", "content": "surviving writer"}));
    *backend.hold_batch_containing.lock().unwrap() = Some("committed before cancellation");

    let first = {
        let session = session.clone();
        let item = first_item.clone();
        tokio::spawn(async move { session.add_items(vec![item]).await })
    };
    tokio::time::timeout(Duration::from_secs(5), backend.held.notified())
        .await
        .expect("the first append never reached the backend");
    let newer = {
        let session = session.clone();
        let item = newer_item.clone();
        tokio::spawn(async move { session.add_items(vec![item]).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!newer.is_finished(), "the newer writer must wait its turn");
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), newer)
        .await
        .expect("the surviving writer never finished")
        .unwrap()
        .unwrap();
    session
        .run_compaction(openai_agents_sdk::memory::OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    let inputs = compact_inputs(&captured);
    assert_eq!(
        inputs.last().unwrap(),
        &vec![first_item.0.clone(), newer_item.0.clone()]
    );
    assert_eq!(backend.stored(), vec![first_item, newer_item]);
}

// ---------------------------------------------------------------------------------------
// Resumed approval whose Session append is cancelled
// ---------------------------------------------------------------------------------------

/// Python's `_FailingResumeSession.block_next_add`: the next append waits, before writing
/// anything, until released.
#[derive(Debug)]
struct BlockingSession {
    id: String,
    items: Mutex<Vec<InputItem>>,
    block_next_add: Mutex<bool>,
    started: Notify,
    release: Notify,
}

impl BlockingSession {
    fn new(id: &str) -> Arc<Self> {
        Arc::new(BlockingSession {
            id: id.to_string(),
            items: Mutex::new(Vec::new()),
            block_next_add: Mutex::new(false),
            started: Notify::new(),
            release: Notify::new(),
        })
    }

    fn stored(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }
}

#[async_trait]
impl Session for BlockingSession {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let items = self.stored();
        Ok(match limit {
            Some(n) if items.len() > n => items[items.len() - n..].to_vec(),
            _ => items,
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let block = std::mem::take(&mut *self.block_next_add.lock().unwrap());
        if block {
            self.started.notify_one();
            self.release.notified().await;
        }
        self.items.lock().unwrap().extend(items);
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self.items.lock().unwrap().pop())
    }

    async fn clear_session(&self) -> Result<()> {
        self.items.lock().unwrap().clear();
        Ok(())
    }
}

#[derive(Deserialize, JsonSchema)]
struct Amount {
    amount: i64,
}

async fn drive(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Arc<dyn Session>,
    streamed: bool,
    slot: &RunCheckpoint,
) -> std::result::Result<Option<Value>, AgentsError> {
    let options = RunOptions::new(()).session(session).checkpoint(slot);
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(result.final_output().cloned())
    } else {
        Ok(Runner::run_with_options(agent, input, options)
            .await?
            .final_output
            .clone())
    }
}

fn call_pair(items: &[InputItem], call_id: &str) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.0.get("call_id").and_then(Value::as_str) == Some(call_id))
        .filter_map(|i| i.0.get("type").and_then(Value::as_str).map(str::to_string))
        .collect()
}

/// Python: `test_resumed_session_append_cancellation_retains_recoverable_state`. While the
/// Session holds the resumed batch's append, a second resume of the same checkpoint is
/// refused; when the first run is dropped, the checkpoint it published still reconciles with
/// the append that lands afterwards and the approved tool is never run again.
#[tokio::test]
async fn resumed_session_append_cancellation_retains_recoverable_state() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        let effects = Arc::new(Mutex::new(Vec::<i64>::new()));
        let model = Arc::new(ScriptedModel::new());
        for step in [
            vec![call("charge", json!({"amount": 7}), "charge-1")],
            vec![assistant_message("done")],
        ] {
            model.enqueue(ScriptedStep::output(step)).unwrap();
        }
        let seen = effects.clone();
        let charge = FunctionTool::new("charge", "", move |_c, a: Amount| {
            let seen = seen.clone();
            async move {
                seen.lock().unwrap().push(a.amount);
                Ok("receipt-7".to_string())
            }
        })
        .with_needs_approval(true);
        let agent = Arc::new(
            Agent::<()>::new("payment")
                .with_model_instance(model.clone())
                .with_tool(charge),
        );
        let session = BlockingSession::new("resume-cancel");
        let dyn_session: Arc<dyn Session> = session.clone();

        let slot = RunCheckpoint::new();
        let paused = {
            let options = RunOptions::new(())
                .session(dyn_session.clone())
                .checkpoint(&slot);
            Runner::run_with_options(&*agent, "charge 7", options)
                .await
                .unwrap()
        };
        let mut state = round_trip(&paused.to_state());
        let interruption = state.get_interruptions()[0].clone();
        state.approve(&interruption, false);

        *session.block_next_add.lock().unwrap() = true;
        let attempt_slot = RunCheckpoint::new();
        // Dropping the future is how a non-streamed run is cancelled; a streamed run executes on
        // its own task, so it is stopped with the stream's immediate cancel.
        let stop: Box<
            dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>,
        > = if streamed {
            let options = RunOptions::new(())
                .session(dyn_session.clone())
                .checkpoint(&attempt_slot);
            let mut result = Runner::run_streamed_with_options(&*agent, state.clone(), options);
            let handle = result.cancel_handle();
            let consumer = tokio::spawn(async move { drain(&mut result).await });
            Box::new(move || {
                Box::pin(async move {
                    handle.cancel(CancelMode::Immediate);
                    consumer.await.unwrap();
                })
            })
        } else {
            let attempt = {
                let agent = agent.clone();
                let session = dyn_session.clone();
                let slot = attempt_slot.clone();
                let state = state.clone();
                tokio::spawn(async move { drive(&agent, state, session, false, &slot).await })
            };
            Box::new(move || {
                Box::pin(async move {
                    attempt.abort();
                    assert!(attempt.await.unwrap_err().is_cancelled());
                })
            })
        };
        tokio::time::timeout(Duration::from_secs(5), session.started.notified())
            .await
            .expect("the resumed append never reached the Session");
        let published = attempt_slot
            .take()
            .expect("the run publishes its checkpoint before the append");

        let refused = drive(
            &agent,
            published.clone(),
            dyn_session.clone(),
            !streamed,
            &RunCheckpoint::new(),
        )
        .await
        .expect_err("a second resume must wait for the first append");
        assert!(
            matches!(refused, AgentsError::UserError { .. })
                && refused
                    .to_string()
                    .contains("pending Session write is already in progress"),
            "streamed={streamed}: {refused}"
        );
        assert_eq!(model.calls().len(), 1, "streamed={streamed}");

        stop().await;
        // The Session is still writing the dropped run's batch, so the checkpoint stays claimed.
        let still_refused = drive(
            &agent,
            published.clone(),
            dyn_session.clone(),
            streamed,
            &RunCheckpoint::new(),
        )
        .await
        .expect_err("the checkpoint stays in progress while the append is held");
        assert!(
            still_refused
                .to_string()
                .contains("pending Session write is already in progress"),
            "streamed={streamed}: {still_refused}"
        );
        session.release.notify_one();
        // The Session finishes the append the dropped run had handed it.
        for _ in 0..200 {
            if call_pair(&session.stored(), "charge-1")
                .contains(&"function_call_output".to_string())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            call_pair(&session.stored(), "charge-1").contains(&"function_call_output".to_string()),
            "streamed={streamed}: the append never landed: {:?}",
            session.stored()
        );

        let restored = round_trip(&published);
        let output = drive(
            &agent,
            restored,
            dyn_session.clone(),
            !streamed,
            &RunCheckpoint::new(),
        )
        .await
        .unwrap();
        assert_eq!(output, Some(json!("done")), "streamed={streamed}");
        assert_eq!(*effects.lock().unwrap(), vec![7], "streamed={streamed}");
        assert_eq!(
            call_pair(&session.stored(), "charge-1"),
            vec![
                "function_call".to_string(),
                "function_call_output".to_string()
            ],
            "streamed={streamed}"
        );
    }
}
