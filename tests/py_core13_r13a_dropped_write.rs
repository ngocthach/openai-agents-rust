//! The session write of a run dropped while its output guardrails ran
//! (`test_output_guardrail_cancellation_preserves_final_output_in_session`) cannot report its own
//! failure to the dropped caller, as Python's shielded `finally` can. The failure is instead
//! observable: it is logged under the data redaction policy and, when the run was given a
//! `RunCheckpoint`, recorded there as a pending session write that a resume settles.
//!
//! What stays impossible: the dropped caller never receives the error, and a run without a
//! checkpoint leaves only the log line.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::error::Result as AgentsResult;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputItem, OutputGuardrail, RunState, Runner,
};
use serde_json::Value;
use tokio::sync::Notify;

/// A list session whose appends fail `failures` times, after the first `passes` have succeeded.
/// The run saves its input before the model is called, so the first append is the input.
#[derive(Debug)]
struct FlakySession {
    id: &'static str,
    inner: InMemorySession,
    passes: AtomicUsize,
    failures: AtomicUsize,
    failed: Notify,
}

impl FlakySession {
    fn new(id: &'static str, passes: usize, failures: usize) -> Arc<Self> {
        Arc::new(FlakySession {
            id,
            inner: InMemorySession::new(id),
            passes: AtomicUsize::new(passes),
            failures: AtomicUsize::new(failures),
            failed: Notify::new(),
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
impl Session for FlakySession {
    fn session_id(&self) -> &str {
        self.id
    }
    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        let passed = self
            .passes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok();
        if !passed
            && self
                .failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        {
            self.failed.notify_one();
            return Err(AgentsError::other("disk full: secret-token-123"));
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

static LOGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        LOGS.lock().unwrap().push(format!("{}", record.args()));
    }
    fn flush(&self) {}
}

fn failure_logs() -> Vec<String> {
    LOGS.lock()
        .unwrap()
        .iter()
        .filter(|line| line.contains("dropped during its output guardrails"))
        .cloned()
        .collect()
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

/// Drop a run while its output guardrail is parked, and wait for the detached write to fail.
async fn drop_the_run(
    session: &Arc<FlakySession>,
    checkpoint: Option<&RunCheckpoint>,
    model: &Arc<FakeModel>,
) {
    let entered = Arc::new(Notify::new());
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_output_guardrail(parked_guardrail(entered.clone())),
    );
    let mut options = RunOptions::new(()).session(session.clone() as Arc<dyn Session>);
    if let Some(checkpoint) = checkpoint {
        options = options.checkpoint(checkpoint);
    }
    let task = tokio::spawn(async move {
        let result = Runner::run_with_options(agent, "user_message", options).await;
        let _ = result;
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .expect("the guardrail never started");
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), session.failed.notified())
        .await
        .expect("the detached write never ran");
    // The failure handler runs just after the append returns.
    tokio::time::sleep(Duration::from_millis(300)).await;
}

/// A failed detached write is logged without the error text (the default data redaction
/// policy; running with `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` or `..._TOOL_DATA` set to `0` shows
/// the text and fails that one assertion), records itself as the checkpoint's
/// pending session write, and a resume from that checkpoint appends the turn exactly once. A run
/// without a checkpoint leaves only the log line.
#[tokio::test]
async fn a_failed_detached_write_is_observable_and_recoverable() {
    log::set_logger(&Capture).ok();
    log::set_max_level(log::LevelFilter::Trace);
    openai_agents_sdk::tracing::set_tracing_disabled(true);

    // Without a checkpoint: the log line is all there is.
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("first")]);
    let orphan = FlakySession::new("orphan", 1, 1);
    drop_the_run(&orphan, None, &model).await;
    let lines = failure_logs();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(!lines[0].contains("secret-token-123"), "{lines:?}");
    assert_eq!(orphan.kinds().await, ["user"], "only the input was saved");

    // With a checkpoint: the failed write rides along as a pending session write.
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("second")]);
    let session = FlakySession::new("recorded", 1, 1);
    let checkpoint = RunCheckpoint::new();
    drop_the_run(&session, Some(&checkpoint), &model).await;
    assert_eq!(failure_logs().len(), 2);
    assert_eq!(session.kinds().await, ["user"], "the final append failed");
    let state = checkpoint
        .take()
        .expect("the dropped run left a checkpoint");
    let pending = state
        .pending_session_write
        .clone()
        .expect("the failed write is recorded");
    assert_eq!(pending["session_id"], "recorded");
    assert_eq!(pending["items"].as_array().map(Vec::len), Some(1));
    assert_eq!(pending["items"][0]["role"], "assistant");
    assert_eq!(pending["before"].as_array().map(Vec::len), Some(1));
    assert!(!pending.to_string().contains("secret-token-123"));

    // It survives serialization, and a resume settles it against the (now healthy) session.
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    model.add_turn(vec![assistant_message("resumed")]);
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    Runner::run_with_options(
        agent,
        restored,
        RunOptions::new(()).session(session.clone() as Arc<dyn Session>),
    )
    .await
    .expect("the resume settles the write");

    let kinds = session.kinds().await;
    assert_eq!(
        kinds.iter().filter(|k| *k == "user").count(),
        1,
        "the user turn is appended once: {kinds:?}"
    );
    assert_eq!(&kinds[..2], ["user", "message"], "{kinds:?}");
    assert_eq!(
        kinds.iter().filter(|k| *k == "message").count(),
        2,
        "the lost message once, then the resumed turn's: {kinds:?}"
    );
}

/// A detached write that succeeds leaves no pending write behind.
#[tokio::test]
async fn a_successful_detached_write_records_nothing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("kept")]);
    let session = FlakySession::new("healthy", 1, 0);
    let checkpoint = RunCheckpoint::new();
    let entered = Arc::new(Notify::new());
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model)
            .with_output_guardrail(parked_guardrail(entered.clone())),
    );
    let options = RunOptions::new(())
        .session(session.clone() as Arc<dyn Session>)
        .checkpoint(&checkpoint);
    let task = tokio::spawn(async move {
        let _ = Runner::run_with_options(agent, "user_message", options).await;
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .expect("the guardrail never started");
    task.abort();
    let _ = task.await;
    for _ in 0..200 {
        if session.kinds().await == ["user", "message"] {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert_eq!(session.kinds().await, ["user", "message"]);
    let state = checkpoint
        .take()
        .expect("the dropped run left a checkpoint");
    assert!(state.pending_session_write.is_none());
}
