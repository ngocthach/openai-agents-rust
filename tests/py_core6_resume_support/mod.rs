//! Shared fixtures for the `py_core6_resume_*` parity tests: a Session whose appends can fail
//! the way Python's `_FailingResumeSession` does, a driver that works streamed or not, and the
//! approval-gated `charge` scenario most of the resume tests start from.
#![allow(dead_code, unused_imports)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::{InputItem, OutputItem, RunInput, ToolApprovalItem};
use openai_agents_sdk::memory::{Session, SessionSettings};
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunOptions, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Default)]
pub struct Store {
    items: Vec<InputItem>,
    failure: Option<&'static str>,
    fail_on_output: Option<&'static str>,
    add_calls: usize,
    forbid_reads: bool,
    fail_nth: Option<usize>,
}

/// Python's `_FailingResumeSession`: the next append fails before writing (`before`), after
/// writing (`after`, a lost acknowledgement) or after writing only its first item (`partial`).
#[derive(Debug)]
pub struct FailingSession {
    id: String,
    store: Mutex<Store>,
}

impl FailingSession {
    pub fn new(id: &str) -> Arc<Self> {
        Arc::new(FailingSession {
            id: id.to_string(),
            store: Mutex::new(Store::default()),
        })
    }

    pub fn fail_next(&self, mode: &'static str) {
        self.store.lock().unwrap().failure = Some(mode);
    }

    pub fn fail_on_output(&self, needle: &'static str) {
        self.store.lock().unwrap().fail_on_output = Some(needle);
    }

    pub fn fail_nth_add(&self, n: usize) {
        self.store.lock().unwrap().fail_nth = Some(n);
    }

    pub fn forbid_reads(&self) {
        self.store.lock().unwrap().forbid_reads = true;
    }

    pub fn items(&self) -> Vec<InputItem> {
        self.store.lock().unwrap().items.clone()
    }

    pub fn push(&self, item: Value) {
        self.store.lock().unwrap().items.push(InputItem(item));
    }
}

#[async_trait]
impl Session for FailingSession {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        if self.store.lock().unwrap().forbid_reads {
            return Err(AgentsError::other("session reconciliation must not run"));
        }
        let items = self.store.lock().unwrap().items.clone();
        Ok(match limit {
            Some(n) if items.len() > n => items[items.len() - n..].to_vec(),
            _ => items,
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let mut store = self.store.lock().unwrap();
        store.add_calls += 1;
        if store.fail_nth == Some(store.add_calls) {
            store.fail_nth = None;
            return Err(AgentsError::other("session append failed"));
        }
        if let Some(needle) = store.fail_on_output
            && items.iter().any(|i| i.0.to_string().contains(needle))
        {
            store.fail_on_output = None;
            return Err(AgentsError::other("session append failed"));
        }
        let failure = store.failure.take();
        if failure == Some("before") {
            return Err(AgentsError::other("session append failed"));
        }
        if failure == Some("partial") {
            store.items.extend(items.into_iter().take(1));
            return Err(AgentsError::other("session append failed"));
        }
        store.items.extend(items);
        if failure == Some("after") {
            return Err(AgentsError::other("session append failed"));
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(self.store.lock().unwrap().items.pop())
    }

    async fn clear_session(&self) -> Result<()> {
        self.store.lock().unwrap().items.clear();
        Ok(())
    }
}

pub struct Outcome {
    pub final_output: Option<Value>,
    pub interruptions: Vec<ToolApprovalItem>,
    pub state: Option<RunState>,
    pub input_list: Vec<InputItem>,
}

pub async fn drive(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Option<Arc<FailingSession>>,
    streamed: bool,
    slot: &RunCheckpoint,
) -> std::result::Result<Outcome, AgentsError> {
    drive_with(agent, input, session, streamed, slot, |o| o).await
}

/// `drive` with a hook to configure the run options (hooks, turn budget, error handlers).
pub async fn drive_with(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    session: Option<Arc<FailingSession>>,
    streamed: bool,
    slot: &RunCheckpoint,
    configure: impl FnOnce(RunOptions<()>) -> RunOptions<()>,
) -> std::result::Result<Outcome, AgentsError> {
    let mut options = configure(RunOptions::new(()).checkpoint(slot));
    if let Some(session) = session {
        options = options.session(session);
    }
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(Outcome {
            final_output: result.final_output().cloned(),
            interruptions: result.interruptions(),
            state: result.to_state(),
            input_list: result
                .result()
                .map(|r| r.to_input_list())
                .unwrap_or_default(),
        })
    } else {
        let result = Runner::run_with_options(agent, input, options).await?;
        Ok(Outcome {
            final_output: result.final_output.clone(),
            interruptions: result.interruptions.clone(),
            state: Some(result.to_state()),
            input_list: result.to_input_list(),
        })
    }
}

pub fn round_trip(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

pub fn call(name: &str, args: Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

pub fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

#[derive(Deserialize, JsonSchema)]
pub struct Amount {
    amount: i64,
}

pub fn charge_tool(effects: &Arc<Mutex<Vec<i64>>>) -> FunctionTool<()> {
    let effects = effects.clone();
    FunctionTool::new("charge", "", move |_c, a: Amount| {
        let effects = effects.clone();
        async move {
            effects.lock().unwrap().push(a.amount);
            Ok("receipt-7".to_string())
        }
    })
    .with_needs_approval(true)
}

pub fn call_pair(items: &[InputItem], call_id: &str) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.0.get("call_id").and_then(Value::as_str) == Some(call_id))
        .filter_map(|i| i.0.get("type").and_then(Value::as_str).map(str::to_string))
        .collect()
}

pub fn charge_pair(items: &[InputItem]) -> Vec<String> {
    call_pair(items, "charge-1")
}

pub const PAIR: [&str; 2] = ["function_call", "function_call_output"];

pub fn pair() -> Vec<String> {
    PAIR.iter().map(|s| s.to_string()).collect()
}

pub struct Paused {
    pub agent: Agent<()>,
    pub model: Arc<ScriptedModel>,
    pub session: Arc<FailingSession>,
    pub state: RunState,
    pub effects: Arc<Mutex<Vec<i64>>>,
}

/// Python's `_approved_session_state`.
pub async fn approved_session_state(streamed: bool) -> Paused {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::new()));
    let model = scripted(vec![
        vec![call("charge", json!({"amount": 7}), "charge-1")],
        vec![assistant_message("done")],
        vec![assistant_message("fresh")],
    ]);
    let agent = Agent::<()>::new("payment")
        .with_model_instance(model.clone())
        .with_tool(charge_tool(&effects));
    let session = FailingSession::new("resume");
    let paused = drive(
        &agent,
        "charge 7",
        Some(session.clone()),
        streamed,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    let mut state = round_trip(paused.state.as_ref().unwrap());
    let interruption = state.get_interruptions()[0].clone();
    state.approve(&interruption, false);
    Paused {
        agent,
        model,
        session,
        state,
        effects,
    }
}

pub fn is_session_failure(error: &AgentsError) -> bool {
    error.to_string().contains("session append failed")
}

/// A resumed approval whose append failed with `failure`, left as a checkpoint.
pub async fn failed_resume(
    p: &Paused,
    state: RunState,
    streamed: bool,
    failure: &'static str,
) -> RunState {
    p.session.fail_next(failure);
    let slot = RunCheckpoint::new();
    let err = drive(&p.agent, state, Some(p.session.clone()), streamed, &slot)
        .await
        .err()
        .expect("the append must fail");
    assert!(is_session_failure(&err), "{err}");
    slot.take().expect("a failed run leaves its checkpoint")
}

pub fn assert_user_error(err: AgentsError, needle: &str) {
    assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
    assert!(err.to_string().contains(needle), "{needle}: {err}");
}
