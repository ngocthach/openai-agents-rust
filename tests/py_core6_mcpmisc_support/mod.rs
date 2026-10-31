//! Shared fixtures of the round-6 MCP and memory ports: a mock `responses/compact` endpoint, run
//! helpers, session stacks, and the model-input filters the Python tests pass.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::RunInput;
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionSession, Session, SessionSettings, SqliteSession,
};
use openai_agents_sdk::run_config::ModelInputData;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunConfig, RunOptions, RunResult, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::py_mcpmem_support::{ListSession, Request, mock_openai_with};

pub const LOCAL_OUTPUT: &str = "synthetic-local-only-result";

#[derive(Deserialize, JsonSchema)]
pub struct NoArgs {}

pub fn lookup_tool(needs_approval: bool) -> FunctionTool<()> {
    lookup_returning(LOCAL_OUTPUT, needs_approval)
}

pub fn lookup_returning(result: &'static str, needs_approval: bool) -> FunctionTool<()> {
    FunctionTool::new("lookup", "lookup", move |_ctx, _a: NoArgs| async move {
        Ok(result.to_string())
    })
    .with_needs_approval(needs_approval)
}

/// Python's `IdStrippingSession`: a store that reassigns item ids, so replayed items must be
/// matched without them.
#[derive(Debug)]
pub struct IdStrippingSession(pub ListSession);

#[async_trait::async_trait]
impl Session for IdStrippingSession {
    fn session_id(&self) -> &str {
        self.0.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.0.session_settings()
    }
    fn ignores_item_ids_for_matching(&self) -> bool {
        true
    }
    async fn get_items(
        &self,
        limit: Option<usize>,
    ) -> openai_agents_sdk::error::Result<Vec<InputItem>> {
        self.0.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::error::Result<()> {
        self.0.add_items(items).await
    }
    async fn pop_item(&self) -> openai_agents_sdk::error::Result<Option<InputItem>> {
        self.0.pop_item().await
    }
    async fn clear_session(&self) -> openai_agents_sdk::error::Result<()> {
        self.0.clear_session().await
    }
}

pub fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

pub fn without_tools(items: Vec<InputItem>) -> Vec<InputItem> {
    items
        .into_iter()
        .filter(|i| {
            !matches!(
                i.item_type(),
                Some("function_call") | Some("function_call_output")
            )
        })
        .collect()
}

pub fn text(items: &[InputItem]) -> String {
    items
        .iter()
        .map(|i| i.as_value().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn values(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|i| i.as_value().clone()).collect()
}

pub fn of_type<'a>(items: &'a [Value], kind: &str) -> Vec<&'a Value> {
    items.iter().filter(|i| i["type"] == kind).collect()
}

/// A compaction endpoint whose replies come from `reply(request body)`.
pub struct Compaction {
    base: String,
    captured: Arc<Mutex<Vec<Request>>>,
}

impl Compaction {
    pub async fn replying(output: Vec<Value>) -> Compaction {
        Self::with(move |_| json!({"id": "cmp", "output": output.clone(), "usage": usage()})).await
    }
    pub async fn with(f: impl Fn(&Value) -> Value + Send + Sync + 'static) -> Compaction {
        let (base, captured) = mock_openai_with(move |req| f(&req.body)).await;
        Compaction { base, captured }
    }
    pub fn client(&self) -> OpenAIClient {
        OpenAIClient::new("sk-test").with_base_url(self.base.clone())
    }
    pub fn requests(&self) -> Vec<Value> {
        self.captured
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.path.ends_with("/responses/compact"))
            .map(|r| r.body.clone())
            .collect()
    }
    pub fn inputs(&self) -> Vec<Vec<Value>> {
        self.requests()
            .iter()
            .map(|b| b["input"].as_array().cloned().unwrap_or_default())
            .collect()
    }
}

pub fn usage() -> Value {
    json!({"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})
}

pub fn summary() -> Value {
    json!({"type": "compaction", "id": "cmp-test", "encrypted_content": "synthetic-summary"})
}

pub fn compaction_session(
    id: &str,
    underlying: Arc<dyn Session>,
    endpoint: &Compaction,
    mode: CompactionMode,
) -> OpenAIResponsesCompactionSession {
    OpenAIResponsesCompactionSession::new(id, underlying)
        .unwrap()
        .with_client(endpoint.client())
        .with_compaction_mode(mode)
}

pub fn always(session: OpenAIResponsesCompactionSession) -> OpenAIResponsesCompactionSession {
    session.with_should_trigger_compaction(|_| true)
}

pub async fn run(
    agent: &Agent<()>,
    prompt: impl Into<RunInput>,
    session: &Arc<dyn Session>,
    streamed: bool,
    config: Option<RunConfig<()>>,
) -> RunResult<()> {
    let mut options = RunOptions::new(()).session(session.clone());
    if let Some(config) = config {
        options = options.run_config(config);
    }
    if !streamed {
        return Runner::run_with_options(agent, prompt, options)
            .await
            .unwrap();
    }
    let mut stream = Runner::run_streamed_with_options(agent, prompt, options);
    {
        let events = stream.stream_events();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    stream
        .into_result()
        .expect("a finished stream has a result")
}

pub fn as_session(session: OpenAIResponsesCompactionSession) -> Arc<dyn Session> {
    Arc::new(session)
}

pub fn model_with(turns: Vec<Vec<openai_agents_sdk::items::OutputItem>>) -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    for turn in turns {
        model.add_turn(turn);
    }
    model
}

pub fn texts(turns: &[&str]) -> Vec<Vec<openai_agents_sdk::items::OutputItem>> {
    turns.iter().map(|t| vec![assistant_message(*t)]).collect()
}

pub fn model_input(model: &FakeModel, call: usize) -> Vec<InputItem> {
    model.calls()[call].input.clone()
}

pub fn model_input_values(model: &FakeModel, call: usize) -> Vec<Value> {
    values(&model_input(model, call))
}

pub fn calls_without_id(items: &[Value], call_id: &str) -> Vec<String> {
    items
        .iter()
        .filter(|i| i["call_id"] == call_id)
        .map(|i| i["type"].as_str().unwrap_or_default().to_string())
        .collect()
}

pub fn sqlite(id: &str, limit: Option<usize>) -> Arc<SqliteSession> {
    let session = SqliteSession::in_memory(id).unwrap();
    Arc::new(match limit {
        Some(limit) => session.with_settings(SessionSettings::with_limit(limit)),
        None => session,
    })
}

pub fn limit_100() -> RunConfig<()> {
    RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(100))
}

#[cfg(feature = "encrypt")]
pub fn encrypted(id: &str, inner: Arc<dyn Session>, ttl: u64) -> Arc<dyn Session> {
    Arc::new(
        openai_agents_sdk::memory::EncryptedSession::new(id, inner, "synthetic-test-key", ttl)
            .unwrap(),
    )
}

// ---------------------------------------------------------------------------------------------
// test_automatic_compaction_respects_backend_id_matching_policy (2)
// ---------------------------------------------------------------------------------------------

pub fn count_of(items: &[Value], wanted: &Value) -> usize {
    items.iter().filter(|i| *i == wanted).count()
}

pub fn model_call_filter<F>(f: F) -> RunConfig<()>
where
    F: Fn(Vec<Value>) -> Vec<Value> + Send + Sync + 'static,
{
    let f = Arc::new(f);
    RunConfig::<()>::default().with_call_model_input_filter(move |data| {
        let f = f.clone();
        async move {
            let input = f(values(&data.model_data.input))
                .into_iter()
                .map(InputItem::from_value)
                .collect();
            Ok(ModelInputData {
                input,
                instructions: data.model_data.instructions,
            })
        }
    })
}

pub fn session_callback<F>(f: F) -> RunConfig<()>
where
    F: Fn(Vec<Value>, Vec<Value>) -> Vec<Value> + Send + Sync + 'static,
{
    let f = Arc::new(f);
    RunConfig::<()>::default().with_session_input_callback(move |history, new| {
        let f = f.clone();
        async move {
            Ok(f(values(&history), values(&new))
                .into_iter()
                .map(InputItem::from_value)
                .collect())
        }
    })
}

// ---------------------------------------------------------------------------------------------
// test_backend_read_limit_cannot_hide_retained_history_from_compaction (2 x 2)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "encrypt")]
pub const TTL: u64 = 3;

#[cfg(feature = "encrypt")]
pub async fn let_items_expire() {
    tokio::time::sleep(std::time::Duration::from_millis(4200)).await;
}

/// A compaction session over an encrypted view of `backend` (`outer == false`) or an encrypted
/// session over a compaction session (`outer == true`), with the compaction session handle
/// that manual `run_compaction` calls need.
#[cfg(feature = "encrypt")]
pub fn expiring_stack(
    outer: bool,
    id: &str,
    backend: Arc<dyn Session>,
    endpoint: &Compaction,
    hook: Option<bool>,
) -> (Arc<dyn Session>, Arc<OpenAIResponsesCompactionSession>) {
    let build = |store: Arc<dyn Session>| {
        let session = compaction_session(id, store, endpoint, CompactionMode::Input);
        Arc::new(match hook {
            Some(answer) => session.with_should_trigger_compaction(move |_| answer),
            None => session,
        })
    };
    if outer {
        let compaction = build(backend);
        (encrypted(id, compaction.clone(), TTL), compaction)
    } else {
        let compaction = build(encrypted(id, backend, TTL));
        (compaction.clone(), compaction)
    }
}

pub fn force() -> openai_agents_sdk::memory::OpenAIResponsesCompactionArgs {
    openai_agents_sdk::memory::OpenAIResponsesCompactionArgs {
        force: true,
        ..Default::default()
    }
}
