//! Ports of the registration-policy cases of `tests/deferred_session_writes/test_pending_write.py`:
//! a held batch keeps the server-assigned reasoning ids that a remote-history session needs.
//!
//! Python reads the record through `_pending_session_write`; here the record is the
//! `pending_session_write` object of the serialized checkpoint.

#![allow(clippy::field_reassign_with_default)]

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::error::Result as SdkResult;
use openai_agents_sdk::items::ReasoningOutput;
use openai_agents_sdk::memory::Session;
use openai_agents_sdk::run_config::{ReasoningItemIdPolicy, RunConfig};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, GuardrailFunctionOutput, InputItem, OutputGuardrail, OutputItem, RunOptions, RunState,
    Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::ListSession;

/// A store that owns its history, like the Conversations session.
#[derive(Debug)]
struct RemoteHistorySession {
    inner: ListSession,
}

#[async_trait]
impl Session for RemoteHistorySession {
    fn session_id(&self) -> &str {
        "conv_abc"
    }
    fn manages_history_remotely(&self) -> bool {
        true
    }
    async fn get_items(&self, limit: Option<usize>) -> SdkResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> SdkResult<()> {
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> SdkResult<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> SdkResult<()> {
        self.inner.clear_session().await
    }
}

#[derive(Deserialize, JsonSchema)]
struct Query {
    #[serde(default)]
    query: String,
}

fn gated(name: &'static str) -> FunctionTool<()> {
    FunctionTool::new(name, "writes", |_ctx, q: Query| async move {
        Ok(format!("wrote:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn reasoning(id: &str) -> OutputItem {
    OutputItem::Reasoning(ReasoningOutput {
        id: id.into(),
        summary: vec![json!({"type": "summary_text", "text": "because"})],
        content: None,
        encrypted_content: None,
        status: None,
        extra: Default::default(),
    })
}

fn deferring_agent(steps: Vec<ScriptedStep>) -> Agent<()> {
    let model = Arc::new(ScriptedModel::new());
    model.extend(steps).unwrap();
    Agent::<()>::new("deferred")
        .with_tool(gated("write_thing"))
        .with_tool(gated("write_other"))
        .with_output_guardrail(OutputGuardrail::new(
            "always_fine",
            |_c, _a, _o: Value| async { Ok(GuardrailFunctionOutput::new(Value::Null, false)) },
        ))
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["finish".into()],
        }))
        .with_model_instance(model)
}

fn omit_ids() -> RunConfig<()> {
    RunConfig::<()>::default().with_reasoning_item_id_policy(ReasoningItemIdPolicy::Omit)
}

fn held_reasoning_ids(state: &RunState) -> Vec<Value> {
    let payload = state.to_json().unwrap();
    assert_eq!(payload["pending_session_write"]["held"], json!(true));
    payload["pending_session_write"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .map(|item| item["id"].clone())
        .collect()
}

/// Python: `test_registration_forces_the_conversations_reasoning_policy` and
/// `test_the_park_records_the_conversion_policy_it_used`. A session that owns its history keeps
/// a server-identified reasoning item persistable even when the run omits reasoning ids, and
/// the record states that no conversion policy rewrote its items.
#[tokio::test]
async fn a_remote_history_park_keeps_server_reasoning_ids_under_an_omit_policy() {
    openai_agents_sdk::set_tracing_disabled(true);
    let agent = deferring_agent(vec![
        ScriptedStep::output(vec![
            reasoning("rs_server_1"),
            function_call_with_id("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
        ])
        .with_response_id(Some("resp_parked".into())),
    ]);
    let session = Arc::new(RemoteHistorySession {
        inner: ListSession::new("conv_abc"),
    });
    let options = RunOptions::new(()).session(session).run_config(omit_ids());

    let result = Runner::run_with_options(&agent, "go", options)
        .await
        .unwrap();

    assert_eq!(result.interruptions.len(), 1);
    let state = result.to_state();
    assert_eq!(held_reasoning_ids(&state), vec![json!("rs_server_1")]);
    assert_eq!(
        state.to_json().unwrap()["pending_session_write"]["reasoning_item_id_policy"],
        Value::Null
    );
}

/// Python: `test_a_detached_re_park_folds_under_the_batch_registration_policy`. A detached
/// resume (no session) that parks again folds its new items into the standing record without
/// stripping the server reasoning id the run's own policy would omit.
#[tokio::test]
async fn a_detached_re_park_keeps_the_server_reasoning_id_of_the_batch() {
    openai_agents_sdk::set_tracing_disabled(true);
    let agent = deferring_agent(vec![
        ScriptedStep::output(vec![function_call_with_id(
            "write_thing",
            r#"{"query":"x"}"#,
            "call_PARKED",
        )])
        .with_response_id(Some("resp_parked".into())),
        ScriptedStep::output(vec![
            reasoning("rs_SERVER_ID"),
            function_call_with_id("write_other", r#"{"query":"y"}"#, "call_SECOND"),
        ])
        .with_response_id(Some("resp_second".into())),
    ]);
    let session = Arc::new(RemoteHistorySession {
        inner: ListSession::new("conv_abc"),
    });
    let first = Runner::run_with_options(
        &agent,
        "go",
        RunOptions::new(()).session(session).run_config(omit_ids()),
    )
    .await
    .unwrap();
    let mut state = first.to_state();
    let parked = state.get_interruptions()[0].clone();
    state.approve(&parked, false);
    // Round-trip through the checkpoint, then resume with no session at all.
    let mut restored = RunState::from_json(state.to_json().unwrap()).unwrap();
    restored.approve(&restored.get_interruptions()[0].clone(), false);

    let second =
        Runner::run_with_options(&agent, restored, RunOptions::new(()).run_config(omit_ids()))
            .await
            .unwrap();

    assert_eq!(second.interruptions.len(), 1);
    assert_eq!(
        held_reasoning_ids(&second.to_state()),
        vec![json!("rs_SERVER_ID")]
    );
}
