//! Ports of the context-aware session cases of `tests/memory/test_session_context_wrapper.py`
//! and `tests/deferred_session_writes/test_resume.py`.
//!
//! Python opts a session in by giving every history method a `wrapper` argument. Rust sessions
//! opt in by overriding the defaulted `*_with_context` methods of `Session`; a session that does
//! not override them keeps receiving the plain calls.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::memory::{Session, SessionRunContext};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, InputItem, OutputGuardrail,
    RunOptions, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug)]
struct TenantContext {
    tenant_id: &'static str,
}

const DEFAULT_SCOPE: &str = "default";

/// Python's `ContextAwareSession`: history is scoped by the tenant of the run context.
#[derive(Debug, Default)]
struct ContextAwareSession {
    items_by_scope: Mutex<HashMap<String, Vec<InputItem>>>,
    /// `(method, carried a run context)`.
    calls: Mutex<Vec<(&'static str, bool)>>,
    /// Calls that arrived without a run context, by method name.
    wrapperless: Mutex<Vec<&'static str>>,
    wrapper_contexts: Mutex<Vec<Arc<TenantContext>>>,
}

impl ContextAwareSession {
    fn scope(ctx: Option<SessionRunContext<'_>>) -> String {
        ctx.and_then(|c| c.context::<TenantContext>())
            .map(|c| c.tenant_id.to_string())
            .unwrap_or_else(|| DEFAULT_SCOPE.to_string())
    }

    fn record(&self, method: &'static str, ctx: Option<SessionRunContext<'_>>) {
        match ctx.and_then(|c| c.run_context::<TenantContext>()) {
            Some(run_context) => {
                let arc = run_context.context_arc();
                self.calls.lock().unwrap().push((method, true));
                self.wrapper_contexts.lock().unwrap().push(arc);
            }
            None => {
                self.calls.lock().unwrap().push((method, false));
                self.wrapperless.lock().unwrap().push(method);
            }
        }
    }

    fn get(&self, ctx: Option<SessionRunContext<'_>>, limit: Option<usize>) -> Vec<InputItem> {
        let mut map = self.items_by_scope.lock().unwrap();
        let items = map.entry(Self::scope(ctx)).or_default();
        match limit {
            None => items.clone(),
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        }
    }

    fn add(&self, ctx: Option<SessionRunContext<'_>>, items: Vec<InputItem>) {
        self.items_by_scope
            .lock()
            .unwrap()
            .entry(Self::scope(ctx))
            .or_default()
            .extend(items);
    }

    fn scoped(&self, scope: &str) -> Vec<InputItem> {
        self.items_by_scope
            .lock()
            .unwrap()
            .get(scope)
            .cloned()
            .unwrap_or_default()
    }

    fn seed(&self, scope: &str, items: Vec<InputItem>) {
        self.items_by_scope
            .lock()
            .unwrap()
            .insert(scope.to_string(), items);
    }

    fn call_names(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().iter().map(|(n, _)| *n).collect()
    }

    fn clear_calls(&self) {
        self.calls.lock().unwrap().clear();
        self.wrapperless.lock().unwrap().clear();
        self.wrapper_contexts.lock().unwrap().clear();
    }
}

#[async_trait]
impl Session for ContextAwareSession {
    fn session_id(&self) -> &str {
        "context-aware"
    }

    async fn get_items(&self, limit: Option<usize>) -> openai_agents_sdk::Result<Vec<InputItem>> {
        self.record("get_items", None);
        Ok(self.get(None, limit))
    }

    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::Result<()> {
        self.record("add_items", None);
        self.add(None, items);
        Ok(())
    }

    async fn pop_item(&self) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.record("pop_item", None);
        Ok(self
            .items_by_scope
            .lock()
            .unwrap()
            .entry(DEFAULT_SCOPE.into())
            .or_default()
            .pop())
    }

    async fn clear_session(&self) -> openai_agents_sdk::Result<()> {
        self.record("clear_session", None);
        self.seed(DEFAULT_SCOPE, Vec::new());
        Ok(())
    }

    async fn get_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        limit: Option<usize>,
    ) -> openai_agents_sdk::Result<Vec<InputItem>> {
        self.record("get_items", Some(ctx));
        Ok(self.get(Some(ctx), limit))
    }

    async fn add_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        items: Vec<InputItem>,
    ) -> openai_agents_sdk::Result<()> {
        self.record("add_items", Some(ctx));
        self.add(Some(ctx), items);
        Ok(())
    }

    async fn pop_item_with_context(
        &self,
        ctx: SessionRunContext<'_>,
    ) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.record("pop_item", Some(ctx));
        Ok(self
            .items_by_scope
            .lock()
            .unwrap()
            .entry(Self::scope(Some(ctx)))
            .or_default()
            .pop())
    }

    async fn clear_session_with_context(
        &self,
        ctx: SessionRunContext<'_>,
    ) -> openai_agents_sdk::Result<()> {
        self.record("clear_session", Some(ctx));
        self.seed(&Self::scope(Some(ctx)), Vec::new());
        Ok(())
    }
}

/// Python's `LegacySession`: only the plain methods exist.
#[derive(Debug, Default)]
struct LegacySession {
    items: Mutex<Vec<InputItem>>,
    get_calls: Mutex<usize>,
}

#[async_trait]
impl Session for LegacySession {
    fn session_id(&self) -> &str {
        "legacy"
    }

    async fn get_items(&self, limit: Option<usize>) -> openai_agents_sdk::Result<Vec<InputItem>> {
        *self.get_calls.lock().unwrap() += 1;
        let items = self.items.lock().unwrap();
        Ok(match limit {
            None => items.clone(),
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::Result<()> {
        self.items.lock().unwrap().extend(items);
        Ok(())
    }

    async fn pop_item(&self) -> openai_agents_sdk::Result<Option<InputItem>> {
        Ok(self.items.lock().unwrap().pop())
    }

    async fn clear_session(&self) -> openai_agents_sdk::Result<()> {
        self.items.lock().unwrap().clear();
        Ok(())
    }
}

fn user_item(text: &str) -> InputItem {
    InputItem(json!({"role": "user", "content": text}))
}

fn tenant_a() -> TenantContext {
    TenantContext {
        tenant_id: "tenant-a",
    }
}

async fn drive<C: Send + Sync + 'static>(
    agent: &Agent<C>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    options: RunOptions<C>,
    streamed: bool,
) -> Result<
    (
        Option<String>,
        Vec<openai_agents_sdk::items::ToolApprovalItem>,
        Option<RunState>,
        Arc<C>,
    ),
    AgentsError,
> {
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        let ctx = result
            .result()
            .expect("a finished streamed run has a result")
            .context_wrapper
            .context_arc();
        Ok((
            result.final_output_text(),
            result.interruptions(),
            result.to_state(),
            ctx,
        ))
    } else {
        let result = Runner::run_with_options(agent, input, options).await?;
        Ok((
            result.final_output_text(),
            result.interruptions.clone(),
            Some(result.to_state()),
            result.context_wrapper.context_arc(),
        ))
    }
}

fn agent_with_reply(reply: &str) -> Agent<TenantContext> {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message(reply)]);
    Agent::<TenantContext>::new("test").with_model_instance(model)
}

/// Python: `test_runner_passes_same_wrapper_to_context_aware_session`.
#[tokio::test]
async fn the_runner_passes_the_run_context_to_a_context_aware_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(ContextAwareSession::default());
        let agent = agent_with_reply("ok");
        let options = RunOptions::new(tenant_a()).session(session.clone());

        let (text, _, _, run_context) = drive(&agent, "hello", options, streamed).await.unwrap();

        assert_eq!(text.as_deref(), Some("ok"), "streamed={streamed}");
        assert_eq!(
            session.call_names(),
            ["get_items", "add_items", "add_items"],
            "streamed={streamed}"
        );
        assert!(session.wrapperless.lock().unwrap().is_empty());
        // Every call saw the run's own context object.
        for arc in session.wrapper_contexts.lock().unwrap().iter() {
            assert!(Arc::ptr_eq(arc, &run_context), "streamed={streamed}");
        }
        assert!(session.scoped(DEFAULT_SCOPE).is_empty());
        assert_eq!(session.scoped("tenant-a").len(), 2);
    }
}

/// Python: `test_runner_preserves_legacy_session_call_shapes`. A session that overrides only
/// the plain methods keeps working: the defaulted context methods forward to them.
#[tokio::test]
async fn a_session_without_context_methods_gets_the_plain_calls() {
    openai_agents_sdk::set_tracing_disabled(true);
    let session = Arc::new(LegacySession::default());
    let agent = agent_with_reply("ok");
    let options = RunOptions::new(tenant_a()).session(session.clone());

    let (text, _, _, _) = drive(&agent, "hello", options, false).await.unwrap();

    assert_eq!(text.as_deref(), Some("ok"));
    assert_eq!(*session.get_calls.lock().unwrap(), 1);
    assert_eq!(session.items.lock().unwrap().len(), 2);
}

/// Python: `test_input_guardrail_persists_in_the_context_scope`.
#[tokio::test]
async fn an_input_guardrail_trip_persists_the_input_in_the_context_scope() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let session = Arc::new(ContextAwareSession::default());
        session.seed(DEFAULT_SCOPE, vec![user_item("keep")]);
        let agent = agent_with_reply("not persisted")
            .with_input_guardrail(InputGuardrail::new("trips", |_c, _a, _i| async {
                Ok(GuardrailFunctionOutput::new(Value::Null, true))
            }));
        let options = RunOptions::new(tenant_a()).session(session.clone());

        let err = drive(&agent, "hello", options, streamed)
            .await
            .expect_err("the guardrail trips");

        assert!(
            matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
            "streamed={streamed}: {err:?}"
        );
        assert_eq!(session.scoped(DEFAULT_SCOPE), vec![user_item("keep")]);
        let stored = session.scoped("tenant-a");
        assert_eq!(stored.len(), 1, "streamed={streamed}");
        assert_eq!(stored[0].as_value()["role"], "user");
        assert_eq!(stored[0].as_value()["content"], "hello");
        assert!(
            session.wrapperless.lock().unwrap().is_empty(),
            "streamed={streamed}: {:?}",
            session.wrapperless.lock().unwrap()
        );
    }
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn gated_tool<C: Send + Sync + 'static>() -> FunctionTool<C> {
    FunctionTool::new("test_tool", "gated", |_ctx, _args: Empty| async move {
        Ok("tool result".to_string())
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

/// Python: `test_resumed_run_persists_in_the_context_scope`.
#[tokio::test]
async fn a_resumed_run_persists_in_the_context_scope() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call_with_id(
            "test_tool",
            "{}",
            "call-resume",
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<TenantContext>::new("test")
            .with_tool(gated_tool::<TenantContext>())
            .with_model_instance(model);
        let session = Arc::new(ContextAwareSession::default());
        session.seed(DEFAULT_SCOPE, vec![user_item("keep")]);

        let (_, interruptions, state, _) = drive(
            &agent,
            "hello",
            RunOptions::new(tenant_a()).session(session.clone()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(interruptions.len(), 1, "streamed={streamed}");
        let mut state = state.unwrap();
        state.approve(&interruptions[0], false);
        session.clear_calls();

        let (text, _, _, run_context) = drive(
            &agent,
            state,
            RunOptions::new(tenant_a()).session(session.clone()),
            streamed,
        )
        .await
        .unwrap();

        assert_eq!(text.as_deref(), Some("done"), "streamed={streamed}");
        assert_eq!(session.scoped(DEFAULT_SCOPE), vec![user_item("keep")]);
        assert!(session.wrapperless.lock().unwrap().is_empty());
        for arc in session.wrapper_contexts.lock().unwrap().iter() {
            assert!(Arc::ptr_eq(arc, &run_context), "streamed={streamed}");
        }
        assert!(
            session.scoped("tenant-a").iter().any(|item| {
                item.item_type() == Some("function_call_output")
                    && item.as_value()["call_id"] == "call-resume"
            }),
            "streamed={streamed}"
        );
    }
}

/// Python: `test_compaction_session_keeps_context_aware_underlying_on_legacy_scope`. A wrapper
/// session does not forward the run context, so its underlying store stays on the plain scope.
#[tokio::test]
async fn a_compaction_session_keeps_its_underlying_store_on_the_plain_scope() {
    use openai_agents_sdk::memory::OpenAIResponsesCompactionSession;
    openai_agents_sdk::set_tracing_disabled(true);
    let underlying = Arc::new(ContextAwareSession::default());
    underlying.seed(DEFAULT_SCOPE, vec![user_item("existing")]);
    let session = Arc::new(
        OpenAIResponsesCompactionSession::new("compaction", underlying.clone())
            .unwrap()
            .with_should_trigger_compaction(|_| false),
    );
    let agent = agent_with_reply("done");
    let options = RunOptions::new(tenant_a()).session(session);

    let (text, _, _, _) = drive(&agent, "hello", options, false).await.unwrap();

    assert_eq!(text.as_deref(), Some("done"));
    assert!(underlying.scoped("tenant-a").is_empty());
    assert_eq!(underlying.scoped(DEFAULT_SCOPE).len(), 3);
    assert!(!underlying.call_names().is_empty());
    assert!(
        underlying
            .calls
            .lock()
            .unwrap()
            .iter()
            .all(|(_, has_ctx)| !has_ctx)
    );
}

// --- deferred session writes: the settle path reaches a context-aware session -----------------

#[derive(Deserialize, JsonSchema)]
struct Query {
    query: String,
}

fn typed_write_thing() -> FunctionTool<TenantContext> {
    FunctionTool::new("write_thing", "writes", |_ctx, q: Query| async move {
        Ok(format!("wrote:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn typed_look_up() -> FunctionTool<TenantContext> {
    FunctionTool::new("look_up", "looks up", |_ctx, q: Query| async move {
        Ok(format!("schema for {}", q.query))
    })
}

fn typed_ok_guardrail() -> OutputGuardrail<TenantContext> {
    OutputGuardrail::new("always_fine", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, false))
    })
}

/// Python: `test_settle_reaches_a_context_aware_session_through_the_wrapper`.
#[tokio::test]
async fn settling_a_held_write_reaches_a_context_aware_session_with_the_context() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call_with_id(
            "look_up",
            r#"{"query":"x"}"#,
            "call_LOOKUP",
        )]);
        model.add_turn(vec![function_call_with_id(
            "write_thing",
            r#"{"query":"x"}"#,
            "call_PARKED",
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<TenantContext>::new("deferred repro")
            .with_tool(typed_look_up())
            .with_tool(typed_write_thing())
            .with_output_guardrail(typed_ok_guardrail())
            .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
                stop_at_tool_names: vec!["finish".into()],
            }))
            .with_model_instance(model.clone());
        let session = Arc::new(ContextAwareSession::default());

        let (_, interruptions, state, _) = drive(
            &agent,
            "do the thing",
            RunOptions::new(tenant_a()).session(session.clone()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(interruptions.len(), 1);
        let mut state = RunState::from_json(state.unwrap().to_json().unwrap()).unwrap();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        session.clear_calls();
        model.add_turn(vec![assistant_message("done")]);

        drive(
            &agent,
            state,
            RunOptions::new(tenant_a()).session(session.clone()),
            streamed,
        )
        .await
        .unwrap();

        assert!(
            session.wrapperless.lock().unwrap().is_empty(),
            "streamed={streamed}: {:?}",
            session.wrapperless.lock().unwrap()
        );
        let stored = session.scoped("tenant-a");
        let parked: Vec<&str> = stored
            .iter()
            .filter(|i| i.as_value().get("call_id").and_then(Value::as_str) == Some("call_PARKED"))
            .filter_map(|i| i.item_type())
            .collect();
        assert_eq!(
            parked,
            ["function_call", "function_call_output"],
            "streamed={streamed}"
        );
    }
}
