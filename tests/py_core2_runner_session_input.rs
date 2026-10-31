//! Ports of the session preparation and persistence tests in `tests/test_agent_runner.py`
//! (`prepare_input_with_session*`, `save_result_to_session*`, the guardrail-trip persistence
//! helper and the session retry tests, Python lines ~3353-5320).
//!
//! Python calls the private `prepare_input_with_session` / `save_result_to_session` helpers
//! directly. Rust has no such seam, so each port drives a real run: "prepared input" is what the
//! model receives on its first call, and "session items" are what the run appended to the
//! session for its input, read back from the session afterwards.

#![allow(clippy::field_reassign_with_default)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::error::Result as AgentsResult;
use openai_agents_sdk::items::{OutputItem, RunInput};
use openai_agents_sdk::memory::{InMemorySession, Session, SessionSettings};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, ModelStep, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, InputItem, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

fn tool_returning(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

fn user(content: &str) -> Value {
    json!({"role": "user", "content": content})
}

fn values(items: Vec<InputItem>) -> Vec<Value> {
    items.into_iter().map(|i| i.0).collect()
}

fn contents(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .map(|i| i["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// A session that may report remote history ownership (Python's
/// `OpenAIConversationsSession` subclass in these tests).
#[derive(Debug)]
struct ListSession {
    inner: InMemorySession,
    remote: bool,
    pops: AtomicUsize,
}

impl ListSession {
    async fn new(history: Vec<Value>, remote: bool, limit: Option<usize>) -> Arc<Self> {
        let mut inner = InMemorySession::new("list");
        if let Some(limit) = limit {
            inner = inner.with_settings(SessionSettings::with_limit(limit));
        }
        inner
            .add_items(history.into_iter().map(InputItem).collect())
            .await
            .unwrap();
        Arc::new(ListSession {
            inner,
            remote,
            pops: AtomicUsize::new(0),
        })
    }

    async fn all(&self) -> Vec<Value> {
        values(self.inner.get_items(Some(usize::MAX)).await.unwrap())
    }
}

#[async_trait]
impl Session for ListSession {
    fn session_id(&self) -> &str {
        "list"
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    fn manages_history_remotely(&self) -> bool {
        self.remote
    }
    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> AgentsResult<Option<InputItem>> {
        self.pops.fetch_add(1, Ordering::SeqCst);
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> AgentsResult<()> {
        self.inner.clear_session().await
    }
}

type Callback = openai_agents_sdk::memory::SessionInputCallback;

fn callback<F>(f: F) -> Callback
where
    F: Fn(Vec<Value>, Vec<Value>) -> Vec<Value> + Send + Sync + 'static,
{
    Arc::new(move |history: Vec<InputItem>, new: Vec<InputItem>| {
        let out: Vec<InputItem> = f(values(history), values(new))
            .into_iter()
            .map(InputItem)
            .collect();
        Box::pin(async move { Ok(out) })
    })
}

struct Prepared {
    /// What the model received on its first call.
    model_input: Vec<Value>,
    /// What the run appended to the session for its input (the assistant reply removed).
    session_items: Vec<Value>,
}

/// Python: `prepare_input_with_session(input, session, callback, settings)`.
async fn prepare(
    history: Vec<Value>,
    input: impl Into<RunInput>,
    cb: Option<Callback>,
    limit: Option<usize>,
    remote: bool,
) -> Prepared {
    let model = setup();
    model.add_turn(vec![assistant_message("reply")]);
    let session = ListSession::new(history.clone(), remote, None).await;
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let mut config = RunConfig::default();
    config.session_input_callback = cb;
    config.session_settings = limit.map(SessionSettings::with_limit);
    Runner::run_with_options(
        &agent,
        input,
        RunOptions::new(())
            .session(session.clone())
            .run_config(config),
    )
    .await
    .unwrap();
    let all = session.all().await;
    let mut session_items = all[history.len()..].to_vec();
    let reply = session_items.pop().expect("the reply is saved");
    assert_eq!(reply["role"], "assistant");
    Prepared {
        model_input: values(model.calls()[0].input.clone()),
        session_items,
    }
}

// ------------------------------------------------------------------- orphan pruning

/// Python: `test_prepare_input_with_session_keeps_orphan_output_without_limit`.
/// The durable variant of the retry session: SQLite when the feature is enabled, otherwise the
/// always-available in-memory session, so the test runs in both builds.
fn retry_session() -> Arc<dyn Session> {
    #[cfg(feature = "sqlite")]
    {
        Arc::new(openai_agents_sdk::memory::SqliteSession::in_memory("retry-session").unwrap())
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Arc::new(openai_agents_sdk::memory::InMemorySession::new(
            "retry-session",
        ))
    }
}

#[tokio::test]
async fn an_orphan_history_output_is_kept_without_a_limit() {
    let history =
        json!({"type": "function_call_output", "call_id": "call_prepare", "output": "ok"});
    let p = prepare(vec![history.clone()], "hello", None, None, false).await;
    assert_eq!(p.model_input, vec![history, user("hello")]);
    assert_eq!(p.session_items, vec![user("hello")]);
}

/// Python: `test_prepare_input_with_session_drops_limited_orphan_history_function_call_outputs`.
#[tokio::test]
async fn a_limited_orphan_history_output_is_dropped() {
    let history =
        json!({"type": "function_call_output", "call_id": "call_prepare", "output": "ok"});
    let p = prepare(vec![history], "hello", None, Some(1), false).await;
    assert_eq!(p.model_input, vec![user("hello")]);
    assert_eq!(p.session_items, vec![user("hello")]);
}

/// Python: `test_prepare_input_with_session_preserves_new_function_call_outputs`.
#[tokio::test]
async fn a_new_function_call_output_is_kept() {
    let new_output =
        json!({"type": "function_call_output", "call_id": "call_prepare", "output": "ok"});
    let p = prepare(
        vec![],
        vec![InputItem(new_output.clone())],
        None,
        Some(1),
        false,
    )
    .await;
    assert_eq!(p.model_input, vec![new_output.clone()]);
    assert_eq!(p.session_items, vec![new_output]);
}

/// Python: `test_prepare_input_with_session_leaves_custom_callback_output_unchanged`.
#[tokio::test]
async fn a_custom_callback_output_is_not_pruned() {
    let history =
        json!({"type": "function_call_output", "call_id": "call_callback", "output": "ok"});
    let cb = callback(|mut h, n| {
        h.extend(n);
        h
    });
    let p = prepare(vec![history.clone()], "hello", Some(cb), Some(1), false).await;
    assert_eq!(p.model_input, vec![history, user("hello")]);
    assert_eq!(p.session_items, vec![user("hello")]);
}

/// Python: `test_prepare_input_with_session_drops_output_for_program_owned_call_pruned_with_parent`.
#[tokio::test]
async fn a_program_owned_output_goes_with_its_pruned_program() {
    let program = json!({"type": "program", "call_id": "program_orphan",
        "code": "return await tools.lookup({});", "fingerprint": "fingerprint:orphan"});
    let call = json!({"type": "function_call", "call_id": "call_orphan", "name": "lookup",
        "arguments": "{}", "caller": {"type": "program", "caller_id": "program_orphan"}});
    let output = json!({"type": "function_call_output", "call_id": "call_orphan", "output": "ok"});
    let p = prepare(vec![program, call, output], "hello", None, Some(3), false).await;
    assert_eq!(p.model_input, vec![user("hello")]);
    assert_eq!(p.session_items, vec![user("hello")]);
}

/// Python: `test_prepare_input_with_session_keeps_paired_history_function_call_outputs`.
#[tokio::test]
async fn a_paired_history_call_and_output_are_kept() {
    let call = json!({"type": "function_call", "call_id": "call_prepare", "name": "lookup", "arguments": "{}"});
    let output = json!({"type": "function_call_output", "call_id": "call_prepare", "output": "ok"});
    let p = prepare(
        vec![call.clone(), output.clone()],
        "hello",
        None,
        Some(2),
        false,
    )
    .await;
    assert_eq!(p.model_input, vec![call, output, user("hello")]);
    assert_eq!(p.session_items, vec![user("hello")]);
}

/// Python: `test_prepare_input_with_session_prefers_latest_function_call_output`.
#[tokio::test]
async fn the_latest_function_call_output_wins() {
    let history = json!({"type": "function_call_output", "call_id": "call_latest", "output": "history-output"});
    let latest =
        json!({"type": "function_call_output", "call_id": "call_latest", "output": "new-output"});
    let p = prepare(vec![history], vec![InputItem(latest)], None, None, false).await;
    let outputs: Vec<&Value> = p
        .model_input
        .iter()
        .filter(|i| i["type"] == "function_call_output" && i["call_id"] == "call_latest")
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0]["output"], "new-output");
    assert_eq!(p.session_items.len(), 1);
    assert_eq!(p.session_items[0]["output"], "new-output");
}

/// Python: `test_prepare_input_with_session_drops_orphan_function_calls`.
#[tokio::test]
async fn an_orphan_history_call_is_dropped() {
    let orphan = json!({"type": "function_call", "call_id": "orphan_call", "name": "tool_orphan", "arguments": "{}"});
    let p = prepare(vec![orphan], "hello", None, None, false).await;
    assert_eq!(p.session_items.len(), 1);
    assert!(!p.model_input.iter().any(|i| i["call_id"] == "orphan_call"));
    assert!(p.model_input.contains(&user("hello")));
}

/// Python: `test_prepare_input_with_session_preserves_pending_new_shell_calls`.
#[tokio::test]
async fn a_pending_new_shell_call_is_kept() {
    let orphan = json!({"type": "function_call", "call_id": "orphan_call", "name": "tool_orphan", "arguments": "{}"});
    let shell = json!({"type": "shell_call", "id": "shell_1", "call_id": "manual_shell",
        "status": "completed", "action": {"commands": ["echo hi"], "timeout_ms": null}});
    let p = prepare(
        vec![orphan],
        vec![InputItem(shell.clone())],
        None,
        None,
        false,
    )
    .await;
    assert_eq!(p.session_items, vec![shell]);
    assert!(!p.model_input.iter().any(|i| i["call_id"] == "orphan_call"));
    assert!(
        p.model_input
            .iter()
            .any(|i| i["type"] == "shell_call" && i["call_id"] == "manual_shell")
    );
}

/// Python: `test_ensure_api_input_item_preserves_object_output`.
#[tokio::test]
async fn an_object_tool_output_reaches_the_model_unchanged() {
    let call =
        json!({"type": "function_call", "call_id": "call_object", "name": "t", "arguments": "{}"});
    let output = json!({"type": "function_call_output", "call_id": "call_object", "output": {"complex": "value"}});
    let p = prepare(
        vec![],
        vec![InputItem(call), InputItem(output)],
        None,
        None,
        false,
    )
    .await;
    assert_eq!(p.model_input[1]["output"], json!({"complex": "value"}));
}

// ------------------------------------------------------------------ input callbacks

/// Python: `test_prepare_input_with_session_uses_sync_callback`,
/// `test_prepare_input_with_session_awaits_async_callback`.
#[tokio::test]
async fn a_callback_merges_history_and_new_input() {
    let cb = callback(|h, n| {
        assert_eq!(h[0]["role"], "user");
        let mut out = h;
        out.extend(n);
        out
    });
    let p = prepare(vec![user("hi")], "second", Some(cb), None, false).await;
    assert_eq!(p.model_input.len(), 2);
    assert_eq!(p.model_input[1], user("second"));
    assert_eq!(p.session_items, vec![user("second")]);

    let async_cb: Callback = Arc::new(|history: Vec<InputItem>, new: Vec<InputItem>| {
        Box::pin(async move {
            tokio::task::yield_now().await;
            let mut out = history;
            out.extend(new);
            Ok(out)
        })
    });
    let p = prepare(vec![user("initial")], "later", Some(async_cb), None, false).await;
    assert_eq!(contents(&p.model_input), vec!["initial", "later"]);
    assert_eq!(p.session_items, vec![user("later")]);
}

/// Python: `test_prepare_input_with_session_callback_drops_new_items`.
#[tokio::test]
async fn a_callback_that_drops_new_items_saves_none() {
    let cb = callback(|h, _n| h);
    let p = prepare(vec![user("history")], "new", Some(cb), None, false).await;
    assert_eq!(p.model_input, vec![user("history")]);
    assert!(p.session_items.is_empty(), "{:?}", p.session_items);
}

/// Python: `test_prepare_input_with_session_callback_reorders_new_items`.
#[tokio::test]
async fn a_callback_that_reorders_new_items_saves_its_order() {
    let cb = callback(|h, n| vec![n[1].clone(), h[0].clone(), n[0].clone()]);
    let p = prepare(
        vec![user("history")],
        vec![InputItem::user("first"), InputItem::user("second")],
        Some(cb),
        None,
        false,
    )
    .await;
    assert_eq!(contents(&p.model_input), vec!["second", "history", "first"]);
    assert_eq!(contents(&p.session_items), vec!["second", "first"]);
}

/// Python: `test_prepare_input_with_session_callback_accepts_extra_items`.
#[tokio::test]
async fn a_callback_may_add_extra_items() {
    let extra = json!({"role": "assistant", "content": "extra"});
    let e = extra.clone();
    let cb = callback(move |h, n| vec![e.clone(), h[0].clone(), n[0].clone()]);
    let p = prepare(vec![user("history")], "new", Some(cb), None, false).await;
    assert_eq!(contents(&p.model_input), vec!["extra", "history", "new"]);
    assert_eq!(contents(&p.session_items), vec!["extra", "new"]);
}

/// Python: `test_prepare_input_with_session_matches_copied_items_by_content`.
#[tokio::test]
async fn callback_items_are_matched_by_content() {
    let cb = callback(|h, n| vec![h[0].clone(), n[0].clone()]);
    let p = prepare(vec![user("history")], "new", Some(cb), None, false).await;
    assert_eq!(contents(&p.model_input), vec!["history", "new"]);
    assert_eq!(contents(&p.session_items), vec!["new"]);
}

/// Python: `test_prepare_input_with_session_rejects_non_list_callback_result` (the Rust
/// equivalent of a bad callback is one that fails).
#[tokio::test]
async fn a_failing_callback_fails_the_run() {
    let model = setup();
    model.add_turn(vec![assistant_message("reply")]);
    let session = ListSession::new(vec![], false, None).await;
    let mut config = RunConfig::default();
    config.session_input_callback = Some(Arc::new(|_h, _n| {
        Box::pin(async {
            Err(AgentsError::user(
                "Session input callback must return a list of input items.",
            ))
        })
    }));
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let err = Runner::run_with_options(
        &agent,
        "hello",
        RunOptions::new(()).session(session).run_config(config),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("Session input callback"), "{err}");
    assert!(model.calls().is_empty());
}

// ------------------------------------------------- remotely managed (conversation) history

fn conversation_history() -> Vec<Value> {
    vec![
        json!({"id": "conv_item_user", "type": "message", "role": "user",
               "content": "user history", "provider_data": {"server": "metadata"}}),
        json!({"id": "conv_item_assistant", "type": "message", "role": "assistant",
               "content": "history", "provider_data": {"server": "metadata"}}),
        json!({"id": "conv_item_call", "type": "function_call", "call_id": "call_history",
               "name": "lookup", "arguments": "{}"}),
        json!({"id": "conv_item_output", "type": "function_call_output",
               "call_id": "call_history", "output": "ok"}),
    ]
}

/// Python: `test_prepare_input_with_openai_conversation_strips_assistant_history_ids`.
#[tokio::test]
async fn remote_history_assistant_messages_lose_their_ids() {
    let p = prepare(conversation_history(), "new", None, None, true).await;
    let m = &p.model_input;
    assert_eq!(m[0]["role"], "user");
    assert_eq!(m[0]["id"], "conv_item_user");
    assert!(m[0].get("provider_data").is_some());
    assert_eq!(m[1]["role"], "assistant");
    assert!(m[1].get("id").is_none(), "{}", m[1]);
    assert!(m[1].get("provider_data").is_none());
    assert_eq!(m[2]["id"], "conv_item_call");
    assert_eq!(m[3]["id"], "conv_item_output");
    assert_eq!(m[4], user("new"));
    assert_eq!(contents(&p.session_items), vec!["new"]);
}

/// Python: `test_prepare_input_with_regular_session_preserves_history_ids`.
#[tokio::test]
async fn regular_history_keeps_its_ids() {
    let history =
        json!({"id": "message_id", "type": "message", "role": "assistant", "content": "history"});
    let p = prepare(vec![history], "new", None, None, false).await;
    assert_eq!(p.model_input[0]["id"], "message_id");
}

fn strip_copy() -> Callback {
    callback(|h, n| {
        let mut copy = h[0].clone();
        copy.as_object_mut().unwrap().remove("id");
        copy.as_object_mut().unwrap().remove("provider_data");
        vec![copy, n[0].clone()]
    })
}

/// Python: `test_prepare_input_with_openai_conversation_callback_matches_assistant_no_ids`.
#[tokio::test]
async fn a_remote_assistant_copy_without_ids_is_still_history() {
    let history = json!({"id": "conv_item_assistant", "type": "message", "role": "assistant",
        "content": "history", "provider_data": {"server": "metadata"}});
    let p = prepare(vec![history], "new", Some(strip_copy()), None, true).await;
    assert_eq!(contents(&p.model_input), vec!["history", "new"]);
    assert_eq!(contents(&p.session_items), vec!["new"]);
}

/// Python: `test_prepare_input_with_openai_conversation_callback_keeps_user_ids_distinct`.
#[tokio::test]
async fn a_remote_user_copy_without_ids_is_new() {
    let history = json!({"id": "conv_item_user", "type": "message", "role": "user",
        "content": "history", "provider_data": {"server": "metadata"}});
    let p = prepare(vec![history], "new", Some(strip_copy()), None, true).await;
    assert_eq!(contents(&p.model_input), vec!["history", "new"]);
    assert_eq!(contents(&p.session_items), vec!["history", "new"]);
}

// ------------------------------------------------------------- guardrail-trip persistence

/// Python: `test_input_guardrail_tripwire_does_not_save_assistant_message_to_session`,
/// `test_persist_session_items_for_guardrail_trip_uses_original_input_when_missing`.
#[tokio::test]
async fn an_input_guardrail_trip_saves_only_the_user_input() {
    let model = setup();
    model.add_turn(vec![assistant_message("should_not_be_saved")]);
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_input_guardrail(InputGuardrail::new("slow", |_c, _a, _i| async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(GuardrailFunctionOutput::new(Value::Null, true))
        }));
    let session = ListSession::new(vec![], false, None).await;
    let err = Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        AgentsError::InputGuardrailTripwireTriggered { .. }
    ));
    let items = session.all().await;
    assert_eq!(items, vec![user("user_message")]);
}

// ---------------------------------------------------------------------- model retries

fn retry_once() -> openai_agents_sdk::model_settings::ModelSettings {
    use openai_agents_sdk::model_settings::{
        ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings,
    };
    ModelSettings {
        retry: Some(
            ModelRetrySettings {
                max_retries: 1,
                backoff: ModelRetryBackoffSettings {
                    initial_delay: 0.0,
                    max_delay: 0.0,
                    multiplier: 1.0,
                    jitter: false,
                },
                ..Default::default()
            }
            .with_policy(openai_agents_sdk::retry::retry_policies::http_status(vec![
                500,
            ])),
        ),
        ..Default::default()
    }
}

/// Python: `test_non_streamed_model_retry_does_not_rewind_committed_session_input[memory]`
/// (the sqlite variant runs the same flow against `SqliteSession`).
#[tokio::test]
async fn a_model_retry_does_not_rewind_committed_session_input() {
    for sqlite in [false, cfg!(feature = "sqlite")] {
        let model = setup();
        model.enqueue(ModelStep::ApiError(500));
        model.add_turn(vec![assistant_message("done")]);
        let settings = retry_once();
        let agent = Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_model_settings(settings);
        let session: Arc<dyn Session> = if sqlite {
            let s = retry_session();
            s.add_items(vec![InputItem::user("previous")])
                .await
                .unwrap();
            s
        } else {
            ListSession::new(vec![user("previous")], false, None).await
        };
        let result =
            Runner::run_with_options(&agent, "test", RunOptions::new(()).session(session.clone()))
                .await
                .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        let saved = values(session.get_items(None).await.unwrap());
        let roles: Vec<&str> = saved.iter().filter_map(|i| i["role"].as_str()).collect();
        assert_eq!(roles, vec!["user", "user", "assistant"], "sqlite={sqlite}");
        assert_eq!(
            contents(&saved[..2]),
            vec!["previous", "test"],
            "sqlite={sqlite}"
        );
        assert_eq!(model.calls().len(), 2);
    }
}

/// Python: `test_get_new_response_uses_agent_retry_settings`.
#[tokio::test]
async fn agent_retry_settings_count_every_request() {
    let model = setup();
    model.enqueue(ModelStep::ApiError(500));
    model.add_turn(vec![assistant_message("ok")]);
    model.set_usage(openai_agents_sdk::usage::Usage {
        requests: 1,
        ..Default::default()
    });
    let settings = retry_once();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_model_settings(settings);
    let result = Runner::run(&agent, "hello").await.unwrap();
    assert_eq!(result.raw_responses.len(), 1);
    assert_eq!(result.raw_responses[0].usage.requests, 2);
}

// --------------------------------------------------------------- saved run results

async fn run_with_session(
    input: Vec<Value>,
    turns: Vec<Vec<OutputItem>>,
    tool_value: &str,
) -> Vec<Value> {
    let model = setup();
    for t in turns {
        model.add_turn(t);
    }
    let session = ListSession::new(vec![], false, None).await;
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool_returning("tool_ordered", tool_value));
    Runner::run_with_options(
        &agent,
        input.into_iter().map(InputItem).collect::<Vec<_>>(),
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    session.all().await
}

/// Python: `test_save_result_to_session_preserves_function_outputs`.
#[tokio::test]
async fn input_and_generated_function_outputs_are_both_saved() {
    let saved = run_with_session(
        vec![json!({"type": "function_call_output", "call_id": "call_original", "output": "1"})],
        vec![
            vec![function_call_with_id("tool_ordered", "{}", "call_result")],
            vec![assistant_message("done")],
        ],
        "2",
    )
    .await;
    let outputs: Vec<&Value> = saved
        .iter()
        .filter(|i| i["type"] == "function_call_output")
        .collect();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|o| o.get("output").is_some()));
}

/// Python: `test_save_result_to_session_prefers_latest_duplicate_function_outputs`,
/// `test_save_result_to_session_keeps_latest_output_after_its_call`.
#[tokio::test]
async fn the_latest_duplicate_output_is_saved_after_its_call() {
    let saved = run_with_session(
        vec![json!({"type": "function_call_output", "call_id": "call_ordered", "output": "old"})],
        vec![
            vec![function_call_with_id("tool_ordered", "{}", "call_ordered")],
            vec![assistant_message("done")],
        ],
        "new",
    )
    .await;
    // The caller's input is stored first, as Python's runner does, so its stale output stays
    // where it was written; the call and the output the tool produced follow, in that order.
    let relevant: Vec<&Value> = saved
        .iter()
        .filter(|i| i["call_id"] == "call_ordered")
        .collect();
    let types: Vec<&str> = relevant.iter().filter_map(|i| i["type"].as_str()).collect();
    assert_eq!(
        types,
        vec![
            "function_call_output",
            "function_call",
            "function_call_output"
        ]
    );
    assert_eq!(relevant[0]["output"], "old");
    assert_eq!(relevant[2]["output"], "new");
}

/// Python: `test_save_result_to_session_keeps_tool_call_before_its_output`.
#[tokio::test]
async fn a_replayed_call_stays_before_its_output() {
    let saved = run_with_session(
        vec![
            json!({"type": "function_call", "call_id": "call_ordered", "name": "tool_ordered", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_ordered", "output": "result"}),
        ],
        vec![
            vec![function_call_with_id("tool_ordered", "{}", "call_ordered")],
            vec![assistant_message("done")],
        ],
        "result",
    )
    .await;
    // The caller's input is stored first, so the pair it supplied stays as written and the
    // model's replayed call and its output follow as their own batch, call first.
    let types: Vec<&str> = saved
        .iter()
        .filter(|i| i["call_id"] == "call_ordered")
        .filter_map(|i| i["type"].as_str())
        .collect();
    assert_eq!(
        types,
        vec![
            "function_call",
            "function_call_output",
            "function_call",
            "function_call_output"
        ]
    );
}

/// Python: `test_save_result_to_session_does_not_increment_counter_when_nothing_saved`,
/// `test_save_result_to_session_returns_count_and_updates_state` (the persisted part: an
/// approval item is never written to the session).
#[tokio::test]
async fn approval_items_are_never_saved() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("ok"),
        function_call_with_id("tool_ordered", "{}", "call-2"),
    ]);
    let session = ListSession::new(vec![], false, None).await;
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool_returning("tool_ordered", "r").with_needs_approval(true));
    let result = Runner::run_with_options(
        &agent,
        "input",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    let saved = session.all().await;
    assert!(
        !saved
            .iter()
            .any(|i| i["type"] == "tool_approval_item" || i.get("raw_item").is_some())
    );
    assert!(saved.iter().any(|i| i["content"] == "input"));
}

/// Python: `test_save_result_to_session_omits_reasoning_ids_when_policy_is_omit`.
#[tokio::test]
async fn the_omit_policy_strips_reasoning_ids_on_save() {
    let model = setup();
    model.add_turn(vec![
        OutputItem::from_value(json!({"type": "reasoning", "id": "rs_stream", "summary": []})),
        assistant_message("done"),
    ]);
    let session = ListSession::new(vec![], false, None).await;
    let mut config = RunConfig::default();
    config.reasoning_item_id_policy = openai_agents_sdk::run_config::ReasoningItemIdPolicy::Omit;
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    Runner::run_with_options(
        &agent,
        "input",
        RunOptions::new(())
            .session(session.clone())
            .run_config(config),
    )
    .await
    .unwrap();
    let saved = session.all().await;
    let reasoning = saved.iter().find(|i| i["type"] == "reasoning").unwrap();
    assert!(reasoning.get("id").is_none(), "{reasoning}");
}

/// Python: `test_save_result_to_session_keeps_tool_call_payload_api_safe`,
/// `test_save_result_to_session_sanitizes_original_input_items`,
/// `test_prepare_input_with_session_strips_internal_tool_call_metadata`,
/// `test_prepare_input_with_session_sanitizes_new_tool_call_session_items`: SDK-internal
/// tool-call metadata never reaches the session or the model.
#[tokio::test]
async fn internal_tool_call_metadata_never_leaves_the_sdk() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "tool_ordered",
        "{}",
        "call_session",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let session = ListSession::new(vec![], false, None).await;
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool_returning("tool_ordered", "r"));
    Runner::run_with_options(
        &agent,
        "input",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    let mut seen = session.all().await;
    for call in model.calls() {
        seen.extend(values(call.input));
    }
    for item in &seen {
        for key in ["description", "title"] {
            assert!(item.get(key).is_none(), "{key} leaked: {item}");
        }
        assert!(
            item.as_object()
                .is_none_or(|m| !m.keys().any(|k| k.starts_with("_agents"))),
            "{item}"
        );
    }
}

// ----------------------------------------------------------------- only new step items

/// Python: `test_session_persists_only_new_step_items`: each turn appends only the items it
/// generated, never the items earlier turns already persisted.
#[tokio::test]
async fn each_turn_persists_only_its_own_items() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("old"),
        function_call_with_id("tool_ordered", "{}", "c1"),
    ]);
    model.add_turn(vec![assistant_message("new")]);
    let session = ListSession::new(vec![], false, None).await;
    let writes = Arc::new(Mutex::new(0usize));
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(tool_returning("tool_ordered", "r"));
    let result = Runner::run_with_options(
        &agent,
        "hello",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    let _ = writes;
    assert_eq!(result.final_output_text().as_deref(), Some("new"));
    let saved = session.all().await;
    let olds = saved
        .iter()
        .filter(|i| i.to_string().contains("\"old\""))
        .count();
    let news = saved
        .iter()
        .filter(|i| i.to_string().contains("\"new\""))
        .count();
    assert_eq!((olds, news), (1, 1), "{saved:?}");
}
