//! Parity port of `tests/memory/test_compaction_model_visibility.py`: automatic compaction only
//! ever summarizes history the model was actually shown. A filter, a session input callback or a
//! handoff input filter that hides stored items keeps those items out of the compaction request
//! until a run replays them in full.
//!
//! The compaction endpoint is a local mock server; the model is `FakeModel`. Python's
//! `MagicMock` client assertions become assertions on the captured `/responses/compact`
//! requests.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "sqlite")]

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::items::RunInput;
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionSession, Session, SessionSettings, SqliteSession,
};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::run_config::{ModelInputData, ReasoningItemIdPolicy};
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool};
use openai_agents_sdk::{
    Agent, InputItem, OpenAIClient, RunConfig, RunOptions, RunResult, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{ListSession, Request, mock_openai_with};

const LOCAL_OUTPUT: &str = "synthetic-local-only-result";

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn lookup_tool(needs_approval: bool) -> FunctionTool<()> {
    lookup_returning(LOCAL_OUTPUT, needs_approval)
}

fn lookup_returning(result: &'static str, needs_approval: bool) -> FunctionTool<()> {
    FunctionTool::new("lookup", "lookup", move |_ctx, _a: NoArgs| async move {
        Ok(result.to_string())
    })
    .with_needs_approval(needs_approval)
}

/// Python's `IdStrippingSession`: a store that reassigns item ids, so replayed items must be
/// matched without them.
#[derive(Debug)]
struct IdStrippingSession(ListSession);

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

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

fn without_tools(items: Vec<InputItem>) -> Vec<InputItem> {
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

fn text(items: &[InputItem]) -> String {
    items
        .iter()
        .map(|i| i.as_value().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn values(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|i| i.as_value().clone()).collect()
}

fn of_type<'a>(items: &'a [Value], kind: &str) -> Vec<&'a Value> {
    items.iter().filter(|i| i["type"] == kind).collect()
}

/// A compaction endpoint whose replies come from `reply(request body)`.
struct Compaction {
    base: String,
    captured: Arc<Mutex<Vec<Request>>>,
}

impl Compaction {
    async fn replying(output: Vec<Value>) -> Compaction {
        Self::with(move |_| json!({"id": "cmp", "output": output.clone(), "usage": usage()})).await
    }
    async fn with(f: impl Fn(&Value) -> Value + Send + Sync + 'static) -> Compaction {
        let (base, captured) = mock_openai_with(move |req| f(&req.body)).await;
        Compaction { base, captured }
    }
    fn client(&self) -> OpenAIClient {
        OpenAIClient::new("sk-test").with_base_url(self.base.clone())
    }
    fn requests(&self) -> Vec<Value> {
        self.captured
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.path.ends_with("/responses/compact"))
            .map(|r| r.body.clone())
            .collect()
    }
    fn inputs(&self) -> Vec<Vec<Value>> {
        self.requests()
            .iter()
            .map(|b| b["input"].as_array().cloned().unwrap_or_default())
            .collect()
    }
}

fn usage() -> Value {
    json!({"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})
}

fn summary() -> Value {
    json!({"type": "compaction", "id": "cmp-test", "encrypted_content": "synthetic-summary"})
}

fn compaction_session(
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

fn always(session: OpenAIResponsesCompactionSession) -> OpenAIResponsesCompactionSession {
    session.with_should_trigger_compaction(|_| true)
}

async fn run(
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

fn as_session(session: OpenAIResponsesCompactionSession) -> Arc<dyn Session> {
    Arc::new(session)
}

fn model_with(turns: Vec<Vec<openai_agents_sdk::items::OutputItem>>) -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    for turn in turns {
        model.add_turn(turn);
    }
    model
}

fn texts(turns: &[&str]) -> Vec<Vec<openai_agents_sdk::items::OutputItem>> {
    turns.iter().map(|t| vec![assistant_message(*t)]).collect()
}

fn model_input(model: &FakeModel, call: usize) -> Vec<InputItem> {
    model.calls()[call].input.clone()
}

fn model_input_values(model: &FakeModel, call: usize) -> Vec<Value> {
    values(&model_input(model, call))
}

// ---------------------------------------------------------------------------------------------
// test_automatic_compaction_retains_hidden_tool_output_until_full_replay (2 x 3)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn automatic_compaction_retains_hidden_tool_output_until_full_replay() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        for filter_kind in ["model", "session", "redact"] {
            let endpoint = Compaction::replying(vec![summary()]).await;
            let session = as_session(always(compaction_session(
                "visibility",
                Arc::new(ListSession::new("visibility")),
                &endpoint,
                CompactionMode::Auto,
            )));
            let model = model_with(vec![
                vec![function_call_with_id("lookup", "{}", "call_1")],
                vec![assistant_message("ok")],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("worker")
                .with_model_instance(model.clone())
                .with_model_settings(ModelSettings::default().with_store(false))
                .with_tool(lookup_tool(false))
                .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
            run(&agent, "first", &session, streamed, None).await;
            assert!(endpoint.requests().is_empty(), "{filter_kind}");

            let config = match filter_kind {
                "session" => RunConfig::<()>::default().with_session_input_callback(
                    |history: Vec<InputItem>, new: Vec<InputItem>| async move {
                        let mut out = without_tools(history);
                        out.extend(new);
                        Ok(out)
                    },
                ),
                "redact" => {
                    RunConfig::<()>::default().with_call_model_input_filter(|data| async move {
                        let input = data
                            .model_data
                            .input
                            .into_iter()
                            .map(|item| {
                                if item.item_type() == Some("function_call_output") {
                                    let mut v = item.as_value().clone();
                                    v["output"] = json!("redacted");
                                    InputItem::from_value(v)
                                } else {
                                    item
                                }
                            })
                            .collect();
                        Ok(ModelInputData {
                            input,
                            instructions: data.model_data.instructions,
                        })
                    })
                }
                _ => RunConfig::<()>::default().with_call_model_input_filter(|data| async move {
                    Ok(ModelInputData {
                        input: without_tools(data.model_data.input),
                        instructions: data.model_data.instructions,
                    })
                }),
            };
            run(&agent, "second", &session, streamed, Some(config)).await;
            assert!(!text(&model_input(&model, 1)).contains(LOCAL_OUTPUT));
            assert!(endpoint.requests().is_empty(), "{filter_kind}");
            assert!(text(&session.get_items(None).await.unwrap()).contains(LOCAL_OUTPUT));

            run(&agent, "third", &session, streamed, None).await;
            assert!(text(&model_input(&model, 2)).contains(LOCAL_OUTPUT));
            assert_eq!(endpoint.requests().len(), 1, "{filter_kind}");
            assert!(
                endpoint.inputs()[0]
                    .iter()
                    .any(|i| i.to_string().contains(LOCAL_OUTPUT))
            );
            assert_eq!(
                values(&session.get_items(None).await.unwrap()),
                vec![summary()]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_manual_compaction_can_explicitly_replace_filtered_history (2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn manual_compaction_can_explicitly_replace_filtered_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    for mode in [CompactionMode::Input, CompactionMode::PreviousResponseId] {
        let endpoint = Compaction::replying(vec![]).await;
        let session = Arc::new(always(compaction_session(
            "manual",
            Arc::new(ListSession::with_history(
                "manual",
                vec![user(LOCAL_OUTPUT)],
            )),
            &endpoint,
            mode,
        )));
        let dyn_session: Arc<dyn Session> = session.clone();
        let model = model_with(texts(&["ok"]));
        let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
        run(
            &agent,
            "visible",
            &dyn_session,
            false,
            Some(RunConfig::<()>::default().with_session_input_callback(
                |_history: Vec<InputItem>, new: Vec<InputItem>| async move { Ok(new) },
            )),
        )
        .await;
        assert!(endpoint.requests().is_empty());
        assert!(text(&session.get_items(None).await.unwrap()).contains(LOCAL_OUTPUT));
        session
            .run_compaction(openai_agents_sdk::memory::OpenAIResponsesCompactionArgs {
                force: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(endpoint.requests().len(), 1, "{mode:?}");
        if mode == CompactionMode::Input {
            assert!(
                endpoint.inputs()[0]
                    .iter()
                    .any(|i| i.to_string().contains(LOCAL_OUTPUT))
            );
        }
        assert!(session.get_items(None).await.unwrap().is_empty());
    }
}

fn calls_without_id(items: &[Value], call_id: &str) -> Vec<String> {
    items
        .iter()
        .filter(|i| i["call_id"] == call_id)
        .map(|i| i["type"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn sqlite(id: &str, limit: Option<usize>) -> Arc<SqliteSession> {
    let session = SqliteSession::in_memory(id).unwrap();
    Arc::new(match limit {
        Some(limit) => session.with_settings(SessionSettings::with_limit(limit)),
        None => session,
    })
}

fn limit_100() -> RunConfig<()> {
    RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(100))
}

#[cfg(feature = "encrypt")]
fn encrypted(id: &str, inner: Arc<dyn Session>, ttl: u64) -> Arc<dyn Session> {
    Arc::new(
        openai_agents_sdk::memory::EncryptedSession::new(id, inner, "synthetic-test-key", ttl)
            .unwrap(),
    )
}

// ---------------------------------------------------------------------------------------------
// test_automatic_compaction_respects_backend_id_matching_policy (2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn automatic_compaction_respects_backend_id_matching_policy() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let endpoint = Compaction::replying(vec![]).await;
        let backend = Arc::new(IdStrippingSession(
            ListSession::new("id-stripping").stripping_ids(),
        ));
        let duplicate = user("repeated request");
        backend
            .add_items(vec![duplicate.clone(), duplicate.clone()])
            .await
            .unwrap();
        let session = as_session(always(compaction_session(
            "id-stripping",
            backend.clone(),
            &endpoint,
            CompactionMode::Input,
        )));
        let model = model_with(texts(&["ok", "done"]));
        let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
        run(
            &agent,
            "filtered",
            &session,
            streamed,
            Some(RunConfig::<()>::default().with_session_input_callback(
                |history: Vec<InputItem>, new: Vec<InputItem>| async move {
                    let mut out = history[1..].to_vec();
                    out.extend(new);
                    Ok(out)
                },
            )),
        )
        .await;
        assert!(endpoint.requests().is_empty());
        let stored = values(&session.get_items(None).await.unwrap());
        assert_eq!(
            stored
                .iter()
                .filter(|i| **i == *duplicate.as_value())
                .count(),
            2
        );
        assert!(stored.iter().all(|i| i.get("id").is_none()));

        run(&agent, "complete", &session, streamed, None).await;
        assert_eq!(endpoint.requests().len(), 1);
        let compact_input = &endpoint.inputs()[0];
        assert_eq!(
            compact_input
                .iter()
                .filter(|i| **i == *duplicate.as_value())
                .count(),
            2
        );
        assert!(compact_input.iter().any(|i| i.to_string().contains("done")));
        assert!(session.get_items(None).await.unwrap().is_empty());
    }
}

// ---------------------------------------------------------------------------------------------
// test_automatic_compaction_hook_approves_the_reloaded_snapshot (2 x 2 x 2)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn automatic_compaction_hook_approves_the_reloaded_snapshot() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        for use_encryption in [false, true] {
            for deferred in [false, true] {
                let endpoint = Compaction::replying(vec![]).await;
                let backend = sqlite("hook-snapshot", Some(1));
                let approved_views = Arc::new(Mutex::new(Vec::<String>::new()));
                let views = approved_views.clone();
                let compaction = compaction_session(
                    "hook-snapshot",
                    backend.clone(),
                    &endpoint,
                    CompactionMode::Input,
                )
                .with_should_trigger_compaction(move |context| {
                    let rendered = text(&context.session_items);
                    views.lock().unwrap().push(rendered.clone());
                    !rendered.contains(LOCAL_OUTPUT)
                });
                let session: Arc<dyn Session> = if use_encryption {
                    encrypted("hook-snapshot", Arc::new(compaction), 600)
                } else {
                    Arc::new(compaction)
                };
                session
                    .add_items(vec![user(LOCAL_OUTPUT), user("visible tail")])
                    .await
                    .unwrap();
                let mut turns = Vec::new();
                if deferred {
                    turns.push(vec![function_call("lookup", "{}")]);
                }
                turns.push(vec![assistant_message("ok")]);
                let model = model_with(turns);
                let agent = Agent::<()>::new("worker")
                    .with_model_instance(model.clone())
                    .with_tool(lookup_returning("tool result", false));
                run(&agent, "full read", &session, streamed, Some(limit_100())).await;
                let calls = model.calls();
                assert!(text(&calls.last().unwrap().input).contains(LOCAL_OUTPUT));
                let views = approved_views.lock().unwrap();
                assert!(
                    !views[0].contains(LOCAL_OUTPUT),
                    "first approval saw hidden history"
                );
                assert!(endpoint.requests().is_empty());
                assert!(views.last().unwrap().contains(LOCAL_OUTPUT));
                assert!(text(&session.get_items(Some(100)).await.unwrap()).contains(LOCAL_OUTPUT));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_handoff_filter_cannot_expose_omitted_history_through_compaction (2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn handoff_filter_cannot_expose_omitted_history_through_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let endpoint = Compaction::replying(vec![]).await;
        let session = as_session(always(compaction_session(
            "handoff",
            Arc::new(ListSession::new("handoff")),
            &endpoint,
            CompactionMode::Input,
        )));
        let target_model = model_with(texts(&["done"]));
        let target = Agent::<()>::new("target").with_model_instance(target_model.clone());
        let source_model = model_with(vec![vec![function_call("transfer_to_target", "{}")]]);
        let source = Agent::<()>::new("source")
            .with_model_instance(source_model)
            .with_handoff(
                openai_agents_sdk::handoffs::handoff(target).with_input_filter(|mut data| {
                    data.input_history = Vec::new();
                    data
                }),
            );
        run(&source, LOCAL_OUTPUT, &session, streamed, None).await;
        assert!(!text(&target_model.calls()[0].input).contains(LOCAL_OUTPUT));
        assert!(endpoint.requests().is_empty());
        assert!(text(&session.get_items(None).await.unwrap()).contains(LOCAL_OUTPUT));
    }
}

// ---------------------------------------------------------------------------------------------
// test_encrypted_session_checks_decrypted_history
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn encrypted_session_checks_decrypted_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![]).await;
    let backend = Arc::new(ListSession::new("encrypted"));
    let compaction = always(compaction_session(
        "encrypted",
        backend.clone(),
        &endpoint,
        CompactionMode::Input,
    ));
    let session = encrypted("encrypted", Arc::new(compaction), 600);
    session.add_items(vec![user(LOCAL_OUTPUT)]).await.unwrap();
    let model = model_with(texts(&["ok", "done"]));
    let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
    run(
        &agent,
        "visible",
        &session,
        true,
        Some(RunConfig::<()>::default().with_session_input_callback(
            |_history: Vec<InputItem>, new: Vec<InputItem>| async move { Ok(new) },
        )),
    )
    .await;
    assert!(endpoint.requests().is_empty());
    assert!(text(&session.get_items(None).await.unwrap()).contains(LOCAL_OUTPUT));
    assert!(!text(&backend.get_items(None).await.unwrap()).contains(LOCAL_OUTPUT));
    run(&agent, "full replay", &session, true, None).await;
    assert_eq!(endpoint.requests().len(), 1);
    assert!(
        endpoint.inputs()[0]
            .iter()
            .any(|i| i.to_string().contains(LOCAL_OUTPUT))
    );
}

// ---------------------------------------------------------------------------------------------
// test_resumed_tool_output_stays_local_when_filtered (2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn resumed_tool_output_stays_local_when_filtered() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let endpoint = Compaction::replying(vec![]).await;
        let session = as_session(always(compaction_session(
            "resume",
            Arc::new(ListSession::new("resume")),
            &endpoint,
            CompactionMode::Input,
        )));
        let model = model_with(vec![
            vec![function_call("lookup", "{}")],
            vec![assistant_message("ok")],
        ]);
        let agent = Agent::<()>::new("worker")
            .with_model_instance(model.clone())
            .with_tool(lookup_tool(true))
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
        let interrupted = run(&agent, "first", &session, streamed, None).await;
        let mut state = interrupted.to_state();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        let serialized = state.to_json().unwrap();
        assert!(
            !serialized
                .to_string()
                .contains("_session_compaction_model_exchange")
        );
        let restored = RunState::from_json(serialized).unwrap();
        let resumed = run(&agent, restored, &session, streamed, None).await;
        assert_eq!(resumed.final_output_text().as_deref(), Some(LOCAL_OUTPUT));
        run(
            &agent,
            "second",
            &session,
            streamed,
            Some(
                RunConfig::<()>::default().with_call_model_input_filter(|data| async move {
                    Ok(ModelInputData {
                        input: without_tools(data.model_data.input),
                        instructions: data.model_data.instructions,
                    })
                }),
            ),
        )
        .await;
        assert!(!text(&model.calls().last().unwrap().input).contains(LOCAL_OUTPUT));
        assert!(endpoint.requests().is_empty());
        assert!(text(&session.get_items(None).await.unwrap()).contains(LOCAL_OUTPUT));
    }
}

// ---------------------------------------------------------------------------------------------
// test_pending_approval_defers_automatic_compaction (2 x 2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn pending_approval_defers_automatic_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        for approve in [false, true] {
            for atomic_backend in [false, true] {
                let endpoint = Compaction::replying(vec![summary()]).await;
                let backend: Arc<dyn Session> = if atomic_backend {
                    sqlite("approval", None)
                } else {
                    Arc::new(ListSession::new("approval"))
                };
                let session = as_session(always(compaction_session(
                    "approval",
                    backend,
                    &endpoint,
                    CompactionMode::Input,
                )));
                let model = model_with(vec![
                    vec![function_call_with_id("lookup", "{}", "call_lookup")],
                    vec![assistant_message("resolved")],
                    vec![assistant_message("next")],
                    vec![assistant_message("done")],
                ]);
                let agent = Agent::<()>::new("worker")
                    .with_model_instance(model.clone())
                    .with_tool(lookup_tool(true));
                let interrupted = run(&agent, "first", &session, streamed, None).await;
                assert_eq!(interrupted.interruptions.len(), 1);
                assert!(endpoint.requests().is_empty());
                let stored = values(&session.get_items(None).await.unwrap());
                assert!(stored.iter().any(|i| i["call_id"] == "call_lookup"));

                let mut state =
                    RunState::from_json(interrupted.to_state().to_json().unwrap()).unwrap();
                let item = interrupted.interruptions[0].clone();
                if approve {
                    state.approve(&item, false);
                } else {
                    state.reject(&item, false, None);
                }
                run(&agent, state, &session, streamed, None).await;
                for items in [
                    model_input_values(&model, 1),
                    values(&session.get_items(None).await.unwrap()),
                ] {
                    assert_eq!(
                        calls_without_id(&items, "call_lookup"),
                        ["function_call", "function_call_output"],
                        "approve={approve} atomic={atomic_backend}: {items:#?}"
                    );
                }

                run(&agent, "second", &session, streamed, None).await;
                assert_eq!(endpoint.requests().len(), 1);
                assert_eq!(
                    calls_without_id(&endpoint.inputs()[0], "call_lookup"),
                    ["function_call", "function_call_output"]
                );
                assert_eq!(
                    values(&session.get_items(None).await.unwrap()),
                    vec![summary()]
                );

                run(&agent, "third", &session, streamed, None).await;
                assert_eq!(
                    model_input_values(&model, 3),
                    vec![summary(), json!({"role": "user", "content": "third"})]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_pending_hosted_mcp_approval_defers_automatic_compaction (2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn pending_hosted_mcp_approval_defers_automatic_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    let request = json!({
        "id": "approval-1", "type": "mcp_approval_request", "arguments": "{}",
        "name": "lookup", "server_label": "synthetic",
    });
    let is_request = |item: &Value| {
        item["type"] == "mcp_approval_request"
            && item["id"] == "approval-1"
            && item["name"] == "lookup"
            && item["server_label"] == "synthetic"
            && item["arguments"] == "{}"
    };
    for streamed in [false, true] {
        for approve in [false, true] {
            let endpoint = Compaction::replying(vec![summary()]).await;
            let session = as_session(always(compaction_session(
                "mcp-approval",
                sqlite("mcp-approval", None),
                &endpoint,
                CompactionMode::Input,
            )));
            let model = model_with(vec![
                vec![openai_agents_sdk::items::OutputItem::from_value(
                    request.clone(),
                )],
                vec![assistant_message("resolved")],
                vec![assistant_message("next")],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("worker")
                .with_model_instance(model.clone())
                .with_tool(HostedMCPTool::new(json!({
                    "type": "mcp",
                    "server_label": "synthetic",
                    "server_url": "https://synthetic.example/mcp",
                    "require_approval": "always",
                })));
            let interrupted = run(&agent, "first", &session, streamed, None).await;
            assert_eq!(interrupted.interruptions.len(), 1);
            assert!(endpoint.requests().is_empty());
            assert!(
                values(&session.get_items(None).await.unwrap())
                    .iter()
                    .any(is_request)
            );

            let mut state = RunState::from_json(interrupted.to_state().to_json().unwrap()).unwrap();
            let item = interrupted.interruptions[0].clone();
            if approve {
                state.approve(&item, false);
            } else {
                state.reject(&item, false, None);
            }
            run(&agent, state, &session, streamed, None).await;
            for items in [
                model_input_values(&model, 1),
                values(&session.get_items(None).await.unwrap()),
            ] {
                assert!(items.iter().any(is_request));
                let responses = of_type(&items, "mcp_approval_response");
                assert_eq!(responses.len(), 1);
                assert_eq!(responses[0]["approval_request_id"], "approval-1");
                assert_eq!(responses[0]["approve"], approve);
            }

            run(&agent, "second", &session, streamed, None).await;
            assert_eq!(endpoint.requests().len(), 1);
            let compact_input = &endpoint.inputs()[0];
            assert!(compact_input.iter().any(is_request));
            assert!(
                compact_input
                    .iter()
                    .any(|i| i["type"] == "mcp_approval_response"
                        && i["approval_request_id"] == "approval-1"
                        && i["approve"] == approve)
            );
            assert_eq!(
                values(&session.get_items(None).await.unwrap()),
                vec![summary()]
            );

            run(&agent, "third", &session, streamed, None).await;
            assert_eq!(
                model_input_values(&model, 3),
                vec![summary(), json!({"role": "user", "content": "third"})]
            );
        }
    }
}

fn count_of(items: &[Value], wanted: &Value) -> usize {
    items.iter().filter(|i| *i == wanted).count()
}

fn model_call_filter<F>(f: F) -> RunConfig<()>
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

fn session_callback<F>(f: F) -> RunConfig<()>
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
#[tokio::test]
async fn backend_read_limit_cannot_hide_retained_history_from_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        for use_encryption in [false, true] {
            let endpoint = Compaction::replying(vec![]).await;
            let backend = sqlite("limited", Some(1));
            let compaction = always(compaction_session(
                "limited",
                backend.clone(),
                &endpoint,
                CompactionMode::Input,
            ));
            let session: Arc<dyn Session> = if use_encryption {
                encrypted("limited", Arc::new(compaction), 600)
            } else {
                Arc::new(compaction)
            };
            session
                .add_items(vec![user(LOCAL_OUTPUT), user("visible tail")])
                .await
                .unwrap();
            let model = model_with(texts(&["ok", "done"]));
            let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
            run(&agent, "limited read", &session, streamed, None).await;
            assert!(!text(&model_input(&model, 0)).contains(LOCAL_OUTPUT));
            assert_eq!(endpoint.requests().len(), 1, "encrypted={use_encryption}");
            assert!(
                !endpoint.inputs()[0]
                    .iter()
                    .any(|i| i.to_string().contains(LOCAL_OUTPUT))
            );
            assert!(text(&session.get_items(Some(100)).await.unwrap()).contains(LOCAL_OUTPUT));

            run(&agent, "full read", &session, streamed, Some(limit_100())).await;
            assert!(text(&model_input(&model, 1)).contains(LOCAL_OUTPUT));
            assert_eq!(endpoint.requests().len(), 2);
            assert!(
                endpoint.inputs()[1]
                    .iter()
                    .any(|i| i.to_string().contains(LOCAL_OUTPUT))
            );
            assert!(session.get_items(Some(100)).await.unwrap().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_automatic_compaction_preserves_filtered_duplicate_occurrences (2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn automatic_compaction_preserves_filtered_duplicate_occurrences() {
    openai_agents_sdk::set_tracing_disabled(true);
    let duplicate = json!({"role": "user", "content": "repeated request"});
    for streamed in [false, true] {
        for mode in [CompactionMode::Input, CompactionMode::PreviousResponseId] {
            let endpoint = Compaction::replying(vec![]).await;
            let history = vec![
                InputItem::from_value(duplicate.clone()),
                InputItem::from_value(duplicate.clone()),
            ];
            let session = as_session(always(compaction_session(
                "duplicates",
                Arc::new(ListSession::with_history("duplicates", history)),
                &endpoint,
                mode,
            )));
            let model = model_with(texts(&["ok", "done"]));
            let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
            run(
                &agent,
                "filtered",
                &session,
                streamed,
                Some(session_callback(|history, new| {
                    history[1..].iter().cloned().chain(new).collect()
                })),
            )
            .await;
            assert_eq!(count_of(&model_input_values(&model, 0), &duplicate), 1);
            assert!(endpoint.requests().is_empty());
            let stored = values(&session.get_items(None).await.unwrap());
            assert_eq!(count_of(&stored, &duplicate), 2);

            run(&agent, "complete", &session, streamed, None).await;
            assert_eq!(count_of(&model_input_values(&model, 1), &duplicate), 2);
            assert_eq!(endpoint.requests().len(), 1, "{mode:?}");
            if mode == CompactionMode::Input {
                assert_eq!(count_of(&endpoint.inputs()[0], &duplicate), 2);
            }
            assert!(session.get_items(None).await.unwrap().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_automatic_compaction_preserves_reordered_history (2 x 2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn automatic_compaction_preserves_reordered_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    let first = json!({"role": "user", "content": "first stored request"});
    let second = json!({"role": "user", "content": "second stored request"});
    let model_only = json!({"role": "user", "content": "model-only context"});
    for streamed in [false, true] {
        for mode in [CompactionMode::Input, CompactionMode::PreviousResponseId] {
            for filter_kind in ["model", "session"] {
                let endpoint = Compaction::replying(vec![]).await;
                let history = vec![
                    InputItem::from_value(first.clone()),
                    InputItem::from_value(second.clone()),
                ];
                let session = as_session(always(compaction_session(
                    "ordered",
                    Arc::new(ListSession::with_history("ordered", history)),
                    &endpoint,
                    mode,
                )));
                let model = model_with(texts(&["ok", "done"]));
                let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
                let config = if filter_kind == "model" {
                    model_call_filter(|items| {
                        let mut out = vec![items[1].clone(), items[0].clone()];
                        out.extend(items[2..].iter().cloned());
                        out
                    })
                } else {
                    session_callback(|history, new| history.into_iter().rev().chain(new).collect())
                };
                run(&agent, "reordered", &session, streamed, Some(config)).await;
                assert_eq!(
                    model_input_values(&model, 0)[..2],
                    [second.clone(), first.clone()]
                );
                assert!(endpoint.requests().is_empty(), "{filter_kind} {mode:?}");
                assert_eq!(
                    values(&session.get_items(None).await.unwrap())[..2],
                    [first.clone(), second.clone()]
                );

                let extra = model_only.clone();
                run(
                    &agent,
                    "ordered",
                    &session,
                    streamed,
                    Some(model_call_filter(move |items| {
                        let mut out = vec![items[0].clone(), extra.clone()];
                        out.extend(items[1..].iter().cloned());
                        out
                    })),
                )
                .await;
                assert_eq!(
                    model_input_values(&model, 1)[..3],
                    [first.clone(), model_only.clone(), second.clone()]
                );
                assert_eq!(endpoint.requests().len(), 1, "{filter_kind} {mode:?}");
                if mode == CompactionMode::Input {
                    let compact_input = &endpoint.inputs()[0];
                    assert_eq!(compact_input[..2], [first.clone(), second.clone()]);
                    assert!(!compact_input.contains(&model_only));
                }
                assert!(session.get_items(None).await.unwrap().is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// reasoning item id policy (2 x 2, 2, 2 x 2)
// ---------------------------------------------------------------------------------------------

fn reasoning_output(id: &str) -> openai_agents_sdk::items::OutputItem {
    openai_agents_sdk::items::OutputItem::from_value(
        json!({"id": id, "type": "reasoning", "summary": []}),
    )
}

#[tokio::test]
async fn automatic_compaction_applies_current_reasoning_id_policy() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        for mode in [CompactionMode::Input, CompactionMode::PreviousResponseId] {
            let endpoint = Compaction::replying(vec![]).await;
            let backend = sqlite("reasoning-policy", None);
            let approving = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let decisions = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));
            let (flag, seen) = (approving.clone(), decisions.clone());
            let session = as_session(
                compaction_session("reasoning-policy", backend.clone(), &endpoint, mode)
                    .with_should_trigger_compaction(move |context| {
                        if !flag.load(std::sync::atomic::Ordering::SeqCst) {
                            return false;
                        }
                        seen.lock().unwrap().push(values(&context.session_items));
                        true
                    }),
            );
            let model = model_with(vec![
                vec![reasoning_output("rs_stored"), assistant_message("old")],
                vec![assistant_message("done")],
            ]);
            let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
            run(&agent, "before policy", &session, streamed, None).await;
            let stored = values(&backend.get_items(None).await.unwrap());
            let stored_reasoning = of_type(&stored, "reasoning");
            assert_eq!(stored_reasoning[0]["id"], "rs_stored");

            approving.store(true, std::sync::atomic::Ordering::SeqCst);
            run(
                &agent,
                "adopt policy",
                &session,
                streamed,
                Some(
                    RunConfig::<()>::default()
                        .with_reasoning_item_id_policy(ReasoningItemIdPolicy::Omit),
                ),
            )
            .await;
            let sent = model_input_values(&model, 1);
            let sent_reasoning = of_type(&sent, "reasoning")[0].clone();
            assert!(sent_reasoning.get("id").is_none());
            let decisions = decisions.lock().unwrap();
            assert_eq!(decisions.len(), 2, "{mode:?}");
            let approved = of_type(&decisions[decisions.len() - 1], "reasoning")[0].clone();
            assert_eq!(approved, sent_reasoning);
            assert_eq!(endpoint.requests().len(), 1, "{mode:?}");
            if mode == CompactionMode::Input {
                let compact_input = &endpoint.inputs()[0];
                assert_eq!(*of_type(compact_input, "reasoning")[0], sent_reasoning);
            }
            assert!(backend.get_items(None).await.unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn reasoning_policy_does_not_authorize_other_input_changes() {
    openai_agents_sdk::set_tracing_disabled(true);
    let reasoning = json!({
        "type": "reasoning", "id": "rs_stored", "summary": [],
        "encrypted_content": "synthetic-content",
    });
    for omit_policy in [false, true] {
        let endpoint = Compaction::replying(vec![]).await;
        let session = as_session(always(compaction_session(
            "reasoning-filter",
            Arc::new(ListSession::with_history(
                "reasoning-filter",
                vec![InputItem::from_value(reasoning.clone())],
            )),
            &endpoint,
            CompactionMode::Input,
        )));
        let model = model_with(texts(&["done"]));
        let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
        let mut config = model_call_filter(move |mut items| {
            if omit_policy {
                items[0]["encrypted_content"] = json!("different-content");
            } else if let Value::Object(map) = &mut items[0] {
                map.remove("id");
            }
            items
        });
        if omit_policy {
            config = config.with_reasoning_item_id_policy(ReasoningItemIdPolicy::Omit);
        }
        run(&agent, "filtered", &session, false, Some(config)).await;
        assert!(model_input_values(&model, 0)[0].get("id").is_none());
        assert!(endpoint.requests().is_empty(), "omit={omit_policy}");
        assert_eq!(
            values(&session.get_items(None).await.unwrap())[0],
            reasoning
        );
    }
}

#[tokio::test]
async fn omit_policy_preserves_model_visible_reasoning_ids() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        for restore_history_id in [false, true] {
            let endpoint = Compaction::replying(vec![]).await;
            let backend = sqlite("reasoning-caller-input", None);
            let decisions = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));
            let seen = decisions.clone();
            let session = as_session(
                compaction_session(
                    "reasoning-caller-input",
                    backend.clone(),
                    &endpoint,
                    CompactionMode::Input,
                )
                .with_should_trigger_compaction(move |context| {
                    seen.lock().unwrap().push(values(&context.session_items));
                    true
                }),
            );
            session
                .add_items(vec![InputItem::from_value(
                    json!({"type": "reasoning", "id": "rs_stored", "summary": []}),
                )])
                .await
                .unwrap();
            let model = model_with(texts(&["done"]));
            let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
            let prompt = vec![
                InputItem::from_value(
                    json!({"type": "reasoning", "id": "rs_caller", "summary": []}),
                ),
                user("continue"),
            ];
            let mut config =
                session_callback(|history, new| history.into_iter().chain(new).collect())
                    .with_reasoning_item_id_policy(ReasoningItemIdPolicy::Omit);
            if restore_history_id {
                let restore = |mut items: Vec<Value>| {
                    items[0]["id"] = json!("rs_stored");
                    items
                };
                let callback = model_call_filter(restore);
                config = config.with_call_model_input_filter(move |data| {
                    let mut items = values(&data.model_data.input);
                    items[0]["id"] = json!("rs_stored");
                    let _ = &callback;
                    async move {
                        Ok(ModelInputData {
                            input: items.into_iter().map(InputItem::from_value).collect(),
                            instructions: data.model_data.instructions,
                        })
                    }
                });
            }
            run(&agent, prompt, &session, streamed, Some(config)).await;
            let sent = model_input_values(&model, 0);
            let sent_reasoning: Vec<Value> =
                of_type(&sent, "reasoning").into_iter().cloned().collect();
            assert_eq!(
                sent_reasoning[0].get("id").and_then(Value::as_str),
                restore_history_id.then_some("rs_stored")
            );
            assert_eq!(sent_reasoning[1]["id"], "rs_caller");
            assert_eq!(endpoint.requests().len(), 1);
            let compact_reasoning: Vec<Value> = of_type(&endpoint.inputs()[0], "reasoning")
                .into_iter()
                .cloned()
                .collect();
            assert_eq!(compact_reasoning, sent_reasoning);
            let last = decisions.lock().unwrap().last().unwrap().clone();
            let decided: Vec<Value> = of_type(&last, "reasoning").into_iter().cloned().collect();
            assert_eq!(decided, sent_reasoning);
            assert!(backend.get_items(None).await.unwrap().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Encrypted history that expires. Python moves a mocked clock; here the items' lifetime is a few
// seconds of real time, so each test lets the first items lapse and then finishes inside the
// lifetime of the items written afterwards.
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "encrypt")]
const TTL: u64 = 3;

#[cfg(feature = "encrypt")]
async fn let_items_expire() {
    tokio::time::sleep(std::time::Duration::from_millis(4200)).await;
}

/// A compaction session over an encrypted view of `backend` (`outer == false`) or an encrypted
/// session over a compaction session (`outer == true`), with the compaction session handle
/// that manual `run_compaction` calls need.
#[cfg(feature = "encrypt")]
fn expiring_stack(
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

fn force() -> openai_agents_sdk::memory::OpenAIResponsesCompactionArgs {
    openai_agents_sdk::memory::OpenAIResponsesCompactionArgs {
        force: true,
        ..Default::default()
    }
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn manual_compaction_refreshes_expired_encrypted_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    for outer in [false, true] {
        // Echo the request so stale input would also be written back with a fresh lifetime.
        let endpoint = Compaction::with(
            |body| json!({"id": "cmp", "output": body["input"].clone(), "usage": usage()}),
        )
        .await;
        let backend = sqlite("manual-expiry", None);
        let (session, compaction) = expiring_stack(
            outer,
            "manual-expiry",
            backend.clone(),
            &endpoint,
            Some(false),
        );
        session
            .add_items(vec![InputItem::from_value(
                json!({"role": "assistant", "content": "synthetic expired item"}),
            )])
            .await
            .unwrap();
        // A declined policy check can populate the cache.
        compaction.run_compaction(Default::default()).await.unwrap();
        assert!(endpoint.requests().is_empty());

        let_items_expire().await;
        assert!(session.get_items(None).await.unwrap().is_empty());
        compaction.run_compaction(force()).await.unwrap();
        assert_eq!(endpoint.requests().len(), 1, "outer={outer}");
        assert_eq!(endpoint.requests()[0]["model"], "gpt-4.1");
        assert_eq!(endpoint.inputs()[0], Vec::<Value>::new());
        assert!(session.get_items(None).await.unwrap().is_empty());

        let live = json!({"role": "user", "content": "synthetic live item"});
        session
            .add_items(vec![InputItem::from_value(live.clone())])
            .await
            .unwrap();
        compaction.run_compaction(force()).await.unwrap();
        assert_eq!(endpoint.inputs()[1], vec![live.clone()], "outer={outer}");
        assert_eq!(values(&session.get_items(None).await.unwrap()), vec![live]);
    }
}

#[cfg(feature = "encrypt")]
async fn recovers_after_expiry(streamed: bool, outer: bool, limited_default: bool) {
    let endpoint = Compaction::replying(vec![]).await;
    let backend = sqlite("expiry", limited_default.then_some(20));
    let (session, _) = expiring_stack(outer, "expiry", backend.clone(), &endpoint, None);
    let model = model_with(texts(&["ok"; 15]));
    let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
    session
        .add_items(vec![user("expired request")])
        .await
        .unwrap();
    let_items_expire().await;
    for index in 0..15 {
        run(
            &agent,
            format!("turn {index}"),
            &session,
            streamed,
            limited_default.then(limit_100),
        )
        .await;
    }
    let label = format!("streamed={streamed} outer={outer} limited={limited_default}");
    assert_eq!(endpoint.requests().len(), 1, "{label}");
    assert_eq!(endpoint.inputs()[0].len(), 20, "{label}");
    assert!(
        !endpoint.inputs()[0]
            .iter()
            .any(|i| i.to_string().contains("expired request")),
        "{label}"
    );
    assert_eq!(
        backend.get_items(Some(100)).await.unwrap().len(),
        10,
        "{label}"
    );
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn automatic_compaction_recovers_after_encrypted_history_expires() {
    openai_agents_sdk::set_tracing_disabled(true);
    let mut cases = Vec::new();
    for streamed in [false, true] {
        for outer in [false, true] {
            for limited_default in [false, true] {
                cases.push(recovers_after_expiry(streamed, outer, limited_default));
            }
        }
    }
    futures::future::join_all(cases).await;
}

/// Records the limit of every history read and snapshot read, delegating to a real store.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<SqliteSession>,
    reads: Mutex<Vec<Option<usize>>>,
    snapshot_reads: Mutex<Vec<usize>>,
}

impl CountingStore {
    fn new(inner: Arc<SqliteSession>) -> Arc<Self> {
        Arc::new(CountingStore {
            inner,
            reads: Mutex::new(Vec::new()),
            snapshot_reads: Mutex::new(Vec::new()),
        })
    }
    fn reset(&self) {
        self.reads.lock().unwrap().clear();
        self.snapshot_reads.lock().unwrap().clear();
    }
}

#[async_trait::async_trait]
impl Session for CountingStore {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    async fn get_items(
        &self,
        limit: Option<usize>,
    ) -> openai_agents_sdk::error::Result<Vec<InputItem>> {
        self.reads.lock().unwrap().push(limit);
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::error::Result<()> {
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> openai_agents_sdk::error::Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> openai_agents_sdk::error::Result<()> {
        self.inner.clear_session().await
    }
    async fn compaction_snapshot(
        &self,
        limit: usize,
    ) -> openai_agents_sdk::error::Result<Option<openai_agents_sdk::memory::CompactionSnapshot>>
    {
        self.snapshot_reads.lock().unwrap().push(limit);
        self.inner.compaction_snapshot(limit).await
    }
    async fn replace_compaction_suffix(
        &self,
        handles: &[openai_agents_sdk::memory::SnapshotHandle],
        items: Vec<InputItem>,
    ) -> openai_agents_sdk::error::Result<bool> {
        self.inner.replace_compaction_suffix(handles, items).await
    }
    async fn replace_compaction_suffix_pruning(
        &self,
        handles: &[openai_agents_sdk::memory::SnapshotHandle],
        items: Vec<InputItem>,
        prune_prefix: openai_agents_sdk::memory::PrunePrefix,
    ) -> openai_agents_sdk::error::Result<bool> {
        self.inner
            .replace_compaction_suffix_pruning(handles, items, prune_prefix)
            .await
    }
}

// ---------------------------------------------------------------------------------------------
// test_automatic_compaction_does_not_reload_unbounded_hidden_history (2 x 3)
//
// Python counts reads on the SQLite backend. Encryption cases wait for the first 100 items to
// expire instead of moving a clock.
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "encrypt")]
async fn does_not_reload_unbounded_history(approve: bool, encryption: &str) {
    let endpoint = Compaction::replying(vec![]).await;
    let sqlite_backend = sqlite("bounded", Some(1));
    let backend = CountingStore::new(sqlite_backend.clone());
    let build = |store: Arc<dyn Session>| {
        Arc::new(
            compaction_session("bounded", store, &endpoint, CompactionMode::Input)
                .with_should_trigger_compaction(move |_| approve),
        )
    };
    let compaction: Arc<OpenAIResponsesCompactionSession>;
    let session: Arc<dyn Session> = match encryption {
        "none" => {
            compaction = build(backend.clone());
            compaction.clone()
        }
        "outer" => {
            compaction = build(backend.clone());
            encrypted("bounded", compaction.clone(), TTL)
        }
        _ => {
            compaction = build(encrypted("bounded", backend.clone(), TTL));
            compaction.clone()
        }
    };
    let items: Vec<InputItem> = (0..100).map(|i| user(&format!("retained-{i}"))).collect();
    session.add_items(items).await.unwrap();
    if encryption != "none" {
        let_items_expire().await;
        session.add_items(vec![user("fresh tail")]).await.unwrap();
    }
    let model = model_with(texts(&["ok", "done"]));
    let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
    for prompt in ["first", "second"] {
        backend.reset();
        run(&agent, prompt, &session, false, None).await;
        let limits = backend.reads.lock().unwrap().clone();
        let snapshots = backend.snapshot_reads.lock().unwrap().clone();
        let label = format!("approve={approve} encryption={encryption} {prompt}");
        if approve {
            assert!(
                snapshots.iter().all(|limit| *limit <= 4),
                "{label}: {snapshots:?}"
            );
            assert_eq!(snapshots.len(), 1, "{label}");
        } else {
            assert!(
                limits.iter().all(|limit| limit.is_none_or(|n| n == 1)),
                "{label}: {limits:?}"
            );
            assert!(snapshots.is_empty(), "{label}");
            assert!(endpoint.requests().is_empty(), "{label}");
        }
    }
    assert_eq!(endpoint.requests().len(), if approve { 2 } else { 0 });
    let raw = sqlite_backend.get_items(Some(1000)).await.unwrap().len();
    let expected = match (approve, encryption != "none") {
        (true, true) => 0,
        (true, false) => 98,
        (false, true) => 105,
        (false, false) => 104,
    };
    assert_eq!(raw, expected, "approve={approve} encryption={encryption}");
    // Explicit manual compaction can still process the application's approved logical history.
    compaction.run_compaction(force()).await.unwrap();
    let requests = endpoint.requests().len();
    assert_eq!(requests, if approve { 3 } else { 1 });
    if !approve {
        assert!(
            endpoint
                .inputs()
                .last()
                .unwrap()
                .iter()
                .any(|i| i.to_string().contains("done"))
        );
    }
    assert!(
        sqlite_backend
            .get_items(Some(1000))
            .await
            .unwrap()
            .is_empty()
    );
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn automatic_compaction_does_not_reload_unbounded_hidden_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    let mut cases = Vec::new();
    for approve in [false, true] {
        for encryption in ["none", "outer", "inner"] {
            cases.push(does_not_reload_unbounded_history(approve, encryption));
        }
    }
    futures::future::join_all(cases).await;
}

// ---------------------------------------------------------------------------------------------
// test_expired_read_overhead_does_not_authorize_hidden_history (2)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "encrypt")]
async fn expired_read_overhead(outer: bool) {
    let endpoint = Compaction::replying(vec![]).await;
    let sqlite_backend = sqlite("expiry-hidden", Some(1));
    let backend = CountingStore::new(sqlite_backend.clone());
    let (session, _) = expiring_stack(
        outer,
        "expiry-hidden",
        backend.clone(),
        &endpoint,
        Some(true),
    );
    let hide = || {
        model_call_filter(|items| {
            let mut out: Vec<Value> = items
                .into_iter()
                .filter(|i| i["content"] != LOCAL_OUTPUT)
                .collect();
            out.push(json!({"role": "user", "content": "model-only context"}));
            out
        })
    };
    let model = model_with(texts(&["ok", "ok", "ok"]));
    let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
    session
        .add_items((0..10).map(|_| user("expired")).collect())
        .await
        .unwrap();
    let_items_expire().await;
    session.add_items(vec![user(LOCAL_OUTPUT)]).await.unwrap();
    run(
        &agent,
        "filtered",
        &session,
        false,
        Some(hide().with_session_settings(SessionSettings::with_limit(100))),
    )
    .await;
    let label = format!("outer={outer}");
    assert_eq!(endpoint.requests().len(), 1, "{label}");
    assert!(
        !text(&model_input(&model, 0)).contains(LOCAL_OUTPUT),
        "{label}"
    );
    assert!(
        !endpoint.inputs()[0]
            .iter()
            .any(|i| i.to_string().contains(LOCAL_OUTPUT)),
        "{label}"
    );
    assert_eq!(
        sqlite_backend.get_items(Some(100)).await.unwrap().len(),
        1,
        "{label}"
    );
    assert!(text(&session.get_items(Some(100)).await.unwrap()).contains(LOCAL_OUTPUT));

    backend.reset();
    run(&agent, "bounded", &session, false, Some(hide())).await;
    // The previous full read's overhead is not carried into a new bounded turn.
    let limits = backend.reads.lock().unwrap().clone();
    assert!(
        limits.iter().all(|limit| limit.is_none_or(|n| n <= 4)),
        "{label}: {limits:?}"
    );
    assert_eq!(endpoint.requests().len(), 2, "{label}");
    assert!(
        !endpoint.inputs()[1]
            .iter()
            .any(|i| i.to_string().contains(LOCAL_OUTPUT)),
        "{label}"
    );
    run(&agent, "visible", &session, false, Some(limit_100())).await;
    assert_eq!(endpoint.requests().len(), 3, "{label}");
    assert!(
        endpoint.inputs()[2]
            .iter()
            .any(|i| i.to_string().contains(LOCAL_OUTPUT)),
        "{label}"
    );
    assert!(
        session.get_items(Some(100)).await.unwrap().is_empty(),
        "{label}"
    );
}

#[cfg(feature = "encrypt")]
#[tokio::test]
async fn expired_read_overhead_does_not_authorize_hidden_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    futures::future::join_all([expired_read_overhead(false), expired_read_overhead(true)]).await;
}
