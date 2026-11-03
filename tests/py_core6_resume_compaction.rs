//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py`: a Session append that
//! fails around a resumed approval or handoff must leave Responses compaction recoverable: the
//! committed history is what compaction sees, the owed compaction runs once, and a rolled-back
//! replacement keeps its ownership without repeating the append.
//!
//! Python injects a `SimpleNamespace` compact client; Rust's compaction session talks HTTP, so a
//! local mock `/v1/responses/compact` endpoint echoes the request `input` back as its output and
//! records each request.

mod py_core6_resume_support;
mod py_mcpmem_support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::{InputItem, RunItemKind};
use openai_agents_sdk::memory::{
    CompactionContext, CompactionMode, OpenAIResponsesCompactionSession, Session, SessionSettings,
};
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, AgentsError, OpenAIClient};
use py_core6_resume_support::*;
use py_mcpmem_support::{Request, mock_openai_with};
use serde_json::{Value, json};

/// A backing store with scripted append failures, standing in for Python's
/// `_LostAckSQLiteSession`, `_FailSecondAddItemsSession` and `FailReplacementSession`.
#[derive(Debug)]
struct Backend {
    id: String,
    items: Mutex<Vec<InputItem>>,
    settings: Option<SessionSettings>,
    /// Fail the next append after committing it (a lost acknowledgement).
    fail_after_commit: Mutex<bool>,
    /// Fail the n-th append (1-based) before writing.
    fail_nth: Mutex<Option<usize>>,
    adds: Mutex<usize>,
    /// Replacement rollback: fail the append of exactly this batch once, then fail restores.
    replacement: Mutex<Option<(&'static str, Option<&'static str>, bool)>>,
}

impl Backend {
    fn new(id: &str) -> Arc<Self> {
        Arc::new(Backend {
            id: id.into(),
            items: Mutex::new(Vec::new()),
            settings: None,
            fail_after_commit: Mutex::new(false),
            fail_nth: Mutex::new(None),
            adds: Mutex::new(0),
            replacement: Mutex::new(None),
        })
    }

    fn stored(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }
}

#[async_trait]
impl Session for Backend {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let items = self.stored();
        Ok(match limit {
            Some(n) if items.len() > n => items[items.len() - n..].to_vec(),
            _ => items,
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let n = {
            let mut adds = self.adds.lock().unwrap();
            *adds += 1;
            *adds
        };
        if *self.fail_nth.lock().unwrap() == Some(n) {
            *self.fail_nth.lock().unwrap() = None;
            return Err(AgentsError::other("session append failed"));
        }
        {
            let mut replacement = self.replacement.lock().unwrap();
            if let Some((target, restore_failure, failed)) = replacement.as_mut() {
                if !*failed && items.iter().any(|i| i.0.to_string().contains(*target)) {
                    *failed = true;
                    return Err(AgentsError::other("replacement failed before commit"));
                }
                if *failed {
                    match restore_failure {
                        Some("partial") => {
                            self.items.lock().unwrap().extend(items.into_iter().take(1));
                            return Err(AgentsError::other("restoration failed"));
                        }
                        Some(_) => return Err(AgentsError::other("restoration failed")),
                        None => {}
                    }
                }
            }
        }
        self.items.lock().unwrap().extend(items);
        let mut lost = self.fail_after_commit.lock().unwrap();
        if std::mem::take(&mut *lost) {
            return Err(AgentsError::other("session append failed"));
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

/// A compaction session over `backend` whose compact endpoint echoes the request input (or the
/// fixed `output`) and whose decision hook is `trigger`.
async fn compaction_over(
    backend: Arc<Backend>,
    output: Option<Vec<Value>>,
    trigger: impl Fn(&CompactionContext) -> bool + Send + Sync + 'static,
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

fn compact_inputs(captured: &Captured) -> Vec<Vec<InputItem>> {
    captured
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.path.ends_with("/responses/compact"))
        .map(|r| {
            r.body["input"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(InputItem)
                .collect()
        })
        .collect()
}

async fn drive_dyn(
    agent: &Agent<()>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    session: Option<Arc<dyn Session>>,
    streamed: bool,
    slot: &RunCheckpoint,
) -> std::result::Result<Outcome, AgentsError> {
    use futures::StreamExt;
    let mut options = openai_agents_sdk::RunOptions::new(()).checkpoint(slot);
    if let Some(session) = session {
        options = options.session(session);
    }
    if streamed {
        let mut result =
            openai_agents_sdk::Runner::run_streamed_with_options(agent, input, options);
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
        let result = openai_agents_sdk::Runner::run_with_options(agent, input, options).await?;
        Ok(Outcome {
            final_output: result.final_output.clone(),
            interruptions: result.interruptions.clone(),
            state: Some(result.to_state()),
            input_list: result.to_input_list(),
        })
    }
}

fn dyn_session(s: &Arc<OpenAIResponsesCompactionSession>) -> Option<Arc<dyn Session>> {
    Some(s.clone() as Arc<dyn Session>)
}

fn response_step(output: Vec<openai_agents_sdk::items::OutputItem>, id: &str) -> ScriptedStep {
    ScriptedStep::output(output).with_response_id(Some(id.to_string()))
}

fn handoff_agents(
    model: &Arc<ScriptedModel>,
    retain_messages_only: bool,
) -> (Agent<()>, Arc<Mutex<Vec<String>>>) {
    let calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = calls.clone();
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let mut route = handoff(delegate).with_on_handoff(move |_c| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push("handoff".into());
            Ok(())
        }
    });
    if retain_messages_only {
        route = route.with_input_filter(|mut data| {
            data.new_items
                .retain(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }));
            data
        });
    }
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_handoff(route);
    (triage, calls)
}

fn handoff_only_model(with_text: bool) -> Arc<ScriptedModel> {
    let mut first = Vec::new();
    if with_text {
        first.push(assistant_message("delegating"));
    }
    first.push(call("transfer_to_delegate", json!({}), "handoff-1"));
    let model = Arc::new(ScriptedModel::new());
    model.enqueue(response_step(first, "resp-handoff")).unwrap();
    model
        .enqueue(response_step(
            vec![assistant_message("done")],
            "resp-delegate",
        ))
        .unwrap();
    model
}

#[tokio::test]
async fn fresh_streamed_handoff_replays_deferred_compaction_after_resume() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for json_trip in [false, true] {
        let backend = Backend::new("compaction-handoff-test");
        let hook_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = hook_calls.clone();
        let (session, captured) = compaction_over(backend.clone(), None, move |ctx| {
            let id = ctx.response_id.clone().unwrap_or_default();
            seen.lock().unwrap().push(id.clone());
            id == "resp-handoff"
        })
        .await;
        let model = handoff_only_model(false);
        let (triage, _) = handoff_agents(&model, false);
        // The initial input append succeeds; the next one (the handoff batch) fails.
        *backend.fail_nth.lock().unwrap() = Some(2);
        let slot = RunCheckpoint::new();
        let err = drive_dyn(&triage, "hello", dyn_session(&session), true, &slot)
            .await
            .err()
            .expect("the append fails");
        assert!(err.to_string().contains("session append failed"), "{err}");
        assert!(hook_calls.lock().unwrap().is_empty());
        assert!(compact_inputs(&captured).is_empty());
        let mut state = slot.take().expect("checkpoint");
        let pending = state.pending_session_write.clone().expect("pending write");
        assert_eq!(pending["response_id"], json!("resp-handoff"));
        assert_eq!(pending["has_local_tool_outputs"], json!(true));
        if json_trip {
            state = round_trip(&state);
        }
        let result = drive_dyn(
            &triage,
            state,
            dyn_session(&session),
            false,
            &RunCheckpoint::new(),
        )
        .await
        .unwrap();
        assert_eq!(result.final_output, Some(json!("done")));
        // The handoff's own response id is evaluated by the decision hook and deferred, and the
        // deferred decision is forced through on the delegate's save.
        assert_eq!(
            *hook_calls.lock().unwrap(),
            vec!["resp-handoff"],
            "json={json_trip}"
        );
        assert_eq!(compact_inputs(&captured).len(), 1, "json={json_trip}");
    }
}

/// Wraps a compaction session so its first completion notice fails, standing in for Python's
/// decision hook that raises once (a Rust hook returns a bool and cannot fail).
#[derive(Debug)]
struct FlakyNotice {
    inner: Arc<OpenAIResponsesCompactionSession>,
    fail_first: Mutex<bool>,
}

#[async_trait]
impl Session for FlakyNotice {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
    fn wants_model_exchange(&self) -> bool {
        self.inner.wants_model_exchange()
    }
    async fn on_run_completed(
        &self,
        info: &openai_agents_sdk::memory::SessionRunCompletion,
    ) -> Result<openai_agents_sdk::usage::Usage> {
        if std::mem::take(&mut *self.fail_first.lock().unwrap()) {
            return Err(AgentsError::other("compaction decision hook exploded"));
        }
        self.inner.on_run_completed(info).await
    }
}

#[tokio::test]
async fn fresh_streamed_handoff_retains_checkpoint_when_post_write_compaction_fails() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let backend = Backend::new("compaction-handoff-failure-test");
    let (inner, captured) = compaction_over(backend.clone(), None, |ctx| {
        ctx.response_id.as_deref() == Some("resp-handoff")
    })
    .await;
    let flaky = Arc::new(FlakyNotice {
        inner,
        fail_first: Mutex::new(false),
    });
    let session: Arc<dyn Session> = flaky.clone();
    let model = handoff_only_model(false);
    let (triage, _) = handoff_agents(&model, false);
    *backend.fail_nth.lock().unwrap() = Some(2);
    let slot = RunCheckpoint::new();
    let err = drive_dyn(&triage, "hello", Some(session.clone()), true, &slot)
        .await
        .err()
        .expect("the append fails");
    assert!(err.to_string().contains("session append failed"), "{err}");
    let state = slot.take().expect("checkpoint");
    assert!(state.pending_session_write.is_some());

    // The append now succeeds, but the completion notice for the handoff's response fails.
    *flaky.fail_first.lock().unwrap() = true;
    let slot = RunCheckpoint::new();
    let err = drive_dyn(&triage, state, Some(session.clone()), false, &slot)
        .await
        .err()
        .expect("the notice fails");
    assert!(
        err.to_string()
            .contains("compaction decision hook exploded"),
        "{err}"
    );
    // The checkpoint survives so a retry can redo just the compaction step.
    let state = slot.take().expect("checkpoint");
    let pending = state
        .pending_session_write
        .clone()
        .expect("pending write retained");
    assert_eq!(pending["response_id"], json!("resp-handoff"));

    // The retry must not repeat the append, and compaction must actually run this time.
    let result = drive_dyn(
        &triage,
        state,
        Some(session.clone()),
        false,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(json!("done")));
    assert_eq!(compact_inputs(&captured).len(), 1);
    let stored = backend.stored();
    let handoff_pair: Vec<String> = stored
        .iter()
        .filter(|i| i.0.get("call_id").and_then(Value::as_str) == Some("handoff-1"))
        .filter_map(|i| i.0.get("type").and_then(Value::as_str).map(str::to_string))
        .collect();
    // The forced compaction replaced the stored history, so the pair appears at most once.
    assert!(handoff_pair.len() <= 2, "{handoff_pair:?}");
}

#[tokio::test]
async fn resumed_committed_append_refreshes_compaction_input() {
    use std::sync::atomic::{AtomicBool, Ordering};
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for json_trip in [false, true] {
            let label = format!("streamed={streamed} json={json_trip}");
            let backend = Backend::new("compaction-recovery");
            let enabled = Arc::new(AtomicBool::new(false));
            let flag = enabled.clone();
            let (session, captured) =
                compaction_over(backend.clone(), None, move |_| flag.load(Ordering::SeqCst)).await;
            let effects = Arc::new(Mutex::new(Vec::new()));
            let model = Arc::new(ScriptedModel::new());
            for (output, id) in [
                (
                    vec![call("charge", json!({"amount": 7}), "charge-1")],
                    "resp-1",
                ),
                (vec![assistant_message("done")], "resp-2"),
                (vec![assistant_message("fresh")], "resp-3"),
            ] {
                model.enqueue(response_step(output, id)).unwrap();
            }
            let agent = Agent::<()>::new("payment")
                .with_model_instance(model.clone())
                .with_tool(charge_tool(&effects));
            let paused = drive_dyn(
                &agent,
                "charge 7",
                dyn_session(&session),
                streamed,
                &RunCheckpoint::new(),
            )
            .await
            .unwrap();
            let mut state = round_trip(paused.state.as_ref().unwrap());
            let first = state.get_interruptions()[0].clone();
            state.approve(&first, false);
            // A declined compaction initializes the retained wrapper's history cache.
            session.run_compaction(Default::default()).await.unwrap();
            assert!(compact_inputs(&captured).is_empty(), "{label}");

            *backend.fail_after_commit.lock().unwrap() = true;
            let slot = RunCheckpoint::new();
            let err = drive_dyn(&agent, state, dyn_session(&session), streamed, &slot)
                .await
                .err()
                .expect("the lost acknowledgement surfaces");
            assert!(
                err.to_string().contains("session append failed"),
                "{label}: {err}"
            );
            assert_eq!(
                charge_pair(&backend.get_items(Some(100)).await.unwrap()),
                pair(),
                "{label}"
            );
            let mut state = slot.take().unwrap();
            if json_trip {
                state = round_trip(&state);
            }
            enabled.store(true, Ordering::SeqCst);
            let result = drive_dyn(
                &agent,
                state,
                dyn_session(&session),
                streamed,
                &RunCheckpoint::new(),
            )
            .await
            .unwrap();
            assert_eq!(result.final_output, Some(json!("done")), "{label}");
            assert_eq!(*effects.lock().unwrap(), vec![7], "{label}");
            assert_eq!(model.calls().len(), 2, "{label}");
            let inputs = compact_inputs(&captured);
            assert_eq!(inputs.len(), 1, "{label}");
            assert_eq!(charge_pair(&inputs[0]), pair(), "{label}");
            assert_eq!(
                charge_pair(&backend.get_items(Some(100)).await.unwrap()),
                pair(),
                "{label}"
            );
            assert_eq!(charge_pair(&result.input_list), pair(), "{label}");
            assert!(
                result
                    .state
                    .unwrap()
                    .to_json()
                    .unwrap()
                    .get("pending_session_write")
                    .is_none(),
                "{label}"
            );
        }
    }
}

fn text_of(item: &InputItem) -> String {
    item.0.to_string()
}

#[tokio::test]
async fn handoff_resume_retries_rolled_back_compaction() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let compacted = json!({"type": "message", "role": "assistant", "content": "compacted history"});
    let mut failures: Vec<String> = Vec::new();
    for retry_streamed in [false, true] {
        for json_trip in [false, true] {
            for restore_failure in [None, Some("empty"), Some("partial")] {
                let label =
                    format!("rs={retry_streamed} json={json_trip} restore={restore_failure:?}");
                if let Err(e) =
                    rolled_back_case(&compacted, retry_streamed, json_trip, restore_failure).await
                {
                    failures.push(format!("{label}: {e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

async fn rolled_back_case(
    compacted: &Value,
    retry_streamed: bool,
    json_trip: bool,
    restore_failure: Option<&'static str>,
) -> std::result::Result<(), String> {
    macro_rules! ck {
        ($c:expr) => {
            if !$c {
                return Err(stringify!($c).to_string());
            }
        };
    }
    let backend = Backend::new("rollback-compaction");
    *backend.replacement.lock().unwrap() = Some(("compacted history", restore_failure, false));
    let (session, captured) =
        compaction_over(backend.clone(), Some(vec![compacted.clone()]), |ctx| {
            ctx.response_id.as_deref() == Some("resp-handoff")
        })
        .await;
    let model = handoff_only_model(true);
    let (triage, handoff_calls) = handoff_agents(&model, true);
    let slot = RunCheckpoint::new();
    let err = drive_dyn(&triage, "hello", dyn_session(&session), true, &slot)
        .await
        .err()
        .ok_or_else(|| {
            format!(
                "the replacement must fail (compactions={}, stored={:?})",
                compact_inputs(&captured).len(),
                backend
                    .stored()
                    .iter()
                    .map(|i| i.0.to_string())
                    .collect::<Vec<_>>()
            )
        })?;
    ck!(err.to_string().contains("replacement failed before commit"));
    let restored_history = backend.stored();
    ck!(model.calls().len() == 1);
    let mut state = slot.take().ok_or("no checkpoint")?;
    if json_trip {
        state = round_trip(&state);
    }
    if let Some(mode) = restore_failure {
        ck!(restored_history.len() == if mode == "partial" { 1 } else { 0 });
        let err = drive_dyn(
            &triage,
            state.clone(),
            dyn_session(&session),
            retry_streamed,
            &RunCheckpoint::new(),
        )
        .await
        .err()
        .ok_or("the resume must be refused")?;
        ck!(matches!(err, AgentsError::UserError { .. }));
        ck!(err
            .to_string()
            .contains("Cannot reconcile the pending Session write"));
        ck!(state
            .pending_session_write
            .as_ref()
            .is_some_and(|p| p["append_acknowledged"] == json!(true)));
        ck!(backend.stored() == restored_history);
        ck!(model.calls().len() == 1);
        ck!(compact_inputs(&captured).len() == 1);
        ck!(*handoff_calls.lock().unwrap() == vec!["handoff"]);
        return Ok(());
    }
    ck!(restored_history.len() == 2);
    let resumed = drive_dyn(
        &triage,
        state,
        dyn_session(&session),
        retry_streamed,
        &RunCheckpoint::new(),
    )
    .await
    .map_err(|e| format!("retry failed: {e}"))?;
    ck!(resumed.final_output == Some(json!("done")));
    let inputs = compact_inputs(&captured);
    ck!(inputs.len() == 2);
    if inputs[0] != inputs[1] {
        return Err(format!(
            "{:?}\n{:?}",
            inputs[0].iter().map(text_of).collect::<Vec<_>>(),
            inputs[1].iter().map(text_of).collect::<Vec<_>>()
        ));
    }
    ck!(*handoff_calls.lock().unwrap() == vec!["handoff"]);
    ck!(model.calls().len() == 2);
    let stored = backend.stored();
    ck!(stored.len() == 2);
    ck!(text_of(&stored[0]).contains("compacted history"));
    ck!(text_of(&stored[1]).contains("done"));
    ck!(resumed
        .state
        .unwrap()
        .to_json()
        .unwrap()
        .get("pending_session_write")
        .is_none());
    Ok(())
}
