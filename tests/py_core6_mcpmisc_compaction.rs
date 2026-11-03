//! Ports of `json_compaction_retry_does_not_promote_filtered_session_history` and
//! `compaction_reload_preserves_session_retrieval_window` (2 cases) from
//! `tests/test_run_impl_resume_paths.py`: a compaction owed by a resumed append is authorized by the
//! model exchange the failed attempt recorded on the checkpoint, and a compaction reads the
//! window the store's retrieval settings expose.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs,
    clippy::collapsible_if
)]
#![cfg(feature = "sqlite")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, Session, SessionSettings, SqliteSession,
};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, RunState, Runner};
use serde_json::{Value, json};

mod py_core6_mcpmisc_support;
mod py_core6_resume_support;
mod py_mcpmem_support;
use py_core6_mcpmisc_support::{Compaction, compaction_session, values};
use py_core6_resume_support::FailingSession;

#[tokio::test]
async fn json_compaction_retry_does_not_promote_filtered_session_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    let backend = FailingSession::new("filtered-retry");
    let hidden = json!({"role": "user", "content": "synthetic omitted history"});
    backend.push(hidden.clone());
    backend.fail_on_output("delegating");
    let endpoint = Compaction::replying(vec![]).await;
    let session: Arc<dyn Session> = Arc::new(
        compaction_session(
            "filtered-retry",
            backend.clone(),
            &endpoint,
            CompactionMode::Input,
        )
        .with_should_trigger_compaction(|ctx| ctx.response_id.as_deref() == Some("resp-handoff")),
    );
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(
            ScriptedStep::output(vec![
                assistant_message("delegating"),
                function_call_with_id("transfer_to_delegate", "{}", "handoff-1"),
            ])
            .with_response_id(Some("resp-handoff".into())),
        )
        .unwrap();
    model
        .enqueue(
            ScriptedStep::output(vec![assistant_message("done")])
                .with_response_id(Some("resp-delegate".into())),
        )
        .unwrap();
    let delegate = Agent::<()>::new("delegate").with_model_instance(model.clone());
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_handoff(handoff(delegate).with_input_filter(|mut data| {
            data.new_items
                .retain(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }));
            data
        }));
    let config = RunConfig::<()>::default().with_session_input_callback(
        |_history: Vec<InputItem>, new: Vec<InputItem>| async move { Ok(new) },
    );
    let mut failed = Runner::run_streamed_with_options(
        &triage,
        "hello",
        RunOptions::new(())
            .session(session.clone())
            .run_config(config.clone()),
    );
    let mut error = None;
    {
        let events = failed.stream_events();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
                break;
            }
        }
    }
    assert!(
        error
            .expect("the append fails")
            .to_string()
            .contains("session append failed")
    );
    let payload = failed.to_state().expect("checkpoint").to_json().unwrap();
    assert!(payload["pending_session_write"]["compaction_model_exchange"].is_object());
    let state = RunState::from_json(payload).unwrap();
    let resumed = Runner::run_with_options(
        &triage,
        state,
        RunOptions::new(())
            .session(session.clone())
            .run_config(config),
    )
    .await
    .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert!(
        calls
            .iter()
            .all(|c| !values(&c.request.input).contains(&hidden))
    );
    assert!(endpoint.requests().is_empty());
    assert_eq!(values(&session.get_items(None).await.unwrap())[0], hidden);
}

/// Python's `_LostAckSQLiteSession`: the next append commits and then reports a failure.
#[derive(Debug)]
struct LostAck {
    inner: Arc<SqliteSession>,
    fail_after_commit: AtomicBool,
}

#[async_trait::async_trait]
impl Session for LostAck {
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
        self.inner.add_items(items).await?;
        if self.fail_after_commit.swap(false, Ordering::SeqCst) {
            return Err(AgentsError::other("commit acknowledgement failed"));
        }
        Ok(())
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
}

#[tokio::test]
async fn compaction_reload_preserves_session_retrieval_window() {
    for mode in [CompactionMode::Input, CompactionMode::Auto] {
        let dir = tempfile::tempdir().unwrap();
        let sqlite = Arc::new(
            SqliteSession::open("bounded-compaction", dir.path().join("history.db"))
                .unwrap()
                .with_settings(SessionSettings::with_limit(1)),
        );
        let backend = Arc::new(LostAck {
            inner: sqlite,
            fail_after_commit: AtomicBool::new(false),
        });
        // The mock echoes the request input back as the compacted output.
        let inputs = Arc::new(Mutex::new(Vec::<Value>::new()));
        let log = inputs.clone();
        let endpoint = Compaction::with(move |body| {
            log.lock().unwrap().push(body["input"].clone());
            json!({"id": "cmp", "output": body["input"].clone(), "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})
        })
        .await;
        let session = compaction_session("bounded-compaction", backend.clone(), &endpoint, mode);
        let old_items: Vec<InputItem> = (0..12)
            .map(|i| {
                InputItem::from_value(
                    json!({"role": "assistant", "content": format!("old message {i}")}),
                )
            })
            .collect();
        backend.add_items(old_items.clone()).await.unwrap();
        // The configured window has one candidate, so the default threshold is not met.
        session
            .run_compaction(OpenAIResponsesCompactionArgs {
                response_id: Some("unstored-response".into()),
                store: Some(false),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(inputs.lock().unwrap().is_empty(), "{mode:?}");
        assert_eq!(
            values(&backend.get_items(Some(100)).await.unwrap()),
            values(&old_items)
        );

        let recovered =
            InputItem::from_value(json!({"role": "assistant", "content": "committed reply"}));
        backend.fail_after_commit.store(true, Ordering::SeqCst);
        assert!(session.add_items(vec![recovered.clone()]).await.is_err());
        let mut all = old_items.clone();
        all.push(recovered.clone());
        assert_eq!(
            values(&backend.get_items(Some(100)).await.unwrap()),
            values(&all)
        );

        session
            .run_compaction(OpenAIResponsesCompactionArgs {
                force: true,
                store: Some(false),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            *inputs.lock().unwrap(),
            vec![json!(values(&[recovered.clone()]))],
            "{mode:?}"
        );
        assert_eq!(
            values(&backend.get_items(Some(100)).await.unwrap()),
            values(&[recovered]),
            "{mode:?}"
        );
    }
}
