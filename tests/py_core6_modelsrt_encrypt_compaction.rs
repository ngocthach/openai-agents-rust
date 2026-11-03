//! Ports of the encryption-outermost compaction tests in
//! `tests/extensions/memory/test_encrypt_session.py`:
//! `test_runner_decrypts_existing_compaction_history_with_ttl_and_limit`,
//! `test_runner_defers_compaction_using_decrypted_candidates`,
//! `test_encrypted_compaction_replaces_complete_logical_history_after_expiry` and
//! `test_runner_recovers_encrypted_resumed_append`.
//!
//! Python freezes `cryptography.fernet.time.time`; Rust reads the real clock, so the expired
//! envelopes are written with a 2 second TTL and the test sleeps past it. The compaction
//! endpoint is a local mock server, so no network or real API key is used.
#![cfg(all(feature = "sqlite", feature = "encrypt"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::memory::{
    CompactionContext, CompactionMode, EncryptedSession, OpenAIResponsesCompactionArgs,
    OpenAIResponsesCompactionSession, Session, SessionSettings, SqliteSession,
};
use openai_agents_sdk::testing::{
    FakeModel, ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{item_text, mock_openai};

const KEY: &str = "0123456789abcdef0123456789abcdef";
const TTL: u64 = 2;
const PAST_TTL: Duration = Duration::from_millis(3100);

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn compact_reply(output: Vec<Value>) -> Value {
    json!({
        "id": "cmp_1",
        "output": output,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    })
}

async fn run(
    agent: &Agent<()>,
    input: &str,
    session: Arc<dyn Session>,
    streamed: bool,
) -> openai_agents_sdk::Result<()> {
    openai_agents_sdk::set_tracing_disabled(true);
    let options = RunOptions::new(()).session(session);
    if !streamed {
        return Runner::run_with_options(agent, input, options)
            .await
            .map(|_| ());
    }
    use futures::StreamExt;
    let mut result = Runner::run_streamed_with_options(agent, input, options);
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event?;
        }
    }
    result.finish().await
}

#[tokio::test]
async fn the_runner_decrypts_existing_compaction_history_with_ttl_and_limit() {
    for streamed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        // The limit of one forces an expired tail to widen the encrypted read window.
        let backend = Arc::new(
            SqliteSession::open("encrypted-history", dir.path().join("history.db"))
                .unwrap()
                .with_settings(SessionSettings::with_limit(1)),
        );
        let compaction =
            OpenAIResponsesCompactionSession::new("encrypted-history", backend.clone())
                .unwrap()
                .with_client(OpenAIClient::new("sk-test").with_base_url("http://127.0.0.1:1"))
                .with_should_trigger_compaction(|_| false);
        let session: Arc<dyn Session> = Arc::new(
            EncryptedSession::new("encrypted-history", Arc::new(compaction), KEY, TTL).unwrap(),
        );

        session
            .add_items(vec![InputItem::from_value(
                json!({"role": "assistant", "content": "expired history"}),
            )])
            .await
            .unwrap();
        let expired = backend.get_items(None).await.unwrap()[0].clone();
        tokio::time::sleep(PAST_TTL).await;
        session
            .add_items(vec![user("retained history")])
            .await
            .unwrap();
        // An expired tail forces the encrypted session to expand its one-item retrieval window.
        backend.add_items(vec![expired]).await.unwrap();

        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("answer")]);
        let agent = Agent::<()>::new("test").with_model_instance(model.clone());
        run(&agent, "next", session.clone(), streamed)
            .await
            .unwrap();

        assert_eq!(
            texts(&model.last_call().unwrap().input),
            vec!["retained history", "next"],
            "streamed={streamed}"
        );
    }
}

#[tokio::test]
async fn the_runner_defers_compaction_using_decrypted_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let backend =
        Arc::new(SqliteSession::open("encrypted-deferred", dir.path().join("history.db")).unwrap());
    let (base, captured) = mock_openai(vec![compact_reply(vec![])]).await;
    let decisions: Arc<Mutex<Vec<CompactionContext>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = decisions.clone();
    let compaction = OpenAIResponsesCompactionSession::new("encrypted-deferred", backend.clone())
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_compaction_mode(CompactionMode::Input)
        .with_should_trigger_compaction(move |context| {
            recorded.lock().unwrap().push(context.clone());
            context.response_id.as_deref() == Some("resp-tool")
                && context
                    .compaction_candidate_items
                    .iter()
                    .any(|item| item.item_type() == Some("function_call_output"))
        });
    let session: Arc<dyn Session> = Arc::new(
        EncryptedSession::new("encrypted-deferred", Arc::new(compaction), KEY, 600).unwrap(),
    );

    let lookup = FunctionTool::<()>::new_raw(
        "lookup",
        "look something up",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async move { Ok("private lookup result".to_string()) },
    );
    let model = ScriptedModel::with_steps(vec![
        ScriptedStep {
            output: vec![function_call_with_id("lookup", "{}", "lookup-1")],
            response_id: Some("resp-tool".into()),
            ..Default::default()
        },
        ScriptedStep {
            output: vec![assistant_message("private answer")],
            response_id: Some("resp-final".into()),
            ..Default::default()
        },
    ])
    .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(Arc::new(model))
        .with_tool(lookup);

    run(&agent, "private request", session.clone(), false)
        .await
        .unwrap();

    // The policy saw the decrypted tool call and its output as candidates.
    let saw_candidates = decisions.lock().unwrap().iter().any(|context| {
        let types: Vec<Option<&str>> = context
            .compaction_candidate_items
            .iter()
            .map(|item| item.item_type())
            .collect();
        context.response_id.as_deref() == Some("resp-tool")
            && types == vec![Some("function_call"), Some("function_call_output")]
    });
    assert!(saw_candidates, "{:?}", decisions.lock().unwrap());

    // The compaction request carried plaintext history, never an encrypted envelope.
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    let sent = requests[0].body["input"].as_array().unwrap().clone();
    let types: Vec<&str> = sent
        .iter()
        .map(|item| item["type"].as_str().unwrap_or("message"))
        .collect();
    assert_eq!(
        types,
        vec![
            "message",
            "function_call",
            "function_call_output",
            "message"
        ]
    );
    assert_eq!(sent[0]["content"], json!("private request"));
    assert_eq!(sent[2]["call_id"], json!("lookup-1"));
    assert_eq!(sent[2]["output"], json!("private lookup result"));
    assert!(!requests[0].body.to_string().contains("__enc__"));
    // Compaction replaced the whole history; the empty replacement left nothing stored.
    assert!(backend.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn encrypted_compaction_replaces_complete_logical_history_after_expiry() {
    let mut cases = Vec::new();
    for streamed in [false, true] {
        for mode in [CompactionMode::Input, CompactionMode::PreviousResponseId] {
            cases.push(replaces_logical_history_case(streamed, mode));
        }
    }
    futures::future::join_all(cases).await;
}

async fn replaces_logical_history_case(streamed: bool, mode: CompactionMode) {
    let case = format!("streamed={streamed} mode={mode:?}");
    let dir = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        SqliteSession::open("encrypted-active-compaction", dir.path().join("history.db")).unwrap(),
    );
    let compacted = json!({"type": "compaction", "id": "cmp-1", "encrypted_content": "summary"});
    let (base, captured) = mock_openai(vec![
        compact_reply(vec![compacted.clone()]),
        compact_reply(vec![compacted.clone()]),
        compact_reply(vec![compacted.clone()]),
    ])
    .await;
    let decisions: Arc<Mutex<Vec<CompactionContext>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = decisions.clone();
    let compaction = Arc::new(
        OpenAIResponsesCompactionSession::new("encrypted-active-compaction", backend.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(base))
            .with_compaction_mode(mode)
            .with_should_trigger_compaction(move |context| {
                recorded.lock().unwrap().push(context.clone());
                true
            }),
    );
    let session: Arc<dyn Session> = Arc::new(
        EncryptedSession::new("encrypted-active-compaction", compaction.clone(), KEY, TTL).unwrap(),
    );

    let output = assistant_message("private answer");
    let next_output = assistant_message("next answer");
    let model = ScriptedModel::with_steps(vec![
        ScriptedStep {
            output: vec![output],
            response_id: Some("resp-1".into()),
            ..Default::default()
        },
        ScriptedStep {
            output: vec![next_output],
            response_id: Some("resp-2".into()),
            ..Default::default()
        },
    ])
    .unwrap();
    let model = Arc::new(model);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    session
        .add_items(vec![InputItem::from_value(
            json!({"role": "assistant", "content": "expired private history"}),
        )])
        .await
        .unwrap();
    tokio::time::sleep(PAST_TTL).await;
    session
        .add_items(vec![user("retained history")])
        .await
        .unwrap();
    run(&agent, "private input", session.clone(), streamed)
        .await
        .unwrap();

    let first = decisions.lock().unwrap()[0].clone();
    assert_eq!(
        texts(&first.session_items[..2]),
        vec!["retained history", "private input"],
        "{case}"
    );
    assert_eq!(first.session_items.len(), 3, "{case}");
    assert_eq!(first.compaction_candidate_items.len(), 1, "{case}");
    assert_eq!(
        first.compaction_candidate_items[0], first.session_items[2],
        "{case}"
    );

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "{case}");
    assert_eq!(requests[0].body["model"], json!("gpt-4.1"), "{case}");
    match mode {
        CompactionMode::Input => {
            let sent = requests[0].body["input"].as_array().unwrap().clone();
            assert_eq!(sent.len(), 3, "{case}: {sent:?}");
            assert_eq!(sent[0]["content"], json!("retained history"), "{case}");
            assert_eq!(sent[1]["content"], json!("private input"), "{case}");
        }
        _ => assert_eq!(
            requests[0].body["previous_response_id"],
            json!("resp-1"),
            "{case}"
        ),
    }
    // The compaction output replaced the complete logical history, stored encrypted.
    assert_eq!(backend.get_items(None).await.unwrap().len(), 1, "{case}");
    let stored = session.get_items(None).await.unwrap();
    assert_eq!(stored.len(), 1, "{case}");
    assert_eq!(stored[0].item_type(), Some("compaction"), "{case}");

    run(&agent, "next", session.clone(), streamed)
        .await
        .unwrap();
    let call = model.calls().last().unwrap().request.input.clone();
    assert_eq!(call.len(), 2, "{case}");
    assert_eq!(call[0].item_type(), Some("compaction"), "{case}");
    assert_eq!(item_text(&call[1]), "next", "{case}");
    let second = decisions.lock().unwrap()[1].clone();
    assert_eq!(second.session_items.len(), 3, "{case}");
    assert_eq!(
        second.session_items[0].item_type(),
        Some("compaction"),
        "{case}"
    );
    assert_eq!(second.compaction_candidate_items.len(), 1, "{case}");
    assert_eq!(
        second.compaction_candidate_items[0], second.session_items[2],
        "{case}"
    );
    assert_eq!(backend.get_items(None).await.unwrap().len(), 1, "{case}");

    // Expiry changes the logical snapshot without changing the mutation generation.
    tokio::time::sleep(PAST_TTL).await;
    compaction
        .run_compaction(OpenAIResponsesCompactionArgs {
            compaction_mode: Some(CompactionMode::Input),
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.last().unwrap().body["input"], json!([]), "{case}");
}

// ---------------------------------------------------------------------------------------------
// test_runner_recovers_encrypted_resumed_append
// ---------------------------------------------------------------------------------------------

/// Commits the batch, then reports a failure once: the append happened but its acknowledgement
/// was lost, as in Python's `LostAckSQLiteSession`.
#[derive(Debug)]
struct LostAck {
    inner: Arc<SqliteSession>,
    fail_after_commit: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl Session for LostAck {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    async fn get_items(&self, limit: Option<usize>) -> openai_agents_sdk::Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::Result<()> {
        self.inner.add_items(items).await?;
        if self
            .fail_after_commit
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(openai_agents_sdk::AgentsError::other(
                "append acknowledgement lost",
            ));
        }
        Ok(())
    }
    async fn pop_item(&self) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn pop_item_validated(
        &self,
        validate: openai_agents_sdk::memory::PopValidator,
    ) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.inner.pop_item_validated(validate).await
    }
    async fn clear_session(&self) -> openai_agents_sdk::Result<()> {
        self.inner.clear_session().await
    }
}

#[tokio::test]
async fn the_runner_recovers_an_encrypted_resumed_append() {
    use openai_agents_sdk::RunState;
    use openai_agents_sdk::testing::function_call_with_id;
    use openai_agents_sdk::tool::FunctionTool;
    use std::sync::atomic::{AtomicUsize, Ordering};

    for streamed in [false, true] {
        let case = format!("streamed={streamed}");
        let dir = tempfile::tempdir().unwrap();
        let sqlite = Arc::new(
            SqliteSession::open("encrypted-resume", dir.path().join("history.db")).unwrap(),
        );
        let backend = Arc::new(LostAck {
            inner: sqlite.clone(),
            fail_after_commit: std::sync::atomic::AtomicBool::new(false),
        });
        let compaction = OpenAIResponsesCompactionSession::new("encrypted-resume", backend.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url("http://127.0.0.1:1"))
            .with_should_trigger_compaction(|_| false);
        let session: Arc<dyn Session> = Arc::new(
            EncryptedSession::new("encrypted-resume", Arc::new(compaction), KEY, 600).unwrap(),
        );

        let effects = Arc::new(AtomicUsize::new(0));
        let counter = effects.clone();
        let record = FunctionTool::<()>::new_raw(
            "record",
            "records",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok("private tool receipt".to_string())
                }
            },
        )
        .with_needs_approval_fn(|_ctx, _args, _id| async move { true });
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call_with_id("record", "{}", "record-1")]);
        model.add_turn(vec![assistant_message("private final answer")]);
        let agent = Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(record);

        openai_agents_sdk::set_tracing_disabled(true);
        let paused = Runner::run_with_options(
            &agent,
            "private request",
            RunOptions::new(()).session(session.clone()),
        )
        .await
        .unwrap();
        let mut state = paused.to_state();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        backend
            .fail_after_commit
            .store(true, std::sync::atomic::Ordering::SeqCst);

        let slot = openai_agents_sdk::run::RunCheckpoint::new();
        let failed = if streamed {
            let mut result = Runner::run_streamed_with_options(
                &agent,
                state,
                RunOptions::new(())
                    .session(session.clone())
                    .checkpoint(&slot),
            );
            use futures::StreamExt;
            let mut stream_error = None;
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(event) = events.next().await {
                    if let Err(error) = event {
                        stream_error = Some(error);
                        break;
                    }
                }
            }
            match stream_error {
                Some(error) => Err(error),
                None => result.finish().await.map(|_| ()),
            }
        } else {
            Runner::run_with_options(
                &agent,
                state,
                RunOptions::new(())
                    .session(session.clone())
                    .checkpoint(&slot),
            )
            .await
            .map(|_| ())
        };
        let error = failed
            .expect_err(&format!("{case}: the append must fail"))
            .to_string();
        assert!(
            error.contains("append acknowledgement lost"),
            "{case}: {error}"
        );
        assert_eq!(effects.load(Ordering::SeqCst), 1, "{case}");
        assert_eq!(model.calls().len(), 1, "{case}");

        // The failed run left a checkpoint holding the append whose acknowledgement was lost.
        let checkpoint = slot.take().expect("the failed run leaves a checkpoint");
        let state = RunState::from_json(checkpoint.to_json().unwrap()).unwrap();
        // The retry runs in the same mode as the failed run.
        let (final_text, state_json) = if streamed {
            use futures::StreamExt;
            let mut result = Runner::run_streamed_with_options(
                &agent,
                state,
                RunOptions::new(()).session(session.clone()),
            );
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(event) = events.next().await {
                    event.unwrap();
                }
            }
            (
                result.final_output_text(),
                result.to_state().unwrap().to_json().unwrap(),
            )
        } else {
            let result = Runner::run_with_options(
                &agent,
                state,
                RunOptions::new(()).session(session.clone()),
            )
            .await
            .unwrap();
            (
                result.final_output_text(),
                result.to_state().to_json().unwrap(),
            )
        };

        assert_eq!(
            final_text.as_deref(),
            Some("private final answer"),
            "{case}"
        );
        assert_eq!(effects.load(Ordering::SeqCst), 1, "{case}");
        assert!(
            !state_json.to_string().contains("pending_session_write"),
            "{case}"
        );
        let stored = session.get_items(None).await.unwrap();
        let types: Vec<Option<&str>> = stored.iter().map(|i| i.item_type()).collect();
        assert_eq!(
            types,
            vec![
                None,
                Some("function_call"),
                Some("function_call_output"),
                Some("message")
            ],
            "{case}: {stored:?}"
        );
        assert_eq!(item_text(&stored[0]), "private request", "{case}");
        assert_eq!(
            stored[2].as_value()["output"],
            json!("private tool receipt"),
            "{case}"
        );
    }
}
