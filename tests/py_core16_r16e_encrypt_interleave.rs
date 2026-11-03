//! Ports of the interleaving cases of `tests/extensions/memory/test_encrypt_session.py`:
//! `test_encrypted_runner_skips_compaction_after_interleaved_outer_write` (before and after the
//! outer append), `test_encrypted_compaction_restores_original_tokens_before_concurrent_append`
//! (the failure case) and `test_clear_revokes_encrypted_deferred_compaction`.
//!
//! Python pauses inside subclasses of `EncryptedSession` and of the storage session; Rust wraps
//! the same boundaries in forwarding `Session` implementations. The compaction endpoint is a
//! local mock server.
#![cfg(all(feature = "sqlite", feature = "encrypt"))]
#![allow(clippy::type_complexity)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::memory::{
    CompactionMode, CompactionSnapshot, EncryptedSession, OpenAIResponsesCompactionArgs, Session,
    SessionRunCompletion, SessionSettings,
};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, InputItem, RunOptions, Runner};
use serde_json::json;
use tokio::sync::Notify;

mod py_core6_mcpmisc_support;
mod py_mcpmem_support;
use py_core6_mcpmisc_support::{Compaction, compaction_session, summary, text};
use py_mcpmem_support::ListSession;

const KEY: &str = "0123456789abcdef0123456789abcdef";

fn scripted(output: &'static str, response_id: &'static str) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(
            ScriptedStep::output(vec![assistant_message(output)])
                .with_response_id(Some(response_id.into())),
        )
        .unwrap();
    model
}

/// Forwards every `Session` hook to `inner`; the first append whose batch mentions `marker`
/// pauses before or after it reaches `inner`.
#[derive(Debug)]
struct OuterGate {
    inner: Arc<dyn Session>,
    marker: &'static str,
    pause_before: bool,
    armed: AtomicBool,
    paused: Notify,
    release: Notify,
}

impl OuterGate {
    fn new(inner: Arc<dyn Session>, marker: &'static str, pause_before: bool) -> Arc<OuterGate> {
        Arc::new(OuterGate {
            inner,
            marker,
            pause_before,
            armed: AtomicBool::new(true),
            paused: Notify::new(),
            release: Notify::new(),
        })
    }
}

#[async_trait::async_trait]
impl Session for OuterGate {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    fn manages_history_remotely(&self) -> bool {
        self.inner.manages_history_remotely()
    }
    fn ignores_item_ids_for_matching(&self) -> bool {
        self.inner.ignores_item_ids_for_matching()
    }
    fn wants_model_exchange(&self) -> bool {
        self.inner.wants_model_exchange()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let hit = text(&items).contains(self.marker) && self.armed.swap(false, Ordering::SeqCst);
        if hit && self.pause_before {
            self.paused.notify_one();
            self.release.notified().await;
        }
        self.inner.add_items(items).await?;
        if hit && !self.pause_before {
            self.paused.notify_one();
            self.release.notified().await;
        }
        Ok(())
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
    async fn on_run_completed(&self, info: &SessionRunCompletion) -> Result<Usage> {
        self.inner.on_run_completed(info).await
    }
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        self.inner.compaction_snapshot(limit).await
    }
}

async fn run_one(agent: &Agent<()>, input: &str, session: &Arc<dyn Session>) {
    Runner::run_with_options(agent, input, RunOptions::new(()).session(session.clone()))
        .await
        .unwrap();
}

async fn interleaved_outer_write(pause_before_append: bool) {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![]).await;
    let store = Arc::new(ListSession::new("encrypted-concurrent"));
    let compaction = compaction_session(
        "encrypted-concurrent",
        store.clone(),
        &endpoint,
        CompactionMode::PreviousResponseId,
    )
    .with_should_trigger_compaction(|ctx| ctx.response_id.as_deref() == Some("resp-a"));
    let encrypted: Arc<dyn Session> = Arc::new(
        EncryptedSession::new("encrypted-concurrent", Arc::new(compaction), KEY, 600).unwrap(),
    );
    let gate = OuterGate::new(encrypted.clone(), "answer-A", pause_before_append);
    let session: Arc<dyn Session> = gate.clone();

    let agent_a = Agent::<()>::new("A").with_model_instance(scripted("answer-A", "resp-a"));
    let agent_b = Agent::<()>::new("B").with_model_instance(scripted("answer-B", "resp-b"));
    let run_a = run_one(&agent_a, "input-A", &session);
    let interleave = async {
        gate.paused.notified().await;
        run_one(&agent_b, "input-B", &session).await;
        gate.release.notify_one();
    };
    tokio::join!(run_a, interleave);

    assert!(
        endpoint.requests().is_empty(),
        "pause_before={pause_before_append}: {:?}",
        endpoint.requests()
    );
    let stored = encrypted.get_items(None).await.unwrap();
    let texts: Vec<String> = stored
        .iter()
        .map(|i| text(std::slice::from_ref(i)))
        .collect();
    let expected: Vec<&str> = if pause_before_append {
        vec!["input-A", "input-B", "answer-B", "answer-A"]
    } else {
        vec!["input-A", "answer-A", "input-B", "answer-B"]
    };
    assert_eq!(texts.len(), expected.len(), "{texts:?}");
    for (actual, wanted) in texts.iter().zip(&expected) {
        assert!(
            actual.contains(wanted),
            "pause_before={pause_before_append}: {texts:?}"
        );
    }
}

#[tokio::test]
async fn an_encrypted_runner_skips_compaction_after_an_outer_write_interleaves_before_the_append() {
    interleaved_outer_write(true).await;
}

#[tokio::test]
async fn an_encrypted_runner_skips_compaction_after_an_outer_write_interleaves_after_the_append() {
    interleaved_outer_write(false).await;
}

/// Parks the compaction's write of its replacement summary and then fails it before anything is
/// committed. The restore path then puts the originals back while a concurrent append lands
/// (Python commits the replacement and raises; the Rust restore runs after a failed write).
#[derive(Debug)]
struct FailingReplacement {
    inner: Arc<ListSession>,
    armed: AtomicBool,
    reached: Notify,
    release: Notify,
}

#[async_trait::async_trait]
impl Session for FailingReplacement {
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
        // The first write after arming is the compaction's replacement: the concurrent append is
        // only started once this one is parked.
        if self.armed.swap(false, Ordering::SeqCst) {
            self.reached.notify_one();
            self.release.notified().await;
            return Err(AgentsError::other("replacement acknowledgement lost"));
        }
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        self.inner.compaction_snapshot(limit).await
    }
}

/// A committed replacement whose acknowledgement fails is rolled back to the original tokens
/// before a concurrent append lands, so the survivor sits after the restored history.
#[tokio::test]
async fn a_failed_encrypted_compaction_restores_the_original_tokens_before_a_concurrent_append() {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![summary()]).await;
    let store = Arc::new(ListSession::new("encrypted-rollback"));
    let backend = Arc::new(FailingReplacement {
        inner: store.clone(),
        armed: AtomicBool::new(false),
        reached: Notify::new(),
        release: Notify::new(),
    });
    let compaction = Arc::new(compaction_session(
        "encrypted-rollback",
        backend.clone(),
        &endpoint,
        CompactionMode::Input,
    ));
    let session: Arc<dyn Session> =
        Arc::new(EncryptedSession::new("encrypted-rollback", compaction.clone(), KEY, 2).unwrap());

    session
        .add_items(vec![InputItem::user("expired history")])
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(3100)).await;
    let valid = InputItem::user("valid history");
    session.add_items(vec![valid.clone()]).await.unwrap();
    let original_tokens = store.items();
    backend.armed.store(true, Ordering::SeqCst);

    let compacting = {
        let compaction = compaction.clone();
        tokio::spawn(async move {
            compaction
                .run_compaction(OpenAIResponsesCompactionArgs {
                    force: true,
                    ..Default::default()
                })
                .await
        })
    };
    tokio::time::timeout(Duration::from_secs(2), backend.reached.notified())
        .await
        .expect("the replacement is about to run");
    let writer = {
        let session = session.clone();
        tokio::spawn(async move {
            session
                .add_items(vec![InputItem::user("concurrent survivor")])
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    backend.release.notify_one();
    let outcome = compacting.await.unwrap();
    assert!(
        outcome
            .expect_err("the acknowledgement was lost")
            .to_string()
            .contains("replacement acknowledgement lost")
    );
    writer.await.unwrap().unwrap();

    // Python compares the restored ciphertext tokens byte for byte. Rust re-encrypts the restored
    // history with fresh nonces, so the observable contract is the decrypted history: the
    // unexpired original and the concurrent append, in order, with the expired item gone.
    assert_eq!(original_tokens.len(), 2, "expired and valid tokens");
    assert_eq!(store.items().len(), 2);
    let decrypted = session.get_items(None).await.unwrap();
    let texts: Vec<String> = decrypted
        .iter()
        .map(|i| text(std::slice::from_ref(i)))
        .collect();
    assert_eq!(texts.len(), 2, "{texts:?}");
    assert!(texts[0].contains("valid history"), "{texts:?}");
    assert!(texts[1].contains("concurrent survivor"), "{texts:?}");
}

/// A clear that arrives while a deferred compaction decision is paused revokes that decision.
#[tokio::test]
async fn a_clear_revokes_an_encrypted_deferred_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    let endpoint = Compaction::replying(vec![]).await;
    let store = Arc::new(ListSession::new("encrypted-deferred-clear"));

    // Pauses the first deferred-compaction snapshot that already contains the tool output.
    #[derive(Debug)]
    struct PausedSnapshot {
        inner: Arc<dyn Session>,
        armed: AtomicBool,
        read: Notify,
        release: Notify,
    }
    #[async_trait::async_trait]
    impl Session for PausedSnapshot {
        fn session_id(&self) -> &str {
            self.inner.session_id()
        }
        fn session_settings(&self) -> Option<SessionSettings> {
            self.inner.session_settings()
        }
        fn wants_model_exchange(&self) -> bool {
            self.inner.wants_model_exchange()
        }
        async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
            let items = self.inner.get_items(limit).await?;
            // This store holds ciphertext, so the read is recognised by its size: the user input,
            // the tool call and its output are stored once the tool output has settled.
            let has_output = items.len() >= 3;
            if has_output && self.armed.swap(false, Ordering::SeqCst) {
                self.read.notify_one();
                self.release.notified().await;
            }
            Ok(items)
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
        async fn on_run_completed(&self, info: &SessionRunCompletion) -> Result<Usage> {
            self.inner.on_run_completed(info).await
        }
        async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
            self.inner.compaction_snapshot(limit).await
        }
    }
    let paused = Arc::new(PausedSnapshot {
        inner: store.clone(),
        armed: AtomicBool::new(true),
        read: Notify::new(),
        release: Notify::new(),
    });
    let compaction = compaction_session(
        "encrypted-deferred-clear",
        paused.clone(),
        &endpoint,
        CompactionMode::Input,
    )
    .with_should_trigger_compaction(|ctx| ctx.response_id.as_deref() == Some("resp-tool"));
    let encrypted: Arc<dyn Session> = Arc::new(
        EncryptedSession::new("encrypted-deferred-clear", Arc::new(compaction), KEY, 600).unwrap(),
    );
    let session: Arc<dyn Session> = encrypted.clone();

    let lookup = FunctionTool::<()>::new_raw(
        "lookup",
        "look something up",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async move { Ok("private tool output".to_string()) },
    );
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(
            ScriptedStep::output(vec![function_call_with_id("lookup", "{}", "lookup-a")])
                .with_response_id(Some("resp-tool".into())),
        )
        .unwrap();
    model
        .enqueue(
            ScriptedStep::output(vec![assistant_message("answer-A")])
                .with_response_id(Some("resp-a".into())),
        )
        .unwrap();
    let agent_a = Agent::<()>::new("A")
        .with_model_instance(model)
        .with_tool(lookup);
    let agent_b = Agent::<()>::new("B").with_model_instance(scripted("answer-B", "resp-b"));

    // Run A parks in its deferred-compaction snapshot, which runs after the tool output settles
    // and before its second model call. Clearing while it is parked must revoke that compaction.
    let run_a = run_one(&agent_a, "input-A", &session);
    let driver = async {
        paused.read.notified().await;
        let clearing = {
            let session = session.clone();
            tokio::spawn(async move { session.clear_session().await.unwrap() })
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        paused.release.notify_one();
        clearing.await.unwrap();
        run_one(&agent_b, "input-B", &session).await;
    };
    tokio::join!(run_a, driver);

    assert!(
        endpoint.requests().is_empty(),
        "a cleared history is never compacted: {:?}",
        endpoint.requests()
    );
    let stored = encrypted.get_items(None).await.unwrap();
    let texts: Vec<String> = stored
        .iter()
        .map(|i| text(std::slice::from_ref(i)))
        .collect();
    assert!(texts.iter().any(|t| t.contains("input-B")), "{texts:?}");
    assert!(texts.iter().any(|t| t.contains("answer-B")), "{texts:?}");
    assert!(
        !texts.iter().any(|t| t.contains("private tool output")),
        "{texts:?}"
    );
}
