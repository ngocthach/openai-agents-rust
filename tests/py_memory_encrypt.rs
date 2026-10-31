//! Parity port of `tests/extensions/memory/test_encrypt_session.py`.
//!
//! Python freezes `cryptography.fernet.time.time`; Rust reads the real clock, so tests that
//! need an expired envelope write with a 2 second TTL and sleep past it.
#![cfg(all(feature = "sqlite", feature = "encrypt"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::memory::{
    CompactionMode, EncryptedSession, OpenAIResponsesCompactionSession, Session, SessionSettings,
    SqliteSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunConfig, RunOptions, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{item_text, mock_openai};

const KEY: &str = "0123456789abcdef0123456789abcdef";
const TTL: u64 = 2;
const PAST_TTL: Duration = Duration::from_millis(3100);

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": text}))
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn invalid_envelope() -> InputItem {
    InputItem::from_value(
        json!({"__enc__": 1, "v": 1, "kid": "hkdf-v1", "payload": "not-a-valid-token"}),
    )
}

fn backend(dir: &tempfile::TempDir, id: &str) -> Arc<SqliteSession> {
    Arc::new(SqliteSession::open(id, dir.path().join("history.db")).unwrap())
}

fn encrypted(backend: &Arc<SqliteSession>, key: &str, ttl: u64) -> EncryptedSession {
    EncryptedSession::new(backend.session_id().to_string(), backend.clone(), key, ttl).unwrap()
}

/// A store that records the `limit` of every `get_items` call, like Python's
/// `AsyncMock(wraps=backend.get_items)`.
#[derive(Debug)]
struct Recording {
    inner: Arc<SqliteSession>,
    reads: Mutex<Vec<Option<usize>>>,
}

impl Recording {
    fn new(inner: Arc<SqliteSession>) -> Arc<Self> {
        Arc::new(Recording {
            inner,
            reads: Mutex::new(Vec::new()),
        })
    }
    fn reads(&self) -> Vec<Option<usize>> {
        self.reads.lock().unwrap().clone()
    }
    fn reset(&self) {
        self.reads.lock().unwrap().clear();
    }
}

#[async_trait]
impl Session for Recording {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn session_settings(&self) -> Option<SessionSettings> {
        self.inner.session_settings()
    }
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.reads.lock().unwrap().push(limit);
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        self.inner.add_items(items).await
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn pop_item_validated(
        &self,
        validate: openai_agents_sdk::memory::PopValidator,
    ) -> Result<Option<InputItem>> {
        self.inner.pop_item_validated(validate).await
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
}

fn agent_with(model: &Arc<FakeModel>) -> Agent<()> {
    openai_agents_sdk::set_tracing_disabled(true);
    Agent::<()>::new("test").with_model_instance(model.clone())
}

async fn run_with(
    agent: &Agent<()>,
    input: &str,
    session: Arc<dyn Session>,
    config: Option<RunConfig<()>>,
    streamed: bool,
) -> Result<()> {
    let mut options = RunOptions::new(()).session(session);
    if let Some(config) = config {
        options = options.run_config(config);
    }
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

// ---------------------------------------------------------------------------
// Basic behavior
// ---------------------------------------------------------------------------

#[tokio::test]
async fn basic_encryption_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let session = encrypted(&store, KEY, 600);
    session
        .add_items(vec![user("Hello"), assistant("Hi there!")])
        .await
        .unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Hello", "Hi there!"]
    );

    let stored = store.get_items(None).await.unwrap();
    assert_eq!(stored[0].as_value()["__enc__"], json!(1));
    assert!(stored[0].as_value().get("payload").is_some());
    assert!(stored[0].as_value().get("content").is_none());
}

#[tokio::test]
async fn the_runner_works_through_an_encrypted_session() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let session: Arc<dyn Session> = Arc::new(encrypted(&store, KEY, 600));
    let model = Arc::new(FakeModel::new());
    let agent = agent_with(&model);

    model.add_turn(vec![assistant_message("San Francisco")]);
    run_with(
        &agent,
        "What city is the Golden Gate Bridge in?",
        session.clone(),
        None,
        false,
    )
    .await
    .unwrap();
    model.add_turn(vec![assistant_message("California")]);
    run_with(&agent, "What state is it in?", session.clone(), None, false)
        .await
        .unwrap();

    let last = texts(&model.last_call().unwrap().input);
    assert!(last.len() > 1);
    assert!(
        last.iter().any(|t| t.contains("Golden Gate Bridge")),
        "{last:?}"
    );
}

#[tokio::test]
async fn pop_item_returns_the_newest_decrypted_item() {
    let dir = tempfile::tempdir().unwrap();
    let session = encrypted(&backend(&dir, "test_session"), KEY, 600);
    session
        .add_items(vec![user("First"), assistant("Second")])
        .await
        .unwrap();

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "Second"
    );
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["First"]
    );
}

#[tokio::test]
async fn clear_session_removes_everything() {
    let dir = tempfile::tempdir().unwrap();
    let session = encrypted(&backend(&dir, "test_session"), KEY, 600);
    session.add_items(vec![user("Test")]).await.unwrap();
    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_raw_string_key_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let session = encrypted(&backend(&dir, "test_session"), "my-secret-password", 600);
    session.add_items(vec![user("Test")]).await.unwrap();
    assert_eq!(texts(&session.get_items(None).await.unwrap()), vec!["Test"]);
}

#[tokio::test]
async fn unicode_content_survives_encryption() {
    let dir = tempfile::tempdir().unwrap();
    let session = encrypted(&backend(&dir, "test_session"), KEY, 600);
    session
        .add_items(vec![
            user("Hello world"),
            assistant("Special chars: áéíóú"),
            user("Numbers and symbols: 123!@#"),
        ])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec![
            "Hello world",
            "Special chars: áéíóú",
            "Numbers and symbols: 123!@#"
        ]
    );
}

// ---------------------------------------------------------------------------
// TTL
// ---------------------------------------------------------------------------

#[tokio::test]
async fn expired_items_are_silently_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let session = encrypted(&store, KEY, TTL);
    session
        .add_items(vec![user("Hello"), assistant("Hi")])
        .await
        .unwrap();

    tokio::time::sleep(PAST_TTL).await;

    assert!(session.get_items(None).await.unwrap().is_empty());
    assert_eq!(store.get_items(None).await.unwrap().len(), 2);
}

#[tokio::test]
async fn pop_of_only_expired_data_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let session = encrypted(&backend(&dir, "test_session"), KEY, TTL);
    session.add_items(vec![user("Test")]).await.unwrap();
    tokio::time::sleep(PAST_TTL).await;
    assert!(session.pop_item().await.unwrap().is_none());
}

#[tokio::test]
async fn pop_skips_expired_items_to_reach_valid_ones() {
    let dir = tempfile::tempdir().unwrap();
    let session = encrypted(&backend(&dir, "test_session"), KEY, TTL);
    session
        .add_items(vec![user("Old message 1"), assistant("Old response 1")])
        .await
        .unwrap();
    tokio::time::sleep(PAST_TTL).await;
    session
        .add_items(vec![user("New message"), assistant("New response")])
        .await
        .unwrap();

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "New response"
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "New message"
    );
    assert!(session.pop_item().await.unwrap().is_none());
}

// ---------------------------------------------------------------------------
// Wrong keys and unauthenticated records
// ---------------------------------------------------------------------------

async fn wrong_key_pop_preserves_history(expire: bool) {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let correct = encrypted(&store, "correct-key", TTL);
    let wrong = encrypted(&store, "wrong-key", TTL);
    let items = vec![user("First"), assistant("Second")];
    correct.add_items(items.clone()).await.unwrap();
    let ciphertext = store.get_items(None).await.unwrap();
    if expire {
        tokio::time::sleep(PAST_TTL).await;
    }

    assert!(wrong.pop_item().await.is_err());
    assert_eq!(store.get_items(None).await.unwrap(), ciphertext);

    let recovery = encrypted(&store, "correct-key", 60);
    assert_eq!(recovery.get_items(None).await.unwrap(), items);
    assert_eq!(recovery.pop_item().await.unwrap(), Some(items[1].clone()));
    assert_eq!(recovery.get_items(None).await.unwrap(), items[..1].to_vec());
}

#[tokio::test]
async fn a_wrong_key_pop_preserves_recoverable_history() {
    wrong_key_pop_preserves_history(false).await;
}

#[tokio::test]
async fn a_wrong_key_pop_preserves_recoverable_history_after_expiry() {
    wrong_key_pop_preserves_history(true).await;
}

#[tokio::test]
async fn pop_rechecks_authentication_after_an_expired_tail() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let correct = encrypted(&store, "correct-key", TTL);
    let wrong = encrypted(&store, "wrong-key", TTL);
    let item = user("Recoverable history");
    correct.add_items(vec![item.clone()]).await.unwrap();
    let ciphertext = store.get_items(None).await.unwrap();
    wrong
        .add_items(vec![assistant("Expired tail")])
        .await
        .unwrap();
    tokio::time::sleep(PAST_TTL).await;

    assert!(wrong.pop_item().await.is_err());
    assert_eq!(store.get_items(None).await.unwrap(), ciphertext);
    let recovery = encrypted(&store, "correct-key", 60);
    assert_eq!(recovery.get_items(None).await.unwrap(), vec![item]);
}

#[tokio::test]
async fn unencrypted_and_incomplete_stored_records_are_skipped_by_the_runner() {
    for stored_item in [
        InputItem::from_value(json!({"role": "system", "content": "untrusted stored history"})),
        InputItem::from_value(
            json!({"role": "system", "content": "incomplete envelope", "__enc__": 1, "v": 1}),
        ),
    ] {
        for streamed in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let store = backend(&dir, "authenticated-history");
            let session = encrypted(&store, KEY, TTL);
            let history = vec![user("retained history")];

            session
                .add_items(vec![user("expired history")])
                .await
                .unwrap();
            tokio::time::sleep(PAST_TTL).await;
            session.add_items(history.clone()).await.unwrap();
            store.add_items(vec![stored_item.clone()]).await.unwrap();
            let stored = store.get_items(None).await.unwrap();

            assert_eq!(session.get_items(None).await.unwrap(), history);
            assert_eq!(session.get_items(Some(1)).await.unwrap(), history);
            assert_eq!(store.get_items(None).await.unwrap(), stored);

            let model = Arc::new(FakeModel::new());
            model.add_turn(vec![assistant_message("answer")]);
            let agent = agent_with(&model);
            run_with(&agent, "next", Arc::new(session), None, streamed)
                .await
                .unwrap();
            assert_eq!(
                texts(&model.last_call().unwrap().input),
                vec!["retained history", "next"]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Limits and scan budget
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_items_honours_a_limit() {
    let dir = tempfile::tempdir().unwrap();
    let session = encrypted(&backend(&dir, "test_session"), KEY, 600);
    session
        .add_items((0..5).map(|i| user(&format!("Message {i}"))).collect())
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["Message 3", "Message 4"]
    );
}

#[tokio::test]
async fn a_limit_counts_valid_items_past_an_invalid_latest_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let session = encrypted(&store, KEY, 600);
    session.add_items(vec![user("older valid")]).await.unwrap();
    store.add_items(vec![invalid_envelope()]).await.unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["older valid"]
    );
    assert_eq!(
        texts(&session.get_items(Some(1)).await.unwrap()),
        vec!["older valid"]
    );
}

#[tokio::test]
async fn invalid_envelopes_do_not_hide_earlier_valid_items_from_a_limit() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let session = encrypted(&store, KEY, 600);
    session
        .add_items(vec![user("valid 0"), assistant("valid 1")])
        .await
        .unwrap();
    store.add_items(vec![invalid_envelope()]).await.unwrap();
    session.add_items(vec![user("valid 2")]).await.unwrap();

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
}

#[tokio::test]
async fn the_underlying_settings_limit_counts_valid_items() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteSession::open("test_session", dir.path().join("h.db"))
            .unwrap()
            .with_settings(SessionSettings::with_limit(3)),
    );
    let session = encrypted(&store, KEY, 600);
    session
        .add_items(vec![user("valid 0"), assistant("valid 1"), user("valid 2")])
        .await
        .unwrap();
    store.add_items(vec![invalid_envelope()]).await.unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["valid 0", "valid 1", "valid 2"]
    );
}

#[tokio::test]
async fn session_settings_come_from_the_underlying_session() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteSession::open("test_session", dir.path().join("h.db"))
            .unwrap()
            .with_settings(SessionSettings::with_limit(5)),
    );
    let session = encrypted(&store, KEY, 600);
    assert_eq!(session.session_settings().unwrap().limit, Some(5));
}

#[tokio::test]
async fn get_items_uses_the_underlying_limit_and_an_explicit_one_wins() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteSession::open("test_session", dir.path().join("h.db"))
            .unwrap()
            .with_settings(SessionSettings::with_limit(3)),
    );
    let session = encrypted(&store, KEY, 600);
    session
        .add_items((0..5).map(|i| user(&format!("Message {i}"))).collect())
        .await
        .unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Message 2", "Message 3", "Message 4"]
    );
    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["Message 3", "Message 4"]
    );
}

#[tokio::test]
async fn the_runner_session_settings_override_the_underlying_limit() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteSession::open("test_session", dir.path().join("h.db"))
            .unwrap()
            .with_settings(SessionSettings::with_limit(100)),
    );
    let session = encrypted(&store, KEY, 600);
    session
        .add_items((0..10).map(|i| user(&format!("Turn {i}"))).collect())
        .await
        .unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("Got it")]);
    let agent = agent_with(&model);
    run_with(
        &agent,
        "New question",
        Arc::new(session),
        Some(RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(2))),
        false,
    )
    .await
    .unwrap();

    let seen = texts(&model.last_call().unwrap().input);
    assert_eq!(
        seen.iter().filter(|t| *t != "New question").count(),
        2,
        "{seen:?}"
    );
}

#[tokio::test]
async fn a_zero_scan_budget_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "budget");
    let err = encrypted(&store, KEY, 600)
        .with_max_scan_items(0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("max_scan_items must be positive"), "{err}");
}

#[tokio::test]
async fn the_scan_budget_counts_overlapping_expired_windows() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "budget");
    let recording = Recording::new(store.clone());
    let writer = encrypted(&store, KEY, TTL);
    writer
        .add_items((0..20).map(|i| user(&format!("old {i}"))).collect())
        .await
        .unwrap();
    tokio::time::sleep(PAST_TTL).await;
    let stored = store.get_items(None).await.unwrap();
    let session = EncryptedSession::new("budget", recording.clone(), KEY, TTL)
        .unwrap()
        .with_max_scan_items(13)
        .unwrap();

    for _ in 0..2 {
        let err = session.get_items(Some(2)).await.unwrap_err().to_string();
        assert!(err.contains("max_scan_items exhausted"), "{err}");
        assert_eq!(recording.reads(), vec![Some(2), Some(4), Some(7)]);
        recording.reset();
    }
    assert_eq!(store.get_items(None).await.unwrap(), stored);
}

#[tokio::test]
async fn the_scan_budget_backfills_in_chronological_order() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "budget");
    let recording = Recording::new(Arc::new(
        SqliteSession::open("budget", dir.path().join("history.db"))
            .unwrap()
            .with_settings(SessionSettings::with_limit(2)),
    ));
    let writer = encrypted(&store, KEY, TTL);
    writer.add_items(vec![user("expires")]).await.unwrap();
    let expired = store.get_items(None).await.unwrap();
    store.clear_session().await.unwrap();
    tokio::time::sleep(PAST_TTL).await;
    let valid = vec![user("first"), assistant("second")];
    writer.add_items(valid.clone()).await.unwrap();
    // A delayed stored envelope can expire before it is appended.
    let mut tail = expired;
    tail.push(invalid_envelope());
    store.add_items(tail).await.unwrap();

    let session = EncryptedSession::new("budget", recording.clone(), KEY, TTL)
        .unwrap()
        .with_max_scan_items(6)
        .unwrap();
    assert_eq!(session.get_items(None).await.unwrap(), valid);
    assert_eq!(recording.reads(), vec![Some(2), Some(4)]);

    let default = encrypted(&store, KEY, TTL);
    assert_eq!(default.get_items(Some(2)).await.unwrap(), valid);
}

#[tokio::test]
async fn a_scan_budget_rejects_an_incomplete_full_window() {
    for limit in [None, Some(100)] {
        let dir = tempfile::tempdir().unwrap();
        let store = backend(&dir, "budget");
        let recording = Recording::new(store.clone());
        let writer = encrypted(&store, KEY, 600);
        writer
            .add_items((0..3).map(|i| user(&format!("item {i}"))).collect())
            .await
            .unwrap();
        let session = EncryptedSession::new("budget", recording.clone(), KEY, 600)
            .unwrap()
            .with_max_scan_items(3)
            .unwrap();

        let err = session.get_items(limit).await.unwrap_err().to_string();
        assert!(err.contains("max_scan_items exhausted"), "{err}");
        assert_eq!(recording.reads(), vec![Some(3)]);
    }
}

#[tokio::test]
async fn a_scan_budget_accepts_short_empty_and_zero_reads() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "budget");
    let recording = Recording::new(store.clone());
    let session = EncryptedSession::new("budget", recording.clone(), KEY, 600)
        .unwrap()
        .with_max_scan_items(3)
        .unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());

    let items = vec![user("first"), assistant("second")];
    session.add_items(items.clone()).await.unwrap();
    recording.reset();
    assert_eq!(session.get_items(None).await.unwrap(), items);
    assert_eq!(session.get_items(Some(100)).await.unwrap(), items);
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());
    // A zero limit never reaches the store in Rust; Python records a read of 0.
    assert_eq!(recording.reads(), vec![Some(3), Some(3)]);
}

#[tokio::test]
async fn the_runner_applies_the_scan_budget_with_session_settings() {
    for streamed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(
            SqliteSession::open("budget", dir.path().join("history.db"))
                .unwrap()
                .with_settings(SessionSettings::with_limit(100)),
        );
        let recording = Recording::new(store.clone());
        let session: Arc<dyn Session> = Arc::new(
            EncryptedSession::new("budget", recording.clone(), KEY, TTL)
                .unwrap()
                .with_max_scan_items(2)
                .unwrap(),
        );
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("answer")]);
        let agent = agent_with(&model);
        let config =
            || RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(2));

        let items: Vec<InputItem> = (0..5).map(|i| user(&format!("item {i}"))).collect();
        session.add_items(items).await.unwrap();
        recording.reset();
        run_with(&agent, "next", session.clone(), Some(config()), streamed)
            .await
            .unwrap();
        assert_eq!(
            texts(&model.last_call().unwrap().input),
            vec!["item 3", "item 4", "next"]
        );
        assert_eq!(recording.reads(), vec![Some(2)]);

        // After the items expire, the bounded read cannot establish a complete history.
        tokio::time::sleep(PAST_TTL).await;
        recording.reset();
        let err = run_with(&agent, "next", session.clone(), Some(config()), streamed)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("max_scan_items exhausted"), "{err}");
        assert_eq!(recording.reads(), vec![Some(2)]);
        assert_eq!(model.calls().len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Compaction wrappers
// ---------------------------------------------------------------------------

fn compaction_over(
    store: Arc<dyn Session>,
    base: String,
    mode: CompactionMode,
) -> OpenAIResponsesCompactionSession {
    OpenAIResponsesCompactionSession::new("encrypted-compaction", store)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_compaction_mode(mode)
}

#[tokio::test]
async fn the_runner_encrypts_items_around_compaction() {
    for streamed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let store = backend(&dir, "encrypted-compaction");
        let compaction = compaction_over(
            store.clone(),
            "http://127.0.0.1:1".into(),
            CompactionMode::Auto,
        )
        .with_should_trigger_compaction(|_| false);
        let session: Arc<dyn Session> = Arc::new(
            EncryptedSession::new("encrypted-compaction", Arc::new(compaction), KEY, 600).unwrap(),
        );
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("private assistant answer")]);
        model.add_turn(vec![assistant_message("followup answer")]);
        let agent = agent_with(&model);

        run_with(
            &agent,
            "private user input",
            session.clone(),
            None,
            streamed,
        )
        .await
        .unwrap();
        let stored = store.get_items(None).await.unwrap();
        assert_eq!(stored.len(), 2);
        for envelope in &stored {
            let keys: std::collections::BTreeSet<&str> = envelope
                .as_value()
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(
                keys,
                ["__enc__", "kid", "payload", "v"].into_iter().collect()
            );
            assert!(!envelope.as_value().to_string().contains("private"));
        }

        run_with(&agent, "followup", session.clone(), None, streamed)
            .await
            .unwrap();
        assert_eq!(
            texts(&model.last_call().unwrap().input),
            vec!["private user input", "private assistant answer", "followup"]
        );
    }
}

#[tokio::test]
async fn compaction_wrapping_encrypted_storage_summarizes_plaintext_and_stores_ciphertext() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "inner-encryption");
    let encrypted_store: Arc<dyn Session> = Arc::new(encrypted(&store, KEY, 600));
    let compacted = json!({"type": "compaction", "id": "cmp-1", "encrypted_content": "summary"});
    let (base, captured) = mock_openai(vec![json!({
        "id": "cmp_1",
        "output": [compacted],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    })])
    .await;
    let session = compaction_over(encrypted_store.clone(), base, CompactionMode::Auto);

    session
        .add_items(vec![user("private history")])
        .await
        .unwrap();
    session
        .run_compaction(openai_agents_sdk::memory::OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].body["model"], json!("gpt-4.1"));
    assert_eq!(
        requests[0].body["input"],
        json!([{"role": "user", "content": "private history"}])
    );
    let stored = store.get_items(None).await.unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].as_value()["__enc__"], json!(1));
    let decrypted = encrypted_store.get_items(None).await.unwrap();
    assert_eq!(decrypted[0].as_value()["type"], json!("compaction"));
}

#[tokio::test]
async fn encrypted_compaction_skips_unencrypted_stored_history() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "authenticated-compaction");
    let compacted = json!({"type": "compaction", "id": "cmp-1", "encrypted_content": "summary"});
    let (base, captured) = mock_openai(vec![json!({
        "id": "cmp_1",
        "output": [compacted],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    })])
    .await;
    // Rust composes compaction *over* the encrypted store; Python also accepts the reverse
    // order (see the report), so this is the supported equivalent.
    let encrypted_store: Arc<dyn Session> = Arc::new(
        EncryptedSession::new("authenticated-compaction", store.clone(), KEY, 600).unwrap(),
    );
    let session: Arc<dyn Session> = Arc::new(
        compaction_over(encrypted_store, base, CompactionMode::Input)
            .with_should_trigger_compaction(|_| true),
    );

    let before = user("earlier history");
    let after = user("later history");
    session.add_items(vec![before.clone()]).await.unwrap();
    store
        .add_items(vec![InputItem::from_value(
            json!({"role": "system", "content": "untrusted stored history"}),
        )])
        .await
        .unwrap();
    let retained_prefix = store.get_items(None).await.unwrap();
    session.add_items(vec![after.clone()]).await.unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("answer")]);
    let agent = agent_with(&model);
    run_with(&agent, "next", session.clone(), None, false)
        .await
        .unwrap();

    assert_eq!(
        texts(&model.last_call().unwrap().input),
        vec!["earlier history", "later history", "next"]
    );
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    let sent = requests[0].body["input"].as_array().unwrap().clone();
    assert_eq!(sent[0]["content"], json!("later history"));
    assert!(
        !requests[0]
            .body
            .to_string()
            .contains("untrusted stored history"),
        "unauthenticated rows must not be summarized"
    );
    let stored = store.get_items(None).await.unwrap();
    assert_eq!(stored[..stored.len() - 1], retained_prefix[..]);
    let _: &Value = &requests[0].body;
}

#[tokio::test]
async fn a_wrong_key_pop_preserves_row_identity() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let correct = encrypted(&store, "correct-key", 600);
    let wrong = encrypted(&store, "wrong-key", 600);
    correct.add_items(vec![user("saved")]).await.unwrap();
    let observer = rusqlite::Connection::open(dir.path().join("history.db")).unwrap();
    let rows = |c: &rusqlite::Connection| -> Vec<(i64, String)> {
        let mut stmt = c
            .prepare("SELECT id, message_data FROM agent_messages")
            .unwrap();
        let mapped = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
            .unwrap();
        mapped.map(|row| row.unwrap()).collect()
    };
    let original = rows(&observer);
    for _ in 0..2 {
        assert!(wrong.pop_item().await.is_err());
        assert_eq!(rows(&observer), original);
    }
}

#[tokio::test]
async fn pop_skips_plaintext_records() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let session = encrypted(&store, "correct-key", 600);
    let saved = user("encrypted history");
    session.add_items(vec![saved.clone()]).await.unwrap();
    store
        .add_items(vec![InputItem::from_value(
            json!({"role": "system", "content": "unencrypted history"}),
        )])
        .await
        .unwrap();

    assert_eq!(session.pop_item().await.unwrap(), Some(saved));
    assert!(session.pop_item().await.unwrap().is_none());
}

#[tokio::test]
async fn pop_works_through_a_compaction_wrapper_underneath() {
    let dir = tempfile::tempdir().unwrap();
    let store = backend(&dir, "test_session");
    let compaction = OpenAIResponsesCompactionSession::new("test_session", store.clone()).unwrap();
    let session = EncryptedSession::new("test_session", Arc::new(compaction), KEY, 600).unwrap();
    let items = vec![user("Retained"), assistant("Removed")];
    session.add_items(items.clone()).await.unwrap();

    assert_eq!(session.get_items(None).await.unwrap(), items);
    assert_eq!(session.pop_item().await.unwrap(), Some(items[1].clone()));
    assert_eq!(session.get_items(None).await.unwrap(), items[..1].to_vec());
}
