//! Parity port of `tests/extensions/memory/test_sqlalchemy_session.py`.
//!
//! Python's `SQLAlchemySession` maps to Rust's `SqlxSession`; the SQLite-backed Python tests
//! run here against temporary SQLite files (`sqlite://...?mode=rwc`), with the tests that
//! inspect or edit rows going through the session's own pool.
#![cfg(all(feature = "sqlx", feature = "sqlite"))]

use std::sync::Arc;

use openai_agents_sdk::memory::{Session, SessionSettings, SqlxDialect, SqlxSession};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::json;
use sqlx::Row;

mod py_mcpmem_support;
use py_mcpmem_support::item_text;

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": text}))
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn url(dir: &tempfile::TempDir, name: &str) -> String {
    format!("sqlite://{}?mode=rwc", dir.path().join(name).display())
}

async fn open(dir: &tempfile::TempDir, id: &str) -> SqlxSession {
    SqlxSession::from_url(id, &url(dir, "sessions.db"), true)
        .await
        .unwrap()
}

async fn stored_rows(session: &SqlxSession, id: &str) -> Vec<String> {
    sqlx::query("SELECT message_data FROM agent_messages WHERE session_id = ? ORDER BY id")
        .bind(id.to_string())
        .fetch_all(session.pool())
        .await
        .unwrap()
        .iter()
        .map(|r| r.get::<String, _>(0))
        .collect()
}

async fn insert_raw(session: &SqlxSession, id: &str, raw: &str) {
    sqlx::query("INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)")
        .bind(id.to_string())
        .bind(raw.to_string())
        .execute(session.pool())
        .await
        .unwrap();
}

#[tokio::test]
async fn direct_ops() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "direct").await;
    assert_eq!(session.dialect(), SqlxDialect::Sqlite);

    session
        .add_items(vec![user("Hello"), assistant("Hi there!")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Hello", "Hi there!"]
    );

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "Hi there!"
    );
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["Hello"]
    );

    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn non_ascii_is_stored_escaped_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "default_ascii_storage").await;
    let item = user("café");
    session.add_items(vec![item.clone()]).await.unwrap();

    let stored = stored_rows(&session, "default_ascii_storage").await;
    assert_eq!(stored.len(), 1);
    assert!(stored[0].contains("\\u00e9"), "{}", stored[0]);
    assert!(!stored[0].contains("café"));
    assert_eq!(session.get_items(None).await.unwrap(), vec![item]);
}

#[tokio::test]
async fn non_ascii_can_be_stored_without_escaping() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "non_ascii_storage")
        .await
        .with_ensure_ascii(false);
    let item = user("café");
    session.add_items(vec![item.clone()]).await.unwrap();

    let stored = stored_rows(&session, "non_ascii_storage").await;
    assert!(stored[0].contains("café"), "{}", stored[0]);
    assert!(!stored[0].contains("\\u00e9"));
    assert_eq!(session.get_items(None).await.unwrap(), vec![item]);
}

#[tokio::test]
async fn astral_characters_escape_as_surrogate_pairs_and_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "astral").await;
    session.add_items(vec![user("😊👍")]).await.unwrap();
    let stored = stored_rows(&session, "astral").await;
    assert!(stored[0].contains("\\ud83d\\ude0a"), "{}", stored[0]);
    assert_eq!(texts(&session.get_items(None).await.unwrap()), vec!["😊👍"]);
}

#[tokio::test]
async fn sessions_are_isolated() {
    let dir = tempfile::tempdir().unwrap();
    let path = url(&dir, "iso.db");
    let a = SqlxSession::from_url("a", &path, true).await.unwrap();
    let b = SqlxSession::from_url("b", &path, true).await.unwrap();
    a.add_items(vec![user("from a")]).await.unwrap();
    b.add_items(vec![user("from b")]).await.unwrap();
    assert_eq!(texts(&a.get_items(None).await.unwrap()), vec!["from a"]);
    assert_eq!(texts(&b.get_items(None).await.unwrap()), vec!["from b"]);
    a.clear_session().await.unwrap();
    assert_eq!(texts(&b.get_items(None).await.unwrap()), vec!["from b"]);
}

#[tokio::test]
async fn get_items_with_limit() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "limit").await;
    session
        .add_items(vec![user("1"), assistant("2"), user("3"), assistant("4")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["3", "4"]
    );
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
    assert_eq!(session.get_items(Some(10)).await.unwrap().len(), 4);
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());
}

#[tokio::test]
async fn pop_from_an_empty_session_and_empty_adds() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "empty").await;
    assert!(session.pop_item().await.unwrap().is_none());
    session.add_items(vec![]).await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn concurrent_pops_return_each_row_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = url(&dir, "concurrent_pop.db");
    let writer = SqlxSession::from_url("concurrent_pop", &path, true)
        .await
        .unwrap();
    let other = SqlxSession::from_url("concurrent_pop", &path, false)
        .await
        .unwrap();
    writer
        .add_items(vec![user("first"), assistant("second")])
        .await
        .unwrap();

    let (a, b) = tokio::join!(writer.pop_item(), other.pop_item());
    let mut popped: Vec<String> = [a.unwrap(), b.unwrap()]
        .into_iter()
        .flatten()
        .map(|i| item_text(&i))
        .collect();
    popped.sort();
    assert_eq!(popped, vec!["first", "second"]);
}

#[tokio::test]
async fn pop_skips_a_corrupt_most_recent_row() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "pop_corrupt").await;
    session.add_items(vec![user("valid")]).await.unwrap();
    insert_raw(&session, "pop_corrupt", "not valid json {{{").await;

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "valid"
    );
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn pop_returns_none_after_dropping_only_corrupt_rows() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "only_corrupt").await;
    session.add_items(vec![user("seed")]).await.unwrap();
    session.pop_item().await.unwrap();
    insert_raw(&session, "only_corrupt", "not valid json {{{").await;

    assert!(session.pop_item().await.unwrap().is_none());
    assert!(stored_rows(&session, "only_corrupt").await.is_empty());
}

#[tokio::test]
async fn limit_counts_valid_rows_past_corrupt_newest_ones() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "limit_corrupt").await;
    session
        .add_items(vec![user("valid 0"), assistant("valid 1"), user("valid 2")])
        .await
        .unwrap();
    insert_raw(&session, "limit_corrupt", "not valid json {{{").await;

    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
    assert_eq!(session.get_items(Some(10)).await.unwrap().len(), 3);
}

#[tokio::test]
async fn limit_widening_stops_at_the_end_of_history() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "exhausted").await;
    session.add_items(vec![user("only valid")]).await.unwrap();
    assert_eq!(
        texts(&session.get_items(Some(5)).await.unwrap()),
        vec!["only valid"]
    );
}

#[tokio::test]
async fn first_writes_from_concurrent_sessions_are_all_kept() {
    let dir = tempfile::tempdir().unwrap();
    let path = url(&dir, "first_access.db");
    let first = SqlxSession::from_url("first_access", &path, true)
        .await
        .unwrap();
    let mut handles = Vec::new();
    for i in 0..8 {
        let path = path.clone();
        handles.push(tokio::spawn(async move {
            let session = SqlxSession::from_url("first_access", &path, true)
                .await
                .unwrap();
            session
                .add_items(vec![user(&format!("m{i}"))])
                .await
                .unwrap();
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
    assert_eq!(first.get_items(None).await.unwrap().len(), 8);
}

#[tokio::test]
async fn items_keep_insertion_order_even_with_identical_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "same_timestamp_test").await;
    session.add_items(vec![user("older")]).await.unwrap();
    session
        .add_items(vec![assistant("reasoning"), assistant("message")])
        .await
        .unwrap();

    // Give every row the same timestamp: ordering must fall back to the row id.
    sqlx::query("UPDATE agent_messages SET created_at = '2025-10-15 17:26:39'")
        .execute(session.pool())
        .await
        .unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["older", "reasoning", "message"]
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "message"
    );
    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["older", "reasoning"]
    );
}

#[tokio::test]
async fn the_pool_is_exposed_for_url_and_external_construction() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "engine").await;
    session.add_items(vec![user("x")]).await.unwrap();

    let reuse =
        SqlxSession::with_pool("engine", session.pool().clone(), SqlxDialect::Sqlite, false)
            .await
            .unwrap();
    assert_eq!(texts(&reuse.get_items(None).await.unwrap()), vec!["x"]);
}

#[tokio::test]
async fn custom_table_names_are_validated_and_used() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqlxSession::from_url("custom", &url(&dir, "custom.db"), false)
        .await
        .unwrap()
        .with_tables("my_sessions", "my_messages")
        .unwrap();
    session.create_tables().await.unwrap();
    session.add_items(vec![user("hi")]).await.unwrap();
    assert_eq!(texts(&session.get_items(None).await.unwrap()), vec!["hi"]);

    let bad = SqlxSession::from_url("custom", &url(&dir, "custom.db"), false)
        .await
        .unwrap()
        .with_tables("ok", "bad table; DROP");
    assert!(bad.is_err());
}

#[tokio::test]
async fn session_settings_default_constructor_and_limits() {
    let dir = tempfile::tempdir().unwrap();
    assert!(open(&dir, "s").await.session_settings().is_none());

    let limited = open(&dir, "settings")
        .await
        .with_settings(SessionSettings::with_limit(3));
    assert_eq!(limited.session_settings().unwrap().limit, Some(3));
    limited
        .add_items((0..5).map(|i| user(&format!("Message {i}"))).collect())
        .await
        .unwrap();
    assert_eq!(
        texts(&limited.get_items(None).await.unwrap()),
        vec!["Message 2", "Message 3", "Message 4"]
    );
    assert_eq!(
        texts(&limited.get_items(Some(2)).await.unwrap()),
        vec!["Message 3", "Message 4"]
    );
}

#[tokio::test]
async fn the_runner_stores_turns_and_honours_a_run_config_limit() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(open(&dir, "runner").await);
    session
        .add_items((0..10).map(|i| user(&format!("Turn {i}"))).collect())
        .await
        .unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("Got it")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let result = Runner::run_with_options(
        &agent,
        "New question",
        RunOptions::new(()).session(session.clone()).run_config(
            RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(2)),
        ),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Got it"));

    let seen = texts(&model.last_call().unwrap().input);
    assert_eq!(
        seen.iter().filter(|c| *c != "New question").count(),
        2,
        "{seen:?}"
    );
    let stored = texts(&session.get_items(None).await.unwrap());
    assert!(stored.contains(&"New question".to_string()) && stored.contains(&"Got it".to_string()));
}

#[tokio::test]
async fn compaction_snapshots_replace_a_suffix_and_detect_concurrent_writes() {
    let dir = tempfile::tempdir().unwrap();
    let session = open(&dir, "compact").await;
    session
        .add_items(vec![user("keep"), assistant("old 1"), assistant("old 2")])
        .await
        .unwrap();

    let snapshot = session.compaction_snapshot(2).await.unwrap().unwrap();
    assert_eq!(texts(&snapshot.items), vec!["old 1", "old 2"]);
    assert!(!snapshot.complete);

    session.add_items(vec![user("newer")]).await.unwrap();
    assert!(
        !session
            .replace_compaction_suffix(&snapshot.handles, vec![assistant("summary")])
            .await
            .unwrap(),
        "a write after the snapshot must make the replacement refuse"
    );

    let fresh = session.compaction_snapshot(2).await.unwrap().unwrap();
    assert!(
        session
            .replace_compaction_suffix(&fresh.handles, vec![assistant("summary")])
            .await
            .unwrap()
    );
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["keep", "old 1", "summary"]
    );
}
