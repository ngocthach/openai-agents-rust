//! Parity port of `tests/extensions/memory/test_dapr_session.py`.
//!
//! Python swaps in a `FakeDaprClient` over gRPC; Rust's `DaprSession` speaks the Dapr HTTP
//! state API, so these tests run an in-process sidecar (same shape as `mock_dapr` in
//! `tests/sessions_tests.rs`) with ETags, `first-write` concurrency, seedable 409 conflicts and
//! a store the test can read and corrupt directly.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{DaprConsistency, DaprSession, Session, SessionSettings};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text, serve};

#[derive(Default)]
struct Sidecar {
    /// key -> (value as the sidecar returns it, etag)
    store: HashMap<String, (String, String)>,
    requests: Vec<Request>,
    /// Remaining 409 answers for conditional writes to keys ending with this suffix.
    conflicts: HashMap<String, u32>,
    /// Keys ending with this suffix always answer 409 on conditional writes.
    always_conflict: Option<String>,
    revision: u64,
}

async fn mock_dapr() -> (String, Arc<Mutex<Sidecar>>) {
    let shared = Arc::new(Mutex::new(Sidecar::default()));
    let state = shared.clone();
    let base = serve(Arc::new(move |req: &Request| {
        let path = req.path.split('?').next().unwrap_or_default().to_string();
        if path == "/v1.0/metadata" {
            return Reply::json(json!({"id": "mock"}));
        }
        let Some(rest) = path.strip_prefix("/v1.0/state/statestore") else {
            return Reply::empty(404);
        };
        let key = rest.trim_start_matches('/').to_string();
        let mut state = state.lock().unwrap();
        state.requests.push(req.clone());
        match req.method.as_str() {
            "GET" => match state.store.get(&key) {
                None => Reply::empty(204),
                Some((value, etag)) => Reply {
                    status: 200,
                    headers: vec![
                        ("content-type".into(), "application/json".into()),
                        ("etag".into(), etag.clone()),
                    ],
                    body: value.clone(),
                },
            },
            "POST" => {
                let entry = req.body[0].clone();
                let entry_key = entry["key"].as_str().unwrap_or_default().to_string();
                let sent_etag = entry["etag"].as_str().map(str::to_string);
                let current = state.store.get(&entry_key).map(|(_, e)| e.clone());
                let mut conflict = false;
                if sent_etag.is_some() {
                    let suffix = state
                        .conflicts
                        .keys()
                        .find(|s| entry_key.ends_with(s.as_str()))
                        .cloned();
                    if let Some(suffix) = suffix {
                        let left = state.conflicts.get_mut(&suffix).unwrap();
                        if *left > 0 {
                            *left -= 1;
                            conflict = true;
                        }
                    }
                    if state
                        .always_conflict
                        .as_deref()
                        .is_some_and(|s| entry_key.ends_with(s))
                    {
                        conflict = true;
                    }
                    if sent_etag != current {
                        conflict = true;
                    }
                }
                if conflict {
                    return Reply::status(409, json!({"errorCode": "ERR_STATE_SAVE"}).to_string());
                }
                state.revision += 1;
                let revision = state.revision.to_string();
                state
                    .store
                    .insert(entry_key, (entry["value"].to_string(), revision));
                Reply::empty(204)
            }
            "DELETE" => {
                state.store.remove(&key);
                Reply::empty(204)
            }
            _ => Reply::empty(405),
        }
    }))
    .await;
    (base, shared)
}

fn session(base: &str, id: &str) -> DaprSession {
    DaprSession::new(id, "statestore", base)
}

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": text}))
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

fn inject(state: &Arc<Mutex<Sidecar>>, key: &str, raw: &str) {
    state
        .lock()
        .unwrap()
        .store
        .insert(key.into(), (raw.into(), "9000".into()));
}

fn metadata(state: &Arc<Mutex<Sidecar>>, key: &str) -> serde_json::Value {
    let raw = state.lock().unwrap().store[key].0.clone();
    serde_json::from_str(&raw).unwrap()
}

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn direct_ops() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "direct");

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
    assert!(
        state.lock().unwrap().store.is_empty(),
        "clear removes messages and metadata"
    );
}

#[tokio::test]
async fn sessions_are_isolated() {
    let (base, _) = mock_dapr().await;
    let a = session(&base, "a");
    let b = session(&base, "b");
    a.add_items(vec![user("from a")]).await.unwrap();
    b.add_items(vec![user("from b")]).await.unwrap();
    assert_eq!(texts(&a.get_items(None).await.unwrap()), vec!["from a"]);
    assert_eq!(texts(&b.get_items(None).await.unwrap()), vec!["from b"]);
}

#[tokio::test]
async fn get_items_with_limit() {
    let (base, _) = mock_dapr().await;
    let session = session(&base, "limit");
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
async fn pop_from_empty_and_empty_add() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "empty");
    assert!(session.pop_item().await.unwrap().is_none());
    session.add_items(vec![]).await.unwrap();
    assert!(state.lock().unwrap().store.is_empty());
}

#[tokio::test]
async fn unicode_special_characters_and_problematic_strings_round_trip() {
    let (base, _) = mock_dapr().await;
    let session = session(&base, "text");
    let samples = [
        "こんにちは",
        "😊👍",
        "{\"a\": [1, 2]}",
        "line1\nline2\r\n\ttabbed",
        "quote \" backslash \\",
        "O'Reilly",
        "",
    ];
    session
        .add_items(samples.iter().map(|s| user(s)).collect())
        .await
        .unwrap();
    assert_eq!(texts(&session.get_items(None).await.unwrap()), samples);
}

#[tokio::test]
async fn concurrent_writes_are_all_kept() {
    let (base, _) = mock_dapr().await;
    let session = Arc::new(session(&base, "concurrent"));
    let mut handles = Vec::new();
    for i in 0..10 {
        let session = session.clone();
        handles.push(tokio::spawn(async move {
            session
                .add_items(vec![user(&format!("Message {i}"))])
                .await
                .unwrap();
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
    let mut seen = texts(&session.get_items(None).await.unwrap());
    seen.sort();
    let mut expected: Vec<String> = (0..10).map(|i| format!("Message {i}")).collect();
    expected.sort();
    assert_eq!(seen, expected);
}

#[tokio::test]
async fn ping_reports_connectivity() {
    let (base, _) = mock_dapr().await;
    assert!(session(&base, "ping").ping().await);
    assert!(!session("http://127.0.0.1:1", "ping").ping().await);
}

// ---------------------------------------------------------------------------
// Concurrency, TTL, consistency
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_items_retries_on_an_etag_conflict() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "retry_add");
    // Conditional (etag) writes only exist once the key does.
    session.add_items(vec![user("seed")]).await.unwrap();
    state
        .lock()
        .unwrap()
        .conflicts
        .insert(":messages".into(), 2);

    session.add_items(vec![user("survives")]).await.unwrap();

    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["seed", "survives"]
    );
    assert_eq!(
        state.lock().unwrap().conflicts[":messages"],
        0,
        "both conflicts were consumed"
    );
}

#[tokio::test]
async fn pop_item_retries_on_an_etag_conflict() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "retry_pop");
    session.add_items(vec![user("a"), user("b")]).await.unwrap();
    state
        .lock()
        .unwrap()
        .conflicts
        .insert(":messages".into(), 2);

    assert_eq!(item_text(&session.pop_item().await.unwrap().unwrap()), "b");
    assert_eq!(texts(&session.get_items(None).await.unwrap()), vec!["a"]);
}

#[tokio::test]
async fn a_conflict_that_never_clears_eventually_fails() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "never");
    session.add_items(vec![user("seed")]).await.unwrap();
    state.lock().unwrap().always_conflict = Some(":messages".into());
    assert!(session.add_items(vec![user("lost")]).await.is_err());
    assert_eq!(texts(&session.get_items(None).await.unwrap()), vec!["seed"]);
}

#[tokio::test]
async fn ttl_is_sent_as_item_metadata() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "ttl_test").with_ttl(60);
    session.add_items(vec![user("expires")]).await.unwrap();

    let requests = state.lock().unwrap().requests.clone();
    let writes: Vec<&Request> = requests.iter().filter(|r| r.method == "POST").collect();
    assert!(!writes.is_empty());
    for write in writes {
        assert_eq!(write.body[0]["metadata"]["ttlInSeconds"], json!("60"));
    }
    assert_eq!(session.get_items(None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn consistency_levels_are_sent_with_reads_and_writes() {
    for (level, expected) in [
        (DaprConsistency::Eventual, "eventual"),
        (DaprConsistency::Strong, "strong"),
    ] {
        let (base, state) = mock_dapr().await;
        let session = session(&base, "consistency").with_consistency(level);
        session.add_items(vec![user("x")]).await.unwrap();
        session.get_items(None).await.unwrap();

        let requests = state.lock().unwrap().requests.clone();
        let read = requests.iter().find(|r| r.method == "GET").unwrap();
        assert!(
            read.path.contains(&format!("consistency={expected}")),
            "{}",
            read.path
        );
        let write = requests.iter().find(|r| r.method == "POST").unwrap();
        assert_eq!(write.body[0]["options"]["consistency"], json!(expected));
    }
}

#[tokio::test]
async fn the_api_token_header_is_sent_when_configured() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "token").with_api_token("secret-token");
    session.add_items(vec![user("x")]).await.unwrap();
    let requests = state.lock().unwrap().requests.clone();
    assert!(
        requests
            .iter()
            .all(|r| r.header("dapr-api-token") == Some("secret-token")),
        "every request carries the token"
    );
}

// ---------------------------------------------------------------------------
// Corrupt state
// ---------------------------------------------------------------------------

#[tokio::test]
async fn limit_counts_valid_entries_past_corrupt_newest_ones() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "limit_corrupt");
    inject(
        &state,
        "limit_corrupt:messages",
        &json!([
            json!({"role": "user", "content": "valid 0"}).to_string(),
            json!({"role": "assistant", "content": "valid 1"}).to_string(),
            json!({"role": "user", "content": "valid 2"}).to_string(),
            "not valid json {{{",
        ])
        .to_string(),
    );
    assert_eq!(
        texts(&session.get_items(Some(2)).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
    assert_eq!(session.get_items(Some(10)).await.unwrap().len(), 3);
}

#[tokio::test]
async fn pop_skips_corrupt_newest_entries_then_reports_none() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "pop_corrupt");
    inject(
        &state,
        "pop_corrupt:messages",
        &json!([
            json!({"role": "user", "content": "valid"}).to_string(),
            "not valid json {{{"
        ])
        .to_string(),
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "valid"
    );
    assert!(session.pop_item().await.unwrap().is_none());

    inject(
        &state,
        "pop_corrupt:messages",
        &json!(["not valid json {{{"]).to_string(),
    );
    assert!(session.pop_item().await.unwrap().is_none());
}

#[tokio::test]
async fn add_items_refuses_to_overwrite_corrupted_aggregate_state() {
    for raw in ["invalid json data", "\u{fffd}", "{\"some\": \"object\"}"] {
        let (base, state) = mock_dapr().await;
        let session = session(&base, "corruption_test");
        inject(&state, "corruption_test:messages", raw);

        assert!(session.get_items(None).await.unwrap().is_empty(), "{raw:?}");
        let err = session
            .add_items(vec![user("valid after corruption")])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("stored Dapr session messages"),
            "{raw:?}: {err}"
        );
        assert_eq!(
            state.lock().unwrap().store["corruption_test:messages"].0,
            raw
        );
    }
}

#[tokio::test]
async fn pop_item_refuses_to_rewrite_corrupted_aggregate_state() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "pop_aggregate");
    inject(&state, "pop_aggregate:messages", "invalid json data");
    assert!(session.pop_item().await.is_err());
    assert_eq!(
        state.lock().unwrap().store["pop_aggregate:messages"].0,
        "invalid json data"
    );
}

#[tokio::test]
async fn non_list_state_reads_as_empty() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "not_list_test");
    inject(
        &state,
        "not_list_test:messages",
        &json!({"some": "object"}).to_string(),
    );
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn messages_stored_as_plain_objects_are_read() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "deserialized_test");
    inject(
        &state,
        "deserialized_test:messages",
        &json!([
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
        ])
        .to_string(),
    );
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["First message", "Second message"]
    );
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn metadata_keeps_created_at_and_advances_updated_at() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "meta");
    session.add_items(vec![user("first")]).await.unwrap();
    let first = metadata(&state, "meta:metadata");
    assert_eq!(first["session_id"], json!("meta"));

    // Force an older created_at so a regression that rewrites it is visible.
    inject(
        &state,
        "meta:metadata",
        &json!({"session_id": "meta", "created_at": "100", "updated_at": "100"}).to_string(),
    );
    session.add_items(vec![user("second")]).await.unwrap();
    let second = metadata(&state, "meta:metadata");
    assert_eq!(second["created_at"], json!("100"));
    assert_ne!(second["updated_at"], json!("100"));
}

#[tokio::test]
async fn the_first_metadata_write_is_unconditional() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "first_meta");
    session.add_items(vec![user("first")]).await.unwrap();

    let requests = state.lock().unwrap().requests.clone();
    let first_metadata_write = requests
        .iter()
        .find(|r| r.method == "POST" && r.body[0]["key"] == json!("first_meta:metadata"))
        .unwrap();
    assert!(
        first_metadata_write.body[0].get("etag").is_none(),
        "no etag for a new key"
    );
}

#[tokio::test]
async fn a_stale_metadata_etag_retries_and_keeps_created_at() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "stale_meta");
    inject(
        &state,
        "stale_meta:metadata",
        &json!({"session_id": "stale_meta", "created_at": "100", "updated_at": "100"}).to_string(),
    );
    state
        .lock()
        .unwrap()
        .conflicts
        .insert(":metadata".into(), 1);

    session.add_items(vec![user("x")]).await.unwrap();

    let meta = metadata(&state, "stale_meta:metadata");
    assert_eq!(meta["created_at"], json!("100"));
    assert_ne!(meta["updated_at"], json!("100"));
}

#[tokio::test]
async fn add_items_survives_the_metadata_write_giving_up() {
    let (base, state) = mock_dapr().await;
    state.lock().unwrap().always_conflict = Some(":metadata".into());
    let session = session(&base, "survives");
    inject(
        &state,
        "survives:metadata",
        &json!({"session_id": "survives", "created_at": "1", "updated_at": "1"}).to_string(),
    );

    // The items committed, so the failed bookkeeping must not surface as an error that a caller
    // would retry into a duplicate append.
    session
        .add_items(vec![user("committed once")])
        .await
        .unwrap();
    assert_eq!(
        texts(&session.get_items(None).await.unwrap()),
        vec!["committed once"]
    );
}

// ---------------------------------------------------------------------------
// Session settings and the runner
// ---------------------------------------------------------------------------

#[tokio::test]
async fn session_settings_default_constructor_and_limits() {
    let (base, _) = mock_dapr().await;
    assert!(session(&base, "s").session_settings().is_none());

    let limited = session(&base, "settings").with_settings(SessionSettings::with_limit(3));
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
    let (base, _) = mock_dapr().await;
    let session = Arc::new(session(&base, "runner"));
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
async fn the_key_layout_matches_the_python_backend() {
    let (base, state) = mock_dapr().await;
    let session = session(&base, "customer:archive");
    assert_eq!(session.messages_key(), "customer:archive:messages");
    assert_eq!(session.metadata_key(), "customer:archive:metadata");
    session.add_items(vec![user("x")]).await.unwrap();
    let store = state.lock().unwrap();
    assert!(store.store.contains_key("customer:archive:messages"));
    assert!(store.store.contains_key("customer:archive:metadata"));
}
