//! Parity port of the SQLite cases of `tests/memory/test_compaction_suffix.py`: automatic
//! compaction replaces only the stored suffix a run's model exchange covers, in one transaction,
//! and leaves expired, unverifiable and still-needed rows alone.
//!
//! Python parametrizes every case over `SQLiteSession` and `AsyncSQLiteSession`; Rust has one
//! store, so only the first variant exists here. Python also moves a mocked `cryptography`
//! clock; here the items' lifetime is a few seconds of real time and the tests let it lapse.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(all(feature = "sqlite", feature = "encrypt"))]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{
    CompactionMode, EncryptedSession, OpenAIResponsesCompactionSession, Session, SessionSettings,
    SqliteSession,
};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::json;

mod py_core6_mcpmisc_support;
mod py_mcpmem_support;
use py_core6_mcpmisc_support::*;

const TTL_SECONDS: u64 = 2;

async fn let_ttl_lapse() {
    tokio::time::sleep(std::time::Duration::from_millis(3300)).await;
}

fn db() -> (tempfile::TempDir, std::path::PathBuf, Arc<SqliteSession>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    let backend = SqliteSession::open("suffix", &path)
        .unwrap()
        .with_settings(SessionSettings::with_limit(20));
    (dir, path, Arc::new(backend))
}

fn raw_rows(path: &std::path::Path) -> Vec<String> {
    let conn = rusqlite::Connection::open(path).unwrap();
    let mut stmt = conn
        .prepare("SELECT message_data FROM agent_messages ORDER BY id")
        .unwrap();
    stmt.query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn add_trigger(path: &std::path::Path, name: &str, condition: &str) {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.execute_batch(&format!(
        "CREATE TRIGGER {name} BEFORE INSERT ON agent_messages {condition} \
         BEGIN SELECT RAISE(ABORT, 'synthetic insert failure'); END"
    ))
    .unwrap();
}

fn key_session(inner: Arc<dyn Session>, key: &str, ttl: u64) -> Arc<EncryptedSession> {
    Arc::new(EncryptedSession::new("suffix", inner, key, ttl).unwrap())
}

/// The compaction session and the session the runner talks to, with encryption below
/// (`outer == false`) or above (`outer == true`) the compaction session.
fn stack(
    outer: bool,
    backend: Arc<dyn Session>,
    endpoint: &Compaction,
    mode: CompactionMode,
    hook: Option<bool>,
    ttl: u64,
) -> (Arc<dyn Session>, Arc<OpenAIResponsesCompactionSession>) {
    let build = |store: Arc<dyn Session>| {
        let session = compaction_session("suffix", store, endpoint, mode);
        Arc::new(match hook {
            Some(answer) => session.with_should_trigger_compaction(move |_| answer),
            None => session,
        })
    };
    if outer {
        let compaction = build(backend);
        (
            key_session(compaction.clone(), "synthetic-key", ttl),
            compaction,
        )
    } else {
        let compaction = build(key_session(backend, "synthetic-key", ttl));
        (compaction.clone(), compaction)
    }
}

fn live_items() -> Vec<InputItem> {
    (0..20)
        .map(|i| {
            InputItem::from_value(json!({
                "role": if i % 2 == 1 { "assistant" } else { "user" },
                "content": format!("live {i}"),
            }))
        })
        .collect()
}

fn config_limit(limit: usize) -> RunConfig<()> {
    RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(limit))
}

fn four_users() -> Vec<InputItem> {
    (0..4).map(|i| user(&i.to_string())).collect()
}

async fn run_once(
    agent: &Agent<()>,
    prompt: &str,
    session: &Arc<dyn Session>,
    config: Option<RunConfig<()>>,
) -> Result<(), openai_agents_sdk::error::AgentsError> {
    let mut options = RunOptions::new(()).session(session.clone());
    if let Some(config) = config {
        options = options.run_config(config);
    }
    Runner::run_with_options(agent, prompt, options)
        .await
        .map(|_| ())
}

fn worker(turns: &[&str]) -> (Arc<openai_agents_sdk::testing::FakeModel>, Agent<()>) {
    let model = model_with(texts(turns));
    let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
    (model, agent)
}

// test_expired_prefix_does_not_block_full_live_window (sqlite x 2 x 2)
async fn expired_prefix_case(outer: bool, expired_count: usize) {
    let endpoint = Compaction::replying(vec![]).await;
    let (_dir, _path, backend) = db();
    let (session, _) = stack(
        outer,
        backend.clone(),
        &endpoint,
        CompactionMode::Auto,
        None,
        TTL_SECONDS,
    );
    session
        .add_items((0..expired_count).map(|_| user("expired")).collect())
        .await
        .unwrap();
    let_ttl_lapse().await;
    session.add_items(live_items()).await.unwrap();
    let (_, agent) = worker(&["done"]);
    run_once(&agent, "continue", &session, None).await.unwrap();
    let label = format!("outer={outer} expired={expired_count}");
    assert_eq!(endpoint.requests().len(), 1, "{label}");
    assert!(
        !endpoint.requests()[0].to_string().contains("expired"),
        "{label}"
    );
    if expired_count > 1 {
        // Auto mode must scope the request to the suffix, not the whole response chain.
        assert_eq!(endpoint.inputs()[0].len(), 22, "{label}");
    }
    assert!(
        backend.get_items(Some(100)).await.unwrap().is_empty(),
        "{label}"
    );
}

#[tokio::test]
async fn expired_prefix_does_not_block_full_live_window() {
    openai_agents_sdk::set_tracing_disabled(true);
    futures::future::join_all([
        expired_prefix_case(false, 1),
        expired_prefix_case(false, 40),
        expired_prefix_case(true, 1),
        expired_prefix_case(true, 40),
    ])
    .await;
}

// test_suffix_transaction_rolls_back_failed_insert
#[tokio::test]
async fn suffix_transaction_rolls_back_failed_insert() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (_dir, path, backend) = db();
    backend.add_items(four_users()).await.unwrap();
    add_trigger(
        &path,
        "reject_summary",
        "WHEN NEW.message_data LIKE '%compaction%'",
    );
    let endpoint = Compaction::replying(vec![summary()]).await;
    let session = as_session(always(compaction_session(
        "suffix",
        backend.clone(),
        &endpoint,
        CompactionMode::Input,
    )));
    let (_, agent) = worker(&["done"]);
    let error = run_once(&agent, "continue", &session, Some(config_limit(1)))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("synthetic insert failure"),
        "{error}"
    );
    let stored = values(&backend.get_items(Some(100)).await.unwrap());
    assert_eq!(stored.len(), 6);
    assert_eq!(stored[..4], values(&four_users())[..]);
    assert_eq!(stored[4], json!({"role": "user", "content": "continue"}));
    assert!(stored[5].to_string().contains("done"));
}

// test_new_backend_write_invalidates_snapshot
#[tokio::test]
async fn new_backend_write_invalidates_snapshot() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (_dir, path, backend) = db();
    backend.add_items(four_users()).await.unwrap();
    let newer = json!({"role": "user", "content": "concurrent backend append"});
    let (writer_path, row) = (path.clone(), newer.to_string());
    let endpoint = Compaction::with(move |_| {
        let conn = rusqlite::Connection::open(&writer_path).unwrap();
        conn.execute(
            "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
            rusqlite::params!["suffix", row],
        )
        .unwrap();
        json!({"id": "cmp", "output": [], "usage": usage()})
    })
    .await;
    let session = as_session(always(compaction_session(
        "suffix",
        backend.clone(),
        &endpoint,
        CompactionMode::Input,
    )));
    let (_, agent) = worker(&["done"]);
    run_once(&agent, "continue", &session, Some(config_limit(1)))
        .await
        .unwrap();
    let stored = values(&backend.get_items(Some(100)).await.unwrap());
    assert_eq!(stored.len(), 7);
    assert_eq!(stored[4], json!({"role": "user", "content": "continue"}));
    assert!(stored[5].to_string().contains("done"));
    assert_eq!(stored[6], newer);
}

// test_partial_previous_response_mode_preserves_history
#[tokio::test]
async fn partial_previous_response_mode_preserves_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (_dir, _path, backend) = db();
    backend.add_items(four_users()).await.unwrap();
    let endpoint = Compaction::replying(vec![]).await;
    let session = as_session(always(compaction_session(
        "suffix",
        backend.clone(),
        &endpoint,
        CompactionMode::PreviousResponseId,
    )));
    let (_, agent) = worker(&["done"]);
    run_once(&agent, "continue", &session, Some(config_limit(1)))
        .await
        .unwrap();
    assert!(endpoint.requests().is_empty());
    assert_eq!(backend.get_items(Some(100)).await.unwrap().len(), 6);
}

// test_failed_encrypted_suffix_transaction_preserves_original_tokens (sqlite x 2)
async fn failed_encrypted_transaction_case(outer: bool) {
    let (_dir, path, backend) = db();
    let saved = Arc::new(Mutex::new(Vec::<String>::new()));
    let (trigger_path, record) = (path.clone(), saved.clone());
    let endpoint = Compaction::with(move |_| {
        *record.lock().unwrap() = raw_rows(&trigger_path);
        add_trigger(&trigger_path, "reject_all", "");
        json!({"id": "cmp", "output": [summary()], "usage": usage()})
    })
    .await;
    let (session, _) = stack(
        outer,
        backend.clone(),
        &endpoint,
        CompactionMode::Input,
        Some(true),
        TTL_SECONDS,
    );
    session
        .add_items((0..4).map(|_| user("expired")).collect())
        .await
        .unwrap();
    let_ttl_lapse().await;
    session.add_items(four_users()).await.unwrap();
    let (_, agent) = worker(&["done"]);
    let error = run_once(&agent, "continue", &session, Some(config_limit(1)))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("synthetic insert failure"),
        "outer={outer}: {error}"
    );
    let saved = saved.lock().unwrap().clone();
    assert_eq!(saved.len(), 10, "outer={outer}");
    assert_eq!(raw_rows(&path), saved, "outer={outer}");
}

#[tokio::test]
async fn failed_encrypted_suffix_transaction_preserves_original_tokens() {
    openai_agents_sdk::set_tracing_disabled(true);
    futures::future::join_all([
        failed_encrypted_transaction_case(false),
        failed_encrypted_transaction_case(true),
    ])
    .await;
}

// test_repeated_ttl_bursts_do_not_accumulate_expired_rows (sqlite x 2)
async fn ttl_bursts_case(outer: bool) {
    let endpoint = Compaction::replying(vec![]).await;
    let (_dir, _path, backend) = db();
    let (session, _) = stack(
        outer,
        backend.clone(),
        &endpoint,
        CompactionMode::Auto,
        None,
        1,
    );
    let turns = vec!["done"; 20];
    let (_, agent) = worker(&turns);
    for cycle in 0..2 {
        for _ in 0..3 {
            for _ in 0..3 {
                run_once(&agent, "below threshold", &session, None)
                    .await
                    .unwrap();
            }
            tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
        }
        assert_eq!(endpoint.requests().len(), cycle, "outer={outer}");
        session.add_items(live_items()).await.unwrap();
        run_once(&agent, "cross threshold", &session, None)
            .await
            .unwrap();
        assert_eq!(endpoint.requests().len(), cycle + 1, "outer={outer}");
        assert!(
            backend.get_items(Some(100)).await.unwrap().is_empty(),
            "outer={outer}"
        );
    }
}

#[tokio::test]
async fn repeated_ttl_bursts_do_not_accumulate_expired_rows() {
    openai_agents_sdk::set_tracing_disabled(true);
    futures::future::join_all([ttl_bursts_case(false), ttl_bursts_case(true)]).await;
}

// test_prefix_cleanup_stops_before_retained_history (sqlite x 2)
async fn prefix_cleanup_case(untrusted: bool) {
    let (_dir, path, backend) = db();
    let encrypted = key_session(backend.clone(), "synthetic-key", TTL_SECONDS);
    encrypted
        .add_items((0..40).map(|_| user("expired")).collect())
        .await
        .unwrap();
    let mut retained: Vec<InputItem> = Vec::new();
    if untrusted {
        let other = key_session(backend.clone(), "other-synthetic-key", TTL_SECONDS);
        other.add_items(vec![user("unknown key")]).await.unwrap();
        retained.extend(backend.get_items(Some(1)).await.unwrap());
    }
    let_ttl_lapse().await;
    encrypted
        .add_items(vec![user("retained live"), user("retained live")])
        .await
        .unwrap();
    retained.extend(backend.get_items(Some(2)).await.unwrap());
    encrypted.add_items(live_items()).await.unwrap();
    let endpoint = Compaction::replying(vec![]).await;
    let session = as_session(compaction_session(
        "suffix",
        encrypted.clone(),
        &endpoint,
        CompactionMode::Auto,
    ));
    let (_, agent) = worker(&["done"]);
    run_once(&agent, "cross threshold", &session, None)
        .await
        .unwrap();
    let label = format!("untrusted={untrusted}");
    assert_eq!(endpoint.requests().len(), 1, "{label}");
    assert!(
        !endpoint.requests()[0].to_string().contains("retained live"),
        "{label}"
    );
    let retained_rows: Vec<String> = retained.iter().map(|i| i.as_value().to_string()).collect();
    let _ = path;
    assert_eq!(
        backend
            .get_items(Some(100))
            .await
            .unwrap()
            .iter()
            .map(|i| i.as_value().to_string())
            .collect::<Vec<_>>(),
        retained_rows,
        "{label}"
    );
}

#[tokio::test]
async fn prefix_cleanup_stops_before_retained_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    futures::future::join_all([prefix_cleanup_case(false), prefix_cleanup_case(true)]).await;
}

// test_partial_compaction_does_not_split_tool_call_and_output
#[tokio::test]
async fn partial_compaction_does_not_split_tool_call_and_output() {
    openai_agents_sdk::set_tracing_disabled(true);
    let (_dir, _path, backend) = db();
    let history = vec![
        InputItem::from_value(
            json!({"type": "function_call", "call_id": "call", "name": "lookup", "arguments": "{}"}),
        ),
        InputItem::from_value(
            json!({"type": "function_call_output", "call_id": "call", "output": "synthetic"}),
        ),
    ];
    backend.add_items(history.clone()).await.unwrap();
    let endpoint = Compaction::replying(vec![]).await;
    let session = as_session(always(compaction_session(
        "suffix",
        backend.clone(),
        &endpoint,
        CompactionMode::Input,
    )));
    let (model, agent) = worker(&["done"]);
    let config = model_call_filter(|mut items| {
        items[0] = json!({"type": "function_call", "call_id": "call", "name": "lookup", "arguments": "{\"query\":\"model-visible\"}"});
        items
    });
    run_once(&agent, "continue", &session, Some(config))
        .await
        .unwrap();
    assert!(text(&model_input(&model, 0)).contains("model-visible"));
    assert_eq!(endpoint.requests().len(), 1);
    assert_eq!(
        endpoint.inputs()[0][0],
        json!({"role": "user", "content": "continue"})
    );
    assert_eq!(
        values(&backend.get_items(Some(100)).await.unwrap()),
        values(&history)
    );
}

// test_unverifiable_rows_inside_snapshot_are_retained (sqlite x 2 x 2)
async fn unverifiable_rows_case(outer: bool, interspersed: bool) {
    let (_dir, _path, backend) = db();
    let endpoint = Compaction::replying(vec![]).await;
    let (session, _) = stack(
        outer,
        backend.clone(),
        &endpoint,
        CompactionMode::Auto,
        Some(true),
        600,
    );
    if interspersed {
        session
            .add_items(vec![user("earlier visible history")])
            .await
            .unwrap();
    }
    let other = key_session(backend.clone(), "other-synthetic-key", 600);
    other
        .add_items(vec![user("retained under another key")])
        .await
        .unwrap();
    let retained = backend.get_items(Some(100)).await.unwrap();
    let (_, agent) = worker(&["done"]);
    run_once(&agent, "continue", &session, None).await.unwrap();
    let label = format!("outer={outer} interspersed={interspersed}");
    assert_eq!(endpoint.requests().len(), 1, "{label}");
    assert_eq!(
        backend.get_items(Some(100)).await.unwrap(),
        retained,
        "{label}"
    );
    let input = &endpoint.inputs()[0];
    assert_eq!(input.len(), 2, "{label}");
    assert_eq!(input[0], json!({"role": "user", "content": "continue"}));
    assert!(input[1].to_string().contains("done"));
}

#[tokio::test]
async fn unverifiable_rows_inside_snapshot_are_retained() {
    openai_agents_sdk::set_tracing_disabled(true);
    for outer in [false, true] {
        for interspersed in [false, true] {
            unverifiable_rows_case(outer, interspersed).await;
        }
    }
}

// test_partial_compaction_preserves_reasoning_and_its_follower (sqlite x 2 x 2)
async fn reasoning_follower_case(tool_follower: bool, filtered: bool) {
    let (_dir, _path, backend) = db();
    let mut history = vec![InputItem::from_value(
        json!({"type": "reasoning", "id": "rs_stored", "summary": []}),
    )];
    if tool_follower {
        history.push(InputItem::from_value(json!({"type": "function_call", "call_id": "call", "name": "lookup", "arguments": "{}"})));
        history.push(InputItem::from_value(
            json!({"type": "function_call_output", "call_id": "call", "output": "synthetic"}),
        ));
    } else {
        history.push(InputItem::from_value(json!({
            "id": "msg_stored", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "stored reply", "annotations": []}],
        })));
    }
    backend.add_items(history.clone()).await.unwrap();
    let summary_item = json!({"type": "compaction", "id": "cmp", "encrypted_content": "synthetic"});
    let endpoint = Compaction::replying(vec![summary_item.clone()]).await;
    let hook = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let flag = hook.clone();
    let session = as_session(
        compaction_session("suffix", backend.clone(), &endpoint, CompactionMode::Input)
            .with_should_trigger_compaction(move |_| {
                flag.load(std::sync::atomic::Ordering::SeqCst)
            }),
    );
    let (model, agent) = worker(&["done", "replayed"]);
    let config = if filtered {
        model_call_filter(|items| {
            items
                .into_iter()
                .filter(|i| i["type"] != "reasoning")
                .collect()
        })
    } else {
        config_limit(history.len() - 1)
    };
    run_once(&agent, "continue", &session, Some(config))
        .await
        .unwrap();
    let label = format!("tool={tool_follower} filtered={filtered}");
    assert!(
        model_input_values(&model, 0)
            .iter()
            .all(|i| i["type"] != "reasoning"),
        "{label}"
    );
    assert_eq!(endpoint.requests().len(), 1, "{label}");
    let input = &endpoint.inputs()[0];
    assert_eq!(input.len(), 2, "{label}");
    assert_eq!(input[0], json!({"role": "user", "content": "continue"}));
    assert!(input[1].to_string().contains("done"));
    let mut expected = values(&history);
    expected.push(summary_item);
    assert_eq!(
        values(&backend.get_items(Some(100)).await.unwrap()),
        expected,
        "{label}"
    );

    hook.store(false, std::sync::atomic::Ordering::SeqCst);
    run_once(&agent, "replay", &session, Some(config_limit(100)))
        .await
        .unwrap();
    assert_eq!(
        model_input_values(&model, 1)[..history.len()],
        values(&history)[..],
        "{label}"
    );
}

#[tokio::test]
async fn partial_compaction_preserves_reasoning_and_its_follower() {
    openai_agents_sdk::set_tracing_disabled(true);
    for tool_follower in [true, false] {
        for filtered in [false, true] {
            reasoning_follower_case(tool_follower, filtered).await;
        }
    }
}
