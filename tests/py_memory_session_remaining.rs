//! Remaining parity ports of `tests/memory/test_session.py`: the `run_sync` / `run_streamed`
//! variants of the parametrized runner tests that `py_memory_session.rs` only ports for `run`.
//!
//! Python-only (no Rust counterpart, grouped):
//! - `_await_mutation` cancellation leak test and the three `post_commit_cancellation` tests:
//!   asyncio task cancellation semantics; a dropped Rust future cannot be re-awaited.
//! - `close` / worker-thread connection tests (`close_closes_worker_thread_connections`,
//!   `reaps_connections_from_exited_worker_threads`, `closes_retired_connections...`,
//!   `keeps_connections_of_live_worker_threads`, `closed_rejects_empty_add_items`,
//!   `rollback_failure_evicts_connection`, `close_retries_quarantined_connection`): Python opens
//!   a connection per executor thread and tracks them; Rust keeps one `Arc<Mutex<Connection>>`
//!   that closes on drop, so there is nothing to reap, evict or quarantine.
//! - `file_lock_is_shared_across_instances`: a Python process-local lock registry; Rust relies on
//!   SQLite's own file locking (`two_sessions_on_one_file_write_independently` covers the
//!   observable behaviour).
//! - `session_settings_default` / `constructor_normalizes_dictionary` /
//!   `rejects_unknown_dictionary_fields`: Python's dict coercion; Rust has typed settings.

#![cfg(feature = "sqlite")]

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::memory::{Session, SessionSettings, SqliteSession};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};

mod py_mcpmem_support;
use py_mcpmem_support::item_text;

#[derive(Clone, Copy, Debug)]
enum Method {
    Run,
    RunSync,
    RunStreamed,
}

const METHODS: [Method; 3] = [Method::Run, Method::RunSync, Method::RunStreamed];

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(serde_json::json!({"role": "assistant", "content": text}))
}

fn texts(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

/// Python's `run_agent_async`: run `input` with the given runner method and return the final
/// output text.
async fn run_agent(
    method: Method,
    agent: &Arc<Agent<()>>,
    input: impl Into<openai_agents_sdk::RunInput> + Send + 'static,
    options: RunOptions<()>,
) -> Option<String> {
    match method {
        Method::Run => Runner::run_with_options(agent, input, options)
            .await
            .unwrap()
            .final_output_text(),
        Method::RunSync => {
            let agent = agent.clone();
            // `run_sync` refuses to run inside a runtime, and blocking-pool threads still carry
            // the runtime context, so it gets a plain OS thread of its own.
            tokio::task::spawn_blocking(move || {
                std::thread::spawn(move || {
                    Runner::run_sync_with_options(&agent, input, options)
                        .unwrap()
                        .final_output_text()
                })
                .join()
                .unwrap()
            })
            .await
            .unwrap()
        }
        Method::RunStreamed => {
            let mut result = Runner::run_streamed_with_options(agent, input, options);
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(event) = events.next().await {
                    event.unwrap();
                }
            }
            result.final_output_text()
        }
    }
}

fn fixtures() -> (Arc<FakeModel>, Arc<Agent<()>>) {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    (model, agent)
}

fn opts(session: &Arc<SqliteSession>) -> RunOptions<()> {
    RunOptions::new(()).session(session.clone())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explicit_session_instance_keeps_history_for_every_runner_method() {
    for method in METHODS {
        let dir = tempfile::tempdir().unwrap();
        let session = Arc::new(SqliteSession::open("explicit", dir.path().join("m.db")).unwrap());
        let (model, agent) = fixtures();

        model.add_turn(vec![assistant_message("Hello")]);
        let out = run_agent(method, &agent, "Hi there", opts(&session)).await;
        assert_eq!(out.as_deref(), Some("Hello"), "{method:?}");

        model.add_turn(vec![assistant_message("I remember you said hi")]);
        let out = run_agent(
            method,
            &agent,
            "Do you remember what I said?",
            opts(&session),
        )
        .await;
        assert_eq!(out.as_deref(), Some("I remember you said hi"), "{method:?}");
        let seen = texts(&model.last_call().unwrap().input);
        assert!(seen.iter().any(|t| t == "Hi there"), "{method:?}: {seen:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disabled_memory_sends_only_the_new_message_for_every_runner_method() {
    for method in METHODS {
        let (model, agent) = fixtures();

        model.add_turn(vec![assistant_message("Hello")]);
        let out = run_agent(method, &agent, "Hi there", RunOptions::new(())).await;
        assert_eq!(out.as_deref(), Some("Hello"), "{method:?}");

        model.add_turn(vec![assistant_message("I don't remember")]);
        let out = run_agent(
            method,
            &agent,
            "Do you remember what I said?",
            RunOptions::new(()),
        )
        .await;
        assert_eq!(out.as_deref(), Some("I don't remember"), "{method:?}");
        assert_eq!(model.last_call().unwrap().input.len(), 1, "{method:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn different_sessions_stay_separate_for_every_runner_method() {
    for method in METHODS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.db");
        let session_1 = Arc::new(SqliteSession::open("session_1", &path).unwrap());
        let session_2 = Arc::new(SqliteSession::open("session_2", &path).unwrap());
        let (model, agent) = fixtures();

        model.add_turn(vec![assistant_message("I like cats")]);
        run_agent(method, &agent, "I like cats", opts(&session_1)).await;
        model.add_turn(vec![assistant_message("I like dogs")]);
        run_agent(method, &agent, "I like dogs", opts(&session_2)).await;

        model.add_turn(vec![assistant_message("Yes, you mentioned cats")]);
        let out = run_agent(method, &agent, "What did I say I like?", opts(&session_1)).await;
        assert_eq!(
            out.as_deref(),
            Some("Yes, you mentioned cats"),
            "{method:?}"
        );

        let seen = texts(&model.last_call().unwrap().input);
        assert!(
            seen.iter().any(|t| t == "I like cats"),
            "{method:?}: {seen:?}"
        );
        assert!(
            !seen.iter().any(|t| t == "I like dogs"),
            "{method:?}: {seen:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_input_is_appended_to_history_for_every_runner_method() {
    for method in METHODS {
        let dir = tempfile::tempdir().unwrap();
        let session = Arc::new(SqliteSession::open("validation", dir.path().join("m.db")).unwrap());
        session
            .add_items(vec![user("Earlier message"), assistant("Saved reply")])
            .await
            .unwrap();
        let (model, agent) = fixtures();

        model.add_turn(vec![assistant_message("This should run")]);
        run_agent(method, &agent, vec![user("Test message")], opts(&session)).await;

        assert_eq!(
            texts(&model.last_call().unwrap().input),
            vec!["Earlier message", "Saved reply", "Test message"],
            "{method:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_input_callback_prepares_input_for_every_runner_method() {
    for method in METHODS {
        let dir = tempfile::tempdir().unwrap();
        let session = Arc::new(SqliteSession::open("callback", dir.path().join("m.db")).unwrap());
        session
            .add_items(vec![
                user("Hello there."),
                assistant("Hi, I'm here to assist you."),
            ])
            .await
            .unwrap();
        let (model, agent) = fixtures();
        model.add_turn(vec![assistant_message("I'm gpt-4o")]);

        let config = RunConfig::<()>::default().with_session_input_callback(
            |history: Vec<InputItem>, new_input: Vec<InputItem>| async move {
                let mut out: Vec<InputItem> = history
                    .into_iter()
                    .filter(|i| i.as_value().get("role").and_then(|r| r.as_str()) == Some("user"))
                    .collect();
                out.extend(new_input);
                Ok(out)
            },
        );
        run_agent(
            method,
            &agent,
            vec![user("What your name?")],
            opts(&session).run_config(config),
        )
        .await;

        assert_eq!(
            texts(&model.last_call().unwrap().input),
            vec!["Hello there.", "What your name?"],
            "{method:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_repeating_callback_does_not_grow_the_session_for_every_runner_method() {
    for method in METHODS {
        let dir = tempfile::tempdir().unwrap();
        let session = Arc::new(SqliteSession::open("repeat", dir.path().join("m.db")).unwrap());
        let (model, agent) = fixtures();

        for turn in 0..3 {
            model.add_turn(vec![assistant_message(format!("assistant {turn}"))]);
            let config = RunConfig::<()>::default().with_session_input_callback(
                |history: Vec<InputItem>, new_input: Vec<InputItem>| async move {
                    if history.is_empty() {
                        return Ok(new_input);
                    }
                    let mut out = history.clone();
                    out.push(history[0].clone());
                    out.extend(new_input);
                    Ok(out)
                },
            );
            run_agent(
                method,
                &agent,
                format!("user {turn}"),
                opts(&session).run_config(config),
            )
            .await;
        }

        let stored = session.get_items(None).await.unwrap();
        let users: Vec<String> = stored
            .iter()
            .filter(|i| i.as_value().get("role").and_then(|r| r.as_str()) == Some("user"))
            .map(item_text)
            .collect();
        assert_eq!(users, vec!["user 0", "user 1", "user 2"], "{method:?}");
        assert_eq!(stored.len(), 6, "{method:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_runner_settings_override_wins_for_every_runner_method() {
    for method in METHODS {
        let dir = tempfile::tempdir().unwrap();
        let session = Arc::new(
            SqliteSession::open("override", dir.path().join("m.db"))
                .unwrap()
                .with_settings(SessionSettings::with_limit(100)),
        );
        session
            .add_items((0..10).map(|i| user(&format!("Turn {i}"))).collect())
            .await
            .unwrap();
        let (model, agent) = fixtures();
        model.add_turn(vec![assistant_message("Got it")]);

        run_agent(
            method,
            &agent,
            "New question",
            opts(&session).run_config(
                RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(2)),
            ),
        )
        .await;

        let seen = texts(&model.last_call().unwrap().input);
        let history: Vec<&String> = seen.iter().filter(|t| *t != "New question").collect();
        assert_eq!(history.len(), 2, "{method:?}: {seen:?}");
    }
}

#[test]
fn session_settings_constructor_stores_the_limit() {
    let session = SqliteSession::in_memory("constructor_settings_test")
        .unwrap()
        .with_settings(SessionSettings::with_limit(5));
    assert_eq!(session.session_settings().unwrap().limit, Some(5));
    // A zero limit is a real setting, not "unset".
    let zero = SqliteSession::in_memory("zero")
        .unwrap()
        .with_settings(SessionSettings::with_limit(0));
    assert_eq!(zero.session_settings().unwrap().limit, Some(0));
}
