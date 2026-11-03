//! Parity port of `tests/memory/test_session_limit.py` across all three Runner entry points
//! (`run`, `run_sync`, `run_streamed`): a `SessionSettings` limit bounds the history the model
//! sees, and a limit boundary never passes an output whose call it cut off.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "sqlite")]

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::memory::{Session, SessionSettings, SqliteSession};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::{Value, json};

const METHODS: [&str; 3] = ["run", "run_sync", "run_streamed"];

async fn run_agent(
    method: &str,
    agent: &Arc<Agent<()>>,
    input: &str,
    session: &Arc<SqliteSession>,
    limit: Option<Option<usize>>,
) {
    let mut options = RunOptions::new(()).session(session.clone());
    if let Some(limit) = limit {
        let settings = match limit {
            Some(limit) => SessionSettings::with_limit(limit),
            None => SessionSettings::default(),
        };
        options = options.run_config(RunConfig::<()>::default().with_session_settings(settings));
    }
    match method {
        "run" => {
            Runner::run_with_options(agent.clone(), input, options)
                .await
                .unwrap();
        }
        "run_sync" => {
            // `run_sync` refuses to run inside a runtime, so it gets a plain thread.
            let (agent, input) = (agent.clone(), input.to_string());
            let (done, finished) = tokio::sync::oneshot::channel();
            std::thread::spawn(move || {
                let outcome = Runner::run_sync_with_options(agent, input.as_str(), options);
                let _ = done.send(outcome.map(|_| ()));
            });
            finished.await.unwrap().unwrap();
        }
        _ => {
            let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
            let events = stream.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
    }
}

fn setup(
    name: &str,
) -> (
    tempfile::TempDir,
    Arc<SqliteSession>,
    Arc<FakeModel>,
    Arc<Agent<()>>,
) {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(SqliteSession::open(name, dir.path().join("history.db")).unwrap());
    let model = Arc::new(FakeModel::new());
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    (dir, session, model, agent)
}

fn last_input(model: &FakeModel) -> Vec<Value> {
    model
        .last_call()
        .unwrap()
        .input
        .iter()
        .map(|i| i.as_value().clone())
        .collect()
}

async fn turn(
    method: &str,
    agent: &Arc<Agent<()>>,
    model: &FakeModel,
    session: &Arc<SqliteSession>,
    n: usize,
    limit: Option<Option<usize>>,
) {
    model.add_turn(vec![assistant_message(format!("Reply {n}"))]);
    run_agent(method, agent, &format!("Message {n}"), session, limit).await;
}

#[tokio::test]
async fn session_limit_parameter() {
    for method in METHODS {
        let (_dir, session, model, agent) = setup("limit_test");
        for n in 1..=3 {
            turn(method, &agent, &model, &session, n, None).await;
        }
        assert_eq!(session.get_items(None).await.unwrap().len(), 6, "{method}");
        turn(method, &agent, &model, &session, 4, Some(Some(2))).await;
        let input = last_input(&model);
        assert_eq!(input.len(), 3, "{method}");
        assert_eq!(input[0]["content"], "Message 3", "{method}");
        assert_eq!(input[1]["content"][0]["text"], "Reply 3", "{method}");
        assert_eq!(input[2]["content"], "Message 4", "{method}");
    }
}

#[tokio::test]
async fn session_limit_drops_unmatched_history_function_call_output() {
    for method in METHODS {
        let (_dir, session, model, agent) = setup("limit_tool_pair");
        let history: Vec<Value> = vec![
            json!({"role": "user", "content": "What is the weather?"}),
            json!({"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "sunny"}),
            json!({"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "It is sunny.", "annotations": []}]}),
        ];
        session
            .add_items(history.iter().cloned().map(InputItem::from_value).collect())
            .await
            .unwrap();
        let tail: Vec<Value> = session
            .get_items(Some(2))
            .await
            .unwrap()
            .iter()
            .map(|i| i.as_value().clone())
            .collect();
        assert_eq!(tail, history[2..], "{method}");

        model.add_turn(vec![assistant_message("Tomorrow is sunny too.")]);
        run_agent(
            method,
            &agent,
            "What about tomorrow?",
            &session,
            Some(Some(2)),
        )
        .await;
        assert_eq!(
            last_input(&model),
            vec![
                history[3].clone(),
                json!({"role": "user", "content": "What about tomorrow?"})
            ],
            "{method}"
        );
    }
}

#[tokio::test]
async fn session_limit_zero() {
    for method in METHODS {
        let (_dir, session, model, agent) = setup("limit_zero_test");
        turn(method, &agent, &model, &session, 1, None).await;
        turn(method, &agent, &model, &session, 2, None).await;
        turn(method, &agent, &model, &session, 3, Some(Some(0))).await;
        let input = last_input(&model);
        assert_eq!(input.len(), 1, "{method}");
        assert_eq!(input[0]["content"], "Message 3", "{method}");
    }
}

#[tokio::test]
async fn session_limit_none_gets_all_history() {
    for method in METHODS {
        let (_dir, session, model, agent) = setup("limit_none_test");
        for n in 1..=5 {
            turn(method, &agent, &model, &session, n, None).await;
        }
        assert_eq!(session.get_items(None).await.unwrap().len(), 10, "{method}");
        turn(method, &agent, &model, &session, 6, Some(None)).await;
        let input = last_input(&model);
        assert_eq!(input.len(), 11, "{method}");
        assert_eq!(input[0]["content"], "Message 1", "{method}");
        assert_eq!(input[10]["content"], "Message 6", "{method}");
    }
}

#[tokio::test]
async fn session_limit_larger_than_history() {
    for method in METHODS {
        let (_dir, session, model, agent) = setup("limit_large_test");
        turn(method, &agent, &model, &session, 1, None).await;
        turn(method, &agent, &model, &session, 2, Some(Some(100))).await;
        let input = last_input(&model);
        assert_eq!(input.len(), 3, "{method}");
        assert_eq!(input[0]["content"], "Message 1", "{method}");
        assert_eq!(input[1]["content"][0]["text"], "Reply 1", "{method}");
        assert_eq!(input[2]["content"], "Message 2", "{method}");
    }
}
