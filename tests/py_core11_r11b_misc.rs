//! Parity ports of cases the earlier audits filed as Python-only although Rust has the
//! behaviour: `tests/memory/test_session.py::test_session_settings_constructor` and
//! `tests/test_agent_runner_streamed.py::test_streamed_omit_policy_strips_reasoning_ids_already_stored_in_the_session`.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{ReasoningItemIdPolicy, RunConfig};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, AgentsError, Runner, handoff};
use serde_json::{Value, json};

fn reasoning(id: &str) -> OutputItem {
    OutputItem::from_value(json!({"type": "reasoning", "id": id, "summary": []}))
}

fn find_reasoning(items: &[Value]) -> Option<Value> {
    items
        .iter()
        .find(|i| i.get("type").and_then(Value::as_str) == Some("reasoning"))
        .cloned()
}

async fn drain(
    agent: Arc<Agent<()>>,
    input: &str,
    options: RunOptions<()>,
) -> Result<openai_agents_sdk::result::RunResult<()>, AgentsError> {
    let mut stream = Runner::run_streamed_with_options(agent, input, options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream
            .into_result()
            .expect("a finished stream has a result")),
    }
}

/// `test_streamed_omit_policy_strips_reasoning_ids_already_stored_in_the_session`
#[tokio::test]
async fn the_streamed_omit_policy_strips_reasoning_ids_already_in_the_session() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let specialist = Arc::new(Agent::<()>::new("specialist").with_model_instance(model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(specialist)),
    );
    let session = Arc::new(InMemorySession::new("issue-2020-streamed"));
    model.add_turn(vec![
        reasoning("rs_triage"),
        openai_agents_sdk::testing::function_call("transfer_to_specialist", "{}"),
    ]);
    model.add_turn(vec![assistant_message("handled")]);

    // The first turn predates the mitigation, so the session records the reasoning id.
    let first = drain(
        triage.clone(),
        "hello",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(first.final_output_text().as_deref(), Some("handled"));
    let stored: Vec<Value> = session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|i| i.0)
        .collect();
    assert_eq!(find_reasoning(&stored).unwrap()["id"], "rs_triage");

    // The server no longer resolves the stored id: replaying it would fail.
    model.add_responder(|request| {
        if request.input.iter().any(|i| {
            i.item_type() == Some("reasoning") && i.0.get("id") == Some(&json!("rs_triage"))
        }) {
            return Err(AgentsError::other("Item with id 'rs_triage' not found."));
        }
        Ok(vec![assistant_message("done")])
    });
    let config = RunConfig::<()> {
        reasoning_item_id_policy: ReasoningItemIdPolicy::Omit,
        ..Default::default()
    };
    let second = drain(
        triage,
        "anything else?",
        RunOptions::new(()).session(session).run_config(config),
    )
    .await
    .unwrap();

    assert_eq!(second.final_output_text().as_deref(), Some("done"));
    let replayed: Vec<Value> = model
        .calls()
        .last()
        .unwrap()
        .input
        .iter()
        .map(|i| i.0.clone())
        .collect();
    assert!(find_reasoning(&replayed).unwrap().get("id").is_none());
}

/// `test_session_settings_constructor`
#[cfg(feature = "sqlite")]
#[test]
fn a_sqlite_session_keeps_the_settings_it_was_built_with() {
    use openai_agents_sdk::memory::{SessionSettings, SqliteSession};

    let session = SqliteSession::in_memory("constructor_settings_test")
        .unwrap()
        .with_settings(SessionSettings::with_limit(5));

    assert_eq!(
        session
            .session_settings()
            .and_then(|settings| settings.limit),
        Some(5)
    );
}
