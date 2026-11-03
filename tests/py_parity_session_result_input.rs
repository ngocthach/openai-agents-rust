//! After a run with a client-managed session, `RunResult::input` is the input the session
//! prepared (the stored history plus the new message), as Python keeps `original_input_for_state`
//! in `RunResult.input`; `to_input_list()` then carries the whole conversation. Pinned by the
//! live parity run `session_two_turns` (`parity/scenarios/live/session_two_turns`).

use std::sync::Arc;

use openai_agents_sdk::memory::InMemorySession;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, InputItem, RunOptions, Runner, Session, handoff};
use serde_json::Value;

#[tokio::test]
async fn a_session_run_reports_the_prepared_input() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("San Francisco.")]);
    model.add_turn(vec![assistant_message("California.")]);
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Reply very concisely.")
        .with_model_instance(model);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("conversation"));

    let first = Runner::run_with_options(
        &agent,
        "What city is the Golden Gate Bridge in?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(first.input.len(), 1);
    assert_eq!(first.to_input_list().len(), 2);

    let second = Runner::run_with_options(
        &agent,
        "What state is it in?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    // The two stored items, then the new message.
    assert_eq!(second.input.len(), 3);
    assert_eq!(
        second.input[2].text_content().as_deref(),
        Some("What state is it in?")
    );
    assert_eq!(second.to_input_list().len(), 4);
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
}

/// A handoff with nested history rewrites the run input; after a session run the result
/// reports that nested history (Python's `original_input = nested.input_history`), not the
/// input the session prepared.
#[tokio::test]
async fn a_nested_handoff_after_a_session_run_reports_the_nested_history() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(delegate)),
    );
    model.add_turn(vec![
        assistant_message("triage summary"),
        function_call("transfer_to_delegate", "{}"),
    ]);
    model.add_turn(vec![assistant_message("resolution")]);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("conversation"));
    session
        .add_items(vec![
            InputItem::user("earlier question"),
            InputItem::assistant("earlier answer"),
        ])
        .await
        .unwrap();
    let mut config = RunConfig::default();
    config.nest_handoff_history = true;
    let result = Runner::run_with_options(
        triage,
        "user_message",
        RunOptions::new(()).run_config(config).session(session),
    )
    .await
    .unwrap();
    let input: Vec<Value> = result.input.iter().map(|i| i.0.clone()).collect();
    assert_eq!(input.len(), 3, "{input:?}");
    let summary = input[0]["content"].as_str().unwrap();
    assert!(summary.contains("<CONVERSATION HISTORY>"), "{summary}");
    assert!(summary.contains("earlier question"), "{summary}");
    assert!(summary.contains("user_message"), "{summary}");
    assert_eq!(result.last_agent.name, "delegate");
}
