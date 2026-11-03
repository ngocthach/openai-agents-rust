//! After a run with a client-managed session, `RunResult::input` is the input the session
//! prepared (the stored history plus the new message), as Python keeps `original_input_for_state`
//! in `RunResult.input`; `to_input_list()` then carries the whole conversation. Pinned by the
//! live parity run `session_two_turns` (`parity/scenarios/live/session_two_turns`).

use std::sync::Arc;

use openai_agents_sdk::memory::InMemorySession;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, RunOptions, Runner, Session};

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
