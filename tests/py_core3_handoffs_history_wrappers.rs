//! Port of `test_wrapper_reset_does_not_change_nested_history_ownership` from the Python SDK's
//! `tests/test_handoff_history_duplication.py`.
//!
//! The history wrapper markers are process-global, so this case lives in its own test binary.

use std::sync::Arc;

use openai_agents_sdk::handoffs::{
    reset_conversation_history_wrappers, set_conversation_history_wrappers,
};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, InputItem, Runner};

/// Replay ownership must not depend on wrappers that are current after the run. The markers are
/// changed for the run and restored afterwards, and the handoff message is still replayed once.
#[tokio::test]
async fn wrapper_reset_does_not_change_nested_history_ownership() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let first_model = Arc::new(FakeModel::new());
    let second_model = Arc::new(FakeModel::new());
    second_model.add_turn(vec![assistant_message("done")]);
    let second = Arc::new(Agent::<()>::new("second").with_model_instance(second_model.clone()));
    let first = Agent::<()>::new("first")
        .with_model_instance(first_model.clone())
        .with_handoff(second);
    first_model.add_turn(vec![
        assistant_message("handoff message"),
        function_call("transfer_to_second", "{}"),
    ]);

    set_conversation_history_wrappers(Some("<<START>>"), Some("<<END>>"));
    let result = Runner::run_with_options(
        Arc::new(first),
        "start",
        RunOptions::new(()).run_config(RunConfig::<()>::default().with_nest_handoff_history(true)),
    )
    .await;
    reset_conversation_history_wrappers();
    let result = result.unwrap();

    let replayed = result
        .to_input_list()
        .iter()
        .filter(|i: &&InputItem| i.text_content().as_deref() == Some("handoff message"))
        .count();
    assert_eq!(replayed, 1);
}
