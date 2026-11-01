//! Port of `test_run_sync_errors_when_loop_already_running` (`tests/test_agent_runner_sync.py`):
//! the blocking runner refuses to start inside a running async runtime.

use std::sync::Arc;

use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, Runner};

#[tokio::test]
async fn run_sync_errors_inside_a_running_runtime() {
    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![vec![assistant_message("done")]]);
    let agent = Agent::<()>::new("test-agent").with_model_instance(model.clone());

    let error = Runner::run_sync(&agent, "input").expect_err("a runtime is already running");

    assert!(error.to_string().contains("run_sync"), "{error}");
    // The model was never asked: the refusal comes before any work.
    assert!(model.calls().is_empty());
}

#[test]
fn run_sync_runs_outside_a_runtime() {
    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![vec![assistant_message("done")]]);
    let agent = Agent::<()>::new("test-agent").with_model_instance(model);

    let result = Runner::run_sync(&agent, "input").expect("run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}
