//! Python `tests/test_agent_scoped_approvals.py`: a snapshot restored onto a graph whose
//! same-named agents can no longer be told apart must ask again instead of guessing the owner.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunState, Runner};

type Log = Arc<Mutex<Vec<String>>>;

fn make_agent(label: &str, instructions: &str, executed: &Log) -> Arc<Agent<()>> {
    let executed = executed.clone();
    let label = label.to_string();
    let tool = FunctionTool::<()>::new_no_args("operate", "", move |_ctx| {
        let executed = executed.clone();
        let label = label.clone();
        async move {
            executed.lock().unwrap().push(label.clone());
            Ok(label)
        }
    })
    .with_needs_approval(true);
    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![
        vec![function_call_with_id("operate", "{}", "pending")],
        vec![assistant_message("done")],
    ]);
    Arc::new(
        Agent::<()>::new("same")
            .with_instructions(instructions)
            .with_tool(tool)
            .with_model_instance(model),
    )
}

/// `test_tied_duplicate_owner_restore_requires_reapproval`.
#[tokio::test]
async fn a_restore_onto_tied_duplicate_owners_asks_again() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for tied_at_write in [false, true] {
        for always in [false, true] {
            let ctx = format!("tied_at_write={tied_at_write} always={always}");
            let executed: Log = Arc::default();
            let first = make_agent(
                "first",
                if tied_at_write { "same" } else { "first" },
                &executed,
            );
            let second = make_agent(
                "second",
                if tied_at_write { "same" } else { "second" },
                &executed,
            );
            let root_model = Arc::new(FakeModel::new());
            root_model.add_turns(vec![vec![function_call_with_id(
                "to_first", "{}", "handoff",
            )]]);
            let root = Arc::new(
                Agent::<()>::new("root")
                    .with_handoff(Handoff::new(first.clone()).with_tool_name("to_first"))
                    .with_handoff(Handoff::new(second.clone()).with_tool_name("to_second"))
                    .with_model_instance(root_model),
            );
            let result = Runner::run(&root, "start").await.unwrap();
            assert_eq!(result.interruptions.len(), 1, "{ctx}");
            let mut state = result.to_state();
            state.approve(&result.interruptions[0], always);
            let payload = state.to_json_string().unwrap();

            // Identical visible signatures cannot tell these different closures apart.
            let restored_first = make_agent("restored-first", "same", &executed);
            let restored_second = make_agent("restored-second", "same", &executed);
            let root_model = Arc::new(FakeModel::new());
            root_model.add_turns(vec![vec![function_call_with_id(
                "to_first", "{}", "handoff",
            )]]);
            let configured = Arc::new(
                Agent::<()>::new("root")
                    .with_handoff(Handoff::new(restored_second).with_tool_name("to_second"))
                    .with_handoff(Handoff::new(restored_first).with_tool_name("to_first"))
                    .with_model_instance(root_model),
            );
            let restored = RunState::from_json_str(&payload).unwrap();
            let resumed = Runner::run(&configured, restored).await.unwrap();
            assert_eq!(resumed.interruptions.len(), 1, "{ctx}");
            assert!(executed.lock().unwrap().is_empty(), "{ctx}");
        }
    }
}
