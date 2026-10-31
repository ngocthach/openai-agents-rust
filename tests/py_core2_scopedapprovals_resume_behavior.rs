//! Port of `tests/test_resume_tool_use_behavior.py`: resuming a turn after an approval keeps
//! the tool results that already completed and feeds them to the agent's tool-use behaviour.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::items::RunInput;
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunResult, RunState, Runner};
use serde_json::{Value, json};

async fn run(agent: &Agent<()>, input: impl Into<RunInput>, streamed: bool) -> RunResult<()> {
    if streamed {
        let mut stream = Runner::run_streamed(agent, input);
        stream.finish().await.unwrap();
        stream.into_result().unwrap()
    } else {
        Runner::run(agent, input).await.unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
enum Behavior {
    StopAt,
    First,
    Custom,
}

async fn scenario(gated_first: bool, streamed: bool, serialized: bool, reject: bool, b: Behavior) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));

    let e = effects.clone();
    let submit_order = FunctionTool::<()>::new_no_args("submit_order", "", move |_ctx| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push("submit");
            Ok("order submitted".to_string())
        }
    });
    let e = effects.clone();
    let notify_manager = FunctionTool::<()>::new_no_args("notify_manager", "", move |_ctx| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push("notify");
            Ok("manager notified".to_string())
        }
    })
    .with_needs_approval(true);

    let mut calls = vec![
        function_call_with_id("submit_order", "{}", "submit"),
        function_call_with_id("notify_manager", "{}", "notify"),
    ];
    if gated_first {
        calls.reverse();
    }
    let model = Arc::new(FakeModel::new());
    model.add_turn(calls);

    let s = seen.clone();
    let behavior = match b {
        Behavior::StopAt => ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["submit_order".into()],
        }),
        Behavior::First => ToolUseBehavior::StopOnFirstTool,
        Behavior::Custom => ToolUseBehavior::Custom(Arc::new(move |_ctx, results| {
            let s = s.clone();
            Box::pin(async move {
                s.lock()
                    .unwrap()
                    .extend(results.iter().map(|r| r.tool_name.clone()));
                let first = match &results[0].output {
                    openai_agents_sdk::tool::ToolOutput::Text(t) => json!(t),
                    other => json!(format!("{other:?}")),
                };
                Ok(ToolsToFinalOutputResult::final_output(first))
            })
        })),
    };
    let agent = Agent::<()>::new("orders")
        .with_model_instance(model.clone())
        .with_tool(submit_order)
        .with_tool(notify_manager)
        .with_tool_use_behavior(behavior);

    let result = run(&agent, "submit an order", streamed).await;
    assert_eq!(*effects.lock().unwrap(), ["submit"]);
    let mut state = result.to_state();
    if serialized {
        state = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    }
    for interruption in state.get_interruptions() {
        if reject {
            state.reject(&interruption, false, Some("notification rejected".into()));
        } else {
            state.approve(&interruption, false);
        }
    }
    let result = run(&agent, state, streamed).await;

    let mut expected = "order submitted";
    if gated_first && !matches!(b, Behavior::StopAt) {
        expected = if reject {
            "notification rejected"
        } else {
            "manager notified"
        };
    }
    let ctx = format!(
        "gated_first={gated_first} streamed={streamed} serialized={serialized} reject={reject} {b:?}"
    );
    assert_eq!(result.final_output, Some(Value::from(expected)), "{ctx}");
    assert_eq!(model.calls().len(), 1, "{ctx}");
    let want: &[&str] = if reject {
        &["submit"]
    } else {
        &["submit", "notify"]
    };
    assert_eq!(*effects.lock().unwrap(), want, "{ctx}");
    if matches!(b, Behavior::Custom) {
        let want = if gated_first {
            ["notify_manager", "submit_order"]
        } else {
            ["submit_order", "notify_manager"]
        };
        assert_eq!(*seen.lock().unwrap(), want, "{ctx}");
    }
}

/// Python: `test_resume_preserves_completed_tool_results` (48 parametrizations).
#[tokio::test]
async fn resume_preserves_completed_tool_results() {
    for b in [Behavior::StopAt, Behavior::First, Behavior::Custom] {
        for reject in [false, true] {
            for serialized in [false, true] {
                for streamed in [false, true] {
                    for gated_first in [false, true] {
                        scenario(gated_first, streamed, serialized, reject, b).await;
                    }
                }
            }
        }
    }
}
