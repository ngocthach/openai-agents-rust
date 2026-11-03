//! `test_deferred_sticky_decision_preserves_mirrored_per_call_binding` of
//! `test_tool_approval_call_id_reuse.py`: a deferred top-level tool (called under a namespace
//! equal to its own name) keeps the per-call binding of an earlier call id after a later sticky
//! decision about the same tool, so reusing that call id with other arguments is refused.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, function_call_full};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, Runner};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct ValueArgs {
    value: i64,
}

fn lookup_call(call_id: &str, value: i64) -> openai_agents_sdk::items::OutputItem {
    function_call_full(
        "lookup",
        format!(r#"{{"value":{value}}}"#),
        call_id,
        None,
        Some("lookup".into()),
    )
}

#[tokio::test]
async fn deferred_sticky_decision_preserves_mirrored_per_call_binding() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for approve in [true, false] {
        let executed = Arc::new(Mutex::new(Vec::<i64>::new()));
        let recorded = executed.clone();
        let lookup = FunctionTool::new("lookup", "lookup", move |_c, a: ValueArgs| {
            let recorded = recorded.clone();
            async move {
                recorded.lock().unwrap().push(a.value);
                Ok(format!("looked up {}", a.value))
            }
        })
        .with_needs_approval(true)
        .with_defer_loading(true);

        let model = Arc::new(ScriptedModel::new());
        for output in [
            lookup_call("call_0", 1),
            lookup_call("call_1", 2),
            lookup_call("call_0", 3),
        ] {
            model.enqueue(ScriptedStep::output(vec![output])).unwrap();
        }
        let agent = Agent::<()>::new("agent")
            .with_tool(lookup)
            .with_model_instance(model);

        let first = Runner::run(&agent, "go").await.unwrap();
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], false);
        let second = Runner::run(&agent, state).await.unwrap();
        let mut state = second.to_state();
        if approve {
            state.approve(&second.interruptions[0], true);
        } else {
            state.reject(&second.interruptions[0], true, None);
        }
        let err = Runner::run(&agent, state).await.unwrap_err();
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. })
                && err.to_string().contains("unique call ID"),
            "{err:?}"
        );
        let expected: &[i64] = if approve { &[1, 2] } else { &[1] };
        assert_eq!(*executed.lock().unwrap(), expected, "approve={approve}");
    }
}
