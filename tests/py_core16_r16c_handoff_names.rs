//! Ports of `tests/test_handoff_tool.py`:
//! `test_handoff_derived_name_collision_allows_explicit_override` and
//! `test_handoffs_ignore_disabled_derived_name_collision`. Python reads `get_handoffs`; Rust
//! observes the handoff tools the model is offered.

use std::sync::Arc;

use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::run_config::{RunConfig, ToolNameCollisionPolicy};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, RunOptions, Runner};

/// The handoff tool names the model was offered by a one-turn run under the `error` policy.
async fn offered_handoffs(handoffs: Vec<Handoff<()>>) -> Vec<String> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_handoffs(handoffs);
    let config =
        RunConfig::<()>::default().with_tool_name_collision_policy(ToolNameCollisionPolicy::Error);
    Runner::run_with_options(&triage, "hi", RunOptions::new(()).run_config(config))
        .await
        .expect("no collision");
    model
        .last_call()
        .unwrap()
        .request
        .handoffs
        .iter()
        .map(|h| h.name.clone())
        .collect()
}

#[tokio::test]
async fn an_explicit_tool_name_resolves_a_derived_name_collision() {
    // "Billing Agent" and "billing agent" derive the same tool name.
    let billing = Handoff::new(Arc::new(Agent::<()>::new("Billing Agent")));
    let normalized = Handoff::new(Arc::new(Agent::<()>::new("billing agent")))
        .with_tool_name("transfer_to_normalized_billing");
    assert_eq!(
        offered_handoffs(vec![billing, normalized]).await,
        vec![
            "transfer_to_billing_agent",
            "transfer_to_normalized_billing"
        ]
    );
}

#[tokio::test]
async fn a_disabled_handoff_does_not_take_part_in_a_derived_name_collision() {
    let billing = Handoff::new(Arc::new(Agent::<()>::new("Billing Agent")));
    let normalized =
        Handoff::new(Arc::new(Agent::<()>::new("billing agent"))).with_is_enabled(false);
    assert_eq!(
        offered_handoffs(vec![billing, normalized]).await,
        vec!["transfer_to_billing_agent"]
    );
}
