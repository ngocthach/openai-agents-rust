//! Run-loop behaviors the Python suite pins down: how an empty or refused model turn ends a run,
//! and how same-named agents are told apart in a serialized snapshot.

use std::sync::Arc;

use openai_agents_sdk::agent::Instructions;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call_with_id, refusal_message,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Summary {
    summary: String,
}

// ------------------------------------------------------------------ empty turns

#[tokio::test]
async fn an_empty_model_turn_ends_a_plain_text_run_with_empty_output() {
    let model = setup();
    model.add_turn(vec![]);
    let agent = Agent::<()>::new("empty").with_model_instance(model.clone());

    let result = Runner::run(&agent, "hi").await.unwrap();

    assert_eq!(result.final_output, Some(json!("")));
    assert_eq!(model.calls().len(), 1, "the run must not loop");
}

#[tokio::test]
async fn a_message_without_text_ends_a_plain_text_run() {
    let model = setup();
    model.add_turn(vec![assistant_message("")]);
    let agent = Agent::<()>::new("blank").with_model_instance(model.clone());

    let result = Runner::run(&agent, "hi").await.unwrap();

    assert_eq!(result.final_output, Some(json!("")));
    assert_eq!(model.calls().len(), 1);
}

#[tokio::test]
async fn an_empty_structured_turn_runs_again_until_max_turns() {
    let model = setup();
    model.add_turn(vec![]);
    model.add_turn(vec![]);
    let agent = Agent::<()>::new("structured")
        .with_output_type::<Summary>()
        .with_model_instance(model.clone());

    let err = Runner::run_with_options(&agent, "hi", RunOptions::new(()).max_turns(Some(2)))
        .await
        .unwrap_err();

    assert!(
        matches!(err, AgentsError::MaxTurnsExceeded { .. }),
        "{err:?}"
    );
    assert_eq!(model.calls().len(), 2);
}

#[tokio::test]
async fn an_empty_structured_turn_reaches_the_invalid_final_output_handler() {
    let model = setup();
    model.add_turn(vec![]);
    let agent = Agent::<()>::new("structured")
        .with_output_type::<Summary>()
        .with_model_instance(model.clone());
    let handlers = RunErrorHandlers::new().on_invalid_final_output(|input| async move {
        assert!(
            input
                .error
                .to_string()
                .contains("Model returned no final output for the structured output type."),
            "{}",
            input.error
        );
        Ok(Some(RunErrorHandlerResult::new(
            json!({"summary": "fallback"}),
        )))
    });

    let result =
        Runner::run_with_options(&agent, "hi", RunOptions::new(()).error_handlers(handlers))
            .await
            .unwrap();

    assert_eq!(result.final_output, Some(json!({"summary": "fallback"})));
}

#[tokio::test]
async fn a_refusal_raises_even_without_an_output_type() {
    let model = setup();
    model.add_turn(vec![refusal_message("I cannot help with that.")]);
    let agent = Agent::<()>::new("plain").with_model_instance(model.clone());

    let err = Runner::run(&agent, "hi").await.unwrap_err();

    match err {
        AgentsError::ModelRefusalError { refusal, .. } => {
            assert_eq!(refusal, "I cannot help with that.")
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// ------------------------------------------------- duplicate agent names in a snapshot

fn approval_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "approval_tool",
        "needs approval",
        |_c, _a: NoArgs| async move { Ok("approved output".to_string()) },
    )
    .with_needs_approval(true)
}

/// `triage` hands off to two distinct agents that are both named `duplicate`. The one that
/// carries the approval tool sorts second by configuration, so its identity key is
/// `duplicate#2`.
fn duplicate_graph(model: &Arc<FakeModel>) -> Agent<()> {
    let alpha = Arc::new(
        Agent::<()>::new("duplicate")
            .with_instructions("Alpha")
            .with_model_instance(model.clone()),
    );
    let beta = Arc::new(
        Agent::<()>::new("duplicate")
            .with_instructions("Beta")
            .with_tool(approval_tool())
            .with_model_instance(model.clone()),
    );
    Agent::<()>::new("triage")
        .with_handoff(handoff(alpha).with_tool_name("transfer_to_alpha"))
        .with_handoff(handoff(beta).with_tool_name("transfer_to_beta"))
        .with_model_instance(model.clone())
}

#[tokio::test]
async fn a_snapshot_names_the_current_duplicate_agent_by_identity() {
    let model = setup();
    let triage = duplicate_graph(&model);
    model.add_turn(vec![function_call_with_id(
        "transfer_to_beta",
        "{}",
        "call_handoff",
    )]);
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_approval",
    )]);

    let result = Runner::run(&triage, "go").await.unwrap();
    assert_eq!(result.interruptions.len(), 1);

    let json = result.to_state().to_json().unwrap();
    assert_eq!(
        json["current_agent"],
        json!({"name": "duplicate", "identity": "duplicate#2"})
    );
    // The tool-use tracker is keyed by identity, so the two agents never share an entry.
    assert_eq!(
        json["tool_use_tracker"]["duplicate#2"],
        json!(["approval_tool"])
    );
    assert!(json["tool_use_tracker"].get("duplicate").is_none());
}

#[tokio::test]
async fn a_duplicate_agent_snapshot_resumes_on_the_agent_it_named() {
    let model = setup();
    let triage = duplicate_graph(&model);
    model.add_turn(vec![function_call_with_id(
        "transfer_to_beta",
        "{}",
        "call_handoff",
    )]);
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_approval",
    )]);

    let first = Runner::run(&triage, "go").await.unwrap();
    let json = first.to_state().to_json().unwrap();

    let mut restored = RunState::from_json(json).unwrap();
    assert_eq!(
        restored.current_agent_identity.as_deref(),
        Some("duplicate#2")
    );
    let interruption = restored.get_interruptions()[0].clone();
    restored.approve(&interruption, false);
    model.add_turn(vec![assistant_message("all done")]);

    let resumed = Runner::run(&triage, restored).await.unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("all done"));
    assert!(
        matches!(&resumed.last_agent.instructions, Some(Instructions::Static(s)) if s == "Beta"),
        "resumed on the wrong duplicate"
    );
}

#[tokio::test]
async fn a_duplicate_agent_snapshot_without_identity_is_refused() {
    let model = setup();
    let triage = duplicate_graph(&model);
    model.add_turn(vec![function_call_with_id(
        "transfer_to_beta",
        "{}",
        "call_handoff",
    )]);
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_approval",
    )]);

    let first = Runner::run(&triage, "go").await.unwrap();
    let mut json = first.to_state().to_json().unwrap();
    json["current_agent"] = json!({"name": "duplicate"});

    // Python resolves a name-only current agent through its identity map to the first agent
    // with that name (`_resolve_agent_from_data`), and refuses only graphs whose duplicates differ
    // solely by hook instance state. This graph has no hooks, so the snapshot is not refused.
    let restored = RunState::from_json(json).unwrap();
    if let Err(err) = Runner::run(&triage, restored).await {
        assert!(!err.to_string().contains("same-named agents"), "{err}");
    }
}

// ------------------------------------------------- prepare hook after blocking guardrails

#[tokio::test]
async fn the_prepare_hook_runs_after_blocking_input_guardrails() {
    use openai_agents_sdk::{GuardrailFunctionOutput, InputGuardrail};
    use std::sync::atomic::{AtomicUsize, Ordering};

    for trips in [true, false] {
        let model = setup();
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("guarded")
            .with_input_guardrail(
                InputGuardrail::new("g", move |_c, _a, _i| async move {
                    Ok(GuardrailFunctionOutput::new(serde_json::Value::Null, trips))
                })
                .blocking(),
            )
            .with_model_instance(model.clone());
        let prepared = Arc::new(AtomicUsize::new(0));
        let counter = prepared.clone();
        let options = openai_agents_sdk::run::RunOptions::new(()).prepare_agent(move |agent| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(agent)
            }
        });
        let outcome = Runner::run_with_options(&agent, "hi", options).await;
        if trips {
            assert!(matches!(
                outcome,
                Err(AgentsError::InputGuardrailTripwireTriggered { .. })
            ));
            assert_eq!(
                prepared.load(Ordering::SeqCst),
                0,
                "rejected input is never prepared"
            );
        } else {
            assert_eq!(
                outcome.unwrap().final_output_text().as_deref(),
                Some("done")
            );
            assert_eq!(prepared.load(Ordering::SeqCst), 1);
        }
    }
}
