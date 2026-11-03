//! Port of `test_sandbox_agents_as_tools_example_serializes_structured_reviews` of
//! `tests/test_example_workflows.py`. The example module only supplies the scenario; the SDK
//! behaviour under test is that agents with structured output types, exposed as tools with a
//! custom output extractor, return the extractor's string verbatim as their tool output, in
//! call order, next to an ordinary function tool called in the same turn.

use std::collections::BTreeMap;
use std::sync::Arc;

use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct PricingPacketReview {
    requested_discount_percent: i64,
    requested_term_months: i64,
    pricing_risk: String,
    summary: String,
    recommended_next_step: String,
    evidence_files: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct RolloutRiskReview {
    rollout_risk: String,
    summary: String,
    blockers: Vec<String>,
    recommended_next_step: String,
    evidence_files: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct DiscountArgs {
    discount_percent: i64,
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model
        .extend(steps.into_iter().map(ScriptedStep::output).collect())
        .unwrap();
    model
}

fn call(name: &str, args: Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

/// The example's extractor: the structured final output as JSON with sorted keys.
fn sorted_json(value: &Value) -> String {
    let sorted: BTreeMap<String, Value> = value
        .as_object()
        .expect("structured output")
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    serde_json::to_string(&sorted).unwrap()
}

fn options(name: &str, description: &str) -> AgentAsToolOptions<()> {
    AgentAsToolOptions::new()
        .tool_name(name)
        .tool_description(description)
        .custom_output_extractor(|result| async move {
            Ok(sorted_json(
                result.final_output.as_ref().expect("final output"),
            ))
        })
}

/// Python: `test_sandbox_agents_as_tools_example_serializes_structured_reviews`.
#[tokio::test]
async fn structured_reviews_reach_the_orchestrator_through_the_extractor_in_call_order() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let pricing = json!({
        "requested_discount_percent": 15,
        "requested_term_months": 24,
        "pricing_risk": "medium",
        "summary": "Discount ask is above target band.",
        "recommended_next_step": "Trade discount for a stronger give-get.",
        "evidence_files": ["pricing_summary.md", "commercial_notes.md"],
    });
    let rollout = json!({
        "rollout_risk": "medium",
        "summary": "Launch timing is compressed.",
        "blockers": [
            "Regional admin training is incomplete.",
            "SSO migration lands in week 2.",
        ],
        "recommended_next_step": "Require a phased rollout plan.",
        "evidence_files": ["rollout_plan.md", "support_history.md"],
    });
    let pricing_model = scripted(vec![vec![assistant_message(pricing.to_string())]]);
    let rollout_model = scripted(vec![vec![assistant_message(rollout.to_string())]]);
    let orchestrator_model = scripted(vec![
        vec![
            call(
                "review_pricing_packet",
                json!({"input": "Review pricing"}),
                "outer_pricing",
            ),
            call(
                "review_rollout_risk",
                json!({"input": "Review rollout"}),
                "outer_rollout",
            ),
            call(
                "get_discount_approval_rule",
                json!({"discount_percent": 15}),
                "outer_approval",
            ),
        ],
        vec![assistant_message("Recommendation complete")],
    ]);

    let approval_rule = FunctionTool::<()>::new(
        "get_discount_approval_rule",
        "",
        |_c, args: DiscountArgs| async move {
            Ok(match args.discount_percent {
                p if p <= 10 => "AE",
                p if p <= 15 => "RSD",
                _ => "Finance + RSD",
            }
            .to_string())
        },
    );
    let pricing_agent = Agent::<()>::new("pricing")
        .with_model_instance(pricing_model)
        .with_output_type::<PricingPacketReview>();
    let rollout_agent = Agent::<()>::new("rollout")
        .with_model_instance(rollout_model)
        .with_output_type::<RolloutRiskReview>();
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_model_instance(orchestrator_model.clone())
        .with_tool(pricing_agent.as_tool_with(options("review_pricing_packet", "Pricing review")))
        .with_tool(rollout_agent.as_tool_with(options("review_rollout_risk", "Rollout review")))
        .with_tool(approval_rule)
        .with_model_settings(ModelSettings {
            tool_choice: Some(ToolChoice::required()),
            ..Default::default()
        });

    let result = Runner::run(&orchestrator, "Review the renewal")
        .await
        .unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("Recommendation complete")
    );

    let last = orchestrator_model
        .calls()
        .last()
        .cloned()
        .expect("second call");
    let outputs: Vec<Value> = last
        .request
        .input
        .iter()
        .map(|item| serde_json::to_value(item).unwrap())
        .filter(|v| v["type"] == "function_call_output")
        .map(|v| json!({"call_id": v["call_id"], "output": v["output"]}))
        .collect();
    assert_eq!(
        outputs,
        vec![
            json!({"call_id": "outer_pricing", "output": sorted_json(&pricing)}),
            json!({"call_id": "outer_rollout", "output": sorted_json(&rollout)}),
            json!({"call_id": "outer_approval", "output": "RSD"}),
        ]
    );
}
