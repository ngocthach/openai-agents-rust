//! Ports of `tests/test_tool_custom_data.py` and `tests/test_tool_use_tracker.py`.
//!
//! Custom data is observed on the run items and the replayed input. The tool-use tracker is
//! observed through `RunResult::to_state().tool_use_tracker` after real runs, since
//! `AgentToolUseTracker` is internal to the run loop.

use std::sync::Arc;

use openai_agents_sdk::items::{OutputItem, RunInput};
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::ToolSearchTool;
use openai_agents_sdk::tool::{CustomTool, FunctionTool, tool_namespace};
use openai_agents_sdk::{Agent, Result, RunItemKind, Runner};
use serde_json::{Value, json};

/// Scripted model with the given turns (each a list of output items).
fn scripted(turns: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(turns.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

/// The input items sent to the model on its last request, as JSON.
fn last_input(model: &ScriptedModel) -> Vec<Value> {
    model
        .last_call()
        .expect("a model call")
        .request
        .input
        .iter()
        .map(|i| i.as_value().clone())
        .collect()
}

fn no_key_anywhere(items: &[Value], key: &str) -> bool {
    items.iter().all(|item| item.get(key).is_none())
}

// ---------------------------------------------------------------------------
// test_tool_custom_data.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn function_tool_custom_data_is_attached_but_not_replayed() {
    // Python's extractor also mutates `ctx.raw_item` to show it is not replayed; Rust hands the
    // extractor an immutable `&Value`, so only the attached data and the replay are checked here.
    let tool = FunctionTool::<()>::new_no_args("get_data", "d", |_ctx| async { Ok("tool_result") })
        .with_custom_data_extractor(|ctx| {
            let call_id = ctx.raw_item["call_id"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let output = ctx.output.to_model_string();
            async move { Some(json!({"call_id": call_id, "output": output})) }
        });

    let model = scripted(vec![
        vec![
            assistant_message("call tool"),
            function_call_full("get_data", "{}", "call-1", None, None),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let result = Runner::run(agent, "user").await.unwrap();

    let output = result
        .new_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .expect("a tool output item");
    assert_eq!(
        output.custom_data(),
        Some(&json!({"call_id": "call-1", "output": "tool_result"}))
    );
    let replay = output.to_input_item();
    assert!(replay.as_value().get("custom_data").is_none());
    assert!(replay.as_value().get("renderer").is_none());
    assert!(no_key_anywhere(&last_input(&model), "custom_data"));
}

#[tokio::test]
async fn custom_tool_custom_data_is_attached() {
    let tool = CustomTool::<()>::new(
        "raw_editor",
        "Edit raw text.",
        |_ctx, input: String| async move { Ok(input.to_uppercase()) },
    )
    .with_format(json!({"type": "text"}))
    .with_custom_data_extractor(|ctx| {
        let input = ctx.input.to_string();
        let output = ctx.output.to_string();
        async move { Some(json!({"input": input, "output": output})) }
    });

    let model = scripted(vec![
        vec![OutputItem::Other(json!({
            "type": "custom_tool_call",
            "name": "raw_editor",
            "call_id": "call_custom",
            "input": "hello",
        }))],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("custom-agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let result = Runner::run(agent, "user").await.unwrap();

    let output = result
        .new_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .expect("a tool output item");
    assert_eq!(
        output.custom_data(),
        Some(&json!({"input": "hello", "output": "HELLO"}))
    );
    let raw = output.to_input_item();
    assert!(raw.as_value().get("custom_data").is_none());
    assert!(raw.as_value().get("renderer").is_none());
}

// Covered by `tests/core_audit_fixes_tools.rs`:
// - `computer_tool_custom_data_reaches_the_output_item` (custom data `{"source": ...}` and the
//   computer call's screenshot output reach the extractor and the item);
// - `apply_patch_custom_data_carries_the_operations_and_status` (operations and status reach the
//   extractor and the item).

// ---------------------------------------------------------------------------
// test_tool_use_tracker.py
// ---------------------------------------------------------------------------

fn tracker_of(
    result: &openai_agents_sdk::RunResult<()>,
) -> std::collections::BTreeMap<String, Vec<String>> {
    result.to_state().tool_use_tracker
}

fn noop_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args(name, "d", |_ctx| async { Ok("ok") })
}

#[tokio::test]
async fn tool_use_tracker_snapshot_is_sorted_by_tool_name() {
    // Python: `add_tool_use(agent, ["beta", "alpha"])` serializes as `["alpha", "beta"]`.
    let model = scripted(vec![
        vec![
            function_call_full("beta", "{}", "call-b", None, None),
            function_call_full("alpha", "{}", "call-a", None, None),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("runtime-agent")
        .with_tool(noop_tool("beta"))
        .with_tool(noop_tool("alpha"))
        .with_model_instance(model);
    let result = Runner::run(agent, "go").await.unwrap();

    assert_eq!(
        tracker_of(&result)["runtime-agent"],
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[tokio::test]
async fn tool_use_tracker_records_trace_names_for_namespaced_and_deferred_functions() {
    // Python: a namespaced tool is recorded as `billing.lookup_account`, a deferred top-level
    // tool called under its synthetic namespace as the bare `get_weather`.
    let billing = tool_namespace(
        "billing",
        "Billing tools",
        vec![noop_tool("lookup_account")],
    )
    .unwrap()
    .remove(0);
    let deferred = noop_tool("get_weather").with_defer_loading(true);

    let model = scripted(vec![
        vec![
            function_call_full(
                "lookup_account",
                "{}",
                "call-1",
                None,
                Some("billing".into()),
            ),
            function_call_full(
                "get_weather",
                "{}",
                "call-2",
                None,
                Some("get_weather".into()),
            ),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("tracked-agent")
        .with_tool(billing)
        .with_tool(deferred)
        .with_tool(ToolSearchTool::new())
        .with_model_instance(model);
    let result = Runner::run(agent, "go").await.unwrap();

    assert_eq!(
        tracker_of(&result)["tracked-agent"],
        vec![
            "billing.lookup_account".to_string(),
            "get_weather".to_string()
        ]
    );
}

#[tokio::test]
async fn a_function_named_tool_search_is_recorded_as_a_function() {
    let model = scripted(vec![
        vec![function_call_full(
            "tool_search",
            "{}",
            "call-ts",
            None,
            None,
        )],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("tracked-agent")
        .with_tool(noop_tool("tool_search"))
        .with_model_instance(model);
    let result = Runner::run(agent, "go").await.unwrap();

    assert_eq!(
        tracker_of(&result)["tracked-agent"],
        vec!["tool_search".to_string()]
    );
}

#[tokio::test]
async fn hosted_tool_search_does_not_reset_a_required_tool_choice() {
    // Python `record_processed_response` ignores hosted tool search, so `tool_choice="required"`
    // survives it. The reset happens only after a real function call.
    let search_call = OutputItem::from_value(json!({
        "id": "tsc_1", "type": "tool_search_call",
        "arguments": {"paths": [], "query": "lookup"}, "execution": "server",
        "status": "completed",
    }));
    let search_output = OutputItem::from_value(json!({
        "id": "tso_1", "type": "tool_search_output",
        "execution": "server", "status": "completed", "tools": [],
    }));
    let model = scripted(vec![
        vec![search_call, search_output],
        vec![function_call_full("lookup", "{}", "call-l", None, None)],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("tool-search-agent")
        .with_tool(noop_tool("lookup"))
        .with_tool(ToolSearchTool::new())
        .with_model_settings(ModelSettings {
            tool_choice: Some(ToolChoice::Named("required".into())),
            ..Default::default()
        })
        .with_model_instance(model.clone());
    Runner::run(agent, "go").await.unwrap();

    let calls = model.calls();
    assert_eq!(calls.len(), 3);
    let choice = |i: usize| calls[i].request.model_settings.tool_choice.clone();
    assert_eq!(choice(1), Some(ToolChoice::Named("required".into())));
    assert_ne!(choice(2), Some(ToolChoice::Named("required".into())));
}

#[tokio::test]
async fn a_resumed_snapshot_is_hydrated_and_serialized_sorted() {
    // Python `from_serializable` / `hydrate_tool_use_tracker`: a snapshot restored into a run
    // comes back out, sorted, through the next snapshot.
    // The first run ends with "done"; the resumed run gets its own final turn.
    let model = scripted(vec![
        vec![function_call_full("tool-1", "{}", "call-1", None, None)],
        vec![assistant_message("done")],
        vec![assistant_message("resumed")],
    ]);
    let agent = Agent::<()>::new("agent")
        .with_tool(noop_tool("tool-1"))
        .with_tool(noop_tool("tool-2"))
        .with_model_instance(model);
    let mut state: RunState = Runner::run(agent.clone(), "go").await.unwrap().to_state();
    state.tool_use_tracker = [(
        "agent".to_string(),
        vec!["tool-2".to_string(), "tool-1".to_string()],
    )]
    .into_iter()
    .collect();
    state
        .tool_use_tracker
        .insert("missing-agent".into(), vec!["missing".into()]);

    let resumed: Result<_> =
        Runner::run_with_options(agent, RunInput::State(Box::new(state)), Default::default()).await;
    let snapshot = resumed.unwrap().to_state().tool_use_tracker;

    assert_eq!(
        snapshot["agent"],
        vec!["tool-1".to_string(), "tool-2".to_string()]
    );
    assert!(!snapshot.contains_key("missing-agent"));
}
