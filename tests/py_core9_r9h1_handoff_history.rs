//! Ports of the nested-history replay cases of the Python SDK's
//! `tests/test_handoff_history_duplication.py` that the earlier audits classed as
//! implementation-private: the released schema 1.12 snapshot layout and the partial loss of an
//! ownership ledger. Both are asserted through the replayed input.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::{
    InputItem, MessageOutput, OutputItem, RunInput, RunItem, RunItemKind,
};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn nested() -> RunConfig<()> {
    RunConfig::<()>::default().with_nest_handoff_history(true)
}

fn transfer(target: &str) -> OutputItem {
    function_call(format!("transfer_to_{target}"), "{}")
}

fn item_text(item: &InputItem) -> String {
    item.text_content().unwrap_or_default()
}

fn count(items: &[InputItem], t: &str) -> usize {
    items.iter().filter(|i| item_text(i) == t).count()
}

async fn run_mode(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> Result<RunResult<()>, AgentsError> {
    if !streamed {
        return Runner::run_with_options(agent.clone(), input, options).await;
    }
    let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream.into_result().expect("finished")),
    }
}

/// Python: `test_first_nested_handoff_after_restore_uses_explicit_occurrence_lineage`, for the
/// current snapshot (the tool search items are replayed once) and the released 1.12 layout
/// without an ownership ledger, which cannot prove which items an earlier handoff forwarded and
/// so replays them twice.
#[tokio::test]
async fn first_nested_handoff_after_restore_uses_explicit_occurrence_lineage() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    for streamed in [false, true] {
        for legacy_snapshot in [false, true] {
            let label = format!("streamed={streamed} legacy={legacy_snapshot}");
            let approval = FunctionTool::new(
                "approval_tool",
                "needs approval",
                |_ctx, _a: NoArgs| async { Ok("approved".to_string()) },
            )
            .with_needs_approval(true);
            let first_model = setup();
            let second_model = setup();
            second_model.add_turn(vec![assistant_message("done")]);
            let second =
                Arc::new(Agent::<()>::new("second").with_model_instance(second_model.clone()));
            let first = Arc::new(
                Agent::<()>::new("first")
                    .with_model_instance(first_model.clone())
                    .with_tool(approval)
                    .with_handoff(second),
            );
            first_model.add_turn(vec![
                OutputItem::from_value(json!({
                    "type": "tool_search_call", "id": "tool_search_call", "call_id": "search",
                    "arguments": {"query": "profile"}, "execution": "server",
                    "status": "completed",
                })),
                OutputItem::from_value(json!({
                    "type": "tool_search_output", "id": "tool_search_output", "call_id": "search",
                    "execution": "server", "status": "completed", "tools": [],
                })),
                function_call_with_id("approval_tool", "{}", "approval"),
            ]);
            first_model.add_turn(vec![transfer("second")]);

            let interrupted = run_mode(
                &first,
                "start",
                RunOptions::new(()).run_config(nested()),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(interrupted.interruptions.len(), 1, "{label}");
            let mut state = interrupted.to_state();
            state.approve(&interrupted.interruptions[0], false);
            let mut wire = state.to_json().unwrap();
            if legacy_snapshot {
                // The released 1.12 layout: per-call approvals live in `approvals`, and the
                // ownership ledger and generated/session index map do not exist yet.
                let context = wire["context"].as_object_mut().unwrap();
                let entries = context
                    .remove("function_tool_approvals")
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();
                for entry in entries {
                    let key = entry["tool_key"].as_str().unwrap().to_string();
                    context["approvals"][key] = entry["decision"].clone();
                }
                wire["$schemaVersion"] = json!("1.12");
                let wire = wire.as_object_mut().unwrap();
                wire.remove("nested_history_owned_session_item_refs");
                wire.remove("generated_session_item_indexes");
            }
            let restored = RunState::from_json(wire).unwrap();

            let resumed = run_mode(
                &first,
                restored,
                RunOptions::new(()).run_config(nested()),
                streamed,
            )
            .await
            .unwrap();
            let replay = resumed.to_input_list();
            let expected = if legacy_snapshot { 2 } else { 1 };
            for kind in ["tool_search_call", "tool_search_output"] {
                let found = replay.iter().filter(|i| i.0["type"] == kind).count();
                assert_eq!(found, expected, "{label}: {kind}");
            }
        }
    }
}

/// Python: `test_nested_history_session_resolution_keeps_surviving_owned_items`. Dropping one
/// owned `new_items` entry must not discard the ownership of another: the surviving message is
/// still replayed once, and the removed one is still present once, through the input that
/// forwarded it.
#[tokio::test]
async fn removing_one_owned_new_item_keeps_the_others_owned() {
    for streamed in [false, true] {
        let first_model = setup();
        let second_model = setup();
        second_model.add_turn(vec![assistant_message("done")]);
        let second = Arc::new(Agent::<()>::new("second").with_model_instance(second_model));
        first_model.add_turn(vec![
            assistant_message("kept"),
            assistant_message("removed"),
            transfer("second"),
        ]);
        let first = Arc::new(
            Agent::<()>::new("first")
                .with_model_instance(first_model)
                .with_handoff(handoff(second)),
        );
        let mut result = run_mode(
            &first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        let replay = result.to_input_list();
        assert_eq!(count(&replay, "kept"), 1, "streamed={streamed}");
        assert_eq!(count(&replay, "removed"), 1, "streamed={streamed}");

        result.new_items.retain(|item: &RunItem| {
            !matches!(
                &item.kind,
                RunItemKind::MessageOutput { raw_item: MessageOutput { .. } }
                    if item.to_input_item().text_content().as_deref() == Some("removed")
            )
        });
        let replay = result.to_input_list();
        assert_eq!(count(&replay, "kept"), 1, "streamed={streamed}");
        assert_eq!(count(&replay, "removed"), 1, "streamed={streamed}");
    }
}
