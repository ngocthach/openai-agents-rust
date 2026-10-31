//! Parity port of the central scenario of `tests/memory/test_compaction_model_visibility.py`:
//! automatic compaction must retain history the model was not shown (a tool output hidden by a
//! `call_model_input_filter` or `session_input_callback`) until a run replays it in full.

use std::sync::Arc;

use openai_agents_sdk::agent::ToolUseBehavior;
use openai_agents_sdk::memory::{OpenAIResponsesCompactionSession, Session};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::run_config::ModelInputData;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunConfig, RunOptions, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{ListSession, mock_openai};

const LOCAL_OUTPUT: &str = "synthetic-local-only-result";

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn lookup_tool() -> FunctionTool<()> {
    FunctionTool::new("lookup", "lookup", |_ctx, _a: NoArgs| async move {
        Ok(LOCAL_OUTPUT.to_string())
    })
}

fn without_tools(items: Vec<InputItem>) -> Vec<InputItem> {
    items
        .into_iter()
        .filter(|i| {
            !matches!(
                i.item_type(),
                Some("function_call") | Some("function_call_output")
            )
        })
        .collect()
}

fn text(items: &[InputItem]) -> String {
    items
        .iter()
        .map(|i| i.as_value().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

async fn scenario(filter_kind: &str) {
    openai_agents_sdk::set_tracing_disabled(true);
    let summary = json!({
        "type": "compaction", "id": "cmp-test", "encrypted_content": "synthetic-summary"
    });
    let (base, captured) = mock_openai(vec![json!({
        "id": "cmp_1",
        "output": [summary.clone()],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    })])
    .await;
    let underlying = Arc::new(ListSession::new("visibility"));
    let session = Arc::new(
        OpenAIResponsesCompactionSession::new("visibility", underlying.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(base))
            .with_should_trigger_compaction(|_| true),
    );

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id("lookup", "{}", "call_1")]);
    model.add_turn(vec![assistant_message("ok")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("worker")
        .with_model_instance(model.clone())
        .with_model_settings(ModelSettings::default().with_store(false))
        .with_tool(lookup_tool())
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);

    Runner::run_with_options(
        &agent,
        "first",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert!(
        captured.lock().unwrap().is_empty(),
        "a run that executed a local tool defers compaction"
    );

    let config = match filter_kind {
        "session" => RunConfig::<()>::default().with_session_input_callback(
            |history: Vec<InputItem>, new_input: Vec<InputItem>| async move {
                let mut out = without_tools(history);
                out.extend(new_input);
                Ok(out)
            },
        ),
        "redact" => RunConfig::<()>::default().with_call_model_input_filter(|data| async move {
            let input = data
                .model_data
                .input
                .into_iter()
                .map(|item| {
                    if item.item_type() == Some("function_call_output") {
                        let mut v: Value = item.as_value().clone();
                        v["output"] = json!("redacted");
                        InputItem::from_value(v)
                    } else {
                        item
                    }
                })
                .collect();
            Ok(ModelInputData {
                input,
                instructions: data.model_data.instructions,
            })
        }),
        _ => RunConfig::<()>::default().with_call_model_input_filter(|data| async move {
            Ok(ModelInputData {
                input: without_tools(data.model_data.input),
                instructions: data.model_data.instructions,
            })
        }),
    };
    Runner::run_with_options(
        &agent,
        "second",
        RunOptions::new(())
            .session(session.clone())
            .run_config(config),
    )
    .await
    .unwrap();

    let calls = model.calls();
    assert!(
        !text(&calls[1].input).contains(LOCAL_OUTPUT),
        "{filter_kind}: the filtered output must not reach the model"
    );
    assert!(
        captured.lock().unwrap().is_empty(),
        "{filter_kind}: history the model did not see must not be compacted"
    );
    assert!(text(&session.get_items(None).await.unwrap()).contains(LOCAL_OUTPUT));

    // A run that replays the history in full compacts it.
    Runner::run_with_options(
        &agent,
        "third",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert!(text(&model.calls()[2].input).contains(LOCAL_OUTPUT));
    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "{filter_kind}");
    assert!(requests[0].body["input"].to_string().contains(LOCAL_OUTPUT));
    assert_eq!(session.get_items(None).await.unwrap().len(), 1);
    assert_eq!(
        session.get_items(None).await.unwrap()[0].item_type(),
        Some("compaction")
    );
}

/// Compaction records the input the model actually saw (after `call_model_input_filter`), so
/// history the filter hid is retained until a run replays it in full.
#[tokio::test]
async fn compaction_retains_output_hidden_by_a_model_input_filter() {
    scenario("model").await;
}

#[tokio::test]
async fn compaction_retains_output_hidden_by_a_session_input_callback() {
    scenario("session").await;
}

/// Compaction records the input the model actually saw (after `call_model_input_filter`), so
/// history the filter hid is retained until a run replays it in full.
#[tokio::test]
async fn compaction_retains_output_redacted_by_a_model_input_filter() {
    scenario("redact").await;
}
