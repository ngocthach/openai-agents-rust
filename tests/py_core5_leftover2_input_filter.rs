//! Port of the model-input deduplication cases of `test_call_model_input_filter.py`: the input a
//! `call_model_input_filter` returns is deduplicated (latest value wins) before the model sees
//! it, without moving a repeated call behind its output.
//!
//! The first two Python cases (sync/async filter replacing input and instructions) are covered
//! by `py_core3_runstatectx_context.rs`; the invalid-return-type case cannot exist because the
//! Rust filter returns `ModelInputData`. The span-error case is in
//! `py_core5_leftover2_spans.rs`.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{ModelInputData, RunConfig};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, Runner};
use serde_json::{Value, json};

type Items = Vec<Value>;

/// Run one turn with a filter that appends `extra` to the model input, in blocking and streamed
/// mode, and return the input the model received for each mode.
async fn sent_inputs(extra: Items, prepend: bool) -> Vec<Vec<InputItem>> {
    let mut sent = Vec::new();
    for streamed in [false, true] {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("ok")]);
        let agent = Agent::<()>::new("test").with_model_instance(model.clone());
        let extra = extra.clone();
        let config = RunConfig::<()>::new().with_call_model_input_filter(move |data| {
            let extra: Vec<InputItem> = extra.iter().cloned().map(InputItem).collect();
            async move {
                let input = if prepend {
                    extra.into_iter().chain(data.model_data.input).collect()
                } else {
                    data.model_data.input.into_iter().chain(extra).collect()
                };
                Ok(ModelInputData {
                    input,
                    instructions: data.model_data.instructions,
                })
            }
        });
        let options = RunOptions::new(()).run_config(config);
        if streamed {
            let mut run = Runner::run_streamed_with_options(agent, "start", options);
            let events = run.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        } else {
            Runner::run_with_options(&agent, "start", options)
                .await
                .unwrap();
        }
        sent.push(model.calls().last().unwrap().input.clone());
    }
    sent
}

fn types(input: &[InputItem]) -> Vec<String> {
    input
        .iter()
        .filter_map(|item| item.0.get("type").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Python: `test_call_model_input_filter_prefers_latest_duplicate_outputs_non_streamed` and the
/// streamed twin.
#[tokio::test]
async fn the_latest_duplicate_output_wins() {
    for (call_id, input) in [("dup-call", 0), ("dup-call-stream", 1)] {
        let sent = sent_inputs(
            vec![
                json!({"type": "function_call_output", "call_id": call_id, "output": "old-value"}),
                json!({"type": "function_call_output", "call_id": call_id, "output": "new-value"}),
            ],
            true,
        )
        .await;
        let outputs: Vec<&InputItem> = sent[input]
            .iter()
            .filter(|item| item.0["type"] == "function_call_output" && item.0["call_id"] == call_id)
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0["output"], "new-value");
    }
}

fn call(call_id: &str) -> Value {
    json!({"type": "function_call", "call_id": call_id, "name": "tool_ordered", "arguments": "{}"})
}

fn output(call_id: &str, value: &str) -> Value {
    json!({"type": "function_call_output", "call_id": call_id, "output": value})
}

/// Python: `test_call_model_input_filter_keeps_duplicate_item_order_non_streamed` and the
/// streamed twin. Collapsing the repeated call must not move it behind its output.
#[tokio::test]
async fn a_repeated_call_stays_ahead_of_its_output() {
    let sent = sent_inputs(
        vec![
            call("ordered-call"),
            output("ordered-call", "result"),
            call("ordered-call"),
        ],
        false,
    )
    .await;
    for input in &sent {
        assert_eq!(types(input), ["function_call", "function_call_output"]);
    }
}

/// Python: `test_call_model_input_filter_keeps_duplicate_output_order_non_streamed` and the
/// streamed twin.
#[tokio::test]
async fn a_repeated_output_keeps_the_latest_value_after_its_call() {
    let sent = sent_inputs(
        vec![
            output("ordered-output", "old"),
            call("ordered-output"),
            output("ordered-output", "new"),
        ],
        false,
    )
    .await;
    for input in &sent {
        assert_eq!(types(input), ["function_call", "function_call_output"]);
        assert_eq!(input.last().unwrap().0["output"], "new");
    }
}

/// Python: `test_call_model_input_filter_keeps_reasoning_before_required_follower`.
#[tokio::test]
async fn a_repeated_reasoning_item_stays_before_its_follower() {
    let sent = sent_inputs(
        vec![
            json!({"type": "reasoning", "id": "ordered-reasoning",
                   "summary": [{"type": "summary_text", "text": "old"}]}),
            json!({"type": "function_call", "call_id": "reasoning-call",
                   "name": "tool_ordered", "arguments": "{}"}),
            json!({"type": "reasoning", "id": "ordered-reasoning",
                   "summary": [{"type": "summary_text", "text": "new"}]}),
        ],
        false,
    )
    .await;
    for input in &sent {
        assert_eq!(types(input), ["reasoning", "function_call"]);
        let reasoning: Vec<&InputItem> = input
            .iter()
            .filter(|item| item.0["type"] == "reasoning")
            .collect();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(
            reasoning[0].0["summary"],
            json!([{"type": "summary_text", "text": "new"}])
        );
    }
}
