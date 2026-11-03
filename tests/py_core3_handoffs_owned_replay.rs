//! Ports of the Python SDK's `tests/test_handoff_history_duplication.py` cases about ownership of
//! forwarded history: items a nested handoff already carried into the next agent's input must
//! not be replayed a second time, whether the history came from the runner, from a handoff input
//! filter that calls `nest_handoff_history`, or from a later filter that keeps the earlier
//! history. Also covers multi-turn nesting and caller edits to `result.new_items` / `result.input`.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::handoffs::{Handoff, HandoffInputData, handoff, nest_handoff_history};
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

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn text(t: &str) -> OutputItem {
    assistant_message(t)
}

fn transfer(target: &str) -> OutputItem {
    function_call(format!("transfer_to_{target}"), "{}")
}

fn nested() -> RunConfig<()> {
    RunConfig::<()>::default().with_nest_handoff_history(true)
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

/// `first` says `message` and hands off to `second`, which answers `done`.
struct Chain {
    first: Arc<Agent<()>>,
}

fn chain(message: &str, configure: impl FnOnce(Handoff<()>) -> Handoff<()>) -> Chain {
    let first_model = setup();
    let second_model = setup();
    second_model.add_turn(vec![text("done")]);
    let second = Arc::new(Agent::<()>::new("second").with_model_instance(second_model.clone()));
    first_model.add_turn(vec![text(message), transfer("second")]);
    let first = Arc::new(
        Agent::<()>::new("first")
            .with_model_instance(first_model.clone())
            .with_handoff(configure(handoff(second))),
    );
    Chain { first }
}

async fn run_nested(chain: &Chain, streamed: bool) -> RunResult<()> {
    run_mode(
        &chain.first,
        "start",
        RunOptions::new(()).run_config(nested()),
        streamed,
    )
    .await
    .unwrap()
}

/// Python: `test_nested_handoff_history_preserves_identical_messages_across_turns`. Equal
/// messages from different turns are different occurrences, so each one is replayed.
#[tokio::test]
async fn identical_messages_across_turns_are_all_kept() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    for streamed in [false, true] {
        let continue_work =
            FunctionTool::new("continue_work", "continue", |_ctx, _a: NoArgs| async {
                Ok("continue".to_string())
            });
        let first_model = setup();
        let second_model = setup();
        let final_model = setup();
        final_model.add_turn(vec![text("same")]);
        let final_agent =
            Arc::new(Agent::<()>::new("final").with_model_instance(final_model.clone()));
        second_model.add_turn(vec![text("same"), function_call("continue_work", "{}")]);
        second_model.add_turn(vec![text("same"), transfer("final")]);
        let second = Arc::new(
            Agent::<()>::new("second")
                .with_model_instance(second_model.clone())
                .with_tool(continue_work)
                .with_handoff(final_agent),
        );
        let first = Arc::new(
            Agent::<()>::new("first")
                .with_model_instance(first_model.clone())
                .with_handoff(second),
        );
        first_model.add_turn(vec![text("same"), transfer("second")]);

        let result = run_mode(
            &first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        let final_input = final_model.calls().last().unwrap().input.clone();
        assert_eq!(
            item_text(&final_input[0]).matches("same").count(),
            2,
            "streamed={streamed}"
        );
        assert_eq!(
            count(&result.to_input_list(), "same"),
            4,
            "streamed={streamed}"
        );
    }
}

/// Python: `test_nested_handoff_history_partition_survives_interruption_resume`.
#[tokio::test]
async fn nested_history_partition_survives_an_interruption_and_restore() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    let approval = FunctionTool::new(
        "approval_tool",
        "needs approval",
        |_ctx, _a: NoArgs| async { Ok("approved".to_string()) },
    )
    .with_needs_approval(true);
    let first_model = setup();
    let second_model = setup();
    let final_model = setup();
    final_model.add_turn(vec![text("done")]);
    let final_agent = Arc::new(Agent::<()>::new("final").with_model_instance(final_model.clone()));
    let second = Arc::new(
        Agent::<()>::new("second")
            .with_model_instance(second_model.clone())
            .with_tool(approval)
            .with_handoff(final_agent),
    );
    let first = Arc::new(
        Agent::<()>::new("first")
            .with_model_instance(first_model.clone())
            .with_handoff(second),
    );
    first_model.add_turn(vec![text("once"), transfer("second")]);
    second_model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "approval",
    )]);
    second_model.add_turn(vec![text("once"), transfer("final")]);

    let config = nested();
    let interrupted = run_mode(
        &first,
        "start",
        RunOptions::new(()).run_config(config.clone()),
        false,
    )
    .await
    .unwrap();
    assert_eq!(interrupted.interruptions.len(), 1);
    assert_eq!(count(&interrupted.to_input_list(), "once"), 1);

    let mut state = interrupted.to_state();
    assert_eq!(state.nested_history_owned_items.len(), 1);
    state.approve(&interrupted.interruptions[0], false);
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    assert_eq!(
        restored.nested_history_owned_items,
        state.nested_history_owned_items
    );

    let resumed = run_mode(
        &first,
        restored,
        RunOptions::new(()).run_config(config),
        false,
    )
    .await
    .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let final_input = final_model.calls().last().unwrap().input.clone();
    assert_eq!(item_text(&final_input[0]).matches("once").count(), 1);
    assert_eq!(count(&resumed.to_input_list(), "once"), 2);
}

/// Python: `test_public_nested_history_filter_preserves_ownership_after_deepcopy`.
#[tokio::test]
async fn nested_history_copied_by_a_filter_is_replayed_once() {
    let chain = chain("same", |h| {
        h.with_input_filter(|data: HandoffInputData| {
            let mut nested = nest_handoff_history(data, None);
            nested.input_history = nested.input_history.clone();
            nested
        })
    });
    assert_eq!(
        count(&run_nested(&chain, false).await.to_input_list(), "same"),
        1
    );
}

/// Python: `test_public_nested_history_filter_preserves_ownership_after_dict_rebuild`.
#[tokio::test]
async fn nested_history_rebuilt_as_fresh_mappings_is_replayed_once() {
    let chain = chain("same", |h| {
        h.with_input_filter(|data: HandoffInputData| {
            let mut nested = nest_handoff_history(data, None);
            nested.input_history = nested
                .input_history
                .iter()
                .map(|item| InputItem(item.0.clone()))
                .collect();
            nested
        })
    });
    assert_eq!(
        count(&run_nested(&chain, false).await.to_input_list(), "same"),
        1
    );
}

/// Python: `test_public_nested_history_filter_preserves_ownership`, direct and chained with
/// `remove_all_tools`.
#[tokio::test]
async fn public_nested_history_filter_preserves_ownership() {
    for chained in [false, true] {
        let chain = chain("handoff message", |h| {
            h.with_input_filter(move |data: HandoffInputData| {
                let nested = nest_handoff_history(data, None);
                if chained {
                    openai_agents_sdk::extensions::handoff_filters::remove_all_tools(nested)
                } else {
                    nested
                }
            })
        });
        let result = run_mode(&chain.first, "start", RunOptions::new(()), false)
            .await
            .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "handoff message"),
            1,
            "chained={chained}"
        );
    }
}

/// Python: `test_public_nested_history_filter_rebases_owned_input_after_insertion`.
#[tokio::test]
async fn owned_input_is_rebased_after_a_clean_item_is_inserted() {
    for streamed in [false, true] {
        let chain = chain("same", |h| {
            h.with_input_filter(|data: HandoffInputData| {
                let mut nested = nest_handoff_history(data, None);
                nested.input_history.insert(
                    0,
                    InputItem(serde_json::json!({"role": "user", "content": "inserted"})),
                );
                nested
            })
        });
        let result = run_mode(&chain.first, "start", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "same"),
            1,
            "streamed={streamed}"
        );
    }
}

/// Python: `test_public_nested_history_filter_preserves_ownership_after_new_items_copy`.
#[tokio::test]
async fn nested_history_with_copied_new_items_is_replayed_once() {
    let chain = chain("same", |h| {
        h.with_input_filter(|data: HandoffInputData| {
            let mut nested = nest_handoff_history(data, None);
            nested.new_items = nested.new_items.clone();
            nested
        })
    });
    assert_eq!(
        count(&run_nested(&chain, false).await.to_input_list(), "same"),
        1
    );
}

/// Python: `test_nested_history_preserves_repeated_run_item_reference_occurrences`. One RunItem
/// listed twice is two logical occurrences, so it is replayed twice.
#[tokio::test]
async fn repeated_run_item_is_replayed_as_two_occurrences() {
    for streamed in [false, true] {
        let chain = chain("same", |h| {
            h.with_input_filter(|data: HandoffInputData| {
                let message = data.new_items[0].clone();
                let mut new_items = vec![message.clone(), message];
                new_items.extend(data.new_items[1..].iter().cloned());
                nest_handoff_history(HandoffInputData { new_items, ..data }, None)
            })
        });
        let result = run_mode(&chain.first, "start", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "same"),
            2,
            "streamed={streamed}"
        );
    }
}

/// A three-agent chain `first -> second -> final`; `second_handoff` configures the second
/// handoff. Returns the first agent in a [`Chain`].
fn three_agents(
    first_message: &str,
    second_message: &str,
    second_handoff: impl FnOnce(Handoff<()>) -> Handoff<()>,
) -> Chain {
    let first_model = setup();
    let second_model = setup();
    let final_model = setup();
    final_model.add_turn(vec![text("done")]);
    let final_agent = Arc::new(Agent::<()>::new("final").with_model_instance(final_model.clone()));
    second_model.add_turn(vec![text(second_message), transfer("final")]);
    let second = Arc::new(
        Agent::<()>::new("second")
            .with_model_instance(second_model.clone())
            .with_handoff(second_handoff(handoff(final_agent))),
    );
    first_model.add_turn(vec![text(first_message), transfer("second")]);
    let first = Arc::new(
        Agent::<()>::new("first")
            .with_model_instance(first_model.clone())
            .with_handoff(second),
    );
    Chain { first }
}

/// Python: `test_identity_filter_preserves_prior_nested_history_ownership`.
#[tokio::test]
async fn identity_filter_preserves_prior_nested_history_ownership() {
    for streamed in [false, true] {
        let chain = three_agents("first message", "second message", |h| {
            h.with_input_filter(|data: HandoffInputData| data)
        });
        let result = run_mode(
            &chain.first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "first message"),
            1,
            "streamed={streamed}"
        );
    }
}

/// Python: `test_non_nested_copying_filter_preserves_prior_nested_history_ownership`.
#[tokio::test]
async fn non_nested_copying_filter_preserves_prior_nested_history_ownership() {
    for streamed in [false, true] {
        let chain = three_agents("first message", "second message", |h| {
            h.with_nest_handoff_history(false)
                .with_input_filter(|data: HandoffInputData| HandoffInputData {
                    input_history: data.input_history.clone(),
                    ..data
                })
        });
        let result = run_mode(
            &chain.first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "first message"),
            1,
            "streamed={streamed}"
        );
    }
}

/// Python: `test_nested_history_ownership_remaps_after_new_items_insertion`. A caller inserting a
/// new item must not make the owned message disappear from the replay.
#[tokio::test]
async fn ownership_remaps_after_a_new_item_is_inserted() {
    let chain = chain("handoff message", |h| h);
    let mut result = run_nested(&chain, false).await;
    result.new_items.insert(
        0,
        RunItem::new(
            "first",
            RunItemKind::MessageOutput {
                raw_item: MessageOutput::text("inserted"),
            },
        ),
    );
    let replay = result.to_input_list();
    assert_eq!(count(&replay, "inserted"), 1);
    assert_eq!(count(&replay, "handoff message"), 1);
}

/// Python: `test_nested_history_ownership_revalidates_after_input_removal`. Removing the
/// forwarded message from `result.input` puts its session copy back into the replay.
#[tokio::test]
async fn ownership_revalidates_after_the_forwarded_input_is_removed() {
    let chain = chain("handoff message", |h| h);
    let mut result = run_nested(&chain, false).await;
    result.input.retain(|i| item_text(i) != "handoff message");
    assert_eq!(count(&result.to_input_list(), "handoff message"), 1);
}

/// Python: `test_nested_history_new_item_removal_does_not_claim_an_equal_item`. Replacing the
/// owned RunItem with an equal one leaves the equal item as a separate occurrence.
#[tokio::test]
async fn an_equal_replacement_for_the_owned_new_item_is_a_separate_occurrence() {
    let chain = chain("same", |h| h);
    let mut result = run_nested(&chain, false).await;
    let equal = RunItem::new(
        "first",
        RunItemKind::MessageOutput {
            raw_item: MessageOutput::text("same"),
        },
    );
    let is_same = |item: &RunItem| item.to_input_item().text_content().as_deref() == Some("same");
    let others: Vec<RunItem> = result
        .new_items
        .iter()
        .filter(|i| !is_same(i))
        .cloned()
        .collect();
    result.new_items = std::iter::once(equal).chain(others).collect();
    assert_eq!(count(&result.to_input_list(), "same"), 2);
}

/// Python: `test_public_nested_history_filter_does_not_own_equal_replacement`. An equal new item
/// that a filter puts in place of the forwarded one is a separate occurrence.
#[tokio::test]
async fn nested_history_filter_does_not_own_an_equal_replacement() {
    let chain = chain("same", |h| {
        h.with_input_filter(|data: HandoffInputData| {
            let nested = nest_handoff_history(data, None);
            let original = nested.new_items[0].clone();
            let replacement = RunItem::new(
                original.agent.clone(),
                RunItemKind::MessageOutput {
                    raw_item: MessageOutput::text("same"),
                },
            );
            let mut new_items = vec![replacement];
            new_items.extend(nested.new_items[1..].iter().cloned());
            HandoffInputData {
                new_items,
                ..nested
            }
        })
    });
    assert_eq!(
        count(&run_nested(&chain, false).await.to_input_list(), "same"),
        2
    );
}
