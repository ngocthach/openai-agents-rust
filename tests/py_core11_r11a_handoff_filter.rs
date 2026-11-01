//! A handoff input filter may await (Python's `HandoffInputFilter` is `MaybeAwaitable`). The
//! ownership record that ties a filter's `nest_handoff_history` output to the run items it
//! forwarded must follow the filter across await points and worker threads, so these cases run
//! on a multi-thread runtime and make the filter yield both before and after the call.
//!
//! Ports of `test_public_nested_history_filter_preserves_ownership` for filters that are async.

use std::sync::Arc;

use openai_agents_sdk::handoffs::{Handoff, HandoffInputData, handoff, nest_handoff_history};
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, Runner, async_handoff_input_filter};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn text(t: &str) -> OutputItem {
    assistant_message(t)
}

fn count(items: &[InputItem], t: &str) -> usize {
    items
        .iter()
        .filter(|i| i.text_content().unwrap_or_default() == t)
        .count()
}

/// `first` says `message` and hands off to `second`, which answers `done`. Returns the first
/// agent and the model the second one calls.
fn chain(
    message: &str,
    configure: impl FnOnce(Handoff<()>) -> Handoff<()>,
) -> (Arc<Agent<()>>, Arc<FakeModel>) {
    let (first_model, second_model) = (setup(), setup());
    second_model.add_turn(vec![text("done")]);
    let second = Arc::new(Agent::<()>::new("second").with_model_instance(second_model.clone()));
    first_model.add_turn(vec![
        text(message),
        function_call("transfer_to_second", "{}"),
    ]);
    let first = Arc::new(
        Agent::<()>::new("first")
            .with_model_instance(first_model)
            .with_handoff(configure(handoff(second))),
    );
    (first, second_model)
}

async fn yield_a_few_times() {
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
}

/// A filter that awaits, calls `nest_handoff_history`, and awaits again: the items it forwarded
/// are still recognised as owned by the input, so the message is replayed once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_async_filter_keeps_the_nested_history_ownership_across_threads() {
    for round in 0..20 {
        let (first, second_model) = chain("handoff message", |h| {
            h.with_input_filter_handler(async_handoff_input_filter(
                |data: HandoffInputData| async move {
                    yield_a_few_times().await;
                    let nested = nest_handoff_history(data, None);
                    yield_a_few_times().await;
                    Ok(nested)
                },
            ))
        });
        let result: RunResult<()> = Runner::run(first, "start").await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            count(&result.to_input_list(), "handoff message"),
            1,
            "round {round}"
        );
        let input = second_model.last_call().unwrap().input;
        assert!(
            input.iter().any(|item| item
                .text_content()
                .unwrap_or_default()
                .contains("<CONVERSATION HISTORY>")),
            "round {round}: the next agent receives the summary"
        );
    }
}

/// Concurrent runs each keep their own ownership record: no run sees (or loses) the entries of
/// another while their filters interleave.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_runs_with_async_filters_keep_separate_ownership_records() {
    let mut runs = Vec::new();
    for index in 0..16 {
        let message = format!("handoff message {index}");
        let (first, _second_model) = chain(&message, |h| {
            h.with_input_filter_handler(async_handoff_input_filter(
                |data: HandoffInputData| async move {
                    yield_a_few_times().await;
                    Ok(nest_handoff_history(data, None))
                },
            ))
        });
        runs.push(tokio::spawn(async move {
            let result = Runner::run(first, "start").await.unwrap();
            (message, result.to_input_list())
        }));
    }
    for run in runs {
        let (message, items) = run.await.unwrap();
        assert_eq!(count(&items, &message), 1, "{message}");
    }
}

/// An async filter that only rewrites the history it is given: the next agent's input is what the
/// filter returned after awaiting, and the model call sees nothing else.
#[tokio::test]
async fn an_async_filter_rewrites_the_input_the_next_agent_sees() {
    let (first, second_model) = chain("handoff message", |h| {
        h.with_input_filter_handler(async_handoff_input_filter(
            |mut data: HandoffInputData| async move {
                tokio::task::yield_now().await;
                data.input_history = vec![InputItem::user("rewritten by an async filter")];
                data.pre_handoff_items = Vec::new();
                data.new_items = Vec::new();
                data.input_items = None;
                Ok(data)
            },
        ))
    });
    let result = Runner::run(first, "start").await.unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let input = second_model.last_call().unwrap().input;
    assert_eq!(input.len(), 1);
    assert_eq!(
        input[0].text_content().as_deref(),
        Some("rewritten by an async filter")
    );
}
