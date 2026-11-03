//! Port of the identical-occurrence cases of `tests/test_run_state_pending_input.py`:
//! `test_server_tracker_distinguishes_identical_input_occurrences_after_restore` and
//! `test_server_managed_resume_sends_identical_late_input_in_later_occurrence` (both params).
//!
//! Under a server-managed conversation, every admitted input is its own occurrence
//! (`RunItem::input_id`, Python's `InputItem.input_id`). A later input identical to one already
//! delivered is sent again; an occurrence already delivered is never resent, and one that was
//! admitted but not delivered (a failed request) is sent once on retry.
//!
//! Python's `RunState` is mutated by the run it resumes; a Rust run consumes its state, so the
//! admitted items are read from `result.to_state()`.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::{InputItem, RunInput, RunItem, RunItemKind};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_state::{RunState, ServerConversationState};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, RunItemStreamEventName, RunResult, Runner, ServerConversationTracker, StreamEvent,
};
use serde_json::{Value, json};

type Calls = Arc<Mutex<Vec<String>>>;

fn server_options() -> RunOptions<()> {
    let mut options = RunOptions::new(()).run_config(RunConfig::new());
    options.auto_previous_response_id = true;
    options
}

fn message_text(item: &InputItem) -> Option<String> {
    let obj = item.0.as_object()?;
    if obj.get("role")?.as_str()? != "user" {
        return None;
    }
    match obj.get("content")? {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect(),
        ),
        _ => None,
    }
}

fn count_text(items: &[InputItem], text: &str) -> usize {
    items
        .iter()
        .filter(|i| message_text(i).as_deref() == Some(text))
        .count()
}

fn destination_call(destination: &str, call_id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id(
        "record_destination",
        json!({"destination": destination}).to_string(),
        call_id,
    )
}

/// Stream `state` (or a fresh input) until the first tool output, cancel after that turn, and
/// return the state it leaves. Python's `cancel(mode="after_turn")` loop.
async fn cancel_after_first_tool_output(agent: &Arc<Agent<()>>, input: RunInput) -> RunState {
    let mut streamed = Runner::run_streamed_with_options(agent, input, server_options());
    let handle = streamed.cancel_handle();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem {
                name: RunItemStreamEventName::ToolOutput,
                ..
            } = event.unwrap()
            {
                handle.cancel(CancelMode::AfterTurn);
            }
        }
    }
    streamed
        .to_state()
        .expect("a cancelled run leaves its state")
}

async fn run_to_end(agent: &Arc<Agent<()>>, state: RunState, streamed: bool) -> RunResult<()> {
    if streamed {
        let mut result =
            Runner::run_streamed_with_options(agent, RunInput::from(state), server_options());
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        return result
            .into_result()
            .expect("completed stream holds its result");
    }
    Runner::run_with_options(agent, RunInput::from(state), server_options())
        .await
        .unwrap()
}

fn admitted(state: &RunState) -> Vec<&RunItem> {
    state
        .generated_items
        .iter()
        .filter(|item| matches!(item.kind, RunItemKind::Input { .. }))
        .collect()
}

fn admitted_input(text: &str, input_id: &str) -> RunItem {
    let mut item = RunItem::new(
        "assistant",
        RunItemKind::Input {
            raw_item: InputItem(json!({"role": "user", "content": text})),
        },
    );
    item.input_id = Some(input_id.to_string());
    item
}

/// A restored tracker that already delivered one occurrence of "Repeat" selects only the second,
/// identical occurrence. Python hydrates the delivered occurrence from the generated items; a
/// Rust snapshot carries it in `ServerConversationState::accepted_input_ids`.
#[test]
fn server_tracker_distinguishes_identical_input_occurrences_after_restore() {
    let first = admitted_input("Repeat", "input-first");
    let second = admitted_input("Repeat", "input-second");
    let state = ServerConversationState {
        accepted_input_ids: ["input-first".to_string()].into_iter().collect(),
        ..Default::default()
    };
    let mut tracker = ServerConversationTracker::default();
    tracker.hydrate_from_state(
        &state,
        &[InputItem(
            json!({"role": "user", "content": "Initial request"}),
        )],
        std::slice::from_ref(&first),
        &[],
        true,
    );

    let selected: Vec<&RunItem> = [&first, &second]
        .into_iter()
        .filter(|item| tracker.needs_send_item(item, &item.to_input_item()))
        .collect();
    assert_eq!(selected, vec![&second]);
    assert!(tracker.was_sent_item(&first));
    assert!(!tracker.was_sent_item(&second));
}

async fn sends_identical_late_input_in_later_occurrence(streamed_second_resume: bool) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let calls: Calls = Arc::default();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![destination_call("Paris", "call-destination")]);
    let log = calls.clone();
    let tool = FunctionTool::<()>::new(
        "record_destination",
        "Records a destination",
        move |_c, a: Value| {
            let log = log.clone();
            async move {
                let destination = a["destination"].as_str().unwrap_or_default().to_string();
                log.lock().unwrap().push(destination.clone());
                Ok(format!("recorded:{destination}"))
            }
        },
    );
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(model.clone())
            .with_tools(vec![tool]),
    );
    let mut state = cancel_after_first_tool_output(&agent, RunInput::from("Initial request")).await;

    state.add_input("Repeat").unwrap();
    model.add_turn(vec![destination_call("Rome", "call-second-destination")]);
    let parked = cancel_after_first_tool_output(&agent, RunInput::from(state)).await;
    assert_eq!(
        count_text(&model.last_call().unwrap().input, "Repeat"),
        1,
        "the first occurrence is delivered by the first resume"
    );

    let mut state = RunState::from_json(parked.to_json().unwrap()).unwrap();
    let before = admitted(&state);
    assert_eq!(before.len(), 1);
    let before_id = before[0].input_id.clone().expect("an admitted occurrence");
    state.add_input("Repeat").unwrap();
    model.add_turn(vec![assistant_message("Done")]);

    let result = run_to_end(&agent, state, streamed_second_resume).await;

    assert_eq!(result.final_output_text().as_deref(), Some("Done"));
    assert_eq!(
        *calls.lock().unwrap(),
        vec!["Paris".to_string(), "Rome".to_string()]
    );
    assert_eq!(count_text(&model.last_call().unwrap().input, "Repeat"), 1);
    let final_state = result.to_state();
    let after: Vec<Option<String>> = admitted(&final_state)
        .into_iter()
        .map(|item| item.input_id.clone())
        .collect();
    assert_eq!(after.len(), 2);
    assert_eq!(
        after
            .iter()
            .filter(|id| id.as_deref() == Some(before_id.as_str()))
            .count(),
        1
    );
    assert_ne!(after[0], after[1]);
}

#[tokio::test]
async fn server_managed_resume_sends_identical_late_input_in_later_occurrence_blocking() {
    sends_identical_late_input_in_later_occurrence(false).await;
}

#[tokio::test]
async fn server_managed_resume_sends_identical_late_input_in_later_occurrence_streamed() {
    sends_identical_late_input_in_later_occurrence(true).await;
}
