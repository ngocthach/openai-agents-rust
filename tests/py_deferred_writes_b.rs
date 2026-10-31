//! Parity port of end-state tests from `tests/deferred_session_writes/test_recovery.py` and
//! `tests/deferred_session_writes/test_filters.py`.
//!
//! The Python tests inspect the held record (`pending_session_write`, `_pending_session_write`)
//! and the internal settle predicates. Only the session's final content and the run outcome
//! are asserted here, through `Runner`, `RunState` and `Session`.

use std::sync::Mutex;

use async_trait::async_trait;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::handoffs::{HandoffInputData, handoff};
use openai_agents_sdk::items::{InputItem, RunItemKind};
use openai_agents_sdk::memory::{Session, SessionSettings};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

mod py_mcpmem_support;
use py_mcpmem_support::ListSession;

#[derive(Deserialize, JsonSchema)]
struct Query {
    query: String,
}

/// Python's `always_fine` output guardrail. Its presence is what makes an interruption withhold
/// its session write.
fn always_fine() -> OutputGuardrail<()> {
    OutputGuardrail::new("always_fine", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, false))
    })
}

fn write_thing() -> FunctionTool<()> {
    FunctionTool::new("write_thing", "writes", |_ctx, q: Query| async move {
        Ok(format!("wrote:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn write_other() -> FunctionTool<()> {
    FunctionTool::new("write_other", "writes", |_ctx, q: Query| async move {
        Ok(format!("other:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn look_up() -> FunctionTool<()> {
    FunctionTool::new("look_up", "looks up", |_ctx, q: Query| async move {
        Ok(format!("schema for {}", q.query))
    })
}

/// A session whose next `add_items` fails either before or after the write is applied
/// (Python's `_FailingResumeSession`: "atomic-failure" versus "lost-ack").
#[derive(Debug)]
struct FlakySession {
    inner: ListSession,
    failure: Mutex<Option<&'static str>>,
}

impl FlakySession {
    fn new() -> Self {
        FlakySession {
            inner: ListSession::new("test"),
            failure: Mutex::new(None),
        }
    }

    fn arm(&self, failure: &'static str) {
        *self.failure.lock().unwrap() = Some(failure);
    }

    fn items(&self) -> Vec<InputItem> {
        self.inner.items()
    }
}

#[async_trait]
impl Session for FlakySession {
    fn session_id(&self) -> &str {
        "test"
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let failure = self.failure.lock().unwrap().take();
        if failure == Some("before") {
            return Err(AgentsError::other("session append failed"));
        }
        self.inner.add_items(items).await?;
        if failure == Some("after") {
            return Err(AgentsError::other("session append failed"));
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.inner.pop_item().await
    }

    async fn clear_session(&self) -> Result<()> {
        self.inner.clear_session().await
    }
}

fn call_ids(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some("function_call"))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn output_ids(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.item_type() == Some("function_call_output"))
        .filter_map(|i| i.as_value().get("call_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn orphaned_outputs(items: &[InputItem]) -> Vec<String> {
    let calls = call_ids(items);
    output_ids(items)
        .into_iter()
        .filter(|o| !calls.contains(o))
        .collect()
}

fn parked_pair(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.as_value().get("call_id").and_then(Value::as_str) == Some("call_PARKED"))
        .filter_map(|i| i.item_type().map(str::to_string))
        .collect()
}

const EXPECTED_PAIR: [&str; 2] = ["function_call", "function_call_output"];

fn deferring_agent(model: &std::sync::Arc<FakeModel>) -> Agent<()> {
    model.add_turn(vec![function_call_with_id(
        "look_up",
        r#"{"query":"x"}"#,
        "call_LOOKUP",
    )]);
    model.add_turn(vec![function_call_with_id(
        "write_thing",
        r#"{"query":"x"}"#,
        "call_PARKED",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    Agent::<()>::new("deferred repro")
        .with_tool(look_up())
        .with_tool(write_thing())
        .with_output_guardrail(always_fine())
        .with_tool_use_behavior(openai_agents_sdk::agent::ToolUseBehavior::StopAtTools(
            openai_agents_sdk::agent::StopAtTools {
                stop_at_tool_names: vec!["finish".into()],
            },
        ))
        .with_model_instance(model.clone())
}

/// Python's `test_a_failed_settle_of_the_held_batch_is_recovered_on_the_next_resume`, atomic
/// failure: the append raises before anything is written, so the retry settles it once.
#[tokio::test]
async fn a_failed_settle_is_recovered_when_the_append_fails_atomically() {
    recovery_cases("before").await;
}

/// The lost-ack parametrization: the append is written and then reports failure. The retry
/// recognizes the written batch at the tail of the session and does not append it again.
#[tokio::test]
async fn a_failed_settle_is_recovered_when_the_append_is_lost_after_writing() {
    recovery_cases("after").await;
}

async fn recovery_cases(failure: &'static str) {
    openai_agents_sdk::set_tracing_disabled(true);
    {
        for streamed in [false, true] {
            for round_trip in [false, true] {
                for retry_streamed in [false, true] {
                    let label = format!("{failure}/{streamed}/{round_trip}/{retry_streamed}");
                    let model = std::sync::Arc::new(FakeModel::new());
                    let agent = deferring_agent(&model);
                    let session = std::sync::Arc::new(FlakySession::new());

                    let first = Runner::run_with_options(
                        &agent,
                        "do the thing",
                        RunOptions::new(()).session(session.clone()),
                    )
                    .await
                    .unwrap();
                    let mut state =
                        RunState::from_json(first.to_state().to_json().unwrap()).unwrap();
                    let item = state.get_interruptions()[0].clone();
                    state.approve(&item, false);

                    session.arm(failure);
                    let failed = if streamed {
                        let mut result = Runner::run_streamed_with_options(
                            &agent,
                            state.clone(),
                            RunOptions::new(()).session(session.clone()),
                        );
                        let mut err = None;
                        {
                            use futures::StreamExt;
                            let mut events = std::pin::pin!(result.stream_events());
                            while let Some(event) = events.next().await {
                                if let Err(e) = event {
                                    err = Some(e);
                                    break;
                                }
                            }
                        }
                        err
                    } else {
                        Runner::run_with_options(
                            &agent,
                            state.clone(),
                            RunOptions::new(()).session(session.clone()),
                        )
                        .await
                        .err()
                    };
                    let failed = failed.unwrap_or_else(|| panic!("{label}: the append must fail"));
                    assert!(
                        failed.to_string().contains("session append failed"),
                        "{label}: {failed}"
                    );

                    if round_trip {
                        state = RunState::from_json(state.to_json().unwrap()).unwrap();
                    }
                    let finished = if retry_streamed {
                        let mut result = Runner::run_streamed_with_options(
                            &agent,
                            state,
                            RunOptions::new(()).session(session.clone()),
                        );
                        {
                            use futures::StreamExt;
                            let mut events = std::pin::pin!(result.stream_events());
                            while let Some(event) = events.next().await {
                                event.unwrap();
                            }
                        }
                        result.final_output_text()
                    } else {
                        Runner::run_with_options(
                            &agent,
                            state,
                            RunOptions::new(()).session(session.clone()),
                        )
                        .await
                        .unwrap()
                        .final_output_text()
                    };
                    assert_eq!(finished.as_deref(), Some("done"), "{label}");

                    let items = session.items();
                    assert!(orphaned_outputs(&items).is_empty(), "{label}");
                    assert_eq!(parked_pair(&items), EXPECTED_PAIR, "{label}");
                }
            }
        }
    }
}

/// Python's `test_a_filter_that_drops_one_output_takes_its_held_call_with_it`: a handoff filter
/// removes one of two resolved outputs; the session must keep the surviving pair and must not
/// settle the filtered call from the held batch.
#[tokio::test]
async fn a_filter_that_drops_one_output_takes_its_held_call_with_it() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = std::sync::Arc::new(FakeModel::new());
        model.add_turn(vec![
            function_call_with_id("write_thing", r#"{"query":"x"}"#, "call_PARKED"),
            function_call_with_id("write_other", r#"{"query":"x"}"#, "call_PARKED_2"),
            function_call_with_id("transfer_to_target", "{}", "call_HANDOFF"),
        ]);
        let target_model = std::sync::Arc::new(FakeModel::new());
        target_model.add_turn(vec![assistant_message("done")]);
        let target = Agent::<()>::new("target")
            .with_instructions("x")
            .with_model_instance(target_model);
        let drop_one = handoff(target).with_input_filter(|data: HandoffInputData| {
            let keep = |items: Vec<openai_agents_sdk::items::RunItem>| {
                items
                    .into_iter()
                    .filter(|item| {
                        !matches!(
                            &item.kind,
                            RunItemKind::ToolCallOutput { raw_item, .. }
                                if raw_item.get("call_id").and_then(Value::as_str) == Some("call_PARKED_2")
                        )
                    })
                    .collect()
            };
            HandoffInputData {
                input_history: data.input_history,
                pre_handoff_items: keep(data.pre_handoff_items),
                new_items: keep(data.new_items),
                input_items: data.input_items,
            }
        });
        let agent = Agent::<()>::new("deferred repro (partial filter)")
            .with_instructions("x")
            .with_tool(write_thing())
            .with_tool(write_other())
            .with_handoff(drop_one)
            .with_output_guardrail(always_fine())
            .with_tool_use_behavior(openai_agents_sdk::agent::ToolUseBehavior::StopAtTools(
                openai_agents_sdk::agent::StopAtTools {
                    stop_at_tool_names: vec!["finish".into()],
                },
            ))
            .with_model_instance(model.clone());
        let session = std::sync::Arc::new(ListSession::new("test"));

        let first =
            Runner::run_with_options(&agent, "go", RunOptions::new(()).session(session.clone()))
                .await
                .unwrap();
        let mut state = RunState::from_json(first.to_state().to_json().unwrap()).unwrap();
        for interruption in state.get_interruptions() {
            state.approve(&interruption, false);
        }
        if streamed {
            use futures::StreamExt;
            let mut result = Runner::run_streamed_with_options(
                &agent,
                state,
                RunOptions::new(()).session(session.clone()),
            );
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        } else {
            Runner::run_with_options(&agent, state, RunOptions::new(()).session(session.clone()))
                .await
                .unwrap();
        }

        let items = session.items();
        let calls = call_ids(&items);
        let outputs = output_ids(&items);
        for call in &calls {
            assert!(
                outputs.contains(call),
                "dangling call {call} (streamed={streamed})"
            );
        }
        for output in &outputs {
            assert!(
                calls.contains(output),
                "orphaned output {output} (streamed={streamed})"
            );
        }
        assert!(
            calls.contains(&"call_PARKED".to_string()),
            "streamed={streamed}"
        );
        assert!(
            !calls.contains(&"call_PARKED_2".to_string()),
            "the filtered output must not settle from the batch (streamed={streamed})"
        );
    }
}
