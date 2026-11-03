//! Ports of the caller-visible recovery cases in the Python SDK's
//! `tests/test_handoff_failure_error_function.py`.
//!
//! Python-only cases, not ported here:
//! - `async_formatter=True`, the re-raise half and `failure="formatter"`: ported in
//!   `py_core10_r10b_failure_formatters.rs` (the formatter is a `ToolErrorFn`, which may await
//!   and may fail).
//! - `failure="cancel"`: `asyncio.CancelledError` is an asyncio control-flow exception; Rust
//!   cancels a run by dropping its future or through the stream cancel handle.
//! - `failure="filter"`: a Rust input filter returns a plain value, so it cannot fail.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, default_tool_error_function};
use openai_agents_sdk::{Agent, AgentsError, InputItem, RunContext, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const FORMATTED: &str = "An error occurred while running the tool. Please try again.";

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

/// The input of the handoff tool: `name` must be `"a"`, anything else is a model error.
#[derive(Debug, Deserialize, JsonSchema)]
enum Letter {
    #[serde(rename = "a")]
    A,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Pick {
    name: Letter,
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn companion_tool() -> FunctionTool<()> {
    FunctionTool::new("companion", "a tool", |_ctx, _a: NoArgs| async {
        Ok("companion result".to_string())
    })
}

fn output_for(items: &[InputItem], call_id: &str) -> Vec<Value> {
    items
        .iter()
        .filter(|i| {
            i.0["type"] == "function_call_output" && i.0["call_id"].as_str() == Some(call_id)
        })
        .map(|i| i.0["output"].clone())
        .collect()
}

#[derive(Default)]
struct RecordingHooks {
    handoffs: Mutex<Vec<(String, String)>>,
}

#[async_trait::async_trait]
impl RunHooks<()> for RecordingHooks {
    async fn on_handoff(
        &self,
        _ctx: &RunContext<()>,
        from_agent: &Agent<()>,
        to_agent: &Agent<()>,
    ) {
        self.handoffs
            .lock()
            .unwrap()
            .push((from_agent.name.clone(), to_agent.name.clone()));
    }
}

/// Runs `agent` either directly or streamed, returning the result and the stream events (empty
/// for a direct run).
async fn run_caller(
    agent: &Arc<Agent<()>>,
    options: RunOptions<()>,
    streamed: bool,
) -> (Result<RunResult<()>, AgentsError>, Vec<StreamEvent<()>>) {
    if !streamed {
        let result = Runner::run_with_options(agent.clone(), "go", options).await;
        return (result, Vec::new());
    }
    let mut stream = Runner::run_streamed_with_options(agent.clone(), "go", options);
    let mut events = Vec::new();
    let mut error = None;
    {
        let mut pinned = std::pin::pin!(stream.stream_events());
        while let Some(event) = pinned.next().await {
            match event {
                Ok(event) => events.push(event),
                Err(e) => error = Some(e),
            }
        }
    }
    match error {
        Some(e) => (Err(e), events),
        None => (Ok(stream.into_result().expect("finished")), events),
    }
}

/// Python: `test_failed_handoff_retries_on_caller_then_transfers`.
#[tokio::test]
async fn failed_handoff_retries_on_caller_then_transfers() {
    for streamed in [false, true] {
        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = received.clone();
        let filter_calls = Arc::new(AtomicUsize::new(0));
        let filter_seen = filter_calls.clone();
        let target_model = setup();
        target_model.add_turn(vec![assistant_message("target answer")]);
        let target = Arc::new(Agent::<()>::new("target").with_model_instance(target_model.clone()));
        let transfer = handoff(target.clone())
            .with_on_handoff_input(move |_ctx, data: Pick| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(format!("{:?}", data.name));
                    Ok(())
                }
            })
            .with_input_filter(move |data| {
                filter_seen.fetch_add(1, Ordering::SeqCst);
                data
            })
            .with_failure_error_function(default_tool_error_function);

        let caller_model = setup();
        caller_model.add_turn(vec![
            function_call_with_id("companion", "{}", "companion"),
            function_call_with_id("transfer_to_target", r#"{"name":"b"}"#, "bad"),
        ]);
        caller_model.add_turn(vec![function_call_with_id(
            "transfer_to_target",
            r#"{"name":"a"}"#,
            "good",
        )]);
        let caller = Arc::new(
            Agent::<()>::new("caller")
                .with_model_instance(caller_model.clone())
                .with_tool(companion_tool())
                .with_handoff(transfer),
        );
        let hooks = Arc::new(RecordingHooks::default());
        let session = Arc::new(InMemorySession::new("handoff-retry"));
        let options = RunOptions::new(())
            .hooks(hooks.clone())
            .session(session.clone());

        let (result, events) = run_caller(&caller, options, streamed).await;
        let result = result.unwrap();
        let history = session.get_items(None).await.unwrap();

        assert_eq!(result.final_output_text().as_deref(), Some("target answer"));
        assert_eq!(result.last_agent.name, "target");
        assert_eq!(*received.lock().unwrap(), vec!["A".to_string()]);
        assert_eq!(caller_model.calls().len(), 2);
        assert_eq!(target_model.calls().len(), 1);
        assert_eq!(filter_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *hooks.handoffs.lock().unwrap(),
            vec![("caller".to_string(), "target".to_string())]
        );

        let expected = [("companion", "companion result"), ("bad", FORMATTED)];
        let second_caller_input = caller_model.calls()[1].input.clone();
        let target_input = target_model.calls()[0].input.clone();
        for items in [second_caller_input, target_input, history] {
            for (call_id, output) in expected {
                assert_eq!(
                    output_for(&items, call_id),
                    vec![json!(output)],
                    "{call_id}"
                );
            }
        }
        assert_eq!(
            result
                .new_items
                .iter()
                .filter(|i| matches!(i.kind, RunItemKind::HandoffOutput { .. }))
                .count(),
            1
        );
        if streamed {
            let agents: Vec<String> = events
                .iter()
                .filter_map(|e| match e {
                    StreamEvent::AgentUpdated { new_agent } => Some(new_agent.name.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(agents, vec!["caller", "target"]);
            let handoff_events = events
                .iter()
                .filter(|e| {
                    matches!(
                        e,
                        StreamEvent::RunItem { name, .. }
                            if name.as_str() == "handoff_occured"
                    )
                })
                .count();
            assert_eq!(handoff_events, 1);
        }
    }
}

/// Python: `test_default_handoff_failure_still_raises`. Without a failure formatter the invalid
/// input ends the run with a model behavior error, and the callback never runs.
#[tokio::test]
async fn default_handoff_failure_still_raises() {
    for streamed in [false, true] {
        let callbacks = Arc::new(AtomicUsize::new(0));
        let seen = callbacks.clone();
        let target_model = setup();
        let target = Arc::new(Agent::<()>::new("target").with_model_instance(target_model.clone()));
        let transfer = handoff(target).with_on_handoff_input(move |_ctx, _data: Pick| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        });
        let model = setup();
        model.add_turn(vec![function_call_with_id(
            "transfer_to_target",
            r#"{"name":"b"}"#,
            "bad",
        )]);
        let caller = Arc::new(
            Agent::<()>::new("caller")
                .with_model_instance(model.clone())
                .with_handoff(transfer),
        );

        let (result, _) = run_caller(&caller, RunOptions::new(()), streamed).await;
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. }),
            "{err:?}"
        );
        assert_eq!(callbacks.load(Ordering::SeqCst), 0);
        assert_eq!(model.calls().len(), 1);
        assert!(target_model.calls().is_empty());
    }
}

/// Python: `test_custom_handoff_can_handle_or_reraise_application_error`, the synchronous
/// formatter half. The formatter's text reaches the calling agent and no handoff output is made.
#[tokio::test]
async fn custom_handoff_failure_is_formatted_for_the_caller() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let formatter_seen = seen.clone();
    let transfer =
        failing_custom_handoff().with_failure_error_function(move |_ctx, caught: &AgentsError| {
            formatter_seen.lock().unwrap().push(caught.to_string());
            "Transfer unavailable; answer here.".to_string()
        });
    let model = setup();
    model.add_turn(vec![function_call_with_id("transfer", "{}", "call")]);
    model.add_turn(vec![assistant_message("caller answer")]);
    let caller = Arc::new(
        Agent::<()>::new("caller")
            .with_model_instance(model.clone())
            .with_handoff(transfer),
    );

    let result = Runner::run(&caller, "go").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("caller answer"));
    assert_eq!(result.last_agent.name, "caller");
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(seen.lock().unwrap()[0].contains("synthetic application failure"));
    let second_input = model.calls()[1].input.clone();
    assert_eq!(
        second_input.last().unwrap().0,
        json!({
            "type": "function_call_output",
            "call_id": "call",
            "output": "Transfer unavailable; answer here.",
        })
    );
    assert!(
        !result
            .new_items
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::HandoffOutput { .. }))
    );
}

/// The custom handoff used by the failure formatter cases: its invocation always fails with
/// an application error.
fn failing_custom_handoff() -> openai_agents_sdk::handoffs::Handoff<()> {
    openai_agents_sdk::handoffs::Handoff::custom(
        "transfer",
        "Transfer to target",
        "target",
        json!({}),
        |_ctx, _args: String| async {
            Err::<Arc<Agent<()>>, _>(AgentsError::user("synthetic application failure"))
        },
    )
}

/// Python: `test_handoff_recovery_run_sync`. The synchronous entry point recovers the same way.
#[test]
fn handoff_recovery_run_sync() {
    let target_model = setup();
    let target = Arc::new(Agent::<()>::new("target").with_model_instance(target_model.clone()));
    let transfer = handoff(target)
        .with_on_handoff_input(|_ctx, _data: Pick| async {
            panic!("invalid input must not invoke callback")
        })
        .with_failure_error_function(default_tool_error_function);
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "transfer_to_target",
        r#"{"name":"b"}"#,
        "bad",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let caller = Agent::<()>::new("caller")
        .with_model_instance(model.clone())
        .with_handoff(transfer);

    let result = Runner::run_sync(&caller, "go").unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(result.last_agent.name, "caller");
}
