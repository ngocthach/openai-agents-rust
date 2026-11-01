//! Ports of `tests/test_run_impl_resume_paths.py::test_resumed_stream_waits_for_event_consumption_before_continuing`
//! and of the persisted-count cases around resumed session saves, driven through the public
//! runner instead of monkeypatching the run loop.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunOptions, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;

#[derive(Deserialize, JsonSchema)]
struct Amount {
    #[allow(dead_code)]
    amount: i64,
}

fn call(name: &str, args: Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

/// The `charge` tool waits for `gate` once it is approved, so the test can attach its consumer
/// before the resumed turn finishes (Python waits for the first agent-updated event instead).
fn charge_tool(gate: Arc<Notify>) -> FunctionTool<()> {
    FunctionTool::new("charge", "", move |_c, _a: Amount| {
        let gate = gate.clone();
        async move {
            gate.notified().await;
            Ok("receipt-7".to_string())
        }
    })
    .with_needs_approval(true)
}

/// A turn that parks on the `charge` approval; with `handoff_in_turn` the model also asked to
/// hand off to `delegate` in the same response.
async fn approved_state(
    handoff_in_turn: bool,
    gate: Arc<Notify>,
) -> (Agent<()>, Arc<Agent<()>>, Arc<ScriptedModel>, RunState) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::new());
    let mut first = vec![call("charge", json!({"amount": 7}), "charge-1")];
    if handoff_in_turn {
        first.push(call("transfer_to_delegate", json!({}), "handoff-1"));
    }
    model.enqueue(ScriptedStep::output(first)).unwrap();
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("unexpected")]))
        .unwrap();
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let mut agent = Agent::<()>::new("resume-agent")
        .with_model_instance(model.clone())
        .with_tool(charge_tool(gate.clone()));
    if handoff_in_turn {
        agent = agent.with_handoff(handoff(delegate.clone()));
    }
    // Approval is requested before the tool body runs, so the first run never reaches the gate.
    let paused = Runner::run(&agent, "go").await.unwrap();
    let mut state = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
    let interruption = state.get_interruptions()[0].clone();
    state.approve(&interruption, false);
    (agent, delegate, model, state)
}

/// Python: `test_resumed_stream_waits_for_event_consumption_before_continuing`. A consumer
/// that is still handling the resumed tool's output event keeps the run from starting the next
/// model turn; cancelling after the turn from there stops the run at the continuation's agent.
#[tokio::test]
async fn resumed_stream_waits_for_event_consumption_before_continuing() {
    for continuation in ["run_again", "handoff"] {
        let gate = Arc::new(Notify::new());
        let (agent, delegate, model, state) =
            approved_state(continuation == "handoff", gate.clone()).await;
        let mut result = Runner::run_streamed_with_options(&agent, state, RunOptions::new(()));
        let handle = result.cancel_handle();
        let attached = Arc::new(Notify::new());
        let suspended = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let calls_while_suspended = Arc::new(Mutex::new(Vec::<usize>::new()));
        let consumer = {
            let (attached, suspended, release) =
                (attached.clone(), suspended.clone(), release.clone());
            let model = model.clone();
            let seen = calls_while_suspended.clone();
            tokio::spawn(async move {
                {
                    let mut events = std::pin::pin!(result.stream_events());
                    while let Some(event) = events.next().await {
                        // The first event proves the consumer is attached to the stream.
                        attached.notify_one();
                        if let Ok(StreamEvent::RunItem {
                            name: RunItemStreamEventName::ToolOutput,
                            ..
                        }) = event
                        {
                            suspended.notify_one();
                            release.notified().await;
                            // The model must not have been asked for another turn meanwhile.
                            seen.lock().unwrap().push(model.calls().len());
                            handle.cancel(CancelMode::AfterTurn);
                        }
                    }
                }
                result
            })
        };
        // The resumed tool finishes only once the consumer has seen its first event.
        tokio::time::timeout(Duration::from_secs(5), attached.notified())
            .await
            .unwrap_or_else(|_| panic!("{continuation}: the consumer never received an event"));
        gate.notify_one();
        tokio::time::timeout(Duration::from_secs(5), suspended.notified())
            .await
            .unwrap_or_else(|_| panic!("{continuation}: the tool output was never announced"));
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert_eq!(
                model.calls().len(),
                1,
                "{continuation}: the next turn started while the consumer was suspended"
            );
        }
        release.notify_one();
        let result = tokio::time::timeout(Duration::from_secs(5), consumer)
            .await
            .unwrap_or_else(|_| panic!("{continuation}: the stream never ended"))
            .unwrap();
        assert_eq!(*calls_while_suspended.lock().unwrap(), vec![1]);
        assert_eq!(model.calls().len(), 1, "{continuation}");
        assert!(result.final_output().is_none(), "{continuation}");
        let expected = if continuation == "handoff" {
            delegate.name.as_str()
        } else {
            agent.name.as_str()
        };
        assert_eq!(result.current_agent().name, expected, "{continuation}");
        assert_eq!(
            result.to_state().expect("a state").current_agent_name,
            expected,
            "{continuation}"
        );
    }
}

/// Python: `test_resumed_run_again_resets_persisted_count`. After a resumed approval the next
/// turn's own tool call and output are written to the session, whatever the interrupted turn
/// had already saved.
#[tokio::test]
async fn resumed_run_again_saves_the_next_turn_items() {
    use openai_agents_sdk::items::InputItem;
    use openai_agents_sdk::memory::Session;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(ScriptedModel::new());
        for step in [
            vec![call("charge", json!({"amount": 7}), "charge-1")],
            vec![call("lookup", json!({}), "lookup-1")],
            vec![assistant_message("final")],
        ] {
            model.enqueue(ScriptedStep::output(step)).unwrap();
        }
        let gate = Arc::new(Notify::new());
        gate.notify_one();
        let lookup = FunctionTool::new_no_args("lookup", "", |_c| async move { Ok("found") });
        let agent = Agent::<()>::new("resume-agent")
            .with_model_instance(model.clone())
            .with_tool(charge_tool(gate))
            .with_tool(lookup);
        let session = Arc::new(openai_agents_sdk::InMemorySession::new("resume-count"));
        let dyn_session: Arc<dyn Session> = session.clone();
        let paused = Runner::run_with_options(
            &agent,
            "go",
            RunOptions::new(()).session(dyn_session.clone()),
        )
        .await
        .unwrap();
        let mut state = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
        let interruption = state.get_interruptions()[0].clone();
        state.approve(&interruption, false);

        let options = RunOptions::new(()).session(dyn_session.clone());
        let output = if streamed {
            let mut result = Runner::run_streamed_with_options(&agent, state, options);
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(event) = events.next().await {
                    event.unwrap();
                }
            }
            result.final_output().cloned()
        } else {
            Runner::run_with_options(&agent, state, options)
                .await
                .unwrap()
                .final_output
                .clone()
        };
        assert_eq!(output, Some(json!("final")), "streamed={streamed}");
        let items: Vec<InputItem> = session.get_items(None).await.unwrap();
        let kinds: Vec<(String, String)> = items
            .iter()
            .filter_map(|i| {
                Some((
                    i.0.get("type")?.as_str()?.to_string(),
                    i.0.get("call_id")?.as_str()?.to_string(),
                ))
            })
            .collect();
        let expected: Vec<(String, String)> = [
            ("function_call", "charge-1"),
            ("function_call_output", "charge-1"),
            ("function_call", "lookup-1"),
            ("function_call_output", "lookup-1"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        assert_eq!(kinds, expected, "streamed={streamed}");
    }
}

/// Python: `test_resumed_interruption_passes_server_managed_conversation_flag`. A resumed
/// interruption whose snapshot came from a server-managed run (conversation id, previous
/// response id or automatic chaining) is resolved as server-managed: a handoff input filter is
/// refused when the parked handoff runs, whichever of the three modes the run used.
#[tokio::test]
async fn resumed_interruption_passes_server_managed_conversation_flag() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for mode in ["conversation_id", "previous_response_id", "auto"] {
        let model = Arc::new(ScriptedModel::new());
        model
            .enqueue(
                ScriptedStep::output(vec![
                    call("charge", json!({"amount": 7}), "charge-1"),
                    call("transfer_to_delegate", json!({}), "handoff-1"),
                ])
                .with_response_id(Some("resp-1".to_string())),
            )
            .unwrap();
        model
            .enqueue(ScriptedStep::output(vec![assistant_message("unexpected")]))
            .unwrap();
        let gate = Arc::new(Notify::new());
        gate.notify_one();
        let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
        let agent = Agent::<()>::new("resume-agent")
            .with_model_instance(model.clone())
            .with_tool(charge_tool(gate))
            .with_handoff(handoff(delegate).with_input_filter(|data| data));
        let configure = |options: RunOptions<()>| match mode {
            "conversation_id" => options.conversation_id("conv_1"),
            "previous_response_id" => options.previous_response_id("resp_prev"),
            _ => options.auto_previous_response_id(true),
        };
        let paused = Runner::run_with_options(&agent, "go", configure(RunOptions::new(())))
            .await
            .unwrap();
        let mut state = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
        let interruption = state.get_interruptions()[0].clone();
        state.approve(&interruption, false);
        let err = Runner::run_with_options(&agent, state, RunOptions::new(()))
            .await
            .err()
            .unwrap_or_else(|| panic!("{mode}: the resumed handoff filter must be refused"));
        assert!(
            err.to_string()
                .contains("Server-managed conversations do not support handoff input filters"),
            "{mode}: {err}"
        );
    }
}
