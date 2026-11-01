//! Ports of the same-name cases of `tests/test_agent_scoped_approvals.py`: two distinct agents
//! sharing a name own separate approval decisions, told apart by the identity key a snapshot
//! already uses (`same-name` for the starting agent, `same-name#2` for the other).

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::items::{OutputItem, RunInput, RunItemKind};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunContext, RunResult, RunState, Runner};
use serde_json::json;

type Log = Arc<Mutex<Vec<String>>>;

/// A late-bound agent: filled in after the handoff that targets it is built.
type AgentSlot = Arc<OnceLock<Arc<Agent<()>>>>;

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn setup() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn model(turns: Vec<Vec<OutputItem>>) -> Arc<FakeModel> {
    let m = Arc::new(FakeModel::new());
    m.add_turns(turns);
    m
}

fn recording_tool(name: &str, output: &str, executed: &Log) -> FunctionTool<()> {
    let executed = executed.clone();
    let output = output.to_string();
    FunctionTool::new_no_args(name, "", move |_ctx| {
        let executed = executed.clone();
        let output = output.clone();
        async move {
            executed.lock().unwrap().push(output.clone());
            Ok(output)
        }
    })
}

/// A policy that records each call id it is asked about and always requires approval.
fn recording_policy(tool: FunctionTool<()>, policies: &Log) -> FunctionTool<()> {
    let policies = policies.clone();
    tool.with_needs_approval_fn(move |_ctx, _args, call_id| {
        let policies = policies.clone();
        async move {
            policies.lock().unwrap().push(call_id);
            true
        }
    })
}

/// A handoff back to an agent built later, so two agents can hand off to each other.
fn late_handoff(tool_name: &str, target_name: &str) -> (Handoff<()>, AgentSlot) {
    let slot: AgentSlot = Arc::new(OnceLock::new());
    let target = slot.clone();
    let handoff = Handoff::custom(
        tool_name,
        "Hand back.",
        target_name,
        json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false}),
        move |_ctx, _args| {
            let target = target.clone();
            async move { Ok(target.get().expect("handoff target set").clone()) }
        },
    );
    (handoff, slot)
}

async fn run(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    streamed: bool,
    ctx: Option<RunContext<()>>,
) -> openai_agents_sdk::Result<RunResult<()>> {
    let options = match ctx {
        Some(ctx) => RunOptions::with_run_context(ctx),
        None => RunOptions::default(),
    };
    if streamed {
        let mut stream = Runner::run_streamed_with_options(agent, input, options);
        stream.finish().await?;
        Ok(stream.into_result().expect("streamed result"))
    } else {
        Runner::run_with_options(agent, input, options).await
    }
}

fn roundtrip(state: &RunState) -> RunState {
    RunState::from_json_str(&state.to_json_string().unwrap()).unwrap()
}

/// `(agent, call_id, output)` of every tool output item.
fn outputs(result: &RunResult<()>) -> Vec<(String, String, String)> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput {
                raw_item, output, ..
            } => Some((
                item.agent.clone(),
                raw_item["call_id"].as_str().unwrap_or_default().to_string(),
                output
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| output.to_string()),
            )),
            _ => None,
        })
        .collect()
}

/// Python: `test_sticky_approval_stays_with_agent_across_handoffs` with both agents named
/// `same-name` (streamed x serialized x approve).
#[tokio::test]
async fn sticky_approval_stays_with_the_same_named_agent_across_handoffs() {
    setup();
    for streamed in [false, true] {
        for serialized in [false, true] {
            for approve in [false, true] {
                let ctx = format!("streamed={streamed} serialized={serialized} approve={approve}");
                let executed = log();
                let policies = log();
                let first_tool =
                    recording_policy(recording_tool("operate", "first", &executed), &policies);
                let second_tool =
                    recording_policy(recording_tool("operate", "second", &executed), &policies);
                let (to_first, first_slot) = late_handoff("to_first", "same-name");
                let second = Arc::new(
                    Agent::<()>::new("same-name")
                        .with_tool(second_tool)
                        .with_handoff(to_first)
                        .with_model_instance(model(vec![
                            vec![function_call_with_id("operate", "{}", "second-1")],
                            vec![function_call_with_id("operate", "{}", "second-2")],
                            vec![function_call_with_id("to_first", "{}", "handoff-2")],
                        ])),
                );
                let first = Arc::new(
                    Agent::<()>::new("same-name")
                        .with_tool(first_tool)
                        .with_handoff(Handoff::new(second.clone()).with_tool_name("to_second"))
                        .with_model_instance(model(vec![
                            vec![function_call_with_id("operate", "{}", "first-1")],
                            vec![function_call_with_id("to_second", "{}", "handoff-1")],
                            vec![function_call_with_id("operate", "{}", "first-2")],
                            vec![assistant_message("done")],
                        ])),
                );
                first_slot.set(first.clone()).unwrap();

                let result = run(&first, "start", streamed, None).await.unwrap();
                let mut state = result.to_state();
                if approve {
                    state.approve(&result.interruptions[0], true);
                } else {
                    state.reject(&result.interruptions[0], true, Some("first denied".into()));
                }
                if serialized {
                    state = roundtrip(&state);
                }
                let result = run(&first, state, streamed, None).await.unwrap();
                assert_eq!(result.interruptions.len(), 1, "{ctx}");
                // The second agent is told apart from the first by its identity key.
                assert_eq!(result.interruptions[0].agent, "same-name#2", "{ctx}");
                assert_eq!(*policies.lock().unwrap(), ["first-1", "second-1"], "{ctx}");
                let want: &[&str] = if approve { &["first"] } else { &[] };
                assert_eq!(*executed.lock().unwrap(), want, "{ctx}");

                let mut state = result.to_state();
                if approve {
                    state.reject(&result.interruptions[0], true, Some("second denied".into()));
                } else {
                    state.approve(&result.interruptions[0], true);
                }
                if serialized {
                    state = roundtrip(&state);
                }
                let result = run(&first, state, streamed, None).await.unwrap();
                assert_eq!(result.final_output_text().as_deref(), Some("done"), "{ctx}");
                assert!(result.interruptions.is_empty(), "{ctx}");
                assert_eq!(*policies.lock().unwrap(), ["first-1", "second-1"], "{ctx}");
                let want = if approve {
                    ["first", "first"]
                } else {
                    ["second", "second"]
                };
                assert_eq!(*executed.lock().unwrap(), want, "{ctx}");
                let (rejected_calls, reason) = if approve {
                    (["second-1", "second-2"], "second denied")
                } else {
                    (["first-1", "first-2"], "first denied")
                };
                let rejections: Vec<String> = outputs(&result)
                    .into_iter()
                    .filter(|(_, call_id, _)| rejected_calls.contains(&call_id.as_str()))
                    .map(|(_, _, o)| o)
                    .collect();
                assert_eq!(rejections, [reason, reason], "{ctx}");
            }
        }
    }
}

/// A snapshot written for same-named agents names every owner by identity, and still reads
/// back when the owner carries only a name (a snapshot from before identities existed).
#[tokio::test]
async fn a_snapshot_names_same_named_owners_and_accepts_name_only_owners() {
    setup();
    let executed = log();
    let policies = log();
    let first_tool = recording_policy(recording_tool("operate", "first", &executed), &policies);
    let second = Arc::new(
        Agent::<()>::new("same-name")
            .with_tool(recording_policy(
                recording_tool("operate", "second", &executed),
                &policies,
            ))
            .with_model_instance(model(vec![vec![function_call_with_id(
                "operate", "{}", "second-1",
            )]])),
    );
    let first = Arc::new(
        Agent::<()>::new("same-name")
            .with_tool(first_tool)
            .with_handoff(Handoff::new(second.clone()).with_tool_name("to_second"))
            .with_model_instance(model(vec![
                vec![function_call_with_id("operate", "{}", "first-1")],
                vec![function_call_with_id("to_second", "{}", "handoff-1")],
            ])),
    );
    let result = run(&first, "start", false, None).await.unwrap();
    let mut state = result.to_state();
    state.approve(&result.interruptions[0], true);
    let result = run(&first, state, false, None).await.unwrap();
    assert_eq!(result.interruptions[0].agent, "same-name#2");
    let mut state = result.to_state();
    state.approve(&result.interruptions[0], true);

    let wire = state.to_json().unwrap();
    let owners: Vec<(String, Option<String>)> = wire["context"]["function_tool_approvals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            (
                entry["agent"]["name"].as_str().unwrap().to_string(),
                entry["agent"]["identity"].as_str().map(str::to_string),
            )
        })
        .collect();
    assert_eq!(
        owners,
        [
            ("same-name".to_string(), None),
            ("same-name".to_string(), Some("same-name#2".to_string())),
        ]
    );

    // An older snapshot names the owner by name alone: it is the first agent of that name.
    let mut legacy = wire.clone();
    for entry in legacy["context"]["function_tool_approvals"]
        .as_array_mut()
        .unwrap()
    {
        entry["agent"] = json!({"name": entry["agent"]["name"].clone()});
    }
    let restored = RunState::from_json(legacy).unwrap();
    assert_eq!(
        restored
            .approvals
            .0
            .keys()
            .filter(|key| key.ends_with("operate"))
            .count(),
        1,
        "both decisions fold onto the one owner a name-only reference can name"
    );
}
