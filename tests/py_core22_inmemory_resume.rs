//! What an in-memory `RunState` (from `to_state()`, never written to JSON) does when it is
//! resumed with the same agent, with a rebuilt agent graph, or after a JSON round trip.
//!
//! Python's `Runner.run(agent, state)` runs `state._current_agent`, the agent object the state
//! was taken from, and ignores the agent it is given for an in-memory state; the state's
//! decisions are bound to that object, so they always apply. A JSON-restored state resolves its
//! agents from the graph given to `RunState.from_string` and binds its decisions to them. In both
//! cases an approved call runs without asking again (checked against the Python SDK for every
//! row below, streamed and not, per-call and sticky, single agent and handoff target).
//!
//! A Rust `RunState` holds no agent: every resume runs the graph it is given, found by name, so
//! a rebuilt graph runs its own tool where Python runs the original one. The decision still
//! applies and nothing is asked again, which is the outcome Python shows.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Handoff, RunState, Runner};

type Log = Arc<Mutex<Vec<String>>>;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn entries(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

/// A tool named `operate` that needs approval and records `label` when it runs.
fn operate(label: &str, executed: &Log) -> FunctionTool<()> {
    let executed = executed.clone();
    let label = label.to_string();
    FunctionTool::new("operate", "Operate.", move |_c, _a: NoArgs| {
        let executed = executed.clone();
        let label = label.clone();
        async move {
            executed.lock().unwrap().push(label.clone());
            Ok(label)
        }
    })
    .with_needs_approval(true)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Resume {
    /// The very `Arc<Agent>` the state was taken from (Python's `memory-same`).
    Same,
    /// A rebuilt graph with the same names and configuration, in-memory state.
    Rebuilt,
    /// A rebuilt graph, state written to JSON and read back.
    Json,
}

/// The agent whose tool pauses: a single agent `owner`, or `dest` reached from `root` through a
/// handoff. `first_run` scripts the whole conversation; otherwise only the turns after the
/// pause, which is all a rebuilt agent sees.
fn build(
    label: &str,
    executed: &Log,
    graph: bool,
    always: bool,
    first_run: bool,
) -> Arc<Agent<()>> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    if first_run {
        model.add_turn(vec![function_call_with_id("operate", "{}", "c1")]);
    }
    if always {
        model.add_turn(vec![function_call_with_id("operate", "{}", "c2")]);
    }
    model.add_turn(vec![assistant_message("done")]);
    let name = if graph { "dest" } else { "owner" };
    let agent = Agent::<()>::new(name)
        .with_tool(operate(label, executed))
        .with_model_instance(model);
    if !graph {
        return Arc::new(agent);
    }
    let root_model = Arc::new(FakeModel::new());
    if first_run {
        root_model.add_turn(vec![function_call_with_id("to_dest", "{}", "h")]);
    }
    Arc::new(
        Agent::<()>::new("root")
            .with_handoff(Handoff::new(Arc::new(agent)).with_tool_name("to_dest"))
            .with_model_instance(root_model),
    )
}

async fn run(agent: &Arc<Agent<()>>, state: RunState, streamed: bool) -> (usize, Option<String>) {
    if streamed {
        let mut stream = Runner::run_streamed(agent.clone(), state);
        {
            let mut events = std::pin::pin!(stream.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        (stream.interruptions().len(), stream.final_output_text())
    } else {
        let result = Runner::run(agent.clone(), state).await.unwrap();
        (result.interruptions.len(), result.final_output_text())
    }
}

/// Every row of the Python probe: an approved call (per call, or sticky with a second call)
/// runs once the state resumes, and the run finishes without asking again.
#[tokio::test]
async fn an_approved_in_memory_state_runs_without_asking_again() {
    for graph in [false, true] {
        for streamed in [false, true] {
            for always in [false, true] {
                for mode in [Resume::Same, Resume::Rebuilt, Resume::Json] {
                    let label =
                        format!("{mode:?} graph={graph} streamed={streamed} always={always}");
                    let executed: Log = Arc::default();
                    let original = build("orig", &executed, graph, always, true);
                    let first = Runner::run(original.clone(), "start").await.unwrap();
                    assert_eq!(first.interruptions.len(), 1, "{label}");
                    let mut state = first.to_state();
                    state.approve(&first.interruptions[0], always);
                    let (agent, state) = match mode {
                        Resume::Same => (original.clone(), state),
                        Resume::Rebuilt => {
                            (build("rebuilt", &executed, graph, always, false), state)
                        }
                        Resume::Json => (
                            build("rebuilt", &executed, graph, always, false),
                            RunState::from_json(state.to_json().unwrap()).unwrap(),
                        ),
                    };
                    let (interruptions, final_output) = run(&agent, state, streamed).await;
                    assert_eq!(interruptions, 0, "{label}");
                    assert_eq!(final_output.as_deref(), Some("done"), "{label}");
                    // Python runs the original tool for an in-memory state; Rust runs the graph
                    // it is given, so a rebuilt graph runs its own.
                    let ran = if mode == Resume::Same {
                        "orig"
                    } else {
                        "rebuilt"
                    };
                    let calls = if always { 2 } else { 1 };
                    assert_eq!(entries(&executed), vec![ran.to_string(); calls], "{label}");
                }
            }
        }
    }
}
