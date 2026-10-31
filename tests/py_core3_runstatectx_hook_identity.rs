//! Port of `tests/test_run_state_hook_identity.py`: hook lifecycle state must not redirect a
//! resumed run or its approval owner, and same-named agents restore through their identity key.
//!
//! Restore is not a separate step in Rust: `RunState::from_json` only parses, and the agent graph
//! is bound when the run resumes. So a Python `await RunState.from_json(root, payload)` that raises
//! becomes a resume that errors before any tool runs; the cases assert the same message and that
//! no tool ran.
//!
//! Python-only cases (not ported):
//! - the `hook_type` and `hook_category` discriminators of `test_legacy_hook_order_is_preserved...`:
//!   they distinguish hook classes by Python type name and dataclass-ness. Rust hook identity
//!   records only that a hook is present, so the discriminators do not exist.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::{RunInput, ToolApprovalItem};
use openai_agents_sdk::lifecycle::AgentHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunContext, handoff};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

/// Counts `on_start` calls; `name` stands in for Python's per-instance attributes.
#[derive(Default)]
struct CountingHooks {
    starts: AtomicUsize,
}

impl CountingHooks {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            starts: AtomicUsize::new(0),
        })
    }
    fn starts(&self) -> usize {
        self.starts.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl AgentHooks<()> for CountingHooks {
    async fn on_start(&self, _ctx: &RunContext<()>, _agent: &Agent<()>) {
        self.starts.fetch_add(1, Ordering::SeqCst);
    }
}

/// The three-agent graph of the Python tests: `root` hands off to `x` and `y`, and both children
/// run an approval-gated `operate` tool that records their owner.
#[derive(Clone)]
struct Layout {
    restoring: bool,
    unique: bool,
    /// Handoff targets in the order the root lists them (`"x"` or `"y"`).
    root_targets: Vec<&'static str>,
    root_name: String,
    x_hooks: Option<Arc<CountingHooks>>,
    y_hooks: Option<Arc<CountingHooks>>,
    x_instructions: String,
    y_instructions: String,
    x_description: Option<String>,
    y_description: Option<String>,
}

impl Layout {
    fn new(restoring: bool, unique: bool) -> Self {
        Self {
            restoring,
            unique,
            root_targets: if restoring {
                vec!["y", "x"]
            } else {
                vec!["x", "y"]
            },
            root_name: "root".into(),
            x_hooks: Some(CountingHooks::new()),
            y_hooks: Some(CountingHooks::new()),
            x_instructions: "b".into(),
            y_instructions: "a".into(),
            x_description: None,
            y_description: None,
        }
    }
}

struct Graph {
    root: Arc<Agent<()>>,
    y: Arc<Agent<()>>,
    x_model: Arc<FakeModel>,
    y_hooks: Option<Arc<CountingHooks>>,
}

fn child(
    owner: &str,
    layout: &Layout,
    effects: &Log,
    hooks: Option<Arc<CountingHooks>>,
    instructions: &str,
    description: Option<&str>,
) -> (Arc<Agent<()>>, Arc<FakeModel>) {
    let model = Arc::new(FakeModel::new());
    let mut turns = Vec::new();
    if !layout.restoring {
        turns.push(vec![function_call_with_id("operate", "{}", "saved-call")]);
    }
    turns.push(vec![function_call_with_id("operate", "{}", "follow-up")]);
    turns.push(vec![assistant_message(format!("done-{owner}"))]);
    model.add_turns(turns);

    let name = if layout.unique {
        format!("sub-{owner}")
    } else {
        "sub".to_string()
    };
    let effects = effects.clone();
    let owner_name = owner.to_string();
    let operate = FunctionTool::<()>::new("operate", "Records the owner", move |_c, _a: Value| {
        let effects = effects.clone();
        let owner_name = owner_name.clone();
        async move {
            effects.lock().unwrap().push(owner_name.clone());
            Ok(owner_name)
        }
    })
    .with_needs_approval(true);

    let mut agent = Agent::<()>::new(name)
        .with_instructions(instructions)
        .with_tools(vec![operate])
        .with_model_instance(model.clone());
    if let Some(description) = description {
        agent = agent.with_handoff_description(description);
    }
    if let Some(hooks) = hooks {
        agent = agent.with_hooks(hooks);
    }
    (Arc::new(agent), model)
}

fn build_graph(effects: &Log, layout: &Layout) -> Graph {
    let (x, x_model) = child(
        "x",
        layout,
        effects,
        layout.x_hooks.clone(),
        &layout.x_instructions,
        layout.x_description.as_deref(),
    );
    let (y, _y_model) = child(
        "y",
        layout,
        effects,
        layout.y_hooks.clone(),
        &layout.y_instructions,
        layout.y_description.as_deref(),
    );
    let root_model = Arc::new(FakeModel::new());
    root_model.add_turn(vec![function_call_with_id("to_y", "{}", "handoff")]);
    let handoffs: Vec<_> = layout
        .root_targets
        .iter()
        .map(|target| {
            let agent = if *target == "x" { x.clone() } else { y.clone() };
            handoff(agent).with_tool_name(format!("to_{target}"))
        })
        .collect();
    let root = Arc::new(
        Agent::<()>::new(layout.root_name.clone())
            .with_handoffs(handoffs)
            .with_model_instance(root_model),
    );
    Graph {
        root,
        y,
        x_model,
        y_hooks: layout.y_hooks.clone(),
    }
}

/// What a finished run leaves behind, for both the streamed and the plain run.
struct Outcome {
    final_output: Option<String>,
    last_agent: Arc<Agent<()>>,
    interruptions: Vec<ToolApprovalItem>,
    state: RunState,
}

async fn run_graph(
    root: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    streamed: bool,
) -> Result<Outcome, AgentsError> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let config = RunConfig::new();
    if streamed {
        let mut result = openai_agents_sdk::Runner::run_streamed_with_options(
            root,
            input,
            RunOptions::new(()).run_config(config),
        );
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        let run = result
            .result()
            .expect("a completed stream holds its result");
        return Ok(Outcome {
            final_output: run.final_output_text(),
            last_agent: run.last_agent.clone(),
            interruptions: run.interruptions.clone(),
            state: run.to_state(),
        });
    }
    let run = openai_agents_sdk::Runner::run_with_options(
        root,
        input,
        RunOptions::new(()).run_config(config),
    )
    .await?;
    Ok(Outcome {
        final_output: run.final_output_text(),
        last_agent: run.last_agent.clone(),
        interruptions: run.interruptions.clone(),
        state: run.to_state(),
    })
}

/// The first run: root hands off to `y`, which parks on its approval.
async fn first_run(graph: &Graph, streamed: bool) -> Outcome {
    let first = run_graph(&graph.root, "start", streamed).await.unwrap();
    assert!(Arc::ptr_eq(&first.last_agent, &graph.y));
    first
}

async fn hook_reset_case(streamed: bool, always: bool) {
    {
        let effects: Log = Arc::default();
        let graph = build_graph(&effects, &Layout::new(false, false));
        let first = first_run(&graph, streamed).await;
        assert_eq!(graph.y_hooks.as_ref().unwrap().starts(), 1);
        let mut state = first.state.clone();
        state.approve(&first.interruptions[0], always);
        let payload = state.to_json().unwrap();

        let graph = build_graph(&effects, &Layout::new(true, false));
        let restored = RunState::from_json(payload).unwrap();
        let resumed = run_graph(&graph.root, restored, streamed).await.unwrap();
        assert!(Arc::ptr_eq(&resumed.last_agent, &graph.y));
        let expected: Vec<String> = if always {
            vec!["y".into(), "y".into()]
        } else {
            vec!["y".into()]
        };
        assert_eq!(*effects.lock().unwrap(), expected);
        assert!(graph.x_model.calls().is_empty());
        if always {
            assert_eq!(resumed.final_output.as_deref(), Some("done-y"));
            assert!(resumed.interruptions.is_empty());
        } else {
            assert_eq!(resumed.interruptions.len(), 1);
            assert_eq!(resumed.interruptions[0].call_id(), "follow-up");
        }
    }
}

#[tokio::test]
async fn hook_reset_preserves_resumed_execution_and_approval_owner_per_call() {
    hook_reset_case(false, false).await;
}

#[tokio::test]
async fn hook_reset_preserves_resumed_execution_and_approval_owner_always() {
    hook_reset_case(false, true).await;
}

#[tokio::test]
async fn hook_reset_preserves_resumed_execution_and_approval_owner_streamed_per_call() {
    hook_reset_case(true, false).await;
}

#[tokio::test]
async fn hook_reset_preserves_resumed_execution_and_approval_owner_streamed_always() {
    hook_reset_case(true, true).await;
}

/// Legacy (`1.18` / `1.20`) snapshots with duplicate names and hooks: refused unless names are
/// unique, and refused before any tool runs.
async fn legacy_identity_case(unique: bool, version: &str) {
    let effects: Log = Arc::default();
    let graph = build_graph(&effects, &Layout::new(false, unique));
    let first = first_run(&graph, false).await;
    let mut state = first.state.clone();
    state.approve(&first.interruptions[0], true);
    let mut payload = state.to_json().unwrap();
    assert_eq!(
        payload["$schemaVersion"],
        openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION
    );
    payload["$schemaVersion"] = json!(version);

    let graph = build_graph(&effects, &Layout::new(true, unique));
    let restored = RunState::from_json(payload).unwrap();
    let outcome = run_graph(&graph.root, restored, false).await;
    if unique {
        let resumed = outcome.unwrap();
        assert!(Arc::ptr_eq(&resumed.last_agent, &graph.y));
        assert_eq!(
            *effects.lock().unwrap(),
            vec!["y".to_string(), "y".to_string()]
        );
    } else {
        let err = outcome
            .err()
            .expect("legacy duplicate-agent identities must be refused");
        assert!(
            err.to_string()
                .contains("legacy duplicate-agent identities with hooks"),
            "unexpected error: {err}"
        );
        assert!(effects.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn legacy_hook_identity_restore_refuses_duplicate_names_1_18() {
    legacy_identity_case(false, "1.18").await;
}

#[tokio::test]
async fn legacy_hook_identity_restore_refuses_duplicate_names_1_20() {
    legacy_identity_case(false, "1.20").await;
}

#[tokio::test]
async fn legacy_hook_identity_restore_accepts_unique_names_1_18() {
    legacy_identity_case(true, "1.18").await;
}

#[tokio::test]
async fn legacy_hook_identity_restore_accepts_unique_names_1_20() {
    legacy_identity_case(true, "1.20").await;
}

#[tokio::test]
async fn legacy_root_priority_does_not_depend_on_hook_state() {
    let effects: Log = Arc::default();
    // Only the root and one non-root agent share a name; root priority cannot flip.
    let mut layout = Layout::new(false, false);
    layout.root_name = "sub".into();
    layout.root_targets = vec!["y"];
    let graph = build_graph(&effects, &layout);
    let first = first_run(&graph, false).await;
    let mut payload = first.state.to_json().unwrap();
    payload["$schemaVersion"] = json!("1.18");

    let mut layout = Layout::new(true, false);
    layout.root_name = "sub".into();
    layout.root_targets = vec!["y"];
    let graph = build_graph(&effects, &layout);
    let mut restored = RunState::from_json(payload).unwrap();
    let pending = restored.get_interruptions()[0].clone();
    assert_eq!(pending.agent, "sub");
    restored.approve(&pending, true);
    let resumed = run_graph(&graph.root, restored, false).await.unwrap();
    assert!(Arc::ptr_eq(&resumed.last_agent, &graph.y));
    assert_eq!(
        *effects.lock().unwrap(),
        vec!["y".to_string(), "y".to_string()]
    );
}

#[tokio::test]
async fn legacy_single_hook_duplicate_group_preserves_presence_order() {
    let effects: Log = Arc::default();
    let mut layout = Layout::new(false, false);
    layout.x_hooks = None;
    let graph = build_graph(&effects, &layout);
    let first = first_run(&graph, false).await;
    let mut state = first.state.clone();
    state.approve(&first.interruptions[0], true);
    let payload = state.to_json().unwrap();
    // The legacy writer sorted JSON null before the normalized hook object.
    assert_eq!(
        payload["current_agent"],
        json!({"name": "sub", "identity": "sub#2"})
    );
    let mut payload = payload;
    payload["$schemaVersion"] = json!("1.18");

    let mut layout = Layout::new(true, false);
    layout.x_hooks = None;
    let graph = build_graph(&effects, &layout);
    let restored = RunState::from_json(payload).unwrap();
    let resumed = run_graph(&graph.root, restored, false).await.unwrap();
    assert!(Arc::ptr_eq(&resumed.last_agent, &graph.y));
    assert_eq!(
        *effects.lock().unwrap(),
        vec!["y".to_string(), "y".to_string()]
    );
}

#[tokio::test]
async fn legacy_hook_order_is_preserved_by_handoff_description() {
    let effects: Log = Arc::default();
    let configure = |restoring: bool| {
        let mut layout = Layout::new(restoring, false);
        layout.x_description = Some("a".into());
        layout.y_description = Some("b".into());
        layout
    };
    let graph = build_graph(&effects, &configure(false));
    let first = first_run(&graph, false).await;
    let mut state = first.state.clone();
    state.approve(&first.interruptions[0], true);
    let payload = state.to_json().unwrap();
    // Both descriptions precede hook instance state in the released writer.
    assert_eq!(
        payload["current_agent"],
        json!({"name": "sub", "identity": "sub#2"})
    );
    let mut payload = payload;
    payload["$schemaVersion"] = json!("1.18");

    let graph = build_graph(&effects, &configure(true));
    let restored = RunState::from_json(payload).unwrap();
    let resumed = run_graph(&graph.root, restored, false).await.unwrap();
    assert!(Arc::ptr_eq(&resumed.last_agent, &graph.y));
    assert_eq!(
        *effects.lock().unwrap(),
        vec!["y".to_string(), "y".to_string()]
    );
}

/// Two same-named agents whose only difference is hook instance state cannot be told apart
/// once the hooks are gone, so the snapshot must not be written.
#[tokio::test]
async fn hook_only_discriminator_rejects_serialization() {
    let effects: Log = Arc::default();
    let mut layout = Layout::new(false, false);
    layout.x_instructions = "same".into();
    layout.y_instructions = "same".into();
    layout.x_hooks = Some(CountingHooks::new());
    layout.y_hooks = Some(CountingHooks::new());
    let graph = build_graph(&effects, &layout);
    let first = first_run(&graph, false).await;
    let mut state = first.state.clone();
    state.approve(&first.interruptions[0], true);

    let err = state
        .to_json()
        .expect_err("indistinguishable same-name agents must not serialize");
    assert!(
        err.to_string()
            .contains("cannot distinguish same-name agents"),
        "{err}"
    );
    assert!(effects.lock().unwrap().is_empty());
}

/// A reconstructed graph that removed its durable distinction must not use traversal order to
/// pick the owner of the saved operation.
#[tokio::test]
async fn restore_rejects_newly_indistinguishable_hook_owners() {
    let effects: Log = Arc::default();
    let graph = build_graph(&effects, &Layout::new(false, false));
    let first = first_run(&graph, false).await;
    let mut state = first.state.clone();
    state.approve(&first.interruptions[0], true);
    let payload = state.to_json().unwrap();

    let mut layout = Layout::new(true, false);
    layout.x_instructions = "same".into();
    layout.y_instructions = "same".into();
    let graph = build_graph(&effects, &layout);
    let restored = RunState::from_json(payload).unwrap();
    let err = run_graph(&graph.root, restored, false)
        .await
        .err()
        .expect("indistinguishable same-name owners must be refused");
    assert!(
        err.to_string()
            .contains("cannot distinguish same-name agents"),
        "{err}"
    );
    assert!(effects.lock().unwrap().is_empty());
}
