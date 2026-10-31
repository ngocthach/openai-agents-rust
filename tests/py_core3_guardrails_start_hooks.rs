//! Ported from the Python SDK's `tests/test_start_hook_tools.py`: agent-start hooks that change
//! which tools the model sees, and approval-gated tools added by a hook.
//!
//! Python hooks reassign `agent.tools` in place. Rust hooks receive `&Agent`, so an agent's tool
//! set cannot change during a run. The "replace" cases therefore install the post-hook tool set
//! through `RunOptions::prepare_agent`, which runs at the same point (after the start of the run,
//! before the first model call), and the "disable" cases use the run context, as Python does.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentHooks, AgentsError, RunContext, RunHooks, RunOptions, RunState, Runner,
    ToolUseBehavior,
};
use serde_json::Value;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

/// The run context: the `enabled` flag that the `removed` tool's enablement check reads.
type Flag = Arc<AtomicBool>;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Change {
    Replace,
    Disable,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum HookKind {
    Run,
    Agent,
}

/// Records the agents it sees and applies `change` at agent start, for both hook kinds.
#[derive(Default)]
struct StartHookState {
    calls: Mutex<Vec<String>>,
    enablement: Mutex<Vec<bool>>,
}

struct RunStartHook {
    state: Arc<StartHookState>,
    change: Change,
}

#[async_trait]
impl RunHooks<Flag> for RunStartHook {
    async fn on_agent_start(&self, context: &RunContext<Flag>, agent: &Agent<Flag>) {
        self.state.calls.lock().unwrap().push(agent.name.clone());
        if self.change == Change::Disable {
            context.context().store(false, Ordering::SeqCst);
        }
    }
}

struct AgentStartHook {
    state: Arc<StartHookState>,
    change: Change,
}

#[async_trait]
impl AgentHooks<Flag> for AgentStartHook {
    async fn on_start(&self, context: &RunContext<Flag>, agent: &Agent<Flag>) {
        self.state.calls.lock().unwrap().push(agent.name.clone());
        if self.change == Change::Disable {
            context.context().store(false, Ordering::SeqCst);
        }
    }
}

fn tool_names(tools: &[openai_agents_sdk::models::ToolSpec]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|spec| match spec {
            openai_agents_sdk::models::ToolSpec::Function(f) => Some(f.name.clone()),
            _ => None,
        })
        .collect()
}

async fn start_hooks_update_model_tools_and_dispatch(
    streamed: bool,
    hook_kind: HookKind,
    change: Change,
    call_removed: bool,
) {
    disable_tracing();
    let effects = Arc::new(Mutex::new(Vec::<String>::new()));
    let state = Arc::new(StartHookState::default());
    let enabled: Flag = Arc::new(AtomicBool::new(true));

    let removed = {
        let effects = effects.clone();
        let state = state.clone();
        FunctionTool::new("removed", "", move |_c, _a: Value| {
            let effects = effects.clone();
            async move {
                effects.lock().unwrap().push("removed".into());
                Ok("removed".to_string())
            }
        })
        .with_is_enabled_fn({
            let state = state.clone();
            move |ctx: RunContext<Flag>, _agent: String| {
                let state = state.clone();
                async move {
                    let enabled = ctx.context().load(Ordering::SeqCst);
                    state.enablement.lock().unwrap().push(enabled);
                    enabled
                }
            }
        })
    };
    let permitted = {
        let effects = effects.clone();
        FunctionTool::new("permitted", "", move |_c, _a: Value| {
            let effects = effects.clone();
            async move {
                effects.lock().unwrap().push("permitted".into());
                Ok("permitted".to_string())
            }
        })
    };

    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![function_call(
            if call_removed { "removed" } else { "permitted" },
            "{}",
        )]))
        .unwrap();
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();

    // "replace" starts with `removed` only, and the hook installs `permitted`; "disable" starts
    // with both tools and the hook turns `removed` off through the context.
    let mut agent = Agent::<Flag>::new("request-local")
        .with_model_instance(model.clone())
        .with_tool(removed.clone());
    if change == Change::Disable {
        agent = agent.with_tool(permitted.clone());
    }
    if hook_kind == HookKind::Agent {
        agent = agent.with_hooks(Arc::new(AgentStartHook {
            state: state.clone(),
            change,
        }));
    }
    let agent = Arc::new(agent);

    let mut options = RunOptions::new(enabled.clone());
    if hook_kind == HookKind::Run {
        options = options.hooks(Arc::new(RunStartHook {
            state: state.clone(),
            change,
        }));
    }
    if change == Change::Replace {
        // The replacement is the same agent with the post-hook tool set, so it keeps its hooks.
        let mut replacement = Agent::<Flag>::new("request-local")
            .with_model_instance(model.clone())
            .with_tool(permitted.clone());
        if hook_kind == HookKind::Agent {
            replacement = replacement.with_hooks(Arc::new(AgentStartHook {
                state: state.clone(),
                change,
            }));
        }
        let permitted_only = Arc::new(replacement);
        options = options.prepare_agent(move |_agent| {
            let permitted_only = permitted_only.clone();
            async move { Ok(permitted_only) }
        });
    }

    let outcome: Result<String, AgentsError> = if streamed {
        let mut result = Runner::run_streamed_with_options(&agent, "go", options);
        let error = {
            let mut events = std::pin::pin!(result.stream_events());
            let mut error = None;
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                    break;
                }
            }
            error
        };
        match error {
            Some(e) => Err(e),
            None => Ok(result.final_output_text().unwrap_or_default()),
        }
    } else {
        Runner::run_with_options(&agent, "go", options)
            .await
            .map(|r| r.final_output_text().unwrap_or_default())
    };

    if call_removed {
        let err = outcome.expect_err("calling a removed tool must fail");
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { ref message, .. } if message.contains("removed")),
            "{err:?}"
        );
        assert!(effects.lock().unwrap().is_empty());
    } else {
        assert_eq!(outcome.unwrap(), "done");
        assert_eq!(*effects.lock().unwrap(), vec!["permitted".to_string()]);
        let calls = model.calls();
        assert_eq!(calls.len(), 2);
        for call in &calls {
            assert_eq!(
                tool_names(&call.request.tools),
                vec!["permitted".to_string()]
            );
        }
    }

    assert_eq!(
        *state.calls.lock().unwrap(),
        vec!["request-local".to_string()]
    );
    if change == Change::Disable {
        let enablement = state.enablement.lock().unwrap().clone();
        assert!(!enablement.is_empty());
        assert!(enablement.iter().all(|enabled| !enabled));
    }
}

macro_rules! start_hook_case {
    ($name:ident, $streamed:expr, $kind:expr, $change:expr, $removed:expr) => {
        #[tokio::test]
        async fn $name() {
            start_hooks_update_model_tools_and_dispatch($streamed, $kind, $change, $removed).await;
        }
    };
}

start_hook_case!(
    run_hook_replace_dispatches_permitted_non_streamed,
    false,
    HookKind::Run,
    Change::Replace,
    false
);
start_hook_case!(
    run_hook_replace_dispatches_permitted_streamed,
    true,
    HookKind::Run,
    Change::Replace,
    false
);
start_hook_case!(
    run_hook_replace_rejects_removed_non_streamed,
    false,
    HookKind::Run,
    Change::Replace,
    true
);
start_hook_case!(
    run_hook_replace_rejects_removed_streamed,
    true,
    HookKind::Run,
    Change::Replace,
    true
);
start_hook_case!(
    run_hook_disable_dispatches_permitted_non_streamed,
    false,
    HookKind::Run,
    Change::Disable,
    false
);
start_hook_case!(
    run_hook_disable_dispatches_permitted_streamed,
    true,
    HookKind::Run,
    Change::Disable,
    false
);
start_hook_case!(
    run_hook_disable_rejects_removed_non_streamed,
    false,
    HookKind::Run,
    Change::Disable,
    true
);
start_hook_case!(
    run_hook_disable_rejects_removed_streamed,
    true,
    HookKind::Run,
    Change::Disable,
    true
);
start_hook_case!(
    agent_hook_replace_dispatches_permitted_non_streamed,
    false,
    HookKind::Agent,
    Change::Replace,
    false
);
start_hook_case!(
    agent_hook_replace_dispatches_permitted_streamed,
    true,
    HookKind::Agent,
    Change::Replace,
    false
);
start_hook_case!(
    agent_hook_replace_rejects_removed_non_streamed,
    false,
    HookKind::Agent,
    Change::Replace,
    true
);
start_hook_case!(
    agent_hook_replace_rejects_removed_streamed,
    true,
    HookKind::Agent,
    Change::Replace,
    true
);
start_hook_case!(
    agent_hook_disable_dispatches_permitted_non_streamed,
    false,
    HookKind::Agent,
    Change::Disable,
    false
);
start_hook_case!(
    agent_hook_disable_dispatches_permitted_streamed,
    true,
    HookKind::Agent,
    Change::Disable,
    false
);
start_hook_case!(
    agent_hook_disable_rejects_removed_non_streamed,
    false,
    HookKind::Agent,
    Change::Disable,
    true
);
start_hook_case!(
    agent_hook_disable_rejects_removed_streamed,
    true,
    HookKind::Agent,
    Change::Disable,
    true
);

/// Installs the approval-gated tool at agent start, after the paused run was recorded. The
/// tool must be part of the agent the caller rebuilds; a tool that exists only in the hook's
/// run is not found on resume.
async fn approval_tool_requires_reconstruction(streamed: bool, rebuild_tools: bool) {
    disable_tracing();
    let effects = Arc::new(Mutex::new(Vec::<String>::new()));
    let approved_action = {
        let effects = effects.clone();
        FunctionTool::new("approved_action", "", move |_c, _a: Value| {
            let effects = effects.clone();
            async move {
                effects.lock().unwrap().push("executed".into());
                Ok("done".to_string())
            }
        })
        .with_needs_approval(true)
    };

    let hook_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    struct InstallCallRecorder(Arc<Mutex<Vec<String>>>);
    #[async_trait]
    impl RunHooks<()> for InstallCallRecorder {
        async fn on_agent_start(&self, _context: &RunContext<()>, agent: &Agent<()>) {
            self.0.lock().unwrap().push(agent.name.clone());
        }
    }
    let hooks: Arc<dyn RunHooks<()>> = Arc::new(InstallCallRecorder(hook_calls.clone()));

    let original_model = Arc::new(ScriptedModel::new());
    original_model
        .enqueue(ScriptedStep::output(vec![function_call(
            "approved_action",
            "{}",
        )]))
        .unwrap();
    let original = Arc::new(
        Agent::<()>::new("approval-agent")
            .with_model_instance(original_model)
            .with_tool(approved_action.clone())
            .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool),
    );

    let run = |agent: Arc<Agent<()>>, input: Option<RunState>| {
        let hooks = hooks.clone();
        async move {
            let options = RunOptions::new(()).hooks(hooks);
            if streamed {
                let mut result = match input {
                    Some(state) => Runner::run_streamed_with_options(&agent, state, options),
                    None => Runner::run_streamed_with_options(&agent, "go", options),
                };
                let error = {
                    let mut events = std::pin::pin!(result.stream_events());
                    let mut error = None;
                    while let Some(event) = events.next().await {
                        if let Err(e) = event {
                            error = Some(e);
                            break;
                        }
                    }
                    error
                };
                if let Some(error) = error {
                    return Err(error);
                }
                Ok(result.into_result().expect("completed result"))
            } else {
                match input {
                    Some(state) => Runner::run_with_options(&agent, state, options).await,
                    None => Runner::run_with_options(&agent, "go", options).await,
                }
            }
        }
    };

    let paused = run(original.clone(), None).await.unwrap();
    assert_eq!(paused.interruptions.len(), 1);
    assert!(effects.lock().unwrap().is_empty());

    let mut restored_builder = Agent::<()>::new("approval-agent")
        .with_model_instance(Arc::new(ScriptedModel::new()))
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
    if rebuild_tools {
        restored_builder = restored_builder.with_tool(approved_action.clone());
    }
    let restored_agent = Arc::new(restored_builder);
    let mut restored = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
    let approval = restored.get_interruptions()[0].clone();
    restored.approve(&approval, true);

    if rebuild_tools {
        let completed = run(restored_agent.clone(), Some(restored)).await.unwrap();
        assert_eq!(completed.final_output_text().as_deref(), Some("done"));
        assert_eq!(*effects.lock().unwrap(), vec!["executed".to_string()]);
    } else {
        let err = run(restored_agent.clone(), Some(restored))
            .await
            .expect_err("the tool is not part of the restored agent");
        assert!(
            err.to_string().contains("Tool approved_action not found"),
            "{err}"
        );
        assert!(effects.lock().unwrap().is_empty());
    }
    assert_eq!(
        *hook_calls.lock().unwrap(),
        vec!["approval-agent".to_string()]
    );
}

#[tokio::test]
async fn hook_added_approval_tool_is_executed_when_rebuilt_non_streamed() {
    approval_tool_requires_reconstruction(false, true).await;
}

#[tokio::test]
async fn hook_added_approval_tool_is_executed_when_rebuilt_streamed() {
    approval_tool_requires_reconstruction(true, true).await;
}

#[tokio::test]
async fn hook_added_approval_tool_must_be_reconstructed_non_streamed() {
    approval_tool_requires_reconstruction(false, false).await;
}

#[tokio::test]
async fn hook_added_approval_tool_must_be_reconstructed_streamed() {
    approval_tool_requires_reconstruction(true, false).await;
}
