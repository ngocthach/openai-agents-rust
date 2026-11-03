//! Port of `tests/test_start_hook_tool_concurrency.py`.
//!
//! The Python file guards one hazard: start hooks, handoff callbacks and enablement checks may
//! assign `agent.tools` on an `Agent` that several runs share, so the SDK registers each run's
//! view of the agent and rejects every overlapping run with "concurrent runs" once one of them
//! changes the shared list. A Rust `Agent` is read-only while it runs: hooks receive `&Agent`,
//! and a run that wants a different tool set builds a replacement per run through
//! `RunOptions::prepare_agent`, so there is no shared mutable list to conflict on. The 35
//! mutation-conflict cases (shared-list replacement, in-place append, MCP discovery mutation,
//! failed or cancelled hooks, restore-after-replace, `run_sync` threads, model waits, resumed
//! agents, sequential guardrails and handoff-target registration) therefore have no Rust
//! counterpart. The cases that survive are the isolation guarantees: context-driven enablement
//! stays per run, and a per-run tool replacement never reaches a concurrent run.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::RunInput;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::models::interface::ToolSpec;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentModel, RunContext, RunOptions, RunResult, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::{Barrier, Notify};

#[derive(Deserialize, JsonSchema, Default)]
struct NoArgs {}

fn privileged(effects: &Arc<Mutex<Vec<String>>>) -> FunctionTool<String> {
    let effects = effects.clone();
    FunctionTool::new(
        "privileged",
        "a synthetic privileged effect",
        move |ctx: openai_agents_sdk::ToolContext<String>, _a: NoArgs| {
            let effects = effects.clone();
            async move {
                effects.lock().unwrap().push(ctx.context().clone());
                Ok("synthetic effect".to_string())
            }
        },
    )
}

fn final_model() -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message("done")])])
            .unwrap(),
    )
}

fn tool_names(model: &ScriptedModel, call: usize) -> Vec<String> {
    model.calls()[call]
        .request
        .tools
        .iter()
        .filter_map(|tool| match tool {
            ToolSpec::Function(spec) => Some(spec.name.clone()),
            ToolSpec::Hosted { .. } => None,
        })
        .collect()
}

/// Run `agent` with `model` for `role`, streamed or not, like the Python `_run` helper.
async fn run(
    agent: &Agent<String>,
    model: &Arc<ScriptedModel>,
    role: &str,
    streamed: bool,
    options: RunOptions<String>,
) -> RunResult<String> {
    let config = RunConfig::<String> {
        model: Some(AgentModel::Instance(model.clone())),
        tracing_disabled: true,
        ..Default::default()
    };
    let options = options.run_config(config);
    let _ = role;
    let result = if streamed {
        let mut result =
            Runner::run_streamed_with_options(agent.clone(), RunInput::from("go"), options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        result.into_result().expect("a finished streamed run")
    } else {
        Runner::run_with_options(agent.clone(), RunInput::from("go"), options)
            .await
            .unwrap()
    };
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    result
}

/// Makes both runs reach their start hook before either continues.
struct BothStarted {
    barrier: Barrier,
}

#[async_trait::async_trait]
impl RunHooks<String> for BothStarted {
    async fn on_agent_start(&self, _context: &RunContext<String>, _agent: &Agent<String>) {
        self.barrier.wait().await;
    }
}

/// Python: `test_shared_agent_with_context_enablement_can_run_concurrently[False, True]`.
#[tokio::test]
async fn shared_agent_with_context_enablement_can_run_concurrently() {
    for streamed in [false, true] {
        let effects = Arc::new(Mutex::new(Vec::new()));
        let tool = privileged(&effects)
            .with_is_enabled_fn(|ctx, _agent| async move { ctx.context() == "high" });
        let agent = Agent::<String>::new("shared").with_tool(tool);
        let hooks = Arc::new(BothStarted {
            barrier: Barrier::new(2),
        });
        let (low_model, high_model) = (final_model(), final_model());

        let low_options = RunOptions::new("low".to_string()).hooks(hooks.clone());
        let high_options = RunOptions::new("high".to_string()).hooks(hooks.clone());
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(
                run(&agent, &low_model, "low", streamed, low_options),
                run(&agent, &high_model, "high", streamed, high_options),
            )
        })
        .await
        .expect("both runs reach their start hook concurrently");

        assert!(tool_names(&low_model, 0).is_empty(), "streamed={streamed}");
        assert_eq!(
            tool_names(&high_model, 0),
            vec!["privileged"],
            "streamed={streamed}"
        );
        assert_eq!(agent.tools.len(), 1, "the shared agent is untouched");
    }
}

/// Holds the first run in its start hook until the survivor has started, and the survivor until
/// the first run has finished.
struct Overlap {
    first_started: Notify,
    survivor_started: Notify,
    first_finished: Notify,
}

#[async_trait::async_trait]
impl RunHooks<String> for Overlap {
    async fn on_agent_start(&self, context: &RunContext<String>, _agent: &Agent<String>) {
        match context.context().as_str() {
            "first" => {
                self.first_started.notify_one();
                self.survivor_started.notified().await;
            }
            "survivor" => {
                self.survivor_started.notify_one();
                self.first_finished.notified().await;
            }
            _ => {}
        }
    }
}

/// Python: `test_surviving_run_can_update_tools_after_harmless_overlap[False, True]`. The
/// survivor's tool set is a per-run replacement, so the first run never sees it and the shared
/// agent is left as it was.
#[tokio::test]
async fn surviving_run_can_update_tools_after_harmless_overlap() {
    for streamed in [false, true] {
        let effects = Arc::new(Mutex::new(Vec::new()));
        let added = privileged(&effects);
        let agent = Agent::<String>::new("shared");
        let hooks = Arc::new(Overlap {
            first_started: Notify::new(),
            survivor_started: Notify::new(),
            first_finished: Notify::new(),
        });
        let first_model = final_model();
        let survivor_model = Arc::new(
            ScriptedModel::with_steps(vec![
                ScriptedStep::output(vec![function_call_with_id("privileged", "{}", "call_1")]),
                ScriptedStep::output(vec![assistant_message("done")]),
            ])
            .unwrap(),
        );

        let first_options = RunOptions::new("first".to_string()).hooks(hooks.clone());
        let survivor_options = RunOptions::new("survivor".to_string())
            .hooks(hooks.clone())
            .prepare_agent(move |current| {
                let added = added.clone();
                async move { Ok(Arc::new(current.as_ref().clone().with_tool(added))) }
            });

        let first = async {
            let result = run(&agent, &first_model, "first", streamed, first_options).await;
            hooks.first_finished.notify_one();
            result
        };
        let survivor = async {
            hooks.first_started.notified().await;
            run(
                &agent,
                &survivor_model,
                "survivor",
                streamed,
                survivor_options,
            )
            .await
        };
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(first, survivor)
        })
        .await
        .expect("the overlapping runs finish");

        assert!(
            tool_names(&first_model, 0).is_empty(),
            "streamed={streamed}"
        );
        assert_eq!(
            tool_names(&survivor_model, 0),
            vec!["privileged"],
            "streamed={streamed}"
        );
        assert_eq!(*effects.lock().unwrap(), vec!["survivor".to_string()]);
        assert!(agent.tools.is_empty(), "the shared agent is untouched");

        // A later independent run on the same agent starts from the unchanged tool set.
        let later = final_model();
        run(
            &agent,
            &later,
            "later",
            streamed,
            RunOptions::new("later".to_string()),
        )
        .await;
        assert!(tool_names(&later, 0).is_empty());
    }
}
