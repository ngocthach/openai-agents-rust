//! Ports of the nested-approval cases of `test_agent_as_tool.py`.
//!
//! Python drives these through dummy results and its global agent-tool run registry
//! (`record_agent_tool_run_result`, scope ids, `_rebuild_approvals`); these tests run the real
//! outer -> agent tool -> nested tool flow and decide through `RunState`. Python's registry
//! scope ids, legacy approval rebuilds, hosted-MCP `hosted_tool_call` wrappers and the
//! cross-scope `ToolContext` guard have no Rust counterpart (the nested state lives in the run
//! context, so no registry can alias two runs).

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{AgentAsToolOptions, AgentToolStreamEvent};
use openai_agents_sdk::items::ToolApprovalItem;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call_full, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, tool_namespace};
use openai_agents_sdk::{Agent, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

struct Outcome {
    final_text: Option<String>,
    interruptions: Vec<ToolApprovalItem>,
    state: RunState,
    requests: u64,
}

async fn run<C: Clone + Send + Sync + 'static>(
    agent: &Agent<C>,
    input: impl Into<openai_agents_sdk::items::RunInput>,
    context: C,
    streamed: bool,
) -> Outcome {
    let options = RunOptions::new(context);
    if streamed {
        let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
        {
            let events = stream.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        let result = stream
            .into_result()
            .expect("a finished stream has a result");
        Outcome {
            final_text: result.final_output_text(),
            interruptions: result.interruptions.clone(),
            state: result.to_state(),
            requests: result.usage().requests,
        }
    } else {
        let result = Runner::run_with_options(agent, input, options)
            .await
            .unwrap();
        Outcome {
            final_text: result.final_output_text(),
            interruptions: result.interruptions.clone(),
            state: result.to_state(),
            requests: result.usage().requests,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct ConfirmArgs {
    #[serde(default)]
    value: String,
}

fn confirm_tool(log: &Arc<Mutex<Vec<String>>>) -> FunctionTool<()> {
    let log = log.clone();
    FunctionTool::<()>::new(
        "confirm",
        "Needs approval",
        move |_ctx, args: ConfirmArgs| {
            let log = log.clone();
            async move {
                log.lock().unwrap().push(args.value.clone());
                Ok(format!("confirmed:{}", args.value))
            }
        },
    )
    .with_needs_approval(true)
}

/// An outer agent that calls `delegate` (an agent tool around `inner`) once, then answers.
fn outer_over(inner: Agent<()>, outer_model: &Arc<FakeModel>) -> Agent<()> {
    outer_model.add_turn(vec![function_call_with_id(
        "delegate",
        r#"{"input":"go"}"#,
        "outer",
    )]);
    outer_model.add_turn(vec![assistant_message("outer done")]);
    Agent::<()>::new("outer")
        .with_tool(inner.as_tool("delegate", "Delegate"))
        .with_model_instance(outer_model.clone())
}

/// Python: `test_agent_as_tool_cached_resume_rebinds_usage_to_outer_context`. The turns a nested
/// run takes after a resume are billed on the outer run's usage.
#[tokio::test]
async fn nested_turns_after_a_resume_accrue_on_the_outer_usage() {
    let inner_model = fake();
    inner_model.add_turn(vec![function_call_with_id(
        "confirm",
        r#"{"value":"one"}"#,
        "inner",
    )]);
    inner_model.add_turn(vec![assistant_message("inner done")]);
    let log = Arc::default();
    let inner = Agent::<()>::new("inner")
        .with_tool(confirm_tool(&log))
        .with_model_instance(inner_model);
    let outer = outer_over(inner, &fake());

    let paused = run(&outer, "go", (), false).await;
    assert_eq!(paused.interruptions.len(), 1);
    // outer turn 1 + nested turn 1
    assert_eq!(paused.requests, 2);
    let mut state = paused.state;
    state.approve(&paused.interruptions[0], false);
    let resumed = run(&outer, state, (), false).await;
    assert_eq!(resumed.final_text.as_deref(), Some("outer done"));
    // + nested turn 2 + outer turn 2: the nested resume's turn is not lost.
    assert_eq!(resumed.requests, 4);
    assert_eq!(*log.lock().unwrap(), ["one"]);
}

fn billing_tool(log: &Arc<Mutex<Vec<String>>>) -> FunctionTool<()> {
    let log = log.clone();
    tool_namespace(
        "billing",
        "Billing tools",
        vec![
            FunctionTool::<()>::new(
                "lookup_account",
                "Look up",
                move |_ctx, args: ConfirmArgs| {
                    let log = log.clone();
                    async move {
                        log.lock().unwrap().push(args.value.clone());
                        Ok(format!("account:{}", args.value))
                    }
                },
            )
            .with_needs_approval(true),
        ],
    )
    .unwrap()
    .remove(0)
}

/// Python: `test_agent_as_tool_namespaced_nested_always_approve_stays_permanent`.
#[tokio::test]
async fn a_permanent_namespaced_approval_carries_into_the_nested_resume() {
    let inner_model = fake();
    inner_model.add_turn(vec![function_call_full(
        "lookup_account",
        r#"{"value":"first"}"#,
        "inner-1",
        None,
        Some("billing".into()),
    )]);
    inner_model.add_turn(vec![function_call_full(
        "lookup_account",
        r#"{"value":"second"}"#,
        "inner-2",
        None,
        Some("billing".into()),
    )]);
    inner_model.add_turn(vec![assistant_message("inner done")]);
    let log = Arc::default();
    let inner = Agent::<()>::new("inner")
        .with_tool(billing_tool(&log))
        .with_model_instance(inner_model);
    let outer = outer_over(inner, &fake());

    let paused = run(&outer, "go", (), false).await;
    assert_eq!(paused.interruptions.len(), 1);
    assert_eq!(paused.interruptions[0].tool_name, "billing.lookup_account");
    let mut state = paused.state;
    state.approve(&paused.interruptions[0], true);
    let resumed = run(&outer, state, (), false).await;
    // The second call, a different call id, is approved without another pause.
    assert!(resumed.interruptions.is_empty());
    assert_eq!(resumed.final_text.as_deref(), Some("outer done"));
    assert_eq!(*log.lock().unwrap(), ["first", "second"]);
}

/// Python: `test_agent_as_tool_hosted_mcp_nested_sticky_decision_stays_scoped` (the
/// non-legacy cases, with a function tool in place of the hosted MCP request: a sticky decision
/// covers a follow-up call, a one-off decision does not).
#[tokio::test]
async fn a_nested_sticky_decision_covers_the_follow_up_call_and_a_one_off_does_not() {
    for (approve, sticky) in [(true, true), (false, true), (true, false), (false, false)] {
        let inner_model = fake();
        inner_model.add_turn(vec![function_call_with_id(
            "confirm",
            r#"{"value":"one"}"#,
            "inner-1",
        )]);
        inner_model.add_turn(vec![function_call_with_id(
            "confirm",
            r#"{"value":"two"}"#,
            "inner-2",
        )]);
        inner_model.add_turn(vec![assistant_message("inner done")]);
        let log = Arc::default();
        let inner = Agent::<()>::new("inner")
            .with_tool(confirm_tool(&log))
            .with_model_instance(inner_model.clone());
        let outer = outer_over(inner, &fake());

        let paused = run(&outer, "go", (), false).await;
        let mut state = paused.state;
        if approve {
            state.approve(&paused.interruptions[0], sticky);
        } else {
            state.reject(
                &paused.interruptions[0],
                sticky,
                Some("server-a denied".into()),
            );
        }
        let resumed = run(&outer, state, (), false).await;
        let label = format!("approve={approve} sticky={sticky}");
        match (approve, sticky) {
            (true, true) => {
                assert!(resumed.interruptions.is_empty(), "{label}");
                assert_eq!(*log.lock().unwrap(), ["one", "two"], "{label}");
            }
            (false, true) => {
                // The sticky rejection also refuses the follow-up call with its message.
                assert!(resumed.interruptions.is_empty(), "{label}");
                assert!(log.lock().unwrap().is_empty(), "{label}");
                let last_input = inner_model.calls().last().unwrap().input.clone();
                let sent =
                    serde_json::to_string(&last_input.iter().map(|i| &i.0).collect::<Vec<_>>())
                        .unwrap();
                assert!(
                    sent.matches("server-a denied").count() >= 2,
                    "{label}: {sent}"
                );
            }
            _ => {
                // A one-off decision decides only its own call; the follow-up pauses again.
                assert_eq!(resumed.interruptions.len(), 1, "{label}");
                assert_eq!(resumed.interruptions[0].call_id(), "inner-2", "{label}");
                assert_eq!(log.lock().unwrap().len(), usize::from(approve), "{label}");
            }
        }
    }
}

#[derive(Clone)]
struct UserContext {
    user: String,
}

/// Python: `test_fresh_runs_isolate_nested_agent_tool_approvals` (blocking and streamed).
#[tokio::test]
async fn fresh_runs_isolate_their_nested_agent_tool_approvals() {
    for streamed in [false, true] {
        let executed = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let log = executed.clone();
        let confirm = FunctionTool::<UserContext>::new(
            "confirm",
            "Needs approval",
            move |ctx, args: ConfirmArgs| {
                let log = log.clone();
                let user = ctx.context().user.clone();
                async move {
                    log.lock().unwrap().push((user, args.value.clone()));
                    Ok(args.value)
                }
            },
        )
        .with_needs_approval(true);
        let inner_model = fake();
        inner_model.add_turns(vec![
            vec![function_call_with_id(
                "confirm",
                r#"{"value":"first"}"#,
                "inner",
            )],
            vec![function_call_with_id(
                "confirm",
                r#"{"value":"second"}"#,
                "inner",
            )],
            vec![assistant_message("second confirmed")],
            vec![assistant_message("first confirmed")],
        ]);
        let inner = Agent::<UserContext>::new("inner")
            .with_tool(confirm)
            .with_model_instance(inner_model.clone());
        let mut options = AgentAsToolOptions::new()
            .tool_name("delegate")
            .tool_description("Delegate");
        if streamed {
            options = options.on_stream(|_event: AgentToolStreamEvent<UserContext>| async {});
        }
        let outer_model = fake();
        outer_model.add_turns(vec![
            vec![function_call_with_id(
                "delegate",
                r#"{"input":"go"}"#,
                "outer",
            )],
            vec![function_call_with_id(
                "delegate",
                r#"{"input":"go"}"#,
                "outer",
            )],
            vec![assistant_message("second done")],
            vec![assistant_message("first done")],
        ]);
        let outer = Agent::<UserContext>::new("outer")
            .with_tool(inner.as_tool_with(options))
            .with_model_instance(outer_model.clone());

        let first_ctx = UserContext {
            user: "first".into(),
        };
        let second_ctx = UserContext {
            user: "second".into(),
        };
        let first = run(&outer, "go", first_ctx.clone(), streamed).await;
        // The first result stays paused while a second fresh run uses the same call fields.
        let second = run(&outer, "go", second_ctx.clone(), streamed).await;
        assert_eq!(first.interruptions.len(), 1);
        assert_eq!(second.interruptions.len(), 1);
        let arguments = |item: &ToolApprovalItem| {
            serde_json::from_str::<serde_json::Value>(&item.raw_item.arguments).unwrap()
        };
        assert_eq!(
            arguments(&first.interruptions[0]),
            json!({"value": "first"})
        );
        assert_eq!(
            arguments(&second.interruptions[0]),
            json!({"value": "second"})
        );
        assert!(executed.lock().unwrap().is_empty());

        let mut second_state = second.state;
        second_state.approve(&second.interruptions[0], false);
        let second_done = run(&outer, second_state, second_ctx.clone(), streamed).await;
        assert_eq!(second_done.final_text.as_deref(), Some("second done"));
        assert!(second_done.interruptions.is_empty());
        assert_eq!(
            *executed.lock().unwrap(),
            [("second".to_string(), "second".to_string())]
        );

        // Completing the newer run must not consume the older run's pending approval.
        let still_pending = run(&outer, first.state, first_ctx.clone(), streamed).await;
        assert_eq!(still_pending.interruptions.len(), 1);
        assert_eq!(executed.lock().unwrap().len(), 1);
        let mut first_state = still_pending.state;
        first_state.approve(&still_pending.interruptions[0], false);
        let first_done = run(&outer, first_state, first_ctx.clone(), streamed).await;
        assert_eq!(first_done.final_text.as_deref(), Some("first done"));
        assert!(first_done.interruptions.is_empty());
        assert_eq!(
            *executed.lock().unwrap(),
            [
                ("second".to_string(), "second".to_string()),
                ("first".to_string(), "first".to_string()),
            ]
        );
        assert!(inner_model.remaining_steps() == 0 && outer_model.remaining_steps() == 0);
    }
}

/// Python: `test_replaced_agent_as_tool_preserves_agent_markers_for_build_agent_map`. A renamed
/// copy of an agent tool still resumes its nested agent from a serialized state.
#[tokio::test]
async fn a_renamed_agent_tool_still_resumes_its_nested_agent() {
    let inner_model = fake();
    inner_model.add_turn(vec![function_call_with_id(
        "confirm",
        r#"{"value":"one"}"#,
        "inner",
    )]);
    inner_model.add_turn(vec![assistant_message("inner done")]);
    let log = Arc::default();
    let inner = Agent::<()>::new("nested_agent")
        .with_tool(confirm_tool(&log))
        .with_model_instance(inner_model);
    let mut replaced = inner.as_tool("nested_agent_tool", "Nested agent tool");
    replaced.name = "replaced_nested_agent_tool".to_string();
    let outer_model = fake();
    outer_model.add_turn(vec![function_call_with_id(
        "replaced_nested_agent_tool",
        r#"{"input":"go"}"#,
        "outer",
    )]);
    outer_model.add_turn(vec![assistant_message("outer done")]);
    let outer = Agent::<()>::new("parent_agent")
        .with_tool(replaced)
        .with_model_instance(outer_model);

    let paused = run(&outer, "go", (), false).await;
    assert_eq!(paused.interruptions.len(), 1);
    let mut state = RunState::from_json(paused.state.to_json().unwrap()).unwrap();
    state.approve(&paused.interruptions[0], false);
    let resumed = run(&outer, state, (), false).await;
    assert_eq!(resumed.final_text.as_deref(), Some("outer done"));
    assert_eq!(*log.lock().unwrap(), ["one"]);
}
