//! Port of the Python SDK's `tests/test_agent_tracing.py`: the trace and span tree a run produces
//! for single, repeated, wrapped and mixed runs (streamed and not), for resumed runs (from the
//! live state and from serialized state), for approval resumes, and for the task and turn spans
//! with their aggregate usage.
//!
//! Python's `fetch_*` helpers read a global span processor. Here each test installs its own
//! [`SpanRecorder`] through [`Pipeline`] while holding the shared tracing lock.

#[path = "py_core4_tracerunner_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::{OutputItem, ToolApprovalItem};
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{Span, TracingProcessor, current_span, custom_span, trace};
use openai_agents_sdk::usage::{InputTokensDetails, Usage};
use openai_agents_sdk::{Agent, AgentsError, RunOptions, RunState, Runner, ToolUseBehavior};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use support::{Pipeline, SpanRecorder, lock_async};

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NoArgs {
    #[serde(default)]
    a: String,
}

/// A scripted model that replays `steps`, one list of output items per call.
fn scripted(steps: Vec<Vec<OutputItem>>, emit: bool) -> Arc<ScriptedModel> {
    let model = ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect())
        .expect("scripted steps");
    Arc::new(model.with_emit_traces(emit))
}

fn text(s: &str) -> Vec<OutputItem> {
    vec![assistant_message(s)]
}

fn call(name: &str, call_id: &str) -> Vec<OutputItem> {
    vec![function_call_with_id(name, "{}", call_id)]
}

/// One normalized `agent` child, as the Python snapshots spell it.
fn agent_child(name: &str, handoffs: &[&str], tools: &[&str]) -> Value {
    json!({
        "type": "agent",
        "data": {
            "name": name,
            "handoffs": handoffs,
            "tools": tools,
            "output_type": "str",
        },
    })
}

async fn recording() -> (
    tokio::sync::MutexGuard<'static, ()>,
    Arc<SpanRecorder>,
    Pipeline,
) {
    let lock = lock_async().await;
    let rec = SpanRecorder::new();
    let pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    (lock, rec, pipeline)
}

fn export_of(span: &Span) -> Value {
    span.export().expect("exported span")
}

/// The span's type as Python's `span.span_data.type` names it. Task and turn spans are custom
/// spans on the wire, tagged with `sdk_span_type`.
fn span_type(span: &Span) -> String {
    let export = export_of(span);
    let data = &export["span_data"];
    if data["type"] == "custom"
        && let Some(sdk) = data["data"]["sdk_span_type"].as_str()
    {
        return sdk.to_string();
    }
    data["type"].as_str().unwrap_or_default().to_string()
}

fn of_type(rec: &SpanRecorder, ty: &str) -> Vec<Span> {
    rec.ordered_spans()
        .into_iter()
        .filter(|s| span_type(s) == ty)
        .collect()
}

fn usage_of(span: &Span) -> Value {
    export_of(span)["span_data"]["data"]["usage"].clone()
}

/// The outcome of one run driven to its end, whether it was streamed or not.
struct Outcome {
    final_output: Option<String>,
    interruptions: Vec<ToolApprovalItem>,
    state: Option<RunState>,
    error: Option<AgentsError>,
}

async fn drive_text(
    agent: &Agent<()>,
    input: &str,
    streamed: bool,
    options: RunOptions<()>,
) -> Outcome {
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, input, options);
        let mut error = None;
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                    break;
                }
            }
        }
        Outcome {
            final_output: result.final_output_text(),
            interruptions: result.interruptions(),
            state: result.to_state(),
            error,
        }
    } else {
        match Runner::run_with_options(agent, input, options).await {
            Ok(r) => Outcome {
                final_output: r.final_output_text(),
                interruptions: r.interruptions.clone(),
                state: Some(r.to_state()),
                error: None,
            },
            Err(e) => Outcome {
                final_output: None,
                interruptions: Vec::new(),
                state: None,
                error: Some(e),
            },
        }
    }
}

async fn drive_state(
    agent: &Agent<()>,
    state: RunState,
    streamed: bool,
    options: RunOptions<()>,
) -> Outcome {
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, state, options);
        let mut error = None;
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                    break;
                }
            }
        }
        Outcome {
            final_output: result.final_output_text(),
            interruptions: result.interruptions(),
            state: result.to_state(),
            error,
        }
    } else {
        match Runner::run_with_options(agent, state, options).await {
            Ok(r) => Outcome {
                final_output: r.final_output_text(),
                interruptions: r.interruptions.clone(),
                state: Some(r.to_state()),
                error: None,
            },
            Err(e) => Outcome {
                final_output: None,
                interruptions: Vec::new(),
                state: None,
                error: Some(e),
            },
        }
    }
}

/// Run `agent` once with `input`, streamed or not, inside a default run configuration.
async fn run_once(agent: &Agent<()>, input: &str, streamed: bool) -> Outcome {
    drive_text(agent, input, streamed, RunOptions::new(())).await
}

fn approval_tool(calls: Arc<Mutex<usize>>) -> FunctionTool<()> {
    FunctionTool::new("approval_tool", "approval tool", move |_c, _a: NoArgs| {
        let calls = calls.clone();
        async move {
            *calls.lock().unwrap() += 1;
            Ok("ok".to_string())
        }
    })
    .with_needs_approval(true)
}

/// The `approval agent` whose only tool asks for approval, as Python's `_make_approval_agent`.
fn approval_agent(model: Arc<ScriptedModel>) -> Agent<()> {
    Agent::<()>::new("test_agent")
        .with_model_instance(model)
        .with_tool(approval_tool(Arc::new(Mutex::new(0))))
}

// -------------------------------------------------------------------------------------------
// single and multiple runs
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn single_run_is_single_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(vec![text("first_test")], false));

    run_once(&agent, "first_test", false).await;

    assert_eq!(
        rec.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [agent_child("test_agent", &[], &[])],
        })]
    );
}

#[tokio::test]
async fn agent_span_uses_resolved_tool_name_collision_view() {
    for surface in ["function_tool", "handoff", "mixed"] {
        for streamed in [false, true] {
            let (_lock, rec, _p) = recording().await;
            let model = scripted(vec![text("done")], false);
            let (agent, expected_tools, expected_handoffs): (Agent<()>, Vec<&str>, Vec<&str>) =
                match surface {
                    "function_tool" => {
                        let first = FunctionTool::new("lookup", "first", |_c, _a: NoArgs| async {
                            Ok("first".to_string())
                        });
                        let second =
                            FunctionTool::new("lookup", "second", |_c, _a: NoArgs| async {
                                Ok("second".to_string())
                            });
                        (
                            Agent::<()>::new("test_agent")
                                .with_model_instance(model)
                                .with_tool(first)
                                .with_tool(second),
                            vec!["lookup"],
                            vec![],
                        )
                    }
                    "handoff" => (
                        Agent::<()>::new("test_agent")
                            .with_model_instance(model)
                            .with_handoff(Agent::<()>::new("Billing Agent"))
                            .with_handoff(Agent::<()>::new("billing agent")),
                        vec![],
                        vec!["billing agent"],
                    ),
                    _ => (
                        Agent::<()>::new("test_agent")
                            .with_model_instance(model)
                            .with_tool(
                                Agent::<()>::new("transfer to Billing Agent")
                                    .as_tool("transfer_to_billing_agent", "Billing tool"),
                            )
                            .with_handoff(Agent::<()>::new("Billing Agent")),
                        vec![],
                        vec!["Billing Agent"],
                    ),
                };

            run_once(&agent, "test", streamed).await;

            let agent_spans = of_type(&rec, "agent");
            let ctx = format!("{surface} streamed={streamed}");
            assert_eq!(agent_spans.len(), 1, "{ctx}");
            let data = export_of(&agent_spans[0])["span_data"].clone();
            let tools: Vec<&str> = data["tools"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let handoffs: Vec<&str> = data["handoffs"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            assert_eq!(tools, expected_tools, "{ctx}");
            assert_eq!(handoffs, expected_handoffs, "{ctx}");
        }
    }
}

#[tokio::test]
async fn task_and_turn_spans_export_aggregate_usage() {
    let (_lock, rec, _p) = recording().await;
    let foo = FunctionTool::new("foo_tool", "foo", |_c, _a: NoArgs| async {
        Ok("foo result".to_string())
    });
    let model = ScriptedModel::with_steps(vec![
        ScriptedStep::output(call("foo_tool", "call-1")),
        ScriptedStep::output(text("done")),
    ])
    .unwrap()
    .with_emit_traces(true);
    model.set_default_usage(Some(Usage {
        requests: 1,
        input_tokens: 10,
        output_tokens: 3,
        total_tokens: 13,
        input_tokens_details: InputTokensDetails {
            cached_tokens: 2,
            cache_write_tokens: 3,
        },
        ..Default::default()
    }));
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(Arc::new(model))
        .with_tool(foo);

    run_once(&agent, "first_test", false).await;

    let tasks = of_type(&rec, "task");
    let turns = of_type(&rec, "turn");
    let agents = of_type(&rec, "agent");
    let generations = of_type(&rec, "generation");

    assert_eq!(tasks.len(), 1);
    let task = export_of(&tasks[0]);
    assert_eq!(
        task["span_data"],
        json!({
            "type": "custom",
            "name": "task",
            "data": {
                "sdk_span_type": "task",
                "name": "Agent workflow",
                "usage": {
                    "requests": 2,
                    "input_tokens": 20,
                    "output_tokens": 6,
                    "total_tokens": 26,
                    "cached_input_tokens": 4,
                    "cache_write_input_tokens": 6,
                },
            },
        })
    );
    assert!(task.get("metadata").is_none_or(Value::is_null));
    let turn_usages: Vec<Value> = turns.iter().map(usage_of).collect();
    let per_turn = json!({
        "input_tokens": 10,
        "output_tokens": 3,
        "cached_input_tokens": 2,
        "cache_write_input_tokens": 3,
    });
    assert_eq!(turn_usages, vec![per_turn.clone(), per_turn]);
    let turn_data: Vec<Value> = turns
        .iter()
        .map(|t| export_of(t)["span_data"].clone())
        .collect();
    assert_eq!(
        turn_data,
        vec![
            json!({"type": "custom", "name": "turn", "data": {
                "sdk_span_type": "turn", "turn": 1, "agent_name": "test_agent",
                "usage": {"input_tokens": 10, "output_tokens": 3,
                    "cached_input_tokens": 2, "cache_write_input_tokens": 3}}}),
            json!({"type": "custom", "name": "turn", "data": {
                "sdk_span_type": "turn", "turn": 2, "agent_name": "test_agent",
                "usage": {"input_tokens": 10, "output_tokens": 3,
                    "cached_input_tokens": 2, "cache_write_input_tokens": 3}}}),
        ]
    );
    assert_eq!(task["span_data"]["data"]["usage"]["requests"], 2);
    assert_eq!(agents.len(), 1);
    assert_eq!(generations.len(), 2);
    assert_eq!(task["parent_id"], Value::Null);
    let task_id = task["id"].as_str().unwrap();
    assert_eq!(export_of(&agents[0])["parent_id"].as_str(), Some(task_id));
    let agent_id = export_of(&agents[0])["id"].as_str().unwrap().to_string();
    for turn in &turns {
        assert_eq!(
            export_of(turn)["parent_id"].as_str(),
            Some(agent_id.as_str())
        );
    }
    for generation in &generations {
        let parent = export_of(generation)["parent_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(turns.iter().any(|t| export_of(t)["id"] == parent.as_str()));
    }
}

#[tokio::test]
async fn task_and_turn_spans_can_be_disabled() {
    let (_lock, rec, _p) = recording().await;
    let foo = FunctionTool::new("foo_tool", "foo", |_c, _a: NoArgs| async {
        Ok("foo result".to_string())
    });
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(
            vec![call("foo_tool", "call-1"), text("done")],
            true,
        ))
        .with_tool(foo);
    let options = RunOptions::new(()).run_config(RunConfig {
        include_task_and_turn_spans: false,
        ..RunConfig::new()
    });

    let result = drive_text(&agent, "first_test", false, options).await;

    assert_eq!(result.final_output.as_deref(), Some("done"));
    let spans = rec.ordered_spans();
    assert!(
        !spans
            .iter()
            .any(|s| matches!(span_type(s).as_str(), "task" | "turn"))
    );
    let agents = of_type(&rec, "agent");
    let generations = of_type(&rec, "generation");
    let functions = of_type(&rec, "function");
    assert_eq!(agents.len(), 1);
    assert!(export_of(&agents[0])["parent_id"].is_null());
    assert_eq!(generations.len(), 2);
    assert_eq!(functions.len(), 1);
    let agent_id = export_of(&agents[0])["id"].as_str().unwrap().to_string();
    for span in generations.iter().chain(functions.iter()) {
        assert_eq!(
            export_of(span)["parent_id"].as_str(),
            Some(agent_id.as_str())
        );
    }
}

#[tokio::test]
async fn task_and_turn_spans_can_be_explicitly_enabled() {
    let (_lock, rec, _p) = recording().await;
    let agent =
        Agent::<()>::new("test_agent").with_model_instance(scripted(vec![text("done")], true));
    let options = RunOptions::new(()).run_config(RunConfig {
        include_task_and_turn_spans: true,
        ..RunConfig::new()
    });

    drive_text(&agent, "first_test", false, options).await;

    let types: Vec<String> = rec.ordered_spans().iter().map(span_type).collect();
    assert_eq!(types.iter().filter(|t| *t == "task").count(), 1);
    assert_eq!(types.iter().filter(|t| *t == "turn").count(), 1);
}

#[tokio::test]
async fn task_span_is_closed_when_the_run_fails_inside_it() {
    // Python fails the run during setup, inside the task span, with a monkeypatched sandbox check.
    // The Rust run loop has no such check, so the failing model call is the point where the run
    // leaves its task span. The same invariants apply: the task span exists, has no parent, and
    // the caller's next span is not parented to it.
    let (_lock, rec, _p) = recording().await;
    let model = ScriptedModel::new().with_emit_traces(true);
    model
        .enqueue(ScriptedStep::raise_error(AgentsError::other(
            "setup failed",
        )))
        .unwrap();
    let agent = Agent::<()>::new("test_agent").with_model_instance(Arc::new(model));

    let workflow = trace("test_workflow").build();
    let after_id = Arc::new(Mutex::new(None::<Value>));
    let after_for_run = after_id.clone();
    workflow
        .clone()
        .run(async {
            let outcome = Runner::run_with_options(&agent, "first_test", RunOptions::new(())).await;
            assert!(outcome.unwrap_err().to_string().contains("setup failed"));
            let after = custom_span("after_setup_failure", json!({}));
            after.start();
            after.finish();
            *after_for_run.lock().unwrap() = after.export();
        })
        .await;

    let after = after_id.lock().unwrap().clone().expect("after span");
    assert!(after["parent_id"].is_null());
    assert!(current_span().is_none());
    let tasks = of_type(&rec, "task");
    assert_eq!(tasks.len(), 1);
    assert!(export_of(&tasks[0])["parent_id"].is_null());
}

#[tokio::test]
async fn multiple_runs_are_multiple_traces() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent_1").with_model_instance(scripted(
        vec![text("first_test"), text("second_test")],
        false,
    ));

    run_once(&agent, "first_test", false).await;
    run_once(&agent, "second_test", false).await;

    let one = json!({
        "workflow_name": "Agent workflow",
        "children": [agent_child("test_agent_1", &[], &[])],
    });
    assert_eq!(rec.normalized(false, false), vec![one.clone(), one]);
}

// -------------------------------------------------------------------------------------------
// approval resume
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn approved_tool_executes_inside_resumed_agent_span() {
    for outcome_kind in ["continue", "stop", "error", "handoff"] {
        for streamed in [false, true] {
            for include_task_and_turn_spans in [false, true] {
                let (_lock, rec, _p) = recording().await;
                let calls = Arc::new(Mutex::new(0usize));
                let calls_in_tool = calls.clone();
                let error_outcome = outcome_kind == "error";
                let approved =
                    FunctionTool::new("approved_tool", "approved", move |_c, _a: NoArgs| {
                        let calls = calls_in_tool.clone();
                        async move {
                            *calls.lock().unwrap() += 1;
                            if error_outcome {
                                return Err(AgentsError::other("synthetic tool failure"));
                            }
                            Ok("tool result".to_string())
                        }
                    })
                    .with_needs_approval(true)
                    .without_failure_error_function();

                let target = Agent::<()>::new("target")
                    .with_model_instance(scripted(vec![text("done")], false));
                let mut first_response = call("approved_tool", "call-1");
                let handoff = outcome_kind == "handoff";
                if handoff {
                    first_response.push(openai_agents_sdk::testing::function_call_with_id(
                        "transfer_to_target",
                        "{}",
                        "handoff-1",
                    ));
                }
                let stop = outcome_kind == "stop";
                let mut agent = Agent::<()>::new("approval agent")
                    .with_tool(approved)
                    .with_model_instance(scripted(vec![first_response, text("done")], false));
                if handoff {
                    agent = agent.with_handoff(target);
                }
                if stop {
                    agent = agent.with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool);
                }
                let config = RunConfig {
                    include_task_and_turn_spans,
                    ..RunConfig::new()
                };
                let max_turns = if stop { 1 } else { 2 };
                let opts = || {
                    RunOptions::new(())
                        .max_turns(Some(max_turns))
                        .run_config(config.clone())
                };

                let first = drive_text(&agent, "run the tool", streamed, opts()).await;
                let ctx = format!(
                    "outcome={outcome_kind} streamed={streamed} include={include_task_and_turn_spans}"
                );
                assert_eq!(*calls.lock().unwrap(), 0, "{ctx}");
                assert_eq!(first.interruptions.len(), 1, "{ctx}");
                let mut state = first.state.clone().expect("state");
                state.approve(&first.interruptions[0], false);
                let initial_ids: Vec<String> = rec
                    .ordered_spans()
                    .iter()
                    .map(|s| s.span_id().to_string())
                    .collect();

                let resumed = drive_state(&agent, state, streamed, opts()).await;
                if error_outcome {
                    let err = resumed.error.as_ref().expect("the tool failure surfaces");
                    assert!(
                        err.to_string().contains("synthetic tool failure"),
                        "{ctx}: {err}"
                    );
                } else {
                    assert!(resumed.error.is_none(), "{ctx}: {:?}", resumed.error);
                    assert_eq!(
                        resumed.final_output.as_deref(),
                        Some(if stop { "tool result" } else { "done" }),
                        "{ctx}"
                    );
                }

                assert_eq!(*calls.lock().unwrap(), 1, "{ctx}");
                let spans: Vec<Span> = rec
                    .ordered_spans()
                    .into_iter()
                    .filter(|s| !initial_ids.iter().any(|id| id == s.span_id()))
                    .collect();
                let agent_spans: Vec<&Span> =
                    spans.iter().filter(|s| span_type(s) == "agent").collect();
                let function_spans: Vec<&Span> = spans
                    .iter()
                    .filter(|s| span_type(s) == "function")
                    .collect();
                assert_eq!(agent_spans.len(), if handoff { 2 } else { 1 }, "{ctx}");
                assert_eq!(function_spans.len(), 1, "{ctx}");
                let agent_span = agent_spans[0];
                let agent_export = export_of(agent_span);
                let agent_id = agent_export["id"].as_str().unwrap();
                assert_eq!(
                    export_of(function_spans[0])["parent_id"].as_str(),
                    Some(agent_id),
                    "{ctx}"
                );
                assert_eq!(agent_export["span_data"]["name"], "approval agent", "{ctx}");
                assert_eq!(
                    agent_export["span_data"]["tools"],
                    json!(["approved_tool"]),
                    "{ctx}"
                );
                assert_eq!(
                    agent_export["span_data"]["handoffs"],
                    if handoff {
                        json!(["target"])
                    } else {
                        json!([])
                    },
                    "{ctx}"
                );
                if handoff {
                    let target_span = agent_spans[1];
                    assert_eq!(
                        export_of(target_span)["span_data"]["name"],
                        "target",
                        "{ctx}"
                    );
                    assert_eq!(
                        export_of(target_span)["parent_id"],
                        agent_export["parent_id"],
                        "{ctx}"
                    );
                    let handoffs: Vec<&Span> =
                        spans.iter().filter(|s| span_type(s) == "handoff").collect();
                    assert_eq!(handoffs.len(), 1, "{ctx}");
                    assert_eq!(
                        export_of(handoffs[0])["parent_id"].as_str(),
                        Some(agent_id),
                        "{ctx}"
                    );
                }
                if error_outcome {
                    assert_eq!(
                        agent_export["error"]["message"], "Error in agent run",
                        "{ctx}"
                    );
                    assert_eq!(
                        export_of(function_spans[0])["error"]["message"],
                        "Error running tool",
                        "{ctx}"
                    );
                } else {
                    assert!(agent_export["error"].is_null(), "{ctx}");
                }
                assert!(
                    spans.iter().all(|s| export_of(s)["ended_at"].is_string()),
                    "{ctx}"
                );
                let events = rec.events();
                assert_eq!(
                    events.iter().filter(|e| **e == "span_start").count(),
                    events.iter().filter(|e| **e == "span_end").count(),
                    "{ctx}"
                );
                assert!(current_span().is_none(), "{ctx}");
            }
        }
    }
}

#[tokio::test]
async fn resumed_run_reuses_original_trace_without_duplicate_trace_start() {
    let (_lock, rec, _p) = recording().await;
    let agent = approval_agent(scripted(
        vec![call("approval_tool", "call-1"), text("done")],
        false,
    ));

    let first = run_once(&agent, "first_test", false).await;
    assert!(!first.interruptions.is_empty());
    let mut state = first.state.expect("state");
    state.approve(&first.interruptions[0], false);

    let resumed = drive_state(&agent, state, false, RunOptions::new(())).await;

    assert_eq!(resumed.final_output.as_deref(), Some("done"));
    let traces = rec.traces();
    assert_eq!(traces.len(), 1);
    let events = rec.events();
    assert_eq!(events.iter().filter(|e| **e == "trace_start").count(), 1);
    assert_eq!(events.iter().filter(|e| **e == "trace_end").count(), 1);
    assert!(
        rec.ordered_spans()
            .iter()
            .all(|s| s.trace_id() == traces[0].trace_id())
    );
}

#[tokio::test]
async fn resumed_run_task_span_usage_is_run_local_delta() {
    let (_lock, rec, _p) = recording().await;
    let model = ScriptedModel::with_steps(vec![
        ScriptedStep::output(call("approval_tool", "call-1")),
        ScriptedStep::output(text("done")),
    ])
    .unwrap();
    model.set_default_usage(Some(Usage::single(10, 3)));
    let agent = approval_agent(Arc::new(model));

    let first = run_once(&agent, "first_test", false).await;
    let mut state = first.state.expect("state");
    state.approve(&first.interruptions[0], false);
    let resumed = drive_state(&agent, state, false, RunOptions::new(())).await;

    assert_eq!(resumed.final_output.as_deref(), Some("done"));
    let usages: Vec<Value> = of_type(&rec, "task")
        .iter()
        .map(|t| export_of(t)["span_data"]["data"]["usage"].clone())
        .collect();
    let expected = json!({
        "requests": 1,
        "input_tokens": 10,
        "output_tokens": 3,
        "total_tokens": 13,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
    });
    assert_eq!(usages, vec![expected.clone(), expected]);
}

#[tokio::test]
async fn resumed_run_from_serialized_state_reuses_original_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = approval_agent(scripted(
        vec![call("approval_tool", "call-1"), text("done")],
        false,
    ));

    let first = run_once(&agent, "first_test", false).await;
    let serialized = first.state.expect("state").to_json_string().unwrap();
    let mut restored = RunState::from_json_str(&serialized).unwrap();
    let interruptions = restored.get_interruptions();
    assert_eq!(interruptions.len(), 1);
    restored.approve(&interruptions[0], false);

    let resumed = drive_state(&agent, restored, false, RunOptions::new(())).await;

    assert_eq!(resumed.final_output.as_deref(), Some("done"));
    let traces = rec.traces();
    assert_eq!(traces.len(), 1);
    let events = rec.events();
    assert_eq!(events.iter().filter(|e| **e == "trace_start").count(), 1);
    assert_eq!(events.iter().filter(|e| **e == "trace_end").count(), 1);
    assert!(
        rec.ordered_spans()
            .iter()
            .all(|s| s.trace_id() == traces[0].trace_id())
    );
}

#[tokio::test]
async fn resumed_run_from_serialized_state_preserves_explicit_trace_key() {
    let (_lock, rec, _p) = recording().await;
    let agent = approval_agent(scripted(
        vec![call("approval_tool", "call-1"), text("done")],
        false,
    ));
    let config = RunConfig {
        tracing_api_key: Some("trace-key".into()),
        ..RunConfig::new()
    };

    let first = drive_text(
        &agent,
        "first_test",
        false,
        RunOptions::new(()).run_config(config.clone()),
    )
    .await;
    let serialized = first.state.expect("state").to_json_string().unwrap();
    let mut restored = RunState::from_json_str(&serialized).unwrap();
    let interruptions = restored.get_interruptions();
    assert_eq!(interruptions.len(), 1);
    restored.approve(&interruptions[0], false);

    let resumed = drive_state(
        &agent,
        restored,
        false,
        RunOptions::new(()).run_config(config.clone()),
    )
    .await;

    assert_eq!(resumed.final_output.as_deref(), Some("done"));
    let traces = rec.traces();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].tracing_api_key(), Some("trace-key"));
    let events = rec.events();
    assert_eq!(events.iter().filter(|e| **e == "trace_start").count(), 1);
    assert_eq!(events.iter().filter(|e| **e == "trace_end").count(), 1);
    assert!(
        rec.ordered_spans()
            .iter()
            .all(|s| s.trace_id() == traces[0].trace_id())
    );
    assert!(
        rec.ordered_spans()
            .iter()
            .all(|s| s.tracing_api_key() == Some("trace-key"))
    );
}

#[tokio::test]
async fn resumed_run_with_workflow_override_starts_new_trace() {
    let (_lock, rec, _p) = recording().await;
    let trace_id = format!("trace_{}", openai_agents_sdk::tracing::gen_trace_id());
    let agent = approval_agent(scripted(
        vec![call("approval_tool", "call-1"), text("done")],
        false,
    ));

    let first = drive_text(
        &agent,
        "first_test",
        false,
        RunOptions::new(()).run_config(RunConfig {
            workflow_name: "original_workflow".into(),
            trace_id: Some(trace_id.clone()),
            group_id: Some("group-1".into()),
            ..RunConfig::new()
        }),
    )
    .await;
    let mut state = first.state.expect("state");
    state.approve(&first.interruptions[0], false);

    let resumed = drive_state(
        &agent,
        state,
        false,
        RunOptions::new(()).run_config(RunConfig {
            workflow_name: "override_workflow".into(),
            ..RunConfig::new()
        }),
    )
    .await;

    assert_eq!(resumed.final_output.as_deref(), Some("done"));
    let traces = rec.traces();
    assert_eq!(traces.len(), 2);
    let events = rec.events();
    assert_eq!(events.iter().filter(|e| **e == "trace_start").count(), 2);
    assert_eq!(events.iter().filter(|e| **e == "trace_end").count(), 2);
    assert_eq!(
        traces
            .iter()
            .map(|t| t.trace_id().to_string())
            .collect::<Vec<_>>(),
        vec![trace_id.clone(), trace_id]
    );
    assert_eq!(
        traces
            .iter()
            .map(|t| t.name().to_string())
            .collect::<Vec<_>>(),
        vec![
            "original_workflow".to_string(),
            "override_workflow".to_string()
        ]
    );
}

// -------------------------------------------------------------------------------------------
// wrapped traces, disabling and configuration
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn wrapped_trace_is_single_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent_1").with_model_instance(scripted(
        vec![text("first_test"), text("second_test"), text("third_test")],
        false,
    ));

    trace("test_workflow")
        .run(async {
            run_once(&agent, "first_test", false).await;
            run_once(&agent, "second_test", false).await;
            run_once(&agent, "third_test", false).await;
        })
        .await;

    let child = agent_child("test_agent_1", &[], &[]);
    assert_eq!(
        rec.normalized(false, false),
        vec![json!({
            "workflow_name": "test_workflow",
            "children": [child.clone(), child.clone(), child],
        })]
    );
}

#[tokio::test]
async fn parent_disabled_trace_disabled_agent_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(vec![text("first_test")], false));

    trace("test_workflow")
        .disabled(true)
        .run(async {
            run_once(&agent, "first_test", false).await;
        })
        .await;

    assert!(rec.traces().is_empty());
    assert!(rec.ordered_spans().is_empty());
}

#[tokio::test]
async fn manual_disabling_works() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(vec![text("first_test")], false));

    drive_text(
        &agent,
        "first_test",
        false,
        RunOptions::new(()).run_config(RunConfig::new().with_tracing_disabled(true)),
    )
    .await;

    assert!(rec.traces().is_empty());
    assert!(rec.ordered_spans().is_empty());
}

#[tokio::test]
async fn trace_config_works() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(vec![text("first_test")], false));

    drive_text(
        &agent,
        "first_test",
        false,
        RunOptions::new(()).run_config(RunConfig {
            workflow_name: "Foo bar".into(),
            group_id: Some("123".into()),
            trace_id: Some("trace_456".into()),
            ..RunConfig::new()
        }),
    )
    .await;

    assert_eq!(
        rec.normalized(false, true),
        vec![json!({
            "id": "trace_456",
            "workflow_name": "Foo bar",
            "group_id": "123",
            "children": [agent_child("test_agent", &[], &[])],
        })]
    );
}

#[tokio::test]
async fn not_starting_streaming_creates_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(vec![text("first_test")], false));

    let mut result = Runner::run_streamed_with_options(&agent, "first_test", RunOptions::new(()));

    // The consumer never reads the stream; the run still completes on its own task.
    while !result.is_complete() {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    assert_eq!(
        rec.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [agent_child("test_agent", &[], &[])],
        })]
    );

    // Drain the stream so the run's task is not left unconsumed.
    let mut events = std::pin::pin!(result.stream_events());
    while events.next().await.is_some() {}
}

#[tokio::test]
async fn streaming_single_run_is_single_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(vec![text("first_test")], false));

    run_once(&agent, "first_test", true).await;

    assert_eq!(
        rec.normalized(false, false),
        vec![json!({
            "workflow_name": "Agent workflow",
            "children": [agent_child("test_agent", &[], &[])],
        })]
    );
}

#[tokio::test]
async fn multiple_streamed_runs_are_multiple_traces() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent_1").with_model_instance(scripted(
        vec![text("first_test"), text("second_test")],
        false,
    ));

    run_once(&agent, "first_test", true).await;
    run_once(&agent, "second_test", true).await;

    let one = json!({
        "workflow_name": "Agent workflow",
        "children": [agent_child("test_agent_1", &[], &[])],
    });
    assert_eq!(rec.normalized(false, false), vec![one.clone(), one]);
}

#[tokio::test]
async fn resumed_streaming_run_reuses_original_trace_without_duplicate_trace_start() {
    let (_lock, rec, _p) = recording().await;
    let agent = approval_agent(scripted(
        vec![call("approval_tool", "call-1"), text("done")],
        false,
    ));

    let first = run_once(&agent, "first_test", true).await;
    assert!(!first.interruptions.is_empty());
    let mut state = first.state.expect("state");
    state.approve(&first.interruptions[0], false);

    let resumed = drive_state(&agent, state, true, RunOptions::new(())).await;

    assert_eq!(resumed.final_output.as_deref(), Some("done"));
    let traces = rec.traces();
    assert_eq!(traces.len(), 1);
    let events = rec.events();
    assert_eq!(events.iter().filter(|e| **e == "trace_start").count(), 1);
    assert_eq!(events.iter().filter(|e| **e == "trace_end").count(), 1);
    assert!(
        rec.ordered_spans()
            .iter()
            .all(|s| s.trace_id() == traces[0].trace_id())
    );
}

#[tokio::test]
async fn resumed_streaming_run_task_span_usage_is_run_local_delta() {
    let (_lock, rec, _p) = recording().await;
    let model = ScriptedModel::with_steps(vec![
        ScriptedStep::output(call("approval_tool", "call-1")),
        ScriptedStep::output(text("done")),
    ])
    .unwrap();
    model.set_default_usage(Some(Usage::single(11, 4)));
    let agent = approval_agent(Arc::new(model));

    let first = run_once(&agent, "first_test", true).await;
    let mut state = first.state.expect("state");
    state.approve(&first.interruptions[0], false);
    let resumed = drive_state(&agent, state, true, RunOptions::new(())).await;

    assert_eq!(resumed.final_output.as_deref(), Some("done"));
    let usages: Vec<Value> = of_type(&rec, "task")
        .iter()
        .map(|t| export_of(t)["span_data"]["data"]["usage"].clone())
        .collect();
    let expected = json!({
        "requests": 1,
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
    });
    assert_eq!(usages, vec![expected.clone(), expected]);
}

#[tokio::test]
async fn wrapped_streaming_trace_is_single_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent_1").with_model_instance(scripted(
        vec![text("first_test"), text("second_test"), text("third_test")],
        false,
    ));

    trace("test_workflow")
        .run(async {
            run_once(&agent, "first_test", true).await;
            run_once(&agent, "second_test", true).await;
            run_once(&agent, "third_test", true).await;
        })
        .await;

    let child = agent_child("test_agent_1", &[], &[]);
    assert_eq!(
        rec.normalized(false, false),
        vec![json!({
            "workflow_name": "test_workflow",
            "children": [child.clone(), child.clone(), child],
        })]
    );
}

#[tokio::test]
async fn wrapped_streaming_run_creates_root_task_span() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(scripted(vec![text("first_test")], true));

    trace("test_workflow")
        .run(async {
            run_once(&agent, "first_test", true).await;
        })
        .await;

    let spans = rec.ordered_spans();
    let tasks = of_type(&rec, "task");
    let agents = of_type(&rec, "agent");
    let turns = of_type(&rec, "turn");
    let generations = of_type(&rec, "generation");
    assert_eq!(tasks.len(), 1);
    let task_export = export_of(&tasks[0]);
    assert!(task_export["parent_id"].is_null());
    assert_eq!(agents.len(), 1);
    let agent_export = export_of(&agents[0]);
    assert_eq!(agent_export["parent_id"], task_export["id"]);
    assert_eq!(turns.len(), 1);
    let turn_export = export_of(&turns[0]);
    assert_eq!(turn_export["parent_id"], agent_export["id"]);
    assert_eq!(generations.len(), 1);
    assert_eq!(export_of(&generations[0])["parent_id"], turn_export["id"]);
    assert!(spans.len() >= 4);
}

#[tokio::test]
async fn wrapped_run_task_span_uses_run_workflow_name() {
    let (_lock, rec, _p) = recording().await;
    let make_agent = || {
        Agent::<()>::new("test_agent")
            .with_model_instance(scripted(vec![text("first_test")], false))
    };
    let run_config = RunConfig {
        workflow_name: "inner_workflow".into(),
        ..RunConfig::new()
    };

    trace("outer_workflow")
        .run(async {
            drive_text(
                &make_agent(),
                "first_test",
                false,
                RunOptions::new(()).run_config(run_config.clone()),
            )
            .await;
            drive_text(
                &make_agent(),
                "first_test",
                true,
                RunOptions::new(()).run_config(run_config.clone()),
            )
            .await;
        })
        .await;

    // A task span names one Runner invocation, so both runs use their own workflow name rather
    // than the enclosing trace's name.
    let names: Vec<String> = of_type(&rec, "task")
        .iter()
        .map(|t| {
            export_of(t)["span_data"]["data"]["name"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(names, vec!["inner_workflow", "inner_workflow"]);
}

#[tokio::test]
async fn wrapped_streaming_run_can_disable_task_and_turn_spans() {
    let (_lock, rec, _p) = recording().await;
    let agent =
        Agent::<()>::new("test_agent").with_model_instance(scripted(vec![text("done")], true));
    let mut final_output = None;

    trace("test_workflow")
        .run(async {
            let result = drive_text(
                &agent,
                "first_test",
                true,
                RunOptions::new(()).run_config(RunConfig {
                    include_task_and_turn_spans: false,
                    ..RunConfig::new()
                }),
            )
            .await;
            final_output = result.final_output;
        })
        .await;

    let spans = rec.ordered_spans();
    let agents = of_type(&rec, "agent");
    let generations = of_type(&rec, "generation");
    assert_eq!(final_output.as_deref(), Some("done"));
    assert!(
        !spans
            .iter()
            .any(|s| matches!(span_type(s).as_str(), "task" | "turn"))
    );
    assert_eq!(agents.len(), 1);
    assert!(export_of(&agents[0])["parent_id"].is_null());
    assert_eq!(generations.len(), 1);
    assert_eq!(
        export_of(&generations[0])["parent_id"],
        export_of(&agents[0])["id"]
    );
}

#[tokio::test]
async fn wrapped_mixed_trace_is_single_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent_1").with_model_instance(scripted(
        vec![text("first_test"), text("second_test"), text("third_test")],
        false,
    ));

    trace("test_workflow")
        .run(async {
            run_once(&agent, "first_test", true).await;
            run_once(&agent, "second_test", false).await;
            run_once(&agent, "third_test", true).await;
        })
        .await;

    let child = agent_child("test_agent_1", &[], &[]);
    assert_eq!(
        rec.normalized(false, false),
        vec![json!({
            "workflow_name": "test_workflow",
            "children": [child.clone(), child.clone(), child],
        })]
    );
}

#[tokio::test]
async fn parent_disabled_trace_disables_streaming_agent_trace() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent").with_model_instance(scripted(
        vec![text("first_test"), text("second_test")],
        false,
    ));

    trace("test_workflow")
        .disabled(true)
        .run(async {
            run_once(&agent, "first_test", true).await;
        })
        .await;

    assert!(rec.traces().is_empty());
    assert!(rec.ordered_spans().is_empty());
}

#[tokio::test]
async fn manual_streaming_disabling_works() {
    let (_lock, rec, _p) = recording().await;
    let agent = Agent::<()>::new("test_agent").with_model_instance(scripted(
        vec![text("first_test"), text("second_test")],
        false,
    ));

    drive_text(
        &agent,
        "first_test",
        true,
        RunOptions::new(()).run_config(RunConfig::new().with_tracing_disabled(true)),
    )
    .await;

    assert!(rec.traces().is_empty());
    assert!(rec.ordered_spans().is_empty());
}
