//! Ports of `tests/test_function_tool_approval_arguments.py`: a conditional approval rule
//! observes exactly the invocation that can execute.
//!
//! Python prepares a call's arguments with the tool's pydantic model and consults the rule only
//! when the prepared arguments equal the raw ones. Rust's equivalent for a typed tool
//! (`FunctionTool::new`) is: the arguments parse into the tool's argument type and name every
//! declared parameter and nothing else. Anything else (a default the tool would fill in, an
//! undeclared key, a value that does not parse) requires approval without consulting the rule,
//! and the parse error surfaces only after approval, through the tool's failure policy.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::items::{RunInput, RunItemKind};
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, Tool};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{Agent, AgentsError, RunResult, RunState, Runner};
use serde::Deserialize;
use serde_json::{Value, json};

type Log<T> = Arc<Mutex<Vec<T>>>;

fn log<T>() -> Log<T> {
    Arc::new(Mutex::new(Vec::new()))
}

fn setup() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

/// An agent whose model calls `tool` once with `arguments`, then answers `done`.
fn scripted_agent(tool: FunctionTool<()>, arguments: &str) -> Agent<()> {
    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![
        vec![function_call_with_id(
            tool.name.clone(),
            arguments,
            "call_test",
        )],
        vec![assistant_message("done")],
    ]);
    Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(tool)
}

async fn run_agent(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    streamed: bool,
    hooks: Option<Arc<dyn RunHooks<()>>>,
) -> openai_agents_sdk::Result<RunResult<()>> {
    let mut options = RunOptions::default();
    if let Some(hooks) = hooks {
        options = options.hooks(hooks);
    }
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

fn outputs(result: &RunResult<()>) -> Vec<String> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(
                output
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| output.to_string()),
            ),
            _ => None,
        })
        .collect()
}

/// A dynamic rule that records the arguments it sees and answers with `decide`.
fn with_policy(
    tool: FunctionTool<()>,
    seen: &Log<Value>,
    decide: fn(&Value) -> bool,
) -> FunctionTool<()> {
    let seen = seen.clone();
    tool.with_needs_approval_fn(move |_ctx, args, _id| {
        let seen = seen.clone();
        async move {
            let decision = decide(&args);
            seen.lock().unwrap().push(args);
            decision
        }
    })
}

#[derive(Deserialize, schemars::JsonSchema)]
struct EndpointArgs {
    #[serde(rename = "endpoint", default = "protected")]
    target: String,
}

fn protected() -> String {
    "protected".into()
}

/// Python: `test_defaults_and_alias_keep_existing_policy` (streamed x 3 argument cases).
#[tokio::test]
async fn defaults_and_alias_keep_existing_policy() {
    setup();
    for streamed in [false, true] {
        for (arguments, expected_calls, interrupted) in [
            ("{}", 0, true),
            (r#"{"endpoint":"protected"}"#, 1, true),
            (r#"{"endpoint":"safe"}"#, 1, false),
        ] {
            let seen = log();
            let executed = log::<String>();
            let e = executed.clone();
            let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: EndpointArgs| {
                let e = e.clone();
                async move {
                    e.lock().unwrap().push(args.target.clone());
                    Ok(args.target)
                }
            });
            let tool = with_policy(tool, &seen, |p| p["endpoint"] == "protected");
            let result = run_agent(&scripted_agent(tool, arguments), "go", streamed, None)
                .await
                .unwrap();
            let label = format!("streamed={streamed} {arguments}");
            assert_eq!(!result.interruptions.is_empty(), interrupted, "{label}");
            assert_eq!(seen.lock().unwrap().len(), expected_calls, "{label}");
            let want: &[&str] = if interrupted { &[] } else { &["safe"] };
            assert_eq!(*executed.lock().unwrap(), want, "{label}");
        }
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct EnvironmentArgs {
    #[serde(default = "safe")]
    environment: String,
}

fn safe() -> String {
    "safe".into()
}

fn environment_tool(executed: &Log<String>) -> FunctionTool<()> {
    let e = executed.clone();
    FunctionTool::<()>::new("operation", "", move |_ctx, args: EnvironmentArgs| {
        let e = e.clone();
        async move {
            e.lock().unwrap().push(args.environment.clone());
            Ok(args.environment)
        }
    })
}

/// Python: `test_benign_omission_requires_approval_and_explicit_input_runs` (streamed).
#[tokio::test]
async fn benign_omission_requires_approval_and_explicit_input_runs() {
    setup();
    for streamed in [false, true] {
        let seen = log();
        let executed = log();
        let tool = || {
            with_policy(environment_tool(&executed), &seen, |p| {
                p.get("environment").is_none()
            })
        };
        let omitted = run_agent(&scripted_agent(tool(), "{}"), "go", streamed, None)
            .await
            .unwrap();
        assert!(!omitted.interruptions.is_empty());
        assert!(seen.lock().unwrap().is_empty() && executed.lock().unwrap().is_empty());
        let explicit = run_agent(
            &scripted_agent(tool(), r#"{"environment":"safe"}"#),
            "go",
            streamed,
            None,
        )
        .await
        .unwrap();
        assert_eq!(explicit.final_output_text().as_deref(), Some("done"));
        assert!(explicit.interruptions.is_empty());
        assert_eq!(*seen.lock().unwrap(), [json!({"environment": "safe"})]);
        assert_eq!(*executed.lock().unwrap(), ["safe"]);
    }
}

/// Python: `test_manual_decision_roundtrip_does_not_rerun_predicate` (streamed x serialize x
/// approve).
#[tokio::test]
async fn manual_decision_roundtrip_does_not_rerun_predicate() {
    setup();
    for streamed in [false, true] {
        for serialize in [false, true] {
            for approve in [false, true] {
                let label = format!("streamed={streamed} serialize={serialize} approve={approve}");
                let calls = log();
                let effects = log();
                let tool = with_policy(environment_tool(&effects), &calls, |_| false);
                let agent = scripted_agent(tool, "{}");
                let paused = run_agent(&agent, "go", streamed, None).await.unwrap();
                let mut state = paused.to_state();
                if serialize {
                    state = roundtrip(&state);
                }
                let pending = state.get_interruptions()[0].clone();
                if approve {
                    state.approve(&pending, false);
                } else {
                    state.reject(&pending, false, None);
                }
                let result = run_agent(&agent, state, streamed, None).await.unwrap();
                assert!(result.interruptions.is_empty(), "{label}");
                assert!(calls.lock().unwrap().is_empty(), "{label}");
                let want: &[&str] = if approve { &["safe"] } else { &[] };
                assert_eq!(*effects.lock().unwrap(), want, "{label}");
            }
        }
    }
}

struct StartHooks(Log<&'static str>);

#[async_trait]
impl RunHooks<()> for StartHooks {
    async fn on_tool_start(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &Tool<()>,
    ) {
        self.0.lock().unwrap().push("start");
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct NumberArgs {
    number: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Failure {
    Default,
    Custom,
    Raise,
}

/// Python: `test_invalid_typed_input_requires_approval_before_failure_policy` (failure x
/// {runner, streamed} x approve). The realtime mode is listed in the parity report.
#[tokio::test]
async fn invalid_typed_input_requires_approval_before_failure_policy() {
    setup();
    for failure in [Failure::Default, Failure::Custom, Failure::Raise] {
        for streamed in [false, true] {
            for approve in [false, true] {
                let label = format!("{failure:?} streamed={streamed} approve={approve}");
                let seen = log();
                let effects = log::<i64>();
                let formatted = log::<bool>();
                let callbacks = log::<&'static str>();
                let cb = callbacks.clone();
                let guardrail = ToolInputGuardrail::<()>::new("guardrail", move |_data| {
                    let cb = cb.clone();
                    async move {
                        cb.lock().unwrap().push("guardrail");
                        Ok(ToolGuardrailFunctionOutput::allow())
                    }
                });
                let e = effects.clone();
                let mut tool =
                    FunctionTool::<()>::new("operation", "", move |_ctx, args: NumberArgs| {
                        let e = e.clone();
                        async move {
                            e.lock().unwrap().push(args.number);
                            Ok(args.number.to_string())
                        }
                    })
                    .with_input_guardrail(guardrail);
                tool = with_policy(tool, &seen, |_| true);
                match failure {
                    Failure::Default => {}
                    Failure::Custom => {
                        let f = formatted.clone();
                        tool = tool.with_failure_error_function(move |_ctx, error| {
                            f.lock()
                                .unwrap()
                                .push(matches!(error, AgentsError::ModelBehaviorError { .. }));
                            "correct the number".to_string()
                        });
                    }
                    Failure::Raise => tool = tool.without_failure_error_function(),
                }
                let hooks =
                    || Some(Arc::new(StartHooks(callbacks.clone())) as Arc<dyn RunHooks<()>>);
                let agent = scripted_agent(tool, r#"{"number":"invalid"}"#);
                let paused = run_agent(&agent, "go", streamed, hooks()).await.unwrap();
                assert!(!paused.interruptions.is_empty(), "{label}");
                assert!(seen.lock().unwrap().is_empty(), "{label}");
                assert!(formatted.lock().unwrap().is_empty(), "{label}");
                assert!(callbacks.lock().unwrap().is_empty(), "{label}");
                assert!(effects.lock().unwrap().is_empty(), "{label}");
                let mut state = roundtrip(&paused.to_state());
                let pending = state.get_interruptions()[0].clone();
                if !approve {
                    state.reject(&pending, false, None);
                    let result = run_agent(&agent, state, streamed, hooks()).await.unwrap();
                    assert!(result.interruptions.is_empty(), "{label}");
                    assert!(callbacks.lock().unwrap().is_empty(), "{label}");
                } else {
                    state.approve(&pending, false);
                    if failure == Failure::Raise {
                        let err = run_agent(&agent, state, streamed, hooks())
                            .await
                            .unwrap_err();
                        assert!(
                            matches!(err, AgentsError::ModelBehaviorError { .. }),
                            "{label}: {err:?}"
                        );
                    } else {
                        let result = run_agent(&agent, state, streamed, hooks()).await.unwrap();
                        assert!(result.interruptions.is_empty(), "{label}");
                        let out = outputs(&result);
                        assert_eq!(out.len(), 1, "{label}");
                        if failure == Failure::Custom {
                            assert_eq!(out, ["correct the number"], "{label}");
                        }
                    }
                    assert!(callbacks.lock().unwrap().contains(&"start"), "{label}");
                }
                assert!(seen.lock().unwrap().is_empty(), "{label}");
                assert!(effects.lock().unwrap().is_empty(), "{label}");
                let want: &[bool] = if approve && failure == Failure::Custom {
                    &[true]
                } else {
                    &[]
                };
                assert_eq!(*formatted.lock().unwrap(), want, "{label}");
                let guardrails = callbacks
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|c| **c == "guardrail")
                    .count();
                assert_eq!(guardrails, usize::from(approve), "{label}");
            }
        }
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RequestArgs {
    request: HashMap<String, String>,
}

/// Python: `test_preparation_is_preserved_on_copied_tools` (`copy.copy` and
/// `dataclasses.replace`; both are a `Clone` of the tool in Rust, the second with a field
/// replaced). A rule that edits the arguments it was given cannot change the invocation.
#[tokio::test]
async fn preparation_is_preserved_on_copied_tools() {
    setup();
    for replace in [false, true] {
        let seen = log::<Value>();
        let effects = log::<String>();
        let e = effects.clone();
        let s = seen.clone();
        let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: RequestArgs| {
            let e = e.clone();
            async move {
                let environment = args.request["environment"].clone();
                e.lock().unwrap().push(environment.clone());
                Ok(environment)
            }
        })
        .with_needs_approval_fn(move |_ctx, mut params, _id| {
            let s = s.clone();
            async move {
                s.lock().unwrap().push(params.clone());
                // A rule can mutate its own input without changing the prepared arguments.
                params["request"]["environment"] = json!("changed");
                false
            }
        });
        let mut copied = tool.clone();
        if replace {
            copied.description = "replaced".into();
        }
        let result = run_agent(
            &scripted_agent(copied, r#"{"request":{"environment":"safe"}}"#),
            "go",
            false,
            None,
        )
        .await
        .unwrap();
        assert!(result.interruptions.is_empty());
        assert_eq!(*effects.lock().unwrap(), ["safe"]);
        assert_eq!(
            *seen.lock().unwrap(),
            [json!({"request": {"environment": "safe"}})]
        );
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ValueRequest {
    request: HashMap<String, i64>,
}

/// Python: `test_parallel_calls_keep_distinct_prepared_values`: the rules of two calls run
/// concurrently and each call executes with its own arguments.
#[tokio::test]
async fn parallel_calls_keep_distinct_prepared_values() {
    setup();
    let first_started = Arc::new(tokio::sync::Notify::new());
    let second_finished = Arc::new(tokio::sync::Notify::new());
    let executed = log::<i64>();
    let e = executed.clone();
    let (fs, sf) = (first_started.clone(), second_finished.clone());
    let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: ValueRequest| {
        let e = e.clone();
        async move {
            let value = args.request["value"];
            e.lock().unwrap().push(value);
            Ok(value.to_string())
        }
    })
    .with_needs_approval_fn(move |_ctx, params, _id| {
        let (fs, sf) = (fs.clone(), sf.clone());
        async move {
            if params["request"]["value"] == 1 {
                fs.notify_one();
                sf.notified().await;
            } else {
                fs.notified().await;
                sf.notify_one();
            }
            false
        }
    });
    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![
        vec![
            function_call_with_id("operation", r#"{"request":{"value":1}}"#, "one"),
            function_call_with_id("operation", r#"{"request":{"value":2}}"#, "two"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(tool);
    let result = tokio::time::timeout(Duration::from_secs(5), run_agent(&agent, "go", false, None))
        .await
        .expect("the two rules ran concurrently")
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(*executed.lock().unwrap(), [2, 1]);
}

#[derive(Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct IntValue {
    value: i64,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct FloatValue {
    value: f64,
}

/// Python: `test_changed_or_uninspectable_argument_requires_manual_approval`, the cases with a
/// Rust counterpart: a string or a boolean for an integer, an undeclared key, and a float that
/// needs no conversion.
#[tokio::test]
async fn changed_or_uninspectable_argument_requires_manual_approval() {
    setup();
    async fn case<A>(arguments: &str, interrupted: bool)
    where
        A: for<'de> Deserialize<'de> + schemars::JsonSchema + Send + 'static,
    {
        let seen = log();
        let effects = log::<()>();
        let e = effects.clone();
        let tool = FunctionTool::<()>::new("operation", "", move |_ctx, _args: A| {
            let e = e.clone();
            async move {
                e.lock().unwrap().push(());
                Ok("ok".to_string())
            }
        });
        let tool = with_policy(tool, &seen, |_| false);
        let result = run_agent(&scripted_agent(tool, arguments), "go", false, None)
            .await
            .unwrap();
        assert_eq!(!result.interruptions.is_empty(), interrupted, "{arguments}");
        let expected = usize::from(!interrupted);
        assert_eq!(seen.lock().unwrap().len(), expected, "{arguments}");
        assert_eq!(effects.lock().unwrap().len(), expected, "{arguments}");
    }
    case::<IntValue>(r#"{"value":"1"}"#, true).await;
    case::<IntValue>(r#"{"value":true}"#, true).await;
    case::<FloatValue>(r#"{"value":1.0}"#, false).await;
    case::<IntValue>(r#"{"value":1,"extra":2}"#, true).await;
}

#[derive(Deserialize, schemars::JsonSchema)]
struct TwoArgs {
    first: String,
    second: String,
}

/// Python: `test_root_parameter_order_does_not_change_approval`: the rule sees the keys in the
/// order the model sent them.
#[tokio::test]
async fn root_parameter_order_does_not_change_approval() {
    setup();
    let seen = log::<Vec<String>>();
    let s = seen.clone();
    let tool = FunctionTool::<()>::new("operation", "", |_ctx, args: TwoArgs| async move {
        Ok(args.first + &args.second)
    })
    .with_needs_approval_fn(move |_ctx, params, _id| {
        let s = s.clone();
        async move {
            let keys = params.as_object().unwrap().keys().cloned().collect();
            s.lock().unwrap().push(keys);
            false
        }
    });
    let result = run_agent(
        &scripted_agent(tool, r#"{"second":"b","first":"a"}"#),
        "go",
        false,
        None,
    )
    .await
    .unwrap();
    assert!(result.interruptions.is_empty());
    assert_eq!(*seen.lock().unwrap(), [vec!["second", "first"]]);
}

#[derive(Deserialize, schemars::JsonSchema)]
struct StrValue {
    value: String,
}

/// Python: `test_resume_selection_reuses_preparation_or_requires_approval` (invalid). Python
/// drives the private `resolve_interrupted_turn`; the same selection runs for a call the
/// runner has not decided yet: a valid call consults the rule once and runs, an invalid one
/// requires approval without the rule or the failure formatter.
#[tokio::test]
async fn selection_reuses_preparation_or_requires_approval() {
    setup();
    for invalid in [false, true] {
        let policies = log();
        let effects = log::<String>();
        let errors = log::<()>();
        let e = effects.clone();
        let errs = errors.clone();
        let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: StrValue| {
            let e = e.clone();
            async move {
                e.lock().unwrap().push(args.value.clone());
                Ok(args.value)
            }
        })
        .with_failure_error_function(move |_ctx, _error| {
            errs.lock().unwrap().push(());
            "correct input".to_string()
        });
        let tool = with_policy(tool, &policies, |_| false);
        let arguments = if invalid {
            r#"{"value":3}"#
        } else {
            r#"{"value":"safe"}"#
        };
        let result = run_agent(&scripted_agent(tool, arguments), "go", false, None)
            .await
            .unwrap();
        let want: &[&str] = if invalid { &[] } else { &["safe"] };
        assert_eq!(*effects.lock().unwrap(), want);
        assert_eq!(policies.lock().unwrap().len(), usize::from(!invalid));
        assert!(errors.lock().unwrap().is_empty());
        assert_eq!(!result.interruptions.is_empty(), invalid);
        let want: &[&str] = if invalid { &[] } else { &["safe"] };
        assert_eq!(outputs(&result), want);
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct StrRequest {
    request: HashMap<String, String>,
}

/// Python: `test_cancelled_policy_does_not_reuse_preparation_for_a_new_run`: a run cancelled
/// while its rule waits leaves nothing behind for the next run.
#[tokio::test]
async fn cancelled_policy_does_not_reuse_preparation_for_a_new_run() {
    setup();
    let started = Arc::new(tokio::sync::Notify::new());
    let policies = log::<Value>();
    let effects = log::<String>();
    let make = || {
        let (p, s, e) = (policies.clone(), started.clone(), effects.clone());
        FunctionTool::<()>::new("operation", "", move |_ctx, args: StrRequest| {
            let e = e.clone();
            async move {
                let value = args.request["value"].clone();
                e.lock().unwrap().push(value.clone());
                Ok(value)
            }
        })
        .with_needs_approval_fn(move |_ctx, params, _id| {
            let (p, s) = (p.clone(), s.clone());
            async move {
                let first = {
                    let mut p = p.lock().unwrap();
                    p.push(params);
                    p.len() == 1
                };
                if first {
                    s.notify_one();
                    std::future::pending::<()>().await;
                }
                false
            }
        })
    };
    let agent = scripted_agent(make(), r#"{"request":{"value":"first"}}"#);
    let task = tokio::spawn(async move { run_agent(&agent, "go", false, None).await.map(|_| ()) });
    tokio::time::timeout(Duration::from_secs(1), started.notified())
        .await
        .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let result = run_agent(
        &scripted_agent(make(), r#"{"request":{"value":"second"}}"#),
        "go",
        false,
        None,
    )
    .await
    .unwrap();
    assert!(result.interruptions.is_empty());
    assert_eq!(
        *policies.lock().unwrap(),
        [
            json!({"request": {"value": "first"}}),
            json!({"request": {"value": "second"}})
        ]
    );
    assert_eq!(*effects.lock().unwrap(), ["second"]);
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ListArg {
    value: Vec<String>,
}

/// Python: `test_input_guardrail_blocks_application_validation` (approved): with pre-approval
/// guardrails, a blocking guardrail answers without approval; an allowing one lets the call
/// ask, and the guardrail runs again with the approved invocation.
#[tokio::test]
async fn input_guardrail_blocks_application_validation() {
    use openai_agents_sdk::run_config::{RunConfig, ToolExecutionConfig};
    setup();
    for approved in [false, true] {
        let events = log::<&'static str>();
        let block = Arc::new(Mutex::new(!approved));
        let (ev, bl) = (events.clone(), block.clone());
        let guardrail = ToolInputGuardrail::<()>::new("guardrail", move |_d| {
            let (ev, bl) = (ev.clone(), bl.clone());
            async move {
                ev.lock().unwrap().push("guardrail");
                Ok(if *bl.lock().unwrap() {
                    ToolGuardrailFunctionOutput::reject_content("blocked")
                } else {
                    ToolGuardrailFunctionOutput::allow()
                })
            }
        });
        let ev = events.clone();
        let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: ListArg| {
            let ev = ev.clone();
            async move {
                ev.lock().unwrap().push("body");
                Ok(args.value[0].clone())
            }
        })
        .with_input_guardrail(guardrail);
        let ev = events.clone();
        let tool = tool.with_needs_approval_fn(move |_ctx, _args, _id| {
            let ev = ev.clone();
            async move {
                ev.lock().unwrap().push("policy");
                true
            }
        });
        let agent = scripted_agent(tool, r#"{"value":["safe"]}"#);
        let config = RunConfig::<()>::default().with_tool_execution(ToolExecutionConfig {
            pre_approval_tool_input_guardrails: true,
            ..Default::default()
        });
        let options = || RunOptions::default().run_config(config.clone());
        let mut result = Runner::run_with_options(&agent, "go", options())
            .await
            .unwrap();
        if approved {
            assert!(!result.interruptions.is_empty());
            assert_eq!(*events.lock().unwrap(), ["policy", "guardrail"]);
            let mut state = result.to_state();
            let pending = state.get_interruptions()[0].clone();
            state.approve(&pending, false);
            *block.lock().unwrap() = true;
            result = Runner::run_with_options(&agent, state, options())
                .await
                .unwrap();
            assert_eq!(
                *events.lock().unwrap(),
                ["policy", "guardrail", "guardrail"]
            );
        } else {
            assert_eq!(*events.lock().unwrap(), ["policy", "guardrail"]);
        }
        assert!(result.interruptions.is_empty());
        assert_eq!(outputs(&result), ["blocked"]);
    }
}
