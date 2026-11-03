//! A dynamic approval rule is never consulted for a tool whose argument type runs application
//! code while parsing, once the tool says so with `FunctionTool::with_args_have_validators`.
//! Python skips the rule for any argument model with custom validators, validator plugins,
//! `model_post_init` or default factories (`can_prepare_without_user_code`), because inspecting
//! the arguments would run that code; the call pauses for a manual decision first, and the
//! validation happens only after approval.
//!
//! Rust cannot see a `deserialize_with` function or a custom `Deserialize`, so the tool declares
//! it. Every case of `test_function_tool_approval_arguments.py` that puts application code in
//! argument validation is exercised here with a rule that records itself: the recorded events
//! must never contain `policy`, before the decision or after it. The neighbouring file
//! `py_core9_r9h3_validation.rs` holds the same cases without the marker, where the parse still
//! never runs before approval.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{RunInput, RunItemKind};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunResult, RunState, Runner};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

type Log<T> = Arc<Mutex<Vec<T>>>;

fn log<T>() -> Log<T> {
    Arc::new(Mutex::new(Vec::new()))
}

/// Validation and body events of the case that holds `GUARD`.
static EVENTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
static GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static KIND: AtomicUsize = AtomicUsize::new(0);

fn record(event: &'static str) {
    EVENTS.lock().unwrap_or_else(|e| e.into_inner()).push(event);
}

fn events() -> Vec<&'static str> {
    EVENTS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn reset() {
    EVENTS.lock().unwrap_or_else(|e| e.into_inner()).clear();
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn lower<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    record("validate");
    Ok(String::deserialize(d)?.to_lowercase())
}

fn lower_all<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    record("validate");
    Ok(Vec::<String>::deserialize(d)?
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect())
}

fn checked_number<'de, D: Deserializer<'de>>(d: D) -> Result<i64, D::Error> {
    record("validate");
    i64::deserialize(d)
}

/// Raises on anything but a string, as `value.lower()` raises on an int.
fn strict_text<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    record("validate");
    match Value::deserialize(d)? {
        Value::String(s) => Ok(s.to_lowercase()),
        other => Err(serde::de::Error::custom(format!(
            "'{other}' has no attribute 'lower'"
        ))),
    }
}

fn transform_targets<'de, D: Deserializer<'de>>(d: D) -> Result<HashMap<String, String>, D::Error> {
    record("validate");
    let mut targets = HashMap::<String, String>::deserialize(d)?;
    match KIND.load(Ordering::SeqCst) {
        0 => targets.values_mut().for_each(|v| *v = v.to_lowercase()),
        1 => {
            let sorted: BTreeMap<_, _> = targets.into_iter().collect();
            targets = sorted.into_iter().collect();
        }
        2 => {
            targets.insert("z".into(), "prod".into());
        }
        _ => {}
    }
    Ok(targets)
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Annotated {
    #[serde(deserialize_with = "lower_all")]
    value: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Inner {
    #[serde(deserialize_with = "lower")]
    value: String,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Nested {
    value: Vec<Inner>,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct PostInit {
    value: String,
}

impl<'de> Deserialize<'de> for PostInit {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            value: String,
        }
        let raw = Raw::deserialize(d)?;
        record("validate");
        Ok(PostInit { value: raw.value })
    }
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct InitArgs {
    value: PostInit,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
#[serde(untagged)]
enum IntOrText {
    Int(i64),
    Text(#[serde(deserialize_with = "lower")] String),
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct UnionArgs {
    value: IntOrText,
}

fn operation<A>(body: fn(A) -> String) -> FunctionTool<()>
where
    A: for<'de> Deserialize<'de> + JsonSchema + Send + 'static,
{
    operation_with(true, body)
}

fn operation_with<A>(strict: bool, body: fn(A) -> String) -> FunctionTool<()>
where
    A: for<'de> Deserialize<'de> + JsonSchema + Send + 'static,
{
    FunctionTool::<()>::try_new_with_strictness(
        "operation",
        "",
        strict,
        move |_ctx, args: A| async move {
            record("body");
            Ok(body(args))
        },
    )
    .unwrap()
    .with_args_have_validators(true)
    .with_needs_approval_fn(|_ctx, _args, _id| async {
        record("policy");
        // Whatever the rule would answer, the marker makes the call pause first.
        false
    })
}

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
) -> openai_agents_sdk::Result<RunResult<()>> {
    if streamed {
        let mut stream = Runner::run_streamed(agent, input);
        stream.finish().await?;
        Ok(stream.into_result().expect("streamed result"))
    } else {
        Runner::run(agent, input).await
    }
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

/// Run until the call pauses for approval and return the decided, serialized state.
async fn decided_state(agent: &Agent<()>, streamed: bool, approve: bool, label: &str) -> RunState {
    let paused = run_agent(agent, "go", streamed).await.unwrap();
    assert!(!paused.interruptions.is_empty(), "{label}");
    assert!(events().is_empty(), "{label}: ran before the decision");
    let mut state = RunState::from_json_str(&paused.to_state().to_json_string().unwrap()).unwrap();
    let pending = state.get_interruptions()[0].clone();
    if approve {
        state.approve(&pending, false);
    } else {
        state.reject(&pending, false, None);
    }
    state
}

async fn waits_for_approval<A>(arguments: &str, body: fn(A) -> String)
where
    A: for<'de> Deserialize<'de> + JsonSchema + Send + 'static,
{
    for streamed in [false, true] {
        for approve in [false, true] {
            reset();
            let label = format!("streamed={streamed} approve={approve}");
            let agent = scripted_agent(operation::<A>(body), arguments);
            let state = decided_state(&agent, streamed, approve, &label).await;
            let result = run_agent(&agent, state, streamed).await.unwrap();
            assert!(result.interruptions.is_empty(), "{label}");
            let want: &[&str] = if approve { &["validate", "body"] } else { &[] };
            assert_eq!(events(), want, "{label}");
        }
    }
}

/// Python: `test_application_validation_waits_for_explicit_approval[annotated]` (streamed x
/// approve): a validator on every list item.
#[tokio::test]
async fn an_item_validator_waits_for_approval() {
    let _g = GUARD.lock().await;
    waits_for_approval::<Annotated>(r#"{"value":["SAFE"]}"#, |_| "ok".into()).await;
}

/// Python: `..._waits_for_explicit_approval[nested]`: a validator on a nested model.
#[tokio::test]
async fn a_nested_model_validator_waits_for_approval() {
    let _g = GUARD.lock().await;
    waits_for_approval::<Nested>(r#"{"value":[{"value":"SAFE"}]}"#, |_| "ok".into()).await;
}

/// Python: `..._waits_for_explicit_approval[post_init]`: a hook that runs when a model is built.
#[tokio::test]
async fn a_post_construction_hook_waits_for_approval() {
    let _g = GUARD.lock().await;
    waits_for_approval::<InitArgs>(r#"{"value":{"value":"safe"}}"#, |_| "ok".into()).await;
}

/// Python: `..._waits_for_explicit_approval[union]`: a validator on one branch of a union.
#[tokio::test]
async fn a_union_branch_validator_waits_for_approval() {
    let _g = GUARD.lock().await;
    waits_for_approval::<UnionArgs>(r#"{"value":"SAFE"}"#, |_| "ok".into()).await;
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Targets {
    #[serde(deserialize_with = "transform_targets")]
    targets: HashMap<String, String>,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct TargetsRequest {
    request: Targets,
}

/// Python: `test_nested_transformations_require_approval` (case, order, in_place, unchanged). A
/// validator that rewrites the input never runs for the approval decision, nor when the call is
/// rejected.
#[tokio::test]
async fn a_rewriting_validator_never_runs_for_a_rejected_call() {
    let _g = GUARD.lock().await;
    for (kind, name) in ["case", "order", "in_place", "unchanged"]
        .into_iter()
        .enumerate()
    {
        reset();
        KIND.store(kind, Ordering::SeqCst);
        let raw = if name == "case" {
            r#"{"request":{"targets":{"z":"PROD","a":"prod"}}}"#
        } else {
            r#"{"request":{"targets":{"z":"test","a":"prod"}}}"#
        };
        let agent = scripted_agent(
            operation_with::<TargetsRequest>(false, |_| "ok".into()),
            raw,
        );
        let state = decided_state(&agent, false, false, name).await;
        let result = run_agent(&agent, state, false).await.unwrap();
        assert!(result.interruptions.is_empty(), "{name}");
        assert!(events().is_empty(), "{name}: {:?}", events());
    }
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NumberValue {
    #[serde(deserialize_with = "checked_number")]
    value: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Runner,
    Streamed,
    #[cfg(feature = "realtime")]
    Realtime,
}

#[allow(unused_mut)]
fn modes() -> Vec<Mode> {
    let mut modes = vec![Mode::Runner, Mode::Streamed];
    #[cfg(feature = "realtime")]
    modes.push(Mode::Realtime);
    modes
}

#[cfg(feature = "realtime")]
mod rt {
    use std::time::Duration;

    use openai_agents_sdk::realtime::{
        FakeRealtimeModel, RealtimeAgent, RealtimeModelEvent, RealtimeModelToolCallEvent,
        RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
    };
    use openai_agents_sdk::tool::FunctionTool;
    use serde_json::json;

    const WAIT: Duration = Duration::from_secs(5);

    pub async fn start(
        model: &std::sync::Arc<FakeRealtimeModel>,
        tool: FunctionTool<()>,
    ) -> RealtimeSession<()> {
        let agent = RealtimeAgent::<()>::new("test").with_tool(tool).into_arc();
        let config = RealtimeRunConfig {
            async_tool_calls: false,
            ..Default::default()
        };
        RealtimeSession::connect(model.clone(), agent, (), None, Some(config))
            .await
            .expect("session connects")
    }

    /// Emit a call and wait until the session handled it; returns the approval requests seen.
    pub async fn handle_call(
        model: &FakeRealtimeModel,
        session: &RealtimeSession<()>,
        arguments: &str,
    ) -> usize {
        model.emit(RealtimeModelEvent::FunctionCall(
            RealtimeModelToolCallEvent::new("operation", "one", arguments),
        ));
        model.emit(RealtimeModelEvent::Other {
            data: json!("barrier"),
        });
        let mut approvals = 0;
        loop {
            let event = tokio::time::timeout(WAIT, session.recv())
                .await
                .expect("timed out")
                .expect("stream ended");
            match event {
                RealtimeSessionEvent::ToolApprovalRequired { .. } => approvals += 1,
                RealtimeSessionEvent::RawModelEvent {
                    data: RealtimeModelEvent::Other { data },
                    ..
                } if data == "barrier" => return approvals,
                _ => {}
            }
        }
    }

    pub fn sent_outputs(model: &FakeRealtimeModel) -> Vec<String> {
        model
            .sent_events()
            .iter()
            .filter_map(|e| e.as_tool_output().map(|(_, output, _)| output.to_string()))
            .collect()
    }
}

/// Drive one pausing call in `mode`, decide it, and return the outputs. `Err` carries the error
/// of an approved call whose failure policy re-raises.
async fn drive(
    mode: Mode,
    tool: FunctionTool<()>,
    arguments: &str,
    approve: bool,
    label: &str,
) -> Result<Vec<String>, AgentsError> {
    match mode {
        Mode::Runner | Mode::Streamed => {
            let streamed = mode == Mode::Streamed;
            let agent = scripted_agent(tool, arguments);
            let state = decided_state(&agent, streamed, approve, label).await;
            let result = run_agent(&agent, state, streamed).await?;
            assert!(result.interruptions.is_empty(), "{label}");
            Ok(outputs(&result))
        }
        #[cfg(feature = "realtime")]
        Mode::Realtime => {
            use openai_agents_sdk::realtime::FakeRealtimeModel;
            let model = Arc::new(FakeRealtimeModel::new());
            let session = rt::start(&model, tool).await;
            let approvals = rt::handle_call(&model, &session, arguments).await;
            assert_eq!(approvals, 1, "{label}");
            assert!(events().is_empty(), "{label}: ran before the decision");
            assert!(rt::sent_outputs(&model).is_empty(), "{label}");
            if approve {
                session.approve_tool_call("one", false).await?;
            } else {
                session.reject_tool_call("one", false, None).await?;
            }
            Ok(rt::sent_outputs(&model))
        }
    }
}

/// Python: `test_plugin_validation_waits_for_approval` (runner, streamed, realtime x approve).
/// The validator hook runs once, between the approval and the body.
#[tokio::test]
async fn a_validation_hook_runs_between_the_approval_and_the_body() {
    let _g = GUARD.lock().await;
    for mode in modes() {
        for approve in [false, true] {
            reset();
            let label = format!("{mode:?} approve={approve}");
            let tool = operation::<NumberValue>(|args| args.value.to_string());
            let out = drive(mode, tool, r#"{"value":1}"#, approve, &label)
                .await
                .unwrap();
            let want: &[&str] = if approve { &["validate", "body"] } else { &[] };
            assert_eq!(events(), want, "{label}");
            assert_eq!(out.len(), 1, "{label}");
            if approve {
                assert_eq!(out, ["1"], "{label}");
            }
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Patterned {
    #[schemars(inner(regex(pattern = "(?i)^a+$")))]
    value: Vec<String>,
}

/// Python: `test_compiled_pattern_requires_explicit_approval` (runner, streamed, realtime x
/// approve), with the case-insensitive pattern in the schema. The pattern is checked after the
/// approval; a matching value reaches the body.
#[tokio::test]
async fn a_case_insensitive_pattern_is_checked_after_approval() {
    let _g = GUARD.lock().await;
    for mode in modes() {
        for approve in [false, true] {
            reset();
            let label = format!("{mode:?} approve={approve}");
            let tool = operation::<Patterned>(|args| args.value[0].clone());
            let out = drive(mode, tool, r#"{"value":["AAA"]}"#, approve, &label)
                .await
                .unwrap();
            let want: &[&str] = if approve { &["body"] } else { &[] };
            assert_eq!(events(), want, "{label}");
            assert_eq!(out.len(), 1, "{label}");
            if approve {
                assert_eq!(out, ["AAA"], "{label}");
            }
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Checked {
    #[serde(deserialize_with = "strict_text")]
    value: String,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct CheckedRequest {
    request: Checked,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Failure {
    Default,
    Custom,
    Raise,
}

/// Python: `test_validator_exception_preserves_failure_policy` (failure x runner, streamed,
/// realtime). A validator that fails on a value of the wrong kind runs only after the approval and
/// its failure goes through the tool's failure policy. (Python's `AttributeError` is a
/// `ModelBehaviorError` here: every parse failure is one.)
#[tokio::test]
async fn a_failing_validator_uses_the_failure_policy_after_approval() {
    let _g = GUARD.lock().await;
    for failure in [Failure::Default, Failure::Custom, Failure::Raise] {
        for mode in modes() {
            reset();
            let label = format!("{failure:?} {mode:?}");
            let errors = log::<bool>();
            let mut tool = operation::<CheckedRequest>(|args| args.request.value);
            match failure {
                Failure::Default => {}
                Failure::Custom => {
                    let errors = errors.clone();
                    tool = tool.with_failure_error_function(move |_ctx, error| {
                        errors
                            .lock()
                            .unwrap()
                            .push(matches!(error, AgentsError::ModelBehaviorError { .. }));
                        "correct value".to_string()
                    });
                }
                Failure::Raise => tool = tool.without_failure_error_function(),
            }
            let result = drive(mode, tool, r#"{"request":{"value":3}}"#, true, &label).await;
            if failure == Failure::Raise {
                let err = result.expect_err(&label);
                assert!(
                    matches!(err, AgentsError::ModelBehaviorError { .. }),
                    "{label}: {err:?}"
                );
            } else {
                let out = result.unwrap();
                assert_eq!(out.len(), 1, "{label}");
                if failure == Failure::Custom {
                    assert_eq!(out, ["correct value"], "{label}");
                }
            }
            assert_eq!(events(), ["validate"], "{label}: the body must not run");
            let want: &[bool] = if failure == Failure::Custom {
                &[true]
            } else {
                &[]
            };
            assert_eq!(*errors.lock().unwrap(), want, "{label}");
        }
    }
}

#[cfg(feature = "realtime")]
/// Python: `test_realtime_application_validation_waits_for_approval` (approve): a validator on
/// every list item in a realtime session.
#[tokio::test]
async fn a_realtime_item_validator_waits_for_approval() {
    let _g = GUARD.lock().await;
    for approve in [false, true] {
        reset();
        let tool = operation::<Annotated>(|args| args.value[0].clone());
        let out = drive(
            Mode::Realtime,
            tool,
            r#"{"value":["SAFE"]}"#,
            approve,
            &format!("approve={approve}"),
        )
        .await
        .unwrap();
        let want: &[&str] = if approve { &["validate", "body"] } else { &[] };
        assert_eq!(events(), want, "approve={approve}");
        assert_eq!(out.len(), 1);
    }
}

/// Without the marker the same tool is judged by its schema alone: the arguments match, so the
/// rule is consulted (and here lets the call through), and the validator then runs with the body.
#[tokio::test]
async fn without_the_marker_the_rule_decides_for_arguments_that_match_the_schema() {
    let _g = GUARD.lock().await;
    reset();
    let tool = FunctionTool::<()>::try_new_with_strictness(
        "operation",
        "",
        true,
        |_ctx, args: Annotated| async move {
            record("body");
            Ok(args.value.join(","))
        },
    )
    .unwrap()
    .with_needs_approval_fn(|_ctx, _args, _id| async {
        record("policy");
        false
    });
    let agent = scripted_agent(tool, r#"{"value":["SAFE"]}"#);
    let result = run_agent(&agent, "go", false).await.unwrap();
    assert!(result.interruptions.is_empty());
    assert_eq!(events(), ["policy", "validate", "body"]);
    assert_eq!(outputs(&result), ["safe"]);
}

/// The marker only changes what a dynamic rule sees. A static `needs_approval` is not a rule:
/// `false` still runs the tool without a pause (Python never prepares the arguments for it).
#[tokio::test]
async fn the_marker_does_not_affect_a_static_setting() {
    let _g = GUARD.lock().await;
    for needs_approval in [false, true] {
        reset();
        let tool = FunctionTool::<()>::try_new_with_strictness(
            "operation",
            "",
            true,
            |_ctx, args: Annotated| async move {
                record("body");
                Ok(args.value.join(","))
            },
        )
        .unwrap()
        .with_args_have_validators(true)
        .with_needs_approval(needs_approval);
        let agent = scripted_agent(tool, r#"{"value":["SAFE"]}"#);
        let result = run_agent(&agent, "go", false).await.unwrap();
        assert_eq!(
            !result.interruptions.is_empty(),
            needs_approval,
            "needs_approval={needs_approval}"
        );
        let want: &[&str] = if needs_approval {
            &[]
        } else {
            &["validate", "body"]
        };
        assert_eq!(events(), want, "needs_approval={needs_approval}");
    }
}
