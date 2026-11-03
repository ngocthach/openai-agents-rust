//! Ports of the `execute_approved_tools` tests in `tests/test_agent_runner.py` (Python lines
//! ~6970-7640).
//!
//! Python drives the private `execute_approved_tools` helper with hand-built approval items.
//! Rust has no such seam: an approval decision only reaches a tool through a resumed run, so
//! each port runs the real flow (a `needs_approval` tool interrupts, the state is approved or
//! rejected, the run resumes) and asserts the tool output the resumed run produced. The
//! expected values were checked against the same public-API flow on the Python SDK.

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call_full, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, ToolTimeoutBehavior, tool_namespace};
use openai_agents_sdk::{Agent, AgentsError, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NoArgs {}

type Calls = Arc<Mutex<Vec<String>>>;

fn recording_tool(name: &str, output: &str, calls: &Calls) -> FunctionTool<()> {
    let (output, calls) = (output.to_string(), calls.clone());
    FunctionTool::new(name, "a tool", move |_ctx, _a: NoArgs| {
        let (output, calls) = (output.clone(), calls.clone());
        async move {
            calls.lock().unwrap().push(output.clone());
            Ok(output)
        }
    })
    .with_needs_approval(true)
}

fn tool_outputs<C>(result: &RunResult<C>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Decision {
    Approve,
    Reject,
    None,
}

/// Interrupt on `call`, decide, then resume; returns the resumed run.
async fn decide_and_resume(
    tools: Vec<FunctionTool<()>>,
    call: OutputItem,
    decision: Decision,
    config: impl Fn() -> RunConfig<()>,
    rejection: Option<&str>,
) -> Result<RunResult<()>, AgentsError> {
    let model = setup();
    model.add_turn(vec![call]);
    model.add_turn(vec![assistant_message("done")]);
    let mut agent = Agent::<()>::new("A").with_model_instance(model.clone());
    for t in tools {
        agent = agent.with_tool(t);
    }
    let agent = Arc::new(agent);
    let first = Runner::run_with_options(
        agent.clone(),
        "hi",
        RunOptions::new(()).run_config(config()),
    )
    .await?;
    assert_eq!(first.interruptions.len(), 1);
    let mut state = first.to_state();
    match decision {
        Decision::Approve => state.approve(&first.interruptions[0], false),
        Decision::Reject => state.reject(
            &first.interruptions[0],
            false,
            rejection.map(str::to_string),
        ),
        Decision::None => {}
    }
    let state = RunState::from_json(state.to_json().unwrap()).unwrap();
    Runner::run_with_options(agent, state, RunOptions::new(()).run_config(config())).await
}

fn formatter_config() -> RunConfig<()> {
    let mut c = RunConfig::default();
    c.tool_error_formatter = Some(openai_agents_sdk::tool_error_formatter_fn(|args| {
        Some(format!(
            "run-level {} denied ({})",
            args.tool_name, args.call_id
        ))
    }));
    c
}

/// Python: `test_execute_approved_tools_with_rejected_tool`.
#[tokio::test]
async fn a_rejected_tool_is_not_run() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![recording_tool("test_tool", "tool_result", &calls)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::Reject,
        RunConfig::default,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        tool_outputs(&r),
        vec![json!("Tool execution was not approved.")]
    );
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_execute_approved_tools_with_rejected_tool_uses_run_level_formatter`.
#[tokio::test]
async fn a_rejection_uses_the_run_level_formatter() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![recording_tool("test_tool", "tool_result", &calls)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::Reject,
        formatter_config,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        tool_outputs(&r),
        vec![json!("run-level test_tool denied (c1)")]
    );
}

/// Python: `test_execute_approved_tools_with_rejected_tool_prefers_explicit_message`.
#[tokio::test]
async fn an_explicit_rejection_message_beats_the_formatter() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![recording_tool("test_tool", "tool_result", &calls)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::Reject,
        formatter_config,
        Some("explicit rejection message"),
    )
    .await
    .unwrap();
    assert_eq!(tool_outputs(&r), vec![json!("explicit rejection message")]);
}

/// Python: `test_execute_approved_tools_with_rejected_tool_formatter_none_uses_default`.
#[tokio::test]
async fn a_formatter_returning_none_keeps_the_default_rejection() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![recording_tool("test_tool", "tool_result", &calls)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::Reject,
        || {
            let mut c = RunConfig::default();
            c.tool_error_formatter = Some(openai_agents_sdk::tool_error_formatter_fn(|_args| None));
            c
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        tool_outputs(&r),
        vec![json!("Tool execution was not approved.")]
    );
}

/// Python: `test_execute_approved_tools_with_unclear_status`. Through the public API an
/// undecided approval interrupts again and the tool does not run.
#[tokio::test]
async fn an_undecided_approval_interrupts_again() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![recording_tool("test_tool", "tool_result", &calls)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::None,
        RunConfig::default,
        None,
    )
    .await
    .unwrap();
    assert_eq!(r.interruptions.len(), 1);
    assert!(r.final_output.is_none());
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_execute_approved_tools_instance_method`.
#[tokio::test]
async fn an_approved_tool_runs() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![recording_tool("test_tool", "tool_result", &calls)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_outputs(&r), vec![json!("tool_result")]);
    assert_eq!(*calls.lock().unwrap(), vec!["tool_result"]);
    assert_eq!(r.final_output_text().as_deref(), Some("done"));
}

fn slow_tool(behavior: ToolTimeoutBehavior) -> FunctionTool<()> {
    FunctionTool::new("test_tool", "slow", |_ctx, _a: NoArgs| async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok("tool_result".to_string())
    })
    .with_needs_approval(true)
    .with_timeout(Duration::from_millis(10))
    .with_timeout_behavior(behavior)
}

/// Python: `test_execute_approved_tools_timeout_returns_error_as_result`.
#[tokio::test]
async fn an_approved_tool_timeout_becomes_its_output() {
    let r = decide_and_resume(
        vec![slow_tool(ToolTimeoutBehavior::ErrorAsResult)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap();
    let outputs = tool_outputs(&r);
    assert_eq!(outputs.len(), 1);
    assert!(
        outputs[0]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("timed out"),
        "{outputs:?}"
    );
}

/// Python: `test_execute_approved_tools_timeout_can_raise_exception`.
#[tokio::test]
async fn an_approved_tool_timeout_can_raise() {
    let err = decide_and_resume(
        vec![slow_tool(ToolTimeoutBehavior::RaiseException)],
        function_call_with_id("test_tool", "{}", "c1"),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::ToolTimeoutError { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("timed out"), "{err}");
}

fn namespaced(ns: &str, output: &str, calls: &Calls) -> FunctionTool<()> {
    tool_namespace(
        ns,
        &format!("{ns} tools"),
        vec![recording_tool("lookup_account", output, calls)],
    )
    .unwrap()
    .remove(0)
}

/// Python: `test_execute_approved_tools_does_not_resolve_explicit_namespaced_tool_by_bare_name`.
#[tokio::test]
async fn a_bare_call_never_resolves_to_a_namespaced_tool() {
    let calls = Calls::default();
    let err = decide_and_resume(
        vec![
            namespaced("crm", "crm", &calls),
            namespaced("billing", "billing", &calls),
        ],
        function_call_with_id("lookup_account", "{}", "call-ambiguous"),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("Tool lookup_account not found in agent A"),
        "{err}"
    );
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_execute_approved_tools_does_not_fallback_from_namespaced_approval_to_bare_tool`.
#[tokio::test]
async fn a_namespaced_call_never_falls_back_to_a_bare_tool() {
    let calls = Calls::default();
    let err = decide_and_resume(
        vec![recording_tool("lookup_account", "bare", &calls)],
        function_call_full(
            "lookup_account",
            "{}",
            "call-billing",
            None,
            Some("billing".into()),
        ),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("billing.lookup_account") && err.to_string().contains("not found"),
        "{err}"
    );
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_execute_approved_tools_prefers_visible_top_level_function_over_deferred_same_name_tool`.
#[tokio::test]
async fn a_visible_top_level_tool_beats_a_deferred_one() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![
            recording_tool("lookup_account", "visible", &calls),
            recording_tool("lookup_account", "deferred", &calls).with_defer_loading(true),
        ],
        function_call_with_id("lookup_account", "{}", "call-visible"),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_outputs(&r), vec![json!("visible")]);
    assert_eq!(*calls.lock().unwrap(), vec!["visible"]);
}

fn deferred_pair(calls: &Calls) -> Vec<FunctionTool<()>> {
    vec![
        recording_tool("lookup_account.lookup_account", "visible", calls),
        recording_tool("lookup_account", "deferred", calls).with_defer_loading(true),
    ]
}

fn deferred_call() -> OutputItem {
    function_call_full(
        "lookup_account",
        "{}",
        "call-deferred",
        None,
        Some("lookup_account".into()),
    )
}

/// Python: `test_execute_approved_tools_uses_internal_lookup_key_for_deferred_top_level_calls`.
#[tokio::test]
async fn a_deferred_top_level_call_uses_its_internal_lookup_key() {
    let calls = Calls::default();
    let r = decide_and_resume(
        deferred_pair(&calls),
        deferred_call(),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_outputs(&r), vec![json!("deferred")]);
    assert_eq!(*calls.lock().unwrap(), vec!["deferred"]);
}

/// Python: `test_deferred_collision_rejection_prefers_explicit_message`.
#[tokio::test]
async fn a_deferred_collision_rejection_keeps_the_explicit_message() {
    let calls = Calls::default();
    let r = decide_and_resume(
        deferred_pair(&calls),
        deferred_call(),
        Decision::Reject,
        formatter_config,
        Some("explicit rejection message"),
    )
    .await
    .unwrap();
    assert_eq!(tool_outputs(&r), vec![json!("explicit rejection message")]);
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_execute_approved_tools_with_rejected_deferred_tool_uses_display_name`.
#[tokio::test]
async fn a_rejected_deferred_tool_is_named_without_its_synthetic_namespace() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![recording_tool("get_weather", "sunny", &calls).with_defer_loading(true)],
        function_call_full("get_weather", "{}", "c", None, Some("get_weather".into())),
        Decision::Reject,
        formatter_config,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        tool_outputs(&r),
        vec![json!("run-level get_weather denied (c)")]
    );
}

/// Python: `test_execute_approved_tools_uses_last_duplicate_top_level_function`.
#[tokio::test]
async fn the_last_duplicate_top_level_tool_runs() {
    let calls = Calls::default();
    let r = decide_and_resume(
        vec![
            recording_tool("lookup_account", "first", &calls),
            recording_tool("lookup_account", "second", &calls),
        ],
        function_call_with_id("lookup_account", "{}", "call-shadow"),
        Decision::Approve,
        RunConfig::default,
        None,
    )
    .await
    .unwrap();
    assert_eq!(tool_outputs(&r), vec![json!("second")]);
    assert_eq!(*calls.lock().unwrap(), vec!["second"]);
}
