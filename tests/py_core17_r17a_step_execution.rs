//! Ports of the portable halves of `tests/test_run_step_execution.py` that earlier rounds
//! filed as asyncio-only: a sibling failing while another call is in its post-invoke phase or
//! hung must still surface the right failure promptly.
//!
//! Python drives `execute_tools_and_side_effects`; here the same batch runs through `Runner`.
//! A cancelled Python task is a dropped Rust future (observed with [`DropFlag`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool, MCPToolApprovalResult};
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolOutputGuardrail, ToolOutputGuardrailData,
};
use openai_agents_sdk::{Agent, AgentsError, RunResult, RunState, Runner};
use serde_json::{Value, json};
use tokio::sync::Notify;

#[derive(Default)]
struct Event {
    flag: AtomicBool,
    notify: Notify,
}

impl Event {
    fn new() -> Arc<Self> {
        Arc::new(Event::default())
    }
    fn set(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            futures::pin_mut!(notified);
            notified.as_mut().enable();
            if self.is_set() {
                return;
            }
            notified.await;
        }
    }
}

struct DropFlag(Arc<Event>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.set();
    }
}

fn model(output: Vec<openai_agents_sdk::items::OutputItem>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(output),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    )
}

fn calls() -> Vec<openai_agents_sdk::items::OutputItem> {
    vec![
        function_call_full("ok_tool", "{}", "1", None, None),
        function_call_full("error_tool", "{}", "2", None, None),
    ]
}

async fn run(agent: Agent<()>, streaming: bool, wait: Duration) -> Result<(), AgentsError> {
    let options =
        RunOptions::default().run_config(RunConfig::default().with_tool_post_invoke_wait(wait));
    let outcome = tokio::time::timeout(Duration::from_secs(10), async {
        if streaming {
            let mut streamed = Runner::run_streamed_with_options(agent, "go", options);
            let events = streamed.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event?;
            }
            Ok(())
        } else {
            Runner::run_with_options(agent, "go", options)
                .await
                .map(|_| ())
        }
    })
    .await;
    outcome.expect("the run did not settle in time")
}

fn error_tool(started: &Arc<Event>, after: Option<Arc<Event>>) -> FunctionTool<()> {
    let started = started.clone();
    FunctionTool::<()>::new_no_args("error_tool", "fails", move |_c| {
        let started = started.clone();
        let after = after.clone();
        async move {
            started.wait().await;
            if let Some(after) = after {
                // Release the sibling only after this failure has been returned.
                tokio::spawn(async move {
                    tokio::task::yield_now().await;
                    after.set();
                });
            }
            Err::<String, _>(AgentsError::other("boom"))
        }
    })
    .without_failure_error_function()
}

fn ok_tool_with_guardrail(
    started: &Arc<Event>,
    release: Option<Arc<Event>>,
    delay: Option<Duration>,
    tripwire: bool,
) -> FunctionTool<()> {
    let started = started.clone();
    FunctionTool::<()>::new_no_args("ok_tool", "ok", |_c| async { Ok("ok".to_string()) })
        .without_failure_error_function()
        .with_output_guardrail(ToolOutputGuardrail::new(
            "post_invoke",
            move |_d: ToolOutputGuardrailData<()>| {
                let started = started.clone();
                let release = release.clone();
                async move {
                    started.set();
                    if let Some(release) = release {
                        release.wait().await;
                    }
                    if let Some(delay) = delay {
                        tokio::time::sleep(delay).await;
                    }
                    Ok(if tripwire {
                        ToolGuardrailFunctionOutput::raise_exception()
                            .with_output_info(json!({"status": "late-tripwire"}))
                    } else {
                        ToolGuardrailFunctionOutput::allow().with_output_info(json!("done"))
                    })
                }
            },
        ))
}

/// `test_multiple_tool_calls_surface_sleeping_post_invoke_failure_before_sibling_error`: a
/// sibling's output guardrail that is still sleeping when another call fails wins arbitration
/// when it trips within the post-invoke wait.
#[tokio::test]
async fn sleeping_post_invoke_failure_surfaces_before_sibling_error() {
    for streaming in [false, true] {
        let started = Event::new();
        let agent = Agent::<()>::new("test")
            .with_model_instance(model(calls()))
            .with_tool(ok_tool_with_guardrail(
                &started,
                None,
                Some(Duration::from_millis(50)),
                true,
            ))
            .with_tool(error_tool(&started, None));
        let err = run(agent, streaming, Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                AgentsError::ToolOutputGuardrailTripwireTriggered { .. }
            ),
            "streaming={streaming}: {err:?}"
        );
    }
}

/// `test_multiple_tool_calls_surface_post_invoke_failure_unblocked_during_settle_turns`: the
/// guardrail is released only after the sibling's failure was returned, and still wins.
#[tokio::test]
async fn post_invoke_failure_unblocked_after_the_sibling_failed_still_surfaces() {
    for streaming in [false, true] {
        let started = Event::new();
        let release = Event::new();
        let agent = Agent::<()>::new("test")
            .with_model_instance(model(calls()))
            .with_tool(ok_tool_with_guardrail(
                &started,
                Some(release.clone()),
                None,
                true,
            ))
            .with_tool(error_tool(&started, Some(release.clone())));
        let err = run(agent, streaming, Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                AgentsError::ToolOutputGuardrailTripwireTriggered { .. }
            ),
            "streaming={streaming}: {err:?}"
        );
    }
}

/// `test_multiple_tool_calls_do_not_wait_indefinitely_for_sleeping_post_invoke_sibling`: a
/// post-invoke sibling that never finishes cannot hold back the sibling error.
#[tokio::test]
async fn a_blocked_post_invoke_sibling_does_not_hold_back_the_error() {
    for streaming in [false, true] {
        let started = Event::new();
        let release = Event::new();
        let agent = Agent::<()>::new("test")
            .with_model_instance(model(calls()))
            .with_tool(ok_tool_with_guardrail(&started, Some(release), None, false))
            .with_tool(error_tool(&started, None));
        let err = run(agent, streaming, Duration::from_millis(50))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Error running tool error_tool: boom"),
            "streaming={streaming}: {err:?}"
        );
    }
}

/// `test_multiple_tool_calls_do_not_wait_for_cancelled_sibling_tool_before_raising`,
/// `..._bound_cancelled_sibling_self_rescheduling_cleanup` and
/// `..._preserve_triggering_error_over_cancelled_sibling_cleanup_error`: a sibling that never
/// returns is cancelled (dropped) right away and cannot replace or delay the triggering error.
#[tokio::test]
async fn a_hung_sibling_is_dropped_and_the_triggering_error_surfaces() {
    for streaming in [false, true] {
        let started = Event::new();
        let dropped = Event::new();
        let (s, d) = (started.clone(), dropped.clone());
        let hung = FunctionTool::<()>::new_no_args("ok_tool", "hangs", move |_c| {
            let (s, d) = (s.clone(), d.clone());
            async move {
                let _flag = DropFlag(d);
                s.set();
                std::future::pending::<()>().await;
                Ok("unreachable".to_string())
            }
        })
        .without_failure_error_function();
        let agent = Agent::<()>::new("test")
            .with_model_instance(model(calls()))
            .with_tool(hung)
            .with_tool(error_tool(&started, None));
        let err = run(agent, streaming, Duration::from_millis(100))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Error running tool error_tool: boom"),
            "streaming={streaming}: {err:?}"
        );
        assert!(
            dropped.is_set(),
            "streaming={streaming}: sibling not cancelled"
        );
    }
}

// ---------------------------------------------------------------------------
// Hosted MCP approval identities
//
// Python drives the private `collect_manual_mcp_approvals` and `resolve_interrupted_turn` with
// hand-built pending items. Here the pending item is a real interruption whose stored identity
// is rewritten in the serialized snapshot, then resumed through `Runner`.
// ---------------------------------------------------------------------------

fn mcp_request(id: &str, server: &str, name: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "id": id, "type": "mcp_approval_request", "name": name,
        "server_label": server, "arguments": "{}",
    }))
}

fn mcp_tool(server: &str) -> HostedMCPTool<()> {
    HostedMCPTool::new(json!({
        "type": "mcp", "server_label": server,
        "server_url": format!("https://{server}.example/mcp"),
    }))
}

fn mcp_model(request: OutputItem) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![request]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    )
}

fn mcp_responses(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::MCPApprovalResponse { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

/// `test_manual_hosted_mcp_approval_uses_current_scoped_decision_after_identity_conflict` and
/// `test_resolve_interrupted_turn_uses_current_scoped_hosted_mcp_decision`: a stale pending
/// item for another server under the same request id does not hide the current request's own
/// permanent decision.
#[tokio::test]
async fn a_stale_pending_identity_does_not_hide_the_current_scoped_decision() {
    let agent = Agent::<()>::new("agent")
        .with_tool(mcp_tool("server-b"))
        .with_model_instance(mcp_model(mcp_request(
            "shared-request",
            "server-b",
            "lookup_account",
        )));
    let first = Runner::run(&agent, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], true);
    let mut json = state.to_json().unwrap();
    json["current_step"]["data"]["interruptions"][0]["raw_item"]["server_label"] =
        json!("server-a");
    let restored = RunState::from_json(json).unwrap();
    let resumed = Runner::run(&agent, restored).await.unwrap();
    assert!(
        resumed.interruptions.is_empty(),
        "{:?}",
        resumed.interruptions
    );
    let responses = mcp_responses(&resumed);
    assert_eq!(responses.len(), 1, "{responses:?}");
    assert_eq!(responses[0]["approval_request_id"], "shared-request");
    assert_eq!(responses[0]["approve"], true);
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
}

/// `test_resolve_interrupted_turn_keeps_callback_owned_hosted_mcp_request_out_of_pending`: a
/// pending item for a request that a callback now owns is answered by the callback and never
/// stays pending.
#[tokio::test]
async fn a_callback_owned_request_does_not_stay_pending() {
    let manual = Agent::<()>::new("agent")
        .with_tool(mcp_tool("server-a"))
        .with_model_instance(mcp_model(mcp_request(
            "callback-request",
            "server-a",
            "lookup_account",
        )));
    let first = Runner::run(&manual, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let state = first.to_state();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let with_callback = mcp_tool("server-a").with_on_approval_request(move |_request| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            MCPToolApprovalResult {
                approve: true,
                reason: None,
            }
        }
    });
    let agent = Agent::<()>::new("agent")
        .with_tool(with_callback)
        .with_model_instance(mcp_model(mcp_request(
            "callback-request",
            "server-a",
            "lookup_account",
        )));
    let resumed = Runner::run(&agent, state).await.unwrap();
    assert!(
        resumed.interruptions.is_empty(),
        "{:?}",
        resumed.interruptions
    );
    let responses = mcp_responses(&resumed);
    assert_eq!(responses.len(), 1, "{responses:?}");
    assert_eq!(responses[0]["approve"], true);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// `test_manual_hosted_mcp_approval_reprompts_for_partial_pending_identity`: a pending item
/// that lost its server label cannot be answered by a permanent decision, so the request asks
/// again.
#[tokio::test]
async fn a_partial_pending_identity_is_asked_again() {
    let agent = Agent::<()>::new("agent")
        .with_tool(mcp_tool("server-a"))
        .with_model_instance(mcp_model(mcp_request(
            "shared-request",
            "server-a",
            "lookup_account",
        )));
    let first = Runner::run(&agent, "go").await.unwrap();
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], true);
    let mut json = state.to_json().unwrap();
    // A permanent decision restored without an invocation binding cannot answer this request.
    json["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    json["current_step"]["data"]["interruptions"][0]["raw_item"]
        .as_object_mut()
        .unwrap()
        .remove("server_label");
    let restored = RunState::from_json(json).unwrap();
    let resumed = Runner::run(&agent, restored).await.unwrap();
    assert_eq!(resumed.interruptions.len(), 1);
    assert!(mcp_responses(&resumed).is_empty());
}

/// `test_hosted_mcp_approval_does_not_reuse_legacy_name_for_a_different_current_tool`: a
/// legacy decision for one tool name never answers a request for a different tool.
#[tokio::test]
async fn a_legacy_decision_for_another_tool_name_is_asked_again() {
    let agent = Agent::<()>::new("agent")
        .with_tool(mcp_tool("server-a"))
        .with_model_instance(mcp_model(mcp_request(
            "shared-request",
            "server-a",
            "delete_account",
        )));
    let first = Runner::run(&agent, "go").await.unwrap();
    let mut json = first.to_state().to_json().unwrap();
    json["context"]
        .as_object_mut()
        .unwrap()
        .remove("tool_invocations");
    json["context"]["approvals"] =
        json!({"lookup_account": {"approved": ["shared-request"], "rejected": []}});
    json["current_step"]["data"]["interruptions"][0]["raw_item"]["name"] = json!("lookup_account");
    let restored = RunState::from_json(json).unwrap();
    let resumed = Runner::run(&agent, restored).await.unwrap();
    assert_eq!(resumed.interruptions.len(), 1);
    assert!(mcp_responses(&resumed).is_empty());
}

/// `test_execute_tools_uses_public_agent_for_hosted_mcp_interruptions` and
/// `test_resolve_interrupted_turn_uses_public_agent_for_resumed_hosted_mcp_approvals`: the
/// interruption and the resumed approval response belong to the public agent, and the response
/// keeps the programmatic caller of its request. A Rust run has one agent per name, so the
/// public agent is the run's agent.
#[tokio::test]
async fn hosted_mcp_interruption_and_resumed_response_keep_the_public_agent() {
    let program_caller = json!({"type": "program", "caller_id": "program-resume"});
    let request = OutputItem::from_value(json!({
        "id": "mcp-approval-public-agent", "type": "mcp_approval_request",
        "name": "list_repo_languages", "server_label": "test_mcp_server", "arguments": "{}",
        "caller": program_caller,
    }));
    let program = OutputItem::from_value(json!({
        "type": "program", "id": "program-resume", "call_id": "program-resume",
        "code": "x", "fingerprint": "f",
    }));
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![program, request]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let tool = HostedMCPTool::<()>::new(json!({
        "type": "mcp", "server_label": "test_mcp_server", "server_url": "https://example.com",
        "require_approval": "always", "allowed_callers": ["direct", "programmatic"],
    }));
    let agent = Agent::<()>::new("public")
        .with_model_instance(model)
        .with_tool(tool)
        .with_tool(openai_agents_sdk::tool::ProgrammaticToolCallingTool);
    let first = Runner::run(&agent, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    assert_eq!(first.interruptions[0].agent, "public");
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let resumed = Runner::run(&agent, state).await.unwrap();
    let responses: Vec<_> = resumed
        .new_items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::MCPApprovalResponse { .. }))
        .collect();
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].agent, "public");
    assert_eq!(responses[0].to_input_item().0["caller"], program_caller);
}
