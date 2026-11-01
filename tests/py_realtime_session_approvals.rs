//! Ports of the Python `tests/realtime/test_session_approvals.py` approval, rejection and
//! pre-approval guardrail ordering tests.

#![cfg(feature = "realtime")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModel, RealtimeModelConfig, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimeModelToolCallEvent, RealtimeRunConfig,
    RealtimeSession, RealtimeSessionEvent, RealtimeToolExecutionConfig,
};
use openai_agents_sdk::tool::{FunctionTool, tool_namespace};
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use serde_json::{Value, json};
use tokio::sync::Notify;

type Event = RealtimeSessionEvent<()>;

const WAIT: Duration = Duration::from_secs(5);

fn schema() -> Value {
    json!({"type": "object", "properties": {}})
}

fn tool(name: &str, output: &str) -> FunctionTool<()> {
    let output = output.to_string();
    FunctionTool::new_raw(name, "test tool", schema(), move |_ctx, _args| {
        let output = output.clone();
        async move { Ok(output) }
    })
}

fn counting_tool(name: &str, output: &str, runs: Arc<AtomicUsize>) -> FunctionTool<()> {
    let output = output.to_string();
    FunctionTool::new_raw(name, "counting tool", schema(), move |_ctx, _args| {
        runs.fetch_add(1, Ordering::SeqCst);
        let output = output.clone();
        async move { Ok(output) }
    })
}

fn sync_config() -> RealtimeRunConfig<()> {
    RealtimeRunConfig {
        async_tool_calls: false,
        ..Default::default()
    }
}

fn pre_approval_config(async_tool_calls: bool) -> RealtimeRunConfig<()> {
    RealtimeRunConfig {
        async_tool_calls,
        tool_execution: RealtimeToolExecutionConfig {
            pre_approval_tool_input_guardrails: true,
        },
        ..Default::default()
    }
}

fn call(name: &str, call_id: &str, arguments: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent::new(name, call_id, arguments))
}

async fn start_with(
    model: Arc<dyn RealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
    run: RealtimeRunConfig<()>,
) -> RealtimeSession<()> {
    RealtimeSession::connect(model, agent, (), None, Some(run))
        .await
        .expect("session connects")
}

async fn start(
    model: &Arc<FakeRealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
    run: RealtimeRunConfig<()>,
) -> RealtimeSession<()> {
    start_with(model.clone(), agent, run).await
}

async fn recv(session: &RealtimeSession<()>) -> Event {
    tokio::time::timeout(WAIT, session.recv())
        .await
        .expect("timed out waiting for a session event")
        .expect("session stream ended")
}

async fn until(session: &RealtimeSession<()>, predicate: impl Fn(&Event) -> bool) -> Vec<Event> {
    let mut seen = Vec::new();
    loop {
        let event = recv(session).await;
        let done = predicate(&event);
        seen.push(event);
        if done {
            return seen;
        }
    }
}

fn is_marker(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::RawModelEvent { data: RealtimeModelEvent::Other { data }, .. } if data == "barrier")
}

async fn drain(model: &FakeRealtimeModel, session: &RealtimeSession<()>) -> Vec<Event> {
    assert!(model.emit(RealtimeModelEvent::Other {
        data: json!("barrier"),
    }));
    let mut events = until(session, is_marker).await;
    events.pop();
    events
}

fn types(events: &[Event]) -> Vec<&'static str> {
    events.iter().map(Event::event_type).collect()
}

fn outputs(model: &FakeRealtimeModel) -> Vec<(String, String, bool)> {
    model
        .sent_events()
        .iter()
        .filter_map(|event| {
            event
                .as_tool_output()
                .map(|(call, output, start)| (call.call_id.clone(), output.to_string(), start))
        })
        .collect()
}

fn output_strings(model: &FakeRealtimeModel) -> Vec<String> {
    outputs(model).into_iter().map(|(_, o, _)| o).collect()
}

fn is_error(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::Error { .. })
}

fn is_tool_end(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::ToolEnd { .. })
}

fn error_message(event: &Event) -> String {
    match event {
        RealtimeSessionEvent::Error { error, .. } => error["message"].as_str().unwrap().to_string(),
        other => panic!("expected an error event, got {}", other.event_type()),
    }
}

fn approvals(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. }))
        .count()
}

#[tokio::test]
async fn an_approval_resumes_with_the_tool_that_was_pending() {
    let approved = tool("approval_tool", "approved implementation").with_needs_approval(true);
    let initial = RealtimeAgent::<()>::new("initial")
        .with_tool(approved)
        .into_arc();
    let replacement = RealtimeAgent::<()>::new("replacement")
        .with_tool(tool("approval_tool", "replacement implementation"))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, initial, sync_config()).await;

    model.emit(call("approval_tool", "call_pending_snapshot", "{}"));
    let events = drain(&model, &session).await;
    assert_eq!(approvals(&events), 1);

    session.update_agent(replacement).await.unwrap();
    session
        .approve_tool_call("call_pending_snapshot", false)
        .await
        .unwrap();
    assert_eq!(output_strings(&model), ["approved implementation"]);
}

async fn sticky_decisions_follow_the_agent(approve: bool) {
    let operate = |output: &str| tool("operate", output).with_needs_approval(true);
    let first = RealtimeAgent::<()>::new("first-agent")
        .with_tool(operate("first"))
        .into_arc();
    let second = RealtimeAgent::<()>::new("second-agent")
        .with_tool(operate("second"))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, first.clone(), sync_config()).await;

    model.emit(call("operate", "first", "{}"));
    drain(&model, &session).await;
    if approve {
        session.approve_tool_call("first", true).await.unwrap();
    } else {
        session
            .reject_tool_call("first", true, Some("deny first".into()))
            .await
            .unwrap();
    }
    session.update_agent(second).await.unwrap();
    model.emit(call("operate", "second", "{}"));
    let events = drain(&model, &session).await;
    assert_eq!(approvals(&events), 1, "the new agent asks again");
    session
        .reject_tool_call("second", true, Some("deny second".into()))
        .await
        .unwrap();
    session.update_agent(first).await.unwrap();
    model.emit(call("operate", "first-again", "{}"));
    let events = drain(&model, &session).await;
    assert_eq!(
        approvals(&events),
        0,
        "the first agent's decision still applies"
    );

    let first_output = if approve { "first" } else { "deny first" };
    assert_eq!(
        output_strings(&model),
        [first_output, "deny second", first_output]
    );
}

#[tokio::test]
async fn sticky_rejections_remain_with_the_agent_after_update() {
    sticky_decisions_follow_the_agent(false).await;
}

#[tokio::test]
async fn sticky_approvals_remain_with_the_agent_after_update() {
    sticky_decisions_follow_the_agent(true).await;
}

#[tokio::test]
async fn callable_approval_fails_closed_for_invalid_arguments() {
    for arguments in [
        "",
        r#"{"subject": "refund""#,
        "null",
        "[]",
        r#"{"amount": NaN}"#,
        r#"{"amount": Infinity}"#,
        r#"{"amount": -Infinity}"#,
    ] {
        let approval_inputs = Arc::new(Mutex::new(Vec::<Value>::new()));
        let runs = Arc::new(AtomicUsize::new(0));
        let send_email =
            counting_tool("send_email", "sent", runs.clone()).with_needs_approval_fn({
                let approval_inputs = approval_inputs.clone();
                move |_ctx, params, _call_id| {
                    approval_inputs.lock().unwrap().push(params);
                    async { false }
                }
            });
        let agent = RealtimeAgent::<()>::new("agent")
            .with_tool(send_email)
            .into_arc();
        let model = Arc::new(FakeRealtimeModel::new());
        let session = start(&model, agent, sync_config()).await;

        model.emit(call("send_email", "call-invalid", arguments));
        let events = drain(&model, &session).await;
        assert_eq!(approvals(&events), 1, "arguments: {arguments:?}");
        assert!(
            approval_inputs.lock().unwrap().is_empty(),
            "arguments: {arguments:?}"
        );
        assert_eq!(runs.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn callable_approval_receives_valid_object_arguments() {
    let approval_inputs = Arc::new(Mutex::new(Vec::<Value>::new()));
    let tool_inputs = Arc::new(Mutex::new(Vec::<String>::new()));
    let send_email = FunctionTool::new_raw(
        "send_email",
        "Send an email.",
        json!({"type": "object", "properties": {"subject": {"type": "string"}}}),
        {
            let tool_inputs = tool_inputs.clone();
            move |_ctx, raw| {
                tool_inputs.lock().unwrap().push(raw);
                async { Ok("sent".to_string()) }
            }
        },
    )
    .with_needs_approval_fn({
        let approval_inputs = approval_inputs.clone();
        move |_ctx, params, _call_id| {
            approval_inputs.lock().unwrap().push(params);
            async { false }
        }
    });
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(send_email)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config()).await;

    let arguments = r#"{"subject": "status update"}"#;
    model.emit(call("send_email", "call-valid", arguments));
    let events = drain(&model, &session).await;
    assert_eq!(approvals(&events), 0);
    assert_eq!(
        *approval_inputs.lock().unwrap(),
        [json!({"subject": "status update"})]
    );
    assert_eq!(*tool_inputs.lock().unwrap(), [arguments]);
}

fn rejecting_guardrail(message: &'static str) -> ToolInputGuardrail<()> {
    ToolInputGuardrail::<()>::new("reject", move |_data| async move {
        Ok(ToolGuardrailFunctionOutput::reject_content(message))
    })
}

fn counting_guardrail(runs: Arc<AtomicUsize>) -> ToolInputGuardrail<()> {
    ToolInputGuardrail::<()>::new("count", move |_data| {
        runs.fetch_add(1, Ordering::SeqCst);
        async { Ok(ToolGuardrailFunctionOutput::allow()) }
    })
}

#[tokio::test]
async fn a_tool_input_guardrail_rejects_before_execution() {
    let runs = Arc::new(AtomicUsize::new(0));
    let guarded = counting_tool("test_function", "ok", runs.clone())
        .with_input_guardrail(rejecting_guardrail("blocked before execution"));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config()).await;
    model.emit(call("test_function", "call_guardrail_reject", "{}"));
    drain(&model, &session).await;

    assert_eq!(runs.load(Ordering::SeqCst), 0);
    let sent = outputs(&model);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].1, "blocked before execution");
    assert!(sent[0].2);
}

#[tokio::test]
async fn a_pending_approval_skips_tool_input_guardrails_by_default() {
    let guardrail_runs = Arc::new(AtomicUsize::new(0));
    let guarded = tool("test_function", "ok")
        .with_needs_approval(true)
        .with_input_guardrail(counting_guardrail(guardrail_runs.clone()));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config()).await;
    model.emit(call("test_function", "call_guardrail_pending", "{}"));
    let events = drain(&model, &session).await;
    assert_eq!(approvals(&events), 1);
    assert_eq!(guardrail_runs.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_pre_approval_guardrail_rejects_a_pending_approval() {
    let runs = Arc::new(AtomicUsize::new(0));
    let guarded = counting_tool("test_function", "ok", runs.clone())
        .with_needs_approval(true)
        .with_input_guardrail(rejecting_guardrail("blocked before approval"));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, pre_approval_config(false)).await;
    model.emit(call("test_function", "call_pre_approval_reject", "{}"));
    let events = drain(&model, &session).await;

    assert_eq!(approvals(&events), 0);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    let sent = outputs(&model);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].1, "blocked before approval");
    assert!(sent[0].2);
}

#[tokio::test]
async fn pre_approval_guardrails_run_again_after_approval() {
    let guardrail_runs = Arc::new(AtomicUsize::new(0));
    let runs = Arc::new(AtomicUsize::new(0));
    let guarded = counting_tool("test_function", "ok", runs.clone())
        .with_needs_approval(true)
        .with_input_guardrail(counting_guardrail(guardrail_runs.clone()));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, pre_approval_config(false)).await;

    model.emit(call("test_function", "call_pre_approval_rerun", "{}"));
    drain(&model, &session).await;
    assert_eq!(guardrail_runs.load(Ordering::SeqCst), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 0);

    session
        .approve_tool_call("call_pre_approval_rerun", false)
        .await
        .unwrap();
    assert_eq!(guardrail_runs.load(Ordering::SeqCst), 2);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let sent = outputs(&model);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].1, "ok");
    assert!(sent[0].2);
}

#[tokio::test]
async fn a_duplicate_pending_approval_is_ignored_and_the_approval_runs_once() {
    let runs = Arc::new(AtomicUsize::new(0));
    let guarded = counting_tool("test_function", "ok", runs.clone()).with_needs_approval(true);
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config()).await;

    model.emit(call("test_function", "call_duplicate_approval", "{}"));
    model.emit(call("test_function", "call_duplicate_approval", "{}"));
    model.emit(call(
        "test_function",
        "call_duplicate_approval",
        r#"{"changed":true}"#,
    ));
    let events = drain(&model, &session).await;
    assert_eq!(approvals(&events), 1);
    let errors: Vec<_> = events.iter().filter(|e| is_error(e)).collect();
    assert_eq!(errors.len(), 1);
    assert!(error_message(errors[0]).contains("unique call ID"));

    session
        .approve_tool_call("call_duplicate_approval", false)
        .await
        .unwrap();
    model.emit(call("test_function", "call_duplicate_approval", "{}"));
    model.emit(call(
        "test_function",
        "call_duplicate_approval",
        r#"{"changed":true}"#,
    ));
    let events = drain(&model, &session).await;
    assert!(types(&events).contains(&"tool_end"));
    assert_eq!(events.iter().filter(|e| is_error(e)).count(), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(outputs(&model).len(), 1);
}

#[tokio::test]
async fn an_approved_call_reserves_its_id_while_the_tool_runs() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let approved_runs = Arc::new(AtomicUsize::new(0));
    let approved = {
        let (started, release, approved_runs) =
            (started.clone(), release.clone(), approved_runs.clone());
        FunctionTool::new_raw("test_function", "approved", schema(), move |_ctx, _args| {
            let (started, release, approved_runs) =
                (started.clone(), release.clone(), approved_runs.clone());
            async move {
                approved_runs.fetch_add(1, Ordering::SeqCst);
                started.notify_one();
                release.notified().await;
                Ok("approved_result".to_string())
            }
        })
        .with_needs_approval(true)
    };
    let agent = RealtimeAgent::<()>::new("approved_agent")
        .with_tool(approved)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, Default::default()).await;

    model.emit(call("test_function", "call_async_approval_race", "{}"));
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    let approver = tokio::spawn({
        let session = session.clone();
        async move {
            session
                .approve_tool_call("call_async_approval_race", false)
                .await
        }
    });
    tokio::time::timeout(WAIT, started.notified())
        .await
        .unwrap();

    // A redelivered call while the approved tool runs must not start a second run.
    model.emit(call("test_function", "call_async_approval_race", "{}"));
    drain(&model, &session).await;
    release.notify_one();
    approver.await.unwrap().unwrap();
    assert_eq!(approved_runs.load(Ordering::SeqCst), 1);
    assert_eq!(output_strings(&model), ["approved_result"]);
}

fn namespaced(name: &str, runs: Arc<AtomicUsize>) -> FunctionTool<()> {
    tool_namespace(
        "crm",
        "CRM tools",
        vec![counting_tool(name, "account", runs).with_needs_approval(true)],
    )
    .unwrap()
    .remove(0)
}

#[tokio::test]
async fn always_approving_a_namespaced_tool_does_not_approve_a_bare_tool() {
    let runs = Arc::new(AtomicUsize::new(0));
    let namespaced_agent = RealtimeAgent::<()>::new("crm_agent")
        .with_tool(namespaced("lookup_account", runs.clone()))
        .into_arc();
    let bare_agent = RealtimeAgent::<()>::new("bare_agent")
        .with_tool(
            counting_tool("lookup_account", "account", runs.clone()).with_needs_approval(true),
        )
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, namespaced_agent, sync_config()).await;

    model.emit(call("lookup_account", "call_first", "{}"));
    drain(&model, &session).await;
    session.approve_tool_call("call_first", true).await.unwrap();
    session.update_agent(bare_agent).await.unwrap();
    model.emit(call("lookup_account", "call_second", "{}"));
    let events = drain(&model, &session).await;

    // The bare tool still asks: the namespaced tool's standing approval does not carry over.
    assert_eq!(approvals(&events), 1);
    assert_eq!(
        session
            .context()
            .is_tool_approved("bare_agent", "lookup_account", "call_second"),
        None
    );
    assert_eq!(outputs(&model).len(), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_pending_rejection_does_not_send_twice_when_the_call_is_redelivered() {
    /// Holds the first tool-output send until released.
    #[derive(Debug)]
    struct BlockingModel {
        inner: FakeRealtimeModel,
        started: Notify,
        release: Notify,
        block_next: AtomicBool,
    }

    #[async_trait::async_trait]
    impl RealtimeModel for BlockingModel {
        async fn connect(
            &self,
            config: RealtimeModelConfig,
        ) -> openai_agents_sdk::error::Result<RealtimeModelEvents> {
            self.inner.connect(config).await
        }
        async fn send_event(
            &self,
            event: RealtimeModelSendEvent,
        ) -> openai_agents_sdk::error::Result<()> {
            if event.as_tool_output().is_some() && self.block_next.swap(false, Ordering::SeqCst) {
                self.started.notify_one();
                self.release.notified().await;
            }
            self.inner.send_event(event).await
        }
        async fn close(&self) -> openai_agents_sdk::error::Result<()> {
            self.inner.close().await
        }
    }

    let model = Arc::new(BlockingModel {
        inner: FakeRealtimeModel::new(),
        started: Notify::new(),
        release: Notify::new(),
        block_next: AtomicBool::new(true),
    });
    let runs = Arc::new(AtomicUsize::new(0));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(counting_tool("test_function", "ok", runs.clone()).with_needs_approval(true))
        .into_arc();
    let session = start_with(model.clone(), agent, Default::default()).await;

    model
        .inner
        .emit(call("test_function", "call_reject_race", "{}"));
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    let rejecter = tokio::spawn({
        let session = session.clone();
        async move {
            session
                .reject_tool_call("call_reject_race", false, None)
                .await
        }
    });
    tokio::time::timeout(WAIT, model.started.notified())
        .await
        .unwrap();

    model
        .inner
        .emit(call("test_function", "call_reject_race", "{}"));
    drain(&model.inner, &session).await;
    model.release.notify_one();
    rejecter.await.unwrap().unwrap();

    assert_eq!(outputs(&model.inner).len(), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

fn denial_formatter() -> openai_agents_sdk::realtime::RealtimeToolErrorFormatter<()> {
    openai_agents_sdk::tool_error_formatter_fn(|args| {
        Some(format!(
            "run-level {} denied ({})",
            args.tool_name, args.call_id
        ))
    })
}

#[tokio::test]
async fn a_rejection_uses_the_run_level_formatter() {
    let guarded = tool("test_function", "ok").with_needs_approval(true);
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(guarded)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let run = RealtimeRunConfig {
        tool_error_formatter: Some(denial_formatter()),
        ..Default::default()
    };
    let session = start(&model, agent, run).await;

    model.emit(call("test_function", "call_reject_custom", "{}"));
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    session
        .reject_tool_call("call_reject_custom", false, None)
        .await
        .unwrap();

    let sent = outputs(&model);
    assert_eq!(
        sent[0].1,
        "run-level test_function denied (call_reject_custom)"
    );
    assert!(sent[0].2);
    let events = until(&session, is_tool_end).await;
    match events.last().unwrap() {
        RealtimeSessionEvent::ToolEnd { output, .. } => assert_eq!(
            output.to_model_string(),
            "run-level test_function denied (call_reject_custom)"
        ),
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn always_rejecting_a_namespaced_tool_reuses_the_explicit_message() {
    let runs = Arc::new(AtomicUsize::new(0));
    let agent = RealtimeAgent::<()>::new("crm_agent")
        .with_tool(namespaced("lookup_account", runs.clone()))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, Default::default()).await;

    model.emit(call("lookup_account", "call_reject_first", "{}"));
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    session
        .reject_tool_call(
            "call_reject_first",
            true,
            Some("explicit crm rejection".into()),
        )
        .await
        .unwrap();
    model.emit(call("lookup_account", "call_reject_second", "{}"));
    let events = drain(&model, &session).await;

    assert_eq!(approvals(&events), 0);
    assert_eq!(
        output_strings(&model),
        ["explicit crm rejection", "explicit crm rejection"]
    );
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_sticky_rejection_does_not_bind_a_changed_payload_for_the_same_call_id() {
    let runs = Arc::new(AtomicUsize::new(0));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(counting_tool("test_function", "ok", runs.clone()).with_needs_approval(true))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config()).await;

    model.emit(call("test_function", "call-sticky-reject", "{}"));
    drain(&model, &session).await;
    session
        .reject_tool_call("call-sticky-reject", true, None)
        .await
        .unwrap();
    model.emit(call(
        "test_function",
        "call-sticky-reject",
        r#"{"changed":true}"#,
    ));
    let events = drain(&model, &session).await;
    let errors: Vec<_> = events.iter().filter(|e| is_error(e)).collect();
    assert_eq!(errors.len(), 1);
    assert!(error_message(errors[0]).contains("unique call ID"));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(outputs(&model).len(), 1);
}

#[tokio::test]
async fn a_sticky_rejection_skips_the_dynamic_approval_checker() {
    let checker_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let runs = Arc::new(AtomicUsize::new(0));
    let send_email = counting_tool("send_email", "should-not-run", runs.clone())
        .with_needs_approval_fn({
            let checker_calls = checker_calls.clone();
            move |_ctx, _params, call_id| {
                checker_calls.lock().unwrap().push(call_id);
                async { true }
            }
        });
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(send_email)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config()).await;

    model.emit(call("send_email", "call-reject-first", "{}"));
    drain(&model, &session).await;
    session
        .reject_tool_call("call-reject-first", true, None)
        .await
        .unwrap();
    model.emit(call("send_email", "call-reject-second", "{}"));
    let events = drain(&model, &session).await;

    assert_eq!(*checker_calls.lock().unwrap(), ["call-reject-first"]);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(approvals(&events), 0);
    assert_eq!(outputs(&model).len(), 2);
}

#[tokio::test]
async fn a_sticky_rejection_wins_while_the_dynamic_approval_checker_is_pending() {
    let checker_started = Arc::new(Notify::new());
    let checker_release = Arc::new(Notify::new());
    let checker_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let runs = Arc::new(AtomicUsize::new(0));
    let send_email = counting_tool("send_email", "should-not-run", runs.clone())
        .with_needs_approval_fn({
            let (checker_started, checker_release, checker_calls) = (
                checker_started.clone(),
                checker_release.clone(),
                checker_calls.clone(),
            );
            move |_ctx, _params, call_id| {
                checker_calls.lock().unwrap().push(call_id.clone());
                let (checker_started, checker_release) =
                    (checker_started.clone(), checker_release.clone());
                async move {
                    if call_id == "call-pending-checker" {
                        checker_started.notify_one();
                        checker_release.notified().await;
                        return false;
                    }
                    true
                }
            }
        });
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(send_email)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, Default::default()).await;

    model.emit(call("send_email", "call-reject-first", "{}"));
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    model.emit(call("send_email", "call-pending-checker", "{}"));
    tokio::time::timeout(WAIT, checker_started.notified())
        .await
        .unwrap();
    session
        .reject_tool_call("call-reject-first", true, Some("sticky rejection".into()))
        .await
        .unwrap();
    checker_release.notify_one();
    // Both calls finish with the sticky rejection: the first at once, the pending one once its
    // checker returns.
    until(&session, is_tool_end).await;
    until(&session, is_tool_end).await;

    assert_eq!(
        *checker_calls.lock().unwrap(),
        ["call-reject-first", "call-pending-checker"]
    );
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(
        output_strings(&model),
        ["sticky rejection", "sticky rejection"]
    );
}

async fn sticky_decision_while_a_rejecting_pre_approval_guardrail_is_pending(approved: bool) {
    let guardrail_started = Arc::new(Notify::new());
    let guardrail_release = Arc::new(Notify::new());
    let guardrail_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let runs = Arc::new(AtomicUsize::new(0));
    let guardrail = ToolInputGuardrail::<()>::new("blocking", {
        let (guardrail_started, guardrail_release, guardrail_calls) = (
            guardrail_started.clone(),
            guardrail_release.clone(),
            guardrail_calls.clone(),
        );
        move |data| {
            let call_id = data.context.tool_call_id.clone();
            guardrail_calls.lock().unwrap().push(call_id.clone());
            let (guardrail_started, guardrail_release) =
                (guardrail_started.clone(), guardrail_release.clone());
            async move {
                if call_id == "call-pending-guardrail" {
                    guardrail_started.notify_one();
                    guardrail_release.notified().await;
                    return Ok(ToolGuardrailFunctionOutput::reject_content(
                        "guardrail rejection",
                    ));
                }
                Ok(ToolGuardrailFunctionOutput::allow())
            }
        }
    });
    let send_email = counting_tool("send_email", "tool output", runs.clone())
        .with_needs_approval(true)
        .with_input_guardrail(guardrail);
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(send_email)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, pre_approval_config(true)).await;

    model.emit(call("send_email", "call-reject-first", "{}"));
    until(&session, |e| {
        matches!(e, RealtimeSessionEvent::ToolApprovalRequired { .. })
    })
    .await;
    model.emit(call("send_email", "call-pending-guardrail", "{}"));
    tokio::time::timeout(WAIT, guardrail_started.notified())
        .await
        .unwrap();
    if approved {
        session
            .approve_tool_call("call-reject-first", true)
            .await
            .unwrap();
        // The approved first call ran its tool; that output is not part of this comparison.
        model.clear();
    } else {
        session
            .reject_tool_call("call-reject-first", true, Some("sticky rejection".into()))
            .await
            .unwrap();
        model.clear();
    }
    // The guardrail blocks on every run of this call, so leave a permit for the second run.
    guardrail_release.notify_one();
    guardrail_release.notify_one();
    until(&session, is_tool_end).await;
    // Wait for the pending call's own output.
    let mut waited = 0;
    while !outputs(&model)
        .iter()
        .any(|(id, _, _)| id == "call-pending-guardrail")
        && waited < 200
    {
        tokio::time::sleep(Duration::from_millis(10)).await;
        waited += 1;
    }

    let pending_outputs: Vec<_> = outputs(&model)
        .into_iter()
        .filter(|(id, _, _)| id == "call-pending-guardrail")
        .map(|(_, output, _)| output)
        .collect();
    let calls: Vec<String> = guardrail_calls
        .lock()
        .unwrap()
        .iter()
        .filter(|id| *id == "call-pending-guardrail")
        .cloned()
        .collect();
    if approved {
        // The standing approval sends the call back through the guardrails before it may run.
        assert_eq!(calls, ["call-pending-guardrail", "call-pending-guardrail"]);
        assert_eq!(pending_outputs, ["guardrail rejection"]);
    } else {
        assert_eq!(calls, ["call-pending-guardrail"]);
        assert_eq!(pending_outputs, ["sticky rejection"]);
    }
    // Only the first call (approved) ever reaches the tool body, never the pending one.
    assert_eq!(runs.load(Ordering::SeqCst), usize::from(approved));
}

#[tokio::test]
async fn an_approval_decision_wins_while_a_rejecting_guardrail_is_pending() {
    sticky_decision_while_a_rejecting_pre_approval_guardrail_is_pending(true).await;
}

#[tokio::test]
async fn a_rejection_decision_wins_while_a_rejecting_guardrail_is_pending() {
    sticky_decision_while_a_rejecting_pre_approval_guardrail_is_pending(false).await;
}

// Keeps the error type imported for the helper-free ports above.
#[allow(dead_code)]
fn _error_type(_: AgentsError) {}
