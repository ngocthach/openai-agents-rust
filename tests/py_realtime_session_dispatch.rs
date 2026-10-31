//! Ports of the Python `tests/realtime/test_session.py` tool dispatch tests (call-id identity,
//! timeouts, handoffs and snapshots).

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModelConfig, RealtimeModelEvent,
    RealtimeModelToolCallEvent, RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
    RealtimeSessionModelSettings, RealtimeVoice, realtime_handoff,
};
use openai_agents_sdk::tool::{FunctionTool, ToolTimeoutBehavior};
use serde_json::{Value, json};
use tokio::sync::Notify;

type Event = RealtimeSessionEvent<()>;

const WAIT: Duration = Duration::from_secs(5);

fn raw_tool(name: &str, output: &str) -> FunctionTool<()> {
    let output = output.to_string();
    FunctionTool::new_raw(
        name,
        "test tool",
        json!({"type": "object", "properties": {}}),
        move |_ctx, _args| {
            let output = output.clone();
            async move { Ok(output) }
        },
    )
}

/// A tool that signals `started`, then waits for `release` before answering `output`.
fn gated_tool(
    name: &str,
    output: &str,
    started: Arc<Notify>,
    release: Arc<Notify>,
    runs: Arc<AtomicUsize>,
) -> FunctionTool<()> {
    let output = output.to_string();
    FunctionTool::new_raw(
        name,
        "gated tool",
        json!({"type": "object", "properties": {}}),
        move |_ctx, _args| {
            let (output, started, release, runs) = (
                output.clone(),
                started.clone(),
                release.clone(),
                runs.clone(),
            );
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                started.notify_one();
                release.notified().await;
                Ok(output)
            }
        },
    )
}

fn sync_config() -> RealtimeRunConfig<()> {
    RealtimeRunConfig {
        async_tool_calls: false,
        ..Default::default()
    }
}

fn ambiguous_agent(name: &str) -> Arc<RealtimeAgent<()>> {
    RealtimeAgent::<()>::new(name)
        .with_tool(raw_tool("transfer_to_billing", "ok"))
        .with_handoff(realtime_handoff(
            RealtimeAgent::<()>::new("billing").into_arc(),
        ))
        .into_arc()
}

fn call(name: &str, call_id: &str, arguments: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::FunctionCall(RealtimeModelToolCallEvent::new(name, call_id, arguments))
}

async fn start(
    model: &Arc<FakeRealtimeModel>,
    agent: Arc<RealtimeAgent<()>>,
    run: RealtimeRunConfig<()>,
    model_config: Option<RealtimeModelConfig>,
) -> RealtimeSession<()> {
    RealtimeSession::connect(model.clone(), agent, (), model_config, Some(run))
        .await
        .expect("session connects")
}

async fn recv(session: &RealtimeSession<()>) -> Event {
    tokio::time::timeout(WAIT, session.recv())
        .await
        .expect("timed out waiting for a session event")
        .expect("session stream ended")
}

/// Collect events up to and including the first one matching `predicate`.
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

/// Everything the session emits for the events fed so far. The marker is raw-only, so once it
/// comes back every earlier event (handled in order by the session) has been delivered.
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

fn tool_outputs(model: &FakeRealtimeModel) -> Vec<(String, String, bool)> {
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

fn outputs_only(model: &FakeRealtimeModel) -> Vec<String> {
    tool_outputs(model).into_iter().map(|(_, o, _)| o).collect()
}

fn is_tool_end(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::ToolEnd { .. })
}

fn is_error(event: &Event) -> bool {
    matches!(event, RealtimeSessionEvent::Error { .. })
}

fn error_message(event: &Event) -> String {
    match event {
        RealtimeSessionEvent::Error { error, .. } => error["message"].as_str().unwrap().to_string(),
        other => panic!("expected an error event, got {}", other.event_type()),
    }
}

// ---------------------------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn tool_calls_run_without_blocking_event_delivery_by_default() {
    let model = Arc::new(FakeRealtimeModel::new());
    let (started, release, runs) = (
        Arc::new(Notify::new()),
        Arc::new(Notify::new()),
        Arc::new(AtomicUsize::new(0)),
    );
    let tool = gated_tool(
        "slow",
        "slow output",
        started.clone(),
        release.clone(),
        runs,
    );
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, Default::default(), None).await;

    model.emit(call("slow", "call_async", "{}"));
    tokio::time::timeout(WAIT, started.notified())
        .await
        .unwrap();
    // The tool is parked, yet later transport events still reach the caller.
    let events = drain(&model, &session).await;
    assert!(!types(&events).contains(&"tool_end"));
    release.notify_one();
    until(&session, is_tool_end).await;
    assert_eq!(outputs_only(&model), ["slow output"]);
}

#[tokio::test]
async fn an_async_tool_call_uses_the_dispatch_snapshot_from_event_time() {
    let model = Arc::new(FakeRealtimeModel::new());
    let (started, release, runs) = (
        Arc::new(Notify::new()),
        Arc::new(Notify::new()),
        Arc::new(AtomicUsize::new(0)),
    );
    let initial_tool = gated_tool(
        "snapshot_tool",
        "initial implementation",
        started.clone(),
        release.clone(),
        runs,
    );
    let initial = RealtimeAgent::<()>::new("initial")
        .with_tool(initial_tool)
        .into_arc();
    let replacement = RealtimeAgent::<()>::new("replacement")
        .with_tool(raw_tool("snapshot_tool", "replacement implementation"))
        .into_arc();
    let session = start(&model, initial, Default::default(), None).await;

    model.emit(call("snapshot_tool", "call_async_snapshot", "{}"));
    tokio::time::timeout(WAIT, started.notified())
        .await
        .unwrap();
    session.update_agent(replacement).await.unwrap();
    release.notify_one();
    until(&session, is_tool_end).await;
    assert_eq!(outputs_only(&model), ["initial implementation"]);
}

#[tokio::test]
async fn a_duplicate_function_tool_call_id_is_ignored() {
    let model = Arc::new(FakeRealtimeModel::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let tool = {
        let runs = runs.clone();
        FunctionTool::new_raw(
            "test_function",
            "counts",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                runs.fetch_add(1, Ordering::SeqCst);
                async move { Ok("function_result".to_string()) }
            },
        )
    };
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, Default::default(), None).await;
    model.emit(call("test_function", "call_duplicate", "{}"));
    model.emit(call("test_function", "call_duplicate", "{}"));
    drain(&model, &session).await;
    // Wait for the (single) run to finish by draining once more after the tool task settles.
    let events = until(&session, is_tool_end).await;
    assert!(types(&events).contains(&"tool_end"));
    drain(&model, &session).await;
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(tool_outputs(&model).len(), 1);
}

fn slow_tool(behavior: ToolTimeoutBehavior) -> FunctionTool<()> {
    let mut tool = FunctionTool::new_raw(
        "slow_tool",
        "slow",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok("done".to_string())
        },
    )
    .with_timeout(Duration::from_millis(10))
    .with_timeout_behavior(behavior);
    tool.timeout_error_function = None;
    tool
}

#[tokio::test]
async fn a_tool_timeout_returns_a_result_message() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(slow_tool(ToolTimeoutBehavior::ErrorAsResult))
        .into_arc();
    let session = start(&model, agent, sync_config(), None).await;
    model.emit(call("slow_tool", "call_timeout", "{}"));
    drain(&model, &session).await;
    let sent = tool_outputs(&model);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].0, "call_timeout");
    assert!(sent[0].1.to_lowercase().contains("timed out"));
    assert!(sent[0].2, "the output starts a response");
}

#[tokio::test]
async fn a_tool_timeout_can_raise_and_ends_the_session() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(slow_tool(ToolTimeoutBehavior::RaiseException))
        .into_arc();
    // Both the awaited and the background dispatch paths fail the same way.
    for run in [sync_config(), Default::default()] {
        model.clear();
        let session = RealtimeSession::connect(model.clone(), agent.clone(), (), None, Some(run))
            .await
            .unwrap();
        model.emit(call("slow_tool", "call_timeout_raise", "{}"));
        let events = until(&session, is_error).await;
        assert!(types(&events).contains(&"tool_start"));
        assert_eq!(
            error_message(events.last().unwrap()),
            "Tool call task failed"
        );
        assert!(tool_outputs(&model).is_empty());
        while session.recv().await.is_some() {}
        match session.take_error() {
            Some(AgentsError::ToolTimeoutError { tool_name, .. }) => {
                assert_eq!(tool_name, "slow_tool");
            }
            other => panic!("expected a tool timeout error, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_timeout_error_function_formats_the_result() {
    let model = Arc::new(FakeRealtimeModel::new());
    let tool =
        slow_tool(ToolTimeoutBehavior::ErrorAsResult).with_timeout_error_function(|_ctx, error| {
            match error {
                AgentsError::ToolTimeoutError {
                    tool_name,
                    timeout_seconds,
                    ..
                } => format!("async-timeout:{tool_name}:{timeout_seconds}"),
                other => panic!("unexpected {other:?}"),
            }
        });
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, sync_config(), None).await;
    model.emit(call("slow_tool", "call_timeout_custom", "{}"));
    let events = drain(&model, &session).await;
    assert_eq!(outputs_only(&model), ["async-timeout:slow_tool:0.01"]);
    let end = events.iter().find(|e| is_tool_end(e)).expect("tool_end");
    match end {
        RealtimeSessionEvent::ToolEnd { output, .. } => {
            assert_eq!(output.to_model_string(), "async-timeout:slow_tool:0.01");
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn the_right_tool_runs_among_several() {
    let model = Arc::new(FakeRealtimeModel::new());
    let first_runs = Arc::new(AtomicUsize::new(0));
    let tool_one = {
        let first_runs = first_runs.clone();
        FunctionTool::new_raw(
            "tool_one",
            "one",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                first_runs.fetch_add(1, Ordering::SeqCst);
                async move { Ok("result_one".to_string()) }
            },
        )
    };
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(tool_one)
        .with_tool(raw_tool("tool_two", "result_two"))
        .with_handoff(realtime_handoff(
            RealtimeAgent::<()>::new("handoff_tool").into_arc(),
        ))
        .into_arc();
    let session = start(&model, agent, sync_config(), None).await;
    model.emit(call("tool_two", "call_456", r#"{"test": "data"}"#));
    drain(&model, &session).await;
    assert_eq!(first_runs.load(Ordering::SeqCst), 0);
    assert_eq!(outputs_only(&model), ["result_two"]);
}

#[tokio::test]
async fn a_handoff_validation_failure_keeps_the_current_agent() {
    let first = RealtimeAgent::<()>::new("first_agent")
        .with_instructions("first_agent_instructions")
        .with_handoff(realtime_handoff(ambiguous_agent("invalid_agent")))
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, first, sync_config(), None).await;

    model.emit(call("transfer_to_invalid_agent", "call_invalid", "{}"));
    let events = until(&session, is_error).await;
    assert_eq!(
        error_message(events.last().unwrap()),
        "Tool call task failed"
    );
    while session.recv().await.is_some() {}
    assert!(
        session
            .take_error()
            .unwrap()
            .to_string()
            .contains("Duplicate")
    );
    assert_eq!(session.current_agent_name(), "first_agent");
    assert!(model.sent_events().is_empty());
    assert!(tool_outputs(&model).is_empty());
}

#[tokio::test]
async fn a_handoff_session_update_preserves_a_custom_voice() {
    let second = RealtimeAgent::<()>::new("second_agent")
        .with_instructions("second_agent_instructions")
        .into_arc();
    let first = RealtimeAgent::<()>::new("first_agent")
        .with_instructions("first_agent_instructions")
        .with_handoff(second)
        .into_arc();
    let config = RealtimeModelConfig {
        initial_model_settings: Some(RealtimeSessionModelSettings {
            voice: Some(RealtimeVoice::Custom {
                id: "voice_test".into(),
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, first, sync_config(), Some(config)).await;
    model.emit(call("transfer_to_second_agent", "call_789", "{}"));
    drain(&model, &session).await;

    let sent = model.sent_events();
    let update = sent[0].as_session_update().expect("session update first");
    assert_eq!(
        update.voice,
        Some(RealtimeVoice::Custom {
            id: "voice_test".into()
        })
    );
    let (_, _, start_response) = sent[1].as_tool_output().expect("then the output");
    assert!(start_response);
}

#[tokio::test]
async fn a_changed_call_id_fails_while_the_first_call_is_still_running() {
    let model = Arc::new(FakeRealtimeModel::new());
    let (started, release, runs) = (
        Arc::new(Notify::new()),
        Arc::new(Notify::new()),
        Arc::new(AtomicUsize::new(0)),
    );
    let tool = gated_tool(
        "test_function",
        "ok",
        started.clone(),
        release.clone(),
        runs.clone(),
    );
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, Default::default(), None).await;

    model.emit(call(
        "test_function",
        "call_dispatch_pending",
        r#"{"value":"safe"}"#,
    ));
    tokio::time::timeout(WAIT, started.notified())
        .await
        .unwrap();
    model.emit(call(
        "test_function",
        "call_dispatch_pending",
        r#"{"value":"changed"}"#,
    ));
    let events = until(&session, is_error).await;
    assert!(error_message(events.last().unwrap()).contains("unique call ID"));

    release.notify_one();
    until(&session, is_tool_end).await;
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(tool_outputs(&model).len(), 1);
}

#[tokio::test]
async fn a_changed_completed_call_id_fails_before_execution() {
    let model = Arc::new(FakeRealtimeModel::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let tool = {
        let runs = runs.clone();
        FunctionTool::new_raw(
            "test_function",
            "counts",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                runs.fetch_add(1, Ordering::SeqCst);
                async move { Ok("function_result".to_string()) }
            },
        )
    };
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let session = start(&model, agent, sync_config(), None).await;
    model.emit(call("test_function", "call-reused", r#"{"value":"safe"}"#));
    model.emit(call(
        "test_function",
        "call-reused",
        r#"{"value":"changed"}"#,
    ));
    let events = drain(&model, &session).await;
    let errors: Vec<_> = events.iter().filter(|e| is_error(e)).collect();
    assert_eq!(errors.len(), 1);
    assert!(error_message(errors[0]).contains("unique call ID"));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(tool_outputs(&model).len(), 1);
}

#[tokio::test]
async fn a_reused_call_id_fails_when_the_role_changes_between_agents() {
    for run in [sync_config(), Default::default()] {
        let target = RealtimeAgent::<()>::new("target").into_arc();
        let route = realtime_handoff(target.clone()).tool_name;
        let runs = Arc::new(AtomicUsize::new(0));
        let function_tool = {
            let runs = runs.clone();
            FunctionTool::new_raw(
                route.clone(),
                "function with the handoff's name",
                json!({"type": "object", "properties": {}}),
                move |_ctx, _args| {
                    runs.fetch_add(1, Ordering::SeqCst);
                    async move { Ok("function result".to_string()) }
                },
            )
        };
        let function_agent = RealtimeAgent::<()>::new("function")
            .with_tool(function_tool)
            .into_arc();
        let handoff_agent = RealtimeAgent::<()>::new("handoff")
            .with_handoff(target)
            .into_arc();
        let model = Arc::new(FakeRealtimeModel::new());
        let session = start(&model, function_agent, run, None).await;

        model.emit(call(&route, "shared", "{}"));
        until(&session, is_tool_end).await;
        session.update_agent(handoff_agent).await.unwrap();
        model.emit(call(&route, "shared", "{}"));
        let events = until(&session, is_error).await;
        assert!(error_message(events.last().unwrap()).contains("unique call ID"));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn a_handoff_retry_after_a_settings_failure_does_not_repeat_the_callback() {
    let fail = Arc::new(AtomicBool::new(true));
    let target = {
        let fail = fail.clone();
        RealtimeAgent::<()>::new("target")
            .with_instructions_fn(move |_ctx, _agent| {
                let fail = fail.load(Ordering::SeqCst);
                async move {
                    if fail {
                        Err(AgentsError::other("settings failed"))
                    } else {
                        Ok("target".to_string())
                    }
                }
            })
            .into_arc()
    };
    // The handoff to `target` is built with a callback that counts invocations.
    let callbacks = Arc::new(AtomicUsize::new(0));
    let route = {
        let callbacks = callbacks.clone();
        let mut route = realtime_handoff(target.clone()).with_on_handoff_input::<Value, _, _>(
            move |_ctx, _input| {
                callbacks.fetch_add(1, Ordering::SeqCst);
                async { Ok(()) }
            },
        );
        route.tool_name = "route".into();
        route
    };
    let agent = RealtimeAgent::<()>::new("source")
        .with_handoff(route)
        .into_arc();
    // `target` fails its instructions only after connect, so connect itself succeeds.
    fail.store(false, Ordering::SeqCst);
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config(), None).await;
    fail.store(true, Ordering::SeqCst);

    model.emit(call("route", "shared", "{}"));
    let events = until(&session, is_error).await;
    assert_eq!(
        error_message(events.last().unwrap()),
        "Tool call task failed"
    );
    while session.recv().await.is_some() {}
    assert!(
        session
            .take_error()
            .unwrap()
            .to_string()
            .contains("settings failed")
    );
    assert_eq!(callbacks.load(Ordering::SeqCst), 1);
    // The failure ended the session, so a retry cannot be delivered; the callback ran once and
    // the current agent did not change.
    assert_eq!(session.current_agent_name(), "source");
}

#[tokio::test]
async fn a_tool_call_without_a_call_id_is_rejected() {
    let model = Arc::new(FakeRealtimeModel::new());
    let agent = RealtimeAgent::<()>::new("source")
        .with_handoff(realtime_handoff(
            RealtimeAgent::<()>::new("target").into_arc(),
        ))
        .into_arc();
    let session = start(&model, agent, Default::default(), None).await;
    model.emit(call("transfer_to_target", "", "{}"));
    let events = until(&session, is_error).await;
    assert!(error_message(events.last().unwrap()).contains("non-empty string call ID"));
    assert_eq!(session.current_agent_name(), "source");
    assert!(tool_outputs(&model).is_empty());
}

#[tokio::test]
async fn complex_arguments_reach_the_tool_and_the_events_unchanged() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let tool = {
        let seen = seen.clone();
        FunctionTool::new_raw(
            "test_function",
            "records",
            json!({"type": "object", "properties": {}}),
            move |ctx, args| {
                seen.lock().unwrap().push((ctx.tool_call_id.clone(), args));
                async move { Ok("function_result".to_string()) }
            },
        )
    };
    let agent = RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(&model, agent, sync_config(), None).await;
    let complex = r#"{"nested": {"data": [1, 2, 3]}, "bool": true, "null": null}"#;
    model.emit(call("test_function", "custom_call_id_12345", complex));
    let events = drain(&model, &session).await;

    assert_eq!(
        *seen.lock().unwrap(),
        [("custom_call_id_12345".to_string(), complex.to_string())]
    );
    let start_args = events.iter().find_map(|e| match e {
        RealtimeSessionEvent::ToolStart { arguments, .. } => Some(arguments.clone()),
        _ => None,
    });
    let end_args = events.iter().find_map(|e| match e {
        RealtimeSessionEvent::ToolEnd { arguments, .. } => Some(arguments.clone()),
        _ => None,
    });
    assert_eq!(start_args.as_deref(), Some(complex));
    assert_eq!(end_args.as_deref(), Some(complex));
}
