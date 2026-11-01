//! Ports of the `tests/realtime/test_session.py` and `test_realtime_model_settings.py` cases whose
//! connect-time settings carry live tools or handoffs (`initial_model_settings["tools"]` /
//! `["handoffs"]`): `RealtimeRunConfig::with_initial_tools` / `with_initial_handoffs`.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::realtime::{
    FakeRealtimeModel, OpenAIRealtimeSIPModel, RealtimeAgent, RealtimeHandoff, RealtimeModelEvent,
    RealtimeModelToolCallEvent, RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

fn tool(name: &str, output: &str) -> FunctionTool<()> {
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

fn billing_handoff(enabled: bool) -> RealtimeHandoff<()> {
    RealtimeHandoff::new(RealtimeAgent::<()>::new("billing").into_arc())
        .with_tool_name("transfer_to_billing")
        .with_tool_description("Transfer to billing")
        .with_is_enabled(enabled)
}

fn sync_config() -> RealtimeRunConfig<()> {
    RealtimeRunConfig {
        async_tool_calls: false,
        ..Default::default()
    }
}

async fn connect(
    agent: Arc<RealtimeAgent<()>>,
    config: RealtimeRunConfig<()>,
) -> (Arc<FakeRealtimeModel>, RealtimeSession<()>) {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = RealtimeSession::connect(model.clone(), agent, (), None, Some(config))
        .await
        .expect("connects");
    (model, session)
}

async fn call_and_wait(model: &FakeRealtimeModel, session: &RealtimeSession<()>, name: &str) {
    model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new(name, "call_1", "{}"),
    ));
    loop {
        let event = tokio::time::timeout(Duration::from_secs(5), session.recv())
            .await
            .expect("event")
            .expect("stream open");
        if matches!(
            event,
            RealtimeSessionEvent::ToolEnd { .. } | RealtimeSessionEvent::Handoff { .. }
        ) {
            return;
        }
    }
}

fn outputs(model: &FakeRealtimeModel) -> Vec<String> {
    model
        .sent_events()
        .iter()
        .filter_map(|e| e.as_tool_output().map(|(_, o, _)| o.to_string()))
        .collect()
}

fn tool_names(model: &FakeRealtimeModel) -> Vec<String> {
    model
        .last_session_settings()
        .unwrap()
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.name)
        .collect()
}

/// `test_initial_settings_handoff_override_does_not_block_function_dispatch`.
#[tokio::test]
async fn an_empty_handoff_override_does_not_block_a_function_tool_of_the_same_name() {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(tool("transfer_to_billing", "function ok"))
        .with_handoff(billing_handoff(true))
        .into_arc();
    let (model, session) = connect(agent, sync_config().with_initial_handoffs(Vec::new())).await;
    assert_eq!(tool_names(&model), ["transfer_to_billing"]);
    call_and_wait(&model, &session, "transfer_to_billing").await;
    assert_eq!(outputs(&model), ["function ok"]);
}

/// `test_initial_settings_function_tool_override_is_dispatchable`.
#[tokio::test]
async fn an_override_tool_is_advertised_and_dispatched() {
    let agent = RealtimeAgent::<()>::new("agent").into_arc();
    let config = sync_config().with_initial_tools(vec![tool("override_tool", "override ok")]);
    let (model, session) = connect(agent, config).await;
    assert_eq!(tool_names(&model), ["override_tool"]);
    call_and_wait(&model, &session, "override_tool").await;
    assert_eq!(outputs(&model), ["override ok"]);
}

/// `test_initial_settings_same_name_function_tool_override_is_dispatched`.
#[tokio::test]
async fn a_same_name_override_tool_replaces_the_agent_implementation() {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(tool("shared_tool", "agent implementation"))
        .into_arc();
    let config =
        sync_config().with_initial_tools(vec![tool("shared_tool", "override implementation")]);
    let (model, session) = connect(agent, config).await;
    call_and_wait(&model, &session, "shared_tool").await;
    assert_eq!(outputs(&model), ["override implementation"]);
}

/// `test_initial_settings_handoff_override_is_dispatchable`.
#[tokio::test]
async fn an_override_handoff_is_advertised_and_dispatched() {
    let agent = RealtimeAgent::<()>::new("agent").into_arc();
    let config = sync_config().with_initial_handoffs(vec![billing_handoff(true)]);
    let (model, session) = connect(agent, config).await;
    assert_eq!(tool_names(&model), ["transfer_to_billing"]);
    call_and_wait(&model, &session, "transfer_to_billing").await;
    assert_eq!(
        outputs(&model),
        [json!({"assistant": "billing"}).to_string()]
    );
    assert_eq!(
        model
            .last_session_settings()
            .unwrap()
            .instructions
            .as_deref(),
        Some("")
    );
}

/// `test_aenter_filters_disabled_override_handoff_name_conflict` and
/// `test_build_model_settings_filters_disabled_starting_handoff_name_conflict`: a disabled
/// override handoff sharing a name with an agent tool is dropped, not a duplicate.
#[tokio::test]
async fn a_disabled_override_handoff_does_not_conflict_with_an_agent_tool() {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(tool("transfer_to_billing", "ok"))
        .into_arc();
    let config =
        RealtimeRunConfig::<()>::default().with_initial_handoffs(vec![billing_handoff(false)]);
    let (model, _session) = connect(agent, config).await;
    assert_eq!(tool_names(&model), ["transfer_to_billing"]);
}

/// `test_aenter_filters_disabled_override_tool_name_conflict`: a disabled override tool does not
/// conflict with the agent's handoff and the handoff stays.
#[tokio::test]
async fn a_disabled_override_tool_does_not_conflict_with_an_agent_handoff() {
    let agent = RealtimeAgent::<()>::new("agent")
        .with_handoff(billing_handoff(true))
        .into_arc();
    let disabled = tool("transfer_to_billing", "ok").with_is_enabled(false);
    let config = RealtimeRunConfig::<()>::default().with_initial_tools(vec![disabled]);
    let (model, _session) = connect(agent, config).await;
    assert_eq!(tool_names(&model), ["transfer_to_billing"]);
    let spec = &model.last_session_settings().unwrap().tools.unwrap()[0];
    assert_eq!(spec.description, "Transfer to billing");
}

/// `test_build_model_settings_evaluates_starting_tool_is_enabled_callable`: an override tool's
/// enablement callable runs once, with the session context and the agent.
#[tokio::test]
async fn an_override_tool_enablement_callable_is_evaluated() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let gated = tool("transfer_to_billing", "ok").with_is_enabled_fn(move |_ctx, _agent| {
        seen.fetch_add(1, Ordering::SeqCst);
        async { false }
    });
    let agent = RealtimeAgent::<()>::new("agent").into_arc();
    let config = RealtimeRunConfig::<()>::default().with_initial_tools(vec![gated]);
    let (model, _session) = connect(agent, config).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(tool_names(&model).is_empty());
}

/// `test_updated_model_settings_validates_final_tool_names_after_overrides` with live objects:
/// an enabled override that keeps the conflict is refused.
#[tokio::test]
async fn an_enabled_override_that_conflicts_is_refused() {
    let agent = RealtimeAgent::<()>::new("agent").into_arc();
    let config = RealtimeRunConfig::<()>::default()
        .with_initial_tools(vec![tool("transfer_to_billing", "ok")])
        .with_initial_handoffs(vec![billing_handoff(true)]);
    let model = Arc::new(FakeRealtimeModel::new());
    let Err(error) = RealtimeSession::connect(model, agent, (), None, Some(config)).await else {
        panic!("the duplicate names are refused");
    };
    assert!(
        error.to_string().contains("Duplicate realtime tool names"),
        "{error}"
    );
}

/// `test_updated_model_settings_evaluates_override_handoff_is_enabled_callable`.
#[tokio::test]
async fn an_override_handoff_enablement_callable_is_evaluated() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let gated = billing_handoff(true).with_is_enabled_fn(move |_ctx, _agent| {
        seen.fetch_add(1, Ordering::SeqCst);
        async { false }
    });
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(tool("transfer_to_billing", "ok"))
        .into_arc();
    let config = RealtimeRunConfig::<()>::default().with_initial_handoffs(vec![gated]);
    let (model, _session) = connect(agent, config).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool_names(&model), ["transfer_to_billing"]);
}

fn payload_tool_names(payload: &serde_json::Value) -> Vec<String> {
    payload["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|tool| tool["name"].as_str().map(str::to_string))
        .collect()
}

/// `test_sip_initial_session_payload_filters_disabled_initial_model_settings_handoff` and
/// `..._override_handoff`: a disabled live handoff override does not duplicate a tool name.
#[tokio::test]
async fn a_sip_payload_drops_a_disabled_override_handoff() {
    let agent = RealtimeAgent::<()>::new("parent")
        .with_tool(tool("transfer_to_billing", "ok"))
        .into_arc();
    let config =
        RealtimeRunConfig::<()>::default().with_initial_handoffs(vec![billing_handoff(false)]);
    let payload = OpenAIRealtimeSIPModel::build_initial_session_payload(
        &agent,
        (),
        None,
        Some(&config),
        None,
    )
    .await
    .unwrap();
    assert_eq!(payload_tool_names(&payload), ["transfer_to_billing"]);
}

/// `test_sip_initial_session_payload_filters_disabled_initial_model_settings_tool` and
/// `..._override_tool`: a disabled live tool override leaves the agent's handoff.
#[tokio::test]
async fn a_sip_payload_drops_a_disabled_override_tool() {
    let agent = RealtimeAgent::<()>::new("parent")
        .with_handoff(billing_handoff(true))
        .into_arc();
    let disabled = tool("transfer_to_billing", "ok").with_is_enabled(false);
    let config = RealtimeRunConfig::<()>::default().with_initial_tools(vec![disabled]);
    let payload = OpenAIRealtimeSIPModel::build_initial_session_payload(
        &agent,
        (),
        None,
        Some(&config),
        None,
    )
    .await
    .unwrap();
    assert_eq!(payload_tool_names(&payload), ["transfer_to_billing"]);
}
