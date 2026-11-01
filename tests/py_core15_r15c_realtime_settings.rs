//! Ports of `tests/realtime/test_session.py` model-settings cases: the names that reach the
//! model are the final ones, after connect-time overrides.
#![cfg(feature = "realtime")]

use std::sync::Arc;

use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeFunctionToolSpec, RealtimeModelConfig,
    RealtimeSession, RealtimeSessionModelSettings, realtime_handoff,
};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

fn raw_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new_raw(
        name,
        "test tool",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async { Ok("ok".to_string()) },
    )
}

fn conflicting_agent() -> Arc<RealtimeAgent<()>> {
    // The default handoff tool name for an agent called "billing" is `transfer_to_billing`.
    RealtimeAgent::<()>::new("agent")
        .with_tool(raw_tool("transfer_to_billing"))
        .with_handoff(realtime_handoff(
            RealtimeAgent::<()>::new("billing").into_arc(),
        ))
        .into_arc()
}

async fn connect(
    agent: Arc<RealtimeAgent<()>>,
    overrides: Option<RealtimeSessionModelSettings>,
) -> (
    Arc<FakeRealtimeModel>,
    Result<RealtimeSession<()>, openai_agents_sdk::error::AgentsError>,
) {
    let model = Arc::new(FakeRealtimeModel::new());
    let config = RealtimeModelConfig {
        initial_model_settings: overrides,
        ..Default::default()
    };
    let session = RealtimeSession::connect(model.clone(), agent, (), Some(config), None).await;
    (model, session)
}

/// `test_updated_model_settings_validates_final_tool_names_after_overrides` and
/// `test_aenter_validates_initial_model_settings_before_listener_registration`: a tool and a
/// handoff that share a name are refused, unless the connect-time override replaces them.
#[tokio::test]
async fn an_override_that_replaces_the_conflicting_names_is_accepted() {
    let (_, refused) = connect(conflicting_agent(), None).await;
    let Err(error) = refused else {
        panic!("the duplicate names are refused");
    };
    assert!(
        error.to_string().contains("Duplicate realtime tool names"),
        "{error}"
    );

    let mut overrides = RealtimeSessionModelSettings::new();
    overrides.tools = Some(vec![RealtimeFunctionToolSpec::new(
        "lookup_account",
        "",
        json!({"type": "object", "properties": {}}),
    )]);
    let (model, session) = connect(conflicting_agent(), Some(overrides)).await;
    let _session = session.expect("the final names are unique");
    let tools = model.last_session_settings().unwrap().tools.unwrap();
    assert_eq!(
        tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["lookup_account"]
    );
}

/// An override that keeps both names still fails.
#[tokio::test]
async fn an_override_that_keeps_the_conflict_is_refused() {
    let spec = |name: &str| RealtimeFunctionToolSpec::new(name, "", json!({}));
    let mut overrides = RealtimeSessionModelSettings::new();
    overrides.tools = Some(vec![spec("same"), spec("same")]);
    let (_, session) = connect(
        RealtimeAgent::<()>::new("agent").into_arc(),
        Some(overrides),
    )
    .await;
    let Err(error) = session else {
        panic!("refused");
    };
    assert!(error.to_string().contains("same"), "{error}");
}
