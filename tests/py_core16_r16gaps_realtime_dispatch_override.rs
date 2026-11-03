//! A tool list given in the connect-time model settings replaces the agent's: a tool the list
//! no longer names is neither offered to the model nor dispatched when the model calls it,
//! because Python builds its dispatch snapshot from the merged settings.
#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeFunctionToolSpec, RealtimeModelConfig,
    RealtimeModelEvent, RealtimeModelToolCallEvent, RealtimeRunConfig, RealtimeSession,
    RealtimeSessionEvent, RealtimeSessionModelSettings,
};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

fn counting_tool(name: &str, runs: Arc<AtomicUsize>) -> FunctionTool<()> {
    FunctionTool::new_raw(
        name,
        "test tool",
        json!({"type": "object", "properties": {}}),
        move |_ctx, _args| {
            let runs = runs.clone();
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                Ok("ok".to_string())
            }
        },
    )
}

#[tokio::test]
async fn a_tool_removed_by_the_connect_time_list_is_not_dispatched() {
    let kept = Arc::new(AtomicUsize::new(0));
    let removed = Arc::new(AtomicUsize::new(0));
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(counting_tool("kept", kept.clone()))
        .with_tool(counting_tool("removed", removed.clone()))
        .into_arc();
    let mut overrides = RealtimeSessionModelSettings::new();
    overrides.tools = Some(vec![RealtimeFunctionToolSpec::new(
        "kept",
        "test tool",
        json!({"type": "object", "properties": {}}),
    )]);
    let model = Arc::new(FakeRealtimeModel::new());
    let session = RealtimeSession::connect(
        model.clone(),
        agent,
        (),
        Some(RealtimeModelConfig {
            initial_model_settings: Some(overrides),
            ..Default::default()
        }),
        Some(RealtimeRunConfig {
            async_tool_calls: false,
            ..Default::default()
        }),
    )
    .await
    .expect("connects");

    for (name, id) in [("removed", "call_removed"), ("kept", "call_kept")] {
        model.emit(RealtimeModelEvent::FunctionCall(
            RealtimeModelToolCallEvent::new(name, id, "{}"),
        ));
    }
    // Wait for the kept tool's end event: both calls have been handled by then.
    loop {
        let event = tokio::time::timeout(Duration::from_secs(5), session.recv())
            .await
            .expect("an event arrives")
            .expect("the stream is open");
        if matches!(event, RealtimeSessionEvent::ToolEnd { .. }) {
            break;
        }
    }

    assert_eq!(kept.load(Ordering::SeqCst), 1);
    assert_eq!(removed.load(Ordering::SeqCst), 0);
}
