//! Ports of the Python SDK's `tests/realtime/test_realtime_handoffs.py`.

#![cfg(feature = "realtime")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{RealtimeAgent, RealtimeHandoff, realtime_handoff};
use openai_agents_sdk::tool::IsEnabled;
use openai_agents_sdk::{Agent, RunContext, handoff};
use schemars::JsonSchema;
use serde::Deserialize;

fn ctx() -> RunContext<()> {
    RunContext::new(())
}

#[test]
fn realtime_handoff_creation() {
    let agent = RealtimeAgent::<()>::new("test_agent").into_arc();
    let handoff_obj = realtime_handoff(agent);

    assert_eq!(handoff_obj.agent_name, "test_agent");
    assert_eq!(handoff_obj.tool_name, "transfer_to_test_agent");
    // Realtime handoffs carry no input filter field at all.
    assert!(matches!(handoff_obj.is_enabled, IsEnabled::Static(true)));
}

#[test]
fn realtime_handoff_with_custom_params() {
    let agent = RealtimeAgent::<()>::new("helper_agent")
        .with_handoff_description("Helps with general tasks")
        .into_arc();
    let default = realtime_handoff(agent.clone());
    assert!(
        default
            .tool_description
            .contains("Helps with general tasks")
    );

    let handoff_obj = realtime_handoff(agent)
        .with_tool_name("custom_handoff")
        .with_tool_description("Custom handoff description")
        .with_is_enabled(IsEnabled::Static(false));

    assert_eq!(handoff_obj.agent_name, "helper_agent");
    assert_eq!(handoff_obj.tool_name, "custom_handoff");
    assert_eq!(handoff_obj.tool_description, "Custom handoff description");
    assert!(matches!(handoff_obj.is_enabled, IsEnabled::Static(false)));
}

#[tokio::test]
async fn realtime_handoff_execution() {
    let agent = RealtimeAgent::<()>::new("target_agent").into_arc();
    let handoff_obj = realtime_handoff(agent.clone());

    let result = handoff_obj.invoke(ctx(), String::new()).await.unwrap();
    assert!(Arc::ptr_eq(&result, &agent));
}

#[tokio::test]
async fn realtime_handoff_with_on_handoff_callback() {
    let agent = RealtimeAgent::<()>::new("callback_agent").into_arc();
    let called = Arc::new(Mutex::new(Vec::new()));
    let sink = called.clone();
    let handoff_obj = realtime_handoff(agent).with_on_handoff(move |_ctx| {
        let sink = sink.clone();
        async move {
            sink.lock().unwrap().push(true);
            Ok(())
        }
    });

    handoff_obj.invoke(ctx(), String::new()).await.unwrap();
    assert_eq!(*called.lock().unwrap(), vec![true]);
    assert_eq!(handoff_obj.agent_name, "callback_agent");
}

#[test]
fn regular_agent_handoff_still_works() {
    let regular = Agent::<()>::new("regular_agent");
    let handoff_obj = handoff(regular);

    assert_eq!(handoff_obj.agent_name, "regular_agent");
    assert_eq!(handoff_obj.tool_name, "transfer_to_regular_agent");
    // Regular handoffs, unlike realtime ones, support input filters.
    assert!(handoff_obj.input_filter.is_none());
}

fn model_behavior_message(error: AgentsError) -> String {
    assert!(
        matches!(error, AgentsError::ModelBehaviorError { .. }),
        "expected a model behavior error, got {error:?}"
    );
    error.to_string()
}

#[tokio::test]
async fn realtime_handoff_missing_input_json_raises_model_error() {
    let agent = RealtimeAgent::<()>::new("x").into_arc();
    let handoff_obj = realtime_handoff(agent)
        .with_on_handoff_input::<i64, _, _>(|_ctx, _value| async move { Ok(()) });

    let error = handoff_obj
        .invoke(ctx(), "null".to_string())
        .await
        .unwrap_err();
    model_behavior_message(error);
}

#[tokio::test]
async fn realtime_handoff_is_enabled_async() {
    let agent = RealtimeAgent::<()>::new("x").into_arc();
    let handoff_obj =
        realtime_handoff(agent).with_is_enabled_fn(|_ctx, _agent_name| async move { true });

    assert!(matches!(handoff_obj.is_enabled, IsEnabled::Dynamic(_)));
    assert!(handoff_obj.is_enabled.evaluate(&ctx(), "x").await);
}

#[tokio::test]
async fn realtime_handoff_rejects_none_input() {
    // Python passes `None` where a string belongs. Rust has no null string, so the closest
    // missing input is the empty string, which a typed handoff must reject as well.
    let agent = RealtimeAgent::<()>::new("x").into_arc();
    let handoff_obj = realtime_handoff(agent)
        .with_on_handoff_input::<i64, _, _>(|_ctx, _value| async move { Ok(()) });

    let error = handoff_obj.invoke(ctx(), String::new()).await.unwrap_err();
    model_behavior_message(error);
}

#[tokio::test]
async fn realtime_handoff_sync_is_enabled_callable() {
    let agent = RealtimeAgent::<()>::new("x").into_arc();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let sink = calls.clone();
    let handoff_obj = realtime_handoff(agent).with_is_enabled_fn(move |_ctx, agent_name| {
        let sink = sink.clone();
        async move {
            sink.lock().unwrap().push(agent_name);
            false
        }
    });

    assert!(!handoff_obj.is_enabled.evaluate(&ctx(), "x").await);
    assert_eq!(*calls.lock().unwrap(), vec!["x".to_string()]);
}

#[tokio::test]
async fn realtime_handoff_sync_on_handoff_executes() {
    let agent = RealtimeAgent::<()>::new("sync").into_arc();
    let called = Arc::new(Mutex::new(Vec::new()));
    let sink = called.clone();
    let handoff_obj =
        realtime_handoff(agent.clone()).with_on_handoff_input::<i64, _, _>(move |_ctx, value| {
            let sink = sink.clone();
            async move {
                sink.lock().unwrap().push(value);
                Ok(())
            }
        });

    let result = handoff_obj.invoke(ctx(), "5".to_string()).await.unwrap();
    assert!(Arc::ptr_eq(&result, &agent));
    assert_eq!(*called.lock().unwrap(), vec![5]);
}

#[tokio::test]
async fn realtime_handoff_on_handoff_without_input_runs() {
    let agent = RealtimeAgent::<()>::new("no_input").into_arc();
    let called = Arc::new(Mutex::new(Vec::new()));
    let sink = called.clone();
    let handoff_obj = realtime_handoff(agent.clone()).with_on_handoff(move |_ctx| {
        let sink = sink.clone();
        async move {
            sink.lock().unwrap().push(true);
            Ok(())
        }
    });

    let result = handoff_obj.invoke(ctx(), String::new()).await.unwrap();
    assert!(Arc::ptr_eq(&result, &agent));
    assert_eq!(*called.lock().unwrap(), vec![true]);
}

#[tokio::test]
async fn realtime_handoff_async_on_handoff_without_input_runs() {
    // Every Rust callback is async; this variant awaits a yield inside the callback to prove the
    // handoff waits for it before returning the agent.
    let agent = RealtimeAgent::<()>::new("async_no_input").into_arc();
    let called = Arc::new(Mutex::new(Vec::new()));
    let sink = called.clone();
    let handoff_obj = realtime_handoff(agent.clone()).with_on_handoff(move |_ctx| {
        let sink = sink.clone();
        async move {
            tokio::task::yield_now().await;
            sink.lock().unwrap().push(true);
            Ok(())
        }
    });

    let result = handoff_obj.invoke(ctx(), String::new()).await.unwrap();
    assert!(Arc::ptr_eq(&result, &agent));
    assert_eq!(*called.lock().unwrap(), vec![true]);
}

#[derive(JsonSchema, Deserialize)]
struct StrictInput {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    age: i64,
}

#[tokio::test]
async fn realtime_handoff_strict_json_rejects_type_coercion() {
    let agent = RealtimeAgent::<()>::new("strict_test").into_arc();
    let handoff_obj: RealtimeHandoff<()> = realtime_handoff(agent.clone())
        .with_on_handoff_input::<StrictInput, _, _>(|_ctx, _data| async move { Ok(()) });

    // `age` is a string "25": strict parsing must reject it instead of coercing.
    let error = handoff_obj
        .invoke(ctx(), r#"{"name": "Alice", "age": "25"}"#.to_string())
        .await
        .unwrap_err();
    let message = model_behavior_message(error);
    assert!(message.contains("Invalid JSON"), "{message}");

    // Correctly typed input is still accepted.
    let result = handoff_obj
        .invoke(ctx(), r#"{"name": "Alice", "age": 25}"#.to_string())
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&result, &agent));
}
