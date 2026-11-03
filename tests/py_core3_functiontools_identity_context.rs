//! Ports of `tests/test_tool_identity.py`, `tests/test_tool_context.py` and
//! `tests/test_custom_tool.py`.
//!
//! The `_tool_identity` helpers are private in Rust and operate on typed calls, so the observable
//! behaviour is asserted through `FunctionTool`, `ToolContext` and the Runner. Python-only
//! inputs (raw dicts, `None` names, annotation-based dispatch) are not ported.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{FunctionCall, OutputItem};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{CustomTool, FunctionTool, tool_namespace};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunConfig, Runner};
use serde_json::{Value, json};

fn lookup(name: &str) -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args(name, "d", |_ctx| async { Ok("ok") })
}

/// A function call with an explicit namespace in the wire `extra` map, as the model sends it.
fn call_with_namespace(name: &str, call_id: &str, namespace: Option<&str>) -> FunctionCall {
    let mut call = FunctionCall::new(name, "{}");
    call.call_id = call_id.to_string();
    if let Some(ns) = namespace {
        call.extra.insert("namespace".into(), json!(ns));
    }
    call
}

fn tool_context(call: &FunctionCall, agent: Option<Arc<Agent<()>>>) -> ToolContext<()> {
    let ctx = ToolContext::from_call(RunContext::new(()), "agent", call);
    match agent {
        Some(agent) => ctx.with_agent(agent),
        None => ctx,
    }
}

// ---------------------------------------------------------------------------
// test_tool_identity.py
// ---------------------------------------------------------------------------

#[test]
fn qualified_name_without_namespace_is_the_name() {
    assert_eq!(lookup("search").qualified_name(), "search");
}

#[test]
fn qualified_name_with_namespace_is_dotted() {
    let tool = tool_namespace("tools", "Tools", vec![lookup("search")])
        .unwrap()
        .remove(0);
    assert_eq!(tool.qualified_name(), "tools.search");
}

#[test]
fn reserved_synthetic_namespace_is_detected_by_name_equality() {
    // Python: `is_reserved_synthetic_tool_namespace(name, namespace)` is true only when they match.
    let err = tool_namespace("search", "Same-name", vec![lookup("search")]).unwrap_err();
    assert!(
        err.to_string().contains("reserves the synthetic namespace"),
        "{err}"
    );
    assert!(tool_namespace("tools", "Tools", vec![lookup("search")]).is_ok());
}

#[test]
fn validate_namespace_shape_rejects_synthetic_and_accepts_real_namespaces() {
    // Python: `validate_function_tool_namespace_shape(name, namespace)`.
    assert!(tool_namespace("tools", "Tools", vec![lookup("search")]).is_ok());
    let err = tool_namespace("search", "Tools", vec![lookup("search")]).unwrap_err();
    assert!(
        err.to_string().contains("reserves the synthetic namespace"),
        "{err}"
    );
}

#[test]
fn trace_and_lookup_names_follow_the_call_namespace() {
    // Python `get_tool_call_qualified_name` / `get_tool_call_trace_name` / lookup keys. Rust
    // derives them from the call's namespace in the tool context.
    let real = call_with_namespace("search", "c1", Some("tools"));
    let synthetic = call_with_namespace("search", "c2", Some("search"));
    let bare = call_with_namespace("search", "c3", None);

    assert_eq!(
        tool_context(&real, None).qualified_tool_name(),
        "tools.search"
    );
    assert_eq!(
        tool_context(&synthetic, None).qualified_tool_name(),
        "search"
    );
    assert_eq!(tool_context(&bare, None).qualified_tool_name(), "search");
}

#[test]
fn an_empty_namespace_in_the_call_is_ignored() {
    let call = call_with_namespace("search", "c1", Some(""));
    let ctx = tool_context(&call, None);
    assert_eq!(ctx.tool_namespace, None);
    assert_eq!(ctx.qualified_tool_name(), "search");
}

// ---------------------------------------------------------------------------
// test_tool_context.py
// ---------------------------------------------------------------------------

#[test]
fn tool_context_from_call_populates_fields() {
    let call = FunctionCall::new("test_tool", r#"{"a": 1}"#);
    let mut call = call;
    call.call_id = "call-123".to_string();
    let agent = Arc::new(Agent::<()>::new("agent"));

    let ctx = tool_context(&call, Some(agent.clone()));

    assert_eq!(ctx.tool_name, "test_tool");
    assert_eq!(ctx.tool_call_id, "call-123");
    assert_eq!(ctx.tool_arguments, r#"{"a": 1}"#);
    assert!(Arc::ptr_eq(ctx.agent.as_ref().unwrap(), &agent));
}

#[test]
fn tool_context_agent_is_none_by_default() {
    let call = FunctionCall::new("test_tool", "{}");
    assert!(tool_context(&call, None).agent.is_none());
}

#[test]
fn tool_context_accepts_an_agent_directly() {
    let agent = Arc::new(Agent::<()>::new("direct-agent"));
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "my_tool", "{}")
        .with_agent(agent.clone());
    assert!(Arc::ptr_eq(ctx.agent.as_ref().unwrap(), &agent));
}

#[test]
fn tool_context_infers_namespace_from_tool_call() {
    let call = call_with_namespace("lookup_account", "call-2", Some("billing"));
    let ctx = tool_context(&call, None);

    assert_eq!(ctx.tool_namespace.as_deref(), Some("billing"));
    assert_eq!(ctx.qualified_tool_name(), "billing.lookup_account");
}

#[test]
fn tool_context_qualified_tool_name_collapses_synthetic_namespace() {
    let call = call_with_namespace("get_weather", "call-weather", Some("get_weather"));
    let ctx = tool_context(&call, None);

    assert_eq!(ctx.tool_namespace.as_deref(), Some("get_weather"));
    assert_eq!(ctx.qualified_tool_name(), "get_weather");
}

#[test]
fn tool_context_run_config_is_the_explicit_one() {
    let config = Arc::new(RunConfig::<()>::default().with_tracing_disabled(true));
    let call = FunctionCall::new("test_tool", "{}");
    let ctx = tool_context(&call, None).with_run_config(config.clone());
    assert!(Arc::ptr_eq(ctx.run_config.as_ref().unwrap(), &config));
    assert!(ctx.run_config.as_ref().unwrap().tracing_disabled);
}

#[tokio::test]
async fn invoke_preserves_the_tool_context_fields() {
    // Python: a tool that asks for the `ToolContext` receives the same object back.
    let tool = FunctionTool::<String>::new_raw(
        "tool_context_tool",
        "test",
        json!({"type": "object", "properties": {}}),
        |ctx: ToolContext<String>, _args: String| async move {
            Ok(format!(
                "{}|{}|{}",
                ctx.tool_name,
                ctx.tool_call_id,
                ctx.context()
            ))
        },
    );
    let ctx = ToolContext::direct(
        RunContext::new("Stormy".to_string()),
        "agent",
        "tool_context_tool",
        "{}",
    );
    let out = tool.invoke(ctx.clone(), "{}".to_string()).await.unwrap();
    assert_eq!(
        out.to_model_string(),
        format!("tool_context_tool|{}|Stormy", ctx.tool_call_id)
    );
}

// ---------------------------------------------------------------------------
// test_custom_tool.py
// ---------------------------------------------------------------------------

fn custom_output(items: &[Value]) -> Value {
    items
        .iter()
        .find(|i| i["type"] == "custom_tool_call_output")
        .expect("a custom_tool_call_output")
        .clone()
}

fn custom_call(name: &str, input: &str) -> OutputItem {
    OutputItem::Other(json!({
        "type": "custom_tool_call",
        "name": name,
        "call_id": "call_custom",
        "input": input,
    }))
}

#[tokio::test]
async fn custom_tool_action_returns_custom_tool_call_output() {
    let seen_name = Arc::new(Mutex::new(String::new()));
    let seen = seen_name.clone();
    let tool = CustomTool::<()>::new("raw_editor", "Edit raw text.", move |ctx, input: String| {
        *seen.lock().unwrap() = ctx.tool_name.clone();
        async move { Ok(input.to_uppercase()) }
    })
    .with_format(json!({"type": "text"}));

    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![custom_call("raw_editor", "hello")]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("custom-agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    Runner::run(agent, "start").await.unwrap();

    assert_eq!(*seen_name.lock().unwrap(), "raw_editor");
    let items: Vec<Value> = model
        .last_call()
        .unwrap()
        .request
        .input
        .iter()
        .map(|i| i.as_value().clone())
        .collect();
    assert_eq!(
        custom_output(&items),
        json!({
            "type": "custom_tool_call_output",
            "call_id": "call_custom",
            "output": "HELLO",
        })
    );
}

#[tokio::test]
async fn custom_tool_on_approval_auto_rejects_with_reason() {
    let tool = CustomTool::<()>::new(
        "raw_editor",
        "Edit raw text.",
        |_ctx, _input: String| async {
            Err::<String, _>(AgentsError::Other(
                "rejected custom tool should not execute".into(),
            ))
        },
    )
    .with_format(json!({"type": "text"}))
    .with_needs_approval(true)
    .with_on_approval(|_ctx, _item| async {
        Some(openai_agents_sdk::tool::ApprovalDecision::reject(Some(
            "Not allowed".to_string(),
        )))
    });

    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![custom_call("raw_editor", "hello")]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("custom-agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let result = Runner::run(agent, "start").await.unwrap();
    assert!(result.interruptions.is_empty());

    let items: Vec<Value> = model
        .last_call()
        .unwrap()
        .request
        .input
        .iter()
        .map(|i| i.as_value().clone())
        .collect();
    assert_eq!(
        custom_output(&items),
        json!({
            "type": "custom_tool_call_output",
            "call_id": "call_custom",
            "output": "Not allowed",
        })
    );
}
