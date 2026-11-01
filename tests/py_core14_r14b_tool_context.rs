//! Python `tests/test_tool_context.py`: how a `ToolContext` is populated from a tool call.
use std::sync::Arc;

use openai_agents_sdk::Agent;
use openai_agents_sdk::items::FunctionCall;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool_context::ToolContext;
use serde_json::json;

fn call(name: &str, call_id: &str, arguments: &str) -> FunctionCall {
    let mut call = FunctionCall::new(name, arguments);
    call.call_id = call_id.to_string();
    call
}

/// Python: `test_tool_context_from_agent_context_populates_fields` and
/// `test_tool_context_agent_none_by_default`.
#[test]
fn a_tool_context_copies_the_call_fields_and_has_no_agent_unless_given_one() {
    let tool_call = call("test_tool", "call-123", r#"{"a": 1}"#);
    let plain = ToolContext::<()>::from_call(RunContext::new(()), "agent", &tool_call);

    assert_eq!(plain.tool_name, "test_tool");
    assert_eq!(plain.tool_call_id, "call-123");
    assert_eq!(plain.tool_arguments, r#"{"a": 1}"#);
    assert!(plain.agent.is_none());

    let agent = Arc::new(Agent::<()>::new("agent"));
    let with_agent = plain.with_agent(agent.clone());
    assert!(Arc::ptr_eq(with_agent.agent.as_ref().unwrap(), &agent));
}

/// Python: `test_tool_context_constructor_infers_namespace_from_tool_call`.
#[test]
fn the_namespace_comes_from_the_tool_call() {
    let mut tool_call = call("lookup_account", "call-2", "{}");
    tool_call.extra.insert("namespace".into(), json!("billing"));

    let context = ToolContext::<()>::from_call(RunContext::new(()), "agent", &tool_call);

    assert_eq!(context.tool_namespace.as_deref(), Some("billing"));
    assert_eq!(context.qualified_tool_name(), "billing.lookup_account");
}

/// Python: `test_tool_context_from_agent_context_prefers_explicit_run_config`.
#[test]
fn an_explicit_run_config_is_kept_as_given() {
    let tool_call = call("test_tool", "call-1", "{}");
    let config = Arc::new(RunConfig::<()>::default());

    let context = ToolContext::<()>::from_call(RunContext::new(()), "agent", &tool_call)
        .with_run_config(config.clone());

    assert!(Arc::ptr_eq(context.run_config.as_ref().unwrap(), &config));
}
