//! Port of the behavioural cases of `tests/test_source_compat_constructors.py`.
//!
//! Almost every case in that file pins a positional constructor signature (`RunConfig(None, ...)`,
//! `ModelSettings(...)`, `RunResultStreaming(...)`, ...) or a keyword alias kept for older callers.
//! Rust has no positional arguments for these types and no alias keywords, so those cases are
//! python-only. The ones below check defaults and field values that the Rust types do expose.
//!
//! Python-only groups:
//! - positional `RunConfig`, `ModelSettings`, `ToolContext`, `RunResult` and `RunResultStreaming`
//!   constructors, and the `FunctionTool` positional guardrail slots: Rust builds these with
//!   named fields and builders.
//! - `RunErrorDetails` mutable-default aliasing: Rust vectors are owned per value, so two
//!   instances cannot share a default list.
//! - `ToolExecutionConfig` rejecting a non-bool guardrail flag: the field is typed `bool`.
//! - `AgentHookContext`: Rust has no such type; hook callbacks receive the run context directly.
//! - keyword aliases such as `_run_impl_task` and `run_loop_task`: Rust has one field name.

use openai_agents_sdk::run_config::ToolExecutionConfig;
use openai_agents_sdk::tool::{FunctionTool, ToolTimeoutBehavior};
use openai_agents_sdk::{Agent, RunContext, ToolContext};

#[test]
fn tool_execution_config_keeps_concurrency_and_pre_approval_flag() {
    let config = ToolExecutionConfig {
        max_function_tool_concurrency: Some(2),
        pre_approval_tool_input_guardrails: true,
    };
    assert_eq!(config.max_function_tool_concurrency, Some(2));
    assert!(config.pre_approval_tool_input_guardrails);

    let defaults = ToolExecutionConfig::default();
    assert_eq!(defaults.max_function_tool_concurrency, None);
    assert!(!defaults.pre_approval_tool_input_guardrails);
}

#[test]
fn function_tool_defaults_have_no_timeout_and_error_as_result() {
    let tool = FunctionTool::<()>::new_no_args("tool_name", "tool_description", |_c| async {
        Ok("ok".to_string())
    });
    assert_eq!(tool.timeout, None);
    assert_eq!(tool.timeout_behavior, ToolTimeoutBehavior::ErrorAsResult);
    assert!(tool.timeout_error_function.is_none());
    assert!(matches!(
        tool.needs_approval,
        openai_agents_sdk::tool::NeedsApproval::Static(false)
    ));
}

#[test]
fn tool_context_carries_the_agent_it_was_built_with() {
    let agent = std::sync::Arc::new(Agent::<()>::new("agent"));
    let context =
        ToolContext::<()>::direct(RunContext::new(()), "agent", "tool_name", r#"{"x":1}"#)
            .with_agent(agent.clone());

    assert_eq!(context.tool_name, "tool_name");
    assert_eq!(context.tool_arguments, r#"{"x":1}"#);
    assert!(std::sync::Arc::ptr_eq(
        context.agent.as_ref().unwrap(),
        &agent
    ));
}
