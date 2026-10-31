//! Ports of `tests/sandbox/capabilities/test_tool_input_errors.py`: invalid model input comes
//! back as one safe line naming only the failed fields, never echoing a value, and never
//! reaches the backend.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::RunContext;
use openai_agents_sdk::sandbox::capabilities::pty_tools::{exec_command_tool, write_stdin_tool};
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding, Filesystem};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::testing::{ScriptedSandboxSession, scripted_sandbox_session};
use openai_agents_sdk::sandbox::workspace_paths::SandboxWorkspaceScope;
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;

fn session() -> Arc<ScriptedSandboxSession> {
    Arc::new(
        scripted_sandbox_session(vec![]).with_manifest(Manifest::new().with_root("/workspace")),
    )
}

fn function(tool: Tool<()>) -> Arc<FunctionTool<()>> {
    match tool {
        Tool::Function(function) => function,
        _ => panic!("a function tool"),
    }
}

fn tool(name: &str, session: &Arc<ScriptedSandboxSession>) -> Arc<FunctionTool<()>> {
    match name {
        "exec_command" => function(exec_command_tool::<()>(
            session.clone(),
            None,
            SandboxWorkspaceScope::default(),
            false,
        )),
        "write_stdin" => function(write_stdin_tool::<()>(session.clone(), false)),
        _ => Capability::<()>::tools(&Filesystem::new(), &CapabilityBinding::new(session.clone()))
            .unwrap()
            .into_iter()
            .find_map(|tool| match tool {
                Tool::Function(function) if function.name == name => Some(function),
                _ => None,
            })
            .expect("view_image"),
    }
}

/// Python: `test_invalid_model_input_returns_safe_feedback_without_backend_calls`.
#[tokio::test]
async fn invalid_model_input_returns_safe_feedback_without_backend_calls() {
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "exec_command",
            r#"{"cmd": "pwd", "workdir": "/outside/private-sentinel"}"#.into(),
            "Invalid workdir.",
        ),
        (
            "exec_command",
            r#"{"cmd": ""}"#.into(),
            "Invalid tool arguments for cmd.",
        ),
        (
            "exec_command",
            r#"{"cmd": "private-sentinel", "max_output_tokens": 0}"#.into(),
            "Invalid tool arguments for max_output_tokens.",
        ),
        (
            "exec_command",
            r#"{"cmd": "private-sentinel""#.into(),
            "Invalid tool arguments.",
        ),
        (
            "view_image",
            r#"{"path": "/outside/private-sentinel"}"#.into(),
            "Invalid path.",
        ),
        (
            "view_image",
            r#"{"path": {"private-sentinel": "value"}}"#.into(),
            "Invalid tool arguments for path.",
        ),
        (
            "write_stdin",
            r#"{"chars": "private-sentinel"}"#.into(),
            "Invalid tool arguments for session_id.",
        ),
    ];
    for (name, arguments, expected) in cases {
        let session = session();
        let tool = tool(name, &session);
        let ctx = ToolContext::direct(RunContext::new(()), "agent", name, &arguments);
        let output = tool.invoke(ctx, arguments.clone()).await.expect("feedback");
        let ToolOutput::Text(text) = output else {
            panic!("text feedback for {arguments}");
        };
        assert!(text.starts_with(expected), "{arguments}: {text}");
        assert!(!text.contains("private-sentinel"), "{text}");
        assert!(session.calls().is_empty(), "{arguments}: no backend call");
    }
}

/// Several failing fields are listed in declaration order.
#[tokio::test]
async fn failing_fields_are_listed_in_declaration_order() {
    let session = session();
    let tool = tool("exec_command", &session);
    let arguments = r#"{"cmd": "", "tty": "yes", "max_output_tokens": 0}"#.to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "exec_command", &arguments);
    let ToolOutput::Text(text) = tool.invoke(ctx, arguments).await.unwrap() else {
        panic!("text");
    };
    assert_eq!(
        text,
        "Invalid tool arguments for cmd, tty, max_output_tokens. Check the tool schema and try again."
    );
}
