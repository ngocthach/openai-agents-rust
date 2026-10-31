//! Ports of `tests/sandbox/capabilities/test_tool_input_errors.py`: invalid model input comes
//! back as one safe line naming only the failed fields, never echoing a value, and never
//! reaches the backend.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::sandbox::capabilities::pty_tools::{exec_command_tool, write_stdin_tool};
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding, Filesystem};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::testing::{ScriptedSandboxSession, scripted_sandbox_session};
use openai_agents_sdk::sandbox::workspace_paths::SandboxWorkspaceScope;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, RunContext, Runner};
use serde_json::Value;

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

fn cases() -> Vec<(&'static str, String, &'static str)> {
    vec![
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
    ]
}

/// Python: `test_invalid_model_input_returns_safe_feedback_without_backend_calls`, run through
/// the runner in both modes (`streamed` x 7 cases). The model's tool call is rejected with
/// feedback, the model gets that feedback as its next input, and the backend is never called.
#[tokio::test]
async fn invalid_model_input_returns_safe_feedback_without_backend_calls() {
    for streamed in [false, true] {
        for (name, arguments, expected) in cases() {
            let session = session();
            let tool = tool(name, &session);
            let model = Arc::new(FakeModel::new());
            model.add_turn(vec![function_call_with_id(
                name,
                arguments.clone(),
                "bad-input",
            )]);
            model.add_turn(vec![assistant_message("recovered")]);
            let agent = Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(Tool::Function(tool));
            let config = RunConfig::default().with_trace_include_sensitive_data(false);
            let (final_output, items, last_input) = if streamed {
                let mut run = Runner::run_streamed_with_options(
                    agent,
                    "go",
                    RunOptions::new(()).run_config(config),
                );
                {
                    let events = run.stream_events();
                    futures::pin_mut!(events);
                    while let Some(event) = events.next().await {
                        event.expect("the stream does not fail");
                    }
                }
                let result = run.into_result().expect("a completed run");
                (
                    result.final_output_text(),
                    result.new_items.clone(),
                    model.calls()[1].input.last().cloned(),
                )
            } else {
                let result =
                    Runner::run_with_options(agent, "go", RunOptions::new(()).run_config(config))
                        .await
                        .expect("the run completes");
                (
                    result.final_output_text(),
                    result.new_items.clone(),
                    model.calls()[1].input.last().cloned(),
                )
            };

            assert_eq!(
                final_output.as_deref(),
                Some("recovered"),
                "{name} {arguments}"
            );
            let outputs: Vec<Value> = items
                .iter()
                .filter_map(|item| match &item.kind {
                    RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(outputs.len(), 1, "{name} {arguments}");
            let text = outputs[0].as_str().expect("text feedback").to_string();
            assert!(text.starts_with(expected), "{arguments}: {text}");
            assert!(!text.contains("private-sentinel"), "{text}");
            assert_eq!(
                last_input.expect("the model saw the feedback").0["output"],
                outputs[0],
                "{name} {arguments}"
            );
            assert!(session.calls().is_empty(), "{arguments}: no backend call");
        }
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
