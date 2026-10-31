//! Parity port of the end-to-end cases of `tests/mcp/test_mcp_resume_binding.py`: an approval
//! granted to an MCP tool call stays bound to the server that offered it, across a resume.
//!
//! Python also resumes from a serialized `RunState` and from `Runner.run_streamed`; here the
//! in-memory resume is ported. `RunState::from_json` takes no agent in Rust, so the restore
//! variants ("before_restore", "after_restore", schema-version cases) have no equivalent
//! entry point.
#![cfg(feature = "mcp")]

use std::sync::Arc;

use openai_agents_sdk::mcp::{MCPServer, RequireApproval};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::{Agent, Runner};
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::FakeMcpServer;

fn fake(name: &str, with_tool: bool) -> Arc<FakeMcpServer> {
    let mut server =
        FakeMcpServer::new(name).configure(|c| c.require_approval = RequireApproval::Always);
    if with_tool {
        server = server.add_tool("search", json!({"type": "object", "properties": {}}));
    }
    Arc::new(server)
}

fn dyn_server(server: &Arc<FakeMcpServer>) -> Arc<dyn MCPServer> {
    server.clone()
}

#[tokio::test]
async fn approved_call_runs_on_the_original_recipient() {
    openai_agents_sdk::set_tracing_disabled(true);
    let original = fake("docs", true);
    let other = fake("docs", false);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("search", "{}")]);
    let agent: Agent<()> = Agent::new("test")
        .with_mcp_server(dyn_server(&original))
        .with_mcp_server(dyn_server(&other))
        .with_model_instance(model.clone());

    let first = Runner::run(&agent, "search").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    model.add_turn(vec![assistant_message("done")]);

    let result = Runner::run(&agent, state).await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(original.tool_calls(), vec!["search"]);
    assert!(other.tool_calls().is_empty());
}

#[tokio::test]
#[ignore = "known cross-area divergence: RunState does not bind an approved MCP call to its server, so the call is re-resolved against whichever server now offers the tool (Python raises 'different recipient'); fix lives in src/run_state*/src/run_internal, not src/mcp"]
async fn approval_is_rejected_when_the_recipient_changes() {
    // `test_mcp_recipient_survives_absent_tool_during_restore` (changed_recipient=True):
    // the tool moves to the other same-named server before the approved call resumes.
    openai_agents_sdk::set_tracing_disabled(true);
    let original = fake("docs", true);
    let other = fake("docs", false);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("search", "{}")]);
    let agent: Agent<()> = Agent::new("test")
        .with_mcp_server(dyn_server(&original))
        .with_mcp_server(dyn_server(&other))
        .with_model_instance(model.clone());

    let first = Runner::run(&agent, "search").await.unwrap();
    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);

    original.set_tools(Vec::new());
    other.set_tools(vec![py_mcpmem_support::fake_tool("search")]);
    model.add_turn(vec![assistant_message("done")]);

    let error = Runner::run(&agent, state).await.unwrap_err();
    assert!(error.to_string().contains("different recipient"), "{error}");
    assert!(original.tool_calls().is_empty());
    assert!(other.tool_calls().is_empty());
}

#[tokio::test]
async fn rejected_call_continues_after_the_recipient_changes() {
    // `test_rejected_mcp_call_continues_after_recipient_changes` (in_memory path).
    openai_agents_sdk::set_tracing_disabled(true);
    let original = fake("docs", true);
    let other = fake("docs", false);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("search", "{}")]);
    let agent: Agent<()> = Agent::new("test")
        .with_mcp_server(dyn_server(&original))
        .with_mcp_server(dyn_server(&other))
        .with_model_instance(model.clone());

    let first = Runner::run(&agent, "search").await.unwrap();
    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.reject(&item, false, Some("MCP request declined".into()));

    original.set_tools(Vec::new());
    other.set_tools(vec![py_mcpmem_support::fake_tool("search")]);
    model.add_turn(vec![assistant_message("done")]);

    let result = Runner::run(&agent, state).await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert!(original.tool_calls().is_empty());
    assert!(other.tool_calls().is_empty());
}

#[tokio::test]
#[ignore = "known cross-area divergence: a handoff named like an MCP tool makes get_all_function_tools fail with 'Duplicate tool names' even without include_server_in_tool_names (Python only reserves handoff names when prefixing) and the 'MCP tool call as a handoff' guard does not exist; fix lives in src/mcp/util.rs and src/run_internal"]
async fn mcp_approval_cannot_authorize_a_handoff_with_the_same_name() {
    // `test_mcp_approval_cannot_authorize_handoff` (restore=False, streaming=False).
    openai_agents_sdk::set_tracing_disabled(true);
    for approved in [true, false] {
        let server = fake("docs", true);
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call("search", "{}")]);
        let agent: Agent<()> = Agent::new("test")
            .with_mcp_server(dyn_server(&server))
            .with_model_instance(model.clone());
        let first = Runner::run(&agent, "search").await.unwrap();
        let mut state = first.to_state();
        let item = state.get_interruptions()[0].clone();
        if approved {
            state.approve(&item, false);
        } else {
            state.reject(&item, false, Some("MCP request declined".into()));
        }

        // The handoff is attached only after the interruption, under the MCP tool's name.
        let delegate: Agent<()> = Agent::new("delegate").with_model_instance(model.clone());
        let handoff_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = handoff_calls.clone();
        let handoff = openai_agents_sdk::handoffs::handoff(delegate)
            .with_tool_name("search")
            .with_on_handoff(move |_ctx| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(())
                }
            });
        let resumed: Agent<()> = Agent::new("test")
            .with_mcp_server(dyn_server(&server))
            .with_handoff(handoff)
            .with_model_instance(model.clone());
        model.add_turn(vec![assistant_message("done")]);

        let outcome = Runner::run(&resumed, state).await;
        if approved {
            let error = outcome.unwrap_err();
            assert!(
                error.to_string().contains("MCP tool call as a handoff"),
                "{error}"
            );
        } else {
            let result = outcome.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("done"));
        }
        assert!(server.tool_calls().is_empty());
        assert_eq!(handoff_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
