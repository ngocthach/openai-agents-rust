//! Parity port of `tests/mcp/test_mcp_resume_binding.py`: an approval or rejection recorded for
//! a local MCP tool call stays bound to the MCP server and raw tool that offered it, across an
//! in-memory resume and across a serialized snapshot.
//!
//! Rust idioms: `RunState::from_json` takes no agent, so "restore" is `from_json` followed by a
//! resume on the agent the test built; where Python mutates `agent.tools` / `agent.handoffs`
//! after the first run, the tests build a second agent with the same name instead.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "mcp")]

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::items::{RunInput, RunItemKind};
use openai_agents_sdk::mcp::{MCPServer, RequireApproval};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, ToolOrigin};
use openai_agents_sdk::{Agent, MCPConfig, RunContext, RunState, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{FakeMcpServer, fake_tool};

type Run = openai_agents_sdk::result::RunResult<()>;

fn server(name: &str, tools: &[&str]) -> Arc<FakeMcpServer> {
    let s = Arc::new(
        FakeMcpServer::new(name).configure(|c| c.require_approval = RequireApproval::Always),
    );
    s.set_tools(tools.iter().map(|t| fake_tool(t)).collect());
    s
}

fn script(steps: Vec<Vec<openai_agents_sdk::items::OutputItem>>) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

fn call_then_done() -> Vec<Vec<openai_agents_sdk::items::OutputItem>> {
    vec![
        vec![function_call("search", "{}")],
        vec![assistant_message("done")],
    ]
}

struct Build<'a> {
    servers: Vec<&'a Arc<FakeMcpServer>>,
    model: &'a Arc<ScriptedModel>,
    prefix: bool,
    tools: Vec<FunctionTool<()>>,
    handoffs: Vec<openai_agents_sdk::handoffs::Handoff<()>>,
}

impl<'a> Build<'a> {
    fn new(servers: Vec<&'a Arc<FakeMcpServer>>, model: &'a Arc<ScriptedModel>) -> Self {
        Build {
            servers,
            model,
            prefix: false,
            tools: Vec::new(),
            handoffs: Vec::new(),
        }
    }
    fn prefixed(mut self) -> Self {
        self.prefix = true;
        self
    }
    fn tool(mut self, tool: FunctionTool<()>) -> Self {
        self.tools.push(tool);
        self
    }
    fn handoff(mut self, h: openai_agents_sdk::handoffs::Handoff<()>) -> Self {
        self.handoffs.push(h);
        self
    }
    fn agent(self) -> Arc<Agent<()>> {
        let mut agent: Agent<()> = Agent::new("test").with_model_instance(self.model.clone());
        for s in self.servers {
            agent = agent.with_mcp_server(s.clone() as Arc<dyn MCPServer>);
        }
        for t in self.tools {
            agent = agent.with_tool(t);
        }
        for h in self.handoffs {
            agent = agent.with_handoff(h);
        }
        if self.prefix {
            let mut config = MCPConfig::<()>::default();
            config.include_server_in_tool_names = true;
            agent = agent.with_mcp_config(config);
        }
        Arc::new(agent)
    }
}

async fn drive(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    streaming: bool,
) -> Result<Run, AgentsError> {
    if streaming {
        let mut stream = Runner::run_streamed(agent.clone(), input);
        {
            let events = stream.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(stream
            .into_result()
            .expect("a finished stream has a result"))
    } else {
        Runner::run(agent.clone(), input).await
    }
}

fn restore(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

/// Rewrite a snapshot the way a pre-binding SDK wrote it: per-tool approvals folded into
/// `approvals`, schema 1.17, and no `mcp_tool_bindings`.
fn legacy_snapshot(mut snapshot: Value) -> Value {
    let context = snapshot["context"].as_object_mut().unwrap();
    if let Some(Value::Array(entries)) = context.remove("function_tool_approvals") {
        for entry in entries {
            context["approvals"][entry["tool_key"].as_str().unwrap()] = entry["decision"].clone();
        }
    }
    snapshot["$schemaVersion"] = json!("1.17");
    snapshot["last_processed_response"]
        .as_object_mut()
        .unwrap()
        .remove("mcp_tool_bindings");
    snapshot
}

fn assert_user_error(result: Result<Run, AgentsError>, needle: &str) {
    match result {
        Ok(run) => panic!("expected an error containing {needle:?}, got {run:?}"),
        Err(error) => assert!(error.to_string().contains(needle), "{error}"),
    }
}

fn rejected_output_seen(result: &Run, message: &str) -> bool {
    result.new_items.iter().any(|item| {
        matches!(&item.kind, RunItemKind::ToolCallOutput { output, .. } if output == &json!(message))
    })
}

fn local_search(calls: &Arc<Mutex<Vec<String>>>, needs_approval: bool) -> FunctionTool<()> {
    let calls = calls.clone();
    FunctionTool::new_raw(
        "search",
        "local search",
        json!({"type": "object", "properties": {"query": {"type": "string"}}, "additionalProperties": true}),
        move |_ctx, args: String| {
            let calls = calls.clone();
            async move {
                let query = serde_json::from_str::<Value>(&args)
                    .ok()
                    .and_then(|v| v["query"].as_str().map(str::to_string))
                    .unwrap_or_else(|| "search".to_string());
                calls.lock().unwrap().push(query);
                Ok::<_, AgentsError>("local result".to_string())
            }
        },
    )
    .with_needs_approval(needs_approval)
}

// ---------------------------------------------------------------------------------------------
// test_mcp_recipient_survives_absent_tool_during_restore (2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn recipient_survives_absent_tool_during_restore() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streaming in [false, true] {
        for changed_recipient in [false, true] {
            let original = server("docs", &["search"]);
            let other = server("docs", &[]);
            let model = script(call_then_done());
            let agent = Build::new(vec![&original, &other], &model).agent();
            let first = drive(&agent, "search", false).await.unwrap();
            let mut state = first.to_state();
            let item = state.get_interruptions()[0].clone();
            state.approve(&item, false);
            let snapshot = state.to_json().unwrap();
            original.set_tools(Vec::new());
            let restored = RunState::from_json(snapshot).unwrap();
            // Saving while discovery has no candidate must retain the original recipient.
            let restored = restore(&restored);
            if changed_recipient {
                other.set_tools(vec![fake_tool("search")]);
            } else {
                original.set_tools(vec![fake_tool("search")]);
            }
            let result = drive(&agent, restored, streaming).await;
            if changed_recipient {
                assert_user_error(result, "different recipient");
                assert!(original.tool_calls().is_empty() && other.tool_calls().is_empty());
            } else {
                assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
                assert_eq!(original.tool_calls(), vec!["search"]);
                assert!(other.tool_calls().is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_mcp_approval_cannot_authorize_handoff (2 x 2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn mcp_approval_cannot_authorize_handoff() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streaming in [false, true] {
        for restore_state in [false, true] {
            for approved in [false, true] {
                let handoff_calls = Arc::new(Mutex::new(Vec::<String>::new()));
                let srv = server("server", &["search"]);
                let model = script(call_then_done());
                let agent = Build::new(vec![&srv], &model).agent();
                let first = drive(&agent, "search", false).await.unwrap();
                let mut state = first.to_state();
                let item = state.get_interruptions()[0].clone();
                if approved {
                    state.approve(&item, false);
                } else {
                    state.reject(&item, false, Some("MCP request declined".into()));
                }
                let delegate: Agent<()> = Agent::new("delegate").with_model_instance(model.clone());
                let calls = handoff_calls.clone();
                let handoff = openai_agents_sdk::handoffs::handoff(delegate)
                    .with_tool_name("search")
                    .with_on_handoff(move |_ctx| {
                        let calls = calls.clone();
                        async move {
                            calls.lock().unwrap().push("handoff".into());
                            Ok(())
                        }
                    });
                let resumed = Build::new(vec![&srv], &model).handoff(handoff).agent();
                if restore_state {
                    state = restore(&state);
                }
                let result = drive(&resumed, state, streaming).await;
                if approved {
                    assert_user_error(result, "MCP tool call as a handoff");
                } else {
                    let result = result.unwrap();
                    assert_eq!(result.final_output_text().as_deref(), Some("done"));
                    assert_eq!(result.last_agent.name, "test");
                    assert!(rejected_output_seen(&result, "MCP request declined"));
                }
                assert!(srv.tool_calls().is_empty());
                assert!(handoff_calls.lock().unwrap().is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_serialized_mcp_approval_preserves_recipient (2 x 3)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn serialized_mcp_approval_preserves_recipient() {
    openai_agents_sdk::set_tracing_disabled(true);
    let arguments = r#"{"query":"synthetic document"}"#;
    for streaming in [false, true] {
        for listing_change in ["none", "before_restore", "after_restore"] {
            let trusted = server("docs", &[]);
            let other = server("docs", &[]);
            for s in [&trusted, &other] {
                s.set_tools(vec![{
                    let mut t = fake_tool("search");
                    t.input_schema = json!({"type": "object", "properties": {}});
                    t
                }]);
            }
            let model = script(vec![
                vec![function_call("mcp_docs__search_15de6fa1", arguments)],
                vec![assistant_message("done")],
            ]);
            let agent = Build::new(vec![&trusted, &other], &model)
                .prefixed()
                .agent();
            let first = drive(&agent, "search", streaming).await.unwrap();
            assert_eq!(first.interruptions.len(), 1);
            let mut state = first.to_state();
            let item = state.get_interruptions()[0].clone();
            state.approve(&item, false);
            let snapshot = state.to_json_string().unwrap();
            let payload: Value = serde_json::from_str(&snapshot).unwrap();
            assert_eq!(
                payload["$schemaVersion"],
                openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION
            );
            assert_eq!(
                payload["last_processed_response"]["mcp_tool_bindings"][item.call_id()],
                json!(["docs", "search", 0])
            );
            assert!(trusted.tool_calls().is_empty() && other.tool_calls().is_empty());

            let mut restored = (listing_change != "before_restore")
                .then(|| RunState::from_json_str(&snapshot).unwrap());
            if listing_change != "none" {
                let mut renamed = fake_tool("search_15de6fa1");
                renamed.input_schema = json!({"type": "object", "properties": {}});
                other.set_tools(vec![renamed]);
                // The original public name now belongs to the other server's new raw tool.
                let names: Vec<String> = agent
                    .get_all_tools(&RunContext::new(()))
                    .await
                    .unwrap()
                    .iter()
                    .map(|t| t.name().to_string())
                    .collect();
                assert_eq!(names, ["mcp_docs__search", "mcp_docs__search_15de6fa1"]);
                if listing_change == "before_restore" {
                    restored = Some(RunState::from_json_str(&snapshot).unwrap());
                }
                let result = drive(&agent, restored.unwrap(), streaming).await;
                assert_user_error(result, "different recipient");
                assert!(trusted.tool_calls().is_empty() && other.tool_calls().is_empty());
            } else {
                let resumed = drive(&agent, restored.unwrap(), streaming).await.unwrap();
                assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
                assert_eq!(trusted.tool_calls(), vec!["search"]);
                // The shared fake server echoes `result_<tool>` without the arguments Python's
                // helper appends, so only the raw tool name is observable here.
                assert_eq!(trusted.tool_results(), vec!["result_search"]);
                assert!(other.tool_calls().is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_unprefixed_mcp_resume_rejects_different_server_with_same_raw_tool
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn unprefixed_resume_rejects_different_server_with_same_raw_tool() {
    openai_agents_sdk::set_tracing_disabled(true);
    let first_server = server("docs", &["search"]);
    let second_server = server("docs", &[]);
    let model = script(vec![vec![function_call("search", "{}")]]);
    let agent = Build::new(vec![&first_server, &second_server], &model).agent();
    let result = drive(&agent, "search", false).await.unwrap();
    let mut state = result.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let snapshot = state.to_json().unwrap();
    first_server.set_tools(Vec::new());
    second_server.set_tools(vec![fake_tool("search")]);
    let restored = RunState::from_json(snapshot).unwrap();
    assert_user_error(drive(&agent, restored, false).await, "different recipient");
    assert!(first_server.tool_calls().is_empty() && second_server.tool_calls().is_empty());
}

// ---------------------------------------------------------------------------------------------
// test_legacy_pending_mcp_call_requires_new_run
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn legacy_pending_mcp_call_requires_new_run() {
    openai_agents_sdk::set_tracing_disabled(true);
    let srv = server("server", &["search"]);
    let model = script(vec![vec![function_call("search", "{}")]]);
    let agent = Build::new(vec![&srv], &model).agent();
    let result = drive(&agent, "search", false).await.unwrap();
    let mut state = result.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let snapshot = legacy_snapshot(state.to_json().unwrap());
    let restored = RunState::from_json(snapshot).unwrap();
    assert_user_error(
        drive(&agent, restored, false).await,
        "missing or different recipient binding",
    );
    assert!(srv.tool_calls().is_empty());
}

// ---------------------------------------------------------------------------------------------
// test_mcp_resume_rejects_different_raw_tool_on_same_server
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn resume_rejects_different_raw_tool_on_same_server() {
    openai_agents_sdk::set_tracing_disabled(true);
    let srv = server("docs", &["search!", "search?"]);
    let model = Arc::new(ScriptedModel::new());
    let agent = Build::new(vec![&srv], &model).prefixed().agent();
    let original_tools = agent.get_all_tools(&RunContext::new(())).await.unwrap();
    let public_name = original_tools[0].name().to_string();
    model
        .enqueue(ScriptedStep::output(vec![function_call(
            &public_name,
            "{}",
        )]))
        .unwrap();
    let result = drive(&agent, "search", false).await.unwrap();
    let mut state = result.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let snapshot = state.to_json().unwrap();
    srv.set_tools(vec![
        fake_tool("search!"),
        fake_tool(public_name.strip_prefix("mcp_docs__").unwrap()),
    ]);
    let current = agent.get_all_tools(&RunContext::new(())).await.unwrap();
    assert_eq!(current[1].name(), public_name);
    let restored = RunState::from_json(snapshot).unwrap();
    assert_user_error(drive(&agent, restored, false).await, "different recipient");
    assert!(srv.tool_calls().is_empty());
}

// ---------------------------------------------------------------------------------------------
// test_completed_mcp_sibling_does_not_block_function_approval (1.14, 1.17, 1.18, 1.19)
// ---------------------------------------------------------------------------------------------

fn schema_tuple(version: &str) -> (u32, u32) {
    let (major, minor) = version.split_once('.').unwrap();
    (major.parse().unwrap(), minor.parse().unwrap())
}

#[tokio::test]
async fn completed_mcp_sibling_does_not_block_function_approval() {
    openai_agents_sdk::set_tracing_disabled(true);
    for schema_version in ["1.14", "1.17", "1.18", "1.19"] {
        let function_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let calls = function_calls.clone();
        let gated = FunctionTool::<()>::new_raw(
            "gated",
            "gated",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args: String| {
                let calls = calls.clone();
                async move {
                    calls.lock().unwrap().push("gated".into());
                    Ok::<_, AgentsError>("ok".to_string())
                }
            },
        )
        .with_needs_approval(true);
        let srv = Arc::new(FakeMcpServer::new("server"));
        srv.set_tools(vec![fake_tool("search")]);
        let model = script(vec![
            vec![
                function_call_with_id("search", "{}", "completed_mcp"),
                function_call_with_id("gated", "{}", "pending_function"),
            ],
            vec![assistant_message("done")],
        ]);
        let agent = Build::new(vec![&srv], &model).tool(gated).agent();
        let result = drive(&agent, "search and run gated", false).await.unwrap();
        assert_eq!(srv.tool_calls(), vec!["search"]);
        let names: Vec<String> = result
            .interruptions
            .iter()
            .map(|i| i.tool_name.clone())
            .collect();
        assert_eq!(names, ["gated"]);
        let mut state = result.to_state();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        let mut snapshot = state.to_json().unwrap();
        if schema_tuple(schema_version) < (1, 18) {
            snapshot = legacy_snapshot(snapshot);
        }
        snapshot["$schemaVersion"] = json!(schema_version);
        if schema_version == "1.14" {
            snapshot["context"]
                .as_object_mut()
                .unwrap()
                .remove("tool_invocations");
        }
        let restored = RunState::from_json(snapshot).unwrap();
        let resumed = drive(&agent, restored, false).await.unwrap();
        assert_eq!(
            resumed.final_output_text().as_deref(),
            Some("done"),
            "{schema_version}"
        );
        assert_eq!(srv.tool_calls(), vec!["search"]);
        assert_eq!(*function_calls.lock().unwrap(), vec!["gated"]);
    }
}

// ---------------------------------------------------------------------------------------------
// test_legacy_mcp_call_missing_during_restore_cannot_rebind (2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn legacy_mcp_call_missing_during_restore_cannot_rebind() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streaming in [false, true] {
        let original = server("docs", &["search"]);
        let other = server("docs", &[]);
        let model = script(vec![
            vec![function_call("search", r#"{"query":"synthetic document"}"#)],
            vec![assistant_message("done")],
        ]);
        let agent = Build::new(vec![&original, &other], &model).agent();
        let result = drive(&agent, "search", false).await.unwrap();
        let mut state = result.to_state();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        let snapshot = legacy_snapshot(state.to_json().unwrap());
        original.set_tools(Vec::new());
        let restored = RunState::from_json(snapshot).unwrap();
        other.set_tools(vec![fake_tool("search")]);
        assert_user_error(
            drive(&agent, restored, streaming).await,
            "missing or different recipient binding",
        );
        assert!(original.tool_calls().is_empty() && other.tool_calls().is_empty());
    }
}

// ---------------------------------------------------------------------------------------------
// test_rejected_mcp_call_continues_after_recipient_changes (2 x 3)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn rejected_mcp_call_continues_after_recipient_changes() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streaming in [false, true] {
        for resume_path in ["in_memory", "before_restore", "after_restore"] {
            let original = server("docs", &["search"]);
            let other = server("docs", &[]);
            let model = script(call_then_done());
            let agent = Build::new(vec![&original, &other], &model).agent();
            let first = drive(&agent, "search", false).await.unwrap();
            let mut state = first.to_state();
            let item = state.get_interruptions()[0].clone();
            state.reject(&item, false, Some("MCP request declined".into()));
            let snapshot = state.to_json().unwrap();
            if resume_path == "after_restore" {
                state = RunState::from_json(snapshot.clone()).unwrap();
            }
            original.set_tools(Vec::new());
            other.set_tools(vec![fake_tool("search")]);
            if resume_path == "before_restore" {
                state = RunState::from_json(snapshot).unwrap();
            }
            let result = drive(&agent, state, streaming).await.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("done"));
            assert!(
                rejected_output_seen(&result, "MCP request declined"),
                "{resume_path}"
            );
            assert!(original.tool_calls().is_empty() && other.tool_calls().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_restored_rejection_retains_recipient_when_later_approved (2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn restored_rejection_retains_recipient_when_later_approved() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streaming in [false, true] {
        let original = server("docs", &["search"]);
        let other = server("docs", &[]);
        let model = script(vec![vec![function_call("search", "{}")]]);
        let agent = Build::new(vec![&original, &other], &model).agent();
        let first = drive(&agent, "search", false).await.unwrap();
        let mut state = first.to_state();
        let item = state.get_interruptions()[0].clone();
        state.reject(&item, false, None);
        let snapshot = state.to_json().unwrap();
        original.set_tools(Vec::new());
        other.set_tools(vec![fake_tool("search")]);
        let restored = RunState::from_json(snapshot).unwrap();
        // A second snapshot must still identify the original server, even after discovery.
        let mut restored = restore(&restored);
        restored.approve(&item, false);
        assert_user_error(
            drive(&agent, restored, streaming).await,
            "different recipient",
        );
        assert!(original.tool_calls().is_empty() && other.tool_calls().is_empty());
    }
}

// ---------------------------------------------------------------------------------------------
// test_mcp_approval_cannot_authorize_local_replacement (2 x 2 x 3)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn mcp_approval_cannot_authorize_local_replacement() {
    openai_agents_sdk::set_tracing_disabled(true);
    let call = || function_call("search", r#"{"query":"synthetic input"}"#);
    for streaming in [false, true] {
        for restore_state in [false, true] {
            for policy in ["always", "conditional", "never"] {
                let local_calls = Arc::new(Mutex::new(Vec::<String>::new()));
                let approval_checks = Arc::new(Mutex::new(Vec::<Value>::new()));
                let make_local = || {
                    let tool = local_search(&local_calls, policy == "always");
                    if policy == "conditional" {
                        let checks = approval_checks.clone();
                        tool.with_needs_approval_fn(move |_ctx, args, _id| {
                            let checks = checks.clone();
                            async move {
                                checks.lock().unwrap().push(args);
                                true
                            }
                        })
                    } else {
                        tool
                    }
                };
                let srv = server("server", &["search"]);
                let model = script(vec![vec![call()], vec![assistant_message("done")]]);
                let agent = Build::new(vec![&srv], &model).agent();
                let first = drive(&agent, "search", false).await.unwrap();
                let mut state = first.to_state();
                let item = state.get_interruptions()[0].clone();
                state.approve(&item, false);
                let with_local = Build::new(vec![&srv], &model).tool(make_local()).agent();
                if restore_state {
                    state = restore(&state);
                }
                assert_user_error(
                    drive(&with_local, state, streaming).await,
                    "different recipient",
                );
                assert!(local_calls.lock().unwrap().is_empty(), "{policy}");
                assert!(approval_checks.lock().unwrap().is_empty(), "{policy}");
                assert!(srv.tool_calls().is_empty());

                // A new run selects the intended local recipient and evaluates its own policy.
                let fresh_model = script(vec![vec![call()], vec![assistant_message("done")]]);
                let fresh_agent = Build::new(vec![&srv], &fresh_model)
                    .tool(make_local())
                    .agent();
                let mut fresh = drive(&fresh_agent, "search", false).await.unwrap();
                if policy != "never" {
                    assert_eq!(fresh.interruptions.len(), 1, "{policy}");
                    assert!(local_calls.lock().unwrap().is_empty());
                    let mut fresh_state = fresh.to_state();
                    let pending = fresh_state.get_interruptions()[0].clone();
                    fresh_state.approve(&pending, false);
                    fresh = drive(&fresh_agent, fresh_state, false).await.unwrap();
                }
                assert_eq!(
                    fresh.final_output_text().as_deref(),
                    Some("done"),
                    "{policy}"
                );
                assert_eq!(*local_calls.lock().unwrap(), vec!["synthetic input"]);
                let expected: Vec<Value> = if policy == "conditional" {
                    vec![json!({"query": "synthetic input"})]
                } else {
                    Vec::new()
                };
                assert_eq!(*approval_checks.lock().unwrap(), expected, "{policy}");
                assert!(srv.tool_calls().is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_rejected_mcp_call_does_not_invoke_local_replacement (2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn rejected_mcp_call_does_not_invoke_local_replacement() {
    openai_agents_sdk::set_tracing_disabled(true);
    let expected_origin = ToolOrigin::mcp("docs");
    for streaming in [false, true] {
        for restore_state in [false, true] {
            let local_calls = Arc::new(Mutex::new(Vec::<String>::new()));
            let srv = server("docs", &["search"]);
            let model = script(call_then_done());
            let agent = Build::new(vec![&srv], &model).agent();
            let first = drive(&agent, "search", false).await.unwrap();
            assert_eq!(
                first.interruptions[0].tool_origin,
                Some(expected_origin.clone())
            );
            let mut state = first.to_state();
            let item = state.get_interruptions()[0].clone();
            state.reject(&item, false, Some("MCP request declined".into()));
            let with_local = Build::new(vec![&srv], &model)
                .tool(local_search(&local_calls, true))
                .agent();
            if restore_state {
                state = restore(&state);
            }
            let result = drive(&with_local, state, streaming).await.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("done"));
            let calls: Vec<_> = result
                .new_items
                .iter()
                .filter(|i| matches!(i.kind, RunItemKind::ToolCall { .. }))
                .collect();
            let outputs: Vec<_> = result
                .new_items
                .iter()
                .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
                .collect();
            assert_eq!((calls.len(), outputs.len()), (1, 1));
            assert_eq!(calls[0].tool_origin, Some(expected_origin.clone()));
            assert_eq!(outputs[0].tool_origin, Some(expected_origin.clone()));
            assert!(rejected_output_seen(&result, "MCP request declined"));
            let snapshot = result.to_state().to_json().unwrap();
            let serialized: Vec<&Value> = snapshot["generated_items"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|i| i["type"] == "tool_call_output_item")
                .collect();
            assert_eq!(serialized.len(), 1);
            assert_eq!(
                serialized[0]["tool_origin"],
                json!({"type": "mcp", "mcp_server_name": "docs"})
            );
            let restored = RunState::from_json(snapshot).unwrap();
            let restored_outputs: Vec<_> = restored
                .generated_items
                .iter()
                .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
                .collect();
            assert_eq!(restored_outputs.len(), 1);
            assert_eq!(
                restored_outputs[0].tool_origin,
                Some(expected_origin.clone())
            );
            assert!(local_calls.lock().unwrap().is_empty() && srv.tool_calls().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_legacy_approval_cannot_authorize_local_replacement_of_mcp (2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn legacy_approval_cannot_authorize_local_replacement_of_mcp() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streaming in [false, true] {
        for original_mcp in [false, true] {
            let local_calls = Arc::new(Mutex::new(Vec::<String>::new()));
            let srv = server("docs", &["search"]);
            let model = script(call_then_done());
            let first_agent = if original_mcp {
                Build::new(vec![&srv], &model).agent()
            } else {
                Build::new(vec![], &model)
                    .tool(local_search(&local_calls, true))
                    .agent()
            };
            let first = drive(&first_agent, "search", false).await.unwrap();
            let mut state = first.to_state();
            let item = state.get_interruptions()[0].clone();
            state.approve(&item, false);
            let snapshot = legacy_snapshot(state.to_json().unwrap());
            let servers = if original_mcp { vec![&srv] } else { vec![] };
            let resumed_agent = Build::new(servers, &model)
                .tool(local_search(&local_calls, true))
                .agent();
            let restored = RunState::from_json(snapshot).unwrap();
            let result = drive(&resumed_agent, restored, streaming).await;
            if original_mcp {
                assert_user_error(result, "missing or different recipient binding");
                assert!(local_calls.lock().unwrap().is_empty());
            } else {
                assert_eq!(result.unwrap().final_output_text().as_deref(), Some("done"));
                assert_eq!(local_calls.lock().unwrap().len(), 1);
            }
            assert!(srv.tool_calls().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_removed_local_override_preserves_original_mcp_recipient (2 x 2)
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn removed_local_override_preserves_original_mcp_recipient() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streaming in [false, true] {
        for round_trip in [false, true] {
            let local_calls = Arc::new(Mutex::new(Vec::<String>::new()));
            let srv = server("docs", &["search"]);
            let model = script(call_then_done());
            let agent = Build::new(vec![&srv], &model).agent();
            let first = drive(&agent, "search", false).await.unwrap();
            let mut state = first.to_state();
            let item = state.get_interruptions()[0].clone();
            state.approve(&item, false);
            // Restoring while a local override exists must not rebind the call to it.
            let _with_local = Build::new(vec![&srv], &model)
                .tool(local_search(&local_calls, false))
                .agent();
            let mut restored = restore(&state);
            if round_trip {
                restored = restore(&restored);
            }
            let result = drive(&agent, restored, streaming).await.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("done"));
            assert_eq!(srv.tool_calls(), vec!["search"]);
            assert!(local_calls.lock().unwrap().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// test_cancelled_rejection_preserves_recipient_for_later_approval (2)
//
// Python cancels a task that is resuming a shared, mutable RunState and then reads the state.
// Rust moves the state into the run, so the analogue aborts a run over a copy and checks that
// the snapshot taken before it still names the original recipient for a later approval.
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn cancelled_rejection_preserves_recipient_for_later_approval() {
    openai_agents_sdk::set_tracing_disabled(true);
    for local_override in [false, true] {
        let local_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let original = server("docs", &["search"]);
        let other = server("docs", &[]);
        let model = script(call_then_done());
        let agent = Build::new(vec![&original, &other], &model).agent();
        let first = drive(&agent, "search", false).await.unwrap();
        let mut state = first.to_state();
        let approval = state.get_interruptions()[0].clone();
        state.reject(&approval, false, None);
        original.set_tools(Vec::new());
        other.set_tools(vec![fake_tool("search")]);
        let run_agent = if local_override {
            Build::new(vec![&original, &other], &model)
                .tool(local_search(&local_calls, false))
                .agent()
        } else {
            agent.clone()
        };
        let state = restore(&state);
        let task = {
            let (run_agent, state) = (run_agent.clone(), state.clone());
            tokio::spawn(async move { Runner::run(run_agent, state).await.map(|_| ()) })
        };
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;

        let snapshot = state.to_json().unwrap();
        assert_eq!(
            snapshot["last_processed_response"]["mcp_tool_bindings"][approval.call_id()],
            json!(["docs", "search", 0])
        );
        let mut restored = RunState::from_json(snapshot).unwrap();
        restored.approve(&approval, false);
        let bare = Build::new(vec![&original, &other], &model).agent();
        assert_user_error(drive(&bare, restored, false).await, "different recipient");
        assert!(original.tool_calls().is_empty() && other.tool_calls().is_empty());
        assert!(local_calls.lock().unwrap().is_empty());
    }
}
