#![cfg(feature = "mcp")]
//! Port of the agent-as-tool and handoff-snapshot cases of Python's
//! `tests/test_tool_name_collision_policy.py`.
//!
//! Python's nested-state cases edit the interrupted outer run's private fields. Here the outer
//! run is resumed through the public API, and the nested state is restored from its JSON
//! snapshot where Python round-trips it.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool, RequireApproval};
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, Result, Runner};
use serde_json::{Map, Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn record_tool(
    name: &str,
    needs_approval: bool,
    calls: &Arc<Mutex<Vec<String>>>,
    which: &'static str,
) -> FunctionTool<()> {
    let calls = calls.clone();
    FunctionTool::new(name, "", move |_ctx, _args: Value| {
        let calls = calls.clone();
        async move {
            calls.lock().unwrap().push(which.to_string());
            Ok::<_, AgentsError>(which.to_string())
        }
    })
    .with_needs_approval(needs_approval)
}

fn restore(value: Value) -> RunState {
    RunState::from_json(value).unwrap()
}

/// Approve every interruption of `state`.
fn approve_all(state: &mut RunState) {
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
}

/// Echo MCP server exposing one `search` tool that never needs approval.
#[derive(Debug)]
struct SearchServer {
    config: MCPServerConfig,
}

#[async_trait]
impl MCPServer for SearchServer {
    fn name(&self) -> &str {
        "fake_mcp_server"
    }
    fn config(&self) -> &MCPServerConfig {
        &self.config
    }
    async fn connect(&self) -> Result<()> {
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        Ok(())
    }
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Ok(vec![MCPTool {
            meta: None,
            name: "search".into(),
            title: None,
            description: None,
            input_schema: json!({"type": "object", "properties": {}}),
            annotations: None,
        }])
    }
    async fn call_tool(&self, name: &str, _a: Option<Map<String, Value>>) -> Result<Value> {
        Ok(json!({"content": [{"type": "text", "text": format!("result_{name}")}]}))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(vec![])
    }
    async fn get_prompt(
        &self,
        _n: &str,
        _a: Option<std::collections::BTreeMap<String, String>>,
    ) -> Result<Value> {
        Ok(Value::Null)
    }
}

/// Python: `test_missing_interrupted_agent_tool_preserves_nested_state_for_retry`.
#[tokio::test]
async fn missing_interrupted_agent_tool_preserves_nested_state_for_retry() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let inner_model = setup();
    inner_model.add_turn(vec![
        function_call_with_id("before_pause", "{}", "before_call"),
        function_call_with_id("sensitive", "{}", "sensitive_call"),
    ]);
    inner_model.add_turn(vec![assistant_message("inner done")]);
    let inner = Agent::<()>::new("inner")
        .with_model_instance(inner_model)
        .with_tool(record_tool("before_pause", false, &calls, "before"))
        .with_tool(record_tool("sensitive", true, &calls, "sensitive"));
    let nested_tool = inner.as_tool("lookup", "Look up a value with the inner agent.");
    let outer_model = setup();
    outer_model.add_turn(vec![function_call_with_id(
        "lookup",
        r#"{"input":"hi"}"#,
        "outer_call",
    )]);
    outer_model.add_turn(vec![assistant_message("outer done")]);
    let outer_with = Agent::<()>::new("outer")
        .with_model_instance(outer_model.clone())
        .with_tool(nested_tool.clone());

    let initial = Runner::run(&outer_with, "Look this up").await.unwrap();
    let mut state = restore(initial.to_state().to_json().unwrap());
    approve_all(&mut state);
    assert_eq!(*calls.lock().unwrap(), vec!["before"]);

    let outer_without = Agent::<()>::new("outer").with_model_instance(outer_model.clone());
    let err = Runner::run(&outer_without, state.clone())
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Tool lookup not found in agent outer"),
        "{err}"
    );

    let resumed = Runner::run(&outer_with, state).await.unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("outer done"));
    assert_eq!(*calls.lock().unwrap(), vec!["before", "sensitive"]);
}

/// Python: `test_replacing_interrupted_agent_tool_fails_before_side_effects`.
#[tokio::test]
async fn replacing_interrupted_agent_tool_fails_before_side_effects() {
    let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let inner = Agent::<()>::new("inner")
        .with_model_instance({
            let m = setup();
            m.add_turn(vec![function_call_with_id(
                "sensitive",
                "{}",
                "call_sensitive",
            )]);
            m
        })
        .with_tool(
            FunctionTool::new("sensitive", "", |_ctx, _args: Value| async {
                Ok::<_, AgentsError>("sensitive".to_string())
            })
            .with_needs_approval(true),
        );
    let nested_tool = inner.as_tool("lookup", "Look up a value with the inner agent.");
    let outer_model = setup();
    outer_model.add_turn(vec![function_call_with_id(
        "lookup",
        r#"{"input":"hi"}"#,
        "call_lookup",
    )]);
    let outer = Agent::<()>::new("outer")
        .with_model_instance(outer_model.clone())
        .with_tool(nested_tool);

    let initial = Runner::run(&outer, "Look this up").await.unwrap();
    let mut state = restore(initial.to_state().to_json().unwrap());
    approve_all(&mut state);

    // A local tool now answers to the name `lookup` that the queued nested call uses.
    let replaced = Agent::<()>::new("outer")
        .with_model_instance(outer_model.clone())
        .with_tool(FunctionTool::new("lookup", "", {
            let calls = calls.clone();
            move |_ctx, args: Value| {
                let calls = calls.clone();
                async move {
                    calls.lock().unwrap().push(args["input"].to_string());
                    Ok::<_, AgentsError>("local".to_string())
                }
            }
        }));
    let err = Runner::run(&replaced, state).await.unwrap_err();

    assert!(
        err.to_string()
            .contains("Cannot reconcile queued tool lookup with a new tool"),
        "{err}"
    );
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_preserves_multiple_agent_tool_calls`.
#[tokio::test]
async fn resume_preserves_multiple_agent_tool_calls() {
    let inner_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let inner_model = setup();
    inner_model.add_turns(vec![
        vec![function_call_with_id("inner_hitl_tool", "{}", "inner-1")],
        vec![function_call_with_id("inner_hitl_tool", "{}", "inner-2")],
        vec![assistant_message("inner done")],
        vec![assistant_message("inner done")],
    ]);
    let inner_hitl_tool = {
        let inner_calls = inner_calls.clone();
        FunctionTool::new("inner_hitl_tool", "", move |_ctx, _args: Value| {
            let inner_calls = inner_calls.clone();
            async move {
                inner_calls.lock().unwrap().push("inner".into());
                Ok::<_, AgentsError>("ok".to_string())
            }
        })
        .with_needs_approval(true)
    };
    let inner = Agent::<()>::new("inner")
        .with_model_instance(inner_model)
        .with_tool(inner_hitl_tool);
    let agent_tool = inner.as_tool_with(
        openai_agents_sdk::agent::AgentAsToolOptions::new()
            .tool_name("inner_agent_tool")
            .tool_description("Run the inner agent.")
            .needs_approval(false),
    );
    let outer_model = setup();
    outer_model.add_turn(vec![
        function_call_with_id("inner_agent_tool", r#"{"input":"a"}"#, "outer-a"),
        function_call_with_id("inner_agent_tool", r#"{"input":"b"}"#, "outer-b"),
    ]);
    outer_model.add_turn(vec![assistant_message("done")]);
    let outer = Agent::<()>::new("outer")
        .with_model_instance(outer_model)
        .with_tool(agent_tool);
    let initial = Runner::run(&outer, "start").await.unwrap();
    let mut state = initial.to_state();
    approve_all(&mut state);

    let resumed = Runner::run(&outer, state).await.unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*inner_calls.lock().unwrap(), vec!["inner", "inner"]);
    let outer_outputs = resumed
        .new_items
        .iter()
        .filter(|item| match &item.kind {
            openai_agents_sdk::items::RunItemKind::ToolCallOutput { raw_item, .. } => {
                raw_item["type"] == "function_call_output"
                    && matches!(raw_item["call_id"].as_str(), Some("outer-a" | "outer-b"))
            }
            _ => false,
        })
        .count();
    assert_eq!(outer_outputs, 2);
}

/// Python: `test_resume_reuses_handoff_snapshot_for_delegating_overrides`, the `get_mcp_tools`
/// variant. Python's `get_all_tools` subclass variant has no Rust counterpart (see the report).
#[tokio::test]
async fn resume_reuses_handoff_snapshot_for_mcp_agents() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let resume_phase = Arc::new(Mutex::new(false));
    let enablement_checks: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let server = Arc::new(SearchServer {
        config: MCPServerConfig {
            require_approval: RequireApproval::Never,
            ..Default::default()
        },
    });

    let is_enabled = {
        let resume_phase = resume_phase.clone();
        let enablement_checks = enablement_checks.clone();
        move |_ctx: openai_agents_sdk::run_context::RunContext<()>, _agent: String| {
            let resume_phase = resume_phase.clone();
            let enablement_checks = enablement_checks.clone();
            async move {
                if !*resume_phase.lock().unwrap() {
                    return true;
                }
                let mut checks = enablement_checks.lock().unwrap();
                let enabled = checks.is_empty();
                checks.push(enabled);
                enabled
            }
        }
    };
    let target = Agent::<()>::new("target").with_model_instance({
        let m = setup();
        m.add_turn(vec![assistant_message("done")]);
        m
    });
    let route_handoff = {
        let calls = calls.clone();
        handoff(Arc::new(target))
            .with_tool_name("route")
            .with_on_handoff(move |_ctx| {
                let calls = calls.clone();
                async move {
                    calls.lock().unwrap().push("handoff".into());
                    Ok(())
                }
            })
            .with_is_enabled_fn(is_enabled)
    };
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("approval_tool", "{}", "approval_call"),
        function_call_with_id("route", "{}", "handoff_call"),
    ]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model)
        .with_tool(record_tool("approval_tool", true, &calls, "approved"))
        .with_handoff(route_handoff)
        .with_mcp_server(server);

    let initial = Runner::run(&agent, "Route this request").await.unwrap();
    let mut state = initial.to_state();
    approve_all(&mut state);
    *resume_phase.lock().unwrap() = true;

    let resumed = Runner::run(&agent, state).await.unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*enablement_checks.lock().unwrap(), vec![true]);
    assert_eq!(*calls.lock().unwrap(), vec!["approved", "handoff"]);
}

/// Python: `test_nested_rebind_is_not_committed_before_later_strict_missing_error`.
#[tokio::test]
async fn nested_rebind_is_not_committed_before_later_strict_missing_error() {
    let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let inner_model = setup();
    inner_model.add_turn(vec![
        function_call_with_id("before", "{}", "before_call"),
        function_call_with_id("sensitive", "{}", "sensitive_call"),
    ]);
    inner_model.add_turn(vec![assistant_message("inner done")]);
    let inner = Agent::<()>::new("inner")
        .with_model_instance(inner_model)
        .with_tool(record_tool("before", false, &calls, "before"))
        .with_tool(record_tool("sensitive", true, &calls, "sensitive"));
    let nested_tool = inner.as_tool("nested", "Run the inner agent.");
    let outer_model = setup();
    outer_model.add_turn(vec![
        function_call_with_id("nested", r#"{"input":"go"}"#, "nested_call"),
        function_call_with_id("missing", "{}", "missing_call"),
    ]);
    outer_model.add_turn(vec![assistant_message("outer done")]);
    let missing_tool = record_tool("missing", true, &calls, "missing");

    let outer_full = Agent::<()>::new("outer")
        .with_model_instance(outer_model.clone())
        .with_tool(nested_tool.clone())
        .with_tool(missing_tool.clone());
    let initial = Runner::run(&outer_full, "Start").await.unwrap();
    assert_eq!(*calls.lock().unwrap(), vec!["before"]);
    let mut state = initial.to_state();
    approve_all(&mut state);

    let outer_nested_only = Agent::<()>::new("outer")
        .with_model_instance(outer_model.clone())
        .with_tool(nested_tool.clone());
    let err = Runner::run(&outer_nested_only, state.clone())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Tool missing not found"), "{err}");

    let restored = restore(state.to_json().unwrap());
    let resumed = Runner::run(&outer_full, restored).await.unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("outer done"));
    let mut seen = calls.lock().unwrap().clone();
    assert_eq!(seen[0], "before");
    seen.remove(0);
    seen.sort();
    assert_eq!(seen, vec!["missing", "sensitive"]);
}
