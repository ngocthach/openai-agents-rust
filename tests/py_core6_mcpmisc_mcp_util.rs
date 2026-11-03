//! Parity ports of MCP cases that earlier rounds could not port for missing seams:
//! `FunctionTool::mcp_annotations` (`test_mcp_util.py`, `test_runner_calls_mcp.py`), per-server
//! guardrail isolation and disabled-handoff names (`test_mcp_util.py`), and an approval policy
//! that sees the run context (`test_mcp_approval.py`).
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "mcp")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::mcp::util::{get_all_function_tools, to_function_tool};
use openai_agents_sdk::mcp::{MCPServer, MCPTool, MCPToolPolicy, RequireApproval, ToolFilter};
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{Agent, MCPConfig, RunContext, RunOptions, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::FakeMcpServer;

fn plain_server(name: &str) -> Arc<FakeMcpServer> {
    Arc::new(FakeMcpServer::new(name))
}

fn mcp_tool(name: &str, annotations: Option<Value>) -> MCPTool {
    MCPTool {
        meta: None,
        name: name.to_string(),
        title: None,
        description: None,
        input_schema: json!({}),
        annotations,
    }
}

// test_mcp_annotations_are_isolated_discovery_metadata
#[tokio::test]
async fn mcp_annotations_are_isolated_discovery_metadata() {
    let server = plain_server("fake_mcp_server");
    let mut tool = mcp_tool("search", Some(json!({"readOnlyHint": true})));
    let function_tool: FunctionTool<()> = to_function_tool(
        server.clone() as Arc<dyn MCPServer>,
        &tool,
        "search".into(),
        false,
        "agent",
    )
    .await;
    let mut annotations = function_tool.mcp_annotations().expect("annotations");
    assert_eq!(annotations, json!({"readOnlyHint": true}));

    // Consumers and later discovery updates cannot rewrite the converted tool's snapshot.
    annotations["title"] = json!("consumer edit");
    tool.annotations.as_mut().unwrap()["title"] = json!("server edit");
    assert_eq!(
        function_tool.mcp_annotations(),
        Some(json!({"readOnlyHint": true}))
    );
    assert!(
        !function_tool
            .evaluate_needs_approval(&RunContext::new(()), "{}", "call")
            .await
    );
    assert_eq!(
        function_tool.tool_origin().to_json_dict(),
        json!({"type": "mcp", "mcp_server_name": "fake_mcp_server"})
    );
}

// test_non_mcp_function_tool_has_no_mcp_annotations
#[test]
fn non_mcp_function_tool_has_no_mcp_annotations() {
    let ordinary =
        FunctionTool::<()>::new_raw("ordinary", "", json!({}), |_ctx, _args: String| async {
            Ok::<_, openai_agents_sdk::error::AgentsError>("done".to_string())
        });
    assert_eq!(ordinary.mcp_annotations(), None);
}

// test_runner_hooks_can_read_mcp_annotations (streaming x 2)
#[derive(Default)]
struct AnnotationHooks {
    observed: Mutex<Vec<(String, String, Option<Value>)>>,
}

impl AnnotationHooks {
    fn record(&self, event: &str, tool: &Tool<()>) {
        let Tool::Function(function) = tool else {
            panic!("expected a function tool");
        };
        self.observed.lock().unwrap().push((
            event.to_string(),
            function.name.clone(),
            function.mcp_annotations(),
        ));
    }
}

#[async_trait]
impl RunHooks<()> for AnnotationHooks {
    async fn on_tool_start(&self, _ctx: &ToolContext<()>, _agent: &Agent<()>, tool: &Tool<()>) {
        self.record("start", tool);
    }
    async fn on_tool_end(
        &self,
        _ctx: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &ToolOutput,
    ) {
        self.record("end", tool);
    }
}

#[tokio::test]
async fn runner_hooks_can_read_mcp_annotations() {
    openai_agents_sdk::set_tracing_disabled(true);
    let hints = json!({
        "readOnlyHint": false, "destructiveHint": true,
        "idempotentHint": false, "openWorldHint": false,
    });
    for streaming in [false, true] {
        let server = plain_server("fake_mcp_server");
        server.set_tools(vec![
            mcp_tool("write_record", Some(hints.clone())),
            mcp_tool("unclassified", None),
        ]);
        let hooks = Arc::new(AnnotationHooks::default());
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![function_call("write_record", "{}")]);
        model.add_turn(vec![function_call_with_id(
            "unclassified",
            "{}",
            "unclassified_call",
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("test")
            .with_model_instance(model)
            .with_mcp_server(server.clone() as Arc<dyn MCPServer>);
        let options = RunOptions::new(()).hooks(hooks.clone());
        let result = if streaming {
            let mut stream = Runner::run_streamed_with_options(&agent, "user_message", options);
            {
                let events = stream.stream_events();
                futures::pin_mut!(events);
                while let Some(event) = events.next().await {
                    event.unwrap();
                }
            }
            stream.into_result().unwrap()
        } else {
            Runner::run_with_options(&agent, "user_message", options)
                .await
                .unwrap()
        };
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(server.tool_calls(), ["write_record", "unclassified"]);
        let observed = hooks.observed.lock().unwrap().clone();
        let expected = vec![
            (
                "start".to_string(),
                "write_record".to_string(),
                Some(hints.clone()),
            ),
            (
                "end".to_string(),
                "write_record".to_string(),
                Some(hints.clone()),
            ),
            ("start".to_string(), "unclassified".to_string(), None),
            ("end".to_string(), "unclassified".to_string(), None),
        ];
        assert_eq!(observed, expected, "streaming={streaming}");
    }
}

// test_mcp_server_guardrails_do_not_leak_across_servers_or_filtered_tools
#[tokio::test]
async fn mcp_server_guardrails_do_not_leak_across_servers_or_filtered_tools() {
    let guardrail = ToolInputGuardrail::<()>::new("allow", |_data| async {
        Ok(ToolGuardrailFunctionOutput::allow())
    });
    let policy = MCPToolPolicy::<()> {
        tool_input_guardrails: vec![guardrail],
        ..Default::default()
    };
    let guarded = Arc::new(
        FakeMcpServer::new("guarded_server")
            .configure(move |config| {
                config.tool_filter = ToolFilter::Static {
                    allowed_tool_names: Some(vec!["guarded".into()]),
                    blocked_tool_names: None,
                };
                *config = std::mem::take(config).with_tool_policy(policy);
            })
            .add_tool("guarded", json!({}))
            .add_tool("filtered_out", json!({})),
    );
    let unguarded =
        Arc::new(FakeMcpServer::new("unguarded_server").add_tool("unguarded", json!({})));
    let servers: Vec<Arc<dyn MCPServer>> = vec![guarded, unguarded];
    let tools = get_all_function_tools(
        &servers,
        &MCPConfig::<()>::default(),
        &RunContext::new(()),
        "test_agent",
    )
    .await
    .unwrap();
    let names: Vec<String> = tools.iter().map(Tool::name).collect();
    assert_eq!(names, ["guarded", "unguarded"]);
    let (Tool::Function(guarded), Tool::Function(unguarded)) = (&tools[0], &tools[1]) else {
        panic!("expected function tools");
    };
    assert_eq!(guarded.tool_input_guardrails.len(), 1);
    assert!(guarded.tool_output_guardrails.is_empty());
    assert!(unguarded.tool_input_guardrails.is_empty());
    assert!(unguarded.tool_output_guardrails.is_empty());
}

// test_agent_get_mcp_tools_ignores_disabled_handoff_tool_names_when_prefixing
#[tokio::test]
async fn agent_ignores_disabled_handoff_tool_names_when_prefixing() {
    let server = Arc::new(FakeMcpServer::new("calendar").add_tool("search", json!({})));
    let handoff_agent = Agent::<()>::new("calendar_agent");
    let agent = Arc::new(
        Agent::<()>::new("test_agent")
            .with_handoff(
                openai_agents_sdk::handoffs::handoff(handoff_agent)
                    .with_tool_name("mcp_calendar__search")
                    .with_is_enabled(false),
            )
            .with_mcp_server(server as Arc<dyn MCPServer>)
            .with_mcp_config({
                let mut config = MCPConfig::<()>::default();
                config.include_server_in_tool_names = true;
                config
            }),
    );
    let tools = agent.get_all_tools(&RunContext::new(())).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "mcp_calendar__search");
}

// test_mcp_require_approval_async_callable_uses_run_context
#[tokio::test]
async fn mcp_require_approval_async_callable_uses_run_context() {
    openai_agents_sdk::set_tracing_disabled(true);
    #[derive(Clone, Default)]
    struct Policy {
        needs_approval: bool,
    }
    let seen = Arc::new(Mutex::new(Vec::<bool>::new()));
    let record = seen.clone();
    let server = Arc::new(
        FakeMcpServer::new("fake_mcp_server")
            .configure(move |config| {
                config.require_approval = RequireApproval::dynamic_with_context(
                    move |ctx: RunContext<Policy>, _agent: &str, _tool: &MCPTool| {
                        let needs = ctx.context().needs_approval;
                        record.lock().unwrap().push(needs);
                        async move {
                            tokio::task::yield_now().await;
                            needs
                        }
                    },
                );
            })
            .add_tool("conditional", json!({"type": "object", "properties": {}})),
    );
    let model = Arc::new(FakeModel::new());
    let agent = Agent::<Policy>::new("TestAgent")
        .with_model_instance(model.clone())
        .with_mcp_server(server.clone() as Arc<dyn MCPServer>);

    model.add_turn(vec![function_call("conditional", "{}")]);
    model.add_turn(vec![assistant_message("approved path")]);
    let first = Runner::run_with_options(
        &agent,
        "call conditional",
        RunOptions::new(Policy {
            needs_approval: true,
        }),
    )
    .await
    .unwrap();
    assert!(
        !first.interruptions.is_empty(),
        "run context should be able to trigger approval"
    );

    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, true);
    let resumed = Runner::run_with_options(
        &agent,
        state,
        RunOptions::new(Policy {
            needs_approval: true,
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        resumed.final_output_text().as_deref(),
        Some("approved path")
    );

    model.add_turn(vec![function_call("conditional", "{}")]);
    model.add_turn(vec![assistant_message("no approval path")]);
    let second = Runner::run_with_options(
        &agent,
        "call conditional",
        RunOptions::new(Policy {
            needs_approval: false,
        }),
    )
    .await
    .unwrap();
    assert!(
        second.interruptions.is_empty(),
        "run context should be able to skip approval"
    );
    assert_eq!(
        second.final_output_text().as_deref(),
        Some("no approval path")
    );
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.first(), Some(&true));
    assert_eq!(seen.last(), Some(&false));
}
