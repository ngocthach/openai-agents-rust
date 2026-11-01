//! Round-9 ports of cases the earlier audits filed as Python-only: output-schema names and
//! generic-mapping output types (`test_output_tool.py`), the whitespace handoff-name case
//! (`test_handoff_tool.py`), the hosted MCP callback retry ledger and the persisted-item counter
//! on a streamed resume (`test_hitl_error_scenarios.py`), and the decorator timeout and
//! deferred-loading configuration (`test_function_tool_decorator.py`).
//!
//! The `log` subscriber is process-global; each test filters by a needle no other test uses.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::agent_output::{AgentOutputSchema, AgentOutputSchemaBase};
use openai_agents_sdk::handoffs;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{
    FunctionTool, HostedMCPTool, MCPToolApprovalResult, Tool, ToolTimeoutBehavior,
};
use openai_agents_sdk::{Agent, AgentsError, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

// ------------------------------------------------------------------ output schema

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Foo {
    bar: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
enum Verdict {
    Ok,
    Done,
}

/// Python: `test_structured_output_generic_dict_is_not_wrapped`.
#[test]
fn a_generic_mapping_output_is_not_wrapped() {
    let schema = AgentOutputSchema::new_with_strict::<HashMap<String, i64>>(false).unwrap();
    assert!(!schema.is_wrapped());
    let json_schema = schema.json_schema().unwrap();
    assert!(
        json_schema
            .get("properties")
            .and_then(|p| p.get("response"))
            .is_none(),
        "{json_schema}"
    );
    let validated = schema.validate_json(r#"{"foo": 1}"#).unwrap();
    assert_eq!(validated, json!({"foo": 1}));
}

/// Python: `test_structured_output_generic_dict_rejects_wrapper_shape`.
#[test]
fn a_generic_mapping_output_rejects_the_wrapper_shape() {
    let schema = AgentOutputSchema::new_with_strict::<HashMap<String, i64>>(false).unwrap();
    let err = schema
        .validate_json(&json!({"response": {"foo": 1}}).to_string())
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );
}

/// Python: `test_structured_output_literal_name_handles_literal_values` and the nested-list
/// variant. The name is the readable type, never a module path or a dangling `>`.
#[test]
fn output_schema_names_read_as_the_type() {
    assert_eq!(AgentOutputSchema::new::<Foo>().unwrap().name(), "Foo");
    assert_eq!(
        AgentOutputSchema::new::<Verdict>().unwrap().name(),
        "Verdict"
    );
    assert_eq!(
        AgentOutputSchema::new::<Vec<String>>().unwrap().name(),
        "Vec<String>"
    );
    assert_eq!(
        AgentOutputSchema::new::<Vec<Verdict>>().unwrap().name(),
        "Vec<Verdict>"
    );
    assert_eq!(
        AgentOutputSchema::new_with_strict::<HashMap<String, i64>>(false)
            .unwrap()
            .name(),
        "HashMap<String, i64>"
    );
}

/// The agent's output type name reaches its `Debug` form unchanged.
#[test]
fn the_agent_debug_form_shows_the_readable_output_type() {
    let agent = Agent::<()>::new("a").with_output_type::<Vec<String>>();
    let rendered = format!("{agent:?}");
    assert!(rendered.contains("Vec<String>"), "{rendered}");
}

// ------------------------------------------------------------------ handoff name

#[derive(Default)]
struct Collect(Mutex<Vec<String>>);

impl log::Log for Collect {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

static SINK: std::sync::OnceLock<Collect> = std::sync::OnceLock::new();

fn sink() -> &'static Collect {
    let sink: &'static Collect = SINK.get_or_init(Collect::default);
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

fn logged(needle: &str) -> Vec<String> {
    sink()
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|m| m.contains(needle))
        .cloned()
        .collect()
}

/// Python: `test_default_handoff_tool_name_allows_whitespace_without_warning`.
#[test]
fn a_default_handoff_name_with_whitespace_does_not_warn() {
    sink();
    let tool_name = handoffs::default_tool_name("Refund agent");
    assert_eq!(tool_name, "transfer_to_refund_agent");
    assert!(logged("transfer_to_Refund agent").is_empty());
}

// ------------------------------------------------------------------ MCP callback ledger

fn mcp_request(id: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "mcp_approval_request", "id": id, "server_label": "server",
        "arguments": "{}", "name": "hosted_mcp"
    }))
}

fn hosted_mcp(calls: &Arc<Mutex<Vec<String>>>) -> Tool<()> {
    let calls = calls.clone();
    HostedMCPTool::<()>::new(json!({
        "type": "mcp", "server_label": "server", "server_url": "https://example.invalid",
        "require_approval": "always"
    }))
    .with_on_approval_request(move |req| {
        let calls = calls.clone();
        async move {
            calls
                .lock()
                .unwrap()
                .push(req.data["id"].as_str().unwrap_or_default().to_string());
            MCPToolApprovalResult {
                approve: true,
                reason: Some("ok".into()),
            }
        }
    })
    .into()
}

/// Python: `test_mcp_callback_exact_retry_reuses_stored_decision`. A request id the callback
/// already decided on the shared run context is answered from the stored decision.
#[tokio::test]
async fn an_exact_mcp_retry_reuses_the_stored_decision() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(hosted_mcp(&calls));
    let ctx = RunContext::new(());
    let mut approvals = Vec::new();
    for _ in 0..2 {
        model.add_turns(vec![
            vec![mcp_request("mcp-callback-retry")],
            vec![assistant_message("done")],
        ]);
        let result = Runner::run_with_options(
            &agent,
            "handle mcp",
            RunOptions::with_run_context(ctx.clone()),
        )
        .await
        .unwrap();
        approvals.extend(result.new_items.iter().filter_map(|item| match &item.kind {
            openai_agents_sdk::items::RunItemKind::MCPApprovalResponse { raw_item } => {
                Some(raw_item["approve"].clone())
            }
            _ => None,
        }));
    }
    assert_eq!(*calls.lock().unwrap(), ["mcp-callback-retry"]);
    assert_eq!(approvals, [Value::Bool(true), Value::Bool(true)]);
}

/// Python: `test_mcp_callback_failure_is_not_retried_for_same_request`. A callback that started
/// but never recorded its answer (here: its run was abandoned mid-callback) is not run again
/// for the same request on the same context (the run refuses before the callback).
#[tokio::test]
async fn an_abandoned_mcp_callback_is_not_retried() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let hang = {
        let calls = calls.clone();
        HostedMCPTool::<()>::new(json!({
            "type": "mcp", "server_label": "server", "server_url": "https://example.invalid",
            "require_approval": "always"
        }))
        .with_on_approval_request(move |req| {
            let calls = calls.clone();
            async move {
                calls
                    .lock()
                    .unwrap()
                    .push(req.data["id"].as_str().unwrap_or_default().to_string());
                std::future::pending::<()>().await;
                MCPToolApprovalResult {
                    approve: true,
                    reason: None,
                }
            }
        })
    };
    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(hang);
    let ctx = RunContext::new(());
    model.add_turn(vec![mcp_request("mcp-callback-failure")]);
    let abandoned = tokio::time::timeout(
        Duration::from_millis(100),
        Runner::run_with_options(
            &agent,
            "handle mcp",
            RunOptions::with_run_context(ctx.clone()),
        ),
    )
    .await;
    assert!(abandoned.is_err(), "the callback never answers");

    model.add_turn(vec![mcp_request("mcp-callback-failure")]);
    let err = Runner::run_with_options(
        &agent,
        "handle mcp",
        RunOptions::with_run_context(ctx.clone()),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. })
            && err.to_string().contains("not committed"),
        "{err:?}"
    );
    assert_eq!(*calls.lock().unwrap(), ["mcp-callback-failure"]);
}

// ------------------------------------------------------------------ tool search availability

fn tool_names(tools: &[Tool<bool>]) -> Vec<String> {
    tools.iter().map(Tool::name).collect()
}

/// Python: `test_get_all_tools_preserves_explicit_tool_search_when_deferred_tools_are_disabled`.
#[tokio::test]
async fn an_explicit_tool_search_survives_disabled_deferred_tools() {
    let deferred = FunctionTool::<bool>::new_no_args("deferred_lookup", "", |_c| async move {
        Ok("loaded".to_string())
    })
    .with_defer_loading(true)
    .with_is_enabled_fn(|ctx, _agent| async move { *ctx.context() });
    let agent = Arc::new(
        Agent::<bool>::new("t")
            .with_tool(deferred)
            .with_tool(openai_agents_sdk::tool::ToolSearchTool::new()),
    );

    let disabled = agent.get_all_tools(&RunContext::new(false)).await.unwrap();
    assert_eq!(tool_names(&disabled), ["tool_search"]);

    let enabled = agent.get_all_tools(&RunContext::new(true)).await.unwrap();
    assert_eq!(tool_names(&enabled), ["deferred_lookup", "tool_search"]);
}

/// Python: `test_get_all_tools_keeps_tool_search_for_namespace_only_tools`.
#[tokio::test]
async fn tool_search_is_kept_beside_namespaced_tools() {
    let lookup = FunctionTool::<bool>::new_no_args("lookup_account", "", |_c| async move {
        Ok("ok".to_string())
    });
    let namespaced = openai_agents_sdk::tool::tool_namespace("crm", "CRM tools", vec![lookup])
        .unwrap()
        .remove(0);
    assert_eq!(namespaced.namespace(), Some("crm"));
    let agent = Arc::new(
        Agent::<bool>::new("t")
            .with_tool(namespaced)
            .with_tool(openai_agents_sdk::tool::ToolSearchTool::new()),
    );
    let tools = agent.get_all_tools(&RunContext::new(false)).await.unwrap();
    assert_eq!(tool_names(&tools), ["lookup_account", "tool_search"]);
}

/// Python: `test_get_all_tools_keeps_tool_search_for_deferred_hosted_mcp`.
#[tokio::test]
async fn tool_search_is_kept_beside_a_deferred_hosted_mcp_tool() {
    let hosted = HostedMCPTool::<bool>::new(json!({
        "type": "mcp", "server_label": "crm_server", "server_url": "https://example.com/mcp",
        "defer_loading": true
    }));
    let agent = Arc::new(
        Agent::<bool>::new("t")
            .with_tool(hosted)
            .with_tool(openai_agents_sdk::tool::ToolSearchTool::new()),
    );
    let tools = agent.get_all_tools(&RunContext::new(false)).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert!(matches!(tools[0], Tool::HostedMCP(_)), "{:?}", tools[0]);
    assert!(matches!(tools[1], Tool::ToolSearch(_)));
}

// ------------------------------------------------------------------ decorator configuration

/// Python: `test_decorator_timeout_configuration_is_applied` and
/// `test_function_tool_defer_loading`.
#[test]
fn timeout_and_deferred_loading_configuration_is_kept() {
    let tool =
        FunctionTool::<()>::new_no_args("slow", "", |_ctx| async move { Ok("ok".to_string()) })
            .with_timeout(Duration::from_millis(1250))
            .with_timeout_behavior(ToolTimeoutBehavior::RaiseException)
            .with_timeout_error_function(|_ctx, _error| "too slow".to_string())
            .with_defer_loading(true);
    assert_eq!(tool.timeout, Some(Duration::from_millis(1250)));
    assert_eq!(tool.timeout_behavior, ToolTimeoutBehavior::RaiseException);
    assert!(tool.timeout_error_function.is_some());
    assert!(tool.defer_loading);
}

// ------------------------------------------------------------------ persisted counter

#[cfg(feature = "sqlite")]
mod persisted_counter {
    use super::*;
    use openai_agents_sdk::items::{InputItem, RunInput};
    use openai_agents_sdk::memory::{Session, SqliteSession};
    use openai_agents_sdk::testing::function_call_with_id;

    fn types(items: &[InputItem]) -> Vec<String> {
        items
            .iter()
            .map(|i| {
                let v = serde_json::to_value(i).unwrap();
                v.get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("message")
                    .to_string()
            })
            .collect()
    }

    /// Python: `test_preserve_persisted_item_counter_when_resuming_streamed_runs`. A state
    /// interrupted mid-persistence keeps its saved-item count on resume, so the items that were
    /// not saved are written and the ones that were are not written again, streamed or not.
    #[tokio::test]
    async fn a_resume_keeps_the_saved_item_count() {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        for streamed in [false, true] {
            let model = Arc::new(FakeModel::new());
            let tool =
                FunctionTool::<()>::new_no_args(
                    "gated",
                    "",
                    |_c| async move { Ok("ran".to_string()) },
                )
                .with_needs_approval(true);
            let agent = Agent::<()>::new("agent")
                .with_model_instance(model.clone())
                .with_tool(tool);
            model.add_turns(vec![
                vec![function_call_with_id("gated", "{}", "c1")],
                vec![assistant_message("done")],
            ]);
            let result = Runner::run(&agent, "go").await.unwrap();
            let mut state = result.to_state();
            assert_eq!(state.session_saved_items, 0);
            // The first generated item (the call) reached the session before the interruption.
            state.session_saved_items = 1;
            for interruption in state.get_interruptions() {
                state.approve(&interruption, false);
            }
            let session: Arc<dyn Session> = Arc::new(SqliteSession::in_memory("s").unwrap());
            let opts = RunOptions::new(()).session(session.clone());
            let out = if streamed {
                let mut stream =
                    Runner::run_streamed_with_options(&agent, RunInput::from(state), opts);
                stream.finish().await.unwrap();
                stream.into_result().unwrap()
            } else {
                Runner::run_with_options(&agent, RunInput::from(state), opts)
                    .await
                    .unwrap()
            };
            assert_eq!(out.final_output_text().as_deref(), Some("done"));
            let saved = types(&session.get_items(None).await.unwrap());
            assert!(
                !saved.contains(&"function_call".to_string()),
                "streamed={streamed}: {saved:?}"
            );
            assert!(
                saved.contains(&"function_call_output".to_string()),
                "streamed={streamed}: {saved:?}"
            );
        }
    }
}

// ------------------------------------------------------------------ serialized manifest

/// Python: `test_run_config_accepts_serialized_manifest_without_path_grants` (python and json
/// dumps). A manifest that was serialized without path grants is accepted as configuration.
#[cfg(feature = "sandbox")]
#[test]
fn a_serialized_manifest_without_path_grants_is_accepted_as_configuration() {
    use openai_agents_sdk::sandbox::manifest::Manifest;

    let payload = Manifest::new().with_root("/workspace").to_value().unwrap();
    let manifest = Manifest::from_untrusted_value(&payload, "sandbox.manifest").unwrap();
    assert_eq!(manifest.root(), "/workspace");
    assert!(manifest.extra_path_grants().is_empty());

    let text = serde_json::to_string(&payload).unwrap();
    let reparsed: Value = serde_json::from_str(&text).unwrap();
    let manifest = Manifest::from_untrusted_value(&reparsed, "sandbox.manifest").unwrap();
    assert_eq!(manifest.root(), "/workspace");
    assert!(manifest.extra_path_grants().is_empty());
}
