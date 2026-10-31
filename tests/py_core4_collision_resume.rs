#![cfg(feature = "mcp")]
//! Port of the resume-path cases of Python's `tests/test_tool_name_collision_policy.py`.
//!
//! Python edits RunState private fields (`_model_responses`, `_last_processed_response`, the
//! approval items' raw items and lookup keys) to forge a state. Here the same state is reached
//! through the public fields (`model_responses`, `last_processed_response`, `interruptions`) or
//! by editing the serialized state, then restored with `RunState::from_json`. Nested
//! agent-as-tool cases live in `py_core4_collision_nested.rs`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool, RequireApproval};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{RunConfig, ToolNotFoundBehavior};
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call_full, function_call_with_id,
};
use openai_agents_sdk::tool::{
    ApplyPatchTool, FunctionTool, ProgrammaticToolCallingTool, Tool, tool_namespace,
};
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

/// Run until the model's first call pauses for approval, then approve every interruption.
async fn approved_state(agent: &Agent<()>) -> RunState {
    let first = Runner::run(agent, "Look this up").await.unwrap();
    let mut state = first.to_state();
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    state
}

fn restore(value: Value) -> RunState {
    RunState::from_json(value).unwrap()
}

/// Applies `edit` to every serialized approval item for `call_id`. The item appears in the
/// interrupted step and in the run's generated items, and both must agree on its identity.
fn edit_approvals(value: &mut Value, call_id: &str, edit: &mut dyn FnMut(&mut Map<String, Value>)) {
    match value {
        Value::Object(map) => {
            let is_target = map.get("type").and_then(Value::as_str) == Some("tool_approval_item")
                && map
                    .get("raw_item")
                    .and_then(|raw| raw.get("call_id"))
                    .and_then(Value::as_str)
                    == Some(call_id);
            if is_target {
                edit(map);
            }
            for child in map.values_mut() {
                edit_approvals(child, call_id, edit);
            }
        }
        Value::Array(items) => {
            for child in items {
                edit_approvals(child, call_id, edit);
            }
        }
        _ => {}
    }
}

fn set_raw_item(map: &mut Map<String, Value>, raw: Value) {
    map.insert("raw_item".into(), raw);
}

/// Echo MCP server with a fixed tool list; records the names it was asked to call.
#[derive(Debug)]
struct ApprovalServer {
    config: MCPServerConfig,
    calls: Mutex<Vec<String>>,
}

impl ApprovalServer {
    fn new(approval: RequireApproval) -> Arc<Self> {
        Arc::new(ApprovalServer {
            config: MCPServerConfig {
                require_approval: approval,
                ..Default::default()
            },
            calls: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl MCPServer for ApprovalServer {
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
            name: "lookup".into(),
            title: None,
            description: None,
            input_schema: json!({"type": "object", "properties": {}}),
            annotations: None,
        }])
    }
    async fn call_tool(&self, name: &str, _a: Option<Map<String, Value>>) -> Result<Value> {
        self.calls.lock().unwrap().push(name.to_string());
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

/// Python: `test_resume_warn_mode_rejects_mcp_approval_for_local_winner`.
#[tokio::test]
async fn resume_warn_mode_rejects_mcp_approval_for_local_winner() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let server = ApprovalServer::new(RequireApproval::Always);
    let model = setup();
    model.add_turn(vec![function_call_with_id("lookup", "{}", "lookup_call")]);
    let base = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_mcp_server(server.clone());

    let initial = Runner::run(&base, "Look this up").await.unwrap();
    let mut state = initial.to_state();
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    model.add_turn(vec![assistant_message("done")]);

    // A local `lookup` now wins the name while the MCP server stays attached.
    let with_local = base
        .clone()
        .with_tool(record_tool("lookup", false, &calls, "local"));
    let err = Runner::run(&with_local, state).await.unwrap_err();

    assert!(err.to_string().contains("different recipient"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
    assert!(server.calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_rejects_function_approval_reclassified_as_handoff` (two parametrized
/// cases: the state is used live, or after a JSON round trip).
async fn function_approval_reclassified_as_handoff(deserialize: bool) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![function_call_with_id("route", "{}", "route_call")]);
    let target = Agent::<()>::new("target").with_model_instance({
        let m = setup();
        m.add_turn(vec![assistant_message("target done")]);
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
    };
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("route", true, &calls, "function"));
    let first = Runner::run(&agent, "Route this request").await.unwrap();
    let mut state = if deserialize {
        restore(first.to_state().to_json().unwrap())
    } else {
        first.to_state()
    };
    state.model_responses.last_mut().unwrap().output.clear();
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }

    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_handoff(route_handoff);
    let err = Runner::run(&agent, state).await.unwrap_err();

    assert!(err.to_string().contains("unique call ID"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn resume_rejects_function_approval_reclassified_as_handoff_live_state() {
    function_approval_reclassified_as_handoff(false).await;
}

#[tokio::test]
async fn resume_rejects_function_approval_reclassified_as_handoff_deserialized_state() {
    function_approval_reclassified_as_handoff(true).await;
}

/// Python: `test_reclassified_handoff_is_rejected_before_run_hook`.
#[tokio::test]
async fn reclassified_handoff_is_rejected_before_run_hook() {
    struct RecordingHandoffHooks {
        hook_calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl RunHooks<()> for RecordingHandoffHooks {
        async fn on_handoff(
            &self,
            _context: &openai_agents_sdk::run_context::RunContext<()>,
            _from_agent: &Agent<()>,
            _to_agent: &Agent<()>,
        ) {
            self.hook_calls.lock().unwrap().push("handoff".into());
        }
    }

    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![function_call_with_id("route", "{}", "route")]);
    let target = Agent::<()>::new("target").with_model_instance({
        let m = setup();
        m.add_turn(vec![assistant_message("target done")]);
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
    };
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("route", true, &calls, "function"));
    let first = Runner::run(&agent, "Route this request").await.unwrap();
    let mut state = first.to_state();
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    state.model_responses.last_mut().unwrap().output.clear();

    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_handoff(route_handoff);
    let hook_calls = Arc::new(Mutex::new(Vec::new()));
    let hooks = Arc::new(RecordingHandoffHooks {
        hook_calls: hook_calls.clone(),
    });
    let err = Runner::run_with_options(&agent, state, RunOptions::<()>::new(()).hooks(hooks))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("unique call ID"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
    assert!(hook_calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_rejects_queued_handoff_reclassified_as_function`.
#[tokio::test]
async fn resume_rejects_queued_handoff_reclassified_as_function() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("approval_tool", "{}", "approval_call"),
        function_call_with_id("route", "{}", "route_call"),
    ]);
    let target = Agent::<()>::new("target");
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("approval_tool", true, &calls, "approved"))
        .with_handoff(handoff(Arc::new(target.clone())).with_tool_name("route"));
    let mut state = approved_state(&agent).await;
    state.model_responses.last_mut().unwrap().output.clear();

    // The queued `route` call now names a plain function instead of the handoff.
    let reclassified = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("approval_tool", true, &calls, "approved"))
        .with_tool(record_tool("route", false, &calls, "route"));
    let err = Runner::run(&reclassified, state).await.unwrap_err();

    assert!(err.to_string().contains("unique call ID"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_rejects_conflicting_persisted_identity_before_sibling_effects`.
#[tokio::test]
async fn resume_rejects_conflicting_persisted_identity_before_sibling_effects() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("lookup", "{}", "conflicting_call"),
        function_call_with_id("sibling", "{}", "sibling_call"),
    ]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("lookup", true, &calls, "conflicting"))
        .with_tool(record_tool("sibling", true, &calls, "sibling"));
    let mut state = approved_state(&agent).await;
    let mut value = state.to_json().unwrap();
    edit_approvals(&mut value, "conflicting_call", &mut |map| {
        set_raw_item(
            map,
            json!({
                "type": "function_call",
                "name": "lookup",
                "namespace": "current",
                "call_id": "conflicting_call",
            }),
        );
        map.insert(
            "tool_lookup_key".into(),
            json!({"kind": "namespaced", "namespace": "legacy", "name": "lookup"}),
        );
    });
    state = restore(value);

    let err = Runner::run(&agent, state).await.unwrap_err();

    assert!(err.to_string().contains("Persisted tool identity"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_rejects_legacy_approval_name_change_before_side_effects`.
#[tokio::test]
async fn resume_rejects_legacy_approval_name_change_before_side_effects() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "old_lookup",
        "{}",
        "lookup_call",
    )]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("old_lookup", true, &calls, "old"));
    let mut state = approved_state(&agent).await;
    let mut value = state.to_json().unwrap();
    edit_approvals(&mut value, "lookup_call", &mut |map| {
        map.remove("tool_lookup_key");
        set_raw_item(
            map,
            json!({
                "type": "function_call",
                "name": "new_lookup",
                "arguments": "{}",
                "call_id": "lookup_call",
            }),
        );
    });
    state = restore(value);
    let renamed = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("new_lookup", false, &calls, "new"));

    let err = Runner::run(&renamed, state).await.unwrap_err();

    assert!(err.to_string().contains("Persisted tool identity"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_treats_apply_patch_prefixed_queued_function_as_function`. Four
/// parametrized cases: bare or namespaced, raise or return the error to the model.
async fn apply_patch_prefixed_function_resumes_as_function(
    namespace: Option<&'static str>,
    return_error_to_model: bool,
) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let tools = |calls: &Arc<Mutex<Vec<String>>>| -> Vec<Tool<()>> {
        let base = record_tool("apply_patch_lookup", true, calls, "function");
        match namespace {
            Some(ns) => tool_namespace(ns, "Lookup tools", vec![base])
                .unwrap()
                .into_iter()
                .map(Into::into)
                .collect(),
            None => vec![base.into()],
        }
    };
    let model = setup();
    model.add_turn(vec![function_call_full(
        "apply_patch_lookup",
        "{}",
        "lookup_call",
        None,
        namespace.map(str::to_string),
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tools(tools(&calls));
    let state = approved_state(&agent).await;

    let without_tools = Agent::<()>::new("agent").with_model_instance(model.clone());
    let behavior = if return_error_to_model {
        ToolNotFoundBehavior::ReturnErrorToModel
    } else {
        ToolNotFoundBehavior::RaiseError
    };
    let config = RunConfig::<()>::default().with_tool_not_found_behavior(behavior);
    let options = RunOptions::<()>::new(()).run_config(config);

    if return_error_to_model {
        let resumed = Runner::run_with_options(&without_tools, state, options)
            .await
            .unwrap();
        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    } else {
        let err = Runner::run_with_options(&without_tools, state, options)
            .await
            .unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("apply_patch_lookup not found in agent agent"),
            "{text}"
        );
    }
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn apply_patch_prefixed_queued_function_bare_raises() {
    apply_patch_prefixed_function_resumes_as_function(None, false).await;
}

#[tokio::test]
async fn apply_patch_prefixed_queued_function_bare_returns_error() {
    apply_patch_prefixed_function_resumes_as_function(None, true).await;
}

#[tokio::test]
async fn apply_patch_prefixed_queued_function_namespaced_raises() {
    apply_patch_prefixed_function_resumes_as_function(Some("tools"), false).await;
}

#[tokio::test]
async fn apply_patch_prefixed_queued_function_namespaced_returns_error() {
    apply_patch_prefixed_function_resumes_as_function(Some("tools"), true).await;
}

/// Python: `test_approved_malformed_approval_only_stays_pending_without_side_effects` (two
/// parametrized malformed raw items).
async fn malformed_approval_only_stays_pending(malformed_raw_item: Value) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![function_call_with_id("lookup", "{}", "lookup_call")]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("lookup", true, &calls, "lookup"));
    let first = Runner::run(&agent, "Look this up").await.unwrap();
    let mut state = first.to_state();
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    if let Some(processed) = state.last_processed_response.as_mut() {
        processed.functions.clear();
    }
    state.model_responses.last_mut().unwrap().output.clear();
    let mut value = state.to_json().unwrap();
    edit_approvals(&mut value, "lookup_call", &mut |map| {
        set_raw_item(map, malformed_raw_item.clone());
    });
    state = restore(value);

    let resumed = Runner::run(&agent, state).await.unwrap();

    // Python returns the same approval object; Rust returns the malformed item, so the tool
    // name identifies it.
    assert_eq!(resumed.interruptions.len(), 1);
    assert_eq!(resumed.interruptions[0].name(), "lookup");
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn approved_malformed_approval_without_arguments_stays_pending() {
    malformed_approval_only_stays_pending(json!({
        "type": "function_call",
        "name": "lookup",
        "call_id": "lookup_call",
    }))
    .await;
}

#[tokio::test]
async fn approved_malformed_approval_with_item_id_only_stays_pending() {
    malformed_approval_only_stays_pending(json!({
        "type": "function_call",
        "name": "lookup",
        "arguments": "{}",
        "id": "lookup_call",
    }))
    .await;
}

/// Python: `test_resume_uses_queued_arguments_instead_of_mutated_approval_arguments`.
#[tokio::test]
async fn resume_uses_queued_arguments_instead_of_mutated_approval_arguments() {
    let calls: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    let lookup = FunctionTool::new("lookup", "", move |_ctx, args: Value| {
        let recorded = recorded.clone();
        async move {
            let amount = args["amount"].as_i64().unwrap_or_default();
            recorded.lock().unwrap().push(amount);
            Ok::<_, AgentsError>(amount.to_string())
        }
    })
    .with_needs_approval(true);
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "lookup",
        r#"{"amount":10}"#,
        "lookup_call",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(lookup);
    let first = Runner::run(&agent, "Look this up").await.unwrap();
    let mut state = restore(first.to_state().to_json().unwrap());
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    state.interruptions[0].raw_item.arguments = r#"{"amount":999}"#.into();

    let resumed = Runner::run(&agent, state).await.unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*calls.lock().unwrap(), vec![10]);
}

/// Python: `test_resume_snapshots_program_parent_context_before_inventory_await`. The forged
/// caller is present before the run; the Python in-flight mutation of the parent program
/// during `get_all_tools` cannot be expressed (see the report).
#[tokio::test]
async fn resume_rejects_forged_program_caller_against_parent_program() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let program = OutputItem::from_value(json!({
        "type": "program",
        "id": "program_item",
        "call_id": "legit_program",
        "code": "lookup()",
        "fingerprint": "fingerprint",
    }));
    let lookup_call = OutputItem::from_value(json!({
        "type": "function_call",
        "name": "lookup",
        "arguments": "{}",
        "call_id": "lookup_call",
        "caller": {"type": "program", "caller_id": "legit_program"},
    }));
    let model = setup();
    model.add_turn(vec![program.clone(), lookup_call]);
    let lookup = record_tool("lookup", true, &calls, "lookup")
        .with_allowed_callers(vec!["programmatic".into()]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(lookup);
    let first = Runner::run(&agent, "Look this up").await.unwrap();
    let mut state = first.to_state();
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    if let Some(processed) = state.last_processed_response.as_mut() {
        processed.functions.clear();
    }
    state.model_responses.last_mut().unwrap().output = vec![program];
    let mut value = state.to_json().unwrap();
    edit_approvals(&mut value, "lookup_call", &mut |map| {
        let mut raw = map["raw_item"].clone();
        raw["caller"]["caller_id"] = json!("forged_program");
        set_raw_item(map, raw);
    });
    let state = restore(value);

    let err = Runner::run(&agent, state).await.unwrap_err();

    assert!(
        err.to_string()
            .contains("does not match a parent program item"),
        "{err}"
    );
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_rejects_response_backed_approval_lookup_mismatch_before_effects`.
#[tokio::test]
async fn resume_rejects_response_backed_approval_lookup_mismatch_before_effects() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("lookup", "{}", "lookup_call"),
        function_call_with_id("sibling", "{}", "sibling_call"),
    ]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("lookup", true, &calls, "lookup"))
        .with_tool(record_tool("sibling", true, &calls, "sibling"));
    let first = Runner::run(&agent, "Look these up").await.unwrap();
    let mut state = restore(first.to_state().to_json().unwrap());
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    if let Some(processed) = state.last_processed_response.as_mut() {
        processed
            .functions
            .retain(|run| run.tool_call["call_id"] != "lookup_call");
    }
    let mut value = state.to_json().unwrap();
    edit_approvals(&mut value, "lookup_call", &mut |map| {
        let mut raw = map["raw_item"].clone();
        raw["name"] = json!("other");
        set_raw_item(map, raw);
        map.insert("tool_name".into(), json!("other"));
        map.insert(
            "tool_lookup_key".into(),
            json!({"kind": "bare", "name": "other"}),
        );
    });
    let state = restore(value);

    let resumed = Runner::run(&agent, state).await.unwrap();

    assert_eq!(resumed.interruptions.len(), 2);
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_resume_preserves_function_shaped_apply_patch_owner`.
#[tokio::test]
async fn resume_preserves_function_shaped_apply_patch_owner() {
    #[derive(Clone, Default)]
    struct CountingEditor {
        operations: Arc<Mutex<usize>>,
    }

    #[async_trait]
    impl ApplyPatchEditor for CountingEditor {
        async fn create_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
            *self.operations.lock().unwrap() += 1;
            Ok(ApplyPatchResult {
                status: None,
                output: Some("created".into()),
            })
        }
        async fn update_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
            *self.operations.lock().unwrap() += 1;
            Ok(ApplyPatchResult {
                status: None,
                output: Some("updated".into()),
            })
        }
        async fn delete_file(&self, _op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
            *self.operations.lock().unwrap() += 1;
            Ok(ApplyPatchResult {
                status: None,
                output: Some("deleted".into()),
            })
        }
    }

    let editor = CountingEditor::default();
    let operations = editor.operations.clone();
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "apply_patch",
        r#"{"type":"update_file","path":"test.md","diff":"-a\n+b\n"}"#,
        "patch_call",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(ApplyPatchTool::new(editor).with_needs_approval(true));
    let state = approved_state(&agent).await;

    let resumed = Runner::run(&agent, state).await.unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*operations.lock().unwrap(), 1);
}

/// Python: `test_resume_rejects_cross_kind_approval_identity_before_sibling_effects`.
#[tokio::test]
async fn resume_rejects_cross_kind_approval_identity_before_sibling_effects() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("lookup", "{}", "lookup_call"),
        function_call_with_id("sibling", "{}", "sibling_call"),
    ]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("lookup", true, &calls, "lookup"))
        .with_tool(record_tool("sibling", true, &calls, "sibling"));
    let mut state = approved_state(&agent).await;
    let mut value = state.to_json().unwrap();
    edit_approvals(&mut value, "lookup_call", &mut |map| {
        set_raw_item(
            map,
            json!({
                "type": "custom_tool_call",
                "name": "evil",
                "call_id": "lookup_call",
                "input": "{}",
            }),
        );
    });
    state = restore(value);

    let err = Runner::run(&agent, state).await.unwrap_err();

    assert!(err.to_string().contains("Persisted tool identity"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
}

/// Python: `test_approved_malformed_queued_approval_stays_pending_without_side_effects` (two
/// parametrized raw items).
async fn malformed_queued_approval_stays_pending(malformed_raw_item: Value) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("lookup", "{}", "lookup_call"),
        function_call_with_id("sibling", "{}", "sibling_call"),
    ]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("lookup", true, &calls, "lookup"))
        .with_tool(record_tool("sibling", true, &calls, "sibling"));
    let mut state = approved_state(&agent).await;
    let mut value = state.to_json().unwrap();
    edit_approvals(&mut value, "lookup_call", &mut |map| {
        set_raw_item(map, malformed_raw_item.clone());
    });
    state = restore(value);

    let resumed = Runner::run(&agent, state).await.unwrap();

    assert!(
        resumed.interruptions.iter().any(|i| i.name() == "lookup"),
        "the malformed lookup approval stays pending"
    );
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn approved_malformed_queued_approval_without_type_stays_pending() {
    malformed_queued_approval_stays_pending(json!({"name": "lookup"})).await;
}

#[tokio::test]
async fn approved_malformed_queued_approval_with_changed_call_id_stays_pending() {
    malformed_queued_approval_stays_pending(json!({
        "type": "function_call",
        "name": "lookup",
        "arguments": "{}",
        "call_id": "changed_call",
    }))
    .await;
}

/// Python: `test_resume_preserves_model_order_for_function_outcomes`. Previously ignored: the
/// not-found output of a resumed approved call was dropped (run_internal/mod.rs).
#[tokio::test]
async fn resume_preserves_model_order_for_function_outcomes() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("missing_lookup", "{}", "missing_call"),
        function_call_with_id("rejected_lookup", "{}", "rejected_call"),
        function_call_with_id("available_lookup", "{}", "available_call"),
    ]);
    let agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("missing_lookup", true, &calls, "missing"))
        .with_tool(record_tool("rejected_lookup", true, &calls, "rejected"))
        .with_tool(record_tool("available_lookup", true, &calls, "available"));

    let first = Runner::run(&agent, "Look these up").await.unwrap();
    let mut state = first.to_state();
    for interruption in state.get_interruptions() {
        if interruption.name() == "rejected_lookup" {
            state.reject(&interruption, false, None);
        } else {
            state.approve(&interruption, false);
        }
    }
    let resumed_agent = Agent::<()>::new("agent")
        .with_model_instance(model.clone())
        .with_tool(record_tool("rejected_lookup", true, &calls, "rejected"))
        .with_tool(record_tool("available_lookup", true, &calls, "available"));
    model.add_turn(vec![assistant_message("done")]);
    let config = RunConfig::<()>::default()
        .with_tool_not_found_behavior(ToolNotFoundBehavior::ReturnErrorToModel);

    let resumed = Runner::run_with_options(
        &resumed_agent,
        state,
        RunOptions::<()>::new(()).run_config(config),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(*calls.lock().unwrap(), vec!["available".to_string()]);
    let output_ids: Vec<String> = resumed
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            openai_agents_sdk::items::RunItemKind::ToolCallOutput { raw_item, .. }
                if raw_item["type"] == "function_call_output" =>
            {
                raw_item["call_id"].as_str().map(str::to_string)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        output_ids,
        vec!["missing_call", "rejected_call", "available_call"]
    );
}
