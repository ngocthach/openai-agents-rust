//! Ported from the Python SDK's `tests/test_run_internal_items.py`.
//!
//! Python calls `agents.run_internal.items` helpers directly. Rust keeps those helpers private,
//! so each port drives the same behaviour through the runner:
//! - orphan pruning (`drop_orphan_function_calls`, `prepare_model_input_items`) runs on stored
//!   session history, observed in the input the model receives;
//! - `deduplicate_input_items_preferring_latest` runs on every model input, observed the same way;
//! - `run_item_to_input_item` is `RunItem::to_input_item`.

use std::sync::Arc;

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run_config::ReasoningItemIdPolicy;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, InMemorySession, InputItem, ReasoningOutput, RunConfig, RunItem, RunOptions, RunState,
    Runner, Session,
};
use serde_json::{Value, json};

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

fn items(values: Vec<Value>) -> Vec<InputItem> {
    values.into_iter().map(InputItem).collect()
}

/// Seed a session with `history`, run one turn, and return what the model received, minus the
/// trailing user message the run itself added.
async fn model_input_after_session(history: Vec<Value>) -> Vec<Value> {
    disable_tracing();
    let session = Arc::new(InMemorySession::new("items"));
    session.add_items(items(history)).await.unwrap();
    let model = scripted(vec![vec![assistant_message("done")]]);
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    Runner::run_with_options(&agent, "next", RunOptions::new(()).session(session))
        .await
        .unwrap();
    let mut input: Vec<Value> = model
        .first_call()
        .unwrap()
        .request
        .input
        .into_iter()
        .map(|i| i.0)
        .collect();
    let last = input.pop().expect("run input");
    assert_eq!(last.get("content"), Some(&json!("next")));
    input
}

/// Run with `input` as the caller input and return what the model received.
async fn model_input_for(input: Vec<Value>) -> Vec<Value> {
    disable_tracing();
    let model = scripted(vec![vec![assistant_message("done")]]);
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    Runner::run_with_options(&agent, input, RunOptions::new(()))
        .await
        .unwrap();
    model
        .first_call()
        .unwrap()
        .request
        .input
        .into_iter()
        .map(|i| i.0)
        .collect()
}

/// Resume a finished snapshot whose generated items are `generated` and return what the model
/// received after the original input. This is the `prepare_model_input_items` /
/// `normalize_resumed_input` route: runner-generated history, pruned without a scope.
async fn model_input_after_resume(generated: Vec<Value>) -> Vec<Value> {
    disable_tracing();
    let model = scripted(vec![
        vec![assistant_message("first")],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    let first = Runner::run(&agent, "go").await.unwrap();
    let mut state: RunState = first.to_state();
    state.generated_items = generated
        .into_iter()
        .map(
            |raw_item| match raw_item.get("type").and_then(Value::as_str) {
                Some(t) if t.ends_with("_output") => RunItem::new(
                    "A",
                    RunItemKind::ToolCallOutput {
                        output: raw_item.get("output").cloned().unwrap_or(Value::Null),
                        raw_item,
                        custom_data: None,
                    },
                ),
                _ => RunItem::new("A", RunItemKind::ToolCall { raw_item }),
            },
        )
        .collect();
    Runner::run(&agent, state).await.unwrap();
    let mut input: Vec<Value> = model
        .last_call()
        .unwrap()
        .request
        .input
        .into_iter()
        .map(|i| i.0)
        .collect();
    let first = input.remove(0);
    assert_eq!(first["content"], "go");
    input
}

fn types(items: &[Value]) -> Vec<Option<String>> {
    items
        .iter()
        .map(|i| i.get("type").and_then(Value::as_str).map(str::to_string))
        .collect()
}

fn has(items: &[Value], ty: &str, call_id: &str) -> bool {
    items.iter().any(|i| {
        i.get("type").and_then(Value::as_str) == Some(ty)
            && i.get("call_id").and_then(Value::as_str) == Some(call_id)
    })
}

// ---------------------------------------------------------------------------------------------
// function_tool_error_output: direct calls keep plain-text rejection messages
// ---------------------------------------------------------------------------------------------

async fn rejected_output(caller: Option<Value>, has_schema: bool) -> Value {
    disable_tracing();
    let mut tool =
        FunctionTool::<()>::new("lookup", "", |_ctx, _a: Value| async move { Ok("ran") })
            .with_needs_approval(true);
    if has_schema {
        tool = tool
            .with_output_json_schema(json!({"type": "object", "properties": {}}))
            .unwrap();
    }
    let mut call = openai_agents_sdk::testing::function_call_with_id("lookup", "{}", "call-42");
    if let (Some(caller), OutputItem::FunctionCall(fc)) = (caller, &mut call) {
        fc.extra.insert("caller".into(), caller);
    }
    let model = scripted(vec![vec![call], vec![assistant_message("done")]]);
    let agent = Agent::<()>::new("A")
        .with_model_instance(model.clone())
        .with_tool(tool);
    let first = Runner::run(&agent, "go").await.unwrap();
    let mut state = first.to_state();
    let pending = state.get_interruptions()[0].clone();
    state.reject(&pending, false, Some("Request rejected.".into()));
    let resumed = Runner::run(&agent, state).await.unwrap();
    resumed
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => raw_item.get("output").cloned(),
            _ => None,
        })
        .expect("rejection output")
}

#[tokio::test]
async fn direct_function_tool_errors_preserve_plain_text_untyped_no_caller() {
    assert_eq!(
        rejected_output(None, false).await,
        json!("Request rejected.")
    );
}

#[tokio::test]
async fn direct_function_tool_errors_preserve_plain_text_untyped_direct() {
    assert_eq!(
        rejected_output(Some(json!({"type": "direct"})), false).await,
        json!("Request rejected.")
    );
}

#[tokio::test]
async fn direct_function_tool_errors_preserve_plain_text_typed_no_caller() {
    assert_eq!(
        rejected_output(None, true).await,
        json!("Request rejected.")
    );
}

#[tokio::test]
async fn direct_function_tool_errors_preserve_plain_text_typed_direct() {
    assert_eq!(
        rejected_output(Some(json!({"type": "direct"})), true).await,
        json!("Request rejected.")
    );
}

// ---------------------------------------------------------------------------------------------
// drop_orphan_function_calls over stored history
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn drop_orphan_function_calls_preserves_non_mapping_entries() {
    let input = model_input_after_session(vec![
        json!({"type": "message", "role": "user", "content": "hello"}),
        json!({"type": "function_call", "call_id": "orphan_call", "name": "orphan", "arguments": "{}"}),
        json!({"type": "function_call", "call_id": "paired_call", "name": "paired", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "paired_call", "output": "ok"}),
        json!({"call_id": "not-a-tool-call"}),
    ])
    .await;
    assert_eq!(input[0]["type"], "message");
    assert!(has(&input, "function_call", "paired_call"));
    assert!(!has(&input, "function_call", "orphan_call"));
    assert!(
        input
            .iter()
            .any(|i| i.get("call_id") == Some(&json!("not-a-tool-call")))
    );
}

#[tokio::test]
async fn replay_pruning_drops_orphan_program_call_chain() {
    let generated = vec![
        json!({"type": "program", "call_id": "program_orphan",
               "code": "return await tools.lookup({});", "fingerprint": "fingerprint:orphan"}),
        json!({"type": "function_call", "call_id": "function_orphan", "name": "lookup",
               "arguments": "{}", "caller": {"type": "program", "caller_id": "program_orphan"}}),
    ];
    // Stored history with a pruning scope drops the chain too.
    assert_eq!(
        model_input_after_session(generated.clone()).await,
        Vec::<Value>::new()
    );

    // `prepare_model_input_items` / `normalize_resumed_input` route: generated items.
    assert_eq!(
        model_input_after_resume(generated).await,
        Vec::<Value>::new()
    );
}

#[tokio::test]
async fn drop_orphan_function_calls_preserves_active_program_call_chain() {
    let payload = vec![
        json!({"type": "program", "call_id": "program_active",
               "code": "return await tools.lookup({});", "fingerprint": "fingerprint:active"}),
        json!({"type": "function_call", "call_id": "function_completed", "name": "lookup",
               "arguments": "{}", "caller": {"type": "program", "caller_id": "program_active"}}),
        json!({"type": "function_call_output", "call_id": "function_completed", "output": "done",
               "caller": {"type": "program", "caller_id": "program_active"}}),
    ];
    assert_eq!(model_input_after_resume(payload.clone()).await, payload);
}

async fn program_owned_hosted_item_case(hosted_item_type: &str) {
    let payload = vec![
        json!({"type": "program", "call_id": "program_pending",
               "code": "return await tools.lookup({});", "fingerprint": "fingerprint:pending"}),
        json!({"type": hosted_item_type, "id": "hosted_item",
               "caller": {"type": "program", "caller_id": "program_pending"}}),
    ];
    assert_eq!(
        model_input_after_resume(payload.clone()).await,
        payload,
        "{hosted_item_type}"
    );
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_file_search_call() {
    program_owned_hosted_item_case("file_search_call").await;
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_web_search_call() {
    program_owned_hosted_item_case("web_search_call").await;
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_code_interpreter_call() {
    program_owned_hosted_item_case("code_interpreter_call").await;
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_image_generation_call() {
    program_owned_hosted_item_case("image_generation_call").await;
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_mcp_list_tools() {
    program_owned_hosted_item_case("mcp_list_tools").await;
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_mcp_call() {
    program_owned_hosted_item_case("mcp_call").await;
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_mcp_approval_request() {
    program_owned_hosted_item_case("mcp_approval_request").await;
}

#[tokio::test]
async fn replay_pruning_preserves_program_owned_hosted_items_mcp_approval_response() {
    program_owned_hosted_item_case("mcp_approval_response").await;
}

#[tokio::test]
async fn drop_orphan_function_calls_drops_reasoning_preceding_dropped_tool_call() {
    let input = model_input_after_session(vec![
        json!({"role": "user", "content": "hi"}),
        json!({"type": "reasoning", "id": "rs_orphan_a", "summary": []}),
        json!({"type": "reasoning", "id": "rs_orphan_b", "summary": []}),
        json!({"type": "function_call", "call_id": "orphan_call", "name": "orphan", "arguments": "{}"}),
        json!({"type": "reasoning", "id": "rs_paired", "summary": []}),
        json!({"type": "function_call", "call_id": "paired_call", "name": "paired", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "paired_call", "output": "ok"}),
    ])
    .await;
    let reasoning_ids: Vec<&Value> = input
        .iter()
        .filter(|i| i.get("type") == Some(&json!("reasoning")))
        .map(|i| &i["id"])
        .collect();
    assert_eq!(reasoning_ids, vec![&json!("rs_paired")]);
    assert!(!has(&input, "function_call", "orphan_call"));
}

#[tokio::test]
async fn drop_orphan_function_calls_keeps_lone_reasoning_when_no_tool_calls_dropped() {
    let payload = vec![json!({"type": "reasoning", "id": "rs_lone", "summary": []})];
    assert_eq!(model_input_after_session(payload.clone()).await, payload);
}

#[tokio::test]
async fn drop_orphan_function_calls_handles_tool_search_calls() {
    let input = model_input_after_session(vec![
        json!({"type": "tool_search_call", "call_id": "tool_search_orphan",
               "arguments": {"query": "orphan"}, "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_call", "call_id": "tool_search_keep",
               "arguments": {"query": "keep"}, "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_output", "call_id": "tool_search_keep",
               "execution": "server", "status": "completed", "tools": []}),
    ])
    .await;
    assert!(has(&input, "tool_search_call", "tool_search_keep"));
    assert!(!has(&input, "tool_search_call", "tool_search_orphan"));
}

#[tokio::test]
async fn drop_orphan_function_calls_preserves_hosted_tool_search_pairs_without_call_ids() {
    let input = model_input_after_session(vec![
        json!({"type": "tool_search_call", "call_id": null, "arguments": {"query": "keep"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_output", "call_id": null, "execution": "server",
               "status": "completed", "tools": []}),
    ])
    .await;
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["type"], "tool_search_call");
    assert_eq!(input[1]["type"], "tool_search_output");
}

#[tokio::test]
async fn drop_orphan_function_calls_matches_latest_anonymous_tool_search_call() {
    let input = model_input_after_session(vec![
        json!({"type": "tool_search_call", "call_id": null, "arguments": {"query": "orphan"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_call", "call_id": null, "arguments": {"query": "paired"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_output", "call_id": null, "execution": "server",
               "status": "completed", "tools": []}),
    ])
    .await;
    assert_eq!(
        types(&input),
        vec![
            Some("tool_search_call".to_string()),
            Some("tool_search_output".to_string())
        ]
    );
    assert_eq!(input[0]["arguments"], json!({"query": "paired"}));
}

#[tokio::test]
async fn drop_orphan_function_calls_does_not_pair_named_tool_search_with_anonymous_output() {
    let input = model_input_after_session(vec![
        json!({"type": "tool_search_call", "call_id": "orphan_search", "arguments": {"query": "keep"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_output", "call_id": null, "execution": "server",
               "status": "completed", "tools": []}),
    ])
    .await;
    assert_eq!(types(&input), vec![Some("tool_search_output".to_string())]);
}

#[tokio::test]
async fn drop_orphan_function_calls_keeps_reasoning_chain_before_non_dropped_item() {
    let input = model_input_after_session(vec![
        json!({"type": "reasoning", "id": "rs_1", "summary": []}),
        json!({"type": "reasoning", "id": "rs_2", "summary": []}),
        json!({"type": "message", "role": "assistant", "content": []}),
        json!({"type": "function_call", "call_id": "orphan_call", "name": "orphan", "arguments": "{}"}),
    ])
    .await;
    assert_eq!(input[0]["id"], "rs_1");
    assert_eq!(input[1]["id"], "rs_2");
    assert_eq!(
        types(&input),
        vec![
            Some("reasoning".to_string()),
            Some("reasoning".to_string()),
            Some("message".to_string())
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// Internal session metadata never reaches the model
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn normalize_input_items_for_api_strips_internal_tool_call_metadata() {
    let input = model_input_after_session(vec![
        json!({"type": "function_call", "call_id": "call_123", "name": "lookup_account",
               "arguments": "{}", "_agents_tool_description": "Lookup customer records.",
               "_agents_tool_title": "Lookup Account"}),
        json!({"type": "function_call_output", "call_id": "call_123", "output": "ok"}),
    ])
    .await;
    assert!(input[0].get("_agents_tool_description").is_none());
    assert!(input[0].get("_agents_tool_title").is_none());
    assert_eq!(input[0]["name"], "lookup_account");
}

// ---------------------------------------------------------------------------------------------
// deduplicate_input_items_preferring_latest over the model input
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn deduplicate_input_items_handles_fake_ids_and_approval_request_ids() {
    let input = model_input_for(vec![
        json!({"type": "function_call_output", "id": "__fake_id__", "call_id": "call-1", "output": "first"}),
        json!({"type": "function_call_output", "id": "__fake_id__", "call_id": "call-1", "output": "latest"}),
        json!({"type": "mcp_approval_response", "approval_request_id": "req-1", "approve": true}),
        json!({"type": "mcp_approval_response", "approval_request_id": "req-1", "approve": false}),
        json!("plain"),
    ])
    .await;
    assert_eq!(input.len(), 3);
    assert_eq!(input[2], json!("plain"));
    let outputs: Vec<&Value> = input
        .iter()
        .filter(|i| i.get("type") == Some(&json!("function_call_output")))
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0]["output"], "latest");
}

#[tokio::test]
async fn deduplicate_input_items_preferring_latest_keeps_original_order() {
    let call =
        json!({"type": "function_call", "call_id": "call-1", "name": "tool", "arguments": "{}"});
    let input = model_input_for(vec![
        call.clone(),
        json!({"type": "function_call_output", "call_id": "call-1", "output": "result"}),
        json!({"role": "assistant", "content": "ack"}),
        call,
    ])
    .await;
    assert_eq!(
        types(&input),
        vec![
            Some("function_call".to_string()),
            Some("function_call_output".to_string()),
            None
        ]
    );
    assert_eq!(input[2]["role"], "assistant");
}

#[tokio::test]
async fn deduplicate_input_items_preferring_latest_keeps_latest_output_position() {
    let input = model_input_for(vec![
        json!({"type": "function_call_output", "call_id": "call-1", "output": "old"}),
        json!({"role": "user", "content": "next"}),
        json!({"type": "function_call_output", "call_id": "call-1", "output": "new"}),
    ])
    .await;
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["content"], "next");
    assert_eq!(input[1]["output"], "new");
}

#[tokio::test]
async fn deduplicate_input_items_preferring_latest_keeps_output_after_matching_call() {
    let input = model_input_for(vec![
        json!({"type": "function_call_output", "call_id": "call-1", "output": "old"}),
        json!({"type": "function_call", "call_id": "call-1", "name": "tool", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call-1", "output": "new"}),
    ])
    .await;
    assert_eq!(
        types(&input),
        vec![
            Some("function_call".to_string()),
            Some("function_call_output".to_string())
        ]
    );
    assert_eq!(input[1]["output"], "new");
}

#[tokio::test]
async fn deduplicate_input_items_preferring_latest_keeps_reasoning_before_follower() {
    let input = model_input_for(vec![
        json!({"type": "reasoning", "id": "rs-1", "summary": [{"type": "summary_text", "text": "old"}]}),
        json!({"type": "function_call", "call_id": "call-1", "name": "tool", "arguments": "{}"}),
        json!({"type": "reasoning", "id": "rs-1", "summary": [{"type": "summary_text", "text": "new"}]}),
    ])
    .await;
    assert_eq!(
        types(&input),
        vec![
            Some("reasoning".to_string()),
            Some("function_call".to_string())
        ]
    );
    assert_eq!(
        input[0]["summary"],
        json!([{"type": "summary_text", "text": "new"}])
    );
}

#[tokio::test]
async fn deduplicate_input_items_preferring_latest_keeps_approval_request_before_response() {
    let input = model_input_for(vec![
        json!({"type": "mcp_approval_request", "id": "approval-1", "arguments": "old",
               "name": "lookup", "server_label": "server"}),
        json!({"type": "mcp_approval_response", "approval_request_id": "approval-1", "approve": true}),
        json!({"type": "mcp_approval_request", "id": "approval-1", "arguments": "new",
               "name": "lookup", "server_label": "server"}),
    ])
    .await;
    assert_eq!(
        types(&input),
        vec![
            Some("mcp_approval_request".to_string()),
            Some("mcp_approval_response".to_string())
        ]
    );
    assert_eq!(input[0]["arguments"], "new");
}

#[tokio::test]
async fn deduplicate_input_items_preferring_latest_leaves_unique_items_untouched() {
    let payload = vec![
        json!({"role": "user", "content": "hi"}),
        json!({"type": "function_call", "call_id": "call-1", "name": "tool", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call-1", "output": "result"}),
        json!({"role": "user", "content": "hi"}),
    ];
    assert_eq!(model_input_for(payload.clone()).await, payload);
}

// ---------------------------------------------------------------------------------------------
// run_item_to_input_item == RunItem::to_input_item
// ---------------------------------------------------------------------------------------------

fn reasoning_item(id: &str) -> OutputItem {
    OutputItem::Reasoning(ReasoningOutput {
        id: id.to_string(),
        summary: vec![],
        content: None,
        encrypted_content: None,
        status: None,
        extra: Default::default(),
    })
}

async fn replayed_reasoning(policy: Option<ReasoningItemIdPolicy>) -> Value {
    disable_tracing();
    let tool = FunctionTool::<()>::new("noop", "", |_ctx, _a: Value| async move { Ok("ok") });
    let model = scripted(vec![
        vec![
            reasoning_item("rs_123"),
            openai_agents_sdk::testing::function_call_with_id("noop", "{}", "call_1"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("A")
        .with_model_instance(model.clone())
        .with_tool(tool);
    let mut config = RunConfig::default();
    if let Some(policy) = policy {
        config = config.with_reasoning_item_id_policy(policy);
    }
    let result = Runner::run_with_options(&agent, "go", RunOptions::new(()).run_config(config))
        .await
        .unwrap();
    let reasoning = result
        .new_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::Reasoning { .. }))
        .unwrap();
    assert_eq!(reasoning.to_input_item().0["id"], "rs_123");
    model
        .last_call()
        .unwrap()
        .request
        .input
        .into_iter()
        .map(|i| i.0)
        .find(|i| i.get("type") == Some(&json!("reasoning")))
        .unwrap()
}

#[tokio::test]
async fn run_item_to_input_item_preserves_reasoning_item_ids_by_default() {
    let replayed = replayed_reasoning(None).await;
    assert_eq!(replayed["type"], "reasoning");
    assert_eq!(replayed["id"], "rs_123");
}

#[tokio::test]
async fn run_item_to_input_item_omits_reasoning_item_ids_when_configured() {
    let replayed = replayed_reasoning(Some(ReasoningItemIdPolicy::Omit)).await;
    assert_eq!(replayed["type"], "reasoning");
    assert!(replayed.get("id").is_none());
}

#[test]
fn run_item_to_input_item_preserves_tool_search_items() {
    let call = RunItem::new(
        "A",
        RunItemKind::ToolCall {
            raw_item: json!({"type": "tool_search_call", "queries": [{"search_term": "profile"}]}),
        },
    );
    let output = RunItem::new(
        "A",
        RunItemKind::ToolCallOutput {
            raw_item: json!({"type": "tool_search_output", "results": [{"text": "Customer profile"}]}),
            output: Value::Null,
            custom_data: None,
        },
    );
    assert_eq!(call.to_input_item().0["type"], "tool_search_call");
    assert_eq!(output.to_input_item().0["type"], "tool_search_output");
}

#[test]
fn run_item_to_input_item_strips_tool_search_created_by() {
    let call = RunItem::new(
        "A",
        RunItemKind::ToolCall {
            raw_item: json!({"id": "tsc_123", "type": "tool_search_call", "arguments": {"query": "profile"},
                             "execution": "client", "status": "completed", "created_by": "server"}),
        },
    );
    let output = RunItem::new(
        "A",
        RunItemKind::ToolCallOutput {
            raw_item: json!({"id": "tso_123", "type": "tool_search_output", "execution": "client",
                             "status": "completed", "tools": [], "created_by": "server"}),
            output: Value::Null,
            custom_data: None,
        },
    );
    let call = call.to_input_item().0;
    let output = output.to_input_item().0;
    assert_eq!(call["type"], "tool_search_call");
    assert!(call.get("created_by").is_none());
    assert_eq!(output["type"], "tool_search_output");
    assert!(output.get("created_by").is_none());
}

#[test]
fn run_item_to_input_item_strips_function_call_created_by() {
    let call = RunItem::new(
        "A",
        RunItemKind::ToolCall {
            raw_item: json!({"id": "fc_1", "arguments": "{}", "call_id": "call_1", "name": "lookup",
                             "type": "function_call", "created_by": "server"}),
        },
    );
    let converted = call.to_input_item().0;
    assert_eq!(converted["type"], "function_call");
    assert!(converted.get("created_by").is_none());
}

#[test]
fn run_item_to_input_item_omits_tool_call_metadata() {
    let call = RunItem::new(
        "A",
        RunItemKind::ToolCall {
            raw_item: json!({"id": "fc_123", "call_id": "call_123", "name": "lookup_account",
                             "arguments": "{}", "type": "function_call", "status": "completed"}),
        },
    )
    .with_tool_details(
        Some("Lookup customer records.".into()),
        Some("Lookup Account".into()),
    );
    let converted = call.to_input_item().0;
    assert_eq!(converted["type"], "function_call");
    assert!(converted.get("description").is_none());
    assert!(converted.get("title").is_none());
}

#[tokio::test]
async fn run_result_to_input_list_preserves_tool_search_items() {
    disable_tracing();
    let model = scripted(vec![vec![
        OutputItem::from_value(
            json!({"type": "tool_search_call", "id": "tsc_1", "call_id": "ts_1",
                                      "arguments": {"query": "profile"}, "execution": "server",
                                      "status": "completed"}),
        ),
        OutputItem::from_value(
            json!({"type": "tool_search_output", "id": "tso_1", "call_id": "ts_1",
                                      "execution": "server", "status": "completed", "tools": []}),
        ),
        assistant_message("done"),
    ]]);
    let agent = Agent::<()>::new("A").with_model_instance(model);
    let result = Runner::run(&agent, "Find CRM tools").await.unwrap();
    let input = result.to_input_list();
    assert_eq!(input.len(), 4);
    assert_eq!(input[1].0["type"], "tool_search_call");
    assert_eq!(input[2].0["type"], "tool_search_output");
}
