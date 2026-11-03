//! Runner replay, continuation, state, session, handoff and non-function output cases of Python's
//! `tests/test_programmatic_tool_calling.py` for programmatic calls.
//!
//! Server-managed continuation is `previous_response_id` plus an input delta. Python's
//! `previous_response_id` value `resp-789` is the Rust scripted default response id.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::handoffs::default_tool_name;
use openai_agents_sdk::items::{InputItem, OutputItem, RunItemKind};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::run_config::{ModelInputData, RunConfig};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{
    ApplyPatchTool, CustomTool, FunctionTool, ProgrammaticToolCallingTool, ShellTool,
};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunOptions, RunResult, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const PROGRAM_CALL_ID: &str = "call_program";
const FUNCTION_CALL_ID: &str = "call_lookup";

fn program_caller() -> Value {
    json!({"type": "program", "caller_id": PROGRAM_CALL_ID})
}

fn program_item() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "program", "id": "program_item", "call_id": PROGRAM_CALL_ID,
        "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint",
    }))
}

fn lookup_call() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
        "name": "lookup_inventory", "arguments": "{\"sku\":\"A-1\"}",
        "caller": program_caller(),
    }))
}

fn program_output(status: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "program_output", "id": "program_output_item", "call_id": PROGRAM_CALL_ID,
        "result": "{\"sku\":\"A-1\",\"available_units\":42}", "status": status,
    }))
}

fn model(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let m = Arc::new(ScriptedModel::new());
    m.extend(steps.into_iter().map(ScriptedStep::output).collect())
        .unwrap();
    m
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct InventoryOutput {
    sku: String,
    available_units: i64,
}

#[derive(Deserialize, JsonSchema)]
struct SkuArgs {
    sku: String,
}

fn lookup(executed: Arc<AtomicBool>) -> FunctionTool<()> {
    FunctionTool::new(
        "lookup_inventory",
        "",
        move |_ctx: ToolContext<()>, args: SkuArgs| {
            let executed = executed.clone();
            async move {
                executed.store(true, Ordering::SeqCst);
                Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
            }
        },
    )
    .with_allowed_callers(vec!["programmatic".into()])
    .with_output_type::<InventoryOutput>()
    .unwrap()
}

fn function_outputs(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. }
                if raw_item["type"] == "function_call_output" =>
            {
                Some(raw_item.clone())
            }
            _ => None,
        })
        .collect()
}

fn tool_outputs(result: &RunResult<()>) -> usize {
    result
        .new_items
        .iter()
        .filter(|item| matches!(item.kind, RunItemKind::ToolCallOutput { .. }))
        .count()
}

/// Items of a streamed run, from its events.
async fn stream_items(run: &mut openai_agents_sdk::RunResultStreaming<()>) -> Vec<StreamEvent<()>> {
    let mut events = Vec::new();
    let mut stream = Box::pin(run.stream_events());
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }
    drop(stream);
    events
}

fn streamed_function_outputs(events: &[StreamEvent<()>]) -> Vec<Value> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::RunItem {
                item:
                    openai_agents_sdk::RunItem {
                        kind: RunItemKind::ToolCallOutput { raw_item, .. },
                        ..
                    },
                ..
            } if raw_item["type"] == "function_call_output" => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

fn tool_choice_of(request_settings: &ModelSettings) -> Option<String> {
    request_settings
        .tool_choice
        .as_ref()
        .and_then(|choice| match choice {
            openai_agents_sdk::model_settings::ToolChoice::Named(name) => Some(name.clone()),
            _ => None,
        })
}

// ---------------------------------------------------------------------------------------------
// Execution and replay
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn runner_executes_and_replays_programmatic_function_calls() {
    // Python's streamed and non-streamed cases.
    for streamed in [false, true] {
        let m = model(vec![
            vec![program_item(), lookup_call()],
            vec![
                program_output("completed"),
                assistant_message("42 units are available"),
            ],
        ]);
        let executed = Arc::new(AtomicBool::new(false));
        let agent = Agent::new("inventory")
            .with_model_instance(m.clone())
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(lookup(executed.clone()))
            .with_model_settings(
                ModelSettings::default().with_tool_choice("programmatic_tool_calling"),
            );

        let (final_text, outputs) = if streamed {
            let mut run = Runner::run_streamed(agent, "Check inventory");
            let events = stream_items(&mut run).await;
            assert!(events.iter().any(|event| matches!(
                event,
                StreamEvent::RunItem { name: RunItemStreamEventName::ToolCalled, item }
                    if matches!(&item.kind, RunItemKind::ToolCall { raw_item } if raw_item["type"] == "program")
            )));
            run.finish().await.unwrap();
            (
                run.final_output_text().unwrap(),
                streamed_function_outputs(&events),
            )
        } else {
            let result = Runner::run(agent, "Check inventory").await.unwrap();
            (
                result.final_output_text().unwrap(),
                function_outputs(&result),
            )
        };

        assert_eq!(final_text, "42 units are available", "streamed={streamed}");
        assert!(executed.load(Ordering::SeqCst));
        let calls = m.calls();
        assert_eq!(
            tool_choice_of(&calls[0].request.model_settings).as_deref(),
            Some("programmatic_tool_calling")
        );
        assert_eq!(
            tool_choice_of(&calls.last().unwrap().request.model_settings),
            None
        );

        assert_eq!(outputs.len(), 1, "streamed={streamed}");
        assert_eq!(
            serde_json::from_str::<Value>(outputs[0]["output"].as_str().unwrap()).unwrap(),
            json!({"sku": "A-1", "available_units": 42})
        );
        assert_eq!(outputs[0]["caller"], program_caller());

        let replayed = calls
            .last()
            .unwrap()
            .request
            .input
            .iter()
            .map(|item| item.0.clone())
            .find(|v| v["type"] == "function_call_output")
            .expect("replayed output");
        assert_eq!(replayed["caller"], program_caller(), "streamed={streamed}");
    }
}

#[tokio::test]
async fn runner_handles_multiple_pauses_from_one_program() {
    let m = model(vec![
        vec![program_item(), lookup_call()],
        vec![
            program_output("incomplete"),
            OutputItem::from_value(json!({
                "type": "function_call", "id": "function_item_2", "call_id": "call_lookup_2",
                "name": "lookup_inventory", "arguments": "{\"sku\":\"B-2\"}",
                "caller": program_caller(),
            })),
        ],
        vec![
            OutputItem::from_value(json!({
                "type": "program_output", "id": "program_output_item_2",
                "call_id": PROGRAM_CALL_ID, "result": "{\"total\":84}", "status": "completed",
            })),
            assistant_message("84 units are available"),
        ],
    ]);
    let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        move |_ctx: ToolContext<()>, args: SkuArgs| {
            let recorded = recorded.clone();
            async move {
                recorded.lock().unwrap().push(args.sku.clone());
                Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
            }
        },
    )
    .with_allowed_callers(vec!["programmatic".into()]);
    let agent = Agent::new("inventory")
        .with_model_instance(m.clone())
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(tool)
        .with_model_settings(
            ModelSettings::default().with_tool_choice("programmatic_tool_calling"),
        );

    let result = Runner::run(agent, "Check two SKUs").await.unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("84 units are available")
    );
    assert_eq!(
        *calls.lock().unwrap(),
        vec!["A-1".to_string(), "B-2".to_string()]
    );
    assert_eq!(
        tool_choice_of(&m.calls().last().unwrap().request.model_settings),
        None
    );
    assert_eq!(tool_outputs(&result), 4);
}

#[tokio::test]
async fn runner_executes_programmatic_batch_calls_concurrently() {
    let calls: Vec<OutputItem> = (0..9)
        .map(|index| {
            OutputItem::from_value(json!({
                "type": "function_call", "id": format!("function_item_{index}"),
                "call_id": format!("call_lookup_{index}"), "name": "lookup_inventory",
                "arguments": json!({"sku": format!("SKU-{index}")}).to_string(),
                "caller": program_caller(),
            }))
        })
        .collect();
    let mut first = vec![program_item()];
    first.extend(calls);
    let m = model(vec![
        first,
        vec![
            program_output("completed"),
            assistant_message("batch complete"),
        ],
    ]);

    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let (a, mx) = (active.clone(), max_active.clone());
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        move |_ctx: ToolContext<()>, args: SkuArgs| {
            let (a, mx) = (a.clone(), mx.clone());
            async move {
                let now = a.fetch_add(1, Ordering::SeqCst) + 1;
                mx.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                a.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
            }
        },
    )
    .with_allowed_callers(vec!["programmatic".into()]);
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(tool);

    let result = Runner::run(agent, "Check nine SKUs").await.unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("batch complete")
    );
    assert_eq!(result.raw_responses.len(), 2);
    let outputs = function_outputs(&result);
    assert_eq!(outputs.len(), 9);
    assert_eq!(max_active.load(Ordering::SeqCst), 9);
    assert!(outputs.iter().all(|o| o["caller"] == program_caller()));
}

// ---------------------------------------------------------------------------------------------
// Server-managed continuation
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn previous_response_id_continuation_sends_only_program_function_output() {
    let m = model(vec![
        vec![program_item(), lookup_call()],
        vec![program_output("completed"), assistant_message("done")],
    ]);
    let agent = Agent::new("inventory")
        .with_model_instance(m.clone())
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(lookup(Arc::new(AtomicBool::new(false))));

    let result = Runner::run_with_options(
        agent,
        "Check inventory",
        RunOptions::new(()).auto_previous_response_id(true),
    )
    .await
    .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let last = m.calls().last().unwrap().request.clone();
    assert_eq!(last.previous_response_id.as_deref(), Some("resp-789"));
    assert_eq!(last.input.len(), 1);
    let function_output = &last.input[0].0;
    assert_eq!(function_output["type"], "function_call_output");
    assert_eq!(function_output["caller"], program_caller());
}

#[tokio::test]
async fn previous_response_id_continuation_accepts_server_owned_program_output() {
    // Python's streamed and non-streamed cases: the submitted delta is the program's output.
    for streamed in [false, true] {
        let m = model(vec![vec![
            program_output("completed"),
            assistant_message("done"),
        ]]);
        let agent = Agent::new("inventory")
            .with_model_instance(m.clone())
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(lookup(Arc::new(AtomicBool::new(false))));
        let delta = vec![json!({
            "type": "function_call_output", "call_id": FUNCTION_CALL_ID,
            "output": "{\"sku\":\"A-1\",\"available_units\":42}", "caller": program_caller(),
        })];
        let options = RunOptions::new(()).previous_response_id("response_with_program_parent");

        let final_text = if streamed {
            let mut run = Runner::run_streamed_with_options(agent, delta, options);
            stream_items(&mut run).await;
            run.finish().await.unwrap();
            run.final_output_text().unwrap()
        } else {
            Runner::run_with_options(agent, delta, options)
                .await
                .unwrap()
                .final_output_text()
                .unwrap()
        };

        assert_eq!(final_text, "done", "streamed={streamed}");
        assert_eq!(
            m.calls()
                .last()
                .unwrap()
                .request
                .previous_response_id
                .as_deref(),
            Some("response_with_program_parent")
        );
    }
}

#[tokio::test]
async fn previous_response_id_continuation_accepts_repeated_program_pause() {
    let m = model(vec![
        vec![lookup_call()],
        vec![program_output("completed"), assistant_message("done")],
    ]);
    let executed = Arc::new(AtomicBool::new(false));
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(lookup(executed.clone()));
    let delta = vec![json!({
        "type": "function_call_output", "call_id": "call_previous_lookup",
        "output": "{\"sku\":\"A-0\",\"available_units\":21}", "caller": program_caller(),
    })];

    let result = Runner::run_with_options(
        agent,
        delta,
        RunOptions::new(()).previous_response_id("response_with_program_parent"),
    )
    .await
    .unwrap();

    assert!(executed.load(Ordering::SeqCst));
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(result.raw_responses.len(), 2);
}

#[tokio::test]
async fn run_state_round_trip_preserves_server_owned_program_parent() {
    // Python's two sources of the server-owned parent: a caller on the submitted delta, or an
    // incomplete program output submitted as an item.
    for parent_source in ["caller", "incomplete_program_output"] {
        let m = model(vec![
            vec![lookup_call()],
            vec![program_output("completed"), assistant_message("done")],
        ]);
        let executed = Arc::new(AtomicBool::new(false));
        let agent = Agent::new("inventory")
            .with_model_instance(m.clone())
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(lookup(executed.clone()).with_needs_approval(true));
        let delta: Vec<Value> = if parent_source == "caller" {
            vec![json!({
                "type": "function_call_output", "call_id": "call_previous_lookup",
                "output": "{\"sku\":\"A-0\",\"available_units\":21}", "caller": program_caller(),
            })]
        } else {
            vec![json!({
                "type": "program_output", "id": "program_output_item",
                "call_id": PROGRAM_CALL_ID, "result": "{}", "status": "incomplete",
            })]
        };

        let first = Runner::run_with_options(
            agent.clone(),
            delta,
            RunOptions::new(()).previous_response_id("response_with_program_parent"),
        )
        .await
        .unwrap();
        assert_eq!(first.interruptions.len(), 1, "{parent_source}");

        let json_state = first.to_state().to_json().unwrap();
        let mut restored = RunState::from_json(json_state).unwrap();
        let approval = restored.get_interruptions().remove(0);
        restored.approve(&approval, false);
        let result = Runner::run(agent, restored).await.unwrap();

        assert!(executed.load(Ordering::SeqCst), "{parent_source}");
        assert_eq!(
            result.final_output_text().as_deref(),
            Some("done"),
            "{parent_source}"
        );
        assert_eq!(
            m.calls()
                .last()
                .unwrap()
                .request
                .previous_response_id
                .as_deref(),
            Some("resp-789"),
            "{parent_source}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Session history and handoff summaries
// ---------------------------------------------------------------------------------------------

fn raw_type(item: &Value) -> Option<String> {
    item.get("type").and_then(Value::as_str).map(str::to_string)
}

#[tokio::test]
async fn sqlite_session_round_trip_preserves_program_history_and_caller() {
    let m = model(vec![
        vec![program_item(), lookup_call()],
        vec![program_output("completed"), assistant_message("done")],
    ]);
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(lookup(Arc::new(AtomicBool::new(false))));
    let session = Arc::new(InMemorySession::new("programmatic-tool-calling"));

    let result = Runner::run_with_options(
        agent,
        "Check inventory",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let items: Vec<Value> = session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|item| item.0)
        .collect();
    let types: Vec<Option<String>> = items.iter().map(raw_type).collect();
    assert_eq!(
        types,
        vec![
            None,
            Some("program".into()),
            Some("function_call".into()),
            Some("function_call_output".into()),
            Some("program_output".into()),
            Some("message".into()),
        ]
    );
    let function_output = items
        .iter()
        .find(|item| raw_type(item).as_deref() == Some("function_call_output"))
        .unwrap();
    assert_eq!(function_output["caller"], program_caller());
}

#[tokio::test]
async fn nested_handoff_summarizes_complete_programmatic_transcript() {
    let m = Arc::new(ScriptedModel::new());
    let delegate = Agent::new("delegate").with_model_instance(m.clone());
    let handoff_name = default_tool_name("delegate");
    m.extend(vec![
        ScriptedStep::output(vec![program_item(), lookup_call()]),
        ScriptedStep::output(vec![
            program_output("completed"),
            OutputItem::from_value(json!({
                "type": "function_call", "id": "handoff_item", "call_id": "call_handoff",
                "name": handoff_name, "arguments": "{}",
            })),
        ]),
        ScriptedStep::output(vec![assistant_message("done")]),
    ])
    .unwrap();

    let triage = Agent::new("triage")
        .with_model_instance(m.clone())
        .with_handoff(delegate)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(lookup(Arc::new(AtomicBool::new(false))));

    let captured: Arc<Mutex<Vec<Vec<InputItem>>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let config = RunConfig::default()
        .with_nest_handoff_history(true)
        .with_call_model_input_filter(
            move |data: openai_agents_sdk::run_config::CallModelData<()>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(data.model_data.input.clone());
                    Ok::<ModelInputData, AgentsError>(data.model_data)
                }
            },
        );

    let result = Runner::run_with_options(
        triage,
        "Check inventory and delegate the final response.",
        RunOptions::new(()).run_config(config),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let handoff_input: Vec<Value> = captured
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .iter()
        .map(|item| item.0.clone())
        .collect();
    let types: Vec<Option<String>> = handoff_input.iter().map(raw_type).collect();
    for kind in [
        "program",
        "function_call",
        "function_call_output",
        "program_output",
    ] {
        assert!(
            !types.contains(&Some(kind.to_string())),
            "{kind} in {types:?}"
        );
    }
    let summary_text = handoff_input
        .iter()
        .filter_map(|item| item.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    for kind in [
        "program",
        "function_call",
        "function_call_output",
        "program_output",
    ] {
        assert!(
            summary_text.contains(&format!("\"type\": \"{kind}\"")),
            "{kind} missing from {summary_text}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Non-function programmatic outputs
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn non_function_programmatic_outputs_preserve_caller() {
    let caller = program_caller();
    let shell_call = OutputItem::from_value(json!({
        "id": "shell_item", "call_id": "call_shell", "action": {"commands": ["echo ok"]},
        "status": "completed", "type": "shell_call", "caller": caller.clone(),
    }));
    let custom_call = OutputItem::from_value(json!({
        "id": "custom_item", "call_id": "call_custom", "input": "input", "name": "custom",
        "type": "custom_tool_call", "caller": caller.clone(),
    }));
    let patch_call = OutputItem::from_value(json!({
        "id": "apply_patch_item", "call_id": "call_apply_patch",
        "operation": {"type": "create_file", "path": "example.txt", "diff": "hello"},
        "status": "completed", "type": "apply_patch_call", "caller": caller.clone(),
    }));

    // Shell.
    let m = model(vec![
        vec![program_item(), shell_call],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("tool agent")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(
            ShellTool::new(|_req| async { Ok::<_, AgentsError>("shell done".to_string()) })
                .with_allowed_callers(vec!["programmatic".into()]),
        );
    let result = Runner::run(agent, "Run the tool").await.unwrap();
    assert_eq!(first_tool_output(&result)["caller"], caller);

    // Custom.
    let m = model(vec![
        vec![program_item(), custom_call],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("tool agent")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(
            CustomTool::new(
                "custom",
                "Custom tool",
                |_ctx: ToolContext<()>, _input: String| async {
                    Ok::<_, AgentsError>("custom done".to_string())
                },
            )
            .with_allowed_callers(vec!["programmatic".into()]),
        );
    let result = Runner::run(agent, "Run the tool").await.unwrap();
    assert_eq!(first_tool_output(&result)["caller"], caller);

    // Apply patch.
    let m = model(vec![
        vec![program_item(), patch_call],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("tool agent")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(
            ApplyPatchTool::new(NoopEditor).with_allowed_callers(vec!["programmatic".into()]),
        );
    let result = Runner::run(agent, "Run the tool").await.unwrap();
    assert_eq!(first_tool_output(&result)["caller"], caller);
}

fn first_tool_output(result: &RunResult<()>) -> Value {
    result
        .new_items
        .iter()
        .find_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("tool output")
}

struct NoopEditor;

#[async_trait::async_trait]
impl openai_agents_sdk::editor::ApplyPatchEditor for NoopEditor {
    async fn create_file(
        &self,
        _operation: &openai_agents_sdk::editor::ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<openai_agents_sdk::editor::ApplyPatchResult> {
        Ok(Default::default())
    }
    async fn update_file(
        &self,
        _operation: &openai_agents_sdk::editor::ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<openai_agents_sdk::editor::ApplyPatchResult> {
        Ok(Default::default())
    }
    async fn delete_file(
        &self,
        _operation: &openai_agents_sdk::editor::ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<openai_agents_sdk::editor::ApplyPatchResult> {
        Ok(Default::default())
    }
}
