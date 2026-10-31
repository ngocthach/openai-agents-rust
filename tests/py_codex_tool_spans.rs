//! Span-asserting tests of the Python SDK's `test_codex_tool.py`. They share the global tracing
//! processors, so every test holds `LOCK` and installs its own recorder.
//!
//! The Codex CLI is faked with a POSIX shell script, so these tests only run on Unix.
#![cfg(unix)]
#![allow(dead_code, unused_imports)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::codex::{
    Codex, CodexOptions, CodexToolOptions, ThreadEvent, codex_tool,
};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tracing::{
    Span, Trace, TracingProcessor, function_span, set_trace_processors, trace,
};
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner};
use serde_json::{Value, json};
use tempfile::TempDir;

/// `thread.started` + an agent message + usage, the shape every Python fixture uses.
fn done_events(thread_id: &str) -> String {
    format!(
        "{}\n{}\n{}\n",
        json!({"type": "thread.started", "thread_id": thread_id}),
        json!({
            "type": "item.completed",
            "item": {"id": "agent-1", "type": "agent_message", "text": "Codex done."},
        }),
        json!({
            "type": "turn.completed",
            "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
        }),
    )
}

/// A temp directory holding the fixture's recordings and canned event files.
struct FakeCodex {
    dir: TempDir,
    env: BTreeMap<String, String>,
}

impl FakeCodex {
    fn with_events(events: &str) -> Self {
        let dir = TempDir::new().expect("temp dir");
        let record_dir = dir.path().join("record");
        std::fs::create_dir_all(&record_dir).expect("record dir");
        let events_path = dir.path().join("events.jsonl");
        std::fs::write(&events_path, events).expect("events file");

        let env = BTreeMap::from([
            (
                "FAKE_CODEX_RECORD_DIR".to_string(),
                record_dir.to_string_lossy().into_owned(),
            ),
            (
                "FAKE_CODEX_EVENTS".to_string(),
                events_path.to_string_lossy().into_owned(),
            ),
        ]);
        FakeCodex { dir, env }
    }

    fn with_env(mut self, key: &str, value: &str) -> Self {
        self.env.insert(key.to_string(), value.to_string());
        self
    }

    /// Serve `events` on the given 1-based turn instead of the default file.
    fn with_turn_events(mut self, turn: usize, events: &str) -> Self {
        let path = self.dir.path().join(format!("events-{turn}.jsonl"));
        std::fs::write(&path, events).expect("turn events file");
        self.env.insert(
            format!("FAKE_CODEX_EVENTS_{turn}"),
            path.to_string_lossy().into_owned(),
        );
        self
    }

    fn options(&self) -> CodexOptions {
        CodexOptions::new()
            .with_codex_path_override(fixture_path().to_string_lossy().into_owned())
            .with_api_key("sk-test")
            .with_env(self.env.clone())
    }

    /// Process options without an explicit API key, so the key is resolved from the environment.
    fn options_without_api_key(&self) -> CodexOptions {
        CodexOptions::new()
            .with_codex_path_override(fixture_path().to_string_lossy().into_owned())
            .with_env(self.env.clone())
    }

    fn client(&self) -> Codex {
        Codex::new(self.options())
    }

    fn record_dir(&self) -> PathBuf {
        self.dir.path().join("record")
    }

    fn argv(&self, turn: usize) -> Vec<String> {
        read_lines(&self.record_dir().join(format!("argv-{turn}.txt")))
    }

    fn stdin(&self, turn: usize) -> String {
        std::fs::read_to_string(self.record_dir().join(format!("stdin-{turn}.txt")))
            .expect("recorded stdin")
    }

    fn env_record(&self, turn: usize) -> BTreeMap<String, String> {
        read_lines(&self.record_dir().join(format!("env-{turn}.txt")))
            .into_iter()
            .filter_map(|line| {
                line.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
            })
            .collect()
    }

    fn schema(&self, turn: usize) -> Value {
        let bytes = std::fs::read(self.record_dir().join(format!("schema-{turn}.json")))
            .expect("recorded schema");
        serde_json::from_slice(&bytes).expect("schema json")
    }

    fn turns(&self) -> usize {
        std::fs::read_dir(self.record_dir())
            .expect("record dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("argv-"))
            .count()
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/py_codex_support/fake_codex.sh")
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .lines()
        .map(str::to_string)
        .collect()
}

/// The tool arguments Python's fixtures send.
fn text_arguments(text: &str) -> String {
    json!({"inputs": [{"type": "text", "text": text, "path": ""}]}).to_string()
}

/// Invoke a tool the way `on_invoke_tool` is called directly in Python.
async fn invoke<C: Send + Sync + 'static>(
    tool: &FunctionTool<C>,
    context: RunContext<C>,
    arguments: &str,
) -> openai_agents_sdk::error::Result<Value> {
    let ctx = ToolContext::direct(context, "test", &tool.name, arguments);
    let output = tool.invoke(ctx, arguments.to_string()).await?;
    Ok(serde_json::from_str(&output.to_model_string()).expect("tool result json"))
}

fn thread_id_context() -> Arc<Mutex<Option<String>>> {
    Arc::new(Mutex::new(None))
}

/// A tool that keeps its thread id on the run context, as `use_run_context_thread_id` does.
fn with_store(
    options: CodexToolOptions<Mutex<Option<String>>>,
) -> CodexToolOptions<Mutex<Option<String>>> {
    options.with_thread_id_store(
        |context: &Mutex<Option<String>>| context.lock().expect("poisoned").clone(),
        |context: &Mutex<Option<String>>, thread_id: &str| {
            *context.lock().expect("poisoned") = Some(thread_id.to_string());
        },
    )
}

fn tool_outputs(items: &[openai_agents_sdk::items::RunItem]) -> Vec<Value> {
    items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect()
}

/// Serializes the tests that share the global trace processor. Async-aware because the guard is
/// held across awaits.
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Records the export of every span at its start and end, like Python's `SPAN_PROCESSOR_TESTING`
/// plus the `on_span_start` snapshot hook.
#[derive(Debug, Default)]
struct Recorder {
    starts: Mutex<Vec<Value>>,
    ends: Mutex<Vec<Value>>,
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, _trace: &Trace) {}
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, span: &Span) {
        if let Some(export) = span.export() {
            self.starts.lock().expect("poisoned").push(export);
        }
    }
    fn on_span_end(&self, span: &Span) {
        if let Some(export) = span.export() {
            self.ends.lock().expect("poisoned").push(export);
        }
    }
}

fn install() -> Arc<Recorder> {
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    let recorder = Arc::new(Recorder::default());
    set_trace_processors(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
    recorder
}

fn custom_spans(spans: &[Value]) -> Vec<&Value> {
    spans
        .iter()
        .filter(|s| {
            s["span_data"]["type"] == "custom"
                && s["span_data"]["name"] == "Codex command execution"
        })
        .collect()
}

fn command_events(command: &str, outputs: [&str; 3], failed: bool) -> String {
    let mut lines = Vec::new();
    for (index, event_type) in ["item.started", "item.updated", "item.completed"]
        .iter()
        .enumerate()
    {
        let status = if index == 2 {
            if failed { "failed" } else { "completed" }
        } else {
            "in_progress"
        };
        let mut item = json!({
            "id": "command-1", "type": "command_execution", "command": command,
            "aggregated_output": outputs[index], "status": status,
        });
        if index == 2 {
            item["exit_code"] = json!(if failed { 1 } else { 0 });
        }
        lines.push(json!({"type": event_type, "item": item}).to_string());
    }
    lines.push(
        json!({"type": "item.completed",
               "item": {"id": "message-1", "type": "agent_message", "text": "SYNTHETIC_APPLICATION_RESPONSE"}})
        .to_string(),
    );
    lines.join("\n") + "\n"
}

/// Port of `test_codex_tool_streams_events_and_updates_usage`.
#[tokio::test]
async fn codex_tool_streams_events_and_updates_usage() {
    let _guard = LOCK.lock().await;
    let recorder = install();
    let events = [
        json!({"type": "thread.started", "thread_id": "thread-1"}),
        json!({"type": "turn.started"}),
        json!({"type": "item.started", "item": {"id": "reason-1", "type": "reasoning", "text": "Initial reasoning"}}),
        json!({"type": "item.updated", "item": {"id": "reason-1", "type": "reasoning", "text": "Refined reasoning"}}),
        json!({"type": "item.completed", "item": {"id": "reason-1", "type": "reasoning", "text": "Final reasoning"}}),
        json!({"type": "item.started", "item": {"id": "cmd-1", "type": "command_execution", "command": "pytest", "aggregated_output": "", "status": "in_progress"}}),
        json!({"type": "item.updated", "item": {"id": "cmd-1", "type": "command_execution", "command": "pytest", "aggregated_output": "Running tests", "status": "in_progress"}}),
        json!({"type": "item.completed", "item": {"id": "cmd-1", "type": "command_execution", "command": "pytest", "aggregated_output": "All good", "exit_code": 0, "status": "completed"}}),
        json!({"type": "item.started", "item": {"id": "mcp-1", "type": "mcp_tool_call", "server": "gitmcp", "tool": "search_codex_code", "arguments": {"query": "foo"}, "status": "in_progress"}}),
        json!({"type": "item.completed", "item": {"id": "mcp-1", "type": "mcp_tool_call", "server": "gitmcp", "tool": "search_codex_code", "arguments": {"query": "foo"}, "status": "completed", "result": {"content": [], "structured_content": null}}}),
        json!({"type": "item.completed", "item": {"id": "agent-1", "type": "agent_message", "text": "Codex finished."}}),
        json!({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 1, "output_tokens": 5}}),
    ]
    .iter()
    .map(Value::to_string)
    .collect::<Vec<_>>()
    .join("\n")
        + "\n";
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let arguments = text_arguments("Diagnose failure");
    let ctx = ToolContext::direct(RunContext::new(()), "test", &tool.name, &arguments);

    let usage_ctx = ctx.clone();
    trace("codex-test")
        .run(async {
            function_span(&tool.name, None)
                .run(async {
                    let output = tool
                        .invoke(ctx, arguments.clone())
                        .await
                        .expect("tool result");
                    let result: Value = serde_json::from_str(&output.to_model_string()).unwrap();
                    assert_eq!(result["thread_id"], "thread-1");
                    assert_eq!(result["response"], "Codex finished.");
                    assert_eq!(
                        result["usage"],
                        json!({"input_tokens": 10, "cached_input_tokens": 1, "output_tokens": 5})
                    );
                })
                .await;
        })
        .await;

    assert_eq!(usage_ctx.usage().total_tokens, 15);
    assert_eq!(usage_ctx.usage().requests, 1);

    let ends = recorder.ends.lock().unwrap().clone();
    let function_span_obj = ends
        .iter()
        .find(|s| s["span_data"]["type"] == "function" && s["span_data"]["name"] == tool.name)
        .expect("function span");
    let customs: Vec<&Value> = ends
        .iter()
        .filter(|s| s["span_data"]["type"] == "custom")
        .collect();
    assert_eq!(customs.len(), 1);
    assert_eq!(customs[0]["parent_id"], function_span_obj["id"]);
    let data = &customs[0]["span_data"]["data"];
    assert_eq!(data["command"], "pytest");
    assert_eq!(data["status"], "completed");
    assert_eq!(data["output"], "All good");
    assert_eq!(data["exit_code"], 0);
}

/// Port of `test_codex_tool_keeps_command_output_when_completed_missing_output`.
#[tokio::test]
async fn codex_tool_keeps_command_output_when_completed_missing_output() {
    let _guard = LOCK.lock().await;
    let recorder = install();
    let events = [
        json!({"type": "thread.started", "thread_id": "thread-1"}),
        json!({"type": "item.started", "item": {"id": "cmd-1", "type": "command_execution", "command": "ls", "aggregated_output": "", "status": "in_progress"}}),
        json!({"type": "item.updated", "item": {"id": "cmd-1", "type": "command_execution", "command": "ls", "aggregated_output": "first output", "status": "in_progress"}}),
        json!({"type": "item.completed", "item": {"id": "cmd-1", "type": "command_execution", "command": "ls", "exit_code": 0, "status": "completed"}}),
        json!({"type": "item.completed", "item": {"id": "agent-1", "type": "agent_message", "text": "Codex finished."}}),
        json!({"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}}),
    ]
    .iter()
    .map(Value::to_string)
    .collect::<Vec<_>>()
    .join("\n")
        + "\n";
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    trace("codex-test")
        .run(async {
            invoke(&tool, RunContext::new(()), &text_arguments("List files"))
                .await
                .expect("tool result");
        })
        .await;
    let ends = recorder.ends.lock().unwrap().clone();
    let spans = custom_spans(&ends);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0]["span_data"]["data"]["output"], "first output");
}

/// Port of `test_codex_tool_handle_item_started_ignores_missing_id` and
/// `test_codex_tool_handle_item_updated_ignores_missing_span`, driven through the event stream.
#[tokio::test]
async fn codex_tool_ignores_item_events_without_a_span() {
    let _guard = LOCK.lock().await;
    let recorder = install();
    let events = [
        json!({"type": "item.started", "item": {"id": "", "type": "command_execution", "command": "ls", "aggregated_output": "", "status": "in_progress"}}),
        json!({"type": "item.updated", "item": {"id": "missing", "type": "reasoning", "text": "hi"}}),
        json!({"type": "item.updated", "item": {"id": "missing", "type": "command_execution", "command": "ls", "aggregated_output": "", "status": "in_progress"}}),
        json!({"type": "item.completed", "item": {"id": "agent-1", "type": "agent_message", "text": "done"}}),
    ]
    .iter()
    .map(Value::to_string)
    .collect::<Vec<_>>()
    .join("\n")
        + "\n";
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    trace("codex-test")
        .run(async {
            let result = invoke(&tool, RunContext::new(()), &text_arguments("hi"))
                .await
                .expect("tool result");
            assert_eq!(result["response"], "done");
        })
        .await;
    let ends = recorder.ends.lock().unwrap().clone();
    assert!(custom_spans(&ends).is_empty());
}

/// Port of `test_codex_command_spans_respect_run_tracing_policy` (all six parameters).
#[tokio::test]
async fn codex_command_spans_respect_run_tracing_policy() {
    let _guard = LOCK.lock().await;
    let command = "SYNTHETIC_PRIVATE_COMMAND";
    let outputs = [
        "SYNTHETIC_START_OUTPUT",
        "SYNTHETIC_UPDATE_OUTPUT",
        "SYNTHETIC_FINAL_OUTPUT",
    ];
    let response = "SYNTHETIC_APPLICATION_RESPONSE";

    for include in [Some(false), Some(true), None] {
        for failed in [false, true] {
            let recorder = install();
            let fake = FakeCodex::with_events(&command_events(command, outputs, failed));
            let seen = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
            let recorded = seen.clone();
            let tool: FunctionTool<()> = codex_tool(
                CodexToolOptions::new()
                    .with_codex(fake.client())
                    .with_on_stream(move |event| {
                        let recorded = recorded.clone();
                        async move {
                            let detail = match &event.event {
                                ThreadEvent::ItemStarted { item }
                                | ThreadEvent::ItemUpdated { item }
                                | ThreadEvent::ItemCompleted { item } => {
                                    item.to_value()["aggregated_output"]
                                        .as_str()
                                        .or(item.as_agent_message())
                                        .unwrap_or_default()
                                        .to_string()
                                }
                                _ => String::new(),
                            };
                            recorded
                                .lock()
                                .unwrap()
                                .push((event.event.event_type().to_string(), detail));
                        }
                    }),
            )
            .unwrap();
            let arguments = text_arguments("Run the test task");

            match include {
                None => {
                    trace("direct-codex")
                        .run(async {
                            let result = invoke(&tool, RunContext::new(()), &arguments)
                                .await
                                .expect("tool result");
                            assert_eq!(result["response"], response);
                        })
                        .await;
                }
                Some(include) => {
                    let model = Arc::new(FakeModel::new());
                    model.add_turn(vec![function_call("codex", arguments.clone())]);
                    model.add_turn(vec![assistant_message("done")]);
                    let agent = Agent::<()>::new("test")
                        .with_tool(tool.clone())
                        .with_model_instance(model.clone());
                    let options = RunOptions::new(())
                        .run_config(RunConfig::new().with_trace_include_sensitive_data(include));
                    let result = Runner::run_with_options(&agent, "Run the test task", options)
                        .await
                        .expect("run");
                    let outs = tool_outputs(&result.new_items);
                    let payload: Value = serde_json::from_str(outs[0].as_str().unwrap()).unwrap();
                    assert_eq!(payload["response"], response);
                }
            }

            let starts = recorder.starts.lock().unwrap().clone();
            let ends = recorder.ends.lock().unwrap().clone();
            assert_eq!(custom_spans(&starts).len(), 1);
            let final_span = custom_spans(&ends)[0].clone();
            let data = &final_span["span_data"]["data"];
            assert_eq!(data["status"], if failed { "failed" } else { "completed" });
            assert_eq!(data["exit_code"], if failed { 1 } else { 0 });
            if include == Some(false) {
                let serialized = serde_json::to_string(&json!([starts, ends])).unwrap();
                for secret in [command, outputs[0], outputs[1], outputs[2], response] {
                    assert!(!serialized.contains(secret), "{secret} leaked into a span");
                }
                assert!(data.get("command").is_none());
                assert!(data.get("output").is_none());
                if failed {
                    assert_eq!(final_span["error"]["data"], json!({"exit_code": 1}));
                }
            } else {
                assert_eq!(
                    custom_spans(&starts)[0]["span_data"]["data"]["command"],
                    command
                );
                if failed {
                    assert_eq!(final_span["error"]["data"]["output"], outputs[2]);
                }
            }
            let callbacks = seen.lock().unwrap().clone();
            let got: Vec<&str> = callbacks[..3].iter().map(|(_, o)| o.as_str()).collect();
            assert_eq!(got, outputs);
            assert_eq!(callbacks.last().unwrap().1, response);
        }
    }
}

/// Port of `test_codex_command_tracing_disabled_with_caller_trace`.
#[tokio::test]
async fn codex_command_tracing_disabled_with_caller_trace() {
    let _guard = LOCK.lock().await;
    let recorder = install();
    let events = format!(
        "{}\n",
        json!({"type": "item.started", "item": {"id": "command-1", "type": "command_execution",
               "command": "SYNTHETIC_COMMAND", "aggregated_output": "SYNTHETIC_OUTPUT", "status": "in_progress"}})
    );
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("codex", text_arguments("Run the task"))]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(tool)
        .with_model_instance(model.clone());
    trace("caller-owned")
        .run(async {
            let options =
                RunOptions::new(()).run_config(RunConfig::new().with_tracing_disabled(true));
            let result = Runner::run_with_options(&agent, "Run the task", options)
                .await
                .expect("run");
            let outs = tool_outputs(&result.new_items);
            let payload: Value = serde_json::from_str(outs[0].as_str().unwrap()).unwrap();
            assert_eq!(payload["response"], "Codex task completed with inputs.");
        })
        .await;
    let starts = recorder.starts.lock().unwrap().clone();
    assert!(custom_spans(&starts).is_empty());
}
