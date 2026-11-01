//! Port of `test_codex_tool_persists_thread_id_for_handled_parallel_cancellation`: a Codex turn
//! that is dropped because a sibling tool call failed has already persisted its thread id.
#![cfg(unix)]
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::codex::{Codex, CodexOptions, CodexToolOptions, codex_tool};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{FakeModel, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, RunOptions, Runner};
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

/// A tool that fails after the Codex turn has started, with no failure handler: the run raises.
fn failing_tool() -> FunctionTool<Mutex<Option<String>>> {
    FunctionTool::<Mutex<Option<String>>>::new_no_args("error_tool", "fails", |_ctx| async {
        tokio::time::sleep(Duration::from_millis(800)).await;
        Err::<String, _>(openai_agents_sdk::AgentsError::Other("boom".to_string()))
    })
    .without_failure_error_function()
}

#[tokio::test]
async fn a_turn_dropped_by_a_failing_sibling_has_persisted_its_thread_id() {
    let started = format!(
        "{}\n",
        json!({"type": "thread.started", "thread_id": "thread-next"})
    );
    let fake = FakeCodex::with_events(&done_events("thread-next"))
        .with_turn_events(1, &started)
        .with_env("FAKE_CODEX_SLEEP", "60")
        .with_env("FAKE_CODEX_SLEEP_TURN", "1");
    let codex: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new().with_codex(fake.client()),
    ))
    .unwrap();
    let arguments = text_arguments("Continue thread");

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        function_call("codex", arguments.clone()),
        function_call("error_tool", "{}"),
    ]);
    let agent = Agent::<Mutex<Option<String>>>::new("Dev")
        .with_tool(codex.clone())
        .with_tool(failing_tool())
        .with_model_instance(model.clone());
    let context = thread_id_context();

    let error = Runner::run_with_options(
        &agent,
        "Work",
        RunOptions::with_run_context(RunContext::from_arc(context.clone())),
    )
    .await
    .expect_err("the failing sibling raises");
    assert!(error.to_string().contains("boom"), "{error}");

    assert_eq!(
        context.lock().expect("poisoned").as_deref(),
        Some("thread-next")
    );
    assert_eq!(fake.turns(), 1);

    // The next call resumes the thread the dropped turn started.
    let result = invoke(&codex, RunContext::from_arc(context.clone()), &arguments)
        .await
        .expect("tool result");
    assert_eq!(result["thread_id"], "thread-next");
    assert!(fake.argv(2).contains(&"resume".to_string()));
    assert!(fake.argv(2).contains(&"thread-next".to_string()));
}
