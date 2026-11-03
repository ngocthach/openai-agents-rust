//! Parity port of the Python SDK's `tests/extensions/experiemental/codex/test_codex_tool.py`,
//! except the tests that assert on exported spans (see `py_codex_tool_spans.rs`).
//!
//! The Codex CLI is faked with a POSIX shell script, so these tests only run on Unix.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::codex::{
    Codex, CodexOptions, CodexToolOptions, Input, JsonPrimitive, OutputSchema,
    OutputSchemaDescriptor, OutputSchemaField, OutputSchemaProperty, SandboxMode, ThreadOptions,
    TurnOptions, UserInput, codex_tool, codex_tool_params_schema,
};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
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

// ---------------------------------------------------------------------------
// Options, schemas and names
// ---------------------------------------------------------------------------

/// Port of `test_codex_tool_accepts_output_schema_descriptor`.
#[tokio::test]
async fn codex_tool_accepts_output_schema_descriptor() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let descriptor = OutputSchemaDescriptor::new()
        .with_title("Summary")
        .with_property(
            OutputSchemaProperty::new(
                "summary",
                OutputSchemaField::string().with_description("Summary field"),
            )
            .with_description("Short summary"),
        );
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_output_schema(descriptor),
    )
    .unwrap();

    invoke(&tool, RunContext::new(()), &text_arguments("Check schema"))
        .await
        .expect("tool result");

    let schema = fake.schema(1);
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["properties"]["summary"]["type"], "string");
    // The property description wins over the field description, as in Python.
    assert_eq!(
        schema["properties"]["summary"]["description"],
        "Short summary"
    );
    assert_eq!(schema["required"], json!([]));
}

/// Port of `test_codex_tool_resolve_output_schema_validation_errors`.
///
/// Python's fourth case feeds `{"type": "bogus"}` as a property schema; Rust's typed
/// [`OutputSchemaField`] cannot express an unknown primitive, so that case has no analogue.
#[test]
fn codex_tool_resolve_output_schema_validation_errors() {
    let no_properties = OutputSchemaDescriptor::new().to_json_schema().unwrap_err();
    assert!(
        no_properties
            .to_string()
            .contains("must include properties"),
        "{no_properties}"
    );

    let missing_required = OutputSchemaDescriptor::new()
        .with_property(OutputSchemaProperty::new(
            "name",
            OutputSchemaField::string(),
        ))
        .with_required(vec!["missing".into()])
        .to_json_schema()
        .unwrap_err();
    assert!(
        missing_required.to_string().contains("Required property"),
        "{missing_required}"
    );

    let not_an_object = OutputSchema::JsonSchema(json!({"type": "string"}))
        .to_json_schema()
        .unwrap_err();
    assert!(
        not_an_object.to_string().contains("type \"object\""),
        "{not_an_object}"
    );
}

/// Port of `test_codex_tool_resolve_output_schema_does_not_mutate_input`.
#[test]
fn codex_tool_resolve_output_schema_does_not_mutate_input() {
    let nested = json!({"type": "object", "properties": {"y": {"type": "string"}}});
    let option = json!({"type": "object", "properties": {"inner": nested.clone()}});
    let snapshot = option.clone();

    let result = OutputSchema::JsonSchema(option.clone())
        .to_json_schema()
        .expect("schema");

    assert_eq!(option, snapshot);
    assert_eq!(
        nested,
        json!({"type": "object", "properties": {"y": {"type": "string"}}})
    );
    // The resolved copy is strict, which the untouched input is not.
    assert_eq!(result["properties"]["inner"]["additionalProperties"], false);
}

/// Port of `test_codex_tool_resolve_output_schema_descriptor`.
#[test]
fn codex_tool_resolve_output_schema_descriptor() {
    let schema = OutputSchemaDescriptor::new()
        .with_title("Report")
        .with_description("Structured output")
        .with_property(
            OutputSchemaProperty::new(
                "tags",
                OutputSchemaField::array(
                    OutputSchemaField::primitive(JsonPrimitive::String)
                        .with_description("Tag value"),
                )
                .with_description("Tags array"),
            )
            .with_description("Tag list"),
        )
        .with_property(
            OutputSchemaProperty::new("summary", OutputSchemaField::string())
                .with_description("Summary text"),
        )
        .with_required(vec!["tags".into()])
        .to_json_schema()
        .expect("schema");

    assert_eq!(schema["title"], "Report");
    assert_eq!(schema["description"], "Structured output");
    assert_eq!(schema["properties"]["tags"]["type"], "array");
    assert_eq!(schema["properties"]["tags"]["description"], "Tag list");
    assert_eq!(
        schema["properties"]["tags"]["items"]["description"],
        "Tag value"
    );
    assert_eq!(schema["properties"]["tags"]["items"]["type"], "string");
    assert_eq!(schema["required"], json!(["tags"]));
}

/// Port of `test_codex_tool_build_turn_options_merges_output_schema`: the configured output
/// schema reaches the turn, and other turn defaults survive.
#[tokio::test]
async fn codex_tool_build_turn_options_merges_output_schema() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_output_schema(json!({"type": "object", "properties": {}}))
            .with_default_turn_options(
                TurnOptions::new().with_idle_timeout(Duration::from_secs(1)),
            ),
    )
    .unwrap();

    invoke(&tool, RunContext::new(()), &text_arguments("Check schema"))
        .await
        .expect("tool result");

    assert_eq!(
        fake.schema(1),
        json!({"type": "object", "properties": {}, "additionalProperties": false, "required": []})
    );
}

/// Port of `test_codex_tool_build_turn_options_overrides_schema`: the tool's own schema wins
/// over one set in the turn defaults.
#[tokio::test]
async fn codex_tool_build_turn_options_overrides_schema() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_default_turn_options(TurnOptions::new().with_output_schema(
                json!({"type": "object", "properties": {"x": {"type": "string"}}}),
            ))
            .with_output_schema(json!({
                "type": "object", "properties": {}, "additionalProperties": false
            })),
    )
    .unwrap();

    invoke(&tool, RunContext::new(()), &text_arguments("Check schema"))
        .await
        .expect("tool result");

    let schema = fake.schema(1);
    assert_eq!(schema["properties"], json!({}));
}

/// Port of `test_codex_tool_passes_idle_timeout_seconds`: the configured idle timeout is applied
/// to the turn.
#[tokio::test]
async fn codex_tool_passes_idle_timeout_seconds() {
    let fake = FakeCodex::with_events("").with_env("FAKE_CODEX_SLEEP", "60");
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_default_turn_options(
                TurnOptions::new().with_idle_timeout(Duration::from_millis(10)),
            ),
    )
    .unwrap();

    let error = invoke(&tool, RunContext::new(()), &text_arguments("Check timeout"))
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Codex stream idle for 0.01 seconds.");
}

/// Port of `test_codex_tool_allows_non_alnum_suffix_when_run_context_thread_id_disabled`.
#[test]
fn codex_tool_allows_non_alnum_suffix() {
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_name("codex_a-b")).unwrap();
    assert_eq!(tool.name, "codex_a-b");
}

/// Port of `test_codex_tool_accepts_all_keyword_overrides`.
#[test]
fn codex_tool_accepts_all_keyword_overrides() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new()
            .with_name("codex_overrides")
            .with_description("desc")
            .with_params_json_schema(codex_tool_params_schema(false))
            .with_output_schema(json!({
                "type": "object", "properties": {}, "additionalProperties": false
            }))
            .with_codex(fake.client())
            .with_codex_options(CodexOptions::new().with_api_key("from-options"))
            .with_default_thread_options(ThreadOptions::new().with_model("gpt"))
            .with_thread_id("thread-1")
            .with_sandbox_mode(SandboxMode::ReadOnly)
            .with_working_directory("/work")
            .with_skip_git_repo_check(true)
            .with_default_turn_options(TurnOptions::new().with_idle_timeout(Duration::from_secs(1)))
            .with_span_data_max_chars(Some(10))
            .with_include_sensitive_span_data(false)
            .with_persist_session(true)
            .with_on_stream(|_event| async {}),
    ))
    .unwrap();

    assert_eq!(tool.name, "codex_overrides");
    assert_eq!(tool.description, "desc");
}

/// Port of `test_codex_tool_resolve_codex_options_reads_env_override`: the API key is taken from
/// the process-options environment when none is configured.
#[tokio::test]
async fn codex_tool_resolve_codex_options_reads_env_override() {
    let fake =
        FakeCodex::with_events(&done_events("thread-1")).with_env("CODEX_API_KEY", "env-key");
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex_options(fake.options_without_api_key()))
            .unwrap();

    invoke(&tool, RunContext::new(()), &text_arguments("Check key"))
        .await
        .expect("tool result");

    assert_eq!(
        fake.env_record(1).get("CODEX_API_KEY").map(String::as_str),
        Some("env-key")
    );
}

/// Port of `test_codex_tool_defaults_to_openai_api_key`: `OPENAI_API_KEY` is the fallback.
#[tokio::test]
async fn codex_tool_defaults_to_openai_api_key() {
    let fake =
        FakeCodex::with_events(&done_events("thread-1")).with_env("OPENAI_API_KEY", "openai-key");
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex_options(fake.options_without_api_key()))
            .unwrap();

    invoke(
        &tool,
        RunContext::new(()),
        &text_arguments("Check default key"),
    )
    .await
    .expect("tool result");

    assert_eq!(
        fake.env_record(1).get("CODEX_API_KEY").map(String::as_str),
        Some("openai-key")
    );
}

// ---------------------------------------------------------------------------
// Tool input
// ---------------------------------------------------------------------------

/// Port of `test_codex_tool_input_thread_id_validation_errors`.
#[tokio::test]
async fn codex_tool_input_thread_id_validation_errors() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let arguments =
        json!({"inputs": [{"type": "text", "text": "hello", "path": ""}], "thread_id": "   "})
            .to_string();

    let error = invoke(&tool, RunContext::new(()), &arguments)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("non-empty string"), "{error}");
    // Nothing was spawned.
    assert_eq!(fake.turns(), 0);
}

/// Port of `test_codex_tool_build_codex_input_empty`.
#[test]
fn codex_tool_build_codex_input_empty() {
    assert_eq!(Input::Items(Vec::new()).normalize().0, "");
    assert_eq!(
        Input::Items(vec![UserInput::text("hello")]).normalize().0,
        "hello"
    );
}

// ---------------------------------------------------------------------------
// Thread id on the run context
// ---------------------------------------------------------------------------

/// Port of `test_codex_tool_uses_run_context_thread_id_and_persists_latest`.
#[tokio::test]
async fn codex_tool_uses_run_context_thread_id_and_persists_latest() {
    let fake = FakeCodex::with_events(&done_events("thread-next"));
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new().with_codex(fake.client()),
    ))
    .unwrap();

    let context = thread_id_context();
    *context.lock().expect("poisoned") = Some("thread-prev".into());
    let result = invoke(
        &tool,
        RunContext::from_arc(context.clone()),
        &text_arguments("Continue thread"),
    )
    .await
    .expect("tool result");

    assert_eq!(result["thread_id"], "thread-next");
    assert!(fake.argv(1).contains(&"resume".to_string()));
    assert!(fake.argv(1).contains(&"thread-prev".to_string()));
    assert_eq!(
        context.lock().expect("poisoned").as_deref(),
        Some("thread-next")
    );
}

/// Port of `test_codex_tool_persists_thread_started_id_when_thread_object_id_is_none`.
#[tokio::test]
async fn codex_tool_persists_thread_started_id_when_thread_object_id_is_none() {
    let fake = FakeCodex::with_events(&done_events("thread-next"));
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new().with_codex(fake.client()),
    ))
    .unwrap();
    let context = thread_id_context();

    let first = invoke(
        &tool,
        RunContext::from_arc(context.clone()),
        &text_arguments("Continue thread"),
    )
    .await
    .expect("tool result");
    let second = invoke(
        &tool,
        RunContext::from_arc(context.clone()),
        &text_arguments("Continue thread"),
    )
    .await
    .expect("tool result");

    assert_eq!(first["thread_id"], "thread-next");
    assert_eq!(second["thread_id"], "thread-next");
    assert_eq!(
        context.lock().expect("poisoned").as_deref(),
        Some("thread-next")
    );
    assert!(!fake.argv(1).contains(&"resume".to_string()));
    assert_eq!(
        fake.argv(2),
        vec![
            "exec",
            "--experimental-json",
            "resume",
            "--",
            "thread-next",
            "-"
        ]
    );
}

/// Port of `test_codex_tool_persists_thread_id_for_recoverable_turn_failure`: the failure is
/// handed to the model, and the thread id survives for the next call.
#[tokio::test]
async fn codex_tool_persists_thread_id_for_recoverable_turn_failure() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let events = format!(
        "{}\n{}\n",
        json!({"type": "thread.started", "thread_id": "thread-next"}),
        json!({"type": "turn.failed", "error": {"message": "boom"}}),
    );
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new().with_codex(fake.client()),
    ))
    .unwrap()
    .with_failure_error_function(|_ctx, _error| "handled".to_string());

    let model = Arc::new(FakeModel::new());
    let arguments = text_arguments("Continue thread");
    model.add_turn(vec![function_call("codex", arguments.clone())]);
    model.add_turn(vec![function_call("codex", arguments)]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<Mutex<Option<String>>>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let context = thread_id_context();
    let result = Runner::run_with_options(
        &agent,
        "Work twice",
        RunOptions::with_run_context(RunContext::from_arc(context.clone())),
    )
    .await
    .expect("run");

    let outputs = tool_outputs(&result.new_items);
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].as_str(), Some("handled"));
    assert_eq!(outputs[1].as_str(), Some("handled"));
    assert_eq!(
        context.lock().expect("poisoned").as_deref(),
        Some("thread-next")
    );
    assert_eq!(fake.turns(), 2);
    assert!(fake.argv(2).contains(&"thread-next".to_string()));
}

/// Port of `test_codex_tool_persists_thread_id_for_raised_turn_failure`.
#[tokio::test]
async fn codex_tool_persists_thread_id_for_raised_turn_failure() {
    let events = format!(
        "{}\n{}\n",
        json!({"type": "thread.started", "thread_id": "thread-next"}),
        json!({"type": "turn.failed", "error": {"message": "boom"}}),
    );
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new().with_codex(fake.client()),
    ))
    .unwrap();
    let context = thread_id_context();

    for _ in 0..2 {
        let error = invoke(
            &tool,
            RunContext::from_arc(context.clone()),
            &text_arguments("Continue thread"),
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "Codex turn failed: boom");
        assert_eq!(
            context.lock().expect("poisoned").as_deref(),
            Some("thread-next")
        );
    }

    assert!(!fake.argv(1).contains(&"resume".to_string()));
    assert!(fake.argv(2).contains(&"thread-next".to_string()));
}

/// Port of `test_codex_tool_persists_thread_id_for_cancelled_turn`: a turn aborted mid-stream
/// still persists the thread id, and the next call resumes it.
///
/// Python aborts with `asyncio.CancelledError` raised from the event stream; the Rust analogue
/// is the idle timeout, which is the only way a turn is cut short after `thread.started`.
#[tokio::test]
async fn codex_tool_persists_thread_id_for_cancelled_turn() {
    let started = format!(
        "{}\n",
        json!({"type": "thread.started", "thread_id": "thread-next"})
    );
    let fake = FakeCodex::with_events(&done_events("thread-next"))
        .with_turn_events(1, &started)
        .with_env("FAKE_CODEX_SLEEP", "60")
        .with_env("FAKE_CODEX_SLEEP_TURN", "1");
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_default_turn_options(
                TurnOptions::new().with_idle_timeout(Duration::from_millis(1500)),
            ),
    ))
    .unwrap();
    let context = thread_id_context();

    let error = invoke(
        &tool,
        RunContext::from_arc(context.clone()),
        &text_arguments("Continue thread"),
    )
    .await
    .unwrap_err();
    assert!(
        error.to_string().starts_with("Codex stream idle for"),
        "{error}"
    );
    assert_eq!(
        context.lock().expect("poisoned").as_deref(),
        Some("thread-next")
    );

    let result = invoke(
        &tool,
        RunContext::from_arc(context.clone()),
        &text_arguments("Continue thread"),
    )
    .await
    .expect("tool result");
    assert_eq!(result["thread_id"], "thread-next");
    assert!(!fake.argv(1).contains(&"resume".to_string()));
    assert!(fake.argv(2).contains(&"thread-next".to_string()));
}

/// Port of `test_codex_tool_falls_back_to_call_thread_id_when_thread_object_id_is_none`: with no
/// `thread.started` event the thread id of the call is kept and persisted.
#[tokio::test]
async fn codex_tool_falls_back_to_call_thread_id_when_thread_object_id_is_none() {
    let events = format!(
        "{}\n{}\n",
        json!({
            "type": "item.completed",
            "item": {"id": "agent-1", "type": "agent_message", "text": "Codex done."},
        }),
        json!({
            "type": "turn.completed",
            "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
        }),
    );
    let fake = FakeCodex::with_events(&events);
    // Python passes `parameters=CodexToolParameters` to re-expose the hidden `thread_id` field.
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_params_json_schema(codex_tool_params_schema(true)),
    ))
    .unwrap();
    let context = thread_id_context();

    let first_arguments = json!({
        "inputs": [{"type": "text", "text": "Continue thread", "path": ""}],
        "thread_id": "thread-explicit",
    })
    .to_string();
    // The re-exposed schema marks `thread_id` required (strict mode), so the second call sends null.
    let second_arguments = json!({
        "inputs": [{"type": "text", "text": "Continue thread", "path": ""}],
        "thread_id": null,
    })
    .to_string();
    let first = invoke(
        &tool,
        RunContext::from_arc(context.clone()),
        &first_arguments,
    )
    .await
    .expect("tool result");
    let second = invoke(
        &tool,
        RunContext::from_arc(context.clone()),
        &second_arguments,
    )
    .await
    .expect("tool result");

    assert_eq!(first["thread_id"], "thread-explicit");
    assert_eq!(second["thread_id"], "thread-explicit");
    assert_eq!(
        context.lock().expect("poisoned").as_deref(),
        Some("thread-explicit")
    );
    for turn in 1..=2 {
        assert!(fake.argv(turn).contains(&"resume".to_string()));
        assert!(fake.argv(turn).contains(&"thread-explicit".to_string()));
    }
}

/// Port of `test_codex_tool_tool_input_thread_id_overrides_run_context_thread_id`.
#[tokio::test]
async fn codex_tool_tool_input_thread_id_overrides_run_context_thread_id() {
    let fake = FakeCodex::with_events(&done_events("thread-from-tool-input"));
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(with_store(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_params_json_schema(codex_tool_params_schema(true)),
    ))
    .unwrap();

    let context = thread_id_context();
    *context.lock().expect("poisoned") = Some("thread-from-context".into());
    let arguments = json!({
        "inputs": [{"type": "text", "text": "Continue thread", "path": ""}],
        "thread_id": "thread-from-args",
    })
    .to_string();

    invoke(&tool, RunContext::from_arc(context), &arguments)
        .await
        .expect("tool result");

    assert!(fake.argv(1).contains(&"thread-from-args".to_string()));
    assert!(!fake.argv(1).contains(&"thread-from-context".to_string()));
}

// ---------------------------------------------------------------------------
// Name collisions and failure policies
// ---------------------------------------------------------------------------

/// Port of `test_codex_tool_name_collision_with_other_tool_fails_fast`.
#[tokio::test]
async fn codex_tool_name_collision_with_other_tool_fails_fast() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let codex: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let other = FunctionTool::<()>::new_no_args("codex", "Other", |_c| async { Ok("ok") });
    let agent = Arc::new(Agent::<()>::new("test").with_tool(codex).with_tool(other));

    let error = agent
        .get_all_tools(&RunContext::new(()))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Duplicate Codex tool names found"),
        "{error}"
    );
}

/// Port of `test_replaced_codex_tool_preserves_codex_collision_markers`: a renamed clone is
/// still a Codex tool for the duplicate-name check.
#[tokio::test]
async fn replaced_codex_tool_preserves_codex_collision_markers() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let built: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let mut first = built.clone();
    first.name = "shared_codex_tool".into();
    let mut second = built.clone();
    second.name = "shared_codex_tool".into();
    let agent = Arc::new(Agent::<()>::new("test").with_tool(first).with_tool(second));

    let error = agent
        .get_all_tools(&RunContext::new(()))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Duplicate Codex tool names found: shared_codex_tool"),
        "{error}"
    );
}

/// Port of `test_codex_tool_on_invoke_tool_handles_failure_error_function_sync`.
#[tokio::test]
async fn codex_tool_on_invoke_tool_handles_failure_error_function() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<()> = codex_tool(CodexToolOptions::new().with_codex(fake.client()))
        .unwrap()
        .with_failure_error_function(|_ctx, _error| "handled".to_string());

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("codex", "{bad")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "Run codex").await.expect("run");
    let outputs = tool_outputs(&result.new_items);
    assert_eq!(outputs[0].as_str(), Some("handled"));
}

/// Port of `test_codex_tool_on_invoke_tool_raises_without_failure_handler` and
/// `test_replaced_codex_tool_normal_failure_uses_replaced_policy`.
#[tokio::test]
async fn codex_tool_without_failure_handler_raises() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let built: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    // Directly, the invalid arguments surface as a model behavior error.
    let error = invoke(&built, RunContext::new(()), "{bad")
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("Invalid JSON input for codex tool"),
        "{error}"
    );

    // A clone whose failure policy was replaced re-raises instead of answering the model.
    let replaced = built.clone().without_failure_error_function();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("codex", "{bad")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(replaced)
        .with_model_instance(model.clone());

    let error = Runner::run(&agent, "Run codex").await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Invalid JSON input for codex tool"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// Event consumption
// ---------------------------------------------------------------------------

/// Port of `test_codex_tool_consume_events_default_response`: a turn with no agent message
/// answers with the default text.
#[tokio::test]
async fn codex_tool_consume_events_default_response() {
    let events = format!(
        "{}\n{}\n",
        json!({"type": "thread.started", "thread_id": "thread-1"}),
        json!({
            "type": "turn.completed",
            "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
        }),
    );
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    let result = invoke(&tool, RunContext::new(()), &text_arguments("hello"))
        .await
        .expect("tool result");
    assert_eq!(result["response"], "Codex task completed with inputs.");
    assert_eq!(
        result["usage"],
        json!({"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1})
    );
    assert_eq!(result["thread_id"], "thread-1");
}

/// Port of `test_codex_tool_consume_events_turn_failed`.
#[tokio::test]
async fn codex_tool_consume_events_turn_failed() {
    let events = format!(
        "{}\n",
        json!({"type": "turn.failed", "error": {"message": "boom"}})
    );
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    let error = invoke(&tool, RunContext::new(()), &text_arguments("hello"))
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Codex turn failed: boom");
}

/// Port of `test_codex_tool_consume_events_error_event`.
#[tokio::test]
async fn codex_tool_consume_events_error_event() {
    let events = format!("{}\n", json!({"type": "error", "message": "boom"}));
    let fake = FakeCodex::with_events(&events);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    let error = invoke(&tool, RunContext::new(()), &text_arguments("hello"))
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Codex stream error: boom");
}

/// Port of `test_codex_tool_consume_events_with_on_stream_error`: a stream handler that blows up
/// never fails the turn.
///
/// Python's handler raises `RuntimeError`; the Rust handler returns `()`, so the analogue is a
/// panic inside the dispatcher task.
#[tokio::test]
async fn codex_tool_consume_events_with_on_stream_error() {
    let events = format!(
        "{}\n{}\n{}\n",
        json!({"type": "thread.started", "thread_id": "thread-1"}),
        json!({
            "type": "item.started",
            "item": {"id": "cmd-1", "type": "command_execution", "command": "ls",
                     "aggregated_output": "", "status": "in_progress"},
        }),
        json!({
            "type": "item.completed",
            "item": {"id": "agent-1", "type": "agent_message", "text": "done"},
        }),
    );
    let fake = FakeCodex::with_events(&events);
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let recorder = seen.clone();
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_on_stream(move |event| {
                let recorder = recorder.clone();
                async move {
                    let event_type = event.event.event_type().to_string();
                    recorder.lock().expect("poisoned").push(event_type.clone());
                    if event_type == "item.started" {
                        panic!("SECRET_CODEX_STREAM_PAYLOAD");
                    }
                }
            }),
    )
    .unwrap();

    let result = invoke(&tool, RunContext::new(()), &text_arguments("hello"))
        .await
        .expect("tool result");

    assert_eq!(result["response"], "done");
    assert_eq!(result["thread_id"], "thread-1");
    assert!(
        seen.lock()
            .expect("poisoned")
            .contains(&"item.started".to_string())
    );
}

/// Port of `test_codex_tool_streaming_drains_events_before_stream_error`: the terminal event is
/// handed to a slow handler before the turn error is raised.
#[tokio::test]
async fn codex_tool_streaming_drains_events_before_stream_error() {
    let events = format!(
        "{}\n{}\n",
        json!({"type": "turn.started"}),
        json!({"type": "turn.failed", "error": {"message": "boom"}}),
    );
    let fake = FakeCodex::with_events(&events);
    let handled = Arc::new(Mutex::new(Vec::<String>::new()));
    let recorder = handled.clone();
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_on_stream(move |event| {
                let recorder = recorder.clone();
                async move {
                    // The first handler is slow, so the terminal event queues behind it.
                    if recorder.lock().expect("poisoned").is_empty() {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    recorder
                        .lock()
                        .expect("poisoned")
                        .push(event.event.event_type().to_string());
                }
            }),
    )
    .unwrap();

    let error = invoke(&tool, RunContext::new(()), &text_arguments("hello"))
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Codex turn failed: boom");
    assert_eq!(
        *handled.lock().expect("poisoned"),
        vec!["turn.started".to_string(), "turn.failed".to_string()]
    );
}

/// The tool joins the structured inputs the way Python's `_build_codex_input` does.
#[tokio::test]
async fn codex_tool_joins_structured_inputs() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let arguments = json!({"inputs": [
        {"type": "text", "text": "first", "path": ""},
        {"type": "local_image", "path": "/a.png", "text": null},
        {"type": "text", "text": "second", "path": ""},
    ]})
    .to_string();

    invoke(&tool, RunContext::new(()), &arguments)
        .await
        .expect("tool result");

    assert_eq!(fake.stdin(1), "first\n\nsecond");
    assert_eq!(
        fake.argv(1),
        vec!["exec", "--experimental-json", "--image", "/a.png", "-"]
    );
}

/// Port of `test_codex_tool_input_item_validation_errors` (all four parameters).
#[tokio::test]
async fn codex_tool_input_item_validation_errors() {
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let cases = [
        (
            json!({"type": "text", "text": "", "path": ""}),
            "non-empty \"text\"",
        ),
        (
            json!({"type": "text", "text": "hello", "path": "x"}),
            "\"path\" is not allowed",
        ),
        (
            json!({"type": "local_image", "path": ""}),
            "non-empty \"path\"",
        ),
        (
            json!({"type": "local_image", "path": "img.png", "text": "hi"}),
            "\"text\" is not allowed",
        ),
    ];
    for (payload, message) in cases {
        let arguments = json!({"inputs": [payload]}).to_string();
        let error = invoke(&tool, RunContext::new(()), &arguments)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(message), "{message}: {error}");
    }
    assert_eq!(fake.turns(), 0);
}

/// Port of `test_codex_tool_argument_errors_respect_tool_data_redaction` (all four parameters).
#[tokio::test]
async fn codex_tool_argument_errors_respect_tool_data_redaction() {
    let secret = "SECRET_CODEX_TOOL_ARGUMENT_123";
    let fake = FakeCodex::with_events(&done_events("thread-1"));
    let tool: FunctionTool<()> = codex_tool(CodexToolOptions::new().with_codex(fake.client()))
        .unwrap()
        .without_failure_error_function();
    let inputs = [
        format!("{{\"inputs\": \"{secret}\"}}"),
        format!("not valid json {secret}"),
    ];
    for redact in [true, false] {
        openai_agents_sdk::debug::set_dont_log_tool_data(redact);
        for arguments in &inputs {
            let error = invoke(&tool, RunContext::new(()), arguments)
                .await
                .unwrap_err();
            let message = error.to_string();
            if redact {
                assert_eq!(message, "Invalid JSON input for codex tool");
            } else {
                assert!(
                    message.starts_with("Invalid JSON input for codex tool: "),
                    "{message}"
                );
            }
            if redact {
                assert!(!message.contains(secret));
            }
        }
    }
    openai_agents_sdk::debug::reset_dont_log_tool_data();
}
