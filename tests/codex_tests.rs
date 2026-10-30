//! The experimental Codex extension driven by a fake `codex` CLI fixture.
//!
//! The fixture is a POSIX shell script, so these tests only run on Unix.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::extensions::codex::{
    Codex, CodexOptions, CodexToolOptions, OutputSchema, SandboxMode, ThreadEvent, ThreadItem,
    ThreadOptions, TurnOptions, Usage, codex_tool,
};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunContext, RunOptions, Runner};
use serde_json::{Value, json};
use tempfile::TempDir;

const EVENTS: &str = concat!(
    r#"{"type":"thread.started","thread_id":"thread-1"}"#,
    "\n",
    r#"{"type":"turn.started"}"#,
    "\n",
    r#"{"type":"item.started","item":{"id":"cmd-1","type":"command_execution","command":"pytest","aggregated_output":"","status":"in_progress"}}"#,
    "\n",
    r#"{"type":"item.updated","item":{"id":"cmd-1","type":"command_execution","command":"pytest","aggregated_output":"Running","status":"in_progress"}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"id":"cmd-1","type":"command_execution","command":"pytest","aggregated_output":"All good","exit_code":0,"status":"completed"}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"id":"agent-1","type":"agent_message","text":"Codex finished."}}"#,
    "\n",
    r#"{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":1,"output_tokens":5}}"#,
    "\n",
);

const RESUME_EVENTS: &str = concat!(
    r#"{"type":"turn.started"}"#,
    "\n",
    r#"{"type":"item.completed","item":{"id":"agent-2","type":"agent_message","text":"Second turn done."}}"#,
    "\n",
    r#"{"type":"turn.completed","usage":{"input_tokens":2,"cached_input_tokens":0,"output_tokens":3}}"#,
    "\n",
);

/// A temp directory holding the fixture's recordings and canned event files.
struct FakeCodex {
    dir: TempDir,
    env: BTreeMap<String, String>,
}

impl FakeCodex {
    fn new() -> Self {
        Self::with_events(EVENTS, Some(RESUME_EVENTS))
    }

    fn with_events(events: &str, resume_events: Option<&str>) -> Self {
        let dir = TempDir::new().expect("temp dir");
        let record_dir = dir.path().join("record");
        std::fs::create_dir_all(&record_dir).expect("record dir");
        let events_path = dir.path().join("events.jsonl");
        std::fs::write(&events_path, events).expect("events file");

        let mut env = BTreeMap::from([
            (
                "FAKE_CODEX_RECORD_DIR".to_string(),
                record_dir.to_string_lossy().into_owned(),
            ),
            (
                "FAKE_CODEX_EVENTS".to_string(),
                events_path.to_string_lossy().into_owned(),
            ),
        ]);
        if let Some(resume_events) = resume_events {
            let resume_path = dir.path().join("resume-events.jsonl");
            std::fs::write(&resume_path, resume_events).expect("resume events file");
            env.insert(
                "FAKE_CODEX_EVENTS_RESUME".to_string(),
                resume_path.to_string_lossy().into_owned(),
            );
        }
        FakeCodex { dir, env }
    }

    fn with_env(mut self, key: &str, value: &str) -> Self {
        self.env.insert(key.to_string(), value.to_string());
        self
    }

    /// Codex options pointing at the fixture, with a dummy key and no inherited environment.
    fn options(&self) -> CodexOptions {
        CodexOptions::new()
            .with_codex_path_override(fixture_path().to_string_lossy().into_owned())
            .with_api_key("sk-test")
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

    fn turns(&self) -> usize {
        std::fs::read_dir(self.record_dir())
            .expect("record dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("argv-"))
            .count()
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_codex.sh")
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .lines()
        .map(str::to_string)
        .collect()
}

#[tokio::test]
async fn thread_run_maps_options_to_cli_args_and_aggregates_events() {
    let fake = FakeCodex::new();
    let thread = fake.client().start_thread(Some(
        ThreadOptions::new()
            .with_model("gpt-5-codex")
            .with_sandbox_mode(SandboxMode::WorkspaceWrite)
            .with_working_directory("/repo")
            .with_skip_git_repo_check(true)
            .with_network_access_enabled(true)
            .with_additional_directories(vec!["/extra".into()]),
    ));

    let turn = thread.run("Diagnose the failure", None).await.unwrap();

    assert_eq!(turn.final_response, "Codex finished.");
    assert_eq!(
        turn.usage,
        Some(Usage {
            input_tokens: 10,
            cached_input_tokens: 1,
            output_tokens: 5,
        })
    );
    assert_eq!(turn.items.len(), 2);
    assert_eq!(turn.items[0].item_type(), "command_execution");
    assert_eq!(turn.items[1].as_agent_message(), Some("Codex finished."));
    assert_eq!(thread.id().as_deref(), Some("thread-1"));

    assert_eq!(
        fake.argv(1),
        vec![
            "exec",
            "--experimental-json",
            "--model",
            "gpt-5-codex",
            "--sandbox",
            "workspace-write",
            "--cd",
            "/repo",
            "--add-dir",
            "/extra",
            "--skip-git-repo-check",
            "--config",
            "sandbox_workspace_write.network_access=true",
            "-",
        ]
    );
    assert_eq!(fake.stdin(1), "Diagnose the failure");
    let env = fake.env_record(1);
    assert_eq!(
        env.get("CODEX_API_KEY").map(String::as_str),
        Some("sk-test")
    );
    assert_eq!(
        env.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE")
            .map(String::as_str),
        Some("codex_sdk_ts")
    );
}

#[tokio::test]
async fn thread_resumes_with_the_captured_thread_id() {
    let fake = FakeCodex::new();
    let thread = fake.client().start_thread(None);

    let first = thread.run("first", None).await.unwrap();
    assert_eq!(first.final_response, "Codex finished.");
    let second = thread.run("second", None).await.unwrap();
    assert_eq!(second.final_response, "Second turn done.");

    assert!(!fake.argv(1).contains(&"resume".to_string()));
    assert_eq!(
        fake.argv(2),
        vec![
            "exec",
            "--experimental-json",
            "resume",
            "--",
            "thread-1",
            "-"
        ]
    );
}

#[tokio::test]
async fn run_streamed_yields_every_event() {
    let fake = FakeCodex::new();
    let thread = fake.client().start_thread(None);
    let stream = thread.run_streamed("stream it", None).await.unwrap();
    let events: Vec<ThreadEvent> = stream
        .map(|event| event.expect("codex event"))
        .collect()
        .await;

    let types: Vec<&str> = events.iter().map(ThreadEvent::event_type).collect();
    assert_eq!(
        types,
        vec![
            "thread.started",
            "turn.started",
            "item.started",
            "item.updated",
            "item.completed",
            "item.completed",
            "turn.completed",
        ]
    );
    match &events[4] {
        ThreadEvent::ItemCompleted {
            item:
                ThreadItem::CommandExecution {
                    command,
                    status,
                    aggregated_output,
                    exit_code,
                    ..
                },
        } => {
            assert_eq!(command, "pytest");
            assert_eq!(status, "completed");
            assert_eq!(aggregated_output, "All good");
            assert_eq!(*exit_code, Some(0));
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[tokio::test]
async fn structured_input_attaches_images_and_joins_text() {
    let fake = FakeCodex::new();
    let thread = fake.client().start_thread(None);
    thread
        .run(
            vec![
                openai_agents_sdk::extensions::codex::UserInput::text("first"),
                openai_agents_sdk::extensions::codex::UserInput::local_image("/a.png"),
                openai_agents_sdk::extensions::codex::UserInput::text("second"),
            ],
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        fake.argv(1),
        vec!["exec", "--experimental-json", "--image", "/a.png", "-"]
    );
    assert_eq!(fake.stdin(1), "first\n\nsecond");
}

#[tokio::test]
async fn non_zero_exit_fails_the_turn() {
    let fake = FakeCodex::with_events("", None).with_env("FAKE_CODEX_EXIT", "3");
    let thread = fake.client().start_thread(None);
    let error = thread.run("boom", None).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.starts_with("Codex exec exited with code 3:"),
        "unexpected error: {message}"
    );
    assert!(message.contains("fake codex failure"), "{message}");
}

#[tokio::test]
async fn turn_failure_and_stream_errors_are_reported() {
    let fake = FakeCodex::with_events(
        concat!(
            r#"{"type":"thread.started","thread_id":"thread-9"}"#,
            "\n",
            r#"{"type":"turn.failed","error":{"message":"sandbox denied"}}"#,
            "\n",
        ),
        None,
    );
    let error = fake
        .client()
        .start_thread(None)
        .run("do it", None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "sandbox denied");

    let fake = FakeCodex::with_events(r#"{"type":"error","message":"stream broke"}"#, None);
    let error = fake
        .client()
        .start_thread(None)
        .run("do it", None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Codex stream error: stream broke");
}

#[tokio::test]
async fn output_schema_is_written_for_the_turn_and_removed_afterwards() {
    let fake = FakeCodex::new();
    let thread = fake.client().start_thread(None);
    let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
    thread
        .run(
            "structured",
            Some(
                TurnOptions::new()
                    .with_output_schema(OutputSchema::JsonSchema(schema).to_json_schema().unwrap()),
            ),
        )
        .await
        .unwrap();

    let argv = fake.argv(1);
    let schema_flag = argv
        .iter()
        .position(|arg| arg == "--output-schema")
        .expect("--output-schema is passed");
    let schema_path = PathBuf::from(&argv[schema_flag + 1]);
    let recorded: Value = serde_json::from_slice(
        &std::fs::read(fake.record_dir().join("schema-1.json")).expect("recorded schema"),
    )
    .expect("schema json");
    assert_eq!(recorded["additionalProperties"], false);
    assert_eq!(recorded["required"], json!(["answer"]));
    assert!(
        !schema_path.exists(),
        "the temp schema file should be removed after the turn"
    );
}

fn codex_text_call(text: &str) -> openai_agents_sdk::items::OutputItem {
    function_call(
        "codex",
        json!({"inputs": [{"type": "text", "text": text, "path": null}], "thread_id": null})
            .to_string(),
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

#[tokio::test]
async fn codex_tool_runs_end_to_end_through_the_runner() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();

    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_sandbox_mode(SandboxMode::ReadOnly)
            .with_on_stream(move |event| {
                let recorder = recorder.clone();
                async move {
                    recorder
                        .lock()
                        .expect("poisoned")
                        .push(event.event.event_type().to_string());
                }
            }),
    )
    .unwrap();
    assert_eq!(tool.name, "codex");

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![codex_text_call("Diagnose the failure")]);
    model.add_turn(vec![assistant_message("All tests pass now")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "Fix the tests").await.unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("All tests pass now")
    );
    let outputs = tool_outputs(&result.new_items);
    assert_eq!(outputs.len(), 1);
    let payload: Value =
        serde_json::from_str(outputs[0].as_str().expect("tool output string")).unwrap();
    assert_eq!(payload["thread_id"], "thread-1");
    assert_eq!(payload["response"], "Codex finished.");
    assert_eq!(payload["usage"]["input_tokens"], 10);

    // Two model requests (10/5 tokens each) plus the Codex turn (10/5).
    assert_eq!(result.usage().requests, 3);
    assert_eq!(result.usage().total_tokens, 45);
    assert_eq!(result.usage().input_tokens_details.cached_tokens, 1);

    assert_eq!(
        fake.argv(1),
        vec!["exec", "--experimental-json", "--sandbox", "read-only", "-"]
    );
    assert_eq!(fake.stdin(1), "Diagnose the failure");
    assert_eq!(
        seen.lock().expect("poisoned").first().map(String::as_str),
        Some("thread.started")
    );
}

#[tokio::test]
async fn codex_tool_reuses_the_thread_across_calls_when_persisting() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::new();
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_persist_session(true),
    )
    .unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![codex_text_call("first")]);
    model.add_turn(vec![codex_text_call("second")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "Work twice").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    assert_eq!(fake.turns(), 2);
    assert!(!fake.argv(1).contains(&"resume".to_string()));
    assert_eq!(
        fake.argv(2),
        vec![
            "exec",
            "--experimental-json",
            "resume",
            "--",
            "thread-1",
            "-"
        ]
    );
    let outputs = tool_outputs(&result.new_items);
    let second: Value =
        serde_json::from_str(outputs[1].as_str().expect("tool output string")).unwrap();
    assert_eq!(second["thread_id"], "thread-1");
    assert_eq!(second["response"], "Second turn done.");
}

#[tokio::test]
async fn codex_tool_resumes_the_thread_id_from_the_model() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::new();
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call(
        "codex",
        json!({
            "inputs": [{"type": "text", "text": "continue", "path": null}],
            "thread_id": "thread-7",
        })
        .to_string(),
    )]);
    model.add_turn(vec![assistant_message("resumed")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    Runner::run(&agent, "Continue the thread").await.unwrap();

    assert_eq!(
        fake.argv(1),
        vec![
            "exec",
            "--experimental-json",
            "resume",
            "--",
            "thread-7",
            "-"
        ]
    );
}

#[tokio::test]
async fn codex_tool_reports_a_failed_turn_to_the_model() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::with_events(
        concat!(
            r#"{"type":"thread.started","thread_id":"thread-5"}"#,
            "\n",
            r#"{"type":"turn.failed","error":{"message":"sandbox denied"}}"#,
            "\n",
        ),
        None,
    );
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![codex_text_call("break it")]);
    model.add_turn(vec![assistant_message("reported")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "Try").await.unwrap();
    let outputs = tool_outputs(&result.new_items);
    assert_eq!(
        outputs[0].as_str(),
        Some("An error occurred while running the tool. Please try again.")
    );
}

#[tokio::test]
async fn codex_tool_stores_the_thread_id_on_the_run_context() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::new();
    let tool: FunctionTool<Mutex<Option<String>>> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_thread_id_store(
                |context: &Mutex<Option<String>>| context.lock().expect("poisoned").clone(),
                |context: &Mutex<Option<String>>, thread_id: &str| {
                    *context.lock().expect("poisoned") = Some(thread_id.to_string());
                },
            ),
    )
    .unwrap();
    // Run-context mode hides the thread id from the model.
    assert!(
        tool.params_json_schema["properties"]
            .get("thread_id")
            .is_none()
    );

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call(
        "codex",
        json!({"inputs": [{"type": "text", "text": "first", "path": null}]}).to_string(),
    )]);
    model.add_turn(vec![function_call(
        "codex",
        json!({"inputs": [{"type": "text", "text": "second", "path": null}]}).to_string(),
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<Mutex<Option<String>>>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let context = Arc::new(Mutex::new(None));
    let result = Runner::run_with_options(
        &agent,
        "Work twice",
        RunOptions::with_run_context(RunContext::from_arc(context.clone())),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        context.lock().expect("poisoned").as_deref(),
        Some("thread-1")
    );
    assert_eq!(
        fake.argv(2),
        vec![
            "exec",
            "--experimental-json",
            "resume",
            "--",
            "thread-1",
            "-"
        ]
    );
}

#[tokio::test]
async fn a_custom_parameters_schema_may_carry_extra_fields() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::new();
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_params_json_schema(json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "inputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "type": {"type": "string", "enum": ["text"]},
                                "text": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["type", "text", "reason"]
                        }
                    }
                },
                "required": ["inputs"]
            })),
    )
    .unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call(
        "codex",
        json!({"inputs": [{"type": "text", "text": "look", "reason": "because"}]}).to_string(),
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "Use the custom schema").await.unwrap();
    let outputs = tool_outputs(&result.new_items);
    let payload: Value =
        serde_json::from_str(outputs[0].as_str().expect("tool output string")).unwrap();
    assert_eq!(payload["response"], "Codex finished.");
    assert_eq!(fake.stdin(1), "look");
}

#[tokio::test]
async fn the_tool_result_is_byte_identical_to_pythons_json_dumps() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let events = concat!(
        r#"{"type":"thread.started","thread_id":"thread-1"}"#,
        "\n",
        r#"{"type":"item.completed","item":{"id":"agent-1","type":"agent_message","text":"Terminé ✅"}}"#,
        "\n",
        r#"{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":1,"output_tokens":5}}"#,
        "\n",
    );
    let fake = FakeCodex::with_events(events, None);
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![codex_text_call("Translate it")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("Dev")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let result = Runner::run(&agent, "Run codex").await.unwrap();
    let outputs = tool_outputs(&result.new_items);
    assert_eq!(
        outputs[0].as_str().expect("tool output string"),
        "{\"thread_id\": \"thread-1\", \"response\": \"Termin\\u00e9 \\u2705\", \
         \"usage\": {\"input_tokens\": 10, \"cached_input_tokens\": 1, \"output_tokens\": 5}}"
    );
}

#[tokio::test]
async fn a_cancellation_token_kills_the_codex_process() {
    let fake = FakeCodex::with_events(EVENTS, None).with_env("FAKE_CODEX_SLEEP", "60");
    let thread = fake.client().start_thread(None);
    let cancel = tokio_util::sync::CancellationToken::new();
    let mut stream = thread
        .run_streamed("hang", Some(TurnOptions::new().with_cancel(cancel.clone())))
        .await
        .expect("stream");

    assert!(matches!(
        stream.next_event().await.expect("event").expect("ok"),
        ThreadEvent::ThreadStarted { .. }
    ));
    cancel.cancel();

    let mut last = None;
    while let Some(event) = stream.next_event().await {
        last = Some(event);
    }
    let error = last.expect("a final item").unwrap_err();
    assert!(
        error.to_string().starts_with("Codex exec exited with code"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn a_line_over_the_subprocess_stream_limit_fails_the_turn() {
    let events = format!(
        "{}\n{}\n",
        r#"{"type":"thread.started","thread_id":"thread-1"}"#,
        format_args!(
            r#"{{"type":"item.completed","item":{{"id":"a","type":"agent_message","text":"{}"}}}}"#,
            "x".repeat(100_000)
        ),
    );
    let fake = FakeCodex::with_events(&events, None);
    let codex = Codex::new(
        fake.options()
            .with_codex_subprocess_stream_limit_bytes(65536),
    );
    let mut stream = codex
        .start_thread(None)
        .run_streamed("hi", None)
        .await
        .expect("stream");

    assert!(matches!(
        stream.next_event().await.expect("event").expect("ok"),
        ThreadEvent::ThreadStarted { .. }
    ));
    let error = stream.next_event().await.expect("event").unwrap_err();
    assert_eq!(
        error.to_string(),
        "Codex stdout exceeded the 65536 byte subprocess stream limit: Separator is not found, \
         and chunk exceed the limit"
    );
}

#[tokio::test]
async fn an_out_of_range_subprocess_stream_limit_is_rejected() {
    let fake = FakeCodex::new();
    let codex = Codex::new(
        fake.options()
            .with_codex_subprocess_stream_limit_bytes(1024),
    );
    let error = codex
        .start_thread(None)
        .run_streamed("hi", None)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "codex_subprocess_stream_limit_bytes must be between 65536 and 67108864 bytes."
    );
}

#[test]
fn codex_tool_rejects_names_without_the_codex_prefix() {
    let error = codex_tool(CodexToolOptions::<()>::new().with_name("review")).unwrap_err();
    assert!(error.to_string().contains("must be \"codex\""));
}

#[tokio::test]
async fn two_codex_tools_sharing_a_name_fail_fast() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::new();
    // A Codex tool owns per-instance state, so a name collision would silently route calls to
    // the wrong instance. Port of `_validate_codex_tool_name_collisions`.
    let a: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let b: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let agent = Arc::new(Agent::<()>::new("Dev").with_tool(a).with_tool(b));
    let err = agent
        .get_all_tools(&openai_agents_sdk::RunContext::new(()))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Duplicate Codex tool names found: codex"),
        "{err}"
    );
    assert!(err.contains("codex_tool(name=...)"), "{err}");
}

#[tokio::test]
async fn codex_tools_with_distinct_names_are_accepted() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = FakeCodex::new();
    let a: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();
    let b: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_name("codex_review"),
    )
    .unwrap();
    let agent = Arc::new(Agent::<()>::new("Dev").with_tool(a).with_tool(b));
    let tools = agent
        .get_all_tools(&openai_agents_sdk::RunContext::new(()))
        .await
        .unwrap();
    assert_eq!(tools.len(), 2);
}
