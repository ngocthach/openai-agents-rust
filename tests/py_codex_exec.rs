//! Parity port of the Python SDK's `tests/extensions/experiemental/codex/test_codex_exec_thread.py`
//! and `test_payloads.py`.
//!
//! The Codex CLI is faked with a POSIX shell script, so these tests only run on Unix.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use openai_agents_sdk::extensions::codex::{
    ApprovalMode, Codex, CodexExec, CodexExecArgs, CodexOptions, ModelReasoningEffort,
    OutputSchemaFile, SandboxMode, ThreadEvent, ThreadItem, ThreadOptions, TodoItem, TurnOptions,
    WebSearchMode, build_command_args,
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

const EVENTS: &str = concat!(
    r#"{"type":"thread.started","thread_id":"thread-1"}"#,
    "\n",
    r#"{"type":"item.completed","item":{"id":"agent-1","type":"agent_message","text":"Codex finished."}}"#,
    "\n",
    r#"{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":1,"output_tokens":5}}"#,
    "\n",
);

/// A temp directory holding the fixture's recordings and canned event files.
struct FakeCodex {
    dir: TempDir,
    env: BTreeMap<String, String>,
}

impl FakeCodex {
    fn new() -> Self {
        Self::with_events(EVENTS)
    }

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

    /// Write `bytes` verbatim to the fixture's stderr, preserving non-UTF-8 sequences.
    fn with_stderr(mut self, bytes: &[u8]) -> Self {
        let path = self.dir.path().join("stderr.bin");
        std::fs::write(&path, bytes).expect("stderr file");
        self.env.insert(
            "FAKE_CODEX_STDERR_FILE".to_string(),
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

    /// The value the fixture was given for `flag` on `turn`.
    fn argv_value(&self, turn: usize, flag: &str) -> Option<String> {
        let argv = self.argv(turn);
        let index = argv.iter().position(|arg| arg == flag)?;
        argv.get(index + 1).cloned()
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

// ---------------------------------------------------------------------------
// output_schema_file
// ---------------------------------------------------------------------------

/// Port of `test_output_schema_file_none_schema`: no schema means no `--output-schema` flag.
#[tokio::test]
async fn output_schema_file_none_schema() {
    let fake = FakeCodex::new();
    fake.client()
        .start_thread(None)
        .run("hello", Some(TurnOptions::new()))
        .await
        .expect("turn");
    assert!(!fake.argv(1).contains(&"--output-schema".to_string()));
}

/// Port of `test_output_schema_file_rejects_non_object`.
#[test]
fn output_schema_file_rejects_non_object() {
    let error = OutputSchemaFile::create(&json!(["not", "an", "object"])).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("output_schema must be a plain JSON object"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// Codex client
// ---------------------------------------------------------------------------

/// Port of `test_codex_start_and_resume_thread`.
#[test]
fn codex_start_and_resume_thread() {
    let codex = Codex::new(CodexOptions::new().with_codex_path_override("/bin/codex"));
    let thread = codex.start_thread(Some(ThreadOptions::new().with_model("gpt")));
    assert_eq!(thread.id(), None);
    let resumed = codex.resume_thread("thread-1", Some(ThreadOptions::new().with_model("gpt")));
    assert_eq!(resumed.id().as_deref(), Some("thread-1"));
}

/// Port of `test_codex_init_accepts_stream_limit_kwarg`: the configured limit reaches the client.
#[test]
fn codex_init_accepts_stream_limit_option() {
    let codex = Codex::new(
        CodexOptions::new()
            .with_codex_path_override("/bin/codex")
            .with_codex_subprocess_stream_limit_bytes(123456),
    );
    assert_eq!(codex.executable_path(), "/bin/codex");
    assert_eq!(
        codex.options().codex_subprocess_stream_limit_bytes,
        Some(123456)
    );
}

// ---------------------------------------------------------------------------
// CodexExec argument and environment building
// ---------------------------------------------------------------------------

/// Port of `test_codex_exec_run_builds_command_args_and_env`.
#[test]
fn codex_exec_run_builds_command_args_and_env() {
    let exec = CodexExec::new(
        Some("/bin/codex".into()),
        Some(BTreeMap::from([("FOO".to_string(), "bar".to_string())])),
    );
    let args = CodexExecArgs {
        input: "hello".into(),
        base_url: Some("https://example.com".into()),
        api_key: Some("api-key".into()),
        thread_id: Some("thread-123".into()),
        images: vec!["/tmp/img.png".into()],
        model: Some("gpt-4.1-mini".into()),
        sandbox_mode: Some(SandboxMode::ReadOnly),
        working_directory: Some("/work".into()),
        additional_directories: vec!["/extra-a".into(), "/extra-b".into()],
        skip_git_repo_check: Some(true),
        output_schema_file: Some("/tmp/schema.json".into()),
        model_reasoning_effort: Some(ModelReasoningEffort::High),
        network_access_enabled: Some(true),
        web_search_mode: Some(WebSearchMode::Live),
        approval_policy: Some(ApprovalMode::OnRequest),
        ..Default::default()
    };

    assert_eq!(
        build_command_args(&args),
        vec![
            "exec",
            "--experimental-json",
            "--model",
            "gpt-4.1-mini",
            "--sandbox",
            "read-only",
            "--cd",
            "/work",
            "--add-dir",
            "/extra-a",
            "--add-dir",
            "/extra-b",
            "--skip-git-repo-check",
            "--output-schema",
            "/tmp/schema.json",
            "--config",
            "model_reasoning_effort=\"high\"",
            "--config",
            "sandbox_workspace_write.network_access=true",
            "--config",
            "web_search=\"live\"",
            "--config",
            "approval_policy=\"on-request\"",
            "resume",
            "--image",
            "/tmp/img.png",
            "--",
            "thread-123",
            "-",
        ]
    );

    let env = exec.build_env(&args);
    assert_eq!(env.get("FOO").map(String::as_str), Some("bar"));
    assert_eq!(
        env.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE")
            .map(String::as_str),
        Some("codex_sdk_ts")
    );
    assert_eq!(
        env.get("OPENAI_BASE_URL").map(String::as_str),
        Some("https://example.com")
    );
    assert_eq!(
        env.get("CODEX_API_KEY").map(String::as_str),
        Some("api-key")
    );
}

/// Port of `test_codex_exec_run_treats_option_like_thread_id_as_positional` (both parameters).
#[test]
fn codex_exec_run_treats_option_like_thread_id_as_positional() {
    for images in [Vec::new(), vec!["/tmp/img.png".to_string()]] {
        let args = CodexExecArgs {
            input: "hello".into(),
            thread_id: Some("--thread-option".into()),
            images: images.clone(),
            ..Default::default()
        };
        let mut expected = vec![
            "exec".to_string(),
            "--experimental-json".to_string(),
            "resume".to_string(),
        ];
        if let Some(image) = images.first() {
            expected.push("--image".to_string());
            expected.push(image.clone());
        }
        expected.extend([
            "--".to_string(),
            "--thread-option".to_string(),
            "-".to_string(),
        ]);
        assert_eq!(build_command_args(&args), expected);
    }
}

// ---------------------------------------------------------------------------
// CodexExec process behavior
// ---------------------------------------------------------------------------

/// Port of `test_codex_exec_run_handles_large_single_line_events`: a line far longer than
/// asyncio's 64 KiB default is delivered whole under the SDK's own 8 MiB limit.
#[tokio::test]
async fn codex_exec_run_handles_large_single_line_events() {
    let payload = "x".repeat((1 << 16) + 1);
    let events = format!(
        "{}\n",
        json!({
            "type": "item.completed",
            "item": {"id": "agent-1", "type": "agent_message", "text": payload},
        })
    );
    let fake = FakeCodex::with_events(&events);
    let turn = fake
        .client()
        .start_thread(None)
        .run("hello", None)
        .await
        .expect("turn");
    assert_eq!(turn.final_response, payload);
}

/// Port of `test_codex_exec_run_raises_on_non_zero_exit` (both parameters).
#[tokio::test]
async fn codex_exec_run_raises_on_non_zero_exit() {
    for stderr_text in ["bad", "找不到命令"] {
        let fake = FakeCodex::with_events("")
            .with_env("FAKE_CODEX_EXIT", "2")
            .with_stderr(stderr_text.as_bytes());
        let error = fake
            .client()
            .start_thread(None)
            .run("hello", None)
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("Codex exec exited with code 2: {stderr_text}")
        );
    }
}

/// Port of `test_codex_thread_preserves_exit_error_with_non_utf8_stderr` (both parameters).
#[tokio::test]
async fn codex_thread_preserves_exit_error_with_non_utf8_stderr() {
    // Captured Windows CP950 diagnostic from the npm Codex shim in issue #5185.
    let stderr: &[u8] = b"'\"node\"' \xa4\xa3\xacO\xa4\xba\xb3\xa1\xa9\xce\xa5~\xb3\xa1\xa9R\xa5O\xa1B\xa5i\xb0\xf5\xa6\xe6\xaa\xba\xb5{\xa6\xa1\xa9\xce\xa7\xe5\xa6\xb8\xc0\xc9\xa1C\r\n";

    for streamed in [false, true] {
        let fake = FakeCodex::with_events("")
            .with_env("FAKE_CODEX_EXIT", "1")
            .with_stderr(stderr);
        let thread = fake.client().start_thread(None);
        let error = if streamed {
            let mut stream = thread.run_streamed("hello", None).await.expect("stream");
            let mut last = None;
            while let Some(event) = stream.next_event().await {
                last = Some(event);
            }
            last.expect("a final item").unwrap_err()
        } else {
            thread.run("hello", None).await.unwrap_err()
        };

        let message = error.to_string();
        assert!(
            message.starts_with("Codex exec exited with code 1: '\"node\"' "),
            "unexpected error: {message}"
        );
        assert!(message.contains('\u{fffd}'), "{message}");
        assert!(message.ends_with("\r\n"), "{message:?}");
    }
}

// ---------------------------------------------------------------------------
// Thread
// ---------------------------------------------------------------------------

/// Port of `test_thread_run_streamed_passes_options_and_updates_id`.
#[tokio::test]
async fn thread_run_streamed_passes_options_and_updates_id() {
    let events = concat!(
        r#"{"type":"thread.started","thread_id":"thread-42"}"#,
        "\n",
        r#"{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}"#,
        "\n",
    );
    let fake = FakeCodex::with_events(events);
    let codex = Codex::new(
        fake.options()
            .with_base_url("https://example.com")
            .with_api_key("api-key"),
    );
    let thread = codex.start_thread(Some(
        ThreadOptions::new()
            .with_model("gpt-4.1-mini")
            .with_sandbox_mode(SandboxMode::ReadOnly)
            .with_working_directory("/work")
            .with_skip_git_repo_check(true)
            .with_model_reasoning_effort(ModelReasoningEffort::Low)
            .with_network_access_enabled(false)
            .with_web_search_mode(WebSearchMode::Cached)
            .with_approval_policy(ApprovalMode::OnRequest)
            .with_additional_directories(vec!["/extra".into()]),
    ));

    let mut stream = thread
        .run_streamed(
            vec![
                openai_agents_sdk::extensions::codex::UserInput::text("hello"),
                openai_agents_sdk::extensions::codex::UserInput::local_image("/tmp/a.png"),
            ],
            Some(TurnOptions::new().with_output_schema(json!({"type": "object"}))),
        )
        .await
        .expect("stream");

    let first = stream.next_event().await.expect("event").expect("ok");
    assert_eq!(first.event_type(), "thread.started");
    while stream.next_event().await.is_some() {}
    drop(stream);

    assert_eq!(thread.id().as_deref(), Some("thread-42"));
    let schema_path = fake
        .argv_value(1, "--output-schema")
        .expect("--output-schema is passed");
    // The schema file is owned by the turn and removed once the stream is dropped.
    assert!(!PathBuf::from(&schema_path).exists());

    assert_eq!(
        fake.argv(1),
        vec![
            "exec",
            "--experimental-json",
            "--model",
            "gpt-4.1-mini",
            "--sandbox",
            "read-only",
            "--cd",
            "/work",
            "--add-dir",
            "/extra",
            "--skip-git-repo-check",
            "--output-schema",
            &schema_path,
            "--config",
            "model_reasoning_effort=\"low\"",
            "--config",
            "sandbox_workspace_write.network_access=false",
            "--config",
            "web_search=\"cached\"",
            "--config",
            "approval_policy=\"on-request\"",
            "--image",
            "/tmp/a.png",
            "-",
        ]
    );
    assert_eq!(fake.stdin(1), "hello");
    let env: BTreeMap<String, String> = read_lines(&fake.record_dir().join("env-1.txt"))
        .into_iter()
        .filter_map(|line| {
            line.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect();
    assert_eq!(
        env.get("OPENAI_BASE_URL").map(String::as_str),
        Some("https://example.com")
    );
    assert_eq!(
        env.get("CODEX_API_KEY").map(String::as_str),
        Some("api-key")
    );
}

/// Port of `test_thread_run_cleans_up_owned_resources` (all three parameters): the owned schema
/// file is removed and the Codex process is killed whether the turn failed or completed.
#[tokio::test]
async fn thread_run_cleans_up_owned_resources() {
    let cases: [(&str, Option<&str>); 3] = [
        (
            r#"{"type":"turn.failed","error":{"message":"boom"}}"#,
            Some("boom"),
        ),
        (
            r#"{"type":"error","message":"boom"}"#,
            Some("Codex stream error: boom"),
        ),
        (
            r#"{"type":"turn.completed","usage":{"input_tokens":2,"cached_input_tokens":1,"output_tokens":3}}"#,
            None,
        ),
    ];

    for (event, error_message) in cases {
        let mut fake = FakeCodex::with_events(&format!("{event}\n"));
        if error_message.is_some() {
            // Keep the subprocess live after its event, so cleanup is explicit rather than
            // the process simply having exited.
            fake = fake.with_env("FAKE_CODEX_SLEEP", "60");
        }
        let thread = fake.client().start_thread(None);
        let options = TurnOptions::new().with_output_schema(json!({"type": "object"}));

        match error_message {
            Some(message) => {
                let error = thread.run("hello", Some(options)).await.unwrap_err();
                assert_eq!(error.to_string(), message);
            }
            None => {
                let turn = thread.run("hello", Some(options)).await.expect("turn");
                assert_eq!(
                    turn.usage,
                    Some(openai_agents_sdk::extensions::codex::Usage {
                        input_tokens: 2,
                        cached_input_tokens: 1,
                        output_tokens: 3,
                    })
                );
            }
        }

        let schema_path = fake
            .argv_value(1, "--output-schema")
            .expect("--output-schema is passed");
        assert!(
            !PathBuf::from(&schema_path).exists(),
            "the schema file should be removed after the turn: {schema_path}"
        );
    }
}

/// Port of `test_thread_run_streamed_raises_on_parse_error`.
#[tokio::test]
async fn thread_run_streamed_raises_on_parse_error() {
    let fake = FakeCodex::with_events("not-json\n");
    let mut stream = fake
        .client()
        .start_thread(None)
        .run_streamed("hello", None)
        .await
        .expect("stream");
    let error = stream.next_event().await.expect("event").unwrap_err();
    assert_eq!(error.to_string(), "Failed to parse event: not-json");
}

/// Port of `test_thread_run_streamed_idle_timeout_sets_signal`: the caller's cancellation token
/// is fired when the Codex stream goes idle.
#[tokio::test]
async fn thread_run_streamed_idle_timeout_sets_signal() {
    let fake = FakeCodex::with_events("").with_env("FAKE_CODEX_SLEEP", "60");
    let cancel = CancellationToken::new();
    let error = fake
        .client()
        .start_thread(None)
        .run(
            "hello",
            Some(
                TurnOptions::new()
                    .with_cancel(cancel.clone())
                    .with_idle_timeout(Duration::from_millis(10)),
            ),
        )
        .await
        .unwrap_err();

    assert_eq!(error.to_string(), "Codex stream idle for 0.01 seconds.");
    assert!(cancel.is_cancelled());
}

/// Port of `test_thread_run_streamed_idle_timeout_creates_signal`: an idle timeout without a
/// caller-supplied token still aborts the turn.
#[tokio::test]
async fn thread_run_streamed_idle_timeout_creates_signal() {
    let fake = FakeCodex::with_events("").with_env("FAKE_CODEX_SLEEP", "60");
    let error = fake
        .client()
        .start_thread(None)
        .run(
            "hello",
            Some(TurnOptions::new().with_idle_timeout(Duration::from_millis(10))),
        )
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Codex stream idle for 0.01 seconds.");
}

// ---------------------------------------------------------------------------
// test_payloads.py
// ---------------------------------------------------------------------------

/// Port of `test_dict_like_as_dict_recursively_converts_nested_dataclasses`.
#[test]
fn item_as_dict_recursively_converts_nested_items() {
    let item = ThreadItem::TodoList {
        id: "todo-list-1".into(),
        items: vec![
            TodoItem {
                text: "write tests".into(),
                completed: true,
            },
            TodoItem {
                text: "run tests".into(),
                completed: false,
            },
        ],
    };
    let value: Value = item.to_value();
    assert_eq!(
        value,
        json!({
            "id": "todo-list-1",
            "items": [
                {"text": "write tests", "completed": true},
                {"text": "run tests", "completed": false},
            ],
            "type": "todo_list",
        })
    );
}

/// The streamed event carries the parsed item, which is how Rust exposes the payload mapping
/// that `_DictLike` provides in Python.
#[tokio::test]
async fn completed_items_round_trip_through_the_stream() {
    let fake = FakeCodex::new();
    let mut stream = fake
        .client()
        .start_thread(None)
        .run_streamed("hello", None)
        .await
        .expect("stream");
    let mut items: Vec<ThreadItem> = Vec::new();
    while let Some(event) = stream.next_event().await {
        if let ThreadEvent::ItemCompleted { item } = event.expect("ok") {
            items.push(item);
        }
    }
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0].to_value(),
        json!({"type": "agent_message", "id": "agent-1", "text": "Codex finished."})
    );
}
