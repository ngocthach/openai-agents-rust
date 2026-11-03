//! Parity port of `tests/extensions/experiemental/codex/test_codex_tool.py::test_codex_tool_persists_session`:
//! a tool built with `persist_session` keeps one thread across invocations, so the second call
//! continues it instead of starting a fresh thread.
//!
//! The Codex CLI is faked with the POSIX shell fixture shared with `py_codex_tool.rs`.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use openai_agents_sdk::extensions::codex::{Codex, CodexOptions, CodexToolOptions, codex_tool};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use serde_json::json;
use tempfile::TempDir;

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

struct FakeCodex {
    dir: TempDir,
    env: BTreeMap<String, String>,
}

impl FakeCodex {
    fn new(thread_id: &str) -> Self {
        let dir = TempDir::new().expect("temp dir");
        let record_dir = dir.path().join("record");
        std::fs::create_dir_all(&record_dir).expect("record dir");
        let events_path = dir.path().join("events.jsonl");
        std::fs::write(&events_path, done_events(thread_id)).expect("events file");
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

    fn client(&self) -> Codex {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/py_codex_support/fake_codex.sh");
        Codex::new(
            CodexOptions::new()
                .with_codex_path_override(fixture.to_string_lossy().into_owned())
                .with_api_key("sk-test")
                .with_env(self.env.clone()),
        )
    }

    fn argv(&self, turn: usize) -> Vec<String> {
        std::fs::read_to_string(self.dir.path().join(format!("record/argv-{turn}.txt")))
            .expect("recorded argv")
            .lines()
            .map(str::to_string)
            .collect()
    }
}

async fn call_twice(tool: &FunctionTool<()>) {
    let arguments =
        json!({"inputs": [{"type": "text", "text": "First call", "path": ""}]}).to_string();
    for _ in 0..2 {
        let ctx = ToolContext::direct(RunContext::new(()), "test", &tool.name, &arguments);
        tool.invoke(ctx, arguments.clone())
            .await
            .expect("tool result");
    }
}

#[tokio::test]
async fn codex_tool_persists_session() {
    let fake = FakeCodex::new("thread-1");
    let tool: FunctionTool<()> = codex_tool(
        CodexToolOptions::new()
            .with_codex(fake.client())
            .with_persist_session(true),
    )
    .unwrap();

    call_twice(&tool).await;

    assert!(
        !fake.argv(1).contains(&"resume".to_string()),
        "{:?}",
        fake.argv(1)
    );
    let second = fake.argv(2);
    assert!(second.contains(&"resume".to_string()), "{second:?}");
    assert!(second.contains(&"thread-1".to_string()), "{second:?}");
}

#[tokio::test]
async fn codex_tool_without_persist_session_starts_a_fresh_thread_each_call() {
    let fake = FakeCodex::new("thread-1");
    let tool: FunctionTool<()> =
        codex_tool(CodexToolOptions::new().with_codex(fake.client())).unwrap();

    call_twice(&tool).await;

    assert!(!fake.argv(1).contains(&"resume".to_string()));
    assert!(
        !fake.argv(2).contains(&"resume".to_string()),
        "{:?}",
        fake.argv(2)
    );
}
