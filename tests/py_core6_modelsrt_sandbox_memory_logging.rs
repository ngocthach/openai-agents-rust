//! Port of `test_sandbox_memory_enqueue_failure_follows_both_data_policies`
//! (`tests/sandbox/test_memory.py`): a failed rollout enqueue never fails the run, and its
//! warning keeps the error text only when neither `dont_log_model_data` nor `dont_log_tool_data`
//! is set.
#![cfg(feature = "sandbox")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::AgentModel;
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::sandbox::capabilities::{Filesystem, Memory, Shell};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::memory::MemoryGenerateConfig;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    Manifest, SandboxAgent, SandboxClient, SandboxResult, SandboxRunConfig, SandboxRunner,
    SandboxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use serde_json::json;

struct LogSink(Mutex<Vec<(log::Level, String)>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0
                .lock()
                .unwrap()
                .push((record.level(), record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let sink = SINK.get_or_init(|| LogSink(Mutex::new(Vec::new())));
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

/// Restores the redaction flags to the environment defaults when a test ends.
struct PolicyGuard;

impl PolicyGuard {
    fn set(model_redacted: bool, tool_redacted: bool) -> Self {
        set_dont_log_model_data(model_redacted);
        set_dont_log_tool_data(tool_redacted);
        PolicyGuard
    }
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

fn shell_call(call_id: &str, command: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": [command]}
    }))
}

fn phase_one_model() -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message(
        json!({
            "rollout_slug": "task_memory",
            "rollout_summary": "# Task summary\n",
            "raw_memory": "raw\n",
        })
        .to_string(),
    )]);
    model
}

fn consolidating_model() -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![shell_call(
        "consolidate",
        "printf 'memories\\n' > memories/MEMORY.md && printf 'summary\\n' > memories/memory_summary.md",
    )]);
    model.add_turn(vec![assistant_message("consolidated")]);
    model
}

/// A client that delegates to the Unix-local backend and records the root of every workspace it
/// deletes.
#[derive(Debug, Default)]
struct DeleteTrackingClient {
    inner: UnixLocalSandboxClient,
    deleted_roots: Mutex<Vec<PathBuf>>,
}

#[async_trait]
impl SandboxClient for DeleteTrackingClient {
    fn backend_id(&self) -> &str {
        self.inner.backend_id()
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.inner.create(manifest).await
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        self.deleted_roots
            .lock()
            .expect("deleted roots")
            .push(PathBuf::from(session.workspace_root()));
        self.inner.delete(session).await
    }
}

#[tokio::test]
async fn sandbox_memory_enqueue_failure_follows_both_data_policies() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _turn = SERIAL.lock().await;
    let sink = sink();

    for streamed in [false, true] {
        for (model_redacted, tool_redacted) in [(true, false), (false, true), (false, false)] {
            let _policy = PolicyGuard::set(model_redacted, tool_redacted);
            sink.0.lock().unwrap().clear();

            let client = Arc::new(DeleteTrackingClient::default());
            let memory = Memory::new().with_generate(Some(
                MemoryGenerateConfig::new()
                    .with_phase_one_model(AgentModel::Instance(phase_one_model()))
                    .with_phase_two_model(AgentModel::Instance(consolidating_model()))
                    .with_phase_one_model_settings(None)
                    .with_phase_two_model_settings(None),
            ));
            let model = Arc::new(FakeModel::new());
            model.add_turn(vec![assistant_message("done")]);
            let agent = SandboxAgent::<()>::new("worker")
                .with_model_instance(model)
                .with_instructions("Worker.")
                .with_capabilities(vec![
                    Arc::new(Filesystem::new()),
                    Arc::new(Shell::new().with_pty(false)),
                    Arc::new(memory),
                ])
                // A file where the sessions directory has to go makes every enqueue fail.
                .with_default_manifest(
                    Manifest::new().with_entry("sessions", Entry::text_file("in the way")),
                );
            let config = SandboxRunConfig::with_client(client.clone());

            let expected_message = if streamed {
                let mut run = SandboxRunner::run_streamed(&agent, "hello", &config)
                    .await
                    .expect("the stream starts");
                {
                    let stream = run.streaming.stream_events();
                    futures::pin_mut!(stream);
                    while stream.next().await.is_some() {}
                }
                assert_eq!(run.streaming.final_output_text().as_deref(), Some("done"));
                "Failed to enqueue sandbox memory after streamed run"
            } else {
                let result = SandboxRunner::run(&agent, "hello", &config)
                    .await
                    .expect("run");
                assert_eq!(result.final_output_text().as_deref(), Some("done"));
                "Failed to enqueue sandbox memory after run"
            };

            let deleted = client.deleted_roots.lock().expect("deleted").clone();
            assert_eq!(deleted.len(), 1);
            assert!(!deleted[0].exists());

            let case = format!("streamed={streamed} model={model_redacted} tool={tool_redacted}");
            let records: Vec<(log::Level, String)> = sink
                .0
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, message)| message.starts_with(expected_message))
                .cloned()
                .collect();
            assert_eq!(records.len(), 1, "{case}: {records:?}");
            let (level, record) = &records[0];
            assert_eq!(*level, log::Level::Warn, "{case}");
            if model_redacted || tool_redacted {
                assert_eq!(record, expected_message, "{case}");
            } else {
                assert!(
                    record.starts_with(&format!("{expected_message}: ")),
                    "{case}: {record}"
                );
            }
        }
    }
}
