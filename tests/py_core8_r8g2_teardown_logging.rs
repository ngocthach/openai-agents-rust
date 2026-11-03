//! What a failed teardown writes to the log, from
//! `tests/sandbox/test_runtime.py::test_runner_owned_cleanup_redacts_client_delete_failure`.
//!
//! Python raises a fixed "protected mount configuration" error when the session holds a
//! credentialed mount, so a provider's message (which can repeat the mount's configuration) never
//! reaches the caller. A Rust run is not failed by a teardown error, so the same failure is
//! logged instead, and the log is where the provider's text must stay out of. The `log` backend
//! and the redaction flags are process-wide, so this binary holds one test.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::RunOptions;
use openai_agents_sdk::debug::{reset_dont_log_tool_data, set_dont_log_tool_data};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, SandboxAgent, SandboxClient, SandboxError, SandboxResult,
    SandboxRunConfig, SandboxRunner, SandboxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};

struct LogSink(Mutex<Vec<String>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn log_sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let sink = SINK.get_or_init(|| LogSink(Mutex::new(Vec::new())));
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

#[derive(Debug)]
struct FakeSession {
    manifest: Manifest,
}

#[async_trait]
impl SandboxSession for FakeSession {
    fn id(&self) -> &str {
        "fake-session"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "fake"
    }
    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }
    async fn exec(&self, _request: ExecRequest) -> SandboxResult<ExecResult> {
        Ok(ExecResult {
            exit_code: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
    async fn read_file(&self, _path: &str) -> SandboxResult<Vec<u8>> {
        Ok(Vec::new())
    }
    async fn write_file(&self, _path: &str, _contents: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }
}

/// A client whose `delete` fails with a message that repeats a secret, as a provider error that
/// quotes the mount configuration would.
#[derive(Debug)]
struct FailingDeleteClient {
    secret: &'static str,
}

#[async_trait]
impl SandboxClient for FailingDeleteClient {
    fn backend_id(&self) -> &str {
        "fake"
    }
    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        Ok(Arc::new(FakeSession {
            manifest: manifest.unwrap_or_default(),
        }))
    }
    async fn delete(&self, _session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        Err(SandboxError::new(
            ErrorCode::WorkspaceStopError,
            format!("delete failed with {}", self.secret),
        ))
    }
    async fn resume(
        &self,
        _state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        Ok(Arc::new(FakeSession { manifest }))
    }
}

fn credentialed_manifest() -> Manifest {
    Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::docker_volume(
                S3Mount::new("bucket").with_credentials("access-key", "client-delete-secret"),
                "rclone",
            )
            .mount(),
        ),
    )
}

async fn run_with(manifest: Manifest) -> Vec<String> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    log_sink().0.lock().unwrap().clear();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_instructions("Base instructions.")
        .with_default_manifest(manifest)
        .with_capabilities(Vec::new());
    let config = SandboxRunConfig::with_client(Arc::new(FailingDeleteClient {
        secret: "client-delete-secret",
    }));

    let outcome = SandboxRunner::run_to_outcome(&agent, "hello", &config, RunOptions::default())
        .await
        .expect("a teardown failure does not fail a finished run");
    assert_eq!(outcome.result.final_output_text().as_deref(), Some("done"));
    std::mem::take(&mut *log_sink().0.lock().unwrap())
}

/// Python: `test_runner_owned_cleanup_redacts_client_delete_failure` (both parameter sets).
#[tokio::test]
async fn a_failed_teardown_never_logs_a_credentialed_mounts_provider_text() {
    let _ = log_sink();

    // Whatever the logging policy says, a session with a credentialed mount logs the sentence.
    for dont_log_tool_data in [true, false] {
        set_dont_log_tool_data(dont_log_tool_data);
        let records = run_with(credentialed_manifest()).await;
        let record = records
            .iter()
            .find(|record| record.contains("Failed to tear down the sandbox session"))
            .unwrap_or_else(|| panic!("a teardown failure record in {records:?}"));
        assert_eq!(record, "Failed to tear down the sandbox session");
        assert!(records.iter().all(|r| !r.contains("client-delete-secret")));
    }

    // Without a credentialed mount the text follows the tool-data policy.
    set_dont_log_tool_data(true);
    let records = run_with(Manifest::new().with_root("/workspace")).await;
    assert!(
        records.iter().all(|r| !r.contains("delete failed with")),
        "{records:?}"
    );
    set_dont_log_tool_data(false);
    let records = run_with(Manifest::new().with_root("/workspace")).await;
    assert!(
        records.iter().any(|r| r.contains("delete failed with")),
        "{records:?}"
    );

    reset_dont_log_tool_data();
}
