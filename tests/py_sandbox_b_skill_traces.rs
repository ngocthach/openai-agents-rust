//! The span rows of `tests/sandbox/capabilities/test_skills_capability.py`: the "is this skill
//! loaded yet?" probe leaves its span clean when the file is missing, and marks it when the read
//! fails for any other reason. Trace processors are process-global, so one test runs both.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::capabilities::{LocalDirLazySkillSource, Skills};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::events::{CallbackSink, InstrumentedSession};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::{SandboxPathGrant, SandboxWorkspaceScope};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, TracingProcessor, set_trace_processors, trace,
};
use serde_json::Value;

/// A session whose reads fail with a chosen error, over a real workspace for everything else.
#[derive(Debug)]
struct FailingReads {
    inner: Arc<dyn SandboxSession>,
    error: Option<SandboxError>,
    writes: std::sync::Mutex<usize>,
}

#[async_trait]
impl SandboxSession for FailingReads {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn manifest(&self) -> &Manifest {
        self.inner.manifest()
    }
    fn backend_id(&self) -> &str {
        "failing-reads"
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
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        self.inner.exec(request).await
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => self.inner.read_file(path).await,
        }
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        *self.writes.lock().unwrap() += 1;
        self.inner.write_file(path, contents).await
    }
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.hydrate_workspace(archive).await
    }
}

fn reads(processor: &InMemoryTraceProcessor) -> Vec<Value> {
    processor
        .items()
        .into_iter()
        .filter(|item| {
            item["object"] == "trace.span" && item["span_data"]["name"] == "sandbox.read"
        })
        .collect()
}

async fn load(
    error: Option<SandboxError>,
) -> (
    SandboxResult<openai_agents_sdk::sandbox::capabilities::skills::LoadedSkill>,
    usize,
) {
    let temp = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(temp.path()).unwrap();
    let source = base.join("skills");
    std::fs::create_dir_all(source.join("dynamic-skill")).unwrap();
    std::fs::write(source.join("dynamic-skill/SKILL.md"), "# dynamic skill\n").unwrap();
    let workspace = base.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let client = UnixLocalSandboxClient::new();
    let real = client
        .create(Some(
            Manifest::new()
                .with_root(workspace.to_string_lossy())
                .with_extra_path_grant(SandboxPathGrant::new(source.to_string_lossy()).unwrap()),
        ))
        .await
        .unwrap();
    real.start().await.unwrap();
    let failing = Arc::new(FailingReads {
        inner: real,
        error,
        writes: std::sync::Mutex::new(0),
    });
    let session = InstrumentedSession::new(failing.clone(), Arc::new(CallbackSink::new(|_| {})));
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&source)));
    let result = trace("lazy_skill_probe")
        .run(async {
            skills
                .load_skill(
                    &session,
                    &SandboxWorkspaceScope::root(),
                    None,
                    "dynamic-skill",
                )
                .await
        })
        .await;
    let writes = *failing.writes.lock().unwrap();
    (result, writes)
}

#[tokio::test]
async fn the_loaded_probe_marks_only_unexpected_read_failures() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    // Python: `test_first_load_omits_only_expected_probe_span_error`. A missing SKILL.md is the
    // expected answer on a first load.
    let (loaded, _) = load(None).await;
    assert_eq!(loaded.unwrap().status, "loaded");
    let spans = reads(&processor);
    assert_eq!(spans.len(), 1);
    assert!(spans[0]["error"].is_null(), "{}", spans[0]);

    // The same, when the backend itself reports the file as not found.
    processor.clear();
    let (loaded, _) = load(Some(SandboxError::new(
        ErrorCode::WorkspaceReadNotFound,
        "missing",
    )))
    .await;
    assert_eq!(loaded.unwrap().status, "loaded");
    let spans = reads(&processor);
    assert_eq!(spans.len(), 1);
    assert!(spans[0]["error"].is_null());

    // Python: `test_load_skill_propagates_non_not_found_read_error`.
    processor.clear();
    let (failed, writes) = load(Some(SandboxError::new(
        ErrorCode::WorkspaceArchiveReadError,
        "simulated permission failure",
    )))
    .await;
    assert_eq!(
        failed.unwrap_err().code,
        ErrorCode::WorkspaceArchiveReadError
    );
    assert_eq!(writes, 0, "nothing was materialized");
    let spans = reads(&processor);
    assert_eq!(spans.len(), 1);
    assert!(spans[0]["error"].is_object(), "{}", spans[0]);
}
