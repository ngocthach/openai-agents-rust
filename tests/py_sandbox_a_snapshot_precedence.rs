//! Which snapshot a runner-created session is given.
//!
//! Python's client takes its snapshot only as the `create(snapshot=...)` argument, and the runner
//! fills that argument with the SDK default when the run config names none
//! (`_resolve_snapshot_spec`). A Rust client can also be built with a snapshot, so the order is:
//! the run config's snapshot, then the client's own, then the SDK default.

#![cfg(feature = "sandbox")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    Manifest, SandboxAgent, SandboxClient, SandboxResult, SandboxRunConfig, SandboxRunner,
    SandboxSession, SandboxSessionState, SnapshotSpec,
};

/// Forwards to a real `unix_local` client and records the snapshot each create was given.
#[derive(Debug)]
struct RecordingClient {
    inner: UnixLocalSandboxClient,
    configured: Option<SnapshotSpec>,
    seen: Mutex<Vec<SnapshotSpec>>,
}

impl RecordingClient {
    fn new(configured: Option<SnapshotSpec>) -> Arc<Self> {
        Arc::new(RecordingClient {
            inner: UnixLocalSandboxClient::new(),
            configured,
            seen: Mutex::new(Vec::new()),
        })
    }

    fn seen(&self) -> Vec<SnapshotSpec> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl SandboxClient for RecordingClient {
    fn backend_id(&self) -> &str {
        "unix_local"
    }

    fn configured_snapshot(&self) -> Option<SnapshotSpec> {
        self.configured.clone()
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.inner.create(manifest).await
    }

    async fn create_with_snapshot(
        &self,
        manifest: Option<Manifest>,
        snapshot: SnapshotSpec,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.seen.lock().unwrap().push(snapshot);
        self.inner.create(manifest).await
    }

    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        self.inner.delete(session).await
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.inner.resume(state, manifest).await
    }
}

fn local(path: &str) -> SnapshotSpec {
    SnapshotSpec::Local {
        base_path: PathBuf::from(path),
    }
}

async fn prepared_snapshot(client: Arc<RecordingClient>, config: SandboxRunConfig) -> SnapshotSpec {
    let agent = SandboxAgent::<()>::new("Sandbox agent");
    let prepared = SandboxRunner::prepare(&agent, &config)
        .await
        .expect("prepare");
    prepared.guard.close().await.expect("close");
    let seen = client.seen();
    assert_eq!(seen.len(), 1, "one session was created: {seen:?}");
    seen[0].clone()
}

#[tokio::test]
async fn an_explicit_run_config_snapshot_wins_over_the_clients_own() {
    let client = RecordingClient::new(Some(local("/tmp/from-the-client")));
    let config = SandboxRunConfig::with_client(client.clone()).snapshot(local("/tmp/from-the-run"));

    assert_eq!(
        prepared_snapshot(client, config).await,
        local("/tmp/from-the-run")
    );
}

#[tokio::test]
async fn a_snapshot_the_client_was_built_with_beats_the_sdk_default() {
    let client = RecordingClient::new(Some(local("/tmp/from-the-client")));
    let config = SandboxRunConfig::with_client(client.clone());

    assert_eq!(
        prepared_snapshot(client, config).await,
        local("/tmp/from-the-client")
    );
}
