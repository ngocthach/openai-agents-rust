//! The sandbox is prepared after the first turn's blocking input guardrails, as in Python: input
//! a guardrail rejects never costs a session, and an accepted run creates and releases exactly
//! one.
#![cfg(all(feature = "sandbox", unix))]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxResult, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{AgentsError, GuardrailFunctionOutput, InputGuardrail};

/// A client that counts what the runner asks of it.
#[derive(Debug, Default)]
struct CountingClient {
    inner: UnixLocalSandboxClient,
    creates: AtomicUsize,
    deletes: AtomicUsize,
}

#[async_trait]
impl SandboxClient for CountingClient {
    fn backend_id(&self) -> &str {
        self.inner.backend_id()
    }
    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Arc<dyn SandboxSession>> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        self.inner.create(manifest).await
    }
    async fn delete(&self, session: Arc<dyn SandboxSession>) -> SandboxResult<()> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
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

fn agent(trips: bool) -> SandboxAgent<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let mut agent = SandboxAgent::<()>::new("guarded").with_model_instance(model);
    agent.agent = agent.agent.with_input_guardrail(
        InputGuardrail::new("g", move |_c, _a, _i| async move {
            Ok(GuardrailFunctionOutput::new(serde_json::Value::Null, trips))
        })
        .blocking(),
    );
    agent
}

#[tokio::test]
async fn rejected_input_never_creates_a_session() {
    let client = Arc::new(CountingClient::default());
    let config = SandboxRunConfig::with_client(client.clone());

    let outcome = SandboxRunner::run(&agent(true), "hi", &config).await;

    assert!(matches!(
        outcome,
        Err(AgentsError::InputGuardrailTripwireTriggered { .. })
    ));
    assert_eq!(client.creates.load(Ordering::SeqCst), 0);
    assert_eq!(client.deletes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rejected_streamed_input_never_creates_a_session() {
    let client = Arc::new(CountingClient::default());
    let config = SandboxRunConfig::with_client(client.clone());

    let mut run = SandboxRunner::run_streamed(&agent(true), "hi", &config)
        .await
        .expect("a stream");
    let outcome = run.finish().await;

    assert!(outcome.is_err());
    assert_eq!(client.creates.load(Ordering::SeqCst), 0);
    assert_eq!(client.deletes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn accepted_input_creates_and_releases_one_session() {
    let client = Arc::new(CountingClient::default());
    let config = SandboxRunConfig::with_client(client.clone());

    let result = SandboxRunner::run(&agent(false), "hi", &config)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(client.creates.load(Ordering::SeqCst), 1);
    assert_eq!(client.deletes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn accepted_streamed_input_creates_and_releases_one_session() {
    let client = Arc::new(CountingClient::default());
    let config = SandboxRunConfig::with_client(client.clone());

    let mut run = SandboxRunner::run_streamed(&agent(false), "hi", &config)
        .await
        .expect("a stream");
    run.finish().await.expect("the stream completes");

    assert_eq!(client.creates.load(Ordering::SeqCst), 1);
    assert_eq!(client.deletes.load(Ordering::SeqCst), 1);
}
