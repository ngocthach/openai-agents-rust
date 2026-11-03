//! Port of `test_sandbox_memory_flush_parent_cancellation_stops_worker`
//! (`tests/sandbox/test_memory.py`): cancelling a flush (dropping its future in Rust) stops the
//! phase-one run, never starts phase two, and ends the manager's registration.
#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::models::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::sandbox::capabilities::Memory;
use openai_agents_sdk::sandbox::memory::{MemoryGenerateConfig, MemoryLayoutConfig};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient};
use openai_agents_sdk::{AgentModel, RunConfig};
use serde_json::json;
use tokio::sync::Notify;

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A phase-one model that blocks until the run is cancelled.
#[derive(Debug)]
struct BlockingModel {
    started: Arc<Notify>,
    cancelled: Arc<AtomicBool>,
}

#[async_trait]
impl Model for BlockingModel {
    async fn get_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        let _flag = DropFlag(self.cancelled.clone());
        self.started.notify_one();
        futures::future::pending().await
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        unreachable!("the memory phases do not stream")
    }
}

/// A phase-two model that records that it was reached.
#[derive(Debug)]
struct CountingModel(Arc<AtomicUsize>);

#[async_trait]
impl Model for CountingModel {
    async fn get_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        self.0.fetch_add(1, Ordering::SeqCst);
        futures::future::pending().await
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        unreachable!("the memory phases do not stream")
    }
}

#[tokio::test]
async fn flush_cancellation_stops_the_phase_run_and_ends_the_registration() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(AtomicBool::new(false));
    let phase_two_calls = Arc::new(AtomicUsize::new(0));
    let generate = MemoryGenerateConfig::new()
        .with_phase_one_model(AgentModel::Instance(Arc::new(BlockingModel {
            started: started.clone(),
            cancelled: cancelled.clone(),
        })))
        .with_phase_two_model(AgentModel::Instance(Arc::new(CountingModel(
            phase_two_calls.clone(),
        ))));
    let capability = Memory::new()
        .with_layout(MemoryLayoutConfig::new())
        .with_generate(Some(generate));

    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(Manifest::new())).await.expect("create");
    session.start().await.expect("start");
    let manager = capability.manager_for_session(&session).unwrap().unwrap();
    manager
        .enqueue_rollout_payload(
            session.as_ref(),
            "cancelled-flush",
            &json!({
                "updated_at": "2026-08-05T00:00:00+00:00",
                "input": [],
                "generated_items": [],
                "terminal_metadata": {"terminal_state": "completed", "has_final_output": false},
            }),
        )
        .await
        .expect("enqueue");

    let provider = RunConfig::<()>::default().model_provider;
    let flush = tokio::spawn({
        let manager = manager.clone();
        let session = session.clone();
        async move { manager.flush(session, provider).await }
    });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("phase one never started");
    flush.abort();
    assert!(flush.await.unwrap_err().is_cancelled());

    assert!(cancelled.load(Ordering::SeqCst), "phase one kept running");
    assert_eq!(phase_two_calls.load(Ordering::SeqCst), 0);
    let fresh = capability.manager_for_session(&session).unwrap().unwrap();
    assert!(!Arc::ptr_eq(&manager, &fresh), "the registration survived");
    client.delete(session).await.expect("delete");
}

/// A phase-one model whose worker fails fatally.
#[derive(Debug)]
struct PanickingModel;

#[async_trait]
impl Model for PanickingModel {
    async fn get_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        panic!("fatal memory worker failure");
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        unreachable!("the memory phases do not stream")
    }
}

/// `test_sandbox_memory_flush_propagates_worker_base_exception_without_hanging[base_exception]`:
/// a fatal failure in the phase-one worker reaches the caller of `flush` instead of hanging it,
/// and the manager's registration has ended.
#[tokio::test]
async fn a_fatal_phase_one_failure_propagates_without_hanging() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let generate = MemoryGenerateConfig::new()
        .with_phase_one_model(AgentModel::Instance(Arc::new(PanickingModel)))
        .with_phase_two_model(AgentModel::Instance(Arc::new(PanickingModel)));
    let capability = Memory::new()
        .with_layout(MemoryLayoutConfig::new())
        .with_generate(Some(generate));
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(Manifest::new())).await.expect("create");
    session.start().await.expect("start");
    let manager = capability.manager_for_session(&session).unwrap().unwrap();
    manager
        .enqueue_rollout_payload(
            session.as_ref(),
            "fatal-worker",
            &json!({
                "updated_at": "2026-08-05T00:00:00+00:00",
                "input": [],
                "generated_items": [],
                "terminal_metadata": {"terminal_state": "completed", "has_final_output": false},
            }),
        )
        .await
        .expect("enqueue");

    let provider = RunConfig::<()>::default().model_provider;
    let flush = tokio::spawn({
        let manager = manager.clone();
        let session = session.clone();
        async move { manager.flush(session, provider).await }
    });
    let joined = tokio::time::timeout(Duration::from_secs(5), flush)
        .await
        .expect("the flush hung");
    assert!(joined.unwrap_err().is_panic());

    let fresh = capability.manager_for_session(&session).unwrap().unwrap();
    assert!(!Arc::ptr_eq(&manager, &fresh), "the registration survived");
    client.delete(session).await.expect("delete");
}
