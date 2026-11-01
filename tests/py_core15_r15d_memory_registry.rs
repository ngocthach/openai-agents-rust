//! Ports of the per-session memory manager registry cases of `tests/sandbox/test_memory.py`:
//! one generation manager per (session, layout), conflicting capabilities refused with Python's
//! messages, and the registration ends when the session closes.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::AgentModel;
use openai_agents_sdk::RunConfig;
use openai_agents_sdk::sandbox::capabilities::Memory;
use openai_agents_sdk::sandbox::memory::{MemoryGenerateConfig, MemoryLayoutConfig};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSession};
use openai_agents_sdk::testing::FakeModel;

async fn local_session() -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(Manifest::new())).await.expect("create");
    session.start().await.expect("start");
    (client, session)
}

fn generate() -> MemoryGenerateConfig {
    MemoryGenerateConfig::new()
        .with_phase_one_model(AgentModel::Instance(Arc::new(FakeModel::new())))
        .with_phase_two_model(AgentModel::Instance(Arc::new(FakeModel::new())))
}

fn memory(layout: MemoryLayoutConfig, generate: MemoryGenerateConfig) -> Memory {
    Memory::new()
        .with_layout(layout)
        .with_generate(Some(generate))
}

fn message(error: openai_agents_sdk::AgentsError) -> String {
    error.to_string()
}

/// Python: `test_sandbox_memory_rejects_different_generate_configs_for_same_layout`.
#[tokio::test]
async fn a_different_generate_config_for_the_same_layout_is_refused() {
    let (client, session) = local_session().await;
    let first = memory(MemoryLayoutConfig::new(), generate());
    let manager = first.manager_for_session(&session).unwrap().unwrap();
    // The same configuration shares the manager.
    let again = first.manager_for_session(&session).unwrap().unwrap();
    assert!(Arc::ptr_eq(&manager, &again));

    let different = memory(MemoryLayoutConfig::new(), generate());
    let error = different.manager_for_session(&session).unwrap_err();
    assert!(message(error).contains("different Memory generation config"),);
    client.delete(session).await.expect("delete");
}

/// Python: `test_sandbox_memory_rejects_different_sessions_dirs_for_same_memories_dir`.
#[tokio::test]
async fn the_same_memories_dir_with_another_sessions_dir_is_refused() {
    let (client, session) = local_session().await;
    let config = generate();
    let layout = |sessions: &str| {
        MemoryLayoutConfig::new()
            .with_memories_dir("shared_memory")
            .with_sessions_dir(sessions)
    };
    memory(layout("sessions_a"), config.clone())
        .manager_for_session(&session)
        .unwrap();
    let error = memory(layout("sessions_b"), config)
        .manager_for_session(&session)
        .unwrap_err();
    assert!(message(error).contains("already has a Memory generation capability"));
    client.delete(session).await.expect("delete");
}

/// Python: `test_sandbox_memory_rejects_shared_sessions_dir_for_different_memories_dirs`.
#[tokio::test]
async fn a_shared_sessions_dir_for_different_memories_dirs_is_refused() {
    let (client, session) = local_session().await;
    let config = generate();
    let layout = |memories: &str| {
        MemoryLayoutConfig::new()
            .with_memories_dir(memories)
            .with_sessions_dir("shared_sessions")
    };
    memory(layout("memory_a"), config.clone())
        .manager_for_session(&session)
        .unwrap();
    let error = memory(layout("memory_b"), config)
        .manager_for_session(&session)
        .unwrap_err();
    assert!(message(error).contains("sessions_dir=\"shared_sessions\""));
    client.delete(session).await.expect("delete");
}

/// Different sessions never share a manager.
#[tokio::test]
async fn managers_are_scoped_to_their_session() {
    let (client_a, session_a) = local_session().await;
    let (client_b, session_b) = local_session().await;
    let config = generate();
    let capability = memory(MemoryLayoutConfig::new(), config);
    let a = capability.manager_for_session(&session_a).unwrap().unwrap();
    let b = capability.manager_for_session(&session_b).unwrap().unwrap();
    assert!(!Arc::ptr_eq(&a, &b));
    client_a.delete(session_a).await.expect("delete");
    client_b.delete(session_b).await.expect("delete");
}

/// Python: `test_sandbox_memory_unregisters_manager_on_session_close`.
#[tokio::test]
async fn closing_the_session_unregisters_the_manager() {
    let (client, session) = local_session().await;
    let capability = memory(MemoryLayoutConfig::new(), generate());
    let manager = capability.manager_for_session(&session).unwrap().unwrap();
    let shared = capability.manager_for_session(&session).unwrap().unwrap();
    assert!(Arc::ptr_eq(&manager, &shared));

    let provider = RunConfig::<()>::default().model_provider;
    manager
        .flush(session.clone(), provider)
        .await
        .expect("flush");

    // A conflicting config is accepted now: nothing is registered for the layout any more.
    let other = memory(MemoryLayoutConfig::new(), generate());
    let fresh = other.manager_for_session(&session).unwrap().unwrap();
    assert!(!Arc::ptr_eq(&fresh, &manager));
    client.delete(session).await.expect("delete");
}
