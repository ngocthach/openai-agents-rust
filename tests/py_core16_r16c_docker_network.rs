//! Port of `test_docker_options_reject_exposed_ports_with_network_mode_none` from
//! `tests/sandbox/test_docker_network_mode.py`. Python spells the setting `network_mode="none"`;
//! Rust spells it `network_disabled`. A container without a network stack cannot publish a
//! port, so a client configured with both refuses to create a session and never reaches the
//! daemon.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::sandboxes::docker::{
    DockerRunner, DockerSandboxClient, DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{ErrorCode, SandboxClient, SandboxResult};

#[derive(Debug, Default)]
struct Recorder {
    calls: Mutex<Vec<Vec<String>>>,
}

#[async_trait::async_trait]
impl DockerRunner for Recorder {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().unwrap().push(argv.to_vec());
        Ok(ExecResult {
            stdout: b"container-id\n".to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
        })
    }
}

#[tokio::test]
async fn exposed_ports_are_rejected_for_a_container_without_a_network() {
    let runner = Arc::new(Recorder::default());
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new("python:3.14-slim")
            .with_exposed_ports([8080])
            .with_network_disabled(true),
    )
    .with_runner(runner.clone());

    let error = match client.create(None).await {
        Ok(_) => panic!("both settings together must be refused"),
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(error.to_string().contains("exposed_ports"), "{error}");
    assert!(
        runner.calls.lock().unwrap().is_empty(),
        "the daemon is never called"
    );
}

#[tokio::test]
async fn exposed_ports_alone_and_a_disabled_network_alone_are_accepted() {
    for options in [
        DockerSandboxClientOptions::new("python:3.14-slim").with_exposed_ports([8080]),
        DockerSandboxClientOptions::new("python:3.14-slim").with_network_disabled(true),
    ] {
        let runner = Arc::new(Recorder::default());
        let client = DockerSandboxClient::with_options(options).with_runner(runner.clone());
        client
            .create(None)
            .await
            .expect("a valid configuration creates");
        assert!(!runner.calls.lock().unwrap().is_empty());
    }
}
