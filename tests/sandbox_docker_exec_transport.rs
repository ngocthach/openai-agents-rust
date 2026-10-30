//! The `docker exec` transport, exercised against a real daemon when one is available.
//!
//! Python drives exec over a hijacked raw socket (`_DockerExecSocket`) and frames stdin by byte
//! count (`_LENGTH_FRAMED_STDIN_SCRIPT`) because `docker-py` cannot half-close a TLS stdin. This
//! port shells out to `docker exec -i`, where the CLI owns the half close, so the framing is
//! unnecessary — but the observable behaviour it protects still has to hold: exit codes, stdin
//! EOF, separate stdout and stderr, and a large payload that outruns the pipe buffers.
//!
//! Every test here skips when no daemon or no local image is reachable, and never fails for it.
//! The unit tests beside `run_piped` in `src/sandbox/sandboxes/docker.rs` cover the same
//! transport properties with no daemon at all.

#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::sandboxes::docker::{
    DockerSandboxClient, DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::{ExecRequest, Manifest, SandboxClient, SandboxSession};
use std::sync::Arc;

/// Never pulled: a missing image skips the test instead of reaching the network.
const IMAGE: &str = "alpine:latest";

/// A started Docker session, or `None` when this machine cannot run one.
async fn session() -> Option<(DockerSandboxClient, Arc<dyn SandboxSession>)> {
    if !DockerSandboxClient::is_available().await {
        eprintln!("skipping: no reachable Docker daemon");
        return None;
    }
    if !DockerSandboxClient::image_exists(IMAGE).await {
        eprintln!("skipping: image `{IMAGE}` is not present locally");
        return None;
    }
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(IMAGE).with_network_disabled(true),
    );
    let session = client.create(Some(Manifest::new())).await.ok()?;
    session.start().await.ok()?;
    Some((client, session))
}

#[tokio::test]
async fn docker_exec_reports_exit_codes_and_keeps_the_streams_apart() {
    let Some((client, session)) = session().await else {
        return;
    };

    let result = session
        .exec(ExecRequest::shell("echo out; echo err >&2; exit 7"))
        .await
        .unwrap();
    assert_eq!(result.exit_code, 7);
    assert_eq!(result.stdout_text().trim(), "out");
    assert_eq!(result.stderr_text().trim(), "err");

    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn docker_write_file_survives_a_payload_larger_than_the_pipe_buffers() {
    let Some((client, session)) = session().await else {
        return;
    };

    // Well past any pipe buffer: a transport that fed stdin to completion before draining the
    // child's output would wedge here rather than return.
    let payload: Vec<u8> = (0..4 * 1024 * 1024u32).map(|i| (i % 251) as u8).collect();
    session.write_file("big.bin", &payload).await.unwrap();
    let read_back = session.read_file("big.bin").await.unwrap();
    assert_eq!(read_back.len(), payload.len());
    assert_eq!(read_back, payload);

    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn docker_exec_closes_stdin_so_a_draining_command_finishes() {
    let Some((client, session)) = session().await else {
        return;
    };

    // `cat` in the container only exits when its stdin reaches EOF. A transport that never
    // half-closes would hang until the test harness gave up.
    session.write_file("input.txt", b"hello").await.unwrap();
    let result = session
        .exec(ExecRequest::shell("cat < /workspace/input.txt"))
        .await
        .unwrap();
    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout_text(), "hello");

    client.delete(session).await.unwrap();
}
