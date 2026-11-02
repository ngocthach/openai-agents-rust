//! The Docker sandbox client against a real Docker daemon.
//!
//! Every other Docker test drives a recording [`DockerRunner`]; these run the real `docker` CLI
//! against a real daemon, so they only run when `OPENAI_AGENTS_DOCKER_LIVE=1` is set and skip
//! (with the reason printed) otherwise, or when the image is not present locally. The image is
//! `OPENAI_AGENTS_DOCKER_LIVE_IMAGE`, defaulting to the SDK's default image; it is never pulled.
//!
//! Each test removes its containers when it ends, including on a failed assertion.
//!
//! [`DockerRunner`]: openai_agents_sdk::sandbox::sandboxes::docker::DockerRunner
#![cfg(feature = "sandbox")]

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::pty::PtyExecRequest;
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DEFAULT_PYTHON_SANDBOX_IMAGE, DOCKER_NETWORK_DISABLED_KEY, DockerSandboxClient,
    DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::session::{
    ExecRequest, SandboxClient, SandboxSession, persist_session_snapshot,
};
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::types::User;

const LIVE_ENV: &str = "OPENAI_AGENTS_DOCKER_LIVE";
const IMAGE_ENV: &str = "OPENAI_AGENTS_DOCKER_LIVE_IMAGE";
const TEST_LABEL: &str = "openai-agents-live-test";

/// The image to run, or `None` (with the reason printed) when the live tests must skip.
async fn live_image() -> Option<String> {
    if std::env::var(LIVE_ENV).as_deref() != Ok("1") {
        eprintln!("skipping live Docker test: set {LIVE_ENV}=1 to run it");
        return None;
    }
    if !DockerSandboxClient::is_available().await {
        eprintln!("skipping live Docker test: no Docker daemon answers `docker info`");
        return None;
    }
    let image = std::env::var(IMAGE_ENV).unwrap_or_else(|_| DEFAULT_PYTHON_SANDBOX_IMAGE.into());
    if !DockerSandboxClient::image_exists(&image).await {
        eprintln!("skipping live Docker test: image `{image}` is not present locally");
        return None;
    }
    Some(image)
}

/// Force-removes every container it was told about when the test ends, however it ends.
#[derive(Default)]
struct Containers(Mutex<Vec<String>>);

impl Containers {
    fn track(&self, session: &Arc<dyn SandboxSession>) {
        self.0.lock().unwrap().push(session.id().to_string());
    }
}

impl Drop for Containers {
    fn drop(&mut self) {
        for id in self.0.lock().unwrap().drain(..) {
            let _ = std::process::Command::new("docker")
                .args(["rm", "-f", "-v", &id])
                .output();
        }
    }
}

/// `docker container inspect` of `id`, or `None` when it does not exist.
fn inspect(id: &str) -> Option<serde_json::Value> {
    let output = std::process::Command::new("docker")
        .args(["container", "inspect", id])
        .output()
        .unwrap();
    if !output.status.success() {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    parsed.as_array()?.first().cloned()
}

fn options(image: &str) -> DockerSandboxClientOptions {
    DockerSandboxClientOptions::new(image).with_label(TEST_LABEL, "1")
}

async fn shell(session: &Arc<dyn SandboxSession>, command: &str) -> (i32, String) {
    let result = session
        .exec(ExecRequest::shell(command))
        .await
        .expect("exec");
    (result.exit_code, result.stdout_text())
}

/// Create, materialize, operate on the workspace through every file primitive, and delete.
#[tokio::test]
async fn live_docker_session_materializes_operates_and_deletes() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_env("GREETING", "hello from the manifest")
        .with_user(User::new("sandboxer"))
        .with_entry("README.md", Entry::text_file("# live\n"))
        .with_entry(
            "src",
            Entry::dir().with_child("main.py", Entry::text_file("print('hi')\n")),
        );
    let client = DockerSandboxClient::with_options(options(&image));

    let session = client.create(Some(manifest)).await.expect("create");
    containers.track(&session);
    session
        .start()
        .await
        .expect("start materializes the manifest");
    assert!(session.running().await.unwrap());

    // The container carries the SDK label and the configured one.
    let attrs = inspect(session.id()).expect("the container exists");
    assert_eq!(attrs["Config"]["Labels"]["openai-agents-sandbox"], "1");
    assert_eq!(attrs["Config"]["Labels"][TEST_LABEL], "1");

    // Manifest entries landed, nested ones included.
    assert_eq!(session.read_file("README.md").await.unwrap(), b"# live\n");
    assert_eq!(
        session.read_file("src/main.py").await.unwrap(),
        b"print('hi')\n"
    );
    let listed: Vec<String> = session
        .ls(".")
        .await
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    assert!(
        listed.iter().any(|path| path.ends_with("README.md"))
            && listed.iter().any(|path| path.ends_with("src")),
        "{listed:?}"
    );

    // Commands see the manifest environment, the workspace as their directory, `cwd`, and the
    // provisioned account.
    assert_eq!(
        shell(&session, "printf '%s|%s' \"$GREETING\" \"$PWD\"").await,
        (0, "hello from the manifest|/workspace".to_string())
    );
    let in_src = session
        .exec(ExecRequest::argv(["pwd"]).with_cwd(Some("src".into())))
        .await
        .unwrap();
    assert_eq!(in_src.stdout_text().trim(), "/workspace/src");
    let as_user = session
        .exec(ExecRequest::argv(["id", "-un"]).with_user(Some(User::new("sandboxer"))))
        .await
        .unwrap();
    assert_eq!(as_user.stdout_text().trim(), "sandboxer", "{as_user:?}");

    // A payload far larger than a pipe buffer survives the stdin path both ways.
    let payload: Vec<u8> = (0..3 * 1024 * 1024u32).map(|i| (i % 251) as u8).collect();
    session
        .write_file("out/data.bin", &payload)
        .await
        .expect("write creates missing parents");
    assert_eq!(session.read_file("out/data.bin").await.unwrap(), payload);
    assert_eq!(
        session.read_bounded("out/data.bin", 10).await.unwrap(),
        payload[..10]
    );

    session.mkdir("a/b/c", true).await.unwrap();
    session
        .mv("out/data.bin", "a/b/c/data.bin")
        .await
        .expect("mv");
    assert_eq!(session.read_file("a/b/c/data.bin").await.unwrap(), payload);

    session.rm("README.md", false).await.unwrap();
    let missing = session.read_file("README.md").await.unwrap_err();
    assert_eq!(
        missing.code,
        ErrorCode::WorkspaceReadNotFound,
        "{missing:?}"
    );
    session.rm("a", true).await.unwrap();
    assert_eq!(shell(&session, "test -e /workspace/a").await.0, 1);

    // A link the sandbox made cannot carry a read or a write out of the workspace.
    assert_eq!(shell(&session, "ln -s /etc /workspace/escape").await.0, 0);
    let escaped = session.read_file("escape/passwd").await.unwrap_err();
    assert_eq!(
        escaped.context.get("reason").map(String::as_str),
        Some("escape_root"),
        "{escaped:?}"
    );
    let escaped = session
        .write_file("escape/planted", b"x")
        .await
        .unwrap_err();
    assert_eq!(
        escaped.context.get("reason").map(String::as_str),
        Some("escape_root"),
        "{escaped:?}"
    );

    // A command deadline is enforced.
    let timed_out = session
        .exec(ExecRequest::shell("sleep 10").with_timeout(Some(Duration::from_secs(1))))
        .await
        .unwrap_err();
    assert_eq!(timed_out.code, ErrorCode::ExecTimeout);

    // An interactive process runs in the container and reports its exit.
    let mut update = session
        .pty_exec_start(PtyExecRequest::new("echo pty-$((6*7))").with_yield_time_ms(5_000))
        .await
        .expect("pty start");
    let mut output = update.output_text();
    while update.exit_code.is_none()
        && let Some(process_id) = update.process_id
    {
        update = session
            .pty_write_stdin(
                openai_agents_sdk::sandbox::pty::PtyWriteRequest::new(process_id, "")
                    .with_yield_time_ms(2_000),
            )
            .await
            .expect("pty poll");
        output.push_str(&update.output_text());
    }
    assert!(output.contains("pty-42"), "{output:?}");
    assert_eq!(update.exit_code, Some(0));

    let id = session.id().to_string();
    client.delete(session).await.expect("delete");
    assert!(inspect(&id).is_none(), "the container is removed");
}

/// Resume reattaches a stopped container, and replaces a deleted one from its snapshot.
#[tokio::test]
async fn live_docker_resume_reattaches_then_replaces_from_the_snapshot() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    let snapshots = tempfile::tempdir().unwrap();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("seed.txt", Entry::text_file("seed\n"));
    let client =
        DockerSandboxClient::with_options(options(&image)).with_snapshot(SnapshotSpec::Local {
            base_path: snapshots.path().to_path_buf(),
        });

    let session = client.create(Some(manifest.clone())).await.expect("create");
    containers.track(&session);
    session.start().await.expect("start");
    session
        .write_file("notes/today.txt", b"written before the pause")
        .await
        .unwrap();
    persist_session_snapshot(session.as_ref())
        .await
        .expect("persist the workspace snapshot");
    let state = session.session_state();
    let original = session.id().to_string();
    session.stop().await.expect("stop");
    assert_eq!(
        inspect(&original).unwrap()["State"]["Running"],
        false,
        "stopped, not removed"
    );

    // The stopped container still exists: resume reattaches to it.
    let reattached = client
        .resume(&state, manifest.clone())
        .await
        .expect("resume reattaches");
    assert_eq!(reattached.id(), original);
    reattached.start().await.expect("restart");
    assert_eq!(
        reattached.read_file("notes/today.txt").await.unwrap(),
        b"written before the pause"
    );
    client.delete(reattached).await.expect("delete");
    assert!(inspect(&original).is_none());

    // The container is gone: resume creates a replacement and restores the snapshot into it.
    let replaced = client
        .resume(&state, manifest)
        .await
        .expect("resume replaces");
    containers.track(&replaced);
    assert_ne!(replaced.id(), original);
    replaced.start().await.expect("start the replacement");
    assert_eq!(
        replaced.read_file("notes/today.txt").await.unwrap(),
        b"written before the pause"
    );
    assert_eq!(replaced.read_file("seed.txt").await.unwrap(), b"seed\n");
    let id = replaced.id().to_string();
    client.delete(replaced).await.expect("delete");
    assert!(inspect(&id).is_none());
}

/// `network_disabled` gives the container no network stack, and the persisted setting survives
/// both resume branches.
#[tokio::test]
async fn live_docker_network_isolation_survives_resume() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    let manifest = Manifest::new().with_root("/workspace");
    let client = DockerSandboxClient::with_options(options(&image).with_network_disabled(true));

    let interfaces = |session: Arc<dyn SandboxSession>| async move {
        // `/proc/net/dev` lists every interface of the container's network namespace.
        let (code, stdout) = shell(
            &session,
            "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' ' | sort | tr '\\n' ' '",
        )
        .await;
        assert_eq!(code, 0);
        stdout.trim().to_string()
    };

    let session = client.create(Some(manifest.clone())).await.expect("create");
    containers.track(&session);
    session.start().await.expect("start");
    let original = session.id().to_string();
    assert_eq!(
        inspect(&original).unwrap()["HostConfig"]["NetworkMode"],
        "none"
    );
    assert_eq!(interfaces(session.clone()).await, "lo");
    let state = session.session_state();
    assert_eq!(
        state.provider_metadata(DOCKER_NETWORK_DISABLED_KEY),
        Some(&serde_json::Value::Bool(true))
    );
    session.stop().await.unwrap();

    // A client configured *with* networking still honours the persisted isolation.
    let networked = DockerSandboxClient::with_options(options(&image));
    let reattached = networked
        .resume(&state, manifest.clone())
        .await
        .expect("reattach passes the network check");
    assert_eq!(reattached.id(), original);
    reattached.start().await.unwrap();
    assert_eq!(interfaces(reattached.clone()).await, "lo");
    networked.delete(reattached).await.unwrap();

    let replaced = networked
        .resume(&state, manifest)
        .await
        .expect("resume replaces");
    containers.track(&replaced);
    assert_ne!(replaced.id(), original);
    assert_eq!(
        inspect(replaced.id()).unwrap()["HostConfig"]["NetworkMode"],
        "none"
    );
    replaced.start().await.unwrap();
    assert_eq!(interfaces(replaced.clone()).await, "lo");
    networked.delete(replaced).await.unwrap();
}

/// A published port resolves to the daemon-assigned `127.0.0.1` port, and a server in the
/// sandbox answers on it from the host.
#[tokio::test]
async fn live_docker_exposed_port_reaches_a_server_in_the_sandbox() {
    let Some(image) = live_image().await else {
        return;
    };
    let containers = Containers::default();
    let client = DockerSandboxClient::with_options(options(&image).with_exposed_port(8000));
    let session = client
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create");
    containers.track(&session);
    session.start().await.expect("start");
    if shell(&session, "command -v python3").await.0 != 0 {
        eprintln!("skipping the port check: `{image}` has no python3");
        client.delete(session).await.unwrap();
        return;
    }
    session.write_file("index.html", b"served").await.unwrap();
    let (code, _) = shell(
        &session,
        "nohup python3 -m http.server 8000 --bind 0.0.0.0 >/dev/null 2>&1 &",
    )
    .await;
    assert_eq!(code, 0);

    let endpoint = session
        .resolve_exposed_port(8000)
        .await
        .expect("the port is published");
    assert_eq!(endpoint.host, "127.0.0.1");
    assert_ne!(endpoint.port, 0);
    let mut body = String::new();
    for _ in 0..50 {
        body.clear();
        if let Ok(mut stream) =
            std::net::TcpStream::connect((endpoint.host.as_str(), endpoint.port))
        {
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let _ = stream.write_all(b"GET /index.html HTTP/1.0\r\nHost: localhost\r\n\r\n");
            let _ = stream.read_to_string(&mut body);
            if body.contains("served") {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(body.starts_with("HTTP/1.0 200"), "{body:?}");
    assert!(body.ends_with("served"), "{body:?}");

    // A port the client never published is refused rather than guessed.
    let refused = session.resolve_exposed_port(9).await.unwrap_err();
    assert_eq!(
        refused.context.get("reason").map(String::as_str),
        Some("not_configured"),
        "{refused:?}"
    );
    client.delete(session).await.unwrap();
}
