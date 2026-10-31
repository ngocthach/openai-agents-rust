//! Ports of the exec, file-operation, archive and resume tests in `tests/sandbox/test_docker.py`.
//!
//! Python drives `docker-py` fakes; this port drives the `DockerRunner` seam, so every assertion
//! is on the argument vector that would reach the `docker` executable and the bytes piped into
//! it. No daemon is involved.

#![cfg(feature = "sandbox")]

use std::collections::BTreeMap;
use std::io::Write as _;
use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::mounts::{Mount, docker_volume_name};
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DOCKER_LABELS_KEY, DOCKER_SESSION_ID_KEY, DockerRunner, DockerSandboxClient,
    DockerSandboxClientOptions, DockerSandboxSession,
};
use openai_agents_sdk::sandbox::session::ExecRequest;
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxResult, SandboxSession,
};

// ----------------------------------------------------------------------------------------------
// The fake `docker` executable
// ----------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DockerCall {
    argv: Vec<String>,
    stdin: Option<Vec<u8>>,
}

struct FakeDocker {
    calls: Mutex<Vec<DockerCall>>,
    #[allow(clippy::type_complexity)]
    responder: Box<dyn Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync>,
}

impl std::fmt::Debug for FakeDocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeDocker").finish_non_exhaustive()
    }
}

impl FakeDocker {
    fn new() -> Arc<Self> {
        FakeDocker::scripted(|argv| Ok(default_response(argv)))
    }

    fn scripted(
        responder: impl Fn(&[String]) -> SandboxResult<ExecResult> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(FakeDocker {
            calls: Mutex::new(Vec::new()),
            responder: Box::new(responder),
        })
    }

    fn calls(&self) -> Vec<DockerCall> {
        self.calls.lock().expect("calls").clone()
    }

    fn joined(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .map(|call| call.argv.join(" "))
            .collect()
    }

    /// The bytes piped into the first invocation that received any.
    fn piped(&self) -> Option<Vec<u8>> {
        self.calls().into_iter().find_map(|call| call.stdin)
    }
}

#[async_trait::async_trait]
impl DockerRunner for FakeDocker {
    async fn run(&self, argv: &[String], stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().expect("calls").push(DockerCall {
            argv: argv.to_vec(),
            stdin: stdin.map(<[u8]>::to_vec),
        });
        (self.responder)(argv)
    }
}

fn out(stdout: &str) -> ExecResult {
    ExecResult {
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
        exit_code: 0,
    }
}

fn bytes_out(stdout: Vec<u8>) -> ExecResult {
    ExecResult {
        stdout,
        stderr: Vec::new(),
        exit_code: 0,
    }
}

fn fail(exit_code: i32, stderr: &str) -> ExecResult {
    ExecResult {
        stdout: Vec::new(),
        stderr: stderr.as_bytes().to_vec(),
        exit_code,
    }
}

/// Whether `argv` is a call to the in-container path-resolution helper.
fn is_helper(argv: &[String]) -> bool {
    argv.iter()
        .any(|arg| arg.contains("resolve-workspace-path-"))
}

/// What the real helper answers for a path inside the workspace: the candidate, unchanged.
fn helper_response(argv: &[String]) -> ExecResult {
    let at = argv
        .iter()
        .position(|arg| arg.contains("resolve-workspace-path-"))
        .expect("helper path");
    // `test -x <helper>` has the path last; a resolve call is `<helper> <root> <candidate> ...`.
    match argv.get(at + 2) {
        Some(candidate) => out(candidate),
        None => out(""),
    }
}

fn default_response(argv: &[String]) -> ExecResult {
    if is_helper(argv) {
        return helper_response(argv);
    }
    match argv.first().map(String::as_str) {
        Some("create") => out("container-1\n"),
        Some("container") | Some("inspect") => fail(1, "Error: No such object"),
        _ => out(""),
    }
}

/// A session attached to `container-under-test`, with no client behind it.
fn session(manifest: Manifest, runner: Arc<FakeDocker>) -> DockerSandboxSession {
    DockerSandboxSession::new("container-under-test", manifest).with_runner(runner)
}

/// A one-member gzipped tar, as the container's own `tar -czf -` produces.
fn gz_tar(members: &[(&str, Option<&str>)]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, link_target) in members {
        if name.ends_with('/') {
            let mut header = tar::Header::new_gnu();
            header.set_mode(0o755);
            header.set_entry_type(tar::EntryType::Directory);
            header.set_size(0);
            header.as_old_mut().name[..name.len()].copy_from_slice(name.as_bytes());
            header.set_cksum();
            builder.append(&header, std::io::empty()).expect("dir");
            continue;
        }
        // The name is written into the header by hand: `tar::Builder` refuses the absolute and
        // `..` names these fixtures exist to carry.
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o644);
        header.as_old_mut().name[..name.len()].copy_from_slice(name.as_bytes());
        match link_target {
            Some(target) => {
                header.set_entry_type(tar::EntryType::Symlink);
                header.set_size(0);
                header.as_old_mut().linkname[..target.len()].copy_from_slice(target.as_bytes());
                header.set_cksum();
                builder.append(&header, std::io::empty()).expect("link");
            }
            None => {
                header.set_size(2);
                header.set_cksum();
                builder.append(&header, &b"hi"[..]).expect("data");
            }
        }
    }
    let raw = builder.into_inner().expect("tar");
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(&raw).expect("gzip");
    encoder.finish().expect("gzip")
}

/// Every `(name, link target)` pair in a gzipped tar.
fn gz_tar_links(archive: &[u8]) -> BTreeMap<String, Option<String>> {
    let decoder = flate2::read::GzDecoder::new(archive);
    let mut tar = tar::Archive::new(decoder);
    tar.entries()
        .expect("entries")
        .map(|entry| {
            let entry = entry.expect("entry");
            let name = entry
                .path()
                .expect("path")
                .to_string_lossy()
                .trim_start_matches("./")
                .to_string();
            let target = entry
                .link_name()
                .ok()
                .flatten()
                .map(|path| path.to_string_lossy().into_owned());
            (name, target)
        })
        .collect()
}

fn host_grant(path: &str, host: &str, read_only: bool) -> SandboxPathGrant {
    SandboxPathGrant::new(path)
        .expect("grant path")
        .with_host_path(host)
        .expect("host path")
        .read_only(read_only)
}

/// A `container inspect` document with the attributes the resume checks read.
fn inspect_document(
    mounts: serde_json::Value,
    labels: serde_json::Value,
    ports: serde_json::Value,
) -> String {
    serde_json::json!([{
        "HostConfig": {"NetworkMode": "bridge"},
        "NetworkSettings": {"Networks": {"bridge": {}}, "Ports": ports},
        "Mounts": mounts,
        "Config": {"Labels": labels},
    }])
    .to_string()
}

// ----------------------------------------------------------------------------------------------
// `docker exec`
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_exec_uses_manifest_root_as_workdir_after_workspace_ready` and
/// `test_docker_exec_uses_native_docker_user_without_sudo` — a requested user becomes
/// `docker exec -u <user>`, never a `sudo` prefix inside the container.
#[tokio::test]
async fn exec_runs_in_the_workspace_root_and_switches_user_natively() {
    let runner = FakeDocker::new();
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    session
        .exec(ExecRequest::shell("whoami").with_user(Some("sandbox-user".into())))
        .await
        .expect("exec");

    assert_eq!(
        runner.calls()[0].argv,
        vec![
            "exec",
            "-u",
            "sandbox-user",
            "-w",
            "/workspace",
            "container-under-test",
            "sh",
            "-lc",
            "whoami"
        ]
    );
    // No `sudo` anywhere: Docker switches the user itself.
    assert!(
        !runner.joined()[0].contains("sudo"),
        "{:?}",
        runner.joined()
    );
}

/// Python: `test_docker_exec_timeout_uses_shared_executor` — the point that survives the
/// architecture change is that a command that outlives its timeout fails as a timeout, naming
/// the command, rather than hanging or being reported as a transport error.
#[tokio::test]
async fn exec_reports_a_timeout_and_names_the_command() {
    #[derive(Debug)]
    struct Slow;
    #[async_trait::async_trait]
    impl DockerRunner for Slow {
        async fn run(&self, _argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Ok(out(""))
        }
    }
    let session = DockerSandboxSession::new(
        "container-under-test",
        Manifest::new().with_root("/workspace"),
    )
    .with_runner(Arc::new(Slow));

    let error = session
        .exec(
            ExecRequest::argv(["sleep", "10"])
                .with_timeout(Some(std::time::Duration::from_millis(10))),
        )
        .await
        .expect_err("a command that outlives its timeout fails");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert!(error.message.contains("sleep 10"), "{error}");
}

/// A command that is cut short with no exit status is still reported with a code, so a caller
/// never sees a success it did not get. Python raises `ExecTransportError` with
/// `reason=missing_exit_code` (`test_docker_exec_unknown_exit_code_is_transport_error`); this
/// port's CLI transport reports `-1`, which `ok()` refuses just the same.
#[tokio::test]
async fn exec_never_reports_a_missing_exit_status_as_success() {
    let runner = FakeDocker::scripted(|_| {
        Ok(ExecResult {
            stdout: b"partial stdout".to_vec(),
            stderr: b"partial stderr".to_vec(),
            exit_code: -1,
        })
    });
    let session = session(Manifest::new().with_root("/workspace"), runner);

    let result = session
        .exec(ExecRequest::argv(["find", "."]))
        .await
        .expect("exec");

    // DEFERRED: Python raises `ExecTransportError{reason: missing_exit_code}`; the CLI transport
    // in `run_piped` cannot distinguish "no status" from "killed by a signal", so it reports
    // `-1`. A caller that checks `ok()` — which every SDK caller does — behaves identically.
    assert!(!result.ok());
    assert_eq!(result.exit_code, -1);
    assert_eq!(result.stdout_text(), "partial stdout");
    assert_eq!(result.stderr_text(), "partial stderr");
}

/// Python: `test_docker_exists_returns_false_for_missing_container` — a container the daemon no
/// longer knows is not running.
#[tokio::test]
async fn a_missing_container_is_not_running() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("inspect") {
            return Ok(fail(1, "Error: No such object: missing"));
        }
        Ok(default_response(argv))
    });
    let session = session(Manifest::new(), runner.clone());

    assert!(!session.running().await.expect("running"));
    assert_eq!(
        runner.calls()[0].argv,
        vec![
            "inspect",
            "-f",
            "{{.State.Running}}",
            "container-under-test"
        ]
    );
}

// ----------------------------------------------------------------------------------------------
// Published ports
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_resolve_exposed_port_reads_published_port_mapping`. The daemon assigns
/// the host port, so the endpoint can only be read back from the container.
#[tokio::test]
async fn resolve_exposed_port_reads_the_published_port_mapping() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("container") {
            return Ok(out(
                r#"{"8765/tcp":[{"HostIp":"127.0.0.1","HostPort":"45123"}]}"#,
            ));
        }
        Ok(default_response(argv))
    });
    let session = DockerSandboxSession::new("container-under-test", Manifest::new())
        .with_runner(runner)
        .with_resume_identity(false, BTreeMap::new(), vec![8765]);

    let endpoint = session.resolve_exposed_port(8765).await.expect("endpoint");

    assert_eq!(endpoint.host, "127.0.0.1");
    assert_eq!(endpoint.port, 45123);
    assert!(!endpoint.tls);
}

/// An empty `HostIp` means every interface; Python substitutes the loopback address, because the
/// SDK publishes on the loopback in the first place.
#[tokio::test]
async fn an_empty_host_ip_resolves_to_the_loopback_address() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("container") {
            return Ok(out(r#"{"8765/tcp":[{"HostIp":"","HostPort":"45123"}]}"#));
        }
        Ok(default_response(argv))
    });
    let session = DockerSandboxSession::new("container-under-test", Manifest::new())
        .with_runner(runner)
        .with_resume_identity(false, BTreeMap::new(), vec![8765]);

    assert_eq!(
        session
            .resolve_exposed_port(8765)
            .await
            .expect("endpoint")
            .host,
        "127.0.0.1"
    );
}

/// A port the client never declared is refused without asking the daemon, and the refusal is
/// non-retryable: it can never start working.
#[tokio::test]
async fn a_port_that_was_never_published_is_refused_without_asking_the_daemon() {
    let runner = FakeDocker::new();
    let session = session(Manifest::new(), runner.clone());

    let error = session
        .resolve_exposed_port(8765)
        .await
        .expect_err("an undeclared port is refused");
    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("not_configured")
    );
    assert!(runner.calls().is_empty(), "{:?}", runner.joined());
}

/// A declared port the daemon has not published yet is a backend failure, not a configuration
/// one — Python's `detail=port_not_published`.
#[tokio::test]
async fn a_declared_port_the_daemon_has_not_published_is_a_backend_failure() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("container") {
            return Ok(out(r#"{"8765/tcp":null}"#));
        }
        Ok(default_response(argv))
    });
    let session = DockerSandboxSession::new("container-under-test", Manifest::new())
        .with_runner(runner)
        .with_resume_identity(false, BTreeMap::new(), vec![8765]);

    let error = session
        .resolve_exposed_port(8765)
        .await
        .expect_err("an unpublished port is refused");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("backend_unavailable")
    );
    assert_eq!(
        error.context.get("detail").map(String::as_str),
        Some("port_not_published")
    );
}

// ----------------------------------------------------------------------------------------------
// Workspace file operations
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_read_and_write_reject_paths_outside_workspace_root`.
#[tokio::test]
async fn read_and_write_refuse_a_path_outside_the_workspace_root() {
    let runner = FakeDocker::new();
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    for path in ["../secret.txt", "/etc/passwd"] {
        let read = session.read_file(path).await.expect_err("read is refused");
        assert_eq!(read.code, ErrorCode::InvalidManifestPath, "{read}");
        let write = session
            .write_file(path, b"nope")
            .await
            .expect_err("write is refused");
        assert_eq!(write.code, ErrorCode::InvalidManifestPath, "{write}");
    }
    assert!(runner.calls().is_empty(), "{:?}", runner.joined());
}

/// Python: `test_docker_read_returns_file_bytes_without_archive_api` — a read is a `cat`, not an
/// archive download, so binary content comes back byte for byte.
#[tokio::test]
async fn read_file_returns_the_bytes_without_an_archive_api() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.last().map(String::as_str) == Some("/workspace/hello.bin") {
            return Ok(bytes_out(b"hello\x00world".to_vec()));
        }
        Ok(default_response(argv))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    let data = session.read_file("hello.bin").await.expect("read");

    assert_eq!(data, b"hello\x00world");
    let cat = runner
        .calls()
        .into_iter()
        .find(|call| call.argv.contains(&"cat".to_string()))
        .expect("a cat call");
    assert_eq!(
        cat.argv,
        vec![
            "exec",
            "-w",
            "/workspace",
            "container-under-test",
            "cat",
            "--",
            "/workspace/hello.bin"
        ]
    );
    assert!(
        !runner.joined().iter().any(|call| call.contains("tar")),
        "{:?}",
        runner.joined()
    );
}

/// Python: `test_docker_read_missing_path_raises_not_found_error`.
#[tokio::test]
async fn read_file_reports_a_missing_path_as_not_found() {
    let runner = FakeDocker::scripted(|argv| {
        if is_helper(argv) {
            return Ok(default_response(argv));
        }
        Ok(fail(
            1,
            "cat: /workspace/missing.txt: No such file or directory",
        ))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner);

    let error = session
        .read_file("missing.txt")
        .await
        .expect_err("a missing path is not found");
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
}

/// Python: `test_docker_read_existing_unreadable_path_raises_archive_error` — reading a
/// directory is an error, distinct from "not found".
#[tokio::test]
async fn read_file_reports_an_unreadable_path_as_a_different_error_than_not_found() {
    let runner = FakeDocker::scripted(|argv| {
        if is_helper(argv) {
            return Ok(default_response(argv));
        }
        Ok(fail(1, "cat: /workspace/directory: Is a directory"))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner);

    let error = session
        .read_file("directory")
        .await
        .expect_err("a directory cannot be read");
    // Python raises `WorkspaceArchiveReadError`, distinct from the not-found error.
    assert_ne!(error.code, ErrorCode::WorkspaceReadNotFound);
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
}

/// Python: `test_docker_write_rejects_read_only_extra_path_grant` — the SDK refuses the write
/// itself, with the grant named in the error context, before anything reaches the container.
#[tokio::test]
async fn write_file_refuses_a_read_only_path_grant_before_the_daemon_is_touched() {
    let runner = FakeDocker::new();
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(
            SandboxPathGrant::new("/tmp")
                .expect("grant")
                .read_only(true),
        );
    let session = session(manifest, runner.clone());

    let error = session
        .write_file("/tmp/result.txt", b"scratch output")
        .await
        .expect_err("a read-only grant refuses a write");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );
    assert_eq!(
        error.context.get("grant_path").map(String::as_str),
        Some("/tmp")
    );
    assert!(runner.calls().is_empty(), "{:?}", runner.joined());

    // Reading through the same grant still works, which is the point of a read-only grant.
    let read_only_read = session.read_file("/tmp/result.txt").await;
    assert!(read_only_read.is_ok(), "{read_only_read:?}");
}

/// Python: `test_docker_read_uses_sandbox_target_for_split_path_grant` — a grant whose host path
/// differs from its sandbox path is read at the *sandbox* path, because that is where the
/// container sees it.
#[tokio::test]
async fn read_file_uses_the_sandbox_target_of_a_split_path_grant() {
    let runner = FakeDocker::scripted(|argv| {
        if is_helper(argv) {
            return Ok(default_response(argv));
        }
        if argv.last().map(String::as_str) == Some("/tmp/result.txt") {
            return Ok(bytes_out(b"scratch output".to_vec()));
        }
        Ok(fail(1, "No such file or directory"))
    });
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(host_grant("/tmp", "/host/native-source", false));
    let session = session(manifest, runner.clone());

    let data = session.read_file("/tmp/result.txt").await.expect("read");

    assert_eq!(data, b"scratch output");
    // The host path never appears in the container-side command.
    assert!(
        !runner.joined()[0].contains("/host/native-source"),
        "{:?}",
        runner.joined()
    );
}

/// The write pipes the payload into the container and creates the parent directory, and the
/// destination is passed positionally so a path with shell metacharacters cannot inject.
#[tokio::test]
async fn write_file_pipes_the_payload_and_never_interpolates_the_destination() {
    let runner = FakeDocker::new();
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    session
        .write_file("sub/$(touch pwned).txt", b"payload")
        .await
        .expect("write");

    let calls = runner.calls();
    let argv = &calls
        .iter()
        .find(|call| call.argv.get(1).map(String::as_str) == Some("-i"))
        .expect("the piped write")
        .argv;
    assert_eq!(
        argv[..5],
        ["exec", "-i", "-w", "/workspace", "container-under-test"]
    );
    assert_eq!(
        argv[5..9],
        [
            "sh",
            "-c",
            "mkdir -p \"$(dirname \"$1\")\" && cat > \"$1\"",
            "sh"
        ]
    );
    assert_eq!(argv[9], "/workspace/sub/$(touch pwned).txt");
    assert_eq!(runner.piped().as_deref(), Some(&b"payload"[..]));
}

// ----------------------------------------------------------------------------------------------
// Hydration
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_hydrate_workspace_rejects_unsafe_tar_members`, both parametrizations.
/// The archive is judged on the host, before a single byte reaches the container's `tar`.
#[tokio::test]
async fn hydrate_refuses_an_unsafe_member_before_anything_is_extracted() {
    for member in ["/etc/passwd", "../escape.txt"] {
        let runner = FakeDocker::new();
        let session = session(Manifest::new().with_root("/workspace"), runner.clone());

        let error = session
            .hydrate_workspace(&gz_tar(&[(member, None)]))
            .await
            .expect_err("an unsafe member is refused");

        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveWriteError,
            "{member}"
        );
        assert!(runner.calls().is_empty(), "{member}: {:?}", runner.joined());
    }
}

/// Python: `test_docker_hydrate_workspace_rejects_workspace_root_symlink` — an archive that
/// would replace the workspace root itself with a symlink never reaches `tar`.
#[tokio::test]
async fn hydrate_refuses_an_archive_that_replaces_the_workspace_root_with_a_link() {
    let runner = FakeDocker::new();
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    let error = session
        .hydrate_workspace(&gz_tar(&[(".", Some("/tmp/outside"))]))
        .await
        .expect_err("an archive root symlink is refused");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert!(runner.calls().is_empty(), "{:?}", runner.joined());
}

/// Python: `test_docker_hydrate_workspace_reads_archive_in_bounded_chunks` asserts the stream is
/// consumed in bounded reads and handed to `tar -x -C <root>`. The CLI transport streams the
/// bytes into `docker exec -i`, so the observable is the command and the piped payload.
#[tokio::test]
async fn hydrate_streams_the_archive_into_tar_under_the_workspace_root() {
    let runner = FakeDocker::new();
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());
    let archive = gz_tar(&[("hello.txt", None)]);

    session.hydrate_workspace(&archive).await.expect("hydrate");

    assert_eq!(
        runner.calls()[0].argv,
        vec![
            "exec",
            "-i",
            "container-under-test",
            "sh",
            "-c",
            "mkdir -p /workspace && tar -xzf - -C /workspace"
        ]
    );
    assert_eq!(runner.piped().as_deref(), Some(archive.as_slice()));
}

// ----------------------------------------------------------------------------------------------
// Workspace capture
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_persist_workspace_stages_copy_before_get_archive` — the archive holds
/// workspace-relative members, never a `workspace/` prefix, and the workspace root itself is
/// never downloaded as an archive. This port gets that by archiving `-C <root> .` in the
/// container rather than by staging a copy, so the assertion is on the capture command.
#[tokio::test]
async fn persist_captures_the_archive_relative_to_the_workspace_root() {
    let archive = gz_tar(&[("./README.md", None)]);
    let captured = archive.clone();
    let runner = FakeDocker::scripted(move |argv| {
        if argv.iter().any(|arg| arg.contains("tar -czf -")) {
            return Ok(bytes_out(captured.clone()));
        }
        Ok(default_response(argv))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    let persisted = session.persist_workspace_once().await.expect("persist");

    let command = runner.joined().join(" | ");
    assert!(command.contains("tar -czf - -C /workspace"), "{command}");
    assert!(command.ends_with(" ."), "{command}");
    assert_eq!(persisted, archive);
    assert!(
        gz_tar_links(&persisted)
            .keys()
            .all(|name| !name.starts_with("workspace/")),
        "{:?}",
        gz_tar_links(&persisted)
    );
}

/// Python: `test_docker_persist_workspace_prunes_ephemeral_entries_from_staged_copy`,
/// `..._prunes_mount_paths_without_mount_lifecycle`, `..._prunes_explicit_mount_path...` and
/// `..._prunes_nested_mount_paths...`. Python prunes a staged copy; this port excludes the same
/// paths from the container's own `tar`, which is the same archive without the copy.
#[tokio::test]
async fn persist_excludes_ephemeral_entries_and_every_mount_path() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.iter().any(|arg| arg.contains("tar -czf -")) {
            return Ok(bytes_out(gz_tar(&[("./keep.txt", None)])));
        }
        Ok(default_response(argv))
    });
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry(
            "skip.txt",
            Entry::file(b"skip".to_vec()).with_ephemeral(true),
        )
        .with_entry(
            "repo",
            Entry::dir().with_child(
                "mount",
                Entry::mount(Mount::new(Arc::new(ConfiguredMountStrategy::rclone(
                    S3Mount::new("bucket"),
                )))),
            ),
        )
        .with_entry(
            "logical",
            Entry::mount(
                Mount::new(Arc::new(ConfiguredMountStrategy::rclone(S3Mount::new(
                    "bucket",
                ))))
                .with_mount_path("actual"),
            ),
        );
    let session = session(manifest, runner.clone());

    session.persist_workspace_once().await.expect("persist");

    let command = runner
        .joined()
        .into_iter()
        .find(|call| call.contains("tar -czf -"))
        .expect("the capture command");
    for excluded in ["./skip.txt", "./repo/mount", "./actual"] {
        assert!(
            command.contains(&format!("--exclude {excluded}")),
            "{excluded} is not excluded: {command}"
        );
    }
    // The sibling of a pruned mount is still archived.
    assert!(!command.contains("--exclude ./keep.txt"), "{command}");
}

/// Python: `test_docker_persist_and_hydrate_keep_absolute_workspace_symlinks_resolving`. A link
/// recorded against the container's own root is rewritten relative to the link's directory, so
/// the strict hydrate extractor — which refuses an absolute target — can restore it.
#[tokio::test]
async fn persist_rebases_an_absolute_in_workspace_symlink_so_hydrate_accepts_it() {
    let captured = gz_tar(&[
        ("./", None),
        ("./sub/", None),
        ("./sub/data.txt", None),
        ("./sub/abs_up", Some("/workspace/sub/data.txt")),
    ]);
    let runner = FakeDocker::scripted(move |argv| {
        if argv.iter().any(|arg| arg.contains("tar -czf -")) {
            return Ok(bytes_out(captured.clone()));
        }
        Ok(default_response(argv))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    let persisted = session.persist_workspace_once().await.expect("persist");

    assert_eq!(
        gz_tar_links(&persisted).get("sub/abs_up"),
        Some(&Some("../sub/data.txt".to_string()))
    );
    // And the rebased archive is one hydrate accepts.
    session
        .validate_restorable_archive(&persisted)
        .expect("a rebased archive is restorable");
}

/// Python: `test_docker_persist_keeps_unsupported_symlink_targets_absolute_for_hydrate`. A
/// target the archive cannot prove stays inside the workspace keeps its absolute form, and
/// strict hydration refuses it rather than restoring a link out of the sandbox.
#[tokio::test]
async fn persist_leaves_an_unprovable_symlink_target_absolute_and_hydrate_refuses_it() {
    for target in ["/workspace/alias/../data.txt", "/etc/passwd"] {
        let captured = gz_tar(&[
            ("./data.txt", None),
            ("./alias", Some("sub/deep")),
            ("./abs_alias", Some(target)),
        ]);
        let runner = FakeDocker::scripted(move |argv| {
            if argv.iter().any(|arg| arg.contains("tar -czf -")) {
                return Ok(bytes_out(captured.clone()));
            }
            Ok(default_response(argv))
        });
        let session = session(Manifest::new().with_root("/workspace"), runner);

        let persisted = session.persist_workspace_once().await.expect("persist");

        assert_eq!(
            gz_tar_links(&persisted).get("abs_alias"),
            Some(&Some(target.to_string())),
            "{target}"
        );
        // A relative link inside the workspace is untouched.
        assert_eq!(
            gz_tar_links(&persisted).get("alias"),
            Some(&Some("sub/deep".to_string()))
        );
        let error = session
            .validate_restorable_archive(&persisted)
            .expect_err("an absolute target is refused");
        assert_eq!(
            error.code,
            ErrorCode::WorkspaceArchiveWriteError,
            "{target}"
        );
    }
}

// ----------------------------------------------------------------------------------------------
// Resume
// ----------------------------------------------------------------------------------------------

/// Python: `test_docker_resume_reconnects_serialized_credentialless_state` and
/// `test_docker_resume_preserves_workspace_readiness_from_state`. A container the daemon still
/// knows is reattached, keeping its id and the persisted readiness flag.
#[tokio::test]
async fn resume_reattaches_to_a_live_container_and_keeps_the_persisted_readiness() {
    for ready in [true, false] {
        let runner = FakeDocker::scripted(|argv| {
            if argv.first().map(String::as_str) == Some("container") {
                return Ok(out(&inspect_document(
                    serde_json::json!([]),
                    serde_json::json!({}),
                    serde_json::json!({}),
                )));
            }
            Ok(default_response(argv))
        });
        let client = DockerSandboxClient::new().with_runner(runner.clone());
        let manifest = Manifest::new().with_root("/workspace");
        let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
            .with_provider_id(Some("container".to_string()))
            .with_workspace_root_ready(ready);

        let resumed = client.resume(&state, manifest).await.expect("resume");

        assert_eq!(resumed.id(), "container");
        assert_eq!(resumed.session_state().workspace_root_ready, ready);
        // Reattaching inspects the container; it never creates one.
        assert!(
            !runner
                .joined()
                .iter()
                .any(|call| call.starts_with("create ")),
            "{:?}",
            runner.joined()
        );
    }
}

/// Python: `test_docker_resume_resets_workspace_readiness_when_container_is_recreated` and
/// `test_docker_resume_forwards_persisted_labels_when_recreating_container`. A container that
/// is gone is replaced, the replacement is created with the persisted labels and ports, and the
/// workspace is no longer considered ready.
#[tokio::test]
async fn resume_replaces_a_missing_container_with_the_persisted_identity() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new().with_root("/workspace");
    let mut state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("missing".to_string()))
        .with_workspace_root_ready(true)
        .with_provider_metadata(
            DOCKER_LABELS_KEY,
            serde_json::json!({"com.example.owner": "worker-123"}),
        )
        .with_provider_metadata(DOCKER_SESSION_ID_KEY, serde_json::json!("sess-volumes"));
    state.exposed_ports = vec![8765];

    let resumed = client.resume(&state, manifest).await.expect("resume");

    assert_eq!(resumed.id(), "container-1");
    assert!(!resumed.session_state().workspace_root_ready);
    let create = runner
        .joined()
        .into_iter()
        .find(|call| call.starts_with("create "))
        .expect("a replacement is created");
    assert!(
        create.contains("--label com.example.owner=worker-123"),
        "{create}"
    );
    assert!(create.contains("-p 127.0.0.1::8765/tcp"), "{create}");
    // The replacement keeps the paused session's volume identity, so it attaches the volumes the
    // first container created — and its own `shutdown` removes them again.
    assert_eq!(
        resumed
            .session_state()
            .provider_metadata(DOCKER_SESSION_ID_KEY)
            .and_then(serde_json::Value::as_str),
        Some("sess-volumes")
    );
}

/// Python: `test_docker_resume_reuses_container_with_matching_labels` and
/// `test_docker_resume_rejects_mismatched_existing_labels` (both parametrizations). Extra labels
/// on the container are fine; a persisted label that is missing or different is not.
#[tokio::test]
async fn resume_checks_the_persisted_labels_against_the_live_container() {
    async fn resume_with(labels: serde_json::Value) -> SandboxResult<Arc<dyn SandboxSession>> {
        let runner = FakeDocker::scripted(move |argv| {
            if argv.first().map(String::as_str) == Some("container") {
                return Ok(out(&inspect_document(
                    serde_json::json!([]),
                    labels.clone(),
                    serde_json::json!({}),
                )));
            }
            Ok(default_response(argv))
        });
        let client = DockerSandboxClient::new().with_runner(runner);
        let manifest = Manifest::new().with_root("/workspace");
        let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
            .with_provider_id(Some("container".to_string()))
            .with_provider_metadata(
                DOCKER_LABELS_KEY,
                serde_json::json!({"com.example.owner": "worker-123"}),
            );
        client.resume(&state, manifest).await
    }

    // An extra label the SDK never asked for does not stop the reattach.
    let reused = resume_with(serde_json::json!({
        "com.example.owner": "worker-123",
        "com.example.extra": "preserved",
    }))
    .await
    .expect("a container with the persisted labels is reused");
    assert_eq!(reused.id(), "container");

    for actual in [
        serde_json::json!({}),
        serde_json::json!({"com.example.owner": "different"}),
    ] {
        let error = resume_with(actual.clone())
            .await
            .err()
            .unwrap_or_else(|| panic!("{actual} must be refused"));
        assert_eq!(error.code, ErrorCode::SessionStateMismatch);
        assert!(error.message.contains("labels"), "{error}");
    }
}

/// Python: `test_docker_resume_requires_existing_host_mount_to_match_trusted_state`, both root
/// parametrizations. A container whose bind mount matches the trusted manifest is reattached; one
/// whose mounts do not is refused.
#[tokio::test]
async fn resume_requires_a_live_bind_mount_to_match_the_trusted_manifest() {
    for root in ["/workspace", "/mnt/sibling/../workspace"] {
        let matching = serde_json::json!([{
            "Type": "bind",
            "Source": "/host/shared-data",
            "Destination": "/mnt/shared-data",
            "RW": false,
        }]);
        for (mounts, should_resume) in [
            (matching.clone(), true),
            (serde_json::json!([]), false),
            (
                serde_json::json!([{
                    "Type": "bind",
                    "Source": "/host/shared-data",
                    "Destination": "/mnt/shared-data",
                    "RW": true,
                }]),
                false,
            ),
        ] {
            let runner = FakeDocker::scripted(move |argv| {
                if argv.first().map(String::as_str) == Some("container") {
                    return Ok(out(&inspect_document(
                        mounts.clone(),
                        serde_json::json!({}),
                        serde_json::json!({}),
                    )));
                }
                Ok(default_response(argv))
            });
            let client = DockerSandboxClient::new().with_runner(runner);
            let manifest = Manifest::new()
                .with_root(root)
                .with_extra_path_grant(host_grant("/mnt/shared-data", "/host/shared-data", true));
            let state = SandboxSessionState::new("docker", "sess-1", root, &manifest)
                .with_provider_id(Some("container".to_string()));

            let resumed = client.resume(&state, manifest).await;
            if should_resume {
                assert_eq!(
                    resumed.expect("a matching mount reattaches").id(),
                    "container"
                );
            } else {
                let Err(error) = resumed else {
                    panic!("a mismatched mount is refused");
                };
                assert_eq!(error.code, ErrorCode::SessionStateMismatch);
                assert!(
                    error
                        .message
                        .contains("does not match the current trusted manifest")
                        || error
                            .message
                            .contains("not present in the current trusted manifest"),
                    "{error}"
                );
            }
        }
    }
}

/// Python: `test_docker_resume_rejects_stale_bind_mount_for_path_only_grant` and
/// `..._rejects_bind_mount_when_persisted_grants_are_removed`. A bind mount the trusted manifest
/// does not describe is how a tampered state would smuggle a host directory in, so it is refused
/// even when the manifest has no host grants at all.
#[tokio::test]
async fn resume_rejects_a_bind_mount_the_trusted_manifest_does_not_describe() {
    for manifest in [
        // A path-only grant asks for no mount, so a bind mount at its target is stale.
        Manifest::new()
            .with_root("/workspace")
            .with_extra_path_grant(SandboxPathGrant::new("/mnt/shared-data").expect("grant")),
        // And a manifest with no grants at all must not inherit one from the container.
        Manifest::new().with_root("/workspace"),
    ] {
        let runner = FakeDocker::scripted(|argv| {
            if argv.first().map(String::as_str) == Some("container") {
                return Ok(out(&inspect_document(
                    serde_json::json!([{
                        "Type": "bind",
                        "Source": "/host/shared-data",
                        "Destination": "/mnt/shared-data",
                        "RW": true,
                    }]),
                    serde_json::json!({}),
                    serde_json::json!({}),
                )));
            }
            Ok(default_response(argv))
        });
        let client = DockerSandboxClient::new().with_runner(runner);
        let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
            .with_provider_id(Some("container".to_string()));

        let Err(error) = client.resume(&state, manifest).await else {
            panic!("an undescribed bind mount is refused");
        };
        assert_eq!(error.code, ErrorCode::SessionStateMismatch);
        assert!(
            error
                .message
                .contains("not present in the current trusted manifest"),
            "{error}"
        );
    }
}

/// Python: `test_docker_resume_rejects_host_grant_inside_normalized_persisted_root`, both root
/// parametrizations. The persisted root is normalized before the comparison, so a state that
/// spells the root `/mnt/sibling/..` cannot smuggle a grant inside it.
#[tokio::test]
async fn resume_rejects_a_host_grant_inside_the_normalized_persisted_root() {
    for root in ["/mnt/sibling/..", "//mnt/sibling/.."] {
        let runner = FakeDocker::new();
        let client = DockerSandboxClient::new().with_runner(runner.clone());
        let manifest = Manifest::new()
            .with_root(root)
            .with_extra_path_grant(host_grant("/mnt/shared-data", "/host/shared-data", false));
        let state = SandboxSessionState::new("docker", "sess-1", root, &manifest)
            .with_provider_id(Some("container".to_string()))
            .with_workspace_root_ready(true);

        let error = client
            .resume(&state, manifest)
            .await
            .err()
            .unwrap_or_else(|| panic!("`{root}` must be refused"));
        assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
        assert!(
            error
                .message
                .contains("host_path target must be outside the workspace root"),
            "{error}"
        );
        // Refused before the daemon is touched, and the state is untouched.
        assert!(runner.calls().is_empty(), "{:?}", runner.joined());
        assert_eq!(state.provider_id.as_deref(), Some("container"));
        assert!(state.workspace_root_ready);
    }
}

/// Python: `test_docker_resume_does_not_remove_persisted_volume_selected_by_state` — the volumes
/// a resumed session removes are the ones its persisted identity names, not fresh ones.
#[tokio::test]
async fn a_resumed_session_removes_the_volumes_its_persisted_identity_names() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.first().map(String::as_str) == Some("container") {
            return Ok(out(&inspect_document(
                serde_json::json!([]),
                serde_json::json!({}),
                serde_json::json!({}),
            )));
        }
        Ok(default_response(argv))
    });
    let client = DockerSandboxClient::new().with_runner(runner.clone());
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(Mount::new(Arc::new(
            ConfiguredMountStrategy::docker_volume(S3Mount::new("bucket"), "rclone"),
        ))),
    );
    let state = SandboxSessionState::new("docker", "sess-1", "/workspace", &manifest)
        .with_provider_id(Some("container".to_string()))
        .with_provider_metadata(
            DOCKER_SESSION_ID_KEY,
            serde_json::json!("persisted-session"),
        );

    let resumed = client
        .resume(&state, manifest)
        .await
        .expect("resume")
        .clone();
    resumed.shutdown().await.expect("shutdown");

    let expected = docker_volume_name(Some("persisted-session"), "/workspace/data");
    assert!(
        runner
            .joined()
            .iter()
            .any(|call| call == &format!("volume rm {expected}")),
        "{:?}",
        runner.joined()
    );
}

/// Python: `test_docker_volume_name_is_collision_safe_for_separator_aliases` and
/// `..._uses_strictly_safe_suffix_characters`. The digest differs from Python's (it is not a
/// wire format), so the assertions are the two properties the Python strings encode.
#[test]
fn a_volume_name_is_collision_safe_and_uses_only_characters_docker_accepts() {
    let underscore = docker_volume_name(Some("sess"), "/workspace/a_b");
    let separator = docker_volume_name(Some("sess"), "/workspace/a/b");
    assert_ne!(
        underscore, separator,
        "two mount paths that sanitize to the same suffix must not share a volume"
    );

    let unsafe_path = docker_volume_name(None, "/workspace/data set/@prod");
    assert!(
        unsafe_path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')),
        "{unsafe_path}"
    );
    assert!(
        unsafe_path.ends_with("workspace_data_set__prod"),
        "{unsafe_path}"
    );
    // Without a session id the name carries no session prefix, which is what lets a direct
    // `DockerVolumeMountStrategy` be compared across sessions.
    assert!(!unsafe_path.contains("sess"), "{unsafe_path}");
}

/// A session created with a client that disables the network cannot also publish a port; the
/// refusal happens before `docker create` runs.
#[tokio::test]
async fn a_network_less_container_cannot_publish_a_port() {
    let runner = FakeDocker::new();
    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new("python:3.14-slim")
            .with_network_disabled(true)
            .with_exposed_port(8765),
    )
    .with_runner(runner.clone());

    let error = client
        .create(None)
        .await
        .expect_err("a network-less container cannot publish a port");
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(runner.calls().is_empty(), "{:?}", runner.joined());
}

// ----------------------------------------------------------------------------------------------
// Symlink containment
// ----------------------------------------------------------------------------------------------

/// A runner whose helper answers with `exit_code` and `stderr`, as the real script does when the
/// resolved path leaves the workspace (111) or lands under a read-only grant (114).
fn helper_refusing(exit_code: i32, stderr: &'static str) -> Arc<FakeDocker> {
    FakeDocker::scripted(move |argv| {
        if is_helper(argv) && argv.iter().any(|arg| arg.starts_with("/workspace/")) {
            return Ok(fail(exit_code, stderr));
        }
        Ok(default_response(argv))
    })
}

/// Python: `test_docker_workspace_file_ops_reject_symlink_escape`. A link in the workspace that
/// points outside it is resolved in the container, where the link is real, and the read or write
/// is refused rather than followed.
#[tokio::test]
async fn a_symlink_that_escapes_the_workspace_is_refused_for_read_and_write() {
    let runner = helper_refusing(111, "workspace escape");
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    let read = session
        .read_file("link/secret.txt")
        .await
        .expect_err("a read through an escaping link is refused");
    assert_eq!(read.code, ErrorCode::InvalidManifestPath);
    assert!(read.message.contains("must not escape root"), "{read}");

    let write = session
        .write_file("link/secret.txt", b"overwrite")
        .await
        .expect_err("a write through an escaping link is refused");
    assert_eq!(write.code, ErrorCode::InvalidManifestPath);
    assert!(write.message.contains("must not escape root"), "{write}");

    // Nothing was read or written: only the helper ran.
    assert!(
        !runner
            .joined()
            .iter()
            .any(|call| call.contains(" cat ") || call.contains("cat >")),
        "{:?}",
        runner.joined()
    );
}

/// Python: `test_docker_write_rejects_workspace_symlink_to_read_only_extra_path_grant` and the
/// nested variant. The link is lexically inside the workspace, so only the in-container
/// resolution can see that the write lands under a read-only grant.
#[tokio::test]
async fn a_write_through_a_symlink_into_a_read_only_grant_is_refused() {
    let runner = helper_refusing(
        114,
        "read-only extra path grant: /tmp\nresolved path: /tmp/result.txt\n",
    );
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(
            SandboxPathGrant::new("/tmp")
                .expect("grant")
                .read_only(true),
        );
    let session = session(manifest, runner);

    let error = session
        .write_file("tmp-link/result.txt", b"scratch output")
        .await
        .expect_err("the grant is read-only");

    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.message,
        "failed to write archive for path: /workspace/tmp-link/result.txt"
    );
    assert_eq!(
        error.context.get("path").map(String::as_str),
        Some("/workspace/tmp-link/result.txt")
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );
    assert_eq!(
        error.context.get("grant_path").map(String::as_str),
        Some("/tmp")
    );
    assert_eq!(
        error.context.get("resolved_path").map(String::as_str),
        Some("/tmp/result.txt")
    );
}

/// Python: `test_docker_normalize_path_preserves_safe_leaf_symlink_path`. A leaf link that stays
/// inside the workspace is accepted and read at the path the caller wrote.
#[tokio::test]
async fn a_safe_leaf_symlink_is_read_at_the_path_the_caller_wrote() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.last().map(String::as_str) == Some("/workspace/link.txt") && !is_helper(argv) {
            return Ok(bytes_out(b"hello".to_vec()));
        }
        Ok(default_response(argv))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner);

    assert_eq!(session.read_file("link.txt").await.expect("read"), b"hello");
}

/// The helper is installed once per session, not once per file operation.
#[tokio::test]
async fn the_path_helper_is_installed_once_and_then_reused() {
    let runner = FakeDocker::scripted(|argv| {
        if argv.iter().any(|arg| arg == "test") && is_helper(argv) {
            return Ok(fail(1, ""));
        }
        Ok(default_response(argv))
    });
    let session = session(Manifest::new().with_root("/workspace"), runner.clone());

    session.read_file("a.txt").await.expect("first");
    session.read_file("b.txt").await.expect("second");

    let installs = runner
        .joined()
        .iter()
        .filter(|call| call.contains("OPENAI_AGENTS_HELPER_"))
        .count();
    assert_eq!(installs, 1, "{:?}", runner.joined());
}
