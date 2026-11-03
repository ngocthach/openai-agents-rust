//! Ports of the PTY, tar-hydration and helper groups of `tests/extensions/sandbox/test_modal.py`.
//!
//! The Rust provider drives a Python helper over a JSON protocol, so what Python asserts about
//! the Modal SDK's stdin and exec objects is observed here through the requests the helper
//! receives (`tests/fixtures/fake_modal_helper_scripted.py`), and through the real bundled
//! `helper.py` run against fake SDK objects (`tests/fixtures/modal_helper_driver.py`).
#![cfg(feature = "sandbox")]

use std::time::Duration;

use base64::Engine as _;
use openai_agents_sdk::extensions::sandbox::modal::{
    ModalSandboxClient, ModalSandboxClientOptions,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyWriteRequest};
use openai_agents_sdk::sandbox::{ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSession};
use serde_json::{Value, json};

struct FakeModal {
    directory: tempfile::TempDir,
    pty: &'static str,
}

impl FakeModal {
    fn new() -> FakeModal {
        FakeModal::with_pty("")
    }

    fn with_pty(pty: &'static str) -> FakeModal {
        FakeModal {
            directory: tempfile::tempdir().expect("tempdir"),
            pty,
        }
    }

    fn path(&self, name: &str) -> std::path::PathBuf {
        self.directory.path().join(name)
    }

    fn rules(&self, rules: Value) {
        std::fs::write(self.path("rules.json"), rules.to_string()).expect("write rules");
    }

    fn command(&self) -> Vec<String> {
        vec![
            "/usr/bin/env".to_string(),
            format!("FAKE_MODAL_LOG={}", self.path("requests.jsonl").display()),
            format!("FAKE_MODAL_STATE={}", self.path("state.json").display()),
            format!("FAKE_MODAL_RULES={}", self.path("rules.json").display()),
            format!("FAKE_MODAL_PTY={}", self.pty),
            "python3".to_string(),
            format!(
                "{}/tests/fixtures/fake_modal_helper_scripted.py",
                env!("CARGO_MANIFEST_DIR")
            ),
        ]
    }

    fn of_op(&self, op: &str) -> Vec<Value> {
        std::fs::read_to_string(self.path("requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|request| request["op"] == op)
            .collect()
    }

    fn options(&self) -> ModalSandboxClientOptions {
        ModalSandboxClientOptions::new("agents-app").with_helper_command(self.command())
    }
}

// ---------------------------------------------------------------------------------------------
// PTY
// ---------------------------------------------------------------------------------------------

/// `test_modal_pty_start_and_write_stdin`: a started process answers a write, and `pty_terminate_all`
/// kills the process that the helper proxies inside the sandbox.
#[tokio::test]
async fn pty_start_and_write_stdin() {
    let fake = FakeModal::with_pty("cat");
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let started = session
        .pty_exec_start(PtyExecRequest::new("cat").with_tty(true))
        .await
        .expect("start");
    assert!(started.output_text().contains("ready"), "{started:?}");
    let process_id = started.process_id.expect("still running");

    let mut write = PtyWriteRequest::new(process_id, "hi\n");
    write.yield_time_ms = 800;
    let echoed = session.pty_write_stdin(write).await.expect("write");
    assert!(echoed.output_text().contains("echo:hi"), "{echoed:?}");

    session.pty_terminate_all().await.expect("terminate");
    let scripts: Vec<String> = fake
        .of_op("exec")
        .into_iter()
        .filter_map(|request| request["argv"][2].as_str().map(str::to_string))
        .collect();
    assert!(
        scripts.iter().any(|script| script.contains("kill -TERM")),
        "the sandbox-side process is killed: {scripts:?}"
    );
    // The PTY request travels on the helper's command line, and names the wrapped command.
    let pty = &fake.of_op("pty")[0];
    assert_eq!(pty["tty"], true);
    assert_eq!(pty["workdir"], "/workspace");
    let wrapped = pty["argv"][2].as_str().expect("inner script");
    assert!(wrapped.contains("exec sh -lc cat"), "{wrapped}");
}

/// `test_modal_pty_start_drains_all_buffered_output_after_exit`: a process that finished while its
/// output was still buffered reports all of it together with the exit code.
#[tokio::test]
async fn pty_start_drains_all_buffered_output_after_exit() {
    let fake = FakeModal::with_pty("burst");
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let mut request = PtyExecRequest::new("seq 200");
    request.yield_time_ms = 1500;
    request.max_output_tokens = Some(100_000);
    let started = session.pty_exec_start(request).await.expect("start");
    let mut text = started.output_text();
    let mut update = started;
    if let Some(process_id) = update.process_id {
        // The interpreter was slow to start; poll the process to its end.
        let mut poll = PtyWriteRequest::new(process_id, "");
        poll.yield_time_ms = 5000;
        update = session.pty_write_stdin(poll).await.expect("poll");
        text.push_str(&update.output_text());
    }
    assert!(text.contains("line 0"), "{text}");
    assert!(text.contains("line 199"), "{text}");
    assert_eq!(update.exit_code, Some(0));
    assert_eq!(update.process_id, None);
}

/// `test_modal_pty_keeps_session_live_until_delayed_exit_tail_reaches_eof`: output that arrives
/// after the first window is returned by a later poll, together with the exit code.
#[tokio::test]
async fn pty_keeps_the_session_live_until_the_delayed_tail_arrives() {
    let fake = FakeModal::with_pty("delayed");
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let mut request = PtyExecRequest::new("slow").with_tty(true);
    request.yield_time_ms = 250;
    let mut update = session.pty_exec_start(request).await.expect("start");
    let mut saw_head_while_live = false;
    let mut output = String::new();
    for _ in 0..60 {
        output.push_str(&update.output_text());
        if update.output_text().contains("head") && update.exit_code.is_none() {
            saw_head_while_live = true;
        }
        let Some(process_id) = update.process_id else {
            break;
        };
        let mut poll = PtyWriteRequest::new(process_id, "");
        poll.yield_time_ms = 250;
        update = session.pty_write_stdin(poll).await.expect("poll");
    }

    assert!(saw_head_while_live, "{output}");
    assert!(output.contains("tail"), "{output}");
    assert_eq!(update.exit_code, Some(3));
    assert_eq!(update.process_id, None);
}

/// `test_modal_pty_closes_failed_stream_after_known_exit`: a process that fails reports its exit
/// code and what it wrote to stderr, and leaves no live session behind.
#[tokio::test]
async fn pty_failure_reports_the_exit_code_and_stderr() {
    let fake = FakeModal::with_pty("fail");
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let mut request = PtyExecRequest::new("broken");
    request.yield_time_ms = 1500;
    let started = session.pty_exec_start(request).await.expect("start");
    // A slow interpreter start can outlast the first window; the process is then polled to its
    // end like any other.
    let mut output = started.output_text();
    let mut update = started;
    if let Some(process_id) = update.process_id {
        let mut poll = PtyWriteRequest::new(process_id, "");
        poll.yield_time_ms = 5000;
        update = session.pty_write_stdin(poll).await.expect("poll");
        output.push_str(&update.output_text());
    }

    assert_eq!(update.exit_code, Some(7));
    assert_eq!(update.process_id, None);
    assert!(output.contains("pty failed"), "{output}");
}

/// `test_modal_pty_start_*` request shape: a command that never reaches a shell keeps its argv
/// whole, and the working directory is resolved against the workspace root.
#[tokio::test]
async fn pty_start_resolves_the_working_directory() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let mut request = PtyExecRequest::new("pwd");
    request.cwd = Some("repo".to_string());
    request.yield_time_ms = 1000;
    let update = session.pty_exec_start(request).await.expect("start");

    assert!(update.output_text().starts_with("pty: "), "{update:?}");
    assert_eq!(fake.of_op("pty")[0]["workdir"], "/workspace/repo");
}

// ---------------------------------------------------------------------------------------------
// Start and exec failures
// ---------------------------------------------------------------------------------------------

/// `test_modal_start_wraps_exec_details`: a helper failure while preparing the workspace root is a
/// start error that carries the provider's detail.
#[tokio::test]
async fn start_wraps_exec_failures() {
    let fake = FakeModal::new();
    fake.rules(json!([{
        "op": "exec",
        "contains": "mkdir",
        "response": {"error": "FileNotFoundError: missing-shell"},
    }]));
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");

    let error = session.start().await.expect_err("start fails");

    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
    assert!(
        error.message.starts_with("failed to start session: ")
            && error.message.contains("Modal helper `exec` failed"),
        "{}",
        error.message
    );
    assert!(error.message.contains("FileNotFoundError: missing-shell"));
}

/// `test_modal_pty_start_maps_timeout_failures` / `..._maps_modal_exec_timeout_failures`, for the
/// command path that has a timeout: an overrunning command is `ExecTimeout`, and because Modal
/// cannot cancel a running command the sandbox is terminated.
#[tokio::test]
async fn exec_timeout_terminates_the_sandbox() {
    let fake = FakeModal::new();
    fake.rules(
        json!([{"op": "exec", "contains": "slow", "sleep": 3.0, "response": {"exit_code": 0}}]),
    );
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");
    session.start().await.expect("start");

    let error = session
        .exec(ExecRequest::argv(["slow"]).with_timeout(Some(Duration::from_millis(300))))
        .await
        .expect_err("the command overruns");

    assert_eq!(error.code, ErrorCode::ExecTimeout);
    assert_eq!(fake.of_op("terminate").len(), 1);
}

// ---------------------------------------------------------------------------------------------
// Hydration of a tar archive
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Kind {
    File,
    Dir,
    Symlink,
    Hardlink,
    Fifo,
    CharDevice,
}

/// A raw tar header with `name` written verbatim, so a traversal name survives.
fn member(name: &str, kind: Kind, link: &str) -> tar::Header {
    let mut header = tar::Header::new_gnu();
    {
        let old = header.as_old_mut();
        let bytes = name.as_bytes();
        old.name[..bytes.len()].copy_from_slice(bytes);
    }
    header.set_mode(0o644);
    header.set_size(if matches!(kind, Kind::File) { 1 } else { 0 });
    header.set_entry_type(match kind {
        Kind::File => tar::EntryType::Regular,
        Kind::Dir => tar::EntryType::Directory,
        Kind::Symlink => tar::EntryType::Symlink,
        Kind::Hardlink => tar::EntryType::Link,
        Kind::Fifo => tar::EntryType::Fifo,
        Kind::CharDevice => tar::EntryType::Char,
    });
    if !link.is_empty() {
        header.set_link_name(link).expect("link name");
    }
    header.set_cksum();
    header
}

fn tar_bytes(members: Vec<tar::Header>) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for header in members {
        let body: &[u8] = if header.entry_type() == tar::EntryType::Regular {
            b"x"
        } else {
            b""
        };
        builder.append(&header, body).expect("append");
    }
    builder.into_inner().expect("finish")
}

/// A session whose manifest owns `logs` as an ephemeral directory.
async fn hydration_session(fake: &FakeModal) -> std::sync::Arc<dyn SandboxSession> {
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry("main.py", Entry::file(b"print('hi')\n".to_vec()))
        .with_entry("logs", Entry::dir().with_ephemeral(true));
    ModalSandboxClient::new(fake.options())
        .create(Some(manifest))
        .await
        .expect("create")
}

/// `test_modal_hydrate_tar_rejects_protected_members_before_extracting`: an archive member that
/// overlaps a path the manifest owns as ephemeral is refused before anything is uploaded or run.
#[tokio::test]
async fn hydrate_rejects_protected_members_before_anything_runs() {
    let cases: Vec<(&str, tar::Header, &str)> = vec![
        (
            "a member claiming a path the manifest owns",
            member("logs/events.jsonl", Kind::File, ""),
            "archive member overlaps protected path: logs",
        ),
        (
            "a fifo under the protected prefix",
            member("logs/pipe", Kind::Fifo, ""),
            "unsupported member type",
        ),
        (
            "a character device under the protected prefix",
            member("logs/dev", Kind::CharDevice, ""),
            "unsupported member type",
        ),
        (
            "a symlink under the protected prefix",
            member("logs/escape", Kind::Symlink, "/etc/passwd"),
            "archive member overlaps protected path: logs",
        ),
        (
            "a hardlink under the protected prefix",
            member("logs/link", Kind::Hardlink, "/etc/passwd"),
            "hardlink member not allowed",
        ),
        (
            "a traversal normalizing under the protected prefix",
            member("logs/../../etc/passwd", Kind::File, ""),
            "parent traversal",
        ),
    ];
    for (label, offender, reason) in cases {
        let fake = FakeModal::new();
        let session = hydration_session(&fake).await;
        let archive = tar_bytes(vec![member("main.py", Kind::File, ""), offender]);

        let error = session.hydrate_workspace(&archive).await.expect_err(label);

        assert!(error.message.contains(reason), "{label}: {}", error.message);
        assert!(fake.of_op("exec").is_empty(), "{label}: nothing ran");
        assert!(
            fake.of_op("write_file").is_empty(),
            "{label}: nothing uploaded"
        );
    }
}

/// `test_modal_hydrate_tar_accepts_an_archive_without_protected_paths`: the control. A snapshot
/// that leaves out the ephemeral paths, and keeps a symlink that stays inside the archive, is
/// uploaded whole and extracted into the workspace root.
#[tokio::test]
async fn hydrate_accepts_an_archive_without_protected_paths() {
    let fake = FakeModal::new();
    let session = hydration_session(&fake).await;
    let archive = tar_bytes(vec![
        member(".", Kind::Dir, ""),
        member("./main.py", Kind::File, ""),
        member("./.venv", Kind::Dir, ""),
        member("./.venv/lib64", Kind::Symlink, "lib"),
    ]);

    session.hydrate_workspace(&archive).await.expect("hydrate");

    let writes = fake.of_op("write_file");
    assert_eq!(writes.len(), 1);
    let uploaded = base64::engine::general_purpose::STANDARD
        .decode(writes[0]["data"].as_str().expect("data"))
        .expect("base64");
    assert_eq!(uploaded, archive);
    let scripts: Vec<String> = fake
        .of_op("exec")
        .into_iter()
        .map(|request| request["argv"].to_string())
        .collect();
    assert!(
        scripts
            .iter()
            .any(|argv| argv.contains("tar -xzf") && argv.contains("-C /workspace")),
        "{scripts:?}"
    );
}

/// `test_modal_hydrate_tar_chunks_large_payload_before_draining` (the host half): a large archive
/// reaches the helper in one `write_file` request, never as a command argument, whose size the
/// kernel bounds.
#[tokio::test]
async fn hydrate_stages_a_large_archive_through_write_file() {
    let fake = FakeModal::new();
    let session = hydration_session(&fake).await;
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    let payload = vec![b'q'; 3 * 1024 * 1024];
    header.set_size(payload.len() as u64);
    header.set_mode(0o644);
    header.set_entry_type(tar::EntryType::Regular);
    builder
        .append_data(&mut header, "large.bin", payload.as_slice())
        .expect("append");
    let archive = builder.into_inner().expect("finish");

    session.hydrate_workspace(&archive).await.expect("hydrate");

    let writes = fake.of_op("write_file");
    let staged = base64::engine::general_purpose::STANDARD
        .decode(writes[0]["data"].as_str().expect("data"))
        .expect("base64");
    assert_eq!(staged.len(), archive.len());
    for request in fake.of_op("exec") {
        let argv = request["argv"].to_string();
        assert!(argv.len() < 4096, "no payload in a command line");
    }
}

/// `test_modal_write_chunks_large_payload_before_draining` (the host half): a large file is sent
/// whole in a `write_file` request.
#[tokio::test]
async fn write_file_sends_a_large_payload_in_one_request() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(fake.options())
        .create(None)
        .await
        .expect("create");
    let payload = vec![b'z'; 2 * 1024 * 1024 + 7];

    session
        .write_file("nested/file.bin", &payload)
        .await
        .expect("write");

    let writes = fake.of_op("write_file");
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0]["path"], "/workspace/nested/file.bin");
    let sent = base64::engine::general_purpose::STANDARD
        .decode(writes[0]["data"].as_str().expect("data"))
        .expect("base64");
    assert_eq!(sent, payload);
}

// ---------------------------------------------------------------------------------------------
// The bundled helper, against fake SDK objects
// ---------------------------------------------------------------------------------------------

fn run_driver(case: &str) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let output = std::process::Command::new("python3")
        .arg(format!("{manifest}/tests/fixtures/modal_helper_driver.py"))
        .arg(format!("{manifest}/src/extensions/sandbox/modal/helper.py"))
        .arg(case)
        .output()
        .expect("run python3");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("driver output")
}

/// `test_modal_write_chunks_large_payload_before_draining` and
/// `test_modal_hydrate_tar_chunks_large_payload_before_draining` (the helper half): the helper
/// writes a payload larger than the SDK's stdin buffer in drained chunks, then closes stdin.
#[test]
fn helper_writes_a_large_payload_in_drained_chunks() {
    let result = run_driver("write_chunks");

    assert_eq!(result["response"], json!({"ok": true}));
    let total = 20 * 1024 * 1024 + 3;
    assert_eq!(result["total"], total);
    let chunks = result["chunks"].as_array().expect("chunks");
    assert!(
        chunks
            .iter()
            .filter_map(Value::as_u64)
            .all(|chunk| chunk <= 8 * 1024 * 1024)
    );
    // Every payload chunk is drained, and one more drain flushes the end-of-file.
    assert_eq!(chunks.len(), 4);
    assert_eq!(result["eof"], 1);
    assert_eq!(
        result["commands"][0],
        json!(["mkdir", "-p", "--", "/workspace/nested"])
    );
    assert_eq!(
        result["commands"][1],
        json!([
            "sh",
            "-c",
            "cat > \"$1\"",
            "sh",
            "/workspace/nested/file.bin"
        ])
    );
}

/// `test_modal_pty_start_marks_typed_not_found_non_retryable` and
/// `test_modal_pty_start_marks_typed_internal_failure_retryable` (the classification half): the
/// helper maps Modal exception types to a retryable flag.
#[test]
fn helper_classifies_modal_errors_for_retry() {
    let result = run_driver("retryability");

    for permanent in [
        "NotFoundError",
        "InvalidError",
        "AuthError",
        "PermissionDeniedError",
        "SandboxFilesystemNotFoundError",
    ] {
        assert_eq!(result[permanent], json!(false), "{permanent}");
    }
    // An HTTP status that is not a server failure says nothing either way.
    assert_eq!(result["status_404"], Value::Null);
    for transient in [
        "ConnectionError",
        "InternalError",
        "ServiceError",
        "status_503",
    ] {
        assert_eq!(result[transient], json!(true), "{transient}");
    }
    assert_eq!(result["unknown"], Value::Null);
    assert_eq!(result["wrapped_not_found"], json!(false));
}
