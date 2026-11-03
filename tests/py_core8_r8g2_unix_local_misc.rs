//! Unix-local cases the earlier audits filed as Python-only:
//!
//! * `test_unix_local_special_files.py::test_fifo_replacement_is_rejected_before_io`: a file
//!   replaced by a FIFO between validation and open is refused without blocking or writing.
//! * `test_unix_local.py::test_pty_exit_waits_for_output_close_before_terminal_cleanup`: output
//!   a process wrote just before exiting is delivered before the session is reported finished.
#![cfg(all(unix, feature = "sandbox"))]

use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyWriteRequest};
use openai_agents_sdk::sandbox::sandboxes::unix_local::{
    UnixLocalSandboxClient, UnixLocalSandboxSession,
};
use openai_agents_sdk::sandbox::{ExecShell, Manifest, SandboxClient, SandboxSession};

/// `O_NONBLOCK` for the platforms these tests run on.
#[cfg(target_os = "linux")]
const O_NONBLOCK: i32 = 0o4000;
#[cfg(not(target_os = "linux"))]
const O_NONBLOCK: i32 = 0x0004;

fn mkfifo(path: &Path) {
    let status = std::process::Command::new("mkfifo")
        .arg(path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo failed");
}

/// Python: `test_fifo_replacement_is_rejected_before_io[read]` and `[write]`.
///
/// The hook runs after the path is validated and before it is opened, and swaps the regular file
/// for a FIFO. A peer holds the FIFO open for reading and writing so that even a writable open
/// succeeds; the operation must still be refused, no byte may reach the peer, and the original
/// file keeps its content.
#[tokio::test]
async fn a_file_replaced_by_a_fifo_is_refused_before_any_io() {
    for operation in ["read", "write"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = std::fs::canonicalize(dir.path()).expect("canonical root");
        let target = workspace.join("pipe");
        std::fs::write(&target, b"original").expect("write the original");

        let peer: Arc<std::sync::Mutex<Option<std::fs::File>>> = Arc::default();
        let swapped = Arc::new(AtomicBool::new(false));
        let (hook_peer, hook_swapped) = (peer.clone(), swapped.clone());
        let hook_workspace = workspace.clone();
        let session = UnixLocalSandboxSession::new(
            Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
            false,
        )
        .expect("a session")
        .with_after_validate_hook(Arc::new(move |host: &Path| {
            if host.ends_with("pipe") && !hook_swapped.swap(true, Ordering::SeqCst) {
                let pipe = hook_workspace.join("pipe");
                std::fs::rename(&pipe, hook_workspace.join("original")).expect("move aside");
                mkfifo(&pipe);
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .custom_flags(O_NONBLOCK)
                    .open(&pipe)
                    .expect("open the peer");
                *hook_peer.lock().unwrap() = Some(file);
            }
        }));

        let attempt = async {
            if operation == "read" {
                session.read_file("pipe").await.map(|_| ())
            } else {
                session.write_file("pipe", b"payload").await
            }
        };
        let error = tokio::time::timeout(Duration::from_secs(10), attempt)
            .await
            .unwrap_or_else(|_| panic!("{operation}: the open blocked on the FIFO"))
            .expect_err("a FIFO is never a file");

        assert!(
            matches!(
                error.code,
                ErrorCode::WorkspaceArchiveReadError | ErrorCode::WorkspaceArchiveWriteError
            ),
            "{operation}: {error:?}"
        );
        assert!(swapped.load(Ordering::SeqCst));
        assert_eq!(
            std::fs::read(workspace.join("original")).unwrap(),
            b"original",
            "{operation}"
        );
        assert!(
            std::fs::symlink_metadata(&target)
                .unwrap()
                .file_type()
                .is_fifo_like(),
            "{operation}: the FIFO is still there"
        );
        let mut peer_file = peer.lock().unwrap().take().expect("the peer is open");
        let mut byte = [0u8; 1];
        let received = std::io::Read::read(&mut peer_file, &mut byte);
        assert!(
            matches!(received, Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock),
            "{operation}: the rejected FIFO received a payload: {received:?}"
        );
    }
}

trait FifoLike {
    fn is_fifo_like(&self) -> bool;
}

impl FifoLike for std::fs::FileType {
    fn is_fifo_like(&self) -> bool {
        std::os::unix::fs::FileTypeExt::is_fifo(self)
    }
}

/// Python: `test_pty_exit_waits_for_output_close_before_terminal_cleanup`. The Python test drives
/// private collector helpers with a hand-built entry; what a caller sees is that everything the
/// process printed before it exited arrives, ahead of the update that reports the exit.
#[tokio::test]
async fn output_written_just_before_exit_is_delivered_before_the_exit() {
    const PAYLOAD: usize = 200_000;
    let client = UnixLocalSandboxClient::new();
    let session = client.create(None).await.expect("create");
    session.start().await.expect("start");

    let started = session
        .pty_exec_start(
            PtyExecRequest::new(format!(
                "head -c {PAYLOAD} /dev/zero | tr '\\000' x; exit 3"
            ))
            .with_shell(ExecShell::Plain)
            .with_tty(true)
            .with_yield_time_ms(50),
        )
        .await
        .expect("start a pty session");
    let mut output = started.output.clone();
    let mut update = started;
    for _ in 0..40 {
        let Some(process_id) = update.process_id else {
            break;
        };
        update = session
            .pty_write_stdin(PtyWriteRequest::new(process_id, "").with_yield_time_ms(500))
            .await
            .expect("poll the session");
        output.extend_from_slice(&update.output);
    }

    assert_eq!(update.process_id, None, "the process finished");
    assert_eq!(update.exit_code, Some(3));
    assert_eq!(
        output.iter().filter(|byte| **byte == b'x').count(),
        PAYLOAD,
        "every byte written before the exit arrived"
    );
    client.delete(session).await.expect("delete");
}
