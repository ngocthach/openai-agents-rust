//! Ports of the last two `tests/sandbox/test_bounded_read.py` rows that had no Rust test: the
//! instrumented session forwards the byte limit, and the Unix-local backend answers a bounded
//! read from the file itself.
#![cfg(all(feature = "sandbox", unix))]

use std::sync::Arc;

use openai_agents_sdk::sandbox::ErrorCode;
use openai_agents_sdk::sandbox::events::{CallbackSink, InstrumentedSession};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};

async fn session_with(
    root: &std::path::Path,
    name: &str,
    content: &[u8],
) -> Arc<dyn SandboxSession> {
    std::fs::write(root.join(name), content).unwrap();
    let manifest = Manifest::new().with_root(root.to_string_lossy());
    let session = UnixLocalSandboxClient::new()
        .create(Some(manifest))
        .await
        .expect("create");
    session.start().await.expect("start");
    session
}

/// Python: `test_read_bounded_wrapper_forwards_limit`.
#[tokio::test]
async fn instrumented_session_forwards_the_byte_limit() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let inner = session_with(&root, "out.jsonl", b"abcdefgh").await;
    let wrapper = InstrumentedSession::new(inner, Arc::new(CallbackSink::new(|_| {})));

    assert_eq!(wrapper.read_bounded("out.jsonl", 3).await.unwrap(), b"abc");
    assert_eq!(
        wrapper.read_bounded("out.jsonl", 100).await.unwrap(),
        b"abcdefgh"
    );
}

/// Python: `test_native_bounded_read_needs_no_process`. The backend reads the prefix of binary
/// content itself, a missing file is `WorkspaceReadNotFound`, and a zero limit is refused.
#[tokio::test]
async fn native_bounded_read_returns_the_prefix_of_binary_content() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let session = session_with(&root, "out.jsonl", b"\x00\xffbinary data").await;

    assert_eq!(
        session.read_bounded("out.jsonl", 4).await.unwrap(),
        b"\x00\xffbi"
    );
    let missing = session.read_bounded("missing", 4).await.unwrap_err();
    assert_eq!(missing.code, ErrorCode::WorkspaceReadNotFound);
    let zero = session.read_bounded("out.jsonl", 0).await.unwrap_err();
    assert_eq!(zero.code, ErrorCode::SandboxConfigInvalid);
}

/// A file far larger than the limit answers with exactly the prefix.
#[tokio::test]
async fn native_bounded_read_of_a_large_file_returns_exactly_the_prefix() {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    let mut content = vec![b'x'; 4 * 1024 * 1024];
    content[..5].copy_from_slice(b"start");
    let session = session_with(&root, "big.log", &content).await;

    let prefix = session.read_bounded("big.log", 5).await.unwrap();

    assert_eq!(prefix, b"start");
}
