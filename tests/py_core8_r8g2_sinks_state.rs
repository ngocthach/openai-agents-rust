//! Cases the earlier audits filed as Python-only that have an observable Rust counterpart:
//!
//! * `test_session_manager.py`: a raising sink's failure reaches the caller of the delivery.
//! * `test_session_sinks.py`: an `HttpProxySink` keeps the headers it was built with.
//! * `test_workspace_jsonl_sink.py`: a legacy backend's read failure keeps the file and the
//!   pending event and never repeats the failure's text.
//! * `test_session_state_roundtrip.py`: a malformed persisted state is rejected without quoting
//!   the payload (the Python tests also walk traceback frames, which Rust does not have).
//! * `test_tar_utils.py`: `strip_tar_member_prefix` keeps a large payload and relativizes the
//!   absolute link beside it.
#![cfg(feature = "sandbox")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::sandbox::SandboxOp;
use openai_agents_sdk::sandbox::archive::{
    TarValidationOptions, extract_tar_to, read_tar_members, strip_tar_member_prefix,
    validate_tar_bytes,
};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::events::{
    CallbackSink, EventSink, HttpProxySink, Instrumentation, SandboxSessionEvent,
    WorkspaceJsonlSink,
};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;

// ----------------------------------------------------------------------------------------------
// test_session_manager.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_async_sink_raise_propagates_to_emit`. A sink that raises on error fails the
/// delivery; Rust's `Instrumentation::emit` is the logging form, so the raising form is
/// `deliver`. The failure carries a fixed message instead of the sink's own text.
#[tokio::test]
async fn a_raising_sink_failure_reaches_the_caller_of_the_delivery() {
    let sink =
        CallbackSink::raising(|_| Err(SandboxError::new(ErrorCode::ExecTransportError, "boom")));
    let instrumentation = Instrumentation::new(Arc::new(sink), "session", "unix_local");
    let event =
        SandboxSessionEvent::start("session", "unix_local", 1, "span_running", SandboxOp::Exec);

    let error = instrumentation
        .deliver(event)
        .await
        .expect_err("the sink raises");

    assert_eq!(error.code, ErrorCode::EventSinkFailed);
    assert!(!error.message.contains("boom"), "{}", error.message);
}

// ----------------------------------------------------------------------------------------------
// test_session_sinks.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_http_proxy_sink_snapshots_headers`. Python mutates the caller's dict after
/// construction; the Rust builder takes its own copy, so the same mutation of the caller's string
/// must not reach the request.
#[tokio::test]
async fn an_http_proxy_sink_sends_the_headers_it_was_built_with() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut chunk = [0u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream.read(&mut chunk).unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .unwrap();
        String::from_utf8_lossy(&request).to_lowercase()
    });

    let mut authorization = String::from("Bearer original");
    let sink = HttpProxySink::new(format!("http://127.0.0.1:{port}/events"))
        .with_header("authorization", authorization.clone());
    authorization.replace_range(.., "Bearer changed");
    let event = SandboxSessionEvent::start("session", "unix_local", 1, "span", SandboxOp::Write);

    sink.emit(&event).await;

    let request = server.join().unwrap();
    assert!(
        request.contains("authorization: bearer original"),
        "{request}"
    );
    assert!(!request.contains("bearer changed"), "{request}");
}

// ----------------------------------------------------------------------------------------------
// test_workspace_jsonl_sink.py
// ----------------------------------------------------------------------------------------------

/// A backend with only a whole-file read whose read fails, as `_LegacyReadSession` with a
/// synthetic failing stream.
#[derive(Debug)]
struct LegacyFsSession {
    inner: Arc<dyn SandboxSession>,
    read_error: Mutex<Option<SandboxError>>,
    writes: Mutex<usize>,
}

#[async_trait]
impl SandboxSession for LegacyFsSession {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn manifest(&self) -> &Manifest {
        self.inner.manifest()
    }
    fn backend_id(&self) -> &str {
        "legacy-fs-session"
    }
    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        self.inner.exec(request).await
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        if let Some(error) = self.read_error.lock().unwrap().clone() {
            return Err(error);
        }
        self.inner.read_file(path).await
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        *self.writes.lock().unwrap() += 1;
        self.inner.write_file(path, contents).await
    }
    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        self.inner.mkdir(path, parents).await
    }
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.hydrate_workspace(archive).await
    }
}

/// Python: `test_workspace_jsonl_sink_legacy_backend_read_failure_closes_stream`. The stream
/// close has no Rust counterpart (reads return bytes); the rest does: the failure is reported,
/// the file keeps its bytes, the event stays pending, and the provider's text is not repeated.
#[tokio::test]
async fn a_legacy_backend_read_failure_keeps_the_file_and_hides_the_text() {
    let client = UnixLocalSandboxClient::new();
    let inner = client.create(None).await.unwrap();
    inner.start().await.unwrap();
    let session = Arc::new(LegacyFsSession {
        inner: inner.clone(),
        read_error: Mutex::new(None),
        writes: Mutex::new(0),
    });
    session
        .write_file("out.jsonl", b"original\n")
        .await
        .unwrap();
    let writes_before = *session.writes.lock().unwrap();
    *session.read_error.lock().unwrap() = Some(SandboxError::new(
        ErrorCode::ExecTransportError,
        "synthetic-private-payload",
    ));
    let sink = WorkspaceJsonlSink::new(session.clone(), "out.jsonl")
        .with_max_bytes(Some(8 * 1024 * 1024))
        .unwrap();
    let event = SandboxSessionEvent::start(
        session.id(),
        "legacy-fs-session",
        1,
        "test-span",
        SandboxOp::Write,
    );

    let error = sink.try_emit(&event).await.expect_err("the read fails");

    assert!(!format!("{error:?}").contains("synthetic-private-payload"));
    assert_eq!(
        *session.writes.lock().unwrap(),
        writes_before,
        "no replacement write"
    );
    assert_eq!(
        std::fs::read(std::path::Path::new(inner.workspace_root()).join("out.jsonl")).unwrap(),
        b"original\n"
    );
    assert!(sink.pending_bytes().await > 0);
    client.delete(inner).await.unwrap();
}

// ----------------------------------------------------------------------------------------------
// test_session_state_roundtrip.py
// ----------------------------------------------------------------------------------------------

fn valid_state_json() -> serde_json::Value {
    let client_state =
        SandboxSessionState::new("unix_local", "session-1", "/workspace", &Manifest::new());
    serde_json::from_str(&client_state.to_json_string().unwrap()).unwrap()
}

/// Python: `test_parse_redacts_malformed_payload_errors`,
/// `test_direct_model_validation_redacts_malformed_mount_authority`,
/// `test_direct_model_validation_redacts_non_mapping_manifest` and
/// `test_model_validate_json_redacts_malformed_json`: a rejected payload is described by kind
/// only. The traceback-frame half of the Python tests has no Rust counterpart.
#[test]
fn a_malformed_state_is_rejected_without_quoting_the_payload() {
    const SECRET: &str = "session-state-parse-secret";
    let mut payloads: Vec<(String, &str)> = Vec::new();
    for (field, value) in [
        ("exposed_ports", serde_json::json!([SECRET])),
        ("session_id", serde_json::json!([SECRET])),
        ("snapshot", serde_json::json!(SECRET)),
        ("snapshot", serde_json::json!({"type": SECRET})),
        ("workspace_root_owned", serde_json::json!(SECRET)),
        ("provider_id", serde_json::json!({SECRET: SECRET})),
        ("$schemaVersion", serde_json::json!(SECRET)),
    ] {
        let mut payload = valid_state_json();
        payload[field] = value;
        payloads.push((payload.to_string(), "payload"));
    }
    // A valid shape that is not an object at all, and values that are not JSON.
    payloads.push((format!("\"{SECRET}\""), "payload"));
    payloads.push((format!("{{\"backend\":\"{SECRET}\""), "JSON"));
    payloads.push((format!("{{\"{SECRET}\": }}"), "JSON"));
    payloads.push((format!("{SECRET} is not json"), "JSON"));

    for (payload, kind) in payloads {
        let error = SandboxSessionState::from_json_str(&payload)
            .expect_err(&format!("expected a rejection for {payload}"));
        assert_eq!(error.code, ErrorCode::SessionStateInvalid, "{payload}");
        assert_eq!(
            error.message,
            format!("sandbox session state {kind} is invalid"),
            "{payload}"
        );
        assert!(!format!("{error:?}").contains(SECRET), "{payload}");
    }
}

/// Python: `test_generic_session_state_parser_sanitizes_legacy_mount_authority`,
/// `test_raw_state_sanitization_clears_pattern_authority`,
/// `test_raw_state_sanitization_strips_opaque_fields_with_known_strategy_type` and
/// `test_raw_state_sanitization_strips_opaque_nested_pattern_fields`. A payload that still embeds
/// a manifest (as older builds wrote) is parsed without it: the state has no manifest field, so
/// no embedded credential, pattern option or opaque field survives a parse and a re-serialize.
#[test]
fn an_embedded_manifest_never_survives_a_parse() {
    let secrets = [
        "legacy-session-state-secret",
        "credential-file-secret",
        "pattern-secret",
        "raw-strategy-secret",
        "nested-pattern-secret",
        "nested-options-secret",
    ];
    let mut payload = valid_state_json();
    payload["manifest"] = serde_json::json!({
        "entries": {
            "credentials.conf": {"type": "file", "content": "credential-file-secret"},
            "data": {
                "type": "s3_mount",
                "bucket": "bucket",
                "access_key_id": "access-key",
                "secret_access_key": "legacy-session-state-secret",
                "mount_strategy": {
                    "type": "docker_volume",
                    "driver": "rclone",
                    "api_token": "raw-strategy-secret",
                },
            },
            "s3files": {
                "type": "s3_files_mount",
                "mount_strategy": {
                    "type": "in_container",
                    "pattern": {
                        "type": "s3files",
                        "api_token": "nested-pattern-secret",
                        "options": {
                            "authorization": "nested-options-secret",
                            "extra_options": {"secret_access_key": "pattern-secret"},
                        },
                    },
                },
            },
        },
    });

    let state = SandboxSessionState::from_json_str(&payload.to_string()).expect("parse");

    let rendered = format!("{state:?}{}", state.to_json_string().unwrap());
    for secret in secrets {
        assert!(!rendered.contains(secret), "{secret} survived: {rendered}");
    }
}

// ----------------------------------------------------------------------------------------------
// test_tar_utils.py
// ----------------------------------------------------------------------------------------------

fn tar_member(name: &str, type_flag: u8, link_name: &str, contents: &[u8], mode: u32) -> Vec<u8> {
    fn put(header: &mut [u8; 512], offset: usize, bytes: &[u8]) {
        header[offset..offset + bytes.len()].copy_from_slice(bytes);
    }
    let mut header = [0u8; 512];
    put(&mut header, 0, name.as_bytes());
    put(&mut header, 100, format!("{mode:07o}\0").as_bytes());
    put(&mut header, 108, b"0000000\0");
    put(&mut header, 116, b"0000000\0");
    put(
        &mut header,
        124,
        format!("{:011o}\0", contents.len()).as_bytes(),
    );
    put(&mut header, 136, b"00000000000\0");
    put(&mut header, 148, b"        ");
    header[156] = type_flag;
    put(&mut header, 157, link_name.as_bytes());
    put(&mut header, 257, b"ustar\0");
    put(&mut header, 263, b"00");
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    put(&mut header, 148, format!("{checksum:06o}\0 ").as_bytes());
    let mut out = header.to_vec();
    out.extend_from_slice(contents);
    out.extend(std::iter::repeat_n(0u8, (512 - contents.len() % 512) % 512));
    out
}

/// Python: `test_strip_tar_member_prefix_retains_only_one_archive_payload` (2 parameter sets).
/// The storage-budget half monkeypatches `tempfile.TemporaryFile`; Rust works on the finished
/// bytes. What a caller can observe is the stripped archive: a payload of over a megabyte arrives
/// intact, passes strict hydrate validation, and the absolute link beside it is rebased.
#[test]
fn a_large_payload_survives_stripping_and_passes_strict_validation() {
    let payload = b"workspace content\n".repeat(65536);
    for absolute_link in [false, true] {
        let mut members = vec![tar_member("workspace", b'5', "", b"", 0o755)];
        if absolute_link {
            members.push(tar_member(
                "workspace/link",
                b'2',
                "/workspace/data.txt",
                b"",
                0o777,
            ));
        }
        members.push(tar_member("workspace/data.txt", b'0', "", &payload, 0o644));
        let mut raw: Vec<u8> = members.into_iter().flatten().collect();
        raw.extend(std::iter::repeat_n(0u8, 1024));

        let normalized = strip_tar_member_prefix(&raw, "workspace", Some("/workspace")).unwrap();

        validate_tar_bytes(&normalized, &TarValidationOptions::hydrate()).unwrap();
        let root = tempfile::tempdir().unwrap();
        extract_tar_to(root.path(), &normalized, &TarValidationOptions::hydrate()).unwrap();
        assert_eq!(
            std::fs::read(root.path().join("data.txt")).unwrap(),
            payload
        );
        if absolute_link {
            let link = read_tar_members(&normalized)
                .unwrap()
                .into_iter()
                .find(|member| member.name == "link")
                .expect("the link member");
            assert_eq!(link.link_target.as_deref(), Some("data.txt"));
        }
    }
}
