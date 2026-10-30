//! Tests for the Modal, Cloudflare and Blaxel sandbox providers.
//!
//! The HTTP providers are driven against a mock worker/control plane served from a local
//! `TcpListener`, which records every request so the tests can assert on paths, auth headers and
//! bodies. Modal is driven against `tests/fixtures/fake_modal_helper.py`, which speaks the same
//! JSON protocol as the bundled helper and logs each request. No test touches the network or a
//! real credential.

#![cfg(feature = "sandbox")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelCloudBucketMountConfig, BlaxelCloudBucketMountStrategy, BlaxelDriveMount,
    BlaxelDriveMountConfig, BlaxelDriveMountStrategy, BlaxelSandboxClient,
    BlaxelSandboxClientOptions, BlaxelTimeouts, DEFAULT_BLAXEL_WORKSPACE_ROOT, encode_sandbox_path,
};
use openai_agents_sdk::extensions::sandbox::cloudflare::{
    CloudflareBucketMountConfig, CloudflareBucketMountStrategy, CloudflareSandboxClient,
    CloudflareSandboxClientOptions,
};
use openai_agents_sdk::extensions::sandbox::modal::{
    ModalCloudBucketMountConfig, ModalCloudBucketMountStrategy, ModalImageSelector,
    ModalSandboxClient, ModalSandboxClientOptions, ModalSandboxSelector, ModalSandboxSession,
    ModalWorkspacePersistence, decode_snapshot_directory_ref, decode_snapshot_filesystem_ref,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::{ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxSession};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// ---------------------------------------------------------------------------------------------
// Workspace archive fixtures
// ---------------------------------------------------------------------------------------------

/// A well-formed single-file `tar` archive.
///
/// Workspace archives are validated member by member before a provider unpacks them, so a probe
/// archive has to be a real tar rather than an opaque marker string. The marker lives in the
/// member payload, which is where the assertions below look for it.
fn probe_tar(marker: &str) -> Vec<u8> {
    fn put(header: &mut [u8; 512], offset: usize, bytes: &[u8]) {
        header[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    let mut header = [0u8; 512];
    put(&mut header, 0, b"marker.txt");
    put(&mut header, 100, b"0000644\0");
    put(&mut header, 108, b"0000000\0");
    put(&mut header, 116, b"0000000\0");
    put(
        &mut header,
        124,
        format!("{:011o}\0", marker.len()).as_bytes(),
    );
    put(&mut header, 136, b"00000000000\0");
    put(&mut header, 148, b"        ");
    put(&mut header, 156, b"0");
    put(&mut header, 257, b"ustar\0");
    put(&mut header, 263, b"00");
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    put(&mut header, 148, format!("{checksum:06o}\0 ").as_bytes());

    let mut archive = header.to_vec();
    archive.extend_from_slice(marker.as_bytes());
    archive.extend(std::iter::repeat_n(0u8, (512 - marker.len() % 512) % 512));
    archive.extend(std::iter::repeat_n(0u8, 1024));
    archive
}

// ---------------------------------------------------------------------------------------------
// A tiny recording HTTP server
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RecordedRequest {
    method: String,
    path: String,
    query: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl RecordedRequest {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

#[derive(Debug, Clone)]
struct MockResponse {
    status: u16,
    content_type: &'static str,
    /// Written in order, so a test can split an SSE event across socket writes.
    chunks: Vec<Vec<u8>>,
}

impl MockResponse {
    fn json(value: Value) -> Self {
        MockResponse {
            status: 200,
            content_type: "application/json",
            chunks: vec![value.to_string().into_bytes()],
        }
    }

    fn bytes(body: impl Into<Vec<u8>>) -> Self {
        MockResponse {
            status: 200,
            content_type: "application/octet-stream",
            chunks: vec![body.into()],
        }
    }

    fn sse<I, S>(chunks: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        MockResponse {
            status: 200,
            content_type: "text/event-stream",
            chunks: chunks
                .into_iter()
                .map(|chunk| chunk.as_ref().as_bytes().to_vec())
                .collect(),
        }
    }

    fn status(status: u16, body: Value) -> Self {
        MockResponse {
            status,
            content_type: "application/json",
            chunks: vec![body.to_string().into_bytes()],
        }
    }

    fn empty() -> Self {
        MockResponse {
            status: 200,
            content_type: "application/json",
            chunks: vec![b"{}".to_vec()],
        }
    }
}

type Handler = Arc<dyn Fn(&RecordedRequest) -> MockResponse + Send + Sync>;

struct MockServer {
    base_url: String,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
}

impl MockServer {
    async fn start(handler: Handler) -> MockServer {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests: Arc<Mutex<Vec<RecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let handler = handler.clone();
                let recorded = recorded.clone();
                tokio::spawn(async move {
                    let Some(request) = read_request(&mut socket).await else {
                        return;
                    };
                    recorded.lock().unwrap().push(request.clone());
                    let response = handler(&request);
                    let length: usize = response.chunks.iter().map(Vec::len).sum();
                    let head = format!(
                        "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\
                         Connection: close\r\n\r\n",
                        response.status, response.content_type, length
                    );
                    if socket.write_all(head.as_bytes()).await.is_err() {
                        return;
                    }
                    for chunk in response.chunks {
                        if socket.write_all(&chunk).await.is_err() {
                            return;
                        }
                        socket.flush().await.ok();
                        // Give the client a chance to decode this piece on its own, so a CR at
                        // the end of a chunk really does arrive before the LF that follows it.
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    socket.shutdown().await.ok();
                });
            }
        });
        MockServer {
            base_url: format!("http://{address}"),
            requests,
        }
    }

    fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn find(&self, method: &str, path_contains: &str) -> RecordedRequest {
        self.requests()
            .into_iter()
            .find(|request| request.method == method && request.path.contains(path_contains))
            .unwrap_or_else(|| {
                panic!(
                    "no `{method} …{path_contains}…` request; saw {:?}",
                    self.requests()
                        .iter()
                        .map(|r| format!("{} {}", r.method, r.path))
                        .collect::<Vec<_>>()
                )
            })
    }

    fn count(&self, method: &str, path_contains: &str) -> usize {
        self.requests()
            .into_iter()
            .filter(|request| request.method == method && request.path.contains(path_contains))
            .count()
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Option<RecordedRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = buffer
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
        {
            break index;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let mut lines = head.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), query.to_string()),
        None => (target, String::new()),
    };
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let length: usize = headers
        .get("content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let mut body = buffer[header_end..].to_vec();
    while body.len() < length {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(length);
    Some(RecordedRequest {
        method,
        path,
        query,
        headers,
        body,
    })
}

/// The SSE stream a Cloudflare worker returns for one command.
fn exec_sse(stdout: &str, exit_code: i32) -> MockResponse {
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(stdout);
    MockResponse::sse([format!(
        "event: stdout\ndata: {encoded}\n\nevent: exit\ndata: {{\"exit_code\": {exit_code}}}\n\n"
    )])
}

// ---------------------------------------------------------------------------------------------
// Cloudflare
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn cloudflare_runs_a_full_session_against_the_worker_protocol() {
    let handler: Handler =
        Arc::new(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v1/sandbox") => MockResponse::json(json!({"id": "sbx-77"})),
                ("POST", "/v1/sandbox/sbx-77/exec") => exec_sse("ok\n", 0),
                ("PUT", path) if path.starts_with("/v1/sandbox/sbx-77/file/") => {
                    MockResponse::empty()
                }
                ("GET", path) if path.starts_with("/v1/sandbox/sbx-77/file/") => {
                    MockResponse::bytes("file contents")
                }
                ("GET", "/v1/sandbox/sbx-77/running") => {
                    MockResponse::json(json!({"running": true}))
                }
                ("POST", "/v1/sandbox/sbx-77/persist") => MockResponse::bytes("tar-bytes"),
                ("POST", "/v1/sandbox/sbx-77/hydrate") => MockResponse::empty(),
                ("DELETE", "/v1/sandbox/sbx-77") => MockResponse::empty(),
                _ => MockResponse::status(404, json!({"error": "unexpected"})),
            },
        );
    let server = MockServer::start(handler).await;

    let client = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url)
            .with_api_key("cf-token")
            .with_exposed_ports([8080]),
    );
    let manifest = Manifest::new()
        .with_env("FOO", "bar")
        .with_entry("task.md", Entry::text_file("fix it"))
        .with_entry("cache", Entry::dir().with_ephemeral(true));
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();

    // Creation asks the worker for the sandbox id, with the bearer token attached.
    let create = server.find("POST", "/v1/sandbox");
    assert_eq!(
        create.headers.get("authorization").map(String::as_str),
        Some("Bearer cf-token")
    );

    // `start` creates the workspace root first, because the worker's image ships none. The
    // manifest environment becomes an `env` prefix, since the worker's exec body is argv only.
    let first_exec = server.find("POST", "/exec");
    assert_eq!(
        first_exec.json()["argv"],
        json!(["env", "FOO=bar", "mkdir", "-p", "--", "/workspace"])
    );
    assert_eq!(first_exec.json()["timeout_ms"], 30000);

    // The manifest's file entry is materialized through the worker's file endpoint.
    let write = server.find("PUT", "/file/workspace/task.md");
    assert_eq!(write.text(), "fix it");
    assert_eq!(
        write.headers.get("content-type").map(String::as_str),
        Some("application/octet-stream")
    );

    let result = session
        .exec(
            ExecRequest::shell("echo ok")
                .with_timeout(Some(Duration::from_secs(7)))
                .with_env("FOO", "request"),
        )
        .await
        .unwrap();
    assert_eq!(result.stdout_text(), "ok\n");
    assert!(result.ok());
    let timed = server
        .requests()
        .into_iter()
        .rfind(|r| r.path.ends_with("/exec"))
        .unwrap();
    assert_eq!(timed.json()["timeout_ms"], 7000);
    // A per-command variable wins over the manifest value of the same name.
    assert_eq!(
        timed.json()["argv"][1],
        "FOO=request",
        "{}",
        timed.json()["argv"]
    );

    assert_eq!(
        session.read_file("task.md").await.unwrap(),
        b"file contents"
    );
    session.write_file("out/result.txt", b"done").await.unwrap();
    assert_eq!(
        server.find("PUT", "/file/workspace/out/result.txt").text(),
        "done"
    );

    assert!(session.running().await.unwrap());

    let archive = session.persist_workspace().await.unwrap();
    assert_eq!(archive, b"tar-bytes");
    // Ephemeral entries and mounts are runtime state, so they are excluded from the archive.
    assert_eq!(
        server.find("POST", "/persist").query,
        "excludes=.sandbox-blobfuse-cache%2C.sandbox-blobfuse-config%2C.sandbox-mountpoint-env%2C\
         .sandbox-rclone-config%2Ccache"
    );

    session
        .hydrate_workspace(&probe_tar("restored"))
        .await
        .unwrap();
    assert!(server.find("POST", "/hydrate").text().contains("restored"));

    session.shutdown().await.unwrap();
    assert_eq!(server.count("DELETE", "/v1/sandbox/sbx-77"), 1);
    // A second shutdown is a no-op rather than a second delete.
    session.shutdown().await.unwrap();
    assert_eq!(server.count("DELETE", "/v1/sandbox/sbx-77"), 1);
}

#[tokio::test]
async fn cloudflare_exec_decodes_an_sse_stream_split_across_writes() {
    let handler: Handler = Arc::new(|request| {
        if request.path.ends_with("/exec") {
            // The CR of the first event's terminator lands at the end of one write and its LF
            // opens the next, which is the boundary a naive line splitter gets wrong.
            return MockResponse::sse(vec![
                "event: stdout\r\ndata: aGVsbG8=\r\n\r",
                "\nevent: stderr\r\ndata: b29wcw==\r\n\r\n",
                "event: exit\r\ndata: {\"exit_code\": 3}\r\n\r\n",
            ]);
        }
        MockResponse::json(json!({"id": "sbx-1"}))
    });
    let server = MockServer::start(handler).await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));
    let session = client.create(None).await.unwrap();

    let result = session
        .exec(ExecRequest::shell("echo hello"))
        .await
        .unwrap();
    assert_eq!(result.stdout_text(), "hello");
    assert_eq!(result.stderr_text(), "oops");
    assert_eq!(result.exit_code, 3);
}

#[tokio::test]
async fn cloudflare_maps_a_missing_file_and_a_worker_error() {
    let handler: Handler = Arc::new(|request| match request.method.as_str() {
        "GET" => MockResponse::status(404, json!({"error": "not found"})),
        _ if request.path.ends_with("/exec") => {
            MockResponse::status(503, json!({"error": "overloaded", "code": "E503"}))
        }
        _ => MockResponse::json(json!({"id": "sbx-1"})),
    });
    let server = MockServer::start(handler).await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));
    let session = client.create(None).await.unwrap();

    let error = session.read_file("missing.txt").await.unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);

    let error = session.exec(ExecRequest::shell("ls")).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert!(error.message.contains("HTTP 503"));
    assert!(error.message.contains("E503: overloaded"));
}

#[tokio::test]
async fn cloudflare_bucket_mount_activates_through_the_worker() {
    let handler: Handler = Arc::new(|request| {
        if request.path.ends_with("/exec") {
            return exec_sse("", 0);
        }
        if request.path == "/v1/sandbox" {
            return MockResponse::json(json!({"id": "sbx-9"}));
        }
        MockResponse::empty()
    });
    let server = MockServer::start(handler).await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));
    let manifest = Manifest::new().with_entry(
        "assets",
        Entry::mount(CloudflareBucketMountStrategy::mount(
            CloudflareBucketMountConfig::r2("assets", "acct")
                .with_credentials("AKIA", "secret")
                .with_key_prefix("images"),
        )),
    );
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();

    let mount = server.find("POST", "/mount");
    assert_eq!(
        mount.json(),
        json!({
            "bucket": "assets",
            "mountPath": "/workspace/assets",
            "options": {
                "endpoint": "https://acct.r2.cloudflarestorage.com",
                "readOnly": true,
                "prefix": "/images/",
                "credentials": {"accessKeyId": "AKIA", "secretAccessKey": "secret"},
            },
        })
    );

    session
        .manifest()
        .clone()
        .unmount_all(&*session)
        .await
        .unwrap();
    assert_eq!(
        server.find("POST", "/unmount").json(),
        json!({"mountPath": "/workspace/assets"})
    );
}

#[tokio::test]
async fn cloudflare_and_blaxel_sessions_support_pty() {
    let server = MockServer::start(Arc::new(|_| MockResponse::json(json!({"id": "sbx-1"})))).await;
    let cloudflare = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url).with_api_key("cf-token"),
    )
    .create(None)
    .await
    .unwrap();
    // Both providers drive a terminal over their WebSocket endpoint.
    assert!(cloudflare.supports_pty());

    let blaxel_server = {
        let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let url_slot = shared.clone();
        let handler: Handler = Arc::new(move |request| {
            let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
            blaxel_handler(sandbox_url)(request)
        });
        let server = MockServer::start(handler).await;
        *shared.lock().unwrap() = Some(server.base_url.clone());
        server
    };
    let blaxel = BlaxelSandboxClient::new(blaxel_options(&blaxel_server.base_url))
        .create(None)
        .await
        .unwrap();
    assert!(blaxel.supports_pty());

    // Without an API key the terminal URL cannot be built, so the session says so up front.
    let keyless = BlaxelSandboxClient::new(
        BlaxelSandboxClientOptions::new()
            .with_name("agents-test")
            .with_workspace("acme")
            .with_api_base_url(format!("{}/v0", blaxel_server.base_url)),
    )
    .create(None)
    .await
    .unwrap();
    assert!(!keyless.supports_pty());
}

#[tokio::test]
async fn cloudflare_rejects_a_non_workspace_manifest_root() {
    let server = MockServer::start(Arc::new(|_| MockResponse::json(json!({"id": "x"})))).await;
    let client =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url));
    let error = client
        .create(Some(Manifest::new().with_root("/srv")))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    // The worker is never contacted when the manifest cannot work.
    assert!(server.requests().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Blaxel
// ---------------------------------------------------------------------------------------------

fn blaxel_handler(sandbox_url: String) -> Handler {
    Arc::new(
        move |request| match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v0/sandboxes") => MockResponse::json(json!({
                "metadata": {"name": "agents-test", "url": sandbox_url},
                "status": "DEPLOYED",
            })),
            ("GET", "/v0/sandboxes/agents-test") => MockResponse::json(json!({
                "metadata": {"name": "agents-test", "url": sandbox_url},
            })),
            ("DELETE", "/v0/sandboxes/agents-test") => MockResponse::empty(),
            ("POST", "/v0/sandboxes/agents-test/snapshots") => {
                MockResponse::json(json!({"metadata": {"name": "snap-1"}}))
            }
            ("POST", "/v0/sandboxes/agents-test/snapshots/snap-1/restore") => MockResponse::empty(),
            ("POST", "/process") => MockResponse::json(json!({
                "exitCode": 0,
                "stdout": "ran\n",
                "stderr": "",
                "status": "completed",
            })),
            ("PUT", path) if path.starts_with("/filesystem/") => {
                MockResponse::json(json!({"success": true}))
            }
            ("DELETE", path) if path.starts_with("/filesystem/") => {
                MockResponse::json(json!({"success": true}))
            }
            ("GET", path) if path.starts_with("/filesystem/") => {
                if request.headers.get("accept").map(String::as_str)
                    == Some("application/octet-stream")
                {
                    MockResponse::bytes("blob")
                } else {
                    MockResponse::json(json!({
                        "name": "workspace",
                        "path": "/workspace",
                        "files": [{
                            "name": "task.md",
                            "path": "/workspace/task.md",
                            "size": 6,
                            "permissions": "-rw-r--r--",
                            "owner": "root",
                            "group": "root",
                            "lastModified": "now",
                        }],
                        "subdirectories": [{"name": "repo", "path": "/workspace/repo"}],
                    }))
                }
            }
            ("POST", "/drives/mount") => MockResponse::json(json!({"success": true})),
            ("DELETE", path) if path.starts_with("/drives/mount/") => {
                MockResponse::json(json!({"success": true}))
            }
            _ => MockResponse::status(404, json!({"error": {"code": "ROUTE_NOT_FOUND"}})),
        },
    )
}

fn blaxel_options(base_url: &str) -> BlaxelSandboxClientOptions {
    BlaxelSandboxClientOptions::new()
        .with_name("agents-test")
        .with_api_key("bl-token")
        .with_workspace("acme")
        .with_region("us-pdx-1")
        .with_image("blaxel/base-image:latest")
        .with_memory(2048)
        .with_env_var("BASE", "1")
        .with_api_base_url(format!("{base_url}/v0"))
        .with_timeouts(BlaxelTimeouts {
            exec: Duration::from_secs(5),
            ..BlaxelTimeouts::default()
        })
}

#[tokio::test]
async fn blaxel_runs_a_full_session_against_the_sandbox_api() {
    // The control plane and the sandbox share one mock, with `metadata.url` pointing back at it.
    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        blaxel_handler(sandbox_url)(request)
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());

    let client = BlaxelSandboxClient::new(blaxel_options(&server.base_url));
    let manifest = Manifest::new()
        .with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
        .with_env("FOO", "bar")
        .with_entry("task.md", Entry::text_file("fix it"));
    let session = client.create(Some(manifest)).await.unwrap();

    let create = server.find("POST", "/v0/sandboxes");
    assert_eq!(create.query, "createIfNotExist=true");
    assert_eq!(
        create
            .headers
            .get("x-blaxel-authorization")
            .map(String::as_str),
        Some("Bearer bl-token")
    );
    assert_eq!(
        create.headers.get("x-blaxel-workspace").map(String::as_str),
        Some("acme")
    );
    assert_eq!(
        create.json(),
        json!({
            "metadata": {"name": "agents-test"},
            "spec": {
                "runtime": {
                    "image": "blaxel/base-image:latest",
                    "memory": 2048,
                    "envs": [{"name": "BASE", "value": "1"}],
                },
                "region": "us-pdx-1",
            },
        })
    );

    session.start().await.unwrap();
    // The workspace root is created through the filesystem API, with its leading slash encoded.
    let mkdir = server.find(
        "PUT",
        &format!("/filesystem/{}", encode_sandbox_path("/workspace")),
    );
    assert_eq!(
        mkdir.json(),
        json!({"isDirectory": true, "permissions": "0755"})
    );

    // The file entry is uploaded as multipart form data, as the Blaxel SDK does.
    let write = server.find("PUT", "task.md");
    assert!(
        write
            .headers
            .get("content-type")
            .unwrap()
            .starts_with("multipart/form-data; boundary=")
    );
    assert!(write.text().contains("filename=\"binary-file.bin\""));
    assert!(write.text().contains("fix it"));

    let result = session
        .exec(ExecRequest::argv(["ls", "-la"]).with_env("EXTRA", "2"))
        .await
        .unwrap();
    assert_eq!(result.stdout_text(), "ran\n");
    // Materializing the manifest also runs commands, so the assertion names this one.
    let process = server
        .requests()
        .into_iter()
        .find(|request| request.path == "/process" && request.json()["command"] == "ls -la")
        .expect("the exec reached the sandbox process API");
    assert_eq!(
        process.json(),
        json!({
            "command": "ls -la",
            "workingDir": "/workspace",
            "waitForCompletion": true,
            "timeout": 5000,
            // The sandbox's creation environment, then the manifest's, then this command's.
            "env": {"BASE": "1", "EXTRA": "2", "FOO": "bar"},
        })
    );

    assert_eq!(session.read_file("task.md").await.unwrap(), b"blob");

    let entries = session.ls("/workspace").await.unwrap();
    assert_eq!(entries.len(), 2);
    assert!(entries[0].is_dir);
    assert_eq!(entries[0].path, "/workspace/repo");
    assert_eq!(entries[1].name, "task.md");
    assert_eq!(entries[1].size, Some(6));

    session.rm("task.md", true).await.unwrap();
    let remove = server.find("DELETE", "task.md");
    assert_eq!(remove.query, "recursive=true");

    session.shutdown().await.unwrap();
    assert_eq!(server.count("DELETE", "/v0/sandboxes/agents-test"), 1);
}

#[tokio::test]
async fn blaxel_persists_and_hydrates_through_a_tar_file() {
    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        blaxel_handler(sandbox_url)(request)
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());

    let client = BlaxelSandboxClient::new(blaxel_options(&server.base_url));
    let manifest = Manifest::new()
        .with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
        .with_entry("cache", Entry::dir().with_ephemeral(true));
    let session = client.create(Some(manifest)).await.unwrap();

    let archive = session.persist_workspace().await.unwrap();
    assert_eq!(archive, b"blob");
    let tar = server.find("POST", "/process");
    assert_eq!(
        tar.json()["command"],
        "sh -lc 'tar --exclude ./.sandbox-blobfuse-cache --exclude ./.sandbox-blobfuse-config \
         --exclude ./.sandbox-mountpoint-env --exclude ./.sandbox-rclone-config --exclude ./cache \
         -C /workspace -cf /tmp/openai-agents-sandbox-blaxel-persist-agents-test.tar .'"
    );
    // The archive is pulled back out of the sandbox and the scratch file removed.
    assert!(
        server.count(
            "DELETE",
            "openai-agents-sandbox-blaxel-persist-agents-test.tar"
        ) >= 1
    );

    session
        .hydrate_workspace(&probe_tar("payload"))
        .await
        .unwrap();
    let upload = server.find(
        "PUT",
        "openai-agents-sandbox-blaxel-hydrate-agents-test.tar",
    );
    assert!(upload.text().contains("payload"));
    let extract = server
        .requests()
        .into_iter()
        .rfind(|request| request.path == "/process")
        .unwrap();
    assert_eq!(
        extract.json()["command"],
        "sh -lc 'tar -C /workspace -xf \
         /tmp/openai-agents-sandbox-blaxel-hydrate-agents-test.tar'"
    );
}

#[tokio::test]
async fn blaxel_snapshots_and_restores_through_the_control_plane() {
    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        blaxel_handler(sandbox_url)(request)
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());

    let client = BlaxelSandboxClient::new(blaxel_options(&server.base_url));
    // Snapshots are Blaxel-only, so the test takes the concrete session.
    let session = client
        .create_blaxel_session(Some(
            Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        ))
        .await
        .unwrap();
    let snapshot = session.create_sandbox_snapshot().await.unwrap();
    assert_eq!(snapshot, "snap-1");
    let created = server.find("POST", "/v0/sandboxes/agents-test/snapshots");
    assert_eq!(
        created
            .headers
            .get("x-blaxel-authorization")
            .map(String::as_str),
        Some("Bearer bl-token")
    );

    session.restore_sandbox_snapshot(&snapshot).await.unwrap();
    assert_eq!(
        server.count("POST", "/v0/sandboxes/agents-test/snapshots/snap-1/restore"),
        1
    );
}

#[tokio::test]
async fn blaxel_mounts_a_drive_and_a_bucket() {
    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        blaxel_handler(sandbox_url)(request)
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());

    let client = BlaxelSandboxClient::new(blaxel_options(&server.base_url));
    let manifest = Manifest::new()
        .with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
        .with_entry(
            "drive",
            Entry::mount(BlaxelDriveMountStrategy::mount(
                BlaxelDriveMountConfig::new("shared").with_drive_path("/data"),
            )),
        )
        .with_entry(
            "bucket",
            Entry::mount(BlaxelCloudBucketMountStrategy::mount(
                BlaxelCloudBucketMountConfig::s3("public-bucket"),
            )),
        );
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();

    assert_eq!(
        server.find("POST", "/drives/mount").json(),
        json!({
            "driveName": "shared",
            "mountPath": "/workspace/drive",
            "drivePath": "/data",
            "readOnly": false,
        })
    );
    // The public bucket needs no credential file, so the mount runs `s3fs` directly.
    let mount_command = server
        .requests()
        .into_iter()
        .filter(|request| request.path == "/process")
        .find(|request| {
            // The probe that checks whether the image already ships `s3fs` mentions it too, so
            // the mount itself is identified by an option only the mount command carries.
            request.json()["command"]
                .as_str()
                .unwrap_or("")
                .contains("public_bucket=1")
        })
        .expect("the bucket mount runs s3fs inside the sandbox");
    let command = mount_command.json()["command"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(command.contains("public_bucket=1"), "{command}");
    assert!(command.contains("/workspace/bucket"), "{command}");
}

#[tokio::test]
async fn blaxel_mounts_a_drive_declared_as_a_mount_entry() {
    // The same drive, described as a `BlaxelDriveMount` entry rather than as a strategy
    // configuration, reaches the drives API with the same body.
    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        blaxel_handler(sandbox_url)(request)
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());

    let entry = BlaxelDriveMount::new("shared")
        .with_drive_path("/data")
        .with_read_only(true);
    let client = BlaxelSandboxClient::new(blaxel_options(&server.base_url));
    let manifest = Manifest::new()
        .with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
        .with_entry("drive", Entry::mount(entry.mount()));
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();

    assert_eq!(
        server.find("POST", "/drives/mount").json(),
        json!({
            "driveName": "shared",
            "mountPath": "/workspace/drive",
            "drivePath": "/data",
            "readOnly": true,
        })
    );

    // The entry builds exactly the strategy the configuration form builds, so the rest of the
    // lifecycle is the one `blaxel_mounts_a_drive_and_a_bucket` already covers.
    assert_eq!(
        BlaxelDriveMountStrategy::for_mount(&entry)
            .unwrap()
            .config(),
        &BlaxelDriveMountConfig::new("shared")
            .with_drive_path("/data")
            .with_read_only(true)
    );
}

#[tokio::test]
async fn blaxel_requires_a_workspace() {
    // SAFETY-equivalent note: the test only reads its own process environment, and the provider
    // reports the missing workspace instead of guessing one.
    let client =
        BlaxelSandboxClient::new(BlaxelSandboxClientOptions::new().with_api_key("bl-token"));
    if std::env::var("BL_WORKSPACE").is_ok() {
        return;
    }
    let error = client.create(None).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
    assert!(error.message.contains("BL_WORKSPACE"));
}

// ---------------------------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------------------------

struct FakeModal {
    _directory: tempfile::TempDir,
    log: std::path::PathBuf,
    state: std::path::PathBuf,
}

impl FakeModal {
    fn new() -> FakeModal {
        let directory = tempfile::tempdir().unwrap();
        FakeModal {
            log: directory.path().join("requests.jsonl"),
            state: directory.path().join("state.json"),
            _directory: directory,
        }
    }

    fn fixture() -> String {
        format!(
            "{}/tests/fixtures/fake_modal_helper.py",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    /// The helper command, with the log and state paths wired in through `env`.
    fn command(&self) -> Vec<String> {
        vec![
            "/usr/bin/env".to_string(),
            format!("FAKE_MODAL_LOG={}", self.log.display()),
            format!("FAKE_MODAL_STATE={}", self.state.display()),
            "FAKE_MODAL_SLEEP=5".to_string(),
            "python3".to_string(),
            FakeModal::fixture(),
        ]
    }

    fn requests(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    fn of_op(&self, op: &str) -> Vec<Value> {
        self.requests()
            .into_iter()
            .filter(|request| request["op"] == op)
            .collect()
    }
}

#[tokio::test]
async fn modal_runs_a_full_session_through_the_helper() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_image_tag("python:3.14-slim")
            .with_memory_mb(2048)
            .with_helper_command(fake.command()),
    );
    let manifest = Manifest::new()
        .with_env("FOO", "bar")
        .with_entry("task.md", Entry::text_file("fix it"))
        .with_entry(
            "assets",
            Entry::mount(ModalCloudBucketMountStrategy::mount(
                ModalCloudBucketMountConfig::s3("bucket")
                    .with_secret_name("aws-creds")
                    .with_key_prefix("prefix/"),
            )),
        );
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();

    let create = &fake.of_op("create")[0];
    assert_eq!(create["app_name"], "agents-app");
    assert_eq!(create["image_tag"], "python:3.14-slim");
    assert_eq!(create["workdir"], "/workspace");
    assert_eq!(create["env"], json!({"FOO": "bar"}));
    assert_eq!(create["memory"], 2048);
    // Modal attaches buckets at creation, so the mount travels in `volumes`.
    assert_eq!(
        create["volumes"],
        json!([{
            "mount_path": "/workspace/assets",
            "bucket_name": "bucket",
            "key_prefix": "prefix/",
            "secret_name": "aws-creds",
            "read_only": true,
        }])
    );

    // `start` creates the workspace root from `/` before materializing the manifest.
    let first_exec = &fake.of_op("exec")[0];
    assert_eq!(first_exec["sandbox_id"], "sb-1");
    assert_eq!(
        first_exec["argv"],
        json!(["mkdir", "-p", "--", "/workspace"])
    );
    assert_eq!(first_exec["workdir"], "/");

    let written = &fake.of_op("write_file")[0];
    assert_eq!(written["path"], "/workspace/task.md");
    assert_eq!(session.read_file("task.md").await.unwrap(), b"fix it");

    let result = session
        .exec(ExecRequest::argv(["echo", "hi"]).with_cwd(Some("repo".into())))
        .await
        .unwrap();
    assert_eq!(result.stdout_text(), "ran: echo hi\n");
    let last_exec = fake.of_op("exec").pop().unwrap();
    assert_eq!(last_exec["workdir"], "/workspace/repo");

    let failed = session.exec(ExecRequest::argv(["false"])).await.unwrap();
    assert_eq!(failed.exit_code, 3);
    assert_eq!(failed.stderr_text(), "boom\n");

    assert!(session.running().await.unwrap());
    session.shutdown().await.unwrap();
    assert_eq!(fake.of_op("terminate").len(), 1);
}

#[tokio::test]
async fn modal_persists_and_restores_a_filesystem_snapshot() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_workspace_persistence(ModalWorkspacePersistence::SnapshotFilesystem)
            .with_helper_command(fake.command()),
    );
    let session = client.create(None).await.unwrap();

    let archive = session.persist_workspace().await.unwrap();
    assert_eq!(
        decode_snapshot_filesystem_ref(&archive).as_deref(),
        Some("im-sb-1")
    );
    assert_eq!(fake.of_op("snapshot_filesystem").len(), 1);

    // Restoring boots a replacement sandbox from the snapshot image and retires the old one.
    session.hydrate_workspace(&archive).await.unwrap();
    let creates = fake.of_op("create");
    assert_eq!(creates.len(), 2);
    assert_eq!(creates[1]["image_id"], "im-sb-1");
    assert_eq!(fake.of_op("terminate")[0]["sandbox_id"], "sb-1");
    // The first create was unbounded; the restore is bounded by the restore timeout rather than
    // by the capture timeout or by `sandbox_create_timeout`.
    assert!(creates[0].get("create_timeout").is_none());
    assert_eq!(creates[1]["create_timeout"], 60.0);
}

#[tokio::test]
async fn modal_bounds_a_snapshot_restore_with_its_own_timeout() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_workspace_persistence(ModalWorkspacePersistence::SnapshotFilesystem)
            .with_sandbox_create_timeout(Duration::from_secs(45))
            .with_snapshot_filesystem_timeout(Duration::from_secs(30))
            .with_snapshot_filesystem_restore_timeout(Duration::from_secs(90))
            .with_helper_command(fake.command()),
    );
    let session = client.create(None).await.unwrap();
    let archive = session.persist_workspace().await.unwrap();
    session.hydrate_workspace(&archive).await.unwrap();

    assert_eq!(fake.of_op("snapshot_filesystem")[0]["timeout"], 30);
    let creates = fake.of_op("create");
    assert_eq!(creates[0]["create_timeout"], 45.0);
    assert_eq!(creates[1]["create_timeout"], 90.0);
}

#[tokio::test]
async fn modal_sends_a_cpu_and_memory_range_to_the_helper() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_cpu_range(2.0, 4.0)
            .with_memory_mb_range(4096, 8192)
            .with_helper_command(fake.command()),
    );
    let _session = client.create(None).await.unwrap();

    let create = fake.of_op("create").remove(0);
    assert_eq!(create["cpu"], json!([2.0, 4.0]));
    assert_eq!(create["memory"], json!([4096, 8192]));
}

#[tokio::test]
async fn modal_bundled_helper_turns_a_resource_pair_into_a_tuple() {
    // Modal's resource conversion tests `isinstance(value, tuple)`, so a JSON array reaching
    // `Sandbox.create` as a list would drop both the request and the limit without an error.
    // The fake helper never calls Modal, so only the bundled helper's own conversion proves it.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("helper.py");
    std::fs::write(
        &path,
        openai_agents_sdk::extensions::sandbox::modal::HELPER_SOURCE,
    )
    .unwrap();
    let script = format!(
        "import pathlib, types; module = types.ModuleType('helper'); \
         exec(compile(pathlib.Path({:?}).read_text(), 'helper.py', 'exec'), module.__dict__); \
         print(repr(module._resource_pair([2.0, 4.0])), repr(module._resource_pair(2.0)), \
         repr(module._resource_pair(None)))",
        path.to_string_lossy()
    );
    let Ok(output) = tokio::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .await
    else {
        // A machine without `python3` cannot run the provider at all.
        return;
    };
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "(2.0, 4.0) 2.0 None"
    );
}

#[tokio::test]
async fn modal_persists_a_tar_archive_by_default() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app").with_helper_command(fake.command()),
    );
    let manifest = Manifest::new().with_entry("cache", Entry::dir().with_ephemeral(true));
    let session = client.create(Some(manifest)).await.unwrap();

    assert_eq!(
        session.persist_workspace().await.unwrap(),
        b"fake-tar-bytes"
    );
    let tar = fake.of_op("exec").pop().unwrap();
    assert_eq!(
        tar["argv"],
        json!([
            "sh",
            "-lc",
            "tar -czf - -C /workspace --exclude ./cache --exclude ./.sandbox-rclone-config \
             --exclude ./.sandbox-blobfuse-config --exclude ./.sandbox-blobfuse-cache \
             --exclude ./.sandbox-mountpoint-env ."
        ])
    );

    let archive = probe_tar("payload");
    session.hydrate_workspace(&archive).await.unwrap();
    // The archive is staged with the helper's chunked `write_file`, never as a command argument,
    // because a real workspace would blow past the kernel's per-argument limit.
    let staged = fake.of_op("write_file").pop().unwrap();
    assert!(
        staged["path"]
            .as_str()
            .unwrap()
            .starts_with("/tmp/openai-agents-sandbox-modal-hydrate-"),
        "{staged}"
    );
    // The whole archive travels as one base64 payload.
    assert_eq!(
        staged["data"].as_str().unwrap().len(),
        4 * archive.len().div_ceil(3)
    );
    let extract = fake
        .of_op("exec")
        .into_iter()
        .find(|request| {
            request["argv"][2]
                .as_str()
                .unwrap_or("")
                .contains("tar -xzf")
        })
        .expect("the archive is extracted inside the sandbox");
    let script = extract["argv"][2].as_str().unwrap();
    assert!(script.contains("-C /workspace"), "{script}");
    assert!(
        script.contains(staged["path"].as_str().unwrap()),
        "{script}"
    );
    // Extraction must not inherit a short default timeout.
    assert!(extract.get("timeout").is_none(), "{extract}");
}

#[tokio::test]
async fn modal_terminates_the_sandbox_when_a_command_times_out() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_exec_timeout(Duration::from_millis(200))
            .with_helper_command(fake.command()),
    );
    let session = client.create(None).await.unwrap();

    let error = session
        .exec(ExecRequest::argv(["sleep", "30"]).with_timeout(Some(Duration::from_millis(200))))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ExecTimeout);
    // Modal cannot cancel a running command, so the port retires the sandbox instead.
    assert_eq!(fake.of_op("terminate").len(), 1);
    assert!(!session.running().await.unwrap());
}

#[tokio::test]
async fn modal_runs_a_command_without_a_timeout_unbounded() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_exec_timeout(Duration::from_millis(50))
            .with_helper_command(fake.command()),
    );
    let session = client.create(None).await.unwrap();

    // A command with no timeout of its own must run to completion rather than inherit one that
    // would retire the whole sandbox, which is what the Python port does.
    let result = session
        .exec(ExecRequest::argv(["echo", "slow"]))
        .await
        .unwrap();
    assert!(result.ok());
    assert!(fake.of_op("terminate").is_empty());
    let request = fake.of_op("exec").pop().unwrap();
    assert!(request.get("timeout").is_none(), "{request}");
}

#[tokio::test]
async fn modal_reports_a_missing_file_and_a_helper_failure() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app").with_helper_command(fake.command()),
    );
    let session = client.create(None).await.unwrap();

    let error = session.read_file("missing.txt").await.unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);

    let broken = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_helper_command(["/nonexistent/python-interpreter"]),
    );
    let error = broken.create(None).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceStartError);
}

#[tokio::test]
async fn modal_starts_a_pty_session_through_the_helper() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app").with_helper_command(fake.command()),
    );
    let session = client.create(None).await.unwrap();
    assert!(session.supports_pty());

    let update = session
        .pty_exec_start(
            openai_agents_sdk::sandbox::PtyExecRequest::new("echo hi")
                .with_tty(true)
                .with_yield_time_ms(400),
        )
        .await
        .unwrap();
    assert!(
        update.output_text().contains("pty: sh -c echo $$ >"),
        "{}",
        update.output_text()
    );
    // The PTY request travels on the command line, because stdin belongs to the terminal.
    let pty = &fake.of_op("pty")[0];
    assert_eq!(pty["tty"], true);
    assert_eq!(pty["workdir"], "/workspace");
    let inner = pty["argv"][2].as_str().unwrap();
    assert!(inner.contains("exec sh -lc 'echo hi'"), "{inner}");

    // Killing the helper would leave the command running inside Modal, so teardown kills the
    // sandbox-side process by the pid the wrapper recorded.
    let kill = fake
        .of_op("exec")
        .into_iter()
        .find(|request| {
            request["argv"].as_array().is_some_and(|argv| {
                argv.iter()
                    .any(|part| part.as_str().unwrap_or("").contains("kill -TERM"))
            })
        })
        .expect("the PTY teardown kills the sandbox-side process");
    assert!(
        kill["argv"][2]
            .as_str()
            .unwrap()
            .contains("/tmp/openai-agents-sandbox-modal-pty-"),
        "{kill}"
    );
}

#[tokio::test]
async fn modal_passes_per_command_environment_to_the_helper() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app").with_helper_command(fake.command()),
    );
    let session = client.create(None).await.unwrap();

    session
        .exec(ExecRequest::argv(["env"]).with_env("TOKEN", "abc"))
        .await
        .unwrap();
    let request = fake.of_op("exec").pop().unwrap();
    assert_eq!(request["env"], json!({"TOKEN": "abc"}));
}

#[tokio::test]
async fn modal_bundled_helper_is_valid_python_and_covers_every_op() {
    // The helper only ever runs inside an interpreter, so a syntax error would otherwise surface
    // at the first real Modal call rather than in the test suite.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("helper.py");
    std::fs::write(
        &path,
        openai_agents_sdk::extensions::sandbox::modal::HELPER_SOURCE,
    )
    .unwrap();
    let compiled = tokio::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import pathlib; compile(pathlib.Path({:?}).read_text(), 'helper.py', 'exec')",
            path.to_string_lossy()
        ))
        .output()
        .await;
    let Ok(compiled) = compiled else {
        // A machine without `python3` cannot run the provider at all, so there is nothing to
        // check here.
        return;
    };
    assert!(
        compiled.status.success(),
        "the bundled Modal helper does not compile: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    for op in [
        "create",
        "exec",
        "read_file",
        "write_file",
        "poll",
        "terminate",
        "snapshot_filesystem",
        "pty",
    ] {
        assert!(
            openai_agents_sdk::extensions::sandbox::modal::HELPER_SOURCE
                .contains(&format!("\"{op}\"")),
            "the bundled Modal helper does not handle the `{op}` op"
        );
    }
}

#[tokio::test]
async fn modal_bundled_helper_parses_a_command_line_request() {
    // The `pty` op passes its request on the command line, and only the bundled helper runs in
    // production, so its own argv parsing is exercised here rather than the fake's.
    let output = tokio::process::Command::new("python3")
        .arg("-c")
        .arg(openai_agents_sdk::extensions::sandbox::modal::HELPER_SOURCE)
        .arg(r#"{"op":"definitely-not-an-op"}"#)
        .stdin(std::process::Stdio::null())
        .output()
        .await;
    let Ok(output) = output else {
        return;
    };
    let reported = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // Either the interpreter has no `modal` package, or the op is rejected by name; both mean
    // the request itself was read from argv rather than from the empty stdin.
    assert!(
        reported.contains("modal") || reported.contains("definitely-not-an-op"),
        "{reported}"
    );
    assert!(!reported.contains("empty request"), "{reported}");
    assert!(!reported.contains("malformed request"), "{reported}");
}

// ---------------------------------------------------------------------------------------------
// Modal selectors, directory snapshots and tunnels; Cloudflare and Blaxel exposed ports
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn modal_adopts_a_sandbox_named_by_a_selector() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_image(ModalImageSelector::from_id("im-existing"))
            .with_sandbox(ModalSandboxSelector::from_id("sb-existing"))
            .with_helper_command(fake.command()),
    );
    let session = client.create(None).await.expect("create");

    // Adopting replaces `Sandbox.create` entirely; the helper only checks the sandbox is live.
    assert!(fake.of_op("create").is_empty());
    assert_eq!(fake.of_op("poll")[0]["sandbox_id"], "sb-existing");
    let modal = session
        .as_any()
        .and_then(|any| any.downcast_ref::<ModalSandboxSession>())
        .expect("a Modal session");
    assert_eq!(modal.sandbox_id(), "sb-existing");
    assert_eq!(modal.image_id().as_deref(), Some("im-existing"));
}

#[tokio::test]
async fn modal_image_selectors_pick_the_id_or_the_tag() {
    let fake = FakeModal::new();
    ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_image_id("im-stale")
            // A later selector replaces the earlier form rather than stacking with it.
            .with_image(ModalImageSelector::from_tag("debian:bookworm"))
            .with_helper_command(fake.command()),
    )
    .create(None)
    .await
    .expect("create");

    let create = &fake.of_op("create")[0];
    assert_eq!(create["image_tag"], "debian:bookworm");
    assert!(create.get("image_id").is_none());
}

#[tokio::test]
async fn modal_round_trips_a_directory_snapshot() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_workspace_persistence(ModalWorkspacePersistence::SnapshotDirectory)
            .with_helper_command(fake.command()),
    );
    let session = client
        .create(Some(Manifest::new().with_root("/workspace")))
        .await
        .expect("create");

    let archive = session.persist_workspace().await.expect("persist");
    // The image captures the whole directory, so the staged rclone configuration and every
    // ephemeral entry are moved aside first and put back afterwards.
    let scripts: Vec<String> = fake
        .of_op("exec")
        .into_iter()
        .filter_map(|request| request["argv"][2].as_str().map(str::to_string))
        .collect();
    assert!(
        scripts.iter().any(|script| script.contains("tar cf")
            && script.contains(".sandbox-rclone-config")
            && script.contains("rm -rf")),
        "{scripts:?}"
    );
    assert!(
        scripts.iter().any(|script| script.contains("tar xf")),
        "{scripts:?}"
    );
    assert_eq!(
        decode_snapshot_directory_ref(&archive).as_deref(),
        Some("dir-sb-1")
    );
    // The two snapshot kinds have distinct magics, so neither decodes the other's reference.
    assert!(decode_snapshot_filesystem_ref(&archive).is_none());
    let snapshot = &fake.of_op("snapshot_directory")[0];
    assert_eq!(snapshot["path"], "/workspace");

    session.hydrate_workspace(&archive).await.expect("hydrate");
    let mount = &fake.of_op("mount_image")[0];
    assert_eq!(mount["path"], "/workspace");
    assert_eq!(mount["image_id"], "dir-sb-1");
    // Remounting keeps the running sandbox, unlike a filesystem snapshot restore.
    assert_eq!(fake.of_op("create").len(), 1);
}

#[tokio::test]
async fn modal_refuses_a_directory_snapshot_under_a_bucket_mount() {
    let fake = FakeModal::new();
    let client = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_workspace_persistence(ModalWorkspacePersistence::SnapshotDirectory)
            .with_helper_command(fake.command()),
    );
    let manifest = Manifest::new().with_root("/workspace").with_entry(
        "assets",
        Entry::mount(ModalCloudBucketMountStrategy::mount(
            ModalCloudBucketMountConfig::s3("bucket").with_secret_name("aws-creds"),
        )),
    );
    let error = client
        .create(Some(manifest))
        .await
        .expect_err("a bucket under the snapshot root");
    assert_eq!(error.code, ErrorCode::MountConfigInvalid);
    assert_eq!(
        error.context.get("mount_path").map(String::as_str),
        Some("/workspace/assets")
    );
    assert!(fake.of_op("create").is_empty());
}

#[tokio::test]
async fn modal_resolves_an_exposed_port_through_a_tunnel() {
    let fake = FakeModal::new();
    let session = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app")
            .with_exposed_ports([8080, 9000])
            .with_helper_command(fake.command()),
    )
    .create(None)
    .await
    .expect("create");
    let tls = session.resolve_exposed_port(8080).await.expect("resolve");
    assert_eq!(tls.host, "tunnel.modal.host");
    assert_eq!(tls.port, 443);
    assert!(tls.tls);
    assert_eq!(tls.url_for("ws").unwrap(), "wss://tunnel.modal.host/");

    let plain = session.resolve_exposed_port(9000).await.expect("resolve");
    assert_eq!(plain.port, 31000);
    assert!(!plain.tls);

    let error = session
        .resolve_exposed_port(7000)
        .await
        .expect_err("undeclared");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("not_configured")
    );
}

#[tokio::test]
async fn cloudflare_reports_that_it_cannot_resolve_an_exposed_port() {
    let handler: Handler =
        Arc::new(
            |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v1/sandbox") => MockResponse::json(json!({"id": "sbx-77"})),
                _ => MockResponse::status(404, json!({"error": "unexpected"})),
            },
        );
    let server = MockServer::start(handler).await;
    let session = CloudflareSandboxClient::new(
        CloudflareSandboxClientOptions::new(&server.base_url).with_exposed_ports([8080]),
    )
    .create(None)
    .await
    .unwrap();
    // The worker opens the port inside the sandbox but publishes no endpoint for it, so, as in
    // Python, a declared port still cannot be resolved.
    let error = session
        .resolve_exposed_port(8080)
        .await
        .expect_err("cloudflare has no endpoint");
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("backend_unavailable")
    );
    let undeclared = session
        .resolve_exposed_port(9090)
        .await
        .expect_err("undeclared");
    assert_eq!(
        undeclared.context.get("reason").map(String::as_str),
        Some("not_configured")
    );
}

#[tokio::test]
async fn blaxel_creates_a_preview_and_a_token_for_a_private_port() {
    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/v0/sandboxes/agents-test/previews/port-3000") => {
                MockResponse::status(404, json!({"error": "not found"}))
            }
            ("POST", "/v0/sandboxes/agents-test/previews") => MockResponse::json(json!({
                "metadata": {"name": "port-3000"},
                "spec": {"port": 3000, "public": false, "url": "https://preview.blaxel.app"},
            })),
            ("POST", "/v0/sandboxes/agents-test/previews/port-3000/tokens") => {
                MockResponse::json(json!({
                    "metadata": {"name": "tok-1"},
                    "spec": {"token": "preview-token", "expiresAt": "2030-01-01T00:00:00Z"},
                }))
            }
            _ => blaxel_handler(sandbox_url)(request),
        }
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());

    let mut options = blaxel_options(&server.base_url);
    options.exposed_port_public = false;
    options.exposed_port_url_ttl_secs = 600;
    let session = BlaxelSandboxClient::new(options)
        .create(Some(
            Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        ))
        .await
        .unwrap();
    let endpoint = session.resolve_exposed_port(3000).await.expect("resolve");
    assert_eq!(endpoint.host, "preview.blaxel.app");
    assert!(endpoint.tls);
    // A private preview is only reachable with its time-limited token.
    assert_eq!(endpoint.query, "bl_preview_token=preview-token");
    assert_eq!(
        endpoint.url_for("http").unwrap(),
        "https://preview.blaxel.app/?bl_preview_token=preview-token"
    );

    let create = server.find("POST", "/v0/sandboxes/agents-test/previews");
    assert_eq!(
        create.json(),
        json!({
            "metadata": {"name": "port-3000"},
            "spec": {"port": 3000, "public": false},
        })
    );
    let token = server.find(
        "POST",
        "/v0/sandboxes/agents-test/previews/port-3000/tokens",
    );
    assert!(
        token.json()["spec"]["expiresAt"].is_string(),
        "{}",
        token.text()
    );
}

#[tokio::test]
async fn blaxel_reuses_an_existing_public_preview() {
    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        match (request.method.as_str(), request.path.as_str()) {
            // `create_if_not_exists` reads first, so an existing preview is never re-created.
            ("POST", "/v0/sandboxes/agents-test/previews") => {
                MockResponse::status(409, json!({"error": "already exists"}))
            }
            ("GET", "/v0/sandboxes/agents-test/previews/port-3000") => MockResponse::json(json!({
                "metadata": {"name": "port-3000"},
                "spec": {"port": 3000, "public": true, "url": "https://public.blaxel.app:8443/"},
            })),
            _ => blaxel_handler(sandbox_url)(request),
        }
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());

    let session = BlaxelSandboxClient::new(blaxel_options(&server.base_url))
        .create(Some(
            Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT),
        ))
        .await
        .unwrap();
    let endpoint = session.resolve_exposed_port(3000).await.expect("resolve");
    assert_eq!(endpoint.host, "public.blaxel.app");
    assert_eq!(endpoint.port, 8443);
    // A public preview mints no token.
    assert!(endpoint.query.is_empty());
    assert!(
        server
            .requests()
            .iter()
            .all(|request| !request.path.ends_with("/tokens"))
    );
    // Reading first means an existing preview is never re-created.
    assert!(
        server
            .requests()
            .iter()
            .all(|request| request.path != "/v0/sandboxes/agents-test/previews")
    );
}

// ---------------------------------------------------------------------------------------------
// The core hooks the sandbox-core area added: bounded reads and write grants
// ---------------------------------------------------------------------------------------------

/// A Cloudflare session against a worker that serves `body` for every file read.
async fn cloudflare_reading(body: Vec<u8>) -> (MockServer, Arc<dyn SandboxSession>) {
    let served = Arc::new(body);
    let handler: Handler =
        Arc::new(
            move |request| match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v1/sandbox") => MockResponse::json(json!({"id": "sbx-77"})),
                ("GET", path) if path.starts_with("/v1/sandbox/sbx-77/file/") => {
                    MockResponse::bytes(served.as_ref().clone())
                }
                _ => MockResponse::status(404, json!({"error": "unexpected"})),
            },
        );
    let server = MockServer::start(handler).await;
    let session =
        CloudflareSandboxClient::new(CloudflareSandboxClientOptions::new(&server.base_url))
            .create(None)
            .await
            .unwrap();
    (server, session)
}

#[tokio::test]
async fn cloudflare_bounds_a_raw_read_at_max_bytes() {
    let (_server, session) = cloudflare_reading(vec![b'a'; 4096]).await;
    let read = session
        .read_bounded("/workspace/big.log", 64)
        .await
        .expect("bounded read");
    // A raw body is bounded by `max_bytes` itself, not by the wire budget.
    assert_eq!(read.len(), 64);
    assert!(read.iter().all(|byte| *byte == b'a'));
}

#[tokio::test]
async fn cloudflare_refuses_an_sse_body_over_the_bounded_read_wire_limit() {
    // The worker's SSE framing base64s every chunk, so the wire form of a file is several times
    // its size; bounding only the decoded result would let the sandbox decide how much this
    // process buffers. The budget is `8 * max_bytes + 65536`.
    let max_bytes = 64usize;
    let wire_limit = 8 * max_bytes + 65536;
    let mut body = b"data: {\"type\":\"metadata\",\"isBinary\":false}\n\n".to_vec();
    while body.len() <= wire_limit {
        body.extend_from_slice(b"data: {\"type\":\"chunk\",\"data\":\"");
        body.extend_from_slice(&vec![b'A'; 4096]);
        body.extend_from_slice(b"\"}\n\n");
    }
    let (_server, session) = cloudflare_reading(body).await;

    let error = session
        .read_bounded("/workspace/big.log", max_bytes)
        .await
        .expect_err("the wire limit is enforced");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveReadError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("bounded_read_wire_limit")
    );
    // A refused request is a configuration fact, not something a retry fixes.
    assert_eq!(error.retryable, Some(false));
}

#[tokio::test]
async fn cloudflare_decodes_an_sse_body_inside_the_wire_limit() {
    let payload = "hello from the worker";
    let body = format!(
        "data: {{\"type\":\"metadata\",\"isBinary\":false}}\n\n\
         data: {{\"type\":\"chunk\",\"data\":\"{payload}\"}}\n\n\
         data: {{\"type\":\"complete\"}}\n\n"
    );
    let (_server, session) = cloudflare_reading(body.into_bytes()).await;
    let read = session
        .read_bounded("/workspace/note.txt", 5)
        .await
        .expect("bounded read");
    assert_eq!(read, b"hello");
}

#[tokio::test]
async fn modal_and_blaxel_refuse_a_write_through_a_read_only_path_grant() {
    let grant = openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant::new("/shared/data")
        .expect("grant")
        .read_only(true);

    let fake = FakeModal::new();
    let modal = ModalSandboxClient::new(
        ModalSandboxClientOptions::new("agents-app").with_helper_command(fake.command()),
    )
    .create(Some(Manifest::new().with_extra_path_grant(grant.clone())))
    .await
    .expect("create");
    let error = modal
        .write_file("/shared/data/out.txt", b"x")
        .await
        .expect_err("a write through a read-only grant");
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("read_only_extra_path_grant")
    );
    // The refusal is the SDK's, so the helper is never asked to write.
    assert!(fake.of_op("write_file").is_empty());

    let shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let url_slot = shared.clone();
    let handler: Handler = Arc::new(move |request| {
        let sandbox_url = url_slot.lock().unwrap().clone().unwrap_or_default();
        blaxel_handler(sandbox_url)(request)
    });
    let server = MockServer::start(handler).await;
    *shared.lock().unwrap() = Some(server.base_url.clone());
    let blaxel = BlaxelSandboxClient::new(blaxel_options(&server.base_url))
        .create(Some(
            Manifest::new()
                .with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
                .with_extra_path_grant(grant),
        ))
        .await
        .unwrap();
    assert_eq!(
        blaxel
            .write_file("/shared/data/out.txt", b"x")
            .await
            .expect_err("a write through a read-only grant")
            .code,
        ErrorCode::WorkspaceArchiveWriteError
    );
}
