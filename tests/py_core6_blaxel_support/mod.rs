//! A mock Blaxel control plane and sandbox API shared by the `py_core6_blaxel_*` test binaries.
//!
//! It serves both on one origin, records every request (through [`MockServer`]), and plays the
//! in-sandbox `resolve-workspace-path` helper the session runs before each read, write and mkdir.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use openai_agents_sdk::extensions::sandbox::blaxel::{
    BlaxelSandboxClient, BlaxelSandboxClientOptions, BlaxelTimeouts, DEFAULT_BLAXEL_WORKSPACE_ROOT,
};
use openai_agents_sdk::sandbox::errors::SandboxError;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSession};
use serde_json::{Value, json};

use crate::support::{self, MockResponse, MockServer, Recorded, Shared, percent_decode};

pub const HELPER_PREFIX: &str = "/tmp/openai-agents/bin/resolve-workspace-path";

// ---------------------------------------------------------------------------------------------
// The mock sandbox
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct World {
    /// Files by absolute sandbox path.
    pub files: HashMap<String, Vec<u8>>,
    /// Workspace symlinks the path resolver sees, `link -> target`.
    pub symlinks: HashMap<String, String>,
    /// Results for the next ordinary commands, as `(exit_code, stdout, stderr)`.
    pub exec_queue: VecDeque<(i64, String, String)>,
    pub exec_delay_ms: u64,
    /// A failing `POST /process`, as `(status, body)`.
    pub exec_failure: Option<(u16, Value)>,
    pub fs_read_status: Option<u16>,
    pub fs_write_status: Option<u16>,
    /// How many of the next filesystem writes fail with a 500 before writes succeed again.
    pub fs_write_failures: usize,
    /// The status of `POST /drives/mount`, and of `DELETE /drives/mount/...`.
    pub drive_mount_status: Option<u16>,
    pub drive_unmount_status: Option<u16>,
    pub fs_delete_status: Option<u16>,
    pub fs_list_status: Option<u16>,
    pub delete_status: Option<u16>,
    pub get_status: Option<u16>,
    /// What `POST /sandboxes` reports as `metadata.url`: `None` points back at the mock.
    pub sandbox_url: Option<Option<String>>,
    /// Previews that already exist, by name.
    pub previews: HashMap<String, Value>,
    /// The reply to `POST .../previews`.
    pub preview_create: Option<(u16, Value)>,
    /// The reply to `POST .../previews/{name}/tokens`.
    pub token_reply: Option<(u16, Value)>,
}

pub fn json_body(request: &Recorded) -> Value {
    request.json()
}

/// The multipart `file` part of a filesystem write.
pub fn multipart_file(request: &Recorded) -> Vec<u8> {
    let marker = b"name=\"file\"";
    let body = &request.body;
    let start = support::find_subslice(body, marker).expect("a file part");
    let after = support::find_subslice(&body[start..], b"\r\n\r\n").expect("part header end") + 4;
    let content_start = start + after;
    let end = support::find_subslice(&body[content_start..], b"\r\n--openai-agents-")
        .expect("part end boundary");
    body[content_start..content_start + end].to_vec()
}

/// What the in-sandbox `resolve-workspace-path` helper answers for `tokens`.
pub fn emulate_resolver(world: &World, tokens: &[&str]) -> (i64, String, String) {
    let root = tokens[1];
    let lexical = tokens[2];
    let mode = tokens[3];
    let mut resolved = lexical.to_string();
    for (link, target) in &world.symlinks {
        if lexical == link || lexical.starts_with(&format!("{link}/")) {
            resolved = format!("{target}{}", &lexical[link.len()..]);
        }
    }
    let within = |path: &str, base: &str| {
        path == base || path.starts_with(&format!("{}/", base.trim_end_matches('/')))
    };
    let grants: Vec<(&str, bool)> = tokens[4..]
        .chunks(2)
        .map(|pair| (pair[0], pair[1] == "1"))
        .collect();
    if mode != "0"
        && let Some((grant, _)) = grants
            .iter()
            .find(|(grant, read_only)| *read_only && within(&resolved, grant))
    {
        return (
            114,
            String::new(),
            format!("read-only extra path grant: {grant}\nresolved path: {resolved}\n"),
        );
    }
    let allowed =
        within(&resolved, root) || grants.iter().any(|(grant, _)| within(&resolved, grant));
    if !allowed {
        return (111, String::new(), format!("workspace escape: {resolved}"));
    }
    (0, resolved, String::new())
}

pub fn process_reply(world: &Shared<World>, request: &Recorded) -> MockResponse {
    let command = json_body(request)["command"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let (exit_code, stdout, stderr) = if command.starts_with("test -x") {
        (0, String::new(), String::new())
    } else if command.starts_with(HELPER_PREFIX) {
        world.with(|w| {
            let tokens: Vec<&str> = command.split_whitespace().collect();
            emulate_resolver(w, &tokens)
        })
    } else {
        world.with(|w| {
            w.exec_queue
                .pop_front()
                .unwrap_or((0, String::new(), String::new()))
        })
    };
    let (failure, delay) = world.with(|w| (w.exec_failure.clone(), w.exec_delay_ms));
    if let Some((status, body)) = failure
        && !command.starts_with("test -x")
        && !command.starts_with(HELPER_PREFIX)
    {
        return MockResponse::status(status, body);
    }
    let body = json!({
        "exitCode": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "status": "completed",
    });
    if delay > 0 && !command.starts_with(HELPER_PREFIX) && !command.starts_with("test -x") {
        return MockResponse::streamed(
            "application/json",
            vec![(serde_json::to_vec(&body).unwrap(), delay)],
        );
    }
    MockResponse::json(body)
}

pub fn filesystem_reply(world: &Shared<World>, request: &Recorded) -> MockResponse {
    let encoded = request.path.trim_start_matches("/filesystem/");
    let path = percent_decode(encoded);
    let w = world.get();
    match request.method.as_str() {
        "GET" => {
            if request.header("accept") == Some("application/octet-stream") {
                if let Some(status) = w.fs_read_status {
                    return MockResponse::error(status, "read failed");
                }
                return match w.files.get(&path) {
                    Some(bytes) => MockResponse::bytes(bytes.clone()),
                    None => MockResponse::error(404, "file not found"),
                };
            }
            if let Some(status) = w.fs_list_status {
                return MockResponse::error(status, "list failed");
            }
            MockResponse::json(json!({
                "name": "root", "path": path, "files": [], "subdirectories": [],
            }))
        }
        "PUT" => {
            if let Some(status) = w.fs_write_status {
                return MockResponse::error(status, "write failed");
            }
            let flaky = world.with(|state| {
                let flaky = state.fs_write_failures > 0;
                state.fs_write_failures = state.fs_write_failures.saturating_sub(1);
                flaky
            });
            if flaky {
                return MockResponse::error(500, "write failed");
            }
            if request
                .header("content-type")
                .is_some_and(|value| value.starts_with("multipart/form-data"))
            {
                let bytes = multipart_file(request);
                world.with(|state| state.files.insert(path, bytes));
            }
            MockResponse::json(json!({ "success": true }))
        }
        "DELETE" => match w.fs_delete_status {
            Some(status) => MockResponse::error(status, "delete failed"),
            None => {
                world.with(|state| state.files.remove(&path));
                MockResponse::json(json!({ "success": true }))
            }
        },
        _ => MockResponse::status(405, json!({})),
    }
}

pub async fn start(world: Shared<World>) -> MockServer {
    let url = Shared::new(String::new());
    let slot = url.clone();
    let server = MockServer::start(move |request: &Recorded| {
        let own_url = slot.get();
        let method = request.method.as_str();
        let path = request.path.as_str();
        let w = world.get();
        let metadata_url = match &w.sandbox_url {
            None => Some(own_url.clone()),
            Some(value) => value.clone(),
        };
        let sandbox_json = match metadata_url {
            Some(url) => json!({ "metadata": { "name": "agents-test", "url": url } }),
            None => json!({ "metadata": { "name": "agents-test" } }),
        };
        match (method, path) {
            ("POST", "/v0/sandboxes") => MockResponse::json(sandbox_json),
            ("GET", "/v0/sandboxes/agents-test") => match w.get_status {
                Some(status) => MockResponse::error(status, "gone"),
                None => MockResponse::json(sandbox_json),
            },
            ("DELETE", "/v0/sandboxes/agents-test") => match w.delete_status {
                Some(status) => MockResponse::error(status, "cannot delete"),
                None => MockResponse::empty(204),
            },
            ("POST", "/process") => process_reply(&world, request),
            ("POST", "/drives/mount") => match w.drive_mount_status {
                Some(status) => MockResponse::error(status, "drive mount failed"),
                None => MockResponse::json(json!({ "success": true })),
            },
            ("DELETE", p) if p.starts_with("/drives/mount/") => match w.drive_unmount_status {
                Some(status) => MockResponse::error(status, "drive unmount failed"),
                None => MockResponse::json(json!({ "success": true })),
            },
            (_, p) if p.starts_with("/filesystem/") => filesystem_reply(&world, request),
            ("GET", p) if p.starts_with("/v0/sandboxes/agents-test/previews/") => {
                let name = p.rsplit('/').next().unwrap();
                match w.previews.get(name) {
                    Some(preview) => MockResponse::json(preview.clone()),
                    None => MockResponse::error(404, "no such preview"),
                }
            }
            ("POST", "/v0/sandboxes/agents-test/previews") => match w.preview_create {
                Some((status, body)) => MockResponse::status(status, body),
                None => MockResponse::json(json!({
                    "metadata": { "name": "port-0" },
                    "spec": { "url": "https://preview.example.com" },
                })),
            },
            ("POST", p) if p.contains("/previews/") && p.ends_with("/tokens") => {
                match w.token_reply {
                    Some((status, body)) => MockResponse::status(status, body),
                    None => MockResponse::json(json!({ "spec": { "token": "tok-1" } })),
                }
            }
            _ => MockResponse::status(404, json!({ "error": { "code": "ROUTE_NOT_FOUND" } })),
        }
    })
    .await;
    url.set(server.base_url.clone());
    server
}

pub fn options(base_url: &str) -> BlaxelSandboxClientOptions {
    BlaxelSandboxClientOptions::new()
        .with_name("agents-test")
        .with_api_key("bl-token")
        .with_workspace("acme")
        .with_region("us-pdx-1")
        .with_api_base_url(format!("{base_url}/v0"))
        .with_timeouts(BlaxelTimeouts {
            exec: Duration::from_secs(5),
            ..BlaxelTimeouts::default()
        })
}

pub fn root_manifest() -> Manifest {
    Manifest::new().with_root(DEFAULT_BLAXEL_WORKSPACE_ROOT)
}

pub async fn session_with(
    server: &MockServer,
    options: BlaxelSandboxClientOptions,
    manifest: Manifest,
) -> Arc<dyn SandboxSession> {
    let _ = server;
    BlaxelSandboxClient::new(options)
        .create(Some(manifest))
        .await
        .expect("create")
}

pub async fn session(server: &MockServer) -> Arc<dyn SandboxSession> {
    session_with(server, options(&server.base_url), root_manifest()).await
}

pub fn protected_grant() -> SandboxPathGrant {
    SandboxPathGrant::new("/tmp/protected")
        .expect("grant")
        .read_only(true)
}

pub fn count_prefixed(server: &MockServer, method: &str, prefix: &str) -> usize {
    server
        .requests()
        .iter()
        .filter(|request| request.method == method && request.path.starts_with(prefix))
        .count()
}

pub fn context(error: &SandboxError, key: &str) -> Option<String> {
    error.context.get(key).cloned()
}
