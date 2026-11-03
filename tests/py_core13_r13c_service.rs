//! Ports of `tests/sandbox/test_docker_removal.py` (and the shared `_docker_removal_helpers.py`):
//! the host-side removal service, its pause/bind/remove protocol, and the Docker client wiring.
//!
//! Python replaces `DockerClient`, the container object and the worker with recording doubles.
//! Here the daemon is a fake [`DockerRunner`] (inspect, pause, unpause, start, rm, exec), `/proc`
//! is a fake [`RemovalHost`], and the worker is a recording [`RemovalWorker`] that asserts the
//! container is paused whenever it is spoken to. A few cases use a real worker transport: a shell
//! script standing in for the `docker-removal-worker` executable.
#![cfg(all(feature = "docker-removal", unix))]

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DockerRunner, DockerSandboxClient, DockerSandboxSession,
};
use openai_agents_sdk::sandbox::sandboxes::docker_removal::{
    DockerRemovalOptions, DockerRemovalService, RemovalHost, RemovalWorker, WorkerFailure,
    WorkerLauncher,
};
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxError, SandboxResult, SandboxSession, User,
    clear_workspace_root_on_resume,
};
use serde_json::{Value, json};
use tokio::sync::Notify;

fn id_of(letter: char) -> String {
    letter.to_string().repeat(64)
}

fn out(stdout: &str) -> SandboxResult<ExecResult> {
    Ok(ExecResult {
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
        exit_code: 0,
    })
}

fn fail(stderr: &str) -> SandboxResult<ExecResult> {
    Ok(ExecResult {
        stdout: Vec::new(),
        stderr: stderr.as_bytes().to_vec(),
        exit_code: 1,
    })
}

fn grant(path: &str) -> SandboxPathGrant {
    SandboxPathGrant::new(path).expect("grant path")
}

fn manifest() -> Manifest {
    Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(grant("/external"))
        .with_extra_path_grant(grant("/grant-alias").read_only(true))
}

// --------------------------------------------------------------------------------------------
// The fake daemon (Python's `RecordingContainer` plus `DockerClient`)
// --------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Container {
    paused: bool,
    pid: u32,
    mounts: Value,
    host_config: Value,
    user: String,
    events: Vec<String>,
    removed_pauses: Vec<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RmBehavior {
    Removed,
    NotFound,
    Error,
}

type ExecHook = Box<dyn Fn(&[String]) -> Option<SandboxResult<ExecResult>> + Send + Sync>;

struct Daemon {
    containers: Mutex<BTreeMap<String, Container>>,
    calls: Mutex<Vec<Vec<String>>>,
    create_ids: Mutex<VecDeque<String>>,
    info: Mutex<Value>,
    version: Mutex<String>,
    fail_inspect: AtomicBool,
    fail_unpause: AtomicBool,
    rm: Mutex<RmBehavior>,
    exec_hook: Mutex<Option<ExecHook>>,
}

impl std::fmt::Debug for Daemon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Daemon").finish_non_exhaustive()
    }
}

impl Daemon {
    fn new() -> Arc<Self> {
        Arc::new(Daemon {
            containers: Mutex::new(BTreeMap::new()),
            calls: Mutex::new(Vec::new()),
            create_ids: Mutex::new(VecDeque::new()),
            info: Mutex::new(json!({
                "SecurityOptions": ["name=seccomp,profile=builtin"],
                "DefaultRuntime": "runc",
            })),
            version: Mutex::new("26.0.0".to_string()),
            fail_inspect: AtomicBool::new(false),
            fail_unpause: AtomicBool::new(false),
            rm: Mutex::new(RmBehavior::Removed),
            exec_hook: Mutex::new(None),
        })
    }

    fn add(&self, id: &str, pid: u32) {
        self.containers.lock().unwrap().insert(
            id.to_string(),
            Container {
                paused: false,
                pid,
                mounts: json!([]),
                host_config: json!({}),
                user: "1000:1000".to_string(),
                events: Vec::new(),
                removed_pauses: Vec::new(),
            },
        );
    }

    fn with<T>(&self, id: &str, f: impl FnOnce(&mut Container) -> T) -> T {
        f(self
            .containers
            .lock()
            .unwrap()
            .get_mut(id)
            .expect("container"))
    }

    fn paused(&self, id: &str) -> bool {
        self.with(id, |c| c.paused)
    }

    fn events(&self, id: &str) -> Vec<String> {
        self.with(id, |c| c.events.clone())
    }

    fn push_event(&self, id: &str, event: &str) {
        self.with(id, |c| c.events.push(event.to_string()));
    }

    fn calls(&self) -> Vec<Vec<String>> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl DockerRunner for Daemon {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().unwrap().push(argv.to_vec());
        let parts: Vec<&str> = argv.iter().map(String::as_str).collect();
        match parts.as_slice() {
            ["container", "inspect", id] => {
                if self.fail_inspect.load(Ordering::SeqCst) {
                    return fail("cleanup unavailable");
                }
                let containers = self.containers.lock().unwrap();
                let Some(container) = containers.get(*id) else {
                    return fail("Error: No such object");
                };
                out(&json!([{
                    "Id": id,
                    "State": {
                        "Running": true,
                        "Paused": container.paused,
                        "Pid": container.pid,
                        "StartedAt": "incarnation",
                        "Restarting": false,
                    },
                    "HostConfig": container.host_config,
                    "Config": {"User": container.user},
                    "Mounts": container.mounts,
                }])
                .to_string())
            }
            ["info", ..] => out(&self.info.lock().unwrap().to_string()),
            ["version", ..] => out(&self.version.lock().unwrap().clone()),
            ["pause", id] => {
                self.with(id, |c| {
                    c.events.push("pause".to_string());
                    c.paused = true;
                });
                out("")
            }
            ["unpause", id] => {
                if self.fail_unpause.load(Ordering::SeqCst) {
                    return fail("cleanup unavailable");
                }
                self.with(id, |c| {
                    c.events.push("unpause".to_string());
                    c.paused = false;
                });
                out("")
            }
            ["start", id] => {
                self.push_event(id, "start");
                out("")
            }
            ["create", ..] => {
                let id = self
                    .create_ids
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("a scripted container id");
                out(&format!("{id}\n"))
            }
            ["rm", "-f", id] => {
                let behavior = *self.rm.lock().unwrap();
                self.with(id, |c| {
                    let paused = c.paused;
                    c.removed_pauses.push(paused);
                });
                match behavior {
                    RmBehavior::Removed => out(""),
                    RmBehavior::NotFound => fail(&format!("Error: No such container: {id}")),
                    RmBehavior::Error => fail("Cannot connect to the Docker daemon"),
                }
            }
            ["stop", ..] | ["volume", ..] => out(""),
            ["exec", ..] => {
                if let Some(hook) = self.exec_hook.lock().unwrap().as_ref()
                    && let Some(result) = hook(argv)
                {
                    return result;
                }
                out("ok\n")
            }
            _ => out(""),
        }
    }
}

struct FakeHost {
    daemon: Arc<Daemon>,
    cgroup: Mutex<Option<String>>,
}

impl RemovalHost for FakeHost {
    fn read_proc(&self, pid: u32, name: &str) -> io::Result<String> {
        let id = self
            .daemon
            .containers
            .lock()
            .unwrap()
            .iter()
            .find(|(_, container)| container.pid == pid)
            .map(|(id, _)| id.clone())
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))?;
        Ok(match name {
            "cgroup" => self
                .cgroup
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| format!("0::/system.slice/docker-{id}.scope\n")),
            _ => "0 0 4294967295\n".to_string(),
        })
    }
}

// --------------------------------------------------------------------------------------------
// The recording worker (Python's `RecordingWorker`)
// --------------------------------------------------------------------------------------------

struct Gate {
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

type Pre = Box<dyn Fn(&Value) -> Option<Result<Value, WorkerFailure>> + Send + Sync>;
type Post =
    Box<dyn Fn(&Value, Result<Value, WorkerFailure>) -> Result<Value, WorkerFailure> + Send + Sync>;

struct WorkerShared {
    daemon: Arc<Daemon>,
    container: String,
    uncertain: AtomicBool,
    aliases: Mutex<HashMap<String, String>>,
    calls: Mutex<Vec<Value>>,
    removed: Mutex<Vec<String>>,
    selected: Mutex<String>,
    pre: Mutex<Option<Pre>>,
    post: Mutex<Option<Post>>,
    remove_gate: Mutex<Option<Gate>>,
    close_gate: Mutex<Option<Gate>>,
    close_error: AtomicBool,
    close_calls: AtomicUsize,
}

impl WorkerShared {
    fn new(daemon: &Arc<Daemon>, container: &str) -> Arc<Self> {
        Arc::new(WorkerShared {
            daemon: daemon.clone(),
            container: container.to_string(),
            uncertain: AtomicBool::new(false),
            aliases: Mutex::new(HashMap::from([(
                "/grant-alias".to_string(),
                "/external/protected".to_string(),
            )])),
            calls: Mutex::new(Vec::new()),
            removed: Mutex::new(Vec::new()),
            selected: Mutex::new(String::new()),
            pre: Mutex::new(None),
            post: Mutex::new(None),
            remove_gate: Mutex::new(None),
            close_gate: Mutex::new(None),
            close_error: AtomicBool::new(false),
            close_calls: AtomicUsize::new(0),
        })
    }

    fn alias(&self, from: &str, to: &str) {
        self.aliases
            .lock()
            .unwrap()
            .insert(from.to_string(), to.to_string());
    }

    fn operations(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|call| call["operation"].as_str().unwrap().to_string())
            .collect()
    }

    fn removed(&self) -> Vec<String> {
        self.removed.lock().unwrap().clone()
    }

    fn set_pre(
        &self,
        hook: impl Fn(&Value) -> Option<Result<Value, WorkerFailure>> + Send + Sync + 'static,
    ) {
        *self.pre.lock().unwrap() = Some(Box::new(hook));
    }

    fn set_post(
        &self,
        hook: impl Fn(&Value, Result<Value, WorkerFailure>) -> Result<Value, WorkerFailure>
        + Send
        + Sync
        + 'static,
    ) {
        *self.post.lock().unwrap() = Some(Box::new(hook));
    }

    fn gate_remove(&self) -> (Arc<Notify>, Arc<Notify>) {
        let gate = Gate {
            entered: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let handles = (gate.entered.clone(), gate.release.clone());
        *self.remove_gate.lock().unwrap() = Some(gate);
        handles
    }
}

struct RecordingWorker(Arc<WorkerShared>);

#[async_trait]
impl RemovalWorker for RecordingWorker {
    async fn request(&mut self, request: Value) -> Result<Value, WorkerFailure> {
        let shared = &self.0;
        assert!(
            shared.daemon.paused(&shared.container),
            "the worker is only spoken to while the container is paused"
        );
        let operation = request["operation"].as_str().unwrap().to_string();
        if operation == "remove" {
            let gate = shared
                .remove_gate
                .lock()
                .unwrap()
                .as_ref()
                .map(|gate| (gate.entered.clone(), gate.release.clone()));
            if let Some((entered, release)) = gate {
                entered.notify_one();
                release.notified().await;
            }
        }
        if let Some(pre) = shared.pre.lock().unwrap().as_ref()
            && let Some(result) = pre(&request)
        {
            return result;
        }
        shared.calls.lock().unwrap().push(request.clone());
        let aliases = shared.aliases.lock().unwrap().clone();
        let result = match operation.as_str() {
            "bind" => {
                let paths: Vec<Value> = request["paths"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|path| {
                        let path = path.as_str().unwrap();
                        Value::String(
                            aliases
                                .get(path)
                                .cloned()
                                .unwrap_or_else(|| path.to_string()),
                        )
                    })
                    .collect();
                Ok(json!({"ok": true, "paths": paths}))
            }
            "inspect" => {
                let path = request["path"].as_str().unwrap();
                let selected = aliases
                    .get(path)
                    .cloned()
                    .unwrap_or_else(|| path.to_string());
                *shared.selected.lock().unwrap() = selected.clone();
                Ok(json!({"ok": true, "path": selected, "is_directory": true}))
            }
            _ => {
                let selected = shared.selected.lock().unwrap().clone();
                shared.removed.lock().unwrap().push(selected);
                Ok(json!({"ok": true}))
            }
        };
        match shared.post.lock().unwrap().as_ref() {
            Some(post) => post(&request, result),
            None => result,
        }
    }

    fn uncertain(&self) -> bool {
        self.0.uncertain.load(Ordering::SeqCst)
    }

    async fn close(&mut self) -> io::Result<()> {
        let shared = &self.0;
        shared.close_calls.fetch_add(1, Ordering::SeqCst);
        let gate = shared
            .close_gate
            .lock()
            .unwrap()
            .as_ref()
            .map(|gate| (gate.entered.clone(), gate.release.clone()));
        if let Some((entered, release)) = gate {
            entered.notify_one();
            release.notified().await;
        }
        if shared.close_error.load(Ordering::SeqCst) {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "cleanup failed"));
        }
        shared.daemon.push_event(&shared.container, "close");
        Ok(())
    }
}

struct QueueLauncher(Mutex<VecDeque<Arc<WorkerShared>>>);

#[async_trait]
impl WorkerLauncher for QueueLauncher {
    async fn launch(&self, _pid: u32) -> Result<Box<dyn RemovalWorker>, WorkerFailure> {
        match self.0.lock().unwrap().pop_front() {
            Some(shared) => Ok(Box::new(RecordingWorker(shared))),
            None => Err(WorkerFailure::Transport("no scripted worker".to_string())),
        }
    }
}

struct Harness {
    daemon: Arc<Daemon>,
    host: Arc<FakeHost>,
    service: DockerRemovalService,
    workers: Vec<Arc<WorkerShared>>,
    ids: Vec<String>,
}

impl Harness {
    fn worker(&self) -> &Arc<WorkerShared> {
        &self.workers[0]
    }

    fn id(&self) -> &str {
        &self.ids[0]
    }

    fn events(&self) -> Vec<String> {
        self.daemon.events(self.id())
    }

    fn session(&self, manifest: Manifest) -> DockerSandboxSession {
        DockerSandboxSession::new(self.id(), manifest).with_removal_service(self.service.clone())
    }

    fn session_for(&self, index: usize, manifest: Manifest) -> DockerSandboxSession {
        DockerSandboxSession::new(&self.ids[index], manifest)
            .with_removal_service(self.service.clone())
    }

    fn client(&self) -> DockerSandboxClient {
        DockerSandboxClient::new().with_removal_service(self.service.clone())
    }
}

fn harness_of(containers: usize, options: DockerRemovalOptions) -> Harness {
    let daemon = Daemon::new();
    let mut workers = Vec::new();
    let mut ids = Vec::new();
    for index in 0..containers {
        let id = id_of((b'a' + index as u8) as char);
        daemon.add(&id, 123 + index as u32);
        daemon.create_ids.lock().unwrap().push_back(id.clone());
        workers.push(WorkerShared::new(&daemon, &id));
        ids.push(id);
    }
    let host = Arc::new(FakeHost {
        daemon: daemon.clone(),
        cgroup: Mutex::new(None),
    });
    let service = DockerRemovalService::try_from_parts(
        options,
        daemon.clone(),
        host.clone(),
        Arc::new(QueueLauncher(Mutex::new(workers.iter().cloned().collect()))),
    )
    .expect("service");
    Harness {
        daemon,
        host,
        service,
        workers,
        ids,
    }
}

fn limits() -> DockerRemovalOptions {
    DockerRemovalOptions::new(4, 100_000, 10)
}

fn harness() -> Harness {
    harness_of(1, limits())
}

async fn bound(manifest: &Manifest) -> Harness {
    let h = harness();
    h.service.bind_new(h.id(), manifest).await.expect("bind");
    h
}

fn reason(error: &SandboxError) -> &str {
    error
        .context
        .get("reason")
        .map(String::as_str)
        .unwrap_or("")
}

async fn eventually(mut condition: impl FnMut() -> bool) {
    for _ in 0..300 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the condition never became true");
}

// --------------------------------------------------------------------------------------------
// Binding
// --------------------------------------------------------------------------------------------

#[tokio::test]
async fn binding_rejects_canonical_backslashes() {
    for bound_path in ["/workspace", "/external", "/grant-alias"] {
        let h = harness();
        h.worker().alias(bound_path, r"/private/\root");
        let error = h.service.bind_new(h.id(), &manifest()).await.unwrap_err();
        assert!(
            error
                .message
                .contains("canonical paths containing backslashes"),
            "{error}"
        );
        assert!(!h.service.is_bound(h.id()));
        assert!(h.worker().removed().is_empty());
        assert_eq!(h.events(), ["pause", "close", "unpause"], "{bound_path}");
    }
}

#[tokio::test]
async fn writable_shared_mounts_cannot_acquire_authority() {
    let h = harness();
    let configured = Manifest::new()
        .with_root("/workspace")
        .with_extra_path_grant(
            grant("/toolchain")
                .with_host_path("/host/toolchain")
                .unwrap(),
        );
    let error = h.service.bind_new(h.id(), &configured).await.unwrap_err();
    assert!(error.message.contains("shared host paths"), "{error}");
    assert!(h.worker().calls.lock().unwrap().is_empty());
    assert!(!h.daemon.paused(h.id()));
}

#[tokio::test]
async fn read_only_host_binding_rejects_untrusted_mount_layout() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().canonicalize().unwrap().join("toolchain");
    let source = source.to_str().unwrap();
    for invalid in ["source", "writable", "missing", "workspace_alias"] {
        let h = harness();
        let configured = Manifest::new()
            .with_root("/workspace")
            .with_extra_path_grant(
                grant("/toolchain")
                    .with_host_path(source)
                    .unwrap()
                    .read_only(true),
            );
        let mut mount = json!({
            "Type": "bind", "Source": source, "Destination": "/toolchain",
            "RW": false, "Propagation": "rprivate",
        });
        if invalid == "source" {
            mount["Source"] = json!(format!("{source}-other"));
        }
        if invalid == "writable" {
            mount["RW"] = json!(true);
        }
        h.daemon.with(h.id(), |c| {
            c.mounts = if invalid == "missing" {
                json!([])
            } else {
                json!([mount])
            };
        });
        if invalid == "workspace_alias" {
            h.worker().alias("/toolchain", "/workspace/mounted");
        }
        assert!(
            h.service.bind_new(h.id(), &configured).await.is_err(),
            "{invalid}"
        );
        assert!(!h.service.is_bound(h.id()), "{invalid}");
        assert!(h.worker().removed().is_empty());
        assert!(!h.daemon.paused(h.id()), "{invalid}");
    }
}

#[tokio::test]
async fn existing_pause_is_owned_by_its_caller() {
    let h = harness();
    h.daemon.with(h.id(), |c| c.paused = true);
    h.service.bind_new(h.id(), &manifest()).await.unwrap();
    h.service
        .remove(h.id(), &manifest(), "build", None)
        .await
        .unwrap();
    assert!(h.daemon.paused(h.id()));
    assert!(h.events().is_empty(), "{:?}", h.events());
    let calls = h.worker().calls.lock().unwrap().clone();
    assert_eq!(calls.last().unwrap()["user"], "1000:1000");
}

#[tokio::test]
async fn binding_cannot_be_reconstructed_from_persisted_configuration() {
    let h = harness();
    let error = h
        .service
        .assert_bound(h.id(), &manifest())
        .await
        .unwrap_err();
    assert!(error.message.contains("original live authority"), "{error}");
}

#[tokio::test]
async fn changed_configuration_requires_new_authority() {
    let h = bound(&manifest()).await;
    let error = h
        .service
        .remove(
            h.id(),
            &Manifest::new().with_root("/workspace"),
            "build",
            None,
        )
        .await
        .unwrap_err();
    assert!(error.message.contains("original live authority"), "{error}");
    assert!(h.worker().removed().is_empty());
}

#[tokio::test]
async fn live_manifest_update_preserves_removal_authority() {
    for (is_bound, change_grants) in [(true, true), (true, false), (false, true)] {
        let h = harness();
        let configured = manifest();
        let current = if is_bound {
            h.service.bind_new(h.id(), &configured).await.unwrap();
            h.session(configured.clone())
        } else {
            DockerSandboxSession::new(h.id(), configured.clone()).with_runner(h.daemon.clone())
        };
        let mut updated = configured.clone();
        if change_grants {
            updated = updated.with_extra_path_grant(grant("/new-grant").read_only(true));
        }
        let verdict = current
            .validate_manifest_application(&updated, Some(true))
            .await;
        if is_bound && change_grants {
            let error = verdict.unwrap_err();
            assert!(
                error.message.contains("original live authority binding"),
                "{error}"
            );
            assert_eq!(h.worker().operations(), ["bind"]);
        } else {
            verdict.expect("an unchanged or unbound session accepts the update");
        }
        if is_bound {
            current.rm("build", true).await.unwrap();
            assert_eq!(h.worker().removed(), ["/workspace/build"]);
        }
    }
}

// --------------------------------------------------------------------------------------------
// Removal
// --------------------------------------------------------------------------------------------

#[tokio::test]
async fn removal_rejects_canonical_backslashes() {
    for is_directory in [true, false] {
        let configured = Manifest::new()
            .with_root("/workspace")
            .with_extra_path_grant(grant("/protected").read_only(true));
        let h = bound(&configured).await;
        h.worker()
            .alias("/workspace/link/data", r"/protected/\..\..\workspace/data");
        h.worker().set_post(move |request, result| {
            let mut value = result?;
            if request["operation"] == "inspect" {
                value["is_directory"] = json!(is_directory);
            }
            Ok(value)
        });
        let current = h.session(configured.clone());
        let error = current.rm("link/data", true).await.unwrap_err();
        assert_eq!(reason(&error), "docker_removal_canonical_path");
        assert_eq!(h.worker().operations(), ["bind", "inspect"]);
        assert!(h.worker().removed().is_empty());
        assert!(!h.daemon.paused(h.id()));
        current.rm("build", true).await.unwrap();
        assert_eq!(h.worker().removed(), ["/workspace/build"]);
    }
}

#[tokio::test]
async fn recursive_removal_preserves_unrelated_writable_trees() {
    for target in ["build", "/external/build"] {
        let h = bound(&manifest()).await;
        h.session(manifest())
            .rm_as(target, true, Some(&User::new("developer")))
            .await
            .unwrap();
        let expected = if target == "build" {
            "/workspace/build"
        } else {
            target
        };
        assert_eq!(h.worker().removed(), [expected]);
        let calls = h.worker().calls.lock().unwrap().clone();
        assert_eq!(
            *calls.last().unwrap(),
            json!({
                "operation": "remove", "user": "developer",
                "max_entry_visits": 100_000, "max_cpu_seconds": 10,
            })
        );
        assert_eq!(h.events(), ["pause", "unpause", "pause", "unpause"]);
    }
}

#[tokio::test]
async fn removal_forwards_application_resource_limits() {
    let h = harness_of(1, DockerRemovalOptions::new(4, 3, 1));
    h.service.bind_new(h.id(), &manifest()).await.unwrap();
    h.session(manifest()).rm("build", true).await.unwrap();
    let calls = h.worker().calls.lock().unwrap().clone();
    assert_eq!(
        *calls.last().unwrap(),
        json!({
            "operation": "remove", "user": "1000:1000",
            "max_entry_visits": 3, "max_cpu_seconds": 1,
        })
    );
}

#[tokio::test]
async fn service_rejects_nonpositive_limits_before_connecting() {
    let daemon = Daemon::new();
    for limit in [
        "max_concurrent_removals",
        "max_entry_visits",
        "max_cpu_seconds",
    ] {
        let mut options = limits();
        match limit {
            "max_concurrent_removals" => options.max_concurrent_removals = 0,
            "max_entry_visits" => options.max_entry_visits = 0,
            _ => options.max_cpu_seconds = 0,
        }
        let host = Arc::new(FakeHost {
            daemon: daemon.clone(),
            cgroup: Mutex::new(None),
        });
        let error = DockerRemovalService::try_from_parts(
            options.clone(),
            daemon.clone(),
            host,
            Arc::new(QueueLauncher(Mutex::new(VecDeque::new()))),
        )
        .map(|_| ())
        .unwrap_err();
        assert!(
            error.message.contains("resource limits must be positive"),
            "{limit}"
        );
        // The public constructor checks the limits before it checks for root or looks for the
        // worker executable.
        let error = DockerRemovalService::new(options).map(|_| ()).unwrap_err();
        assert!(
            error.message.contains("resource limits must be positive"),
            "{limit}"
        );
    }
    assert!(
        daemon.calls().is_empty(),
        "no daemon call before the limits are valid"
    );
}

#[tokio::test]
async fn recursive_removal_preserves_live_binding_roots() {
    for (target, is_directory) in [
        (".", true),
        ("/workspace", true),
        ("/external", true),
        ("/external", false),
    ] {
        let configured = Manifest::new()
            .with_root("/workspace")
            .with_extra_path_grant(grant("/external"))
            .with_extra_path_grant(grant("/protected").read_only(true));
        let h = bound(&configured).await;
        h.worker().set_post(move |request, result| {
            let mut value = result?;
            if request["operation"] == "inspect" {
                value["is_directory"] = json!(is_directory);
            }
            Ok(value)
        });
        let current = h.session(configured.clone());
        let error = current
            .rm_as(target, true, Some(&User::new("0")))
            .await
            .unwrap_err();
        assert_eq!(
            reason(&error),
            "docker_removal_bound_root",
            "{target} {is_directory}"
        );
        assert!(h.worker().removed().is_empty());
        assert!(!h.daemon.paused(h.id()));
        current.rm("build", true).await.unwrap();
        assert_eq!(h.worker().removed(), ["/workspace/build"]);
    }
}

#[tokio::test]
async fn recursive_removal_rejects_alias_to_ancestor_of_workspace() {
    let configured = Manifest::new()
        .with_root("/external/group/workspace")
        .with_extra_path_grant(grant("/external"))
        .with_extra_path_grant(grant("/protected").read_only(true));
    let h = bound(&configured).await;
    h.worker()
        .alias("/external/group/workspace/link/tree", "/external/group");
    let current = h.session(configured);
    let error = current.rm("link/tree", true).await.unwrap_err();
    assert_eq!(reason(&error), "docker_removal_bound_root");
    assert!(h.worker().removed().is_empty());
    current.rm("build", true).await.unwrap();
    assert_eq!(h.worker().removed(), ["/external/group/workspace/build"]);
}

#[tokio::test]
async fn pause_cleanup_preserves_operation_failure() {
    for stage in ["state", "unpause"] {
        for operation in ["denied", "worker_failure", "success"] {
            let h = bound(&manifest()).await;
            let daemon = h.daemon.clone();
            let failing_stage = stage;
            // The first request after the bind arms the cleanup failure, as Python's request
            // wrapper does: only the thaw at the end is affected.
            h.worker().set_pre(move |_| {
                if failing_stage == "state" {
                    daemon.fail_inspect.store(true, Ordering::SeqCst);
                } else {
                    daemon.fail_unpause.store(true, Ordering::SeqCst);
                }
                None
            });
            if operation == "denied" {
                // A policy denial inside the pause: the target is a bound root.
                h.worker().alias("/workspace/build", "/external/protected");
            }
            if operation == "worker_failure" {
                let daemon = h.daemon.clone();
                h.worker().set_pre(move |_| {
                    if failing_stage == "state" {
                        daemon.fail_inspect.store(true, Ordering::SeqCst);
                    } else {
                        daemon.fail_unpause.store(true, Ordering::SeqCst);
                    }
                    Some(Err(WorkerFailure::Transport("worker failure".to_string())))
                });
            }
            let current = h.session(manifest());
            let error = current.rm("build", true).await.unwrap_err();
            match operation {
                "success" => {
                    // The removal completed; only the thaw failed, and that is what is reported.
                    assert!(error.message.contains("cleanup unavailable"), "{error}");
                    assert_eq!(h.worker().removed(), ["/workspace/build"]);
                }
                "denied" => {
                    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
                    assert_eq!(reason(&error), "docker_removal_bound_root");
                    assert!(h.worker().removed().is_empty());
                }
                _ => {
                    assert_eq!(reason(&error), "docker_removal_failed");
                    assert!(!error.message.contains("cleanup unavailable"), "{error}");
                    assert!(h.worker().removed().is_empty());
                }
            }
            assert!(h.daemon.paused(h.id()), "{stage} {operation}: left paused");
        }
    }
}

#[tokio::test]
async fn fixed_grant_alias_cannot_move_protection() {
    let h = bound(&manifest()).await;
    h.worker().alias("/grant-alias", "/unrelated");
    let error = h
        .session(manifest())
        .rm("/external", true)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert!(h.worker().removed().is_empty());
    assert!(!h.daemon.paused(h.id()));
}

#[tokio::test]
async fn target_alias_is_authorized_inside_pause() {
    let h = bound(&manifest()).await;
    h.worker().alias("/workspace/link/tree", "/external");
    let error = h
        .session(manifest())
        .rm("link/tree", true)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert!(h.worker().removed().is_empty());
}

fn listing_hook(entries: &'static str) -> impl Fn(&[String]) -> Option<SandboxResult<ExecResult>> {
    move |argv| {
        argv.join(" ")
            .ends_with("ls -la -- /workspace")
            .then(|| out(entries))
    }
}

const BUILD_LISTING: &str = "total 8\ndrwxr-xr-x 2 root root 4096 Jan  1 00:00 build\n";

#[tokio::test]
async fn pruned_restore_uses_real_rm_with_unrelated_grant() {
    let h = bound(&manifest()).await;
    *h.daemon.exec_hook.lock().unwrap() = Some(Box::new(listing_hook(BUILD_LISTING)));
    let current = h.session(manifest());
    clear_workspace_root_on_resume(&current).await.unwrap();
    assert_eq!(h.worker().removed(), ["/workspace/build"]);
}

#[tokio::test]
async fn transport_uncertainty_leaves_workload_paused() {
    let h = bound(&manifest()).await;
    let worker = h.worker().clone();
    h.worker().set_pre(move |_| {
        worker.uncertain.store(true, Ordering::SeqCst);
        Some(Err(WorkerFailure::Transport("lost worker".to_string())))
    });
    let error = h
        .service
        .remove(h.id(), &manifest(), "build", None)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert!(h.daemon.paused(h.id()));
    assert!(h.worker().removed().is_empty());
    let error = h
        .service
        .remove(h.id(), &manifest(), "build", None)
        .await
        .unwrap_err();
    assert!(error.message.contains("no longer usable"), "{error}");
}

#[tokio::test]
async fn windows_path_is_rejected_before_mutation() {
    let h = bound(&manifest()).await;
    let error = h
        .service
        .remove(h.id(), &manifest(), "C:/workspace/build", None)
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidManifestPath);
    assert!(h.worker().removed().is_empty());
}

#[tokio::test]
async fn relative_parent_segments_use_the_normalized_request() {
    let h = bound(&manifest()).await;
    h.session(manifest())
        .rm("link/../build", true)
        .await
        .unwrap();
    let calls = h.worker().calls.lock().unwrap().clone();
    assert_eq!(
        calls[calls.len() - 2],
        json!({
            "operation": "inspect",
            "path": "/workspace/build",
            "workspace_root": "/workspace",
        })
    );
    assert_eq!(h.worker().removed(), ["/workspace/build"]);
}

#[tokio::test]
async fn missing_target_still_checks_the_requested_user() {
    let h = bound(&manifest()).await;
    let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
    let log = seen.clone();
    h.worker().set_pre(move |request| {
        log.lock().unwrap().push(request.clone());
        Some(if request["operation"] == "inspect" {
            Ok(json!({"ok": true, "path": "", "is_directory": false}))
        } else {
            Err(WorkerFailure::Transport("PermissionError".to_string()))
        })
    });
    let error = h
        .session(manifest())
        .rm_as("private/missing", true, Some(&User::new("developer")))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::WorkspaceArchiveWriteError);
    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            json!({"operation": "inspect", "path": "/workspace/private/missing",
                   "workspace_root": "/workspace"}),
            json!({"operation": "remove", "user": "developer",
                   "max_entry_visits": 100_000, "max_cpu_seconds": 10}),
        ]
    );
    assert!(!h.daemon.paused(h.id()));
}

#[tokio::test]
async fn relative_removal_preserves_workspace_alias_traversal() {
    for accessible in [false, true] {
        let alias = "/private/workspace-alias";
        let h = harness();
        h.worker().alias(alias, "/workspace");
        h.worker()
            .alias(&format!("{alias}/build"), "/workspace/build");
        let configured = Manifest::new().with_root(alias);
        h.service.bind_new(h.id(), &configured).await.unwrap();
        let inspected = Arc::new(Mutex::new(None::<Value>));
        let log = inspected.clone();
        h.worker().set_pre(move |request| {
            match request["operation"].as_str() {
                Some("inspect") => *log.lock().unwrap() = Some(request.clone()),
                Some("remove") => {
                    assert_eq!(request["user"], "developer");
                    let path = log.lock().unwrap().as_ref().unwrap()["path"]
                        .as_str()
                        .unwrap()
                        .to_string();
                    if path.starts_with("/private/") && !accessible {
                        return Some(Err(WorkerFailure::Transport("PermissionError".to_string())));
                    }
                }
                _ => {}
            }
            None
        });
        let current = h.session(configured);
        let result = current
            .rm_as("build", true, Some(&User::new("developer")))
            .await;
        assert_eq!(result.is_ok(), accessible);
        assert_eq!(
            inspected.lock().unwrap().clone().unwrap(),
            json!({"operation": "inspect", "path": format!("{alias}/build"), "workspace_root": alias})
        );
        let expected: Vec<&str> = if accessible {
            vec!["/workspace/build"]
        } else {
            vec![]
        };
        assert_eq!(h.worker().removed(), expected);
    }
}

// --------------------------------------------------------------------------------------------
// Host eligibility
// --------------------------------------------------------------------------------------------

#[tokio::test]
async fn host_service_rejects_uncontrolled_execution_modes() {
    for host_configuration in [
        json!({"Privileged": true}),
        json!({"CapAdd": ["SYS_ADMIN"]}),
        json!({"SecurityOpt": ["seccomp=unconfined"]}),
        json!({"PidMode": "container:other"}),
        json!({"Runtime": "unverified"}),
        json!({"CapDrop": ["NET_RAW"]}),
        json!({"GroupAdd": ["docker"]}),
        json!({"UsernsMode": "private"}),
    ] {
        let h = harness();
        h.daemon
            .with(h.id(), |c| c.host_config = host_configuration.clone());
        let error = h.service.verify_container(h.id()).await.unwrap_err();
        assert!(
            error.message.contains("private container"),
            "{host_configuration}: {error}"
        );
    }
}

#[tokio::test]
async fn host_service_rejects_unsupported_daemons() {
    for (version, security, runtime) in [
        ("25.0.0", json!(["name=seccomp,profile=builtin"]), "runc"),
        ("26.0.0", json!(["name=seccomp,profile=custom"]), "runc"),
        (
            "26.0.0",
            json!(["name=seccomp,profile=builtin", "name=selinux"]),
            "runc",
        ),
        ("26.0.0", json!(["name=seccomp,profile=builtin"]), "crun"),
    ] {
        let h = harness();
        *h.daemon.version.lock().unwrap() = version.to_string();
        *h.daemon.info.lock().unwrap() =
            json!({"SecurityOptions": security, "DefaultRuntime": runtime});
        let error = h.service.verify_container(h.id()).await.unwrap_err();
        assert!(
            error.message.contains("Docker 26+"),
            "{version} {security} {runtime}: {error}"
        );
    }
}

#[tokio::test]
async fn host_service_verifies_local_pid_and_identity_user_maps() {
    let h = harness();
    assert_eq!(
        h.service.verify_container(h.id()).await.unwrap(),
        (123, "incarnation".to_string())
    );
    let bind = |field: &str, value: Value| {
        let mut mount = json!({"Type": "bind", "RW": false, "Propagation": "rprivate"});
        mount[field] = value;
        json!([mount])
    };
    h.daemon
        .with(h.id(), |c| c.mounts = bind("Type", json!("bind")));
    h.service
        .verify_container(h.id())
        .await
        .expect("a read-only private bind mount is fine");
    for (field, value) in [
        ("Type", json!("volume")),
        ("RW", json!(true)),
        ("Propagation", json!("rshared")),
    ] {
        h.daemon
            .with(h.id(), |c| c.mounts = bind(field, value.clone()));
        let error = h.service.verify_container(h.id()).await.unwrap_err();
        assert!(
            error.message.contains("private container"),
            "{field}: {error}"
        );
    }
    h.daemon.with(h.id(), |c| c.mounts = json!([]));
    *h.host.cgroup.lock().unwrap() = Some("0::/unrelated\n".to_string());
    let error = h.service.verify_container(h.id()).await.unwrap_err();
    assert!(
        error.message.contains("not on this service's host"),
        "{error}"
    );
}

#[tokio::test]
async fn client_rejects_a_service_connected_to_another_daemon() {
    let h = harness();
    let other = Daemon::new();
    let client = h.client().with_runner(other.clone());
    let error = client
        .create(Some(manifest()))
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(error.message.contains("connection"), "{error}");
    assert!(other.calls().is_empty() && h.daemon.calls().is_empty());
}

// --------------------------------------------------------------------------------------------
// Client lifecycle
// --------------------------------------------------------------------------------------------

#[tokio::test]
async fn read_only_host_mount_preserves_workspace_cleanup() {
    for restore in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().canonicalize().unwrap().join("toolchain");
        let source = source.to_str().unwrap();
        let configured = Manifest::new()
            .with_root("/workspace")
            .with_extra_path_grant(
                grant("/toolchain")
                    .with_host_path(source)
                    .unwrap()
                    .read_only(true),
            );
        let h = harness();
        h.daemon.with(h.id(), |c| {
            c.mounts = json!([{
                "Type": "bind", "Source": source, "Destination": "/toolchain",
                "RW": false, "Propagation": "rprivate",
            }]);
        });
        let wrapped = h
            .client()
            .create(Some(configured.clone()))
            .await
            .expect("create");
        if restore {
            *h.daemon.exec_hook.lock().unwrap() = Some(Box::new(listing_hook(BUILD_LISTING)));
            clear_workspace_root_on_resume(wrapped.as_ref())
                .await
                .unwrap();
        } else {
            wrapped.rm("build", true).await.unwrap();
        }
        assert_eq!(h.worker().removed(), ["/workspace/build"]);
        h.worker()
            .alias("/workspace/link/child", "/toolchain/child");
        assert!(wrapped.rm("link/child", true).await.is_err());
        assert_eq!(h.worker().removed(), ["/workspace/build"]);
        let changed = Manifest::new()
            .with_root("/workspace")
            .with_extra_path_grant(
                grant("/toolchain")
                    .with_host_path(format!("{source}-replaced"))
                    .unwrap()
                    .read_only(true),
            );
        let error = h.service.assert_bound(h.id(), &changed).await.unwrap_err();
        assert!(error.message.contains("original live authority"), "{error}");
    }
}

#[tokio::test]
async fn create_binds_before_returning_the_session() {
    for failure in [
        None,
        Some("writable_first"),
        Some("read_only_first"),
        Some("workspace"),
        Some("transport"),
        Some("close"),
    ] {
        let h = harness();
        let mut configured = manifest();
        match failure {
            Some("transport") | Some("close") => {
                let worker = h.worker().clone();
                h.worker().set_post(move |request, result| {
                    let value = result?;
                    if request["operation"] == "bind" {
                        worker.uncertain.store(true, Ordering::SeqCst);
                        return Err(WorkerFailure::Transport(
                            "worker transport failed".to_string(),
                        ));
                    }
                    Ok(value)
                });
                if failure == Some("close") {
                    h.worker().close_error.store(true, Ordering::SeqCst);
                }
            }
            Some(kind) => {
                h.worker().alias(
                    "/grant-alias",
                    if kind == "workspace" {
                        "/workspace"
                    } else {
                        "/external"
                    },
                );
                if kind == "read_only_first" {
                    configured = Manifest::new()
                        .with_root("/workspace")
                        .with_extra_path_grant(grant("/grant-alias").read_only(true))
                        .with_extra_path_grant(grant("/external"));
                }
            }
            None => {}
        }
        let result = h.client().create(Some(configured)).await;
        if let Some(kind) = failure {
            let error = result.map(|_| ()).unwrap_err();
            assert!(!h.service.is_bound(h.id()), "{kind}");
            assert_eq!(h.worker().operations(), ["bind"], "{kind}");
            assert!(h.worker().removed().is_empty());
            if kind == "transport" || kind == "close" {
                assert!(error.message.contains("worker transport failed"), "{error}");
                let expected: &[&str] = if kind == "transport" {
                    &["start", "pause", "close"]
                } else {
                    &["start", "pause"]
                };
                assert_eq!(h.events(), expected, "{kind}");
                // The container is removed while still paused: a lost outcome never thaws it.
                assert_eq!(h.daemon.with(h.id(), |c| c.removed_pauses.clone()), [true]);
            } else {
                assert!(
                    error
                        .message
                        .contains("distinct canonical workspace and grant roots"),
                    "{kind}: {error}"
                );
                assert_eq!(h.events(), ["start", "pause", "close", "unpause"], "{kind}");
                assert_eq!(h.daemon.with(h.id(), |c| c.removed_pauses.clone()), [false]);
            }
            if kind == "close" {
                assert_eq!(h.worker().close_calls.load(Ordering::SeqCst), 1);
            }
            continue;
        }
        let wrapped = result.expect("create");
        wrapped.rm("build", true).await.unwrap();
        assert_eq!(h.events()[..3], ["start", "pause", "unpause"]);
        assert_eq!(h.worker().removed(), ["/workspace/build"]);
    }
}

#[tokio::test]
async fn delete_releases_authority_only_after_confirmed_container_removal() {
    for (behavior, label) in [
        (RmBehavior::Removed, "removed"),
        (RmBehavior::NotFound, "missing"),
        (RmBehavior::Error, "error"),
    ] {
        let h = harness();
        let client = h.client();
        let wrapped = client.create(Some(manifest())).await.expect("create");
        *h.daemon.rm.lock().unwrap() = behavior;
        let result = client.delete(wrapped).await;
        if behavior == RmBehavior::Error {
            assert!(result.is_err(), "{label}");
            assert!(h.service.is_bound(h.id()), "{label}");
            assert!(!h.events().contains(&"close".to_string()), "{label}");
        } else {
            result.unwrap_or_else(|error| panic!("{label}: {error}"));
            assert!(!h.service.is_bound(h.id()), "{label}");
            assert_eq!(
                h.events().iter().filter(|event| *event == "close").count(),
                1,
                "{label}"
            );
        }
        assert!(h.worker().removed().is_empty());
    }
}

#[tokio::test]
async fn service_close_attempts_all_workers_after_a_worker_failure() {
    let h = harness_of(2, limits());
    h.service.bind_new(&h.ids[0], &manifest()).await.unwrap();
    h.service.bind_new(&h.ids[1], &manifest()).await.unwrap();
    h.workers[0].close_error.store(true, Ordering::SeqCst);
    h.workers[1].close_error.store(false, Ordering::SeqCst);
    let error = h.service.close().await.unwrap_err();
    assert!(error.message.contains("cleanup failed"), "{error}");
    assert_eq!(h.workers[0].close_calls.load(Ordering::SeqCst), 1);
    assert_eq!(h.workers[1].close_calls.load(Ordering::SeqCst), 1);
    assert!(!h.service.is_bound(&h.ids[0]) && !h.service.is_bound(&h.ids[1]));
    // A closed service accepts no new authority.
    let error = h
        .service
        .bind_new(&h.ids[0], &manifest())
        .await
        .unwrap_err();
    assert!(error.message.contains("closed"), "{error}");
}

// --------------------------------------------------------------------------------------------
// Concurrency and cancellation
// --------------------------------------------------------------------------------------------

#[tokio::test]
async fn caller_cancellation_never_leaves_the_removal_half_done() {
    for outcome in ["success", "failure"] {
        let h = bound(&manifest()).await;
        let (entered, release) = h.worker().gate_remove();
        if outcome == "failure" {
            h.worker().set_pre(|request| {
                (request["operation"] == "remove")
                    .then(|| Err(WorkerFailure::Transport("worker failed".to_string())))
            });
        }
        let current = Arc::new(h.session(manifest()));
        let task = tokio::spawn({
            let current = current.clone();
            async move { current.rm("build", true).await }
        });
        entered.notified().await;
        task.abort();
        tokio::task::yield_now().await;
        assert!(h.daemon.paused(h.id()), "the removal is still running");
        release.notify_one();
        let daemon = h.daemon.clone();
        let id = h.id().to_string();
        eventually(move || !daemon.paused(&id)).await;
        let expected: Vec<&str> = if outcome == "success" {
            vec!["/workspace/build"]
        } else {
            vec![]
        };
        assert_eq!(h.worker().removed(), expected, "{outcome}");
    }
}

#[tokio::test]
async fn service_bounds_concurrent_removal_without_pausing_rejected_work() {
    for outcome in ["success", "failure", "uncertain"] {
        let h = harness_of(2, DockerRemovalOptions::new(1, 100_000, 10));
        h.service.bind_new(&h.ids[0], &manifest()).await.unwrap();
        h.service.bind_new(&h.ids[1], &manifest()).await.unwrap();
        let (entered, release) = h.workers[0].gate_remove();
        if outcome != "success" {
            let worker = h.workers[0].clone();
            let uncertain = outcome == "uncertain";
            h.workers[0].set_pre(move |request| {
                if request["operation"] != "remove" {
                    return None;
                }
                worker.uncertain.store(uncertain, Ordering::SeqCst);
                Some(Err(WorkerFailure::Transport("worker failure".to_string())))
            });
        }
        let first = tokio::spawn({
            let session = h.session_for(0, manifest());
            async move { session.rm("build", true).await }
        });
        entered.notified().await;
        let events = h.daemon.events(&h.ids[1]);
        let second = h.session_for(1, manifest());
        let error = second.rm("build", true).await.unwrap_err();
        assert_eq!(reason(&error), "docker_removal_capacity", "{outcome}");
        assert!(h.workers[1].removed().is_empty());
        assert_eq!(
            h.daemon.events(&h.ids[1]),
            events,
            "a rejected removal pauses nothing"
        );
        assert!(!first.is_finished());
        release.notify_one();
        let result = first.await.unwrap();
        assert_eq!(result.is_ok(), outcome == "success", "{outcome}");
        if outcome == "uncertain" {
            assert!(h.daemon.paused(&h.ids[0]));
            h.service.release(&h.ids[0]).await.unwrap();
            let error = second.rm("build", true).await.unwrap_err();
            assert_eq!(
                reason(&error),
                "docker_removal_capacity",
                "the slot stays consumed"
            );
        } else {
            second.rm("build", true).await.unwrap();
            assert_eq!(h.workers[1].removed(), ["/workspace/build"]);
            assert!(!h.daemon.paused(&h.ids[0]));
        }
    }
}

#[tokio::test]
async fn unrelated_container_progresses_during_a_blocked_removal() {
    for operation in ["rm", "delete"] {
        let h = harness_of(2, limits());
        h.service.bind_new(&h.ids[0], &manifest()).await.unwrap();
        h.service.bind_new(&h.ids[1], &manifest()).await.unwrap();
        let (entered, release) = h.workers[0].gate_remove();
        let first = tokio::spawn({
            let session = h.session_for(0, manifest());
            async move { session.rm("build", true).await }
        });
        entered.notified().await;
        let second = h.session_for(1, manifest());
        let progressed = tokio::time::timeout(Duration::from_secs(5), async {
            if operation == "rm" {
                second.rm("build", true).await
            } else {
                h.client().delete(Arc::new(second)).await
            }
        })
        .await
        .expect("the unrelated container is not blocked");
        progressed.unwrap();
        assert!(!first.is_finished());
        release.notify_one();
        first.await.unwrap().unwrap();
        assert_eq!(h.workers[0].removed(), ["/workspace/build"]);
        assert!(!h.daemon.paused(&h.ids[0]));
        if operation == "rm" {
            assert_eq!(h.workers[1].removed(), ["/workspace/build"]);
        } else {
            assert!(!h.service.is_bound(&h.ids[1]));
            let closed = h
                .daemon
                .events(&h.ids[1])
                .iter()
                .filter(|e| *e == "close")
                .count();
            assert_eq!(closed, 1);
        }
    }
}

#[tokio::test]
async fn delete_waits_for_worker_cleanup_even_when_the_caller_is_cancelled() {
    let h = bound(&manifest()).await;
    let client = h.client();
    let entered = {
        let gate = Gate {
            entered: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let handles = (gate.entered.clone(), gate.release.clone());
        *h.worker().close_gate.lock().unwrap() = Some(gate);
        handles
    };
    let session: Arc<dyn SandboxSession> = Arc::new(h.session(manifest()));
    let task = tokio::spawn(async move { client.delete(session).await });
    entered.0.notified().await;
    assert!(!task.is_finished());
    task.abort();
    tokio::task::yield_now().await;
    assert_eq!(h.worker().close_calls.load(Ordering::SeqCst), 1);
    assert!(
        !h.events().contains(&"close".to_string()),
        "the worker is still closing"
    );
    entered.1.notify_one();
    let daemon = h.daemon.clone();
    let id = h.id().to_string();
    eventually(move || daemon.events(&id).contains(&"close".to_string())).await;
    assert!(!h.service.is_bound(h.id()));
}
