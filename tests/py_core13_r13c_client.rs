//! Ports of `tests/sandbox/test_docker_removal_client.py`: the Docker client's lifecycle with a
//! removal service, over a simulated daemon and the real filesystem binding.
//!
//! The daemon double executes the workspace bootstrap (`mkdir -p`) and `touch` against the host
//! file system, and the worker double binds the real directories with the worker's own
//! `bind_paths` (with the root device pinned to the temp dir's, since a temp dir need not share
//! `/`'s device). The last group drives the real worker transport through a script standing in
//! for the `docker-removal-worker` executable.
#![cfg(all(feature = "docker-removal", unix))]

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::sandboxes::docker::{DockerRunner, DockerSandboxClient};
use openai_agents_sdk::sandbox::sandboxes::docker_removal::{
    DockerRemovalOptions, DockerRemovalService, ProcessLauncher, RemovalHost, RemovalWorker,
    WorkerFailure, WorkerLauncher,
};
use openai_agents_sdk::sandbox::sandboxes::docker_removal_worker::{
    Bindings, FileStat, HostFs, WorkerError, WorkerFs, bind_paths, remove, selected_path,
};
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxClient, SandboxError, SandboxResult, SandboxSession,
    clear_workspace_root_on_resume,
};
use serde_json::{Value, json};

const CONTAINER: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn out(stdout: &str) -> SandboxResult<ExecResult> {
    Ok(ExecResult {
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
        exit_code: 0,
    })
}

fn exit(code: i32) -> SandboxResult<ExecResult> {
    Ok(ExecResult {
        stdout: Vec::new(),
        stderr: b"failed".to_vec(),
        exit_code: code,
    })
}

fn grant(path: &str) -> SandboxPathGrant {
    SandboxPathGrant::new(path).expect("grant path")
}

// --------------------------------------------------------------------------------------------
// The daemon double
// --------------------------------------------------------------------------------------------

#[derive(Debug)]
struct Daemon {
    paused: AtomicBool,
    mounts: Mutex<Value>,
    bootstrap_exit: Mutex<i32>,
    owners: Mutex<BTreeMap<PathBuf, String>>,
    calls: Mutex<Vec<Vec<String>>>,
    removed_pauses: Mutex<Vec<bool>>,
    events: Mutex<Vec<String>>,
    listing: Mutex<Option<(String, String)>>,
}

impl Daemon {
    fn new() -> Arc<Self> {
        Arc::new(Daemon {
            paused: AtomicBool::new(false),
            mounts: Mutex::new(json!([])),
            bootstrap_exit: Mutex::new(0),
            owners: Mutex::new(BTreeMap::new()),
            calls: Mutex::new(Vec::new()),
            removed_pauses: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
            listing: Mutex::new(None),
        })
    }

    fn calls(&self) -> Vec<Vec<String>> {
        self.calls.lock().unwrap().clone()
    }

    fn execs(&self) -> usize {
        self.calls().iter().filter(|call| call[0] == "exec").count()
    }
}

#[async_trait]
impl DockerRunner for Daemon {
    async fn run(&self, argv: &[String], _stdin: Option<&[u8]>) -> SandboxResult<ExecResult> {
        self.calls.lock().unwrap().push(argv.to_vec());
        let parts: Vec<&str> = argv.iter().map(String::as_str).collect();
        match parts.as_slice() {
            ["container", "inspect", id] if *id != CONTAINER => exit(1),
            ["container", "inspect", id] => out(&json!([{
                "Id": id,
                "State": {"Running": true, "Paused": self.paused.load(Ordering::SeqCst),
                          "Pid": 123, "StartedAt": "incarnation", "Restarting": false},
                "HostConfig": {},
                "Config": {"User": "1000:1000"},
                "Mounts": self.mounts.lock().unwrap().clone(),
            }])
            .to_string()),
            ["info", ..] => out(&json!({
                "SecurityOptions": ["name=seccomp,profile=builtin"],
                "DefaultRuntime": "runc",
            })
            .to_string()),
            ["version", ..] => out("26.0.0"),
            ["pause", _] => {
                self.events.lock().unwrap().push("pause".to_string());
                self.paused.store(true, Ordering::SeqCst);
                out("")
            }
            ["unpause", _] => {
                self.events.lock().unwrap().push("unpause".to_string());
                self.paused.store(false, Ordering::SeqCst);
                out("")
            }
            ["start", _] => out(""),
            ["create", ..] => out(&format!("{CONTAINER}\n")),
            ["rm", "-f", _] => {
                self.removed_pauses
                    .lock()
                    .unwrap()
                    .push(self.paused.load(Ordering::SeqCst));
                out("")
            }
            ["exec", ..] => {
                let position = argv
                    .iter()
                    .position(|part| part == CONTAINER)
                    .expect("container");
                let command = &argv[position + 1..];
                let user = argv
                    .iter()
                    .position(|part| part == "-u")
                    .map(|index| argv[index + 1].clone())
                    .unwrap_or_else(|| "1000:1000".to_string());
                let cmd: Vec<&str> = command.iter().map(String::as_str).collect();
                match cmd.as_slice() {
                    ["mkdir", "-p", "--", target] => {
                        let code = *self.bootstrap_exit.lock().unwrap();
                        if code != 0 {
                            return exit(code);
                        }
                        let target = Path::new(target);
                        if !target.exists() {
                            std::fs::create_dir_all(target).unwrap();
                            self.owners
                                .lock()
                                .unwrap()
                                .insert(target.to_path_buf(), user);
                        }
                        out("")
                    }
                    ["touch", target] => {
                        let target = Path::new(target);
                        // A daemon-created 0755 workspace is not writable by the image user.
                        let owner = self
                            .owners
                            .lock()
                            .unwrap()
                            .get(target.parent().unwrap())
                            .cloned()
                            .unwrap_or_else(|| user.clone());
                        if owner != user {
                            return exit(1);
                        }
                        std::fs::write(target, "").unwrap();
                        out("")
                    }
                    _ => {
                        let joined = command.join(" ");
                        if let Some((root, text)) = self.listing.lock().unwrap().as_ref()
                            && joined.ends_with(&format!("ls -la -- {root}"))
                        {
                            return out(text);
                        }
                        out("ok\n")
                    }
                }
            }
            _ => out(""),
        }
    }
}

struct FakeHost;

impl RemovalHost for FakeHost {
    fn read_proc(&self, _pid: u32, name: &str) -> io::Result<String> {
        Ok(match name {
            "cgroup" => format!("0::/system.slice/docker-{CONTAINER}.scope\n"),
            _ => "0 0 4294967295\n".to_string(),
        })
    }
}

/// The worker's filesystem with the root device pinned to the temp dir's own.
struct TempRooted {
    inner: HostFs,
    device: u64,
}

impl WorkerFs for TempRooted {
    type Fd = <HostFs as WorkerFs>::Fd;

    fn canonical(&self, path: &str) -> io::Result<String> {
        self.inner.canonical(path)
    }

    fn root_device(&self) -> io::Result<u64> {
        Ok(self.device)
    }

    fn lstat(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<FileStat> {
        self.inner.lstat(name, parent)
    }

    fn fstat(&self, fd: &Self::Fd) -> io::Result<FileStat> {
        self.inner.fstat(fd)
    }

    fn open_path(&self, path: &str) -> io::Result<Self::Fd> {
        self.inner.open_path(path)
    }

    fn open_dir(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<Self::Fd> {
        self.inner.open_dir(name, parent)
    }

    fn rmdir(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<()> {
        self.inner.rmdir(name, parent)
    }

    fn unlink(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<()> {
        self.inner.unlink(name, parent)
    }

    fn scan(
        &self,
        dir: &Self::Fd,
        visit: &mut dyn FnMut(&OsStr) -> Result<bool, WorkerError>,
    ) -> Result<(), WorkerError> {
        self.inner.scan(dir, visit)
    }
}

struct Record {
    operations: Mutex<Vec<String>>,
    removed: Mutex<Vec<String>>,
    closed: Mutex<usize>,
}

struct BindingWorker {
    fs: TempRooted,
    bindings: Option<Bindings<TempRooted>>,
    selected: String,
    record: Arc<Record>,
    really_remove: bool,
}

fn failure(error: WorkerError) -> WorkerFailure {
    WorkerFailure::Os {
        errno: error.errno(),
        reason: error.reason().to_string(),
    }
}

#[async_trait]
impl RemovalWorker for BindingWorker {
    async fn request(&mut self, request: Value) -> Result<Value, WorkerFailure> {
        let operation = request["operation"].as_str().unwrap().to_string();
        self.record
            .operations
            .lock()
            .unwrap()
            .push(operation.clone());
        match operation.as_str() {
            "bind" => {
                let paths: Vec<String> = request["paths"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|path| path.as_str().unwrap().to_string())
                    .collect();
                let bound = bind_paths(&self.fs, &paths).map_err(failure)?;
                let response = json!({"ok": true, "paths": bound.paths});
                self.bindings = Some(bound);
                Ok(response)
            }
            "inspect" => {
                // The worker re-checks every pinned root before it selects a target.
                if let Some(bound) = &self.bindings {
                    bound.validate(&self.fs).map_err(failure)?;
                }
                let (path, is_directory) =
                    selected_path(&self.fs, request["path"].as_str().unwrap()).map_err(failure)?;
                self.selected = path.clone();
                Ok(json!({"ok": true, "path": path, "is_directory": is_directory}))
            }
            _ => {
                if self.really_remove {
                    remove(&self.fs, &self.selected, 1000).map_err(failure)?;
                }
                self.record
                    .removed
                    .lock()
                    .unwrap()
                    .push(self.selected.clone());
                Ok(json!({"ok": true}))
            }
        }
    }

    fn uncertain(&self) -> bool {
        false
    }

    async fn close(&mut self) -> io::Result<()> {
        *self.record.closed.lock().unwrap() += 1;
        Ok(())
    }
}

struct BindingLauncher {
    device: u64,
    record: Arc<Record>,
    really_remove: bool,
}

#[async_trait]
impl WorkerLauncher for BindingLauncher {
    async fn launch(&self, _pid: u32) -> Result<Box<dyn RemovalWorker>, WorkerFailure> {
        Ok(Box::new(BindingWorker {
            fs: TempRooted {
                inner: HostFs::new(),
                device: self.device,
            },
            bindings: None,
            selected: String::new(),
            record: self.record.clone(),
            really_remove: self.really_remove,
        }))
    }
}

struct Setup {
    daemon: Arc<Daemon>,
    service: DockerRemovalService,
    record: Arc<Record>,
    temp: tempfile::TempDir,
    base: PathBuf,
}

fn setup() -> Setup {
    setup_with(false)
}

fn setup_with(really_remove: bool) -> Setup {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let device = HostFs::new()
        .lstat(OsStr::new(base.to_str().unwrap()), None)
        .unwrap()
        .dev;
    let daemon = Daemon::new();
    let record = Arc::new(Record {
        operations: Mutex::new(Vec::new()),
        removed: Mutex::new(Vec::new()),
        closed: Mutex::new(0),
    });
    let service = DockerRemovalService::from_parts(
        DockerRemovalOptions::new(4, 100_000, 10),
        daemon.clone(),
        Arc::new(FakeHost),
        Arc::new(BindingLauncher {
            device,
            record: record.clone(),
            really_remove,
        }),
    );
    Setup {
        daemon,
        service,
        record,
        temp,
        base,
    }
}

impl Setup {
    fn client(&self) -> DockerSandboxClient {
        DockerSandboxClient::new().with_removal_service(self.service.clone())
    }

    /// `create`, or a `resume` of a state whose container is gone (so a replacement is made).
    async fn start(
        &self,
        manifest: &Manifest,
        resume: bool,
    ) -> SandboxResult<Arc<dyn SandboxSession>> {
        let client = self.client();
        if resume {
            let state = SandboxSessionState::new("docker", "sess-1", manifest.root(), manifest)
                .with_provider_id(Some("missing-container".to_string()))
                .with_workspace_root_ready(true);
            // The daemon double knows only the one container; "missing-container" is not it.
            return client.resume(&state, manifest.clone()).await;
        }
        client.create(Some(manifest.clone())).await
    }
}

#[tokio::test]
async fn client_bootstraps_workspace_before_strict_binding() {
    for resume in [false, true] {
        for workspace_setup in ["missing", "existing", "ancestor_grant"] {
            let s = setup();
            let root = s.base.join("nested/workspace");
            if workspace_setup == "existing" {
                std::fs::create_dir_all(&root).unwrap();
                std::fs::write(root.join("keep.txt"), "existing contents").unwrap();
            }
            let mut configured = Manifest::new().with_root(root.to_str().unwrap());
            if workspace_setup == "ancestor_grant" {
                // The grant root is the workspace's parent, which the bootstrap creates.
                std::fs::create_dir_all(&root).unwrap();
                configured = configured.with_extra_path_grant(
                    grant(root.parent().unwrap().to_str().unwrap()).read_only(true),
                );
            }
            let wrapped = s
                .start(&configured, resume)
                .await
                .unwrap_or_else(|error| panic!("{resume} {workspace_setup}: {error}"));
            assert!(root.is_dir());
            s.service
                .assert_bound(CONTAINER, &configured)
                .await
                .unwrap();
            let created = root.join("created.txt");
            let result = wrapped
                .exec(ExecRequest::argv(["touch", created.to_str().unwrap()]))
                .await
                .unwrap();
            assert!(result.ok(), "{resume} {workspace_setup}");
            assert!(created.is_file());
            if workspace_setup == "existing" {
                assert_eq!(
                    std::fs::read_to_string(root.join("keep.txt")).unwrap(),
                    "existing contents"
                );
            }
            let refused = wrapped.rm(root.to_str().unwrap(), true).await.unwrap_err();
            assert_eq!(refused.context["reason"], "docker_removal_bound_root");
            if workspace_setup == "ancestor_grant" {
                let refused = wrapped
                    .rm(root.parent().unwrap().to_str().unwrap(), true)
                    .await
                    .unwrap_err();
                assert_eq!(refused.context["reason"], "docker_removal_bound_root");
            }
            assert!(s.record.removed.lock().unwrap().is_empty());
            drop(s.temp);
        }
    }
}

#[tokio::test]
async fn client_bootstrap_does_not_create_missing_grant_roots() {
    for resume in [false, true] {
        let s = setup();
        let root = s.base.join("workspace");
        let grant_root = s.base.join("external");
        let configured = Manifest::new()
            .with_root(root.to_str().unwrap())
            .with_extra_path_grant(grant(grant_root.to_str().unwrap()));
        let error = s.start(&configured, resume).await.map(|_| ()).unwrap_err();
        assert!(error.message.contains("FileNotFoundError"), "{error}");
        assert!(root.is_dir());
        assert!(!grant_root.exists());
        assert!(!s.service.is_bound(CONTAINER));
        assert_eq!(*s.daemon.removed_pauses.lock().unwrap(), [false]);
        assert_eq!(*s.record.closed.lock().unwrap(), 1);
    }
}

#[tokio::test]
async fn failed_workspace_bootstrap_cleans_up_before_binding() {
    for resume in [false, true] {
        let s = setup();
        *s.daemon.bootstrap_exit.lock().unwrap() = 1;
        let configured = Manifest::new().with_root(s.base.join("workspace").to_str().unwrap());
        let error = s.start(&configured, resume).await.map(|_| ()).unwrap_err();
        assert!(
            error.message.contains("Unable to create Docker workspace"),
            "{error}"
        );
        assert!(!s.service.is_bound(CONTAINER));
        assert!(
            s.record.operations.lock().unwrap().is_empty(),
            "no worker was ever asked"
        );
        assert_eq!(s.daemon.removed_pauses.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn ineligible_mount_is_rejected_before_bootstrap_writes() {
    for resume in [false, true] {
        for mount_kind in ["host_grant", "image_volume"] {
            let s = setup();
            let shared = s.base.join("shared");
            std::fs::create_dir_all(&shared).unwrap();
            std::fs::write(shared.join("keep.txt"), "host contents").unwrap();
            let alias = s.base.join("image-workspace");
            std::os::unix::fs::symlink(&shared, &alias).unwrap();
            let mut configured = Manifest::new().with_root(alias.join("build").to_str().unwrap());
            if mount_kind == "host_grant" {
                configured = configured.with_extra_path_grant(
                    grant(shared.to_str().unwrap())
                        .with_host_path(shared.to_str().unwrap())
                        .unwrap(),
                );
            }
            *s.daemon.mounts.lock().unwrap() = json!([{
                "Type": if mount_kind == "host_grant" { "bind" } else { "volume" },
                "Destination": shared.to_str().unwrap(),
                "Source": shared.to_str().unwrap(),
                "RW": true,
                "Propagation": "rprivate",
            }]);
            let error = s.start(&configured, resume).await.map(|_| ()).unwrap_err();
            assert!(
                error.message.contains("shared host paths")
                    || error.message.contains("private container"),
                "{mount_kind}: {error}"
            );
            let names: Vec<String> = std::fs::read_dir(&shared)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect();
            assert_eq!(
                names,
                ["keep.txt"],
                "nothing was written through the image alias"
            );
            assert_eq!(
                std::fs::read_to_string(shared.join("keep.txt")).unwrap(),
                "host contents"
            );
            assert_eq!(
                s.daemon.execs(),
                0,
                "{mount_kind}: no exec before the layout is accepted"
            );
            assert!(s.record.operations.lock().unwrap().is_empty());
            assert!(!s.service.is_bound(CONTAINER));
            assert_eq!(s.daemon.removed_pauses.lock().unwrap().len(), 1);
        }
    }
}

#[tokio::test]
async fn workspace_overridden_grant_does_not_invalidate_later_removal() {
    for restore in [false, true] {
        let s = setup_with(true);
        let workspace = s.base.join("workspace");
        let cache = workspace.join("cache");
        let artifacts = workspace.join("build-output");
        let protected = s.base.join("external/protected");
        for directory in [&cache, &artifacts, &protected] {
            std::fs::create_dir_all(directory).unwrap();
            std::fs::write(directory.join("data.txt"), "x").unwrap();
        }
        let configured = Manifest::new()
            .with_root(workspace.to_str().unwrap())
            .with_extra_path_grant(grant(cache.to_str().unwrap()).read_only(true))
            .with_extra_path_grant(grant(protected.to_str().unwrap()).read_only(true));
        let wrapped = s.start(&configured, false).await.expect("create");
        if restore {
            let root = workspace.to_str().unwrap().to_string();
            let text = "total 8\n\
                 drwxr-xr-x 2 root root 4096 Jan  1 00:00 cache\n\
                 drwxr-xr-x 2 root root 4096 Jan  1 00:00 build-output\n"
                .to_string();
            *s.daemon.listing.lock().unwrap() = Some((root, text));
            clear_workspace_root_on_resume(wrapped.as_ref())
                .await
                .unwrap();
        } else {
            wrapped.rm("cache", true).await.unwrap();
            wrapped.rm("build-output", true).await.unwrap();
        }
        // A read-only grant inside the workspace is an ordinary entry of it, and removing it
        // later does not look like a replaced root.
        assert_eq!(
            *s.record.removed.lock().unwrap(),
            [cache.to_str().unwrap(), artifacts.to_str().unwrap()]
        );
        assert!(!cache.exists() && !artifacts.exists());
        assert!(workspace.is_dir() && protected.join("data.txt").is_file());

        // Effective external protection still fails closed if its identity changes.
        std::fs::remove_dir_all(&protected).unwrap();
        std::fs::create_dir_all(&protected).unwrap();
        assert!(wrapped.rm("another-build", true).await.is_err());
        assert_eq!(s.record.removed.lock().unwrap().len(), 2);
    }
}

#[tokio::test]
async fn a_bound_container_is_reattached_only_through_the_same_live_binding() {
    let s = setup();
    let root = s.base.join("workspace");
    let configured = Manifest::new().with_root(root.to_str().unwrap());
    let wrapped = s.start(&configured, false).await.expect("create");
    // The persisted state names the live container: resume reattaches through the binding.
    let state = wrapped.session_state();
    let client = s.client();
    let resumed = client
        .resume(&state, configured.clone())
        .await
        .expect("resume");
    assert_eq!(resumed.manifest().root(), configured.root());
    // A new service holds no such binding, so it cannot reattach the same container.
    let replacement = DockerRemovalService::from_parts(
        DockerRemovalOptions::new(4, 100_000, 10),
        s.daemon.clone(),
        Arc::new(FakeHost),
        Arc::new(BindingLauncher {
            device: 0,
            record: s.record.clone(),
            really_remove: false,
        }),
    );
    let error = DockerSandboxClient::new()
        .with_removal_service(replacement)
        .resume(&state, configured)
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(error.message.contains("original live authority"), "{error}");
}

// --------------------------------------------------------------------------------------------
// The real worker transport, against a script standing in for `docker-removal-worker`
// --------------------------------------------------------------------------------------------

fn write_script(directory: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = directory.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// Launch with a retry: executing a script another thread just wrote can race a `fork` that
/// inherited the write handle (`ETXTBSY`).
async fn launch(launcher: &ProcessLauncher) -> Box<dyn RemovalWorker> {
    for _ in 0..50 {
        match launcher.launch(123).await {
            Ok(worker) => return worker,
            Err(WorkerFailure::Os {
                errno: Some(26), ..
            }) => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Err(other) => panic!("launch failed: {other:?}"),
        }
    }
    panic!("the worker script stayed busy");
}

const WORKER_LOOP: &str = r#"
if [ "$1" = worker ]; then
  while IFS= read -r line; do
    case "$line" in
      *'"bind"'*) echo '{"ok":true,"paths":["/workspace"]}';;
      *'"inspect"'*) echo '{"ok":true,"path":"/workspace/build","is_directory":true}';;
      *'"close"'*) exit 0;;
    esac
  done
  exit 0
fi
"#;

fn removal_request() -> Value {
    json!({"operation": "remove", "user": "0", "max_entry_visits": 1, "max_cpu_seconds": 1})
}

#[tokio::test]
async fn worker_transport_eof_marks_outcome_uncertain() {
    let temp = tempfile::tempdir().unwrap();
    let script = write_script(temp.path(), "worker-exits", "exit 0");
    let mut worker = launch(&ProcessLauncher::new(script)).await;
    let error = worker
        .request(json!({"operation": "inspect", "path": "/workspace/build"}))
        .await
        .unwrap_err();
    assert_eq!(
        error,
        WorkerFailure::Transport("Docker removal worker transport failed".to_string())
    );
    assert!(worker.uncertain());
    // Closing reaps the exited process and never hangs.
    worker.close().await.unwrap();
}

#[tokio::test]
async fn process_worker_speaks_the_json_line_protocol_and_runs_removals_as_a_child() {
    let temp = tempfile::tempdir().unwrap();
    let record = temp.path().join("remove-request.json");
    let environment = temp.path().join("remove-environment.txt");
    let script = write_script(
        temp.path(),
        "worker",
        &format!(
            r#"{WORKER_LOOP}
if [ "$1" = remove ] && [ "$2" = 123 ]; then
  cat > '{record}'
  echo "$LC_ALL:${{HOME:-unset}}" > '{environment}'
  echo '{{"ok":true}}'
  exit 0
fi
exit 3"#,
            record = record.display(),
            environment = environment.display(),
        ),
    );
    let mut worker = launch(&ProcessLauncher::new(script)).await;
    // `remove` before any `inspect` is refused: there is nothing selected to remove.
    let early = worker.request(removal_request()).await.unwrap_err();
    assert_eq!(
        early,
        WorkerFailure::Os {
            errno: None,
            reason: "ValueError".to_string()
        }
    );
    assert_eq!(
        worker
            .request(json!({"operation": "bind", "paths": ["/workspace"]}))
            .await
            .unwrap(),
        json!({"ok": true, "paths": ["/workspace"]})
    );
    assert_eq!(
        worker
            .request(json!({"operation": "inspect", "path": "/workspace/link/build", "workspace_root": null}))
            .await
            .unwrap()["path"],
        "/workspace/build"
    );
    worker
        .request(json!({"operation": "remove", "user": "developer",
                        "max_entry_visits": 7, "max_cpu_seconds": 3}))
        .await
        .unwrap();
    // The child is handed the ORIGINAL requested path, not the canonical one.
    let sent: Value = serde_json::from_str(&std::fs::read_to_string(&record).unwrap()).unwrap();
    assert_eq!(
        sent,
        json!({"path": "/workspace/link/build", "user": "developer",
               "max_entry_visits": 7, "max_cpu_seconds": 3})
    );
    // Host workers inherit no environment beyond a fixed locale.
    assert_eq!(
        std::fs::read_to_string(&environment).unwrap().trim(),
        "C.UTF-8:unset"
    );
    assert!(!worker.uncertain());
    // A removal is consumed: a second one needs a fresh inspect.
    let again = worker.request(removal_request()).await.unwrap_err();
    assert_eq!(
        again,
        WorkerFailure::Os {
            errno: None,
            reason: "ValueError".to_string()
        }
    );
    worker.close().await.unwrap();
}

#[tokio::test]
async fn child_failure_reaches_structured_session_error() {
    for (reason_name, errno) in [
        ("PermissionError", Some(13)),
        ("OSError", Some(7)),
        ("ValueError", None),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let outcome = json!({"ok": false, "reason": reason_name, "errno": errno}).to_string();
        let script = write_script(
            temp.path(),
            "worker",
            &format!(
                r#"{WORKER_LOOP}
if [ "$1" = remove ]; then cat > /dev/null; echo '{outcome}'; exit 0; fi
exit 3"#
            ),
        );
        let daemon = Daemon::new();
        let service = DockerRemovalService::from_parts(
            DockerRemovalOptions::new(4, 100_000, 10),
            daemon.clone(),
            Arc::new(FakeHost),
            Arc::new(ProcessLauncher::new(script)),
        );
        let configured = Manifest::new().with_root("/workspace");
        // Retry the bind while the freshly written script is busy.
        let mut bound = Err(SandboxError::new(
            ErrorCode::WorkspaceStartError,
            "unreached",
        ));
        for _ in 0..50 {
            bound = service.bind_new(CONTAINER, &configured).await;
            if bound.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        bound.expect("bind through the real transport");
        let error = service
            .remove(CONTAINER, &configured, "build", Some("developer"))
            .await
            .unwrap_err();
        assert_eq!(
            error.context["reason"], "docker_removal_failed",
            "{reason_name}"
        );
        assert_eq!(error.context["worker_reason"], reason_name);
        assert_eq!(
            error.context.get("errno").map(String::as_str),
            errno.map(|code: i32| code.to_string()).as_deref()
        );
        assert!(
            !daemon.paused.load(Ordering::SeqCst),
            "a structured failure thaws the container"
        );
        service.close().await.unwrap();
    }
}

#[tokio::test]
async fn removal_child_that_dies_is_a_runtime_failure_not_a_lost_transport() {
    let temp = tempfile::tempdir().unwrap();
    let script = write_script(temp.path(), "worker", &format!("{WORKER_LOOP}\nexit 9"));
    let mut worker = launch(&ProcessLauncher::new(script)).await;
    worker
        .request(json!({"operation": "inspect", "path": "/workspace/build"}))
        .await
        .unwrap();
    let failure = worker.request(removal_request()).await.unwrap_err();
    assert_eq!(
        failure,
        WorkerFailure::Os {
            errno: None,
            reason: "RuntimeError".to_string()
        }
    );
    assert!(!worker.uncertain());
    worker.close().await.unwrap();
}

#[tokio::test]
async fn closing_a_worker_twice_is_harmless() {
    let temp = tempfile::tempdir().unwrap();
    let script = write_script(temp.path(), "worker", WORKER_LOOP);
    let mut worker = launch(&ProcessLauncher::new(script)).await;
    worker.close().await.unwrap();
    worker.close().await.unwrap();
}
