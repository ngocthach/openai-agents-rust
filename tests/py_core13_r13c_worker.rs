//! Ports of `tests/sandbox/test_docker_removal_worker.py`: the trusted worker's logic.
//!
//! Python drives the worker script with syscall doubles; here the same behaviours run against
//! [`WorkerFs`] / [`NamespaceOps`] / [`ChildPrivileges`] doubles (descriptor lifetimes are tracked
//! through `Drop`) and against real directories under a temp dir through `HostFs`. Nothing needs
//! root: the namespace entry and the credential drop are doubled, as they are in Python.
#![cfg(all(feature = "docker-removal", unix))]

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::ffi::OsStr;
use std::io::{self, BufRead, Cursor, Read};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::sandbox::sandboxes::docker_removal_worker::{
    ChildPrivileges, FileStat, HostFs, NamespaceOps, RemoveRequest, WorkerError, WorkerFs,
    bind_paths, enter_container, remove, run_remove_child, selected_path, serve, user_ids,
};
use serde_json::{Value, json};

const DIR: u32 = 0o040_755;
const REG: u32 = 0o100_644;
const LNK: u32 = 0o120_777;

fn errno(code: i32) -> io::Error {
    io::Error::from_raw_os_error(code)
}

type Log = Arc<Mutex<Vec<u32>>>;

/// A descriptor double that records when it is closed.
struct IdFd {
    id: u32,
    closed: Log,
}

impl Drop for IdFd {
    fn drop(&mut self) {
        self.closed.lock().unwrap().push(self.id);
    }
}

fn stat(dev: u64, ino: u64, mode: u32) -> FileStat {
    FileStat { dev, ino, mode }
}

// --------------------------------------------------------------------------------------------
// Binding, validation and path selection
// --------------------------------------------------------------------------------------------

type PathFn = Box<dyn Fn(&str) -> io::Result<String> + Send + Sync>;
type StatFn = Box<dyn Fn(&str) -> io::Result<FileStat> + Send + Sync>;

/// A filesystem whose answers are scripted per test.
struct MapFs {
    canonical: PathFn,
    lstat: StatFn,
    fstat: HashMap<u32, FileStat>,
    root_device: u64,
    opens: Mutex<VecDeque<io::Result<u32>>>,
    fstat_failure_at: Option<usize>,
    fstat_calls: Mutex<usize>,
    closed: Log,
}

impl MapFs {
    fn new() -> Self {
        MapFs {
            canonical: Box::new(|path| Ok(path.to_string())),
            lstat: Box::new(|_| Ok(stat(1, 1, DIR))),
            fstat: HashMap::new(),
            root_device: 1,
            opens: Mutex::new(VecDeque::new()),
            fstat_failure_at: None,
            fstat_calls: Mutex::new(0),
            closed: Arc::default(),
        }
    }
}

impl WorkerFs for MapFs {
    type Fd = IdFd;

    fn canonical(&self, path: &str) -> io::Result<String> {
        (self.canonical)(path)
    }

    fn root_device(&self) -> io::Result<u64> {
        Ok(self.root_device)
    }

    fn lstat(&self, name: &OsStr, _parent: Option<&IdFd>) -> io::Result<FileStat> {
        (self.lstat)(name.to_str().unwrap())
    }

    fn fstat(&self, fd: &IdFd) -> io::Result<FileStat> {
        let mut calls = self.fstat_calls.lock().unwrap();
        *calls += 1;
        if self.fstat_failure_at == Some(*calls) {
            return Err(errno(libc::EIO));
        }
        Ok(*self.fstat.get(&fd.id).expect("scripted descriptor"))
    }

    fn open_path(&self, _path: &str) -> io::Result<IdFd> {
        let id = self
            .opens
            .lock()
            .unwrap()
            .pop_front()
            .expect("scripted open")?;
        Ok(IdFd {
            id,
            closed: self.closed.clone(),
        })
    }
}

#[test]
fn replaced_bound_root_is_rejected_even_through_new_parent_alias() {
    // Same device and inode, but the canonical name moved: a new parent alias must not pass.
    let mut fs = MapFs::new();
    fs.lstat = Box::new(|_| Ok(stat(1, 2, DIR)));
    fs.fstat.insert(123, stat(1, 2, DIR));
    fs.opens.lock().unwrap().push_back(Ok(123));
    let bound = bind_paths(&fs, &["/data/protected".to_string()]).expect("bind");
    bound.validate(&fs).expect("unchanged binding validates");
    // The same filesystem, now reached through a re-pointed parent.
    let mut moved = MapFs::new();
    moved.canonical = Box::new(|_| Ok("/elsewhere/protected".to_string()));
    moved.lstat = Box::new(|_| Ok(stat(1, 2, DIR)));
    moved.fstat.insert(123, stat(1, 2, DIR));
    assert_eq!(
        bound.validate(&moved),
        Err(WorkerError::value("bound_root_replaced"))
    );
}

#[test]
fn worker_pins_external_mount_device_but_requires_private_workspace() {
    for workspace_device in [1u64, 2] {
        let mut fs = MapFs::new();
        fs.fstat.insert(20, stat(workspace_device, 10, DIR));
        fs.fstat.insert(21, stat(2, 11, DIR));
        fs.opens.lock().unwrap().extend([Ok(20), Ok(21)]);
        let paths = vec!["/workspace".to_string(), "/toolchain".to_string()];
        let closed = fs.closed.clone();
        let result = bind_paths(&fs, &paths);
        if workspace_device != 1 {
            let error = result.expect_err("the workspace must remain on the private root");
            assert!(
                error.to_string().contains("private_root_filesystem"),
                "{error}"
            );
            assert_eq!(*closed.lock().unwrap(), vec![20]);
        } else {
            let bound = result.expect("bind");
            assert_eq!(bound.len(), 2);
            assert!(closed.lock().unwrap().is_empty());
            drop(bound);
            // Descriptors close last-opened first.
            assert_eq!(*closed.lock().unwrap(), vec![21, 20]);
        }
    }
}

#[test]
fn bound_descriptors_are_closed_after_partial_or_normal_lifetime() {
    for failure in ["open", "fstat", "none"] {
        let mut fs = MapFs::new();
        fs.fstat.insert(20, stat(1, 1, DIR));
        fs.fstat.insert(21, stat(1, 1, DIR));
        fs.opens.lock().unwrap().push_back(Ok(20));
        fs.opens.lock().unwrap().push_back(if failure == "open" {
            Err(errno(libc::EIO))
        } else {
            Ok(21)
        });
        if failure == "fstat" {
            // fstat calls: workspace, then the second root.
            fs.fstat_failure_at = Some(2);
        }
        let closed = fs.closed.clone();
        let paths = vec!["/workspace".to_string(), "/protected".to_string()];
        let result = bind_paths(&fs, &paths);
        if failure == "none" {
            let bound = result.expect("bind");
            assert_eq!(bound.paths, ["/workspace", "/protected"]);
            assert!(closed.lock().unwrap().is_empty());
            drop(bound);
        } else {
            assert!(result.is_err(), "binding must fail for {failure}");
        }
        let expected: Vec<u32> = if failure == "open" {
            vec![20]
        } else {
            vec![21, 20]
        };
        assert_eq!(*closed.lock().unwrap(), expected, "{failure}");
    }
}

#[test]
fn worker_refuses_filesystem_root_without_filesystem_calls() {
    // A double that refuses every call: the root must be refused before any lookup.
    struct Nothing;
    impl WorkerFs for Nothing {
        type Fd = IdFd;
    }
    for path in ["/", "//", "///", "/workspace/.."] {
        let error = selected_path(&Nothing, path).expect_err(path);
        assert_eq!(error, WorkerError::value("filesystem_root"), "{path}");
    }
}

#[test]
fn selected_path_resolves_parents_but_preserves_a_symlink_leaf() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    std::fs::create_dir_all(root.join("real/inner")).unwrap();
    std::os::unix::fs::symlink(root.join("real"), root.join("link")).unwrap();
    let fs = HostFs::new();
    let through_parent = selected_path(&fs, root.join("link/inner").to_str().unwrap()).unwrap();
    assert_eq!(
        through_parent,
        (root.join("real/inner").to_str().unwrap().to_string(), true)
    );
    // The leaf itself is a link: unlink semantics, so it is neither resolved nor a directory.
    let leaf = selected_path(&fs, root.join("link").to_str().unwrap()).unwrap();
    assert_eq!(
        leaf,
        (root.join("link").to_str().unwrap().to_string(), false)
    );
    // A missing leaf, and a missing parent, are "not there" rather than an error.
    let missing = selected_path(&fs, root.join("real/none").to_str().unwrap()).unwrap();
    assert_eq!(missing, (String::new(), false));
    let missing_parent = selected_path(&fs, root.join("gone/child").to_str().unwrap()).unwrap();
    assert_eq!(missing_parent, (String::new(), false));
}

// --------------------------------------------------------------------------------------------
// Accounts
// --------------------------------------------------------------------------------------------

/// A reader that records when it is dropped (closed).
struct RecordingReader {
    inner: Cursor<Vec<u8>>,
    path: String,
    dropped: Arc<Mutex<Vec<String>>>,
}

impl Read for RecordingReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl BufRead for RecordingReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }
    fn consume(&mut self, amount: usize) {
        self.inner.consume(amount);
    }
}

impl Drop for RecordingReader {
    fn drop(&mut self) {
        self.dropped.lock().unwrap().push(self.path.clone());
    }
}

struct AccountFs {
    files: HashMap<&'static str, String>,
    dropped: Arc<Mutex<Vec<String>>>,
}

impl AccountFs {
    fn new(passwd: &str, group: &str) -> Self {
        AccountFs {
            files: HashMap::from([
                ("/etc/passwd", passwd.to_string()),
                ("/etc/group", group.to_string()),
            ]),
            dropped: Arc::default(),
        }
    }
}

impl WorkerFs for AccountFs {
    type Fd = IdFd;

    fn open_account_file(&self, path: &str) -> Result<Option<Box<dyn BufRead>>, WorkerError> {
        Ok(self.files.get(path).map(|text| {
            Box::new(RecordingReader {
                inner: Cursor::new(text.clone().into_bytes()),
                path: path.to_string(),
                dropped: self.dropped.clone(),
            }) as Box<dyn BufRead>
        }))
    }
}

const PASSWD: &str = "developer:x:1000:1001::/home/developer:/bin/sh\n";
const GROUPS: &str = "tools:x:2000:developer\n";

#[test]
fn requested_user_and_groups_are_preserved() {
    let fs = AccountFs::new(PASSWD, GROUPS);
    for (user, expected) in [
        ("developer", (1000, 1001, vec![2000])),
        ("1000", (1000, 1001, vec![2000])),
        ("1000:3000", (1000, 3000, vec![])),
        ("developer:tools", (1000, 2000, vec![])),
        ("developer:3000", (1000, 3000, vec![])),
    ] {
        assert_eq!(user_ids(&fs, user).unwrap(), expected, "{user}");
    }
}

#[test]
fn unknown_user_and_group_are_refused() {
    let fs = AccountFs::new(PASSWD, GROUPS);
    assert_eq!(
        user_ids(&fs, "nobody-here"),
        Err(WorkerError::value("unknown_user"))
    );
    assert_eq!(
        user_ids(&fs, "developer:missing"),
        Err(WorkerError::value("unknown_group"))
    );
    // A container without account files still resolves a numeric id (to group 0).
    let empty = AccountFs {
        files: HashMap::new(),
        dropped: Arc::default(),
    };
    assert_eq!(user_ids(&empty, "7").unwrap(), (7, 0, vec![]));
}

#[test]
fn account_stream_closes_on_group_conversion_failure() {
    let fs = AccountFs::new(PASSWD, "tools:x:invalid:developer\n");
    assert!(user_ids(&fs, "developer").is_err());
    // The passwd stream closed before the group stream opened, and the group stream closed on
    // the failure.
    assert_eq!(
        *fs.dropped.lock().unwrap(),
        vec!["/etc/passwd".to_string(), "/etc/group".to_string()]
    );
}

#[test]
fn account_special_files_are_rejected_before_open() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    std::fs::create_dir_all(root.join("etc")).unwrap();
    // A symlink and a FIFO, for both account files; a FIFO would block a blocking open.
    for name in ["passwd", "group"] {
        let link = root.join("etc").join(name);
        std::os::unix::fs::symlink("/etc/hostname", &link).unwrap();
        let fs = HostFs::with_account_root(root);
        let error = fs
            .open_account_file(&format!("/etc/{name}"))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error,
            WorkerError::value("account_file_requires_regular_file"),
            "symlink {name}"
        );
        std::fs::remove_file(&link).unwrap();
        let c_path = std::ffi::CString::new(link.to_str().unwrap()).unwrap();
        // SAFETY: `c_path` is a valid NUL-terminated path that outlives the call.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let error = fs
            .open_account_file(&format!("/etc/{name}"))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error,
            WorkerError::value("account_file_requires_regular_file"),
            "fifo {name}"
        );
    }
    // A character device, which a temp dir cannot hold, through the real path.
    let device = HostFs::with_account_root("/")
        .open_account_file("/dev/null")
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        device,
        WorkerError::value("account_file_requires_regular_file")
    );
}

#[test]
fn regular_account_file_is_streamed_and_a_missing_one_is_absent() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join("etc")).unwrap();
    std::fs::write(temp.path().join("etc/passwd"), PASSWD).unwrap();
    std::fs::write(temp.path().join("etc/group"), GROUPS).unwrap();
    let fs = HostFs::with_account_root(temp.path());
    assert_eq!(
        user_ids(&fs, "developer").unwrap(),
        (1000, 1001, vec![2000])
    );
    assert!(fs.open_account_file("/etc/shadow").unwrap().is_none());
}

#[test]
fn user_lookup_bounds_account_parsing() {
    const MIB: usize = 1024 * 1024;
    let user_record = "root:x:0:0::/root:/bin/sh\n";
    let group_record = "tools:x:2:root\n";
    for shape in ["newlines", "fields", "members", "oversized"] {
        let (passwd, groups) = match shape {
            "newlines" => (
                format!("{}{user_record}", "\n".repeat(MIB - user_record.len())),
                format!("{}{group_record}", "\n".repeat(MIB - group_record.len())),
            ),
            "fields" => (":".repeat(MIB), ":".repeat(MIB)),
            "members" => (
                user_record.to_string(),
                format!("tools:x:2:{}root\n", ",".repeat(MIB - 15)),
            ),
            _ => (format!("{user_record}{}", "\n".repeat(MIB)), String::new()),
        };
        let fs = AccountFs::new(&passwd, &groups);
        if shape == "oversized" {
            assert_eq!(
                user_ids(&fs, "0"),
                Err(WorkerError::value("account_file_too_large"))
            );
            assert_eq!(*fs.dropped.lock().unwrap(), vec!["/etc/passwd".to_string()]);
        } else {
            let expected_groups = if shape == "fields" { vec![] } else { vec![2] };
            assert_eq!(
                user_ids(&fs, "0").unwrap(),
                (0, 0, expected_groups),
                "{shape}"
            );
            assert_eq!(fs.dropped.lock().unwrap().len(), 2, "{shape}");
        }
    }
}

// --------------------------------------------------------------------------------------------
// Namespace entry
// --------------------------------------------------------------------------------------------

struct EntryOps {
    events: Arc<Mutex<Vec<String>>>,
    root_open_fails: bool,
}

struct NamedFd {
    name: &'static str,
    events: Arc<Mutex<Vec<String>>>,
}

impl Drop for NamedFd {
    fn drop(&mut self) {
        self.events
            .lock()
            .unwrap()
            .push(format!("close {}", self.name));
    }
}

impl NamespaceOps for EntryOps {
    type Fd = NamedFd;

    fn open_mount_namespace(&self, _pid: u32) -> io::Result<NamedFd> {
        Ok(NamedFd {
            name: "mnt",
            events: self.events.clone(),
        })
    }

    fn open_root(&self, _pid: u32) -> io::Result<NamedFd> {
        if self.root_open_fails {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        Ok(NamedFd {
            name: "root",
            events: self.events.clone(),
        })
    }

    fn enter_mount_namespace(&self, fd: &NamedFd) -> io::Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("setns {}", fd.name));
        Ok(())
    }

    fn fchdir(&self, fd: &NamedFd) -> io::Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("fchdir {}", fd.name));
        Ok(())
    }

    fn chroot_here(&self) -> io::Result<()> {
        self.events.lock().unwrap().push("chroot .".to_string());
        Ok(())
    }

    fn chdir_root(&self) -> io::Result<()> {
        self.events.lock().unwrap().push("chdir /".to_string());
        Ok(())
    }
}

#[test]
fn namespace_entry_uses_only_mount_namespace_and_closes_host_directory_handles() {
    let events = Arc::new(Mutex::new(Vec::new()));
    enter_container(
        &EntryOps {
            events: events.clone(),
            root_open_fails: false,
        },
        123,
    )
    .unwrap();
    let recorded = events.lock().unwrap().clone();
    assert_eq!(
        recorded[..4],
        ["setns mnt", "fchdir root", "chroot .", "chdir /"]
    );
    let closed: BTreeSet<&str> = recorded[4..].iter().map(String::as_str).collect();
    assert_eq!(closed, BTreeSet::from(["close mnt", "close root"]));
}

#[test]
fn namespace_entry_closes_mount_handle_when_root_open_fails() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let error = enter_container(
        &EntryOps {
            events: events.clone(),
            root_open_fails: true,
        },
        123,
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    // `setns` was never reached, and the mount handle was closed.
    assert_eq!(*events.lock().unwrap(), vec!["close mnt".to_string()]);
}

// --------------------------------------------------------------------------------------------
// The request loop and the removal child
// --------------------------------------------------------------------------------------------

/// The scripted filesystem of the protocol cases: a canonical rewrite that can re-point after the
/// bind, and a user-scoped `lstat`.
struct ProtocolFs {
    canonical_root: &'static str,
    repoint_to: Option<&'static str>,
    bound: Mutex<bool>,
    current_user: Arc<Mutex<&'static str>>,
    original: &'static str,
    selected: &'static str,
    exists: bool,
    accessible: bool,
    removed: Mutex<Vec<String>>,
    closed: Log,
}

impl WorkerFs for ProtocolFs {
    type Fd = IdFd;

    fn canonical(&self, _path: &str) -> io::Result<String> {
        let rebound = *self.bound.lock().unwrap();
        Ok(match self.repoint_to {
            Some(moved) if rebound => moved.to_string(),
            _ => self.canonical_root.to_string(),
        })
    }

    fn root_device(&self) -> io::Result<u64> {
        Ok(1)
    }

    fn lstat(&self, name: &OsStr, _parent: Option<&IdFd>) -> io::Result<FileStat> {
        let name = name.to_str().unwrap();
        if *self.current_user.lock().unwrap() != "root" && name == self.original && !self.accessible
        {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        if name == self.selected && !self.exists {
            return Err(io::Error::from(io::ErrorKind::NotFound));
        }
        if name == self.original && !self.exists {
            return Err(io::Error::from(io::ErrorKind::NotFound));
        }
        Ok(stat(1, 1, DIR))
    }

    fn fstat(&self, _fd: &IdFd) -> io::Result<FileStat> {
        Ok(stat(1, 1, DIR))
    }

    fn open_path(&self, _path: &str) -> io::Result<IdFd> {
        *self.bound.lock().unwrap() = true;
        Ok(IdFd {
            id: 20,
            closed: self.closed.clone(),
        })
    }

    fn rmdir(&self, name: &OsStr, _parent: Option<&IdFd>) -> io::Result<()> {
        self.removed
            .lock()
            .unwrap()
            .push(name.to_str().unwrap().to_string());
        Ok(())
    }
}

/// Privileges that only record the user the removal ends up running as.
struct UserSwitch {
    current_user: Arc<Mutex<&'static str>>,
    calls: Mutex<Vec<String>>,
    limit_failure: Option<io::ErrorKind>,
}

impl ChildPrivileges for UserSwitch {
    fn set_cpu_limit(&self, seconds: u64) -> io::Result<()> {
        self.calls.lock().unwrap().push(format!("cpu {seconds}"));
        match self.limit_failure {
            Some(kind) => Err(io::Error::from(kind)),
            None => Ok(()),
        }
    }

    fn set_groups(&self, groups: &[u32]) -> io::Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("groups {groups:?}"));
        Ok(())
    }

    fn set_gid(&self, gid: u32) -> io::Result<()> {
        self.calls.lock().unwrap().push(format!("gid {gid}"));
        Ok(())
    }

    fn set_uid(&self, uid: u32) -> io::Result<()> {
        self.calls.lock().unwrap().push(format!("uid {uid}"));
        *self.current_user.lock().unwrap() = "user";
        Ok(())
    }
}

fn serve_lines(fs: &ProtocolFs, requests: &[Value]) -> Vec<Value> {
    let input = requests
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let mut output = Vec::new();
    serve(fs, &mut Cursor::new(input.into_bytes()), &mut output).unwrap();
    String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn worker_protocol_keeps_original_traversal_for_user_scoped_removal() {
    for accessible in [false, true] {
        for exists in [false, true] {
            let original = "/workspace/private/link/build";
            let current_user = Arc::new(Mutex::new("root"));
            let fs = ProtocolFs {
                canonical_root: "/workspace/shared",
                repoint_to: None,
                bound: Mutex::new(false),
                current_user: current_user.clone(),
                original,
                selected: "/workspace/shared/build",
                exists,
                accessible,
                removed: Mutex::new(Vec::new()),
                closed: Arc::default(),
            };
            let responses = serve_lines(
                &fs,
                &[
                    json!({"operation": "bind", "paths": ["/workspace"]}),
                    json!({"operation": "inspect", "path": original}),
                ],
            );
            assert_eq!(
                responses[1],
                json!({
                    "ok": true,
                    "path": if exists { "/workspace/shared/build" } else { "" },
                    "is_directory": exists,
                }),
                "accessible={accessible} exists={exists}"
            );
            // The removal child drops to the user, then removes the ORIGINAL path, so the
            // user's search permissions on the alias apply.
            let privileges = UserSwitch {
                current_user,
                calls: Mutex::new(Vec::new()),
                limit_failure: None,
            };
            let outcome = run_remove_child(
                &fs,
                &privileges,
                &RemoveRequest {
                    path: original.to_string(),
                    user: "1000:1000".to_string(),
                    max_entry_visits: 100_000,
                    max_cpu_seconds: 10,
                },
            );
            assert_eq!(
                outcome,
                if accessible {
                    json!({"ok": true})
                } else {
                    json!({"ok": false, "reason": "PermissionError", "errno": null})
                }
            );
            let removed = fs.removed.lock().unwrap().clone();
            let expected: Vec<String> = if accessible && exists {
                vec![original.to_string()]
            } else {
                Vec::new()
            };
            assert_eq!(removed, expected);
        }
    }
}

#[test]
fn worker_preserves_workspace_alias_permissions_and_bound_identity() {
    let alias = "/private/workspace-alias";
    let original = "/private/workspace-alias/build";
    for outcome in ["allowed", "search_denied", "repointed"] {
        let current_user = Arc::new(Mutex::new("root"));
        let fs = ProtocolFs {
            canonical_root: "/canonical",
            repoint_to: (outcome == "repointed").then_some("/different"),
            bound: Mutex::new(false),
            current_user: current_user.clone(),
            original,
            selected: "/canonical/build",
            exists: true,
            accessible: outcome != "search_denied",
            removed: Mutex::new(Vec::new()),
            closed: Arc::default(),
        };
        let responses = serve_lines(
            &fs,
            &[
                json!({"operation": "bind", "paths": [alias]}),
                json!({"operation": "inspect", "path": original, "workspace_root": alias}),
                json!({"operation": "remove", "user": "developer",
                       "max_entry_visits": 100_000, "max_cpu_seconds": 10}),
            ],
        );
        if outcome == "repointed" {
            assert_eq!(
                responses[1],
                json!({"ok": false, "reason": "ValueError", "errno": null})
            );
            // The worker never removes: only the removal child does, and the service never asks
            // for one after a failed inspect.
            assert_eq!(
                responses[2],
                json!({"ok": false, "reason": "ValueError", "errno": null})
            );
            assert!(fs.removed.lock().unwrap().is_empty());
            continue;
        }
        assert_eq!(responses[1]["path"], "/canonical/build", "{outcome}");
        // `remove` is not a worker operation: it is a sibling child process.
        assert_eq!(
            responses[2],
            json!({"ok": false, "reason": "ValueError", "errno": null})
        );
        let privileges = UserSwitch {
            current_user,
            calls: Mutex::new(Vec::new()),
            limit_failure: None,
        };
        let outcome_json = run_remove_child(
            &fs,
            &privileges,
            &RemoveRequest {
                path: original.to_string(),
                user: "1000:1000".to_string(),
                max_entry_visits: 100_000,
                max_cpu_seconds: 10,
            },
        );
        if outcome == "search_denied" {
            assert_eq!(
                outcome_json,
                json!({"ok": false, "reason": "PermissionError", "errno": null})
            );
            assert!(fs.removed.lock().unwrap().is_empty());
        } else {
            assert_eq!(outcome_json, json!({"ok": true}));
            assert_eq!(*fs.removed.lock().unwrap(), vec![original.to_string()]);
        }
    }
}

#[test]
fn removal_child_reports_every_failure_after_the_limit_and_identity_steps() {
    struct FailingFs {
        failure: Option<io::ErrorKind>,
        raw: Option<i32>,
        user_lookup: bool,
    }
    impl WorkerFs for FailingFs {
        type Fd = IdFd;
        fn open_account_file(&self, _path: &str) -> Result<Option<Box<dyn BufRead>>, WorkerError> {
            assert!(self.user_lookup, "the account files must not be read");
            Ok(None)
        }
        fn lstat(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<FileStat> {
            match (self.failure, self.raw) {
                (_, Some(code)) => Err(errno(code)),
                (Some(kind), None) => Err(io::Error::from(kind)),
                _ => Ok(stat(1, 1, REG)),
            }
        }
        fn unlink(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<()> {
            Ok(())
        }
    }
    let request = |user: &str| RemoveRequest {
        path: "/workspace/build".to_string(),
        user: user.to_string(),
        max_entry_visits: 100_000,
        max_cpu_seconds: 10,
    };
    let switch = |limit_failure| UserSwitch {
        current_user: Arc::new(Mutex::new("root")),
        calls: Mutex::new(Vec::new()),
        limit_failure,
    };

    // Success.
    let privileges = switch(None);
    let ok = run_remove_child(
        &FailingFs {
            failure: None,
            raw: None,
            user_lookup: false,
        },
        &privileges,
        &request("1000:1000"),
    );
    assert_eq!(ok, json!({"ok": true}));
    assert_eq!(
        *privileges.calls.lock().unwrap(),
        ["cpu 10", "groups []", "gid 1000", "uid 1000"]
    );

    // PermissionError(EACCES) and OSError(E2BIG) carry their errno.
    for (raw, reason) in [(libc::EACCES, "PermissionError"), (libc::E2BIG, "OSError")] {
        let outcome = run_remove_child(
            &FailingFs {
                failure: None,
                raw: Some(raw),
                user_lookup: false,
            },
            &switch(None),
            &request("1000:1000"),
        );
        assert_eq!(
            outcome,
            json!({"ok": false, "reason": reason, "errno": raw})
        );
    }

    // A ValueError (an unknown user) has no errno.
    let outcome = run_remove_child(
        &FailingFs {
            failure: None,
            raw: None,
            user_lookup: true,
        },
        &switch(None),
        &request("nobody"),
    );
    assert_eq!(
        outcome,
        json!({"ok": false, "reason": "ValueError", "errno": null})
    );

    // A refused CPU limit is reported, and nothing else ran: no account lookup, no credential
    // change.
    let privileges = switch(Some(io::ErrorKind::PermissionDenied));
    let outcome = run_remove_child(
        &FailingFs {
            failure: None,
            raw: None,
            user_lookup: false,
        },
        &privileges,
        &request("nobody"),
    );
    assert_eq!(
        outcome,
        json!({"ok": false, "reason": "PermissionError", "errno": null})
    );
    assert_eq!(*privileges.calls.lock().unwrap(), ["cpu 10"]);
}

#[test]
fn worker_closes_bindings_when_response_write_fails() {
    struct BrokenOutput;
    impl io::Write for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut fs = MapFs::new();
    fs.fstat.insert(20, stat(1, 1, DIR));
    fs.opens.lock().unwrap().push_back(Ok(20));
    let closed = fs.closed.clone();
    let input = b"{\"operation\":\"bind\",\"paths\":[\"/workspace\"]}\n".to_vec();
    let error = serve(&fs, &mut Cursor::new(input), &mut BrokenOutput).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(*closed.lock().unwrap(), vec![20]);
}

// --------------------------------------------------------------------------------------------
// Removal traversal
// --------------------------------------------------------------------------------------------

#[test]
fn empty_directory_needs_no_search_of_its_contents() {
    struct EmptyDirectory {
        removed: Mutex<Vec<String>>,
    }
    impl WorkerFs for EmptyDirectory {
        type Fd = IdFd;
        fn lstat(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<FileStat> {
            Ok(stat(1, 1, DIR))
        }
        fn rmdir(&self, name: &OsStr, _parent: Option<&IdFd>) -> io::Result<()> {
            self.removed
                .lock()
                .unwrap()
                .push(name.to_str().unwrap().to_string());
            Ok(())
        }
        fn scan(
            &self,
            _dir: &IdFd,
            _visit: &mut dyn FnMut(&OsStr) -> Result<bool, WorkerError>,
        ) -> Result<(), WorkerError> {
            panic!("must not search");
        }
    }
    let fs = EmptyDirectory {
        removed: Mutex::new(Vec::new()),
    };
    remove(&fs, "/workspace/empty", 100_000).unwrap();
    assert_eq!(
        *fs.removed.lock().unwrap(),
        vec!["/workspace/empty".to_string()]
    );
}

/// Python's `_DeepRemovalTree`: a chain of nested directories `/tree/d/d/...`.
struct TreeState {
    remaining: BTreeSet<usize>,
    removed: Vec<usize>,
    fds: HashMap<u64, usize>,
    next_fd: u64,
    open_scans: usize,
    peak_fds: usize,
}

struct TreeFs {
    depth: usize,
    state: Arc<Mutex<TreeState>>,
    fail: Option<&'static str>,
}

struct TreeFd {
    id: u64,
    state: Arc<Mutex<TreeState>>,
}

impl Drop for TreeFd {
    fn drop(&mut self) {
        self.state.lock().unwrap().fds.remove(&self.id);
    }
}

impl TreeFs {
    fn new(depth: usize, fail: Option<&'static str>) -> Self {
        TreeFs {
            depth,
            state: Arc::new(Mutex::new(TreeState {
                remaining: (0..=depth).collect(),
                removed: Vec::new(),
                fds: HashMap::new(),
                next_fd: 100,
                open_scans: 0,
                peak_fds: 0,
            })),
            fail,
        }
    }

    fn lookup(&self, name: &str, parent: Option<&TreeFd>) -> io::Result<usize> {
        if name.len() >= 4096 {
            return Err(errno(libc::ENAMETOOLONG));
        }
        let node = match parent {
            None => {
                assert_eq!(name, "/tree");
                0
            }
            Some(parent) => {
                assert!(
                    name == "d" || name == "..",
                    "descendants must use a single relative component"
                );
                let base = self.state.lock().unwrap().fds[&parent.id];
                if name == "d" {
                    base + 1
                } else {
                    base.wrapping_sub(1)
                }
            }
        };
        if !self.state.lock().unwrap().remaining.contains(&node) {
            return Err(io::Error::from(io::ErrorKind::NotFound));
        }
        Ok(node)
    }
}

impl WorkerFs for TreeFs {
    type Fd = TreeFd;

    fn lstat(&self, name: &OsStr, parent: Option<&TreeFd>) -> io::Result<FileStat> {
        self.lookup(name.to_str().unwrap(), parent)?;
        Ok(stat(1, 1, DIR))
    }

    fn rmdir(&self, name: &OsStr, parent: Option<&TreeFd>) -> io::Result<()> {
        let node = self.lookup(name.to_str().unwrap(), parent)?;
        if self.fail == Some("remove") && node == self.depth {
            return Err(errno(libc::EACCES));
        }
        let mut state = self.state.lock().unwrap();
        if state.remaining.contains(&(node + 1)) {
            return Err(errno(libc::ENOTEMPTY));
        }
        state.remaining.remove(&node);
        state.removed.push(node);
        Ok(())
    }

    fn unlink(&self, _name: &OsStr, _parent: Option<&TreeFd>) -> io::Result<()> {
        panic!("directories only");
    }

    fn open_dir(&self, name: &OsStr, parent: Option<&TreeFd>) -> io::Result<TreeFd> {
        let name = name.to_str().unwrap();
        if (self.fail == Some("child_open") && name == "d")
            || (self.fail == Some("parent_open") && name == "..")
        {
            return Err(errno(libc::EACCES));
        }
        assert_eq!(self.state.lock().unwrap().open_scans, 0);
        let node = self.lookup(name, parent)?;
        let mut state = self.state.lock().unwrap();
        state.next_fd += 1;
        let id = state.next_fd;
        state.fds.insert(id, node);
        state.peak_fds = state.peak_fds.max(state.fds.len());
        assert!(state.peak_fds <= 2, "at most two descriptors are ever open");
        Ok(TreeFd {
            id,
            state: self.state.clone(),
        })
    }

    fn scan(
        &self,
        dir: &TreeFd,
        visit: &mut dyn FnMut(&OsStr) -> Result<bool, WorkerError>,
    ) -> Result<(), WorkerError> {
        let has_child = {
            let mut state = self.state.lock().unwrap();
            let node = state.fds[&dir.id];
            state.open_scans += 1;
            assert_eq!(state.open_scans, 1);
            state.remaining.contains(&(node + 1))
        };
        let result = if self.fail == Some("scan") {
            Err(errno(libc::EACCES).into())
        } else if has_child {
            visit(OsStr::new("d")).map(|_| ())
        } else {
            Ok(())
        };
        self.state.lock().unwrap().open_scans -= 1;
        result
    }
}

#[test]
fn worker_removes_deep_tree_beyond_path_max_with_bounded_descriptors() {
    let fs = TreeFs::new(2500, None);
    assert!("/tree".len() + "/d".len() * fs.depth > 4096);
    remove(&fs, "/tree", 100_000).unwrap();
    let state = fs.state.lock().unwrap();
    assert!(state.remaining.is_empty());
    assert_eq!(state.removed, (0..=fs.depth).rev().collect::<Vec<_>>());
    assert!(state.fds.is_empty());
    assert_eq!(state.open_scans, 0);
    assert_eq!(state.peak_fds, 2);
}

#[test]
fn worker_closes_traversal_descriptors_after_failure() {
    for failure in ["child_open", "parent_open", "scan", "remove", "budget"] {
        let fs = TreeFs::new(3, Some(failure));
        let budget = if failure == "budget" { 2 } else { 100_000 };
        let error = remove(&fs, "/tree", budget).expect_err(failure);
        let expected = if failure == "budget" {
            libc::E2BIG
        } else {
            libc::EACCES
        };
        assert_eq!(error.errno(), Some(expected), "{failure}");
        let state = fs.state.lock().unwrap();
        assert!(state.fds.is_empty(), "{failure}");
        assert_eq!(state.open_scans, 0, "{failure}");
    }
}

#[test]
fn worker_streams_wide_directory_without_buffering_sibling_paths() {
    const COUNT: usize = 10_000;
    struct WideFs {
        leaf_mode: u32,
        deleted: Mutex<usize>,
        root_removed: Mutex<bool>,
        closed: Log,
    }
    impl WorkerFs for WideFs {
        type Fd = IdFd;
        fn lstat(&self, name: &OsStr, _parent: Option<&IdFd>) -> io::Result<FileStat> {
            let mode = if name == "/tree" { DIR } else { self.leaf_mode };
            Ok(stat(1, 1, mode))
        }
        fn rmdir(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<()> {
            if *self.deleted.lock().unwrap() != COUNT {
                return Err(errno(libc::ENOTEMPTY));
            }
            *self.root_removed.lock().unwrap() = true;
            Ok(())
        }
        fn unlink(&self, name: &OsStr, parent: Option<&IdFd>) -> io::Result<()> {
            assert_eq!(parent.map(|fd| fd.id), Some(123));
            let mut deleted = self.deleted.lock().unwrap();
            assert_eq!(name.to_str().unwrap(), deleted.to_string());
            *deleted += 1;
            Ok(())
        }
        fn open_dir(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<IdFd> {
            Ok(IdFd {
                id: 123,
                closed: self.closed.clone(),
            })
        }
        fn scan(
            &self,
            _dir: &IdFd,
            visit: &mut dyn FnMut(&OsStr) -> Result<bool, WorkerError>,
        ) -> Result<(), WorkerError> {
            for index in 0..COUNT {
                assert_eq!(
                    *self.deleted.lock().unwrap(),
                    index,
                    "each leaf must be consumed before fetching the next"
                );
                if !visit(OsStr::new(&index.to_string()))? {
                    break;
                }
            }
            Ok(())
        }
    }
    for leaf_mode in [REG, LNK] {
        let fs = WideFs {
            leaf_mode,
            deleted: Mutex::new(0),
            root_removed: Mutex::new(false),
            closed: Arc::default(),
        };
        remove(&fs, "/tree", 100_000).unwrap();
        assert!(*fs.root_removed.lock().unwrap());
        assert_eq!(*fs.deleted.lock().unwrap(), COUNT);
        assert_eq!(*fs.closed.lock().unwrap(), vec![123]);
    }
}

#[test]
fn worker_stops_entry_visits_before_additional_filesystem_work() {
    struct Visited {
        visited: Mutex<Vec<String>>,
        closed: Log,
    }
    impl WorkerFs for Visited {
        type Fd = IdFd;
        fn lstat(&self, name: &OsStr, _parent: Option<&IdFd>) -> io::Result<FileStat> {
            let name = name.to_str().unwrap().to_string();
            let mode = if name == "/tree" { DIR } else { REG };
            self.visited.lock().unwrap().push(name);
            Ok(stat(1, 1, mode))
        }
        fn rmdir(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<()> {
            Err(errno(libc::ENOTEMPTY))
        }
        fn unlink(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<()> {
            Ok(())
        }
        fn open_dir(&self, _name: &OsStr, _parent: Option<&IdFd>) -> io::Result<IdFd> {
            Ok(IdFd {
                id: 123,
                closed: self.closed.clone(),
            })
        }
        fn scan(
            &self,
            _dir: &IdFd,
            visit: &mut dyn FnMut(&OsStr) -> Result<bool, WorkerError>,
        ) -> Result<(), WorkerError> {
            for index in 0..100 {
                if !visit(OsStr::new(&index.to_string()))? {
                    break;
                }
            }
            Ok(())
        }
    }
    let fs = Visited {
        visited: Mutex::new(Vec::new()),
        closed: Arc::default(),
    };
    let error = remove(&fs, "/tree", 3).unwrap_err();
    assert_eq!(error.errno(), Some(libc::E2BIG));
    assert_eq!(*fs.visited.lock().unwrap(), ["/tree", "0", "1"]);
    assert_eq!(*fs.closed.lock().unwrap(), vec![123]);
}

#[test]
fn host_filesystem_removes_a_real_tree_without_following_links() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let outside = base.join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("keep.txt"), "kept").unwrap();
    let tree = base.join("tree");
    std::fs::create_dir_all(tree.join("a/b/c")).unwrap();
    std::fs::write(tree.join("a/file.txt"), "x").unwrap();
    std::fs::write(tree.join("a/b/c/deep.txt"), "x").unwrap();
    std::os::unix::fs::symlink(&outside, tree.join("a/link")).unwrap();
    std::os::unix::fs::symlink(&outside, tree.join("b-link")).unwrap();
    remove(&HostFs::new(), tree.to_str().unwrap(), 1000).unwrap();
    assert!(!tree.exists());
    // The links were unlinked, never followed.
    assert_eq!(
        std::fs::read_to_string(outside.join("keep.txt")).unwrap(),
        "kept"
    );
    // A missing target is already removed.
    remove(&HostFs::new(), tree.to_str().unwrap(), 1000).unwrap();
}

#[test]
fn host_filesystem_binds_and_validates_real_roots() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let workspace = base.join("workspace");
    let grant = base.join("grant");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&grant).unwrap();
    std::os::unix::fs::symlink(&grant, base.join("grant-alias")).unwrap();
    // The temp directory need not share a device with `/`, so pin the root device to it.
    struct TempRooted(HostFs, u64);
    impl WorkerFs for TempRooted {
        type Fd = <HostFs as WorkerFs>::Fd;
        fn canonical(&self, path: &str) -> io::Result<String> {
            self.0.canonical(path)
        }
        fn root_device(&self) -> io::Result<u64> {
            Ok(self.1)
        }
        fn lstat(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<FileStat> {
            self.0.lstat(name, parent)
        }
        fn fstat(&self, fd: &Self::Fd) -> io::Result<FileStat> {
            self.0.fstat(fd)
        }
        fn open_path(&self, path: &str) -> io::Result<Self::Fd> {
            self.0.open_path(path)
        }
    }
    let device = HostFs::new()
        .lstat(OsStr::new(base.to_str().unwrap()), None)
        .unwrap()
        .dev;
    let fs = TempRooted(HostFs::new(), device);
    let paths = vec![
        workspace.to_str().unwrap().to_string(),
        base.join("grant-alias").to_str().unwrap().to_string(),
    ];
    let bound = bind_paths(&fs, &paths).expect("bind");
    // The alias resolves to its canonical target.
    assert_eq!(bound.paths[1], grant.to_str().unwrap());
    bound.validate(&fs).expect("fresh binding validates");
    // Replacing the bound root with another directory fails closed.
    std::fs::remove_dir(&grant).unwrap();
    std::fs::create_dir(&grant).unwrap();
    assert!(bound.validate(&fs).is_err());
    drop(bound);

    // The workspace itself must be an existing directory.
    let file = base.join("not-a-directory");
    std::fs::write(&file, "x").unwrap();
    let error = bind_paths(&fs, &[file.to_str().unwrap().to_string()]).unwrap_err();
    assert_eq!(
        error,
        WorkerError::value("workspace_requires_existing_directory")
    );
    // The filesystem root can never be bound.
    let error = bind_paths(&fs, &["/".to_string()]).unwrap_err();
    assert_eq!(error, WorkerError::value("filesystem_root"));
}
