//! The trusted host worker of the opt-in Docker removal service. Mirrors
//! `agents.sandbox.sandboxes._docker_removal_worker`.
//!
//! This module is the logic of the `docker-removal-worker` executable that
//! [`DockerRemovalService`](super::docker_removal::DockerRemovalService) launches on the Docker
//! daemon host. It never runs through `docker exec`: it is a host binary that enters the
//! container's mount namespace and root (`setns`, `fchdir`, `chroot`) while the service holds the
//! container paused. **The caller supplies the privileges**: the worker needs root on a Linux
//! Docker host, and nothing in this crate escalates, `sudo`s or grants anything.
//!
//! Two kinds of process use it:
//!
//! - the long-lived *worker* (`docker-removal-worker worker <pid>`) pins the workspace and grant
//!   roots with `O_PATH` descriptors (`bind`) and resolves removal targets (`inspect`);
//! - a short-lived *removal child* (`docker-removal-worker remove <pid>`) enters the container,
//!   caps its CPU time, drops to the requested user and removes one tree.
//!
//! Python forks the removal child from the worker. The crate forbids `unsafe`, so there is no
//! `fork`: the service starts the child as a sibling process inside the same pause window. The
//! request/response shapes of Python's protocol are otherwise unchanged.
//!
//! Everything that touches the filesystem goes through [`WorkerFs`], so the traversal, the
//! binding checks and the account lookup are exercised on ordinary directories without root.

use std::ffi::{OsStr, OsString};
use std::io::{self, BufRead, Read, Write};

use serde_json::{Value, json};

/// A failure of one worker operation: Python's `ValueError` or an `OSError` subclass.
///
/// [`reason`](Self::reason) and [`errno`](Self::errno) are what the worker puts on the wire, and
/// what the service reports back as `worker_reason` and `errno`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerError {
    /// A refusal such as `filesystem_root` or `bound_root_replaced`.
    Value(String),
    /// An operating-system error, with the Python exception class name as its reason.
    Os {
        /// The Python exception class name (`PermissionError`, `OSError`, ...).
        reason: String,
        /// The errno, when there is one.
        errno: Option<i32>,
        /// A short description.
        message: String,
    },
}

impl WorkerError {
    /// A `ValueError`.
    pub fn value(message: impl Into<String>) -> Self {
        WorkerError::Value(message.into())
    }

    /// A bare `OSError(errno, message)`.
    pub fn os(errno: i32, message: impl Into<String>) -> Self {
        WorkerError::Os {
            reason: "OSError".to_string(),
            errno: Some(errno),
            message: message.into(),
        }
    }

    /// The Python exception class name.
    pub fn reason(&self) -> &str {
        match self {
            WorkerError::Value(_) => "ValueError",
            WorkerError::Os { reason, .. } => reason,
        }
    }

    /// The errno, when this is an operating-system error that has one.
    pub fn errno(&self) -> Option<i32> {
        match self {
            WorkerError::Value(_) => None,
            WorkerError::Os { errno, .. } => *errno,
        }
    }

    /// The `{"ok": false, ...}` response for this failure.
    pub fn to_response(&self) -> Value {
        json!({"ok": false, "reason": self.reason(), "errno": self.errno()})
    }
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerError::Value(message) => write!(f, "{message}"),
            WorkerError::Os {
                reason, message, ..
            } => write!(f, "{reason}: {message}"),
        }
    }
}

impl std::error::Error for WorkerError {}

impl From<io::Error> for WorkerError {
    fn from(error: io::Error) -> Self {
        let reason = match error.kind() {
            io::ErrorKind::NotFound => "FileNotFoundError",
            io::ErrorKind::PermissionDenied => "PermissionError",
            io::ErrorKind::AlreadyExists => "FileExistsError",
            io::ErrorKind::NotADirectory => "NotADirectoryError",
            io::ErrorKind::IsADirectory => "IsADirectoryError",
            _ => "OSError",
        };
        WorkerError::Os {
            reason: reason.to_string(),
            errno: error.raw_os_error(),
            message: error.to_string(),
        }
    }
}

impl From<rustix::io::Errno> for WorkerError {
    fn from(error: rustix::io::Errno) -> Self {
        io::Error::from(error).into()
    }
}

const S_IFMT: u32 = 0o170_000;
const S_IFDIR: u32 = 0o040_000;
const S_IFREG: u32 = 0o100_000;

/// The part of `stat` the worker looks at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStat {
    /// The device the entry lives on.
    pub dev: u64,
    /// The inode number.
    pub ino: u64,
    /// The mode, including the file-type bits.
    pub mode: u32,
}

impl FileStat {
    /// Whether the entry is a directory.
    pub fn is_dir(&self) -> bool {
        self.mode & S_IFMT == S_IFDIR
    }

    /// Whether the entry is a regular file.
    pub fn is_reg(&self) -> bool {
        self.mode & S_IFMT == S_IFREG
    }
}

fn unsupported<T>() -> io::Result<T> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// The filesystem operations the worker needs. [`HostFs`] is the real implementation; the
/// defaults refuse, so a double only implements what a case exercises.
pub trait WorkerFs {
    /// An open descriptor. Dropping it closes it.
    type Fd;

    /// The fully resolved absolute path (`realpath`, strict).
    fn canonical(&self, _path: &str) -> io::Result<String> {
        unsupported()
    }

    /// The device of `/`.
    fn root_device(&self) -> io::Result<u64> {
        unsupported()
    }

    /// `lstat` of `name`, relative to `parent` when one is given.
    fn lstat(&self, _name: &OsStr, _parent: Option<&Self::Fd>) -> io::Result<FileStat> {
        unsupported()
    }

    /// `fstat` of an open descriptor.
    fn fstat(&self, _fd: &Self::Fd) -> io::Result<FileStat> {
        unsupported()
    }

    /// Open `path` with `O_PATH | O_NOFOLLOW`.
    fn open_path(&self, _path: &str) -> io::Result<Self::Fd> {
        unsupported()
    }

    /// Open a directory with `O_RDONLY | O_DIRECTORY | O_NOFOLLOW`, relative to `parent`.
    fn open_dir(&self, _name: &OsStr, _parent: Option<&Self::Fd>) -> io::Result<Self::Fd> {
        unsupported()
    }

    /// `rmdir`, relative to `parent` when one is given.
    fn rmdir(&self, _name: &OsStr, _parent: Option<&Self::Fd>) -> io::Result<()> {
        unsupported()
    }

    /// `unlink`, relative to `parent` when one is given.
    fn unlink(&self, _name: &OsStr, _parent: Option<&Self::Fd>) -> io::Result<()> {
        unsupported()
    }

    /// Stream the names in `dir` to `visit` until it returns `Ok(false)`.
    ///
    /// The scan is closed before this returns, so the caller may descend afterwards, and no
    /// sibling list is kept: a name is handed over only once the previous one was consumed.
    fn scan(
        &self,
        _dir: &Self::Fd,
        _visit: &mut dyn FnMut(&OsStr) -> Result<bool, WorkerError>,
    ) -> Result<(), WorkerError> {
        Err(io::Error::from(io::ErrorKind::Unsupported).into())
    }

    /// Open an account file (`/etc/passwd` or `/etc/group`) for streaming, or `None` when the
    /// container has none. A file that is not a regular file is refused before it is opened.
    fn open_account_file(&self, _path: &str) -> Result<Option<Box<dyn BufRead>>, WorkerError> {
        Err(io::Error::from(io::ErrorKind::Unsupported).into())
    }
}

/// `realpath("/" + path.lstrip("/"), strict=True)`.
fn canonical<F: WorkerFs>(fs: &F, path: &str) -> io::Result<String> {
    fs.canonical(&format!("/{}", path.trim_start_matches('/')))
}

/// The descriptors a `bind` pinned, closed last-opened-first.
pub struct Bindings<F: WorkerFs> {
    /// The canonical roots; the first is the workspace.
    pub paths: Vec<String>,
    fds: Vec<F::Fd>,
}

impl<F: WorkerFs> Bindings<F> {
    /// The number of pinned roots.
    pub fn len(&self) -> usize {
        self.fds.len()
    }

    /// Whether nothing is pinned.
    pub fn is_empty(&self) -> bool {
        self.fds.is_empty()
    }

    /// Check that every pinned root still names the same inode at the same canonical path.
    pub fn validate(&self, fs: &F) -> Result<(), WorkerError> {
        let workspace = self.paths.first().map(|first| format!("{first}/"));
        for (path, fd) in self.paths.iter().zip(&self.fds) {
            // Workspace precedence makes nested grants ordinary writable entries.
            if workspace
                .as_ref()
                .is_some_and(|prefix| path.starts_with(prefix.as_str()))
            {
                continue;
            }
            let entry = fs.lstat(OsStr::new(path), None)?;
            let pinned = fs.fstat(fd)?;
            if canonical(fs, path)? != *path || (entry.dev, entry.ino) != (pinned.dev, pinned.ino) {
                return Err(WorkerError::value("bound_root_replaced"));
            }
        }
        Ok(())
    }
}

impl<F: WorkerFs> std::fmt::Debug for Bindings<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bindings")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

impl<F: WorkerFs> Drop for Bindings<F> {
    fn drop(&mut self) {
        while let Some(fd) = self.fds.pop() {
            drop(fd);
        }
    }
}

/// Pin the canonical workspace and grant roots with descriptors.
///
/// The workspace must be an existing directory on the container's private root filesystem; a
/// grant may be a directory or a regular file on a read-only mount.
pub fn bind_paths<F: WorkerFs>(fs: &F, paths: &[String]) -> Result<Bindings<F>, WorkerError> {
    let device = fs.root_device()?;
    // Built incrementally so a failure part-way drops (closes) what was already opened.
    let mut bound = Bindings::<F> {
        paths: Vec::new(),
        fds: Vec::new(),
    };
    for path in paths {
        let resolved = canonical(fs, path)?;
        if resolved == "/" {
            return Err(WorkerError::value("filesystem_root"));
        }
        let fd = fs.open_path(&resolved)?;
        let entry = fs.fstat(&fd)?;
        bound.fds.push(fd);
        // The service validates all shared mounts as read-only grants before binding. The
        // workspace itself must remain on the container's private root filesystem.
        if (bound.paths.is_empty() && entry.dev != device) || !(entry.is_dir() || entry.is_reg()) {
            return Err(WorkerError::value("grant_requires_private_root_filesystem"));
        }
        bound.paths.push(resolved);
    }
    let first = bound
        .fds
        .first()
        .ok_or_else(|| WorkerError::value("workspace_requires_existing_directory"))?;
    if !fs.fstat(first)?.is_dir() {
        return Err(WorkerError::value("workspace_requires_existing_directory"));
    }
    Ok(bound)
}

/// Resolve the parent of `path` but keep unlink semantics for a symlink leaf.
///
/// Returns the resolved path and whether it is a directory, or `("", false)` when it does not
/// exist. The filesystem root is refused before anything is inspected.
pub fn selected_path<F: WorkerFs>(fs: &F, path: &str) -> Result<(String, bool), WorkerError> {
    let path =
        crate::sandbox::workspace_paths::normpath(&format!("/{}", path.trim_start_matches('/')));
    if path == "/" {
        return Err(WorkerError::value("filesystem_root"));
    }
    let (parent, name) = match path.rfind('/') {
        Some(0) => ("/".to_string(), path[1..].to_string()),
        Some(index) => (path[..index].to_string(), path[index + 1..].to_string()),
        None => ("/".to_string(), path.clone()),
    };
    let resolved = match canonical(fs, &parent) {
        Ok(resolved) => resolved,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok((String::new(), false)),
        Err(error) => return Err(error.into()),
    };
    let target = if resolved.ends_with('/') {
        format!("{resolved}{name}")
    } else {
        format!("{resolved}/{name}")
    };
    match fs.lstat(OsStr::new(&target), None) {
        Ok(entry) => Ok((target, entry.is_dir())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok((String::new(), false)),
        Err(error) => Err(error.into()),
    }
}

const ACCOUNT_FILE_LIMIT: usize = 1024 * 1024;

/// Stream the records of an account file, split into at most eight fields, to `visit`.
///
/// A missing file has no records. The file is bounded to one MiB and read line by line, so a
/// file of newlines or delimiters cannot grow an unbounded list.
fn for_each_account<F: WorkerFs>(
    fs: &F,
    path: &str,
    visit: &mut dyn FnMut(&[&str]) -> Result<(), WorkerError>,
) -> Result<(), WorkerError> {
    let Some(mut reader) = fs.open_account_file(path)? else {
        return Ok(());
    };
    let mut remaining = ACCOUNT_FILE_LIMIT as i64;
    loop {
        let mut raw = Vec::new();
        let read = (&mut reader)
            .take((remaining + 1) as u64)
            .read_until(b'\n', &mut raw)?;
        if read == 0 {
            return Ok(());
        }
        remaining -= read as i64;
        if remaining < 0 {
            return Err(WorkerError::value("account_file_too_large"));
        }
        let line =
            String::from_utf8(raw).map_err(|_| WorkerError::value("account_file_requires_utf8"))?;
        let record = line.trim_end_matches(['\r', '\n']);
        let fields: Vec<&str> = record.splitn(8, ':').collect();
        visit(&fields)?;
    }
}

fn is_decimal(value: &str) -> bool {
    !value.is_empty() && value.chars().all(|c| c.is_ascii_digit())
}

fn parse_id(value: &str) -> Result<u32, WorkerError> {
    value
        .parse::<u32>()
        .map_err(|_| WorkerError::value(format!("invalid literal for an id: {value:?}")))
}

/// Resolve `user` (`name`, `uid`, `uid:gid`, `name:group`) to a uid, a gid and supplementary
/// groups using the container's own account files.
pub fn user_ids<F: WorkerFs>(fs: &F, user: &str) -> Result<(u32, u32, Vec<u32>), WorkerError> {
    let (name, separator, group) = match user.split_once(':') {
        Some((name, group)) => (name, true, group),
        None => (user, false, ""),
    };
    if is_decimal(name) && separator && is_decimal(group) {
        return Ok((parse_id(name)?, parse_id(group)?, Vec::new()));
    }
    let mut matched: Option<Vec<String>> = None;
    for_each_account(fs, "/etc/passwd", &mut |entry| {
        if matched.is_none() && entry.len() == 7 && (entry[0] == name || entry[2] == name) {
            matched = Some(entry.iter().map(|field| field.to_string()).collect());
        }
        Ok(())
    })?;
    let uid = if is_decimal(name) {
        parse_id(name)?
    } else if let Some(entry) = &matched {
        parse_id(&entry[2])?
    } else {
        return Err(WorkerError::value("unknown_user"));
    };
    let mut gid = match &matched {
        Some(entry) => parse_id(&entry[3])?,
        None => 0,
    };
    let mut groups: Vec<u32> = Vec::new();
    let mut found: Option<String> = None;
    for_each_account(fs, "/etc/group", &mut |entry| {
        if entry.len() != 4 {
            return Ok(());
        }
        if separator {
            if found.is_none() && entry[0] == group {
                found = Some(entry[2].to_string());
            }
        } else if let Some(account) = &matched
            && !account[0].contains(',')
            && format!(",{},", entry[3]).contains(&format!(",{},", account[0]))
        {
            groups.push(parse_id(entry[2])?);
        }
        Ok(())
    })?;
    if separator {
        if is_decimal(group) {
            gid = parse_id(group)?;
        } else if let Some(found) = found {
            gid = parse_id(&found)?;
        } else {
            return Err(WorkerError::value("unknown_group"));
        }
    }
    Ok((uid, gid, groups))
}

/// Counts entry visits against the application's budget.
struct Visits {
    remaining: u64,
}

impl Visits {
    /// Remove a file, or an empty directory. Returns `false` for a directory with contents.
    fn remove_leaf<F: WorkerFs>(
        &mut self,
        fs: &F,
        name: &OsStr,
        parent: Option<&F::Fd>,
    ) -> Result<bool, WorkerError> {
        if self.remaining == 0 {
            return Err(WorkerError::os(
                rustix::io::Errno::TOOBIG.raw_os_error(),
                "removal_entry_limit",
            ));
        }
        self.remaining -= 1;
        let entry = match fs.lstat(name, parent) {
            Ok(entry) => entry,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(true),
            Err(error) => return Err(error.into()),
        };
        if !entry.is_dir() {
            fs.unlink(name, parent)?;
            return Ok(true);
        }
        // Empty directories require no permission to search their contents.
        match fs.rmdir(name, parent) {
            Ok(()) => Ok(true),
            Err(error)
                if error.raw_os_error() == Some(rustix::io::Errno::NOTEMPTY.raw_os_error())
                    || error.raw_os_error() == Some(rustix::io::Errno::EXIST.raw_os_error()) =>
            {
                Ok(false)
            }
            Err(error) => Err(error.into()),
        }
    }
}

/// Remove `path` and everything below it without following links.
///
/// Names are streamed relative to a pinned directory descriptor, so a tree deeper than
/// `PATH_MAX` is removed with at most two descriptors open and no stack or sibling list.
/// `max_entry_visits` bounds the work, counting every revisit of an ancestor; reaching it fails
/// with `E2BIG` and may leave a partially removed tree.
pub fn remove<F: WorkerFs>(fs: &F, path: &str, max_entry_visits: u64) -> Result<(), WorkerError> {
    let mut visits = Visits {
        remaining: max_entry_visits,
    };
    let root = OsStr::new(path);
    if visits.remove_leaf(fs, root, None)? {
        return Ok(());
    }
    let mut ancestors: Vec<OsString> = Vec::new();
    let mut current = fs.open_dir(root, None)?;
    loop {
        let mut child_directory: Option<OsString> = None;
        // The scanner is closed before descent, and no descriptor stack is kept.
        fs.scan(&current, &mut |name| {
            if visits.remove_leaf(fs, name, Some(&current))? {
                Ok(true)
            } else {
                child_directory = Some(name.to_os_string());
                Ok(false)
            }
        })?;
        if let Some(name) = child_directory {
            let next = fs.open_dir(&name, Some(&current))?;
            current = next;
            ancestors.push(name);
        } else if !ancestors.is_empty() {
            let parent = fs.open_dir(OsStr::new(".."), Some(&current))?;
            current = parent;
            if let Some(name) = ancestors.pop() {
                visits.remove_leaf(fs, &name, Some(&current))?;
            }
        } else if visits.remove_leaf(fs, root, None)? {
            return Ok(());
        }
    }
}

/// One removal, as the service hands it to the removal child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoveRequest {
    /// The original (alias-preserving) absolute path to remove.
    pub path: String,
    /// The Docker user to run the removal as.
    pub user: String,
    /// The entry-visit budget.
    pub max_entry_visits: u64,
    /// The CPU-time limit in seconds.
    pub max_cpu_seconds: u64,
}

impl RemoveRequest {
    /// The wire form.
    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "user": self.user,
            "max_entry_visits": self.max_entry_visits,
            "max_cpu_seconds": self.max_cpu_seconds,
        })
    }

    /// Parse the wire form.
    pub fn from_json(value: &Value) -> Result<Self, WorkerError> {
        let text = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| WorkerError::value(format!("missing {key}")))
        };
        let number = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| WorkerError::value(format!("missing {key}")))
        };
        Ok(RemoveRequest {
            path: text("path")?,
            user: text("user")?,
            max_entry_visits: number("max_entry_visits")?,
            max_cpu_seconds: number("max_cpu_seconds")?,
        })
    }
}

/// The process-wide privilege changes a removal child makes before it touches the tree.
pub trait ChildPrivileges {
    /// Cap CPU time with `RLIMIT_CPU` (soft and hard).
    fn set_cpu_limit(&self, seconds: u64) -> io::Result<()>;
    /// Replace the supplementary groups.
    fn set_groups(&self, groups: &[u32]) -> io::Result<()>;
    /// Set the group id.
    fn set_gid(&self, gid: u32) -> io::Result<()>;
    /// Set the user id. Irreversible: it must come last.
    fn set_uid(&self, uid: u32) -> io::Result<()>;
}

/// Run one removal in the (already entered) container as the requested user.
///
/// Returns the outcome the child writes back: `{"ok": true}` or the structured failure. The
/// order is fixed: the CPU limit first, so a refusal to set it never reads account files, then
/// the account lookup as root, then the credential drop, then the removal.
pub fn run_remove_child<F: WorkerFs, P: ChildPrivileges>(
    fs: &F,
    privileges: &P,
    request: &RemoveRequest,
) -> Value {
    let outcome = (|| -> Result<(), WorkerError> {
        privileges.set_cpu_limit(request.max_cpu_seconds)?;
        let (uid, gid, groups) = user_ids(fs, &request.user)?;
        privileges.set_groups(&groups)?;
        privileges.set_gid(gid)?;
        privileges.set_uid(uid)?;
        remove(fs, &request.path, request.max_entry_visits)
    })();
    match outcome {
        Ok(()) => json!({"ok": true}),
        Err(error) => error.to_response(),
    }
}

/// How the worker reaches a container's mount namespace and root.
pub trait NamespaceOps {
    /// An open descriptor. Dropping it closes it.
    type Fd;
    /// Open `/proc/<pid>/ns/mnt`.
    fn open_mount_namespace(&self, pid: u32) -> io::Result<Self::Fd>;
    /// Open `/proc/<pid>/root` as a directory.
    fn open_root(&self, pid: u32) -> io::Result<Self::Fd>;
    /// `setns(fd, CLONE_NEWNS)`.
    fn enter_mount_namespace(&self, fd: &Self::Fd) -> io::Result<()>;
    /// `fchdir(fd)`.
    fn fchdir(&self, fd: &Self::Fd) -> io::Result<()>;
    /// `chroot(".")`.
    fn chroot_here(&self) -> io::Result<()>;
    /// `chdir("/")`.
    fn chdir_root(&self) -> io::Result<()>;
}

/// Enter the container's mount namespace and root, then close the host handles.
///
/// Only the mount namespace is entered, so the worker keeps its own PID, user and network
/// namespaces. A failure to open the root closes the mount handle and never calls `setns`.
pub fn enter_container<N: NamespaceOps>(ops: &N, pid: u32) -> io::Result<()> {
    let mount = ops.open_mount_namespace(pid)?;
    let root = ops.open_root(pid)?;
    ops.enter_mount_namespace(&mount)?;
    ops.fchdir(&root)?;
    ops.chroot_here()?;
    ops.chdir_root()
}

enum Handled {
    Respond(Value),
    Close,
}

fn handle_request<F: WorkerFs>(
    fs: &F,
    bindings: &mut Option<Bindings<F>>,
    line: &str,
) -> Result<Handled, WorkerError> {
    let request: Value = serde_json::from_str(line)
        .map_err(|error| WorkerError::value(format!("invalid worker request: {error}")))?;
    let operation = request
        .get("operation")
        .and_then(Value::as_str)
        .ok_or_else(|| WorkerError::value("invalid_worker_operation"))?;
    match (operation, bindings.is_some()) {
        ("bind", false) => {
            let paths: Vec<String> = request
                .get("paths")
                .and_then(Value::as_array)
                .ok_or_else(|| WorkerError::value("invalid_worker_operation"))?
                .iter()
                .map(|path| path.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| WorkerError::value("invalid_worker_operation"))?;
            let bound = bind_paths(fs, &paths)?;
            let response = json!({"ok": true, "paths": bound.paths});
            *bindings = Some(bound);
            Ok(Handled::Respond(response))
        }
        ("inspect", true) => {
            let bound = bindings.as_ref().expect("checked above");
            bound.validate(fs)?;
            if let Some(alias) = request.get("workspace_root").and_then(Value::as_str)
                && Some(&canonical(fs, alias)?) != bound.paths.first()
            {
                return Err(WorkerError::value("workspace_alias_changed"));
            }
            let path = request
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| WorkerError::value("invalid_worker_operation"))?;
            let (selected, is_directory) = selected_path(fs, path)?;
            Ok(Handled::Respond(
                json!({"ok": true, "path": selected, "is_directory": is_directory}),
            ))
        }
        ("close", _) => Ok(Handled::Close),
        _ => Err(WorkerError::value("invalid_worker_operation")),
    }
}

/// The worker's request loop: one JSON request per line in, one JSON response per line out.
///
/// `close` ends the loop without a response. The pinned descriptors are closed when this
/// returns, including when writing a response fails.
pub fn serve<F: WorkerFs>(
    fs: &F,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> io::Result<()> {
    let mut bindings: Option<Bindings<F>> = None;
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let response = match handle_request(fs, &mut bindings, line.trim_end_matches(['\r', '\n']))
        {
            Ok(Handled::Respond(response)) => response,
            Ok(Handled::Close) => return Ok(()),
            Err(error) => error.to_response(),
        };
        writeln!(output, "{response}")?;
        output.flush()?;
    }
}

/// The real filesystem, as seen from the process that entered the container.
#[derive(Debug, Clone)]
pub struct HostFs {
    account_root: std::path::PathBuf,
}

impl Default for HostFs {
    fn default() -> Self {
        HostFs::new()
    }
}

impl HostFs {
    /// The filesystem rooted at `/`.
    pub fn new() -> Self {
        HostFs {
            account_root: std::path::PathBuf::from("/"),
        }
    }

    /// Read `/etc/passwd` and `/etc/group` below `root` instead of `/`.
    pub fn with_account_root(root: impl Into<std::path::PathBuf>) -> Self {
        HostFs {
            account_root: root.into(),
        }
    }
}

// The widths of `st_dev`, `st_ino` and `st_mode` differ between platforms.
#[allow(clippy::unnecessary_cast)]
fn to_stat(stat: &rustix::fs::Stat) -> FileStat {
    FileStat {
        dev: stat.st_dev as u64,
        ino: stat.st_ino as u64,
        mode: stat.st_mode as u32,
    }
}

fn dir_fd<'a>(parent: Option<&'a rustix::fd::OwnedFd>) -> rustix::fd::BorrowedFd<'a> {
    use rustix::fd::AsFd;
    parent.map_or(rustix::fs::CWD, |fd| fd.as_fd())
}

impl WorkerFs for HostFs {
    type Fd = rustix::fd::OwnedFd;

    fn canonical(&self, path: &str) -> io::Result<String> {
        std::fs::canonicalize(path)?
            .into_os_string()
            .into_string()
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))
    }

    fn root_device(&self) -> io::Result<u64> {
        Ok(to_stat(&rustix::fs::stat("/")?).dev)
    }

    fn lstat(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<FileStat> {
        Ok(to_stat(&rustix::fs::statat(
            dir_fd(parent),
            name,
            rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
        )?))
    }

    fn fstat(&self, fd: &Self::Fd) -> io::Result<FileStat> {
        Ok(to_stat(&rustix::fs::fstat(fd)?))
    }

    fn open_path(&self, path: &str) -> io::Result<Self::Fd> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let access = rustix::fs::OFlags::PATH;
        // Only Linux has `O_PATH`; the service refuses other hosts before it gets here.
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let access = rustix::fs::OFlags::RDONLY;
        Ok(rustix::fs::openat(
            rustix::fs::CWD,
            path,
            access | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?)
    }

    fn open_dir(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<Self::Fd> {
        Ok(rustix::fs::openat(
            dir_fd(parent),
            name,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?)
    }

    fn rmdir(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<()> {
        Ok(rustix::fs::unlinkat(
            dir_fd(parent),
            name,
            rustix::fs::AtFlags::REMOVEDIR,
        )?)
    }

    fn unlink(&self, name: &OsStr, parent: Option<&Self::Fd>) -> io::Result<()> {
        Ok(rustix::fs::unlinkat(
            dir_fd(parent),
            name,
            rustix::fs::AtFlags::empty(),
        )?)
    }

    fn scan(
        &self,
        dir: &Self::Fd,
        visit: &mut dyn FnMut(&OsStr) -> Result<bool, WorkerError>,
    ) -> Result<(), WorkerError> {
        use std::os::unix::ffi::OsStrExt;
        let entries = rustix::fs::Dir::read_from(dir)?;
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name().to_bytes();
            if name == b"." || name == b".." {
                continue;
            }
            if !visit(OsStr::from_bytes(name))? {
                break;
            }
        }
        Ok(())
    }

    fn open_account_file(&self, path: &str) -> Result<Option<Box<dyn BufRead>>, WorkerError> {
        let full = self.account_root.join(path.trim_start_matches('/'));
        // The workload is paused, so reject special files before opening them.
        match std::fs::symlink_metadata(&full) {
            Ok(metadata) if !metadata.file_type().is_file() => {
                return Err(WorkerError::value("account_file_requires_regular_file"));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        }
        let fd = match rustix::fs::openat(
            rustix::fs::CWD,
            &full,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(None),
            Err(error) => return Err(io::Error::from(error).into()),
        };
        if !to_stat(&rustix::fs::fstat(&fd)?).is_reg() {
            return Err(WorkerError::value("account_file_requires_regular_file"));
        }
        Ok(Some(Box::new(io::BufReader::new(std::fs::File::from(fd)))))
    }
}

/// The Linux implementation of [`NamespaceOps`] and [`ChildPrivileges`].
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, Default)]
pub struct LinuxHost;

#[cfg(target_os = "linux")]
impl NamespaceOps for LinuxHost {
    type Fd = rustix::fd::OwnedFd;

    fn open_mount_namespace(&self, pid: u32) -> io::Result<Self::Fd> {
        Ok(rustix::fs::openat(
            rustix::fs::CWD,
            format!("/proc/{pid}/ns/mnt"),
            rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?)
    }

    fn open_root(&self, pid: u32) -> io::Result<Self::Fd> {
        Ok(rustix::fs::openat(
            rustix::fs::CWD,
            format!("/proc/{pid}/root"),
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?)
    }

    fn enter_mount_namespace(&self, fd: &Self::Fd) -> io::Result<()> {
        use rustix::fd::AsFd;
        Ok(rustix::thread::move_into_link_name_space(
            fd.as_fd(),
            Some(rustix::thread::LinkNameSpaceType::Mount),
        )?)
    }

    fn fchdir(&self, fd: &Self::Fd) -> io::Result<()> {
        Ok(rustix::process::fchdir(fd)?)
    }

    fn chroot_here(&self) -> io::Result<()> {
        Ok(rustix::process::chroot(".")?)
    }

    fn chdir_root(&self) -> io::Result<()> {
        Ok(rustix::process::chdir("/")?)
    }
}

#[cfg(target_os = "linux")]
fn checked_id(raw: u32) -> io::Result<u32> {
    // `u32::MAX` is the "no change" sentinel of the id syscalls, never a real id.
    if raw == u32::MAX {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    Ok(raw)
}

#[cfg(target_os = "linux")]
impl ChildPrivileges for LinuxHost {
    fn set_cpu_limit(&self, seconds: u64) -> io::Result<()> {
        Ok(rustix::process::setrlimit(
            rustix::process::Resource::Cpu,
            rustix::process::Rlimit {
                current: Some(seconds),
                maximum: Some(seconds),
            },
        )?)
    }

    fn set_groups(&self, groups: &[u32]) -> io::Result<()> {
        let groups = groups
            .iter()
            .map(|group| checked_id(*group).map(rustix::process::Gid::from_raw))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(rustix::thread::set_thread_groups(&groups)?)
    }

    fn set_gid(&self, gid: u32) -> io::Result<()> {
        Ok(rustix::thread::set_thread_gid(
            rustix::process::Gid::from_raw(checked_id(gid)?),
        )?)
    }

    fn set_uid(&self, uid: u32) -> io::Result<()> {
        Ok(rustix::thread::set_thread_uid(
            rustix::process::Uid::from_raw(checked_id(uid)?),
        )?)
    }
}

/// The entry point of the `docker-removal-worker` executable. Returns the exit code.
///
/// `worker <pid>` serves the bind/inspect protocol on stdin and stdout. `remove <pid>` reads one
/// [`RemoveRequest`] from stdin, performs it as the requested user and writes one outcome line.
/// Both enter the container of process `pid` first, so they only work as root on a Linux Docker
/// host; any other use fails closed with a non-zero exit code.
pub fn worker_main(args: &[String]) -> i32 {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = args;
        eprintln!("The Docker removal worker requires a Linux host.");
        1
    }
    #[cfg(target_os = "linux")]
    {
        let (mode, pid) = match args {
            [mode, pid] if mode == "worker" || mode == "remove" => match pid.parse::<u32>() {
                Ok(pid) => (mode.as_str(), pid),
                Err(_) => return 2,
            },
            _ => {
                eprintln!("usage: docker-removal-worker (worker|remove) <container-pid>");
                return 2;
            }
        };
        let host = LinuxHost;
        let fs = HostFs::new();
        // The request is read before entering the container so a bad request fails early.
        let stdin = io::stdin();
        let mut input = stdin.lock();
        let request = if mode == "remove" {
            let mut raw = String::new();
            let parsed = input.read_to_string(&mut raw).ok().and_then(|_| {
                serde_json::from_str::<Value>(&raw)
                    .ok()
                    .and_then(|value| RemoveRequest::from_json(&value).ok())
            });
            match parsed {
                Some(request) => Some(request),
                None => return 2,
            }
        } else {
            None
        };
        if enter_container(&host, pid).is_err() {
            return 1;
        }
        let stdout = io::stdout();
        let mut output = stdout.lock();
        match request {
            Some(request) => {
                let outcome = run_remove_child(&fs, &host, &request);
                if writeln!(output, "{outcome}")
                    .and_then(|()| output.flush())
                    .is_err()
                {
                    return 1;
                }
                0
            }
            None => i32::from(serve(&fs, &mut input, &mut output).is_err()),
        }
    }
}
