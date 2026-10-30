//! Value types shared by the sandbox modules. Mirrors `agents.sandbox.types`.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A sandbox user account used for model-facing operations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct User {
    /// The account name.
    pub name: String,
}

impl User {
    /// A user with the given account name.
    pub fn new(name: impl Into<String>) -> Self {
        User { name: name.into() }
    }
}

impl From<&str> for User {
    fn from(value: &str) -> Self {
        User::new(value)
    }
}

impl From<String> for User {
    fn from(value: String) -> Self {
        User::new(value)
    }
}

/// A sandbox group account and its members.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    /// The group name.
    pub name: String,
    /// Users belonging to the group.
    pub users: Vec<User>,
}

impl Group {
    /// A group with the given name and no members.
    pub fn new(name: impl Into<String>) -> Self {
        Group {
            name: name.into(),
            users: Vec::new(),
        }
    }

    /// Add a member.
    pub fn with_user(mut self, user: impl Into<User>) -> Self {
        self.users.push(user.into());
        self
    }
}

/// Permission bits for one `rwx` triplet.
pub mod file_mode {
    /// Read permission.
    pub const READ: u32 = 0b100;
    /// Write permission.
    pub const WRITE: u32 = 0b010;
    /// Execute (or directory traversal) permission.
    pub const EXEC: u32 = 0b001;
    /// Read, write and execute.
    pub const ALL: u32 = 0b111;
    /// No permissions.
    pub const NONE: u32 = 0;
}

/// POSIX permission triplets for a manifest entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permissions {
    /// Owner triplet (0-7).
    pub owner: u32,
    /// Group triplet (0-7).
    pub group: u32,
    /// Other triplet (0-7).
    pub other: u32,
    /// Whether the entry is a directory.
    pub directory: bool,
}

impl Default for Permissions {
    fn default() -> Self {
        Permissions {
            owner: file_mode::ALL,
            group: file_mode::READ | file_mode::EXEC,
            other: file_mode::READ | file_mode::EXEC,
            directory: false,
        }
    }
}

impl Permissions {
    /// Permissions from explicit triplets.
    pub fn new(owner: u32, group: u32, other: u32) -> Self {
        Permissions {
            owner: owner & 0o7,
            group: group & 0o7,
            other: other & 0o7,
            directory: false,
        }
    }

    /// Mark the entry as a directory.
    pub fn as_directory(mut self, directory: bool) -> Self {
        self.directory = directory;
        self
    }

    /// The permission bits as a numeric mode (without the file-type bits).
    pub fn to_mode(self) -> u32 {
        ((self.owner & 0o7) << 6) | ((self.group & 0o7) << 3) | (self.other & 0o7)
    }

    /// Permissions from a numeric mode, as `stat` reports it.
    ///
    /// Mirrors Python's `Permissions.from_mode`: the three triplets come from the low nine bits
    /// and the directory flag from `S_IFDIR`, so a mode read back from the sandbox round-trips
    /// through [`to_mode`](Self::to_mode).
    pub fn from_mode(mode: u32) -> Self {
        /// `stat.S_IFDIR`.
        const S_IFDIR: u32 = 0o040000;
        Permissions {
            owner: (mode >> 6) & 0o7,
            group: (mode >> 3) & 0o7,
            other: mode & 0o7,
            directory: mode & S_IFDIR != 0,
        }
    }

    /// The mode as a four-digit octal string, as `chmod` expects it.
    pub fn octal_string(self) -> String {
        format!("{:04o}", self.to_mode())
    }

    /// Parse the leading ten characters of an `ls -l` mode field.
    pub fn from_ls_field(field: &str) -> Option<Self> {
        let chars: Vec<char> = field.chars().collect();
        let chars = match chars.len() {
            11 if matches!(chars[10], '@' | '+' | '.') => &chars[..10],
            10 => &chars[..10],
            _ => return None,
        };
        let directory = match chars[0] {
            'd' => true,
            '-' | 'l' => false,
            _ => return None,
        };
        let triplet = |offset: usize| -> u32 {
            let mut mask = 0;
            if chars[offset] == 'r' {
                mask |= file_mode::READ;
            }
            if chars[offset + 1] == 'w' {
                mask |= file_mode::WRITE;
            }
            if matches!(chars[offset + 2], 'x' | 's' | 't') {
                mask |= file_mode::EXEC;
            }
            mask
        };
        Some(Permissions {
            owner: triplet(1),
            group: triplet(4),
            other: triplet(7),
            directory,
        })
    }
}

impl fmt::Display for Permissions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let triplet = |bits: u32| {
            format!(
                "{}{}{}",
                if bits & file_mode::READ != 0 {
                    "r"
                } else {
                    "-"
                },
                if bits & file_mode::WRITE != 0 {
                    "w"
                } else {
                    "-"
                },
                if bits & file_mode::EXEC != 0 {
                    "x"
                } else {
                    "-"
                },
            )
        };
        write!(
            f,
            "{}{}{}{}",
            if self.directory { "d" } else { "-" },
            triplet(self.owner),
            triplet(self.group),
            triplet(self.other),
        )
    }
}

/// The result of running a command inside a sandbox session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecResult {
    /// Raw standard output.
    pub stdout: Vec<u8>,
    /// Raw standard error.
    pub stderr: Vec<u8>,
    /// The process exit code.
    pub exit_code: i32,
}

impl ExecResult {
    /// True when the command exited with code zero.
    pub fn ok(&self) -> bool {
        self.exit_code == 0
    }

    /// Standard output decoded as lossy UTF-8.
    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// Standard error decoded as lossy UTF-8.
    pub fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

/// One entry returned by [`SandboxSession::ls`](crate::sandbox::SandboxSession::ls).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Entry name, relative to the listed directory.
    pub name: String,
    /// Absolute sandbox path of the entry.
    pub path: String,
    /// Size in bytes, when reported.
    pub size: Option<u64>,
    /// Permissions, when they could be parsed.
    pub permissions: Option<Permissions>,
    /// Whether the entry is a directory.
    pub is_dir: bool,
}

/// Parse the output of `ls -la` into [`FileEntry`] values.
///
/// `base` is the absolute sandbox path that was listed; `.` and `..` are skipped.
pub fn parse_ls_la(output: &str, base: &str) -> Vec<FileEntry> {
    let mut entries = Vec::new();
    for line in output.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with("total ") {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(mode) = parts.next() else { continue };
        let Some(permissions) = Permissions::from_ls_field(mode) else {
            continue;
        };
        // links, owner, group, size, then the date fields.
        let fields: Vec<&str> = parts.collect();
        if fields.len() < 5 {
            continue;
        }
        let size = fields.get(3).and_then(|s| s.parse::<u64>().ok());
        // The name starts after the mode plus the next eight whitespace-separated fields.
        let name = name_after_fields(line, 8).unwrap_or_default();
        let name = name.split(" -> ").next().unwrap_or(&name).to_string();
        if name.is_empty() || name == "." || name == ".." {
            continue;
        }
        let path = if base.ends_with('/') {
            format!("{base}{name}")
        } else {
            format!("{base}/{name}")
        };
        entries.push(FileEntry {
            name,
            path,
            size,
            permissions: Some(permissions),
            is_dir: permissions.directory,
        });
    }
    entries
}

/// Return the remainder of `line` after `count` whitespace-separated fields.
fn name_after_fields(line: &str, count: usize) -> Option<String> {
    let mut rest = line;
    for _ in 0..count {
        rest = rest.trim_start();
        let end = rest.find(char::is_whitespace)?;
        rest = &rest[end..];
    }
    Some(rest.trim_start().to_string())
}

/// A reachable host endpoint for a port exposed by a sandbox.
///
/// Mirrors `agents.sandbox.types.ExposedPortEndpoint`. A sandbox port is almost never reachable
/// at the number the workload bound: the provider maps it, usually onto a TLS-terminated name
/// with a signed query string, so the host side needs all four pieces to build a URL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExposedPortEndpoint {
    /// The host name or address the port is reachable at.
    pub host: String,
    /// The port on `host`, which is rarely the sandbox-side port.
    pub port: u16,
    /// Whether the endpoint terminates TLS.
    pub tls: bool,
    /// A query string the provider requires, such as a preview token. Never includes the `?`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub query: String,
}

impl ExposedPortEndpoint {
    /// An endpoint with no query string.
    pub fn new(host: impl Into<String>, port: u16, tls: bool) -> Self {
        ExposedPortEndpoint {
            host: host.into(),
            port,
            tls,
            query: String::new(),
        }
    }

    /// Attach a query string, with or without a leading `?`.
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = query.into();
        self
    }

    /// The URL for `scheme`, which must be `http` or `ws`.
    ///
    /// The default port for the scheme is omitted, matching Python's `url_for`.
    pub fn url_for(&self, scheme: &str) -> crate::sandbox::errors::SandboxResult<String> {
        use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxOp};

        let normalized = scheme.to_ascii_lowercase();
        let prefix = match (normalized.as_str(), self.tls) {
            ("http", false) => "http",
            ("http", true) => "https",
            ("ws", false) => "ws",
            ("ws", true) => "wss",
            _ => {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    "scheme must be either 'http' or 'ws'",
                )
                .with_op(SandboxOp::ResolveExposedPort)
                .with_retryable(false));
            }
        };
        let default_port = if self.tls { 443 } else { 80 };
        // A bare IPv6 address has to be bracketed before it can carry a port.
        let host = if self.host.contains(':') && !self.host.starts_with('[') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        let base = if self.port == default_port {
            format!("{prefix}://{host}/")
        } else {
            format!("{prefix}://{host}:{}/", self.port)
        };
        let query = self.query.strip_prefix('?').unwrap_or(&self.query);
        if query.is_empty() {
            Ok(base)
        } else {
            Ok(format!("{base}?{query}"))
        }
    }
}

/// Why a port could not be resolved. Mirrors Python's `reason` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExposedPortUnavailable {
    /// The port was never declared on the client options, so the backend never exposed it.
    NotConfigured,
    /// The backend could not, or cannot, hand out an endpoint for the port.
    BackendUnavailable,
}

impl ExposedPortUnavailable {
    /// The `reason` string Python puts in the error context.
    pub fn as_str(self) -> &'static str {
        match self {
            ExposedPortUnavailable::NotConfigured => "not_configured",
            ExposedPortUnavailable::BackendUnavailable => "backend_unavailable",
        }
    }
}

/// The error Python raises as `ExposedPortUnavailableError`.
///
/// A port that was never declared cannot start working, so that case is explicitly
/// non-retryable; a backend that is merely not ready yet leaves retryability unset.
pub fn exposed_port_unavailable(
    backend: &str,
    port: u16,
    exposed_ports: &[u16],
    reason: ExposedPortUnavailable,
) -> crate::sandbox::errors::SandboxError {
    use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxOp};

    let message = match reason {
        ExposedPortUnavailable::NotConfigured => {
            format!("port {port} is not configured for host exposure")
        }
        ExposedPortUnavailable::BackendUnavailable => {
            format!("port {port} could not be resolved for host exposure")
        }
    };
    let mut error = SandboxError::new(ErrorCode::ExposedPortUnavailable, message)
        .with_op(SandboxOp::ResolveExposedPort)
        .with_context("backend", backend)
        .with_context("port", port.to_string())
        .with_context("reason", reason.as_str())
        .with_context(
            "exposed_ports",
            exposed_ports
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    if reason == ExposedPortUnavailable::NotConfigured {
        error = error.with_retryable(false);
    }
    error
}

#[cfg(test)]
mod tests {

    #[test]
    fn round_trips_a_numeric_mode() {
        let permissions = Permissions::new(7, 5, 4);
        assert_eq!(Permissions::from_mode(permissions.to_mode()), permissions);
        let directory = Permissions::new(7, 0, 0).as_directory(true);
        assert_eq!(
            Permissions::from_mode(directory.to_mode() | 0o040000),
            directory
        );
    }
    use super::*;

    #[test]
    fn permissions_round_trip() {
        let p = Permissions::from_ls_field("drwxr-xr-x").unwrap();
        assert!(p.directory);
        assert_eq!(p.to_mode(), 0o755);
        assert_eq!(p.octal_string(), "0755");
        assert_eq!(p.to_string(), "drwxr-xr-x");
        assert_eq!(Permissions::default().to_mode(), 0o755);
    }

    #[test]
    fn parses_ls_output() {
        let out = "total 8\n\
                   drwxr-xr-x  3 me  staff   96 Jan  1 10:00 .\n\
                   drwxr-xr-x  5 me  staff  160 Jan  1 10:00 ..\n\
                   -rw-r--r--  1 me  staff   12 Jan  1 10:00 task.md\n\
                   drwxr-xr-x  2 me  staff   64 Jan  1 10:00 src\n";
        let entries = parse_ls_la(out, "/workspace/repo");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "task.md");
        assert_eq!(entries[0].path, "/workspace/repo/task.md");
        assert_eq!(entries[0].size, Some(12));
        assert!(!entries[0].is_dir);
        assert!(entries[1].is_dir);
    }
}
