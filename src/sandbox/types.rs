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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    /// Parse the mode field of an `ls -l` line strictly.
    ///
    /// Mirrors Python's `Permissions.from_str`. A trailing `.` (SELinux context), `+` (ACL) or
    /// `@` (macOS extended attributes) is dropped first. The special bits are only valid where
    /// the kernel puts them: `s`/`S` in the owner and group execute positions, `t`/`T` in the
    /// other execute position. A line that puts one anywhere else is not an `ls` listing, and
    /// reading it leniently would hand back a permission nobody granted.
    pub fn try_from_ls_field(field: &str) -> Result<Self, String> {
        let mut chars: Vec<char> = field.chars().collect();
        if chars.len() == 11 && matches!(chars[10], '@' | '+' | '.') {
            chars.pop();
        }
        if chars.len() != 10 {
            return Err(format!("invalid permissions string length: {field:?}"));
        }
        let directory = chars[0] == 'd';
        if !matches!(chars[0], 'd' | '-') {
            return Err(format!("invalid permissions type: {field:?}"));
        }
        let triplet = |offset: usize, special: (char, char)| -> Result<u32, String> {
            let text: String = chars[offset..offset + 3].iter().collect();
            let mut mask = 0;
            match chars[offset] {
                'r' => mask |= file_mode::READ,
                '-' => {}
                _ => return Err(format!("invalid read flag: {text:?}")),
            }
            match chars[offset + 1] {
                'w' => mask |= file_mode::WRITE,
                '-' => {}
                _ => return Err(format!("invalid write flag: {text:?}")),
            }
            let flag = chars[offset + 2];
            if flag == 'x' || flag == special.0 {
                mask |= file_mode::EXEC;
            } else if flag != '-' && flag != special.1 {
                return Err(format!("invalid exec flag: {text:?}"));
            }
            Ok(mask)
        };
        Ok(Permissions {
            owner: triplet(1, ('s', 'S'))?,
            group: triplet(4, ('s', 'S'))?,
            other: triplet(7, ('t', 'T'))?,
            directory,
        })
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
    /// Whether the entry is a symbolic link.
    ///
    /// Mirrors Python's `EntryKind.SYMLINK`. Archive extraction needs it: a member whose path
    /// descends through a link that is *already* in the destination would be written outside the
    /// workspace, and the listing is the only thing that says so before the write happens.
    pub is_symlink: bool,
}

/// What kind of entry a listing line describes. Mirrors Python's `EntryKind` for listings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ListingKind {
    /// A regular file.
    File,
    /// A directory.
    Directory,
    /// A symbolic link.
    Symlink,
    /// A device node, socket, pipe or anything else `ls` can show.
    Other,
}

/// One parsed `ls -la` line, with the columns [`FileEntry`] does not carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListingEntry {
    /// The entry, as [`SandboxSession::ls`](crate::sandbox::SandboxSession::ls) returns it.
    pub entry: FileEntry,
    /// The owning user, as `ls` printed it.
    pub owner: String,
    /// The owning group, as `ls` printed it.
    pub group: String,
    /// What kind of entry the line describes.
    pub kind: ListingKind,
}

/// Parse `ls -la` output, failing on a mode field that is not a valid `ls` mode.
///
/// Mirrors Python's `parse_ls_la`. Lines `ls` could not have produced (too few fields, a size
/// that is not a number) are skipped; a mode field that is present but malformed is an error.
pub fn try_parse_ls_la(output: &str, base: &str) -> Result<Vec<ListingEntry>, String> {
    let mut entries = Vec::new();
    for raw_line in output.lines() {
        let line = raw_line.trim_matches('\n');
        if line.is_empty() || line.starts_with("total") {
            continue;
        }
        // Typical coreutils format:
        //   drwxr-xr-x  2 root root     4096 Jan  1 00:00 dirname
        //   lrwxrwxrwx  1 root root       12 Jan  1 00:00 link -> target
        let parts = split_max(line, 8);
        if parts.len() < 9 {
            continue;
        }
        let mode = parts[0];
        let owner = parts[2].to_string();
        let group = parts[3].to_string();
        let type_char = mode.chars().next().unwrap_or('-');
        let size;
        let name;
        if matches!(type_char, 'c' | 'b') {
            // A device reports an identifier in place of the size column, in one of two shapes.
            // `stat` reports size 0 for them.
            size = 0;
            if parts[4].ends_with(',') {
                // GNU coreutils prints "major, minor", which occupies two fields and shifts
                // every following field by one.
                let shifted = split_max(line, 9);
                if shifted.len() < 10 {
                    continue;
                }
                name = shifted[9].to_string();
            } else {
                // BSD ls prints a single hexadecimal identifier, e.g. 0x3000002.
                name = parts[8].to_string();
            }
        } else {
            let Ok(parsed) = parts[4].parse::<u64>() else {
                continue;
            };
            size = parsed;
            name = parts[8].to_string();
        }
        let kind = match type_char {
            'd' => ListingKind::Directory,
            '-' => ListingKind::File,
            'l' => ListingKind::Symlink,
            _ => ListingKind::Other,
        };
        // Permissions only track rwx bits and directory-ness; for symlink and other entries the
        // leading type marker is normalized to `-` so the rwx bits survive.
        let mut mode_text = mode.to_string();
        if !matches!(type_char, 'd' | '-') && mode_text.len() >= 2 {
            mode_text = format!("-{}", &mode_text[type_char.len_utf8()..]);
        }
        // Only a symlink prints " -> target"; in a regular file's name the arrow is the name.
        let name = if kind == ListingKind::Symlink {
            name.split(" -> ").next().unwrap_or(&name).to_string()
        } else {
            name
        };
        if name == "." || name == ".." {
            continue;
        }
        let permissions = Permissions::try_from_ls_field(&mode_text)?;
        let path = if name.starts_with('/') {
            name.clone()
        } else if base == "/" {
            format!("/{name}")
        } else {
            format!("{}/{name}", base.trim_end_matches('/'))
        };
        let short_name = path.rsplit('/').next().unwrap_or(&path).to_string();
        entries.push(ListingEntry {
            entry: FileEntry {
                name: short_name,
                path,
                size: Some(size),
                permissions: Some(permissions),
                is_dir: permissions.directory,
                is_symlink: kind == ListingKind::Symlink,
            },
            owner,
            group,
            kind,
        });
    }
    Ok(entries)
}

/// Split on whitespace into at most `max_splits + 1` fields, keeping the rest of the line (inner
/// spacing included) in the last one. Mirrors `str.split(maxsplit=n)`.
fn split_max(line: &str, max_splits: usize) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut rest = line.trim_start();
    while fields.len() < max_splits && !rest.is_empty() {
        match rest.find(char::is_whitespace) {
            Some(end) => {
                fields.push(&rest[..end]);
                rest = rest[end..].trim_start();
            }
            None => {
                fields.push(rest);
                rest = "";
            }
        }
    }
    if !rest.is_empty() {
        fields.push(rest);
    }
    fields
}

/// Parse the output of `ls -la` into [`FileEntry`] values.
///
/// `base` is the absolute sandbox path that was listed; `.` and `..` are skipped. A line whose
/// mode field is malformed is skipped here; use [`try_parse_ls_la`] to be told about it.
pub fn parse_ls_la(output: &str, base: &str) -> Vec<FileEntry> {
    output
        .lines()
        .filter_map(|line| try_parse_ls_la(line, base).ok())
        .flatten()
        .map(|listing| listing.entry)
        .collect()
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

/// The default ceiling on the bytes [`SandboxSession::extract`] accepts.
///
/// [`SandboxSession::extract`]: crate::sandbox::SandboxSession::extract
pub const DEFAULT_MAX_ARCHIVE_INPUT_BYTES: u64 = 1024 * 1024 * 1024;

/// The default ceiling on the bytes an archive may declare it extracts.
pub const DEFAULT_MAX_ARCHIVE_EXTRACTED_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The default ceiling on the number of extractable archive members.
pub const DEFAULT_MAX_ARCHIVE_MEMBERS: u64 = 100_000;

/// Resource limits for sandbox archive extraction. Mirrors Python's `SandboxArchiveLimits`.
///
/// Each limit is independently nullable: `None` disables that one ceiling. The defaults are the
/// Python ones, and [`SandboxRunConfig::archive_limits`] is `None` by default, so an application
/// that sets nothing keeps the SDK out of the way.
///
/// [`SandboxRunConfig::archive_limits`]: crate::sandbox::SandboxRunConfig::archive_limits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxArchiveLimits {
    /// Maximum archive input bytes accepted by [`SandboxSession::extract`].
    ///
    /// [`SandboxSession::extract`]: crate::sandbox::SandboxSession::extract
    pub max_input_bytes: Option<u64>,
    /// Maximum bytes an archive may declare that it extracts.
    pub max_extracted_bytes: Option<u64>,
    /// Maximum number of extractable archive members.
    pub max_members: Option<u64>,
}

impl Default for SandboxArchiveLimits {
    fn default() -> Self {
        SandboxArchiveLimits {
            max_input_bytes: Some(DEFAULT_MAX_ARCHIVE_INPUT_BYTES),
            max_extracted_bytes: Some(DEFAULT_MAX_ARCHIVE_EXTRACTED_BYTES),
            max_members: Some(DEFAULT_MAX_ARCHIVE_MEMBERS),
        }
    }
}

impl SandboxArchiveLimits {
    /// The SDK default limits.
    pub fn new() -> Self {
        SandboxArchiveLimits::default()
    }

    /// Set the maximum accepted archive input size.
    pub fn with_max_input_bytes(mut self, bytes: Option<u64>) -> Self {
        self.max_input_bytes = bytes;
        self
    }

    /// Set the maximum declared extracted size.
    pub fn with_max_extracted_bytes(mut self, bytes: Option<u64>) -> Self {
        self.max_extracted_bytes = bytes;
        self
    }

    /// Set the maximum number of extractable members.
    pub fn with_max_members(mut self, members: Option<u64>) -> Self {
        self.max_members = members;
        self
    }

    /// Reject a limit that is set to zero. Mirrors Python's `SandboxArchiveLimits.validate`.
    pub fn validate(&self) -> crate::sandbox::errors::SandboxResult<()> {
        use crate::sandbox::errors::{ErrorCode, SandboxError};

        for (name, value) in [
            ("max_input_bytes", self.max_input_bytes),
            ("max_extracted_bytes", self.max_extracted_bytes),
            ("max_members", self.max_members),
        ] {
            if value == Some(0) {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    format!("archive_limits.{name} must be at least 1"),
                ));
            }
        }
        Ok(())
    }
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
