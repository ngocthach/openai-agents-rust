//! What materializing a manifest produced. Mirrors `agents.sandbox.materialization` and
//! `agents.sandbox.util.checksums`.
//!
//! Applying a [`Manifest`](crate::sandbox::Manifest) writes files into the workspace, and a
//! caller often needs to know *which* file ended up where and whether it is the one they meant:
//! a cache that keys on content, an audit trail, a resume that has to notice a changed input.
//! A list of paths cannot answer that, so each written file is reported with the SHA-256 of the
//! bytes that were written.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::Path;

use futures::stream::{StreamExt, TryStreamExt};
use sha2::{Digest, Sha256};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};

/// How much of a file is hashed at a time.
const CHUNK_BYTES: usize = 1024 * 1024;

/// Run `task_factories` with at most `max_concurrency` in flight, returning results in the order
/// the factories were given. Mirrors Python's `gather_in_order`.
///
/// Materializing a manifest is a pile of independent round trips to a backend: running them one
/// at a time is slow, running all of them at once exhausts the backend's connections. This bounds
/// the middle without letting the finishing order leak into the result — a manifest's
/// [`MaterializationResult`] must read the same whatever order the backend happened to complete
/// the writes in.
///
/// Each factory is only called when a slot frees up, so a factory that opens a host file does not
/// open it until the copy is about to run. The first error short-circuits: the remaining factories
/// are never called and the in-flight futures are dropped, which cancels them.
pub async fn gather_in_order<T, F, Fut>(
    task_factories: impl IntoIterator<Item = F>,
    max_concurrency: Option<usize>,
) -> SandboxResult<Vec<T>>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = SandboxResult<T>>,
{
    // Validated before the empty-input shortcut, as Python does: an impossible limit is a
    // configuration bug worth reporting even when there is nothing to run.
    if max_concurrency == Some(0) {
        return Err(SandboxError::new(
            ErrorCode::SandboxConfigInvalid,
            "max_concurrency must be at least 1",
        ));
    }
    let factories: Vec<F> = task_factories.into_iter().collect();
    if factories.is_empty() {
        return Ok(Vec::new());
    }
    let width = max_concurrency.unwrap_or(factories.len()).max(1);
    futures::stream::iter(factories.into_iter().map(|factory| factory()))
        .buffered(width)
        .try_collect()
        .await
}

/// One file written into the sandbox workspace. Mirrors Python's `MaterializedFile`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedFile {
    /// The absolute path inside the sandbox.
    pub path: String,
    /// The SHA-256 of the bytes written, as lowercase hex.
    pub sha256: String,
}

impl MaterializedFile {
    /// A record of `path` holding `contents`.
    pub fn new(path: impl Into<String>, contents: &[u8]) -> Self {
        MaterializedFile {
            path: path.into(),
            sha256: sha256_hex(contents),
        }
    }
}

/// Everything one materialization wrote. Mirrors Python's `MaterializationResult`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaterializationResult {
    /// The files written, in the order they were written.
    pub files: Vec<MaterializedFile>,
}

impl MaterializationResult {
    /// An empty result.
    pub fn new() -> Self {
        MaterializationResult::default()
    }

    /// Record one written file.
    pub fn push(&mut self, file: MaterializedFile) {
        self.files.push(file);
    }

    /// Merge another result into this one.
    pub fn extend(&mut self, other: MaterializationResult) {
        self.files.extend(other.files);
    }

    /// The absolute sandbox paths that were written.
    pub fn paths(&self) -> Vec<String> {
        self.files.iter().map(|file| file.path.clone()).collect()
    }

    /// The checksums by path, for a caller comparing two materializations.
    pub fn checksums(&self) -> BTreeMap<String, String> {
        self.files
            .iter()
            .map(|file| (file.path.clone(), file.sha256.clone()))
            .collect()
    }

    /// Whether nothing was written.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

impl FromIterator<MaterializedFile> for MaterializationResult {
    fn from_iter<I: IntoIterator<Item = MaterializedFile>>(iter: I) -> Self {
        MaterializationResult {
            files: iter.into_iter().collect(),
        }
    }
}

/// The SHA-256 of `bytes`, as lowercase hex. Mirrors Python's `sha256_io`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The SHA-256 of a host file, read in chunks. Mirrors Python's `sha256_file`.
///
/// The file is streamed rather than read whole, so checksumming a large manifest input does not
/// need to hold it in memory.
pub async fn sha256_file(path: &Path) -> SandboxResult<String> {
    use tokio::io::AsyncReadExt;

    let mut handle = tokio::fs::File::open(path).await.map_err(|error| {
        SandboxError::new(
            ErrorCode::LocalChecksumError,
            format!(
                "Failed to open `{}` for checksumming: {error}",
                path.display()
            ),
        )
        .with_context("path", path.display().to_string())
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; CHUNK_BYTES];
    loop {
        let read = handle.read(&mut buffer).await.map_err(|error| {
            SandboxError::new(
                ErrorCode::LocalChecksumError,
                format!(
                    "Failed to read `{}` while checksumming: {error}",
                    path.display()
                ),
            )
            .with_context("path", path.display().to_string())
        })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_bytes_the_way_sha256sum_does() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[tokio::test]
    async fn hashes_a_host_file_in_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload");
        // Larger than one chunk, so the streaming loop runs more than once.
        let contents = vec![b'x'; CHUNK_BYTES + 17];
        tokio::fs::write(&path, &contents).await.unwrap();
        assert_eq!(sha256_file(&path).await.unwrap(), sha256_hex(&contents));
    }

    #[tokio::test]
    async fn reports_a_missing_file_as_a_checksum_error() {
        let error = sha256_file(Path::new("/nonexistent/for/checksum"))
            .await
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::LocalChecksumError);
    }

    #[test]
    fn collects_paths_and_checksums() {
        let mut result = MaterializationResult::new();
        result.push(MaterializedFile::new("/w/a", b"a"));
        result.extend(MaterializationResult::from_iter([MaterializedFile::new(
            "/w/b", b"b",
        )]));
        assert_eq!(result.paths(), vec!["/w/a", "/w/b"]);
        assert_eq!(result.checksums().len(), 2);
        assert!(!result.is_empty());
    }
}
