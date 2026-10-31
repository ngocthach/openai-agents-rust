//! In-memory sandbox session shared by the apply_patch ports. Mirrors
//! `tests/sandbox/_apply_patch_test_session.py`.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::types::{ExecResult, User};

/// How the store decides that two spellings are one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Store {
    /// Every distinct spelling is a distinct file (Linux, macOS default).
    Sensitive,
    /// Case-folding APFS: `notes.txt` and `Notes.txt` are one entry that keeps its first name.
    CaseFolding,
    /// APFS: case and Unicode normalization are both folded.
    NormalizationFolding,
    /// `/workspace/alias` resolves to `/workspace/real`.
    ParentAlias,
}

#[derive(Debug, Default)]
pub struct State {
    pub files: BTreeMap<String, Vec<u8>>,
    pub directories: BTreeSet<String>,
    pub symlinks: BTreeMap<String, String>,
    pub mkdir_calls: Vec<(String, bool)>,
    pub rm_calls: Vec<(String, bool)>,
    pub mv_calls: Vec<(String, String)>,
    pub fail_next_write: bool,
    pub concurrent_source: Option<String>,
    pub fail_moves: bool,
    /// Report every read as missing, as a backend with an exclusive create would for a probe.
    pub reads_always_missing: bool,
    /// Missing-file reads carry a provider-private path in their message.
    pub provider_private_not_found: bool,
    /// `(operation, user name)` for every call that carried a user.
    pub user_calls: Vec<(String, Option<String>)>,
}

#[derive(Debug)]
pub struct ApplyPatchSession {
    manifest: Manifest,
    store: Store,
    state: Mutex<State>,
}

fn nfc(text: &str) -> String {
    // Enough normalization for the accented pair the Python test uses.
    text.replace("e\u{301}", "\u{e9}")
}

impl ApplyPatchSession {
    pub fn new(store: Store) -> Self {
        Self::with_manifest(store, Manifest::new().with_root("/workspace"))
    }

    pub fn with_manifest(store: Store, manifest: Manifest) -> Self {
        ApplyPatchSession {
            manifest,
            store,
            state: Mutex::new(State::default()),
        }
    }

    pub fn put(&self, path: &str, contents: &[u8]) {
        self.state
            .lock()
            .unwrap()
            .files
            .insert(path.to_string(), contents.to_vec());
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        f(&mut self.state.lock().unwrap())
    }

    fn note(&self, operation: &str, user: Option<&User>) {
        let name = user.map(|user| user.name.clone());
        self.with(|state| state.user_calls.push((operation.to_string(), name)));
    }

    /// The user names the session saw for `operation`.
    pub fn users_for(&self, operation: &str) -> Vec<Option<String>> {
        self.with(|state| {
            state
                .user_calls
                .iter()
                .filter(|(op, _)| op == operation)
                .map(|(_, user)| user.clone())
                .collect()
        })
    }

    pub fn files(&self) -> BTreeMap<String, Vec<u8>> {
        self.with(|state| state.files.clone())
    }

    pub fn mv_calls(&self) -> Vec<(String, String)> {
        self.with(|state| state.mv_calls.clone())
    }

    pub fn rm_calls(&self) -> Vec<(String, bool)> {
        self.with(|state| state.rm_calls.clone())
    }

    /// The key this store holds `path` under.
    fn stored(&self, state: &State, path: &str) -> String {
        match self.store {
            Store::Sensitive => path.to_string(),
            Store::CaseFolding => {
                let folded = path.to_lowercase();
                state
                    .files
                    .keys()
                    .find(|key| key.to_lowercase() == folded)
                    .cloned()
                    .unwrap_or_else(|| path.to_string())
            }
            Store::NormalizationFolding => {
                let folded = nfc(path).to_lowercase();
                state
                    .files
                    .keys()
                    .find(|key| nfc(key).to_lowercase() == folded)
                    .cloned()
                    .unwrap_or_else(|| path.to_string())
            }
            Store::ParentAlias => match path.strip_prefix("/workspace/alias/") {
                Some(rest) => format!("/workspace/real/{rest}"),
                None => path.to_string(),
            },
        }
    }

    fn resolved(&self, state: &State, path: &str) -> String {
        let mut stored = self.stored(state, path);
        let mut seen = BTreeSet::new();
        while let Some(target) = state.symlinks.get(&stored) {
            if !seen.insert(stored.clone()) {
                break;
            }
            stored = self.stored(state, target);
        }
        stored
    }
}

fn not_found(path: &str) -> SandboxError {
    SandboxError::new(ErrorCode::WorkspaceReadNotFound, format!("missing {path}"))
}

#[async_trait]
impl SandboxSession for ApplyPatchSession {
    fn id(&self) -> &str {
        "apply-patch"
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn backend_id(&self) -> &str {
        "apply-patch-test"
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
    async fn exec(&self, _request: ExecRequest) -> SandboxResult<ExecResult> {
        panic!("exec() should not be called");
    }
    async fn hydrate_workspace(&self, _archive: &[u8]) -> SandboxResult<()> {
        Ok(())
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        let normalized = self.normalize_path(path)?;
        self.with(|state| {
            if state.reads_always_missing {
                return Err(not_found(&normalized));
            }
            let stored = match self.store {
                // The case-folding and alias stores look the entry up through `stored`.
                Store::Sensitive => normalized.clone(),
                _ => self.stored(state, &normalized),
            };
            match state.files.get(&stored) {
                Some(bytes) => Ok(bytes.clone()),
                None if state.provider_private_not_found => Err(SandboxError::new(
                    ErrorCode::WorkspaceReadNotFound,
                    format!("missing /provider/private/root{normalized}"),
                )),
                None => Err(not_found(&normalized)),
            }
        })
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        let normalized = self.normalize_path_for_write(path)?;
        self.with(|state| {
            if state.fail_next_write {
                state.fail_next_write = false;
                return Err(SandboxError::new(
                    ErrorCode::ExecTransportError,
                    "ConnectionError: sandbox write failed",
                ));
            }
            if state.directories.contains(&normalized) {
                return Err(SandboxError::new(
                    ErrorCode::WorkspaceWriteTypeError,
                    format!("IsADirectoryError: {normalized}"),
                ));
            }
            let stored = match self.store {
                Store::Sensitive => normalized,
                _ => self.stored(state, &normalized),
            };
            state.files.insert(stored, contents.to_vec());
            Ok(())
        })
    }

    async fn write_new_file(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&User>,
    ) -> SandboxResult<()> {
        let normalized = self.normalize_path_for_write(path)?;
        if self.with(|state| state.files.contains_key(&normalized)) {
            return Err(SandboxError::new(
                ErrorCode::ApplyPatchInvalidPath,
                format!("FileExistsError: `{normalized}` already exists"),
            ));
        }
        // Real backends create the parents inside the primitive, so record that here too.
        if let Some((parent, _)) = normalized.rsplit_once('/') {
            self.mkdir_as(parent, true, user).await?;
        }
        self.write_file_as(&normalized, contents, user).await
    }

    async fn read_file_as(&self, path: &str, user: Option<&User>) -> SandboxResult<Vec<u8>> {
        self.note("read", user);
        self.read_file(path).await
    }

    async fn write_file_as(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&User>,
    ) -> SandboxResult<()> {
        self.note("write", user);
        self.write_file(path, contents).await
    }

    async fn mkdir_as(&self, path: &str, parents: bool, user: Option<&User>) -> SandboxResult<()> {
        self.note("mkdir", user);
        self.mkdir(path, parents).await
    }

    async fn rm_as(&self, path: &str, recursive: bool, user: Option<&User>) -> SandboxResult<()> {
        self.note("rm", user);
        self.rm(path, recursive).await
    }

    async fn mv_as(
        &self,
        source: &str,
        destination: &str,
        user: Option<&User>,
    ) -> SandboxResult<()> {
        self.note("mv", user);
        self.mv(source, destination).await
    }

    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        let normalized = self.normalize_path_for_write(path)?;
        self.with(|state| state.mkdir_calls.push((normalized, parents)));
        Ok(())
    }

    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        let normalized = self.normalize_path_for_write(path)?;
        self.with(|state| {
            let stored = match self.store {
                Store::Sensitive => normalized,
                _ => self.stored(state, &normalized),
            };
            state.rm_calls.push((stored.clone(), recursive));
            state.files.remove(&stored);
        });
        Ok(())
    }

    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        let source_path = self.normalize_path_for_write(source)?;
        let destination_path = self.normalize_path_for_write(destination)?;
        self.with(|state| {
            if state.fail_moves {
                if let Some(concurrent) = state.concurrent_source.clone() {
                    state
                        .files
                        .insert(concurrent, b"written by someone else\n".to_vec());
                }
                return Err(SandboxError::new(
                    ErrorCode::ExecTransportError,
                    "ConnectionError: sandbox move failed",
                ));
            }
            if state.directories.contains(&destination_path) {
                // A real `mv` would move the source inside this directory and report success.
                return Err(SandboxError::new(
                    ErrorCode::ExecNonZero,
                    format!("IsADirectoryError: {destination_path}"),
                ));
            }
            if self.store == Store::ParentAlias {
                let same =
                    self.stored(state, &source_path) == self.stored(state, &destination_path);
                let leaf = |p: &str| p.rsplit('/').next().unwrap_or("").to_string();
                if source_path != destination_path
                    && leaf(&source_path) == leaf(&destination_path)
                    && same
                {
                    return Err(SandboxError::new(
                        ErrorCode::ExecNonZero,
                        "mv: source and destination are the same file",
                    ));
                }
            }
            let stored_source = self.stored(state, &source_path);
            let Some(payload) = state.files.remove(&stored_source) else {
                return Err(SandboxError::new(
                    ErrorCode::ExecNonZero,
                    format!("FileNotFoundError: {stored_source}"),
                ));
            };
            let stored_destination = self.stored(state, &destination_path);
            let destination_exists = state.files.contains_key(&stored_destination);
            if destination_exists && stored_destination != stored_source {
                // APFS keeps an existing entry's spelling when a different inode replaces it
                // through a case-variant path.
                state.files.insert(stored_destination.clone(), payload);
            } else {
                state.files.remove(&stored_destination);
                state.files.insert(destination_path.clone(), payload);
            }
            state.mv_calls.push((stored_source, destination_path));
            Ok(())
        })
    }

    async fn same_file(
        &self,
        left: &str,
        right: &str,
        follow_symlinks: bool,
        user: Option<&User>,
    ) -> SandboxResult<bool> {
        self.note("same_file", user);
        let left = self.normalize_path(left)?;
        let right = self.normalize_path(right)?;
        Ok(self.with(|state| {
            if !follow_symlinks
                && (state.symlinks.contains_key(&self.stored(state, &left))
                    || state.symlinks.contains_key(&self.stored(state, &right)))
            {
                return false;
            }
            self.resolved(state, &left) == self.resolved(state, &right)
        }))
    }
}
