//! Port of `tests/sandbox/integration_tests/test_runner_pause_resume.py` and its `_helpers.py`.
//!
//! A UnixLocal run materializes a manifest with every built-in entry type (inline file, nested
//! directory with an ephemeral child, local file and directory, a git repository, and the four
//! in-container rclone mounts), drives the sandbox through capability tools, and pauses on an
//! approval. The paused run leaves nothing on the host; the resumed run restores the workspace
//! from the saved state, re-applies the ephemeral entries and mounts, and reads the same tree back.
//!
//! The external tools (`git`, `cp`, `rclone`, ...) are the same mock scripts Python puts first on
//! `PATH`, so no network, bucket or FUSE driver is touched. This binary holds one test, because it
//! changes the process environment.
#![cfg(all(feature = "sandbox", unix))]

use std::collections::BTreeSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::AgentsError;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::items::{RunItem, RunItemKind};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding};
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::events::{
    CallbackSink, EventPhase, InstrumentedSession, SandboxSessionEvent,
};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::mount_entries::{
    AzureBlobMount, ConfiguredMountStrategy, GcsMount, R2Mount, S3Mount,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{ExecShell, SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::snapshot::SnapshotSpec;
use openai_agents_sdk::sandbox::state::SandboxSessionState;
use openai_agents_sdk::sandbox::workspace_paths::SandboxPathGrant;
use openai_agents_sdk::sandbox::{
    PtyExecRequest, PtyWriteRequest, SandboxAgent, SandboxResult, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, Tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

type Session = Arc<dyn SandboxSession>;

// ----------------------------------------------------------------------------------------------
// Fixture data (`_helpers.py`)
// ----------------------------------------------------------------------------------------------

const BUILTIN_MANIFEST_ENTRY_TYPES: [&str; 9] = [
    "azure_blob_mount",
    "dir",
    "file",
    "gcs_mount",
    "git_repo",
    "local_dir",
    "local_file",
    "r2_mount",
    "s3_mount",
];

const DURABLE_WORKSPACE_TEXTS: [(&str, &str); 8] = [
    ("inline.txt", "inline file v1\n"),
    ("delete_me.txt", "delete me v1\n"),
    ("tree/nested.txt", "nested file v1\n"),
    ("copied_file.txt", "local file source v1\n"),
    ("copied_dir/child.txt", "local dir child v1\n"),
    (
        "copied_dir/nested/grandchild.txt",
        "local dir grandchild v1\n",
    ),
    ("repo/README.md", "mock git repo readme v1\n"),
    ("repo/pkg/module.py", "VALUE = 'mock git module v1'\n"),
];

const EPHEMERAL_WORKSPACE_TEXTS: [(&str, &str); 1] =
    [("tree/ephemeral.txt", "ephemeral file v1\n")];

const MOUNT_WORKSPACE_TEXTS: [(&str, &str); 4] = [
    ("mounts/s3/.mock-rclone-mounted", "mock rclone mount\n"),
    ("mounts/gcs/.mock-rclone-mounted", "mock rclone mount\n"),
    ("mounts/r2/.mock-rclone-mounted", "mock rclone mount\n"),
    ("mounts/azure/.mock-rclone-mounted", "mock rclone mount\n"),
];

const ARCHIVE_WORKSPACE_TEXTS: [(&str, &str); 1] =
    [("archive_dir/hello.txt", "hello from tar archive\n")];

const RUNTIME_WORKSPACE_TEXTS: [(&str, &str); 1] = [("runtime_note.txt", "runtime note v1\n")];

const PATCHED_WORKSPACE_TEXTS: [(&str, &str); 2] = [
    ("inline.txt", "inline file v2\n"),
    ("created_by_patch.txt", "created by patch"),
];

const RESTORED_WORKSPACE_DIRS: [&str; 11] = [
    "archive_dir",
    "copied_dir",
    "copied_dir/nested",
    "mounts",
    "mounts/azure",
    "mounts/gcs",
    "mounts/r2",
    "mounts/s3",
    "repo",
    "repo/pkg",
    "tree",
];

const RESTORED_WORKSPACE_FILES: [&str; 16] = [
    "archive_dir/hello.txt",
    "bundle.tar",
    "copied_dir/child.txt",
    "copied_dir/nested/grandchild.txt",
    "copied_file.txt",
    "created_by_patch.txt",
    "inline.txt",
    "mounts/azure/.mock-rclone-mounted",
    "mounts/gcs/.mock-rclone-mounted",
    "mounts/r2/.mock-rclone-mounted",
    "mounts/s3/.mock-rclone-mounted",
    "repo/README.md",
    "repo/pkg/module.py",
    "runtime_note.txt",
    "tree/ephemeral.txt",
    "tree/nested.txt",
];

const SANDBOX_INTERNAL_WORKSPACE_DIR_PREFIXES: [&str; 1] = [".sandbox-rclone-config"];

const MOCK_TOOL_NAMES: [&str; 9] = [
    "blobfuse2",
    "cp",
    "fusermount3",
    "git",
    "mount-s3",
    "pkill",
    "rclone",
    "rm",
    "umount",
];

const MOCK_TOOL_SCRIPT: &str = r#"#!/bin/sh
set -eu

tool=$(basename "$0")
log_path="${SANDBOX_INTEGRATION_TOOL_LOG:-}"
if [ -n "$log_path" ]; then
  {
    printf "%s" "$tool"
    for arg in "$@"; do
      printf "\t%s" "$arg"
    done
    printf "\n"
  } >> "$log_path"
fi

case "$tool" in
  git)
    exit 0
    ;;
  cp)
    dest=""
    for arg in "$@"; do
      dest="$arg"
    done
    mkdir -p "$dest/pkg"
    printf "mock git repo readme v1\n" > "$dest/README.md"
    printf "VALUE = 'mock git module v1'\n" > "$dest/pkg/module.py"
    exit 0
    ;;
  rclone)
    if [ "${1:-}" = "mount" ] && [ -n "${3:-}" ]; then
      mkdir -p "$3"
      printf "mock rclone mount\n" > "$3/.mock-rclone-mounted"
    fi
    exit 0
    ;;
  blobfuse2)
    if [ "${1:-}" = "mount" ]; then
      dest=""
      for arg in "$@"; do
        dest="$arg"
      done
      mkdir -p "$dest"
      printf "mock blobfuse mount\n" > "$dest/.mock-blobfuse-mounted"
    fi
    exit 0
    ;;
  mount-s3)
    dest=""
    for arg in "$@"; do
      dest="$arg"
    done
    mkdir -p "$dest"
    printf "mock mount-s3 mount\n" > "$dest/.mock-mount-s3-mounted"
    exit 0
    ;;
  rm)
    recursive=""
    for arg in "$@"; do
      case "$arg" in
        -rf|-fr|-r|-f|--)
          if [ "$arg" = "-rf" ] || [ "$arg" = "-fr" ] || [ "$arg" = "-r" ]; then
            recursive="-r"
          fi
          ;;
        "$HOME"|"$HOME"/*)
          if [ -n "$recursive" ]; then
            /bin/rm -rf -- "$arg"
          else
            /bin/rm -f -- "$arg"
          fi
          ;;
        /*)
          ;;
        *..*)
          ;;
        *)
          if [ -n "$recursive" ]; then
            /bin/rm -rf -- "$arg"
          else
            /bin/rm -f -- "$arg"
          fi
          ;;
      esac
    done
    exit 0
    ;;
  fusermount3|umount|pkill)
    exit 0
    ;;
esac

exit 0
"#;

// ----------------------------------------------------------------------------------------------
// Workspace helpers
// ----------------------------------------------------------------------------------------------

fn fail(message: impl Into<String>) -> AgentsError {
    AgentsError::user(message.into())
}

fn to_agents(error: openai_agents_sdk::sandbox::SandboxError) -> AgentsError {
    AgentsError::from(error)
}

async fn read_workspace_text(session: &Session, path: &str) -> SandboxResult<String> {
    let bytes = session.read_file(path).await?;
    Ok(String::from_utf8(bytes).expect("utf-8 workspace text"))
}

async fn write_workspace_text(session: &Session, path: &str, text: &str) -> SandboxResult<()> {
    session.write_file(path, text.as_bytes()).await
}

async fn assert_workspace_texts(
    session: &Session,
    expected: &[(&str, &str)],
) -> Result<(), AgentsError> {
    let mut actual = Vec::new();
    for (path, _) in expected {
        let text = read_workspace_text(session, path)
            .await
            .map_err(|error| fail(format!("{path}: {error:?}")))?;
        actual.push((path.to_string(), text));
    }
    let wanted: Vec<(String, String)> = expected
        .iter()
        .map(|(path, text)| (path.to_string(), text.to_string()))
        .collect();
    if actual != wanted {
        return Err(fail(format!(
            "workspace texts differ: {actual:?} != {wanted:?}"
        )));
    }
    Ok(())
}

fn manifest_entry_types(session: &Session) -> BTreeSet<String> {
    session
        .manifest()
        .iter_entries()
        .into_iter()
        .map(|(_, entry)| entry.type_discriminator().to_string())
        .collect()
}

fn assert_entry_types(session: &Session) -> Result<(), AgentsError> {
    let expected: BTreeSet<String> = BUILTIN_MANIFEST_ENTRY_TYPES
        .iter()
        .map(|name| name.to_string())
        .collect();
    let actual = manifest_entry_types(session);
    if actual != expected {
        return Err(fail(format!("entry types {actual:?} != {expected:?}")));
    }
    Ok(())
}

async fn assert_manifest_materialized(session: &Session) -> Result<(), AgentsError> {
    assert_entry_types(session)?;
    assert_workspace_texts(session, &DURABLE_WORKSPACE_TEXTS).await?;
    assert_workspace_texts(session, &EPHEMERAL_WORKSPACE_TEXTS).await?;
    assert_workspace_texts(session, &MOUNT_WORKSPACE_TEXTS).await
}

async fn assert_workspace_missing(session: &Session, path: &str) -> Result<(), AgentsError> {
    match read_workspace_text(session, path).await {
        Err(error) if error.code == ErrorCode::WorkspaceReadNotFound => Ok(()),
        other => Err(fail(format!(
            "expected {path} to be missing, got {other:?}"
        ))),
    }
}

fn lifecycle_patch_operations() -> Vec<ApplyPatchOperation> {
    vec![
        ApplyPatchOperation {
            op_type: ApplyPatchOperationType::UpdateFile,
            path: "inline.txt".into(),
            diff: Some("@@\n-inline file v1\n+inline file v2\n".into()),
            move_to: None,
        },
        ApplyPatchOperation {
            op_type: ApplyPatchOperationType::CreateFile,
            path: "created_by_patch.txt".into(),
            diff: Some("+created by patch\n".into()),
            move_to: None,
        },
        ApplyPatchOperation {
            op_type: ApplyPatchOperationType::DeleteFile,
            path: "delete_me.txt".into(),
            diff: None,
            move_to: None,
        },
    ]
}

async fn assert_lifecycle_patch_state(session: &Session) -> Result<(), AgentsError> {
    let mut expected: Vec<(&str, &str)> = DURABLE_WORKSPACE_TEXTS
        .iter()
        .copied()
        .filter(|(path, _)| *path != "delete_me.txt" && *path != "inline.txt")
        .collect();
    expected.extend(RUNTIME_WORKSPACE_TEXTS);
    expected.extend(PATCHED_WORKSPACE_TEXTS);
    assert_workspace_texts(session, &expected).await?;
    assert_workspace_missing(session, "delete_me.txt").await
}

async fn assert_restored_lifecycle_state(session: &Session) -> Result<(), AgentsError> {
    assert_entry_types(session)?;
    assert_lifecycle_patch_state(session).await?;
    assert_workspace_texts(session, &ARCHIVE_WORKSPACE_TEXTS).await?;
    assert_workspace_texts(session, &EPHEMERAL_WORKSPACE_TEXTS).await?;
    assert_workspace_texts(session, &MOUNT_WORKSPACE_TEXTS).await?;
    assert_restored_workspace_tree(session).await
}

// -- escapes ------------------------------------------------------------------------------------

async fn assert_read_blocked(session: &Session, path: &str) -> Result<(), AgentsError> {
    match read_workspace_text(session, path).await {
        Err(error) if error.code == ErrorCode::InvalidManifestPath => Ok(()),
        other => Err(fail(format!(
            "expected read of {path} to be blocked: {other:?}"
        ))),
    }
}

async fn assert_write_blocked(session: &Session, path: &str) -> Result<(), AgentsError> {
    match write_workspace_text(session, path, "outside write\n").await {
        Err(error) if error.code == ErrorCode::InvalidManifestPath => Ok(()),
        other => Err(fail(format!(
            "expected write to {path} to be blocked: {other:?}"
        ))),
    }
}

async fn assert_patch_blocked(session: &Session, path: &str) -> Result<(), AgentsError> {
    let editor = SandboxEditor::new(session.clone());
    let operation = ApplyPatchOperation {
        op_type: ApplyPatchOperationType::CreateFile,
        path: path.to_string(),
        diff: Some("+outside patch\n".into()),
        move_to: None,
    };
    match editor.create_file(&operation).await {
        Err(error)
            if matches!(
                error.sandbox_code(),
                Some("apply_patch_invalid_path") | Some("invalid_manifest_path")
            ) =>
        {
            Ok(())
        }
        other => Err(fail(format!(
            "expected patch of {path} to be blocked: {other:?}"
        ))),
    }
}

async fn assert_symlink_escape_blocked(session: &Session) -> Result<(), AgentsError> {
    let root = PathBuf::from(session.manifest().root());
    let outside = root.parent().expect("a parent").join("symlink-outside.txt");
    let link = root.join("symlink_escape.txt");
    std::fs::write(&outside, "outside symlink target\n").map_err(|e| fail(e.to_string()))?;
    std::os::unix::fs::symlink(&outside, &link).map_err(|e| fail(e.to_string()))?;
    let outcome = async {
        assert_read_blocked(session, "symlink_escape.txt").await?;
        assert_write_blocked(session, "symlink_escape.txt").await?;
        assert_patch_blocked(session, "symlink_escape.txt").await
    }
    .await;
    let _ = std::fs::remove_file(&link);
    let _ = std::fs::remove_file(&outside);
    outcome
}

async fn assert_workspace_escape_blocked(session: &Session) -> Result<(), AgentsError> {
    for path in ["../outside.txt", "/tmp/sandbox-outside.txt"] {
        assert_read_blocked(session, path).await?;
        assert_write_blocked(session, path).await?;
        assert_patch_blocked(session, path).await?;
    }
    assert_symlink_escape_blocked(session).await
}

// -- tree ---------------------------------------------------------------------------------------

fn is_internal_dir(path: &str) -> bool {
    SANDBOX_INTERNAL_WORKSPACE_DIR_PREFIXES
        .iter()
        .any(|prefix| path == *prefix || path.starts_with(&format!("{prefix}/")))
}

async fn workspace_tree(
    session: &Session,
) -> Result<(BTreeSet<String>, BTreeSet<String>), AgentsError> {
    let root = std::fs::canonicalize(session.manifest().root()).map_err(|e| fail(e.to_string()))?;
    let mut dirs = BTreeSet::new();
    let mut files = BTreeSet::new();
    let mut pending = vec![".".to_string()];
    while let Some(directory) = pending.pop() {
        for entry in session.ls(&directory).await.map_err(to_agents)? {
            let relative = Path::new(&entry.path)
                .strip_prefix(&root)
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_else(|_| entry.path.clone());
            if entry.is_symlink {
                return Err(fail(format!(
                    "unexpected symlink in the workspace: {relative}"
                )));
            }
            if entry.is_dir {
                if is_internal_dir(&relative) {
                    continue;
                }
                dirs.insert(relative.clone());
                pending.push(relative);
            } else {
                files.insert(relative);
            }
        }
    }
    Ok((dirs, files))
}

async fn assert_restored_workspace_tree(session: &Session) -> Result<(), AgentsError> {
    let (dirs, files) = workspace_tree(session).await?;
    let want_dirs: BTreeSet<String> = RESTORED_WORKSPACE_DIRS
        .iter()
        .map(|s| s.to_string())
        .collect();
    let want_files: BTreeSet<String> = RESTORED_WORKSPACE_FILES
        .iter()
        .map(|s| s.to_string())
        .collect();
    if dirs != want_dirs {
        return Err(fail(format!(
            "actual dirs {dirs:?} != expected {want_dirs:?}"
        )));
    }
    if files != want_files {
        return Err(fail(format!(
            "actual files {files:?} != expected {want_files:?}"
        )));
    }
    Ok(())
}

fn tar_bytes(members: &[(&str, &str)]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, text) in members {
        let payload = text.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_size(payload.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, name, payload).unwrap();
    }
    builder.into_inner().unwrap()
}

// ----------------------------------------------------------------------------------------------
// Capabilities
// ----------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadFileArgs {
    path: String,
}

fn probe_tool<F, Fut>(name: &'static str, session: &Session, f: F) -> Tool<()>
where
    F: Fn(Session) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, AgentsError>> + Send + 'static,
{
    let session = session.clone();
    Tool::Function(Arc::new(FunctionTool::<()>::new(
        name,
        name,
        move |_ctx, _args: NoArgs| f(session.clone()),
    )))
}

/// `SandboxFileCapability`: `write_file` and `read_file` over the workspace.
#[derive(Debug)]
struct SandboxFileCapability;

#[async_trait]
impl Capability<()> for SandboxFileCapability {
    fn capability_type(&self) -> &str {
        "sandbox-file"
    }

    fn tools(
        &self,
        binding: &CapabilityBinding,
    ) -> openai_agents_sdk::error::Result<Vec<Tool<()>>> {
        let write_session = binding.session.clone();
        let read_session = binding.session.clone();
        let write = FunctionTool::<()>::new(
            "write_file",
            "write a workspace file",
            move |_ctx, args: WriteFileArgs| {
                let session = write_session.clone();
                async move {
                    write_workspace_text(&session, &args.path, &args.content)
                        .await
                        .map_err(to_agents)?;
                    Ok::<_, AgentsError>(format!("wrote {}", args.path))
                }
            },
        );
        let read = FunctionTool::<()>::new(
            "read_file",
            "read a workspace file",
            move |_ctx, args: ReadFileArgs| {
                let session = read_session.clone();
                async move {
                    read_workspace_text(&session, &args.path)
                        .await
                        .map_err(to_agents)
                }
            },
        );
        Ok(vec![
            Tool::Function(Arc::new(write)),
            Tool::Function(Arc::new(read)),
        ])
    }
}

/// `SandboxLifecycleProbeCapability`: tools that assert the sandbox lifecycle from inside.
#[derive(Debug)]
struct SandboxLifecycleProbeCapability;

#[async_trait]
impl Capability<()> for SandboxLifecycleProbeCapability {
    fn capability_type(&self) -> &str {
        "sandbox-lifecycle-probe"
    }

    fn tools(
        &self,
        binding: &CapabilityBinding,
    ) -> openai_agents_sdk::error::Result<Vec<Tool<()>>> {
        let session = &binding.session;
        Ok(vec![
            probe_tool("assert_manifest_materialized", session, |s| async move {
                assert_manifest_materialized(&s).await?;
                Ok("manifest materialized".to_string())
            }),
            probe_tool("apply_lifecycle_patch", session, |s| async move {
                let editor = SandboxEditor::new(s.clone());
                for operation in lifecycle_patch_operations() {
                    let result = match operation.op_type {
                        ApplyPatchOperationType::CreateFile => editor.create_file(&operation).await,
                        ApplyPatchOperationType::UpdateFile => editor.update_file(&operation).await,
                        ApplyPatchOperationType::DeleteFile => editor.delete_file(&operation).await,
                    }?;
                    if result.output.is_none() {
                        return Err(fail("a patch operation reported nothing"));
                    }
                }
                assert_lifecycle_patch_state(&s).await?;
                Ok("lifecycle patch applied".to_string())
            }),
            probe_tool("assert_workspace_escape_blocked", session, |s| async move {
                assert_workspace_escape_blocked(&s).await?;
                Ok("workspace escape blocked".to_string())
            }),
            probe_tool("extract_lifecycle_archive", session, |s| async move {
                s.extract("bundle.tar", &tar_bytes(&ARCHIVE_WORKSPACE_TEXTS), None)
                    .await
                    .map_err(to_agents)?;
                assert_workspace_texts(&s, &ARCHIVE_WORKSPACE_TEXTS).await?;
                Ok("archive extracted".to_string())
            }),
            probe_tool("start_lifecycle_pty", session, |s| async move {
                let script = "printf 'ready\\n'; while IFS= read -r line; do printf 'got:%s\\n' \"$line\"; done";
                let pty = s
                    .pty_exec_start(
                        PtyExecRequest::new(script)
                            .with_shell(ExecShell::Custom(vec!["sh".into(), "-c".into()]))
                            .with_tty(true)
                            .with_yield_time_ms(250),
                    )
                    .await
                    .map_err(to_agents)?;
                let Some(process_id) = pty.process_id else {
                    return Err(fail("the pty has no process id"));
                };
                let output = pty.output_text().replace("\r\n", "\n");
                if output != "ready\n" {
                    return Err(fail(format!("unexpected pty output {output:?}")));
                }
                let update = s
                    .pty_write_stdin(
                        PtyWriteRequest::new(process_id, "hello pty\n").with_yield_time_ms(250),
                    )
                    .await
                    .map_err(to_agents)?;
                let written = update.output_text().replace("\r\n", "\n");
                if written != "hello pty\ngot:hello pty\n"
                    || update.process_id != Some(process_id)
                    || update.exit_code.is_some()
                {
                    return Err(fail(format!("unexpected pty echo {written:?} {update:?}")));
                }
                Ok("pty started and echoed stdin".to_string())
            }),
            probe_tool("assert_restored_lifecycle_state", session, |s| async move {
                assert_restored_lifecycle_state(&s).await?;
                Ok("restored lifecycle state verified".to_string())
            }),
        ])
    }
}

// ----------------------------------------------------------------------------------------------
// Client instrumentation, mock tools, manifest
// ----------------------------------------------------------------------------------------------

/// Wraps every session the UnixLocal client hands out in an [`InstrumentedSession`] feeding one
/// event list, as the Python client's `instrumentation=` does.
#[derive(Debug)]
struct InstrumentedClient {
    inner: UnixLocalSandboxClient,
    events: Arc<Mutex<Vec<SandboxSessionEvent>>>,
}

impl InstrumentedClient {
    fn wrap(&self, session: Session) -> Session {
        let events = self.events.clone();
        Arc::new(InstrumentedSession::new(
            session,
            Arc::new(CallbackSink::new(move |event| {
                events.lock().unwrap().push(event.clone());
            })),
        ))
    }

    fn successful(&self, op: &str) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| {
                event.op == op && event.phase == EventPhase::Finish && event.ok == Some(true)
            })
            .count()
    }
}

#[async_trait]
impl SandboxClient for InstrumentedClient {
    fn backend_id(&self) -> &str {
        self.inner.backend_id()
    }

    async fn create(&self, manifest: Option<Manifest>) -> SandboxResult<Session> {
        Ok(self.wrap(self.inner.create(manifest).await?))
    }

    fn configured_snapshot(&self) -> Option<SnapshotSpec> {
        self.inner.configured_snapshot()
    }

    async fn create_with_snapshot(
        &self,
        manifest: Option<Manifest>,
        snapshot: SnapshotSpec,
    ) -> SandboxResult<Session> {
        Ok(self.wrap(self.inner.create_with_snapshot(manifest, snapshot).await?))
    }

    async fn delete(&self, session: Session) -> SandboxResult<()> {
        self.inner.delete(session).await
    }

    async fn resume(
        &self,
        state: &SandboxSessionState,
        manifest: Manifest,
    ) -> SandboxResult<Session> {
        Ok(self.wrap(self.inner.resume(state, manifest).await?))
    }
}

fn install_mock_external_tools(base: &Path, writable_root: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let bin_dir = base.join("mock-bin");
    std::fs::create_dir(&bin_dir).unwrap();
    let log_path = writable_root.join("mock-tool-calls.tsv");
    std::fs::write(&log_path, "").unwrap();
    for name in MOCK_TOOL_NAMES {
        let tool = bin_dir.join(name);
        std::fs::write(&tool, MOCK_TOOL_SCRIPT).unwrap();
        std::fs::set_permissions(&tool, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let existing = std::env::var("PATH").unwrap_or_default();
    // SAFETY: this binary holds a single test, so nothing else reads the environment while it
    // changes.
    unsafe {
        std::env::set_var("SANDBOX_INTEGRATION_TOOL_LOG", &log_path);
        std::env::set_var("PATH", format!("{}:{existing}", bin_dir.display()));
    }
    log_path
}

fn create_local_sources(base: &Path) -> PathBuf {
    let source_root = base.join("manifest-sources");
    let local_dir = source_root.join("local-dir");
    let nested = local_dir.join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(source_root.join("local-file.txt"), "local file source v1\n").unwrap();
    std::fs::write(local_dir.join("child.txt"), "local dir child v1\n").unwrap();
    std::fs::write(nested.join("grandchild.txt"), "local dir grandchild v1\n").unwrap();
    source_root
}

fn text_of(table: &[(&'static str, &'static str)], path: &str) -> &'static str {
    table.iter().find(|(name, _)| *name == path).unwrap().1
}

fn build_manifest_with_all_entry_types(source_root: &Path) -> Manifest {
    let rclone = |mount: openai_agents_sdk::sandbox::mounts::Mount| Entry::mount(mount);
    Manifest::new()
        .with_extra_path_grant(SandboxPathGrant::new(source_root.to_string_lossy()).unwrap())
        .with_entry(
            "inline.txt",
            Entry::text_file(text_of(&DURABLE_WORKSPACE_TEXTS, "inline.txt")),
        )
        .with_entry(
            "delete_me.txt",
            Entry::text_file(text_of(&DURABLE_WORKSPACE_TEXTS, "delete_me.txt")),
        )
        .with_entry(
            "tree",
            Entry::dir()
                .with_child(
                    "nested.txt",
                    Entry::text_file(text_of(&DURABLE_WORKSPACE_TEXTS, "tree/nested.txt")),
                )
                .with_child(
                    "ephemeral.txt",
                    Entry::text_file(text_of(&EPHEMERAL_WORKSPACE_TEXTS, "tree/ephemeral.txt"))
                        .with_ephemeral(true),
                ),
        )
        .with_entry(
            "copied_file.txt",
            Entry::local_file(source_root.join("local-file.txt")),
        )
        .with_entry(
            "copied_dir",
            Entry::local_dir(source_root.join("local-dir")),
        )
        .with_entry(
            "repo",
            Entry::git_repo("openai/mock-sandbox-fixture", "main"),
        )
        .with_entry(
            "mounts/s3",
            rclone(ConfiguredMountStrategy::rclone(S3Mount::new("s3-bucket")).mount()),
        )
        .with_entry(
            "mounts/gcs",
            rclone(ConfiguredMountStrategy::rclone(GcsMount::new("gcs-bucket")).mount()),
        )
        .with_entry(
            "mounts/r2",
            rclone(
                ConfiguredMountStrategy::rclone(R2Mount::new("r2-bucket", "r2-account-id")).mount(),
            ),
        )
        .with_entry(
            "mounts/azure",
            rclone(
                ConfiguredMountStrategy::rclone(AzureBlobMount::new(
                    "azure-account",
                    "azure-container",
                ))
                .mount(),
            ),
        )
}

fn build_agent(model: Arc<FakeModel>, manifest: Manifest) -> SandboxAgent<()> {
    let approval = FunctionTool::<()>::new(
        "approval_tool",
        "needs approval",
        |_ctx, _args: NoArgs| async move { Ok::<_, AgentsError>("approved".to_string()) },
    )
    .with_needs_approval(true);
    SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Use the sandbox lifecycle tools.")
        .with_default_manifest(manifest)
        .with_tool(Tool::Function(Arc::new(approval)))
        .with_capabilities(vec![
            Arc::new(SandboxFileCapability),
            Arc::new(SandboxLifecycleProbeCapability),
        ])
}

fn tool_outputs(items: &[RunItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { output, .. } if item.agent == "sandbox" => {
                Some(output.as_str().expect("text tool output").to_string())
            }
            _ => None,
        })
        .collect()
}

fn call(model: &FakeModel, name: &str, arguments: Value, call_id: &str) {
    model.add_turn(vec![function_call_with_id(
        name,
        arguments.to_string(),
        call_id,
    )]);
}

// ----------------------------------------------------------------------------------------------
// The test
// ----------------------------------------------------------------------------------------------

/// Python: `test_runner_preserves_unix_local_lifecycle_state_across_pause_and_resume`.
#[tokio::test]
async fn runner_preserves_unix_local_lifecycle_state_across_pause_and_resume() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let temp = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(temp.path()).unwrap();
    let source_root = create_local_sources(&base);
    let log_path = install_mock_external_tools(&base, &source_root);
    assert_eq!(log_path.parent().unwrap(), source_root);
    let manifest = build_manifest_with_all_entry_types(&source_root);

    let client = Arc::new(InstrumentedClient {
        inner: UnixLocalSandboxClient::new(),
        events: Arc::new(Mutex::new(Vec::new())),
    });

    let model = Arc::new(FakeModel::new());
    call(
        &model,
        "assert_manifest_materialized",
        json!({}),
        "call_manifest_materialized",
    );
    call(
        &model,
        "write_file",
        json!({"path": "runtime_note.txt", "content": "runtime note v1\n"}),
        "call_write_runtime_note",
    );
    call(
        &model,
        "apply_lifecycle_patch",
        json!({}),
        "call_apply_lifecycle_patch",
    );
    call(
        &model,
        "assert_workspace_escape_blocked",
        json!({}),
        "call_assert_workspace_escape_blocked",
    );
    call(
        &model,
        "extract_lifecycle_archive",
        json!({}),
        "call_extract_lifecycle_archive",
    );
    call(
        &model,
        "start_lifecycle_pty",
        json!({}),
        "call_start_lifecycle_pty",
    );
    call(&model, "approval_tool", json!({}), "call_approval");

    let agent = build_agent(model.clone(), manifest.clone());
    let config = SandboxRunConfig::with_client(client.clone());
    let first = SandboxRunner::run_to_outcome(
        &agent,
        "verify the UnixLocal sandbox lifecycle and wait for approval",
        &config,
        RunOptions::default().max_turns(Some(20)),
    )
    .await
    .unwrap();

    assert_eq!(
        tool_outputs(&first.result.new_items),
        [
            "manifest materialized",
            "wrote runtime_note.txt",
            "lifecycle patch applied",
            "workspace escape blocked",
            "archive extracted",
            "pty started and echoed stdin",
        ]
    );
    assert_eq!(first.result.interruptions.len(), 1);

    let sandbox_state = first
        .sandbox_state
        .clone()
        .expect("a paused run keeps its state");
    let sandbox_json = sandbox_state.to_json_string().unwrap();
    let state_value: Value = serde_json::from_str(&sandbox_json).unwrap();
    assert_eq!(state_value["snapshot"]["type"], "local");
    assert!(sandbox_state.workspace_root_owned);
    assert!(sandbox_state.workspace_root_ready);
    let workspace_root = PathBuf::from(&sandbox_state.workspace_root);
    // Python reads the root from `session_state["manifest"]["root"]`. A Rust state carries no
    // manifest (a resume takes the trusted manifest from the application), so the root is its
    // own field.
    assert!(state_value.get("manifest").is_none());
    assert!(!workspace_root.exists());
    assert_eq!(client.successful("stop"), 1);
    assert_eq!(client.successful("shutdown"), 1);

    let run_state_json = first.result.to_state().to_json_string().unwrap();
    drop(first);
    let mut run_state = RunState::from_json_str(&run_state_json).unwrap();
    let restored_interruptions = run_state.get_interruptions();
    assert_eq!(restored_interruptions.len(), 1);
    run_state.approve(&restored_interruptions[0].clone(), false);

    let resumed_model = Arc::new(FakeModel::new());
    call(
        &resumed_model,
        "assert_restored_lifecycle_state",
        json!({}),
        "call_assert_restored_lifecycle_state",
    );
    call(
        &resumed_model,
        "read_file",
        json!({"path": "runtime_note.txt"}),
        "call_read_runtime_note",
    );
    resumed_model.add_turn(vec![assistant_message("done")]);
    let resumed_agent = build_agent(resumed_model.clone(), manifest);
    let resume_config = SandboxRunConfig::with_client(client.clone())
        .session_state(SandboxSessionState::from_json_str(&sandbox_json).unwrap());
    let resumed = SandboxRunner::run_with_options(
        &resumed_agent,
        run_state,
        &resume_config,
        RunOptions::default().max_turns(Some(20)),
    )
    .await
    .unwrap();

    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert!(!workspace_root.exists());
    assert_eq!(client.successful("stop"), 2);
    assert_eq!(client.successful("shutdown"), 2);
    let outputs = tool_outputs(&resumed.new_items);
    assert_eq!(
        outputs[outputs.len() - 3..],
        [
            "approved".to_string(),
            "restored lifecycle state verified".to_string(),
            "runtime note v1\n".to_string(),
        ]
    );
}
