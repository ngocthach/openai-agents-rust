//! Parity port of the remaining `tests/sandbox/test_unix_local*.py` cases that need a session
//! wrapper or process-wide state:
//! `test_pty_terminal_signals_interrupt_even_if_parent_ignores_signal`,
//! `test_same_file_*`, `test_public_entry_operations_preserve_leaf_symlinks`,
//! `test_apply_patch_preserves_diverged_alias_source`, `test_base_default_create_*` and the
//! client-driven `test_case_only_move_to_keeps_the_file`.
//!
//! Real workspace directories throughout, as Python's `tmp_path` fixtures.
#![allow(clippy::await_holding_lock)] // the lock serializes process-wide state for the whole test
#![cfg(all(unix, feature = "sandbox"))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use openai_agents_sdk::sandbox::editor::SandboxEditor;
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyWriteRequest};
use openai_agents_sdk::sandbox::sandboxes::unix_local::confinement::{self, HostFacts};
use openai_agents_sdk::sandbox::sandboxes::unix_local::{
    UnixLocalSandboxClient, UnixLocalSandboxSession,
};
use openai_agents_sdk::sandbox::types::{ExecResult, User};
use openai_agents_sdk::sandbox::{
    ExecRequest, ExecShell, Manifest, SandboxClient, SandboxPathGrant, SandboxSession,
};

/// Serializes the signal tests: dispositions are process-wide and the harness runs `#[test]`
/// functions on parallel threads.
static DISPOSITIONS: Mutex<()> = Mutex::new(());

fn unix_local_at(root: &Path) -> UnixLocalSandboxSession {
    UnixLocalSandboxSession::new(
        Manifest::new().with_root(root.to_string_lossy().into_owned()),
        false,
    )
    .expect("a session over an absolute root")
}

fn symlink(target: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(target.as_ref(), link.as_ref()).expect("create a symlink");
}

fn names_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("list directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------------------------------------------
// PTY terminal signals
// ---------------------------------------------------------------------------------------------

/// Ignore `signum` in this process for the duration of `body`, then restore it.
async fn while_ignoring<T, F: std::future::Future<Output = T>>(signum: i32, body: F) -> T {
    let _guard = DISPOSITIONS.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: `signal` with `SIG_IGN` is async-signal-safe, and the lock above keeps every other
    // test in this binary from touching a disposition while it is in effect.
    let previous = unsafe { libc::signal(signum, libc::SIG_IGN) };
    let result = body.await;
    unsafe { libc::signal(signum, previous) };
    result
}

/// Python: `test_pty_terminal_signals_interrupt_even_if_parent_ignores_signal[sigint, sigquit]`.
///
/// The terminal's interrupt characters must reach the child as signals whatever the parent
/// ignores, so a PTY child cannot inherit `SIG_IGN` and shrug off `^C` or `^\`.
#[tokio::test(flavor = "multi_thread")]
async fn pty_terminal_signals_interrupt_even_if_parent_ignores_signal() {
    for (signum, chars, name) in [
        (libc::SIGINT, "\u{3}", "sigint"),
        (libc::SIGQUIT, "\u{1c}", "sigquit"),
    ] {
        let dir = tempfile::tempdir().expect("temp dir");
        let client = UnixLocalSandboxClient::new();
        let session = client
            .create(Some(Manifest::new().with_root(
                dir.path().join("workspace").to_string_lossy().into_owned(),
            )))
            .await
            .expect("create");
        session.start().await.expect("start");

        let interrupted = while_ignoring(signum, async {
            let started = session
                .pty_exec_start(
                    PtyExecRequest::new("sleep 30")
                        .with_shell(ExecShell::Plain)
                        .with_tty(true)
                        .with_yield_time_ms(50),
                )
                .await
                .expect("start a pty session");
            let process_id = started.process_id.expect("a live process id");
            session
                .pty_write_stdin(PtyWriteRequest::new(process_id, chars).with_yield_time_ms(5_500))
                .await
                .expect("send the interrupt character")
        })
        .await;

        assert_eq!(interrupted.process_id, None, "{name}: the process ended");
        let code = interrupted
            .exit_code
            .unwrap_or_else(|| panic!("{name}: no exit code"));
        // A child killed by a signal has no exit status; Python reports `-signum`, which is what
        // tells `Ctrl-C` from an ordinary failure.
        assert_eq!(code, -signum, "{name}: the exit code is the negated signal");
    }
}

// ---------------------------------------------------------------------------------------------
// same_file — device and inode equality
// ---------------------------------------------------------------------------------------------

/// Python: `test_same_file_answers_for_real_paths`.
#[tokio::test]
async fn same_file_answers_for_real_paths() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
        ))
        .await
        .expect("create");
    session.start().await.expect("start");
    session
        .write_file("one.txt", b"one\n")
        .await
        .expect("write one");
    session
        .write_file("two.txt", b"two\n")
        .await
        .expect("write two");

    let one = workspace.join("one.txt").to_string_lossy().into_owned();
    let two = workspace.join("two.txt").to_string_lossy().into_owned();
    assert!(
        session
            .same_file(&one, &one, false, None)
            .await
            .expect("same")
    );
    assert!(
        !session
            .same_file(&one, &two, false, None)
            .await
            .expect("different")
    );
}

/// Python: `test_same_file_answers_by_entry_identity`.
#[tokio::test]
async fn same_file_answers_by_entry_identity() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::write(workspace.join("one.txt"), b"one").expect("write one");
    std::fs::write(workspace.join("two.txt"), b"two").expect("write two");
    std::fs::hard_link(workspace.join("one.txt"), workspace.join("hard.txt")).expect("hardlink");
    let session = unix_local_at(&workspace);

    assert!(
        session
            .same_file("one.txt", "one.txt", true, None)
            .await
            .expect("self")
    );
    assert!(
        session
            .same_file("one.txt", "hard.txt", true, None)
            .await
            .expect("hardlink")
    );
    assert!(
        !session
            .same_file("one.txt", "two.txt", true, None)
            .await
            .expect("different")
    );
    assert!(
        !session
            .same_file("one.txt", "missing.txt", true, None)
            .await
            .expect("a missing entry is not the same file")
    );
}

/// Python: `test_same_file_distinguishes_a_link_from_its_target_when_asked`.
#[tokio::test]
async fn same_file_distinguishes_a_link_from_its_target_when_asked() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::write(workspace.join("target.txt"), b"target").expect("write target");
    symlink(workspace.join("target.txt"), workspace.join("link.txt"));
    let session = unix_local_at(&workspace);

    assert!(
        session
            .same_file("link.txt", "target.txt", true, None)
            .await
            .expect("following the link")
    );
    assert!(
        !session
            .same_file("link.txt", "target.txt", false, None)
            .await
            .expect("not following the link")
    );
}

/// Python: `test_public_entry_operations_preserve_leaf_symlinks[None]`.
///
/// The `example-user` leg drives Python's own `sudo python3` worker body and is Python-only.
#[tokio::test]
async fn public_entry_operations_preserve_leaf_symlinks() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let target = workspace.join("target");
    std::fs::write(&target, b"untouched").expect("write target");
    let link = workspace.join("link");
    symlink("target", &link);
    symlink(".", workspace.join("alias"));
    let session = unix_local_at(&workspace);

    assert!(
        session
            .same_file("link", "target", true, None)
            .await
            .expect("follow")
    );
    assert!(
        !session
            .same_file("link", "target", false, None)
            .await
            .expect("no follow")
    );
    assert!(
        !session
            .same_file("link", "link", false, None)
            .await
            .expect("link vs itself")
    );
    session
        .mv("alias/link", "moved")
        .await
        .expect("move the link");
    assert!(std::fs::symlink_metadata(&link).is_err());
    assert!(
        std::fs::symlink_metadata(workspace.join("moved"))
            .expect("stat moved")
            .file_type()
            .is_symlink()
    );
    assert_eq!(std::fs::read(&target).expect("read target"), b"untouched");

    std::fs::write(workspace.join("replacement"), b"replacement").expect("write replacement");
    session
        .mv("replacement", "moved")
        .await
        .expect("replace the link");
    assert!(
        !std::fs::symlink_metadata(workspace.join("moved"))
            .expect("stat moved")
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        std::fs::read(workspace.join("moved")).expect("read moved"),
        b"replacement"
    );
    assert_eq!(std::fs::read(&target).expect("read target"), b"untouched");
}

// ---------------------------------------------------------------------------------------------
// A session that changes the workspace between the steps of a move
// ---------------------------------------------------------------------------------------------

/// A Unix-local session that, right after `mv` commits, replaces `replace_after_mv`.
///
/// Python pauses `session.mv` after it commits and swaps the source from the test. Every
/// operation the editor reaches is delegated to the real backend; only `mv` has a hook.
#[derive(Debug)]
struct MoveHookSession {
    inner: UnixLocalSandboxSession,
    replace_after_mv: Option<PathBuf>,
}

#[async_trait]
impl SandboxSession for MoveHookSession {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn manifest(&self) -> &Manifest {
        self.inner.manifest()
    }
    fn backend_id(&self) -> &str {
        self.inner.backend_id()
    }
    async fn start(&self) -> SandboxResult<()> {
        self.inner.start().await
    }
    async fn stop(&self) -> SandboxResult<()> {
        self.inner.stop().await
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        self.inner.shutdown().await
    }
    async fn running(&self) -> SandboxResult<bool> {
        self.inner.running().await
    }
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        self.inner.exec(request).await
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.inner.read_file(path).await
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.inner.write_file(path, contents).await
    }
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.hydrate_workspace(archive).await
    }
    fn normalize_path(&self, path: &str) -> SandboxResult<String> {
        self.inner.normalize_path(path)
    }
    fn normalize_path_for_write(&self, path: &str) -> SandboxResult<String> {
        self.inner.normalize_path_for_write(path)
    }
    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        self.inner.mkdir(path, parents).await
    }
    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        self.inner.rm(path, recursive).await
    }
    async fn same_file(
        &self,
        left: &str,
        right: &str,
        follow_symlinks: bool,
        user: Option<&User>,
    ) -> SandboxResult<bool> {
        self.inner
            .same_file(left, right, follow_symlinks, user)
            .await
    }
    async fn write_new_file(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&User>,
    ) -> SandboxResult<()> {
        self.inner.write_new_file(path, contents, user).await
    }
    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        self.inner.mv(source, destination).await?;
        if let Some(source_path) = &self.replace_after_mv {
            std::fs::remove_file(source_path).expect("drop the source");
            std::fs::write(source_path, b"other writer\n").expect("a concurrent writer's file");
        }
        Ok(())
    }
}

/// Python: `test_apply_patch_preserves_diverged_alias_source[False, True]`.
///
/// Source and destination start as hardlinks of one file. Once the destination is committed as
/// its own file the source is no longer the same entry, so the editor must leave it alone and
/// say so — and must not delete a file a concurrent writer put at the source path meanwhile.
#[tokio::test]
async fn apply_patch_preserves_a_diverged_alias_source() {
    for replace_source in [false, true] {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("destination.txt");
        std::fs::write(&source, b"original\n").expect("write source");
        std::fs::hard_link(&source, &destination).expect("hardlink destination");
        let session = Arc::new(MoveHookSession {
            inner: unix_local_at(dir.path()),
            replace_after_mv: replace_source.then(|| source.clone()),
        });

        let error = SandboxEditor::new(session)
            .update_file(&ApplyPatchOperation {
                op_type: ApplyPatchOperationType::UpdateFile,
                path: "source.txt".to_string(),
                diff: Some("@@\n-original\n+updated\n".to_string()),
                move_to: Some("destination.txt".to_string()),
            })
            .await
            .expect_err("a diverged source is not removed");

        let message = format!("{error}");
        assert!(
            message.contains("source was left untouched"),
            "replace_source={replace_source}: {message}"
        );
        assert_eq!(
            std::fs::read(&source).expect("read source"),
            if replace_source {
                b"other writer\n".to_vec()
            } else {
                b"original\n".to_vec()
            },
            "replace_source={replace_source}"
        );
        assert_eq!(
            std::fs::read(&destination).expect("read destination"),
            b"updated\n",
            "replace_source={replace_source}"
        );
        assert_eq!(
            names_in(dir.path()),
            vec!["destination.txt".to_string(), "source.txt".to_string()],
            "replace_source={replace_source}: no staging file is left behind"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Provider defaults for Add File
// ---------------------------------------------------------------------------------------------

/// A host-filesystem session that implements only what the trait requires, so every other
/// behaviour is the trait's own default. Mirrors Python's `FilesystemTestSandboxSession`, a
/// `BaseSandboxSession` double with no process-execution implementation of its own.
#[derive(Debug)]
struct DefaultsOnlySession {
    inner: UnixLocalSandboxSession,
}

#[async_trait]
impl SandboxSession for DefaultsOnlySession {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn manifest(&self) -> &Manifest {
        self.inner.manifest()
    }
    fn backend_id(&self) -> &str {
        "filesystem_test"
    }
    async fn start(&self) -> SandboxResult<()> {
        self.inner.start().await
    }
    async fn stop(&self) -> SandboxResult<()> {
        self.inner.stop().await
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        self.inner.shutdown().await
    }
    async fn running(&self) -> SandboxResult<bool> {
        self.inner.running().await
    }
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        self.inner.exec(request).await
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.inner.read_file(path).await
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.inner.write_file(path, contents).await
    }
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.hydrate_workspace(archive).await
    }
}

fn defaults_only_editor(root: &Path) -> SandboxEditor {
    SandboxEditor::new(Arc::new(DefaultsOnlySession {
        inner: unix_local_at(root),
    }) as Arc<dyn SandboxSession>)
}

fn create_op(path: &str, diff: &str) -> ApplyPatchOperation {
    ApplyPatchOperation {
        op_type: ApplyPatchOperationType::CreateFile,
        path: path.to_string(),
        diff: Some(diff.to_string()),
        move_to: None,
    }
}

/// Python: `test_base_default_create_allows_a_missing_parent`.
#[tokio::test]
async fn base_default_create_allows_a_missing_parent() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");

    defaults_only_editor(&workspace)
        .create_file(&create_op("newdir/file.txt", "+hello\n"))
        .await
        .expect("the provider default creates the parent");

    assert_eq!(
        std::fs::read_to_string(workspace.join("newdir").join("file.txt")).expect("read file"),
        "hello"
    );
}

/// Python: `test_base_default_create_preserves_provider_write_semantics`.
///
/// Python's default create is the released mkdir-then-write: it claims nothing exclusively and
/// adds no probe, so a provider with only `write` sees an overwrite of an existing name.
#[tokio::test]
async fn base_default_create_preserves_provider_write_semantics() {
    let dir = tempfile::tempdir().expect("temp dir");
    let target = dir.path().join("existing.txt");
    std::fs::write(&target, b"previous").expect("write existing");

    defaults_only_editor(dir.path())
        .create_file(&create_op("existing.txt", "+replacement\n"))
        .await
        .expect("the provider default overwrites");

    assert_eq!(
        std::fs::read(&target).expect("read existing"),
        b"replacement"
    );
}

// ---------------------------------------------------------------------------------------------
// Case-only rename through the client
// ---------------------------------------------------------------------------------------------

/// Python: `test_case_only_move_to_keeps_the_file`.
///
/// Created through the client, the way a runner does, and skipped on a volume that does not fold
/// case because it then cannot exercise the bug.
#[tokio::test]
async fn case_only_move_keeps_the_file_through_the_client() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
        ))
        .await
        .expect("create");
    session.start().await.expect("start");
    session
        .write_file("notes.txt", b"alpha\nbeta\n")
        .await
        .expect("write notes.txt");

    let source = workspace.join("notes.txt").to_string_lossy().into_owned();
    let destination = workspace.join("Notes.txt").to_string_lossy().into_owned();
    if !session
        .same_file(&source, &destination, true, None)
        .await
        .expect("same_file")
    {
        // This volume does not fold case, so it cannot exercise the bug.
        return;
    }

    SandboxEditor::new(session.clone())
        .update_file(&ApplyPatchOperation {
            op_type: ApplyPatchOperationType::UpdateFile,
            path: "notes.txt".to_string(),
            diff: Some("@@\n alpha\n-beta\n+gamma\n".to_string()),
            move_to: Some("Notes.txt".to_string()),
        })
        .await
        .expect("a case-only rename");

    assert_eq!(names_in(&workspace), vec!["Notes.txt".to_string()]);
    assert_eq!(
        std::fs::read(workspace.join("Notes.txt")).expect("read Notes.txt"),
        b"alpha\ngamma\n"
    );
}

// ---------------------------------------------------------------------------------------------
// Listings keep exact names (an addition beyond the Python tests, which cover this only for the
// run-as-user listing worker)
// ---------------------------------------------------------------------------------------------

/// A name containing a newline must survive a listing intact, not be split into two entries.
#[tokio::test]
async fn listing_preserves_names_with_newlines_and_spaces() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::write(workspace.join("name\nwith spaces"), b"x").expect("write odd name");
    let session = unix_local_at(&workspace);

    let listing = session.ls(".").await.expect("list");
    assert_eq!(listing.len(), 1, "{listing:?}");
    assert_eq!(listing[0].name, "name\nwith spaces");
    assert_eq!(listing[0].size, Some(1));
    assert!(!listing[0].is_dir);
}

// ---------------------------------------------------------------------------------------------
// macOS `sandbox-exec` confinement — Python `test_runtime.py`
// ---------------------------------------------------------------------------------------------

/// The command words `confined_exec_command` is asked to wrap.
fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| part.to_string()).collect()
}

/// Wrap `command_parts` for `session` as if on macOS, with `which` standing in for `shutil.which`.
fn darwin_command(
    session: &UnixLocalSandboxSession,
    command_parts: &[&str],
    workspace_root: &Path,
    child_path: &str,
    host_path: &str,
    host_home: &Path,
    which: &dyn Fn(&str, Option<&str>) -> Option<PathBuf>,
) -> SandboxResult<Vec<String>> {
    session.confined_exec_command(
        &words(command_parts),
        workspace_root,
        Some(child_path),
        &HostFacts {
            is_darwin: true,
            platform: "darwin",
            host_path: Some(host_path),
            host_home,
            which,
        },
    )
}

/// The profile lines of a wrapped command, which is `[sandbox-exec, -p, <profile>, ...]`.
fn profile_lines(command: &[String]) -> Vec<String> {
    assert_eq!(&command[..2], ["/usr/bin/sandbox-exec", "-p"]);
    command[2].lines().map(str::to_string).collect()
}

fn read_rule(path: &Path) -> String {
    format!(
        "(allow file-read-data file-read-metadata (subpath \"{}\"))",
        path.display()
    )
}

fn write_rule(path: &Path) -> String {
    format!("(allow file-write* (subpath \"{}\"))", path.display())
}

/// A Python virtual environment under `base`, laid out as `host-project/<venv>/bin`.
fn virtual_environment(base: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let project = base.join("host-project");
    let root = project.join(concat!(".v", "env"));
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).expect("create the venv bin");
    std::fs::write(root.join("pyvenv.cfg"), "home = /usr/bin\n").expect("write pyvenv.cfg");
    (project, root, bin)
}

/// Python: `test_unix_local_confined_exec_command_allows_common_darwin_interpreter_roots`.
#[test]
fn confined_exec_command_allows_common_darwin_interpreter_roots() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let session = unix_local_at(&workspace);
    let home = PathBuf::from("/Users/parity-test-user");
    let path_env = format!(
        "/opt/homebrew/bin:/usr/local/bin:{}",
        home.join(".local").join("bin").display()
    );
    let expected_path = path_env.clone();
    let which = move |name: &str, path: Option<&str>| match name {
        "sandbox-exec" => Some(PathBuf::from("/usr/bin/sandbox-exec")),
        "python3" => {
            assert_eq!(path, Some(expected_path.as_str()));
            Some(PathBuf::from("/opt/homebrew/bin/python3"))
        }
        _ => None,
    };

    let command = darwin_command(
        &session,
        &["python3", "-V"],
        &workspace,
        &path_env,
        &path_env,
        &home,
        &which,
    )
    .expect("a confined command");
    let lines = profile_lines(&command);

    assert!(lines.contains(&"(deny lsopen)".to_string()));
    assert!(lines.contains(&read_rule(Path::new("/opt/homebrew"))));
    assert!(lines.contains(&read_rule(Path::new("/usr/local"))));
    assert!(lines.contains(&read_rule(&home.join(".local"))));
    assert!(lines.contains(&"(deny file-write* (subpath \"/opt\"))".to_string()));
    assert!(!lines.contains(&write_rule(Path::new("/opt/homebrew"))));
}

/// Python: `test_unix_local_confined_exec_command_allows_python_virtual_environment_root`
/// (`absolute_command` False and True).
#[test]
fn confined_exec_command_allows_a_python_virtual_environment_root() {
    for absolute_command in [false, true] {
        let dir = tempfile::tempdir().expect("temp dir");
        let workspace = dir.path().join("workspace");
        std::fs::create_dir(&workspace).expect("create workspace");
        let (project, root, bin) = virtual_environment(dir.path());
        let python = bin.join("python");
        std::fs::write(&python, "").expect("write python");
        let session = unix_local_at(&workspace);
        let path_env = bin.to_string_lossy().into_owned();
        let expected_path = path_env.clone();
        let python_text = python.to_string_lossy().into_owned();
        let which = {
            let python = python.clone();
            move |name: &str, path: Option<&str>| {
                if name == "sandbox-exec" {
                    return Some(PathBuf::from("/usr/bin/sandbox-exec"));
                }
                if name == "python" || name == python_text {
                    assert_eq!(path, Some(expected_path.as_str()));
                    return Some(python.clone());
                }
                None
            }
        };
        let program = if absolute_command {
            python.to_string_lossy().into_owned()
        } else {
            "python".to_string()
        };

        let command = darwin_command(
            &session,
            &[program.as_str(), "-V"],
            &workspace,
            &path_env,
            &path_env,
            Path::new("/Users/parity-test-user"),
            &which,
        )
        .expect("a confined command");
        let lines = profile_lines(&command);

        assert!(
            lines.contains(&read_rule(&root)),
            "absolute_command={absolute_command}"
        );
        assert!(
            !lines.contains(&read_rule(&project)),
            "the project directory is not granted"
        );
        assert!(
            !lines.contains(&write_rule(&root)),
            "the virtual environment is never writable"
        );
    }
}

/// Python: `test_unix_local_confined_exec_command_omits_host_paths_removed_from_child_path`.
#[test]
fn confined_exec_command_omits_host_paths_removed_from_the_child_path() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let (_project, root, bin) = virtual_environment(dir.path());
    let host_bin = dir.path().join("host-bin");
    std::fs::create_dir(&host_bin).expect("create host-bin");
    let session = unix_local_at(&workspace);
    let host_path = format!("{}:{}:/usr/bin", bin.display(), host_bin.display());
    let which = |name: &str, path: Option<&str>| match name {
        "sandbox-exec" => Some(PathBuf::from("/usr/bin/sandbox-exec")),
        "sh" => {
            assert_eq!(path, Some("/usr/bin:/bin"));
            Some(PathBuf::from("/bin/sh"))
        }
        _ => None,
    };

    let command = darwin_command(
        &session,
        &["sh", "-c", "true"],
        &workspace,
        "/usr/bin:/bin",
        &host_path,
        Path::new("/Users/parity-test-user"),
        &which,
    )
    .expect("a confined command");
    let profile = &command[2];

    for excluded in [&root, &bin, &host_bin] {
        for spelled in [excluded.clone(), excluded.canonicalize().expect("resolve")] {
            assert!(
                !profile.contains(&format!("(subpath \"{}\")", spelled.display())),
                "{} must not appear in the profile",
                spelled.display()
            );
        }
    }
}

/// Python: `test_unix_local_virtual_environment_grants_follow_client_path_inheritance`
/// (`resume` and `inherit_path`, four combinations).
///
/// The client decides whether a retained host `PATH` entry may widen a `bin` grant to its virtual
/// environment root; the manifest cannot.
#[tokio::test]
async fn virtual_environment_grants_follow_client_path_inheritance() {
    for resume in [false, true] {
        for inherit_path in [false, true] {
            let dir = tempfile::tempdir().expect("temp dir");
            let workspace = dir.path().join("workspace");
            std::fs::create_dir(&workspace).expect("create workspace");
            let (_project, root, bin) = virtual_environment(dir.path());
            let path_env = bin.to_string_lossy().into_owned();
            let manifest = Manifest::new()
                .with_root(workspace.to_string_lossy().into_owned())
                .with_env("PATH", path_env.clone());
            let allowlist: Vec<&str> = if inherit_path {
                vec!["PATH"]
            } else {
                Vec::new()
            };
            let client = UnixLocalSandboxClient::new()
                .with_inherit_host_environment(false)
                .with_host_environment_allowlist(allowlist);
            let session = if resume {
                let original = UnixLocalSandboxClient::new()
                    .create(Some(manifest.clone()))
                    .await
                    .expect("create the original");
                client
                    .resume(&original.session_state(), manifest.clone())
                    .await
                    .expect("resume")
            } else {
                client.create(Some(manifest.clone())).await.expect("create")
            };
            let inner = session
                .as_any()
                .and_then(|any| any.downcast_ref::<UnixLocalSandboxSession>())
                .expect("a Unix-local session");

            let expected_path = path_env.clone();
            let which = move |name: &str, path: Option<&str>| match name {
                "sandbox-exec" => Some(PathBuf::from("/usr/bin/sandbox-exec")),
                "sh" => {
                    assert_eq!(path, Some(expected_path.as_str()));
                    Some(PathBuf::from("/bin/sh"))
                }
                _ => None,
            };
            let command = darwin_command(
                inner,
                &["sh", "-c", "true"],
                &workspace,
                &path_env,
                &path_env,
                Path::new("/Users/parity-test-user"),
                &which,
            )
            .expect("a confined command");
            let lines = profile_lines(&command);

            let label = format!("resume={resume} inherit_path={inherit_path}");
            assert!(lines.contains(&read_rule(&bin)), "{label}");
            assert_eq!(lines.contains(&read_rule(&root)), inherit_path, "{label}");
        }
    }
}

/// Python: `test_unix_local_confined_exec_command_does_not_expand_manifest_virtual_environment`.
#[test]
fn confined_exec_command_does_not_expand_a_manifest_virtual_environment() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let (project, root, bin) = virtual_environment(dir.path());
    let python = bin.join("python");
    std::fs::write(&python, "").expect("write python");
    let session = unix_local_at(&workspace);
    let manifest_path = bin.to_string_lossy().into_owned();
    let expected_path = manifest_path.clone();
    let which = {
        let python = python.clone();
        move |name: &str, path: Option<&str>| match name {
            "sandbox-exec" => Some(PathBuf::from("/usr/bin/sandbox-exec")),
            "python" => {
                assert_eq!(path, Some(expected_path.as_str()));
                Some(python.clone())
            }
            _ => None,
        }
    };

    let command = darwin_command(
        &session,
        &["python", "-V"],
        &workspace,
        &manifest_path,
        "/usr/bin:/bin",
        Path::new("/Users/parity-test-user"),
        &which,
    )
    .expect("a confined command");
    let lines = profile_lines(&command);

    assert!(lines.contains(&read_rule(&bin)));
    assert!(!lines.contains(&read_rule(&root)));
    assert!(!lines.contains(&read_rule(&project)));
}

/// Build a session over `workspace` with `grants`, the way `UnixLocalSandboxSession.from_state`
/// does.
fn session_with_grants(workspace: &Path, grants: &[SandboxPathGrant]) -> UnixLocalSandboxSession {
    let mut manifest = Manifest::new().with_root(workspace.to_string_lossy().into_owned());
    for grant in grants {
        manifest = manifest.with_extra_path_grant(grant.clone());
    }
    UnixLocalSandboxSession::new(manifest, false).expect("a session")
}

fn profile_for(session: &UnixLocalSandboxSession, workspace: &Path) -> Vec<String> {
    let grants = confinement::extra_path_grant_roots(session.manifest().extra_path_grants())
        .expect("grant roots");
    confinement::exec_profile(workspace, &[], &grants)
        .lines()
        .map(str::to_string)
        .collect()
}

/// Python: `test_unix_local_darwin_exec_profile_allows_extra_path_grants`.
#[test]
fn darwin_exec_profile_allows_extra_path_grants() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let read_write = dir.path().join("read-write");
    let read_only = dir.path().join("read-only");
    for path in [&workspace, &read_write, &read_only] {
        std::fs::create_dir(path).expect("create directory");
    }
    let session = session_with_grants(
        &workspace,
        &[
            SandboxPathGrant::new(read_write.to_string_lossy()).expect("grant"),
            SandboxPathGrant::new(read_only.to_string_lossy())
                .expect("grant")
                .read_only(true),
        ],
    );

    let lines = profile_for(&session, &workspace);

    assert!(lines.contains(&read_rule(&read_write)));
    assert!(lines.contains(&write_rule(&read_write)));
    assert!(lines.contains(&read_rule(&read_only)));
    assert!(
        !lines.contains(&write_rule(&read_only)),
        "a read-only grant gets no write rule"
    );
}

/// Python: `test_unix_local_darwin_exec_profile_denies_nested_read_only_extra_path_grant`.
///
/// Rules are ordered and the last match wins, so the read-only deny has to come after the
/// read-write allow it is nested inside.
#[test]
fn darwin_exec_profile_denies_a_nested_read_only_extra_path_grant() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    let read_write = dir.path().join("read-write");
    let read_only = read_write.join("protected");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::create_dir_all(&read_only).expect("create protected");
    let session = session_with_grants(
        &workspace,
        &[
            SandboxPathGrant::new(read_write.to_string_lossy()).expect("grant"),
            SandboxPathGrant::new(read_only.to_string_lossy())
                .expect("grant")
                .read_only(true),
        ],
    );

    let lines = profile_for(&session, &workspace);
    let parent_allow = write_rule(&read_write);
    let child_deny = format!("(deny file-write* (subpath \"{}\"))", read_only.display());
    let position = |rule: &str| lines.iter().position(|line| line == rule);

    assert!(position(&parent_allow).is_some());
    assert!(position(&child_deny).is_some());
    assert!(position(&parent_allow) < position(&child_deny));
    assert!(!lines.contains(&write_rule(&read_only)));
}

/// Python: `test_unix_local_darwin_exec_profile_rejects_extra_path_grant_symlink_to_root`.
#[test]
fn darwin_exec_profile_rejects_an_extra_path_grant_symlink_to_root() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let root_alias = dir.path().join("root-alias");
    symlink("/", &root_alias);
    let session = session_with_grants(
        &workspace,
        &[SandboxPathGrant::new(root_alias.to_string_lossy()).expect("grant")],
    );

    let error = confinement::extra_path_grant_roots(session.manifest().extra_path_grants())
        .expect_err("a grant that resolves to `/` is refused");

    assert_eq!(
        error.message,
        "sandbox path grant path must not resolve to filesystem root"
    );
}

/// Python: `test_unix_local_exec_rejects_when_confinement_is_unavailable`.
#[test]
fn exec_is_rejected_when_confinement_is_unavailable() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let session = unix_local_at(&workspace);
    let which = |_: &str, _: Option<&str>| None;

    let error = darwin_command(
        &session,
        &["pwd"],
        &workspace,
        "/usr/bin:/bin",
        "/usr/bin:/bin",
        Path::new("/Users/parity-test-user"),
        &which,
    )
    .expect_err("no sandbox-exec means no run");

    assert_eq!(error.code, ErrorCode::ExecTransportError);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("unix_local_confinement_unavailable")
    );
    assert_eq!(
        error.context.get("platform").map(String::as_str),
        Some("darwin")
    );
    assert_eq!(
        error.context.get("workspace_root").map(String::as_str),
        Some(workspace.to_string_lossy().as_ref())
    );
}

/// Python: `test_unix_local_exec_runs_without_wrapper_on_linux`.
///
/// Off macOS the command is returned untouched, and a real command still runs in the workspace.
#[tokio::test]
async fn exec_runs_without_a_wrapper_off_macos() {
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create workspace");
    let session = unix_local_at(&workspace);
    let which = |_: &str, _: Option<&str>| None;
    let unchanged = session
        .confined_exec_command(
            &words(&["pwd"]),
            &workspace,
            Some("/usr/bin:/bin"),
            &HostFacts {
                is_darwin: false,
                platform: "linux",
                host_path: None,
                host_home: Path::new("/"),
                which: &which,
            },
        )
        .expect("an unwrapped command");
    assert_eq!(unchanged, words(&["pwd"]));

    let client = UnixLocalSandboxClient::new();
    let started = client
        .create(Some(
            Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
        ))
        .await
        .expect("create");
    started.start().await.expect("start");
    let result = started.exec(ExecRequest::shell("pwd")).await.expect("pwd");
    assert!(result.ok(), "{}", result.stderr_text());
    assert_eq!(
        result.stdout_text().trim(),
        workspace.canonicalize().expect("resolve").to_string_lossy()
    );
}

/// Python: `test_unix_local_exec_confines_commands_to_workspace_root` (macOS only, as Python).
#[cfg(target_os = "macos")]
#[tokio::test]
async fn exec_confines_commands_to_the_workspace_root() {
    if !Path::new("/usr/bin/sandbox-exec").exists() {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let workspace = dir.path().join("unix-local-exec");
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new().with_root(workspace.to_string_lossy().into_owned()),
        ))
        .await
        .expect("create");
    session.start().await.expect("start");

    let result = session
        .exec(ExecRequest::shell("echo hi > note.txt && cat note.txt"))
        .await
        .expect("a command in the workspace");
    assert!(result.ok(), "{}", result.stderr_text());
    assert!(result.stdout_text().trim().ends_with("hi"));

    let forbidden = session
        .exec(ExecRequest::shell("cat /etc/passwd >/dev/null"))
        .await
        .expect("exec");
    assert!(!forbidden.ok(), "reading /etc is denied");

    let outside_write = session
        .exec(ExecRequest::shell("echo nope > /usr/local/test-sandbox"))
        .await
        .expect("exec");
    assert!(!outside_write.ok(), "writing under /usr/local is denied");

    let sibling = dir.path().join("escape.txt");
    let escaped = session
        .exec(ExecRequest::shell("echo nope > ../escape.txt"))
        .await
        .expect("exec");
    assert!(!escaped.ok(), "writing beside the workspace is denied");
    assert!(!sibling.exists());
}

/// The probe half of the Launch Services test. It does nothing unless the parent test asks.
///
/// Python runs `python -c` inside the confined session. A Rust test has no interpreter to hand,
/// so the test binary runs itself: this test, selected by name, asks the OS whether `lsopen` is
/// allowed for the current process and prints the answer.
#[test]
fn lsopen_probe_child() {
    if std::env::var("PY_PARITY_LSOPEN_PROBE").is_err() {
        return;
    }
    #[cfg(target_os = "macos")]
    println!("LSOPEN={}", lsopen_check());
}

/// Ask the OS whether `lsopen` is allowed for this process: `0` allowed, otherwise denied.
#[cfg(target_os = "macos")]
fn lsopen_check() -> i32 {
    // SAFETY: `libsandbox` is a system library, `sandbox_check` has the signature declared here,
    // and the name arguments are NUL-terminated literals.
    unsafe {
        let handle = libc::dlopen(c"/usr/lib/libsandbox.1.dylib".as_ptr(), libc::RTLD_NOW);
        assert!(!handle.is_null(), "libsandbox is available");
        let symbol = libc::dlsym(handle, c"sandbox_check".as_ptr());
        assert!(!symbol.is_null(), "sandbox_check is exported");
        let check: extern "C" fn(libc::pid_t, *const libc::c_char, libc::c_int) -> libc::c_int =
            std::mem::transmute(symbol);
        check(libc::getpid(), c"lsopen".as_ptr(), 0)
    }
}

/// Python: `test_unix_local_exec_denies_launch_services` (macOS only, as Python).
#[cfg(target_os = "macos")]
#[tokio::test]
async fn exec_denies_launch_services() {
    if !Path::new("/usr/bin/sandbox-exec").exists() {
        return;
    }
    // The unconfined test process may use Launch Services; the confined child may not.
    assert_eq!(lsopen_check(), 0, "the test process is not itself confined");

    let dir = tempfile::tempdir().expect("temp dir");
    let client = UnixLocalSandboxClient::new().with_inherit_host_environment(false);
    let session = client
        .create(Some(Manifest::new().with_root(
            dir.path().join("workspace").to_string_lossy().into_owned(),
        )))
        .await
        .expect("create");
    session.start().await.expect("start");
    let executable = std::env::current_exe()
        .expect("the test binary")
        .to_string_lossy()
        .into_owned();

    let result = session
        .exec(
            ExecRequest::argv([
                executable.as_str(),
                "--exact",
                "lsopen_probe_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .with_shell(ExecShell::None)
            .with_timeout(Some(Duration::from_secs(10)))
            .with_env("PY_PARITY_LSOPEN_PROBE", "1"),
        )
        .await
        .expect("run the probe");
    assert!(result.ok(), "{}", result.stderr_text());
    let text = result.stdout_text();
    let answer: i32 = text
        .lines()
        .find_map(|line| line.split_once("LSOPEN=").map(|(_, answer)| answer))
        .unwrap_or_else(|| panic!("the probe printed no answer: {text}"))
        .trim()
        .parse()
        .expect("a number");
    assert!(answer > 0, "Launch Services is denied inside the sandbox");
}
