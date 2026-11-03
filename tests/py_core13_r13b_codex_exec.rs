//! Process-lifetime cases of `tests/extensions/experiemental/codex/test_codex_exec_thread.py`
//! that earlier audits carried over as python-only: closing a stream early, a read that is
//! abandoned while the process is quiet, and the cancellation signal each end a live Codex
//! process instead of leaving it behind.
//!
//! The Codex CLI is faked with a POSIX shell script that records its pid and then stays alive.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures::StreamExt;
use openai_agents_sdk::extensions::codex::{
    Codex, CodexExec, CodexExecArgs, CodexOptions, TurnOptions,
};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

/// Writes its pid, reads the prompt, prints one event (and optionally a large unread tail),
/// then `exec`s `sleep`, so a signal sent to the pid reaches the process that stays alive.
const SCRIPT: &str = r#"#!/bin/sh
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH
echo $$ > "$FAKE_PID_FILE"
cat > /dev/null
if [ -z "$FAKE_QUIET" ]; then
  echo '{"type":"thread.started","thread_id":"thread-1"}'
fi
if [ -n "$FAKE_FLOOD" ]; then
  head -c 400000 /dev/zero | tr '\0' 'x'
fi
exec sleep 60
"#;

struct Fixture {
    dir: TempDir,
    env: BTreeMap<String, String>,
}

impl Fixture {
    fn new(extra: &[(&str, &str)]) -> Self {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("codex.sh");
        std::fs::write(&script, SCRIPT).unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
        std::fs::set_permissions(&script, permissions).unwrap();
        let mut env = BTreeMap::from([(
            "FAKE_PID_FILE".to_string(),
            dir.path().join("pid").to_string_lossy().into_owned(),
        )]);
        for (key, value) in extra {
            env.insert((*key).to_string(), (*value).to_string());
        }
        Fixture { dir, env }
    }

    fn script(&self) -> String {
        self.dir
            .path()
            .join("codex.sh")
            .to_string_lossy()
            .into_owned()
    }

    fn exec(&self) -> CodexExec {
        CodexExec::new(Some(self.script()), Some(self.env.clone()))
    }

    fn client(&self) -> Codex {
        Codex::new(
            CodexOptions::new()
                .with_codex_path_override(self.script())
                .with_api_key("sk-test")
                .with_env(self.env.clone()),
        )
    }

    fn pid_file(&self) -> PathBuf {
        self.dir.path().join("pid")
    }
}

async fn recorded_pid(path: &Path) -> u32 {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(pid) = text.trim().parse()
        {
            return pid;
        }
        assert!(
            Instant::now() < deadline,
            "the fake codex never wrote its pid"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn is_running(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

async fn assert_ends(pid: u32, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while is_running(pid) {
        assert!(
            Instant::now() < deadline,
            "{what}: process {pid} is still running"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn closing_a_stream_early_ends_a_process_that_has_unread_output() {
    // `test_codex_exec_run_drains_stdout_before_waiting_for_live_process`: the process has
    // more output than a pipe holds and is still alive; closing the stream must not hang on it.
    let fixture = Fixture::new(&[("FAKE_FLOOD", "1")]);
    let mut stream = fixture
        .exec()
        .run(CodexExecArgs {
            input: "hello".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let first = stream.next().await.expect("first line").unwrap();
    assert!(first.contains("thread.started"), "{first}");
    let pid = recorded_pid(&fixture.pid_file()).await;
    assert!(is_running(pid));

    let closed = tokio::time::timeout(Duration::from_secs(3), async move { drop(stream) }).await;
    assert!(closed.is_ok(), "closing the stream hung");
    assert_ends(pid, "closing the stream").await;
}

#[tokio::test]
async fn abandoning_a_pending_read_ends_the_process() {
    // `test_codex_exec_run_cancels_pending_timeout_read_before_draining_stdout`: a read that
    // is waiting on a quiet process is dropped; the process does not outlive it.
    let fixture = Fixture::new(&[("FAKE_QUIET", "1")]);
    let mut stream = fixture
        .exec()
        .run(CodexExecArgs {
            input: "hello".into(),
            idle_timeout: Some(Duration::from_secs(60)),
            ..Default::default()
        })
        .await
        .unwrap();
    let pid = recorded_pid(&fixture.pid_file()).await;
    let reader = tokio::spawn(async move { stream.next().await });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(is_running(pid));

    reader.abort();
    assert!(reader.await.unwrap_err().is_cancelled());
    assert_ends(pid, "abandoning the read").await;
}

#[tokio::test]
async fn cancelling_the_signal_terminates_the_process_and_ends_the_turn() {
    // `test_watch_signal_terminates_process`.
    let fixture = Fixture::new(&[("FAKE_QUIET", "1")]);
    let cancel = CancellationToken::new();
    let mut streamed = fixture
        .client()
        .start_thread(None)
        .run_streamed(
            "hello",
            Some(TurnOptions::new().with_cancel(cancel.clone())),
        )
        .await
        .unwrap();
    let pid = recorded_pid(&fixture.pid_file()).await;
    assert!(is_running(pid));

    let waiter = tokio::spawn(async move {
        let mut last = None;
        while let Some(event) = streamed.next_event().await {
            last = Some(event);
        }
        last
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    cancel.cancel();

    let last = tokio::time::timeout(Duration::from_secs(5), waiter)
        .await
        .expect("the turn did not end after the signal")
        .unwrap();
    let error = last.expect("an event").unwrap_err();
    assert!(
        error.to_string().contains("Codex exec exited with code"),
        "{error}"
    );
    assert_ends(pid, "the cancellation signal").await;
}

#[test]
fn a_cleanup_that_cannot_remove_the_schema_file_does_not_fail_the_caller() {
    // `test_output_schema_file_cleanup_swallows_rmtree_errors`: the directory is made
    // read-only, so removing the schema file fails; dropping the handle still returns.
    use std::os::unix::fs::PermissionsExt;
    let schema = serde_json::json!({"type": "object"});
    let file = openai_agents_sdk::extensions::codex::OutputSchemaFile::create(&schema).unwrap();
    let path = file.path().to_path_buf();
    let dir = path.parent().unwrap().to_path_buf();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();

    drop(file);

    // Whether the removal succeeded depends on the privileges the tests run with; either way
    // the drop returned, and the leftovers are cleaned here.
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).ok();
    let _ = std::fs::remove_dir_all(&dir);
}
