//! A PTY child must start with default signal dispositions, whatever the parent ignores.
//!
//! Python's `unix_local.py` passes `preexec_fn=_restore_pty_child_signal_defaults`, resetting
//! `SIGINT` and `SIGQUIT` to `SIG_DFL` between fork and exec. The Rust port never installs such
//! a hook, because `portable-pty` already does the equivalent inside `spawn_command`
//! (`portable-pty-0.9.0/src/unix.rs`: `SIG_DFL` for `SIGCHLD`, `SIGHUP`, `SIGINT`, `SIGQUIT`,
//! `SIGTERM` and `SIGALRM`, then `sigprocmask(SIG_SETMASK, &empty_set)`), and `std`'s own
//! `do_exec` resets `SIGPIPE` — the one disposition the Rust runtime leaves ignored — before any
//! `pre_exec` closure runs.
//!
//! This test proves the behaviour rather than the mechanism: it makes the test process ignore
//! each signal, then has a PTY child signal itself. A child that inherited `SIG_IGN` survives
//! and prints; a child with the default disposition is killed by it.
//!
//! It lives in its own test binary because it mutates process-wide signal dispositions, which
//! must not race other tests. `/proc` is unavailable on macOS, so the child reports by living
//! or dying rather than by reading its own status file.

#![cfg(all(unix, feature = "sandbox"))]

use std::sync::Mutex;

use openai_agents_sdk::sandbox::pty::PtySpawn;
use openai_agents_sdk::sandbox::{PtyExecUpdate, PtyManager};

/// Serializes the tests: signal dispositions are process-wide, and the harness runs `#[test]`
/// functions on parallel threads, so one test's restore must not land inside another's window.
static DISPOSITIONS: Mutex<()> = Mutex::new(());

/// Ignore `signum` for the duration of `body`, restoring the previous disposition afterwards.
fn while_ignoring<T>(signum: i32, body: impl FnOnce() -> T) -> T {
    let _guard = DISPOSITIONS.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: `signal` with `SIG_IGN` is async-signal-safe, and the lock above means no other
    // test in this binary is reading or writing a disposition for the duration.
    let previous = unsafe { libc::signal(signum, libc::SIG_IGN) };
    let result = body();
    unsafe { libc::signal(signum, previous) };
    result
}

async fn pty_output(script: &str) -> PtyExecUpdate {
    let manager = PtyManager::new();
    let (update, _cleanup) = manager
        .start(
            PtySpawn {
                argv: vec!["sh".into(), "-c".into(), script.into()],
                cwd: None,
                env: Vec::new(),
                clear_env: false,
                tty: false,
                remote_kill_command: None,
            },
            // The child signals itself immediately; the window only has to outlast that.
            2_000,
            None,
        )
        .await
        .expect("start a pty process");
    update
}

/// `sh` prints only if it survived signalling itself, which it does only under `SIG_IGN`.
fn script(signal_name: &str) -> String {
    format!("kill -{signal_name} $$; echo INHERITED_SIG_IGN")
}

#[test]
fn a_pty_child_starts_with_default_signal_dispositions() {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");

    for (name, signum) in [
        ("INT", libc::SIGINT),
        ("QUIT", libc::SIGQUIT),
        ("TERM", libc::SIGTERM),
        ("HUP", libc::SIGHUP),
    ] {
        let update = while_ignoring(signum, || runtime.block_on(pty_output(&script(name))));
        assert!(
            !update.output_text().contains("INHERITED_SIG_IGN"),
            "a PTY child inherited SIG_IGN for SIG{name}: {:?}",
            update.output_text()
        );
    }

    // SIGPIPE needs no parent mutation: the Rust runtime already ignores it process-wide, so an
    // unreset disposition would leak into every PTY child on its own.
    let update = runtime.block_on(pty_output(&script("PIPE")));
    assert!(
        !update.output_text().contains("INHERITED_SIG_IGN"),
        "a PTY child inherited the runtime's ignored SIGPIPE: {:?}",
        update.output_text()
    );
}

/// The parent's ignored disposition really is observable when it is inherited, so the assertions
/// above are not vacuously true. A plain `std::process::Command` child does inherit `SIG_IGN`.
#[test]
fn the_probe_detects_an_inherited_sig_ign() {
    let output = while_ignoring(libc::SIGINT, || {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(script("INT"))
            .output()
            .expect("run sh")
    });
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("INHERITED_SIG_IGN"),
        "the probe cannot detect an inherited SIG_IGN, so the PTY assertions prove nothing"
    );
}
