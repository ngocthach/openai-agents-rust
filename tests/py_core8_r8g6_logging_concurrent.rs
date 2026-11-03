//! Port of `test_agents_logging.py::test_enable_verbose_stdout_logging_serializes_handler_initialization`:
//! concurrent `enable_verbose_stdout_logging()` calls install the backend once, every caller
//! sees it installed, and a record is printed once, not once per caller.
//!
//! The `log` backend is a per-process global, so the check runs in a re-executed child process
//! (the technique of `logger_verbose_stdout.rs`) and the parent reads the child's stdout.
//! The other Python cases (`follows_replaced_stdout`, `falls_back_to_stderr`) swap `sys.stdout`
//! on a live handler; the Rust backend writes through the process's stdout handle, so there is
//! no replaceable stream to follow.

use std::process::Command;
use std::sync::{Arc, Barrier};

use openai_agents_sdk::enable_verbose_stdout_logging;
use openai_agents_sdk::logger::LOG_TARGET;

const CHILD_MARKER: &str = "OARS_R8G6_LOGGER_CHILD";

#[test]
fn child_enables_logging_from_many_threads() {
    if std::env::var(CHILD_MARKER).is_err() {
        return;
    }
    let threads = 8;
    let barrier = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                enable_verbose_stdout_logging()
            })
        })
        .collect();
    for handle in handles {
        assert!(
            handle.join().unwrap(),
            "every caller sees the backend installed"
        );
    }
    log::debug!(target: LOG_TARGET, "debug message");
    log::logger().flush();
}

#[test]
fn concurrent_enable_calls_install_one_backend() {
    if std::env::var(CHILD_MARKER).is_ok() {
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .args([
            "child_enables_logging_from_many_threads",
            "--exact",
            "--nocapture",
        ])
        .env(CHILD_MARKER, "1")
        .output()
        .expect("the test binary re-runs");
    assert!(
        output.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let printed = stdout
        .lines()
        .filter(|line| *line == "DEBUG openai.agents: debug message")
        .count();
    assert_eq!(printed, 1, "the record is printed once: {stdout}");
}
