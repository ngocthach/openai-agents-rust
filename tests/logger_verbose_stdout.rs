//! `enable_verbose_stdout_logging` installs a stdout `log` backend for SDK records only.
//!
//! A `log` backend is a per-process global, so each behavior runs in its own process: the tests
//! named `child_*` re-run this binary with a single test selected and an env marker set, and the
//! parent asserts on what the child wrote to stdout.

use std::process::Command;

use log::{Log, Metadata, Record};
use openai_agents_sdk::enable_verbose_stdout_logging;
use openai_agents_sdk::logger::LOG_TARGET;

const CHILD_MARKER: &str = "OARS_LOGGER_TEST_CHILD";

/// Whether this process is the re-executed child of one of the tests below.
fn is_child() -> bool {
    std::env::var(CHILD_MARKER).is_ok()
}

/// Run this test binary again with only `test_name` selected, and return its stdout.
fn run_child(test_name: &str) -> String {
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .args([test_name, "--exact", "--nocapture"])
        .env(CHILD_MARKER, "1")
        .output()
        .expect("the test binary re-runs");
    assert!(
        output.status.success(),
        "child test {test_name} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn child_prints_sdk_records_only() {
    if !is_child() {
        return;
    }
    assert!(enable_verbose_stdout_logging(), "installs the backend");
    assert!(
        enable_verbose_stdout_logging(),
        "a second call is a no-op reporting the same state"
    );

    log::debug!(target: LOG_TARGET, "sdk-debug-line");
    log::debug!(target: "openai.agents.realtime", "sdk-child-target-line");
    log::info!(target: LOG_TARGET, "sdk-info-line");
    log::trace!(target: LOG_TARGET, "sdk-trace-line");
    log::debug!(target: "some_other_crate", "other-crate-line");
    log::logger().flush();
}

#[test]
fn verbose_logging_prints_sdk_debug_records_to_stdout() {
    if is_child() {
        return;
    }
    let stdout = run_child("child_prints_sdk_records_only");

    assert!(
        stdout.contains("DEBUG openai.agents: sdk-debug-line"),
        "SDK debug records reach stdout: {stdout}"
    );
    assert!(
        stdout.contains("sdk-child-target-line"),
        "child targets of openai.agents reach stdout too: {stdout}"
    );
    assert!(
        stdout.contains("INFO openai.agents: sdk-info-line"),
        "records above Debug reach stdout: {stdout}"
    );
    assert!(
        !stdout.contains("sdk-trace-line"),
        "Trace is below the Debug filter: {stdout}"
    );
    assert!(
        !stdout.contains("other-crate-line"),
        "other crates stay quiet: {stdout}"
    );
}

#[test]
fn child_logs_from_an_sdk_module() {
    if !is_child() {
        return;
    }
    assert!(enable_verbose_stdout_logging(), "installs the backend");

    // A real SDK debug line, from a module that does not go through `logger`'s own helpers.
    // Python logs it through `openai.agents` (`models/openai_chatcompletions.py` imports the
    // shared logger), so it has to reach the verbose backend here too.
    openai_agents_sdk::models::openai_chatcompletions::items_to_messages(
        &[openai_agents_sdk::items::InputItem(serde_json::json!({
            "type": "custom_backend_item",
        }))],
        None,
    )
    .expect("an unknown item type is skipped, not an error");
    log::logger().flush();
}

#[test]
fn an_sdk_modules_own_records_reach_the_verbose_backend() {
    if is_child() {
        return;
    }
    let stdout = run_child("child_logs_from_an_sdk_module");
    assert!(
        stdout.contains(
            "DEBUG openai.agents: Skipping item type custom_backend_item not supported by Chat Completions"
        ),
        "a module's own debug record is not filtered out by its module path: {stdout}"
    );
}

/// A backend that drops everything, standing in for an application's own logger.
#[derive(Debug)]
struct SilentLogger;

static SILENT_LOGGER: SilentLogger = SilentLogger;

impl Log for SilentLogger {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        false
    }
    fn log(&self, _record: &Record<'_>) {}
    fn flush(&self) {}
}

#[test]
fn child_reports_false_when_a_backend_is_already_installed() {
    if !is_child() {
        return;
    }
    log::set_logger(&SILENT_LOGGER).expect("the application installs its backend first");
    log::set_max_level(log::LevelFilter::Info);

    assert!(
        !enable_verbose_stdout_logging(),
        "the existing backend is left alone"
    );
    assert_eq!(
        log::max_level(),
        log::LevelFilter::Info,
        "the level filter of the existing backend is not raised"
    );
    log::debug!(target: LOG_TARGET, "must-not-appear");
}

#[test]
fn an_existing_backend_is_never_replaced() {
    if is_child() {
        return;
    }
    let stdout = run_child("child_reports_false_when_a_backend_is_already_installed");
    assert!(
        !stdout.contains("must-not-appear"),
        "nothing is printed when another backend owns the process: {stdout}"
    );
}
