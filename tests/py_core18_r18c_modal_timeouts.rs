//! Ports of the Modal control-plane bound tests in `tests/extensions/sandbox/test_modal.py`.
//!
//! Python wraps each SDK call in `_call_modal(..., call_timeout=...)`. The Rust client drives the
//! bundled `helper.py`, which awaits the same SDK calls, so the driver runs the real helper
//! against fake SDK objects and records the bound every call is awaited under.
#![cfg(feature = "sandbox")]

use serde_json::{Value, json};

fn run_driver(case: &str) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let output = std::process::Command::new("python3")
        .arg(format!("{manifest}/tests/fixtures/modal_helper_driver.py"))
        .arg(format!("{manifest}/src/extensions/sandbox/modal/helper.py"))
        .arg(case)
        .output()
        .expect("run python3");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("driver output")
}

/// Python: `test_modal_ensure_sandbox_bounds_app_lookup` and
/// `test_modal_ensure_sandbox_bounds_image_id_lookup`: creating a sandbox looks the app up under
/// a ten-second bound, and an image id is reused without another control-plane call.
/// (`Image.from_registry` is a lazy constructor in the helper, so it is not awaited at all.)
#[test]
fn creating_a_sandbox_bounds_the_app_lookup() {
    let result = run_driver("call_timeouts");

    let create = &result["create"];
    assert_eq!(create["response"]["sandbox_id"], "sb-new");
    assert_eq!(
        create["calls"],
        json!([
            "App.lookup",
            "Image.from_registry:python:3.14-slim",
            "Sandbox.create"
        ])
    );
    assert_eq!(create["timeouts"], json!([10.0]));

    let from_image = &result["create_from_image"];
    assert_eq!(
        from_image["calls"],
        json!(["App.lookup", "Image.from_id:im-existing", "Sandbox.create"])
    );
    assert_eq!(from_image["timeouts"], json!([10.0]));
}

/// Python: `test_modal_shutdown_rehydrates_sandbox_and_terminates_without_wait_kwarg` (the
/// bounds): reconnecting to a sandbox by id and terminating it are each bounded by the default
/// thirty seconds, and so are a reconnect and a poll.
#[test]
fn reconnecting_polling_and_terminating_are_bounded() {
    let result = run_driver("call_timeouts");

    assert_eq!(result["terminate"]["response"], json!({"ok": true}));
    assert_eq!(
        result["terminate"]["calls"],
        json!(["Sandbox.from_id", "terminate"])
    );
    assert_eq!(result["terminate"]["timeouts"], json!([30.0, 30.0]));

    assert_eq!(result["poll"]["response"], json!({"running": true}));
    assert_eq!(result["poll"]["calls"], json!(["Sandbox.from_id", "poll"]));
    assert_eq!(result["poll"]["timeouts"], json!([30.0, 30.0]));
}
