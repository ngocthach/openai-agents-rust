//! `test_e2b_shutdown_logs_pause_failure_and_falls_back_to_kill`,
//! `test_e2b_shutdown_logs_kill_failure_after_pause_fallback` and
//! `test_e2b_shutdown_logs_direct_kill_failure` of `tests/extensions/sandbox/test_e2b.py`:
//! shutdown never raises, and the failures are logged as warnings whose provider text follows the
//! tool-data policy.
#![cfg(feature = "sandbox")]

#[path = "py_sandbox_support/mod.rs"]
mod support;

use openai_agents_sdk::extensions::sandbox::e2b::{E2BSandboxClient, E2BSandboxClientOptions};
use openai_agents_sdk::sandbox::SandboxClient;
use serde_json::{Value, json};
use support::{MockResponse, Recorded};

struct Capture {
    records: std::sync::Mutex<Vec<(log::Level, String)>>,
}

impl log::Log for Capture {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        self.records
            .lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {}
}

static CAPTURE: Capture = Capture {
    records: std::sync::Mutex::new(Vec::new()),
};

fn created_sandbox() -> Value {
    json!({ "sandboxID": "sb-123", "envdAccessToken": "envd-token" })
}

fn warnings_containing(text: &str) -> Vec<String> {
    CAPTURE
        .records
        .lock()
        .unwrap()
        .iter()
        .filter(|(level, message)| *level == log::Level::Warn && message.contains(text))
        .map(|(_, message)| message.clone())
        .collect()
}

#[derive(Clone, Copy)]
enum Case {
    PauseFailsThenKillWorks,
    PauseFailsThenKillFails,
    DirectKillFails,
}

async fn shutdown(case: Case, secret: &str) -> (usize, usize) {
    let secret = secret.to_string();
    let server = support::start_resolving(support::Resolver::E2b, move |request: &Recorded| {
        let fail_pause = matches!(
            case,
            Case::PauseFailsThenKillWorks | Case::PauseFailsThenKillFails
        );
        let fail_kill = matches!(case, Case::PauseFailsThenKillFails | Case::DirectKillFails);
        if request.path.ends_with("/pause") && fail_pause {
            MockResponse::status(500, json!({ "message": format!("pause {secret}") }))
        } else if request.method == "DELETE" && fail_kill {
            MockResponse::status(500, json!({ "message": format!("kill {secret}") }))
        } else {
            MockResponse::json(created_sandbox())
        }
    })
    .await;
    let pause_on_exit = !matches!(case, Case::DirectKillFails);
    let session = E2BSandboxClient::with_options(
        E2BSandboxClientOptions::new()
            .with_api_key("e2b_test")
            .with_api_base_url(&server.base_url)
            .with_sandbox_base_url(&server.base_url)
            .with_pause_on_exit(pause_on_exit),
    )
    .create(None)
    .await
    .expect("create");

    session
        .shutdown()
        .await
        .expect("shutdown never raises for a release failure");
    let counts = (
        server.count("POST", "/sandboxes/sb-123/pause"),
        server.count("DELETE", "/sandboxes/sb-123"),
    );
    server.stop();
    counts
}

/// One test, because the logger is process-wide.
#[tokio::test]
async fn shutdown_failures_are_logged_and_never_raised() {
    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Debug);

    for redacted in [true, false] {
        openai_agents_sdk::debug::set_dont_log_tool_data(redacted);
        let tag = if redacted { "REDACTED" } else { "VISIBLE" };
        let cases = [
            (
                Case::PauseFailsThenKillWorks,
                "Failed to pause E2B sandbox on shutdown; falling back to kill.",
                (1, 1),
            ),
            (
                Case::PauseFailsThenKillFails,
                "Failed to kill E2B sandbox after pause fallback failure.",
                (1, 1),
            ),
            (
                Case::DirectKillFails,
                "Failed to kill E2B sandbox on shutdown.",
                (0, 1),
            ),
        ];
        for (index, (case, message, expected_calls)) in cases.into_iter().enumerate() {
            let secret = format!("SECRET_{tag}_{index}");
            let calls = shutdown(case, &secret).await;
            assert_eq!(calls, expected_calls, "{message} redacted={redacted}");
            let logged = warnings_containing(message);
            assert!(!logged.is_empty(), "{message} was not logged");
            let leaked = CAPTURE
                .records
                .lock()
                .unwrap()
                .iter()
                .any(|(_, line)| line.contains(&secret));
            assert_eq!(
                leaked, !redacted,
                "{message}: provider text must appear exactly when tool data may be logged"
            );
        }
    }
    openai_agents_sdk::debug::set_dont_log_tool_data(true);
}
