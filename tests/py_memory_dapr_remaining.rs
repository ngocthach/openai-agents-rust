//! Parity port of `test_metadata_write_warning_respects_data_policies` from
//! `tests/extensions/memory/test_dapr_session.py`.
//!
//! The give-up warning for a failed metadata write must obey both data policies: only the mode
//! with model-data and tool-data redaction both off may carry the session id or the provider
//! error. The logging policy and the `log` backend are process-global, so this lives in its own
//! binary and every case holds one lock.
//!
//! Python also inspects `LogRecord` internals (`exc_info`, `extra`); Rust log records carry only
//! a rendered message, so the assertions check the rendered text.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::memory::{DaprSession, Session};
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, item_text, serve};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push(format!("{}", record.args()));
    }
    fn flush(&self) {}
}

fn install_logger() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

/// A Dapr sidecar whose metadata-key writes always fail with `error_text`, while the message
/// key behaves normally.
async fn sidecar_failing_metadata(error_text: &'static str) -> String {
    let store: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
    serve(Arc::new(move |req: &Request| {
        let path = req.path.split('?').next().unwrap_or_default().to_string();
        if path == "/v1.0/metadata" {
            return Reply::json(json!({"id": "mock"}));
        }
        let Some(rest) = path.strip_prefix("/v1.0/state/statestore") else {
            return Reply::empty(404);
        };
        let key = rest.trim_start_matches('/').to_string();
        let mut store = store.lock().unwrap();
        match req.method.as_str() {
            "GET" => match store.get(&key) {
                None => Reply::empty(204),
                Some(value) => Reply {
                    status: 200,
                    headers: vec![
                        ("content-type".into(), "application/json".into()),
                        ("etag".into(), "1".into()),
                    ],
                    body: value.clone(),
                },
            },
            "POST" => {
                let entry = req.body[0].clone();
                let entry_key = entry["key"].as_str().unwrap_or_default().to_string();
                if entry_key.ends_with(":metadata") {
                    return Reply::status(500, json!({"message": error_text}).to_string());
                }
                store.insert(entry_key, entry["value"].to_string());
                Reply::empty(204)
            }
            _ => Reply::empty(405),
        }
    }))
    .await
}

async fn run_case(dont_log_model_data: bool, dont_log_tool_data: bool, redacted: bool) {
    let _guard = LOCK.lock().await;
    install_logger();
    let error_text = "etag mismatch";
    let session_id = format!("metadata_policy_{dont_log_model_data}_{dont_log_tool_data}");
    let base = sidecar_failing_metadata(error_text).await;
    let session = DaprSession::new(&session_id, "statestore", &base);

    set_dont_log_model_data(dont_log_model_data);
    set_dont_log_tool_data(dont_log_tool_data);
    RECORDS.lock().unwrap().clear();
    let added = session.add_items(vec![InputItem::user("kept")]).await;
    let records: Vec<String> = RECORDS
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.contains("could not update"))
        .cloned()
        .collect();
    reset_dont_log_model_data();
    reset_dont_log_tool_data();

    // The append landed regardless of how the failure was logged.
    added.unwrap();
    let items = session.get_items(None).await.unwrap();
    assert_eq!(
        items.iter().map(item_text).collect::<Vec<_>>(),
        vec!["kept"]
    );

    assert_eq!(records.len(), 1, "{records:?}");
    let rendered = &records[0];
    if redacted {
        assert!(!rendered.contains(&session_id), "{rendered}");
        assert!(!rendered.contains(error_text), "{rendered}");
    } else {
        assert!(rendered.contains(&session_id), "{rendered}");
        assert!(rendered.contains(error_text), "{rendered}");
    }
}

#[tokio::test]
async fn the_metadata_warning_is_redacted_when_model_data_is_not_logged() {
    run_case(true, false, true).await;
}

#[tokio::test]
async fn the_metadata_warning_is_redacted_when_tool_data_is_not_logged() {
    run_case(false, true, true).await;
}

#[tokio::test]
async fn the_metadata_warning_carries_diagnostics_only_when_both_policies_allow() {
    run_case(false, false, false).await;
}
