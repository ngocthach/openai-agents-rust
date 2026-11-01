//! Ports of the log-redaction tests of `tests/realtime/test_openai_realtime.py`: an invalid
//! server event and a raw client message that cannot be converted log only a fixed message when
//! model-data logging is off, and keep the payload for diagnostics when it is on.
#![cfg(feature = "realtime")]

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use openai_agents_sdk::debug::{reset_dont_log_model_data, set_dont_log_model_data};
use openai_agents_sdk::realtime::{
    RealtimeModel, RealtimeModelEvent, RealtimeModelEvents, RealtimeModelSendEvent,
};
use serde_json::json;

#[path = "py_realtime_support/mod.rs"]
mod support;
use support::*;

/// The redaction flag and the log sink are process-global, so the tests take turns.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct LogSink(Mutex<Vec<(log::Level, String, String)>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        self.0.lock().unwrap().push((
            record.level(),
            record.target().to_string(),
            record.args().to_string(),
        ));
    }
    fn flush(&self) {}
}

fn sink() -> &'static LogSink {
    static SINK: OnceLock<LogSink> = OnceLock::new();
    let sink = SINK.get_or_init(|| LogSink(Mutex::new(Vec::new())));
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

struct PolicyGuard;

impl PolicyGuard {
    fn set(model_redacted: bool) -> Self {
        set_dont_log_model_data(model_redacted);
        PolicyGuard
    }
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        reset_dont_log_model_data();
    }
}

/// Every SDK log line recorded so far, as `"message"` (the SDK target only).
fn sdk_lines(sink: &LogSink) -> Vec<String> {
    sink.0
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, target, _)| target == openai_agents_sdk::logger::LOG_TARGET)
        .map(|(_, _, line)| line.clone())
        .collect()
}

/// Every log line the SDK itself produced, under any of its targets (never a vendor crate's).
fn sdk_target_lines(sink: &LogSink) -> Vec<String> {
    sink.0
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, target, _)| target.starts_with("openai"))
        .map(|(_, _, line)| line.clone())
        .collect()
}

async fn emitted_until_error(events: &mut RealtimeModelEvents) -> Vec<RealtimeModelEvent> {
    let mut seen = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(event) = events.recv().await {
            let is_error = matches!(event, RealtimeModelEvent::Error { .. });
            seen.push(event);
            if is_error {
                return;
            }
        }
    })
    .await
    .expect("an error event arrived");
    seen
}

#[tokio::test]
async fn an_invalid_server_event_follows_the_model_data_policy() {
    let _turn = SERIAL.lock().await;
    let sink = sink();

    for redacted in [true, false] {
        let _policy = PolicyGuard::set(redacted);
        sink.0.lock().unwrap().clear();
        let mut server = MockRealtimeServer::start().await;
        let (model, mut events) = connect(&server, None, None).await;
        let _ = server.next_frame("session.update").await;
        sink.0.lock().unwrap().clear();

        server.push(json!({
            "type": "diagnostic.event",
            "event_id": "diagnostic_event_id",
            "delta": "diagnostic payload",
        }));
        let emitted = emitted_until_error(&mut events).await;

        // The listener still sees the failure as an error event after the raw event.
        assert!(matches!(
            emitted.last(),
            Some(RealtimeModelEvent::Error { error })
                if error["context"] == "Failed to validate server event"
        ));
        let lines = sdk_lines(sink).join("\n");
        let everything = sdk_target_lines(sink);
        if redacted {
            assert!(
                sdk_lines(sink)
                    .iter()
                    .any(|l| l == "Failed to validate server event"),
                "{lines}"
            );
            for secret in [
                "diagnostic.event",
                "diagnostic_event_id",
                "diagnostic payload",
            ] {
                assert!(
                    everything.iter().all(|l| !l.contains(secret)),
                    "{secret} leaked: {everything:?}"
                );
            }
        } else {
            for diagnostic in [
                "diagnostic.event",
                "diagnostic_event_id",
                "diagnostic payload",
            ] {
                assert!(lines.contains(diagnostic), "{diagnostic} missing: {lines}");
            }
        }
        model.close().await.unwrap();
        server.stop();
    }
}

#[tokio::test]
async fn a_raw_message_conversion_failure_follows_the_model_data_policy() {
    let _turn = SERIAL.lock().await;
    let sink = sink();

    for redacted in [true, false] {
        let _policy = PolicyGuard::set(redacted);
        let mut server = MockRealtimeServer::start().await;
        let (model, _events) = connect(&server, None, None).await;
        let _ = server.next_frame("session.update").await;
        sink.0.lock().unwrap().clear();

        model
            .send_event(RealtimeModelSendEvent::RawMessage {
                message: json!({
                    "type": "diagnostic.raw.event",
                    "other_data": {"transcript": "diagnostic transcript"},
                }),
            })
            .await
            .unwrap();

        let lines = sdk_lines(sink).join("\n");
        let everything = sdk_target_lines(sink);
        if redacted {
            assert!(
                sdk_lines(sink)
                    .iter()
                    .any(|l| l == "Failed to convert raw message"),
                "{lines}"
            );
            for secret in ["diagnostic.raw.event", "diagnostic transcript"] {
                assert!(
                    everything.iter().all(|l| !l.contains(secret)),
                    "{secret} leaked: {everything:?}"
                );
            }
        } else {
            for diagnostic in ["diagnostic.raw.event", "diagnostic transcript"] {
                assert!(lines.contains(diagnostic), "{diagnostic} missing: {lines}");
            }
        }
        model.close().await.unwrap();
        server.stop();
    }
}
