//! Port of `test_output_guardrail_failure_follows_both_data_policies`
//! (`tests/realtime/test_session_guardrails.py`): an output guardrail that raises is skipped, and
//! the log line keeps the error text and the guardrail name only when neither `dont_log_model_data`
//! nor `dont_log_tool_data` is set.
#![cfg(feature = "realtime")]

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeAudioFormat, RealtimeGuardrailsSettings,
    RealtimeModelEvent, RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::{AgentsError, OutputGuardrail};
use serde_json::json;

/// The redaction flags and the log sink are process-global, so the tests take turns.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct LogSink(Mutex<Vec<(log::Level, String)>>);

impl log::Log for LogSink {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0
                .lock()
                .unwrap()
                .push((record.level(), record.args().to_string()));
        }
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

/// Restores the redaction flags to the environment defaults when a test ends.
struct PolicyGuard;

impl PolicyGuard {
    fn set(model_redacted: bool, tool_redacted: bool) -> Self {
        set_dont_log_model_data(model_redacted);
        set_dont_log_tool_data(tool_redacted);
        PolicyGuard
    }
}

impl Drop for PolicyGuard {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

async fn settle() {
    for _ in 0..300 {
        tokio::task::yield_now().await;
    }
}

#[tokio::test]
async fn output_guardrail_failure_follows_both_data_policies() {
    const ERROR: &str = "SECRET_REALTIME_GUARDRAIL_ERROR";
    const NAME: &str = "SECRET_REALTIME_GUARDRAIL_NAME";
    let _turn = SERIAL.lock().await;
    let sink = sink();

    for (model_redacted, tool_redacted) in [(true, false), (false, true), (false, false)] {
        let _policy = PolicyGuard::set(model_redacted, tool_redacted);
        sink.0.lock().unwrap().clear();

        let failing = OutputGuardrail::<()>::new(NAME, |_ctx, _agent, _output| async move {
            Err(AgentsError::other(ERROR))
        });
        let model = Arc::new(FakeRealtimeModel::new());
        let session = RealtimeSession::connect(
            model.clone(),
            RealtimeAgent::<()>::new("agent").into_arc(),
            (),
            None,
            Some(RealtimeRunConfig::<()> {
                output_guardrails: vec![failing],
                guardrails_settings: RealtimeGuardrailsSettings {
                    debounce_text_length: 5,
                },
                ..Default::default()
            }),
        )
        .await
        .expect("session connects");

        model.emit(RealtimeModelEvent::TranscriptDelta {
            item_id: "item_1".into(),
            delta: "model text".into(),
            response_id: "response-id".into(),
        });
        settle().await;
        assert!(model.emit(RealtimeModelEvent::Other {
            data: json!({"marker": true}),
        }));
        settle().await;
        let mut tripped = false;
        while let Ok(Some(event)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), session.recv()).await
        {
            tripped |= matches!(event, RealtimeSessionEvent::GuardrailTripped { .. });
        }
        assert!(!tripped);

        let records: Vec<(log::Level, String)> = sink
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, message)| message.contains("Output guardrail raised an exception"))
            .cloned()
            .collect();
        let case = format!("model={model_redacted} tool={tool_redacted}");
        assert_eq!(records.len(), 1, "{case}: {records:?}");
        let (level, record) = &records[0];
        assert_eq!(*level, log::Level::Warn, "{case}");
        if model_redacted || tool_redacted {
            assert_eq!(
                record, "Output guardrail raised an exception; skipping it",
                "{case}"
            );
            assert!(
                !record.contains(ERROR) && !record.contains(NAME),
                "{case}: {record}"
            );
        } else {
            assert!(record.contains(ERROR), "{case}: {record}");
            assert!(record.contains(NAME), "{case}: {record}");
        }
    }
}

/// `test_to_realtime_audio_format_redacts_unknown_values` and
/// `test_to_realtime_audio_format_preserves_diagnostic_mapping` with the tool-data flag varied:
/// it has no effect on a diagnostic, which follows the model-data flag alone.
#[tokio::test]
async fn audio_format_diagnostics_follow_the_model_policy_for_either_tool_flag() {
    let _turn = SERIAL.lock().await;
    let sink = sink();
    let messages = || -> Vec<String> {
        sink.0
            .lock()
            .unwrap()
            .iter()
            .map(|(_, m)| m.clone())
            .collect()
    };

    for tool_redacted in [false, true] {
        let _policy = PolicyGuard::set(true, tool_redacted);
        let cases = [
            (
                json!("format-secret"),
                "Unknown realtime audio format",
                None,
                "format-secret",
            ),
            (
                json!({"type": "audio/pcm", "rate": "rate-secret"}),
                "Unknown PCM rate in a realtime audio format",
                Some(json!({"type": "audio/pcm", "rate": 24000})),
                "rate-secret",
            ),
            (
                json!({"type": "format-secret", "nested": "mapping-secret"}),
                "Unknown realtime audio format mapping",
                None,
                "mapping-secret",
            ),
        ];
        for (input, expected_message, expected, secret) in cases {
            sink.0.lock().unwrap().clear();
            let format = match input.as_str() {
                Some(name) => RealtimeAudioFormat::from(name),
                None => RealtimeAudioFormat::from(input),
            };
            let value = format.to_session_value();
            assert_eq!(
                value, expected,
                "tool_redacted={tool_redacted} {expected_message}"
            );
            let logged = messages();
            assert!(
                logged.iter().any(|m| m == expected_message),
                "tool_redacted={tool_redacted} expected {expected_message:?}: {logged:?}"
            );
            assert!(
                !logged
                    .iter()
                    .any(|m| m.contains(secret) || m.contains("format-secret")),
                "tool_redacted={tool_redacted}: {logged:?}"
            );
        }

        // With model redaction off the diagnostic keeps the whole mapping.
        drop(_policy);
        let _policy = PolicyGuard::set(false, tool_redacted);
        sink.0.lock().unwrap().clear();
        let value =
            RealtimeAudioFormat::from(json!({"type": "format-secret", "nested": "mapping-secret"}))
                .to_session_value();
        assert_eq!(value, None);
        let logged = messages();
        let line = logged
            .iter()
            .find(|m| m.starts_with("Unknown realtime audio format mapping"))
            .unwrap_or_else(|| panic!("tool_redacted={tool_redacted}: {logged:?}"));
        assert!(
            line.contains("format-secret") && line.contains("mapping-secret"),
            "{line}"
        );
    }
}
