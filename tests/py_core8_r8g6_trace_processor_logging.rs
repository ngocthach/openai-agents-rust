//! Port of `test_error_logging_redaction.py::test_trace_processor_failure_identity_follows_both_data_policies`.
//!
//! A trace processor that fails must not stop the others, and the failure log follows the data
//! policies: with either the model or the tool flag set only the fixed message is logged (no
//! processor identity, no failure text); with both off the identity and the failure text appear.
//! The log backend, the logging flags and the trace provider are process-global, so one test
//! walks every operation and policy in sequence.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::tracing::{
    Span, Trace, TracingProcessor, custom_span, flush_traces, set_trace_processors,
    shutdown_tracing, trace,
};
use serde_json::json;

const SECRET: &str = "SECRET_PROCESSOR_FAILURE_123";

static RECORDS: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS
            .lock()
            .unwrap()
            .push((record.level(), format!("{}", record.args())));
    }
    fn flush(&self) {}
}

fn install() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

#[derive(Debug)]
struct FailingProcessor;

impl TracingProcessor for FailingProcessor {
    fn on_trace_start(&self, _: &Trace) {
        panic!("{SECRET}");
    }
    fn on_trace_end(&self, _: &Trace) {
        panic!("{SECRET}");
    }
    fn on_span_start(&self, _: &Span) {
        panic!("{SECRET}");
    }
    fn on_span_end(&self, _: &Span) {
        panic!("{SECRET}");
    }
    fn shutdown(&self) {
        panic!("{SECRET}");
    }
    fn force_flush(&self) {
        panic!("{SECRET}");
    }
}

#[derive(Debug, Default)]
struct Healthy(Mutex<Vec<&'static str>>);

impl TracingProcessor for Healthy {
    fn on_trace_start(&self, _: &Trace) {
        self.0.lock().unwrap().push("on_trace_start");
    }
    fn on_trace_end(&self, _: &Trace) {
        self.0.lock().unwrap().push("on_trace_end");
    }
    fn on_span_start(&self, _: &Span) {
        self.0.lock().unwrap().push("on_span_start");
    }
    fn on_span_end(&self, _: &Span) {
        self.0.lock().unwrap().push("on_span_end");
    }
    fn force_flush(&self) {
        self.0.lock().unwrap().push("force_flush");
    }
    fn shutdown(&self) {
        self.0.lock().unwrap().push("shutdown");
    }
}

/// The base message Python logs for each operation.
const OPERATIONS: [(&str, &str); 6] = [
    (
        "on_trace_start",
        "Error in trace processor during on_trace_start",
    ),
    (
        "on_trace_end",
        "Error in trace processor during on_trace_end",
    ),
    (
        "on_span_start",
        "Error in trace processor during on_span_start",
    ),
    ("on_span_end", "Error in trace processor during on_span_end"),
    ("force_flush", "Error flushing trace processor"),
    ("shutdown", "Error shutting down trace processor"),
];

#[tokio::test(flavor = "current_thread")]
async fn trace_processor_failure_identity_follows_both_data_policies() {
    install();
    // `catch_unwind` still runs the default hook, which would print a backtrace per panic.
    std::panic::set_hook(Box::new(|_| {}));
    for (model_redacted, tool_redacted) in [(true, false), (false, true), (false, false)] {
        set_dont_log_model_data(model_redacted);
        set_dont_log_tool_data(tool_redacted);
        let healthy = Arc::new(Healthy::default());
        set_trace_processors(vec![Arc::new(FailingProcessor), healthy.clone()]);
        RECORDS.lock().unwrap().clear();

        trace("failing processor")
            .run(async {
                custom_span("work", json!({})).run(async {}).await;
            })
            .await;
        flush_traces();
        shutdown_tracing();

        // A failing processor never stops the one after it.
        assert_eq!(
            *healthy.0.lock().unwrap(),
            [
                "on_trace_start",
                "on_span_start",
                "on_span_end",
                "on_trace_end",
                "force_flush",
                "shutdown"
            ],
            "model_redacted={model_redacted} tool_redacted={tool_redacted}"
        );

        let errors: Vec<String> = RECORDS
            .lock()
            .unwrap()
            .iter()
            .filter(|(level, _)| *level == log::Level::Error)
            .map(|(_, text)| text.clone())
            .collect();
        let redacted = model_redacted || tool_redacted;
        for (operation, base) in OPERATIONS {
            let line = errors
                .iter()
                .find(|text| text.starts_with(base))
                .unwrap_or_else(|| panic!("no error logged for {operation}: {errors:?}"));
            if redacted {
                assert_eq!(line, base, "{operation}");
            } else {
                assert!(line.contains("FailingProcessor"), "{operation}: {line}");
                assert!(line.contains(SECRET), "{operation}: {line}");
            }
        }
        for line in &errors {
            if redacted {
                assert!(!line.contains(SECRET) && !line.contains("FailingProcessor"));
            }
        }
    }
    reset_dont_log_model_data();
    reset_dont_log_tool_data();
    let _ = std::panic::take_hook();
}
