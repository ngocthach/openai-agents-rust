//! Tracing cases that need a captured log or a cleared started-trace cache: ports of
//! `test_trace_lifecycle_logging_omits_caller_data` and `test_disabled_span_logging_respects_data_policy`
//! (`tests/tracing/test_tracing_env_disable.py`) and of
//! `test_trace_logs_warning_when_current_trace_exists` and
//! `test_create_trace_for_run_does_not_reattach_after_trace_state_reload`
//! (`tests/tracing/test_trace_context.py`).
//!
//! The installed provider, the environment and the redaction flags are process-global, so every
//! test takes [`serial`] first.
#![allow(clippy::await_holding_lock)]

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::tracing::{
    DefaultTraceProvider, TraceProvider, TraceState, clear_started_trace_ids,
    create_trace_for_run_with_state, gen_trace_id, reset_trace_provider, set_trace_provider,
};
use serde_json::{Value, json};

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

fn captured() -> Vec<(log::Level, String)> {
    sink().0.lock().unwrap().clone()
}

fn serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    sink().0.lock().unwrap().clear();
    guard
}

/// Sets or removes an environment variable and restores the previous value on drop.
struct EnvGuard {
    name: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(name: &'static str, value: Option<&str>) -> Self {
        let previous = std::env::var(name).ok();
        apply_env(name, value);
        EnvGuard { name, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        apply_env(self.name, self.previous.as_deref());
    }
}

fn apply_env(name: &str, value: Option<&str>) {
    // Safety: every caller holds `serial`, so no other test thread touches the environment.
    unsafe {
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
}

/// Restores the redaction flags to the environment defaults on drop.
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

#[tokio::test]
async fn trace_lifecycle_logging_omits_caller_data() {
    let _lock = serial();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    let workflow_name = "PRIVATE_WORKFLOW_CANARY";
    let private_metadata = "PRIVATE_METADATA_CANARY";

    for (model_redacted, tool_redacted) in
        [(true, true), (true, false), (false, true), (false, false)]
    {
        for disabled_by in [None, Some("call"), Some("provider")] {
            let _policy = PolicyGuard::set(model_redacted, tool_redacted);
            sink().0.lock().unwrap().clear();
            let provider = Arc::new(DefaultTraceProvider::new());
            provider.set_disabled(disabled_by == Some("provider"));
            set_trace_provider(provider);

            let trace_id = gen_trace_id();
            let metadata = json!({"private": private_metadata})
                .as_object()
                .unwrap()
                .clone();
            let created = openai_agents_sdk::tracing::trace(workflow_name)
                .trace_id(trace_id.clone())
                .metadata(metadata.clone())
                .disabled(disabled_by == Some("call"))
                .build();
            let case = format!("{disabled_by:?} model={model_redacted} tool={tool_redacted}");

            let current_id = created
                .clone()
                .run(async {
                    openai_agents_sdk::tracing::current_trace().map(|t| t.trace_id().to_string())
                })
                .await;
            assert_eq!(current_id.as_deref(), Some(created.trace_id()), "{case}");
            assert!(
                openai_agents_sdk::tracing::current_trace().is_none(),
                "{case}"
            );
            if disabled_by.is_none() {
                assert_eq!(created.name(), workflow_name, "{case}");
                assert_eq!(created.trace_id(), trace_id, "{case}");
                let exported = created.export().expect("an enabled trace exports");
                assert_eq!(exported["workflow_name"], json!(workflow_name), "{case}");
                assert_eq!(exported["id"], json!(trace_id), "{case}");
                assert_eq!(exported["metadata"], Value::Object(metadata), "{case}");
            } else {
                assert!(created.is_disabled(), "{case}");
                assert!(created.export().is_none(), "{case}");
            }
            reset_trace_provider();

            let records = captured();
            assert!(!records.is_empty(), "{case}");
            for (_, message) in &records {
                for value in [workflow_name, trace_id.as_str(), private_metadata] {
                    assert!(!message.contains(value), "{case}: {message}");
                }
            }
        }
    }
}

#[test]
fn disabled_span_logging_respects_data_policy() {
    let _lock = serial();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", Some("1"));

    for redacted in [true, false] {
        let _policy = PolicyGuard::set(redacted, redacted);
        sink().0.lock().unwrap().clear();
        set_trace_provider(Arc::new(DefaultTraceProvider::new()));
        let span = openai_agents_sdk::tracing::agent_span("SECRET_SPAN_NAME", None, None, None);
        reset_trace_provider();

        assert!(span.is_disabled());
        let text: String = captured()
            .into_iter()
            .map(|(_, message)| message)
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(!text.contains("SECRET_SPAN_NAME"), redacted, "{text}");
        assert!(
            text.contains("Tracing is disabled. Not creating span"),
            "{text}"
        );
    }
}

#[tokio::test]
async fn trace_logs_warning_when_current_trace_exists() {
    let _lock = serial();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    reset_trace_provider();

    let outer = openai_agents_sdk::tracing::trace("outer")
        .trace_id(gen_trace_id())
        .build();
    let inner = outer
        .run(async {
            openai_agents_sdk::tracing::trace("inner")
                .trace_id(gen_trace_id())
                .build()
        })
        .await;

    assert_eq!(inner.name(), "inner");
    let warned = captured().into_iter().any(|(level, message)| {
        level == log::Level::Warn && message.contains("Trace already exists")
    });
    assert!(warned, "{:?}", captured());
}

#[test]
fn create_trace_for_run_does_not_reattach_after_trace_state_reload() {
    let _lock = serial();
    let _env = EnvGuard::set("OPENAI_AGENTS_DISABLE_TRACING", None);
    reset_trace_provider();
    let meta = json!({"key": "value"});

    let original = openai_agents_sdk::tracing::trace("workflow")
        .trace_id(gen_trace_id())
        .group_id("group-1")
        .metadata(meta.as_object().unwrap().clone())
        .build();
    original.start();
    original.finish();
    let state = TraceState::from_trace(Some(&original)).unwrap();
    assert!(openai_agents_sdk::tracing::trace_id_was_started(
        state.trace_id.as_deref()
    ));

    // A resumed run in a new process has never started this trace.
    clear_started_trace_ids();

    let created = create_trace_for_run_with_state(
        "workflow",
        state.trace_id.clone(),
        state.group_id.clone(),
        Some(meta.as_object().unwrap().clone()),
        None,
        false,
        Some(&state),
        true,
    )
    .unwrap();

    assert!(!created.is_reattached());
    assert_eq!(Some(created.trace_id().to_string()), state.trace_id);
}
