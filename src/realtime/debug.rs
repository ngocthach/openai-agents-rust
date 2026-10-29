//! Redaction of model and tool data in realtime logs. Mirrors `agents._debug`.
//!
//! Realtime sessions carry transcripts, audio and tool arguments, so by default the SDK logs only
//! what happened, never the data it happened to. Opt in to the detail with the
//! `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` and `OPENAI_AGENTS_DONT_LOG_TOOL_DATA` environment
//! variables, or with [`set_dont_log_model_data`] and [`set_dont_log_tool_data`], which take
//! precedence over the environment.
//!
//! Event payloads are never redacted: [`crate::realtime::RealtimeSessionEvent`] is the session's
//! API, not its log.

use std::fmt;
use std::sync::RwLock;

static DONT_LOG_MODEL_DATA: RwLock<Option<bool>> = RwLock::new(None);
static DONT_LOG_TOOL_DATA: RwLock<Option<bool>> = RwLock::new(None);

/// Read a `1`/`true` style flag from the environment.
fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Err(_) => default,
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true"
        }
    }
}

/// Whether model data (transcripts, audio formats, server payloads) is kept out of logs.
///
/// Defaults to `true`, as in the Python SDK.
pub fn dont_log_model_data() -> bool {
    if let Some(value) = *DONT_LOG_MODEL_DATA
        .read()
        .expect("realtime debug flag poisoned")
    {
        return value;
    }
    env_flag("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", true)
}

/// Whether tool data (arguments, outputs) is kept out of logs.
///
/// Defaults to `true`, as in the Python SDK.
pub fn dont_log_tool_data() -> bool {
    if let Some(value) = *DONT_LOG_TOOL_DATA
        .read()
        .expect("realtime debug flag poisoned")
    {
        return value;
    }
    env_flag("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", true)
}

/// Override the model-data redaction flag, ignoring the environment.
pub fn set_dont_log_model_data(value: bool) {
    *DONT_LOG_MODEL_DATA
        .write()
        .expect("realtime debug flag poisoned") = Some(value);
}

/// Override the tool-data redaction flag, ignoring the environment.
pub fn set_dont_log_tool_data(value: bool) {
    *DONT_LOG_TOOL_DATA
        .write()
        .expect("realtime debug flag poisoned") = Some(value);
}

/// Go back to reading the model-data redaction flag from the environment.
pub fn reset_dont_log_model_data() {
    *DONT_LOG_MODEL_DATA
        .write()
        .expect("realtime debug flag poisoned") = None;
}

/// Go back to reading the tool-data redaction flag from the environment.
pub fn reset_dont_log_tool_data() {
    *DONT_LOG_TOOL_DATA
        .write()
        .expect("realtime debug flag poisoned") = None;
}

/// Log a failure involving model data, with the detail only when redaction is off.
pub(crate) fn log_model_action_error(message: &str, detail: &dyn fmt::Display) {
    if dont_log_model_data() {
        log::error!("{message}");
    } else {
        log::error!("{message}: {detail}");
    }
}

/// Warn about a failure involving model data, with the detail only when redaction is off.
pub(crate) fn log_model_action_warning(message: &str, detail: &dyn fmt::Display) {
    if dont_log_model_data() {
        log::warn!("{message}");
    } else {
        log::warn!("{message}: {detail}");
    }
}

/// Log model data at debug level, naming it only when redaction is off.
pub(crate) fn log_model_data_debug(message: &str, detail: &dyn fmt::Display) {
    if dont_log_model_data() {
        log::debug!("{message}");
    } else {
        log::debug!("{message}: {detail}");
    }
}

/// Log a failure involving tool data, with the detail only when redaction is off.
pub(crate) fn log_tool_action_error(message: &str, detail: &dyn fmt::Display) {
    if dont_log_tool_data() {
        log::error!("{message}");
    } else {
        log::error!("{message}: {detail}");
    }
}

/// Warn about a failure involving tool data, with the detail only when redaction is off.
pub(crate) fn log_tool_action_warning(message: &str, detail: &dyn fmt::Display) {
    if dont_log_tool_data() {
        log::warn!("{message}");
    } else {
        log::warn!("{message}: {detail}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_take_precedence_over_the_environment() {
        // These globals are shared, so the test restores them before returning.
        set_dont_log_model_data(false);
        assert!(!dont_log_model_data());
        set_dont_log_model_data(true);
        assert!(dont_log_model_data());
        reset_dont_log_model_data();

        set_dont_log_tool_data(false);
        assert!(!dont_log_tool_data());
        reset_dont_log_tool_data();

        // After a reset the flags come from the environment again, which is the Python default of
        // "on" unless the variables say otherwise.
        assert_eq!(
            dont_log_model_data(),
            env_flag("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", true)
        );
        assert_eq!(
            dont_log_tool_data(),
            env_flag("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", true)
        );
    }

    #[test]
    fn env_flag_accepts_one_and_true() {
        // `env_flag` only reads, so an absent variable exercises the default.
        assert!(env_flag(
            "OPENAI_AGENTS_REALTIME_FLAG_THAT_IS_NOT_SET",
            true
        ));
        assert!(!env_flag(
            "OPENAI_AGENTS_REALTIME_FLAG_THAT_IS_NOT_SET",
            false
        ));
    }
}
