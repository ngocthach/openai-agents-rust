//! Redaction of model and tool data in SDK logs. Mirrors `agents._debug`.
//!
//! By default the SDK logs what happened, never the model or tool data involved. Opt in to the
//! detail with the `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` and `OPENAI_AGENTS_DONT_LOG_TOOL_DATA`
//! environment variables (`0`/`false`), or with [`set_dont_log_model_data`] and
//! [`set_dont_log_tool_data`], which take precedence over the environment.

use std::sync::RwLock;

static DONT_LOG_MODEL_DATA: RwLock<Option<bool>> = RwLock::new(None);
static DONT_LOG_TOOL_DATA: RwLock<Option<bool>> = RwLock::new(None);

/// Read a `1`/`true` style flag from the environment.
pub(crate) fn env_flag(name: &str, default: bool) -> bool {
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
    if let Some(value) = *DONT_LOG_MODEL_DATA.read().expect("debug flag poisoned") {
        return value;
    }
    env_flag("OPENAI_AGENTS_DONT_LOG_MODEL_DATA", true)
}

/// Whether tool data (arguments, outputs) is kept out of logs.
///
/// Defaults to `true`, as in the Python SDK.
pub fn dont_log_tool_data() -> bool {
    if let Some(value) = *DONT_LOG_TOOL_DATA.read().expect("debug flag poisoned") {
        return value;
    }
    env_flag("OPENAI_AGENTS_DONT_LOG_TOOL_DATA", true)
}

/// Override the model-data redaction flag, ignoring the environment.
pub fn set_dont_log_model_data(value: bool) {
    *DONT_LOG_MODEL_DATA.write().expect("debug flag poisoned") = Some(value);
}

/// Override the tool-data redaction flag, ignoring the environment.
pub fn set_dont_log_tool_data(value: bool) {
    *DONT_LOG_TOOL_DATA.write().expect("debug flag poisoned") = Some(value);
}

/// Go back to reading the model-data redaction flag from the environment.
pub fn reset_dont_log_model_data() {
    *DONT_LOG_MODEL_DATA.write().expect("debug flag poisoned") = None;
}

/// Go back to reading the tool-data redaction flag from the environment.
pub fn reset_dont_log_tool_data() {
    *DONT_LOG_TOOL_DATA.write().expect("debug flag poisoned") = None;
}
