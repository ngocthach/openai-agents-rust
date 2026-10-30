//! Logging helpers that honor the model/tool data redaction flags. Mirrors `agents.logger`.
//!
//! The SDK logs *what* failed unconditionally, but the error text itself is only logged when
//! [`crate::debug`] allows the corresponding data class. Use these helpers instead of
//! `log::error!("… {err}")` anywhere an error may carry model transcripts or tool arguments.
//!
//! ```
//! use openai_agents_sdk::logger;
//!
//! let err = std::io::Error::other("secret payload");
//! // Logs `Error getting response` only, unless OPENAI_AGENTS_DONT_LOG_MODEL_DATA=0.
//! logger::log_model_action_error("Error getting response", &err);
//! ```

use std::fmt::Display;

use log::Level;

use crate::debug::{dont_log_model_data, dont_log_tool_data};

/// The `log` target used by every SDK message, matching the Python logger name.
pub const LOG_TARGET: &str = "openai.agents";

/// Placeholder substituted for a payload that the logging policy keeps out of messages.
pub const REDACTED_DATA_PLACEHOLDER: &str = "<redacted>";

fn log_action(level: Level, message: &str, error: &dyn Display, redact: bool) {
    if redact {
        log::log!(target: LOG_TARGET, level, "{message}");
    } else {
        log::log!(target: LOG_TARGET, level, "{message}: {error}");
    }
}

/// Whether a payload containing model data (and optionally tool data) may be logged.
///
/// Mirrors the `should_redact` rule in `agents.util._json.validate_json`.
pub fn may_log_model_data(contains_tool_data: bool) -> bool {
    !(dont_log_model_data() || (contains_tool_data && dont_log_tool_data()))
}

/// Return `payload`, or [`REDACTED_DATA_PLACEHOLDER`] when the logging policy forbids it.
pub fn model_data_payload(payload: &str, contains_tool_data: bool) -> &str {
    if may_log_model_data(contains_tool_data) {
        payload
    } else {
        REDACTED_DATA_PLACEHOLDER
    }
}

/// Error-log a model-data failure according to the model logging policy.
pub fn log_model_action_error(message: &str, error: &dyn Display) {
    log_action(Level::Error, message, error, dont_log_model_data());
}

/// Warning-log a model-data failure according to the model logging policy.
pub fn log_model_action_warning(message: &str, error: &dyn Display) {
    log_action(Level::Warn, message, error, dont_log_model_data());
}

/// Debug-log a model-data failure according to the model logging policy.
pub fn log_model_action_debug(message: &str, error: &dyn Display) {
    log_action(Level::Debug, message, error, dont_log_model_data());
}

/// Error-log a tool-data failure according to the tool logging policy.
pub fn log_tool_action_error(message: &str, error: &dyn Display) {
    log_action(Level::Error, message, error, dont_log_tool_data());
}

/// Warning-log a tool-data failure according to the tool logging policy.
pub fn log_tool_action_warning(message: &str, error: &dyn Display) {
    log_action(Level::Warn, message, error, dont_log_tool_data());
}

/// Debug-log a tool-data failure according to the tool logging policy.
pub fn log_tool_action_debug(message: &str, error: &dyn Display) {
    log_action(Level::Debug, message, error, dont_log_tool_data());
}

fn redact_mixed() -> bool {
    dont_log_model_data() || dont_log_tool_data()
}

/// Error-log a mixed model/tool failure; the text appears only when both policies allow it.
pub fn log_model_and_tool_action_error(message: &str, error: &dyn Display) {
    log_action(Level::Error, message, error, redact_mixed());
}

/// Warning-log a mixed model/tool failure; the text appears only when both policies allow it.
pub fn log_model_and_tool_action_warning(message: &str, error: &dyn Display) {
    log_action(Level::Warn, message, error, redact_mixed());
}

/// Debug-log a mixed model/tool failure; the text appears only when both policies allow it.
pub fn log_model_and_tool_action_debug(message: &str, error: &dyn Display) {
    log_action(Level::Debug, message, error, redact_mixed());
}

/// Warning-log mixed model/tool *data*: `redacted_message`, or `diagnostic()` when both
/// policies allow the detail. Mirrors `log_model_and_tool_data_warning`.
pub fn log_model_and_tool_data_warning(
    redacted_message: &str,
    diagnostic: impl FnOnce() -> String,
) {
    if redact_mixed() {
        log::warn!(target: LOG_TARGET, "{redacted_message}");
    } else {
        log::warn!(target: LOG_TARGET, "{}", diagnostic());
    }
}
