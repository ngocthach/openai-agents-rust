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
//!
//! An application that has not wired up a `log` backend sees nothing at all. Port of Python's
//! `enable_verbose_stdout_logging()`: [`enable_verbose_stdout_logging`] installs a minimal stdout
//! backend for the SDK's own messages, so debugging needs no extra dependency.
//!
//! ```
//! openai_agents_sdk::enable_verbose_stdout_logging();
//! ```

use std::fmt::Display;
use std::io::Write;
use std::sync::OnceLock;

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

/// Whether `target` is the SDK's own log target, or one of its dotted children.
fn is_sdk_target(target: &str) -> bool {
    target
        .strip_prefix(LOG_TARGET)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
}

/// The stdout backend installed by [`enable_verbose_stdout_logging`].
///
/// It only accepts records on [`LOG_TARGET`], so an application that enables it does not also get
/// the debug output of every other crate in the process.
#[derive(Debug)]
struct VerboseStdoutLogger;

static VERBOSE_STDOUT_LOGGER: VerboseStdoutLogger = VerboseStdoutLogger;

impl log::Log for VerboseStdoutLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= Level::Debug && is_sdk_target(metadata.target())
    }

    fn log(&self, record: &log::Record<'_>) {
        // The `log::debug!` family calls `log` without consulting `enabled`, so filter here too.
        if !self.enabled(record.metadata()) {
            return;
        }
        // A logger must never panic, so a closed stdout is ignored rather than unwrapped.
        let mut out = std::io::stdout().lock();
        let _ = writeln!(
            out,
            "{} {}: {}",
            record.level(),
            record.target(),
            record.args()
        );
    }

    fn flush(&self) {
        let _ = std::io::stdout().flush();
    }
}

static VERBOSE_STDOUT_INSTALLED: OnceLock<bool> = OnceLock::new();

/// Enable verbose SDK logging to stdout. Port of Python's `enable_verbose_stdout_logging()`.
///
/// Installs a minimal `log` backend that prints every SDK record of level `Debug` or higher to
/// stdout as `LEVEL target: message`, and raises the global level filter to `Debug`. Records on
/// other targets are dropped, so no other crate becomes verbose.
///
/// Returns `true` when the SDK backend is in place (this call installed it, or an earlier one
/// did), and `false` when another `log` backend already owns the process: the `log` crate allows
/// a single global logger, so nothing is replaced and nothing panics. Raise the verbosity of the
/// `openai.agents` target in that backend instead (for example `RUST_LOG=openai.agents=debug`).
///
/// Verbose logging does not defeat redaction: model and tool payloads still appear as
/// [`REDACTED_DATA_PLACEHOLDER`] unless [`crate::debug::set_dont_log_model_data`] /
/// [`crate::debug::set_dont_log_tool_data`] (or `OPENAI_AGENTS_DONT_LOG_MODEL_DATA=0` /
/// `OPENAI_AGENTS_DONT_LOG_TOOL_DATA=0`) allow them.
///
/// ```
/// use openai_agents_sdk::enable_verbose_stdout_logging;
///
/// // Idempotent: a second call is a no-op that reports the same state.
/// assert!(enable_verbose_stdout_logging());
/// assert!(enable_verbose_stdout_logging());
/// ```
pub fn enable_verbose_stdout_logging() -> bool {
    *VERBOSE_STDOUT_INSTALLED.get_or_init(|| {
        if log::set_logger(&VERBOSE_STDOUT_LOGGER).is_err() {
            return false;
        }
        if log::max_level() < log::LevelFilter::Debug {
            log::set_max_level(log::LevelFilter::Debug);
        }
        true
    })
}
