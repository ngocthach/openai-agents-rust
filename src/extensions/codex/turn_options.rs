//! Per-turn options. Mirrors `agents.extensions.experimental.codex.turn_options`.

use std::time::Duration;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// Options applied to a single Codex turn.
///
/// `PartialEq` is not implemented because [`cancel`](TurnOptions::cancel) holds a cancellation
/// handle.
#[derive(Debug, Clone, Default)]
pub struct TurnOptions {
    /// JSON schema written to a temp file and passed as `--output-schema`.
    pub output_schema: Option<Value>,
    /// Cancels the turn: the Codex CLI is killed and the turn ends with an error. Port of
    /// `TurnOptions.signal`.
    pub cancel: Option<CancellationToken>,
    /// Abort the turn when no event arrives within this duration.
    pub idle_timeout: Option<Duration>,
}

impl TurnOptions {
    /// Default turn options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request structured output against this JSON schema.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Abort the turn when the Codex stream is idle for this long.
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = Some(timeout);
        self
    }

    /// Cancel the turn when `token` fires. Dropping the event stream also kills the Codex CLI;
    /// this gives callers an explicit handle, as Python's `TurnOptions.signal` does.
    pub fn with_cancel(mut self, token: CancellationToken) -> Self {
        self.cancel = Some(token);
        self
    }
}
