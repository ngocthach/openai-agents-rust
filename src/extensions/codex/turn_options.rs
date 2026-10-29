//! Per-turn options. Mirrors `agents.extensions.experimental.codex.turn_options`.

use std::time::Duration;

use serde_json::Value;

/// Options applied to a single Codex turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnOptions {
    /// JSON schema written to a temp file and passed as `--output-schema`.
    pub output_schema: Option<Value>,
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
}
