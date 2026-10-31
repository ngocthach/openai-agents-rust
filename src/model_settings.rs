//! Model settings. Mirrors `agents.model_settings`.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Tool choice for a hosted MCP server tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MCPToolChoice {
    pub server_label: String,
    pub name: String,
}

/// How the model should pick tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// A literal: `auto`, `required`, `none`, or the name of a specific tool / hosted tool type.
    Named(String),
    /// A hosted MCP tool.
    Mcp(MCPToolChoice),
}

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Named("auto".into())
    }
    pub fn required() -> Self {
        ToolChoice::Named("required".into())
    }
    pub fn none() -> Self {
        ToolChoice::Named("none".into())
    }
    /// Force a specific tool by name.
    pub fn tool(name: impl Into<String>) -> Self {
        ToolChoice::Named(name.into())
    }
    /// The string form, if this is a named choice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ToolChoice::Named(s) => Some(s.as_str()),
            ToolChoice::Mcp(_) => None,
        }
    }
}

impl From<&str> for ToolChoice {
    fn from(value: &str) -> Self {
        ToolChoice::Named(value.to_string())
    }
}

impl From<String> for ToolChoice {
    fn from(value: String) -> Self {
        ToolChoice::Named(value)
    }
}

/// Reasoning configuration for reasoning models.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reasoning {
    /// `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// `auto`, `concise`, `detailed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Reasoning mode (Responses API only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Reasoning context carry-over (Responses API only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Provider options this SDK version does not model yet; they are forwarded unchanged.
    #[serde(flatten, default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

impl Reasoning {
    /// Reasoning with only the effort set.
    pub fn effort(effort: impl Into<String>) -> Self {
        Reasoning {
            effort: Some(effort.into()),
            ..Default::default()
        }
    }
}

/// Backoff configuration for runner-managed model retries. A partial JSON object keeps the
/// defaults of the omitted fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelRetryBackoffSettings {
    /// Delay before the first retry, in seconds.
    pub initial_delay: f64,
    /// Upper bound of the computed backoff delay, in seconds.
    pub max_delay: f64,
    /// Multiplier applied to the delay after every retry.
    pub multiplier: f64,
    /// Whether to randomize the delay.
    pub jitter: bool,
}

impl Default for ModelRetryBackoffSettings {
    fn default() -> Self {
        ModelRetryBackoffSettings {
            initial_delay: 0.25,
            max_delay: 2.0,
            multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Runner-managed retry settings for model calls (opt-in).
///
/// With a [`policy`](Self::policy) (see [`crate::retry::retry_policies`]) the policy decides,
/// with the model's retry advice and the runner's replay-safety rules applied. Without one,
/// network errors, timeouts and the HTTP statuses in `retry_statuses` are retried (stateful
/// requests only on 429). Streamed requests are only retried before any output event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRetrySettings {
    /// Number of retries after the initial attempt.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Backoff strategy.
    #[serde(default)]
    pub backoff: ModelRetryBackoffSettings,
    /// HTTP status codes that trigger a retry.
    #[serde(default = "default_retry_statuses")]
    pub retry_statuses: Vec<u16>,
    /// Runtime-only retry policy (not serialized).
    #[serde(skip)]
    pub policy: Option<crate::retry::RetryPolicy>,
}

fn default_max_retries() -> u32 {
    ModelRetrySettings::default().max_retries
}

fn default_retry_statuses() -> Vec<u16> {
    vec![408, 409, 429, 500, 502, 503, 504]
}

impl Default for ModelRetrySettings {
    fn default() -> Self {
        ModelRetrySettings {
            max_retries: 2,
            backoff: ModelRetryBackoffSettings::default(),
            retry_statuses: default_retry_statuses(),
            policy: None,
        }
    }
}

impl ModelRetrySettings {
    /// Use a retry policy.
    pub fn with_policy(mut self, policy: crate::retry::RetryPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Set the retry budget.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Delay for the given retry attempt (1-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let b = &self.backoff;
        let base = b.initial_delay * b.multiplier.powi(attempt.saturating_sub(1) as i32);
        let mut secs = base.min(b.max_delay).max(0.0);
        if b.jitter {
            let factor: f64 = rand::random::<f64>() * 0.5 + 0.75;
            secs *= factor;
        }
        Duration::from_secs_f64(secs)
    }
}

/// Settings to use when calling an LLM. All fields are optional; `None` defers to the provider.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// `auto` or `disabled`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// `low`, `medium`, `high`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// `in_memory` or `24h`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    /// Chat Completions only: request a usage chunk when streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    /// Additional output data to include (Responses `include`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_query: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_headers: Option<HashMap<String, String>>,
    /// Arbitrary extra request fields merged into the request body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<ModelRetrySettings>,
    /// Responses `context_management` entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<Value>>,
    /// Responses `prompt_cache_options`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_options: Option<Value>,
    /// Preserve the raw provider usage payload on `ModelResponse::raw_usage`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preserve_raw_usage: Option<bool>,
    /// Maximum duration of each model-call attempt, in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

macro_rules! overlay {
    ($self:ident, $other:ident, $($field:ident),* $(,)?) => {
        $( if $other.$field.is_some() { $self.$field = $other.$field.clone(); } )*
    };
}

impl ModelSettings {
    /// Produce a new `ModelSettings` by overlaying any `Some` values from `override_` on top of
    /// this instance. `extra_args` maps are merged.
    pub fn resolve(&self, override_: Option<&ModelSettings>) -> ModelSettings {
        let Some(other) = override_ else {
            return self.clone();
        };
        let mut out = self.clone();
        overlay!(
            out,
            other,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            tool_choice,
            parallel_tool_calls,
            truncation,
            max_tokens,
            reasoning,
            verbosity,
            metadata,
            store,
            prompt_cache_retention,
            include_usage,
            response_include,
            top_logprobs,
            extra_query,
            extra_body,
            extra_headers,
            retry,
            context_management,
            prompt_cache_options,
            preserve_raw_usage,
            timeout,
        );
        if self.extra_args.is_some() || other.extra_args.is_some() {
            let mut merged = Map::new();
            if let Some(a) = &self.extra_args {
                merged.extend(a.clone());
            }
            if let Some(b) = &other.extra_args {
                merged.extend(b.clone());
            }
            out.extra_args = if merged.is_empty() {
                None
            } else {
                Some(merged)
            };
        }
        out
    }

    /// Build settings from a JSON object, the Rust form of Python's dictionary
    /// `model_settings`. Unknown keys in the settings and in the SDK-owned nested settings
    /// (`tool_choice`, `retry`, `retry.backoff`, `context_management[i]`,
    /// `prompt_cache_options`) are rejected so typos fail early; omitted fields keep their
    /// defaults and unmodelled `reasoning` options are preserved.
    pub fn from_value(value: Value) -> crate::error::Result<ModelSettings> {
        use crate::error::AgentsError;
        let Value::Object(map) = &value else {
            return Err(AgentsError::user(format!(
                "ModelSettings must be a ModelSettings instance or a dict, got {}",
                json_type_name(&value)
            )));
        };
        reject_unknown(map, MODEL_SETTINGS_FIELDS, "Unknown model settings")?;
        if let Some(Value::Object(choice)) = map.get("tool_choice") {
            reject_unknown(
                choice,
                &["server_label", "name"],
                "Unknown model settings in tool_choice",
            )?;
        }
        if let Some(Value::Object(retry)) = map.get("retry") {
            reject_unknown(
                retry,
                &["max_retries", "backoff", "retry_statuses"],
                "Unknown model settings in retry",
            )?;
            if let Some(Value::Object(backoff)) = retry.get("backoff") {
                reject_unknown(
                    backoff,
                    &["initial_delay", "max_delay", "multiplier", "jitter"],
                    "Unknown model settings in retry.backoff",
                )?;
            }
        }
        if let Some(Value::Array(entries)) = map.get("context_management") {
            for (index, entry) in entries.iter().enumerate() {
                if let Value::Object(entry) = entry {
                    reject_unknown(
                        entry,
                        &["type", "compact_threshold"],
                        &format!("Unknown model settings in context_management[{index}]"),
                    )?;
                }
            }
        }
        if let Some(Value::Object(options)) = map.get("prompt_cache_options") {
            reject_unknown(
                options,
                &["comparison_response_id", "mode", "prewarm", "ttl"],
                "Unknown model settings in prompt_cache_options",
            )?;
        }
        serde_json::from_value(value)
            .map_err(|e| AgentsError::user(format!("Invalid model settings: {e}")))
    }

    /// Serialize the subset of settings that is safe to record in traces.
    pub fn to_traceable_value(&self) -> Value {
        let mut m = Map::new();
        let v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(obj) = v {
            for key in [
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "tool_choice",
                "parallel_tool_calls",
                "truncation",
                "max_tokens",
                "reasoning",
                "verbosity",
                "metadata",
                "store",
                "prompt_cache_retention",
                "include_usage",
                "response_include",
                "top_logprobs",
                "retry",
                "context_management",
                "prompt_cache_options",
                "timeout",
            ] {
                if let Some(val) = obj.get(key) {
                    m.insert(key.to_string(), val.clone());
                }
            }
        }
        Value::Object(m)
    }

    /// Builder-style setter for temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
    /// Builder-style setter for tool choice.
    pub fn with_tool_choice(mut self, tool_choice: impl Into<ToolChoice>) -> Self {
        self.tool_choice = Some(tool_choice.into());
        self
    }
    /// Builder-style setter for max output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
    /// Builder-style setter for reasoning.
    pub fn with_reasoning(mut self, reasoning: Reasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }
    /// Builder-style setter for parallel tool calls.
    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }
    /// Builder-style setter for retry settings.
    pub fn with_retry(mut self, retry: ModelRetrySettings) -> Self {
        self.retry = Some(retry);
        self
    }
    /// Builder-style setter for the per-attempt timeout in seconds.
    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout = Some(seconds);
        self
    }
    /// Builder-style setter for `store`.
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }
}

/// Every [`ModelSettings`] field, for [`ModelSettings::from_value`] validation.
const MODEL_SETTINGS_FIELDS: &[&str] = &[
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "tool_choice",
    "parallel_tool_calls",
    "truncation",
    "max_tokens",
    "reasoning",
    "verbosity",
    "metadata",
    "store",
    "prompt_cache_retention",
    "include_usage",
    "response_include",
    "top_logprobs",
    "extra_query",
    "extra_body",
    "extra_headers",
    "extra_args",
    "retry",
    "context_management",
    "prompt_cache_options",
    "preserve_raw_usage",
    "timeout",
];

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn reject_unknown(
    map: &Map<String, Value>,
    known: &[&str],
    message: &str,
) -> crate::error::Result<()> {
    let mut unknown: Vec<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|key| !known.contains(key))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    Err(crate::error::AgentsError::user(format!(
        "{message}: {}",
        unknown.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_overlays_and_merges_extra_args() {
        let base = ModelSettings {
            temperature: Some(0.1),
            top_p: Some(0.9),
            extra_args: Some(serde_json::json!({"a": 1}).as_object().unwrap().clone()),
            ..Default::default()
        };
        let over = ModelSettings {
            temperature: Some(0.7),
            extra_args: Some(serde_json::json!({"b": 2}).as_object().unwrap().clone()),
            ..Default::default()
        };
        let r = base.resolve(Some(&over));
        assert_eq!(r.temperature, Some(0.7));
        assert_eq!(r.top_p, Some(0.9));
        let args = r.extra_args.unwrap();
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn tool_choice_serializes_as_string_or_object() {
        let s = serde_json::to_value(ToolChoice::required()).unwrap();
        assert_eq!(s, Value::String("required".into()));
        let m = serde_json::to_value(ToolChoice::Mcp(MCPToolChoice {
            server_label: "x".into(),
            name: "y".into(),
        }))
        .unwrap();
        assert_eq!(m["server_label"], "x");
    }
}
