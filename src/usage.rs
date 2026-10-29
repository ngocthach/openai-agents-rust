//! Token usage tracking. Mirrors `agents.usage`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Details about the input tokens of a request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Tokens served from the prompt cache.
    #[serde(default)]
    pub cached_tokens: u64,
    /// Tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
}

/// Details about the output tokens of a request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Tokens spent on reasoning.
    #[serde(default)]
    pub reasoning_tokens: u64,
}

/// Usage of a single model request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
    #[serde(default)]
    pub output_tokens_details: OutputTokensDetails,
}

/// Aggregated usage across a run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of model requests.
    #[serde(default)]
    pub requests: u64,
    /// Total input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
    /// Total output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub output_tokens_details: OutputTokensDetails,
    /// Total tokens (input + output).
    #[serde(default)]
    pub total_tokens: u64,
    /// Per-request breakdown.
    #[serde(default)]
    pub request_usage_entries: Vec<RequestUsage>,
}

impl Usage {
    /// Usage for a single request with the given token counts.
    pub fn single(input_tokens: u64, output_tokens: u64) -> Self {
        Usage {
            requests: 1,
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            ..Default::default()
        }
    }

    /// Add another usage into this one, preserving per-request entries.
    pub fn add(&mut self, other: &Usage) {
        self.requests += other.requests;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.input_tokens_details.cached_tokens += other.input_tokens_details.cached_tokens;
        self.input_tokens_details.cache_write_tokens +=
            other.input_tokens_details.cache_write_tokens;
        self.output_tokens_details.reasoning_tokens +=
            other.output_tokens_details.reasoning_tokens;

        if !other.request_usage_entries.is_empty() {
            self.request_usage_entries
                .extend(other.request_usage_entries.iter().cloned());
        } else if other.requests == 1
            && (other.input_tokens > 0 || other.output_tokens > 0 || other.total_tokens > 0)
        {
            self.request_usage_entries.push(RequestUsage {
                input_tokens: other.input_tokens,
                output_tokens: other.output_tokens,
                total_tokens: other.total_tokens,
                input_tokens_details: other.input_tokens_details.clone(),
                output_tokens_details: other.output_tokens_details.clone(),
            });
        }
    }

    /// Build a single-request usage from a Responses API `usage` object.
    pub fn from_responses_usage(value: &Value) -> Usage {
        let get = |k: &str| value.get(k).and_then(Value::as_u64).unwrap_or(0);
        let input_tokens = get("input_tokens");
        let output_tokens = get("output_tokens");
        let total_tokens = {
            let t = get("total_tokens");
            if t == 0 { input_tokens + output_tokens } else { t }
        };
        let cached = value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let cache_write = value
            .pointer("/input_tokens_details/cache_write_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let reasoning = value
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Usage {
            requests: 1,
            input_tokens,
            output_tokens,
            total_tokens,
            input_tokens_details: InputTokensDetails {
                cached_tokens: cached,
                cache_write_tokens: cache_write,
            },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: reasoning,
            },
            request_usage_entries: Vec::new(),
        }
    }

    /// Build a single-request usage from a Chat Completions `usage` object.
    pub fn from_chat_usage(value: &Value) -> Usage {
        let get = |k: &str| value.get(k).and_then(Value::as_u64).unwrap_or(0);
        let input_tokens = get("prompt_tokens");
        let output_tokens = get("completion_tokens");
        let total_tokens = {
            let t = get("total_tokens");
            if t == 0 { input_tokens + output_tokens } else { t }
        };
        let cached = value
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let reasoning = value
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Usage {
            requests: 1,
            input_tokens,
            output_tokens,
            total_tokens,
            input_tokens_details: InputTokensDetails {
                cached_tokens: cached,
                cache_write_tokens: 0,
            },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: reasoning,
            },
            request_usage_entries: Vec::new(),
        }
    }

    /// Usage in the shape used by generation spans.
    pub fn to_span_usage(&self) -> Value {
        serde_json::json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.input_tokens_details.cached_tokens,
            "cache_write_input_tokens": self.input_tokens_details.cache_write_tokens,
            "reasoning_output_tokens": self.output_tokens_details.reasoning_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_preserves_request_entries() {
        let mut total = Usage::default();
        total.add(&Usage::single(100, 10));
        total.add(&Usage::single(50, 5));
        assert_eq!(total.requests, 2);
        assert_eq!(total.input_tokens, 150);
        assert_eq!(total.total_tokens, 165);
        assert_eq!(total.request_usage_entries.len(), 2);
        assert_eq!(total.request_usage_entries[1].input_tokens, 50);
    }

    #[test]
    fn parses_responses_usage() {
        let v = serde_json::json!({
            "input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 2}
        });
        let u = Usage::from_responses_usage(&v);
        assert_eq!(u.requests, 1);
        assert_eq!(u.input_tokens_details.cached_tokens, 4);
        assert_eq!(u.output_tokens_details.reasoning_tokens, 2);
    }
}
