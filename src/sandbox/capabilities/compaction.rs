//! The compaction capability. Mirrors `agents.sandbox.capabilities.compaction`.
//!
//! Compaction asks the Responses API to compact the conversation once it crosses a token
//! threshold, and drops everything before the newest `compaction` item from the model input.

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::items::InputItem;
use crate::sandbox::capabilities::Capability;

/// Threshold used when the model's context window is unknown.
pub const DEFAULT_COMPACT_THRESHOLD: u64 = 240_000;

/// When the compaction threshold is derived from the context window, this is the fraction used.
pub const DEFAULT_DYNAMIC_THRESHOLD_FRACTION: f64 = 0.9;

/// Normalize a model name the way the Python lookup table does: lowercase, without an
/// `openai/` prefix and without `.` or `-` separators.
fn model_lookup_key(model: &str) -> String {
    model
        .trim()
        .to_ascii_lowercase()
        .strip_prefix("openai/")
        .map(str::to_string)
        .unwrap_or_else(|| model.trim().to_ascii_lowercase())
        .chars()
        .filter(|c| *c != '.' && *c != '-')
        .collect()
}

/// The context window of a known OpenAI model, in tokens.
///
/// Unknown models return `None`, which makes [`Compaction`] fall back to
/// [`DEFAULT_COMPACT_THRESHOLD`].
pub fn context_window_for_model(model: &str) -> Option<u64> {
    const ONE_M: u64 = 1_047_576;
    const FOUR_HUNDRED_K: u64 = 400_000;
    const TWO_HUNDRED_K: u64 = 200_000;
    const ONE_TWENTY_EIGHT_K: u64 = 128_000;

    let key = model_lookup_key(model);
    let families: [(&[&str], u64); 4] = [
        (
            &[
                "gpt54",
                "gpt54pro",
                "gpt55",
                "gpt55pro",
                "gpt56",
                "gpt56sol",
                "gpt56terra",
                "gpt56luna",
                "gpt41",
                "gpt41mini",
                "gpt41nano",
            ],
            ONE_M,
        ),
        (
            &[
                "gpt5",
                "gpt5codex",
                "gpt5mini",
                "gpt5nano",
                "gpt5pro",
                "gpt51",
                "gpt51codex",
                "gpt51codexmax",
                "gpt51codexmini",
                "gpt52",
                "gpt52codex",
                "gpt52pro",
                "gpt53codex",
                "gpt54mini",
                "gpt54nano",
            ],
            FOUR_HUNDRED_K,
        ),
        (
            &[
                "codexminilatest",
                "o1",
                "o1pro",
                "o3",
                "o3deepresearch",
                "o3mini",
                "o3pro",
                "o4mini",
                "o4minideepresearch",
            ],
            TWO_HUNDRED_K,
        ),
        (
            &[
                "gpt4o",
                "gpt4omini",
                "gpt5chatlatest",
                "gpt51chatlatest",
                "gpt52chatlatest",
                "gpt53chatlatest",
            ],
            ONE_TWENTY_EIGHT_K,
        ),
    ];
    for (models, window) in families {
        // Dated snapshots share the base family name followed by the release date.
        if models
            .iter()
            .any(|m| key == *m || key.starts_with(&format!("{m}20")))
        {
            return Some(window);
        }
    }
    None
}

/// How the compaction threshold is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactionPolicy {
    /// A fixed token threshold.
    Static {
        /// The threshold, in tokens.
        threshold: u64,
    },
    /// A fraction of the model's context window.
    Dynamic {
        /// The model's context window, in tokens.
        context_window: u64,
        /// The fraction of the window at which compaction starts (0.0 to 1.0).
        fraction: f64,
    },
}

impl CompactionPolicy {
    /// The resolved threshold, in tokens.
    pub fn threshold(&self) -> u64 {
        match self {
            CompactionPolicy::Static { threshold } => *threshold,
            CompactionPolicy::Dynamic {
                context_window,
                fraction,
            } => (*context_window as f64 * fraction.clamp(0.0, 1.0)) as u64,
        }
    }
}

/// Enables Responses API conversation compaction for the run.
#[derive(Debug, Clone, Default)]
pub struct Compaction {
    policy: Option<CompactionPolicy>,
}

impl Compaction {
    /// Compaction with a policy derived from the run's model.
    pub fn new() -> Self {
        Compaction::default()
    }

    /// Compaction with an explicit policy.
    pub fn with_policy(mut self, policy: CompactionPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// The policy that applies for the given resolved model name.
    pub fn resolve_policy(&self, model: Option<&str>) -> CompactionPolicy {
        if let Some(policy) = self.policy {
            return policy;
        }
        match model.and_then(context_window_for_model) {
            Some(context_window) => CompactionPolicy::Dynamic {
                context_window,
                fraction: DEFAULT_DYNAMIC_THRESHOLD_FRACTION,
            },
            None => CompactionPolicy::Static {
                threshold: DEFAULT_COMPACT_THRESHOLD,
            },
        }
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for Compaction {
    fn capability_type(&self) -> &str {
        "compaction"
    }

    fn sampling_params(&self, sampling_params: &Map<String, Value>) -> Map<String, Value> {
        let model = sampling_params.get("model").and_then(Value::as_str);
        let threshold = self.resolve_policy(model).threshold();
        json!({
            "context_management": [
                {"type": "compaction", "compact_threshold": threshold}
            ]
        })
        .as_object()
        .cloned()
        .unwrap_or_default()
    }

    fn process_context(&self, context: Vec<InputItem>) -> Vec<InputItem> {
        let last = context
            .iter()
            .rposition(|item| item.0.get("type").and_then(Value::as_str) == Some("compaction"));
        match last {
            Some(index) => context[index..].to_vec(),
            None => context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_policies_from_model_names() {
        let compaction = Compaction::new();
        assert_eq!(
            compaction.resolve_policy(Some("gpt-5.5")).threshold(),
            (1_047_576_f64 * 0.9) as u64
        );
        assert_eq!(
            compaction
                .resolve_policy(Some("gpt-5-2025-08-07"))
                .threshold(),
            360_000
        );
        assert_eq!(
            compaction.resolve_policy(Some("unknown-model")).threshold(),
            DEFAULT_COMPACT_THRESHOLD
        );
        assert_eq!(
            compaction.resolve_policy(None).threshold(),
            DEFAULT_COMPACT_THRESHOLD
        );
        let fixed = Compaction::new().with_policy(CompactionPolicy::Static { threshold: 10 });
        assert_eq!(fixed.resolve_policy(Some("gpt-5.5")).threshold(), 10);
    }

    #[test]
    fn emits_context_management_params() {
        let compaction = Compaction::new();
        let mut params = Map::new();
        params.insert("model".into(), Value::String("gpt-4o".into()));
        let out = Capability::<()>::sampling_params(&compaction, &params);
        assert_eq!(
            Value::Object(out),
            json!({"context_management": [{"type": "compaction", "compact_threshold": 115_200}]})
        );
    }

    #[test]
    fn truncates_context_at_the_last_compaction_item() {
        let compaction = Compaction::new();
        let items = vec![
            InputItem::from_value(json!({"type": "message", "role": "user", "content": "a"})),
            InputItem::from_value(json!({"type": "compaction"})),
            InputItem::from_value(json!({"type": "message", "role": "user", "content": "b"})),
        ];
        let out = Capability::<()>::process_context(&compaction, items);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0["type"], "compaction");
    }
}
