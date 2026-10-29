//! Prefix-based routing between providers. Mirrors `agents.models.multi_provider`.

use std::collections::HashMap;
use std::sync::Arc;

use super::interface::{Model, ModelProvider};
use super::openai_provider::OpenAIProvider;
use crate::error::{AgentsError, Result};

/// How unknown or `openai/` prefixes are treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefixMode {
    /// Strip the prefix (for `openai/`) or reject it (for unknown prefixes). Default.
    #[default]
    Alias,
    /// Pass the full namespaced string to the OpenAI provider as the model id.
    ModelId,
}

/// Routes `prefix/model` names to registered providers; names without a prefix go to OpenAI.
#[derive(Debug, Clone)]
pub struct MultiProvider {
    openai: OpenAIProvider,
    providers: HashMap<String, Arc<dyn ModelProvider>>,
    openai_prefix_mode: PrefixMode,
    unknown_prefix_mode: PrefixMode,
}

impl Default for MultiProvider {
    fn default() -> Self {
        MultiProvider {
            openai: OpenAIProvider::new(),
            providers: HashMap::new(),
            openai_prefix_mode: PrefixMode::Alias,
            unknown_prefix_mode: PrefixMode::Alias,
        }
    }
}

impl MultiProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Use a configured OpenAI provider.
    pub fn with_openai_provider(mut self, provider: OpenAIProvider) -> Self {
        self.openai = provider;
        self
    }

    /// Register a provider for a prefix (e.g. `"litellm"` handles `litellm/...`).
    pub fn with_provider(mut self, prefix: impl Into<String>, provider: Arc<dyn ModelProvider>) -> Self {
        self.providers.insert(prefix.into(), provider);
        self
    }

    /// How `openai/...` names are handled.
    pub fn with_openai_prefix_mode(mut self, mode: PrefixMode) -> Self {
        self.openai_prefix_mode = mode;
        self
    }

    /// How names with unknown prefixes are handled.
    pub fn with_unknown_prefix_mode(mut self, mode: PrefixMode) -> Self {
        self.unknown_prefix_mode = mode;
        self
    }
}

impl ModelProvider for MultiProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let Some(name) = model_name else {
            return self.openai.get_model(None);
        };
        let Some((prefix, rest)) = name.split_once('/') else {
            return self.openai.get_model(Some(name));
        };
        if let Some(p) = self.providers.get(prefix) {
            return p.get_model(Some(rest));
        }
        if prefix == "openai" {
            return match self.openai_prefix_mode {
                PrefixMode::Alias => self.openai.get_model(Some(rest)),
                PrefixMode::ModelId => self.openai.get_model(Some(name)),
            };
        }
        match self.unknown_prefix_mode {
            PrefixMode::ModelId => self.openai.get_model(Some(name)),
            PrefixMode::Alias => Err(AgentsError::user(format!("Unknown prefix: {prefix}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_prefixes() {
        let p = MultiProvider::new();
        assert_eq!(p.get_model(Some("openai/gpt-4.1")).unwrap().model_name().as_deref(), Some("gpt-4.1"));
        assert_eq!(p.get_model(Some("gpt-4.1")).unwrap().model_name().as_deref(), Some("gpt-4.1"));
        assert!(p.get_model(Some("foo/bar")).is_err());
        let p = MultiProvider::new().with_unknown_prefix_mode(PrefixMode::ModelId);
        assert_eq!(p.get_model(Some("foo/bar")).unwrap().model_name().as_deref(), Some("foo/bar"));
    }
}
