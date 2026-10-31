//! Prefix-based routing between providers. Mirrors `agents.models.multi_provider`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::interface::{Model, ModelProvider};
use super::openai_provider::OpenAIProvider;
use crate::error::{AgentsError, Result};

/// Prefixes [`MultiProvider`] resolves on its own, without the caller registering a provider.
pub const BUILT_IN_FALLBACK_PREFIXES: &[&str] = &["litellm", "any-llm"];

fn create_fallback_provider(prefix: &str) -> Result<Arc<dyn ModelProvider>> {
    match prefix {
        "litellm" => Ok(Arc::new(crate::extensions::models::LitellmProvider::new())),
        "any-llm" => Ok(Arc::new(
            crate::extensions::models::AnyLLMProvider::default(),
        )),
        _ => Err(AgentsError::user(format!("Unknown prefix: {prefix}"))),
    }
}

/// A map of model-name prefixes to providers. Port of `MultiProviderMap`.
#[derive(Debug, Clone, Default)]
pub struct MultiProviderMap {
    mapping: HashMap<String, Arc<dyn ModelProvider>>,
}

impl MultiProviderMap {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `prefix` is registered.
    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.mapping.contains_key(prefix)
    }

    /// A copy of the current prefix → provider mapping.
    pub fn get_mapping(&self) -> HashMap<String, Arc<dyn ModelProvider>> {
        self.mapping.clone()
    }

    /// Replace the whole mapping.
    pub fn set_mapping(&mut self, mapping: HashMap<String, Arc<dyn ModelProvider>>) {
        self.mapping = mapping;
    }

    /// The provider registered for `prefix`.
    pub fn get_provider(&self, prefix: &str) -> Option<Arc<dyn ModelProvider>> {
        self.mapping.get(prefix).cloned()
    }

    /// Register a provider for `prefix`.
    pub fn add_provider(&mut self, prefix: impl Into<String>, provider: Arc<dyn ModelProvider>) {
        self.mapping.insert(prefix.into(), provider);
    }

    /// Remove the provider registered for `prefix`.
    pub fn remove_provider(&mut self, prefix: &str) -> Option<Arc<dyn ModelProvider>> {
        self.mapping.remove(prefix)
    }
}

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
///
/// The `litellm/` and `any-llm/` prefixes resolve to the built-in
/// [`crate::extensions::models`] providers unless a provider is registered for them.
#[derive(Debug, Clone)]
pub struct MultiProvider {
    openai: OpenAIProvider,
    providers: HashMap<String, Arc<dyn ModelProvider>>,
    fallback_providers: Arc<Mutex<HashMap<String, Arc<dyn ModelProvider>>>>,
    openai_prefix_mode: PrefixMode,
    unknown_prefix_mode: PrefixMode,
}

impl Default for MultiProvider {
    fn default() -> Self {
        MultiProvider {
            openai: OpenAIProvider::new(),
            providers: HashMap::new(),
            fallback_providers: Arc::new(Mutex::new(HashMap::new())),
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

    /// The OpenAI provider names without a prefix (and `openai/...`) resolve through.
    /// Mirrors `MultiProvider.openai_provider`.
    pub fn openai_provider(&self) -> &OpenAIProvider {
        &self.openai
    }

    /// Register a provider for a prefix (e.g. `"litellm"` handles `litellm/...`).
    pub fn with_provider(
        mut self,
        prefix: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
    ) -> Self {
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

    /// Register every provider in `map`, replacing existing entries for the same prefixes.
    pub fn with_provider_map(mut self, map: &MultiProviderMap) -> Self {
        self.providers.extend(map.get_mapping());
        self
    }

    /// The provider for one of the built-in fallback prefixes, created on first use.
    fn fallback_provider(&self, prefix: &str) -> Result<Arc<dyn ModelProvider>> {
        let mut cache = self.fallback_providers.lock().expect("poisoned");
        if let Some(p) = cache.get(prefix) {
            return Ok(p.clone());
        }
        let provider = create_fallback_provider(prefix)?;
        cache.insert(prefix.to_string(), provider.clone());
        Ok(provider)
    }
}

#[async_trait::async_trait]
impl ModelProvider for MultiProvider {
    /// Close the OpenAI provider, every registered provider and every fallback provider.
    async fn close(&self) {
        self.openai.close().await;
        let providers: Vec<Arc<dyn ModelProvider>> = self
            .providers
            .values()
            .cloned()
            .chain(
                self.fallback_providers
                    .lock()
                    .expect("poisoned")
                    .values()
                    .cloned(),
            )
            .collect();
        for provider in providers {
            provider.close().await;
        }
    }

    fn agent_harness_id(&self) -> Option<String> {
        self.openai.agent_harness_id()
    }

    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let Some(name) = model_name else {
            return self.openai.get_model(None);
        };
        let Some((prefix, rest)) = name.split_once('/') else {
            return self.openai.get_model(Some(name));
        };
        // Explicitly registered providers always win over the built-in fallbacks.
        if let Some(p) = self.providers.get(prefix) {
            return p.get_model(Some(rest));
        }
        if BUILT_IN_FALLBACK_PREFIXES.contains(&prefix) {
            return self.fallback_provider(prefix)?.get_model(Some(rest));
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
        assert_eq!(
            p.get_model(Some("openai/gpt-4.1"))
                .unwrap()
                .model_name()
                .as_deref(),
            Some("gpt-4.1")
        );
        assert_eq!(
            p.get_model(Some("gpt-4.1"))
                .unwrap()
                .model_name()
                .as_deref(),
            Some("gpt-4.1")
        );
        assert!(p.get_model(Some("foo/bar")).is_err());
        // Built-in fallbacks resolve without the caller registering anything; the adapter keeps
        // the `provider/model` name it was given.
        assert_eq!(
            p.get_model(Some("litellm/openai/gpt-4.1"))
                .unwrap()
                .model_name()
                .as_deref(),
            Some("openai/gpt-4.1")
        );
        assert!(p.get_model(Some("any-llm/openai/gpt-4.1")).is_ok());
        let p = MultiProvider::new().with_unknown_prefix_mode(PrefixMode::ModelId);
        assert_eq!(
            p.get_model(Some("foo/bar"))
                .unwrap()
                .model_name()
                .as_deref(),
            Some("foo/bar")
        );
    }
}
