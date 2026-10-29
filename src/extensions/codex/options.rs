//! Options for the Codex CLI process. Mirrors
//! `agents.extensions.experimental.codex.codex_options`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Configuration of the Codex CLI subprocess.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodexOptions {
    /// Absolute path to the `codex` binary; otherwise it is resolved from `CODEX_PATH` and `PATH`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_path_override: Option<String>,
    /// Override the OpenAI base URL for the Codex CLI process (`OPENAI_BASE_URL`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// API key passed to the Codex CLI (`CODEX_API_KEY`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Environment of the Codex CLI process. When set, the parent environment is **not**
    /// inherited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<BTreeMap<String, String>>,
}

impl CodexOptions {
    /// Default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use an explicit `codex` binary.
    pub fn with_codex_path_override(mut self, path: impl Into<String>) -> Self {
        self.codex_path_override = Some(path.into());
        self
    }

    /// Override the OpenAI base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the API key passed to the Codex CLI.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Replace the environment of the Codex CLI process.
    pub fn with_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = Some(env);
        self
    }

    /// Add one environment variable, replacing the inherited environment.
    pub fn with_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env
            .get_or_insert_with(BTreeMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// Resolve the API key the way `codex_tool` does: the explicit key, then `CODEX_API_KEY` and
    /// `OPENAI_API_KEY` from the env override, then from the process environment, then the
    /// default OpenAI key configured on the SDK.
    pub(crate) fn resolve_api_key(&self) -> Option<String> {
        if let Some(key) = self.api_key.as_ref().filter(|k| !k.is_empty()) {
            return Some(key.clone());
        }
        if let Some(env) = &self.env {
            for key in ["CODEX_API_KEY", "OPENAI_API_KEY"] {
                if let Some(value) = env.get(key).filter(|v| !v.is_empty()) {
                    return Some(value.clone());
                }
            }
        }
        for key in ["CODEX_API_KEY", "OPENAI_API_KEY"] {
            if let Ok(value) = std::env::var(key)
                && !value.is_empty()
            {
                return Some(value);
            }
        }
        crate::models::default_openai_key()
    }
}
