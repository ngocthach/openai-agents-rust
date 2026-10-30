//! Environment variables a sandbox exports, including ones resolved at start-up. Mirrors the
//! `EnvValue` / `EnvEntry` / `Environment` half of `agents.sandbox.manifest`.
//!
//! Most variables are plain strings, and [`Manifest::with_env`](crate::sandbox::Manifest::with_env)
//! covers those. Some are not: a database password or an API token belongs in a secret store,
//! not in the manifest a program builds and logs. [`EnvValue`] is the hook for those — the
//! manifest records *how* to obtain the value, and it is fetched when the session starts, so the
//! secret lives in the sandbox's environment and nowhere else.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::sandbox::errors::SandboxResult;

/// A source for one environment variable's value. Mirrors Python's `EnvValue`.
///
/// Implement this to read from a secret manager, a credential helper, or anything else that
/// should not be written down in the manifest.
#[async_trait]
pub trait EnvValue: Send + Sync + fmt::Debug {
    /// The value type, matching the Python `type` discriminator.
    fn env_value_type(&self) -> &str;

    /// Produce the value. Called once per session start.
    async fn resolve(&self) -> SandboxResult<String>;
}

/// A literal environment value. Mirrors Python's `StrEnvValue`.
#[derive(Clone, PartialEq, Eq)]
pub struct StrEnvValue {
    value: String,
}

impl StrEnvValue {
    /// A literal value.
    pub fn new(value: impl Into<String>) -> Self {
        StrEnvValue {
            value: value.into(),
        }
    }
}

// The value may be a secret the caller inlined, so it never reaches a log line.
impl fmt::Debug for StrEnvValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StrEnvValue")
            .field("value", &"<redacted>")
            .finish()
    }
}

#[async_trait]
impl EnvValue for StrEnvValue {
    fn env_value_type(&self) -> &str {
        "str"
    }

    async fn resolve(&self) -> SandboxResult<String> {
        Ok(self.value.clone())
    }
}

/// One environment variable and what is known about it. Mirrors Python's `EnvEntry`.
#[derive(Clone, Debug)]
pub struct EnvEntry {
    /// What the variable is for, for the people reading the manifest.
    pub description: Option<String>,
    /// Whether the variable is runtime-only and must not reach a snapshot.
    pub ephemeral: bool,
    /// Where the value comes from.
    pub value: Arc<dyn EnvValue>,
}

impl EnvEntry {
    /// An entry backed by `value`.
    pub fn new(value: Arc<dyn EnvValue>) -> Self {
        EnvEntry {
            description: None,
            ephemeral: false,
            value,
        }
    }

    /// An entry holding a literal string.
    pub fn literal(value: impl Into<String>) -> Self {
        EnvEntry::new(Arc::new(StrEnvValue::new(value)))
    }

    /// Explain what the variable is for.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Mark the variable as runtime-only.
    pub fn ephemeral(mut self, ephemeral: bool) -> Self {
        self.ephemeral = ephemeral;
        self
    }
}

/// The environment a manifest exports. Mirrors Python's `Environment`.
#[derive(Clone, Debug, Default)]
pub struct Environment {
    entries: BTreeMap<String, EnvEntry>,
}

impl Environment {
    /// An empty environment.
    pub fn new() -> Self {
        Environment::default()
    }

    /// Add or replace one variable.
    pub fn with_entry(mut self, key: impl Into<String>, entry: EnvEntry) -> Self {
        self.entries.insert(key.into(), entry);
        self
    }

    /// The variables, by name.
    pub fn entries(&self) -> &BTreeMap<String, EnvEntry> {
        &self.entries
    }

    /// Whether anything is declared here.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The names of the variables that must not reach a snapshot.
    pub fn ephemeral_keys(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.ephemeral)
            .map(|(key, _)| key.as_str())
            .collect()
    }

    /// Resolve every variable.
    ///
    /// Mirrors Python's `Environment.resolve`. Resolution runs sequentially and stops at the
    /// first failure: these are caller-supplied lookups that can reach a secret store, and a
    /// manifest that has already failed should not keep fetching secrets it will not use.
    pub async fn resolve(&self) -> SandboxResult<BTreeMap<String, String>> {
        let mut resolved = BTreeMap::new();
        for (key, entry) in &self.entries {
            resolved.insert(key.clone(), entry.value.resolve().await?);
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::errors::{ErrorCode, SandboxError};

    #[derive(Debug)]
    struct Vault {
        name: &'static str,
    }

    #[async_trait]
    impl EnvValue for Vault {
        fn env_value_type(&self) -> &str {
            "test_vault"
        }

        async fn resolve(&self) -> SandboxResult<String> {
            if self.name == "missing" {
                return Err(SandboxError::new(
                    ErrorCode::SandboxConfigInvalid,
                    "no such secret",
                ));
            }
            Ok(format!("secret-for-{}", self.name))
        }
    }

    #[tokio::test]
    async fn resolves_literal_and_looked_up_values() {
        let environment = Environment::new()
            .with_entry("REGION", EnvEntry::literal("us-east-1"))
            .with_entry(
                "TOKEN",
                EnvEntry::new(Arc::new(Vault { name: "api" }))
                    .with_description("API token")
                    .ephemeral(true),
            );
        let resolved = environment.resolve().await.unwrap();
        assert_eq!(
            resolved.get("REGION").map(String::as_str),
            Some("us-east-1")
        );
        assert_eq!(
            resolved.get("TOKEN").map(String::as_str),
            Some("secret-for-api")
        );
        assert_eq!(environment.ephemeral_keys(), vec!["TOKEN"]);
    }

    #[tokio::test]
    async fn a_failed_lookup_fails_the_whole_environment() {
        let environment = Environment::new()
            .with_entry("TOKEN", EnvEntry::new(Arc::new(Vault { name: "missing" })));
        assert!(environment.resolve().await.is_err());
    }

    #[test]
    fn an_inlined_value_is_not_printed() {
        let entry = EnvEntry::literal("SUPERSECRET");
        assert!(!format!("{entry:?}").contains("SUPERSECRET"));
    }
}
