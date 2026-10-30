//! The run's stable grouping hierarchy and the prompt cache key derived from it.
//! Port of `agents.run_internal.run_grouping` and `agents.run_internal.prompt_cache_key`.

use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::memory::Session;
use crate::model_settings::ModelSettings;
use crate::models::interface::Model;

/// The `ModelSettings::extra_args` key the Responses and Chat Completions APIs read.
pub(crate) const PROMPT_CACHE_KEY_FIELD: &str = "prompt_cache_key";

/// What a run is grouped by, in precedence order.
/// Mirrors `RunGroupingKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunGroupingKind {
    Conversation,
    Session,
    Group,
    Run,
}

impl RunGroupingKind {
    fn as_str(self) -> &'static str {
        match self {
            RunGroupingKind::Conversation => "conversation",
            RunGroupingKind::Session => "session",
            RunGroupingKind::Group => "group",
            RunGroupingKind::Run => "run",
        }
    }
}

/// Resolve the runner's stable grouping: server conversation, then SDK session, then trace
/// group, then a generated per-run value. Port of `resolve_run_grouping`.
pub(crate) fn resolve_run_grouping(
    conversation_id: Option<&str>,
    session: Option<&Arc<dyn Session>>,
    group_id: Option<&str>,
) -> (RunGroupingKind, String) {
    if let Some(id) = conversation_id.map(str::trim).filter(|v| !v.is_empty()) {
        return (RunGroupingKind::Conversation, id.to_string());
    }
    if let Some(id) = session
        .map(|s| s.session_id())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return (RunGroupingKind::Session, id.to_string());
    }
    if let Some(id) = group_id.map(str::trim).filter(|v| !v.is_empty()) {
        return (RunGroupingKind::Group, id.to_string());
    }
    (RunGroupingKind::Run, crate::util::gen_hex32())
}

/// The grouping as a single id, with the generated per-run value prefixed so it cannot be
/// mistaken for a caller-supplied one. Port of `resolve_run_grouping_id`.
pub fn resolve_run_grouping_id(
    conversation_id: Option<&str>,
    session: Option<&Arc<dyn Session>>,
    group_id: Option<&str>,
) -> String {
    match resolve_run_grouping(conversation_id, session, group_id) {
        (RunGroupingKind::Run, value) => format!("run-{value}"),
        (_, value) => value,
    }
}

fn hashed_key(kind: RunGroupingKind, value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("agents-sdk:{}:{}", kind.as_str(), &hex[..32])
}

fn prompt_cache_key_for_grouping(kind: RunGroupingKind, value: &str) -> String {
    match kind {
        // Without a conversation, session or group id the key is reused only inside this run.
        // That helps a multi-turn agent loop without pretending unrelated runs share a cache.
        RunGroupingKind::Run => format!("agents-sdk:run:{value}"),
        other => hashed_key(other, value),
    }
}

/// `true` when the caller already forwards a `prompt_cache_key`, so the runner must not add
/// one. Port of `_model_settings_has_prompt_cache_key`.
fn settings_have_prompt_cache_key(settings: &ModelSettings) -> bool {
    let has = |map: Option<&serde_json::Map<String, serde_json::Value>>| {
        map.is_some_and(|m| m.contains_key(PROMPT_CACHE_KEY_FIELD))
    };
    has(settings.extra_args.as_ref()) || has(settings.extra_body.as_ref())
}

/// One generated prompt cache key per runner invocation.
///
/// The runner asks for a key on every model turn; this returns the same key each time and
/// hands it to `RunState` so a resumed run keeps its cache group.
/// Port of `PromptCacheKeyResolver`.
#[derive(Debug, Clone, Default)]
pub(crate) struct PromptCacheKeyResolver {
    generated_key: Option<String>,
}

impl PromptCacheKeyResolver {
    /// Seed from a resumed run's snapshot. Port of `PromptCacheKeyResolver.from_run_state`.
    pub(crate) fn from_generated_key(generated_key: Option<String>) -> Self {
        PromptCacheKeyResolver { generated_key }
    }

    /// The key generated so far, for `RunState`.
    pub(crate) fn generated_key(&self) -> Option<String> {
        self.generated_key.clone()
    }

    /// The prompt cache key for this model call, or `None` when the runner must not add one.
    /// Port of `PromptCacheKeyResolver.resolve`.
    pub(crate) fn resolve(
        &mut self,
        settings: &ModelSettings,
        model: &Arc<dyn Model>,
        conversation_id: Option<&str>,
        session: Option<&Arc<dyn Session>>,
        group_id: Option<&str>,
    ) -> Option<String> {
        if settings_have_prompt_cache_key(settings) {
            return None;
        }
        if !model.supports_default_prompt_cache_key() {
            return None;
        }
        if self.generated_key.is_none() {
            let (kind, value) = resolve_run_grouping(conversation_id, session, group_id);
            self.generated_key = Some(prompt_cache_key_for_grouping(kind, &value));
        }
        self.generated_key.clone()
    }
}

/// Add the generated key to `extra_args`, unless the caller already supplied one.
/// Port of `model_settings_with_prompt_cache_key`.
pub(crate) fn apply_prompt_cache_key(settings: &mut ModelSettings, prompt_cache_key: Option<&str>) {
    let Some(key) = prompt_cache_key else {
        return;
    };
    if settings_have_prompt_cache_key(settings) {
        return;
    }
    settings
        .extra_args
        .get_or_insert_with(Default::default)
        .insert(
            PROMPT_CACHE_KEY_FIELD.to_string(),
            serde_json::Value::String(key.to_string()),
        );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemorySession;
    use serde_json::json;

    fn session(id: &str) -> Arc<dyn Session> {
        Arc::new(InMemorySession::new(id))
    }

    #[test]
    fn grouping_follows_conversation_session_group_run() {
        assert_eq!(
            resolve_run_grouping(Some("conv_1"), Some(&session("s")), Some("g")),
            (RunGroupingKind::Conversation, "conv_1".to_string())
        );
        assert_eq!(
            resolve_run_grouping(None, Some(&session("s")), Some("g")),
            (RunGroupingKind::Session, "s".to_string())
        );
        assert_eq!(
            resolve_run_grouping(Some("   "), None, Some(" g ")),
            (RunGroupingKind::Group, "g".to_string())
        );
        let (kind, value) = resolve_run_grouping(None, None, None);
        assert_eq!(kind, RunGroupingKind::Run);
        assert_eq!(value.len(), 32);
    }

    #[test]
    fn grouping_id_prefixes_only_the_generated_value() {
        assert_eq!(
            resolve_run_grouping_id(Some("conv_1"), None, None),
            "conv_1"
        );
        assert!(resolve_run_grouping_id(None, None, None).starts_with("run-"));
    }

    #[test]
    fn keys_match_the_python_format() {
        // sha256("conv_1")[:32]
        let key = prompt_cache_key_for_grouping(RunGroupingKind::Conversation, "conv_1");
        assert!(key.starts_with("agents-sdk:conversation:"));
        let digest = key.rsplit(':').next().unwrap();
        assert_eq!(digest.len(), 32);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            key,
            prompt_cache_key_for_grouping(RunGroupingKind::Conversation, "conv_1"),
            "the key is stable for the same grouping"
        );
        assert_ne!(
            key,
            prompt_cache_key_for_grouping(RunGroupingKind::Session, "conv_1"),
            "the kind is part of the key"
        );
        // A per-run grouping keeps its value verbatim rather than hashing it.
        assert_eq!(
            prompt_cache_key_for_grouping(RunGroupingKind::Run, "abc"),
            "agents-sdk:run:abc"
        );
    }

    #[test]
    fn a_caller_supplied_key_is_never_replaced() {
        let mut settings = ModelSettings::default();
        settings
            .extra_args
            .get_or_insert_with(Default::default)
            .insert(PROMPT_CACHE_KEY_FIELD.into(), json!("mine"));
        assert!(settings_have_prompt_cache_key(&settings));
        apply_prompt_cache_key(&mut settings, Some("generated"));
        assert_eq!(
            settings.extra_args.as_ref().unwrap()[PROMPT_CACHE_KEY_FIELD],
            json!("mine")
        );

        let mut extra_body = ModelSettings::default();
        extra_body
            .extra_body
            .get_or_insert_with(Default::default)
            .insert(PROMPT_CACHE_KEY_FIELD.into(), json!("mine"));
        assert!(settings_have_prompt_cache_key(&extra_body));
    }

    #[test]
    fn apply_writes_into_extra_args() {
        let mut settings = ModelSettings::default();
        apply_prompt_cache_key(&mut settings, Some("agents-sdk:run:abc"));
        assert_eq!(
            settings.extra_args.as_ref().unwrap()[PROMPT_CACHE_KEY_FIELD],
            json!("agents-sdk:run:abc")
        );
        apply_prompt_cache_key(&mut settings, None);
        assert_eq!(settings.extra_args.as_ref().unwrap().len(), 1);
    }
}
