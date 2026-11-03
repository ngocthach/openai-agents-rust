//! Default model selection. Mirrors `agents.models.default_models`.

use crate::model_settings::{ModelSettings, Reasoning};

/// Environment variable overriding the default model.
pub const OPENAI_DEFAULT_MODEL_ENV_VARIABLE_NAME: &str = "OPENAI_DEFAULT_MODEL";

/// The SDK default model.
pub const DEFAULT_MODEL: &str = "gpt-5.6-luna";

const CHAT_LATEST: &[&str] = &[
    "gpt-5-chat-latest",
    "gpt-5.1-chat-latest",
    "gpt-5.2-chat-latest",
    "gpt-5.3-chat-latest",
];

/// Strip an optional `-YYYY-MM-DD` snapshot suffix.
fn strip_snapshot(name: &str) -> &str {
    let bytes = name.as_bytes();
    if bytes.len() > 11 {
        let tail = &name[name.len() - 11..];
        let t = tail.as_bytes();
        let is_date = t[0] == b'-'
            && t[1..5].iter().all(u8::is_ascii_digit)
            && t[5] == b'-'
            && t[6..8].iter().all(u8::is_ascii_digit)
            && t[8] == b'-'
            && t[9..11].iter().all(u8::is_ascii_digit);
        if is_date {
            return &name[..name.len() - 11];
        }
    }
    name
}

fn default_reasoning_effort(model: &str) -> Option<&'static str> {
    match model {
        "gpt-5.2-codex" => return Some("low"),
        "gpt-5.3-codex" => return Some("none"),
        _ => {}
    }
    match strip_snapshot(model) {
        "gpt-5" => Some("low"),
        "gpt-5.1" | "gpt-5.2" | "gpt-5.4" | "gpt-5.4-mini" | "gpt-5.4-nano" | "gpt-5.5"
        | "gpt-5.6" | "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna" => Some("none"),
        "gpt-5.2-pro" | "gpt-5.4-pro" => Some("medium"),
        _ => None,
    }
}

/// True if the model is a GPT-5 model that requires reasoning settings.
pub fn gpt_5_reasoning_settings_required(model_name: &str) -> bool {
    if CHAT_LATEST.contains(&model_name) {
        return false;
    }
    model_name.starts_with("gpt-5")
}

/// The default model name (`OPENAI_DEFAULT_MODEL` or `gpt-5.6-luna`).
pub fn get_default_model() -> String {
    std::env::var(OPENAI_DEFAULT_MODEL_ENV_VARIABLE_NAME)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

/// True if the default model is a GPT-5 model.
pub fn is_gpt_5_default() -> bool {
    gpt_5_reasoning_settings_required(&get_default_model())
}

/// Default model settings for a model (or the default model when `None`).
pub fn get_default_model_settings(model: Option<&str>) -> ModelSettings {
    let name = model.map(str::to_string).unwrap_or_else(get_default_model);
    if gpt_5_reasoning_settings_required(&name) {
        let mut s = ModelSettings {
            verbosity: Some("low".into()),
            ..Default::default()
        };
        if let Some(effort) = default_reasoning_effort(&name) {
            s.reasoning = Some(Reasoning::effort(effort));
        }
        return s;
    }
    ModelSettings::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt5_defaults() {
        let s = get_default_model_settings(Some("gpt-5.6-luna"));
        assert_eq!(s.reasoning.unwrap().effort.as_deref(), Some("none"));
        assert_eq!(s.verbosity.as_deref(), Some("low"));
        let s = get_default_model_settings(Some("gpt-5-2025-08-07"));
        assert_eq!(s.reasoning.unwrap().effort.as_deref(), Some("low"));
        let s = get_default_model_settings(Some("gpt-5-chat-latest"));
        assert!(s.reasoning.is_none() && s.verbosity.is_none());
        let s = get_default_model_settings(Some("gpt-4.1"));
        assert_eq!(s, ModelSettings::default());
        let s = get_default_model_settings(Some("gpt-5.9-unknown"));
        assert!(s.reasoning.is_none());
        assert_eq!(s.verbosity.as_deref(), Some("low"));
    }
}
