//! Trace-safe model configuration. Mirrors `agents.models._trace`.

use serde_json::{Map, Value};

use super::interface::ModelTracing;
use crate::model_settings::ModelSettings;

/// Strip credentials and request parameters from a URL before it is exported in a trace.
pub fn sanitize_url_for_trace(url: &str) -> String {
    let url = url.trim();
    let (scheme, rest) = match url.split_once("://") {
        Some((scheme, rest)) => (scheme, rest),
        None => ("", url),
    };
    let rest = rest.split('#').next().unwrap_or_default();
    let rest = rest.split('?').next().unwrap_or_default();
    let (netloc, path) = match rest.split_once('/') {
        Some((netloc, path)) => (netloc, format!("/{path}")),
        None => (rest, String::new()),
    };
    // `user:password@host` — keep only what follows the last `@`.
    let netloc = netloc.rsplit_once('@').map_or(netloc, |(_, host)| host);
    if scheme.is_empty() {
        format!("{netloc}{path}")
    } else {
        format!("{scheme}://{netloc}{path}")
    }
}

/// The `model_config` recorded on a generation span.
///
/// `metadata` is caller-supplied and may carry sensitive data, so it is dropped unless the run
/// includes data in traces.
pub fn model_config_for_trace(
    model_settings: &ModelSettings,
    tracing: ModelTracing,
    base_url: Option<&str>,
) -> Value {
    let mut config = match model_settings.to_traceable_value() {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    if !tracing.include_data() {
        config.remove("metadata");
    }
    if let Some(base_url) = base_url {
        config.insert(
            "base_url".into(),
            Value::String(sanitize_url_for_trace(base_url)),
        );
    }
    Value::Object(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_urls() {
        assert_eq!(
            sanitize_url_for_trace("https://key:secret@api.example.com/v1/?token=abc#frag"),
            "https://api.example.com/v1/"
        );
        assert_eq!(
            sanitize_url_for_trace("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn drops_metadata_without_data() {
        let settings = ModelSettings {
            metadata: Some(
                [("user".to_string(), "alice".to_string())]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };
        let with = model_config_for_trace(&settings, ModelTracing::Enabled, Some("http://x/v1"));
        assert_eq!(with["metadata"]["user"], "alice");
        assert_eq!(with["base_url"], "http://x/v1");
        let without = model_config_for_trace(
            &settings,
            ModelTracing::EnabledWithoutData,
            Some("http://x/v1"),
        );
        assert!(without.get("metadata").is_none());
    }
}
