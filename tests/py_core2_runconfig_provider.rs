//! Ports of `tests/test_config.py` and `tests/test_openai_provider_client_options.py`: the
//! global OpenAI defaults and how `OpenAIProvider` builds the client its models use.
//!
//! The defaults are process-wide, so every test takes [`lock`] and starts from a known state.
//! Python reads `model._client`; Rust reads the client the provider hands its models through
//! `OpenAIProvider::openai_client`, and the model kind from its `Debug` name.

use std::sync::{Arc, OnceLock};

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::models::interface::{Model, ModelProvider};
use openai_agents_sdk::models::{
    OpenAIApi, OpenAIClient, OpenAIProvider, OpenAIResponsesTransport, clear_default_openai_client,
    get_default_openai_responses_transport, set_default_openai_api, set_default_openai_client,
    set_default_openai_key, set_default_openai_responses_transport,
    set_use_responses_websocket_by_default,
};

async fn lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    clear_default_openai_client();
    set_default_openai_api(OpenAIApi::Responses);
    set_default_openai_responses_transport(OpenAIResponsesTransport::Http);
    for key in ["OPENAI_BASE_URL", "OPENAI_WEBSOCKET_BASE_URL"] {
        set_env(key, None);
    }
    guard
}

fn set_env(key: &str, value: Option<&str>) {
    // SAFETY: every test of this binary holds `lock`.
    unsafe {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
}

/// The struct name a model's `Debug` output starts with: the Rust stand-in for `isinstance`.
fn kind(model: &Arc<dyn Model>) -> String {
    format!("{model:?}")
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or_default()
        .to_string()
}

fn get_model(provider: &OpenAIProvider) -> Arc<dyn Model> {
    provider.get_model(Some("gpt-4")).unwrap()
}

// ---------------------------------------------------------------------------
// test_config.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn set_default_openai_key_reaches_both_model_kinds() {
    let _g = lock().await;
    set_default_openai_key("test_key");
    for use_responses in [false, true] {
        let provider = OpenAIProvider::new().with_use_responses(use_responses);
        let model = get_model(&provider);
        let expected = if use_responses {
            "OpenAIResponsesModel"
        } else {
            "OpenAIChatCompletionsModel"
        };
        assert_eq!(kind(&model), expected);
        assert_eq!(
            provider.openai_client().api_key().as_deref(),
            Some("test_key")
        );
    }
}

#[tokio::test]
async fn set_default_openai_client_reaches_both_model_kinds() {
    let _g = lock().await;
    set_default_openai_client(OpenAIClient::new("test_key"), false);
    for use_responses in [false, true] {
        let provider = OpenAIProvider::new().with_use_responses(use_responses);
        get_model(&provider);
        assert_eq!(
            provider.openai_client().api_key().as_deref(),
            Some("test_key")
        );
    }
}

#[tokio::test]
async fn set_default_openai_api_switches_model_kind() {
    let _g = lock().await;
    assert_eq!(
        kind(&get_model(&OpenAIProvider::new())),
        "OpenAIResponsesModel"
    );
    set_default_openai_api(OpenAIApi::ChatCompletions);
    assert_eq!(
        kind(&get_model(&OpenAIProvider::new())),
        "OpenAIChatCompletionsModel"
    );
    set_default_openai_api(OpenAIApi::Responses);
    assert_eq!(
        kind(&get_model(&OpenAIProvider::new())),
        "OpenAIResponsesModel"
    );
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn set_default_openai_responses_transport_switches_model_kind() {
    let _g = lock().await;
    assert_eq!(
        kind(&get_model(&OpenAIProvider::new())),
        "OpenAIResponsesModel"
    );
    set_default_openai_responses_transport(OpenAIResponsesTransport::Websocket);
    assert_eq!(
        kind(&get_model(&OpenAIProvider::new())),
        "OpenAIResponsesWSModel"
    );
    set_default_openai_responses_transport(OpenAIResponsesTransport::Http);
    assert_eq!(
        kind(&get_model(&OpenAIProvider::new())),
        "OpenAIResponsesModel"
    );
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn openai_provider_rejects_client_with_conflicting_args() {
    let _g = lock().await;
    let client = || OpenAIClient::new("test_key");
    let cases = [
        OpenAIProvider::new()
            .with_client(client())
            .with_api_key("other_key"),
        OpenAIProvider::new()
            .with_client(client())
            .with_base_url("https://example.com"),
        OpenAIProvider::new()
            .with_client(client())
            .with_websocket_base_url("wss://example.com"),
        OpenAIProvider::new()
            .with_client(client())
            .with_api_key("other_key")
            .with_base_url("https://example.com")
            .with_websocket_base_url("wss://example.com"),
    ];
    for provider in cases {
        let error = provider.get_model(Some("gpt-4")).unwrap_err();
        assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
        assert!(error.to_string().contains("Don't provide"), "{error}");
    }
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn openai_provider_transport_override_beats_default() {
    let _g = lock().await;
    set_default_openai_responses_transport(OpenAIResponsesTransport::Websocket);
    let http = OpenAIProvider::new()
        .with_use_responses(true)
        .with_responses_websocket(false);
    assert_eq!(kind(&get_model(&http)), "OpenAIResponsesModel");
    let ws = OpenAIProvider::new()
        .with_use_responses(true)
        .with_responses_websocket(true);
    assert_eq!(kind(&get_model(&ws)), "OpenAIResponsesWSModel");
}

#[tokio::test]
async fn legacy_websocket_default_flag_syncs_transport_getter() {
    let _g = lock().await;
    set_use_responses_websocket_by_default(true);
    assert_eq!(
        get_default_openai_responses_transport(),
        OpenAIResponsesTransport::Websocket
    );
    set_use_responses_websocket_by_default(false);
    assert_eq!(
        get_default_openai_responses_transport(),
        OpenAIResponsesTransport::Http
    );
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn openai_provider_uses_base_urls_from_env() {
    let _g = lock().await;
    set_env("OPENAI_BASE_URL", Some("https://proxy.example.test/v1"));
    set_env(
        "OPENAI_WEBSOCKET_BASE_URL",
        Some("wss://proxy.example.test/v1"),
    );
    let provider = OpenAIProvider::new().with_use_responses(true);
    assert_eq!(kind(&get_model(&provider)), "OpenAIResponsesModel");
    let client = provider.openai_client();
    assert_eq!(client.base_url(), "https://proxy.example.test/v1");
    assert_eq!(
        provider.websocket_base_url().as_deref(),
        Some("wss://proxy.example.test/v1")
    );
    set_env("OPENAI_BASE_URL", None);
    set_env("OPENAI_WEBSOCKET_BASE_URL", None);
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn openai_provider_websocket_base_url_arg_overrides_env() {
    let _g = lock().await;
    set_env(
        "OPENAI_WEBSOCKET_BASE_URL",
        Some("wss://env.example.test/v1"),
    );
    let provider = OpenAIProvider::new()
        .with_use_responses(true)
        .with_websocket_base_url("wss://explicit.example.test/v1");
    assert_eq!(kind(&get_model(&provider)), "OpenAIResponsesModel");
    assert_eq!(
        provider.openai_client().websocket_base_url(),
        "wss://explicit.example.test/v1"
    );
    set_env("OPENAI_WEBSOCKET_BASE_URL", None);
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn openai_provider_reuses_websocket_model_instance_for_same_model_name() {
    let _g = lock().await;
    let provider = OpenAIProvider::new()
        .with_use_responses(true)
        .with_responses_websocket(true);
    let first = get_model(&provider);
    let second = get_model(&provider);
    assert_eq!(kind(&first), "OpenAIResponsesWSModel");
    assert!(Arc::ptr_eq(&first, &second));
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn openai_provider_passes_responses_websocket_options_to_model() {
    use openai_agents_sdk::models::OpenAIResponsesWebSocketOptions;
    let _g = lock().await;
    let provider = OpenAIProvider::new()
        .with_use_responses(true)
        .with_responses_websocket(true)
        .with_client(OpenAIClient::new("sk-test"))
        .with_websocket_options(OpenAIResponsesWebSocketOptions {
            ping_interval: Some(30.0),
            ping_timeout: None,
            ..Default::default()
        });
    let model = get_model(&provider);
    assert_eq!(kind(&model), "OpenAIResponsesWSModel");
    let debug = format!("{model:?}");
    assert!(debug.contains("ping_interval: Some(30.0)"), "{debug}");
    assert!(debug.contains("ping_timeout: None"), "{debug}");
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn responses_websocket_session_passes_keepalive_options_to_provider() {
    use openai_agents_sdk::models::OpenAIResponsesWebSocketOptions;
    use openai_agents_sdk::models::openai_responses_ws::ResponsesWebSocketSession;
    let _g = lock().await;
    let session = ResponsesWebSocketSession::new(OpenAIClient::new("test-key"))
        .with_websocket_options(OpenAIResponsesWebSocketOptions {
            ping_interval: None,
            ping_timeout: None,
            ..Default::default()
        });
    let options = session.provider().websocket_options();
    assert_eq!(options.ping_interval, None);
    assert_eq!(options.ping_timeout, None);
}

#[tokio::test]
async fn openai_provider_does_not_reuse_non_websocket_model_instances() {
    let _g = lock().await;
    let provider = OpenAIProvider::new().with_use_responses(true);
    let first = get_model(&provider);
    let second = get_model(&provider);
    assert_eq!(kind(&first), "OpenAIResponsesModel");
    assert_eq!(kind(&second), "OpenAIResponsesModel");
    assert!(!Arc::ptr_eq(&first, &second));
}

#[cfg(feature = "websocket")]
#[tokio::test]
async fn openai_provider_close_closes_cached_models() {
    let _g = lock().await;
    let provider = OpenAIProvider::new()
        .with_use_responses(true)
        .with_responses_websocket(true);
    let first = get_model(&provider);
    provider.close().await;
    // The cache was emptied, so the next lookup builds a fresh model.
    assert!(!Arc::ptr_eq(&get_model(&provider), &first));
}

// ---------------------------------------------------------------------------
// test_openai_provider_client_options.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_provider_rejects_ignored_options_with_explicit_client() {
    let _g = lock().await;
    for provider in [
        OpenAIProvider::new()
            .with_client(OpenAIClient::new("sk-test"))
            .with_organization("org-test"),
        OpenAIProvider::new()
            .with_client(OpenAIClient::new("sk-test"))
            .with_project("proj-test"),
    ] {
        let error = provider.get_model(Some("gpt-4")).unwrap_err();
        assert!(matches!(error, AgentsError::UserError { .. }));
        assert!(
            error.to_string().contains("organization, or project"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn openai_provider_explicit_options_override_default_client() {
    let _g = lock().await;
    let default_client = OpenAIClient::new("sk-default").with_base_url("https://default.test/v1");
    set_default_openai_client(default_client, false);
    type Check = fn(&OpenAIClient) -> bool;
    let cases: Vec<(OpenAIProvider, Check)> = vec![
        (OpenAIProvider::new().with_api_key("sk-provider"), |c| {
            c.api_key().as_deref() == Some("sk-provider")
        }),
        (
            OpenAIProvider::new().with_base_url("https://provider.example.test/v1"),
            |c| c.base_url() == "https://provider.example.test/v1",
        ),
        (
            OpenAIProvider::new().with_organization("org-provider"),
            |c| c.organization() == Some("org-provider"),
        ),
        (OpenAIProvider::new().with_project("proj-provider"), |c| {
            c.project() == Some("proj-provider")
        }),
        #[cfg(feature = "websocket")]
        (
            OpenAIProvider::new().with_websocket_base_url("wss://provider.example.test/v1"),
            |c| c.websocket_base_url() == "wss://provider.example.test/v1",
        ),
    ];
    for (provider, check) in cases {
        let client = provider.openai_client();
        // A provider-level option builds a fresh client instead of reusing the default one.
        assert_ne!(client.base_url(), "https://default.test/v1", "{client:?}");
        assert!(check(&client), "{client:?}");
    }
    clear_default_openai_client();
}

#[tokio::test]
async fn openai_provider_preserves_explicit_empty_options() {
    let _g = lock().await;
    set_default_openai_client(OpenAIClient::new("sk-default"), false);
    set_default_openai_key("sk-global");

    let client = OpenAIProvider::new().with_api_key("").openai_client();
    assert_eq!(client.api_key().as_deref(), Some(""));

    set_env("OPENAI_BASE_URL", Some("https://global.example.test/v1"));
    let client = OpenAIProvider::new().with_base_url("").openai_client();
    assert_eq!(client.base_url(), "");
    set_env("OPENAI_BASE_URL", None);

    #[cfg(feature = "websocket")]
    {
        set_env(
            "OPENAI_WEBSOCKET_BASE_URL",
            Some("https://global.example.test/v1"),
        );
        let provider = OpenAIProvider::new().with_websocket_base_url("");
        assert_eq!(provider.websocket_base_url().as_deref(), Some(""));
        set_env("OPENAI_WEBSOCKET_BASE_URL", None);
    }
    clear_default_openai_client();
}
