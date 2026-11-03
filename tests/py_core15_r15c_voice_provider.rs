//! Ports of `tests/voice/test_openai_model_provider.py` default-client resolution: the process
//! default client is used unless an explicit option is given, in which case a client is built
//! from the environment and that option. Process-wide state, so one test owns it.
#![cfg(feature = "voice")]

use openai_agents_sdk::models::{
    OpenAIClient, clear_default_openai_client, set_default_openai_client,
};
use openai_agents_sdk::voice::models::OpenAIVoiceModelProvider;

#[test]
fn the_default_client_is_used_unless_an_option_is_explicit() {
    let default = OpenAIClient::new("sk-default").with_base_url("https://default.example.test/v1");
    set_default_openai_client(default.clone(), false);

    let provider = OpenAIVoiceModelProvider::new();
    assert_eq!(
        provider.client().base_url(),
        "https://default.example.test/v1"
    );
    assert_eq!(provider.client().api_key().as_deref(), Some("sk-default"));

    for (provider, check) in [
        (
            OpenAIVoiceModelProvider::new().with_api_key("sk-voice"),
            "api_key",
        ),
        (
            OpenAIVoiceModelProvider::new().with_api_key(""),
            "empty api_key",
        ),
        (
            OpenAIVoiceModelProvider::new().with_base_url("https://voice.example.test/v1"),
            "base_url",
        ),
        (
            OpenAIVoiceModelProvider::new().with_base_url(""),
            "empty base_url",
        ),
        (
            OpenAIVoiceModelProvider::new().with_organization("org-voice"),
            "organization",
        ),
        (
            OpenAIVoiceModelProvider::new().with_project("proj-voice"),
            "project",
        ),
    ] {
        // None of these starts from the default client.
        assert_ne!(
            provider.client().base_url(),
            "https://default.example.test/v1",
            "{check}"
        );
    }
    let keyed = OpenAIVoiceModelProvider::new().with_api_key("sk-voice");
    assert_eq!(keyed.client().api_key().as_deref(), Some("sk-voice"));
    let organized = OpenAIVoiceModelProvider::new().with_organization("org-voice");
    assert_eq!(organized.client().organization(), Some("org-voice"));
    let project = OpenAIVoiceModelProvider::new().with_project("proj-voice");
    assert_eq!(project.client().project(), Some("proj-voice"));
    assert_eq!(
        OpenAIVoiceModelProvider::new()
            .with_base_url("https://voice.example.test/v1")
            .client()
            .base_url(),
        "https://voice.example.test/v1"
    );

    clear_default_openai_client();
}
