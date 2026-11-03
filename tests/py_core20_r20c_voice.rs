//! Port of `tests/voice/test_openai_stt_client_config.py::test_streaming_stt_websocket_headers_skip_openai_omission_sentinels`.
//!
//! Python marks the organization and project headers with OpenAI's `omit` / `NOT_GIVEN`
//! sentinels; the Rust client's equivalent is `without_header`. The streamed transcription
//! websocket takes its handshake headers from `OpenAIClient::websocket_headers`: the omitted
//! headers are absent even though the client was configured with values for them, the
//! authorization and custom headers stay, and no `openai-log-session` header is invented.

#![cfg(feature = "voice")]

use openai_agents_sdk::models::OpenAIClient;

#[tokio::test]
async fn streaming_stt_websocket_headers_skip_omitted_openai_headers() {
    let client = OpenAIClient::new("sk-client")
        .with_organization("org-client")
        .with_project("proj-client")
        .with_header("X-Proxy-Token", "proxy-token")
        .without_header("OpenAI-Organization")
        .without_header("OpenAI-Project");

    let headers = client
        .websocket_headers("wss://api.openai.com/v1/realtime?intent=transcription")
        .await
        .expect("headers");

    assert_eq!(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        Some("Bearer sk-client")
    );
    assert_eq!(
        headers.get("x-proxy-token").and_then(|v| v.to_str().ok()),
        Some("proxy-token")
    );
    assert!(headers.get("openai-organization").is_none());
    assert!(headers.get("openai-project").is_none());
    assert!(
        headers
            .keys()
            .all(|name| !name.as_str().eq_ignore_ascii_case("openai-log-session"))
    );
}
