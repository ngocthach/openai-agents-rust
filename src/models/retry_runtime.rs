//! Run-scoped retry switches. Mirrors `agents.models._retry_runtime`.
//!
//! When the SDK's own retry policy owns a call, provider-managed retries have to be turned off
//! for the duration of that call so the two do not multiply. Python uses `ContextVar`s; Rust
//! uses task-locals with the same default (`false`).

use std::future::Future;

tokio::task_local! {
    static DISABLE_PROVIDER_MANAGED_RETRIES: bool;
    static DISABLE_WEBSOCKET_PRE_EVENT_RETRIES: bool;
}

/// Whether the caller asked for provider-managed retries to be suppressed.
pub fn should_disable_provider_managed_retries() -> bool {
    DISABLE_PROVIDER_MANAGED_RETRIES
        .try_with(|v| *v)
        .unwrap_or(false)
}

/// Whether the caller asked for the WebSocket pre-event reconnect-and-resend to be suppressed.
pub fn should_disable_websocket_pre_event_retries() -> bool {
    DISABLE_WEBSOCKET_PRE_EVENT_RETRIES
        .try_with(|v| *v)
        .unwrap_or(false)
}

/// Run `fut` with provider-managed retries disabled (or re-enabled).
pub async fn provider_managed_retries_disabled<F: Future>(disabled: bool, fut: F) -> F::Output {
    DISABLE_PROVIDER_MANAGED_RETRIES.scope(disabled, fut).await
}

/// Run `fut` with the WebSocket pre-event retry disabled (or re-enabled).
pub async fn websocket_pre_event_retries_disabled<F: Future>(disabled: bool, fut: F) -> F::Output {
    DISABLE_WEBSOCKET_PRE_EVENT_RETRIES
        .scope(disabled, fut)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn defaults_to_enabled_and_scopes_the_override() {
        assert!(!should_disable_provider_managed_retries());
        assert!(!should_disable_websocket_pre_event_retries());
        let seen = provider_managed_retries_disabled(true, async {
            should_disable_provider_managed_retries()
        })
        .await;
        assert!(seen);
        assert!(!should_disable_provider_managed_retries());
        let seen = websocket_pre_event_retries_disabled(true, async {
            should_disable_websocket_pre_event_retries()
        })
        .await;
        assert!(seen);
    }
}
