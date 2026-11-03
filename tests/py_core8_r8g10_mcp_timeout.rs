#![cfg(feature = "mcp")]
//! Port of the zero cases of `test_manager_rejects_unsupported_lifecycle_timeouts`
//! (`tests/mcp/test_mcp_server_manager.py`).
//!
//! A `Duration` cannot be negative, NaN or infinite, so zero is the one value Rust has to reject.
//! The manager checks it before any server is touched.

use std::time::Duration;

use openai_agents_sdk::mcp::MCPServerManager;

#[tokio::test]
async fn a_zero_connect_timeout_is_rejected() {
    let mut manager = MCPServerManager::new(vec![]).with_connect_timeout(Some(Duration::ZERO));

    let err = manager.connect_all().await.unwrap_err();
    assert!(
        err.to_string().contains(
            "connect_timeout_seconds must be a positive finite number of seconds or None."
        ),
        "{err}"
    );
    let err = manager.reconnect(false).await.unwrap_err();
    assert!(err.to_string().contains("connect_timeout_seconds"), "{err}");
}

#[tokio::test]
async fn a_zero_cleanup_timeout_is_rejected() {
    let mut manager = MCPServerManager::new(vec![]).with_cleanup_timeout(Some(Duration::ZERO));

    let err = manager.connect_all().await.unwrap_err();
    assert!(
        err.to_string().contains(
            "cleanup_timeout_seconds must be a positive finite number of seconds or None."
        ),
        "{err}"
    );
}

#[tokio::test]
async fn a_zero_timeout_set_after_construction_is_rejected_on_use() {
    let mut manager = MCPServerManager::new(vec![]);
    manager.connect_all().await.unwrap();

    manager.set_connect_timeout(Some(Duration::ZERO));
    assert!(manager.connect_all().await.is_err());

    manager.set_connect_timeout(None);
    manager.connect_all().await.unwrap();
}
