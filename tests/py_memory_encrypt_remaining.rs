//! Parity port of `test_encrypted_pop_preserves_sqlalchemy_history` from
//! `tests/extensions/memory/test_encrypt_session.py`: an `EncryptedSession` over the SQLAlchemy
//! store (Rust: `SqlxSession` on SQLite) pops only encrypted items and leaves history it did
//! not write alone.
#![cfg(all(feature = "sqlx", feature = "sqlite", feature = "encrypt"))]

use std::sync::Arc;

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{EncryptedSession, Session, SqlxSession};
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::item_text;

const KEY: &str = "0123456789abcdef0123456789abcdef";

fn user(text: &str) -> InputItem {
    InputItem::user(text)
}

fn assistant(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": text}))
}

#[tokio::test]
async fn encrypted_pop_preserves_sqlx_history() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!(
        "sqlite://{}?mode=rwc",
        dir.path().join("history.db").display()
    );
    let backend = Arc::new(
        SqlxSession::from_url("test_session", &url, true)
            .await
            .unwrap(),
    );
    let session = EncryptedSession::new("test_session", backend.clone(), KEY, 600).unwrap();

    let items = vec![user("First"), assistant("Second")];
    session.add_items(items.clone()).await.unwrap();
    // History the encrypted wrapper did not write.
    backend
        .add_items(vec![InputItem::from_value(
            json!({"role": "system", "content": "unencrypted history"}),
        )])
        .await
        .unwrap();

    // The plaintext row is skipped, then the newest encrypted item is returned.
    let popped = session.pop_item().await.unwrap().unwrap();
    assert_eq!(item_text(&popped), "Second");
    let remaining = session.get_items(None).await.unwrap();
    assert_eq!(
        remaining.iter().map(item_text).collect::<Vec<_>>(),
        vec!["First"]
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "First"
    );
    assert!(session.pop_item().await.unwrap().is_none());
}
