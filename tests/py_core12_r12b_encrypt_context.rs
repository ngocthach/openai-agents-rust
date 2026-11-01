//! Port of `test_encrypted_session_forwards_wrapper_to_all_underlying_operations` from
//! `tests/extensions/memory/test_encrypt_session.py`: an encrypted session passes the run's
//! context on to every operation of a store that scopes its history by that context.
#![cfg(feature = "encrypt")]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use openai_agents_sdk::InputItem;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::memory::{EncryptedSession, Session, SessionRunContext};
use openai_agents_sdk::run_context::RunContext;
use serde_json::json;

#[derive(Debug)]
struct Tenant(&'static str);

/// A store whose history is scoped by the tenant of the run context; a call without a context
/// is refused, as Python's `ContextAwareUnderlying` raises "Tenant context is required.".
#[derive(Debug, Default)]
struct TenantStore {
    items: Mutex<HashMap<&'static str, Vec<InputItem>>>,
}

impl TenantStore {
    fn tenant(ctx: Option<SessionRunContext<'_>>) -> Result<&'static str> {
        ctx.and_then(|c| c.context::<Tenant>())
            .map(|t| t.0)
            .ok_or_else(|| openai_agents_sdk::AgentsError::other("Tenant context is required."))
    }

    fn history<T>(
        &self,
        ctx: Option<SessionRunContext<'_>>,
        f: impl FnOnce(&mut Vec<InputItem>) -> T,
    ) -> Result<T> {
        let tenant = Self::tenant(ctx)?;
        Ok(f(self.items.lock().unwrap().entry(tenant).or_default()))
    }
}

#[async_trait]
impl Session for TenantStore {
    fn session_id(&self) -> &str {
        "test_session"
    }
    async fn get_items(&self, _limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.history(None, |h| h.clone())
    }
    async fn add_items(&self, _items: Vec<InputItem>) -> Result<()> {
        self.history(None, |_| ())
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.history(None, |_| None)
    }
    async fn clear_session(&self) -> Result<()> {
        self.history(None, |_| ())
    }
    async fn get_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        limit: Option<usize>,
    ) -> Result<Vec<InputItem>> {
        self.history(Some(ctx), |h| match limit {
            None => h.clone(),
            Some(n) => h[h.len().saturating_sub(n)..].to_vec(),
        })
    }
    async fn add_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        items: Vec<InputItem>,
    ) -> Result<()> {
        self.history(Some(ctx), |h| h.extend(items))
    }
    async fn pop_item_with_context(&self, ctx: SessionRunContext<'_>) -> Result<Option<InputItem>> {
        self.history(Some(ctx), |h| h.pop())
    }
    async fn clear_session_with_context(&self, ctx: SessionRunContext<'_>) -> Result<()> {
        self.history(Some(ctx), |h| h.clear())
    }
}

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

#[tokio::test]
async fn an_encrypted_session_forwards_the_run_context_to_every_operation() {
    let session = EncryptedSession::new(
        "test_session",
        Arc::new(TenantStore::default()),
        "0123456789abcdef0123456789abcdef",
        600,
    )
    .unwrap();
    let a = RunContext::new(Tenant("a"));
    let b = RunContext::new(Tenant("b"));
    let (a, b) = (SessionRunContext::new(&a), SessionRunContext::new(&b));

    session
        .add_items_with_context(a, vec![user("hello")])
        .await
        .unwrap();
    session
        .add_items_with_context(b, vec![user("other tenant")])
        .await
        .unwrap();
    assert_eq!(
        session.get_items_with_context(a, None).await.unwrap(),
        vec![user("hello")]
    );
    assert_eq!(
        session.pop_item_with_context(a).await.unwrap(),
        Some(user("hello"))
    );
    assert!(
        session
            .get_items_with_context(a, None)
            .await
            .unwrap()
            .is_empty()
    );
    session
        .add_items_with_context(a, vec![user("clear me")])
        .await
        .unwrap();
    session.clear_session_with_context(a).await.unwrap();

    assert!(
        session
            .get_items_with_context(a, None)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        session.get_items_with_context(b, None).await.unwrap(),
        vec![user("other tenant")]
    );
    // Without a run context the store refuses, exactly as it does when used directly.
    assert!(session.get_items(None).await.is_err());
}
