//! A context-scoped `EncryptedSession` append is shielded from a dropped caller, as Python's
//! `await_session_mutation` shields the encrypted wrapper's write
//! (`tests/extensions/memory/test_encrypt_session.py`): once the append has started it runs to
//! completion on its own task, with the run context it was given.
#![cfg(feature = "encrypt")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::InputItem;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::memory::{EncryptedSession, Session, SessionRunContext};
use openai_agents_sdk::run_context::RunContext;
use serde_json::json;

#[derive(Debug)]
struct Tenant(&'static str);

/// A store scoped by the run context's tenant whose append lands one item at a time with a pause
/// between items, so a caller dropped mid-append would leave a partial batch.
#[derive(Debug, Default)]
struct SlowTenantStore {
    items: Mutex<HashMap<&'static str, Vec<InputItem>>>,
}

fn tenant(ctx: SessionRunContext<'_>) -> Result<&'static str> {
    ctx.context::<Tenant>()
        .map(|t| t.0)
        .ok_or_else(|| openai_agents_sdk::AgentsError::other("Tenant context is required."))
}

#[async_trait]
impl Session for SlowTenantStore {
    fn session_id(&self) -> &str {
        "slow"
    }
    async fn get_items(&self, _limit: Option<usize>) -> Result<Vec<InputItem>> {
        Ok(Vec::new())
    }
    async fn add_items(&self, _items: Vec<InputItem>) -> Result<()> {
        Err(openai_agents_sdk::AgentsError::other(
            "Tenant context is required.",
        ))
    }
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        Ok(None)
    }
    async fn clear_session(&self) -> Result<()> {
        Ok(())
    }
    async fn get_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        _limit: Option<usize>,
    ) -> Result<Vec<InputItem>> {
        let tenant = tenant(ctx)?;
        Ok(self
            .items
            .lock()
            .unwrap()
            .get(tenant)
            .cloned()
            .unwrap_or_default())
    }
    async fn add_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        items: Vec<InputItem>,
    ) -> Result<()> {
        let tenant = tenant(ctx)?;
        for item in items {
            self.items
                .lock()
                .unwrap()
                .entry(tenant)
                .or_default()
                .push(item);
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        Ok(())
    }
}

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

fn session() -> Arc<EncryptedSession> {
    Arc::new(
        EncryptedSession::new(
            "slow",
            Arc::new(SlowTenantStore::default()),
            "0123456789abcdef0123456789abcdef",
            600,
        )
        .unwrap(),
    )
}

/// A caller dropped between the first and the second item of a context-scoped append does not
/// leave half the batch: the append finishes, and the store saw the run's own context.
#[tokio::test]
async fn a_dropped_caller_does_not_leave_a_partial_batch() {
    let session = session();
    let context = RunContext::new(Tenant("a"));

    let outcome = tokio::time::timeout(
        Duration::from_millis(60),
        session.add_items_with_context(
            SessionRunContext::new(&context),
            vec![user("first"), user("second"), user("third")],
        ),
    )
    .await;
    assert!(outcome.is_err(), "the caller was dropped mid-append");
    tokio::time::sleep(Duration::from_millis(700)).await;

    let stored = session
        .get_items_with_context(SessionRunContext::new(&context), None)
        .await
        .unwrap();
    assert_eq!(stored, vec![user("first"), user("second"), user("third")]);
    let other = RunContext::new(Tenant("b"));
    assert!(
        session
            .get_items_with_context(SessionRunContext::new(&other), None)
            .await
            .unwrap()
            .is_empty()
    );
}

/// A caller that is not dropped still sees the append's result.
#[tokio::test]
async fn a_scoped_append_reports_its_own_failure() {
    let session = session();
    let anonymous = RunContext::new(());

    let error = session
        .add_items_with_context(SessionRunContext::new(&anonymous), vec![user("lost")])
        .await
        .expect_err("the store needs a tenant");

    assert!(error.to_string().contains("Tenant context is required"));
}
