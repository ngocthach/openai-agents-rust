//! Ports of the session rewind cases of `tests/test_agent_runner.py` (`rewind_session_items`,
//! `wait_for_session_cleanup`) and `tests/memory/test_session_context_wrapper.py` (retry rewind
//! reads, pops and restores in the run's context scope).

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openai_agents_sdk::ServerConversationTracker;
use openai_agents_sdk::memory::{
    Session, SessionRunContext, fingerprint_input_item, rewind_session_items,
    wait_for_session_cleanup,
};
use openai_agents_sdk::run_state::ServerConversationState;
use openai_agents_sdk::{AgentsError, InputItem, RunContext};
use serde_json::{Value, json};

fn item(value: Value) -> InputItem {
    InputItem(value)
}

/// Python's `SimpleListSession` / `CountingSession` / `IdStrippingSession`.
#[derive(Debug, Default)]
struct ListSession {
    items: Mutex<Vec<InputItem>>,
    pop_calls: AtomicUsize,
    get_calls: AtomicUsize,
    strip_ids: bool,
    fail_gets: bool,
    fail_first_get: bool,
}

impl ListSession {
    fn new(history: Vec<InputItem>) -> Self {
        ListSession {
            items: Mutex::new(history),
            ..Default::default()
        }
    }

    fn saved(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }
}

#[async_trait]
impl Session for ListSession {
    fn session_id(&self) -> &str {
        "test"
    }

    fn ignores_item_ids_for_matching(&self) -> bool {
        self.strip_ids
    }

    async fn get_items(&self, limit: Option<usize>) -> openai_agents_sdk::Result<Vec<InputItem>> {
        let call = self.get_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_gets || (self.fail_first_get && call == 0) {
            return Err(AgentsError::user("tail failure"));
        }
        if self.fail_first_get {
            return Ok(Vec::new());
        }
        let items = self.items.lock().unwrap();
        Ok(match limit {
            None => items.clone(),
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        })
    }

    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::Result<()> {
        let mut store = self.items.lock().unwrap();
        for mut it in items {
            if self.strip_ids
                && let Value::Object(m) = &mut it.0
            {
                m.remove("id");
            }
            store.push(it);
        }
        Ok(())
    }

    async fn pop_item(&self) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.pop_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.items.lock().unwrap().pop())
    }

    async fn clear_session(&self) -> openai_agents_sdk::Result<()> {
        self.items.lock().unwrap().clear();
        Ok(())
    }
}

fn tracker_with(server_item_ids: &[&str], sent: &[&InputItem]) -> ServerConversationTracker {
    let state = ServerConversationState {
        server_item_ids: server_item_ids.iter().map(|s| s.to_string()).collect(),
        sent_item_fingerprints: sent
            .iter()
            .map(|i| fingerprint_input_item(i, false).unwrap())
            .collect::<BTreeSet<_>>(),
        ..Default::default()
    };
    let mut tracker = ServerConversationTracker::default();
    tracker.hydrate_from_state(&state, &[], &[], &[], false);
    tracker
}

/// Python: `test_rewind_handles_id_stripped_sessions`.
#[tokio::test]
async fn rewind_matches_items_whose_ids_the_store_stripped() {
    let session = ListSession {
        strip_ids: true,
        ..Default::default()
    };
    let target = item(json!({"id": "message-1", "type": "message", "content": "hello"}));
    session.add_items(vec![target.clone()]).await.unwrap();

    rewind_session_items(&session, &[target], None, None).await;

    assert_eq!(session.pop_calls.load(Ordering::SeqCst), 1);
    assert!(session.saved().is_empty());
}

/// Python: `test_rewind_skips_mismatched_tail_suffix`.
#[tokio::test]
async fn rewind_skips_a_tail_that_does_not_match() {
    let target = item(json!({"type": "message", "role": "user", "content": "target"}));
    let unrelated =
        item(json!({"type": "message", "role": "user", "content": "unrelated tail item"}));
    let session = ListSession::new(vec![target.clone(), unrelated.clone()]);

    rewind_session_items(&session, std::slice::from_ref(&target), None, None).await;

    assert_eq!(session.pop_calls.load(Ordering::SeqCst), 0);
    assert_eq!(session.saved(), vec![target, unrelated]);
}

/// Python: `test_rewind_preserves_unrelated_tail_items_when_server_tracker_cleanup_runs`.
#[tokio::test]
async fn rewind_keeps_unrelated_items_before_the_target_when_a_tracker_is_present() {
    let known = item(
        json!({"id": "msg_server_1", "type": "message", "role": "assistant", "content": "server item"}),
    );
    let unrelated =
        item(json!({"type": "message", "role": "user", "content": "unrelated tail item"}));
    let target = item(json!({"type": "message", "role": "user", "content": "target"}));
    let session = ListSession::new(vec![known.clone(), unrelated.clone(), target.clone()]);
    let tracker = tracker_with(&["msg_server_1"], &[]);

    rewind_session_items(&session, &[target], Some(&tracker), None).await;

    assert_eq!(session.pop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.saved(), vec![known, unrelated]);
}

/// Python: `test_rewind_strips_only_retry_owned_tail_items_before_known_server_item`.
#[tokio::test]
async fn rewind_strips_retry_owned_items_back_to_the_last_known_server_item() {
    let known = item(
        json!({"id": "msg_server_1", "type": "message", "role": "assistant", "content": "server item"}),
    );
    let owned =
        item(json!({"type": "message", "role": "user", "content": "retry-owned local item"}));
    let target = item(json!({"type": "message", "role": "user", "content": "target"}));
    let session = ListSession::new(vec![known.clone(), owned.clone(), target.clone()]);
    let tracker = tracker_with(&["msg_server_1"], &[&owned]);

    rewind_session_items(&session, &[target], Some(&tracker), None).await;

    assert_eq!(session.pop_calls.load(Ordering::SeqCst), 2);
    assert_eq!(session.saved(), vec![known]);
}

/// Python: `test_collect_retry_owned_tail_serializations_returns_empty_for_empty_session`.
#[tokio::test]
async fn an_emptied_session_has_no_stray_tail_to_strip() {
    let target = item(json!({"type": "message", "role": "user", "content": "target"}));
    let session = ListSession::new(vec![target.clone()]);
    let tracker = tracker_with(&[], &[]);

    rewind_session_items(&session, &[target], Some(&tracker), None).await;

    assert_eq!(session.pop_calls.load(Ordering::SeqCst), 1);
    assert!(session.saved().is_empty());
}

/// Python: `test_wait_for_session_cleanup_retries_after_get_items_error`: a failed read waits
/// 0.1s and reads again.
#[tokio::test]
async fn cleanup_verification_retries_after_a_read_error() {
    let target = item(json!({"id": "msg-1", "type": "message", "content": "hello"}));
    let fp = fingerprint_input_item(&target, false).unwrap();
    let session = ListSession {
        fail_first_get: true,
        ..Default::default()
    };

    let started = Instant::now();
    wait_for_session_cleanup(&session, &[fp], 5, false, None).await;
    let elapsed = started.elapsed();

    assert_eq!(session.get_calls.load(Ordering::SeqCst), 2);
    assert!(elapsed >= Duration::from_millis(100), "{elapsed:?}");
    assert!(elapsed < Duration::from_millis(250), "{elapsed:?}");
}

/// Python: `test_wait_for_session_cleanup_logs_when_targets_linger`: two attempts sleep
/// 0.1s then 0.2s before giving up.
#[tokio::test]
async fn cleanup_verification_gives_up_after_the_attempt_budget() {
    let target = item(json!({"id": "msg-1", "type": "message", "content": "hello"}));
    let fp = fingerprint_input_item(&target, false).unwrap();
    let session = ListSession::new(vec![target]);

    let started = Instant::now();
    wait_for_session_cleanup(&session, &[fp], 2, false, None).await;
    let elapsed = started.elapsed();

    assert_eq!(session.get_calls.load(Ordering::SeqCst), 2);
    assert!(elapsed >= Duration::from_millis(300), "{elapsed:?}");
    assert!(elapsed < Duration::from_millis(550), "{elapsed:?}");
}

// ---------------------------------------------------------------------------------------------
// context scope (tests/memory/test_session_context_wrapper.py)
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct TenantContext {
    tenant_id: &'static str,
}

const DEFAULT_SCOPE: &str = "default";

/// Python's `ContextAwareSession`: history is scoped by the tenant of the run context.
#[derive(Debug, Default)]
struct ContextAwareSession {
    items_by_scope: Mutex<HashMap<String, Vec<InputItem>>>,
    calls: Mutex<Vec<(&'static str, bool)>>,
    pops: AtomicUsize,
    fail_pop_number: Option<usize>,
}

impl ContextAwareSession {
    fn scope(ctx: Option<SessionRunContext<'_>>) -> String {
        ctx.and_then(|c| c.context::<TenantContext>())
            .map(|c| c.tenant_id.to_string())
            .unwrap_or_else(|| DEFAULT_SCOPE.to_string())
    }

    fn record(&self, method: &'static str, ctx: Option<SessionRunContext<'_>>) {
        let scoped = ctx.and_then(|c| c.run_context::<TenantContext>()).is_some();
        self.calls.lock().unwrap().push((method, scoped));
    }

    fn scoped(&self, scope: &str) -> Vec<InputItem> {
        self.items_by_scope
            .lock()
            .unwrap()
            .get(scope)
            .cloned()
            .unwrap_or_default()
    }

    fn seed(&self, scope: &str, items: Vec<InputItem>) {
        self.items_by_scope
            .lock()
            .unwrap()
            .insert(scope.to_string(), items);
    }

    fn names(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().iter().map(|(n, _)| *n).collect()
    }

    fn all_scoped(&self) -> bool {
        self.calls.lock().unwrap().iter().all(|(_, scoped)| *scoped)
    }
}

#[async_trait]
impl Session for ContextAwareSession {
    fn session_id(&self) -> &str {
        "context-aware"
    }

    async fn get_items(&self, _limit: Option<usize>) -> openai_agents_sdk::Result<Vec<InputItem>> {
        self.record("get_items", None);
        Ok(Vec::new())
    }

    async fn add_items(&self, _items: Vec<InputItem>) -> openai_agents_sdk::Result<()> {
        self.record("add_items", None);
        Ok(())
    }

    async fn pop_item(&self) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.record("pop_item", None);
        Ok(None)
    }

    async fn clear_session(&self) -> openai_agents_sdk::Result<()> {
        Ok(())
    }

    async fn get_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        limit: Option<usize>,
    ) -> openai_agents_sdk::Result<Vec<InputItem>> {
        self.record("get_items", Some(ctx));
        let map = self.items_by_scope.lock().unwrap();
        let items = map
            .get(&Self::scope(Some(ctx)))
            .cloned()
            .unwrap_or_default();
        Ok(match limit {
            None => items,
            Some(n) => items[items.len().saturating_sub(n)..].to_vec(),
        })
    }

    async fn add_items_with_context(
        &self,
        ctx: SessionRunContext<'_>,
        items: Vec<InputItem>,
    ) -> openai_agents_sdk::Result<()> {
        self.record("add_items", Some(ctx));
        self.items_by_scope
            .lock()
            .unwrap()
            .entry(Self::scope(Some(ctx)))
            .or_default()
            .extend(items);
        Ok(())
    }

    async fn pop_item_with_context(
        &self,
        ctx: SessionRunContext<'_>,
    ) -> openai_agents_sdk::Result<Option<InputItem>> {
        self.record("pop_item", Some(ctx));
        let count = self.pops.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_pop_number == Some(count) {
            return Err(AgentsError::user("pop failed"));
        }
        Ok(self
            .items_by_scope
            .lock()
            .unwrap()
            .entry(Self::scope(Some(ctx)))
            .or_default()
            .pop())
    }
}

fn two_items() -> Vec<InputItem> {
    vec![
        item(json!({"role": "user", "content": "hello"})),
        item(json!({"role": "assistant", "content": "hi"})),
    ]
}

/// Python: `test_retry_rewind_uses_same_context_scope_for_reads_pops_and_cleanup`.
#[tokio::test]
async fn rewind_reads_pops_and_verifies_cleanup_in_the_run_context_scope() {
    let session = ContextAwareSession::default();
    let context = RunContext::new(TenantContext {
        tenant_id: "tenant-a",
    });
    let keep = item(json!({"role": "user", "content": "keep"}));
    session.seed(DEFAULT_SCOPE, vec![keep.clone()]);
    session.seed("tenant-a", two_items());

    rewind_session_items(
        &session,
        &two_items(),
        None,
        Some(SessionRunContext::new(&context)),
    )
    .await;

    assert!(session.scoped("tenant-a").is_empty());
    assert_eq!(session.scoped(DEFAULT_SCOPE), vec![keep]);
    assert_eq!(
        session.names(),
        vec!["get_items", "pop_item", "pop_item", "get_items"]
    );
    assert!(session.all_scoped());
}

/// Python: `test_retry_rewind_restores_partial_pops_in_the_same_context_scope`.
#[tokio::test]
async fn a_failed_pop_restores_the_popped_items_in_the_same_scope() {
    let session = ContextAwareSession {
        fail_pop_number: Some(2),
        ..Default::default()
    };
    let context = RunContext::new(TenantContext {
        tenant_id: "tenant-a",
    });
    let keep = item(json!({"role": "user", "content": "keep"}));
    session.seed("tenant-a", two_items());
    session.seed(DEFAULT_SCOPE, vec![keep.clone()]);

    rewind_session_items(
        &session,
        &two_items(),
        None,
        Some(SessionRunContext::new(&context)),
    )
    .await;

    assert_eq!(session.scoped("tenant-a"), two_items());
    assert_eq!(session.scoped(DEFAULT_SCOPE), vec![keep]);
    assert_eq!(
        session.names(),
        vec!["get_items", "pop_item", "pop_item", "add_items"]
    );
    assert!(session.all_scoped());
}
