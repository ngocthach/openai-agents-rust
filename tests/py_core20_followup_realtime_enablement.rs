//! Realtime `is_enabled` checks that fail. Ports of:
//!
//! - `tests/realtime/test_realtime_handoffs.py::test_collect_enabled_handoffs_cancels_sibling_checks_on_error`
//! - `tests/realtime/test_realtime_model_settings.py::test_filter_enabled_tools_cancels_sibling_checks_on_error`
//! - `tests/realtime/test_realtime_model_settings.py::test_build_model_settings_cancels_cross_group_enablement_on_error`
//! - `tests/realtime/test_session.py::test_realtime_session_boundaries_cancel_cross_group_enablement_on_error`
//!
//! A failing [`IsEnabled::Fallible`] check raises out of the realtime session, as in Python,
//! instead of counting as disabled. Python cancels the sibling checks still in flight
//! (`gather_with_cancel`); in Rust they are dropped, which a drop flag inside the blocked check
//! observes (Python's `cancelled` and `finished` events both map to that drop).
//!
//! Python's `test_session.py` boundaries map to the Rust session like this:
//! `_get_updated_model_settings_from_agent` is `RealtimeSession::update_agent`;
//! `_resolve_dispatch_snapshot` is the snapshot resolved when the session connects (Rust resolves
//! it at connect and on `update_agent`, never lazily on a call), so it is the connect case below;
//! `_filter_enabled_dispatch_snapshot` is the dispatch-time recheck. Rust rechecks only the tool
//! or handoff the call names, so there is no sibling check to cancel there: the case asserts that
//! the failure ends the session instead of answering "not found".

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeHandoff, RealtimeModelEvent,
    RealtimeModelToolCallEvent, RealtimeRunConfig, RealtimeSession, realtime_handoff,
};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::{FunctionTool, IsEnabled};
use serde_json::json;
use tokio::sync::Notify;

/// Sets its flag when dropped: the blocked check it lives in was cancelled.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// What a blocked sibling check reports.
#[derive(Clone, Default)]
struct Blocked {
    started: Arc<Notify>,
    dropped: Arc<AtomicBool>,
    completed: Arc<AtomicBool>,
}

impl Blocked {
    /// A check that announces it started, then never finishes.
    fn check(&self) -> IsEnabled<()> {
        let this = self.clone();
        IsEnabled::fallible(move |_ctx, _agent| {
            let this = this.clone();
            async move {
                let _flag = DropFlag(this.dropped.clone());
                this.started.notify_one();
                futures::future::pending::<()>().await;
                this.completed.store(true, Ordering::SeqCst);
                Ok(true)
            }
        })
    }

    /// A check that waits for the blocked one to start, then fails with `message`.
    fn failing(&self, message: &'static str) -> IsEnabled<()> {
        let started = self.started.clone();
        IsEnabled::fallible(move |_ctx, _agent| {
            let started = started.clone();
            async move {
                started.notified().await;
                Err(AgentsError::user(message))
            }
        })
    }

    fn assert_cancelled(&self) {
        assert!(
            self.dropped.load(Ordering::SeqCst),
            "the sibling check is cancelled"
        );
        assert!(!self.completed.load(Ordering::SeqCst));
    }
}

fn tool(name: &str, is_enabled: IsEnabled<()>) -> FunctionTool<()> {
    FunctionTool::new_raw(
        name,
        "test tool",
        json!({"type": "object", "properties": {}}),
        |_ctx, _args| async { Ok("ok".to_string()) },
    )
    .with_is_enabled(is_enabled)
}

fn handoff(target: &str, is_enabled: IsEnabled<()>) -> RealtimeHandoff<()> {
    realtime_handoff(RealtimeAgent::<()>::new(target).into_arc()).with_is_enabled(is_enabled)
}

/// Python's `_agent_with_cross_group_enablement_failure`: a tool check that fails once the
/// handoff check, which never finishes, has started.
fn agent_with_cross_group_enablement_failure() -> (Arc<RealtimeAgent<()>>, Blocked) {
    let blocked = Blocked::default();
    let agent = RealtimeAgent::<()>::new("parent")
        .with_tool(tool(
            "failing_tool",
            blocked.failing("tool enablement failed"),
        ))
        .with_handoff(handoff("blocking", blocked.check()))
        .into_arc();
    (agent, blocked)
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a failing check must not wait for its blocked siblings")
}

#[tokio::test]
async fn collect_enabled_handoffs_cancels_sibling_checks_on_error() {
    let blocked = Blocked::default();
    let parent = RealtimeAgent::<()>::new("parent")
        .with_handoff(handoff("slow", blocked.check()))
        .with_handoff(handoff("failing", blocked.failing("enablement failed")));

    let err = within(parent.try_get_enabled_handoffs(&RunContext::new(())))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("enablement failed"), "{err}");
    blocked.assert_cancelled();
}

#[tokio::test]
async fn filter_enabled_tools_cancels_sibling_checks_on_error() {
    let blocked = Blocked::default();
    let agent = RealtimeAgent::<()>::new("parent")
        .with_tool(tool("slow", blocked.check()))
        .with_tool(tool("failing", blocked.failing("enablement failed")));

    let err = within(agent.get_all_tools(&RunContext::new(())))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("enablement failed"), "{err}");
    blocked.assert_cancelled();
}

/// Also the `resolve_dispatch_snapshot` boundary of the `test_session.py` case: Rust resolves the
/// dispatch snapshot when the session connects.
#[tokio::test]
async fn build_model_settings_cancels_cross_group_enablement_on_error() {
    let (agent, blocked) = agent_with_cross_group_enablement_failure();
    let model = Arc::new(FakeRealtimeModel::new());

    let err = within(RealtimeSession::connect(
        model.clone(),
        agent,
        (),
        None,
        None,
    ))
    .await
    .expect_err("a failing enablement check fails the connect");

    assert!(err.to_string().contains("tool enablement failed"), "{err}");
    blocked.assert_cancelled();
    assert!(
        model.connect_calls().is_empty(),
        "nothing is sent to the model"
    );
}

/// `test_realtime_session_boundaries_cancel_cross_group_enablement_on_error[get_updated_model_settings]`.
#[tokio::test]
async fn update_agent_cancels_cross_group_enablement_on_error() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = RealtimeSession::connect(
        model.clone(),
        RealtimeAgent::<()>::new("start").into_arc(),
        (),
        None,
        None,
    )
    .await
    .expect("session connects");
    let updates_before = model.session_updates().len();
    let (agent, blocked) = agent_with_cross_group_enablement_failure();

    let err = within(session.update_agent(agent)).await.unwrap_err();

    assert!(err.to_string().contains("tool enablement failed"), "{err}");
    blocked.assert_cancelled();
    assert_eq!(session.current_agent_name(), "start");
    assert_eq!(model.session_updates().len(), updates_before);
    session.close().await.unwrap();
}

/// `test_realtime_session_boundaries_cancel_cross_group_enablement_on_error[filter_enabled_dispatch_snapshot]`:
/// a check that fails when a call is dispatched ends the session with that error.
#[tokio::test]
async fn dispatch_time_enablement_failure_ends_the_session() {
    let failing = Arc::new(AtomicBool::new(false));
    let ran = Arc::new(AtomicBool::new(false));
    let check = {
        let failing = failing.clone();
        IsEnabled::fallible(move |_ctx, _agent| {
            let failing = failing.load(Ordering::SeqCst);
            async move {
                if failing {
                    Err(AgentsError::user("tool enablement failed"))
                } else {
                    Ok(true)
                }
            }
        })
    };
    let lookup = {
        let ran = ran.clone();
        FunctionTool::new_raw(
            "lookup",
            "test tool",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                ran.store(true, Ordering::SeqCst);
                async { Ok("ok".to_string()) }
            },
        )
        .with_is_enabled(check)
    };
    let agent = RealtimeAgent::<()>::new("agent")
        .with_tool(lookup)
        .into_arc();
    let model = Arc::new(FakeRealtimeModel::new());
    let run = RealtimeRunConfig {
        async_tool_calls: false,
        ..Default::default()
    };
    let session = RealtimeSession::connect(model.clone(), agent, (), None, Some(run))
        .await
        .expect("session connects");
    failing.store(true, Ordering::SeqCst);

    assert!(model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("lookup", "call_lookup", "{}")
    )));
    within(async { while session.recv().await.is_some() {} }).await;

    let err = session
        .take_error()
        .expect("the failed check ends the session");
    assert!(err.to_string().contains("tool enablement failed"), "{err}");
    assert!(!ran.load(Ordering::SeqCst));
    assert!(
        model.tool_outputs().is_empty(),
        "the call is not answered as a missing tool"
    );
}
