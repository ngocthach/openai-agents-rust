//! Port of `test_scripted_sandbox_retains_step_after_matcher_exception` from
//! `tests/test_scripted_sandbox.py`.
//!
//! A Rust matcher returns `bool`, so the way it fails is a panic: an `assert!` on the arguments,
//! which is how the step's documentation suggests checking them. The failed call must leave the
//! step in place, and the session usable, so a retry is answered by the same step.
#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};
use openai_agents_sdk::sandbox::types::ExecResult;

#[tokio::test]
async fn a_failing_matcher_leaves_its_step_for_a_retry() {
    let matcher_calls = Arc::new(AtomicUsize::new(0));
    let counter = matcher_calls.clone();
    let session = Arc::new(scripted_sandbox_session([SandboxStep::new(
        SandboxMethod::Exec,
    )
    .matching(move |_call| {
        let calls = counter.fetch_add(1, Ordering::SeqCst) + 1;
        assert!(calls > 1, "matcher failed");
        true
    })
    .returning_exec(ExecResult {
        exit_code: 0,
        stdout: b"ok".to_vec(),
        stderr: Vec::new(),
    })]));

    let first = {
        let session = session.clone();
        tokio::spawn(async move { session.exec(ExecRequest::argv(["first"])).await }).await
    };
    let panic = first.expect_err("the matcher's failure reaches the caller");
    assert!(panic.is_panic());
    assert_eq!(session.remaining_steps(), 1);

    let retried = session
        .exec(ExecRequest::argv(["retry"]))
        .await
        .expect("the retained step answers the retry");
    assert_eq!(retried.stdout, b"ok");
    assert_eq!(matcher_calls.load(Ordering::SeqCst), 2);
    session.assert_complete();
}
