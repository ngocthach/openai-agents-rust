//! Port of the streaming tests (`stream_response_with_retry`) and the replay-safety policy tests
//! of `tests/models/test_model_retry.py`, plus the provider-managed-retry flag tests.
//!
//! See `py_models_retry_support` for how Python's direct calls are mapped onto the runner. A
//! streamed run appends a terminal `response.completed` event (the runner needs one), so the
//! event lists below end with it where Python's end with the last scripted event.
//!
//! Tests marked `#[ignore]` fail because `src/run_internal/model_call.rs`, which is outside the
//! models/tracing area, does not implement the corresponding Python rule; each message names it
//! and `plans/reports/py-test-parity-models-tracing.md` gives the minimal fix.

mod py_models_retry_support;

use std::sync::{Arc, Mutex};

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::retry::{
    ModelRetryAdvice, ModelRetryAdviceRequest, ModelRetryNormalizedError, ReplaySafety,
    RetryDecision, RetryPolicy, RetryPolicyContext, get_openai_retry_advice, retry_policies,
};
use openai_agents_sdk::usage::Usage;
use py_models_retry_support::*;

const CROSS_AREA_PROVIDER_FLAGS: &str = "cross-area: run_internal/model_call.rs never scopes \
     provider_managed_retries_disabled / websocket_pre_event_retries_disabled around an attempt";
const CROSS_AREA_REPLAY_UNSAFE: &str = "cross-area: run_internal/model_call.rs has no \
     replay_unsafe_request (set when a ProgrammaticToolCalling tool is present)";

/// What a policy saw: (stream, response_started, replay_safety, stateful_request).
type PolicyView = (bool, bool, ReplaySafety, bool);

fn created_then_completed() -> Vec<String> {
    vec!["response.created".into(), "response.completed".into()]
}

// ---------------------------------------------------------------------------------------------
// retries before output
// ---------------------------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn a_stream_is_retried_before_the_first_event() {
    let model = Scripted::new(vec![
        Step::Stream(vec![Item::Err(connection_error().await)]),
        Step::Stream(vec![event("response.created")]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.25, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.25]);
    assert_eq!(out.types(), created_then_completed());
    // The failed attempt still counts as a request.
    assert_eq!(requests(out.result.as_ref().unwrap()), 2);
}

#[tokio::test(start_paused = true)]
async fn a_stream_timeout_before_output_is_retried() {
    let model = Scripted::new(vec![
        Step::Stream(vec![Item::Hang]),
        Step::Stream(vec![event("response.created")]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            timeout: Some(0.01),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(out.types(), created_then_completed());
}

#[tokio::test(start_paused = true)]
async fn a_stream_timeout_after_output_is_not_retried() {
    let model = Scripted::new(vec![
        Step::Stream(vec![event("response.output_item.added"), Item::Hang]),
        Step::Stream(vec![event("response.created")]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            timeout: Some(0.01),
            ..Default::default()
        },
    )
    .await;
    let err = out.error.expect("timeout error");
    assert!(
        matches!(err, AgentsError::ModelTimeoutError { .. }),
        "{err}"
    );
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_stream_timeout_reports_the_configured_attempt_timeout() {
    let model = Scripted::new(vec![Step::Stream(vec![
        event("response.created"),
        Item::Hang,
    ])]);
    let out = run_streamed(
        &model,
        Opts {
            timeout: Some(0.05),
            ..Default::default()
        },
    )
    .await;
    let err = out.error.expect("timeout error");
    match &err {
        AgentsError::ModelTimeoutError {
            timeout_seconds, ..
        } => assert_eq!(*timeout_seconds, 0.05),
        other => panic!("expected a timeout, got {other}"),
    }
    assert_eq!(err.to_string(), "Model call timed out after 0.05 seconds.");
}

#[tokio::test(start_paused = true)]
async fn stream_retries_stop_when_the_budget_is_exhausted() {
    let model = Scripted::new(vec![
        Step::Stream(vec![Item::Err(connection_error().await)]),
        Step::Stream(vec![Item::Err(connection_error().await)]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.25, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await;
    assert!(is_connect_error(&out.error.expect("error")));
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.25]);
}

#[tokio::test(start_paused = true)]
async fn a_stream_is_retried_after_a_pre_output_event() {
    let model = Scripted::new(vec![
        Step::Stream(vec![
            event("response.created"),
            Item::Err(connection_error().await),
        ]),
        Step::Stream(vec![
            event("response.created"),
            event("response.in_progress"),
        ]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.25, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.25]);
    assert_eq!(
        out.types(),
        [
            "response.created",
            "response.created",
            "response.in_progress",
            "response.completed"
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn a_stream_is_not_retried_after_an_output_event() {
    let model = Scripted::new(vec![
        Step::Stream(vec![
            event("response.output_item.added"),
            Item::Err(connection_error().await),
        ]),
        Step::Stream(vec![event("response.created")]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await;
    assert!(is_connect_error(&out.error.expect("error")));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_stream_retry_reports_conversation_locked_compatibility() {
    let model = Scripted::new(vec![
        Step::Stream(vec![Item::Err(conversation_locked_error())]),
        Step::Stream(vec![event("response.created")]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [1.0]);
    assert_eq!(out.types(), created_then_completed());
    assert_eq!(requests(out.result.as_ref().unwrap()), 2);
}

#[tokio::test(start_paused = true)]
async fn a_timeout_error_from_the_stream_is_retryable() {
    // Python raises the built-in `TimeoutError` from a websocket receive.
    let model = Scripted::new(vec![
        Step::Stream(vec![Item::Err(AgentsError::ModelTimeoutError {
            timeout_seconds: 0.5,
            run_data: None,
        })]),
        Step::Stream(vec![event("response.created")]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.25, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.25]);
}

// ---------------------------------------------------------------------------------------------
// stateful streamed retries
// ---------------------------------------------------------------------------------------------

fn safe_advice() -> impl Fn(&ModelRetryAdviceRequest<'_>) -> Option<ModelRetryAdvice> {
    |_| {
        Some(ModelRetryAdvice {
            suggested: Some(true),
            replay_safety: Some(ReplaySafety::Safe),
            ..Default::default()
        })
    }
}

#[tokio::test(start_paused = true)]
async fn a_stateful_stream_retry_is_allowed_when_the_provider_marks_it_safe() {
    let model = Scripted::with_advice(
        vec![
            Step::Stream(vec![Item::Err(connection_error().await)]),
            Step::Stream(vec![event("response.created")]),
        ],
        safe_advice(),
    );
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.25, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.25]);
    assert_eq!(out.types(), created_then_completed());
}

#[tokio::test(start_paused = true)]
async fn a_stateful_stream_retry_is_allowed_for_http_failure_advice() {
    let model = Scripted::with_advice(
        vec![
            Step::Stream(vec![Item::Err(status_error(500, "server_error"))]),
            Step::Stream(vec![event("response.created")]),
        ],
        get_openai_retry_advice,
    );
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.25, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.25]);
}

#[tokio::test(start_paused = true)]
async fn a_custom_policy_can_override_a_provider_veto_while_streaming() {
    let model = Scripted::with_advice(
        vec![
            Step::Stream(vec![Item::Err(status_error(429, "rate_limit"))]),
            Step::Stream(vec![event("response.created")]),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(false),
                reason: Some("server veto".into()),
                normalized: Some(ModelRetryNormalizedError {
                    status_code: Some(429),
                    ..Default::default()
                }),
                ..Default::default()
            })
        },
    );
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.25, Some(retry_policies::http_status(vec![429])))),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.25]);
}

#[tokio::test(start_paused = true)]
async fn a_stateful_stream_retry_without_replay_safety_is_rejected() {
    let model = Scripted::with_advice(
        vec![
            Step::Stream(vec![Item::Err(connection_error().await)]),
            Step::Stream(vec![event("response.created")]),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(true),
                ..Default::default()
            })
        },
    );
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await;
    assert!(is_connect_error(&out.error.expect("error")));
    assert_eq!(model.calls(), 1);
}

// ---------------------------------------------------------------------------------------------
// provider-managed and websocket pre-event retry flags
// ---------------------------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn provider_retries_stay_on_for_the_first_attempt_only() {
    let _ = CROSS_AREA_PROVIDER_FLAGS;
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("resp_provider_retry_flag", usage_requests_1()),
    ]);
    run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.provider_retry_flags(), [false, true]);
}

#[tokio::test(start_paused = true)]
async fn provider_retries_are_off_from_the_first_attempt_on_a_stateful_provider_hint() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp_stateful_provider_retry_flag", usage_requests_1()),
        ],
        safe_advice(),
    );
    run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.provider_retry_flags(), [true, true]);
}

#[tokio::test(start_paused = true)]
async fn a_narrow_stateful_policy_disables_provider_retries() {
    let model = Scripted::new(vec![Step::Err(connection_error().await)]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::http_status(vec![429])))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.calls(), 1);
    assert_eq!(model.provider_retry_flags(), [true]);
}

#[tokio::test(start_paused = true)]
async fn an_omitted_budget_keeps_stateful_provider_retries() {
    // Python's `ModelRetrySettings(policy=...)` has no `max_retries`; the Rust field is a plain
    // number, so the omitted budget is a policy with no retry settings at all.
    let model = Scripted::new(vec![Step::Err(connection_error().await)]);
    let err = run(
        &model,
        Opts {
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.provider_retry_flags(), [false]);
}

#[tokio::test(start_paused = true)]
async fn a_network_only_policy_disables_stateful_provider_retries() {
    let model = Scripted::new(vec![Step::Err(status_error(500, "server_error"))]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_status(&err, 500));
    assert_eq!(model.provider_retry_flags(), [true]);
}

#[tokio::test(start_paused = true)]
async fn a_partial_policy_disables_stateful_provider_retries() {
    let policy = retry_policies::any(vec![
        retry_policies::network_error(),
        retry_policies::http_status(vec![500]),
    ]);
    let model = Scripted::new(vec![Step::Err(status_error(429, "rate_limit_exceeded"))]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_status(&err, 429));
    assert_eq!(model.provider_retry_flags(), [true]);
}

#[tokio::test(start_paused = true)]
async fn an_explicit_no_retry_budget_disables_provider_retries() {
    let model = Scripted::new(vec![Step::Ok(
        "resp_provider_retry_preserved",
        usage_requests_1(),
    )]);
    run(
        &model,
        Opts {
            retry: Some(retry(0, 0.0, Some(retry_policies::never()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 1);
    assert_eq!(model.provider_retry_flags(), [true]);
}

#[tokio::test(start_paused = true)]
async fn provider_retries_are_kept_without_a_runner_policy() {
    let model = Scripted::new(vec![Step::Ok(
        "resp_provider_retry_without_policy",
        usage_requests_1(),
    )]);
    run(
        &model,
        Opts {
            retry: Some(retry(2, 0.0, None)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 1);
    assert_eq!(model.provider_retry_flags(), [false]);
}

#[tokio::test(start_paused = true)]
async fn a_stateful_compat_replay_disables_provider_retries() {
    let model = Scripted::new(vec![
        Step::Err(conversation_locked_error()),
        Step::Ok("resp_stateful_compat_disable_none", usage_requests_1()),
    ]);
    run(
        &model,
        Opts {
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(model.provider_retry_flags(), [false, true]);
    assert_eq!(model.delays(), [1.0]);
}

#[tokio::test(start_paused = true)]
async fn a_streamed_retry_flags_the_provider_on_retry_attempts_only() {
    let model = Scripted::new(vec![
        Step::Stream(vec![Item::Err(connection_error().await)]),
        Step::Stream(vec![event("response.created")]),
    ]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none());
    assert_eq!(model.provider_retry_flags(), [false, true]);
}

#[tokio::test(start_paused = true)]
async fn a_streamed_stateful_provider_hint_disables_provider_retries_from_the_start() {
    let model = Scripted::with_advice(
        vec![
            Step::Stream(vec![Item::Err(connection_error().await)]),
            Step::Stream(vec![event("response.created")]),
        ],
        safe_advice(),
    );
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await;
    assert!(out.error.is_none());
    assert_eq!(model.provider_retry_flags(), [true, true]);
}

#[tokio::test(start_paused = true)]
async fn websocket_pre_event_retries_are_disabled_when_the_runner_manages_retries() {
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("resp_disable_ws_hidden_retry", usage_requests_1()),
    ]);
    run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.websocket_flags(), [true, true]);
}

#[tokio::test(start_paused = true)]
async fn websocket_pre_event_retries_stay_on_with_an_unrelated_policy() {
    let model = Scripted::new(vec![Step::Stream(vec![Item::Err(
        connection_error().await,
    )])]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::http_status(vec![429])))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await;
    assert!(is_connect_error(&out.error.expect("error")));
    assert_eq!(model.calls(), 1);
    assert_eq!(model.websocket_flags(), [false]);
}

#[tokio::test(start_paused = true)]
async fn websocket_pre_event_retries_stay_on_for_a_partial_all_policy() {
    let policy = retry_policies::all(vec![
        retry_policies::network_error(),
        retry_policies::http_status(vec![500]),
    ]);
    let model = Scripted::new(vec![Step::Stream(vec![Item::Err(
        connection_error().await,
    )])]);
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await;
    assert!(is_connect_error(&out.error.expect("error")));
    assert_eq!(model.calls(), 1);
    assert_eq!(model.websocket_flags(), [false]);
}

#[tokio::test(start_paused = true)]
async fn websocket_pre_event_retries_are_disabled_for_an_explicit_no_retry_budget() {
    let model = Scripted::new(vec![Step::Ok(
        "resp_disable_ws_hidden_retry_zero",
        usage_requests_1(),
    )]);
    run(
        &model,
        Opts {
            retry: Some(retry(0, 0.0, Some(retry_policies::never()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.websocket_flags(), [true]);
}

// ---------------------------------------------------------------------------------------------
// replay safety and approval
// ---------------------------------------------------------------------------------------------

/// Advice matching the Responses WebSocket adapter after a response-started disconnect.
fn ws_response_started_advice(request: &ModelRetryAdviceRequest<'_>) -> Option<ModelRetryAdvice> {
    Some(ModelRetryAdvice {
        suggested: Some(false),
        replay_safety: Some(ReplaySafety::Unsafe),
        reason: Some(request.error.to_string()),
        response_started: true,
        ..Default::default()
    })
}

#[test]
fn provider_suggested_keeps_the_public_veto_for_unsafe_advice() {
    let error = AgentsError::other("connection error");
    let normalized = ModelRetryNormalizedError {
        is_network_error: true,
        ..Default::default()
    };
    let advice = ModelRetryAdvice {
        suggested: Some(false),
        replay_safety: Some(ReplaySafety::Unsafe),
        reason: Some("provider veto".into()),
        ..Default::default()
    };
    let ctx = RetryPolicyContext {
        error: &error,
        attempt: 1,
        max_retries: 1,
        stream: false,
        normalized: &normalized,
        provider_advice: Some(&advice),
        previous_response_id: None,
        conversation_id: None,
    };
    let decision = retry_policies::provider_suggested().evaluate(&ctx);
    assert!(!decision.retry);
    assert_eq!(decision.reason.as_deref(), Some("provider veto"));
}

#[tokio::test(start_paused = true)]
async fn a_provider_unsafe_replay_retries_when_the_policy_approves_it_explicitly() {
    let seen: Arc<Mutex<Vec<PolicyView>>> = Arc::default();
    let recorded = seen.clone();
    let policy = RetryPolicy::new(move |ctx| {
        recorded.lock().unwrap().push((
            ctx.stream,
            ctx.response_started(),
            ctx.replay_safety(),
            ctx.stateful_request(),
        ));
        RetryDecision::retry()
            .with_unsafe_replay_approved()
            .with_reason("Approved one replay of a read-only model turn")
    });
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp", Usage::default()),
        ],
        ws_response_started_advice,
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(response_id(&result).as_deref(), Some("resp"));
    assert_eq!(model.calls(), 2);
    // The policy scopes its approval through the structured context.
    assert_eq!(
        *seen.lock().unwrap(),
        [(false, true, ReplaySafety::Unsafe, false)]
    );
}

#[tokio::test(start_paused = true)]
async fn a_provider_unsafe_replay_still_blocks_an_ordinary_retry_decision() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry());
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", Usage::default()),
        ],
        ws_response_started_advice,
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn provider_unsafe_replay_approval_is_ignored_for_streamed_requests() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry().with_unsafe_replay_approved());
    let model = Scripted::with_advice(
        vec![
            Step::Stream(vec![Item::Err(connection_error().await)]),
            Step::Stream(vec![event("response.created")]),
        ],
        ws_response_started_advice,
    );
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            ..Default::default()
        },
    )
    .await;
    assert!(is_connect_error(&out.error.expect("error")));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn unsafe_replay_approval_resolves_the_provider_veto_in_policy_combinators() {
    for combinator in ["all", "any"] {
        for provider_first in [false, true] {
            let approving =
                RetryPolicy::new(|_| RetryDecision::retry().with_unsafe_replay_approved());
            let provider = retry_policies::provider_suggested();
            let policies = if provider_first {
                vec![provider, approving]
            } else {
                vec![approving, provider]
            };
            let combined = match combinator {
                "all" => retry_policies::all(policies),
                _ => retry_policies::any(policies),
            };
            let model = Scripted::with_advice(
                vec![
                    Step::Err(connection_error().await),
                    Step::Ok("resp", Usage::default()),
                ],
                ws_response_started_advice,
            );
            let result = run(
                &model,
                Opts {
                    retry: Some(retry(1, 0.0, Some(combined))),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let label = format!("{combinator} provider_first={provider_first}");
            assert_eq!(response_id(&result).as_deref(), Some("resp"), "{label}");
            assert_eq!(model.calls(), 2, "{label}");
        }
    }
}

#[tokio::test(start_paused = true)]
async fn an_unsafe_replay_approval_applies_to_a_stateful_request() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry().with_unsafe_replay_approved());
    // `auto_previous_response_id` produces this shape: a stateful request that fails closed by
    // default but carries no application-local side effects.
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp", Usage::default()),
        ],
        ws_response_started_advice,
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            previous_response_id: Some("resp_1"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(response_id(&result).as_deref(), Some("resp"));
    assert_eq!(model.calls(), 2);
}

#[tokio::test(start_paused = true)]
async fn a_stateful_request_still_fails_closed_without_explicit_approval() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry());
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", Usage::default()),
        ],
        ws_response_started_advice,
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            conversation_id: Some("conv_1"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_unsafe_approval_does_not_lift_a_stateful_request_with_unknown_safety() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry().with_unsafe_replay_approved());
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("never", Usage::default()),
    ]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            previous_response_id: Some("resp_1"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_policy_cannot_promote_unknown_provider_replay_safety() {
    // Python's policy tries to rewrite `provider_advice.replay_safety`; a Rust policy receives
    // an immutable context, so the closest check is that unknown safety stays unknown and the
    // stateful request is still not replayed.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = seen.clone();
    let policy = RetryPolicy::new(move |ctx| {
        recorded.lock().unwrap().push(ctx.replay_safety());
        RetryDecision::retry()
    });
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", Usage::default()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(true),
                replay_safety: Some(ReplaySafety::Unknown),
                ..Default::default()
            })
        },
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            previous_response_id: Some("resp_1"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.calls(), 1);
    assert_eq!(*seen.lock().unwrap(), [ReplaySafety::Unknown]);
}

#[tokio::test(start_paused = true)]
async fn a_policy_cannot_revoke_provider_safe_replay_evidence() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = seen.clone();
    let policy = RetryPolicy::new(move |ctx| {
        recorded.lock().unwrap().push(ctx.replay_safety());
        RetryDecision::retry()
    });
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp", Usage::default()),
        ],
        safe_advice(),
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            previous_response_id: Some("resp_1"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(response_id(&result).as_deref(), Some("resp"));
    assert_eq!(model.calls(), 2);
    assert_eq!(*seen.lock().unwrap(), [ReplaySafety::Safe]);
}

/// Python also asserts a provider cannot clear an abort inferred from the raw exception class.
/// A Rust error carries no exception chain to infer an abort from, so only the other half is
/// portable: provider normalization may add abort evidence, which vetoes any policy decision.
#[tokio::test(start_paused = true)]
async fn provider_abort_evidence_vetoes_the_policy() {
    let policy_calls = Arc::new(Mutex::new(0));
    let counter = policy_calls.clone();
    let policy = RetryPolicy::new(move |_| {
        *counter.lock().unwrap() += 1;
        RetryDecision::retry().with_unsafe_replay_approved()
    });
    let model = Scripted::with_advice(
        vec![
            Step::Err(AgentsError::other("cancelled")),
            Step::Ok("never", Usage::default()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                normalized: Some(ModelRetryNormalizedError {
                    is_abort: true,
                    ..Default::default()
                }),
                ..Default::default()
            })
        },
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("cancelled"));
    assert_eq!(model.calls(), 1);
    assert_eq!(*policy_calls.lock().unwrap(), 0);
}

// ---------------------------------------------------------------------------------------------
// request-level replay veto (Programmatic Tool Calling)
// ---------------------------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn a_programmatic_request_disables_hidden_provider_retries() {
    let _ = CROSS_AREA_REPLAY_UNSAFE;
    let model = Scripted::new(vec![Step::Ok("resp", Usage::default())]);
    let result = run(
        &model,
        Opts {
            programmatic_tool_calling: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(response_id(&result).as_deref(), Some("resp"));
    assert_eq!(model.provider_retry_flags(), [true]);
}

#[tokio::test(start_paused = true)]
async fn a_programmatic_request_requires_replay_approval_for_a_runner_retry() {
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("never", Usage::default()),
    ]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            programmatic_tool_calling: true,
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_unsafe_replay_approval_does_not_replay_programmatic_tool_calling() {
    let policy_calls = Arc::new(Mutex::new(0));
    let counter = policy_calls.clone();
    let policy = RetryPolicy::new(move |_| {
        *counter.lock().unwrap() += 1;
        RetryDecision::retry().with_unsafe_replay_approved()
    });
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", Usage::default()),
        ],
        ws_response_started_advice,
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            programmatic_tool_calling: true,
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err));
    assert_eq!(model.calls(), 1);
    // The request-level veto stops the retry before the policy is consulted.
    assert_eq!(*policy_calls.lock().unwrap(), 0);
}
