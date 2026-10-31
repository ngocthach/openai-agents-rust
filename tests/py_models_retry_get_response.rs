//! Port of the non-streaming tests of `tests/models/test_model_retry.py`
//! (`get_response_with_retry`) and of the settings checks at the top of that file.
//!
//! Python calls `get_response_with_retry(get_response=..., rewind=..., ...)` directly. The Rust
//! equivalent is private to the runner, so these tests drive it through `Runner::run_with_options`
//! with a scripted [`Model`]: each attempt is one `get_response` call, `ModelSettings::retry`
//! carries the retry settings and `RunOptions` the stateful ids. Python patches `asyncio.sleep`
//! to record delays; here the tokio clock is paused, so a delay is the virtual time between two
//! attempts and the tests run instantly.
//!
//! Python's `rewind` callback (conversation-state rewind between attempts) has no counterpart in
//! the Rust runner, so the `rewinds == n` assertions are expressed through the attempt count.

mod py_models_retry_support;

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::model_settings::{ModelRetryBackoffSettings, ModelSettings};
use openai_agents_sdk::retry::{
    ModelRetryAdvice, ModelRetryAdviceRequest, ModelRetryNormalizedError, ReplaySafety,
    RetryDecision, RetryPolicy, get_openai_retry_advice, retry_policies,
};
use openai_agents_sdk::usage::Usage;
use py_models_retry_support::*;
use serde_json::json;

// ---------------------------------------------------------------------------------------------
// settings validation (top of the file)
// ---------------------------------------------------------------------------------------------

#[test]
fn zero_backoff_values_are_allowed() {
    let backoff = ModelRetryBackoffSettings {
        initial_delay: 0.0,
        max_delay: 0.0,
        multiplier: 0.0,
        jitter: true,
    };
    assert_eq!(backoff.initial_delay, 0.0);
    assert_eq!(backoff.max_delay, 0.0);
    assert_eq!(backoff.multiplier, 0.0);
}

#[test]
fn a_finite_model_call_timeout_is_accepted_and_traced() {
    let settings = ModelSettings {
        timeout: Some(1.25),
        ..Default::default()
    };
    assert_eq!(settings.timeout, Some(1.25));
    assert_eq!(settings.to_traceable_value()["timeout"], json!(1.25));
}

// ---------------------------------------------------------------------------------------------
// timeouts
// ---------------------------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn a_timed_out_stateless_attempt_is_retried() {
    let model = Scripted::new(vec![Step::Hang, Step::Ok("resp", Usage::default())]);
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            timeout: Some(0.01),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(response_id(&result).as_deref(), Some("resp"));
}

#[tokio::test(start_paused = true)]
async fn an_unsafe_stateful_timeout_replay_is_blocked() {
    let model = Scripted::new(vec![Step::Hang, Step::Ok("never", Usage::default())]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            conversation_id: Some("conv_123"),
            timeout: Some(0.01),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelTimeoutError { .. }),
        "{err}"
    );
    assert!(
        err.to_string().contains("timed out after 0.01 seconds"),
        "{err}"
    );
    assert_eq!(model.calls(), 1);
}

// ---------------------------------------------------------------------------------------------
// retries and usage
// ---------------------------------------------------------------------------------------------

#[test]
fn a_non_retryable_status_gets_no_provider_advice() {
    let err = status_error(400, "invalid_request_error");
    let advice = get_openai_retry_advice(&ModelRetryAdviceRequest {
        error: &err,
        attempt: 1,
        stream: false,
        previous_response_id: None,
        conversation_id: None,
    });
    assert!(advice.is_none());
}

#[tokio::test(start_paused = true)]
async fn a_retry_augments_usage_and_sleeps_the_backoff() {
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("resp_123", usage_requests_1()),
    ]);
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.5, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.5]);
    assert_eq!(requests(&result), 2);
}

#[tokio::test(start_paused = true)]
async fn the_successful_request_keeps_its_own_usage_entry() {
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok(
            "resp_usage_entries",
            Usage {
                requests: 1,
                input_tokens: 11,
                output_tokens: 7,
                total_tokens: 18,
                ..Default::default()
            },
        ),
    ]);
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.5, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let usage = result.usage();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.request_usage_entries.len(), 2);
    assert_eq!(usage.request_usage_entries[0].total_tokens, 0);
    assert_eq!(usage.request_usage_entries[1].input_tokens, 11);
    assert_eq!(usage.request_usage_entries[1].output_tokens, 7);
    assert_eq!(usage.request_usage_entries[1].total_tokens, 18);
}

#[tokio::test(start_paused = true)]
async fn a_zero_token_successful_request_still_gets_an_entry() {
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("resp_zero_usage_entries", usage_requests_1()),
    ]);
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.5, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let usage = result.usage();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.request_usage_entries.len(), 2);
    assert_eq!(usage.request_usage_entries[0].total_tokens, 0);
    assert_eq!(usage.request_usage_entries[1].total_tokens, 0);
}

#[tokio::test(start_paused = true)]
async fn inferred_normalized_error_flags_survive_a_partial_provider_override() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp_partial_normalized", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                normalized: Some(ModelRetryNormalizedError {
                    status_code: Some(429),
                    ..Default::default()
                }),
                ..Default::default()
            })
        },
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_partial_normalized")
    );
}

#[tokio::test(start_paused = true)]
async fn an_explicit_false_provider_override_suppresses_the_network_retry() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                normalized: Some(
                    ModelRetryNormalizedError::default()
                        .with_explicit(["is_network_error", "is_timeout"]),
                ),
                ..Default::default()
            })
        },
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_explicit_none_retry_after_override_suppresses_retry_after_retries() {
    let mut headers = std::collections::BTreeMap::new();
    headers.insert("retry-after-ms".to_string(), "1250".to_string());
    let rate_limited = AgentsError::ApiError {
        status: 429,
        message: "rate_limit".into(),
        body: Some(Box::new(
            json!({"error": {"code": "rate_limit", "message": "rate_limit"}}),
        )),
        request_id: None,
        headers: Some(headers),
    };
    let model = Scripted::with_advice(
        vec![
            Step::Err(rate_limited),
            Step::Ok("never", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                normalized: Some(
                    ModelRetryNormalizedError::default().with_explicit(["retry_after"]),
                ),
                ..Default::default()
            })
        },
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::retry_after()))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_status(&err, 429), "{err}");
    assert_eq!(model.calls(), 1);
}

// ---------------------------------------------------------------------------------------------
// conversation_locked compatibility
// ---------------------------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn conversation_locked_is_retried_without_retry_settings() {
    let model = Scripted::new(vec![
        Step::Err(conversation_locked_error()),
        Step::Ok(
            "resp_compat",
            Usage {
                requests: 1,
                input_tokens: 3,
                output_tokens: 2,
                total_tokens: 5,
                ..Default::default()
            },
        ),
    ]);
    let result = run(&model, Opts::default()).await.unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [1.0]);
    let usage = result.usage();
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.request_usage_entries.len(), 2);
    assert_eq!(usage.request_usage_entries[0].total_tokens, 0);
    assert_eq!(usage.request_usage_entries[1].total_tokens, 5);
}

#[tokio::test(start_paused = true)]
async fn conversation_locked_is_retried_when_the_request_is_stateful() {
    let model = Scripted::new(vec![
        Step::Err(conversation_locked_error()),
        Step::Ok("resp_stateful_compat_disable_none", usage_requests_1()),
    ]);
    let result = run(
        &model,
        Opts {
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [1.0]);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_stateful_compat_disable_none")
    );
}

#[tokio::test(start_paused = true)]
async fn an_explicit_no_retry_budget_suppresses_the_conversation_locked_retry() {
    // `max_retries=0` with `never()` is an explicit opt-out, so no compatibility retry either.
    let model = Scripted::new(vec![
        Step::Err(conversation_locked_error()),
        Step::Ok("never", usage_requests_1()),
    ]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(0, 0.0, Some(retry_policies::never()))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_status(&err, 400), "{err}");
    assert_eq!(model.calls(), 1);
    assert!(model.delays().is_empty());
}

#[tokio::test(start_paused = true)]
async fn conversation_locked_compatibility_also_applies_with_retry_settings() {
    let model = Scripted::new(vec![
        Step::Err(conversation_locked_error()),
        Step::Ok("resp_locked_retry_enabled", usage_requests_1()),
    ]);
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [1.0]);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_locked_retry_enabled")
    );
}

#[tokio::test(start_paused = true)]
async fn conversation_locked_compatibility_retries_are_capped() {
    let model = Scripted::new(vec![
        Step::Err(conversation_locked_error()),
        Step::Err(conversation_locked_error()),
        Step::Err(conversation_locked_error()),
        Step::Err(conversation_locked_error()),
    ]);
    let err = run(&model, Opts::default()).await.unwrap_err();
    assert!(is_status(&err, 400), "{err}");
    assert_eq!(model.calls(), 4);
    assert_eq!(model.delays(), [1.0, 2.0, 4.0]);
}

// ---------------------------------------------------------------------------------------------
// stateful replay safety
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
async fn a_stateful_retry_is_allowed_when_the_provider_marks_it_safe() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp_stateful", usage_requests_1()),
        ],
        safe_advice(),
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(requests(&result), 2);
}

#[tokio::test(start_paused = true)]
async fn a_stateful_retry_is_allowed_for_http_failure_advice() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(status_error(429, "rate_limit")),
            Step::Ok("resp_stateful_http_failure", usage_requests_1()),
        ],
        get_openai_retry_advice_fn,
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_stateful_http_failure")
    );
    assert_eq!(requests(&result), 2);
}

fn get_openai_retry_advice_fn(request: &ModelRetryAdviceRequest<'_>) -> Option<ModelRetryAdvice> {
    get_openai_retry_advice(request)
}

#[tokio::test(start_paused = true)]
async fn provider_safe_stateful_retry_works_for_a_generic_policy() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp_stateful_generic_policy", usage_requests_1()),
        ],
        safe_advice(),
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(requests(&result), 2);
}

#[tokio::test(start_paused = true)]
async fn a_stateful_retry_without_replay_safety_is_rejected() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(true),
                ..Default::default()
            })
        },
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::provider_suggested()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn the_provider_error_code_reaches_a_custom_policy() {
    let policy = RetryPolicy::new(|ctx| ctx.normalized.error_code.as_deref() == Some("rate_limit"));
    let model = Scripted::with_advice(
        vec![
            Step::Err(status_error(429, "rate_limit")),
            Step::Ok("resp_rate_limit_retry", usage_requests_1()),
        ],
        get_openai_retry_advice_fn,
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
    assert_eq!(model.calls(), 2);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_rate_limit_retry")
    );
    assert_eq!(requests(&result), 2);
}

// ---------------------------------------------------------------------------------------------
// budget, delays and vetoes
// ---------------------------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn retries_stop_when_the_budget_is_exhausted() {
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Err(connection_error().await),
    ]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.5, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [0.5]);
}

#[tokio::test(start_paused = true)]
async fn retry_after_wins_over_the_backoff() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp_retry_after", Usage::default()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(true),
                retry_after: Some(1.75),
                ..Default::default()
            })
        },
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 5.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.delays(), [1.75]);
    assert_eq!(requests(&result), 2);
}

#[tokio::test(start_paused = true)]
async fn a_provider_hard_veto_stops_every_policy() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(false),
                reason: Some("server veto".into()),
                ..Default::default()
            })
        },
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(
                1,
                0.0,
                Some(retry_policies::any(vec![
                    retry_policies::provider_suggested(),
                    retry_policies::network_error(),
                ])),
            )),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_custom_policy_can_override_a_provider_veto() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(status_error(429, "rate_limit")),
            Step::Ok("resp_custom_policy_override", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(false),
                retry_after: Some(1.75),
                reason: Some("server veto".into()),
                normalized: Some(ModelRetryNormalizedError {
                    status_code: Some(429),
                    retry_after: Some(1.75),
                    ..Default::default()
                }),
                ..Default::default()
            })
        },
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::retry_after()))),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(model.delays(), [1.75]);
    assert_eq!(requests(&result), 2);
}

#[test]
fn any_merges_later_positive_metadata() {
    let policy = retry_policies::any(vec![
        retry_policies::network_error(),
        retry_policies::retry_after(),
    ]);
    let error = AgentsError::other("connection error");
    let normalized = ModelRetryNormalizedError {
        is_network_error: true,
        retry_after: Some(1.75),
        ..Default::default()
    };
    let advice = ModelRetryAdvice {
        retry_after: Some(1.75),
        ..Default::default()
    };
    let ctx = openai_agents_sdk::retry::RetryPolicyContext {
        error: &error,
        attempt: 1,
        max_retries: 2,
        stream: false,
        normalized: &normalized,
        provider_advice: Some(&advice),
        previous_response_id: None,
        conversation_id: None,
    };
    let decision: RetryDecision = policy.evaluate(&ctx);
    assert!(decision.retry);
    assert_eq!(decision.delay, Some(1.75));
}

#[tokio::test(start_paused = true)]
async fn an_unsafe_replay_is_vetoed() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(true),
                replay_safety: Some(ReplaySafety::Unsafe),
                ..Default::default()
            })
        },
    );
    let err = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_explicit_unsafe_replay_approval_lets_the_policy_retry() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry().with_unsafe_replay_approved());
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp_unsafe_approved", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                suggested: Some(true),
                replay_safety: Some(ReplaySafety::Unsafe),
                ..Default::default()
            })
        },
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
    assert_eq!(model.calls(), 2);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_unsafe_approved")
    );
}

#[tokio::test(start_paused = true)]
async fn an_ordinary_retry_decision_does_not_bypass_a_provider_unsafe_veto() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry());
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                replay_safety: Some(ReplaySafety::Unsafe),
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
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_unsafe_replay_approval_applies_to_a_stateful_request() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry().with_unsafe_replay_approved());
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("resp_stateful_unsafe_approved", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                replay_safety: Some(ReplaySafety::Unsafe),
                ..Default::default()
            })
        },
    );
    let result = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(policy))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 2);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_stateful_unsafe_approved")
    );
}

#[tokio::test(start_paused = true)]
async fn a_stateful_request_fails_closed_without_explicit_approval() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry());
    let model = Scripted::with_advice(
        vec![
            Step::Err(connection_error().await),
            Step::Ok("never", usage_requests_1()),
        ],
        |_| {
            Some(ModelRetryAdvice {
                replay_safety: Some(ReplaySafety::Unsafe),
                ..Default::default()
            })
        },
    );
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
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_unsafe_approval_does_not_lift_a_stateful_request_with_unknown_safety() {
    let policy = RetryPolicy::new(|_| RetryDecision::retry().with_unsafe_replay_approved());
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("never", usage_requests_1()),
    ]);
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
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_narrow_policy_does_not_retry_an_unrelated_error() {
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("never", usage_requests_1()),
    ]);
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
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_omitted_retry_budget_does_not_retry_a_policy_error() {
    // `ModelRetrySettings(policy=...)` without `max_retries` has a budget of zero in Python.
    let model = Scripted::new(vec![
        Step::Err(connection_error().await),
        Step::Ok("never", usage_requests_1()),
    ]);
    let err = run(
        &model,
        Opts {
            retry: Some(retry(0, 0.0, Some(retry_policies::network_error()))),
            previous_response_id: Some("resp_prev"),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    assert!(is_connect_error(&err), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_network_only_policy_does_not_retry_a_stateful_500() {
    let model = Scripted::new(vec![
        Step::Err(status_error(500, "server_error")),
        Step::Ok("never", usage_requests_1()),
    ]);
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
    assert!(is_status(&err, 500), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_partial_any_policy_does_not_retry_a_stateful_429() {
    let policy = retry_policies::any(vec![
        retry_policies::network_error(),
        retry_policies::http_status(vec![500]),
    ]);
    let model = Scripted::new(vec![
        Step::Err(status_error(429, "rate_limit_exceeded")),
        Step::Ok("never", usage_requests_1()),
    ]);
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
    assert!(is_status(&err, 429), "{err}");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_policy_less_retry_budget_does_not_retry_a_success() {
    let model = Scripted::new(vec![Step::Ok(
        "resp_provider_retry_without_policy",
        usage_requests_1(),
    )]);
    let result = run(
        &model,
        Opts {
            retry: Some(retry(2, 0.0, None)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(model.calls(), 1);
    assert_eq!(
        response_id(&result).as_deref(),
        Some("resp_provider_retry_without_policy")
    );
}
