//! Parity ports of two `tests/models/test_model_retry.py` cases the earlier ports left as
//! Python-only: a websocket request invalidated by a client close is never replayed, however the
//! retry policy is written, on both the non-streamed and the streamed path.
//!
//! The provider advice comes from the real `OpenAIResponsesWSModel`, as in Python.
#![cfg(feature = "websocket")]

mod py_models_retry_support;

use openai_agents_sdk::AgentsError;
use openai_agents_sdk::models::openai_responses_ws::{
    OpenAIResponsesWSModel, WS_CLOSE_INVALIDATED_MESSAGE,
};
use openai_agents_sdk::models::{Model, OpenAIClient};
use openai_agents_sdk::retry::retry_policies;
use py_models_retry_support::*;

fn close_invalidated() -> AgentsError {
    AgentsError::other(WS_CLOSE_INVALIDATED_MESSAGE)
}

fn ws_advice(
    request: &openai_agents_sdk::retry::ModelRetryAdviceRequest<'_>,
) -> Option<openai_agents_sdk::retry::ModelRetryAdvice> {
    OpenAIResponsesWSModel::new("gpt-4", OpenAIClient::new("sk-test")).get_retry_advice(request)
}

#[tokio::test(start_paused = true)]
async fn a_websocket_close_invalidated_request_is_not_replayed() {
    let model = Scripted::with_advice(
        vec![
            Step::Err(close_invalidated()),
            Step::Ok("never", openai_agents_sdk::usage::Usage::default()),
        ],
        ws_advice,
    );
    let error = run(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();

    assert!(
        error.to_string().contains("closed while establishing"),
        "{error}"
    );
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_streamed_websocket_close_invalidated_request_is_not_replayed() {
    let model = Scripted::with_advice(
        vec![
            Step::Stream(vec![Item::Err(close_invalidated())]),
            Step::Stream(vec![event("response.created")]),
        ],
        ws_advice,
    );
    let out = run_streamed(
        &model,
        Opts {
            retry: Some(retry(1, 0.0, Some(retry_policies::network_error()))),
            ..Default::default()
        },
    )
    .await;

    let error = out.error.expect("the stream must fail");
    assert!(
        error.to_string().contains("closed while establishing"),
        "{error}"
    );
    assert_eq!(model.calls(), 1);
}
