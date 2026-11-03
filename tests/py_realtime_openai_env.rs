//! Ports of the API-key tests of `tests/realtime/test_openai_realtime.py` and
//! `tests/realtime/test_openai_realtime_conversions.py` that depend on `OPENAI_API_KEY`.
//!
//! They live in their own test binary and share one lock, because the environment is
//! process-global and the Rust test harness runs tests on parallel threads.

#![cfg(feature = "realtime")]

use std::sync::Mutex;

use openai_agents_sdk::realtime::{
    OpenAIRealtimeWebSocketModel, RealtimeApiKey, RealtimeModel, RealtimeModelConfig, get_api_key,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Run `body` with `OPENAI_API_KEY` set to `value` (or removed), restoring it afterwards.
fn with_api_key_env<T>(value: Option<&str>, body: impl FnOnce() -> T) -> T {
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous = std::env::var("OPENAI_API_KEY").ok();
    // SAFETY: every test that touches the variable holds `ENV_LOCK`.
    unsafe {
        match value {
            Some(value) => std::env::set_var("OPENAI_API_KEY", value),
            None => std::env::remove_var("OPENAI_API_KEY"),
        }
    }
    let result = body();
    // SAFETY: as above.
    unsafe {
        match previous {
            Some(previous) => std::env::set_var("OPENAI_API_KEY", previous),
            None => std::env::remove_var("OPENAI_API_KEY"),
        }
    }
    result
}

fn run<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

/// `test_get_api_key_from_env`
#[test]
fn get_api_key_falls_back_to_the_environment() {
    let key = with_api_key_env(Some("env-key"), || run(get_api_key(None)));
    assert_eq!(key.unwrap(), "env-key");
}

/// `test_get_api_key_from_callable_async`
#[test]
fn get_api_key_resolves_an_async_provider() {
    let provider = RealtimeApiKey::Provider(std::sync::Arc::new(|| {
        Box::pin(async { Ok("k".to_string()) })
    }));
    let key = with_api_key_env(None, || run(get_api_key(Some(&provider))));
    assert_eq!(key.unwrap(), "k");
}

/// `test_connect_missing_api_key_raises_error`
#[test]
fn connect_without_any_api_key_is_a_user_error() {
    let error = with_api_key_env(None, || {
        run(async {
            OpenAIRealtimeWebSocketModel::new()
                .connect(RealtimeModelConfig::new().with_url("ws://127.0.0.1:9/v1/realtime"))
                .await
        })
    })
    .expect_err("no key anywhere must fail before connecting");
    assert!(
        error.to_string().contains("API key is required"),
        "unexpected error: {error}"
    );
}
