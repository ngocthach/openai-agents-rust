//! `OpenAIProvider::get_model` without an API key, and where a run reaches it.
//!
//! Python's provider builds its `AsyncOpenAI` client lazily, in `get_model`, and the client
//! refuses to exist without a key. A run resolves the model inside `run_single_turn` (after the
//! blocking input guardrails and the agent-start hooks; a non-streamed turn after the
//! `call_model_input_filter`, a streamed one before it), inside the task that races the parallel
//! input guardrails. Python's test suite supplies `OPENAI_API_KEY=test_key` from its conftest;
//! these tests remove the variable (and never set a process default key) to reach the keyless
//! path, and set the variable where Python's tests rely on the conftest key.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::guardrail::{GuardrailFunctionOutput, InputGuardrail};
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::models::interface::ModelProvider;
use openai_agents_sdk::models::{MultiProvider, OpenAIClient, OpenAIProvider};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::{Agent, AgentsError, InputItem, RunConfig, RunContext, Runner};

const MISSING_KEY: &str = "The api_key client option must be set either by passing api_key to \
                           the client or by setting the OPENAI_API_KEY environment variable";

static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Sets (or removes) `OPENAI_API_KEY` and removes `OPENAI_BASE_URL` for one test, restoring
/// both on drop. Callers hold [`ENV_LOCK`].
struct KeyEnv {
    key: Option<String>,
    base_url: Option<String>,
}

impl KeyEnv {
    fn set(key: Option<&str>) -> Self {
        let saved = KeyEnv {
            key: std::env::var("OPENAI_API_KEY").ok(),
            base_url: std::env::var("OPENAI_BASE_URL").ok(),
        };
        // Safety: every test of this binary that touches the environment holds `ENV_LOCK`.
        unsafe {
            match key {
                Some(key) => std::env::set_var("OPENAI_API_KEY", key),
                None => std::env::remove_var("OPENAI_API_KEY"),
            }
            std::env::remove_var("OPENAI_BASE_URL");
        }
        saved
    }
}

impl Drop for KeyEnv {
    fn drop(&mut self) {
        // Safety: see `KeyEnv::set`.
        unsafe {
            match &self.key {
                Some(value) => std::env::set_var("OPENAI_API_KEY", value),
                None => std::env::remove_var("OPENAI_API_KEY"),
            }
            if let Some(value) = &self.base_url {
                std::env::set_var("OPENAI_BASE_URL", value);
            }
        }
    }
}

fn assert_missing_key(result: Result<impl std::fmt::Debug, AgentsError>) {
    match result {
        Err(AgentsError::UserError { message, .. }) => assert_eq!(message, MISSING_KEY),
        other => panic!("expected the missing-key user error, got {other:?}"),
    }
}

/// Python's provider fails building its client without a key; Responses and Chat Completions
/// alike, and through the multi-provider's `openai/` prefix.
#[tokio::test]
async fn get_model_without_a_key_fails_like_python() {
    let _lock = ENV_LOCK.lock().await;
    let _env = KeyEnv::set(None);
    for use_responses in [true, false] {
        let provider = OpenAIProvider::new().with_use_responses(use_responses);
        assert_missing_key(provider.get_model(Some("gpt-4.1")));
        assert_missing_key(provider.get_model(None));
    }
    assert_missing_key(MultiProvider::new().get_model(Some("openai/gpt-4.1")));
    // Explicit options other than the key still build the client and still need the key.
    assert_missing_key(
        OpenAIProvider::new()
            .with_organization("org")
            .get_model(Some("gpt-4.1")),
    );
}

/// A key from the provider, the environment, or an explicit (even empty) value lets the client
/// be built, as Python accepts `api_key=""`.
#[tokio::test]
async fn get_model_with_a_key_succeeds() {
    let _lock = ENV_LOCK.lock().await;
    {
        let _env = KeyEnv::set(None);
        for key in ["sk-test", ""] {
            assert!(
                OpenAIProvider::new()
                    .with_api_key(key)
                    .get_model(Some("gpt-4.1"))
                    .is_ok()
            );
        }
    }
    let _env = KeyEnv::set(Some("test_key"));
    assert!(OpenAIProvider::new().get_model(Some("gpt-4.1")).is_ok());
    assert!(MultiProvider::new().get_model(Some("gpt-4.1")).is_ok());
}

/// Rust-only, documented: a client the caller built, or one aimed at another base URL (a local
/// OpenAI-compatible server), may stay keyless.
#[tokio::test]
async fn keyless_clients_are_kept_for_explicit_clients_and_other_base_urls() {
    let _lock = ENV_LOCK.lock().await;
    let _env = KeyEnv::set(None);
    assert!(
        OpenAIProvider::new()
            .with_base_url("http://127.0.0.1:11434/v1")
            .get_model(Some("llama"))
            .is_ok()
    );
    assert!(
        OpenAIProvider::new()
            .with_client(OpenAIClient::from_env())
            .get_model(Some("gpt-4.1"))
            .is_ok()
    );
}

#[derive(Default)]
struct Calls {
    agent_start: AtomicUsize,
    llm_start: AtomicUsize,
}

#[async_trait]
impl RunHooks<()> for Calls {
    async fn on_agent_start(&self, _context: &RunContext<()>, _agent: &Agent<()>) {
        self.agent_start.fetch_add(1, Ordering::SeqCst);
    }

    async fn on_llm_start(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _system_prompt: Option<&str>,
        _input_items: &[InputItem],
    ) {
        self.llm_start.fetch_add(1, Ordering::SeqCst);
    }
}

async fn run(
    agent: &Agent<()>,
    options: RunOptions<()>,
    streamed: bool,
) -> Result<(), AgentsError> {
    if !streamed {
        return Runner::run_with_options(agent, "user_message", options)
            .await
            .map(|_| ());
    }
    let mut stream = Runner::run_streamed_with_options(agent, "user_message", options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = futures::StreamExt::next(&mut events).await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// A keyless run fails where Python's `get_model` runs: after the agent-start hooks and before
/// `on_llm_start`; a non-streamed turn calls the `call_model_input_filter` first, a streamed one
/// does not.
#[tokio::test]
async fn a_keyless_run_fails_where_python_resolves_the_model() {
    let _lock = ENV_LOCK.lock().await;
    let _env = KeyEnv::set(None);
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        let hooks = Arc::new(Calls::default());
        let filtered = Arc::new(Mutex::new(0usize));
        let seen = filtered.clone();
        let config = RunConfig::<()>::new().with_call_model_input_filter(move |data| {
            *seen.lock().unwrap() += 1;
            async move { Ok(data.model_data) }
        });
        let options = RunOptions::new(()).hooks(hooks.clone()).run_config(config);
        assert_missing_key(run(&Agent::<()>::new("test"), options, streamed).await);
        assert_eq!(hooks.agent_start.load(Ordering::SeqCst), 1, "{streamed}");
        assert_eq!(hooks.llm_start.load(Ordering::SeqCst), 0, "{streamed}");
        assert_eq!(
            *filtered.lock().unwrap(),
            usize::from(!streamed),
            "{streamed}"
        );
    }
}

fn guardrail(trip: bool, wait: bool) -> InputGuardrail<()> {
    InputGuardrail::new(
        "guardrail_function",
        move |_ctx, _agent, _input| async move {
            if wait {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Ok(if trip {
                GuardrailFunctionOutput::tripwire()
            } else {
                GuardrailFunctionOutput::pass()
            })
        },
    )
}

/// Input guardrails run before the model is resolved where Python runs them first: a blocking
/// one that trips always wins; a parallel one that trips without waiting wins (Python starts the
/// guardrail task while the model task is still in its agent-start hooks); a parallel one that
/// passes leaves the missing-key error.
#[tokio::test]
async fn input_guardrails_and_a_keyless_model() {
    let _lock = ENV_LOCK.lock().await;
    let _env = KeyEnv::set(None);
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for (guard, expect_trip) in [
            (guardrail(true, true).blocking(), true),
            (guardrail(true, false), true),
            (guardrail(false, false), false),
            (guardrail(false, true), false),
        ] {
            let agent = Agent::<()>::new("test").with_input_guardrail(guard);
            let result = run(&agent, RunOptions::new(()), streamed).await;
            if expect_trip {
                assert!(
                    matches!(
                        result,
                        Err(AgentsError::InputGuardrailTripwireTriggered { .. })
                    ),
                    "{streamed}: {result:?}"
                );
            } else {
                assert_missing_key(result);
            }
        }
    }
}

/// A streamed run reports a parallel guardrail's trip in place of the resolution error even
/// when the guardrail finishes later, as it does for a failed model call.
#[tokio::test]
async fn a_streamed_run_waits_for_a_parallel_guardrail_before_the_resolution_error() {
    let _lock = ENV_LOCK.lock().await;
    let _env = KeyEnv::set(None);
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let agent = Agent::<()>::new("test").with_input_guardrail(guardrail(true, true));
    let streamed = run(&agent, RunOptions::new(()), true).await;
    assert!(
        matches!(
            streamed,
            Err(AgentsError::InputGuardrailTripwireTriggered { .. })
        ),
        "{streamed:?}"
    );
    // A non-streamed run drops a guardrail still running when the model fails.
    let agent = Agent::<()>::new("test").with_input_guardrail(guardrail(true, true));
    assert_missing_key(run(&agent, RunOptions::new(()), false).await);
}
