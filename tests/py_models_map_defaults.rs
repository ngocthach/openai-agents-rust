//! Parity ports of the Python suites `tests/models/test_default_models.py`,
//! `tests/models/test_map.py`, `tests/models/test_agent_registration.py`,
//! `tests/models/test_openai_client_utils.py` and `tests/models/test_generation_metadata.py`.
//!
//! Python reads `Agent(...).model_settings` straight after construction; Rust resolves the
//! implicit per-model defaults when the run starts, so the equivalent assertions run an agent
//! against a recording provider and inspect the settings the runner handed to the model.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::agent::AgentModel;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::default_models::{
    DEFAULT_MODEL, get_default_model, get_default_model_settings,
    gpt_5_reasoning_settings_required, is_gpt_5_default,
};
use openai_agents_sdk::models::interface::{
    Model, ModelProvider, ModelRequest, ResponseEventStream,
};
use openai_agents_sdk::models::multi_provider::{MultiProvider, MultiProviderMap, PrefixMode};
use openai_agents_sdk::models::openai_client::OpenAIClient;
use openai_agents_sdk::models::openai_provider::OpenAIProvider;
use openai_agents_sdk::models::{
    OPENAI_HARNESS_ID_TRACE_METADATA_KEY, is_official_openai_base_url, model_config_for_trace,
    set_default_openai_harness,
};
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::tracing::{
    Span, Trace, TracingProcessor, set_trace_processors, set_tracing_disabled,
};
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner};
use serde_json::{Map, Value};

const DEFAULT_MODEL_ENV: &str = "OPENAI_DEFAULT_MODEL";
const HARNESS_ENV: &str = "OPENAI_AGENT_HARNESS_ID";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Model resolution and harness-id resolution both read process-wide environment variables, and
/// the trace processor list is process-global, so every test touching either takes this lock.
async fn env_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

fn set_env(key: &str, value: &str) {
    // SAFETY: every test touching the environment holds `env_lock`.
    unsafe { std::env::set_var(key, value) };
}

fn clear_env(key: &str) {
    // SAFETY: every test touching the environment holds `env_lock`.
    unsafe { std::env::remove_var(key) };
}

/// `_gpt_5_default_settings` from the Python suite.
fn gpt5_defaults(effort: Option<&str>) -> ModelSettings {
    ModelSettings {
        reasoning: effort.map(Reasoning::effort),
        verbosity: Some("low".into()),
        ..Default::default()
    }
}

/// A model that records the settings the runner resolved for it and answers with `"ok"`.
#[derive(Debug)]
struct RecordingModel {
    name: String,
    settings: Arc<Mutex<Option<ModelSettings>>>,
}

#[async_trait]
impl Model for RecordingModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        *self.settings.lock().unwrap() = Some(request.model_settings.clone());
        // Real models record a `model_config` on their generation span; mirror that so the
        // sensitive-capture assertions have something to read.
        let span = openai_agents_sdk::tracing::generation_span(
            None,
            Some(self.name.clone()),
            Some(model_config_for_trace(
                &request.model_settings,
                request.tracing,
                Some("https://example.com/v1"),
            )),
            request.tracing.is_disabled(),
        );
        span.start();
        span.finish();
        Ok(ModelResponse {
            output: vec![assistant_message("ok")],
            usage: Usage::single(0, 0),
            response_id: Some("resp".into()),
            request_id: None,
            raw_usage: None,
        })
    }

    async fn stream_response(&self, _request: ModelRequest) -> Result<ResponseEventStream> {
        unreachable!("the default-settings ports never stream")
    }

    fn model_name(&self) -> Option<String> {
        Some(self.name.clone())
    }
}

/// A provider mirroring `OpenAIProvider`'s name resolution: an unnamed model falls back to
/// [`get_default_model`].
#[derive(Debug, Default)]
struct RecordingProvider {
    settings: Arc<Mutex<Option<ModelSettings>>>,
    harness_id: Option<String>,
}

#[async_trait]
impl ModelProvider for RecordingProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        Ok(Arc::new(RecordingModel {
            name: model_name
                .map(str::to_string)
                .unwrap_or_else(get_default_model),
            settings: self.settings.clone(),
        }))
    }

    fn agent_harness_id(&self) -> Option<String> {
        self.harness_id.clone()
    }
}

/// The model settings the runner resolves for `agent`: the Rust equivalent of reading
/// `agent.model_settings` in Python.
async fn resolved_settings(agent: &Agent<()>) -> ModelSettings {
    let provider = Arc::new(RecordingProvider::default());
    let settings = provider.settings.clone();
    let config = RunConfig {
        model_provider: provider,
        tracing_disabled: true,
        ..Default::default()
    };
    Runner::run_with_options(agent, "hello", RunOptions::default().run_config(config))
        .await
        .expect("run");
    let recorded = settings.lock().unwrap().clone();
    recorded.expect("model was called")
}

/// The struct name a model's `Debug` output starts with: the Rust stand-in for `isinstance`.
fn kind(model: &Arc<dyn Model>) -> String {
    let text = format!("{model:?}");
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or_default()
        .to_string()
}

// ---------------------------------------------------------------------------
// test_default_models.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_model_is_gpt_5_6_luna() {
    let _guard = env_lock().await;
    clear_env(DEFAULT_MODEL_ENV);
    assert_eq!(get_default_model(), "gpt-5.6-luna");
    assert_eq!(DEFAULT_MODEL, "gpt-5.6-luna");
    assert!(is_gpt_5_default());
    assert!(gpt_5_reasoning_settings_required(&get_default_model()));
    assert_eq!(
        get_default_model_settings(None),
        gpt5_defaults(Some("none"))
    );
}

#[tokio::test]
async fn is_gpt_5_default_with_real_model_name() {
    let _guard = env_lock().await;
    set_env(DEFAULT_MODEL_ENV, "gpt-5.4");
    assert_eq!(get_default_model(), "gpt-5.4");
    assert!(is_gpt_5_default());
    clear_env(DEFAULT_MODEL_ENV);
}

#[tokio::test]
async fn is_gpt_5_default_returns_false_for_non_gpt_5_default_model() {
    let _guard = env_lock().await;
    set_env(DEFAULT_MODEL_ENV, "gpt-4.1");
    assert_eq!(get_default_model(), "gpt-4.1");
    assert!(!is_gpt_5_default());
    clear_env(DEFAULT_MODEL_ENV);
}

#[test]
fn gpt_5_reasoning_settings_required_detects_gpt_5_models_while_ignoring_chat_latest() {
    for model in [
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.2-pro",
        "gpt-5.4-pro",
        "gpt-5.5",
        "gpt-5-mini",
        "gpt-5-nano",
    ] {
        assert!(gpt_5_reasoning_settings_required(model), "{model}");
    }
    for model in [
        "gpt-5-chat-latest",
        "gpt-5.1-chat-latest",
        "gpt-5.2-chat-latest",
        "gpt-5.3-chat-latest",
    ] {
        assert!(!gpt_5_reasoning_settings_required(model), "{model}");
    }
}

#[test]
fn gpt_5_reasoning_settings_required_returns_false_for_non_gpt_5_models() {
    assert!(!gpt_5_reasoning_settings_required("gpt-4.1"));
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_1_models() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5.1")),
        gpt5_defaults(Some("none"))
    );
    assert_eq!(
        get_default_model_settings(Some("gpt-5.1-2025-11-13")),
        gpt5_defaults(Some("none"))
    );
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_2_models() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5.2")),
        gpt5_defaults(Some("none"))
    );
    assert_eq!(
        get_default_model_settings(Some("gpt-5.2-2025-12-11")),
        gpt5_defaults(Some("none"))
    );
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_3_codex_models() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5.3-codex")),
        gpt5_defaults(Some("none"))
    );
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_4_models() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5.4")),
        gpt5_defaults(Some("none"))
    );
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_4_snapshot_families() {
    for model in [
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano-2026-03-17",
    ] {
        assert_eq!(
            get_default_model_settings(Some(model)),
            gpt5_defaults(Some("none")),
            "{model}"
        );
    }
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_4_mini_and_nano() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5.4-mini")),
        gpt5_defaults(Some("none"))
    );
    assert_eq!(
        get_default_model_settings(Some("gpt-5.4-nano")),
        gpt5_defaults(Some("none"))
    );
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_5_models() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5.5")),
        gpt5_defaults(Some("none"))
    );
    assert_eq!(
        get_default_model_settings(Some("gpt-5.5-2026-04-23")),
        gpt5_defaults(Some("none"))
    );
}

#[test]
fn get_default_model_settings_returns_none_reasoning_defaults_for_gpt_5_6_models() {
    for model in ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] {
        assert_eq!(
            get_default_model_settings(Some(model)),
            gpt5_defaults(Some("none")),
            "{model}"
        );
    }
}

#[tokio::test]
async fn agent_uses_gpt_5_6_model_settings_from_default_model_env() {
    let _guard = env_lock().await;
    for model in ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] {
        set_env(DEFAULT_MODEL_ENV, &model.to_uppercase());
        assert_eq!(get_default_model(), model);
        let agent = Agent::<()>::new("test");
        assert!(agent.model.is_none());
        assert_eq!(
            resolved_settings(&agent).await,
            gpt5_defaults(Some("none")),
            "{model}"
        );
    }
    clear_env(DEFAULT_MODEL_ENV);
}

#[test]
fn get_default_model_settings_returns_low_reasoning_defaults_for_base_gpt_5() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5")),
        gpt5_defaults(Some("low"))
    );
    assert_eq!(
        get_default_model_settings(Some("gpt-5-2025-08-07")),
        gpt5_defaults(Some("low"))
    );
}

#[test]
fn get_default_model_settings_returns_low_reasoning_defaults_for_gpt_5_2_codex() {
    assert_eq!(
        get_default_model_settings(Some("gpt-5.2-codex")),
        gpt5_defaults(Some("low"))
    );
}

#[test]
fn get_default_model_settings_returns_medium_reasoning_defaults_for_gpt_5_pro_models() {
    for model in [
        "gpt-5.2-pro",
        "gpt-5.2-pro-2025-12-11",
        "gpt-5.4-pro",
        "gpt-5.4-pro-2026-03-05",
    ] {
        assert_eq!(
            get_default_model_settings(Some(model)),
            gpt5_defaults(Some("medium")),
            "{model}"
        );
    }
}

#[test]
fn get_default_model_settings_omits_reasoning_for_unconfirmed_gpt_5_variants() {
    for model in [
        "gpt-5-mini",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano",
        "gpt-5-nano-2025-08-07",
        "gpt-5.1-codex",
    ] {
        assert_eq!(
            get_default_model_settings(Some(model)),
            gpt5_defaults(None),
            "{model}"
        );
    }
}

#[test]
fn get_default_model_settings_returns_empty_settings_for_gpt_5_chat_latest_aliases() {
    for model in [
        "gpt-5-chat-latest",
        "gpt-5.1-chat-latest",
        "gpt-5.2-chat-latest",
        "gpt-5.3-chat-latest",
    ] {
        assert_eq!(
            get_default_model_settings(Some(model)),
            ModelSettings::default(),
            "{model}"
        );
    }
}

#[test]
fn get_default_model_settings_returns_empty_settings_for_non_gpt_5_models() {
    assert_eq!(
        get_default_model_settings(Some("gpt-4.1")),
        ModelSettings::default()
    );
}

#[tokio::test]
async fn agent_uses_gpt_5_default_model_settings() {
    let _guard = env_lock().await;
    set_env(DEFAULT_MODEL_ENV, "gpt-5");
    let agent = Agent::<()>::new("test");
    assert!(agent.model.is_none());
    let settings = resolved_settings(&agent).await;
    clear_env(DEFAULT_MODEL_ENV);
    assert_eq!(
        settings.reasoning.and_then(|r| r.effort).as_deref(),
        Some("low")
    );
    assert_eq!(settings.verbosity.as_deref(), Some("low"));
}

#[tokio::test]
async fn agent_uses_model_specific_settings_for_explicit_gpt_5_models() {
    let agent = Agent::<()>::new("test").with_model("gpt-5");
    assert!(matches!(&agent.model, Some(AgentModel::Name(n)) if n == "gpt-5"));
    let settings = resolved_settings(&agent).await;
    assert_eq!(settings, get_default_model_settings(Some("gpt-5")));
    assert_eq!(
        settings.reasoning.and_then(|r| r.effort).as_deref(),
        Some("low")
    );
}

#[tokio::test]
async fn agent_uses_empty_settings_for_explicit_non_gpt_5_models() {
    let agent = Agent::<()>::new("test").with_model("gpt-4.1");
    assert!(matches!(&agent.model, Some(AgentModel::Name(n)) if n == "gpt-4.1"));
    assert_eq!(resolved_settings(&agent).await, ModelSettings::default());
}

#[tokio::test]
async fn agent_clone_recomputes_implicit_settings_when_model_changes() {
    let agent = Agent::<()>::new("test").with_model("gpt-5");
    let cloned = agent.clone_with(|a| a.model = Some(AgentModel::Name("gpt-5.4-mini".into())));
    assert!(matches!(&cloned.model, Some(AgentModel::Name(n)) if n == "gpt-5.4-mini"));
    let settings = resolved_settings(&cloned).await;
    assert_eq!(settings, get_default_model_settings(Some("gpt-5.4-mini")));
    assert_eq!(
        settings.reasoning.and_then(|r| r.effort).as_deref(),
        Some("none")
    );
}

#[tokio::test]
async fn agent_clone_preserves_explicit_settings_when_model_changes() {
    let explicit = ModelSettings {
        temperature: Some(0.3),
        ..Default::default()
    };
    let agent = Agent::<()>::new("test")
        .with_model("gpt-5")
        .with_model_settings(explicit.clone());
    let cloned = agent.clone_with(|a| a.model = Some(AgentModel::Name("gpt-5.4-mini".into())));
    assert!(matches!(&cloned.model, Some(AgentModel::Name(n)) if n == "gpt-5.4-mini"));
    assert_eq!(resolved_settings(&cloned).await, explicit);
}

#[tokio::test]
async fn agent_resets_model_settings_for_non_gpt_5_models() {
    let _guard = env_lock().await;
    set_env(DEFAULT_MODEL_ENV, "gpt-5");
    let agent = Agent::<()>::new("test").with_model("gpt-4.1");
    let settings = resolved_settings(&agent).await;
    clear_env(DEFAULT_MODEL_ENV);
    assert_eq!(settings, ModelSettings::default());
}

#[tokio::test]
async fn explicit_default_settings_survive_construction_and_clone() {
    let _guard = env_lock().await;
    set_env(DEFAULT_MODEL_ENV, "gpt-5.6-luna");
    let expected = get_default_model_settings(None);
    for model_name in [None, Some("gpt-4o"), Some("gpt-5")] {
        let mut agent = Agent::<()>::new("test").with_model_settings(expected.clone());
        if let Some(name) = model_name {
            agent = agent.with_model(name);
        }
        let cloned = agent.clone_with(|a| a.model = Some(AgentModel::Name("gpt-4.1".into())));
        assert_eq!(resolved_settings(&agent).await, expected, "{model_name:?}");
        assert_eq!(resolved_settings(&cloned).await, expected, "{model_name:?}");
    }
    clear_env(DEFAULT_MODEL_ENV);
}

#[tokio::test]
async fn clone_retains_implicit_defaults_without_copying_public_settings() {
    let agent = Agent::<()>::new("test").with_model("gpt-5");
    let cloned = agent.clone_with(|a| a.name = "clone".into());
    let from_agent = resolved_settings(&agent).await;
    assert_eq!(resolved_settings(&cloned).await, from_agent);
    let regrafted = cloned.clone_with(|a| a.model = Some(AgentModel::Name("gpt-4.1".into())));
    assert_eq!(
        resolved_settings(&regrafted).await,
        ModelSettings::default()
    );
}

#[tokio::test]
async fn clone_explicit_settings_equal_to_defaults_are_not_recomputed() {
    // Python asserts object identity (`is explicit`); Rust has no settings identity, so the
    // portable claim is that the explicit settings are not recomputed for the new model.
    let explicit = get_default_model_settings(Some("gpt-5"));
    let cloned = Agent::<()>::new("test")
        .with_model("gpt-5")
        .with_model_settings(explicit.clone())
        .clone_with(|a| a.model = Some(AgentModel::Name("gpt-4.1".into())));
    assert_eq!(resolved_settings(&cloned).await, explicit);
}

#[tokio::test]
async fn clone_preserves_settings_customized_after_construction() {
    // replacement = false: a field mutated in place on the agent's own settings.
    let mut agent = Agent::<()>::new("test").with_model("gpt-5");
    agent.model_settings_mut().temperature = Some(0.3);
    let cloned = agent.clone_with(|a| a.model = Some(AgentModel::Name("gpt-4.1".into())));
    assert_eq!(resolved_settings(&cloned).await.temperature, Some(0.3));

    // replacement = true: assigning an equal object is still an explicit settings choice.
    let mut agent = Agent::<()>::new("test").with_model("gpt-5");
    *agent.model_settings_mut() = get_default_model_settings(Some("gpt-5"));
    let expected = agent.model_settings().clone();
    let cloned = agent.clone_with(|a| a.model = Some(AgentModel::Name("gpt-4.1".into())));
    assert_eq!(resolved_settings(&cloned).await, expected);
}

#[tokio::test]
async fn explicit_empty_dictionary_settings_are_preserved() {
    let _guard = env_lock().await;
    set_env(DEFAULT_MODEL_ENV, "gpt-4o");
    let agent = Agent::<()>::new("test")
        .with_model("gpt-5")
        .with_model_settings(ModelSettings::default());
    let cloned = agent.clone_with(|a| a.model = Some(AgentModel::Name("gpt-5.6-luna".into())));
    let agent_settings = resolved_settings(&agent).await;
    let cloned_settings = resolved_settings(&cloned).await;
    clear_env(DEFAULT_MODEL_ENV);
    assert_eq!(agent_settings, ModelSettings::default());
    assert_eq!(cloned_settings, ModelSettings::default());
}

// ---------------------------------------------------------------------------
// test_map.py
// ---------------------------------------------------------------------------

#[test]
fn no_prefix_is_openai() {
    let model = MultiProvider::new().get_model(Some("gpt-4o")).unwrap();
    assert_eq!(kind(&model), "OpenAIResponsesModel");
    assert_eq!(model.model_name().as_deref(), Some("gpt-4o"));
}

#[test]
fn openai_prefix_is_openai() {
    let model = MultiProvider::new()
        .get_model(Some("openai/gpt-4o"))
        .unwrap();
    assert_eq!(kind(&model), "OpenAIResponsesModel");
    assert_eq!(model.model_name().as_deref(), Some("gpt-4o"));
}

#[test]
fn litellm_prefix_is_litellm() {
    // Python uses `litellm/foo/bar`; Rust's `LitellmProvider` validates the inner provider name
    // at construction, so the port names a real provider (see the parity report).
    let model = MultiProvider::new()
        .get_model(Some("litellm/openai/gpt-4o"))
        .unwrap();
    assert_eq!(kind(&model), "LitellmModel");
    assert_eq!(model.model_name().as_deref(), Some("openai/gpt-4o"));

    let error = MultiProvider::new()
        .get_model(Some("litellm/foo/bar"))
        .unwrap_err();
    assert!(
        error.to_string().contains("Unknown model provider 'foo'"),
        "{error}"
    );
}

#[test]
fn any_llm_prefix_uses_any_llm_provider() {
    let model = MultiProvider::new()
        .get_model(Some("any-llm/openrouter/openai/gpt-5.4-mini"))
        .unwrap();
    assert_eq!(kind(&model), "AnyLLMModel");
    assert_eq!(
        model.model_name().as_deref(),
        Some("openrouter/openai/gpt-5.4-mini")
    );
}

#[cfg(feature = "websocket")]
#[test]
fn no_prefix_can_use_openai_responses_websocket() {
    let provider = MultiProvider::new()
        .with_openai_provider(OpenAIProvider::new().with_responses_websocket(true));
    let model = provider.get_model(Some("gpt-4o")).unwrap();
    assert_eq!(kind(&model), "OpenAIResponsesWSModel");
}

#[cfg(feature = "websocket")]
#[test]
fn openai_prefix_can_use_openai_responses_websocket() {
    let provider = MultiProvider::new()
        .with_openai_provider(OpenAIProvider::new().with_responses_websocket(true));
    let model = provider.get_model(Some("openai/gpt-4o")).unwrap();
    assert_eq!(kind(&model), "OpenAIResponsesWSModel");
    assert_eq!(model.model_name().as_deref(), Some("gpt-4o"));
}

#[cfg(feature = "websocket")]
#[test]
fn multi_provider_passes_websocket_base_url_to_openai_provider() {
    let provider = MultiProvider::new().with_openai_provider(
        OpenAIProvider::new().with_websocket_base_url("wss://proxy.example.test/v1"),
    );
    assert_eq!(
        provider.openai_provider().websocket_base_url().as_deref(),
        Some("wss://proxy.example.test/v1")
    );
}

#[test]
fn multi_provider_forwards_openai_buffer_streamed_tool_calls_to_chat_model() {
    let openai = OpenAIProvider::new()
        .with_client(OpenAIClient::new("sk-test"))
        .with_use_responses(false)
        .with_buffer_streamed_tool_calls(true);
    assert!(openai.buffer_streamed_tool_calls());

    let provider = MultiProvider::new().with_openai_provider(openai);
    assert!(provider.openai_provider().buffer_streamed_tool_calls());

    let model = provider.get_model(Some("gpt-4o")).unwrap();
    assert_eq!(kind(&model), "OpenAIChatCompletionsModel");
    assert!(model.is_chat_completions());
}

#[test]
fn openai_prefix_defaults_to_alias_mode() {
    let model = MultiProvider::new()
        .get_model(Some("openai/gpt-4o"))
        .unwrap();
    assert_eq!(model.model_name().as_deref(), Some("gpt-4o"));
}

#[test]
fn openai_prefix_can_be_preserved_as_literal_model_id() {
    let model = MultiProvider::new()
        .with_openai_prefix_mode(PrefixMode::ModelId)
        .get_model(Some("openai/gpt-4o"))
        .unwrap();
    assert_eq!(model.model_name().as_deref(), Some("openai/gpt-4o"));
}

#[test]
fn unknown_prefix_defaults_to_error() {
    let error = MultiProvider::new()
        .get_model(Some("openrouter/openai/gpt-4o"))
        .unwrap_err();
    assert!(
        error.to_string().contains("Unknown prefix: openrouter"),
        "{error}"
    );
}

#[test]
fn unknown_prefix_can_be_preserved_for_openai_compatible_model_ids() {
    let model = MultiProvider::new()
        .with_unknown_prefix_mode(PrefixMode::ModelId)
        .get_model(Some("openrouter/openai/gpt-4o"))
        .unwrap();
    assert_eq!(kind(&model), "OpenAIResponsesModel");
    assert_eq!(
        model.model_name().as_deref(),
        Some("openrouter/openai/gpt-4o")
    );
}

/// A provider recording the model names it was asked for and when it was closed.
#[derive(Debug, Default)]
struct CapturingProvider {
    seen: Arc<Mutex<Vec<Option<String>>>>,
    closed: Arc<Mutex<Vec<String>>>,
    name: String,
}

#[async_trait]
impl ModelProvider for CapturingProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        self.seen
            .lock()
            .unwrap()
            .push(model_name.map(str::to_string));
        Ok(Arc::new(RecordingModel {
            name: model_name.unwrap_or_default().to_string(),
            settings: Arc::new(Mutex::new(None)),
        }))
    }

    async fn close(&self) {
        self.closed.lock().unwrap().push(self.name.clone());
    }
}

#[test]
fn provider_map_entries_override_openai_prefix_mode() {
    let custom = Arc::new(CapturingProvider::default());
    let seen = custom.seen.clone();
    let mut map = MultiProviderMap::new();
    map.add_provider("openai", custom);

    let provider = MultiProvider::new()
        .with_provider_map(&map)
        .with_openai_prefix_mode(PrefixMode::ModelId);
    provider.get_model(Some("openai/gpt-4o")).unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![Some("gpt-4o".to_string())]);
}

#[test]
fn provider_map_routes_to_registered_prefix() {
    // Python's `FalseyProvider.__bool__` guards a truthiness test in the lookup; Rust's lookup
    // is an `Option`, so the portable half is that a registered prefix routes with its prefix
    // stripped.
    let custom = Arc::new(CapturingProvider::default());
    let seen = custom.seen.clone();
    let mut map = MultiProviderMap::new();
    map.add_provider("custom", custom);

    let model = MultiProvider::new()
        .with_provider_map(&map)
        .get_model(Some("custom/test-model"))
        .unwrap();

    assert_eq!(model.model_name().as_deref(), Some("test-model"));
    assert_eq!(*seen.lock().unwrap(), vec![Some("test-model".to_string())]);
}

#[tokio::test]
async fn multi_provider_aclose_continues_and_closes_every_provider() {
    // Python also asserts the first `aclose` failure is re-raised; Rust's `ModelProvider::close`
    // is infallible, so only the "every provider is closed" half ports.
    let closed = Arc::new(Mutex::new(Vec::new()));
    let mut map = MultiProviderMap::new();
    for name in ["failing", "later"] {
        map.add_provider(
            name,
            Arc::new(CapturingProvider {
                seen: Arc::new(Mutex::new(Vec::new())),
                closed: closed.clone(),
                name: name.to_string(),
            }),
        );
    }
    let provider = MultiProvider::new().with_provider_map(&map);

    provider.close().await;

    let mut names = closed.lock().unwrap().clone();
    names.sort();
    assert_eq!(names, vec!["failing".to_string(), "later".to_string()]);
}

// ---------------------------------------------------------------------------
// test_agent_registration.py
// ---------------------------------------------------------------------------

/// Records the metadata of every trace that started.
#[derive(Debug, Default)]
struct TraceMetadataRecorder {
    metadata: Mutex<Vec<Option<Map<String, Value>>>>,
}

impl TracingProcessor for TraceMetadataRecorder {
    fn on_trace_start(&self, t: &Trace) {
        self.metadata.lock().unwrap().push(t.metadata().cloned());
    }
    fn on_trace_end(&self, _t: &Trace) {}
    fn on_span_start(&self, _s: &Span) {}
    fn on_span_end(&self, _s: &Span) {}
    fn force_flush(&self) {}
    fn shutdown(&self) {}
}

/// Records every span export.
#[derive(Debug, Default)]
struct SpanExportRecorder {
    exports: Mutex<Vec<Value>>,
}

impl TracingProcessor for SpanExportRecorder {
    fn on_trace_start(&self, _t: &Trace) {}
    fn on_trace_end(&self, _t: &Trace) {}
    fn on_span_start(&self, _s: &Span) {}
    fn on_span_end(&self, s: &Span) {
        if let Some(export) = s.export() {
            self.exports.lock().unwrap().push(export);
        }
    }
    fn force_flush(&self) {}
    fn shutdown(&self) {}
}

/// Run one agent turn with `processor` installed and `provider` resolving the model.
async fn run_traced(
    provider: Arc<dyn ModelProvider>,
    explicit: Option<Map<String, Value>>,
    processor: Arc<dyn TracingProcessor>,
) {
    set_tracing_disabled(false);
    set_trace_processors(vec![processor]);
    let config = RunConfig {
        model_provider: provider,
        trace_metadata: explicit,
        ..Default::default()
    };
    let agent = Agent::<()>::new("test");
    Runner::run_with_options(&agent, "hello", RunOptions::default().run_config(config))
        .await
        .expect("run");
    set_trace_processors(vec![]);
    set_tracing_disabled(true);
}

/// The trace metadata the runner resolved: the Rust equivalent of Python's
/// `resolve_trace_settings(...)[3]`.
async fn trace_metadata(
    provider: Arc<dyn ModelProvider>,
    explicit: Option<Map<String, Value>>,
) -> Option<Map<String, Value>> {
    let recorder = Arc::new(TraceMetadataRecorder::default());
    run_traced(provider, explicit, recorder.clone()).await;
    let recorded = recorder.metadata.lock().unwrap().clone();
    recorded.into_iter().next().flatten()
}

fn harness_provider(harness_id: Option<&str>) -> Arc<dyn ModelProvider> {
    Arc::new(RecordingProvider {
        settings: Arc::new(Mutex::new(None)),
        harness_id: harness_id.map(str::to_string),
    })
}

fn harness_metadata(value: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(
        OPENAI_HARNESS_ID_TRACE_METADATA_KEY.to_string(),
        Value::String(value.into()),
    );
    m
}

#[tokio::test]
async fn agent_registration_config_precedence() {
    let _guard = env_lock().await;
    set_env(HARNESS_ENV, "env-harness");
    set_default_openai_harness(Some("default-harness".into()));
    let metadata = trace_metadata(harness_provider(Some("explicit-harness")), None).await;
    set_default_openai_harness(None);
    clear_env(HARNESS_ENV);
    assert_eq!(metadata, Some(harness_metadata("explicit-harness")));
}

#[tokio::test]
async fn agent_registration_uses_default_before_env() {
    let _guard = env_lock().await;
    set_env(HARNESS_ENV, "env-harness");
    set_default_openai_harness(Some("default-harness".into()));
    let metadata = trace_metadata(harness_provider(None), None).await;
    set_default_openai_harness(None);
    clear_env(HARNESS_ENV);
    assert_eq!(metadata, Some(harness_metadata("default-harness")));
}

#[tokio::test]
async fn agent_registration_uses_env() {
    let _guard = env_lock().await;
    set_env(HARNESS_ENV, "env-harness");
    let metadata = trace_metadata(harness_provider(None), None).await;
    clear_env(HARNESS_ENV);
    assert_eq!(metadata, Some(harness_metadata("env-harness")));
}

#[tokio::test]
async fn set_default_openai_harness_helper() {
    let _guard = env_lock().await;
    set_env(HARNESS_ENV, "env-harness");
    set_default_openai_harness(Some("helper-harness".into()));
    let metadata = trace_metadata(harness_provider(None), None).await;
    set_default_openai_harness(None);
    clear_env(HARNESS_ENV);
    assert_eq!(metadata, Some(harness_metadata("helper-harness")));
}

#[tokio::test]
async fn agent_registration_disabled_without_config() {
    let _guard = env_lock().await;
    clear_env(HARNESS_ENV);
    set_default_openai_harness(None);
    let metadata = trace_metadata(harness_provider(None), None).await;
    assert_eq!(metadata, None);
}

#[test]
fn agent_registration_provider_constructor_config() {
    let openai_provider = OpenAIProvider::new().with_agent_harness_id("provider-harness");
    let multi_provider = MultiProvider::new().with_openai_provider(openai_provider.clone());

    assert_eq!(
        openai_provider.agent_harness_id().as_deref(),
        Some("provider-harness")
    );
    assert_eq!(
        multi_provider
            .openai_provider()
            .agent_harness_id()
            .as_deref(),
        Some("provider-harness")
    );
    assert_eq!(
        multi_provider.agent_harness_id().as_deref(),
        Some("provider-harness")
    );
}

#[test]
fn harness_id_resolves_from_the_model_provider() {
    let provider = OpenAIProvider::new().with_agent_harness_id("private-harness");
    assert_eq!(
        provider.agent_harness_id().as_deref(),
        Some("private-harness")
    );
}

#[tokio::test]
async fn harness_id_is_added_to_trace_metadata() {
    let _guard = env_lock().await;
    clear_env(HARNESS_ENV);
    set_default_openai_harness(None);
    let metadata = trace_metadata(harness_provider(Some("provider-harness")), None).await;
    assert_eq!(metadata, Some(harness_metadata("provider-harness")));
}

#[tokio::test]
async fn harness_id_preserves_explicit_trace_metadata() {
    let _guard = env_lock().await;
    clear_env(HARNESS_ENV);
    set_default_openai_harness(None);
    let mut explicit = harness_metadata("explicit-harness");
    explicit.insert("source".into(), Value::String("test".into()));
    let metadata = trace_metadata(
        harness_provider(Some("provider-harness")),
        Some(explicit.clone()),
    )
    .await;
    assert_eq!(metadata, Some(explicit));
}

#[tokio::test]
async fn env_harness_id_is_added_to_trace_metadata() {
    let _guard = env_lock().await;
    set_env(HARNESS_ENV, "env-harness");
    set_default_openai_harness(None);
    let metadata = trace_metadata(Arc::new(RecordingProvider::default()), None).await;
    clear_env(HARNESS_ENV);
    assert_eq!(metadata, Some(harness_metadata("env-harness")));
}

#[tokio::test]
async fn harness_id_trace_metadata_propagates_to_spans() {
    let _guard = env_lock().await;
    clear_env(HARNESS_ENV);
    set_default_openai_harness(None);
    let recorder = Arc::new(SpanExportRecorder::default());
    run_traced(
        harness_provider(Some("provider-harness")),
        None,
        recorder.clone(),
    )
    .await;

    let exports = recorder.exports.lock().unwrap().clone();
    let agent_span = exports
        .iter()
        .find(|e| e.pointer("/span_data/type").and_then(Value::as_str) == Some("agent"))
        .expect("an agent span was exported");
    assert_eq!(
        agent_span.pointer("/metadata"),
        Some(&Value::Object(harness_metadata("provider-harness")))
    );
}

// ---------------------------------------------------------------------------
// test_openai_client_utils.py (continued below after the generation-metadata ports)
// ---------------------------------------------------------------------------

#[test]
fn official_openai_base_url_matches_exact_host() {
    for base_url in ["https://api.openai.com", "https://api.openai.com/v1/"] {
        assert!(is_official_openai_base_url(base_url, false), "{base_url}");
    }
}

#[test]
fn official_openai_base_url_rejects_non_openai_hosts() {
    for base_url in [
        "https://api.openai.com.evil/v1/",
        "https://api.openai.com.proxy.local/v1/",
        "http://api.openai.com/v1/",
        "https://custom.example.test/v1/",
    ] {
        assert!(!is_official_openai_base_url(base_url, false), "{base_url}");
    }
}

#[test]
fn official_openai_websocket_base_url_matches_exact_host() {
    assert!(is_official_openai_base_url(
        "wss://api.openai.com/v1/",
        true
    ));
    assert!(!is_official_openai_base_url(
        "wss://api.openai.com.proxy.local/v1/",
        true
    ));
}

// ---------------------------------------------------------------------------
// test_generation_metadata.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generation_metadata_respects_sensitive_capture() {
    const SENTINEL: &str = "synthetic-request-metadata-sentinel";
    let _guard = env_lock().await;
    clear_env(HARNESS_ENV);
    set_default_openai_harness(None);

    for include_data in [false, true] {
        let metadata: std::collections::HashMap<String, String> =
            [("request_label".to_string(), SENTINEL.to_string())]
                .into_iter()
                .collect();
        let settings = ModelSettings {
            temperature: Some(0.25),
            metadata: Some(metadata.clone()),
            ..Default::default()
        };
        let mut trace_meta = Map::new();
        trace_meta.insert(
            "metadata".into(),
            Value::String("intentional-trace-label".into()),
        );

        let provider = Arc::new(RecordingProvider::default());
        let recorded_settings = provider.settings.clone();
        let traces = Arc::new(TraceMetadataRecorder::default());
        let spans = Arc::new(SpanExportRecorder::default());

        set_tracing_disabled(false);
        set_trace_processors(vec![traces.clone(), spans.clone()]);
        let config = RunConfig {
            model_provider: provider,
            trace_metadata: Some(trace_meta.clone()),
            trace_include_sensitive_data: include_data,
            ..Default::default()
        };
        let agent = Agent::<()>::new("test").with_model_settings(settings.clone());
        let result =
            Runner::run_with_options(&agent, "hello", RunOptions::default().run_config(config))
                .await
                .expect("run");
        set_trace_processors(vec![]);
        set_tracing_disabled(true);

        assert_eq!(result.final_output_text().as_deref(), Some("ok"));
        // The request carried the caller's metadata, and the caller's objects are untouched.
        assert_eq!(
            recorded_settings.lock().unwrap().as_ref().unwrap().metadata,
            Some(metadata.clone())
        );
        assert_eq!(settings.metadata, Some(metadata));

        let recorded_traces = traces.metadata.lock().unwrap().clone();
        assert_eq!(recorded_traces.len(), 1);
        assert_eq!(recorded_traces[0], Some(trace_meta));

        let exports = spans.exports.lock().unwrap().clone();
        let generations: Vec<&Value> = exports
            .iter()
            .filter(|e| e.pointer("/span_data/type").and_then(Value::as_str) == Some("generation"))
            .collect();
        assert_eq!(generations.len(), 1);
        let model_config = generations[0]
            .pointer("/span_data/model_config")
            .expect("model_config");
        assert_eq!(
            model_config.pointer("/temperature"),
            Some(&Value::from(0.25))
        );
        if include_data {
            assert_eq!(
                model_config.pointer("/metadata/request_label"),
                Some(&Value::from(SENTINEL))
            );
        } else {
            assert!(model_config.get("metadata").is_none(), "{model_config}");
            let dumped = serde_json::to_string(&exports).unwrap();
            assert!(!dumped.contains(SENTINEL), "{dumped}");
        }
    }
}

#[tokio::test]
async fn caller_authored_trace_metadata_remains_explicit() {
    let _guard = env_lock().await;
    clear_env(HARNESS_ENV);
    set_default_openai_harness(None);
    let mut metadata = Map::new();
    metadata.insert(
        "metadata".into(),
        Value::String("caller-owned-label".into()),
    );

    let recorded = trace_metadata(
        Arc::new(RecordingProvider::default()),
        Some(metadata.clone()),
    )
    .await;

    assert_eq!(recorded, Some(metadata));
}

// ---------------------------------------------------------------------------
// test_openai_client_utils.py
// ---------------------------------------------------------------------------

#[test]
fn official_openai_client_rejects_client_without_official_base_url() {
    // Python passes a bare `object()` with no `base_url`; Rust's client always carries one, so
    // the portable equivalent is a client pointed at a non-OpenAI endpoint.
    let client = OpenAIClient::new("sk-test").with_base_url("https://custom.example.test/v1");
    assert!(!client.is_openai_api());
}
