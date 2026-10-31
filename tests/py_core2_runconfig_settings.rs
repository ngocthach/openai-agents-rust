//! Ports of `tests/test_run_config.py`: run configuration defaults, model selection and the
//! model settings the runner resolves for the selected model.
//!
//! Python's `RunConfig` also accepts plain dictionaries and coerces them into the typed config
//! objects; Rust's config is typed at compile time, so the dictionary-only rows are not ported
//! (see the parity report).
//!
//! Every test takes [`env`]: `Agent::new` captures the defaults of `OPENAI_DEFAULT_MODEL` and
//! `RunConfig::default` reads `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA`, both process-wide.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::AgentModel;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::extensions::models::{AnyLLMModel, AnyLLMProvider, LitellmModel};
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::default_models::get_default_model_settings;
use openai_agents_sdk::models::interface::{
    Model, ModelProvider, ModelRequest, ResponseEventStream,
};
use openai_agents_sdk::models::multi_provider::MultiProvider;
use openai_agents_sdk::models::openai_chatcompletions::OpenAIChatCompletionsModel;
use openai_agents_sdk::models::openai_client::OpenAIClient;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{
    OutputGuardrailBlockedMessage, RunConfig, ToolExecutionConfig, ToolNameCollisionPolicy,
    ToolNotFoundBehavior,
};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const DEFAULT_MODEL_ENV: &str = "OPENAI_DEFAULT_MODEL";
const SENSITIVE_ENV: &str = "OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA";

/// Serialize the process environment for this binary and set `OPENAI_DEFAULT_MODEL`.
async fn env(default_model: Option<&str>) -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    set_default_model(default_model);
    guard
}

fn set_default_model(value: Option<&str>) {
    // SAFETY: every test of this binary holds the `env` lock.
    unsafe {
        match value {
            Some(v) => std::env::set_var(DEFAULT_MODEL_ENV, v),
            None => std::env::remove_var(DEFAULT_MODEL_ENV),
        }
    }
}

fn set_var(key: &str, value: Option<&str>) {
    // SAFETY: every test of this binary holds the `env` lock.
    unsafe {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
}

fn effort(effort: &str) -> ModelSettings {
    ModelSettings {
        reasoning: Some(Reasoning::effort(effort)),
        ..Default::default()
    }
}

fn gpt5(effort: &str) -> ModelSettings {
    ModelSettings {
        reasoning: Some(Reasoning::effort(effort)),
        verbosity: Some("low".into()),
        ..Default::default()
    }
}

fn scripted(text: &str) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message(text)])])
            .unwrap(),
    )
}

/// `NamedScriptedModel` (a model reporting a string `model` name) and the plain unnamed
/// `ScriptedModel`, plus a model supplying its own defaults without a name.
#[derive(Debug)]
struct Named {
    name: Option<String>,
    defaults: Option<ModelSettings>,
    inner: Arc<ScriptedModel>,
}

impl Named {
    fn new(name: Option<&str>, text: &str) -> Arc<Self> {
        Arc::new(Named {
            name: name.map(str::to_string),
            defaults: None,
            inner: scripted(text),
        })
    }

    fn settings(&self, index: usize) -> ModelSettings {
        self.inner.calls()[index].request.model_settings.clone()
    }
}

#[async_trait]
impl Model for Named {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.inner.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        self.inner.stream_response(request).await
    }

    fn model_name(&self) -> Option<String> {
        self.name.clone()
    }

    fn default_model_settings(&self) -> Option<ModelSettings> {
        match &self.defaults {
            Some(d) => Some(d.clone()),
            None => self
                .name
                .as_deref()
                .map(|n| get_default_model_settings(Some(n))),
        }
    }
}

/// `DummyProvider`: returns one model and records the name it was asked for.
#[derive(Debug)]
struct DummyProvider {
    last_requested: Mutex<Option<Option<String>>>,
    model: Arc<dyn Model>,
}

impl DummyProvider {
    fn new(model: Arc<dyn Model>) -> Arc<Self> {
        Arc::new(DummyProvider {
            last_requested: Mutex::new(None),
            model,
        })
    }

    /// `None` when never called, `Some(name)` with the requested name otherwise.
    fn last_requested(&self) -> Option<Option<String>> {
        self.last_requested.lock().unwrap().clone()
    }
}

impl ModelProvider for DummyProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        *self.last_requested.lock().unwrap() = Some(model_name.map(str::to_string));
        Ok(self.model.clone())
    }
}

/// A model that keeps an adapter's identity and defaults but answers from a script: the Rust
/// stand-in for `monkeypatch.setattr(adapter, "get_response", scripted.get_response)`.
#[derive(Debug)]
struct Adapted {
    adapter: Arc<dyn Model>,
    scripted: Arc<ScriptedModel>,
}

#[async_trait]
impl Model for Adapted {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.scripted.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        self.scripted.stream_response(request).await
    }

    fn model_name(&self) -> Option<String> {
        self.adapter.model_name()
    }

    fn default_model_settings(&self) -> Option<ModelSettings> {
        self.adapter.default_model_settings()
    }
}

/// Wraps every model `inner` resolves in [`Adapted`].
#[derive(Debug)]
struct AdaptingProvider {
    inner: Arc<dyn ModelProvider>,
    scripted: Arc<ScriptedModel>,
}

impl ModelProvider for AdaptingProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        Ok(Arc::new(Adapted {
            adapter: self.inner.get_model(model_name)?,
            scripted: self.scripted.clone(),
        }))
    }
}

async fn run(agent: &Agent<()>, config: RunConfig<()>, streamed: bool) -> Option<String> {
    let options = RunOptions::default().run_config(config);
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, "hi", options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.expect("stream event");
            }
        }
        result.final_output_text()
    } else {
        Runner::run_with_options(agent, "hi", options)
            .await
            .expect("run")
            .final_output_text()
    }
}

fn provider_config(provider: Arc<dyn ModelProvider>) -> RunConfig<()> {
    RunConfig::default().with_model_provider(provider)
}

// ---------------------------------------------------------------------------
// typed configuration values
// ---------------------------------------------------------------------------

#[cfg(feature = "sandbox")]
#[tokio::test]
async fn sandbox_run_config_rejects_invalid_cwd() {
    use openai_agents_sdk::sandbox::workspace_paths::normalize_sandbox_cwd;
    let _g = env(None).await;
    for (cwd, message) in [
        ("", "sandbox.cwd must be non-empty"),
        (
            "/workspace/tasks/a",
            "sandbox.cwd must be workspace-relative",
        ),
        ("tasks/../a", "sandbox.cwd must not contain parent segments"),
        (r"tasks\a", "sandbox.cwd must use POSIX path separators"),
        // `PureWindowsPath("C:/tasks/a")` reaches the validator in its POSIX spelling.
        ("C:/tasks/a", "sandbox.cwd must be workspace-relative"),
    ] {
        let error = normalize_sandbox_cwd(cwd).expect_err(cwd);
        assert!(error.to_string().contains(message), "{cwd}: {error}");
    }
    assert_eq!(normalize_sandbox_cwd("tasks/a").unwrap(), "tasks/a");
}

#[tokio::test]
async fn run_config_accepts_output_guardrail_blocked_message_customizers() {
    let _g = env(None).await;
    let config = RunConfig::<()>::default().with_output_guardrail_blocked_message(
        OutputGuardrailBlockedMessage::Text("custom".into()),
    );
    assert!(matches!(
        config.output_guardrail_blocked_message,
        Some(OutputGuardrailBlockedMessage::Text(ref t)) if t == "custom"
    ));

    let formatter: openai_agents_sdk::run_config::BlockedMessageFormatter<()> =
        Arc::new(|_args| Some("custom".to_string()));
    let config = RunConfig::<()>::default().with_output_guardrail_blocked_message(
        OutputGuardrailBlockedMessage::Formatter(formatter.clone()),
    );
    match config.output_guardrail_blocked_message {
        Some(OutputGuardrailBlockedMessage::Formatter(f)) => assert!(Arc::ptr_eq(&f, &formatter)),
        other => panic!("unexpected {other:?}"),
    }
}

/// Python rejects `output_guardrail_blocked_message=""` when the config is built; Rust's
/// config is a plain struct, so the run rejects it before the model is called.
#[tokio::test]
async fn run_config_rejects_empty_output_guardrail_blocked_message() {
    let _g = env(None).await;
    let model = scripted("done");
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let config = RunConfig::default()
        .with_output_guardrail_blocked_message(OutputGuardrailBlockedMessage::Text(String::new()));
    let error = Runner::run_with_options(&agent, "hi", RunOptions::default().run_config(config))
        .await
        .expect_err("empty message is rejected");
    assert!(
        error
            .to_string()
            .contains("output_guardrail_blocked_message must be non-empty"),
        "{error}"
    );
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn runner_accepts_run_configuration_with_model_settings() {
    let _g = env(None).await;
    let model = scripted("done");
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let config = RunConfig::default().with_model_settings(ModelSettings {
        temperature: Some(0.0),
        ..Default::default()
    });
    assert_eq!(run(&agent, config, false).await.as_deref(), Some("done"));
    assert_eq!(
        model.calls()[0].request.model_settings.temperature,
        Some(0.0)
    );
}

#[tokio::test]
async fn trace_include_sensitive_data_defaults_to_true_when_env_not_set() {
    let _g = env(None).await;
    set_var(SENSITIVE_ENV, None);
    assert!(RunConfig::<()>::default().trace_include_sensitive_data);
}

#[tokio::test]
async fn trace_include_sensitive_data_follows_env_value() {
    let _g = env(None).await;
    for (value, expected) in [
        ("true", true),
        ("True", true),
        ("1", true),
        ("yes", true),
        ("on", true),
        ("false", false),
        ("False", false),
        ("0", false),
        ("no", false),
        ("off", false),
    ] {
        set_var(SENSITIVE_ENV, Some(value));
        assert_eq!(
            RunConfig::<()>::default().trace_include_sensitive_data,
            expected,
            "{value}"
        );
    }
    set_var(SENSITIVE_ENV, None);
}

#[tokio::test]
async fn trace_include_sensitive_data_explicit_override_takes_precedence() {
    let _g = env(None).await;
    set_var(SENSITIVE_ENV, Some("false"));
    let config = RunConfig::<()>::default().with_trace_include_sensitive_data(true);
    assert!(config.trace_include_sensitive_data);

    set_var(SENSITIVE_ENV, Some("true"));
    let config = RunConfig::<()>::default().with_trace_include_sensitive_data(false);
    assert!(!config.trace_include_sensitive_data);
    set_var(SENSITIVE_ENV, None);
}

/// Python rejects `max_function_tool_concurrency=0` when the config is built; Rust's config is
/// a plain struct, so the run rejects it before the model is called.
#[tokio::test]
async fn tool_execution_config_rejects_invalid_function_tool_concurrency() {
    let _g = env(None).await;
    let model = scripted("done");
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let config = RunConfig::default().with_tool_execution(ToolExecutionConfig {
        max_function_tool_concurrency: Some(0),
        ..Default::default()
    });
    let error = Runner::run_with_options(&agent, "hi", RunOptions::default().run_config(config))
        .await
        .expect_err("zero concurrency is rejected");
    assert!(
        error
            .to_string()
            .contains("tool_execution.max_function_tool_concurrency must be at least 1"),
        "{error}"
    );
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn tool_execution_config_is_public_from_agents_package() {
    let _g = env(None).await;
    let config =
        RunConfig::<()>::default().with_tool_execution(openai_agents_sdk::ToolExecutionConfig {
            max_function_tool_concurrency: Some(2),
            ..Default::default()
        });
    assert_eq!(config.tool_execution.max_function_tool_concurrency, Some(2));
}

#[tokio::test]
async fn tool_not_found_behavior_defaults_to_raise_error() {
    let _g = env(None).await;
    assert_eq!(
        RunConfig::<()>::default().tool_not_found_behavior,
        ToolNotFoundBehavior::RaiseError
    );
}

#[tokio::test]
async fn tool_not_found_behavior_is_public_from_agents_package() {
    let _g = env(None).await;
    let behavior: openai_agents_sdk::ToolNotFoundBehavior =
        openai_agents_sdk::ToolNotFoundBehavior::ReturnErrorToModel;
    let config = RunConfig::<()>::default().with_tool_not_found_behavior(behavior);
    assert_eq!(
        config.tool_not_found_behavior,
        ToolNotFoundBehavior::ReturnErrorToModel
    );
}

#[tokio::test]
async fn tool_name_collision_policy_defaults_to_warn() {
    let _g = env(None).await;
    assert_eq!(
        RunConfig::<()>::default().tool_name_collision_policy,
        ToolNameCollisionPolicy::Warn
    );
}

#[tokio::test]
async fn tool_name_collision_policy_is_public_from_run_module() {
    let _g = env(None).await;
    // `"ToolNameCollisionPolicy" in agents.run.__all__`: the type is reachable from the run
    // configuration module the runner re-exports.
    let policy: openai_agents_sdk::run_config::ToolNameCollisionPolicy =
        ToolNameCollisionPolicy::Error;
    let config = RunConfig::<()>::default().with_tool_name_collision_policy(policy);
    assert_eq!(
        config.tool_name_collision_policy,
        ToolNameCollisionPolicy::Error
    );
}

// ---------------------------------------------------------------------------
// model selection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_provider_on_run_config_is_used_for_agent_model_name() {
    let _g = env(None).await;
    let model = scripted("from-provider");
    let provider = DummyProvider::new(model);
    let agent = Agent::<()>::new("test").with_model("test-model");
    let output = run(&agent, provider_config(provider.clone()), false).await;
    assert_eq!(provider.last_requested(), Some(Some("test-model".into())));
    assert_eq!(output.as_deref(), Some("from-provider"));
}

#[tokio::test]
async fn run_config_model_name_override_takes_precedence() {
    let _g = env(None).await;
    let provider = DummyProvider::new(scripted("override-name"));
    let agent = Agent::<()>::new("test").with_model("agent-model");
    let config = provider_config(provider.clone()).with_model("override-name");
    let output = run(&agent, config, false).await;
    assert_eq!(
        provider.last_requested(),
        Some(Some("override-name".into()))
    );
    assert_eq!(output.as_deref(), Some("override-name"));
}

#[tokio::test]
async fn run_config_model_name_override_uses_model_specific_default_settings() {
    for (model_name, reasoning_effort) in [("gpt-5", "low"), ("gpt-5.6", "none")] {
        let _g = env(Some("gpt-5.4-mini")).await;
        let model = scripted("override-name");
        let provider = DummyProvider::new(model.clone());
        let agent = Agent::<()>::new("test");
        let config = provider_config(provider).with_model(model_name);
        assert_eq!(
            run(&agent, config, false).await.as_deref(),
            Some("override-name")
        );
        let settings = model.calls()[0].request.model_settings.clone();
        assert_eq!(
            settings.reasoning.and_then(|r| r.effort).as_deref(),
            Some(reasoning_effort),
            "{model_name}"
        );
        assert_eq!(settings.verbosity.as_deref(), Some("low"));
    }
}

#[tokio::test]
async fn run_config_model_settings_override_implicit_model_specific_defaults() {
    let _g = env(Some("gpt-5.4-mini")).await;
    let model = scripted("override-name");
    let agent = Agent::<()>::new("test");
    let config = provider_config(DummyProvider::new(model.clone()))
        .with_model("gpt-5")
        .with_model_settings(ModelSettings {
            temperature: Some(0.3),
            ..Default::default()
        });
    assert_eq!(
        run(&agent, config, false).await.as_deref(),
        Some("override-name")
    );
    let settings = model.calls()[0].request.model_settings.clone();
    assert_eq!(
        settings.reasoning.and_then(|r| r.effort).as_deref(),
        Some("low")
    );
    assert_eq!(settings.verbosity.as_deref(), Some("low"));
    assert_eq!(settings.temperature, Some(0.3));
}

#[tokio::test]
async fn provider_chosen_model_settings() {
    type Case = (
        &'static str,
        Option<&'static str>,
        Option<ModelSettings>,
        Option<ModelSettings>,
        ModelSettings,
    );
    let temperature = ModelSettings {
        temperature: Some(0.3),
        ..Default::default()
    };
    let cases: Vec<Case> = vec![
        (
            "non-gpt5",
            Some("other-model"),
            None,
            None,
            ModelSettings::default(),
        ),
        ("gpt5", Some("gpt-5"), None, None, gpt5("low")),
        (
            "customized-agent",
            Some("other-model"),
            Some(temperature.clone()),
            None,
            temperature.clone(),
        ),
        (
            "explicit-empty-agent",
            Some("gpt-5"),
            Some(ModelSettings::default()),
            None,
            ModelSettings::default(),
        ),
        (
            "run-override-matches-old-default",
            Some("other-model"),
            None,
            Some(effort("none")),
            effort("none"),
        ),
        ("unnamed-model", None, None, None, gpt5("none")),
    ];
    for streamed in [false, true] {
        for (id, model_name, agent_settings, run_settings, expected) in cases.clone() {
            let _g = env(Some("gpt-5.6-luna")).await;
            let model = Named::new(model_name, "chosen");
            let provider = DummyProvider::new(model.clone());
            let agent = match agent_settings {
                Some(s) => Agent::<()>::new("test").with_model_settings(s),
                None => Agent::<()>::new("test"),
            };
            let original_settings = agent.model_settings.clone();
            let mut config = provider_config(provider.clone());
            config.model_settings = run_settings;
            assert_eq!(
                run(&agent, config, streamed).await.as_deref(),
                Some("chosen"),
                "{id}"
            );
            assert_eq!(provider.last_requested(), Some(None), "{id}");
            assert_eq!(model.settings(0), expected, "{id} streamed={streamed}");
            assert_eq!(agent.model_settings, original_settings, "{id}");
        }
    }
}

#[tokio::test]
async fn explicit_model_settings_precede_provider_reported_name() {
    for source in ["agent-name", "run-name", "agent-object", "run-object"] {
        let _g = env(Some("gpt-5.6-luna")).await;
        let by_name = source.ends_with("name");
        let model = Named::new(
            Some(if by_name { "other-model" } else { "gpt-5" }),
            "chosen",
        );
        let provider = DummyProvider::new(model.clone());
        let selection = if by_name {
            AgentModel::Name("gpt-5".into())
        } else {
            AgentModel::Instance(model.clone())
        };
        let mut agent = Agent::<()>::new("test");
        let mut config = provider_config(provider);
        if source.starts_with("agent") {
            agent = agent.with_model(selection);
        } else {
            config = config.with_model(selection);
        }
        assert_eq!(
            run(&agent, config, false).await.as_deref(),
            Some("chosen"),
            "{source}"
        );
        let expected = if by_name {
            gpt5("low")
        } else {
            ModelSettings::default()
        };
        assert_eq!(model.settings(0), expected, "{source}");
    }
}

/// A canned Chat Completions server recording request bodies.
async fn chat_completions_server(text: &str) -> (String, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let recorded = bodies.clone();
    let reply = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "other-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    })
    .to_string();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let body = loop {
                let Ok(n) = sock.read(&mut tmp).await else {
                    return;
                };
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                let Some(idx) = text.find("\r\n\r\n") else {
                    continue;
                };
                let len: usize = text[..idx]
                    .lines()
                    .filter_map(|l| l.split_once(':'))
                    .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, v)| v.trim().parse().ok())
                    .unwrap_or(0);
                if buf.len() < idx + 4 + len {
                    continue;
                }
                break serde_json::from_slice::<Value>(&buf[idx + 4..idx + 4 + len])
                    .unwrap_or(Value::Null);
            };
            recorded.lock().unwrap().push(body);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 connection: close\r\n\r\n{}",
                reply.len(),
                reply
            );
            sock.write_all(response.as_bytes()).await.ok();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}/v1"), bodies)
}

#[tokio::test]
async fn provider_chosen_chat_model_omits_gpt5_fields() {
    let _g = env(Some("gpt-5.6-luna")).await;
    let (url, bodies) = chat_completions_server("chosen").await;
    let client = OpenAIClient::new("test-key").with_base_url(url);
    let provider = DummyProvider::new(Arc::new(OpenAIChatCompletionsModel::new(
        "other-model",
        client,
    )));
    let output = run(&Agent::<()>::new("test"), provider_config(provider), false).await;
    assert_eq!(output.as_deref(), Some("chosen"));
    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0]["model"], "other-model");
    assert!(bodies[0].get("reasoning_effort").is_none(), "{}", bodies[0]);
    assert!(bodies[0].get("verbosity").is_none(), "{}", bodies[0]);
}

#[tokio::test]
async fn run_config_model_override_object_takes_precedence() {
    let _g = env(None).await;
    let model = scripted("override-object");
    let agent = Agent::<()>::new("test").with_model("agent-model");
    let config = RunConfig::default().with_model(AgentModel::Instance(model));
    assert_eq!(
        run(&agent, config, false).await.as_deref(),
        Some("override-object")
    );
}

#[tokio::test]
async fn agent_model_object_is_used_when_present() {
    let _g = env(None).await;
    let provider = DummyProvider::new(scripted("unused"));
    let agent = Agent::<()>::new("test").with_model_instance(scripted("from-agent-object"));
    let output = run(&agent, provider_config(provider.clone()), false).await;
    assert_eq!(provider.last_requested(), None);
    assert_eq!(output.as_deref(), Some("from-agent-object"));
}

#[tokio::test]
async fn provider_preserves_explicit_settings_equal_to_environment_defaults() {
    for streamed in [false, true] {
        for default_name in ["gpt-4o", "gpt-5.6-luna"] {
            let _g = env(Some(default_name)).await;
            let explicit = get_default_model_settings(None);
            let agent = Agent::<()>::new("test").with_model_settings(explicit.clone());
            let model = Named::new(
                Some(if default_name == "gpt-4o" {
                    "gpt-5"
                } else {
                    "other-model"
                }),
                "chosen",
            );
            let config = provider_config(DummyProvider::new(model.clone()));
            assert_eq!(
                run(&agent, config, streamed).await.as_deref(),
                Some("chosen")
            );
            assert_eq!(model.settings(0), explicit, "{default_name} {streamed}");
            assert_eq!(agent.model_settings, explicit);
        }
    }
}

#[tokio::test]
async fn any_llm_provider_default_uses_canonical_model_settings() {
    let _g = env(Some("gpt-5.6-luna")).await;
    let any_llm = Arc::new(AnyLLMProvider {
        api_key: Some("sk-test".into()),
        base_url: Some("http://127.0.0.1:9/v1".into()),
        ..Default::default()
    });
    let model = any_llm.get_model(None).unwrap();
    assert_eq!(model.default_model_settings(), Some(gpt5("none")));

    let replies = scripted("chosen");
    let provider = Arc::new(AdaptingProvider {
        inner: any_llm,
        scripted: replies.clone(),
    });
    run(&Agent::<()>::new("test"), provider_config(provider), false).await;
    assert_eq!(replies.calls()[0].request.model_settings, gpt5("none"));
}

#[tokio::test]
async fn provider_model_can_supply_defaults_without_a_string_identifier() {
    let _g = env(None).await;
    let model = Arc::new(Named {
        name: None,
        defaults: Some(ModelSettings {
            temperature: Some(0.2),
            ..Default::default()
        }),
        inner: scripted("chosen"),
    });
    let config =
        provider_config(DummyProvider::new(model.clone())).with_model_settings(ModelSettings {
            top_p: Some(0.5),
            ..Default::default()
        });
    run(&Agent::<()>::new("test"), config, false).await;
    assert_eq!(
        model.settings(0),
        ModelSettings {
            temperature: Some(0.2),
            top_p: Some(0.5),
            ..Default::default()
        }
    );
}

fn routed_cases() -> Vec<(&'static str, ModelSettings)> {
    vec![
        ("openrouter/openai/gpt-5.4-mini", gpt5("none")),
        ("openai/gpt-5", gpt5("low")),
        (
            "openrouter/anthropic/claude-sonnet-4.5",
            ModelSettings::default(),
        ),
        (
            "openrouter/other-vendor/gpt-5.4-mini",
            ModelSettings::default(),
        ),
        ("ollama/gpt-5.4-mini", ModelSettings::default()),
    ]
}

#[tokio::test]
async fn provider_selected_routed_model_defaults() {
    for adapter in ["any-llm", "litellm"] {
        for streamed in [false, true] {
            for (model_name, expected) in routed_cases() {
                let _g = env(Some("gpt-5.6-luna")).await;
                let url = Some("http://127.0.0.1:9/v1");
                let model: Arc<dyn Model> = if adapter == "any-llm" {
                    Arc::new(AnyLLMModel::new(model_name, url, Some("sk-test"), None).unwrap())
                } else {
                    Arc::new(
                        LitellmModel::with_options(model_name, url, Some("sk-test"), None).unwrap(),
                    )
                };
                let replies = scripted("chosen");
                let adapted = Arc::new(Adapted {
                    adapter: model.clone(),
                    scripted: replies.clone(),
                });
                let config = provider_config(DummyProvider::new(adapted));
                assert_eq!(
                    run(&Agent::<()>::new("test"), config, streamed)
                        .await
                        .as_deref(),
                    Some("chosen")
                );
                assert_eq!(
                    replies.calls()[0].request.model_settings,
                    expected,
                    "{adapter} {model_name} streamed={streamed}"
                );
                assert_eq!(model.model_name().as_deref(), Some(model_name));
            }
        }
    }
}

#[tokio::test]
async fn unnamed_provider_model_retains_captured_defaults() {
    for streamed in [false, true] {
        for (initial_model, later_model, expected) in [
            ("gpt-5.4-mini", "gpt-4.1", gpt5("none")),
            ("gpt-4.1", "gpt-5.4-mini", ModelSettings::default()),
        ] {
            let _g = env(Some(initial_model)).await;
            let agent = Agent::<()>::new("test");
            set_default_model(Some(later_model));
            let model = Named::new(None, "chosen");
            let config = provider_config(DummyProvider::new(model.clone()));
            assert_eq!(
                run(&agent, config, streamed).await.as_deref(),
                Some("chosen")
            );
            assert_eq!(
                model.settings(0),
                expected,
                "{initial_model} -> {later_model}"
            );
        }
    }
}

#[tokio::test]
async fn routed_string_defaults_use_resolved_adapter() {
    for streamed in [false, true] {
        for selection in ["agent", "run"] {
            for explicit in [false, true] {
                for (route, route_effort) in [
                    ("any-llm/openrouter/openai/gpt-5.4-mini", "none"),
                    ("litellm/openai/gpt-5", "low"),
                ] {
                    let _g = env(Some("gpt-4.1")).await;
                    set_var("OPENROUTER_API_KEY", Some("sk-test"));
                    set_var("OPENAI_API_KEY", Some("sk-test"));
                    let replies = scripted("chosen");
                    let provider = Arc::new(AdaptingProvider {
                        inner: Arc::new(MultiProvider::default()),
                        scripted: replies.clone(),
                    });
                    let mut agent = Agent::<()>::new("test");
                    if selection == "agent" {
                        agent = agent.with_model(route);
                    }
                    if explicit {
                        agent = agent.with_model_settings(ModelSettings {
                            reasoning: Some(Reasoning::effort("high")),
                            verbosity: Some("medium".into()),
                            ..Default::default()
                        });
                    }
                    let mut config = provider_config(provider).with_model_settings(ModelSettings {
                        temperature: Some(0.25),
                        ..Default::default()
                    });
                    if selection == "run" {
                        config = config.with_model(route);
                    }
                    assert_eq!(
                        run(&agent, config, streamed).await.as_deref(),
                        Some("chosen")
                    );
                    let settings = replies.calls()[0].request.model_settings.clone();
                    let case = format!("{route} {selection} explicit={explicit} {streamed}");
                    assert_eq!(
                        settings.reasoning.and_then(|r| r.effort).as_deref(),
                        Some(if explicit { "high" } else { route_effort }),
                        "{case}"
                    );
                    assert_eq!(
                        settings.verbosity.as_deref(),
                        Some(if explicit { "medium" } else { "low" }),
                        "{case}"
                    );
                    assert_eq!(settings.temperature, Some(0.25), "{case}");
                    set_var("OPENROUTER_API_KEY", None);
                    set_var("OPENAI_API_KEY", None);
                }
            }
        }
    }
}

/// `test_equal_in_place_assignment_preserves_explicit_settings`, `field="reasoning"`.
///
/// Python shares one settings object between an agent and its shallow clone, so the in-place
/// write reaches both; a Rust clone is a deep copy, so the write goes to the agent that runs.
/// The `field="verbosity"` half is not ported: it relies on Python detecting an in-place write
/// of a value equal to the captured default through `ModelSettings.__setattr__`.
#[tokio::test]
async fn in_place_reasoning_assignment_preserves_explicit_settings() {
    for streamed in [false, true] {
        for clone_timing in ["none", "before", "after"] {
            let _g = env(Some("gpt-5.6-luna")).await;
            let original = Agent::<()>::new("test");
            let mut agent = match clone_timing {
                "before" => original.clone_with(|_| {}),
                _ => original,
            };
            agent.model_settings.reasoning = Some(Reasoning::effort("none"));
            if clone_timing == "after" {
                agent = agent.clone_with(|_| {});
            }
            let model = Named::new(Some("gpt-5"), "chosen");
            let config = provider_config(DummyProvider::new(model.clone()));
            assert_eq!(
                run(&agent, config, streamed).await.as_deref(),
                Some("chosen")
            );
            assert_eq!(
                model.settings(0).reasoning,
                Some(Reasoning::effort("none")),
                "{clone_timing} {streamed}"
            );
        }
    }
}
