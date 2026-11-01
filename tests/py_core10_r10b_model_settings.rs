//! `tests/test_run_config.py::test_equal_in_place_assignment_preserves_explicit_settings`.
//!
//! Python's `ModelSettings.__setattr__` marks an agent's SDK-default settings as touched on any
//! write, even of a value equal to the default, so the defaults captured at construction stay
//! in force instead of following the model the run selects. The Rust analogue is
//! `Agent::model_settings_mut`, the tracked way to edit settings in place.
//!
//! Python shares one settings object between an agent and its shallow clone, so an edit made
//! through the original reaches the clone; a Rust clone is an independent copy, so the
//! `before` timing edits the clone that runs (the clone of an untouched agent must still track
//! the edit) and `after` clones the edited agent.
//!
//! `Agent::new` captures the defaults of `OPENAI_DEFAULT_MODEL`, a process-wide variable, so
//! every test takes the [`env`] lock.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::default_models::get_default_model_settings;
use openai_agents_sdk::models::interface::{
    Model, ModelProvider, ModelRequest, ResponseEventStream,
};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, Runner};

async fn env(default_model: &str) -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    // SAFETY: every test of this binary holds the `env` lock.
    unsafe { std::env::set_var("OPENAI_DEFAULT_MODEL", default_model) };
    guard
}

/// A scripted model reporting the name `gpt-5` and that model's defaults.
#[derive(Debug)]
struct Named {
    inner: ScriptedModel,
}

impl Named {
    fn new() -> Arc<Self> {
        Arc::new(Named {
            inner: ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message(
                "chosen",
            )])])
            .unwrap(),
        })
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
        Some("gpt-5".into())
    }

    fn default_model_settings(&self) -> Option<ModelSettings> {
        Some(get_default_model_settings(Some("gpt-5")))
    }
}

#[derive(Debug)]
struct Provider(Arc<dyn Model>);

impl ModelProvider for Provider {
    fn get_model(&self, _name: Option<&str>) -> Result<Arc<dyn Model>> {
        Ok(self.0.clone())
    }
}

async fn run(agent: &Agent<()>, model: &Arc<Named>, streamed: bool) -> ModelSettings {
    let config = RunConfig::default().with_model_provider(Arc::new(Provider(model.clone())));
    let options = RunOptions::default().run_config(config);
    if streamed {
        let mut result = Runner::run_streamed_with_options(agent, "hi", options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.expect("stream event");
            }
        }
        assert_eq!(result.final_output_text().as_deref(), Some("chosen"));
    } else {
        let result = Runner::run_with_options(agent, "hi", options)
            .await
            .expect("run");
        assert_eq!(result.final_output_text().as_deref(), Some("chosen"));
    }
    model.inner.calls()[0].request.model_settings.clone()
}

/// `field="reasoning"` and `field="verbosity"`, `clone_timing` none/before/after, direct and
/// streamed: the model selected at run time is `gpt-5` (reasoning `low`), but the agent was
/// created under `gpt-5.6-luna` (reasoning `none`), and an in-place edit, equal to the default
/// or not, keeps the captured defaults.
#[tokio::test]
async fn an_in_place_edit_preserves_the_captured_defaults() {
    for streamed in [false, true] {
        for clone_timing in ["none", "before", "after"] {
            for field in ["reasoning", "verbosity"] {
                let _g = env("gpt-5.6-luna").await;
                let mut agent = Agent::<()>::new("test");
                if clone_timing == "before" {
                    agent = agent.clone_with(|_| {});
                }
                match field {
                    "reasoning" => {
                        agent.model_settings_mut().reasoning = Some(Reasoning::effort("none"))
                    }
                    _ => agent.model_settings_mut().verbosity = Some("low".into()),
                }
                if clone_timing == "after" {
                    agent = agent.clone_with(|_| {});
                }
                let settings = run(&agent, &Named::new(), streamed).await;
                let label = format!("{field} {clone_timing} streamed={streamed}");
                assert_eq!(
                    settings.reasoning,
                    Some(Reasoning::effort("none")),
                    "{label}"
                );
                assert_eq!(settings.verbosity.as_deref(), Some("low"), "{label}");
            }
        }
    }
}

/// Without an edit the settings keep following the selected model, and a direct field
/// assignment is not tracked (unset fields still follow the selected model).
#[tokio::test]
async fn untouched_and_directly_assigned_settings_follow_the_selected_model() {
    let _g = env("gpt-5.6-luna").await;
    let untouched = Agent::<()>::new("test");
    let settings = run(&untouched, &Named::new(), false).await;
    assert_eq!(settings, get_default_model_settings(Some("gpt-5")));

    let mut assigned = Agent::<()>::new("test");
    assigned.model_settings.verbosity = Some("low".into());
    let settings = run(&assigned, &Named::new(), false).await;
    assert_eq!(
        settings.reasoning,
        get_default_model_settings(Some("gpt-5")).reasoning
    );
    assert_eq!(settings.verbosity.as_deref(), Some("low"));
}
