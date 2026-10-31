//! The streamed row of `test_provider_defaults_are_resolved_before_sandbox_capability_settings`
//! (`tests/sandbox/test_runtime.py`): the provider is asked for its default model once and the
//! capability's request parameters merge with the run config's instead of replacing them, for a
//! streamed run as well as a plain one (the plain run is in `py_sandbox_a_runtime_7.rs`).
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::RunOptions;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::models::ModelProvider;
use openai_agents_sdk::models::interface::Model;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::sandbox::capabilities::{Compaction, CompactionPolicy};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use serde_json::{Value, json};

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect())
            .expect("scripted model"),
    )
}

/// Hands out one fixed model and counts how it was asked.
#[derive(Debug)]
struct FixedProvider {
    model: Arc<ScriptedModel>,
    requests: Mutex<Vec<Option<String>>>,
}

impl ModelProvider for FixedProvider {
    fn get_model(
        &self,
        model_name: Option<&str>,
    ) -> openai_agents_sdk::error::Result<Arc<dyn Model>> {
        self.requests
            .lock()
            .unwrap()
            .push(model_name.map(str::to_string));
        Ok(self.model.clone())
    }
}

#[tokio::test]
async fn a_streamed_run_merges_provider_defaults_and_capability_settings() {
    let model = scripted(vec![vec![assistant_message("done")]]);
    let provider = Arc::new(FixedProvider {
        model: model.clone(),
        requests: Mutex::new(Vec::new()),
    });
    let provider_dyn: Arc<dyn ModelProvider> = provider.clone();
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_default_manifest(Manifest::new())
        .with_capabilities(vec![Arc::new(
            Compaction::new().with_policy(CompactionPolicy::Static { threshold: 123 }),
        )]);
    let run_config = RunConfig::<()> {
        model_provider: provider_dyn,
        model_settings: Some(openai_agents_sdk::model_settings::ModelSettings {
            extra_args: Some(
                json!({"request_option": "override"})
                    .as_object()
                    .cloned()
                    .expect("object"),
            ),
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut run = SandboxRunner::run_streamed_with_options(
        &agent,
        "hello",
        &SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new())),
        RunOptions::default().run_config(run_config),
    )
    .await
    .expect("the stream starts");
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while let Some(event) = stream.next().await {
            event.expect("event");
        }
    }

    assert_eq!(run.streaming.final_output_text().as_deref(), Some("done"));
    assert_eq!(*provider.requests.lock().unwrap(), vec![None]);
    let call = model.first_call().expect("the model was called");
    assert_eq!(
        call.request
            .model_settings
            .extra_args
            .clone()
            .map(Value::Object),
        Some(json!({
            "context_management": [{"type": "compaction", "compact_threshold": 123}],
            "request_option": "override",
        }))
    );
}
