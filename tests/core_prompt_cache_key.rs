//! The SDK-generated prompt cache key and the run grouping it is derived from.

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::memory::InMemorySession;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, RunConfig, RunOptions, RunState, Runner, Session, resolve_run_grouping_id,
};
use serde_json::{Value, json};

/// A model that reports official-OpenAI support, so the runner adds a generated key.
#[derive(Debug)]
struct CachingModel(Arc<FakeModel>);

#[async_trait]
impl Model for CachingModel {
    fn supports_default_prompt_cache_key(&self) -> bool {
        true
    }

    async fn get_response(
        &self,
        request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        self.0.get_response(request).await
    }

    async fn stream_response(
        &self,
        request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        self.0.stream_response(request).await
    }
}

fn prompt_cache_key(fake: &FakeModel, index: usize) -> Option<Value> {
    fake.calls()[index]
        .model_settings
        .extra_args
        .as_ref()?
        .get("prompt_cache_key")
        .cloned()
}

fn ping_tool() -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args("ping", "Ping", |_c| async { Ok("pong") })
}

#[tokio::test]
async fn a_conversation_run_gets_one_hashed_key_for_every_turn() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = Arc::new(FakeModel::new());
    fake.add_turn(vec![function_call("ping", "{}")]);
    fake.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("A")
        .with_tool(ping_tool())
        .with_model_instance(Arc::new(CachingModel(fake.clone())));

    Runner::run_with_options(
        &agent,
        "hi",
        RunOptions::new(()).conversation_id("conv_cache"),
    )
    .await
    .unwrap();

    let first = prompt_cache_key(&fake, 0).expect("a key was generated");
    let key = first.as_str().unwrap();
    assert!(
        key.starts_with("agents-sdk:conversation:"),
        "conversation grouping wins: {key}"
    );
    let digest = key.rsplit(':').next().unwrap();
    assert_eq!(digest.len(), 32);
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(
        !key.contains("conv_cache"),
        "the grouping value is hashed, never sent verbatim: {key}"
    );
    assert_eq!(
        prompt_cache_key(&fake, 1),
        Some(first),
        "every turn of the run reuses one key"
    );
}

#[tokio::test]
async fn a_session_run_and_a_group_run_get_their_own_keys() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);

    let fake = Arc::new(FakeModel::new());
    fake.add_turn(vec![assistant_message("one")]);
    let agent = Agent::<()>::new("A").with_model_instance(Arc::new(CachingModel(fake.clone())));
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("sess_cache"));
    Runner::run_with_options(&agent, "hi", RunOptions::new(()).session(session))
        .await
        .unwrap();
    let session_key = prompt_cache_key(&fake, 0).unwrap();
    assert!(
        session_key
            .as_str()
            .unwrap()
            .starts_with("agents-sdk:session:")
    );

    let fake = Arc::new(FakeModel::new());
    fake.add_turn(vec![assistant_message("two")]);
    let agent = Agent::<()>::new("A").with_model_instance(Arc::new(CachingModel(fake.clone())));
    Runner::run_with_options(
        &agent,
        "hi",
        RunOptions::new(()).run_config(RunConfig::new().with_group_id("grp_cache")),
    )
    .await
    .unwrap();
    let group_key = prompt_cache_key(&fake, 0).unwrap();
    assert!(group_key.as_str().unwrap().starts_with("agents-sdk:group:"));
    assert_ne!(session_key, group_key);
}

#[tokio::test]
async fn an_ungrouped_run_keeps_its_key_to_itself() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = Arc::new(FakeModel::new());
    fake.add_turn(vec![function_call("ping", "{}")]);
    fake.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("A")
        .with_tool(ping_tool())
        .with_model_instance(Arc::new(CachingModel(fake.clone())));
    Runner::run(&agent, "hi").await.unwrap();

    let key = prompt_cache_key(&fake, 0).unwrap();
    let key = key.as_str().unwrap();
    assert!(key.starts_with("agents-sdk:run:"), "{key}");
    assert_eq!(prompt_cache_key(&fake, 1).unwrap(), json!(key));

    // A second ungrouped run must not be treated as the same cache group.
    let other = Arc::new(FakeModel::new());
    other.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("A").with_model_instance(Arc::new(CachingModel(other.clone())));
    Runner::run(&agent, "hi").await.unwrap();
    assert_ne!(prompt_cache_key(&other, 0).unwrap(), json!(key));
}

#[tokio::test]
async fn a_model_without_support_and_a_caller_supplied_key_are_both_left_alone() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);

    // A plain FakeModel reports no support, like a third-party endpoint.
    let fake = Arc::new(FakeModel::new());
    fake.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("A").with_model_instance(fake.clone());
    Runner::run_with_options(&agent, "hi", RunOptions::new(()).conversation_id("conv"))
        .await
        .unwrap();
    assert_eq!(prompt_cache_key(&fake, 0), None);

    // A caller-supplied key is never replaced.
    let fake = Arc::new(FakeModel::new());
    fake.add_turn(vec![assistant_message("done")]);
    let mut settings = openai_agents_sdk::ModelSettings::default();
    settings
        .extra_args
        .get_or_insert_with(Default::default)
        .insert("prompt_cache_key".into(), json!("mine"));
    let agent = Agent::<()>::new("A")
        .with_model_settings(settings)
        .with_model_instance(Arc::new(CachingModel(fake.clone())));
    Runner::run_with_options(&agent, "hi", RunOptions::new(()).conversation_id("conv"))
        .await
        .unwrap();
    assert_eq!(prompt_cache_key(&fake, 0), Some(json!("mine")));
}

#[tokio::test]
async fn the_generated_key_survives_a_run_state_round_trip() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let fake = Arc::new(FakeModel::new());
    fake.add_turn(vec![function_call("gated", "{}")]);
    let tool = FunctionTool::<()>::new_no_args("gated", "Gated", |_c| async { Ok("ok") })
        .with_needs_approval(true);
    let agent = Agent::<()>::new("A")
        .with_tool(tool)
        .with_model_instance(Arc::new(CachingModel(fake.clone())));

    let paused = Runner::run_with_options(
        &agent,
        "hi",
        RunOptions::new(()).conversation_id("conv_resume"),
    )
    .await
    .unwrap();
    assert_eq!(paused.interruptions.len(), 1);
    let first_key = prompt_cache_key(&fake, 0).unwrap();

    let mut state = RunState::from_json(paused.to_state().to_json().unwrap()).unwrap();
    assert_eq!(
        state.generated_prompt_cache_key.as_deref(),
        first_key.as_str(),
        "the key is persisted in the snapshot"
    );
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);

    fake.add_turn(vec![assistant_message("done")]);
    Runner::run_with_options(
        &agent,
        state,
        RunOptions::new(()).conversation_id("conv_resume"),
    )
    .await
    .unwrap();
    assert_eq!(
        prompt_cache_key(&fake, 1),
        Some(first_key),
        "a resumed run stays in the same cache group"
    );
}

#[test]
fn run_grouping_id_follows_the_same_precedence() {
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("sess"));
    assert_eq!(
        resolve_run_grouping_id(Some("conv"), Some(&session), Some("grp")),
        "conv"
    );
    assert_eq!(
        resolve_run_grouping_id(None, Some(&session), Some("grp")),
        "sess"
    );
    assert_eq!(resolve_run_grouping_id(None, None, Some("grp")), "grp");
    let generated = resolve_run_grouping_id(None, None, None);
    assert!(generated.starts_with("run-"), "{generated}");
}
