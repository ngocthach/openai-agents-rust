//! Port of `test_agent_config.py`: system instructions, handoff resolution, cloning, the final
//! output schema, and dictionary-form model settings.
//!
//! Python's `Agent.__post_init__` type checks (`Agent(name=1)`, `tools="not_a_list"`,
//! `hooks="invalid"`, `tool_use_behavior=123`) and the "no constructor keyword for model
//! settings" checks are rejected by the Rust type system at compile time; the dictionary form of
//! `model_settings` maps to `ModelSettings::from_value`.

use std::sync::Arc;

use openai_agents_sdk::agent::AgentModel;
use openai_agents_sdk::handoffs::{self, handoff};
use openai_agents_sdk::model_settings::ModelSettings;
use openai_agents_sdk::{Agent, AgentsError, RunContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn ctx() -> RunContext<()> {
    RunContext::new(())
}

/// Python: `test_system_instructions` (static, sync callable, async callable, callable object).
#[tokio::test]
async fn system_instructions_resolve_for_static_and_dynamic_forms() {
    let agent = Agent::<()>::new("test").with_instructions("abc123");
    assert_eq!(
        Arc::new(agent.clone())
            .get_system_prompt(&ctx())
            .await
            .unwrap(),
        Some("abc123".to_string())
    );

    let sync = agent
        .clone_with(|a| a.instructions = None)
        .with_dynamic_instructions(|_ctx, _agent| async { Ok("sync_123".to_string()) });
    assert_eq!(
        Arc::new(sync).get_system_prompt(&ctx()).await.unwrap(),
        Some("sync_123".to_string())
    );

    let asynchronous = agent
        .clone_with(|a| a.instructions = None)
        .with_dynamic_instructions(|_ctx, _agent| async { Ok("async_123".to_string()) });
    assert_eq!(
        Arc::new(asynchronous)
            .get_system_prompt(&ctx())
            .await
            .unwrap(),
        Some("async_123".to_string())
    );

    // A callable object is a closure capturing state in Rust.
    let prefix = "async_callable".to_string();
    let callable = agent
        .clone_with(|a| a.instructions = None)
        .with_dynamic_instructions(move |_ctx, _agent| {
            let value = format!("{prefix}_123");
            async move { Ok(value) }
        });
    assert_eq!(
        Arc::new(callable).get_system_prompt(&ctx()).await.unwrap(),
        Some("async_callable_123".to_string())
    );
}

/// Python: `test_handoff_with_agents`.
#[tokio::test]
async fn handoff_targets_given_as_agents_are_resolved() {
    let agent_3 = Agent::<()>::new("agent_3")
        .with_handoffs([Agent::<()>::new("agent_1"), Agent::<()>::new("agent_2")]);
    let resolved = agent_3.get_enabled_handoffs(&ctx()).await;
    assert_eq!(resolved.len(), 2);
    assert_eq!(resolved[0].agent_name, "agent_1");
    assert_eq!(resolved[1].agent_name, "agent_2");
    assert_eq!(
        resolved[0].invoke(ctx(), String::new()).await.unwrap().name,
        "agent_1"
    );
    assert_eq!(
        resolved[1].invoke(ctx(), String::new()).await.unwrap().name,
        "agent_2"
    );
}

/// Python: `test_handoff_with_handoff_obj`.
#[tokio::test]
async fn handoff_objects_keep_their_overrides() {
    let agent_1 = Agent::<()>::new("agent_1");
    let agent_2 = Agent::<()>::new("agent_2");
    let agent_3 = Agent::<()>::new("agent_3").with_handoffs([
        handoff(agent_1),
        handoff(agent_2)
            .with_tool_name("transfer_to_2")
            .with_tool_description("description_2"),
    ]);
    let resolved = agent_3.get_enabled_handoffs(&ctx()).await;
    assert_eq!(resolved.len(), 2);
    assert_eq!(resolved[0].agent_name, "agent_1");
    assert_eq!(resolved[1].agent_name, "agent_2");
    assert_eq!(
        resolved[0].tool_name,
        handoffs::default_tool_name("agent_1")
    );
    assert_eq!(resolved[1].tool_name, "transfer_to_2");
    assert_eq!(
        resolved[0].tool_description,
        handoffs::default_tool_description("agent_1", None)
    );
    assert_eq!(resolved[1].tool_description, "description_2");
    assert_eq!(
        resolved[0].invoke(ctx(), String::new()).await.unwrap().name,
        "agent_1"
    );
    assert_eq!(
        resolved[1].invoke(ctx(), String::new()).await.unwrap().name,
        "agent_2"
    );
}

/// Python: `test_handoff_with_handoff_obj_and_agent`.
#[tokio::test]
async fn handoff_objects_and_agents_can_be_mixed() {
    let agent_3 = Agent::<()>::new("agent_3")
        .with_handoff(handoff(Agent::<()>::new("agent_1")))
        .with_handoff(Agent::<()>::new("agent_2"));
    let resolved = agent_3.get_enabled_handoffs(&ctx()).await;
    assert_eq!(resolved.len(), 2);
    for (index, name) in ["agent_1", "agent_2"].into_iter().enumerate() {
        assert_eq!(resolved[index].agent_name, name);
        assert_eq!(resolved[index].tool_name, handoffs::default_tool_name(name));
        assert_eq!(
            resolved[index].tool_description,
            handoffs::default_tool_description(name, None)
        );
        assert_eq!(
            resolved[index]
                .invoke(ctx(), String::new())
                .await
                .unwrap()
                .name,
            name
        );
    }
}

/// Python: `test_agent_cloning`.
#[test]
fn cloning_overrides_only_the_given_fields() {
    let agent = Agent::<()>::new("test")
        .with_handoff_description("test_description")
        .with_model("o3-mini");
    let cloned = agent.clone_with(|a| {
        a.handoff_description = Some("new_description".to_string());
        a.model = Some(AgentModel::from("o1"));
    });
    assert_eq!(cloned.name, "test");
    assert_eq!(
        cloned.handoff_description.as_deref(),
        Some("new_description")
    );
    assert!(matches!(&cloned.model, Some(AgentModel::Name(name)) if name == "o1"));
    // The original is untouched.
    assert_eq!(
        agent.handoff_description.as_deref(),
        Some("test_description")
    );
    assert!(matches!(&agent.model, Some(AgentModel::Name(name)) if name == "o3-mini"));
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Foo {
    bar: String,
}

/// Python: `test_agent_final_output`.
#[test]
fn a_structured_output_type_gets_a_strict_schema() {
    let agent = Agent::<()>::new("test").with_output_type::<Foo>();
    let schema = agent.output_type.as_ref().expect("an output schema");
    assert!(schema.is_strict_json_schema());
    assert!(schema.json_schema().is_ok());
    assert!(!schema.is_plain_text());
}

/// Python: `TestAgentValidation::test_model_settings_validation`.
#[test]
fn model_settings_accept_typed_values_and_dictionaries_only() {
    assert_eq!(
        ModelSettings::from_value(json!({})).unwrap(),
        ModelSettings::default()
    );
    let settings = ModelSettings::from_value(json!({"temperature": 0.25})).unwrap();
    assert_eq!(settings.temperature, Some(0.25));

    let err = ModelSettings::from_value(json!("invalid")).unwrap_err();
    assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
    assert!(
        err.to_string()
            .contains("ModelSettings must be a ModelSettings instance or a dict"),
        "{err}"
    );
}

/// Python: `test_agent_model_settings_dictionary_preserves_openai_reasoning_extensions`.
#[test]
fn dictionary_model_settings_preserve_reasoning_extensions() {
    let settings = ModelSettings::from_value(json!({
        "reasoning": {"context": "all_turns", "future_reasoning_option": "enabled"},
        "context_management": [{"type": "compaction", "compact_threshold": 244800}],
        "retry": {"max_retries": 0, "backoff": {"jitter": false}},
    }))
    .unwrap();

    let reasoning = settings.reasoning.as_ref().unwrap();
    assert_eq!(reasoning.context.as_deref(), Some("all_turns"));
    assert_eq!(
        serde_json::Value::Object(reasoning.extra.clone()),
        json!({"future_reasoning_option": "enabled"})
    );
    // The extension survives serialization onto the wire.
    assert_eq!(
        serde_json::to_value(reasoning).unwrap(),
        json!({"context": "all_turns", "future_reasoning_option": "enabled"})
    );
    assert_eq!(
        settings.context_management,
        Some(vec![
            json!({"type": "compaction", "compact_threshold": 244800})
        ])
    );
    let retry = settings.retry.as_ref().unwrap();
    assert_eq!(retry.max_retries, Some(0));
    let backoff = retry.backoff.as_ref().unwrap();
    assert_eq!(backoff.jitter, Some(false));
    // Omitted backoff fields stay unset and use the documented defaults when the delay is computed.
    assert_eq!(backoff.initial_delay, None);
}

/// Python: `test_agent_rejects_unknown_first_party_dictionary_model_settings` (4 cases).
#[test]
fn dictionary_model_settings_reject_unknown_keys() {
    let cases = [
        (
            json!({"temperatur": 0.2}),
            "Unknown model settings: temperatur",
        ),
        (
            json!({"retry": {"max_retry": 2}}),
            "Unknown model settings in retry: max_retry",
        ),
        (
            json!({"retry": {"backoff": {"initial_delai": 1}}}),
            "Unknown model settings in retry.backoff: initial_delai",
        ),
        (
            json!({"context_management": [
                {"type": "compaction", "compact_threshold_typo": 1}
            ]}),
            "Unknown model settings in context_management[0]: compact_threshold_typo",
        ),
    ];
    for (settings, message) in cases {
        let err = ModelSettings::from_value(settings).unwrap_err();
        assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
        assert!(err.to_string().contains(message), "{err}");
    }
}
