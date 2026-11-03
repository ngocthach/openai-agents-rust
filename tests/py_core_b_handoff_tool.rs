//! Ported from the Python SDK's `tests/test_handoff_tool.py`.
//!
//! Python's `get_handoffs` and `resolve_tool_name_collisions` are module-private in Rust, so the
//! enablement-filtering and collision-policy behaviors are asserted through the public runner
//! (the tool list the model is handed, and the error the run raises) instead.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::handoffs::{self, Handoff, HandoffInputData};
use openai_agents_sdk::items::{InputItem, MessageOutput, RunItem, RunItemKind};
use openai_agents_sdk::models::interface::FunctionToolSpec;
use openai_agents_sdk::run_config::ToolNameCollisionPolicy;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::IsEnabled;
use openai_agents_sdk::{Agent, AgentsError, RunConfig, RunContext, RunOptions, Runner, handoff};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct StrictInput {
    name: String,
    age: i64,
}

fn message_item(content: &str, agent: &str) -> RunItem {
    RunItem::new(
        agent,
        RunItemKind::MessageOutput {
            raw_item: MessageOutput::text(content),
        },
    )
}

fn get_len(data: &HandoffInputData) -> usize {
    data.input_history.len() + data.pre_handoff_items.len() + data.new_items.len()
}

/// Names of the handoff tools offered to the model. Python's `get_handoffs` result is sent as
/// the request's `handoffs` list, which Rust keeps separate from `tools`.
fn handoff_names(handoffs: &[FunctionToolSpec]) -> Vec<String> {
    handoffs.iter().map(|h| h.name.clone()).collect()
}

#[test]
fn default_tool_name_and_description() {
    assert_eq!(handoffs::default_tool_name("test_1"), "transfer_to_test_1");
    assert_eq!(
        handoffs::default_tool_name("Refund agent"),
        "transfer_to_refund_agent"
    );
    // A `/` is not whitespace, so it is also replaced.
    assert_eq!(
        handoffs::default_tool_name("Refund/agent"),
        "transfer_to_refund_agent"
    );
    assert!(
        handoffs::default_tool_description("test_1", None)
            .to_lowercase()
            .contains("test_1")
    );
}

#[test]
fn single_handoff_setup() {
    let agent_1 = Arc::new(Agent::<()>::new("test_1"));
    let agent_2 = Agent::<()>::new("test_2").with_handoff(agent_1.clone());

    assert!(agent_1.handoffs.is_empty());
    assert_eq!(agent_2.handoffs.len(), 1);

    let obj = &agent_2.handoffs[0];
    assert_eq!(obj.tool_name, handoffs::default_tool_name("test_1"));
    assert_eq!(
        obj.tool_description,
        handoffs::default_tool_description("test_1", None)
    );
    assert_eq!(obj.agent_name, "test_1");
}

#[test]
fn multiple_handoffs_setup() {
    let agent_1 = Arc::new(Agent::<()>::new("test_1"));
    let agent_2 = Arc::new(Agent::<()>::new("test_2"));
    let agent_3 = Agent::<()>::new("test_3")
        .with_handoff(agent_1.clone())
        .with_handoff(agent_2.clone());

    assert_eq!(agent_3.handoffs.len(), 2);
    assert_eq!(
        agent_3.handoffs[0].tool_name,
        handoffs::default_tool_name("test_1")
    );
    assert_eq!(
        agent_3.handoffs[1].tool_name,
        handoffs::default_tool_name("test_2")
    );
    assert_eq!(agent_3.handoffs[0].agent_name, "test_1");
    assert_eq!(agent_3.handoffs[1].agent_name, "test_2");
}

#[test]
fn custom_handoff_setup() {
    let agent_1 = Arc::new(Agent::<()>::new("test_1"));
    let agent_2 = Arc::new(Agent::<()>::new("test_2"));
    let agent_3 = Agent::<()>::new("test_3")
        .with_handoff(agent_1.clone())
        .with_handoff(
            handoff(agent_2.clone())
                .with_tool_name("custom_tool_name")
                .with_tool_description("custom tool description"),
        );

    assert_eq!(agent_3.handoffs.len(), 2);
    let first = &agent_3.handoffs[0];
    assert_eq!(first.tool_name, handoffs::default_tool_name("test_1"));
    assert_eq!(
        first.tool_description,
        handoffs::default_tool_description("test_1", None)
    );
    assert_eq!(first.agent_name, "test_1");

    let second = &agent_3.handoffs[1];
    assert_eq!(second.tool_name, "custom_tool_name");
    assert_eq!(second.tool_description, "custom tool description");
    assert_eq!(second.agent_name, "test_2");
}

#[tokio::test]
async fn handoff_input_type_rejects_invalid_json() {
    disable_tracing();
    let agent = Arc::new(Agent::<()>::new("test"));
    let obj =
        handoff(agent.clone()).with_on_handoff_input(|_ctx, _input: Foo| async move { Ok(()) });

    // The schema of the input type becomes the handoff tool's parameters.
    assert_eq!(obj.input_json_schema["type"], "object");
    assert_eq!(obj.input_json_schema["properties"]["bar"]["type"], "string");

    let ctx = RunContext::new(());
    let err = obj
        .invoke(ctx.clone(), "not json".to_string())
        .await
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );

    // Valid JSON transfers to the target agent.
    let invoked = obj
        .invoke(ctx, json!({"bar": "baz"}).to_string())
        .await
        .unwrap();
    assert_eq!(invoked.name, agent.name);
}

#[tokio::test]
async fn on_handoff_called_with_input() {
    disable_tracing();
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = calls.clone();
    let agent = Arc::new(Agent::<()>::new("test"));
    let obj = handoff(agent.clone()).with_on_handoff_input(move |_ctx, input: Foo| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push(input);
            Ok(())
        }
    });

    let invoked = obj
        .invoke(RunContext::new(()), json!({"bar": "baz"}).to_string())
        .await
        .unwrap();
    assert_eq!(invoked.name, agent.name);
    assert_eq!(
        *calls.lock().unwrap(),
        vec![Foo {
            bar: "baz".to_string()
        }]
    );
}

#[tokio::test]
async fn on_handoff_without_input_called() {
    disable_tracing();
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    let agent = Arc::new(Agent::<()>::new("test"));
    let obj = handoff(agent.clone()).with_on_handoff(move |_ctx| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    });

    let invoked = obj
        .invoke(RunContext::new(()), String::new())
        .await
        .unwrap();
    assert_eq!(invoked.name, agent.name);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn handoff_input_schema_is_strict() {
    let agent = Arc::new(Agent::<()>::new("test"));
    let obj = handoff(agent).with_on_handoff_input(|_ctx, _input: Foo| async move { Ok(()) });
    assert!(obj.strict_json_schema, "input schema should be strict");
    assert_eq!(obj.input_json_schema["additionalProperties"], json!(false));
}

#[tokio::test]
async fn handoff_strict_json_rejects_type_coercion() {
    disable_tracing();
    let agent = Arc::new(Agent::<()>::new("test"));
    let obj =
        handoff(agent.clone()).with_on_handoff_input(|_ctx, _i: StrictInput| async move { Ok(()) });
    assert!(obj.strict_json_schema);

    let ctx = RunContext::new(());
    let err = obj
        .invoke(ctx.clone(), r#"{"name": "Alice", "age": "25"}"#.to_string())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Invalid JSON"), "{err}");

    let ok = obj
        .invoke(ctx, r#"{"name": "Alice", "age": 25}"#.to_string())
        .await
        .unwrap();
    assert_eq!(ok.name, agent.name);
}

#[test]
fn get_transfer_message_is_valid_json() {
    let message = Handoff::<()>::get_transfer_message("foo");
    let parsed: serde_json::Value = serde_json::from_str(&message).unwrap();
    assert_eq!(parsed, json!({"assistant": "foo"}));
}

#[test]
fn handoff_input_data_lengths() {
    let data = HandoffInputData {
        input_history: Vec::new(),
        pre_handoff_items: Vec::new(),
        new_items: Vec::new(),
        input_items: None,
    };
    assert_eq!(get_len(&data), 0);

    let data = HandoffInputData {
        input_history: vec![InputItem(json!({"role": "user", "content": "foo"}))],
        pre_handoff_items: Vec::new(),
        new_items: Vec::new(),
        input_items: None,
    };
    assert_eq!(get_len(&data), 1);

    let data = HandoffInputData {
        input_history: vec![
            InputItem(json!({"role": "user", "content": "foo"})),
            InputItem(json!({"role": "assistant", "content": "bar"})),
        ],
        pre_handoff_items: Vec::new(),
        new_items: Vec::new(),
        input_items: None,
    };
    assert_eq!(get_len(&data), 2);

    let data = HandoffInputData {
        input_history: vec![InputItem(json!({"role": "user", "content": "foo"}))],
        pre_handoff_items: vec![message_item("foo", "a"), message_item("foo2", "a")],
        new_items: vec![message_item("bar", "a"), message_item("baz", "a")],
        input_items: None,
    };
    assert_eq!(get_len(&data), 5);

    let data = HandoffInputData {
        input_history: vec![
            InputItem(json!({"role": "user", "content": "foo"})),
            InputItem(json!({"role": "assistant", "content": "bar"})),
        ],
        pre_handoff_items: vec![message_item("baz", "a")],
        new_items: vec![message_item("baz", "a"), message_item("qux", "a")],
        input_items: None,
    };
    assert_eq!(get_len(&data), 5);
}

#[test]
fn handoff_is_enabled_defaults_to_true() {
    let agent = Arc::new(Agent::<()>::new("test"));
    assert!(matches!(
        handoff(agent.clone()).is_enabled,
        IsEnabled::Static(true)
    ));
    assert!(matches!(
        handoff(agent.clone()).with_is_enabled(true).is_enabled,
        IsEnabled::Static(true)
    ));
    assert!(matches!(
        handoff(agent).with_is_enabled(false).is_enabled,
        IsEnabled::Static(false)
    ));
}

/// Python's `test_handoff_is_enabled_filtering_integration`, asserted through the tool list the
/// model is offered.
#[tokio::test]
async fn disabled_handoffs_are_not_offered_to_the_model() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();

    let main_agent = Agent::<()>::new("main_agent")
        .with_model_instance(model.clone())
        .with_handoff(handoff(Arc::new(Agent::<()>::new("agent_1"))).with_is_enabled(true))
        .with_handoff(handoff(Arc::new(Agent::<()>::new("agent_2"))).with_is_enabled(false))
        .with_handoff(
            handoff(Arc::new(Agent::<()>::new("agent_3")))
                .with_is_enabled_fn(|_ctx, _agent| async move { true }),
        );

    Runner::run(&main_agent, "hi").await.unwrap();

    let call = model.last_call().unwrap();
    let names = handoff_names(&call.request.handoffs);
    assert!(
        names.contains(&"transfer_to_agent_1".to_string()),
        "{names:?}"
    );
    assert!(
        names.contains(&"transfer_to_agent_3".to_string()),
        "{names:?}"
    );
    assert!(
        !names.contains(&"transfer_to_agent_2".to_string()),
        "a disabled handoff must not be offered: {names:?}"
    );
}

#[tokio::test]
async fn sync_is_enabled_false_filters_handoff() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let main_agent = Agent::<()>::new("main")
        .with_model_instance(model.clone())
        .with_handoff(
            handoff(Arc::new(Agent::<()>::new("target")))
                .with_is_enabled_fn(|_ctx, _agent| async move { false }),
        );

    Runner::run(&main_agent, "hi").await.unwrap();
    let call = model.last_call().unwrap();
    assert!(
        call.request.handoffs.is_empty(),
        "{:?}",
        call.request.handoffs
    );
}

/// Python's `test_handoffs_reject_colliding_derived_names`: two agents whose names normalize to
/// the same handoff tool name are rejected under the `error` collision policy.
#[tokio::test]
async fn colliding_derived_handoff_names_are_rejected() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model)
        .with_handoff(Arc::new(Agent::<()>::new("Billing Agent")))
        .with_handoff(Arc::new(Agent::<()>::new("billing agent")));

    let config =
        RunConfig::<()>::default().with_tool_name_collision_policy(ToolNameCollisionPolicy::Error);
    let err = Runner::run_with_options(&triage, "hi", RunOptions::new(()).run_config(config))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("transfer_to_billing_agent"),
        "{err}"
    );
}

#[tokio::test]
async fn colliding_derived_handoff_names_allow_an_explicit_override() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_handoff(Arc::new(Agent::<()>::new("Billing Agent")))
        .with_handoff(
            handoff(Arc::new(Agent::<()>::new("billing agent")))
                .with_tool_name("transfer_to_normalized_billing"),
        );

    let config =
        RunConfig::<()>::default().with_tool_name_collision_policy(ToolNameCollisionPolicy::Error);
    Runner::run_with_options(&triage, "hi", RunOptions::new(()).run_config(config))
        .await
        .unwrap();

    let call = model.last_call().unwrap();
    let names = handoff_names(&call.request.handoffs);
    assert_eq!(
        names,
        vec![
            "transfer_to_billing_agent",
            "transfer_to_normalized_billing"
        ]
    );
}

#[tokio::test]
async fn a_disabled_handoff_does_not_collide() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model.clone())
        .with_handoff(Arc::new(Agent::<()>::new("Billing Agent")))
        .with_handoff(handoff(Arc::new(Agent::<()>::new("billing agent"))).with_is_enabled(false));

    let config =
        RunConfig::<()>::default().with_tool_name_collision_policy(ToolNameCollisionPolicy::Error);
    Runner::run_with_options(&triage, "hi", RunOptions::new(()).run_config(config))
        .await
        .unwrap();

    let call = model.last_call().unwrap();
    let names = handoff_names(&call.request.handoffs);
    assert_eq!(names, vec!["transfer_to_billing_agent"]);
}
