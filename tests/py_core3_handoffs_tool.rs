//! Ports of the remaining cases in the Python SDK's `tests/test_handoff_tool.py`: derived-name
//! collisions and warnings, copied and renamed handoffs, and callable `is_enabled` checks. The
//! cases already in `py_core_b_handoff_tool.rs` are not repeated here.
//!
//! The `log` subscriber is process-global. Each test filters the records it cares about by a
//! needle that no other test in this binary uses, so parallel tests do not interfere.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::handoffs::{self, Handoff};
use openai_agents_sdk::run_config::{RunConfig, ToolNameCollisionPolicy};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::{Agent, RunContext, RunOptions, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

#[derive(Default)]
struct Collect(Mutex<Vec<String>>);

impl log::Log for Collect {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

static SINK: std::sync::OnceLock<Collect> = std::sync::OnceLock::new();

/// Installs the process-wide sink once and returns it.
fn sink() -> &'static Collect {
    let sink: &'static Collect = SINK.get_or_init(Collect::default);
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink
}

/// Records whose text contains `needle`.
fn logged(needle: &str) -> Vec<String> {
    sink()
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|m| m.contains(needle))
        .cloned()
        .collect()
}

fn names(handoffs: &[openai_agents_sdk::models::interface::FunctionToolSpec]) -> Vec<String> {
    handoffs.iter().map(|h| h.name.clone()).collect()
}

fn error_policy() -> RunConfig<()> {
    RunConfig::<()>::default().with_tool_name_collision_policy(ToolNameCollisionPolicy::Error)
}

/// Python: `test_handoff_rejects_distinct_agents_with_the_same_name`. Two distinct agents whose
/// handoff tool names are identical are rejected under the `error` policy.
#[tokio::test]
async fn distinct_agents_with_the_same_name_are_rejected() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let triage = Agent::<()>::new("triage")
        .with_model_instance(model)
        .with_handoff(Arc::new(Agent::<()>::new("Billing")))
        .with_handoff(Arc::new(Agent::<()>::new("Billing")));

    let err = Runner::run_with_options(
        &triage,
        "hi",
        RunOptions::new(()).run_config(error_policy()),
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("handoff tool name `transfer_to_billing` is used by multiple handoffs"),
        "{err}"
    );
}

/// Python: `test_handoff_warns_and_keeps_last_distinct_agent_with_the_same_name`. Under the
/// `warn` policy the collision is logged and the last handoff with the name is the one kept.
#[tokio::test]
async fn warn_policy_keeps_the_last_distinct_agent_with_the_same_name() {
    disable_tracing();
    sink();
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    let triage_model = Arc::new(ScriptedModel::new());
    triage_model
        .enqueue(ScriptedStep::output(vec![function_call(
            "transfer_to_billing",
            "{}",
        )]))
        .unwrap();
    let first_model = Arc::new(ScriptedModel::new());
    let second_model = Arc::new(ScriptedModel::new());
    second_model
        .enqueue(ScriptedStep::output(vec![assistant_message("from second")]))
        .unwrap();
    let first = Arc::new(Agent::<()>::new("Billing").with_model_instance(first_model.clone()));
    let second = Arc::new(Agent::<()>::new("Billing").with_model_instance(second_model.clone()));
    let triage = Agent::<()>::new("triage")
        .with_model_instance(triage_model.clone())
        .with_handoff(first)
        .with_handoff(second.clone());

    let config =
        RunConfig::<()>::default().with_tool_name_collision_policy(ToolNameCollisionPolicy::Warn);
    let result = Runner::run_with_options(&triage, "hi", RunOptions::new(()).run_config(config))
        .await
        .unwrap();

    let request = triage_model.last_call().unwrap().request;
    assert_eq!(names(&request.handoffs), vec!["transfer_to_billing"]);
    assert!(Arc::ptr_eq(&result.last_agent, &second));
    assert_eq!(result.final_output_text().as_deref(), Some("from second"));
    assert!(
        !logged("handoff tool name `transfer_to_billing` is used by multiple handoffs").is_empty()
    );
    assert!(first_model.calls().is_empty());
}

/// Python: `test_handoff_default_identity_tracks_the_current_tool_name`. A copy of a derived
/// handoff still derives the colliding name, while a renamed one no longer collides.
#[tokio::test]
async fn default_identity_tracks_the_current_tool_name() {
    disable_tracing();
    let billing = Arc::new(Agent::<()>::new("Billing Agent"));
    let normalized = Arc::new(Agent::<()>::new("billing agent"));
    let derived = handoff(billing.clone());

    // An unchanged copy keeps the derived name and collides with `billing agent`.
    let copied_model = Arc::new(ScriptedModel::new());
    copied_model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let copied_triage = Agent::<()>::new("copied")
        .with_model_instance(copied_model)
        .with_handoff(derived.clone())
        .with_handoff(normalized.clone());
    let err = Runner::run_with_options(
        &copied_triage,
        "hi",
        RunOptions::new(()).run_config(error_policy()),
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("Ambiguous handoff configuration"),
        "{err}"
    );

    // A copy whose tool name was changed no longer collides.
    let renamed_model = Arc::new(ScriptedModel::new());
    renamed_model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let renamed_triage = Agent::<()>::new("renamed")
        .with_model_instance(renamed_model.clone())
        .with_handoff(derived.with_tool_name("transfer_to_primary_billing"))
        .with_handoff(normalized);
    Runner::run_with_options(
        &renamed_triage,
        "hi",
        RunOptions::new(()).run_config(error_policy()),
    )
    .await
    .unwrap();
    let request = renamed_model.last_call().unwrap().request;
    assert_eq!(
        names(&request.handoffs),
        vec!["transfer_to_primary_billing", "transfer_to_billing_agent"]
    );
}

/// Python: `test_default_handoff_tool_name_warns_for_non_whitespace_invalid_characters`. Only
/// non-whitespace invalid characters are logged; the name is transformed the same way.
#[test]
fn default_handoff_tool_name_warns_for_non_whitespace_invalid_characters() {
    sink();
    let tool_name = handoffs::default_tool_name("Refund/agent");
    assert_eq!(tool_name, "transfer_to_refund_agent");
    let warnings = logged("transfer_to_Refund/agent");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains("contains invalid characters for function calling"),
        "{warnings:?}"
    );
}

/// Python: `test_handoff_is_enabled_callable`. A dynamic `is_enabled` callback is evaluated on
/// every call and its boolean result is used as is.
#[tokio::test]
async fn handoff_is_enabled_callable_returns_its_result() {
    let agent = Arc::new(Agent::<()>::new("test"));
    let ctx = RunContext::new(());

    let enabled = handoff(agent.clone()).with_is_enabled_fn(|_ctx, _name| async { true });
    assert!(enabled.is_enabled.evaluate(&ctx, "test").await);

    let disabled = handoff(agent.clone()).with_is_enabled_fn(|_ctx, _name| async { false });
    assert!(!disabled.is_enabled.evaluate(&ctx, "test").await);

    let flag = Arc::new(AtomicBool::new(true));
    let toggled = {
        let flag = flag.clone();
        handoff(agent).with_is_enabled_fn(move |_ctx, _name| {
            let value = flag.load(Ordering::SeqCst);
            async move { value }
        })
    };
    assert!(toggled.is_enabled.evaluate(&ctx, "test").await);
    flag.store(false, Ordering::SeqCst);
    assert!(!toggled.is_enabled.evaluate(&ctx, "test").await);
}

/// Python: `test_handoff_direct_sync_is_enabled_callable_filters_handoff`. A handoff built
/// directly from its parts, with a disabling `is_enabled`, is filtered out of the enabled list.
#[tokio::test]
async fn directly_built_handoff_with_a_disabling_is_enabled_is_filtered() {
    let handoff_obj = Handoff::custom(
        "transfer_to_target",
        "Transfer to target.",
        "target",
        json!({}),
        |_ctx, _args: String| async { Ok(Arc::new(Agent::<()>::new("target"))) },
    )
    .with_is_enabled_fn(|_ctx, _name| async { false });
    let main = Agent::<()>::new("main").with_handoff(handoff_obj);

    let enabled = main.get_enabled_handoffs(&RunContext::new(())).await;
    assert!(enabled.is_empty());
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Reason {
    #[allow(dead_code)]
    reason: String,
}

/// Python: `test_handoff_rejects_nullable_input_root`. The builder cannot fail, so the run that
/// uses the agent raises the same `UserError` before any model call.
#[tokio::test]
async fn nullable_input_root_is_rejected_before_the_first_model_call() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let transfer = handoff(Arc::new(Agent::<()>::new("target")))
        .with_on_handoff_input(|_ctx, _input: Option<Reason>| async { Ok(()) });
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_handoff(transfer);

    let err = Runner::run_with_options(&agent, "hi", RunOptions::new(()))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("root of a strict JSON schema"),
        "{err}"
    );
    assert!(model.calls().is_empty());
}
