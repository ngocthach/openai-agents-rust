//! Ports of Python tests the earlier rounds filed as python-only although Rust has the behaviour:
//!
//! * `test_agent_instructions_signature.py`: static, `None`, sync-shaped and async-shaped dynamic
//!   instructions resolve through `Agent::get_system_prompt`, and a stateful callable runs once.
//!   The arity checks (`must accept exactly 2 arguments`) and the non-callable refusal are
//!   compile-time facts in Rust: a dynamic instruction function has one fixed signature.
//! * `test_agent_memory_leak.py`: an agent is not kept alive by a finished run (`Arc`/`Weak`
//!   instead of `weakref` and `gc`).
//! * `test_function_schema.py::test_schema_with_mapping_raises_strict_mode_error`: a mapping
//!   parameter cannot be made strict.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, Result, RunContext, Runner};
use serde_json::json;

fn ctx() -> RunContext<()> {
    RunContext::new(())
}

/// Python: `test_valid_async_signature_passes`.
#[tokio::test]
async fn an_async_dynamic_instruction_function_is_awaited() {
    let agent = Arc::new(
        Agent::<()>::new("test_agent").with_dynamic_instructions(|_ctx, _agent| async {
            Ok("Valid async instructions".into())
        }),
    );
    assert_eq!(
        agent.get_system_prompt(&ctx()).await.unwrap().as_deref(),
        Some("Valid async instructions")
    );
}

/// Python: `test_valid_sync_signature_passes`, `test_sync_callable_object_still_works`: a
/// dynamic function that does no awaiting resolves the same way.
#[tokio::test]
async fn a_dynamic_instruction_function_without_awaiting_resolves() {
    let agent = Arc::new(Agent::<()>::new("test_agent").with_dynamic_instructions(
        |_ctx, _agent| std::future::ready(Ok("Valid sync instructions".into())),
    ));
    assert_eq!(
        agent.get_system_prompt(&ctx()).await.unwrap().as_deref(),
        Some("Valid sync instructions")
    );

    let prefix = String::from("Valid sync callable");
    let agent = Arc::new(Agent::<()>::new("test_agent").with_dynamic_instructions(
        move |_ctx, _agent| std::future::ready(Ok(format!("{prefix} instructions"))),
    ));
    assert_eq!(
        agent.get_system_prompt(&ctx()).await.unwrap().as_deref(),
        Some("Valid sync callable instructions")
    );
}

/// Python: `test_async_callable_object_is_awaited`: a stateful async callable is awaited (not
/// handed back as an unawaited coroutine) and runs exactly once per resolution.
#[tokio::test]
async fn a_stateful_async_instruction_callable_runs_once_per_resolution() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let agent = Arc::new(Agent::<()>::new("test_agent").with_dynamic_instructions(
        move |_ctx, _agent| {
            counter.fetch_add(1, Ordering::SeqCst);
            async { Ok("Valid async callable instructions".to_string()) }
        },
    ));
    assert_eq!(
        agent.get_system_prompt(&ctx()).await.unwrap().as_deref(),
        Some("Valid async callable instructions")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Python: `test_string_instructions_still_work`, `test_none_instructions_return_none`.
#[tokio::test]
async fn static_and_missing_instructions_resolve() {
    let agent =
        Arc::new(Agent::<()>::new("test_agent").with_instructions("Static string instructions"));
    assert_eq!(
        agent.get_system_prompt(&ctx()).await.unwrap().as_deref(),
        Some("Static string instructions")
    );
    let agent = Arc::new(Agent::<()>::new("test_agent"));
    assert_eq!(agent.get_system_prompt(&ctx()).await.unwrap(), None);
}

/// Python: `test_agent_is_released_after_run`.
#[tokio::test]
async fn a_finished_run_does_not_keep_the_agent_alive() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("Paris")]);
    let agent = Arc::new(
        Agent::<()>::new("leak-test-agent")
            .with_instructions("Answer questions.")
            .with_model_instance(model),
    );
    let weak = Arc::downgrade(&agent);

    let result = Runner::run(agent.clone(), "What is the capital of France?")
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Paris"));
    drop(result);

    assert_eq!(Arc::strong_count(&agent), 1, "the run kept a reference");
    drop(agent);
    assert!(weak.upgrade().is_none());
}

async fn noop(_ctx: ToolContext<()>, args: String) -> Result<String> {
    Ok(args)
}

/// Python: `test_schema_with_mapping_raises_strict_mode_error`.
#[test]
fn a_mapping_parameter_cannot_be_made_strict() {
    let schema = json!({
        "type": "object",
        "properties": {
            "test_one": {"type": "object", "additionalProperties": {"type": "integer"}}
        },
        "required": ["test_one"],
    });
    let strict =
        FunctionTool::<()>::try_new_raw("func_with_mapping", "", schema.clone(), true, noop);
    assert!(
        matches!(strict, Err(AgentsError::UserError { .. })),
        "{strict:?}"
    );
    // Without strict mode the schema is kept as given.
    let relaxed =
        FunctionTool::<()>::try_new_raw("func_with_mapping", "", schema.clone(), false, noop)
            .expect("non-strict keeps the schema");
    assert_eq!(relaxed.params_json_schema, schema);
}
