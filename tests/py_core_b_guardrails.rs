//! Ported from the Python SDK's `tests/test_guardrails.py`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, InputGuardrail, OutputGuardrail, RunOptions, Runner};
use serde_json::json;
use tokio::sync::Notify;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn agent_only() -> Arc<Agent<()>> {
    Arc::new(Agent::<()>::new("test"))
}

/// Mirrors `get_sync_guardrail` / `get_async_input_guardrail`.
fn input_guardrail(
    name: &str,
    triggers: bool,
    output_info: serde_json::Value,
) -> InputGuardrail<()> {
    InputGuardrail::new(name, move |_ctx, _agent, _input| {
        let output_info = output_info.clone();
        async move { Ok(GuardrailFunctionOutput::new(output_info, triggers)) }
    })
}

fn output_guardrail(
    name: &str,
    triggers: bool,
    output_info: serde_json::Value,
) -> OutputGuardrail<()> {
    OutputGuardrail::new(name, move |_ctx, _agent, _out| {
        let output_info = output_info.clone();
        async move { Ok(GuardrailFunctionOutput::new(output_info, triggers)) }
    })
}

fn model_with(items: Vec<openai_agents_sdk::items::OutputItem>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model.enqueue(ScriptedStep::output(items)).unwrap();
    model
}

fn dangerous_tool(flag: Arc<AtomicBool>) -> FunctionTool<()> {
    FunctionTool::new("dangerous_tool", "", move |_ctx, _a: serde_json::Value| {
        let flag = flag.clone();
        async move {
            flag.store(true, Ordering::SeqCst);
            Ok("tool_executed".to_string())
        }
    })
}

// --- direct guardrail invocation ---

#[tokio::test]
async fn input_guardrail_runs_and_reports_output_info() {
    disable_tracing();
    let ctx = openai_agents_sdk::RunContext::new(());

    let g = input_guardrail("g", false, serde_json::Value::Null);
    let result = g.run(ctx.clone(), agent_only(), Vec::new()).await.unwrap();
    assert!(!result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, serde_json::Value::Null);

    let g = input_guardrail("g", true, serde_json::Value::Null);
    let result = g.run(ctx.clone(), agent_only(), Vec::new()).await.unwrap();
    assert!(result.output.tripwire_triggered);

    let g = input_guardrail("g", true, json!("test"));
    let result = g.run(ctx, agent_only(), Vec::new()).await.unwrap();
    assert!(result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, json!("test"));
}

#[tokio::test]
async fn output_guardrail_runs_and_reports_output_info() {
    disable_tracing();
    let ctx = openai_agents_sdk::RunContext::new(());

    let g = output_guardrail("g", false, serde_json::Value::Null);
    let result = g
        .run(ctx.clone(), agent_only(), json!("test"))
        .await
        .unwrap();
    assert!(!result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, serde_json::Value::Null);
    assert_eq!(result.agent_name, "test");
    assert_eq!(result.agent_output, json!("test"));

    let g = output_guardrail("g", true, json!("test"));
    let result = g.run(ctx, agent_only(), json!("test")).await.unwrap();
    assert!(result.output.tripwire_triggered);
    assert_eq!(result.output.output_info, json!("test"));
}

#[tokio::test]
async fn guardrail_names_are_kept() {
    let g = input_guardrail("Custom name", false, json!("test_2"));
    assert_eq!(g.name, "Custom name");
    let g = output_guardrail("Custom name", false, json!("test_4"));
    assert_eq!(g.name, "Custom name");
}

#[test]
fn input_guardrail_run_in_parallel_default_and_override() {
    let g = input_guardrail("g", false, serde_json::Value::Null);
    assert!(g.run_in_parallel);
    let g = g.blocking();
    assert!(!g.run_in_parallel);
}

// --- parallel vs blocking semantics in a run ---

#[tokio::test]
async fn parallel_guardrail_runs_concurrently_with_agent() {
    disable_tracing();
    let executed = Arc::new(AtomicBool::new(false));
    let flag = executed.clone();
    let guardrail = InputGuardrail::new("parallel_check", move |_c, _a, _i| {
        let flag = flag.clone();
        async move {
            tokio::task::yield_now().await;
            flag.store(true, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(json!("parallel_ok"), false))
        }
    });

    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let result = Runner::run(&agent, "test input").await.unwrap();

    assert!(executed.load(Ordering::SeqCst));
    assert!(result.final_output.is_some());
    assert_eq!(result.input_guardrail_results.len(), 1);
    assert_eq!(
        result.input_guardrail_results[0].output.output_info,
        json!("parallel_ok")
    );
    assert!(!model.calls().is_empty());
}

#[tokio::test]
async fn parallel_guardrail_runs_concurrently_with_agent_streaming() {
    disable_tracing();
    let executed = Arc::new(AtomicBool::new(false));
    let flag = executed.clone();
    let guardrail = InputGuardrail::new("parallel_check", move |_c, _a, _i| {
        let flag = flag.clone();
        async move {
            tokio::task::yield_now().await;
            flag.store(true, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(
                json!("parallel_streaming_ok"),
                false,
            ))
        }
    });

    let model = model_with(vec![assistant_message("hello from stream")]);
    let agent = Agent::<()>::new("streaming_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let mut streamed = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let mut received = false;
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
            received = true;
        }
    }

    assert!(executed.load(Ordering::SeqCst));
    assert!(received);
    assert!(!model.calls().is_empty());
}

#[tokio::test]
async fn blocking_guardrail_prevents_agent_execution() {
    disable_tracing();
    let executed = Arc::new(AtomicBool::new(false));
    let flag = executed.clone();
    let guardrail = InputGuardrail::new("blocking_check", move |_c, _a, _i| {
        let flag = flag.clone();
        async move {
            flag.store(true, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(
                json!("security_violation"),
                true,
            ))
        }
    })
    .blocking();

    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let err = Runner::run(&agent, "test input").await.unwrap_err();
    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(executed.load(Ordering::SeqCst));
    assert!(
        model.calls().is_empty(),
        "model must not be called when a blocking guardrail trips"
    );
}

#[tokio::test]
async fn blocking_guardrail_prevents_agent_execution_streaming() {
    disable_tracing();
    let guardrail = InputGuardrail::new("blocking_check", |_c, _a, _i| async move {
        Ok(GuardrailFunctionOutput::new(
            json!("blocked_streaming"),
            true,
        ))
    })
    .blocking();

    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("streaming_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let mut streamed = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let mut error = None;
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    assert!(matches!(
        error,
        Some(AgentsError::InputGuardrailTripwireTriggered { .. })
    ));
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn blocking_guardrail_prevents_tool_execution() {
    disable_tracing();
    let tool_ran = Arc::new(AtomicBool::new(false));
    let guardrail = InputGuardrail::new("security_check", |_c, _a, _i| async move {
        Ok(GuardrailFunctionOutput::new(
            json!("blocked_dangerous_input"),
            true,
        ))
    })
    .blocking();

    let model = model_with(vec![function_call("dangerous_tool", "{}")]);
    let agent = Agent::<()>::new("agent_with_tools")
        .with_model_instance(model.clone())
        .with_tool(dangerous_tool(tool_ran.clone()))
        .with_input_guardrail(guardrail);

    let err = Runner::run(&agent, "trigger guardrail").await.unwrap_err();
    assert!(matches!(
        err,
        AgentsError::InputGuardrailTripwireTriggered { .. }
    ));
    assert!(!tool_ran.load(Ordering::SeqCst));
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn blocking_guardrail_prevents_tool_execution_streaming() {
    disable_tracing();
    let tool_ran = Arc::new(AtomicBool::new(false));
    let guardrail = InputGuardrail::new("security_check", |_c, _a, _i| async move {
        Ok(GuardrailFunctionOutput::new(
            json!("blocked_dangerous_input_streaming"),
            true,
        ))
    })
    .blocking();

    let model = model_with(vec![function_call("dangerous_tool", "{}")]);
    let agent = Agent::<()>::new("agent_with_tools")
        .with_model_instance(model.clone())
        .with_tool(dangerous_tool(tool_ran.clone()))
        .with_input_guardrail(guardrail);

    let mut streamed =
        Runner::run_streamed_with_options(&agent, "trigger guardrail", RunOptions::new(()));
    let mut error = None;
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    assert!(matches!(
        error,
        Some(AgentsError::InputGuardrailTripwireTriggered { .. })
    ));
    assert!(!tool_ran.load(Ordering::SeqCst));
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn parallel_guardrail_passes_agent_continues() {
    disable_tracing();
    let guardrail = input_guardrail("parallel_check", false, json!("parallel_passed"));
    let model = model_with(vec![assistant_message("success")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let result = Runner::run(&agent, "test input").await.unwrap();
    assert!(result.final_output.is_some());
    assert!(!model.calls().is_empty());
}

#[tokio::test]
async fn blocking_guardrail_passes_agent_continues() {
    disable_tracing();
    let guardrail = input_guardrail("blocking_check", false, json!("blocking_passed")).blocking();
    let model = model_with(vec![assistant_message("success")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let result = Runner::run(&agent, "test input").await.unwrap();
    assert!(result.final_output.is_some());
    assert!(!model.calls().is_empty());
}

#[tokio::test]
async fn blocking_guardrail_completes_before_the_model_is_called() {
    disable_tracing();
    let blocking_done = Arc::new(AtomicBool::new(false));
    let flag = blocking_done.clone();
    let guardrail = InputGuardrail::new("blocking_check", move |_c, _a, _i| {
        let flag = flag.clone();
        async move {
            tokio::task::yield_now().await;
            flag.store(true, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(
                json!("blocking_passed"),
                false,
            ))
        }
    })
    .blocking();

    let seen_at_model_call = Arc::new(AtomicBool::new(false));
    let probe = seen_at_model_call.clone();
    let done = blocking_done.clone();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::respond(move |_call| {
            probe.store(done.load(Ordering::SeqCst), Ordering::SeqCst);
            Ok(ScriptedStep::output(vec![assistant_message("hello")]))
        }))
        .unwrap();

    let agent = Agent::<()>::new("multi_blocking_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let result = Runner::run(&agent, "test input").await.unwrap();
    assert!(result.final_output.is_some());
    assert!(
        seen_at_model_call.load(Ordering::SeqCst),
        "blocking guardrail must complete before the model is called"
    );
}

#[tokio::test]
async fn multiple_blocking_guardrails_one_triggers() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("multi_blocking_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(input_guardrail("passes", false, json!("passes")).blocking())
        .with_input_guardrail(input_guardrail("trips", true, json!("trips")).blocking());

    let err = Runner::run(&agent, "test input").await.unwrap_err();
    assert!(matches!(
        err,
        AgentsError::InputGuardrailTripwireTriggered { .. }
    ));
    assert!(model.calls().is_empty());
}

// --- results reported on the run result and on the raised error ---

fn guardrail_names(results: &[openai_agents_sdk::guardrail::InputGuardrailResult]) -> Vec<String> {
    results.iter().map(|r| r.guardrail_name.clone()).collect()
}

fn output_guardrail_names(
    results: &[openai_agents_sdk::guardrail::OutputGuardrailResult],
) -> Vec<String> {
    results.iter().map(|r| r.guardrail_name.clone()).collect()
}

/// Two input guardrails whose completion order is fixed by an explicit barrier, the second
/// tripping. Mirrors Python's `_ordered_input_guardrails` / `_tripwire_agent`.
fn ordered_input_guardrails(second_triggers: bool) -> (InputGuardrail<()>, InputGuardrail<()>) {
    let first_done = Arc::new(Notify::new());
    let signal = first_done.clone();
    let first = InputGuardrail::new("passes", move |_c, _a, _i| {
        let signal = signal.clone();
        async move {
            signal.notify_waiters();
            signal.notify_one();
            Ok(GuardrailFunctionOutput::new(json!("passes"), false))
        }
    });
    let wait = first_done.clone();
    let second = InputGuardrail::new("trips", move |_c, _a, _i| {
        let wait = wait.clone();
        async move {
            wait.notified().await;
            Ok(GuardrailFunctionOutput::new(
                json!("second"),
                second_triggers,
            ))
        }
    });
    (first, second)
}

#[tokio::test]
async fn input_guardrail_tripwire_reports_results() {
    disable_tracing();
    for blocking in [false, true] {
        let model = model_with(vec![assistant_message("hello")]);
        let (first, second) = ordered_input_guardrails(true);
        let (first, second) = if blocking {
            (first.blocking(), second.blocking())
        } else {
            (first, second)
        };
        let agent = Agent::<()>::new("guardrail_results_agent")
            .with_model_instance(model)
            .with_input_guardrail(first)
            .with_input_guardrail(second);

        let err = Runner::run(&agent, "test input").await.unwrap_err();
        let AgentsError::InputGuardrailTripwireTriggered { run_data, .. } = &err else {
            panic!("expected InputGuardrailTripwireTriggered, got {err:?}");
        };
        let run_data = run_data.as_ref().expect("run data");
        assert_eq!(
            guardrail_names(&run_data.input_guardrail_results),
            vec!["passes", "trips"],
            "blocking={blocking}"
        );
    }
}

#[tokio::test]
async fn input_guardrail_tripwire_reports_results_streamed() {
    disable_tracing();
    for blocking in [false, true] {
        let model = model_with(vec![assistant_message("hello")]);
        let (first, second) = ordered_input_guardrails(true);
        let (first, second) = if blocking {
            (first.blocking(), second.blocking())
        } else {
            (first, second)
        };
        let agent = Agent::<()>::new("guardrail_results_agent")
            .with_model_instance(model)
            .with_input_guardrail(first)
            .with_input_guardrail(second);

        let mut streamed =
            Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
        let mut error = None;
        {
            let mut events = std::pin::pin!(streamed.stream_events());
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                }
            }
        }
        let error = error.expect("tripwire");
        let AgentsError::InputGuardrailTripwireTriggered { run_data, .. } = &error else {
            panic!("expected InputGuardrailTripwireTriggered, got {error:?}");
        };
        assert_eq!(
            guardrail_names(&run_data.as_ref().expect("run data").input_guardrail_results),
            vec!["passes", "trips"],
            "blocking={blocking}"
        );
    }
}

#[test]
fn input_guardrail_tripwire_reports_results_sync() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let (first, second) = ordered_input_guardrails(true);
    let agent = Agent::<()>::new("guardrail_results_agent")
        .with_model_instance(model)
        .with_input_guardrail(first.blocking())
        .with_input_guardrail(second.blocking());

    let err = Runner::run_sync(&agent, "test input").unwrap_err();
    let AgentsError::InputGuardrailTripwireTriggered { run_data, .. } = &err else {
        panic!("expected InputGuardrailTripwireTriggered, got {err:?}");
    };
    assert_eq!(
        guardrail_names(&run_data.as_ref().expect("run data").input_guardrail_results),
        vec!["passes", "trips"]
    );
}

#[tokio::test]
async fn input_guardrail_results_reported_on_success() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_results_agent")
        .with_model_instance(model)
        .with_input_guardrail(
            input_guardrail("blocking", false, serde_json::Value::Null).blocking(),
        )
        .with_input_guardrail(input_guardrail("parallel", false, serde_json::Value::Null));

    let result = Runner::run(&agent, "test input").await.unwrap();
    assert_eq!(
        guardrail_names(&result.input_guardrail_results),
        vec!["blocking", "parallel"]
    );
}

#[tokio::test]
async fn output_guardrail_tripwire_reports_results() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("output_guardrail_results_agent")
        .with_model_instance(model)
        .with_output_guardrail(output_guardrail("passes", false, json!("passes")))
        .with_output_guardrail(output_guardrail("trips", true, json!("second")));

    let err = Runner::run(&agent, "test input").await.unwrap_err();
    let AgentsError::OutputGuardrailTripwireTriggered { run_data, .. } = &err else {
        panic!("expected OutputGuardrailTripwireTriggered, got {err:?}");
    };
    assert_eq!(
        output_guardrail_names(
            &run_data
                .as_ref()
                .expect("run data")
                .output_guardrail_results
        ),
        vec!["passes", "trips"]
    );
}

#[tokio::test]
async fn output_guardrail_tripwire_reports_results_streamed() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("output_guardrail_results_agent")
        .with_model_instance(model)
        .with_output_guardrail(output_guardrail("passes", false, json!("passes")))
        .with_output_guardrail(output_guardrail("trips", true, json!("second")));

    let mut streamed = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let mut error = None;
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    let error = error.expect("tripwire");
    let AgentsError::OutputGuardrailTripwireTriggered { run_data, .. } = &error else {
        panic!("expected OutputGuardrailTripwireTriggered, got {error:?}");
    };
    assert_eq!(
        output_guardrail_names(
            &run_data
                .as_ref()
                .expect("run data")
                .output_guardrail_results
        ),
        vec!["passes", "trips"]
    );
}

#[tokio::test]
async fn output_guardrail_results_reported_on_success() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("output_guardrail_results_agent")
        .with_model_instance(model)
        .with_output_guardrail(output_guardrail("passes", false, json!("passes")))
        .with_output_guardrail(output_guardrail("trips", false, json!("second")));

    let result = Runner::run(&agent, "test input").await.unwrap();
    assert_eq!(
        output_guardrail_names(&result.output_guardrail_results),
        vec!["passes", "trips"]
    );
}

#[tokio::test]
async fn input_guardrail_error_propagates_and_is_not_swallowed() {
    disable_tracing();
    let guardrail = InputGuardrail::new("raises", |_c, _a, _i| async move {
        Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail boom"))
    });
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_error_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let err = Runner::run(&agent, "trigger guardrail").await.unwrap_err();
    assert!(err.to_string().contains("guardrail boom"), "{err}");
}

#[tokio::test]
async fn output_guardrail_error_propagates_and_is_not_swallowed() {
    disable_tracing();
    let guardrail = OutputGuardrail::new("raises", |_c, _a, _o| async move {
        Err::<GuardrailFunctionOutput, _>(AgentsError::other("guardrail exploded"))
    });
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("output_guardrail_results_agent")
        .with_model_instance(model)
        .with_output_guardrail(guardrail);

    let err = Runner::run(&agent, "test input").await.unwrap_err();
    assert!(err.to_string().contains("guardrail exploded"), "{err}");
}

#[tokio::test]
async fn guardrail_via_agent_and_run_config_are_equivalent() {
    disable_tracing();
    // On the agent.
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("a")
        .with_model_instance(model)
        .with_input_guardrail(input_guardrail("trips", true, json!("trips")));
    let via_agent = Runner::run(&agent, "test input").await.unwrap_err();

    // On the run config.
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("a").with_model_instance(model);
    let mut config = openai_agents_sdk::RunConfig::<()>::default();
    config
        .input_guardrails
        .push(input_guardrail("trips", true, json!("trips")));
    let via_config =
        Runner::run_with_options(&agent, "test input", RunOptions::new(()).run_config(config))
            .await
            .unwrap_err();

    assert!(matches!(
        via_agent,
        AgentsError::InputGuardrailTripwireTriggered { .. }
    ));
    assert!(matches!(
        via_config,
        AgentsError::InputGuardrailTripwireTriggered { .. }
    ));
}
