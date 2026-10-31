//! Ported from the Python SDK's `tests/test_guardrails.py`: parallel and blocking input
//! guardrails, their ordering against the model and tools, result reporting and sibling
//! cancellation, through the public `Runner` API.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::models::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, InputGuardrail, OutputGuardrail, Result, RunOptions, Runner,
};
use serde_json::{Value, json};
use tokio::sync::Notify;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn guard(name: &str, triggers: bool, output_info: Value) -> InputGuardrail<()> {
    let name = name.to_string();
    InputGuardrail::new(name, move |_c, _a, _i| {
        let output_info = output_info.clone();
        async move { Ok(GuardrailFunctionOutput::new(output_info, triggers)) }
    })
}

fn model_with(items: Vec<openai_agents_sdk::items::OutputItem>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model.enqueue(ScriptedStep::output(items)).unwrap();
    model
}

/// Wraps a scripted model and runs `before_call` ahead of every model request, the way the
/// Python tests patch `get_response` / `stream_response` to observe ordering.
#[derive(Clone)]
struct ObservedModel {
    inner: Arc<ScriptedModel>,
    before_call: Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>,
}

impl std::fmt::Debug for ObservedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservedModel").finish_non_exhaustive()
    }
}

#[async_trait]
impl Model for ObservedModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        (self.before_call)().await;
        self.inner.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        (self.before_call)().await;
        self.inner.stream_response(request).await
    }
}

fn observed(
    inner: Arc<ScriptedModel>,
    before_call: impl Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync + 'static,
) -> Arc<ObservedModel> {
    Arc::new(ObservedModel {
        inner,
        before_call: Arc::new(before_call),
    })
}

fn names<T>(results: &[T], name: impl Fn(&T) -> &str) -> Vec<String> {
    results.iter().map(|r| name(r).to_string()).collect()
}

/// Streams the run to completion and returns the streamed result's error, if any.
async fn drain(result: &mut openai_agents_sdk::RunResultStreaming<()>) -> Result<()> {
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        event?;
    }
    Ok(())
}

#[tokio::test]
async fn parallel_guardrail_passes_agent_continues_streaming() {
    disable_tracing();
    let executed = Arc::new(AtomicBool::new(false));
    let flag = executed.clone();
    let guardrail = InputGuardrail::new("parallel_check", move |_c, _a, _i| {
        let flag = flag.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            flag.store(true, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(
                json!("parallel_passed_streaming"),
                false,
            ))
        }
    });
    let model = model_with(vec![assistant_message("success")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    drain(&mut result).await.unwrap();

    assert!(executed.load(Ordering::SeqCst));
    assert!(!model.calls().is_empty(), "Model should have been called");
}

#[tokio::test]
async fn blocking_guardrail_passes_agent_continues_streaming() {
    disable_tracing();
    let executed = Arc::new(AtomicBool::new(false));
    let flag = executed.clone();
    let guardrail = InputGuardrail::new("blocking_check", move |_c, _a, _i| {
        let flag = flag.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            flag.store(true, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(
                json!("blocking_passed_streaming"),
                false,
            ))
        }
    })
    .blocking();
    let model = model_with(vec![assistant_message("success")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(guardrail);

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    drain(&mut result).await.unwrap();

    assert!(executed.load(Ordering::SeqCst));
    assert!(
        !model.calls().is_empty(),
        "Model should have been called after guardrail passed"
    );
}

#[tokio::test]
async fn mixed_blocking_and_parallel_guardrails() {
    disable_tracing();
    let blocking_finished = Arc::new(AtomicBool::new(false));
    let parallel_started = Arc::new(Notify::new());
    let model_called = Arc::new(Notify::new());
    let parallel_finished = Arc::new(AtomicBool::new(false));
    let blocking_at_parallel_start = Arc::new(AtomicBool::new(false));
    let blocking_at_model_call = Arc::new(AtomicBool::new(false));
    let parallel_at_model_call = Arc::new(AtomicBool::new(true));

    let blocking = {
        let finished = blocking_finished.clone();
        InputGuardrail::new("blocking_check", move |_c, _a, _i| {
            let finished = finished.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                finished.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("blocking_passed"),
                    false,
                ))
            }
        })
        .blocking()
    };
    let parallel = {
        let finished = blocking_finished.clone();
        let at_start = blocking_at_parallel_start.clone();
        let started = parallel_started.clone();
        let model_called = model_called.clone();
        let parallel_finished = parallel_finished.clone();
        InputGuardrail::new("parallel_check", move |_c, _a, _i| {
            let finished = finished.clone();
            let at_start = at_start.clone();
            let started = started.clone();
            let model_called = model_called.clone();
            let parallel_finished = parallel_finished.clone();
            async move {
                at_start.store(finished.load(Ordering::SeqCst), Ordering::SeqCst);
                started.notify_one();
                tokio::time::timeout(Duration::from_secs(5), model_called.notified())
                    .await
                    .expect("model call");
                parallel_finished.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("parallel_passed"),
                    false,
                ))
            }
        })
    };

    let inner = model_with(vec![assistant_message("hello")]);
    let model = {
        let finished = blocking_finished.clone();
        let at_model = blocking_at_model_call.clone();
        let parallel_at_model = parallel_at_model_call.clone();
        let parallel_started = parallel_started.clone();
        let parallel_finished = parallel_finished.clone();
        let model_called = model_called.clone();
        observed(inner.clone(), move || {
            let finished = finished.clone();
            let at_model = at_model.clone();
            let parallel_at_model = parallel_at_model.clone();
            let parallel_started = parallel_started.clone();
            let parallel_finished = parallel_finished.clone();
            let model_called = model_called.clone();
            Box::pin(async move {
                at_model.store(finished.load(Ordering::SeqCst), Ordering::SeqCst);
                tokio::time::timeout(Duration::from_secs(5), parallel_started.notified())
                    .await
                    .expect("parallel guardrail start");
                parallel_at_model.store(parallel_finished.load(Ordering::SeqCst), Ordering::SeqCst);
                model_called.notify_one();
            })
        })
    };

    let agent = Agent::<()>::new("mixed_agent")
        .with_model_instance(model)
        .with_input_guardrail(blocking)
        .with_input_guardrail(parallel);

    let result = Runner::run(&agent, "test input").await.unwrap();

    assert!(result.final_output.is_some());
    assert_eq!(result.input_guardrail_results.len(), 2);
    assert!(
        blocking_at_parallel_start.load(Ordering::SeqCst),
        "Blocking must complete before parallel starts"
    );
    assert!(
        blocking_at_model_call.load(Ordering::SeqCst),
        "Blocking must complete before model is called"
    );
    assert!(
        !parallel_at_model_call.load(Ordering::SeqCst),
        "Model called while parallel guardrail still running"
    );
    assert!(parallel_finished.load(Ordering::SeqCst));
    assert!(!inner.calls().is_empty());
}

#[tokio::test]
async fn mixed_blocking_and_parallel_guardrails_streaming() {
    disable_tracing();
    let blocking_finished = Arc::new(AtomicBool::new(false));
    let parallel_started = Arc::new(Notify::new());
    let model_called = Arc::new(Notify::new());
    let parallel_finished = Arc::new(AtomicBool::new(false));
    let blocking_at_parallel_start = Arc::new(AtomicBool::new(false));
    let blocking_at_model_call = Arc::new(AtomicBool::new(false));
    let parallel_at_model_call = Arc::new(AtomicBool::new(true));

    let blocking = {
        let finished = blocking_finished.clone();
        InputGuardrail::new("blocking_check", move |_c, _a, _i| {
            let finished = finished.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                finished.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("blocking_passed"),
                    false,
                ))
            }
        })
        .blocking()
    };
    let parallel = {
        let finished = blocking_finished.clone();
        let at_start = blocking_at_parallel_start.clone();
        let started = parallel_started.clone();
        let model_called = model_called.clone();
        let parallel_finished = parallel_finished.clone();
        InputGuardrail::new("parallel_check", move |_c, _a, _i| {
            let finished = finished.clone();
            let at_start = at_start.clone();
            let started = started.clone();
            let model_called = model_called.clone();
            let parallel_finished = parallel_finished.clone();
            async move {
                at_start.store(finished.load(Ordering::SeqCst), Ordering::SeqCst);
                started.notify_one();
                tokio::time::timeout(Duration::from_secs(5), model_called.notified())
                    .await
                    .expect("model call");
                parallel_finished.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("parallel_passed"),
                    false,
                ))
            }
        })
    };

    let inner = model_with(vec![assistant_message("hello")]);
    let model = {
        let finished = blocking_finished.clone();
        let at_model = blocking_at_model_call.clone();
        let parallel_at_model = parallel_at_model_call.clone();
        let parallel_started = parallel_started.clone();
        let parallel_finished = parallel_finished.clone();
        let model_called = model_called.clone();
        observed(inner.clone(), move || {
            let finished = finished.clone();
            let at_model = at_model.clone();
            let parallel_at_model = parallel_at_model.clone();
            let parallel_started = parallel_started.clone();
            let parallel_finished = parallel_finished.clone();
            let model_called = model_called.clone();
            Box::pin(async move {
                at_model.store(finished.load(Ordering::SeqCst), Ordering::SeqCst);
                tokio::time::timeout(Duration::from_secs(5), parallel_started.notified())
                    .await
                    .expect("parallel guardrail start");
                parallel_at_model.store(parallel_finished.load(Ordering::SeqCst), Ordering::SeqCst);
                model_called.notify_one();
            })
        })
    };

    let agent = Agent::<()>::new("mixed_agent")
        .with_model_instance(model)
        .with_input_guardrail(blocking)
        .with_input_guardrail(parallel);

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    drain(&mut result).await.unwrap();

    assert!(
        blocking_at_parallel_start.load(Ordering::SeqCst),
        "Blocking must complete before parallel starts"
    );
    assert!(
        blocking_at_model_call.load(Ordering::SeqCst),
        "Blocking must complete before model is called"
    );
    assert!(
        !parallel_at_model_call.load(Ordering::SeqCst),
        "Model called while parallel guardrail still running"
    );
    assert!(parallel_finished.load(Ordering::SeqCst));
    assert!(!inner.calls().is_empty());
}

#[tokio::test]
async fn multiple_blocking_guardrails_complete_before_agent() {
    disable_tracing();
    let first_end = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));
    let second_end = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));
    let model_called = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));

    let timed = |slot: Arc<std::sync::Mutex<Option<std::time::Instant>>>, name: &'static str| {
        InputGuardrail::new(name, move |_c, _a, _i| {
            let slot = slot.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                *slot.lock().unwrap() = Some(std::time::Instant::now());
                Ok(GuardrailFunctionOutput::new(json!(name), false))
            }
        })
        .blocking()
    };
    let inner = model_with(vec![assistant_message("hello")]);
    let model = {
        let slot = model_called.clone();
        observed(inner.clone(), move || {
            let slot = slot.clone();
            Box::pin(async move {
                *slot.lock().unwrap() = Some(std::time::Instant::now());
            })
        })
    };
    let agent = Agent::<()>::new("multi_blocking_agent")
        .with_model_instance(model)
        .with_input_guardrail(timed(first_end.clone(), "first_passed"))
        .with_input_guardrail(timed(second_end.clone(), "second_passed"));

    let result = Runner::run(&agent, "test input").await.unwrap();

    assert!(result.final_output.is_some());
    assert_eq!(result.input_guardrail_results.len(), 2);
    let model_at = model_called.lock().unwrap().expect("model called");
    assert!(first_end.lock().unwrap().expect("first ran") <= model_at);
    assert!(second_end.lock().unwrap().expect("second ran") <= model_at);
    assert!(!inner.calls().is_empty());
}

#[tokio::test]
async fn multiple_blocking_guardrails_complete_before_agent_streaming() {
    disable_tracing();
    let first_end = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));
    let second_end = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));
    let model_called = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));

    let timed = |slot: Arc<std::sync::Mutex<Option<std::time::Instant>>>, name: &'static str| {
        InputGuardrail::new(name, move |_c, _a, _i| {
            let slot = slot.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                *slot.lock().unwrap() = Some(std::time::Instant::now());
                Ok(GuardrailFunctionOutput::new(json!(name), false))
            }
        })
        .blocking()
    };
    let inner = model_with(vec![assistant_message("hello")]);
    let model = {
        let slot = model_called.clone();
        observed(inner.clone(), move || {
            let slot = slot.clone();
            Box::pin(async move {
                *slot.lock().unwrap() = Some(std::time::Instant::now());
            })
        })
    };
    let agent = Agent::<()>::new("multi_blocking_agent")
        .with_model_instance(model)
        .with_input_guardrail(timed(first_end.clone(), "first_passed"))
        .with_input_guardrail(timed(second_end.clone(), "second_passed"));

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    drain(&mut result).await.unwrap();

    let model_at = model_called.lock().unwrap().expect("model called");
    assert!(first_end.lock().unwrap().expect("first ran") <= model_at);
    assert!(second_end.lock().unwrap().expect("second ran") <= model_at);
    assert!(!inner.calls().is_empty());
}

#[tokio::test]
async fn multiple_blocking_guardrails_one_triggers_streaming() {
    disable_tracing();
    let first_executed = Arc::new(AtomicBool::new(false));
    let second_executed = Arc::new(AtomicBool::new(false));
    let first = {
        let flag = first_executed.clone();
        InputGuardrail::new("first_blocking_check", move |_c, _a, _i| {
            let flag = flag.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                flag.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(json!("first_passed"), false))
            }
        })
        .blocking()
    };
    let second = {
        let flag = second_executed.clone();
        InputGuardrail::new("second_blocking_check", move |_c, _a, _i| {
            let flag = flag.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                flag.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("second_triggered"),
                    true,
                ))
            }
        })
        .blocking()
    };
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("multi_blocking_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(first)
        .with_input_guardrail(second);

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let err = drain(&mut result).await.unwrap_err();

    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(first_executed.load(Ordering::SeqCst));
    assert!(second_executed.load(Ordering::SeqCst));
    assert!(
        model.calls().is_empty(),
        "Model should not have been called when guardrail triggered"
    );
}

#[tokio::test]
async fn parallel_guardrail_may_not_prevent_tool_execution() {
    disable_tracing();
    let tool_ran = Arc::new(AtomicBool::new(false));
    let tool_executed = Arc::new(Notify::new());
    let guardrail_executed = Arc::new(AtomicBool::new(false));
    let tool = {
        let tool_ran = tool_ran.clone();
        let tool_executed = tool_executed.clone();
        FunctionTool::new("fast_tool", "", move |_ctx, _a: Value| {
            let tool_ran = tool_ran.clone();
            let tool_executed = tool_executed.clone();
            async move {
                tool_ran.store(true, Ordering::SeqCst);
                tool_executed.notify_one();
                Ok("tool_executed".to_string())
            }
        })
    };
    let guardrail = {
        let executed = guardrail_executed.clone();
        let tool_executed = tool_executed.clone();
        InputGuardrail::new("slow_parallel_check", move |_c, _a, _i| {
            let executed = executed.clone();
            let tool_executed = tool_executed.clone();
            async move {
                // Trip only after the tool ran; a gated turn would never reach this point.
                tokio::time::timeout(Duration::from_secs(5), tool_executed.notified())
                    .await
                    .expect("tool must run before the guardrail trips");
                executed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("slow_parallel_triggered"),
                    true,
                ))
            }
        })
    };
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![function_call("fast_tool", "{}")]))
        .unwrap();
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let agent = Agent::<()>::new("agent_with_tools")
        .with_model_instance(model.clone())
        .with_tool(tool)
        .with_input_guardrail(guardrail);

    let err = tokio::time::timeout(
        Duration::from_secs(10),
        Runner::run(&agent, "trigger guardrail"),
    )
    .await
    .expect("run finished")
    .unwrap_err();

    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(guardrail_executed.load(Ordering::SeqCst));
    assert!(
        tool_ran.load(Ordering::SeqCst),
        "Expected tool to execute before slow parallel guardrail triggered"
    );
    assert!(
        !model.calls().is_empty(),
        "Model should have been called in parallel mode"
    );
}

#[tokio::test]
async fn parallel_guardrail_may_not_prevent_tool_execution_streaming() {
    disable_tracing();
    let tool_ran = Arc::new(AtomicBool::new(false));
    let tool_executed = Arc::new(Notify::new());
    let guardrail_executed = Arc::new(AtomicBool::new(false));
    let tool = {
        let tool_ran = tool_ran.clone();
        let tool_executed = tool_executed.clone();
        FunctionTool::new("fast_tool", "", move |_ctx, _a: Value| {
            let tool_ran = tool_ran.clone();
            let tool_executed = tool_executed.clone();
            async move {
                tool_ran.store(true, Ordering::SeqCst);
                tool_executed.notify_one();
                Ok("tool_executed".to_string())
            }
        })
    };
    let guardrail = {
        let executed = guardrail_executed.clone();
        let tool_executed = tool_executed.clone();
        InputGuardrail::new("slow_parallel_check", move |_c, _a, _i| {
            let executed = executed.clone();
            let tool_executed = tool_executed.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(5), tool_executed.notified())
                    .await
                    .expect("tool must run before the guardrail trips");
                executed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("slow_parallel_triggered_streaming"),
                    true,
                ))
            }
        })
    };
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![function_call("fast_tool", "{}")]))
        .unwrap();
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let agent = Agent::<()>::new("agent_with_tools")
        .with_model_instance(model.clone())
        .with_tool(tool)
        .with_input_guardrail(guardrail);

    let mut result =
        Runner::run_streamed_with_options(&agent, "trigger guardrail", RunOptions::new(()));
    let err = tokio::time::timeout(Duration::from_secs(10), drain(&mut result))
        .await
        .expect("stream finished")
        .unwrap_err();

    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(guardrail_executed.load(Ordering::SeqCst));
    assert!(
        tool_ran.load(Ordering::SeqCst),
        "Expected tool to execute before slow parallel guardrail triggered"
    );
    assert!(
        !model.calls().is_empty(),
        "Model should have been called in parallel mode"
    );
}

#[tokio::test]
async fn blocking_guardrail_cancels_remaining_on_trigger() {
    disable_tracing();
    let slow_started = Arc::new(Notify::new());
    let slow_completed = Arc::new(AtomicBool::new(false));
    let slow_cancelled = Arc::new(AtomicBool::new(false));
    let fast_executed = Arc::new(AtomicBool::new(false));

    let fast = {
        let started = slow_started.clone();
        let executed = fast_executed.clone();
        InputGuardrail::new("fast_guardrail_that_triggers", move |_c, _a, _i| {
            let started = started.clone();
            let executed = executed.clone();
            async move {
                // Trip only once the sibling is in flight and therefore cancellable.
                tokio::time::timeout(Duration::from_secs(5), started.notified())
                    .await
                    .expect("sibling started");
                executed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(json!("fast_triggered"), true))
            }
        })
        .blocking()
    };
    let slow = {
        let started = slow_started.clone();
        let completed = slow_completed.clone();
        let cancelled = slow_cancelled.clone();
        InputGuardrail::new(
            "slow_guardrail_that_should_be_cancelled",
            move |_c, _a, _i| {
                let started = started.clone();
                let completed = completed.clone();
                let cancelled = cancelled.clone();
                async move {
                    let _cancel_flag = DropFlag(cancelled);
                    started.notify_one();
                    // Never finishes on its own, so only cancellation can end this guardrail.
                    std::future::pending::<()>().await;
                    completed.store(true, Ordering::SeqCst);
                    Ok(GuardrailFunctionOutput::new(json!("slow_completed"), false))
                }
            },
        )
        .blocking()
    };
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(fast)
        .with_input_guardrail(slow);

    let err = tokio::time::timeout(Duration::from_secs(5), Runner::run(&agent, "test input"))
        .await
        .expect("run finished")
        .unwrap_err();

    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(
        fast_executed.load(Ordering::SeqCst),
        "Fast guardrail should have executed"
    );
    assert!(
        slow_cancelled.load(Ordering::SeqCst),
        "Slow guardrail should have been cancelled"
    );
    assert!(
        !slow_completed.load(Ordering::SeqCst),
        "Slow guardrail should NOT have completed"
    );
    assert!(
        model.calls().is_empty(),
        "Model should not have been called when guardrail triggered"
    );
}

#[tokio::test]
async fn blocking_guardrail_cancels_remaining_on_trigger_streaming() {
    disable_tracing();
    let slow_started = Arc::new(Notify::new());
    let slow_completed = Arc::new(AtomicBool::new(false));
    let slow_cancelled = Arc::new(AtomicBool::new(false));
    let fast_executed = Arc::new(AtomicBool::new(false));

    let fast = {
        let started = slow_started.clone();
        let executed = fast_executed.clone();
        InputGuardrail::new("fast_guardrail_that_triggers", move |_c, _a, _i| {
            let started = started.clone();
            let executed = executed.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(5), started.notified())
                    .await
                    .expect("sibling started");
                executed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(json!("fast_triggered"), true))
            }
        })
        .blocking()
    };
    let slow = {
        let started = slow_started.clone();
        let completed = slow_completed.clone();
        let cancelled = slow_cancelled.clone();
        InputGuardrail::new(
            "slow_guardrail_that_should_be_cancelled",
            move |_c, _a, _i| {
                let started = started.clone();
                let completed = completed.clone();
                let cancelled = cancelled.clone();
                async move {
                    let _cancel_flag = DropFlag(cancelled);
                    started.notify_one();
                    std::future::pending::<()>().await;
                    completed.store(true, Ordering::SeqCst);
                    Ok(GuardrailFunctionOutput::new(json!("slow_completed"), false))
                }
            },
        )
        .blocking()
    };
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model.clone())
        .with_input_guardrail(fast)
        .with_input_guardrail(slow);

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let err = tokio::time::timeout(Duration::from_secs(5), drain(&mut result))
        .await
        .expect("stream finished")
        .unwrap_err();

    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(
        fast_executed.load(Ordering::SeqCst),
        "Fast guardrail should have executed"
    );
    assert!(
        slow_cancelled.load(Ordering::SeqCst),
        "Slow guardrail should have been cancelled"
    );
    assert!(
        !slow_completed.load(Ordering::SeqCst),
        "Slow guardrail should NOT have completed"
    );
    assert!(
        model.calls().is_empty(),
        "Model should not have been called when guardrail triggered"
    );
}

/// Sets the flag when dropped. A future dropped mid-await is the Rust analogue of a task that
/// observed `CancelledError`.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn input_guardrail_raise_cancels_siblings() {
    disable_tracing();
    let sibling_started = Arc::new(Notify::new());
    let sibling_completed = Arc::new(AtomicBool::new(false));
    let sibling_cancelled = Arc::new(AtomicBool::new(false));

    let sibling = {
        let started = sibling_started.clone();
        let completed = sibling_completed.clone();
        let cancelled = sibling_cancelled.clone();
        InputGuardrail::new("slow_sibling_started_first", move |_c, _a, _i| {
            let started = started.clone();
            let completed = completed.clone();
            let cancelled = cancelled.clone();
            async move {
                let _cancel_flag = DropFlag(cancelled);
                started.notify_one();
                tokio::time::sleep(Duration::from_millis(500)).await;
                completed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(json!(null), false))
            }
        })
        .blocking()
    };
    let raising = {
        let started = sibling_started.clone();
        InputGuardrail::new("raise_after_sibling_starts", move |_c, _a, _i| {
            let started = started.clone();
            async move {
                started.notified().await;
                Err(AgentsError::user("boom"))
            }
        })
        .blocking()
    };

    let agent = Agent::<()>::new("t")
        .with_model_instance(model_with(vec![assistant_message("never")]))
        .with_input_guardrail(sibling)
        .with_input_guardrail(raising);
    let err = Runner::run(&agent, "x").await.unwrap_err();

    assert!(err.to_string().contains("boom"), "{err}");
    assert!(
        sibling_cancelled.load(Ordering::SeqCst),
        "Sibling task should have been cancelled"
    );
    assert!(
        !sibling_completed.load(Ordering::SeqCst),
        "Sibling task should not have completed"
    );
}

#[tokio::test]
async fn output_guardrail_raise_cancels_siblings() {
    disable_tracing();
    let sibling_started = Arc::new(Notify::new());
    let sibling_completed = Arc::new(AtomicBool::new(false));
    let sibling_cancelled = Arc::new(AtomicBool::new(false));

    let sibling = {
        let started = sibling_started.clone();
        let completed = sibling_completed.clone();
        let cancelled = sibling_cancelled.clone();
        OutputGuardrail::new("slow_sibling_started_first", move |_c, _a, _o| {
            let started = started.clone();
            let completed = completed.clone();
            let cancelled = cancelled.clone();
            async move {
                let _cancel_flag = DropFlag(cancelled);
                started.notify_one();
                tokio::time::sleep(Duration::from_millis(500)).await;
                completed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(json!(null), false))
            }
        })
    };
    let raising = {
        let started = sibling_started.clone();
        OutputGuardrail::new("raise_after_sibling_starts", move |_c, _a, _o| {
            let started = started.clone();
            async move {
                started.notified().await;
                Err(AgentsError::user("boom"))
            }
        })
    };

    let agent = Agent::<()>::new("t")
        .with_model_instance(model_with(vec![assistant_message("out")]))
        .with_output_guardrail(sibling)
        .with_output_guardrail(raising);
    let err = Runner::run(&agent, "x").await.unwrap_err();

    assert!(err.to_string().contains("boom"), "{err}");
    assert!(
        sibling_cancelled.load(Ordering::SeqCst),
        "Sibling task should have been cancelled"
    );
    assert!(
        !sibling_completed.load(Ordering::SeqCst),
        "Sibling task should not have completed"
    );
}

/// Two guardrails whose completion order is fixed by a barrier: `first` always completes first.
fn ordered_input_guardrails(
    second_triggers: bool,
    second_raises: bool,
    parallel: bool,
) -> Vec<InputGuardrail<()>> {
    let first_done = Arc::new(Notify::new());
    let first = {
        let done = first_done.clone();
        InputGuardrail::new("passes", move |_c, _a, _i| {
            let done = done.clone();
            async move {
                done.notify_one();
                Ok(GuardrailFunctionOutput::new(json!("passes"), false))
            }
        })
    };
    let second_name = if second_raises { "raises" } else { "trips" };
    let second = {
        let done = first_done.clone();
        InputGuardrail::new(second_name, move |_c, _a, _i| {
            let done = done.clone();
            async move {
                done.notified().await;
                if second_raises {
                    return Err(AgentsError::user("guardrail exploded"));
                }
                Ok(GuardrailFunctionOutput::new(
                    json!("second"),
                    second_triggers,
                ))
            }
        })
    };
    let mut guards = vec![first, second];
    if !parallel {
        guards = guards.into_iter().map(InputGuardrail::blocking).collect();
    }
    guards
}

fn guardrail_names_of(
    results: &[openai_agents_sdk::guardrail::InputGuardrailResult],
) -> Vec<String> {
    names(results, |r| r.guardrail_name.as_str())
}

#[tokio::test]
async fn input_guardrail_tripwire_reports_results_blocking() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_results_agent").with_model_instance(model);
    let agent = ordered_input_guardrails(true, false, false)
        .into_iter()
        .fold(agent, |agent, g| agent.with_input_guardrail(g));

    let err = Runner::run(&agent, "test input").await.unwrap_err();

    let AgentsError::InputGuardrailTripwireTriggered {
        guardrail_result, ..
    } = &err
    else {
        panic!("expected a tripwire, got {err:?}");
    };
    assert_eq!(guardrail_result.guardrail_name, "trips");
    let run_data = err.run_data().expect("run data");
    assert_eq!(
        guardrail_names_of(&run_data.input_guardrail_results),
        vec!["passes", "trips"]
    );
}

#[tokio::test]
async fn input_guardrail_tripwire_reports_results_parallel() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_results_agent").with_model_instance(model);
    let agent = ordered_input_guardrails(true, false, true)
        .into_iter()
        .fold(agent, |agent, g| agent.with_input_guardrail(g));

    let err = Runner::run(&agent, "test input").await.unwrap_err();

    let AgentsError::InputGuardrailTripwireTriggered {
        guardrail_result, ..
    } = &err
    else {
        panic!("expected a tripwire, got {err:?}");
    };
    assert_eq!(guardrail_result.guardrail_name, "trips");
    let run_data = err.run_data().expect("run data");
    assert_eq!(
        guardrail_names_of(&run_data.input_guardrail_results),
        vec!["passes", "trips"]
    );
}

#[tokio::test]
async fn input_guardrail_tripwire_reports_results_streamed_blocking() {
    disable_tracing();
    run_streamed_tripwire_reports(false).await;
}

#[tokio::test]
async fn input_guardrail_tripwire_reports_results_streamed_parallel() {
    disable_tracing();
    run_streamed_tripwire_reports(true).await;
}

async fn run_streamed_tripwire_reports(parallel: bool) {
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_results_agent").with_model_instance(model);
    let agent = ordered_input_guardrails(true, false, parallel)
        .into_iter()
        .fold(agent, |agent, g| agent.with_input_guardrail(g));

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let err = drain(&mut result).await.unwrap_err();

    let run_data = err.run_data().expect("run data");
    assert_eq!(
        guardrail_names_of(&run_data.input_guardrail_results),
        vec!["passes", "trips"]
    );
}

#[tokio::test]
async fn input_guardrail_exception_reports_completed_results() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_results_agent").with_model_instance(model);
    let agent = ordered_input_guardrails(false, true, false)
        .into_iter()
        .fold(agent, |agent, g| agent.with_input_guardrail(g));

    let err = Runner::run(&agent, "test input").await.unwrap_err();

    assert!(err.to_string().contains("guardrail exploded"), "{err}");
    let run_data = err.run_data().expect("run data");
    assert_eq!(
        guardrail_names_of(&run_data.input_guardrail_results),
        vec!["passes"]
    );
}

#[tokio::test]
async fn input_guardrail_exception_reports_completed_results_streamed_blocking() {
    disable_tracing();
    run_streamed_exception_reports(false).await;
}

#[tokio::test]
async fn input_guardrail_exception_reports_completed_results_streamed_parallel() {
    disable_tracing();
    run_streamed_exception_reports(true).await;
}

async fn run_streamed_exception_reports(parallel: bool) {
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_results_agent").with_model_instance(model);
    let agent = ordered_input_guardrails(false, true, parallel)
        .into_iter()
        .fold(agent, |agent, g| agent.with_input_guardrail(g));

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let err = drain(&mut result).await.unwrap_err();

    assert!(err.to_string().contains("guardrail exploded"), "{err}");
    let run_data = err.run_data().expect("run data");
    assert_eq!(
        guardrail_names_of(&run_data.input_guardrail_results),
        vec!["passes"]
    );
}

#[tokio::test]
async fn input_guardrail_results_are_reported_on_success_in_completion_order() {
    disable_tracing();
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("guardrail_results_agent")
        .with_model_instance(model)
        .with_input_guardrail(guard("blocking", false, Value::Null).blocking())
        .with_input_guardrail(guard("parallel", false, Value::Null));

    let result = Runner::run(&agent, "test input").await.unwrap();

    assert_eq!(
        guardrail_names_of(&result.input_guardrail_results),
        vec!["blocking", "parallel"]
    );
}

#[tokio::test]
async fn parallel_guardrail_counts_each_invocation_once() {
    disable_tracing();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let guardrail = InputGuardrail::new("counted", move |_c, _a, _i| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(GuardrailFunctionOutput::new(Value::Null, false))
        }
    });
    let model = model_with(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("counted_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let result = Runner::run(&agent, "test input").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.input_guardrail_results.len(), 1);
}

/// Two output guardrails whose completion order is fixed by a barrier: `passes` always completes
/// first, and the second one trips or raises.
fn ordered_output_guardrails(second_raises: bool) -> Vec<OutputGuardrail<()>> {
    let first_done = Arc::new(Notify::new());
    let first = {
        let done = first_done.clone();
        OutputGuardrail::new("passes", move |_c, _a, _o| {
            let done = done.clone();
            async move {
                done.notify_one();
                Ok(GuardrailFunctionOutput::new(json!("passes"), false))
            }
        })
    };
    let second_name = if second_raises { "raises" } else { "trips" };
    let second = {
        let done = first_done.clone();
        OutputGuardrail::new(second_name, move |_c, _a, _o| {
            let done = done.clone();
            async move {
                done.notified().await;
                if second_raises {
                    return Err(AgentsError::user("guardrail exploded"));
                }
                Ok(GuardrailFunctionOutput::new(json!("second"), true))
            }
        })
    };
    vec![first, second]
}

#[tokio::test]
async fn output_guardrail_exception_reports_completed_results() {
    disable_tracing();
    let agent = ordered_output_guardrails(true).into_iter().fold(
        Agent::<()>::new("output_guardrail_results_agent")
            .with_model_instance(model_with(vec![assistant_message("hello")])),
        |agent, g| agent.with_output_guardrail(g),
    );

    let err = Runner::run(&agent, "test input").await.unwrap_err();

    assert!(err.to_string().contains("guardrail exploded"), "{err}");
    let run_data = err.run_data().expect("run data");
    let names = names(&run_data.output_guardrail_results, |r| {
        r.guardrail_name.as_str()
    });
    assert_eq!(names, vec!["passes"]);
}

#[tokio::test]
async fn output_guardrail_exception_reports_completed_results_streamed() {
    disable_tracing();
    let agent = ordered_output_guardrails(true).into_iter().fold(
        Agent::<()>::new("output_guardrail_results_agent")
            .with_model_instance(model_with(vec![assistant_message("hello")])),
        |agent, g| agent.with_output_guardrail(g),
    );

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let err = drain(&mut result).await.unwrap_err();

    assert!(err.to_string().contains("guardrail exploded"), "{err}");
    let run_data = err.run_data().expect("run data");
    let names = names(&run_data.output_guardrail_results, |r| {
        r.guardrail_name.as_str()
    });
    assert_eq!(names, vec!["passes"]);
}

#[test]
fn output_guardrail_tripwire_reports_results_sync() {
    disable_tracing();
    let agent = ordered_output_guardrails(false).into_iter().fold(
        Agent::<()>::new("output_guardrail_results_agent")
            .with_model_instance(model_with(vec![assistant_message("hello")])),
        |agent, g| agent.with_output_guardrail(g),
    );

    let err = Runner::run_sync(&agent, "test input").unwrap_err();

    let AgentsError::OutputGuardrailTripwireTriggered {
        guardrail_result, ..
    } = &err
    else {
        panic!("expected an output tripwire, got {err:?}");
    };
    assert_eq!(guardrail_result.guardrail_name, "trips");
    let run_data = err.run_data().expect("run data");
    let names = names(&run_data.output_guardrail_results, |r| {
        r.guardrail_name.as_str()
    });
    assert_eq!(names, vec!["passes", "trips"]);
}
