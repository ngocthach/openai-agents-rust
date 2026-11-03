//! Ported from the Python SDK's `tests/test_guardrails.py`: parallel input guardrails racing the
//! model call. A tripwire cancels the in-flight model call, a model error cancels a running
//! guardrail, and a guardrail error cancels the model. Python asserts this with
//! `CancelledError` inside the coroutines; here a dropped future sets a `DropFlag`, which is the
//! observable equivalent.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::BoxFuture;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::models::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{
    Agent, AgentsError, InputGuardrail, OutputGuardrail, Result, RunOptions, Runner,
};
use serde_json::json;
use tokio::sync::Notify;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

/// Sets the flag when dropped. A future dropped mid-await is the Rust analogue of a task that
/// observed `CancelledError`.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Runs `before_call` ahead of every model request. The Python tests patch `get_response` and
/// `stream_response` in the same way to slow the model down or fail it.
#[derive(Clone)]
struct GatedModel {
    inner: Arc<ScriptedModel>,
    before_call: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>,
}

impl std::fmt::Debug for GatedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedModel").finish_non_exhaustive()
    }
}

#[async_trait]
impl Model for GatedModel {
    async fn get_response(&self, request: ModelRequest) -> Result<ModelResponse> {
        (self.before_call)().await;
        self.inner.get_response(request).await
    }

    async fn stream_response(&self, request: ModelRequest) -> Result<ResponseEventStream> {
        (self.before_call)().await;
        self.inner.stream_response(request).await
    }
}

fn gated(
    inner: Arc<ScriptedModel>,
    before_call: impl Fn() -> BoxFuture<'static, ()> + Send + Sync + 'static,
) -> Arc<GatedModel> {
    Arc::new(GatedModel {
        inner,
        before_call: Arc::new(before_call),
    })
}

/// A future that never completes on its own, but reports its own cancellation.
fn pending_until_cancelled(cancelled: Arc<AtomicBool>) -> BoxFuture<'static, ()> {
    Box::pin(async move {
        let _flag = DropFlag(cancelled);
        std::future::pending::<()>().await;
    })
}

fn timeout<F: std::future::Future>(fut: F) -> tokio::time::Timeout<F> {
    tokio::time::timeout(Duration::from_secs(5), fut)
}

#[tokio::test]
async fn parallel_guardrail_trip_cancels_model_task() {
    disable_tracing();
    let model_started = Arc::new(Notify::new());
    let model_cancelled = Arc::new(AtomicBool::new(false));
    let inner = Arc::new(ScriptedModel::new());
    inner
        .enqueue(ScriptedStep::output(vec![assistant_message(
            "should_not_finish",
        )]))
        .unwrap();

    let guardrail = {
        let started = model_started.clone();
        InputGuardrail::new("tripwire_after_model_starts", move |_c, _a, _i| {
            let started = started.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(1), started.notified())
                    .await
                    .expect("model started");
                Ok(GuardrailFunctionOutput::new(
                    json!("parallel_tripwire"),
                    true,
                ))
            }
        })
    };
    let model = {
        let started = model_started.clone();
        let cancelled = model_cancelled.clone();
        gated(inner.clone(), move || {
            let started = started.clone();
            let cancelled = cancelled.clone();
            Box::pin(async move {
                started.notify_one();
                pending_until_cancelled(cancelled).await;
            })
        })
    };
    let agent = Agent::<()>::new("parallel_tripwire_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let err = timeout(Runner::run(&agent, "trigger guardrail"))
        .await
        .expect("run finished")
        .unwrap_err();

    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(
        model_cancelled.load(Ordering::SeqCst),
        "the in-flight model call must be cancelled"
    );
    assert!(
        inner.calls().is_empty(),
        "the cancelled model call must not reach the model"
    );
}

#[tokio::test]
async fn model_error_cancels_parallel_input_guardrail_task() {
    disable_tracing();
    let guardrail_started = Arc::new(Notify::new());
    let guardrail_cancelled = Arc::new(AtomicBool::new(false));
    let guardrail_finished = Arc::new(AtomicBool::new(false));

    let guardrail = {
        let started = guardrail_started.clone();
        let cancelled = guardrail_cancelled.clone();
        let finished = guardrail_finished.clone();
        InputGuardrail::new("slow_parallel_check", move |_c, _a, _i| {
            let started = started.clone();
            let cancelled = cancelled.clone();
            let finished = finished.clone();
            async move {
                let _flag = DropFlag(cancelled);
                started.notify_one();
                // Never finishes on its own, so only cancellation can end this guardrail.
                std::future::pending::<()>().await;
                finished.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(json!("parallel_ok"), false))
            }
        })
    };
    let inner = Arc::new(ScriptedModel::new());
    inner
        .enqueue(ScriptedStep::raise_error(AgentsError::user("model boom")))
        .unwrap();
    let model = {
        let started = guardrail_started.clone();
        gated(inner, move || {
            let started = started.clone();
            Box::pin(async move {
                // Only fail once the guardrail is genuinely mid-flight.
                tokio::time::timeout(Duration::from_secs(1), started.notified())
                    .await
                    .expect("guardrail started");
            })
        })
    };
    let agent = Agent::<()>::new("model_error_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let err = timeout(Runner::run(&agent, "trigger guardrail"))
        .await
        .expect("run finished")
        .unwrap_err();

    assert!(err.to_string().contains("model boom"), "{err}");
    assert!(
        guardrail_cancelled.load(Ordering::SeqCst),
        "the guardrail must be cancelled"
    );
    assert!(
        !guardrail_finished.load(Ordering::SeqCst),
        "the guardrail must not complete"
    );
}

#[tokio::test]
async fn parallel_guardrail_non_tripwire_error_is_not_swallowed() {
    disable_tracing();
    let model_started = Arc::new(Notify::new());
    let model_cancelled = Arc::new(AtomicBool::new(false));

    let guardrail = {
        let started = model_started.clone();
        InputGuardrail::new("raising_parallel_check", move |_c, _a, _i| {
            let started = started.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(1), started.notified())
                    .await
                    .expect("model started");
                Err(AgentsError::user("guardrail boom"))
            }
        })
    };
    let inner = Arc::new(ScriptedModel::new());
    inner
        .enqueue(ScriptedStep::output(vec![assistant_message(
            "should_not_finish",
        )]))
        .unwrap();
    let model = {
        let started = model_started.clone();
        let cancelled = model_cancelled.clone();
        gated(inner, move || {
            let started = started.clone();
            let cancelled = cancelled.clone();
            Box::pin(async move {
                started.notify_one();
                pending_until_cancelled(cancelled).await;
            })
        })
    };
    let agent = Agent::<()>::new("guardrail_error_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let err = timeout(Runner::run(&agent, "trigger guardrail"))
        .await
        .expect("run finished")
        .unwrap_err();

    assert!(err.to_string().contains("guardrail boom"), "{err}");
    assert!(
        model_cancelled.load(Ordering::SeqCst),
        "the in-flight model call must be cancelled"
    );
}

#[tokio::test]
async fn parallel_guardrail_error_cancels_streaming_model() {
    disable_tracing();
    let model_started = Arc::new(Notify::new());
    let model_cancelled = Arc::new(AtomicBool::new(false));

    let guardrail = {
        let started = model_started.clone();
        InputGuardrail::new("raising_parallel_check", move |_c, _a, _i| {
            let started = started.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(1), started.notified())
                    .await
                    .expect("model started");
                Err(AgentsError::user("guardrail boom"))
            }
        })
    };
    let inner = Arc::new(ScriptedModel::new());
    let model = {
        let started = model_started.clone();
        let cancelled = model_cancelled.clone();
        gated(inner, move || {
            let started = started.clone();
            let cancelled = cancelled.clone();
            Box::pin(async move {
                started.notify_one();
                pending_until_cancelled(cancelled).await;
            })
        })
    };
    let agent = Agent::<()>::new("streaming_guardrail_error_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let mut result =
        Runner::run_streamed_with_options(&agent, "trigger guardrail", RunOptions::new(()));
    let err = timeout(async {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event?;
        }
        Ok::<(), AgentsError>(())
    })
    .await
    .expect("stream finished")
    .unwrap_err();

    assert!(err.to_string().contains("guardrail boom"), "{err}");
    assert!(
        model_cancelled.load(Ordering::SeqCst),
        "the streaming model call must be cancelled"
    );
}

#[tokio::test]
async fn streamed_model_error_reports_the_guardrail_error_instead() {
    disable_tracing();
    let model_failed = Arc::new(Notify::new());

    let guardrail = {
        let failed = model_failed.clone();
        InputGuardrail::new("raising_parallel_check", move |_c, _a, _i| {
            let failed = failed.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(1), failed.notified())
                    .await
                    .expect("model failed");
                // Let the model's error reach the run loop before the guardrail fails.
                tokio::time::sleep(Duration::from_millis(10)).await;
                Err(AgentsError::user("guardrail boom"))
            }
        })
    };
    let inner = Arc::new(ScriptedModel::new());
    inner
        .enqueue(ScriptedStep::raise_error(AgentsError::user("model boom")))
        .unwrap();
    let model = {
        let failed = model_failed.clone();
        gated(inner, move || {
            let failed = failed.clone();
            Box::pin(async move {
                failed.notify_one();
            })
        })
    };
    let agent = Agent::<()>::new("streaming_model_error_agent")
        .with_model_instance(model)
        .with_input_guardrail(guardrail);

    let mut result =
        Runner::run_streamed_with_options(&agent, "trigger model error", RunOptions::new(()));
    let err = timeout(async {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event?;
        }
        Ok::<(), AgentsError>(())
    })
    .await
    .expect("stream finished")
    .unwrap_err();

    assert!(err.to_string().contains("guardrail boom"), "{err}");
}

#[tokio::test]
async fn parallel_trip_before_tool_execution_stops_streaming_turn() {
    disable_tracing();
    let tool_ran = Arc::new(AtomicBool::new(false));
    let model_started = Arc::new(Notify::new());
    let guardrail_tripped = Arc::new(Notify::new());
    let model_entered = Arc::new(AtomicBool::new(false));

    let tool = {
        let ran = tool_ran.clone();
        openai_agents_sdk::tool::FunctionTool::new(
            "dangerous_tool",
            "",
            move |_ctx, _a: serde_json::Value| {
                let ran = ran.clone();
                async move {
                    ran.store(true, Ordering::SeqCst);
                    Ok("tool_executed".to_string())
                }
            },
        )
    };
    let guardrail = {
        let started = model_started.clone();
        let tripped = guardrail_tripped.clone();
        InputGuardrail::new("tripwire_before_tool_execution", move |_c, _a, _i| {
            let started = started.clone();
            let tripped = tripped.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(1), started.notified())
                    .await
                    .expect("model started");
                tripped.notify_one();
                Ok(GuardrailFunctionOutput::new(
                    json!("parallel_trip_before_tool_execution"),
                    true,
                ))
            }
        })
    };
    let inner = Arc::new(ScriptedModel::new());
    inner
        .enqueue(ScriptedStep::output(vec![
            openai_agents_sdk::testing::function_call("dangerous_tool", "{}"),
        ]))
        .unwrap();
    inner
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let model = {
        let started = model_started.clone();
        let tripped = guardrail_tripped.clone();
        let model_entered = model_entered.clone();
        gated(inner.clone(), move || {
            let started = started.clone();
            let tripped = tripped.clone();
            let entered = model_entered.clone();
            Box::pin(async move {
                entered.store(true, Ordering::SeqCst);
                started.notify_one();
                tokio::time::timeout(Duration::from_secs(1), tripped.notified())
                    .await
                    .expect("guardrail tripped");
                // Give a finished guardrail the chance to win the race, as the Python test does.
                tokio::time::sleep(Duration::from_millis(5)).await;
            })
        })
    };
    let agent = Agent::<()>::new("streaming_guardrail_hardening_agent")
        .with_model_instance(model)
        .with_tool(tool)
        .with_input_guardrail(guardrail);

    let mut result =
        Runner::run_streamed_with_options(&agent, "trigger guardrail", RunOptions::new(()));
    let err = timeout(async {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event?;
        }
        Ok::<(), AgentsError>(())
    })
    .await
    .expect("stream finished")
    .unwrap_err();

    assert!(
        matches!(err, AgentsError::InputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(
        !tool_ran.load(Ordering::SeqCst),
        "the tool must not run after the tripwire"
    );
    assert!(
        model_entered.load(Ordering::SeqCst),
        "the model should have started in parallel mode"
    );
}

#[tokio::test]
async fn parallel_trip_with_a_slow_sibling_keeps_only_the_tripping_result() {
    disable_tracing();
    let model_started = Arc::new(Notify::new());
    let guardrail_tripped = Arc::new(Notify::new());
    let sibling_cancelled = Arc::new(AtomicBool::new(false));
    let sibling_completed = Arc::new(AtomicBool::new(false));
    let tool_ran = Arc::new(AtomicBool::new(false));

    let tool = {
        let ran = tool_ran.clone();
        openai_agents_sdk::tool::FunctionTool::new(
            "dangerous_tool",
            "",
            move |_ctx, _a: serde_json::Value| {
                let ran = ran.clone();
                async move {
                    ran.store(true, Ordering::SeqCst);
                    Ok("tool_executed".to_string())
                }
            },
        )
    };
    let tripping = {
        let started = model_started.clone();
        let tripped = guardrail_tripped.clone();
        InputGuardrail::new("tripwire_before_tool_execution", move |_c, _a, _i| {
            let started = started.clone();
            let tripped = tripped.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(1), started.notified())
                    .await
                    .expect("model started");
                tripped.notify_one();
                Ok(GuardrailFunctionOutput::new(
                    json!("parallel_trip_before_tool_execution_with_slow_cancel"),
                    true,
                ))
            }
        })
    };
    let slow = {
        let cancelled = sibling_cancelled.clone();
        let completed = sibling_completed.clone();
        InputGuardrail::new("slow_to_cancel_guardrail", move |_c, _a, _i| {
            let cancelled = cancelled.clone();
            let completed = completed.clone();
            async move {
                let _flag = DropFlag(cancelled);
                std::future::pending::<()>().await;
                completed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(
                    json!("slow_to_cancel_guardrail_completed"),
                    false,
                ))
            }
        })
    };
    let inner = Arc::new(ScriptedModel::new());
    inner
        .enqueue(ScriptedStep::output(vec![
            openai_agents_sdk::testing::function_call("dangerous_tool", "{}"),
        ]))
        .unwrap();
    inner
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let model = {
        let started = model_started.clone();
        let tripped = guardrail_tripped.clone();
        gated(inner.clone(), move || {
            let started = started.clone();
            let tripped = tripped.clone();
            Box::pin(async move {
                started.notify_one();
                tokio::time::timeout(Duration::from_secs(1), tripped.notified())
                    .await
                    .expect("guardrail tripped");
                tokio::time::sleep(Duration::from_millis(5)).await;
            })
        })
    };
    let agent = Agent::<()>::new("streaming_guardrail_slow_cancel_agent")
        .with_model_instance(model)
        .with_tool(tool)
        .with_input_guardrail(tripping)
        .with_input_guardrail(slow);

    let mut result =
        Runner::run_streamed_with_options(&agent, "trigger guardrail", RunOptions::new(()));
    let err = timeout(async {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event?;
        }
        Ok::<(), AgentsError>(())
    })
    .await
    .expect("stream finished")
    .unwrap_err();

    let run_data = err.run_data().expect("run data");
    let names: Vec<&str> = run_data
        .input_guardrail_results
        .iter()
        .map(|r| r.guardrail_name.as_str())
        .collect();
    assert_eq!(names, vec!["tripwire_before_tool_execution"]);
    assert!(
        sibling_cancelled.load(Ordering::SeqCst),
        "the slow sibling must be cancelled"
    );
    assert!(!sibling_completed.load(Ordering::SeqCst));
    assert!(
        !tool_ran.load(Ordering::SeqCst),
        "the tool must not run after the tripwire"
    );
}

#[tokio::test]
async fn streamed_input_guardrail_error_cancels_a_slow_sibling() {
    disable_tracing();
    let slow_started = Arc::new(Notify::new());
    let slow_cancelled = Arc::new(AtomicBool::new(false));
    let slow_completed = Arc::new(AtomicBool::new(false));

    let slow = {
        let started = slow_started.clone();
        let cancelled = slow_cancelled.clone();
        let completed = slow_completed.clone();
        InputGuardrail::new("slow_guardrail", move |_c, _a, _i| {
            let started = started.clone();
            let cancelled = cancelled.clone();
            let completed = completed.clone();
            async move {
                let _flag = DropFlag(cancelled);
                started.notify_one();
                tokio::time::sleep(Duration::from_millis(500)).await;
                completed.store(true, Ordering::SeqCst);
                Ok(GuardrailFunctionOutput::new(json!(null), false))
            }
        })
    };
    let raising = {
        let started = slow_started.clone();
        InputGuardrail::new("raising_guardrail", move |_c, _a, _i| {
            let started = started.clone();
            async move {
                started.notified().await;
                Err(AgentsError::user("guardrail failed"))
            }
        })
    };
    let model = Arc::new(ScriptedModel::new());
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model)
        .with_input_guardrail(slow)
        .with_input_guardrail(raising);

    let mut result = Runner::run_streamed_with_options(&agent, "test input", RunOptions::new(()));
    let err = timeout(async {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            event?;
        }
        Ok::<(), AgentsError>(())
    })
    .await
    .expect("stream finished")
    .unwrap_err();

    assert!(err.to_string().contains("guardrail failed"), "{err}");
    assert!(
        slow_cancelled.load(Ordering::SeqCst),
        "the slow sibling must be dropped"
    );
    assert!(!slow_completed.load(Ordering::SeqCst));
}

/// The output guardrail that trips cancels its slow sibling, which observes the cancellation
/// before the run returns.
#[tokio::test]
async fn output_guardrail_trip_cancels_sibling_before_the_run_returns() {
    disable_tracing();
    let slow_started = Arc::new(Notify::new());
    let slow_cancelled = Arc::new(AtomicBool::new(false));

    let slow = {
        let started = slow_started.clone();
        let cancelled = slow_cancelled.clone();
        OutputGuardrail::new("slow_then_observe_cancel", move |_c, _a, _o| {
            let started = started.clone();
            let cancelled = cancelled.clone();
            async move {
                let _flag = DropFlag(cancelled);
                started.notify_one();
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(GuardrailFunctionOutput::new(json!(null), false))
            }
        })
    };
    let fast = {
        let started = slow_started.clone();
        OutputGuardrail::new("fast_tripwire", move |_c, _a, _o| {
            let started = started.clone();
            async move {
                // Trip only once the sibling is parked on its sleep.
                started.notified().await;
                Ok(GuardrailFunctionOutput::new(json!(null), true))
            }
        })
    };
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message(
            "agent output",
        )]))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_output_guardrail(slow)
        .with_output_guardrail(fast);

    let err = timeout(Runner::run(&agent, "go"))
        .await
        .expect("run finished")
        .unwrap_err();

    assert!(
        matches!(err, AgentsError::OutputGuardrailTripwireTriggered { .. }),
        "{err:?}"
    );
    assert!(
        slow_cancelled.load(Ordering::SeqCst),
        "the sibling must observe cancellation"
    );
}
