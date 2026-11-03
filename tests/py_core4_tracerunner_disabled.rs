//! Port of the Python SDK's `tests/test_runner_tracing_disabled.py`: a run with tracing turned off
//! must not export spans, must not leak its own spans into the caller's trace, and must leave the
//! caller's current trace and span exactly as they were, including after an error, a cancellation,
//! a resume, and a failure while the session is being prepared.
//!
//! Tracing state is process-global, so every test takes the shared lock and installs its own
//! recorder through [`Pipeline`].

#[path = "py_core4_tracerunner_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::Result as AgentsResult;
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::models::interface::{Model, ModelProvider};
use openai_agents_sdk::models::{OpenAIChatCompletionsModel, OpenAIClient};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{
    Span, TracingProcessor, current_span, current_trace, custom_span, trace,
};
use openai_agents_sdk::{
    Agent, AgentsError, InputGuardrail, OutputGuardrail, RunConfig, RunOptions, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use support::{ModelReply, Pipeline, SpanRecorder, lock_async, start_model_endpoint};

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

/// A scripted model returning `steps` in order, emitting generation spans when `emit`.
fn scripted(steps: Vec<Vec<OutputItem>>, emit: bool) -> Arc<ScriptedModel> {
    let model = ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect())
        .expect("scripted steps");
    Arc::new(model.with_emit_traces(emit))
}

/// Install a recorder for one test and hold the lock that serialises tracing tests.
async fn recording() -> (
    tokio::sync::MutexGuard<'static, ()>,
    Arc<SpanRecorder>,
    Pipeline,
) {
    let lock = lock_async().await;
    let rec = SpanRecorder::new();
    let pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    (lock, rec, pipeline)
}

/// The `name` of each exported span, as Python's `getattr(span.span_data, "name", None)`.
fn names(rec: &SpanRecorder) -> Vec<Option<String>> {
    rec.ordered_spans()
        .iter()
        .map(|span| {
            span.export()
                .and_then(|e| e["span_data"]["name"].as_str().map(str::to_string))
        })
        .collect()
}

fn names_of(rec: &SpanRecorder) -> Vec<String> {
    names(rec)
        .into_iter()
        .map(|n| n.unwrap_or_default())
        .collect()
}

/// The span id of the first exported span called `name`.
fn id_named(rec: &SpanRecorder, name: &str) -> String {
    rec.ordered_spans()
        .iter()
        .find(|s| s.export().is_some_and(|e| e["span_data"]["name"] == name))
        .unwrap_or_else(|| panic!("no span named {name}"))
        .span_id()
        .to_string()
}

fn span_type(span: &Span) -> String {
    span.export()
        .map(|e| {
            e["span_data"]["type"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .unwrap_or_default()
}

/// Current trace is `expected`, by identity (trace id).
fn assert_current_trace(expected: &openai_agents_sdk::tracing::Trace) {
    assert_eq!(
        current_trace().map(|t| t.trace_id().to_string()),
        Some(expected.trace_id().to_string())
    );
}

/// Current span is `expected`, by identity (span id).
fn assert_current_span(expected: &Span) {
    assert_eq!(
        current_span().map(|s| s.span_id().to_string()),
        Some(expected.span_id().to_string())
    );
}

fn text_tool(name: &'static str, output: &'static str) -> FunctionTool<()> {
    FunctionTool::new(name, "a tool", move |_c, _a: AnyArgs| async move {
        Ok(output.to_string())
    })
}

// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn runner_tracing_opt_out_inside_caller_span() {
    for streamed in [false, true] {
        for disabled in [false, true] {
            let (_lock, rec, _p) = recording().await;
            let lookup = FunctionTool::new("lookup", "look up", |_c, _a: AnyArgs| async move {
                custom_span("inside-tool", json!({}))
                    .run(async { Ok("synthetic weather for Paris".to_string()) })
                    .await
            });
            let allow = InputGuardrail::new("allow", |_c, _a, _i| async move {
                Ok(GuardrailFunctionOutput::new(json!(null), false))
            });
            let target = Agent::<()>::new("target")
                .with_model_instance(scripted(vec![vec![assistant_message("done")]], true));
            let agent = Agent::<()>::new("start")
                .with_model_instance(scripted(
                    vec![
                        vec![function_call_with_id(
                            "lookup",
                            r#"{"city":"Paris"}"#,
                            "lookup-1",
                        )],
                        vec![function_call_with_id(
                            "transfer_to_target",
                            "{}",
                            "handoff-1",
                        )],
                    ],
                    true,
                ))
                .with_tool(lookup)
                .with_handoff(target)
                .with_input_guardrail(allow);
            let options =
                || RunOptions::new(()).run_config(RunConfig::new().with_tracing_disabled(disabled));

            let caller = trace("caller").build();
            let caller_for_check = caller.clone();
            caller
                .clone()
                .run(async {
                    let caller_span = custom_span("caller-parent", json!({}));
                    caller_span
                        .clone()
                        .run(async {
                            custom_span("before", json!({})).run(async {}).await;
                            let final_output = if streamed {
                                let mut result =
                                    Runner::run_streamed_with_options(&agent, "hi", options());
                                assert_current_trace(&caller_for_check);
                                assert_current_span(&caller_span);
                                {
                                    let mut events = std::pin::pin!(result.stream_events());
                                    while let Some(event) = events.next().await {
                                        event.unwrap();
                                        assert_current_trace(&caller_for_check);
                                        assert_current_span(&caller_span);
                                    }
                                }
                                result.final_output_text()
                            } else {
                                Runner::run_with_options(&agent, "hi", options())
                                    .await
                                    .unwrap()
                                    .final_output_text()
                            };
                            assert_eq!(final_output.as_deref(), Some("done"));
                            assert_current_trace(&caller_for_check);
                            assert_current_span(&caller_span);
                            custom_span("after", json!({})).run(async {}).await;
                        })
                        .await;
                })
                .await;

            let ctx = format!("streamed={streamed} disabled={disabled}");
            assert_eq!(
                rec.traces()
                    .iter()
                    .map(|t| t.trace_id().to_string())
                    .collect::<Vec<_>>(),
                vec![caller.trace_id().to_string()],
                "{ctx}"
            );
            let spans = rec.ordered_spans();
            if disabled {
                assert_eq!(
                    names_of(&rec),
                    vec!["caller-parent", "before", "after"],
                    "{ctx}"
                );
                let events = rec.events();
                assert_eq!(
                    events.iter().filter(|e| **e == "span_start").count(),
                    3,
                    "{ctx}"
                );
                assert_eq!(
                    events.iter().filter(|e| **e == "span_end").count(),
                    3,
                    "{ctx}"
                );
            } else {
                let types: std::collections::BTreeSet<String> =
                    spans.iter().map(span_type).collect();
                for expected in ["agent", "function", "guardrail", "handoff", "generation"] {
                    assert!(
                        types.contains(expected),
                        "{ctx}: missing {expected} in {types:?}"
                    );
                }
                assert!(names_of(&rec).contains(&"inside-tool".to_string()), "{ctx}");
            }
            assert!(
                spans.iter().all(|s| s.trace_id() == caller.trace_id()),
                "{ctx}"
            );
            let after = spans
                .iter()
                .find(|s| {
                    s.export()
                        .is_some_and(|e| e["span_data"]["name"] == "after")
                })
                .expect("after span");
            assert_eq!(
                after.parent_id(),
                Some(id_named(&rec, "caller-parent").as_str()),
                "{ctx}"
            );
        }
    }
}

#[tokio::test]
async fn disabled_runner_restores_context_after_error() {
    for streamed in [false, true] {
        let (_lock, rec, _p) = recording().await;
        let model = ScriptedModel::new();
        model
            .enqueue(ScriptedStep::raise_error(AgentsError::other("synthetic")))
            .unwrap();
        let agent = Agent::<()>::new("error").with_model_instance(Arc::new(model));
        let config = RunConfig::new().with_tracing_disabled(true);
        let caller = trace("caller").build();
        caller
            .clone()
            .run(async {
                let caller_span = custom_span("caller-parent", json!({}));
                caller_span
                    .clone()
                    .run(async {
                        let outcome: AgentsResult<()> = if streamed {
                            let mut result = Runner::run_streamed_with_options(
                                &agent,
                                "hi",
                                RunOptions::new(()).run_config(config.clone()),
                            );
                            let mut error = None;
                            {
                                let mut events = std::pin::pin!(result.stream_events());
                                while let Some(event) = events.next().await {
                                    if let Err(e) = event {
                                        error = Some(e);
                                        break;
                                    }
                                }
                            }
                            match error {
                                Some(e) => Err(e),
                                None => Ok(()),
                            }
                        } else {
                            Runner::run_with_options(
                                &agent,
                                "hi",
                                RunOptions::new(()).run_config(config.clone()),
                            )
                            .await
                            .map(|_| ())
                        };
                        let err = outcome.expect_err("the synthetic model error must surface");
                        assert!(err.to_string().contains("synthetic"), "{err}");
                        assert_current_trace(&caller);
                        assert_current_span(&caller_span);
                        custom_span("after-error", json!({})).run(async {}).await;
                    })
                    .await;
            })
            .await;

        assert_eq!(
            names_of(&rec),
            vec!["caller-parent", "after-error"],
            "streamed={streamed}"
        );
    }
}

#[tokio::test]
async fn disabled_runner_cancellation_preserves_concurrent_caller() {
    for streamed in [false, true] {
        let (_lock, rec, _p) = recording().await;
        let entered = Arc::new(tokio::sync::Notify::new());
        let cleaned_up = Arc::new(AtomicBool::new(false));

        /// Runs its cleanup when the tool future is dropped (cancelled), like Python's `finally`.
        struct Cleanup(Arc<AtomicBool>);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                // Mirrors `with custom_span("tool-cleanup"): cleaned_up.set()` in the Python test.
                let span = custom_span("tool-cleanup", json!({}));
                span.start();
                self.0.store(true, Ordering::SeqCst);
                span.finish();
            }
        }

        let entered_in_tool = entered.clone();
        let cleaned_in_tool = cleaned_up.clone();
        let block = FunctionTool::new("block", "blocks", move |_c, _a: AnyArgs| {
            let entered = entered_in_tool.clone();
            let cleanup = Cleanup(cleaned_in_tool.clone());
            async move {
                entered.notify_one();
                let _cleanup = cleanup;
                std::future::pending::<()>().await;
                Ok("unreachable".to_string())
            }
        });
        let agent = Agent::<()>::new("blocked")
            .with_model_instance(scripted(
                vec![vec![openai_agents_sdk::testing::function_call(
                    "block", "{}",
                )]],
                false,
            ))
            .with_tool(block);
        let config = RunConfig::new().with_tracing_disabled(true);

        let caller = trace("caller").build();
        caller
            .clone()
            .run(async {
                let caller_span = custom_span("caller-parent", json!({}));
                caller_span
                    .clone()
                    .run(async {
                        let run_wait = async {
                            tokio::time::timeout(
                                std::time::Duration::from_secs(5),
                                entered.notified(),
                            )
                            .await
                            .expect("tool entered");
                        };
                        if streamed {
                            let mut result = Runner::run_streamed_with_options(
                                &agent,
                                "hi",
                                RunOptions::new(()).run_config(config.clone()),
                            );
                            run_wait.await;
                            assert_current_trace(&caller);
                            assert_current_span(&caller_span);
                            custom_span("while-running", json!({})).run(async {}).await;
                            result.cancel(CancelMode::Immediate);
                            let mut events = std::pin::pin!(result.stream_events());
                            while events.next().await.is_some() {}
                        } else {
                            let snapshot =
                                openai_agents_sdk::tracing::scope::ScopeSnapshot::capture();
                            let runner_agent = agent.clone();
                            let runner_config = config.clone();
                            let handle = tokio::spawn(snapshot.enter(async move {
                                Runner::run_with_options(
                                    &runner_agent,
                                    "hi",
                                    RunOptions::new(()).run_config(runner_config),
                                )
                                .await
                            }));
                            run_wait.await;
                            assert_current_trace(&caller);
                            assert_current_span(&caller_span);
                            custom_span("while-running", json!({})).run(async {}).await;
                            handle.abort();
                            assert!(handle.await.unwrap_err().is_cancelled());
                        }
                        assert!(
                            cleaned_up.load(Ordering::SeqCst),
                            "the cancelled tool must run its cleanup"
                        );
                        assert_current_trace(&caller);
                        assert_current_span(&caller_span);
                    })
                    .await;
            })
            .await;

        assert_eq!(
            names_of(&rec),
            vec!["caller-parent", "while-running"],
            "streamed={streamed}"
        );
    }
}

#[test]
fn disabled_runner_sync_preserves_caller_context() {
    let lock = support::lock();
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);
    let agent = Agent::<()>::new("sync")
        .with_model_instance(scripted(vec![vec![assistant_message("done")]], false));
    let caller = trace("caller").build();
    futures::executor::block_on(caller.clone().run(async {
        let caller_span = custom_span("caller-parent", json!({}));
        caller_span
            .clone()
            .run(async {
                let result = Runner::run_sync_with_options(
                    &agent,
                    "hi",
                    RunOptions::new(()).run_config(RunConfig::new().with_tracing_disabled(true)),
                )
                .unwrap();
                assert_eq!(result.final_output_text().as_deref(), Some("done"));
                assert_current_trace(&caller);
                assert_current_span(&caller_span);
            })
            .await;
    }));
    assert_eq!(names_of(&rec), vec!["caller-parent"]);
    // The pipeline resets the global processors when dropped, so it must go before the lock is
    // released, or it would clear the recorder of the next test that takes the lock.
    drop(_pipeline);
    drop(lock);
}

#[tokio::test]
async fn disabled_resume_does_not_inherit_caller_trace() {
    for streamed in [false, true] {
        let (_lock, rec, _p) = recording().await;
        let agent = Agent::<()>::new("approval")
            .with_tool(text_tool("approved_tool", "synthetic result").with_needs_approval(true))
            .with_model_instance(scripted(
                vec![
                    vec![function_call_with_id("approved_tool", "{}", "approval-1")],
                    vec![assistant_message("done")],
                ],
                false,
            ));
        let interrupted = Runner::run_with_options(&agent, "hi", RunOptions::new(()))
            .await
            .unwrap();
        let mut state = interrupted.to_state();
        state.approve(&interrupted.interruptions[0], false);
        rec.clear();

        let caller = trace("caller").build();
        let config = RunConfig::new().with_tracing_disabled(true);
        caller
            .clone()
            .run(async {
                let caller_span = custom_span("caller-parent", json!({}));
                caller_span
                    .clone()
                    .run(async {
                        let trace_state_is_none;
                        if streamed {
                            let mut result = Runner::run_streamed_with_options(
                                &agent,
                                state.clone(),
                                RunOptions::new(()).run_config(config.clone()),
                            );
                            {
                                let mut events = std::pin::pin!(result.stream_events());
                                while let Some(event) = events.next().await {
                                    event.unwrap();
                                }
                            }
                            assert_eq!(result.final_output_text().as_deref(), Some("done"));
                            trace_state_is_none =
                                result.to_state().expect("state").trace_state.is_none();
                        } else {
                            let result = Runner::run_with_options(
                                &agent,
                                state.clone(),
                                RunOptions::new(()).run_config(config.clone()),
                            )
                            .await
                            .unwrap();
                            assert_eq!(result.final_output_text().as_deref(), Some("done"));
                            trace_state_is_none = result.to_state().trace_state.is_none();
                        }
                        assert_current_trace(&caller);
                        assert_current_span(&caller_span);
                        assert!(trace_state_is_none, "streamed={streamed}");
                    })
                    .await;
            })
            .await;

        assert_eq!(
            rec.traces()
                .iter()
                .map(|t| t.trace_id().to_string())
                .collect::<Vec<_>>(),
            vec![caller.trace_id().to_string()],
            "streamed={streamed}"
        );
        assert_eq!(names_of(&rec), vec!["caller-parent"], "streamed={streamed}");
    }
}

/// A session whose reads run inside a named custom span and can be made to fail. Mirrors the
/// Python test's `Session` subclass.
#[derive(Debug)]
struct TracedSession {
    inner: InMemorySession,
    fail_read: bool,
}

#[async_trait]
impl Session for TracedSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }

    async fn get_items(&self, limit: Option<usize>) -> AgentsResult<Vec<InputItem>> {
        custom_span("session-read", json!({}))
            .run(async {
                if self.fail_read {
                    return Err(AgentsError::other("synthetic session-read failure"));
                }
                self.inner.get_items(limit).await
            })
            .await
    }

    async fn add_items(&self, items: Vec<InputItem>) -> AgentsResult<()> {
        self.inner.add_items(items).await
    }

    async fn pop_item(&self) -> AgentsResult<Option<InputItem>> {
        self.inner.pop_item().await
    }

    async fn clear_session(&self) -> AgentsResult<()> {
        self.inner.clear_session().await
    }
}

#[tokio::test]
async fn runner_tracing_opt_out_during_session_preparation() {
    for disabled in [false, true] {
        for failure in [None, Some("session-read"), Some("input-callback")] {
            let (_lock, rec, _p) = recording().await;
            let session = Arc::new(TracedSession {
                inner: InMemorySession::new("session"),
                fail_read: failure == Some("session-read"),
            });
            let calls = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
            let calls_in_callback = calls.clone();
            let fail_callback = failure == Some("input-callback");
            let agent = Agent::<()>::new("session")
                .with_model_instance(scripted(vec![vec![assistant_message("done")]], false));
            let config = RunConfig::new()
                .with_tracing_disabled(disabled)
                .with_session_input_callback(move |history, new_input| {
                    let calls = calls_in_callback.clone();
                    async move {
                        custom_span("input-callback", json!({}))
                            .run(async {
                                calls.lock().unwrap().push("input-callback".into());
                            })
                            .await;
                        if fail_callback {
                            return Err(AgentsError::other("synthetic input-callback failure"));
                        }
                        let mut out = history;
                        out.extend(new_input);
                        Ok(out)
                    }
                });

            let caller = trace("caller").build();
            let outcome: AgentsResult<()> = caller
                .clone()
                .run(async {
                    let caller_span = custom_span("caller-parent", json!({}));
                    caller_span
                        .clone()
                        .run(async {
                            let result = Runner::run_with_options(
                                &agent,
                                "hi",
                                RunOptions::new(())
                                    .session(session.clone())
                                    .run_config(config.clone()),
                            )
                            .await;
                            assert_current_trace(&caller);
                            assert_current_span(&caller_span);
                            custom_span("after-preparation", json!({}))
                                .run(async {})
                                .await;
                            result.map(|_| ())
                        })
                        .await
                })
                .await;

            let ctx = format!("disabled={disabled} failure={failure:?}");
            match failure {
                None => {
                    outcome.unwrap();
                    let saved = session.inner.get_items(None).await.unwrap();
                    assert_eq!(
                        serde_json::to_value(&saved[0]).unwrap(),
                        json!({"role": "user", "content": "hi"}),
                        "{ctx}"
                    );
                }
                Some(text) => {
                    let err = outcome.expect_err(&ctx);
                    assert!(
                        err.to_string()
                            .contains(&format!("synthetic {text} failure")),
                        "{ctx}: {err}"
                    );
                    assert!(
                        session.inner.get_items(None).await.unwrap().is_empty(),
                        "{ctx}"
                    );
                }
            }
            let expected_calls: Vec<String> = match failure {
                Some("session-read") => vec!["session-read".into()],
                Some("input-callback") => vec!["session-read".into(), "input-callback".into()],
                _ => vec!["session-read".into(), "input-callback".into()],
            };
            // `session-read` is recorded by the session itself; the callback records its own call.
            let names = names_of(&rec);
            assert_eq!(
                rec.traces()
                    .iter()
                    .map(|t| t.trace_id().to_string())
                    .collect::<Vec<_>>(),
                vec![caller.trace_id().to_string()],
                "{ctx}"
            );
            if disabled {
                assert_eq!(names, vec!["caller-parent", "after-preparation"], "{ctx}");
            } else {
                for call in &expected_calls {
                    assert!(names.contains(call), "{ctx}: {call} missing from {names:?}");
                }
                let caller_id = id_named(&rec, "caller-parent");
                for span in rec.ordered_spans() {
                    let name = span
                        .export()
                        .and_then(|e| e["span_data"]["name"].as_str().map(str::to_string));
                    if name
                        .as_deref()
                        .is_some_and(|n| expected_calls.iter().any(|c| c == n))
                    {
                        assert_eq!(span.parent_id(), Some(caller_id.as_str()), "{ctx}");
                    }
                }
            }
            assert_eq!(
                rec.ordered_spans().last().unwrap().parent_id(),
                Some(id_named(&rec, "caller-parent").as_str()),
                "{ctx}"
            );
        }
    }
}

/// A provider whose resolution runs inside a custom span and optionally fails, as the Python
/// test's `Provider` does.
#[derive(Debug)]
struct ResolvingProvider {
    model: Arc<dyn Model>,
    resolutions: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    fail: bool,
}

impl ModelProvider for ResolvingProvider {
    fn get_model(&self, model_name: Option<&str>) -> AgentsResult<Arc<dyn Model>> {
        let span = custom_span("provider-resolution", json!({}));
        span.start();
        self.resolutions
            .lock()
            .unwrap()
            .push(model_name.map(str::to_string));
        let outcome = if self.fail {
            Err(AgentsError::other("synthetic provider failure"))
        } else {
            Ok(self.model.clone())
        };
        span.finish();
        outcome
    }
}

#[tokio::test]
async fn streamed_setup_tracing_opt_out_during_model_resolution() {
    for disabled in [false, true] {
        for provider_error in [false, true] {
            let (_lock, rec, _p) = recording().await;
            let resolutions = Arc::new(std::sync::Mutex::new(Vec::new()));
            // The Python test resolves a Chat Completions model: the output-guardrail check that
            // rejects server-managed history exempts it. The model answers from a local server.
            let endpoint = start_model_endpoint(vec![ModelReply::Sse(vec![
                json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
                    "model": "synthetic-model",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"},
                        "finish_reason": null}]}),
                json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
                    "model": "synthetic-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            ])]);
            let provider = Arc::new(ResolvingProvider {
                model: Arc::new(OpenAIChatCompletionsModel::new(
                    "synthetic-model",
                    OpenAIClient::new("sk-test").with_base_url(&endpoint.base_url),
                )),
                resolutions: resolutions.clone(),
                fail: provider_error,
            });
            let allow = OutputGuardrail::new("allow", |_c, _a, _o| async move {
                Ok(GuardrailFunctionOutput::new(json!(null), false))
            });
            let agent = Agent::<()>::new("provider")
                .with_model("synthetic-model")
                .with_output_guardrail(allow);
            let config = RunConfig::new()
                .with_tracing_disabled(disabled)
                .with_model_provider(provider.clone());

            let caller = trace("caller").build();
            caller
                .clone()
                .run(async {
                    let caller_span = custom_span("caller-parent", json!({}));
                    caller_span
                        .clone()
                        .run(async {
                            let mut result = Runner::run_streamed_with_options(
                                &agent,
                                "hi",
                                RunOptions::new(())
                                    .previous_response_id("response-id")
                                    .run_config(config.clone()),
                            );
                            assert_current_trace(&caller);
                            assert_current_span(&caller_span);
                            // Python's `run_streamed` raises here, before the caller's next line
                            // runs. Rust reports a setup failure through the stream instead, so the
                            // caller's line between the two is only reached on the success path.
                            if !provider_error {
                                custom_span("after-setup", json!({})).run(async {}).await;
                            }
                            let mut error = None;
                            {
                                let mut events = std::pin::pin!(result.stream_events());
                                while let Some(event) = events.next().await {
                                    if let Err(e) = event {
                                        error = Some(e);
                                        break;
                                    }
                                }
                            }
                            if provider_error {
                                let err = error.expect("provider failure surfaces");
                                assert!(
                                    err.to_string().contains("synthetic provider failure"),
                                    "{err}"
                                );
                            } else {
                                assert!(error.is_none(), "{error:?}");
                                assert_eq!(result.final_output_text().as_deref(), Some("done"));
                            }
                            assert_current_trace(&caller);
                            assert_current_span(&caller_span);
                            custom_span("after-run", json!({})).run(async {}).await;
                        })
                        .await;
                })
                .await;

            let ctx = format!("disabled={disabled} provider_error={provider_error}");
            // Python checks the single resolution made during setup. The Rust stream resolves the
            // model again while it runs, so only the name and a non-empty record are required.
            let resolved = resolutions.lock().unwrap().clone();
            assert!(!resolved.is_empty(), "{ctx}");
            assert!(
                resolved
                    .iter()
                    .all(|r| r.as_deref() == Some("synthetic-model")),
                "{ctx}: {resolved:?}"
            );
            let names = names_of(&rec);
            assert_eq!(
                rec.traces()
                    .iter()
                    .map(|t| t.trace_id().to_string())
                    .collect::<Vec<_>>(),
                vec![caller.trace_id().to_string()],
                "{ctx}"
            );
            if disabled {
                let expected: Vec<&str> = if provider_error {
                    vec!["caller-parent", "after-run"]
                } else {
                    vec!["caller-parent", "after-setup", "after-run"]
                };
                assert_eq!(names, expected, "{ctx}");
            } else {
                assert_eq!(
                    names.iter().filter(|n| *n == "provider-resolution").count(),
                    resolved.len(),
                    "{ctx}"
                );
            }
        }
    }
}
