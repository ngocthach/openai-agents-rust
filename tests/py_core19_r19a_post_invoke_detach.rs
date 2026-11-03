//! `test_multiple_tool_calls_do_not_wait_indefinitely_for_sleeping_post_invoke_sibling`, second
//! half: once a sibling's failure has surfaced, a call still in its post-invoke phase is not
//! cancelled. Python leaves its task running detached; here it keeps running on its own task,
//! so its output guardrail can still finish after the run returned.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolOutputGuardrail, ToolOutputGuardrailData,
};
use openai_agents_sdk::{Agent, AgentsError, Runner};
use serde_json::json;
use tokio::sync::Notify;

#[derive(Default)]
struct Event {
    flag: AtomicBool,
    notify: Notify,
}

impl Event {
    fn new() -> Arc<Self> {
        Arc::new(Event::default())
    }
    fn set(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            futures::pin_mut!(notified);
            notified.as_mut().enable();
            if self.is_set() {
                return;
            }
            notified.await;
        }
    }
}

fn agent(started: &Arc<Event>, release: &Arc<Event>, finished: &Arc<Event>) -> Agent<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![
                function_call_full("ok_tool", "{}", "1", None, None),
                function_call_full("error_tool", "{}", "2", None, None),
            ]),
            ScriptedStep::output(vec![assistant_message("done")]),
        ])
        .unwrap(),
    );
    let (s, r, f) = (started.clone(), release.clone(), finished.clone());
    let ok_tool =
        FunctionTool::<()>::new_no_args("ok_tool", "ok", |_c| async { Ok("ok".to_string()) })
            .without_failure_error_function()
            .with_output_guardrail(ToolOutputGuardrail::new(
                "blocked_post_invoke_guardrail",
                move |_d: ToolOutputGuardrailData<()>| {
                    let (s, r, f) = (s.clone(), r.clone(), f.clone());
                    async move {
                        s.set();
                        // Only the test releases it, after the sibling error propagated.
                        r.wait().await;
                        f.set();
                        Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("done")))
                    }
                },
            ));
    let started = started.clone();
    let error_tool = FunctionTool::<()>::new_no_args("error_tool", "fails", move |_c| {
        let started = started.clone();
        async move {
            // Fail only once the sibling guardrail is in its post-invoke phase.
            started.wait().await;
            Err::<String, _>(AgentsError::other("boom"))
        }
    })
    .without_failure_error_function();
    Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(ok_tool)
        .with_tool(error_tool)
}

async fn run(agent: Agent<()>, streaming: bool) -> Result<(), AgentsError> {
    let options = RunOptions::default()
        .run_config(RunConfig::default().with_tool_post_invoke_wait(Duration::from_millis(50)));
    let outcome = tokio::time::timeout(Duration::from_secs(5), async {
        if streaming {
            let mut streamed = Runner::run_streamed_with_options(agent, "go", options);
            let events = streamed.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event?;
            }
            Ok(())
        } else {
            Runner::run_with_options(agent, "go", options)
                .await
                .map(|_| ())
        }
    })
    .await;
    outcome.expect("the run did not settle in time")
}

#[tokio::test]
async fn a_post_invoke_sibling_still_pending_at_the_failure_finishes_in_the_background() {
    for streaming in [false, true] {
        let (started, release, finished) = (Event::new(), Event::new(), Event::new());
        let outcome = run(agent(&started, &release, &finished), streaming).await;
        // Release the guardrail even when the assertion fails, so no task stays blocked.
        release.set();
        let err = outcome.unwrap_err();
        assert!(
            err.to_string()
                .contains("Error running tool error_tool: boom"),
            "streaming={streaming}: {err:?}"
        );
        tokio::time::timeout(Duration::from_secs(5), finished.wait())
            .await
            .unwrap_or_else(|_| panic!("streaming={streaming}: the sibling did not finish"));
    }
}

/// The same with a multi-threaded runtime, where the detached task runs on another worker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_detached_sibling_finishes_on_a_multi_threaded_runtime() {
    let (started, release, finished) = (Event::new(), Event::new(), Event::new());
    let outcome = run(agent(&started, &release, &finished), false).await;
    release.set();
    assert!(outcome.is_err());
    tokio::time::timeout(Duration::from_secs(5), finished.wait())
        .await
        .expect("the sibling did not finish");
}
