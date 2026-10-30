//! Recoverable handoff invocation failures, the server-managed `nest_handoff_history`
//! warning, and cancellation of an in-flight model call when a parallel input guardrail trips.
//!
//! The `log` subscriber is process-global, so these tests live in their own integration-test
//! binary.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::models::interface::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::default_tool_error_function;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, RunConfig, RunContext, RunOptions,
    Runner, handoff,
};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item_types(items: &[openai_agents_sdk::items::RunItem]) -> Vec<&str> {
    items.iter().map(|i| i.item_type()).collect()
}

#[tokio::test]
async fn a_failed_handoff_invocation_recovers_when_a_formatter_is_set() {
    let model = setup();
    let target = Arc::new(Agent::<()>::new("B").with_model_instance(model.clone()));
    let transfer = handoff(target)
        .with_on_handoff(|_ctx| async { Err(AgentsError::user("target is offline")) })
        .with_failure_error_function(default_tool_error_function);
    let agent = Agent::<()>::new("A")
        .with_handoff(transfer)
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_b", "{}")]);
    model.add_turn(vec![assistant_message("I will handle it myself")]);

    let r = Runner::run(&agent, "escalate").await.unwrap();
    assert_eq!(
        r.final_output_text().as_deref(),
        Some("I will handle it myself")
    );
    // The failed invocation is not a transfer: the calling agent keeps the turn.
    assert_eq!(r.last_agent.name, "A");
    assert_eq!(
        item_types(&r.new_items),
        vec![
            "handoff_call_item",
            "tool_call_output_item",
            "message_output_item"
        ]
    );
    let output = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        output.as_str(),
        Some("An error occurred while running the tool. Please try again.")
    );
    // The error text reaches the next model call as the handoff tool's output.
    let second = &model.calls()[1];
    assert!(
        second.input.iter().any(|i| {
            i.item_type() == Some("function_call_output")
                && i.0["output"].as_str() == output.as_str()
        }),
        "{:?}",
        second.input
    );
}

#[tokio::test]
async fn a_failed_handoff_invocation_still_ends_the_run_without_a_formatter() {
    let model = setup();
    let target = Arc::new(Agent::<()>::new("B").with_model_instance(model.clone()));
    let transfer =
        handoff(target).with_on_handoff(|_ctx| async { Err(AgentsError::user("offline")) });
    let agent = Agent::<()>::new("A")
        .with_handoff(transfer)
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_b", "{}")]);
    model.add_turn(vec![assistant_message("unreachable")]);

    let err = Runner::run(&agent, "escalate").await.unwrap_err();
    assert!(err.to_string().contains("offline"), "{err}");
}

#[tokio::test]
async fn a_custom_formatter_output_is_sent_to_the_calling_agent() {
    let model = setup();
    let target = Arc::new(Agent::<()>::new("B").with_model_instance(model.clone()));
    let transfer = handoff(target)
        .with_on_handoff(|_ctx| async { Err(AgentsError::user("queue full")) })
        .with_failure_error_function(|_ctx, e| format!("handoff unavailable: {e}"));
    let agent = Agent::<()>::new("A")
        .with_handoff(transfer)
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_b", "{}")]);
    model.add_turn(vec![assistant_message("ok")]);

    let r = Runner::run(&agent, "escalate").await.unwrap();
    let output = r
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .unwrap();
    assert!(
        output.as_str().unwrap().starts_with("handoff unavailable:"),
        "{output}"
    );
    assert!(
        r.new_items
            .iter()
            .all(|i| i.item_type() != "handoff_output_item")
    );
}

// ---------------------------------------------------------------------------------------------
// Server-managed conversations disable nested handoff history, with a warning
// ---------------------------------------------------------------------------------------------

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

fn install_logger() -> &'static Collect {
    let sink: &'static Collect = SINK.get_or_init(Collect::default);
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink.0.lock().unwrap().clear();
    sink
}

#[tokio::test]
async fn nest_handoff_history_under_a_server_managed_conversation_warns() {
    let sink = install_logger();
    let model = setup();
    let target = Arc::new(Agent::<()>::new("B").with_model_instance(model.clone()));
    let agent = Agent::<()>::new("A")
        .with_handoff(handoff(target))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_b", "{}")]);
    model.add_turn(vec![assistant_message("ok")]);

    let opts = RunOptions::new(())
        .conversation_id("conv_nest")
        .run_config(RunConfig::new().with_nest_handoff_history(true));
    let r = Runner::run_with_options(&agent, "question", opts)
        .await
        .unwrap();
    assert_eq!(r.last_agent.name, "B");

    let logs = sink.0.lock().unwrap().clone();
    assert!(
        logs.iter().any(|l| {
            l.contains("do not support nest_handoff_history for handoff A -> B")
                && l.contains("delta-only input")
        }),
        "{logs:?}"
    );
    // The nesting is dropped: the second call sends only the turn's new items.
    let second = &model.calls()[1];
    assert!(
        second.input.iter().all(|i| i
            .text_content()
            .map(|t| !t.contains("<CONVERSATION HISTORY>"))
            != Some(false)),
        "{:?}",
        second.input
    );
}

// ---------------------------------------------------------------------------------------------
// A tripped parallel input guardrail cancels the in-flight model call
// ---------------------------------------------------------------------------------------------

/// Records whether its in-flight call was dropped before it could answer.
#[derive(Debug, Default)]
struct SlowModel {
    started: AtomicUsize,
    completed: AtomicUsize,
    cancelled: Arc<AtomicBool>,
}

/// Flips the shared flag unless the call ran to completion.
struct CancelGuard {
    flag: Arc<AtomicBool>,
    finished: bool,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.flag.store(true, Ordering::SeqCst);
        }
    }
}

#[async_trait::async_trait]
impl Model for SlowModel {
    async fn get_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ModelResponse> {
        self.started.fetch_add(1, Ordering::SeqCst);
        let mut guard = CancelGuard {
            flag: self.cancelled.clone(),
            finished: false,
        };
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        guard.finished = true;
        self.completed.fetch_add(1, Ordering::SeqCst);
        Ok(ModelResponse {
            output: Vec::new(),
            usage: Default::default(),
            response_id: None,
            request_id: None,
            raw_usage: None,
        })
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> openai_agents_sdk::Result<ResponseEventStream> {
        unimplemented!("not used")
    }
}

#[tokio::test]
async fn a_tripped_parallel_guardrail_cancels_the_in_flight_model_call() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(SlowModel::default());
    // The verdict arrives after the model call is already in flight, so the trip has to
    // cancel it rather than simply win the race to start.
    let guardrail = InputGuardrail::<()>::new("block", |_c, _a, _i| async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok(GuardrailFunctionOutput::tripwire())
    });
    let agent = Agent::<()>::new("A")
        .with_input_guardrail(guardrail)
        .with_model_instance(model.clone());

    let err = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        Runner::run(&agent, "hello"),
    )
    .await
    .expect("the run must not wait for the model call it cancelled")
    .unwrap_err();
    assert!(matches!(
        err,
        AgentsError::InputGuardrailTripwireTriggered { .. }
    ));
    assert_eq!(model.started.load(Ordering::SeqCst), 1);
    assert_eq!(model.completed.load(Ordering::SeqCst), 0);
    assert!(
        model.cancelled.load(Ordering::SeqCst),
        "the model future must be dropped when the guardrail trips"
    );
}

#[tokio::test]
async fn a_blocking_guardrail_never_starts_the_model_call() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(SlowModel::default());
    let guardrail = InputGuardrail::<()>::new("block", |_c, _a, _i| async move {
        Ok(GuardrailFunctionOutput::tripwire())
    })
    .blocking();
    let agent = Agent::<()>::new("A")
        .with_input_guardrail(guardrail)
        .with_model_instance(model.clone());

    let err = Runner::run(&agent, "hello").await.unwrap_err();
    assert!(matches!(
        err,
        AgentsError::InputGuardrailTripwireTriggered { .. }
    ));
    assert_eq!(model.started.load(Ordering::SeqCst), 0);
}

/// Keeps `RunContext` in the imports honest: the formatter receives the run context.
#[tokio::test]
async fn the_failure_formatter_receives_the_run_context() {
    let model = setup();
    let seen = Arc::new(AtomicUsize::new(0));
    let s2 = seen.clone();
    let target = Arc::new(Agent::<usize>::new("B").with_model_instance(model.clone()));
    let transfer = handoff(target)
        .with_on_handoff(|_ctx| async { Err(AgentsError::user("down")) })
        .with_failure_error_function(move |ctx: &RunContext<usize>, _e| {
            s2.store(*ctx.context(), Ordering::SeqCst);
            "retry later".to_string()
        });
    let agent = Agent::<usize>::new("A")
        .with_handoff(transfer)
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_b", "{}")]);
    model.add_turn(vec![assistant_message("ok")]);

    Runner::run_with_options(&agent, "go", RunOptions::new(7usize))
        .await
        .unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 7);
}
