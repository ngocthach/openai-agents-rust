//! `test_run_streamed_resume_omits_tool_output_in_log_when_dont_log`: resuming a streamed run
//! from a state that already holds a tool output must not write that output to the logs while
//! tool data logging is off.

use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, Runner};
use serde_json::json;

#[derive(Default)]
struct Collect(Mutex<Vec<String>>);

impl log::Log for Collect {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        self.0.lock().unwrap().push(record.args().to_string());
    }
    fn flush(&self) {}
}

static SINK: OnceLock<Collect> = OnceLock::new();

#[tokio::test]
async fn a_streamed_resume_does_not_log_tool_output_when_tool_data_logging_is_off() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    openai_agents_sdk::debug::set_dont_log_tool_data(true);
    let sink: &'static Collect = SINK.get_or_init(Collect::default);
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);

    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("ok")]))
        .unwrap();
    let agent = Agent::<()>::new("log-agent").with_model_instance(model);
    let mut wire = RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "log-agent".into(),
        max_turns: Some(1),
        context: Some(json!({})),
        ..Default::default()
    }
    .to_json()
    .unwrap();
    wire["generated_items"] = json!([{
        "type": "tool_call_output_item", "agent": {"name": "log-agent"},
        "raw_item": {"type": "function_call_output", "call_id": "call-1", "output": "secret"},
        "output": "secret",
    }]);
    let state = RunState::from_json(wire).unwrap();

    let mut streamed = Runner::run_streamed(&agent, state);
    let mut events = std::pin::pin!(streamed.stream_events());
    while let Some(event) = events.next().await {
        event.unwrap();
    }

    let logged = sink.0.lock().unwrap().join("\n");
    assert!(!logged.contains("secret"), "{logged}");
}
