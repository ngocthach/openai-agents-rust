//! Port of `test_streamed_validated_items_precede_llm_end_hook_failure` in
//! `tests/test_tool_approval_call_id_reuse.py`: a streamed run shows the validated tool call of a
//! response before the `on_llm_end` hook that then fails the run, and runs no tool.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, Result, RunContext, Runner};

struct FailingEnd;

#[async_trait]
impl RunHooks<()> for FailingEnd {
    async fn try_on_llm_end(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _response: &ModelResponse,
    ) -> Result<()> {
        Err(AgentsError::user("llm end failed"))
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ValueArgs {
    value: String,
}

#[tokio::test]
async fn the_tool_called_event_precedes_a_failing_llm_end_hook() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let executed: Arc<Mutex<Vec<String>>> = Arc::default();
    let record = executed.clone();
    let tool = FunctionTool::new("record_value", "record", move |_c, a: ValueArgs| {
        let record = record.clone();
        async move {
            record.lock().unwrap().push(a.value.clone());
            Ok(a.value)
        }
    });
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "record_value",
        r#"{"value":"safe"}"#,
        "call_0",
    )]);
    let agent = Agent::<()>::new("agent")
        .with_tool(tool)
        .with_model_instance(model);

    let mut stream = Runner::run_streamed_with_options(
        &agent,
        "record a value",
        RunOptions::new(()).hooks(Arc::new(FailingEnd)),
    );
    let mut names = Vec::new();
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            match event {
                Ok(StreamEvent::RunItem { name, .. }) => names.push(name.as_str()),
                Ok(_) => {}
                Err(e) => error = Some(e),
            }
        }
    }

    assert!(
        error
            .expect("the run fails")
            .to_string()
            .contains("llm end failed")
    );
    assert_eq!(names, ["tool_called"]);
    assert!(executed.lock().unwrap().is_empty());
}
