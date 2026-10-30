//! Run-scoped model lifecycle: the run owner a model sees, the end-of-run cleanup, and the
//! stream events a failing stream still delivers before its error.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::{ModelResponse, OutputItem};
use openai_agents_sdk::models::interface::{Model, ModelRequest, ResponseEventStream};
use openai_agents_sdk::models::response_terminal::{
    should_drain_stream_events_before_raising, terminal_failure_error,
};
use openai_agents_sdk::models::run_context::{ModelRunOwner, get_model_run_owner};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::usage::Usage;
use openai_agents_sdk::{Agent, AgentsError, Runner};
use serde_json::{Value, json};

/// Records the run owner installed around each model call and each cleanup notification.
#[derive(Debug, Default)]
struct OwnerSpy {
    owners_seen: Mutex<Vec<Option<ModelRunOwner>>>,
    cleanups: Mutex<Vec<ModelRunOwner>>,
}

#[derive(Debug)]
struct SpyModel {
    spy: Arc<OwnerSpy>,
    text: String,
}

fn response(text: &str) -> ModelResponse {
    ModelResponse {
        output: vec![assistant_message(text)],
        usage: Usage::single(1, 1),
        response_id: Some("resp_spy".into()),
        request_id: None,
        raw_usage: None,
    }
}

#[async_trait]
impl Model for SpyModel {
    async fn get_response(&self, _request: ModelRequest) -> Result<ModelResponse, AgentsError> {
        self.spy
            .owners_seen
            .lock()
            .unwrap()
            .push(get_model_run_owner());
        Ok(response(&self.text))
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> Result<ResponseEventStream, AgentsError> {
        self.spy
            .owners_seen
            .lock()
            .unwrap()
            .push(get_model_run_owner());
        let r = response(&self.text);
        let events = vec![Ok(json!({
            "type": "response.completed",
            "response": {
                "id": "resp_spy",
                "output": r.output.iter().map(OutputItem::to_value).collect::<Vec<_>>(),
            },
        }))];
        Ok(futures::stream::iter(events).boxed())
    }

    async fn cleanup_on_run_end(&self, owner: ModelRunOwner) {
        self.spy.cleanups.lock().unwrap().push(owner);
    }
}

#[tokio::test]
async fn every_model_call_runs_under_the_run_owner_and_is_cleaned_up_once() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let spy = Arc::new(OwnerSpy::default());
    let model = Arc::new(SpyModel {
        spy: spy.clone(),
        text: "done".into(),
    });
    let agent = Agent::<()>::new("A").with_model_instance(model.clone() as Arc<dyn Model>);

    Runner::run(&agent, "hi").await.unwrap();
    let first_owner = {
        let seen = spy.owners_seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        seen[0].expect("the model call must see a run owner")
    };
    let cleanups = spy.cleanups.lock().unwrap().clone();
    assert_eq!(
        cleanups,
        vec![first_owner],
        "the run ends by telling the model its own owner, exactly once"
    );
    drop(cleanups);

    // A second run of the same model gets its own owner, so per-run state cannot be released
    // by an unrelated run.
    Runner::run(&agent, "again").await.unwrap();
    let seen = spy.owners_seen.lock().unwrap().clone();
    let second_owner = seen[1].expect("owner");
    assert_ne!(first_owner, second_owner);
    assert_eq!(
        *spy.cleanups.lock().unwrap(),
        vec![first_owner, second_owner]
    );
}

#[tokio::test]
async fn a_streamed_run_cleans_up_its_model_and_scopes_the_stream_to_the_owner() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let spy = Arc::new(OwnerSpy::default());
    let model = Arc::new(SpyModel {
        spy: spy.clone(),
        text: "streamed".into(),
    });
    let agent = Agent::<()>::new("A").with_model_instance(model.clone() as Arc<dyn Model>);

    let mut stream = Runner::run_streamed(&agent, "hi");
    {
        let mut events = Box::pin(stream.stream_events());
        while let Some(ev) = events.next().await {
            ev.unwrap();
        }
    }
    let owner = spy.owners_seen.lock().unwrap()[0].expect("owner");
    assert_eq!(*spy.cleanups.lock().unwrap(), vec![owner]);
}

/// A model whose stream emits output and then fails with a terminal event.
#[derive(Debug)]
struct FailingStreamModel {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Model for FailingStreamModel {
    async fn get_response(&self, _request: ModelRequest) -> Result<ModelResponse, AgentsError> {
        unreachable!("this test only streams")
    }

    async fn stream_response(
        &self,
        _request: ModelRequest,
    ) -> Result<ResponseEventStream, AgentsError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let failure = json!({
            "type": "response.failed",
            "response": {"id": "resp_bad", "error": {"message": "upstream exploded"}},
        });
        let events: Vec<Result<Value, AgentsError>> = vec![
            Ok(json!({"type": "response.created", "response": {"id": "resp_bad"}})),
            Ok(json!({"type": "response.output_text.delta", "delta": "partial"})),
            Err(terminal_failure_error("response.failed", &failure)),
        ];
        Ok(futures::stream::iter(events).boxed())
    }
}

#[tokio::test]
async fn a_terminal_stream_failure_delivers_its_queued_events_before_the_error() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FailingStreamModel {
        calls: Arc::new(AtomicUsize::new(0)),
    });
    let agent = Agent::<()>::new("A").with_model_instance(model.clone() as Arc<dyn Model>);

    let mut stream = Runner::run_streamed(&agent, "hi");
    let mut raw_types: Vec<String> = Vec::new();
    let mut error: Option<AgentsError> = None;
    {
        let mut events = Box::pin(stream.stream_events());
        while let Some(ev) = events.next().await {
            match ev {
                Ok(StreamEvent::RawResponse(v)) => raw_types.push(
                    v.get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                ),
                Ok(_) => {}
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
    }

    let error = error.expect("the terminal failure reaches the consumer");
    // The error is the drainable kind, and the events the model produced before it failed
    // were all delivered first. Port of `_should_drain_stream_events_before_raising`:
    // the Rust stream carries events and the terminal error over one ordered channel, so a
    // drainable error can never overtake the output already produced.
    assert!(
        should_drain_stream_events_before_raising(&error),
        "{error:?}"
    );
    assert!(error.to_string().contains("response.failed"), "{error}");
    assert_eq!(
        raw_types,
        vec![
            "response.created".to_string(),
            "response.output_text.delta".to_string()
        ]
    );
    assert!(stream.run_loop_error().is_some());
}

#[tokio::test]
async fn max_turns_exceeded_also_delivers_the_events_of_the_last_turn() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(openai_agents_sdk::testing::FakeModel::new());
    model.add_turn(vec![openai_agents_sdk::testing::function_call(
        "noop", "{}",
    )]);
    let tool =
        openai_agents_sdk::tool::FunctionTool::<()>::new_no_args("noop", "Noop", |_c| async {
            Ok("ok")
        });
    let agent = Agent::<()>::new("A")
        .with_tool(tool)
        .with_model_instance(model.clone() as Arc<dyn Model>);

    let mut stream = Runner::run_streamed_with_options(
        &agent,
        "hi",
        openai_agents_sdk::RunOptions::new(()).max_turns(Some(1)),
    );

    let mut item_events = 0usize;
    let mut error: Option<AgentsError> = None;
    {
        let mut events = Box::pin(stream.stream_events());
        while let Some(ev) = events.next().await {
            match ev {
                Ok(StreamEvent::RunItem { .. }) => item_events += 1,
                Ok(_) => {}
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
    }
    assert!(matches!(error, Some(AgentsError::MaxTurnsExceeded { .. })));
    assert!(
        item_events > 0,
        "the turn's items are delivered before the max-turns error"
    );
}
