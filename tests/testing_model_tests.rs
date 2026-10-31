//! `ScriptedModel`: strict step matching, the script-error hierarchy and the synthesized
//! Responses stream. Mirrors the behaviour of `agents.testing.model`.

use std::sync::Arc;

use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    DEFAULT_SCRIPTED_RESPONSE_ID, ModelScriptError, ModelStepReason, ScriptedModel, ScriptedStep,
    assistant_message, assistant_message_with_id, function_call_full, refusal_message,
};
use openai_agents_sdk::{Agent, AgentsError, Runner};
use serde_json::{Value, json};

fn event_types(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .map(|e| e["type"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[tokio::test]
async fn strict_matching_reports_unexpected_calls_and_leftovers() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            ScriptedStep::output(vec![assistant_message("one")]),
            ScriptedStep::output(vec![assistant_message("two")]),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());

    let result = Runner::run(&agent, "hi").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("one"));
    assert_eq!(
        result.raw_responses[0].response_id.as_deref(),
        Some(DEFAULT_SCRIPTED_RESPONSE_ID)
    );

    // A step is left over, so the script is not complete.
    match model.assert_complete() {
        Err(ModelScriptError::UnconsumedModelSteps { remaining_steps }) => {
            assert_eq!(remaining_steps, 1);
        }
        other => panic!("{other:?}"),
    }

    Runner::run(&agent, "again").await.unwrap();
    model.assert_complete().unwrap();

    // One call too many: the call is still recorded, then the run fails.
    let err = Runner::run(&agent, "third").await.unwrap_err().to_string();
    assert!(err.contains("no scripted steps remain"), "{err}");
    assert_eq!(model.calls().len(), 3);
    assert!(
        model
            .first_call()
            .unwrap()
            .request
            .conversation_id
            .is_none()
    );
    assert!(!model.last_call().unwrap().streamed);
}

#[tokio::test]
async fn conflicting_outcomes_are_rejected_before_queueing() {
    let mut step = ScriptedStep::raise_error(AgentsError::other("boom"));
    step.stream_events = Some(Vec::new());
    let model = ScriptedModel::new();
    match model.enqueue(step) {
        Err(ModelScriptError::InvalidModelStep { reason, .. }) => {
            assert_eq!(reason, ModelStepReason::ConflictingOutcomes);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(model.remaining_steps(), 0, "nothing was queued");

    // Retry advice without an error has nothing to attach to.
    let orphan = ScriptedStep::output(vec![]).with_retry_advice(Default::default());
    match model.enqueue(orphan) {
        Err(ModelScriptError::InvalidModelStep { reason, .. }) => {
            assert_eq!(reason, ModelStepReason::InvalidRetryAdvice);
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn responders_see_the_recorded_call() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::respond(|call| {
            assert!(!call.streamed);
            let text = call.request.input[0].text_content().unwrap_or_default();
            Ok(ScriptedStep::output(vec![assistant_message(format!(
                "saw: {text}"
            ))]))
        })])
        .unwrap(),
    );
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    let result = Runner::run(&agent, "ping").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("saw: ping"));
}

#[tokio::test]
async fn synthesized_stream_covers_every_event_type() {
    use futures::StreamExt;
    use openai_agents_sdk::models::interface::{Model, ModelRequest};

    openai_agents_sdk::set_tracing_disabled(true);
    let mut annotated = assistant_message_with_id("hello", "msg_1");
    if let openai_agents_sdk::items::OutputItem::Message(m) = &mut annotated
        && let openai_agents_sdk::items::MessageContent::OutputText { annotations, .. } =
            &mut m.content[0]
    {
        annotations.push(json!({"type": "url_citation", "url": "https://x"}));
    }
    let reasoning = openai_agents_sdk::items::OutputItem::from_value(json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "thinking"}],
        "content": [{"type": "reasoning_text", "text": "because"}],
    }));
    let model = ScriptedModel::with_steps(vec![ScriptedStep::output(vec![
        reasoning,
        annotated,
        refusal_message("nope"),
        function_call_full("add", "{\"a\":1}", "call_1", None, Some("tools".into())),
    ])])
    .unwrap();

    let events: Vec<Value> = model
        .stream_response(ModelRequest {
            system_instructions: None,
            input: vec![],
            model_settings: Default::default(),
            tools: vec![],
            output_schema: None,
            handoffs: vec![],
            tracing: Default::default(),
            previous_response_id: None,
            conversation_id: None,
            prompt: None,
        })
        .await
        .unwrap()
        .map(|e| e.unwrap())
        .collect()
        .await;

    assert_eq!(
        event_types(&events),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.annotation.added",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.refusal.delta",
            "response.refusal.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    // Sequence numbers are monotonic from zero.
    let sequence: Vec<u64> = events
        .iter()
        .map(|e| e["sequence_number"].as_u64().unwrap())
        .collect();
    assert_eq!(sequence, (0..events.len() as u64).collect::<Vec<_>>());
    let by_type = |name: &str| {
        events
            .iter()
            .find(|e| e["type"] == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .clone()
    };
    assert_eq!(
        by_type("response.reasoning_summary_text.done")["text"],
        "thinking"
    );
    assert_eq!(by_type("response.reasoning_text.done")["text"], "because");
    assert_eq!(by_type("response.output_text.delta")["delta"], "hello");
    assert_eq!(by_type("response.output_text.delta")["item_id"], "msg_1");
    assert_eq!(
        by_type("response.output_text.annotation.added")["annotation"]["url"],
        "https://x"
    );
    assert_eq!(by_type("response.refusal.done")["refusal"], "nope");
    assert_eq!(
        by_type("response.function_call_arguments.done")["name"],
        "add"
    );
    assert_eq!(
        by_type("response.function_call_arguments.done")["item_id"],
        "call_1"
    );
    // `output_item.added` shows the item before it is filled in.
    let added: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "response.output_item.added")
        .collect();
    assert_eq!(added[1]["item"]["content"], json!([]));
    assert_eq!(added[1]["item"]["status"], "in_progress");
    assert_eq!(added[0]["item"]["summary"], json!([]));
}

#[tokio::test]
async fn the_runner_consumes_the_synthesized_stream() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(
        ScriptedModel::with_steps(vec![ScriptedStep::output(vec![assistant_message_with_id(
            "streamed answer",
            "msg_1",
        )])])
        .unwrap(),
    );
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());

    let mut result = Runner::run_streamed(&agent, "hi");
    let mut names = Vec::new();
    let mut deltas = String::new();
    {
        let mut events = std::pin::pin!(result.stream_events());
        use futures::StreamExt;
        while let Some(event) = events.next().await {
            match event.unwrap() {
                StreamEvent::RunItem { name, .. } => names.push(name),
                StreamEvent::RawResponse(raw) => {
                    if raw["type"] == "response.output_text.delta" {
                        deltas.push_str(raw["delta"].as_str().unwrap_or_default());
                    }
                }
                StreamEvent::AgentUpdated { .. } => {}
            }
        }
    }
    assert_eq!(names, [RunItemStreamEventName::MessageOutputCreated]);
    assert_eq!(deltas, "streamed answer");
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("streamed answer")
    );
    assert!(model.last_call().unwrap().streamed);
}

#[tokio::test]
async fn exact_stream_events_bypass_synthesis() {
    use futures::StreamExt;
    use openai_agents_sdk::models::interface::{Model, ModelRequest};

    openai_agents_sdk::set_tracing_disabled(true);
    let scripted = vec![json!({"type": "response.created", "response": {"id": "r"}})];
    let model = ScriptedModel::with_steps(vec![ScriptedStep::stream(scripted.clone())]).unwrap();
    let events: Vec<Value> = model
        .stream_response(ModelRequest {
            system_instructions: None,
            input: vec![],
            model_settings: Default::default(),
            tools: vec![],
            output_schema: None,
            handoffs: vec![],
            tracing: Default::default(),
            previous_response_id: None,
            conversation_id: None,
            prompt: None,
        })
        .await
        .unwrap()
        .map(|e| e.unwrap())
        .collect()
        .await;
    assert_eq!(events, scripted);
}
