//! Parity port of `test_runner_resume_does_not_replay_partial_repeated_history` and
//! `test_runner_resume_handles_periodic_history_for_session_atomicity` from
//! `tests/memory/test_openai_conversations_session.py`: a run whose history append fails part
//! way leaves a pending-write checkpoint, and a resume must neither replay the part that landed
//! nor guess about a partial append whose tail cannot be told from older, repeated history.
//!
//! The Conversations API is a local mock; a failed non-streamed run hands its checkpoint to the
//! caller through `RunCheckpoint`.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::guardrail::{GuardrailFunctionOutput, OutputGuardrail};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::memory::{OpenAIConversationsSession, Session};
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, assistant_message_with_id, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, serve};

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn message_json(id: &str, text: &str) -> Value {
    json!({
        "id": id, "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

/// A Conversations mock whose items list is `stored`; `on_write(writes_so_far, batch)` returns
/// `(append, fail)` for each items-create request.
struct Backend {
    client: OpenAIClient,
    stored: Arc<Mutex<Vec<Value>>>,
    writes: Arc<Mutex<Vec<Vec<Value>>>>,
}

async fn backend(
    stored: Vec<Value>,
    on_write: impl Fn(usize) -> (bool, bool) + Send + Sync + 'static,
) -> Backend {
    let stored = Arc::new(Mutex::new(stored));
    let writes = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));
    let (data, log) = (stored.clone(), writes.clone());
    let base = serve(Arc::new(move |req: &Request| {
        if req.method == "GET" {
            let mut items = data.lock().unwrap().clone();
            if req.path.contains("order=desc") {
                items.reverse();
            }
            return Reply::json(json!({"object": "list", "data": items, "has_more": false}));
        }
        let batch: Vec<Value> = req.body["items"].as_array().cloned().unwrap_or_default();
        let count = {
            let mut log = log.lock().unwrap();
            log.push(batch.clone());
            log.len()
        };
        let (append, fail) = on_write(count);
        if append {
            let mut data = data.lock().unwrap();
            for item in batch {
                let mut item = item;
                item["id"] = json!(format!("item_{}", data.len()));
                data.push(item);
            }
        }
        if fail {
            return Reply::status(
                400,
                json!({"error": {"message": "synthetic failure", "type": "invalid_request_error"}})
                    .to_string(),
            );
        }
        Reply::json(json!({"object": "list", "data": [], "has_more": false}))
    }))
    .await;
    Backend {
        client: OpenAIClient::new("test-placeholder").with_base_url(format!("{base}/v1")),
        stored,
        writes,
    }
}

fn ok_guardrail() -> OutputGuardrail<()> {
    OutputGuardrail::new("gate", |_c, _a, _o: Value| async {
        Ok(GuardrailFunctionOutput::new(Value::Null, false))
    })
}

fn text_of(item: &Value) -> String {
    item["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

fn lens(writes: &Arc<Mutex<Vec<Vec<Value>>>>) -> Vec<usize> {
    writes.lock().unwrap().iter().map(Vec::len).collect()
}

async fn drive(
    agent: &Agent<()>,
    value: impl Into<openai_agents_sdk::items::RunInput>,
    session: &Arc<dyn Session>,
    stream: bool,
    checkpoint: &RunCheckpoint,
) -> Result<Option<String>, (AgentsError, Option<RunState>)> {
    let options = RunOptions::new(())
        .session(session.clone())
        .checkpoint(checkpoint);
    if stream {
        let mut result = Runner::run_streamed_with_options(agent, value, options);
        let mut error = None;
        {
            let events = result.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                    break;
                }
            }
        }
        return match error {
            Some(e) => Err((e, result.to_state())),
            None => Ok(result.final_output_text()),
        };
    }
    match Runner::run_with_options(agent, value, options).await {
        Ok(result) => Ok(result.final_output_text()),
        Err(e) => Err((e, checkpoint.take())),
    }
}

// test_runner_resume_does_not_replay_partial_repeated_history (2 x 4)
async fn repeated_history_case(stream: bool, history_size: usize, failure: &'static str) {
    let mut outputs: Vec<OutputItem> = (0..19)
        .map(|i| assistant_message_with_id(format!("message {i}"), format!("msg_{i}")))
        .collect();
    outputs.push(assistant_message_with_id("S", "msg_19"));
    let old: Vec<Value> = (0..history_size - 1)
        .map(|i| {
            message_json(&format!("old_{i}"), &{
                let k = i % 20;
                if k == 19 {
                    "S".to_string()
                } else {
                    format!("message {k}")
                }
            })
        })
        .collect();
    let service = backend(old, move |count| {
        let fail = count == if failure == "before" { 2 } else { 3 };
        (!fail || failure == "lost-ack", fail)
    })
    .await;
    let effects = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let log = effects.clone();
    let lookup = FunctionTool::<()>::new("lookup", "lookup", move |_c, _a: NoArgs| {
        let log = log.clone();
        async move {
            log.lock().unwrap().push("lookup");
            Ok("found".to_string())
        }
    })
    .with_needs_approval(true);
    let finish = FunctionTool::<()>::new("finish", "finish", |_c, _a: NoArgs| async {
        Ok("done".to_string())
    });
    let model = Arc::new(FakeModel::new());
    let mut first_turn = outputs.clone();
    first_turn.push(function_call_with_id("lookup", "{}", "lookup_1"));
    model.add_turn(first_turn);
    model.add_turn(vec![function_call_with_id("finish", "{}", "finish_1")]);
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(lookup)
        .with_tool(finish)
        .with_output_guardrail(ok_guardrail())
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["finish".into()],
        }));
    let session: Arc<dyn Session> = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        service.client.clone(),
    ));
    let label = format!("stream={stream} size={history_size} {failure}");

    let original = InputItem::from_value(json!({
        "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "S", "annotations": []}],
    }));
    let checkpoint = RunCheckpoint::new();
    // Park on the approval; the run itself is a plain run.
    let paused = Runner::run_with_options(
        &agent,
        vec![original],
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(
        service.stored.lock().unwrap().len(),
        history_size,
        "{label}"
    );
    let mut state = paused.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let (error, failed_state) = match drive(&agent, state, &session, stream, &checkpoint).await {
        Ok(_) => panic!("{label}: the resumed append must fail"),
        Err(failure) => failure,
    };
    assert!(
        error.to_string().contains("synthetic failure"),
        "{label}: {error}"
    );
    assert_eq!(*effects.lock().unwrap(), ["lookup"], "{label}");
    assert_eq!(model.calls().len(), 1, "{label}");
    let state = RunState::from_json(
        failed_state
            .expect("a failed run leaves a checkpoint")
            .to_json()
            .unwrap(),
    )
    .unwrap();

    if failure == "partial" {
        assert_eq!(
            service.stored.lock().unwrap().len(),
            history_size + 20,
            "{label}"
        );
        let snapshot = service.stored.lock().unwrap().clone();
        let again = drive(&agent, state, &session, false, &RunCheckpoint::new()).await;
        let error = again
            .expect_err(&format!("{label}: a partial append cannot be reconciled"))
            .0;
        assert!(
            error
                .to_string()
                .contains("Cannot reconcile the pending Session write"),
            "{label}: {error}"
        );
        assert_eq!(*service.stored.lock().unwrap(), snapshot, "{label}");
        assert_eq!(lens(&service.writes), [1, 20, 2], "{label}");
        assert_eq!(model.calls().len(), 1, "{label}");
    } else {
        let result = drive(&agent, state, &session, stream, &RunCheckpoint::new())
            .await
            .unwrap_or_else(|(e, _)| panic!("{label}: {e}"));
        assert_eq!(result.as_deref(), Some("done"), "{label}");
        let texts: Vec<String> = service
            .stored
            .lock()
            .unwrap()
            .iter()
            .filter(|i| i["type"] == "message")
            .map(text_of)
            .collect();
        let mut expected = vec!["S".to_string()];
        expected.extend((0..19).map(|i| format!("message {i}")));
        expected.push("S".to_string());
        assert_eq!(texts, expected, "{label}");
        assert_eq!(model.calls().len(), 2, "{label}");
    }
    assert_eq!(*effects.lock().unwrap(), ["lookup"], "{label}");
}

#[tokio::test]
async fn runner_resume_does_not_replay_partial_repeated_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    for stream in [false, true] {
        for (history_size, failure) in [
            (1, "partial"),
            (40, "partial"),
            (1, "before"),
            (1, "lost-ack"),
        ] {
            repeated_history_case(stream, history_size, failure).await;
        }
    }
}

/// Python's `LostAckSession`: an id-stripping local store whose second append lands but
/// reports a failure, so the caller cannot tell whether the batch was written.
#[derive(Debug)]
struct LostAckSession {
    inner: py_mcpmem_support::ListSession,
    writes: Arc<Mutex<Vec<Vec<Value>>>>,
}

#[async_trait::async_trait]
impl Session for LostAckSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn ignores_item_ids_for_matching(&self) -> bool {
        true
    }
    async fn get_items(
        &self,
        limit: Option<usize>,
    ) -> openai_agents_sdk::error::Result<Vec<InputItem>> {
        self.inner.get_items(limit).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::error::Result<()> {
        let count = {
            let mut writes = self.writes.lock().unwrap();
            writes.push(items.iter().map(|i| i.as_value().clone()).collect());
            writes.len()
        };
        self.inner.add_items(items).await?;
        if count == 2 {
            return Err(AgentsError::other("synthetic failure"));
        }
        Ok(())
    }
    async fn pop_item(&self) -> openai_agents_sdk::error::Result<Option<InputItem>> {
        self.inner.pop_item().await
    }
    async fn clear_session(&self) -> openai_agents_sdk::error::Result<()> {
        self.inner.clear_session().await
    }
}

// test_runner_resume_handles_periodic_history_for_session_atomicity (5 cases)
async fn periodic_history_case(atomic: bool, output_count: usize, history_size: usize) {
    let label = format!("atomic={atomic} outputs={output_count} history={history_size}");
    let letter = |i: usize| ["A", "B", "C"][i % 3].to_string();
    let old: Vec<Value> = (0..history_size - 1)
        .map(|i| message_json(&format!("old_{i}"), &letter(i)))
        .collect();
    let service = backend(old.clone(), move |count| {
        let fail = count == if output_count <= 20 { 2 } else { 3 };
        (!fail || output_count <= 20 || history_size == 1, fail)
    })
    .await;
    let (session, writes): (Arc<dyn Session>, Arc<Mutex<Vec<Vec<Value>>>>) = if atomic {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let history: Vec<InputItem> = old
            .iter()
            .cloned()
            .map(|mut item| {
                item.as_object_mut().unwrap().remove("id");
                InputItem::from_value(item)
            })
            .collect();
        (
            Arc::new(LostAckSession {
                inner: py_mcpmem_support::ListSession::with_history("atomic", history),
                writes: writes.clone(),
            }),
            writes,
        )
    } else {
        (
            Arc::new(OpenAIConversationsSession::from_conversation_id(
                "conv_test",
                service.client.clone(),
            )),
            service.writes.clone(),
        )
    };
    let mut first_turn: Vec<OutputItem> = (0..output_count)
        .map(|i| assistant_message_with_id(letter(i), format!("msg_{i}")))
        .collect();
    first_turn.push(function_call_with_id(
        "transfer_to_delegate",
        "{}",
        "handoff_1",
    ));
    let model = Arc::new(FakeModel::new());
    model.add_turn(first_turn);
    model.add_turn(vec![assistant_message("done")]);
    let delegate = Agent::<()>::new("delegate").with_model_instance(model.clone());
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_handoff(
            openai_agents_sdk::handoffs::handoff(delegate).with_input_filter(|mut data| {
                data.new_items.retain(|item| {
                    matches!(
                        item.kind,
                        openai_agents_sdk::items::RunItemKind::MessageOutput { .. }
                    )
                });
                data
            }),
        );
    let original = InputItem::from_value(json!({
        "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": letter(29), "annotations": []}],
    }));
    let (error, failed_state) = match drive(
        &agent,
        vec![original],
        &session,
        true,
        &RunCheckpoint::new(),
    )
    .await
    {
        Ok(_) => panic!("{label}: the append must fail"),
        Err(failure) => failure,
    };
    assert!(
        error.to_string().contains("synthetic failure"),
        "{label}: {error}"
    );
    let snapshot = session.get_items(None).await.unwrap();
    let complete = atomic || output_count <= 20 || history_size == 1;
    assert_eq!(
        snapshot.len(),
        if complete {
            history_size + output_count
        } else {
            history_size + 20
        },
        "{label}"
    );
    let expected_writes = if atomic || output_count <= 20 {
        vec![1, output_count]
    } else {
        vec![1, 20, output_count - 20]
    };
    assert_eq!(lens(&writes), expected_writes, "{label}");
    let state = RunState::from_json(failed_state.expect("checkpoint").to_json().unwrap()).unwrap();
    if complete {
        let result = drive(&agent, state, &session, false, &RunCheckpoint::new())
            .await
            .unwrap_or_else(|(e, _)| panic!("{label}: {e}"));
        assert_eq!(result.as_deref(), Some("done"), "{label}");
        let history = session.get_items(None).await.unwrap();
        assert_eq!(history[..history.len() - 1], snapshot[..], "{label}");
        assert_eq!(history.len(), history_size + output_count + 1, "{label}");
        let mut expected_writes = if atomic || output_count <= 20 {
            vec![1, output_count]
        } else {
            vec![1, 20, output_count - 20]
        };
        expected_writes.push(1);
        assert_eq!(lens(&writes), expected_writes, "{label}");
        assert_eq!(model.calls().len(), 2, "{label}");
    } else {
        let again = drive(
            &agent,
            state.clone(),
            &session,
            false,
            &RunCheckpoint::new(),
        )
        .await;
        let error = again
            .expect_err(&format!("{label}: a partial append cannot be reconciled"))
            .0;
        assert!(
            error
                .to_string()
                .contains("Cannot reconcile the pending Session write"),
            "{label}: {error}"
        );
        assert_eq!(session.get_items(None).await.unwrap(), snapshot, "{label}");
        assert_eq!(writes.lock().unwrap().len(), 3, "{label}");
        assert_eq!(model.calls().len(), 1, "{label}");
        // Missing messages remain pending rather than being discarded.
        let pending = state.to_json().unwrap();
        assert_eq!(
            pending["pending_session_write"]["items"]
                .as_array()
                .map(Vec::len),
            Some(23),
            "{label}"
        );
    }
}

#[tokio::test]
async fn runner_resume_handles_periodic_history_for_session_atomicity() {
    openai_agents_sdk::set_tracing_disabled(true);
    for (atomic, output_count, history_size) in [
        (false, 23, 30),
        (true, 23, 30),
        (false, 2, 30),
        (false, 23, 1),
        (false, 21, 1),
    ] {
        periodic_history_case(atomic, output_count, history_size).await;
    }
}

// test_runner_persists_program_item_ids: program items keep the id the Conversations
// create-item schema requires, while ids the schema leaves optional stay stripped.
#[derive(Deserialize, JsonSchema)]
struct Sku {
    sku: String,
}

#[tokio::test]
async fn runner_persists_program_item_ids() {
    openai_agents_sdk::set_tracing_disabled(true);
    let service = backend(Vec::new(), |_| (true, false)).await;
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![OutputItem::from_value(json!({
        "id": "program_item", "call_id": "call_program",
        "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint", "type": "program",
    }))]);
    model.add_turn(vec![
        OutputItem::from_value(json!({
            "id": "program_output_item", "call_id": "call_program",
            "result": "{\"sku\":\"A-1\",\"available_units\":42}", "status": "completed",
            "type": "program_output",
        })),
        assistant_message("done"),
    ]);
    let lookup =
        FunctionTool::<()>::new("lookup_inventory", "lookup", |_c, args: Sku| async move {
            Ok(args.sku)
        })
        .with_allowed_callers(vec!["programmatic".into()]);
    let agent = Agent::<()>::new("inventory")
        .with_model_instance(model)
        .with_tool(openai_agents_sdk::tool::ProgrammaticToolCallingTool)
        .with_tool(lookup);
    let session: Arc<dyn Session> = Arc::new(OpenAIConversationsSession::from_conversation_id(
        "conv_test",
        service.client.clone(),
    ));
    let result = Runner::run_with_options(
        &agent,
        "Check inventory",
        RunOptions::new(()).session(session),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let saved: Vec<Value> = service
        .writes
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .cloned()
        .collect();
    let by_type = |kind: &str| {
        saved
            .iter()
            .find(|i| i["type"] == kind)
            .cloned()
            .unwrap_or(Value::Null)
    };
    assert_eq!(by_type("program")["id"], "program_item");
    assert_eq!(by_type("program_output")["id"], "program_output_item");
    assert!(by_type("message").get("id").is_none());
}
