//! Resume `RunState` snapshots produced by the real Python SDK.
//!
//! The fixtures under `tests/fixtures/python_run_state/` are written by
//! `tests/fixtures/python_run_state/generate.py`, which drives the Python `openai-agents`
//! package with `agents.testing.ScriptedModel`. This suite never shells out to Python: it
//! only reads the committed JSON.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run_state::{ContextRestoreOptions, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message, assistant_message_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner, handoff};
use serde_json::{Value, json};

fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/tests/fixtures/python_run_state/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn load(name: &str) -> RunState {
    RunState::from_json(fixture(name)).unwrap_or_else(|e| panic!("from_json {name}: {e}"))
}

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[test]
fn python_completed_run_state_round_trips() {
    let state = load("completed.json");
    assert_eq!(state.current_agent_name, "Weather Agent");
    assert_eq!(state.current_turn, 2);
    assert_eq!(state.max_turns, Some(10));
    assert_eq!(state.usage.requests, 2);
    assert_eq!(state.usage.input_tokens, 30);
    assert_eq!(state.usage.output_tokens, 12);
    assert_eq!(state.model_responses.len(), 2);
    assert_eq!(state.generated_items.len(), 3);
    assert!(state.interruptions.is_empty());
    // Python only persists the trace of a run that parked; a finished run has none.
    assert_eq!(state.trace_id, None);
    assert_eq!(state.trace_state, None);
    assert_eq!(state.last_processed_response, None);

    // The original input Python stored as a bare string survives as one user message.
    assert_eq!(
        state.original_input[0].text_content().as_deref(),
        Some("What is the weather in Hanoi?")
    );
}

#[tokio::test]
async fn python_completed_run_state_resumes_with_new_input() {
    let model = fake();
    model.add_turn(vec![assistant_message("Still sunny.")]);
    let agent = Agent::<()>::new("Weather Agent").with_model_instance(model.clone());

    let mut state = load("completed.json");
    state.add_input("And tomorrow?").unwrap();
    let result = Runner::run(&agent, state).await.unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("Still sunny."));
    // The Python turn's usage is carried into the resumed run.
    assert_eq!(result.usage().requests, 3);
    let call = model.last_call().unwrap();
    assert!(
        call.input
            .iter()
            .any(|i| i.text_content().as_deref() == Some("And tomorrow?"))
    );
}

#[tokio::test]
async fn python_interrupted_approval_resumes_and_runs_the_tool() {
    let state = load("interrupted_approval.json");
    assert_eq!(state.current_agent_name, "File Agent");
    assert_eq!(state.interruptions.len(), 1);
    assert_eq!(
        state.trace_id.as_deref(),
        Some("trace_0123456789abcdef0123456789abcd")
    );
    assert_eq!(
        state
            .trace_state
            .as_ref()
            .and_then(|t| t.workflow_name.as_deref()),
        Some("fixture-workflow")
    );
    assert_eq!(state.interruptions[0].name(), "delete_file");
    assert_eq!(state.interruptions[0].raw_item.call_id, "call_delete_1");
    assert!(state.pending_tool_processing);
    assert_eq!(
        state.tool_use_tracker.get("File Agent").map(Vec::as_slice),
        Some(["delete_file".to_string()].as_slice())
    );

    // Python wrote the processed response of the interrupted turn; Rust reads it back.
    let processed = state.last_processed_response.as_ref().expect("processed");
    assert_eq!(processed.tools_used, vec!["delete_file".to_string()]);
    assert_eq!(processed.functions.len(), 1);
    assert_eq!(processed.functions[0].tool_name, "delete_file");
    assert_eq!(
        processed.functions[0].tool_call["call_id"],
        json!("call_delete_1")
    );
    assert_eq!(processed.new_items.len(), 1);
    assert_eq!(processed.interruptions.len(), 1);

    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let tool = FunctionTool::<()>::new("delete_file", "Delete a file", move |_c, a: Value| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(format!("deleted {}", a["path"].as_str().unwrap_or("")))
        }
    })
    .with_needs_approval(true);

    let model = fake();
    model.add_turn(vec![assistant_message("Deleted the report.")]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let mut state = state;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let result = Runner::run(&agent, state).await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("Deleted the report.")
    );
    assert!(
        result
            .new_items
            .iter()
            .any(|i| i.tool_output() == Some(&json!("deleted /tmp/report.txt")))
    );
}

#[tokio::test]
async fn python_interrupted_approval_rejection_is_honoured() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let tool = FunctionTool::<()>::new("delete_file", "Delete a file", move |_c, _a: Value| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok("deleted".to_string())
        }
    })
    .with_needs_approval(true);

    let model = fake();
    model.add_turn(vec![assistant_message("Left it alone.")]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let mut state = load("interrupted_approval.json");
    let item = state.get_interruptions()[0].clone();
    state.reject(&item, false, Some("Not allowed".into()));
    let result = Runner::run(&agent, state).await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        result
            .new_items
            .iter()
            .any(|i| i.tool_output() == Some(&json!("Not allowed")))
    );
}

#[tokio::test]
async fn python_handoff_state_resumes_on_the_target_agent() {
    let state = load("handoff.json");
    assert_eq!(state.current_agent_name, "Spanish Agent");
    assert_eq!(state.generated_items.len(), 3);
    assert_eq!(
        state
            .generated_items
            .iter()
            .filter(|i| matches!(i.kind, RunItemKind::HandoffCall { .. }))
            .count(),
        1
    );

    let model = fake();
    model.add_turn(vec![assistant_message("Muy bien.")]);
    let spanish = Agent::<()>::new("Spanish Agent").with_model_instance(model.clone());
    let triage = Agent::<()>::new("Triage Agent")
        .with_handoff(handoff(spanish))
        .with_model_instance(model.clone());

    let mut state = state;
    state.add_input("Y tu?").unwrap();
    let result = Runner::run(&triage, state).await.unwrap();
    // The resumed run continues as the handoff target, not as the starting agent.
    assert_eq!(result.last_agent.name, "Spanish Agent");
    assert_eq!(result.final_output_text().as_deref(), Some("Muy bien."));
}

#[test]
fn python_usage_and_guardrail_results_survive_the_read() {
    let state = load("usage_guardrails.json");
    assert_eq!(state.usage.requests, 2);
    assert_eq!(state.usage.input_tokens, 75);
    assert_eq!(state.usage.output_tokens, 13);
    assert_eq!(state.usage.total_tokens, 88);

    assert_eq!(state.input_guardrail_results.len(), 1);
    assert_eq!(
        state.input_guardrail_results[0].guardrail_name,
        "length_guardrail"
    );
    assert!(!state.input_guardrail_results[0].output.tripwire_triggered);
    assert_eq!(
        state.input_guardrail_results[0].output.output_info,
        json!({"checked": "input"})
    );

    assert_eq!(state.output_guardrail_results.len(), 1);
    assert_eq!(
        state.output_guardrail_results[0].guardrail_name,
        "politeness_guardrail"
    );
    assert_eq!(
        state.output_guardrail_results[0].agent_name,
        "Guarded Agent"
    );
    assert_eq!(
        state.output_guardrail_results[0].agent_output,
        json!("Sunny in Lisbon.")
    );
}

#[tokio::test]
async fn python_state_re_serialized_by_rust_keeps_the_python_envelope() {
    for name in [
        "completed.json",
        "interrupted_approval.json",
        "handoff.json",
        "usage_guardrails.json",
        "second_turn_interrupted.json",
        "mixed_batch_interrupted.json",
    ] {
        let original = fixture(name);
        let state = load(name);
        let wire = state.to_json().unwrap();
        assert_eq!(wire["$schemaVersion"], json!("1.21"), "{name}");
        assert_eq!(wire["current_agent"], original["current_agent"], "{name}");
        assert_eq!(wire["current_turn"], original["current_turn"], "{name}");
        assert_eq!(wire["max_turns"], original["max_turns"], "{name}");
        assert_eq!(
            wire["context"]["usage"]["total_tokens"], original["context"]["usage"]["total_tokens"],
            "{name}"
        );
        assert_eq!(wire["trace"]["id"], original["trace"]["id"], "{name}");
        assert_eq!(
            wire["trace"]["group_id"], original["trace"]["group_id"],
            "{name}"
        );
        assert_eq!(
            wire["trace"]["workflow_name"], original["trace"]["workflow_name"],
            "{name}"
        );
        assert_eq!(
            wire["trace"]["metadata"], original["trace"]["metadata"],
            "{name}"
        );
        assert_eq!(
            wire["last_processed_response"].is_null(),
            original["last_processed_response"].is_null(),
            "{name}"
        );
        // A Rust re-read of what Rust wrote is identical to the Rust read of Python's payload.
        let reread = RunState::from_json(wire).unwrap();
        assert_eq!(reread, state, "{name}");
    }
}

#[tokio::test]
async fn python_interrupted_state_resumed_by_rust_is_still_resumable_by_python() {
    // The envelope Rust writes after reading Python's interrupted snapshot must keep the
    // keys Python needs: `last_processed_response` plus the ownership range.
    let state = load("interrupted_approval.json");
    let wire = state.to_json().unwrap();
    let processed = &wire["last_processed_response"];
    assert_eq!(
        processed["functions"][0]["tool"]["name"],
        json!("delete_file")
    );
    assert_eq!(
        processed["functions"][0]["tool_call"]["call_id"],
        json!("call_delete_1")
    );
    assert_eq!(processed["tools_used"], json!(["delete_file"]));
    assert_eq!(processed["new_items"].as_array().unwrap().len(), 1);
    assert_eq!(
        processed["interruptions"][0]["tool_name"],
        json!("delete_file")
    );
    let ownership = &wire["current_response_generated_item_ownership"];
    assert_eq!(ownership["start"], json!(0));
    assert_eq!(ownership["end"], json!(2));
    assert_eq!(ownership["interruptions"], json!([1]));
}

#[tokio::test]
async fn python_second_turn_interruption_keeps_the_turn_boundary() {
    let state = load("second_turn_interrupted.json");
    // Turn 1 ran a tool to completion, so the parked turn does not start at index 0.
    assert_eq!(state.generated_items.len(), 4);
    assert_eq!(state.turn_start_index, 2);
    assert_eq!(state.current_turn, 2);
    let processed = state.last_processed_response.as_ref().expect("processed");
    assert_eq!(processed.new_items.len(), 1);
    assert_eq!(processed.functions[0].tool_name, "delete_file");

    // The boundary survives a Rust re-serialization.
    let wire = state.to_json().unwrap();
    assert_eq!(
        wire["current_response_generated_item_ownership"]["start"],
        json!(2)
    );
    assert_eq!(
        wire["current_response_generated_item_ownership"]["end"],
        json!(4)
    );
    assert_eq!(
        wire["current_response_generated_item_ownership"]["interruptions"],
        json!([3])
    );

    let deleted = Arc::new(AtomicUsize::new(0));
    let weather = Arc::new(AtomicUsize::new(0));
    let (d, w) = (deleted.clone(), weather.clone());
    let model = fake();
    model.add_turn(vec![assistant_message("Done.")]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(counting_tool("delete_file", d, "deleted"))
        .with_tool(counting_tool("get_weather", w, "sunny"))
        .with_model_instance(model.clone());

    let mut state = state;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let result = Runner::run(&agent, state).await.unwrap();
    // Only the parked call runs again; the completed turn-1 call is not replayed.
    assert_eq!(deleted.load(Ordering::SeqCst), 1);
    assert_eq!(weather.load(Ordering::SeqCst), 0);
    assert_eq!(result.final_output_text().as_deref(), Some("Done."));
}

#[tokio::test]
async fn python_mixed_batch_resumes_only_the_parked_call() {
    let state = load("mixed_batch_interrupted.json");
    assert_eq!(state.generated_items.len(), 4);
    assert_eq!(state.turn_start_index, 0);
    let processed = state.last_processed_response.as_ref().expect("processed");
    // Both calls of the response are in `new_items`; the executed output is not.
    assert_eq!(processed.new_items.len(), 2);
    assert_eq!(processed.functions.len(), 2);
    assert_eq!(processed.interruptions.len(), 1);
    assert!(
        processed
            .new_items
            .iter()
            .all(|i| matches!(i.kind, RunItemKind::ToolCall { .. }))
    );

    let wire = state.to_json().unwrap();
    let ownership = &wire["current_response_generated_item_ownership"];
    assert_eq!(ownership["start"], json!(0));
    assert_eq!(ownership["end"], json!(4));
    assert_eq!(ownership["interruptions"], json!([3]));

    let deleted = Arc::new(AtomicUsize::new(0));
    let weather = Arc::new(AtomicUsize::new(0));
    let (d, w) = (deleted.clone(), weather.clone());
    let model = fake();
    model.add_turn(vec![assistant_message("Both handled.")]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(counting_tool("delete_file", d, "deleted"))
        .with_tool(counting_tool("get_weather", w, "sunny"))
        .with_model_instance(model.clone());

    let mut state = state;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let result = Runner::run(&agent, state).await.unwrap();
    assert_eq!(deleted.load(Ordering::SeqCst), 1);
    // The sibling already produced its output; it is not executed a second time.
    assert_eq!(weather.load(Ordering::SeqCst), 0);
    assert_eq!(result.final_output_text().as_deref(), Some("Both handled."));
}

#[test]
fn the_python_tool_invocation_ledger_is_read_and_rewritten() {
    let state = load("mixed_batch_interrupted.json");
    let ledger = &state.tool_invocations;
    assert_eq!(ledger.len(), 2);
    let parked = &ledger["call_delete_3"];
    assert_eq!(parked.invocation_type, "function_call");
    assert!(!parked.executed);
    assert!(!parked.completed);
    let done = &ledger["call_weather_4"];
    assert!(done.executed);
    assert!(done.completed);
    assert_eq!(
        done.approval_scope,
        "ebfaafc0dd470214bab3421e81676cb703ed4adbebb486804c7298f6ed89cf6a"
    );

    let wire = state.to_json().unwrap();
    assert_eq!(
        wire["context"]["tool_invocations"]["call_delete_3"]["fingerprint"],
        json!("2c2e1289df12dcc7af612784f5be9888261d85a79b69b7549d55013b7125d23d")
    );
    assert_eq!(
        wire["context"]["tool_invocations"]["call_weather_4"]["completed"],
        json!(true)
    );
}

#[tokio::test]
async fn a_snapshot_with_mutated_tool_arguments_is_rejected() {
    let mut wire = fixture("interrupted_approval.json");
    // Edit the parked call's arguments without touching the ledger, as a tampered or
    // hand-edited snapshot would.
    for path in ["generated_items", "session_items"] {
        for item in wire[path].as_array_mut().unwrap() {
            if let Some(args) = item["raw_item"].get_mut("arguments") {
                *args = json!(r#"{"path":"/etc/passwd"}"#);
            }
        }
    }
    wire["current_step"]["data"]["interruptions"][0]["raw_item"]["arguments"] =
        json!(r#"{"path":"/etc/passwd"}"#);

    let state = RunState::from_json(wire).unwrap();
    let model = fake();
    model.add_turn(vec![assistant_message("never reached")]);
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = Agent::<()>::new("File Agent")
        .with_tool(counting_tool("delete_file", calls.clone(), "deleted"))
        .with_model_instance(model.clone());

    let mut state = state;
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let err = Runner::run(&agent, state).await.unwrap_err().to_string();
    assert!(err.contains("reused a tool call ID"), "{err}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_snapshot_python_marked_unrecoverable_is_refused() {
    let mut wire = fixture("interrupted_approval.json");
    wire["terminal_unrecoverable"] = json!(true);
    let state = RunState::from_json(wire).unwrap();
    assert!(state.terminal_unrecoverable);
    // The flag survives a Rust re-serialization, so the refusal is not lost in transit.
    assert_eq!(
        state.to_json().unwrap()["terminal_unrecoverable"],
        json!(true)
    );

    let model = fake();
    let agent = Agent::<()>::new("File Agent")
        .with_tool(counting_tool(
            "delete_file",
            Arc::new(AtomicUsize::new(0)),
            "deleted",
        ))
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, state).await.unwrap_err().to_string();
    assert!(err.contains("unrecoverable boundary"), "{err}");
}

fn counting_tool(
    name: &'static str,
    counter: Arc<AtomicUsize>,
    reply: &'static str,
) -> FunctionTool<()> {
    FunctionTool::<()>::new(name, name, move |_c, _a: Value| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(reply.to_string())
        }
    })
    .with_needs_approval(name == "delete_file")
}

/// Write the Rust-produced snapshots that `tests/fixtures/python_run_state/resume_rust_state.py`
/// loads with `RunState.from_json` and resumes with the Python SDK. Opt-in so the Rust suite
/// never depends on Python: set `OARS_RUST_STATE_OUT` to a directory to refresh them.
#[tokio::test]
async fn write_rust_states_for_the_python_consumer() {
    let Ok(dir) = std::env::var("OARS_RUST_STATE_OUT") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    // Keys are sorted before writing: `serde_json::Map` is insertion-ordered when a
    // dependency turns on `serde_json/preserve_order` (the `mongodb` feature does), and a
    // committed fixture must not change with the feature set it was generated under.
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort_unstable();
                Value::Object(
                    keys.into_iter()
                        .map(|k| (k.clone(), sorted(&map[k])))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            scalar => scalar.clone(),
        }
    }
    let write = |name: &str, state: &RunState| {
        let json = serde_json::to_string_pretty(&sorted(&state.to_json().unwrap())).unwrap();
        std::fs::write(format!("{dir}/{name}"), json + "\n").unwrap();
    };

    // The run config pins everything that would otherwise vary between runs.
    let config = || {
        let mut c = RunConfig::<()>::default()
            .with_workflow_name("rust-workflow")
            .with_group_id("rust-group");
        c.trace_id = Some("trace_fedcba9876543210fedcba987654".into());
        c
    };

    // 1. Interrupted by a function-tool approval.
    let tool = FunctionTool::<()>::new("delete_file", "Delete a file", |_c, a: Value| async move {
        Ok(format!("deleted {}", a["path"].as_str().unwrap_or("")))
    })
    .with_params_json_schema(json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": false,
    }))
    .with_needs_approval(true);
    let model = fake();
    model.add_turn(vec![openai_agents_sdk::testing::function_call_with_id(
        "delete_file",
        r#"{"path":"/tmp/report.txt"}"#,
        "call_rust_delete_1",
    )]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let result = Runner::run_with_options(
        &agent,
        "Delete /tmp/report.txt",
        RunOptions::new(()).run_config(config()),
    )
    .await
    .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    write("rust_interrupted_approval.json", &result.to_state());

    // 2. A completed run, which Python resumes by adding input.
    let model = fake();
    model.add_turn(vec![assistant_message_with_id(
        "It is sunny in Hanoi.",
        "msg_rust_completed_1",
    )]);
    let agent = Agent::<()>::new("Weather Agent").with_model_instance(model.clone());
    let result = Runner::run_with_options(
        &agent,
        "What is the weather in Hanoi?",
        RunOptions::new(()).run_config(config()),
    )
    .await
    .unwrap();
    write("rust_completed.json", &result.to_state());

    // 3. A handoff, so Python resumes on the target agent.
    let model = fake();
    model.add_turn(vec![openai_agents_sdk::testing::function_call_with_id(
        "transfer_to_spanish_agent",
        "{}",
        "call_rust_handoff_1",
    )]);
    model.add_turn(vec![assistant_message_with_id(
        "Hola, como estas?",
        "msg_rust_handoff_1",
    )]);
    let spanish = Agent::<()>::new("Spanish Agent").with_model_instance(model.clone());
    let triage = Agent::<()>::new("Triage Agent")
        .with_handoff(handoff(spanish))
        .with_model_instance(model.clone());
    let result =
        Runner::run_with_options(&triage, "Hola", RunOptions::new(()).run_config(config()))
            .await
            .unwrap();
    assert_eq!(result.last_agent.name, "Spanish Agent");
    write("rust_handoff.json", &result.to_state());

    // 4. One response asking for two tools: one runs, the other parks.
    let model = fake();
    model.add_turn(vec![
        openai_agents_sdk::testing::function_call_with_id(
            "get_weather",
            r#"{"city":"Porto"}"#,
            "call_rust_weather_1",
        ),
        openai_agents_sdk::testing::function_call_with_id(
            "delete_file",
            r#"{"path":"/tmp/porto.txt"}"#,
            "call_rust_delete_2",
        ),
    ]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(counting_tool(
            "delete_file",
            Arc::new(AtomicUsize::new(0)),
            "deleted",
        ))
        .with_tool(counting_tool(
            "get_weather",
            Arc::new(AtomicUsize::new(0)),
            "sunny",
        ))
        .with_model_instance(model.clone());
    let result = Runner::run_with_options(
        &agent,
        "Weather in Porto, then delete it",
        RunOptions::new(()).run_config(config()),
    )
    .await
    .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    write("rust_mixed_batch_interrupted.json", &result.to_state());

    // 5. Turn 1 completes a tool call, turn 2 parks, so the turn does not start at index 0.
    let model = fake();
    model.add_turn(vec![openai_agents_sdk::testing::function_call_with_id(
        "get_weather",
        r#"{"city":"Oslo"}"#,
        "call_rust_weather_2",
    )]);
    model.add_turn(vec![openai_agents_sdk::testing::function_call_with_id(
        "delete_file",
        r#"{"path":"/tmp/oslo.txt"}"#,
        "call_rust_delete_3",
    )]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(counting_tool(
            "delete_file",
            Arc::new(AtomicUsize::new(0)),
            "deleted",
        ))
        .with_tool(counting_tool(
            "get_weather",
            Arc::new(AtomicUsize::new(0)),
            "sunny",
        ))
        .with_model_instance(model.clone());
    let result = Runner::run_with_options(
        &agent,
        "Check Oslo then delete the file",
        RunOptions::new(()).run_config(config()),
    )
    .await
    .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    write("rust_second_turn_interrupted.json", &result.to_state());
}

/// `interrupted_approval.json` with the parked call already approved in the serialized
/// approvals bucket, so a resume needs no `RunState::approve` call of its own.
fn preapproved_interruption_fixture() -> Value {
    let mut wire = fixture("interrupted_approval.json");
    wire["context"]["approvals"] = json!({
        "delete_file": { "approved": ["call_delete_1"], "rejected": [] }
    });
    wire
}

/// Resume `wire` against a `delete_file` tool that needs approval, reporting how many times the
/// tool body ran and how many approvals the run came back asking for.
async fn resume_preapproved(wire: Value) -> (usize, usize) {
    let state = RunState::from_json(wire).expect("from_json");
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let tool = FunctionTool::<()>::new("delete_file", "Delete a file", move |_c, _a: Value| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok("deleted".to_string())
        }
    })
    .with_needs_approval(true);

    let model = fake();
    model.add_turn(vec![assistant_message("Deleted the report.")]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(tool)
        .with_model_instance(model);

    let result = Runner::run(&agent, state).await.expect("run");
    (calls.load(Ordering::SeqCst), result.interruptions.len())
}

#[tokio::test]
async fn an_approval_the_ledger_cannot_bind_is_requested_again() {
    let mut wire = preapproved_interruption_fixture();
    // A decision for a call the snapshot never bound to an invocation. Python stops honouring
    // this from schema 1.15 on, and asks the user again.
    wire["context"]["tool_invocations"] = json!({});

    let state = RunState::from_json(wire.clone()).expect("from_json");
    assert!(
        state
            .restored_unbound_approval_call_ids
            .contains("call_delete_1")
    );

    let (ran, interruptions) = resume_preapproved(wire).await;
    assert_eq!(ran, 0, "an unbound approval must not authorize the call");
    assert_eq!(interruptions, 1, "the approval is raised again");
}

#[tokio::test]
async fn a_renewed_approval_clears_the_unbound_call_id() {
    let mut wire = preapproved_interruption_fixture();
    wire["context"]["tool_invocations"] = json!({});
    let mut state = RunState::from_json(wire).expect("from_json");
    assert_eq!(state.restored_unbound_approval_call_ids.len(), 1);

    // A fresh decision replaces the unbound one, exactly as Python's
    // `_apply_approval_decision` discards the call id.
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    assert!(state.restored_unbound_approval_call_ids.is_empty());

    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let tool = FunctionTool::<()>::new("delete_file", "Delete a file", move |_c, _a: Value| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok("deleted".to_string())
        }
    })
    .with_needs_approval(true);
    let model = fake();
    model.add_turn(vec![assistant_message("Deleted the report.")]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(tool)
        .with_model_instance(model);

    let result = Runner::run(&agent, state).await.expect("run");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(result.interruptions.is_empty());
}

#[tokio::test]
async fn an_approval_the_ledger_binds_still_authorizes_the_call() {
    // The same payload, with the ledger entry Python writes for the parked call.
    let (ran, interruptions) = resume_preapproved(preapproved_interruption_fixture()).await;
    assert_eq!(ran, 1);
    assert_eq!(interruptions, 0);
}

#[tokio::test]
async fn a_pre_binding_schema_keeps_the_tolerant_approval_rule() {
    let mut wire = preapproved_interruption_fixture();
    wire["context"]["tool_invocations"] = json!({});
    // 1.14 predates the binding; Python clears the ledger instead of enforcing it, so the
    // decision still stands. The numeric compare matters here: "1.14" sorts above "1.15" as a
    // string.
    wire["$schemaVersion"] = json!("1.14");

    let state = RunState::from_json(wire.clone()).expect("from_json");
    assert!(state.restored_unbound_approval_call_ids.is_empty());

    let (ran, interruptions) = resume_preapproved(wire).await;
    assert_eq!(ran, 1, "a pre-1.15 snapshot keeps the tolerant rule");
    assert_eq!(interruptions, 0);
}

#[tokio::test]
async fn a_legacy_flat_rust_snapshot_keeps_the_tolerant_approval_rule() {
    // The flat `"1.0"` shape this crate wrote before the Python envelope. It never recorded a
    // ledger, so every decision in it would read as unbound; the tolerance stays.
    let state = load("interrupted_approval.json");
    let mut flat = serde_json::to_value(&state).expect("serialize flat");
    flat["$schemaVersion"] = json!("1.0");
    flat["approvals"] = json!({
        "\u{1}delete_file": { "approved_call_ids": ["call_delete_1"] }
    });
    flat["tool_invocations"] = json!({});
    assert!(flat.get("current_agent_name").is_some());

    let restored = RunState::from_json(flat.clone()).expect("from_json");
    assert!(restored.restored_unbound_approval_call_ids.is_empty());

    let (ran, interruptions) = resume_preapproved(flat).await;
    assert_eq!(ran, 1, "a flat 1.0 snapshot predates the ledger");
    assert_eq!(interruptions, 0);
}

#[tokio::test]
async fn an_un_renewed_approval_stays_unbound_across_a_re_snapshot() {
    let mut wire = preapproved_interruption_fixture();
    wire["context"]["tool_invocations"] = json!({});
    let state = RunState::from_json(wire).expect("from_json");

    let tool = FunctionTool::<()>::new("delete_file", "Delete a file", |_c, _a: Value| async {
        Ok("deleted".to_string())
    })
    .with_needs_approval(true);
    let model = fake();
    model.add_turn(vec![assistant_message("Deleted the report.")]);
    let agent = Agent::<()>::new("File Agent")
        .with_tool(tool)
        .with_model_instance(model);

    let result = Runner::run(&agent, state).await.expect("run");
    assert_eq!(result.interruptions.len(), 1);

    // Re-snapshotting must not quietly bind the call the user never re-approved: the next
    // resume would otherwise honour the stale decision.
    let next = result.to_state();
    assert!(!next.tool_invocations.contains_key("call_delete_1"));
    assert!(
        next.restored_unbound_approval_call_ids
            .contains("call_delete_1")
    );
}

#[test]
fn a_non_object_context_is_refused_by_a_strict_serialization() {
    let state = load("completed.json").with_context(json!("an opaque handle"));
    // Python's deserializer requires a mapping, so this snapshot would be unreadable there.
    let err = state.to_json_strict().expect_err("strict serialize");
    assert!(
        err.to_string().contains("JSON object"),
        "unexpected message: {err}"
    );
    // The tolerant form still writes it, and says so in the metadata.
    let wire = state.to_json().expect("to_json");
    assert_eq!(
        wire["context"]["context_meta"]["requires_deserializer"],
        json!(true)
    );
    assert_eq!(wire["context"]["context"], json!("an opaque handle"));

    // An object context is serializable either way.
    let ok = load("completed.json").with_context(json!({ "tenant": "acme" }));
    let wire = ok.to_json_strict().expect("strict serialize");
    assert_eq!(wire["context"]["context"], json!({ "tenant": "acme" }));
    assert_eq!(
        wire["context"]["context_meta"]["requires_deserializer"],
        json!(false)
    );
}

#[test]
fn a_context_that_needs_restoring_is_an_error_only_under_strict_context() {
    let mut wire = fixture("completed.json");
    wire["context"]["context"] = Value::Null;
    wire["context"]["context_meta"] = json!({
        "original_type": "custom",
        "serialized_via": "omitted",
        "requires_deserializer": true,
        "omitted": true,
        "class_path": "app.context.Tenant",
    });

    // Tolerant read: the state loads, and reports that its context needs restoring.
    let state = RunState::from_json(wire.clone()).expect("from_json");
    assert!(state.context_requires_restoration);
    assert_eq!(state.context, None);

    // Strict read without an override: refused, naming what is missing.
    let err = RunState::from_json_with(wire.clone(), ContextRestoreOptions::strict())
        .expect_err("strict read");
    assert!(
        err.to_string().contains("omitted during serialization"),
        "unexpected message: {err}"
    );

    // An override supplies the context and satisfies the strict read.
    let state = RunState::from_json_with(
        wire,
        ContextRestoreOptions::strict().with_override(json!({ "tenant": "acme" })),
    )
    .expect("override read");
    assert!(!state.context_requires_restoration);
    assert_eq!(state.context, Some(json!({ "tenant": "acme" })));
}

#[test]
fn a_restorable_context_passes_a_strict_read_untouched() {
    let mut wire = fixture("completed.json");
    wire["context"]["context"] = json!({ "tenant": "acme" });
    let state = RunState::from_json_with(wire, ContextRestoreOptions::strict()).expect("strict");
    assert!(!state.context_requires_restoration);
    assert_eq!(state.context, Some(json!({ "tenant": "acme" })));
}

#[test]
fn a_rust_written_context_round_trips_through_the_strict_read() {
    let state = load("completed.json").with_context(json!({ "tenant": "acme" }));
    let wire = state.to_json_strict().expect("strict serialize");
    let back = RunState::from_json_with(wire, ContextRestoreOptions::strict()).expect("strict");
    assert_eq!(back.context, Some(json!({ "tenant": "acme" })));
}
