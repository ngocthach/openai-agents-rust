//! Ports of the Python SDK's `_deserialize_items` skip cases (`tests/test_run_state.py`,
//! `tests/test_error_logging_redaction.py`): an item whose agent is not in the graph, that has
//! no agent or no type, or whose payload its typed model rejects is skipped with a warning, and
//! the rest of the snapshot restores.

use std::sync::{Arc, Mutex, OnceLock};

use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, Runner};
use serde_json::{Value, json};

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOG_LOCK: Mutex<()> = Mutex::new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.level() <= log::Level::Warn {
            RECORDS.lock().unwrap().push(format!("{}", record.args()));
        }
    }
    fn flush(&self) {}
}

fn capture_logs() -> std::sync::MutexGuard<'static, ()> {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Warn);
    });
    let guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    RECORDS.lock().unwrap().clear();
    guard
}

fn logged() -> String {
    RECORDS.lock().unwrap().join("\n")
}

fn make_state() -> RunState {
    RunState {
        schema_version: CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: "TestAgent".into(),
        original_input: vec![InputItem::user("test")],
        max_turns: Some(5),
        context: Some(json!({})),
        ..Default::default()
    }
}

fn message(text: &str) -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

fn message_item(agent: Value, text: &str) -> Value {
    json!({"type": "message_output_item", "agent": agent, "raw_item": message(text)})
}

fn items_wire(items: Value) -> Value {
    let mut wire = make_state().to_json().unwrap();
    wire["session_items"] = items.clone();
    wire["generated_session_item_indexes"] = (0..items.as_array().map_or(0, Vec::len))
        .collect::<Vec<_>>()
        .into();
    wire["generated_items"] = items;
    wire
}

fn restored_texts(wire: Value) -> Vec<String> {
    RunState::from_json(wire)
        .unwrap()
        .generated_items
        .iter()
        .map(|item| {
            serde_json::to_value(item).unwrap()["raw_item"]["content"][0]["text"]
                .as_str()
                .unwrap_or("<none>")
                .to_string()
        })
        .collect()
}

fn restored_len(wire: Value) -> usize {
    RunState::from_json(wire).unwrap().generated_items.len()
}

/// Python: `test_deserialization_handles_unknown_agent_gracefully`. The graph is known only when
/// a run restores the state, so the item goes with the agent graph the resume supplies.
#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn an_item_of_an_agent_outside_the_graph_is_skipped_on_resume() {
    let _logs = capture_logs();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(Agent::<()>::new("TestAgent").with_model_instance(model.clone()));
    let wire = items_wire(json!([
        message_item(json!({"name": "TestAgent"}), "kept"),
        message_item(json!({"name": "UnknownAgent"}), "dropped"),
    ]));
    let state = RunState::from_json(wire).unwrap();
    assert_eq!(state.generated_items.len(), 2, "the graph is not known yet");

    let result = Runner::run_with_options(agent, state, RunOptions::new(()))
        .await
        .unwrap();
    let history: Vec<String> = result
        .to_input_list()
        .iter()
        .filter_map(|i| i.text_content())
        .collect();
    assert!(history.contains(&"kept".to_string()), "{history:?}");
    assert!(!history.contains(&"dropped".to_string()), "{history:?}");
    assert!(
        logged().contains("not found, skipping item"),
        "{}",
        logged()
    );
}

/// Python: `test_deserialize_items_handles_missing_agent_name`,
/// `test_deserialize_items_handles_handoff_output_no_source_no_target`.
#[test]
fn an_item_without_an_agent_is_skipped() {
    let _logs = capture_logs();
    let wire = items_wire(json!([
        {"type": "message_output_item", "raw_item": message("no agent")},
        {"type": "handoff_output_item",
         "raw_item": {"role": "assistant", "content": "Handoff message"}},
        message_item(json!({"name": "TestAgent"}), "kept"),
    ]));
    assert_eq!(restored_texts(wire), vec!["kept"]);
    assert!(
        logged().contains("Item missing agent field, skipping"),
        "{}",
        logged()
    );
}

/// Python: `test_deserialize_items_handles_missing_type`.
#[test]
fn an_item_without_a_type_is_skipped() {
    let _logs = capture_logs();
    let wire = items_wire(json!([
        {"agent": {"name": "TestAgent"}, "raw_item": message("no type")},
        message_item(json!({"name": "TestAgent"}), "kept"),
    ]));
    assert_eq!(restored_texts(wire), vec!["kept"]);
    assert!(
        logged().contains("Item missing type field, skipping"),
        "{}",
        logged()
    );
}

/// Python: `test_deserialization_handles_malformed_items_gracefully`.
#[test]
fn a_malformed_item_is_skipped() {
    let _logs = capture_logs();
    for raw in [json!({"type": "message"}), json!("not-an-object")] {
        let wire = items_wire(json!([{
            "type": "message_output_item", "agent": {"name": "TestAgent"}, "raw_item": raw,
        }]));
        assert_eq!(restored_len(wire), 0);
    }
    assert!(
        logged().contains("Failed to deserialize item"),
        "{}",
        logged()
    );
}

/// Python: the union-adapter fallback case of `_deserialize_items`: a tool output whose raw item
/// has no `type` is not a valid output and is skipped.
#[test]
fn a_tool_output_whose_raw_item_has_no_type_is_skipped() {
    let _logs = capture_logs();
    let wire = items_wire(json!([{
        "type": "tool_call_output_item", "agent": {"name": "TestAgent"},
        "raw_item": {"call_id": "call123", "output": "result"}, "output": "result",
    }]));
    assert_eq!(restored_len(wire), 0);
}

/// Python: `test_deserialize_items_handles_handoff_output_*`: a handoff output needs both its
/// source and its target.
#[test]
fn a_handoff_output_missing_a_side_is_skipped() {
    let _logs = capture_logs();
    let handoff = |extra: Value| {
        let mut item = json!({
            "type": "handoff_output_item", "agent": {"name": "TestAgent"},
            "raw_item": {"role": "assistant", "content": "Handoff message"},
        });
        item.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        item
    };
    let wire = items_wire(json!([
        handoff(json!({"source_agent": {"name": "TestAgent"}})),
        handoff(json!({"target_agent": {"name": "TestAgent"}})),
        handoff(json!({"source_agent": {"name": "TestAgent"}, "target_agent": "TestAgent"})),
    ]));
    assert_eq!(restored_len(wire), 1);
    assert!(
        logged().contains("could not resolve agents"),
        "{}",
        logged()
    );
}

/// Python: `test_run_state_deserialization_warnings_follow_both_data_policies`: the warning
/// names the payload value only when both the model and the tool data policies allow it.
#[test]
fn skip_warnings_follow_both_data_policies() {
    let _logs = capture_logs();
    for (model_redacted, tool_redacted) in
        [(true, false), (false, true), (true, true), (false, false)]
    {
        for (scenario, secret, redacted) in [
            (
                "missing_agent_field",
                "SECRET_ITEM_TYPE",
                "Item missing agent field, skipping",
            ),
            (
                "missing_handoff_agents",
                "SECRET_SOURCE_AGENT",
                "Skipping handoff output item: could not resolve agents",
            ),
        ] {
            RECORDS.lock().unwrap().clear();
            set_dont_log_model_data(model_redacted);
            set_dont_log_tool_data(tool_redacted);
            let item = if scenario == "missing_agent_field" {
                json!({"type": secret, "raw_item": {}})
            } else {
                json!({
                    "type": "handoff_output_item", "agent": {"name": "TestAgent"},
                    "source_agent": secret, "target_agent": "", "raw_item": {},
                })
            };
            let wire = items_wire(json!([item]));
            let _ = RunState::from_json(wire);
            let text = logged();
            reset_dont_log_model_data();
            reset_dont_log_tool_data();
            let label = format!("{scenario} model={model_redacted} tool={tool_redacted}");
            if model_redacted || tool_redacted {
                assert_eq!(text, redacted, "{label}");
                assert!(!text.contains(secret), "{label}");
            } else if scenario == "missing_agent_field" {
                // A typeless value is not a type at all, so only the typed scenario names it.
                assert!(text.contains(redacted), "{label}: {text}");
            } else {
                assert!(text.contains(secret), "{label}: {text}");
            }
        }
    }
}

/// Python: `test_nested_history_ownership_remaps_after_skipped_session_item`: skipping an
/// unrelated item must not shift the ownership of the one that survives. The state keeps one
/// list, so the position of the owned item is what moves.
#[test]
fn a_skipped_item_does_not_shift_the_owned_items_that_survive() {
    let _logs = capture_logs();
    let owned = message("owned");
    // The digest Python computes for this message (`digest_input_item`).
    let digest = "8359850d0a341251272d79a50a45b89e915b2978946e4f3d4811edbaa36d16b3";
    let mut wire = items_wire(json!([
        {"type": "message_output_item", "agent": {"name": "UnknownAgent"},
         "raw_item": message("skip me")},
        message_item(json!({"name": "TestAgent"}), "owned"),
    ]));
    wire["session_items"] = wire["generated_items"].clone();
    wire["generated_session_item_indexes"] = json!([0, 1]);
    wire["original_input"] = json!([owned]);
    wire["nested_history_owned_session_item_refs"] =
        json!([{"index": 1, "digest": digest, "input_index": 0}]);
    let state = RunState::from_json(wire).unwrap();
    assert_eq!(state.generated_items.len(), 2);
    assert_eq!(state.nested_history_owned_items, vec![1]);
}

/// Python: `test_nested_history_ownership_for_skipped_session_item_is_ignored`: ownership of an
/// item that cannot be restored is ignored, with a warning, instead of landing on another item.
#[test]
fn ownership_of_a_skipped_item_is_ignored() {
    let _logs = capture_logs();
    let skipped = message("skipped");
    let digest = "ac68af685398897ec0fe93be230f10ba29605ad27d732c1e3ee0086d36b8d537";
    let mut wire = items_wire(json!([
        {"type": "message_output_item", "raw_item": skipped.clone()},
        message_item(json!({"name": "TestAgent"}), "kept"),
    ]));
    wire["session_items"] = wire["generated_items"].clone();
    wire["generated_session_item_indexes"] = json!([0, 1]);
    wire["original_input"] = json!([skipped]);
    wire["nested_history_owned_session_item_refs"] =
        json!([{"index": 0, "digest": digest, "input_index": 0}]);
    let state = RunState::from_json(wire).unwrap();
    assert_eq!(state.generated_items.len(), 1);
    assert!(state.nested_history_owned_items.is_empty());
    assert!(
        logged().contains("Ignoring nested history ownership for skipped session item"),
        "{}",
        logged()
    );
}
