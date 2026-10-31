//! Tracks what a server-managed conversation already holds, so a resumed or retried run
//! neither resends nor skips deltas. Port of
//! `agents.run_internal.oai_conversation.OpenAIServerConversationTracker`.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::items::{InputItem, ModelResponse, RunItem, RunItemKind};
use crate::run_state::ServerConversationState;

/// Placeholder id emitted by non-Responses providers; never a server identity.
const FAKE_RESPONSES_ID: &str = "__fake_id__";

fn normalize_server_item_id(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty() && *id != FAKE_RESPONSES_ID)
        .map(str::to_string)
}

/// Session and output-only metadata that must not reach the model or a fingerprint.
fn strip_internal_metadata(mut item: Value) -> Value {
    if let Value::Object(m) = &mut item {
        m.remove("created_by");
        m.remove("_session_description");
        m.remove("_session_title");
        // The keys the Python SDK writes (`TOOL_CALL_SESSION_*_KEY`).
        m.remove("_agents_tool_description");
        m.remove("_agents_tool_title");
    }
    item
}

/// A stable content fingerprint. `serde_json` writes object keys in insertion order for a
/// `Map`, so the payload is re-serialized through a sorted intermediate first.
fn fingerprint(item: &Value, ignore_ids: bool) -> Option<String> {
    let mut payload = strip_internal_metadata(item.clone());
    if ignore_ids && let Value::Object(m) = &mut payload {
        m.remove("id");
    }
    Some(canonical_json(&payload))
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body = keys
                .iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        Value::String((*k).clone()),
                        canonical_json(&map[*k])
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        Value::Array(items) => {
            let body = items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        other => other.to_string(),
    }
}

fn is_tool_search_item(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("tool_search_call") | Some("tool_search_output")
    )
}

/// `tool_search` items have no stable provider identity, so they are matched on content.
fn anonymous_tool_search_fingerprint(item: &Value) -> Option<String> {
    if !is_tool_search_item(item) {
        return None;
    }
    fingerprint(item, true)
}

fn call_id_of(item: &Value) -> Option<String> {
    item.get("call_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn has_output_payload(item: &Value) -> bool {
    item.get("output").is_some()
}

/// What a server-managed conversation has already accepted.
#[derive(Debug, Clone, Default)]
pub(crate) struct ServerConversationTracker {
    state: ServerConversationState,
    restored_anonymous_tool_search_fingerprints: BTreeSet<String>,
    sent_initial_input: bool,
    primed_from_state: bool,
}

impl ServerConversationTracker {
    /// The durable half of the tracker, for `RunState`.
    pub(crate) fn state(&self) -> ServerConversationState {
        self.state.clone()
    }

    /// Mark the run input as delivered (used when restoring a legacy index-based snapshot).
    pub(crate) fn set_sent_initial_input(&mut self, sent: bool) {
        self.sent_initial_input = sent;
    }

    /// The id a resumed run chains from: the most recent response that carries one. A response
    /// without an id (a non-Responses provider) does not break the chain. Port of the
    /// `latest_response_id` selection in `hydrate_from_state`.
    pub(crate) fn latest_response_id(model_responses: &[ModelResponse]) -> Option<String> {
        model_responses
            .iter()
            .rev()
            .find_map(|response| response.response_id.clone())
    }

    /// Seed tracking from a prior run state so a resume does not replay sent content.
    /// Port of `hydrate_from_state`.
    pub(crate) fn hydrate_from_state(
        &mut self,
        state: &ServerConversationState,
        original_input: &[InputItem],
        generated_items: &[RunItem],
        model_responses: &[ModelResponse],
        input_already_sent: bool,
    ) {
        self.state = state.clone();
        self.primed_from_state = true;

        if input_already_sent {
            for item in original_input {
                if let Some(id) = normalize_server_item_id(item.0.get("id")) {
                    self.state.server_item_ids.insert(id);
                }
                if let Some(fp) = fingerprint(&item.0, false) {
                    self.state.sent_item_fingerprints.insert(fp);
                }
                if let Some(fp) = anonymous_tool_search_fingerprint(&item.0) {
                    self.restored_anonymous_tool_search_fingerprints.insert(fp);
                }
            }
            self.sent_initial_input = true;
        }

        for response in model_responses {
            self.track_server_items(response);
        }

        // An interrupted or retried turn stops before its next model request, so every tool
        // output it built locally for the latest response is still unsent. Port of
        // `get_unsent_tool_call_ids_for_interrupted_state`.
        let unsent_tool_call_ids: BTreeSet<String> = model_responses
            .last()
            .map(|response| {
                response
                    .output
                    .iter()
                    .filter_map(|o| call_id_of(&o.to_value()))
                    .collect()
            })
            .unwrap_or_default();

        for item in generated_items {
            if matches!(item.kind, RunItemKind::ToolApproval { .. }) {
                continue;
            }
            let raw = item.to_input_item().0;
            if has_output_payload(&raw)
                && let Some(call_id) = call_id_of(&raw)
                && unsent_tool_call_ids.contains(&call_id)
            {
                continue;
            }
            if let Some(id) = normalize_server_item_id(raw.get("id")) {
                self.state.server_item_ids.insert(id);
            }
            if let Some(call_id) = call_id_of(&raw)
                && has_output_payload(&raw)
            {
                self.state.server_tool_call_ids.insert(call_id);
            }
            if let Some(fp) = fingerprint(&raw, false) {
                if is_tool_search_item(&raw) {
                    self.state.server_output_fingerprints.insert(fp.clone());
                }
                // An admitted input item is delivered only by a model call that carries it
                // (`mark_input_as_sent`). One that was admitted but never sent stays unsent.
                if !matches!(item.kind, RunItemKind::Input { .. }) {
                    self.state.sent_item_fingerprints.insert(fp);
                }
            }
            if let Some(fp) = anonymous_tool_search_fingerprint(&raw) {
                self.restored_anonymous_tool_search_fingerprints.insert(fp);
            }
        }
    }

    /// Record what the server produced for a response. Port of `track_server_items`.
    pub(crate) fn track_server_items(&mut self, response: &ModelResponse) {
        for output in &response.output {
            let raw = output.to_value();
            if let Some(id) = normalize_server_item_id(raw.get("id")) {
                self.state.server_item_ids.insert(id);
            }
            if let Some(call_id) = call_id_of(&raw)
                && has_output_payload(&raw)
            {
                self.state.server_tool_call_ids.insert(call_id);
            }
            // The server already holds everything it produced. Rust has no object identity
            // for a JSON item, so the content fingerprint plays the role Python's
            // `server_items` identity list plays.
            if let Some(fp) = fingerprint(&strip_internal_metadata(raw.clone()), false) {
                self.state.server_output_fingerprints.insert(fp);
            }
        }
    }

    /// Whether the server already holds this input item, by the same fingerprint that selects
    /// deltas. Staged input that was never delivered reads as unsent.
    pub(crate) fn was_sent(&self, item: &InputItem) -> bool {
        fingerprint(&item.0, false)
            .is_some_and(|fp| self.state.sent_item_fingerprints.contains(&fp))
    }

    /// Record the items a successful model call delivered. Port of `mark_input_as_sent`.
    pub(crate) fn mark_input_as_sent(&mut self, items: &[InputItem]) {
        for item in items {
            if let Some(id) = normalize_server_item_id(item.0.get("id")) {
                self.state.server_item_ids.insert(id);
            }
            if let Some(fp) = fingerprint(&item.0, false) {
                self.state.sent_item_fingerprints.insert(fp);
            }
            if let Some(fp) = anonymous_tool_search_fingerprint(&item.0) {
                self.restored_anonymous_tool_search_fingerprints.insert(fp);
            }
        }
    }

    /// Whether this generated item still has to be sent to the server.
    /// Port of the filter inside `prepare_input`.
    pub(crate) fn needs_send(&self, converted: &InputItem) -> bool {
        let raw = &converted.0;
        if let Some(id) = normalize_server_item_id(raw.get("id"))
            && self.state.server_item_ids.contains(&id)
        {
            return false;
        }
        let call_id = call_id_of(raw);
        if let Some(call_id) = &call_id
            && has_output_payload(raw)
            && self.state.server_tool_call_ids.contains(call_id)
        {
            return false;
        }
        let fp = fingerprint(raw, false);
        if let Some(fp) = &fp
            && self.state.server_output_fingerprints.contains(fp)
        {
            return false;
        }
        // A generated item delivered by an earlier turn of this run (or of the run a state
        // was taken from) is already on the server. Python skips it by object identity
        // (`sent_items`); the content fingerprint plays that role here.
        if let Some(fp) = &fp
            && self.state.sent_item_fingerprints.contains(fp)
        {
            return false;
        }
        if self.primed_from_state
            && normalize_server_item_id(raw.get("id")).is_none()
            && call_id.is_none()
            && let Some(fp) = anonymous_tool_search_fingerprint(raw)
            && self
                .restored_anonymous_tool_search_fingerprints
                .contains(&fp)
        {
            return false;
        }
        true
    }

    /// Mark the run input as delivered. Port of the `sent_initial_input` half of
    /// `prepare_input`.
    pub(crate) fn take_initial_input(&mut self, original_input: &[InputItem]) -> Vec<InputItem> {
        if self.sent_initial_input {
            return Vec::new();
        }
        self.sent_initial_input = true;
        original_input.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys() {
        let a = json!({"b": 1, "a": 2});
        let b = json!({"a": 2, "b": 1});
        assert_eq!(canonical_json(&a), canonical_json(&b));
    }

    #[test]
    fn an_item_the_server_already_holds_is_not_resent() {
        let mut tracker = ServerConversationTracker::default();
        let item = InputItem(json!({"type": "message", "role": "user", "content": "hi"}));
        assert!(tracker.needs_send(&item));
        tracker.mark_input_as_sent(std::slice::from_ref(&item));
        // A delivered item is not sent again, in a live run as after a resume.
        assert!(!tracker.needs_send(&item));
    }

    #[test]
    fn a_tool_output_the_server_produced_is_not_resent() {
        let mut tracker = ServerConversationTracker::default();
        let response = ModelResponse {
            output: vec![crate::items::OutputItem::Other(
                json!({"type": "function_call_output", "call_id": "c1", "output": "ok"}),
            )],
            usage: Default::default(),
            response_id: None,
            request_id: None,
            raw_usage: None,
        };
        tracker.track_server_items(&response);
        let item = InputItem(json!({
            "type": "function_call_output", "call_id": "c1", "output": "ok"
        }));
        assert!(!tracker.needs_send(&item));
    }

    #[test]
    fn hydration_keeps_the_latest_response_outputs_unsent() {
        let mut tracker = ServerConversationTracker::default();
        let response = ModelResponse {
            output: vec![crate::items::OutputItem::Other(
                json!({"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"}),
            )],
            usage: Default::default(),
            response_id: Some("resp_1".into()),
            request_id: None,
            raw_usage: None,
        };
        let generated = vec![RunItem::new(
            "A",
            RunItemKind::ToolCallOutput {
                custom_data: None,
                raw_item: json!({
                    "type": "function_call_output", "call_id": "c1", "output": "ok"
                }),
                output: json!("ok"),
            },
        )];
        tracker.hydrate_from_state(
            &ServerConversationState::default(),
            &[],
            &generated,
            std::slice::from_ref(&response),
            true,
        );
        let output = InputItem(json!({
            "type": "function_call_output", "call_id": "c1", "output": "ok"
        }));
        assert!(
            tracker.needs_send(&output),
            "the output of the interrupted turn still has to reach the server"
        );
    }

    #[test]
    fn hydration_marks_generated_items_as_already_sent() {
        let mut tracker = ServerConversationTracker::default();
        let generated = vec![RunItem::new(
            "A",
            RunItemKind::ToolCall {
                raw_item: json!({"type": "function_call", "call_id": "c1", "name": "t"}),
            },
        )];
        tracker.hydrate_from_state(
            &ServerConversationState::default(),
            &[],
            &generated,
            &[],
            true,
        );
        let item = InputItem(json!({"type": "function_call", "call_id": "c1", "name": "t"}));
        assert!(!tracker.needs_send(&item));
    }

    #[test]
    fn resume_chains_from_the_latest_response_that_has_an_id() {
        let response = |id: Option<&str>| ModelResponse {
            output: Vec::new(),
            usage: Default::default(),
            response_id: id.map(str::to_string),
            request_id: None,
            raw_usage: None,
        };
        let responses = vec![
            response(Some("resp_first")),
            response(Some("resp_second")),
            response(None),
        ];
        assert_eq!(
            ServerConversationTracker::latest_response_id(&responses).as_deref(),
            Some("resp_second")
        );
        assert_eq!(ServerConversationTracker::latest_response_id(&[]), None);
    }
}
