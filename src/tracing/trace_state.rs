//! Serializable trace metadata for run-state persistence. Mirrors the `TraceState` /
//! `reattach_trace` half of `agents.tracing.traces`.
//!
//! A run that is snapshotted and resumed later should keep exporting into the *same* trace
//! instead of starting a new one. [`TraceState`] is the part of a [`Trace`] that survives
//! serialization, and [`reattach_trace`] rebuilds a live trace from it without re-emitting the
//! trace-start event the original run already sent.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::spans::Trace;

/// How many recently started trace ids are remembered, as in Python.
const MAX_STARTED_TRACE_IDS: usize = 4096;

struct StartedIds {
    order: VecDeque<String>,
    set: std::collections::HashSet<String>,
}

fn started_ids() -> &'static Mutex<StartedIds> {
    static STARTED: OnceLock<Mutex<StartedIds>> = OnceLock::new();
    STARTED.get_or_init(|| {
        Mutex::new(StartedIds {
            order: VecDeque::new(),
            set: std::collections::HashSet::new(),
        })
    })
}

/// Remember that `trace_id` was started in this process. Port of `_mark_trace_id_started`.
pub(crate) fn mark_trace_id_started(trace_id: &str) {
    if trace_id.is_empty() || trace_id == "no-op" {
        return;
    }
    let mut guard = started_ids().lock().expect("started ids poisoned");
    if guard.set.contains(trace_id) {
        guard.order.retain(|id| id != trace_id);
    } else {
        guard.set.insert(trace_id.to_string());
    }
    guard.order.push_back(trace_id.to_string());
    while guard.order.len() > MAX_STARTED_TRACE_IDS {
        if let Some(oldest) = guard.order.pop_front() {
            guard.set.remove(&oldest);
        }
    }
}

/// Forget every started trace id, as a process restart would. Port of `_clear_started_trace_ids`;
/// a resumed run in a fresh process must not reattach to a trace this process never started.
pub fn clear_started_trace_ids() {
    let mut guard = started_ids().lock().expect("started ids poisoned");
    guard.order.clear();
    guard.set.clear();
}

/// Whether `trace_id` was already started in this process. Port of `_trace_id_was_started`.
pub fn trace_id_was_started(trace_id: Option<&str>) -> bool {
    let Some(trace_id) = trace_id.filter(|id| !id.is_empty() && *id != "no-op") else {
        return false;
    };
    started_ids()
        .lock()
        .expect("started ids poisoned")
        .set
        .contains(trace_id)
}

/// A fingerprint of a tracing API key, so a snapshot can prove the resumed run supplied the
/// same explicit key without persisting the secret. Port of `_hash_tracing_api_key`.
pub fn hash_tracing_api_key(tracing_api_key: Option<&str>) -> Option<String> {
    let key = tracing_api_key?;
    let digest = Sha256::digest(key.as_bytes());
    Some(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// Trace metadata that survives serialization of a run state. Port of `TraceState`.
///
/// Two serializations exist and they are not interchangeable. [`TraceState::to_json`] /
/// [`TraceState::from_json`] produce the **trace wire payload** Python persists
/// (`{"object", "id", "workflow_name", ...}` plus the `extra` keys inline) and are what a run
/// snapshot should carry. The derived `Serialize`/`Deserialize` instead produce this struct's
/// own Rust field names with `extra` as a nested object, which is convenient inside a larger
/// Rust-only payload but is **not** the Python format.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceState {
    /// The trace id (`trace_<32 hex>`).
    pub trace_id: Option<String>,
    /// The workflow name.
    pub workflow_name: Option<String>,
    /// The group id linking traces of one conversation.
    pub group_id: Option<String>,
    /// Trace metadata.
    pub metadata: Option<Map<String, Value>>,
    /// The explicit export API key, when the snapshot is allowed to carry it.
    pub tracing_api_key: Option<String>,
    /// A fingerprint of the export API key, kept even when the key itself is stripped.
    pub tracing_api_key_hash: Option<String>,
    /// The `object` discriminator of the serialized payload (`"trace"`).
    pub object_type: Option<String>,
    /// Any other keys of the serialized payload, preserved on round-trip.
    #[serde(default)]
    pub extra: Map<String, Value>,
}

fn take_str(data: &mut Map<String, Value>, key: &str) -> Option<String> {
    match data.remove(key) {
        Some(Value::String(s)) => Some(s),
        _ => None,
    }
}

impl TraceState {
    /// Capture the state of a live trace (including its API key). Port of `TraceState.from_trace`.
    pub fn from_trace(trace: Option<&Trace>) -> Option<Self> {
        Self::from_json(trace?.to_json(true).as_ref())
    }

    /// Parse a serialized trace payload. Port of `TraceState.from_json`.
    pub fn from_json(payload: Option<&Value>) -> Option<Self> {
        let mut data = match payload {
            Some(Value::Object(map)) if !map.is_empty() => map.clone(),
            _ => return None,
        };
        let object_type = take_str(&mut data, "object");
        let trace_id = take_str(&mut data, "id").or_else(|| take_str(&mut data, "trace_id"));
        let workflow_name = take_str(&mut data, "workflow_name");
        let group_id = take_str(&mut data, "group_id");
        let metadata = match data.remove("metadata") {
            Some(Value::Object(m)) => Some(m),
            _ => None,
        };
        let tracing_api_key = take_str(&mut data, "tracing_api_key");
        let stored_hash = take_str(&mut data, "tracing_api_key_hash");
        let tracing_api_key_hash = hash_tracing_api_key(tracing_api_key.as_deref()).or(stored_hash);
        Some(TraceState {
            trace_id,
            workflow_name,
            group_id,
            metadata,
            tracing_api_key,
            tracing_api_key_hash,
            object_type,
            extra: data,
        })
    }

    /// Serialize back to a trace payload. Port of `TraceState.to_json`.
    pub fn to_json(&self, include_tracing_api_key: bool) -> Option<Value> {
        if self.trace_id.is_none()
            && self.workflow_name.is_none()
            && self.group_id.is_none()
            && self.metadata.is_none()
            && self.tracing_api_key.is_none()
            && self.tracing_api_key_hash.is_none()
            && self.object_type.is_none()
            && self.extra.is_empty()
        {
            return None;
        }
        let mut payload = Map::new();
        if let Some(t) = self.object_type.as_ref().filter(|t| !t.is_empty()) {
            payload.insert("object".into(), Value::String(t.clone()));
        }
        if let Some(id) = self.trace_id.as_ref().filter(|i| !i.is_empty()) {
            payload.insert("id".into(), Value::String(id.clone()));
        }
        if let Some(n) = &self.workflow_name {
            payload.insert("workflow_name".into(), Value::String(n.clone()));
        }
        if let Some(g) = &self.group_id {
            payload.insert("group_id".into(), Value::String(g.clone()));
        }
        if let Some(m) = &self.metadata {
            payload.insert("metadata".into(), Value::Object(m.clone()));
        }
        if include_tracing_api_key
            && let Some(k) = self.tracing_api_key.as_ref().filter(|k| !k.is_empty())
        {
            payload.insert("tracing_api_key".into(), Value::String(k.clone()));
        }
        if let Some(h) = self.tracing_api_key_hash.as_ref().filter(|h| !h.is_empty()) {
            payload.insert("tracing_api_key_hash".into(), Value::String(h.clone()));
        }
        for (k, v) in &self.extra {
            payload.entry(k.clone()).or_insert_with(|| v.clone());
        }
        Some(Value::Object(payload))
    }

    /// Whether this state describes the trace a resumed run is about to create.
    /// Port of `_trace_state_matches_effective_settings`.
    pub fn matches_effective_settings(
        &self,
        workflow_name: &str,
        trace_id: Option<&str>,
        group_id: Option<&str>,
        metadata: Option<&Map<String, Value>>,
        tracing_api_key: Option<&str>,
    ) -> bool {
        if self.trace_id.is_none() || self.trace_id.as_deref() != trace_id {
            return false;
        }
        if self.workflow_name.as_deref() != Some(workflow_name) {
            return false;
        }
        if self.group_id.as_deref() != group_id {
            return false;
        }
        if self.metadata.as_ref() != metadata {
            return false;
        }
        if self.tracing_api_key.is_some() {
            return self.tracing_api_key.as_deref() == tracing_api_key;
        }
        if self.tracing_api_key_hash.is_some() {
            return self.tracing_api_key_hash == hash_tracing_api_key(tracing_api_key);
        }
        tracing_api_key.is_none()
    }
}

/// Rebuild a live trace from persisted state without notifying processors.
/// Port of `reattach_trace`.
pub fn reattach_trace(state: &TraceState, tracing_api_key: Option<String>) -> Option<Trace> {
    let trace_id = state.trace_id.clone()?;
    Some(Trace::reattached(
        state
            .workflow_name
            .clone()
            .unwrap_or_else(|| "Agent workflow".to_string()),
        trace_id,
        state.group_id.clone(),
        state.metadata.clone(),
        state.tracing_api_key.clone().or(tracing_api_key),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trips_a_payload_and_keeps_unknown_keys() {
        let payload = json!({
            "object": "trace",
            "id": "trace_abc",
            "workflow_name": "wf",
            "group_id": "group_1",
            "metadata": {"k": "v"},
            "tracing_api_key": "sk-test",
            "custom": 7,
        });
        let state = TraceState::from_json(Some(&payload)).unwrap();
        assert_eq!(state.trace_id.as_deref(), Some("trace_abc"));
        assert_eq!(state.extra.get("custom"), Some(&json!(7)));
        assert_eq!(
            state.tracing_api_key_hash,
            hash_tracing_api_key(Some("sk-test"))
        );
        // Without the key the fingerprint still travels.
        let stripped = state.to_json(false).unwrap();
        assert!(stripped.get("tracing_api_key").is_none());
        assert!(stripped.get("tracing_api_key_hash").is_some());
        assert_eq!(stripped["custom"], json!(7));
        assert_eq!(state.to_json(true).unwrap()["tracing_api_key"], "sk-test");
        assert!(TraceState::from_json(None).is_none());
        assert!(TraceState::default().to_json(true).is_none());
    }

    #[test]
    fn matches_only_the_same_effective_settings() {
        let state = TraceState::from_json(Some(&json!({
            "object": "trace", "id": "trace_abc", "workflow_name": "wf",
            "group_id": null, "metadata": null, "tracing_api_key": "sk-test",
        })))
        .unwrap();
        assert!(state.matches_effective_settings(
            "wf",
            Some("trace_abc"),
            None,
            None,
            Some("sk-test")
        ));
        assert!(!state.matches_effective_settings(
            "other",
            Some("trace_abc"),
            None,
            None,
            Some("sk-test")
        ));
        assert!(!state.matches_effective_settings(
            "wf",
            Some("trace_zzz"),
            None,
            None,
            Some("sk-test")
        ));
        assert!(!state.matches_effective_settings("wf", Some("trace_abc"), None, None, None));

        // A stripped snapshot still matches through the fingerprint.
        let mut stripped = state.clone();
        stripped.tracing_api_key = None;
        assert!(stripped.matches_effective_settings(
            "wf",
            Some("trace_abc"),
            None,
            None,
            Some("sk-test")
        ));
        assert!(!stripped.matches_effective_settings(
            "wf",
            Some("trace_abc"),
            None,
            None,
            Some("sk-other")
        ));
    }

    #[test]
    fn reattaches_without_re_emitting_the_trace() {
        let state = TraceState::from_json(Some(&json!({
            "object": "trace", "id": "trace_reattach_1", "workflow_name": "wf",
        })))
        .unwrap();
        let trace = reattach_trace(&state, Some("sk-test".into())).unwrap();
        assert!(trace.is_reattached());
        assert_eq!(trace.trace_id(), "trace_reattach_1");
        assert_eq!(trace.tracing_api_key(), Some("sk-test"));
        assert!(!trace_id_was_started(Some("trace_reattach_1")));
        trace.start();
        assert!(trace_id_was_started(Some("trace_reattach_1")));
        trace.finish();
        assert!(!trace_id_was_started(Some("no-op")));
        assert!(!trace_id_was_started(None));
    }
}
