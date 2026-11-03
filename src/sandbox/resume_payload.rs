//! The per-agent sandbox section of a run state: Python's `RunState._sandbox`.
//!
//! The section maps each agent of a handoff graph to the session state its sandbox was
//! suspended with, so a resumed multi-agent run restores every agent's session rather than only
//! the starting agent's. Port of the payload logic in
//! `agents.sandbox.runtime_session_manager.SandboxRuntimeSessionManager`:
//!
//! ```json
//! {
//!   "backend_id": "unix_local",
//!   "current_agent_key": "worker",
//!   "current_agent_name": "worker",
//!   "session_state": { ... },
//!   "sessions_by_agent": { "triage": { "agent_name": "triage", "session_state": { ... } } }
//! }
//! ```
//!
//! Keys are the agent identity keys (`name`, `name#2`, ...). A snapshot written by another
//! build may name a session under a different key, so the key an agent resumes from is matched
//! by key first and by agent name second. Sessions that no agent of the current graph claims are
//! written back untouched.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::{Map, Value};

use crate::error::{AgentsError, Result};
use crate::sandbox::state::SandboxSessionState;

const BACKEND_MISMATCH: &str =
    "RunState sandbox backend does not match the configured sandbox client";
const MISSING_SESSION_STATE: &str = "RunState sandbox payload is missing `session_state`";
const INVALID_SESSION_ITEM: &str =
    "RunState sandbox payload has an invalid `sessions_by_agent` item";
const SCHEMA_VERSION_KEY: &str = "$schemaVersion";

/// One session a run suspended, as the section stores it.
pub(crate) struct SuspendedSession<'a> {
    /// Identity key of the agent the session belongs to.
    pub key: &'a str,
    /// The agent's own name.
    pub name: &'a str,
    pub state: &'a SandboxSessionState,
}

/// The sandbox section a run was resumed from, plus which of its entries this run claimed.
#[derive(Debug, Default)]
pub(crate) struct ResumeSource {
    payload: Option<Map<String, Value>>,
    /// The agent the saved run stopped at: its identity key, else its name.
    current_hint: Option<String>,
    current_name: Option<String>,
    claimed: BTreeSet<String>,
    /// Graph key to the stored key the agent's session is filed under.
    assigned: HashMap<String, String>,
    /// Graph key to the stored key its session was resumed from.
    sources: HashMap<String, String>,
}

impl ResumeSource {
    pub(crate) fn new(
        payload: Option<&Value>,
        current_agent_identity: Option<&str>,
        current_agent_name: Option<&str>,
    ) -> Self {
        ResumeSource {
            payload: payload.and_then(Value::as_object).cloned(),
            current_hint: current_agent_identity
                .or(current_agent_name)
                .map(str::to_string),
            current_name: current_agent_name.map(str::to_string),
            ..Default::default()
        }
    }

    /// Whether the section saved a rollout id for the session of agent `key`.
    pub(crate) fn has_rollout_id(&self, key: &str) -> bool {
        let Some(payload) = &self.payload else {
            return false;
        };
        let has = |session: Option<&Value>| {
            let session = session
                .and_then(|entry| entry.get("session_state").or(Some(entry)))
                .and_then(Value::as_object);
            session.is_some_and(|session| session.get("rollout_id").is_some_and(Value::is_string))
        };
        if let Some(sessions) = self.sessions() {
            return has(sessions.get(key));
        }
        payload.get("current_agent_key").and_then(Value::as_str) == Some(key)
            && has(payload.get("session_state"))
    }

    /// The session state saved for the agent with identity key `graph_key`, if any.
    ///
    /// A section written for another backend is refused. An entry that is not in this crate's
    /// session-state format (a snapshot another SDK wrote) is skipped, so that agent gets a
    /// fresh session.
    pub(crate) fn state_for(
        &mut self,
        backend_id: &str,
        graph_key: &str,
        name: &str,
    ) -> Result<Option<SandboxSessionState>> {
        let Some(payload) = &self.payload else {
            return Ok(None);
        };
        if payload.get("backend_id").and_then(Value::as_str) != Some(backend_id) {
            return Err(AgentsError::user(BACKEND_MISMATCH));
        }
        let has_sessions = matches!(payload.get("sessions_by_agent"), Some(Value::Object(_)));
        if has_sessions && let Some(key) = self.assign(graph_key, name) {
            let entry = self
                .sessions()
                .and_then(|sessions| sessions.get(&key).cloned());
            if let Some(session) = session_payload_from_entry(entry.as_ref())? {
                self.sources.insert(graph_key.to_string(), key);
                return parse_session_state(&session);
            }
        }
        let payload = self.payload.as_ref().expect("checked above");
        let session = match payload.get("session_state") {
            None | Some(Value::Null) => return Ok(None),
            Some(Value::Object(session)) => session.clone(),
            Some(_) => return Err(AgentsError::user(MISSING_SESSION_STATE)),
        };
        if let Some(current_key) = payload
            .get("current_agent_key")
            .and_then(Value::as_str)
            .map(str::to_string)
        {
            if self.assign(graph_key, name).as_deref() != Some(current_key.as_str()) {
                return Ok(None);
            }
            self.sources.insert(graph_key.to_string(), current_key);
            return parse_session_state(&session);
        }
        let current_name = payload
            .get("current_agent_name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.current_name.clone());
        match current_name {
            Some(current_name) if current_name != name => Ok(None),
            Some(current_name) => {
                self.sources.insert(graph_key.to_string(), current_name);
                parse_session_state(&session)
            }
            None => parse_session_state(&session),
        }
    }

    /// The key the saved section files an agent's session under: the agent's own key when an
    /// entry for the same agent exists there, else the saved current-agent key for the agent the
    /// run stopped at, else the first unclaimed entry for an agent of that name.
    fn assign(&mut self, graph_key: &str, name: &str) -> Option<String> {
        if let Some(existing) = self.assigned.get(graph_key) {
            return Some(existing.clone());
        }
        self.payload.as_ref()?;
        let current_key = self
            .payload
            .as_ref()
            .and_then(|p| p.get("current_agent_key"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let is_current = self.current_hint.as_deref() == Some(graph_key);
        let claim = |this: &mut Self, key: &str| {
            this.claimed.insert(key.to_string());
            this.assigned.insert(graph_key.to_string(), key.to_string());
            Some(key.to_string())
        };
        if !self.claimed.contains(graph_key) && self.entry_matches(graph_key, name) {
            return claim(self, graph_key);
        }
        if let Some(current_key) = &current_key
            && is_current
            && !self.claimed.contains(current_key)
            && self.entry_matches(current_key, name)
        {
            return claim(self, current_key);
        }
        let candidates = self.keys_by_name().remove(name).unwrap_or_default();
        for key in candidates {
            if self.claimed.contains(&key) {
                continue;
            }
            if current_key.as_deref() == Some(key.as_str()) && !is_current {
                continue;
            }
            return claim(self, &key);
        }
        None
    }

    fn sessions(&self) -> Option<&Map<String, Value>> {
        self.payload
            .as_ref()?
            .get("sessions_by_agent")
            .and_then(Value::as_object)
    }

    fn entry_matches(&self, key: &str, name: &str) -> bool {
        self.sessions()
            .and_then(|sessions| sessions.get(key))
            .and_then(|entry| entry_agent_name(key, entry))
            .as_deref()
            == Some(name)
    }

    fn keys_by_name(&self) -> BTreeMap<String, Vec<String>> {
        let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Some(sessions) = self.sessions() {
            for (key, entry) in sessions {
                if let Some(name) = entry_agent_name(key, entry) {
                    grouped.entry(name).or_default().push(key.clone());
                }
            }
        }
        grouped
    }

    /// Entries of the section being resumed, as the next section starts from them.
    fn legacy_entries(&self) -> Map<String, Value> {
        let Some(payload) = &self.payload else {
            return Map::new();
        };
        if let Some(Value::Object(sessions)) = payload.get("sessions_by_agent") {
            return sessions.clone();
        }
        let Some(Value::Object(session)) = payload.get("session_state") else {
            return Map::new();
        };
        let key = payload
            .get("current_agent_key")
            .and_then(Value::as_str)
            .or_else(|| payload.get("current_agent_name").and_then(Value::as_str))
            .or(self.current_name.as_deref());
        match key {
            Some(key) => Map::from_iter([(key.to_string(), Value::Object(session.clone()))]),
            None => Map::new(),
        }
    }

    fn used_keys(&self) -> BTreeSet<String> {
        let mut used: BTreeSet<String> = self.legacy_entries().keys().cloned().collect();
        used.extend(self.assigned.values().cloned());
        used
    }

    /// The key an agent's session is written under.
    fn resume_key(&mut self, graph_key: &str, name: &str) -> String {
        if let Some(existing) = self.assigned.get(graph_key) {
            return existing.clone();
        }
        if !self.used_keys().contains(graph_key) {
            self.assigned
                .insert(graph_key.to_string(), graph_key.to_string());
            return graph_key.to_string();
        }
        if let Some(key) = self.assign(graph_key, name) {
            return key;
        }
        let used = self.used_keys();
        let mut candidate = name.to_string();
        let mut index = 1;
        while used.contains(&candidate) {
            index += 1;
            candidate = format!("{name}#{index}");
        }
        self.assigned
            .insert(graph_key.to_string(), candidate.clone());
        candidate
    }

    /// The section a run that suspended `sessions` writes.
    ///
    /// Sessions of the section being resumed that this run did not touch are kept; a session
    /// resumed from one key and now filed under another leaves its old entry behind.
    pub(crate) fn serialize(
        &mut self,
        backend_id: &str,
        current: (&str, &str),
        current_state: &SandboxSessionState,
        sessions: &[SuspendedSession<'_>],
    ) -> Result<Value> {
        let mut by_agent = self.legacy_entries();
        for session in sessions {
            let key = self.resume_key(session.key, session.name);
            if let Some(source) = self.sources.get(session.key)
                && *source != key
            {
                by_agent.shift_remove(source);
            }
            let mut entry = Map::new();
            entry.insert("agent_name".into(), Value::String(session.name.to_string()));
            entry.insert("session_state".into(), state_value(session.state)?);
            by_agent.insert(key, Value::Object(entry));
        }
        let current_key = self.resume_key(current.0, current.1);
        let mut payload = Map::new();
        payload.insert("backend_id".into(), Value::String(backend_id.to_string()));
        payload.insert("current_agent_key".into(), Value::String(current_key));
        payload.insert(
            "current_agent_name".into(),
            Value::String(current.1.to_string()),
        );
        payload.insert("session_state".into(), state_value(current_state)?);
        payload.insert("sessions_by_agent".into(), Value::Object(by_agent));
        Ok(Value::Object(payload))
    }
}

fn state_value(state: &SandboxSessionState) -> Result<Value> {
    let text = state.to_json_string()?;
    serde_json::from_str(&text)
        .map_err(|_| AgentsError::user("sandbox session state could not be serialized"))
}

/// The agent a stored entry belongs to: its recorded name, else its key.
fn entry_agent_name(key: &str, entry: &Value) -> Option<String> {
    let entry = entry.as_object()?;
    match (entry.get("agent_name"), entry.get("session_state")) {
        (Some(Value::String(name)), Some(Value::Object(_))) => Some(name.clone()),
        _ => Some(key.to_string()),
    }
}

/// The session state of a stored entry: its `session_state`, or the entry itself when it is a
/// bare state.
fn session_payload_from_entry(entry: Option<&Value>) -> Result<Option<Map<String, Value>>> {
    match entry {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(entry)) => Ok(Some(match entry.get("session_state") {
            Some(Value::Object(state)) => state.clone(),
            _ => entry.clone(),
        })),
        Some(_) => Err(AgentsError::user(INVALID_SESSION_ITEM)),
    }
}

fn parse_session_state(session: &Map<String, Value>) -> Result<Option<SandboxSessionState>> {
    // A state another SDK wrote carries no schema version and cannot be reattached here.
    if !session.contains_key(SCHEMA_VERSION_KEY) {
        return Ok(None);
    }
    Ok(Some(SandboxSessionState::from_json_str(
        &Value::Object(session.clone()).to_string(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::manifest::Manifest;
    use serde_json::json;

    fn state(id: &str) -> SandboxSessionState {
        SandboxSessionState::new("fake", id, "/workspace", &Manifest::new())
    }

    fn payload(current_key: &str, current: &SandboxSessionState) -> Value {
        json!({
            "backend_id": "fake",
            "current_agent_key": current_key,
            "current_agent_name": "sandbox",
            "session_state": state_value(current).unwrap(),
            "sessions_by_agent": {
                "sandbox": {"agent_name": "sandbox", "session_state": state_value(&state("first")).unwrap()},
                "sandbox#2": {"agent_name": "sandbox", "session_state": state_value(&state("second")).unwrap()},
            },
        })
    }

    fn id_of(state: Option<SandboxSessionState>) -> Option<String> {
        state.map(|state| state.session_id)
    }

    /// Python: `test_session_manager_reserves_current_duplicate_resume_key_for_current_agent`.
    #[test]
    fn the_current_duplicate_keeps_its_own_key() {
        let section = payload("sandbox#2", &state("second"));
        let mut source = ResumeSource::new(Some(&section), Some("sandbox#2"), Some("sandbox"));
        // Asking for the first agent last must not hand it the current agent's entry.
        assert_eq!(
            id_of(source.state_for("fake", "sandbox", "sandbox").unwrap()).as_deref(),
            Some("first")
        );
        assert_eq!(
            id_of(source.state_for("fake", "sandbox#2", "sandbox").unwrap()).as_deref(),
            Some("second")
        );
    }

    /// Python: `test_session_manager_uses_run_state_starting_agent_for_duplicate_resume_keys`:
    /// the run stopped at an agent that holds no session, and each duplicate still finds its
    /// own entry by key.
    #[test]
    fn duplicates_find_their_entries_when_the_current_agent_has_no_session() {
        let section = payload("sandbox#2", &state("second"));
        let mut source = ResumeSource::new(Some(&section), Some("approver"), Some("approver"));
        assert_eq!(
            id_of(source.state_for("fake", "sandbox#2", "sandbox").unwrap()).as_deref(),
            Some("second")
        );
        assert_eq!(
            id_of(source.state_for("fake", "sandbox", "sandbox").unwrap()).as_deref(),
            Some("first")
        );
    }

    /// Python: `test_session_manager_restores_duplicate_name_sessions_when_only_sandbox_config_differs`:
    /// a restored graph whose keys come out in the other order still hands each agent the entry
    /// recorded for its own key.
    #[test]
    fn entries_are_matched_by_key_before_name() {
        let section = payload("sandbox#2", &state("second"));
        let mut source = ResumeSource::new(Some(&section), Some("sandbox#2"), Some("sandbox"));
        assert_eq!(
            id_of(source.state_for("fake", "sandbox#2", "sandbox").unwrap()).as_deref(),
            Some("second")
        );
        assert_eq!(
            id_of(source.state_for("fake", "sandbox", "sandbox").unwrap()).as_deref(),
            Some("first")
        );
    }

    /// A key the saved section does not hold falls back to an unclaimed entry of that name; the
    /// entry of the agent the saved run stopped at stays reserved for that agent.
    #[test]
    fn an_unknown_key_claims_an_unclaimed_entry_of_its_name() {
        let section = payload("sandbox", &state("first"));
        let mut source = ResumeSource::new(Some(&section), Some("sandbox"), Some("sandbox"));
        assert_eq!(
            id_of(source.state_for("fake", "renamed", "sandbox").unwrap()).as_deref(),
            Some("second")
        );
        assert!(
            source
                .state_for("fake", "renamed#2", "sandbox")
                .unwrap()
                .is_none(),
            "the current agent's entry is not handed to another agent"
        );
    }

    #[test]
    fn a_section_for_another_backend_is_refused() {
        let section = payload("sandbox", &state("first"));
        let mut source = ResumeSource::new(Some(&section), None, None);
        let error = source
            .state_for("docker", "sandbox", "sandbox")
            .unwrap_err();
        assert_eq!(error.to_string(), BACKEND_MISMATCH);
    }

    /// Without `sessions_by_agent` only the agent the saved run stopped at resumes; the others
    /// start fresh (Python's legacy payload).
    #[test]
    fn a_legacy_section_resumes_only_the_current_agent() {
        let section = json!({
            "backend_id": "fake",
            "current_agent_name": "worker",
            "session_state": state_value(&state("legacy")).unwrap(),
        });
        let mut source = ResumeSource::new(Some(&section), None, Some("worker"));
        assert_eq!(
            id_of(source.state_for("fake", "worker", "worker").unwrap()).as_deref(),
            Some("legacy")
        );
        assert!(
            source
                .state_for("fake", "triage", "triage")
                .unwrap()
                .is_none()
        );
    }

    /// A session state with no `$schemaVersion` (another SDK wrote it) is not resumable here: the
    /// agent starts a fresh session.
    #[test]
    fn a_foreign_session_state_is_skipped() {
        let section = json!({
            "backend_id": "fake",
            "sessions_by_agent": {"sandbox": {"agent_name": "sandbox", "session_state": {"snapshot": {"id": "x"}}}},
        });
        let mut source = ResumeSource::new(Some(&section), None, None);
        assert!(
            source
                .state_for("fake", "sandbox", "sandbox")
                .unwrap()
                .is_none()
        );
    }

    /// Python: `test_session_manager_preserves_untouched_run_state_sessions_on_cleanup`.
    #[test]
    fn untouched_entries_are_written_back_and_touched_ones_replaced() {
        let section = payload("sandbox", &state("first"));
        let mut source = ResumeSource::new(Some(&section), Some("sandbox"), Some("sandbox"));
        let fresh = state("fresh");
        let written = source
            .serialize(
                "fake",
                ("sandbox", "sandbox"),
                &fresh,
                &[SuspendedSession {
                    key: "sandbox",
                    name: "sandbox",
                    state: &fresh,
                }],
            )
            .unwrap();
        let sessions = written["sessions_by_agent"].as_object().unwrap();
        assert_eq!(sessions["sandbox"]["session_state"]["session_id"], "fresh");
        assert_eq!(
            sessions["sandbox#2"],
            section["sessions_by_agent"]["sandbox#2"]
        );
        assert_eq!(written["current_agent_key"], "sandbox");
        assert_eq!(written["session_state"]["session_id"], "fresh");
    }

    /// Python: `test_session_manager_generates_collision_free_resume_keys_for_literal_suffix_names`:
    /// a key already held by a saved entry for another agent is never reused.
    #[test]
    fn a_new_agent_never_takes_a_key_held_by_another_entry() {
        let section = json!({
            "backend_id": "fake",
            "sessions_by_agent": {
                "sandbox": {"agent_name": "sandbox", "session_state": state_value(&state("a")).unwrap()},
                "sandbox#2": {"agent_name": "sandbox#2", "session_state": state_value(&state("b")).unwrap()},
            },
        });
        let mut source = ResumeSource::new(Some(&section), None, None);
        let fresh = state("fresh");
        // A third agent called `sandbox`, with a graph key that is already taken by saved
        // sessions of other agents and that no entry matches by name: it gets `sandbox#3`.
        let written = source
            .serialize(
                "fake",
                ("graph-key", "sandbox"),
                &fresh,
                &[SuspendedSession {
                    key: "graph-key",
                    name: "sandbox",
                    state: &fresh,
                }],
            )
            .unwrap();
        let keys: Vec<&str> = written["sessions_by_agent"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert!(
            keys.contains(&"sandbox") && keys.contains(&"sandbox#2"),
            "{keys:?}"
        );
        assert_eq!(keys.len(), 3, "{keys:?}");
    }
}
