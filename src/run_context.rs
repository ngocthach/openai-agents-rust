//! Run context: the app-defined context object plus SDK runtime metadata. Mirrors
//! `agents.run_context.RunContextWrapper`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::items::{InputItem, ToolApprovalItem};
use crate::usage::Usage;

/// Approval decisions for one tool identity (`agent name` + `tool name`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRecord {
    /// `true` means always approved; otherwise the set of approved call ids.
    #[serde(default)]
    pub approved_always: bool,
    #[serde(default)]
    pub approved_call_ids: BTreeSet<String>,
    /// `true` means always rejected; otherwise the set of rejected call ids.
    #[serde(default)]
    pub rejected_always: bool,
    #[serde(default)]
    pub rejected_call_ids: BTreeSet<String>,
    /// Custom rejection messages keyed by call id.
    #[serde(default)]
    pub rejection_messages: BTreeMap<String, String>,
    /// Rejection message for a sticky rejection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sticky_rejection_message: Option<String>,
}

/// The decision recorded for one tool call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalStatus {
    /// `Some(true)` approved, `Some(false)` rejected, `None` undecided.
    pub approved: Option<bool>,
    /// The model-visible rejection message, when one was recorded.
    pub rejection_message: Option<String>,
}

/// Serializable approval decisions keyed by tool identity (`agent` + `tool`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ApprovalsState(pub BTreeMap<String, ApprovalRecord>);

fn approval_key(agent: &str, tool_name: &str) -> String {
    format!("{agent}\u{1}{tool_name}")
}

/// Tool names whose sticky decisions are scoped by invocation type: the native `apply_patch_call`
/// tool and a custom tool can both be called `apply_patch`, and approving one must not approve the
/// other. Mirrors `RunContextWrapper._approval_keys_for_scope`.
const SCOPED_APPROVAL_TOOL: &str = "apply_patch";

/// The key a decision about `tool_name` is recorded under, given how the tool was invoked
/// (`apply_patch_call`, `custom_tool_call`, `function_call`, ...). Only `apply_patch` is scoped;
/// every other name keys as itself.
pub fn scoped_tool_key(tool_name: &str, invocation_type: Option<&str>) -> String {
    if tool_name == SCOPED_APPROVAL_TOOL {
        format!("{tool_name}:{}", invocation_type.unwrap_or("function_call"))
    } else {
        tool_name.to_string()
    }
}

/// [`scoped_tool_key`] for the tool call an approval item stands for.
pub fn approval_tool_key(tool_name: &str, call: &crate::items::FunctionCall) -> String {
    scoped_tool_key(tool_name, call.extra.get("type").and_then(Value::as_str))
}

/// The names a lookup consults, most specific first: a scoped `apply_patch:<type>` key falls back
/// to the bare name for decisions recorded before scoping existed.
fn lookup_names(tool_name: &str) -> impl Iterator<Item = &str> {
    let bare = tool_name
        .strip_prefix(SCOPED_APPROVAL_TOOL)
        .filter(|rest| rest.starts_with(':'))
        .map(|_| SCOPED_APPROVAL_TOOL);
    std::iter::once(tool_name).chain(bare)
}

impl ApprovalsState {
    /// Approve a call; `always` approves future calls to the same tool by the same agent.
    pub fn approve(&mut self, item: &ToolApprovalItem, always: bool) {
        let rec = self
            .0
            .entry(approval_key(
                &item.agent,
                &approval_tool_key(&item.tool_name, &item.raw_item),
            ))
            .or_default();
        if always {
            rec.approved_always = true;
            rec.rejected_always = false;
        }
        rec.approved_call_ids.insert(item.call_id().to_string());
        rec.rejected_call_ids.remove(item.call_id());
    }

    /// Reject a call; `always` rejects future calls to the same tool by the same agent.
    pub fn reject(&mut self, item: &ToolApprovalItem, always: bool, message: Option<String>) {
        let rec = self
            .0
            .entry(approval_key(
                &item.agent,
                &approval_tool_key(&item.tool_name, &item.raw_item),
            ))
            .or_default();
        if always {
            rec.rejected_always = true;
            rec.approved_always = false;
            if message.is_some() {
                rec.sticky_rejection_message = message.clone();
            }
        }
        rec.rejected_call_ids.insert(item.call_id().to_string());
        rec.approved_call_ids.remove(item.call_id());
        if let Some(m) = message {
            rec.rejection_messages.insert(item.call_id().to_string(), m);
        }
    }

    fn record_status(rec: &ApprovalRecord, call_id: &str) -> Option<bool> {
        if rec.rejected_call_ids.contains(call_id) {
            return Some(false);
        }
        if rec.approved_call_ids.contains(call_id) {
            return Some(true);
        }
        if rec.rejected_always {
            return Some(false);
        }
        if rec.approved_always {
            return Some(true);
        }
        None
    }

    /// `Some(true)` approved, `Some(false)` rejected, `None` undecided.
    ///
    /// Decisions recorded without an owning agent (Python's agent-agnostic `approvals`
    /// bucket, and snapshots written before it bound function tools to their agent) apply to
    /// every agent and are consulted when the agent-specific record is undecided.
    pub fn status(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<bool> {
        let decided = |owner: &str| {
            lookup_names(tool_name).find_map(|name| {
                self.0
                    .get(&approval_key(owner, name))
                    .and_then(|rec| Self::record_status(rec, call_id))
            })
        };
        decided(agent).or_else(|| decided(""))
    }

    /// Custom rejection message for a call.
    pub fn rejection_message(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<String> {
        let message = |owner: &str| {
            lookup_names(tool_name).find_map(|name| {
                self.0.get(&approval_key(owner, name)).and_then(|rec| {
                    rec.rejection_messages
                        .get(call_id)
                        .cloned()
                        .or_else(|| rec.sticky_rejection_message.clone())
                })
            })
        };
        message(agent).or_else(|| message(""))
    }

    /// Merge decisions from another state.
    pub fn merge(&mut self, other: &ApprovalsState) {
        for (k, v) in &other.0 {
            let rec = self.0.entry(k.clone()).or_default();
            rec.approved_always |= v.approved_always;
            rec.rejected_always |= v.rejected_always;
            rec.approved_call_ids
                .extend(v.approved_call_ids.iter().cloned());
            rec.rejected_call_ids
                .extend(v.rejected_call_ids.iter().cloned());
            for id in &v.approved_call_ids {
                if !v.rejected_call_ids.contains(id) {
                    rec.rejected_call_ids.remove(id);
                }
            }
            for id in &v.rejected_call_ids {
                if !v.approved_call_ids.contains(id) {
                    rec.approved_call_ids.remove(id);
                }
            }
            rec.rejection_messages.extend(
                v.rejection_messages
                    .iter()
                    .map(|(a, b)| (a.clone(), b.clone())),
            );
            if v.sticky_rejection_message.is_some() {
                rec.sticky_rejection_message = v.sticky_rejection_message.clone();
            }
        }
    }
}

#[derive(Default)]
struct Inner {
    usage: Usage,
    /// The input list of the turn currently being executed.
    turn_input: Vec<InputItem>,
    approvals: ApprovalsState,
    /// Serialized nested `Agent::as_tool` run states awaiting approval, keyed by tool call id.
    nested_states: BTreeMap<String, Value>,
    /// Interruptions raised by nested agent tool runs, keyed by tool call id.
    nested_interruptions: BTreeMap<String, Vec<ToolApprovalItem>>,
}

/// Wrapper around the app context with shared usage and approval state.
///
/// Cloning is cheap: all clones share the same context, usage and approvals.
pub struct RunContext<C> {
    context: Arc<C>,
    inner: Arc<Mutex<Inner>>,
    tool_input: Option<Arc<Value>>,
}

impl<C> Clone for RunContext<C> {
    fn clone(&self) -> Self {
        RunContext {
            context: self.context.clone(),
            inner: self.inner.clone(),
            tool_input: self.tool_input.clone(),
        }
    }
}

impl<C> fmt::Debug for RunContext<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunContext")
            .field("usage", &self.usage())
            .finish_non_exhaustive()
    }
}

impl<C> RunContext<C> {
    /// Wrap an app context.
    pub fn new(context: C) -> Self {
        RunContext {
            context: Arc::new(context),
            inner: Arc::new(Mutex::new(Inner::default())),
            tool_input: None,
        }
    }

    /// Wrap an already shared app context.
    pub fn from_arc(context: Arc<C>) -> Self {
        RunContext {
            context,
            inner: Arc::new(Mutex::new(Inner::default())),
            tool_input: None,
        }
    }

    /// The app-defined context object.
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Shared handle to the app context.
    pub fn context_arc(&self) -> Arc<C> {
        self.context.clone()
    }

    /// Snapshot of the usage accumulated so far.
    pub fn usage(&self) -> Usage {
        self.inner
            .lock()
            .expect("run context poisoned")
            .usage
            .clone()
    }

    /// Add usage to the run total.
    pub fn add_usage(&self, usage: &Usage) {
        self.inner
            .lock()
            .expect("run context poisoned")
            .usage
            .add(usage);
    }

    /// Replace the usage total (used when resuming from a state).
    pub fn set_usage(&self, usage: Usage) {
        self.inner.lock().expect("run context poisoned").usage = usage;
    }

    /// Structured input of the current `Agent::as_tool` invocation, if any.
    pub fn tool_input(&self) -> Option<&Value> {
        self.tool_input.as_deref()
    }

    /// Return a clone of this context that carries the given nested tool input.
    pub fn with_tool_input(&self, tool_input: Value) -> Self {
        RunContext {
            context: self.context.clone(),
            inner: self.inner.clone(),
            tool_input: Some(Arc::new(tool_input)),
        }
    }

    /// Approve a pending tool call. With `always_approve`, future calls to the same tool by the
    /// same agent are approved automatically.
    pub fn approve_tool(&self, item: &ToolApprovalItem, always_approve: bool) {
        self.lock().approvals.approve(item, always_approve);
    }

    /// Reject a pending tool call, optionally with a custom model-visible message.
    pub fn reject_tool(
        &self,
        item: &ToolApprovalItem,
        always_reject: bool,
        rejection_message: Option<String>,
    ) {
        self.lock()
            .approvals
            .reject(item, always_reject, rejection_message);
    }

    /// Approval status for a tool call: `Some(true)` approved, `Some(false)` rejected, `None`
    /// undecided.
    pub fn is_tool_approved(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<bool> {
        self.lock().approvals.status(agent, tool_name, call_id)
    }

    /// Custom rejection message for a rejected call, if one was provided.
    pub fn rejection_message(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<String> {
        self.lock()
            .approvals
            .rejection_message(agent, tool_name, call_id)
    }

    /// The input list of the turn currently being executed, as the model sees it. Empty
    /// before the first model call. Mirrors `RunContextWrapper.turn_input`.
    pub fn turn_input(&self) -> Vec<InputItem> {
        self.lock().turn_input.clone()
    }

    /// Replace the current turn's input list.
    pub fn set_turn_input(&self, input: Vec<InputItem>) {
        self.lock().turn_input = input;
    }

    /// Approval status plus the rejection message for a tool call, in one lookup.
    /// Mirrors `RunContextWrapper.get_approval_status`.
    pub fn get_approval_status(
        &self,
        agent: &str,
        tool_name: &str,
        call_id: &str,
    ) -> ApprovalStatus {
        let inner = self.lock();
        ApprovalStatus {
            approved: inner.approvals.status(agent, tool_name, call_id),
            rejection_message: inner.approvals.rejection_message(agent, tool_name, call_id),
        }
    }

    /// Export approval state (for `RunState`).
    pub fn export_approvals(&self) -> ApprovalsState {
        self.lock().approvals.clone()
    }

    /// Import approval state (for `RunState`), merging with existing decisions.
    pub fn import_approvals(&self, approvals: &ApprovalsState) {
        self.lock().approvals.merge(approvals);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("run context poisoned")
    }

    pub(crate) fn nested_states(&self) -> BTreeMap<String, Value> {
        self.lock().nested_states.clone()
    }

    pub(crate) fn import_nested_states(&self, states: &BTreeMap<String, Value>) {
        self.lock()
            .nested_states
            .extend(states.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    pub(crate) fn take_nested_state(&self, call_id: &str) -> Option<Value> {
        self.lock().nested_states.remove(call_id)
    }

    pub(crate) fn set_nested_interruption(
        &self,
        call_id: &str,
        state: Value,
        interruptions: Vec<ToolApprovalItem>,
    ) {
        let mut inner = self.lock();
        inner.nested_states.insert(call_id.to_string(), state);
        inner
            .nested_interruptions
            .insert(call_id.to_string(), interruptions);
    }

    pub(crate) fn take_nested_interruptions(&self, call_id: &str) -> Option<Vec<ToolApprovalItem>> {
        self.lock().nested_interruptions.remove(call_id)
    }
}

impl<C: Default> Default for RunContext<C> {
    fn default() -> Self {
        RunContext::new(C::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::FunctionCall;

    #[test]
    fn approvals_are_scoped_per_agent_and_tool() {
        let ctx = RunContext::new(());
        let item = ToolApprovalItem {
            agent: "a".into(),
            raw_item: FunctionCall::new("t", "{}"),
            tool_name: "t".into(),
            tool_origin: None,
        };
        assert_eq!(ctx.is_tool_approved("a", "t", item.call_id()), None);
        ctx.approve_tool(&item, false);
        assert_eq!(ctx.is_tool_approved("a", "t", item.call_id()), Some(true));
        assert_eq!(ctx.is_tool_approved("a", "t", "other"), None);
        assert_eq!(ctx.is_tool_approved("b", "t", item.call_id()), None);
        ctx.reject_tool(&item, true, Some("no".into()));
        assert_eq!(ctx.is_tool_approved("a", "t", "other"), Some(false));
        assert_eq!(
            ctx.rejection_message("a", "t", "other").as_deref(),
            Some("no")
        );
    }
}
