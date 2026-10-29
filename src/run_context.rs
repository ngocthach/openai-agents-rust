//! Run context: the app-defined context object plus SDK runtime metadata. Mirrors
//! `agents.run_context.RunContextWrapper`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::items::ToolApprovalItem;
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

/// Serializable approval state keyed by `"{agent}\u{1}{tool}"`.
pub type ApprovalsState = BTreeMap<String, ApprovalRecord>;

fn approval_key(agent: &str, tool_name: &str) -> String {
    format!("{agent}\u{1}{tool_name}")
}

#[derive(Default)]
struct Inner {
    usage: Usage,
    approvals: ApprovalsState,
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
        self.inner.lock().expect("run context poisoned").usage.clone()
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
        let mut inner = self.inner.lock().expect("run context poisoned");
        let rec = inner
            .approvals
            .entry(approval_key(&item.agent, &item.tool_name))
            .or_default();
        if always_approve {
            rec.approved_always = true;
        }
        rec.approved_call_ids.insert(item.call_id().to_string());
        rec.rejected_call_ids.remove(item.call_id());
    }

    /// Reject a pending tool call, optionally with a custom model-visible message.
    pub fn reject_tool(
        &self,
        item: &ToolApprovalItem,
        always_reject: bool,
        rejection_message: Option<String>,
    ) {
        let mut inner = self.inner.lock().expect("run context poisoned");
        let rec = inner
            .approvals
            .entry(approval_key(&item.agent, &item.tool_name))
            .or_default();
        if always_reject {
            rec.rejected_always = true;
            if rejection_message.is_some() {
                rec.sticky_rejection_message = rejection_message.clone();
            }
        }
        rec.rejected_call_ids.insert(item.call_id().to_string());
        rec.approved_call_ids.remove(item.call_id());
        if let Some(msg) = rejection_message {
            rec.rejection_messages.insert(item.call_id().to_string(), msg);
        }
    }

    /// Approval status for a tool call: `Some(true)` approved, `Some(false)` rejected, `None`
    /// undecided.
    pub fn is_tool_approved(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<bool> {
        let inner = self.inner.lock().expect("run context poisoned");
        let rec = inner.approvals.get(&approval_key(agent, tool_name))?;
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

    /// Custom rejection message for a rejected call, if one was provided.
    pub fn rejection_message(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<String> {
        let inner = self.inner.lock().expect("run context poisoned");
        let rec = inner.approvals.get(&approval_key(agent, tool_name))?;
        rec.rejection_messages
            .get(call_id)
            .cloned()
            .or_else(|| rec.sticky_rejection_message.clone())
    }

    /// Export approval state (for `RunState`).
    pub fn export_approvals(&self) -> ApprovalsState {
        self.inner.lock().expect("run context poisoned").approvals.clone()
    }

    /// Import approval state (for `RunState`), merging with existing decisions.
    pub fn import_approvals(&self, approvals: &ApprovalsState) {
        let mut inner = self.inner.lock().expect("run context poisoned");
        for (k, v) in approvals {
            let rec = inner.approvals.entry(k.clone()).or_default();
            rec.approved_always |= v.approved_always;
            rec.rejected_always |= v.rejected_always;
            rec.approved_call_ids.extend(v.approved_call_ids.iter().cloned());
            rec.rejected_call_ids.extend(v.rejected_call_ids.iter().cloned());
            rec.rejection_messages
                .extend(v.rejection_messages.iter().map(|(a, b)| (a.clone(), b.clone())));
            if v.sticky_rejection_message.is_some() {
                rec.sticky_rejection_message = v.sticky_rejection_message.clone();
            }
        }
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
        };
        assert_eq!(ctx.is_tool_approved("a", "t", item.call_id()), None);
        ctx.approve_tool(&item, false);
        assert_eq!(ctx.is_tool_approved("a", "t", item.call_id()), Some(true));
        assert_eq!(ctx.is_tool_approved("a", "t", "other"), None);
        assert_eq!(ctx.is_tool_approved("b", "t", item.call_id()), None);
        ctx.reject_tool(&item, true, Some("no".into()));
        assert_eq!(ctx.is_tool_approved("a", "t", "other"), Some(false));
        assert_eq!(ctx.rejection_message("a", "t", "other").as_deref(), Some("no"));
    }
}
