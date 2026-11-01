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

/// Prefix of the key a deferred top-level function tool's decisions are recorded under. Mirrors
/// Python's `deferred_top_level:<name>` approval key.
const DEFERRED_TOP_LEVEL_PREFIX: &str = "deferred_top_level:";

/// [`scoped_tool_key`] for the tool call an approval item stands for.
///
/// A deferred top-level tool is called under a namespace equal to its own name. Its decisions
/// are recorded under their own key, so approving it permanently never authorizes a visible tool
/// that has the same bare name.
pub fn approval_tool_key(tool_name: &str, call: &crate::items::FunctionCall) -> String {
    let namespace = call
        .extra
        .get("namespace")
        .and_then(Value::as_str)
        .filter(|ns| !ns.is_empty());
    if namespace == Some(tool_name) {
        return format!("{DEFERRED_TOP_LEVEL_PREFIX}{tool_name}");
    }
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

/// Reserved marker that starts the tool half of a hosted MCP approval key. No tool name can
/// contain it, so hosted records never collide with function or native tool records.
const HOSTED_MARKER: char = '\u{2}';
/// Separator between the fields of a hosted MCP approval key.
const HOSTED_SEP: char = '\u{1f}';

/// Key of a sticky hosted MCP decision, scoped by server label and tool name. Hosted MCP
/// decisions belong to no agent. Mirrors Python's `("hosted_mcp", server_label, tool_name)`.
pub(crate) fn hosted_server_tool_key(server_label: &str, tool_name: &str) -> String {
    format!("\u{1}{HOSTED_MARKER}hosted_mcp{HOSTED_SEP}{server_label}{HOSTED_SEP}{tool_name}")
}

/// Key of an exact-call hosted MCP decision whose request carries no full identity. Mirrors
/// Python's `("hosted_mcp_call", request_id)`.
pub(crate) fn hosted_request_key(request_id: &str) -> String {
    format!("\u{1}{HOSTED_MARKER}hosted_mcp_call{HOSTED_SEP}{request_id}")
}

/// Key of the lookup-only mirror that lets a name-based query see one exact hosted MCP
/// decision. It never authorizes anything. Mirrors Python's `("hosted_mcp_query", tool, id)`.
pub(crate) fn hosted_query_key(tool_name: &str, request_id: &str) -> String {
    format!("\u{1}{HOSTED_MARKER}hosted_mcp_query{HOSTED_SEP}{tool_name}{HOSTED_SEP}{request_id}")
}

/// A typed hosted MCP approval identity, decoded from a reserved key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HostedApprovalIdentity {
    ServerTool {
        server_label: String,
        tool_name: String,
    },
    Request {
        request_id: String,
    },
    Query {
        tool_name: String,
        request_id: String,
    },
}

/// Decode a reserved hosted MCP key; `None` for function and native tool keys.
pub(crate) fn parse_hosted_key(key: &str) -> Option<HostedApprovalIdentity> {
    let rest = key.strip_prefix('\u{1}')?.strip_prefix(HOSTED_MARKER)?;
    let parts: Vec<&str> = rest.split(HOSTED_SEP).collect();
    match parts.as_slice() {
        ["hosted_mcp", label, tool] => Some(HostedApprovalIdentity::ServerTool {
            server_label: (*label).to_string(),
            tool_name: (*tool).to_string(),
        }),
        ["hosted_mcp_call", id] => Some(HostedApprovalIdentity::Request {
            request_id: (*id).to_string(),
        }),
        ["hosted_mcp_query", tool, id] => Some(HostedApprovalIdentity::Query {
            tool_name: (*tool).to_string(),
            request_id: (*id).to_string(),
        }),
        _ => None,
    }
}

/// The identity fields of a hosted MCP approval request. Port of
/// `HostedMCPApprovalRequestIdentity`: every field is `None` unless it is a non-empty string.
struct HostedRequest<'a> {
    request_id: Option<&'a str>,
    server_label: Option<&'a str>,
    tool_name: Option<&'a str>,
}

impl HostedRequest<'_> {
    fn identity(&self) -> Option<(&str, &str)> {
        self.server_label.zip(self.tool_name)
    }
}

fn non_empty(s: &str) -> Option<&str> {
    (!s.is_empty()).then_some(s)
}

/// The hosted MCP request an approval item stands for, if it is one. The tool name falls back
/// to the item's own `tool_name` the way Python reads persisted legacy items.
fn hosted_request(item: &ToolApprovalItem) -> Option<HostedRequest<'_>> {
    let raw = &item.raw_item;
    if raw.extra.get("type").and_then(Value::as_str) != Some("mcp_approval_request") {
        return None;
    }
    Some(HostedRequest {
        request_id: non_empty(&raw.call_id),
        server_label: raw
            .extra
            .get("server_label")
            .and_then(Value::as_str)
            .and_then(non_empty),
        tool_name: non_empty(&raw.name),
    })
}

/// The bare tool name a legacy (pre-scoping) hosted MCP record was stored under.
fn hosted_legacy_name<'a>(item: &'a ToolApprovalItem, request: &HostedRequest<'a>) -> &'a str {
    request.tool_name.unwrap_or(&item.tool_name)
}

/// Record one decision on a record. Port of the per-record branch of
/// `RunContextWrapper._apply_approval_decision`: a sticky decision replaces the opposite
/// decision wholesale, a per-call decision moves only its call id.
fn apply_decision(
    rec: &mut ApprovalRecord,
    call_id: &str,
    sticky: bool,
    approve: bool,
    message: Option<String>,
) {
    if sticky {
        if approve {
            rec.approved_always = true;
            rec.rejected_always = false;
            rec.rejected_call_ids.clear();
            rec.rejection_messages.clear();
            rec.sticky_rejection_message = None;
        } else {
            rec.approved_always = false;
            rec.approved_call_ids.clear();
            rec.rejected_always = true;
            match &message {
                Some(m) => {
                    rec.rejection_messages
                        .insert(call_id.to_string(), m.clone());
                }
                None => {
                    rec.rejection_messages.remove(call_id);
                }
            }
            rec.sticky_rejection_message = message;
        }
        return;
    }
    if approve {
        rec.rejected_call_ids.remove(call_id);
        if !rec.approved_always {
            rec.approved_call_ids.insert(call_id.to_string());
        }
        rec.rejection_messages.remove(call_id);
    } else {
        rec.approved_call_ids.remove(call_id);
        if !rec.rejected_always {
            rec.rejected_call_ids.insert(call_id.to_string());
        }
        match message {
            Some(m) => {
                rec.rejection_messages.insert(call_id.to_string(), m);
            }
            None => {
                rec.rejection_messages.remove(call_id);
            }
        }
    }
}

/// Only an exact-call decision. Port of `_get_per_call_approval_status_for_record`.
fn per_call_status(rec: &ApprovalRecord, call_id: &str) -> Option<bool> {
    if rec.approved_call_ids.contains(call_id) {
        Some(true)
    } else if rec.rejected_call_ids.contains(call_id) {
        Some(false)
    } else {
        None
    }
}

/// The model-visible message of a decided record. Port of `_get_rejection_message_for_key`.
fn record_rejection_message(rec: &ApprovalRecord, call_id: &str) -> Option<String> {
    if rec.approved_call_ids.contains(call_id) {
        return None;
    }
    if rec.rejected_always {
        return rec
            .rejection_messages
            .get(call_id)
            .or(rec.sticky_rejection_message.as_ref())
            .cloned();
    }
    if rec.rejected_call_ids.contains(call_id) {
        return rec.rejection_messages.get(call_id).cloned();
    }
    None
}

impl ApprovalsState {
    /// Approve a call; `always` approves future calls to the same tool by the same agent.
    ///
    /// An item that cannot carry the decision (a hosted MCP request without a request id, or a
    /// sticky decision without a server label and tool name) records nothing; use
    /// [`try_approve`](Self::try_approve) to get the reason.
    pub fn approve(&mut self, item: &ToolApprovalItem, always: bool) {
        let _ = self.try_approve(item, always);
    }

    /// Reject a call; `always` rejects future calls to the same tool by the same agent.
    /// Records nothing for an item that cannot carry the decision; see
    /// [`try_reject`](Self::try_reject).
    pub fn reject(&mut self, item: &ToolApprovalItem, always: bool, message: Option<String>) {
        let _ = self.try_reject(item, always, message);
    }

    /// [`approve`](Self::approve), failing the way Python's `approve_tool` raises.
    pub fn try_approve(&mut self, item: &ToolApprovalItem, always: bool) -> crate::Result<()> {
        self.decide(item, always, true, None)
    }

    /// [`reject`](Self::reject), failing the way Python's `reject_tool` raises.
    pub fn try_reject(
        &mut self,
        item: &ToolApprovalItem,
        always: bool,
        message: Option<String>,
    ) -> crate::Result<()> {
        self.decide(item, always, false, message)
    }

    fn decide(
        &mut self,
        item: &ToolApprovalItem,
        always: bool,
        approve: bool,
        message: Option<String>,
    ) -> crate::Result<()> {
        let Some(request) = hosted_request(item) else {
            let key = approval_tool_key(&item.tool_name, &item.raw_item);
            // A per-call decision about a deferred top-level tool also answers a query by its
            // bare name; a permanent one stays under the deferred key.
            if !always && let Some(bare) = key.strip_prefix(DEFERRED_TOP_LEVEL_PREFIX) {
                let rec = self.0.entry(approval_key(&item.agent, bare)).or_default();
                apply_decision(rec, item.call_id(), false, approve, message.clone());
            }
            let rec = self.0.entry(approval_key(&item.agent, &key)).or_default();
            apply_decision(rec, item.call_id(), always, approve, message);
            return Ok(());
        };
        // Validate everything before touching the ledger, so a refused decision leaves it as
        // it was. Port of the hosted branch of `_apply_approval_decision`.
        let request_id = request.request_id.ok_or_else(|| {
            crate::AgentsError::user(
                "Hosted MCP approval decisions require a non-empty request id.",
            )
        })?;
        let identity = request.identity();
        if always && identity.is_none() {
            return Err(crate::AgentsError::user(
                "Persistent hosted MCP approval decisions require a non-empty server_label \
                 and tool name.",
            ));
        }
        let Some((label, tool)) = identity else {
            return Err(crate::AgentsError::model_behavior(
                "Approval decisions require a canonical invocation identity.",
            ));
        };
        let rec = self
            .0
            .entry(hosted_server_tool_key(label, tool))
            .or_default();
        apply_decision(rec, request_id, always, approve, message.clone());
        // Exact name-based lookup without adding an authorization source.
        let query = self
            .0
            .entry(hosted_query_key(tool, request_id))
            .or_default();
        apply_decision(query, request_id, false, approve, message);
        Ok(())
    }

    /// Python's `_get_approval_status_for_record`: an exact call decision wins over the sticky
    /// one, and approval wins when both sticky decisions are set.
    fn record_status(rec: &ApprovalRecord, call_id: &str) -> Option<bool> {
        if let Some(status) = per_call_status(rec, call_id) {
            return Some(status);
        }
        if rec.approved_always {
            return Some(true);
        }
        if rec.rejected_always {
            return Some(false);
        }
        None
    }

    /// The first record, in lookup order, that decides `call_id`.
    fn deciding_record(
        &self,
        agent: &str,
        tool_name: &str,
        call_id: &str,
    ) -> Option<&ApprovalRecord> {
        let decided = |owner: &str| {
            lookup_names(tool_name).find_map(|name| {
                self.0
                    .get(&approval_key(owner, name))
                    .filter(|rec| Self::record_status(rec, call_id).is_some())
            })
        };
        decided(agent).or_else(|| decided(""))
    }

    /// The decision for a function tool call. Port of the function branch of
    /// `_approval_records_for_tool`: an ownerless (pre-1.18) record contributes only its exact
    /// call decisions, never a sticky grant, so an old permanent decision never picks an owner.
    pub(crate) fn function_tool_status(
        &self,
        agent: &str,
        tool_name: &str,
        call_id: &str,
    ) -> ApprovalStatus {
        let owned = lookup_names(tool_name).find_map(|name| {
            self.0
                .get(&approval_key(agent, name))
                .filter(|rec| Self::record_status(rec, call_id).is_some())
        });
        if let Some(rec) = owned {
            return ApprovalStatus {
                approved: Self::record_status(rec, call_id),
                rejection_message: record_rejection_message(rec, call_id),
            };
        }
        // A per-call decision recorded about a deferred top-level tool before it had its own key
        // sits under the bare name; it still answers for its own call, never as a sticky grant.
        let bare_per_call = tool_name
            .strip_prefix(DEFERRED_TOP_LEVEL_PREFIX)
            .and_then(|bare| self.0.get(&approval_key(agent, bare)))
            .filter(|rec| per_call_status(rec, call_id).is_some());
        let legacy = bare_per_call.or_else(|| {
            lookup_names(tool_name).find_map(|name| {
                self.0
                    .get(&approval_key("", name))
                    .filter(|rec| per_call_status(rec, call_id).is_some())
            })
        });
        let Some(rec) = legacy else {
            return ApprovalStatus::default();
        };
        let approved = per_call_status(rec, call_id);
        ApprovalStatus {
            approved,
            rejection_message: (approved == Some(false))
                .then(|| rec.rejection_messages.get(call_id).cloned())
                .flatten(),
        }
    }

    /// `Some(true)` approved, `Some(false)` rejected, `None` undecided.
    ///
    /// Decisions recorded without an owning agent (Python's agent-agnostic `approvals`
    /// bucket, and snapshots written before it bound function tools to their agent) apply to
    /// every agent and are consulted when the agent-specific record is undecided. The runner's
    /// function-tool path does not use this fallback for sticky grants; see
    /// `function_tool_status`.
    pub fn status(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<bool> {
        self.deciding_record(agent, tool_name, call_id)
            .and_then(|rec| Self::record_status(rec, call_id))
    }

    /// Custom rejection message for a call: the message of the record that decides it.
    pub fn rejection_message(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<String> {
        self.deciding_record(agent, tool_name, call_id)
            .and_then(|rec| record_rejection_message(rec, call_id))
    }

    /// The exact hosted MCP decision a name-only query sees for `call_id`, if one was made.
    /// The outer `Option` says whether the mirror decides the call at all.
    fn hosted_query(&self, tool_name: &str, call_id: &str) -> Option<(bool, Option<String>)> {
        let rec = self.0.get(&hosted_query_key(tool_name, call_id))?;
        let status = per_call_status(rec, call_id)?;
        Some((status, record_rejection_message(rec, call_id)))
    }

    /// The decision for a hosted MCP request with a full identity. Port of
    /// `_resolve_hosted_mcp_approval_decision`: the `(server_label, tool)` record decides first;
    /// otherwise only an exact-call decision under a legacy name counts, so a legacy sticky
    /// grant never authorizes a server it was not scoped to.
    pub(crate) fn hosted_decision(
        &self,
        agent: &str,
        server_label: &str,
        tool_name: &str,
        request_id: &str,
    ) -> ApprovalStatus {
        let decided = |rec: &ApprovalRecord, exact_only: bool| {
            let status = if exact_only {
                per_call_status(rec, request_id)
            } else {
                Self::record_status(rec, request_id)
            };
            status.map(|approved| ApprovalStatus {
                approved: Some(approved),
                rejection_message: record_rejection_message(rec, request_id),
            })
        };
        if let Some(found) = self
            .0
            .get(&hosted_server_tool_key(server_label, tool_name))
            .and_then(|rec| decided(rec, false))
        {
            return found;
        }
        // Snapshots this crate wrote before hosted identities were typed keyed the decision
        // by the owning agent and `label:tool`.
        let scoped = format!("{server_label}:{tool_name}");
        if let Some(found) = [agent, ""]
            .iter()
            .find_map(|owner| self.0.get(&approval_key(owner, &scoped)))
            .and_then(|rec| decided(rec, false))
        {
            return found;
        }
        self.hosted_legacy_exact(agent, tool_name, request_id)
    }

    /// An exact-call decision recorded under the request id or the bare tool name.
    fn hosted_legacy_exact(
        &self,
        agent: &str,
        tool_name: &str,
        request_id: &str,
    ) -> ApprovalStatus {
        let exact = |rec: &ApprovalRecord| {
            per_call_status(rec, request_id).map(|approved| ApprovalStatus {
                approved: Some(approved),
                rejection_message: record_rejection_message(rec, request_id),
            })
        };
        self.0
            .get(&hosted_request_key(request_id))
            .and_then(exact)
            .or_else(|| {
                [agent, ""]
                    .iter()
                    .find_map(|owner| self.0.get(&approval_key(owner, tool_name)).and_then(exact))
            })
            .unwrap_or_default()
    }

    /// The decision that governs `call_id` when it runs as the call `item` stands for. Port of
    /// `RunContextWrapper.get_approval_status(..., existing_pending=item)` together with
    /// `get_rejection_message`.
    ///
    /// A hosted MCP request without a server label and tool name cannot be bound to a
    /// canonical invocation, so it is always undecided; only its legacy exact rejection
    /// message stays readable.
    pub fn status_for_item(&self, item: &ToolApprovalItem, call_id: &str) -> ApprovalStatus {
        let Some(request) = hosted_request(item) else {
            let key = approval_tool_key(&item.tool_name, &item.raw_item);
            return ApprovalStatus {
                approved: self.status(&item.agent, &key, call_id),
                rejection_message: self.rejection_message(&item.agent, &key, call_id),
            };
        };
        match request.identity() {
            Some((label, tool)) => self.hosted_decision(&item.agent, label, tool, call_id),
            None => ApprovalStatus {
                approved: None,
                rejection_message: self
                    .hosted_legacy_exact(&item.agent, hosted_legacy_name(item, &request), call_id)
                    .rejection_message,
            },
        }
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
    /// The invocations decisions were recorded for, keyed by call id. Mirrors
    /// `RunContextWrapper._tool_invocations`: a decision binds the exact call it was made for.
    invocations: BTreeMap<String, crate::run_state::ToolInvocationRecord>,
}

/// The invocation a function-tool decision binds: the call id, the tool it routes to and its
/// arguments. Port of the `_tool_invocation_status` registration in `_apply_approval_decision`.
/// Other tool kinds are bound by the run that raises them.
pub(crate) fn approval_binding(
    item: &ToolApprovalItem,
) -> Option<(String, crate::run_state::ToolInvocationRecord)> {
    let mut raw = item.raw_item.to_value();
    let map = raw.as_object_mut()?;
    let ty = map
        .entry("type")
        .or_insert_with(|| Value::String("function_call".into()))
        .as_str()?
        .to_string();
    // A native or custom call travels as a `FunctionCall` whose `name`/`arguments` the SDK
    // filled in; its own payload is the rest. Native calls are scoped by their tool's name.
    let tool_name = match ty.as_str() {
        "function_call" => None,
        "custom_tool_call" => {
            map.remove("arguments");
            None
        }
        "shell_call" | "apply_patch_call" | "computer_call" | "local_shell_call" => {
            map.remove("arguments");
            map.remove("name");
            Some(item.tool_name.as_str())
        }
        _ => return None,
    };
    let id = crate::run_internal::tool_invocation::tool_invocation_identity_named(
        &raw, tool_name, None,
    )?;
    Some((
        id.call_id,
        crate::run_state::ToolInvocationRecord {
            invocation_type: id.invocation_type,
            approval_scope: id.approval_scope,
            fingerprint: id.fingerprint,
            executed: false,
            completed: false,
        },
    ))
}

/// Wrapper around the app context with shared usage and approval state.
///
/// Cloning is cheap: all clones share the same context, usage and approvals.
pub struct RunContext<C> {
    context: Arc<C>,
    inner: Arc<Mutex<Inner>>,
    tool_input: Option<Arc<Value>>,
    /// Call ids whose tool started running but whose output was never committed. Shared by
    /// clones (every run that reuses this context), fresh for a nested agent-tool context.
    /// Mirrors the `executed`/`completed` flags of `RunContextWrapper._tool_invocations`.
    uncommitted: Arc<Mutex<BTreeSet<String>>>,
    /// Computers lazily created by computer tools for this run, keyed by tool id. Shared by
    /// clones like `uncommitted`, fresh for a nested agent-tool context.
    computers: Arc<Mutex<BTreeMap<u64, crate::tool::ResolvedComputer>>>,
}

impl<C> Clone for RunContext<C> {
    fn clone(&self) -> Self {
        RunContext {
            context: self.context.clone(),
            inner: self.inner.clone(),
            tool_input: self.tool_input.clone(),
            uncommitted: self.uncommitted.clone(),
            computers: self.computers.clone(),
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
            uncommitted: Arc::default(),
            computers: Arc::default(),
        }
    }

    /// Wrap an already shared app context.
    pub fn from_arc(context: Arc<C>) -> Self {
        RunContext {
            context,
            inner: Arc::new(Mutex::new(Inner::default())),
            tool_input: None,
            uncommitted: Arc::default(),
            computers: Arc::default(),
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

    /// The computer this run already created for the computer tool `tool_id`, if any.
    pub(crate) fn run_computer(&self, tool_id: u64) -> Option<Arc<dyn crate::computer::Computer>> {
        self.computers
            .lock()
            .expect("run context poisoned")
            .get(&tool_id)
            .map(|c| c.computer.clone())
    }

    /// Keep a computer created for this run. If another resolve stored one first, that one is
    /// returned and `created` is dropped.
    pub(crate) fn store_run_computer(
        &self,
        tool_id: u64,
        created: crate::tool::ResolvedComputer,
    ) -> Arc<dyn crate::computer::Computer> {
        self.computers
            .lock()
            .expect("run context poisoned")
            .entry(tool_id)
            .or_insert(created)
            .computer
            .clone()
    }

    /// Remove every computer this run created, so the caller can dispose of them.
    pub(crate) fn take_run_computers(&self) -> Vec<crate::tool::ResolvedComputer> {
        let mut computers = self.computers.lock().expect("run context poisoned");
        std::mem::take(&mut *computers).into_values().collect()
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
            uncommitted: self.uncommitted.clone(),
            computers: self.computers.clone(),
        }
    }

    /// A clone that carries no nested tool input.
    pub(crate) fn without_tool_input(&self) -> Self {
        RunContext {
            tool_input: None,
            ..self.clone()
        }
    }

    /// A clone that shares everything but the invocation lifecycle, for a nested run.
    pub(crate) fn with_fresh_invocations(&self) -> Self {
        RunContext {
            uncommitted: Arc::default(),
            computers: Arc::default(),
            ..self.clone()
        }
    }

    /// Record that the tool behind `call_id` is about to run its body.
    pub(crate) fn mark_invocation_executed(&self, call_id: &str) {
        self.uncommitted
            .lock()
            .expect("run context poisoned")
            .insert(call_id.to_string());
    }

    /// Record that the output of `call_id` was committed to the run.
    pub(crate) fn mark_invocation_completed(&self, call_id: &str) {
        self.uncommitted
            .lock()
            .expect("run context poisoned")
            .remove(call_id);
    }

    /// Record that the output of a decided call was committed, so a context reused for a later
    /// run does not run it again.
    pub(crate) fn mark_bound_invocation_completed(&self, call_id: &str) {
        if let Some(record) = self.lock().invocations.get_mut(call_id) {
            record.executed = true;
            record.completed = true;
        }
    }

    /// Whether `call_id` already ran in a run that failed before committing its output.
    pub(crate) fn invocation_executed_uncommitted(&self, call_id: &str) -> bool {
        self.uncommitted
            .lock()
            .expect("run context poisoned")
            .contains(call_id)
    }

    /// Approve a pending tool call. With `always_approve`, future calls to the same tool by the
    /// same agent are approved automatically.
    pub fn approve_tool(&self, item: &ToolApprovalItem, always_approve: bool) {
        let _ = self.try_approve_tool(item, always_approve);
    }

    /// Reject a pending tool call, optionally with a custom model-visible message.
    pub fn reject_tool(
        &self,
        item: &ToolApprovalItem,
        always_reject: bool,
        rejection_message: Option<String>,
    ) {
        let _ = self.try_reject_tool(item, always_reject, rejection_message);
    }

    /// [`approve_tool`](Self::approve_tool), returning why a decision could not be recorded:
    /// a hosted MCP request needs a request id, a sticky hosted decision needs a server label
    /// and tool name, and a per-call one needs a complete invocation identity. Nothing is
    /// recorded on error. Mirrors the exceptions `RunContextWrapper.approve_tool` raises.
    pub fn try_approve_tool(
        &self,
        item: &ToolApprovalItem,
        always_approve: bool,
    ) -> crate::Result<()> {
        let mut inner = self.lock();
        inner.approvals.try_approve(item, always_approve)?;
        Self::bind(&mut inner, item);
        Ok(())
    }

    fn bind(inner: &mut Inner, item: &ToolApprovalItem) {
        if let Some((call_id, record)) = approval_binding(item) {
            inner.invocations.entry(call_id).or_insert(record);
        }
    }

    /// The invocations this context's decisions are bound to, keyed by call id. Carry them
    /// with [`export_approvals`](Self::export_approvals) when moving a context elsewhere, as
    /// Python's context converters carry `_tool_invocations`: a later run then refuses a call
    /// that reuses a decided call id for different arguments, and does not repeat a call that
    /// already completed.
    pub fn export_tool_invocations(
        &self,
    ) -> BTreeMap<String, crate::run_state::ToolInvocationRecord> {
        self.lock().invocations.clone()
    }

    /// The invocation a decision about `call_id` was bound to, if any.
    pub(crate) fn bound_invocation(
        &self,
        call_id: &str,
    ) -> Option<crate::run_state::ToolInvocationRecord> {
        self.lock().invocations.get(call_id).cloned()
    }

    /// Restore invocations exported with [`export_tool_invocations`](Self::export_tool_invocations).
    /// Entries already known keep their current state.
    pub fn import_tool_invocations(
        &self,
        invocations: &BTreeMap<String, crate::run_state::ToolInvocationRecord>,
    ) {
        let mut inner = self.lock();
        for (call_id, record) in invocations {
            inner
                .invocations
                .entry(call_id.clone())
                .or_insert_with(|| record.clone());
        }
    }

    /// [`reject_tool`](Self::reject_tool), returning why a decision could not be recorded.
    pub fn try_reject_tool(
        &self,
        item: &ToolApprovalItem,
        always_reject: bool,
        rejection_message: Option<String>,
    ) -> crate::Result<()> {
        let mut inner = self.lock();
        inner
            .approvals
            .try_reject(item, always_reject, rejection_message)?;
        Self::bind(&mut inner, item);
        Ok(())
    }

    /// Approval status for a tool call: `Some(true)` approved, `Some(false)` rejected, `None`
    /// undecided.
    ///
    /// An exact hosted MCP decision about a tool of this name and this call id answers first,
    /// as Python's name-only query does; it is lookup-only and never extends to other calls.
    pub fn is_tool_approved(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<bool> {
        let inner = self.lock();
        if let Some((approved, _)) = inner.approvals.hosted_query(tool_name, call_id) {
            return Some(approved);
        }
        inner.approvals.status(agent, tool_name, call_id)
    }

    /// The decision for a function tool call, without an old snapshot's ownerless sticky
    /// grants; see `ApprovalsState::function_tool_status`.
    pub(crate) fn function_tool_status(
        &self,
        agent: &str,
        tool_name: &str,
        call_id: &str,
    ) -> ApprovalStatus {
        self.lock()
            .approvals
            .function_tool_status(agent, tool_name, call_id)
    }

    /// Custom rejection message for a rejected call, if one was provided.
    pub fn rejection_message(&self, agent: &str, tool_name: &str, call_id: &str) -> Option<String> {
        let inner = self.lock();
        if let Some((_, message)) = inner.approvals.hosted_query(tool_name, call_id) {
            return message;
        }
        inner.approvals.rejection_message(agent, tool_name, call_id)
    }

    /// The decision governing `call_id` when it runs as the call `item` stands for: the same
    /// tool, agent and (for hosted MCP) server. Mirrors
    /// `RunContextWrapper.get_approval_status(tool_name, call_id, existing_pending=item)`
    /// plus `get_rejection_message` with the same arguments.
    pub fn approval_status_for_item(
        &self,
        item: &ToolApprovalItem,
        call_id: &str,
    ) -> ApprovalStatus {
        self.lock().approvals.status_for_item(item, call_id)
    }

    /// The decision for a hosted MCP request, scoped by its server label and tool name.
    pub(crate) fn hosted_mcp_decision(
        &self,
        agent: &str,
        server_label: &str,
        tool_name: &str,
        request_id: &str,
    ) -> ApprovalStatus {
        self.lock()
            .approvals
            .hosted_decision(agent, server_label, tool_name, request_id)
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
        ApprovalStatus {
            approved: self.is_tool_approved(agent, tool_name, call_id),
            rejection_message: self.rejection_message(agent, tool_name, call_id),
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
