//! Stable agent-graph identities. Port of `agents._run_state_agent_identity`.
//!
//! Agents are identified by name in the run loop, but a graph may contain several distinct agents
//! with one name. Serialized state therefore names an agent by an identity key: the bare name for
//! the first agent with that name, `<name>#2`, `<name>#3`, ... for the others. The key does not
//! depend on the order the graph happens to be reachable in: duplicates are ranked by a signature
//! of their configuration, with the starting agent always first.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::agent::Agent;
use crate::agent::{AgentModel, Instructions, ToolUseBehavior};
use crate::items::ToolApprovalItem;
use crate::run_state::ProcessedResponseSnapshot;
use crate::tool::Tool;

/// Agents reachable from `start` through handoffs, breadth first, each exactly once.
pub(crate) fn iter_agent_graph<C>(start: &Arc<Agent<C>>) -> Vec<Arc<Agent<C>>> {
    let mut out = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(agent) = queue.pop_front() {
        if !seen.insert(Arc::as_ptr(&agent) as usize) {
            continue;
        }
        for handoff in &agent.handoffs {
            if let Some(target) = handoff.agent.as_ref() {
                queue.push_back(target.clone());
            }
        }
        out.push(agent);
    }
    out
}

fn tool_signature<C>(tool: &Tool<C>) -> String {
    let namespace = match tool {
        Tool::Function(f) => f.namespace.clone(),
        _ => None,
    };
    json!({"name": tool.name(), "namespace": namespace}).to_string()
}

/// A deterministic description of an agent's configuration, used only to rank same-named agents.
///
/// Keys are listed alphabetically so the serialized text orders like Python's.
fn agent_signature<C>(agent: &Agent<C>) -> String {
    let mut handoffs: Vec<String> = agent
        .handoffs
        .iter()
        .map(|h| json!({"agent_name": h.agent_name, "tool_name": h.tool_name}).to_string())
        .collect();
    handoffs.sort();
    let mut tools: Vec<String> = agent.tools.iter().map(tool_signature).collect();
    tools.sort();
    let mut input_guardrails: Vec<String> = agent
        .input_guardrails
        .iter()
        .map(|g| g.name.clone())
        .collect();
    input_guardrails.sort();
    let mut output_guardrails: Vec<String> = agent
        .output_guardrails
        .iter()
        .map(|g| g.name.clone())
        .collect();
    output_guardrails.sort();
    let instructions = match &agent.instructions {
        None => Value::Null,
        Some(Instructions::Static(s)) => json!(s),
        Some(Instructions::Dynamic(_)) => json!({"callable": "instructions"}),
    };
    let model = match &agent.model {
        None => Value::Null,
        Some(AgentModel::Name(n)) => json!(n),
        Some(AgentModel::Instance(m)) => json!({"type": m.model_name()}),
    };
    let model_settings = serde_json::to_value(&agent.model_settings).unwrap_or(Value::Null);
    let tool_use_behavior = match &agent.tool_use_behavior {
        ToolUseBehavior::RunLlmAgain => json!("run_llm_again"),
        ToolUseBehavior::StopOnFirstTool => json!("stop_on_first_tool"),
        ToolUseBehavior::StopAtTools(s) => json!({"stop_at_tool_names": s.stop_at_tool_names}),
        ToolUseBehavior::Custom(_) => json!({"callable": "tool_use_behavior"}),
    };
    // `serde_json::Map` is ordered by key, so this text is stable.
    json!({
        "agent_type": "agent",
        "handoff_description": agent.handoff_description,
        "handoffs": handoffs,
        "hooks": agent.hooks.as_ref().map(|_| json!({"object": "hooks"})),
        "input_guardrails": input_guardrails,
        "instructions": instructions,
        "model": model,
        "model_settings": model_settings,
        "output_guardrails": output_guardrails,
        "output_type": agent.output_type.as_ref().map(|s| s.name()),
        "reset_tool_choice": agent.reset_tool_choice,
        "tool_use_behavior": tool_use_behavior,
        "tools": tools,
    })
    .to_string()
}

/// Agents sharing one name, with their position in graph order.
type NameGroup<C> = (String, Vec<(usize, Arc<Agent<C>>)>);

/// Identity keys of every agent reachable from `start`, in the order the keys were allocated.
///
/// Mirrors `_build_agent_identity_map`: the first agent of a name (ranked with the root first,
/// then by configuration signature, then by traversal order) keeps the bare name; the rest get
/// `#2`, `#3`, ... skipping any key that collides with another agent's literal name.
pub(crate) fn build_agent_identity_map<C>(start: &Arc<Agent<C>>) -> Vec<(String, Arc<Agent<C>>)> {
    let ordered = iter_agent_graph(start);
    let literal_names: HashSet<&str> = ordered.iter().map(|a| a.name.as_str()).collect();
    let mut by_name: Vec<NameGroup<C>> = Vec::new();
    for (index, agent) in ordered.iter().enumerate() {
        match by_name.iter_mut().find(|(name, _)| *name == agent.name) {
            Some((_, group)) => group.push((index, agent.clone())),
            None => by_name.push((agent.name.clone(), vec![(index, agent.clone())])),
        }
    }

    let mut used: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for (name, mut group) in by_name {
        group.sort_by_cached_key(|(index, agent)| {
            (
                u8::from(!Arc::ptr_eq(agent, start)),
                agent_signature(agent),
                *index,
            )
        });
        let mut iter = group.into_iter();
        let (_, base) = iter.next().expect("a name group is never empty");
        used.insert(name.clone());
        out.push((name.clone(), base));
        let mut next_index = 2usize;
        for (_, duplicate) in iter {
            let mut candidate = format!("{name}#{next_index}");
            while used.contains(&candidate) || literal_names.contains(candidate.as_str()) {
                next_index += 1;
                candidate = format!("{name}#{next_index}");
            }
            used.insert(candidate.clone());
            out.push((candidate, duplicate));
            next_index += 1;
        }
    }
    out
}

/// The identity key of `agent` within the graph rooted at `start`.
pub(crate) fn identity_key<C>(start: &Arc<Agent<C>>, agent: &Arc<Agent<C>>) -> String {
    build_agent_identity_map(start)
        .into_iter()
        .find(|(_, a)| Arc::ptr_eq(a, agent))
        .map(|(key, _)| key)
        .unwrap_or_else(|| agent.name.clone())
}

/// Identity keys by agent pointer.
pub(crate) fn identity_keys_by_ptr<C>(start: &Arc<Agent<C>>) -> HashMap<usize, String> {
    build_agent_identity_map(start)
        .into_iter()
        .map(|(key, agent)| (Arc::as_ptr(&agent) as usize, key))
        .collect()
}

/// The agent registered under `identity`, falling back to the first agent with that name.
pub(crate) fn resolve_identity<C>(start: &Arc<Agent<C>>, identity: &str) -> Option<Arc<Agent<C>>> {
    let map = build_agent_identity_map(start);
    if let Some((_, agent)) = map.iter().find(|(key, _)| key == identity) {
        return Some(agent.clone());
    }
    map.into_iter()
        .find(|(_, agent)| agent.name == identity)
        .map(|(_, agent)| agent)
}

/// Whether same-named agents in the graph have identical configuration signatures while hooks
/// are present. Port of `_has_ambiguous_hook_identities`: the saved owner of an operation then
/// depends on traversal order, and the hook instance state that could tell them apart is not
/// stored.
pub(crate) fn has_ambiguous_hook_identities<C>(start: &Arc<Agent<C>>) -> bool {
    let mut groups: HashMap<(String, bool, String), (usize, bool)> = HashMap::new();
    for agent in iter_agent_graph(start) {
        let key = (
            agent.name.clone(),
            Arc::ptr_eq(&agent, start),
            agent_signature(&agent),
        );
        let entry = groups.entry(key).or_default();
        entry.0 += 1;
        entry.1 |= agent.hooks.is_some();
    }
    groups
        .values()
        .any(|(count, has_hooks)| *count > 1 && *has_hooks)
}

/// Whether a legacy (pre-`1.21`) snapshot could not name its duplicate owners: two non-root
/// agents with hooks share a name and every signature field up to `hooks`, so the ordinal the
/// old format saved depended on hook state it did not record. Port of
/// `_has_legacy_hook_identity_ambiguity`.
pub(crate) fn has_legacy_hook_identity_ambiguity<C>(start: &Arc<Agent<C>>) -> bool {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for agent in iter_agent_graph(start) {
        if Arc::ptr_eq(&agent, start) || agent.hooks.is_none() {
            continue;
        }
        let signature: Value =
            serde_json::from_str(&agent_signature(&agent)).unwrap_or(Value::Null);
        let prefix: serde_json::Map<String, Value> = signature
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(key, _)| key.as_str() <= "hooks")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !seen.insert((agent.name.clone(), Value::Object(prefix).to_string())) {
            return true;
        }
    }
    false
}

/// Whether a parked approval may end the run once it resolves, so staged input has no model call
/// left to join. Port of the `stops_before_next_model` / callable check in Python's
/// `RunState.add_input`. The names that count are the parked calls' tool names plus, when the
/// response carries no handoff, the completed sibling calls' names (bare and qualified).
pub(crate) fn interruption_may_end_run<C>(
    agent: &Agent<C>,
    interruptions: &[ToolApprovalItem],
    processed: Option<&ProcessedResponseSnapshot>,
) -> bool {
    if interruptions.is_empty() {
        return false;
    }
    match &agent.tool_use_behavior {
        ToolUseBehavior::StopOnFirstTool | ToolUseBehavior::Custom(_) => true,
        ToolUseBehavior::RunLlmAgain => false,
        ToolUseBehavior::StopAtTools(stop) => {
            let mut names: HashSet<String> = interruptions
                .iter()
                .map(|item| item.tool_name.clone())
                .collect();
            if let Some(processed) = processed.filter(|p| p.handoffs.is_empty()) {
                for function in &processed.functions {
                    names.insert(function.tool_name.clone());
                    if let Some(ns) = function.namespace.as_deref()
                        && ns != function.tool_name
                    {
                        names.insert(format!("{ns}.{}", function.tool_name));
                    }
                }
            }
            stop.stop_at_tool_names
                .iter()
                .any(|name| names.contains(name))
        }
    }
}
