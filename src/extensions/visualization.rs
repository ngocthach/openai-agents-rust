//! Agent graph visualization in Graphviz DOT format. Mirrors
//! `agents.extensions.visualization`.
//!
//! Node identifiers are derived from labels when a label is unique, and generated otherwise, so
//! two tools (or an agent and a tool) sharing a name stay distinct nodes and a node literally
//! named `__start__` cannot collide with the graph terminals.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::agent::Agent;

/// Escape a name for use inside a Graphviz double-quoted ID or label.
fn escape_label(name: &str) -> String {
    name.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace("\r\n", "\\n")
        .replace(['\r', '\n'], "\\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NodeKey {
    Agent(usize),
    /// `(owning agent, index in `Agent::tools`)`.
    Tool(usize, usize),
    /// `(owning agent, index in `Agent::mcp_servers`)`.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    Mcp(usize, usize),
    /// A handoff whose target agent is not materialized: `(owning agent, index)`.
    Handoff(usize, usize),
}

impl NodeKey {
    fn kind(self) -> &'static str {
        match self {
            NodeKey::Agent(_) => "agent",
            NodeKey::Tool(..) => "tool",
            NodeKey::Mcp(..) => "mcp",
            NodeKey::Handoff(..) => "handoff",
        }
    }
}

fn agent_ptr<C>(agent: &Agent<C>) -> usize {
    agent as *const Agent<C> as usize
}

/// Stable DOT identifiers that never conflate two nodes sharing a label.
#[derive(Debug, Default)]
struct GraphNodeIds {
    ids: HashMap<NodeKey, String>,
}

impl GraphNodeIds {
    fn new<C>(agent: &Agent<C>, initially_visited: &HashSet<String>) -> Self {
        let mut nodes: Vec<(NodeKey, String)> = Vec::new();
        let mut previsited: Vec<(NodeKey, String)> = Vec::new();
        let mut seen_keys: HashSet<NodeKey> = HashSet::new();
        let mut visited: HashSet<usize> = HashSet::new();
        collect_nodes(
            agent,
            initially_visited,
            &mut visited,
            &mut seen_keys,
            &mut nodes,
            &mut previsited,
        );

        let escaped: Vec<(NodeKey, String, String)> = nodes
            .into_iter()
            .map(|(key, label)| {
                let e = escape_label(&label);
                (key, label, e)
            })
            .collect();
        let escaped_previsited: Vec<(NodeKey, String, String)> = previsited
            .into_iter()
            .map(|(key, label)| {
                let e = escape_label(&label);
                (key, label, e)
            })
            .collect();
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for (_, _, e) in &escaped {
            *label_counts.entry(e.as_str()).or_default() += 1;
        }
        let raw_labels: HashSet<&str> = escaped_previsited
            .iter()
            .chain(escaped.iter())
            .map(|(_, _, e)| e.as_str())
            .collect();
        let mut used_ids: HashSet<String> = ["__start__".to_string(), "__end__".to_string()].into();
        let mut generated: usize = 0;
        let mut ids = HashMap::new();
        let mut generate_id = |key: NodeKey, used_ids: &HashSet<String>| loop {
            let candidate = format!("__agents_graph_{}_{generated}__", key.kind());
            generated += 1;
            if !raw_labels.contains(candidate.as_str()) && !used_ids.contains(candidate.as_str()) {
                break candidate;
            }
        };
        // An agent the caller already drew keeps its label as id and reserves it.
        for (key, label, escaped_label) in &escaped_previsited {
            let node_id = if used_ids.contains(escaped_label.as_str()) {
                generate_id(*key, &used_ids)
            } else {
                label.clone()
            };
            used_ids.insert(escape_label(&node_id));
            ids.insert(*key, node_id);
        }
        for (key, label, escaped_label) in &escaped {
            let node_id = if label_counts[escaped_label.as_str()] == 1
                && !used_ids.contains(escaped_label.as_str())
            {
                label.clone()
            } else {
                generate_id(*key, &used_ids)
            };
            used_ids.insert(escape_label(&node_id));
            ids.insert(*key, node_id);
        }
        GraphNodeIds { ids }
    }

    fn id(&self, key: NodeKey) -> String {
        escape_label(self.ids.get(&key).map(String::as_str).unwrap_or_default())
    }
}

fn collect_nodes<C>(
    agent: &Agent<C>,
    initially_visited: &HashSet<String>,
    visited: &mut HashSet<usize>,
    seen_keys: &mut HashSet<NodeKey>,
    out: &mut Vec<(NodeKey, String)>,
    previsited: &mut Vec<(NodeKey, String)>,
) {
    let ptr = agent_ptr(agent);
    if !visited.insert(ptr) {
        return;
    }
    if initially_visited.contains(&agent.name) {
        previsited.push((NodeKey::Agent(ptr), agent.name.clone()));
        return;
    }
    fn add(
        seen_keys: &mut HashSet<NodeKey>,
        out: &mut Vec<(NodeKey, String)>,
        key: NodeKey,
        label: String,
    ) {
        if seen_keys.insert(key) {
            out.push((key, label));
        }
    }
    add(seen_keys, out, NodeKey::Agent(ptr), agent.name.clone());
    for (i, tool) in agent.tools.iter().enumerate() {
        add(seen_keys, out, NodeKey::Tool(ptr, i), tool.name());
    }
    #[cfg(feature = "mcp")]
    for (i, server) in agent.mcp_servers.iter().enumerate() {
        add(
            seen_keys,
            out,
            NodeKey::Mcp(ptr, i),
            server.name().to_string(),
        );
    }
    for (i, handoff) in agent.handoffs.iter().enumerate() {
        match handoff.agent.as_ref() {
            Some(target) => collect_nodes(
                target.as_ref(),
                initially_visited,
                visited,
                seen_keys,
                out,
                previsited,
            ),
            None => add(
                seen_keys,
                out,
                NodeKey::Handoff(ptr, i),
                handoff.agent_name.clone(),
            ),
        }
    }
}

const AGENT_NODE: &str = "shape=box, style=filled, fillcolor=lightyellow, width=1.5, height=0.8";
const HANDOFF_NODE: &str =
    "shape=box, style=\"filled,rounded\", fillcolor=lightyellow, width=1.5, height=0.8";

/// Traversal state shared by the node and edge renderers.
struct Walk<'a> {
    ids: &'a GraphNodeIds,
    /// Names the caller had already drawn before the call; they are skipped entirely.
    initially_visited: &'a HashSet<String>,
    /// Names reached by this traversal, reported back to the caller.
    visited_names: &'a mut HashSet<String>,
    visited: HashSet<usize>,
}

impl Walk<'_> {
    /// Marks `agent` as drawn; false when it is already drawn or was drawn by the caller.
    fn enter<C>(&mut self, agent: &Agent<C>) -> bool {
        if self.visited.contains(&agent_ptr(agent)) || self.initially_visited.contains(&agent.name)
        {
            return false;
        }
        self.visited.insert(agent_ptr(agent));
        self.visited_names.insert(agent.name.clone());
        true
    }
}

fn nodes<C>(agent: &Agent<C>, is_root: bool, walk: &mut Walk<'_>, out: &mut String) {
    if !walk.enter(agent) {
        return;
    }
    let ptr = agent_ptr(agent);
    let ids = walk.ids;
    if is_root {
        out.push_str("\"__start__\" [label=\"__start__\", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];");
        out.push_str("\"__end__\" [label=\"__end__\", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];");
        out.push_str(&format!(
            "\"{}\" [label=\"{}\", {AGENT_NODE}];",
            ids.id(NodeKey::Agent(ptr)),
            escape_label(&agent.name)
        ));
    }
    for (i, tool) in agent.tools.iter().enumerate() {
        out.push_str(&format!(
            "\"{}\" [label=\"{}\", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];",
            ids.id(NodeKey::Tool(ptr, i)),
            escape_label(&tool.name())
        ));
    }
    #[cfg(feature = "mcp")]
    for (i, server) in agent.mcp_servers.iter().enumerate() {
        out.push_str(&format!(
            "\"{}\" [label=\"{}\", shape=box, style=filled, fillcolor=lightgrey, width=1, height=0.5];",
            ids.id(NodeKey::Mcp(ptr, i)),
            escape_label(server.name())
        ));
    }
    for (i, handoff) in agent.handoffs.iter().enumerate() {
        match handoff.agent.as_ref() {
            Some(target) => {
                // A target already drawn keeps its single node, as in the Python SDK.
                if !walk.visited.contains(&agent_ptr(target.as_ref()))
                    && !walk.initially_visited.contains(&target.name)
                {
                    out.push_str(&format!(
                        "\"{}\" [label=\"{}\", {HANDOFF_NODE}];",
                        ids.id(NodeKey::Agent(agent_ptr(target.as_ref()))),
                        escape_label(&target.name)
                    ));
                }
                nodes(target.as_ref(), false, walk, out);
            }
            None => out.push_str(&format!(
                "\"{}\" [label=\"{}\", {HANDOFF_NODE}];",
                ids.id(NodeKey::Handoff(ptr, i)),
                escape_label(&handoff.agent_name)
            )),
        }
    }
}

fn edges<C>(agent: &Agent<C>, is_root: bool, walk: &mut Walk<'_>, out: &mut String) {
    if !walk.enter(agent) {
        return;
    }
    let ptr = agent_ptr(agent);
    let ids = walk.ids;
    let name = ids.id(NodeKey::Agent(ptr));
    if is_root {
        out.push_str(&format!("\"__start__\" -> \"{name}\";"));
    }
    for (i, _) in agent.tools.iter().enumerate() {
        let tn = ids.id(NodeKey::Tool(ptr, i));
        out.push_str(&format!(
            "\"{name}\" -> \"{tn}\" [style=dotted, penwidth=1.5];"
        ));
        out.push_str(&format!(
            "\"{tn}\" -> \"{name}\" [style=dotted, penwidth=1.5];"
        ));
    }
    #[cfg(feature = "mcp")]
    for (i, _) in agent.mcp_servers.iter().enumerate() {
        let sn = ids.id(NodeKey::Mcp(ptr, i));
        out.push_str(&format!(
            "\"{name}\" -> \"{sn}\" [style=dashed, penwidth=1.5];"
        ));
        out.push_str(&format!(
            "\"{sn}\" -> \"{name}\" [style=dashed, penwidth=1.5];"
        ));
    }
    for (i, handoff) in agent.handoffs.iter().enumerate() {
        match handoff.agent.as_ref() {
            Some(target) => {
                out.push_str(&format!(
                    "\"{name}\" -> \"{}\";",
                    ids.id(NodeKey::Agent(agent_ptr(target.as_ref())))
                ));
                edges(target.as_ref(), false, walk, out);
            }
            None => out.push_str(&format!(
                "\"{name}\" -> \"{}\";",
                ids.id(NodeKey::Handoff(ptr, i))
            )),
        }
    }
    if agent.handoffs.is_empty() {
        out.push_str(&format!("\"{name}\" -> \"__end__\";"));
    }
}

/// The DOT node declarations for `agent`, its tools, MCP servers and handoff graph.
pub fn get_all_nodes<C>(agent: &Agent<C>) -> String {
    get_all_nodes_with_visited(agent, &mut HashSet::new())
}

/// Like [`get_all_nodes`], with the caller's set of already-drawn agent names. Names in the set
/// on entry are skipped but keep their DOT id reserved; every agent drawn is added to the set.
pub fn get_all_nodes_with_visited<C>(agent: &Agent<C>, visited: &mut HashSet<String>) -> String {
    let initially = visited.clone();
    let ids = GraphNodeIds::new(agent, &initially);
    let mut walk = Walk {
        ids: &ids,
        initially_visited: &initially,
        visited_names: visited,
        visited: HashSet::new(),
    };
    let mut out = String::new();
    nodes(agent, true, &mut walk, &mut out);
    out
}

/// The DOT edge declarations for `agent`, its tools, MCP servers and handoff graph.
pub fn get_all_edges<C>(agent: &Agent<C>) -> String {
    get_all_edges_with_visited(agent, &mut HashSet::new())
}

/// Like [`get_all_edges`], with the caller's set of already-drawn agent names (see
/// [`get_all_nodes_with_visited`]).
pub fn get_all_edges_with_visited<C>(agent: &Agent<C>, visited: &mut HashSet<String>) -> String {
    let initially = visited.clone();
    let ids = GraphNodeIds::new(agent, &initially);
    let mut walk = Walk {
        ids: &ids,
        initially_visited: &initially,
        visited_names: visited,
        visited: HashSet::new(),
    };
    let mut out = String::new();
    edges(agent, true, &mut walk, &mut out);
    out
}

/// The DOT source describing `agent`, its tools and its handoff graph.
pub fn get_main_graph<C>(agent: &Agent<C>) -> String {
    let none = HashSet::new();
    let ids = GraphNodeIds::new(agent, &none);
    let mut out = String::from(
        "\n    digraph G {\n        graph [splines=true];\n        node [fontname=\"Arial\"];\n        edge [penwidth=1.5];\n    ",
    );
    for render in [nodes::<C>, edges::<C>] {
        let mut names = HashSet::new();
        let mut walk = Walk {
            ids: &ids,
            initially_visited: &none,
            visited_names: &mut names,
            visited: HashSet::new(),
        };
        render(agent, true, &mut walk, &mut out);
    }
    out.push('}');
    out
}

/// Write the DOT graph to `path` (render it with Graphviz, e.g. `dot -Tpng`).
pub fn draw_graph<C>(
    agent: &Arc<Agent<C>>,
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    std::fs::write(path, get_main_graph(agent.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoffs::handoff;
    use crate::tool::FunctionTool;

    #[test]
    fn renders_graph() {
        let b = Arc::new(Agent::<()>::new("B"));
        let a = Agent::<()>::new("A")
            .with_tool(FunctionTool::new_no_args("t", "d", |_c| async { Ok("x") }))
            .with_handoff(handoff(b));
        let g = get_main_graph(&a);
        assert!(g.contains("\"__start__\" -> \"A\";"));
        assert!(g.contains("\"A\" -> \"B\";"));
        assert!(g.contains("\"B\" -> \"__end__\";"));
        assert!(g.contains("\"A\" -> \"t\" [style=dotted"));
        assert!(g.contains(&get_all_nodes(&a)));
        assert!(g.contains(&get_all_edges(&a)));
    }

    #[test]
    fn distinguishes_nodes_sharing_a_label() {
        let agent =
            Agent::<()>::new("dup").with_tool(FunctionTool::new_no_args("dup", "d", |_c| async {
                Ok("x")
            }));
        let g = get_main_graph(&agent);
        // The agent and the tool share the label `dup`, so both get generated ids.
        assert!(g.contains("[label=\"dup\""));
        assert!(g.contains("__agents_graph_agent_0__"));
        assert!(g.contains("__agents_graph_tool_1__"));
        assert!(!g.contains("\"dup\" -> \"dup\""));
    }

    #[test]
    fn a_node_named_like_a_terminal_does_not_collide() {
        let agent = Agent::<()>::new("__start__");
        let g = get_main_graph(&agent);
        assert!(g.contains("\"__start__\" [label=\"__start__\", shape=ellipse"));
        assert!(g.contains("__agents_graph_agent_0__"));
    }

    #[test]
    fn escapes_line_breaks() {
        let agent = Agent::<()>::new("two\nlines");
        let g = get_main_graph(&agent);
        assert!(g.contains("label=\"two\\nlines\""));
        assert!(!g.contains("two\nlines"));
    }
}
