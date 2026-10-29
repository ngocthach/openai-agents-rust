//! Agent graph visualization in Graphviz DOT format. Mirrors
//! `agents.extensions.visualization`.

use std::collections::HashSet;
use std::sync::Arc;

use crate::agent::Agent;

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn nodes<C>(agent: &Agent<C>, is_root: bool, visited: &mut HashSet<usize>, out: &mut String) {
    let ptr = agent as *const Agent<C> as usize;
    if !visited.insert(ptr) {
        return;
    }
    if is_root {
        out.push_str("\"__start__\" [label=\"__start__\", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];");
        out.push_str("\"__end__\" [label=\"__end__\", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];");
        out.push_str(&format!(
            "\"{}\" [label=\"{}\", shape=box, style=filled, fillcolor=lightyellow, width=1.5, height=0.8];",
            escape(&agent.name),
            escape(&agent.name)
        ));
    }
    for t in &agent.tools {
        let n = escape(&t.name());
        out.push_str(&format!(
            "\"{n}\" [label=\"{n}\", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];"
        ));
    }
    #[cfg(feature = "mcp")]
    for s in &agent.mcp_servers {
        let n = escape(s.name());
        out.push_str(&format!(
            "\"{n}\" [label=\"{n}\", shape=box, style=filled, fillcolor=lightgrey, width=1, height=0.5];"
        ));
    }
    for h in &agent.handoffs {
        let n = escape(&h.agent_name);
        out.push_str(&format!(
            "\"{n}\" [label=\"{n}\", shape=box, style=\"filled,rounded\", fillcolor=lightyellow, width=1.5, height=0.8];"
        ));
        if let Some(target) = h.agent.as_ref() {
            nodes(target.as_ref(), false, visited, out);
        }
    }
}

fn edges<C>(agent: &Agent<C>, is_root: bool, visited: &mut HashSet<usize>, out: &mut String) {
    let ptr = agent as *const Agent<C> as usize;
    if !visited.insert(ptr) {
        return;
    }
    let name = escape(&agent.name);
    if is_root {
        out.push_str(&format!("\"__start__\" -> \"{name}\";"));
    }
    for t in &agent.tools {
        let tn = escape(&t.name());
        out.push_str(&format!("\"{name}\" -> \"{tn}\" [style=dotted, penwidth=1.5];"));
        out.push_str(&format!("\"{tn}\" -> \"{name}\" [style=dotted, penwidth=1.5];"));
    }
    #[cfg(feature = "mcp")]
    for s in &agent.mcp_servers {
        let sn = escape(s.name());
        out.push_str(&format!("\"{name}\" -> \"{sn}\" [style=dashed, penwidth=1.5];"));
        out.push_str(&format!("\"{sn}\" -> \"{name}\" [style=dashed, penwidth=1.5];"));
    }
    for h in &agent.handoffs {
        out.push_str(&format!("\"{name}\" -> \"{}\";", escape(&h.agent_name)));
        if let Some(target) = h.agent.as_ref() {
            edges(target.as_ref(), false, visited, out);
        }
    }
    if agent.handoffs.is_empty() {
        out.push_str(&format!("\"{name}\" -> \"__end__\";"));
    }
}

/// The DOT source describing `agent`, its tools and its handoff graph.
pub fn get_main_graph<C>(agent: &Agent<C>) -> String {
    let mut out = String::from(
        "\n    digraph G {\n        graph [splines=true];\n        node [fontname=\"Arial\"];\n        edge [penwidth=1.5];\n    ",
    );
    nodes(agent, true, &mut HashSet::new(), &mut out);
    edges(agent, true, &mut HashSet::new(), &mut out);
    out.push('}');
    out
}

/// Write the DOT graph to `path` (render it with Graphviz, e.g. `dot -Tpng`).
pub fn draw_graph<C>(agent: &Arc<Agent<C>>, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
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
    }
}
