//! Parity port of `tests/test_visualization.py`: the DOT graph renderers in
//! `agents.extensions.visualization`.
//!
//! Python-only: `graphviz.Source` return type and the `Source.render` monkeypatch (Rust's
//! `draw_graph` writes the DOT source to a path), `Mock(spec=...)` property tripwires, and cycles
//! (agents are immutable `Arc`s in Rust, so a handoff cycle cannot be built; a diamond covers the
//! "each agent is drawn once" rule).
#![cfg(feature = "mcp")]

use std::collections::HashSet;
use std::sync::Arc;

use openai_agents_sdk::Agent;
use openai_agents_sdk::extensions::visualization::{
    draw_graph, get_all_edges, get_all_edges_with_visited, get_all_nodes,
    get_all_nodes_with_visited, get_main_graph,
};
use openai_agents_sdk::handoffs::{Handoff, handoff};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::FakeMcpServer;

fn tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new_no_args(name, "d", |_c| async { Ok("x") })
}

fn stub_handoff(agent_name: &str) -> Handoff<()> {
    Handoff::custom("t", "d", agent_name, json!({}), |_c, _a| async {
        Err(openai_agents_sdk::AgentsError::user("never invoked"))
    })
}

/// `mock_agent` fixture: Agent1 with Tool1, Tool2, a name-only Handoff1 stub and MCPServer1.
fn mock_agent_named(name: &str, tool1: &str, tool2: &str) -> Agent<()> {
    Agent::<()>::new(name)
        .with_tool(tool(tool1))
        .with_tool(tool(tool2))
        .with_handoff(stub_handoff("Handoff1"))
        .with_mcp_server(Arc::new(FakeMcpServer::new("MCPServer1")))
}

fn mock_agent() -> Agent<()> {
    mock_agent_named("Agent1", "Tool1", "Tool2")
}

/// Every DOT id declared with `[label="<label>"`, in order (the regex
/// `"([^"]+)" \[label="<label>"` of the Python tests).
fn ids_with_label(source: &str, label: &str) -> Vec<String> {
    let needle = format!("\" [label=\"{label}\"");
    let mut ids = Vec::new();
    let mut from = 0;
    while let Some(pos) = source[from..].find(&needle) {
        let end = from + pos;
        let start = source[..end].rfind('"').expect("opening quote") + 1;
        ids.push(source[start..end].to_string());
        from = end + needle.len();
    }
    ids
}

fn distinct(ids: &[String]) -> usize {
    ids.iter().collect::<HashSet<_>>().len()
}

fn assert_mcp_nodes(source: &str) {
    assert!(source.contains(
        "\"MCPServer1\" [label=\"MCPServer1\", shape=box, style=filled, fillcolor=lightgrey, width=1, height=0.5];"
    ));
}

fn assert_mcp_edges(source: &str) {
    assert!(source.contains("\"Agent1\" -> \"MCPServer1\" [style=dashed, penwidth=1.5];"));
    assert!(source.contains("\"MCPServer1\" -> \"Agent1\" [style=dashed, penwidth=1.5];"));
}

fn assert_fixture_nodes(source: &str) {
    for expected in [
        "\"__start__\" [label=\"__start__\", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];",
        "\"__end__\" [label=\"__end__\", shape=ellipse, style=filled, fillcolor=lightblue, width=0.5, height=0.3];",
        "\"Agent1\" [label=\"Agent1\", shape=box, style=filled, fillcolor=lightyellow, width=1.5, height=0.8];",
        "\"Tool1\" [label=\"Tool1\", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];",
        "\"Tool2\" [label=\"Tool2\", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];",
        "\"Handoff1\" [label=\"Handoff1\", shape=box, style=\"filled,rounded\", fillcolor=lightyellow, width=1.5, height=0.8];",
    ] {
        assert!(source.contains(expected), "missing {expected}\n{source}");
    }
    assert_mcp_nodes(source);
}

#[test]
fn get_main_graph_renders_every_node_kind() {
    let result = get_main_graph(&mock_agent());
    assert!(result.contains("digraph G"));
    assert!(result.contains("graph [splines=true];"));
    assert!(result.contains("node [fontname=\"Arial\"];"));
    assert!(result.contains("edge [penwidth=1.5];"));
    assert_fixture_nodes(&result);
}

#[test]
fn get_all_nodes_renders_every_node_kind() {
    assert_fixture_nodes(&get_all_nodes(&mock_agent()));
}

#[test]
fn get_all_edges_renders_every_edge_kind() {
    let result = get_all_edges(&mock_agent());
    for expected in [
        "\"__start__\" -> \"Agent1\";",
        "\"Agent1\" -> \"Tool1\" [style=dotted, penwidth=1.5];",
        "\"Tool1\" -> \"Agent1\" [style=dotted, penwidth=1.5];",
        "\"Agent1\" -> \"Tool2\" [style=dotted, penwidth=1.5];",
        "\"Tool2\" -> \"Agent1\" [style=dotted, penwidth=1.5];",
        "\"Agent1\" -> \"Handoff1\";",
    ] {
        assert!(result.contains(expected), "missing {expected}");
    }
    assert_mcp_edges(&result);
}

#[test]
fn draw_graph_writes_the_dot_source() {
    let dir = std::env::temp_dir().join(format!("py_core5_viz_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("agent_graph.dot");
    draw_graph(&Arc::new(mock_agent()), &path).unwrap();
    let source = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert!(source.contains("digraph G"));
    assert!(source.contains("graph [splines=true];"));
    assert_fixture_nodes(&source);
}

#[test]
fn a_shared_handoff_target_is_drawn_once() {
    // Python's `test_cycle_detection` builds A <-> B; Rust agents are immutable, so a diamond
    // (A -> B, A -> C, B -> C) exercises the same "visit each agent once" rule.
    let c = Arc::new(Agent::<()>::new("C"));
    let b = Arc::new(Agent::<()>::new("B").with_handoff(handoff(c.clone())));
    let a = Agent::<()>::new("A")
        .with_handoff(handoff(b))
        .with_handoff(handoff(c));
    let nodes = get_all_nodes(&a);
    let edges = get_all_edges(&a);
    assert_eq!(nodes.matches("\"A\" [label=\"A\"").count(), 1);
    assert_eq!(nodes.matches("\"B\" [label=\"B\"").count(), 1);
    assert_eq!(nodes.matches("\"C\" [label=\"C\"").count(), 1);
    assert!(edges.contains("\"A\" -> \"B\""));
    assert!(edges.contains("\"B\" -> \"C\""));
    assert!(edges.contains("\"A\" -> \"C\""));
    assert_eq!(edges.matches("\"C\" -> \"__end__\"").count(), 1);
}

#[test]
fn graph_keeps_different_node_types_with_the_same_name_distinct() {
    let agent = Agent::<()>::new("shared")
        .with_tool(tool("shared"))
        .with_mcp_server(Arc::new(FakeMcpServer::new("shared")))
        .with_handoff(stub_handoff("shared"));
    let source = get_main_graph(&agent);
    let ids = ids_with_label(&source, "shared");
    assert_eq!(ids.len(), 4);
    assert_eq!(distinct(&ids), 4);
    let (agent_id, tool_id, server_id, handoff_id) = (&ids[0], &ids[1], &ids[2], &ids[3]);
    assert!(source.contains(&format!("\"{agent_id}\" -> \"{tool_id}\" [style=dotted")));
    assert!(source.contains(&format!("\"{agent_id}\" -> \"{server_id}\" [style=dashed")));
    assert!(source.contains(&format!("\"{agent_id}\" -> \"{handoff_id}\";")));
    for id in &ids {
        assert!(!source.contains(&format!("\"{id}\" -> \"{id}\"")));
    }
}

#[test]
fn graph_keeps_names_that_escape_to_the_same_id_distinct() {
    let agent = Agent::<()>::new("shared\n")
        .with_tool(tool("shared\r"))
        .with_handoff(stub_handoff("shared\r\n"));
    let source = get_main_graph(&agent);
    let ids = ids_with_label(&source, "shared\\n");
    assert_eq!(ids.len(), 3);
    assert_eq!(distinct(&ids), 3);
    let (agent_id, tool_id, handoff_id) = (&ids[0], &ids[1], &ids[2]);
    assert!(source.contains(&format!("\"{agent_id}\" -> \"{tool_id}\" [style=dotted")));
    assert!(source.contains(&format!("\"{agent_id}\" -> \"{handoff_id}\";")));
    for id in &ids {
        assert!(!source.contains(&format!("\"{id}\" -> \"{id}\"")));
    }
}

fn child_handoff(child: Agent<()>, use_handoff_object: bool) -> Handoff<()> {
    if use_handoff_object {
        handoff(child)
    } else {
        Handoff::from(child)
    }
}

#[test]
fn graph_traverses_different_agents_with_the_same_name() {
    for use_handoff_object in [false, true] {
        let child = Agent::<()>::new("duplicate").with_tool(tool("child_tool"));
        let parent =
            Agent::<()>::new("duplicate").with_handoff(child_handoff(child, use_handoff_object));
        let source = get_main_graph(&parent);
        let ids = ids_with_label(&source, "duplicate");
        assert_eq!(ids.len(), 2, "{source}");
        assert_eq!(distinct(&ids), 2);
        let (parent_id, child_id) = (&ids[0], &ids[1]);
        assert!(source.contains(&format!("\"{parent_id}\" -> \"{child_id}\";")));
        assert!(source.contains(&format!("\"{child_id}\" -> \"child_tool\" [style=dotted")));
    }
}

#[test]
fn get_all_nodes_honors_prepopulated_visited_names() {
    for use_handoff_object in [false, true] {
        let child = Agent::<()>::new("child");
        let parent =
            Agent::<()>::new("parent").with_handoff(child_handoff(child, use_handoff_object));
        let mut visited: HashSet<String> = ["child".to_string()].into();
        let nodes = get_all_nodes_with_visited(&parent, &mut visited);
        assert!(!nodes.contains("\"child\" [label=\"child\""));
        let expected: HashSet<String> = ["parent".to_string(), "child".to_string()].into();
        assert_eq!(visited, expected);
    }
}

#[test]
fn a_previsited_root_is_not_drawn_or_inspected() {
    let agent = Agent::<()>::new("visited").with_tool(tool("t"));
    let seed = || -> HashSet<String> { ["visited".to_string()].into() };
    assert_eq!(get_all_nodes_with_visited(&agent, &mut seed()), "");
    assert_eq!(get_all_edges_with_visited(&agent, &mut seed()), "");
}

#[test]
fn a_previsited_subgraph_does_not_affect_included_node_ids() {
    let child = Agent::<()>::new("child").with_tool(tool("shared"));
    let parent = Agent::<()>::new("parent")
        .with_tool(tool("shared"))
        .with_handoff(child);
    let seed = || -> HashSet<String> { ["child".to_string()].into() };
    for source in [
        get_all_nodes_with_visited(&parent, &mut seed()),
        get_all_edges_with_visited(&parent, &mut seed()),
    ] {
        assert!(source.contains("\"shared\""));
        assert!(!source.contains("__agents_graph_tool_"));
    }
}

#[test]
fn a_previsited_agent_reserves_its_id() {
    for use_handoff_object in [false, true] {
        let child = Agent::<()>::new("shared");
        let parent = Agent::<()>::new("parent")
            .with_tool(tool("shared"))
            .with_handoff(child_handoff(child, use_handoff_object));
        let seed = || -> HashSet<String> { ["shared".to_string()].into() };
        let nodes = get_all_nodes_with_visited(&parent, &mut seed());
        let edges = get_all_edges_with_visited(&parent, &mut seed());
        let tool_ids = ids_with_label(&nodes, "shared");
        assert_eq!(tool_ids.len(), 1, "{nodes}");
        assert!(tool_ids[0].starts_with("__agents_graph_tool_"));
        assert!(edges.contains(&format!("\"parent\" -> \"{}\" [style=dotted", tool_ids[0])));
        assert!(edges.contains("\"parent\" -> \"shared\";"));
    }
}

#[test]
fn collision_free_escaped_ids_keep_legacy_names() {
    let agent = Agent::<()>::new(r"shared\n").with_tool(tool("shared\n"));
    let source = get_main_graph(&agent);
    assert!(source.contains("\"shared\\\\n\" [label=\"shared\\\\n\""));
    assert!(source.contains("\"shared\\n\" [label=\"shared\\n\""));
    assert!(!source.contains("__agents_graph_"));
}

#[test]
fn graph_reserves_start_and_end_node_ids() {
    let source = get_main_graph(&Agent::<()>::new("__start__"));
    let start_ids = ids_with_label(&source, "__start__");
    assert_eq!(start_ids.len(), 2);
    assert_eq!(distinct(&start_ids), 2);
    assert_eq!(start_ids[0], "__start__");
    assert!(source.contains(&format!("\"__start__\" -> \"{}\";", start_ids[1])));
}

#[test]
fn names_with_quotes_and_backslashes_are_escaped() {
    let agent = mock_agent_named("Weird\"Name", "Back\\slash", "Tool2");
    let nodes = get_all_nodes(&agent);
    let edges = get_all_edges(&agent);
    assert!(nodes.contains("\"Weird\\\"Name\" [label=\"Weird\\\"Name\""));
    assert!(!nodes.contains("\"Weird\"Name\""));
    assert!(nodes.contains("\"Back\\\\slash\""));
    assert!(edges.contains("\"__start__\" -> \"Weird\\\"Name\";"));
}

#[test]
fn names_with_line_breaks_are_escaped() {
    let agent = mock_agent_named("Agent\nName", "CRLF\r\nTool", "CR\rTool");
    let nodes = get_all_nodes(&agent);
    let edges = get_all_edges(&agent);
    assert!(nodes.contains("\"Agent\\nName\" [label=\"Agent\\nName\""));
    assert!(nodes.contains("\"CRLF\\nTool\""));
    assert!(nodes.contains("\"CR\\nTool\""));
    assert!(edges.contains("\"__start__\" -> \"Agent\\nName\";"));
    assert!(!nodes.contains("\"Agent\nName\""));
}

#[test]
fn draw_graph_with_a_real_agent_without_handoffs() {
    let source =
        get_main_graph(&Agent::<()>::new("TestAgent").with_instructions("Test instructions"));
    assert!(source.contains("\"TestAgent\""));
    assert!(source.contains("\"__start__\" -> \"TestAgent\""));
    assert!(source.contains("\"TestAgent\" -> \"__end__\""));
}

#[test]
fn draw_graph_with_a_real_agent_with_handoffs() {
    let child = Agent::<()>::new("ChildAgent").with_instructions("Child instructions");
    let parent = Agent::<()>::new("ParentAgent")
        .with_instructions("Parent instructions")
        .with_handoff(child);
    let source = get_main_graph(&parent);
    assert!(source.contains("\"ParentAgent\""));
    assert!(source.contains("\"ChildAgent\""));
    assert!(source.contains("\"ParentAgent\" -> \"ChildAgent\""));
    assert!(!source.contains("\"ParentAgent\" -> \"__end__\""));
    assert!(source.contains("\"ChildAgent\" -> \"__end__\""));
}

#[test]
fn draw_graph_with_a_real_handoff_object() {
    let child = Agent::<()>::new("ChildAgent")
        .with_instructions("Child instructions")
        .with_tool(tool("ChildTool"));
    let parent = Agent::<()>::new("ParentAgent")
        .with_instructions("Parent instructions")
        .with_handoff(handoff(child));
    let source = get_main_graph(&parent);
    assert!(source.contains("\"ParentAgent\""));
    assert!(source.contains(
        "\"ChildAgent\" [label=\"ChildAgent\", shape=box, style=\"filled,rounded\", fillcolor=lightyellow, width=1.5, height=0.8];"
    ));
    assert!(source.contains(
        "\"ChildTool\" [label=\"ChildTool\", shape=ellipse, style=filled, fillcolor=lightgreen, width=0.5, height=0.3];"
    ));
    assert!(source.contains("\"ParentAgent\" -> \"ChildAgent\";"));
    assert!(!source.contains("\"ParentAgent\" -> \"__end__\""));
    assert!(source.contains("\"ChildAgent\" -> \"__end__\""));
}

#[test]
fn draw_graph_keeps_a_stub_for_a_handoff_without_an_agent_ref() {
    let parent = Agent::<()>::new("ParentAgent")
        .with_instructions("Parent instructions")
        .with_handoff(stub_handoff("ExternalHandoff"));
    let source = get_main_graph(&parent);
    assert!(source.contains(
        "\"ExternalHandoff\" [label=\"ExternalHandoff\", shape=box, style=\"filled,rounded\", fillcolor=lightyellow, width=1.5, height=0.8];"
    ));
    assert!(source.contains("\"ParentAgent\" -> \"ExternalHandoff\";"));
    assert!(!source.contains("\"ExternalHandoff\" -> \"__end__\""));
}
