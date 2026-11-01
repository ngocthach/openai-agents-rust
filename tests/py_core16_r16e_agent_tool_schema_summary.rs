//! Ports of the schema-summary cases of `tests/test_agent_tool_input.py`:
//! `test_build_structured_input_schema_info_generates_summary_for_simple_fields`,
//! `test_schema_summary_returns_none_for_unsupported_shapes` and
//! `test_private_schema_helper_edge_cases`. Python reaches the field and label helpers directly;
//! here the same shapes go through the public summary builder, which is how their output is
//! observed.

use openai_agents_sdk::agent::build_schema_summary;
use serde_json::{Value, json};

#[test]
fn a_summary_describes_simple_fields_in_property_order() {
    let schema = json!({
        "type": "object",
        "description": "Tool arguments.",
        "properties": {
            "mode": {"enum": ["fast", "safe"], "description": "Execution mode."},
            "status": {"const": "ok", "description": "Status marker."},
            "count": {"type": ["integer", "null"], "description": "Optional count."},
            "enabled": {"type": "boolean", "description": "Feature toggle."},
        },
        "required": ["mode", "status"],
    });

    let summary = build_schema_summary(&schema).expect("a summary");

    assert_eq!(
        summary,
        [
            "Description: Tool arguments.",
            "- mode (enum(\"fast\" | \"safe\"), required) - Execution mode.",
            "- status (literal(\"ok\"), required) - Status marker.",
            "- count (integer | null, optional) - Optional count.",
            "- enabled (boolean, optional) - Feature toggle.",
        ]
        .join("\n")
    );
}

#[test]
fn a_summary_is_none_for_unsupported_shapes() {
    assert_eq!(build_schema_summary(&json!({"type": "array"})), None);
    assert_eq!(
        build_schema_summary(&json!({"type": "object", "properties": []})),
        None
    );
    assert_eq!(
        build_schema_summary(&json!({
            "type": "object",
            "properties": {
                "nested": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        })),
        None
    );
}

fn summary_of_field(field: Value) -> Option<String> {
    build_schema_summary(&json!({
        "type": "object",
        "description": "Described.",
        "properties": {"field": field},
    }))
}

#[test]
fn fields_the_summary_cannot_describe_make_it_none() {
    assert_eq!(summary_of_field(json!("not-a-dict")), None);
    assert_eq!(
        summary_of_field(json!({"type": ["integer", "string"]})),
        None
    );
    assert_eq!(summary_of_field(json!({"type": "array"})), None);
    assert_eq!(summary_of_field(json!({})), None);
}

#[test]
fn enum_and_literal_labels_follow_the_python_formats() {
    let label = |field: Value| -> String {
        let summary = summary_of_field(field).expect("a summary");
        let line = summary.lines().nth(1).expect("a field line").to_string();
        let start = line.find('(').expect("a label") + 1;
        let end = line.rfind(", optional)").expect("an optional field");
        line[start..end].to_string()
    };

    assert_eq!(label(json!({"enum": []})), "enum");
    assert_eq!(
        label(json!({"enum": [1, 2, 3, 4, 5, 6]})),
        "enum(1 | 2 | 3 | 4 | 5 | ...)"
    );
    assert_eq!(
        label(json!({"enum": [1, 2, 3, 4, 5]})),
        "enum(1 | 2 | 3 | 4 | 5)"
    );
    assert_eq!(label(json!({"const": null})), "literal(null)");
}
