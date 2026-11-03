//! Parity port of `tests/test_mcp_tool_metadata.py`: the display-metadata helpers of
//! `agents._mcp_tool_metadata`.
//!
//! Python's helpers take `Any` and read a mapping key or an attribute. The Rust equivalents are
//! the `*_value` forms (raw JSON, as hosted MCP delivers tools) and the `MCPTool` forms (local
//! MCP). The Python dict cases use the JSON forms; the attribute-object cases use `MCPTool`.
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;

use openai_agents_sdk::mcp::{
    MCPTool, MCPToolMetadata, collect_mcp_list_tools_metadata, extract_mcp_tool_metadata,
    extract_mcp_tool_metadata_value, resolve_mcp_tool_description,
    resolve_mcp_tool_description_for_model, resolve_mcp_tool_description_for_model_value,
    resolve_mcp_tool_description_value, resolve_mcp_tool_title, resolve_mcp_tool_title_value,
};
use serde_json::{Value, json};

/// The `_ToolObj` stand-in.
fn tool_obj(title: Option<&str>, description: Option<&str>, annotations: Option<Value>) -> MCPTool {
    MCPTool {
        name: String::new(),
        title: title.map(str::to_string),
        description: description.map(str::to_string),
        input_schema: json!({}),
        annotations,
        meta: None,
    }
}

fn meta(description: Option<&str>, title: Option<&str>) -> MCPToolMetadata {
    MCPToolMetadata {
        description: description.map(str::to_string),
        title: title.map(str::to_string),
    }
}

fn key(server: &str, tool: &str) -> (String, String) {
    (server.to_string(), tool.to_string())
}

// ---- resolve_mcp_tool_title ----

#[test]
fn explicit_title_wins() {
    let tool = json!({"title": "Explicit", "annotations": {"title": "Annotated"}});
    assert_eq!(
        resolve_mcp_tool_title_value(&tool).as_deref(),
        Some("Explicit")
    );
}

#[test]
fn falls_back_to_annotations_title() {
    let tool = json!({"annotations": {"title": "Annotated"}});
    assert_eq!(
        resolve_mcp_tool_title_value(&tool).as_deref(),
        Some("Annotated")
    );
}

#[test]
fn title_is_none_when_neither_present() {
    assert_eq!(resolve_mcp_tool_title_value(&json!({})), None);
}

#[test]
fn skips_empty_explicit_title() {
    let tool = json!({"title": "", "annotations": {"title": "Annotated"}});
    assert_eq!(
        resolve_mcp_tool_title_value(&tool).as_deref(),
        Some("Annotated")
    );
}

#[test]
fn skips_non_string_explicit_title() {
    let tool = json!({"title": 123, "annotations": {"title": "Annotated"}});
    assert_eq!(
        resolve_mcp_tool_title_value(&tool).as_deref(),
        Some("Annotated")
    );
}

#[test]
fn title_works_with_attribute_objects() {
    let tool = tool_obj(Some("Explicit"), None, None);
    assert_eq!(resolve_mcp_tool_title(&tool).as_deref(), Some("Explicit"));
}

#[test]
fn title_works_with_attribute_annotations() {
    let tool = tool_obj(None, None, Some(json!({"title": "Annotated"})));
    assert_eq!(resolve_mcp_tool_title(&tool).as_deref(), Some("Annotated"));
}

#[test]
fn handles_missing_annotations_attribute() {
    assert_eq!(resolve_mcp_tool_title(&tool_obj(None, None, None)), None);
}

// ---- resolve_mcp_tool_description ----

#[test]
fn returns_description() {
    assert_eq!(
        resolve_mcp_tool_description_value(&json!({"description": "Long form"})).as_deref(),
        Some("Long form")
    );
}

#[test]
fn description_is_none_when_empty() {
    assert_eq!(
        resolve_mcp_tool_description_value(&json!({"description": ""})),
        None
    );
}

#[test]
fn description_is_none_when_missing() {
    assert_eq!(resolve_mcp_tool_description_value(&json!({})), None);
}

#[test]
fn description_is_none_when_non_string() {
    assert_eq!(
        resolve_mcp_tool_description_value(&json!({"description": 123})),
        None
    );
}

#[test]
fn description_works_with_attribute_object() {
    let tool = tool_obj(None, Some("Long form"), None);
    assert_eq!(
        resolve_mcp_tool_description(&tool).as_deref(),
        Some("Long form")
    );
}

// ---- resolve_mcp_tool_description_for_model ----

#[test]
fn model_description_uses_description_when_present() {
    let tool = json!({"description": "Long form", "title": "Short"});
    assert_eq!(
        resolve_mcp_tool_description_for_model_value(&tool),
        "Long form"
    );
    let obj = tool_obj(Some("Short"), Some("Long form"), None);
    assert_eq!(resolve_mcp_tool_description_for_model(&obj), "Long form");
}

#[test]
fn model_description_falls_back_to_title_when_description_missing() {
    assert_eq!(
        resolve_mcp_tool_description_for_model_value(&json!({"title": "Short"})),
        "Short"
    );
}

#[test]
fn model_description_falls_back_to_annotations_title() {
    let tool = json!({"annotations": {"title": "Annotated"}});
    assert_eq!(
        resolve_mcp_tool_description_for_model_value(&tool),
        "Annotated"
    );
    let obj = tool_obj(None, None, Some(json!({"title": "Annotated"})));
    assert_eq!(resolve_mcp_tool_description_for_model(&obj), "Annotated");
}

#[test]
fn model_description_is_empty_when_nothing_resolvable() {
    assert_eq!(resolve_mcp_tool_description_for_model_value(&json!({})), "");
    assert_eq!(
        resolve_mcp_tool_description_for_model(&tool_obj(None, None, None)),
        ""
    );
}

// ---- extract_mcp_tool_metadata ----

#[test]
fn collects_both_fields() {
    let tool = json!({"description": "Long form", "title": "Short"});
    assert_eq!(
        extract_mcp_tool_metadata_value(&tool),
        meta(Some("Long form"), Some("Short"))
    );
    let obj = tool_obj(Some("Short"), Some("Long form"), None);
    assert_eq!(
        extract_mcp_tool_metadata(&obj),
        meta(Some("Long form"), Some("Short"))
    );
}

#[test]
fn returns_empty_metadata_when_nothing_present() {
    assert_eq!(
        extract_mcp_tool_metadata_value(&json!({})),
        MCPToolMetadata::default()
    );
    assert_eq!(
        extract_mcp_tool_metadata(&tool_obj(None, None, None)),
        MCPToolMetadata::default()
    );
}

// ---- collect_mcp_list_tools_metadata ----

#[test]
fn collects_from_raw_payload() {
    let items = [json!({
        "type": "mcp_list_tools",
        "server_label": "github",
        "tools": [
            {"name": "search", "description": "Search repos", "title": "Search"},
            {"name": "create", "description": "Create issue"}
        ]
    })];
    let expected = BTreeMap::from([
        (
            key("github", "search"),
            meta(Some("Search repos"), Some("Search")),
        ),
        (key("github", "create"), meta(Some("Create issue"), None)),
    ]);
    assert_eq!(collect_mcp_list_tools_metadata(&items), expected);
}

#[test]
fn unwraps_run_item_with_raw_item() {
    let run_item = json!({"raw_item": {
        "type": "mcp_list_tools",
        "server_label": "internal",
        "tools": [{"name": "ping", "description": "Ping"}]
    }});
    let expected = BTreeMap::from([(key("internal", "ping"), meta(Some("Ping"), None))]);
    assert_eq!(collect_mcp_list_tools_metadata(&[run_item]), expected);
}

#[test]
fn skips_items_without_correct_type() {
    let items = [json!({
        "type": "mcp_call", "server_label": "github", "tools": [{"name": "ignored"}]
    })];
    assert!(collect_mcp_list_tools_metadata(&items).is_empty());
}

#[test]
fn skips_items_without_server_label() {
    let items = [json!({"type": "mcp_list_tools", "tools": [{"name": "search"}]})];
    assert!(collect_mcp_list_tools_metadata(&items).is_empty());
}

#[test]
fn skips_items_with_non_list_tools() {
    let items = [json!({
        "type": "mcp_list_tools", "server_label": "github", "tools": "not-a-list"
    })];
    assert!(collect_mcp_list_tools_metadata(&items).is_empty());
}

#[test]
fn skips_tools_without_name() {
    let items = [json!({
        "type": "mcp_list_tools",
        "server_label": "github",
        "tools": [
            {"description": "no name"},
            {"name": "", "description": "empty name"},
            {"name": "good", "description": "kept"}
        ]
    })];
    let expected = BTreeMap::from([(key("github", "good"), meta(Some("kept"), None))]);
    assert_eq!(collect_mcp_list_tools_metadata(&items), expected);
}

#[test]
fn returns_empty_for_empty_input() {
    assert!(collect_mcp_list_tools_metadata(&[]).is_empty());
}

#[test]
fn later_entry_for_same_key_wins() {
    let items = [
        json!({"type": "mcp_list_tools", "server_label": "github",
               "tools": [{"name": "search", "description": "first"}]}),
        json!({"type": "mcp_list_tools", "server_label": "github",
               "tools": [{"name": "search", "description": "second"}]}),
    ];
    let expected = BTreeMap::from([(key("github", "search"), meta(Some("second"), None))]);
    assert_eq!(collect_mcp_list_tools_metadata(&items), expected);
}
