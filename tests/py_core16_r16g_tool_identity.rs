//! Ports of the observable cases of `tests/test_tool_identity.py`: how a tool call and a pending
//! approval are named, and the refusal of the synthetic `name.name` namespace.

use openai_agents_sdk::items::{FunctionCall, ToolApprovalItem};
use openai_agents_sdk::tool::{FunctionTool, tool_namespace};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn approval(name: &str, namespace: Option<&str>) -> ToolApprovalItem {
    let mut call = FunctionCall::new(name, "{}");
    if let Some(namespace) = namespace {
        call.extra.insert("namespace".into(), json!(namespace));
    }
    ToolApprovalItem {
        agent: "agent".into(),
        raw_item: call,
        tool_name: name.into(),
        tool_origin: None,
    }
}

/// Python: `TestToolQualifiedName`, `TestToolTraceName` and the `get_tool_call_*` extractors.
#[test]
fn a_call_is_named_by_its_real_namespace_only() {
    assert_eq!(approval("search", None).qualified_name(), "search");
    assert_eq!(
        approval("search", Some("tools")).qualified_name(),
        "tools.search"
    );
    // An empty namespace is ignored.
    assert_eq!(approval("search", Some("")).qualified_name(), "search");
    assert_eq!(approval("search", Some("")).tool_namespace(), None);
    assert_eq!(
        approval("search", Some("tools")).tool_namespace(),
        Some("tools")
    );
    // The synthetic namespace of a deferred top-level tool collapses to the bare name.
    assert_eq!(
        approval("search", Some("search")).qualified_name(),
        "search"
    );
}

/// Python: `test_validate_function_tool_namespace_shape_rejects_synthetic`.
#[test]
fn the_synthetic_namespace_shape_is_refused() {
    let tool = |name: &str| {
        FunctionTool::<()>::new(name, "a tool", |_ctx, _a: NoArgs| async move {
            Ok("ok".to_string())
        })
    };

    assert!(tool_namespace("tools", "A group", vec![tool("search")]).is_ok());
    let error = tool_namespace("search", "A group", vec![tool("search")]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("reserves the synthetic namespace"),
        "{error}"
    );
}
