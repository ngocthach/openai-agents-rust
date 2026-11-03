//! Ports of the construction, schema, namespace, enablement and guardrail tests in
//! `tests/test_function_tool.py`.
//!
//! Tool bodies are always async in Rust, so the Python sync/async pairs collapse to one case.
//! Schema rejections go through [`FunctionTool::try_new_raw`]; the infallible `new_raw` keeps its
//! silent non-strict fallback for MCP-style callers.

use std::sync::Arc;

use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool, Tool, ToolSearchTool, tool_namespace};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct BoolCtx {
    enable_tools: bool,
}

#[derive(Deserialize, JsonSchema)]
struct SimpleArgs {
    a: i64,
    #[serde(default = "five")]
    b: i64,
}

fn five() -> i64 {
    5
}

#[derive(Deserialize, JsonSchema)]
struct Foo {
    a: i64,
    #[serde(default = "five")]
    b: i64,
}

#[derive(Deserialize, JsonSchema)]
struct Bar {
    x: String,
    y: i64,
}

#[derive(Deserialize, JsonSchema)]
struct ComplexArgs {
    foo: Foo,
    bar: Bar,
    #[serde(default = "hello")]
    baz: String,
}

fn hello() -> String {
    "hello".to_string()
}

/// `Field(ge=0, le=10)` analogue: a validated scalar that fails while the arguments are parsed.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(try_from = "i64")]
#[schemars(with = "i64")]
struct Score(i64);

impl TryFrom<i64> for Score {
    type Error = String;
    fn try_from(value: i64) -> std::result::Result<Self, String> {
        if (0..=10).contains(&value) {
            Ok(Score(value))
        } else {
            Err(format!("score {value} is outside 0..=10"))
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ScoreArgs {
    score: Score,
}

/// Invoke `tool` directly with raw JSON arguments.
async fn invoke_direct(tool: &FunctionTool<()>, args: &str) -> Result<String> {
    let ctx = ToolContext::direct(RunContext::new(()), "test", &tool.name, args);
    tool.invoke(ctx, args.to_string())
        .await
        .map(|out| out.to_model_string())
}

fn ok_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new_no_args(name, "desc", |_ctx| async { Ok("ok") })
}

// ---------------------------------------------------------------------------
// Namespaces
// ---------------------------------------------------------------------------

#[test]
fn tool_namespace_copies_tools_with_metadata() {
    let tool = ok_tool("argless_function");
    let namespaced = tool_namespace("crm", "CRM tools", vec![tool.clone()]).unwrap();

    assert_eq!(namespaced.len(), 1);
    assert_eq!(namespaced[0].namespace(), Some("crm"));
    assert_eq!(
        namespaced[0].to_spec().namespace_description.as_deref(),
        Some("CRM tools")
    );
    assert_eq!(namespaced[0].qualified_name(), "crm.argless_function");
    assert_eq!(tool.namespace(), None);
    assert_eq!(tool.qualified_name(), "argless_function");
}

#[test]
fn tool_namespace_requires_non_empty_description() {
    for description in ["", "   "] {
        let err = tool_namespace("crm", description, vec![ok_tool("t")]).unwrap_err();
        assert!(
            err.to_string().contains("non-empty description"),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn tool_namespace_rejects_reserved_same_name_shape() {
    let tool = ok_tool("lookup_account");
    let err = tool_namespace("lookup_account", "Same-name namespace", vec![tool]).unwrap_err();
    assert!(
        err.to_string()
            .contains("synthetic namespace `lookup_account.lookup_account`"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Argument handling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn argless_function() {
    let tool = ok_tool("argless_function");
    assert_eq!(tool.name, "argless_function");
    assert_eq!(invoke_direct(&tool, "").await.unwrap(), "ok");
}

#[tokio::test]
async fn argless_with_context_ignores_extra_json() {
    let tool =
        FunctionTool::<()>::new_no_args("argless_with_context", "desc", |_ctx| async { Ok("ok") });
    assert_eq!(invoke_direct(&tool, "").await.unwrap(), "ok");
    // Extra JSON should not raise an error.
    assert_eq!(invoke_direct(&tool, r#"{"a": 1}"#).await.unwrap(), "ok");
}

#[tokio::test]
async fn simple_function_applies_defaults_and_rejects_missing_required() {
    let tool = FunctionTool::<()>::new(
        "simple_function",
        "desc",
        |_ctx, args: SimpleArgs| async move { Ok(args.a + args.b) },
    )
    .with_strict_json_schema(true);
    assert_eq!(tool.name, "simple_function");

    assert_eq!(invoke_direct(&tool, r#"{"a": 1}"#).await.unwrap(), "6");
    assert_eq!(
        invoke_direct(&tool, r#"{"a": 1, "b": 2}"#).await.unwrap(),
        "3"
    );

    // Missing required argument should raise an error.
    let err = invoke_direct(&tool, "").await.unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn bounded_argument_is_rejected_before_the_body_runs() {
    // Python validates `Field(ge=0, le=10)` before the call; the Rust port validates in the
    // argument type's `Deserialize`, so the body never sees the value.
    let calls = Arc::new(std::sync::Mutex::new(Vec::<i64>::new()));
    let seen = calls.clone();
    let tool = FunctionTool::<()>::new("score", "desc", move |_ctx, args: ScoreArgs| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push(args.score.0);
            Ok(args.score.0)
        }
    });

    for invalid in [-1, 11] {
        let err = invoke_direct(&tool, &format!(r#"{{"score": {invalid}}}"#))
            .await
            .unwrap_err();
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. }),
            "{err:?}"
        );
        assert!(calls.lock().unwrap().is_empty());
    }

    assert_eq!(
        invoke_direct(&tool, r#"{"score": 10}"#).await.unwrap(),
        "10"
    );
    assert_eq!(invoke_direct(&tool, r#"{"score": 0}"#).await.unwrap(), "0");
    assert_eq!(*calls.lock().unwrap(), vec![10, 0]);
}

#[tokio::test]
async fn complex_args_function() {
    let tool = FunctionTool::<()>::new(
        "complex_args_function",
        "desc",
        |_ctx, args: ComplexArgs| async move {
            Ok(format!(
                "{} {}{} {}",
                args.foo.a + args.foo.b,
                args.bar.x,
                args.bar.y,
                args.baz
            ))
        },
    );
    assert_eq!(tool.name, "complex_args_function");

    let valid = json!({"foo": {"a": 1}, "bar": {"x": "hello", "y": 10}});
    assert_eq!(
        invoke_direct(&tool, &valid.to_string()).await.unwrap(),
        "6 hello10 hello"
    );

    let valid = json!({"foo": {"a": 1, "b": 2}, "bar": {"x": "hello", "y": 10}});
    assert_eq!(
        invoke_direct(&tool, &valid.to_string()).await.unwrap(),
        "3 hello10 hello"
    );

    let valid = json!({"foo": {"a": 1, "b": 2}, "bar": {"x": "hello", "y": 10}, "baz": "world"});
    assert_eq!(
        invoke_direct(&tool, &valid.to_string()).await.unwrap(),
        "3 hello10 world"
    );

    // Missing required argument should raise an error.
    let err = invoke_direct(&tool, r#"{"foo": {"a": 1}}"#)
        .await
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );
}

#[test]
fn function_config_overrides_are_set_at_construction() {
    let tool = FunctionTool::<()>::new_no_args("custom_name", "custom description", |_ctx| async {
        Ok("ok")
    });
    assert_eq!(tool.name, "custom_name");
    assert_eq!(tool.description, "custom description");
}

#[test]
fn func_schema_is_strict() {
    let simple =
        FunctionTool::<()>::new("simple_function", "desc", |_ctx, _args: SimpleArgs| async {
            Ok("ok")
        });
    assert!(simple.strict_json_schema, "Should be strict by default");
    assert_eq!(
        simple.params_json_schema["additionalProperties"],
        json!(false)
    );

    let complex = FunctionTool::<()>::new("complex", "desc", |_ctx, _args: ComplexArgs| async {
        Ok("ok")
    });
    assert!(complex.strict_json_schema, "Should be strict by default");
    assert_eq!(
        complex.params_json_schema["additionalProperties"],
        json!(false)
    );
}

// ---------------------------------------------------------------------------
// Manually supplied schemas
// ---------------------------------------------------------------------------

async fn noop_raw(_ctx: ToolContext<()>, args: String) -> Result<String> {
    Ok(args)
}

#[test]
fn manual_function_tool_normalizes_typeless_object_schemas() {
    let tool = FunctionTool::<()>::try_new_raw(
        "test",
        "Processes nested data",
        json!({
            "properties": {
                "config": {"properties": {"key": {"type": "string"}}},
                "optional": {
                    "type": ["object", "null"],
                    "properties": {"value": {"type": "integer"}},
                },
            }
        }),
        true,
        noop_raw,
    )
    .unwrap();

    assert!(tool.strict_json_schema);
    assert_eq!(
        tool.params_json_schema,
        json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "additionalProperties": false,
                    "required": ["key"],
                },
                "optional": {
                    "type": ["object", "null"],
                    "properties": {"value": {"type": "integer"}},
                    "additionalProperties": false,
                    "required": ["value"],
                },
            },
            "additionalProperties": false,
            "required": ["config", "optional"],
        })
    );
}

/// `(definition name, ref)` pairs whose resolved schema must be the `string` definition.
const URI_FRAGMENT_CASES: &[(&str, &str)] = &[
    ("Value", "#/$defs/Value"),
    ("A B", "#/$defs/A%20B"),
    ("café", "#/$defs/caf%C3%A9"),
    ("A%20B", "#/$defs/A%2520B"),
    ("a/b", "#/$defs/a%7E1b"),
    ("a~1b", "#/$defs/a%7E01b"),
    ("A+B", "#/$defs/A+B"),
    ("A+B", "#/$defs/A%2BB"),
    ("Value", "#/%24defs%2FValue"),
];

#[test]
fn function_tool_uri_fragment_preserves_schema_meaning() {
    for (definition_name, reference) in URI_FRAGMENT_CASES {
        let schema = json!({
            "type": "object",
            "properties": {"value": {"$ref": *reference, "description": "value"}},
            "$defs": {*definition_name: {"type": "string"}},
        });
        let tool = FunctionTool::<()>::try_new_raw("test", "test", schema, true, noop_raw)
            .unwrap_or_else(|e| panic!("{definition_name} {reference}: {e}"));
        assert_eq!(
            tool.params_json_schema["properties"]["value"],
            json!({"type": "string", "description": "value"}),
            "{definition_name} {reference}"
        );
    }
}

#[test]
fn function_tool_uri_fragment_does_not_select_literal_percent_name() {
    let schema = json!({
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/A%20B", "description": "value"}},
        "$defs": {"A B": {"type": "string"}, "A%20B": {"type": "integer"}},
    });
    let tool = FunctionTool::<()>::try_new_raw("test", "test", schema, true, noop_raw).unwrap();
    assert_eq!(
        tool.params_json_schema["properties"]["value"],
        json!({"type": "string", "description": "value"})
    );
}

#[test]
fn function_tool_uri_fragment_rejects_invalid_utf8() {
    // Python raises UnicodeDecodeError while percent-decoding `%FF`. Rust `serde_json` values are
    // always UTF-8, so the only remaining case is an escape that decodes to an invalid sequence.
    let schema = json!({
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/%FF", "description": "value"}},
        "$defs": {"\u{FFFD}": {"type": "string"}},
    });
    assert!(FunctionTool::<()>::try_new_raw("test", "test", schema, true, noop_raw).is_err());
}

#[test]
fn function_tool_uri_fragment_rejects_nested_resource() {
    let schema = json!({
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/A%20B", "description": "value"}},
        "$defs": {"A B": {"$id": "https://example.test/nested", "type": "string"}},
    });
    let err = FunctionTool::<()>::try_new_raw("test", "test", schema, true, noop_raw).unwrap_err();
    assert!(err.to_string().contains("nested `$id` resource"), "{err}");
}

#[test]
fn manual_function_tool_rejects_root_union() {
    let schema = json!({
        "anyOf": [
            {"properties": {"value": {"type": "string"}}},
            {"type": "null"},
        ]
    });
    let err =
        FunctionTool::<()>::try_new_raw("test", "Processes nullable data", schema, true, noop_raw)
            .unwrap_err();
    assert!(
        err.to_string().contains("root of a strict JSON schema"),
        "{err}"
    );
}

#[test]
fn manual_function_tool_rejects_nested_typeless_open_map() {
    let schema = json!({
        "type": "object",
        "properties": {"metadata": {"additionalProperties": {"type": "string"}}},
    });
    let err = FunctionTool::<()>::try_new_raw("test", "Processes metadata", schema, true, noop_raw)
        .unwrap_err();
    assert!(err.to_string().contains("additionalProperties"), "{err}");
}

#[derive(Deserialize, JsonSchema)]
struct DataArgs {
    data: String,
}

#[tokio::test]
async fn manual_function_tool_creation_works() {
    let schema = serde_json::to_value(schemars::schema_for!(DataArgs)).unwrap();
    let tool = FunctionTool::<()>::try_new_raw(
        "test",
        "Processes extracted user data",
        schema.clone(),
        true,
        |_ctx, args: String| async move {
            let parsed: DataArgs = serde_json::from_str(&args)?;
            Ok(format!("{}_done", parsed.data))
        },
    )
    .unwrap();

    assert_eq!(tool.name, "test");
    assert_eq!(tool.description, "Processes extracted user data");
    assert!(tool.strict_json_schema);
    assert_eq!(tool.params_json_schema["properties"], schema["properties"]);
    assert_eq!(
        invoke_direct(&tool, r#"{"data": "hello"}"#).await.unwrap(),
        "hello_done"
    );

    let tool_not_strict = FunctionTool::<()>::try_new_raw(
        "test",
        "Processes extracted user data",
        schema,
        false,
        |_ctx, args: String| async move {
            let parsed: DataArgs = serde_json::from_str(&args)?;
            Ok(format!("{}_done", parsed.data))
        },
    )
    .unwrap();
    assert!(!tool_not_strict.strict_json_schema);
    assert!(
        tool_not_strict
            .params_json_schema
            .get("additionalProperties")
            .is_none()
    );
    assert_eq!(
        invoke_direct(&tool_not_strict, r#"{"data": "hello", "bar": "baz"}"#)
            .await
            .unwrap(),
        "hello_done"
    );
}

#[test]
fn function_tool_does_not_mutate_the_caller_schema() {
    let schema = json!({"type": "object", "properties": {"x": {"type": "string"}}});
    let snapshot = schema.clone();
    let tool = FunctionTool::<()>::try_new_raw("t", "d", schema.clone(), true, noop_raw).unwrap();

    assert_eq!(schema, snapshot);
    assert_eq!(
        tool.params_json_schema["additionalProperties"],
        json!(false)
    );
    assert_eq!(tool.params_json_schema["required"], json!(["x"]));
}

// ---------------------------------------------------------------------------
// Single `allOf` wrappers
// ---------------------------------------------------------------------------

fn entry() -> Value {
    json!({"type": "object", "properties": {"value": {"type": "string"}}})
}

#[test]
fn function_tool_rejects_single_all_of_that_broadens_closed_parent() {
    let parents = [
        json!({"type": "object", "properties": {}}),
        json!({"type": "object"}),
        json!({}),
    ];
    for parent in parents {
        for use_ref in [true, false] {
            let mut schema = parent.as_object().unwrap().clone();
            schema.insert("additionalProperties".into(), json!(false));
            schema.insert("$defs".into(), json!({"Entry": entry()}));
            let all_of_entry = if use_ref {
                json!({"$ref": "#/$defs/Entry"})
            } else {
                entry()
            };
            schema.insert("allOf".into(), json!([all_of_entry]));
            let schema = Value::Object(schema);
            let original = schema.clone();

            let err = FunctionTool::<()>::try_new_raw("t", "d", schema.clone(), true, noop_raw)
                .unwrap_err();
            assert!(err.to_string().contains("singleton `allOf`"), "{err}");

            // Non-strict keeps the schema exactly as given.
            let non_strict =
                FunctionTool::<()>::try_new_raw("t", "d", schema, false, noop_raw).unwrap();
            assert_eq!(non_strict.params_json_schema, original);
        }
    }
}

#[test]
fn function_tool_single_all_of_wrapper_remains_strict() {
    for parent in [
        json!({"type": "object"}),
        json!({"type": "object", "properties": {}}),
    ] {
        let mut schema = parent.as_object().unwrap().clone();
        schema.insert("description".into(), json!("wrapper"));
        schema.insert("allOf".into(), json!([entry()]));
        let original = Value::Object(schema.clone());

        let tool = FunctionTool::<()>::try_new_raw("t", "d", Value::Object(schema), true, noop_raw)
            .unwrap();
        assert_eq!(tool.params_json_schema["description"], "wrapper");
        assert_eq!(tool.params_json_schema["required"], json!(["value"]));
        assert_ne!(original, tool.params_json_schema);
    }
}

#[test]
fn function_tool_nested_single_all_of_ref_wrapper_remains_strict() {
    let schema = json!({
        "type": "object",
        "components": {
            "schemas": {
                "Inner": {"type": "object", "properties": {"value": {"type": "string"}}},
                "Outer": {"type": "object", "allOf": [{"$ref": "#/components/schemas/Inner"}]},
            }
        },
        "allOf": [{"$ref": "#/components/schemas/Outer"}],
    });
    let tool = FunctionTool::<()>::try_new_raw("t", "d", schema, true, noop_raw).unwrap();
    assert_eq!(tool.params_json_schema["required"], json!(["value"]));
    assert!(tool.params_json_schema.get("allOf").is_none());
}

#[test]
fn function_tool_single_all_of_closed_empty_object_remains_strict() {
    let schema = json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false,
        "allOf": [{"type": "object", "properties": {}}],
    });
    let tool = FunctionTool::<()>::try_new_raw("t", "d", schema, true, noop_raw).unwrap();
    assert_eq!(
        tool.params_json_schema["additionalProperties"],
        json!(false)
    );
    assert_eq!(tool.params_json_schema["properties"], json!({}));
}

// ---------------------------------------------------------------------------
// Depth limits
// ---------------------------------------------------------------------------

fn nested_object_schema(depth: usize) -> Value {
    let mut root = json!({"type": "object", "properties": {}});
    let mut current = &mut root;
    for _ in 0..depth {
        current["properties"]["child"] = json!({"type": "object", "properties": {}});
        current = current["properties"].get_mut("child").unwrap();
    }
    root
}

fn chained_ref_schema(depth: usize) -> Value {
    let mut definitions = serde_json::Map::new();
    for i in 0..depth {
        definitions.insert(
            format!("L{i}"),
            json!({"$ref": format!("#/$defs/L{}", i + 1)}),
        );
    }
    definitions.insert(format!("L{depth}"), json!({"type": "string"}));
    json!({
        "$defs": definitions,
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/L0", "description": "value"}},
    })
}

#[test]
fn function_tool_rejects_deep_schema_before_copying() {
    let schema = nested_object_schema(200);

    let err = FunctionTool::<()>::try_new_raw("strict_tool", "d", schema.clone(), true, noop_raw)
        .unwrap_err();
    assert!(err.to_string().contains("too deeply nested"), "{err}");

    let non_strict =
        FunctionTool::<()>::try_new_raw("non_strict_tool", "d", schema.clone(), false, noop_raw)
            .unwrap();
    assert_eq!(non_strict.params_json_schema, schema);
}

#[test]
fn function_tool_rejects_deeply_chained_refs_before_conversion() {
    let err = FunctionTool::<()>::try_new_raw(
        "strict_tool",
        "d",
        chained_ref_schema(200),
        true,
        noop_raw,
    )
    .unwrap_err();
    assert!(err.to_string().contains("too deeply nested"), "{err}");
}

#[test]
fn function_tool_rejects_deep_output_schema_before_copying() {
    let tool = FunctionTool::<()>::new_raw(
        "output_tool",
        "d",
        json!({"type": "object", "properties": {}}),
        noop_raw,
    );
    let err = tool
        .with_output_json_schema(nested_object_schema(200))
        .unwrap_err();
    assert!(err.to_string().contains("too deeply nested"), "{err}");
}

// ---------------------------------------------------------------------------
// Enablement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn is_enabled_bool_and_callable() {
    let disabled =
        FunctionTool::<BoolCtx>::new_no_args("disabled_tool", "d", |_ctx| async { Ok("nope") })
            .with_is_enabled(false);
    let another =
        FunctionTool::<BoolCtx>::new_no_args("another_tool", "d", |_ctx| async { Ok("hi") })
            .with_is_enabled_fn(|ctx, _agent| async move { ctx.context().enable_tools });
    let third = FunctionTool::<BoolCtx>::new_raw(
        "third_tool",
        "third tool",
        json!({}),
        |_ctx, _args| async { Ok("third") },
    )
    .with_is_enabled_fn(|ctx, _agent| async move { ctx.context().enable_tools });

    let agent = Arc::new(
        Agent::<BoolCtx>::new("t")
            .with_tool(disabled)
            .with_tool(another)
            .with_tool(third),
    );
    let context_1 = RunContext::new(BoolCtx {
        enable_tools: false,
    });
    let context_2 = RunContext::new(BoolCtx { enable_tools: true });

    assert!(agent.get_all_tools(&context_1).await.unwrap().is_empty());

    let tools = agent.get_all_tools(&context_2).await.unwrap();
    let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
    assert_eq!(names, ["another_tool", "third_tool"]);
}

#[tokio::test]
async fn get_all_tools_preserves_explicit_tool_search_when_deferred_tools_are_disabled() {
    let deferred =
        FunctionTool::<BoolCtx>::new_no_args("deferred_lookup", "d", |_ctx| async { Ok("loaded") })
            .with_defer_loading(true)
            .with_is_enabled_fn(|ctx, _agent| async move { ctx.context().enable_tools });
    let deferred_name = deferred.name.clone();

    let agent = Arc::new(
        Agent::<BoolCtx>::new("t")
            .with_tool(deferred)
            .with_tool(ToolSearchTool::new()),
    );

    let disabled = agent
        .get_all_tools(&RunContext::new(BoolCtx {
            enable_tools: false,
        }))
        .await
        .unwrap();
    assert_eq!(disabled.len(), 1);
    assert!(matches!(disabled[0], Tool::ToolSearch(_)));

    let enabled = agent
        .get_all_tools(&RunContext::new(BoolCtx { enable_tools: true }))
        .await
        .unwrap();
    assert_eq!(enabled[0].name(), deferred_name);
    assert!(matches!(enabled[1], Tool::ToolSearch(_)));
}

#[tokio::test]
async fn get_all_tools_keeps_tool_search_for_namespace_only_tools() {
    let namespaced = tool_namespace(
        "crm",
        "CRM tools",
        vec![FunctionTool::<BoolCtx>::new_no_args(
            "lookup_account",
            "d",
            |_ctx| async { Ok("ok") },
        )],
    )
    .unwrap()
    .remove(0);

    let agent = Arc::new(
        Agent::<BoolCtx>::new("t")
            .with_tool(namespaced)
            .with_tool(ToolSearchTool::new()),
    );
    let tools = agent
        .get_all_tools(&RunContext::new(BoolCtx {
            enable_tools: false,
        }))
        .await
        .unwrap();
    assert_eq!(tools[0].name(), "lookup_account");
    assert!(matches!(tools[1], Tool::ToolSearch(_)));
}

#[tokio::test]
async fn get_all_tools_keeps_tool_search_for_deferred_hosted_mcp() {
    let hosted = HostedMCPTool::<BoolCtx>::new(json!({
        "type": "mcp",
        "server_label": "crm_server",
        "server_url": "https://example.com/mcp",
        "defer_loading": true,
    }));
    let agent = Arc::new(
        Agent::<BoolCtx>::new("t")
            .with_tool(hosted)
            .with_tool(ToolSearchTool::new()),
    );
    let tools = agent
        .get_all_tools(&RunContext::new(BoolCtx {
            enable_tools: false,
        }))
        .await
        .unwrap();
    assert!(matches!(tools[0], Tool::HostedMCP(_)));
    assert!(matches!(tools[1], Tool::ToolSearch(_)));
}

// ---------------------------------------------------------------------------
// Guardrail wiring
// ---------------------------------------------------------------------------

#[test]
fn function_tool_accepts_guardrail_arguments() {
    use openai_agents_sdk::tool_guardrails::{
        ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData,
        ToolOutputGuardrail, ToolOutputGuardrailData,
    };

    let reject =
        ToolInputGuardrail::<()>::new("reject_args", |_data: ToolInputGuardrailData<()>| async {
            Ok(ToolGuardrailFunctionOutput::reject_content("blocked"))
        });
    let allow = ToolOutputGuardrail::<()>::new(
        "allow_output",
        |data: ToolOutputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow()
                .with_output_info(json!({"echo": data.output.to_model_string()})))
        },
    );
    let tool = FunctionTool::<()>::new_no_args("guarded", "d", |_ctx| async { Ok("ok") })
        .with_input_guardrail(reject)
        .with_output_guardrail(allow);

    assert_eq!(tool.tool_input_guardrails.len(), 1);
    assert_eq!(tool.tool_output_guardrails.len(), 1);
    assert_eq!(tool.tool_input_guardrails[0].name, "reject_args");
    assert_eq!(tool.tool_output_guardrails[0].name, "allow_output");
}

#[allow(dead_code)] // only its schema and deserialization are exercised
#[derive(Deserialize, JsonSchema)]
struct OpenMapArgs {
    metadata: std::collections::HashMap<String, String>,
}

#[test]
fn typed_tool_with_open_map_argument_is_rejected_by_try_new() {
    // Python raises UserError at construction for a typed open map; `try_new` returns it.
    let err = FunctionTool::<()>::try_new("open_map", "d", |_ctx, _args: OpenMapArgs| async {
        Ok("ok")
    })
    .unwrap_err();
    assert!(err.to_string().contains("additionalProperties"), "{err}");
}

#[test]
#[should_panic(expected = "additionalProperties")]
fn typed_tool_with_open_map_argument_panics_in_new() {
    let _ = FunctionTool::<()>::new("open_map", "d", |_ctx, _args: OpenMapArgs| async {
        Ok("ok")
    });
}

#[allow(dead_code)] // only its schema and deserialization are exercised
#[derive(Deserialize, JsonSchema)]
struct AnyValueArgs {
    payload: serde_json::Value,
}

#[test]
fn typed_tool_with_unconstrained_value_argument_builds_strict() {
    // Python's `Any` parameter (pydantic `{}`) is accepted; schemars emits `true` for `Value`.
    let tool = FunctionTool::<()>::try_new("any_value", "d", |_ctx, _args: AnyValueArgs| async {
        Ok("ok")
    })
    .unwrap();
    assert!(tool.strict_json_schema);
    assert_eq!(tool.params_json_schema["required"], json!(["payload"]));
}
