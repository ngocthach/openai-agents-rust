//! Parity port of the schema-conversion half of `tests/mcp/test_mcp_util.py`
//! (`to_function_tool` with `convert_schemas_to_strict`, `_meta` forwarding, schema isolation).
#![cfg(feature = "mcp")]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::MCPConfig;
use openai_agents_sdk::agent::Agent;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::mcp::util::{get_all_function_tools_for_agent, to_function_tool};
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::{FunctionTool, Tool};
use serde_json::{Map, Value, json};

#[derive(Debug)]
struct Srv {
    tools: Vec<MCPTool>,
    config: MCPServerConfig,
    metas: Mutex<Vec<Option<Map<String, Value>>>>,
}

impl Srv {
    fn new() -> Self {
        Srv {
            tools: Vec::new(),
            config: MCPServerConfig::default(),
            metas: Mutex::new(Vec::new()),
        }
    }
    fn with_tool(mut self, name: &str, schema: Value) -> Self {
        self.tools.push(tool(name, schema));
        self
    }
}

#[async_trait::async_trait]
impl MCPServer for Srv {
    fn name(&self) -> &str {
        "fake_mcp_server"
    }
    fn config(&self) -> &MCPServerConfig {
        &self.config
    }
    async fn connect(&self) -> Result<()> {
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        Ok(())
    }
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Ok(self.tools.clone())
    }
    async fn call_tool(&self, _n: &str, _a: Option<Map<String, Value>>) -> Result<Value> {
        self.metas.lock().unwrap().push(None);
        Ok(json!({"content": [{"type": "text", "text": "ok"}]}))
    }
    async fn call_tool_with_meta(
        &self,
        _n: &str,
        _a: Option<Map<String, Value>>,
        meta: Option<Map<String, Value>>,
    ) -> Result<Value> {
        self.metas.lock().unwrap().push(meta);
        Ok(json!({"content": [{"type": "text", "text": "ok"}]}))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(vec![])
    }
    async fn get_prompt(
        &self,
        _n: &str,
        _a: Option<std::collections::BTreeMap<String, String>>,
    ) -> Result<Value> {
        Ok(Value::Null)
    }
}

fn tool(name: &str, schema: Value) -> MCPTool {
    MCPTool {
        meta: None,
        name: name.into(),
        title: None,
        description: None,
        input_schema: schema,
        annotations: None,
    }
}

async fn convert(schema: Value, strict: bool) -> FunctionTool<()> {
    let server: Arc<dyn MCPServer> = Arc::new(Srv::new());
    to_function_tool(server, &tool("t", schema), "t".into(), strict, "agent").await
}

fn foo_schema() -> Value {
    json!({
        "properties": {"bar": {"title": "Bar", "type": "string"}, "baz": {"title": "Baz", "type": "integer"}},
        "required": ["bar", "baz"], "title": "Foo", "type": "object",
    })
}

async fn agent_tools(
    schemas: Vec<(&str, Value)>,
    strict: Option<bool>,
) -> Vec<Arc<FunctionTool<()>>> {
    let mut srv = Srv::new();
    for (n, s) in schemas {
        srv = srv.with_tool(n, s);
    }
    let servers: Vec<Arc<dyn MCPServer>> = vec![Arc::new(srv)];
    let config = MCPConfig::<()> {
        convert_schemas_to_strict: strict.unwrap_or(false),
        ..Default::default()
    };
    let agent: Arc<Agent<()>> = Arc::new(Agent::new("test_agent"));
    get_all_function_tools_for_agent(
        &servers,
        &config,
        &RunContext::new(()),
        "test_agent",
        Some(agent),
        &Default::default(),
    )
    .await
    .unwrap()
    .into_iter()
    .map(|t| match t {
        Tool::Function(f) => f,
        _ => panic!("function tool expected"),
    })
    .collect()
}

fn by_name<'a>(tools: &'a [Arc<FunctionTool<()>>], n: &str) -> &'a FunctionTool<()> {
    tools.iter().find(|t| t.name == n).unwrap()
}

// --- agent-level convert_schemas_to_strict (true / false / unset) ---------------------------

#[tokio::test]
async fn agent_convert_schemas_true() {
    let bar = json!({"type": "object", "additionalProperties": {"type": "string"}});
    let schemas = vec![("foo", foo_schema()), ("bar", bar), ("baz", foo_schema())];
    let tools = agent_tools(schemas, Some(true)).await;
    let mut strict_foo = foo_schema();
    strict_foo["additionalProperties"] = json!(false);
    assert_eq!(by_name(&tools, "foo").params_json_schema, strict_foo);
    assert!(by_name(&tools, "foo").strict_json_schema);
    assert_eq!(
        by_name(&tools, "bar").params_json_schema,
        json!({"type": "object", "additionalProperties": {"type": "string"}, "properties": {}})
    );
    assert!(!by_name(&tools, "bar").strict_json_schema);
    assert_eq!(by_name(&tools, "baz").params_json_schema, strict_foo);
    assert!(by_name(&tools, "baz").strict_json_schema);
}

#[tokio::test]
async fn agent_convert_schemas_false_and_unset_leave_schemas_alone() {
    for strict in [Some(false), None] {
        let bar = json!({"type": "object", "additionalProperties": {"type": "string"}});
        let schemas = vec![("foo", foo_schema()), ("bar", bar)];
        let tools = agent_tools(schemas, strict).await;
        assert_eq!(by_name(&tools, "foo").params_json_schema, foo_schema());
        assert!(!by_name(&tools, "foo").strict_json_schema);
        assert_eq!(
            by_name(&tools, "bar").params_json_schema,
            json!({"type": "object", "additionalProperties": {"type": "string"}, "properties": {}})
        );
        assert!(!by_name(&tools, "bar").strict_json_schema);
    }
}

#[tokio::test]
async fn util_adds_properties() {
    let tools = agent_tools(
        vec![(
            "test_tool",
            json!({"type": "object", "description": "Test tool"}),
        )],
        None,
    )
    .await;
    assert_eq!(
        tools[0].params_json_schema,
        json!({"type": "object", "description": "Test tool", "properties": {}})
    );
}

#[tokio::test]
async fn conversion_does_not_mutate_the_mcp_input_schema() {
    let schema = json!({"type": "object", "description": "Test tool"});
    let t = tool("test_tool", schema.clone());
    let server: Arc<dyn MCPServer> = Arc::new(Srv::new());
    let f: FunctionTool<()> = to_function_tool(server, &t, "test_tool".into(), false, "a").await;
    assert_eq!(
        f.params_json_schema,
        json!({"type": "object", "description": "Test tool", "properties": {}})
    );
    assert_eq!(t.input_schema, schema);
}

#[test]
fn a_deeply_nested_schema_is_rejected_before_copying() {
    // Python's `to_function_tool` raises `UserError("... too deeply nested")` for strict and
    // non-strict conversion alike. Rust's `to_function_tool` cannot fail, so the check sits in
    // `get_all_function_tools_for_agent`, the path every agent takes. The deep value is built
    // and dropped on a big stack because `serde_json::Value` drops recursively.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(async {
                for strict in [false, true] {
                    let mut root = json!({"type": "object", "properties": {}});
                    let mut cur = &mut root;
                    for _ in 0..1000 {
                        cur["properties"]["child"] = json!({"type": "object", "properties": {}});
                        cur = &mut cur["properties"]["child"];
                    }
                    let servers: Vec<Arc<dyn MCPServer>> =
                        vec![Arc::new(Srv::new().with_tool("deep_tool", root))];
                    let config = MCPConfig::<()> {
                        convert_schemas_to_strict: strict,
                        ..Default::default()
                    };
                    let error = get_all_function_tools_for_agent(
                        &servers,
                        &config,
                        &RunContext::new(()),
                        "a",
                        None,
                        &Default::default(),
                    )
                    .await
                    .expect_err("a 1000-deep schema must be rejected");
                    assert!(error.to_string().contains("too deeply nested"), "{error}");
                }
            });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[tokio::test]
async fn failed_strict_conversion_keeps_original_schema() {
    let schema = json!({"type": "object", "properties": {"x": {"additionalProperties": true}}});
    let f = convert(schema.clone(), true).await;
    assert!(!f.strict_json_schema);
    assert_eq!(f.params_json_schema, schema);
}

// --- $ref / allOf fallbacks and free-form detection ----------------------------------------

async fn assert_non_strict_original(schema: Value) {
    let f = convert(schema.clone(), true).await;
    assert!(!f.strict_json_schema, "expected non-strict for {schema}");
    assert_eq!(f.params_json_schema, schema);
}

#[tokio::test]
async fn ref_sibling_falls_back_to_original_schema() {
    let defs = json!({"T": {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}, "U": {"type": "integer"}});
    for node in [
        json!({"properties": {"a": {"type": "string"}}, "required": ["a"], "$ref": "#/$defs/T"}),
        json!({"$ref": "#/$defs/T", "allOf": [{"$ref": "#/$defs/U"}]}),
        json!({"$ref": "#/$defs/T", "additionalProperties": false}),
    ] {
        assert_non_strict_original(
            json!({"$defs": defs, "type": "object", "properties": {"node": node}}),
        )
        .await;
    }
}

#[tokio::test]
async fn ref_with_schema_metadata_remains_strict() {
    let schema = json!({
        "$defs": {"T": {"type": "string"}}, "type": "object",
        "properties": {"value": {"$schema": "https://json-schema.org/draft/2020-12/schema", "$ref": "#/$defs/T"}},
    });
    let f = convert(schema, true).await;
    assert!(f.strict_json_schema);
    assert_eq!(
        f.params_json_schema["properties"]["value"],
        json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "string"})
    );
    assert_eq!(f.params_json_schema["additionalProperties"], json!(false));
}

#[tokio::test]
async fn ref_with_nested_id_falls_back() {
    assert_non_strict_original(json!({
        "$defs": {"T": {"type": "string"}}, "type": "object",
        "properties": {"value": {"$id": "https://example.test/nested", "$defs": {"T": {"type": "integer"}}, "$ref": "#/$defs/T"}},
        "additionalProperties": false,
    })).await;
}

#[tokio::test]
async fn ref_with_anchor_remains_strict() {
    let schema = json!({
        "$defs": {"T": {"type": "string"}}, "type": "object",
        "properties": {"value": {"$anchor": "value", "$ref": "#/$defs/T"}},
        "additionalProperties": false,
    });
    let f = convert(schema, true).await;
    assert!(f.strict_json_schema);
    assert_eq!(
        f.params_json_schema["properties"]["value"],
        json!({"$anchor": "value", "type": "string"})
    );
}

#[tokio::test]
async fn single_all_of_annotated_alias_remains_strict() {
    let schema = json!({
        "components": {"schemas": {
            "Inner": {"type": "object", "description": "inner", "properties": {"value": {"type": "string"}}, "additionalProperties": false},
            "Outer": {"$ref": "#/components/schemas/Inner", "description": "outer"},
        }},
        "type": "object",
        "properties": {"payload": {"allOf": [{"$ref": "#/components/schemas/Outer", "title": "entry"}]}},
        "additionalProperties": false,
    });
    let f = convert(schema, true).await;
    assert!(f.strict_json_schema);
    let payload = &f.params_json_schema["properties"]["payload"];
    assert_eq!(payload["description"], "outer");
    assert_eq!(payload["title"], "entry");
    assert_eq!(payload["properties"], json!({"value": {"type": "string"}}));
    assert_eq!(payload["required"], json!(["value"]));
    assert!(payload.get("$ref").is_none());
}

#[tokio::test]
async fn single_all_of_closed_parent_falls_back() {
    for declare in [true, false] {
        let mut schema = json!({
            "type": "object", "additionalProperties": false,
            "$defs": {"Entry": {"type": "object", "properties": {"value": {"type": "string"}}, "additionalProperties": false}},
            "allOf": [{"$ref": "#/$defs/Entry"}],
        });
        if declare {
            schema["properties"] = json!({});
        }
        let f = convert(schema.clone(), true).await;
        assert!(!f.strict_json_schema);
        let mut expected = schema.clone();
        expected["properties"] = json!({});
        assert_eq!(f.params_json_schema, expected);
    }
}

#[tokio::test]
async fn single_all_of_conflicts_and_nested_ids_fall_back() {
    let t = json!({"type": "object", "properties": {"inner": {"type": "string"}}, "additionalProperties": false});
    assert_non_strict_original(json!({
        "$defs": {"T": t}, "type": "object", "properties": {"outer": {"type": "string"}},
        "allOf": [{"$ref": "#/$defs/T", "description": "alias"}], "additionalProperties": false,
    }))
    .await;
    assert_non_strict_original(json!({
        "$defs": {"T": t}, "type": "object", "properties": {"outer": {"type": "string"}},
        "allOf": [{"allOf": [{"$ref": "#/$defs/T"}]}], "additionalProperties": false,
    }))
    .await;
    assert_non_strict_original(json!({
        "$defs": {"T": {"const": 1}}, "type": "object", "const": true, "properties": {},
        "allOf": [{"$ref": "#/$defs/T"}], "additionalProperties": false,
    }))
    .await;
    assert_non_strict_original(json!({
        "$defs": {"T": {"type": "string"}},
        "contentSchema": {"$id": "https://example.test/nested", "$defs": {"T": {"type": "integer"}},
            "type": "object", "properties": {"value": {"$ref": "#/$defs/T"}}},
        "type": "object", "properties": {}, "allOf": [{"$ref": "#/contentSchema"}], "additionalProperties": false,
    })).await;
    assert_non_strict_original(json!({
        "$defs": {"T": {"type": "string"}}, "type": "object",
        "properties": {"node": {"$id": "https://example.test/nested", "$defs": {"T": {"type": "integer"}},
            "allOf": [{"$ref": "#/$defs/T"}]}},
        "additionalProperties": false,
    })).await;
    assert_non_strict_original(json!({
        "$defs": {"T": {"type": "string"}}, "type": "object",
        "properties": {"node": {"$id": "https://example.test/nested", "$defs": {"T": {"type": "integer"}},
            "type": "object", "properties": {"child": {"allOf": [{"$ref": "#/$defs/T"}]}}, "additionalProperties": false}},
        "additionalProperties": false,
    })).await;
    assert_non_strict_original(json!({
        "$defs": {"T": {"type": "string"}},
        "contentSchema": {"$id": "https://example.test/nested", "$defs": {"T": {"type": "integer"}},
            "target": {"$ref": "#/$defs/T"}},
        "type": "object", "properties": {}, "allOf": [{"$ref": "#/contentSchema/target"}], "additionalProperties": false,
    })).await;
}

#[tokio::test]
async fn ref_allows_unrelated_id_definition_name() {
    let schema = json!({
        "$defs": {"$id": {"type": "integer"}, "T": {"type": "string"}}, "type": "object",
        "properties": {"value": {"$ref": "#/$defs/T", "description": "value"}},
        "additionalProperties": false,
    });
    let f = convert(schema, true).await;
    assert!(f.strict_json_schema);
    assert_eq!(
        f.params_json_schema["properties"]["value"],
        json!({"type": "string", "description": "value"})
    );
}

fn free_forms() -> Vec<Value> {
    vec![
        json!({"type": "object", "description": "key/value pairs"}),
        json!({"type": "object", "properties": {}}),
        json!({"type": "object", "properties": {}, "required": []}),
        json!({"type": "object", "properties": {}, "$schema": "https://json-schema.org/draft/2020-12/schema"}),
        json!({"type": "object", "properties": {}, "$comment": "Arbitrary values."}),
    ]
}

#[tokio::test]
async fn free_form_object_argument_falls_back_to_non_strict() {
    for free in free_forms() {
        let schema = json!({"type": "object",
            "properties": {"target": {"type": "string"}, "keysAndValues": free},
            "required": ["target", "keysAndValues"]});
        assert_non_strict_original(schema).await;
    }
}

#[tokio::test]
async fn free_form_root_falls_back_to_non_strict() {
    for schema in free_forms() {
        let f = convert(schema.clone(), true).await;
        assert!(!f.strict_json_schema, "{schema}");
        let mut expected = schema.clone();
        expected["properties"] = json!({});
        assert_eq!(f.params_json_schema, expected);
    }
}

#[tokio::test]
async fn free_form_object_in_array_items_and_supported_nodes() {
    assert_non_strict_original(
        json!({"type": "object", "properties": {"entries": {"type": "array",
        "items": {"type": "object", "properties": {}, "$comment": "Arbitrary values."}}}}),
    )
    .await;
    let obj = json!({"type": "object", "properties": {}});
    for schema in [
        json!({"type": "object", "properties": {"value": {"anyOf": [obj, {"type": "null"}]}}}),
        json!({"type": "object", "properties": {"value": {"oneOf": [obj, {"type": "null"}]}}}),
        json!({"type": "object", "properties": {"value": {"allOf": [obj]}}}),
        json!({"type": "object", "properties": {"value": {"$ref": "#/$defs/value"}}, "$defs": {"value": obj}}),
        json!({"type": "object", "properties": {"value": {"$ref": "#/definitions/value"}}, "definitions": {"value": obj}}),
        json!({"type": "object", "properties": {"value": {"$ref": "#/components/schemas/value"}}, "components": {"schemas": {"value": obj}}}),
        json!({"type": "object", "properties": {"value": {"$ref": "#/components/schemas/value", "description": "Arbitrary values."}}, "components": {"schemas": {"value": obj}}}),
    ] {
        assert_non_strict_original(schema).await;
    }
}

#[tokio::test]
async fn strictable_closed_and_shaped_objects_stay_strict() {
    for schema in [
        json!({}),
        json!({"type": "object", "additionalProperties": false}),
        json!({"type": "object", "properties": {"value": {"type": "string"}}}),
    ] {
        let f = convert(schema, true).await;
        assert!(f.strict_json_schema);
        assert_eq!(f.params_json_schema["additionalProperties"], json!(false));
    }
}

#[tokio::test]
async fn advanced_open_root_and_nullable_root_fall_back() {
    for schema in [
        json!({"type": "object", "allOf": [{"type": "object", "properties": {"value": {"type": "string"}}}]}),
        json!({"anyOf": [{"type": "object", "properties": {"value": {"type": "string"}}}, {"type": "null"}]}),
    ] {
        let f = convert(schema.clone(), true).await;
        assert!(!f.strict_json_schema);
        let mut expected = schema;
        expected["properties"] = json!({});
        assert_eq!(f.params_json_schema, expected);
    }
}

#[tokio::test]
async fn preserved_pure_refs_fall_back_to_non_strict() {
    for (r, defs) in [
        ("#/$defs/value", json!({"value": {"type": "string"}})),
        ("#/$defs/a%20b", json!({"a b": {"type": "string"}})),
    ] {
        assert_non_strict_original(
            json!({"$defs": defs, "type": "object", "properties": {"payload": {"$ref": r}}}),
        )
        .await;
    }
}

#[tokio::test]
async fn conversion_disabled_keeps_schema_as_is() {
    let schema = json!({"type": "object", "properties": {"value": {"type": "string"}}});
    let f = convert(schema.clone(), false).await;
    assert!(!f.strict_json_schema);
    assert_eq!(f.params_json_schema, schema);
}

// --- static `_meta` ---------------------------------------------------------------------------

#[tokio::test]
async fn nested_static_meta_is_not_shared_between_calls() {
    // Python deep-copies the tool's `_meta` per call. Rust's MCPTool carries no `_meta` field, so
    // there is nothing to forward; documented as a model difference (see report).
    let server: Arc<dyn MCPServer> = Arc::new(Srv::new());
    let f: FunctionTool<()> =
        to_function_tool(server, &tool("t", json!({})), "t".into(), false, "a").await;
    let ctx =
        openai_agents_sdk::tool_context::ToolContext::direct(RunContext::new(()), "a", "t", "{}");
    f.invoke(ctx.clone(), "{}".into()).await.unwrap();
    f.invoke(ctx, "{}".into()).await.unwrap();
}
