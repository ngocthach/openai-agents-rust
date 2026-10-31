//! Port of `tests/models/test_openai_responses_converter.py`.
//!
//! Python exercises `Converter.convert_tool_choice` / `convert_tools` / `get_response_format`
//! directly. Rust has no `Converter` class: the same conversion happens inside
//! `OpenAIResponsesModel::build_request_body`, so every case goes through that, reading
//! `body["tool_choice"]`, `body["tools"]`, `body["include"]` and `body["text"]`.

use std::sync::Arc;

use openai_agents_sdk::agent_output::{AgentOutputSchema, AgentOutputSchemaBase};
use openai_agents_sdk::model_settings::{MCPToolChoice, ModelSettings, ToolChoice};
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::models::interface::{FunctionToolSpec, ModelRequest, ToolSpec};
use openai_agents_sdk::models::openai_responses::OpenAIResponsesModel;
use openai_agents_sdk::prompts::Prompt;
use openai_agents_sdk::tool::{FileSearchTool, Tool, WebSearchTool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn model(name: &str) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new(name, OpenAIClient::new("sk-test"))
}

fn func(name: &str) -> FunctionToolSpec {
    FunctionToolSpec {
        name: name.into(),
        description: "d".into(),
        parameters: json!({"type": "object", "properties": {}, "additionalProperties": false}),
        strict: true,
        ..Default::default()
    }
}

fn tool(name: &str) -> ToolSpec {
    ToolSpec::Function(func(name))
}

fn deferred(name: &str) -> ToolSpec {
    ToolSpec::Function(FunctionToolSpec {
        defer_loading: true,
        ..func(name)
    })
}

/// A function tool inside `tool_namespace(name=ns, description=desc, ...)`.
fn namespaced(ns: &str, desc: &str, name: &str) -> ToolSpec {
    ToolSpec::Function(FunctionToolSpec {
        namespace: Some(ns.into()),
        namespace_description: Some(desc.into()),
        ..func(name)
    })
}

fn hosted(param: Value) -> ToolSpec {
    ToolSpec::Hosted {
        param,
        includes: Vec::new(),
    }
}

fn hosted_with(param: Value, includes: &[&str]) -> ToolSpec {
    ToolSpec::Hosted {
        param,
        includes: includes.iter().map(|s| s.to_string()).collect(),
    }
}

fn tool_search() -> ToolSpec {
    hosted(json!({"type": "tool_search"}))
}

fn computer_tool() -> ToolSpec {
    hosted(json!({"type": "computer", "environment": "mac",
        "display_width": 800, "display_height": 600}))
}

fn request(tools: Vec<ToolSpec>, choice: Option<&str>) -> ModelRequest {
    ModelRequest {
        tools,
        model_settings: ModelSettings {
            tool_choice: choice.map(ToolChoice::from),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// The converted `tool_choice`, or `None` when the key is absent (Python's `omit`).
fn choice_of(m: &OpenAIResponsesModel, req: &ModelRequest) -> Option<Value> {
    m.build_request_body(req, false)
        .unwrap()
        .get("tool_choice")
        .cloned()
}

fn choice(tools: Vec<ToolSpec>, c: &str) -> Value {
    choice_of(&model("gpt-4.1"), &request(tools, Some(c))).unwrap()
}

fn choice_err(tools: Vec<ToolSpec>, c: &str) -> String {
    model("gpt-4.1")
        .build_request_body(&request(tools, Some(c)), false)
        .unwrap_err()
        .to_string()
}

fn tools_of(m: &OpenAIResponsesModel, req: &ModelRequest) -> Vec<Value> {
    m.build_request_body(req, false)
        .unwrap()
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn includes_of(m: &OpenAIResponsesModel, req: &ModelRequest) -> Vec<String> {
    m.build_request_body(req, false)
        .unwrap()
        .get("include")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------- tool choice

#[test]
fn convert_tool_choice_standard_values() {
    let m = model("gpt-4.1");
    // `None` leaves the key out entirely (Python returns `omit`).
    assert_eq!(choice_of(&m, &request(vec![], None)), None);
    assert_eq!(choice(vec![], "auto"), json!("auto"));
    assert_eq!(choice(vec![], "required"), json!("required"));
    assert_eq!(choice(vec![], "none"), json!("none"));
    assert_eq!(
        choice(vec![], "file_search"),
        json!({"type": "file_search"})
    );
    assert_eq!(
        choice(vec![], "web_search_preview"),
        json!({"type": "web_search_preview"})
    );
    assert_eq!(
        choice(vec![], "my_function"),
        json!({"type": "function", "name": "my_function"})
    );
}

#[test]
fn convert_tool_choice_computer_variants_follow_effective_model() {
    let ga =
        |c: &str| choice_of(&model("gpt-5.4"), &request(vec![computer_tool()], Some(c))).unwrap();
    assert_eq!(ga("computer"), json!({"type": "computer"}));
    assert_eq!(ga("computer_use"), json!({"type": "computer"}));
    assert_eq!(ga("computer_use_preview"), json!({"type": "computer"}));

    let preview = |c: &str| {
        choice_of(
            &model("computer-use-preview"),
            &request(vec![computer_tool()], Some(c)),
        )
        .unwrap()
    };
    assert_eq!(
        preview("computer_use_preview"),
        json!({"type": "computer_use_preview"})
    );
    assert_eq!(preview("computer"), json!({"type": "computer_use_preview"}));
    assert_eq!(
        preview("computer_use"),
        json!({"type": "computer_use_preview"})
    );

    // Prompt-managed request: the model is omitted, so a GA alias pins the GA selector.
    let unresolved = |c: &str| {
        let mut req = request(vec![computer_tool()], Some(c));
        req.prompt = Some(Prompt::new("pmpt_123"));
        choice_of(&model("gpt-4").with_explicit_model(false), &req).unwrap()
    };
    assert_eq!(unresolved("computer_use"), json!({"type": "computer"}));
    assert_eq!(unresolved("computer"), json!({"type": "computer"}));
}

#[test]
fn convert_tool_choice_allows_function_named_computer_without_computer_tool() {
    assert_eq!(
        choice(vec![tool("computer")], "computer"),
        json!({"type": "function", "name": "computer"})
    );
    assert_eq!(
        choice(vec![tool("computer_use")], "computer_use"),
        json!({"type": "function", "name": "computer_use"})
    );
}

#[test]
fn convert_tool_choice_builtin_shell_and_apply_patch() {
    assert_eq!(
        choice(
            vec![hosted(
                json!({"type": "shell", "environment": {"type": "local"}})
            )],
            "shell"
        ),
        json!({"type": "shell"})
    );
    assert_eq!(
        choice(vec![hosted(json!({"type": "apply_patch"}))], "apply_patch"),
        json!({"type": "apply_patch"})
    );
}

#[test]
fn convert_tool_choice_allows_function_named_tool_search() {
    assert_eq!(
        choice(vec![tool("tool_search")], "tool_search"),
        json!({"type": "function", "name": "tool_search"})
    );
}

#[test]
fn convert_tool_choice_rejects_hosted_tool_search_choice() {
    let err = choice_err(
        vec![deferred("lookup_weather"), tool_search()],
        "tool_search",
    );
    assert!(
        err.contains("is not supported for ToolSearchTool()"),
        "{err}"
    );
}

#[test]
fn convert_tool_choice_rejects_tool_search_without_matching_definition() {
    let err = choice_err(
        vec![namespaced("crm", "CRM tools", "lookup_weather")],
        "tool_search",
    );
    assert!(
        err.contains(
            "requires ToolSearchTool() or a real top-level function tool named `tool_search`"
        ),
        "{err}"
    );
}

#[test]
fn convert_tool_choice_allows_function_named_tool_search_with_hosted_tool_search() {
    assert_eq!(
        choice(
            vec![
                tool("tool_search"),
                deferred("lookup_weather"),
                tool_search()
            ],
            "tool_search"
        ),
        json!({"type": "function", "name": "tool_search"})
    );
}

#[test]
fn convert_tool_choice_required_allows_eager_namespace_tools_without_tool_search() {
    assert_eq!(
        choice(
            vec![namespaced("crm", "CRM tools", "lookup_account")],
            "required"
        ),
        json!("required")
    );
}

#[test]
fn convert_tool_choice_required_allows_eager_namespace_tools_with_tool_search() {
    assert_eq!(
        choice(
            vec![
                namespaced("crm", "CRM tools", "lookup_account"),
                tool_search()
            ],
            "required"
        ),
        json!("required")
    );
}

#[test]
fn convert_tool_choice_required_rejects_deferred_function_tools() {
    let err = choice_err(vec![deferred("lookup_account")], "required");
    assert!(err.contains("ToolSearchTool()"), "{err}");
}

#[test]
fn convert_tool_choice_required_allows_deferred_function_tools_with_tool_search() {
    assert_eq!(
        choice(vec![deferred("lookup_account"), tool_search()], "required"),
        json!("required")
    );
}

#[test]
fn convert_tool_choice_required_allows_deferred_hosted_mcp_tools_with_tool_search() {
    let mcp = hosted(json!({"type": "mcp", "server_label": "crm_server",
        "server_url": "https://example.com/mcp", "defer_loading": true}));
    assert_eq!(
        choice(vec![mcp, tool_search()], "required"),
        json!("required")
    );
}

#[test]
fn convert_tool_choice_allows_qualified_namespaced_function_tools() {
    assert_eq!(
        choice(
            vec![namespaced("crm", "CRM tools", "lookup_account")],
            "crm.lookup_account"
        ),
        json!({"type": "function", "name": "crm.lookup_account"})
    );
}

#[test]
fn convert_tool_choice_rejects_namespace_wrapper_and_bare_inner_name() {
    for c in ["lookup_account", "crm"] {
        let err = choice_err(vec![namespaced("crm", "CRM tools", "lookup_account")], c);
        assert!(err.contains("tool_namespace()"), "{c}: {err}");
    }
}

#[test]
fn convert_tool_choice_allows_top_level_function_with_namespaced_tools_present() {
    assert_eq!(
        choice(
            vec![
                tool("lookup_account"),
                namespaced("crm", "CRM tools", "lookup_account")
            ],
            "lookup_account"
        ),
        json!({"type": "function", "name": "lookup_account"})
    );
}

#[test]
fn convert_tool_choice_allows_handoff_with_namespaced_function_name_clash() {
    let mut req = request(
        vec![namespaced("crm", "CRM tools", "lookup_account")],
        Some("lookup_account"),
    );
    req.handoffs = vec![func("lookup_account")];
    assert_eq!(
        choice_of(&model("gpt-4.1"), &req).unwrap(),
        json!({"type": "function", "name": "lookup_account"})
    );
}

#[test]
fn convert_tool_choice_rejects_deferred_only_function_tools() {
    let err = choice_err(
        vec![deferred("lookup_account"), tool_search()],
        "lookup_account",
    );
    assert!(err.contains("deferred-loading function tools"), "{err}");
}

#[test]
fn convert_tool_choice_allows_visible_top_level_function_with_deferred_peer() {
    assert_eq!(
        choice(
            vec![
                tool("lookup_account"),
                deferred("lookup_account"),
                tool_search()
            ],
            "lookup_account"
        ),
        json!({"type": "function", "name": "lookup_account"})
    );
}

#[test]
fn convert_tools_prompt_managed_computer_accepts_mcp_tool_choice() {
    let mut req = request(vec![computer_tool()], None);
    req.model_settings.tool_choice = Some(ToolChoice::Mcp(MCPToolChoice {
        server_label: "crm".into(),
        name: "lookup".into(),
    }));
    req.prompt = Some(Prompt::new("pmpt_123"));
    let m = model("gpt-4").with_explicit_model(false);
    assert_eq!(
        choice_of(&m, &req).unwrap(),
        json!({"type": "mcp", "server_label": "crm", "name": "lookup"})
    );
    // With no resolved model and no GA alias, the preview payload is used.
    assert_eq!(tools_of(&m, &req)[0]["type"], "computer_use_preview");
}

#[test]
fn convert_tools_prompt_managed_computer_respects_explicit_ga_tool_choice() {
    let mut req = request(vec![computer_tool()], Some("computer"));
    req.prompt = Some(Prompt::new("pmpt_123"));
    let m = model("gpt-4").with_explicit_model(false);
    assert_eq!(tools_of(&m, &req), vec![json!({"type": "computer"})]);
}

// ------------------------------------------------------------ response format

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct OutModel {
    foo: i64,
    bar: String,
}

#[test]
fn get_response_format_plain_text_and_json_schema() {
    let m = model("gpt-4.1");
    // No schema at all: no `text` key.
    assert!(
        m.build_request_body(&ModelRequest::default(), false)
            .unwrap()
            .get("text")
            .is_none()
    );
    // Rust models plain text as `output_schema: None`; there is no plain-text schema object.

    let schema = AgentOutputSchema::new::<OutModel>().unwrap();
    let strict = schema.is_strict_json_schema();
    let structured = ModelRequest {
        output_schema: Some(Arc::new(schema)),
        ..Default::default()
    };
    let body = m.build_request_body(&structured, false).unwrap();
    let inner = &body["text"]["format"];
    assert_eq!(inner["type"], "json_schema");
    assert_eq!(inner["name"], "final_output");
    assert!(inner["schema"].is_object());
    assert_eq!(inner["strict"], json!(strict));
}

// ----------------------------------------------------------------- tool lists

#[test]
fn convert_tools_basic_types_and_includes() {
    let file_search = hosted_with(
        json!({"type": "file_search", "vector_store_ids": ["vs1"], "max_num_results": 3}),
        &["file_search_call.results"],
    );
    let web_search = hosted(json!({"type": "web_search", "filters": null,
        "user_location": null, "search_context_size": "medium"}));
    let req = request(
        vec![tool("do_thing"), file_search, web_search, computer_tool()],
        None,
    );
    let m = model("gpt-4.1");
    let tools = tools_of(&m, &req);
    assert_eq!(tools.len(), 4);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "do_thing");
    assert_eq!(tools[1]["type"], "file_search");
    assert_eq!(tools[1]["vector_store_ids"], json!(["vs1"]));
    assert_eq!(tools[1]["max_num_results"], 3);
    assert_eq!(tools[2]["type"], "web_search");
    // A GA model emits the bare GA computer tool.
    assert_eq!(tools[3], json!({"type": "computer"}));
    assert_eq!(includes_of(&m, &req), vec!["file_search_call.results"]);

    // Only one computer tool is allowed.
    let two = request(vec![computer_tool(), computer_tool()], None);
    let err = m.build_request_body(&two, false).unwrap_err().to_string();
    assert!(
        err.contains("You can only provide one computer tool. Got 2"),
        "{err}"
    );
}

#[test]
fn convert_tools_includes_web_search_content_types_and_image_settings() {
    let t = hosted_with(
        json!({"type": "web_search", "search_content_types": ["text", "image"],
            "image_settings": {"max_images": 2}}),
        &["web_search_call.results"],
    );
    let m = model("gpt-4.1");
    let req = request(vec![t], None);
    let tools = tools_of(&m, &req);
    assert_eq!(tools[0]["search_content_types"], json!(["text", "image"]));
    assert_eq!(tools[0]["image_settings"], json!({"max_images": 2}));
    assert_eq!(includes_of(&m, &req), vec!["web_search_call.results"]);
}

#[test]
fn convert_tools_text_only_content_types_adds_no_include() {
    let t = hosted(json!({"type": "web_search", "search_content_types": ["text"]}));
    let m = model("gpt-4.1");
    let req = request(vec![t], None);
    assert_eq!(
        tools_of(&m, &req)[0]["search_content_types"],
        json!(["text"])
    );
    assert!(includes_of(&m, &req).is_empty());
}

#[test]
fn convert_tools_includes_explicit_false_external_web_access() {
    let t = hosted(json!({"type": "web_search", "external_web_access": false}));
    let tools = tools_of(&model("gpt-4.1"), &request(vec![t], None));
    assert_eq!(tools[0]["external_web_access"], json!(false));
}

#[test]
fn convert_tools_uses_preview_computer_payload_for_preview_model() {
    let tools = tools_of(
        &model("computer-use-preview"),
        &request(vec![computer_tool()], None),
    );
    assert_eq!(
        tools[0],
        json!({"type": "computer_use_preview", "environment": "mac",
            "display_width": 800, "display_height": 600})
    );
}

#[test]
fn convert_tools_prompt_managed_computer_defaults_to_preview_payload() {
    let mut req = request(vec![computer_tool()], None);
    req.prompt = Some(Prompt::new("pmpt_123"));
    let tools = tools_of(&model("gpt-4").with_explicit_model(false), &req);
    assert_eq!(tools[0]["type"], "computer_use_preview");
    assert_eq!(tools[0]["environment"], "mac");
}

#[test]
fn convert_tools_shell_local_environment() {
    let t = hosted(json!({"type": "shell", "environment": {"type": "local"}}));
    let tools = tools_of(&model("gpt-4.1"), &request(vec![t], None));
    assert_eq!(tools[0]["environment"], json!({"type": "local"}));
}

#[test]
fn convert_tools_shell_container_reference_environment() {
    let t = hosted(json!({"type": "shell",
        "environment": {"type": "container", "container_id": "c1"}}));
    let tools = tools_of(&model("gpt-4.1"), &request(vec![t], None));
    assert_eq!(
        tools[0]["environment"],
        json!({"type": "container", "container_id": "c1"})
    );
}

#[test]
fn convert_tools_shell_container_auto_environment() {
    let t = hosted(json!({"type": "shell",
        "environment": {"type": "container_auto", "image": "python:3.12"}}));
    let tools = tools_of(&model("gpt-4.1"), &request(vec![t], None));
    assert_eq!(tools[0]["environment"]["type"], "container_auto");
    assert_eq!(tools[0]["environment"]["image"], "python:3.12");
}

#[test]
fn convert_tools_shell_environment_passes_through_unknown_fields() {
    let t = hosted(json!({"type": "shell",
        "environment": {"type": "container", "container_id": "c1", "future_field": 7}}));
    let tools = tools_of(&model("gpt-4.1"), &request(vec![t], None));
    assert_eq!(tools[0]["environment"]["future_field"], 7);
}

#[test]
fn convert_tools_tool_search_and_namespaces() {
    let req = request(
        vec![
            namespaced("crm", "CRM tools", "lookup_account"),
            tool("plain"),
            namespaced("crm", "CRM tools", "create_lead"),
            tool_search(),
        ],
        None,
    );
    let tools = tools_of(&model("gpt-4.1"), &req);
    assert_eq!(tools.len(), 3);
    assert_eq!(tools[0]["type"], "namespace");
    assert_eq!(tools[0]["name"], "crm");
    assert_eq!(tools[0]["description"], "CRM tools");
    let members: Vec<&str> = tools[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(members, vec!["lookup_account", "create_lead"]);
    assert_eq!(tools[1]["name"], "plain");
    assert_eq!(tools[2], json!({"type": "tool_search"}));

    // Namespace members must agree on the description.
    let clash = request(
        vec![
            namespaced("crm", "CRM tools", "a"),
            namespaced("crm", "Other", "b"),
        ],
        None,
    );
    let err = model("gpt-4.1")
        .build_request_body(&clash, false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("All tools in namespace 'crm' must share the same description."),
        "{err}"
    );
}

#[test]
fn convert_tools_top_level_deferred_function_requires_tool_search() {
    let err = model("gpt-4.1")
        .build_request_body(&request(vec![deferred("lookup")], None), false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Deferred-loading Responses tools require ToolSearchTool()"),
        "{err}"
    );
}

#[test]
fn convert_tools_rejects_tool_search_without_deferred_function() {
    let err = model("gpt-4.1")
        .build_request_body(&request(vec![tool("plain"), tool_search()], None), false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("ToolSearchTool() requires at least one searchable Responses surface"),
        "{err}"
    );
}

#[test]
fn convert_tools_allows_prompt_managed_tool_search_without_local_surface() {
    let mut req = request(vec![tool_search()], None);
    req.prompt = Some(Prompt::new("pmpt_123"));
    let tools = tools_of(&model("gpt-4").with_explicit_model(false), &req);
    assert_eq!(tools, vec![json!({"type": "tool_search"})]);
}

#[test]
fn convert_tools_rejects_duplicate_tool_search_tools() {
    let err = model("gpt-4.1")
        .build_request_body(
            &request(vec![deferred("x"), tool_search(), tool_search()], None),
            false,
        )
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Only one ToolSearchTool() is allowed when using OpenAI Responses models."),
        "{err}"
    );
}

#[test]
fn convert_tools_top_level_deferred_function_with_tool_search() {
    let tools = tools_of(
        &model("gpt-4.1"),
        &request(vec![deferred("lookup"), tool_search()], None),
    );
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "lookup");
    assert_eq!(tools[0]["defer_loading"], json!(true));
    assert_eq!(tools[1], json!({"type": "tool_search"}));
}

#[test]
fn convert_tools_preserves_tool_search_config_fields() {
    let search = hosted(json!({"type": "tool_search", "description": "find tools",
        "execution": {"type": "client"},
        "parameters": {"type": "object", "properties": {}}}));
    let tools = tools_of(
        &model("gpt-4.1"),
        &request(vec![deferred("lookup"), search], None),
    );
    assert_eq!(tools[1]["description"], "find tools");
    assert_eq!(tools[1]["execution"], json!({"type": "client"}));
    assert_eq!(tools[1]["parameters"]["type"], "object");
}

#[test]
fn convert_tools_allows_client_executed_tool_search_for_manual_flows() {
    let search = hosted(json!({"type": "tool_search", "execution": {"type": "client"}}));
    let tools = tools_of(
        &model("gpt-4.1"),
        &request(vec![namespaced("crm", "CRM tools", "lookup"), search], None),
    );
    assert_eq!(tools[0]["type"], "namespace");
    assert_eq!(tools[1]["execution"], json!({"type": "client"}));
}

#[test]
fn convert_tools_namespace_only_allows_eager_namespaces_without_tool_search() {
    let tools = tools_of(
        &model("gpt-4.1"),
        &request(vec![namespaced("crm", "CRM tools", "lookup")], None),
    );
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "namespace");
    assert_eq!(tools[0]["tools"][0]["name"], "lookup");
    // An eager namespaced tool carries no `defer_loading` flag.
    assert!(tools[0]["tools"][0].get("defer_loading").is_none());
}

#[test]
fn convert_tools_allows_tool_search_with_namespace_only_tools() {
    let tools = tools_of(
        &model("gpt-4.1"),
        &request(
            vec![namespaced("crm", "CRM tools", "lookup"), tool_search()],
            None,
        ),
    );
    assert_eq!(tools[0]["type"], "namespace");
    assert_eq!(tools[1], json!({"type": "tool_search"}));
}

#[test]
fn convert_tools_deferred_hosted_mcp_requires_tool_search() {
    let mcp = hosted(json!({"type": "mcp", "server_label": "crm",
        "server_url": "https://example.com/mcp", "defer_loading": true}));
    let err = model("gpt-4.1")
        .build_request_body(&request(vec![mcp], None), false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Deferred-loading Responses tools require ToolSearchTool()"),
        "{err}"
    );
}

#[test]
fn convert_tools_deferred_hosted_mcp_with_tool_search() {
    let mcp = hosted(json!({"type": "mcp", "server_label": "crm",
        "server_url": "https://example.com/mcp", "defer_loading": true}));
    let tools = tools_of(&model("gpt-4.1"), &request(vec![mcp, tool_search()], None));
    assert_eq!(tools[0]["type"], "mcp");
    assert_eq!(tools[0]["defer_loading"], json!(true));
    assert_eq!(tools[1], json!({"type": "tool_search"}));
}

#[test]
fn convert_tools_allows_dotted_non_function_tool_name_with_namespaced_function() {
    // A hosted tool whose name happens to look qualified does not clash with a namespace.
    let hosted_dotted = hosted(json!({"type": "mcp", "server_label": "crm.lookup"}));
    let tools = tools_of(
        &model("gpt-4.1"),
        &request(
            vec![namespaced("crm", "CRM tools", "lookup"), hosted_dotted],
            None,
        ),
    );
    assert_eq!(tools[0]["type"], "namespace");
    assert_eq!(tools[1]["server_label"], "crm.lookup");
}

#[test]
fn convert_tools_includes_handoffs() {
    let mut req = request(vec![tool("do_thing")], None);
    req.handoffs = vec![FunctionToolSpec {
        name: "transfer_to_other".into(),
        description: "Handoff".into(),
        parameters: json!({"type": "object", "properties": {}}),
        strict: true,
        ..Default::default()
    }];
    let tools = tools_of(&model("gpt-4.1"), &req);
    // Handoffs are appended after the tools, as in Python.
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], "do_thing");
    assert_eq!(tools[1]["type"], "function");
    assert_eq!(tools[1]["name"], "transfer_to_other");
    assert_eq!(tools[1]["description"], "Handoff");
}

#[test]
fn convert_tools_accepts_unresolved_computer_initializer() {
    // A GA model emits `{"type": "computer"}` without needing environment/dimensions.
    for name in ["gpt-5.4", "gpt-5.5"] {
        let bare = hosted(json!({"type": "computer"}));
        let tools = tools_of(&model(name), &request(vec![bare], None));
        assert_eq!(tools, vec![json!({"type": "computer"})]);
    }
}

#[test]
fn convert_tools_preview_tool_choice_uses_ga_payload_for_ga_model() {
    for name in ["gpt-5.4", "gpt-5.5"] {
        let m = model(name);
        let req = request(vec![computer_tool()], Some("computer_use_preview"));
        assert_eq!(tools_of(&m, &req), vec![json!({"type": "computer"})]);
        assert_eq!(choice_of(&m, &req).unwrap(), json!({"type": "computer"}));
    }
}

#[test]
fn resolve_computer_tool_model_returns_none_when_request_model_is_omitted() {
    // No computer tool: the computer model resolves to `None`, so the preview default applies
    // to nothing and a named choice still converts as a function.
    assert_eq!(
        choice(vec![tool("other")], "other"),
        json!({"type": "function", "name": "other"})
    );
    // With the model omitted and no GA alias, the computer payload stays on preview.
    let mut req = request(vec![computer_tool()], None);
    req.prompt = Some(Prompt::new("pmpt_123"));
    let tools = tools_of(
        &model("computer-use-preview").with_explicit_model(false),
        &req,
    );
    assert_eq!(tools[0]["type"], "computer_use_preview");
}

// ---------------------------------------------------------- hosted tool structs

/// Convert a hosted tool through the same path the runner uses, then into a request body.
fn hosted_tool(tool: Tool<()>) -> ToolSpec {
    let (param, includes) = tool.to_responses_param();
    ToolSpec::Hosted { param, includes }
}

#[test]
fn convert_tools_image_only_uses_default_image_settings() {
    let tool = WebSearchTool {
        search_content_types: Some(vec!["image".into()]),
        ..Default::default()
    };
    let m = model("gpt-4.1");
    let req = request(vec![hosted_tool(Tool::WebSearch(tool))], None);
    assert_eq!(includes_of(&m, &req), vec!["web_search_call.results"]);
    let tools = tools_of(&m, &req);
    assert_eq!(tools[0]["search_content_types"], json!(["image"]));
    assert!(tools[0].get("image_settings").is_none());
}

#[test]
fn web_search_filters_preserve_the_provider_payload() {
    let filters = json!({"allowed_domains": ["example.com"]});
    let tool = WebSearchTool {
        filters: Some(filters.clone()),
        ..Default::default()
    };
    let tools = tools_of(
        &model("gpt-5.4"),
        &request(vec![hosted_tool(Tool::WebSearch(tool))], None),
    );
    assert_eq!(tools[0]["filters"], filters);
    assert_eq!(tools[0]["search_context_size"], "medium");
}

#[test]
fn web_search_filters_preserve_forward_compatible_fields() {
    let filters = json!({"future_filter": ["example.com"]});
    let tool = WebSearchTool {
        filters: Some(filters.clone()),
        ..Default::default()
    };
    let tools = tools_of(
        &model("gpt-5.4"),
        &request(vec![hosted_tool(Tool::WebSearch(tool))], None),
    );
    assert_eq!(tools[0]["filters"], filters);
}

#[test]
fn convert_tools_rejects_reserved_same_name_namespace_shape() {
    let invalid = namespaced("lookup_account", "Same-name namespace", "lookup_account");
    let err = model("gpt-4.1")
        .build_request_body(&request(vec![invalid, tool_search()], None), false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("synthetic namespace `lookup_account.lookup_account`"),
        "{err}"
    );
}

#[test]
fn convert_tools_rejects_qualified_name_collision_with_dotted_top_level_tool() {
    let err = model("gpt-4.1")
        .build_request_body(
            &request(
                vec![
                    tool("crm.lookup_account"),
                    namespaced("crm", "CRM tools", "lookup_account"),
                    tool_search(),
                ],
                None,
            ),
            false,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("qualified name `crm.lookup_account`"), "{err}");
}

#[test]
fn convert_tools_rejects_duplicate_deferred_top_level_names() {
    let err = model("gpt-4.1")
        .build_request_body(
            &request(
                vec![
                    deferred("lookup_account"),
                    deferred("lookup_account"),
                    tool_search(),
                ],
                None,
            ),
            false,
        )
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("deferred top-level tool name `lookup_account`"),
        "{err}"
    );
}

#[test]
fn convert_file_search_tool_preserves_supported_result_limits() {
    for n in [1, 50] {
        let tool = FileSearchTool {
            vector_store_ids: vec!["v".into()],
            max_num_results: Some(n),
            ..Default::default()
        };
        let tools = tools_of(
            &model("gpt-4.1"),
            &request(vec![hosted_tool(Tool::FileSearch(tool))], None),
        );
        assert_eq!(tools[0]["max_num_results"], json!(n));
    }
}

#[test]
fn convert_file_search_tool_omits_provider_default_result_limit() {
    for limit in [None, Some(0)] {
        let tool = FileSearchTool {
            vector_store_ids: vec!["v".into()],
            max_num_results: limit,
            ..Default::default()
        };
        let tools = tools_of(
            &model("gpt-4.1"),
            &request(vec![hosted_tool(Tool::FileSearch(tool))], None),
        );
        assert!(tools[0].get("max_num_results").is_none(), "{limit:?}");
    }
}

#[test]
fn convert_file_search_tool_rejects_unsupported_result_limits() {
    for bad in [
        json!(-1),
        json!(51),
        json!(true),
        json!(0.0),
        json!(1.5),
        json!("3"),
    ] {
        let tool = hosted(json!({"type": "file_search", "vector_store_ids": ["v"],
            "max_num_results": bad}));
        let err = model("gpt-4.1")
            .build_request_body(&request(vec![tool], None), false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(
                "FileSearchTool max_num_results must be zero, an integer between 1 and 50, or None."
            ),
            "{bad}: {err}"
        );
    }
}
