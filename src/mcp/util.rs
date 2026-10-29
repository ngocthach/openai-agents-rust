//! Conversion of MCP tools into function tools. Mirrors `agents.mcp.util`.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::{Map, Value, json};

use super::{MCPServer, MCPTool, ToolFilterContext};
use crate::agent::MCPConfig;
use crate::error::{AgentsError, Result};
use crate::run_context::RunContext;
use crate::strict_schema::ensure_strict_json_schema;
use crate::tool::{FunctionTool, NeedsApproval, Tool, ToolOutput, ToolOutputPart};
use crate::tracing::{SpanData, mcp_tools_span};

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Convert an MCP `CallToolResult` (as JSON) into a tool output.
pub fn convert_call_tool_result(result: &Value, use_structured_content: bool) -> ToolOutput {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if use_structured_content
        && !is_error
        && let Some(sc) = result.get("structuredContent").filter(|v| !v.is_null())
        && !(sc.is_object() && sc.as_object().map(Map::is_empty).unwrap_or(false))
    {
        return ToolOutput::Text(sc.to_string());
    }
    let content = result
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let parts: Vec<ToolOutputPart> = content
        .iter()
        .map(|c| match c.get("type").and_then(Value::as_str) {
            Some("text") => {
                ToolOutputPart::text(c.get("text").and_then(Value::as_str).unwrap_or(""))
            }
            Some("image") => {
                let data = c.get("data").and_then(Value::as_str).unwrap_or("");
                let mime = c
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                ToolOutputPart::image_url(format!("data:{mime};base64,{data}"))
            }
            _ => ToolOutputPart::text(c.to_string()),
        })
        .collect();
    match parts.as_slice() {
        [] => ToolOutput::Text(String::new()),
        [ToolOutputPart::Text { text }] => ToolOutput::Text(text.clone()),
        _ => ToolOutput::Parts(parts),
    }
}

fn prepare_schema(schema: &Value, strict: bool) -> (Value, bool) {
    let mut schema = schema.clone();
    if let Value::Object(m) = &mut schema {
        m.entry("properties").or_insert_with(|| json!({}));
        m.remove("$schema");
    }
    if strict && let Ok(s) = ensure_strict_json_schema(&schema) {
        return (s, true);
    }
    (schema, false)
}

/// Convert one MCP tool into a function tool that calls the server.
pub fn to_function_tool<C: Send + Sync + 'static>(
    server: Arc<dyn MCPServer>,
    tool: &MCPTool,
    exposed_name: String,
    convert_to_strict: bool,
    agent_name: &str,
) -> FunctionTool<C> {
    let (schema, strict) = prepare_schema(&tool.input_schema, convert_to_strict);
    let original = tool.name.clone();
    let use_structured = server.config().use_structured_content;
    let srv = server.clone();
    let mut f = FunctionTool::new_raw(
        exposed_name,
        tool.description.clone().unwrap_or_default(),
        schema,
        move |_ctx, args: String| {
            let srv = srv.clone();
            let original = original.clone();
            async move {
                let arguments: Option<Map<String, Value>> = if args.trim().is_empty() {
                    None
                } else {
                    match serde_json::from_str::<Value>(&args) {
                        Ok(Value::Object(m)) => Some(m),
                        Ok(_) | Err(_) => {
                            return Err(AgentsError::model_behavior(format!(
                                "Invalid JSON input for tool {original}: {args}"
                            )));
                        }
                    }
                };
                let result = srv.call_tool(&original, arguments).await?;
                Ok(convert_call_tool_result(&result, use_structured))
            }
        },
    );
    f.strict_json_schema = strict;
    f.mcp_server_name = Some(server.name().to_string());
    f.needs_approval =
        NeedsApproval::Static(server.config().require_approval.requires(agent_name, tool));
    f
}

/// List the (filtered) tools of every server and convert them to function tools.
pub async fn get_all_function_tools<C: Send + Sync + 'static>(
    servers: &[Arc<dyn MCPServer>],
    config: &MCPConfig,
    _ctx: &RunContext<C>,
    agent_name: &str,
) -> Result<Vec<Tool<C>>> {
    let mut out = Vec::new();
    let mut names = HashSet::new();
    for server in servers {
        let span = mcp_tools_span(server.name(), None);
        span.start();
        let listed = server.list_tools().await;
        let tools = match listed {
            Ok(t) => t,
            Err(e) => {
                span.set_error(
                    "Error fetching MCP tools",
                    Some(json!({"error": e.to_string()})),
                );
                span.finish();
                return Err(e);
            }
        };
        let fctx = ToolFilterContext {
            agent_name: agent_name.to_string(),
            server_name: server.name().to_string(),
        };
        let tools: Vec<MCPTool> = tools
            .into_iter()
            .filter(|t| server.config().tool_filter.allows(&fctx, t))
            .collect();
        span.update_data(|d| {
            if let SpanData::MCPListTools { result, .. } = d {
                *result = Some(tools.iter().map(|t| t.name.clone()).collect());
            }
        });
        span.finish();
        for t in &tools {
            let exposed = if config.include_server_in_tool_names {
                let mut n = format!("{}_{}", sanitize(server.name()), sanitize(&t.name));
                n.truncate(64);
                n
            } else {
                t.name.clone()
            };
            if !names.insert(exposed.clone()) {
                return Err(AgentsError::user(format!(
                    "Duplicate tool names found across MCP servers: {exposed}. Set include_server_in_tool_names to disambiguate."
                )));
            }
            out.push(Tool::Function(Arc::new(to_function_tool(
                server.clone(),
                t,
                exposed,
                config.convert_schemas_to_strict,
                agent_name,
            ))));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_results() {
        let r = json!({"content": [{"type": "text", "text": "hi"}]});
        assert_eq!(
            convert_call_tool_result(&r, false),
            ToolOutput::Text("hi".into())
        );
        let r = json!({"content": [{"type": "text", "text": "a"}, {"type": "image", "data": "AAA", "mimeType": "image/jpeg"}]});
        match convert_call_tool_result(&r, false) {
            ToolOutput::Parts(p) => assert_eq!(p.len(), 2),
            other => panic!("{other:?}"),
        }
        let r = json!({"content": [], "structuredContent": {"x": 1}});
        assert_eq!(
            convert_call_tool_result(&r, true),
            ToolOutput::Text("{\"x\":1}".into())
        );
    }
}
