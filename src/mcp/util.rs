//! Conversion of MCP tools into function tools. Mirrors `agents.mcp.util`.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use serde_json::{Map, Value, json};

use super::logging::{get_mcp_server_log_name, resolve_mcp_tool_description_for_model};
use super::{MCPServer, MCPTool, ToolFilterContext};
use crate::agent::MCPConfig;
use crate::error::{AgentsError, Result};
use crate::run_context::RunContext;
use crate::strict_schema::ensure_strict_json_schema;
use crate::tool::{FunctionTool, NeedsApproval, Tool, ToolOutput, ToolOutputPart};
use crate::tracing::{SpanData, mcp_tools_span};

/// Maximum length of a function-tool name derived from an MCP tool.
const TOOL_NAME_MAX_LENGTH: usize = 64;
/// Length of the hash suffix appended to disambiguate a derived tool name.
const TOOL_NAME_HASH_LENGTH: usize = 8;

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

/// Keep only ASCII alphanumerics, `_` and `-`, trimming the separators from both ends.
fn safe_tool_name_part(value: &str, fallback: &str) -> String {
    let safe = sanitize(value);
    let safe = safe.trim_matches(['_', '-']);
    if safe.is_empty() {
        fallback.to_string()
    } else {
        safe.to_string()
    }
}

/// The first [`TOOL_NAME_HASH_LENGTH`] hex characters of `seed`'s SHA-1 digest.
///
/// Only used to disambiguate derived tool names, so what matters is that it is deterministic and
/// matches Python's `hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]` byte for byte.
fn name_hash(seed: &str) -> String {
    use sha1::{Digest, Sha1};
    let digest = Sha1::digest(seed.as_bytes());
    let mut hex = String::with_capacity(TOOL_NAME_HASH_LENGTH);
    for byte in digest.iter().take(TOOL_NAME_HASH_LENGTH.div_ceil(2)) {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex.truncate(TOOL_NAME_HASH_LENGTH);
    hex
}

/// Shorten `base_name` to the tool-name length limit, appending a hash of `seed`.
///
/// A name that already fits is returned unchanged unless `force_hash` asks for the suffix,
/// which is how a collision is broken.
fn shorten_tool_name(base_name: &str, seed: &str, force_hash: bool) -> String {
    if !force_hash && base_name.len() <= TOOL_NAME_MAX_LENGTH {
        return base_name.to_string();
    }
    let suffix = format!("_{}", name_hash(seed));
    let stem_length = TOOL_NAME_MAX_LENGTH - suffix.len();
    let mut stem = base_name.to_string();
    stem.truncate(stem_length);
    while !stem.is_char_boundary(stem.len()) {
        stem.pop();
    }
    let stem = stem.trim_end_matches(['_', '-']);
    let stem = if stem.is_empty() { "mcp" } else { stem };
    format!("{stem}{suffix}")
}

/// The base public name of an MCP tool exposed with its server name: `mcp_<server>__<tool>`.
fn prefixed_tool_base_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "mcp_{}__{}",
        safe_tool_name_part(server_name, "server"),
        safe_tool_name_part(tool_name, "tool")
    )
}

/// Allocate one public function-tool name per `(server index, tool index)` pair.
///
/// Names longer than the limit, and names two servers would both produce, get a deterministic
/// hash suffix instead of colliding. `reserved` holds names already taken by the agent's own
/// tools, which are never handed out here.
fn build_prefixed_tool_name_overrides(
    batches: &[(usize, String, Vec<String>)],
    reserved: &HashSet<String>,
) -> BTreeMap<(usize, usize), String> {
    let mut base_name_counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_, server_name, tools) in batches {
        for tool in tools {
            *base_name_counts
                .entry(prefixed_tool_base_name(server_name, tool))
                .or_default() += 1;
        }
    }

    // (initial_name, seed, server_index, tool_index) -> deterministic allocation order.
    let mut candidates: Vec<(String, String, usize, usize, String)> = Vec::new();
    for (server_index, server_name, tools) in batches {
        for (tool_index, tool) in tools.iter().enumerate() {
            let base_name = prefixed_tool_base_name(server_name, tool);
            let seed = format!("{server_name}\0{tool}");
            let force_hash = base_name_counts.get(&base_name).copied().unwrap_or(0) > 1
                || reserved.contains(&base_name);
            let initial = shorten_tool_name(&base_name, &seed, force_hash);
            candidates.push((initial, seed, *server_index, tool_index, base_name));
        }
    }
    candidates.sort();

    let mut used: HashSet<String> = reserved.clone();
    let mut out = BTreeMap::new();
    for (initial, seed, server_index, tool_index, base_name) in candidates {
        let mut public_name = initial;
        let mut collision = 1;
        while used.contains(&public_name) {
            public_name = shorten_tool_name(&base_name, &format!("{seed}\0{collision}"), true);
            collision += 1;
        }
        used.insert(public_name.clone());
        out.insert((server_index, tool_index), public_name);
    }
    out
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

/// Overlay `explicit` on top of `resolved`, as `MCPUtil._merge_mcp_meta` does.
///
/// An explicit key set by the caller wins over the same key from the server's resolver.
pub fn merge_mcp_meta(
    resolved: Option<Map<String, Value>>,
    explicit: Option<Map<String, Value>>,
) -> Option<Map<String, Value>> {
    match (resolved, explicit) {
        (None, None) => None,
        (Some(m), None) | (None, Some(m)) => Some(m),
        (Some(mut resolved), Some(explicit)) => {
            resolved.extend(explicit);
            Some(resolved)
        }
    }
}

/// Call `tool` on `server` with the model's raw JSON arguments and return its output.
///
/// This is the body behind every function tool derived from an MCP server, exposed so callers
/// can drive one tool directly. `meta` is merged on top of whatever the server's
/// [`crate::mcp::MCPToolMetaResolver`] produced and sent as the request's `_meta`.
/// `tool_display_name` names the tool in error messages; it defaults to the MCP tool name.
pub async fn invoke_mcp_tool(
    server: &Arc<dyn MCPServer>,
    tool: &MCPTool,
    input_json: &str,
    meta: Option<Map<String, Value>>,
    tool_display_name: Option<&str>,
) -> Result<ToolOutput> {
    let display = tool_display_name.unwrap_or(&tool.name);
    let arguments: Option<Map<String, Value>> = if input_json.trim().is_empty() {
        None
    } else {
        match serde_json::from_str::<Value>(input_json) {
            Ok(Value::Object(m)) => Some(m),
            Ok(_) | Err(_) => {
                let message = if crate::debug::dont_log_tool_data() {
                    format!("Invalid JSON input for tool {display}")
                } else {
                    format!("Invalid JSON input for tool {display}: {input_json}")
                };
                return Err(AgentsError::model_behavior(message));
            }
        }
    };

    let resolved = server.config().tool_meta_resolver.as_ref().and_then(|r| {
        r(&crate::mcp::MCPToolMetaContext {
            server_name: server.name().to_string(),
            tool_name: tool.name.clone(),
            arguments: arguments.clone(),
        })
    });
    let merged = merge_mcp_meta(resolved, meta);

    let result = match merged {
        None => server.call_tool(&tool.name, arguments).await,
        Some(meta) => {
            server
                .call_tool_with_meta(&tool.name, arguments, Some(meta))
                .await
        }
    };
    let result = result.map_err(|e| {
        crate::logger::log_tool_action_error(
            &crate::mcp::get_mcp_server_log_message(
                &format!("Error invoking MCP tool {display} on server"),
                server.name(),
            ),
            &e,
        );
        // A misconfiguration or a malformed model call is not a transport failure, so it keeps
        // its own kind instead of being flattened into a generic invocation error.
        match e {
            AgentsError::UserError { .. } | AgentsError::ModelBehaviorError { .. } => e,
            _ => AgentsError::other(format!(
                "Error invoking MCP tool {display} on server '{}': {e}",
                get_mcp_server_log_name(server.name())
            )),
        }
    })?;
    Ok(convert_call_tool_result(
        &result,
        server.config().use_structured_content,
    ))
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
    let srv = server.clone();
    let mcp_tool = tool.clone();
    let display = exposed_name.clone();
    let mut f = FunctionTool::new_raw(
        exposed_name,
        resolve_mcp_tool_description_for_model(tool),
        schema,
        move |_ctx, args: String| {
            let srv = srv.clone();
            let mcp_tool = mcp_tool.clone();
            let display = display.clone();
            async move { invoke_mcp_tool(&srv, &mcp_tool, &args, None, Some(&display)).await }
        },
    );
    f.strict_json_schema = strict;
    f.mcp_server_name = Some(get_mcp_server_log_name(server.name()));
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
    get_all_function_tools_with_reserved(servers, config, _ctx, agent_name, &HashSet::new()).await
}

/// Like [`get_all_function_tools`], but keeping `reserved` names free.
///
/// `reserved` holds the names of tools the agent already defines, so a derived MCP tool name
/// never shadows one of them.
pub async fn get_all_function_tools_with_reserved<C: Send + Sync + 'static>(
    servers: &[Arc<dyn MCPServer>],
    config: &MCPConfig,
    _ctx: &RunContext<C>,
    agent_name: &str,
    reserved: &HashSet<String>,
) -> Result<Vec<Tool<C>>> {
    let mut out = Vec::new();
    let mut batches: Vec<(usize, String, Vec<String>)> = Vec::new();
    let mut listings: Vec<Vec<MCPTool>> = Vec::new();
    let mut names: HashSet<String> = reserved.clone();
    for (server_index, server) in servers.iter().enumerate() {
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
        batches.push((
            server_index,
            get_mcp_server_log_name(server.name()),
            tools.iter().map(|t| t.name.clone()).collect(),
        ));
        listings.push(tools);
    }

    // With the server name in the tool name, collisions and over-long names are resolved with a
    // deterministic hash suffix instead of failing the run.
    let overrides = if config.include_server_in_tool_names {
        build_prefixed_tool_name_overrides(&batches, reserved)
    } else {
        BTreeMap::new()
    };

    for (server_index, (server, tools)) in servers.iter().zip(&listings).enumerate() {
        for (tool_index, t) in tools.iter().enumerate() {
            let exposed = match overrides.get(&(server_index, tool_index)) {
                Some(name) => name.clone(),
                None => t.name.clone(),
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

    #[test]
    fn disambiguates_prefixed_tool_names() {
        // Two servers whose sanitized names collide still get distinct, bounded tool names.
        let batches = vec![
            (0, "my server".to_string(), vec!["read".to_string()]),
            (1, "my/server".to_string(), vec!["read".to_string()]),
            (
                2,
                "long".to_string(),
                vec!["x".repeat(80), "plain".to_string()],
            ),
        ];
        let reserved: HashSet<String> = ["mcp_long__plain".to_string()].into_iter().collect();
        let names = build_prefixed_tool_name_overrides(&batches, &reserved);

        // The exact names Python's `_build_prefixed_tool_name_overrides` produces for this
        // batch, hash suffixes included.
        assert_eq!(names[&(0, 0)], "mcp_my_server__read_9e4bab56");
        assert_eq!(names[&(1, 0)], "mcp_my_server__read_97f2cfe8");
        assert_eq!(
            names[&(2, 0)],
            format!("mcp_long__{}_ad4e26c9", "x".repeat(45))
        );
        assert_eq!(names[&(2, 1)], "mcp_long__plain_a2634f9a");

        let allocated: Vec<&String> = names.values().collect();
        let unique: HashSet<&String> = allocated.iter().copied().collect();
        assert_eq!(
            allocated.len(),
            unique.len(),
            "names collide: {allocated:?}"
        );
        assert!(allocated.iter().all(|n| n.len() <= TOOL_NAME_MAX_LENGTH));
        assert!(
            !unique.contains(&"mcp_long__plain".to_string()),
            "reserved name reused"
        );
        // The 80-character tool name is cut down to exactly the limit, hash suffix included.
        assert_eq!(names[&(2, 0)].len(), TOOL_NAME_MAX_LENGTH);
        // Allocation is deterministic.
        assert_eq!(
            build_prefixed_tool_name_overrides(&batches, &reserved),
            names
        );
    }

    #[test]
    fn hashes_match_python_sha1() {
        // `hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]`.
        assert_eq!(name_hash("srv\u{0}tool"), "f1a04abf");
        assert_eq!(name_hash("long\u{0}plain"), "a2634f9a");
        // A short, unique, unreserved name keeps its base form.
        assert_eq!(
            shorten_tool_name("mcp_srv__tool", "srv\u{0}tool", false),
            "mcp_srv__tool"
        );
        assert_eq!(
            shorten_tool_name("mcp_srv__tool", "srv\u{0}tool", true),
            "mcp_srv__tool_f1a04abf"
        );
    }

    #[test]
    fn sanitizes_name_parts() {
        assert_eq!(safe_tool_name_part("__", "fallback"), "fallback");
        assert_eq!(safe_tool_name_part("a b/c", "x"), "a_b_c");
        assert_eq!(prefixed_tool_base_name("srv", "tool"), "mcp_srv__tool");
    }
}
