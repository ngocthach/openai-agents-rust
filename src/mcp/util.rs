//! Conversion of MCP tools into function tools. Mirrors `agents.mcp.util`.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use serde_json::{Map, Value, json};

use super::logging::{
    get_mcp_server_log_name, resolve_mcp_tool_description_for_model, resolve_mcp_tool_title,
};
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

/// Quote `value` the way Python's `repr()` quotes a `str`.
///
/// Duplicate tool names are reported back to the user, and a tool name comes from the server,
/// so it may carry control characters that would forge log lines or terminal escapes. Python
/// escapes them because it formats the names with `repr`; this reproduces that escaping for the
/// characters a name can realistically contain: backslash, the quote in use, ASCII and C1
/// control characters, and the format/line/paragraph separators that are invisible when printed.
fn python_repr(value: &str) -> String {
    // Python prefers single quotes and switches to double quotes only when that avoids escaping.
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_python_unprintable(c) => {
                let code = c as u32;
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x10000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Whether `str.isprintable()` would reject `c`, for the ranges a tool name can contain.
///
/// Covers Cc (ASCII and C1 controls), the Cf format characters and the Zl/Zp separators. Other
/// unassigned or private-use code points are treated as printable; they are escaped by Python
/// but cannot forge a log line, so the difference is cosmetic.
fn is_python_unprintable(c: char) -> bool {
    let code = c as u32;
    matches!(code,
        0x00..=0x1f
        | 0x7f..=0x9f
        | 0x00ad
        | 0x0600..=0x0605
        | 0x061c
        | 0x06dd
        | 0x070f
        | 0x180e
        | 0x200b..=0x200f
        | 0x2028..=0x202e
        | 0x2060..=0x2064
        | 0x2066..=0x206f
        | 0xfeff
        | 0xfff9..=0xfffb)
}

/// The `UserError` Python raises when two MCP servers expose the same tool name.
///
/// `names` are the colliding names; `prefixed` says whether tool names already carry their
/// server name, in which case suggesting the same option again would be useless, so Python
/// leaves the hint out.
fn duplicate_tool_names_error(names: &[String], prefixed: bool) -> AgentsError {
    let quoted: Vec<String> = names.iter().map(|n| python_repr(n)).collect();
    let listed = quoted.join(", ");
    if prefixed {
        AgentsError::user(format!(
            "Duplicate tool names found across MCP servers: {listed}"
        ))
    } else {
        AgentsError::user(format!(
            "Duplicate tool names found across MCP servers: {listed}. \
             Pass `include_server_in_tool_names=True` to \
             `MCPUtil.get_all_function_tools()` or set \
             `mcp_config={{'include_server_in_tool_names': True}}` on the \
             agent to prefix tool names with their server name and avoid \
             collisions."
        ))
    }
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
        // Python returns the empty list for a result with no content blocks, which reaches the
        // model as `[]` rather than as an empty string.
        [] => ToolOutput::Json(json!([])),
        [ToolOutputPart::Text { text }] => ToolOutput::Text(text.clone()),
        _ => ToolOutput::Parts(parts),
    }
}

fn prepare_schema(schema: &Value, strict: bool) -> (Value, bool) {
    let mut schema = schema.clone();
    if let Value::Object(m) = &mut schema {
        // The MCP spec does not require `properties`, but the OpenAI function schema does.
        // Everything else the server declared — `$schema` included — is passed through, as
        // `MCPUtil.to_function_tool` does.
        m.entry("properties").or_insert_with(|| json!({}));
    }
    if strict && let Ok(s) = ensure_strict_json_schema(&schema) {
        return (s, true);
    }
    (schema, false)
}

/// Deepest nesting a tool input schema may have, matching Python's `_MAX_SCHEMA_DEPTH`.
const MAX_SCHEMA_DEPTH: usize = 100;

/// Reject a schema nested deeper than [`MAX_SCHEMA_DEPTH`] before anything recurses into it.
///
/// Python's `_copy_json_schema` raises the same `UserError` for strict and non-strict conversion
/// alike. The walk is iterative so a hostile schema cannot overflow the stack here.
fn check_schema_depth(schema: &Value) -> Result<()> {
    let mut stack: Vec<(&Value, usize)> = vec![(schema, 1)];
    while let Some((value, depth)) = stack.pop() {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(AgentsError::user(
                "JSON schema is too deeply nested to process safely. Simplify or flatten the schema.",
            ));
        }
        match value {
            Value::Object(map) => stack.extend(map.values().map(|v| (v, depth + 1))),
            Value::Array(items) => stack.extend(items.iter().map(|v| (v, depth + 1))),
            _ => {}
        }
    }
    Ok(())
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
            Ok(_) => {
                return Err(AgentsError::model_behavior(format!(
                    "Invalid JSON input for tool {display}: expected a JSON object"
                )));
            }
            Err(_) => {
                // The payload may hold secrets, so it reaches the log and the error message
                // only when tool-data logging is allowed.
                let message = if crate::debug::dont_log_tool_data() {
                    log::debug!("Invalid JSON input for MCP tool");
                    format!("Invalid JSON input for tool {display}")
                } else {
                    let message = format!("Invalid JSON input for tool {display}: {input_json}");
                    log::debug!("{message}");
                    message
                };
                return Err(AgentsError::model_behavior(message));
            }
        }
    };
    if crate::debug::dont_log_tool_data() {
        log::debug!("Invoking MCP tool");
    } else {
        log::debug!("Invoking MCP tool {display} with input {input_json}");
    }

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
        // Redacted mode names neither the tool nor the server, as Python's does.
        let log_message = if crate::debug::dont_log_tool_data() {
            "Error invoking MCP tool".to_string()
        } else {
            crate::mcp::get_mcp_server_log_message(
                &format!("Error invoking MCP tool {display} on server"),
                server.name(),
            )
        };
        crate::logger::log_tool_action_error(&log_message, &e);
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
pub async fn to_function_tool<C: Send + Sync + 'static>(
    server: Arc<dyn MCPServer>,
    tool: &MCPTool,
    exposed_name: String,
    convert_to_strict: bool,
    agent_name: &str,
) -> FunctionTool<C> {
    to_function_tool_with_failure(
        server,
        tool,
        exposed_name,
        convert_to_strict,
        agent_name,
        Some(crate::tool::shared_default_tool_error_function::<C>()),
    )
    .await
}

/// [`to_function_tool`] with the agent's MCP failure formatter.
///
/// `failure_error_function` is the agent's `MCPConfig::failure_error_function`. The server's
/// own [`crate::mcp::MCPToolPolicy`] wins when it set one explicitly, mirroring Python's
/// `_get_failure_error_function` and its `_UNSET` sentinel; `None` re-raises MCP tool errors
/// instead of reporting them to the model.
pub async fn to_function_tool_with_failure<C: Send + Sync + 'static>(
    server: Arc<dyn MCPServer>,
    tool: &MCPTool,
    exposed_name: String,
    convert_to_strict: bool,
    agent_name: &str,
    failure_error_function: Option<crate::tool::ToolErrorFn<C>>,
) -> FunctionTool<C> {
    let (schema, strict) = prepare_schema(&tool.input_schema, convert_to_strict);
    let prepared_schema = schema.clone();
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
            async move {
                // A fresh copy of the static `_meta` per call, as Python's partial passes it.
                let static_meta = mcp_tool.meta.clone();
                invoke_mcp_tool(&srv, &mcp_tool, &args, static_meta, Some(&display)).await
            }
        },
    );
    // `FunctionTool::new_raw` strictifies whatever it is given; the MCP tool's schema must
    // reach the model exactly as `prepare_schema` decided, strict or not.
    f.params_json_schema = prepared_schema;
    f.strict_json_schema = strict;
    f.mcp_title = resolve_mcp_tool_title(tool);
    f.mcp_server_name = Some(get_mcp_server_log_name(server.name()));
    f.needs_approval = NeedsApproval::Static(
        server
            .config()
            .require_approval
            .requires(agent_name, tool)
            .await,
    );

    let policy = server.config().tool_policy::<C>();
    let failure = match policy {
        Some(p) if p.failure_error_function_set => p.failure_error_function.clone(),
        _ => failure_error_function,
    };
    f = match failure {
        Some(handler) => {
            // The default policy keeps the error detail out of traces.
            let is_default = crate::tool::is_shared_default_tool_error_function(&handler);
            let mut f = f.with_failure_error_function(move |ctx, e| handler(ctx, e));
            f.default_failure_error_function = is_default;
            f
        }
        None => f.without_failure_error_function(),
    };
    if let Some(p) = policy {
        f.tool_input_guardrails = p.tool_input_guardrails.clone();
        f.tool_output_guardrails = p.tool_output_guardrails.clone();
        if let Some(extractor) = &p.custom_data_extractor {
            f.custom_data_extractor = Some(extractor.clone());
        }
    }
    f
}

/// List the (filtered) tools of every server and convert them to function tools.
pub async fn get_all_function_tools<C: Send + Sync + 'static>(
    servers: &[Arc<dyn MCPServer>],
    config: &MCPConfig<C>,
    ctx: &RunContext<C>,
    agent_name: &str,
) -> Result<Vec<Tool<C>>> {
    get_all_function_tools_with_reserved(servers, config, ctx, agent_name, &HashSet::new()).await
}

/// Like [`get_all_function_tools`], but keeping `reserved` names free.
///
/// `reserved` holds the names of tools the agent already defines, so a derived MCP tool name
/// never shadows one of them.
pub async fn get_all_function_tools_with_reserved<C: Send + Sync + 'static>(
    servers: &[Arc<dyn MCPServer>],
    config: &MCPConfig<C>,
    ctx: &RunContext<C>,
    agent_name: &str,
    reserved: &HashSet<String>,
) -> Result<Vec<Tool<C>>> {
    get_all_function_tools_for_agent(servers, config, ctx, agent_name, None, reserved).await
}

/// [`get_all_function_tools_with_reserved`] with the agent itself, so a dynamic
/// [`crate::mcp::ToolFilter`] sees `ToolFilterContext::agent` as Python's does.
pub async fn get_all_function_tools_for_agent<C: Send + Sync + 'static>(
    servers: &[Arc<dyn MCPServer>],
    config: &MCPConfig<C>,
    ctx: &RunContext<C>,
    agent_name: &str,
    agent: Option<Arc<crate::agent::Agent<C>>>,
    reserved: &HashSet<String>,
) -> Result<Vec<Tool<C>>> {
    let run_context_any: Arc<dyn std::any::Any + Send + Sync> = Arc::new(ctx.clone());
    let agent_any: Option<Arc<dyn std::any::Any + Send + Sync>> =
        agent.map(|a| Arc::new(a) as Arc<dyn std::any::Any + Send + Sync>);
    let mut out = Vec::new();
    let mut batches: Vec<(usize, String, Vec<String>)> = Vec::new();
    let mut listings: Vec<Vec<MCPTool>> = Vec::new();
    let mut names: HashSet<String> = reserved.clone();
    for (server_index, server) in servers.iter().enumerate() {
        // The span is exported, so it carries the credential-free name, never the raw URL.
        let span = mcp_tools_span(&get_mcp_server_log_name(server.name()), None);
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
            run_context_any: Some(run_context_any.clone()),
            agent_any: agent_any.clone(),
        };
        let mut kept: Vec<MCPTool> = Vec::with_capacity(tools.len());
        for t in tools {
            if server.config().tool_filter.allows(&fctx, &t).await {
                kept.push(t);
            }
        }
        let tools = kept;
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
        // Python compares one server's whole name set against the names seen so far and reports
        // every collision at once, sorted, before adding that server's tools.
        let exposed_names: Vec<String> = tools
            .iter()
            .enumerate()
            .map(
                |(tool_index, t)| match overrides.get(&(server_index, tool_index)) {
                    Some(name) => name.clone(),
                    None => t.name.clone(),
                },
            )
            .collect();
        let mut duplicates: Vec<String> = exposed_names
            .iter()
            .filter(|n| names.contains(*n))
            .cloned()
            .collect();
        duplicates.sort();
        duplicates.dedup();
        if !duplicates.is_empty() {
            return Err(duplicate_tool_names_error(
                &duplicates,
                config.include_server_in_tool_names,
            ));
        }
        names.extend(exposed_names.iter().cloned());

        for (tool_index, t) in tools.iter().enumerate() {
            check_schema_depth(&t.input_schema)?;
            let exposed = exposed_names[tool_index].clone();
            out.push(Tool::Function(Arc::new(
                to_function_tool_with_failure(
                    server.clone(),
                    t,
                    exposed,
                    config.convert_schemas_to_strict,
                    agent_name,
                    config.failure_error_function.clone(),
                )
                .await,
            )));
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
