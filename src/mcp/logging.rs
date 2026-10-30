//! Log-safe MCP server names and tool display metadata. Mirrors `agents.mcp._logging` and
//! `agents._mcp_tool_metadata`.
//!
//! A server name is often the URL it was derived from, which may embed credentials in its
//! userinfo and secrets in its query string. [`get_mcp_server_log_name`] keeps only the parts
//! that identify the server, so a name can be logged, put in a span or embedded in an error.

use std::collections::BTreeMap;

use serde_json::Value;

use super::MCPTool;

/// Prefixes the SDK puts in front of a URL-derived server name.
const URL_DERIVED_NAME_PREFIXES: [&str; 3] = ["sse: ", "streamable_http: ", "streamable-http: "];

const INVALID_URL: &str = "<invalid-url>";

/// A URL split into the pieces `get_mcp_server_log_name` needs.
struct SplitUrl {
    scheme: String,
    netloc: String,
    path: String,
}

/// Whether `value` can be a URL scheme (`[A-Za-z][A-Za-z0-9+.-]*`).
fn scheme_chars_ok(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
}

/// Split a URL the way Python's `urllib.parse.urlsplit` does, for the fields used here.
///
/// Returns `Err(())` for the inputs `urlsplit` rejects: a netloc with unbalanced `[`/`]`.
fn split_url(url: &str) -> std::result::Result<SplitUrl, ()> {
    let mut rest = url;
    let mut scheme = String::new();
    if let Some(i) = rest.find(':')
        && i > 0
        && scheme_chars_ok(&rest[..i])
    {
        scheme = rest[..i].to_ascii_lowercase();
        rest = &rest[i + 1..];
    }

    let mut netloc = String::new();
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        netloc = after[..end].to_string();
        rest = &after[end..];
        let opens = netloc.contains('[');
        let closes = netloc.contains(']');
        if opens != closes {
            return Err(());
        }
    }

    // Drop the fragment, then the query; both are removed from the log name anyway.
    let rest = rest.split('#').next().unwrap_or("");
    let path = rest.split('?').next().unwrap_or("").to_string();

    Ok(SplitUrl {
        scheme,
        netloc,
        path,
    })
}

/// The `(hostname, port)` pair of a netloc, as Python's `SplitResult._hostinfo` computes it.
fn host_info(netloc: &str) -> (String, Option<&str>) {
    let hostinfo = match netloc.rsplit_once('@') {
        Some((_, after)) => after,
        None => netloc,
    };
    let (hostname, port) = match hostinfo.split_once('[') {
        Some((_, bracketed)) => {
            let (host, after) = bracketed.split_once(']').unwrap_or((bracketed, ""));
            (host, after.split_once(':').map(|(_, p)| p))
        }
        None => match hostinfo.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (hostinfo, None),
        },
    };
    (
        hostname.to_ascii_lowercase(),
        port.filter(|p| !p.is_empty()),
    )
}

/// Remove URL credentials, query parameters and fragments from an MCP server name.
///
/// Names that are not `http`/`https` URLs are returned unchanged. A URL the parser rejects, or
/// one whose port is not a valid number, becomes `<invalid-url>` behind its transport prefix.
///
/// ```
/// # #[cfg(feature = "mcp")] {
/// use openai_agents_sdk::mcp::get_mcp_server_log_name;
///
/// assert_eq!(
///     get_mcp_server_log_name("sse: https://user:pw@example.com/mcp?token=abc#frag"),
///     "sse: https://example.com/mcp"
/// );
/// assert_eq!(get_mcp_server_log_name("my server"), "my server");
/// # }
/// ```
pub fn get_mcp_server_log_name(name: &str) -> String {
    let prefix = URL_DERIVED_NAME_PREFIXES
        .iter()
        .find(|p| name.starts_with(**p))
        .copied()
        .unwrap_or("");
    let candidate = &name[prefix.len()..];

    let looks_like_http = {
        let lower = candidate.to_ascii_lowercase();
        lower.starts_with("http://") || lower.starts_with("https://")
    };
    let invalid = || {
        if !prefix.is_empty() || looks_like_http {
            format!("{prefix}{INVALID_URL}")
        } else {
            name.to_string()
        }
    };

    let Ok(parsed) = split_url(candidate) else {
        return invalid();
    };
    if parsed.scheme != "http" && parsed.scheme != "https" {
        return name.to_string();
    }

    let (hostname, port) = host_info(&parsed.netloc);
    if let Some(port) = port
        && !(port.chars().all(|c| c.is_ascii_digit())
            && port.parse::<u32>().is_ok_and(|p| p <= 65535))
    {
        return format!("{prefix}{INVALID_URL}");
    }
    if parsed.netloc.is_empty() || hostname.is_empty() || hostname.chars().any(char::is_whitespace)
    {
        return format!("{prefix}{INVALID_URL}");
    }

    let mut host = if hostname.contains(':') {
        format!("[{hostname}]")
    } else {
        hostname
    };
    if let Some(port) = port {
        host = format!("{host}:{port}");
    }
    let mut path = parsed.path;
    if !path.is_empty() && !path.starts_with('/') {
        path.insert(0, '/');
    }
    format!("{prefix}{}://{host}{path}", parsed.scheme)
}

/// Build a log message that names a server only when tool data may be logged.
///
/// Mirrors `get_mcp_server_log_message`: in redacted mode the server name is left out entirely.
pub fn get_mcp_server_log_message(message: &str, server_name: &str) -> String {
    if crate::debug::dont_log_tool_data() {
        message.to_string()
    } else {
        format!("{message} '{}'", get_mcp_server_log_name(server_name))
    }
}

/// Characters that cannot appear inside a URL and therefore end one inside a sentence.
fn url_terminator(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '"' | '\'' | '<' | '>' | '`' | '|' | '\\' | '^' | '{' | '}'
        )
}

/// Strip userinfo, query and fragment from one URL, whatever its scheme.
///
/// Unlike [`get_mcp_server_log_name`] this does not require `http`/`https`, because a transport
/// error can name a `ws://`, `unix://` or proxy URL that carries the same credentials.
fn redact_url(url: &str) -> String {
    let Some((scheme, after)) = url.split_once("://") else {
        return url.to_string();
    };
    let split = after
        .find(['/', '?', '#'])
        .map_or((after, ""), |i| (&after[..i], &after[i..]));
    let (netloc, rest) = split;
    let netloc = match netloc.rsplit_once('@') {
        Some((_, host)) => host,
        None => netloc,
    };
    let path = rest.split(['?', '#']).next().unwrap_or("");
    format!("{scheme}://{netloc}{path}")
}

/// Remove credentials and query secrets from every URL embedded in a message.
///
/// Python cannot rewrite the URLs inside an `httpx` exception, so it discards the whole
/// exception and substitutes a fixed string (`_credential_safe_exception_group` /
/// `_credential_safe_exception_leaf`). A Rust transport error is already a string, so the same
/// invariant — no credential ever reaches a log, a span or an error message — is met by
/// rewriting each URL in place, which keeps the diagnostic text useful.
///
/// ```
/// # #[cfg(feature = "mcp")] {
/// use openai_agents_sdk::mcp::redact_credentials;
///
/// assert_eq!(
///     redact_credentials("send failed for https://u:pw@example.com/mcp?token=abc (timeout)"),
///     "send failed for https://example.com/mcp (timeout)"
/// );
/// # }
/// ```
pub fn redact_credentials(message: &str) -> String {
    let mut out = String::with_capacity(message.len());
    let bytes = message.as_bytes();
    let mut i = 0usize;
    while i < message.len() {
        if !message.is_char_boundary(i) {
            i += 1;
            continue;
        }
        if bytes[i..].starts_with(b"://") {
            // Walk back over the scheme, which must be `[A-Za-z][A-Za-z0-9+.-]*`.
            let mut start = i;
            while start > 0 {
                let c = bytes[start - 1] as char;
                if c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.' {
                    start -= 1;
                } else {
                    break;
                }
            }
            while start < i && !(bytes[start] as char).is_ascii_alphabetic() {
                start += 1;
            }
            if start < i {
                let end = message[i..]
                    .find(url_terminator)
                    .map_or(message.len(), |o| i + o);
                let url = &message[start..end];
                out.truncate(out.len() - (i - start));
                out.push_str(&redact_url(url));
                i = end;
                continue;
            }
        }
        let ch = message[i..].chars().next().expect("char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Resolved display metadata for an MCP tool. Mirrors `agents._mcp_tool_metadata.MCPToolMetadata`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MCPToolMetadata {
    /// The tool's long-form description, when it has a non-empty one.
    pub description: Option<String>,
    /// The tool's display title, from `title` or `annotations.title`.
    pub title: Option<String>,
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value.filter(|v| !v.is_empty()).map(str::to_string)
}

/// The MCP display title, preferring an explicit `title` over `annotations.title`.
pub fn resolve_mcp_tool_title(tool: &MCPTool) -> Option<String> {
    non_empty(tool.title.as_deref()).or_else(|| {
        non_empty(
            tool.annotations
                .as_ref()
                .and_then(|a| a.get("title"))
                .and_then(Value::as_str),
        )
    })
}

/// The tool's description, when it has a non-empty one.
pub fn resolve_mcp_tool_description(tool: &MCPTool) -> Option<String> {
    non_empty(tool.description.as_deref())
}

/// The best model-facing description for an MCP tool.
///
/// MCP separates a long-form description from a short display title. When the description is
/// absent the title is used, so a title-only tool does not reach the model as a blank function.
pub fn resolve_mcp_tool_description_for_model(tool: &MCPTool) -> String {
    resolve_mcp_tool_description(tool)
        .or_else(|| resolve_mcp_tool_title(tool))
        .unwrap_or_default()
}

/// Resolve both display fields of an MCP tool at once.
pub fn extract_mcp_tool_metadata(tool: &MCPTool) -> MCPToolMetadata {
    MCPToolMetadata {
        description: resolve_mcp_tool_description(tool),
        title: resolve_mcp_tool_title(tool),
    }
}

/// [`extract_mcp_tool_metadata`] for a hosted-MCP tool, which arrives as raw JSON.
///
/// Python's resolvers take `Any` and read either a mapping key or an attribute; the Rust
/// equivalents are this `Value` form for hosted MCP and the [`MCPTool`] form for local MCP.
pub fn extract_mcp_tool_metadata_value(tool: &Value) -> MCPToolMetadata {
    let field = |key: &str| non_empty(tool.get(key).and_then(Value::as_str));
    MCPToolMetadata {
        description: field("description"),
        title: field("title").or_else(|| {
            non_empty(
                tool.get("annotations")
                    .and_then(|a| a.get("title"))
                    .and_then(Value::as_str),
            )
        }),
    }
}

/// Index the display metadata of every hosted `mcp_list_tools` payload in `items`.
///
/// Accepts raw payloads and run items whose `raw_item` holds the payload, as Python's
/// `collect_mcp_list_tools_metadata` does. The key is `(server_label, tool name)`.
pub fn collect_mcp_list_tools_metadata(
    items: &[Value],
) -> BTreeMap<(String, String), MCPToolMetadata> {
    let mut out = BTreeMap::new();
    for item in items {
        let raw = item.get("raw_item").unwrap_or(item);
        if raw.get("type").and_then(Value::as_str) != Some("mcp_list_tools") {
            continue;
        }
        let Some(server_label) = non_empty(raw.get("server_label").and_then(Value::as_str)) else {
            continue;
        };
        let Some(tools) = raw.get("tools").and_then(Value::as_array) else {
            continue;
        };
        for tool in tools {
            let Some(name) = non_empty(tool.get("name").and_then(Value::as_str)) else {
                continue;
            };
            out.insert(
                (server_label.clone(), name),
                extract_mcp_tool_metadata_value(tool),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str) -> MCPTool {
        MCPTool {
            name: name.to_string(),
            title: None,
            description: None,
            input_schema: json!({"type": "object"}),
            annotations: None,
        }
    }

    #[test]
    fn strips_credentials_query_and_fragment() {
        assert_eq!(
            get_mcp_server_log_name("https://user:secret@example.com/mcp?token=t#f"),
            "https://example.com/mcp"
        );
        assert_eq!(
            get_mcp_server_log_name("streamable_http: https://u@EXAMPLE.com:8443/a/b?x=1"),
            "streamable_http: https://example.com:8443/a/b"
        );
        assert_eq!(
            get_mcp_server_log_name("sse: http://[2001:db8::1]:9000/sse"),
            "sse: http://[2001:db8::1]:9000/sse"
        );
    }

    #[test]
    fn keeps_non_url_names() {
        assert_eq!(get_mcp_server_log_name("math server"), "math server");
        assert_eq!(get_mcp_server_log_name("stdio: python"), "stdio: python");
        assert_eq!(get_mcp_server_log_name(""), "");
    }

    #[test]
    fn reports_invalid_urls() {
        assert_eq!(
            get_mcp_server_log_name("https://exa mple.com/x"),
            "<invalid-url>"
        );
        assert_eq!(
            get_mcp_server_log_name("sse: https://example.com:port/x"),
            "sse: <invalid-url>"
        );
        assert_eq!(
            get_mcp_server_log_name("sse: https://[2001:db8::1/x"),
            "sse: <invalid-url>"
        );
    }

    #[test]
    fn redacts_urls_inside_error_text() {
        assert_eq!(
            redact_credentials("POST https://user:pw@example.com/mcp?token=abc failed"),
            "POST https://example.com/mcp failed"
        );
        // Any scheme, not just http(s): a transport error can name a websocket or proxy URL.
        assert_eq!(redact_credentials("ws://a:b@host:9/x?k=v"), "ws://host:9/x");
        // A bare fragment or an unterminated URL at the end of the message.
        assert_eq!(redact_credentials("see http://h/p#frag"), "see http://h/p");
        // Nothing that looks like a URL is left alone, including a stray `://`.
        assert_eq!(redact_credentials("no urls here"), "no urls here");
        assert_eq!(redact_credentials("weird ://x"), "weird ://x");
        // Several URLs in one message, and non-ASCII text around them.
        assert_eq!(
            redact_credentials("é https://u@a/1 and https://v:w@b/2?q=1 é"),
            "é https://a/1 and https://b/2 é"
        );
    }

    #[test]
    fn indexes_hosted_mcp_list_tools_metadata() {
        let items = vec![
            json!({"type": "other"}),
            json!({"raw_item": {"type": "mcp_list_tools", "server_label": "srv", "tools": [
                {"name": "a", "description": "Desc A"},
                {"name": "b", "annotations": {"title": "Title B"}},
                {"name": ""},
            ]}}),
            // No server label: skipped entirely.
            json!({"type": "mcp_list_tools", "tools": [{"name": "c"}]}),
        ];
        let map = collect_mcp_list_tools_metadata(&items);
        assert_eq!(map.len(), 2);
        assert_eq!(
            map[&("srv".to_string(), "a".to_string())]
                .description
                .as_deref(),
            Some("Desc A")
        );
        assert_eq!(
            map[&("srv".to_string(), "b".to_string())].title.as_deref(),
            Some("Title B")
        );
    }

    #[test]
    fn resolves_display_metadata() {
        let mut t = tool("a");
        assert_eq!(resolve_mcp_tool_description_for_model(&t), "");
        t.annotations = Some(json!({"title": "Annotated"}));
        assert_eq!(resolve_mcp_tool_description_for_model(&t), "Annotated");
        t.title = Some("Explicit".into());
        assert_eq!(resolve_mcp_tool_description_for_model(&t), "Explicit");
        t.description = Some("Long form".into());
        assert_eq!(resolve_mcp_tool_description_for_model(&t), "Long form");
        assert_eq!(
            extract_mcp_tool_metadata(&t),
            MCPToolMetadata {
                description: Some("Long form".into()),
                title: Some("Explicit".into()),
            }
        );
    }
}
