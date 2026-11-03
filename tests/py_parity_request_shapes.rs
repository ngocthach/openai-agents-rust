//! Request-shape details pinned by the payload-parity harness (`parity/`), where the Python SDK
//! and this crate must send the same thing: the default agent-as-tool parameter schema carries
//! Python's model docstring as its description, and a code interpreter tool adds no `include`
//! entry (only file search with results and web search with images do).

use openai_agents_sdk::tool::{CodeInterpreterTool, FileSearchTool, Tool, WebSearchTool};
use openai_agents_sdk::{Agent, FunctionTool};
use serde_json::json;

/// Python derives the default schema from its `AgentToolInput` model, whose docstring becomes
/// the schema description the model can see.
#[test]
fn the_default_agent_tool_schema_matches_python() {
    let agent = Agent::<()>::new("spanish_agent").with_instructions("Translate to Spanish.");
    let tool: FunctionTool<()> = agent.as_tool("translate_to_spanish", "Translate to Spanish.");
    let param = tool.to_responses_param();
    assert_eq!(
        param["parameters"],
        json!({
            "type": "object",
            "description": "Default input schema for agent-as-tool calls.",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "additionalProperties": false,
        })
    );
}

/// Python requests `include` entries only for file search results and web search images.
#[test]
fn only_file_and_web_search_add_include_entries() {
    let code_interpreter: Tool<()> = CodeInterpreterTool::new().into();
    assert_eq!(
        code_interpreter.to_responses_param().1,
        Vec::<String>::new()
    );

    let file_search: Tool<()> = FileSearchTool {
        vector_store_ids: vec!["vs_1".into()],
        include_search_results: true,
        ..Default::default()
    }
    .into();
    assert_eq!(
        file_search.to_responses_param().1,
        vec!["file_search_call.results".to_string()]
    );

    let plain_file_search: Tool<()> = FileSearchTool::new(vec!["vs_1".into()]).into();
    assert_eq!(
        plain_file_search.to_responses_param().1,
        Vec::<String>::new()
    );

    let web_search: Tool<()> = WebSearchTool {
        search_content_types: Some(vec!["image".into()]),
        ..Default::default()
    }
    .into();
    assert_eq!(
        web_search.to_responses_param().1,
        vec!["web_search_call.results".to_string()]
    );
}
