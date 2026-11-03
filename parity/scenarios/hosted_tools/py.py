"""Python side of `hosted_tools`: every hosted tool config both SDKs expose, as pure request
payload (web search with location and filters, file search with results included, code
interpreter, image generation, hosted MCP). Mirrors examples/tools/*.py."""

import asyncio

from agents import (
    Agent,
    CodeInterpreterTool,
    FileSearchTool,
    HostedMCPTool,
    ImageGenerationTool,
    Runner,
    WebSearchTool,
)


async def main() -> None:
    agent = Agent(
        name="Hosted tools",
        instructions="Use the hosted tools when they help.",
        model="gpt-6-luna",
        tools=[
            WebSearchTool(
                user_location={"type": "approximate", "city": "Tokyo"},
                filters={"allowed_domains": ["openai.com"]},
                search_context_size="low",
            ),
            FileSearchTool(
                vector_store_ids=["vs_parity"], max_num_results=3, include_search_results=True
            ),
            CodeInterpreterTool(
                tool_config={"type": "code_interpreter", "container": {"type": "auto"}}
            ),
            ImageGenerationTool(tool_config={"type": "image_generation"}),
            HostedMCPTool(
                tool_config={
                    "type": "mcp",
                    "server_label": "deepwiki",
                    "server_url": "https://mcp.deepwiki.com/mcp",
                    "require_approval": "never",
                }
            ),
        ],
    )
    result = await Runner.run(agent, "What's new in the openai-agents SDK?")
    print(result.final_output)


asyncio.run(main())
