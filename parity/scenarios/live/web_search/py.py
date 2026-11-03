"""Live L10: one hosted web search; the answer must carry a citation annotation."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner, WebSearchTool  # noqa: E402


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Researcher",
        instructions="Use web search to answer, then reply in one sentence with a citation.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=600),
        tools=[WebSearchTool(search_context_size="low")],
    )
    try:
        result = await Runner.run(agent, "What is the current stable version of the Rust compiler?")
        shape.write(
            "web_search",
            started,
            result,
            search_calls=[i.type for i in result.new_items].count("tool_call_item"),
        )
    except Exception as e:  # noqa: BLE001
        shape.write("web_search", started, error=e)


asyncio.run(main())
