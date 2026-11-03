"""Python side of `streaming`: a streamed run with one tool call. The request bodies carry
`stream: true`; the sequence of events the SDK emits from the same SSE is written to
`$PARITY_OUT/py.events.json` in the shared vocabulary (`raw:<api event type>`,
`item:<run item event name>`, `agent_updated:<agent name>`). Mirrors
examples/basic/stream_items.py."""

import asyncio
import json
import os
from pathlib import Path

from agents import Agent, Runner
from agents.decorators import tool


@tool
def how_many_jokes() -> int:
    """Return a number of jokes."""
    return 3


async def main() -> None:
    agent = Agent(
        name="Joker",
        instructions="First call the `how_many_jokes` tool, then tell that many jokes.",
        tools=[how_many_jokes],
    )
    result = Runner.run_streamed(agent, input="Hello")
    events: list[str] = []
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            events.append(f"raw:{event.data.type}")
        elif event.type == "run_item_stream_event":
            events.append(f"item:{event.name}")
        elif event.type == "agent_updated_stream_event":
            events.append(f"agent_updated:{event.new_agent.name}")
    print(result.final_output)
    Path(os.environ["PARITY_OUT"], "py.events.json").write_text(json.dumps(events, indent=1))


asyncio.run(main())
