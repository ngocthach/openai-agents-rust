"""Live L5: a streamed run with one tool call; the item-level event sequence is compared, the
raw API event sequence only has to start and end correctly."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner  # noqa: E402
from agents.decorators import tool  # noqa: E402


@tool
def how_many_jokes() -> int:
    """Return the number of jokes to tell."""
    return 2


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Joker",
        instructions="First call the `how_many_jokes` tool, then tell that many very short jokes.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=2000),
        tools=[how_many_jokes],
    )
    events: list[str] = []
    try:
        result = Runner.run_streamed(agent, "Hello")
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                events.append(f"raw:{event.data.type}")
            elif event.type == "run_item_stream_event":
                events.append(f"item:{event.name}")
            elif event.type == "agent_updated_stream_event":
                events.append(f"agent_updated:{event.new_agent.name}")
        shape.write(
            "streaming",
            started,
            result,
            stream_event_types=events,
            stream_item_events=[e for e in events if not e.startswith("raw:")],
            stream_item_events_stable=[
                e for e in events if not e.startswith("raw:") and e != "item:reasoning_item_created"
            ],
        )
    except Exception as e:  # noqa: BLE001
        shape.write("streaming", started, error=e, stream_event_types=events)


asyncio.run(main())
