"""Live L2: a function tool the model must call (the prompt pins the city), then answer."""

import asyncio
import os
import sys
from typing import Annotated

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner  # noqa: E402
from agents.decorators import tool  # noqa: E402


@tool
def get_weather(city: Annotated[str, "The city to get the weather for"]) -> str:
    """Get the current weather information for a specified city."""
    return f"Sunny in {city}, 14-20C, light wind."


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Weather agent",
        instructions="Always use the get_weather tool to answer weather questions. Be concise.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=400),
        tools=[get_weather],
    )
    try:
        result = await Runner.run(agent, "What's the weather in Tokyo?")
        shape.write("tools_two_turn", started, result)
    except Exception as e:  # noqa: BLE001
        shape.write("tools_two_turn", started, error=e)


asyncio.run(main())
