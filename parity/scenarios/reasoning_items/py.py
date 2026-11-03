"""Python side of `reasoning_items`: the model returns a reasoning item before a tool call; the
second request must send that reasoning item back along with the call and its output.
Mirrors examples/reasoning_content/runner_example.py."""

import asyncio
from typing import Annotated

from openai.types.shared import Reasoning

from agents import Agent, ModelSettings, Runner
from agents.decorators import tool


@tool
def get_weather(city: Annotated[str, "The city to get the weather for"]) -> str:
    """Get the current weather information for a specified city."""
    return f"Sunny in {city}"


async def main() -> None:
    agent = Agent(
        name="Hello world",
        instructions="You are a helpful agent.",
        model="gpt-6-luna",
        model_settings=ModelSettings(reasoning=Reasoning(effort="low", summary="auto")),
        tools=[get_weather],
    )
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


asyncio.run(main())
