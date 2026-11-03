"""Python side of `chat_completions`: the same tool-calling agent through the Chat Completions
model (messages mapping, tools format, tool result message). Mirrors
integration_tests/openai/test_chat_completions.py."""

import asyncio
from typing import Annotated

from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.decorators import tool


@tool
def get_weather(city: Annotated[str, "The city to get the weather for"]) -> str:
    """Get the current weather information for a specified city."""
    return f"Sunny in {city}"


async def main() -> None:
    model = OpenAIChatCompletionsModel(model="gpt-6-luna", openai_client=AsyncOpenAI())
    agent = Agent(
        name="Hello world",
        instructions="You are a helpful agent.",
        tools=[get_weather],
        model=model,
    )
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


asyncio.run(main())
