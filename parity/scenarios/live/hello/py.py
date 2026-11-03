"""Live L1: the hello-world agent on the live model."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner  # noqa: E402


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=2000),
    )
    try:
        result = await Runner.run(agent, "Tell me about recursion in programming.")
        shape.write("hello", started, result)
    except Exception as e:  # noqa: BLE001
        shape.write("hello", started, error=e)


asyncio.run(main())
