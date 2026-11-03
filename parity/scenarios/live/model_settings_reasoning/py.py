"""Live L8: reasoning effort and verbosity settings must be accepted by the live model; a
reasoning item is recorded when the API returns one."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from openai.types.shared import Reasoning  # noqa: E402

from agents import Agent, ModelSettings, Runner  # noqa: E402


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Assistant",
        instructions="Answer in one short sentence.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(
            max_tokens=2000, reasoning=Reasoning(effort="low", summary="auto"), verbosity="low"
        ),
    )
    try:
        result = await Runner.run(agent, "Why is the sky blue?")
        shape.write(
            "model_settings_reasoning",
            started,
            result,
            has_reasoning_item="reasoning_item" in [i.type for i in result.new_items],
        )
    except Exception as e:  # noqa: BLE001
        shape.write("model_settings_reasoning", started, error=e)


asyncio.run(main())
