"""Live L9: an orchestrator that must call one specialist agent as a tool (inner run), then
answer: three requests in all."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner  # noqa: E402


async def main() -> None:
    started = shape.start()
    settings = ModelSettings(max_tokens=2000)
    model = os.environ["PARITY_MODEL"]
    spanish = Agent(
        name="spanish_agent",
        instructions="You translate the user's message to Spanish. Reply with the translation only.",
        model=model,
        model_settings=settings,
    )
    orchestrator = Agent(
        name="orchestrator_agent",
        instructions=(
            "You are a translation agent. You never translate yourself: always call the "
            "translate_to_spanish tool, then reply with its result only."
        ),
        model=model,
        model_settings=settings,
        tools=[
            spanish.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate the user's message to Spanish",
            )
        ],
    )
    try:
        result = await Runner.run(orchestrator, "Translate to Spanish: Good morning, friend.")
        shape.write("agents_as_tools", started, result)
    except Exception as e:  # noqa: BLE001
        shape.write("agents_as_tools", started, error=e)


asyncio.run(main())
