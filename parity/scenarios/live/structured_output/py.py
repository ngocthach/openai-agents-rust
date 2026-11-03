"""Live L3: a typed output the model must fill."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from agents import Agent, ModelSettings, Runner  # noqa: E402


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Calendar extractor",
        instructions="Extract calendar events from text",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=400),
        output_type=CalendarEvent,
    )
    try:
        result = await Runner.run(agent, "Alice and Bob are going to a science fair on Friday.")
        event = result.final_output_as(CalendarEvent)
        shape.write(
            "structured_output",
            started,
            result,
            participants_sorted=sorted(p.lower() for p in event.participants),
        )
    except Exception as e:  # noqa: BLE001
        shape.write("structured_output", started, error=e)


asyncio.run(main())
