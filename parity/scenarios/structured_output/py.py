"""Python side of `structured_output`: a pydantic output type becomes a strict `text.format`
JSON schema. Mirrors the Rust structured_output example and the README quickstart."""

import asyncio

from pydantic import BaseModel

from agents import Agent, Runner


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


async def main() -> None:
    agent = Agent(
        name="Calendar extractor",
        instructions="Extract calendar events from text",
        output_type=CalendarEvent,
    )
    result = await Runner.run(agent, "Alice and Bob are going to a science fair on Friday.")
    event = result.final_output_as(CalendarEvent)
    print(f"{event.name} on {event.date} with {', '.join(event.participants)}")


asyncio.run(main())
