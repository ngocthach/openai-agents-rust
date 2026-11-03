"""Live L6: a tool that needs approval pauses the run; approving after a JSON round trip of
the state resumes it and the tool runs once."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner, RunState  # noqa: E402
from agents.decorators import tool  # noqa: E402

RUNS = []


@tool(needs_approval=True)
async def get_temperature(city: str) -> str:
    """Get the temperature for a city."""
    RUNS.append(city)
    return f"The temperature in {city} is 20° Celsius"


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Weather assistant",
        instructions="Always use the get_temperature tool to answer temperature questions. Be concise.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=400),
        tools=[get_temperature],
    )
    interruptions: list[int] = []
    try:
        result = await Runner.run(agent, "What is the temperature in Oakland?")
        interruptions.append(len(result.interruptions))
        while result.interruptions:
            state = await RunState.from_json(agent, result.to_state().to_json())
            for item in result.interruptions:
                state.approve(item)
            result = await Runner.run(agent, state)
            interruptions.append(len(result.interruptions))
        shape.write("approval_resume", started, result, interruptions=interruptions, tool_runs=len(RUNS))
    except Exception as e:  # noqa: BLE001
        shape.write("approval_resume", started, error=e, interruptions=interruptions)


asyncio.run(main())
