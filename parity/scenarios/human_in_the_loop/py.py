"""Python side of `human_in_the_loop`: a tool that needs approval for Oakland pauses the run
(no second request); approving after a JSON round trip of the state resumes with the tool
output, rejecting resumes with the rejection text. Mirrors
examples/agent_patterns/human_in_the_loop.py reduced to one tool."""

import asyncio

from agents import Agent, Runner, RunState
from agents.decorators import tool


async def _needs_approval(_ctx, params, _call_id) -> bool:
    return "Oakland" in params.get("city", "")


@tool(needs_approval=_needs_approval)
async def get_temperature(city: str) -> str:
    """Get the temperature"""
    return f"The temperature in {city} is 20° Celsius"


agent = Agent(
    name="Weather assistant",
    instructions="Answer weather questions with the provided tools.",
    tools=[get_temperature],
)


async def run(approve: bool) -> None:
    result = await Runner.run(agent, "What is the temperature in Oakland?")
    while result.interruptions:
        state = await RunState.from_json(agent, result.to_state().to_json())
        for item in result.interruptions:
            if approve:
                state.approve(item)
            else:
                state.reject(item)
        result = await Runner.run(agent, state)
    print(result.final_output)


async def main() -> None:
    await run(approve=True)
    await run(approve=False)


asyncio.run(main())
