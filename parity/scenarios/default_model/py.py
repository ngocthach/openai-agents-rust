"""Python side of `default_model`: the hello-world agent with no model set, so the request
carries the SDK's default model and default settings. Mirrors examples/basic/hello_world.py."""

import asyncio

from agents import Agent, Runner


async def main() -> None:
    agent = Agent(name="Assistant", instructions="You only respond in haikus.")
    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)


asyncio.run(main())
