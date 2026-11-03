"""Python side of `explicit_gpt6_luna`: the hello-world agent with `model="gpt-6-luna"`, a name
no default-settings pattern matches, so the request must carry no implicit reasoning or
verbosity settings."""

import asyncio

from agents import Agent, Runner


async def main() -> None:
    agent = Agent(
        name="Assistant", instructions="You only respond in haikus.", model="gpt-6-luna"
    )
    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)


asyncio.run(main())
