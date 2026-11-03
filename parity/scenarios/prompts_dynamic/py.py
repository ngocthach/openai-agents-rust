"""Python side of `prompts_dynamic`: a Responses prompt template (`prompt {id, version,
variables}`) and instructions computed at run time. Mirrors examples/basic/prompt_template.py and
dynamic_system_prompt.py."""

import asyncio

from agents import Agent, Runner


async def main() -> None:
    poet = Agent(
        name="Poet",
        prompt={"id": "pmpt_parity", "version": "1", "variables": {"poem_style": "haiku"}},
    )
    result = await Runner.run(poet, "Write a poem about recursion.")
    print(result.final_output)

    name = "Sora"

    async def custom_instructions(context, agent) -> str:
        return f"The user's name is {name}. Greet them."

    greeter = Agent(name="Greeter", instructions=custom_instructions)
    result = await Runner.run(greeter, "Hi")
    print(result.final_output)


asyncio.run(main())
