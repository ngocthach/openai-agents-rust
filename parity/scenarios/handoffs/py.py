"""Python side of `handoffs`: a triage agent with two handoffs and the recommended prompt
prefix; the scripted handoff makes the second request go to the history tutor. Mirrors the
Rust handoffs example and the README quickstart."""

import asyncio

from agents import Agent, Runner, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


async def main() -> None:
    history_tutor = Agent(
        name="History Tutor",
        handoff_description="Specialist agent for historical questions",
        instructions="You answer history questions clearly and concisely.",
    )
    math_tutor = Agent(
        name="Math Tutor",
        handoff_description="Specialist agent for math questions",
        instructions="You explain math step by step and include worked examples.",
    )
    triage = Agent(
        name="Triage Agent",
        instructions=prompt_with_handoff_instructions(
            "Route each homework question to the right specialist."
        ),
        handoffs=[handoff(history_tutor), handoff(math_tutor)],
    )
    result = await Runner.run(triage, "Who was the first president of the United States?")
    print(result.final_output)
    print(f"Answered by: {result.last_agent.name}")


asyncio.run(main())
