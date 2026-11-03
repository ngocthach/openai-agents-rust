"""Python side of `agents_as_tools`: an orchestrator calling a specialist agent as a tool. The
scripted call runs the inner agent (its own request) before the orchestrator's second turn.
Mirrors examples/agent_patterns/agents_as_tools.py reduced to one tool call."""

import asyncio

from agents import Agent, Runner


async def main() -> None:
    spanish = Agent(
        name="spanish_agent", instructions="You translate the user's message to Spanish"
    )
    french = Agent(
        name="french_agent", instructions="You translate the user's message to French"
    )
    orchestrator = Agent(
        name="orchestrator_agent",
        instructions=(
            "You are a translation agent. You use the tools given to you to translate. "
            "If asked for multiple translations, you call the relevant tools."
        ),
        tools=[
            spanish.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate the user's message to Spanish",
            ),
            french.as_tool(
                tool_name="translate_to_french",
                tool_description="Translate the user's message to French",
            ),
        ],
    )
    result = await Runner.run(orchestrator, "Say 'Hello, how are you?' in Spanish.")
    print(result.final_output)


asyncio.run(main())
