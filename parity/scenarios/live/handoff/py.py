"""Live L4: a triage agent that must hand off a Spanish message to the Spanish assistant."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner, handoff  # noqa: E402
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions  # noqa: E402


async def main() -> None:
    started = shape.start()
    settings = ModelSettings(max_tokens=400)
    model = os.environ["PARITY_MODEL"]
    spanish = Agent(
        name="Spanish Assistant",
        handoff_description="A Spanish-speaking assistant for messages written in Spanish.",
        instructions="You only speak Spanish and are extremely concise.",
        model=model,
        model_settings=settings,
    )
    english = Agent(
        name="English Assistant",
        handoff_description="An English-speaking assistant for messages written in English.",
        instructions="You only speak English and are extremely concise.",
        model=model,
        model_settings=settings,
    )
    triage = Agent(
        name="Triage",
        instructions=prompt_with_handoff_instructions(
            "Hand off every message to the assistant that speaks its language. Never answer yourself."
        ),
        model=model,
        model_settings=settings,
        handoffs=[handoff(spanish), handoff(english)],
    )
    try:
        result = await Runner.run(triage, "Hola, ¿cómo estás? Responde en una frase.")
        shape.write("handoff", started, result)
    except Exception as e:  # noqa: BLE001
        shape.write("handoff", started, error=e)


asyncio.run(main())
