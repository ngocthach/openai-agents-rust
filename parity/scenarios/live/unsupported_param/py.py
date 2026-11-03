"""Live L12: `temperature` on the live model: both SDKs must see the same outcome (both
succeed, or both get the same HTTP status), which documents the model's parameter rules."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner  # noqa: E402


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Assistant",
        instructions="Answer in one word.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=400, temperature=0.2),
    )
    try:
        result = await Runner.run(agent, "What colour is the sky on a clear day?")
        shape.write("unsupported_param", started, result, outcome="ok")
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status_code", None)
        shape.write("unsupported_param", started, error=e, outcome=f"error:{status}")


asyncio.run(main())
