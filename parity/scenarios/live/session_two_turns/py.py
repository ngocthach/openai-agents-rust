"""Live L7: two turns through a SQLite session; the second answer must use the first turn's
fact, and the second run sends one request whose input carries the stored history."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from agents import Agent, ModelSettings, Runner, SQLiteSession  # noqa: E402


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
        model=os.environ["PARITY_MODEL"],
        model_settings=ModelSettings(max_tokens=2000),
    )
    session = SQLiteSession("live_123", Path(os.environ["PARITY_OUT"]) / "py_session.db")
    try:
        await Runner.run(agent, "My favourite colour is turquoise. Just acknowledge it.", session=session)
        result = await Runner.run(agent, "What is my favourite colour? Answer with the colour only.", session=session)
        # Reasoning items come and go with the model's mood; count the conversation without them.
        stored = await session.get_items()
        shape.write(
            "session_two_turns",
            started,
            result,
            mentions_fact="turquoise" in result.final_output.lower(),
            session_items=len(stored),
            session_items_stable=len([i for i in stored if i.get("type") != "reasoning"]),
            input_list_len_stable=len(
                [i for i in result.to_input_list() if i.get("type") != "reasoning"]
            ),
        )
    except Exception as e:  # noqa: BLE001
        shape.write("session_two_turns", started, error=e)


asyncio.run(main())
