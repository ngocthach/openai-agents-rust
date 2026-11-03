"""Python side of `sessions`: three turns through a SQLite session (each request carries the
stored history plus the new message), two turns chained with `previous_response_id`, and one
run with `conversation_id`. Mirrors examples/memory/sqlite_session_example.py and
examples/basic/previous_response_id.py."""

import asyncio
import os
from pathlib import Path

from agents import Agent, Runner, SQLiteSession


async def main() -> None:
    out = Path(os.environ["PARITY_OUT"])
    agent = Agent(name="Assistant", instructions="Reply very concisely.")
    session = SQLiteSession("conversation_123", out / "py_session.db")
    for question in [
        "What city is the Golden Gate Bridge in?",
        "What state is it in?",
        "What's the population of that state?",
    ]:
        result = await Runner.run(agent, question, session=session)
        print(f"Assistant: {result.final_output}")

    concise = Agent(name="Assistant", instructions="You are a helpful assistant. be VERY concise.")
    result = await Runner.run(concise, "What is the largest country in South America?")
    print(result.final_output)
    result = await Runner.run(
        concise, "What is the capital of that country?", previous_response_id=result.last_response_id
    )
    print(result.final_output)

    helper = Agent(name="Assistant", instructions="You are a helpful assistant.")
    result = await Runner.run(helper, "Hello", conversation_id="conv_parity_123")
    print(result.final_output)


asyncio.run(main())
