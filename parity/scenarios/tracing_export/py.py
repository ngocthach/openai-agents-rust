"""Python side of `tracing_export`: a tool-calling run with the trace exporter pointed at the
capture server; the `/v1/traces/ingest` payload (trace + agent/generation/function spans) is
recorded like a model request. Mirrors integration_tests/openai/test_tracing.py."""

import asyncio
import os
from typing import Annotated

from agents import Agent, Runner
from agents.decorators import tool
from agents.tracing import get_trace_provider, set_trace_processors
from agents.tracing.processors import BackendSpanExporter, BatchTraceProcessor


@tool
def get_weather(city: Annotated[str, "The city to get the weather for"]) -> str:
    """Get the current weather information for a specified city."""
    return f"Sunny in {city}"


async def main() -> None:
    endpoint = os.environ["OPENAI_BASE_URL"] + "/traces/ingest"
    set_trace_processors([BatchTraceProcessor(BackendSpanExporter(endpoint=endpoint))])
    agent = Agent(
        name="Hello world",
        instructions="You are a helpful agent.",
        model="gpt-6-luna",
        tools=[get_weather],
    )
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)
    get_trace_provider().force_flush()


asyncio.run(main())
