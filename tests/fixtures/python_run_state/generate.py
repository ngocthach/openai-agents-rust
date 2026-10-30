#!/usr/bin/env python3
"""Regenerate the `RunState` fixtures this crate reads in `tests/core_python_run_state.rs`.

The fixtures are produced by the real Python SDK (`openai-agents`), driven by
`agents.testing.ScriptedModel`, so the Rust reader is checked against snapshots a Python
process actually wrote instead of against hand-built envelopes.

Usage::

    uv venv pyenv
    uv pip install --python ./pyenv/bin/python -e /path/to/openai-agents-python
    ./pyenv/bin/python tests/fixtures/python_run_state/generate.py

Every fixture is deterministic: fixed call ids, item ids, response ids, trace id and group
id, so a regeneration that changes nothing produces a byte-identical file.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
    handoff,
    input_guardrail,
    output_guardrail,
)
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call
from agents.tracing import set_trace_processors
from agents.usage import Usage

# Keep trace ids in the fixtures but never export a span: these runs are offline.
set_trace_processors([])

OUT_DIR = Path(__file__).resolve().parent

TRACE_ID = "trace_0123456789abcdef0123456789abcd"
GROUP_ID = "fixture-group"
WORKFLOW = "fixture-workflow"


def usage(requests: int = 1, input_tokens: int = 11, output_tokens: int = 7) -> Usage:
    return Usage(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def run_config() -> RunConfig:
    return RunConfig(
        workflow_name=WORKFLOW,
        trace_id=TRACE_ID,
        group_id=GROUP_ID,
        trace_metadata={"fixture": "python_run_state"},
    )


@function_tool
def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    return f"The weather in {city} is sunny."


@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """Delete a file. Always needs approval."""
    return f"deleted {path}"


@input_guardrail
async def length_guardrail(
    ctx: RunContextWrapper[Any], agent: Agent[Any], agent_input: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info={"checked": "input"}, tripwire_triggered=False
    )


@output_guardrail
async def politeness_guardrail(
    ctx: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info={"checked": "output"}, tripwire_triggered=False
    )


async def completed_run() -> dict[str, Any]:
    """A run that calls one function tool and then produces a final message."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        "get_weather",
                        {"city": "Hanoi"},
                        call_id="call_weather_1",
                        item_id="fc_weather_1",
                    )
                ],
                usage=usage(),
                response_id="resp_completed_1",
            ),
            ModelStep(
                output=[assistant_message("It is sunny in Hanoi.", item_id="msg_completed_1")],
                usage=usage(input_tokens=19, output_tokens=5),
                response_id="resp_completed_2",
            ),
        ]
    )
    agent = Agent(
        name="Weather Agent",
        instructions="Help with weather.",
        tools=[get_weather],
        model=model,
    )
    result = await Runner.run(agent, "What is the weather in Hanoi?", run_config=run_config())
    model.assert_complete()
    return result.to_state().to_json()


async def interrupted_run() -> dict[str, Any]:
    """A run stopped by a function tool that needs approval."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        "delete_file",
                        {"path": "/tmp/report.txt"},
                        call_id="call_delete_1",
                        item_id="fc_delete_1",
                    )
                ],
                usage=usage(),
                response_id="resp_interrupted_1",
            )
        ]
    )
    agent = Agent(
        name="File Agent", instructions="Manage files.", tools=[delete_file], model=model
    )
    result = await Runner.run(agent, "Delete /tmp/report.txt", run_config=run_config())
    assert result.interruptions, "expected an approval interruption"
    model.assert_complete()
    return result.to_state().to_json()


async def handoff_run() -> dict[str, Any]:
    """A run that hands off to a second agent, which produces the final output."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        "transfer_to_spanish_agent",
                        {},
                        call_id="call_handoff_1",
                        item_id="fc_handoff_1",
                    )
                ],
                usage=usage(),
                response_id="resp_handoff_1",
            ),
            ModelStep(
                output=[assistant_message("Hola, como estas?", item_id="msg_handoff_1")],
                usage=usage(input_tokens=23, output_tokens=6),
                response_id="resp_handoff_2",
            ),
        ]
    )
    spanish = Agent(name="Spanish Agent", instructions="Reply in Spanish.", model=model)
    triage = Agent(
        name="Triage Agent",
        instructions="Route the request.",
        handoffs=[handoff(spanish)],
        model=model,
    )
    result = await Runner.run(triage, "Hola", run_config=run_config())
    model.assert_complete()
    return result.to_state().to_json()


async def guardrails_run() -> dict[str, Any]:
    """A run with input and output guardrails and non-zero usage across two turns."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        "get_weather",
                        {"city": "Lisbon"},
                        call_id="call_weather_2",
                        item_id="fc_weather_2",
                    )
                ],
                usage=usage(input_tokens=31, output_tokens=9),
                response_id="resp_guardrails_1",
            ),
            ModelStep(
                output=[assistant_message("Sunny in Lisbon.", item_id="msg_guardrails_1")],
                usage=usage(input_tokens=44, output_tokens=4),
                response_id="resp_guardrails_2",
            ),
        ]
    )
    agent = Agent(
        name="Guarded Agent",
        instructions="Answer politely.",
        tools=[get_weather],
        input_guardrails=[length_guardrail],
        output_guardrails=[politeness_guardrail],
        model=model,
    )
    result = await Runner.run(agent, "Weather in Lisbon?", run_config=run_config())
    model.assert_complete()
    return result.to_state().to_json()


async def second_turn_interrupted_run() -> dict[str, Any]:
    """Turn 1 runs a tool to completion; turn 2 parks, so the turn does not start at index 0."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        "get_weather",
                        {"city": "Oslo"},
                        call_id="call_weather_3",
                        item_id="fc_weather_3",
                    )
                ],
                usage=usage(),
                response_id="resp_two_turn_1",
            ),
            ModelStep(
                output=[
                    function_call(
                        "delete_file",
                        {"path": "/tmp/oslo.txt"},
                        call_id="call_delete_2",
                        item_id="fc_delete_2",
                    )
                ],
                usage=usage(input_tokens=27, output_tokens=8),
                response_id="resp_two_turn_2",
            ),
        ]
    )
    agent = Agent(
        name="File Agent",
        instructions="Manage files.",
        tools=[get_weather, delete_file],
        model=model,
    )
    result = await Runner.run(agent, "Check Oslo then delete the file", run_config=run_config())
    assert result.interruptions, "expected an approval interruption"
    model.assert_complete()
    return result.to_state().to_json()


async def mixed_batch_run() -> dict[str, Any]:
    """One response asks for two tools: one runs, the other parks for approval."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        "get_weather",
                        {"city": "Porto"},
                        call_id="call_weather_4",
                        item_id="fc_weather_4",
                    ),
                    function_call(
                        "delete_file",
                        {"path": "/tmp/porto.txt"},
                        call_id="call_delete_3",
                        item_id="fc_delete_3",
                    ),
                ],
                usage=usage(input_tokens=17, output_tokens=12),
                response_id="resp_mixed_1",
            )
        ]
    )
    agent = Agent(
        name="File Agent",
        instructions="Manage files.",
        tools=[get_weather, delete_file],
        model=model,
    )
    result = await Runner.run(agent, "Weather in Porto, then delete it", run_config=run_config())
    assert len(result.interruptions) == 1, "exactly one call needs approval"
    model.assert_complete()
    return result.to_state().to_json()


FIXTURES = {
    "completed.json": completed_run,
    "interrupted_approval.json": interrupted_run,
    "handoff.json": handoff_run,
    "usage_guardrails.json": guardrails_run,
    "second_turn_interrupted.json": second_turn_interrupted_run,
    "mixed_batch_interrupted.json": mixed_batch_run,
}


async def main() -> None:
    for name, builder in FIXTURES.items():
        payload = await builder()
        path = OUT_DIR / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
