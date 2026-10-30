#!/usr/bin/env python3
"""Resume Rust-written `RunState` snapshots with the Python SDK.

This is the reverse direction of `generate.py`: the files under `rust/` are written by the
Rust test `core_python_run_state::write_rust_states_for_the_python_consumer` (run it with
`OARS_RUST_STATE_OUT=<this dir>/rust`), and this script loads each one with
`RunState.from_json` and finishes the run with `agents.testing.ScriptedModel`.

The Rust test suite does not need Python; this script is the interop check that the envelope
Rust writes for an *interrupted* run is actually resumable by the Python runner, which needs
`last_processed_response` to continue the parked turn.

Usage::

    ./pyenv/bin/python tests/fixtures/python_run_state/resume_rust_state.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agents import Agent, Runner, function_tool, handoff
from agents.exceptions import UserError
from agents.run_state import RunState
from agents.testing import ModelStep, ScriptedModel, assistant_message
from agents.tracing import set_trace_processors

set_trace_processors([])

RUST_DIR = Path(__file__).resolve().parent / "rust"

executed: list[str] = []
weather_calls: list[str] = []


@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """Delete a file."""
    executed.append(path)
    return f"deleted {path}"


@function_tool
def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    weather_calls.append(city)
    return f"The weather in {city} is sunny."


def load(name: str) -> dict[str, Any]:
    return json.loads((RUST_DIR / name).read_text())


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def resume_interrupted_approval() -> None:
    """Approve the parked call and assert the Python runner actually ran the tool."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[assistant_message("Deleted the report.", item_id="msg_rust_1")],
                response_id="resp_rust_resume_1",
            )
        ]
    )
    agent = Agent(
        name="File Agent", instructions="Manage files.", tools=[delete_file], model=model
    )
    state = await RunState.from_json(agent, load("rust_interrupted_approval.json"))

    interruptions = state.get_interruptions()
    check(len(interruptions) == 1, f"expected 1 interruption, got {len(interruptions)}")
    check(
        interruptions[0].raw_item.call_id == "call_rust_delete_1",
        f"unexpected call id {interruptions[0].raw_item.call_id!r}",
    )
    state.approve(interruptions[0])

    executed.clear()
    result = await Runner.run(agent, state)
    model.assert_complete()

    # Python silently skips a processed-response entry whose tool does not resolve, so the
    # only meaningful assertion is that the approved tool really executed.
    check(executed == ["/tmp/report.txt"], f"tool did not run: {executed!r}")
    outputs = [
        item.output
        for item in result.new_items
        if getattr(item, "type", None) == "tool_call_output_item"
    ]
    check(
        "deleted /tmp/report.txt" in [str(o) for o in outputs],
        f"missing tool output, got {outputs!r}",
    )
    check(
        result.final_output == "Deleted the report.",
        f"unexpected final output {result.final_output!r}",
    )
    check(not result.interruptions, "run is still interrupted")
    print("ok  rust_interrupted_approval.json -> tool executed, run finished")


async def load_completed() -> None:
    """A finished Rust run reads back whole; Python treats a finished snapshot as terminal.

    Rust lets `add_input` continue a snapshot that produced a final output, Python does not
    (`current_step` is cleared when the run ends), so this direction only checks the read.
    """
    agent = Agent(name="Weather Agent", instructions="Help with weather.")
    state = await RunState.from_json(agent, load("rust_completed.json"))
    items = state._generated_items
    check(len(items) == 1, f"expected 1 generated item, got {len(items)}")
    check(
        items[0].raw_item.content[0].text == "It is sunny in Hanoi.",
        "assistant message did not survive the read",
    )
    check(state._current_turn == 1, f"unexpected turn {state._current_turn}")
    try:
        state.add_input("And tomorrow?")
    except UserError:
        pass
    else:
        raise AssertionError("Python accepted input on a terminal state")
    print("ok  rust_completed.json -> read back whole, terminal as Python expects")


async def load_handoff() -> None:
    """A finished Rust handoff run reads back with the target agent and both handoff items."""
    spanish = Agent(name="Spanish Agent", instructions="Reply in Spanish.")
    triage = Agent(
        name="Triage Agent",
        instructions="Route the request.",
        handoffs=[handoff(spanish)],
    )
    state = await RunState.from_json(triage, load("rust_handoff.json"))
    check(
        state._current_agent.name == "Spanish Agent",
        f"resumed on {state._current_agent.name!r}",
    )
    types = [getattr(i, "type", None) for i in state._generated_items]
    check(
        types == ["handoff_call_item", "handoff_output_item", "message_output_item"],
        f"items did not survive the read: {types!r}",
    )
    print("ok  rust_handoff.json -> read back with the handoff target and all items")


async def resume_mixed_batch() -> None:
    """One response asked for two tools: only the parked one runs on resume."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[assistant_message("Both handled.", item_id="msg_rust_4")],
                response_id="resp_rust_resume_4",
            )
        ]
    )
    agent = Agent(
        name="File Agent",
        instructions="Manage files.",
        tools=[delete_file, get_weather],
        model=model,
    )
    state = await RunState.from_json(agent, load("rust_mixed_batch_interrupted.json"))
    interruptions = state.get_interruptions()
    check(len(interruptions) == 1, f"expected 1 interruption, got {len(interruptions)}")
    state.approve(interruptions[0])

    executed.clear()
    weather_calls.clear()
    result = await Runner.run(agent, state)
    model.assert_complete()
    check(executed == ["/tmp/porto.txt"], f"parked tool did not run: {executed!r}")
    check(weather_calls == [], f"sibling tool ran again: {weather_calls!r}")
    outputs = [
        str(item.output)
        for item in result.new_items
        if getattr(item, "type", None) == "tool_call_output_item"
    ]
    check(
        outputs.count("The weather in Porto is sunny.") == 0,
        f"the already-executed output was re-emitted: {outputs!r}",
    )
    check(result.final_output == "Both handled.", f"unexpected {result.final_output!r}")
    print("ok  rust_mixed_batch_interrupted.json -> only the parked call ran")


async def resume_second_turn() -> None:
    """The parked turn starts at index 2; turn 1's completed call is not replayed."""
    model = ScriptedModel(
        [
            ModelStep(
                output=[assistant_message("Done.", item_id="msg_rust_5")],
                response_id="resp_rust_resume_5",
            )
        ]
    )
    agent = Agent(
        name="File Agent",
        instructions="Manage files.",
        tools=[delete_file, get_weather],
        model=model,
    )
    state = await RunState.from_json(agent, load("rust_second_turn_interrupted.json"))
    state.approve(state.get_interruptions()[0])

    executed.clear()
    weather_calls.clear()
    result = await Runner.run(agent, state)
    model.assert_complete()
    check(executed == ["/tmp/oslo.txt"], f"parked tool did not run: {executed!r}")
    check(weather_calls == [], f"turn 1 was replayed: {weather_calls!r}")
    check(result.final_output == "Done.", f"unexpected {result.final_output!r}")
    print("ok  rust_second_turn_interrupted.json -> turn boundary honoured")


async def reject_tampered_arguments() -> None:
    """A Rust snapshot whose arguments were edited after the fact is rejected by Python.

    The ledger Rust now writes under `context.tool_invocations` is what makes this possible:
    without it Python has nothing to compare the restored call against.
    """
    data = load("rust_interrupted_approval.json")
    tampered = json.dumps(data)
    tampered = tampered.replace("/tmp/report.txt", "/etc/passwd")
    data = json.loads(tampered)

    agent = Agent(
        name="File Agent", instructions="Manage files.", tools=[delete_file]
    )
    executed.clear()
    try:
        state = await RunState.from_json(agent, data)
        state.approve(state.get_interruptions()[0])
        await Runner.run(agent, state)
    except Exception as error:  # ModelBehaviorError or UserError, depending on the path
        check(
            "reused a tool call ID" in str(error) or "approval" in str(error).lower(),
            f"unexpected rejection: {error!r}",
        )
        check(not executed, f"tool ran before the rejection: {executed!r}")
        print(f"ok  tampered arguments rejected: {type(error).__name__}")
        return
    check(not executed, f"tampered call executed: {executed!r}")
    print("ok  tampered arguments did not execute the tool")


async def main() -> None:
    await resume_interrupted_approval()
    await resume_mixed_batch()
    await resume_second_turn()
    await reject_tampered_arguments()
    await load_completed()
    await load_handoff()
    print("all Rust-written snapshots resumed by the Python SDK")


if __name__ == "__main__":
    asyncio.run(main())
