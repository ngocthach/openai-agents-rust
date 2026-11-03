"""Behaviour-shape record for live parity runs (Python side).

Both SDKs write the same JSON keys to ``$PARITY_OUT/<sdk>.json``; ``compare_live.py`` checks
them against the scenario's ``expect.json``. Free text is recorded but never asserted on.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def start() -> float:
    return time.monotonic()


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def from_result(result: Any) -> dict[str, Any]:
    """Shape fields derived from a RunResult / RunResultStreaming."""
    final = result.final_output
    parsed = final if isinstance(final, (dict, list)) else _json_safe(final) if isinstance(final, BaseModel) else None
    items = list(result.new_items)
    tool_calls = []
    annotations = 0
    for item in items:
        raw = item.raw_item
        if item.type == "tool_call_item":
            name = getattr(raw, "name", None) or getattr(raw, "type", "unknown")
            arguments = getattr(raw, "arguments", None)
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except ValueError:
                pass
            tool_calls.append({"name": name, "arguments": arguments})
        if item.type == "message_output_item":
            for part in getattr(raw, "content", []) or []:
                annotations += len(getattr(part, "annotations", []) or [])
    usage = result.context_wrapper.usage
    return {
        "final_output": final if isinstance(final, str) else json.dumps(_json_safe(final)),
        "final_output_parsed": parsed,
        "last_agent": result.last_agent.name,
        "new_item_types": [item.type for item in items],
        "tool_calls": tool_calls,
        "handoffs": [
            f"{item.source_agent.name}->{item.target_agent.name}"
            for item in items
            if item.type == "handoff_output_item"
        ],
        "annotations": annotations,
        "usage": {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
        "response_ids": len(result.raw_responses),
        "input_list_len": len(result.to_input_list()),
    }


def write(scenario: str, started: float, result: Any = None, error: BaseException | None = None, **extra: Any) -> None:
    record: dict[str, Any] = {
        "scenario": scenario,
        "sdk": "py",
        "model": os.environ.get("PARITY_MODEL"),
        "error": None,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
    if result is not None:
        record.update(from_result(result))
    if error is not None:
        record["error"] = {
            "type": type(error).__name__,
            "status": getattr(error, "status_code", None),
            "message": str(error)[:200],
        }
    record.update(extra)
    Path(os.environ["PARITY_OUT"], "py.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({k: record[k] for k in ("scenario", "error", "usage", "elapsed_ms") if k in record}, default=str))
