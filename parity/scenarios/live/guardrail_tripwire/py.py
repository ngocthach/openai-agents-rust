"""Live L11: an agent-based input guardrail must trip on a math-homework question in both
SDKs, raising the tripwire error; the request count is compared."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import shape  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from agents import (  # noqa: E402
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    Runner,
    input_guardrail,
)

MODEL = os.environ["PARITY_MODEL"]
SETTINGS = ModelSettings(max_tokens=2000)


class MathHomeworkOutput(BaseModel):
    reasoning: str
    is_math_homework: bool


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    model=MODEL,
    model_settings=SETTINGS,
    output_type=MathHomeworkOutput,
)


@input_guardrail(name="math_homework", run_in_parallel=False)
async def math_guardrail(context, agent, input) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=context.context)
    out = result.final_output_as(MathHomeworkOutput)
    return GuardrailFunctionOutput(output_info=out, tripwire_triggered=out.is_math_homework)


async def main() -> None:
    started = shape.start()
    agent = Agent(
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
        model=MODEL,
        model_settings=SETTINGS,
        input_guardrails=[math_guardrail],
    )
    try:
        result = await Runner.run(agent, "Can you help me solve for x: 2x + 5 = 11")
        shape.write("guardrail_tripwire", started, result, tripped=False)
    except InputGuardrailTripwireTriggered as e:
        shape.write(
            "guardrail_tripwire",
            started,
            error=e,
            tripped=True,
            guardrail_requests=e.guardrail_result.run_data.context_wrapper.usage.requests if getattr(e.guardrail_result, "run_data", None) else None,
        )
    except Exception as e:  # noqa: BLE001
        shape.write("guardrail_tripwire", started, error=e, tripped=False)


asyncio.run(main())
