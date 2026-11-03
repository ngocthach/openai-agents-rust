"""Python side of `guardrails`: an agent-based input guardrail run in parallel with the first
model call, the same guardrail blocking, and a function output guardrail. The verdict is the
set of model requests each configuration sends and the error raised. Mirrors
examples/agent_patterns/input_guardrails.py and integration_tests/openai/test_guardrails.py."""

import asyncio

from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    input_guardrail,
    output_guardrail,
)


class MathHomeworkOutput(BaseModel):
    reasoning: str
    is_math_homework: bool


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)


async def check_math(context, agent, input) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=context.context)
    out = result.final_output_as(MathHomeworkOutput)
    return GuardrailFunctionOutput(output_info=out, tripwire_triggered=out.is_math_homework)


parallel_guardrail = input_guardrail(name="math_homework")(check_math)
blocking_guardrail = input_guardrail(name="math_homework", run_in_parallel=False)(check_math)


@output_guardrail(name="no_equations")
async def no_equations(context, agent, output: str) -> GuardrailFunctionOutput:
    has_equation = "=" in output
    return GuardrailFunctionOutput(
        output_info={"has_equation": has_equation}, tripwire_triggered=has_equation
    )


INSTRUCTIONS = "You are a customer support agent. You help customers with their questions."
QUESTION = "Can you help me solve for x: 2x + 5 = 11"


async def main() -> None:
    cases = [
        ("parallel", Agent(name="Customer support agent", instructions=f"{INSTRUCTIONS} (parallel)", input_guardrails=[parallel_guardrail])),
        ("blocking", Agent(name="Customer support agent", instructions=f"{INSTRUCTIONS} (blocking)", input_guardrails=[blocking_guardrail])),
        ("output", Agent(name="Customer support agent", instructions=f"{INSTRUCTIONS} (output)", output_guardrails=[no_equations])),
    ]
    for label, agent in cases:
        try:
            result = await Runner.run(agent, QUESTION)
            print(f"{label}: no tripwire: {result.final_output}")
        except InputGuardrailTripwireTriggered:
            print(f"{label}: input tripwire")
        except OutputGuardrailTripwireTriggered:
            print(f"{label}: output tripwire")


asyncio.run(main())
