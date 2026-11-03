"""Python side of `model_settings`: every model setting both SDKs expose, on a model with no
implicit defaults. Mirrors integration_tests/openai/test_model_settings.py."""

import asyncio

from openai.types.shared import Reasoning

from agents import Agent, ModelSettings, Runner


async def main() -> None:
    settings = ModelSettings(
        temperature=0.3,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        tool_choice="auto",
        parallel_tool_calls=False,
        truncation="auto",
        max_tokens=256,
        reasoning=Reasoning(effort="low", summary="auto"),
        verbosity="low",
        metadata={"run": "parity"},
        store=False,
        prompt_cache_retention="in_memory",
        include_usage=True,
        response_include=["reasoning.encrypted_content"],
        top_logprobs=2,
        extra_query={"q": "1"},
        extra_body={"prompt_cache_key": "parity"},
        extra_headers={"x-parity": "1"},
    )
    agent = Agent(
        name="Settings",
        instructions="Reply with exactly PACKAGED_SETTINGS_OK.",
        model="gpt-6-luna",
        model_settings=settings,
    )
    result = await Runner.run(agent, "Confirm the packaged settings path.")
    print(result.final_output)


asyncio.run(main())
