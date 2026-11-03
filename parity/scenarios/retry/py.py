"""Python side of `retry`: the first model request gets a 429, the SDK-level retry (one
retry, tiny backoff, 429 retried) sends it again. The underlying client's own retries are
switched off so the request count is the SDK's. Mirrors examples/basic/retry.py and
integration_tests/openai/test_retry.py."""

import asyncio

from openai import AsyncOpenAI

from agents import (
    Agent,
    ModelRetryBackoffSettings,
    ModelRetrySettings,
    ModelSettings,
    Runner,
    retry_policies,
    set_default_openai_client,
)


async def main() -> None:
    set_default_openai_client(AsyncOpenAI(max_retries=0))
    agent = Agent(
        name="Assistant",
        instructions="Reply with exactly RETRY_OK.",
        model="gpt-6-luna",
        model_settings=ModelSettings(
            retry=ModelRetrySettings(
                max_retries=1,
                backoff=ModelRetryBackoffSettings(initial_delay=0.01, max_delay=0.02, jitter=False),
                policy=retry_policies.http_status([429]),
            )
        ),
    )
    result = await Runner.run(agent, "Confirm the retry path.")
    print(result.final_output)


asyncio.run(main())
