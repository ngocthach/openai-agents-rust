# Phase 03: Models

Files: `src/models/{mod,interface,openai_client,openai_responses,openai_chatcompletions,openai_provider,multi_provider,default_models,retry}.rs`

- `Model` trait: `get_response(ModelRequest) -> ModelResponse`, `stream_response(ModelRequest) -> BoxStream<Result<ResponseStreamEvent>>`.
- `OpenAIClient`: api key / base url / org / project / headers; POST json; SSE via reqwest byte stream.
- Responses model: request builder (instructions, input, tools, tool_choice, parallel_tool_calls, text.format json_schema, reasoning, store, include, previous_response_id, conversation, metadata, extras); parse `Response` object; stream: emit raw events and build final `ModelResponse` from `response.completed`.
- Chat Completions model: items -> messages conversion, tools -> functions, response_format json_schema, parse choice -> output items, stream deltas -> synthesized Responses-style events.
- `OpenAIProvider` (use_responses flag), `MultiProvider` (prefix routing), default model + settings resolution, env `OPENAI_DEFAULT_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`.
- Retry: `ModelRetrySettings` with backoff and simple policy (status codes / network error).

Validation: unit tests with a mock HTTP server (`wiremock`? -> use `httpmock`-free approach: tests spin a tiny hyper/axum? keep simple: unit test converters; live smoke test gated on env).
