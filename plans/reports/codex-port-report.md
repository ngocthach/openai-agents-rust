# Codex extension port report

Port of `agents.extensions.experimental.codex` (Python SDK 0.24) to
`openai_agents_sdk::extensions::codex`.

Date: 2026-10-29

## Files

Created (nothing outside the allowed scope was touched):

- `src/extensions/codex/mod.rs` — module docs and re-exports (replaced the stub)
- `src/extensions/codex/options.rs` — `CodexOptions` (`codex_options.py`)
- `src/extensions/codex/client.rs` — `Codex` (`codex.py`)
- `src/extensions/codex/exec.rs` — `CodexExec`, `CodexExecArgs`, argv/env builders (`exec.py`)
- `src/extensions/codex/events.rs` — `ThreadEvent`, `Usage`, `ThreadError` (`events.py`)
- `src/extensions/codex/items.rs` — `ThreadItem` and helpers (`items.py`)
- `src/extensions/codex/thread.rs` — `Thread`, `Turn`, `StreamedTurn`, `Input` (`thread.py`)
- `src/extensions/codex/thread_options.rs` — `ThreadOptions` + enums (`thread_options.py`)
- `src/extensions/codex/turn_options.rs` — `TurnOptions` (`turn_options.py`)
- `src/extensions/codex/output_schema.rs` — schema descriptor + temp file (`output_schema_file.py`,
  schema helpers of `codex_tool.py`)
- `src/extensions/codex/tool.rs` — `codex_tool`, `CodexToolOptions`, `CodexToolResult` (`codex_tool.py`)
- `tests/codex_tests.rs` — 13 integration tests
- `tests/fixtures/fake_codex.sh` — fake `codex` CLI (records argv/stdin/env/schema, prints canned JSONL)
- `examples/codex_tool.rs` — agent with the Codex tool (auto-discovered, no `Cargo.toml` change needed)

## Public API

```rust
// client
Codex::new(CodexOptions) -> Codex
Codex::start_thread(Option<ThreadOptions>) -> Thread
Codex::resume_thread(impl Into<String>, Option<ThreadOptions>) -> Thread
Codex::options() / executable_path()

// options (builder style)
CodexOptions { codex_path_override, base_url, api_key, env }
ThreadOptions { model, sandbox_mode, working_directory, skip_git_repo_check,
                model_reasoning_effort, network_access_enabled, web_search_mode,
                web_search_enabled, approval_policy, additional_directories }
TurnOptions { output_schema: Option<Value>, idle_timeout: Option<Duration> }
enum SandboxMode / ApprovalMode / ModelReasoningEffort / WebSearchMode  // ::as_str()

// thread
Thread::id() -> Option<String>
Thread::run(impl Into<Input>, Option<TurnOptions>) -> Result<Turn>
Thread::run_streamed(impl Into<Input>, Option<TurnOptions>) -> Result<StreamedTurn>
StreamedTurn: Stream<Item = Result<ThreadEvent>> + next_event()
Turn { items: Vec<ThreadItem>, final_response: String, usage: Option<Usage> }
Input::{Text, Items} / UserInput::{Text, LocalImage} / Input::normalize()

// stream types (serde via from_value/to_value, with Unknown fallbacks)
ThreadEvent::{ThreadStarted, TurnStarted, TurnCompleted, TurnFailed, ItemStarted,
              ItemUpdated, ItemCompleted, Error, Unknown}
ThreadItem::{AgentMessage, Reasoning, CommandExecution, FileChange, McpToolCall,
             WebSearch, TodoList, Error, Unknown}  // ::id(), ::item_type(), ::as_agent_message()
Usage { input_tokens, cached_input_tokens, output_tokens } // ::to_agent_usage()

// exec (low level)
CodexExec::new(Option<String>, Option<BTreeMap<String, String>>)
CodexExec::run(CodexExecArgs) -> Result<CodexLineStream>
build_command_args(&CodexExecArgs) -> Vec<String>
find_codex_path() -> String

// output schema
OutputSchema::{Descriptor, JsonSchema}::to_json_schema() -> Result<Value>
OutputSchemaDescriptor / OutputSchemaProperty / OutputSchemaField / JsonPrimitive
OutputSchemaFile::create(&Value) -> Result<OutputSchemaFile>   // removed on Drop

// tool
codex_tool<C>(CodexToolOptions<C>) -> Result<FunctionTool<C>>
CodexToolOptions<C>: name, description, params_json_schema, output_schema, codex,
  codex_options, default_thread_options, thread_id, sandbox_mode, working_directory,
  skip_git_repo_check, default_turn_options, span_data_max_chars,
  include_sensitive_span_data, persist_session, with_on_stream(..), with_thread_id_store(..)
CodexToolResult { thread_id, response, usage }  // Display = Python's json.dumps order
CodexToolStreamEvent { event, thread, tool_call }
codex_tool_params_schema(include_thread_id: bool) -> Value
DEFAULT_CODEX_TOOL_NAME / CODEX_TOOL_NAME_PREFIX / DEFAULT_CODEX_TOOL_DESCRIPTION /
DEFAULT_SPAN_DATA_MAX_CHARS
```

## Ported faithfully

- Argv mapping copies `exec.py:63–118` exactly, including flag order, the quoting of `--config`
  values, `network_access` emitted whenever it is set (not only when true), `web_search_mode`
  winning over the legacy `web_search_enabled`, and `resume … -- <thread id>` placed after the
  `--image` flags with the trailing `-` for stdin. Covered by unit tests with full `Vec<String>`
  equality plus the recorded argv in the integration tests.
- Env: an override replaces the inherited environment (`env_clear()`),
  `CODEX_INTERNAL_ORIGINATOR_OVERRIDE=codex_sdk_ts` is added only when absent, `base_url` sets
  `OPENAI_BASE_URL`, `api_key` sets `CODEX_API_KEY`. API-key fallback order matches
  `_resolve_default_codex_api_key`: option, override `CODEX_API_KEY`/`OPENAI_API_KEY`, process env,
  then the SDK default key.
- Event/item parsing with the same required/optional field split (`id`, `command`, `status`,
  `server`, `tool` required; `text`, `query`, `aggregated_output`, `exit_code` optional) and the
  `Unknown` fallbacks, including `item.*` events without an `item` becoming an unknown item.
  Malformed lines fail with `Failed to parse event: {line}`.
- `Thread::run` aggregation, `thread.started` capturing the id for resume (shared between clones
  of a `Thread`), `Codex stream error: …` for error events and the bare message for `turn.failed`.
- Non-zero exit becomes the last stream item: `Codex exec exited with code {n}: {stderr}`.
- Output schema written to a temp dir (`std::env::temp_dir()` + uuid) and removed when the turn
  ends (RAII `Drop`); descriptor validation errors match the Python messages.
- `codex_tool`: name validation (`codex` or `codex_*`), default description, parameter schema
  (`inputs` with `minItems: 1`, per-item `type`/`text`/`path`, optional `thread_id`), input
  validation messages, thread resolution order (tool argument → run context store → configured
  id), `persist_session` reuse with the verbatim mismatch error, `Codex turn failed[: msg]` /
  `Codex stream error: msg`, the `Codex task completed with[out] inputs.` fallback response, usage
  added to the run context (`requests: 1`, cached input tokens), the `Codex command execution`
  custom span lifecycle with `set_error` on failed commands, and the result JSON in Python's field
  order (`thread_id`, `response`, `usage`).

## Known divergences

- **Result string format.** Field order matches Python (`thread_id`, `response`, `usage`), but the
  JSON is compact UTF-8 while Python's `json.dumps` uses `", "`/`": "` separators and
  `ensure_ascii` escaping. The parsed value is identical; the model-visible bytes differ.
- **`on_stream` is awaited inline** instead of Python's producer/consumer queue, so a slow callback
  applies backpressure to the Codex pipe. Event order is preserved either way.
- **Custom parameters schema.** With `params_json_schema` set, extra fields in the arguments are
  ignored instead of rejected (Python validates against the caller's own Pydantic model). The
  default schema still rejects them, matching `extra="forbid"`.
- **`tests/codex_tests.rs` is `#![cfg(unix)]`** because the fixture is a POSIX shell script. CI runs
  on `ubuntu-latest`, and the lib unit tests spawn nothing, so coverage is unaffected there.

## Not ported, and why

- `use_run_context_thread_id` / `run_context_thread_id_key` key derivation. Python writes the id
  onto an arbitrary context object through `setattr`/mapping access; `RunContext<C>` hands out
  `&C`, so there is no generic equivalent and no key namespace to derive. Replaced by
  `CodexToolOptions::with_thread_id_store(get, set)`, which has the same effect (read before the
  turn, write after, `thread_id` hidden from the model's schema) using caller-provided closures.
  Tested in `codex_tool_stores_the_thread_id_on_the_run_context`.
- `TurnOptions.signal` (`asyncio.Event`). Dropping `StreamedTurn` kills the process
  (`kill_on_drop(true)`), which is the idiomatic Rust cancellation path.
- `codex_subprocess_stream_limit_bytes` and
  `OPENAI_AGENTS_CODEX_SUBPROCESS_STREAM_LIMIT_BYTES`: asyncio's 64 KiB `readline` limit has no
  analogue in `tokio::io::Lines`, which grows its buffer as needed.
- Vendored binary lookup (`vendor/<target-triple>/codex/codex`). The Rust crate ships no vendored
  binaries, so resolution is `CODEX_PATH` → `PATH` → the bare name `codex`.
- Full `_enforce_span_data_budget` allocator (the ~110-line proportional budget redistribution
  across `arguments`/`command`/`output`/… keys). Ported: the span lifecycle, the per-value
  truncation with Python's `... [truncated, N chars]` suffix, and dropping sensitive keys. Not
  ported: redistributing one global JSON character budget across keys. `span_data_max_chars` is
  therefore a per-value limit, not a whole-payload limit.
- `_is_codex_tool` marker and the duplicate-codex-tool fail-fast (see below).
- `_DictLike` (`payloads.py`): Python-only mapping sugar. `ThreadEvent`/`ThreadItem` expose
  `to_value()` plus `Serialize`/`Deserialize` instead.
- Per-run `trace_include_sensitive_data` / `tracing_disabled` lookups: `RunConfig` is not reachable
  from `ToolContext`. `custom_span` already honours globally disabled tracing, and sensitive span
  data is controlled by `CodexToolOptions::include_sensitive_span_data` (default `true`).
- Python's keyword-argument overload of `codex_tool(...)` and the mapping/`coerce_*` input forms:
  Rust uses one typed options struct with builder methods.

## Requested core changes (worked around, none blocking)

1. `FunctionTool` has no extension/marker field, so the Python
   `function_tool._is_codex_tool = True` marker and the runner-side fail-fast on two Codex tools
   with the same name (or a Codex tool colliding with another tool) could not be ported. Adding a
   `pub(crate)` marker (or a generic `extra: Option<Value>`) to `FunctionTool` plus the duplicate
   check in `run_internal` would restore it.
2. `ToolContext` does not expose the active `RunConfig`, so per-run tracing policy
   (`trace_include_sensitive_data`) cannot be honoured by tools. Exposing a read-only handle would
   let the Codex spans follow the run's policy instead of a tool-level flag.
3. `tests/fixtures/fake_codex.sh` must keep its executable bit in git; nothing else in the build
   needs changing (the example is auto-discovered and needs no `[[example]]` section).

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/codex
rustfmt --edition 2024 src/extensions/codex/*.rs tests/codex_tests.rs examples/codex_tool.rs
cargo clippy --all-targets -- -D warnings   # exit 0, no warnings
cargo test --lib extensions::codex          # 38 passed
cargo test --test codex_tests               # 14 passed
cargo test --doc extensions::codex          # 3 passed
cargo build --example codex_tool            # ok
```

The integration tests drive the fake CLI fixture only: no network, no real `codex` binary, and the
API key is always the dummy `sk-test` passed explicitly (the fixture's environment is fully
replaced, so a developer's real `OPENAI_API_KEY` cannot leak in). They assert argument mapping,
stdin, the originator/API-key environment, event parsing, the final response and usage, thread-id
reuse on a second turn (both at thread level and through the tool with `persist_session`),
structured input with images, the output schema file contents and its removal, non-zero exit
failures, turn/stream failures, a custom parameters schema with extra fields, and the tool running
end to end through `Runner` with `FakeModel`.
