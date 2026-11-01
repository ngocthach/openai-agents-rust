# Round 8 python-only re-audit: r8g8 (last third of top-level tests/*.py)

Totals: 42 files, 472 python-only cases before. 32 moved to ported, 9 to covered, 431 remain valid python-only.
CSV: `plans/reports/py-test-parity-core8-r8g8.csv` (before - ported - covered = remaining on every row).

New Rust tests (all green with `--all-features`):
- `tests/py_core8_r8g8_runstate_identity.rs` (9): duplicate agents differing only by `tool_use_behavior`, missing saved identity, value-free agent name/identity errors, malformed JSON with a caller context, supplied context keeps tool input and ledger, lookup-key round trip and malformed keys.
- `tests/py_core8_r8g8_tool_batch.rs` (5): dropped sibling skips post-invoke work, shell/function sibling dropped when a computer call fails, finished sibling's `on_tool_end` completes after a failure, dropping the run drops the running tool.
- `tests/py_core8_r8g8_stream_events.rs` (2), `py_core8_r8g8_transforms_log.rs` (1), `py_core8_r8g8_streaming_logging.rs` (1), `py_core8_r8g8_cyclic_handoff_spans.rs` (2, streamed and not; cycle built with `Handoff::custom`), `py_core8_r8g8_approval_reuse.rs` (1).
- `src/strict_schema.rs` unit tests (3): opt-in open-object rejection, preserved pure refs, node budget on a 100k-hop `allOf` ref chain.

## Per-file table

| file | before | ported | covered | remaining | reason for the remaining group |
|---|--:|--:|--:|--:|---|
| test_run_state_agent_identity | 3 | 0 | 0 | 3 | capability identity (SandboxAgent capabilities are folded into the prepared plain Agent), legacy `.agent` handoff shapes x2 (dynamic typing) |
| test_run_state_compatibility_corpus | 68 | 6 | 0 | 62 | exception/traceback/BaseException/cancellation ownership, monkeypatched `json.loads`, frame-metadata trust, foreign mappings and context subclasses, test-helper self-tests |
| test_run_state_hook_identity | 2 | 0 | 0 | 2 | discriminators on Python hook class names and dataclass-ness |
| test_run_state_pending_input | 11 | 0 | 0 | 11 | raising hooks (8: Rust hooks return `()`), private tracker call, object-identity rewrite (2) (see concerns) |
| test_run_state_pydantic_context | 8 | 0 | 0 | 8 | pydantic default_factory / model_validate of the app context |
| test_run_step_execution | 53 | 8 | 0 | 45 | `CancelledError` raised by tools, BaseException, contextvars, eager task factory, unhashable dict args, exception objects as outputs, raising hooks and `is_enabled`, hand-built processed responses |
| test_runner_model_provider_lifecycle | 1 | 0 | 0 | 1 | raising `aclose` (`ModelProvider::close` returns `()`) |
| test_runner_tracing_disabled | 2 | 0 | 0 | 2 | subclassed `DefaultTraceProvider.create_trace` / `NoOpTrace` with contextvars |
| test_runtime_symmetry_contract | 2 | 0 | 0 | 2 | `__dataclass_fields__` inspection |
| test_scripted_model | 29 | 0 | 0 | 29 | aliasing/snapshots of mutable objects, dict step specs, mistyped fields, contextvars, `aclose` failures, user `__str__` |
| test_scripted_sandbox | 25 | 0 | 0 | 25 | BytesIO/StringIO, dict step-spec validation, introspection, deepcopy, matcher exceptions |
| test_server_conversation_tracker | 5 | 0 | 0 | 5 | monkeypatched `id()`/address reuse (3), tracker policy argument with no reachable Rust scenario (2) |
| test_shell_call_serialization | 3 | 0 | 0 | 3 | bytes/bytearray commands, string outcome (typed) |
| test_shell_examples | 21 | 0 | 0 | 21 | Python examples |
| test_source_compat_constructors | 17 | 0 | 0 | 17 | positional constructor binding and keyword aliases |
| test_start_hook_tool_concurrency | 35 | 0 | 0 | 35 | mutation of a shared `agent.tools`; a Rust `Agent` is read-only during a run |
| test_stream_events | 3 | 1 | 1 | 1 | private helper logging for an unknown/compaction item |
| test_streaming_logging | 1 | 1 | 0 | 0 | |
| test_strict_schema | 8 | 4 | 0 | 4 | mapping subclasses, fresh-copy aliasing, Python value identity |
| test_test_environment | 6 | 0 | 0 | 6 | pytest conftest |
| test_tool_approval_call_id_reuse | 12 | 2 | 0 | 10 | pydantic warnings (3), async formatter cancel (1), private mutation (1), raising hooks (5) |
| test_tool_batch_failure_history | 39 | 0 | 7 | 32 | object identity/late mutation/cyclic outputs, CancelledError (incl. the `cancel`, `cancel_at_verdict`, `tool_cancel` verdicts), raising hooks |
| test_tool_context | 11 | 0 | 0 | 11 | hashability, required-field and dict run_config typing, `from_agent_context(ToolContext)` |
| test_tool_custom_data | 4 | 0 | 0 | 4 | non-JSON object and NaN/inf (`serde_json::Value` cannot hold them) |
| test_tool_exception_formatter | 7 | 0 | 0 | 7 | formatter that raises or returns a non-string |
| test_tool_failure_boundaries | 10 | 0 | 0 | 10 | raising hooks and MCP callbacks, cancellation |
| test_tool_guardrails | 2 | 0 | 0 | 2 | invalid guardrail callables |
| test_tool_identity | 8 | 5 | 0 | 3 | non-string names and attribute-object calls |
| test_tool_name_collision_policy | 6 | 0 | 0 | 6 | formatter cancellation, overridable `get_all_tools`, in-flight mutation |
| test_tool_origin | 3 | 0 | 0 | 3 | weakref/gc, positional dataclass arg |
| test_tool_output_conversion | 1 | 0 | 0 | 1 | `__str__` override |
| test_tool_use_behavior | 6 | 0 | 0 | 6 | wrong behavior types, callable objects |
| test_trace_processor | 14 | 0 | 0 | 14 | non-JSON values |
| test_trace_redaction_example | 6 | 0 | 0 | 6 | Python example |
| test_tracing_errors | 6 | 1 | 0 | 5 | exception `__str__` raising or counted |
| test_tracing_errors_streamed | 1 | 1 | 0 | 0 | |
| test_tracing_provider_safe_debug | 1 | 0 | 0 | 1 | Python logging handler streams |
| test_transforms | 3 | 3 | 0 | 0 | |
| test_update_rclone_pin | 19 | 0 | 0 | 19 | repo CI script |
| test_usage | 5 | 0 | 1 | 4 | `model_construct` None normalisation, `snapshot_usage`/`usage_delta` object helpers |
| test_usaspending_setup_db | 4 | 0 | 0 | 4 | Python example |
| test_visualization | 1 | 0 | 0 | 1 | monkeypatched `graphviz.Source.render` |

Covered:
- `test_stream_events` result-to-queue: `py_core6_streamedleft_stream_events.rs::stream_events_emit_tool_search_items`.
- `test_tool_batch_failure_history` 7 cases (`streamed_` and `nonstreamed_partial_history_waits_for_input_verdict` x pass/reject/error, and `nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail`): `py_core6_nativetools_guardrail_race.rs` (`streamed_partial_history_waits_for_input_verdict`, `nonstreamed_partial_history_waits_for_input_verdict`, `nonstreamed_tool_failure_without_output_cancels_pending_input_guardrail`). Read and run: they use function tools, a gated parallel input guardrail and assert tools ran while the run waits, nothing admitted to history/session until the verdict, partial history kept only on pass, and the tool error raised for every verdict. The earlier claim that Rust awaits the guardrail before running tools was stale: `run_turn` races the guardrail with the model call and leaves it pending (`pending_input_guard`) through tool execution, settled by `settle_pending_input_guard`. No source change was needed.
- `test_usage` snapshot/delta/span cache-write: `py_core4_tracerunner_agent.rs::task_and_turn_spans_export_aggregate_usage` (task span usage `cache_write_input_tokens` 6 and per-turn 3, computed from the run's usage baseline in `src/run_internal/mod.rs`).

## Source bugs fixed
- `src/run_internal/agent_identity.rs`: the duplicate-agent signature ignored `tool_use_behavior`, so same-named agents differing only by it were ranked by traversal order and a restore could land on the wrong agent. It is now part of the signature.
- `src/run_internal/mod.rs`: a snapshot naming an agent identity absent from the restored graph now fails with Python's "Run state references an agent identity that is not present in the restored graph" (value-free); a name-only snapshot keeps "Run state agent not found in agent map".
- `src/strict_schema.rs`: tests only.

## Breaking changes
None to public API. The agent-identity ranking of duplicates that differ only by `tool_use_behavior` can differ from snapshots written by an earlier build.

## Concerns and unresolved questions
1. `test_run_state_pending_input.py`: identical late input is not resent under server-managed conversations (fingerprint-based, documented divergence). Kept as python-only pending a decision.
2. Not re-audited case by case: `test_scripted_model`, `test_scripted_sandbox`, `test_start_hook_tool_concurrency` keep the earlier groups' reasons (typed API, aliasing, ownership).
3. `py_core8_r8g8_cyclic_handoff_spans.rs` reuses `tests/py_core3_tracing_support/mod.rs` via `#[path]` instead of copying it.
4. Whole-crate clippy `--all-targets` fails on other agents' in-flight code (`src/memory/advanced_sqlite_session.rs`, `src/mcp/mod.rs` needless_borrow, other `py_core8_*` files); my targets are clean with `-A clippy::needless_borrow`.
