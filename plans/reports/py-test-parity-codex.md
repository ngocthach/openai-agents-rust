# Python test parity: experimental Codex extension

Python: `tests/extensions/experiemental/codex/` (directory misspelled upstream). Ledger:
`/private/tmp/claude-501/-Users-ngocthach-myGit-openai-agents-rust/df34ec6b-67cb-4052-99ee-9636fbfd755b/scratchpad/tp/ledger-codex.tsv` (132 rows, one bucket each).

Counted by grep `def test_`, parametrize not expanded (parametrized cases are ported as loops inside one Rust test).

| file | #tests | ported | covered | python-only | failures found |
|---|---|---|---|---|---|
| test_codex_exec_thread.py | 44 | 14 | 13 | 17 | 0 |
| test_codex_tool.py | 85 | 39 | 20 | 26 | 2 (blank thread_id accepted; serde detail leaked under redaction) |
| test_payloads.py | 3 | 1 | 0 | 2 | 0 |
| total | 132 | 54 | 33 | 45 | 2 |

Totals: PORTED 54 + COVERED 33 + PYTHON-ONLY 45 = 132 (derived from the ledger).

## New files
- `tests/py_codex_exec.rs` (exec + thread + payloads)
- `tests/py_codex_tool.rs` (tool, no span assertions)
- `tests/py_codex_tool_spans.rs` (span tests; own binary because tracing processors are global)
- `tests/py_codex_support/fake_codex.sh` (superset fixture: per-turn events, raw stderr file, per-turn sleep)

## Rust bugs fixed (src/extensions/codex/**)
- `tool.rs` `parse_tool_input`: a blank `thread_id` (`"   "`) was silently turned into `None`; Python's `validate_thread_id` rejects it ("When provided, \"thread_id\" must be a non-empty string."). Now rejected. Inline test updated.
- `tool.rs` `invalid_json`: serde error text (which echoes the offending argument value, e.g. `invalid type: string "SECRET"`) was always included in the `Invalid JSON input for codex tool` error. Now dropped while `debug::dont_log_tool_data()` is true (Python `DONT_LOG_TOOL_DATA`).
- `mod.rs` doc: stale claim that the duplicate-Codex-tool check is not performed; it is (`mark_codex_tool`).

## Notes (not bugs)
- A custom schema built from `codex_tool_params_schema(true)` marks `thread_id` required (strict mode), so a call omitting it fails; Python's pydantic model treats it as optional. Ported test sends `"thread_id": null`. Not changed (schema contract used by existing tests).
- Idle timeout error text matches Python's shape (`Codex stream idle for 0.01 seconds.`).
- `on_stream` panic stands in for a raising handler; the log-redaction half of that test is not portable.

## Cross-area bugs deferred
None found.

## Shared-file edits
None (src/lib.rs and Cargo.toml untouched).

## PYTHON-ONLY groups (one reason each)
- Mapping/kwargs/`inspect.signature` option coercion (typed builders in Rust).
- Python stdlib monkeypatching (shutil/json), process-env mutation.
- asyncio process/task internals (fake asyncio processes, task cancellation, BaseException propagation).
- Vendored-binary target-triple fallback (crate ships no binaries).
- Run-context `setattr`/mapping/pydantic/dataclass key mechanics (replaced by `with_thread_id_store`).
- Pydantic parameters class, lazy-resolver object identity, circular reference, async failure function, `_DictLike` mapping protocol.

## Verification
- `cargo test --all-features --test py_codex_exec` 16 pass; `py_codex_tool` 32 pass; `py_codex_tool_spans` 5 pass; `codex_tests` 20 pass; lib `extensions::codex` 63 pass.
- `cargo clippy --all-features --all-targets -- -D warnings` fails only in other areas' files (src/extensions/models/mod.rs:1016, src/models/openai_responses_ws.rs:553/556, src/sandbox/dependencies.rs:188, src/sandbox/sandboxes/unix_local.rs:1149); no diagnostics in src/extensions/codex/** or my tests, but the lib error stops clippy before it reaches the test targets, so test-file lints are unverified.
- New test files rustfmt-ed individually.
