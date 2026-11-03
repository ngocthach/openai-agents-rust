# Round 8, group 10: four controller-decided items

| file | python_only before | ported | covered | python-only after | reason for the rest |
|---|---|---|---|---|---|
| model_settings/test_serialization.py | 4 | 2 | 0 | 2 | pydantic/validate_python introspection cases (not touched) |
| memory/test_session_context_wrapper.py | 15 | 8 | 0 | 7 | 5 signature-introspection cases (`inspect.signature`, kwargs opt-in, partial opt-in, uninspectable); 2 retry-rewind cases: Rust has no session rewind on a `conversation_locked` retry (feature gap, see concerns) |
| deferred_session_writes/test_resume.py | 3 | 2 | 0 | 1 | not part of this item, left as classified |
| mcp/test_mcp_server_manager.py | 50 | 2 | 0 | 48 | the two zero-timeout cases ported; the rest is outside this item |
| sandbox/test_docker_removal.py | 74 | 0 | 0 | 74 | see Docker decision |
| sandbox/test_docker_removal_client.py | 14 | 0 | 0 | 14 | see Docker decision |
| sandbox/test_docker_removal_worker.py | 60 | 0 | 0 | 60 | see Docker decision |

Totals: 14 ported, 0 covered, 206 remain python-only in these files.

## Source changes

1. Retry merge (`src/model_settings.rs`, `src/run_internal/model_call.rs`). `ModelSettings::resolve` now merges `retry` field by field and `backoff` field by field, the override winning per field, as Python's `_merge_retry_settings`.
   To tell "unset" from "default", `ModelRetrySettings.{max_retries, backoff, retry_statuses}` and every `ModelRetryBackoffSettings` field are now `Option`. Defaults apply at use time (`max_retries_or_default()` = 2 as before, `retry_statuses_or_default()`, backoff 0.25/2.0/2.0/jitter on). Added `ModelRetrySettings::merge` and `ModelRetryBackoffSettings::merge`. `None` fields are skipped when serializing.
2. Context-aware sessions (`src/memory/mod.rs`, `src/run_internal/mod.rs`, `src/run_internal/session_write.rs`). New `SessionRunContext<'a>` (type-erased view with `run_context::<C>()` and `context::<C>()`) and four defaulted `Session` methods: `get_items_with_context`, `add_items_with_context`, `pop_item_with_context`, `clear_session_with_context`, each defaulting to the plain method. The runner calls the `_with_context` variants for every history read and write (load, append, guardrail-trip persist, resume settle/reconcile). Wrapper sessions (compaction, encrypted) keep the defaults, so their underlying store stays on the plain scope, as in Python.
3. MCP manager (`src/mcp/mod.rs`). A zero connect or cleanup timeout makes `connect_all` and `reconnect` return a user error `"<field>_timeout_seconds must be a positive finite number of seconds or None."` before any server is touched. Builders and setters stay infallible; validation is at use, because `cleanup_all` cannot return an error. A `Duration` cannot be negative, NaN or infinite, so zero is the only invalid value.

## Docker removal decision: leave python-only

`DockerRemovalService` is an opt-in, host-side, root-only helper: it pauses the container, spawns host Python workers (`_docker_removal_worker.py`, run with `-I -S`) that enter the container's mount namespace through `/proc/<pid>` using ctypes and O_PATH descriptors, and releases "live authority" bindings that are never serialized. Rust has no counterpart service, no `removal_service` client argument and no worker protocol; it makes sense only inside Python's privileged deployment. I read all three test files' fixtures: every case builds the service (`RecordingContainer`/`RecordingWorker`, monkeypatched `Popen`, `worker_code.os`, `exec_run`) and asserts the service or worker protocol (pause/unpause ordering, bind/inspect requests, uid and namespace checks, descriptor closing, transport-uncertainty). None asserts user-visible Docker session behaviour (request shape, label handling) independently of the service, so there was nothing to port through the Rust Docker client. Ordinary Docker recursive removal and mount handling are already covered by `py_sandbox_b_removal.rs` and the docker tests. Reasons per group: fixtures/monkeypatch of Python internals plus a Python-only host worker script.

## TRACE handshake logging

Documented only, no change: tungstenite logs the websocket handshake at TRACE through the `log` crate.

## Breaking changes

- `ModelRetrySettings` fields `max_retries`, `backoff`, `retry_statuses` are `Option`; `ModelRetryBackoffSettings` fields are `Option` and its `Default` is all-`None` (not the numeric defaults). `with_max_retries` still takes `u32`. Updated call sites in 10 test files.
- New defaulted `Session` methods and `memory::SessionRunContext` (additive).
- Zero MCP lifecycle timeouts now error instead of timing out immediately.

## Tests

New: `tests/py_core8_r8g10_retry_merge.rs` (3), `py_core8_r8g10_session_context.rs` (6), `py_core8_r8g10_mcp_timeout.rs` (3, `mcp` gate). 123 binaries touching model settings, sessions, retry, MCP manager: 1869 passed, 0 failed. rustfmt applied to my files.
Clippy `-D warnings`: my files clean in both feature sets; remaining errors are in other agents' in-flight files (`src/memory/advanced_sqlite_session.rs`, `tests/py_core8_r8g6_docstrings.rs`).

## Unresolved

- Rust does not rewind session items when a `conversation_locked` retry happens with server-managed conversations (Python `rewind_session_items`); the two rewind cases cannot be ported without implementing that.
- Changelog/README not edited (left for the controller's round-8 entry).
