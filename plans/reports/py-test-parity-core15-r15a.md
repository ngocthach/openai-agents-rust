# Core15 r15a: re-audit of the python-only bucket, work list r15_0 (37 of 37 files)

Crate 0.16.0 plus other agents' uncommitted work. Files done: 37. Python-only before 405; moved to ported 4; moved to covered 58; remaining 343. Python test bodies were opened for every candidate that could carry observable behaviour; names alone were not trusted (the Rust ports mostly use renamed tests, so each counterpart was located by content).

## Source changes (all additive or fail-closed)

1. `RunState::try_approve` / `try_reject` (src/run_state.rs): a decision whose invocation identity (call id, tool scope, payload fingerprint) is carried by more than one pending approval of the same agent is refused with Python's message (`Cannot apply approval because multiple current pending approvals contain the same tool invocation identity ...`) and records nothing. The infallible `approve`/`reject` are unchanged. Test: `tests/py_core15_r15a_run_state_approvals.rs`.

2. `ServerConversationTracker::set_omit_reasoning_item_ids` (src/run_internal/oai_conversation.rs, set by the runner in src/run_internal/mod.rs from the reasoning-id policy, including a resumed snapshot's): under the omit policy a reasoning item that was delivered without its id is recognised as sent when it comes up again with the id (`needs_send` checks both fingerprints). Before, it would be sent again. Test: `tests/py_core15_r15a_server_tracker.rs`.

3. `RunContext::try_approve_tool` / `try_reject_tool` and the approval ledger (src/run_context.rs `hosted_request`): a `hosted_tool_call` item wrapping an `mcp_approval_request` in `provider_data` is now a hosted MCP request whose identity is the provider payload (`id`, `server_label`, `name`). A non-string or empty provider id fails with `non-empty request id` and never falls back to the wrapper's call id; before, the wrapper was decided as an ordinary function call keyed by its outer call id. Test: `tests/py_core15_r15a_hosted_mcp_ids.rs`.

4. `tests/py_core15_r15a_usage.rs`: the span usage of a per-call cache-write delta (no source change).

No public API is removed or changed; (2) adds one public method.

## Per-file table

| file | before | ported | covered | after | what remains and why |
|---|--:|--:|--:|--:|---|
| tests/test_run_state.py | 134 | 1 | 37 | 96 | ported: ambiguous current approval identity (new RunState::try_approve/try_reject check). covered by renamed tests in py_core2_runstate_items/envelope/boundary/approvals/sidecars/resume + r10a. remaining: context (de)serializer for dataclass/pydantic contexts (Rust context is JSON), falsy objects, get_interruptions pydantic/typed-extra hardening (~30), object-identity routing of detached approvals to nested states (Rust has no routing), hand-built ProcessedResponse tool-object resolution (Rust rebuilds from model_responses), generated/session list identity, monkeypatched registries, gc |
| tests/models/test_openai_responses.py | 34 | 0 | 6 | 28 | covered: ws stream drop/close/terminal-close, release of the request lock, max_size options (py_models_responses_ws.rs). remaining: monkeypatch of websockets.connect/_open_websocket_connection, omit/NOT_GIVEN sentinels, legacy httpx Timeout/URL, httpx client default timeout, asyncio event-loop/lock/close-handshake internals |
| tests/test_scripted_model.py | 29 | 0 | 0 | 29 | aliasing/snapshot of mutable Python objects, dict step specs, mistyped envelopes, aclose/__str__/BaseException failure modes, contextvar spans |
| tests/test_scripted_sandbox.py | 25 | 0 | 0 | 25 | hasattr/dir introspection, io.BytesIO/StringIO snapshots, dict step validation, deepcopy failure, matcher exceptions |
| tests/test_agent_as_tool.py | 21 | 0 | 0 | 21 | monkeypatch of Runner.run with dummy states/results, falsy extractors/builders, dynamic builder output, asyncio BaseException/stream close |
| tests/test_code_change_verification_runner.py | 19 | 0 | 0 | 19 | bash CI runner script of the Python repo |
| tests/extensions/memory/test_advanced_sqlite_session.py | 14 | 0 | 0 | 14 | close(), injected rollback/init failure and connection invalidation, asyncio cancel settlement, sqlite3 authorizer |
| tests/test_trace_processor.py | 14 | 0 | 7 | 7 | covered: Retry-After wait (unit tests in src/tracing/processors.rs), 4 string-truncation and zero-budget cases, top-level flush_traces export. remaining: non-JSON values (uuid, surrogate, int keys, cyclic), monotonic clock monkeypatch, atexit subprocess, httpx close |
| tests/memory/test_openai_responses_compaction_session.py | 12 | 0 | 5 | 7 | covered: fine-tuned and invalid model names, init validates/accepts model, non-positive rollback budget (py_memory_compaction.rs, py_memory_responses_compaction_a.rs). remaining: cancelled restores x4 (asyncio), falsy default client / falsy decision hook, pydantic model_dump warning suppression, type-guard on MagicMock(spec=Session) |
| tests/test_handoff_history_duplication.py | 10 | 0 | 0 | 10 | nested-history ownership by copy lineage of Python objects (deepcopy of RunItem/input keeps or loses a private occurrence key); Rust values carry equality only, so ownership is claimed by equal input (documented in py_core3_handoffs_owned_replay.rs); dataclass-field introspection |
| tests/voice/test_openai_stt.py | 10 | 0 | 0 | 10 | asyncio generator athrow/cancel during cleanup, close while websocket.close is pending, caplog record shape, numpy dtype, Python 3.10 TimeoutError class, lazy Python session setup vs Rust create_session awaiting setup |
| tests/sandbox/test_unix_local_create.py | 8 | 0 | 0 | 8 | bound-user variants run Python's sudo worker via exec of the worker source; close-failure injection via monkeypatched os.fdopen (the other variants are ported in py_core8_r8g2_create_failures.rs) |
| tests/realtime/test_session_tool_outputs.py | 7 | 0 | 0 | 7 | asyncio cancellation while publishing the tool-end event, monkeypatched _serialize_tool_output, pydantic/dataclass dump failure fallbacks (serde values cannot fail to dump) |
| tests/realtime/test_realtime_handoffs.py | 6 | 0 | 0 | 6 | sibling is_enabled cancellation on a raising check (Rust is_enabled returns bool), on_handoff arity/callable validation and type-annotation tests (typed in Rust) |
| tests/test_run_serial_tests.py | 6 | 0 | 0 | 6 | tests of .github/scripts/run_serial_tests.py (Python CI script) |
| tests/memory/test_session_context_wrapper.py | 5 | 0 | 0 | 5 | signature-inspection opt-in rules (legacy kwargs, failed inspection, partial opt-in, named wrapper parameter); Rust opts in by overriding *_with_context methods (the scope cases are ported in py_core8_r8g10_session_context.rs) |
| tests/test_server_conversation_tracker.py | 5 | 1 | 0 | 4 | ported: reasoning item marked sent without its id is not resent (new omit-ids switch on ServerConversationTracker, set by the runner from the reasoning-id policy; py_core15_r15a_server_tracker.rs). remaining: monkeypatched id() / address-reuse cases (2), and the policy-applies-to-prepared-items case (Rust strips ids in the runner after the tracker selected items; no runner path hands the tracker an item the server does not already hold) |
| tests/sandbox/test_client_options.py | 4 | 0 | 0 | 4 | pydantic: BaseSandboxClientOptions.parse of an instance, model_dump(exclude_unset), duplicate/absent type-discriminator registration |
| tests/test_extension_filters.py | 4 | 0 | 0 | 4 | unserializable Python objects and patch.object(json, 'dumps'); bare str input_history (a list in Rust, ported with a one-message list) |
| tests/test_usage.py | 4 | 1 | 0 | 3 | ported: span usage of a cache-write delta (py_core15_r15a_usage.rs; the per-turn delta itself is covered by py_core4_tracerunner_agent). remaining: raw-usage snapshot detachment/pydantic-unset/non-JSON (3, Rust raw usage is a serde Value), aliasing of synthesized request entries, None token details (u64 in Rust) |
| tests/models/test_agent_registration.py | 3 | 0 | 0 | 3 | dictionary normalisation and unknown-field TypeError of untyped registration settings (Rust takes a typed harness id), resolution via a private _agent_registration attribute |
| tests/sandbox/capabilities/test_apply_patch_tool.py | 3 | 0 | 0 | 3 | approval resolved concurrently while an awaited checker callback runs (asyncio interleaving; the Rust policy closure has no agent identity to consult decisions with), monkeypatched PureWindowsPath |
| tests/sandbox/test_unix_local_special_files.py | 3 | 0 | 0 | 3 | Linux-only fcntl F_SETLEASE kernel lease test (not runnable on this macOS host; Rust opens with O_NONBLOCK like Python), monkeypatched os.open lease conflict |
| tests/test_guardrails.py | 3 | 0 | 0 | 3 | non-callable guardrail UserError (typed closures in Rust), Temporal replay compat switch |
| tests/test_shell_call_serialization.py | 3 | 0 | 0 | 3 | bytes/bytearray commands and the string outcome branch (executor output is a typed enum in Rust) |
| tests/voice/test_input.py | 3 | 0 | 1 | 2 | covered: float32 buffer not mutated by to_base64 (py_voice_misc.rs). remaining: numpy dtype rejection, numpy-array-typed constructor cases |
| tests/memory/test_compaction_suffix.py | 2 | 0 | 0 | 2 | asyncio cancellation of a suffix transaction settling before a newer append (Rust cancels by dropping futures) |
| tests/sandbox/test_dependencies.py | 2 | 0 | 1 | 1 | covered: factory failure then retry (py_sandbox_a_entries_1.rs; the Python case adds the 3.12 eager task factory). remaining: asyncio task cleanup when a rebind races a pending resolve, aclose cancellation |
| tests/sandbox/test_tar_utils.py | 2 | 0 | 0 | 2 | monkeypatched tarfile.extractfile payload failing mid-write and monkeypatched TemporaryFile stream-close accounting (Rust extracts from in-memory bytes, validated before any write) |
| tests/test_run_context_approvals.py | 2 | 1 | 0 | 1 | ported: provider_data wrapper with a non-string request id (py_core15_r15a_hosted_mcp_ids.rs). remaining: bare hosted request with an integer id (Rust request ids are strings), deferred-top-level bare-name alias key |
| tests/voice/test_openai_stt_session_config.py | 2 | 0 | 0 | 2 | dataclass field order and positional subclass construction of STTModelSettings |
| tests/realtime/test_conversion_helpers.py | 1 | 0 | 0 | 1 | pydantic ValidationError for a tool output whose call id is None (Mock object; call id is a String in Rust). The three raw-message conversion cases are ported in py_core7_rtvoicemcp_realtime_wire.rs |
| tests/realtime/test_session_history.py | 1 | 0 | 0 | 1 | duck-typed fake audio content forcing the merge error path with a patched module logger |
| tests/sandbox/test_workspace_jsonl_sink.py | 1 | 0 | 1 | 0 | covered: flushed buffer cleared after each event with flush_every=1 and lines delivered in order (py_sandbox_b_jsonl_sink.rs::flushes_do_not_duplicate_lines_and_clear_the_buffer). The ephemeral-outbox archive case is also ported there |
| tests/test_call_model_input_filter_unit.py | 1 | 0 | 0 | 1 | filter returning an invalid type (typed in Rust) |
| tests/test_runner_model_provider_lifecycle.py | 1 | 0 | 0 | 1 | monkeypatched MultiProvider.aclose raising while the run error propagates |
| tests/tracing/test_reasoning_export.py | 1 | 0 | 0 | 1 | UserList/UserDict message sequences (Python container types) |

Note on fix 2: `needs_send` checks both the id-bearing and the id-stripped fingerprint, Python only the stripped one; the extra check can only make the tracker send less, never more.

## Tests added (4 binaries, 6 tests, all green with `--all-features`)
`py_core15_r15a_run_state_approvals` (2), `py_core15_r15a_server_tracker` (2), `py_core15_r15a_usage` (1), `py_core15_r15a_hosted_mcp_ids` (1 test covering approve and reject). Regression binaries re-run green after the source changes: `py_core2_runstate_sidecars`, `_runstate_approvals`, `_scopedapprovals_hitl`, `_approvals_context`, `py_core12_r12b_agent_tool`, `py_core5_leftover2_agent_tool_approvals`, `py_core6_mcpmisc_resume_binding`, `py_core5_leftover1_server_conversation`, `py_core2_runner_routing`.

## Method and honesty notes
- The ledger counts are per Python test name and the earlier reports gave groups, not case ids, so a candidate list per file was rebuilt from the Python names that no Rust test name matched, then each Rust counterpart was located by content (renamed ports are common: for example the whole `deserialize_items_*` family of `test_run_state.py` is covered by four tests in `py_core2_runstate_items.rs`). Where my candidate list was larger than the ledger count the moves are capped by what I could verify, so per-file moved numbers may overlap with cases the ledger already counted as ported.
- `deserialize_processed_response_*` cases of `test_run_state.py` were NOT counted as covered: they re-resolve tool objects from a snapshot, which Rust never does (it rebuilds execution from `model_responses.last()`); the observable resume paths are covered by the approval/resume binaries.
- Files whose remaining cases are only asyncio cancellation, monkeypatched Python internals, pydantic/dataclass/numpy introspection, Python CI scripts or dynamically typed arguments were confirmed by reading the test bodies of every candidate that could carry user-visible behaviour; the reason per group is in the table.

## Breaking changes
None. One additive public method (`ServerConversationTracker::set_omit_reasoning_item_ids`). Behaviour changes that fail closed or fix a miss: ambiguous pending approvals are refused by `try_approve`/`try_reject`; a wrapped hosted MCP request is decided by its provider identity.

## Gates
- Test binaries listed above pass. Clippy `-D warnings` was run on the lib and the four new test binaries only (a `--all-targets` pass over ~570 test binaries was not feasible on the shared host at load ~40), with `--all-features` and with default features. rustfmt `--check` is clean on every file touched (including `src/run_state.rs`, `src/run_internal/mod.rs`, `src/run_internal/oai_conversation.rs`, `src/run_context.rs`).

## Unresolved questions
- `test_unix_local_special_files.py::test_regular_file_lease_rejected` needs Linux `fcntl(F_SETLEASE)`; this host is macOS, so it could not be run. Rust opens with `O_NONBLOCK` like Python, so a Linux-gated port is probably straightforward.
- `apply_patch` custom-tool approval checker: Python stops evaluating later operations once a decision for the call exists (`_get_approval_status_for_key`); the Rust policy closure receives no agent identity, so the same check cannot be written without widening the policy callback. Left as Python-only (asyncio interleaving in the test).
- Rust `create_session` for STT awaits setup, Python sets up lazily, so a transcript received before the session is established is delivered by Python and dropped by Rust (`test_listener_timeout_drains_buffered_transcript_before_setup`). Left as a lifecycle difference.

Status: DONE_WITH_CONCERNS
Summary: 37 of 37 files read; python-only before 405, moved to ported 4, moved to covered 58, remaining 343. Three source fixes (ambiguous approval refusal, reasoning-id tracker fix, wrapped hosted MCP request identity) plus 4 new test binaries.
Concerns: moved counts are capped by what could be matched to the ledger basis; the three open questions above; clippy scoped to the lib plus the new tests, not `--all-targets`.
