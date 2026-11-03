# Round 9, r9gaps: five controller-decided feature gaps

Counts below are cases of the Python files, parametrisations counted separately. "Before" is the remaining python-only count after the round-8 reports.

| file | python_only before | ported | covered | python-only after | reason for the rest |
|---|---|---|---|---|---|
| tests/memory/test_session_context_wrapper.py | 7 | 2 | 0 | 5 | the 5 `inspect.signature` introspection cases |
| tests/test_agent_runner.py | 43 | 10 | 1 | 32 | rewind helpers ported (below); the rest as classified in r8g6 |
| tests/sandbox/test_runtime.py | 76 | 10 | 0 | 66 | see sandbox notes |
| tests/test_agent_scoped_approvals.py | 48 | 8 | 0 | 40 | Temporal converter, gc/weakref, object-identity name-only queries on a context |
| tests/test_run_state.py | 150 | 2 | 0 | 148 | unchanged groups |

Totals: 32 ported, 1 covered.

## 1. conversation_locked retry and session rewind

Finding: Python's `rewind_session_items` is called from the retry loop on every retry, but it acts only when a server conversation tracker exists, and the runner passes `session_items_to_rewind` only when `session_persistence_enabled` (`session is not None and server_conversation_tracker is None`). The streamed path never passes it. So in a real run the rewind is a no-op (the item list is empty whenever it could act); only the helper itself is observable. Rust already retries `conversation_locked` identically (`call_model`, `py_models_retry_get_response`), and never writes the model input to a session before the call when the run is server managed, so there is nothing to roll back. I therefore did not add a rewind hook to `call_model`: it would be dead code with no observable effect.

Implemented the helper and made it public: `memory::rewind_session_items(session, items, tracker, ctx)`, `memory::wait_for_session_cleanup`, `memory::fingerprint_input_item` (`src/memory/session_rewind.rs`). It matches the tail by fingerprint (ids ignored when `Session::ignores_item_ids_for_matching`), pops with the run context scope, restores popped items on a mismatch or failed pop, and strips retry-owned tail items back to the last server item when a `ServerConversationTracker` is given. `oai_conversation::fingerprint` became `pub(crate)`.

Tests: `tests/py_core9_r9gaps_rewind.rs` (9: id-stripped store, mismatched tail, preserves unrelated, strips only retry-owned, empty-session strays, cleanup retry after a read error with 0.1 s, cleanup budget with 0.1+0.2 s, two context-scope cases) and `tests/py_core9_r9gaps_rewind_logs.rs` (2: debug target logging under both data policies, shared failure message). `test_conversation_lock_rewind_skips_when_no_snapshot` is covered by `py_models_retry_get_response::conversation_locked_*` plus `a_model_retry_does_not_rewind_committed_session_input` (a retry never touches the session).

## 2. Sandbox pre-stop hooks and `aclose` (additive)

New `src/sandbox/session_lifecycle.rs`, exported from `sandbox`: `PreStopHook`, `register_pre_stop_hook`, `run_pre_stop_hooks`, `pre_stop_hooks_failed`, `aclose_session` and the `SandboxSessionLifecycle` extension trait (`session.register_pre_stop_hook(..)`, `.run_pre_stop_hooks()`, `.aclose()` on `Arc<dyn SandboxSession>`). A session is any trait impl, so the hook list lives in a registry keyed by the shared `Arc` (entries drop with the last handle).

Semantics follow Python: hooks run once, in order, all of them even after a failure, first failure returned, and a failed hook skips the capture and `stop` from then on (shutdown still runs; a failed capture or stop skips shutdown, as Python's single `try`). `aclose` is serialised, closes the session `Dependencies` last, returns the first error, and redacts errors for manifests with credentialed mounts (`redact_for_protected_mounts`). `SandboxSessionGuard::close` and `suspend` now run the hooks first (owned sessions only), skip the snapshot capture when a hook failed, close the dependencies, and redact delete and capture failures.

Not done, with reasons:
- Skipping `session.stop()` itself after a hook failure on the guard path: a Rust `SandboxClient::delete` performs stop and shutdown as one release (unix_local and the other backends), and `stop` here unmounts and frees the workspace. What Python guarantees, that the workspace is never persisted after a hook failure, holds; the stop assertions (`stop_calls == 0`) are therefore asserted on `aclose`, not on the runner-owned path.
- Dependency close failure (1 case): `Dependencies::close` returns `()`, a closer cannot fail, so the case cannot exist (type system).
- CancelledError cases (5): asyncio cancellation semantics. The drop-while-queued case is ported.
- The runner-owned teardown failure is logged (sanitised for protected mounts) and does not fail a finished run; Python raises it. Kept: existing Rust contract, asserted through the log.

Tests: `tests/py_core9_r9gaps_sandbox_lifecycle.rs` (11).

## 3. Duplicate agent names in a sandbox handoff graph

`SandboxAgent` graphs no longer reject two agents with one name. `GraphWalk` (`src/sandbox/agent.rs`) assigns identity keys before the walk: the root and then the agent ranking first by configuration signature (instructions, base instructions, capability types, handoff names, manifest, run_as) keep the bare name; the others are `name#2`, `name#3`, ..., skipping any literal name in the graph (`sandbox`, `sandbox#2`, `sandbox`: keys `sandbox`, `sandbox#2`, `sandbox#3`). Keys do not depend on handoff declaration order. Everything keyed by name is now keyed by this key (plain names whenever unique): `graph_agent_names`, `graph_agent`, `PreparedSandboxGraph::{agents_by_name, prepared_by_name}`, per-agent sessions, `SandboxRunOutcome::sandbox_states_by_agent`. The per-turn `process_context` filter and the active memory manager find the agent by pointer (`process_context_for_agent`, `key_for_prepared`) so same-named agents apply their own capabilities. The streamed completion only has the last agent's name, so its memory manager lookup still picks the first of a name.

Not done: per-agent resume from a stored map. Rust resumes only the starting agent's session (`SandboxRunConfig::session_state`), and has no `RunState._sandbox` section; the manager tests that call private `_resume_state_payload_for_agent` / `_ensure_resume_key` (about 5) therefore stay python-only (private helpers plus the Python-only sandbox section of RunState), as does the full run/resume test (`test_runner_serializes_unique_sandbox_resume_keys...`, resume half). Ported: the collision-free key case, the stable-identity case, the paused-run key case (new), the own-workspace run (new). The old test that asserted rejection now asserts the keys.

Tests: `tests/py_core9_r9gaps_sandbox_duplicates.rs` (4), `tests/sandbox_tests.rs::duplicate_agent_names_get_identity_keys_in_the_handoff_graph` (replaces the rejection test).

## 4. Approvals keyed by agent identity

Approval owners are the agent identity key already used for run-state restore (`agent_identity::identity_key`): the name for a unique or first agent, `name#N` for further same-named agents. The runner creates `ToolApprovalItem`s with `agent` = that key and looks function, native and hosted decisions up under it (`ToolEnv::agent_key`, hosted branch of `execute_side_effects_inner`). For unique names nothing changes (key == name). A duplicate's interruption item therefore reports `agent == "same-name#2"` (documented break of the field's meaning for duplicates only).

Wire: `function_tool_approvals` entries carry `{name, identity}` (identity only when it differs from the name), pending approvals are rewritten to their agent's real name plus identity, and reading takes `identity` when present, else the bare name, so every snapshot written before this change reads back unchanged (a name-only owner is the first agent of that name, as before). `RunState` gained two read-derived, never-serialised fields: `agent_owner_names` (key to name, set by the run and when reading) and `ambiguous_agent_owners` (port of `_get_ambiguous_agent_ids`: same-named agents with identical configuration carry no decisions into a snapshot). Ambiguous owners are not filtered on import: in-process decisions on them still work, as in Python.

Not done: Python's context-level object-identity query API (`ToolApprovalItem(agent=<Agent>)` against a bare `RunContextWrapper`). A `RunContext` has no agent graph; its owner is the string on the item, so name-only inspection cases (`test_name_only_query_does_not_choose_between_agent_owners`, the converter owner cases) stay as classified; with distinct owner strings they are already covered by `py_core2_scopedapprovals_agent`.

Tests: `tests/py_core9_r9gaps_approval_identity.rs` (2: the full same-name sticky matrix, streamed x serialized x approve = 8 Python cases, and the wire shape plus name-only legacy read).

## 5. Hosted MCP interruption with `arguments` removed

`RunState::from_json` now marks a parked hosted MCP request whose serialized `raw_item` has no `arguments` as unbound (`wire::hosted_pending_without_arguments`): a live request always carries them (the runner writes `"{}"` when the server sent none), so a missing key means a snapshot that cannot be bound to one invocation. The resumed run asks again whatever decision or ledger entry exists, and a fresh `approve` clears it, as for other unbound ids.

Tests: `tests/py_core9_r9gaps_hosted_arguments.rs` (2).

## Breaking and additive changes

- `ToolApprovalItem.agent` holds the identity key (differs from the name only for second and later same-named agents). One existing assertion changed accordingly: `py_core3_runstatectx_hook_identity::legacy_root_priority_does_not_depend_on_hook_state` now expects `sub#2` for the non-root `sub`'s parked approval.
- Sandbox graphs with same-named agents are accepted (previously an error).
- `SandboxSessionGuard` teardown: pre-stop hooks, dependency close, redacted delete/capture errors.
- `RunState` has two new `#[serde(skip)]` fields; `ToolEnv` has `agent_key` (crate-private).
- New public: `memory::{rewind_session_items, wait_for_session_cleanup, fingerprint_input_item}`, `sandbox::{PreStopHook, SandboxSessionLifecycle, aclose_session, register_pre_stop_hook, run_pre_stop_hooks, pre_stop_hooks_failed}`, `PreparedSandboxGraph::{process_context_for_agent, key_for_prepared}`.

## Unresolved

- Streamed sandbox runs find the active memory manager by name only (`CompletionInfo` carries no agent handle).
- `RunItem.agent` and `new_items` keep the plain agent name for duplicates; only approval owners use keys.

## Gates

- 238 test binaries touching runner, run_state, approvals, handoffs, sandbox, sessions, MCP and the new files: 3273 passed, 0 failed (`--all-features`).
- clippy `--all-features --all-targets -D warnings`: clean. Default features: lib and all my targets clean; the full `--all-targets` run stops on `tests/py_core9_r9h3_validation.rs:339` (`unused_mut`, another agent's in-flight file).
- rustfmt applied to my files; `src/run_state/wire.rs` still has a pre-existing unformatted schema table that I did not reformat.

Status: DONE_WITH_CONCERNS
