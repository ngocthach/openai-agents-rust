# Parity core21: four documented divergences closed

Closes four known gaps listed in `docs/project-changelog.md` (0.20.0–0.23.0): `get_model` without an API key, the
voice provider's explicit-client conflict, the sqlx suffix replacement under a dropped caller, and identical agent
clones sharing function-tool approvals. 16 Python cases move out of the Python-only bucket (16 ported, 0 covered);
per-file counts in `py-test-parity-core21-divergences.csv`. No live model call was made; no real API key was used.

## 1. `OpenAIProvider::get_model` without an API key

Python (`src/agents/models/openai_provider.py`): `_get_client` builds `AsyncOpenAI` lazily inside `get_model`, and
the client raises `OpenAIError("The api_key client option must be set ...")` without a key. An explicit
`openai_client` or the process default client is used as given. In a run, `get_model` is called inside
`run_single_turn` (`run_internal/run_loop.py`): after the blocking input guardrails (`run.py`, awaited before the
model task is created), after the agent-start hooks, tools, system prompt and handoffs; in a non-streamed turn after
`maybe_filter_model_input` (`call_model_input_filter`), in a streamed turn before it; always before `on_llm_start`.
The model task races the parallel input guardrails (`asyncio.gather(guardrail_task, model_task)`); the model task
first suspends in the agent-start hooks' `gather`, so a parallel guardrail that trips without waiting wins.
`InputGuardrail.run_in_parallel` defaults to true, which is why `test_guardrail_error` (Rust
`py_core3_tracing_errors.rs`, run with no key) expects the tripwire.

Rust before: `run_turn_inner` resolved the model after the blocking guardrails but *before* the parallel-guardrail
race, so a keyless failure would have pre-empted a parallel tripwire (the cause of the earlier breakage).

Change:

- `src/models/openai_provider.rs`: `get_model` fails with `AgentsError::UserError` carrying Python's message when the
  provider builds its own client (no `with_client`, and either an explicit option or no process default client),
  the client targets the official OpenAI base URL, and no key resolves (explicit, `set_default_openai_key`,
  `OPENAI_API_KEY`). An explicit empty key is accepted, as Python accepts `api_key=""`. Documented Rust-only
  leniency: a built client aimed at another base URL (local / self-hosted OpenAI-compatible server) and an explicit
  `with_client` client may stay keyless (Python would raise building the former).
- `src/run_internal/mod.rs`: the streamed turn resolves the model before the `call_model_input_filter`, the
  non-streamed one after it (Python's order). A resolution error goes through `fail_model_resolution`: with
  parallel guardrails, they are polled first (one poll for a non-streamed run, as the existing `biased` select does
  for a model error; awaited for a streamed run, which reports the guardrail outcome in place of a model error), so a
  tripwire wins; otherwise the error is returned. `on_llm_start` never runs on a resolution failure. The early
  resolution for server-managed output guardrails is unchanged (Python's
  `validate_output_guardrails_with_server_managed_conversation` resolves early too).

Tests that called `get_model` without a key now get one the way Python's conftest provides it:
`src/models/multi_provider.rs::routes_prefixes` and `tests/py_models_map_defaults.rs` hand the OpenAI provider
`with_api_key("test_key")`; `tests/py_core2_runconfig_provider.rs` sets `set_default_openai_key("test_key")` in its
serial lock helper. No assertion was weakened.

New tests `tests/py_core21_keyless_model.rs` (own env lock, removes/sets `OPENAI_API_KEY`, never sets a default
key): `test_cc_no_default_key_errors` / `test_resp_no_default_key_errors` ported (plus the multi-provider prefix
and an organization-only provider); keys from provider/env/empty value succeed; keyless explicit client and
non-OpenAI base URL succeed; a keyless run runs `on_agent_start` but not `on_llm_start`, and calls the input filter
only when non-streamed; blocking-trip, immediate-parallel-trip, passing parallel guardrails (streamed and not); a
slow parallel trip wins only when streamed.

## 2. Voice provider: explicit client plus key/base URL/organization/project

Python (`voice/models/openai_model_provider.py`): the constructor raises `UserError("Don't provide api_key,
base_url, organization, or project if you provide openai_client")`.

Change (`src/voice/models/openai_model_provider.rs`, `openai_stt.rs`, `openai_tts.rs`): builders stay infallible.
The options are always recorded; when an explicit client is also set (either order) the provider remembers the
conflict and leaves the explicit client unchanged (it used to apply the options to it). New additive
`OpenAIVoiceModelProvider::validate(&self) -> Result<()>` returns Python's error; `get_stt_model` / `get_tts_model`
hand the models a crate-private config error so `transcribe`, `create_session` (before consuming the audio input)
and `run` fail with it as their first step.

Tests (`tests/py_voice_openai_models.rs`): `voice_provider_rejects_client_with_conflicting_args` (the 5 Python rows,
both builder orders, `validate` plus all three model operations) and
`voice_provider_accepts_client_without_conflicting_args`. The former
`voice_provider_explicit_options_override_the_default_client` was a mis-port (it emulated Python's monkeypatched
*process default* client with an explicit client, which is now a conflict); that Python case is already ported
correctly in `tests/py_core15_r15c_voice_provider.rs`, so the duplicate was replaced.

## 3. sqlx suffix replacement under a dropped caller

Python (`extensions/memory/sqlalchemy_session.py`): the suffix replacement runs under `_await_mutation`, which shields
it; a cancelled caller lets it settle, and a later append lands after it.

Change (`src/memory/sqlx_session.rs`): `SqlxSession` gained a per-session mutation lock (`Arc<tokio::sync::Mutex>`),
taken by `add_items`, `pop_item`, `clear_session` and the replacement. `replace_compaction_suffix(_pruning)` serializes
its items, takes the lock (`lock_owned`, still cancellable while waiting, like Python's acquisition), and runs the
transaction (moved into an owned `SuffixReplacement`) on a `tokio::spawn`ed task that holds the lock guard until it
settles; the caller awaits the `JoinHandle`, so dropping the caller detaches the task instead of aborting it. A
panic in the task is resumed in the caller. There is no `replace_history_atomically` override in the sqlx store, so
"whole" replacement is the same suffix path with every handle. `add_items` / `pop_item` / `clear_session` still run
in the caller's future (a dropped one rolls back), documented on the type.

Test hook: none was needed. The deterministic gate is the pool's only connection (`AnyPoolOptions::max_connections(1)`
plus `pool.acquire()` in the test), not a SQLite write lock (sqlx's deferred transaction fails at once with
`database is locked` instead of waiting, verified in round 20). `tests/py_core21_sqlx_suffix_cancel.rs`:

- `a_dropped_replacement_commits_before_a_newer_append`: the caller future is dropped by a 100 ms timeout after its
  first poll (which already spawned the task), a later `add_items` stays pending, then the connection is released:
  history is `original[:3] + [summary, newer]`. Fully deterministic; fails without the change (the replacement is
  rolled back).
- `a_dropped_run_settles_its_suffix_replacement_before_a_newer_append`: port of the `AsyncSQLiteSession` variant of
  `test_cancelled_suffix_transaction_settles_before_newer_append` through `OpenAIResponsesCompactionSession` and an
  aborted run. Uses a 500 ms sleep for the run to reach the replacement after the mock reply, like the existing
  `SqliteSession` variant.

## 4. Identical agent clones sharing function-tool approvals

Python (`run_context.py` `_FunctionToolApprovalKey`): a decision is keyed by an owner token tied to the agent object
through a weakref (`_function_approval_identity`); a clone or `copy.copy` gets a new token. A converted context
(`_from_serialized`) is detached and `_resolve_function_approval_owners` reattaches it only to a live agent of the
starting graph holding that token; otherwise it stays dormant and the call is asked again. A `RunState` JSON restore
binds each entry to the owner resolved by identity key (`run_state.py`), skipping ambiguous owners.

Can the SDK's internal clones be told apart from a user's clone? Yes. Removing `Clone` from `Agent` and compiling
the library with all features lists every internal agent-value clone; there are exactly five, all standing for the
same agent: `IntoAgentArc for &Agent` (`run.rs`), `From<&Agent> for Handoff` (`handoffs.rs`), `Agent::as_tool`
(`agent.rs`), the sandbox plain-form starting agent (`sandbox/runtime.rs`) and the sandbox-prepared agent
(`sandbox/agent.rs`, previously `clone_with`). The same audit on `SandboxAgent` lists its internal clones
(`as_tool_with`, three graph walks, the preparation hook in `sandbox/runtime.rs`). Builders (`with_*`) take `self`
by value and handoff targets are `Arc`s, so they never clone.

Change:

- `src/agent.rs`: private field `approval_owner` (a v4 UUID, process-unique; a counter could collide across
  processes). `Agent::new` mints one; `Clone::clone` (hence `clone_with`, Rust's `agent.clone(...)`) mints a new one;
  crate-private `clone_same_identity()` keeps it and is used at the five sites above. `SandboxAgent`: `Clone` fresh,
  `clone_same_identity()` at its internal sites. Struct literals of `Agent` were already impossible outside the crate
  (`pub(crate)` fields), so the new field is not breaking at the type level.
- `src/run_context.rs` / `src/run_internal/tools.rs`: the capability a decision binds to is the public
  `function_tool_capability_fingerprint` (unchanged, still pinned stable) followed by `\u{4}` and the owner's
  identity (`owned_capability`). The runner and `approve_tool_for_agent` / `reject_tool_for_agent` use it, so a
  bound decision authorizes only that agent object (and the SDK's same-identity copies of it).
- Snapshots: `RunState::to_json` writes the fingerprint without the owner (the `capability` field and the
  `rust.approval_capabilities` registry), and every resume (in-memory or JSON) detaches owners
  (`run_context::detach_capabilities`; an owned record that lands on an existing detached key is merged over it,
  the later sticky flag winning). `function_tool_status` tries the owned key, then the detached one, so a snapshot's
  decision decides for the agent filed under the same owner key with the same fingerprint — Python's identity-key
  binding on restore, which also covers agents run as tools (not in the handoff graph; an earlier attempt to
  re-attach owners from the handoff graph broke nested agent-tool resumes, including an inner agent sharing the
  outer agent's name). Ambiguous owners are still dropped at write and at restore (existing code).
- Context transport (`export_approvals` / `import_approvals`, Python's context converters) keeps the owner, so a
  transported decision authorizes only the same live agent in this process; a clone, a copy, a rebuilt agent or
  another process is asked again (Python's dormant records).

Deviation kept on purpose: an in-memory `RunState` resumed with a rebuilt agent graph is treated like its JSON (the
decision applies to the agent filed under the same key with the same fingerprint); Python keeps an in-memory state
bound to the original objects. The first attempt kept owners on in-memory resumes and broke ~45 existing Rust tests
that resume `result.to_state()` with a rebuilt agent; resuming the JSON of the same state already behaved this way.

Tests `tests/py_core21_agent_identity_approvals.rs`:

- `runner_requires_reapproval_for_unresolved_context_owners`: rows `source` (source and a `clone_with` sibling both
  approved), `clone` (`clone_with`), `copy` (`Clone::clone`, Python's `copy.copy`), and the handoff-to-source row
  (configured is a clone whose handoffs contain `source`); each interrupts on `owner`, runs nothing, and a fresh
  rejection round-trips through `RunState` JSON. The `missing` row stays in `py_core18_approval_fingerprint.rs`.
- `tied_duplicate_owner_restore_requires_reapproval` (`tied_at_write` x `always`, 4 rows).
- Controls: the owning agent keeps its transported decision; a handoff target keeps its identity while a clone in
  its place re-prompts; a JSON run state resumed with a rebuilt agent runs the approved call.

Existing tests adapted (`tests/py_core18_approval_fingerprint.rs`): `runner_executes_converted_context_decisions`
runs `source` itself (Python's `configured = source; configured.model = ...`) instead of `source.clone()`;
`bound_approvals_round_trip_through_json` looks the live key up by its `fingerprint\u{4}` prefix and expects the
restored state to carry the detached key.

Unrelated test fix: `tests/py_core4_tracerunner_disabled.rs::disabled_runner_sync_preserves_caller_context` released
the shared tracing lock before dropping its `Pipeline`, whose drop resets the global processors, so it could clear the
recorder of the next test (`disabled_runner_restores_context_after_error` failed about half the time once timings
shifted). The pipeline is now dropped first; 8/8 runs of the binary pass.

## Gates

`cargo test --workspace --all-features --no-fail-fast`, `cargo test --workspace --no-fail-fast`,
`cargo clippy --workspace --all-targets [--all-features] -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc
--all-features --no-deps`: all pass (all features: 7,976 passed, 0 failed; default features: 3,747 passed, 0
failed). A first run is where the tracing-test race above showed up.

## Count caveats

- `test_agent_scoped_approvals.py`: the ledger's 22 Python-only cases have no per-case ids; round 18 listed the
  unresolved-owner rows (4) and tied-duplicate rows (4) among the candidates whose recount exceeded the ledger. 8 are
  moved here; the remainder is Temporal-converter rows and gc/weakref cases.
- `voice/test_openai_model_provider.py`: the 5 rows of `test_voice_provider_rejects_client_with_conflicting_args`;
  `test_voice_provider_accepts_client_without_conflicting_args` was already counted as ported.
- `test_config.py`: the two `*_no_default_key_errors` cases recorded as the divergence in round 19.
- `memory/test_compaction_suffix.py`: the one remaining `AsyncSQLiteSession` variant.

## Suggested changelog paragraph

> Closes four known gaps. `OpenAIProvider::get_model` now fails with Python's "The api_key client option must be set
> ..." user error when it builds a client for the official OpenAI API without a key; a run reaches it where Python
> does (after the blocking input guardrails and agent-start hooks, racing the parallel guardrails, before
> `on_llm_start`), and a client for another base URL or an explicit client may still be keyless.
> `OpenAIVoiceModelProvider` remembers an explicit client combined with a key, base URL, organization or project:
> the new `validate()` and the models' first `transcribe` / `create_session` / `run` fail with Python's error.
> `SqlxSession`'s compaction suffix replacement runs on its own task holding a new per-session mutation lock, so a
> dropped caller no longer rolls it back and a later write lands after it. Function-tool approvals are bound to the
> agent object as well as the capability fingerprint: `Agent::clone` (and `clone_with`) makes a different agent whose
> calls are asked again, while the SDK's own copies (run starting agent, handoff targets, agents as tools, sandbox
> preparation) keep the identity; run-state snapshots store the fingerprint alone and decide for the resumed graph's
> agent with the same key and fingerprint. 16 more parity cases ported.
>
> Breaking changes: `get_model` without a key now errors (tests and keyless setups need a key, an explicit client or a
> non-OpenAI base URL); a cloned agent no longer inherits approvals bound to the original, including ones exported
> with `export_approvals`; `ApprovalCapability::fingerprint` / bound `ApprovalsState` keys of a live context carry
> `\u{4}<owner>`; an explicit voice client is no longer modified by `with_api_key` / `with_base_url` /
> `with_organization` / `with_project`.

## Unresolved questions

- Should an in-memory `RunState` resumed with a rebuilt graph keep Python's stricter object binding? It would break
  the common Rust pattern of resuming `to_state()` with a re-created agent.
- `add_items` / `pop_item` / `clear_session` of `SqlxSession` are not shielded (Python's
  `test_mutation_cancellation_waits_for_transaction_exit` covers them); out of scope here.
