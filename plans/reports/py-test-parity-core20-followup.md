# Python test parity, round 20 follow-up

Date: 2026-11-02. Base: 0.22.0 plus the uncommitted round-20 work. Four tasks were decided by the
user; this report covers what was done for each.

## 1. Server-managed pending input: identical occurrences

**Change.** Each admitted staged input is now its own occurrence:

- `RunItem::input_id: Option<String>` is new. It mirrors Python's `InputItem.input_id`. A uuid is
  set when `add_input` input is admitted at turn start.
- The snapshot writes `input_id` on `input_item` entries. The reader keeps a string `input_id` on
  an input item and drops it anywhere else, as Python does. Before this change the reader stripped
  it as "Python-only".
- `ServerConversationState::accepted_input_ids` is new. It holds the ids of the occurrences that a
  model call delivered, and it is persisted with `#[serde(default)]`.
- `ServerConversationTracker` has three new methods:
  - `needs_send_item` selects an occurrence that has an id by that id. It does not consult the
    content fingerprint, but it still applies the server-id, call-id and server-output checks.
  - `was_sent_item` asks the same question for an item already in the history.
  - `set_prepared_input` records what the next call is built from.
- `mark_input_as_sent` matches each delivered item back to the first prepared item with the same
  content. When that item carries an input id, the id is marked accepted. This mirrors
  `prepared_item_sources_by_fingerprint`, and it still works after a `call_model_input_filter`
  has rebuilt the items.
- An item without an id keeps the old fingerprint selection. That covers legacy snapshots and
  every non-input item.
- `build_model_input` and `return_undelivered_staged_input` in `src/run_internal/mod.rs` now use
  the item-level methods.

**Retry and resume:**

- A delivered occurrence is in `accepted_input_ids`, so neither a resume nor a retry sends it again.
- An occurrence that was admitted but never delivered is not accepted. A failed request is one
  example. It is sent exactly once on retry; `server_managed_staged_input_is_admitted_at_turn_start_and_sent_once_on_retry`
  still passes.

**Why the old expectation was wrong.** `identical_late_input_is_not_resent_by_fingerprint_*`
asserted that the second "Repeat" never reached the model. That claim was made to stop a retry or
resume from resending an item. A content fingerprint cannot tell "this exact item was delivered"
apart from "an equal item was delivered", so it also swallowed a new message the user sent on
purpose. The model never saw the user's second message, while the run's own history recorded it
as admitted. The server conversation and the local history therefore disagreed. Python avoids
this by identity (`input_id`). The retry and resume guarantee needs only the per-occurrence
identity, not content dedupe, so the old test asserted a side effect of the mechanism rather than
the intended behaviour. It was removed from `py_core3_runstatectx_pending_b.rs`, together with
its two now-unused helpers.

**Tests:** `tests/py_core20_followup_pending_input.rs`, 3 cases:

- `server_tracker_distinguishes_identical_input_occurrences_after_restore`
- `server_managed_resume_sends_identical_late_input_in_later_occurrence_{blocking,streamed}`

The second case asserts that the last call carries "Repeat" once. It also asserts that the admitted
ids are distinct and include the first occurrence's id exactly once, as Python checks `input_id`.

## 2. Dropped Vercel / Cloudflare create and resume futures

**Vercel.**

- `create` and the replacement branch of `resume` now wait for `running` through
  `wait_until_running_or_stop_on_drop`.
- While that wait runs, an armed `StopOnDrop` guard holds the new session. If the future is
  dropped, the guard spawns `session.stop()` (the `/stop` request, never `DELETE`) on the current
  Tokio runtime. With no runtime, it logs a warning instead.
- The guard is disarmed when the wait ends, whether it succeeded or failed. Only a dropped future
  changes behaviour.
- The reattach branch of `resume` is not guarded. A dropped reconnect leaves the original sandbox
  alone and creates nothing, as Python's `[cancelled]` variant asserts.

**Cloudflare: no code change.**

- `create` and `resume` have no await point after the worker returns a sandbox id. `create` returns
  right after `POST /v1/sandbox`, and `resume` returns right after its replacement `POST`.
- A drop can therefore land only before the id is known, where no cleanup is possible.
- The Python cancellation cases in `test_cloudflare.py` (`test_cloudflare_resume_start_*`) cover
  something else: `session.start()` cancelled while a resumed workspace is unmounted for restore,
  where Python shields the remount. That is the mount-lifecycle drop concern, not a half-created
  sandbox. It is not ported here (see Concerns).

**Tests:** `tests/py_core20_followup_sandbox_cancel.rs`, 3 cases, against the existing mock server:

- `a_dropped_create_stops_the_allocated_sandbox` ports `test_installed_provider_cancelled_creation_cleans_allocation`.
- `a_resume_dropped_while_reconnecting_creates_nothing_and_keeps_the_original` ports
  `test_installed_provider_closes_unadopted_reconnect_client[cancelled]`.
- `a_resume_dropped_while_its_replacement_starts_stops_the_replacement` is Rust-only and covers the
  replacement branch.

## 3. Realtime fallible `is_enabled`

**Change.** No realtime path calls `IsEnabled::evaluate` any more; every check uses `try_evaluate`.

- `RealtimeAgent::get_all_tools` filters function tools with the new `filter_enabled_tools`. The
  checks run concurrently; the first error is returned unchanged and the siblings are dropped.
- `RealtimeAgent::try_get_enabled_handoffs(ctx) -> Result<Vec<RealtimeHandoff<C>>>` is new public API.
  It ports `collect_enabled_handoffs` and uses the same `filter_enabled_handoffs` behaviour.
- `build_model_settings_unvalidated` resolves the instructions, the tools and the handoffs with
  `try_join3`. This gives the cross-group cancellation of Python's `gather_with_cancel`, so it
  fails `connect` and `update_agent`.
- Connect-time tool and handoff overrides use the same filters.
- The dispatch-time recheck in `dispatch_tool_call` propagates the error. The session then ends
  through `end_with_tool_task_error`; previously the call was answered "Tool X not found".

**Tests:** `tests/py_core20_followup_realtime_enablement.rs`, 5 cases:

- `collect_enabled_handoffs_cancels_sibling_checks_on_error`
- `filter_enabled_tools_cancels_sibling_checks_on_error`
- `build_model_settings_cancels_cross_group_enablement_on_error`. This is also the
  `resolve_dispatch_snapshot` boundary: Rust resolves the dispatch snapshot at connect, never
  lazily.
- `update_agent_cancels_cross_group_enablement_on_error`, the `get_updated_model_settings`
  boundary.
- `dispatch_time_enablement_failure_ends_the_session`, the `filter_enabled_dispatch_snapshot`
  boundary. Rust rechecks only the tool the call names, so this case asserts that the error ends
  the session, not that a sibling check is cancelled.

Python's "cancelled" and "finished" events are both observed through a drop flag inside the
blocked check.

## 4. `tests/test_tool_output_trimmer`

Skipped, as decided.

## Ledger

`plans/reports/py-test-parity-core20-followup.csv` has one row per Python file ported from;
`remaining_python_only` is `-1` (not recounted).

| Python file | ported |
|---|---|
| tests/test_run_state_pending_input.py | 3 |
| tests/realtime/test_realtime_handoffs.py | 1 |
| tests/realtime/test_realtime_model_settings.py | 2 |
| tests/realtime/test_session.py | 1 (3 params mapped to 3 Rust boundaries) |
| tests/extensions/sandbox/test_vercel_provider_contract.py | 2 |

## Gates

- **New and changed binaries, both feature sets:** `py_core20_followup_{pending_input,realtime_enablement,sandbox_cancel}`
  and `py_core3_runstatectx_pending_b` pass.
- **Server conversation / run_state:** the 21 related binaries plus the four above pass under
  `--all-features` after the final edits (404 tests, 0 failed). Under default features, the same
  set plus `--lib` passes (607 tests, 0 failed).
- **Realtime + Vercel + Cloudflare, `--all-features`:** all 72 binaries that use `realtime::` or the
  Vercel/Cloudflare clients pass: 858 tests, 0 failed. These binaries are `cfg`-gated to empty
  under default features.
- **`--lib --all-features`:** 675 passed.
- **clippy `-D warnings`:** clean under default and `--all-features`, for the lib and the four
  test files.
- **rustfmt:** the touched files are formatted; `src/lib.rs` is untouched.
- **`RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps`:** clean.

## Suggested changelog paragraph (not applied to docs/project-changelog.md)

> Python parity, round 20 follow-up. **Server-managed conversations** now track each admitted
> `RunState::add_input` occurrence by identity (`RunItem::input_id`, persisted in snapshots, and
> `ServerConversationState::accepted_input_ids`). A later input identical to one already
> delivered is sent to the model again, as in Python. A delivered occurrence is still never
> resent, and an undelivered one is sent once on retry. **Vercel:** a `create` or replacement
> `resume` future dropped while the new sandbox starts now stops that sandbox on the Tokio
> runtime instead of leaving it running. **Realtime:** a failing `IsEnabled::Fallible` check is
> now an error. It fails `connect`/`update_agent`, or ends the session at dispatch, instead of
> counting as disabled. Sibling checks are cancelled, and the new
> `RealtimeAgent::try_get_enabled_handoffs` returns the error. Ported 9 Python cases.
>
> Breaking: `ServerConversationState` and `RunItem` gain public fields, which affects struct
> literals that do not use `..Default::default()`. `RunItem` already could not be built by literal
> outside the crate.

## Concerns

1. **Vercel mid-POST window.** Python cleans up a creation cancelled while `POST /v3/sandboxes` is
   still in flight by looking the sandbox up by its client-chosen name. The Rust client has no
   verified get-by-name endpoint, and inventing one was out of bounds, so a drop during the POST
   itself still cannot clean up. The ported test cancels during the `running` wait instead; that
   is where the Rust client actually awaits after allocation.
2. **Failed `running` wait.** A Vercel `create` whose `running` wait fails or times out still hands
   back the error without stopping the sandbox. This is unchanged and Python behaves the same in
   `_ensure_sandbox`. Stopping on that path too would be a one-line change: keep the guard armed
   on error.
3. **Cloudflare cancellation cases not ported.** `test_cloudflare.py`'s three
   `resume_start_*_cancellation` cases are about `CloudflareSandboxSession::start` remounting after
   a dropped restore. That needs a drop guard holding an `Arc` of the session inside `start(&self)`,
   which is a larger change than the half-created-sandbox cleanup requested.
4. **Filter ambiguity.** When a `call_model_input_filter` rebuilds items and the initial input
   contains the same content as a staged input, the delivered item is attributed to the first
   prepared match, as in Python. Python additionally refuses ambiguous rebuilds
   (`validate_pending_input_filter`); that refusal is not ported.
