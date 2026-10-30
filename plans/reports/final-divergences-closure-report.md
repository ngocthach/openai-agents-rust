# Final divergences closure

Area: items previously recorded as "inherent" or "unverified". Each was re-verified against
current source before any change. Build dir `/Users/ngocthach/.cache/oars-build/fin1`.

Verdicts: **6 closed, 1 closed-with-residual, 0 proven-impossible.** Plus one extra row
(`strict_context`) added mid-task by the coordinator: closed.

---

## 1. Legacy approvals divergence — CLOSED

**Was:** Rust honoured every restored per-call approval, because it had no schema cut-off.

**Python's rule, re-read:** `run_state.py:4307` sets
`_allow_legacy_approval_binding_reconstruction = (schema_major, schema_minor) < (1, 15)`, and
`:4310-4318` rebuilds the ledger only at `>= (1,15)`, else clears `_tool_invocations`.
`run_context.py:1650 _mark_restored_unbound_approval_call_ids` then collects every call id in a
per-call `approved`/`rejected` list with no ledger entry. `_tool_invocation_status`
(`run_context.py:586-589`) returns `None` for such an id *without registering it*, so
`get_approval_status` returns `None` (`:1532-1536` sticky branch, `:1552-1576` per-call) — the
approval is requested again. A fresh decision clears it: `_apply_approval_decision` ends with
`_restored_unbound_approval_call_ids.discard(call_id)` (`:1339`).

**Implemented** (cut-off 1.15, numeric compare):

- `run_state/wire.rs`: `parse_schema_version` / `schema_version_at_least`. Numeric is
  load-bearing — `"1.9" > "1.15"` as strings would put every 1.9+ snapshot on the wrong side.
- `run_state.rs`: `STRICT_APPROVAL_BINDING_MIN_SCHEMA_VERSION = (1, 15)`; new
  `RunState::restored_unbound_approval_call_ids` (`#[serde(skip)]`, derived on read, never on
  the wire) and `unbound_approval_call_ids()` (port of
  `_mark_restored_unbound_approval_call_ids`). Applied only when the payload is the
  Python-compatible envelope **and** `>= 1.15`; a detected flat Rust `"1.0"` snapshot
  (`is_legacy_rust_envelope`, shape-based) keeps the tolerance, as it predates the ledger.
  `approve`/`reject` remove the call id, mirroring Python's `discard`.
- `run_internal/`: `RunLoop.unbound_approvals`, populated in `apply_state`; new
  `tools::bound_approval_status` masks the three consult sites
  (`tools.rs:203`, `local_tools.rs` approval gate, `mod.rs` hosted-MCP).

**Two hazards found and handled, neither in the brief:**

1. **Ledger laundering (would have defeated the whole rule).** A masked snapshot that
   re-interrupts gets re-snapshotted, and `tool_invocation_ledger` rebuilt the entry from
   `generated_items` — so the *next* resume would have found the stale decision bound and
   honoured it. Fixed by never registering an unbound id: `tool_invocation_ledger` skips them
   and `process_response` has a `None if unbound.contains(...) => {}` arm, which is exactly
   Python's `is_restored_unbound` early return. Covered by
   `an_un_renewed_approval_stays_unbound_across_a_re_snapshot`.
2. **The handler re-read.** `local_tools.rs` re-read the decision after an `on_approval`
   handler. Reading it unmasked would resurrect the stale decision when the handler declined;
   reading it masked would discard a decision the handler had just made. Now gated on whether
   the handler actually recorded one.

**Checked, not a problem:** hosted-MCP approval requests *are* ledgered on both sides —
`mcp_approval_request` is in Rust's `TOOL_INVOCATION_TYPES` and Python's
`_TOOL_INVOCATION_TYPES` (`_tool_invocation.py:23`); `is_mcp_approval_invocation` is only the
fallback when identity is `None`. And every committed fixture already carries its parked call
in the ledger, so no existing resume path changes.

**Residual (one sentence, narrow):** Python has a *second* unbound source at
`run_state.py:5080-5086` — every call id present in restored items/responses but absent from the
ledger, regardless of whether a per-call decision exists, which also nullifies *sticky* grants
for those calls. Rust's `unbound_approval_call_ids()` reads only the per-call decision lists (the
source the brief names), so such a call is registered on resume and a sticky grant still applies.
Narrow in practice: Python ledgers during response processing, so a restored item without a
ledger entry means a snapshot written before 1.15 or hand-edited.

**Nested `Agent::as_tool` states: covered for free.** `src/agent.rs:1063` rehydrates a nested
snapshot with the same `RunState::from_json`, and those snapshots are written with `to_json()`
(so they carry the 1.21 envelope). The nested runner's `apply_state` then picks up the mask like
any other resume — no separate handling needed.

**Tests** (`tests/core_python_run_state.rs`, all 5 directions + 1 regression):
`an_approval_the_ledger_cannot_bind_is_requested_again`,
`a_renewed_approval_clears_the_unbound_call_id`,
`an_approval_the_ledger_binds_still_authorizes_the_call`,
`a_pre_binding_schema_keeps_the_tolerant_approval_rule` (1.14),
`a_legacy_flat_rust_snapshot_keeps_the_tolerant_approval_rule` (flat 1.0),
`an_un_renewed_approval_stays_unbound_across_a_re_snapshot`.
**Python fixture suite re-run: 24/24 pass.**

---

## 1b. `strict_context` (coordinator addition) — CLOSED

Python: `to_json(strict_context=True)` refuses a non-mapping context without a serializer
(`run_state.py:1692-1697`); `from_string` takes `context_override` / `context_deserializer` /
`strict_context` and errors-or-warns via `_context_meta_requires_deserializer` (`:4250-4262`).

Rust never serializes the context automatically — the application puts its own JSON on
`RunState::context` — so there is **no `context_deserializer` equivalent**: rebuilding a value
from stored JSON *is* the application's own deserialization. What does port is the decision.

- `to_json_strict()` refuses a context that is not a JSON object. This is a real
  interoperability bug, not a formality: Python's deserializer requires a mapping, so a Rust
  snapshot with a scalar/array context was **unreadable by the Python SDK**.
- `wire.rs` `context_meta` no longer lies — a non-object context now reports
  `requires_deserializer: true` instead of claiming `"none"` while carrying a value.
- `from_json_with(value, ContextRestoreOptions)` with `context_override` + `strict_context`;
  `from_json` warns on the same condition (Python logs the identical warning).
  New `RunState::context_requires_restoration` exposes it after a tolerant read.
- Ports of `_context_meta_requires_deserializer` / `_context_meta_warning_message` in `wire.rs`.

Tests: `a_non_object_context_is_refused_by_a_strict_serialization`,
`a_context_that_needs_restoring_is_an_error_only_under_strict_context`,
`a_restorable_context_passes_a_strict_read_untouched`,
`a_rust_written_context_round_trips_through_the_strict_read`.

---

## 2. PTY signal reset — CLOSED (the previous "inherent" row was stale)

The old row said `portable-pty` "exposes no hook" for `preexec_fn`. **No hook is needed: the
crate already does the reset itself.** `~/.cargo/registry/src/*/portable-pty-0.9.0/src/unix.rs`
`spawn_command`, lines 238-255, installs a `pre_exec` closure that calls
`libc::signal(signo, SIG_DFL)` for `SIGCHLD, SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGALRM`, then
`sigprocmask(SIG_SETMASK, &empty_set, NULL)`, then `setsid()` + `TIOCSCTTY`.

That is a strict **superset** of Python's `_PTY_CHILD_SIGNAL_DEFAULTS = (SIGINT, SIGQUIT)`
(`unix_local.py:85`, applied `:459-471`). `SIGPIPE` — the one disposition the Rust runtime
leaves ignored — is reset to `SIG_DFL` by std's own `do_exec` before any user `pre_exec` closure
runs. `src/sandbox/pty.rs:481` spawns through `spawn_command`, so it inherits all of it. No code
change was required.

**Proved empirically**, new `tests/sandbox_pty_signal_defaults.rs` (own test binary, because it
mutates process-wide dispositions). `/proc` is absent on macOS, so the child reports by living
or dying: the test sets `SIG_IGN` in the test process, then runs `sh -c 'kill -<SIG> $$; echo
INHERITED_SIG_IGN'` on the PTY, for INT/QUIT/TERM/HUP, plus PIPE with no parent mutation.
A second test, `the_probe_detects_an_inherited_sig_ign`, runs the same script under a plain
`std::process::Command` and asserts it **does** print — so the PTY assertions are not vacuous.
Both pass.

Residual, unchanged and matching Python: a plain (non-PTY) child *does* inherit `SIG_IGN`.
Python only resets in the PTY spawn, so this is parity, not a gap.

---

## 3. Docker raw-socket exec transport — CLOSED, and it found a real bug

Audited the six observable behaviours. Python's `_DockerExecSocket` +
`_LENGTH_FRAMED_STDIN_SCRIPT` exist for the reason Python's own comment gives
(`docker.py:796-802`): `docker-py` cannot half-close a TLS stdin, so it frames by byte count.
The CLI *can* half-close (`moby/client/hijack.go`), so the framing is genuinely unnecessary.

| Behaviour | Verdict |
| --- | --- |
| Exit codes | Kept — `docker exec` propagates the container process's code. |
| Stdin EOF | Kept — the CLI half-closes; this is what the framing works around. |
| Timeouts | Kept — `kill_on_drop(true)` was already set, so a timed-out proxy is killed. The in-container process itself survives a timed-out exec on **both** SDKs (closing a hijacked `/exec/{id}/start` socket does not kill the exec either), so this is parity, not a CLI-path weakness. |
| stdout/stderr separate | Kept — separate pipes, as Python's demuxed stream. |
| TTY | **No divergence.** Both SDKs refuse stdin on a non-tty PTY session with the identical message "stdin is not available for this process" (`docker.py:1193` vs `pty.rs:600-611`). |
| **Large stdin** | **Was broken.** Fixed. |

`run_docker` wrote stdin to completion *before* draining stdout/stderr. A child that writes more
than a pipe buffer before finishing its read blocks on its own stdout, stops draining stdin, and
the caller blocks mid-write — a permanent deadlock. Reachable from `write_file` and the archive
extract path, both of which pipe caller-sized payloads.

Fixed by extracting `run_piped(command, label, stdin)`, which pumps the three streams
concurrently under `tokio::try_join!`. A `BrokenPipe` from a child that exits without reading is
treated as the child's business (reported by its exit code), not a transport error.

Tests — four unit tests beside `run_piped` that need **no daemon** (the deadlock one would hang
and fail on a 60 s timeout under the old code):
`a_large_stdin_payload_survives_a_child_that_floods_stdout_first` (4 MiB),
`stdout_and_stderr_stay_separate_and_the_exit_code_survives`,
`stdin_reaches_eof_so_a_draining_child_can_finish`,
`a_child_that_ignores_its_stdin_is_not_a_transport_error`.
Plus `tests/sandbox_docker_exec_transport.rs`: three daemon-backed tests that **skip** (never
fail) without a reachable daemon or a local `alpine:latest`, and never pull.

---

## 4. Bedrock auth route — CLOSED (was "unverified"); implementation was already correct

AWS documents **both**, explicitly:

- <https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html> —
  the endpoint table gives `https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions`
  with Authentication "**AWS credentials (SigV4) or Amazon Bedrock API key**", and states "The
  `bedrock-runtime` endpoint supports AWS SigV4 authentication and Amazon Bedrock API key
  authentication." Both curl examples are shown against that URL:
  `Authorization: Bearer $OPENAI_API_KEY` and `--aws-sigv4 "aws:amz:us-east-1:bedrock"`.
- <https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html> — `AWS_BEARER_TOKEN_BEDROCK`
  or `Authorization: Bearer <key>`; short-term keys (SigV4-derived, ≤12 h, recommended) and
  long-term (IAM service-specific credential); usage gated by `bedrock:CallWithBearerToken`.

So the SigV4 path is **not** dead code, and the service name `bedrock` in the Rust adapter
matches the documented `aws:amz:<region>:bedrock`. Precedence (bearer env wins, else SigV4) is
correct and now documented in-code with the doc URLs.

**any-llm cross-check** (`mozilla-ai/any-llm`, `src/any_llm/providers/bedrock/bedrock.py`): it
does **not** use the `/openai/v1` route at all — it drives boto3 `converse`/`converse_stream`,
with `ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"` and otherwise the default boto3 credential
chain. Same precedence, different transport; our use of the OpenAI-compatible route is a
deliberate port choice AWS explicitly supports.

No behaviour change. Both existing tests already cover the two paths
(`bedrock_signs_with_sigv4` asserts `/eu-west-1/bedrock/aws4_request`;
`bedrock_uses_a_long_lived_api_key_when_one_is_set` asserts `Bearer bedrock-key`). Minor: that
test's name says "long-lived", but short-term keys use the same variable — name only.

---

## 5. WebSocket DEBUG-log suppression — CLOSED-WITH-RESIDUAL

Previously filed as having "no Rust equivalent". A library genuinely cannot filter another
crate's target — only the installed logger can, and that belongs to the application — but it
**can ship the filter**, which is the Rust-idiomatic equivalent and was not previously offered.

Verified the leak is real, not theoretical: both crates log under their own module-path targets,
and `tungstenite-0.30.0/src/protocol/mod.rs:475` is `trace!("Received message {message}")` —
whole decoded frames, i.e. model input and output. Close frames and handshake detail are at
`debug!`. Python's `_OpenAIWebSocketLoggerAdapter.isEnabledFor` drops everything
`<= logging.DEBUG` on `websockets.client`; the Rust stack splits the same material across DEBUG
**and** TRACE, so both are suppressed.

Added to `src/models/openai_responses_ws.rs` (not `logger.rs` — see shared-file note):
`WEBSOCKET_LOG_TARGETS`, `WEBSOCKET_LOG_FILTER_DIRECTIVES`
(`"tungstenite=info,tokio_tungstenite=info"`), `is_suppressed_websocket_record`, and
`WebSocketDebugFilter<L>: log::Log` that the application wraps its own logger in. `INFO` and
above pass through — suppressing those would hide real faults. Confirmed the SDK's own WS path
emits no log records at all, so nothing of ours needed redacting.

Tests: `the_websocket_stacks_verbose_records_are_dropped` (frame TRACE, handshake DEBUG and
compat TRACE dropped; a `WARN` from the same stack, a same-prefix-different-crate target
`tungstenite_extra`, and `openai.agents` all preserved) and
`enabled_reports_the_same_decision_as_log`.

**Residual:** the SDK still installs no logger, so an application that installs one without
wrapping it gets the dependency's DEBUG/TRACE. That is not fixable from a library without
seizing the global logger, which would be wrong. Not wired into `examples/*` (`env_logger::init`
sites) because examples were outside the granted scope — one-line follow-up per example.

---

## 6. `ephemeral` signal a `MountStrategy` can read — CLOSED

Root cause found: in Python, `ephemeral` is a field on **`Mount` itself** —
`Mount(BaseEntry)` with `ephemeral: bool = True` (`entries/mounts/base.py:373-385`) — and
`_validate_s3_mount` reads `mount.ephemeral` directly
(`extensions/sandbox/vercel/mounts.py:279-283`). The Rust port had put it on `Entry` only, which
is why a strategy could never see it.

- `src/sandbox/mounts.rs`: `Mount::ephemeral: bool` (default `true`, as in Python) plus
  `Mount::with_ephemeral`. `MountStrategy::validate(&self, mount: &Mount)` already received the
  `Mount`, so this alone closes the structural hole.
- `src/extensions/sandbox/vercel/mounts.rs`: the missing `if !mount.ephemeral` branch — the only
  one of `_validate_s3_mount`'s checks that was absent; all others were already ported.

Tests: `sandbox::mounts::tests::a_strategy_can_read_whether_its_mount_is_ephemeral` (default
holds; a durable mount reaches the strategy through `Mount::validate`) and
`extensions::sandbox::vercel::mounts::tests::refuses_a_mount_that_asks_not_to_be_ephemeral`.

**Cross-area ask (one line, `src/sandbox/entries.rs:92`, not owned):** that line is
`let ephemeral = matches!(kind, EntryKind::Mount { .. });` — it forces `Entry.ephemeral = true`
for every mount regardless of `Mount::ephemeral`, so the two fields can disagree. Python has one
field. Suggested: `EntryKind::Mount { mount } => mount.ephemeral`. Not a correctness bug today
(the strategy reads the authoritative `Mount` field), but the duplication should collapse.

---

## Shared-file edits

1. **`Cargo.toml`** — one additive block:
   ```toml
   [target.'cfg(unix)'.dev-dependencies]
   libc = "0.2"
   ```
   dev-only, needed solely by the PTY signal test to set `SIG_IGN` in the test process. Already
   in `Cargo.lock` at 0.2.191 (transitive). No runtime dependency added.
2. **`src/lib.rs`** — **not touched.**
3. **`src/run_internal/mod.rs`** — mine for item 1, but the coordinator later reassigned it.
   All item-1 edits predate that message. One further line was unavoidable afterwards:
   `context_requires_restoration: false` in the `RunState` struct literal at the snapshot site,
   forced by the new field (the literal is exhaustive, so the crate would not compile without
   it). Nothing else was changed there after the reassignment.
4. **`src/logger.rs`** — **not touched** (reassigned); item 5 went into
   `src/models/openai_responses_ws.rs` instead, which is already `pub mod` in
   `src/models/mod.rs`, so the new items are reachable with no re-export edit
   (`WebSocketDebugFilter`'s doctest compiles against the public path).
   Follow-up for that file's owner: `logger::enable_verbose_stdout_logging` (added in parallel)
   could apply `WEBSOCKET_LOG_FILTER_DIRECTIVES` / wrap itself in `WebSocketDebugFilter`, which
   would close item 5's residual for anyone using the SDK's own logger.

## Breaking changes

1. **`RunState` gained two public fields** (`restored_unbound_approval_call_ids`,
   `context_requires_restoration`), both `#[serde(skip)]`. Exhaustive struct literals need them;
   `..Default::default()` and all serialized forms are unaffected.
2. **`Mount` gained a public field `ephemeral`.** Struct-literal construction needs it;
   `Mount::new` defaults it to `true`, preserving current behaviour.
3. **A restored per-call approval with no ledger entry no longer authorizes its call** on
   Python-envelope snapshots at schema >= 1.15. This is the intended behaviour change: such a
   resume now raises the approval again instead of running the tool. Flat Rust `"1.0"` and
   pre-1.15 snapshots are unaffected.
4. **`VercelCloudBucketMountStrategy::validate` now rejects a non-ephemeral mount**
   (`MountConfigInvalid`). Only reachable via the new `Mount::with_ephemeral(false)`.

## Gates

- `cargo clippy --all-features --all-targets -- -D warnings` — **clean.**
- `rustfmt --edition 2024` on all 12 touched files.
- `cargo test --all-features`, lib + every integration target: **all of my area passes.**
  `lib 629`, `core_python_run_state 24`, `core_run_state 9`, `sandbox_pty_signal_defaults 2`,
  `sandbox_docker_exec_transport 3`, `extensions_models_provider_tests 11`,
  `responses_ws_tests 2`, `sandbox_core_parity_tests 28`, `runner_tests 27`,
  `local_tools_tests 6` — 0 failures.

### Pre-existing failures in another agent's in-flight work (not mine, not fixed by me)

At the time of the final run, a parallel agent's `src/sandbox/archive.rs` / `FileEntry` work was
mid-edit. It leaves `tests/sandbox_archive_tests.rs` **not compiling**
(`E0506: cannot assign to header[_] because it is borrowed`) and 5 tar/archive tests failing:

- `sandbox_e2b_daytona_vercel_tests::daytona_round_trips_a_workspace_archive`
- `sandbox_modal_cloudflare_blaxel_tests::{blaxel_persists_and_hydrates_through_a_tar_file,
  cloudflare_runs_a_full_session_against_the_worker_protocol,
  modal_persists_a_tar_archive_by_default}` — all `invalid tar stream`
- `sandbox_tests::an_archive_cannot_write_through_a_symlink`

All five are tar/symlink archive paths; none touch run state, approvals, PTY, the docker exec
transport, mounts, Bedrock or the WebSocket logger. The tree was also transiently
un-compilable twice mid-task from the same work (`FileEntry::is_symlink` across three provider
files; `SandboxArchiveLimits` / `CompressionScheme` in `session.rs`).

### One pre-existing test updated by this change

`tests/core_run_state.rs::round_trips_through_the_envelope` began failing, correctly:
`sample_state()` is a hand-built state that approves a call its empty ledger does not bind, so
reading it back at 1.21 now flags that approval for renewal. The test now asserts that derived
field explicitly, then clears it and keeps the exact round-trip assertion on everything else. A
snapshot a real run produces always ledgers its parked call, so this shape does not arise
outside the fixture.

---

Status: DONE_WITH_CONCERNS
Summary: All six assigned rows closed (5 closed, 1 closed-with-residual, 0 proven-impossible),
plus the coordinator's added `strict_context` row. Two were stale rather than inherent — the PTY
signal reset is already done by `portable-pty`'s own `pre_exec` (proved with a non-vacuous test),
and Bedrock's `/openai/v1` route is documented by AWS to accept both SigV4 and bearer keys, so
the existing adapter was correct. The audit found two real bugs: a `docker exec` stdin/stdout
deadlock on payloads larger than a pipe buffer, and a ledger-laundering hole that would have let
an un-renewed approval re-bind itself across a re-snapshot and defeat the new rule entirely.
Concerns/Blockers:
- Five tar/archive test failures and one non-compiling test target (`sandbox_archive_tests`) come
  from another agent's in-flight `archive.rs`/`FileEntry` work. Not mine, not fixed by me; none
  touch run state, approvals, PTY, the docker exec transport, mounts, Bedrock or the WS logger.
- Cross-area ask: `src/sandbox/entries.rs:92` still forces `Entry.ephemeral = true` for every
  mount, so it can disagree with the new `Mount::ephemeral`. One line; not a correctness bug
  today because strategies read the authoritative `Mount` field.
- Item 1 residual: Python's second unbound source (restored items absent from the ledger, which
  also nullifies sticky grants) is not ported; see the item 1 section.
- Item 5 residual: a library cannot filter another crate's log target without seizing the global
  logger, so an application that installs a logger without wrapping it still sees the dependency's
  DEBUG/TRACE. Examples were outside the granted scope and are not wired up.
