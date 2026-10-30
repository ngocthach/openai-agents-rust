# Audit fixes (small): findings 6 and 8

Date: 2026-10-30. Source: `plans/reports/independent-parity-audit-report.md`, findings 6 and 8.
Both re-verified against Python before changing code.

## Closed rows

### Finding 6 — `enable_verbose_stdout_logging()`

Python (`__init__.py:347-360`): sets the `openai.agents` logger to DEBUG and attaches a stdout
`StreamHandler`, guarded by a module global + lock so repeated calls reuse one handler.

Rust (`src/logger.rs`):

- `VerboseStdoutLogger`: `log::Log` impl accepting only records on `LOG_TARGET`
  (`openai.agents`) or its dotted children, at `Debug` or more severe. The filter is re-checked
  inside `log()` because the `log::debug!` family calls `log()` without consulting `enabled()`.
  Writes with `writeln!(stdout().lock(), "{level} {target}: {args}")` and ignores I/O errors — a
  logger must not panic on a closed pipe.
- `pub fn enable_verbose_stdout_logging() -> bool`: `OnceLock`-guarded, so idempotent; calls
  `log::set_logger` and raises `max_level` to `Debug` only on success. Returns `true` when the
  SDK backend is in place (this or an earlier call), `false` when another `log` backend already
  owns the process — no panic, nothing replaced, the other backend's `max_level` untouched.
- Works with no backend configured by the application, which was the point of the row.
- Redaction still applies; documented on the function and in the README.

Tests: `tests/logger_verbose_stdout.rs` (new). A `log` backend is a process global, so each
behavior runs in its own process: the `child_*` tests re-exec the test binary with
`--exact <name>` and an env marker, and the parent asserts on the child's real stdout —
SDK debug + info lines appear, `openai.agents.realtime` appears, `Trace` and other crates'
targets do not; and with an application backend installed first, the call returns `false` and
nothing is printed.

Docs: module docs in `src/logger.rs` (crate docs) + doctest on the function; new `## Logging`
section in `README.md` (compiles under the README doctest) and a `Python to Rust map` row.

Known limitation (pre-existing, not introduced here): 90 `log::*!` sites under `src/` log on the
default module-path target instead of `LOG_TARGET` (`src/tracing/processors.rs`,
`src/memory/openai_responses_compaction_session.rs`, `src/mcp/*`,
`src/models/openai_chatcompletions.rs`, `src/extensions/**`, ...). The stdout backend filters to
`LOG_TARGET` as specified, so those lines stay hidden. Python routes them all through the
`openai.agents` logger. Fixing it means adding `target: LOG_TARGET` at those call sites — files
outside this area, so it is left to their owners.

### Finding 8 — typed `RealtimeReasoningConfig`

Python (`realtime/config.py:173-177`): `RealtimeReasoningConfig` is a `total=False` TypedDict with
exactly one key, `effort: RealtimeReasoningEffort` (`Literal["minimal","low","medium","high","xhigh"] | str`).

Rust (`src/realtime/config.rs`):

- New `RealtimeReasoningConfig { effort: Option<String>, #[serde(flatten)] extra: Map<String, Value> }`
  — `Option<String>` matches the local convention for `Literal | str` (`Reasoning::effort`,
  `RealtimeTurnDetectionConfig::eagerness`); `extra` is the `Value` escape hatch.
  Helpers: `new(effort)`, `with_extra(key, value)`, `to_session_value()`, `From<&str>`/`From<String>`.
- `RealtimeSessionModelSettings::reasoning` is now `Option<RealtimeReasoningConfig>`; new
  `with_reasoning(impl Into<RealtimeReasoningConfig>)` builder; `merge` unchanged (field already listed).
- `src/realtime/openai_model.rs:1149` serializes through `to_session_value()`. Wire payload is
  identical: `skip_serializing_if` + flatten means `{"effort":"low"}` in, `{"effort":"low"}` out.
- `src/realtime/mod.rs`: added `RealtimeReasoningConfig` to the existing `pub use config::{...}`.

Tests: `tests/realtime_tests.rs::typed_reasoning_config_keeps_the_wire_payload` — payload shape,
extra-key passthrough + round-trip back into the struct, no `reasoning` key when unset, and merge
override precedence.

## Breaking changes

- `RealtimeSessionModelSettings::reasoning` changed from `Option<serde_json::Value>` to
  `Option<RealtimeReasoningConfig>` (public, crate is published at 0.4.0). Callers that passed
  `json!({"effort": "low"})` now pass `RealtimeReasoningConfig::new("low")` or `"low".into()`,
  or keep an existing object with `serde_json::from_value::<RealtimeReasoningConfig>(value)`
  (unknown keys land in `extra`). A non-object `Value` is no longer representable. No in-repo
  caller besides the transport.
- `enable_verbose_stdout_logging` is additive.
- `docs/project-changelog.md` is outside this area: an unreleased entry is still needed for both.

## Shared-file edits

- `src/lib.rs`: one additive line, `pub use logger::enable_verbose_stdout_logging;` (placed
  alphabetically between the `lifecycle` and `memory` re-exports). Not rustfmt'ed.
  The crate-doc header was deliberately left untouched to keep the shared file to one line; the
  rendered crate docs still carry the new API via `src/logger.rs` module docs and the README,
  which is included as crate docs at `src/lib.rs:133`.
- `README.md`: new `## Logging` section + one `Python to Rust map` row.
- `Cargo.toml`: not touched (no new dependency; the backend is ~20 lines over the existing `log`).

## Gates

- `cargo test --all-features --test logger_verbose_stdout --test realtime_tests` — pass (4 + 41).
- `cargo test --features realtime --lib realtime` — pass (41).
- `cargo test --all-features --doc` — pass (49), includes the README and `logger.rs` doctests.
- `cargo clippy --features realtime --lib --test realtime_tests --test logger_verbose_stdout -- -D warnings` — clean.
- `cargo clippy --all-features --all-targets -- -D warnings` — re-run after everything above;
  still **fails in another area's files**, not fixed per the rules: `tests/sandbox_docker_exec_transport.rs:19` imports
  `sandbox::types::ExecRequest` (does not exist), and `sandbox::types::FileEntry` gained an
  `is_symlink` field whose initializers are missing in `src/extensions/sandbox/daytona.rs:2518`
  and `src/extensions/sandbox/e2b/mod.rs:2104`. Parallel sandbox work in flight; my files compile
  clean with `--all-features` (verified before that breakage landed) and under `--features realtime`.
- `rustfmt --edition 2024` on `src/logger.rs`, `src/realtime/config.rs`, `src/realtime/mod.rs`,
  `src/realtime/openai_model.rs`, `tests/logger_verbose_stdout.rs`, `tests/realtime_tests.rs`.

Status: DONE
Summary: Findings 6 and 8 closed — `enable_verbose_stdout_logging()` ships a dependency-free,
idempotent, non-panicking stdout `log` backend filtered to `openai.agents` (re-exported from
`lib.rs`, documented in module docs and README, process-isolated tests), and realtime `reasoning`
is now the typed `RealtimeReasoningConfig` with a flattened extra-key escape hatch and a
byte-identical wire payload.
Concerns/Blockers: `--all-features --all-targets` clippy is red from in-flight sandbox work in
another area (`ExecRequest` import, `FileEntry::is_symlink`). 90 SDK log sites log on module-path
targets rather than `LOG_TARGET`, so they stay invisible to the new backend; that is a pre-existing
divergence from Python spread over files outside this area.
