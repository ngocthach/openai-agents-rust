# Independent Parity Audit — Python SDK → Rust Port

Date: 2026-10-30. Read-only audit. No code under `src/` or `tests/` touched.
Python source: `/Users/ngocthach/myGit/openai-agents-python/src/agents/` (paths below are relative to it).
Rust source: `/Users/ngocthach/myGit/openai-agents-rust/src/` (paths below are relative to it).

Gaps only. Verified-present and renamed rows are not listed; see counts at the end.

---

## P1 — Public API / behavior

### 1. `Handoff.failure_error_function` is absent → handoff invocation errors abort the run
- Python: `handoffs/__init__.py:211` (field), `run_internal/turn_resolution.py:605-632`. On an exception from `on_invoke_handoff`/`on_handoff`, Python emits a `ToolCallOutputItem` with the formatter's text and returns `NextStepRunAgain()` — the calling agent can retry or answer without transferring.
- Rust: `handoffs.rs:82-101` (struct has no such field); `run_internal/mod.rs:2779-2790` returns `Err(e)` unconditionally after setting the span error.
- Impact: any application relying on recoverable handoff failures gets a terminated run instead. Also a silent behavior change for `handoff(..., failure_error_function=...)` ports.
- Fix: add `pub failure_error_function: Option<ToolErrorFn<C>>` to `Handoff<C>` plus a `with_failure_error_function` builder, and in `run_internal/mod.rs:2781` branch to a tool-output item + `NextStep::RunAgain` when it is set.

### 2. No `extract()` API, and archive members are never validated in-process
Two parts, both from the same Python module:

**2a. The public `extract()` entry point is missing.**
- Python: `sandbox/session/base_sandbox_session.py:1338` (`async def extract(...)`, `compression_scheme: "tar" | "zip"`), dispatching through `sandbox/session/archive_ops.py:16` (`extract_archive`), `:67` (`extract_tar_archive`), `:84` (`extract_zip_archive`).
- Rust: `sandbox/session.rs:216-809` — the `SandboxSession` trait has `persist_workspace`/`hydrate_workspace` but no `extract`; zip is not handled anywhere (`grep -rn "zip::" src/` → no hits).
- Impact: a public sandbox capability (upload an archive and extract it to an arbitrary workspace path) has no Rust equivalent, and zip archives cannot be ingested at all.

**2b. Archive extraction is delegated to the system `tar` with no member validation.**
- Python validates every member before anything is written: `sandbox/util/tar_utils.py:78` (`safe_tar_member_rel_path` — rejects absolute paths, `..` traversal, symlink members, hardlink members and unsupported member types), plus `validate_tarfile`, `validate_tar_bytes`, `should_skip_tar_member`, `safe_extract_tarfile`, `UnsafeTarMemberError` (all exported from `sandbox/util/__init__.py`), and `safe_zip_member_rel_path` in `sandbox/session/archive_ops.py`.
- Rust: `sandbox/sandboxes/unix_local.rs:752-759` (`hydrate_workspace`) pipes the archive bytes straight into `extract_tar_gz`, which spawns `tar -xzf - -C <root>` (`sandbox/sandboxes/unix_local.rs:764-782`). No member is inspected in-process, and the remote providers shell out to `tar` in the same way. Whether `..` or symlink members are refused therefore depends on the host's `tar` implementation and version, not on the SDK, and no member-count or extracted-size ceiling is applied anywhere.
- The existing test `rejects_path_traversal_and_symlink_escapes` (`sandbox/sandboxes/unix_local.rs:1004-1024`) covers only the `read_file`/`write_file` path policy; it does not exercise `hydrate_workspace` or any archive member.
- Impact: snapshot restore and workspace hydration have weaker, host-dependent guarantees than Python's, and finding 3's limits have nothing to enforce them.
- Fix: add `async fn extract(&self, path, data, compression_scheme)` to `SandboxSession` in `sandbox/session.rs`, and route both it and `extract_tar_gz` through a new `sandbox/archive.rs` that ports `safe_tar_member_rel_path` semantics (and a zip equivalent) in-process before any member reaches the filesystem.

### 3. `SandboxArchiveLimits` has no Rust counterpart
- Python: `run_config.py:192-212` (`max_input_bytes`, `max_extracted_bytes`, `max_members`, each nullable), validated at `run_config.py:218-222`, applied via `BaseSandboxSession._set_archive_limits` (`sandbox/session/base_sandbox_session.py:262`) and `SandboxRunConfig.archive_limits` (`run_config.py:254`).
- Rust: `grep -rn "archive_limits\|ArchiveLimits" src/` → zero hits; `sandbox/runtime.rs` `SandboxRunConfig` has no such field.
- Fix: add `SandboxArchiveLimits` next to `sandbox/types.rs` and an `archive_limits: Option<SandboxArchiveLimits>` field on `SandboxRunConfig`, enforced by the extractor from gap 2.

### 4. `custom_data_extractor` is missing on `ComputerTool`, `ApplyPatchTool` and `CustomTool`
- Python: `tool.py:922` (`ComputerTool.custom_data_extractor`), `tool.py:1565` (`ApplyPatchTool`), `tool.py:1598` (`CustomTool`); context types `ComputerToolCustomDataContext` (`tool.py:149`), `ApplyPatchToolCustomDataContext` (`tool.py:169`), `CustomToolCustomDataContext` (`tool.py:129`).
- Rust: the mechanism exists — `tool.rs:386` (`CustomDataExtractor<C>`), `tool.rs:470` (`FunctionTool.custom_data_extractor`), `mcp/mod.rs:318` (MCP server level) — but the three local tool structs lack it: `tool/local.rs:877` (`ComputerTool`: only `source`, `on_safety_check`), `tool/local.rs:667` (`ApplyPatchTool`), `tool/local.rs:1025` (`CustomTool`).
- Impact: applications cannot attach SDK-only data to computer / apply-patch / custom tool output items, so the `ToolContext::custom_data` round-trip is incomplete for those tools.
- Fix: add `pub custom_data_extractor: Option<CustomDataExtractor<C>>` + `with_custom_data_extractor` to each of the three structs and invoke it where the tool output item is built in `run_internal/local_tools.rs`.

---

## P2 — Behavior divergence and convenience API

### 5. Server-managed conversation + `nest_handoff_history`: Python warns, Rust is silent
- Python: `run_internal/turn_resolution.py:534-540` logs `"Server-managed conversations do not support nest_handoff_history for handoff %s -> %s. Disabling nested handoff history and continuing with delta-only input."`.
- Rust: `run_internal/mod.rs:2846-2849` — the `else if` guard simply includes `&& !self.server_managed()`, so nesting is dropped with no diagnostic.
- Impact: a user who configured `nest_handoff_history` and a `conversation_id` silently gets delta-only input with no hint.
- Fix: log a warning through `logger::log_model_action_warning` at `run_internal/mod.rs:2847` when `nest_handoff_history` is requested and `server_managed()` is true.

### 6. `enable_verbose_stdout_logging()` has no Rust equivalent
- Python: `__init__.py:347-360` — one call sets the `openai.agents` logger to DEBUG with a stdout handler.
- Rust: `logger.rs:22-102` exposes `LOG_TARGET` and the redaction helpers but no debug-enabling entry point.
- Impact: the documented one-liner from `docs/config.md` does not port; users must wire `env_logger`/`tracing-subscriber` themselves.
- Fix: add `pub fn enable_verbose_stdout_logging()` to `logger.rs` that installs a `log` subscriber filtered to `LOG_TARGET` at `Debug` (idempotent), or document the equivalent explicitly in the crate docs.

---

## P3 — Minor / typed-surface weakening

### 7. Git-clone sandbox helpers absent
- Python: `sandbox/util/github.py` (`clone_repo`, `ensure_git_available`), exported from `sandbox/util/__init__.py`.
- Rust: `sandbox/util.rs` has no git helpers (`grep -rn "clone_repo\|ensure_git" src/` → zero hits).
- Note: these are not called from anywhere else inside the Python SDK, so impact is limited to users who imported them for sandbox setup scripts.
- Fix: port as `sandbox/util.rs::clone_repo` / `ensure_git_available` driven through `SandboxSession::exec`, or document the omission.

### 8. Realtime reasoning config is untyped
- Python: `realtime/config.py:173-177` (`RealtimeReasoningConfig`, a `TypedDict` with an `effort` key), used at `realtime/config.py:235` (`RealtimeSessionModelSettings.reasoning`). (`RealtimeReasoningEffort` at `realtime/config.py:62` is a `Literal | str` alias — counted as a typing artifact, not a gap.)
- Rust: `realtime/config.rs:425-426` — `pub reasoning: Option<Value>` with a doc comment showing `{"effort": "low"}`.
- Impact: the reasoning payload's shape is unchecked; a misspelled key reaches the wire.
- Fix: add a `RealtimeReasoningConfig` struct in `realtime/config.rs` and type the field, keeping a `Value` escape hatch.

### 9. `ShellTool` environment payloads are untyped
- Python: `tool.py:1377` (`ShellToolLocalEnvironment`), `:1384` (`ShellToolContainerAutoEnvironment`), `:1394` (`ShellToolContainerReferenceEnvironment`), `:1357`/`:1365` (`ShellToolContainerNetworkPolicyAllowlist` / `…Disabled`), plus `ShellToolHostedEnvironment`.
- Rust: `tool/local.rs:377` — `pub environment: Value`, validated only at runtime in `ShellTool::hosted` (`tool/local.rs:439-443`).
- Impact: misconfigured container environments or network policies fail at run time instead of compile time. (Skills are typed — `ShellToolSkillReference` / `ShellToolInlineSkill` exist in `tool/local.rs`.)
- Fix: add an `enum ShellEnvironment` with the five shapes in `tool/local.rs` and serialize it into the existing `environment` payload.

### 10. `RunState` deserialization has no `strict_context` guard
- Python: `run_state.py:1909-1914` (`to_json(context_serializer=, strict_context=, include_tracing_api_key=)`), `run_state.py:2339-2346` (`from_string(context_override=, context_deserializer=, strict_context=)`).
- Rust: `run_state.rs:105` exposes `pub context: Option<Value>` and `run_state.rs:310` `with_context(Value)`; serde covers the serializer/deserializer hooks idiomatically, but there is no equivalent of `strict_context=True` (reject a snapshot whose context cannot be restored to the original type). `include_tracing_api_key` is covered at `tracing/trace_state.rs:147`.
- Impact: low — a resumed run silently proceeds with a `None`/default context instead of raising, where Python would raise `UserError`.
- Fix: add `RunState::require_context()` or a `strict` flag on `from_json` that errors when `context` is absent.

---

## Counts

Surface enumeration (script-driven, `ast` over every `.py` under `src/agents/`):

| Measure | Count |
|---|---|
| `__all__` blocks parsed | 71 |
| Total `__all__` entries | 1185 (918 distinct names) |
| Public packages audited (`agents`, `realtime`, `voice`, `sandbox`, `mcp`, `memory`, `tracing`, `models`, `testing`, + 9 `extensions/**` and `sandbox/**` sub-packages) | 18 |
| Distinct public names checked in those packages | **661** |
| Present as a Rust `pub` definition under the same canonical name | 511 |
| Present under a verified Rust rename (enum variant, idiom, error code) | 105 |
| Python typing/runtime artifacts with no Rust counterpart required (`TypeVar`, `TypedDict` `*Dict`, `*Callable`, `SessionABC`, `AgentBase`, `RealtimeReasoningEffort`, `np`, `tarfile`, asyncio-cancellation `MCPToolCancellationError`) | 22 |
| **Real gaps** | **23 names** (→ findings 2, 3, 4, 6, 7, 8, 9) |

Member / parameter diffs (31 target types: `Agent`, `RunConfig`, `ModelSettings`, `Runner.run`/`run_sync`/`run_streamed`, every tool class in `tool.py`, `Handoff`, `SessionSettings`, `RunResult`, `RunResultStreaming`, `Usage`, `AgentOutputSchema`, MCP params, realtime config, voice config):

| Measure | Count |
|---|---|
| Fields + public methods compared | **258** |
| Raw diffs reported by the matcher | 41 |
| Diffs that are Rust renames / trait methods / Python-runtime idioms (`httpx_client_factory`→`http_client`, `to_json_dict`, `RunResult.max_turns`→`max_turns()`, `clone()`→`Clone`, `trace_name`→`tool_span(env,"computer",…)`) | 37 |
| **Real gaps** | **4** (→ findings 1, 4) |

Docs coverage (`docs/**.md`, English only — `ja`/`ko`/`zh`/`ref` excluded):

| Measure | Count |
|---|---|
| Distinct backticked identifiers extracted | **926** |
| Not found anywhere in `src/` | 96 |
| Of those: Python stdlib/third-party names, example-local identifiers, or verified renames | 96 |
| **Additional gaps beyond the list above** | **0** |

Env vars: 9 distinct `OPENAI_AGENTS_*` in Python source+docs vs 11 in Rust. Python-only: `OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH` (a patch for the Python `litellm` package) and `OPENAI_AGENTS_INTEGRATION_*` (`integration_tests/conftest.py` only, not SDK). `OPENAI_AGENT_HARNESS_ID` is read at `models/mod.rs:92`. **0 gaps.**

Behavior checks on the high-traffic paths — 8 decision points read on both sides:

| Point | Verdict |
|---|---|
| `tool_use_behavior` four modes + `StopAtTools` + custom fn | present (`agent.rs:121-161`, `:281`) |
| Handoff input-filter precedence (handoff over `RunConfig`) and server-managed rejection | matches (`run_internal/mod.rs:2820-2828` vs `run_internal/turn_resolution.py:677-680`, `:525-529`) |
| `nest_handoff_history` under a server-managed conversation | **divergent — finding 5** |
| Handoff invocation error recovery | **divergent — finding 1** |
| Default tool-error text | identical string (`tool.rs:423` vs `tool.py:2002`) |
| `output_type` set ⇒ plain text is not final output | present (`run_internal/mod.rs:1245`, `:2659`, `:2723`) |
| Streaming event names and `AgentUpdated` placement | full parity: all 11 run-item names (`stream_events.rs:19-46` vs `stream_events.py:29-42`), and `AgentUpdated` is emitted once before the first turn (`run_internal/mod.rs:1107`) and again immediately after each handoff before the next turn's items (`run_internal/mod.rs:1206`), matching Python |
| Input guardrails first-turn-only + parallel partition; output guardrails on the final agent; `session_input_callback` | present (`run_internal/mod.rs:1755-1778`, `:1272-1273`, `run_config.rs:241`) |

**Total: 1845 names/members/identifiers checked, 10 findings covering 27 Python names.**

## Notes on scope

- `run_internal/**` names (~230 helper functions in Python) were **not** audited for name parity: they are internal module structure, legitimately reorganized in `src/run_internal/`. They were audited for *behavior* via the 8 points above.
- Examples: Python has ~50 example topic directories, Rust has 20 example files. Every feature behind the Python-only example topics (`basic/trace_redaction`, `basic/non_strict_output_type`, `basic/retry`, `agent_patterns/hosted_multi_agent_beta`, `agent_patterns/forcing_tool_use`, `decisions/classify`, `model_providers/litellm`, `extensions/visualization`) was verified present in `src/`. Missing example *files* are not counted as parity gaps.
- Not verified, outside the task's eight behavior points: whether Rust cancels the in-flight model call when a *parallel* input guardrail trips mid-turn (Python: `run_internal/agent_runner_helpers.py` `should_cancel_parallel_model_task_on_input_guardrail_trip`). The partition of blocking vs parallel guardrails is present (`run_internal/mod.rs:1766-1771`); the cancellation semantics were not read on either side.
- Verified-present-under-rename highlights, recorded so a future audit does not re-flag them: `GenerationSpanData`→`SpanData::Generation` (`tracing/span_data.rs:31`), `RealtimeAgentStartEvent`→`RealtimeSessionEvent::AgentStart` (`realtime/events.rs:40`), `RealtimeModelSend*`→`RealtimeModelSendEvent`, `get_current_span/trace`→`tracing::scope::current_span/current_trace` (`tracing/scope.rs:18,23`), `get_trace_provider`→`tracing::provider::provider` (`tracing/provider.rs:77`), `ToolOutputText/Image/FileContent`→`ToolOutputPart` (`tool.rs:27`), `set_default_openai_agent_registration`→`set_default_openai_harness` (`models/mod.rs:80`), `SandboxSessionStartEvent/FinishEvent`→`SandboxSessionEvent` + `EventPhase` (`sandbox/events.rs:36,109`), `UnexpectedSandboxCall`/`UnconsumedSandboxSteps`→ error codes in `sandbox/testing.rs:305,337`, `truncate_with_token_budget`→`TruncationPolicy`/`truncate_text` (`sandbox/util.rs:137,274`), `DependencyKey`/`DependenciesError*`→`Dependencies` + `SandboxError` (`sandbox/dependencies.rs:45`).

---

Status: DONE
Summary: Independent mechanical audit of 1845 Python names, members and doc identifiers against the Rust port found 10 gaps covering 27 Python names; two are behavior divergences on the handoff path, two are the missing sandbox `extract()` API and the absence of in-process archive-member validation and limits, and the rest are a missing per-tool `custom_data_extractor`, a missing verbose-logging entry point, and four typed-surface weakenings.
Concerns/Blockers: Finding 1 (`Handoff.failure_error_function`) is a silent behavior change, not just a missing name — a ported application that relied on recoverable handoff failures will see terminated runs. Finding 2b is security-adjacent: Rust hydrates workspaces by piping bytes into the system `tar` (`sandbox/sandboxes/unix_local.rs:764`) while Python validates every member in-process first (`sandbox/util/tar_utils.py:78`), so traversal and symlink safety depends on the host `tar` rather than the SDK, and finding 3's limits have nothing to enforce them. No prior report in `plans/reports/` claimed either row closed, so these are genuine omissions rather than regressions.
