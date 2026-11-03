---
phase: 3
title: Fix divergences
status: completed
priority: P1
dependencies:
  - 2
---

# Phase 3: Fix divergences

## Overview

Close every `rust-bug` row from the phase-2 report in the Rust crate, with a regression
test per fix, then re-run the whole capture matrix. Sign-off here gates the live runs.

## Requirements

- Functional: after this phase `bash parity/run.sh capture all` reports zero unwaived diffs.
- Non-functional: no public-contract change unless the diff cannot be closed otherwise;
  breaking changes get a changelog "Breaking" entry and a minor version bump.

## Architecture

Per fix: prove the cause from the diff and the Python source line → change the Rust code →
add a focused test in `tests/` (same `py_*` naming as the existing ports when the behaviour
maps to an upstream test, otherwise a descriptive name) → re-run the single scenario → run
the gate.

Divergences already known and deliberately kept (do not reopen; list them in the report):

- Rebuilt-agent resume runs the given graph's tools, Python runs the original object's.
- JSON-restored state cannot fall back to the original agent when the given graph lacks it.

## Related Code Files

<!-- Updated: Validation Session 1 - strict-schema path corrected -->
- Modify: whatever the report names under `src/` (likely candidates: `src/models/openai_responses.rs` request building, `src/model_settings.rs`, `src/strict_schema.rs`, `src/extensions/handoff_prompt.rs` prompt prefix, `src/run_internal/mod.rs` guardrail ordering)
- Create: one test file per fix under `tests/`
- Modify: `docs/project-changelog.md` (new entry, including any Breaking line)

## Implementation Steps

1. Order the rust-bug list by blast radius (request body fields first, counts second).
2. For each: read the Python source that produces the field, fix the Rust side, add the
   test, run `bash parity/run.sh capture <scenario>` until `OK`.
3. If a diff turns out to be a Python bug (upstream sends something the API rejects or
   contradicts its docs), do not copy it: record it as `python-only` with the upstream
   line reference.
4. Run the gate script (`gate070.sh` in the scratchpad: fmt, clippy both feature sets,
   tests all-features and default, each feature alone, ring=0, docs, doctests, MSRV).
5. TLS smoke: `OPENAI_API_KEY=sk-invalid-tls-check cargo run --example hello_world` → 401.
6. Secrets scan over the diff; commits in the usual three groups (feat/fix, test, docs),
   conventional format, no AI references.
7. Sign-off line in the report: "capture matrix: N scenarios, 0 unwaived diffs, gate green,
   commit <sha>".

## Success Criteria

- [ ] `bash parity/run.sh capture all` → every scenario `OK`.
- [ ] Every fix has a test that fails on the previous commit.
- [ ] Gate green; TLS smoke 401; working tree clean after commits.
- [ ] Changelog entry drafted (version decided in phase 5).

## Risk Assessment

- A fix that changes a public type signature → stop, note the trade-off in the report,
  prefer an additive API; the user accepted "best option" decisions, so pick the one that
  keeps existing callers compiling.
- Guardrails (scenario 8): decide per sub-case from the harness output. With the same
  parallel/blocking flag on both sides the request counts must match; a mismatch is a
  `rust-bug` unless an upstream test asserts Python's count is incidental. Do not pre-decide
  it as `rust-ahead`.
