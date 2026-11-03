---
phase: 5
title: "Report and release"
status: pending
priority: P2
dependencies: [4]
---

# Phase 5: Report and release

## Overview

Publish the findings and, if phase 3 changed crate code, release the fixed version.

## Requirements

- Functional: one report with the full capture matrix, the live matrix, every waiver and
  every known gap; docs updated; crate released if code changed.
- Non-functional: no secret, no absolute home path, no raw live transcript in the report
  beyond the shape fields.

## Related Code Files

- Create: `plans/reports/parity-live-and-capture-261103-0651-python-vs-rust-report.md`
  (merges the phase-2 and phase-4 results; the phase-2 report may be folded in)
- Modify: `docs/project-changelog.md` (release entry + known gaps), `docs/project-roadmap.md`
  (parity harness section, remaining gaps), `docs/parity-harness.md` (final commands),
  `Cargo.toml` + `macros/Cargo.toml` versions if releasing

## Implementation Steps

1. Report sections: scope, how to reproduce (three commands), capture matrix table, live
   matrix table, waivers with reasons, divergences fixed (scenario → commit), divergences
   kept (with the trade-off), spend, unresolved questions.
<!-- Updated: Validation Session 1 - release trigger covers phase-3 and phase-4 fixes -->
2. Version: patch (`0.24.3`) if fixes are internal; minor (`0.25.0`) if any public contract
   changed. Fixes from phase 3 and from phase-4 live divergences both count. No release if
   nothing under `src/` changed (harness-only commits still land).
3. Release protocol: gate → TLS smoke → secrets scan → commits (feat/fix, test, docs) →
   `cargo publish` macros first, then `cargo publish --all-features` → local tag → confirm
   the crates.io API lists the version and docs.rs `status.json` is green for both crates.
4. Do not push to GitHub (not requested); tags stay local.

## Success Criteria

- [ ] Report exists, links resolve, numbers match `parity/out/` verdict lines.
- [ ] Changelog and roadmap updated with dates and the version.
- [ ] If released: crates.io shows the version; docs.rs build green; `git tag` lists it.
- [ ] Working tree clean.

## Risk Assessment

- docs.rs `status: false` can mean "still building"; poll again before calling it a
  failure (already learned once).
- A late divergence found while writing the report → go back to phase 3 for that one item
  rather than shipping a known regression.
