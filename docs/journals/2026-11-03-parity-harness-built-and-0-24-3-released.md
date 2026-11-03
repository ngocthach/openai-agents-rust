# 2026-11-03 — Parity harness built, three divergences fixed, 0.24.3 released

## What happened

Executed the five-phase plan in `plans/261103-0651-python-parity-harness-live-gpt6-luna/`.
A new `parity/` harness runs the same scenario through the Python SDK 0.24.0 and this crate.
In capture mode both talk to a local stand-in server (one process, routed by
`/<scenario>/<sdk>/v1`) that records the request bodies and answers from a script; the bodies
are normalized and diffed. In live mode both hit the real API with `gpt-6-luna` and a
behaviour-shape record is compared.

Results: 18/18 capture scenarios identical after normalization; 12/12 live scenarios match
across SDKs. Three Rust fixes shipped in 0.24.3 (agent-as-tool schema description, no
`code_interpreter_call.outputs` include, `RunResult::input` after a session run). An
independent review of 0.24.3 then found the session fix ignored later rewrites of the run
input (handoff input filters, nested handoff history, sandbox `process_input`); 0.24.4 fixes
that with a session + nested-handoff test. Both crates published at 0.24.3 and 0.24.4, tags
`v0.24.3`/`v0.24.4` local only, docs.rs builds pending at the time of writing.

## What was hard

- A gate failure came from my own fix: an existing test encoded the old description-less
  default schema. Reading Python's summary builder showed the new expectation (a summary built
  from the docstring description) is the upstream behaviour, so the test moved with the fix.
- The live model is non-deterministic in one specific way: it returns a `reasoning_item` before
  the message in some runs and not others. The first full pass flagged this on two scenarios;
  two re-runs flipped which side had the item, so the comparison now ignores reasoning items.
- `max_output_tokens` 400 cut a haiku run on both SDKs (identical `response.incomplete` error):
  the model spends output tokens on reasoning even at the default effort. Cap raised to 2000.
- crates.io's API refuses requests without a User-Agent, so the publish-and-poll loop spun for
  five minutes; the main-crate publish then refused the "dirty" tree because of the untracked
  `.idea/` folder. Fixed with a UA header and `--allow-dirty` only after confirming nothing else
  was dirty.

## Decisions

- Parity over preference: the Rust-only `include` entry and the description-less schema were
  removed/aligned even though both were arguably harmless.
- `RunState::original_input` was left alone (Rust's resume path reads it as the input to
  persist); only `RunResult::input` changed. Recorded as an open difference.
- Secrets: the real key only ever enters the two live child processes from `.env`; capture mode
  uses `sk-parity-dummy`; `parity/out/` is gitignored.
