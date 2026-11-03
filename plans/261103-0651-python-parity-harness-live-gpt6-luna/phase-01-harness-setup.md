---
phase: 1
title: Harness setup
status: completed
priority: P1
dependencies: []
---

# Phase 1: Harness setup

## Overview

Build the capture server, normalizer, runner and the Python environment, then prove the
pipeline with the `default_model` scenario end to end: both SDKs hit the server, both
records exist, the diff is empty (or shows only the expected `user-agent` line).

## Requirements

- Functional: one command runs a scenario in both SDKs and prints a verdict.
- Non-functional: no token spent; no secret printed or written; the user's Python checkout
  is not modified; works with `cargo test` running scenarios concurrently.

## Architecture

```
parity/run.sh capture <scenario>
  ├─ start capture_server.py on a free port  → PORT, PID
  ├─ Python: uv run ... parity/scenarios/<s>/py.py
  │     env OPENAI_BASE_URL=http://127.0.0.1:PORT/<s>/py/v1  OPENAI_API_KEY=sk-parity-dummy
  ├─ Rust:   cargo test --test parity_capture <s> -- --exact   (one process per scenario)
  │     env OPENAI_BASE_URL=http://127.0.0.1:PORT/<s>/rs/v1  OPENAI_API_KEY=sk-parity-dummy
  │         OPENAI_AGENTS_PARITY_CAPTURE=1
  ├─ stop server
  └─ compare.py <s>  → parity/out/<s>/diff.txt, verdict line, exit code
```

Capture server request handling (`/<scenario>/<sdk>/v1/<path>`):

- Append `{"path", "method", "headers": {subset, authorization: "present"|"absent"}, "body"}`
  to `parity/out/<scenario>/<sdk>.jsonl`.
- Load `parity/scenarios/<scenario>/responses.json`; pick entry by this SDK's request count
  for that path family (`responses`, `chat/completions`, `traces/ingest`).
- Entry shapes: a Responses object; a Chat Completions object; `{"status": 429, "body": {...}}`;
  `{"status": 204}` for traces ingest.
- `body.stream == true` → serialize the object as SSE using the same event list as
  `openai_agents_sdk::testing::response_to_stream_events` (`response.created`,
  `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`,
  `response.output_text.done`, `response.content_part.done`, `response.output_item.done`,
  `response.completed`; function-call items use `response.function_call_arguments.delta/done`).
  Chat Completions streams send one `chat.completion.chunk` per content piece then `[DONE]`.
- Missing entry → HTTP 500 with a clear body (the scenario script is wrong, not the SDK).

## Related Code Files

- Create: `parity/capture_server.py`, `parity/compare.py`, `parity/run.sh`,
  `parity/waivers.json`, `parity/scenarios/default_model/{py.py,responses.json}`,
  `tests/parity_capture.rs` (with the `default_model` test only), `docs/parity-harness.md`
- Modify: `.gitignore` (add `parity/out/`)
- Reference: `tests/sandbox_docker_live.rs` (env-gate style),
  `tests/py_core15_r15d_models.rs:65-68` (`OpenAIClient::with_base_url`),
  `src/testing.rs:171` (`response_to_stream_events`),
  `/Users/ngocthach/myGit/openai-agents-python/src/agents/models/openai_provider.py:165`

## Implementation Steps

1. Python env: `UV_PROJECT_ENVIRONMENT=$HOME/.cache/oars-build/py-venv uv sync --project
   /Users/ngocthach/myGit/openai-agents-python --python 3.13 --all-extras` (put the command in
   `parity/run.sh setup`; never type `.venv` in a shell command). Confirm
   `uv run ... python -c "import agents; print(agents.__file__)"` points at the checkout.
2. Write `parity/capture_server.py` as above. Flags: `--port 0 --out parity/out --scenarios
   parity/scenarios`; print the chosen port on stdout as the first line.
3. Write `parity/compare.py`: load both jsonl files, pair requests by index, normalize,
   diff (`difflib.unified_diff` of pretty-printed canonical JSON), apply `waivers.json`
   (`{"<scenario>|*": {"<json-pointer>": "reason"}}`), print `PARITY <scenario> OK|DIFF n`,
   write `diff.txt`, exit 1 on unwaived diffs. Different request counts are a diff.
4. Write `parity/run.sh` with subcommands `setup`, `capture <scenario|all>`, `live
   <scenario|all>` (live wired in phase 4). Absolute paths for both repos; export
   `CARGO_TARGET_DIR=$HOME/.cache/oars-build/parity`; `OPENAI_AGENTS_DISABLE_TRACING=1`
   unless the scenario is `tracing_export`; trap to stop the server.
5. Rust side: `tests/parity_capture.rs` with an `enabled()` helper that returns early unless
   `OPENAI_AGENTS_PARITY_CAPTURE=1`. Tests build agents exactly like the examples and use
   the default provider, so `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `get_default_model` come
   from the process env the runner set. The `default_model` test:
   `Agent::new("Assistant").with_instructions("You only respond in haikus.")`,
   `Runner::run(agent, "Tell me about recursion in programming.")`. `run.sh` launches one
   `cargo test --test parity_capture <scenario> -- --exact` per scenario.
6. Python side `parity/scenarios/default_model/py.py`: same agent, `Runner.run`; the model
   provider reads `OPENAI_BASE_URL` from env. `responses.json`: one message output "ok".
7. Run `bash parity/run.sh capture default_model`. Expect: two jsonl files with one request
   each, diff limited to `user-agent` (waive it with the reason "SDK identity").
8. Write `docs/parity-harness.md`: what the harness is, the three commands, where results
   land, how to add a scenario (3 files), the waiver format.

## Success Criteria

- [ ] `bash parity/run.sh capture default_model` ends with a `PARITY default_model OK|DIFF n`
      verdict line; `parity/out/default_model/{py,rs}.jsonl` both have 1 line and
      `diff.txt` exists.
- [ ] Both bodies contain `model: "gpt-5.6-luna"` and identical `instructions` and `input`.
      Any other body difference (e.g. Python's explicit `"stream": false`) is logged as a
      phase-2 row, not a phase-1 failure; only `user-agent` is waived here.
- [ ] `git status` shows no files under `parity/out/`; `git grep -n 'sk-' parity tests`
      finds only `sk-parity-dummy`.
- [ ] The user's Python checkout has no new untracked files (`git -C <py-repo> status`).
- [ ] Gate script still passes (new test file compiles with all feature sets and is a
      no-op without the env var).

## Risk Assessment

- Python 3.14 is the system default and the SDK is tested up to 3.13 → pin `--python 3.13`.
- The Python `openai` client appends `/responses` to `base_url`; if a trailing slash makes a
  double slash, strip it in `run.sh` (Rust trims both sides already).
- SSE fidelity: keep the event list identical to `response_to_stream_events`; the streaming
  scenario in phase 2 is the check.
- Hook blocks: run everything through `bash parity/run.sh ...`; never put the build-dir
  word in a typed command.
