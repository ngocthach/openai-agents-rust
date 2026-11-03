# Parity harness

Compares this crate with the Python SDK (`/Users/ngocthach/myGit/openai-agents-python`,
0.24.0) feature by feature. Two modes:

- **capture** — both SDKs talk to a local stand-in server through `OPENAI_BASE_URL`; the
  server records every request body (the "final prompt") and answers from a script, so both
  SDKs see the same model output. The normalized requests are diffed. No API call is made.
- **live** — a curated set of scenarios runs against the real API in both SDKs and the
  behaviour shape is compared (tools called, handoff target, parsed output, usage, event
  sequence); free text is recorded but never asserted on.

## Commands

```sh
bash parity/run.sh setup                  # once: install the Python SDK (uv, Python 3.13)
bash parity/run.sh capture default_model  # one scenario, no API call
bash parity/run.sh capture all            # every scenario under parity/scenarios/
bash parity/run.sh live hello             # one live scenario (real key from .env)
PARITY_MODEL=gpt-5.6-luna bash parity/run.sh live all   # every scenario under parity/scenarios/live/
```

Live mode loads `OPENAI_API_KEY` from the repo's `.env` inside the runner only, uses
`PARITY_MODEL` (default `gpt-6-luna`) and `max_output_tokens` 400 per call, and runs the
Rust tests one at a time. Results land in `parity/out/live/<scenario>/`: `py.json` and
`rs.json` (the shape records), `verdict.txt` (failed assertions, empty on `OK`).

A live scenario is `parity/scenarios/live/<name>/py.py` (writes its record through
`parity/shape.py`), `expect.json` (assertions evaluated on each record, and the paths that
must be equal across SDKs; see the docstring of `parity/compare_live.py`) and a
`#[tokio::test] async fn <name>()` in `tests/parity_live.rs` writing the same keys.

The Python environment is `~/.cache/oars-build/py-venv` (`UV_PROJECT_ENVIRONMENT`), the Rust
artifacts go to `~/.cache/oars-build/parity`. Capture mode uses `OPENAI_API_KEY=sk-parity-dummy`
and disables tracing except for the `tracing_export` scenario.

## Results

`parity/out/<scenario>/` (gitignored):

- `py.jsonl`, `rs.jsonl` — one recorded request per line: method, path, headers
  (authorization value replaced by `present`), body.
- `diff.txt` — the unified diff after normalization and waivers; empty when the verdict is
  `OK`.
- `server.log` — the capture server's log (one level up, `parity/out/server.log`).

The verdict line is `PARITY <scenario> OK` or `PARITY <scenario> DIFF <lines>`.

## Adding a scenario

Three files, all named after the scenario:

1. `parity/scenarios/<name>/responses.json` — the scripted replies, in order, one per model
   request (a minimal Responses object: `{"output": [...]}`; the server fills ids, usage and
   the other required fields; `{"status": 429, "body": {...}}` for an error reply).
2. `parity/scenarios/<name>/py.py` — the Python side, copied from the upstream example or
   integration test it mirrors.
3. A `#[tokio::test] async fn <name>()` in `tests/parity_capture.rs` that starts with
   `if !enabled() { return; }` and builds the same agent with the same prompts, tool names and
   descriptions.

Keep prompts and names byte-identical between the two sides; the diff is only useful when the
inputs are the same.

## Waivers

`parity/waivers.json` maps JSON pointers into a recorded request to a reason. `*` applies to
every scenario; a scenario name applies to that one. Pointers support `*` for any list index
or key and a trailing `*` on a key for a prefix match. Every entry needs a reason.

The normalizer itself drops transport-only headers (`content-length`, `host`,
`accept-encoding`, `connection`, `accept`), masks ids (`resp_…`, `msg_…`, `call_…`, …) and
`created_at`, sorts the order-free `required` and `include` arrays, and treats an empty
top-level `tools` or `include` list as absent (the Python client sends `[]` where the Rust
client omits the field; the API reads both the same way).
