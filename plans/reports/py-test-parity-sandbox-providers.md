# Python->Rust test parity: sandbox providers (tests/extensions/sandbox/)

Status: PARTIAL. 162 new Rust tests, all green. About 560 of the 794 Python test cases are NOT yet
individually classified or ported (see "Remaining work"). Do not read the table as full coverage.

## Counting method
`ast` walk of each file counting `test_*` functions and multiplying by `len()` of each
`@pytest.mark.parametrize` list (script: scratchpad/count_sbx.py). "funcs" = `def test_`.

| python file | funcs | expanded | Rust tests added (file) | python cases ported (approx.) | python-only (grouped) | not yet classified |
|---|---|---|---|---|---|---|
| test_rclone.py | 6 | 15 | 6 (py_sandbox_rclone_mounts) | 15 | 0 | 0 |
| test_blaxel_mounts.py | 2 | 2 | 2 (same) | 2 | 0 | 0 |
| test_e2b.py | 76 | 80 | 43 (py_sandbox_e2b) | ~35 | ~10 (re-exports, import hooks, pydantic round-trips, `_FakeMountSession` guard) | ~35 |
| test_daytona.py | 63 | 72 | 41 (py_sandbox_daytona) | ~34 | ~8 (re-exports, session-guard, pydantic manifest parse, session-reader asyncio internals) | ~30 |
| test_cloudflare.py | 80 | 96 | 24 (py_sandbox_cloudflare) | ~35 | ~10 (SSE decoder unit tests live in src/.../sse.rs; aiohttp fakes) | ~51 |
| test_vercel.py | 104 | 130 | 16 (py_sandbox_vercel) | ~22 | ~10 (re-exports, from_state/deepcopy/traceback redaction) | ~98 (mostly mount lifecycle; partly COVERED by sandbox_e2b_daytona_vercel_tests::vercel_*) |
| test_vercel_provider_contract.py | 12 | 29 | 0 new (token-omission is in the vercel file) | ~2 | ~24 (monkeypatches installed `vercel.sandbox` SDK) | ~3 |
| test_modal.py | 110 | 119 | 12 (py_sandbox_modal) | ~13 | ~10 (re-exports, pydantic state defaults) | ~96 |
| test_blaxel.py | 227 | 251 | 18 (py_sandbox_blaxel) | ~22 | ~30 (re-exports, `_import_blaxel_sdk`, aiohttp availability, frozen/pydantic dataclass checks) | ~199 |
| **total** | 680 | 794 | 162 | ~180 | ~100 | ~510 |

Counts in the "approx." and "python-only" columns are estimates from reading the test names and bodies,
not an exact per-test ledger. Per-test mapping was not recorded.

Python-only reasons (grouped): re-export/import-path tests; monkeypatching `builtins.__import__`;
pydantic `model_validate`/`model_dump` and frozen-model checks; asyncio task/cancellation internals;
fixtures that replace the vendor SDK (`vercel.sandbox`, `aiohttp.ClientSession`).

## Rust bugs fixed (src/extensions/sandbox/**)
1. cloudflare/sandbox.rs: error `retryable` ignored the worker's documented status table. Now 400/500
   -> not retryable, 503 -> retryable, other -> unclassified (Python `_cloudflare_retryability_for_status`),
   for exec and persist/hydrate. Persist previously retried a 500; now only 503 is retried.
2. cloudflare/sandbox.rs: a 200 reply whose body is a JSON error (not an event stream) now reports
   `POST /exec returned non-SSE error body: <code: error>` instead of "ended without exit event".
3. vercel/mod.rs: exec errors for 408 and 425 were marked non-retryable; Python retries 408/425/429/5xx.
   Local `vercel_status_is_retryable` now applies the Python table for `cmd` failures.
4. vercel/mod.rs: when the tar command failed during persist, the scratch tarball was left in the sandbox;
   it is now removed on that path too (Python removes it).

## Cross-area / cross-cutting findings (not edited)
- src/sandbox/http.rs:34 `status_is_retryable`: lacks 408 and 425. Python Vercel (and the generic table)
  retries them. Minimal fix: add `|| matches!(status.as_u16(), 408 | 425)`. Vercel's `cmd` path is patched
  locally; vercel read/write/snapshot calls still go through the core helper.
- Archive error codes: Python raises `WorkspaceArchiveReadError`/`WorkspaceArchiveWriteError` from provider
  persist/hydrate; every Rust provider (and core unix_local/docker) uses `SnapshotPersistError`/
  `SnapshotRestoreError`. Changing it is a public-contract change spanning ~31 sites, so it was left
  alone and the ported tests assert the Rust codes.
- E2B and Daytona `rclone` install: Rust installs differently from Python's Daytona helper (distribution
  package vs pinned release); behaviour (rclone on PATH) matches, command sequence differs. Test documents it.
- Vercel allows `user=root` (maps to `sudo`); Python refuses every user. Existing deliberate deviation.

## Test premises that are Python-specific (no Rust change)
- Blaxel resume reconnects by name only with `pause_on_exit` (matches Python); `running()` probes the
  sandbox filesystem API.
- Daytona: `_ensure_fuse_support` probes `/dev/fuse` and the kernel module in two calls in Python, one
  combined probe in Rust.
- Modal: mount strategy validation runs when the manifest is applied at `start`, not at `create`.

## Remaining work
~510 Python cases are not yet ported or classified. Largest blocks: Blaxel (PTY, mounts, tar persist/hydrate
error paths, preview ports), Modal (snapshot filesystem/directory persistence, PTY, tar chunking),
Vercel (S3 mount lifecycle, which the existing sandbox_e2b_daytona_vercel_tests partly covers),
Cloudflare (resume/mount lifecycle, PTY websocket, redaction), E2B/Daytona (PTY + mount remount ordering).
A fork-per-provider split was attempted but blocked by the concurrent-subagent limit.

## Files
- tests/py_sandbox_support/mod.rs (shared mock HTTP server; `#[path]`-included, not a test crate)
- tests/py_sandbox_{rclone_mounts,e2b,daytona,cloudflare,vercel,modal,blaxel}.rs
- src edits: src/extensions/sandbox/cloudflare/sandbox.rs, src/extensions/sandbox/vercel/mod.rs
- Shared-file edits (lib.rs, Cargo.toml): none.

## Gates
- All 7 new test binaries green: 18+24+41+43+12+8+16 = 162 (18 is blaxel; 8 is rclone/blaxel_mounts);
  existing sandbox_e2b_daytona_vercel_tests (41) and sandbox_modal_cloudflare_blaxel_tests (37) still green.
- `cargo clippy --all-features` on my test targets: no warnings in tests/py_sandbox_* or
  src/extensions/sandbox/**. `-D warnings` on the lib fails in other areas' files
  (src/extensions/models/mod.rs:1016, src/models/openai_responses_ws.rs:553/556,
  src/sandbox/dependencies.rs:188, src/sandbox/sandboxes/unix_local.rs:1149), not touched.
- rustfmt run on my own files only.

Status: DONE_WITH_CONCERNS
Summary: Built a shared mock-server harness and 162 green Rust tests across all 9 Python files; fixed 4 provider bugs (cloudflare retry table, cloudflare non-SSE error body, vercel 408/425 retryability, vercel tar scratch cleanup).
Concerns/Blockers: only about 180 of 794 Python cases ported and about 100 classified python-only; about 510 remain unclassified. Cross-area: src/sandbox/http.rs status_is_retryable lacks 408/425. Archive error code deviation (SnapshotPersistError vs WorkspaceArchiveReadError) left unchanged.
