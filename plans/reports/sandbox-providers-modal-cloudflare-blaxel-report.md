# Cloud sandbox providers: Modal, Cloudflare, Blaxel

Port of `agents.extensions.sandbox.{modal,cloudflare,blaxel}` onto the Rust core sandbox API
(`src/sandbox/**`). Date: 2026-10-29.

## What landed

| File | Contents |
| --- | --- |
| `src/extensions/sandbox/cloudflare/{mod,sandbox,sse,pty}.rs`, `mounts.rs` | Worker-backed client/session, SSE decoders, WebSocket PTY, R2/S3/GCS bucket mounts |
| `src/extensions/sandbox/blaxel/{mod,sandbox,mounts,pty}.rs` | Control-plane + sandbox-API client/session, WebSocket PTY, FUSE bucket mounts, Drive mounts |
| `src/extensions/sandbox/modal/{mod,sandbox,mounts}.rs`, `helper.py` | Helper-driven client/session with PTY, cloud bucket mounts, bundled Python helper |
| `tests/sandbox_modal_cloudflare_blaxel_tests.rs` | 17 tests against a recording mock HTTP server and a fake Modal helper |
| `tests/fixtures/fake_modal_helper.py` | Protocol-compatible helper double that logs every request |

The one-line stubs `cloudflare.rs`, `blaxel.rs` and `modal.rs` were replaced by directory modules;
`src/extensions/sandbox/mod.rs` was not touched. Phase 2 added the authorized additive core
changes listed below and no others; the E2B/Daytona/Vercel modules were not touched.

## Cloudflare

The Python port talks to a deployed Worker over HTTP, so the protocol is reproduced exactly with
`reqwest`:

- `POST {worker}/v1/sandbox` → `{"id"}`; `DELETE {worker}/v1/sandbox/{id}` on shutdown.
- `POST …/{id}/exec` with `{argv, timeout_ms}`, answered as SSE (`stdout`/`stderr` base64,
  `exit` → `{"exit_code"}`, `error`). The `_SSELineDecoder` / `_SSEDecoder` pair is ported
  faithfully, including the CR-at-end-of-chunk rule, and the response is consumed as a stream.
- `GET`/`PUT …/{id}/file/{path}` (paths percent-encoded with `/` kept literal), `GET …/running`,
  `POST …/persist?excludes=…`, `POST …/hydrate`, `POST …/mount`, `POST …/unmount`.
- Bodies that arrive SSE-encoded (`metadata`/`chunk`/`complete`, `isBinary`) are decoded by the
  same `decode_streamed_payload` logic Python uses, for both file reads and `persist`.
- PTY: `wss://{worker}/v1/sandbox/{id}/pty?cols=&rows=`. The session waits for the worker's
  `{"type":"ready"}` frame (30 s), writes the command as raw bytes, streams binary frames as
  output and ends on `{"type":"exit","code":N}` or `{"type":"error"}`, matching Python frame for
  frame. `supports_pty()` is now `true`.
- Auth: `Authorization: Bearer …` from options or `CLOUDFLARE_SANDBOX_API_KEY`, on the WebSocket
  handshake as well.
- The manifest root is pinned to `/workspace`, as in Python, because persist/hydrate are fixed to
  it. The worker's exec body carries no `cwd` or `env`, so the manifest environment becomes a
  sorted `env K=V …` prefix and an explicit `cwd` becomes a
  `sh -c 'cd -- "$1" && shift && exec "$@"'` wrapper.

## Blaxel

Python wraps `blaxel.core.sandbox.SandboxInstance`; this port calls the two HTTP surfaces that SDK
uses (confirmed against `blaxel-ai/sdk-python` sources and docs.blaxel.ai):

- Control plane `https://api.blaxel.ai/v0`, headers `X-Blaxel-Authorization: Bearer …` and
  `X-Blaxel-Workspace` (the SDK's own headers, not the `Authorization` header in some docs):
  `POST /sandboxes?createIfNotExist=true`, `GET /sandboxes/{name}`, `DELETE /sandboxes/{name}`,
  `POST /sandboxes/{name}/snapshots`, `POST /sandboxes/{name}/snapshots/{id}/restore`.
- Sandbox API at `metadata.url`, falling back to `https://run.blaxel.ai/{workspace}/sandboxes/{name}`:
  `POST /process` (`{command, workingDir, waitForCompletion, timeout, env}`),
  `GET /filesystem/{path}` (octet-stream for reads, JSON `Directory` for `ls`),
  `PUT /filesystem/{path}` (multipart for writes, `{"isDirectory":true}` for `mkdir`),
  `DELETE /filesystem/{path}?recursive=`, `POST /drives/mount`, `DELETE /drives/mount/{path}`.
- Env: `BL_API_KEY`, `BL_WORKSPACE`, `BL_REGION` (default `us-pdx-1`), plus base-URL overrides for
  tests. `pause_on_exit` keeps the sandbox and makes `resume` reconnect without re-materializing.
- Snapshots use Python's flow: `tar` into `/tmp/bl-persist-<sandbox>.tar`, download, delete.
  Ephemeral entries and mounts are excluded. Scratch paths in `/tmp` use absolute-path file
  helpers, since `normalize_path` (correctly) rejects anything outside the workspace.
- `reqwest`'s `multipart` feature is not enabled by `sandbox`, so the `multipart/form-data` body
  (`permissions`, `path`, `file`/`binary-file.bin`) is assembled by hand, matching the SDK's
  `write_binary`.
- PTY: `wss://{sandbox}/terminal/ws?token=&cols=&rows=&sessionId=&workingDir=`, with JSON frames
  in both directions (`{"type":"input"}` out, `{"type":"output"}`/`{"type":"error"}` in) and
  percent-encoded query values, as in Python. The terminal reports no exit code, so a closed
  stream simply ends the session. `supports_pty()` is `true` whenever an API key is configured,
  because the token travels in the query string.
- Mounts: `BlaxelCloudBucketMountStrategy` installs and runs `s3fs`/`gcsfuse` inside the sandbox,
  writing a `chmod 600` credential file that is removed whether or not the mount came up, and
  reports `MountCredentialAuthority::MountScoped` when credentials are present, so the manifest
  must acknowledge the mount path. `BlaxelDriveMountStrategy` uses the drives API.

## Modal: why a helper process

Modal's client speaks gRPC to its control plane and publishes no REST surface — `Sandbox.create`,
`Sandbox.exec` and `Sandbox.snapshot_filesystem` exist only in Modal's own SDKs, and the `modal`
CLI has no sandbox lifecycle commands (checked against `modal-labs/modal-client`
`py/modal/sandbox/_sandbox.py` and a web search for a public REST API). Three options were on the
table: reimplement the gRPC protocol (large, undocumented, version-fragile), shell out to the CLI
(no commands for this), or drive the official Python SDK. The port drives the SDK through a
bundled helper (`helper.py`, embedded with `include_str!` and materialized into a content-hashed
temp file):

- One JSON request on stdin, one JSON response on stdout, one process per operation
  (`create`, `exec`, `read_file`, `write_file`, `poll`, `terminate`, `snapshot_filesystem`, `pty`).
  The `pty` op takes its request as the first command-line argument instead, because a PTY's stdin
  belongs to the terminal. Each call re-resolves the sandbox by id, so invocations are stateless
  and safe concurrently.
- The source is handed to the interpreter as `python3 -c <source>`; nothing is written to a shared
  temporary directory, where a predictable path would be an executable another user could plant
  and this process would run it with the Modal tokens in its environment.
- Cost: one interpreter start-up plus `import modal` per operation.
  `ModalSandboxClientOptions::with_helper_command` (or `OPENAI_AGENTS_MODAL_HELPER`) swaps in a
  long-lived bridge or a test double; `with_python` / `OPENAI_AGENTS_MODAL_PYTHON` pick the
  interpreter. Credentials stay in the environment (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
  `MODAL_ENVIRONMENT`) and are never written to disk by the helper.
- PTY is real here: `pty_exec_start` runs the helper's `pty` op under the core `PtyManager`, so the
  model gets a terminal over Modal's own `pty=True` exec. The helper process is only a proxy, so
  the command is wrapped as `sh -c 'echo $$ > <pid>; exec <argv>'` and the session's
  `remote_kill_command` kills that pid inside the sandbox on teardown — without it, killing the
  local helper would leave a dev server or a `sleep` running in Modal until the sandbox expired.
- Snapshots: `tar` mode is the core flow; `snapshot_filesystem` mode calls Modal and stores the
  `MODAL_SANDBOX_FS_SNAPSHOT_V1\n{…}` envelope Python uses, and hydrating such an envelope boots a
  replacement sandbox from that image and terminates the old one.
- An exec timeout terminates the sandbox, as in Python, because Modal cannot cancel a running
  command — and, also as in Python, a command *without* an explicit timeout runs unbounded, so an
  ordinary long command can never retire the sandbox by outliving a default.
- The `tar` hydrate path stages the archive through the helper's chunked `write_file` op and then
  extracts it, rather than passing base64 as a command argument, which would exceed the kernel's
  128 KiB per-argument limit for any real workspace. Neither the extraction nor its cleanup is
  bounded by a timeout.
- A bucket mount not built with `ModalCloudBucketMountStrategy::mount` cannot be read back by the
  client, so it would silently never attach; `activate` rejects it instead.
- Bucket mounts are attached at creation, so `ModalCloudBucketMountStrategy::mount` registers its
  configuration and the client collects the `volumes` list from the manifest; `activate` is a
  no-op that only validates.

## Core changes made in phase 2 (all additive)

| Change | File | Why |
| --- | --- | --- |
| `MountStrategy::as_any` and `SandboxSession::as_any`, both defaulting to `None` | `mounts.rs`, `session.rs` | A mount strategy only receives a `&dyn SandboxSession` but needs the provider's transport; Modal's client needs to read a bucket configuration back out of a `dyn MountStrategy`. The `Option<&dyn Any>` shape with a `None` default keeps every backend that does not override it compiling unchanged. |
| `PROVIDER_SCRATCH_PREFIX`, `provider_scratch_path`, `is_provider_scratch_path` | `workspace_paths.rs` | One SDK-owned place for files a backend stages outside the workspace. The stem is reduced to `[A-Za-z0-9._-]` and the recognizer rejects anything holding a `/`, so nothing under the prefix can point elsewhere. |
| `SandboxSession::resolve_internal_path`, defaulting to the scratch check then `normalize_path` | `session.rs` | The escape hatch itself. Model-provided paths still go through `normalize_path`, so the workspace guard the model faces is unchanged. |
| `ExecRequest::env`, `with_env`/`with_envs`, `resolved_env` | `session.rs` | A per-command environment, merged over the manifest's so a command's own value wins. Honored by `docker` (`-e`), `unix_local` (process env), Cloudflare (`env` prefix), Blaxel (`env` map) and Modal (the helper's per-exec `env`). The two existing `ExecRequest { … }` literals use `..request.clone()` spreads, so the field breaks nothing; E2B, Daytona and Vercel simply do not read it yet, which is a one-line merge each for their owner. |
| `RemotePtyProcess` and `RemotePtyManager` | `pty.rs` | `PtyManager` owns a local pty and its child. A hosted terminal arrives as a stream the SDK does not own, so these provide the same ids, yield windows, limits and pruning for a provider-driven transport, and both WebSocket providers share them instead of duplicating the logic. Their `collect` carries an incomplete UTF-8 tail *within* a window, because a frame splits characters far more often than a local pty read does. |

Core files touched, all additively: `session.rs`, `mounts.rs`, `pty.rs`, `workspace_paths.rs`,
`mod.rs`, and — for the `env` field only — `sandboxes/docker.rs` (`docker exec -e`) and
`sandboxes/unix_local.rs` (process environment). Those last two change backend behaviour, though
only for a command that sets `ExecRequest::env`, which nothing did before.

Both connection registries and the Modal `Arc`-identity registry are gone, replaced by `as_any`
downcasts. Blaxel's and Modal's scratch files now use `provider_scratch_path` and go through
`resolve_internal_path`.

No type in the tree wraps another `SandboxSession` today (every `impl SandboxSession for` is a
concrete backend), so the downcast is safe; a future delegating wrapper must forward `as_any` to
the session it holds, which its doc comment now says.

### Notes

- **Session ids.** Cloudflare, Blaxel and Modal mint their own session id (as Python does) and
  record the provider's id in `SandboxSessionState::provider_id`. Worth knowing because a backend
  keyed on `id()` being the provider's id would be wrong for these three.
- **Modal needs Python.** Every operation shells out to an interpreter with the `modal` package,
  because Modal publishes no REST API. `with_helper_command` swaps in a long-lived bridge when
  that per-call start-up matters.

## Not ported

Exposed-port resolution (`ExposedPortEndpoint` has no core equivalent), Modal's
`snapshot_directory` persistence mode, Blaxel previews/tokens, Cloudflare's bounded-read wire
limits, and Python's retry/instrumentation decorators.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/cloud2
rustfmt --edition 2024 <all new files>                        # clean
cargo clippy --features sandbox --all-targets -- -D warnings  # clean
cargo test --features sandbox --test sandbox_modal_cloudflare_blaxel_tests  # 21 passed
cargo test --features sandbox --lib sandbox                   # 171 passed (core + all providers)
cargo test --features sandbox --test sandbox_tests            # 45 passed (unchanged)
cargo test --features sandbox --test sandbox_e2b_daytona_vercel_tests  # 19 passed (unchanged)
python3 compile check on both helper scripts                  # clean
```

Tests cover, per provider: request paths, auth headers, request bodies, response parsing, one
end-to-end flow (create → start → exec → read/write → persist/hydrate → shutdown), error mapping
(404 → `WorkspaceReadNotFound`, worker 503 → `ExecTransportError`), mounts, and for Modal the
snapshot round-trip, the timeout-terminates-the-sandbox rule, that a command without a timeout
runs unbounded, that the hydrate archive is staged rather than passed as an argument, and a PTY
session. Two tests run the real bundled helper (not the fake) to check that it compiles and that
it parses a command-line request. The Cloudflare SSE decoder is additionally exercised with a
stream split mid-`CRLF` across socket writes.

The WebSocket PTY paths are covered by unit tests inside the provider modules, which run a real
`tokio_tungstenite` server on a loopback socket: command delivery (including the `cd` prefix),
output collection, `write_stdin` with and without a TTY, an unknown session id, exit-code
reporting, teardown, and a terminal that closes before it is ready. They live in `src/` rather
than `tests/` because `tokio-tungstenite` is a dependency of the crate, not a dev-dependency, so
an integration test cannot run the server side. `RemotePtyProcess`/`RemotePtyManager` have their
own core tests, including the split-character carry and pruning of finished sessions. The Modal
PTY test additionally asserts that teardown kills the sandbox-side process by the recorded pid.

No test performs real network I/O or uses a real credential.

Status: DONE_WITH_CONCERNS
Summary: All three providers are complete, including PTY for all three (Cloudflare and Blaxel over
their WebSocket terminals, Modal through its helper, with the sandbox-side process killed on
teardown), plus all four authorized additive core changes — `as_any`, `ExecRequest::env`, the
scratch-path hatch and a shared remote-PTY manager — and every provider-local registry workaround
deleted; 21 integration tests, 171 core and provider unit tests and the neighbouring sandbox suite
all pass, with my files clippy-clean.
Concerns/Blockers: `ExecRequest::env` is now a public field that the E2B, Daytona and Vercel
sessions silently ignore (docker, unix_local, Cloudflare, Blaxel and Modal honor it). It defaults
to empty so nothing regresses, but a caller using `with_env` against those three would be misled —
please route the one-line merge to that agent.
