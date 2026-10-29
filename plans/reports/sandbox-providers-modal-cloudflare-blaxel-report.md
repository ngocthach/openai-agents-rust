# Cloud sandbox providers: Modal, Cloudflare, Blaxel

Port of `agents.extensions.sandbox.{modal,cloudflare,blaxel}` onto the Rust core sandbox API
(`src/sandbox/**`). Date: 2026-10-29.

## What landed

| File | Contents |
| --- | --- |
| `src/extensions/sandbox/cloudflare/{mod,sandbox,sse}.rs`, `mounts.rs` | Worker-backed client/session, SSE decoders, R2/S3/GCS bucket mounts |
| `src/extensions/sandbox/blaxel/{mod,sandbox,mounts}.rs` | Control-plane + sandbox-API client/session, FUSE bucket mounts, Drive mounts |
| `src/extensions/sandbox/modal/{mod,sandbox,mounts}.rs`, `helper.py` | Helper-driven client/session with PTY, cloud bucket mounts, bundled Python helper |
| `tests/sandbox_modal_cloudflare_blaxel_tests.rs` | 17 tests against a recording mock HTTP server and a fake Modal helper |
| `tests/fixtures/fake_modal_helper.py` | Protocol-compatible helper double that logs every request |

The one-line stubs `cloudflare.rs`, `blaxel.rs` and `modal.rs` were replaced by directory modules;
`src/extensions/sandbox/mod.rs` was not touched. Nothing under `src/sandbox/**` was edited.

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
- Auth: `Authorization: Bearer …` from options or `CLOUDFLARE_SANDBOX_API_KEY`.
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
  model gets a terminal over Modal's own `pty=True` exec. This is the only one of the three
  providers with working PTY support.
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

## Core API notes (no changes made)

1. **`MountStrategy` and `SandboxSession` are not downcastable.** Cloudflare bucket mounts and
   Blaxel drives need the session's provider endpoint, and Modal's client needs to read a bucket
   configuration back out of a manifest entry. Both are solved locally: each session registers its
   connection in a module-private `Vec<(session_id, Weak<Conn>)>` and the strategy looks it up by
   `session.id()`; Modal keys its configurations by the registered `Arc`, compared with
   `Arc::ptr_eq`. An `as_any(&self) -> &dyn Any` on either trait would remove both registries.
2. **Session ids.** Cloudflare, Blaxel and Modal now mint their own session id (as Python does) and
   record the provider's id in `SandboxSessionState::provider_id`. Before that, keying the
   connection registry by the provider id let two live sessions for one sandbox name collide.
3. **`ExecRequest` has no `env`.** Each provider derives it from `manifest.environment()`; a
   first-class field would remove three different workarounds.
4. **PTY over WebSocket is not reachable.** Cloudflare's `/v1/sandbox/{id}/pty` and Blaxel's
   `/terminal/ws` are WebSocket endpoints and `tokio-tungstenite` is not enabled by the `sandbox`
   feature, so both sessions report `supports_pty() == false`. The exact change needed:
   `sandbox = ["dep:base64", "dep:portable-pty", "dep:sha2", "dep:tokio-tungstenite"]` in
   `Cargo.toml` (the dependency is already declared for `realtime`/`voice`). Hand-rolling WebSocket
   framing was deliberately not attempted.
5. **Absolute scratch paths.** Blaxel's snapshot flow and Modal's hydrate flow both need `/tmp`
   file access, which `normalize_path` rejects.
   Blaxel carries its own absolute-path file helpers and Modal calls its helper's `write_file`
   op directly. A core "outside the workspace, provider-internal" file path would be cleaner.

## Not ported

Exposed-port resolution (`ExposedPortEndpoint` has no core equivalent), Modal's
`snapshot_directory` persistence mode, Blaxel previews/tokens, Cloudflare's bounded-read wire
limits, and Python's retry/instrumentation decorators.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/cloud2
rustfmt --edition 2024 <all new files>                       # clean
cargo clippy --features sandbox --all-targets -- -D warnings  # clean
cargo test --features sandbox --test sandbox_modal_cloudflare_blaxel_tests  # 19 passed
cargo test --features sandbox --lib extensions::sandbox       # 70 passed
python3 compile check on both helper scripts                  # clean
```

Tests cover, per provider: request paths, auth headers, request bodies, response parsing, one
end-to-end flow (create → start → exec → read/write → persist/hydrate → shutdown), error mapping
(404 → `WorkspaceReadNotFound`, worker 503 → `ExecTransportError`), mounts, and for Modal the
snapshot round-trip, the timeout-terminates-the-sandbox rule, that a command without a timeout
runs unbounded, that the hydrate archive is staged rather than passed as an argument, and a PTY
session. Two tests run the real bundled helper (not the fake) to check that it compiles and that
it parses a command-line request. The Cloudflare SSE decoder is additionally exercised with a
stream split mid-`CRLF` across socket writes. No test
performs real network I/O or uses a real credential.

Status: DONE_WITH_CONCERNS
Summary: Modal, Cloudflare and Blaxel sandbox providers are ported with clients, sessions,
lifecycle, file and exec operations, mounts and snapshots, covered by 19 integration tests against
mock servers and a fake Modal helper plus 70 unit tests, all clippy-clean.
Concerns/Blockers: Cloudflare and Blaxel PTY need `dep:tokio-tungstenite` added to the `sandbox`
feature (WebSocket endpoints; `supports_pty()` is `false` until then — Modal's PTY works). Modal
necessarily depends on a Python interpreter with the `modal` package, because Modal publishes no
REST API. `as_any` on `MountStrategy`/`SandboxSession` would let two provider-local connection
registries be deleted.
