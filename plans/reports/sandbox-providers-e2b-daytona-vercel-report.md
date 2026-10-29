# Cloud sandbox providers: E2B, Daytona, Vercel

Port of `agents.extensions.sandbox.{e2b,daytona,vercel}` from the Python `openai-agents` SDK to
the Rust crate `openai-agents-sdk` (feature `sandbox`).

Date: 2026-10-29

## Files

| Path | Contents |
| --- | --- |
| `src/extensions/sandbox/e2b/mod.rs` | `E2BSandboxClient`, `E2BSandboxClientOptions`, `E2BSandboxSession`, `E2BCloudBucketMountStrategy`, Connect-RPC framing, remote PTY registry, shared HTTP helpers |
| `src/extensions/sandbox/e2b/rclone.rs` | `RcloneMountPattern` plus the `rclone` / FUSE provisioning helpers, shared with Daytona |
| `src/extensions/sandbox/daytona.rs` | `DaytonaSandboxClient`, `DaytonaSandboxClientOptions`, `DaytonaSandboxResources`, `DaytonaSandboxSession`, `DaytonaCloudBucketMountStrategy`, single-part multipart encoder |
| `src/extensions/sandbox/vercel/mod.rs` | `VercelSandboxClient`, `VercelSandboxClientOptions`, `NetworkPolicy`, `VercelSandboxSession`, `VercelCloudBucketMountStrategy` |
| `src/extensions/sandbox/vercel/tar_gz.rs` | `ustar` writer and gzip framer for `fs/write` |
| `tests/sandbox_e2b_daytona_vercel_tests.rs` | 13 integration tests against a recording mock HTTP server |

`src/extensions/sandbox/{e2b,vercel}.rs` were replaced by directory modules; `pub mod e2b;` and
`pub mod vercel;` in `src/extensions/sandbox/mod.rs` are unchanged. Nothing under `src/sandbox/**`,
`Cargo.toml` or `src/lib.rs` was touched.

## Design

Python wraps each vendor's own SDK. Rust calls the vendors' public HTTP APIs with `reqwest`, so no
new dependency is needed. Each provider exposes a `<Provider>SandboxClient` implementing
`SandboxClient` and a `<Provider>SandboxSession` implementing `SandboxSession`.

Common decisions across the three:

- **Credentials** come from the same environment variables as Python, with an options override and
  a base-URL override for tests. An *explicitly empty* key or token is treated as a
  misconfiguration (`ErrorCode::SandboxConfigInvalid`) rather than falling back to the
  environment, so a test can assert the error without depending on the machine's environment.
- **`Debug` is hand-written** for every options struct and session, because `SandboxClient:
  Debug` would otherwise print the key. Secret values render as `<redacted>` and environment maps
  render as key lists only.
- **`SandboxSessionState` stays credential-free.** `provider_id` carries only the provider's
  resource identity; resume re-fetches whatever token the data plane needs. Provider options come
  from the client on resume, not from the persisted state (Python serialises them into its own
  `*SessionState` subclasses).
- **`id()` is a stable UUID** owned by the session, so it survives a sandbox replacement; the
  provider identity lives behind a `std::sync::RwLock` because `session_state()` is synchronous
  and a native snapshot restore swaps the backing sandbox.
- **Mutable provider identity**: E2B and Vercel can replace their sandbox during
  `hydrate_workspace`, and both kill/stop the sandbox they replaced so the account does not leak
  resources.
- **Exec user**: `ExecRequest::resolved_argv()` would prepend `sudo -u <name> --`. Where the
  provider takes the user out of band (E2B basic auth, Vercel `sudo`), the request is cloned with
  `user: None` first so the wrapper is not built twice.
- **Exec timeouts** are both sent to the provider and enforced locally with
  `tokio::time::timeout`, mapping expiry to `ErrorCode::ExecTimeout`. A request with no timeout
  sends the 24-hour "unbounded" sentinel from Python's timeout tables, because E2B defaults to 60s
  and Daytona to 10s otherwise.
- **Binary safety**: all three providers report command output as text (JSON strings or NDJSON
  log lines), so the default `persist_workspace` — which pipes `tar -czf -` through stdout —
  would corrupt archives. All three override `persist_workspace` / `hydrate_workspace` and move
  the bytes over the file API instead. `read_file` / `write_file` never shell out to `cat`.
- **Workspace-scoped paths**: the trait methods normalise paths through `normalize_path`, which
  rejects anything outside the workspace root. Archive handling needs `/tmp`, so each session has
  private `download_at` / `upload_at` / `remove_at` helpers that take an already-absolute path;
  the public methods normalise and delegate to them.
- **State polling** (Daytona `started`, Vercel `running`) is bounded and its interval is
  configurable, so the mock tests run in milliseconds.
- **`stop()` keeps what `shutdown()` releases**, matching the Docker backend: E2B `stop` pauses
  and `shutdown` kills (or pauses when `pause_on_exit`); Daytona `stop` stops the sandbox and
  `shutdown` deletes it (or stops it when `pause_on_exit`). A Vercel session stop is terminal, so
  both are the same there. A "released" guard makes `shutdown` idempotent and makes a `stop` after
  a `shutdown` a no-op, and a "paused"/"stopped" flag makes `shutdown` skip a second pause or
  stop: `POST /sandboxes/{id}/pause` documents a 409, and the `pause_on_exit` fallback from a
  failed pause is a kill, which would destroy a sandbox the caller asked to keep.
- **`mkdir` and `rm` follow `mkdir -p` / `rm -f` semantics.** `Manifest::apply` creates the
  workspace root again right after `start()` made it, and the core `rm` default is `rm -f`, so an
  existing directory or a missing path is not an error. E2B's envd reports these as the Connect
  codes `already_exists` and `not_found` (the vendor SDK's `make_dir` swallows the first and
  returns `False`), and Daytona's `files/folder` is treated the same way on HTTP 409. Vercel
  keeps no `mkdir` override at all, so the trait's `mkdir -p` default runs, exactly as Python's
  Vercel backend does. E2B's `Remove` always deletes recursively, so a non-recursive `rm` runs
  `rm -f` through `exec` instead: a model removing "a file" must not be able to wipe a tree.
- **Resume adopts the recorded workspace root** when the supplied manifest still carries the
  SDK-wide `/workspace` default. A provider that rewrites that default at create time (Vercel,
  and Daytona for `create(None)`) records the rewritten root in the manifest digest, so verifying
  an unrewritten manifest against it would always fail. A manifest that *did* choose a root is
  left alone, so the digest check still binds it.

### E2B

Two hosts, matching the published OpenAPI description (`https://docs.e2b.dev/openapi-public.yml`):

- Platform API `https://api.<domain>` with `X-API-Key`: `POST /v2/sandboxes`,
  `GET /sandboxes/{id}`, `DELETE /sandboxes/{id}`, `POST /sandboxes/{id}/pause`,
  `POST /v2/sandboxes/{id}/connect`, `POST /sandboxes/{id}/snapshots`.
- envd through the shared sandbox host `https://sandbox.<domain>`, with `E2b-Sandbox-Id`,
  `E2b-Sandbox-Port: 49983` and the per-sandbox `X-Access-Token`:
  `GET`/`POST /files`, the unary `filesystem.Filesystem/{ListDir,MakeDir,Remove,Move}` and
  `process.Process/{SendInput,SendSignal}` calls, and the server-streaming
  `process.Process/Start`.

`Start` uses Connect's enveloped framing. The module contains an encoder and an incremental
decoder (flag byte, big-endian length, JSON payload; `flags & 0x02` marks the trailer, which may
carry an `error`). `exec` buffers the stream; `pty_exec_start` keeps it open on a background task
that feeds a `PtyProcess` buffer, and the yield-window collector mirrors `sandbox::pty::collect`,
reusing `clamp_pty_yield_time_ms`, `resolve_pty_write_yield_time_ms`, the PTY id/limit constants
and `truncate_text_by_tokens`. `PtyManager` itself is local-process only and does not fit — which also means the core's new
`(update, cleanup)` return value does not apply here. The provider registries do the equivalent
themselves: both `pty_terminate_all` and the prune that enforces `PTY_PROCESSES_MAX` return the
sandbox-side pids and the session sends `SendSignal SIGKILL` for each, so an evicted process does
not keep running (nor keep its 24-hour stream open) inside the sandbox.

proto3 JSON omits zero values, so a missing `exitCode` on an end event is read as `0`. A stream
that closes *without* an end event is an `ExecTransportError` rather than a silent success,
because the exit code is then unknown.

Restoring a native snapshot goes back through the client's own create path with the snapshot as
the template, so the replacement sandbox keeps the configured timeout, metadata, network and
lifecycle options rather than a hand-written subset.

`workspace_persistence = Snapshot` takes a native snapshot
(`POST /sandboxes/{id}/snapshots`, `memory: false`) and stores a reference with Python's
`E2B_SANDBOX_SNAPSHOT_V1\n` magic; hydration creates a sandbox from that snapshot template.
`Tar` persistence pipes `tar -czf - … | base64 -w0` exactly as Python does, because envd returns
command output as text.

### Daytona

- Platform API `https://app.daytona.io/api` with `Authorization: Bearer`, plus
  `X-Daytona-Organization-ID` and `X-Daytona-Source`: `POST /sandbox`, `GET /sandbox/{id}`,
  `DELETE /sandbox/{id}`, `POST /sandbox/{id}/stop`.
- Toolbox API at `<toolboxProxyUrl>/<sandbox id>`, taken from the sandbox record (falling back to
  `<api>/toolbox/<id>/toolbox`), carrying the same headers: `process/execute`,
  `process/session[/…]`, `files`, `files/download`, `files/upload`, `files/folder`, `files/move`.

`exec` creates a throwaway toolbox session, runs the command with `runAsync: false`, then deletes
the session — the same shape as Python, and the only route that reports stdout and stderr
separately. The command is wrapped as `cd <cwd> && env -- K=V … <argv>` because the toolbox takes
one command string with no working directory or environment of its own. An older toolbox that
returns only a combined `output` is split by exit code, as Python does.

`files/upload-v2` (the current upload route in `daytona-toolbox-api-client` 0.226.1 — plain
`files/upload` no longer exists) takes `multipart/form-data` with one `file` part. `reqwest`'s
`multipart` feature is not enabled by the `sandbox` feature, so the module builds that part by
hand with a UUID boundary.

Resume starts a sandbox that a previous `stop()` left stopped or archived
(`POST /sandbox/{id}/start`) before waiting for `started`, as the Python SDK does; otherwise the
wait would simply time out. `pty_terminate_all` deletes the toolbox session, which is the only
command teardown the toolbox API exposes.

### Vercel

One base URL (`https://vercel.com/api`, overridable — the vendor SDK documents the same
`base_url` option for testing), `Authorization: Bearer`, and a `teamId` query parameter on every
request. Routes were read from the published `@vercel/sandbox` npm package (3.9.1):
`POST /v3/sandboxes` (or `/v2` when a runtime alias is pinned),
`GET|POST /v2/sandboxes/sessions/{id}[/cmd|/fs/read|/fs/write|/fs/mkdir|/stop|/snapshot]`.

`cmd` with `wait` and `logs` answers with NDJSON: the command record, `{stream, data}` log lines,
then the finished record carrying `exitCode`. A `{"stream":"error"}` line becomes
`ErrorCode::ExecTransportError`.

`fs/write` accepts only a gzipped tar whose entry names are relative to the `x-cwd` header. With
no `tar` or `flate2` dependency, `vercel/tar_gz.rs` writes `ustar` headers (including the
`prefix` split for long names and the `%06o\0 ` checksum) and frames the result as gzip with
DEFLATE *stored* blocks plus a CRC-32. A unit test extracts a generated archive with the system
`tar -xzf` to prove real compatibility. The uploads this produces are single small files; a
whole-workspace archive is produced by `tar` *inside* the sandbox and is already compressed, and
`hydrate_workspace` posts it to `fs/write` unchanged.

Vercel has no filesystem listing endpoint, so `ls`, `rm` and `mv` fall through to the trait's
`exec`-based defaults.

### Provider mounts

| Python | Rust |
| --- | --- |
| `e2b.mounts.E2BCloudBucketMountStrategy` | `e2b::E2BCloudBucketMountStrategy` |
| `daytona.mounts.DaytonaCloudBucketMountStrategy` | `daytona::DaytonaCloudBucketMountStrategy` |
| `vercel.mounts.VercelCloudBucketMountStrategy` | `vercel::VercelCloudBucketMountStrategy` |

E2B and Daytona both wrap the core `InContainerMountStrategy` with an `rclone` FUSE pattern after
a provider-specific preflight (FUSE capability check, `user_allow_other` in `/etc/fuse.conf`,
installing `rclone` / `fuse3`). Vercel runs `mount-s3`, which its runtime images ship.

All three report `MountCredentialAuthority::MountScoped` when they carry credentials and call
`session.authorize_mount_activation` before activating, so the manifest's credential-exposure
acknowledgement applies.

Credentials never reach a command line, because anything in argv is readable from the sandbox's
process table and, on Vercel, is recorded in the provider's command log. `rclone` reads them from
a `600` configuration file written through the session's file API; `mount-s3` reads them from a
`600` file that the mount command sources with `set -a` and removes before `mount-s3` runs, under
a `trap … EXIT INT TERM HUP` so a failure between staging and mounting still removes it; the Rust
side removes it too when the command never ran.
`ExecRequest` has no `env` field, which is why the file detour is needed (see *Requested core
changes*). Both files live under the workspace root, matching Python's
`RcloneMountPattern.config_file_path`, which also resolves relative paths inside the workspace —
the session's path guard rejects writes outside the root, so `/tmp` is not an option for them.
Both staged files live under the workspace root, so `workspace_excludes` adds
`./.openai-agents-rclone-*` and `./.openai-agents-mount-s3-*` to every provider's `tar --exclude`
list: a `persist_workspace` taken while a mount is attached can never capture them. The rclone
configuration file name is derived from a hash of the mount path, so two mounts in one workspace
cannot overwrite each other's credentials. Mount failures never echo the command back, because it
names the remote.

## Divergences from Python

1. **E2B `sandbox_type`** selects the default template name only (`base` vs
   `code-interpreter-v1`). Both Python SDK classes speak the same HTTP API, so there is no
   separate transport.
2. **Session state carries no provider options.** Python's `*SessionState` subclasses serialise
   the template, timeouts, resources and so on; the Rust `SandboxSessionState` is deliberately
   credential- and authority-free, so resume takes those from the client.
3. **Vercel `provider_id` is `"<sandbox name>:<runtime session id>"`**, because a Vercel sandbox
   needs both and the core state has one `provider_id` field.
4. **Daytona PTY** uses the session log and input endpoints rather than
   `process/pty/{id}/connect`, which is a websocket: `tokio-tungstenite` is not enabled by the
   `sandbox` feature. Output streams and `write_stdin` work; there is no terminal resize and no
   true TTY allocation. E2B has a real PTY; Vercel has none in Python either, so
   `supports_pty()` stays `false` there.
5. **Vercel exec user**: Python rejects any `user` argument. This port maps `root` onto the API's
   `sudo` flag (so the provider's own `mount-s3` strategy can run) and still rejects every other
   user with `ErrorCode::SandboxConfigInvalid`.
6. **`RcloneMountPattern` is narrowed.** Python's core pattern also supports an NFS mode, named
   remotes derived per session, and reading the configuration from a file already in the
   workspace. The Rust pattern supports FUSE with an explicit remote name and inline
   configuration. It also installs `rclone` from the image's package manager instead of
   downloading a release pinned by SHA-256, keeping the image's signed repositories as the trust
   root.
7. **Daytona's default workspace root** (`/home/daytona/workspace`) replaces the SDK default only
   when the caller supplies no manifest at all, as in Python; Vercel's replaces the SDK default
   whenever the manifest still carries it, also as in Python. Resume resolves the root the same
   way in both.
8. **Not ported** (same set across the three): exposed-port endpoint resolution and preview URLs,
   sandbox metrics, MCP server configuration (E2B), drives/forks (Vercel), OIDC token refresh
   (Vercel), and Python's retry/instrumentation wrappers.

## Requested core changes

None are required — everything compiles and passes against the current `src/sandbox/**`. Two
placements are awkward and would be better in the core, if `src/sandbox/**` is open for edits
later:

1. **`RcloneMountPattern` lives in `src/extensions/sandbox/e2b/rclone.rs` and `daytona.rs`
   imports it from there.** Python keeps it in `agents.sandbox.entries.mounts.patterns`, and
   `src/sandbox/mod.rs` currently documents the FUSE/`mount-s3`/`rclone` patterns as belonging
   "with those providers". Two providers now share one, so `src/sandbox/mounts.rs` is the natural
   home. Moving it would also let Blaxel/Cloudflare/Modal reuse it.
2. **Generic HTTP helpers (`transport`, `http_error`, `parse_json`, `expect_success`) are
   `pub(crate)` in `src/extensions/sandbox/e2b/mod.rs` and reused by `daytona.rs` and
   `vercel/mod.rs`.** They map `reqwest` failures and non-2xx responses onto `SandboxError`, and
   belong next to `src/sandbox/errors.rs`. The task scoped edits to the three provider files, so
   the alternative was duplicating them three times.

3. **`ExecRequest` has no `env` field.** A mount strategy that needs to hand a credential to a
   command has to stage it in a file and source it, because putting it in argv would expose it
   through `/proc` and through Vercel's command log. All three provider APIs accept a per-command
   environment (`process.ProcessConfig.envs`, the Daytona session wrapper, Vercel `cmd.env`), so
   `ExecRequest::with_env` would let the strategies pass secrets out of band and would also let
   `exec` override manifest variables per command.

A fourth, smaller note: `SandboxSession::persist_workspace` has a useful default that an override
cannot reuse, so each provider re-derives the `--exclude` list from `Manifest::ephemeral_paths()`.
A `pub fn workspace_exclude_patterns(manifest, root)` in `src/sandbox/session.rs` would remove
that duplication (the Rust version of it lives in `e2b::workspace_excludes`).

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/cloud1
rustfmt --edition 2024 <the five source files and the test file>
cargo clippy --features sandbox --lib --test sandbox_e2b_daytona_vercel_tests -- -D warnings
cargo test --features sandbox --test sandbox_e2b_daytona_vercel_tests   # 18 passed
cargo test --features sandbox --lib extensions::sandbox                 # provider unit tests
```

`cargo clippy --features sandbox --all-targets -- -D warnings` is clean for the whole workspace,
and `cargo test --features sandbox --lib extensions::sandbox` and
`--test sandbox_tests` both pass against the committed core sandbox phase 3 (the
`AgentsError::Sandbox`, `SandboxStreamedRun` and `PtyManager` cleanup changes do not reach these
providers, which keep their own remote PTY registries and never touch `PtyManager`).

### Test coverage

Every test points a provider's base-URL override at a `tokio::net::TcpListener` that records each
request (method, path, query, headers, body) and answers from a per-test closure. No network, no
real keys (`e2b_test`, `dtn_test`, `vercel_test`), and no `std::env::set_var` — tests pass the key
and base URL explicitly, which also keeps them safe to run in parallel. The mock supports
delayed, `Content-Length`-free streaming replies so the E2B PTY path can be exercised with a
stream that stays open past the first yield window.

- **E2B** — end-to-end create → start → write → exec → read → delete with assertions on
  `X-API-Key`, the create body (`templateID`, `timeout`, `metadata`, `envVars`, `autoPause`,
  `autoResume`), the envd routing headers and access token, and the Connect envelope of
  `process.Process/Start`; non-zero exit codes and 404 → `WorkspaceReadNotFound`; missing API key;
  PTY streaming plus `SendInput` (pid addressing, base64 payload, `pty.size`, `stdin: true`);
  resume through `/connect` including the manifest-digest mismatch; native snapshot reference.
- **Daytona** — end-to-end flow with assertions on bearer auth, the organization header, the
  toolbox proxy URL derivation, the `cd … && env -- …` command wrapping, and the multipart upload;
  the interactive session (`runAsync`, incremental log delivery, `/input`); the workspace archive
  round trip through `tar` plus `files/download` / `files/upload`.
- **Vercel** — end-to-end flow with assertions on bearer auth, the `teamId` query parameter, the
  create body, NDJSON parsing, and the `application/gzip` + `x-cwd` upload; polling a `pending`
  session to `running` and failing on a terminal state; native snapshot persist and restore
  (including stopping the replaced session); `root` → `sudo` and the rejection of other users.
- **Regressions** — `start` surviving an `already_exists` `MakeDir` on the second call; an E2B
  `rclone` cloud bucket mount activating end to end (configuration written inside the workspace
  under its derived per-mount name, `--read-only` passed, and the configuration text absent from
  the command line); `create` → `session_state()` → JSON round trip → `resume` for Vercel (which
  rewrites the default root), for Daytona created without a manifest, and for a Daytona sandbox a
  previous run left stopped, which must be started again before the toolbox answers.

Unit tests in the source files cover redacted `Debug` output, base-URL and template resolution,
Connect framing in both directions, proto3 zero-value handling, snapshot-reference round trips,
entry mapping, command wrapping, the stdout/stderr split, multipart encoding, `mount-s3` and
`rclone` argument construction, network-policy serialisation, CRC-32 against known vectors, the
`ustar` checksum, and extraction of a generated archive by the system `tar`.
