# Cloud sandbox providers: E2B, Daytona, Vercel

Port of `agents.extensions.sandbox.{e2b,daytona,vercel}` from the Python `openai-agents` SDK to
the Rust crate `openai-agents-sdk` (feature `sandbox`).

Date: 2026-10-29 (phase 1 + phase 2)

## Files

| Path | Contents |
| --- | --- |
| `src/extensions/sandbox/e2b/mod.rs` | `E2BSandboxClient`, `E2BSandboxClientOptions`, `E2BSandboxSession`, `E2BCloudBucketMountStrategy`, Connect-RPC framing, remote PTY registry, shared HTTP helpers |
| `src/extensions/sandbox/e2b/rclone.rs` | Provider-side `rclone` / FUSE provisioning, shared with Daytona: the pinned release download and the package-manager install |
| `src/extensions/sandbox/daytona.rs` | `DaytonaSandboxClient`, `DaytonaSandboxClientOptions`, `DaytonaSandboxResources`, `DaytonaSandboxSession`, `DaytonaCloudBucketMountStrategy`, single-part multipart encoder |
| `src/extensions/sandbox/vercel/mod.rs` | `VercelSandboxClient`, `VercelSandboxClientOptions`, `NetworkPolicy`, `VercelSandboxSession`, `VercelCloudBucketMountStrategy` |
| `src/extensions/sandbox/vercel/tar_gz.rs` | `ustar` writer and gzip framer for `fs/write` |
| `tests/sandbox_e2b_daytona_vercel_tests.rs` | 21 integration tests against a recording mock HTTP/WebSocket server |

Phase 2 also added two additive core modules, with the controller's authorization:

| Path | Contents |
| --- | --- |
| `src/sandbox/http.rs` (new) | `transport`, `http_error`, `parse_json`, `expect_success` — the HTTP-to-`SandboxError` mapping every hosted backend needs |
| `src/sandbox/mounts.rs` (appended) | `RcloneMountPattern`, `RcloneMountMode`, `RCLONE_CONFIG_DIR` and the remote-selection option keys |
| `src/sandbox/session.rs` (appended) | `workspace_archive_excludes`, now used by the default `persist_workspace` and by all three provider overrides |

`src/extensions/sandbox/{e2b,vercel}.rs` were replaced by directory modules; `pub mod e2b;` and
`pub mod vercel;` in `src/extensions/sandbox/mod.rs` are unchanged. `Cargo.toml` and `src/lib.rs`
were not touched. Phase 1 touched nothing under `src/sandbox/**`; phase 2's additions there are
listed above and are purely additive.

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
process table and, on Vercel, is recorded in the provider's command log. `mount-s3` receives them
through `ExecRequest::env`; `rclone` reads them from a `600` configuration file written through
the session's file API, because that is the only channel `rclone mount` offers. That file lives
under the workspace root (the session's path guard rejects writes outside it, so `/tmp` is not an
option) in `.sandbox-rclone-config/`, which `workspace_archive_excludes` keeps out of every
provider's `persist_workspace` archive. Its name is derived from a hash of the mount path, so two
mounts in one workspace cannot overwrite each other's credentials, and the mount's own copy is
always separate from any configuration the caller placed, so teardown never deletes the caller's
file. Mount failures never echo the command back, because it names the remote.

## Divergences from Python

1. **E2B `sandbox_type`** selects the default template name only (`base` vs
   `code-interpreter-v1`). Both Python SDK classes speak the same HTTP API, so there is no
   separate transport.
2. **Session state carries no provider options.** Python's `*SessionState` subclasses serialise
   the template, timeouts, resources and so on; the Rust `SandboxSessionState` is deliberately
   credential- and authority-free, so resume takes those from the client.
3. **Vercel `provider_id` is `"<sandbox name>:<runtime session id>"`**, because a Vercel sandbox
   needs both and the core state has one `provider_id` field.
4. **Daytona PTY** is now the real websocket terminal, with resize, plus the session log path
   for non-TTY processes — the same split Python makes. The only divergence left is the
   subprotocol fallback described under *Phase 2*. E2B has a real PTY; Vercel has none in Python
   either, so `supports_pty()` stays `false` there.
5. **Vercel exec user**: Python rejects any `user` argument. This port maps `root` onto the API's
   `sudo` flag (so the provider's own `mount-s3` strategy can run) and still rejects every other
   user with `ErrorCode::SandboxConfigInvalid`.
6. **`RcloneMountPattern` is at parity** on modes, flags, remote-name derivation, configuration
   sources, placement and teardown (see *Phase 3*). Daytona installs `rclone` from the
   distribution package rather than the pinned release, matching its own Python `mounts.py`.
7. **Daytona's default workspace root** (`/home/daytona/workspace`) replaces the SDK default only
   when the caller supplies no manifest at all, as in Python; Vercel's replaces the SDK default
   whenever the manifest still carries it, also as in Python. Resume resolves the root the same
   way in both.
8. **Not ported** (same set across the three): exposed-port endpoint resolution and preview URLs,
   sandbox metrics, MCP server configuration (E2B), drives/forks (Vercel), OIDC token refresh
   (Vercel), and Python's retry/instrumentation wrappers.

## Core changes made in phase 2

Everything phase 1 listed as a requested core change has landed, three of them in the core by the
other agent and three here:

1. **`RcloneMountPattern` moved to `src/sandbox/mounts.rs`**, where Python keeps it
   (`agents.sandbox.entries.mounts.patterns`). Blaxel, Cloudflare and Modal can reuse it now.
   What stayed in `e2b/rclone.rs` is only what Python keeps in `extensions/sandbox/_rclone.py`:
   installing the tool.
2. **The HTTP helpers moved to a new `src/sandbox/http.rs`**, next to `errors.rs`, with their own
   tests for the two provider error-body shapes and for the truncation that stops an HTML error
   page flooding a log line.
3. **`workspace_archive_excludes` moved to `src/sandbox/session.rs`** and is now the single
   source for both the default `persist_workspace` and the three provider overrides. It also adds
   `RCLONE_CONFIG_DIR`, so no backend — including Docker and unix-local, which can now use the
   rclone pattern — can archive a staged mount credential.
4. **`ExecRequest::env`** (added by the other agent) is honoured by all three providers, see
   below.
5. **`SandboxSession::as_any`** (added by the other agent) is implemented by all three sessions.
   The mount strategies use it to refuse a session from another backend, mirroring Python's
   `_assert_e2b_session` and its Daytona and Vercel equivalents, and the Daytona terminal resize
   is reachable through it from a `SandboxSession` trait object.
6. **`provider_scratch_path` / `resolve_internal_path`** (added by the other agent) replaced the
   private path escape hatches: a workspace archive is staged at an SDK-owned scratch path that
   the core validates, instead of a hand-built `/tmp` path.

## Phase 2: parity work

### Daytona PTY over the toolbox websocket

Python opens the PTY websocket only when the model asks for a writable terminal; a read-only
interactive process goes through the session log stream. Both paths are now ported, and the
provider half is the only thing the provider owns — session ids, the yield window, the output
UTF-8 carry and the `PTY_PROCESSES_MAX` pruning all come from the core `RemotePtyManager`, which
E2B uses too.

The terminal matches the vendor SDK's `create_pty_session`:

- `GET <toolbox>/process/pty/create-connect?id=&cols=&rows=&cwd=`, upgraded to `ws`/`wss`, with
  the REST credentials on the handshake.
- The sandbox environment travels as the `X-Daytona-Pty-Envs~<base64url>` subprotocol token, and
  `X-Daytona-Pty-Exit-Control` asks the daemon for an `exited` control frame.
- Output arrives as binary frames and as text frames that are not `{"type":"control"}`; the exit
  code arrives in the control frame or, failing that, in the JSON close reason (a clean 1000
  close with no reason means success).
- Input is sent as binary frames, with a 20-second ping so a silent terminal is not dropped by an
  intermediate proxy.
- `resize_terminal` posts `{cols, rows}` to `/process/pty/{id}/resize`; teardown is
  `DELETE /process/pty/{id}`.

**One deliberate divergence.** `tungstenite` fails a handshake with `NoSubProtocol` when the
client advertised subprotocols and the server selected none (verified in
`tungstenite-0.30.0/src/handshake/client.rs`); RFC 6455 permits that server behaviour and
`aiohttp`, which the vendor SDK uses, tolerates it. The Daytona daemon's source is no longer
public on GitHub, so this could not be settled by reading it. Rather than gamble, the client
retries once without any subprotocol and then exports the environment on the typed command line,
exactly as the non-terminal path does; the exit code still arrives in the close frame, which is
the documented behaviour for a client that does not send the exit-control token.

The retry asks for a **fresh session id**. By the time `tungstenite` rejects the response the
server has already answered 101, so `create-connect` has created that PTY session and started its
shell; the vendor documents that reusing an id is an error, and reusing it would also orphan the
first shell. So the fallback kills the first session and connects under a new id, and the session
records the id the connection actually used, so teardown targets the right one. The test's mock
answers a reused id with 409, which proves the retry does not reuse it.

### `ExecRequest::env`

All three backends now resolve a command's environment with the core's
`ExecRequest::resolved_env` (manifest first, per-command second) and express it natively:

- **E2B** sets `process.ProcessConfig.envs` on the `Start` request.
- **Vercel** sets `cmd.env`.
- **Daytona** has no environment parameter — the toolbox takes one command string — so it lands
  in the command as `env -- K=V …`, which is how Python already passes the manifest environment
  there. A PTY has no per-request environment in the core type, so a terminal runs with the
  session's.

The precedence is the client options' environment, then the manifest's, then the request's, which
is the order Python resolves per command; `ExecRequest::resolved_env` covers only the last two, so
each provider starts from its own session environment and extends it.

A command started without a TTY has no stdin on Daytona either now: `pty_write_stdin` with input
returns `ErrorCode::PtyStdinUnavailable`, matching Python and the E2B port. An empty write is
still a poll, which is how the model reads the rest of a logged command's output.

### Mount credentials

`VercelCloudBucketMountStrategy` now takes typed credentials and passes them to `mount-s3` as
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` and `AWS_REGION` through
`ExecRequest::with_envs` — the keys Python's `_runtime_s3_mount_environment` sets. The staged
credentials file and its shell trap are gone; nothing secret reaches argv, `/proc` or the
provider's command log, and the strategy's `Debug` redacts the secret.

### `RcloneMountPattern` parity

The core pattern now matches Python's: it only *checks* for `rclone`
(`ErrorCode::MountMissingTool`), writes its configuration to `.sandbox-rclone-config/<hash>.conf`
inside the workspace with mode `600`, supports both modes, and tears down with
`fusermount`/`umount` plus `pkill -f 'rclone (mount|serve nfs) <remote>:'`.

- **FUSE**: `rclone mount remote:path <path> [--read-only] --config <conf> --daemon [extra]`.
- **NFS**: a `rclone serve nfs --help` probe first (a build without it fails with
  `MountMissingTool`), then a backgrounded `rclone serve nfs … --addr`, then `mount -t nfs` with
  Python's default options (`vers=4.1,tcp,port=…,soft,timeo=50,retrans=1`), its three-attempt
  retry loop and its `timeout 10s` prefix when the image provides `timeout`.

The configuration directory is created before the credential is written, as Python does, so the
first mount cannot fail on a backend whose upload endpoint does not create parents.

`--allow-other` is added by the E2B and Daytona strategies in FUSE mode only, the way Python's
`rclone_pattern_for_session` does, rather than being baked into the pattern.

Installing the tool stayed with the providers, as in Python. E2B now downloads the pinned release
(`rclone v1.74.4`), verifies its published SHA-256 inside the sandbox and only then moves the
binary into `/usr/local/bin`; Daytona installs the distribution package with `apt-get` or `apk`.
A unit test pipes the generated install script through `sh -n`, because a quoting slip in a
shell script assembled from twenty pieces would otherwise only surface inside a sandbox.

### The three assumed HTTP behaviours, re-checked

| Assumption | Status |
| --- | --- |
| Vercel `fs/read` answers 404 for a missing path | **Verified** — `@vercel/sandbox` 3.9.1 `readFile` returns `null` on `response.status === 404`. |
| An E2B envd Connect error body carries a string `code` | **Verified** — `e2b` 2.62.0 `envd/client_shared.py` distinguishes a string `code` (a Connect error) from an integer one (a plain HTTP error page), and `sandbox_async/filesystem.py` `make_dir` branches on `Code.ALREADY_EXISTS`. The port tolerates the named codes, and a plain 409/404 maps the same way the vendor's `PLAIN_HTTP_ERROR_CODES` table does. |
| Daytona `files/folder` is idempotent | **Inferred, not verified.** The generated client documents only a 201 and the vendor SDK catches nothing, while the Python Agents SDK creates the workspace root twice through it — so the daemon almost certainly uses a `MkdirAll`-style create. The port additionally treats a 409 as success, so either behaviour works. |

### A caller-placed configuration is not excluded from archives

`workspace_archive_excludes` excludes `.sandbox-rclone-config/`, which covers every configuration
the pattern writes. A file the caller places through `config_file_path` is somewhere else in the
workspace by definition, so `persist_workspace` and snapshots capture it along with whatever
credentials it holds. Python has the same shape (`register_persist_workspace_skip_path` only
covers the generated directory), so this is parity, not a regression, but a caller who puts
credentials in the workspace should expect them in the archive. Fixing it would mean letting a
pattern register its own skip path, which is `session.rs`, outside this phase's scope.

### Remote-name derivation has no in-tree caller

The pattern is at parity, but nothing in this tree reaches the derivation yet. Python exercises it
from `entries/mounts/providers/{s3,gcs,r2,azure_blob,box}.py`, which build the configuration at
activation time with the derived name in the `[section]` header; none of those entries is ported.
The two Rust providers that use `RcloneMountPattern` (E2B, Daytona) pin a name, and an inline
`config_text` is built before a session id exists. So the derivation is reachable today only
through `config_file_path`, where the caller names the section themselves, or by a future port of
those entries. Porting them is outside this phase.

### Archive exclusions beyond these three providers

`workspace_archive_excludes` now backs the core default `persist_workspace` (Docker and
unix-local) and all three overrides here. **Cloudflare, Blaxel and Modal still override
`persist_workspace` with their own exclude logic**, and this task could not edit them. Now that
`RcloneMountPattern` is in the core, those backends can use it, so their archives could capture a
staged mount credential. Routing them through `workspace_archive_excludes` is a one-line change
each and is left for the controller.

## Phase 3: the last two `RcloneMountPattern` gaps

### Deriving the remote name from the session

`RcloneMountPattern::resolve_remote_name(session_id, config)` ports Python's
`resolve_remote_name`. A name pinned through the `remote_name` option wins, because the provider
that set it also generated the matching `[section]`; then one pinned on the pattern itself through
the new `remote_name` field; with neither, the name is `sandbox_{remote_kind}_{session_id}` from
the new `RCLONE_OPTION_REMOTE_KIND` option, and a mount with no kind either fails with
`MountConfigInvalid`, as Python's does. E2B and Daytona pin a name, so their behaviour is
unchanged.

The derivation is per session, so two sessions built from one workspace image never share a
mutable configuration section, and it is stable across a resume of any of these three providers,
whose `session.id()` is the client-side id carried in `SandboxSessionState` (E2B
`with_session_id(state.session_id)`, Daytona and Vercel the same). Only the teardown `pkill`
depends on that stability; removing the configuration is keyed on the mount path. Two deliberate
additions to Python:

- The resolved name is rejected when it contains whitespace or `:`, `/`, `[`, `]` or `\`. The name
  becomes both an INI section header and the `name:path` argument, so a separator inside it would
  silently attach a different remote. Python never validates it.
- Python uses `session_id.hex`; these sessions carry a hyphenated UUID string, so the derived name
  keeps the hyphens. rclone accepts them in a remote name.

`unapply` resolves the name the same way, so the `pkill` that reaps a daemonized `rclone mount`
still matches when the name was derived.

### Reading a configuration the caller placed in the workspace

`RcloneMountPattern::config_file_path` names a configuration the caller has already written into
the workspace; `read_config_text(session, remote_name)` reads it. A relative path resolves inside
the workspace root, as Python's `_resolve_config_path` does. The file must not be blank and must
declare the `[remote_name]` section this mount attaches, so a configuration for a different remote
fails before any credential is staged; both checks return `MountConfigInvalid`, while a missing
file keeps `WorkspaceReadNotFound`, matching Python's re-raise of `WorkspaceReadNotFoundError`.
The section test is `line.trim() == "[name]"` over the file's lines, which is what Python's
`^\s*\[name\]\s*$` with `MULTILINE` matches, without needing a regex dependency.

Python resolves the text one layer above `apply`, in `_build_rclone_config`; this port does it in
`apply`, where the session is already in hand, with the same precedence:

| `config_file_path` | `config_text` | Result |
| --- | --- | --- |
| set | set | the file, with the provider's settings merged into its `[remote]` section |
| set | unset | the file as written |
| unset | set | the provider's text (what every current caller does) |
| unset | unset | `MountConfigInvalid` |

The merge is `supplement_rclone_config_text`, a port of Python's
`_supplement_rclone_config_text`: the provider's settings are appended to that one section, after
dropping the header it repeats, so the sections of other remotes in the same file keep their
values and rclone's last-wins parsing lets the provider override what the caller wrote.

The text always goes into the mount's own `600` copy under `.sandbox-rclone-config/`, as Python
does, so a shared source configuration is never mutated and teardown never removes it. `apply`
refuses a `config_file_path` that resolves to the generated path, and `unapply` skips the removal
in that case too, so neither half can delete the caller's file.

## Verification

```
export CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/cloud1
rustfmt --edition 2024 <the five source files and the test file>
cargo clippy --features sandbox --lib --test sandbox_e2b_daytona_vercel_tests -- -D warnings
cargo test --features sandbox --test sandbox_e2b_daytona_vercel_tests   # 21 passed
cargo test --features sandbox                                           # whole suite
cargo test --features sandbox --lib extensions::sandbox                 # provider unit tests
```

`cargo clippy --features sandbox --all-targets -- -D warnings` is clean for the whole workspace
and `cargo test --features sandbox` passes end to end (282 lib tests plus every integration
suite), against the committed core.

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
  under its derived per-mount name, `--allow-other` and `--read-only` passed, and the
  configuration text absent from the command line); `create` → `session_state()` → JSON round
  trip → `resume` for Vercel (which rewrites the default root), for Daytona created without a
  manifest, and for a Daytona sandbox a previous run left stopped, which must be started again
  before the toolbox answers.
- **Phase 2** — a Daytona terminal driven over a real websocket (the mock completes the upgrade
  by hand and then speaks frames), covering the handshake credentials, the terminal size, the
  environment subprotocol, the typed command line, `write_stdin`, and the exit code read out of
  the close frame; the same flow against a server that selects *no* subprotocol, asserting the
  retry connects and the environment moves to the typed line; `resize_terminal` reached through
  `as_any` with its `{cols, rows}` body; and `ExecRequest::with_env` reaching all three
  providers' wire formats with a per-command variable overriding the manifest.

Unit tests additionally cover the core `RcloneMountPattern`'s FUSE and NFS command construction,
per-mount configuration paths, remote-name resolution (pinned, derived, rejected) and the
validation and merging of a caller-placed configuration; one of them drives `read_config_text`
end to end against a real `unix_local` session, which is the only way to check that a missing file
keeps `WorkspaceReadNotFound`. Others cover the HTTP error-body shapes, the pinned installer's
architecture table and checksum ordering (plus an `sh -n` parse of the generated script), and the
Vercel mount credentials staying out of argv while appearing in the command environment.

Unit tests in the source files cover redacted `Debug` output, base-URL and template resolution,
Connect framing in both directions, proto3 zero-value handling, snapshot-reference round trips,
entry mapping, command wrapping, the stdout/stderr split, multipart encoding, `mount-s3` and
`rclone` argument construction, network-policy serialisation, CRC-32 against known vectors, the
`ustar` checksum, and extraction of a generated archive by the system `tar`.
