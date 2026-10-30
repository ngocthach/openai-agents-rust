# Cloud sandbox providers: final parity pass

Area: `src/extensions/sandbox/**` plus `tests/sandbox_e2b_daytona_vercel_tests.rs` and
`tests/sandbox_modal_cloudflare_blaxel_tests.rs`. Python reference:
`/Users/ngocthach/myGit/openai-agents-python/src/agents/extensions/sandbox/**`.
No file under `src/sandbox/**` was edited.

Every row below was re-verified against the current tree before anything changed.

## 1. Provider inventory (task item 4)

Python directories under `src/agents/extensions/sandbox/`: `blaxel`, `cloudflare`, `daytona`,
`e2b`, `modal`, `vercel`, plus the shared `_rclone.py`. **All six have a Rust counterpart**
(`blaxel/`, `cloudflare/`, `daytona.rs`, `e2b/`, `modal/`, `vercel/`); `_rclone.py` is
`e2b/rclone.rs`. `grep -rni runloop` over the whole Python `src/agents/` returns zero hits — there
is no runloop provider, and nothing else has been added. **Nothing is missing; nothing was
ported.**

Every name in Python's `extensions/sandbox/__init__.py` `__all__` has a Rust counterpart except
the six `*SandboxSessionState` classes, which the Rust port unifies into the one core
`SandboxSessionState` (a pre-existing, documented design difference).

## 2. Vercel `CustomRules` network-policy wire shape (task item 3) — **fixed**

Checked against the Vercel REST API docs (`POST /v3/sandboxes`, and the v2
`POST /v2/sandboxes/sessions/{id}/network-policy`, which carries the fuller schema).
`networkPolicy` is a `oneOf` of two objects:

1. the **mode form**: `{mode, allowedDomains, allowedCIDRs, deniedCIDRs, injectionRules}`, `mode`
   ∈ `allow-all | deny-all | custom | default-allow | default-deny`;
2. the **rule form**: `{allow, subnets}`, where `allow` is `oneOf [array, object]` and `subnets`
   is `{allow: [], deny: []}`.

Both findings were then confirmed against the vendor's own serializer,
`vercel/sandbox` `packages/vercel-sandbox/src/utils/network-policy.ts`:

```ts
const apiPolicy =
  policy === "allow-all" || policy === "deny-all" ? { mode: policy } : policy;
```

- `Custom` and `CustomRules` were **already right**. A custom policy is sent **verbatim** as
  `{allow, subnets}` — the `: policy` branch above — and
  `packages/sandbox/src/util/network-policy.ts` builds `allow` as
  `Record<domain, NetworkPolicyRule[]>` with `allow[domain] = [{ transform: [{ headers }] }]` and
  `allow[domain] = []` for a domain with no rules. That is exactly `CustomRules`'s body. Left
  unchanged.
- `AllowAll` and `DenyAll` were **wrong**. They emitted `{"allow": "all"}` and `{"allow": []}`.
  A bare string matches neither branch of the rule form's `allow`, so `allow-all` was not a valid
  body at all; the SDK sends `{mode: "allow-all"}` / `{mode: "deny-all"}`.

Fixed: `AllowAll` → `{"mode": "allow-all"}`, `DenyAll` → `{"mode": "deny-all"}`. One unit test
updated, one integration test added (`vercel_sends_the_mode_form_for_allow_all_and_deny_all`).

## 3. Modal `ModalImageSelector` / `ModalSandboxSelector` (task item 2) — **closed**

Python's selectors are three-way (`image | id | tag`) and two-way (`sandbox | id`); the `image`
and `sandbox` kinds hold live `modal.Image` / `modal.Sandbox` objects. Those two kinds are
inherently un-portable — this port drives Modal through a stateless subprocess that addresses
everything by id over one JSON request/response. **Python itself persists only `object_id` from a
live object** (`modal/sandbox.py` L2265, L2294), so the id form carries everything that survives a
process boundary.

Added:

- `ModalImageSelector::{Id, Tag}` with `from_id` / `from_tag`. `Tag` is the name form (a registry
  tag such as `debian:bookworm`). Python has **no** name form for a sandbox — `Sandbox` is
  selected by object or id only — so `ModalSandboxSelector` has only `Id` / `from_id`.
- `ModalSandboxClientOptions::with_image(selector)` and `with_sandbox(selector)`; a selector
  clears the other image form so the last call wins. `with_image_tag` / `with_image_id` stay as
  sugar.
- `ModalSandboxClientOptions.sandbox_id`, and the adoption path it enables:
  `ModalSandboxClient::create` now attaches to an existing sandbox (polling it first, failing
  with `WorkspaceStartError` when it is not running) instead of calling `Sandbox.create`. This was
  the real gap — the port previously had no way to run against a sandbox the caller owns.

Tests: `modal_adopts_a_sandbox_named_by_a_selector`,
`modal_image_selectors_pick_the_id_or_the_tag`.

## 4. Rows closed from the three provider reports and the parity audit

### Exposed-port resolution — all six providers (was "not ported" in both provider reports)

Python declares `resolve_exposed_port` on the session base class and every provider implements
`_resolve_exposed_port`. Nothing of this existed in Rust.

New shared module `src/extensions/sandbox/ports.rs`, the port of
`agents.sandbox.types.ExposedPortEndpoint` plus `ExposedPortUnavailableError`:
`ExposedPortEndpoint {host, port, tls, query}` with `url_for("http"|"ws")` (default-port elision
and IPv6 bracketing exactly as Python), `assert_exposed_port_configured`,
`exposed_port_unavailable`, `endpoint_from_url` and `endpoint_from_authority`. No new dependency:
the URL/authority split is a 30-line parser, since `url` is not a direct dependency of the crate.

Per provider, matching its Python counterpart:

| Provider | How | Python |
| --- | --- | --- |
| E2B | `{port}-{sandbox id}.{domain}`, no round trip | `get_host` is client-side string building; `_e2b_endpoint_from_host` gives TLS/443 for a bare host |
| Daytona | `GET /sandbox/{id}/ports/{port}/signed-preview-url?expiresInSeconds=…`, the signature kept in `query` | `create_signed_preview_url` |
| Vercel | the `routes` the create/session response carries, now stored on the session | `sandbox.domain(port)` |
| Modal | new `tunnels` helper op | `sandbox.tunnels.aio()` |
| Blaxel | `GET …/previews/{name}` first, 404 → `POST /sandboxes/{name}/previews`, 409 → re-read; then `POST …/previews/{name}/tokens` for a private preview, token in `bl_preview_token` | `previews.create_if_not_exists` (read-first, which the SDK source confirms) + `preview.tokens.create` |
| Cloudflare | always refuses, with the "no port-resolution endpoint" detail | identical: Python raises for every port |

Endpoint paths and bodies were read off the providers' own sources, not guessed: the Daytona
endpoint from `daytona/skills` `references/api/sandbox.md`, the Blaxel endpoints and the
`spec.url` / `spec.token` / `spec.expiresAt` field names from `blaxel-ai/sdk-python`
(`client/api/compute/create_sandbox_preview{,_token}.py`, `models/preview_spec.py`,
`models/preview_token_spec.py`), the Vercel `routes` shape from the v3 OpenAPI, the E2B host
pattern from E2B's own docs, and `Sandbox.tunnels` from Modal's SDK reference.

New options: `DaytonaSandboxClientOptions::exposed_port_url_ttl_secs` (3600),
`BlaxelSandboxClientOptions::{exposed_port_public (true), exposed_port_url_ttl_secs (3600)}` —
Python's defaults, which is why `BlaxelSandboxClientOptions` now has a manual `Default` instead of
a derived one.

`ModalSandboxSession` did not implement `as_any`, so no caller could downcast to it; it does now
(the other five already did).

Tests: `e2b_sends_the_mcp_configuration_and_resolves_an_exposed_port`,
`daytona_resolves_an_exposed_port_through_a_signed_preview_url`,
`vercel_resolves_an_exposed_port_from_the_create_routes`,
`modal_resolves_an_exposed_port_through_a_tunnel`,
`blaxel_creates_a_preview_and_a_token_for_a_private_port`,
`blaxel_reuses_an_existing_public_preview`,
`cloudflare_reports_that_it_cannot_resolve_an_exposed_port`, plus six unit tests in `ports.rs`.

### Modal `snapshot_directory` persistence (was "not ported"; also the gap-closure report's third deferred item)

`ModalWorkspacePersistence::SnapshotDirectory`, with `snapshot_directory` and `mount_image`
helper ops, `encode_snapshot_directory_ref` / `decode_snapshot_directory_ref` (byte-identical to
Python's `_encode_snapshot_directory_ref`, magic `MODAL_SANDBOX_DIR_SNAPSHOT_V1\n`), and the
restore bound by `snapshot_filesystem_restore_timeout`. This also closes the deferred
`_restore_snapshot_directory_image` row, which had no Rust counterpart to bound.

Unlike a filesystem snapshot, restore remounts the workspace on the **running** sandbox, so the
sandbox id and open PTYs survive — matching Python, whose `_restore_snapshot_directory_image`
calls `sandbox.mount_image` rather than creating a replacement.

`ModalSandboxClient::create` now refuses a manifest with a Modal cloud bucket mount at or under
the workspace root when this mode is selected, mirroring
`_validate_manifest_for_workspace_persistence` (`ModalCloudBucketMountStrategy::mount_paths` is
the new helper it uses).

**The image captures the whole directory**, so, as in Python, every path a workspace archive
excludes — ephemeral entries and the staged rclone configuration, which holds a mount credential —
is tarred to a scratch path and deleted before the snapshot, then restored afterwards whether or
not the snapshot succeeded. Without that, a credential would be baked into an image that outlives
the sandbox. (Python detaches mounts here too; this port cannot need to, because a bucket mount at
or under the root is refused outright.)

Tests: `modal_round_trips_a_directory_snapshot` (which asserts the stash and the restore),
`modal_refuses_a_directory_snapshot_under_a_bucket_mount`, and three new fixture ops in
`tests/fixtures/fake_modal_helper.py`.

### E2B `mcp` server configuration (was "not ported")

`E2BSandboxClientOptions.mcp: BTreeMap<String, BTreeMap<String, String>>` and
`with_mcp_server(name, config)`, emitted as the top-level `NewSandbox.mcp` object. The field name
and its free-form object type were read off `e2b-dev/E2B` `spec/openapi.yml`, not memory.

### Retry wrappers (was "not ported": "Python's retry/instrumentation wrappers")

Core landed `src/sandbox/retry.rs` this round, so every `@retry_async` Python puts on a provider
call now has one:

| Provider | Retried call | Python |
| --- | --- | --- |
| Vercel | sandbox create; `fs/write` archive upload | L150, L1525 |
| Cloudflare | the one workspace archive request behind persist and hydrate (now classified `retryable` on a transport failure or a 5xx) | L1445, L1488 |
| Blaxel | the `tar` archive command (transport or timeout) | L664 |
| Daytona | the `tar` archive command only, not the download that follows | L1033 |
| Modal | the tar persist path | L1811 |
| E2B | the `tar` archive command | L1340 |

**Which failures are retried.** All six predicates are `is_transient_sandbox_error`, which reads
the `retryable` flag `src/sandbox/http.rs` sets per HTTP status (429 plus 5xx other than 501, and
any request that never reached the provider). A 400/401/402 fails on the first attempt, matching
`_is_transient_create_error`. Blaxel additionally retries `ExecTimeout`, as its Python predicate
does. See §11.

### Instrumentation decorators — **closed as already provided**

Python wraps each session at the client (`_wrap_session(inner, instrumentation=…)`). Core's
`src/sandbox/events.rs` landed `Instrumentation` and `InstrumentedSession`, which the caller wraps
a session with. Same capability, applied one level up; no provider change is needed, and adding a
provider-side wrapper would duplicate it.

### Vercel `_s3_mounts_started` (gap-closure deferred item 1) — **closed**

`VercelSandboxSession.s3_mounts_started`: `start()` on a session whose manifest carries a Vercel
S3 mount fails the second time with Python's message, "Vercel S3 mount topology is fixed when the
sandbox is created; starting the same mounted session again is not supported". A session without
S3 mounts is unaffected.

The flag closes exactly when Python's does — on success, or on failure **only once a mount has
actually attached**. A start that failed before any credential reached the sandbox stays
retryable rather than locking the session out permanently. Tests:
`vercel_refuses_to_start_a_mounted_session_twice` and
`vercel_lets_a_mounted_session_start_again_after_an_early_failure`.

### Vercel OIDC credentials (was "not ported": "OIDC token refresh") — **closed, re-scoped**

There is no token *refresh* anywhere in Python: `vercel.oidc.credentials.get_credentials` resolves
once and the SDK's credential factory returns fixed values. The real gap was **claim-based
resolution**: when `VERCEL_OIDC_TOKEN` is set but `VERCEL_PROJECT_ID` / `VERCEL_TEAM_ID` are not
both set, Python decodes the token payload and takes `project_id` and `owner_id` from it. Rust
read the environment only, so a deployment carrying only an OIDC token sent no project or team.

`resolved_project_id` / `resolved_team_id` now fall back to those claims (payload decoded, never
verified — the API validates the token). **Deliberately not adopted**: Python *raises* when token,
project and team are not all three resolvable. The Rust client omits `projectId`/`teamId` when
unknown, which works for a personal-scope API token against the REST API; turning that into a hard
error would break configurations that work today, so it is listed here rather than changed.

### Daytona `--allow-other` divergence — **resolved toward Python**

The previous round flagged this as an open behavioral question. Re-verified: Python's
`daytona/mounts.py` `_delegate` passes `self.pattern` through untouched and never calls
`rclone_pattern_for_session`; only E2B does. `DaytonaCloudBucketMountStrategy::fuse_pattern` was
appending `--allow-other`. Removed, so Daytona now matches Python exactly. A FUSE mount that must
be readable by a non-root command user carries the flag in the pattern's `extra_args`, which is
the same escape hatch a Python user has. No test asserted the old behavior.

### Sandbox metrics, Vercel drives/forks — **closed as non-existent**

Both appeared in the E2B/Daytona/Vercel report's "not ported" list. Neither exists in the Python
source: `grep -rn metrics` over `extensions/sandbox/` returns nothing, and the only `drive`/`fork`
hit is `build_docker_volume_driver_config`, an unrelated helper. No code was written.

### `adopt_recorded_root` imported from E2B by Daytona and Vercel — **not a parity gap**

It is a shared root-adoption helper with three callers, not the snapshot encoder. Copying it would
duplicate logic. Left as is.

## 5. Rows re-verified as already closed

Rows 60, 119, 120 and 121 of `parity-audit-261030-0505-python-to-rust-report.md` were spot-checked
against the current tree, not trusted from the report: the `rpm`/`dnf`/`findmnt`/`find -mindepth`/
`realpath` probes in `vercel/mounts.rs`, `rclone_pattern_for_session` + `default_user_ids` in
`e2b/rclone.rs`, `UC_VERCEL_SNAPSHOT_V1` + `RUNTIME_IMAGES` in `vercel/mod.rs`,
`snapshot_filesystem_restore_timeout` + `cpu_limit`/`memory_limit_mb` in `modal/sandbox.rs`, and
`MOUNT_TYPE_BLAXEL_DRIVE` + `impl ConfiguredMount for BlaxelDriveMount` in `blaxel/mounts.rs`. All
present.

Also re-verified from the E2B/Daytona/Vercel report: its open item **"Cloudflare, Blaxel and Modal
still override `persist_workspace` with their own exclude logic"** is already closed — all three
call `crate::sandbox::session::workspace_archive_excludes`, so none of their archives can capture
a staged mount credential.

## 6. Rows that remain open, with the reason

| Row | Reason |
| --- | --- |
| `ModalImageSelector`'s `image` kind and `ModalSandboxSelector`'s `sandbox` kind | **Inherent.** They hold live `modal.Image` / `modal.Sandbox` objects, which cannot cross the helper's JSON boundary. The `Id`/`Tag` forms carry everything Python itself persists from them. |
| Vercel `ephemeral` requirement in `_validate_s3_mount` | **Structural, with one hole.** `src/sandbox/entries.rs` forces `ephemeral = true` for every mount entry, but `Entry::with_ephemeral(false)` can still override it, and a `MountStrategy` only sees the `Mount`, never the `Entry`. Enforcing it needs a core change. |
| Vercel exposed ports after a **reattach-resume** | **Known limitation, not inherent.** Routes are taken from the create response and refreshed on a snapshot restore. The reattach path reads `GET /v2/sandboxes/sessions/{id}`, a session-scoped response that does not carry the sandbox's routes, so `resolve_exposed_port` on a reattached session reports `backend_unavailable`. Closing it means refetching the named sandbox (`GET /v3/sandboxes/{name}`), whose response shape was not verified this round. |

The four core-owned rows this report asked for in an earlier pass — `ErrorCode::ExposedPortUnavailable`,
`ExposedPortEndpoint` with `resolve_exposed_port` on the trait, `read_bounded`, and `retryable`
per HTTP status — all landed and are **adopted**; see §12.

## 7. Breaking changes

1. **`NetworkPolicy::AllowAll` / `DenyAll` wire shape** is now `{"mode": "allow-all"}` /
   `{"mode": "deny-all"}` instead of `{"allow": "all"}` / `{"allow": []}`. The old `allow-all`
   body was not valid against the documented schema.
2. **`ModalWorkspacePersistence` gained `SnapshotDirectory`**, so an exhaustive match on it breaks.
3. **`ModalSandboxClientOptions` gained a public `sandbox_id` field** on a struct that is not
   `#[non_exhaustive]`, breaking struct-literal construction. Nothing in `src/`, `tests/` or
   `examples/` constructs it outside `new()`.
4. **`DaytonaSandboxClientOptions` gained `exposed_port_url_ttl_secs`** and
   **`BlaxelSandboxClientOptions` gained `exposed_port_public` and `exposed_port_url_ttl_secs`**;
   same struct-literal caveat. `BlaxelSandboxClientOptions::default()` is now a manual impl, so
   its two new fields default to Python's `true` / `3600` rather than `false` / `0`.
5. **`E2BSandboxClientOptions` gained `mcp`**; same struct-literal caveat.
6. **Daytona no longer adds `--allow-other`** to a FUSE mount pattern. A Daytona mount that needs
   it must now put it in the pattern's `extra_args`, exactly as in Python.
7. **A second `start()` on a Vercel session with an S3 mount now fails** with
   `MountConfigInvalid` instead of re-mounting.
8. **`VercelSandboxClientOptions` resolves the project and team from the OIDC token's claims**
   when the environment names neither, so a request that previously omitted `projectId`/`teamId`
   may now carry them.

From the follow-up pass that adopts the core hooks:

9. **`resolve_exposed_port` is a trait method, not an inherent one.** Calling it on a concrete
   session type now needs `SandboxSession` in scope. It also no longer returns
   `SandboxConfigInvalid`: the code is `ErrorCode::ExposedPortUnavailable`.
10. **`extensions::sandbox::ports::ExposedPortEndpoint` is a re-export** of
    `crate::sandbox::types::ExposedPortEndpoint`. The path still resolves; the type is a
    different one, and it now derives `Serialize`/`Deserialize`.
11. **A `write_file` under a read-only `SandboxPathGrant` is refused** on all six providers,
    with `WorkspaceArchiveWriteError` and `reason = "read_only_extra_path_grant"`.

## 8. Shared-file edits

**None.** `src/lib.rs` and `Cargo.toml` were not touched: the new `ports` module is declared in
`src/extensions/sandbox/mod.rs` (one `pub mod ports;` line), which this area owns, and no new
dependency was needed — `chrono` and `base64` are already crate dependencies, `base64` under the
`sandbox` feature.

## 9. For the coordinator

- The nine Vercel breaking changes from the previous round are **still not** in the 0.3.0
  breaking-changes doc; the eight above join them. `docs/` is outside this area.
- Four of the five core follow-ups this report asked for have landed and are adopted (§11):
  `ErrorCode::ExposedPortUnavailable`, `ExposedPortEndpoint` plus the
  `SandboxSession::resolve_exposed_port` trait method, `read_bounded`, and `retryable` per HTTP
  status. **One is still open**: an `ephemeral` signal a `MountStrategy` can read, so Vercel's
  `_validate_s3_mount` can enforce it rather than rely on `src/sandbox/entries.rs` defaulting it.
- `cargo check --features sandbox` (that feature alone) fails in `src/models/openai_client.rs`,
  which uses `reqwest::multipart` without the `voice` feature that enables it. One line, outside
  this area; `--all-features` is unaffected.

## 10. Files changed

| File | What |
| --- | --- |
| `src/extensions/sandbox/ports.rs` (new) | re-export of the core port types, plus the provider-side URL/authority parsing and the declared-port check |
| `src/extensions/sandbox/mod.rs` | `pub mod ports;` |
| `src/extensions/sandbox/modal/sandbox.rs` | selectors, sandbox adoption, `SnapshotDirectory`, tunnels, `as_any`, tar persist retry, `exposed_ports`/`resolve_exposed_port` overrides, write grant |
| `src/extensions/sandbox/modal/mounts.rs` | `mount_paths` |
| `src/extensions/sandbox/modal/mod.rs` | re-exports |
| `src/extensions/sandbox/modal/helper.py` | `snapshot_directory`, `mount_image`, `tunnels` ops |
| `src/extensions/sandbox/e2b/mod.rs` | `mcp`, exposed-port override, persist retry, write grant |
| `src/extensions/sandbox/daytona.rs` | signed preview URL, ttl option, persist retry, `--allow-other` removed, exposed-port override, write grant |
| `src/extensions/sandbox/vercel/mod.rs` | network-policy mode form, routes + exposed-port override, `s3_mounts_started`, OIDC claims, create/write retries, write grant |
| `src/extensions/sandbox/blaxel/sandbox.rs` | previews + tokens, preview options, manual `Default`, persist retry, exposed-port override, write grant |
| `src/extensions/sandbox/cloudflare/sandbox.rs` | `read_bounded` wire limit, exposed-port overrides, archive retry + retryability, write grant |
| `tests/sandbox_e2b_daytona_vercel_tests.rs` | 8 tests |
| `tests/sandbox_modal_cloudflare_blaxel_tests.rs` | 11 tests |
| `tests/fixtures/fake_modal_helper.py` | 3 fixture ops |

## 11. Adopting the core hooks (follow-up pass)

The sandbox-core area landed the four additions §9 asked for, plus a read-only path-grant check
on writes. All five are now used; nothing in `src/sandbox/**` was touched.

### Read-only path grants reach `write_file`

`write_file` is the one file operation each backend implements itself, so the SDK-side
read-only-grant check had to be requested per provider. All six now call
`normalize_path_for_write` instead of `normalize_path`, so a write under a
`SandboxPathGrant { read_only: true }` fails with `WorkspaceArchiveWriteError` and
`reason = "read_only_extra_path_grant"` before the backend is asked to do anything. Reads
through the same grant still succeed.

Tests: `providers_refuse_a_write_through_a_read_only_path_grant` (E2B, Vercel, Daytona) and
`modal_and_blaxel_refuse_a_write_through_a_read_only_path_grant`, which also asserts the Modal
helper is never asked to write.

### `ports.rs` is now a re-export

`ExposedPortEndpoint`, `ExposedPortUnavailable` and `exposed_port_unavailable` come from
`crate::sandbox::types`; the local copies are gone, and with them the `SandboxConfigInvalid`
stand-in — port failures now carry `ErrorCode::ExposedPortUnavailable`. What stays in
`extensions/sandbox/ports.rs` is the provider-side parsing the core should not know about:
`endpoint_from_url` (Daytona, Blaxel, Vercel), `endpoint_from_authority` (E2B's `get_host`
shape) and `assert_exposed_port_configured`, which an override needs because overriding
`resolve_exposed_port` replaces the trait default that would otherwise perform that check.

### `resolve_exposed_port` is a trait override on all six

Each provider's inherent method became an override of the new default, and each implements
`exposed_ports()`. A caller now resolves a port through `&dyn SandboxSession` without a
downcast, which is what the tests do. Blaxel keeps its own override precisely because Python
overrides `_assert_exposed_port_configured` to a no-op there — a Blaxel preview is minted for
any port on demand. Cloudflare's override exists only to carry Python's explanatory detail; the
refusal itself is what the trait default would do.

### Cloudflare bounds the wire, not just the result

`CloudflareSandboxSession::read_bounded` overrides the default. The worker answers a file either
as raw bytes or as an SSE stream whose base64 chunks are several times the file's size, so
bounding only the decoded result would still let the sandbox decide how much this process
buffers. The override reads the body in chunks, stops at `max_bytes` for a raw body and at
`8 * max_bytes + 65536` for an SSE body, and refuses an over-budget SSE body with
`reason = "bounded_read_wire_limit"` — the convention the core doc names — rather than decoding
it. A 404 is still `WorkspaceReadNotFound`; a 403 is explicitly non-retryable, as in Python.

Tests: `cloudflare_bounds_a_raw_read_at_max_bytes`,
`cloudflare_refuses_an_sse_body_over_the_bounded_read_wire_limit`,
`cloudflare_decodes_an_sse_body_inside_the_wire_limit`.

### Retries read `retryable` instead of guessing from the code

`http.rs` now sets `retryable` per status, so the hand-rolled status classification added in the
previous pass is gone: the two Vercel sites are back to plain `parse_json` / `expect_success`,
and all six predicates are `is_transient_sandbox_error` (Blaxel additionally keeps Python's
timeout case). A 402 on `POST /v3/sandboxes` now fails on the first attempt while a 503 is
retried three times, which `vercel_fails_a_create_once_on_a_permanent_status` asserts by
counting requests. The §4 caveat about the exec-driven retries being unable to tell a 4xx from a
dropped connection no longer applies.

## 12. Verification

`CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-build/cloud2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_INCREMENTAL=0`:

- `cargo clippy --all-features --all-targets -- -D warnings` — **clean, workspace-wide.**
- `cargo test --all-features` — 34 suites, **0 failures**, including 616 lib tests.
- `cargo test --all-features --test sandbox_e2b_daytona_vercel_tests --test
  sandbox_modal_cloudflare_blaxel_tests` — 41 + 37 passed.
- `cargo test --all-features --lib extensions::sandbox` — 102 passed (three fewer than the
  previous pass: the `url_for` and scheme tests moved to core with the type).
- `cargo doc --all-features --no-deps` — no warning in any file this area owns.
- `rustfmt --edition 2024 --check` — clean on all twelve changed Rust files (ten under `src/`,
  two under `tests/`). `src/lib.rs` was never formatted or edited.
- `python3 -m py_compile` on both helper scripts — clean.

No network, no real credential, no git command.

**One pre-existing failure outside this area**, found while running the gates and not fixed here:
`cargo check --features sandbox` (the `sandbox` feature *alone*) fails in
`src/models/openai_client.rs`, which calls `reqwest::multipart` — a `reqwest` feature only the
`voice` feature enables. `--all-features` and `--features voice` are unaffected, which is why the
gates above are green. The fix is one line in `Cargo.toml` or in that file, both outside this
area.

Status: DONE
Summary: Every deferred/partial provider row is closed, or has an inherent reason (Modal's two
live-object selector kinds) or a stated limitation (Vercel routes after a reattach-resume). This
pass also adopted the five hooks the sandbox-core area landed: all six `write_file`s honour a
read-only path grant, `ports.rs` is a re-export of the core types, `resolve_exposed_port` is a
trait override on all six providers, Cloudflare bounds the read wire as well as the result, and
every retry predicate reads `retryable` per HTTP status. `cargo clippy --all-features
--all-targets -- -D warnings` is clean workspace-wide and `cargo test --all-features` passes 34
suites with no failures.
Concerns/Blockers: one core ask is still open — an `ephemeral` signal a `MountStrategy` can read,
so Vercel's `_validate_s3_mount` enforces it rather than relying on `entries.rs` defaulting it.
Separately, `cargo check --features sandbox` alone fails in `src/models/openai_client.rs`
(`reqwest::multipart` without the `voice` feature that enables it) — one line, outside this area,
and `--all-features` is unaffected. The eleven breaking changes in §7 still need folding into the
0.3.0 breaking-changes doc (`docs/` is outside this area).
