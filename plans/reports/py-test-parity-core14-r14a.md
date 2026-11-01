# Round 14, r14a: Redis Cluster, persisted sandbox provider fields, Docker removal leftovers

Crate 0.16.0 plus the changes below. Per-file numbers: `py-test-parity-core14-r14a.csv` (before - ported - covered = remaining on every row; the "before" column is the "remaining" of the core13 reports).

| file | before | ported | covered | after | what remains and why |
|---|--:|--:|--:|--:|---|
| memory/test_redis_session.py | 48 | 4 | 0 | 44 | ported: the three `rejects_cluster_transactions_before_pipeline[add/clear/compact]` cases and `cluster_reads_and_subclass_writes[False]`. Remaining: `[True]` (the test overrides `add_items`/`clear_session` on a Python subclass; a Rust session is a concrete type), plus the groups r13b already listed (redis-py pool/WATCH internals, `close()` terminal state, decode_responses, `_get_next_id`, asyncio cancel) |
| sandbox/test_docker_removal.py | 2 | 1 | 0 | 1 | ported: `worker_close_reaps_and_closes_output_after_broken_input_pipe`. Remaining: `repeated_cancellation_waits_for_actual_host_completion[base_exception]` (Python `BaseException` out of a worker thread; the success and exception variants are ported, a Rust panic in a worker is a bug, not an outcome) |
| sandbox/test_docker_removal_worker.py | 2 | 0 | 0 | 2 | `bound_descriptors_are_closed...[register]` (patches `ExitStack.callback`; Rust has no registration step, an opened descriptor is an `OwnedFd` that is closed by drop, and the `open`, `fstat` and `none` variants are ported), `removal_child_always_exits_without_resuming_parent[KeyboardInterrupt]` (signal/`os._exit` of a forked child; the other five variants are ported) |
| sandbox/test_e2b.py | 5 | 0 | 0 | 5 | unchanged: the state-field assertions (`state.timeouts`, `state.metadata`, hydrate-from-snapshot create call) were already counted as ported or covered by earlier rounds; the persistence work below adds Rust tests for them but no case leaves the Python-only list |
| sandbox/test_vercel.py | 35 | 0 | 0 | 35 | same: `supports_pty_is_disabled...` was ported earlier (`py_sandbox_vercel::vercel_supports_pty_is_disabled`); the remaining groups are the ones r12a/r13a named |

Totals: 5 files, 92 python-only before, 5 ported, 0 covered, 87 remain. The state-field and `close()` work also closes three documented gaps, so the unresolved questions of r13a/r13b/r13c on those points are resolved.

## 1. Redis Cluster

- `Cargo.toml`: the `redis` dependency now enables `cluster-async`.
- `src/memory/redis_session.rs`: a private `Connection` enum (`Single(MultiplexedConnection)`, `Cluster(ClusterConnection)`) behind one `run` helper. New constructors `RedisSession::from_cluster_nodes(session_id, nodes)` and `with_cluster_connection(session_id, ClusterConnection)`. `get_items`, `pop_item` and `compaction_snapshot` work on a cluster. Every write (`add_items` with items, `clear_session`, `replace_history_atomically`, `replace_compaction_suffix`) fails with a user error before anything is sent: `RedisSession does not support transactions with Redis Cluster connections. Use a standalone connection (redis::aio::MultiplexedConnection), or implement a custom Session backend for Redis Cluster.` An empty `add_items` is a no-op, as in Python. A caller-owned cluster connection outlives the session.
- Why writes stay rejected: the write scripts touch three keys that live on different slots; Python rejects them for the same reason.
- Tests: `tests/py_core14_r14a_redis_cluster.rs` (4), against a local RESP endpoint that answers the cluster topology query and serves `LRANGE`/`RPOP` (read and pop, caller-owned connection, writes rejected with no command sent, manual compaction reports the rejection with no write command sent).

## 2. Persisted sandbox provider fields

The provider record under `provider_metadata.provider_options` now also carries:

| provider | new fields | rule |
|---|---|---|
| E2B (`E2BPersistedOptions`) | `template`, `metadata`, `envs` (Python `base_envs`), `timeouts` | metadata and envs redacted; template kept only when it does not look like a credential; timeouts absent in an old record |
| Vercel (`VercelPersistedOptions`) | `project_id`, `team_id`, `env`, `interactive` | env redacted; ids kept only when they do not look like a credential |

- Redaction (shared helpers in `src/extensions/sandbox/mod.rs`, moved out of the E2B module): an entry whose name looks secret (`key`, `token`, `secret`, `password`, `auth`, `credential`, `cookie`, `bearer`, `signature`, `session`) is dropped, as for the MCP shape; an entry whose value is shaped like a known credential (`sk-`, `ghp_`, `xox`, `AKIA`, `eyJ`, `Bearer `, a PEM block, ...) is dropped whatever its name; URL values lose userinfo, query and fragment. An identifier is kept only if it is short, uses a plain id alphabet, is not a URL and has no secret-looking word.
- Resume layering: metadata, envs and env merge key by key with the resuming client's value winning (the client holds the secrets the record omits); the recorded template, timeouts, project id, team id and `interactive` are the original session's and replace the client's when present; a missing field leaves the client's. Old states read through serde defaults.
- `VercelSandboxClientOptions::interactive` / `with_interactive` is added: accepted and persisted, enables nothing (Python's docstring says the same).
- Tests: `tests/py_core14_r14a_sandbox_state.rs` (9): E2B state records template/metadata/envs/timeouts with no secret in the serialized text, replacement sandbox built from the record with client values winning, secret-looking template not recorded, older record reads; Vercel state records scope/env/interactive, replacement uses the recorded project and team with merged env, client supplies a missing project and team (`test_vercel_resume_uses_client_project_and_team_fallbacks`), secret-looking ids not recorded, serde defaults. Existing E2B/Vercel/Modal binaries still pass.
- Deviations from Python, kept on purpose: the record is nested (not wire-compatible with a Python payload); env values and credential-shaped ids are not stored; Rust resolves `VERCEL_PROJECT_ID`/`VERCEL_TEAM_ID` from the environment (and the OIDC claims) when the options leave them unset, so the persisted id can come from there, unlike Python which passes `None` to the SDK.

## 3. Docker removal leftovers

- `src/sandbox/sandboxes/docker_removal.rs`: `ProcessWorker::close` now delegates to a generic `close_transport(stdin, wait, release_output)` (shut input, reap the process, release stdout, every step attempted, first failure reported), and `stdout` is an `Option` so `close` releases it. Behaviour with real pipes is unchanged. Two in-crate tests (`a_broken_input_pipe_still_reaps_and_releases_the_output`, `a_failed_wait_is_reported_when_the_input_closes_cleanly`) port `test_worker_close_reaps_and_closes_output_after_broken_input_pipe` with a mock writer whose shutdown fails with a broken pipe: events `stdin.close`, `wait`, `stdout.close`, the broken-pipe error is the one returned.
- The registration-failure variant stays Python-only (no registration step in Rust, see the table). `[base_exception]` and `[KeyboardInterrupt]` stay Python-only.

## Source changes and breaking changes

Files: `Cargo.toml`, `src/memory/redis_session.rs`, `src/extensions/sandbox/{mod.rs,e2b/mod.rs,vercel/mod.rs}`, `src/sandbox/sandboxes/docker_removal.rs`, `docs/project-changelog.md` (Unreleased).

Public API additions only: `RedisSession::{from_cluster_nodes, with_cluster_connection}`, `E2BPersistedOptions` and `VercelPersistedOptions` fields, `VercelSandboxClientOptions::{interactive, with_interactive}`. A struct literal of `E2BPersistedOptions`, `VercelPersistedOptions` or `VercelSandboxClientOptions` written without `..Default::default()` stops compiling. A resumed session now takes the recorded E2B template and timeouts, and the recorded Vercel project/team, over the resuming client's own.

## Gates

- `--all-features`: `py_core14_r14a_redis_cluster` 4, `py_core14_r14a_sandbox_state` 9, lib `docker_removal` 2; related binaries re-run (see Status). Default features: both new binaries compile (gated) and pass.
- clippy `-D warnings`, all-features and default, clean for the lib, bins and my test targets. The full `--all-targets` runs currently fail only on other agents' in-flight files (`py_core14_r14c_example_workflows`, `py_core14_r14c_public_api_contract`).
- rustfmt applied to every touched file.

## Unresolved questions

- The Redis Cluster test endpoint is a one-node mock; no real multi-node cluster (slot redirection) was available.
- E2B persists `sandbox_timeout`, `allow_internet_access`, `pause_on_exit`, `on_timeout` and `auto_resume` in Python; the Rust record still omits them (not in this task's list).

Status: DONE_WITH_CONCERNS
Summary: Redis Cluster, persisted provider fields and the broken-pipe `close()` port are done; 5 python-only cases moved to ported (4 Redis, 1 Docker removal), 0 covered, 87 remain.
Concerns: the Redis cluster path is tested against a mock endpoint only; the state-field work moves no case out of the Python-only list because those cases were already counted; full-workspace clippy blocked by other agents' in-flight test files.
