# Round 8 re-audit: tests/sandbox first half (r8g1)

Before: 318 python-only cases in 22 files. After: 41 moved to ported, 2 to covered, 271 valid python-only.

## Per-file table

| file | before | ported | covered | remaining | reason for what remains |
|---|---|---|---|---|---|
| capabilities/test_apply_patch_tool.py | 3 | 0 | 0 | 3 | `PureWindowsPath` monkeypatch (1); asyncio.Event interleaving of an approval resolved mid-checker (2, see questions) |
| capabilities/test_compaction_capability.py | 1 | 0 | 0 | 1 | pydantic policy validation |
| capabilities/test_filesystem_capability.py | 2 | 0 | 0 | 2 | unbound mutable Capability / configure_tools tool replacement; Rust capabilities are stateless |
| capabilities/test_shell_capability.py | 3 | 0 | 0 | 3 | unbound capability, configure_tools callbacks |
| capabilities/test_skills_capability.py | 9 | 6 | 0 | 3 | combining several skill sources / directory content: one Rust constructor per source, bytes content |
| capabilities/test_tool_input_errors.py | 2 | 0 | 0 | 2 | pydantic ValidationError and CancelledError as injected exceptions |
| test_bounded_read.py | 2 | 2 | 0 | 0 | |
| test_client_options.py | 13 | 9 | 0 | 4 | existing-instance passthrough, `exclude_unset` dump, duplicate/missing `type` class registration |
| test_compatibility_guards.py | 27 | 0 | 0 | 27 | `__all__`, dataclass/`inspect.signature` field order, positional constructor order |
| test_dependencies.py | 3 | 0 | 1 | 2 | private `_pending`/`_active_tasks` with loop ordering; CancelledError propagation. Eager-task retry is covered by `py_sandbox_a_entries_1::dependencies_cached_factory_failure_allows_retry` |
| test_docker.py | 28 | 4 | 1 | 23 | docker-py exec sockets/spool/pull parsing/container lookup/PTY sockets, asyncio deferred cleanup and cancellation, probe-script internals, Mount subclass trust list, exec workdir omission. RunState labels round trip covered by `py_core2_runstate_sidecars::run_state_round_trip_preserves_serialized_sandbox_session_snapshot_fields` |
| test_docker_network_mode.py | 4 | 3 | 0 | 1 | `typing_extensions` import AST check |
| test_docker_removal.py | 74 | 0 | 0 | 74 | Linux-root mount-namespace removal service, docker-py objects, Python worker subprocess; feature absent in Rust |
| test_docker_removal_client.py | 14 | 0 | 0 | 14 | same |
| test_docker_removal_worker.py | 60 | 0 | 0 | 60 | same |
| test_entries.py | 3 | 0 | 0 | 3 | monkeypatch of `_OPEN_SUPPORTS_DIR_FD` / `os.open` |
| test_external_provider_example.py | 8 | 0 | 0 | 8 | example script and pydantic option registry |
| test_extract.py | 3 | 0 | 0 | 3 | `io.IOBase` stream adapters |
| test_manifest.py | 11 | 6 | 0 | 5 | pydantic subclass registry, `@model_serializer`, `exclude_unset`/`exclude_defaults` dumps, duplicate registration |
| test_manifest_config.py | 18 | 5 | 0 | 13 | dict carriers via RunConfig/SandboxAgent (Rust takes a typed `Manifest`), typed object inside a dict |
| test_memory.py | 22 | 6 | 0 | 16 | pydantic model-settings config (~7), dataclass positional args, bound-session state, asyncio flush cancellation (2), private `_sandbox_memory_input` / RunResultStreaming helpers (3), conflicting-layout rejections (3, see questions) |
| test_mount_lifecycle.py | 4 | 0 | 0 | 4 | asyncio cancellation shielding |

Skills ledger arithmetic: 13 candidate cases were identified by name against 9 counted; 6 ported, remaining 3 are the valid ones; the cache-state and bound-session rows were evidently counted ported earlier (Rust capabilities are stateless and re-list per call).

## New tests
- tests/py_core8_r8g1_manifest.rs (14), _skills.rs (3), _bounded_read.rs (3), _client_options.rs (6), _docker.rs (7), _memory.rs (6).

## Source changes
- Manifest serde (additive): `Manifest::from_value`, `from_untrusted_value`, `to_value`; `persisted_registry::{parse_manifest, parse_untrusted_manifest, manifest_to_value}`. Matches Python's manifest dict shape; untrusted reader rejects host sources and extra path grants; credential acknowledgements are never written and never read. Literal env vars write as plain strings and lookups in long `{description, ephemeral, value}` form (pydantic keeps bare/entry as built); only the `str` env value kind is registered.
- Bug: client options payloads that named only some fields failed (`missing field network_disabled`). Added serde defaults to Docker, E2B, Daytona, Vercel, Blaxel, Modal, Cloudflare options (required fields stay required).
- Bug: `DockerSandboxSession::start` echoed the daemon's text for manifests with protected mounts; now redacted like create/delete.
- UnixLocal `read_bounded` now reads only the prefix natively (previously read the whole file, then truncated).
- Files: src/sandbox/persisted_registry.rs, manifest.rs, sandboxes/docker.rs, sandboxes/unix_local.rs, extensions/sandbox/{e2b/mod.rs,daytona.rs,vercel/mod.rs,blaxel/sandbox.rs,modal/sandbox.rs,cloudflare/sandbox.rs}.

## Breaking changes
None. Serde defaults only make previously rejected payloads parse.

## Unresolved questions
1. The 148 docker-removal cases are a feature gap (root host-side removal service), not test portability.
2. Apply-patch checker: Python stops evaluating later operations once a context approval resolves mid-check; Rust's checker has no agent identity to look the decision up, so that behaviour is absent.
3. Do conflicting generate configs / sessions dirs for one layout get rejected in Rust? Not verified.
4. `["docker"]` (a JSON array) parses as docker options through serde's tagged-enum quirk.

Gates: the six r8g1 test binaries pass (all-features). All-features and default clippy `--all-targets -D warnings` were blocked by another agent's in-flight `tests/py_core8_r8g6_docstrings.rs` compile errors (not mine); no diagnostics were reported against r8g1 files or the edited src files.

Status: DONE_WITH_CONCERNS
Summary: 318 cases audited; 41 ported, 2 covered, 271 remain python-only; 3 source bugs fixed, manifest serde added.
Concerns: docker-removal feature gap (148 cases), clippy unconfirmed due to another agent's file.
