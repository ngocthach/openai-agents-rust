# Changelog

## 0.3.0 (2026-10-29)

Completes the port. See the 0.3.0 section of `docs/project-roadmap.md`.

Breaking changes (pre-1.0):

- `AgentsError::ApiError` has a `headers` field; `AgentsError::Sandbox` is a new variant.
- `EncryptedSession` now uses Fernet (Python-compatible); items written by 0.2.0 are not readable.
- `SandboxRunner::run_streamed*` returns `SandboxStreamedRun<C>`.
- New `Session` and `Model` trait methods have default implementations.

## 0.2.0 (2026-10-29)

Ports the rest of the Python SDK surface. See the 0.2.0 section of `docs/project-roadmap.md`.

Breaking changes (pre-1.0):

- `Tool<C>` has new variants (computer, shell, local shell, apply patch, custom, tool search,
  programmatic tool calling); exhaustive matches need updating.
- `FunctionToolSpec`, `RunConfig` and `ModelRetrySettings` have new public fields; use
  `..Default::default()` or the builders.
- The `sqlite` feature now uses `rusqlite` 0.39 (to share `libsqlite3-sys` with `sqlx`).
- Approvals of namespaced function tools are keyed by `namespace.name`.

## 0.1.0 (2026-10-29)

Initial release: Rust port of the OpenAI Agents SDK (Python 0.24.0) core. See
`docs/project-roadmap.md` for scope and the features not yet ported.
