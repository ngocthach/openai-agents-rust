# Code standards

- Rust edition 2024, MSRV 1.89 (`rusqlite` requires 1.89; the core builds on 1.88).
- `cargo fmt`, `cargo clippy --all-features --all-targets -- -D warnings` and
  `cargo test --all-features` must pass before merging. CI enforces all three.
- `#![forbid(unsafe_code)]`; every public type implements `Debug`.
- Types generic over the context `C` implement `Clone` and `Debug` by hand. Do not derive them:
  derives add `C: Clone`/`C: Debug` bounds that break non-cloneable contexts.
- Builders use `with_*` methods returning `Self`; public fields stay available for direct edits.
- Errors use `AgentsError`. Large payloads in variants are boxed (clippy `result_large_err`).
- Callbacks are `Arc<dyn Fn(..) -> BoxFuture<..> + Send + Sync>`; public constructors accept plain
  async closures.
- Tests: unit tests next to the code, behavior tests in `tests/` using `testing::FakeModel`, wire
  format tests against a local mock HTTP server, MCP tests against `tests/fixtures/mock_mcp_server.py`.
- Keep the Python SDK as the behavioral reference; note intentional differences in docs.
