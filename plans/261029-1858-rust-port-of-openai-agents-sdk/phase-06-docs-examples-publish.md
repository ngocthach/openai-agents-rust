# Phase 06: Docs, examples, CI, publish

- README.md with quickstart parity to Python docs; docs/ (system architecture, code standards, roadmap).
- examples/: hello_world, tools, handoffs, structured_output, stream_text, stream_items, guardrails, sessions, human_in_the_loop, agents_as_tools, chat_completions, mcp_stdio, tracing_custom_processor.
- GitHub Actions workflow: fmt, clippy, test.
- Cargo metadata (license MIT, keywords, categories, repository, docs.rs all-features).
- `cargo publish --dry-run` then `cargo publish`.
- Commit + push to origin.
