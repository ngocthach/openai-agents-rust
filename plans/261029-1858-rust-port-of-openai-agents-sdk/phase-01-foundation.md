# Phase 01: Foundation

Files: `Cargo.toml`, `src/lib.rs`, `src/error.rs`, `src/usage.rs`, `src/model_settings.rs`, `src/items.rs`, `src/agent_output.rs`, `src/strict_schema.rs`, `src/util.rs`

Steps
- Cargo manifest with features `sqlite`, `mcp`; deps: tokio, reqwest(rustls, stream, json), serde, serde_json, schemars, thiserror, futures, async-trait, uuid, chrono, tokio-util, bytes, log.
- `AgentsError` enum mirroring Python exceptions (MaxTurnsExceeded, ModelBehaviorError, ModelRefusalError, ModelTimeoutError, UserError, ToolTimeoutError, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, ToolInputGuardrailTripwireTriggered, ToolOutputGuardrailTripwireTriggered, Http/Api, Serde). Each carries `RunErrorDetails` when raised by the runner.
- `Usage` with `add`, `RequestUsage`, token details.
- `ModelSettings` + `resolve` overlay semantics; `ToolChoice`; `Reasoning`.
- Items: `InputItem` (Value newtype w/ constructors), `OutputItem` typed enum with fallback, `RunItem` enum, `ModelResponse`, `ItemHelpers`.
- `AgentOutputSchema` from `schemars`, strict-schema transformer.

Validation: unit tests for strict schema conversion, usage add, model settings resolve, item parsing.
