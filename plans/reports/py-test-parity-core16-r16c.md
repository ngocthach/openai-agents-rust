# py-test-parity core16 / r16c: full-body re-read of 32 python-only files

Work list: `scratchpad/r16_2.txt` (32 files). Rust tests: `tests/py_core16_r16c_*.rs`. CSV: `py-test-parity-core16-r16c.csv`
(`before - ported - covered = remaining` on every row). Method: the Python test names were checked against every Rust
file (`cand16.py`), the bodies of every uncited case were read, and each was classified per the round-8 rules.

## Totals (32 of 32 files done)

python-only before 316, ported 10, covered 2, remaining 304.

| file | python-only before | ported | covered | remaining | read_depth | note |
|---|--:|--:|--:|--:|---|---|
| tests/test_error_logging_redaction.py | 81 | 0 | 0 | 81 | full | remaining cases: traceback frame locals, exception groups/BaseException, hostile __str__/__bool__, logging.LogRecord caller location; behavioural ones already in py_core2_toolerrors_logging.rs |
| tests/mcp/test_mcp_server_manager.py | 38 | 2 | 0 | 36 | full | 2 ported (strict failed connect keeps active with drop_failed_servers=False; cleanup timeout blocks retry). Rest: worker tasks, asyncio cancellation, lifecycle lock, one-shot iterables |
| tests/extensions/sandbox/test_blaxel.py | 28 | 0 | 0 | 28 | full | import guards, SDK duck typing, pydantic frozen/ge=1, asyncio cancel shielding, PTY entry internals, fake test doubles |
| tests/extensions/sandbox/test_modal.py | 24 | 0 | 0 | 24 | full | asyncio cancellation settling (13), SDK call timeouts/kwargs in the Python helper, examples runner (4), PureWindowsPath, PTY fallback-read SDK polling |
| tests/sandbox/test_docker.py | 20 | 0 | 0 | 20 | full | docker SDK objects (http response close, exec_run workdir, images.pull parsing, socket handling), asyncio executors; rest covered in py_sandbox_a_docker_* |
| tests/realtime/test_openai_realtime.py | 17 | 2 | 2 | 13 | full | 2 ported (failed connect releases socket and restores model name: source fixed; attached call keeps audio formats), 2 covered (py_core8_r8g5_realtime_logging.rs raw-message conversion failure). Rest: monkeypatched private send/emit, listener callbacks, asyncio task state |
| tests/realtime/test_twilio_sip_server.py | 14 | 0 | 0 | 14 | full | unit tests of examples/realtime/twilio_sip/server.py (a Python FastAPI example); Rust has no counterpart |
| tests/test_asyncio_tasks.py | 12 | 0 | 0 | 12 | full | agents.util._asyncio_tasks gather/producer-consumer helpers: asyncio task cancellation internals |
| tests/test_run_config.py | 10 | 0 | 0 | 10 | full | dictionary/dynamic RunConfig normalisation and type validation (RunConfig is typed in Rust), PureWindowsPath cwd |
| tests/test_run_impl_resume_paths.py | 9 | 1 | 0 | 8 | full | 1 ported (end_hook variant of callback retry via try_on_tool_end). Rest: asyncio.CancelledError in callback, monkeypatched private turn_resolution / run_single_turn |
| tests/test_asyncio_progress.py | 7 | 0 | 0 | 7 | full | asyncio task progress-deadline introspection of function tool tasks |
| tests/test_handoff_tool.py | 6 | 2 | 0 | 4 | full | 2 ported (explicit tool name resolves derived collision; disabled handoff ignored). Rest: dynamic on_handoff signature validation, pydantic lenient coercion, asyncio sibling cancellation, bool is_enabled attr |
| tests/extensions/sandbox/test_e2b.py | 5 | 0 | 0 | 5 | full | package re-exports, SDK class resolution/connect wrapper, positional dataclass args, PTY waiter task internals |
| tests/test_default_tool_error_redaction.py | 5 | 0 | 0 | 5 | full | asyncio.CancelledError raised inside a tool (4), hostile __str__ exception (1); Rust errors are concrete and tool panics are not caught |
| tests/mcp/test_streamable_http_client_factory.py | 4 | 1 | 0 | 3 | full | 1 ported (default client does not follow redirects: source fixed in src/mcp/mod.rs). Rest: patched streamablehttp_client kwargs, type annotations |
| tests/test_agent_runner_sync.py | 4 | 0 | 0 | 4 | full | run_sync event-loop management (default loop reuse/replace, async generator finalization) |
| tests/test_tool_custom_data.py | 4 | 0 | 0 | 4 | full | non-JSON-compatible / NaN extractor data: serde_json::Value cannot hold them |
| tests/models/test_litellm_chatcompletions_stream.py | 3 | 0 | 0 | 3 | full | LiteLLM provider stream close/cancellation semantics; no Rust LiteLLM |
| tests/sandbox/test_entries.py | 3 | 0 | 0 | 3 | full | monkeypatched os.open / dir_fd flags to race a symlink swap |
| tests/test_financial_research_budget.py | 3 | 0 | 0 | 3 | full | tests the examples/financial_research_agent Python example |
| tests/test_run_state_pending_input.py | 3 | 0 | 0 | 3 | full | object-identity server tracker (Rust uses documented content-fingerprint delta selection, asserted in py_core3_runstatectx_pending_b.rs); acceptance-refusal case already covered |
| tests/voice/test_pipeline_config.py | 3 | 0 | 0 | 3 | full | dataclass positional-argument compatibility |
| tests/realtime/test_agent.py | 2 | 0 | 0 | 2 | full | dynamic type validation and mutated attribute types; typed in Rust |
| tests/sandbox/test_session_utils.py | 2 | 0 | 0 | 2 | full | best_effort_stream_len sizes Python file-like objects; Rust writes take &[u8] |
| tests/test_process_model_response.py | 2 | 0 | 0 | 2 | full | falsy-object truthiness of Python tool instances |
| tests/mcp/test_mcp_auth_params.py | 1 | 1 | 0 | 0 | full | 1 ported (default SSE client does not follow redirects). Rest: patched sse_client kwargs |
| tests/models/test_kwargs_functionality.py | 1 | 0 | 0 | 1 | full | LiteLLM dictionary model_settings normalisation (dynamic dict, no Rust LiteLLM) |
| tests/realtime/test_session_close_iteration.py | 1 | 0 | 0 | 1 | full | asyncio.CancelledError close failure variant |
| tests/sandbox/test_docker_network_mode.py | 1 | 1 | 0 | 0 | full | 1 ported (exposed_ports refused with network disabled) |
| tests/test_agent_tool_enablement_snapshot.py | 1 | 0 | 0 | 1 | full | asyncio concurrent mutation of agent.tools during is_enabled |
| tests/test_image_generation_tool_reference.py | 1 | 0 | 0 | 1 | full | griffe docstring introspection of Python source |
| tests/test_visualization.py | 1 | 0 | 0 | 1 | full | graphviz.Source.render with filename; Rust writes DOT text |

## Source fixes

- `src/realtime/openai_model.rs` `connect`: a failed connect restores the previous model name (Python resets `self.model`).
- `src/mcp/mod.rs`, `src/mcp/sse.rs`: the default MCP HTTP/SSE client no longer follows redirects (Python hardened default `follow_redirects=False`); a caller-supplied `http_client` keeps its own policy. Behaviour change: a redirecting MCP URL now fails to connect.

## New tests (`tests/py_core16_r16c_*.rs`, 12 tests, all pass; clippy -D warnings clean on both feature sets, rustfmt clean)

mcp_manager (2), realtime_connect (2), resume_end_hook (1), handoff_names (2), docker_network (2), mcp_redirect (3).

## Notes

- `test_run_state_pending_input`: Rust deliberately selects server deltas by content fingerprint, not object identity (documented in py_core3_runstatectx_pending_b.rs); left as a recorded divergence, not reversed.
- Rust lacks `follow_redirects`-style knobs beyond supplying `http_client`.
