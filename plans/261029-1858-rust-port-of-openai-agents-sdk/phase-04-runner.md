# Phase 04: Runner

Files: `src/run.rs`, `src/run_config.rs`, `src/run_state.rs`, `src/result.rs`, `src/stream_events.rs`, `src/run_internal/{mod,turn,tool_execution,handoff_exec,guardrails_exec,session_io}.rs`, `src/run_error_handlers.rs`

- Runner::run / run_sync / run_streamed with RunOptions (context, max_turns, hooks, run_config, error_handlers, previous_response_id, auto_previous_response_id, conversation_id, session).
- Loop: prepare input (session history + callback), input guardrails (parallel/blocking, first agent only), model call (with call_model_input_filter), classify output: final output / handoffs / tool calls; execute function tools concurrently (bounded), approvals -> interruptions; tool_use_behavior; reset_tool_choice; handoffs (input filter, on_handoff, transfer message item); output guardrails; session persistence; usage.
- RunState: serializable snapshot (JSON) with approvals, pending tool calls, generated items, original input, current turn; approve/reject/add_input; resume path.
- Streaming: background task + mpsc; raw events, run item events, agent updated events; cancel (immediate / after_turn).
- Results: RunResult, RunResultStreaming (final_output, new_items, last_agent, raw_responses, to_input_list, to_state, interruptions, guardrail results, last_response_id, context).
- Error handlers for max_turns, model_refusal, invalid_final_output.

Validation: integration tests with a FakeModel (scripted) covering: plain text, structured output, tool loop, handoff, guardrail tripwire, max turns, approval pause/resume, streaming event order, session persistence.
