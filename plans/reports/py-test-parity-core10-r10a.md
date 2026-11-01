# Python test parity, round 10, r10a (restore skips, dropped tools, schema 1.12 replay)

Three documented divergences from earlier rounds were closed in the source, and the Python cases they blocked were ported. The counts are approximate where the earlier reports gave "about".

## Source changes

- `src/run_state/wire.rs`: `run_item_from_wire` returns `Option`. Like Python's `_deserialize_items`, it skips (with a redacted warning, no error) an item with no `type`, no agent, a handoff output missing its source or target, a tool output whose raw item has no `type`, a message without id/role/content, or a payload its typed model rejects. Generated items keep a placeholder while the snapshot is read, so recorded positions line up; the placeholders are then removed and turn start, model-input start, saved/sent counts and owned items remapped (`RunState::remove_generated_items`). Warning text follows the model/tool data policies.
- `src/run_state.rs`: `remove_generated_items`, `drop_items_of_unknown_agents` (pub(crate)), and the read-derived field `nested_history_lineage_unknown` (snapshots older than 1.13).
- `src/run_state/nested_history.rs`: ownership of a skipped session item is ignored with Python's warning instead of landing on another item.
- `src/run_internal/mod.rs`: `restore` drops items of agents outside the resumed graph (including handoff outputs whose source or target is outside it). A resumed parked function call whose tool is gone no longer fails the resume: a call nothing waits on is dropped, an undecided approval stays pending, a rejected one still answers the model; an approved one still fails with "Tool X not found in agent Y" (Python does the same under the default `raise_error`). For a pre-1.13 snapshot, a nested handoff does not claim the restored items as already forwarded, so they are replayed twice, as Python does.

## Ported

- `tests/py_core10_r10a_item_skips.rs` (9 tests): unknown agent, missing agent, missing type, malformed item, typeless tool output, handoff output missing a side, both-data-policy warnings, ownership remap after a skipped item, ownership of a skipped item ignored. Replaces the three `rust_refuses_*` pins in `py_core2_runstate_sidecars.rs`.
- `tests/py_core10_r10a_resume_dropped_tool.rs` (5 tests, streamed and not, disabled and removed): undecided / rejected / approved / completed lost tool, and a retained agent tool keeping its nested state.
- `tests/py_core9_r9h1_handoff_history.rs`: the schema 1.12 half now asserts Python's two occurrences (2 cases, previously a pinned divergence).

## Table

| file | python_only before | ported | covered | remaining |
| --- | --: | --: | --: | --: |
| test_run_state.py | 149 | 13 | 0 | 136 |
| test_handoff_history_duplication.py | 12 | 2 | 0 | 10 |
| test_error_logging_redaction.py | 86 | 3 | 0 | 83 |
| test_run_impl_resume_paths.py | 11 | 0 | 0 | 11 |

Remaining groups keep their earlier reasons (pydantic/deepcopy, asyncio, private helpers, object identity). Of the "Divergences left unchanged" in r9h1: equal copies of `result.input` and hand-rebuilt `HandoffInputData` are object-identity cases (valid python-only); fallible failure formatters would change a public type shared with function tools and were left.

## Breaking changes

Behaviour: `RunState::from_json` no longer errors on unreadable items (skips them); a resume no longer errors on a lost unapproved tool. `RunState` gains one `#[serde(skip)]` public field (`nested_history_lineage_unknown`); struct literals need `..Default::default()`.

## Gates

Binaries (runner, run_state, resume, hitl, approvals, collision, py_core2/3/4/6/8/9 families, lib): 167 green with `--all-features`; new binaries also green on default features. Clippy `-D warnings` clean on all-features and default `--all-targets`; rustfmt applied. Three existing tests were updated to Python's behaviour: `core_run_state::an_unreadable_item_is_skipped_like_python`, `py_core2_runstate_items::deserialize_handoff_output_items_with_string_agents` (a handoff output with no source is skipped), and the 1.12 case of the handoff-history test (two occurrences). A call whose paused nested agent run still exists keeps the not-found refusal (`py_core4_collision_nested`).

## Unresolved

- The rejected-lost-tool stand-in is a hidden always-approval tool; if a hook observes it, it sees a tool the agent does not declare.

Status: DONE
Summary: 18 cases ported (run_state 13, handoff history 2, error-logging 3); three divergences closed in source.
