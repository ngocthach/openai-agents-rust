# Python test parity, round 9, r9h3 (run-state / runner top-level files and realtime leftovers)

18 files, 299 python-only cases before. Every case was matched to its Python test and decided. Result: 65 ported, 0 covered, 234 remain valid python-only.

| file | before | ported | covered | remaining | reason for what remains |
| --- | --: | --: | --: | --: | --- |
| test_function_tool_approval_arguments.py | 58 | 54 | 0 | 4 | compiled `re.Pattern` objects (2), `tuple[int, ...]` annotation (1), `**kwargs` parameter (1) |
| test_integration_runner.py | 58 | 0 | 0 | 58 | `.github/scripts` CI script |
| test_release_automation.py | 56 | 0 | 0 | 56 | CI release script |
| test_run_examples_script.py | 33 | 0 | 0 | 33 | `examples/run_examples.py` script |
| test_repository_workflow_interfaces.py | 18 | 0 | 0 | 18 | Makefile / GitHub workflow files |
| test_release_provenance.py | 14 | 0 | 0 | 14 | CI release verifier |
| test_result_cast.py | 11 | 0 | 0 | 11 | gc / weakref / `release_agents`, `dataclasses.asdict`, pydantic model rebuild, frozen dataclass |
| test_programmatic_tool_calling.py | 10 | 3 | 0 | 7 | return-annotation inference through `Awaitable`/`Coroutine` aliases and PEP 695 (Rust names the output type) |
| test_hosted_mcp_connectors_example.py | 4 | 4 | 0 | 0 | |
| test_image_generation_tool_typing.py | 3 | 0 | 0 | 3 | pyright canaries, pydantic `TypeAdapter`, dataclass fields |
| test_run_context_approvals.py | 3 | 1 | 0 | 2 | non-string request ids (ids are `String`) |
| test_run_internal_approvals.py | 3 | 0 | 0 | 3 | private `_build_function_tool_call_for_approval_error`; the namespace it keeps never reaches the output item |
| test_run_internal_error_handlers.py | 2 | 2 | 0 | 0 | |
| test_image_generation_tool_reference.py | 1 | 0 | 0 | 1 | griffe docstring introspection |
| realtime/test_realtime_handoffs.py | 6 | 0 | 0 | 6 | fallible async `is_enabled` + sibling cancellation, `isinstance(Handoff)`, `inspect` arity checks, non-callable / callable-object `on_handoff`, `input_type` without `on_handoff` (typed builder) |
| realtime/test_realtime_model_settings.py | 9 | 0 | 0 | 9 | live tool/handoff objects with `is_enabled` in initial settings or overrides (Rust settings carry wire specs, which have no `is_enabled`), sibling cancellation of fallible checks |
| realtime/test_session_approvals.py | 3 | 1 | 0 | 2 | monkeypatched `_debug` + logger with a raising formatter (the formatter is infallible), asyncio cancel of the formatter |
| realtime/test_session_tool_outputs.py | 7 | 0 | 0 | 7 | `_serialize_tool_output` fallbacks for pydantic / dataclass / arbitrary objects / bytes (`ToolOutput::json` returns the error), `asyncio.CancelledError` at event publication, monkeypatched private serializer |
| **total** | **299** | **65** | **0** | **234** | |

## Ported

- `tests/py_core9_r9h3_approval_args.rs` (3 tests, 8 cases): an application default (`#[serde(default = "fn")]`) runs only after approval (streamed x approve, 4); a pattern-constrained value is checked after approval and reported through the failure policy (2); an input guardrail runs before the failure policy after approval, blocking or allowing (2).
- `tests/py_core9_r9h3_misc.rs`:
  - free-form output type rejected, output type description reaches the schema, an untyped tool has no output schema (programmatic, 3);
  - handler output rendered through, and refused by, a custom output schema (error handlers, 2);
  - hosted MCP connector tool request and answer over a local Responses server, non-streamed and streamed (4; the `verbose` flag only changes what the example prints);
  - deferred top-level decisions, per-call and permanent (run_context_approvals, 1).
- `tests/py_core9_r9h3_realtime.rs` (2 tests): the explicit rejection message wins over the run-level formatter (1 counted). A second test, the default rejection message sent once even when the call is redelivered, backs the cases the earlier report counted as covered.

## Source bug fixed

`src/run_context.rs`: a permanent approval of a deferred top-level tool authorized a visible tool with the same bare name, because both were recorded under the bare name (Python: `deferred_top_level:<name>` key; the visible tool keeps the bare one). Found by `a_permanent_deferred_approval_does_not_authorize_the_visible_tool_of_the_same_name`, which failed before the fix with both tools executed. The existing Rust tests used a hand-built item shaped differently from what the runner emits, so they did not see it.

- `approval_tool_key` returns `deferred_top_level:<name>` when the call's namespace equals the tool name.
- A per-call decision about such a tool is also recorded under the bare name (Python's bare-name alias), so `is_tool_approved(agent, name, call_id)` still finds it.
- `function_tool_status` reads a per-call decision recorded under the bare name before this change as that call's decision only, never as a sticky grant, so an in-flight snapshot still resumes.

Breaking change: persisted `RunState` approvals written by earlier versions as a sticky ("always") decision for a deferred top-level tool are no longer honored and the user is asked again (the safe direction; Python also requires re-approval for legacy keys).

## Verified

`cargo test --workspace --all-features --no-fail-fast`: 7244 passed, 1 failed. The failure is `py_core6_resume_name_fallback` (schema 1.6 same-named agents, `executed=[]`): it concerns agent identity for duplicate agent names, another agent's in-flight area, and does not touch a namespaced or schema-checked tool. Default-feature and all-feature clippy with `-D warnings` are clean; my files and `src/tool.rs` are rustfmt-clean.

## Decision applied: schema-only pre-approval inspection

The coordinator decided to match Python: the pre-approval check no longer runs the argument type's `Deserialize`. `src/tool.rs` `approval_args_check` now judges the JSON arguments against the tool's schema (`schema_shape_accepts`, plus `validate_value_constraints`) and the declared-key rule. It is conservative: an absent declared property (at any depth), an undeclared key, a type mismatch, a keyword it does not understand, or an integer literal for a `number` parameter (Python's int-to-float change) all require manual approval without consulting the rule. The full parse, including `deserialize_with` functions, custom `Deserialize` impls and application defaults, runs only after approval. The already-ported invalid-typed-input cases pass unchanged.

One difference remains and cannot be closed: Rust cannot see that a `deserialize_with` or custom `Deserialize` exists, so a dynamic rule is still consulted for arguments that match the schema (Python skips it for any model with user code). The ported cases therefore assert what both languages share, that no validation code ran before the decision, and do not assert that the rule went unconsulted.

`tests/py_core9_r9h3_validation.rs` (11 tests, 46 cases): item, nested, post-construction and union-branch validators wait for approval (16); a rewriting validator never runs for a rejected call (4); a validation hook runs between approval and body, runner/streamed/realtime x approve (6); a case-insensitive pattern is checked after approval, same modes (6); a failing validator uses the failure policy after approval, failure x mode (9; Python's `AttributeError` is a `ModelBehaviorError` here); a realtime item validator (2); nested aliases keep the original key for the rule (2); an integer for a float parameter requires manual approval (1).

## Breaking changes

1. A persisted "always approve" decision for a deferred top-level tool written by an earlier version is no longer honored; the user is asked again.
2. A dynamic approval rule is no longer consulted for a call whose arguments hold an integer literal for a float (`number`) parameter, an absent optional nested property, or a value the schema check cannot confirm: such calls now pause for manual approval first (as in Python). Previously the rule saw them whenever they parsed.
3. Custom `Deserialize` / `deserialize_with` code no longer runs before approval.

## Unresolved questions

1. `docs/project-changelog.md` has no entry for the approval-key fix or the breaks above (left to the coordinator as instructed).
