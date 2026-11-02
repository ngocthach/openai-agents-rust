# Parity round 22: in-memory resume, immutable-file removal (live)

Date: 2026-11-02 · Crate: openai-agents-sdk 0.24.0 · Branch: feat/rust-port-of-agents-sdk

## Summary

- **Task 1: no behaviour change was made, because Python does not behave the way the task assumed.**
  The task assumed that Python asks again for approval when an in-memory `to_state()` is resumed
  with a rebuilt agent graph. It does not. For an in-memory state, Python ignores the agent passed
  to `Runner.run` and runs the state's own `_current_agent`, which is the original object. That
  object owns the decision, so the decision always applies.
  - If Rust adopted the "same identity only" rule, it would ask again in cases where Python runs
    the tool. That would be a new divergence in the opposite direction.
  - The ~45 existing tests that resume with a rebuilt agent already show Python's outcome: the
    tool runs and nothing is asked again.
  - The one remaining difference is which closure runs; see "Remaining difference" below.
  - `tests/py_core22_inmemory_resume.rs` pins the Rust side of every row of the Python probe.
- **Task 2: done.** A live case in `tests/sandbox_docker_removal_live.rs`:
  1. sets `chattr +i` on a file inside the sandbox container, from the privileged root builder,
     through `/proc/<pid>/root`;
  2. checks that a recursive removal through the service fails with `PermissionError` / `EPERM`
     (1) and leaves the file intact;
  3. checks that a single-file `rm` (which does not use the service) fails with the sandbox's
     "Operation not permitted" and also leaves the file intact;
  4. checks that removal succeeds after `chattr -i`.

  Run for real on OrbStack: 5/5 live removal tests passed. Every Docker object I created is
  gone, and the user's containers were never touched.

## Task 1: in-memory `RunState` resume

### What Python does (source)

- `src/agents/run.py` takes the current agent of a resumed run from the state. Non-streamed:
  `run.py:977-981`, `current_agent = run_state._current_agent`. Streamed: `run.py:2756-2758` and
  `2891-2893`. The `starting_agent` argument is used for that only when the state has no current
  agent.
- `src/agents/run_context.py`, `_FunctionToolApprovalKey`: an in-memory decision holds a weakref
  to its owner. `_resolve_function_approval_owners(starting_agent)` (`run_context.py:237-266`,
  called at `run.py:692`) touches only keys whose `_agent_ref is None`. Those are keys from a
  context converter or from JSON. An in-memory `to_state()` key is never re-resolved. Its owner is
  the agent that runs.
- A JSON restore (`RunState.from_string(agent, payload)`) resolves the state's agents from the
  graph it is given. It binds each decision to the owner it finds by identity key, and skips
  ambiguous owners. Rust already matches this.

### What Python does (measured)

The probe is `/private/tmp/claude-501/-Users-ngocthach-myGit-openai-agents-rust/df34ec6b-67cb-4052-99ee-9636fbfd755b/scratchpad/c22/probe1.py`, run from `/Users/ngocthach/myGit/openai-agents-python` with
`PYTHONPATH=.` (output: `/private/tmp/claude-501/-Users-ngocthach-myGit-openai-agents-rust/df34ec6b-67cb-4052-99ee-9636fbfd755b/scratchpad/c22/probe1.out`).

The probe works like this:

- `make(label)` builds an `owner` agent with an `operate` tool (`needs_approval=True`) that appends
  `label` to `executed`.
- The run pauses and the state is approved (`always_approve` per row).
- The state is resumed with the mode's agent.

Raw output:

```text
memory-same     streamed=False always=False graph=False -> interruptions=0 executed=['orig'] final='done' last_agent=same
memory-rebuilt  streamed=False always=False graph=False -> interruptions=0 executed=['orig'] final='done' last_agent=orig
memory-renamed  streamed=False always=False graph=False -> interruptions=0 executed=['orig'] final='done' last_agent=orig
json-rebuilt    streamed=False always=False graph=False -> interruptions=0 executed=['rebuilt'] final='done' last_agent=passed
memory-same     streamed=False always=True  graph=False -> interruptions=0 executed=['orig', 'orig'] final='done' last_agent=same
memory-rebuilt  streamed=False always=True  graph=False -> interruptions=0 executed=['orig', 'orig'] final='done' last_agent=orig
memory-renamed  streamed=False always=True  graph=False -> interruptions=0 executed=['orig', 'orig'] final='done' last_agent=orig
json-rebuilt    streamed=False always=True  graph=False -> interruptions=0 executed=['rebuilt', 'rebuilt'] final='done' last_agent=passed
memory-same     streamed=True  always=False graph=False -> interruptions=0 executed=['orig'] final='done' last_agent=same
memory-rebuilt  streamed=True  always=False graph=False -> interruptions=0 executed=['orig'] final='done' last_agent=orig
memory-renamed  streamed=True  always=False graph=False -> interruptions=0 executed=['orig'] final='done' last_agent=orig
json-rebuilt    streamed=True  always=False graph=False -> interruptions=0 executed=['rebuilt'] final='done' last_agent=passed
memory-same     streamed=True  always=True  graph=False -> interruptions=0 executed=['orig', 'orig'] final='done' last_agent=same
memory-rebuilt  streamed=True  always=True  graph=False -> interruptions=0 executed=['orig', 'orig'] final='done' last_agent=orig
memory-renamed  streamed=True  always=True  graph=False -> interruptions=0 executed=['orig', 'orig'] final='done' last_agent=orig
json-rebuilt    streamed=True  always=True  graph=False -> interruptions=0 executed=['rebuilt', 'rebuilt'] final='done' last_agent=passed
memory-same     streamed=False always=False graph=True  -> interruptions=0 executed=['orig-dest'] final='done' last_agent=same
memory-rebuilt  streamed=False always=False graph=True  -> interruptions=0 executed=['orig-dest'] final='done' last_agent=orig
json-rebuilt    streamed=False always=False graph=True  -> interruptions=0 executed=['rebuilt-dest'] final='done' last_agent=passed
memory-same     streamed=False always=True  graph=True  -> interruptions=0 executed=['orig-dest', 'orig-dest'] final='done' last_agent=same
memory-rebuilt  streamed=False always=True  graph=True  -> interruptions=0 executed=['orig-dest', 'orig-dest'] final='done' last_agent=orig
json-rebuilt    streamed=False always=True  graph=True  -> interruptions=0 executed=['rebuilt-dest', 'rebuilt-dest'] final='done' last_agent=passed
memory-same     streamed=True  always=False graph=True  -> interruptions=0 executed=['orig-dest'] final='done' last_agent=same
memory-rebuilt  streamed=True  always=False graph=True  -> interruptions=0 executed=['orig-dest'] final='done' last_agent=orig
json-rebuilt    streamed=True  always=False graph=True  -> interruptions=0 executed=['rebuilt-dest'] final='done' last_agent=passed
memory-same     streamed=True  always=True  graph=True  -> interruptions=0 executed=['orig-dest', 'orig-dest'] final='done' last_agent=same
memory-rebuilt  streamed=True  always=True  graph=True  -> interruptions=0 executed=['orig-dest', 'orig-dest'] final='done' last_agent=orig
json-rebuilt    streamed=True  always=True  graph=True  -> interruptions=0 executed=['rebuilt-dest', 'rebuilt-dest'] final='done' last_agent=passed
```
 Each tool records its label
when it runs. The four resume modes:

- `memory-same`: the original agent object.
- `memory-rebuilt`: a new agent with the same name and configuration.
- `memory-renamed`: a new agent with a different name.
- `json-rebuilt`: `from_string` with a rebuilt graph.

| Rows (each: streamed and not, per-call and `always_approve`) | Python | Rust 0.24.0 |
|---|---|---|
| single agent, `memory-same` | no interruption, original tool runs | same |
| single agent, `memory-rebuilt` | no interruption, **original** tool runs (`last_agent is` original) | no interruption, **rebuilt** tool runs |
| single agent, `memory-renamed` | no interruption, original tool runs | error `Run state agent not found in agent map` |
| single agent, `json-rebuilt` | no interruption, rebuilt tool runs | same |
| handoff target (`root -> dest`), `memory-same` / `memory-rebuilt` / `json-rebuilt` | as the three rows above | as the three rows above |

In none of the 28 Python rows does Python ask again.

### Why Rust was not changed

The task said: "if a test genuinely rebuilds the agent in Python too, then Python re-prompts there
and the Rust test must expect that." That premise is false; Python never asks again on an
in-memory resume.

Keeping owners on in-memory resumes would make Rust's `memory-rebuilt` rows interrupt where
Python's run to completion. The 0.24.0 behaviour gives Python's observable outcome: the decision
applies and nothing is asked again. It also keeps the existing tests valid as they are. So:

- no source change was made;
- none of the ~45 tests that rebuild the agent was touched;
- no assertion was changed.

The core21 report says Python "keeps an in-memory state bound to the original objects". That is
true, but the consequence it drew ("re-prompts with a rebuilt graph") does not follow, because
Python also *runs* those original objects.

### Remaining difference (needs a user decision)

A Rust `RunState` is plain data with no agent references. Every resume runs the graph it is given,
found by name. So:

1. **Which closure runs.** With a rebuilt graph, Python runs the original tool and Rust runs the
   rebuilt one. This is only visible when the rebuilt tool behaves differently.
2. **A renamed agent.** Python ignores the passed agent. Rust fails with "Run state agent not found
   in agent map".

Closing both would mean an in-memory `RunState` carries the original `Arc<Agent<C>>` graph
(type-erased, skipped by serde), and `Runner::run(agent, state)` runs that graph instead of
`agent`.

- Cost: a public behaviour change, and it would break every Rust test that resumes with a fresh
  `FakeModel`. Rust cannot do Python's in-place `configured.model = ScriptedModel(...)` on a shared
  agent.
- Options:
  - (a) keep 0.24.0 (recommended; the outcome already matches);
  - (b) carry the graph in memory and prefer it on resume;
  - (c) (b) behind an opt-in.

### Tests

`tests/py_core22_inmemory_resume.rs`, `an_approved_in_memory_state_runs_without_asking_again`
covers 24 rows:

- single agent and handoff target;
- streamed and not;
- per-call and sticky;
- `Same`, `Rebuilt` and `Json`.

Each row asserts no interruption, the final output `done`, and which closure ran and how many
times. The `memory-renamed` row is not pinned, because Rust's error there differs from Python.

### Parity cases

No Python case becomes portable, because no behaviour changed.

- I checked the ledger and the round reports for Python-only cases classified as such because of
  in-memory resume; there are none.
- `test_agent_scoped_approvals.py`: the 14 remaining Python-only cases are Temporal-converter rows
  and gc/weakref cases (core21 report).
- `test_sticky_approval_stays_with_agent_across_handoffs`, including the `serialized=False` rows,
  is already ported in `tests/py_core2_scopedapprovals_agent.rs` and
  `tests/py_core9_r9gaps_approval_identity.rs`.

The new test file pins the probe rows, which are not Python test cases. It is not counted.

## Task 2: immutable file, docker-removal live

### What Python expects

No test under `tests/sandbox/test_docker_removal*.py` uses an immutable file. The closest is the
`PermissionError` row of `test_child_failure_reaches_structured_session_error`
(`test_docker_removal_worker.py:259-315`): a worker `OSError` reaches the session error as
`reason=docker_removal_failed`, `worker_reason=PermissionError`, and its `errno`.

On Linux, `unlink(2)` of an immutable file fails with `EPERM` for every caller, root included.
Python's worker reports `PermissionError` for both `EPERM` and `EACCES`. The expected errno is
therefore 1 (`EPERM`). Python has no test that pins it for this cause.

### The case

`live_removal_service_stops_at_an_immutable_file`:

1. Create a sandbox with `/workspace/frozen/{file,sibling}`.
2. Read the container's host pid from `docker inspect`.
3. Run `chattr +i /proc/<pid>/root/workspace/frozen/file` from the test process. It runs as root
   in the `--privileged --pid=host` builder, so it is on the Docker host.
4. Check that the sandbox itself cannot `rm -f` the file.
5. Remove `frozen` recursively. It fails with `reason=docker_removal_failed`,
   `worker_reason=PermissionError`, `errno=1`. The file's content is intact and the container is
   running and not paused.
6. Remove `frozen/file` non-recursively. Python's `DockerSandboxSession.rm`
   (`sandbox/sandboxes/docker.py:343`) uses the service only for `recursive=True`; Rust does the
   same, so this is an `rm` exec in the sandbox. It fails with `stderr` "Operation not permitted"
   and no removal `reason`. The file is intact and the container is running and not paused.
7. Run `chattr -i`.
8. `rm frozen` succeeds and the tree is gone.
9. Delete the container.

A `Mutable` drop guard, declared after the `Containers` guard so it drops first, runs `chattr -i`
however the test ends, unless the test already deleted the container. The flag is therefore
cleared before `docker rm -f`. Without that, the container's file system could not be removed.
The first run below exercised this path.

The test's containers also carry the label `oars-live-test=1`.

### Run

Scripts are in `/private/tmp/claude-501/-Users-ngocthach-myGit-openai-agents-rust/df34ec6b-67cb-4052-99ee-9636fbfd755b/scratchpad/c22/docker/`: `builder-up.sh`, `run.sh`, `cleanup.sh`. Logs are
`run-1.log`, `run-2.log` and `run-3.log`.

**Setup:**

- Before starting I recorded the images, volumes and containers that already existed.
- I pulled `rust:1.89-bookworm`; it was not present before.
- I started a builder named `oars-live-builder` (label `oars-live=1`) with the flags of the
  0.23.1 run: `--privileged --pid=host --cgroupns=host`, the Docker socket, the repo read-only at
  `/src`, and the volumes `oars-live-cargo-out` (CARGO_TARGET_DIR) and `oars-live-cargo-home`.
- I installed the static Docker CLI 29.0.0 in it. `chattr` comes with the image.

**Runs:**

| Run | Result |
|---|---|
| 1 | Recursive row passed. The non-recursive row failed: my first version expected the service there, but that call is a plain `rm` (see step 6). The drop guard cleared the flag and the container was removed; nothing was left behind. |
| 2 | `live_removal_service_stops_at_an_immutable_file` passed. |
| 3 | The whole file passed: 5/5 (`removes_refuses_and_releases`, `enforces_the_entry_visit_budget`, `preserves_a_read_only_host_grant`, `refuses_an_ineligible_container`, `stops_at_an_immutable_file`), in 8.9 s. |

Run 3 confirms:

- `chattr +i` through `/proc/<pid>/root` works on OrbStack's overlay root file system.
- The worker reports the kernel's `EPERM` as `PermissionError`, `errno=1`.

**Linux clippy:** after `rustup component add clippy` in the builder,
`cargo clippy --features docker-removal --test sandbox_docker_removal_live -- -D warnings` is
clean.

**Cleanup** (`cleanup.sh`): removed only

- containers labelled `oars-live=1` / `oars-live-test=1` that were not in the baseline (the
  builder);
- the two `oars-live-` volumes;
- `rust:1.89-bookworm`;
- `linux-tmp`.

After cleanup, the diff against the baseline is empty for images, volumes and containers. The
user's containers (`starthub-postgres`, `vibrant_swartz`, `agent-platform-minio`,
`gentube-*`) are still up, and nothing was pruned. The other live tests label their containers
`openai-agents-live-test=1` and remove them themselves; none were left.

## CSV

`py-test-parity-core22.csv`: zero moves. The rows list the files this round read.

## Gates

macOS, `CARGO_TARGET_DIR=/Users/ngocthach/.cache/oars-out/r22` (deleted afterwards):

| Gate | Result |
|---|---|
| `cargo test --workspace --all-features --no-fail-fast` | 7,978 passed, 0 failed, 5 ignored |
| `cargo test --workspace --no-fail-fast` | 3,748 passed, 0 failed |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps` | clean |

On macOS the docker-removal live tests skip, because `OPENAI_AGENTS_DOCKER_REMOVAL_LIVE` is
unset. They ran in the Linux builder (Task 2).

## Suggested changelog paragraph

> Documents and pins how an in-memory `RunState` resumes: a decision taken on `to_state()` applies
> when the state is resumed with the same agent, with a rebuilt agent graph, or after a JSON round
> trip, and the run never asks again. That is the outcome Python shows, because Python runs the
> state's own agent objects for an in-memory resume. Unlike Python, a Rust resume runs the graph it
> is given, so a rebuilt graph runs its own tools. The docker-removal live suite gains an
> immutable-file case: a file made immutable from the Docker host stops a removal with
> `PermissionError` / `EPERM` and stays intact, and once the flag is cleared, the removal
> succeeds. No library behaviour changed.

## Unresolved questions

- Should an in-memory `RunState` carry its original agent graph so that a resume runs it, as Python
  does (options (a)/(b)/(c) above)? I recommend (a).
