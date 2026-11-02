//! Ports of the capability-binding cases of the Python SDK's `tests/test_agent_scoped_approvals.py`.
//!
//! Python binds a function-tool decision to the live agent object that owns it (an owner token
//! that a transported context can only reattach to that same object). Rust has no object
//! identity that survives serialization, so it binds a decision to a capability fingerprint
//! instead: a SHA-256 of the agent name and the tool's name, namespace, description, parameter
//! schema and origin. Two adaptations follow from that and are applied throughout:
//!
//! - Python's same-named tools differ only by their Python function body. A fingerprint cannot
//!   see a closure, so the Rust tools that stand for two distinct capabilities also differ in
//!   description (`"Return public data."` vs `"Return private data."`).
//! - Python attaches a decision to an owner through the item's `agent` object. A Rust item names
//!   its agent, so a hand-built item is bound with [`RunContext::approve_tool_for_agent`] /
//!   [`RunContext::reject_tool_for_agent`]; a decision on an interruption the runner raised is
//!   bound automatically.
//!
//! Python's `python` / `json` converters map to an in-memory copy and a JSON string round trip of
//! the exported approvals and tool invocations.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::items::{FunctionCall, RunItemKind, ToolApprovalItem};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_context::{
    ApprovalCapability, ApprovalsState, function_tool_capability_fingerprint,
};
use openai_agents_sdk::run_state::ToolInvocationRecord;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunContext, RunInput, RunResult, RunState, Runner};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

const PUBLIC: &str = "Return public data.";
const PRIVATE: &str = "Return private data.";

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn entries(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

fn model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

/// A tool named `operate` that needs approval, records `label` and returns `output`.
fn operate(description: &str, label: &str, output: &str, executed: &Log) -> FunctionTool<()> {
    let executed = executed.clone();
    let label = label.to_string();
    let output = output.to_string();
    FunctionTool::new("operate", description, move |_c, _a: NoArgs| {
        let executed = executed.clone();
        let label = label.clone();
        let output = output.clone();
        async move {
            executed.lock().unwrap().push(label);
            Ok(output)
        }
    })
    .with_needs_approval(true)
}

fn agent(name: &str, tool: FunctionTool<()>) -> Agent<()> {
    Agent::<()>::new(name).with_tool(tool)
}

fn call_item(agent: &str, call_id: &str) -> ToolApprovalItem {
    let mut raw = FunctionCall::new("operate", "{}");
    raw.call_id = call_id.to_string();
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: raw,
        tool_name: "operate".into(),
        tool_origin: None,
    }
}

fn shell_item(agent: &str, call_id: &str) -> ToolApprovalItem {
    ToolApprovalItem {
        agent: agent.into(),
        raw_item: serde_json::from_value(json!({
            "type": "shell_call", "name": "shell", "id": call_id, "call_id": call_id,
            "status": "completed",
            "action": {"commands": ["echo", "ok"]},
        }))
        .unwrap(),
        tool_name: "shell".into(),
        tool_origin: None,
    }
}

/// Move a context's approval state into a fresh context: `python` copies it in memory, `json`
/// round-trips it through a JSON string.
fn convert(ctx: &RunContext<()>, converter: &str) -> RunContext<()> {
    let approvals = ctx.export_approvals();
    let invocations = ctx.export_tool_invocations();
    let capabilities = ctx.export_approval_capabilities();
    let (approvals, invocations, capabilities): (
        ApprovalsState,
        BTreeMap<String, ToolInvocationRecord>,
        BTreeMap<String, ApprovalCapability>,
    ) = match converter {
        "python" => (approvals, invocations, capabilities),
        "json" => {
            let text = serde_json::to_string(&(approvals, invocations, capabilities)).unwrap();
            serde_json::from_str(&text).unwrap()
        }
        other => panic!("unknown converter {other}"),
    };
    let restored = RunContext::new(());
    restored.import_approvals(&approvals);
    restored.import_tool_invocations(&invocations);
    restored.import_approval_capabilities(&capabilities);
    restored
}

/// Run streamed or not, draining the stream.
async fn run_mode(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> openai_agents_sdk::Result<RunResult<()>> {
    if !streamed {
        return Runner::run_with_options(agent, input, options).await;
    }
    let mut stream = Runner::run_streamed_with_options(agent, input, options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => stream
            .into_result()
            .ok_or_else(|| AgentsError::user("stream produced no result")),
    }
}

fn tool_outputs(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect()
}

/// `test_converted_context_does_not_authorize_same_named_distinct_capability`: an approval
/// granted for `low_privilege` (exposed as `operate` by agent `owner`) does not let a different
/// `operate` tool on another agent named `owner` run after the context is transported. The run
/// interrupts; approving that interruption on the run state binds the new tool, which then runs.
#[tokio::test]
async fn converted_context_does_not_authorize_same_named_distinct_capability() {
    for converter in ["python", "json"] {
        for always in [false, true] {
            for streamed in [false, true] {
                let label = format!("converter={converter} always={always} streamed={streamed}");
                let executed = log();
                let source = agent("owner", operate(PUBLIC, "low", "public data", &executed));
                let ctx = RunContext::new(());
                ctx.approve_tool_for_agent(&source, &call_item("owner", "approved"), always)
                    .unwrap();
                let restored = convert(&ctx, converter);

                let requested = if always { "next" } else { "approved" };
                let model = model();
                model.add_turn(vec![function_call_with_id("operate", "{}", requested)]);
                model.add_turn(vec![assistant_message("done")]);
                let configured = agent(
                    "owner",
                    operate(PRIVATE, "privileged", "private data", &executed),
                )
                .with_model_instance(model);

                let result = run_mode(
                    &configured,
                    "start",
                    RunOptions::with_run_context(restored),
                    streamed,
                )
                .await
                .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(entries(&executed).is_empty(), "{label}");
                assert_eq!(result.interruptions.len(), 1, "{label}");
                assert_eq!(result.interruptions[0].agent, "owner", "{label}");
                assert!(tool_outputs(&result).is_empty(), "{label}");

                let mut state = result.to_state();
                state.approve(&result.interruptions[0], false);
                let resumed = run_mode(&configured, state, RunOptions::new(()), streamed)
                    .await
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(resumed.final_output, Some(json!("done")), "{label}");
                assert_eq!(entries(&executed), vec!["privileged"], "{label}");
            }
        }
    }
}

/// The same scenario, but the decision on the interruption is taken after the run state went
/// through JSON: the capability the runner asked for travels with the state, so the approval is
/// still bound, and a sticky one keeps authorizing that tool only.
#[tokio::test]
async fn decision_after_state_round_trip_stays_bound_to_the_asking_tool() {
    for streamed in [false, true] {
        let label = format!("streamed={streamed}");
        let executed = log();
        let model = model();
        model.add_turn(vec![function_call_with_id("operate", "{}", "first")]);
        model.add_turn(vec![function_call_with_id("operate", "{}", "second")]);
        model.add_turn(vec![assistant_message("done")]);
        let configured = agent(
            "owner",
            operate(PRIVATE, "privileged", "private", &executed),
        )
        .with_model_instance(model);
        let result = run_mode(&configured, "start", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(result.interruptions.len(), 1, "{label}");

        let restored = RunState::from_json(result.to_state().to_json().unwrap()).unwrap();
        let capability = restored.approval_capabilities.get("first").cloned();
        assert_eq!(
            capability.as_ref().map(|c| c.fingerprint.clone()),
            Some(function_tool_capability_fingerprint(
                "owner",
                &operate(PRIVATE, "x", "x", &log())
            )),
            "{label}"
        );
        let mut restored = restored;
        restored.approve(&result.interruptions[0], true);
        let bound_keys: Vec<String> = restored
            .approvals
            .0
            .keys()
            .filter(|key| key.contains('\u{3}'))
            .cloned()
            .collect();
        assert_eq!(bound_keys.len(), 1, "{label}: {bound_keys:?}");

        let resumed = run_mode(&configured, restored, RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert!(resumed.interruptions.is_empty(), "{label}");
        assert_eq!(resumed.final_output, Some(json!("done")), "{label}");
        assert_eq!(
            entries(&executed),
            vec!["privileged", "privileged"],
            "{label}"
        );
    }
}

/// `test_context_converters_preserve_distinct_same_named_owners`: two decisions of same-named
/// owners about distinct capabilities both survive transport, and a name-only query cannot
/// choose between their sticky decisions.
#[test]
fn context_converters_preserve_distinct_same_named_owners() {
    for converter in ["python", "json"] {
        let executed = log();
        let first = agent("same-name", operate(PUBLIC, "a", "a", &executed));
        let second = agent("same-name", operate(PRIVATE, "b", "b", &executed));
        let ctx = RunContext::new(());
        ctx.approve_tool_for_agent(&first, &call_item("same-name", "call-0"), true)
            .unwrap();
        ctx.reject_tool_for_agent(
            &second,
            &call_item("same-name", "call-1"),
            true,
            Some("Second owner".into()),
        )
        .unwrap();
        let restored = convert(&convert(&ctx, converter), converter);
        assert_eq!(
            restored.is_tool_approved("same-name", "operate", "call-0"),
            None,
            "{converter}"
        );
        assert_eq!(
            restored.rejection_message("same-name", "operate", "call-1"),
            None,
            "{converter}"
        );
        assert_eq!(restored.export_approvals().0.len(), 2, "{converter}");
    }
}

/// `test_name_only_query_does_not_choose_between_agent_owners`: one owner's sticky approval
/// answers a name-only query; once a same-named owner of another capability rejects stickily,
/// the query has no answer. (Python's `get_approval_status` without an invocation returns
/// `None` even for one owner; Rust's name-based `get_approval_status` is the same query as
/// `is_tool_approved`, so that assertion has no Rust counterpart.)
#[test]
fn name_only_query_does_not_choose_between_agent_owners() {
    let executed = log();
    let first = agent("same", operate(PUBLIC, "a", "a", &executed));
    let second = agent("same", operate(PRIVATE, "b", "b", &executed));
    let ctx = RunContext::new(());
    ctx.approve_tool_for_agent(&first, &call_item("same", "first"), true)
        .unwrap();
    assert_eq!(
        ctx.is_tool_approved("same", "operate", "future"),
        Some(true)
    );
    ctx.reject_tool_for_agent(
        &second,
        &call_item("same", "second"),
        true,
        Some("second denied".into()),
    )
    .unwrap();
    assert_eq!(ctx.is_tool_approved("same", "operate", "future"), None);
    assert_eq!(ctx.rejection_message("same", "operate", "future"), None);
}

/// `test_name_only_inspection_selects_a_unique_exact_call`: an exact call decision identifies
/// its owner for a name-only query even when another owner of the same name decided too.
///
/// Python also checks that the inspection is not an execution grant for the other owner
/// (`get_approval_status(current_invocation=other_owner)`); Rust's item-based query carries no
/// tool, so that half is checked through the runner in
/// `converted_context_does_not_authorize_same_named_distinct_capability`.
#[test]
fn name_only_inspection_selects_a_unique_exact_call() {
    for converter in ["none", "python", "json"] {
        let executed = log();
        let first = agent("same", operate(PUBLIC, "a", "a", &executed));
        let second = agent("same", operate(PRIVATE, "b", "b", &executed));
        let ctx = RunContext::new(());
        let approved = call_item("same", "call-a");
        let rejected = call_item("same", "call-b");
        ctx.approve_tool_for_agent(&first, &approved, false)
            .unwrap();
        ctx.reject_tool_for_agent(&second, &rejected, false, Some("Second call denied".into()))
            .unwrap();
        let inspect = || match converter {
            "none" => ctx.clone(),
            other => convert(&ctx, other),
        };

        let observed = inspect();
        assert_eq!(
            observed.is_tool_approved("same", "operate", "call-a"),
            Some(true),
            "{converter}"
        );
        assert_eq!(
            observed.is_tool_approved("same", "operate", "call-b"),
            Some(false),
            "{converter}"
        );
        assert_eq!(
            observed.rejection_message("same", "operate", "call-a"),
            None
        );
        assert_eq!(
            observed
                .rejection_message("same", "operate", "call-b")
                .as_deref(),
            Some("Second call denied"),
            "{converter}"
        );
        assert_eq!(observed.is_tool_approved("same", "operate", "future"), None);

        ctx.approve_tool_for_agent(&second, &rejected, true)
            .unwrap();
        ctx.reject_tool_for_agent(&first, &approved, false, Some("First call denied".into()))
            .unwrap();
        let observed = inspect();
        assert_eq!(
            observed.is_tool_approved("same", "operate", "call-a"),
            Some(false),
            "{converter}"
        );
        assert_eq!(
            observed
                .rejection_message("same", "operate", "call-a")
                .as_deref(),
            Some("First call denied"),
            "{converter}"
        );
        // A unique exact decision wins inspection over another owner's sticky default.
        assert_eq!(
            observed.is_tool_approved("same", "operate", "future"),
            None,
            "{converter}"
        );
        assert_eq!(
            observed.rejection_message("same", "operate", "future"),
            None
        );
    }
}

/// `test_context_converters_preserve_function_decisions`: a bound decision survives transport,
/// answers name-only queries, and leaves a native tool's own rejection alone.
#[test]
fn context_converters_preserve_function_decisions() {
    for converter in ["python", "json"] {
        for decision in ["approve", "reject", "always_approve", "always_reject"] {
            let label = format!("converter={converter} decision={decision}");
            let executed = log();
            let owner = agent("same-name", operate(PUBLIC, "a", "a", &executed));
            let approval = call_item("same-name", "first");
            let ctx = RunContext::new(());
            let approved = decision.ends_with("approve");
            let always = decision.starts_with("always_");
            if approved {
                ctx.approve_tool_for_agent(&owner, &approval, always)
                    .unwrap();
            } else {
                ctx.reject_tool_for_agent(&owner, &approval, always, Some("Owner rejected".into()))
                    .unwrap();
            }
            let native = shell_item("same-name", "native");
            ctx.reject_tool(&native, false, Some("Native rejected".into()));

            let restored = convert(&convert(&ctx, converter), converter);
            assert_eq!(
                restored.is_tool_approved("same-name", "operate", "first"),
                Some(approved),
                "{label}"
            );
            assert_eq!(
                restored.is_tool_approved("same-name", "operate", "next"),
                always.then_some(approved),
                "{label}"
            );
            assert_eq!(
                restored.rejection_message("same-name", "operate", "first"),
                (!approved).then(|| "Owner rejected".to_string()),
                "{label}"
            );
            let native_status = restored.approval_status_for_item(&native, "native");
            assert_eq!(native_status.approved, Some(false), "{label}");
            assert_eq!(
                native_status.rejection_message.as_deref(),
                Some("Native rejected"),
                "{label}"
            );
            assert_eq!(
                ctx.approval_status_for_item(&approval, "first").approved,
                Some(approved),
                "{label}"
            );
        }
    }
}

/// `test_runner_executes_converted_context_decisions`: a bound decision, transported, is
/// executed by a run of an agent exposing the same capability: approved calls run, rejected ones
/// answer with the rejection message, sticky decisions cover the next call too.
#[tokio::test]
async fn runner_executes_converted_context_decisions() {
    for converter in ["python", "json"] {
        for decision in ["approve", "reject", "always_approve", "always_reject"] {
            for streamed in [false, true] {
                let label =
                    format!("converter={converter} decision={decision} streamed={streamed}");
                let executed = log();
                let source = agent("owner", operate(PUBLIC, "executed", "result", &executed));
                let ctx = RunContext::new(());
                let approved = decision.ends_with("approve");
                let always = decision.starts_with("always_");
                let approval = call_item("owner", "first");
                if approved {
                    ctx.approve_tool_for_agent(&source, &approval, always)
                        .unwrap();
                } else {
                    ctx.reject_tool_for_agent(
                        &source,
                        &approval,
                        always,
                        Some("Owner rejected".into()),
                    )
                    .unwrap();
                }
                let restored = convert(&convert(&ctx, converter), converter);
                let model = model();
                model.add_turn(vec![function_call_with_id("operate", "{}", "first")]);
                if always {
                    model.add_turn(vec![function_call_with_id("operate", "{}", "next")]);
                }
                model.add_turn(vec![assistant_message("done")]);
                let configured = source.clone().with_model_instance(model);
                let result = run_mode(
                    &configured,
                    "start",
                    RunOptions::with_run_context(restored),
                    streamed,
                )
                .await
                .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(result.interruptions.is_empty(), "{label}");
                assert_eq!(result.final_output, Some(json!("done")), "{label}");
                let calls = if always { 2 } else { 1 };
                let expected_runs = if approved { calls } else { 0 };
                assert_eq!(entries(&executed).len(), expected_runs, "{label}");
                let output = if approved { "result" } else { "Owner rejected" };
                assert_eq!(tool_outputs(&result), vec![json!(output); calls], "{label}");
                RunState::from_json(result.to_state().to_json().unwrap())
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
            }
        }
    }
}

/// `test_runner_requires_reapproval_for_unresolved_context_owners`, `missing` row: a decision
/// transported to a run whose agent has another name is asked again, and a fresh decision on
/// that interruption leaves a serializable state.
#[tokio::test]
async fn runner_requires_reapproval_for_a_renamed_owner() {
    let executed = log();
    let source = agent("owner", operate(PUBLIC, "unexpected", "x", &executed));
    let ctx = RunContext::new(());
    ctx.approve_tool_for_agent(&source, &call_item("owner", "first"), true)
        .unwrap();
    let restored = convert(&ctx, "json");
    let model = model();
    model.add_turn(vec![function_call_with_id("operate", "{}", "first")]);
    let configured = agent("different", operate(PUBLIC, "unexpected", "x", &executed))
        .with_model_instance(model);
    let result =
        Runner::run_with_options(&configured, "start", RunOptions::with_run_context(restored))
            .await
            .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].agent, "different");
    assert!(entries(&executed).is_empty());
    let mut state = result.to_state();
    state.reject(
        &result.interruptions[0],
        false,
        Some("Fresh decision".into()),
    );
    RunState::from_json(state.to_json().unwrap()).unwrap();
}

/// A bound decision whose capability matches no tool of the agent is refused, and so is binding
/// a hosted MCP request; nothing is recorded either way.
#[test]
fn binding_requires_a_function_tool_of_the_agent() {
    let executed = log();
    let owner = agent("owner", operate(PUBLIC, "a", "a", &executed));
    let ctx = RunContext::new(());
    let mut other = call_item("owner", "c1");
    other.tool_name = "missing".into();
    other.raw_item.name = "missing".into();
    let err = ctx
        .approve_tool_for_agent(&owner, &other, true)
        .unwrap_err();
    assert!(
        err.to_string().contains("no function tool named 'missing'"),
        "{err}"
    );
    let mut hosted = call_item("owner", "req-1");
    hosted
        .raw_item
        .extra
        .insert("type".into(), json!("mcp_approval_request"));
    hosted
        .raw_item
        .extra
        .insert("server_label".into(), json!("server"));
    assert!(
        ctx.reject_tool_for_agent(&owner, &hosted, false, None)
            .is_err()
    );
    assert!(ctx.export_approvals().0.is_empty());
}

/// The fingerprint is a pure function of what the agent exposes: rebuilt tools agree, and each
/// of agent name, description and parameter schema changes it. The value is pinned so a change
/// of the hashed payload, which would re-prompt every persisted bound decision, is deliberate.
#[test]
fn capability_fingerprint_is_stable_and_covers_the_visible_signature() {
    let executed = log();
    let base = function_tool_capability_fingerprint("owner", &operate(PUBLIC, "a", "a", &executed));
    assert_eq!(
        base,
        function_tool_capability_fingerprint("owner", &operate(PUBLIC, "b", "b", &executed))
    );
    assert_eq!(base.len(), 64);
    assert_eq!(base, PINNED_FINGERPRINT);
    assert_ne!(
        base,
        function_tool_capability_fingerprint("other", &operate(PUBLIC, "a", "a", &executed))
    );
    assert_ne!(
        base,
        function_tool_capability_fingerprint("owner", &operate(PRIVATE, "a", "a", &executed))
    );
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct WithArg {
        #[allow(dead_code)]
        value: String,
    }
    let with_arg = FunctionTool::<()>::new("operate", PUBLIC, |_c, _a: WithArg| async move {
        Ok("x".to_string())
    });
    assert_ne!(
        base,
        function_tool_capability_fingerprint("owner", &with_arg)
    );
}

const PINNED_FINGERPRINT: &str = "33466ad2b6431239f5a7f69aab38c4f80e99275edd1a2efdacbf484ec372fe11";

/// The approvals JSON keeps an unbound record's shape and adds bound records under their own
/// keys; a run state writes bound records as Python agent-owned entries plus a `capability`
/// field, and the registered capabilities under its Rust-only `rust` key.
#[test]
fn bound_approvals_round_trip_through_json() {
    let executed = log();
    let owner = agent("owner", operate(PUBLIC, "a", "a", &executed));
    let ctx = RunContext::new(());
    ctx.approve_tool(&call_item("owner", "plain"), false);
    ctx.approve_tool_for_agent(&owner, &call_item("owner", "bound"), true)
        .unwrap();
    let approvals = ctx.export_approvals();
    let text = serde_json::to_value(&approvals).unwrap();
    let fp = function_tool_capability_fingerprint("owner", &operate(PUBLIC, "a", "a", &executed));
    assert_eq!(
        text["owner\u{1}operate"],
        json!({
            "approved_always": false,
            "approved_call_ids": ["plain"],
            "rejected_always": false,
            "rejected_call_ids": [],
            "rejection_messages": {},
        })
    );
    assert_eq!(
        text[format!("owner\u{1}operate\u{3}{fp}")]["approved_always"],
        json!(true)
    );
    let back: ApprovalsState = serde_json::from_value(text).unwrap();
    assert_eq!(back, approvals);

    let mut state = RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.into(),
        current_agent_name: "owner".into(),
        approvals,
        ..Default::default()
    };
    state.approval_capabilities.insert(
        "pending".into(),
        ApprovalCapability {
            agent: "owner".into(),
            tool_key: "operate".into(),
            fingerprint: fp.clone(),
        },
    );
    let wire = state.to_json().unwrap();
    // Both records are agent-owned entries Python can read; the bound one also names its
    // capability, a field Python ignores.
    let records = wire["context"]["function_tool_approvals"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(records.len(), 2, "{records:?}");
    assert!(records.iter().all(|r| r["tool_key"] == json!("operate")));
    assert_eq!(
        records
            .iter()
            .filter_map(|r| r.get("capability"))
            .collect::<Vec<_>>(),
        vec![&json!(fp)]
    );
    assert_eq!(
        wire["rust"]["approval_capabilities"]["pending"]["fingerprint"],
        json!(fp)
    );
    let restored = RunState::from_json(wire).unwrap();
    assert_eq!(restored.approvals, state.approvals);
    assert_eq!(restored.approval_capabilities, state.approval_capabilities);

    // A decision on the registered call after the round trip is bound to its capability.
    let mut restored = restored;
    restored.reject(&call_item("owner", "pending"), false, None);
    assert_eq!(
        restored.approvals.0[&format!("owner\u{1}operate\u{3}{fp}")].rejected_call_ids,
        ["pending".to_string()].into_iter().collect()
    );
}

/// An `ApprovalsState` serialized before decisions were bound still loads, still answers
/// name-based queries, and its unbound sticky approval still lets the runner execute without
/// asking, whatever the tool's description.
#[tokio::test]
async fn pre_binding_approvals_json_still_authorizes_by_name() {
    let legacy = r#"{"owner\u0001operate":{"approved_always":true,"approved_call_ids":[],"rejected_always":false,"rejected_call_ids":[],"rejection_messages":{}}}"#;
    let approvals: ApprovalsState = serde_json::from_str(legacy).unwrap();
    for streamed in [false, true] {
        let ctx = RunContext::new(());
        ctx.import_approvals(&approvals);
        assert_eq!(ctx.is_tool_approved("owner", "operate", "any"), Some(true));
        let executed = log();
        let model = model();
        model.add_turn(vec![function_call_with_id("operate", "{}", "c1")]);
        model.add_turn(vec![assistant_message("done")]);
        let configured =
            agent("owner", operate(PRIVATE, "ran", "ok", &executed)).with_model_instance(model);
        let result = run_mode(
            &configured,
            "start",
            RunOptions::with_run_context(ctx),
            streamed,
        )
        .await
        .unwrap();
        assert!(result.interruptions.is_empty(), "streamed={streamed}");
        assert_eq!(entries(&executed), vec!["ran"], "streamed={streamed}");
    }
}
