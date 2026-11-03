//! Ports of the agent-as-tool cases of `test_agent_as_tool.py` that were filed as Python-only.
//!
//! Python mutates the nested `RunState` in place while an approved nested resume is in flight
//! (`record_agent_tool_resume_state`); a Rust run consumes its state, so the progress lands in the
//! `RunCheckpoint` the caller gave the outer run.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openai_agents_sdk::items::{FunctionCall, OutputItem, ToolApprovalItem};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::run_context::ApprovalsState;
use openai_agents_sdk::testing::{
    FakeModel, ScriptedModel, ScriptedStep, assistant_message, function_call_full,
    function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool};
use openai_agents_sdk::{Agent, RunOptions, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Notify;

fn call(name: &str, args: serde_json::Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

/// Python: `test_agent_as_tool_resume_survives_cancellation_after_nested_output_commit`.
/// The nested run commits the approved tool's output, then its next model call is cancelled
/// together with the outer run. Resuming from the outer run's checkpoint must not run the nested
/// tool again, and the nested model is called once more (three calls in total).
async fn resume_survives_cancellation(streamed: bool) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let attempts = Arc::new(Mutex::new(Vec::<String>::new()));
    let waiting = Arc::new(Notify::new());

    let sensitive = {
        let attempts = attempts.clone();
        FunctionTool::<()>::new("sensitive", "Needs approval", move |_ctx, _args: NoArgs| {
            let attempts = attempts.clone();
            async move {
                attempts.lock().unwrap().push("ran".into());
                Ok("inner value".to_string())
            }
        })
        .with_needs_approval(true)
    };
    let inner_model = Arc::new(ScriptedModel::new());
    inner_model
        .extend(vec![
            ScriptedStep::output(vec![call("sensitive", json!({}), "inner_call")]),
            {
                let waiting = waiting.clone();
                ScriptedStep::respond_async(move |_call| {
                    let waiting = waiting.clone();
                    async move {
                        waiting.notify_one();
                        std::future::pending().await
                    }
                })
            },
            ScriptedStep::output(vec![assistant_message("inner done")]),
        ])
        .unwrap();
    let inner = Agent::<()>::new("inner")
        .with_model_instance(inner_model.clone())
        .with_tool(sensitive);
    let outer_model = Arc::new(ScriptedModel::new());
    outer_model
        .extend(vec![
            ScriptedStep::output(vec![call("delegate", json!({"input": "hi"}), "outer_call")]),
            ScriptedStep::output(vec![assistant_message("outer done")]),
        ])
        .unwrap();
    let outer = Arc::new(
        Agent::<()>::new("outer")
            .with_model_instance(outer_model.clone())
            .with_tool(inner.as_tool("delegate", "Delegate")),
    );

    let interrupted = Runner::run(&*outer, "go").await.unwrap();
    let mut state = RunState::from_json(interrupted.to_state().to_json().unwrap()).unwrap();
    let interruption = state.get_interruptions()[0].clone();
    state.approve(&interruption, false);

    let slot = RunCheckpoint::new();
    let cancel: Box<
        dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>,
    > = if streamed {
        let options = RunOptions::new(()).checkpoint(&slot);
        let mut result = Runner::run_streamed_with_options(&*outer, state.clone(), options);
        let handle = result.cancel_handle();
        let consumer = tokio::spawn(async move {
            let mut events = std::pin::pin!(result.stream_events());
            while events.next().await.is_some() {}
        });
        Box::new(move || {
            Box::pin(async move {
                handle.cancel(CancelMode::Immediate);
                consumer.await.unwrap();
            })
        })
    } else {
        let task = {
            let outer = outer.clone();
            let state = state.clone();
            let slot = slot.clone();
            tokio::spawn(async move {
                let options = RunOptions::new(()).checkpoint(&slot);
                let _ = Runner::run_with_options(&*outer, state, options).await;
            })
        };
        Box::new(move || {
            Box::pin(async move {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            })
        })
    };
    tokio::time::timeout(Duration::from_secs(5), waiting.notified())
        .await
        .expect("the nested model call never started");
    assert_eq!(*attempts.lock().unwrap(), vec!["ran".to_string()]);
    assert_eq!(inner_model.calls().len(), 2);
    cancel().await;

    let checkpoint = slot
        .take()
        .unwrap_or_else(|| panic!("streamed={streamed}: the cancelled run left no checkpoint"));
    let result = Runner::run(&*outer, checkpoint).await.unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("outer done"),
        "streamed={streamed}"
    );
    assert_eq!(
        *attempts.lock().unwrap(),
        vec!["ran".to_string()],
        "streamed={streamed}: the nested tool ran again"
    );
    assert_eq!(inner_model.calls().len(), 3, "streamed={streamed}");
}

#[tokio::test]
async fn agent_tool_resume_survives_cancellation_after_nested_output_commit_non_streamed() {
    resume_survives_cancellation(false).await;
}

#[tokio::test]
async fn agent_tool_resume_survives_cancellation_after_nested_output_commit_streamed() {
    resume_survives_cancellation(true).await;
}

fn mcp_request(id: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "mcp_approval_request", "id": id, "server_label": "accounts",
        "name": "lookup_account", "arguments": "{}"
    }))
}

/// An outer agent delegating once to an inner agent that owns a hosted MCP tool whose requests
/// always need approval; the inner model asks twice (`inner-1`, then `inner-2`).
fn hosted_mcp_pair() -> (Agent<()>, Arc<FakeModel>) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let inner_model = Arc::new(FakeModel::new());
    inner_model.add_turn(vec![mcp_request("inner-1")]);
    inner_model.add_turn(vec![mcp_request("inner-2")]);
    inner_model.add_turn(vec![assistant_message("inner done")]);
    let inner = Agent::<()>::new("inner")
        .with_model_instance(inner_model.clone())
        .with_tool(HostedMCPTool::<()>::new(json!({
            "type": "mcp", "server_label": "accounts", "server_url": "https://example.invalid",
            "require_approval": "always"
        })));
    let outer_model = Arc::new(FakeModel::new());
    outer_model.add_turn(vec![function_call_with_id(
        "delegate",
        r#"{"input":"go"}"#,
        "outer",
    )]);
    outer_model.add_turn(vec![assistant_message("outer done")]);
    let outer = Agent::<()>::new("outer")
        .with_model_instance(outer_model)
        .with_tool(inner.as_tool("delegate", "Delegate"));
    (outer, inner_model)
}

fn mcp_responses_sent(model: &FakeModel) -> Vec<serde_json::Value> {
    model
        .calls()
        .last()
        .unwrap()
        .input
        .iter()
        .map(|item| item.0.clone())
        .filter(|item| item["type"] == "mcp_approval_response")
        .collect()
}

/// Python: `test_agent_as_tool_hosted_mcp_nested_sticky_decision_stays_scoped` with the real
/// hosted MCP request the nested run raises (the earlier port used a function tool): a sticky
/// decision covers the follow-up request with the same message, a one-off decision does not.
#[tokio::test]
async fn nested_hosted_mcp_sticky_decision_covers_the_follow_up_request() {
    for (approve, sticky) in [(true, true), (false, true), (true, false), (false, false)] {
        let (outer, inner_model) = hosted_mcp_pair();
        let paused = Runner::run(&outer, "go").await.unwrap();
        let label = format!("approve={approve} sticky={sticky}");
        assert_eq!(paused.interruptions.len(), 1, "{label}");
        assert_eq!(paused.interruptions[0].call_id(), "inner-1", "{label}");
        let mut state = paused.to_state();
        if approve {
            state.approve(&paused.interruptions[0], sticky);
        } else {
            state.reject(
                &paused.interruptions[0],
                sticky,
                Some("server-a denied".into()),
            );
        }
        let resumed = Runner::run(&outer, state).await.unwrap();
        if sticky {
            assert!(resumed.interruptions.is_empty(), "{label}");
            assert_eq!(resumed.final_output_text().as_deref(), Some("outer done"));
            let sent = mcp_responses_sent(&inner_model);
            assert_eq!(sent.len(), 2, "{label}: {sent:?}");
            for response in &sent {
                assert_eq!(response["approve"], json!(approve), "{label}");
                if !approve {
                    assert_eq!(response["reason"], "server-a denied", "{label}");
                }
            }
        } else {
            assert_eq!(resumed.interruptions.len(), 1, "{label}");
            assert_eq!(resumed.interruptions[0].call_id(), "inner-2", "{label}");
        }
    }
}

/// Python: `test_agent_as_tool_wrapped_hosted_mcp_exact_decision_resumes_run`. A hosted MCP
/// request carried inside a `hosted_tool_call` wrapper keeps its exact rejection and message.
#[test]
fn wrapped_hosted_mcp_exact_rejection_keeps_its_message() {
    let mut call = FunctionCall::new("lookup_account", "{}");
    call.call_id = "inner-1".into();
    call.extra.insert("type".into(), json!("hosted_tool_call"));
    call.extra.insert(
        "provider_data".into(),
        json!({
            "type": "mcp_approval_request", "id": "inner-1", "name": "lookup_account",
            "server_label": "accounts", "arguments": "{}"
        }),
    );
    let item = ToolApprovalItem {
        agent: "outer".into(),
        raw_item: call,
        tool_name: "lookup_account".into(),
        tool_origin: None,
    };
    let mut approvals = ApprovalsState::default();
    approvals
        .try_reject(&item, false, Some("exact denial".into()))
        .unwrap();
    let decided = approvals.status_for_item(&item, "inner-1");
    assert_eq!(decided.approved, Some(false));
    assert_eq!(decided.rejection_message.as_deref(), Some("exact denial"));
    // A different request id of the same tool is not covered by the exact decision.
    assert_eq!(approvals.status_for_item(&item, "inner-2").approved, None);
}
