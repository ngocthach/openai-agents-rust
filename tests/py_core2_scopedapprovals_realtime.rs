//! Realtime halves of `tests/test_function_tool_approval_arguments.py`: a realtime session's
//! conditional approval sees the same invocation that can execute, as the runner's does.

#![cfg(feature = "realtime")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::error::AgentsError;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModelEvent, RealtimeModelToolCallEvent,
    RealtimeRunConfig, RealtimeSession, RealtimeSessionEvent,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use serde::Deserialize;
use serde_json::{Value, json};

type Log<T> = Arc<Mutex<Vec<T>>>;

const WAIT: Duration = Duration::from_secs(5);

fn log<T>() -> Log<T> {
    Arc::new(Mutex::new(Vec::new()))
}

async fn start(model: &Arc<FakeRealtimeModel>, tool: FunctionTool<()>) -> RealtimeSession<()> {
    let agent = RealtimeAgent::<()>::new("test").with_tool(tool).into_arc();
    let config = RealtimeRunConfig {
        async_tool_calls: false,
        ..Default::default()
    };
    RealtimeSession::connect(model.clone(), agent, (), None, Some(config))
        .await
        .expect("session connects")
}

/// Emit a tool call, then wait until the session has handled it.
async fn handle_call(
    model: &FakeRealtimeModel,
    session: &RealtimeSession<()>,
    arguments: &str,
) -> usize {
    model.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("operation", "one", arguments),
    ));
    model.emit(RealtimeModelEvent::Other {
        data: json!("barrier"),
    });
    let mut approvals = 0;
    loop {
        let event = tokio::time::timeout(WAIT, session.recv())
            .await
            .expect("timed out")
            .expect("stream ended");
        match event {
            RealtimeSessionEvent::ToolApprovalRequired { .. } => approvals += 1,
            RealtimeSessionEvent::RawModelEvent {
                data: RealtimeModelEvent::Other { data },
                ..
            } if data == "barrier" => return approvals,
            _ => {}
        }
    }
}

fn sent_outputs(model: &FakeRealtimeModel) -> Vec<String> {
    model
        .sent_events()
        .iter()
        .filter_map(|e| e.as_tool_output().map(|(_, output, _)| output.to_string()))
        .collect()
}

fn recording_policy(tool: FunctionTool<()>, seen: &Log<Value>, decide: bool) -> FunctionTool<()> {
    let seen = seen.clone();
    tool.with_needs_approval_fn(move |_ctx, args, _id| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push(args);
            decide
        }
    })
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ValueArg {
    #[serde(default = "safe")]
    value: String,
}

fn safe() -> String {
    "safe".into()
}

/// Python: `test_realtime_prepared_approval_and_errors` (3 argument cases).
#[tokio::test]
async fn realtime_prepared_approval_and_errors() {
    for (arguments, pending, policy_count) in [
        ("{}", true, 0),
        (r#"{"value":"safe"}"#, false, 1),
        (r#"{"value":3}"#, true, 0),
    ] {
        let seen = log();
        let effects = log::<String>();
        let e = effects.clone();
        let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: ValueArg| {
            let e = e.clone();
            async move {
                e.lock().unwrap().push(args.value.clone());
                Ok(args.value)
            }
        });
        let tool = recording_policy(tool, &seen, false);
        let model = Arc::new(FakeRealtimeModel::new());
        let session = start(&model, tool).await;
        let approvals = handle_call(&model, &session, arguments).await;
        assert_eq!(approvals > 0, pending, "{arguments}");
        assert_eq!(seen.lock().unwrap().len(), policy_count, "{arguments}");
        if pending {
            assert!(effects.lock().unwrap().is_empty());
            session.approve_tool_call("one", false).await.unwrap();
            let want: &[&str] = if arguments == r#"{"value":3}"# {
                &[]
            } else {
                &["safe"]
            };
            assert_eq!(*effects.lock().unwrap(), want, "{arguments}");
        } else {
            assert_eq!(*effects.lock().unwrap(), ["safe"]);
        }
        assert_eq!(sent_outputs(&model).len(), 1, "{arguments}");
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct NumberArgs {
    number: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Failure {
    Default,
    Custom,
    Raise,
}

/// Python: `test_invalid_typed_input_requires_approval_before_failure_policy`, realtime mode
/// (failure x approve).
#[tokio::test]
async fn realtime_invalid_typed_input_requires_approval_before_failure_policy() {
    for failure in [Failure::Default, Failure::Custom, Failure::Raise] {
        for approve in [false, true] {
            let label = format!("{failure:?} approve={approve}");
            let seen = log();
            let effects = log::<i64>();
            let formatted = log::<bool>();
            let guardrails = log::<()>();
            let g = guardrails.clone();
            let e = effects.clone();
            let mut tool =
                FunctionTool::<()>::new("operation", "", move |_ctx, args: NumberArgs| {
                    let e = e.clone();
                    async move {
                        e.lock().unwrap().push(args.number);
                        Ok(args.number.to_string())
                    }
                })
                .with_input_guardrail(ToolInputGuardrail::new(
                    "guardrail",
                    move |_d| {
                        let g = g.clone();
                        async move {
                            g.lock().unwrap().push(());
                            Ok(ToolGuardrailFunctionOutput::allow())
                        }
                    },
                ));
            tool = recording_policy(tool, &seen, true);
            match failure {
                Failure::Default => {}
                Failure::Custom => {
                    let f = formatted.clone();
                    tool = tool.with_failure_error_function(move |_ctx, error| {
                        f.lock()
                            .unwrap()
                            .push(matches!(error, AgentsError::ModelBehaviorError { .. }));
                        "correct the number".to_string()
                    });
                }
                Failure::Raise => tool = tool.without_failure_error_function(),
            }
            let model = Arc::new(FakeRealtimeModel::new());
            let session = start(&model, tool).await;
            let approvals = handle_call(&model, &session, r#"{"number":"invalid"}"#).await;
            assert_eq!(approvals, 1, "{label}");
            assert!(seen.lock().unwrap().is_empty(), "{label}");
            assert!(guardrails.lock().unwrap().is_empty(), "{label}");
            assert!(sent_outputs(&model).is_empty(), "{label}");
            if !approve {
                session.reject_tool_call("one", false, None).await.unwrap();
            } else if failure == Failure::Raise {
                let err = session.approve_tool_call("one", false).await.unwrap_err();
                assert!(
                    matches!(err, AgentsError::ModelBehaviorError { .. }),
                    "{label}: {err:?}"
                );
                assert!(sent_outputs(&model).is_empty(), "{label}");
            } else {
                session.approve_tool_call("one", false).await.unwrap();
                let out = sent_outputs(&model);
                assert_eq!(out.len(), 1, "{label}");
                if failure == Failure::Custom {
                    assert_eq!(out, ["correct the number"], "{label}");
                }
            }
            assert!(seen.lock().unwrap().is_empty(), "{label}");
            assert!(effects.lock().unwrap().is_empty(), "{label}");
            let want: &[bool] = if approve && failure == Failure::Custom {
                &[true]
            } else {
                &[]
            };
            assert_eq!(*formatted.lock().unwrap(), want, "{label}");
            assert_eq!(
                guardrails.lock().unwrap().len(),
                usize::from(approve),
                "{label}"
            );
        }
    }
}
