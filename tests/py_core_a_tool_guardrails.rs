//! Port of `test_tool_guardrails.py`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailBehavior, ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData,
    ToolOutputGuardrail, ToolOutputGuardrailData,
};
use openai_agents_sdk::{Agent, AgentsError, RunContext, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Query {
    #[serde(default)]
    query: String,
}

fn mock_tool_context(arguments: &str) -> ToolContext<()> {
    ToolContext::direct(RunContext::new(()), "test", "test_tool", arguments)
}

// ------------------------------------------------- guardrail function outputs

#[tokio::test]
async fn tool_input_guardrail_runs_and_reports_behavior() {
    let allowing =
        ToolInputGuardrail::new("allowing", |_d: ToolInputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow())
        });
    let data = ToolInputGuardrailData {
        context: mock_tool_context(r#"{"param": "value"}"#),
        agent_name: "test".into(),
    };
    let result = allowing.run(data.clone()).await.unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::Allow);
    assert_eq!(result.output_info, json!(null));

    let raising = ToolInputGuardrail::new("raising", |_d: ToolInputGuardrailData<()>| async move {
        Ok(ToolGuardrailFunctionOutput::raise_exception())
    });
    let result = raising.run(data.clone()).await.unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::RaiseException);

    let with_info = ToolInputGuardrail::new("info", |_d: ToolInputGuardrailData<()>| async move {
        Ok(ToolGuardrailFunctionOutput::raise_exception().with_output_info(json!("test_info")))
    });
    let result = with_info.run(data).await.unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::RaiseException);
    assert_eq!(result.output_info, json!("test_info"));
}

#[tokio::test]
async fn tool_output_guardrail_runs_and_reports_behavior() {
    let data = ToolOutputGuardrailData {
        context: mock_tool_context(r#"{"param": "value"}"#),
        agent_name: "test".into(),
        output: openai_agents_sdk::tool::ToolOutput::text("test output"),
    };

    let allowing =
        ToolOutputGuardrail::new("allowing", |_d: ToolOutputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow())
        });
    let result = allowing.run(data.clone()).await.unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::Allow);
    assert_eq!(result.output_info, json!(null));

    let raising =
        ToolOutputGuardrail::new("raising", |_d: ToolOutputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::raise_exception().with_output_info(json!("test_info")))
        });
    let result = raising.run(data).await.unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::RaiseException);
    assert_eq!(result.output_info, json!("test_info"));
}

#[tokio::test]
async fn guardrails_carry_their_name() {
    let guardrail = ToolInputGuardrail::new(
        "Custom input name",
        |_d: ToolInputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("test_2")))
        },
    );
    assert_eq!(guardrail.name, "Custom input name");
    let out = guardrail
        .run(ToolInputGuardrailData {
            context: mock_tool_context("{}"),
            agent_name: "test".into(),
        })
        .await
        .unwrap();
    assert_eq!(out.output_info, json!("test_2"));

    let guardrail = ToolOutputGuardrail::new(
        "Custom output name",
        |_d: ToolOutputGuardrailData<()>| async move {
            Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("test_4")))
        },
    );
    assert_eq!(guardrail.name, "Custom output name");
}

#[tokio::test]
async fn password_blocking_input_guardrail() {
    let guardrail =
        ToolInputGuardrail::new("password", |d: ToolInputGuardrailData<()>| async move {
            if d.context.tool_arguments.to_lowercase().contains("password") {
                Ok(ToolGuardrailFunctionOutput::reject_content(
                    "Tool call blocked: contains password",
                )
                .with_output_info(json!({"blocked_word": "password"})))
            } else {
                Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("safe_input")))
            }
        });

    let result = guardrail
        .run(ToolInputGuardrailData {
            context: mock_tool_context(r#"{"message": "Hello password world"}"#),
            agent_name: "test".into(),
        })
        .await
        .unwrap();
    assert_eq!(
        result.behavior,
        ToolGuardrailBehavior::RejectContent {
            message: "Tool call blocked: contains password".into()
        }
    );
    assert_eq!(result.output_info["blocked_word"], json!("password"));

    let result = guardrail
        .run(ToolInputGuardrailData {
            context: mock_tool_context(r#"{"message": "Hello safe world"}"#),
            agent_name: "test".into(),
        })
        .await
        .unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::Allow);
    assert_eq!(result.output_info, json!("safe_input"));
}

#[tokio::test]
async fn mixed_behavior_output_guardrail() {
    let guardrail =
        ToolOutputGuardrail::new("mixed", |d: ToolOutputGuardrailData<()>| async move {
            let text = format!("{:?}", d.output).to_lowercase();
            if text.contains("dangerous") {
                Ok(ToolGuardrailFunctionOutput::raise_exception()
                    .with_output_info(json!({"reason": "dangerous_content"})))
            } else if text.contains("sensitive") {
                Ok(
                    ToolGuardrailFunctionOutput::reject_content("Content was filtered")
                        .with_output_info(json!({"reason": "sensitive_content"})),
                )
            } else {
                Ok(ToolGuardrailFunctionOutput::allow()
                    .with_output_info(json!({"status": "clean"})))
            }
        });

    let make = |output: &str| ToolOutputGuardrailData {
        context: mock_tool_context("{}"),
        agent_name: "test".into(),
        output: openai_agents_sdk::tool::ToolOutput::text(output),
    };

    let result = guardrail
        .run(make("This is dangerous content"))
        .await
        .unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::RaiseException);
    assert_eq!(result.output_info["reason"], json!("dangerous_content"));

    let result = guardrail.run(make("This is sensitive data")).await.unwrap();
    assert_eq!(
        result.behavior,
        ToolGuardrailBehavior::RejectContent {
            message: "Content was filtered".into()
        }
    );
    assert_eq!(result.output_info["reason"], json!("sensitive_content"));

    let result = guardrail.run(make("This is clean content")).await.unwrap();
    assert_eq!(result.behavior, ToolGuardrailBehavior::Allow);
    assert_eq!(result.output_info["status"], json!("clean"));
}

// -------------------------------------------------------- end-to-end reporting

fn guarded_agent(
    input_guardrails: Vec<ToolInputGuardrail<()>>,
    output_guardrails: Vec<ToolOutputGuardrail<()>>,
) -> (Agent<()>, Arc<FakeModel>) {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "guarded",
        r#"{"query": "secret"}"#,
        "guarded_1",
    )]);
    model.add_turn(vec![function_call_with_id(
        "guarded",
        r#"{"query": "secret"}"#,
        "guarded_2",
    )]);
    let mut tool = FunctionTool::new("guarded", "guarded", |_c, _a: Query| async move {
        Ok("tool output".to_string())
    });
    for g in input_guardrails {
        tool = tool.with_input_guardrail(g);
    }
    for g in output_guardrails {
        tool = tool.with_output_guardrail(g);
    }
    let agent = Agent::<()>::new("guarded_tool_agent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    (agent, model)
}

async fn run_until_max_turns(agent: &Agent<()>, streaming: bool) -> AgentsError {
    if streaming {
        let mut stream =
            Runner::run_streamed_with_options(agent, "go", RunOptions::new(()).max_turns(Some(2)));
        let mut error = None;
        {
            let mut events = std::pin::pin!(stream.stream_events());
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                }
            }
        }
        error.expect("stream surfaces the error")
    } else {
        Runner::run_with_options(agent, "go", RunOptions::new(()).max_turns(Some(2)))
            .await
            .unwrap_err()
    }
}

async fn tool_input_guardrail_results_reported_on_max_turns(streaming: bool) {
    let guardrail = ToolInputGuardrail::new(
        "input_rejects",
        |_d: ToolInputGuardrailData<()>| async move {
            Ok(
                ToolGuardrailFunctionOutput::reject_content("blocked by policy")
                    .with_output_info(json!("input_rejected")),
            )
        },
    );
    let (agent, _model) = guarded_agent(vec![guardrail], vec![]);
    let err = run_until_max_turns(&agent, streaming).await;
    let AgentsError::MaxTurnsExceeded { run_data, .. } = &err else {
        panic!("unexpected error: {err:?}");
    };
    let data = run_data.as_ref().expect("run data");
    let names: Vec<&str> = data
        .tool_input_guardrail_results
        .iter()
        .map(|r| r.guardrail_name.as_str())
        .collect();
    assert_eq!(names, vec!["input_rejects", "input_rejects"]);
    assert!(data.tool_output_guardrail_results.is_empty());
}

#[tokio::test]
async fn tool_input_guardrail_results_reported_on_max_turns_non_streamed() {
    tool_input_guardrail_results_reported_on_max_turns(false).await;
}

#[tokio::test]
async fn tool_input_guardrail_results_reported_on_max_turns_streamed() {
    tool_input_guardrail_results_reported_on_max_turns(true).await;
}

async fn tool_output_guardrail_results_reported_on_max_turns(streaming: bool) {
    let guardrail = ToolOutputGuardrail::new(
        "output_rejects",
        |_d: ToolOutputGuardrailData<()>| async move {
            Ok(
                ToolGuardrailFunctionOutput::reject_content("blocked by policy")
                    .with_output_info(json!("output_rejected")),
            )
        },
    );
    let (agent, _model) = guarded_agent(vec![], vec![guardrail]);
    let err = run_until_max_turns(&agent, streaming).await;
    let AgentsError::MaxTurnsExceeded { run_data, .. } = &err else {
        panic!("unexpected error: {err:?}");
    };
    let data = run_data.as_ref().expect("run data");
    let names: Vec<&str> = data
        .tool_output_guardrail_results
        .iter()
        .map(|r| r.guardrail_name.as_str())
        .collect();
    assert_eq!(names, vec!["output_rejects", "output_rejects"]);
    assert!(data.tool_input_guardrail_results.is_empty());
}

#[tokio::test]
async fn tool_output_guardrail_results_reported_on_max_turns_non_streamed() {
    tool_output_guardrail_results_reported_on_max_turns(false).await;
}

#[tokio::test]
async fn tool_output_guardrail_results_reported_on_max_turns_streamed() {
    tool_output_guardrail_results_reported_on_max_turns(true).await;
}

async fn tool_tripwire_preserves_completed_turn_results(streaming: bool) {
    let runs = Arc::new(AtomicUsize::new(0));
    let counter = runs.clone();
    let guardrail =
        ToolInputGuardrail::new("allow_then_raise", move |_d: ToolInputGuardrailData<()>| {
            let counter = counter.clone();
            async move {
                let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                if n == 2 {
                    Ok(ToolGuardrailFunctionOutput::raise_exception()
                        .with_output_info(json!("second_turn")))
                } else {
                    Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("first_turn")))
                }
            }
        });
    let (agent, _model) = guarded_agent(vec![guardrail], vec![]);

    let err = if streaming {
        let mut stream = Runner::run_streamed(&agent, "go");
        let mut error = None;
        {
            let mut events = std::pin::pin!(stream.stream_events());
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                }
            }
        }
        error.expect("stream surfaces the error")
    } else {
        Runner::run(&agent, "go").await.unwrap_err()
    };

    let AgentsError::ToolInputGuardrailTripwireTriggered {
        guardrail_name,
        output,
        run_data,
    } = &err
    else {
        panic!("unexpected error: {err:?}");
    };
    assert_eq!(guardrail_name, "allow_then_raise");
    assert_eq!(output.output_info, json!("second_turn"));
    let data = run_data.as_ref().expect("run data");
    let infos: Vec<&serde_json::Value> = data
        .tool_input_guardrail_results
        .iter()
        .map(|r| &r.output.output_info)
        .collect();
    assert_eq!(infos, vec![&json!("first_turn")]);
    assert!(data.tool_output_guardrail_results.is_empty());
}

#[tokio::test]
async fn tool_tripwire_preserves_completed_turn_results_non_streamed() {
    tool_tripwire_preserves_completed_turn_results(false).await;
}

#[tokio::test]
async fn tool_tripwire_preserves_completed_turn_results_streamed() {
    tool_tripwire_preserves_completed_turn_results(true).await;
}
