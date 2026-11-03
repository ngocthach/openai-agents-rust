//! Ported from the Python SDK's `tests/test_max_turns.py`.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, ItemHelpers, RunOptions, Runner, Session,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

/// Mirrors `tests/test_responses.py::get_function_tool`.
fn get_function_tool<C: Send + Sync + 'static>(name: &str, result: &str) -> FunctionTool<C> {
    let result = result.to_string();
    FunctionTool::new(name, "", move |_ctx, _a: serde_json::Value| {
        let result = result.clone();
        async move { Ok(result) }
    })
}

/// Mirrors `tests/test_responses.py::get_function_tool_call`.
fn get_function_tool_call(name: &str, arguments: &str, call_id: &str) -> OutputItemAlias {
    openai_agents_sdk::testing::function_call_full(
        name,
        arguments,
        call_id,
        Some("1".to_string()),
        None,
    )
}

type OutputItemAlias = openai_agents_sdk::items::OutputItem;

fn get_text_message(text: &str) -> OutputItemAlias {
    assistant_message(text)
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct FooModel {
    summary: String,
}

#[tokio::test]
async fn non_streamed_max_turns() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let func_output = json!({"a": "b"}).to_string();
    for i in 1..=5 {
        model
            .enqueue(ScriptedStep::output(vec![
                get_text_message(&i.to_string()),
                get_function_tool_call("some_function", &func_output, &i.to_string()),
            ]))
            .unwrap();
    }
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("some_function", "result"));

    let err = Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).max_turns(Some(3)),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::MaxTurnsExceeded { .. }),
        "expected MaxTurnsExceeded, got {err:?}"
    );
}

#[tokio::test]
async fn non_streamed_max_turns_none_disables_limit() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let func_output = json!({"a": "b"}).to_string();
    for i in 1..=4 {
        model
            .enqueue(ScriptedStep::output(vec![
                get_text_message(&i.to_string()),
                get_function_tool_call("some_function", &func_output, &i.to_string()),
            ]))
            .unwrap();
    }
    model
        .enqueue(ScriptedStep::output(vec![get_text_message("done")]))
        .unwrap();
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("some_function", "result"));

    let result =
        Runner::run_with_options(&agent, "user_message", RunOptions::new(()).max_turns(None))
            .await
            .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

#[tokio::test]
async fn streamed_max_turns() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let func_output = json!({"a": "b"}).to_string();
    for i in 1..=5 {
        model
            .enqueue(ScriptedStep::output(vec![
                get_text_message(&i.to_string()),
                get_function_tool_call("some_function", &func_output, &i.to_string()),
            ]))
            .unwrap();
    }
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("some_function", "result"));

    let mut streamed = Runner::run_streamed_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).max_turns(Some(3)),
    );
    let mut error = None;
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    let error = error.expect("stream must surface MaxTurnsExceeded");
    assert!(matches!(error, AgentsError::MaxTurnsExceeded { .. }));
}

#[tokio::test]
async fn streamed_max_turns_none_disables_limit() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let func_output = json!({"a": "b"}).to_string();
    for i in 1..=4 {
        model
            .enqueue(ScriptedStep::output(vec![
                get_text_message(&i.to_string()),
                get_function_tool_call("some_function", &func_output, &i.to_string()),
            ]))
            .unwrap();
    }
    model
        .enqueue(ScriptedStep::output(vec![get_text_message("done")]))
        .unwrap();
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("some_function", "result"));

    let mut streamed = Runner::run_streamed_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).max_turns(None),
    );
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    assert_eq!(streamed.final_output_text().as_deref(), Some("done"));
    assert_eq!(streamed.max_turns(), None);
}

#[tokio::test]
async fn non_streamed_max_turns_handler_returns_output() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let agent = Agent::<()>::new("test_1").with_model_instance(model.clone());
    let options = RunOptions::new(()).max_turns(Some(0)).error_handlers(
        RunErrorHandlers::new().on_max_turns(|input| async move {
            Ok(Some(RunErrorHandlerResult::new(format!(
                "summary:{}",
                input.run_data.history.len()
            ))))
        }),
    );
    let result = Runner::run_with_options(&agent, "user_message", options)
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("summary:1"));
    assert_eq!(text_message_outputs(&result.new_items), "summary:1");
}

#[tokio::test]
async fn non_streamed_max_turns_handler_skip_history() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let agent = Agent::<()>::new("test_1").with_model_instance(model.clone());
    let options = RunOptions::new(()).max_turns(Some(0)).error_handlers(
        RunErrorHandlers::new().on_max_turns(|_input| async move {
            Ok(Some(
                RunErrorHandlerResult::new("summary").without_history(),
            ))
        }),
    );
    let result = Runner::run_with_options(&agent, "user_message", options)
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("summary"));
    assert!(result.new_items.is_empty());
}

#[tokio::test]
async fn structured_output_max_turns_handler_json_output() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model.clone())
        .with_output_type::<FooModel>();
    let options = RunOptions::new(()).max_turns(Some(0)).error_handlers(
        RunErrorHandlers::new().on_max_turns(|_input| async move {
            Ok(Some(RunErrorHandlerResult::new(json!({"summary": "ok"}))))
        }),
    );
    let result = Runner::run_with_options(&agent, "user_message", options)
        .await
        .unwrap();
    assert_eq!(
        result.final_output_as::<FooModel>().unwrap(),
        FooModel {
            summary: "ok".into()
        }
    );
    assert_eq!(
        text_message_outputs(&result.new_items),
        r#"{"summary":"ok"}"#
    );
}

#[tokio::test]
async fn streamed_max_turns_handler_returns_output() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    let agent = Agent::<()>::new("test_1").with_model_instance(model.clone());
    let options = RunOptions::new(()).max_turns(Some(0)).error_handlers(
        RunErrorHandlers::new()
            .on_max_turns(|_input| async move { Ok(Some(RunErrorHandlerResult::new("summary"))) }),
    );
    let mut streamed = Runner::run_streamed_with_options(&agent, "user_message", options);
    let mut messages = Vec::new();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem { name, item, .. } = event.unwrap()
                && let RunItemKind::MessageOutput { raw_item } = &item.kind
            {
                messages.push((name, raw_item.text_content()));
            }
        }
    }
    assert_eq!(streamed.final_output_text().as_deref(), Some("summary"));
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].1, "summary");
    assert_eq!(
        messages[0].0,
        openai_agents_sdk::stream_events::RunItemStreamEventName::MessageOutputCreated
    );
}

#[tokio::test]
async fn non_streamed_max_turns_handler_persists_output_to_session() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![get_function_tool_call(
            "some_function",
            &json!({"a": "b"}).to_string(),
            "2",
        )]))
        .unwrap();
    let agent = Agent::<()>::new("test_1")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("some_function", "result"));
    let session = Arc::new(InMemorySession::new("max-turns-handler"));
    let options = RunOptions::new(())
        .max_turns(Some(1))
        .session(session.clone())
        .error_handlers(RunErrorHandlers::new().on_max_turns(|_input| async move {
            Ok(Some(RunErrorHandlerResult::new("fallback answer")))
        }));
    let result = Runner::run_with_options(&agent, "user_message", options)
        .await
        .unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("fallback answer")
    );
    assert_eq!(
        session_item_types(session.as_ref()).await,
        vec!["user", "function_call", "function_call_output", "message"]
    );
}

async fn session_item_types(session: &InMemorySession) -> Vec<String> {
    session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|item| {
            item.0
                .get("type")
                .or_else(|| item.0.get("role"))
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

fn text_message_outputs(items: &[openai_agents_sdk::items::RunItem]) -> String {
    ItemHelpers::text_message_outputs(items)
}
