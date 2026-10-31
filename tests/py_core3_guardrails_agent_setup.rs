//! Ported from the Python SDK's `tests/test_agent_clone_shallow_copy.py`,
//! `tests/test_agent_prompt.py` and `tests/test_agent_tool_input.py`: agent cloning, prompt
//! resolution (including the cancellation of sibling instructions when a prompt fails), and the
//! structured input an agent-as-tool receives.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use futures::future::BoxFuture;
use openai_agents_sdk::agent::{
    AgentAsToolOptions, StructuredToolInput, default_tool_input_builder,
};
use openai_agents_sdk::models::interface::Model;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, OpenAIClient, OpenAIResponsesModel, Prompt, RunConfig, RunContext, Runner,
};
use serde_json::{Value, json};
use tokio::sync::Notify;

/// What the structured-input builder saw: the summary and the full schema.
type SeenInput = (Option<String>, Option<Value>);

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn greet() -> FunctionTool<()> {
    FunctionTool::new("greet", "", |_c, _a: Value| async move {
        Ok("Hello".to_string())
    })
}

fn farewell() -> FunctionTool<()> {
    FunctionTool::new("farewell", "", |_c, _a: Value| async move {
        Ok("Goodbye".to_string())
    })
}

// --- clone -----------------------------------------------------------------------------------

#[test]
fn agent_clone_copies_the_scalar_fields_and_the_tool_list() {
    let original = Agent::<()>::new("Original")
        .with_instructions("Testing clone shallow copy")
        .with_tool(greet());
    let cloned = original.clone_with(|a| a.name = "Cloned".into());

    assert_eq!(cloned.name, "Cloned");
    assert_eq!(original.name, "Original");
    assert_eq!(
        cloned.instructions.is_some(),
        original.instructions.is_some()
    );
    let cloned_names: Vec<String> = cloned.tools.iter().filter_map(function_tool).collect();
    assert_eq!(cloned_names, vec!["greet"]);
}

#[test]
fn agent_clone_keeps_the_list_attributes_it_is_not_given() {
    let original = Agent::<()>::new("Original").with_tool(greet());

    let cloned = original.clone_with(|a| a.name = "Cloned".into());

    assert_eq!(cloned.tools.len(), original.tools.len());
    assert_eq!(cloned.tools.len(), 1);
}

#[test]
fn agent_clone_uses_a_given_list_as_is() {
    let original = Agent::<()>::new("Original").with_tool(greet());

    let cloned = original.clone_with(|a| a.tools = vec![farewell().into()]);

    let cloned_names: Vec<String> = cloned.tools.iter().filter_map(function_tool).collect();
    assert_eq!(cloned_names, vec!["farewell"]);
    let original_names: Vec<String> = original.tools.iter().filter_map(function_tool).collect();
    assert_eq!(original_names, vec!["greet"]);
}

/// Returns the name of a function tool, skipping hosted tools.
fn function_tool(tool: &openai_agents_sdk::Tool<()>) -> Option<String> {
    match tool {
        openai_agents_sdk::Tool::Function(f) => Some(f.name.clone()),
        _ => None,
    }
}

// Python's `test_agent_clone_still_shares_when_given_the_original_list` and
// `test_agent_clone_shared_list_mutation_affects_both_agents` assert that two agents hold one
// Python list object. Rust agents own their `Vec`s, so the two copies cannot alias; that is
// Python-only.

// --- prompt ----------------------------------------------------------------------------------

#[tokio::test]
async fn static_prompt_is_resolved_correctly() {
    let agent = Arc::new(
        Agent::<()>::new("test").with_prompt(Prompt {
            id: "my_prompt".into(),
            version: Some("1".into()),
            variables: Some(
                [(
                    "some_var".to_string(),
                    openai_agents_sdk::PromptVariable::from("some_value"),
                )]
                .into_iter()
                .collect(),
            ),
        }),
    );

    let resolved = agent
        .get_prompt(&RunContext::new(()))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(resolved.id, "my_prompt");
    assert_eq!(resolved.version.as_deref(), Some("1"));
    assert_eq!(resolved.variables.as_ref().map(|v| v.len()), Some(1));
}

#[tokio::test]
async fn dynamic_prompt_is_resolved_correctly() {
    let agent = Arc::new(Agent::<()>::new("test").with_dynamic_prompt(
        |_ctx: RunContext<()>, _agent: Arc<Agent<()>>| async move {
            Ok(Some(Prompt {
                id: "dyn_prompt".into(),
                version: Some("2".into()),
                variables: None,
            }))
        },
    ));

    let resolved = agent
        .get_prompt(&RunContext::new(()))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(resolved.id, "dyn_prompt");
    assert_eq!(resolved.version.as_deref(), Some("2"));
    assert!(resolved.variables.is_none());
}

#[tokio::test]
async fn prompt_is_passed_to_model() {
    disable_tracing();
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_prompt(Prompt {
            id: "model_prompt".into(),
            version: None,
            variables: None,
        });

    Runner::run(&agent, "hello").await.unwrap();

    let calls = model.calls();
    let prompt = calls[0]
        .request
        .prompt
        .as_ref()
        .expect("prompt sent to the model");
    assert_eq!(prompt.id, "model_prompt");
    assert!(prompt.version.is_none());
    assert!(prompt.variables.is_none());
}

/// Serves one canned Responses API reply and records every request body.
async fn responses_mock(replies: Vec<Value>) -> (String, Arc<Mutex<Vec<(String, Value)>>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let (path, body) = loop {
                let n = sock.read(&mut tmp).await.unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let headers = &text[..idx];
                    let len: usize = headers
                        .lines()
                        .find_map(|l| {
                            let l = l.to_lowercase();
                            l.strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        let path = headers.split_whitespace().nth(1).unwrap().to_string();
                        let body: Value = serde_json::from_slice(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                        break (path, body);
                    }
                }
            };
            cap.lock().unwrap().push((path, body));
            let reply = replies.next().expect("unexpected request").to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                reply.len(),
                reply
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}/v1"), captured)
}

#[tokio::test]
async fn agent_prompt_with_default_model_omits_model_and_tools_parameters() {
    disable_tracing();
    let (base, captured) = responses_mock(vec![json!({
        "id": "resp_1", "object": "response", "status": "completed",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    })])
    .await;
    let client = OpenAIClient::new("sk-test").with_base_url(base);
    let model: Arc<dyn Model> =
        Arc::new(OpenAIResponsesModel::new("gpt-4.1", client).with_explicit_model(false));
    let agent = Agent::<()>::new("prompt-agent")
        .with_model_instance(model)
        .with_prompt(Prompt {
            id: "pmpt_agent".into(),
            version: None,
            variables: None,
        });

    Runner::run_with_options(
        &agent,
        "hi",
        openai_agents_sdk::RunOptions::new(()).run_config(RunConfig::new()),
    )
    .await
    .unwrap();

    let bodies = captured.lock().unwrap();
    let body = &bodies[0].1;
    assert_eq!(body["prompt"]["id"], "pmpt_agent");
    assert!(
        body.get("model").is_none(),
        "the prompt's own model must win: {body}"
    );
    assert!(
        body.get("tools").is_none(),
        "no tools were configured: {body}"
    );
}

/// Blocks in instruction resolution until cancelled, recording the cancellation and the end of
/// its cleanup, as the Python test's `CancelledError` handler does.
fn slow_instructions(
    started: Arc<Notify>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    finished: Arc<std::sync::atomic::AtomicBool>,
) -> impl Fn(RunContext<()>, Arc<Agent<()>>) -> BoxFuture<'static, openai_agents_sdk::Result<String>>
+ Send
+ Sync
+ 'static {
    move |_ctx, _agent| {
        let started = started.clone();
        let cancelled = cancelled.clone();
        let finished = finished.clone();
        Box::pin(async move {
            struct Flags(
                Arc<std::sync::atomic::AtomicBool>,
                Arc<std::sync::atomic::AtomicBool>,
            );
            impl Drop for Flags {
                fn drop(&mut self) {
                    self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                    self.1.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
            let _flags = Flags(cancelled, finished);
            started.notify_one();
            std::future::pending::<()>().await;
            Ok("unreachable".to_string())
        })
    }
}

/// A prompt resolver that fails once the sibling instructions have started.
fn failing_prompt(
    started: Arc<Notify>,
) -> impl Fn(
    RunContext<()>,
    Arc<Agent<()>>,
) -> BoxFuture<'static, openai_agents_sdk::Result<Option<Prompt>>>
+ Send
+ Sync
+ 'static {
    move |_ctx, _agent| {
        let started = started.clone();
        Box::pin(async move {
            started.notified().await;
            Err(AgentsError::user("prompt resolution failed"))
        })
    }
}

#[tokio::test]
async fn run_cancels_sibling_instructions_when_prompt_resolution_fails() {
    disable_tracing();
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let agent = Agent::<()>::new("prompt-agent")
        .with_model_instance(Arc::new(ScriptedModel::new()))
        .with_dynamic_instructions(slow_instructions(
            started.clone(),
            cancelled.clone(),
            finished.clone(),
        ))
        .with_dynamic_prompt(failing_prompt(started));

    let err = tokio::time::timeout(Duration::from_secs(5), Runner::run(&agent, "hi"))
        .await
        .expect("run finished")
        .unwrap_err();

    assert!(
        err.to_string().contains("prompt resolution failed"),
        "{err}"
    );
    assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
    assert!(finished.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test]
async fn run_streamed_cancels_sibling_instructions_when_prompt_resolution_fails() {
    disable_tracing();
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let agent = Agent::<()>::new("prompt-agent")
        .with_model_instance(Arc::new(ScriptedModel::new()))
        .with_dynamic_instructions(slow_instructions(
            started.clone(),
            cancelled.clone(),
            finished.clone(),
        ))
        .with_dynamic_prompt(failing_prompt(started));

    let mut result =
        Runner::run_streamed_with_options(&agent, "hi", openai_agents_sdk::RunOptions::new(()));
    let error = tokio::time::timeout(Duration::from_secs(5), async {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                return Some(e);
            }
        }
        None
    })
    .await
    .expect("stream finished")
    .expect("the run must fail");

    assert!(
        error.to_string().contains("prompt resolution failed"),
        "{error}"
    );
    assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
    assert!(finished.load(std::sync::atomic::Ordering::SeqCst));
}

// --- agent-as-tool structured input ------------------------------------------------------------

/// Runs a parent whose model calls `tool` with `arguments`, and returns the text the nested
/// agent received as its user input.
async fn nested_input_for(arguments: &str) -> Result<String, AgentsError> {
    disable_tracing();
    let worker_model = Arc::new(ScriptedModel::new());
    worker_model
        .enqueue(ScriptedStep::output(vec![assistant_message("ok")]))
        .unwrap();
    let parent_model = Arc::new(ScriptedModel::new());
    parent_model
        .enqueue(ScriptedStep::output(vec![function_call(
            "worker_tool",
            arguments,
        )]))
        .unwrap();
    parent_model
        .enqueue(ScriptedStep::output(vec![assistant_message("parent done")]))
        .unwrap();
    let worker = Arc::new(Agent::<()>::new("worker").with_model_instance(worker_model.clone()));
    let nested = worker.as_tool_with(AgentAsToolOptions {
        tool_name: Some("worker_tool".into()),
        tool_description: Some("Call the worker".into()),
        ..Default::default()
    });
    let parent = Agent::<()>::new("parent")
        .with_model_instance(parent_model)
        .with_tool(nested);
    Runner::run(&parent, "go").await?;
    let calls = worker_model.calls();
    let input = &calls[0].request.input;
    let text = match &input.last().expect("nested input").0["content"] {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    Ok(text)
}

#[tokio::test]
async fn agent_as_tool_input_string_is_passed_through() {
    let text = nested_input_for(r#"{"input": "hello"}"#).await.unwrap();
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn agent_as_tool_input_without_an_input_field_is_passed_as_json() {
    let text = nested_input_for(r#"{"foo": "bar"}"#).await.unwrap();
    assert_eq!(text, r#"{"foo": "bar"}"#);
}

#[tokio::test]
async fn agent_as_tool_input_with_extra_fields_keeps_every_field() {
    let text = nested_input_for(r#"{"input": "hello", "target": "world"}"#)
        .await
        .unwrap();
    assert_eq!(text, r#"{"input": "hello", "target": "world"}"#);
}

/// Runs an agent-as-tool whose structured input is rendered by `input_builder`, returning what
/// the builder saw (`summary`, `json_schema`) and the text the nested agent received.
async fn structured_builder_view(
    parameters: Option<Value>,
    include_input_schema: bool,
    arguments: &str,
) -> (Option<String>, Option<Value>, String) {
    let seen: Arc<Mutex<Option<SeenInput>>> = Arc::new(Mutex::new(None));
    let recorder = seen.clone();
    let builder = move |options: &StructuredToolInput<'_>| {
        *recorder.lock().unwrap() = Some((
            options.summary.map(str::to_string),
            options.json_schema.cloned(),
        ));
        "built input".to_string()
    };
    let worker_model = Arc::new(ScriptedModel::new());
    worker_model
        .enqueue(ScriptedStep::output(vec![assistant_message("ok")]))
        .unwrap();
    let parent_model = Arc::new(ScriptedModel::new());
    parent_model
        .enqueue(ScriptedStep::output(vec![function_call(
            "worker_tool",
            arguments,
        )]))
        .unwrap();
    parent_model
        .enqueue(ScriptedStep::output(vec![assistant_message("parent done")]))
        .unwrap();
    let worker = Arc::new(Agent::<()>::new("worker").with_model_instance(worker_model.clone()));
    let nested = worker.as_tool_with(AgentAsToolOptions {
        tool_name: Some("worker_tool".into()),
        tool_description: Some("Call the worker".into()),
        parameters,
        input_builder: Some(Arc::new(builder)),
        include_input_schema,
        ..Default::default()
    });
    let parent = Agent::<()>::new("parent")
        .with_model_instance(parent_model)
        .with_tool(nested);
    Runner::run(&parent, "go").await.unwrap();
    let (summary, schema) = seen.lock().unwrap().clone().expect("builder ran");
    let nested_text = match &worker_model.calls()[0].request.input.last().unwrap().0["content"] {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    (summary, schema, nested_text)
}

#[tokio::test]
async fn agent_as_tool_custom_input_builder_replaces_the_input() {
    let (_summary, _schema, nested_text) =
        structured_builder_view(None, false, r#"{"foo": "bar"}"#).await;
    assert_eq!(nested_text, "built input");
}

#[tokio::test]
async fn agent_as_tool_input_builder_without_descriptions_sees_no_summary() {
    // The default parameters carry no descriptions, so there is no summary, and the schema is
    // only passed on when `include_input_schema` is set.
    let (summary, schema, _) = structured_builder_view(None, false, r#"{"foo": "bar"}"#).await;
    assert!(summary.is_none());
    assert!(schema.is_none());
}

#[tokio::test]
async fn agent_as_tool_input_builder_sees_summary_for_simple_fields() {
    let parameters = json!({
        "type": "object",
        "description": "Tool arguments.",
        "properties": {
            "mode": {"enum": ["fast", "safe"], "description": "Execution mode."},
            "status": {"const": "ok", "description": "Status marker."},
            "count": {"type": ["integer", "null"], "description": "Optional count."},
            "enabled": {"type": "boolean", "description": "Feature toggle."},
        },
        "required": ["mode", "status"],
    });
    let (summary, schema, _) = structured_builder_view(
        Some(parameters.clone()),
        true,
        r#"{"mode": "fast", "status": "ok"}"#,
    )
    .await;

    let summary = summary.expect("summary");
    assert!(
        summary.contains("Description: Tool arguments."),
        "{summary}"
    );
    assert!(
        summary.contains(r#"- mode (enum("fast" | "safe"), required) - Execution mode."#),
        "{summary}"
    );
    assert!(
        summary.contains(r#"- status (literal("ok"), required) - Status marker."#),
        "{summary}"
    );
    assert!(
        summary.contains("- count (integer | null, optional) - Optional count."),
        "{summary}"
    );
    assert!(
        summary.contains("- enabled (boolean, optional) - Feature toggle."),
        "{summary}"
    );
    assert_eq!(schema, Some(parameters));
}

#[tokio::test]
async fn agent_as_tool_input_builder_sees_no_summary_for_unsupported_shapes() {
    for parameters in [
        json!({"type": "array"}),
        json!({"type": "object", "properties": []}),
        json!({"type": "object", "properties": {"nested": {"type": "object", "properties": {"x": {"type": "string"}}}}}),
    ] {
        let (summary, _, _) =
            structured_builder_view(Some(parameters), false, r#"{"x": "1"}"#).await;
        assert!(summary.is_none());
    }
}

#[tokio::test]
async fn agent_as_tool_input_builder_summarizes_edge_case_fields() {
    let parameters = json!({
        "type": "object",
        "properties": {
            "many": {"enum": [1, 2, 3, 4, 5, 6], "description": "Many values."},
            "empty": {"enum": [], "description": "No values."},
        },
    });
    let (summary, _, _) = structured_builder_view(Some(parameters), false, r#"{"many": 1}"#).await;
    let summary = summary.expect("summary");
    assert!(summary.contains("..."), "{summary}");
    assert!(
        summary.contains(r#"- empty (enum, optional) - No values."#),
        "{summary}"
    );
}

#[test]
fn default_tool_input_builder_includes_the_summary_when_there_is_no_schema() {
    let params = json!({"foo": "bar"});
    let built = default_tool_input_builder(&StructuredToolInput {
        params: &params,
        summary: Some("Summary"),
        json_schema: None,
    });
    assert!(built.contains("Input Schema Summary:"), "{built}");
    assert!(built.contains("Summary"), "{built}");
}

#[tokio::test]
async fn agent_as_tool_input_builder_drops_the_summary_for_undescribable_fields() {
    for field in [
        json!({"type": ["integer", "string"], "description": "Two scalar types."}),
        json!({"type": "array", "description": "A list."}),
        json!({"description": "No type or enum."}),
        json!({"type": "unknown", "description": "Unsupported type."}),
    ] {
        let parameters = json!({
            "type": "object",
            "description": "Arguments.",
            "properties": {"field": field},
        });
        let (summary, _, _) =
            structured_builder_view(Some(parameters), false, r#"{"field": 1}"#).await;
        assert!(
            summary.is_none(),
            "a field that cannot be described hides the summary"
        );
    }
}
