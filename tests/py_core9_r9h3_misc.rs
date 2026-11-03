//! Portable cases the earlier python-only buckets hid, for several small Python test files:
//! programmatic output schemas (`test_programmatic_tool_calling.py`), run error handler output
//! rendering through a custom output schema (`test_run_internal_error_handlers.py`), and the hosted
//! MCP connector example's request and output (`test_hosted_mcp_connectors_example.py`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent_output::AgentOutputSchemaBase;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool};
use openai_agents_sdk::{
    Agent, AgentsError, OpenAIClient, OpenAIResponsesModel, Result, RunResult, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------- programmatic output schemas

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct LooseOutput {
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[derive(Deserialize, JsonSchema)]
#[schemars(description = "Inventory result")]
#[allow(dead_code)]
struct InventoryOutput {
    sku: String,
    available_units: u32,
}

fn inventory_tool() -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args("lookup_inventory", "", |_ctx| async {
        Ok(json!({"sku": "A-1", "available_units": 42}).to_string())
    })
    .with_allowed_callers(vec!["programmatic".to_string()])
}

/// Python: `test_function_tool_rejects_loose_programmatic_output_annotation`
/// (`-> dict[str, Any]`). An output type with free-form keys is not a strict object schema.
#[test]
fn a_free_form_output_type_is_rejected() {
    let err = inventory_tool()
        .with_output_type::<HashMap<String, Value>>()
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::UserError { .. })
            && err.to_string().contains("strict JSON object"),
        "{err}"
    );
    let err = inventory_tool()
        .with_output_type::<LooseOutput>()
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AgentsError::UserError { .. }), "{err}");
}

/// Python: `test_function_tool_preserves_annotated_structured_return_metadata` (a `Field`
/// description on the return annotation reaches the output schema).
#[test]
fn output_type_metadata_reaches_the_output_schema() {
    let tool = inventory_tool()
        .with_output_type::<InventoryOutput>()
        .unwrap();
    let schema = tool.output_json_schema.expect("an output schema");
    assert_eq!(schema["description"], "Inventory result");
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
}

/// Python: `test_function_tool_treats_annotated_plain_returns_as_untyped` and
/// `test_function_tool_does_not_infer_non_programmatic_output`. Without a declared output type a
/// tool has no output schema, whatever it returns; a plain-text output type is refused.
#[test]
fn an_untyped_tool_has_no_output_schema() {
    let tool = inventory_tool();
    assert!(tool.output_json_schema.is_none());
    let plain = FunctionTool::<()>::new_no_args("plain", "", |_ctx| async { Ok("ok") });
    assert!(plain.output_json_schema.is_none());
    let err = plain.with_output_type::<String>().map(|_| ()).unwrap_err();
    assert!(matches!(err, AgentsError::UserError { .. }), "{err}");
}

// ------------------------------------------------- error handler output with a custom schema

/// A structured output schema that is not one of the SDK's own: any JSON object passes, and the
/// payload can be told to fail validation.
#[derive(Debug)]
struct CustomSchema {
    reject: bool,
}

impl AgentOutputSchemaBase for CustomSchema {
    fn is_plain_text(&self) -> bool {
        false
    }
    fn name(&self) -> String {
        "CustomSchema".into()
    }
    fn json_schema(&self) -> Result<Value> {
        Ok(json!({"type": "object"}))
    }
    fn is_strict_json_schema(&self) -> bool {
        true
    }
    fn validate_json(&self, json_str: &str) -> Result<Value> {
        if self.reject {
            return Err(AgentsError::model_behavior("rejected by the custom schema"));
        }
        serde_json::from_str(json_str).map_err(|e| AgentsError::model_behavior(e.to_string()))
    }
}

fn looping_agent(schema: CustomSchema) -> Agent<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id("ping", "{}", "call-1")]);
    model.add_turn(vec![function_call_with_id("ping", "{}", "call-2")]);
    let ping =
        FunctionTool::<()>::new_no_args("ping", "Ping", |_c| async { Ok("pong".to_string()) });
    Agent::<()>::new("custom-format")
        .with_model_instance(model)
        .with_tools(vec![ping])
        .with_output_schema(Arc::new(schema))
}

fn options_with(handlers: RunErrorHandlers<()>) -> RunOptions<()> {
    let mut options = RunOptions::new(()).run_config(RunConfig::new());
    options.max_turns = Some(1);
    options.error_handlers = handlers;
    options
}

fn handler_returning(output: Value) -> RunErrorHandlers<()> {
    RunErrorHandlers::<()>::default().on_max_turns(move |_input| {
        let output = output.clone();
        async move {
            Ok(Some(RunErrorHandlerResult {
                final_output: output,
                include_in_history: true,
            }))
        }
    })
}

/// Python: `test_format_final_output_text_uses_custom_schema_and_fallback` (the custom schema
/// half; `str(object())` has no Rust counterpart because handler output is always JSON). A
/// handler's output is validated by, and rendered as JSON for, an output schema of any kind.
#[tokio::test]
async fn a_handler_output_is_rendered_through_a_custom_schema() {
    let agent = looping_agent(CustomSchema { reject: false });
    let result = Runner::run_with_options(
        &agent,
        "start",
        options_with(handler_returning(json!({"ok": true}))),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(json!({"ok": true})));
    let message = last_message_text(&result);
    assert_eq!(
        serde_json::from_str::<Value>(&message).unwrap(),
        json!({"ok": true})
    );
}

/// Python: `test_validate_handler_final_output_raises_for_unserializable_data` (the invalid
/// payload half: a set cannot be a Rust `Value`). A handler output the schema refuses is a user
/// error, not a model error.
#[tokio::test]
async fn a_handler_output_the_schema_refuses_is_a_user_error() {
    let agent = looping_agent(CustomSchema { reject: true });
    let err = Runner::run_with_options(
        &agent,
        "start",
        options_with(handler_returning(json!({"bad": [1, 2]}))),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, AgentsError::UserError { .. })
            && err
                .to_string()
                .contains("Invalid run error handler final_output"),
        "{err:?}"
    );
}

fn last_message_text(result: &RunResult<()>) -> String {
    use openai_agents_sdk::items::RunItemKind;
    result
        .new_items
        .iter()
        .rev()
        .find_map(|item| match &item.kind {
            RunItemKind::MessageOutput { raw_item } => Some(raw_item.to_value().to_string()),
            _ => None,
        })
        .map(|raw| {
            let raw: Value = serde_json::from_str(&raw).unwrap();
            raw["content"][0]["text"].as_str().unwrap().to_string()
        })
        .expect("a message output")
}

// ----------------------------------------------------------- hosted MCP connector example

#[derive(Clone)]
enum Reply {
    Json(Value),
    Sse(Vec<Value>),
}

async fn mock_server(reply: Reply) -> (String, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap = captured.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let body = loop {
                let n = sock.read(&mut tmp).await.unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let len: usize = text[..idx]
                        .lines()
                        .find_map(|l| {
                            l.to_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break serde_json::from_slice::<Value>(&buf[idx + 4..idx + 4 + len])
                            .unwrap_or(Value::Null);
                    }
                }
            };
            cap.lock().unwrap().push(body);
            let resp = match &reply {
                Reply::Json(v) => {
                    let b = v.to_string();
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}",
                        b.len()
                    )
                }
                Reply::Sse(events) => {
                    let mut b = String::new();
                    for e in events {
                        b.push_str(&format!(
                            "event: {}\ndata: {}\n\n",
                            e["type"].as_str().unwrap_or("message"),
                            e
                        ));
                    }
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}",
                        b.len()
                    )
                }
            };
            sock.write_all(resp.as_bytes()).await.unwrap();
            sock.shutdown().await.ok();
        }
    });
    (format!("http://{addr}/v1"), captured)
}

/// Python: `test_connector_example_output` (verbose x streaming). The example runs an agent with a
/// connector-backed hosted MCP tool through the real runner and request conversion: the request
/// carries the connector configuration unchanged, the answer is the model's text, and the items
/// are the listed tools and the message.
#[tokio::test]
async fn a_hosted_mcp_connector_tool_reaches_the_request_and_the_answer_comes_back() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let authorization = "synthetic-calendar-authorization";
    let answer = "No events scheduled today.";
    for stream in [false, true] {
        let response = json!({
            "id": "resp_1", "object": "response", "status": "completed",
            "output": [
                {"id": "mcp_1", "type": "mcp_list_tools", "server_label": "google_calendar", "tools": []},
                {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": answer, "annotations": []}]},
            ],
            "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        });
        let reply = if stream {
            Reply::Sse(vec![
                json!({"type": "response.output_text.delta", "sequence_number": 0, "item_id": "1",
                       "output_index": 1, "content_index": 0, "delta": answer, "logprobs": []}),
                json!({"type": "response.completed", "sequence_number": 1, "response": response}),
            ])
        } else {
            Reply::Json(response)
        };
        let (base, captured) = mock_server(reply).await;
        let client = OpenAIClient::new("synthetic-openai-key").with_base_url(base);
        let agent = Agent::<()>::new("Assistant")
            .with_instructions(
                "You are a helpful assistant that can help a user with their calendar.",
            )
            .with_tool(HostedMCPTool::<()>::new(json!({
                "type": "mcp",
                "server_label": "google_calendar",
                "connector_id": "connector_googlecalendar",
                "authorization": authorization,
                "require_approval": "never",
            })))
            .with_model_instance(Arc::new(OpenAIResponsesModel::new("gpt-4.1", client)));

        let (final_text, item_types) = if stream {
            let mut result = Runner::run_streamed(&agent, "What is my schedule for today?");
            let mut text = String::new();
            {
                let mut events = std::pin::pin!(result.stream_events());
                while let Some(event) = events.next().await {
                    if let Some(delta) = event.unwrap().text_delta() {
                        text.push_str(delta);
                    }
                }
            }
            assert_eq!(result.final_output_text().as_deref(), Some(answer));
            let items = result
                .into_result()
                .expect("streamed result")
                .new_items
                .iter()
                .map(|i| i.item_type())
                .collect::<Vec<_>>();
            (text, items)
        } else {
            let result = Runner::run(&agent, "What is my schedule for today?")
                .await
                .unwrap();
            let items = result.new_items.iter().map(|i| i.item_type()).collect();
            (result.final_output_text().unwrap(), items)
        };

        assert_eq!(final_text, answer, "stream={stream}");
        assert_eq!(
            item_types,
            ["mcp_list_tools_item", "message_output_item"],
            "stream={stream}"
        );
        let requests = captured.lock().unwrap().clone();
        assert_eq!(requests.len(), 1, "stream={stream}");
        assert_eq!(
            requests[0]["stream"].as_bool().unwrap_or(false),
            stream,
            "stream={stream}"
        );
        assert_eq!(
            requests[0]["tools"],
            json!([{
                "type": "mcp",
                "server_label": "google_calendar",
                "connector_id": "connector_googlecalendar",
                "authorization": authorization,
                "require_approval": "never",
            }]),
            "stream={stream}"
        );
    }
}

// ------------------------------------------------------------------ deferred top-level tools

fn deferred_call(call_id: &str) -> openai_agents_sdk::items::OutputItem {
    openai_agents_sdk::testing::function_call_full(
        "get_weather",
        "{}",
        call_id,
        None,
        Some("get_weather".into()),
    )
}

/// Python: `test_deferred_top_level_per_call_approval_keeps_bare_name_lookup` and
/// `test_deferred_top_level_permanent_approval_does_not_alias_to_bare_name`, through the runner
/// with a real deferred tool (called under a namespace equal to its own name). A per-call
/// approval answers a bare-name query for that call; a permanent one authorizes later calls of
/// the tool but does not answer a bare-name query for an unrelated call.
#[tokio::test]
async fn deferred_top_level_decisions_follow_the_per_call_and_permanent_rules() {
    use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep};
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for always in [false, true] {
        let executed = Arc::new(Mutex::new(Vec::<String>::new()));
        let recorded = executed.clone();
        let tool = FunctionTool::<()>::new_no_args("get_weather", "weather", move |ctx| {
            let recorded = recorded.clone();
            async move {
                recorded.lock().unwrap().push(ctx.tool_call_id.clone());
                Ok("sunny".to_string())
            }
        })
        .with_needs_approval(true)
        .with_defer_loading(true);
        let model = Arc::new(ScriptedModel::new());
        model
            .enqueue(ScriptedStep::output(vec![deferred_call("call-weather")]))
            .unwrap();
        model
            .enqueue(ScriptedStep::output(vec![deferred_call("call-weather-2")]))
            .unwrap();
        model
            .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
            .unwrap();
        let agent = Agent::<()>::new("test-agent")
            .with_tool(tool)
            .with_model_instance(model);

        let first = Runner::run(&agent, "go").await.unwrap();
        assert_eq!(first.interruptions.len(), 1, "always={always}");
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], always);
        let second = Runner::run(&agent, state).await.unwrap();

        let ctx = &second.context_wrapper;
        if !always {
            assert_eq!(
                ctx.is_tool_approved("test-agent", "get_weather", "call-weather"),
                Some(true),
                "the bare-name query finds the per-call decision"
            );
        }
        if always {
            // The permanent decision authorized the second call without asking again.
            assert!(
                second.interruptions.is_empty(),
                "{:?}",
                second.interruptions
            );
            assert_eq!(
                *executed.lock().unwrap(),
                ["call-weather", "call-weather-2"]
            );
            assert_eq!(
                ctx.is_tool_approved("test-agent", "get_weather", "call-never-made"),
                None
            );
        } else {
            // A per-call decision does not extend to the next call.
            assert_eq!(second.interruptions.len(), 1);
            assert_eq!(*executed.lock().unwrap(), ["call-weather"]);
        }
    }
}

/// Python: `test_deferred_top_level_approval_does_not_alias_to_visible_bare_sibling`, through the
/// runner. A deferred top-level tool and a visible tool of the same bare name are two tools: a
/// permanent approval of the deferred one must not authorize the visible one.
#[tokio::test]
async fn a_permanent_deferred_approval_does_not_authorize_the_visible_tool_of_the_same_name() {
    use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, function_call_full};
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let executed = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let recorder = |label: &'static str| {
        let executed = executed.clone();
        FunctionTool::<()>::new_no_args("get_weather", "weather", move |_ctx| {
            let executed = executed.clone();
            async move {
                executed.lock().unwrap().push(label);
                Ok(label.to_string())
            }
        })
        .with_needs_approval(true)
    };
    let deferred = recorder("deferred").with_defer_loading(true);
    let visible = recorder("visible");
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![function_call_full(
            "get_weather",
            "{}",
            "call-weather",
            None,
            Some("get_weather".into()),
        )]))
        .unwrap();
    model
        .enqueue(ScriptedStep::output(vec![function_call_full(
            "get_weather",
            "{}",
            "call-bare",
            None,
            None,
        )]))
        .unwrap();
    model
        .enqueue(ScriptedStep::output(vec![assistant_message("done")]))
        .unwrap();
    let agent = Agent::<()>::new("test-agent")
        .with_tools(vec![deferred, visible])
        .with_model_instance(model);

    let first = Runner::run(&agent, "go").await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], true);
    let second = Runner::run(&agent, state).await.unwrap();

    assert_eq!(*executed.lock().unwrap(), ["deferred"]);
    assert_eq!(
        second.interruptions.len(),
        1,
        "the visible tool still asks: {:?}",
        second.interruptions
    );
    assert_eq!(second.interruptions[0].call_id(), "call-bare");
}
