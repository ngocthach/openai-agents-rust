//! Items and tool-output helpers left over from earlier rounds:
//! `test_items_helpers.py` (`tool_call_item_to_input_item_keeps_payload_api_safe`, the
//! `ToolCallItem.tool_name` / `call_id` and `ToolCallOutputItem.call_id` accessors).

use openai_agents_sdk::items::{RunItem, RunItemKind};
use serde_json::{Value, json};

fn call(raw: Value) -> RunItem {
    RunItem::new("test", RunItemKind::ToolCall { raw_item: raw })
}

fn output(raw: Value) -> RunItem {
    RunItem::new(
        "test",
        RunItemKind::ToolCallOutput {
            raw_item: raw,
            output: json!("ok"),
            custom_data: None,
        },
    )
}

#[test]
fn tool_call_item_to_input_item_keeps_payload_api_safe() {
    let item = call(json!({
        "id": "fc_1", "call_id": "call_1", "name": "my_tool", "arguments": "{}",
        "type": "function_call", "status": "completed",
    }))
    .with_tool_details(Some("A helpful tool".into()), Some("My Tool".into()));
    let input = item.to_input_item().0;
    assert_eq!(input["type"], "function_call");
    assert!(input.get("title").is_none());
    assert!(input.get("description").is_none());
}

#[test]
fn tool_call_item_tool_name() {
    let typed = call(json!({"id": "fc1", "call_id": "call_1", "name": "my_tool",
                            "arguments": "{}", "type": "function_call"}));
    assert_eq!(typed.tool_name().as_deref(), Some("my_tool"));
    let dict = call(
        json!({"type": "function_call", "name": "dict_tool", "call_id": "call_1",
                           "arguments": "{}"}),
    );
    assert_eq!(dict.tool_name().as_deref(), Some("dict_tool"));
    let nameless = call(json!({"id": "fs1", "queries": ["q"], "status": "completed",
                               "type": "file_search_call"}));
    assert_eq!(nameless.tool_name(), None);
}

#[test]
fn tool_call_item_call_id() {
    let typed = call(
        json!({"id": "fc1", "call_id": "call_abc", "name": "t", "arguments": "{}",
                            "type": "function_call"}),
    );
    assert_eq!(typed.call_id().as_deref(), Some("call_abc"));
    let falls_back_to_id = call(
        json!({"id": "fs_xyz", "queries": ["q"], "status": "completed",
                                       "type": "file_search_call"}),
    );
    assert_eq!(falls_back_to_id.call_id().as_deref(), Some("fs_xyz"));
    let dict = call(
        json!({"type": "function_call", "name": "t", "call_id": "call_dict_id",
                           "arguments": "{}"}),
    );
    assert_eq!(dict.call_id().as_deref(), Some("call_dict_id"));
}

#[test]
fn tool_call_output_item_call_id() {
    let with_id = output(
        json!({"type": "function_call_output", "call_id": "call_out_1",
                                "output": "ok"}),
    );
    assert_eq!(with_id.call_id().as_deref(), Some("call_out_1"));
    let missing = output(json!({"type": "function_call_output", "output": "ok"}));
    assert_eq!(missing.call_id(), None);
}

// ----------------------------------------------------------------------------------------
// `test_pretty_print.py` leftovers
// ----------------------------------------------------------------------------------------

mod pretty {
    use std::sync::Arc;

    use futures::StreamExt;
    use openai_agents_sdk::error::RunErrorDetails;
    use openai_agents_sdk::testing::{
        ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
    };
    use openai_agents_sdk::tool::FunctionTool;
    use openai_agents_sdk::{Agent, RunOptions, Runner};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    struct Foo {
        bar: String,
    }

    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}

    fn model_with(output: Vec<openai_agents_sdk::items::OutputItem>) -> Arc<ScriptedModel> {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let model = Arc::new(ScriptedModel::new());
        model.enqueue(ScriptedStep::output(output)).unwrap();
        model
    }

    /// `test_pretty_result_handles_none_final_output`: a result without a final output. Rust
    /// cannot construct a `RunResult` by hand, so the run pauses for an approval, which leaves
    /// `final_output` empty the same way.
    #[tokio::test]
    async fn pretty_result_handles_none_final_output() {
        let model = model_with(vec![function_call_with_id("gate", "{}", "call_1")]);
        let tool = FunctionTool::new("gate", "gate", |_c, _a: NoArgs| async { Ok("ok") })
            .with_needs_approval(true);
        let agent = Agent::<()>::new("none_agent")
            .with_tool(tool)
            .with_model_instance(model);
        let result = Runner::run(&agent, "Hello").await.unwrap();
        assert!(result.final_output.is_none());
        assert_eq!(
            result.pretty_print(),
            "RunResult:\n\
             - Last agent: Agent(name=\"none_agent\", ...)\n\
             - Final output (NoneType):\n\
             \x20   None\n\
             - 2 new item(s)\n\
             - 1 raw response(s)\n\
             - 0 input guardrail result(s)\n\
             - 0 output guardrail result(s)\n\
             (See `RunResult` for more details)"
        );
    }

    #[test]
    fn pretty_run_error_details() {
        let details = RunErrorDetails {
            last_agent: "error_agent".into(),
            ..Default::default()
        };
        let expected = "RunErrorDetails:\n\
             - Last agent: Agent(name=\"error_agent\", ...)\n\
             - 0 new item(s)\n\
             - 0 raw response(s)\n\
             - 0 input guardrail result(s)\n\
             - 0 output guardrail result(s)\n\
             - 0 tool input guardrail result(s)\n\
             - 0 tool output guardrail result(s)\n\
             (See `RunErrorDetails` for more details)";
        assert_eq!(details.pretty_print(), expected);
        assert_eq!(details.to_string(), expected);
    }

    #[tokio::test]
    async fn pretty_run_result_streaming_list_structured_output() {
        let payload = json!({"response": [{"bar": "Test"}, {"bar": "Test 2"}]}).to_string();
        let model = Arc::new(ScriptedModel::new());
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        model
            .enqueue(ScriptedStep::output(vec![
                assistant_message("Test"),
                assistant_message(&payload),
            ]))
            .unwrap();
        let agent = Agent::<()>::new("test_agent")
            .with_model_instance(model)
            .with_output_type::<Vec<Foo>>();
        let mut streamed = Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
        {
            let mut events = std::pin::pin!(streamed.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        let rendered = streamed.pretty_print();
        assert!(
            rendered.starts_with(
                "RunResultStreaming:\n- Current agent: Agent(name=\"test_agent\", ...)\n\
                 - Current turn: 1\n- Max turns: 10\n- Is complete: True\n\
                 - Final output (list):\n"
            ),
            "{rendered}"
        );
        assert!(rendered.contains("\"bar\": \"Test 2\""), "{rendered}");
        assert!(
            rendered.ends_with(
                "- 2 new item(s)\n- 1 raw response(s)\n- 0 input guardrail result(s)\n\
                 - 0 output guardrail result(s)\n(See `RunResultStreaming` for more details)"
            ),
            "{rendered}"
        );
    }

    /// `test_result_str_escapes_terminal_controls[model]`, non-streamed and streamed: JSON
    /// already escapes the C0 controls; the printer must also escape C1 and DEL.
    #[tokio::test]
    async fn result_str_escapes_terminal_controls_in_structured_output() {
        let payload = "Hello\u{1b}[2J\u{1b}]0;title\u{7}\u{8}\u{0}\u{7f}\u{9b}31m\u{9d}0;title\u{9c}\u{b}\u{c}\u{1c}\u{1d}\u{1e}\u{85}";
        let agent_name = "agent\u{1b}[2J\u{9b}31m";
        for streaming in [false, true] {
            let reply = serde_json::to_string(&Foo {
                bar: payload.into(),
            })
            .unwrap();
            let model = model_with(vec![assistant_message(&reply)]);
            let agent = Agent::<()>::new(agent_name)
                .with_model_instance(model)
                .with_output_type::<Foo>();
            let rendered = if streaming {
                let mut streamed =
                    Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
                {
                    let mut events = std::pin::pin!(streamed.stream_events());
                    while let Some(event) = events.next().await {
                        event.unwrap();
                    }
                }
                streamed.pretty_print()
            } else {
                let result = Runner::run(&agent, "Hello").await.unwrap();
                let foo: Foo = result.final_output_as().unwrap();
                assert_eq!(foo.bar, payload);
                result.pretty_print()
            };
            assert!(
                rendered.contains(r#"Agent(name="agent\x1b[2J\x9b31m", ...)"#),
                "{rendered}"
            );
            assert!(rendered.contains(r"\u001b[2J"), "{rendered}");
            assert!(rendered.contains(r"\x9b31m\x9d0;title\x9c"), "{rendered}");
            assert!(rendered.contains(r"\x85"), "{rendered}");
            assert_eq!(agent.name, agent_name);
            assert!(
                rendered.chars().all(|c| c >= ' ' || c == '\n' || c == '\t'),
                "{rendered}"
            );
            assert!(
                !rendered.chars().any(|c| ('\u{7f}'..='\u{9f}').contains(&c)),
                "{rendered}"
            );
        }
    }
}

// ----------------------------------------------------------------------------------------
// `test_tool_output_conversion.py`: untyped JSON outputs
// ----------------------------------------------------------------------------------------

mod tool_output {
    use openai_agents_sdk::items::{FunctionCall, ItemHelpers};
    use openai_agents_sdk::tool::{ToolOutput, ToolOutputPart};
    use serde_json::{Value, json};

    fn convert(output: ToolOutput) -> Value {
        let mut call = FunctionCall::new("dummy", "{}");
        call.id = Some("call-1".into());
        call.call_id = "call-1".into();
        let payload = ItemHelpers::tool_call_output_item(&call, &output);
        assert_eq!(payload["type"], "function_call_output");
        assert_eq!(payload["call_id"], "call-1");
        payload["output"].clone()
    }

    #[test]
    fn dict_variants_with_valid_shape_become_structured_parts() {
        let cases = [
            (
                json!({"type": "text", "text": "hey"}),
                json!({"type": "input_text", "text": "hey"}),
            ),
            (
                json!({"type": "image", "image_url": "http://example.com/img.png", "detail": "auto"}),
                json!({"type": "input_image", "image_url": "http://example.com/img.png",
                       "detail": "auto"}),
            ),
            (
                json!({"type": "image", "file_id": "file_123"}),
                json!({"type": "input_image", "file_id": "file_123"}),
            ),
            (
                json!({"type": "file", "file_data": "foobar", "filename": "report.pdf"}),
                json!({"type": "input_file", "file_data": "foobar", "filename": "report.pdf"}),
            ),
            (
                json!({"type": "file", "file_url": "https://example.com/report.pdf",
                       "filename": "report.pdf"}),
                json!({"type": "input_file", "file_url": "https://example.com/report.pdf",
                       "filename": "report.pdf"}),
            ),
            (
                json!({"type": "file", "file_id": "file_123", "filename": "report.pdf"}),
                json!({"type": "input_file", "file_id": "file_123", "filename": "report.pdf"}),
            ),
            // Extra fields are ignored.
            (
                json!({"type": "image", "image_url": "http://example.com/img.png", "foobar": 213}),
                json!({"type": "input_image", "image_url": "http://example.com/img.png"}),
            ),
        ];
        for (input, expected) in cases {
            let output = convert(ToolOutput::Json(input.clone()));
            assert_eq!(output, json!([expected]), "{input}");
        }
    }

    #[test]
    fn mixed_list_with_valid_dicts_is_converted() {
        let output = convert(ToolOutput::Json(json!([
            {"type": "text", "text": "hello"},
            {"type": "image", "image_url": "http://example.com/img.png"},
            {"type": "file", "file_id": "file_123"},
        ])));
        assert_eq!(
            output,
            json!([
                {"type": "input_text", "text": "hello"},
                {"type": "input_image", "image_url": "http://example.com/img.png"},
                {"type": "input_file", "file_id": "file_123"},
            ])
        );
    }

    /// The nine "not converted" dict/list cases and the empty list: the value is sent as its
    /// JSON text instead (Python sends its `repr`).
    #[test]
    fn values_without_a_valid_structured_shape_are_stringified() {
        let cases = [
            json!({"type": "text"}),
            json!({"type": "image"}),
            json!({"type": "file"}),
            json!({}),
            json!({"msg": "1234"}),
            json!({"type": "image", "location": "/path/to/img.png"}),
            json!({"type": "file", "path": "/path/to/file.txt"}),
            json!([{"msg": "foobar"}]),
            json!([{"type": "text", "text": "hello"}, {"msg": "foobar"}]),
            json!([]),
        ];
        for input in cases {
            let output = convert(ToolOutput::Json(input.clone()));
            assert_eq!(output, Value::String(input.to_string()), "{input}");
        }
    }

    #[test]
    fn typed_parts_still_convert() {
        let output = convert(ToolOutput::Parts(vec![ToolOutputPart::text("hello")]));
        assert_eq!(output, json!([{"type": "input_text", "text": "hello"}]));
    }
}
