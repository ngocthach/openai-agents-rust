//! Ported from the Python SDK's `tests/test_tool_output_conversion.py` and the portable part of
//! `tests/test_result_cast.py`.
//!
//! Python's `ItemHelpers.tool_call_output_item` has to guess whether a `dict` the user returned is
//! a structured content part; Rust's `ToolOutput` is a typed enum, so the "untyped dict" half of
//! the Python file has no Rust counterpart and is recorded as PYTHON-ONLY in the parity report.

use std::sync::Arc;

use openai_agents_sdk::items::{FunctionCall, ItemHelpers};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{ToolOutput, ToolOutputPart};
use openai_agents_sdk::{Agent, Runner};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

fn make_tool_call() -> FunctionCall {
    let mut call = FunctionCall::new("dummy", "{}");
    call.id = Some("call-1".into());
    call.call_id = "call-1".into();
    call
}

#[test]
fn tool_call_output_item_text_part() {
    let call = make_tool_call();
    let payload = ItemHelpers::tool_call_output_item(
        &call,
        &ToolOutput::Parts(vec![ToolOutputPart::text("hello")]),
    );
    assert_eq!(payload["type"], "function_call_output");
    assert_eq!(payload["call_id"], "call-1");
    assert_eq!(payload["output"].as_array().unwrap().len(), 1);
    assert_eq!(payload["output"][0]["type"], "input_text");
    assert_eq!(payload["output"][0]["text"], "hello");
}

#[test]
fn tool_call_output_item_image_part() {
    let call = make_tool_call();
    let payload = ItemHelpers::tool_call_output_item(
        &call,
        &ToolOutput::Parts(vec![ToolOutputPart::image_url(
            "data:image/png;base64,AAAA",
        )]),
    );
    assert_eq!(payload["output"][0]["type"], "input_image");
    assert_eq!(
        payload["output"][0]["image_url"],
        "data:image/png;base64,AAAA"
    );
}

#[test]
fn tool_call_output_item_file_part() {
    let call = make_tool_call();
    let part = ToolOutputPart::File {
        file_data: Some("ZmFrZS1kYXRh".into()),
        file_url: None,
        file_id: None,
        filename: Some("foo.txt".into()),
    };
    let payload = ItemHelpers::tool_call_output_item(&call, &ToolOutput::Parts(vec![part]));
    assert_eq!(payload["output"][0]["type"], "input_file");
    assert_eq!(payload["output"][0]["file_data"], "ZmFrZS1kYXRh");
    assert_eq!(payload["output"][0]["filename"], "foo.txt");
}

#[test]
fn tool_call_output_item_mixed_list() {
    let call = make_tool_call();
    let parts = vec![
        ToolOutputPart::text("a"),
        ToolOutputPart::image_url("http://example/img.png"),
        ToolOutputPart::File {
            file_data: Some("ZmlsZS1kYXRh".into()),
            file_url: None,
            file_id: None,
            filename: None,
        },
    ];
    let payload = ItemHelpers::tool_call_output_item(&call, &ToolOutput::Parts(parts));
    let items = payload["output"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0]["type"], "input_text");
    assert_eq!(items[0]["text"], "a");
    assert_eq!(items[1]["type"], "input_image");
    assert_eq!(items[1]["image_url"], "http://example/img.png");
    assert_eq!(items[2]["type"], "input_file");
    assert_eq!(items[2]["file_data"], "ZmlsZS1kYXRh");
}

#[test]
fn tool_call_output_item_image_forwards_file_id_and_detail() {
    let call = make_tool_call();
    let part = ToolOutputPart::Image {
        image_url: None,
        file_id: Some("file_123".into()),
        detail: Some("high".into()),
    };
    let payload = ItemHelpers::tool_call_output_item(&call, &ToolOutput::Parts(vec![part]));
    assert_eq!(payload["output"][0]["type"], "input_image");
    assert_eq!(payload["output"][0]["file_id"], "file_123");
    assert_eq!(payload["output"][0]["detail"], "high");
    // Fields that were not set are omitted rather than sent as null.
    assert!(payload["output"][0].get("image_url").is_none());
}

#[test]
fn tool_call_output_item_file_forwards_file_id_and_filename() {
    let call = make_tool_call();
    let part = ToolOutputPart::File {
        file_data: None,
        file_url: None,
        file_id: Some("file_456".into()),
        filename: Some("report.pdf".into()),
    };
    let payload = ItemHelpers::tool_call_output_item(&call, &ToolOutput::Parts(vec![part]));
    assert_eq!(payload["output"][0]["type"], "input_file");
    assert_eq!(payload["output"][0]["file_id"], "file_456");
    assert_eq!(payload["output"][0]["filename"], "report.pdf");
}

#[test]
fn tool_call_output_item_file_forwards_file_url() {
    let call = make_tool_call();
    let part = ToolOutputPart::File {
        file_data: None,
        file_url: Some("https://example.com/report.pdf".into()),
        file_id: None,
        filename: None,
    };
    let payload = ItemHelpers::tool_call_output_item(&call, &ToolOutput::Parts(vec![part]));
    assert_eq!(payload["output"][0]["type"], "input_file");
    assert_eq!(
        payload["output"][0]["file_url"],
        "https://example.com/report.pdf"
    );
}

#[test]
fn tool_call_output_item_text_output_is_sent_verbatim() {
    let call = make_tool_call();
    let payload = ItemHelpers::tool_call_output_item(&call, &ToolOutput::Text("hey".into()));
    assert_eq!(payload["output"], "hey");
}

#[test]
fn tool_call_output_item_json_output_is_stringified() {
    let call = make_tool_call();
    let payload =
        ItemHelpers::tool_call_output_item(&call, &ToolOutput::Json(json!({"msg": "1234"})));
    assert!(payload["output"].is_string());
    assert_eq!(payload["output"], r#"{"msg":"1234"}"#);
}

// --- tests/test_result_cast.py (the part that does not depend on Python weakrefs) ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: i64,
}

async fn run_with_final_output(final_output: &str) -> openai_agents_sdk::RunResult<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message(final_output)]))
        .unwrap();
    let agent = Agent::<()>::new("test").with_model_instance(model);
    Runner::run(&agent, "test").await.unwrap()
}

#[tokio::test]
async fn result_cast_typechecks() {
    let result = run_with_final_output("test").await;
    assert_eq!(result.final_output_as::<String>().unwrap(), "test");

    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![assistant_message(
            json!({"bar": 1}).to_string(),
        )]))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_output_type::<Foo>();
    let result = Runner::run(&agent, "test").await.unwrap();
    assert_eq!(result.final_output_as::<Foo>().unwrap(), Foo { bar: 1 });
}

/// Python's `final_output_as(..., raise_if_incorrect_type=True)` raises `TypeError`; Rust's
/// `final_output_as` is always checked and returns `Err` instead.
#[tokio::test]
async fn bad_cast_returns_an_error() {
    let result = run_with_final_output("test").await;
    assert!(result.final_output_as::<Foo>().is_err());
    assert!(result.final_output_as::<i64>().is_err());
}

#[tokio::test]
async fn agent_tool_invocation_is_none_for_a_plain_run() {
    let result = run_with_final_output("ok").await;
    assert!(result.agent_tool_invocation.is_none());
}
