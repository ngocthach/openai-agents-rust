//! Port of `tests/models/test_openai_chatcompletions_converter.py`: the Chat Completions
//! converter in both directions (message -> output items, items -> chat messages).

use std::sync::Arc;

use openai_agents_sdk::agent_output::{AgentOutputSchema, AgentOutputSchemaBase};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::models::interface::ModelRequest;
use openai_agents_sdk::models::openai_chatcompletions::{
    ChatConversionOptions, convert_chat_tool_choice, extract_all_content, extract_text_content,
    items_to_messages, items_to_messages_with, parse_chat_completion, parse_chat_completion_with,
};
use openai_agents_sdk::models::{FAKE_RESPONSES_ID, OpenAIChatCompletionsModel, OpenAIClient};
use serde_json::{Value, json};

/// A Chat Completions response carrying one assistant message.
fn completion(message: Value) -> Value {
    json!({"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})
}

fn items(values: Vec<Value>) -> Vec<InputItem> {
    values.into_iter().map(InputItem::from_value).collect()
}

fn convert(values: Vec<Value>) -> Vec<Value> {
    items_to_messages(&items(values), None).unwrap()
}

fn convert_err(values: Vec<Value>) -> String {
    items_to_messages(&items(values), None)
        .unwrap_err()
        .to_string()
}

fn strict_options() -> ChatConversionOptions {
    ChatConversionOptions {
        strict_feature_validation: true,
        ..Default::default()
    }
}

// --------------------------------------------------------------------------------------
// message_to_output_items
// --------------------------------------------------------------------------------------

#[test]
fn message_to_output_items_with_text_only() {
    let r = parse_chat_completion(
        &completion(json!({"role": "assistant", "content": "Hello"})),
        None,
        false,
    )
    .unwrap();
    assert_eq!(r.output.len(), 1);
    let item = r.output[0].to_value();
    assert_eq!(item["id"], FAKE_RESPONSES_ID);
    assert_eq!(item["role"], "assistant");
    assert_eq!(item["type"], "message");
    assert_eq!(item["status"], "completed");
    assert_eq!(item["content"].as_array().unwrap().len(), 1);
    assert_eq!(item["content"][0]["type"], "output_text");
    assert_eq!(item["content"][0]["text"], "Hello");
}

#[test]
fn message_to_output_items_rejects_audio() {
    let err = parse_chat_completion(
        &completion(json!({
            "role": "assistant", "content": Value::Null,
            "audio": {"id": "audio-1", "data": "AAA=", "expires_at": 1, "transcript": "hi"},
        })),
        None,
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("Audio is not currently supported"), "{err}");
}

#[test]
fn message_to_output_items_keeps_url_citation_annotations() {
    let r = parse_chat_completion(
        &completion(json!({
            "role": "assistant", "content": "It will rain tomorrow.",
            "annotations": [{"type": "url_citation", "url_citation": {
                "start_index": 0, "end_index": 22,
                "url": "https://example.com/weather", "title": "Weather"}}],
        })),
        None,
        false,
    )
    .unwrap();
    let item = r.output[0].to_value();
    assert_eq!(
        item["content"][0]["annotations"],
        json!([{
            "end_index": 22, "start_index": 0, "title": "Weather",
            "type": "url_citation", "url": "https://example.com/weather",
        }])
    );
}

#[test]
fn message_to_output_items_with_refusal() {
    let r = parse_chat_completion(
        &completion(json!({"role": "assistant", "refusal": "I'm sorry"})),
        None,
        false,
    )
    .unwrap();
    assert_eq!(r.output.len(), 1);
    let item = r.output[0].to_value();
    assert_eq!(item["content"].as_array().unwrap().len(), 1);
    assert_eq!(item["content"][0]["type"], "refusal");
    assert_eq!(item["content"][0]["refusal"], "I'm sorry");
}

#[test]
fn message_to_output_items_with_tool_call() {
    let r = parse_chat_completion(
        &completion(json!({"role": "assistant", "content": "Hi", "tool_calls": [
            {"id": "tool1", "type": "function",
             "function": {"name": "myfn", "arguments": "{\"x\":1}"}}
        ]})),
        None,
        false,
    )
    .unwrap();
    assert_eq!(r.output.len(), 2);
    assert_eq!(r.output[0].to_value()["type"], "message");
    let call = r.output[1].to_value();
    assert_eq!(call["id"], FAKE_RESPONSES_ID);
    assert_eq!(call["call_id"], "tool1");
    assert_eq!(call["name"], "myfn");
    assert_eq!(call["arguments"], "{\"x\":1}");
    assert_eq!(call["type"], "function_call");
}

#[test]
fn message_to_output_items_with_custom_tool_call_keeps_default_compatibility() {
    let r = parse_chat_completion(
        &completion(json!({"role": "assistant", "tool_calls": [
            {"id": "tool1", "type": "custom", "custom": {"name": "raw_tool", "input": "payload"}}
        ]})),
        None,
        false,
    )
    .unwrap();
    assert!(r.output.is_empty());
}

#[test]
fn message_to_output_items_with_custom_tool_call_raises_in_strict_mode() {
    let err = parse_chat_completion_with(
        &completion(json!({"role": "assistant", "tool_calls": [
            {"id": "tool1", "type": "custom", "custom": {"name": "raw_tool", "input": "payload"}}
        ]})),
        None,
        false,
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("Custom tool calls are not supported"), "{err}");
}

#[test]
fn message_to_output_items_with_mixed_custom_tool_call_raises_in_strict_mode() {
    let err = parse_chat_completion_with(
        &completion(json!({"role": "assistant", "tool_calls": [
            {"id": "function-tool", "type": "function",
             "function": {"name": "myfn", "arguments": "{\"x\":1}"}},
            {"id": "custom-tool", "type": "custom",
             "custom": {"name": "raw_tool", "input": "payload"}}
        ]})),
        None,
        false,
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("Custom tool calls are not supported"), "{err}");
}

// --------------------------------------------------------------------------------------
// items_to_messages
// --------------------------------------------------------------------------------------

#[test]
fn items_to_messages_with_string_user_content() {
    let msgs = items_to_messages(&[InputItem::user("Ask me anything")], None).unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "Ask me anything");
}

#[test]
fn items_to_messages_with_easy_input_message() {
    let msgs = convert(vec![json!({"role": "user", "content": "How are you?"})]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "How are you?");
}

#[test]
fn easy_input_assistant_optional_keys_match_bare_shape() {
    let contents = [
        json!("hello"),
        json!(""),
        json!([]),
        json!([{"type": "input_text", "text": "hello"}]),
    ];
    let optionals = [
        json!({"type": "message"}),
        json!({"phase": "commentary"}),
        json!({"type": "message", "phase": "final_answer"}),
    ];
    for content in &contents {
        let bare = convert(vec![json!({"role": "assistant", "content": content})]);
        for extra in &optionals {
            let mut item = json!({"role": "assistant", "content": content});
            for (k, v) in extra.as_object().unwrap() {
                item[k] = v.clone();
            }
            assert_eq!(convert(vec![item]), bare, "{content} / {extra}");
        }
    }
}

#[test]
fn items_to_messages_accepts_raw_chat_completions_user_content_parts() {
    let msgs = convert(vec![json!({"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {
            "url": "https://example.com/image.png", "detail": "high"}},
    ]})]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(
        msgs[0]["content"],
        json!([
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {
                "url": "https://example.com/image.png", "detail": "high"}},
        ])
    );
}

#[test]
fn items_to_messages_with_output_message_and_function_call() {
    for include_id in [true, false] {
        let mut message = json!({
            "id": "42", "type": "message", "role": "assistant", "status": "completed",
            "content": [
                {"text": "Part 1", "type": "output_text", "annotations": [], "logprobs": []},
                {"refusal": "won't do that", "type": "refusal"},
            ],
        });
        if !include_id {
            message.as_object_mut().unwrap().remove("id");
        }
        let msgs = convert(vec![
            message,
            json!({"id": "99", "call_id": "abc", "name": "math", "arguments": "{}",
                   "type": "function_call"}),
        ]);
        assert_eq!(msgs.len(), 1, "include_id={include_id}");
        let assistant = &msgs[0];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["content"], "Part 1");
        assert_eq!(assistant["refusal"], "won't do that");
        let calls = assistant["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "math");
        assert_eq!(calls[0]["function"]["arguments"], "{}");
    }
}

#[test]
fn idless_assistant_output_rejects_unsupported_content() {
    for content in [
        json!("hello"),
        json!([{"type": "input_text", "text": "hello"}]),
        json!([{"type": "output_audio", "data": "AAAA"}]),
    ] {
        let err = convert_err(vec![json!({
            "type": "message", "role": "assistant", "status": "completed", "content": content,
        })]);
        assert!(err.contains("Unhandled item type or structure"), "{err}");
    }
}

/// One completed tool turn in the order the streaming handler emits it.
fn turn_items_function_call_first() -> Vec<Value> {
    vec![
        json!({"id": "99", "call_id": "abc", "name": "math", "arguments": "{}",
               "type": "function_call"}),
        json!({"id": "42", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"text": "Let me calculate.", "type": "output_text",
                            "annotations": []}]}),
        json!({"type": "function_call_output", "call_id": "abc", "output": "4"}),
    ]
}

#[test]
fn items_to_messages_merges_function_call_before_output_message() {
    let msgs = convert(turn_items_function_call_first());
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[0]["content"], "Let me calculate.");
    let calls = msgs[0]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "math");
    assert_eq!(msgs[1]["role"], "tool");
    assert_eq!(msgs[1]["tool_call_id"], "abc");
}

#[test]
fn items_to_messages_streamed_and_nonstreamed_turn_order_converge() {
    let streamed = turn_items_function_call_first();
    let non_streamed = vec![
        streamed[1].clone(),
        streamed[0].clone(),
        streamed[2].clone(),
    ];
    assert_eq!(convert(streamed), convert(non_streamed));
}

#[test]
fn items_to_messages_does_not_merge_a_later_turn_into_a_tool_turn() {
    let mut values = turn_items_function_call_first();
    values.push(json!({"id": "43", "type": "message", "role": "assistant",
                       "status": "completed",
                       "content": [{"text": "Done.", "type": "output_text",
                                    "annotations": []}]}));
    let msgs = convert(values);
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "assistant");
    assert!(!msgs[0]["tool_calls"].as_array().unwrap().is_empty());
    assert_eq!(msgs[1]["role"], "tool");
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["content"], "Done.");
    assert!(msgs[2].get("tool_calls").is_none());
}

#[test]
fn items_to_messages_merges_refusal_into_pending_tool_call_message() {
    let msgs = convert(vec![
        json!({"id": "99", "call_id": "abc", "name": "math", "arguments": "{}",
               "type": "function_call"}),
        json!({"id": "42", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"refusal": "won't do that", "type": "refusal"}]}),
    ]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["refusal"], "won't do that");
    assert!(!msgs[0]["tool_calls"].as_array().unwrap().is_empty());
}

#[test]
fn items_to_messages_accepts_statusless_output_message() {
    for include_id in [true, false] {
        let mut message = json!({
            "id": "msg_123", "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        });
        if !include_id {
            message.as_object_mut().unwrap().remove("id");
        }
        assert_eq!(
            convert(vec![message]),
            vec![json!({"role": "assistant", "content": "hello"})],
            "include_id={include_id}"
        );
    }
}

// --------------------------------------------------------------------------------------
// tool choice / response format
// --------------------------------------------------------------------------------------

fn body_with(settings: ModelSettings, schema: Option<Arc<dyn AgentOutputSchemaBase>>) -> Value {
    let model = OpenAIChatCompletionsModel::new("gpt-4o", OpenAIClient::new("sk-test"));
    let req = ModelRequest {
        model_settings: settings,
        output_schema: schema,
        ..Default::default()
    };
    model.build_request_body(&req, false).unwrap()
}

/// Rust models plain text as "no schema", but the trait also exposes `is_plain_text`, which
/// `AgentOutputSchema(str)` reports in Python. Both paths must omit `response_format`.
#[derive(Debug)]
struct PlainTextSchema;

impl AgentOutputSchemaBase for PlainTextSchema {
    fn is_plain_text(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        "str".into()
    }

    fn json_schema(&self) -> openai_agents_sdk::Result<Value> {
        Ok(json!({"type": "string"}))
    }

    fn is_strict_json_schema(&self) -> bool {
        true
    }

    fn validate_json(&self, json_str: &str) -> openai_agents_sdk::Result<Value> {
        Ok(json!(json_str))
    }
}

#[test]
fn convert_tool_choice_handles_standard_and_named_options() {
    assert!(
        body_with(ModelSettings::default(), None)
            .get("tool_choice")
            .is_none()
    );
    for name in ["auto", "required", "none"] {
        assert_eq!(
            convert_chat_tool_choice(&ToolChoice::Named(name.to_string())),
            json!(name)
        );
    }
    let named = convert_chat_tool_choice(&ToolChoice::Named("mytool".to_string()));
    assert_eq!(named["type"], "function");
    assert_eq!(named["function"]["name"], "mytool");
}

#[test]
fn convert_tool_choice_allows_tool_search_as_named_function_for_chat_models() {
    let named = convert_chat_tool_choice(&ToolChoice::Named("tool_search".to_string()));
    assert_eq!(named["type"], "function");
    assert_eq!(named["function"]["name"], "tool_search");
}

#[test]
fn convert_response_format_returns_not_given_for_plain_text_and_dict_for_schemas() {
    assert!(
        body_with(ModelSettings::default(), None)
            .get("response_format")
            .is_none()
    );
    let plain: Arc<dyn AgentOutputSchemaBase> = Arc::new(PlainTextSchema);
    assert!(
        body_with(ModelSettings::default(), Some(plain))
            .get("response_format")
            .is_none()
    );
    let schema = AgentOutputSchema::new::<i64>().unwrap();
    let strict = schema.is_strict_json_schema();
    let json_schema = schema.json_schema().unwrap();
    let schema: Arc<dyn AgentOutputSchemaBase> = Arc::new(schema);
    let body = body_with(ModelSettings::default(), Some(schema));
    let rf = &body["response_format"];
    assert_eq!(rf["type"], "json_schema");
    assert_eq!(rf["json_schema"]["name"], "final_output");
    assert_eq!(rf["json_schema"]["strict"], json!(strict));
    assert_eq!(rf["json_schema"]["schema"], json_schema);
}

// --------------------------------------------------------------------------------------
// function call outputs
// --------------------------------------------------------------------------------------

#[test]
fn items_to_messages_with_function_output_item() {
    let msgs = convert(vec![json!({
        "type": "function_call_output", "call_id": "somecall", "output": "{\"foo\": \"bar\"}",
    })]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "tool");
    assert_eq!(msgs[0]["tool_call_id"], "somecall");
    assert_eq!(msgs[0]["content"], "{\"foo\": \"bar\"}");
}

#[test]
fn items_to_messages_with_non_text_only_function_output_uses_placeholder_by_default() {
    let msgs = convert(vec![json!({
        "type": "function_call_output", "call_id": "somecall",
        "output": [{"type": "input_image", "image_url": "https://example.com/image.png"}],
    })]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "tool");
    assert_eq!(msgs[0]["tool_call_id"], "somecall");
    assert_eq!(msgs[0]["content"], "[tool output omitted]");
}

#[test]
fn items_to_messages_with_non_text_only_function_output_raises_in_strict_mode() {
    let err = items_to_messages_with(
        &items(vec![json!({
            "type": "function_call_output", "call_id": "somecall",
            "output": [{"type": "input_image", "image_url": "https://example.com/image.png"}],
        })]),
        None,
        &strict_options(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("cannot be empty or contain only non-text content"),
        "{err}"
    );
}

#[test]
fn items_to_messages_with_empty_function_output_uses_placeholder_by_default() {
    let msgs = convert(vec![json!({
        "type": "function_call_output", "call_id": "somecall", "output": [],
    })]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["content"], "[tool output omitted]");
}

#[test]
fn items_to_messages_with_empty_function_output_raises_in_strict_mode() {
    let err = items_to_messages_with(
        &items(vec![json!({
            "type": "function_call_output", "call_id": "somecall", "output": [],
        })]),
        None,
        &strict_options(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("cannot be empty or contain only non-text content"),
        "{err}"
    );
}

#[test]
fn items_to_messages_with_mixed_function_output_keeps_text_by_default() {
    let msgs = convert(vec![json!({
        "type": "function_call_output", "call_id": "somecall",
        "output": [
            {"type": "input_text", "text": "visible text"},
            {"type": "input_image", "image_url": "https://example.com/image.png"},
        ],
    })]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(
        msgs[0]["content"],
        json!([{"type": "text", "text": "visible text"}])
    );
}

#[test]
fn items_to_messages_can_preserve_non_text_function_output() {
    let options = ChatConversionOptions {
        preserve_tool_output_all_content: true,
        ..Default::default()
    };
    let msgs = items_to_messages_with(
        &items(vec![json!({
            "type": "function_call_output", "call_id": "somecall",
            "output": [{"type": "input_image", "image_url": "https://example.com/image.png"}],
        })]),
        None,
        &options,
    )
    .unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(
        msgs[0]["content"],
        json!([{"type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "detail": "auto"}}])
    );
}

// --------------------------------------------------------------------------------------
// content extraction helpers
// --------------------------------------------------------------------------------------

#[test]
fn extract_all_and_text_content_for_strings_and_lists() {
    assert_eq!(
        extract_all_content(&json!("just text")).unwrap(),
        "just text"
    );
    assert_eq!(
        extract_text_content(&json!("just text")).unwrap(),
        "just text"
    );
    let content = json!([
        {"type": "input_text", "text": "one"},
        {"type": "input_text", "text": "two"},
    ]);
    let all = extract_all_content(&content).unwrap();
    let all = all.as_array().unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0]["type"], "text");
    assert_eq!(all[0]["text"], "one");
    assert_eq!(all[1]["type"], "text");
    assert_eq!(all[1]["text"], "two");
    let text = extract_text_content(&content).unwrap();
    let text = text.as_array().unwrap();
    assert!(text.iter().all(|p| p["type"] == "text"));
    assert_eq!(
        text.iter().map(|p| p["text"].clone()).collect::<Vec<_>>(),
        vec![json!("one"), json!("two")]
    );
}

#[test]
fn extract_all_content_handles_input_audio() {
    let parts = extract_all_content(&json!([
        {"type": "input_audio", "input_audio": {"data": "AAA=", "format": "wav"}}
    ]))
    .unwrap();
    assert_eq!(
        parts,
        json!([{"type": "input_audio", "input_audio": {"data": "AAA=", "format": "wav"}}])
    );
}

#[test]
fn extract_all_content_preserves_prompt_cache_breakpoints() {
    let bp = json!({"mode": "explicit"});
    let parts = extract_all_content(&json!([
        {"type": "input_text", "text": "one", "prompt_cache_breakpoint": bp},
        {"type": "input_image", "image_url": "https://example.com/image.png",
         "prompt_cache_breakpoint": bp},
        {"type": "input_audio", "input_audio": {"data": "AAA=", "format": "wav"},
         "prompt_cache_breakpoint": bp},
        {"type": "input_file", "file_data": "data:text/plain;base64,SGVsbG8=",
         "filename": "hello.txt", "prompt_cache_breakpoint": bp},
    ]))
    .unwrap();
    let got: Vec<Value> = parts
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["prompt_cache_breakpoint"].clone())
        .collect();
    assert_eq!(got, vec![bp.clone(), bp.clone(), bp.clone(), bp]);
}

#[test]
fn raw_chat_content_aliases_preserve_prompt_cache_breakpoints() {
    let bp = json!({"mode": "explicit"});
    let parts = extract_all_content(&json!([
        {"type": "text", "text": "one", "prompt_cache_breakpoint": bp},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"},
         "prompt_cache_breakpoint": bp},
    ]))
    .unwrap();
    let got: Vec<Value> = parts
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["prompt_cache_breakpoint"].clone())
        .collect();
    assert_eq!(got, vec![bp.clone(), bp]);
}

#[test]
fn extract_all_content_rejects_invalid_input_audio() {
    assert!(
        extract_all_content(&json!([
            {"type": "input_audio", "input_audio": {"format": "wav"}}
        ]))
        .is_err()
    );
}

#[test]
fn extract_all_content_supports_input_file_file_id() {
    let parts = extract_all_content(&json!([
        {"type": "input_file", "file_id": "file-abc123", "filename": "hello.txt"}
    ]))
    .unwrap();
    assert_eq!(
        parts,
        json!([{"type": "file", "file": {"file_id": "file-abc123", "filename": "hello.txt"}}])
    );
}

#[test]
fn extract_all_content_prefers_input_file_data_over_file_id() {
    let parts = extract_all_content(&json!([
        {"type": "input_file", "file_data": "data:text/plain;base64,SGVsbG8=",
         "file_id": "file-abc123"}
    ]))
    .unwrap();
    assert_eq!(
        parts,
        json!([{"type": "file", "file": {"file_data": "data:text/plain;base64,SGVsbG8="}}])
    );
}

#[test]
fn extract_all_content_rejects_input_file_without_data_or_id() {
    assert!(
        extract_all_content(&json!([{"type": "input_file", "filename": "hello.txt"}])).is_err()
    );
}

// --------------------------------------------------------------------------------------
// roles, references, unknown items
// --------------------------------------------------------------------------------------

#[test]
fn items_to_messages_handles_system_and_developer_roles() {
    let sys = convert(vec![json!({"role": "system", "content": "setup"})]);
    assert_eq!(sys.len(), 1);
    assert_eq!(sys[0]["role"], "system");
    assert_eq!(sys[0]["content"], "setup");
    let dev = convert(vec![json!({"role": "developer", "content": "debug"})]);
    assert_eq!(dev.len(), 1);
    assert_eq!(dev[0]["role"], "developer");
    assert_eq!(dev[0]["content"], "debug");
}

#[test]
fn maybe_input_message_allows_message_typed_dict() {
    let msgs = convert(vec![
        json!({"type": "message", "role": "user", "content": "hi"}),
    ]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "hi");
}

#[test]
fn tool_call_conversion() {
    let msgs = convert(vec![json!({
        "id": "tool1", "call_id": "abc", "name": "math", "arguments": "{}",
        "type": "function_call",
    })]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "assistant");
    // The content key is present even when it is null, for Chat Completions compatibility.
    assert_eq!(
        msgs[0].as_object().unwrap().get("content"),
        Some(&Value::Null)
    );
    let calls = msgs[0]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "abc");
    assert_eq!(calls[0]["function"]["name"], "math");
    assert_eq!(calls[0]["function"]["arguments"], "{}");
}

#[test]
fn input_message_with_all_roles() {
    for role in ["user", "system", "developer"] {
        let msgs = convert(vec![
            json!({"type": "message", "role": role, "content": "hi"}),
        ]);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], role);
        assert_eq!(msgs[0]["content"], "hi");
    }
}

#[test]
fn item_reference_errors() {
    assert!(
        items_to_messages(
            &items(vec![json!({"type": "item_reference", "id": "item1"})]),
            None
        )
        .is_err()
    );
}

#[test]
fn unknown_object_errors() {
    let err = convert_err(vec![json!({"not": "an item"})]);
    assert!(err.contains("Unhandled item type or structure"), "{err}");
}

#[test]
fn assistant_messages_in_history() {
    let msgs = convert(vec![
        json!({"role": "user", "content": "Hello"}),
        json!({"role": "assistant", "content": "Hello?"}),
        json!({"role": "user", "content": "What was my Name?"}),
    ]);
    assert_eq!(
        msgs,
        vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hello?"}),
            json!({"role": "user", "content": "What was my Name?"}),
        ]
    );
}
