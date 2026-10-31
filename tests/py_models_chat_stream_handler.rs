//! Port of the handler-level tests in `tests/models/test_openai_chatcompletions_stream.py`:
//! Chat Completions chunks in, Responses stream events out, with no transport involved.
//! (`ChatCmplStreamHandler.handle_stream` and `buffer_tool_call_stream`.)

use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::models::openai_chatcompletions::{
    ChatConversionOptions, ChatStreamOptions, buffer_tool_call_chunks, handle_chat_stream_chunks,
    items_to_messages_with, merge_buffered_metadata,
};
use serde_json::{Map, Value, json};

// ---- helpers -----------------------------------------------------------------------------

fn chunk_with(choices: Vec<Value>) -> Value {
    json!({"id": "chunk-id", "created": 1, "model": "fake",
           "object": "chat.completion.chunk", "choices": choices})
}

fn choice(delta: Value) -> Value {
    json!({"index": 0, "delta": delta})
}

fn chunk(delta: Value) -> Value {
    chunk_with(vec![choice(delta)])
}

fn finished(delta: Value, reason: &str) -> Value {
    chunk_with(vec![
        json!({"index": 0, "delta": delta, "finish_reason": reason}),
    ])
}

fn thinking_chunk(delta: Value) -> Value {
    let mut c = chunk(delta);
    c["model"] = json!("anthropic/claude-4-opus");
    c
}

fn events(chunks: &[Value]) -> Vec<Value> {
    handle_chat_stream_chunks(chunks, &ChatStreamOptions::default()).unwrap()
}

fn events_for(chunks: &[Value], model: &str) -> Vec<Value> {
    handle_chat_stream_chunks(
        chunks,
        &ChatStreamOptions {
            model: Some(model.to_string()),
            ..Default::default()
        },
    )
    .unwrap()
}

fn event_types(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

fn completed(events: &[Value]) -> &Value {
    events
        .iter()
        .find(|e| e["type"] == "response.completed")
        .expect("a completed event")
}

fn of_type<'a>(events: &'a [Value], ty: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["type"] == ty).collect()
}

fn output(events: &[Value]) -> &Vec<Value> {
    completed(events)["response"]["output"].as_array().unwrap()
}

fn reasoning_done(events: &[Value]) -> &Value {
    events
        .iter()
        .find(|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "reasoning")
        .expect("a reasoning item done event")
}

fn replay(items: &[Value], model: &str, preserve_thinking_blocks: bool) -> Vec<Value> {
    items_to_messages_with(
        &items
            .iter()
            .cloned()
            .map(InputItem::from_value)
            .collect::<Vec<_>>(),
        None,
        &ChatConversionOptions {
            model: Some(model.to_string()),
            preserve_thinking_blocks,
            ..Default::default()
        },
    )
    .unwrap()
}

fn annotation(url: &str, title: &str) -> Value {
    json!({"type": "url_citation", "url_citation": {
        "start_index": 0, "end_index": 22, "url": url, "title": title}})
}

fn weather_citation() -> Value {
    annotation("https://example.com/weather", "Weather")
}

fn streamed_annotations(events: &[Value]) -> Vec<Value> {
    events.last().unwrap()["response"]["output"][0]["content"][0]["annotations"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn tool_call(index: u64, id: Option<&str>, name: Option<&str>, arguments: &str) -> Value {
    let mut tc = json!({"index": index, "type": "function",
                        "function": {"name": name, "arguments": arguments}});
    if let Some(id) = id {
        tc["id"] = json!(id);
    }
    tc
}

// ---- multiple choices ----------------------------------------------------------------------

#[test]
fn stream_handler_filters_multiple_choices_by_default() {
    let chunks = vec![
        chunk_with(vec![
            json!({"index": 1, "delta": {"content": "ignored-first"}}),
        ]),
        chunk_with(vec![
            json!({"index": 0, "delta": {"content": "kept"}}),
            json!({"index": 1, "delta": {"content": "ignored-second"}}),
        ]),
        {
            let mut c = chunk_with(vec![json!({"index": 2,
                                               "delta": {"content": "ignored-third"}})]);
            c["usage"] = json!({"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3});
            c
        },
    ];
    let events = events(&chunks);
    let deltas: Vec<&str> = of_type(&events, "response.output_text.delta")
        .iter()
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, ["kept"]);
    assert_eq!(output(&events)[0]["content"][0]["text"], "kept");
    assert_eq!(completed(&events)["response"]["usage"]["total_tokens"], 3);
}

#[test]
fn stream_handler_keeps_empty_choice_usage_chunks() {
    let mut c = chunk_with(vec![]);
    c["usage"] = json!({"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 0}});
    let events = events(&[c]);
    assert_eq!(
        event_types(&events),
        ["response.created", "response.completed"]
    );
    assert_eq!(output(&events), &Vec::<Value>::new());
    assert_eq!(completed(&events)["response"]["usage"]["total_tokens"], 3);
}

#[test]
fn stream_handler_rejects_multiple_choices_in_strict_mode() {
    let c = chunk_with(vec![
        json!({"index": 0, "delta": {"content": "first"}}),
        json!({"index": 1, "delta": {"content": "second"}}),
    ]);
    let err = handle_chat_stream_chunks(
        &[c],
        &ChatStreamOptions {
            strict_feature_validation: true,
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("multiple choices or nonzero"), "{err}");
}

#[test]
fn stream_handler_rejects_nonzero_choice_index_in_strict_mode() {
    let c = chunk_with(vec![json!({"index": 1, "delta": {"content": "second"}})]);
    let err = handle_chat_stream_chunks(
        &[c],
        &ChatStreamOptions {
            strict_feature_validation: true,
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("multiple choices or nonzero"), "{err}");
}

// ---- buffering layer -----------------------------------------------------------------------

#[test]
fn buffer_tool_call_stream_merges_provider_metadata() {
    let mut d1 = tool_call(0, Some("tool-id"), Some("my_func"), "{\"a\":");
    d1["provider_specific_fields"] =
        json!({"nested": {"keep": "provider", "stable": {"value": 1}}, "replace": "old"});
    d1["extra_content"] =
        json!({"google": {"thought_signature": "sig-1", "stable": {"value": "kept"}}});
    let mut d2 = tool_call(0, None, None, "1}");
    d2["provider_specific_fields"] = json!({"nested": {"stable": {}, "new": "provider"},
                                            "replace": "new"});
    d2["extra_content"] = json!({"google": {"stable": {}, "new": "extra"}});
    let buffered = buffer_tool_call_chunks(vec![
        chunk(json!({"tool_calls": [d1]})),
        chunk(json!({"tool_calls": [d2]})),
    ])
    .unwrap();
    assert_eq!(buffered.len(), 1);
    let call = &buffered[0]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(call["function"]["arguments"], "{\"a\":1}");
    assert_eq!(
        call["provider_specific_fields"],
        json!({"nested": {"keep": "provider", "stable": {"value": 1}, "new": "provider"},
               "replace": "new"})
    );
    assert_eq!(
        call["extra_content"],
        json!({"google": {"thought_signature": "sig-1", "stable": {"value": "kept"},
                          "new": "extra"}})
    );
}

#[test]
fn merge_buffered_metadata_keeps_existing_scalar_when_empty_dict_arrives() {
    let current: Map<String, Value> = json!({"stable": "keep-me"}).as_object().unwrap().clone();
    let incoming: Map<String, Value> = json!({"stable": {}, "new": {}})
        .as_object()
        .unwrap()
        .clone();
    let merged = merge_buffered_metadata(Some(current), &incoming).unwrap();
    assert_eq!(
        Value::Object(merged),
        json!({"stable": "keep-me", "new": {}})
    );
}

#[test]
fn buffered_tool_call_delta_requires_id_and_name() {
    let no_id = buffer_tool_call_chunks(vec![chunk(json!({"tool_calls": [
        tool_call(0, None, Some("my_func"), "")]}))])
    .unwrap_err()
    .to_string();
    assert!(no_id.contains("without a tool call id"), "{no_id}");
    let no_name = buffer_tool_call_chunks(vec![chunk(json!({"tool_calls": [
        tool_call(0, Some("tool-id"), None, "")]}))])
    .unwrap_err()
    .to_string();
    assert!(no_name.contains("without a function name"), "{no_name}");
}

#[test]
fn audio_delta_raises_like_the_sync_path() {
    let c = chunk(json!({"content": "partial", "audio": {"id": "audio-1", "transcript": "hi"}}));
    let err = handle_chat_stream_chunks(&[c], &ChatStreamOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("Audio is not currently supported"), "{err}");
}

#[test]
fn buffered_audio_only_delta_raises_instead_of_completing_empty() {
    let audio = chunk(json!({"audio": {"id": "audio-1", "transcript": "hi"}}));
    let buffered = buffer_tool_call_chunks(vec![audio]).unwrap();
    let err = handle_chat_stream_chunks(&buffered, &ChatStreamOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("Audio is not currently supported"), "{err}");
}

#[test]
fn buffer_tool_call_stream_preserves_empty_choice_chunks() {
    let c = chunk_with(vec![]);
    assert_eq!(buffer_tool_call_chunks(vec![c.clone()]).unwrap(), vec![c]);
}

#[test]
fn buffer_tool_call_stream_keeps_passthrough_index_passthrough() {
    let custom = json!({"index": 0, "id": "custom-id", "type": "custom"});
    let function = tool_call(0, Some("function-id"), Some("my_func"), "{}");
    let buffered = buffer_tool_call_chunks(vec![
        chunk(json!({"tool_calls": [custom.clone()]})),
        chunk(json!({"tool_calls": [function.clone()]})),
    ])
    .unwrap();
    assert_eq!(buffered.len(), 2);
    assert_eq!(
        buffered[0]["choices"][0]["delta"]["tool_calls"],
        json!([custom])
    );
    assert_eq!(
        buffered[1]["choices"][0]["delta"]["tool_calls"],
        json!([function])
    );
}

#[test]
fn stream_handler_detects_passthrough_delta_shapes() {
    // A delta with output survives buffering; an empty one does not.
    let cases = [
        (json!({}), false),
        (json!({"content": "text"}), true),
        (json!({"refusal": "blocked"}), true),
        (json!({"reasoning_content": "summary"}), true),
        (json!({"reasoning": "scratchpad"}), true),
        (json!({"thinking_blocks": [{"thinking": "hidden"}]}), true),
        (json!({"audio": {"id": "audio-1"}}), true),
    ];
    for (delta, expected) in cases {
        let buffered = buffer_tool_call_chunks(vec![chunk(delta.clone())]).unwrap();
        assert_eq!(!buffered.is_empty(), expected, "{delta}");
    }
}

#[test]
fn stream_handler_ignores_choice_without_delta() {
    let c = chunk_with(vec![json!({"index": 0, "delta": null})]);
    let events = events(&[c]);
    assert_eq!(
        event_types(&events),
        ["response.created", "response.completed"]
    );
    assert_eq!(output(&events), &Vec::<Value>::new());
}

#[test]
fn buffer_tool_call_stream_forwards_content_filter_finish_reason() {
    let buffered = buffer_tool_call_chunks(vec![
        chunk(json!({"content": ""})),
        finished(json!({}), "content_filter"),
    ])
    .unwrap();
    let terminal: Vec<&Value> = buffered
        .iter()
        .flat_map(|c| c["choices"].as_array().unwrap())
        .filter(|c| c["finish_reason"] == "content_filter")
        .collect();
    assert_eq!(terminal.len(), 1);
    // The forwarded copy carries no delta output.
    assert_eq!(terminal[0]["delta"], json!({}));
}

#[test]
fn buffer_tool_call_stream_forwards_length_finish_reason() {
    let buffered = buffer_tool_call_chunks(vec![
        chunk(json!({"content": ""})),
        finished(json!({}), "length"),
    ])
    .unwrap();
    let terminal: Vec<&Value> = buffered
        .iter()
        .flat_map(|c| c["choices"].as_array().unwrap())
        .filter(|c| c["finish_reason"] == "length")
        .collect();
    assert_eq!(terminal.len(), 1);
    assert_eq!(terminal[0]["delta"], json!({}));
}

#[test]
fn buffer_tool_call_stream_does_not_duplicate_tool_calls_finish() {
    let call = tool_call(0, Some("tool-id"), Some("my_func"), "{\"a\": 1}");
    let buffered = buffer_tool_call_chunks(vec![
        chunk(json!({"tool_calls": [call.clone()]})),
        finished(json!({}), "tool_calls"),
    ])
    .unwrap();
    let finish: Vec<&Value> = buffered
        .iter()
        .flat_map(|c| c["choices"].as_array().unwrap())
        .filter(|c| c["finish_reason"] == "tool_calls")
        .collect();
    assert_eq!(finish.len(), 1);
    assert!(
        finish[0]["delta"]
            .get("tool_calls")
            .is_none_or(Value::is_null)
    );
    let last = &buffered.last().unwrap()["choices"][0];
    assert_eq!(last["delta"]["tool_calls"][0]["id"], "tool-id");
    assert_eq!(
        last["delta"]["tool_calls"][0]["function"]["arguments"],
        "{\"a\": 1}"
    );
    assert!(last["finish_reason"].is_null());
}

#[test]
fn stream_handler_buffering_keeps_a_citation_only_delta() {
    let chunks = buffer_tool_call_chunks(vec![
        chunk(json!({"role": "assistant", "content": "It will rain tomorrow."})),
        finished(json!({"annotations": [weather_citation()]}), "stop"),
    ])
    .unwrap();
    let events = events(&chunks);
    let urls: Vec<Value> = streamed_annotations(&events)
        .iter()
        .map(|a| a["url"].clone())
        .collect();
    assert_eq!(urls, [json!("https://example.com/weather")]);
}

// ---- plaintext reasoning and thinking blocks ---------------------------------------------

#[test]
fn stream_handler_converts_third_party_reasoning_text() {
    let events = events_for(
        &[
            chunk(json!({"reasoning": "think "})),
            chunk(json!({"reasoning": "hard"})),
            chunk(json!({"content": "answer"})),
        ],
        "third-party",
    );
    let deltas: Vec<&str> = of_type(&events, "response.reasoning_text.delta")
        .iter()
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, ["think ", "hard"]);
    assert_eq!(
        reasoning_done(&events)["item"]["content"][0]["text"],
        "think hard"
    );
    let item = &output(&events)[0];
    assert_eq!(item["content"][0]["text"], "think hard");
    assert_eq!(
        item["provider_data"],
        json!({"_chat_completions_reasoning_field": "reasoning", "model": "third-party",
               "response_id": "chunk-id"})
    );
    let messages = replay(output(&events), "third-party", false);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["reasoning"], "think hard");
}

#[test]
fn stream_handler_ignores_non_string_plaintext_reasoning() {
    let events = events_for(
        &[
            chunk(json!({"reasoning": {"bad": 1}})),
            chunk(json!({"content": "answer"})),
        ],
        "third-party",
    );
    assert!(!event_types(&events).iter().any(|t| matches!(
        *t,
        "response.reasoning_text.delta" | "response.reasoning_text.done"
    )));
    assert_eq!(output(&events).len(), 1);
    assert_eq!(output(&events)[0]["type"], "message");
    assert_eq!(output(&events)[0]["content"][0]["text"], "answer");
}

#[test]
fn stream_handler_prefers_reasoning_content_over_plaintext_reasoning() {
    for split in [false, true] {
        let deltas = if split {
            vec![
                json!({"reasoning": "raw"}),
                json!({"reasoning_content": "summary"}),
            ]
        } else {
            vec![json!({"reasoning": "raw", "reasoning_content": "summary"})]
        };
        let chunks: Vec<Value> = deltas
            .into_iter()
            .chain([json!({"content": "answer"})])
            .map(chunk)
            .collect();
        let events = events_for(&chunks, "deepseek-r1");
        let item = &output(&events)[0];
        assert_eq!(item["type"], "reasoning");
        assert_eq!(item["summary"][0]["text"], "summary");
        let content: Vec<&str> = item["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["text"].as_str().unwrap())
            .collect();
        assert_eq!(content, ["raw"]);
        assert!(
            item["provider_data"]
                .get("_chat_completions_reasoning_field")
                .is_none()
        );
        let messages = replay(output(&events), "deepseek-r1", false);
        assert_eq!(messages.len(), 1, "split={split}");
        assert_eq!(messages[0]["reasoning_content"], "summary");
        assert!(messages[0].get("reasoning").is_none());
    }
}

#[test]
fn stream_handler_prefers_later_thinking_blocks_over_plaintext_reasoning() {
    let events = events_for(
        &[
            thinking_chunk(json!({"reasoning": "raw"})),
            thinking_chunk(json!({"thinking_blocks": [
                {"type": "thinking", "thinking": "hidden", "signature": "sig"}]})),
            thinking_chunk(json!({"content": "answer"})),
        ],
        "anthropic/claude-4-opus",
    );
    let item = &output(&events)[0];
    assert_eq!(item["type"], "reasoning");
    let content: Vec<&str> = item["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["text"].as_str().unwrap())
        .collect();
    assert_eq!(content, ["raw", "hidden"]);
    assert!(
        item["provider_data"]
            .get("_chat_completions_reasoning_field")
            .is_none()
    );
    let messages = replay(output(&events), "anthropic/claude-4-opus", true);
    assert_eq!(messages.len(), 1);
    assert_eq!(
        messages[0]["thinking_blocks"],
        json!([{"type": "thinking", "thinking": "hidden", "signature": "sig"}])
    );
    assert!(messages[0].get("reasoning").is_none());
}

#[test]
fn stream_handler_preserves_thinking_blocks_with_reasoning_summary() {
    let events = events(&[chunk(json!({
        "reasoning_content": "summary",
        "thinking_blocks": [
            {"thinking": "hidden one ", "signature": "sig-1"},
            {"thinking": "hidden two", "signature": "sig-2"},
        ],
    }))]);
    let item = &output(&events)[0];
    assert_eq!(item["type"], "reasoning");
    assert_eq!(item["summary"][0]["text"], "summary");
    let content: Vec<&str> = item["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["text"].as_str().unwrap())
        .collect();
    assert_eq!(content, ["hidden one hidden two"]);
    assert_eq!(item["encrypted_content"], "sig-1\nsig-2");
    assert_eq!(
        item["provider_data"]["thinking_blocks"],
        json!([{"type": "thinking", "thinking": "hidden one ", "signature": "sig-1"},
               {"type": "thinking", "thinking": "hidden two", "signature": "sig-2"}])
    );
}

fn thinking_delta(thinking: &str, signature: &str) -> Value {
    thinking_chunk(json!({"thinking_blocks": [
        {"type": "thinking", "thinking": thinking, "signature": signature}]}))
}

#[test]
fn stream_handler_segments_thinking_blocks_at_signature_deltas() {
    let events = events(&[
        thinking_chunk(json!({"reasoning_content": "summary"})),
        thinking_delta("step ", ""),
        thinking_delta("one", ""),
        thinking_delta("", "SIG-1"),
        thinking_delta("step two", ""),
        thinking_delta("", "SIG-2"),
    ]);
    let item = &output(&events)[0];
    assert_eq!(
        item["provider_data"]["thinking_blocks"],
        json!([{"type": "thinking", "thinking": "step one", "signature": "SIG-1"},
               {"type": "thinking", "thinking": "step two", "signature": "SIG-2"}])
    );
    let content: Vec<&str> = item["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["text"].as_str().unwrap())
        .collect();
    assert_eq!(content, ["step onestep two"]);
    assert_eq!(item["encrypted_content"], "SIG-1\nSIG-2");
}

#[test]
fn stream_handler_finalizes_thinking_blocks_before_item_done() {
    let events = events(&[
        thinking_chunk(json!({"reasoning_content": "summary"})),
        thinking_delta("hidden", ""),
        thinking_delta("", "SIG"),
    ]);
    let done = reasoning_done(&events);
    assert_eq!(done["item"], output(&events)[0]);
    assert_eq!(
        done["item"]["provider_data"]["thinking_blocks"],
        json!([{"type": "thinking", "thinking": "hidden", "signature": "SIG"}])
    );
}

#[test]
fn stream_handler_preserves_redacted_thinking_without_summary() {
    let events = events(&[
        thinking_chunk(json!({"thinking_blocks": [{"type": "redacted_thinking",
                                                    "data": "BLOB"}]})),
        thinking_chunk(json!({"content": "visible answer"})),
    ]);
    let done = reasoning_done(&events);
    assert_eq!(done["output_index"], 0);
    assert_eq!(
        done["item"]["provider_data"]["thinking_blocks"],
        json!([{"type": "redacted_thinking", "data": "BLOB"}])
    );
    let kinds: Vec<&str> = output(&events)
        .iter()
        .map(|i| i["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["reasoning", "message"]);
    assert_eq!(
        output(&events)[0]["provider_data"]["thinking_blocks"],
        json!([{"type": "redacted_thinking", "data": "BLOB"}])
    );
    let text_events: Vec<&Value> = events
        .iter()
        .filter(|e| {
            matches!(
                e["type"].as_str().unwrap(),
                "response.content_part.added"
                    | "response.output_text.delta"
                    | "response.content_part.done"
            )
        })
        .collect();
    assert!(!text_events.is_empty());
    assert!(
        text_events
            .iter()
            .all(|e| e["output_index"] == 1 && e["content_index"] == 0)
    );
}

#[test]
fn stream_handler_places_refusal_after_redacted_thinking_item() {
    let events = events(&[
        thinking_chunk(json!({"thinking_blocks": [{"type": "redacted_thinking",
                                                    "data": "BLOB"}]})),
        thinking_chunk(json!({"refusal": "blocked"})),
    ]);
    let refusal_events: Vec<&Value> = events
        .iter()
        .filter(|e| {
            matches!(
                e["type"].as_str().unwrap(),
                "response.content_part.added"
                    | "response.refusal.delta"
                    | "response.content_part.done"
            )
        })
        .collect();
    assert!(!refusal_events.is_empty());
    assert!(
        refusal_events
            .iter()
            .all(|e| e["output_index"] == 1 && e["content_index"] == 0)
    );
    assert_eq!(
        output(&events)[1]["content"],
        json!([{"refusal": "blocked", "type": "refusal"}])
    );
}

#[test]
fn stream_handler_keeps_redacted_block_metadata_opaque() {
    let redacted = json!({"type": "redacted_thinking", "data": "BLOB",
                          "thinking": {"opaque": true}, "signature": {"opaque": true}});
    let events = events(&[thinking_chunk(
        json!({"thinking_blocks": [redacted.clone()]}),
    )]);
    let item = &reasoning_done(&events)["item"];
    assert_eq!(item["provider_data"]["thinking_blocks"], json!([redacted]));
    assert!(item.get("content").is_none_or(Value::is_null));
    assert!(item.get("encrypted_content").is_none_or(Value::is_null));
}

#[test]
fn stream_handler_emits_complete_signed_thinking_only_lifecycle() {
    let events = events(&[thinking_delta("hidden", ""), thinking_delta("", "SIG")]);
    assert_eq!(
        event_types(&events),
        [
            "response.created",
            "response.output_item.added",
            "response.reasoning_text.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(events[1]["output_index"], 0);
    assert_eq!(events[2]["output_index"], 0);
    assert_eq!(events[2]["text"], "hidden");
    let done_item = &events[3]["item"];
    assert_eq!(events[3]["output_index"], 0);
    assert_eq!(
        done_item["content"],
        json!([{"text": "hidden", "type": "reasoning_text"}])
    );
    assert_eq!(done_item["encrypted_content"], "SIG");
    assert_eq!(
        done_item["provider_data"]["thinking_blocks"],
        json!([{"type": "thinking", "thinking": "hidden", "signature": "SIG"}])
    );
    assert_eq!(output(&events), &vec![done_item.clone()]);
}

#[test]
fn stream_handler_rejects_non_string_thinking_fields() {
    let cases = [
        ("thinking", json!({"opaque": true}), "dict"),
        ("signature", json!({"opaque": true}), "dict"),
        ("thinking", Value::Null, "NoneType"),
        ("signature", Value::Null, "NoneType"),
    ];
    for (field, value, type_name) in cases {
        let mut block = json!({"type": "thinking", "thinking": "", "signature": ""});
        block[field] = value;
        let err = handle_chat_stream_chunks(
            &[thinking_chunk(json!({"thinking_blocks": [block]}))],
            &ChatStreamOptions::default(),
        )
        .unwrap_err()
        .to_string();
        let expected = format!(
            "Expected streamed thinking block field '{field}' to be a string, got {type_name}"
        );
        assert!(err.contains(&expected), "{err}");
    }
}

#[test]
fn stream_handler_ignores_explicit_null_thinking_block_type() {
    let events = events(&[thinking_chunk(json!({"thinking_blocks": [
        {"type": null, "thinking": "hidden", "signature": "SIG"}]}))]);
    assert_eq!(output(&events), &Vec::<Value>::new());
}

#[test]
fn stream_handler_preserves_redacted_thinking_blocks() {
    let events = events(&[
        thinking_chunk(json!({"reasoning_content": "summary"})),
        thinking_chunk(json!({"thinking_blocks": [{"type": "redacted_thinking",
                                                    "data": "REDACTED-BLOB"}]})),
        thinking_delta("visible", ""),
        thinking_delta("", "SIG"),
    ]);
    let item = &output(&events)[0];
    assert_eq!(
        item["provider_data"]["thinking_blocks"],
        json!([{"type": "redacted_thinking", "data": "REDACTED-BLOB"},
               {"type": "thinking", "thinking": "visible", "signature": "SIG"}])
    );
    // The redacted block contributes no reasoning_text and no signature.
    let content: Vec<&str> = item["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["text"].as_str().unwrap())
        .collect();
    assert_eq!(content, ["visible"]);
    assert_eq!(item["encrypted_content"], "SIG");
}

#[test]
fn streamed_thinking_blocks_replay_through_the_converter() {
    let events = events(&[
        thinking_chunk(json!({"reasoning_content": "summary"})),
        thinking_chunk(
            json!({"thinking_blocks": [{"type": "thinking", "thinking": "alpha",
            "signature": "", "cache_control": {"type": "ephemeral"}}]}),
        ),
        thinking_delta("", "SIG-A"),
        thinking_chunk(json!({"thinking_blocks": [{"type": "redacted_thinking",
                                                    "data": "BLOB"}]})),
        thinking_delta("beta", ""),
        thinking_delta("", "SIG-B"),
        thinking_chunk(json!({"content": "visible answer"})),
    ]);
    let messages = replay(output(&events), "anthropic/claude-4-opus", true);
    let assistants: Vec<&Value> = messages
        .iter()
        .filter(|m| m["role"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 1);
    // The complete sequence replays through the native `thinking_blocks` field, which keeps
    // redacted blocks and per-block signatures unchanged.
    assert_eq!(
        assistants[0]["thinking_blocks"],
        json!([
            {"type": "thinking", "thinking": "alpha", "signature": "SIG-A",
             "cache_control": {"type": "ephemeral"}},
            {"type": "redacted_thinking", "data": "BLOB"},
            {"type": "thinking", "thinking": "beta", "signature": "SIG-B"},
        ])
    );
    assert_eq!(assistants[0]["content"], "visible answer");
}

#[test]
fn stream_handler_adds_third_party_reasoning_text_to_summary_item() {
    let events = events(&[
        chunk(json!({"reasoning_content": "summary"})),
        chunk(json!({"reasoning": "details"})),
    ]);
    let item = &output(&events)[0];
    assert_eq!(item["type"], "reasoning");
    assert_eq!(item["summary"][0]["text"], "summary");
    assert_eq!(item["content"][0]["text"], "details");
}

// ---- refusal / text part ordering ----------------------------------------------------------

#[test]
fn stream_handler_orders_refusal_after_reasoning_and_text() {
    let events = events(&[
        chunk(json!({"reasoning_content": "summary"})),
        chunk(json!({"content": "partial"})),
        chunk(json!({"refusal": "blocked"})),
    ]);
    let message = &output(&events)[1];
    assert_eq!(message["type"], "message");
    assert_eq!(message["content"][0]["type"], "output_text");
    assert_eq!(message["content"][1]["type"], "refusal");
    assert_eq!(message["content"][0]["text"], "partial");
    assert_eq!(message["content"][1]["refusal"], "blocked");
}

#[test]
fn stream_handler_places_text_after_existing_refusal_part() {
    let events = events(&[
        chunk(json!({"refusal": "blocked"})),
        chunk(json!({"content": "partial"})),
    ]);
    let parts = of_type(&events, "response.content_part.added");
    let refusal = parts
        .iter()
        .find(|e| e["part"]["type"] == "refusal")
        .unwrap();
    assert_eq!(refusal["content_index"], 0);
    let text = parts
        .iter()
        .find(|e| e["part"]["type"] == "output_text")
        .unwrap();
    assert_eq!(text["content_index"], 1);
    // The completed content lines up with the indexes announced above.
    let message = &output(&events)[0];
    assert_eq!(message["content"][0]["type"], "refusal");
    assert_eq!(message["content"][1]["type"], "output_text");
    assert_eq!(message["content"][0]["refusal"], "blocked");
    assert_eq!(message["content"][1]["text"], "partial");
}

#[test]
fn stream_handler_announces_assistant_message_once_for_text_and_refusal() {
    let orders = [
        [json!({"refusal": "blocked"}), json!({"content": "partial"})],
        [json!({"content": "partial"}), json!({"refusal": "blocked"})],
    ];
    for deltas in orders {
        let chunks: Vec<Value> = deltas.into_iter().map(chunk).collect();
        let events = events(&chunks);
        let is_message = |e: &&Value| e["item"]["type"] == "message";
        let added: Vec<&Value> = of_type(&events, "response.output_item.added")
            .into_iter()
            .filter(is_message)
            .collect();
        let done: Vec<&Value> = of_type(&events, "response.output_item.done")
            .into_iter()
            .filter(is_message)
            .collect();
        assert_eq!(added.len(), 1);
        assert_eq!(done.len(), 1);
        assert_eq!(added[0]["output_index"], done[0]["output_index"]);
        let kinds = event_types(&events);
        let first_added = kinds
            .iter()
            .position(|t| *t == "response.output_item.added")
            .unwrap();
        let first_part = kinds
            .iter()
            .position(|t| *t == "response.content_part.added")
            .unwrap();
        assert!(first_added < first_part);
        let mut indexes: Vec<u64> = of_type(&events, "response.content_part.added")
            .iter()
            .map(|e| e["content_index"].as_u64().unwrap())
            .collect();
        indexes.sort_unstable();
        assert_eq!(indexes, [0, 1]);
        let mut part_types: Vec<&str> = of_type(&events, "response.content_part.added")
            .iter()
            .map(|e| e["part"]["type"].as_str().unwrap())
            .collect();
        part_types.sort_unstable();
        assert_eq!(part_types, ["output_text", "refusal"]);
    }
}

// ---- content filter / length at handler level --------------------------------------------

#[test]
fn handler_stream_raises_on_length_truncation() {
    let err = handle_chat_stream_chunks(
        &[finished(json!({}), "length")],
        &ChatStreamOptions {
            raise_on_length_truncation: true,
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("finish_reason='length'"), "{err}");
}

#[test]
fn handler_stream_length_is_ignored_without_internal_option() {
    let events = events(&[finished(json!({}), "length")]);
    assert!(!event_types(&events).contains(&"response.refusal.delta"));
    assert_eq!(output(&events), &Vec::<Value>::new());
}

#[test]
fn handler_stream_length_does_not_clobber_text() {
    let events = handle_chat_stream_chunks(
        &[
            chunk(json!({"content": "answer"})),
            finished(json!({}), "length"),
        ],
        &ChatStreamOptions {
            raise_on_length_truncation: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!event_types(&events).contains(&"response.refusal.delta"));
    assert_eq!(output(&events)[0]["content"][0]["text"], "answer");
}

// ---- url citations ---------------------------------------------------------------------------

#[test]
fn stream_handler_keeps_url_citations_on_the_text_delta() {
    let events = events(&[finished(
        json!({"role": "assistant", "content": "It will rain tomorrow.",
               "annotations": [weather_citation()]}),
        "stop",
    )]);
    assert_eq!(
        streamed_annotations(&events),
        vec![
            json!({"type": "url_citation", "start_index": 0, "end_index": 22,
                    "url": "https://example.com/weather", "title": "Weather"})
        ]
    );
}

#[test]
fn stream_handler_keeps_url_citations_reported_after_the_text() {
    let events = events(&[
        chunk(json!({"role": "assistant", "content": "It will rain tomorrow."})),
        finished(json!({"annotations": [weather_citation()]}), "stop"),
    ]);
    let urls: Vec<Value> = streamed_annotations(&events)
        .iter()
        .map(|a| a["url"].clone())
        .collect();
    assert_eq!(urls, [json!("https://example.com/weather")]);
}

#[test]
fn stream_handler_accumulates_url_citations_across_deltas() {
    let events = events(&[
        chunk(
            json!({"role": "assistant", "content": "It will rain tomorrow.",
                     "annotations": [weather_citation()]}),
        ),
        finished(
            json!({"annotations": [annotation("https://example.com/forecast", "Forecast")]}),
            "stop",
        ),
    ]);
    let urls: Vec<Value> = streamed_annotations(&events)
        .iter()
        .map(|a| a["url"].clone())
        .collect();
    assert_eq!(
        urls,
        [
            json!("https://example.com/weather"),
            json!("https://example.com/forecast")
        ]
    );
}

#[test]
fn stream_handler_skips_unsupported_annotation_shapes() {
    let other =
        json!({"type": "file_citation", "file_citation": {"file_id": "file-1", "index": 0}});
    let incomplete = json!({"type": "url_citation",
                            "url_citation": {"url": "https://example.com/partial"}});
    let events = events(&[finished(
        json!({"role": "assistant", "content": "It will rain tomorrow.",
               "annotations": [other, incomplete, weather_citation()]}),
        "stop",
    )]);
    let urls: Vec<Value> = streamed_annotations(&events)
        .iter()
        .map(|a| a["url"].clone())
        .collect();
    assert_eq!(urls, [json!("https://example.com/weather")]);
}

#[test]
fn stream_handler_ignores_annotations_that_are_not_a_sequence() {
    let events = events(&[finished(
        json!({"role": "assistant", "content": "It will rain tomorrow.", "annotations": 5}),
        "stop",
    )]);
    assert_eq!(streamed_annotations(&events), Vec::<Value>::new());
}

#[test]
fn stream_handler_drops_citations_reported_before_any_text() {
    let events = events(&[
        chunk(json!({"role": "assistant", "annotations": [weather_citation()]})),
        finished(json!({"content": "It will rain tomorrow."}), "stop"),
    ]);
    assert_eq!(streamed_annotations(&events), Vec::<Value>::new());
    let message = &output(&events)[0];
    assert_eq!(message["content"].as_array().unwrap().len(), 1);
    assert_eq!(message["content"][0]["text"], "It will rain tomorrow.");
}
