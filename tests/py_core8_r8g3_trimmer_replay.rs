//! `test_canonical_structured_output_replays_through_chat_completions` of
//! `tests/extensions/test_tool_output_trimmer.py`: a structured tool output produced by the SDK's
//! own `tool_call_output_item` stays bounded after trimming and replays through the Chat
//! Completions converter as a plain tool message.

use openai_agents_sdk::extensions::tool_output_trimmer::ToolOutputTrimmer;
use openai_agents_sdk::items::{FunctionCall, InputItem, ItemHelpers};
use openai_agents_sdk::models::openai_chatcompletions::items_to_messages;
use openai_agents_sdk::run_config::ModelInputData;
use openai_agents_sdk::tool::{ToolOutput, ToolOutputPart};
use serde_json::{Value, json};

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn assistant(text: &str) -> Value {
    json!({"role": "assistant", "content": text})
}

#[test]
fn a_canonical_structured_output_replays_through_chat_completions() {
    let call: FunctionCall = serde_json::from_value(json!({
        "id": "fc1",
        "call_id": "c1",
        "name": "render",
        "arguments": "{}",
        "type": "function_call",
    }))
    .unwrap();
    let image_url = format!("image-payload-{}", "I".repeat(1000));
    let file_data = format!("file-payload-{}", "F".repeat(1000));
    let output_item = ItemHelpers::tool_call_output_item(
        &call,
        &ToolOutput::Parts(vec![
            ToolOutputPart::Text {
                text: format!("useful-text-preview-{}", "T".repeat(1000)),
            },
            ToolOutputPart::Image {
                image_url: Some(image_url.clone()),
                file_id: Some("image-file-id".into()),
                detail: Some("high".into()),
            },
            ToolOutputPart::File {
                file_data: Some(file_data.clone()),
                file_url: Some("https://example.com/report.pdf".into()),
                file_id: Some("file-id".into()),
                filename: Some("report.pdf".into()),
            },
        ]),
    );
    let produced = output_item["output"].as_array().unwrap();
    assert_eq!(
        produced[1],
        json!({
            "type": "input_image",
            "image_url": image_url,
            "file_id": "image-file-id",
            "detail": "high",
        })
    );
    assert_eq!(
        produced[2],
        json!({
            "type": "input_file",
            "file_data": file_data,
            "file_url": "https://example.com/report.pdf",
            "file_id": "file-id",
            "filename": "report.pdf",
        })
    );

    for max_output_chars in [
        1usize,
        "[Trimmed]".len(),
        "[Trimmed: input_image]".len(),
        70,
        200,
    ] {
        let items: Vec<InputItem> = vec![
            user("q1"),
            call.to_value(),
            output_item.clone(),
            assistant("a1"),
            user("q2"),
            assistant("a2"),
            user("q3"),
            assistant("a3"),
        ]
        .into_iter()
        .map(InputItem)
        .collect();
        let original = items.clone();
        let trimmer = ToolOutputTrimmer::new()
            .with_max_output_chars(max_output_chars)
            .with_preview_chars(20);
        let result = trimmer.apply(ModelInputData {
            input: items.clone(),
            instructions: Some("You are helpful.".into()),
        });
        let trimmed = result.input[2].0["output"]
            .as_str()
            .expect("the trimmed output is a string")
            .to_string();

        assert!(trimmed.chars().count() <= max_output_chars, "{trimmed:?}");
        assert!(!trimmed.contains("image-payload"), "{trimmed:?}");
        assert!(!trimmed.contains("file-payload"), "{trimmed:?}");
        assert_eq!(items, original);
        let again = trimmer.apply(ModelInputData {
            input: result.input.clone(),
            instructions: result.instructions.clone(),
        });
        assert_eq!(again.input, result.input);

        let messages = items_to_messages(&[result.input[2].clone()], None).unwrap();
        assert_eq!(
            messages,
            vec![json!({"role": "tool", "tool_call_id": "c1", "content": trimmed})],
            "budget {max_output_chars}"
        );
    }
}
