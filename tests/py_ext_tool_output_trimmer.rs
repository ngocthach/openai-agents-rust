//! Port of `tests/extensions/test_tool_output_trimmer.py`.

use openai_agents_sdk::extensions::tool_output_trimmer::ToolOutputTrimmer;
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::run_config::ModelInputData;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn assistant(text: &str) -> Value {
    json!({"role": "assistant", "content": text})
}

fn func_call(call_id: &str, name: &str) -> Value {
    json!({"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"})
}

fn namespaced_call(call_id: &str, name: &str, namespace: &str) -> Value {
    json!({
        "type": "function_call", "call_id": call_id, "name": name,
        "arguments": "{}", "namespace": namespace
    })
}

fn func_output(call_id: &str, output: Value) -> Value {
    json!({"type": "function_call_output", "call_id": call_id, "output": output})
}

fn items(values: Vec<Value>) -> Vec<InputItem> {
    values.into_iter().map(InputItem).collect()
}

fn data(values: Vec<Value>) -> ModelInputData {
    ModelInputData {
        input: items(values),
        instructions: Some("You are helpful.".into()),
    }
}

/// The `output` field of the item at `idx`.
fn output(result: &ModelInputData, idx: usize) -> Value {
    result.input[idx].0["output"].clone()
}

fn output_str(result: &ModelInputData, idx: usize) -> String {
    output(result, idx).as_str().unwrap().to_string()
}

fn repeat(unit: &str, times: usize) -> String {
    unit.repeat(times)
}

/// The eight-item transcript the Python tests reuse: one old tool turn, then two newer ones.
fn old_turn_transcript(call: Value, out: Value) -> Vec<Value> {
    vec![
        user("q1"),
        call,
        out,
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    ]
}

// ---------------------------------------------------------------------------------------------
// TestDefaults
// ---------------------------------------------------------------------------------------------

#[test]
fn default_values() {
    let trimmer = ToolOutputTrimmer::new();
    assert_eq!(trimmer.recent_turns, 2);
    assert_eq!(trimmer.max_output_chars, 500);
    assert_eq!(trimmer.preview_chars, 200);
    assert_eq!(trimmer.trimmable_tools, None);
}

#[test]
fn trimmable_tools_become_a_set() {
    let trimmer = ToolOutputTrimmer::new().with_trimmable_tools(["a", "b", "a"]);
    let tools = trimmer.trimmable_tools.unwrap();
    assert_eq!(tools.len(), 2);
    assert!(tools.contains("a"));
    assert!(tools.contains("b"));
}

#[test]
fn trimmable_tools_from_a_list() {
    let trimmer = ToolOutputTrimmer::new().with_trimmable_tools(vec!["search", "run_code"]);
    let tools = trimmer.trimmable_tools.unwrap();
    assert!(tools.contains("search"));
    assert!(tools.contains("run_code"));
}

/// Python accepts a bare string and must not treat it as an iterable of characters; the Rust
/// builder takes an iterator of names, so the equivalent check is that one entry matches the
/// whole tool name.
#[test]
fn a_single_allowlist_entry_matches_a_whole_tool_name() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new()
        .with_trimmable_tools(["search"])
        .apply(data(vec![
            user("q1"),
            func_call("c1", "search"),
            func_output("c1", json!(large)),
            func_call("c2", "s"),
            func_output("c2", json!(large)),
            assistant("a1"),
            user("q2"),
            assistant("a2"),
            user("q3"),
            assistant("a3"),
        ]));
    assert!(output_str(&result, 2).contains("[Trimmed:"));
    assert_eq!(output(&result, 4), json!(large));
}

// ---------------------------------------------------------------------------------------------
// TestValidation
// ---------------------------------------------------------------------------------------------

#[test]
fn recent_turns_zero_is_rejected() {
    let err = ToolOutputTrimmer::new()
        .with_recent_turns(0)
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("recent_turns must be >= 1"));
}

#[test]
fn max_output_chars_zero_is_rejected() {
    let err = ToolOutputTrimmer::new()
        .with_max_output_chars(0)
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("max_output_chars must be >= 1"));
}

#[test]
fn preview_chars_zero_is_allowed() {
    let trimmer = ToolOutputTrimmer::new().with_preview_chars(0);
    assert_eq!(trimmer.preview_chars, 0);
    assert!(trimmer.validate().is_ok());
}

// ---------------------------------------------------------------------------------------------
// TestRecentBoundary
// ---------------------------------------------------------------------------------------------

#[test]
fn boundary_of_empty_items() {
    assert_eq!(ToolOutputTrimmer::new().find_recent_boundary(&[]), 0);
}

#[test]
fn boundary_of_a_single_user_message() {
    let input = items(vec![user("hello")]);
    assert_eq!(ToolOutputTrimmer::new().find_recent_boundary(&input), 0);
}

#[test]
fn two_user_messages_put_the_boundary_at_the_first() {
    let input = items(vec![
        user("q1"),
        assistant("a1"),
        user("q2"),
        assistant("a2"),
    ]);
    assert_eq!(
        ToolOutputTrimmer::new()
            .with_recent_turns(2)
            .find_recent_boundary(&input),
        0
    );
}

#[test]
fn three_user_messages_put_the_boundary_at_the_second() {
    let input = items(vec![
        user("q1"),
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    ]);
    assert_eq!(
        ToolOutputTrimmer::new()
            .with_recent_turns(2)
            .find_recent_boundary(&input),
        2
    );
}

#[test]
fn a_custom_recent_turns_moves_the_boundary_back() {
    let input = items(vec![
        user("q1"),
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
        user("q4"),
        assistant("a4"),
    ]);
    // q4 at 6 (count=1), q3 at 4 (count=2), q2 at 2 (count=3) -> boundary = 2.
    assert_eq!(
        ToolOutputTrimmer::new()
            .with_recent_turns(3)
            .find_recent_boundary(&input),
        2
    );
}

// ---------------------------------------------------------------------------------------------
// TestTrimming
// ---------------------------------------------------------------------------------------------

#[test]
fn empty_input_is_returned_unchanged() {
    let result = ToolOutputTrimmer::new().apply(data(vec![]));
    assert!(result.input.is_empty());
}

#[test]
fn nothing_is_trimmed_when_every_turn_is_recent() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new().apply(data(vec![
        user("q"),
        func_call("c1", "search"),
        func_output("c1", json!(large)),
        assistant("a"),
    ]));
    assert_eq!(output(&result, 2), json!(large));
}

#[test]
fn a_large_old_output_is_trimmed() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new().apply(data(old_turn_transcript(
        func_call("c1", "search"),
        func_output("c1", json!(large)),
    )));
    let trimmed = output_str(&result, 2);
    assert!(trimmed.contains("[Trimmed:"));
    assert!(trimmed.contains("search"));
    assert!(trimmed.contains("1000 chars"));
    assert!(trimmed.len() < large.len());
}

#[test]
fn a_small_old_output_is_preserved() {
    let small = repeat("x", 100);
    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(500)
        .apply(data(old_turn_transcript(
            func_call("c1", "search"),
            func_output("c1", json!(small)),
        )));
    assert_eq!(output(&result, 2), json!(small));
}

#[test]
fn a_structured_output_previews_text_and_drops_opaque_parts() {
    let caption = "Revenue chart: Q3 up 12% YoY, driven by EMEA.";
    let image = json!({
        "type": "input_image",
        "image_url": format!("data:image/png;base64,{}", repeat("Q", 3000)),
        "detail": "auto",
    });
    let text = json!({"type": "input_text", "text": caption});

    for opaque_first in [true, false] {
        let parts = if opaque_first {
            json!([image, text])
        } else {
            json!([text, image])
        };
        let trimmer = ToolOutputTrimmer::new()
            .with_max_output_chars(500)
            .with_preview_chars(200);
        let result = trimmer.apply(data(old_turn_transcript(
            func_call("c1", "plot"),
            func_output("c1", parts),
        )));
        let trimmed = output_str(&result, 2);

        assert!(trimmed.contains(caption), "opaque_first={opaque_first}");
        assert!(!trimmed.contains("base64,"));
        assert!(trimmed.contains(&format!("preview {}", caption.chars().count())));
        assert!(trimmed.contains("dropped 1 input_image"));
        assert!(!trimmed.ends_with("..."));

        // Trimming is idempotent: a second pass leaves the already-trimmed item alone.
        let repeated = trimmer.apply(ModelInputData {
            input: result.input.clone(),
            instructions: result.instructions.clone(),
        });
        assert_eq!(repeated.input, result.input);
    }
}

#[test]
fn a_structured_output_truncates_long_text_with_the_exact_preview_length() {
    let text = repeat("abcdefghij", 40);
    let trimmer = ToolOutputTrimmer::new()
        .with_max_output_chars(100)
        .with_preview_chars(40);
    let result = trimmer.apply(data(old_turn_transcript(
        func_call("c1", "search"),
        func_output("c1", json!([{"type": "input_text", "text": text}])),
    )));
    let trimmed = output_str(&result, 2);

    assert!(trimmed.chars().count() <= 100);
    assert!(trimmed.contains("payload 400"));
    assert!(trimmed.contains("preview 40"));
    assert!(trimmed.ends_with(&format!("{}...", repeat("abcdefghij", 4))));

    let repeated = trimmer.apply(ModelInputData {
        input: result.input.clone(),
        instructions: result.instructions.clone(),
    });
    assert_eq!(repeated.input, result.input);
}

#[test]
fn a_tight_structured_budget_keeps_the_text_first() {
    let text = repeat("abcdefghijklmnopqrstuvwxyz", 10);
    let trimmer = ToolOutputTrimmer::new()
        .with_max_output_chars(40)
        .with_preview_chars(100);
    let result = trimmer.apply(data(old_turn_transcript(
        func_call("c1", "search"),
        func_output("c1", json!([{"type": "input_text", "text": text}])),
    )));
    let trimmed = output_str(&result, 2);

    assert!(trimmed.chars().count() <= 40);
    assert!(trimmed.starts_with("[Trimmed]\n"));
    assert!(trimmed.contains(&text[..20]));

    let repeated = trimmer.apply(ModelInputData {
        input: result.input.clone(),
        instructions: result.instructions.clone(),
    });
    assert_eq!(repeated.input, result.input);
}

#[test]
fn a_structured_output_without_text_names_the_dropped_parts() {
    let parts = json!([
        {"type": "input_image", "image_url": format!("data:image/png;base64,{}", repeat("Q", 1000))},
        {"type": "input_file", "file_data": repeat("R", 1000), "filename": "report.pdf"},
    ]);
    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(500)
        .with_preview_chars(200)
        .apply(data(old_turn_transcript(
            func_call("c1", "render"),
            func_output("c1", parts),
        )));
    let trimmed = output_str(&result, 2);

    assert!(!trimmed.contains("base64,"));
    assert!(!trimmed.contains("report.pdf"));
    assert!(trimmed.contains("dropped 1 input_file, 1 input_image"));
    assert!(!trimmed.contains("char preview"));
    assert!(!trimmed.ends_with("..."));
}

#[test]
fn a_mixed_structured_output_keeps_text_over_dropped_part_details() {
    let text = "useful-text-preview-more";
    let parts = json!([
        {"type": "input_text", "text": text},
        {"type": "input_image", "image_url": format!("image-payload-{}", repeat("Q", 1000))},
        {"type": "input_file", "file_data": format!("file-payload-{}", repeat("R", 1000))},
    ]);
    let trimmer = ToolOutputTrimmer::new()
        .with_max_output_chars(70)
        .with_preview_chars(20);
    let result = trimmer.apply(data(old_turn_transcript(
        func_call("c1", "render"),
        func_output("c1", parts),
    )));
    let trimmed = output_str(&result, 2);

    assert!(trimmed.chars().count() <= 70);
    assert!(trimmed.contains(&text[..20]));
    assert!(!trimmed.contains("image-payload"));
    assert!(!trimmed.contains("file-payload"));

    let repeated = trimmer.apply(ModelInputData {
        input: result.input.clone(),
        instructions: result.instructions.clone(),
    });
    assert_eq!(repeated.input, result.input);
}

#[test]
fn an_opaque_structured_output_respects_a_tight_budget() {
    let cases = [
        (
            json!({"type": "input_image", "image_url": "image-payload-12345"}),
            "image-payload",
        ),
        (
            json!({"type": "input_file", "file_data": "file-payload-12345"}),
            "file-payload",
        ),
    ];
    for (part, fragment) in cases {
        let trimmer = ToolOutputTrimmer::new()
            .with_max_output_chars(10)
            .with_preview_chars(0);
        let result = trimmer.apply(data(old_turn_transcript(
            func_call("c1", "render"),
            func_output("c1", json!([part])),
        )));
        let trimmed = output_str(&result, 2);
        assert!(trimmed.chars().count() <= 10, "{fragment}");
        assert!(!trimmed.contains(fragment));

        let repeated = trimmer.apply(ModelInputData {
            input: result.input.clone(),
            instructions: result.instructions.clone(),
        });
        assert_eq!(repeated.input, result.input);
    }
}

#[test]
fn an_opaque_structured_output_uses_the_exact_type_header_when_it_fits() {
    let part = json!({
        "type": "input_image", "image_url": format!("image-payload-{}", repeat("Q", 1000))
    });
    let expected = "[Trimmed: input_image]";
    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(expected.len())
        .with_preview_chars(0)
        .apply(data(old_turn_transcript(
            func_call("c1", "render"),
            func_output("c1", json!([part])),
        )));
    assert_eq!(output(&result, 2), json!(expected));
}

/// A canonical SDK-produced structured output stays bounded and replayable at any budget.
#[test]
fn a_canonical_structured_output_stays_bounded_at_every_budget() {
    let produced = json!([
        {"type": "input_text", "text": format!("useful-text-preview-{}", repeat("T", 1000))},
        {
            "type": "input_image",
            "image_url": format!("image-payload-{}", repeat("I", 1000)),
            "file_id": "image-file-id",
            "detail": "high",
        },
        {
            "type": "input_file",
            "file_data": format!("file-payload-{}", repeat("F", 1000)),
            "file_url": "https://example.com/report.pdf",
            "file_id": "file-id",
            "filename": "report.pdf",
        },
    ]);

    for max_output_chars in [
        1usize,
        "[Trimmed]".len(),
        "[Trimmed: input_image]".len(),
        70,
        200,
    ] {
        let original = old_turn_transcript(
            func_call("c1", "render"),
            func_output("c1", produced.clone()),
        );
        let trimmer = ToolOutputTrimmer::new()
            .with_max_output_chars(max_output_chars)
            .with_preview_chars(20);
        let input = data(original.clone());
        let result = trimmer.apply(input);
        let trimmed = output_str(&result, 2);

        assert!(
            trimmed.chars().count() <= max_output_chars,
            "budget {max_output_chars}: {trimmed:?}"
        );
        assert!(!trimmed.contains("image-payload"));
        assert!(!trimmed.contains("file-payload"));

        let repeated = trimmer.apply(ModelInputData {
            input: result.input.clone(),
            instructions: result.instructions.clone(),
        });
        assert_eq!(repeated.input, result.input);
    }
}

#[test]
fn the_structured_threshold_counts_payload_characters_only() {
    let parts = json!([{"type": "input_text", "text": "short"}]);
    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(5)
        .with_preview_chars(2)
        .apply(data(old_turn_transcript(
            func_call("c1", "search"),
            func_output("c1", parts.clone()),
        )));
    assert_eq!(output(&result, 2), parts);
}

#[test]
fn a_non_canonical_structured_output_is_preserved() {
    let cases = vec![
        json!([{"type": "output_text", "text": repeat("x", 1000)}]),
        json!([{"type": "input_text", "text": repeat("x", 1000), "metadata": "unsupported"}]),
        json!([{"type": "input_image", "image_url": repeat("x", 1000), "detail": "invalid"}]),
        json!(["not a content part"]),
    ];
    for parts in cases {
        let result = ToolOutputTrimmer::new()
            .with_max_output_chars(100)
            .with_preview_chars(40)
            .apply(data(old_turn_transcript(
                func_call("c1", "search"),
                func_output("c1", parts.clone()),
            )));
        assert_eq!(output(&result, 2), parts);
    }
}

// ---------------------------------------------------------------------------------------------
// allowlists
// ---------------------------------------------------------------------------------------------

#[test]
fn only_allowlisted_tools_are_trimmed() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new()
        .with_trimmable_tools(["search"])
        .apply(data(vec![
            user("q1"),
            func_call("c1", "search"),
            func_output("c1", json!(large)),
            func_call("c2", "resolve_entity"),
            func_output("c2", json!(large)),
            assistant("a1"),
            user("q2"),
            assistant("a2"),
            user("q3"),
            assistant("a3"),
        ]));
    assert!(output_str(&result, 2).contains("[Trimmed:"));
    assert_eq!(output(&result, 4), json!(large));
}

#[test]
fn a_qualified_allowlist_entry_matches_a_namespaced_tool() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new()
        .with_trimmable_tools(["billing.lookup_account"])
        .apply(data(old_turn_transcript(
            namespaced_call("c1", "lookup_account", "billing"),
            func_output("c1", json!(large)),
        )));
    let trimmed = output_str(&result, 2);
    assert!(trimmed.contains("[Trimmed:"));
    assert!(trimmed.contains("billing.lookup_account"));
}

#[test]
fn a_bare_allowlist_entry_still_matches_a_namespaced_tool() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new()
        .with_trimmable_tools(["lookup_account"])
        .apply(data(old_turn_transcript(
            namespaced_call("c1", "lookup_account", "billing"),
            func_output("c1", json!(large)),
        )));
    let trimmed = output_str(&result, 2);
    assert!(trimmed.contains("[Trimmed:"));
    assert!(trimmed.contains("billing.lookup_account"));
}

#[test]
fn a_synthetic_same_name_namespace_uses_the_bare_display_name() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new()
        .with_trimmable_tools(["get_weather"])
        .apply(data(old_turn_transcript(
            namespaced_call("c1", "get_weather", "get_weather"),
            func_output("c1", json!(large)),
        )));
    let trimmed = output_str(&result, 2);
    assert!(trimmed.contains("[Trimmed:"));
    assert!(!trimmed.contains("get_weather.get_weather"));
    assert!(trimmed.contains("get_weather"));
}

#[test]
fn every_tool_is_eligible_without_an_allowlist() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new().apply(data(old_turn_transcript(
        func_call("c1", "any_tool"),
        func_output("c1", json!(large)),
    )));
    assert!(output_str(&result, 2).contains("[Trimmed:"));
}

// ---------------------------------------------------------------------------------------------
// tool_search_output
// ---------------------------------------------------------------------------------------------

fn tool_search_transcript(output_item: Value) -> Vec<Value> {
    vec![
        user("q1"),
        json!({"type": "tool_search_call", "call_id": "ts1", "arguments": {"query": "profile"}}),
        output_item,
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    ]
}

#[test]
fn tool_search_output_tool_definitions_are_structurally_trimmed() {
    let verbose_schema = json!({
        "type": "object",
        "description": repeat("schema ", 200),
        "properties": {
            "customer_id": {
                "type": "string",
                "description": repeat("customer id ", 200),
                "default": "cust_123",
            }
        },
        "required": ["customer_id"],
    });
    let tools = json!([{
        "type": "function",
        "name": "lookup_account",
        "description": repeat("tool description ", 200),
        "parameters": verbose_schema,
    }]);
    let original_len = serde_json::to_string(&tools).unwrap().len();

    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(400)
        .with_preview_chars(60)
        .apply(data(tool_search_transcript(json!({
            "type": "tool_search_output", "call_id": "ts1", "tools": tools
        }))));

    let item = &result.input[2].0;
    assert_eq!(item["type"], json!("tool_search_output"));
    let trimmed_tools = &item["tools"];
    assert_eq!(trimmed_tools[0]["name"], json!("lookup_account"));
    assert!(trimmed_tools[0]["parameters"].get("description").is_none());
    assert_eq!(
        trimmed_tools[0]["parameters"]["properties"]["customer_id"]["default"],
        json!("cust_123")
    );
    assert!(serde_json::to_string(trimmed_tools).unwrap().len() < original_len);
}

#[test]
fn parameters_named_like_schema_keywords_survive() {
    let parameters = json!({
        "type": "object",
        "description": repeat("schema prose ", 200),
        "properties": {
            "description": {"type": "string"},
            "title": {"type": "string"},
            "$comment": {"type": "string"},
            "examples": {"type": "string"},
            "query": {"type": "string"},
        },
        "required": ["description", "title", "$comment", "examples", "query"],
        "additionalProperties": false,
    });

    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(400)
        .with_preview_chars(60)
        .apply(data(tool_search_transcript(json!({
            "type": "tool_search_output", "call_id": "ts1",
            "tools": [{
                "type": "function", "name": "create_ticket",
                "description": repeat("tool description ", 200),
                "parameters": parameters,
            }],
        }))));

    let trimmed = &result.input[2].0["tools"][0]["parameters"];
    let mut names: Vec<&str> = trimmed["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["$comment", "description", "examples", "query", "title"]
    );
    // Every required name is still declared, so the schema stays satisfiable.
    for required in trimmed["required"].as_array().unwrap() {
        assert!(
            trimmed["properties"]
                .get(required.as_str().unwrap())
                .is_some()
        );
    }
    // Schema-level prose is still removed.
    assert!(trimmed.get("description").is_none());
}

#[test]
fn definition_names_and_instance_data_survive_schema_trimming() {
    let parameters = json!({
        "type": "object",
        "description": repeat("schema prose ", 200),
        "$defs": {
            "description": {"type": "object", "properties": {"text": {"type": "string"}}},
            "Priority": {"enum": ["low", "high"]},
        },
        "definitions": {"title": {"type": "string"}},
        "patternProperties": {"title": {"type": "string"}},
        "dependentSchemas": {"title": {"required": ["note"]}},
        "dependentRequired": {"title": ["note"]},
        "dependencies": {
            "description": ["note"],
            "title": {"type": "string", "description": repeat("dependency prose ", 200)},
        },
        "properties": {
            "note": {"$ref": "#/$defs/description"},
            "prio": {"$ref": "#/$defs/Priority"},
            "opts": {
                "type": "object",
                "description": repeat("options ", 200),
                "default": {"title": "Untitled", "description": "auto", "retries": 3},
            },
            "mode": {"const": {"title": "A", "kind": "fast"}},
            "choice": {"enum": [{"title": "A", "id": 1}, {"title": "B", "id": 2}]},
        },
        "x-tool-metadata": {
            "description": "application data",
            "nested": {"title": "must survive"},
        },
        "required": ["note"],
    });

    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(400)
        .with_preview_chars(60)
        .apply(data(tool_search_transcript(json!({
            "type": "tool_search_output", "call_id": "ts1",
            "tools": [{
                "type": "function", "name": "make_report",
                "description": repeat("tool description ", 200),
                "parameters": parameters,
            }],
        }))));

    let trimmed = &result.input[2].0["tools"][0]["parameters"];

    let mut defs: Vec<&str> = trimmed["$defs"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    defs.sort_unstable();
    assert_eq!(defs, ["Priority", "description"]);
    // Every `$ref` still resolves.
    for name in ["note", "prio"] {
        let target = trimmed["properties"][name]["$ref"]
            .as_str()
            .unwrap()
            .strip_prefix("#/$defs/")
            .unwrap();
        assert!(trimmed["$defs"].get(target).is_some(), "{name}");
    }

    for (key, expected) in [
        ("definitions", vec!["title"]),
        ("patternProperties", vec!["title"]),
        ("dependentSchemas", vec!["title"]),
    ] {
        let keys: Vec<&str> = trimmed[key]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, expected, "{key}");
    }
    assert_eq!(trimmed["dependentRequired"], json!({"title": ["note"]}));
    let mut deps: Vec<&str> = trimmed["dependencies"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    deps.sort_unstable();
    assert_eq!(deps, ["description", "title"]);
    assert!(
        trimmed["dependencies"]["title"]
            .get("description")
            .is_none()
    );

    // Instance data is preserved verbatim.
    assert_eq!(
        trimmed["properties"]["opts"]["default"],
        json!({"title": "Untitled", "description": "auto", "retries": 3})
    );
    assert_eq!(
        trimmed["properties"]["mode"]["const"],
        json!({"title": "A", "kind": "fast"})
    );
    assert_eq!(
        trimmed["properties"]["choice"]["enum"],
        json!([{"title": "A", "id": 1}, {"title": "B", "id": 2}])
    );
    assert_eq!(
        trimmed["x-tool-metadata"],
        json!({"description": "application data", "nested": {"title": "must survive"}})
    );

    assert!(trimmed.get("description").is_none());
    assert!(trimmed["properties"]["opts"].get("description").is_none());
}

#[test]
fn prose_inside_genuine_subschema_keywords_is_trimmed() {
    let parameters = json!({
        "type": "object",
        "description": repeat("schema prose ", 200),
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string", "description": repeat("a tag ", 200)},
            },
            "bag": {
                "type": "object",
                "propertyNames": {"pattern": "^x", "description": repeat("a key ", 200)},
            },
        },
        "allOf": [{"type": "object", "description": repeat("combined schema ", 200)}],
    });

    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(400)
        .with_preview_chars(60)
        .apply(data(tool_search_transcript(json!({
            "type": "tool_search_output", "call_id": "ts1",
            "tools": [{
                "type": "function", "name": "tag_it",
                "description": repeat("tool description ", 200),
                "parameters": parameters,
            }],
        }))));

    let trimmed = &result.input[2].0["tools"][0]["parameters"];
    assert_eq!(
        trimmed["properties"]["tags"]["items"],
        json!({"type": "string"})
    );
    assert_eq!(
        trimmed["properties"]["bag"]["propertyNames"],
        json!({"pattern": "^x"})
    );
    assert_eq!(trimmed["allOf"], json!([{"type": "object"}]));
}

#[test]
fn legacy_tool_search_output_results_are_trimmed() {
    let large = repeat("x", 2000);
    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(400)
        .with_preview_chars(80)
        .apply(data(tool_search_transcript(json!({
            "type": "tool_search_output", "call_id": "ts1",
            "results": [{"text": large}],
        }))));

    let item = &result.input[2].0;
    assert_eq!(item["type"], json!("tool_search_output"));
    assert!(
        item["results"][0]["text"]
            .as_str()
            .unwrap()
            .contains("[Trimmed: tool_search output")
    );
}

// ---------------------------------------------------------------------------------------------
// passthrough and sliding window
// ---------------------------------------------------------------------------------------------

#[test]
fn a_large_recent_output_is_preserved() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new().apply(data(vec![
        user("q1"),
        assistant("a1"),
        user("q2"),
        func_call("c1", "search"),
        func_output("c1", json!(large)),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    ]));
    assert_eq!(output(&result, 4), json!(large));
}

#[test]
fn the_original_items_are_not_mutated() {
    let large = repeat("x", 1000);
    let original = old_turn_transcript(
        func_call("c1", "search"),
        func_output("c1", json!(large.clone())),
    );
    let input = data(original.clone());
    let snapshot = input.input.clone();
    let result = ToolOutputTrimmer::new().apply(input);
    assert!(output_str(&result, 2).contains("[Trimmed:"));
    assert_eq!(snapshot, items(original));
}

#[test]
fn instructions_pass_through_unchanged() {
    let result = ToolOutputTrimmer::new().apply(ModelInputData {
        input: items(vec![user("hi")]),
        instructions: Some("Custom prompt".into()),
    });
    assert_eq!(result.instructions.as_deref(), Some("Custom prompt"));
}

#[test]
fn several_old_outputs_are_all_trimmed() {
    let large1 = repeat("a", 1000);
    let large2 = repeat("b", 2000);
    let result = ToolOutputTrimmer::new().apply(data(vec![
        user("q1"),
        func_call("c1", "search"),
        func_output("c1", json!(large1)),
        func_call("c2", "execute"),
        func_output("c2", json!(large2)),
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    ]));
    assert!(output_str(&result, 2).contains("[Trimmed:"));
    assert!(output_str(&result, 4).contains("[Trimmed:"));
    assert!(output_str(&result, 2).contains("search"));
    assert!(output_str(&result, 4).contains("execute"));
}

#[test]
fn the_preview_length_follows_preview_chars() {
    let large = repeat("abcdefghij", 100);
    let result = ToolOutputTrimmer::new()
        .with_preview_chars(50)
        .apply(data(old_turn_transcript(
            func_call("c1", "search"),
            func_output("c1", json!(large)),
        )));
    assert!(output_str(&result, 2).contains(&repeat("abcdefghij", 5)));
}

#[test]
fn user_and_assistant_messages_are_never_modified() {
    let transcript = vec![
        user("important"),
        assistant(&repeat("detailed ", 100)),
        user("follow up"),
        assistant("another"),
        user("final"),
        assistant("done"),
    ];
    let result = ToolOutputTrimmer::new().apply(data(transcript.clone()));
    assert_eq!(result.input, items(transcript));
}

#[test]
fn turn_three_trims_turn_one() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new().apply(data(vec![
        user("q1"),
        func_call("c1", "search"),
        func_output("c1", json!(large)),
        assistant("a1"),
        user("q2"),
        func_call("c2", "search"),
        func_output("c2", json!(large)),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    ]));
    assert!(output_str(&result, 2).contains("[Trimmed:"));
    assert_eq!(output(&result, 6), json!(large));
}

#[test]
fn turn_four_trims_turns_one_and_two() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new().apply(data(vec![
        user("q1"),
        func_call("c1", "s"),
        func_output("c1", json!(large)),
        assistant("a1"),
        user("q2"),
        func_call("c2", "s"),
        func_output("c2", json!(large)),
        assistant("a2"),
        user("q3"),
        func_call("c3", "s"),
        func_output("c3", json!(large)),
        assistant("a3"),
        user("q4"),
        assistant("a4"),
    ]));
    assert!(output_str(&result, 2).contains("[Trimmed:"));
    assert!(output_str(&result, 6).contains("[Trimmed:"));
    assert_eq!(output(&result, 10), json!(large));
}

// ---------------------------------------------------------------------------------------------
// TestEdgeCases
// ---------------------------------------------------------------------------------------------

#[test]
fn trimming_is_skipped_when_the_summary_would_be_longer() {
    let borderline = repeat("x", 501);
    let result = ToolOutputTrimmer::new()
        .with_max_output_chars(500)
        .with_preview_chars(490)
        .apply(data(old_turn_transcript(
            func_call("c1", "search"),
            func_output("c1", json!(borderline)),
        )));
    assert_eq!(output(&result, 2), json!(borderline));
}

#[test]
fn an_unresolved_tool_name_falls_back_to_unknown_tool() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new().apply(data(vec![
        user("q1"),
        func_output("orphan_id", json!(large)),
        assistant("a1"),
        user("q2"),
        assistant("a2"),
        user("q3"),
        assistant("a3"),
    ]));
    let trimmed = output_str(&result, 1);
    assert!(trimmed.contains("unknown_tool"));
    assert!(trimmed.contains("[Trimmed:"));
}

#[test]
fn an_unresolved_tool_is_skipped_when_an_allowlist_is_set() {
    let large = repeat("x", 1000);
    let result = ToolOutputTrimmer::new()
        .with_trimmable_tools(["search"])
        .apply(data(vec![
            user("q1"),
            func_output("orphan_id", json!(large)),
            assistant("a1"),
            user("q2"),
            assistant("a2"),
            user("q3"),
            assistant("a3"),
        ]));
    assert_eq!(output(&result, 1), json!(large));
}
