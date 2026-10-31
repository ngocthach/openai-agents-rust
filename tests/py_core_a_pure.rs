//! Ports of the pure-function Python tests: `test_apply_diff.py`, `test_transforms.py`,
//! `test_debug.py`, `test_handoff_prompt.py`, `test_usage.py` (non-runner half) and
//! `test_strict_schema.py`.

use openai_agents_sdk::apply_diff::{ApplyDiffMode, apply_diff};
use openai_agents_sdk::extensions::handoff_prompt::{
    RECOMMENDED_PROMPT_PREFIX, prompt_with_handoff_instructions,
};
use openai_agents_sdk::strict_schema::ensure_strict_json_schema;
use openai_agents_sdk::transform_string_function_style;
use openai_agents_sdk::usage::{InputTokensDetails, OutputTokensDetails, RequestUsage, Usage};
use serde_json::{Value, json};

fn update(input: &str, diff: &str) -> String {
    apply_diff(input, diff, ApplyDiffMode::Default).expect("diff applies")
}

// ---------------------------------------------------------------- apply_diff

#[test]
fn apply_diff_with_floating_hunk_adds_lines() {
    assert_eq!(update("", "@@\n+hello\n+world"), "hello\nworld\n");
}

#[test]
fn apply_diff_with_empty_input_and_crlf_diff_preserves_crlf() {
    assert_eq!(update("", "@@\r\n+hello\r\n+world"), "hello\r\nworld\r\n");
}

#[test]
fn apply_diff_create_mode_requires_plus_prefix() {
    assert!(apply_diff("", "plain line", ApplyDiffMode::Create).is_err());
}

#[test]
fn apply_diff_create_mode_preserves_trailing_newline() {
    let out = apply_diff("", "+hello\n+world\n+", ApplyDiffMode::Create).unwrap();
    assert_eq!(out, "hello\nworld\n");
}

#[test]
fn apply_diff_applies_contextual_replacement() {
    let out = update(
        "line1\nline2\nline3\n",
        "@@ line1\n-line2\n+updated\n line3",
    );
    assert_eq!(out, "line1\nupdated\nline3\n");
}

#[test]
fn apply_diff_applies_stacked_anchors_from_the_tool_description() {
    let input = "class BaseClass\n    def search():\n        pass\n\nclass Subclass\n    def search():\n        pass\n";
    let diff = [
        "@@ class BaseClass",
        "@@     def search():",
        "-        pass",
        "+        raise NotImplementedError()",
        "",
        "@@ class Subclass",
        "@@     def search():",
        "-        pass",
        "+        raise NotImplementedError()",
    ]
    .join("\n");
    assert_eq!(
        update(input, &diff),
        "class BaseClass\n    def search():\n        raise NotImplementedError()\n\nclass Subclass\n    def search():\n        raise NotImplementedError()\n"
    );
}

#[test]
fn apply_diff_reuses_a_prior_parent_anchor_across_stacked_hunks() {
    let input = "class Target\n    def first():\n        pass\n\n    def second():\n        pass\n";
    let diff = [
        "@@ class Target",
        "@@     def first():",
        "-        pass",
        "+        return 1",
        "@@ class Target",
        "@@     def second():",
        "-        pass",
        "+        return 2",
    ]
    .join("\n");
    assert_eq!(
        update(input, &diff),
        "class Target\n    def first():\n        return 1\n\n    def second():\n        return 2\n"
    );
}

#[test]
fn apply_diff_stacked_anchors_narrow_to_the_named_block() {
    let input = "class First\n    def target():\n        return 0\n\nclass Second\n    def helper():\n        pass\n\n    def target():\n        pass\n";
    let diff = [
        "@@ class Second",
        "@@     def target():",
        "-        pass",
        "+        return 1",
    ]
    .join("\n");
    assert_eq!(
        update(input, &diff),
        "class First\n    def target():\n        return 0\n\nclass Second\n    def helper():\n        pass\n\n    def target():\n        return 1\n"
    );
}

#[test]
fn apply_diff_single_anchor_stays_advisory_when_unmatched() {
    assert_eq!(update("a\nb\n", "@@ nope\n-b\n+B"), "a\nB\n");
}

#[test]
fn apply_diff_rejects_partially_matched_stacked_anchors() {
    let input =
        "class Target\n    def helper():\n        pass\n\n    def desired():\n        return 1\n";
    let diff = [
        "@@ class Target",
        "@@     def missing():",
        "-        pass",
        "+        return 99",
    ]
    .join("\n");
    let err = apply_diff(input, &diff, ApplyDiffMode::Default).unwrap_err();
    assert!(
        err.to_string().contains("Invalid Anchor"),
        "unexpected error: {err}"
    );
}

#[test]
fn apply_diff_rejects_stacked_anchors_when_the_first_is_missing() {
    let input = "class Wrong\n    def desired():\n        pass\n";
    let diff = [
        "@@ class Target",
        "@@     def desired():",
        "-        pass",
        "+        return 99",
    ]
    .join("\n");
    let err = apply_diff(input, &diff, ApplyDiffMode::Default).unwrap_err();
    assert!(
        err.to_string().contains("Invalid Anchor"),
        "unexpected error: {err}"
    );
}

#[test]
fn apply_diff_rejects_a_missing_anchor_followed_by_a_bare_marker() {
    let err = apply_diff("a\nb\n", "@@ missing\n@@\n-b\n+B", ApplyDiffMode::Default).unwrap_err();
    assert!(
        err.to_string().contains("Invalid Anchor"),
        "unexpected error: {err}"
    );
}

#[test]
fn apply_diff_stacked_anchors_accept_a_trailing_bare_anchor() {
    let input = "class Only\n    def run():\n        pass\n";
    let diff = "@@ class Only\n@@\n-        pass\n+        return 1";
    assert_eq!(
        update(input, diff),
        "class Only\n    def run():\n        return 1\n"
    );
}

#[test]
fn apply_diff_raises_on_context_mismatch() {
    assert!(
        apply_diff(
            "one\ntwo\n",
            "@@ -1,2 +1,2 @@\n x\n-two\n+2",
            ApplyDiffMode::Default
        )
        .is_err()
    );
}

#[test]
fn apply_diff_with_crlf_input_and_lf_diff_preserves_crlf() {
    let out = update(
        "line1\r\nline2\r\nline3\r\n",
        "@@ line1\n-line2\n+updated\n line3",
    );
    assert_eq!(out, "line1\r\nupdated\r\nline3\r\n");
}

#[test]
fn apply_diff_with_lf_input_and_crlf_diff_preserves_lf() {
    let out = update(
        "line1\nline2\nline3\n",
        "@@ line1\r\n-line2\r\n+updated\r\n line3",
    );
    assert_eq!(out, "line1\nupdated\nline3\n");
}

#[test]
fn apply_diff_with_crlf_input_and_crlf_diff_preserves_crlf() {
    let out = update(
        "line1\r\nline2\r\nline3\r\n",
        "@@ line1\r\n-line2\r\n+updated\r\n line3",
    );
    assert_eq!(out, "line1\r\nupdated\r\nline3\r\n");
}

#[test]
fn apply_diff_create_mode_preserves_crlf_newlines() {
    let out = apply_diff("", "+hello\r\n+world\r\n+", ApplyDiffMode::Create).unwrap();
    assert_eq!(out, "hello\r\nworld\r\n");
}

#[test]
fn apply_diff_eof_hunk_appends_without_a_blank_line() {
    assert_eq!(update("a\nb\n", "@@\n+c\n*** End of File"), "a\nb\nc\n");
    assert_eq!(update("a\nb", "@@\n+c\n*** End of File"), "a\nb\nc");
}

#[test]
fn apply_diff_eof_hunk_matches_the_last_context_occurrence() {
    let out = update("x\nfoo\ny\nfoo\n", " foo\n+added\n*** End of File");
    assert_eq!(out, "x\nfoo\ny\nfoo\nadded\n");
}

#[test]
fn apply_diff_eof_hunk_preserves_crlf() {
    assert_eq!(
        update("a\r\nb\r\n", "@@\n+c\n*** End of File"),
        "a\r\nb\r\nc\r\n"
    );
}

// --------------------------------------------------------------- transforms

#[test]
fn transform_string_function_style_replaces_invalid_characters() {
    assert_eq!(transform_string_function_style("My Tool"), "my_tool");
    assert_eq!(transform_string_function_style("My-Tool"), "my_tool");
}

#[test]
fn transform_string_function_style_handles_case_only_changes() {
    assert_eq!(transform_string_function_style("MyTool"), "mytool");
    assert_eq!(
        transform_string_function_style("transfer_to_Agent"),
        "transfer_to_agent"
    );
    assert_eq!(transform_string_function_style("snake_case"), "snake_case");
}

// ------------------------------------------------------------ handoff prompt

#[test]
fn prompt_with_handoff_instructions_includes_prefix() {
    let prompt = "Handle the transfer smoothly.";
    let result = prompt_with_handoff_instructions(prompt);
    assert!(result.starts_with(RECOMMENDED_PROMPT_PREFIX));
    assert!(result.ends_with(prompt));
}

// -------------------------------------------------------------------- usage

fn input_details(cached: u64, cache_write: u64) -> InputTokensDetails {
    InputTokensDetails {
        cached_tokens: cached,
        cache_write_tokens: cache_write,
    }
}

fn usage_with(
    requests: u64,
    input_tokens: u64,
    cached: u64,
    cache_write: u64,
    output_tokens: u64,
    reasoning: u64,
    total_tokens: u64,
) -> Usage {
    Usage {
        requests,
        input_tokens,
        input_tokens_details: input_details(cached, cache_write),
        output_tokens,
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: reasoning,
        },
        total_tokens,
        request_usage_entries: Vec::new(),
    }
}

#[test]
fn usage_defaults_cache_write_tokens_to_zero() {
    let usage = Usage::default();
    assert_eq!(usage.input_tokens_details.cached_tokens, 0);
    assert_eq!(usage.input_tokens_details.cache_write_tokens, 0);
}

#[test]
fn usage_add_aggregates_all_fields() {
    let mut u1 = usage_with(1, 10, 3, 5, 20, 5, 30);
    let u2 = usage_with(2, 7, 4, 6, 8, 6, 15);
    u1.add(&u2);
    assert_eq!(u1.requests, 3);
    assert_eq!(u1.input_tokens, 17);
    assert_eq!(u1.output_tokens, 28);
    assert_eq!(u1.total_tokens, 45);
    assert_eq!(u1.input_tokens_details.cached_tokens, 7);
    assert_eq!(u1.input_tokens_details.cache_write_tokens, 11);
    assert_eq!(u1.output_tokens_details.reasoning_tokens, 11);
}

#[test]
fn usage_add_aggregates_with_default_values() {
    let mut u1 = Usage::default();
    let u2 = usage_with(2, 7, 4, 0, 8, 6, 15);
    u1.add(&u2);
    assert_eq!(u1.requests, 2);
    assert_eq!(u1.input_tokens, 7);
    assert_eq!(u1.output_tokens, 8);
    assert_eq!(u1.total_tokens, 15);
    assert_eq!(u1.input_tokens_details.cached_tokens, 4);
    assert_eq!(u1.output_tokens_details.reasoning_tokens, 6);
}

#[test]
fn request_usage_creation() {
    let entry = RequestUsage {
        input_tokens: 100,
        output_tokens: 200,
        total_tokens: 300,
        input_tokens_details: input_details(10, 0),
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 20,
        },
    };
    assert_eq!(entry.input_tokens, 100);
    assert_eq!(entry.output_tokens, 200);
    assert_eq!(entry.total_tokens, 300);
    assert_eq!(entry.input_tokens_details.cached_tokens, 10);
    assert_eq!(entry.output_tokens_details.reasoning_tokens, 20);
}

#[test]
fn usage_add_preserves_single_request() {
    let mut u1 = Usage::default();
    u1.add(&usage_with(1, 100, 10, 0, 200, 20, 300));
    assert_eq!(u1.request_usage_entries.len(), 1);
    let entry = &u1.request_usage_entries[0];
    assert_eq!(entry.input_tokens, 100);
    assert_eq!(entry.output_tokens, 200);
    assert_eq!(entry.total_tokens, 300);
    assert_eq!(entry.input_tokens_details.cached_tokens, 10);
    assert_eq!(entry.output_tokens_details.reasoning_tokens, 20);
}

#[test]
fn usage_add_ignores_zero_token_requests() {
    let mut u1 = Usage::default();
    u1.add(&usage_with(1, 0, 0, 0, 0, 0, 0));
    assert!(u1.request_usage_entries.is_empty());
}

#[test]
fn usage_add_ignores_multi_request_usage() {
    let mut u1 = Usage::default();
    u1.add(&usage_with(3, 100, 10, 0, 200, 20, 300));
    assert!(u1.request_usage_entries.is_empty());
}

#[test]
fn usage_add_merges_existing_request_usage_entries() {
    let mut u1 = Usage::default();
    u1.add(&usage_with(1, 100, 10, 0, 200, 20, 300));
    u1.add(&usage_with(1, 50, 5, 0, 75, 10, 125));
    assert_eq!(u1.request_usage_entries.len(), 2);
    assert_eq!(u1.request_usage_entries[0].input_tokens, 100);
    assert_eq!(u1.request_usage_entries[0].output_tokens, 200);
    assert_eq!(u1.request_usage_entries[0].total_tokens, 300);
    assert_eq!(u1.request_usage_entries[1].input_tokens, 50);
    assert_eq!(u1.request_usage_entries[1].output_tokens, 75);
    assert_eq!(u1.request_usage_entries[1].total_tokens, 125);
}

#[test]
fn usage_add_preserves_existing_entries_when_top_level_also_set() {
    let mut u1 = Usage::default();
    let mut u2 = usage_with(1, 100, 0, 0, 50, 0, 150);
    u2.request_usage_entries = vec![RequestUsage {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        input_tokens_details: input_details(10, 0),
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 5,
        },
    }];
    u1.add(&u2);
    assert_eq!(u1.request_usage_entries.len(), 1);
    let entry = &u1.request_usage_entries[0];
    assert_eq!(entry.input_tokens_details.cached_tokens, 10);
    assert_eq!(entry.output_tokens_details.reasoning_tokens, 5);
}

#[test]
fn usage_request_usage_entries_default_empty() {
    assert!(Usage::default().request_usage_entries.is_empty());
}

#[test]
fn anthropic_cost_calculation_scenario() {
    let mut usage = Usage::default();
    usage.add(&usage_with(1, 100_000, 0, 0, 50_000, 0, 150_000));
    usage.add(&usage_with(1, 150_000, 0, 0, 75_000, 0, 225_000));
    usage.add(&usage_with(1, 80_000, 0, 0, 40_000, 0, 120_000));

    assert_eq!(usage.requests, 3);
    assert_eq!(usage.input_tokens, 330_000);
    assert_eq!(usage.output_tokens, 165_000);
    assert_eq!(usage.total_tokens, 495_000);
    assert_eq!(usage.request_usage_entries.len(), 3);
    assert_eq!(usage.request_usage_entries[0].input_tokens, 100_000);
    assert_eq!(usage.request_usage_entries[1].input_tokens, 150_000);
    assert_eq!(usage.request_usage_entries[2].input_tokens, 80_000);
    for entry in &usage.request_usage_entries {
        assert!(entry.input_tokens < 200_000);
        assert!(entry.output_tokens < 200_000);
    }
}

#[test]
fn usage_serialization_preserves_cache_write_tokens() {
    let mut usage = usage_with(1, 20, 3, 7, 5, 0, 25);
    usage.request_usage_entries = vec![RequestUsage {
        input_tokens: 20,
        output_tokens: 5,
        total_tokens: 25,
        input_tokens_details: input_details(3, 7),
        output_tokens_details: OutputTokensDetails::default(),
    }];

    let serialized = serde_json::to_value(&usage).unwrap();
    assert_eq!(
        serialized["input_tokens_details"],
        json!({"cached_tokens": 3, "cache_write_tokens": 7})
    );
    let restored: Usage = serde_json::from_value(serialized).unwrap();
    assert_eq!(restored.input_tokens_details.cache_write_tokens, 7);
    assert_eq!(
        restored.request_usage_entries[0]
            .input_tokens_details
            .cache_write_tokens,
        7
    );
}

#[test]
fn usage_deserialization_defaults_legacy_cache_write_tokens() {
    let restored: Usage = serde_json::from_value(json!({
        "requests": 1,
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
        "input_tokens_details": {"cached_tokens": 3},
        "request_usage_entries": [{
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "input_tokens_details": {"cached_tokens": 3}
        }]
    }))
    .unwrap();
    assert_eq!(restored.input_tokens_details.cached_tokens, 3);
    assert_eq!(restored.input_tokens_details.cache_write_tokens, 0);
    assert_eq!(
        restored.request_usage_entries[0]
            .input_tokens_details
            .cached_tokens,
        3
    );
    assert_eq!(
        restored.request_usage_entries[0]
            .input_tokens_details
            .cache_write_tokens,
        0
    );
}

#[test]
fn usage_from_responses_usage_reads_cache_write_tokens() {
    let usage = Usage::from_responses_usage(&json!({
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
        "output_tokens_details": {"reasoning_tokens": 1}
    }));
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.input_tokens_details.cached_tokens, 4);
    assert_eq!(usage.input_tokens_details.cache_write_tokens, 2);
    assert_eq!(usage.output_tokens_details.reasoning_tokens, 1);
}

#[test]
fn usage_from_chat_usage_maps_prompt_and_completion_details() {
    let usage = Usage::from_chat_usage(&json!({
        "prompt_tokens": 200,
        "completion_tokens": 150,
        "total_tokens": 350,
        "prompt_tokens_details": {"cached_tokens": 50},
        "completion_tokens_details": {"reasoning_tokens": 100}
    }));
    assert_eq!(usage.input_tokens, 200);
    assert_eq!(usage.output_tokens, 150);
    assert_eq!(usage.input_tokens_details.cached_tokens, 50);
    assert_eq!(usage.output_tokens_details.reasoning_tokens, 100);
}

// ------------------------------------------------------------ strict schema

fn strict(schema: Value) -> Value {
    ensure_strict_json_schema(&schema).expect("schema converts")
}

fn strict_err(schema: Value) -> String {
    ensure_strict_json_schema(&schema)
        .expect_err("schema should be rejected")
        .to_string()
}

fn nested_object_schema(depth: usize) -> Value {
    let mut node = json!({"type": "object", "properties": {}});
    for _ in 0..depth {
        node = json!({"type": "object", "properties": {"child": node}});
    }
    node
}

#[test]
fn empty_schema_has_additional_properties_false() {
    assert_eq!(strict(json!({}))["additionalProperties"], json!(false));
}

#[test]
fn empty_schema_returns_fresh_copy() {
    let first = strict(json!({}));
    let second = strict(json!({}));
    assert_eq!(first, second);
    assert_eq!(
        second,
        json!({"additionalProperties": false, "type": "object", "properties": {}, "required": []})
    );
}

#[test]
fn deeply_nested_schema_is_rejected_before_recursive_conversion() {
    // Python uses depth 1000; a 1000-deep `serde_json::Value` overflows the 2 MiB test-thread
    // stack while it is built, so 300 is used instead - still far past `MAX_SCHEMA_DEPTH`.
    let message = strict_err(nested_object_schema(300));
    assert!(
        message.contains("too deeply nested"),
        "unexpected error: {message}"
    );
}

#[test]
fn reasonably_nested_schema_remains_supported() {
    assert_eq!(
        strict(nested_object_schema(10))["additionalProperties"],
        json!(false)
    );
}

#[test]
fn deeply_chained_refs_are_rejected_before_recursive_conversion() {
    let depth = 1_000usize;
    let mut defs = serde_json::Map::new();
    for i in 0..depth {
        defs.insert(
            format!("L{i}"),
            json!({"$ref": format!("#/$defs/L{}", i + 1)}),
        );
    }
    defs.insert(format!("L{depth}"), json!({"type": "string"}));
    let schema = json!({
        "$defs": Value::Object(defs),
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/L0", "description": "value"}}
    });
    let message = strict_err(schema);
    assert!(
        message.contains("too deeply nested"),
        "unexpected error: {message}"
    );
}

#[test]
fn object_without_additional_properties() {
    let result = strict(json!({"type": "object", "properties": {"a": {"type": "string"}}}));
    assert_eq!(result["type"], json!("object"));
    assert_eq!(result["additionalProperties"], json!(false));
    assert_eq!(result["required"], json!(["a"]));
    assert_eq!(result["properties"]["a"], json!({"type": "string"}));
}

#[test]
fn typeless_root_is_normalized_to_object() {
    assert_eq!(
        strict(json!({"properties": {"a": {"type": "string"}}})),
        json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": false,
            "required": ["a"]
        })
    );
}

#[test]
fn nullable_object_root_errors() {
    let message =
        strict_err(json!({"type": ["object", "null"], "properties": {"a": {"type": "string"}}}));
    assert!(
        message.contains("root of a strict JSON schema"),
        "unexpected error: {message}"
    );
}

#[test]
fn open_map_root_errors() {
    assert!(
        ensure_strict_json_schema(&json!({"additionalProperties": {"type": "string"}})).is_err()
    );
}

#[test]
fn nested_typeless_open_map_errors() {
    assert!(
        ensure_strict_json_schema(&json!({
            "type": "object",
            "properties": {"metadata": {"additionalProperties": {"type": "string"}}}
        }))
        .is_err()
    );
}

#[test]
fn union_root_errors() {
    for keyword in ["anyOf", "oneOf"] {
        let schema = json!({
            keyword: [{"properties": {"a": {"type": "string"}}}, {"type": "null"}]
        });
        let message = strict_err(schema);
        assert!(
            message.contains("root of a strict JSON schema"),
            "unexpected error for {keyword}: {message}"
        );
    }
}

#[test]
fn nested_typeless_objects_get_additional_properties() {
    let property = strict(json!({"type": "object", "properties": {"config": {"properties": {}}}}));
    let node = &property["properties"]["config"];
    assert_eq!(node["type"], json!("object"));
    assert_eq!(node["additionalProperties"], json!(false));

    let array = strict(json!({"type": "array", "items": {"properties": {}}}));
    assert_eq!(array["items"]["type"], json!("object"));
    assert_eq!(array["items"]["additionalProperties"], json!(false));

    let any_of = strict(json!({
        "type": "object",
        "properties": {"value": {"anyOf": [{"properties": {}}, {"type": "null"}]}}
    }));
    let node = &any_of["properties"]["value"]["anyOf"][0];
    assert_eq!(node["type"], json!("object"));
    assert_eq!(node["additionalProperties"], json!(false));
}

#[test]
fn nested_nullable_object_preserves_type_union() {
    let result = strict(json!({
        "type": "object",
        "properties": {
            "config": {"type": ["object", "null"], "properties": {"key": {"type": "string"}}}
        }
    }));
    assert_eq!(
        result["properties"]["config"],
        json!({
            "type": ["object", "null"],
            "properties": {"key": {"type": "string"}},
            "additionalProperties": false,
            "required": ["key"]
        })
    );
}

#[test]
fn object_with_true_additional_properties_errors() {
    assert!(
        ensure_strict_json_schema(&json!({
            "type": "object",
            "properties": {"a": {"type": "number"}},
            "additionalProperties": true
        }))
        .is_err()
    );
}

#[test]
fn typeless_object_with_additional_properties_errors() {
    assert!(
        ensure_strict_json_schema(&json!({
            "properties": {"a": {"type": "number"}},
            "additionalProperties": true
        }))
        .is_err()
    );
}

#[test]
fn explicit_non_object_with_properties_is_not_closed() {
    let schema = json!({
        "type": "string",
        "properties": {},
        "required": [],
        "additionalProperties": true
    });
    assert_eq!(strict(schema.clone()), schema);
}

#[test]
fn object_with_empty_dict_additional_properties_errors() {
    assert!(
        ensure_strict_json_schema(&json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": {}
        }))
        .is_err()
    );
}

#[test]
fn object_with_schema_additional_properties_errors() {
    assert!(
        ensure_strict_json_schema(&json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": {"type": "string"}
        }))
        .is_err()
    );
}

#[test]
fn object_with_false_additional_properties_is_allowed() {
    let result = strict(json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": false
    }));
    assert_eq!(result["additionalProperties"], json!(false));
    assert_eq!(result["required"], json!(["a"]));
}

#[test]
fn array_items_processing_and_default_removal() {
    let result = strict(json!({"type": "array", "items": {"type": "number", "default": null}}));
    assert!(result["items"].get("default").is_none());
    assert_eq!(result["items"]["type"], json!("number"));
}

#[test]
fn any_of_processing() {
    let result = strict(json!({
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}},
                    {"type": "number", "default": null}
                ]
            }
        }
    }));
    let variants = &result["properties"]["value"]["anyOf"];
    assert_eq!(variants[0]["type"], json!("object"));
    assert_eq!(variants[0]["additionalProperties"], json!(false));
    assert_eq!(variants[0]["required"], json!(["a"]));
    assert_eq!(variants[1]["type"], json!("number"));
    assert!(variants[1].get("default").is_none());
}

#[test]
fn all_of_single_entry_merging() {
    let result = strict(json!({
        "type": "object",
        "allOf": [{"properties": {"a": {"type": "boolean"}}}]
    }));
    assert!(result.get("allOf").is_none());
    assert_eq!(result["additionalProperties"], json!(false));
    assert_eq!(result["required"], json!(["a"]));
    assert_eq!(result["properties"]["a"]["type"], json!("boolean"));
}

#[test]
fn default_removal_on_non_object() {
    let result = strict(json!({"type": "string", "default": null}));
    assert_eq!(result["type"], json!("string"));
    assert!(result.get("default").is_none());
}

#[test]
fn ref_expansion() {
    let result = strict(json!({
        "definitions": {"refObj": {"type": "string", "default": null}},
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/refObj", "description": "desc"}}
    }));
    let a = &result["properties"]["a"];
    assert_eq!(a["type"], json!("string"));
    assert_eq!(a["description"], json!("desc"));
    assert!(a.get("default").is_none());
}

#[test]
fn ref_no_expansion_when_alone() {
    assert_eq!(
        strict(json!({"$ref": "#/definitions/refObj"})),
        json!({"$ref": "#/definitions/refObj"})
    );
}

#[test]
fn invalid_ref_format_errors() {
    assert!(
        ensure_strict_json_schema(&json!({
            "type": "object",
            "properties": {"a": {"$ref": "invalid", "description": "desc"}}
        }))
        .is_err()
    );
}

#[test]
fn chained_ref_with_sibling_keys_is_resolved() {
    let result = strict(json!({
        "$defs": {"Inner": {"type": "string"}, "Outer": {"$ref": "#/$defs/Inner"}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Outer", "description": "desc"}}
    }));
    let a = &result["properties"]["a"];
    assert_eq!(a["type"], json!("string"));
    assert_eq!(a["description"], json!("desc"));
    assert!(a.get("$ref").is_none());
}

#[test]
fn ref_expansion_bomb_is_rejected() {
    let depth = 30usize;
    let mut defs = serde_json::Map::new();
    for i in 0..depth {
        defs.insert(
            format!("L{i}"),
            json!({
                "type": "object",
                "properties": {
                    "a": {"$ref": format!("#/$defs/L{}", i + 1), "title": "t"},
                    "b": {"$ref": format!("#/$defs/L{}", i + 1), "title": "t"}
                }
            }),
        );
    }
    defs.insert(format!("L{depth}"), json!({"type": "string"}));
    let schema = json!({
        "$defs": Value::Object(defs),
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/L0", "title": "t"}}
    });
    assert!(ensure_strict_json_schema(&schema).is_err());
}

#[test]
fn ref_with_annotation_sibling_is_still_expanded() {
    let result = strict(json!({
        "$defs": {
            "T": {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}
        },
        "type": "object",
        "properties": {
            "node": {
                "contentMediaType": "application/json",
                "description": "a node",
                "title": "Node",
                "$ref": "#/$defs/T"
            }
        }
    }));
    let node = &result["properties"]["node"];
    assert_eq!(node["type"], json!("object"));
    assert_eq!(node["properties"], json!({"b": {"type": "string"}}));
    assert_eq!(node["required"], json!(["b"]));
    assert_eq!(node["description"], json!("a node"));
    assert_eq!(node["title"], json!("Node"));
    assert_eq!(node["contentMediaType"], json!("application/json"));
    assert!(node.get("$ref").is_none());
}

#[test]
fn ref_with_anchor_is_still_expanded() {
    let result = strict(json!({
        "$defs": {"T": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$anchor": "value", "$ref": "#/$defs/T"}}
    }));
    assert_eq!(
        result["properties"]["value"],
        json!({"$anchor": "value", "type": "string"})
    );
}

#[test]
fn all_of_single_ref_entry_merging() {
    let result = strict(json!({
        "$defs": {
            "Inner": {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
            "Outer": {"$ref": "#/$defs/Inner"}
        },
        "type": "object",
        "allOf": [{"$ref": "#/$defs/Outer"}]
    }));
    assert!(result.get("allOf").is_none());
    assert!(result.get("$ref").is_none());
    assert_eq!(result["type"], json!("object"));
    assert_eq!(result["properties"], json!({"b": {"type": "string"}}));
    assert_eq!(result["required"], json!(["b"]));
    assert_eq!(result["additionalProperties"], json!(false));
}

#[test]
fn all_of_single_ref_entry_preserves_annotated_aliases() {
    let result = strict(json!({
        "components": {
            "schemas": {
                "Inner": {
                    "type": "object",
                    "description": "inner",
                    "properties": {"value": {"type": "string"}}
                },
                "Outer": {"$ref": "#/components/schemas/Inner", "description": "outer"}
            }
        },
        "type": "object",
        "allOf": [{"$ref": "#/components/schemas/Outer", "title": "entry"}]
    }));
    assert!(result.get("$ref").is_none());
    assert_eq!(result["description"], json!("outer"));
    assert_eq!(result["title"], json!("entry"));
    assert_eq!(result["properties"], json!({"value": {"type": "string"}}));
    assert_eq!(result["required"], json!(["value"]));
    assert_eq!(result["additionalProperties"], json!(false));
}

#[test]
fn all_of_single_circular_ref_is_rejected() {
    let schema = json!({
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
        "type": "object",
        "allOf": [{"$ref": "#/$defs/A"}]
    });
    assert!(ensure_strict_json_schema(&schema).is_err());
}

#[test]
fn all_of_single_ref_entry_decodes_json_pointer_tokens() {
    for (name, token) in [("a/b", "a~1b"), ("a~b", "a~0b"), ("a~1b", "a~01b")] {
        let schema = json!({
            "$defs": {name: {"type": "object", "properties": {"value": {"type": "string"}}}},
            "type": "object",
            "allOf": [{"$ref": format!("#/$defs/{token}")}]
        });
        let result = strict(schema);
        assert_eq!(
            result["properties"],
            json!({"value": {"type": "string"}}),
            "pointer token {token} did not resolve to {name}"
        );
        assert_eq!(result["required"], json!(["value"]));
        assert_eq!(result["additionalProperties"], json!(false));
    }
}

#[test]
fn all_of_single_entry_cannot_overwrite_strict_object() {
    for additional in [json!(true), json!({})] {
        let schema = json!({
            "properties": {"a": {"type": "string"}},
            "allOf": [{"additionalProperties": additional}]
        });
        assert!(
            ensure_strict_json_schema(&schema).is_err(),
            "additionalProperties {additional} should be rejected"
        );
    }
}

#[test]
fn ref_with_incompatible_sibling_is_rejected() {
    let message = strict_err(json!({
        "$defs": {
            "T": {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}
        },
        "type": "object",
        "properties": {
            "node": {
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "$ref": "#/$defs/T"
            }
        }
    }));
    assert!(
        message.contains("incompatible sibling"),
        "unexpected error: {message}"
    );
}

#[test]
fn ref_with_incompatible_type_sibling_is_rejected() {
    let message = strict_err(json!({
        "$defs": {
            "T": {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}
        },
        "type": "object",
        "properties": {"node": {"type": "string", "$ref": "#/$defs/T"}}
    }));
    assert!(
        message.contains("incompatible sibling"),
        "unexpected error: {message}"
    );
}

#[test]
fn ref_with_single_all_of_sibling_is_rejected() {
    let message = strict_err(json!({
        "$defs": {"A": {"type": "string"}, "B": {"type": "integer"}},
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/A", "allOf": [{"$ref": "#/$defs/B"}]}}
    }));
    assert!(
        message.contains("incompatible sibling"),
        "unexpected error: {message}"
    );
}

#[test]
fn ref_with_validation_sibling_is_rejected() {
    let message = strict_err(json!({
        "$defs": {"T": {"type": "string"}},
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/T", "minLength": 1}}
    }));
    assert!(
        message.contains("incompatible sibling"),
        "unexpected error: {message}"
    );
}

#[test]
fn ref_with_interacting_object_sibling_is_rejected() {
    let message = strict_err(json!({
        "$defs": {
            "T": {
                "type": "object",
                "properties": {"b": {"type": "string"}},
                "required": ["b"],
                "additionalProperties": false
            }
        },
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/T", "additionalProperties": false}}
    }));
    assert!(
        message.contains("incompatible sibling"),
        "unexpected error: {message}"
    );
}

#[test]
fn ref_with_schema_metadata_is_still_expanded() {
    let result = strict(json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"T": {"type": "object", "properties": {"value": {"type": "string"}}}},
        "$ref": "#/$defs/T"
    }));
    assert_eq!(result["type"], json!("object"));
    assert_eq!(result["properties"], json!({"value": {"type": "string"}}));
    assert_eq!(result["required"], json!(["value"]));
    assert_eq!(result["additionalProperties"], json!(false));
    assert!(result.get("$ref").is_none());
}

#[test]
fn root_ref_with_id_is_still_expanded() {
    let result = strict(json!({
        "$id": "https://example.test/root",
        "$defs": {"T": {"type": "object", "properties": {"value": {"type": "string"}}}},
        "$ref": "#/$defs/T"
    }));
    assert_eq!(result["$id"], json!("https://example.test/root"));
    assert_eq!(result["type"], json!("object"));
    assert_eq!(result["properties"], json!({"value": {"type": "string"}}));
    assert_eq!(result["required"], json!(["value"]));
    assert_eq!(result["additionalProperties"], json!(false));
    assert!(result.get("$ref").is_none());
}

#[test]
fn ref_allows_unrelated_id_definition_name() {
    let result = strict(json!({
        "$defs": {"$id": {"type": "integer"}, "T": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/T", "description": "value"}}
    }));
    assert_eq!(
        result["properties"]["value"],
        json!({"type": "string", "description": "value"})
    );
}

// -------------------------------------------------------------------- debug

// Python's `test_debug.py` monkeypatches `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` /
// `OPENAI_AGENTS_DONT_LOG_TOOL_DATA`. Mutating the process environment is `unsafe` in Rust 2024
// and races with the other tests in this binary, so the equivalent override API is exercised
// instead. The tests run serially because the flags are process-global.

#[test]
fn dont_log_model_and_tool_data_default_to_true_and_follow_overrides() {
    use openai_agents_sdk::debug::{
        dont_log_model_data, dont_log_tool_data, reset_dont_log_model_data,
        reset_dont_log_tool_data, set_dont_log_model_data, set_dont_log_tool_data,
    };

    reset_dont_log_model_data();
    reset_dont_log_tool_data();
    assert!(dont_log_model_data());
    assert!(dont_log_tool_data());

    set_dont_log_model_data(false);
    assert!(!dont_log_model_data());
    set_dont_log_model_data(true);
    assert!(dont_log_model_data());

    set_dont_log_tool_data(false);
    assert!(!dont_log_tool_data());
    set_dont_log_tool_data(true);
    assert!(dont_log_tool_data());

    reset_dont_log_model_data();
    reset_dont_log_tool_data();
    assert!(dont_log_model_data());
    assert!(dont_log_tool_data());
}
