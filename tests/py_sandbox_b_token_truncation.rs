//! Port of `tests/sandbox/test_token_truncation.py`. The Python module is
//! `agents.sandbox.util.token_truncation`; in Rust the same helpers live in
//! `openai_agents_sdk::sandbox::util`.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::util::{
    APPROX_BYTES_PER_TOKEN, TruncationMode, TruncationPolicy, approx_bytes_for_tokens,
    approx_token_count, approx_tokens_from_byte_count, formatted_truncate_text,
    formatted_truncate_text_with_token_count, split_budget, split_string, truncate_text,
    truncate_with_byte_estimate, truncate_with_token_budget,
};

fn long_content() -> String {
    (0..20)
        .map(|index| format!("line {index}: {}", "value ".repeat(8).trim_end()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Python: `test_truncation_policy_clamps_negative_limits_and_converts_budgets`. Rust's limit is
/// a `usize`, so a negative limit cannot be expressed; zero is the clamped value Python reaches.
#[test]
fn truncation_policy_converts_budgets() {
    let byte_policy = TruncationPolicy::bytes(0);
    let token_policy = TruncationPolicy::tokens(0);

    assert_eq!(byte_policy.limit, 0);
    assert_eq!(byte_policy.token_budget(), 0);
    assert_eq!(byte_policy.byte_budget(), 0);
    assert_eq!(token_policy.limit, 0);
    assert_eq!(token_policy.token_budget(), 0);
    assert_eq!(token_policy.byte_budget(), 0);

    assert_eq!(TruncationPolicy::bytes(8).mode, TruncationMode::Bytes);
    assert_eq!(TruncationPolicy::tokens(8).mode, TruncationMode::Tokens);
    assert_eq!(TruncationPolicy::bytes(8).token_budget(), 2);
    assert_eq!(TruncationPolicy::tokens(2).byte_budget(), 8);
    assert_eq!(APPROX_BYTES_PER_TOKEN, 4);
}

/// Python: `test_formatted_truncate_text_returns_short_content_unchanged`.
#[test]
fn formatted_truncate_text_returns_short_content_unchanged() {
    assert_eq!(
        formatted_truncate_text("short", TruncationPolicy::bytes(20)),
        "short"
    );
}

/// Python: `test_formatted_truncate_text_adds_line_count_when_truncated`.
#[test]
fn formatted_truncate_text_adds_line_count_when_truncated() {
    let result = formatted_truncate_text("alpha\nbeta\ngamma", TruncationPolicy::bytes(8));

    assert!(result.starts_with("Total output lines: 3\n\n"), "{result}");
    assert!(result.contains("chars truncated"), "{result}");
}

/// Python: `test_formatted_truncate_text_keeps_token_metadata_within_budget`.
#[test]
fn formatted_truncate_text_keeps_token_metadata_within_budget() {
    let content = long_content();

    let result = formatted_truncate_text(&content, TruncationPolicy::tokens(32));

    assert!(result.starts_with("Total output lines: 20\n\n"), "{result}");
    assert!(result.contains("tokens truncated"), "{result}");
    assert!(approx_token_count(&result) <= 32, "{result}");
}

/// Python: `test_formatted_truncate_text_with_token_count_handles_none_and_short_content`.
#[test]
fn formatted_truncate_text_with_token_count_handles_none_and_short_content() {
    assert_eq!(
        formatted_truncate_text_with_token_count("short", None),
        ("short".to_string(), None)
    );
    assert_eq!(
        formatted_truncate_text_with_token_count("short", Some(10)),
        ("short".to_string(), None)
    );
}

/// Python: `test_formatted_truncate_text_with_token_count_reports_original_count`.
#[test]
fn formatted_truncate_text_with_token_count_reports_original_count() {
    let (result, original_token_count) =
        formatted_truncate_text_with_token_count("abcdefghi", Some(1));

    assert!(approx_token_count(&result) <= 1, "{result}");
    assert_eq!(original_token_count, Some(approx_token_count("abcdefghi")));
}

/// Python: `test_formatted_truncate_text_with_token_count_keeps_metadata_within_budget`.
#[test]
fn formatted_truncate_text_with_token_count_keeps_metadata_within_budget() {
    let content = long_content();

    let (result, original_token_count) =
        formatted_truncate_text_with_token_count(&content, Some(32));

    assert!(result.starts_with("Total output lines: 20\n\n"), "{result}");
    assert!(result.contains("tokens truncated"), "{result}");
    assert!(approx_token_count(&result) <= 32, "{result}");
    assert_eq!(original_token_count, Some(approx_token_count(&content)));
}

/// Python: `test_truncate_text_dispatches_byte_and_token_modes`.
#[test]
fn truncate_text_dispatches_byte_and_token_modes() {
    assert!(
        truncate_text("abcdef", TruncationPolicy::bytes(4)).starts_with('a'),
        "{}",
        truncate_text("abcdef", TruncationPolicy::bytes(4))
    );
    let token_result = truncate_text(
        &"abcdefghijklmnopqrstuvwxyz".repeat(2),
        TruncationPolicy::tokens(8),
    );
    assert!(token_result.contains("tokens truncated"), "{token_result}");
    assert!(approx_token_count(&token_result) <= 8, "{token_result}");
}

/// Python: `test_truncate_with_token_budget_handles_empty_and_short_content`.
#[test]
fn truncate_with_token_budget_handles_empty_and_short_content() {
    assert_eq!(
        truncate_with_token_budget("", TruncationPolicy::tokens(1)),
        (String::new(), None)
    );
    assert_eq!(
        truncate_with_token_budget("abc", TruncationPolicy::tokens(1)),
        ("abc".to_string(), None)
    );
}

/// Python: `test_truncate_with_token_budget_includes_marker_within_budget`.
#[test]
fn truncate_with_token_budget_includes_marker_within_budget() {
    let content = "abcdefghijklmnopqrstuvwxyz".repeat(2);
    let (result, original_token_count) =
        truncate_with_token_budget(&content, TruncationPolicy::tokens(8));

    assert!(result.contains("tokens truncated"), "{result}");
    assert!(approx_token_count(&result) <= 8, "{result}");
    assert_eq!(original_token_count, Some(approx_token_count(&content)));
}

/// Python: `test_formatted_truncate_text_with_zero_token_budget_returns_empty_payload`.
#[test]
fn formatted_truncate_text_with_zero_token_budget_returns_empty_payload() {
    let (result, original_token_count) =
        formatted_truncate_text_with_token_count("content", Some(0));

    assert_eq!(result, "");
    assert_eq!(original_token_count, Some(approx_token_count("content")));
}

/// Python: `test_truncate_with_byte_estimate_handles_empty_zero_and_short_content`.
#[test]
fn truncate_with_byte_estimate_handles_empty_zero_and_short_content() {
    assert_eq!(
        truncate_with_byte_estimate("", TruncationPolicy::bytes(0)),
        ""
    );
    let zero_budget = truncate_with_byte_estimate("abc", TruncationPolicy::bytes(0));
    assert!(zero_budget.contains("chars truncated"), "{zero_budget}");
    assert_eq!(
        truncate_with_byte_estimate("abc", TruncationPolicy::bytes(10)),
        "abc"
    );
}

/// Python: `test_split_string_preserves_utf8_boundaries`.
#[test]
fn split_string_preserves_utf8_boundaries() {
    let (removed_chars, prefix, suffix) = split_string("aあbいc", 2, 4);

    assert_eq!(prefix, "a");
    assert_eq!(suffix, "いc");
    assert_eq!(removed_chars, 2);
}

/// Python: `test_split_string_handles_empty_content`.
#[test]
fn split_string_handles_empty_content() {
    assert_eq!(split_string("", 10, 10), (0, "", ""));
}

/// Python: `test_formatting_and_estimate_helpers`.
#[test]
fn formatting_and_estimate_helpers() {
    let byte_policy = TruncationPolicy::bytes(8);
    let token_policy = TruncationPolicy::tokens(2);

    assert!(byte_policy.marker(3).contains("chars truncated"));
    assert!(token_policy.marker(2).contains("tokens truncated"));
    assert_eq!(split_budget(5), (2, 3));
    assert_eq!(byte_policy.removed_units(10, 4), 4);
    assert_eq!(token_policy.removed_units(9, 4), 3);
    assert_eq!(approx_token_count("abcde"), 2);
    assert_eq!(approx_bytes_for_tokens(0), 0);
    assert_eq!(approx_tokens_from_byte_count(0), 0);
    assert_eq!(approx_tokens_from_byte_count(5), 2);
}
