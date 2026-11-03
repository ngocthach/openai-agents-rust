//! Ports of `tests/test_usage.py` cases the earlier audits left as Python-only.

use openai_agents_sdk::usage::{InputTokensDetails, OutputTokensDetails, Usage};
use serde_json::json;

fn usage(requests: u64, input: u64, output: u64, cached: u64, cache_write: u64) -> Usage {
    Usage {
        requests,
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
        input_tokens_details: InputTokensDetails {
            cached_tokens: cached,
            cache_write_tokens: cache_write,
        },
        output_tokens_details: OutputTokensDetails::default(),
        ..Default::default()
    }
}

/// `test_usage_snapshot_delta_and_span_preserve_cache_write_tokens`: the usage a span carries
/// for a per-call delta keeps both cache counters. Rust takes the delta by subtracting two
/// reads of the run usage (the turn span does exactly that), so the delta is built the same way
/// here and serialized through `to_span_usage`.
#[test]
fn the_span_usage_of_a_delta_keeps_cache_write_tokens() {
    let start = usage(1, 10, 4, 3, 2);
    let end = usage(2, 30, 10, 8, 9);

    let delta = usage(
        end.requests - start.requests,
        end.input_tokens - start.input_tokens,
        end.output_tokens - start.output_tokens,
        end.input_tokens_details.cached_tokens - start.input_tokens_details.cached_tokens,
        end.input_tokens_details.cache_write_tokens - start.input_tokens_details.cache_write_tokens,
    );
    let span = delta.to_span_usage();

    assert_eq!(
        span["input_tokens_details"],
        json!({"cached_tokens": 5, "cache_write_tokens": 7})
    );
    assert_eq!(span["requests"], json!(1));
    assert_eq!(span["input_tokens"], json!(20));
}
