//! Ports of the `BackendSpanExporter` tests in `tests/test_trace_processor.py`: HTTP status
//! handling, retries and their logging, plus the OpenAI payload sanitizer.
//!
//! The HTTP tests post to a local mock ingest server. Retry waits are shortened with millisecond
//! backoffs and asserted through request counts rather than by patching the clock.

#[path = "py_core3_tracing_support/mod.rs"]
mod support;

use std::net::TcpListener;
use std::time::{Duration, Instant};

use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::tracing::processors::{
    BackendSpanExporter, ExportItem, TracingExporter, sanitize_for_openai,
};
use openai_agents_sdk::tracing::{SpanData, Trace};
use serde_json::{Value, json};
use support::{
    Ingest, NoApiKey, Reply, capture_logs, clear_logs, lock, logged_text, new_span, new_trace,
};

const MAX_FIELD_BYTES: usize = 100_000;
const SUFFIX: &str = "... [truncated]";

/// A finished span payload, as the processor would hand it to the exporter.
fn span_item(data: SpanData) -> ExportItem {
    let trace: Trace = new_trace("exporter");
    let span = new_span(&trace, data);
    ExportItem {
        payload: span.export().expect("span payload"),
        api_key: None,
        object: None,
    }
}

/// An exporter pointed at `endpoint` with millisecond backoffs and a fixed key.
fn exporter(endpoint: &str, max_retries: u32) -> BackendSpanExporter {
    BackendSpanExporter::new()
        .with_endpoint(endpoint)
        .with_api_key("test_key")
        .with_max_retries(max_retries)
        .with_base_delay(Duration::from_millis(1))
        .with_max_delay(Duration::from_millis(2))
}

fn json_size(value: &Value) -> usize {
    value.to_string().len()
}

// ---------------------------------------------------------------------------------------------
// posting and status handling
// ---------------------------------------------------------------------------------------------

#[test]
fn backend_span_exporter_no_items() {
    let _serial = lock();
    let ingest = Ingest::start(vec![Reply::new(200)]);
    exporter(&ingest.endpoint, 3).export(Vec::new());
    assert_eq!(ingest.count(), 0, "no request for an empty batch");
}

#[test]
fn backend_span_exporter_no_api_key() {
    let _serial = lock();
    let _key = NoApiKey::new();
    capture_logs();
    clear_logs();
    let ingest = Ingest::start(vec![Reply::new(200)]);
    let keyless = BackendSpanExporter::new().with_endpoint(ingest.endpoint.clone());
    keyless.export(vec![span_item(SpanData::Custom {
        name: "x".into(),
        data: json!({}),
    })]);
    assert_eq!(ingest.count(), 0, "without a key nothing is sent");
    assert!(
        logged_text().contains("OPENAI_API_KEY is not set"),
        "{}",
        logged_text()
    );
}

#[test]
fn backend_span_exporter_2xx_success() {
    let _serial = lock();
    let ingest = Ingest::start(vec![Reply::new(200)]);
    let item = |name: &str| {
        span_item(SpanData::Custom {
            name: name.into(),
            data: json!({}),
        })
    };
    exporter(&ingest.endpoint, 3).export(vec![item("a"), item("b")]);
    assert_eq!(ingest.count(), 1, "one request carries the whole batch");
    assert_eq!(ingest.data()[0].len(), 2);
}

#[test]
fn backend_span_exporter_4xx_client_error() {
    // Python parametrizes this over the redaction flags. A redacted run must neither log nor read
    // the response body.
    for redacted in [true, false] {
        let _serial = lock();
        let _key = NoApiKey::new();
        capture_logs();
        clear_logs();
        set_dont_log_model_data(redacted);
        set_dont_log_tool_data(redacted);
        let ingest = Ingest::start(vec![Reply::with_body(400, "SECRET_TRACE_RESPONSE_BODY")]);
        exporter(&ingest.endpoint, 3).export(vec![span_item(SpanData::Custom {
            name: "x".into(),
            data: json!({}),
        })]);
        reset_dont_log_model_data();
        reset_dont_log_tool_data();

        assert_eq!(
            ingest.count(),
            1,
            "a 4xx is not retried (redacted={redacted})"
        );
        let text = logged_text();
        assert_eq!(
            !text.contains("SECRET_TRACE_RESPONSE_BODY"),
            redacted,
            "redacted={redacted}: {text}"
        );
        if redacted {
            assert!(text.contains("Response data is redacted"), "{text}");
        }
    }
}

#[test]
fn backend_span_exporter_retries_transient_client_errors() {
    for status in [408, 409, 429] {
        let _serial = lock();
        let _key = NoApiKey::new();
        let ingest = Ingest::start(vec![Reply::new(status)]);
        exporter(&ingest.endpoint, 3).export(vec![span_item(SpanData::Custom {
            name: "x".into(),
            data: json!({}),
        })]);
        assert_eq!(
            ingest.count(),
            3,
            "status {status} is retried up to max_retries"
        );
    }
}

#[test]
fn backend_span_exporter_rate_limit_obeys_x_should_retry_false() {
    let _serial = lock();
    let _key = NoApiKey::new();
    capture_logs();
    clear_logs();
    let ingest = Ingest::start(vec![Reply::with_headers(
        429,
        "x-should-retry: false\r\nretry-after: 1\r\n",
    )]);
    let started = Instant::now();
    exporter(&ingest.endpoint, 3).export(vec![span_item(SpanData::Custom {
        name: "x".into(),
        data: json!({}),
    })]);
    assert_eq!(ingest.count(), 1);
    assert!(
        started.elapsed() < Duration::from_millis(900),
        "no backoff was taken"
    );
    assert!(
        logged_text().contains("Tracing client error 429"),
        "{}",
        logged_text()
    );
}

#[test]
fn backend_span_exporter_rate_limit_recovers_after_retry() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(429), Reply::new(200)]);
    exporter(&ingest.endpoint, 3).export(vec![span_item(SpanData::Custom {
        name: "x".into(),
        data: json!({}),
    })]);
    assert_eq!(ingest.count(), 2);
}

#[test]
fn backend_span_exporter_5xx_retry() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(500)]);
    exporter(&ingest.endpoint, 3).export(vec![span_item(SpanData::Custom {
        name: "x".into(),
        data: json!({}),
    })]);
    assert_eq!(
        ingest.count(),
        3,
        "retried up to max_retries requests in total"
    );
}

#[test]
fn backend_span_exporter_5xx_obeys_x_should_retry_false() {
    let _serial = lock();
    let _key = NoApiKey::new();
    capture_logs();
    clear_logs();
    let ingest = Ingest::start(vec![Reply::with_headers(503, "x-should-retry: false\r\n")]);
    exporter(&ingest.endpoint, 3).export(vec![span_item(SpanData::Custom {
        name: "x".into(),
        data: json!({}),
    })]);
    assert_eq!(ingest.count(), 1);
    assert!(
        logged_text().contains("server forbade retry for 503"),
        "{}",
        logged_text()
    );
}

#[test]
fn backend_span_exporter_deadline_stops_during_5xx_retry_backoff() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(504)]);
    let exporter = BackendSpanExporter::new()
        .with_endpoint(ingest.endpoint.clone())
        .with_api_key("test_key")
        .with_max_retries(3)
        .with_base_delay(Duration::from_secs(1))
        .with_max_delay(Duration::from_secs(1));
    let started = Instant::now();
    exporter.export_with_deadline(
        vec![span_item(SpanData::Custom {
            name: "x".into(),
            data: json!({}),
        })],
        Some(Instant::now() + Duration::from_millis(10)),
    );
    assert_eq!(
        ingest.count(),
        1,
        "the deadline ends the export before the retry"
    );
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn backend_span_exporter_request_error() {
    // A connection that is refused is a transport error: retried, then given up on.
    let _serial = lock();
    let _key = NoApiKey::new();
    capture_logs();
    clear_logs();
    let closed = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let endpoint = format!("http://{closed}/v1/traces/ingest");
    exporter(&endpoint, 2).export(vec![span_item(SpanData::Custom {
        name: "x".into(),
        data: json!({}),
    })]);
    let text = logged_text();
    assert_eq!(text.matches("Tracing: request failed").count(), 2, "{text}");
    assert!(text.contains("max retries reached"), "{text}");
}

#[test]
fn backend_span_exporter_keeps_valid_custom_span_data_when_repairing() {
    // Python also covers int keys here, which JSON values cannot hold; string keys exercise the
    // identifiers and the custom payload on both the OpenAI and a custom endpoint.
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(200)]);
    let trace = new_trace("t");
    let span = new_span(
        &trace,
        SpanData::Custom {
            name: "lookup".into(),
            data: json!({"status_counts": {"200": 3}}),
        },
    );
    let payload = span.export().unwrap();
    let custom = exporter(&ingest.endpoint, 1);
    custom.export(vec![ExportItem {
        payload: payload.clone(),
        api_key: None,
        object: None,
    }]);
    let sent = ingest.items();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["id"], payload["id"]);
    assert_eq!(sent[0]["trace_id"], payload["trace_id"]);
    assert_eq!(
        sent[0]["span_data"],
        json!({"type": "custom", "name": "lookup", "data": {"status_counts": {"200": 3}}})
    );

    let openai = sanitize_for_openai(payload.clone());
    assert_eq!(openai["id"], payload["id"]);
    assert_eq!(openai["parent_id"], payload["parent_id"]);
    assert_eq!(openai["span_data"], payload["span_data"]);
}

// ---------------------------------------------------------------------------------------------
// OpenAI payload sanitizer (applied only on the OpenAI endpoint)
// ---------------------------------------------------------------------------------------------

#[test]
fn backend_span_exporter_sanitizes_generation_usage_for_openai_tracing() {
    let original = json!({
        "object": "trace.span",
        "span_data": {
            "type": "generation",
            "usage": {
                "requests": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 1},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        },
    });
    let sent = sanitize_for_openai(original.clone());
    let usage = &sent["span_data"]["usage"];
    assert!(usage.get("requests").is_none());
    assert!(usage.get("total_tokens").is_none());
    assert!(usage.get("input_tokens_details").is_none());
    assert!(usage.get("output_tokens_details").is_none());
    assert_eq!(usage["input_tokens"], 10);
    assert_eq!(usage["output_tokens"], 5);
    assert_eq!(
        usage["details"],
        json!({
            "requests": 1,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens_details": {"reasoning_tokens": 2},
        })
    );
    // The original payload is not mutated.
    assert_eq!(original["span_data"]["usage"]["total_tokens"], 15);
}

#[test]
fn backend_span_exporter_truncates_large_input_for_openai_tracing() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "generation", "input": "x".repeat(MAX_FIELD_BYTES + 5_000)},
    });
    let sent = sanitize_for_openai(payload.clone());
    let input = sent["span_data"]["input"].as_str().expect("string input");
    assert!(input.ends_with(SUFFIX));
    assert!(json_size(&sent["span_data"]["input"]) <= MAX_FIELD_BYTES);
    assert_ne!(payload["span_data"]["input"], sent["span_data"]["input"]);
}

#[test]
fn backend_span_exporter_truncates_large_structured_input_without_stringifying() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {
            "type": "generation",
            "input": {"blob": "x".repeat(MAX_FIELD_BYTES + 5_000)},
        },
    });
    let sent = sanitize_for_openai(payload);
    let input = &sent["span_data"]["input"];
    assert!(input.is_object(), "{input}");
    assert!(input["blob"].as_str().unwrap().ends_with(SUFFIX));
    assert!(json_size(input) <= MAX_FIELD_BYTES);
}

#[test]
fn backend_span_exporter_keeps_generation_usage_for_custom_endpoint() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(200)]);
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "generation", "usage": {"requests": 1, "input_tokens": 10, "output_tokens": 5}},
    });
    exporter(&ingest.endpoint, 1).export(vec![ExportItem {
        payload,
        api_key: None,
        object: None,
    }]);
    let usage = &ingest.items()[0]["span_data"]["usage"];
    assert_eq!(usage["requests"], 1);
    assert_eq!(usage["input_tokens"], 10);
    assert_eq!(usage["output_tokens"], 5);
}

#[test]
fn backend_span_exporter_drops_non_generation_usage_for_openai_endpoint() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "function", "usage": {"requests": 1}},
    });
    assert!(
        sanitize_for_openai(payload)["span_data"]
            .get("usage")
            .is_none()
    );
}

#[test]
fn backend_span_exporter_keeps_non_generation_usage_for_custom_endpoint() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(200)]);
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "function", "usage": {"requests": 1}},
    });
    exporter(&ingest.endpoint, 1).export(vec![ExportItem {
        payload,
        api_key: None,
        object: None,
    }]);
    assert_eq!(
        ingest.items()[0]["span_data"]["usage"],
        json!({"requests": 1})
    );
}

#[test]
fn backend_span_exporter_keeps_large_input_for_custom_endpoint() {
    let _serial = lock();
    let _key = NoApiKey::new();
    let ingest = Ingest::start(vec![Reply::new(200)]);
    let large = "x".repeat(MAX_FIELD_BYTES + 5_000);
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "generation", "input": large.clone()},
    });
    exporter(&ingest.endpoint, 1).export(vec![ExportItem {
        payload,
        api_key: None,
        object: None,
    }]);
    assert_eq!(
        ingest.items()[0]["span_data"]["input"],
        Value::String(large)
    );
}

#[test]
fn sanitize_for_openai_tracing_api_keeps_allowed_generation_usage() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "generation", "usage": {"input_tokens": 1, "output_tokens": 2}},
    });
    assert_eq!(sanitize_for_openai(payload.clone()), payload);
}

#[test]
fn sanitize_for_openai_tracing_api_moves_unsupported_generation_usage_to_details() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {
            "type": "generation",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
                "details": {"provider": "litellm"},
            },
        },
    });
    let sanitized = sanitize_for_openai(payload);
    assert_eq!(
        sanitized["span_data"]["usage"],
        json!({
            "input_tokens": 1,
            "output_tokens": 2,
            "details": {
                "provider": "litellm",
                "total_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        })
    );
}

#[test]
fn sanitize_for_openai_tracing_api_drops_non_dict_generation_usage_details() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "generation", "usage": {"input_tokens": 1, "output_tokens": 2, "details": "invalid"}},
    });
    assert_eq!(
        sanitize_for_openai(payload)["span_data"]["usage"],
        json!({"input_tokens": 1, "output_tokens": 2})
    );
}

#[test]
fn sanitize_for_openai_tracing_api_drops_generation_usage_missing_required_tokens() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {
            "type": "generation",
            "usage": {
                "input_tokens": 1,
                "total_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    });
    assert_eq!(
        sanitize_for_openai(payload)["span_data"],
        json!({"type": "generation"})
    );
}

#[test]
fn sanitize_for_openai_tracing_api_rejects_boolean_token_counts() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {
            "type": "generation",
            "usage": {
                "input_tokens": true,
                "output_tokens": false,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    });
    assert_eq!(
        sanitize_for_openai(payload)["span_data"],
        json!({"type": "generation"})
    );
}

#[test]
fn sanitize_for_openai_tracing_api_skips_non_dict_generation_usage() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "generation", "usage": null},
    });
    assert_eq!(sanitize_for_openai(payload.clone()), payload);
}

#[test]
fn sanitize_for_openai_tracing_api_keeps_small_input_without_mutation() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {
            "type": "generation",
            "input": "short input",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
    });
    assert_eq!(sanitize_for_openai(payload.clone()), payload);
}

#[test]
fn sanitize_for_openai_tracing_api_truncates_oversized_output() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "function", "output": "x".repeat(MAX_FIELD_BYTES + 5_000)},
    });
    let sanitized = sanitize_for_openai(payload.clone());
    let output = sanitized["span_data"]["output"].as_str().unwrap();
    assert!(output.ends_with(SUFFIX));
    assert!(json_size(&sanitized["span_data"]["output"]) <= MAX_FIELD_BYTES);
    assert_ne!(
        payload["span_data"]["output"],
        sanitized["span_data"]["output"]
    );
}

#[test]
fn sanitize_for_openai_tracing_api_preserves_generation_input_list_shape() {
    let payload = json!({
        "object": "trace.span",
        "span_data": {
            "type": "generation",
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": "x".repeat(MAX_FIELD_BYTES + 5_000), "format": "wav"},
                }],
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    });
    let sanitized = sanitize_for_openai(payload);
    let input = sanitized["span_data"]["input"]
        .as_array()
        .expect("list kept");
    assert_eq!(input[0]["role"], "user");
    assert!(json_size(&sanitized["span_data"]["input"]) <= MAX_FIELD_BYTES);
}

#[test]
fn sanitize_for_openai_tracing_api_handles_none_content_under_tight_budget() {
    let output: Vec<Value> = (0..8)
        .map(|_| json!({"role": "assistant", "content": null, "name": "a".repeat(25_000), "tool_calls": []}))
        .collect();
    let payload = json!({
        "object": "trace.span",
        "span_data": {"type": "generation", "output": output.clone(), "usage": {"input_tokens": 1, "output_tokens": 1}},
    });
    let sanitized = sanitize_for_openai(payload);
    let sanitized_output = sanitized["span_data"]["output"]
        .as_array()
        .expect("list kept");
    assert_ne!(sanitized_output, &output);
    assert!(json_size(&sanitized["span_data"]["output"]) <= MAX_FIELD_BYTES);
    assert!(sanitized_output.iter().any(|item| *item == json!({})));
}

#[test]
fn sanitize_for_openai_tracing_api_bounds_wide_mapping_work() {
    // Python also counts the size calls made by the sanitizer. The observable contract is the
    // result: the leading keys are dropped, the kept ones are the trailing keys in insertion
    // order, and the value stays under the limit.
    for width in [1_000usize, 2_000] {
        let mut mapping = serde_json::Map::new();
        for index in 0..width {
            mapping.insert(format!("{index}:\"\\雪{}", "k".repeat(120)), json!("value"));
        }
        let keys: Vec<String> = mapping.keys().cloned().collect();
        let payload = json!({
            "object": "trace.span",
            "span_data": {"type": "generation", "input": [Value::Object(mapping.clone())]},
        });
        let sanitized = sanitize_for_openai(payload);
        let sent = sanitized["span_data"]["input"][0]
            .as_object()
            .expect("mapping kept");
        assert!(
            !sent.is_empty() && sent.len() < width,
            "kept {}",
            sent.len()
        );
        let kept: Vec<String> = sent.keys().cloned().collect();
        assert_eq!(
            kept,
            keys[keys.len() - kept.len()..],
            "trailing keys survive in order"
        );
        assert!(
            sent.values().all(|v| v == ""),
            "values are cut to empty strings"
        );
        assert!(json_size(&sanitized["span_data"]["input"]) <= MAX_FIELD_BYTES);
        assert_eq!(mapping.len(), width);
    }
}
