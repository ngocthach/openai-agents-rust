//! Port of `test_backend_span_exporter_bounds_multipart_normalization_work` from the Python SDK's
//! `tests/test_trace_processor.py`.
//!
//! A chat run with a very wide multipart user message produces a generation span whose input is
//! large. Exporting it must send a bounded payload that keeps the message structure, must not
//! change the span, and must not delay the spans queued after it.
//!
//! Python counts calls to its private `_value_json_size_bytes` through a monkeypatch. Rust has no
//! such hook, so the work bound is checked as wall-clock time (generous, to stay stable) and the
//! observable contract (payload size, structure, non-mutation, batching) is checked exactly.

#[path = "py_core4_tracerunner_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::models::{OpenAIChatCompletionsModel, OpenAIClient};
use openai_agents_sdk::tracing::processors::{
    BackendSpanExporter, BatchTraceProcessor, ExportItem, TracingExporter,
};
use openai_agents_sdk::tracing::{TracingProcessor, trace};
use openai_agents_sdk::{Agent, RunOptions, Runner};
use serde_json::{Value, json};
use std::sync::Mutex;

use support::{
    ModelReply, Pipeline, SpanRecorder, agent_data, new_span, new_trace, official_http_client,
    start_model_endpoint, start_tls_model_endpoint,
};

fn chat_completion() -> Value {
    json!({
        "id": "synthetic-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }],
    })
}

/// Every span the exporter posted, in arrival order, from the recorded request bodies.
fn posted_items(bodies: &Arc<Mutex<Vec<Value>>>) -> Vec<Value> {
    bodies
        .lock()
        .unwrap()
        .iter()
        .flat_map(|body| body["data"].as_array().cloned().unwrap_or_default())
        .collect()
}

fn json_len(value: &Value) -> usize {
    serde_json::to_string(value).expect("json").len()
}

/// The exporter and the batch processor block on their own runtime, so this test is synchronous and
/// drives the agent run through a runtime it owns.
#[test]
fn backend_span_exporter_bounds_multipart_normalization_work() {
    if !support::openssl_available() {
        eprintln!("{}", support::OPENSSL_SKIP_MESSAGE);
        return;
    }
    let _lock = support::lock();
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let rec = SpanRecorder::new();
    let _pipeline = Pipeline::install(vec![rec.clone() as Arc<dyn TracingProcessor>]);

    // Every chat request the model sends, so the exported input can be compared to it.
    let endpoint = start_model_endpoint(vec![ModelReply::Json(chat_completion())]);
    let model = OpenAIChatCompletionsModel::new(
        "test-model",
        OpenAIClient::new("test_key").with_base_url(&endpoint.base_url),
    );
    let agent = Agent::<()>::new("multipart-test").with_model_instance(Arc::new(model));

    // The exporter posts to the official ingest URL, which only the sanitizer's gate recognises,
    // so the host is routed to a local TLS server that records every posted payload.
    let ingest = start_tls_model_endpoint(vec![ModelReply::Json(json!({}))]);
    let exporter = BackendSpanExporter::new()
        .with_http_client(official_http_client(&ingest))
        .with_api_key("sk-test");

    let mut measured = Vec::new();
    for width in [1_000usize, 2_000] {
        let parts: Vec<Value> = (0..width)
            .map(|_| json!({"type": "input_text", "text": "synthetic ".repeat(20)}))
            .collect();
        let input = vec![InputItem::user_parts(parts)];
        let outcome = runtime.block_on(trace("multipart").run(async {
            Runner::run_with_options(&agent, input, RunOptions::new(()))
                .await
                .expect("run")
        }));
        assert_eq!(outcome.final_output_text().as_deref(), Some("ok"));

        let span = rec
            .ordered_spans()
            .into_iter()
            .filter(|s| {
                s.export()
                    .is_some_and(|e| e["span_data"]["type"] == "generation")
            })
            .last()
            .expect("generation span");
        let original = span.export().expect("exported span");
        let provider_messages = endpoint
            .bodies
            .lock()
            .unwrap()
            .last()
            .expect("request body")["messages"]
            .clone();
        assert_eq!(original["span_data"]["input"], provider_messages);
        assert_eq!(
            provider_messages[0]["content"].as_array().map(Vec::len),
            Some(width)
        );

        let started = Instant::now();
        exporter.export(vec![ExportItem::new(original.clone(), None)]);
        measured.push(started.elapsed());
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "exporting a {width}-part input must stay bounded"
        );

        let sent = posted_items(&ingest.bodies)
            .last()
            .cloned()
            .expect("the export was posted");
        let sent_input = sent["span_data"]["input"].clone();
        assert!(
            json_len(&sent_input) <= 100_000,
            "sent {} bytes",
            json_len(&sent_input)
        );
        assert_eq!(sent_input[0]["role"], "user");
        assert!(sent_input[0]["content"].is_array());
        assert!(
            sent_input[0]["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|part| part
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|t| !t.is_empty())),
            "a text part survives the bound"
        );
        assert_eq!(span.export().expect("still exported"), original);
    }
    assert!(measured[1] <= Duration::from_secs(10));

    // An ordinary span queued after the large one still arrives, in its own batch.
    let processor = BatchTraceProcessor::with_options(
        Arc::new(exporter.clone()),
        8192,
        1,
        Duration::from_millis(10),
        0.7,
    );
    let before = posted_items(&ingest.bodies).len();
    let root = new_trace("normal");
    let normal = new_span(&root, agent_data("normal"));
    let large = rec
        .ordered_spans()
        .into_iter()
        .rfind(|s| {
            s.export()
                .is_some_and(|e| e["span_data"]["type"] == "generation")
        })
        .expect("large span");
    processor.on_span_end(&large);
    processor.on_span_end(&normal);
    processor.force_flush();
    processor.shutdown_with_timeout(Duration::from_secs(1));

    let received = posted_items(&ingest.bodies);
    assert_eq!(received.len() - before, 2, "two batches, one span each");
    let tail = &received[received.len() - 2..];
    assert_eq!(tail[0]["id"], large.span_id());
    assert_eq!(tail[1], normal.export().expect("normal export"));
}
