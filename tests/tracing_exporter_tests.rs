//! `BackendSpanExporter` retry behaviour: `Retry-After`, the `x-should-retry` override and the
//! configurable backoff.
//!
//! Plain `#[test]`s: the exporter builds its own current-thread runtime and calls `block_on`,
//! which panics inside a tokio runtime.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use openai_agents_sdk::tracing::{BackendSpanExporter, ExportItem, TracingExporter};
use serde_json::json;

/// Serve `replies` in order (status, extra headers) and count the requests received.
fn mock_ingest(replies: Vec<(u16, &'static str)>) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let counter = requests.clone();
    std::thread::spawn(move || {
        for (status, headers) in replies {
            let Ok((mut sock, _)) = listener.accept() else {
                return;
            };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            // Read until the body is in; the client always sends Content-Length.
            loop {
                let Ok(n) = sock.read(&mut tmp) else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let len: usize = text[..idx]
                        .lines()
                        .find_map(|l| {
                            l.to_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break;
                    }
                }
            }
            counter.fetch_add(1, Ordering::SeqCst);
            let body = "{}";
            let reason = if status < 300 { "OK" } else { "Error" };
            let resp = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
        }
    });
    (format!("http://{addr}/traces/ingest"), requests)
}

fn one_span() -> Vec<ExportItem> {
    vec![ExportItem {
        payload: json!({"object": "trace.span", "id": "span_1"}),
        api_key: Some("sk-test".into()),
    }]
}

#[test]
fn x_should_retry_false_stops_after_one_request() {
    // A second reply is queued so a wrongly attempted retry would be counted, not refused.
    let (endpoint, requests) = mock_ingest(vec![(500, "x-should-retry: false\r\n"), (200, "")]);
    BackendSpanExporter::new()
        .with_endpoint(endpoint)
        .with_base_delay(Duration::from_millis(1))
        .export(one_span());
    assert_eq!(requests.load(Ordering::SeqCst), 1);
}

#[test]
fn retry_after_is_honored_and_bounded() {
    let (endpoint, requests) = mock_ingest(vec![(429, "retry-after-ms: 1\r\n"), (200, "")]);
    let started = std::time::Instant::now();
    BackendSpanExporter::new()
        .with_endpoint(endpoint)
        .with_base_delay(Duration::from_millis(1))
        .with_max_delay(Duration::from_millis(50))
        .export(one_span());
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    // `max_delay` bounds the wait, so a server-sent delay cannot stall the export thread.
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn a_client_error_is_not_retried() {
    let (endpoint, requests) = mock_ingest(vec![(400, ""), (200, "")]);
    BackendSpanExporter::new()
        .with_endpoint(endpoint)
        .with_base_delay(Duration::from_millis(1))
        .export(one_span());
    assert_eq!(requests.load(Ordering::SeqCst), 1);
}
