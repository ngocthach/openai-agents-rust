//! Ports of the `close()` cases of `tests/extensions/memory/test_dapr_session.py`.
//!
//! Python's session owns its gRPC client when built with `from_address` and leaves an injected
//! client alone. Rust's `DaprSession::new` builds and owns its HTTP client, and
//! `with_http_client` injects one, so the same ownership split applies. The sidecar is a small
//! in-process HTTP server that counts requests and can hold a read open.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{DaprSession, Session};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;

#[derive(Default)]
struct Sidecar {
    store: Mutex<HashMap<String, String>>,
    requests: AtomicUsize,
    /// While set, the next state read waits for `release` after signalling `held`.
    hold_next_read: Mutex<bool>,
    held: Notify,
    release: Notify,
}

async fn read_request(stream: &mut tokio::net::TcpStream) -> Option<(String, String, String)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    while buf.len() < header_end + length {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let mut parts = head.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let body = String::from_utf8_lossy(&buf[header_end..]).to_string();
    Some((method, path, body))
}

async fn answer(sidecar: &Sidecar, method: &str, path: &str, body: &str) -> (u16, String) {
    let path = path.split('?').next().unwrap_or_default();
    if path == "/v1.0/metadata" {
        return (200, json!({"id": "mock"}).to_string());
    }
    let Some(key) = path.strip_prefix("/v1.0/state/statestore") else {
        return (404, String::new());
    };
    let key = key.trim_start_matches('/').to_string();
    match method {
        "GET" => {
            let hold = std::mem::take(&mut *sidecar.hold_next_read.lock().unwrap());
            if hold {
                sidecar.held.notify_one();
                sidecar.release.notified().await;
            }
            match sidecar.store.lock().unwrap().get(&key) {
                Some(value) => (200, value.clone()),
                None => (204, String::new()),
            }
        }
        "POST" => {
            let entries: serde_json::Value = serde_json::from_str(body).unwrap();
            let entry = &entries[0];
            let key = entry["key"].as_str().unwrap_or_default().to_string();
            sidecar
                .store
                .lock()
                .unwrap()
                .insert(key, entry["value"].to_string());
            (204, String::new())
        }
        "DELETE" => {
            sidecar.store.lock().unwrap().remove(&key);
            (204, String::new())
        }
        _ => (405, String::new()),
    }
}

async fn mock_sidecar() -> (String, Arc<Sidecar>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let sidecar = Arc::new(Sidecar::default());
    let shared = sidecar.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let sidecar = shared.clone();
            tokio::spawn(async move {
                let Some((method, path, body)) = read_request(&mut stream).await else {
                    return;
                };
                sidecar.requests.fetch_add(1, Ordering::SeqCst);
                let (status, body) = answer(&sidecar, &method, &path, &body).await;
                let response = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (base, sidecar)
}

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

fn contents(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.as_value()["content"].as_str().unwrap().to_string())
        .collect()
}

fn assert_closed<T: std::fmt::Debug>(result: openai_agents_sdk::Result<T>) {
    let error = result.expect_err("a closed session must refuse the operation");
    assert!(
        error.to_string().contains("DaprSession is closed"),
        "{error}"
    );
}

/// `test_dapr_session_closed_operations_raise_runtime_error`, and the owned half of
/// `test_internal_client_ownership` and `test_close_method_coverage`: closing a session that
/// owns its client makes it terminal, and no operation reaches the sidecar afterwards.
#[tokio::test]
async fn closed_operations_on_an_owned_client_fail_without_a_request() {
    let (base, sidecar) = mock_sidecar().await;
    let session = DaprSession::new("closed_state_test", "statestore", &base);
    session.add_items(vec![user("before close")]).await.unwrap();
    assert!(!session.is_closed());

    session.close().await;

    assert!(session.is_closed());
    let before = sidecar.requests.load(Ordering::SeqCst);
    assert_closed(session.get_items(None).await);
    assert_closed(session.add_items(vec![user("after close")]).await);
    assert_closed(session.pop_item().await);
    assert_closed(session.clear_session().await);
    assert_eq!(sidecar.requests.load(Ordering::SeqCst), before);
}

/// `test_dapr_session_closed_rejects_empty_add_items`.
#[tokio::test]
async fn a_closed_session_rejects_an_empty_add_items() {
    let (base, _sidecar) = mock_sidecar().await;
    let session = DaprSession::new("closed_empty_add_test", "statestore", &base);
    session.close().await;
    assert_closed(session.add_items(Vec::new()).await);
}

/// `test_dapr_session_close_is_idempotent`: concurrent and repeated closes are safe.
#[tokio::test]
async fn repeated_and_concurrent_closes_are_safe() {
    let (base, _sidecar) = mock_sidecar().await;
    let session = DaprSession::new("close_idempotent_test", "statestore", &base);
    tokio::join!(session.close(), session.close());
    session.close().await;
    assert_closed(session.get_items(None).await);
}

/// `test_dapr_session_ping_raises_after_close`: `ping` must not use a client the session
/// closed; it fails with `DaprSession is closed` and sends nothing.
#[tokio::test]
async fn ping_after_close_sends_no_request() {
    let (base, sidecar) = mock_sidecar().await;
    let session = DaprSession::new("ping_after_close_test", "statestore", &base);
    assert!(session.ping().await.unwrap());
    session.close().await;
    let before = sidecar.requests.load(Ordering::SeqCst);
    assert_closed(session.ping().await);
    assert_eq!(sidecar.requests.load(Ordering::SeqCst), before);
}

/// `test_dapr_session_close_is_noop_for_injected_client`, `test_external_client_not_closed` and
/// the injected half of `test_close_method_coverage`: with an injected client, close does
/// nothing and the session and its client stay usable.
#[tokio::test]
async fn close_is_a_no_op_for_an_injected_client() {
    let (base, _sidecar) = mock_sidecar().await;
    let client = reqwest::Client::new();
    let session = DaprSession::new("injected_client_noop_test", "statestore", &base)
        .with_http_client(client.clone());
    session.add_items(vec![user("before close")]).await.unwrap();

    session.close().await;

    assert!(!session.is_closed());
    session.add_items(vec![user("after close")]).await.unwrap();
    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        ["before close", "after close"]
    );
    assert!(session.ping().await.unwrap());

    let other =
        DaprSession::new("injected_client_noop_test", "statestore", &base).with_http_client(client);
    assert_eq!(other.get_items(None).await.unwrap().len(), 2);
}

/// `test_dapr_session_operation_waiting_behind_close_raises`: an operation queued behind
/// `close()` fails instead of running after it. Rust's close has nothing to await, so it queues
/// behind a read the sidecar holds open, and a second read queues behind the close.
#[tokio::test]
async fn an_operation_queued_behind_close_fails() {
    let (base, sidecar) = mock_sidecar().await;
    let session = DaprSession::new("close_interleave_test", "statestore", &base);
    session.add_items(vec![user("kept")]).await.unwrap();
    *sidecar.hold_next_read.lock().unwrap() = true;

    let in_flight = session.get_items(None);
    let queued = async {
        sidecar.held.notified().await;
        // The first read holds the session lock; close queues behind it and the second read
        // behind the close (the lock is first-in, first-out).
        let close = session.close();
        let read = session.get_items(None);
        let release = async {
            tokio::task::yield_now().await;
            sidecar.release.notify_one();
        };
        let ((), read, ()) = tokio::join!(close, read, release);
        read
    };
    let (first, second) = tokio::join!(in_flight, queued);

    assert_eq!(contents(&first.unwrap()), ["kept"]);
    assert_closed(second);
    assert!(session.is_closed());
}
