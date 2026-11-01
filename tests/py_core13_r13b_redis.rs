//! Client ownership cases of `tests/extensions/memory/test_redis_session.py`
//! (`test_external_client_not_closed`, `test_internal_client_ownership`,
//! `test_redis_session_close_is_noop_for_injected_client`). Python closes the client a session
//! created itself and leaves a caller's client open on `close()`; Rust has no `close()`, the
//! connection a session built is released when the session is dropped and a connection the
//! caller passed in belongs to the caller.
#![cfg(feature = "redis")]

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use openai_agents_sdk::InputItem;
use openai_agents_sdk::memory::{RedisSession, Session};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Parse one RESP array of bulk strings from the front of `buf`.
fn parse_command(buf: &[u8]) -> Option<(Vec<String>, usize)> {
    fn line(buf: &[u8], from: usize) -> Option<(&str, usize)> {
        let end = buf[from..].windows(2).position(|w| w == b"\r\n")? + from;
        Some((std::str::from_utf8(&buf[from..end]).ok()?, end + 2))
    }
    let (head, mut pos) = line(buf, 0)?;
    let count: usize = head.strip_prefix('*')?.parse().ok()?;
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        let (len_line, next) = line(buf, pos)?;
        let len: usize = len_line.strip_prefix('$')?.parse().ok()?;
        if buf.len() < next + len + 2 {
            return None;
        }
        args.push(String::from_utf8_lossy(&buf[next..next + len]).into_owned());
        pos = next + len + 2;
    }
    Some((args, pos))
}

/// A RESP endpoint that answers `PING` and counts its open connections.
async fn endpoint() -> (String, Arc<AtomicI64>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("redis://{}/", listener.local_addr().unwrap());
    let open = Arc::new(AtomicI64::new(0));
    let counter = open.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let counter = counter.clone();
            counter.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let mut pending: Vec<u8> = Vec::new();
                let mut buf = [0u8; 4096];
                'connection: loop {
                    while let Some((args, used)) = parse_command(&pending) {
                        pending.drain(..used);
                        let reply: &[u8] = if args[0].eq_ignore_ascii_case("PING") {
                            b"+PONG\r\n"
                        } else {
                            b"+OK\r\n"
                        };
                        if sock.write_all(reply).await.is_err() {
                            break 'connection;
                        }
                    }
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => pending.extend_from_slice(&buf[..n]),
                    }
                }
                counter.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    (url, open)
}

async fn wait_for(open: &AtomicI64, expected: i64, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while open.load(Ordering::SeqCst) != expected {
        assert!(
            Instant::now() < deadline,
            "{what}: {} connections open, expected {expected}",
            open.load(Ordering::SeqCst)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn a_connection_the_caller_passed_in_outlives_the_session() {
    let (url, open) = endpoint().await;
    let client = redis::Client::open(url).unwrap();
    let mut shared = client.get_multiplexed_async_connection().await.unwrap();
    let session = RedisSession::with_connection("external_client_test", shared.clone());
    assert!(session.ping().await);

    drop(session);
    // Give a (wrong) close time to land, then use the caller's own handle.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(open.load(Ordering::SeqCst), 1);
    let pong: String = redis::cmd("PING").query_async(&mut shared).await.unwrap();
    assert_eq!(pong, "PONG");
}

#[tokio::test]
async fn a_connection_the_session_opened_is_released_with_it() {
    let (url, open) = endpoint().await;
    let session = RedisSession::from_url("internal_client_test", &url)
        .await
        .unwrap();
    assert!(session.ping().await);
    wait_for(&open, 1, "while the session lives").await;

    drop(session);
    wait_for(&open, 0, "after the session is dropped").await;
}

/// An endpoint that hangs up on the first write script and counts how many it was sent.
async fn hanging_up_on_writes() -> (String, Arc<AtomicI64>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("redis://{}/", listener.local_addr().unwrap());
    let scripts = Arc::new(AtomicI64::new(0));
    let counter = scripts.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let counter = counter.clone();
            tokio::spawn(async move {
                let mut pending: Vec<u8> = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    while let Some((args, used)) = parse_command(&pending) {
                        pending.drain(..used);
                        if args[0].to_uppercase().starts_with("EVAL") {
                            counter.fetch_add(1, Ordering::SeqCst);
                            return;
                        }
                        let reply: &[u8] = if args[0].eq_ignore_ascii_case("PING") {
                            b"+PONG\r\n"
                        } else {
                            b"+OK\r\n"
                        };
                        if sock.write_all(reply).await.is_err() {
                            return;
                        }
                    }
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => pending.extend_from_slice(&buf[..n]),
                    }
                }
            });
        }
    });
    (url, scripts)
}

#[tokio::test]
async fn a_write_whose_outcome_is_unknown_is_not_retried() {
    // `test_ambiguous_exec_watch_error_is_not_retried`: the connection drops after the write
    // was sent, so the server may or may not have applied it; resending could duplicate it.
    let (url, scripts) = hanging_up_on_writes().await;
    let session = RedisSession::from_url("ambiguous", &url).await.unwrap();

    let result = session.add_items(vec![InputItem::user("once")]).await;

    assert!(result.is_err(), "the interrupted write must be reported");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        scripts.load(Ordering::SeqCst),
        1,
        "the write was sent again"
    );
}
