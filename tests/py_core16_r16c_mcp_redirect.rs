//! Port of `test_default_streamable_http_client_matches_expected_defaults`
//! (`tests/mcp/test_streamable_http_client_factory.py`): the default MCP HTTP client never
//! follows a redirect (`follow_redirects` is False), so a credential header sent to the
//! configured URL is not replayed at another origin. A caller-built client replaces the default
//! and keeps its own redirect policy.
#![cfg(feature = "mcp")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use openai_agents_sdk::mcp::{
    MCPServer, MCPServerSse, MCPServerSseParams, MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A server that answers every request with `response` and counts the requests it got.
async fn serve(response: impl Fn() -> String + Send + 'static) -> (u16, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buffer = vec![0u8; 8192];
            let _ = stream.read(&mut buffer).await;
            counter.fetch_add(1, Ordering::SeqCst);
            let _ = stream.write_all(response().as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });
    (port, hits)
}

/// A redirecting server and the server it points at.
async fn redirect_pair() -> (String, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let (target_port, target_hits) =
        serve(|| "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string())
            .await;
    let location = format!("http://127.0.0.1:{target_port}/mcp");
    let (port, redirect_hits) = serve(move || {
        format!(
            "HTTP/1.1 307 Temporary Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
    })
    .await;
    (
        format!("http://127.0.0.1:{port}/mcp"),
        redirect_hits,
        target_hits,
    )
}

#[tokio::test]
async fn the_default_streamable_http_client_does_not_follow_redirects() {
    let (url, redirect_hits, target_hits) = redirect_pair().await;
    let params = MCPServerStreamableHttpParams {
        url,
        ..Default::default()
    };
    let server = MCPServerStreamableHttp::new("redirecting", params);
    assert!(server.connect().await.is_err());
    assert!(redirect_hits.load(Ordering::SeqCst) >= 1);
    assert_eq!(
        target_hits.load(Ordering::SeqCst),
        0,
        "the redirect was followed"
    );
}

#[tokio::test]
async fn the_default_sse_client_does_not_follow_redirects() {
    let (url, redirect_hits, target_hits) = redirect_pair().await;
    let params = MCPServerSseParams {
        url,
        ..Default::default()
    };
    let server = MCPServerSse::new("redirecting", params);
    assert!(server.connect().await.is_err());
    assert!(redirect_hits.load(Ordering::SeqCst) >= 1);
    assert_eq!(
        target_hits.load(Ordering::SeqCst),
        0,
        "the redirect was followed"
    );
}

#[tokio::test]
async fn a_caller_built_client_keeps_its_own_redirect_policy() {
    let (url, _redirect_hits, target_hits) = redirect_pair().await;
    let params = MCPServerStreamableHttpParams {
        url,
        http_client: Some(reqwest::Client::new()),
        ..Default::default()
    };
    let server = MCPServerStreamableHttp::new("redirecting", params);
    // The target is not an MCP server, so the handshake still fails; what matters is that this
    // client reached it.
    assert!(server.connect().await.is_err());
    assert!(target_hits.load(Ordering::SeqCst) >= 1);
}
