//! Ports of the client-metadata cases of `tests/extensions/memory/test_mongodb_session.py`
//! (`test_injected_client_receives_append_metadata`, `test_from_uri_passes_driver_info_to_constructor`):
//! the session identifies the SDK in the MongoDB connection handshake, both for a client it builds
//! from a URI and for a client the caller injects. A tiny OP_MSG server records the handshake.
#![cfg(feature = "mongodb")]

use std::sync::{Arc, Mutex};

use mongodb::Client;
use mongodb::bson::{Bson, Document, doc};
use openai_agents_sdk::memory::MongoDBSession;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The first (command) section of an OP_MSG body.
fn parse_op_msg(body: &[u8]) -> Document {
    let mut pos = 4; // flag bits
    let mut command = Document::new();
    while pos < body.len() {
        let kind = body[pos];
        pos += 1;
        let size = i32::from_le_bytes(body[pos..pos + 4].try_into().unwrap()) as usize;
        if kind == 0 {
            command = Document::from_reader(&mut &body[pos..pos + size]).unwrap();
        }
        pos += size;
    }
    command
}

/// A server that answers every command and records the `client` metadata of each handshake.
async fn recording_server() -> (String, Arc<Mutex<Vec<Document>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handshakes = Arc::new(Mutex::new(Vec::new()));
    let seen = handshakes.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let seen = seen.clone();
            tokio::spawn(async move {
                loop {
                    let mut header = [0u8; 16];
                    if sock.read_exact(&mut header).await.is_err() {
                        return;
                    }
                    let len = i32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
                    let request_id = i32::from_le_bytes(header[4..8].try_into().unwrap());
                    let mut body = vec![0u8; len - 16];
                    if sock.read_exact(&mut body).await.is_err() {
                        return;
                    }
                    let command = parse_op_msg(&body);
                    let name = command.keys().next().cloned().unwrap_or_default();
                    let reply = if matches!(name.as_str(), "hello" | "isMaster" | "ismaster") {
                        if let Some(Bson::Document(client)) = command.get("client") {
                            seen.lock().unwrap().push(client.clone());
                        }
                        doc! {
                            "ismaster": true,
                            "isWritablePrimary": true,
                            "helloOk": true,
                            "maxBsonObjectSize": 16777216,
                            "maxMessageSizeBytes": 48000000,
                            "maxWriteBatchSize": 100000,
                            "minWireVersion": 0,
                            "maxWireVersion": 17,
                            "ok": 1.0,
                        }
                    } else {
                        doc! {"ok": 1.0}
                    };
                    let mut payload = Vec::new();
                    payload.extend_from_slice(&0u32.to_le_bytes());
                    payload.push(0);
                    reply.to_writer(&mut payload).unwrap();
                    let mut out = Vec::new();
                    out.extend_from_slice(&((payload.len() + 16) as i32).to_le_bytes());
                    out.extend_from_slice(&1i32.to_le_bytes());
                    out.extend_from_slice(&request_id.to_le_bytes());
                    out.extend_from_slice(&2013i32.to_le_bytes());
                    out.extend_from_slice(&payload);
                    if sock.write_all(&out).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    let uri = format!(
        "mongodb://127.0.0.1:{port}/?directConnection=true&serverSelectionTimeoutMS=2000\
         &connectTimeoutMS=2000&heartbeatFrequencyMS=500"
    );
    (uri, handshakes)
}

/// Whether any handshake the server saw names the SDK. The driver's monitor connection may
/// handshake before the metadata is appended, so the check looks at every handshake.
fn names_the_sdk(handshakes: &Arc<Mutex<Vec<Document>>>) -> bool {
    let seen = handshakes.lock().unwrap();
    assert!(!seen.is_empty(), "the server must have seen a handshake");
    seen.iter().any(|client| {
        client
            .get_document("driver")
            .and_then(|driver| driver.get_str("name"))
            .is_ok_and(|name| name.contains("openai-agents"))
    })
}

/// Python: `test_from_uri_passes_driver_info_to_constructor`.
#[tokio::test]
async fn a_session_built_from_a_uri_identifies_the_sdk_in_the_handshake() {
    let (uri, handshakes) = recording_server().await;
    let session = MongoDBSession::from_uri("uri-test", &uri, "t")
        .await
        .unwrap();
    assert!(session.ping().await);
    assert!(names_the_sdk(&handshakes));
}

/// Python: `test_injected_client_receives_append_metadata`.
#[tokio::test]
async fn an_injected_client_receives_the_sdk_identity_before_it_connects() {
    let (uri, handshakes) = recording_server().await;
    let client = Client::with_uri_str(&uri).await.unwrap();
    let session = MongoDBSession::with_client("meta-test", &client, "agents_test");
    assert!(session.ping().await);
    assert!(names_the_sdk(&handshakes));
}
