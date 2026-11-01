//! Redis Cluster cases of `tests/extensions/memory/test_redis_session.py`
//! (`test_redis_session_rejects_cluster_transactions_before_pipeline`, `test_redis_cluster_reads_and_subclass_writes`).
//!
//! Python's `RedisSession` accepts a `RedisCluster` for reads and pops and rejects every write
//! with a `TypeError` before creating a pipeline. The Rust session does the same with a
//! `ClusterConnection`, tested against a local RESP endpoint that answers the cluster topology
//! query and serves `LRANGE`/`RPOP` from a list. The "subclass writes" half of the Python test
//! overrides `add_items`/`clear_session` on a subclass; a Rust session is a concrete type, so the
//! read half is ported.
#![cfg(feature = "redis")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession, RedisSession,
    Session,
};
use openai_agents_sdk::{InputItem, OpenAIClient};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

mod py_mcpmem_support;
use py_mcpmem_support::mock_openai_with;

const MESSAGES_KEY: &str = "agents:session:cluster-history:messages";

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

fn bulk(s: &str) -> Vec<u8> {
    format!("${}\r\n{s}\r\n", s.len()).into_bytes()
}

struct Cluster {
    node: String,
    list: Arc<Mutex<Vec<String>>>,
    commands: Arc<Mutex<Vec<Vec<String>>>>,
}

impl Cluster {
    /// Names of every command the endpoint has received, upper-cased.
    fn command_names(&self) -> Vec<String> {
        self.commands
            .lock()
            .unwrap()
            .iter()
            .map(|c| c[0].to_uppercase())
            .collect()
    }
}

/// A one-node "cluster" owning every slot, holding one list at [`MESSAGES_KEY`].
async fn cluster(initial: &[InputItem]) -> Cluster {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let list = Arc::new(Mutex::new(
        initial
            .iter()
            .map(|i| serde_json::to_string(i.as_value()).unwrap())
            .collect::<Vec<_>>(),
    ));
    let commands = Arc::new(Mutex::new(Vec::new()));
    let (stored, seen) = (list.clone(), commands.clone());
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let (stored, seen) = (stored.clone(), seen.clone());
            tokio::spawn(async move {
                let mut pending: Vec<u8> = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    while let Some((args, used)) = parse_command(&pending) {
                        pending.drain(..used);
                        seen.lock().unwrap().push(args.clone());
                        let name = args[0].to_uppercase();
                        let reply: Vec<u8> = match name.as_str() {
                            "CLUSTER" => {
                                // [[0, 16383, [ip, port, id]]]
                                let mut out = b"*1\r\n*3\r\n:0\r\n:16383\r\n*3\r\n".to_vec();
                                out.extend(bulk("127.0.0.1"));
                                out.extend(format!(":{}\r\n", addr.port()).into_bytes());
                                out.extend(bulk("node-1"));
                                out
                            }
                            "PING" => b"+PONG\r\n".to_vec(),
                            "LRANGE" => {
                                let list = stored.lock().unwrap();
                                let len = list.len() as i64;
                                let norm = |i: i64| if i < 0 { (len + i).max(0) } else { i };
                                let (from, to) = (
                                    norm(args[2].parse().unwrap()),
                                    norm(args[3].parse().unwrap()).min(len - 1),
                                );
                                let window: Vec<&String> = if from > to {
                                    Vec::new()
                                } else {
                                    list[from as usize..=to as usize].iter().collect()
                                };
                                let mut out = format!("*{}\r\n", window.len()).into_bytes();
                                for entry in window {
                                    out.extend(bulk(entry));
                                }
                                out
                            }
                            "RPOP" => match stored.lock().unwrap().pop() {
                                Some(entry) => bulk(&entry),
                                None => b"$-1\r\n".to_vec(),
                            },
                            _ => b"+OK\r\n".to_vec(),
                        };
                        if sock.write_all(&reply).await.is_err() {
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
    Cluster {
        node: format!("redis://{addr}"),
        list,
        commands,
    }
}

fn history() -> Vec<InputItem> {
    vec![InputItem::user("hello"), InputItem::assistant("hi")]
}

async fn session(cluster: &Cluster) -> RedisSession {
    RedisSession::from_cluster_nodes("cluster-history", vec![cluster.node.clone()])
        .await
        .unwrap()
}

#[tokio::test]
async fn a_cluster_session_reads_and_pops_history() {
    let cluster = cluster(&history()).await;
    let session = session(&cluster).await;

    assert_eq!(session.get_items(None).await.unwrap(), history());
    assert_eq!(
        session.get_items(Some(1)).await.unwrap(),
        history()[1..].to_vec()
    );
    assert_eq!(
        session.pop_item().await.unwrap(),
        Some(InputItem::assistant("hi"))
    );
    assert_eq!(session.get_items(None).await.unwrap(), history()[..1]);
    assert_eq!(cluster.list.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_cluster_session_never_closes_a_caller_owned_connection() {
    let cluster = cluster(&history()).await;
    let client = redis::cluster::ClusterClient::new(vec![cluster.node.clone()]).unwrap();
    let mut shared = client.get_async_connection().await.unwrap();
    let session = RedisSession::with_cluster_connection("cluster-history", shared.clone());
    assert_eq!(session.get_items(None).await.unwrap(), history());
    drop(session);
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut lrange = redis::cmd("LRANGE");
    lrange.arg(MESSAGES_KEY).arg(0).arg(-1);
    let raw: Vec<String> = lrange.query_async(&mut shared).await.unwrap();
    assert_eq!(raw.len(), 2);
}

fn assert_cluster_rejection(err: &openai_agents_sdk::error::AgentsError) {
    let text = err.to_string();
    assert!(
        text.contains("RedisSession does not support transactions with Redis Cluster"),
        "{text}"
    );
    assert!(text.contains("standalone"), "{text}");
    assert!(text.contains("custom Session backend"), "{text}");
}

#[tokio::test]
async fn cluster_writes_are_rejected_before_anything_is_sent() {
    let cluster = cluster(&history()).await;
    let session = session(&cluster).await;
    let before = cluster.command_names().len();

    // An empty add is a no-op, as in Python.
    session.add_items(Vec::new()).await.unwrap();

    assert_cluster_rejection(
        &session
            .add_items(vec![InputItem::user("saved history")])
            .await
            .unwrap_err(),
    );
    assert_cluster_rejection(&session.clear_session().await.unwrap_err());
    assert_cluster_rejection(
        &session
            .replace_history_atomically(vec![InputItem::user("x")])
            .await
            .unwrap_err(),
    );
    assert_cluster_rejection(
        &session
            .replace_compaction_suffix(&[], vec![InputItem::user("x")])
            .await
            .unwrap_err(),
    );

    assert_eq!(cluster.command_names().len(), before, "nothing was sent");
    assert_eq!(cluster.list.lock().unwrap().len(), 2, "history untouched");
}

#[tokio::test]
async fn compaction_over_a_cluster_session_reports_the_rejection() {
    // The `compact` variant: manual compaction reads the history, asks the model, and the
    // replacement write is refused without a single write command reaching the cluster.
    openai_agents_sdk::set_tracing_disabled(true);
    let cluster = cluster(&history()).await;
    let inner: Arc<dyn Session> = Arc::new(session(&cluster).await);
    let (base, _requests) = mock_openai_with(|_| {
        json!({"id": "cmp_1", "output": [{"type": "message", "role": "user", "content": "saved history"}],
               "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})
    })
    .await;
    let compaction = OpenAIResponsesCompactionSession::new("cluster-history", inner)
        .unwrap()
        .with_client(OpenAIClient::new("sk-test").with_base_url(base))
        .with_compaction_mode(CompactionMode::Input);

    let err = compaction
        .run_compaction(OpenAIResponsesCompactionArgs {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap_err();

    assert_cluster_rejection(&err);
    let names = cluster.command_names();
    assert!(
        !names
            .iter()
            .any(|n| matches!(n.as_str(), "EVAL" | "EVALSHA" | "MULTI" | "DEL" | "RPUSH")),
        "{names:?}"
    );
    assert_eq!(cluster.list.lock().unwrap().len(), 2);
}
