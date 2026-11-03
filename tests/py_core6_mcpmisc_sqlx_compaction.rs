//! The `outer=True, reject_insert=True` variant of `test_encrypted_prefix_cleanup_is_atomic`
//! from `tests/memory/test_sqlalchemy_compaction.py`, on `SqlxSession` with a file-backed SQLite
//! database: encryption wraps the compaction session, expired envelopes are reclaimed together
//! with the replacement, and a rejected summary insert rolls both back.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(all(feature = "sqlx", feature = "sqlite", feature = "encrypt"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::memory::{
    EncryptedSession, OpenAIResponsesCompactionSession, Session, SessionSettings, SqlxSession,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunOptions, Runner};
use serde_json::{Value, json};

mod py_mcpmem_support;
use py_mcpmem_support::{Reply, Request, serve};

const KEY: &str = "0123456789abcdef0123456789abcdef";
const TTL: u64 = 2;
const PAST_TTL: Duration = Duration::from_millis(3200);

fn summary() -> Value {
    json!({"type": "compaction", "id": "cmp", "encrypted_content": "synthetic-summary"})
}

fn alternating(count: usize) -> Vec<InputItem> {
    (0..count)
        .map(|i| {
            let role = if i % 2 == 1 { "assistant" } else { "user" };
            InputItem::from_value(json!({"role": role, "content": format!("live {i}")}))
        })
        .collect()
}

async fn encrypted_outermost_cleanup(reject_insert: bool) {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let path: std::path::PathBuf = dir.path().join("history.db");
    let backend = Arc::new(
        SqlxSession::from_url(
            "bounded",
            &format!("sqlite://{}?mode=rwc", path.display()),
            true,
        )
        .await
        .unwrap()
        .with_settings(SessionSettings::with_limit(20)),
    );
    let compactions = Arc::new(Mutex::new(Vec::<Value>::new()));
    let (record, trigger_path) = (compactions.clone(), path.clone());
    let base = serve(Arc::new(move |req: &Request| {
        record.lock().unwrap().push(req.body.clone());
        if reject_insert {
            let path = trigger_path.clone();
            std::thread::spawn(move || {
                rusqlite::Connection::open(path)
                    .unwrap()
                    .execute_batch(
                        "CREATE TRIGGER reject_summary BEFORE INSERT ON agent_messages \
                         BEGIN SELECT RAISE(ABORT, 'synthetic summary insertion failure'); END",
                    )
                    .unwrap();
            })
            .join()
            .unwrap();
        }
        Reply::json(json!({
            "id": "cmp_1", "output": [summary()],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }))
    }))
    .await;
    let compaction = Arc::new(
        OpenAIResponsesCompactionSession::new("bounded", backend.clone())
            .unwrap()
            .with_client(OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"))),
    );
    let session: Arc<dyn Session> =
        Arc::new(EncryptedSession::new("bounded", compaction, KEY, TTL).unwrap());
    session
        .add_items((0..40).map(|_| InputItem::user("expired")).collect())
        .await
        .unwrap();
    tokio::time::sleep(PAST_TTL).await;
    session.add_items(alternating(20)).await.unwrap();
    let stored_before = backend.get_items(Some(100)).await.unwrap();
    assert_eq!(stored_before.len(), 60);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("worker").with_model_instance(model);
    let result = Runner::run_with_options(
        &agent,
        "continue",
        RunOptions::new(()).session(session.clone()),
    )
    .await;

    let requests = compactions.lock().unwrap().clone();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].to_string().contains("expired"));
    if reject_insert {
        let error = result
            .expect_err("the rejected summary insert must surface")
            .to_string();
        assert!(
            error.contains("synthetic summary insertion failure"),
            "{error}"
        );
        let after = backend.get_items(Some(100)).await.unwrap();
        assert_eq!(&after[..60], &stored_before[..]);
        assert_eq!(after.len(), 62);
    } else {
        result.unwrap();
        assert_eq!(backend.get_items(Some(100)).await.unwrap().len(), 1);
        let visible = session.get_items(None).await.unwrap();
        assert_eq!(visible.len(), 1);
        assert_eq!(
            visible[0].as_value()["encrypted_content"],
            "synthetic-summary"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_summary_insert_rolls_back_the_prefix_cleanup_with_outer_encryption() {
    encrypted_outermost_cleanup(true).await;
}
