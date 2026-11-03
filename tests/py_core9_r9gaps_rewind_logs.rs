//! Ports of the rewind logging cases of `tests/test_agent_runner.py`. The logger and the
//! `debug` flags are process-global, so these live in their own binary and run one at a time.

use std::sync::Mutex;

use async_trait::async_trait;
use log::{Level, Log, Metadata, Record};
use openai_agents_sdk::memory::{Session, rewind_session_items};
use openai_agents_sdk::{AgentsError, InputItem};
use serde_json::{Value, json};

static LOGS: Mutex<Vec<(Level, String)>> = Mutex::new(Vec::new());
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl Log for Capture {
    fn enabled(&self, _: &Metadata) -> bool {
        true
    }
    fn log(&self, record: &Record) {
        LOGS.lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }
    fn flush(&self) {}
}

fn install() {
    let _ = log::set_logger(&Capture);
    log::set_max_level(log::LevelFilter::Debug);
    LOGS.lock().unwrap().clear();
}

#[derive(Debug, Default)]
struct Strip {
    items: Mutex<Vec<InputItem>>,
    fail_reads: bool,
}

#[async_trait]
impl Session for Strip {
    fn session_id(&self) -> &str {
        "strip"
    }
    fn ignores_item_ids_for_matching(&self) -> bool {
        true
    }
    async fn get_items(&self, _limit: Option<usize>) -> openai_agents_sdk::Result<Vec<InputItem>> {
        if self.fail_reads {
            return Err(AgentsError::user("tail failure"));
        }
        Ok(self.items.lock().unwrap().clone())
    }
    async fn add_items(&self, items: Vec<InputItem>) -> openai_agents_sdk::Result<()> {
        let mut store = self.items.lock().unwrap();
        for mut it in items {
            if let Value::Object(m) = &mut it.0 {
                m.remove("id");
            }
            store.push(it);
        }
        Ok(())
    }
    async fn pop_item(&self) -> openai_agents_sdk::Result<Option<InputItem>> {
        Ok(self.items.lock().unwrap().pop())
    }
    async fn clear_session(&self) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
}

/// Python: `test_rewind_debug_logging_respects_model_and_tool_policies[True/False]`.
#[tokio::test]
async fn rewind_debug_logging_respects_model_and_tool_policies() {
    let _guard = SERIAL.lock().await;
    install();
    for redacted in [true, false] {
        openai_agents_sdk::debug::set_dont_log_model_data(redacted);
        openai_agents_sdk::debug::set_dont_log_tool_data(redacted);
        LOGS.lock().unwrap().clear();
        let secret = "SECRET_REWIND_SESSION_CONTENT";
        let session = Strip::default();
        let target = InputItem(
            json!({"id": "message-1", "type": "message", "role": "user", "content": secret}),
        );
        session.add_items(vec![target.clone()]).await.unwrap();

        rewind_session_items(&session, &[target], None, None).await;

        let logged = format!("{:?}", LOGS.lock().unwrap());
        assert_eq!(!logged.contains(secret), redacted, "redacted={redacted}");
    }
    openai_agents_sdk::debug::reset_dont_log_model_data();
    openai_agents_sdk::debug::reset_dont_log_tool_data();
}

/// Python: `test_rewind_failure_uses_placeholder_free_shared_logger_message`.
#[tokio::test]
async fn a_failed_tail_read_warns_with_the_shared_message() {
    let _guard = SERIAL.lock().await;
    install();
    let target = InputItem(json!({"type": "message", "role": "user", "content": "hi"}));
    let session = Strip {
        items: Mutex::new(vec![target.clone()]),
        fail_reads: true,
    };

    rewind_session_items(&session, &[target], None, None).await;

    let logs = LOGS.lock().unwrap();
    assert!(
        logs.iter().any(|(level, msg)| *level == Level::Warn
            && msg.starts_with("Failed to rewind session item")),
        "{logs:?}"
    );
}
