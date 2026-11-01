//! Port of `test_close_waits_for_background_finalizers_before_model_close` from
//! `tests/realtime/test_session.py`: the transport closes only after the running tool call has
//! been torn down.
#![cfg(feature = "realtime")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openai_agents_sdk::error::Result;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeModel, RealtimeModelConfig, RealtimeModelEvent,
    RealtimeModelEvents, RealtimeModelSendEvent, RealtimeModelToolCallEvent, RealtimeSession,
};
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;
use tokio::sync::Notify;

type Order = Arc<Mutex<Vec<&'static str>>>;

#[derive(Debug)]
struct OrderingModel {
    inner: FakeRealtimeModel,
    order: Order,
}

#[async_trait::async_trait]
impl RealtimeModel for OrderingModel {
    async fn connect(&self, config: RealtimeModelConfig) -> Result<RealtimeModelEvents> {
        self.inner.connect(config).await
    }
    async fn send_event(&self, event: RealtimeModelSendEvent) -> Result<()> {
        self.inner.send_event(event).await
    }
    async fn close(&self) -> Result<()> {
        self.order.lock().unwrap().push("model");
        self.inner.close().await
    }
}

struct RecordOnDrop(Order);

impl Drop for RecordOnDrop {
    fn drop(&mut self) {
        self.0.lock().unwrap().push("tool");
    }
}

#[tokio::test]
async fn close_tears_down_a_running_tool_before_closing_the_model() {
    let order: Order = Arc::default();
    let started = Arc::new(Notify::new());
    let finished = Arc::new(AtomicBool::new(false));
    let tool = {
        let (order, started, finished) = (order.clone(), started.clone(), finished.clone());
        FunctionTool::<()>::new_raw(
            "wait_forever",
            "never returns",
            json!({"type": "object", "properties": {}}),
            move |_ctx, _args| {
                let (order, started, finished) = (order.clone(), started.clone(), finished.clone());
                async move {
                    let _guard = RecordOnDrop(order);
                    started.notify_one();
                    std::future::pending::<()>().await;
                    finished.store(true, Ordering::SeqCst);
                    Ok("never".to_string())
                }
            },
        )
    };
    let model = Arc::new(OrderingModel {
        inner: FakeRealtimeModel::new(),
        order: order.clone(),
    });
    let session = RealtimeSession::connect(
        model.clone(),
        RealtimeAgent::<()>::new("agent").with_tool(tool).into_arc(),
        (),
        None,
        None,
    )
    .await
    .unwrap();
    model.inner.emit(RealtimeModelEvent::FunctionCall(
        RealtimeModelToolCallEvent::new("wait_forever", "call-1", "{}"),
    ));
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("the tool started");

    session.close().await.unwrap();

    assert_eq!(*order.lock().unwrap(), ["tool", "model"]);
    assert!(!finished.load(Ordering::SeqCst));
}
