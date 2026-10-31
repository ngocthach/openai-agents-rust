//! Ports of the Python realtime session tests for history merging, transcript preservation,
//! reader termination around `close`, transport exceptions and tool failure events.

#![cfg(feature = "realtime")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::realtime::{
    FakeRealtimeModel, RealtimeAgent, RealtimeContent, RealtimeGuardrailsSettings, RealtimeItem,
    RealtimeMessageItem, RealtimeMessageRole, RealtimeModel, RealtimeModelConfig,
    RealtimeModelEvent, RealtimeModelSendEvent, RealtimeModelToolCallEvent, RealtimeRunConfig,
    RealtimeRunner, RealtimeSession, RealtimeSessionEvent, RealtimeStep, RealtimeToolCallItem,
    ScriptedRealtimeModel,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{AgentsError, GuardrailFunctionOutput, OutputGuardrail};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;

const TIMEOUT: Duration = Duration::from_secs(5);

type Session = RealtimeSession<()>;

async fn start(model: Arc<dyn RealtimeModel>) -> Session {
    start_with(
        model,
        RealtimeRunConfig::default(),
        RealtimeAgent::<()>::new("agent"),
    )
    .await
}

async fn start_with(
    model: Arc<dyn RealtimeModel>,
    config: RealtimeRunConfig<()>,
    agent: RealtimeAgent<()>,
) -> Session {
    RealtimeRunner::new(agent.into_arc())
        .with_model(model)
        .with_config(config)
        .run(())
        .await
        .expect("session connects")
}

async fn recv(session: &Session) -> RealtimeSessionEvent<()> {
    tokio::time::timeout(TIMEOUT, session.recv())
        .await
        .expect("timed out waiting for an event")
        .expect("event stream ended early")
}

async fn recv_types(session: &Session, count: usize) -> Vec<&'static str> {
    let mut types = Vec::new();
    for _ in 0..count {
        types.push(recv(session).await.event_type());
    }
    types
}

/// Read every remaining event until the stream ends.
async fn collect(session: &Session) -> Vec<&'static str> {
    tokio::time::timeout(TIMEOUT, async {
        let mut types = Vec::new();
        while let Some(event) = session.recv().await {
            types.push(event.event_type());
        }
        types
    })
    .await
    .expect("timed out draining the session")
}

/// A session with the initial `history_updated` already consumed.
async fn started_fake() -> (Arc<FakeRealtimeModel>, Session) {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone()).await;
    assert_eq!(recv_types(&session, 1).await, ["history_updated"]);
    (model, session)
}

/// Feed an item update and return the history event type that followed the raw event.
async fn put(
    model: &FakeRealtimeModel,
    session: &Session,
    item: impl Into<RealtimeItem>,
) -> &'static str {
    assert!(model.emit(RealtimeModelEvent::item_updated(item)));
    let types = recv_types(session, 2).await;
    assert_eq!(types[0], "raw_model_event");
    types[1]
}

fn assistant(id: &str, content: Vec<RealtimeContent>) -> RealtimeMessageItem {
    RealtimeMessageItem::new(id, RealtimeMessageRole::Assistant, content)
}

fn user(id: &str, content: Vec<RealtimeContent>) -> RealtimeMessageItem {
    RealtimeMessageItem::new(id, RealtimeMessageRole::User, content)
}

fn text_of(item: &RealtimeItem) -> String {
    item.as_message().expect("message item").text()
}

fn ids(session: &Session) -> Vec<String> {
    session
        .history()
        .iter()
        .map(|item| item.item_id().to_string())
        .collect()
}

fn input_audio(audio: Option<&str>, transcript: Option<&str>) -> RealtimeContent {
    RealtimeContent::InputAudio {
        audio: audio.map(str::to_string),
        transcript: transcript.map(str::to_string),
        extra: Default::default(),
    }
}

fn output_audio(transcript: Option<&str>) -> RealtimeContent {
    RealtimeContent::Audio {
        audio: None,
        transcript: transcript.map(str::to_string),
        extra: Default::default(),
    }
}

fn tool_item(status: &str, output: Option<&str>) -> RealtimeToolCallItem {
    RealtimeToolCallItem {
        item_id: "fc_1".to_string(),
        previous_item_id: None,
        call_id: Some("call_1".to_string()),
        status: status.to_string(),
        arguments: r#"{"city": "Oakland"}"#.to_string(),
        name: "get_weather".to_string(),
        output: output.map(str::to_string),
        extra: Default::default(),
    }
}

fn transcription(item_id: &str, transcript: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::InputAudioTranscriptionCompleted {
        item_id: item_id.to_string(),
        transcript: transcript.to_string(),
    }
}

// ---- test_session_history.py -------------------------------------------------------------

#[tokio::test]
async fn transcription_completed_adds_new_user_item() {
    let (model, session) = started_fake().await;
    assert!(model.emit(transcription("item1", "hello")));
    assert_eq!(
        recv_types(&session, 2).await,
        ["raw_model_event", "history_added"]
    );
    let history = session.history();
    assert_eq!(history.len(), 1);
    let message = history[0].as_message().unwrap();
    assert_eq!(message.role, RealtimeMessageRole::User);
}

#[tokio::test]
async fn transcription_completed_event_updates_history() {
    let (model, session) = started_fake().await;
    let initial = user("item_1", vec![input_audio(None, None)]);
    assert_eq!(put(&model, &session, initial).await, "history_added");

    assert!(model.emit(transcription("item_1", "Hello world")));
    let raw = recv(&session).await;
    assert_eq!(raw.event_type(), "raw_model_event");
    match recv(&session).await {
        RealtimeSessionEvent::HistoryUpdated { history, .. } => assert_eq!(history.len(), 1),
        other => panic!("expected history_updated, got {}", other.event_type()),
    }
    let history = session.history();
    let message = history[0].as_message().unwrap();
    assert_eq!(message.content[0].transcript(), Some("Hello world"));
    assert_eq!(message.status.as_deref(), Some("completed"));
}

#[tokio::test]
async fn item_updated_event_updates_existing_item() {
    let (model, session) = started_fake().await;
    let initial = assistant("existing_item", vec![RealtimeContent::text("Initial")]);
    assert_eq!(put(&model, &session, initial).await, "history_added");
    let updated = assistant("existing_item", vec![RealtimeContent::text("Updated")]);
    assert_eq!(put(&model, &session, updated).await, "history_updated");
    assert_eq!(session.history().len(), 1);
    assert_eq!(text_of(&session.history()[0]), "Updated");
}

#[tokio::test]
async fn item_updated_event_completes_tool_call() {
    let (model, session) = started_fake().await;
    let in_progress = tool_item("in_progress", None);
    assert_eq!(
        put(&model, &session, RealtimeItem::FunctionCall(in_progress)).await,
        "history_added"
    );
    let completed = tool_item("completed", Some("sunny"));
    assert!(model.emit(RealtimeModelEvent::item_updated(
        RealtimeItem::FunctionCall(completed)
    )));
    assert_eq!(recv(&session).await.event_type(), "raw_model_event");
    match recv(&session).await {
        RealtimeSessionEvent::HistoryUpdated { history, .. } => {
            assert_eq!(
                history[0].as_function_call().unwrap().output.as_deref(),
                Some("sunny")
            );
        }
        other => panic!("expected history_updated, got {}", other.event_type()),
    }
    let history = session.history();
    assert_eq!(history.len(), 1);
    let stored = history[0].as_function_call().unwrap();
    assert_eq!(stored.status, "completed");
    assert_eq!(stored.output.as_deref(), Some("sunny"));
}

#[tokio::test]
async fn item_deleted_event_removes_item() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        assistant("item_1", vec![RealtimeContent::text("First")]),
    )
    .await;
    put(
        &model,
        &session,
        assistant("item_2", vec![RealtimeContent::text("Second")]),
    )
    .await;
    assert!(model.emit(RealtimeModelEvent::ItemDeleted {
        item_id: "item_1".to_string()
    }));
    assert_eq!(
        recv_types(&session, 2).await,
        ["raw_model_event", "history_updated"]
    );
    assert_eq!(ids(&session), ["item_2"]);
}

#[tokio::test]
async fn merge_transcript_into_existing_audio_message() {
    let (model, session) = started_fake().await;
    let initial = user(
        "item_1",
        vec![
            RealtimeContent::input_text("Before audio"),
            input_audio(Some("audio_data"), None),
            RealtimeContent::input_text("After audio"),
        ],
    );
    put(&model, &session, initial).await;
    assert!(model.emit(transcription("item_1", "Hello world")));
    recv_types(&session, 2).await;

    let history = session.history();
    assert_eq!(history.len(), 1);
    let message = history[0].as_message().unwrap();
    assert_eq!(message.status.as_deref(), Some("completed"));
    assert_eq!(message.content.len(), 3);
    assert_eq!(message.content[0].text_value(), Some("Before audio"));
    match &message.content[1] {
        RealtimeContent::InputAudio {
            audio, transcript, ..
        } => {
            assert_eq!(transcript.as_deref(), Some("Hello world"));
            assert_eq!(audio.as_deref(), Some("audio_data"));
        }
        other => panic!("expected input audio, got {other:?}"),
    }
    assert_eq!(message.content[2].text_value(), Some("After audio"));
}

#[tokio::test]
async fn merge_transcript_preserves_other_items() {
    let (model, session) = started_fake().await;
    let item1 = user("item_1", vec![RealtimeContent::input_text("First message")]);
    let item2 = user("item_2", vec![input_audio(None, None)]);
    let item3 = assistant("item_3", vec![RealtimeContent::text("Third message")]);
    put(&model, &session, item1.clone()).await;
    put(&model, &session, item2).await;
    put(&model, &session, item3.clone()).await;

    assert!(model.emit(transcription("item_2", "Transcribed audio")));
    recv_types(&session, 2).await;

    let history = session.history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0], RealtimeItem::Message(item1));
    assert_eq!(history[2], RealtimeItem::Message(item3));
    let second = history[1].as_message().unwrap();
    assert_eq!(second.content[0].transcript(), Some("Transcribed audio"));
    assert_eq!(second.status.as_deref(), Some("completed"));
}

#[tokio::test]
async fn merge_transcript_only_affects_audio_content() {
    let (model, session) = started_fake().await;
    let item = user(
        "item_1",
        vec![
            RealtimeContent::input_text("Text content"),
            input_audio(Some("audio1"), None),
            input_audio(Some("audio2"), Some("existing")),
            RealtimeContent::input_text("More text"),
        ],
    );
    put(&model, &session, item).await;
    assert!(model.emit(transcription("item_1", "New transcript")));
    recv_types(&session, 2).await;

    let history = session.history();
    let message = history[0].as_message().unwrap();
    assert_eq!(message.content[0].text_value(), Some("Text content"));
    assert_eq!(message.content[3].text_value(), Some("More text"));
    // Every audio part takes the new transcript, including one that already had one.
    assert_eq!(message.content[1].transcript(), Some("New transcript"));
    assert_eq!(message.content[2].transcript(), Some("New transcript"));
}

#[tokio::test]
async fn update_existing_item_by_id() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        assistant("item_1", vec![RealtimeContent::text("Original")]),
    )
    .await;
    put(
        &model,
        &session,
        assistant("item_1", vec![RealtimeContent::text("Updated")]),
    )
    .await;
    assert_eq!(ids(&session), ["item_1"]);
    assert_eq!(text_of(&session.history()[0]), "Updated");
}

#[tokio::test]
async fn update_existing_item_preserves_order() {
    let (model, session) = started_fake().await;
    for (id, text) in [
        ("item_1", "First"),
        ("item_2", "Second"),
        ("item_3", "Third"),
    ] {
        put(
            &model,
            &session,
            assistant(id, vec![RealtimeContent::text(text)]),
        )
        .await;
    }
    put(
        &model,
        &session,
        assistant("item_2", vec![RealtimeContent::text("Updated Second")]),
    )
    .await;
    assert_eq!(ids(&session), ["item_1", "item_2", "item_3"]);
    let history = session.history();
    assert_eq!(text_of(&history[0]), "First");
    assert_eq!(text_of(&history[1]), "Updated Second");
    assert_eq!(text_of(&history[2]), "Third");
}

#[tokio::test]
async fn insert_new_item_after_previous_item() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        assistant("item_1", vec![RealtimeContent::text("First")]),
    )
    .await;
    put(
        &model,
        &session,
        assistant("item_3", vec![RealtimeContent::text("Third")]),
    )
    .await;
    let mut second = assistant("item_2", vec![RealtimeContent::text("Second")]);
    second.previous_item_id = Some("item_1".to_string());
    put(&model, &session, second).await;
    assert_eq!(ids(&session), ["item_1", "item_2", "item_3"]);
    assert_eq!(text_of(&session.history()[1]), "Second");
}

#[tokio::test]
async fn insert_new_item_after_nonexistent_previous_item_appends() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        assistant("item_1", vec![RealtimeContent::text("First")]),
    )
    .await;
    let mut second = assistant("item_2", vec![RealtimeContent::text("Second")]);
    second.previous_item_id = Some("nonexistent".to_string());
    put(&model, &session, second).await;
    assert_eq!(ids(&session), ["item_1", "item_2"]);
}

#[tokio::test]
async fn add_new_item_to_end_when_no_previous_item_id() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        assistant("item_1", vec![RealtimeContent::text("First")]),
    )
    .await;
    put(
        &model,
        &session,
        assistant("item_2", vec![RealtimeContent::text("Second")]),
    )
    .await;
    assert_eq!(ids(&session), ["item_1", "item_2"]);
}

#[tokio::test]
async fn tool_call_item_update_replaces_existing_entry() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        RealtimeItem::FunctionCall(tool_item("in_progress", None)),
    )
    .await;
    put(
        &model,
        &session,
        RealtimeItem::FunctionCall(tool_item("completed", Some("sunny"))),
    )
    .await;
    let history = session.history();
    assert_eq!(history.len(), 1);
    let updated = history[0].as_function_call().unwrap();
    assert_eq!(updated.status, "completed");
    assert_eq!(updated.output.as_deref(), Some("sunny"));
}

#[tokio::test]
async fn tool_call_item_update_preserves_other_items() {
    let (model, session) = started_fake().await;
    let before = user(
        "item_0",
        vec![RealtimeContent::input_text("what's the weather?")],
    );
    let after = assistant("item_2", vec![RealtimeContent::text("It is sunny.")]);
    let mut in_progress = tool_item("in_progress", None);
    in_progress.item_id = "item_1".to_string();
    put(&model, &session, before.clone()).await;
    put(
        &model,
        &session,
        RealtimeItem::FunctionCall(in_progress.clone()),
    )
    .await;
    put(&model, &session, after.clone()).await;

    in_progress.status = "completed".to_string();
    in_progress.output = Some("sunny".to_string());
    put(&model, &session, RealtimeItem::FunctionCall(in_progress)).await;

    assert_eq!(ids(&session), ["item_0", "item_1", "item_2"]);
    let history = session.history();
    assert_eq!(history[0], RealtimeItem::Message(before));
    assert_eq!(history[2], RealtimeItem::Message(after));
    assert_eq!(
        history[1].as_function_call().unwrap().output.as_deref(),
        Some("sunny")
    );
}

#[tokio::test]
async fn add_first_item_to_empty_history() {
    let (model, session) = started_fake().await;
    assert!(session.history().is_empty());
    assert_eq!(
        put(
            &model,
            &session,
            assistant("item_1", vec![RealtimeContent::text("First")])
        )
        .await,
        "history_added"
    );
    assert_eq!(ids(&session), ["item_1"]);
}

#[tokio::test]
async fn complex_insertion_scenario() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        assistant("A", vec![RealtimeContent::text("A")]),
    )
    .await;
    put(
        &model,
        &session,
        assistant("C", vec![RealtimeContent::text("C")]),
    )
    .await;

    let mut b = assistant("B", vec![RealtimeContent::text("B")]);
    b.previous_item_id = Some("A".to_string());
    put(&model, &session, b).await;
    assert_eq!(ids(&session), ["A", "B", "C"]);

    let mut d = assistant("D", vec![RealtimeContent::text("D")]);
    d.previous_item_id = Some("B".to_string());
    put(&model, &session, d).await;
    assert_eq!(ids(&session), ["A", "B", "D", "C"]);

    put(
        &model,
        &session,
        assistant("B", vec![RealtimeContent::text("Updated B")]),
    )
    .await;
    assert_eq!(ids(&session), ["A", "B", "D", "C"]);
    assert_eq!(text_of(&session.history()[1]), "Updated B");
}

#[tokio::test]
async fn assistant_transcript_preserved_on_item_update() {
    let (model, session) = started_fake().await;
    put(
        &model,
        &session,
        assistant("assist_1", vec![output_audio(Some("Hello there"))]),
    )
    .await;
    put(
        &model,
        &session,
        assistant("assist_1", vec![output_audio(None)]),
    )
    .await;
    let history = session.history();
    assert_eq!(history.len(), 1);
    assert_eq!(
        history[0].as_message().unwrap().content[0].transcript(),
        Some("Hello there")
    );
}

#[tokio::test]
async fn assistant_transcript_can_fallback_to_deltas() {
    let (model, session) = started_fake().await;
    assert!(model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "assist_2".to_string(),
        delta: "partial transcript".to_string(),
        response_id: "resp_2".to_string(),
    }));
    assert_eq!(recv_types(&session, 1).await, ["raw_model_event"]);

    put(
        &model,
        &session,
        assistant("assist_2", vec![output_audio(None)]),
    )
    .await;
    put(
        &model,
        &session,
        assistant("assist_2", vec![output_audio(None)]),
    )
    .await;
    let history = session.history();
    assert_eq!(history.len(), 1);
    assert_eq!(
        history[0].as_message().unwrap().content[0].transcript(),
        Some("partial transcript")
    );
}

#[tokio::test]
async fn existing_transcript_not_overwritten_by_stale_deltas() {
    let (model, session) = started_fake().await;
    // A stale delta accumulator for the item, then the complete transcript arrives.
    assert!(model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "assist_3".to_string(),
        delta: "stale partial".to_string(),
        response_id: "resp_3".to_string(),
    }));
    recv_types(&session, 1).await;
    put(
        &model,
        &session,
        assistant(
            "assist_3",
            vec![output_audio(Some("Final complete transcript"))],
        ),
    )
    .await;
    put(
        &model,
        &session,
        assistant("assist_3", vec![output_audio(None)]),
    )
    .await;
    assert_eq!(
        session.history()[0].as_message().unwrap().content[0].transcript(),
        Some("Final complete transcript")
    );
}

// ---- test_session_close_iteration.py -----------------------------------------------------

#[derive(Debug)]
/// A transport whose `close` pauses until released, optionally failing its first attempt.
struct PausedCloseModel {
    inner: ScriptedRealtimeModel,
    started: Notify,
    release: Notify,
    attempts: AtomicUsize,
    fail_first: bool,
}

#[async_trait]
impl RealtimeModel for PausedCloseModel {
    async fn connect(
        &self,
        config: RealtimeModelConfig,
    ) -> openai_agents_sdk::Result<openai_agents_sdk::realtime::RealtimeModelEvents> {
        self.inner.connect(config).await
    }

    async fn send_event(&self, event: RealtimeModelSendEvent) -> openai_agents_sdk::Result<()> {
        self.inner.send_event(event).await
    }

    async fn close(&self) -> openai_agents_sdk::Result<()> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        self.started.notify_one();
        self.release.notified().await;
        if attempt == 1 && self.fail_first {
            return Err(AgentsError::other("transport close failed"));
        }
        self.inner.close().await
    }
}

async fn readers_finish_before_transport_cleanup(fail_first: bool) {
    let model = Arc::new(PausedCloseModel {
        inner: ScriptedRealtimeModel::new([]).lenient(),
        started: Notify::new(),
        release: Notify::new(),
        attempts: AtomicUsize::new(0),
        fail_first,
    });
    let session = start(model.clone()).await;
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let session = session.clone();
            tokio::spawn(async move { collect(&session).await })
        })
        .collect();
    let closer = {
        let session = session.clone();
        tokio::spawn(async move { session.close().await })
    };
    tokio::time::timeout(TIMEOUT, model.started.notified())
        .await
        .expect("close reached the transport");

    let mut results = Vec::new();
    for reader in readers {
        results.push(
            tokio::time::timeout(TIMEOUT, reader)
                .await
                .unwrap()
                .unwrap(),
        );
    }
    results.sort();
    assert_eq!(results, vec![vec![], vec!["history_updated"]]);
    assert!(!closer.is_finished(), "transport close is still paused");

    model.release.notify_one();
    let outcome = tokio::time::timeout(TIMEOUT, closer)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(outcome.is_err(), fail_first);
    assert!(model.attempts.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn readers_finish_before_transport_cleanup_when_close_succeeds() {
    readers_finish_before_transport_cleanup(false).await;
}

#[tokio::test]
async fn readers_finish_before_transport_cleanup_when_close_fails() {
    readers_finish_before_transport_cleanup(true).await;
}

#[tokio::test]
async fn reader_started_after_failed_close_receives_buffered_events() {
    let model = Arc::new(
        ScriptedRealtimeModel::new([])
            .lenient()
            .with_close_error("transport close failed"),
    );
    let session = start(model.clone()).await;
    assert!(model.emit(transcription("buffered", "text")));
    // The session runs on a single thread here, so once the history has changed the pump has
    // already queued the matching events.
    tokio::time::timeout(TIMEOUT, async {
        while session.history().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let error = session.close().await.expect_err("close failure surfaces");
    assert!(error.to_string().contains("transport close failed"));
    // Unlike Python the event is a history item rather than an `other` raw event: the Rust
    // pump needs an observable state change to know the event was handled without sleeping.
    assert_eq!(
        collect(&session).await,
        ["history_updated", "raw_model_event", "history_added"]
    );
    assert!(collect(&session).await.is_empty());
}

#[tokio::test]
async fn processing_reader_cannot_strand_another_reader_during_close() {
    let model = Arc::new(
        ScriptedRealtimeModel::new([])
            .lenient()
            .with_close_error("transport close failed"),
    );
    let session = start(model).await;
    let processing = Arc::new(Notify::new());
    let (resume_tx, resume_rx) = tokio::sync::oneshot::channel::<()>();
    let active = {
        let session = session.clone();
        let processing = processing.clone();
        tokio::spawn(async move {
            let mut types = Vec::new();
            if let Some(event) = session.recv().await {
                types.push(event.event_type());
                processing.notify_one();
                let _ = resume_rx.await;
            }
            while let Some(event) = session.recv().await {
                types.push(event.event_type());
            }
            types
        })
    };
    tokio::time::timeout(TIMEOUT, processing.notified())
        .await
        .unwrap();
    let parked = {
        let session = session.clone();
        tokio::spawn(async move { collect(&session).await })
    };
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    resume_tx.send(()).unwrap();
    assert!(session.close().await.is_err());
    let active = tokio::time::timeout(TIMEOUT, active)
        .await
        .unwrap()
        .unwrap();
    let parked = tokio::time::timeout(TIMEOUT, parked)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(active, ["history_updated"]);
    assert!(parked.is_empty());
}

// ---- test_session_exceptions.py ----------------------------------------------------------

fn exception(message: &str, context: &str) -> RealtimeModelEvent {
    RealtimeModelEvent::Exception {
        message: message.to_string(),
        context: Some(context.to_string()),
    }
}

fn scripted_with(events: Vec<RealtimeModelEvent>) -> Arc<ScriptedRealtimeModel> {
    Arc::new(
        ScriptedRealtimeModel::new([])
            .lenient()
            .with_connect_events(events),
    )
}

#[tokio::test]
async fn exception_is_stored_ends_the_session_and_cleans_up() {
    let model = scripted_with(vec![exception("Test error", "Test context")]);
    let session = start(model.clone()).await;
    let types = collect(&session).await;
    assert_eq!(types, ["history_updated", "raw_model_event", "error"]);
    assert!(session.is_closed());
    assert_eq!(model.close_calls(), 1);
    let error = session.take_error().expect("the failure is kept");
    assert!(
        error.to_string().contains("Test context: Test error"),
        "got {error}"
    );
}

#[tokio::test]
async fn exception_context_is_preserved_for_each_failure() {
    for (context, message) in [
        ("Failed to send audio", "Audio encoding failed"),
        ("WebSocket error in message listener", "Network error"),
        ("Failed to send event: response.create", "Socket closed"),
    ] {
        let model = scripted_with(vec![exception(message, context)]);
        let session = start(model).await;
        collect(&session).await;
        let error = session.take_error().expect("stored");
        assert!(error.to_string().contains(&format!("{context}: {message}")));
        assert!(session.is_closed());
    }
}

#[tokio::test]
async fn first_exception_wins_when_several_arrive() {
    let model = scripted_with(vec![
        exception("First error", "First context"),
        exception("Second error", "Second context"),
    ]);
    let session = start(model).await;
    collect(&session).await;
    let error = session.take_error().expect("stored");
    assert!(error.to_string().contains("First error"), "got {error}");
    assert!(session.is_closed());
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn exception_during_guardrail_processing_cancels_guardrail_tasks() {
    let started = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let guardrail = {
        let started = started.clone();
        let dropped = dropped.clone();
        OutputGuardrail::<()>::new("slow", move |_ctx, _agent, _output| {
            let started = started.clone();
            let flag = DropFlag(dropped.clone());
            async move {
                let _flag = flag;
                started.notify_one();
                std::future::pending::<()>().await;
                Ok(GuardrailFunctionOutput::new(Value::Null, false))
            }
        })
    };
    let config = RealtimeRunConfig::<()> {
        guardrails_settings: RealtimeGuardrailsSettings {
            debounce_text_length: 1,
        },
        ..Default::default()
    };
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start_with(
        model.clone(),
        config,
        RealtimeAgent::<()>::new("agent").with_output_guardrail(guardrail),
    )
    .await;

    assert!(model.emit(RealtimeModelEvent::TranscriptDelta {
        item_id: "item".to_string(),
        delta: "hello".to_string(),
        response_id: "resp".to_string(),
    }));
    tokio::time::timeout(TIMEOUT, started.notified())
        .await
        .unwrap();
    assert!(model.emit(exception("Processing error", "Processing failed")));

    collect(&session).await;
    assert!(
        session
            .take_error()
            .unwrap()
            .to_string()
            .contains("Processing error")
    );
    tokio::time::timeout(TIMEOUT, async {
        while !dropped.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the pending guardrail run was cancelled");
}

#[tokio::test]
async fn normal_events_still_work_before_exception() {
    let model = Arc::new(FakeRealtimeModel::new());
    let session = start(model.clone()).await;
    assert!(model.emit(RealtimeModelEvent::Error {
        error: json!({"message": "Normal error"}),
    }));
    assert!(model.emit(exception("Fatal error", "Fatal context")));

    let mut errors = Vec::new();
    tokio::time::timeout(TIMEOUT, async {
        while let Some(event) = session.recv().await {
            if let RealtimeSessionEvent::Error { error, .. } = event {
                errors.push(error);
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(errors[0], json!({"message": "Normal error"}));
    assert!(
        session
            .take_error()
            .unwrap()
            .to_string()
            .contains("Fatal error")
    );
}

// ---- test_session_tool_error_events.py ---------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct LookupArgs {
    value: String,
}

const ARGUMENT: &str = "synthetic-argument-secret";
const OUTPUT: &str = "synthetic-output-secret";
const SECRET: &str = "synthetic-exception-secret";

fn lookup_call() -> RealtimeModelToolCallEvent {
    RealtimeModelToolCallEvent {
        name: "lookup".to_string(),
        call_id: "synthetic-call-secret".to_string(),
        arguments: format!(r#"{{"value":"{ARGUMENT}"}}"#),
        id: None,
        previous_item_id: None,
    }
}

fn lookup_tool(
    started: Arc<Notify>,
    release: Arc<Notify>,
    invocations: Arc<AtomicUsize>,
    fail: bool,
) -> FunctionTool<()> {
    FunctionTool::new(
        "lookup",
        "look something up",
        move |_ctx, args: LookupArgs| {
            let started = started.clone();
            let release = release.clone();
            let invocations = invocations.clone();
            async move {
                invocations.fetch_add(1, Ordering::SeqCst);
                assert_eq!(args.value, ARGUMENT);
                started.notify_one();
                release.notified().await;
                if fail {
                    Err(AgentsError::other(SECRET))
                } else {
                    Ok(OUTPUT.to_string())
                }
            }
        },
    )
    .without_failure_error_function()
}

#[tokio::test]
async fn async_tool_failure_emits_an_error_event_and_ends_the_session() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let invocations = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedRealtimeModel::new([]).lenient());
    let agent = RealtimeAgent::<()>::new("test").with_tool(lookup_tool(
        started.clone(),
        release.clone(),
        invocations,
        true,
    ));
    let session = start_with(model.clone(), RealtimeRunConfig::default(), agent).await;

    assert!(model.emit(RealtimeModelEvent::FunctionCall(lookup_call())));
    tokio::time::timeout(TIMEOUT, started.notified())
        .await
        .unwrap();
    release.notify_one();

    let mut types = Vec::new();
    let mut error_message = None;
    tokio::time::timeout(TIMEOUT, async {
        while let Some(event) = session.recv().await {
            types.push(event.event_type());
            if let RealtimeSessionEvent::Error { error, .. } = &event {
                error_message = error["message"].as_str().map(str::to_string);
            }
        }
    })
    .await
    .unwrap();
    assert!(types.contains(&"tool_start"));
    assert!(!types.contains(&"tool_end"));
    // The event carries only Python's fixed message; the real error stays with `take_error`.
    assert_eq!(error_message.as_deref(), Some("Tool call task failed"));
    assert!(session.take_error().unwrap().to_string().contains(SECRET));
    assert!(
        model
            .sent_events()
            .iter()
            .all(|event| event.as_tool_output().is_none()),
        "a failed tool sends no output"
    );
}

#[tokio::test]
async fn tool_output_send_failure_is_retried_without_rerunning_the_tool() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let invocations = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedRealtimeModel::new([
        RealtimeStep::new("tool_output").failing(SECRET),
        RealtimeStep::new("tool_output"),
    ]));
    let agent = RealtimeAgent::<()>::new("test").with_tool(lookup_tool(
        started.clone(),
        release.clone(),
        invocations.clone(),
        false,
    ));
    let session = start_with(model.clone(), RealtimeRunConfig::default(), agent).await;

    assert!(model.emit(RealtimeModelEvent::FunctionCall(lookup_call())));
    tokio::time::timeout(TIMEOUT, started.notified())
        .await
        .unwrap();
    release.notify_one();

    loop {
        match recv(&session).await {
            RealtimeSessionEvent::Error { error, .. } => {
                assert_eq!(
                    error,
                    json!({"message": "Tool output send failed; cached output will be retried"})
                );
                for secret in [SECRET, ARGUMENT, OUTPUT, "synthetic-call-secret"] {
                    assert!(!error.to_string().contains(secret));
                }
                break;
            }
            RealtimeSessionEvent::ToolEnd { .. } => panic!("tool_end before the output landed"),
            _ => {}
        }
    }

    assert!(model.emit(RealtimeModelEvent::FunctionCall(lookup_call())));
    loop {
        if let RealtimeSessionEvent::ToolEnd { output, .. } = recv(&session).await {
            assert_eq!(output.to_model_string(), OUTPUT);
            break;
        }
    }
    assert_eq!(invocations.load(Ordering::SeqCst), 1);
    let outputs: Vec<String> = model
        .sent_events()
        .iter()
        .filter_map(|event| event.as_tool_output().map(|(_, out, _)| out.to_string()))
        .collect();
    assert_eq!(outputs, [OUTPUT, OUTPUT]);
    assert!(session.take_error().is_none());
    model.assert_complete().unwrap();
}
