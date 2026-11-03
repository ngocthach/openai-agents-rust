//! Parity port of `tests/memory/test_session.py` and
//! `tests/extensions/memory/test_async_sqlite_session.py` (both exercise the plain SQLite
//! session; Rust has a single `SqliteSession` covering the sync and async Python classes).
#![cfg(feature = "sqlite")]

use std::sync::Arc;

use openai_agents_sdk::memory::{Session, SessionSettings, SqliteSession};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, RunConfig, RunOptions, Runner};
use serde_json::json;

mod py_mcpmem_support;
use py_mcpmem_support::{FailingSession, item_text};

fn user(content: &str) -> InputItem {
    InputItem::user(content)
}

fn assistant(content: &str) -> InputItem {
    InputItem::from_value(json!({"role": "assistant", "content": content}))
}

fn contents(items: &[InputItem]) -> Vec<String> {
    items.iter().map(item_text).collect()
}

/// Insert a row that is not valid JSON, the way the Python tests do with a raw cursor.
fn insert_corrupt_row(db: &std::path::Path, session_id: &str) {
    let conn = rusqlite::Connection::open(db).unwrap();
    conn.execute(
        "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
        rusqlite::params![session_id, "not valid json {{{"],
    )
    .unwrap();
}

fn drop_table(db: &std::path::Path, table: &str) {
    let conn = rusqlite::Connection::open(db).unwrap();
    conn.execute_batch(&format!("DROP TABLE {table}")).unwrap();
}

// ---------------------------------------------------------------------------
// Runner integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn session_memory_basic_functionality() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let session =
        Arc::new(SqliteSession::open("test_session_123", dir.path().join("m.db")).unwrap());

    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    model.add_turn(vec![assistant_message("San Francisco")]);
    let result1 = Runner::run_with_options(
        &agent,
        "What city is the Golden Gate Bridge in?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(
        result1.final_output_text().as_deref(),
        Some("San Francisco")
    );

    model.add_turn(vec![assistant_message("California")]);
    let result2 = Runner::run_with_options(
        &agent,
        "What state is it in?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result2.final_output_text().as_deref(), Some("California"));

    // The second turn saw the stored conversation, not just the new message.
    let last = model.last_call().unwrap();
    assert!(last.input.len() > 1, "{:?}", last.input);
}

#[tokio::test]
async fn session_memory_disabled_without_a_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    model.add_turn(vec![assistant_message("Hello")]);
    let r1 = Runner::run(&agent, "Hi there").await.unwrap();
    assert_eq!(r1.final_output_text().as_deref(), Some("Hello"));

    model.add_turn(vec![assistant_message("I don't remember")]);
    let r2 = Runner::run(&agent, "Do you remember what I said?")
        .await
        .unwrap();
    assert_eq!(r2.final_output_text().as_deref(), Some("I don't remember"));

    assert_eq!(model.last_call().unwrap().input.len(), 1);
}

#[tokio::test]
async fn session_memory_keeps_different_sessions_apart() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    let session_1 = Arc::new(SqliteSession::open("session_1", &path).unwrap());
    let session_2 = Arc::new(SqliteSession::open("session_2", &path).unwrap());

    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    model.add_turn(vec![assistant_message("I like cats")]);
    Runner::run_with_options(
        &agent,
        "I like cats",
        RunOptions::new(()).session(session_1.clone()),
    )
    .await
    .unwrap();

    model.add_turn(vec![assistant_message("I like dogs")]);
    Runner::run_with_options(
        &agent,
        "I like dogs",
        RunOptions::new(()).session(session_2.clone()),
    )
    .await
    .unwrap();

    model.add_turn(vec![assistant_message("Yes, you mentioned cats")]);
    let r3 = Runner::run_with_options(
        &agent,
        "What did I say I like?",
        RunOptions::new(()).session(session_1.clone()),
    )
    .await
    .unwrap();
    assert_eq!(
        r3.final_output_text().as_deref(),
        Some("Yes, you mentioned cats")
    );

    let seen = contents(&model.last_call().unwrap().input);
    assert!(seen.iter().any(|c| c == "I like cats"), "{seen:?}");
    assert!(!seen.iter().any(|c| c == "I like dogs"), "{seen:?}");
}

#[tokio::test]
async fn session_memory_appends_list_input_by_default() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(SqliteSession::open("list_input", dir.path().join("m.db")).unwrap());
    session
        .add_items(vec![user("Earlier message"), assistant("Saved reply")])
        .await
        .unwrap();

    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    model.add_turn(vec![assistant_message("This should run")]);

    Runner::run_with_options(
        &agent,
        vec![user("Test message")],
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();

    assert_eq!(
        contents(&model.last_call().unwrap().input),
        vec!["Earlier message", "Saved reply", "Test message"]
    );
}

#[tokio::test]
async fn session_input_callback_prepares_the_model_input() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(SqliteSession::open("session_1", dir.path().join("m.db")).unwrap());
    session
        .add_items(vec![
            user("Hello there."),
            assistant("Hi, I'm here to assist you."),
        ])
        .await
        .unwrap();

    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    model.add_turn(vec![assistant_message("I'm gpt-4o")]);

    // Keep only the user messages of the stored history, then the new input.
    let run_config = RunConfig::<()>::default().with_session_input_callback(
        |history: Vec<InputItem>, new_input: Vec<InputItem>| async move {
            let mut out: Vec<InputItem> = history
                .into_iter()
                .filter(|i| i.as_value().get("role").and_then(|r| r.as_str()) == Some("user"))
                .collect();
            out.extend(new_input);
            Ok(out)
        },
    );

    Runner::run_with_options(
        &agent,
        vec![user("What your name?")],
        RunOptions::new(())
            .session(session.clone())
            .run_config(run_config),
    )
    .await
    .unwrap();

    assert_eq!(
        contents(&model.last_call().unwrap().input),
        vec!["Hello there.", "What your name?"]
    );
}

#[tokio::test]
async fn a_repeating_session_callback_does_not_grow_the_session() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(SqliteSession::open("session_repeat", dir.path().join("m.db")).unwrap());

    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    for turn in 0..3 {
        model.add_turn(vec![assistant_message(format!("assistant {turn}"))]);
        let run_config = RunConfig::<()>::default().with_session_input_callback(
            |history: Vec<InputItem>, new_input: Vec<InputItem>| async move {
                if history.is_empty() {
                    return Ok(new_input);
                }
                let mut out = history.clone();
                out.push(history[0].clone());
                out.extend(new_input);
                Ok(out)
            },
        );
        Runner::run_with_options(
            &agent,
            format!("user {turn}"),
            RunOptions::new(())
                .session(session.clone())
                .run_config(run_config),
        )
        .await
        .unwrap();
    }

    let stored = session.get_items(None).await.unwrap();
    let users: Vec<String> = stored
        .iter()
        .filter(|i| i.as_value().get("role").and_then(|r| r.as_str()) == Some("user"))
        .map(item_text)
        .collect();
    assert_eq!(users, vec!["user 0", "user 1", "user 2"]);
    assert_eq!(stored.len(), 6);
}

#[tokio::test]
async fn runner_session_settings_override_the_sessions_own_limit() {
    openai_agents_sdk::set_tracing_disabled(true);
    let dir = tempfile::tempdir().unwrap();
    let session = Arc::new(
        SqliteSession::open("runner_override_test", dir.path().join("m.db"))
            .unwrap()
            .with_settings(SessionSettings::with_limit(100)),
    );
    let history: Vec<InputItem> = (0..10).map(|i| user(&format!("Turn {i}"))).collect();
    session.add_items(history).await.unwrap();

    let model = Arc::new(FakeModel::new());
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    model.add_turn(vec![assistant_message("Got it")]);

    Runner::run_with_options(
        &agent,
        "New question",
        RunOptions::new(()).session(session.clone()).run_config(
            RunConfig::<()>::default().with_session_settings(SessionSettings::with_limit(2)),
        ),
    )
    .await
    .unwrap();

    let seen = contents(&model.last_call().unwrap().input);
    let history_items: Vec<&String> = seen.iter().filter(|c| *c != "New question").collect();
    assert_eq!(history_items.len(), 2, "{seen:?}");
}

#[tokio::test]
async fn a_failing_session_write_propagates_through_the_stream() {
    openai_agents_sdk::set_tracing_disabled(true);
    let session: Arc<dyn Session> =
        Arc::new(FailingSession::new("Simulated session.add_items failure"));

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("This should not be reached")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());

    let mut stream = Runner::run_streamed_with_options(
        &agent,
        "Hello",
        RunOptions::new(()).session(session.clone()),
    );
    let mut error: Option<String> = None;
    {
        use futures::StreamExt;
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e.to_string());
                break;
            }
        }
    }
    if error.is_none() {
        error = stream.finish().await.err().map(|e| e.to_string());
    }
    let error = error
        .or_else(|| stream.run_loop_error().map(str::to_string))
        .expect("the session failure must surface");
    assert!(
        error.contains("Simulated session.add_items failure"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// SqliteSession directly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sqlite_session_round_trips_and_clears() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("direct_test", dir.path().join("d.db")).unwrap();

    session
        .add_items(vec![user("Hello"), assistant("Hi there!")])
        .await
        .unwrap();
    let retrieved = session.get_items(None).await.unwrap();
    assert_eq!(retrieved.len(), 2);
    assert_eq!(
        retrieved[0].as_value().get("role").unwrap().as_str(),
        Some("user")
    );
    assert_eq!(item_text(&retrieved[0]), "Hello");
    assert_eq!(
        retrieved[1].as_value().get("role").unwrap().as_str(),
        Some("assistant")
    );
    assert_eq!(item_text(&retrieved[1]), "Hi there!");

    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn sqlite_session_pop_item() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("pop_test", dir.path().join("p.db")).unwrap();

    assert!(session.pop_item().await.unwrap().is_none());

    session
        .add_items(vec![
            user("Hello"),
            assistant("Hi there!"),
            user("How are you?"),
        ])
        .await
        .unwrap();
    assert_eq!(session.get_items(None).await.unwrap().len(), 3);

    let popped = session.pop_item().await.unwrap().unwrap();
    assert_eq!(item_text(&popped), "How are you?");
    let left = session.get_items(None).await.unwrap();
    assert_eq!(left.len(), 2);
    assert_eq!(item_text(left.last().unwrap()), "Hi there!");

    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "Hi there!"
    );
    assert_eq!(
        item_text(&session.pop_item().await.unwrap().unwrap()),
        "Hello"
    );
    assert!(session.pop_item().await.unwrap().is_none());
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn pop_item_only_affects_its_own_session() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.db");
    let session_1 = SqliteSession::open("session_1", &path).unwrap();
    let session_2 = SqliteSession::open("session_2", &path).unwrap();

    session_1
        .add_items(vec![user("Session 1 message")])
        .await
        .unwrap();
    session_2
        .add_items(vec![
            user("Session 2 message 1"),
            user("Session 2 message 2"),
        ])
        .await
        .unwrap();

    let popped = session_2.pop_item().await.unwrap().unwrap();
    assert_eq!(item_text(&popped), "Session 2 message 2");

    assert_eq!(
        contents(&session_1.get_items(None).await.unwrap()),
        vec!["Session 1 message"]
    );
    assert_eq!(
        contents(&session_2.get_items(None).await.unwrap()),
        vec!["Session 2 message 1"]
    );
}

#[tokio::test]
async fn pop_item_skips_a_corrupt_newest_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.db");
    let session = SqliteSession::open("pop_corrupt", &path).unwrap();
    session.add_items(vec![user("valid")]).await.unwrap();
    insert_corrupt_row(&path, "pop_corrupt");

    let popped = session.pop_item().await.unwrap().unwrap();
    assert_eq!(item_text(&popped), "valid");
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn pop_item_returns_none_after_dropping_only_corrupt_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.db");
    let session = SqliteSession::open("pop_only_corrupt", &path).unwrap();
    // The row references a session that does not exist yet, so create it first.
    session.add_items(vec![user("seed")]).await.unwrap();
    session.pop_item().await.unwrap();
    insert_corrupt_row(&path, "pop_only_corrupt");

    assert!(session.pop_item().await.unwrap().is_none());
    assert!(session.get_items(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn get_items_honours_an_explicit_limit() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("count_test", dir.path().join("l.db")).unwrap();
    session
        .add_items(vec![
            user("Message 1"),
            assistant("Response 1"),
            user("Message 2"),
            assistant("Response 2"),
            user("Message 3"),
            assistant("Response 3"),
        ])
        .await
        .unwrap();

    let all = session.get_items(None).await.unwrap();
    assert_eq!(all.len(), 6);
    assert_eq!(item_text(&all[0]), "Message 1");
    assert_eq!(item_text(all.last().unwrap()), "Response 3");

    assert_eq!(
        contents(&session.get_items(Some(2)).await.unwrap()),
        vec!["Message 3", "Response 3"]
    );
    assert_eq!(
        contents(&session.get_items(Some(4)).await.unwrap()),
        vec!["Message 2", "Response 2", "Message 3", "Response 3"]
    );
    assert_eq!(session.get_items(Some(10)).await.unwrap().len(), 6);
    assert!(session.get_items(Some(0)).await.unwrap().is_empty());
}

#[tokio::test]
async fn get_items_limit_counts_valid_items_past_corrupt_newest_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lc.db");
    let session = SqliteSession::open("limit_corrupt", &path).unwrap();
    session
        .add_items(vec![user("valid 0"), assistant("valid 1"), user("valid 2")])
        .await
        .unwrap();
    insert_corrupt_row(&path, "limit_corrupt");

    assert_eq!(
        contents(&session.get_items(Some(2)).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
}

#[tokio::test]
async fn session_settings_limit_also_counts_valid_items() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sl.db");
    let session = SqliteSession::open("settings_limit_corrupt", &path)
        .unwrap()
        .with_settings(SessionSettings::with_limit(2));
    session
        .add_items(vec![user("valid 0"), assistant("valid 1"), user("valid 2")])
        .await
        .unwrap();
    insert_corrupt_row(&path, "settings_limit_corrupt");

    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec!["valid 1", "valid 2"]
    );
}

#[tokio::test]
async fn get_items_uses_the_sessions_own_limit_and_an_explicit_one_wins() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("uses_settings_limit_test", dir.path().join("s.db"))
        .unwrap()
        .with_settings(SessionSettings::with_limit(3));
    let items: Vec<InputItem> = (0..5).map(|i| user(&format!("Message {i}"))).collect();
    session.add_items(items).await.unwrap();

    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec!["Message 2", "Message 3", "Message 4"]
    );
    assert_eq!(
        contents(&session.get_items(Some(2)).await.unwrap()),
        vec!["Message 3", "Message 4"]
    );
}

#[tokio::test]
async fn session_settings_resolve_prefers_the_override() {
    let base = SessionSettings::with_limit(100);
    let override_settings = SessionSettings::with_limit(50);

    assert_eq!(base.resolve(Some(&override_settings)).limit, Some(50));
    assert_eq!(base.limit, Some(100));
    assert_eq!(base.resolve(None).limit, Some(100));
}

#[tokio::test]
async fn sqlite_session_stores_unicode_content() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("unicode_test", dir.path().join("u.db")).unwrap();
    session
        .add_items(vec![user("こんにちは"), assistant("😊👍"), user("Привет")])
        .await
        .unwrap();

    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec!["こんにちは", "😊👍", "Привет"]
    );
}

#[tokio::test]
async fn sqlite_session_stores_special_characters_and_sql_keywords() {
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("special_chars_test", dir.path().join("s.db")).unwrap();
    let items = vec![
        user("O'Reilly"),
        assistant("DROP TABLE sessions;"),
        user("\"SELECT * FROM users WHERE name = \"admin\";\""),
        assistant("Robert'); DROP TABLE students;--"),
        user("Normal message"),
    ];
    session.add_items(items).await.unwrap();

    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec![
            "O'Reilly",
            "DROP TABLE sessions;",
            "\"SELECT * FROM users WHERE name = \"admin\";\"",
            "Robert'); DROP TABLE students;--",
            "Normal message",
        ]
    );
}

#[tokio::test]
async fn sqlite_session_survives_concurrent_writes() {
    let dir = tempfile::tempdir().unwrap();
    let session =
        Arc::new(SqliteSession::open("concurrent_test", dir.path().join("c.db")).unwrap());

    let mut handles = Vec::new();
    for i in 0..10 {
        let session = session.clone();
        handles.push(tokio::spawn(async move {
            session
                .add_items(vec![user(&format!("Message {i}"))])
                .await
                .unwrap();
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }

    let mut seen = contents(&session.get_items(None).await.unwrap());
    seen.sort();
    let mut expected: Vec<String> = (0..10).map(|i| format!("Message {i}")).collect();
    expected.sort();
    assert_eq!(seen, expected);
}

#[tokio::test]
async fn two_sessions_on_one_file_write_independently() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.db");
    let session_1 = Arc::new(SqliteSession::open("session_1", &path).unwrap());
    let session_2 = Arc::new(SqliteSession::open("session_2", &path).unwrap());

    let (a, b) = tokio::join!(
        session_1.add_items(vec![user("session_1")]),
        session_2.add_items(vec![user("session_2")])
    );
    a.unwrap();
    b.unwrap();

    assert_eq!(
        contents(&session_1.get_items(None).await.unwrap()),
        vec!["session_1"]
    );
    assert_eq!(
        contents(&session_2.get_items(None).await.unwrap()),
        vec!["session_2"]
    );
}

#[tokio::test]
async fn a_failed_mutation_leaves_the_session_usable_and_unlocked() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rollback.db");
    let session = SqliteSession::open("rollback_test", &path).unwrap();
    session.add_items(vec![user("kept")]).await.unwrap();

    // Dropping the message table makes the next write fail after the sessions-table upsert
    // already opened a write transaction, the way Python forces a mid-transaction failure.
    drop_table(&path, "agent_messages");
    assert!(session.add_items(vec![user("doomed")]).await.is_err());

    // An independent writer must be able to take the write lock immediately, which proves the
    // failed transaction was rolled back rather than left open.
    let probe = rusqlite::Connection::open(&path).unwrap();
    probe
        .busy_handler(None)
        .expect("busy handler must be clearable");
    probe
        .execute_batch("CREATE TABLE IF NOT EXISTS probe_lock (x INTEGER)")
        .expect("the write lock must be free after the failed mutation");
    // The message table is gone, so reads report the SQLite failure rather than silently
    // returning an empty history.
    assert!(session.get_items(None).await.is_err());
}

#[tokio::test]
async fn a_failed_clear_session_leaves_the_write_lock_free() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clear_rollback.db");
    let session = SqliteSession::open("clear_rollback", &path).unwrap();
    session.add_items(vec![user("kept")]).await.unwrap();

    drop_table(&path, "agent_sessions");
    assert!(session.clear_session().await.is_err());

    let probe = rusqlite::Connection::open(&path).unwrap();
    probe.busy_handler(None).unwrap();
    probe
        .execute_batch("CREATE TABLE IF NOT EXISTS probe_lock (x INTEGER)")
        .expect("the write lock must be free after the failed clear");
}

#[tokio::test]
async fn a_failed_pop_item_leaves_the_write_lock_free() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pop_rollback.db");
    let session = SqliteSession::open("pop_rollback", &path).unwrap();
    session.add_items(vec![user("kept")]).await.unwrap();

    drop_table(&path, "agent_messages");
    assert!(session.pop_item().await.is_err());

    let probe = rusqlite::Connection::open(&path).unwrap();
    probe.busy_handler(None).unwrap();
    probe
        .execute_batch("CREATE TABLE IF NOT EXISTS probe_lock (x INTEGER)")
        .expect("the write lock must be free after the failed pop");
}

#[tokio::test]
async fn a_rejected_pop_leaves_the_item_in_place() {
    // Port of Python's `_pop_item_with_validation`: an `Err` from the validator rolls the
    // removal back.
    let dir = tempfile::tempdir().unwrap();
    let session = SqliteSession::open("validated_pop", dir.path().join("v.db")).unwrap();
    session
        .add_items(vec![user("first"), user("second")])
        .await
        .unwrap();

    let rejected = session
        .pop_item_validated(Box::new(|_item| {
            Err(openai_agents_sdk::error::AgentsError::other("nope"))
        }))
        .await;
    assert!(rejected.is_err());
    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec!["first", "second"]
    );
}

#[tokio::test]
async fn in_memory_sqlite_sessions_are_independent_of_a_file() {
    let session = SqliteSession::in_memory("mem").unwrap();
    session.add_items(vec![user("only here")]).await.unwrap();
    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec!["only here"]
    );
}

#[tokio::test]
async fn custom_table_names_are_validated() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tables.db");
    let session = SqliteSession::with_tables("u1", &path, "my_sessions", "my_messages").unwrap();
    session.add_items(vec![user("hi")]).await.unwrap();
    assert_eq!(
        contents(&session.get_items(None).await.unwrap()),
        vec!["hi"]
    );

    let bad = SqliteSession::with_tables("u1", &path, "ok", "bad table");
    assert!(bad.is_err());
}
