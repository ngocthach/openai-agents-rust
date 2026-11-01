//! Ports of `test_agent_runner.py` `prepare_input_with_session` cases the earlier round filed as
//! "Python object identity". The three below only depend on content and frequency, which is
//! how a Rust callback (it receives owned copies) is matched back to history and new input:
//! a history item repeated by the callback is still history, and an item equal to the new input
//! stays new.
//!
//! Not portable: `callback_replaces_history_item` and `extra_reconstructed_history_item_stays_new`
//! tell two equal-looking items apart by whether the callback mutated the list it received in
//! place or built a copy, which an owned `Vec` cannot express.

use std::sync::Arc;

use openai_agents_sdk::items::RunInput;
use openai_agents_sdk::memory::{InMemorySession, Session, SessionInputCallback};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{Agent, InputItem, Runner};
use serde_json::{Value, json};

fn user(content: &str) -> Value {
    json!({"role": "user", "content": content})
}

fn contents(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .map(|i| i["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn values(items: Vec<InputItem>) -> Vec<Value> {
    items.into_iter().map(|i| i.0).collect()
}

fn callback<F>(f: F) -> SessionInputCallback
where
    F: Fn(Vec<Value>, Vec<Value>) -> Vec<Value> + Send + Sync + 'static,
{
    Arc::new(move |history: Vec<InputItem>, new: Vec<InputItem>| {
        let out: Vec<InputItem> = f(values(history), values(new))
            .into_iter()
            .map(InputItem)
            .collect();
        Box::pin(async move { Ok(out) })
    })
}

/// Returns (what the model received, what the run saved for its input).
async fn prepare(
    history: Vec<Value>,
    input: impl Into<RunInput>,
    cb: SessionInputCallback,
) -> (Vec<Value>, Vec<Value>) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("reply")]);
    let session = Arc::new(InMemorySession::new("list"));
    session
        .add_items(history.iter().cloned().map(InputItem).collect())
        .await
        .unwrap();
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let config = RunConfig {
        session_input_callback: Some(cb),
        ..RunConfig::default()
    };
    Runner::run_with_options(
        &agent,
        input,
        RunOptions::new(())
            .session(session.clone())
            .run_config(config),
    )
    .await
    .unwrap();
    let all = values(session.get_items(Some(usize::MAX)).await.unwrap());
    let mut saved = all[history.len()..].to_vec();
    assert_eq!(
        saved.pop().expect("the reply is saved")["role"],
        "assistant"
    );
    (values(model.calls()[0].input.clone()), saved)
}

/// Python: `test_prepare_input_with_session_repeated_history_keeps_equal_new_item`.
#[tokio::test]
async fn a_repeated_history_item_does_not_hide_an_equal_new_item() {
    let cb = callback(|h, n| vec![h[0].clone(), h[0].clone(), n[0].clone()]);
    let (model_input, saved) = prepare(vec![user("same")], "same", cb).await;
    assert_eq!(contents(&model_input), ["same", "same", "same"]);
    assert_eq!(contents(&saved), ["same"]);
}

/// Python: `test_prepare_input_with_session_async_callback_moves_repeated_history_item`.
#[tokio::test]
async fn an_async_callback_may_repeat_a_history_item_around_the_new_input() {
    let cb: SessionInputCallback = Arc::new(|mut history: Vec<InputItem>, new: Vec<InputItem>| {
        Box::pin(async move {
            tokio::task::yield_now().await;
            let moved = history.remove(0);
            Ok(vec![moved.clone(), new[0].clone(), moved])
        })
    });
    let (model_input, saved) = prepare(vec![user("history")], "new", cb).await;
    assert_eq!(contents(&model_input), ["history", "new", "history"]);
    assert_eq!(contents(&saved), ["new"]);
}

/// Python: `test_prepare_input_with_session_history_moved_to_new_input_stays_history`.
#[tokio::test]
async fn history_moved_into_the_new_input_stays_history() {
    let cb = callback(|h, mut n| {
        let moved = h[0].clone();
        n.insert(0, moved.clone());
        n.push(moved);
        n
    });
    let (model_input, saved) = prepare(vec![user("history")], "new", cb).await;
    assert_eq!(contents(&model_input), ["history", "new", "history"]);
    assert_eq!(contents(&saved), ["new"]);
}
