//! The sandbox `process_context` chain rewrites the run's original input, as Python's per-turn
//! `SandboxRuntime.prepare_agent` does (`run.py` and `run_loop.py` write `prepared.input` back
//! into `original_input`): the rewrite reaches `RunResult::input` and `to_input_list`, the items
//! the run generated are never passed through the chain, the chain runs once per turn, and a
//! plain-text input is left alone (`prepare_sandbox_input` returns a string unchanged).
#![cfg(feature = "sandbox")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::sandbox::capabilities::{Capability, Compaction, Shell};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use serde_json::{Value, json};

fn client_config() -> SandboxRunConfig {
    SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
}

fn shell_call(call_id: &str, commands: &[&str]) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": commands}
    }))
}

fn user(text: &str) -> InputItem {
    InputItem::from_value(json!({"role": "user", "content": text}))
}

/// Appends `process_calls=N` to the context. Mirrors `_ProcessContextSessionCapability`.
#[derive(Debug, Default)]
struct ProcessCalls(Arc<AtomicUsize>);

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ProcessCalls {
    fn capability_type(&self) -> &str {
        "process-context-session"
    }

    fn process_context(&self, mut context: Vec<InputItem>) -> Vec<InputItem> {
        let calls = self.0.fetch_add(1, Ordering::SeqCst) + 1;
        context.push(user(&format!("process_calls={calls}")));
        context
    }
}

fn two_turn_agent(model: &Arc<FakeModel>, counter: &Arc<AtomicUsize>) -> SandboxAgent<()> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    model.add_turn(vec![shell_call("c1", &["echo one"])]);
    model.add_turn(vec![assistant_message("done")]);
    SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![
            Arc::new(Shell::new().with_pty(false)),
            Arc::new(ProcessCalls(counter.clone())),
        ])
}

fn values(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|item| item.0.clone()).collect()
}

/// What both run modes must show after the two-turn run.
fn assert_rewritten_history(
    model: &FakeModel,
    counter: &AtomicUsize,
    input: &[InputItem],
    replay: &[InputItem],
) {
    let expected_input = vec![
        user("hello"),
        user("process_calls=1"),
        user("process_calls=2"),
    ];
    // Once per turn, never on generated items.
    assert_eq!(counter.load(Ordering::SeqCst), 2);
    assert_eq!(values(input), values(&expected_input));

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(values(&calls[0].input), values(&expected_input[..2]));
    let second = values(&calls[1].input);
    assert_eq!(second[..3], values(&expected_input)[..], "{second:?}");
    let generated: Vec<&str> = second[3..]
        .iter()
        .map(|item| item["type"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(generated, ["shell_call", "shell_call_output"], "{second:?}");

    let replay = values(replay);
    assert_eq!(replay[..3], values(&expected_input)[..], "{replay:?}");
    assert_eq!(
        replay
            .iter()
            .filter(|item| item["content"]
                .as_str()
                .is_some_and(|c| c.starts_with("process_calls=")))
            .count(),
        2,
        "{replay:?}"
    );
    assert!(
        replay
            .iter()
            .any(|item| item["type"] == "shell_call_output"),
        "{replay:?}"
    );
}

#[tokio::test]
async fn the_rewrite_is_kept_in_the_result_input_and_replay() {
    let model = Arc::new(FakeModel::new());
    let counter = Arc::new(AtomicUsize::new(0));
    let agent = two_turn_agent(&model, &counter);

    let result = SandboxRunner::run(&agent, vec![user("hello")], &client_config())
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_rewritten_history(&model, &counter, &result.input, &result.to_input_list());
}

#[tokio::test]
async fn a_streamed_run_keeps_the_rewrite_in_the_result_input_and_replay() {
    let model = Arc::new(FakeModel::new());
    let counter = Arc::new(AtomicUsize::new(0));
    let agent = two_turn_agent(&model, &counter);

    let mut run = SandboxRunner::run_streamed(&agent, vec![user("hello")], &client_config())
        .await
        .expect("the stream starts");
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while let Some(event) = stream.next().await {
            event.expect("stream event");
        }
    }

    let result = run.streaming.result().expect("the run completed");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_rewritten_history(&model, &counter, &result.input, &result.to_input_list());
}

/// `prepare_sandbox_input` returns a string input unchanged, so a run started from text never
/// passes it through the chain.
#[tokio::test]
async fn a_text_input_is_not_rewritten() {
    let model = Arc::new(FakeModel::new());
    let counter = Arc::new(AtomicUsize::new(0));
    let agent = two_turn_agent(&model, &counter);

    let result = SandboxRunner::run(&agent, "hello", &client_config())
        .await
        .expect("the run completes");

    assert_eq!(counter.load(Ordering::SeqCst), 0);
    assert_eq!(values(&result.input), values(&[user("hello")]));
    assert_eq!(
        values(&model.calls()[1].input[..1]),
        values(&[user("hello")])
    );
}

/// `Compaction.process_context` keeps the input from its last compaction item. Python applies it
/// to the run's original input only: an input that carries a compaction item is truncated (and
/// stays truncated in `RunResult::input`), while a compaction item the model returns during the
/// run is a generated item and does not truncate the next turn's input.
#[tokio::test]
async fn compaction_truncates_the_original_input_but_not_generated_items() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        shell_call("c1", &["echo one"]),
        OutputItem::from_value(json!({"type": "compaction", "id": "cmp_2"})),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![
            Arc::new(Shell::new().with_pty(false)),
            Arc::new(Compaction::new()),
        ]);
    let input = vec![
        user("before"),
        InputItem::from_value(json!({"type": "compaction", "id": "cmp_1"})),
        user("after"),
    ];

    let result = SandboxRunner::run(&agent, input, &client_config())
        .await
        .expect("the run completes");

    let kept = vec![
        InputItem::from_value(json!({"type": "compaction", "id": "cmp_1"})),
        user("after"),
    ];
    assert_eq!(values(&result.input), values(&kept));
    let first = values(&model.calls()[0].input);
    assert_eq!(first, values(&kept));
    let second = values(&model.calls()[1].input);
    assert_eq!(second[..2], values(&kept)[..], "{second:?}");
    assert!(
        second.iter().any(|item| item["id"] == "cmp_2")
            && second.iter().any(|item| item["type"] == "shell_call"),
        "{second:?}"
    );
}
