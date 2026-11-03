//! Ports of the handoff, nested-history and handoff-filter tests in `tests/test_agent_runner.py`
//! (Python lines ~2457-3350) and their streamed twins in `tests/test_agent_runner_streamed.py`
//! (lines ~1109-1540). Tests parametrized on `streamed` (or duplicated as a streamed twin) run
//! both modes inside one Rust test.

#![allow(clippy::field_reassign_with_default)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::handoffs::Handoff;
use openai_agents_sdk::items::{OutputItem, RunInput, RunItemKind};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, InputItem, RunContext, RunState, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

fn tool_returning(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

fn approval_tool(name: &str) -> FunctionTool<()> {
    tool_returning(name, "ok").with_needs_approval(true)
}

fn transfer(target: &str) -> OutputItem {
    function_call(format!("transfer_to_{target}"), "{}")
}

fn reasoning(id: &str, text: &str) -> OutputItem {
    OutputItem::from_value(json!({"type": "reasoning", "id": id,
        "summary": [{"type": "summary_text", "text": text}]}))
}

async fn run_mode(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    options: RunOptions<()>,
    streamed: bool,
) -> Result<RunResult<()>, AgentsError> {
    if !streamed {
        return Runner::run_with_options(agent.clone(), input, options).await;
    }
    let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(ev) = events.next().await {
            if let Err(e) = ev {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream.into_result().expect("finished")),
    }
}

fn values(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|i| i.0.clone()).collect()
}

fn types(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .filter_map(|i| i["type"].as_str().map(str::to_string))
        .collect()
}

fn message_text(item: &Value) -> String {
    match &item["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<String>(),
        _ => String::new(),
    }
}

fn nested() -> RunConfig<()> {
    let mut c = RunConfig::default();
    c.nest_handoff_history = true;
    c
}

/// Records the model input of every call through `call_model_input_filter`.
fn capturing(config: &mut RunConfig<()>) -> Arc<Mutex<Vec<Vec<Value>>>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    config.call_model_input_filter = Some(Arc::new(move |data| {
        sink.lock().unwrap().push(values(&data.model_data.input));
        Box::pin(async move { Ok(data.model_data) })
    }));
    seen
}

// ------------------------------------------------------------ nested handoff history

/// Python: `test_nested_handoff_filters_model_input_but_preserves_session_items`.
#[tokio::test]
async fn nested_history_filters_model_input_but_keeps_session_items() {
    let model = setup();
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(delegate))
            .with_tool(tool_returning("some_function", "result")),
    );
    model.add_turn(vec![function_call("some_function", r#"{"a":"b"}"#)]);
    model.add_turn(vec![assistant_message("a_message"), transfer("delegate")]);
    model.add_turn(vec![assistant_message("done")]);
    let mut config = nested();
    let seen = capturing(&mut config);
    let session = Arc::new(InMemorySession::new("nested"));
    let result = Runner::run_with_options(
        triage,
        "user_message",
        RunOptions::new(())
            .run_config(config)
            .session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let seen = seen.lock().unwrap().clone();
    assert!(seen.len() >= 3);
    let handoff_types = types(&seen[2]);
    assert!(!handoff_types.contains(&"function_call".to_string()));
    assert!(!handoff_types.contains(&"function_call_output".to_string()));
    assert!(
        result
            .new_items
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
    );
    assert!(
        result
            .new_items
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::HandoffOutput { .. }))
    );
    let stored = values(&session.get_items(None).await.unwrap());
    assert!(types(&stored).contains(&"function_call_output".to_string()));
}

/// Python: `test_nested_handoff_filters_reasoning_items_from_model_input` and the streamed
/// `test_streamed_nested_handoff_filters_reasoning_items_from_model_input`.
#[tokio::test]
async fn nested_history_drops_reasoning_from_the_next_model_input() {
    for streamed in [false, true] {
        let model = setup();
        let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
        let triage = Arc::new(
            Agent::<()>::new("triage")
                .with_model_instance(model.clone())
                .with_handoff(handoff(delegate)),
        );
        model.add_turn(vec![
            reasoning("reasoning_1", "Thinking about a handoff."),
            transfer("delegate"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let mut config = nested();
        let seen = capturing(&mut config);
        let result = run_mode(
            &triage,
            "user_message",
            RunOptions::new(()).run_config(config),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        let seen = seen.lock().unwrap().clone();
        assert!(seen.len() >= 2);
        assert!(
            !types(&seen[1]).contains(&"reasoning".to_string()),
            "streamed={streamed}"
        );
    }
}

fn call_ids(items: &[Value], ty: &str) -> Vec<String> {
    items
        .iter()
        .filter(|i| i["type"] == ty)
        .filter_map(|i| i["call_id"].as_str().map(str::to_string))
        .collect()
}

/// Python: `test_resume_preserves_filtered_model_input_after_handoff` and the streamed
/// `test_streaming_resume_preserves_filtered_model_input_after_handoff`.
#[tokio::test]
async fn a_resume_keeps_the_filtered_model_input_after_a_handoff() {
    for streamed in [false, true] {
        let model = setup();
        let delegate = Arc::new(
            Agent::<()>::new("delegate")
                .with_model_instance(model.clone())
                .with_tool(approval_tool("approval_tool")),
        );
        let triage = Arc::new(
            Agent::<()>::new("triage")
                .with_model_instance(model.clone())
                .with_handoff(handoff(delegate))
                .with_tool(tool_returning("some_function", "result")),
        );
        model.add_turn(vec![function_call_with_id(
            "some_function",
            r#"{"a":"b"}"#,
            "triage-call",
        )]);
        model.add_turn(vec![assistant_message("a_message"), transfer("delegate")]);
        model.add_turn(vec![function_call_with_id(
            "approval_tool",
            "{}",
            "delegate-call",
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let mut config = nested();
        let seen = capturing(&mut config);
        let first = run_mode(
            &triage,
            "user_message",
            RunOptions::new(()).run_config(config),
            streamed,
        )
        .await
        .unwrap();
        assert!(!first.interruptions.is_empty());
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], false);
        let mut config = nested();
        let resumed_seen = capturing(&mut config);
        let resumed = run_mode(
            &triage,
            state,
            RunOptions::new(()).run_config(config),
            streamed,
        )
        .await
        .unwrap();
        let last = resumed_seen
            .lock()
            .unwrap()
            .last()
            .cloned()
            .or_else(|| seen.lock().unwrap().last().cloned())
            .unwrap();
        let calls = call_ids(&last, "function_call");
        let outputs = call_ids(&last, "function_call_output");
        assert!(
            !calls.contains(&"triage-call".into()),
            "streamed={streamed}"
        );
        assert!(!outputs.contains(&"triage-call".into()));
        assert!(calls.contains(&"delegate-call".into()), "{last:?}");
        assert!(outputs.contains(&"delegate-call".into()));
        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    }
}

/// Python: `test_resumed_state_updates_agent_after_handoff`.
#[tokio::test]
async fn a_resumed_state_follows_the_handoff_to_the_delegate() {
    let model = setup();
    let delegate = Arc::new(
        Agent::<()>::new("delegate")
            .with_model_instance(model.clone())
            .with_tool(approval_tool("delegate_tool")),
    );
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(delegate))
            .with_tool(approval_tool("triage_tool")),
    );
    model.add_turn(vec![function_call_with_id("triage_tool", "{}", "triage-1")]);
    model.add_turn(vec![assistant_message("handoff"), transfer("delegate")]);
    model.add_turn(vec![function_call_with_id(
        "delegate_tool",
        "{}",
        "delegate-1",
    )]);
    let first = Runner::run(triage.clone(), "user_message").await.unwrap();
    assert!(!first.interruptions.is_empty());
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let second = Runner::run(triage, state).await.unwrap();
    assert!(
        second
            .interruptions
            .iter()
            .any(|i| i.tool_name == "delegate_tool")
    );
    assert_eq!(second.last_agent.name, "delegate");
    assert_eq!(second.to_state().current_agent_name, "delegate");
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: String,
}

/// Python: `test_structured_output` (with `nest_handoff_history=True`; the plain structured
/// output half is `py_core_a_runner::structured_output_after_a_handoff`) and the streamed
/// `test_structured_output`.
#[tokio::test]
async fn structured_output_with_nested_history_replays_the_summary() {
    for streamed in [false, true] {
        let model = setup();
        let agent_1 = Arc::new(
            Agent::<()>::new("one")
                .with_tool(tool_returning("bar", "bar_result"))
                .with_output_type::<Foo>()
                .with_model_instance(model.clone()),
        );
        let agent_2 = Arc::new(
            Agent::<()>::new("two")
                .with_tool(tool_returning("foo", "foo_result"))
                .with_handoff(handoff(agent_1))
                .with_model_instance(model.clone()),
        );
        model.add_turn(vec![function_call_with_id(
            "foo",
            r#"{"bar":"baz"}"#,
            "call_foo",
        )]);
        model.add_turn(vec![assistant_message("a_message"), transfer("one")]);
        model.add_turn(vec![
            assistant_message(r#"{"bar":"preamble"}"#),
            function_call_with_id("bar", r#"{"bar":"baz"}"#, "call_bar"),
        ]);
        model.add_turn(vec![assistant_message(r#"{"bar":"baz"}"#)]);
        let result = run_mode(
            &agent_2,
            vec![
                InputItem::user("user_message"),
                InputItem::user("another_message"),
            ],
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(
            result.final_output_as::<Foo>().unwrap(),
            Foo { bar: "baz".into() }
        );
        assert_eq!(result.raw_responses.len(), 4);
        assert_eq!(result.last_agent.name, "one");
        assert_eq!(result.to_input_list().len(), 11, "streamed={streamed}");
        assert_eq!(
            result
                .to_input_list_with(openai_agents_sdk::result::ToInputListMode::Normalized)
                .len(),
            7,
            "streamed={streamed}"
        );
    }
}

/// Python: `test_opt_in_handoff_history_nested_and_filters_respected`.
#[tokio::test]
async fn opt_in_nested_history_respects_a_handoff_filter() {
    let model = setup();
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(delegate)),
    );
    model.add_turn(vec![
        assistant_message("triage summary"),
        transfer("delegate"),
    ]);
    model.add_turn(vec![assistant_message("resolution")]);
    let result = Runner::run_with_options(
        triage,
        "user_message",
        RunOptions::new(()).run_config(nested()),
    )
    .await
    .unwrap();
    let input = values(&result.input);
    assert_eq!(input.len(), 3, "{input:?}");
    assert_eq!(input[0]["role"], "assistant");
    let summary = input[0]["content"].as_str().unwrap();
    assert!(summary.contains("<CONVERSATION HISTORY>"));
    assert!(!summary.contains("triage summary"));
    assert!(summary.contains("user_message"));
    assert_eq!(message_text(&input[1]), "triage summary");
    assert!(message_text(&input[2]).contains("transfer_to_delegate"));
    let delegate_input = values(&model.calls().last().unwrap().input);
    assert_eq!(delegate_input.len(), 3);
    assert_eq!(message_text(&delegate_input[1]), "triage summary");

    let model = setup();
    let delegate = Arc::new(Agent::<()>::new("delegate").with_model_instance(model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(delegate).with_input_filter(|data| data)),
    );
    model.add_turn(vec![
        assistant_message("triage summary"),
        transfer("delegate"),
    ]);
    model.add_turn(vec![assistant_message("resolution")]);
    let filtered = Runner::run_with_options(
        triage,
        "user_message",
        RunOptions::new(()).run_config(nested()),
    )
    .await
    .unwrap();
    assert_eq!(
        values(&filtered.input),
        vec![json!({"role": "user", "content": "user_message"})]
    );
}

/// Python: `test_falsey_per_handoff_input_filter_takes_precedence[non_streamed|streamed]`: a
/// per-handoff filter replaces the run-level one (Rust has no "falsy callable").
#[tokio::test]
async fn a_per_handoff_filter_takes_precedence_over_the_run_filter() {
    for streamed in [false, true] {
        let triage_model = setup();
        let delegate_model = setup();
        let delegate =
            Arc::new(Agent::<()>::new("delegate").with_model_instance(delegate_model.clone()));
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        let triage = Arc::new(
            Agent::<()>::new("triage")
                .with_model_instance(triage_model.clone())
                .with_handoff(handoff(delegate).with_input_filter(move |data| {
                    c.fetch_add(1, Ordering::SeqCst);
                    data
                })),
        );
        triage_model.add_turn(vec![transfer("delegate")]);
        delegate_model.add_turn(vec![assistant_message("done")]);
        let mut config = nested();
        config.handoff_input_filter = Some(Arc::new(|_data| {
            panic!("The run-level filter must not replace the per-handoff filter")
        }));
        let result = run_mode(
            &triage,
            "user_message",
            RunOptions::new(()).run_config(config),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(
            values(&result.input),
            vec![json!({"role": "user", "content": "user_message"})]
        );
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}

/// Python: `test_opt_in_handoff_history_accumulates_across_multiple_handoffs`.
#[tokio::test]
async fn nested_history_accumulates_across_handoffs() {
    let (triage_model, delegate_model, closer_model) = (setup(), setup(), setup());
    let closer = Arc::new(Agent::<()>::new("closer").with_model_instance(closer_model.clone()));
    let delegate = Arc::new(
        Agent::<()>::new("delegate")
            .with_model_instance(delegate_model.clone())
            .with_handoff(handoff(closer)),
    );
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(triage_model.clone())
            .with_handoff(handoff(delegate)),
    );
    triage_model.add_turn(vec![
        assistant_message("triage summary"),
        transfer("delegate"),
    ]);
    delegate_model.add_turn(vec![
        assistant_message("delegate update"),
        transfer("closer"),
    ]);
    closer_model.add_turn(vec![assistant_message("resolution")]);
    let result = Runner::run_with_options(
        triage,
        "user_question",
        RunOptions::new(()).run_config(nested()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("resolution"));
    let closer_input = values(&closer_model.calls()[0].input);
    assert_eq!(closer_input[0]["role"], "assistant");
    let summary = closer_input[0]["content"].as_str().unwrap();
    assert_eq!(summary.matches("<CONVERSATION HISTORY>").count(), 1);
    assert!(summary.contains("triage summary"));
    assert!(!summary.contains("delegate update"));
    assert!(summary.contains("user_question"));
    assert_eq!(closer_input.len(), 3);
    assert_eq!(message_text(&closer_input[1]), "delegate update");
    assert!(message_text(&closer_input[2]).contains("transfer_to_closer"));
}

/// Python: `test_server_managed_handoff_history_auto_disables_with_warning`
/// (`streamed` x `nest_source`).
#[tokio::test]
async fn server_managed_history_disables_nested_handoff_history() {
    for streamed in [false, true] {
        for from_handoff in [false, true] {
            let (triage_model, delegate_model) = (setup(), setup());
            let delegate =
                Arc::new(Agent::<()>::new("delegate").with_model_instance(delegate_model.clone()));
            let mut h = handoff(delegate);
            let mut config = RunConfig::default();
            if from_handoff {
                h = h.with_nest_handoff_history(true);
            } else {
                config.nest_handoff_history = true;
            }
            let triage = Arc::new(
                Agent::<()>::new("triage")
                    .with_model_instance(triage_model.clone())
                    .with_handoff(h),
            );
            triage_model.add_turn(vec![
                assistant_message("triage summary"),
                transfer("delegate"),
            ]);
            delegate_model.add_turn(vec![assistant_message("done")]);
            let result = run_mode(
                &triage,
                "user_message",
                RunOptions::new(())
                    .run_config(config)
                    .auto_previous_response_id(true),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("done"));
            let delegate_input = values(&delegate_model.calls()[0].input);
            assert_eq!(delegate_input.len(), 1, "{delegate_input:?}");
            assert_eq!(delegate_input[0]["type"], "function_call_output");
            assert!(delegate_input[0]["output"].to_string().contains("delegate"));
        }
    }
}

/// Python: `test_server_managed_handoff_input_filters_still_raise` (`streamed` x
/// `filter_source`).
#[tokio::test]
async fn server_managed_history_rejects_handoff_input_filters() {
    for streamed in [false, true] {
        for from_handoff in [false, true] {
            let (triage_model, delegate_model) = (setup(), setup());
            let delegate =
                Arc::new(Agent::<()>::new("delegate").with_model_instance(delegate_model.clone()));
            let mut h = handoff(delegate);
            let mut config = RunConfig::default();
            if from_handoff {
                h = h.with_input_filter(|data| data);
            } else {
                config.handoff_input_filter =
                    Some(openai_agents_sdk::handoff_input_filter(|data| data));
            }
            let triage = Arc::new(
                Agent::<()>::new("triage")
                    .with_model_instance(triage_model.clone())
                    .with_handoff(h),
            );
            triage_model.add_turn(vec![
                assistant_message("triage summary"),
                transfer("delegate"),
            ]);
            delegate_model.add_turn(vec![assistant_message("done")]);
            let err = run_mode(
                &triage,
                "user_message",
                RunOptions::new(())
                    .run_config(config)
                    .auto_previous_response_id(true),
                streamed,
            )
            .await
            .unwrap_err();
            assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
            assert!(
                err.to_string()
                    .contains("Server-managed conversations do not support handoff input filters"),
                "{err}"
            );
            assert!(delegate_model.calls().is_empty());
        }
    }
}

// --------------------------------------------------------------- custom handoffs

fn custom_handoff(target: Arc<Agent<()>>) -> Handoff<()> {
    let name = target.name.clone();
    Handoff::custom(
        openai_agents_sdk::handoffs::default_tool_name(&name),
        openai_agents_sdk::handoffs::default_tool_description(&name, None),
        name,
        json!({}),
        move |_ctx, _input| {
            let target = target.clone();
            async move { Ok(target) }
        },
    )
}

/// Python: `test_async_input_filter_supported` (non-streamed and streamed): a custom handoff
/// with an input filter hands off.
#[tokio::test]
async fn a_custom_handoff_with_a_filter_hands_off() {
    for streamed in [false, true] {
        let model = setup();
        let agent_1 = Arc::new(Agent::<()>::new("one").with_model_instance(model.clone()));
        let agent_2 = Arc::new(
            Agent::<()>::new("two")
                .with_model_instance(model.clone())
                .with_handoff(custom_handoff(agent_1).with_input_filter(|data| data)),
        );
        model.add_turn(vec![
            assistant_message("1"),
            assistant_message("2"),
            transfer("one"),
        ]);
        model.add_turn(vec![assistant_message("last")]);
        let result = run_mode(&agent_2, "user_message", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("last"));
    }
}

/// Python: `test_handoff_on_input`, `test_async_handoff_on_input` (and their streamed twins).
#[tokio::test]
async fn on_handoff_receives_the_parsed_input() {
    for streamed in [false, true] {
        let model = setup();
        let agent_1 = Arc::new(Agent::<()>::new("one").with_model_instance(model.clone()));
        let seen = Arc::new(Mutex::new(None::<String>));
        let s = seen.clone();
        let agent_2 = Arc::new(
            Agent::<()>::new("two")
                .with_model_instance(model.clone())
                .with_handoff(handoff(agent_1).with_on_handoff_input(
                    move |_ctx: RunContext<()>, data: Foo| {
                        let s = s.clone();
                        async move {
                            *s.lock().unwrap() = Some(data.bar);
                            Ok(())
                        }
                    },
                )),
        );
        model.add_turn(vec![
            assistant_message("1"),
            assistant_message("2"),
            function_call("transfer_to_one", r#"{"bar":"test_input"}"#),
        ]);
        model.add_turn(vec![assistant_message("last")]);
        let result = run_mode(&agent_2, "user_message", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("last"));
        assert_eq!(seen.lock().unwrap().as_deref(), Some("test_input"));
    }
}

/// Python: `test_invalid_handoff_input_json_causes_error`.
#[tokio::test]
async fn invalid_handoff_input_json_is_a_model_behavior_error() {
    let target = Arc::new(Agent::<()>::new("test"));
    let h =
        handoff(target).with_on_handoff_input(|_ctx: RunContext<()>, _data: Foo| async { Ok(()) });
    for arguments in ["null", "invalid"] {
        let err = h
            .invoke(RunContext::new(()), arguments.to_string())
            .await
            .unwrap_err();
        assert!(
            matches!(err, AgentsError::ModelBehaviorError { .. }),
            "{arguments}: {err:?}"
        );
    }
}

// ----------------------------------------------------------- resumed handoff state

/// Python: `test_resumed_state_updates_agent_after_handoff` round trip through JSON.
#[tokio::test]
async fn a_serialized_state_resumes_on_the_delegate() {
    let model = setup();
    let delegate = Arc::new(
        Agent::<()>::new("delegate")
            .with_model_instance(model.clone())
            .with_tool(approval_tool("delegate_tool")),
    );
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(delegate)),
    );
    model.add_turn(vec![transfer("delegate")]);
    model.add_turn(vec![function_call_with_id(
        "delegate_tool",
        "{}",
        "delegate-1",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let first = Runner::run(triage.clone(), "go").await.unwrap();
    let mut state = RunState::from_json(first.to_state().to_json().unwrap()).unwrap();
    assert_eq!(state.current_agent_name, "delegate");
    state.approve(&first.interruptions[0], false);
    let resumed = Runner::run(triage, state).await.unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(resumed.last_agent.name, "delegate");
}
