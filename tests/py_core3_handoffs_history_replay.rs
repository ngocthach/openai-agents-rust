//! Ports of the replay and resume cases of the Python SDK's
//! `tests/test_handoff_history_duplication.py`: a message forwarded by a nested handoff must be
//! replayed once, whatever filters, copies, interruptions and restores happen around it.
//!
//! Python checks ownership through private attributes (`_nested_history_owned_session_item_refs`,
//! `_get_nested_history_owned_items`). Those are not reachable here, so each case asserts the
//! replayed input (`to_input_list`, `result.input`, the next model call) instead.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use openai_agents_sdk::handoffs::{
    Handoff, HandoffHistoryMapper, HandoffInputData, default_handoff_history_mapper, handoff,
};
use openai_agents_sdk::items::{
    InputItem, MessageOutput, OutputItem, RunInput, RunItem, RunItemKind,
};
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::{RunResult, ToInputListMode};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, RunContext, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn text(t: &str) -> OutputItem {
    assistant_message(t)
}

fn transfer(target: &str) -> OutputItem {
    function_call(format!("transfer_to_{target}"), "{}")
}

fn nested() -> RunConfig<()> {
    RunConfig::<()>::default().with_nest_handoff_history(true)
}

/// The text of an input item, joining its text parts as Python's `_input_item_text` does.
fn item_text(item: &InputItem) -> String {
    item.text_content().unwrap_or_default()
}

/// How many items carry exactly `t` as their text.
fn count(items: &[InputItem], t: &str) -> usize {
    items.iter().filter(|i| item_text(i) == t).count()
}

fn types(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|i| i.0["type"].as_str().map(str::to_string))
        .collect()
}

fn values(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|i| i.0.clone()).collect()
}

/// Runs `agent` directly or streamed and returns the completed result.
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
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream.into_result().expect("finished")),
    }
}

/// `first` says `message` and hands off to `second`, which answers `done`. `configure` edits
/// the handoff before it is attached to `first`.
struct Chain {
    first: Arc<Agent<()>>,
    second_model: Arc<FakeModel>,
}

fn chain(message: &str, configure: impl FnOnce(Handoff<()>) -> Handoff<()>) -> Chain {
    let first_model = setup();
    let second_model = setup();
    second_model.add_turn(vec![text("done")]);
    let second = Arc::new(Agent::<()>::new("second").with_model_instance(second_model.clone()));
    first_model.add_turn(vec![text(message), transfer("second")]);
    let first = Arc::new(
        Agent::<()>::new("first")
            .with_model_instance(first_model.clone())
            .with_handoff(configure(handoff(second))),
    );
    Chain {
        first,
        second_model,
    }
}

/// Runs the chain with nested history, directly or streamed, and returns the result.
async fn run_nested(chain: &Chain, streamed: bool) -> RunResult<()> {
    run_mode(
        &chain.first,
        "start",
        RunOptions::new(()).run_config(nested()),
        streamed,
    )
    .await
    .unwrap()
}

// ------------------------------------------------------------------- replay after one handoff

/// Python: `test_to_input_list_normalized_uses_filtered_continuation_after_nested_handoff`.
#[tokio::test]
async fn normalized_replay_uses_the_filtered_continuation_after_a_nested_handoff() {
    let triage_model = setup();
    let delegate_model = setup();
    let delegate =
        Arc::new(Agent::<()>::new("delegate").with_model_instance(delegate_model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(triage_model.clone())
            .with_handoff(delegate.clone()),
    );
    triage_model.add_turn(vec![text("triage summary"), transfer("delegate")]);
    delegate_model.add_turn(vec![text("resolution")]);
    delegate_model.add_turn(vec![text("followup answer")]);

    let result = run_mode(
        &triage,
        "user_question",
        RunOptions::new(()).run_config(nested()),
        false,
    )
    .await
    .unwrap();
    let preserve_all = result.to_input_list();
    let normalized = result.to_input_list_with(ToInputListMode::Normalized);

    assert_eq!(preserve_all.len(), 6);
    assert!(types(&preserve_all).contains(&"function_call".to_string()));
    assert!(types(&preserve_all).contains(&"function_call_output".to_string()));
    assert_eq!(count(&preserve_all, "triage summary"), 1);
    assert_eq!(normalized.len(), 4);
    assert!(!types(&normalized).contains(&"function_call".to_string()));
    assert!(!types(&normalized).contains(&"function_call_output".to_string()));

    let replay_model = setup();
    replay_model.add_turn(vec![text("replayed")]);
    let replay_agent =
        Arc::new(Agent::<()>::new("replay").with_model_instance(replay_model.clone()));
    let replayed = run_mode(&replay_agent, preserve_all, RunOptions::new(()), false)
        .await
        .unwrap();
    let replay_input = replay_model.calls()[0].input.clone();
    assert_eq!(count(&replay_input, "triage summary"), 1);
    assert_eq!(replayed.final_output_text().as_deref(), Some("replayed"));

    let mut follow_up = normalized.clone();
    follow_up.push(InputItem(json!({"role": "user", "content": "follow up?"})));
    let follow = run_mode(&delegate, follow_up.clone(), RunOptions::new(()), false)
        .await
        .unwrap();
    assert_eq!(
        follow.final_output_text().as_deref(),
        Some("followup answer")
    );
    assert_eq!(delegate_model.calls().last().unwrap().input, follow_up);
}

/// Python: `test_to_input_list_normalized_keeps_delegate_tool_items_after_nested_handoff`.
#[tokio::test]
async fn normalized_replay_keeps_the_delegate_tool_items() {
    #[derive(Deserialize, JsonSchema)]
    struct City {
        city: String,
    }
    let lookup = FunctionTool::new("lookup_weather", "weather", |_ctx, a: City| async move {
        Ok(format!("weather:{}", a.city))
    });
    let triage_model = setup();
    let delegate_model = setup();
    let delegate = Arc::new(
        Agent::<()>::new("delegate")
            .with_model_instance(delegate_model.clone())
            .with_tool(lookup),
    );
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(triage_model.clone())
            .with_handoff(delegate),
    );
    triage_model.add_turn(vec![text("triage summary"), transfer("delegate")]);
    delegate_model.add_turn(vec![
        text("delegate preamble"),
        function_call("lookup_weather", json!({"city": "Tokyo"}).to_string()),
    ]);
    delegate_model.add_turn(vec![text("resolution")]);

    let result = run_mode(
        &triage,
        "user_question",
        RunOptions::new(()).run_config(nested()),
        false,
    )
    .await
    .unwrap();
    let preserve_all = values(&result.to_input_list());
    let normalized = values(&result.to_input_list_with(ToInputListMode::Normalized));
    let calls = |items: &[Value]| {
        items
            .iter()
            .filter(|i| i["type"] == "function_call")
            .cloned()
            .collect::<Vec<_>>()
    };
    let outputs = |items: &[Value]| {
        items
            .iter()
            .filter(|i| i["type"] == "function_call_output")
            .cloned()
            .collect::<Vec<_>>()
    };

    assert_eq!(calls(&preserve_all).len(), 2);
    assert_eq!(outputs(&preserve_all).len(), 2);
    assert_eq!(calls(&normalized).len(), 1);
    assert_eq!(calls(&normalized)[0]["name"], "lookup_weather");
    assert_eq!(outputs(&normalized).len(), 1);
    assert_eq!(outputs(&normalized)[0]["output"], "weather:Tokyo");
}

/// Python: `test_to_input_list_normalized_uses_custom_filter_input_items`.
#[tokio::test]
async fn normalized_replay_uses_custom_filter_input_items() {
    let keep_messages_only = |data: HandoffInputData| HandoffInputData {
        input_items: Some(
            data.new_items
                .iter()
                .filter(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }))
                .cloned()
                .collect(),
        ),
        ..data
    };
    let triage_model = setup();
    let delegate_model = setup();
    let delegate =
        Arc::new(Agent::<()>::new("delegate").with_model_instance(delegate_model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(triage_model.clone())
            .with_handoff(handoff(delegate).with_input_filter(keep_messages_only)),
    );
    triage_model.add_turn(vec![text("triage summary"), transfer("delegate")]);
    delegate_model.add_turn(vec![text("resolution")]);

    let result = run_mode(&triage, "user_question", RunOptions::new(()), false)
        .await
        .unwrap();
    let preserve_all = result.to_input_list();
    let normalized = result.to_input_list_with(ToInputListMode::Normalized);
    assert_eq!(preserve_all.len(), 5);
    assert!(types(&preserve_all).contains(&"function_call".to_string()));
    assert!(types(&preserve_all).contains(&"function_call_output".to_string()));
    assert_eq!(normalized.len(), 3);
    assert!(!types(&normalized).contains(&"function_call".to_string()));
    assert!(!types(&normalized).contains(&"function_call_output".to_string()));
}

/// Python: `test_non_nested_filtered_handoff_does_not_add_occurrence_lineage`. A filtered
/// handoff that does not nest leaves its message to be replayed exactly once.
#[tokio::test]
async fn non_nested_filtered_handoff_replays_its_message_once() {
    for streamed in [false, true] {
        let chain = chain("same", |h| {
            h.with_input_filter(|data: HandoffInputData| data)
        });
        let result = run_mode(&chain.first, "start", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "same"),
            1,
            "streamed={streamed}"
        );
        assert_eq!(chain.second_model.calls().len(), 1);
    }
}

/// Python: `test_custom_filter_summary_shape_does_not_claim_equal_session_item`. A filter whose
/// summary payloads happen to look like the session items does not change the replay.
#[tokio::test]
async fn custom_filter_summary_shape_does_not_claim_an_equal_session_item() {
    let chain = chain("same", |h| {
        h.with_input_filter(|data: HandoffInputData| {
            let raw_message = data.new_items[0].to_input_item();
            let first_summary = default_handoff_history_mapper(vec![InputItem(
                json!({"role": "user", "content": "custom first"}),
            )]);
            let second_summary = default_handoff_history_mapper(vec![InputItem(
                json!({"role": "user", "content": "custom second"}),
            )]);
            let mut input_history = first_summary;
            input_history.push(raw_message);
            input_history.extend(second_summary);
            HandoffInputData {
                input_history,
                pre_handoff_items: Vec::new(),
                input_items: Some(Vec::new()),
                ..data
            }
        })
    });
    let result = run_nested(&chain, false).await;
    assert_eq!(count(&result.to_input_list(), "same"), 2);
}

/// Python: `test_nested_history_retains_forwarded_pre_handoff_item_provenance`. Tool search
/// items from an earlier turn are forwarded once and survive a state round trip.
#[tokio::test]
async fn forwarded_pre_handoff_items_keep_their_provenance() {
    for streamed in [false, true] {
        let first_model = setup();
        let second_model = setup();
        second_model.add_turn(vec![text("done")]);
        let second = Arc::new(Agent::<()>::new("second").with_model_instance(second_model.clone()));
        let first = Arc::new(
            Agent::<()>::new("first")
                .with_model_instance(first_model.clone())
                .with_handoff(second.clone()),
        );
        first_model.add_turn(vec![
            OutputItem::from_value(json!({
                "type": "tool_search_call", "id": "tool_search_call", "call_id": "search",
                "arguments": {"query": "profile"}, "execution": "server", "status": "completed",
            })),
            OutputItem::from_value(json!({
                "type": "tool_search_output", "id": "tool_search_output", "call_id": "search",
                "execution": "server", "status": "completed", "tools": [],
            })),
        ]);
        first_model.add_turn(vec![transfer("second")]);

        let result = run_mode(
            &first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        let replay = result.to_input_list();
        assert_eq!(
            replay
                .iter()
                .filter(|i| i.0["type"] == "tool_search_call")
                .count(),
            1,
            "streamed={streamed}"
        );
        assert_eq!(
            replay
                .iter()
                .filter(|i| i.0["type"] == "tool_search_output")
                .count(),
            1,
            "streamed={streamed}"
        );
        assert!(
            result
                .input
                .iter()
                .all(|i| !i.0.to_string().contains("_agents_nested_history_token"))
        );
        let state_json = result.to_state().to_json().unwrap();
        RunState::from_json(state_json).unwrap();
    }
}

// --------------------------------------------------------------------- after the run

/// Python: `test_to_input_list_during_active_stream_does_not_mutate_input`. The completed stream
/// keeps one replayed message and no internal markers in its input.
#[tokio::test]
async fn streamed_input_has_no_internal_markers_after_completion() {
    let chain = chain("handoff message", |h| h);
    let result = run_nested(&chain, true).await;
    assert!(
        result
            .input
            .iter()
            .all(|i| !i.0.to_string().contains("_agents_nested_history_token"))
    );
    assert_eq!(count(&result.to_input_list(), "handoff message"), 1);
}

/// Python: `test_nested_history_ownership_survives_result_new_items_copy`.
#[tokio::test]
async fn ownership_survives_a_copy_of_the_result_new_items() {
    let chain = chain("same", |h| h);
    let mut result = run_nested(&chain, false).await;
    result.new_items = result.new_items.clone();
    assert_eq!(count(&result.to_input_list(), "same"), 1);
}

/// Python: `test_result_input_mutation_does_not_change_state_snapshot_ownership`, the remove and
/// reorder mutations, direct and streamed. The snapshot taken before the mutation decides.
#[tokio::test]
async fn interrupted_state_ownership_follows_the_snapshot_not_the_result_view() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    for streamed in [false, true] {
        for mutation in ["remove", "reorder"] {
            let approval = FunctionTool::new(
                "approval_tool",
                "needs approval",
                |_ctx, _a: NoArgs| async { Ok("approved".to_string()) },
            )
            .with_needs_approval(true);
            let first_model = setup();
            let second_model = setup();
            let second = Arc::new(
                Agent::<()>::new("second")
                    .with_model_instance(second_model.clone())
                    .with_tool(approval),
            );
            second_model.add_turn(vec![function_call_with_id(
                "approval_tool",
                "{}",
                "approval",
            )]);
            second_model.add_turn(vec![text("done")]);
            let first = Arc::new(
                Agent::<()>::new("first")
                    .with_model_instance(first_model.clone())
                    .with_handoff(second),
            );
            first_model.add_turn(vec![text("owned once"), transfer("second")]);

            let mut interrupted = run_mode(
                &first,
                "start",
                RunOptions::new(()).run_config(nested()),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(interrupted.interruptions.len(), 1);
            let owned = interrupted
                .input
                .iter()
                .position(|i| item_text(i) == "owned once")
                .expect("forwarded message in input");
            if mutation == "remove" {
                interrupted.input.remove(owned);
            } else {
                let owned_item = interrupted.input.remove(owned);
                interrupted.input.insert(0, owned_item);
            }

            let mut state = interrupted.to_state();
            state.approve(&interrupted.interruptions[0], false);
            let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
            let resumed = run_mode(
                &first,
                restored,
                RunOptions::new(()).run_config(nested()),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
            assert_eq!(
                count(&resumed.to_input_list(), "owned once"),
                1,
                "streamed={streamed} mutation={mutation}"
            );
        }
    }
}

// ------------------------------------------------------------ copies that keep occurrences

/// Python: `test_plain_handoff_preserves_prior_nested_history_ownership`. A later plain
/// (non-nesting) handoff keeps the earlier nested summary and its owned message.
#[tokio::test]
async fn plain_handoff_preserves_prior_nested_history_ownership() {
    for streamed in [false, true] {
        let (chain, final_model) = three_agents("first message", "second message", |h| {
            h.with_nest_handoff_history(false)
        });
        let result = run_mode(
            &chain.first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "first message"),
            1,
            "streamed={streamed}"
        );
        drop(final_model);
    }
}

/// A three-agent chain `first -> second -> final`. The second handoff is configured by
/// `second_handoff`. Returns the chain (with `second` as `first`'s child) and final's model.
fn three_agents(
    first_message: &str,
    second_message: &str,
    second_handoff: impl FnOnce(Handoff<()>) -> Handoff<()>,
) -> (Chain, Arc<FakeModel>) {
    let first_model = setup();
    let second_model = setup();
    let final_model = setup();
    final_model.add_turn(vec![text("done")]);
    let final_agent = Arc::new(Agent::<()>::new("final").with_model_instance(final_model.clone()));
    second_model.add_turn(vec![text(second_message), transfer("final")]);
    let second = Arc::new(
        Agent::<()>::new("second")
            .with_model_instance(second_model.clone())
            .with_handoff(second_handoff(handoff(final_agent))),
    );
    first_model.add_turn(vec![text(first_message), transfer("second")]);
    let first = Arc::new(
        Agent::<()>::new("first")
            .with_model_instance(first_model.clone())
            .with_handoff(second),
    );
    (
        Chain {
            first,
            second_model,
        },
        final_model,
    )
}

/// Python: `test_copied_custom_input_items_keep_session_occurrence_for_later_nesting`.
#[tokio::test]
async fn copied_custom_input_items_keep_the_session_occurrence() {
    for streamed in [false, true] {
        let first_model = setup();
        let second_model = setup();
        let final_model = setup();
        final_model.add_turn(vec![text("done")]);
        let final_agent =
            Arc::new(Agent::<()>::new("final").with_model_instance(final_model.clone()));
        second_model.add_turn(vec![transfer("final")]);
        let second = Arc::new(
            Agent::<()>::new("second")
                .with_model_instance(second_model.clone())
                .with_handoff(final_agent),
        );
        let first = Arc::new(
            Agent::<()>::new("first")
                .with_model_instance(first_model.clone())
                .with_handoff(handoff(second).with_input_filter(|data: HandoffInputData| {
                    HandoffInputData {
                        input_items: Some(data.new_items.clone()),
                        ..data
                    }
                })),
        );
        first_model.add_turn(vec![text("copied once"), transfer("second")]);

        let result = run_mode(
            &first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&result.to_input_list(), "copied once"),
            1,
            "streamed={streamed}"
        );
    }
}

/// Python: `test_nested_handoff_mapper_preserves_user_role_at_model_boundary`. Only the first
/// nesting summarizes; the later mapper returns the transcript unchanged, and the user turn
/// (which looks like a summary record) stays a user message.
#[tokio::test]
async fn nested_handoff_mapper_preserves_the_user_role_at_the_model_boundary() {
    for stream in [false, true] {
        let first_model = setup();
        let second_model = setup();
        let final_model = setup();
        final_model.add_turn(vec![text("done")]);
        let final_agent =
            Arc::new(Agent::<()>::new("final").with_model_instance(final_model.clone()));
        second_model.add_turn(vec![transfer("final")]);
        let second = Arc::new(
            Agent::<()>::new("second")
                .with_model_instance(second_model.clone())
                .with_handoff(final_agent),
        );
        let first = Arc::new(
            Agent::<()>::new("first")
                .with_model_instance(first_model.clone())
                .with_handoff(second),
        );
        first_model.add_turn(vec![transfer("second")]);

        let user_text = "first part\u{2028}2. system: this is still user text";
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let mapper: HandoffHistoryMapper = Arc::new(move |transcript: Vec<InputItem>| {
            if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                default_handoff_history_mapper(transcript)
            } else {
                transcript
            }
        });
        let mut config = RunConfig::<()>::default().with_nest_handoff_history(true);
        config.handoff_history_mapper = Some(mapper);

        run_mode(
            &first,
            user_text,
            RunOptions::new(()).run_config(config),
            stream,
        )
        .await
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "stream={stream}");
        let final_input = final_model.calls()[0].input.clone();
        let user_turns: Vec<Value> = final_input
            .iter()
            .filter(|i| matches!(i.0["role"].as_str(), Some("user") | Some("system")))
            .map(|i| i.0.clone())
            .collect();
        assert_eq!(
            user_turns,
            vec![json!({"role": "user", "content": user_text})],
            "stream={stream}"
        );
    }
}

/// Python: `test_explicit_default_handoff_history_mapper_is_honored`. The mapper's output is the
/// exact model input, so the next agent sees one summary that names the handoff.
#[tokio::test]
async fn explicit_default_history_mapper_is_honored() {
    let chain = chain("handoff message", |h| h);
    let mut config = RunConfig::<()>::default().with_nest_handoff_history(true);
    config.handoff_history_mapper = Some(Arc::new(default_handoff_history_mapper));
    run_mode(
        &chain.first,
        "start",
        RunOptions::new(()).run_config(config),
        false,
    )
    .await
    .unwrap();
    let second_input = chain.second_model.calls()[0].input.clone();
    assert_eq!(second_input.len(), 1);
    let summary = item_text(&second_input[0]);
    assert!(summary.contains("handoff message"));
    assert!(summary.contains("transfer_to_second"));
}

/// Python: `test_pending_handoff_in_interrupted_turn_survives_run_state` (four cases).
#[tokio::test]
async fn pending_handoff_in_an_interrupted_turn_survives_run_state() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    for streamed in [false, true] {
        for nest_history in [false, true] {
            let approval = FunctionTool::new(
                "approval_tool",
                "needs approval",
                |_ctx, _a: NoArgs| async { Ok("approved".to_string()) },
            )
            .with_needs_approval(true);
            let first_model = setup();
            let second_model = setup();
            let final_model = setup();
            final_model.add_turn(vec![text("done")]);
            let final_agent =
                Arc::new(Agent::<()>::new("final").with_model_instance(final_model.clone()));
            let second = Arc::new(
                Agent::<()>::new("second")
                    .with_model_instance(second_model.clone())
                    .with_tool(approval)
                    .with_handoff(final_agent),
            );
            let first = Arc::new(
                Agent::<()>::new("first")
                    .with_model_instance(first_model.clone())
                    .with_handoff(second),
            );
            first_model.add_turn(vec![
                text("first once"),
                function_call_with_id("transfer_to_second", "{}", "first-handoff"),
            ]);
            second_model.add_turn(vec![
                text("second once"),
                function_call_with_id("approval_tool", "{}", "approval"),
                function_call_with_id("transfer_to_final", "{}", "final-handoff"),
            ]);

            let config = RunConfig::<()>::default().with_nest_handoff_history(nest_history);
            let interrupted = run_mode(
                &first,
                "start",
                RunOptions::new(()).run_config(config.clone()),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(interrupted.interruptions.len(), 1);
            let mut state = interrupted.to_state();
            state.approve(&interrupted.interruptions[0], false);
            let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
            assert_eq!(
                restored
                    .last_processed_response
                    .as_ref()
                    .map(|p| p.handoffs.len()),
                Some(1)
            );

            let resumed = run_mode(
                &first,
                restored,
                RunOptions::new(()).run_config(config),
                streamed,
            )
            .await
            .unwrap();
            assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
            assert_eq!(resumed.last_agent.name, "final");
            let handoff_outputs = resumed
                .new_items
                .iter()
                .filter(|i| match &i.kind {
                    RunItemKind::HandoffOutput { raw_item, .. } => {
                        raw_item.0["call_id"] == "final-handoff"
                    }
                    _ => false,
                })
                .count();
            assert_eq!(
                handoff_outputs, 1,
                "streamed={streamed} nest={nest_history}"
            );
            if nest_history {
                let summary = item_text(&final_model.calls().last().unwrap().input[0]);
                assert_eq!(summary.matches("first once").count(), 1);
                assert_eq!(summary.matches("second once").count(), 1);
            }
        }
    }
}

/// Python: `test_resumed_handoff_persists_all_staged_approval_outputs` (eight cases). Each
/// staged approval result reaches the session before the resumed handoff runs.
#[tokio::test]
async fn resumed_handoff_persists_all_staged_approval_outputs() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    #[derive(Default)]
    struct Hooks {
        handoffs: AtomicUsize,
    }
    #[async_trait::async_trait]
    impl RunHooks<()> for Hooks {
        async fn on_handoff(&self, _ctx: &RunContext<()>, _from: &Agent<()>, _to: &Agent<()>) {
            self.handoffs.fetch_add(1, Ordering::SeqCst);
        }
    }
    for streamed in [false, true] {
        for nest_history in [false, true] {
            for second_decision in ["approve", "reject"] {
                let calls = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                let make_tool = |name: &'static str| {
                    let calls = calls.clone();
                    FunctionTool::new(name, "needs approval", move |_ctx, _a: NoArgs| {
                        let calls = calls.clone();
                        async move {
                            calls
                                .lock()
                                .unwrap()
                                .push(name.trim_end_matches("_tool").to_string());
                            Ok(format!("{name} ok"))
                        }
                    })
                    .with_needs_approval(true)
                };
                let source_model = setup();
                let target_model = setup();
                target_model.add_turn(vec![text("done")]);
                let target =
                    Arc::new(Agent::<()>::new("target").with_model_instance(target_model.clone()));
                let source = Arc::new(
                    Agent::<()>::new("source")
                        .with_model_instance(source_model.clone())
                        .with_tool(make_tool("first_tool"))
                        .with_tool(make_tool("second_tool"))
                        .with_handoff(target.clone()),
                );
                source_model.add_turn(vec![
                    function_call_with_id("first_tool", "{}", "first"),
                    function_call_with_id("second_tool", "{}", "second"),
                    function_call_with_id("transfer_to_target", "{}", "handoff"),
                ]);

                let hooks = Arc::new(Hooks::default());
                let session = Arc::new(InMemorySession::new("staged"));
                let config = RunConfig::<()>::default().with_nest_handoff_history(nest_history);
                let options = || {
                    RunOptions::new(())
                        .run_config(config.clone())
                        .hooks(hooks.clone())
                        .session(session.clone())
                };

                let first = run_mode(&source, "start", options(), streamed)
                    .await
                    .unwrap();
                assert_eq!(first.interruptions.len(), 2);
                let mut state =
                    RunState::from_json_str(&first.to_state().to_json_string().unwrap()).unwrap();
                let first_approval = state
                    .get_interruptions()
                    .into_iter()
                    .find(|i| i.raw_item.call_id == "first")
                    .unwrap();
                state.approve(&first_approval, false);

                let staged = run_mode(&source, state, options(), streamed).await.unwrap();
                assert_eq!(staged.interruptions.len(), 1);
                assert_eq!(hooks.handoffs.load(Ordering::SeqCst), 0);

                let mut state =
                    RunState::from_json_str(&staged.to_state().to_json_string().unwrap()).unwrap();
                let second_approval = state.get_interruptions()[0].clone();
                assert_eq!(second_approval.raw_item.call_id, "second");
                if second_decision == "approve" {
                    state.approve(&second_approval, false);
                } else {
                    state.reject(&second_approval, false, None);
                }

                let finished = run_mode(&source, state, options(), streamed).await.unwrap();
                assert_eq!(finished.final_output_text().as_deref(), Some("done"));
                assert_eq!(finished.last_agent.name, "target");
                assert_eq!(hooks.handoffs.load(Ordering::SeqCst), 1);
                let ran = calls.lock().unwrap().clone();
                let expected: Vec<String> = if second_decision == "approve" {
                    vec!["first".into(), "second".into()]
                } else {
                    vec!["first".into()]
                };
                assert_eq!(ran, expected);

                let items = session.get_items(None).await.unwrap();
                for call_id in ["first", "second", "handoff"] {
                    let count_type = |ty: &str| {
                        items
                            .iter()
                            .filter(|i| i.0["type"] == ty && i.0["call_id"] == call_id)
                            .count()
                    };
                    assert_eq!(count_type("function_call"), 1, "{call_id}");
                    assert_eq!(count_type("function_call_output"), 1, "{call_id}");
                }
            }
        }
    }
}

/// Python: `test_nested_history_resume_to_final_preserves_status_less_ownership`. A message
/// without a `status` is still forwarded once and survives the approval resume.
#[tokio::test]
async fn status_less_forwarded_message_is_replayed_once_after_resume() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    for streamed in [false, true] {
        let approval = FunctionTool::new(
            "approval_tool",
            "needs approval",
            |_ctx, _a: NoArgs| async { Ok("approved".to_string()) },
        )
        .with_needs_approval(true);
        let first_model = setup();
        let second_model = setup();
        let second = Arc::new(
            Agent::<()>::new("second")
                .with_model_instance(second_model.clone())
                .with_tool(approval),
        );
        let first = Arc::new(
            Agent::<()>::new("first")
                .with_model_instance(first_model.clone())
                .with_handoff(second),
        );
        first_model.add_turn(vec![
            OutputItem::from_value(json!({
                "type": "message", "id": "msg_none_status", "role": "assistant",
                "content": [{"type": "output_text", "text": "once", "annotations": []}],
            })),
            transfer("second"),
        ]);
        second_model.add_turn(vec![function_call_with_id(
            "approval_tool",
            "{}",
            "approval",
        )]);
        second_model.add_turn(vec![text("done")]);

        let interrupted = run_mode(
            &first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(interrupted.interruptions.len(), 1);
        let mut state = interrupted.to_state();
        state.approve(&interrupted.interruptions[0], false);
        let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();

        let resumed = run_mode(
            &first,
            restored,
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
        let replay = resumed.to_input_list();
        assert_eq!(count(&replay, "once"), 1, "streamed={streamed}");
        assert!(
            second_model
                .calls()
                .last()
                .unwrap()
                .input
                .iter()
                .all(|i| !i.0.to_string().contains("_agents_nested_history_token"))
        );
    }
}

/// Python: `test_first_nested_handoff_after_restore_uses_explicit_occurrence_lineage`, the
/// current-snapshot case. A restored state replays the tool search items once.
#[tokio::test]
async fn first_nested_handoff_after_restore_replays_tool_search_once() {
    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    for streamed in [false, true] {
        let approval = FunctionTool::new(
            "approval_tool",
            "needs approval",
            |_ctx, _a: NoArgs| async { Ok("approved".to_string()) },
        )
        .with_needs_approval(true);
        let first_model = setup();
        let second_model = setup();
        second_model.add_turn(vec![text("done")]);
        let second = Arc::new(Agent::<()>::new("second").with_model_instance(second_model.clone()));
        let first = Arc::new(
            Agent::<()>::new("first")
                .with_model_instance(first_model.clone())
                .with_tool(approval)
                .with_handoff(second),
        );
        first_model.add_turn(vec![
            OutputItem::from_value(json!({
                "type": "tool_search_call", "id": "tool_search_call", "call_id": "search",
                "arguments": {"query": "profile"}, "execution": "server", "status": "completed",
            })),
            OutputItem::from_value(json!({
                "type": "tool_search_output", "id": "tool_search_output", "call_id": "search",
                "execution": "server", "status": "completed", "tools": [],
            })),
            function_call_with_id("approval_tool", "{}", "approval"),
        ]);
        first_model.add_turn(vec![transfer("second")]);

        let interrupted = run_mode(
            &first,
            "start",
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(interrupted.interruptions.len(), 1);
        let mut state = interrupted.to_state();
        state.approve(&interrupted.interruptions[0], false);
        let restored = RunState::from_json(state.to_json().unwrap()).unwrap();

        let resumed = run_mode(
            &first,
            restored,
            RunOptions::new(()).run_config(nested()),
            streamed,
        )
        .await
        .unwrap();
        let replay = resumed.to_input_list();
        assert_eq!(
            replay
                .iter()
                .filter(|i| i.0["type"] == "tool_search_call")
                .count(),
            1,
            "streamed={streamed}"
        );
        assert_eq!(
            replay
                .iter()
                .filter(|i| i.0["type"] == "tool_search_output")
                .count(),
            1,
            "streamed={streamed}"
        );
    }
}

/// Python: `test_nested_history_session_resolution_normalizes_forwarded_status_before_ownership`
/// (the observable part). A forwarded message whose status is `None` carries no `status` key.
#[test]
fn forwarded_message_status_is_normalized_away() {
    let raw: MessageOutput = serde_json::from_value(json!({
        "id": "msg_none_status",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "same", "annotations": []}],
    }))
    .unwrap();
    let data = HandoffInputData {
        input_history: vec![InputItem(json!({"role": "user", "content": "start"}))],
        pre_handoff_items: Vec::new(),
        new_items: vec![RunItem::new(
            "first",
            RunItemKind::MessageOutput { raw_item: raw },
        )],
        input_items: None,
    };
    let nested = openai_agents_sdk::handoffs::nest_handoff_history(data, None);
    let forwarded = nested
        .input_history
        .iter()
        .find(|i| i.0["role"] == "assistant" && i.0.get("type").is_some_and(|t| t == "message"))
        .expect("forwarded message");
    assert!(forwarded.0.get("status").is_none(), "{forwarded:?}");
}
