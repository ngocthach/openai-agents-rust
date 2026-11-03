//! Port of `tests/test_server_conversation_tracker.py`: what a server-managed conversation
//! already holds, and which items a model call still has to carry.
//!
//! The Rust tracker has no `prepare_input`: the runner takes the initial input from
//! `take_initial_input` and keeps the generated items `needs_send` accepts, so `prepare` below
//! performs the same two steps. Python-only: object-identity bookkeeping (`sent_items`,
//! `prepared_item_sources`, monkeypatched `id()`, address reuse) and the reasoning id policy
//! parameter, which the Rust runner applies after the tracker has chosen the items.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::{
    InputItem, ModelResponse, OutputItem, RunInput, RunItem, RunItemKind,
};
use openai_agents_sdk::run_config::ModelInputData;
use openai_agents_sdk::run_state::ServerConversationState;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool};
use openai_agents_sdk::{Agent, RunConfig, RunOptions, Runner, ServerConversationTracker};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const FAKE_RESPONSES_ID: &str = "__fake_id__";

fn item(value: Value) -> InputItem {
    InputItem(value)
}

/// A run item that carries `raw` as its input form, whatever the kind (Python's `DummyRunItem`).
fn dummy(raw: Value) -> RunItem {
    RunItem::new("test", RunItemKind::Compaction { raw_item: raw })
}

fn output_item(call_id: &str, output: &str) -> RunItem {
    RunItem::new(
        "test",
        RunItemKind::ToolCallOutput {
            custom_data: None,
            raw_item: json!({"type": "function_call_output", "call_id": call_id, "output": output}),
            output: json!(output),
        },
    )
}

fn tool_call(call_id: &str, id: &str, name: &str, arguments: &str) -> Value {
    json!({"type": "function_call", "id": id, "call_id": call_id, "name": name,
           "arguments": arguments, "status": "completed"})
}

fn response(output: Vec<Value>, id: &str) -> ModelResponse {
    ModelResponse {
        output: output.into_iter().map(OutputItem::from_value).collect(),
        usage: Default::default(),
        response_id: Some(id.to_string()),
        request_id: None,
        raw_usage: None,
    }
}

/// The two steps the runner takes to assemble a server-managed call's input.
fn prepare(
    tracker: &mut ServerConversationTracker,
    original: &[InputItem],
    generated: &[RunItem],
) -> Vec<InputItem> {
    let mut input = tracker.take_initial_input(original);
    input.extend(
        generated
            .iter()
            .map(RunItem::to_input_item)
            .filter(|converted| tracker.needs_send(converted)),
    );
    input
}

fn hydrate(
    tracker: &mut ServerConversationTracker,
    original: &[InputItem],
    generated: &[RunItem],
    responses: &[ModelResponse],
) {
    tracker.hydrate_from_state(
        &ServerConversationState::default(),
        original,
        generated,
        responses,
        true,
    );
}

fn values(items: &[InputItem]) -> Vec<Value> {
    items.iter().map(|i| i.0.clone()).collect()
}

fn output_call_ids(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.0["type"] == "function_call_output")
        .map(|i| i.0["call_id"].as_str().unwrap().to_string())
        .collect()
}

/// Python: `test_prepare_input_filters_items_seen_by_server_and_tool_calls`.
#[test]
fn prepare_input_filters_items_seen_by_server_and_tool_calls() {
    let mut tracker = ServerConversationTracker::default();
    let original = vec![
        item(json!({"id": "input-1", "type": "message"})),
        item(json!({"id": "input-2", "type": "message"})),
    ];
    let new_raw_item = json!({"type": "message", "content": "hello"});
    let generated = vec![
        dummy(json!({"id": "server-echo", "type": "message"})),
        dummy(new_raw_item.clone()),
        output_item("call-1", "done"),
    ];
    let prior = response(
        vec![json!({"call_id": "call-1", "output": "prior", "type": "function_call_output"})],
        "resp-1",
    );

    hydrate(&mut tracker, &original, &generated, &[prior]);
    let prepared = prepare(&mut tracker, &original, &generated);

    assert_eq!(values(&prepared), vec![new_raw_item]);
    assert!(tracker.sent_initial_input());
    assert!(tracker.remaining_initial_input().is_none());
}

fn interrupted_cleanup_state() -> (Vec<RunItem>, ModelResponse) {
    let cleanup1 = tool_call(
        "call_CLEANUP1",
        "fc_001",
        "run_cleanup",
        r#"{"target": "temp_files"}"#,
    );
    let diagnostic = tool_call(
        "call_DIAG",
        "fc_002",
        "run_diagnostic",
        r#"{"check_name": "thermal"}"#,
    );
    let cleanup2 = tool_call(
        "call_CLEANUP2",
        "fc_003",
        "run_cleanup",
        r#"{"target": "winsxs_cache"}"#,
    );
    let model_response = response(
        vec![cleanup1.clone(), diagnostic.clone(), cleanup2.clone()],
        "resp_002",
    );
    let approval = |raw: &Value| {
        RunItem::new(
            "test",
            RunItemKind::ToolApproval {
                raw_item: serde_json::from_value(raw.clone()).unwrap(),
                tool_name: "run_cleanup".into(),
            },
        )
    };
    let call_item = |raw: &Value| {
        RunItem::new(
            "test",
            RunItemKind::ToolCall {
                raw_item: raw.clone(),
            },
        )
    };
    let generated = vec![
        call_item(&cleanup1),
        call_item(&diagnostic),
        call_item(&cleanup2),
        output_item("call_DIAG", "Diagnostic completed."),
        approval(&cleanup1),
        approval(&cleanup2),
    ];
    (generated, model_response)
}

/// Python: `test_hydrate_from_state_preserves_unsent_outputs_from_interrupted_turn`.
#[test]
fn hydrate_from_state_preserves_unsent_outputs_from_interrupted_turn() {
    let (generated, model_response) = interrupted_cleanup_state();
    let input = vec![item(
        json!({"role": "user", "content": "Run cleanup, diagnostics, and cleanup."}),
    )];
    let mut tracker = ServerConversationTracker::default();
    hydrate(
        &mut tracker,
        &input,
        &generated,
        std::slice::from_ref(&model_response),
    );

    assert!(!tracker.state().server_tool_call_ids.contains("call_DIAG"));

    let mut after_resume: Vec<RunItem> = generated[..4].to_vec();
    after_resume.push(output_item("call_CLEANUP1", "Tool call not approved."));
    after_resume.push(output_item("call_CLEANUP2", "Tool call not approved."));
    let prepared = prepare(&mut tracker, &input, &after_resume);

    assert_eq!(
        output_call_ids(&prepared),
        vec!["call_DIAG", "call_CLEANUP1", "call_CLEANUP2"]
    );
}

/// Python: `test_hydrate_from_state_preserves_unsent_output_without_a_pending_tool_run`.
#[test]
fn hydrate_from_state_preserves_unsent_output_without_a_pending_tool_run() {
    let approval_call = tool_call(
        "call_APPROVAL",
        "fc_101",
        "run_cleanup",
        r#"{"target": "temp_files"}"#,
    );
    let missing_call = tool_call("call_MISSING", "fc_102", "not_a_registered_tool", "{}");
    let model_response = response(
        vec![approval_call.clone(), missing_call.clone()],
        "resp_101",
    );
    let missing_output = output_item("call_MISSING", "Tool not_a_registered_tool not found.");
    let call_item = |raw: &Value| {
        RunItem::new(
            "test",
            RunItemKind::ToolCall {
                raw_item: raw.clone(),
            },
        )
    };
    let generated = vec![
        call_item(&approval_call),
        call_item(&missing_call),
        RunItem::new(
            "test",
            RunItemKind::ToolApproval {
                raw_item: serde_json::from_value(approval_call.clone()).unwrap(),
                tool_name: "run_cleanup".into(),
            },
        ),
        missing_output.clone(),
    ];
    let input = vec![item(
        json!({"role": "user", "content": "Clean up and call the missing tool."}),
    )];
    let mut tracker = ServerConversationTracker::default();
    hydrate(
        &mut tracker,
        &input,
        &generated,
        std::slice::from_ref(&model_response),
    );

    assert!(
        !tracker
            .state()
            .server_tool_call_ids
            .contains("call_MISSING")
    );

    let after_resume = vec![
        call_item(&approval_call),
        call_item(&missing_call),
        missing_output,
        output_item("call_APPROVAL", "Cleanup finished."),
    ];
    let prepared = prepare(&mut tracker, &input, &after_resume);

    assert_eq!(
        output_call_ids(&prepared),
        vec!["call_MISSING", "call_APPROVAL"]
    );
}

/// Python: `test_hydrate_from_state_does_not_track_{string,list}_initial_input_by_object_identity`.
#[test]
fn hydrate_from_state_tracks_initial_input_by_content() {
    for original in [
        vec![InputItem::user("hello")],
        vec![item(json!({"role": "user", "content": "hello"}))],
    ] {
        let mut tracker = ServerConversationTracker::default();
        hydrate(&mut tracker, &original, &[], &[]);

        assert!(tracker.sent_initial_input());
        assert!(tracker.remaining_initial_input().is_none());
        assert_eq!(tracker.state().sent_item_fingerprints.len(), 1);
    }
}

/// Python: `test_mark_input_as_sent_and_rewind_input_respects_remaining_initial_input`.
#[test]
fn mark_input_as_sent_and_rewind_input_respect_remaining_initial_input() {
    let mut tracker = ServerConversationTracker::default();
    let pending_1 = item(json!({"id": "p-1", "type": "message"}));
    let pending_2 = item(json!({"id": "p-2", "type": "message"}));
    tracker.set_remaining_initial_input(Some(vec![pending_1.clone(), pending_2]));

    tracker.mark_input_as_sent(&[
        pending_1.clone(),
        item(json!({"id": "p-2", "type": "message"})),
    ]);
    assert!(tracker.remaining_initial_input().is_none());

    tracker.rewind_input(std::slice::from_ref(&pending_1));
    assert_eq!(
        tracker.remaining_initial_input(),
        Some([pending_1].as_slice())
    );
}

fn tool_search_pair() -> (Value, Value) {
    (
        json!({"type": "tool_search_call", "queries": [{"search_term": "account balance"}]}),
        json!({"type": "tool_search_output", "results": [{"text": "Balance lookup docs"}]}),
    )
}

/// Python: `test_hydrate_from_state_skips_restored_tool_search_items_by_object_identity` and
/// `..._by_fingerprint` (a rebuilt copy has the same content).
#[test]
fn hydrate_from_state_skips_restored_tool_search_items() {
    let (call, result) = tool_search_pair();
    let hydrated = vec![dummy(call.clone()), dummy(result.clone())];
    let rebuilt = vec![dummy(call), dummy(result)];

    for generated in [&hydrated, &rebuilt] {
        let mut tracker = ServerConversationTracker::default();
        hydrate(&mut tracker, &[], &hydrated, &[]);
        assert!(prepare(&mut tracker, &[], generated).is_empty());
    }
}

fn tool_search_items(extra: Value) -> (Value, Value) {
    let mut call = json!({
        "type": "tool_search_call", "arguments": {"query": "account balance"},
        "execution": "server", "status": "completed",
    });
    let mut output = json!({
        "type": "tool_search_output", "execution": "server", "status": "completed", "tools": [],
    });
    for target in [&mut call, &mut output] {
        for (key, value) in extra.as_object().unwrap() {
            target[key] = value.clone();
        }
    }
    (call, output)
}

/// Python: `test_hydrate_from_state_skips_restored_tool_search_items_when_created_by_is_stripped`.
/// Server-managed Rust runs keep no session copy, so the restored history arrives as the original
/// input the state says was already sent.
#[test]
fn hydrate_from_state_skips_restored_tool_search_items_when_created_by_is_stripped() {
    let ids = json!({"call_id": "tool_search_call_1"});
    let (saved_call, saved_output) =
        tool_search_items(json!({"call_id": "tool_search_call_1", "created_by": "server"}));
    let (live_call, live_output) = tool_search_items(ids);
    let mut tracker = ServerConversationTracker::default();
    hydrate(
        &mut tracker,
        &[item(saved_call), item(saved_output)],
        &[],
        &[],
    );

    let prepared = prepare(&mut tracker, &[], &[dummy(live_call), dummy(live_output)]);

    assert!(prepared.is_empty(), "{prepared:?}");
}

/// Python: `test_hydrate_from_state_skips_restored_tool_search_items_when_only_ids_differ`.
#[test]
fn hydrate_from_state_skips_restored_tool_search_items_when_only_ids_differ() {
    let (saved_call, mut saved_output) = tool_search_items(json!({}));
    let mut saved_call = saved_call;
    saved_call["id"] = json!("tool_search_call_saved");
    saved_output["id"] = json!("tool_search_output_saved");
    let (live_call, live_output) = tool_search_items(json!({}));
    let mut tracker = ServerConversationTracker::default();
    hydrate(
        &mut tracker,
        &[item(saved_call), item(saved_output)],
        &[],
        &[],
    );

    let prepared = prepare(&mut tracker, &[], &[dummy(live_call), dummy(live_output)]);

    assert!(prepared.is_empty(), "{prepared:?}");
}

/// Python: `test_prepare_input_keeps_repeated_tool_search_items_with_new_ids`.
#[test]
fn prepare_input_keeps_repeated_tool_search_items_with_new_ids() {
    let mut tracker = ServerConversationTracker::default();
    let (mut prior_call, mut prior_output) = tool_search_items(json!({"created_by": "server"}));
    prior_call["id"] = json!("tool_search_call_saved");
    prior_output["id"] = json!("tool_search_output_saved");
    tracker.track_server_items(&response(
        vec![prior_call, prior_output],
        "resp-tool-search-repeat-1",
    ));

    let (mut repeat_call, mut repeat_output) = tool_search_items(json!({}));
    repeat_call["id"] = json!("tool_search_call_repeat");
    repeat_output["id"] = json!("tool_search_output_repeat");
    let generated = vec![dummy(repeat_call.clone()), dummy(repeat_output.clone())];

    let prepared = prepare(&mut tracker, &[], &generated);

    assert_eq!(values(&prepared), vec![repeat_call, repeat_output]);
}

/// Python: `test_track_server_items_skips_live_tool_search_items_on_next_prepare`.
#[test]
fn track_server_items_skips_live_tool_search_items_on_next_prepare() {
    let mut tracker = ServerConversationTracker::default();
    let live = json!({"call_id": "tool_search_call_live"});
    let (call, output) =
        tool_search_items(json!({"call_id": "tool_search_call_live", "created_by": "server"}));
    tracker.track_server_items(&response(vec![call, output], "resp-tool-search"));

    let (call, output) = tool_search_items(live);
    let prepared = prepare(&mut tracker, &[], &[dummy(call), dummy(output)]);

    assert!(prepared.is_empty(), "{prepared:?}");
}

/// Python: `test_track_server_items_filters_pending_tool_search_by_sanitized_fingerprint`.
#[test]
fn track_server_items_filters_pending_tool_search_by_sanitized_fingerprint() {
    let mut tracker = ServerConversationTracker::default();
    let (pending, _) = tool_search_items(json!({"call_id": "tool_search_pending"}));
    let keep = item(json!({"id": "keep-me", "type": "message"}));
    tracker.set_remaining_initial_input(Some(vec![item(pending), keep.clone()]));

    let (produced, _) =
        tool_search_items(json!({"call_id": "tool_search_pending", "created_by": "server"}));
    tracker.track_server_items(&response(vec![produced], "resp-tool-search-pending"));

    assert_eq!(tracker.remaining_initial_input(), Some([keep].as_slice()));
}

/// Python: `test_track_server_items_filters_remaining_initial_input_by_fingerprint`.
#[test]
fn track_server_items_filters_remaining_initial_input_by_fingerprint() {
    let mut tracker = ServerConversationTracker::default();
    let kept = item(json!({"id": "keep-me", "type": "message"}));
    let filtered =
        item(json!({"type": "function_call_output", "call_id": "call-2", "output": "x"}));
    tracker.set_remaining_initial_input(Some(vec![kept.clone(), filtered]));

    tracker.track_server_items(&response(
        vec![json!({"type": "function_call_output", "call_id": "call-2", "output": "x"})],
        "resp-2",
    ));

    assert_eq!(tracker.remaining_initial_input(), Some([kept].as_slice()));
}

/// Python: `test_prepare_input_does_not_skip_fake_response_ids`.
#[test]
fn prepare_input_does_not_skip_fake_response_ids() {
    let mut tracker = ServerConversationTracker::default();
    tracker.track_server_items(&response(
        vec![json!({"id": FAKE_RESPONSES_ID, "type": "message"})],
        "resp-3",
    ));

    let raw_item = json!({"id": FAKE_RESPONSES_ID, "type": "message", "content": "hello"});
    let prepared = prepare(&mut tracker, &[], &[dummy(raw_item.clone())]);

    assert_eq!(values(&prepared), vec![raw_item]);
}

/// Python: `test_prepare_input_dedupes_same_delivered_tool_output_object`.
#[test]
fn prepare_input_does_not_resend_a_delivered_tool_output() {
    let mut tracker = ServerConversationTracker::default();
    let generated = vec![output_item("call_X", "42")];

    let first = prepare(&mut tracker, &[], &generated);
    assert!(first.iter().any(|i| i.0["call_id"] == "call_X"));

    tracker.mark_input_as_sent(&first);
    let second = prepare(&mut tracker, &[], &generated);

    assert!(
        second.iter().all(|i| i.0["call_id"] != "call_X"),
        "{second:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Through the runner
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema, Default)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

fn lookup_tool() -> FunctionTool<()> {
    FunctionTool::new("lookup", "a tool", |_ctx, _a: AnyArgs| async move {
        Ok("tool result".to_string())
    })
}

fn first_item_only_on_the_first_call() -> RunConfig<()> {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    RunConfig::default().with_call_model_input_filter(move |data| {
        let first = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
        async move {
            let mut input = data.model_data.input;
            if first {
                input.truncate(1);
            }
            Ok(ModelInputData {
                input,
                instructions: data.model_data.instructions,
            })
        }
    })
}

/// Python: `test_get_new_response_marks_filtered_input_as_sent` and
/// `test_run_single_turn_streamed_marks_filtered_input_as_sent`: an initial item the input filter
/// held back was never delivered, so the next call carries it.
#[tokio::test]
async fn a_filtered_out_initial_item_is_offered_again() {
    openai_agents_sdk::set_tracing_disabled(true);
    for streamed in [false, true] {
        let model = Arc::new(
            ScriptedModel::with_steps(vec![
                ScriptedStep::output(vec![function_call_with_id("lookup", "{}", "call_1")]),
                ScriptedStep::output(vec![assistant_message("ok")]),
            ])
            .unwrap(),
        );
        let agent = Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(lookup_tool());
        let options = RunOptions::new(())
            .conversation_id("conv_filtered")
            .run_config(first_item_only_on_the_first_call());
        let input = RunInput::from(vec![
            item(json!({"role": "user", "content": "first"})),
            item(json!({"role": "user", "content": "second"})),
        ]);

        if streamed {
            let mut result = Runner::run_streamed_with_options(agent, input, options);
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        } else {
            Runner::run_with_options(agent, input, options)
                .await
                .unwrap();
        }

        let calls = model.calls();
        assert_eq!(calls.len(), 2, "streamed={streamed}");
        assert_eq!(
            values(&calls[0].request.input),
            vec![json!({"role": "user", "content": "first"})],
            "streamed={streamed}"
        );
        let second_call = values(&calls[1].request.input);
        assert_eq!(
            second_call[0],
            json!({"role": "user", "content": "second"}),
            "streamed={streamed}"
        );
        assert!(
            !second_call.contains(&json!({"role": "user", "content": "first"})),
            "the delivered item is not sent twice: {second_call:?}"
        );
    }
}

/// Python: `test_run_single_turn_streamed_seeds_hosted_mcp_metadata_from_pre_step_items`: the
/// tool listing of an earlier step supplies the description and title of a later hosted call,
/// even when the input filter keeps the listing out of the model input.
#[tokio::test]
async fn streamed_hosted_mcp_call_takes_metadata_from_a_pre_step_listing() {
    openai_agents_sdk::set_tracing_disabled(true);
    let listing = OutputItem::Other(json!({
        "type": "mcp_list_tools",
        "id": "list_docs_server",
        "server_label": "docs_server",
        "tools": [{
            "name": "search_docs",
            "input_schema": {},
            "description": "Search the docs.",
            "annotations": {"title": "Search Docs"},
        }],
    }));
    let mcp_call = OutputItem::Other(json!({
        "type": "mcp_call", "id": "mcp_call_1", "arguments": "{}", "name": "search_docs",
        "server_label": "docs_server", "status": "completed",
    }));
    // Automatic streaming cannot express hosted MCP items, so the events are scripted exactly.
    let exact = |output: Vec<OutputItem>, id: &str| {
        ScriptedStep::stream(openai_agents_sdk::testing::scripted_stream_events(
            &ScriptedStep::output(output.clone()),
            ModelResponse {
                output,
                usage: Default::default(),
                response_id: Some(id.into()),
                request_id: None,
                raw_usage: None,
            },
        ))
    };
    let model = Arc::new(
        ScriptedModel::with_steps(vec![
            exact(
                vec![listing, function_call_with_id("lookup", "{}", "call_1")],
                "resp-listing",
            ),
            exact(vec![mcp_call], "resp-mcp"),
            exact(vec![assistant_message("done")], "resp-done"),
        ])
        .unwrap(),
    );
    let agent = Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_tool(lookup_tool())
        .with_tool(HostedMCPTool::new(json!({
            "type": "mcp",
            "server_label": "docs_server",
            "server_url": "https://example.com/mcp",
        })));
    let run_config = RunConfig::default().with_call_model_input_filter(|data| async move {
        let mut input = data.model_data.input;
        input.truncate(1);
        Ok(ModelInputData {
            input,
            instructions: data.model_data.instructions,
        })
    });

    let mut result = Runner::run_streamed_with_options(
        agent,
        "first",
        RunOptions::new(()).run_config(run_config),
    );
    let mut called: Vec<RunItem> = Vec::new();
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem { item, .. } = event.unwrap()
                && let RunItemKind::ToolCall { raw_item } = &item.kind
                && raw_item["type"] == "mcp_call"
            {
                called.push(item.clone());
            }
        }
    }

    assert_eq!(called.len(), 1);
    assert_eq!(called[0].description.as_deref(), Some("Search the docs."));
    assert_eq!(called[0].title.as_deref(), Some("Search Docs"));
}
