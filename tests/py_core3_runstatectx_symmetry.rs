//! Port of `tests/test_runtime_symmetry_contract.py`: streamed and non-streamed runs, and runs
//! resumed across the two modes, produce the same result, session writes, usage and traces.
//!
//! Python-only: `test_scripted_model_records_every_model_visible_request_field` inspects the
//! dataclass field names of the recorded model call. Rust's `ModelRequest` is a struct whose
//! fields are checked at compile time, so there is no runtime field list to compare.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::items::{RunInput, RunItem};
use openai_agents_sdk::memory::InMemorySession;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tracing::{InMemoryTraceProcessor, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{
    Agent, AgentsError, RunResult, RunResultStreaming, Runner, Session, Usage,
};
use serde_json::{Value, json};

/// Trace capture is process-global, so the tests that read it run one at a time.
static TRACE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// What a finished run leaves behind, for the streamed and the plain run alike.
#[derive(Debug)]
struct Finished {
    final_output: Option<String>,
    last_agent: String,
    new_items: Vec<Value>,
    interruptions: Vec<Value>,
    usage: Usage,
}

fn project_items(items: &[RunItem]) -> Vec<Value> {
    items
        .iter()
        .map(|item| {
            let v = serde_json::to_value(item).unwrap();
            let raw = v.get("raw_item").cloned().unwrap_or(Value::Null);
            json!({
                "type": v.get("type").cloned().unwrap_or(Value::Null),
                "name": raw.get("name").cloned().unwrap_or(Value::Null),
                "call_id": raw.get("call_id").cloned().unwrap_or(Value::Null),
                "output": raw.get("output").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn finished_from(result: &RunResult<()>) -> Finished {
    Finished {
        final_output: result.final_output_text(),
        last_agent: result.last_agent.name.clone(),
        new_items: project_items(&result.new_items),
        interruptions: result
            .interruptions
            .iter()
            .map(|item| json!({"name": item.name(), "call_id": item.call_id()}))
            .collect(),
        usage: result.context_wrapper.usage(),
    }
}

/// Run to completion, streamed or not, and return the finished state.
async fn run_mode(
    agent: &Arc<Agent<()>>,
    input: impl Into<RunInput>,
    streamed: bool,
    session: Option<Arc<dyn Session>>,
) -> Result<Finished, AgentsError> {
    let mut options = RunOptions::new(()).run_config(RunConfig::new());
    options.session = session;
    if streamed {
        let mut result: RunResultStreaming<()> =
            Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event?;
            }
        }
        return Ok(finished_from(
            result.result().expect("completed stream holds its result"),
        ));
    }
    let result = Runner::run_with_options(agent, input, options).await?;
    Ok(finished_from(&result))
}

/// Trace projection of the run's own trace: its spans reduced to kind, name and error state,
/// sorted so exporter ordering does not matter.
async fn trace_projection(processor: &InMemoryTraceProcessor) -> Vec<Value> {
    // Exporting is asynchronous; wait for the run's spans to arrive.
    for _ in 0..200 {
        openai_agents_sdk::tracing::flush_traces();
        if !processor.items().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let items = processor.items();
    let Some(trace_id) = items
        .iter()
        .rev()
        .find_map(|item| item["trace_id"].as_str().map(str::to_string))
    else {
        return Vec::new();
    };
    let mut projected: Vec<Value> = items
        .into_iter()
        .filter(|item| item["trace_id"].as_str() == Some(trace_id.as_str()) || item["id"].as_str() == Some(trace_id.as_str()))
        .map(|item| {
            json!({
                "object": item.get("object").cloned().unwrap_or(Value::Null),
                "span_type": item.get("span_data").and_then(|d| d.get("type")).cloned().unwrap_or(Value::Null),
                "name": item.get("span_data").and_then(|d| d.get("name")).cloned().unwrap_or(Value::Null),
                "has_error": item.get("error").is_some_and(|e| !e.is_null()),
            })
        })
        .collect();
    projected.sort_by_key(|v| v.to_string());
    projected
}

#[allow(clippy::field_reassign_with_default)]
fn detailed_usage_model() -> Arc<FakeModel> {
    let model = Arc::new(FakeModel::new());
    let mut usage = Usage::default();
    usage.requests = 1;
    usage.input_tokens = 11;
    usage.output_tokens = 7;
    usage.total_tokens = 18;
    usage.input_tokens_details.cached_tokens = 3;
    usage.input_tokens_details.cache_write_tokens = 2;
    usage.output_tokens_details.reasoning_tokens = 4;
    model.set_usage(usage);
    model
}

fn assert_detailed_usage(usage: &Usage) {
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
    assert!(usage.total_tokens > 0);
    assert!(usage.input_tokens_details.cached_tokens > 0);
    assert!(usage.input_tokens_details.cache_write_tokens > 0);
    assert!(usage.output_tokens_details.reasoning_tokens > 0);
    assert!(!usage.request_usage_entries.is_empty());
}

async fn streamed_and_nonstreamed_projection(scenario: &str, streamed: bool) -> Value {
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    processor.clear();

    let model = detailed_usage_model();
    let mut agent = Agent::<()>::new("symmetry-agent").with_model_instance(model.clone());
    if scenario == "function-tool" {
        model.add_turns(vec![
            vec![function_call_with_id("release_check", "{}", "call-release")],
            vec![assistant_message("READY")],
        ]);
        let tool = FunctionTool::<()>::new_no_args("release_check", "checked", |_c| async {
            Ok("checked".to_string())
        });
        agent = agent.with_tool(tool);
    } else {
        model.add_turn(vec![assistant_message("READY")]);
    }
    let agent = Arc::new(agent);
    let session = Arc::new(InMemorySession::new(format!("{scenario}-{streamed}")));
    let result = run_mode(&agent, "run the contract", streamed, Some(session.clone()))
        .await
        .unwrap();
    assert_detailed_usage(&result.usage);
    let session_items = session.get_items(None).await.unwrap();
    assert!(!session_items.is_empty());
    let traces = trace_projection(&processor).await;
    assert!(!traces.is_empty());
    set_trace_processors(Vec::new());
    json!({
        "final_output": result.final_output,
        "last_agent": result.last_agent,
        "new_items": result.new_items,
        "interruptions": result.interruptions,
        "session_items": session_items.len(),
        "traces": traces,
    })
}

#[tokio::test]
async fn streamed_and_nonstreamed_runs_match_basic() {
    let _guard = TRACE_LOCK.lock().await;
    let plain = streamed_and_nonstreamed_projection("basic", false).await;
    let streamed = streamed_and_nonstreamed_projection("basic", true).await;
    assert_eq!(plain, streamed);
}

#[tokio::test]
async fn streamed_and_nonstreamed_runs_match_function_tool() {
    let _guard = TRACE_LOCK.lock().await;
    let plain = streamed_and_nonstreamed_projection("function-tool", false).await;
    let streamed = streamed_and_nonstreamed_projection("function-tool", true).await;
    assert_eq!(plain, streamed);
}

#[tokio::test]
async fn streamed_and_nonstreamed_runs_raise_the_same_error_kind() {
    // Tracing is process-global; this test toggles it, so it runs under the same lock.
    let _guard = TRACE_LOCK.lock().await;
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let mut messages = Vec::new();
    for streamed in [false, true] {
        let model = Arc::new(FakeModel::new());
        model.add_error("release contract failure");
        let agent = Arc::new(Agent::<()>::new("symmetry-agent").with_model_instance(model));
        let err = run_mode(&agent, "run the contract", streamed, None)
            .await
            .expect_err("the model error must surface");
        messages.push(err.to_string());
    }
    assert_eq!(messages[0], messages[1]);
    assert!(
        messages[0].contains("release contract failure"),
        "{}",
        messages[0]
    );
}

async fn approval_resume_projection(start_streamed: bool, resume_streamed: bool) -> Value {
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    processor.clear();

    let model = detailed_usage_model();
    model.add_turns(vec![
        vec![function_call_with_id("release_check", "{}", "call-release")],
        vec![assistant_message("READY")],
    ]);
    let tool = FunctionTool::<()>::new_no_args("release_check", "checked", |_c| async {
        Ok("checked".to_string())
    })
    .with_needs_approval(true);
    let agent = Arc::new(
        Agent::<()>::new("symmetry-agent")
            .with_model_instance(model.clone())
            .with_tool(tool),
    );
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new(format!(
        "approval-{start_streamed}-{resume_streamed}"
    )));

    let (first, mut state) = run_mode_with_state(
        &agent,
        "run the contract".into(),
        start_streamed,
        session.clone(),
    )
    .await;
    assert_eq!(first.interruptions.len(), 1);
    state.approve(&first_interruption(&state), false);

    let (resumed, _) =
        run_mode_with_state(&agent, state.into(), resume_streamed, session.clone()).await;
    let session_items = session.get_items(None).await.unwrap();
    assert!(!session_items.is_empty());
    let traces = trace_projection(&processor).await;
    assert!(!traces.is_empty());
    set_trace_processors(Vec::new());
    json!({
        "first": {
            "final_output": first.final_output,
            "last_agent": first.last_agent,
            "new_items": first.new_items,
            "interruptions": first.interruptions,
            "usage_requests": first.usage.requests,
        },
        "resumed": {
            "final_output": resumed.final_output,
            "last_agent": resumed.last_agent,
            "new_items": resumed.new_items,
            "interruptions": resumed.interruptions,
            "usage_requests": resumed.usage.requests,
        },
        "session_items": session_items.len(),
        "traces": traces,
    })
}

fn first_interruption(state: &RunState) -> openai_agents_sdk::items::ToolApprovalItem {
    state.get_interruptions()[0].clone()
}

/// Like [`run_mode`], but also returns the state the run leaves for a later resume.
async fn run_mode_with_state(
    agent: &Arc<Agent<()>>,
    input: RunInput,
    streamed: bool,
    session: Arc<dyn Session>,
) -> (Finished, RunState) {
    let mut options = RunOptions::new(()).run_config(RunConfig::new());
    options.session = Some(session);
    if streamed {
        let mut result: RunResultStreaming<()> =
            Runner::run_streamed_with_options(agent, input, options);
        {
            let mut events = std::pin::pin!(result.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        let run = result.result().expect("completed stream holds its result");
        return (finished_from(run), run.to_state());
    }
    let run = Runner::run_with_options(agent, input, options)
        .await
        .unwrap();
    (finished_from(&run), run.to_state())
}

#[tokio::test]
async fn approval_resume_cross_modes_match() {
    let _guard = TRACE_LOCK.lock().await;
    let a = approval_resume_projection(true, false).await;
    let b = approval_resume_projection(false, true).await;
    assert_eq!(a, b);
}
