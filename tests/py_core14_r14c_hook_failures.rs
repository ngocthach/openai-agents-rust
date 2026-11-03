//! Ports of the hook-failure cases of `tests/test_run_state_pending_input.py`. A lifecycle hook
//! fails a run through its fallible `try_` twin; the run then leaves a checkpoint the caller
//! resumes from. Python reads `state._current_step`; Rust reads the checkpoint the failed run
//! stored in a [`RunCheckpoint`].

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::{InputItem, ModelResponse, RunInput};
use openai_agents_sdk::lifecycle::{AgentHooks, RunHooks};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::run_state::RunState;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, Tool};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, Result, RunItemStreamEventName, Runner, StreamEvent};
use serde_json::{Value, json};

type Calls = Arc<Mutex<Vec<String>>>;

fn options(auto: bool) -> RunOptions<()> {
    let mut options = RunOptions::new(()).run_config(RunConfig::new());
    options.auto_previous_response_id = auto;
    options
}

fn message_text(item: &InputItem) -> Option<String> {
    let obj = item.0.as_object()?;
    if obj.get("role")?.as_str()? != "user" {
        return None;
    }
    obj.get("content")?.as_str().map(str::to_string)
}

fn count_text(items: &[InputItem], text: &str) -> usize {
    items
        .iter()
        .filter(|i| message_text(i).as_deref() == Some(text))
        .count()
}

fn destination_tool(calls: &Calls) -> FunctionTool<()> {
    let calls = calls.clone();
    FunctionTool::<()>::new(
        "record_destination",
        "Records a destination",
        move |_c, a: Value| {
            let calls = calls.clone();
            async move {
                let destination = a["destination"].as_str().unwrap_or_default().to_string();
                calls.lock().unwrap().push(destination.clone());
                Ok(format!("recorded:{destination}"))
            }
        },
    )
}

fn destination_call(destination: &str, call_id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_with_id(
        "record_destination",
        json!({"destination": destination}).to_string(),
        call_id,
    )
}

struct AfterTurn {
    model: Arc<FakeModel>,
    agent: Arc<Agent<()>>,
    state: RunState,
    calls: Calls,
}

/// A server-managed run stopped after its first tool output (`_make_after_turn_state`).
async fn after_turn_state() -> AfterTurn {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let calls: Calls = Arc::default();
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![destination_call("Paris", "call-destination")]);
    let agent = Arc::new(
        Agent::<()>::new("assistant")
            .with_model_instance(model.clone())
            .with_tools(vec![destination_tool(&calls)]),
    );
    let mut streamed = Runner::run_streamed_with_options(&agent, "Initial request", options(true));
    let handle = streamed.cancel_handle();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem {
                name: RunItemStreamEventName::ToolOutput,
                ..
            } = event.unwrap()
            {
                handle.cancel(CancelMode::AfterTurn);
            }
        }
    }
    let state = streamed
        .to_state()
        .expect("a cancelled run leaves its state");
    assert_eq!(*calls.lock().unwrap(), vec!["Paris".to_string()]);
    AfterTurn {
        model,
        agent,
        state,
        calls,
    }
}

/// Run `state` and return the error plus the checkpoint the failed run left.
async fn run_failing(
    agent: &Arc<Agent<()>>,
    state: RunState,
    hooks: Arc<dyn RunHooks<()>>,
    streamed: bool,
) -> (AgentsError, RunState) {
    let checkpoint = RunCheckpoint::new();
    let mut run_options = options(true).checkpoint(&checkpoint);
    run_options.hooks = Some(hooks);
    let error = if streamed {
        let mut failed =
            Runner::run_streamed_with_options(agent, RunInput::from(state), run_options);
        let mut error = None;
        {
            let mut events = std::pin::pin!(failed.stream_events());
            while let Some(event) = events.next().await {
                if let Err(e) = event {
                    error = Some(e);
                }
            }
        }
        error.expect("the streamed run must fail")
    } else {
        Runner::run_with_options(agent, RunInput::from(state), run_options)
            .await
            .unwrap_err()
    };
    let state = checkpoint
        .take()
        .expect("the failed run leaves a checkpoint");
    (error, state)
}

fn reload(state: &RunState) -> RunState {
    RunState::from_json(state.to_json().unwrap()).unwrap()
}

struct FailAfterResponse;

#[async_trait]
impl RunHooks<()> for FailAfterResponse {
    async fn try_on_llm_end(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _response: &ModelResponse,
    ) -> Result<()> {
        Err(AgentsError::user("after response"))
    }
}

struct CountAgentResponseHook(Arc<AtomicUsize>);

#[async_trait]
impl AgentHooks<()> for CountAgentResponseHook {
    async fn on_llm_end(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _response: &ModelResponse,
    ) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

async fn acceptance_commits_before_hook_failure(streamed: bool) {
    let mut at = after_turn_state().await;
    let count = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(
        at.agent
            .clone_with(|a| a.hooks = Some(Arc::new(CountAgentResponseHook(count.clone())))),
    );
    at.state.add_input("Late input").unwrap();
    at.model.add_turn(vec![assistant_message("Accepted")]);

    let (error, state) = run_failing(
        &agent,
        at.state.clone(),
        Arc::new(FailAfterResponse),
        streamed,
    )
    .await;
    assert!(error.to_string().contains("after response"), "{error}");
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Late input"),
        1
    );
    assert!(state.pending_input.is_empty());
    assert_eq!(count.load(Ordering::SeqCst), 1);

    let state = reload(&state);
    let queued = at.model.remaining_steps();
    let recovered = Runner::run_with_options(&agent, RunInput::from(state), options(true))
        .await
        .unwrap();
    assert_eq!(recovered.final_output_text().unwrap(), "Accepted");
    assert_eq!(
        count.load(Ordering::SeqCst),
        1,
        "the end hooks must not rerun"
    );
    assert_eq!(at.model.remaining_steps(), queued, "no second model call");
}

/// Python: `test_server_response_acceptance_commits_before_hook_failure`.
#[tokio::test]
async fn acceptance_commits_before_hook_failure_blocking() {
    acceptance_commits_before_hook_failure(false).await;
}

#[tokio::test]
async fn acceptance_commits_before_hook_failure_streamed() {
    acceptance_commits_before_hook_failure(true).await;
}

struct FailToolHook {
    phase: &'static str,
}

#[async_trait]
impl RunHooks<()> for FailToolHook {
    async fn try_on_tool_start(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &Tool<()>,
    ) -> Result<()> {
        if self.phase == "start" {
            return Err(AgentsError::user("tool hook failed"));
        }
        Ok(())
    }

    async fn try_on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &Tool<()>,
        _result: &openai_agents_sdk::tool::ToolOutput,
    ) -> Result<()> {
        if self.phase == "end" {
            return Err(AgentsError::user("tool hook failed"));
        }
        Ok(())
    }
}

async fn tool_side_effect_failure_is_safe(streamed: bool, phase: &'static str) {
    let mut at = after_turn_state().await;
    at.state.add_input("Late input").unwrap();
    at.model
        .add_turn(vec![destination_call("Rome", "call-retry-destination")]);
    at.model.add_turn(vec![assistant_message("Recovered")]);

    let (error, state) = run_failing(
        &at.agent,
        at.state.clone(),
        Arc::new(FailToolHook { phase }),
        streamed,
    )
    .await;
    assert!(error.to_string().contains("tool hook failed"), "{error}");
    assert!(state.pending_input.is_empty());
    let expected: Vec<String> = if phase == "start" {
        vec!["Paris".into()]
    } else {
        vec!["Paris".into(), "Rome".into()]
    };
    assert_eq!(*at.calls.lock().unwrap(), expected);

    let state = reload(&state);
    if phase == "start" {
        let error = Runner::run_with_options(&at.agent, RunInput::from(state), options(true))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("output was not committed"),
            "{error}"
        );
        assert_eq!(*at.calls.lock().unwrap(), vec!["Paris".to_string()]);
        return;
    }
    let recovered = Runner::run_with_options(&at.agent, RunInput::from(state), options(true))
        .await
        .unwrap();
    assert_eq!(recovered.final_output_text().unwrap(), "Recovered");
    assert_eq!(
        *at.calls.lock().unwrap(),
        vec!["Paris".to_string(), "Rome".to_string()]
    );
    assert_eq!(
        count_text(&at.model.last_call().unwrap().input, "Late input"),
        0
    );
}

/// Python: `test_server_accepted_tool_side_effect_failure_is_safe`.
#[tokio::test]
async fn tool_side_effect_failure_is_safe_start_blocking() {
    tool_side_effect_failure_is_safe(false, "start").await;
}

#[tokio::test]
async fn tool_side_effect_failure_is_safe_start_streamed() {
    tool_side_effect_failure_is_safe(true, "start").await;
}

#[tokio::test]
async fn tool_side_effect_failure_is_safe_end_blocking() {
    tool_side_effect_failure_is_safe(false, "end").await;
}

#[tokio::test]
async fn tool_side_effect_failure_is_safe_end_streamed() {
    tool_side_effect_failure_is_safe(true, "end").await;
}

// ---------------------------------------------------------------------------------------------
// test_server_accepted_computer_start_hook_failure_is_not_replayed
// ---------------------------------------------------------------------------------------------

struct RecordingComputer(Arc<Mutex<Vec<String>>>);

#[async_trait]
impl openai_agents_sdk::computer::Computer for RecordingComputer {
    async fn screenshot(&self) -> Result<String> {
        self.0.lock().unwrap().push("screenshot".into());
        Ok("aW1n".into())
    }
    async fn click(&self, _: i64, _: i64, _: &str, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn double_click(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn scroll(&self, _: i64, _: i64, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn type_text(&self, _: &str) -> Result<()> {
        Ok(())
    }
    async fn wait(&self) -> Result<()> {
        Ok(())
    }
    async fn move_mouse(&self, _: i64, _: i64, _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
    async fn keypress(&self, _: &[String]) -> Result<()> {
        Ok(())
    }
    async fn drag(&self, _: &[(i64, i64)], _: Option<&[String]>) -> Result<()> {
        Ok(())
    }
}

struct FailComputerStart(Arc<AtomicUsize>);

#[async_trait]
impl RunHooks<()> for FailComputerStart {
    async fn try_on_tool_start(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
    ) -> Result<()> {
        if matches!(tool, Tool::Computer(_)) {
            self.0.fetch_add(1, Ordering::SeqCst);
            return Err(AgentsError::user("computer hook failed"));
        }
        Ok(())
    }
}

async fn computer_start_hook_failure_is_not_replayed(streamed: bool) {
    let mut at = after_turn_state().await;
    let screenshots = Arc::new(Mutex::new(Vec::new()));
    let agent = Arc::new(at.agent.clone_with(|a| {
        a.tools = vec![
            openai_agents_sdk::tool::ComputerTool::<()>::new(RecordingComputer(
                screenshots.clone(),
            ))
            .into(),
        ];
    }));
    at.state.add_input("Late input").unwrap();
    at.model
        .add_turn(vec![openai_agents_sdk::items::OutputItem::from_value(
            json!({
                "type": "computer_call", "id": "computer-item", "call_id": "computer-call",
                "status": "completed", "action": {"type": "screenshot"},
                "pending_safety_checks": [],
            }),
        )]);
    let count = Arc::new(AtomicUsize::new(0));

    let (error, state) = run_failing(
        &agent,
        at.state.clone(),
        Arc::new(FailComputerStart(count.clone())),
        streamed,
    )
    .await;
    assert!(
        error.to_string().contains("computer hook failed"),
        "{error}"
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(screenshots.lock().unwrap().is_empty());

    let state = reload(&state);
    let error = Runner::run_with_options(&agent, RunInput::from(state), options(true))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("output was not committed"),
        "{error}"
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(screenshots.lock().unwrap().is_empty());
}

/// Python: `test_server_accepted_computer_start_hook_failure_is_not_replayed`.
#[tokio::test]
async fn computer_start_hook_failure_is_not_replayed_blocking() {
    computer_start_hook_failure_is_not_replayed(false).await;
}

#[tokio::test]
async fn computer_start_hook_failure_is_not_replayed_streamed() {
    computer_start_hook_failure_is_not_replayed(true).await;
}
