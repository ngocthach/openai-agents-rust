//! The agent loop. Port of `agents.run_internal`.

mod history;
mod model_call;
mod tools;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentModel, ToolUseBehavior};
use crate::error::{AgentsError, Result, RunErrorDetails};
use crate::guardrail::{InputGuardrail, InputGuardrailResult, OutputGuardrailResult};
use crate::handoffs::{Handoff, HandoffInputData};
use crate::items::{
    FunctionCall, InputItem, ItemHelpers, MessageOutput, ModelResponse, OutputItem, RunInput,
    RunItem, RunItemKind, ToolApprovalItem,
};
use crate::lifecycle::RunHooks;
use crate::memory::Session;
use crate::model_settings::ModelSettings;
use crate::models::interface::{Model, ModelRequest, ModelTracing, ToolSpec};
use crate::result::{RunResult, StreamMsg};
use crate::run::RunOptions;
use crate::run_config::{
    CallModelData, ModelInputData, ReasoningItemIdPolicy, RunConfig, ToolNotFoundBehavior,
};
use crate::run_context::RunContext;
use crate::run_error_handlers::{
    RunErrorData, RunErrorHandler, RunErrorHandlerInput, RunErrorHandlers,
};
use crate::run_state::{CURRENT_SCHEMA_VERSION, RunState};
use crate::stream_events::{RunItemStreamEventName, StreamEvent};
use crate::tool::{FunctionTool, HostedMCPTool, MCPToolApprovalRequest, Tool, ToolOutput};
use crate::tool_guardrails::{ToolInputGuardrailResult, ToolOutputGuardrailResult};
use crate::tracing::{self as tr, Span, SpanData, scope};

pub(crate) use tools::ToolEnv;

/// Sends stream events from the run task to the streamed result.
pub(crate) struct EventSink<C> {
    pub tx: mpsc::UnboundedSender<StreamMsg<C>>,
    pub cancel_after_turn: Arc<AtomicBool>,
}

impl<C> EventSink<C> {
    pub(crate) fn send(&self, ev: StreamEvent<C>) {
        let _ = self.tx.send(StreamMsg::Event(ev));
    }
}

/// What happens after a turn.
enum NextStep<C> {
    FinalOutput(Value),
    Handoff(Arc<Agent<C>>),
    RunAgain,
    Interruption(Vec<ToolApprovalItem>),
}

/// The classified tool activity of a model response.
struct ProcessedResponse<C> {
    functions: Vec<(FunctionCall, Arc<FunctionTool<C>>)>,
    handoffs: Vec<(FunctionCall, Handoff<C>)>,
    mcp_approvals: Vec<(Value, Option<HostedMCPTool<C>>)>,
    tools_used: Vec<String>,
    last_message: Option<MessageOutput>,
    has_tool_activity: bool,
}

struct RunLoop<C> {
    starting_agent: Arc<Agent<C>>,
    current_agent: Arc<Agent<C>>,
    ctx: RunContext<C>,
    config: RunConfig<C>,
    hooks: Option<Arc<dyn RunHooks<C>>>,
    error_handlers: RunErrorHandlers<C>,
    max_turns: Option<usize>,
    session: Option<Arc<dyn Session>>,
    sink: Option<EventSink<C>>,

    original_input: Vec<InputItem>,
    /// Model input prefix: session history merged with the run input (or the run input).
    model_prefix: Vec<InputItem>,
    /// The run input items to persist to the session.
    session_input: Vec<InputItem>,
    generated_items: Vec<RunItem>,
    /// Items forwarded to the model (differs from `generated_items` after handoff filters).
    model_items: Vec<RunItem>,
    model_responses: Vec<ModelResponse>,
    current_turn: usize,
    input_guardrail_results: Vec<InputGuardrailResult>,
    output_guardrail_results: Vec<OutputGuardrailResult>,
    tool_input_guardrail_results: Vec<ToolInputGuardrailResult>,
    tool_output_guardrail_results: Vec<ToolOutputGuardrailResult>,
    tools_used: HashMap<String, Vec<String>>,
    previous_response_id: Option<String>,
    conversation_id: Option<String>,
    auto_previous_response_id: bool,
    server_sent_items: usize,
    server_input_sent: bool,
    session_saved_items: usize,
    input_saved_to_session: bool,
    pending_input: Vec<InputItem>,
    resume_pending_tools: bool,
    is_resumed: bool,
    should_run_agent_start_hooks: bool,
    agent_span: Option<Span>,
    trace_id: Option<String>,
    /// Index in `generated_items` where the current turn's items start.
    turn_start: usize,
}

fn find_agent<C>(start: &Arc<Agent<C>>, name: &str) -> Option<Arc<Agent<C>>> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(a) = queue.pop_front() {
        if a.name == name {
            return Some(a);
        }
        if !seen.insert(Arc::as_ptr(&a) as usize) {
            continue;
        }
        for h in &a.handoffs {
            if let Some(t) = h.agent.as_ref() {
                queue.push_back(t.clone());
            }
        }
    }
    None
}

fn strip_reasoning_id(item: InputItem, policy: ReasoningItemIdPolicy) -> InputItem {
    if policy == ReasoningItemIdPolicy::Omit && item.item_type() == Some("reasoning") {
        let mut v = item.0;
        if let Value::Object(m) = &mut v {
            m.remove("id");
        }
        return InputItem(v);
    }
    item
}

fn items_to_input(items: &[RunItem]) -> Vec<InputItem> {
    items
        .iter()
        .filter(|i| !matches!(i.kind, RunItemKind::ToolApproval { .. }))
        .map(RunItem::to_input_item)
        .collect()
}

fn call_id_of(item: &RunItem) -> Option<String> {
    match &item.kind {
        RunItemKind::ToolCallOutput { raw_item, .. }
        | RunItemKind::HandoffOutput {
            raw_item: InputItem(raw_item),
            ..
        } => raw_item
            .get("call_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        RunItemKind::MCPApprovalResponse { raw_item } => raw_item
            .get("approval_request_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

fn clone_simple_error(e: &AgentsError) -> AgentsError {
    match e {
        AgentsError::MaxTurnsExceeded { message, .. } => AgentsError::MaxTurnsExceeded {
            message: message.clone(),
            run_data: None,
        },
        AgentsError::ModelRefusalError { refusal, .. } => AgentsError::ModelRefusalError {
            refusal: refusal.clone(),
            run_data: None,
        },
        other => AgentsError::model_behavior(other.to_string()),
    }
}

/// Entry point used by `Runner`.
pub(crate) async fn run_loop<C: Send + Sync + 'static>(
    starting_agent: Arc<Agent<C>>,
    input: RunInput,
    options: RunOptions<C>,
    sink: Option<EventSink<C>>,
) -> Result<RunResult<C>> {
    let RunOptions {
        context,
        max_turns,
        hooks,
        run_config,
        error_handlers,
        previous_response_id,
        auto_previous_response_id,
        conversation_id,
        session,
        shares_parent_context,
    } = options;

    let mut lp = RunLoop {
        starting_agent: starting_agent.clone(),
        current_agent: starting_agent.clone(),
        ctx: context,
        config: run_config,
        hooks,
        error_handlers,
        max_turns,
        session,
        sink,
        original_input: Vec::new(),
        model_prefix: Vec::new(),
        session_input: Vec::new(),
        generated_items: Vec::new(),
        model_items: Vec::new(),
        model_responses: Vec::new(),
        current_turn: 0,
        input_guardrail_results: Vec::new(),
        output_guardrail_results: Vec::new(),
        tool_input_guardrail_results: Vec::new(),
        tool_output_guardrail_results: Vec::new(),
        tools_used: HashMap::new(),
        previous_response_id,
        conversation_id,
        auto_previous_response_id,
        server_sent_items: 0,
        server_input_sent: false,
        session_saved_items: 0,
        input_saved_to_session: false,
        pending_input: Vec::new(),
        resume_pending_tools: false,
        is_resumed: false,
        should_run_agent_start_hooks: true,
        agent_span: None,
        trace_id: None,
        turn_start: 0,
    };

    let server_managed = lp.conversation_id.is_some()
        || lp.previous_response_id.is_some()
        || lp.auto_previous_response_id;
    if lp.session.is_some() && server_managed {
        return Err(AgentsError::user(
            "Session persistence cannot be combined with conversation_id, previous_response_id, or auto_previous_response_id.",
        ));
    }

    match input {
        RunInput::State(state) => lp.restore(*state, !shares_parent_context)?,
        other => {
            let items = other.to_items();
            lp.original_input = items.clone();
            lp.session_input = items.clone();
            lp.model_prefix = items;
        }
    }

    // Tracing: reuse the current trace or start a new one.
    let existing = scope::current_trace();
    let created = if existing.is_none() {
        let mut b = tr::trace(lp.config.workflow_name.clone()).disabled(lp.config.tracing_disabled);
        if let Some(id) = lp.config.trace_id.clone().or_else(|| lp.trace_id.clone()) {
            b = b.trace_id(id);
        }
        if let Some(g) = &lp.config.group_id {
            b = b.group_id(g.clone());
        }
        if let Some(m) = &lp.config.trace_metadata {
            b = b.metadata(m.clone());
        }
        if let Some(k) = &lp.config.tracing_api_key {
            b = b.tracing_api_key(k.clone());
        }
        Some(b.build())
    } else {
        None
    };
    let trace = created.clone().or(existing).expect("trace");
    lp.trace_id = Some(trace.trace_id().to_string());

    let body = async {
        let task_span = if lp.config.include_task_and_turn_spans && !lp.config.tracing_disabled {
            let s = tr::create_span(
                SpanData::Task {
                    name: lp.config.workflow_name.clone(),
                    usage: None,
                },
                None,
                false,
            );
            s.start();
            Some(s)
        } else {
            None
        };
        let res = match &task_span {
            Some(s) => scope::with_span(s.clone(), Box::pin(lp.run())).await,
            None => Box::pin(lp.run()).await,
        };
        if let Some(s) = task_span {
            let usage = lp.ctx.usage();
            s.update_data(|d| {
                if let SpanData::Task { usage: u, .. } = d {
                    *u = Some(json!({
                        "requests": usage.requests,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    }));
                }
            });
            if let Err(e) = &res {
                s.set_error("Run failed", Some(json!({"error": e.to_string()})));
            }
            s.finish();
        }
        if let Some(s) = lp.agent_span.take() {
            s.finish();
        }
        res
    };
    match created {
        Some(t) => t.run(body).await,
        None => body.await,
    }
}

impl<C: Send + Sync + 'static> RunLoop<C> {
    fn restore(&mut self, state: RunState, restore_usage: bool) -> Result<()> {
        if state.schema_version != CURRENT_SCHEMA_VERSION {
            return Err(AgentsError::user(format!(
                "Unsupported run state schema version {}",
                state.schema_version
            )));
        }
        if state.final_output.is_some() {
            return Err(AgentsError::user(
                "This run state already finished; start a new run instead.",
            ));
        }
        let agent =
            find_agent(&self.starting_agent, &state.current_agent_name).ok_or_else(|| {
                AgentsError::user(format!(
                    "Could not find agent '{}' reachable from the starting agent",
                    state.current_agent_name
                ))
            })?;
        self.current_agent = agent;
        self.ctx.import_approvals(&state.approvals);
        self.ctx
            .import_nested_states(&state.nested_agent_tool_states);
        if restore_usage {
            self.ctx.set_usage(state.usage.clone());
        }
        self.original_input = state.original_input.clone();
        self.session_input = state.original_input.clone();
        self.generated_items = state
            .generated_items
            .iter()
            .filter(|i| !matches!(i.kind, RunItemKind::ToolApproval { .. }))
            .cloned()
            .collect();
        self.model_items = self.generated_items.clone();
        self.model_responses = state.model_responses.clone();
        self.current_turn = state.current_turn;
        if let Some(m) = state.max_turns
            && self.max_turns.is_some()
        {
            self.max_turns = Some(m);
        }
        self.input_guardrail_results = state.input_guardrail_results.clone();
        self.output_guardrail_results = state.output_guardrail_results.clone();
        self.tool_input_guardrail_results = state.tool_input_guardrail_results.clone();
        self.tool_output_guardrail_results = state.tool_output_guardrail_results.clone();
        if self.previous_response_id.is_none() {
            self.previous_response_id = state.previous_response_id.clone();
        }
        if self.conversation_id.is_none() {
            self.conversation_id = state.conversation_id.clone();
        }
        self.auto_previous_response_id |= state.auto_previous_response_id;
        self.server_sent_items = state.server_sent_items.min(self.model_items.len());
        self.server_input_sent = state.server_input_sent;
        self.session_saved_items = state.session_saved_items.min(self.generated_items.len());
        self.input_saved_to_session = state.input_saved_to_session;
        self.pending_input = state.pending_input.clone();
        self.resume_pending_tools = state.pending_tool_processing;
        self.trace_id = state.trace_id.clone();
        self.turn_start = state.turn_start_index.min(self.generated_items.len());
        self.is_resumed = true;
        self.should_run_agent_start_hooks = false;
        Ok(())
    }

    fn emit(&self, ev: StreamEvent<C>) {
        if let Some(s) = &self.sink {
            s.send(ev);
        }
    }

    fn emit_item(&self, name: RunItemStreamEventName, item: &RunItem) {
        self.emit(StreamEvent::RunItem {
            name,
            item: item.clone(),
        });
    }

    fn push_item(&mut self, item: RunItem, event: Option<RunItemStreamEventName>) {
        if let Some(name) = event {
            self.emit_item(name, &item);
        }
        self.model_items.push(item.clone());
        self.generated_items.push(item);
    }

    fn run_details(&self) -> RunErrorDetails {
        RunErrorDetails {
            input: self.original_input.clone(),
            new_items: self.generated_items.clone(),
            raw_responses: self.model_responses.clone(),
            last_agent: self.current_agent.name.clone(),
            usage: self.ctx.usage(),
            input_guardrail_results: self.input_guardrail_results.clone(),
            output_guardrail_results: self.output_guardrail_results.clone(),
            tool_input_guardrail_results: self.tool_input_guardrail_results.clone(),
            tool_output_guardrail_results: self.tool_output_guardrail_results.clone(),
        }
    }

    fn server_managed(&self) -> bool {
        self.conversation_id.is_some()
            || self.previous_response_id.is_some()
            || self.auto_previous_response_id
    }

    async fn run(&mut self) -> Result<RunResult<C>> {
        match self.run_inner().await {
            Ok(r) => Ok(r),
            Err(e) => {
                let details = self.run_details();
                Err(e.with_run_data(details))
            }
        }
    }

    fn start_agent_span(&mut self, tools: &[Tool<C>], handoffs: &[Handoff<C>]) {
        if self.agent_span.is_some() {
            return;
        }
        let agent = &self.current_agent;
        let span = tr::create_span(
            SpanData::Agent {
                name: agent.name.clone(),
                handoffs: Some(handoffs.iter().map(|h| h.agent_name.clone()).collect()),
                tools: Some(tools.iter().map(Tool::name).collect()),
                output_type: Some(
                    agent
                        .output_type
                        .as_ref()
                        .map(|o| o.name())
                        .unwrap_or_else(|| "str".to_string()),
                ),
            },
            None,
            false,
        );
        span.start();
        self.agent_span = Some(span);
    }

    async fn run_inner(&mut self) -> Result<RunResult<C>> {
        self.emit(StreamEvent::AgentUpdated {
            new_agent: self.current_agent.clone(),
        });
        loop {
            if let Some(s) = &self.sink
                && s.cancel_after_turn.load(Ordering::SeqCst)
            {
                self.save_session().await?;
                return Ok(self.build_result(None, Vec::new()));
            }

            let step = if self.resume_pending_tools {
                self.resume_pending_tools = false;
                let tools = self.current_agent.get_all_tools(&self.ctx).await?;
                let handoffs = self.current_agent.get_enabled_handoffs(&self.ctx).await;
                self.start_agent_span(&tools, &handoffs);
                let span = self.agent_span.clone().expect("agent span");
                scope::with_span(span, Box::pin(self.resume_tools(tools, handoffs))).await?
            } else {
                if let Some(max) = self.max_turns
                    && self.current_turn >= max
                {
                    return self.handle_max_turns(max).await;
                }
                self.current_turn += 1;
                let tools = self.current_agent.get_all_tools(&self.ctx).await?;
                let handoffs = self.current_agent.get_enabled_handoffs(&self.ctx).await;
                self.start_agent_span(&tools, &handoffs);
                let agent_span = self.agent_span.clone().expect("agent span");
                if self.config.include_task_and_turn_spans && !self.config.tracing_disabled {
                    let turn_span = tr::create_span(
                        SpanData::Turn {
                            turn: self.current_turn as u64,
                            agent_name: self.current_agent.name.clone(),
                            usage: None,
                        },
                        Some(&agent_span),
                        false,
                    );
                    turn_span.start();
                    let before = self.ctx.usage();
                    let res = scope::with_span(
                        turn_span.clone(),
                        Box::pin(self.run_turn(tools, handoffs)),
                    )
                    .await;
                    let after = self.ctx.usage();
                    turn_span.update_data(|d| {
                        if let SpanData::Turn { usage, .. } = d {
                            *usage = Some(json!({
                                "requests": after.requests - before.requests,
                                "input_tokens": after.input_tokens - before.input_tokens,
                                "output_tokens": after.output_tokens - before.output_tokens,
                            }));
                        }
                    });
                    if let Err(e) = &res {
                        turn_span.set_error("Turn failed", Some(json!({"error": e.to_string()})));
                    }
                    turn_span.finish();
                    res?
                } else {
                    scope::with_span(agent_span, Box::pin(self.run_turn(tools, handoffs))).await?
                }
            };

            match step {
                NextStep::FinalOutput(output) => {
                    return Box::pin(self.finish_with_output(output)).await;
                }
                NextStep::Interruption(items) => {
                    self.save_session().await?;
                    self.notify_session_completed().await?;
                    return Ok(self.build_result(None, items));
                }
                NextStep::Handoff(agent) => {
                    if let Some(s) = self.agent_span.take() {
                        s.finish();
                    }
                    self.current_agent = agent;
                    self.should_run_agent_start_hooks = true;
                    self.emit(StreamEvent::AgentUpdated {
                        new_agent: self.current_agent.clone(),
                    });
                    self.save_session().await?;
                }
                NextStep::RunAgain => {
                    self.save_session().await?;
                }
            }
        }
    }

    async fn handle_max_turns(&mut self, max: usize) -> Result<RunResult<C>> {
        let err = AgentsError::MaxTurnsExceeded {
            message: format!("Max turns ({max}) exceeded"),
            run_data: None,
        };
        let handler = self.error_handlers.max_turns.clone();
        self.apply_error_handler(handler, err).await
    }

    async fn apply_error_handler(
        &mut self,
        handler: Option<RunErrorHandler<C>>,
        err: AgentsError,
    ) -> Result<RunResult<C>> {
        let Some(h) = handler else {
            return Err(err);
        };
        let input = RunErrorHandlerInput {
            error: Arc::new(clone_simple_error(&err)),
            context: self.ctx.clone(),
            run_data: RunErrorData {
                input: self.original_input.clone(),
                new_items: self.generated_items.clone(),
                history: {
                    let mut h = self.original_input.clone();
                    h.extend(items_to_input(&self.generated_items));
                    h
                },
                raw_responses: self.model_responses.clone(),
                last_agent: self.current_agent.clone(),
            },
        };
        match h(input).await? {
            None => Err(err),
            Some(res) => {
                if let Some(schema) = &self.current_agent.output_type
                    && !schema.is_plain_text()
                {
                    schema.validate_json(&res.final_output.to_string())?;
                }
                if res.include_in_history {
                    let text = match &res.final_output {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let item = RunItem::new(
                        self.current_agent.name.clone(),
                        RunItemKind::MessageOutput {
                            raw_item: MessageOutput::text(text),
                        },
                    );
                    self.push_item(item, Some(RunItemStreamEventName::MessageOutputCreated));
                }
                self.save_session().await?;
                let mut r = self.build_result(Some(res.final_output), Vec::new());
                r.state.final_output = r.final_output.clone();
                Ok(r)
            }
        }
    }

    async fn finish_with_output(&mut self, output: Value) -> Result<RunResult<C>> {
        let agent = self.current_agent.clone();
        let mut guardrails = agent.output_guardrails.clone();
        guardrails.extend(self.config.output_guardrails.iter().cloned());
        if !guardrails.is_empty() {
            let futs = guardrails.iter().map(|g| {
                let ctx = self.ctx.clone();
                let agent = agent.clone();
                let output = output.clone();
                async move {
                    let span = tr::guardrail_span(&g.name, false);
                    span.start();
                    let r = g.run(ctx, agent, output).await;
                    if let Ok(res) = &r {
                        let tripped = res.output.tripwire_triggered;
                        span.update_data(|d| {
                            if let SpanData::Guardrail { triggered, .. } = d {
                                *triggered = tripped;
                            }
                        });
                    }
                    span.finish();
                    r
                }
            });
            let results = futures::future::join_all(futs).await;
            for r in results {
                let r = r?;
                self.output_guardrail_results.push(r.clone());
                if r.output.tripwire_triggered {
                    self.save_session().await?;
                    return Err(AgentsError::OutputGuardrailTripwireTriggered {
                        guardrail_result: Box::new(r),
                        run_data: None,
                    });
                }
            }
        }
        if let Some(h) = &self.hooks {
            h.on_agent_end(&self.ctx, &agent, &output).await;
        }
        if let Some(h) = &agent.hooks {
            h.on_end(&self.ctx, &agent, &output).await;
        }
        self.save_session().await?;
        self.notify_session_completed().await?;
        let mut r = self.build_result(Some(output), Vec::new());
        r.state.final_output = r.final_output.clone();
        Ok(r)
    }

    fn build_result(
        &self,
        final_output: Option<Value>,
        interruptions: Vec<ToolApprovalItem>,
    ) -> RunResult<C> {
        let state = RunState {
            schema_version: CURRENT_SCHEMA_VERSION.to_string(),
            current_turn: self.current_turn,
            max_turns: self.max_turns,
            current_agent_name: self.current_agent.name.clone(),
            original_input: self.original_input.clone(),
            generated_items: self.generated_items.clone(),
            model_responses: self.model_responses.clone(),
            usage: self.ctx.usage(),
            approvals: self.ctx.export_approvals(),
            nested_agent_tool_states: self.ctx.nested_states(),
            input_guardrail_results: self.input_guardrail_results.clone(),
            output_guardrail_results: self.output_guardrail_results.clone(),
            tool_input_guardrail_results: self.tool_input_guardrail_results.clone(),
            tool_output_guardrail_results: self.tool_output_guardrail_results.clone(),
            pending_tool_processing: !interruptions.is_empty(),
            interruptions: interruptions.clone(),
            pending_input: Vec::new(),
            previous_response_id: self.previous_response_id.clone(),
            conversation_id: self.conversation_id.clone(),
            auto_previous_response_id: self.auto_previous_response_id,
            server_sent_items: self.server_sent_items,
            server_input_sent: self.server_input_sent,
            session_saved_items: self.session_saved_items,
            input_saved_to_session: self.input_saved_to_session,
            final_output: None,
            trace_id: self.trace_id.clone(),
            context: None,
            turn_start_index: self.turn_start,
        };
        RunResult {
            input: self.original_input.clone(),
            new_items: self.generated_items.clone(),
            raw_responses: self.model_responses.clone(),
            final_output,
            last_agent: self.current_agent.clone(),
            input_guardrail_results: self.input_guardrail_results.clone(),
            output_guardrail_results: self.output_guardrail_results.clone(),
            tool_input_guardrail_results: self.tool_input_guardrail_results.clone(),
            tool_output_guardrail_results: self.tool_output_guardrail_results.clone(),
            context_wrapper: self.ctx.clone(),
            interruptions,
            state,
        }
    }

    async fn save_session(&mut self) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let mut items = Vec::new();
        if !self.input_saved_to_session {
            items.extend(self.session_input.clone());
        }
        let new: Vec<InputItem> = items_to_input(&self.generated_items[self.session_saved_items..]);
        items.extend(new);
        if !items.is_empty() {
            session.add_items(items).await?;
        }
        self.input_saved_to_session = true;
        self.session_saved_items = self.generated_items.len();
        Ok(())
    }

    /// Notify the session that the run finished (used for automatic compaction).
    async fn notify_session_completed(&mut self) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let has_local_tool_outputs = self.generated_items
            [self.turn_start.min(self.generated_items.len())..]
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }));
        let info = crate::memory::SessionRunCompletion {
            response_id: self
                .model_responses
                .last()
                .and_then(|r| r.response_id.clone()),
            store: self.current_agent.model_settings.store,
            has_local_tool_outputs,
        };
        let usage = session.on_run_completed(&info).await?;
        if usage.requests > 0 || usage.total_tokens > 0 {
            self.ctx.add_usage(&usage);
        }
        Ok(())
    }

    async fn load_session_history(&mut self) -> Result<()> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let defaults = session.session_settings().unwrap_or_default();
        let settings = defaults.resolve(self.config.session_settings.as_ref());
        let history = session.get_items(settings.limit).await?;
        self.model_prefix = match &self.config.session_input_callback {
            Some(cb) => cb(history, self.session_input.clone()).await?,
            None => {
                let mut h = history;
                h.extend(self.session_input.clone());
                h
            }
        };
        Ok(())
    }

    fn resolve_model(&self, agent: &Agent<C>) -> Result<Arc<dyn Model>> {
        let choice = self.config.model.as_ref().or(agent.model.as_ref());
        match choice {
            Some(AgentModel::Instance(m)) => Ok(m.clone()),
            Some(AgentModel::Name(n)) => self.config.model_provider.get_model(Some(n)),
            None => self.config.model_provider.get_model(None),
        }
    }

    fn resolve_settings(&self, agent: &Agent<C>, model: &Arc<dyn Model>) -> ModelSettings {
        let base = if agent.model_settings_explicit {
            agent.model_settings.clone()
        } else {
            model
                .default_model_settings()
                .map(|d| d.resolve(Some(&agent.model_settings)))
                .unwrap_or_else(|| agent.model_settings.clone())
        };
        let mut s = base.resolve(self.config.model_settings.as_ref());
        if agent.reset_tool_choice
            && self
                .tools_used
                .get(&agent.name)
                .map(|v| !v.is_empty())
                .unwrap_or(false)
        {
            s.tool_choice = None;
        }
        s
    }

    fn build_model_input(&mut self) -> Vec<InputItem> {
        let policy = self.config.reasoning_item_id_policy;
        if self.server_managed() {
            let mut input = Vec::new();
            if !self.server_input_sent {
                input.extend(self.model_prefix.clone());
            }
            input.extend(
                items_to_input(
                    &self.model_items[self.server_sent_items.min(self.model_items.len())..],
                )
                .into_iter()
                .map(|i| strip_reasoning_id(i, policy)),
            );
            input
        } else {
            let mut input = self.model_prefix.clone();
            input.extend(
                items_to_input(&self.model_items)
                    .into_iter()
                    .map(|i| strip_reasoning_id(i, policy)),
            );
            input
        }
    }

    async fn run_input_guardrails(
        &self,
        guardrails: Vec<InputGuardrail<C>>,
        input: Vec<InputItem>,
    ) -> Result<Vec<InputGuardrailResult>> {
        let agent = self.current_agent.clone();
        let futs = guardrails.into_iter().map(|g| {
            let ctx = self.ctx.clone();
            let agent = agent.clone();
            let input = input.clone();
            async move {
                let span = tr::guardrail_span(&g.name, false);
                span.start();
                let r = g.run(ctx, agent, input).await;
                if let Ok(res) = &r {
                    let tripped = res.output.tripwire_triggered;
                    span.update_data(|d| {
                        if let SpanData::Guardrail { triggered, .. } = d {
                            *triggered = tripped;
                        }
                    });
                }
                span.finish();
                r
            }
        });
        futures::future::join_all(futs).await.into_iter().collect()
    }

    fn record_input_guardrails(&mut self, results: Vec<InputGuardrailResult>) -> Result<()> {
        let mut tripped = None;
        for r in results {
            if r.output.tripwire_triggered && tripped.is_none() {
                tripped = Some(r.clone());
            }
            self.input_guardrail_results.push(r);
        }
        match tripped {
            Some(r) => Err(AgentsError::InputGuardrailTripwireTriggered {
                guardrail_result: Box::new(r),
                run_data: None,
            }),
            None => Ok(()),
        }
    }

    async fn run_turn(
        &mut self,
        tools: Vec<Tool<C>>,
        handoffs: Vec<Handoff<C>>,
    ) -> Result<NextStep<C>> {
        let agent = self.current_agent.clone();
        if self.should_run_agent_start_hooks {
            self.should_run_agent_start_hooks = false;
            if let Some(h) = &self.hooks {
                h.on_agent_start(&self.ctx, &agent).await;
            }
            if let Some(h) = &agent.hooks {
                h.on_start(&self.ctx, &agent).await;
            }
        }

        self.turn_start = self.generated_items.len();
        let first_turn = self.current_turn == 1 && !self.is_resumed;
        if first_turn {
            self.load_session_history().await?;
        }

        // Input guardrails apply to the first turn and to staged input on resume.
        let mut guardrail_input: Option<Vec<InputItem>> = None;
        if first_turn {
            guardrail_input = Some(self.original_input.clone());
        }
        if !self.pending_input.is_empty() {
            let staged = std::mem::take(&mut self.pending_input);
            guardrail_input = Some(staged.clone());
            for item in staged {
                let ri = RunItem::new(agent.name.clone(), RunItemKind::Input { raw_item: item });
                self.push_item(ri, None);
            }
        }
        let (blocking, parallel): (Vec<_>, Vec<_>) = if guardrail_input.is_some() {
            let mut all = self.config.input_guardrails.clone();
            all.extend(agent.input_guardrails.iter().cloned());
            all.into_iter().partition(|g| !g.run_in_parallel)
        } else {
            (Vec::new(), Vec::new())
        };
        if !blocking.is_empty() {
            let results = self
                .run_input_guardrails(blocking, guardrail_input.clone().unwrap_or_default())
                .await?;
            self.record_input_guardrails(results)?;
        }

        let system_prompt = agent.get_system_prompt(&self.ctx).await?;
        let prompt = agent.get_prompt(&self.ctx).await?;
        let model = self.resolve_model(&agent)?;
        let settings = self.resolve_settings(&agent, &model);

        let mut input = self.build_model_input();
        let mut instructions = system_prompt;
        if let Some(f) = &self.config.call_model_input_filter {
            let r = f(CallModelData {
                model_data: ModelInputData {
                    input,
                    instructions,
                },
                agent: agent.clone(),
                context: self.ctx.clone(),
            })
            .await?;
            input = r.input;
            instructions = r.instructions;
        }

        let tool_specs: Vec<ToolSpec> = tools
            .iter()
            .map(|t| match t {
                Tool::Function(f) => ToolSpec::Function(f.to_spec()),
                other => {
                    let (param, includes) = other.to_responses_param();
                    ToolSpec::Hosted { param, includes }
                }
            })
            .collect();
        let tracing_mode = if self.config.tracing_disabled {
            ModelTracing::Disabled
        } else if self.config.trace_include_sensitive_data {
            ModelTracing::Enabled
        } else {
            ModelTracing::EnabledWithoutData
        };
        let previous_response_id =
            if self.auto_previous_response_id || self.previous_response_id.is_some() {
                self.previous_response_id.clone()
            } else {
                None
            };
        let request = ModelRequest {
            system_instructions: instructions.clone(),
            input: input.clone(),
            model_settings: settings,
            tools: tool_specs,
            output_schema: agent.output_type.clone(),
            handoffs: handoffs.iter().map(Handoff::to_spec).collect(),
            tracing: tracing_mode,
            previous_response_id,
            conversation_id: self.conversation_id.clone(),
            prompt,
        };

        if let Some(h) = &self.hooks {
            h.on_llm_start(&self.ctx, &agent, instructions.as_deref(), &input)
                .await;
        }
        if let Some(h) = &agent.hooks {
            h.on_llm_start(&self.ctx, &agent, instructions.as_deref(), &input)
                .await;
        }

        let response = if parallel.is_empty() {
            Box::pin(model_call::call_model(&model, request, self.sink.as_ref())).await?
        } else {
            let g_input = guardrail_input.clone().unwrap_or_default();
            let (guard_results, model_result) = {
                let model_fut =
                    Box::pin(model_call::call_model(&model, request, self.sink.as_ref()));
                let guard_fut = self.run_input_guardrails(parallel, g_input);
                tokio::pin!(model_fut);
                tokio::pin!(guard_fut);
                tokio::select! {
                    g = &mut guard_fut => {
                        let g = g?;
                        if g.iter().any(|r| r.output.tripwire_triggered) {
                            (g, None)
                        } else {
                            let m = (&mut model_fut).await;
                            (g, Some(m))
                        }
                    }
                    m = &mut model_fut => {
                        let g = (&mut guard_fut).await?;
                        (g, Some(m))
                    }
                }
            };
            self.record_input_guardrails(guard_results)?;
            model_result.expect("model result")?
        };

        self.ctx.add_usage(&response.usage);
        if let Some(h) = &self.hooks {
            h.on_llm_end(&self.ctx, &agent, &response).await;
        }
        if let Some(h) = &agent.hooks {
            h.on_llm_end(&self.ctx, &agent, &response).await;
        }
        self.model_responses.push(response.clone());

        let processed =
            self.process_response(&agent, &tools, &handoffs, &response, true, &HashSet::new())?;
        if self.server_managed() {
            self.server_input_sent = true;
            if (self.auto_previous_response_id || self.previous_response_id.is_some())
                && let Some(id) = &response.response_id
            {
                self.previous_response_id = Some(id.clone());
            }
            self.server_sent_items = self.model_items.len();
        }
        Box::pin(self.execute_side_effects(&agent, processed)).await
    }

    /// Re-process the last model response after approvals were resolved.
    async fn resume_tools(
        &mut self,
        tools: Vec<Tool<C>>,
        handoffs: Vec<Handoff<C>>,
    ) -> Result<NextStep<C>> {
        let agent = self.current_agent.clone();
        let Some(response) = self.model_responses.last().cloned() else {
            return Ok(NextStep::RunAgain);
        };
        let done: HashSet<String> = self.generated_items.iter().filter_map(call_id_of).collect();
        let processed =
            self.process_response(&agent, &tools, &handoffs, &response, false, &done)?;
        Box::pin(self.execute_side_effects(&agent, processed)).await
    }

    /// Classify the output of a model response. When `append` is true the items are added to
    /// the run (and streamed); `skip` lists call ids that already have outputs.
    fn process_response(
        &mut self,
        agent: &Arc<Agent<C>>,
        tools: &[Tool<C>],
        handoffs: &[Handoff<C>],
        response: &ModelResponse,
        append: bool,
        skip: &HashSet<String>,
    ) -> Result<ProcessedResponse<C>> {
        let mut processed = ProcessedResponse {
            functions: Vec::new(),
            handoffs: Vec::new(),
            mcp_approvals: Vec::new(),
            tools_used: Vec::new(),
            last_message: None,
            has_tool_activity: false,
        };
        let function_map: HashMap<&str, &Arc<FunctionTool<C>>> = tools
            .iter()
            .filter_map(|t| t.as_function().map(|f| (f.name.as_str(), f)))
            .collect();
        let handoff_map: HashMap<&str, &Handoff<C>> =
            handoffs.iter().map(|h| (h.tool_name.as_str(), h)).collect();
        let hosted_mcp: Vec<&HostedMCPTool<C>> = tools
            .iter()
            .filter_map(|t| match t {
                Tool::HostedMCP(m) => Some(m),
                _ => None,
            })
            .collect();
        let name = agent.name.clone();

        for out in &response.output {
            match out {
                OutputItem::Message(m) => {
                    processed.last_message = Some(m.clone());
                    if append {
                        let item = RunItem::new(
                            name.clone(),
                            RunItemKind::MessageOutput {
                                raw_item: m.clone(),
                            },
                        );
                        self.push_item(item, Some(RunItemStreamEventName::MessageOutputCreated));
                    }
                }
                OutputItem::Reasoning(r) => {
                    if append {
                        let item = RunItem::new(
                            name.clone(),
                            RunItemKind::Reasoning {
                                raw_item: r.clone(),
                            },
                        );
                        self.push_item(item, Some(RunItemStreamEventName::ReasoningItemCreated));
                    }
                }
                OutputItem::FunctionCall(fc) => {
                    processed.has_tool_activity = true;
                    if let Some(h) = handoff_map.get(fc.name.as_str()) {
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::HandoffCall {
                                    raw_item: fc.clone(),
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::HandoffRequested));
                        }
                        if !skip.contains(&fc.call_id) {
                            processed.handoffs.push((fc.clone(), (*h).clone()));
                        }
                    } else if let Some(t) = function_map.get(fc.name.as_str()) {
                        if append {
                            let item = RunItem::new(
                                name.clone(),
                                RunItemKind::ToolCall {
                                    raw_item: fc.to_value(),
                                },
                            );
                            self.push_item(item, Some(RunItemStreamEventName::ToolCalled));
                        }
                        processed.tools_used.push(fc.name.clone());
                        if !skip.contains(&fc.call_id) {
                            processed.functions.push((fc.clone(), (*t).clone()));
                        }
                    } else {
                        match self.config.tool_not_found_behavior {
                            ToolNotFoundBehavior::RaiseError => {
                                return Err(AgentsError::model_behavior(format!(
                                    "Tool {} not found in agent {}",
                                    fc.name, agent.name
                                )));
                            }
                            ToolNotFoundBehavior::ReturnErrorToModel => {
                                if append && !skip.contains(&fc.call_id) {
                                    let call_item = RunItem::new(
                                        name.clone(),
                                        RunItemKind::ToolCall {
                                            raw_item: fc.to_value(),
                                        },
                                    );
                                    self.push_item(
                                        call_item,
                                        Some(RunItemStreamEventName::ToolCalled),
                                    );
                                    let msg = tools::tool_not_found_message(
                                        &self.config,
                                        &fc.name,
                                        &fc.call_id,
                                    );
                                    let out = ToolOutput::Text(msg);
                                    let raw = ItemHelpers::tool_call_output_item(fc, &out);
                                    let item = RunItem::new(
                                        name.clone(),
                                        RunItemKind::ToolCallOutput {
                                            raw_item: raw,
                                            output: out.to_json_value(),
                                        },
                                    );
                                    self.push_item(item, Some(RunItemStreamEventName::ToolOutput));
                                }
                            }
                        }
                    }
                }
                OutputItem::Other(v) => {
                    let ty = v.get("type").and_then(Value::as_str).unwrap_or_default();
                    match ty {
                        "mcp_list_tools" => {
                            if append {
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::MCPListTools {
                                        raw_item: v.clone(),
                                    },
                                );
                                self.push_item(item, Some(RunItemStreamEventName::MCPListTools));
                            }
                        }
                        "mcp_approval_request" => {
                            processed.has_tool_activity = true;
                            if append {
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::MCPApprovalRequest {
                                        raw_item: v.clone(),
                                    },
                                );
                                self.push_item(
                                    item,
                                    Some(RunItemStreamEventName::MCPApprovalRequested),
                                );
                            }
                            let id = v.get("id").and_then(Value::as_str).unwrap_or_default();
                            if !skip.contains(id) {
                                let label = v.get("server_label").and_then(Value::as_str);
                                let tool = hosted_mcp
                                    .iter()
                                    .find(|t| t.server_label() == label)
                                    .map(|t| (*t).clone());
                                processed.mcp_approvals.push((v.clone(), tool));
                            }
                        }
                        "tool_search_output" => {
                            if append {
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::ToolCallOutput {
                                        raw_item: v.clone(),
                                        output: Value::Null,
                                    },
                                );
                                self.push_item(
                                    item,
                                    Some(RunItemStreamEventName::ToolSearchOutputCreated),
                                );
                            }
                        }
                        _ => {
                            if append {
                                let event = if ty == "tool_search_call" {
                                    RunItemStreamEventName::ToolSearchCalled
                                } else {
                                    RunItemStreamEventName::ToolCalled
                                };
                                let item = RunItem::new(
                                    name.clone(),
                                    RunItemKind::ToolCall {
                                        raw_item: v.clone(),
                                    },
                                );
                                self.push_item(item, Some(event));
                            }
                            if ty.ends_with("_call") {
                                processed.tools_used.push(ty.to_string());
                            }
                        }
                    }
                }
            }
        }
        if !processed.tools_used.is_empty() {
            self.tools_used
                .entry(agent.name.clone())
                .or_default()
                .extend(processed.tools_used.iter().cloned());
        }
        Ok(processed)
    }

    async fn execute_side_effects(
        &mut self,
        agent: &Arc<Agent<C>>,
        processed: ProcessedResponse<C>,
    ) -> Result<NextStep<C>> {
        let mut interruptions: Vec<ToolApprovalItem> = Vec::new();
        let ran_functions = !processed.functions.is_empty();

        // Function tools.
        let results = {
            let env = ToolEnv {
                ctx: &self.ctx,
                agent,
                hooks: self.hooks.as_ref(),
                config: &self.config,
            };
            Box::pin(tools::execute_function_tools(&env, processed.functions)).await?
        };
        let mut function_results = Vec::new();
        for r in results {
            self.tool_input_guardrail_results
                .extend(r.input_guardrail_results);
            self.tool_output_guardrail_results
                .extend(r.output_guardrail_results);
            match r.outcome {
                tools::ToolOutcome::Output { output, item } => {
                    let ri = RunItem::new(
                        agent.name.clone(),
                        RunItemKind::ToolCallOutput {
                            raw_item: item,
                            output: output.to_json_value(),
                        },
                    );
                    self.push_item(ri.clone(), Some(RunItemStreamEventName::ToolOutput));
                    function_results.push(crate::tool::FunctionToolResult {
                        tool_name: r.tool_name,
                        call_id: r.call.call_id.clone(),
                        output,
                        run_item: Some(ri),
                        interruptions: Vec::new(),
                    });
                }
                tools::ToolOutcome::Interruptions(items) => {
                    for it in &items {
                        let ri = RunItem::new(
                            it.agent.clone(),
                            RunItemKind::ToolApproval {
                                raw_item: it.raw_item.clone(),
                                tool_name: it.tool_name.clone(),
                            },
                        );
                        self.generated_items.push(ri);
                    }
                    interruptions.extend(items);
                }
            }
        }

        // Hosted MCP approval requests.
        let mut answered_mcp = false;
        for (request, tool) in processed.mcp_approvals {
            let request_id = request
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let tool_name = request
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let label = request
                .get("server_label")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let identity = format!("{label}:{tool_name}");
            let decision = match tool.as_ref().and_then(|t| t.on_approval_request().cloned()) {
                Some(cb) => {
                    let r = cb(MCPToolApprovalRequest {
                        ctx: self.ctx.clone(),
                        data: request.clone(),
                    })
                    .await;
                    Some((r.approve, r.reason))
                }
                None => self
                    .ctx
                    .is_tool_approved(&agent.name, &identity, &request_id)
                    .map(|a| {
                        (
                            a,
                            self.ctx
                                .rejection_message(&agent.name, &identity, &request_id),
                        )
                    }),
            };
            match decision {
                Some((approve, reason)) => {
                    let mut resp = json!({
                        "type": "mcp_approval_response",
                        "approval_request_id": request_id,
                        "approve": approve,
                    });
                    if let Some(r) = reason.filter(|_| !approve) {
                        resp["reason"] = Value::String(r);
                    }
                    let ri = RunItem::new(
                        agent.name.clone(),
                        RunItemKind::MCPApprovalResponse { raw_item: resp },
                    );
                    self.push_item(ri, Some(RunItemStreamEventName::MCPApprovalResponse));
                    answered_mcp = true;
                }
                None => {
                    let mut extra = serde_json::Map::new();
                    extra.insert("type".into(), Value::String("mcp_approval_request".into()));
                    extra.insert("server_label".into(), Value::String(label));
                    let item = ToolApprovalItem {
                        agent: agent.name.clone(),
                        raw_item: FunctionCall {
                            id: None,
                            call_id: request_id,
                            name: tool_name,
                            arguments: request
                                .get("arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("{}")
                                .to_string(),
                            status: None,
                            extra,
                        },
                        tool_name: identity,
                    };
                    self.generated_items.push(RunItem::new(
                        agent.name.clone(),
                        RunItemKind::ToolApproval {
                            raw_item: item.raw_item.clone(),
                            tool_name: item.tool_name.clone(),
                        },
                    ));
                    interruptions.push(item);
                }
            }
        }

        if !interruptions.is_empty() {
            return Ok(NextStep::Interruption(interruptions));
        }

        // Handoffs.
        if !processed.handoffs.is_empty() {
            return Box::pin(self.execute_handoffs(agent, processed.handoffs)).await;
        }

        // Tool use behavior.
        if ran_functions && !function_results.is_empty() {
            let decision = match &agent.tool_use_behavior {
                ToolUseBehavior::RunLlmAgain => None,
                ToolUseBehavior::StopOnFirstTool => {
                    Some(function_results[0].output.to_json_value())
                }
                ToolUseBehavior::StopAtTools(s) => function_results
                    .iter()
                    .find(|r| s.stop_at_tool_names.contains(&r.tool_name))
                    .map(|r| r.output.to_json_value()),
                ToolUseBehavior::Custom(f) => {
                    let r = f(self.ctx.clone(), function_results.clone()).await?;
                    if r.is_final_output {
                        Some(r.final_output.unwrap_or(Value::Null))
                    } else {
                        None
                    }
                }
            };
            if let Some(out) = decision {
                return Ok(NextStep::FinalOutput(out));
            }
        }
        if ran_functions || answered_mcp {
            return Ok(NextStep::RunAgain);
        }

        // Final output from the last message.
        let Some(message) = processed.last_message else {
            return Ok(NextStep::RunAgain);
        };
        match &agent.output_type {
            Some(schema) if !schema.is_plain_text() => {
                if let Some(refusal) = message.refusal()
                    && message.text_content().is_empty()
                {
                    let err = AgentsError::ModelRefusalError {
                        refusal,
                        run_data: None,
                    };
                    let handler = self.error_handlers.model_refusal.clone();
                    return self.handled_final(handler, err).await;
                }
                let text = message.text_content();
                if text.is_empty() {
                    return Ok(NextStep::RunAgain);
                }
                match schema.validate_json(&text) {
                    Ok(v) => Ok(NextStep::FinalOutput(v)),
                    Err(e) => {
                        let handler = self.error_handlers.invalid_final_output.clone();
                        self.handled_final(handler, e).await
                    }
                }
            }
            _ => {
                let text = message.text_content();
                if !text.is_empty() {
                    Ok(NextStep::FinalOutput(Value::String(text)))
                } else if let Some(r) = message.refusal() {
                    Ok(NextStep::FinalOutput(Value::String(r)))
                } else {
                    Ok(NextStep::RunAgain)
                }
            }
        }
    }

    /// Apply an error handler during turn resolution, mapping a recovery to a final output.
    async fn handled_final(
        &mut self,
        handler: Option<RunErrorHandler<C>>,
        err: AgentsError,
    ) -> Result<NextStep<C>> {
        let Some(h) = handler else {
            return Err(err);
        };
        let input = RunErrorHandlerInput {
            error: Arc::new(clone_simple_error(&err)),
            context: self.ctx.clone(),
            run_data: RunErrorData {
                input: self.original_input.clone(),
                new_items: self.generated_items.clone(),
                history: {
                    let mut h = self.original_input.clone();
                    h.extend(items_to_input(&self.generated_items));
                    h
                },
                raw_responses: self.model_responses.clone(),
                last_agent: self.current_agent.clone(),
            },
        };
        match h(input).await? {
            None => Err(err),
            Some(res) => {
                if let Some(schema) = &self.current_agent.output_type
                    && !schema.is_plain_text()
                {
                    schema.validate_json(&res.final_output.to_string())?;
                }
                if !res.include_in_history {
                    // Drop the rejected message from history.
                    if let Some(pos) = self
                        .generated_items
                        .iter()
                        .rposition(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }))
                    {
                        self.generated_items.remove(pos);
                    }
                    if let Some(pos) = self
                        .model_items
                        .iter()
                        .rposition(|i| matches!(i.kind, RunItemKind::MessageOutput { .. }))
                    {
                        self.model_items.remove(pos);
                    }
                }
                Ok(NextStep::FinalOutput(res.final_output))
            }
        }
    }

    async fn execute_handoffs(
        &mut self,
        agent: &Arc<Agent<C>>,
        handoffs: Vec<(FunctionCall, Handoff<C>)>,
    ) -> Result<NextStep<C>> {
        let mut iter = handoffs.into_iter();
        let (call, handoff) = iter.next().expect("at least one handoff");
        for (extra_call, _) in iter {
            let raw = InputItem::function_call_output(
                extra_call.call_id.clone(),
                "Multiple handoffs detected, ignoring this one.",
            );
            let ri = RunItem::new(
                agent.name.clone(),
                RunItemKind::ToolCallOutput {
                    raw_item: raw.0,
                    output: Value::String("Multiple handoffs detected, ignoring this one.".into()),
                },
            );
            self.push_item(ri, Some(RunItemStreamEventName::ToolOutput));
        }

        let span = tr::handoff_span(&agent.name, None);
        span.start();
        let new_agent = match scope::with_span(
            span.clone(),
            handoff.invoke(self.ctx.clone(), call.arguments.clone()),
        )
        .await
        {
            Ok(a) => a,
            Err(e) => {
                span.set_error(
                    "Error invoking handoff",
                    Some(json!({"error": e.to_string()})),
                );
                span.finish();
                return Err(e);
            }
        };
        span.update_data(|d| {
            if let SpanData::Handoff { to_agent, .. } = d {
                *to_agent = Some(new_agent.name.clone());
            }
        });
        span.finish();

        let pre_handoff_count = self.turn_start.min(self.generated_items.len());
        let output_item = RunItem::new(
            agent.name.clone(),
            RunItemKind::HandoffOutput {
                raw_item: InputItem::function_call_output(
                    call.call_id.clone(),
                    Handoff::<C>::get_transfer_message(&new_agent.name),
                ),
                source_agent: agent.name.clone(),
                target_agent: new_agent.name.clone(),
            },
        );
        self.push_item(output_item, Some(RunItemStreamEventName::HandoffOccured));

        if let Some(h) = &self.hooks {
            h.on_handoff(&self.ctx, agent, &new_agent).await;
        }
        if let Some(h) = &new_agent.hooks {
            h.on_handoff(&self.ctx, &new_agent, agent).await;
        }

        let filter = handoff
            .input_filter
            .clone()
            .or_else(|| self.config.handoff_input_filter.clone());
        if let Some(filter) = filter {
            if self.server_managed() {
                return Err(AgentsError::user(
                    "Handoff input filters are not supported with server-managed conversations.",
                ));
            }
            let data = HandoffInputData {
                input_history: self.model_prefix.clone(),
                pre_handoff_items: self.generated_items[..pre_handoff_count].to_vec(),
                new_items: self.generated_items[pre_handoff_count..].to_vec(),
                input_items: None,
            };
            let filtered = filter(data);
            self.original_input = filtered.input_history.clone();
            self.model_prefix = filtered.input_history;
            let mut generated = filtered.pre_handoff_items.clone();
            generated.extend(filtered.new_items.clone());
            let mut model_items = filtered.pre_handoff_items;
            model_items.extend(filtered.input_items.unwrap_or(filtered.new_items));
            self.session_saved_items = self.session_saved_items.min(generated.len());
            self.generated_items = generated;
            self.model_items = model_items;
        } else if handoff
            .nest_handoff_history
            .unwrap_or(self.config.nest_handoff_history)
            && !self.server_managed()
        {
            let mut transcript = self.model_prefix.clone();
            transcript.extend(items_to_input(&self.model_items));
            self.model_prefix = match &self.config.handoff_history_mapper {
                Some(m) => m(transcript),
                None => vec![history::build_summary_message(&transcript)],
            };
            self.model_items.clear();
            self.server_sent_items = 0;
        }
        Ok(NextStep::Handoff(new_agent))
    }
}
