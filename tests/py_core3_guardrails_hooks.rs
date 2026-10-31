//! Ported from the Python SDK's `tests/test_agent_hooks.py` and `tests/test_agent_llm_hooks.py`:
//! per-agent `AgentHooks` callbacks counted across handoffs, tools, structured output, streamed
//! runs and the turn input they receive.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::{InputItem, ModelResponse};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentHooks, RunContext, RunOptions, Runner, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn some_function() -> FunctionTool<()> {
    FunctionTool::new("some_function", "", |_ctx, _a: Value| async move {
        Ok("result".to_string())
    })
}

/// Counts every agent-scoped callback, as the Python `AgentHooksForTests` does.
#[derive(Default)]
struct CountingAgentHooks {
    events: Mutex<BTreeMap<String, usize>>,
    tool_context_ids: Mutex<Vec<String>>,
    turn_inputs: Mutex<Vec<Vec<InputItem>>>,
}

impl CountingAgentHooks {
    fn bump(&self, name: &str) {
        *self
            .events
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_default() += 1;
    }

    fn events(&self) -> BTreeMap<String, usize> {
        self.events.lock().unwrap().clone()
    }

    fn reset(&self) {
        self.events.lock().unwrap().clear();
        self.tool_context_ids.lock().unwrap().clear();
    }
}

#[async_trait]
impl AgentHooks<()> for CountingAgentHooks {
    async fn on_start(&self, context: &RunContext<()>, _agent: &Agent<()>) {
        self.bump("on_start");
        self.turn_inputs.lock().unwrap().push(context.turn_input());
    }

    async fn on_end(&self, _context: &RunContext<()>, _agent: &Agent<()>, _output: &Value) {
        self.bump("on_end");
    }

    async fn on_handoff(&self, _context: &RunContext<()>, _agent: &Agent<()>, _source: &Agent<()>) {
        self.bump("on_handoff");
    }

    async fn on_tool_start(
        &self,
        context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &openai_agents_sdk::Tool<()>,
    ) {
        self.bump("on_tool_start");
        self.tool_context_ids
            .lock()
            .unwrap()
            .push(context.tool_call_id.clone());
    }

    async fn on_tool_end(
        &self,
        context: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &openai_agents_sdk::Tool<()>,
        _result: &openai_agents_sdk::tool::ToolOutput,
    ) {
        self.bump("on_tool_end");
        self.tool_context_ids
            .lock()
            .unwrap()
            .push(context.tool_call_id.clone());
    }
}

/// Records the LLM callbacks as well.
#[derive(Default)]
struct LlmCountingHooks {
    events: Mutex<BTreeMap<String, usize>>,
}

impl LlmCountingHooks {
    fn bump(&self, name: &str) {
        *self
            .events
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_default() += 1;
    }

    fn events(&self) -> BTreeMap<String, usize> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait]
impl AgentHooks<()> for LlmCountingHooks {
    async fn on_start(&self, _context: &RunContext<()>, _agent: &Agent<()>) {
        self.bump("on_start");
    }

    async fn on_end(&self, _context: &RunContext<()>, _agent: &Agent<()>, _output: &Value) {
        self.bump("on_end");
    }

    async fn on_llm_start(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _system_prompt: Option<&str>,
        _input_items: &[InputItem],
    ) {
        self.bump("on_llm_start");
    }

    async fn on_llm_end(
        &self,
        _context: &RunContext<()>,
        _agent: &Agent<()>,
        _response: &ModelResponse,
    ) {
        self.bump("on_llm_end");
    }
}

fn expect(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn text(s: &str) -> Vec<openai_agents_sdk::items::OutputItem> {
    vec![assistant_message(s)]
}

#[tokio::test]
async fn agent_llm_hooks_fire_once_each_non_streamed() {
    disable_tracing();
    let hooks = Arc::new(LlmCountingHooks::default());
    let model = Arc::new(ScriptedModel::new());
    model.enqueue(ScriptedStep::output(text("hello"))).unwrap();
    let agent = Agent::<()>::new("A")
        .with_model_instance(model)
        .with_tool(some_function())
        .with_hooks(hooks.clone());

    Runner::run(&agent, "hello").await.unwrap();

    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_end", 1)
        ])
    );
}

#[test]
fn agent_llm_hooks_fire_once_each_sync() {
    disable_tracing();
    let hooks = Arc::new(LlmCountingHooks::default());
    let model = Arc::new(ScriptedModel::new());
    model.enqueue(ScriptedStep::output(text("hello"))).unwrap();
    let agent = Agent::<()>::new("A")
        .with_model_instance(model)
        .with_tool(some_function())
        .with_hooks(hooks.clone());

    Runner::run_sync(&agent, "hello").unwrap();

    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_end", 1)
        ])
    );
}

#[tokio::test]
async fn agent_llm_hooks_fire_once_each_streamed() {
    disable_tracing();
    let hooks = Arc::new(LlmCountingHooks::default());
    let model = Arc::new(ScriptedModel::new());
    model.enqueue(ScriptedStep::output(text("hello"))).unwrap();
    let agent = Agent::<()>::new("A")
        .with_model_instance(model)
        .with_tool(some_function())
        .with_hooks(hooks.clone());

    let mut result = Runner::run_streamed_with_options(&agent, "hello", RunOptions::new(()));
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        event.unwrap();
    }

    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_end", 1)
        ])
    );
}

/// Python `test_falsy_agent_hooks_are_invoked` checks `__bool__` on a hooks object; Rust has no
/// object truthiness, so there is nothing to test beyond the counting itself.
#[tokio::test]
async fn agent_hooks_fire_for_a_tool_call_turn() {
    disable_tracing();
    let hooks = Arc::new(CountingAgentHooks::default());
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(vec![function_call(
            "some_function",
            r#"{"a":"b"}"#,
        )]))
        .unwrap();
    model.enqueue(ScriptedStep::output(text("done"))).unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(some_function())
        .with_hooks(hooks.clone());

    Runner::run(&agent, "user_message").await.unwrap();

    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 1),
            ("on_tool_start", 1),
            ("on_tool_end", 1),
            ("on_end", 1)
        ])
    );
}

/// Builds the three-agent graph of the Python tests: `agent_3` (hooked) hands off to `agent_1`
/// or `agent_2`, and `agent_1` hands back to `agent_3`. Rust agents are immutable, so the back
/// edge goes through a second `agent_3` that shares the hooks; it is never reached for more
/// than one hand-back, which is all the scenarios need.
fn three_agent_graph(
    model: &Arc<ScriptedModel>,
    hooks: Arc<CountingAgentHooks>,
    output_type: bool,
) -> (Arc<Agent<()>>, Arc<Agent<()>>) {
    let agent_3_again = {
        let mut builder = Agent::<()>::new("test_3")
            .with_model_instance(model.clone())
            .with_tool(some_function())
            .with_hooks(hooks.clone());
        if output_type {
            builder = builder.with_output_type::<Foo>();
        }
        Arc::new(builder)
    };
    let agent_1 = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(model.clone())
            .with_handoff(agent_3_again),
    );
    let agent_2 = Arc::new(Agent::<()>::new("test_2").with_model_instance(model.clone()));
    let mut builder = Agent::<()>::new("test_3")
        .with_model_instance(model.clone())
        .with_handoff(agent_1.clone())
        .with_handoff(agent_2.clone())
        .with_tool(some_function())
        .with_hooks(hooks);
    if output_type {
        builder = builder.with_output_type::<Foo>();
    }
    (Arc::new(builder), agent_1)
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Foo {
    #[allow(dead_code)]
    a: String,
}

async fn run_non_streamed_scenario(output_type: bool) {
    let hooks = Arc::new(CountingAgentHooks::default());
    let model = Arc::new(ScriptedModel::new());
    let final_message = if output_type {
        openai_agents_sdk::testing::assistant_message(r#"{"a":"b"}"#)
    } else {
        assistant_message("done")
    };
    let (agent_3, agent_1) = three_agent_graph(&model, hooks.clone(), output_type);

    let first = if output_type {
        assistant_message(r#"{"a":"b"}"#)
    } else {
        assistant_message("user_message")
    };
    model.enqueue(ScriptedStep::output(vec![first])).unwrap();
    Runner::run(&agent_3, "user_message").await.unwrap();
    assert_eq!(hooks.events(), expect(&[("on_start", 1), ("on_end", 1)]));
    hooks.reset();

    model
        .extend(vec![
            ScriptedStep::output(vec![function_call("some_function", r#"{"a":"b"}"#)]),
            ScriptedStep::output(vec![assistant_message("a_message"), handoff_call(&agent_1)]),
            ScriptedStep::output(text("done")),
        ])
        .unwrap();
    Runner::run(&agent_3, "user_message").await.unwrap();
    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 1),
            ("on_tool_start", 1),
            ("on_tool_end", 1),
            ("on_handoff", 1)
        ])
    );
    hooks.reset();

    model
        .extend(vec![
            ScriptedStep::output(vec![function_call("some_function", r#"{"a":"b"}"#)]),
            ScriptedStep::output(vec![
                assistant_message("a_message"),
                function_call("some_function", r#"{"a":"b"}"#),
                handoff_call(&agent_1),
            ]),
            ScriptedStep::output(vec![assistant_message("a_message"), handoff_call(&agent_3)]),
            ScriptedStep::output(vec![final_message]),
        ])
        .unwrap();
    Runner::run(&agent_3, "user_message").await.unwrap();
    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 2),
            ("on_tool_start", 2),
            ("on_tool_end", 2),
            ("on_handoff", 1),
            ("on_end", 1),
        ])
    );
}

fn handoff_call(target: &Agent<()>) -> openai_agents_sdk::items::OutputItem {
    function_call(format!("transfer_to_{}", target.name), "{}")
}

#[tokio::test]
async fn non_streamed_agent_hooks_follow_handoffs_and_tools() {
    disable_tracing();
    run_non_streamed_scenario(false).await;
}

async fn run_streamed_scenario(output_type: bool) {
    let hooks = Arc::new(CountingAgentHooks::default());
    let model = Arc::new(ScriptedModel::new());
    let final_message = if output_type {
        assistant_message(r#"{"a":"b"}"#)
    } else {
        assistant_message("done")
    };
    let (agent_3, agent_1) = three_agent_graph(&model, hooks.clone(), output_type);

    let first = if output_type {
        assistant_message(r#"{"a":"b"}"#)
    } else {
        assistant_message("user_message")
    };
    model.enqueue(ScriptedStep::output(vec![first])).unwrap();
    drain(&agent_3, "user_message").await;
    assert_eq!(hooks.events(), expect(&[("on_start", 1), ("on_end", 1)]));
    hooks.reset();

    model
        .extend(vec![
            ScriptedStep::output(vec![function_call("some_function", r#"{"a":"b"}"#)]),
            ScriptedStep::output(vec![assistant_message("a_message"), handoff_call(&agent_1)]),
            ScriptedStep::output(text("done")),
        ])
        .unwrap();
    drain(&agent_3, "user_message").await;
    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 1),
            ("on_tool_start", 1),
            ("on_tool_end", 1),
            ("on_handoff", 1)
        ])
    );
    hooks.reset();

    model
        .extend(vec![
            ScriptedStep::output(vec![function_call("some_function", r#"{"a":"b"}"#)]),
            ScriptedStep::output(vec![
                assistant_message("a_message"),
                function_call("some_function", r#"{"a":"b"}"#),
                handoff_call(&agent_1),
            ]),
            ScriptedStep::output(vec![assistant_message("a_message"), handoff_call(&agent_3)]),
            ScriptedStep::output(vec![final_message]),
        ])
        .unwrap();
    drain(&agent_3, "user_message").await;
    assert_eq!(
        hooks.events(),
        expect(&[
            ("on_start", 2),
            ("on_tool_start", 2),
            ("on_tool_end", 2),
            ("on_handoff", 1),
            ("on_end", 1),
        ])
    );
}

async fn drain(agent: &Arc<Agent<()>>, input: &str) {
    let mut result = Runner::run_streamed_with_options(agent, input, RunOptions::new(()));
    let mut events = std::pin::pin!(result.stream_events());
    while let Some(event) = events.next().await {
        event.unwrap();
    }
}

#[tokio::test]
async fn streamed_agent_hooks_follow_handoffs_and_tools() {
    disable_tracing();
    run_streamed_scenario(false).await;
}

#[tokio::test]
async fn structured_output_non_streamed_agent_hooks() {
    disable_tracing();
    run_non_streamed_scenario(true).await;
}

#[tokio::test]
async fn structured_output_streamed_agent_hooks() {
    disable_tracing();
    run_streamed_scenario(true).await;
}

#[tokio::test]
async fn base_agent_hooks_with_default_methods_do_not_crash() {
    disable_tracing();

    struct EmptyAgentHooks;
    #[async_trait]
    impl AgentHooks<()> for EmptyAgentHooks {}

    let model = Arc::new(ScriptedModel::new());
    let agent_1 = Arc::new(Agent::<()>::new("test_1").with_model_instance(model.clone()));
    let agent_2 = Arc::new(Agent::<()>::new("test_2").with_model_instance(model.clone()));
    let agent_3 = Arc::new(
        Agent::<()>::new("test_3")
            .with_model_instance(model.clone())
            .with_handoff(agent_1.clone())
            .with_handoff(agent_2.clone())
            .with_tool(some_function())
            .with_hooks(Arc::new(EmptyAgentHooks))
            .with_output_type::<Foo>(),
    );

    model
        .enqueue(ScriptedStep::output(vec![assistant_message(
            r#"{"a":"b"}"#,
        )]))
        .unwrap();
    drain(&agent_3, "user_message").await;
    model
        .extend(vec![
            ScriptedStep::output(vec![function_call("some_function", r#"{"a":"b"}"#)]),
            ScriptedStep::output(vec![assistant_message("a_message"), handoff_call(&agent_1)]),
            ScriptedStep::output(text("done")),
        ])
        .unwrap();
    drain(&agent_3, "user_message").await;
}

#[tokio::test]
async fn agent_hooks_receive_turn_input_for_a_string_input() {
    disable_tracing();
    let hooks = Arc::new(CountingAgentHooks::default());
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(text("response")))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_hooks(hooks.clone());

    Runner::run(&agent, "hello world").await.unwrap();

    let captured = hooks.turn_inputs.lock().unwrap().clone();
    assert_eq!(captured.len(), 1);
    let turn_input = &captured[0];
    assert_eq!(turn_input.len(), 1);
    assert_eq!(turn_input[0].0["content"], "hello world");
    assert_eq!(turn_input[0].0["role"], "user");
}

#[tokio::test]
async fn agent_hooks_receive_turn_input_for_a_list_input() {
    disable_tracing();
    let hooks = Arc::new(CountingAgentHooks::default());
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(text("response")))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_hooks(hooks.clone());
    let input = vec![
        InputItem(json!({"role": "user", "content": "first message"})),
        InputItem(json!({"role": "user", "content": "second message"})),
    ];

    Runner::run(&agent, input).await.unwrap();

    let captured = hooks.turn_inputs.lock().unwrap().clone();
    assert_eq!(captured.len(), 1);
    let turn_input = &captured[0];
    assert_eq!(turn_input.len(), 2);
    assert_eq!(turn_input[0].0["content"], "first message");
    assert_eq!(turn_input[1].0["content"], "second message");
}

#[tokio::test]
async fn agent_hooks_receive_turn_input_in_streamed_mode() {
    disable_tracing();
    let hooks = Arc::new(CountingAgentHooks::default());
    let model = Arc::new(ScriptedModel::new());
    model
        .enqueue(ScriptedStep::output(text("response")))
        .unwrap();
    let agent = Agent::<()>::new("test")
        .with_model_instance(model)
        .with_hooks(hooks.clone());

    drain(&Arc::new(agent), "streamed input").await;

    let captured = hooks.turn_inputs.lock().unwrap().clone();
    assert_eq!(captured.len(), 1);
    let turn_input = &captured[0];
    assert_eq!(turn_input.len(), 1);
    assert_eq!(turn_input[0].0["content"], "streamed input");
}
