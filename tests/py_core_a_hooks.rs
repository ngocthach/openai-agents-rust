//! Ports of `test_run_hooks.py`, `test_global_hooks.py` and `test_run_error_details.py`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::ModelResponse;
use openai_agents_sdk::lifecycle::{AgentHooks, RunHooks};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, InputItem, RunContext, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Args {
    #[serde(default)]
    a: String,
}

fn some_function() -> FunctionTool<()> {
    FunctionTool::new("some_function", "f", |_ctx, _a: Args| async move {
        Ok("result".to_string())
    })
}

#[derive(Default)]
struct CountingRunHooks {
    events: Mutex<BTreeMap<String, usize>>,
    tool_context_ids: Mutex<Vec<String>>,
    tool_results: Mutex<Vec<ToolOutput>>,
    turn_inputs: Mutex<Vec<Vec<InputItem>>>,
}

impl CountingRunHooks {
    fn bump(&self, name: &str) {
        *self
            .events
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert(0) += 1;
    }
    fn count(&self, name: &str) -> usize {
        self.events.lock().unwrap().get(name).copied().unwrap_or(0)
    }
    fn snapshot(&self) -> BTreeMap<String, usize> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait]
impl RunHooks<()> for CountingRunHooks {
    async fn on_llm_start(
        &self,
        _ctx: &RunContext<()>,
        _agent: &Agent<()>,
        _system_prompt: Option<&str>,
        _input_items: &[InputItem],
    ) {
        self.bump("on_llm_start");
    }
    async fn on_llm_end(&self, _ctx: &RunContext<()>, _agent: &Agent<()>, _r: &ModelResponse) {
        self.bump("on_llm_end");
    }
    async fn on_agent_start(&self, ctx: &RunContext<()>, _agent: &Agent<()>) {
        self.bump("on_agent_start");
        self.turn_inputs.lock().unwrap().push(ctx.turn_input());
    }
    async fn on_agent_end(&self, _ctx: &RunContext<()>, _agent: &Agent<()>, _output: &Value) {
        self.bump("on_agent_end");
    }
    async fn on_handoff(&self, _ctx: &RunContext<()>, _from: &Agent<()>, _to: &Agent<()>) {
        self.bump("on_handoff");
    }
    async fn on_tool_start(&self, _ctx: &ToolContext<()>, _agent: &Agent<()>, _tool: &Tool<()>) {
        self.bump("on_tool_start");
    }
    async fn on_tool_end(
        &self,
        ctx: &ToolContext<()>,
        _agent: &Agent<()>,
        _tool: &Tool<()>,
        result: &ToolOutput,
    ) {
        self.bump("on_tool_end");
        self.tool_context_ids
            .lock()
            .unwrap()
            .push(ctx.tool_call_id.clone());
        self.tool_results.lock().unwrap().push(result.clone());
    }
}

#[derive(Default)]
struct CountingAgentHooks {
    events: Mutex<BTreeMap<String, usize>>,
}

impl CountingAgentHooks {
    fn bump(&self, name: &str) {
        *self
            .events
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert(0) += 1;
    }
    fn snapshot(&self) -> BTreeMap<String, usize> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait]
impl AgentHooks<()> for CountingAgentHooks {
    async fn on_start(&self, _ctx: &RunContext<()>, _agent: &Agent<()>) {
        self.bump("on_start");
    }
    async fn on_end(&self, _ctx: &RunContext<()>, _agent: &Agent<()>, _output: &Value) {
        self.bump("on_end");
    }
    async fn on_llm_start(
        &self,
        _ctx: &RunContext<()>,
        _agent: &Agent<()>,
        _system_prompt: Option<&str>,
        _input_items: &[InputItem],
    ) {
        self.bump("on_llm_start");
    }
    async fn on_llm_end(&self, _ctx: &RunContext<()>, _agent: &Agent<()>, _r: &ModelResponse) {
        self.bump("on_llm_end");
    }
}

fn expect(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

#[tokio::test]
async fn async_run_hooks_with_llm() {
    let model = setup();
    model.add_turn(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("A")
        .with_tool(some_function())
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    Runner::run_with_options(&agent, "hello", RunOptions::new(()).hooks(hooks.clone()))
        .await
        .unwrap();
    assert_eq!(
        hooks.snapshot(),
        expect(&[
            ("on_agent_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_agent_end", 1),
        ])
    );
}

#[test]
fn sync_run_hook_with_llm() {
    let model = setup();
    model.add_turn(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("A")
        .with_tool(some_function())
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    Runner::run_sync_with_options(&agent, "hello", RunOptions::new(()).hooks(hooks.clone()))
        .unwrap();
    assert_eq!(
        hooks.snapshot(),
        expect(&[
            ("on_agent_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_agent_end", 1),
        ])
    );
}

#[tokio::test]
async fn streamed_run_hooks_with_llm() {
    let model = setup();
    model.add_turn(vec![assistant_message("hello")]);
    let agent = Agent::<()>::new("A")
        .with_tool(some_function())
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    let mut stream = Runner::run_streamed_with_options(
        &agent,
        "hello",
        RunOptions::new(()).hooks(hooks.clone()),
    );
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    assert_eq!(
        hooks.snapshot(),
        expect(&[
            ("on_agent_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_agent_end", 1),
        ])
    );
}

#[tokio::test]
async fn async_run_hooks_with_agent_hooks_with_llm() {
    let model = setup();
    model.add_turn(vec![assistant_message("hello")]);
    let agent_hooks = Arc::new(CountingAgentHooks::default());
    let agent = Agent::<()>::new("A")
        .with_tool(some_function())
        .with_hooks(agent_hooks.clone())
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    Runner::run_with_options(&agent, "hello", RunOptions::new(()).hooks(hooks.clone()))
        .await
        .unwrap();
    assert_eq!(
        hooks.snapshot(),
        expect(&[
            ("on_agent_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_agent_end", 1),
        ])
    );
    assert_eq!(
        agent_hooks.snapshot(),
        expect(&[
            ("on_start", 1),
            ("on_llm_start", 1),
            ("on_llm_end", 1),
            ("on_end", 1),
        ])
    );
}

#[tokio::test]
async fn run_hooks_llm_error_non_streaming() {
    let model = setup();
    model.add_error("boom");
    let agent = Agent::<()>::new("A")
        .with_tool(some_function())
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    let err = Runner::run_with_options(&agent, "hello", RunOptions::new(()).hooks(hooks.clone()))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("boom"), "{err}");
    assert_eq!(hooks.count("on_agent_start"), 1);
    assert_eq!(hooks.count("on_llm_start"), 1);
    assert_eq!(hooks.count("on_llm_end"), 0);
    assert_eq!(hooks.count("on_agent_end"), 0);
}

#[tokio::test]
async fn streamed_run_hooks_llm_error() {
    let model = setup();
    model.add_error("stream blew up");
    let agent = Agent::<()>::new("A")
        .with_tool(some_function())
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    let mut stream = Runner::run_streamed_with_options(
        &agent,
        "hello",
        RunOptions::new(()).hooks(hooks.clone()),
    );
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    let error = error.expect("stream surfaces the error");
    assert!(error.to_string().contains("stream blew up"), "{error}");
    assert_eq!(hooks.count("on_agent_start"), 1);
    assert_eq!(hooks.count("on_llm_start"), 1);
    assert_eq!(hooks.count("on_llm_end"), 0);
    assert_eq!(hooks.count("on_agent_end"), 0);
}

#[tokio::test]
async fn run_hooks_receives_turn_input_string() {
    let model = setup();
    model.add_turn(vec![assistant_message("response")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    Runner::run_with_options(
        &agent,
        "hello world",
        RunOptions::new(()).hooks(hooks.clone()),
    )
    .await
    .unwrap();
    let turn_inputs = hooks.turn_inputs.lock().unwrap().clone();
    assert_eq!(turn_inputs.len(), 1);
    assert_eq!(turn_inputs[0].len(), 1);
    assert_eq!(
        turn_inputs[0][0].text_content().as_deref(),
        Some("hello world")
    );
}

#[tokio::test]
async fn run_hooks_receives_turn_input_list() {
    let model = setup();
    model.add_turn(vec![assistant_message("response")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    let input = vec![
        InputItem::user("first message"),
        InputItem::user("second message"),
    ];
    Runner::run_with_options(&agent, input, RunOptions::new(()).hooks(hooks.clone()))
        .await
        .unwrap();
    let turn_inputs = hooks.turn_inputs.lock().unwrap().clone();
    assert_eq!(turn_inputs.len(), 1);
    assert_eq!(turn_inputs[0].len(), 2);
    assert_eq!(
        turn_inputs[0][0].text_content().as_deref(),
        Some("first message")
    );
    assert_eq!(
        turn_inputs[0][1].text_content().as_deref(),
        Some("second message")
    );
}

#[tokio::test]
async fn run_hooks_receives_turn_input_streamed() {
    let model = setup();
    model.add_turn(vec![assistant_message("response")]);
    let agent = Agent::<()>::new("test").with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    let mut stream = Runner::run_streamed_with_options(
        &agent,
        "streamed input",
        RunOptions::new(()).hooks(hooks.clone()),
    );
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }
    let turn_inputs = hooks.turn_inputs.lock().unwrap().clone();
    assert_eq!(turn_inputs.len(), 1);
    assert_eq!(
        turn_inputs[0][0].text_content().as_deref(),
        Some("streamed input")
    );
}

#[tokio::test]
async fn run_hooks_count_tool_and_handoff_invocations() {
    let model = setup();
    let agent_1 = Arc::new(Agent::<()>::new("test_1").with_model_instance(model.clone()));
    let agent_2 = Agent::<()>::new("test_2")
        .with_handoff(handoff(agent_1.clone()))
        .with_tool(some_function())
        .with_model_instance(model.clone());

    model.add_turn(vec![function_call("some_function", r#"{"a":"b"}"#)]);
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call("transfer_to_test_1", "{}"),
    ]);
    model.add_turn(vec![assistant_message("done")]);

    let hooks = Arc::new(CountingRunHooks::default());
    Runner::run_with_options(
        &agent_2,
        "user_message",
        RunOptions::new(()).hooks(hooks.clone()),
    )
    .await
    .unwrap();

    assert_eq!(hooks.count("on_tool_start"), 1);
    assert_eq!(hooks.count("on_tool_end"), 1);
    assert_eq!(hooks.count("on_handoff"), 1);
    assert_eq!(hooks.count("on_agent_start"), 2);
    assert_eq!(hooks.count("on_agent_end"), 1);
    assert_eq!(hooks.tool_context_ids.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn streamed_run_hooks_count_tool_and_handoff_invocations() {
    let model = setup();
    let agent_1 = Arc::new(Agent::<()>::new("test_1").with_model_instance(model.clone()));
    let agent_2 = Agent::<()>::new("test_2")
        .with_handoff(handoff(agent_1.clone()))
        .with_tool(some_function())
        .with_model_instance(model.clone());

    model.add_turn(vec![
        function_call_with_id("some_function", r#"{"a":"b"}"#, "call_1"),
        function_call_with_id("some_function", r#"{"a":"b"}"#, "call_2"),
    ]);
    model.add_turn(vec![
        assistant_message("a_message"),
        function_call_with_id("transfer_to_test_1", "{}", "handoff_1"),
    ]);
    model.add_turn(vec![assistant_message("done")]);

    let hooks = Arc::new(CountingRunHooks::default());
    let mut stream = Runner::run_streamed_with_options(
        &agent_2,
        "user_message",
        RunOptions::new(()).hooks(hooks.clone()),
    );
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }

    assert_eq!(hooks.count("on_tool_start"), 2);
    assert_eq!(hooks.count("on_tool_end"), 2);
    assert_eq!(hooks.count("on_handoff"), 1);
    assert_eq!(hooks.count("on_agent_start"), 2);
    assert_eq!(hooks.count("on_agent_end"), 1);
    let ids = hooks.tool_context_ids.lock().unwrap().clone();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"call_1".to_string()));
    assert!(ids.contains(&"call_2".to_string()));
}

#[tokio::test]
async fn tool_end_hooks_receive_the_raw_tool_result() {
    let model = setup();
    model.add_turn(vec![function_call("get_metadata", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let tool = FunctionTool::new("get_metadata", "f", |_ctx, _a: Args| async move {
        Ok(serde_json::json!({"status": "ok", "count": 1}))
    });
    let agent = Agent::<()>::new("test")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingRunHooks::default());
    Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(()).hooks(hooks.clone()),
    )
    .await
    .unwrap();
    let results = hooks.tool_results.lock().unwrap().clone();
    assert_eq!(results.len(), 1);
    match &results[0] {
        ToolOutput::Json(value) => {
            assert_eq!(value, &serde_json::json!({"status": "ok", "count": 1}))
        }
        other => panic!("unexpected tool output: {other:?}"),
    }
}

// --------------------------------------------------------- run error details

#[tokio::test]
async fn run_error_includes_data() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("1"),
        function_call("foo", r#"{"a":"b"}"#),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(FunctionTool::new("foo", "f", |_c, _a: Args| async move {
            Ok("res".to_string())
        }))
        .with_model_instance(model.clone());

    let err = Runner::run_with_options(&agent, "hello", RunOptions::new(()).max_turns(Some(1)))
        .await
        .unwrap_err();
    let AgentsError::MaxTurnsExceeded { run_data, .. } = &err else {
        panic!("unexpected error: {err:?}");
    };
    let data = run_data.as_ref().expect("run data is attached");
    assert_eq!(data.last_agent, "test");
    assert_eq!(data.raw_responses.len(), 1);
    assert!(!data.new_items.is_empty());
}

#[tokio::test]
async fn streamed_run_error_includes_data() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("1"),
        function_call("foo", r#"{"a":"b"}"#),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test")
        .with_tool(FunctionTool::new("foo", "f", |_c, _a: Args| async move {
            Ok("res".to_string())
        }))
        .with_model_instance(model.clone());

    let mut stream =
        Runner::run_streamed_with_options(&agent, "hello", RunOptions::new(()).max_turns(Some(1)));
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                error = Some(e);
            }
        }
    }
    let err = error.expect("stream surfaces the error");
    let AgentsError::MaxTurnsExceeded { run_data, .. } = &err else {
        panic!("unexpected error: {err:?}");
    };
    let data = run_data.as_ref().expect("run data is attached");
    assert_eq!(data.last_agent, "test");
    assert_eq!(data.raw_responses.len(), 1);
    assert!(!data.new_items.is_empty());
}
