//! Ports of the action, hook and safety-check cases of `test_computer_action.py`.
//!
//! Python calls `ComputerAction._execute_action_and_capture` and `ComputerAction.execute`
//! directly; these tests reach the same code through a `computer_call` model output. The
//! function-span cases live in `py_core5_leftover2_computer_spans.rs` (tracing is
//! process-global).
//!
//! Python also covers duck-typed drivers: a sync driver vs an async driver (one `async_trait`
//! here, so both parameter sets run through the same `Computer`), and drivers whose methods do not
//! accept the `keys` keyword (legacy signature, non-introspectable callable, `**kwargs`). The
//! Rust `Computer` trait always takes `keys: Option<&[String]>`, so those three cases cannot
//! exist.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::computer::Computer;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::lifecycle::{AgentHooks, RunHooks};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::testing::{FakeModel, ModelStep, assistant_message};
use openai_agents_sdk::tool::{ComputerTool, Tool, ToolOutput};
use openai_agents_sdk::{Agent, Result, Runner, ToolContext};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

type Calls = Arc<Mutex<Vec<Value>>>;

/// Python's `LoggingComputer` / `LoggingAsyncComputer`: records `[name, args..., keys?]`.
struct LoggingComputer {
    calls: Calls,
    screenshot: String,
    /// Called at every `click`, to observe what had been logged by then.
    on_click: Option<Box<dyn Fn() + Send + Sync>>,
}

impl LoggingComputer {
    fn new(screenshot: &str) -> Arc<Self> {
        Arc::new(LoggingComputer {
            calls: Arc::default(),
            screenshot: screenshot.to_string(),
            on_click: None,
        })
    }

    fn log(&self, name: &str, mut args: Vec<Value>, keys: Option<&[String]>) {
        if let Some(keys) = keys {
            args.push(json!(keys));
        }
        let mut entry = vec![json!(name)];
        entry.extend(args);
        self.calls.lock().unwrap().push(Value::Array(entry));
    }

    fn calls(&self) -> Vec<Value> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Computer for LoggingComputer {
    fn environment(&self) -> Option<openai_agents_sdk::computer::Environment> {
        Some(openai_agents_sdk::computer::Environment::Mac)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((800, 600))
    }
    async fn screenshot(&self) -> Result<String> {
        self.log("screenshot", vec![], None);
        Ok(self.screenshot.clone())
    }
    async fn click(&self, x: i64, y: i64, button: &str, keys: Option<&[String]>) -> Result<()> {
        if let Some(observe) = &self.on_click {
            observe();
        }
        self.log("click", vec![json!(x), json!(y), json!(button)], keys);
        Ok(())
    }
    async fn double_click(&self, x: i64, y: i64, keys: Option<&[String]>) -> Result<()> {
        self.log("double_click", vec![json!(x), json!(y)], keys);
        Ok(())
    }
    async fn scroll(
        &self,
        x: i64,
        y: i64,
        scroll_x: i64,
        scroll_y: i64,
        keys: Option<&[String]>,
    ) -> Result<()> {
        self.log(
            "scroll",
            vec![json!(x), json!(y), json!(scroll_x), json!(scroll_y)],
            keys,
        );
        Ok(())
    }
    async fn type_text(&self, text: &str) -> Result<()> {
        self.log("type", vec![json!(text)], None);
        Ok(())
    }
    async fn wait(&self) -> Result<()> {
        self.log("wait", vec![], None);
        Ok(())
    }
    async fn move_mouse(&self, x: i64, y: i64, keys: Option<&[String]>) -> Result<()> {
        self.log("move", vec![json!(x), json!(y)], keys);
        Ok(())
    }
    async fn keypress(&self, keys: &[String]) -> Result<()> {
        self.log("keypress", vec![json!(keys)], None);
        Ok(())
    }
    async fn drag(&self, path: &[(i64, i64)], keys: Option<&[String]>) -> Result<()> {
        let path: Vec<Value> = path.iter().map(|(x, y)| json!([x, y])).collect();
        self.log("drag", vec![Value::Array(path)], keys);
        Ok(())
    }
}

fn computer_call(call_id: &str, body: Value, checks: Value) -> Value {
    let mut call = json!({
        "type": "computer_call",
        "id": call_id,
        "call_id": call_id,
        "pending_safety_checks": checks,
        "status": "completed",
    });
    for (key, value) in body.as_object().unwrap() {
        call[key] = value.clone();
    }
    call
}

fn single(call_id: &str, action: Value) -> Value {
    computer_call(call_id, json!({"action": action}), json!([]))
}

fn batched(call_id: &str, actions: Value) -> Value {
    computer_call(call_id, json!({"actions": actions}), json!([]))
}

struct Ran {
    output: Value,
}

/// Run `call` against `computer` and return the computer-call output item.
async fn run_call(computer: &Arc<LoggingComputer>, call: Value) -> Ran {
    let model = setup();
    model.add_turn(vec![OutputItem::from_value(call)]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("test_agent")
        .with_tool(ComputerTool::<()>::from_arc(computer.clone()))
        .with_model_instance(model);
    let result = Runner::run(&agent, "go").await.unwrap();
    let output = result
        .new_items
        .iter()
        .find_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("a computer_call_output item");
    Ran { output }
}

fn image_url(ran: &Ran) -> &str {
    ran.output["output"]["image_url"].as_str().unwrap()
}

fn action_cases(first: bool) -> Vec<(Value, Value)> {
    // (action, expected call). The second set mirrors the async-driver parameters.
    if first {
        vec![
            (
                json!({"type": "click", "x": 10, "y": 21, "button": "left"}),
                json!(["click", 10, 21, "left"]),
            ),
            (
                json!({"type": "double_click", "x": 42, "y": 47}),
                json!(["double_click", 42, 47]),
            ),
            (
                json!({"type": "drag", "path": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}),
                json!(["drag", [[1, 2], [3, 4]]]),
            ),
            (
                json!({"type": "keypress", "keys": ["a", "b"]}),
                json!(["keypress", ["a", "b"]]),
            ),
            (
                json!({"type": "move", "x": 100, "y": 200}),
                json!(["move", 100, 200]),
            ),
            (json!({"type": "screenshot"}), json!(["screenshot"])),
            (
                json!({"type": "scroll", "x": 1, "y": 2, "scroll_x": 3, "scroll_y": 4}),
                json!(["scroll", 1, 2, 3, 4]),
            ),
            (
                json!({"type": "type", "text": "hello"}),
                json!(["type", "hello"]),
            ),
            (json!({"type": "wait"}), json!(["wait"])),
        ]
    } else {
        vec![
            (
                json!({"type": "click", "x": 2, "y": 3, "button": "right"}),
                json!(["click", 2, 3, "right"]),
            ),
            (
                json!({"type": "double_click", "x": 12, "y": 13}),
                json!(["double_click", 12, 13]),
            ),
            (
                json!({"type": "drag", "path": [{"x": 5, "y": 6}, {"x": 6, "y": 7}]}),
                json!(["drag", [[5, 6], [6, 7]]]),
            ),
            (
                json!({"type": "keypress", "keys": ["ctrl", "c"]}),
                json!(["keypress", ["ctrl", "c"]]),
            ),
            (
                json!({"type": "move", "x": 8, "y": 9}),
                json!(["move", 8, 9]),
            ),
            (json!({"type": "screenshot"}), json!(["screenshot"])),
            (
                json!({"type": "scroll", "x": 9, "y": 8, "scroll_x": 7, "scroll_y": 6}),
                json!(["scroll", 9, 8, 7, 6]),
            ),
            (
                json!({"type": "type", "text": "world"}),
                json!(["type", "world"]),
            ),
            (json!({"type": "wait"}), json!(["wait"])),
        ]
    }
}

async fn check_actions(first: bool, screenshot: &str, call_id: &str) {
    for (action, expected) in action_cases(first) {
        let computer = LoggingComputer::new(screenshot);
        let ran = run_call(&computer, single(call_id, action.clone())).await;
        if action["type"] == "screenshot" {
            assert_eq!(computer.calls(), vec![json!(["screenshot"])], "{action}");
        } else {
            assert_eq!(
                computer.calls(),
                vec![expected, json!(["screenshot"])],
                "{action}"
            );
        }
        assert_eq!(
            image_url(&ran),
            format!("data:image/png;base64,{screenshot}")
        );
    }
}

/// Python: `test_get_screenshot_sync_executes_action_and_takes_screenshot` (9 actions).
#[tokio::test]
async fn each_action_runs_then_takes_a_screenshot() {
    check_actions(true, "synthetic", "c1").await;
}

/// Python: `test_get_screenshot_async_executes_action_and_takes_screenshot` (9 actions).
#[tokio::test]
async fn each_action_runs_then_takes_a_screenshot_with_a_second_parameter_set() {
    let computer = LoggingComputer::new("async_return");
    assert_eq!(computer.environment().map(|e| e.as_str()), Some("mac"));
    assert_eq!(computer.dimensions(), Some((800, 600)));
    check_actions(false, "async_return", "c2").await;
}

/// Python: `test_get_screenshot_executes_batched_actions_in_order`.
#[tokio::test]
async fn batched_actions_run_in_order() {
    let computer = LoggingComputer::new("batched");
    let ran = run_call(
        &computer,
        batched(
            "c3",
            json!([
                {"type": "click", "x": 11, "y": 12, "button": "left"},
                {"type": "type", "text": "hello"},
            ]),
        ),
    )
    .await;
    assert_eq!(
        computer.calls(),
        vec![
            json!(["click", 11, 12, "left"]),
            json!(["type", "hello"]),
            json!(["screenshot"]),
        ]
    );
    assert_eq!(image_url(&ran), "data:image/png;base64,batched");
}

/// Python: `test_get_screenshot_reuses_terminal_batched_screenshot`.
#[tokio::test]
async fn a_terminal_batched_screenshot_is_reused() {
    let computer = LoggingComputer::new("captured");
    let ran = run_call(&computer, batched("c4", json!([{"type": "screenshot"}]))).await;
    assert_eq!(computer.calls(), vec![json!(["screenshot"])]);
    assert_eq!(image_url(&ran), "data:image/png;base64,captured");
}

/// Python: `test_get_screenshot_preserves_modifier_keys_for_sync_driver`.
#[tokio::test]
async fn modifier_keys_reach_a_click() {
    let computer = LoggingComputer::new("with_keys");
    let ran = run_call(
        &computer,
        single(
            "c5",
            json!({"type": "click", "x": 4, "y": 8, "button": "left", "keys": ["shift", "ctrl"]}),
        ),
    )
    .await;
    assert_eq!(
        computer.calls(),
        vec![
            json!(["click", 4, 8, "left", ["shift", "ctrl"]]),
            json!(["screenshot"]),
        ]
    );
    assert_eq!(image_url(&ran), "data:image/png;base64,with_keys");
}

/// Python: `test_get_screenshot_preserves_modifier_keys_for_async_driver`.
#[tokio::test]
async fn modifier_keys_reach_a_scroll() {
    let computer = LoggingComputer::new("async_keys");
    let ran = run_call(
        &computer,
        single(
            "c6",
            json!({"type": "scroll", "x": 7, "y": 9, "scroll_x": 3, "scroll_y": -2, "keys": ["alt"]}),
        ),
    )
    .await;
    assert_eq!(
        computer.calls(),
        vec![
            json!(["scroll", 7, 9, 3, -2, ["alt"]]),
            json!(["screenshot"])
        ]
    );
    assert_eq!(image_url(&ran), "data:image/png;base64,async_keys");
}

/// Python: `test_get_screenshot_preserves_modifier_keys_for_batched_actions`.
#[tokio::test]
async fn modifier_keys_reach_a_batched_click() {
    let computer = LoggingComputer::new("batched_keys");
    let ran = run_call(
        &computer,
        batched(
            "c10",
            json!([{"type": "click", "x": 11, "y": 12, "button": "left", "keys": ["ctrl"]}]),
        ),
    )
    .await;
    assert_eq!(
        computer.calls(),
        vec![
            json!(["click", 11, 12, "left", ["ctrl"]]),
            json!(["screenshot"])
        ]
    );
    assert_eq!(image_url(&ran), "data:image/png;base64,batched_keys");
}

#[derive(Default)]
struct LoggingHooks {
    started: Mutex<Vec<String>>,
    ended: Mutex<Vec<(String, ToolOutput)>>,
}

#[async_trait]
impl RunHooks<()> for LoggingHooks {
    async fn on_tool_start(&self, _ctx: &ToolContext<()>, agent: &Agent<()>, tool: &Tool<()>) {
        self.started
            .lock()
            .unwrap()
            .push(format!("{}:{}", agent.name, tool.name()));
    }
    async fn on_tool_end(
        &self,
        _ctx: &ToolContext<()>,
        agent: &Agent<()>,
        tool: &Tool<()>,
        result: &ToolOutput,
    ) {
        self.ended
            .lock()
            .unwrap()
            .push((format!("{}:{}", agent.name, tool.name()), result.clone()));
    }
}

#[async_trait]
impl AgentHooks<()> for LoggingHooks {
    async fn on_tool_start(&self, _ctx: &ToolContext<()>, agent: &Agent<()>, tool: &Tool<()>) {
        RunHooks::on_tool_start(self, _ctx, agent, tool).await;
    }
    async fn on_tool_end(
        &self,
        _ctx: &ToolContext<()>,
        agent: &Agent<()>,
        tool: &Tool<()>,
        result: &ToolOutput,
    ) {
        RunHooks::on_tool_end(self, _ctx, agent, tool, result).await;
    }
}

/// Python: `test_execute_invokes_hooks_and_returns_tool_call_output`.
#[tokio::test]
async fn execute_invokes_run_and_agent_hooks_and_returns_the_output_item() {
    let computer = LoggingComputer::new("xyz");
    let model = setup();
    model.add_turn(vec![OutputItem::from_value(single(
        "tool123",
        json!({"type": "click", "x": 1, "y": 2, "button": "left"}),
    ))]);
    model.add_turn(vec![assistant_message("done")]);
    let agent_hooks = Arc::new(LoggingHooks::default());
    let run_hooks = Arc::new(LoggingHooks::default());
    let agent = Agent::<()>::new("test_agent")
        .with_tool(ComputerTool::<()>::from_arc(computer.clone()))
        .with_hooks(agent_hooks.clone())
        .with_model_instance(model);
    let result = Runner::run_with_options(
        &agent,
        "go",
        RunOptions::new(())
            .hooks(run_hooks.clone())
            .run_config(RunConfig::default()),
    )
    .await
    .unwrap();

    let tool_name = ComputerTool::<()>::from_arc(computer.clone())
        .name()
        .to_string();
    for hooks in [&run_hooks, &agent_hooks] {
        assert_eq!(
            *hooks.started.lock().unwrap(),
            vec![format!("test_agent:{tool_name}")]
        );
        let ended = hooks.ended.lock().unwrap();
        assert_eq!(ended.len(), 1);
        assert_eq!(ended[0].0, format!("test_agent:{tool_name}"));
        // The result passed to `on_tool_end` is the raw screenshot string.
        assert_eq!(ended[0].1, ToolOutput::Text("xyz".to_string()));
    }
    assert_eq!(
        computer.calls(),
        vec![json!(["click", 1, 2, "left"]), json!(["screenshot"])]
    );
    let item = result
        .new_items
        .iter()
        .find(|item| matches!(item.kind, RunItemKind::ToolCallOutput { .. }))
        .unwrap();
    assert_eq!(item.agent, "test_agent");
    let RunItemKind::ToolCallOutput {
        raw_item, output, ..
    } = &item.kind
    else {
        unreachable!()
    };
    assert_eq!(output, &json!("data:image/png;base64,xyz"));
    assert_eq!(raw_item["type"], "computer_call_output");
    assert_eq!(raw_item["output"]["type"], "computer_screenshot");
    assert!(
        raw_item["output"]["image_url"]
            .as_str()
            .unwrap()
            .ends_with("xyz")
    );
}

/// Python: `test_pending_safety_check_acknowledged`.
#[tokio::test]
async fn a_pending_safety_check_is_acknowledged_through_the_callback() {
    let computer = LoggingComputer::new("img");
    let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
    let record = seen.clone();
    let tool = ComputerTool::<()>::from_arc(computer.clone()).with_on_safety_check(move |data| {
        let record = record.clone();
        async move {
            record.lock().unwrap().push(data.safety_check.clone());
            true
        }
    });
    let model = setup();
    model.add_turn(vec![OutputItem::from_value(computer_call(
        "t1",
        json!({"action": {"type": "click", "x": 1, "y": 1, "button": "left"}}),
        json!([{"id": "sc", "code": "c", "message": "m"}]),
    ))]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("a")
        .with_tool(tool)
        .with_model_instance(model);
    let result = Runner::run(&agent, "go").await.unwrap();
    let raw = result
        .new_items
        .iter()
        .find_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        raw["acknowledged_safety_checks"],
        json!([{"id": "sc", "code": "c", "message": "m"}])
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["id"], "sc");
}

// ----------------------------------------------- unhandled safety-check warnings

async fn run_maybe_streamed(
    agent: Agent<()>,
    input: &str,
    streamed: bool,
    options: RunOptions<()>,
) -> Result<openai_agents_sdk::result::RunResult<()>> {
    if !streamed {
        return Runner::run_with_options(&agent, input, options).await;
    }
    let mut run = Runner::run_streamed_with_options(agent, input, options);
    {
        let events = run.stream_events();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            event?;
        }
    }
    Ok(run.into_result().expect("a finished stream has a result"))
}

static RECORDS: Mutex<Vec<(log::Level, String, String)>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push((
            record.level(),
            record.target().to_string(),
            format!("{}", record.args()),
        ));
    }
    fn flush(&self) {}
}

fn install_logger() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

fn safety_warnings() -> Vec<(log::Level, String, String)> {
    RECORDS
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, target, message)| {
            target == "openai.agents" && message.contains("no on_safety_check")
        })
        .cloned()
        .collect()
}

/// Python: `test_runner_warns_for_unhandled_computer_safety_checks` (blocking and streamed, four
/// handler/check combinations).
#[tokio::test]
async fn the_runner_warns_for_unhandled_computer_safety_checks() {
    static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _serial = SERIAL.lock().await;
    install_logger();

    for stream in [false, true] {
        for (handler, has_checks) in [
            ("missing", true),
            ("missing", false),
            ("accept", true),
            ("reject", true),
        ] {
            RECORDS.lock().unwrap().clear();
            let should_warn = has_checks && handler == "missing";
            let warned_at_click = Arc::new(Mutex::new(None::<bool>));
            let observed = warned_at_click.clone();
            let computer = Arc::new(LoggingComputer {
                calls: Arc::default(),
                screenshot: "synthetic screenshot".to_string(),
                on_click: Some(Box::new(move || {
                    *observed.lock().unwrap() = Some(!safety_warnings().is_empty());
                })),
            });
            let checks: Vec<Value> = [
                "malicious_instructions",
                "irrelevant_domain",
                "sensitive_domain",
                "future_check",
            ]
            .iter()
            .enumerate()
            .map(|(index, code)| {
                json!({"id": format!("check-{index}"), "code": code,
                       "message": "synthetic private message"})
            })
            .collect();
            let call = computer_call(
                "computer-call",
                json!({"action": {"type": "click", "x": 1, "y": 2, "button": "left"}}),
                if has_checks {
                    Value::Array(checks.clone())
                } else {
                    json!([])
                },
            );
            let model = setup();
            model.enqueue(ModelStep::Output(vec![OutputItem::from_value(call)]));
            model.add_turn(vec![assistant_message("done")]);
            let mut tool = ComputerTool::<()>::from_arc(computer.clone());
            if handler != "missing" {
                let accept = handler == "accept";
                tool = tool.with_on_safety_check(move |_data| async move { accept });
            }
            let agent = Agent::<()>::new("Computer safety checks")
                .with_tool(tool)
                .with_model_instance(model);
            let options = RunOptions::new(()).run_config(RunConfig::default());
            let outcome = run_maybe_streamed(agent, "start", stream, options).await;

            let label = format!("stream={stream} handler={handler} checks={has_checks}");
            if handler == "reject" {
                let error = outcome
                    .err()
                    .unwrap_or_else(|| panic!("{label}: expected error"));
                assert!(
                    error
                        .to_string()
                        .contains("safety check was not acknowledged"),
                    "{label}: {error}"
                );
                assert!(computer.calls().is_empty(), "{label}");
            } else {
                let result = outcome.unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(
                    result.final_output_text().as_deref(),
                    Some("done"),
                    "{label}"
                );
                assert_eq!(
                    computer.calls(),
                    vec![json!(["click", 1, 2, "left"]), json!(["screenshot"])],
                    "{label}"
                );
                let raw = result
                    .new_items
                    .iter()
                    .find_map(|item| match &item.kind {
                        RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
                        _ => None,
                    })
                    .unwrap();
                if handler == "accept" {
                    assert_eq!(
                        raw["acknowledged_safety_checks"],
                        Value::Array(checks.clone()),
                        "{label}"
                    );
                } else {
                    assert!(raw.get("acknowledged_safety_checks").is_none(), "{label}");
                }
                // The warning is logged before the driver acts.
                assert_eq!(
                    *warned_at_click.lock().unwrap(),
                    Some(should_warn),
                    "{label}"
                );
            }

            let warnings = safety_warnings();
            assert_eq!(warnings.len(), usize::from(should_warn), "{label}");
            if let Some((level, _, message)) = warnings.first() {
                assert_eq!(*level, log::Level::Warn);
                assert!(
                    message.contains("proceed without acknowledging"),
                    "{message}"
                );
                assert!(
                    message.contains("Configure ComputerTool.on_safety_check"),
                    "{message}"
                );
                assert!(!message.contains("synthetic private message"), "{message}");
            }
        }
    }
}
