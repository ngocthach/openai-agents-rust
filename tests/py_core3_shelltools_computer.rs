//! Port of the Python SDK's `tests/test_computer_tool_lifecycle.py`.
//!
//! Computers are created through `ComputerTool::from_provider` (Python's `ComputerProvider`).
//! Cases that depend on Python's per-run-context caching, on dispose failures being logged, or
//! on duck-typed `create` attributes are marked where they are declared.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::{ComputerTool, Tool};
use openai_agents_sdk::{Agent, AgentsError, Result, RunContext, Runner, ToolContext};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

/// A computer that records the actions it is asked to perform (Python's `FakeComputer`).
struct FakeComputer {
    dimensions: (u32, u32),
    actions: Arc<Mutex<Vec<String>>>,
}

impl FakeComputer {
    fn new(_label: &str) -> Self {
        FakeComputer {
            dimensions: (1, 1),
            actions: Arc::default(),
        }
    }
}

#[async_trait]
impl Computer for FakeComputer {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Mac)
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        Some(self.dimensions)
    }

    async fn screenshot(&self) -> Result<String> {
        Ok("img".into())
    }

    async fn click(&self, x: i64, y: i64, button: &str, _keys: Option<&[String]>) -> Result<()> {
        self.actions
            .lock()
            .unwrap()
            .push(format!("click {x} {y} {button}"));
        Ok(())
    }

    async fn double_click(&self, x: i64, y: i64, _keys: Option<&[String]>) -> Result<()> {
        self.actions
            .lock()
            .unwrap()
            .push(format!("double_click {x} {y}"));
        Ok(())
    }

    async fn scroll(
        &self,
        x: i64,
        y: i64,
        _scroll_x: i64,
        _scroll_y: i64,
        _keys: Option<&[String]>,
    ) -> Result<()> {
        self.actions.lock().unwrap().push(format!("scroll {x} {y}"));
        Ok(())
    }

    async fn type_text(&self, text: &str) -> Result<()> {
        self.actions.lock().unwrap().push(format!("type {text}"));
        Ok(())
    }

    async fn wait(&self) -> Result<()> {
        self.actions.lock().unwrap().push("wait".into());
        Ok(())
    }

    async fn move_mouse(&self, x: i64, y: i64, _keys: Option<&[String]>) -> Result<()> {
        self.actions.lock().unwrap().push(format!("move {x} {y}"));
        Ok(())
    }

    async fn keypress(&self, keys: &[String]) -> Result<()> {
        self.actions
            .lock()
            .unwrap()
            .push(format!("keypress {}", keys.join("+")));
        Ok(())
    }

    async fn drag(&self, path: &[(i64, i64)], _keys: Option<&[String]>) -> Result<()> {
        self.actions
            .lock()
            .unwrap()
            .push(format!("drag {}", path.len()));
        Ok(())
    }
}

/// A computer-call item for a single screenshot action.
fn screenshot_call(call_id: &str) -> Value {
    json!({
        "type": "computer_call",
        "id": call_id,
        "call_id": call_id,
        "action": {"type": "screenshot"},
        "pending_safety_checks": [],
        "status": "completed",
    })
}

/// A provider whose factory counts its creations and whose dispose records what it was given.
#[derive(Clone, Default)]
struct Provider {
    created: Arc<Mutex<Vec<String>>>,
    disposed: Arc<Mutex<Vec<String>>>,
}

impl Provider {
    fn tool(&self) -> ComputerTool<()> {
        let created = self.created.clone();
        let disposed = self.disposed.clone();
        ComputerTool::from_provider(
            move || {
                let created = created.clone();
                async move {
                    let label = format!("computer-{}", created.lock().unwrap().len() + 1);
                    created.lock().unwrap().push(label.clone());
                    let computer: Arc<dyn Computer> = Arc::new(FakeComputer::new(&label));
                    Ok(computer)
                }
            },
            move |computer| {
                let disposed = disposed.clone();
                async move {
                    let _ = computer;
                    disposed.lock().unwrap().push("disposed".into());
                }
            },
        )
    }
}

#[test]
fn fake_computer_implements_interface() {
    // Every `Computer` method is callable and returns Ok.
    let computer = FakeComputer::new("iface");
    futures::executor::block_on(async {
        computer.screenshot().await.unwrap();
        computer.click(0, 0, "left", None).await.unwrap();
        computer.double_click(0, 0, None).await.unwrap();
        computer.scroll(0, 0, 1, 1, None).await.unwrap();
        computer.type_text("hello").await.unwrap();
        computer.wait().await.unwrap();
        computer.move_mouse(1, 1, None).await.unwrap();
        computer.keypress(&["enter".into()]).await.unwrap();
        computer.drag(&[(0, 0), (1, 1)], None).await.unwrap();
    });
    assert_eq!(computer.actions.lock().unwrap().len(), 8);
}

#[tokio::test]
async fn runner_disposes_computer_after_run() {
    let provider = Provider::default();
    let model = setup();
    let agent = Agent::<()>::new("ComputerAgent")
        .with_tool(provider.tool())
        .with_model_instance(model.clone());
    model.add_turn(vec![assistant_message("done")]);

    let result = Runner::run(&agent, "hello").await.unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    // The computer is resolved for the tool spec the model receives, so it is created and then
    // disposed even when the model never calls it.
    assert_eq!(provider.created.lock().unwrap().len(), 1);
    assert_eq!(provider.disposed.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn runner_creates_and_disposes_computer_for_a_computer_call() {
    let provider = Provider::default();
    let model = setup();
    let agent = Agent::<()>::new("ComputerAgent")
        .with_tool(provider.tool())
        .with_model_instance(model.clone());
    model.add_turn(vec![openai_agents_sdk::items::OutputItem::from_value(
        screenshot_call("computer-call"),
    )]);
    model.add_turn(vec![assistant_message("done")]);

    let result = Runner::run(&agent, "hello").await.unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        *provider.created.lock().unwrap(),
        vec!["computer-1".to_string()]
    );
    assert_eq!(provider.disposed.lock().unwrap().len(), 1);
    // The model saw the computer tool with its environment and display size.
    let tools = &model.calls()[0].tools;
    let computer_param = tools
        .iter()
        .find_map(|t| match t {
            openai_agents_sdk::models::interface::ToolSpec::Hosted { param, .. }
                if param["type"] == "computer" || param["type"] == "computer_use_preview" =>
            {
                Some(param.clone())
            }
            _ => None,
        })
        .expect("computer tool param");
    assert_eq!(computer_param["environment"], "mac");
}

#[tokio::test]
async fn streamed_run_disposes_computer_after_completion() {
    let provider = Provider::default();
    let model = setup();
    let agent = Agent::<()>::new("ComputerAgent")
        .with_tool(provider.tool())
        .with_model_instance(model.clone());
    model.add_turn(vec![openai_agents_sdk::items::OutputItem::from_value(
        screenshot_call("computer-call"),
    )]);
    model.add_turn(vec![assistant_message("done")]);

    let mut streamed = Runner::run_streamed(&agent, "hello");
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }

    assert_eq!(streamed.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        *provider.created.lock().unwrap(),
        vec!["computer-1".to_string()]
    );
    assert_eq!(provider.disposed.lock().unwrap().len(), 1);
}

#[derive(Default)]
struct IdentityHooks {
    started: Mutex<Vec<String>>,
    ended: Mutex<Vec<String>>,
}

#[async_trait]
impl RunHooks<()> for IdentityHooks {
    async fn on_tool_start(&self, _context: &ToolContext<()>, _agent: &Agent<()>, tool: &Tool<()>) {
        self.started.lock().unwrap().push(tool.name());
    }

    async fn on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &openai_agents_sdk::tool::ToolOutput,
    ) {
        self.ended.lock().unwrap().push(tool.name());
    }
}

#[tokio::test]
async fn runner_reports_computer_tool_to_hooks() {
    let tool = ComputerTool::new(FakeComputer::new("concrete"));
    let model = setup();
    let agent = Agent::<()>::new("ComputerAgent")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![openai_agents_sdk::items::OutputItem::from_value(
        screenshot_call("computer-call"),
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let hooks = Arc::new(IdentityHooks::default());

    let result = Runner::run_with_options(
        &agent,
        "hello",
        openai_agents_sdk::run::RunOptions::<()>::new(()).hooks(hooks.clone()),
    )
    .await
    .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        *hooks.started.lock().unwrap(),
        vec!["computer_use_preview".to_string()]
    );
    assert_eq!(
        *hooks.ended.lock().unwrap(),
        vec!["computer_use_preview".to_string()]
    );
    let outputs = result
        .new_items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .count();
    assert_eq!(outputs, 1);
}

/// A start hook swapping `agent.tools` before the run starts. Rust agents are read-only while a
/// run holds them, so the swap cannot happen; the prompt-failure half is ported below.
#[tokio::test]
async fn start_hook_failed_instructions_never_create_computer() {
    let provider = Provider::default();
    let model = setup();
    let agent = Agent::<()>::new("computer")
        .with_tool(provider.tool())
        .with_dynamic_instructions(|_ctx: RunContext<()>, _agent: Arc<Agent<()>>| async move {
            Err::<String, _>(AgentsError::user("prompt failed"))
        })
        .with_model_instance(model.clone());

    let err = Runner::run(&agent, "go").await.unwrap_err();

    assert!(err.to_string().contains("prompt failed"), "{err}");
    assert!(model.calls().is_empty());
    assert!(provider.created.lock().unwrap().is_empty());
    assert!(provider.disposed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn streamed_start_hook_failed_instructions_never_create_computer() {
    let provider = Provider::default();
    let model = setup();
    let agent = Agent::<()>::new("computer")
        .with_tool(provider.tool())
        .with_dynamic_instructions(|_ctx: RunContext<()>, _agent: Arc<Agent<()>>| async move {
            Err::<String, _>(AgentsError::user("prompt failed"))
        })
        .with_model_instance(model.clone());

    let mut streamed = Runner::run_streamed(&agent, "go");
    let mut failed = false;
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if event.is_err() {
                failed = true;
            }
        }
    }

    assert!(
        failed || streamed.result().is_none(),
        "streamed run must fail"
    );
    assert!(model.calls().is_empty());
    assert!(provider.created.lock().unwrap().is_empty());
}

/// Two computers created by the same tool, one per run context.
#[tokio::test]
async fn resolve_computer_per_run_context() {
    let provider = Provider::default();
    let tool = provider.tool();
    let ctx_a = RunContext::new(());
    let ctx_b = RunContext::new(());

    let a1 = tool.resolve(&ctx_a).await.unwrap();
    let a2 = tool.resolve(&ctx_a).await.unwrap();
    let b1 = tool.resolve(&ctx_b).await.unwrap();

    assert!(Arc::ptr_eq(&a1, &a2));
    assert!(!Arc::ptr_eq(&a1, &b1));
    assert_eq!(provider.created.lock().unwrap().len(), 2);
}

/// A model that waits at a barrier before answering, so two runs are in flight together. Each
/// run's first request is recorded with the display width of the computer it sent.
#[derive(Debug)]
struct GatedModel {
    inner: Arc<FakeModel>,
    gate: Arc<tokio::sync::Barrier>,
    first_widths: Mutex<Vec<i64>>,
}

#[async_trait]
impl openai_agents_sdk::models::interface::Model for GatedModel {
    async fn get_response(
        &self,
        request: openai_agents_sdk::models::interface::ModelRequest,
    ) -> Result<openai_agents_sdk::items::ModelResponse> {
        if request.input.len() == 1 {
            let width = request.tools.iter().find_map(|t| match t {
                openai_agents_sdk::models::interface::ToolSpec::Hosted { param, .. } => {
                    param["display_width"].as_i64()
                }
                _ => None,
            });
            self.first_widths.lock().unwrap().extend(width);
        }
        self.gate.wait().await;
        openai_agents_sdk::models::interface::Model::get_response(self.inner.as_ref(), request)
            .await
    }

    async fn stream_response(
        &self,
        _request: openai_agents_sdk::models::interface::ModelRequest,
    ) -> Result<openai_agents_sdk::models::interface::ResponseEventStream> {
        Err(AgentsError::user("streaming is not used by this test"))
    }
}

/// Concurrent runs of one agent each get their own provider instance.
#[tokio::test]
async fn concurrent_runs_keep_computer_provider_instances_isolated() {
    let created: Arc<Mutex<Vec<String>>> = Arc::default();
    let disposed: Arc<Mutex<Vec<String>>> = Arc::default();
    let create_log = created.clone();
    let dispose_log = disposed.clone();
    let tool = ComputerTool::from_provider(
        move || {
            let create_log = create_log.clone();
            async move {
                let mut created = create_log.lock().unwrap();
                let n = created.len() + 1;
                created.push(format!("computer-{n}"));
                let width = 1000 + n as u32;
                let computer: Arc<dyn Computer> = Arc::new(FakeComputer {
                    dimensions: (width, 700),
                    ..FakeComputer::new("sized")
                });
                Ok(computer)
            }
        },
        move |_computer| {
            let dispose_log = dispose_log.clone();
            async move {
                dispose_log.lock().unwrap().push("disposed".into());
            }
        },
    );
    let inner = setup();
    for _ in 0..2 {
        inner.add_turn(vec![openai_agents_sdk::items::OutputItem::from_value(
            screenshot_call("computer-call"),
        )]);
    }
    for _ in 0..2 {
        inner.add_turn(vec![assistant_message("done")]);
    }
    let model = Arc::new(GatedModel {
        inner,
        gate: Arc::new(tokio::sync::Barrier::new(2)),
        first_widths: Mutex::new(Vec::new()),
    });
    let agent = Agent::<()>::new("ComputerAgent")
        .with_tool(tool)
        .with_model_instance(model.clone());

    let (a, b) = tokio::join!(Runner::run(&agent, "run-a"), Runner::run(&agent, "run-b"));

    assert_eq!(a.unwrap().final_output_text().as_deref(), Some("done"));
    assert_eq!(b.unwrap().final_output_text().as_deref(), Some("done"));
    let mut widths = model.first_widths.lock().unwrap().clone();
    widths.sort();
    assert_eq!(widths, vec![1001, 1002]);
    assert_eq!(created.lock().unwrap().len(), 2);
    assert_eq!(disposed.lock().unwrap().len(), 2);
}

/// Captures every log record's message for the dispose-failure test.
struct Capture;

static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());
static CAPTURE: Capture = Capture;

impl log::Log for Capture {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        CAPTURED.lock().unwrap().push(record.args().to_string());
    }

    fn flush(&self) {}
}

/// A dispose failure is logged, with its text redacted unless tool data logging is allowed.
#[tokio::test]
async fn dispose_computer_failure_respects_tool_data_policy() {
    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Warn);
    for redacted in [true, false] {
        openai_agents_sdk::debug::set_dont_log_tool_data(redacted);
        CAPTURED.lock().unwrap().clear();
        let model = setup();
        model.add_turn(vec![openai_agents_sdk::items::OutputItem::from_value(
            screenshot_call("computer-call"),
        )]);
        model.add_turn(vec![assistant_message("done")]);
        let tool = ComputerTool::from_provider(
            || async {
                let computer: Arc<dyn Computer> = Arc::new(FakeComputer::new("dispose"));
                Ok(computer)
            },
            |_computer| async {
                Err::<(), _>(AgentsError::user("SECRET_COMPUTER_DISPOSE_FAILURE"))
            },
        );
        let agent = Agent::<()>::new("ComputerAgent")
            .with_tool(tool)
            .with_model_instance(model.clone());

        Runner::run(&agent, "hello").await.unwrap();

        let logs = CAPTURED.lock().unwrap().join("\n");
        openai_agents_sdk::debug::reset_dont_log_tool_data();
        assert!(
            logs.contains("Failed to dispose computer for run context"),
            "{logs}"
        );
        assert_eq!(
            !logs.contains("SECRET_COMPUTER_DISPOSE_FAILURE"),
            redacted,
            "{logs}"
        );
    }
}

// Python-only, not ported: `resolve_computer_with_create_attribute_returns_instance` relies on a
// duck-typed `create` attribute, which Rust's typed `ComputerTool::new` / `from_provider` do not
// have. The two start-hook cases below that swap `agent.tools` mid-run are Python-only too.
