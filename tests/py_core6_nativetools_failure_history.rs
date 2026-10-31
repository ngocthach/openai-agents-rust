//! Ports of `test_native_tool_failure_history.py`: native tools (computer, custom, apply patch,
//! shell, local shell) run concurrently with the function batch, are cancelled (queued calls
//! never start, an invoking call is dropped) when a sibling function fails, keep a bounded wait
//! for a call that already produced its output, and leave a consistent failure history.
//!
//! Mapping notes. Python's `asyncio.Event` is [`Event`]. The Python tests observe the category
//! failure by monkeypatching `gather_with_cancel`; here the failing tool sets `failed` right
//! before it returns its error (selection follows within the same poll). A cancelled Python
//! task is a dropped Rust future, observed with [`DropFlag`]. The Python 5 s finalization wait
//! is `RunConfig::with_tool_post_invoke_wait`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::items::{InputItem, ModelResponse, OutputItem, RunItem, RunItemKind};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run::{RunCheckpoint, RunOptions};
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, function_call_full, scripted_stream_events,
};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ComputerTool, CustomTool, FunctionTool, LocalShellTool, ShellTool, Tool,
    ToolOutput,
};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, InputGuardrail, Result, RunHooks, Runner, Session,
};
use serde_json::{Value, json};
use tokio::sync::Notify;

#[derive(Default)]
struct Event {
    flag: AtomicBool,
    notify: Notify,
}

impl Event {
    fn new() -> Arc<Self> {
        Arc::new(Event::default())
    }
    fn set(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            futures::pin_mut!(notified);
            notified.as_mut().enable();
            if self.is_set() {
                return;
            }
            notified.await;
        }
    }
    async fn wait_for(&self, what: &str) {
        tokio::time::timeout(Duration::from_secs(5), self.wait())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }
}

/// Sets an event when dropped (a cancelled future drops its locals).
struct DropFlag(Arc<Event>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.set();
    }
}

const SETTLE: Duration = Duration::from_millis(30);

fn config(wait: Duration) -> RunConfig<()> {
    RunConfig::default().with_tool_post_invoke_wait(wait)
}

/// Steps answering both the plain and the streamed path; native call items need their stream
/// events scripted exactly (Python's `get_exact_output_stream_step`).
fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let steps = steps
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            let mut step = ScriptedStep::output(output.clone());
            let response = ModelResponse {
                output,
                usage: Default::default(),
                response_id: Some(format!("resp-{index}")),
                request_id: None,
                raw_usage: None,
            };
            step.stream_events = Some(scripted_stream_events(&step, response));
            step
        })
        .collect();
    Arc::new(ScriptedModel::with_steps(steps).unwrap())
}

fn item(value: Value) -> OutputItem {
    OutputItem::from_value(value)
}

fn call(name: &str, id: &str) -> OutputItem {
    function_call_full(name, "{}", id, None, None)
}

async fn session_items(session: &dyn Session) -> Vec<Value> {
    let items: Vec<InputItem> = session.get_items(None).await.unwrap();
    items.iter().map(|i| i.as_value().clone()).collect()
}

fn call_ids(items: &[Value]) -> Vec<Option<String>> {
    items
        .iter()
        .map(|i| i.get("call_id").and_then(Value::as_str).map(str::to_string))
        .collect()
}

fn output_items(items: &[RunItem]) -> Vec<&RunItem> {
    items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .collect()
}

fn custom_data(item: &RunItem) -> Option<Value> {
    match &item.kind {
        RunItemKind::ToolCallOutput { custom_data, .. } => custom_data.clone(),
        _ => None,
    }
}

fn output_text(item: &RunItem) -> Value {
    match &item.kind {
        RunItemKind::ToolCallOutput { output, .. } => output.clone(),
        _ => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Native tool construction (`_native_tool`)
// ---------------------------------------------------------------------------

struct Probe {
    entered: Arc<Event>,
    release: Arc<Event>,
    finalized: Arc<Event>,
    effects: Arc<Mutex<Vec<String>>>,
}

impl Probe {
    fn new() -> Arc<Self> {
        Arc::new(Probe {
            entered: Event::new(),
            release: Event::new(),
            finalized: Event::new(),
            effects: Arc::new(Mutex::new(Vec::new())),
        })
    }
    fn effects(&self) -> Vec<String> {
        self.effects.lock().unwrap().clone()
    }
}

/// The extractor of the native tools: blocks (the "extractor" phase) or answers at once.
async fn extract(probe: Arc<Probe>, block: bool) -> Option<Value> {
    if block {
        probe.entered.set();
        probe.release.wait().await;
    }
    Some(json!({"finalized": true}))
}

struct StubComputer {
    probe: Arc<Probe>,
}

#[async_trait]
impl Computer for StubComputer {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Browser)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((800, 600))
    }
    async fn screenshot(&self) -> Result<String> {
        self.probe.effects.lock().unwrap().push("computer".into());
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

struct RecordingEditor {
    probe: Arc<Probe>,
}

#[async_trait]
impl ApplyPatchEditor for RecordingEditor {
    async fn create_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::completed(format!("Created {}", op.path)))
    }
    async fn update_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        self.probe.effects.lock().unwrap().push("patch".into());
        Ok(ApplyPatchResult::completed(format!("Updated {}", op.path)))
    }
    async fn delete_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::completed(format!("Deleted {}", op.path)))
    }
}

/// The native tool of `kind`, its call item, and whether the tool takes an extractor.
/// `block_extractor` makes the extractor the blocking finalization step.
fn native_tool(kind: &str, probe: &Arc<Probe>, block_extractor: bool) -> (Tool<()>, OutputItem) {
    let effect = |probe: &Arc<Probe>, kind: &'static str| {
        probe.effects.lock().unwrap().push(kind.to_string());
    };
    match kind {
        "custom" => {
            let p = probe.clone();
            let q = probe.clone();
            let tool =
                CustomTool::<()>::new("native", "Synthetic native tool", move |_ctx, _input| {
                    let p = p.clone();
                    async move {
                        effect(&p, "custom");
                        Ok("completed".to_string())
                    }
                })
                .with_custom_data_extractor(move |_ctx| extract(q.clone(), block_extractor));
            (
                tool.into(),
                item(json!({
                    "type": "custom_tool_call", "name": "native", "call_id": "native",
                    "input": "synthetic",
                })),
            )
        }
        "computer" => {
            let q = probe.clone();
            let tool = ComputerTool::<()>::new(StubComputer {
                probe: probe.clone(),
            })
            .with_custom_data_extractor(move |_ctx| extract(q.clone(), block_extractor));
            (
                tool.into(),
                item(json!({
                    "type": "computer_call", "id": "native", "call_id": "native",
                    "action": {"type": "screenshot"}, "pending_safety_checks": [],
                    "status": "completed",
                })),
            )
        }
        "patch" => {
            let q = probe.clone();
            let tool = ApplyPatchTool::<()>::new(RecordingEditor {
                probe: probe.clone(),
            })
            .with_custom_data_extractor(move |_ctx| extract(q.clone(), block_extractor));
            (
                tool.into(),
                item(json!({
                    "type": "apply_patch_call", "id": "native", "call_id": "native",
                    "status": "completed",
                    "operation": {"type": "update_file", "path": "synthetic.txt", "diff": "-a\n+b\n"},
                })),
            )
        }
        "shell" => {
            let p = probe.clone();
            let tool = ShellTool::<()>::new(move |_req| {
                let p = p.clone();
                async move {
                    effect(&p, "shell");
                    Ok::<_, AgentsError>("completed")
                }
            });
            (
                tool.into(),
                item(json!({
                    "type": "shell_call", "id": "native", "call_id": "native",
                    "status": "in_progress",
                    "action": {"type": "exec", "commands": ["echo test"], "timeout_ms": 1000},
                })),
            )
        }
        _ => {
            let p = probe.clone();
            let tool = LocalShellTool::<()>::new(move |_req| {
                let p = p.clone();
                async move {
                    effect(&p, "local_shell");
                    Ok("completed".to_string())
                }
            });
            (
                tool.into(),
                item(json!({
                    "type": "local_shell_call", "id": "native", "call_id": "native",
                    "status": "completed",
                    "action": {"type": "exec", "command": ["synthetic"], "env": {}},
                })),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Hooks and the run driver
// ---------------------------------------------------------------------------

type HookFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

/// `on_tool_end` runs `action` with the name of the finished tool.
struct EndHook(Box<dyn Fn(String) -> HookFuture + Send + Sync>);

#[async_trait]
impl RunHooks<()> for EndHook {
    async fn on_tool_end(
        &self,
        _context: &ToolContext<()>,
        _agent: &Agent<()>,
        tool: &Tool<()>,
        _result: &ToolOutput,
    ) {
        (self.0)(tool.name().to_string()).await;
    }
}

fn end_hook<F, Fut>(action: F) -> Arc<dyn RunHooks<()>>
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    Arc::new(EndHook(Box::new(move |name| Box::pin(action(name)))))
}

#[derive(Debug)]
struct Outcome {
    result: Result<()>,
    /// Items of the failed run: `run_data.new_items` when not streaming, the failure checkpoint
    /// otherwise.
    items: Vec<RunItem>,
    /// Tool output items announced on the stream.
    output_events: Vec<RunItem>,
}

async fn drive(agent: Agent<()>, streaming: bool, options: RunOptions<()>) -> Outcome {
    let checkpoint = RunCheckpoint::new();
    let options = options.checkpoint(&checkpoint);
    let mut output_events = Vec::new();
    let result = if streaming {
        let mut streamed = Runner::run_streamed_with_options(agent, "go", options);
        let mut failure = None;
        {
            let events = streamed.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                match event {
                    Ok(StreamEvent::RunItem {
                        name: RunItemStreamEventName::ToolOutput,
                        item,
                    }) => output_events.push(item),
                    Ok(_) => {}
                    Err(error) => failure = Some(error),
                }
            }
        }
        failure.map_or(Ok(()), Err)
    } else {
        Runner::run_with_options(agent, "go", options)
            .await
            .map(|_| ())
    };
    let items = match &result {
        Err(error) if !streaming => error
            .run_data()
            .map(|d| d.new_items.clone())
            .unwrap_or_default(),
        _ => checkpoint
            .take()
            .map(|s| s.generated_items.clone())
            .unwrap_or_default(),
    };
    Outcome {
        result,
        items,
        output_events,
    }
}

fn assert_user_error(result: &Result<()>, message: &str, label: &str) {
    let error = result.as_ref().expect_err(label);
    assert!(
        matches!(error, AgentsError::UserError { .. }),
        "{label}: {error:?}"
    );
    assert!(error.to_string().contains(message), "{label}: {error}");
}

async fn join(task: tokio::task::JoinHandle<Outcome>) -> Outcome {
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the run ends")
        .unwrap()
}

// ===========================================================================
// test_native_finalization_survives_sibling_failure
// ===========================================================================

#[tokio::test]
async fn native_finalization_survives_sibling_failure() {
    // `hook_raises` is Python-only: a Rust `RunHooks::on_tool_end` cannot fail.
    let cases = [
        ("computer", "extractor"),
        ("custom", "extractor"),
        ("patch", "extractor"),
        ("custom", "hook"),
        ("shell", "hook"),
        ("local_shell", "hook"),
    ];
    for streaming in [false, true] {
        for (kind, phase) in cases {
            let label = format!("{kind}/{phase} streaming={streaming}");
            let probe = Probe::new();
            let failed = Event::new();
            let (native, native_call) = native_tool(kind, &probe, phase == "extractor");
            let (entered, gate, flag) =
                (probe.entered.clone(), probe.entered.clone(), failed.clone());
            let failed_tool = FunctionTool::<()>::new_no_args("failed_tool", "fails", move |_c| {
                let (gate, flag) = (gate.clone(), flag.clone());
                async move {
                    gate.wait().await;
                    flag.set();
                    Err::<String, _>(AgentsError::other("synthetic sibling failure"))
                }
            })
            .without_failure_error_function();
            let _ = entered;
            let hook_probe = probe.clone();
            let hook_blocks = phase == "hook";
            let hooks = end_hook(move |name| {
                let probe = hook_probe.clone();
                async move {
                    if name == "failed_tool" {
                        return;
                    }
                    if hook_blocks {
                        probe.entered.set();
                        probe.release.wait().await;
                    }
                    probe.finalized.set();
                }
            });
            let agent = Agent::<()>::new("native-agent")
                .with_model_instance(scripted(vec![vec![
                    native_call,
                    call("failed_tool", "failed"),
                ]]))
                .with_tool(native)
                .with_tool(failed_tool);
            let session: Arc<dyn Session> = Arc::new(InMemorySession::new("native"));
            let options = RunOptions::default()
                .session(session.clone())
                .hooks(hooks)
                .run_config(config(Duration::from_secs(5)));
            let task = tokio::spawn(drive(agent, streaming, options));

            failed.wait_for("the category failure").await;
            tokio::time::sleep(SETTLE).await;
            assert_eq!(probe.effects(), vec![kind], "{label}");
            assert!(!probe.finalized.is_set(), "{label}");
            probe.release.set();
            let outcome = join(task).await;
            assert_user_error(&outcome.result, "synthetic sibling failure", &label);
            assert!(probe.finalized.is_set(), "{label}");
            assert_eq!(outcome.items.len(), 2, "{label}: {:?}", outcome.items);
            assert!(
                matches!(outcome.items[0].kind, RunItemKind::ToolCall { .. }),
                "{label}"
            );
            assert!(
                matches!(outcome.items[1].kind, RunItemKind::ToolCallOutput { .. }),
                "{label}"
            );
            if phase == "extractor" {
                assert_eq!(
                    custom_data(&outcome.items[1]),
                    Some(json!({"finalized": true})),
                    "{label}"
                );
            }
            let saved = session_items(session.as_ref()).await;
            assert_eq!(
                call_ids(&saved),
                vec![None, Some("native".into()), Some("native".into())],
                "{label}"
            );
            if streaming {
                assert_eq!(outcome.output_events.len(), 1, "{label}");
                assert_eq!(
                    custom_data(&outcome.output_events[0]),
                    custom_data(&outcome.items[1]),
                    "{label}"
                );
            }
        }
    }
}

// ===========================================================================
// test_native_finalization_preserves_output_without_delaying_parent
// ===========================================================================

#[tokio::test]
async fn native_finalization_preserves_output_after_drain_timeout() {
    let probe = Probe::new();
    let exited = Event::new();
    let (p, ex) = (probe.clone(), exited.clone());
    let tool = CustomTool::<()>::new("native", "Synthetic tool", {
        let p = probe.clone();
        move |_c, _i| {
            let p = p.clone();
            async move {
                p.effects.lock().unwrap().push("custom".into());
                Ok("completed".to_string())
            }
        }
    })
    .with_custom_data_extractor(move |_ctx| {
        let (p, ex) = (p.clone(), ex.clone());
        async move {
            p.entered.set();
            let _flag = DropFlag(ex);
            p.release.wait().await;
            Some(json!({"finalized": true}))
        }
    });
    let gate = probe.entered.clone();
    let sibling = FunctionTool::<()>::new_no_args("sibling", "fails", move |_c| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            Err::<String, _>(AgentsError::other("synthetic sibling failure"))
        }
    })
    .without_failure_error_function();
    let agent = Agent::<()>::new("native-agent")
        .with_model_instance(scripted(vec![vec![
            item(json!({
                "type": "custom_tool_call", "name": "native", "call_id": "native",
                "input": "synthetic",
            })),
            call("sibling", "sibling"),
        ]]))
        .with_tool(tool)
        .with_tool(sibling);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("native-stop"));
    // The finalization budget is exhausted at once (the Python test patches it to 1 ms).
    let options = RunOptions::default()
        .session(session.clone())
        .run_config(config(Duration::from_millis(1)));
    let outcome = join(tokio::spawn(drive(agent, false, options))).await;
    assert_user_error(
        &outcome.result,
        "synthetic sibling failure",
        "drain timeout",
    );
    assert!(exited.is_set(), "the finalization was abandoned");
    assert!(!probe.release.is_set());
    let last = outcome.items.last().expect("the committed output");
    assert_eq!(output_text(last), json!("completed"));
    assert_eq!(custom_data(last), None);
    assert_eq!(probe.effects(), vec!["custom"]);
    let saved = session_items(session.as_ref()).await;
    assert_eq!(saved.last().unwrap()["output"], json!("completed"));
}

#[tokio::test]
async fn native_finalization_is_dropped_when_the_run_is_cancelled() {
    let probe = Probe::new();
    let exited = Event::new();
    let (p, ex) = (probe.clone(), exited.clone());
    let tool = CustomTool::<()>::new("native", "Synthetic tool", {
        let p = probe.clone();
        move |_c, _i| {
            let p = p.clone();
            async move {
                p.effects.lock().unwrap().push("custom".into());
                Ok("completed".to_string())
            }
        }
    })
    .with_custom_data_extractor(move |_ctx| {
        let (p, ex) = (p.clone(), ex.clone());
        async move {
            p.entered.set();
            let _flag = DropFlag(ex);
            p.release.wait().await;
            Some(json!({"finalized": true}))
        }
    });
    let (gate, release) = (probe.entered.clone(), probe.release.clone());
    let sibling = FunctionTool::<()>::new_no_args("sibling", "fails", move |_c| {
        let (gate, release) = (gate.clone(), release.clone());
        async move {
            gate.wait().await;
            release.wait().await;
            Err::<String, _>(AgentsError::other("synthetic sibling failure"))
        }
    })
    .without_failure_error_function();
    let agent = Agent::<()>::new("native-agent")
        .with_model_instance(scripted(vec![vec![
            item(json!({
                "type": "custom_tool_call", "name": "native", "call_id": "native",
                "input": "synthetic",
            })),
            call("sibling", "sibling"),
        ]]))
        .with_tool(tool)
        .with_tool(sibling);
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("native-cancel"));
    let options = RunOptions::default().session(session.clone());
    let task = tokio::spawn(drive(agent, false, options));
    probe.entered.wait_for("the extractor").await;
    task.abort();
    assert!(task.await.expect_err("aborted").is_cancelled());
    exited.wait_for("the extractor to be dropped").await;
    assert_eq!(probe.effects(), vec!["custom"]);
    let saved = session_items(session.as_ref()).await;
    let roles: Vec<_> = saved.iter().map(|i| i["role"].clone()).collect();
    assert_eq!(roles, vec![json!("user")]);
}

// ===========================================================================
// test_input_verdict_during_tool_drain_preserves_selected_tool_error
// ===========================================================================

#[tokio::test]
async fn input_verdict_during_tool_drain_preserves_selected_tool_error() {
    // `tool_cancel` (a tool raising `CancelledError`) and the `cancelled_child` verdict are
    // asyncio-only.
    for verdict in ["pass", "reject", "error"] {
        for kind in ["custom", "function"] {
            for streaming in [false, true] {
                let label = format!("{verdict}/{kind} streaming={streaming}");
                let probe = Probe::new();
                let failed = Event::new();
                let (native, native_call) = if kind == "function" {
                    let p = probe.clone();
                    let completed = FunctionTool::<()>::new_no_args("completed", "ok", move |_c| {
                        let p = p.clone();
                        async move {
                            p.effects.lock().unwrap().push("function".into());
                            Ok("completed".to_string())
                        }
                    });
                    (Tool::from(completed), call("completed", "native"))
                } else {
                    native_tool("custom", &probe, true)
                };
                let (gate, flag) = (probe.entered.clone(), failed.clone());
                let fail = FunctionTool::<()>::new_no_args("fail", "fails", move |_c| {
                    let (gate, flag) = (gate.clone(), flag.clone());
                    async move {
                        gate.wait().await;
                        flag.set();
                        Err::<String, _>(AgentsError::other("primary tool failure"))
                    }
                })
                .without_failure_error_function();
                let (g_failed, g_release) = (failed.clone(), probe.release.clone());
                let guardrail = InputGuardrail::new("input_verdict", move |_c, _a, _i| {
                    let (failed, release) = (g_failed.clone(), g_release.clone());
                    async move {
                        failed.wait().await;
                        release.set();
                        if verdict == "error" {
                            return Err(AgentsError::other("late input failure"));
                        }
                        Ok(GuardrailFunctionOutput::new(
                            Value::Null,
                            verdict == "reject",
                        ))
                    }
                });
                let hook_probe = probe.clone();
                let hooks = end_hook(move |name| {
                    let probe = hook_probe.clone();
                    async move {
                        if name == "completed" {
                            probe.entered.set();
                            probe.release.wait().await;
                        }
                    }
                });
                let agent = Agent::<()>::new("native-agent")
                    .with_model_instance(scripted(vec![vec![native_call, call("fail", "fail")]]))
                    .with_tool(native)
                    .with_tool(fail)
                    .with_input_guardrail(guardrail);
                let session: Arc<dyn Session> = Arc::new(InMemorySession::new("verdict"));
                let options = RunOptions::default()
                    .session(session.clone())
                    .hooks(hooks)
                    .run_config(config(Duration::from_secs(5)));
                let outcome = join(tokio::spawn(drive(agent, streaming, options))).await;
                assert_user_error(&outcome.result, "primary tool failure", &label);
                assert_eq!(probe.effects(), vec![kind], "{label}");
                let saved = session_items(session.as_ref()).await;
                assert_eq!(
                    saved.len(),
                    if verdict == "pass" { 3 } else { 1 },
                    "{label}: {saved:?}"
                );
                if streaming || verdict == "pass" {
                    assert_eq!(
                        outcome.items.len(),
                        if verdict == "pass" { 2 } else { 0 },
                        "{label}"
                    );
                    if verdict == "pass" && kind == "custom" {
                        assert_eq!(
                            custom_data(outcome.items.last().unwrap()),
                            Some(json!({"finalized": true})),
                            "{label}"
                        );
                    }
                }
            }
        }
    }
}

// ===========================================================================
// test_selected_function_failure_stops_queued_native_invocations
// ===========================================================================

#[tokio::test]
async fn selected_function_failure_stops_queued_native_invocations() {
    // `cleanup_error` (a native invocation that turns its cancellation into an error) is
    // asyncio-only: a dropped future cannot run anything.
    for kind in ["custom", "local_shell"] {
        for reject_input in [false, true] {
            for streaming in [false, true] {
                let label = format!("{kind} reject={reject_input} streaming={streaming}");
                let function_finalizing = Event::new();
                let release_function = Event::new();
                let function_finalized = Event::new();
                let native_started = Event::new();
                let native_cancelled = Event::new();
                let selected = Event::new();
                let verdict_returned = Event::new();
                let effects = Arc::new(Mutex::new(Vec::<String>::new()));

                let completed = FunctionTool::<()>::new_no_args("completed", "ok", |_c| async {
                    Ok("completed output".to_string())
                });
                let (finalizing, started, flag) = (
                    function_finalizing.clone(),
                    native_started.clone(),
                    selected.clone(),
                );
                let fail = FunctionTool::<()>::new_no_args("fail", "fails", move |_c| {
                    let (finalizing, started, flag) =
                        (finalizing.clone(), started.clone(), flag.clone());
                    async move {
                        finalizing.wait().await;
                        started.wait().await;
                        flag.set();
                        Err::<String, _>(AgentsError::other("selected function failure"))
                    }
                })
                .without_failure_error_function();
                let invoke = {
                    let (effects, started, cancelled) = (
                        effects.clone(),
                        native_started.clone(),
                        native_cancelled.clone(),
                    );
                    move |value: String| {
                        let (effects, started, cancelled) =
                            (effects.clone(), started.clone(), cancelled.clone());
                        async move {
                            effects.lock().unwrap().push(value.clone());
                            if value == "first" {
                                started.set();
                                let _flag = DropFlag(cancelled);
                                std::future::pending::<()>().await;
                            }
                            "native output".to_string()
                        }
                    }
                };
                let (native, native_calls): (Tool<()>, Vec<OutputItem>) = if kind == "custom" {
                    let tool =
                        CustomTool::<()>::new("native", "Synthetic tool", move |_c, input| {
                            let run = invoke(input);
                            async move { Ok(run.await) }
                        });
                    let calls = ["first", "queued"]
                        .iter()
                        .map(|v| {
                            item(json!({
                                "type": "custom_tool_call", "name": "native",
                                "call_id": v, "input": v,
                            }))
                        })
                        .collect();
                    (tool.into(), calls)
                } else {
                    let tool = LocalShellTool::<()>::new(move |req| {
                        let call_id = req.data["call_id"].as_str().unwrap_or_default().to_string();
                        let run = invoke(call_id);
                        async move { Ok(run.await) }
                    });
                    let calls = ["first", "queued"]
                        .iter()
                        .map(|v| {
                            item(json!({
                                "type": "local_shell_call", "id": v, "call_id": v,
                                "status": "completed",
                                "action": {"type": "exec", "command": ["synthetic"], "env": {}},
                            }))
                        })
                        .collect();
                    (tool.into(), calls)
                };
                let (g_selected, g_returned) = (selected.clone(), verdict_returned.clone());
                let guardrail = InputGuardrail::new("verdict", move |_c, _a, _i| {
                    let (selected, returned) = (g_selected.clone(), g_returned.clone());
                    async move {
                        selected.wait().await;
                        // Let the batch cancel the native calls before the verdict lands.
                        tokio::time::sleep(SETTLE).await;
                        returned.set();
                        Ok(GuardrailFunctionOutput::new(Value::Null, reject_input))
                    }
                });
                let (fin, rel, done) = (
                    function_finalizing.clone(),
                    release_function.clone(),
                    function_finalized.clone(),
                );
                let hooks = end_hook(move |name| {
                    let (fin, rel, done) = (fin.clone(), rel.clone(), done.clone());
                    async move {
                        if name == "completed" {
                            fin.set();
                            rel.wait().await;
                            done.set();
                        }
                    }
                });
                let mut steps = vec![call("completed", "completed"), call("fail", "failed")];
                steps.extend(native_calls);
                let agent = Agent::<()>::new("cancel-queued-native")
                    .with_model_instance(scripted(vec![steps]))
                    .with_tool(completed)
                    .with_tool(fail)
                    .with_tool(native)
                    .with_input_guardrail(guardrail);
                let session: Arc<dyn Session> = Arc::new(InMemorySession::new("cancel-queued"));
                let options = RunOptions::default()
                    .session(session.clone())
                    .hooks(hooks)
                    .run_config(config(Duration::from_secs(5)));
                let task = tokio::spawn(drive(agent, streaming, options));

                verdict_returned.wait_for("the verdict").await;
                native_cancelled.wait_for("the native cancellation").await;
                assert_eq!(*effects.lock().unwrap(), vec!["first"], "{label}");
                assert!(!function_finalized.is_set(), "{label}");
                release_function.set();
                let outcome = join(task).await;
                assert_user_error(&outcome.result, "selected function failure", &label);
                assert!(function_finalized.is_set(), "{label}");
                let outputs = output_items(&outcome.items);
                let saved = session_items(session.as_ref()).await;
                if reject_input {
                    assert!(outputs.is_empty(), "{label}");
                    assert_eq!(saved.len(), 1, "{label}");
                } else {
                    assert_eq!(outputs.len(), 1, "{label}");
                    assert_eq!(
                        output_text(outputs[0]),
                        json!("completed output"),
                        "{label}"
                    );
                    assert_eq!(saved.len(), 3, "{label}: {saved:?}");
                }
            }
        }
    }
}

// ===========================================================================
// test_parent_cancellation_after_function_failure_cancels_pending_finalization
// ===========================================================================

#[tokio::test]
async fn parent_cancellation_after_function_failure_cancels_pending_finalization() {
    for streaming in [false, true] {
        let entered = Event::new();
        let release = Event::new();
        let cancelled = Event::new();
        let failed = Event::new();
        let completed = FunctionTool::<()>::new_no_args("completed", "ok", |_c| async {
            Ok("completed".to_string())
        });
        let (gate, flag) = (entered.clone(), failed.clone());
        let fail = FunctionTool::<()>::new_no_args("fail", "fails", move |_c| {
            let (gate, flag) = (gate.clone(), flag.clone());
            async move {
                gate.wait().await;
                flag.set();
                Err::<String, _>(AgentsError::other("selected function failure"))
            }
        })
        .without_failure_error_function();
        let (e, r, c) = (entered.clone(), release.clone(), cancelled.clone());
        let hooks = end_hook(move |name| {
            let (e, r, c) = (e.clone(), r.clone(), c.clone());
            async move {
                if name == "completed" {
                    e.set();
                    let _flag = DropFlag(c);
                    r.wait().await;
                }
            }
        });
        let agent = Agent::<()>::new("parent-cancel")
            .with_model_instance(scripted(vec![vec![
                call("completed", "completed"),
                call("fail", "failed"),
            ]]))
            .with_tool(completed)
            .with_tool(fail);
        let options = RunOptions::default()
            .hooks(hooks)
            .run_config(config(Duration::from_secs(5)));
        // A dropped plain run is cancelled with its caller; a streamed run is cancelled
        // through its handle (dropping the stream only detaches the consumer).
        let (task, cancel_handle): (tokio::task::JoinHandle<()>, _) = if streaming {
            let mut streamed = Runner::run_streamed_with_options(agent, "go", options);
            let handle = streamed.cancel_handle();
            let consumer = tokio::spawn(async move {
                let events = streamed.stream_events();
                futures::pin_mut!(events);
                while events.next().await.is_some() {}
            });
            (consumer, Some(handle))
        } else {
            let run = tokio::spawn(async move {
                let _ = Runner::run_with_options(agent, "go", options).await;
            });
            (run, None)
        };
        failed.wait_for("the selected failure").await;
        tokio::time::sleep(SETTLE).await;
        match cancel_handle {
            Some(handle) => {
                handle.cancel(CancelMode::Immediate);
                tokio::time::timeout(Duration::from_secs(2), task)
                    .await
                    .expect("the cancelled stream ends")
                    .unwrap();
            }
            None => {
                task.abort();
                assert!(task.await.expect_err("aborted").is_cancelled());
            }
        }
        cancelled.wait_for("the finalization to be dropped").await;
        assert!(!release.is_set(), "streaming={streaming}");
    }
}
