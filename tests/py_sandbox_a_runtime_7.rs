//! Ports of `tests/sandbox/test_runtime.py` (lines 3999 to the end), part 7: whole runs through
//! [`SandboxRunner`] — session history, public agent identity in items, hooks and stream events,
//! handoff preparation, memory managers, shared capabilities and provider/capability model
//! settings. The session manager half is part 5 and the lifecycle half part 6.
#![cfg(feature = "sandbox")]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::items::{InputItem, OutputItem, RunItemKind};
use openai_agents_sdk::lifecycle::{AgentHooks, RunHooks};
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::models::ModelProvider;
use openai_agents_sdk::models::interface::Model;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, Compaction, CompactionPolicy, Filesystem, Memory, Shell,
};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    DEFAULT_SANDBOX_INSTRUCTIONS, SandboxAgent, SandboxRunConfig, SandboxRunner,
};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{RunContext, RunOptions};
use serde_json::{Value, json};

// ----------------------------------------------------------------------------------------------
// Harness
// ----------------------------------------------------------------------------------------------

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect())
            .expect("scripted model"),
    )
}

fn done(text: &str) -> Vec<OutputItem> {
    vec![assistant_message(text)]
}

fn unix_local_config() -> SandboxRunConfig {
    SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
}

/// Contributes one instruction fragment, like `_RecordingCapability(instruction_text=...)`.
#[derive(Debug)]
struct InstructionCapability {
    text: &'static str,
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for InstructionCapability {
    fn capability_type(&self) -> &str {
        "instruction"
    }

    async fn instructions(
        &self,
        _manifest: &Manifest,
    ) -> openai_agents_sdk::error::Result<Option<String>> {
        Ok(Some(self.text.to_string()))
    }
}

/// Reads `README.md` out of the bound session, like `_AwaitableSessionCapability`: what it says
/// proves which session the capability was bound to for this run.
#[derive(Debug)]
struct ReadmeCapability;

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for ReadmeCapability {
    fn capability_type(&self) -> &str {
        "readme"
    }

    async fn bound_instructions(
        &self,
        binding: &CapabilityBinding,
        _manifest: &Manifest,
    ) -> openai_agents_sdk::error::Result<Option<String>> {
        let readme = binding.session.read_file("README.md").await?;
        Ok(Some(String::from_utf8_lossy(&readme).into_owned()))
    }
}

/// Records the name of every agent each hook sees, like `_RecordingRunHooks`.
#[derive(Debug, Default)]
struct HookLog {
    started: Mutex<Vec<String>>,
    ended: Mutex<Vec<String>>,
    llm_started: Mutex<Vec<String>>,
    llm_ended: Mutex<Vec<String>>,
}

impl HookLog {
    fn snapshot(&self) -> [Vec<String>; 4] {
        [
            self.started.lock().unwrap().clone(),
            self.ended.lock().unwrap().clone(),
            self.llm_started.lock().unwrap().clone(),
            self.llm_ended.lock().unwrap().clone(),
        ]
    }
}

#[async_trait]
impl RunHooks<()> for HookLog {
    async fn on_agent_start(
        &self,
        _context: &RunContext<()>,
        agent: &openai_agents_sdk::Agent<()>,
    ) {
        self.started.lock().unwrap().push(agent.name.clone());
    }

    async fn on_agent_end(
        &self,
        _context: &RunContext<()>,
        agent: &openai_agents_sdk::Agent<()>,
        _output: &Value,
    ) {
        self.ended.lock().unwrap().push(agent.name.clone());
    }

    async fn on_llm_start(
        &self,
        _context: &RunContext<()>,
        agent: &openai_agents_sdk::Agent<()>,
        _system_prompt: Option<&str>,
        _input_items: &[InputItem],
    ) {
        self.llm_started.lock().unwrap().push(agent.name.clone());
    }

    async fn on_llm_end(
        &self,
        _context: &RunContext<()>,
        agent: &openai_agents_sdk::Agent<()>,
        _response: &openai_agents_sdk::items::ModelResponse,
    ) {
        self.llm_ended.lock().unwrap().push(agent.name.clone());
    }
}

#[async_trait]
impl AgentHooks<()> for HookLog {
    async fn on_start(&self, _context: &RunContext<()>, agent: &openai_agents_sdk::Agent<()>) {
        self.started.lock().unwrap().push(agent.name.clone());
    }

    async fn on_end(
        &self,
        _context: &RunContext<()>,
        agent: &openai_agents_sdk::Agent<()>,
        _output: &Value,
    ) {
        self.ended.lock().unwrap().push(agent.name.clone());
    }
}

// ----------------------------------------------------------------------------------------------
// Session history and public agent identity
// ----------------------------------------------------------------------------------------------

/// Only this run's new input is added to the session; the old history is not duplicated. Python:
/// `test_sandbox_run_persists_only_new_session_input_items`.
#[tokio::test]
async fn a_sandbox_run_persists_only_the_new_session_input_items() {
    let session = Arc::new(InMemorySession::new("history"));
    session
        .add_items(vec![InputItem::from_value(
            json!({"role": "user", "content": "old"}),
        )])
        .await
        .expect("seed");
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(scripted(vec![done("done")]))
        .with_instructions("Base instructions.");

    let result = SandboxRunner::run_with_options(
        &agent,
        "new",
        &unix_local_config(),
        RunOptions::default().session(session.clone()),
    )
    .await
    .expect("run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let saved_user_items: Vec<Value> = session
        .get_items(None)
        .await
        .expect("items")
        .into_iter()
        .map(|item| item.0)
        .filter(|item| item["role"] == "user")
        .collect();
    assert_eq!(
        saved_user_items,
        vec![
            json!({"role": "user", "content": "old"}),
            json!({"role": "user", "content": "new"}),
        ]
    );
}

/// Reasoning, tool-call and tool-output stream events all name the public agent. Python:
/// `test_runner_streamed_emits_public_agent_for_tool_and_reasoning_events`.
#[tokio::test]
async fn streamed_events_name_the_public_agent() {
    let model = scripted(vec![
        vec![
            OutputItem::from_value(json!({"type": "reasoning", "id": "rs_1", "summary": []})),
            function_call_with_id("tool1", "{}", "call_tool"),
        ],
        done("done"),
    ]);
    let tool = FunctionTool::<()>::new("tool1", "", |_ctx, _args: Value| async move {
        Ok("tool result".to_string())
    });
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_tool(tool);

    let mut run = SandboxRunner::run_streamed(&agent, "hello", &unix_local_config())
        .await
        .expect("the stream starts");
    let mut relevant = Vec::new();
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while let Some(event) = stream.next().await {
            if let StreamEvent::RunItem { name, item } = event.expect("event")
                && matches!(
                    name,
                    RunItemStreamEventName::ReasoningItemCreated
                        | RunItemStreamEventName::ToolCalled
                        | RunItemStreamEventName::ToolOutput
                )
            {
                relevant.push((name, item.agent));
            }
        }
    }

    assert_eq!(relevant.len(), 3, "{relevant:?}");
    assert!(
        relevant.iter().all(|(_, agent)| agent == "sandbox"),
        "{relevant:?}"
    );
}

/// A shell call's output item names the public agent too. Python:
/// `test_runner_uses_public_agent_for_non_function_tool_outputs` (Python drives a local shell
/// tool; the sandbox `Shell` capability's `shell_call` is the Rust equivalent).
#[tokio::test]
async fn non_function_tool_outputs_name_the_public_agent() {
    let model = scripted(vec![
        vec![OutputItem::from_value(json!({
            "type": "shell_call", "id": "sh_1", "call_id": "call_shell",
            "status": "completed", "action": {"commands": ["echo sandbox"]}
        }))],
        done("done"),
    ]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_capabilities(vec![Arc::new(Shell::new().with_pty(false))]);

    let result = SandboxRunner::run(&agent, "hello", &unix_local_config())
        .await
        .expect("run");

    let outputs: Vec<_> = result
        .new_items
        .iter()
        .filter(|item| {
            matches!(
                &item.kind,
                RunItemKind::ToolCallOutput { raw_item, .. }
                    if raw_item["type"] == "shell_call_output"
            )
        })
        .collect();
    assert!(!outputs.is_empty());
    assert!(outputs.iter().all(|item| item.agent == "sandbox"));
}

/// Run hooks, agent hooks and the result all see the public agent, both for a plain and a
/// streamed run. Python: `test_runner_keeps_public_agent_identity_for_hooks_and_streaming`.
#[tokio::test]
async fn hooks_and_results_keep_the_public_agent_identity() {
    let run_hooks = Arc::new(HookLog::default());
    let agent_hooks = Arc::new(HookLog::default());
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(scripted(vec![done("done")]))
        .with_instructions("Base instructions.")
        .with_capabilities(vec![Arc::new(InstructionCapability {
            text: "Capability instructions.",
        })])
        .map_agent({
            let agent_hooks = agent_hooks.clone();
            move |agent| agent.with_hooks(agent_hooks)
        });

    let result = SandboxRunner::run_with_options(
        &agent,
        "hello",
        &unix_local_config(),
        RunOptions::default().hooks(run_hooks.clone()),
    )
    .await
    .expect("run");

    let one = vec!["sandbox".to_string()];
    assert_eq!(result.last_agent.name, "sandbox");
    assert_eq!(
        run_hooks.snapshot(),
        [one.clone(), one.clone(), one.clone(), one.clone()]
    );
    assert_eq!(
        agent_hooks.snapshot()[..2],
        [one.clone(), one.clone()],
        "agent hooks see the same agent"
    );
    assert!(result.new_items.iter().all(|item| item.agent == "sandbox"));

    // The streamed run.
    let streamed_hooks = Arc::new(HookLog::default());
    let streamed_agent = SandboxAgent::<()>::new("streamed-sandbox")
        .with_model_instance(scripted(vec![done("streamed done")]))
        .with_instructions("Base instructions.");
    let mut run = SandboxRunner::run_streamed_with_options(
        &streamed_agent,
        "hello",
        &unix_local_config(),
        RunOptions::default().hooks(streamed_hooks.clone()),
    )
    .await
    .expect("the stream starts");
    let mut item_agents = Vec::new();
    {
        let stream = run.streaming.stream_events();
        futures::pin_mut!(stream);
        while let Some(event) = stream.next().await {
            if let StreamEvent::RunItem { item, .. } = event.expect("event") {
                item_agents.push(item.agent);
            }
        }
    }

    let streamed_one = vec!["streamed-sandbox".to_string()];
    assert_eq!(run.streaming.current_agent().name, "streamed-sandbox");
    assert_eq!(
        streamed_hooks.snapshot(),
        [
            streamed_one.clone(),
            streamed_one.clone(),
            streamed_one.clone(),
            streamed_one.clone()
        ]
    );
    assert!(!item_agents.is_empty());
    assert!(item_agents.iter().all(|agent| agent == "streamed-sandbox"));
}

// ----------------------------------------------------------------------------------------------
// Handoffs
// ----------------------------------------------------------------------------------------------

/// The handoff target runs with its own preparation: its instructions and capability fragment,
/// not the source agent's. Python: `test_runner_reapplies_sandbox_prep_on_handoff`.
#[tokio::test]
async fn a_handoff_target_is_prepared_with_its_own_capabilities() {
    let manifest =
        Manifest::new().with_entry("README.md", Entry::text_file("Shared repo instructions."));
    let worker_model = scripted(vec![done("done")]);
    let worker = Arc::new(
        SandboxAgent::<()>::new("worker")
            .with_model_instance(worker_model.clone())
            .with_instructions("Worker instructions.")
            .with_default_manifest(manifest.clone())
            .with_capabilities(vec![Arc::new(InstructionCapability {
                text: "Worker capability.",
            })]),
    );
    let triage_model = scripted(vec![vec![function_call_with_id(
        "transfer_to_worker",
        "{}",
        "call_handoff",
    )]]);
    let triage = SandboxAgent::<()>::new("triage")
        .with_model_instance(triage_model)
        .with_instructions("Triage instructions.")
        .with_default_manifest(manifest.clone())
        .with_capabilities(vec![Arc::new(InstructionCapability {
            text: "Triage capability.",
        })])
        .with_sandbox_handoff(worker);

    let result = SandboxRunner::run_with_options(
        &triage,
        "route this",
        &unix_local_config(),
        RunOptions::default().max_turns(Some(5)),
    )
    .await
    .expect("run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let call = worker_model.first_call().expect("the worker was called");
    let instructions = call.request.system_instructions.expect("instructions");
    assert!(
        instructions.starts_with(DEFAULT_SANDBOX_INSTRUCTIONS.trim()),
        "{instructions}"
    );
    assert!(
        instructions.contains(
            "# Agent instructions\n\nWorker instructions.\n\n# Sandbox capability instructions\n\nWorker capability."
        ),
        "{instructions}"
    );
    assert!(!instructions.contains("Triage"), "{instructions}");
}

// ----------------------------------------------------------------------------------------------
// Memory managers follow the agent that is prepared
// ----------------------------------------------------------------------------------------------

fn memory_agent(name: &str, memory: Option<Memory>) -> SandboxAgent<()> {
    let mut capabilities: Vec<Arc<dyn Capability<()>>> = vec![
        Arc::new(Filesystem::new()),
        Arc::new(Shell::new().with_pty(false)),
    ];
    if let Some(memory) = memory {
        capabilities.push(Arc::new(memory));
    }
    SandboxAgent::<()>::new(name)
        .with_model_instance(scripted(vec![done("done")]))
        .with_capabilities(capabilities)
}

/// An agent with generation enabled gets a manager; an agent with only reads does not. Python:
/// `test_prepare_agent_uses_active_sandbox_agent_memory_capability_for_handoffs`.
///
/// DEFERRED: Python switches the session's manager as the *active* agent changes through a
/// handoff. `SandboxRunner::prepare` (src/sandbox/runtime.rs) builds the manager once from the
/// starting agent's capability, so a handoff target with a different `Memory` is not reflected.
/// Asserted here: the manager follows the agent each preparation starts from.
#[tokio::test]
async fn the_memory_manager_follows_the_prepared_agent_generate_setting() {
    let generating = memory_agent("triage", Some(Memory::new()));
    let reading_only = memory_agent("reviewer", Some(Memory::new().with_generate(None)));

    let first = SandboxRunner::prepare(&generating, &unix_local_config())
        .await
        .expect("prepare");
    assert!(first.memory_manager().is_some());
    first.guard.close().await.expect("close");

    let second = SandboxRunner::prepare(&reading_only, &unix_local_config())
        .await
        .expect("prepare");
    assert!(second.memory_manager().is_none());
    second.guard.close().await.expect("close");
}

/// An agent without memory gets no manager, and one that adds the capability gets one. Python:
/// `test_prepare_agent_enables_memory_when_handoff_target_adds_capability` (same DEFERRED note).
#[tokio::test]
async fn an_agent_that_adds_the_memory_capability_gets_a_manager() {
    let plain = memory_agent("triage", None);
    let with_memory = memory_agent("worker", Some(Memory::new()));

    let first = SandboxRunner::prepare(&plain, &unix_local_config())
        .await
        .expect("prepare");
    assert!(first.memory_manager().is_none());
    first.guard.close().await.expect("close");

    let second = SandboxRunner::prepare(&with_memory, &unix_local_config())
        .await
        .expect("prepare");
    assert!(second.memory_manager().is_some());
    second.guard.close().await.expect("close");
}

// ----------------------------------------------------------------------------------------------
// Reuse and isolation
// ----------------------------------------------------------------------------------------------

/// Python: `test_runner_rejects_concurrent_reuse_of_same_sandbox_agent`.
///
/// Divergence by design: Rust prepares a fresh agent per run and shares no per-run state on the
/// `SandboxAgent` (see the `Memory` capability docs: one agent serves many concurrent sessions),
/// so a second concurrent run is not an error. Asserted: both runs complete.
#[tokio::test]
async fn one_sandbox_agent_can_serve_concurrent_runs() {
    let model = scripted(vec![done("done"), done("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.");
    let config = unix_local_config();

    let (first, second) = tokio::join!(
        SandboxRunner::run(&agent, "hello", &config),
        SandboxRunner::run(&agent, "again", &config)
    );

    assert_eq!(
        first.expect("first").final_output_text().as_deref(),
        Some("done")
    );
    assert_eq!(
        second.expect("second").final_output_text().as_deref(),
        Some("done")
    );
}

/// One capability shared by two agents running at the same time is bound to each run's own
/// session. Python: `test_runner_isolates_shared_capabilities_per_run`.
#[tokio::test]
async fn a_shared_capability_is_bound_per_run() {
    let shared: Arc<dyn Capability<()>> = Arc::new(ReadmeCapability);
    let build = |name: &str, readme: &str, model: Arc<ScriptedModel>| {
        SandboxAgent::<()>::new(name)
            .with_model_instance(model)
            .with_instructions("Base instructions.")
            .with_default_manifest(
                Manifest::new().with_entry("README.md", Entry::text_file(readme)),
            )
            .with_capabilities(vec![shared.clone()])
    };
    let model_one = scripted(vec![done("done one")]);
    let model_two = scripted(vec![done("done two")]);
    let agent_one = build(
        "sandbox-one",
        "Session one instructions.",
        model_one.clone(),
    );
    let agent_two = build(
        "sandbox-two",
        "Session two instructions.",
        model_two.clone(),
    );
    let config = unix_local_config();

    let (first, second) = tokio::join!(
        SandboxRunner::run(&agent_one, "hello one", &config),
        SandboxRunner::run(&agent_two, "hello two", &config)
    );

    assert_eq!(
        first.expect("first").final_output_text().as_deref(),
        Some("done one")
    );
    assert_eq!(
        second.expect("second").final_output_text().as_deref(),
        Some("done two")
    );
    let one = model_one
        .first_call()
        .expect("call")
        .request
        .system_instructions
        .expect("prompt");
    let two = model_two
        .first_call()
        .expect("call")
        .request
        .system_instructions
        .expect("prompt");
    assert!(one.contains("Session one instructions."), "{one}");
    assert!(!one.contains("Session two instructions."), "{one}");
    assert!(two.contains("Session two instructions."), "{two}");
    assert!(!two.contains("Session one instructions."), "{two}");
}

// ----------------------------------------------------------------------------------------------
// Model settings
// ----------------------------------------------------------------------------------------------

/// Hands out one fixed model and counts how it was asked, like the `Mock(spec=ModelProvider)`.
#[derive(Debug)]
struct FixedProvider {
    model: Arc<ScriptedModel>,
    requests: Mutex<Vec<Option<String>>>,
}

impl ModelProvider for FixedProvider {
    fn get_model(
        &self,
        model_name: Option<&str>,
    ) -> openai_agents_sdk::error::Result<Arc<dyn Model>> {
        self.requests
            .lock()
            .unwrap()
            .push(model_name.map(str::to_string));
        Ok(self.model.clone())
    }
}

/// The provider is asked for its default model once, and the capability's request parameters are
/// merged with the run config's instead of replacing them. Python:
/// `test_provider_defaults_are_resolved_before_sandbox_capability_settings` (the
/// `OPENAI_DEFAULT_MODEL` environment half is Python's provider and is not ported).
#[tokio::test]
async fn provider_defaults_and_capability_settings_are_merged() {
    let model = scripted(vec![done("done")]);
    let provider = Arc::new(FixedProvider {
        model: model.clone(),
        requests: Mutex::new(Vec::new()),
    });
    let provider_dyn: Arc<dyn ModelProvider> = provider.clone();
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_default_manifest(Manifest::new())
        .with_capabilities(vec![Arc::new(
            Compaction::new().with_policy(CompactionPolicy::Static { threshold: 123 }),
        )]);
    let run_config = RunConfig::<()> {
        model_provider: provider_dyn,
        model_settings: Some(openai_agents_sdk::model_settings::ModelSettings {
            extra_args: Some(
                json!({"request_option": "override"})
                    .as_object()
                    .cloned()
                    .expect("object"),
            ),
            ..Default::default()
        }),
        ..Default::default()
    };

    let result = SandboxRunner::run_with_options(
        &agent,
        "hello",
        &unix_local_config(),
        RunOptions::default().run_config(run_config),
    )
    .await
    .expect("run");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(*provider.requests.lock().unwrap(), vec![None]);
    let call = model.first_call().expect("the model was called");
    assert_eq!(
        call.request
            .model_settings
            .extra_args
            .clone()
            .map(Value::Object),
        Some(json!({
            "context_management": [{"type": "compaction", "compact_threshold": 123}],
            "request_option": "override",
        }))
    );
}
