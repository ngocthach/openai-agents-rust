//! Per-turn sandbox preparation. Python re-prepares a sandbox agent before every model call
//! (`SandboxRuntime.prepare_agent` in `run.py` / `run_loop.py`): the prepared agent's
//! instructions callable renders the capability fragments on each call
//! (`build_sandbox_instructions`), and the capability `process_context` chain runs before the
//! caller's `call_model_input_filter`.
//!
//! Python has no runner-level test asserting a second turn's system prompt; these tests pin the
//! behaviour its `_instructions` closure and its skills metadata cache define.
#![cfg(feature = "sandbox")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::RunOptions;
use openai_agents_sdk::error::Result;
use openai_agents_sdk::items::{InputItem, OutputItem};
use openai_agents_sdk::run_config::ModelInputData;
use openai_agents_sdk::sandbox::capabilities::{Capability, Filesystem, Memory, Shell, Skills};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use serde_json::json;

fn exec_call(call_id: &str, cmd: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "function_call", "call_id": call_id, "name": "exec_command",
        "arguments": json!({"cmd": cmd, "login": false}).to_string()
    }))
}

fn prompts(model: &FakeModel) -> Vec<String> {
    model
        .calls()
        .into_iter()
        .map(|call| call.system_instructions.unwrap_or_default())
        .collect()
}

fn client_config() -> SandboxRunConfig {
    SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
}

/// Counts how often its instruction fragment is rendered.
#[derive(Debug, Default)]
struct CountingInstructions(AtomicUsize);

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for CountingInstructions {
    fn capability_type(&self) -> &str {
        "counting-instructions"
    }

    async fn instructions(&self, _manifest: &Manifest) -> Result<Option<String>> {
        let calls = self.0.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Some(format!("instructions_calls={calls}")))
    }
}

/// Python's `_instructions` closure awaits `capability.instructions(manifest)` on every render,
/// so each model call sees a freshly rendered fragment; preparation itself renders none.
#[tokio::test]
async fn capability_instructions_are_rendered_for_every_model_call() {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![exec_call("c1", "true")]);
    model.add_turn(vec![assistant_message("done")]);
    let capability = Arc::new(CountingInstructions::default());
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![Arc::new(Shell::new()), capability.clone()]);

    let result = SandboxRunner::run(&agent, "hello", &client_config())
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let prompts = prompts(&model);
    assert_eq!(prompts.len(), 2);
    assert!(
        prompts[0].contains("instructions_calls=1"),
        "{}",
        prompts[0]
    );
    assert!(
        prompts[1].contains("instructions_calls=2"),
        "{}",
        prompts[1]
    );
    assert_eq!(capability.0.load(Ordering::SeqCst), 2);
}

/// `Memory.instructions` reads `memories/memory_summary.md` from the session on every render,
/// so a summary a tool writes on the first turn reaches the second turn's system prompt.
#[tokio::test]
async fn a_memory_summary_written_mid_run_reaches_the_next_turn_prompt() {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![exec_call(
        "c1",
        "mkdir -p memories && printf 'remember the blue door' > memories/memory_summary.md",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![
            Arc::new(Shell::new()),
            Arc::new(Filesystem::new()),
            Arc::new(Memory::new().with_generate(None)),
        ]);

    let result = SandboxRunner::run(&agent, "hello", &client_config())
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let prompts = prompts(&model);
    assert_eq!(prompts.len(), 2);
    assert!(
        !prompts[0].contains("remember the blue door"),
        "{}",
        prompts[0]
    );
    assert!(
        prompts[1].contains("remember the blue door"),
        "{}",
        prompts[1]
    );
}

/// Python's skills capability caches its index for the prepared run, so a skill directory a
/// tool creates mid-run is not listed on the next turn.
#[tokio::test]
async fn the_skills_index_is_resolved_once_per_prepared_agent() {
    let skill = |name: &str| {
        Entry::dir().with_child(
            "SKILL.md",
            Entry::file(
                format!("---\nname: {name}\ndescription: the {name} skill\n---\n").into_bytes(),
            ),
        )
    };
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![exec_call(
        "c1",
        "mkdir -p .agents/beta && printf -- '---\\nname: beta\\ndescription: the beta skill\\n---\\n' > .agents/beta/SKILL.md",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![
            Arc::new(Shell::new()),
            Arc::new(Skills::from_entry(
                Entry::dir().with_child("alpha", skill("alpha")),
            )),
        ]);

    let result = SandboxRunner::run(&agent, "hello", &client_config())
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let prompts = prompts(&model);
    assert_eq!(prompts.len(), 2);
    for prompt in &prompts {
        assert!(prompt.contains("- alpha: the alpha skill"), "{prompt}");
        assert!(!prompt.contains("- beta:"), "{prompt}");
    }
}

/// Drops every input item mentioning its text.
#[derive(Debug)]
struct DropContextText(&'static str);

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for DropContextText {
    fn capability_type(&self) -> &str {
        "drop-context-text"
    }

    fn process_context(&self, context: Vec<InputItem>) -> Vec<InputItem> {
        context
            .into_iter()
            .filter(|item| !item.0.to_string().contains(self.0))
            .collect()
    }
}

/// Python rewrites the turn input with `process_context` while preparing the agent, before
/// `call_model_input_filter` runs, so the caller's filter only ever sees the processed input.
#[tokio::test]
async fn the_call_model_input_filter_sees_the_processed_context() {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model.clone())
        .with_capabilities(vec![Arc::new(DropContextText("drop me"))]);
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mut options = RunOptions::<()>::default();
    let recorder = seen.clone();
    options.run_config.call_model_input_filter = Some(Arc::new(move |data| {
        let recorder = recorder.clone();
        Box::pin(async move {
            recorder
                .lock()
                .unwrap()
                .extend(data.model_data.input.iter().map(|item| item.0.to_string()));
            Ok(ModelInputData {
                input: data.model_data.input,
                instructions: data.model_data.instructions,
            })
        })
    }));
    let input = vec![
        InputItem::from_value(json!({"role": "user", "content": "keep me"})),
        InputItem::from_value(json!({"role": "user", "content": "drop me"})),
    ];

    let result = SandboxRunner::run_with_options(&agent, input, &client_config(), options)
        .await
        .expect("the run completes");

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert!(seen[0].contains("keep me"));
    let sent = model.calls()[0].input.clone();
    assert_eq!(sent.len(), 1, "{sent:?}");
}
