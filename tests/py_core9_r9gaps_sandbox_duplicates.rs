//! Ports of the duplicate-agent-name cases of `tests/sandbox/test_runtime.py`: a handoff graph
//! may hold several agents with one name, and every per-agent structure (sessions, resume
//! states, capability lookup) tells them apart by identity.

#![cfg(all(feature = "sandbox", unix))]

use std::sync::Arc;

use openai_agents_sdk::RunOptions;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::sandbox::capabilities::Shell;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::runtime::SandboxSessionMode;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{Manifest, SandboxAgent, SandboxRunConfig, SandboxRunner};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use serde_json::json;

fn model() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn handoff_call(tool: &str, call_id: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "function_call", "call_id": call_id, "name": tool, "arguments": "{}"
    }))
}

fn shell_call(call_id: &str, command: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": [command]}
    }))
}

fn instructions(agent: &SandboxAgent<()>) -> String {
    match &agent.agent.instructions {
        Some(openai_agents_sdk::agent::Instructions::Static(text)) => text.clone(),
        other => panic!("expected static instructions, got {other:?}"),
    }
}

/// Python: `test_session_manager_generates_collision_free_resume_keys_for_literal_suffix_names`.
#[test]
fn duplicate_names_get_collision_free_identity_keys() {
    let first = SandboxAgent::<()>::new("sandbox").with_instructions("First.");
    let literal = Arc::new(SandboxAgent::<()>::new("sandbox#2").with_instructions("Literal."));
    let second = Arc::new(SandboxAgent::<()>::new("sandbox").with_instructions("Second."));
    let graph = first
        .with_sandbox_handoff(literal)
        .with_sandbox_handoff(second);

    // Post order of the spanning tree; the keys themselves are what matters.
    let mut keys = graph.graph_agent_names().unwrap();
    keys.sort();
    assert_eq!(keys, ["sandbox", "sandbox#2", "sandbox#3"]);

    // The first agent of a name and the literal-suffix agent keep their own names; the
    // duplicate takes the next free suffix.
    assert_eq!(
        instructions(&graph.graph_agent("sandbox").unwrap()),
        "First."
    );
    assert_eq!(
        instructions(&graph.graph_agent("sandbox#2").unwrap()),
        "Literal."
    );
    assert_eq!(
        instructions(&graph.graph_agent("sandbox#3").unwrap()),
        "Second."
    );
}

/// Python: `test_runner_serializes_unique_sandbox_resume_keys_for_duplicate_agent_names`
/// (the pause half): the sessions of the agents that share a name are suspended under their
/// own keys.
#[tokio::test]
async fn a_paused_run_keys_the_duplicates_sessions_by_identity() {
    let first_model = model();
    let second_model = model();
    let second = Arc::new(
        SandboxAgent::<()>::new("sandbox")
            .with_model_instance(second_model.clone())
            .with_capabilities(vec![Arc::new(
                Shell::new().with_pty(false).with_needs_approval(true),
            )])
            .with_default_manifest(Manifest::new().with_entry("b.txt", Entry::text_file("second"))),
    );
    let first = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(first_model.clone())
        .with_default_manifest(Manifest::new().with_entry("a.txt", Entry::text_file("first")))
        .with_sandbox_handoff(second);
    first_model.add_turn(vec![handoff_call("transfer_to_sandbox", "handoff")]);
    second_model.add_turn(vec![shell_call("c1", "cat b.txt")]);

    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
        .session_mode(SandboxSessionMode::PerAgent);
    let outcome = SandboxRunner::run_to_outcome(
        &first,
        "hello",
        &config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .expect("the run pauses");

    assert_eq!(outcome.result.interruptions.len(), 1);
    assert!(
        outcome.sandbox_state.is_some(),
        "the starting session is suspended"
    );
    let keys: Vec<&String> = outcome.sandbox_states_by_agent.keys().collect();
    assert_eq!(keys, ["sandbox#2"]);
}

/// Two agents sharing a name each run in their own workspace and with their own prepared
/// capabilities: the second one's shell sees only its own entry.
#[tokio::test]
async fn duplicate_named_agents_run_in_their_own_workspaces() {
    let first_model = model();
    let second_model = model();
    let second = Arc::new(
        SandboxAgent::<()>::new("sandbox")
            .with_model_instance(second_model.clone())
            .with_capabilities(vec![Arc::new(Shell::new().with_pty(false))])
            .with_default_manifest(Manifest::new().with_entry("b.txt", Entry::text_file("second"))),
    );
    let first = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(first_model.clone())
        .with_default_manifest(Manifest::new().with_entry("a.txt", Entry::text_file("first")))
        .with_sandbox_handoff(second);
    first_model.add_turn(vec![handoff_call("transfer_to_sandbox", "handoff")]);
    second_model.add_turn(vec![shell_call(
        "c1",
        "cat b.txt; cat a.txt 2>&1 | tail -1",
    )]);
    second_model.add_turn(vec![assistant_message("done")]);

    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
        .session_mode(SandboxSessionMode::PerAgent);
    let outcome = SandboxRunner::run_to_outcome(
        &first,
        "hello",
        &config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .expect("the run completes");

    assert_eq!(outcome.result.final_output_text().as_deref(), Some("done"));
    let stdout = outcome
        .result
        .new_items
        .iter()
        .find_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => {
                raw_item["output"][0]["stdout"].as_str().map(str::to_string)
            }
            _ => None,
        })
        .expect("the second agent's shell output");
    assert!(stdout.contains("second"), "{stdout}");
    assert!(!stdout.contains("first"), "{stdout}");
}

/// Python: `test_duplicate_name_sandbox_identity_map_uses_capability_and_manifest_config`:
/// the key of a duplicate depends on its configuration, not on where the handoff was declared.
#[test]
fn identity_keys_follow_configuration_not_declaration_order() {
    fn variant(readme: &str, text: &str) -> Arc<SandboxAgent<()>> {
        Arc::new(
            SandboxAgent::<()>::new("sandbox")
                .with_instructions(text)
                .with_default_manifest(
                    Manifest::new().with_entry("README.md", Entry::text_file(readme)),
                ),
        )
    }
    fn key_of(root: &SandboxAgent<()>, text: &str) -> String {
        root.graph_agent_names()
            .unwrap()
            .into_iter()
            .find(|key| instructions(&root.graph_agent(key).unwrap()) == text)
            .expect("the agent is in the graph")
    }

    let first_root = SandboxAgent::<()>::new("triage")
        .with_sandbox_handoff(variant("beta", "Beta."))
        .with_sandbox_handoff(variant("alpha", "Alpha."));
    let second_root = SandboxAgent::<()>::new("triage")
        .with_sandbox_handoff(variant("alpha", "Alpha."))
        .with_sandbox_handoff(variant("beta", "Beta."));

    assert_eq!(
        key_of(&first_root, "Alpha."),
        key_of(&second_root, "Alpha.")
    );
    assert_eq!(key_of(&first_root, "Beta."), key_of(&second_root, "Beta."));
    assert_ne!(key_of(&first_root, "Alpha."), key_of(&first_root, "Beta."));
}
