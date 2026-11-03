//! Ports the session-free half of `tests/sandbox/test_memory.py` from the Python SDK: rollout
//! payloads, the memory prompts, the layout and generation configuration, the phase-two input
//! selection and the read-side `Memory` capability. The runner-driven half is in
//! `py_sandbox_a_memory_2.rs`.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::items::FunctionCall;
use openai_agents_sdk::items::{
    InputItem, MessageOutput, ReasoningOutput, RunItem, RunItemKind, ToolApprovalItem,
};
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding, Memory};
use openai_agents_sdk::sandbox::memory::config::{
    MemoryGenerateConfig, MemoryLayoutConfig, MemoryReadConfig,
};
use openai_agents_sdk::sandbox::memory::manager::PHASE_ONE_ROLLOUT_TOKEN_LIMIT;
use openai_agents_sdk::sandbox::memory::prompts::{
    render_memory_consolidation_prompt, render_rollout_extraction_prompt,
};
use openai_agents_sdk::sandbox::memory::rollout_file_name_for_rollout_id;
use openai_agents_sdk::sandbox::memory::rollouts::{
    RolloutTerminalMetadata, TerminalState, build_rollout_payload,
    build_rollout_payload_with_interruptions, dump_rollout_json, render_phase_one_prompt,
};
use openai_agents_sdk::sandbox::memory::storage::{
    PhaseTwoInputSelection, PhaseTwoSelectionItem, SandboxMemoryStorage, updated_at_sort_key,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxSession, SandboxWorkspaceScope,
};
use serde_json::{Value, json};

fn completed() -> RolloutTerminalMetadata {
    RolloutTerminalMetadata::new(TerminalState::Completed, true)
}

fn empty_selection() -> PhaseTwoInputSelection {
    PhaseTwoInputSelection::default()
}

fn run_item(kind: RunItemKind) -> RunItem {
    RunItem::new("test", kind)
}

fn raw_memory_record(
    rollout_id: &str,
    updated_at: &str,
    rollout_summary_file: &str,
    raw_memory: &str,
) -> String {
    format!(
        "rollout_id: {rollout_id}\nupdated_at: {updated_at}\n\
         rollout_path: sessions/{rollout_id}.jsonl\nrollout_summary_file: {rollout_summary_file}\n\
         terminal_state: completed\n\n{}\n",
        raw_memory.trim_end()
    )
}

async fn local_session() -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(Manifest::new()))
        .await
        .expect("create session");
    session.start().await.expect("start session");
    (client, session)
}

// ----------------------------------------------------------------------------------------------
// Rollout file names
// ----------------------------------------------------------------------------------------------

#[test]
fn rollout_file_name_for_rollout_id_uses_file_safe_id_directly() {
    assert_eq!(
        rollout_file_name_for_rollout_id("chat-session.2026_04").expect("file name"),
        "chat-session.2026_04.jsonl"
    );
}

#[test]
fn rollout_file_name_for_rollout_id_rejects_path_like_ids() {
    let error = rollout_file_name_for_rollout_id("../chat-session").expect_err("path-like id");
    assert!(error.message.contains("file-safe ID"), "{}", error.message);
    assert_eq!(error.code, ErrorCode::SandboxConfigInvalid);
}

#[test]
fn rollout_file_name_for_rollout_id_rejects_empty_ids() {
    let error = rollout_file_name_for_rollout_id(" ").expect_err("empty id");
    assert!(error.message.contains("file-safe ID"), "{}", error.message);
}

// ----------------------------------------------------------------------------------------------
// Rollout payloads
// ----------------------------------------------------------------------------------------------

/// Developer and system messages, reasoning and compaction items never reach a rollout, and the
/// payload keys come out in Python's order.
#[test]
fn build_rollout_payload_filters_developer_and_noisy_items() {
    let reasoning: ReasoningOutput =
        serde_json::from_value(json!({"id": "rs_1", "summary": [], "type": "reasoning"}))
            .expect("reasoning");
    let message = MessageOutput::text("assistant");
    let input = vec![
        InputItem::from_value(json!({"role": "developer", "content": "debug"})),
        InputItem::from_value(json!({"role": "system", "content": "system"})),
        InputItem::from_value(json!({"role": "user", "content": "hello"})),
        InputItem::from_value(json!({"type": "reasoning", "summary": []})),
        InputItem::from_value(
            json!({"type": "compaction", "summary": "compact", "encrypted_content": "encrypted"}),
        ),
    ];
    let new_items = vec![
        run_item(RunItemKind::Reasoning {
            raw_item: reasoning,
        }),
        run_item(RunItemKind::Compaction {
            raw_item: json!({"type": "compaction", "summary": "compact", "encrypted_content": "encrypted"}),
        }),
        run_item(RunItemKind::MessageOutput {
            raw_item: message.clone(),
        }),
    ];

    let mut payload =
        build_rollout_payload(&input, &new_items, Some(&json!("done")), 0, &completed());

    let updated_at = payload
        .as_object_mut()
        .expect("object")
        .shift_remove("updated_at")
        .expect("updated_at");
    assert!(
        chrono::DateTime::parse_from_rfc3339(updated_at.as_str().expect("string")).is_ok(),
        "{updated_at}"
    );
    let keys: Vec<&str> = payload
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        vec![
            "input",
            "generated_items",
            "terminal_metadata",
            "final_output"
        ]
    );
    assert_eq!(
        payload["input"],
        json!([{"role": "user", "content": "hello"}])
    );
    assert_eq!(payload["generated_items"], json!([message.to_value()]));
    assert_eq!(payload["final_output"], json!("done"));
}

/// Programmatic tool calling: the program, the function call it issued, and both outputs are
/// kept, and the function call still points back at its program.
#[test]
fn build_rollout_payload_keeps_programmatic_tool_calling_items() {
    let new_items = vec![
        run_item(RunItemKind::ToolCall {
            raw_item: json!({
                "id": "program_item", "call_id": "call_prog_1",
                "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint",
                "type": "program",
            }),
        }),
        run_item(RunItemKind::ToolCall {
            raw_item: json!({
                "id": "function_item", "call_id": "call_fn_1", "name": "lookup_inventory",
                "arguments": "{\"sku\":\"A-1\"}",
                "caller": {"type": "program", "caller_id": "call_prog_1"},
                "type": "function_call",
            }),
        }),
        run_item(RunItemKind::ToolCallOutput {
            raw_item: json!({
                "type": "function_call_output", "call_id": "call_fn_1",
                "output": "{\"available_units\":42}",
            }),
            output: json!("42"),
            custom_data: None,
        }),
        run_item(RunItemKind::ToolCallOutput {
            raw_item: json!({
                "id": "program_output_item", "call_id": "call_prog_1",
                "result": "{\"sku\":\"A-1\",\"available_units\":42}",
                "status": "completed", "type": "program_output",
            }),
            output: json!("42"),
            custom_data: None,
        }),
    ];

    let payload = build_rollout_payload(
        &[InputItem::user("what is in stock?")],
        &new_items,
        Some(&json!("done")),
        0,
        &completed(),
    );

    let generated = payload["generated_items"].as_array().expect("array");
    let types: Vec<&str> = generated
        .iter()
        .map(|item| item["type"].as_str().expect("type"))
        .collect();
    assert_eq!(
        types,
        vec![
            "program",
            "function_call",
            "function_call_output",
            "program_output"
        ]
    );
    // The retained function call points back at the program that issued it, so the program it
    // names has to survive alongside it.
    assert_eq!(
        generated[1]["caller"],
        json!({"type": "program", "caller_id": "call_prog_1"})
    );
    assert_eq!(generated[0]["call_id"], "call_prog_1");
    assert_eq!(generated[0]["code"], "lookup_inventory(sku=\"A-1\")");
    assert_eq!(generated[3]["call_id"], "call_prog_1");
    assert_eq!(
        generated[3]["result"],
        "{\"sku\":\"A-1\",\"available_units\":42}"
    );
}

#[test]
fn build_rollout_payload_keeps_program_items_from_input() {
    let input = vec![
        InputItem::from_value(json!({
            "type": "program", "call_id": "call_prog_1",
            "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint",
        })),
        InputItem::from_value(json!({
            "type": "program_output", "call_id": "call_prog_1",
            "result": "{\"available_units\":42}", "status": "completed",
        })),
    ];
    let metadata = RolloutTerminalMetadata::new(TerminalState::Completed, false);

    let payload = build_rollout_payload(&input, &[], None, 0, &metadata);

    let types: Vec<&str> = payload["input"]
        .as_array()
        .expect("array")
        .iter()
        .map(|item| item["type"].as_str().expect("type"))
        .collect();
    assert_eq!(types, vec!["program", "program_output"]);
}

/// Hosted tool calls with no output half stay out, even though programs are in.
#[test]
fn build_rollout_payload_still_drops_hosted_items_outside_the_included_set() {
    let input = vec![
        InputItem::from_value(json!({"type": "file_search_call", "id": "fs_1", "queries": []})),
        InputItem::from_value(json!({"type": "image_generation_call", "id": "ig_1"})),
        InputItem::from_value(json!({"type": "program", "call_id": "call_prog_1", "code": "x()"})),
    ];
    let metadata = RolloutTerminalMetadata::new(TerminalState::Completed, false);

    let payload = build_rollout_payload(&input, &[], None, 0, &metadata);

    let types: Vec<&str> = payload["input"]
        .as_array()
        .expect("array")
        .iter()
        .map(|item| item["type"].as_str().expect("type"))
        .collect();
    assert_eq!(types, vec!["program"]);
}

/// Interruptions are recorded as the raw tool call dicts that were waiting.
#[test]
fn build_rollout_payload_serializes_model_interruptions_as_dicts() {
    let raw = FunctionCall {
        id: Some("fc_1".into()),
        call_id: "call_1".into(),
        name: "get_weather".into(),
        arguments: "{\"city\":\"Paris\"}".into(),
        status: None,
        extra: serde_json::Map::new(),
    };
    let approval = ToolApprovalItem {
        agent: "test".into(),
        raw_item: raw,
        tool_name: "get_weather".into(),
        tool_origin: None,
    };
    let metadata = RolloutTerminalMetadata::new(TerminalState::Interrupted, false);

    let payload = build_rollout_payload_with_interruptions(
        &[InputItem::user("hello")],
        &[],
        None,
        &[approval],
        &metadata,
    );

    let interruption = &payload["interruptions"][0];
    assert!(interruption.is_object(), "{interruption}");
    assert_eq!(interruption["type"], "function_call");
    assert_eq!(interruption["call_id"], "call_1");
    assert_eq!(interruption["name"], "get_weather");
    assert_eq!(interruption["arguments"], "{\"city\":\"Paris\"}");
    // The key is absent when nothing paused, and `final_output` is absent without one.
    let plain = build_rollout_payload(&[], &[], None, 0, &metadata);
    assert!(plain.get("interruptions").is_none());
    assert!(plain.get("final_output").is_none());
}

/// The terminal metadata serializes with every field, `null` included, as Python's
/// `model_dump(mode="json")` does.
#[test]
fn terminal_metadata_serializes_every_field() {
    let payload = build_rollout_payload(
        &[],
        &[],
        None,
        0,
        &RolloutTerminalMetadata::new(TerminalState::MaxTurnsExceeded, false),
    );
    assert_eq!(
        payload["terminal_metadata"],
        json!({
            "terminal_state": "max_turns_exceeded",
            "exception_type": null,
            "exception_message": null,
            "has_final_output": false,
        })
    );
}

/// A rollout far larger than the phase-one budget keeps its head and tail and says it was cut.
#[test]
fn render_phase_one_prompt_truncates_large_rollout_contents() {
    let payload = json!({
        "input": [{"role": "user", "content": format!("start{}middle{}end", "a".repeat(700_000), "z".repeat(700_000))}],
        "generated_items": [],
        "terminal_metadata": {"terminal_state": "completed", "has_final_output": false},
    });

    let prompt =
        render_phase_one_prompt(&dump_rollout_json(&payload), PHASE_ONE_ROLLOUT_TOKEN_LIMIT)
            .expect("prompt");

    assert!(prompt.contains("start"));
    assert!(prompt.contains("end"));
    assert!(!prompt.contains("middle"));
    assert!(prompt.contains("tokens truncated"));
    assert!(prompt.contains("rollout content omitted"));
    assert!(prompt.contains("Do not assume the rendered rollout below is complete"));
}

// ----------------------------------------------------------------------------------------------
// Prompts
// ----------------------------------------------------------------------------------------------

#[test]
fn render_memory_prompts_omit_extra_prompt_section_by_default() {
    let rollout_prompt = render_rollout_extraction_prompt(None);
    let consolidation_prompt =
        render_memory_consolidation_prompt("memory", &empty_selection(), None);

    assert!(!rollout_prompt.contains("{{ extra_prompt_section }}"));
    assert!(!consolidation_prompt.contains("{{ extra_prompt_section }}"));
    assert!(!rollout_prompt.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));
    assert!(!consolidation_prompt.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));
}

#[test]
fn render_memory_prompts_include_extra_prompt_section() {
    let rollout_prompt = render_rollout_extraction_prompt(Some("Focus on user preferences."));
    let consolidation_prompt = render_memory_consolidation_prompt(
        "memory",
        &empty_selection(),
        Some("Focus on user preferences."),
    );

    assert!(rollout_prompt.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));
    assert!(rollout_prompt.contains("Focus on user preferences."));
    assert!(consolidation_prompt.contains("DEVELOPER-SPECIFIC EXTRA GUIDANCE"));
    assert!(consolidation_prompt.contains("Focus on user preferences."));
}

#[test]
fn render_memory_consolidation_prompt_lists_removed_rollouts() {
    let selection = PhaseTwoInputSelection {
        selected: Vec::new(),
        retained_rollout_ids: Default::default(),
        removed: vec![PhaseTwoSelectionItem {
            rollout_id: "old-rollout".into(),
            updated_at: String::new(),
            rollout_path: "sessions/old-rollout.jsonl".into(),
            rollout_summary_file: "memories/rollout_summaries/old.md".into(),
            terminal_state: "completed".into(),
        }],
    };

    let prompt = render_memory_consolidation_prompt("memory", &selection, None);

    assert!(prompt.contains("- removed from the last successful Phase 2 run: 1"));
    assert!(prompt.contains("rollout_id=old-rollout"));
    assert!(prompt.contains("updated_at=unknown"));
}

// ----------------------------------------------------------------------------------------------
// Phase-two selection
// ----------------------------------------------------------------------------------------------

/// Adapted: Python hands `_updated_at_sort_key` the raw memory text; Rust ranks the already
/// parsed `updated_at` value, so a missing header is the empty value.
#[test]
fn updated_at_sort_key_places_unknown_timestamps_last() {
    assert!(updated_at_sort_key("2025-03-01T00:00:00Z") > updated_at_sort_key("unknown"));
    assert_eq!(updated_at_sort_key("unknown"), updated_at_sort_key(""));
    assert_eq!(updated_at_sort_key("unknown"), updated_at_sort_key("  "));
}

#[tokio::test]
async fn phase_two_selection_tracks_added_retained_and_removed_rollouts() {
    let (client, session) = local_session().await;
    let storage = SandboxMemoryStorage::new(MemoryLayoutConfig::new());
    storage
        .ensure_layout(session.as_ref())
        .await
        .expect("layout");
    let old_item = PhaseTwoSelectionItem {
        rollout_id: "old-rollout".into(),
        updated_at: "2025-03-01T00:00:00Z".into(),
        rollout_path: "sessions/old-rollout.jsonl".into(),
        rollout_summary_file: "rollout_summaries/old-rollout.md".into(),
        terminal_state: "completed".into(),
    };
    storage
        .write_text(
            session.as_ref(),
            &format!("{}/old-rollout.md", storage.raw_memories_dir()),
            &raw_memory_record(
                &old_item.rollout_id,
                &old_item.updated_at,
                &old_item.rollout_summary_file,
                "old raw",
            ),
        )
        .await
        .expect("write old");
    storage
        .write_text(
            session.as_ref(),
            &format!("{}/new-rollout.md", storage.raw_memories_dir()),
            &raw_memory_record(
                "new-rollout",
                "2025-03-02T00:00:00Z",
                "rollout_summaries/new-rollout.md",
                "new raw",
            ),
        )
        .await
        .expect("write new");
    storage
        .write_phase_two_selection(session.as_ref(), std::slice::from_ref(&old_item))
        .await
        .expect("write selection");

    let selection = storage
        .build_phase_two_input_selection(session.as_ref(), 1)
        .await
        .expect("selection");

    let selected: Vec<&str> = selection
        .selected
        .iter()
        .map(|i| i.rollout_id.as_str())
        .collect();
    assert_eq!(selected, vec!["new-rollout"]);
    assert!(selection.retained_rollout_ids.is_empty());
    let removed: Vec<&str> = selection
        .removed
        .iter()
        .map(|i| i.rollout_id.as_str())
        .collect();
    assert_eq!(removed, vec!["old-rollout"]);
    client.delete(session).await.expect("delete");
}

/// A rollout with an unknown timestamp ranks below every dated one, so a capped selection keeps
/// the dated rollout.
#[tokio::test]
async fn capped_selection_prefers_dated_rollouts_over_unknown_ones() {
    let (client, session) = local_session().await;
    let storage = SandboxMemoryStorage::new(MemoryLayoutConfig::new());
    storage
        .ensure_layout(session.as_ref())
        .await
        .expect("layout");
    for (id, updated_at) in [
        ("zz-unknown", "unknown"),
        ("aa-dated", "2025-03-01T00:00:00Z"),
    ] {
        storage
            .write_text(
                session.as_ref(),
                &format!("{}/{id}.md", storage.raw_memories_dir()),
                &raw_memory_record(id, updated_at, &format!("rollout_summaries/{id}.md"), "raw"),
            )
            .await
            .expect("write");
    }

    let selection = storage
        .build_phase_two_input_selection(session.as_ref(), 1)
        .await
        .expect("selection");

    assert_eq!(selection.selected.len(), 1);
    assert_eq!(selection.selected[0].rollout_id, "aa-dated");
    client.delete(session).await.expect("delete");
}

// ----------------------------------------------------------------------------------------------
// Memory capability: configuration
// ----------------------------------------------------------------------------------------------

#[test]
fn memory_capability_rejects_invalid_memories_dir() {
    for (memories_dir, expected) in [
        ("/memory", "memories_dir must be relative"),
        ("../memory", "memories_dir must not escape root"),
        ("", "memories_dir must be non-empty"),
        (".", "memories_dir must be non-empty"),
    ] {
        let memory = Memory::new()
            .with_layout(MemoryLayoutConfig::new().with_memories_dir(memories_dir))
            .with_generate(None);
        let error = memory.validate().expect_err(memories_dir);
        assert!(
            error.to_string().contains(expected),
            "{memories_dir}: {error}"
        );
        let layout_error = MemoryLayoutConfig::new()
            .with_memories_dir(memories_dir)
            .validate()
            .expect_err(memories_dir);
        assert_eq!(layout_error.code, ErrorCode::SandboxConfigInvalid);
    }
}

#[test]
fn memory_capability_rejects_invalid_sessions_dir() {
    for (sessions_dir, expected) in [
        ("/sessions", "sessions_dir must be relative"),
        ("../sessions", "sessions_dir must not escape root"),
        ("", "sessions_dir must be non-empty"),
        (".", "sessions_dir must be non-empty"),
    ] {
        let memory = Memory::new()
            .with_layout(MemoryLayoutConfig::new().with_sessions_dir(sessions_dir))
            .with_generate(None);
        let error = memory.validate().expect_err(sessions_dir);
        assert!(
            error.to_string().contains(expected),
            "{sessions_dir}: {error}"
        );
    }
}

#[test]
fn memory_capability_requires_read_or_generate() {
    let error = Memory::new()
        .with_read(None)
        .with_generate(None)
        .validate()
        .expect_err("neither read nor generate");
    assert!(
        error
            .to_string()
            .contains("Memory requires at least one of `read` or `generate`"),
        "{error}"
    );
}

#[test]
fn memory_generate_config_rejects_non_positive_recent_rollout_limit() {
    let error = MemoryGenerateConfig::new()
        .with_max_raw_memories_for_consolidation(0)
        .validate()
        .expect_err("zero limit");
    assert!(
        error.message.contains(
            "MemoryGenerateConfig.max_raw_memories_for_consolidation must be greater than 0"
        ),
        "{}",
        error.message
    );
}

#[test]
fn memory_generate_config_rejects_too_many_raw_memories() {
    let error = MemoryGenerateConfig::new()
        .with_max_raw_memories_for_consolidation(4097)
        .validate()
        .expect_err("limit above the cap");
    assert!(
        error.message.contains(
            "MemoryGenerateConfig.max_raw_memories_for_consolidation must be less than or equal to 4096"
        ),
        "{}",
        error.message
    );
    MemoryGenerateConfig::new()
        .with_max_raw_memories_for_consolidation(4096)
        .validate()
        .expect("the cap itself is allowed");
}

#[test]
fn memory_layout_config_defaults_match_codex_names() {
    let config = MemoryLayoutConfig::new();
    assert_eq!(config.memories_dir, "memories");
    assert_eq!(config.sessions_dir, "sessions");
}

#[test]
fn memory_generate_config_accepts_renamed_limit_field_and_keeps_default_turn_limit() {
    let config = MemoryGenerateConfig::new().with_max_raw_memories_for_consolidation(123);
    assert_eq!(config.max_raw_memories_for_consolidation, 123);
    assert_eq!(config.phase_two_max_turns, 500);
    assert!(MemoryReadConfig::new().live_update);
}

#[test]
fn memory_generate_config_preserves_typed_and_disabled_model_settings() {
    use openai_agents_sdk::model_settings::ModelSettings;

    let phase_one = ModelSettings {
        temperature: Some(0.1),
        ..Default::default()
    };
    let phase_two = ModelSettings {
        temperature: Some(0.2),
        ..Default::default()
    };
    let config = MemoryGenerateConfig::new()
        .with_phase_one_model_settings(Some(phase_one))
        .with_phase_two_model_settings(Some(phase_two));
    assert_eq!(
        config
            .phase_one_model_settings
            .as_ref()
            .and_then(|s| s.temperature),
        Some(0.1)
    );
    assert_eq!(
        config
            .phase_two_model_settings
            .as_ref()
            .and_then(|s| s.temperature),
        Some(0.2)
    );

    let disabled = MemoryGenerateConfig::new()
        .with_phase_one_model_settings(None)
        .with_phase_two_model_settings(None);
    assert!(disabled.phase_one_model_settings.is_none());
    assert!(disabled.phase_two_model_settings.is_none());
}

// ----------------------------------------------------------------------------------------------
// Memory capability: instructions
// ----------------------------------------------------------------------------------------------

async fn instructions_for(
    memory: &Memory,
    session: &Arc<dyn SandboxSession>,
    scope: Option<SandboxWorkspaceScope>,
) -> Option<String> {
    let mut binding = CapabilityBinding::new(session.clone());
    if let Some(scope) = scope {
        binding = binding.with_scope(scope);
    }
    Capability::<()>::bound_instructions(memory, &binding, session.manifest())
        .await
        .expect("instructions")
}

#[tokio::test]
async fn memory_capability_returns_none_without_memory_summary() {
    let (client, session) = local_session().await;
    let memory = Memory::new().with_generate(None);

    assert!(instructions_for(&memory, &session, None).await.is_none());

    session.mkdir("memories", true).await.expect("mkdir");
    session
        .write_file("memories/memory_summary.md", b"")
        .await
        .expect("write empty summary");

    assert!(instructions_for(&memory, &session, None).await.is_none());
    client.delete(session).await.expect("delete");
}

/// A summary over the token budget is cut in the middle and says so.
#[tokio::test]
async fn memory_capability_injects_truncated_memory_summary() {
    let (client, session) = local_session().await;
    let memory = Memory::new().with_generate(None);
    session.mkdir("memories", true).await.expect("mkdir");
    session
        .write_file(
            "memories/memory_summary.md",
            "abcdefghijklmnopqrstuvwxyz".repeat(5000).as_bytes(),
        )
        .await
        .expect("write summary");

    let instructions = instructions_for(&memory, &session, None)
        .await
        .expect("instructions");

    assert!(
        instructions
            .contains("memories/memory_summary.md (already provided below; do NOT open again)")
    );
    assert!(instructions.contains("MEMORY_SUMMARY BEGINS"));
    assert!(instructions.contains("tokens truncated"));
    client.delete(session).await.expect("delete");
}

#[tokio::test]
async fn memory_capability_live_update_instructions() {
    let (client, session) = local_session().await;
    let memory = Memory::new().with_generate(None);
    session.mkdir("memories", true).await.expect("mkdir");
    session
        .write_file("memories/memory_summary.md", b"summary entry")
        .await
        .expect("write summary");

    let instructions = instructions_for(&memory, &session, None)
        .await
        .expect("instructions");

    assert!(instructions.contains("Memory is writable."));
    assert!(instructions.contains("memories/MEMORY.md"));
    assert!(instructions.contains("same turn"));
    assert!(!instructions.contains("Never update memories."));

    let read_only = Memory::new()
        .with_generate(None)
        .with_read(Some(MemoryReadConfig::new().with_live_update(false)));
    let read_only_instructions = instructions_for(&read_only, &session, None)
        .await
        .expect("instructions");
    assert!(read_only_instructions.contains("Never update memories."));
    assert!(!read_only_instructions.contains("Memory is writable."));
    client.delete(session).await.expect("delete");
}

/// With a run cwd the model resolves relative paths from the cwd, so the paths the capability
/// owns are rendered absolute.
#[tokio::test]
async fn memory_capability_renders_session_owned_paths_as_absolute_with_run_cwd() {
    let (client, session) = local_session().await;
    let memory = Memory::new().with_generate(None);
    session.mkdir("memories", true).await.expect("mkdir");
    session
        .write_file("memories/memory_summary.md", b"summary entry")
        .await
        .expect("write summary");
    let scope = SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).expect("scope");

    let instructions = instructions_for(&memory, &session, Some(scope))
        .await
        .expect("instructions");

    let root = session.manifest().root().trim_end_matches('/').to_string();
    assert!(
        instructions.contains(&format!(
            "{root}/memories/memory_summary.md (already provided below; do NOT open again)"
        )),
        "{instructions}"
    );
    assert!(instructions.contains(&format!("{root}/memories/MEMORY.md")));
    assert!(instructions.contains("summary entry"));
    client.delete(session).await.expect("delete");
}

/// Without a run cwd the layout is rendered the way it was spelled.
#[tokio::test]
async fn memory_capability_preserves_layout_spelling_without_run_cwd() {
    for memories_dir in ["team//memory", r"team\memory"] {
        let (client, session) = local_session().await;
        let memory = Memory::new()
            .with_layout(MemoryLayoutConfig::new().with_memories_dir(memories_dir))
            .with_generate(None);
        session
            .mkdir(memories_dir, true)
            .await
            .unwrap_or_else(|e| panic!("{memories_dir}: mkdir: {e}"));
        session
            .write_file(
                &format!("{memories_dir}/memory_summary.md"),
                b"summary entry",
            )
            .await
            .unwrap_or_else(|e| panic!("{memories_dir}: write: {e}"));

        let instructions = instructions_for(&memory, &session, None)
            .await
            .unwrap_or_else(|| panic!("{memories_dir}: no instructions"));

        assert!(
            instructions.contains(&format!("{memories_dir}/memory_summary.md")),
            "{memories_dir}"
        );
        assert!(
            instructions.contains(&format!("{memories_dir}/MEMORY.md")),
            "{memories_dir}"
        );
        client.delete(session).await.expect("delete");
    }
}

/// A layout spelled with a separator Python treats as part of a POSIX name still resolves under
/// the workspace root when the run has a cwd.
#[tokio::test]
async fn memory_capability_uses_typed_layout_path_with_run_cwd() {
    let memories_dir = r"team\memory";
    let (client, session) = local_session().await;
    let memory = Memory::new()
        .with_layout(MemoryLayoutConfig::new().with_memories_dir(memories_dir))
        .with_generate(None);
    session.mkdir(memories_dir, true).await.expect("mkdir");
    session
        .write_file(
            &format!("{memories_dir}/memory_summary.md"),
            b"summary entry",
        )
        .await
        .expect("write summary");
    let scope = SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).expect("scope");

    let instructions = instructions_for(&memory, &session, Some(scope))
        .await
        .expect("instructions");

    let root = session.manifest().root().trim_end_matches('/').to_string();
    assert!(
        instructions.contains(&format!("{root}/{memories_dir}/memory_summary.md")),
        "{instructions}"
    );
    assert!(instructions.contains(&format!("{root}/{memories_dir}/MEMORY.md")));
    client.delete(session).await.expect("delete");
}

/// One manager per prepared session: two layouts that both generate are two managers, and a
/// manager is rejected when its layout is invalid.
#[test]
fn memory_build_manager_validates_the_layout() {
    assert!(Memory::new().build_manager().expect("manager").is_some());
    let invalid = Memory::new().with_layout(MemoryLayoutConfig::new().with_memories_dir("../x"));
    assert!(invalid.build_manager().is_err());
}

#[allow(dead_code)]
fn unused_value_guard(_: Value) {}
