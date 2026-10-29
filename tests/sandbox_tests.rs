//! Sandbox agents: manifest materialization, capability tools bound to a live session, and
//! end-to-end runs through `Runner` with a scripted model.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding, Shell};
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::sandboxes::docker::{
    DockerSandboxClient, DockerSandboxClientOptions,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{
    ErrorCode, ExecRequest, Manifest, SandboxAgent, SandboxClient, SandboxRunConfig, SandboxRunner,
    SandboxSession, SandboxWorkspaceScope,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::{RunOptions, RunResult, Runner};
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

fn item(v: Value) -> OutputItem {
    OutputItem::from_value(v)
}

fn shell_call(call_id: &str, commands: &[&str]) -> OutputItem {
    item(json!({
        "type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
        "status": "completed", "action": {"commands": commands}
    }))
}

fn apply_patch_call(call_id: &str, operation: Value) -> OutputItem {
    item(json!({
        "type": "apply_patch_call", "call_id": call_id, "status": "completed",
        "operation": operation
    }))
}

fn view_image_call(call_id: &str, path: &str) -> OutputItem {
    item(json!({
        "type": "function_call", "call_id": call_id, "name": "view_image",
        "arguments": json!({"path": path}).to_string()
    }))
}

fn tool_outputs(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

async fn local_session(
    manifest: Option<Manifest>,
) -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(manifest).await.expect("create session");
    session.start().await.expect("start session");
    (client, session)
}

// -------------------------------------------------------------------------------------------
// Manifest materialization
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn materializes_files_dirs_and_local_copies() {
    let host = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(host.path().join("src")).unwrap();
    std::fs::write(host.path().join("README.md"), "readme").unwrap();
    std::fs::write(host.path().join("src/main.rs"), "fn main() {}").unwrap();
    std::fs::write(host.path().join("outside.txt"), "nope").unwrap();

    let manifest = Manifest::new()
        .with_entry("notes.txt", Entry::text_file("inline"))
        .with_entry(
            "repo",
            Entry::local_dir(host.path()).with_description("staged repository"),
        )
        .with_entry(
            "nested",
            Entry::dir().with_child(
                "deep",
                Entry::dir().with_child("leaf.txt", Entry::file(b"leaf".to_vec())),
            ),
        )
        .with_entry(
            "single.md",
            Entry::local_file(host.path().join("README.md")),
        );

    let (client, session) = local_session(Some(manifest)).await;

    let read = |path: &'static str| {
        let session = session.clone();
        async move { String::from_utf8(session.read_file(path).await.unwrap()).unwrap() }
    };
    assert_eq!(read("notes.txt").await, "inline");
    assert_eq!(read("repo/README.md").await, "readme");
    assert_eq!(read("repo/src/main.rs").await, "fn main() {}");
    assert_eq!(read("nested/deep/leaf.txt").await, "leaf");
    assert_eq!(read("single.md").await, "readme");

    let listing = session.ls("repo").await.unwrap();
    let mut names: Vec<&str> = listing.iter().map(|e| e.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["README.md", "outside.txt", "src"]);

    // The rendered layout reaches the model through the instructions.
    let described = session.describe();
    assert!(described.contains("repo/ - staged repository"));

    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn materializes_into_a_caller_supplied_workspace_root() {
    // A root the caller chose is used as-is, is not canonicalized away, and survives `delete`.
    // On macOS such a path resolves through a symlink (`/var` -> `/private/var`), which every
    // workspace prefix check has to tolerate.
    let host = tempfile::tempdir().unwrap();
    let root = host.path().join("workspace");
    std::fs::create_dir(&root).unwrap();

    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new()
                .with_root(root.to_string_lossy().to_string())
                .with_entry(
                    "repo",
                    Entry::dir().with_child("a.txt", Entry::text_file("kept")),
                ),
        ))
        .await
        .unwrap();
    session.start().await.unwrap();

    assert_eq!(
        String::from_utf8(session.read_file("repo/a.txt").await.unwrap()).unwrap(),
        "kept"
    );
    assert!(root.join("repo/a.txt").exists());
    assert!(session.read_file("../escape.txt").await.is_err());

    client.delete(session).await.unwrap();
    assert!(
        root.exists(),
        "a caller-owned workspace root must not be removed"
    );
}

#[tokio::test]
async fn rejects_manifest_entries_that_escape_the_workspace() {
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new().with_entry("../escape.txt", Entry::text_file("x")),
        ))
        .await
        .unwrap();
    let err = session.start().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidManifestPath);
    client.delete(session).await.unwrap();
}

// -------------------------------------------------------------------------------------------
// Capability tools bound to a live session
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn shell_capability_runs_commands_in_the_session() {
    let (client, session) = local_session(Some(
        Manifest::new().with_entry("a.txt", Entry::text_file("hello")),
    ))
    .await;
    let tools =
        Capability::<()>::tools(&Shell::new(), &CapabilityBinding::new(session.clone())).unwrap();
    assert_eq!(tools.len(), 1);

    let model = setup();
    let agent = openai_agents_sdk::Agent::<()>::new("Dev")
        .with_tools(tools)
        .with_model_instance(model.clone());
    model.add_turn(vec![shell_call("c1", &["cat a.txt", "exit 3"])]);
    model.add_turn(vec![assistant_message("done")]);

    let result = Runner::run(&agent, "read it").await.unwrap();
    let out = &tool_outputs(&result)[0];
    assert_eq!(out["type"], "shell_call_output");
    assert_eq!(out["output"][0]["stdout"], "hello");
    assert_eq!(out["output"][0]["outcome"]["exit_code"], 0);
    assert_eq!(out["output"][1]["outcome"]["exit_code"], 3);

    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn shell_capability_reports_timeouts_without_failing_the_call() {
    let (client, session) = local_session(None).await;
    let tools =
        Capability::<()>::tools(&Shell::new(), &CapabilityBinding::new(session.clone())).unwrap();
    let model = setup();
    let agent = openai_agents_sdk::Agent::<()>::new("Dev")
        .with_tools(tools)
        .with_model_instance(model.clone());
    model.add_turn(vec![item(json!({
        "type": "shell_call", "call_id": "slow", "status": "completed",
        "action": {"commands": ["sleep 5"], "timeout_ms": 50}
    }))]);
    model.add_turn(vec![assistant_message("timed out")]);

    let result = Runner::run(&agent, "wait").await.unwrap();
    let out = &tool_outputs(&result)[0];
    assert_eq!(out["status"], "completed");
    assert_eq!(out["output"][0]["outcome"]["type"], "timeout");
    assert_eq!(result.final_output_text().as_deref(), Some("timed out"));

    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn shell_capability_anchors_commands_at_the_run_working_directory() {
    let (client, session) = local_session(Some(Manifest::new().with_entry(
        "repo",
        Entry::dir().with_child("f.txt", Entry::text_file("in repo")),
    )))
    .await;
    let binding = CapabilityBinding::new(session.clone())
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("repo")).unwrap());
    let tools = Capability::<()>::tools(&Shell::new(), &binding).unwrap();

    let model = setup();
    let agent = openai_agents_sdk::Agent::<()>::new("Dev")
        .with_tools(tools)
        .with_model_instance(model.clone());
    model.add_turn(vec![shell_call("c1", &["cat f.txt"])]);
    model.add_turn(vec![assistant_message("ok")]);

    let result = Runner::run(&agent, "read").await.unwrap();
    assert_eq!(tool_outputs(&result)[0]["output"][0]["stdout"], "in repo");

    client.delete(session).await.unwrap();
}

// -------------------------------------------------------------------------------------------
// End-to-end runs through SandboxRunner
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn runs_a_sandbox_agent_end_to_end() {
    let model = setup();
    let agent = SandboxAgent::<()>::new("Sandbox engineer")
        .with_instructions("Read `repo/task.md` before editing files.")
        .with_model_instance(model.clone())
        .with_default_manifest(
            Manifest::new().with_entry(
                "repo",
                Entry::dir()
                    .with_description("staged repository")
                    .with_child("task.md", Entry::text_file("alpha\nbeta\n"))
                    .with_child("logo.png", Entry::file(vec![0x89, 0x50, 0x4e, 0x47])),
            ),
        );

    model.add_turn(vec![shell_call("c_shell", &["cat repo/task.md"])]);
    model.add_turn(vec![apply_patch_call(
        "c_patch",
        json!({
            "type": "update_file",
            "path": "repo/task.md",
            "diff": "@@\n alpha\n-beta\n+gamma\n"
        }),
    )]);
    model.add_turn(vec![view_image_call("c_image", "repo/logo.png")]);
    model.add_turn(vec![assistant_message("Replaced beta with gamma.")]);

    let client = Arc::new(UnixLocalSandboxClient::new());
    let config = SandboxRunConfig::with_client(client);
    let result = SandboxRunner::run(&agent, "Fix repo/task.md", &config)
        .await
        .unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("Replaced beta with gamma.")
    );
    let outputs = tool_outputs(&result);
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0]["output"][0]["stdout"], "alpha\nbeta\n");
    assert_eq!(outputs[1]["type"], "apply_patch_call_output");
    assert_eq!(outputs[1]["status"], "completed");
    assert!(
        outputs[2]
            .to_string()
            .contains("data:image/png;base64,iVBORw==")
    );

    // The sandbox prompt, the agent instructions and the workspace layout all reach the model.
    let instructions = model.calls()[0].system_instructions.clone().unwrap();
    assert!(instructions.contains("# Agent instructions"));
    assert!(instructions.contains("Read `repo/task.md` before editing files."));
    assert!(instructions.contains("# Sandbox capability instructions"));
    assert!(instructions.contains("# Filesystem"));
    assert!(instructions.contains("repo/ - staged repository"));
}

#[tokio::test]
async fn apply_patch_edits_are_visible_to_later_shell_calls() {
    let model = setup();
    let agent = SandboxAgent::<()>::new("Editor")
        .with_model_instance(model.clone())
        .with_default_manifest(Manifest::new().with_entry("notes.md", Entry::text_file("one\n")));

    model.add_turn(vec![apply_patch_call(
        "c_create",
        json!({"type": "create_file", "path": "added.txt", "diff": "+created\n"}),
    )]);
    model.add_turn(vec![shell_call("c_shell", &["cat added.txt"])]);
    model.add_turn(vec![apply_patch_call(
        "c_delete",
        json!({"type": "delete_file", "path": "notes.md"}),
    )]);
    model.add_turn(vec![shell_call("c_check", &["test -f notes.md; echo $?"])]);
    model.add_turn(vec![assistant_message("done")]);

    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));
    let result = SandboxRunner::run_with_options(
        &agent,
        "edit",
        &config,
        RunOptions::default().max_turns(Some(10)),
    )
    .await
    .unwrap();

    let outputs = tool_outputs(&result);
    assert_eq!(outputs[1]["output"][0]["stdout"], "created");
    assert_eq!(outputs[3]["output"][0]["stdout"], "1\n");
}

#[tokio::test]
async fn reports_deleting_a_file_that_does_not_exist() {
    let model = setup();
    let agent = SandboxAgent::<()>::new("Editor").with_model_instance(model.clone());
    model.add_turn(vec![apply_patch_call(
        "c_missing",
        json!({"type": "delete_file", "path": "absent.txt"}),
    )]);
    model.add_turn(vec![assistant_message("nothing to delete")]);

    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));
    let result = SandboxRunner::run(&agent, "delete it", &config)
        .await
        .unwrap();

    let out = &tool_outputs(&result)[0];
    assert_eq!(out["status"], "failed");
    assert!(out.to_string().contains("apply_patch_file_not_found"));
}

#[tokio::test]
async fn rejects_patch_and_image_paths_that_escape_the_workspace() {
    let model = setup();
    let agent = SandboxAgent::<()>::new("Editor").with_model_instance(model.clone());

    model.add_turn(vec![apply_patch_call(
        "c_escape",
        json!({"type": "create_file", "path": "../escape.txt", "diff": "+bad\n"}),
    )]);
    model.add_turn(vec![view_image_call("c_image", "../../etc/hosts")]);
    model.add_turn(vec![assistant_message("blocked")]);

    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));
    let result = SandboxRunner::run(&agent, "try to escape", &config)
        .await
        .unwrap();

    let outputs = tool_outputs(&result);
    assert_eq!(outputs[0]["status"], "failed");
    assert!(outputs[0].to_string().contains("apply_patch_invalid_path"));
    assert!(outputs[1].to_string().contains("Invalid path"));
    assert_eq!(result.final_output_text().as_deref(), Some("blocked"));
}

#[tokio::test]
async fn reuses_a_caller_owned_session_without_tearing_it_down() {
    let (client, session) = local_session(Some(
        Manifest::new().with_entry("state.txt", Entry::text_file("kept")),
    ))
    .await;
    let model = setup();
    let agent = SandboxAgent::<()>::new("Engineer").with_model_instance(model.clone());

    model.add_turn(vec![shell_call("c1", &["cat state.txt"])]);
    model.add_turn(vec![assistant_message("read it")]);

    let config = SandboxRunConfig::with_session(session.clone());
    let result = SandboxRunner::run(&agent, "read state", &config)
        .await
        .unwrap();
    assert_eq!(tool_outputs(&result)[0]["output"][0]["stdout"], "kept");

    // The session survives the run, so a second run can build on the same workspace.
    assert!(session.running().await.unwrap());
    assert_eq!(
        String::from_utf8(session.read_file("state.txt").await.unwrap()).unwrap(),
        "kept"
    );

    client.delete(session).await.unwrap();
}

#[tokio::test]
async fn requires_a_client_or_a_session() {
    let model = setup();
    let agent = SandboxAgent::<()>::new("Engineer").with_model_instance(model);
    let err = SandboxRunner::run(&agent, "hi", &SandboxRunConfig::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("SandboxRunConfig::client"));
}

#[tokio::test]
async fn rejects_a_missing_working_directory() {
    let model = setup();
    let agent = SandboxAgent::<()>::new("Engineer").with_model_instance(model);
    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()))
        .cwd("does/not/exist");
    let err = SandboxRunner::run(&agent, "hi", &config).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("does not exist or is not accessible")
    );
}

#[tokio::test]
async fn streams_a_sandbox_run_and_closes_the_session_afterwards() {
    use futures::StreamExt;

    let model = setup();
    let agent = SandboxAgent::<()>::new("Engineer")
        .with_model_instance(model.clone())
        .with_default_manifest(Manifest::new().with_entry("f.txt", Entry::text_file("streamed")));
    model.add_turn(vec![shell_call("c1", &["cat f.txt"])]);
    model.add_turn(vec![assistant_message("all good")]);

    let config = SandboxRunConfig::with_client(Arc::new(UnixLocalSandboxClient::new()));
    let (mut streaming, guard) =
        SandboxRunner::run_streamed_with_options(&agent, "read it", &config, RunOptions::default())
            .await
            .unwrap();
    let workspace = std::path::PathBuf::from(guard.session().workspace_root().to_string());
    assert!(guard.owns_session());

    {
        let stream = streaming.stream_events();
        futures::pin_mut!(stream);
        while stream.next().await.is_some() {}
    }
    assert_eq!(streaming.final_output_text().as_deref(), Some("all good"));

    guard.close().await.unwrap();
    assert!(!workspace.exists());
}

// -------------------------------------------------------------------------------------------
// Docker backend (skipped unless a daemon and the image are already available locally)
// -------------------------------------------------------------------------------------------

/// The image used by the Docker test. It is never pulled: the test is skipped when it is
/// missing, so the default test run needs no network access.
const DOCKER_TEST_IMAGE: &str = "alpine:latest";

#[tokio::test]
async fn docker_session_runs_commands_and_materializes_the_manifest() {
    if !DockerSandboxClient::is_available().await {
        eprintln!("skipping: no reachable Docker daemon");
        return;
    }
    if !DockerSandboxClient::image_exists(DOCKER_TEST_IMAGE).await {
        eprintln!("skipping: image `{DOCKER_TEST_IMAGE}` is not present locally");
        return;
    }

    let client = DockerSandboxClient::with_options(
        DockerSandboxClientOptions::new(DOCKER_TEST_IMAGE).with_network_disabled(true),
    );
    let manifest = Manifest::new()
        .with_entry(
            "repo",
            Entry::dir().with_child("task.md", Entry::text_file("docker task")),
        )
        .with_env("SANDBOX_MARKER", "container");
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();

    assert!(session.running().await.unwrap());
    assert_eq!(
        String::from_utf8(session.read_file("repo/task.md").await.unwrap()).unwrap(),
        "docker task"
    );
    let result = session
        .exec(ExecRequest::shell("echo $SANDBOX_MARKER"))
        .await
        .unwrap();
    assert_eq!(result.stdout_text().trim(), "container");
    assert!(session.normalize_path("../etc/passwd").is_err());

    client.delete(session).await.unwrap();
}
