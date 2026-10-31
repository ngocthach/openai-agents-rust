//! Ports of the Python sandbox suite's small unit files:
//! `tests/sandbox/test_types.py`, `test_errors.py`, `test_compaction.py`,
//! `test_remote_mount_policy.py`, `test_exposed_ports.py`, `test_retry.py` and
//! `test_posix_tool_paths.py`.
#![cfg(feature = "sandbox")]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use openai_agents_sdk::sandbox::capabilities::compaction::context_window_for_model;
use openai_agents_sdk::sandbox::entries::Entry;
use openai_agents_sdk::sandbox::mount_entries::{ConfiguredMountStrategy, S3Mount};
use openai_agents_sdk::sandbox::mounts::Mount;
use openai_agents_sdk::sandbox::remote_mount_policy::build_remote_mount_policy_instructions;
use openai_agents_sdk::sandbox::retry::{BackoffStrategy, RetryPolicy, retry_async};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::types::{
    ExposedPortEndpoint, ExposedPortUnavailable, Group, Permissions, User, exposed_port_unavailable,
};
use openai_agents_sdk::sandbox::{
    ErrorCode, Manifest, SandboxClient, SandboxError, SandboxOp, SandboxSession,
};

// ----------------------------------------------------------------------------------------------
// tests/sandbox/test_types.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_permissions_is_hashable`.
#[test]
fn permissions_is_hashable() {
    let perms = Permissions::from_mode(0o755);
    let other = Permissions::from_mode(0o755);
    let different = Permissions::from_mode(0o644);

    // Equal values hash equally under one hasher; a different value is a different key.
    use std::hash::BuildHasher as _;
    let state = std::hash::RandomState::new();
    assert_eq!(state.hash_one(perms), state.hash_one(other));
    assert_ne!(state.hash_one(perms), state.hash_one(different));

    let set: HashSet<Permissions> = [perms, other, different].into_iter().collect();
    assert_eq!(set, HashSet::from([perms, different]));

    let map: HashMap<Permissions, &str> = HashMap::from([(perms, "value")]);
    assert_eq!(map[&other], "value");
}

/// Python: `test_user_and_group_remain_hashable`.
#[test]
fn user_and_group_remain_hashable() {
    use std::hash::BuildHasher as _;
    let state = std::hash::RandomState::new();
    assert_eq!(
        state.hash_one(User::new("alice")),
        state.hash_one(User::new("alice"))
    );
    assert_eq!(
        state.hash_one(Group::new("admin")),
        state.hash_one(Group::new("admin"))
    );
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/test_errors.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_sandbox_error_retryable_can_be_set_explicitly`.
#[test]
fn sandbox_error_retryable_can_be_set_explicitly() {
    let error = SandboxError::new(ErrorCode::ExecTransportError, "backend is unavailable")
        .with_op(SandboxOp::Exec)
        .with_retryable(true);

    assert_eq!(error.retryable, Some(true));
}

/// Python: `test_wrapped_sandbox_error_inherits_retryable_from_cause`.
#[test]
fn wrapped_sandbox_error_inherits_retryable_from_cause() {
    let cause =
        SandboxError::new(ErrorCode::WorkspaceArchiveReadError, "/workspace").with_retryable(false);

    let error = SandboxError::new(ErrorCode::WorkspaceStopError, "/workspace").caused_by(&cause);

    assert_eq!(error.retryable, Some(false));
}

/// Python: `test_deterministic_sandbox_errors_are_non_retryable`.
#[test]
fn deterministic_sandbox_errors_are_non_retryable() {
    for code in [
        ErrorCode::WorkspaceReadNotFound,
        ErrorCode::WorkspaceWriteTypeError,
        ErrorCode::ExecTimeout,
    ] {
        assert_eq!(
            SandboxError::new(code, "boom").retryable,
            Some(false),
            "{code} should be non-retryable"
        );
    }
}

/// Python: `test_broad_archive_errors_default_to_unknown_retryability`.
#[test]
fn broad_archive_errors_default_to_unknown_retryability() {
    let error = SandboxError::new(ErrorCode::WorkspaceArchiveReadError, "/workspace");

    assert_eq!(error.retryable, None);
}

/// Python: `test_broad_materialization_and_snapshot_errors_default_to_unknown_retryability`.
#[test]
fn broad_materialization_and_snapshot_errors_default_to_unknown_retryability() {
    for code in [
        ErrorCode::GitCloneError,
        ErrorCode::GitCopyError,
        ErrorCode::SnapshotPersistError,
        ErrorCode::SnapshotRestoreError,
    ] {
        assert_eq!(
            SandboxError::new(code, "boom").retryable,
            None,
            "{code} retryability is unknown"
        );
    }
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/test_compaction.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_compaction_model_info_for_model_returns_context_window` (22 parameter sets).
#[test]
fn compaction_model_info_for_model_returns_context_window() {
    let cases: [(&str, u64); 22] = [
        ("gpt-5.4", 1_047_576),
        ("gpt-5.4-pro", 1_047_576),
        ("gpt-5.5", 1_047_576),
        ("gpt-5.5-2026-04-23", 1_047_576),
        ("gpt-5.5-pro", 1_047_576),
        ("gpt-5.5-pro-2026-04-23", 1_047_576),
        ("gpt-5.6", 1_047_576),
        ("gpt-5.6-sol", 1_047_576),
        ("gpt-5.6-terra", 1_047_576),
        ("gpt-5.6-luna", 1_047_576),
        ("gpt-5.3-codex", 400_000),
        ("gpt-5.4-mini", 400_000),
        ("gpt-4.1", 1_047_576),
        ("o3", 200_000),
        ("gpt-4o", 128_000),
        ("openai/gpt-5.4", 1_047_576),
        ("openai/gpt-5.5", 1_047_576),
        ("gpt-5-2", 400_000),
        ("gpt-5-4", 1_047_576),
        ("gpt-5-5", 1_047_576),
        ("openai/gpt-5-4-mini", 400_000),
        ("gpt-4-1-mini", 1_047_576),
    ];
    for (model, context_window) in cases {
        assert_eq!(
            context_window_for_model(model),
            Some(context_window),
            "context window for `{model}`"
        );
    }
}

/// Python: `test_compaction_model_info_for_model_rejects_unknown_model` and
/// `..._maybe_for_model_returns_none_for_unknown_model`. Rust has one fallible lookup instead of
/// a raising and a non-raising pair, so both land on `None`.
#[test]
fn compaction_model_info_returns_none_for_unknown_model() {
    assert_eq!(context_window_for_model("not-a-model"), None);
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/test_remote_mount_policy.py
// ----------------------------------------------------------------------------------------------

fn s3_mount(read_only: bool) -> Entry {
    let strategy = ConfiguredMountStrategy::rclone(S3Mount::new("example-bucket"));
    Entry::mount(Mount::new(Arc::new(strategy)).with_read_only(read_only))
}

fn policy_for(entries: Vec<(&str, Entry)>) -> String {
    let mut manifest = Manifest::new();
    for (path, entry) in entries {
        manifest = manifest.with_entry(path, entry);
    }
    build_remote_mount_policy_instructions(&manifest).expect("a policy for a manifest with mounts")
}

/// Python: `test_remote_mount_policy_does_not_suggest_direct_edits_for_read_only_mounts`.
#[test]
fn remote_mount_policy_does_not_suggest_direct_edits_for_read_only_mounts() {
    let policy = policy_for(vec![("data", s3_mount(true))]);

    assert!(
        policy.contains("/workspace/data (mounted in read-only mode)"),
        "{policy}"
    );
    assert!(!policy.contains("`apply_patch` directly"), "{policy}");
    assert!(!policy.contains("copy it back"), "{policy}");
    assert!(
        policy.contains("Do not edit paths marked read-only in place"),
        "{policy}"
    );
    assert!(policy.contains("including with `apply_patch`"), "{policy}");
    assert!(
        policy.contains("do not write edited files back"),
        "{policy}"
    );
}

/// Python: `test_remote_mount_policy_keeps_direct_and_copy_back_guidance_for_read_write_mounts`.
#[test]
fn remote_mount_policy_keeps_direct_and_copy_back_guidance_for_read_write_mounts() {
    let policy = policy_for(vec![("data", s3_mount(false))]);

    assert!(
        policy.contains("/workspace/data (mounted in read+write mode)"),
        "{policy}"
    );
    assert!(
        policy.contains("Use `apply_patch` directly for text edits on read+write mounts."),
        "{policy}"
    );
    assert!(
        policy.contains("For shell-based edits on read+write mounts"),
        "{policy}"
    );
    assert!(policy.contains("copy it back"), "{policy}");
    assert!(
        !policy.contains("Do not edit paths marked read-only"),
        "{policy}"
    );
}

/// Python: `test_remote_mount_policy_handles_mixed_read_only_and_read_write_mounts`.
#[test]
fn remote_mount_policy_handles_mixed_read_only_and_read_write_mounts() {
    let policy = policy_for(vec![("input", s3_mount(true)), ("output", s3_mount(false))]);

    assert!(
        policy.contains("/workspace/input (mounted in read-only mode)"),
        "{policy}"
    );
    assert!(
        policy.contains("/workspace/output (mounted in read+write mode)"),
        "{policy}"
    );
    assert!(
        policy.contains("Use `apply_patch` directly for text edits on read+write mounts."),
        "{policy}"
    );
    assert!(
        policy.contains("For shell-based edits on read+write mounts"),
        "{policy}"
    );
    assert!(
        policy.contains("Do not edit paths marked read-only in place"),
        "{policy}"
    );
    assert!(policy.contains("including with `apply_patch`"), "{policy}");
    assert!(
        policy.contains("do not write edited files back"),
        "{policy}"
    );
}

/// Python: `test_remote_mount_policy_returns_none_without_remote_mounts`.
#[test]
fn remote_mount_policy_returns_none_without_remote_mounts() {
    let manifest = Manifest::new().with_entry("local", Entry::dir());

    assert!(build_remote_mount_policy_instructions(&manifest).is_none());
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/test_exposed_ports.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_exposed_port_endpoint_formats_urls`.
#[test]
fn exposed_port_endpoint_formats_urls() {
    let insecure = ExposedPortEndpoint::new("127.0.0.1", 8765, false);
    let secure = ExposedPortEndpoint::new("sandbox.example.test", 443, true);

    assert_eq!(insecure.url_for("http").unwrap(), "http://127.0.0.1:8765/");
    assert_eq!(insecure.url_for("ws").unwrap(), "ws://127.0.0.1:8765/");
    assert_eq!(
        secure.url_for("http").unwrap(),
        "https://sandbox.example.test/"
    );
    assert_eq!(secure.url_for("ws").unwrap(), "wss://sandbox.example.test/");
}

/// Python: `test_exposed_port_endpoint_with_query`.
#[test]
fn exposed_port_endpoint_with_query() {
    let endpoint = ExposedPortEndpoint::new("preview.example.com", 443, true)
        .with_query("bl_preview_token=abc123");

    assert_eq!(
        endpoint.url_for("http").unwrap(),
        "https://preview.example.com/?bl_preview_token=abc123"
    );
    assert_eq!(
        endpoint.url_for("ws").unwrap(),
        "wss://preview.example.com/?bl_preview_token=abc123"
    );
}

/// Python: `test_exposed_port_endpoint_accepts_leading_question_mark_query`.
#[test]
fn exposed_port_endpoint_accepts_leading_question_mark_query() {
    let endpoint = ExposedPortEndpoint::new("preview.example.com", 443, true)
        .with_query("?bl_preview_token=abc123");

    assert_eq!(
        endpoint.url_for("http").unwrap(),
        "https://preview.example.com/?bl_preview_token=abc123"
    );
    assert_eq!(
        endpoint.url_for("ws").unwrap(),
        "wss://preview.example.com/?bl_preview_token=abc123"
    );
}

/// Python: `test_exposed_port_endpoint_empty_query`.
#[test]
fn exposed_port_endpoint_empty_query() {
    let endpoint = ExposedPortEndpoint::new("127.0.0.1", 8080, false).with_query("");

    assert_eq!(endpoint.url_for("http").unwrap(), "http://127.0.0.1:8080/");
}

/// Python: `test_unix_local_resolve_exposed_port_uses_wrapper_and_normalizes_state`.
#[tokio::test]
async fn unix_local_resolve_exposed_port_uses_wrapper_and_normalizes_state() {
    let client = UnixLocalSandboxClient::new().with_exposed_ports([8765, 8765]);
    let session = client.create(None).await.expect("create session");

    let endpoint = session.resolve_exposed_port(8765).await.expect("endpoint");

    assert_eq!(session.exposed_ports(), vec![8765]);
    assert_eq!(session.session_state().exposed_ports, vec![8765]);
    assert_eq!(endpoint, ExposedPortEndpoint::new("127.0.0.1", 8765, false));
    assert_eq!(endpoint.url_for("ws").unwrap(), "ws://127.0.0.1:8765/");

    client.delete(session).await.expect("delete session");
}

/// Python: `test_unix_local_resolve_exposed_port_rejects_undeclared_ports`.
#[tokio::test]
async fn unix_local_resolve_exposed_port_rejects_undeclared_ports() {
    let client = UnixLocalSandboxClient::new().with_exposed_ports([8765]);
    let session = client.create(None).await.expect("create session");

    let error = session
        .resolve_exposed_port(9000)
        .await
        .expect_err("an undeclared port is refused");

    assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("not_configured")
    );
    assert_eq!(
        error.context.get("exposed_ports").map(String::as_str),
        Some("8765")
    );
    // `not_configured` can never start working, so it is explicitly non-retryable.
    assert_eq!(error.retryable, Some(false));

    client.delete(session).await.expect("delete session");
}

/// Companion to the two above: a declared port the backend cannot map is a different reason.
#[test]
fn exposed_port_unavailable_separates_the_two_reasons() {
    let error = exposed_port_unavailable(
        "unix_local",
        8765,
        &[8765],
        ExposedPortUnavailable::BackendUnavailable,
    );
    assert_eq!(
        error.context.get("reason").map(String::as_str),
        Some("backend_unavailable")
    );
    assert_eq!(error.retryable, None);
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/test_retry.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_retry_async_validates_configuration`, for the one check Rust's types do not
/// already make impossible. A negative interval and an unknown backoff cannot be expressed:
/// `Duration` is unsigned and `BackoffStrategy` is an enum.
#[test]
fn retry_policy_clamps_a_zero_attempt_budget_to_one() {
    assert_eq!(RetryPolicy::new().with_max_attempts(0).max_attempts, 1);
}

/// Python: `test_retry_async_retries_with_expected_backoff_and_async_hook` (3 parameter sets),
/// reduced to the delay schedule, which is what the Python test's fake `asyncio.sleep` records.
#[test]
fn retry_async_computes_the_expected_backoff_delays() {
    let interval = Duration::from_millis(500);
    let delays = |backoff: BackoffStrategy| {
        let policy = RetryPolicy::new()
            .with_interval(interval)
            .with_max_attempts(3)
            .with_backoff(backoff);
        vec![policy.delay(1), policy.delay(2)]
    };

    assert_eq!(
        delays(BackoffStrategy::Fixed),
        vec![Duration::from_millis(500), Duration::from_millis(500)]
    );
    assert_eq!(
        delays(BackoffStrategy::Linear),
        vec![Duration::from_millis(500), Duration::from_millis(1000)]
    );
    assert_eq!(
        delays(BackoffStrategy::Exponential),
        vec![Duration::from_millis(500), Duration::from_millis(1000)]
    );
    assert_eq!(BackoffStrategy::Fixed.as_str(), "fixed");
    assert_eq!(BackoffStrategy::Linear.as_str(), "linear");
    assert_eq!(BackoffStrategy::Exponential.as_str(), "exponential");
}

/// Python: `test_retry_async_retries_with_expected_backoff_and_async_hook`, the attempt count and
/// the per-attempt hook.
#[tokio::test]
async fn retry_async_retries_until_the_operation_succeeds() {
    let attempts = Arc::new(AtomicU32::new(0));
    let hook_calls = Arc::new(std::sync::Mutex::new(Vec::new()));

    let seen = attempts.clone();
    let recorded = hook_calls.clone();
    let result: Result<String, SandboxError> = retry_async(
        RetryPolicy::new()
            .with_interval(Duration::ZERO)
            .with_max_attempts(3),
        move || {
            let seen = seen.clone();
            async move {
                if seen.fetch_add(1, Ordering::SeqCst) < 2 {
                    Err(SandboxError::new(ErrorCode::ExecTransportError, "sandbox"))
                } else {
                    Ok("ok:sandbox".to_string())
                }
            }
        },
        move |error, attempt| {
            recorded.lock().unwrap().push((attempt, 3u32));
            error.code == ErrorCode::ExecTransportError
        },
    )
    .await;

    assert_eq!(result.unwrap(), "ok:sandbox");
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    assert_eq!(*hook_calls.lock().unwrap(), vec![(1, 3), (2, 3)]);
}

/// Python: `test_retry_async_stops_without_sleep_when_retry_is_rejected`.
#[tokio::test]
async fn retry_async_stops_when_the_predicate_rejects_the_error() {
    let attempts = Arc::new(AtomicU32::new(0));

    let seen = attempts.clone();
    let result: Result<(), SandboxError> = retry_async(
        // A non-zero interval would be slept through if the loop retried at all.
        RetryPolicy::new()
            .with_interval(Duration::from_secs(30))
            .with_max_attempts(3),
        move || {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Err(SandboxError::new(ErrorCode::ExecTransportError, "stop"))
            }
        },
        |_error, _attempt| false,
    )
    .await;

    assert_eq!(result.unwrap_err().message, "stop");
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

// ----------------------------------------------------------------------------------------------
// tests/sandbox/test_posix_tool_paths.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_shell_workdir_normalizes_backslashes_as_sandbox_separators` and
/// `..._before_unix_local_resolution`. Rust passes the working directory to
/// `ExecRequest::with_cwd` rather than prefixing a `cd` to the command, so the behaviour under
/// test is the path normalization both spellings share.
#[tokio::test]
async fn sandbox_paths_treat_backslashes_as_separators() {
    let session = openai_agents_sdk::sandbox::testing::scripted_sandbox_session([])
        .with_manifest(Manifest::new().with_root("/workspace"));
    assert_eq!(
        session.normalize_path(r"src\project").unwrap(),
        "/workspace/src/project"
    );

    // The same spelling against a real workspace root resolves the same way.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("workspace");
    std::fs::create_dir_all(&root).unwrap();
    let client = UnixLocalSandboxClient::new();
    let session = client
        .create(Some(
            Manifest::new().with_root(root.to_string_lossy().replace('\\', "/")),
        ))
        .await
        .expect("create session");
    assert_eq!(
        session.normalize_path(r"src\project").unwrap(),
        format!("{}/src/project", root.to_string_lossy().replace('\\', "/"))
    );
    client.delete(session).await.expect("delete session");
}
