//! Error redaction around credential-bearing mounts, ported from the cases of
//! `tests/sandbox/test_mount_security.py` that the earlier ports filed as Python-only although the
//! contract is observable through `redact_for_protected_mounts`: what a protected error keeps
//! (code, operation, retry flag), what it loses (message, context, exit code), and that a
//! credentialless manifest never has its errors touched.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::sandboxes::docker::redact_for_protected_mounts;
use openai_agents_sdk::sandbox::types::ExecResult;
use openai_agents_sdk::sandbox::{
    ConfiguredMountStrategy, Entry, ErrorCode, Manifest, S3Mount, SandboxError,
};

const SENTINEL: &str = "provider-operation-secret";

fn protected_manifest() -> Manifest {
    Manifest::new().with_root("/workspace").with_entry(
        "data",
        Entry::mount(
            ConfiguredMountStrategy::rclone(
                S3Mount::new("bucket").with_credentials("access-key", SENTINEL),
            )
            .mount(),
        ),
    )
}

fn leaky(code: ErrorCode, retryable: Option<bool>) -> SandboxError {
    let mut error = SandboxError::new(code, format!("provider failed with {SENTINEL}"))
        .with_context("secret", SENTINEL);
    error.retryable = retryable;
    error
}

/// `test_protected_mount_config_error_preserves_valid_structured_fields`: the error code, the
/// operation and the retry flag a caller branches on survive, and nothing else does.
#[test]
fn a_protected_error_keeps_its_code_operation_and_retry_flag() {
    let mut source = leaky(ErrorCode::ExecTimeout, Some(true));
    source.op = Some(openai_agents_sdk::sandbox::SandboxOp::Exec);

    let safe = redact_for_protected_mounts(&protected_manifest(), source);

    assert_eq!(safe.code, ErrorCode::ExecTimeout);
    assert_eq!(safe.op, Some(openai_agents_sdk::sandbox::SandboxOp::Exec));
    assert_eq!(safe.retryable, Some(true));
    assert!(safe.context.is_empty(), "{:?}", safe.context);
    assert!(!format!("{safe:?}").contains(SENTINEL), "{safe:?}");
}

/// `test_protected_structured_sandbox_error_requires_retryable_field`: an error with no retry
/// decision stays without one, and a decision of `false` stays `false`.
#[test]
fn a_protected_error_does_not_invent_a_retry_decision() {
    let undecided = redact_for_protected_mounts(
        &protected_manifest(),
        leaky(ErrorCode::ExecTransportError, None),
    );
    let refused = redact_for_protected_mounts(
        &protected_manifest(),
        leaky(ErrorCode::ExecTransportError, Some(false)),
    );

    assert_eq!(undecided.retryable, None);
    assert_eq!(refused.retryable, Some(false));
    assert!(!format!("{undecided:?}{refused:?}").contains(SENTINEL));
}

/// `test_protected_structured_sandbox_error_preserves_safe_subtype_state`: a non-zero exit keeps
/// its code, and the command, exit code, stdout and stderr the source carried are blanked.
#[test]
fn a_protected_non_zero_exit_loses_its_output_and_command() {
    let command = vec![SENTINEL.to_string()];
    let source = SandboxError::exec_nonzero(
        &command,
        &ExecResult {
            stdout: SENTINEL.as_bytes().to_vec(),
            stderr: SENTINEL.as_bytes().to_vec(),
            exit_code: 42,
        },
    );
    assert_eq!(source.exit_code, Some(42));

    let safe = redact_for_protected_mounts(&protected_manifest(), source);

    assert_eq!(safe.code, ErrorCode::ExecNonZero);
    assert_eq!(safe.exit_code, None);
    assert!(safe.context.is_empty(), "{:?}", safe.context);
    assert!(!format!("{safe:?}").contains(SENTINEL), "{safe:?}");
}

/// `test_credentialless_external_client_ignores_opaque_provider_link` and
/// `test_credentialless_structured_mount_error_is_unchanged`: with no credential in the manifest
/// the error is handed back exactly as it was, retry flag and context included.
#[test]
fn a_credentialless_manifest_leaves_the_error_unchanged() {
    let mut source = SandboxError::new(ErrorCode::MountFailed, "credentialless-stderr")
        .with_context("command", "credentialless-command");
    source.retryable = Some(true);

    let untouched = redact_for_protected_mounts(&Manifest::new(), source.clone());

    assert_eq!(untouched.code, source.code);
    assert_eq!(untouched.message, "credentialless-stderr");
    assert_eq!(untouched.retryable, Some(true));
    assert_eq!(untouched.context, source.context);
}
