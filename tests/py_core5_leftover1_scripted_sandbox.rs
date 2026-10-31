//! Port of `tests/test_scripted_sandbox.py`: the scripted sandbox session itself.
//!
//! Python-only groups: introspection of dynamic attributes (`hasattr`/`dir`/`inspect`/type
//! hints), `io.BytesIO`/`io.StringIO` result and argument snapshots (the Rust steps own `Vec<u8>`
//! values, and a recorded call is an owned clone), step-spec validation of mistyped dictionaries
//! (the builder makes them unrepresentable), `deepcopy` failures, and matcher exceptions (a
//! matcher returns `bool`). The apply_patch create path and the black-box agent workflow are
//! ported in `py_core3_shelltools_scripted_sandbox.rs`.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::sandbox::SandboxError;
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxSession};
use openai_agents_sdk::sandbox::testing::{
    SandboxCallResult, SandboxMethod, SandboxStep, scripted_sandbox_session,
};
use openai_agents_sdk::sandbox::types::ExecResult;

fn exec_ok(stdout: &str) -> ExecResult {
    ExecResult {
        exit_code: 0,
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
    }
}

fn context<'a>(error: &'a SandboxError, key: &str) -> Option<&'a str> {
    error.context.get(key).map(String::as_str)
}

/// Python: `test_scripted_sandbox_cleanup_does_not_advertise_pty_termination` (the lifecycle
/// half: a closed session is no longer running).
#[tokio::test]
async fn scripted_sandbox_cleanup_stops_the_session() {
    let session = scripted_sandbox_session([]);
    assert!(session.running().await.unwrap());

    session.shutdown().await.unwrap();

    assert!(!session.running().await.unwrap());
}

/// Python: `test_scripted_sandbox_snapshots_manifest_and_derives_pty_support`.
#[tokio::test]
async fn scripted_sandbox_keeps_its_manifest_and_derives_pty_support() {
    let session = scripted_sandbox_session([SandboxStep::new(SandboxMethod::PtyExecStart)])
        .with_manifest(Manifest::new().with_root("/configured"));
    assert_eq!(session.manifest().root(), "/configured");
    assert!(session.supports_pty());

    let pty_session = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::PtyExecStart),
        SandboxStep::new(SandboxMethod::PtyWriteStdin),
    ]);
    assert!(pty_session.supports_pty());

    // No PTY method configured: the capability is off.
    let plain = scripted_sandbox_session([SandboxStep::new(SandboxMethod::Exec)]);
    assert!(!plain.supports_pty());
}

/// Python: `test_scripted_sandbox_exposes_pty_methods_as_one_capability` (both orders).
#[tokio::test]
async fn scripted_sandbox_exposes_pty_methods_as_one_capability() {
    for (configured, missing) in [
        (SandboxMethod::PtyExecStart, SandboxMethod::PtyWriteStdin),
        (SandboxMethod::PtyWriteStdin, SandboxMethod::PtyExecStart),
    ] {
        let session = scripted_sandbox_session([SandboxStep::new(configured)]);
        assert!(session.supports_pty());

        let error = match missing {
            SandboxMethod::PtyExecStart => session
                .pty_exec_start(PtyExecRequest::new("pwd"))
                .await
                .unwrap_err(),
            _ => session
                .pty_write_stdin(PtyWriteRequest::new(1, ""))
                .await
                .unwrap_err(),
        };

        assert_eq!(context(&error, "actual_method"), Some(missing.as_str()));
        assert_eq!(
            context(&error, "expected_method"),
            Some(configured.as_str())
        );
    }
}

/// Python: `test_scripted_sandbox_supports_capability_method_inventory` (the methods the Rust
/// session trait has; there is no `apply_patch` session method, patches go through the editor).
#[tokio::test]
async fn scripted_sandbox_supports_capability_method_inventory() {
    let pty_start = PtyExecUpdate {
        process_id: Some(123),
        output: b"started".to_vec(),
        exit_code: None,
        original_token_count: None,
    };
    let pty_write = PtyExecUpdate {
        process_id: None,
        output: b"done".to_vec(),
        exit_code: Some(0),
        original_token_count: None,
    };
    let session = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::ReadFile).returning_bytes(b"contents".to_vec()),
        SandboxStep::new(SandboxMethod::WriteFile),
        SandboxStep::new(SandboxMethod::Ls).returning_entries(Vec::new()),
        SandboxStep::new(SandboxMethod::Mkdir),
        SandboxStep::new(SandboxMethod::Rm),
        SandboxStep::new(SandboxMethod::PtyExecStart).returning(SandboxCallResult::Pty(pty_start)),
        SandboxStep::new(SandboxMethod::PtyWriteStdin).returning(SandboxCallResult::Pty(pty_write)),
    ]);

    assert_eq!(session.read_file("in.txt").await.unwrap(), b"contents");
    session.write_file("out.txt", b"payload").await.unwrap();
    assert!(session.ls(".").await.unwrap().is_empty());
    session.mkdir("new", true).await.unwrap();
    session.rm("old", true).await.unwrap();
    let mut request = PtyExecRequest::new("sh");
    request.tty = true;
    assert_eq!(
        session.pty_exec_start(request).await.unwrap().process_id,
        Some(123)
    );
    assert_eq!(
        session
            .pty_write_stdin(PtyWriteRequest::new(123, "exit"))
            .await
            .unwrap()
            .exit_code,
        Some(0)
    );

    let calls = session.calls();
    assert_eq!(
        calls.iter().map(|call| call.method).collect::<Vec<_>>(),
        vec![
            SandboxMethod::ReadFile,
            SandboxMethod::WriteFile,
            SandboxMethod::Ls,
            SandboxMethod::Mkdir,
            SandboxMethod::Rm,
            SandboxMethod::PtyExecStart,
            SandboxMethod::PtyWriteStdin,
        ]
    );
    // The write's payload is recorded as the bytes that were sent.
    assert_eq!(calls[1].payload.as_deref(), Some(b"payload".as_slice()));
    session.assert_complete();
}

/// Python: `test_scripted_sandbox_records_detached_fifo_calls`.
#[tokio::test]
async fn scripted_sandbox_records_fifo_calls() {
    let session = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::Mkdir),
        SandboxStep::new(SandboxMethod::Exec).returning_exec(exec_ok("ok")),
    ]);

    session.mkdir("before", true).await.unwrap();
    let result = session.exec(ExecRequest::argv(["pwd"])).await.unwrap();

    assert_eq!(result.stdout, b"ok");
    assert_eq!(session.remaining_steps(), 0);
    session.assert_complete();
    let calls = session.calls();
    assert_eq!(calls[0].call_index, 0);
    assert_eq!(calls[0].method, SandboxMethod::Mkdir);
    assert_eq!(calls[0].arguments, vec!["before", "true"]);
    assert_eq!(calls[1].call_index, 1);
    assert_eq!(calls[1].method, SandboxMethod::Exec);
    assert_eq!(calls[1].arguments, vec!["pwd"]);

    // A recorded call is a detached copy: reading the history again is unchanged.
    let mut returned = session.calls();
    returned[0].arguments[0] = "mutated".into();
    assert_eq!(session.calls()[0].arguments[0], "before");
}

/// Python: `test_scripted_sandbox_supports_matchers_responders_and_errors`.
#[tokio::test]
async fn scripted_sandbox_supports_matchers_responders_and_errors() {
    let session = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.arguments == ["pwd"])
            .responding(|call| {
                Ok(SandboxCallResult::Exec(exec_ok(&format!(
                    "call {}",
                    call.arguments.len()
                ))))
            }),
        SandboxStep::new(SandboxMethod::Exec).failing(SandboxError::new(
            ErrorCode::ExecNonZero,
            "sandbox unavailable",
        )),
    ]);

    assert_eq!(
        session
            .exec(ExecRequest::argv(["pwd"]))
            .await
            .unwrap()
            .stdout,
        b"call 1"
    );
    let error = session.exec(ExecRequest::argv(["next"])).await.unwrap_err();
    assert_eq!(error.code, ErrorCode::ExecNonZero);
    assert_eq!(error.message, "sandbox unavailable");
    session.assert_complete();
}

/// Python: `test_scripted_sandbox_reports_structured_payload_free_failures`.
#[tokio::test]
async fn scripted_sandbox_reports_structured_payload_free_failures() {
    // A call for the wrong method does not consume the step.
    let mismatch = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::Exec).returning_exec(exec_ok("")),
        SandboxStep::new(SandboxMethod::ReadFile),
    ]);
    let error = mismatch.read_file("/secret/payload.txt").await.unwrap_err();
    assert_eq!(context(&error, "script"), Some("unexpected_call"));
    assert_eq!(context(&error, "call_index"), Some("0"));
    assert!(error.message.contains("call #1"), "{}", error.message);
    assert_eq!(context(&error, "actual_method"), Some("read_file"));
    assert_eq!(context(&error, "expected_method"), Some("exec"));
    assert_eq!(context(&error, "remaining_steps"), Some("2"));
    assert_eq!(mismatch.remaining_steps(), 2);
    assert!(!error.message.contains("/secret/payload.txt"));
    mismatch.exec(ExecRequest::argv(["retry"])).await.unwrap();
    assert_eq!(mismatch.remaining_steps(), 1);

    // A rejected matcher does not consume the step either.
    let rejected = scripted_sandbox_session([SandboxStep::new(SandboxMethod::Exec)
        .matching(|call| call.arguments == ["expected"])
        .returning_exec(exec_ok(""))]);
    let error = rejected
        .exec(ExecRequest::argv(["secret command"]))
        .await
        .unwrap_err();
    assert_eq!(context(&error, "script"), Some("matcher_rejected"));
    assert_eq!(context(&error, "call_index"), Some("0"));
    assert!(error.message.contains("call #1"), "{}", error.message);
    assert_eq!(context(&error, "method"), Some("exec"));
    assert!(!error.message.contains("secret command"));
    assert!(!format!("{error:?}").contains("secret command"));
    assert_eq!(rejected.remaining_steps(), 1);
    rejected
        .exec(ExecRequest::argv(["expected"]))
        .await
        .unwrap();
    rejected.assert_complete();

    // A call past the end of the script has no expected method.
    let extra = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::Exec).returning_exec(exec_ok(""))
    ]);
    extra.exec(ExecRequest::argv(["first"])).await.unwrap();
    let error = extra
        .exec(ExecRequest::argv(["second secret"]))
        .await
        .unwrap_err();
    assert_eq!(context(&error, "call_index"), Some("1"));
    assert!(error.message.contains("call #2"), "{}", error.message);
    assert_eq!(context(&error, "expected_method"), None);
    assert_eq!(context(&error, "remaining_steps"), Some("0"));
    assert!(!error.message.contains("second secret"));
}

/// Python: `test_scripted_sandbox_reports_unconsumed_steps`.
#[test]
fn scripted_sandbox_reports_unconsumed_steps() {
    let session = scripted_sandbox_session([
        SandboxStep::new(SandboxMethod::Exec).returning_exec(exec_ok("")),
        SandboxStep::new(SandboxMethod::ReadFile),
    ]);

    let error = session.complete().unwrap_err();

    assert_eq!(context(&error, "script"), Some("unconsumed_steps"));
    assert_eq!(context(&error, "remaining_steps"), Some("2"));
    assert_eq!(context(&error, "pending_methods"), Some("exec,read_file"));
}
