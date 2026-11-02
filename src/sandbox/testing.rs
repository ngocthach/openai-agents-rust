//! A scripted sandbox session for tests. Mirrors `agents.testing.sandbox`.
//!
//! Testing an agent that uses a sandbox usually means choosing between a real backend — slow,
//! and dependent on what is installed — and a hand-rolled fake that drifts from the real trait.
//! [`ScriptedSandboxSession`] is the third option: a FIFO script of the calls the agent is
//! expected to make, each with the answer it should get. A call that does not match the script
//! fails the test where it happens, with the call that was actually made, rather than failing
//! later as a confusing wrong answer.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use openai_agents_sdk::sandbox::testing::{SandboxMethod, SandboxStep, scripted_sandbox_session};
//! use openai_agents_sdk::sandbox::types::ExecResult;
//!
//! let session = scripted_sandbox_session([
//!     SandboxStep::new(SandboxMethod::Exec).returning_exec(ExecResult {
//!         exit_code: 0,
//!         stdout: b"ok\n".to_vec(),
//!         stderr: Vec::new(),
//!     }),
//! ]);
//! let session = Arc::new(session);
//! // `SandboxRunConfig::with_session(session.clone())` drives a run against the script.
//! session.assert_complete();
//! ```

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::pty::{PtyExecRequest, PtyExecUpdate, PtyWriteRequest};
use crate::sandbox::session::{ExecRequest, SandboxSession};
use crate::sandbox::types::{ExecResult, FileEntry};

/// A session method a script can answer. Mirrors Python's `SandboxMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SandboxMethod {
    /// `exec`.
    Exec,
    /// `read_file`.
    ReadFile,
    /// `write_file`.
    WriteFile,
    /// `mkdir`.
    Mkdir,
    /// `rm`.
    Rm,
    /// `mv`.
    Mv,
    /// `ls`.
    Ls,
    /// `pty_exec_start`.
    PtyExecStart,
    /// `pty_write_stdin`.
    PtyWriteStdin,
    /// `persist_workspace`.
    PersistWorkspace,
    /// `hydrate_workspace`.
    HydrateWorkspace,
}

impl SandboxMethod {
    /// The method name, as the session trait spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            SandboxMethod::Exec => "exec",
            SandboxMethod::ReadFile => "read_file",
            SandboxMethod::WriteFile => "write_file",
            SandboxMethod::Mkdir => "mkdir",
            SandboxMethod::Rm => "rm",
            SandboxMethod::Mv => "mv",
            SandboxMethod::Ls => "ls",
            SandboxMethod::PtyExecStart => "pty_exec_start",
            SandboxMethod::PtyWriteStdin => "pty_write_stdin",
            SandboxMethod::PersistWorkspace => "persist_workspace",
            SandboxMethod::HydrateWorkspace => "hydrate_workspace",
        }
    }
}

impl fmt::Display for SandboxMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One call the agent made. Mirrors Python's `SandboxCall`.
///
/// `arguments` holds the call's path or argument vector in order, and `payload` the bytes of a
/// write, so an assertion can look at exactly what was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxCall {
    /// Zero-based position in the call history.
    pub call_index: usize,
    /// Which method was called.
    pub method: SandboxMethod,
    /// The call's string arguments, in order.
    pub arguments: Vec<String>,
    /// The bytes the call carried, for a write or a hydrate.
    pub payload: Option<Vec<u8>>,
}

impl SandboxCall {
    /// Whether any argument equals `value`.
    pub fn has_argument(&self, value: &str) -> bool {
        self.arguments.iter().any(|argument| argument == value)
    }

    /// The payload as lossy UTF-8.
    pub fn payload_text(&self) -> String {
        self.payload
            .as_deref()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_default()
    }
}

/// What a scripted step hands back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxCallResult {
    /// The result of a command.
    Exec(ExecResult),
    /// File contents, or an archive.
    Bytes(Vec<u8>),
    /// A directory listing.
    Entries(Vec<FileEntry>),
    /// A PTY window.
    Pty(PtyExecUpdate),
    /// Nothing; the call only had to succeed.
    Unit,
}

/// How a step answers.
/// Computes a scripted answer from the call that triggered it.
pub type SandboxResponder =
    Arc<dyn Fn(&SandboxCall) -> SandboxResult<SandboxCallResult> + Send + Sync>;

/// Decides whether a call matches the step that is about to answer it.
pub type SandboxMatcher = Arc<dyn Fn(&SandboxCall) -> bool + Send + Sync>;

enum StepOutcome {
    /// Return a fixed value.
    Value(SandboxCallResult),
    /// Compute the value from the call.
    Responder(SandboxResponder),
    /// Fail the call.
    Error(SandboxError),
}

impl fmt::Debug for StepOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepOutcome::Value(value) => f.debug_tuple("Value").field(value).finish(),
            StepOutcome::Responder(_) => f.write_str("Responder(..)"),
            StepOutcome::Error(error) => f.debug_tuple("Error").field(error).finish(),
        }
    }
}

/// One scripted call. Mirrors Python's `SandboxStepSpec`.
///
/// A step names the method it answers and exactly one outcome. The optional matcher gets the
/// call before the outcome is produced, so a test can assert on the arguments at the point they
/// are made instead of afterwards.
pub struct SandboxStep {
    method: SandboxMethod,
    matcher: Option<SandboxMatcher>,
    outcome: StepOutcome,
}

impl fmt::Debug for SandboxStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxStep")
            .field("method", &self.method)
            .field("matcher", &self.matcher.as_ref().map(|_| "..."))
            .field("outcome", &self.outcome)
            .finish()
    }
}

impl SandboxStep {
    /// A step answering `method` with nothing.
    pub fn new(method: SandboxMethod) -> Self {
        SandboxStep {
            method,
            matcher: None,
            outcome: StepOutcome::Value(SandboxCallResult::Unit),
        }
    }

    /// Require the call to satisfy `matcher`.
    pub fn matching<F>(mut self, matcher: F) -> Self
    where
        F: Fn(&SandboxCall) -> bool + Send + Sync + 'static,
    {
        self.matcher = Some(Arc::new(matcher));
        self
    }

    /// Answer with a fixed value.
    pub fn returning(mut self, result: SandboxCallResult) -> Self {
        self.outcome = StepOutcome::Value(result);
        self
    }

    /// Answer a command with `result`.
    pub fn returning_exec(self, result: ExecResult) -> Self {
        self.returning(SandboxCallResult::Exec(result))
    }

    /// Answer a read with `bytes`.
    pub fn returning_bytes(self, bytes: impl Into<Vec<u8>>) -> Self {
        self.returning(SandboxCallResult::Bytes(bytes.into()))
    }

    /// Answer a listing with `entries`.
    pub fn returning_entries(self, entries: Vec<FileEntry>) -> Self {
        self.returning(SandboxCallResult::Entries(entries))
    }

    /// Compute the answer from the call.
    pub fn responding<F>(mut self, responder: F) -> Self
    where
        F: Fn(&SandboxCall) -> SandboxResult<SandboxCallResult> + Send + Sync + 'static,
    {
        self.outcome = StepOutcome::Responder(Arc::new(responder));
        self
    }

    /// Fail the call.
    pub fn failing(mut self, error: SandboxError) -> Self {
        self.outcome = StepOutcome::Error(error);
        self
    }
}

/// The error a script raises when a run does not follow it.
///
/// The message and context name methods and positions only: the arguments of the failing call are
/// never rendered, because they may carry paths or commands the test does not want in a log.
fn script_error(kind: &str, message: String) -> SandboxError {
    SandboxError::new(ErrorCode::SandboxConfigInvalid, message).with_context("script", kind)
}

/// A sandbox session that answers from a FIFO script. Mirrors Python's
/// `ScriptedSandboxSession`.
///
/// Only the methods a script configures are meant to be called; any other call fails rather than
/// returning a plausible-looking default, so a test cannot quietly pass against behaviour nobody
/// scripted.
#[derive(Debug)]
pub struct ScriptedSandboxSession {
    id: String,
    manifest: Manifest,
    /// Path grants a capability changed after the session started.
    live_grants: crate::sandbox::session::LiveGrants,
    steps: Mutex<Vec<SandboxStep>>,
    calls: Mutex<Vec<SandboxCall>>,
    /// Whether the script configures a PTY method; fixed when the script is built.
    pty: bool,
    running: AtomicBool,
}

impl ScriptedSandboxSession {
    /// A session following `steps`.
    pub fn new(steps: impl IntoIterator<Item = SandboxStep>) -> Self {
        let steps: Vec<SandboxStep> = steps.into_iter().collect();
        // The two PTY methods are one capability: configuring either one turns it on.
        let pty = steps.iter().any(|step| {
            matches!(
                step.method,
                SandboxMethod::PtyExecStart | SandboxMethod::PtyWriteStdin
            )
        });
        ScriptedSandboxSession {
            id: "scripted".to_string(),
            manifest: Manifest::new(),
            live_grants: Default::default(),
            steps: Mutex::new(steps),
            calls: Mutex::new(Vec::new()),
            pty,
            running: AtomicBool::new(true),
        }
    }

    /// Use `manifest` as the session's manifest.
    pub fn with_manifest(mut self, manifest: Manifest) -> Self {
        self.manifest = manifest;
        self
    }

    /// The calls made so far, in order.
    pub fn calls(&self) -> Vec<SandboxCall> {
        self.calls.lock().expect("scripted session lock").clone()
    }

    /// How many scripted steps have not been used.
    pub fn remaining_steps(&self) -> usize {
        self.steps.lock().expect("scripted session lock").len()
    }

    /// Panic when scripted steps remain.
    ///
    /// A script that was not consumed means the run did less than the test claims it does, which
    /// an assertion on the output alone will not catch.
    pub fn assert_complete(&self) {
        if let Err(error) = self.complete() {
            panic!("{}", error.message);
        }
    }

    /// The non-panicking form of [`assert_complete`](Self::assert_complete).
    pub fn complete(&self) -> SandboxResult<()> {
        let steps = self.steps.lock().expect("scripted session lock");
        if steps.is_empty() {
            return Ok(());
        }
        let pending: Vec<&str> = steps.iter().map(|step| step.method.as_str()).collect();
        Err(script_error(
            "unconsumed_steps",
            format!(
                "the scripted sandbox session has {} unconsumed step(s): {}",
                steps.len(),
                pending.join(", ")
            ),
        )
        .with_context("remaining_steps", steps.len().to_string())
        .with_context("pending_methods", pending.join(",")))
    }

    /// Record a call and take the answer the script gives it.
    fn invoke(
        &self,
        method: SandboxMethod,
        arguments: Vec<String>,
        payload: Option<Vec<u8>>,
    ) -> SandboxResult<SandboxCallResult> {
        let call = {
            let mut calls = self.calls.lock().expect("scripted session lock");
            let call = SandboxCall {
                call_index: calls.len(),
                method,
                arguments,
                payload,
            };
            calls.push(call.clone());
            call
        };

        // The head step is only consumed once the call is accepted: a call that does not match
        // it, or that its matcher rejects, leaves the script where it was so the test can retry.
        let matcher = {
            let steps = self.steps.lock().expect("scripted session lock");
            let remaining = steps.len();
            let mismatch = |expected: Option<SandboxMethod>, message: String| {
                let mut error = script_error("unexpected_call", message)
                    .with_context("call_index", call.call_index.to_string())
                    .with_context("actual_method", method.as_str())
                    .with_context("remaining_steps", remaining.to_string());
                if let Some(expected) = expected {
                    error = error.with_context("expected_method", expected.as_str());
                }
                error
            };
            let Some(head) = steps.first() else {
                return Err(mismatch(
                    None,
                    format!(
                        "unexpected sandbox `{method}` call #{}: no scripted steps remain",
                        call.call_index + 1
                    ),
                ));
            };
            if head.method != method {
                return Err(mismatch(
                    Some(head.method),
                    format!(
                        "unexpected sandbox `{method}` call #{}; the script expected `{}`",
                        call.call_index + 1,
                        head.method
                    ),
                ));
            }
            head.matcher.clone()
        };
        if let Some(matcher) = matcher
            && !matcher(&call)
        {
            return Err(script_error(
                "matcher_rejected",
                format!(
                    "the scripted sandbox step for `{method}` rejected call #{}",
                    call.call_index + 1
                ),
            )
            .with_context("call_index", call.call_index.to_string())
            .with_context("method", method.as_str()));
        }
        let step = self.steps.lock().expect("scripted session lock").remove(0);

        match step.outcome {
            StepOutcome::Value(value) => Ok(value),
            StepOutcome::Responder(responder) => responder(&call),
            StepOutcome::Error(error) => Err(error),
        }
    }

    /// Fail a call whose scripted answer is the wrong shape.
    fn wrong_result(method: SandboxMethod, expected: &str) -> SandboxError {
        script_error(
            "wrong_result",
            format!("the scripted sandbox step for `{method}` must return {expected}"),
        )
    }
}

#[async_trait]
impl SandboxSession for ScriptedSandboxSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn live_grants(&self) -> Option<&crate::sandbox::session::LiveGrants> {
        Some(&self.live_grants)
    }

    fn backend_id(&self) -> &str {
        "scripted"
    }

    fn supports_pty(&self) -> bool {
        self.pty
    }

    async fn start(&self) -> SandboxResult<()> {
        self.running.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn stop(&self) -> SandboxResult<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn shutdown(&self) -> SandboxResult<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn running(&self) -> SandboxResult<bool> {
        Ok(self.running.load(Ordering::SeqCst))
    }

    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        match self.invoke(SandboxMethod::Exec, request.resolved_argv(), None)? {
            SandboxCallResult::Exec(result) => Ok(result),
            SandboxCallResult::Unit => Ok(ExecResult {
                exit_code: 0,
                stdout: Vec::new(),
                stderr: Vec::new(),
            }),
            _ => Err(Self::wrong_result(SandboxMethod::Exec, "an exec result")),
        }
    }

    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        match self.invoke(SandboxMethod::ReadFile, vec![path.to_string()], None)? {
            SandboxCallResult::Bytes(bytes) => Ok(bytes),
            _ => Err(Self::wrong_result(SandboxMethod::ReadFile, "bytes")),
        }
    }

    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.invoke(
            SandboxMethod::WriteFile,
            vec![path.to_string()],
            Some(contents.to_vec()),
        )?;
        Ok(())
    }

    async fn mkdir(&self, path: &str, parents: bool) -> SandboxResult<()> {
        self.invoke(
            SandboxMethod::Mkdir,
            vec![path.to_string(), parents.to_string()],
            None,
        )?;
        Ok(())
    }

    async fn rm(&self, path: &str, recursive: bool) -> SandboxResult<()> {
        self.invoke(
            SandboxMethod::Rm,
            vec![path.to_string(), recursive.to_string()],
            None,
        )?;
        Ok(())
    }

    async fn mv(&self, source: &str, destination: &str) -> SandboxResult<()> {
        self.invoke(
            SandboxMethod::Mv,
            vec![source.to_string(), destination.to_string()],
            None,
        )?;
        Ok(())
    }

    async fn ls(&self, path: &str) -> SandboxResult<Vec<FileEntry>> {
        match self.invoke(SandboxMethod::Ls, vec![path.to_string()], None)? {
            SandboxCallResult::Entries(entries) => Ok(entries),
            SandboxCallResult::Unit => Ok(Vec::new()),
            _ => Err(Self::wrong_result(SandboxMethod::Ls, "directory entries")),
        }
    }

    async fn pty_exec_start(&self, request: PtyExecRequest) -> SandboxResult<PtyExecUpdate> {
        match self.invoke(SandboxMethod::PtyExecStart, request.resolved_argv(), None)? {
            SandboxCallResult::Pty(update) => Ok(update),
            _ => Err(Self::wrong_result(
                SandboxMethod::PtyExecStart,
                "a PTY update",
            )),
        }
    }

    async fn pty_write_stdin(&self, request: PtyWriteRequest) -> SandboxResult<PtyExecUpdate> {
        match self.invoke(
            SandboxMethod::PtyWriteStdin,
            vec![request.session_id.to_string()],
            Some(request.chars.clone().into_bytes()),
        )? {
            SandboxCallResult::Pty(update) => Ok(update),
            _ => Err(Self::wrong_result(
                SandboxMethod::PtyWriteStdin,
                "a PTY update",
            )),
        }
    }

    async fn persist_workspace(&self) -> SandboxResult<Vec<u8>> {
        match self.invoke(SandboxMethod::PersistWorkspace, Vec::new(), None)? {
            SandboxCallResult::Bytes(bytes) => Ok(bytes),
            SandboxCallResult::Unit => Ok(Vec::new()),
            _ => Err(Self::wrong_result(
                SandboxMethod::PersistWorkspace,
                "archive bytes",
            )),
        }
    }

    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.invoke(
            SandboxMethod::HydrateWorkspace,
            Vec::new(),
            Some(archive.to_vec()),
        )?;
        Ok(())
    }
}

/// A sandbox session that answers from `steps`. Mirrors Python's `scripted_sandbox_session`.
pub fn scripted_sandbox_session(
    steps: impl IntoIterator<Item = SandboxStep>,
) -> ScriptedSandboxSession {
    ScriptedSandboxSession::new(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(stdout: &str) -> ExecResult {
        ExecResult {
            exit_code: 0,
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
        }
    }

    #[tokio::test]
    async fn answers_calls_in_order_and_records_them() {
        let session = scripted_sandbox_session([
            SandboxStep::new(SandboxMethod::WriteFile)
                .matching(|call| call.payload_text() == "hello"),
            SandboxStep::new(SandboxMethod::Exec).returning_exec(ok("done\n")),
        ]);

        session.write_file("/w/a.txt", b"hello").await.unwrap();
        let result = session
            .exec(ExecRequest::argv(["cat", "/w/a.txt"]))
            .await
            .unwrap();
        assert_eq!(result.stdout_text(), "done\n");

        let calls = session.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].method, SandboxMethod::WriteFile);
        assert!(calls[1].has_argument("cat"));
        session.assert_complete();
    }

    #[tokio::test]
    async fn an_unscripted_call_fails_where_it_happens() {
        let session = scripted_sandbox_session([SandboxStep::new(SandboxMethod::Exec)]);
        let error = session.read_file("/w/a.txt").await.unwrap_err();
        assert_eq!(
            error.context.get("script").map(String::as_str),
            Some("unexpected_call")
        );
        assert!(
            error.message.contains("expected `exec`"),
            "{}",
            error.message
        );
    }

    #[tokio::test]
    async fn a_call_past_the_end_of_the_script_fails() {
        let session = scripted_sandbox_session([]);
        let error = session.mkdir("/w", true).await.unwrap_err();
        assert!(error.message.contains("no scripted steps remain"));
    }

    #[tokio::test]
    async fn a_rejected_matcher_names_the_call() {
        let session =
            scripted_sandbox_session([SandboxStep::new(SandboxMethod::Mkdir)
                .matching(|call| call.has_argument("/expected"))]);
        let error = session.mkdir("/other", true).await.unwrap_err();
        assert_eq!(
            error.context.get("script").map(String::as_str),
            Some("matcher_rejected")
        );
    }

    #[tokio::test]
    async fn a_responder_sees_the_call() {
        let session = scripted_sandbox_session([SandboxStep::new(SandboxMethod::ReadFile)
            .responding(|call| {
                Ok(SandboxCallResult::Bytes(
                    call.arguments[0].clone().into_bytes(),
                ))
            })]);
        let bytes = session.read_file("/w/echo").await.unwrap();
        assert_eq!(bytes, b"/w/echo");
    }

    #[tokio::test]
    async fn a_scripted_error_reaches_the_caller() {
        let session = scripted_sandbox_session([SandboxStep::new(SandboxMethod::ReadFile)
            .failing(SandboxError::new(ErrorCode::WorkspaceReadNotFound, "gone"))]);
        let error = session.read_file("/w/missing").await.unwrap_err();
        assert_eq!(error.code, ErrorCode::WorkspaceReadNotFound);
    }

    #[test]
    fn unconsumed_steps_are_reported() {
        let session = scripted_sandbox_session([SandboxStep::new(SandboxMethod::Exec)]);
        assert_eq!(session.remaining_steps(), 1);
        let error = session.complete().unwrap_err();
        assert!(error.message.contains("unconsumed"), "{}", error.message);
    }
}
