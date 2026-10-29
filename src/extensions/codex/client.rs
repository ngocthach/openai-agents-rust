//! The Codex client. Mirrors `agents.extensions.experimental.codex.codex`.

use std::sync::Arc;

use crate::extensions::codex::exec::CodexExec;
use crate::extensions::codex::options::CodexOptions;
use crate::extensions::codex::thread::Thread;
use crate::extensions::codex::thread_options::ThreadOptions;

/// Entry point for driving the Codex CLI: starts and resumes threads.
#[derive(Debug, Clone)]
pub struct Codex {
    exec: Arc<CodexExec>,
    options: CodexOptions,
}

impl Codex {
    /// Create a client with the given process options.
    pub fn new(options: CodexOptions) -> Self {
        let exec = CodexExec::new(options.codex_path_override.clone(), options.env.clone());
        Codex {
            exec: Arc::new(exec),
            options,
        }
    }

    /// The process options of this client.
    pub fn options(&self) -> &CodexOptions {
        &self.options
    }

    /// The resolved path of the `codex` binary.
    pub fn executable_path(&self) -> &str {
        self.exec.executable_path()
    }

    /// Start a new thread.
    pub fn start_thread(&self, options: Option<ThreadOptions>) -> Thread {
        Thread::new(
            self.exec.clone(),
            self.options.clone(),
            options.unwrap_or_default(),
            None,
        )
    }

    /// Resume an existing thread by id.
    pub fn resume_thread(
        &self,
        thread_id: impl Into<String>,
        options: Option<ThreadOptions>,
    ) -> Thread {
        Thread::new(
            self.exec.clone(),
            self.options.clone(),
            options.unwrap_or_default(),
            Some(thread_id.into()),
        )
    }
}

impl Default for Codex {
    fn default() -> Self {
        Codex::new(CodexOptions::default())
    }
}
