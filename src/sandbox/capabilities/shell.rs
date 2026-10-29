//! The shell capability. Mirrors `agents.sandbox.capabilities.shell`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::{AgentsError, Result};
use crate::sandbox::capabilities::{Capability, CapabilityBinding};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::session::ExecRequest;
use crate::tool::{
    ShellCallOutcome, ShellCommandOutput, ShellCommandRequest, ShellResult, ShellTool, Tool,
};

const SHELL_INSTRUCTIONS: &str = "When using the shell:
- Use the `shell` tool for shell execution; each call runs its commands in order.
- Commands run from the run working directory inside the sandbox workspace.
- Prefer `rg` and `rg --files` for text/file discovery when available.
- Avoid using Python scripts just to print large file chunks.";

/// Exposes the sandbox shell to the model as a [`ShellTool`].
///
/// Each `shell_call` command is executed through
/// [`SandboxSession::exec`](crate::sandbox::SandboxSession::exec) and reported back with its
/// stdout, stderr and exit code. A command that exceeds the call's `timeout_ms` is reported as
/// [`ShellCallOutcome::Timeout`] rather than failing the tool call.
#[derive(Debug, Clone, Default)]
pub struct Shell {
    tool_name: Option<String>,
    needs_approval: bool,
}

impl Shell {
    /// The shell capability with the default tool name (`shell`).
    pub fn new() -> Self {
        Shell::default()
    }

    /// Rename the tool exposed to the model.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    /// Require human approval for every shell call.
    pub fn with_needs_approval(mut self, needs_approval: bool) -> Self {
        self.needs_approval = needs_approval;
        self
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for Shell {
    fn capability_type(&self) -> &str {
        "shell"
    }

    fn tools(&self, binding: &CapabilityBinding) -> Result<Vec<Tool<C>>> {
        let session = binding.session.clone();
        let run_as = binding.run_as.clone();
        let cwd = binding.scope.cwd().map(str::to_string);
        let mut tool = ShellTool::<C>::new(move |request: ShellCommandRequest<C>| {
            let session = session.clone();
            let run_as = run_as.clone();
            let cwd = cwd.clone();
            async move {
                let action = &request.data.action;
                let timeout = action.timeout_ms.map(Duration::from_millis);
                let mut output = Vec::with_capacity(action.commands.len());
                for command in &action.commands {
                    let exec = ExecRequest::shell(command.clone())
                        .with_timeout(timeout)
                        .with_cwd(cwd.clone())
                        .with_user(run_as.clone());
                    match session.exec(exec).await {
                        Ok(result) => output.push(ShellCommandOutput {
                            stdout: result.stdout_text(),
                            stderr: result.stderr_text(),
                            outcome: ShellCallOutcome::Exit {
                                exit_code: Some(result.exit_code),
                            },
                            command: Some(command.clone()),
                            provider_data: None,
                        }),
                        Err(error) if error.is_timeout() => output.push(ShellCommandOutput {
                            outcome: ShellCallOutcome::Timeout,
                            command: Some(command.clone()),
                            ..Default::default()
                        }),
                        // A transport or path failure is not a command result: report it as a
                        // failed tool call instead of a fake exit status.
                        Err(error) => return Err(AgentsError::from(error)),
                    }
                }
                Ok::<_, AgentsError>(ShellResult {
                    output,
                    max_output_length: action.max_output_length,
                    provider_data: None,
                })
            }
        })
        .with_needs_approval(self.needs_approval);
        if let Some(name) = &self.tool_name {
            tool = tool.with_name(name.clone());
        }
        Ok(vec![Tool::Shell(Arc::new(tool))])
    }

    async fn instructions(&self, _manifest: &Manifest) -> Result<Option<String>> {
        Ok(Some(SHELL_INSTRUCTIONS.to_string()))
    }
}
