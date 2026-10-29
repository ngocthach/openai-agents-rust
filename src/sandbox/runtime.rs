//! Running a [`SandboxAgent`] through the ordinary [`Runner`]. Mirrors
//! `agents.sandbox.runtime` and `agents.sandbox.runtime_session_manager`.

use std::fmt;
use std::sync::Arc;

use crate::agent::Agent;
use crate::error::{AgentsError, Result};
use crate::items::RunInput;
use crate::result::{RunResult, RunResultStreaming};
use crate::run::{RunOptions, Runner};
use crate::sandbox::agent::SandboxAgent;
use crate::sandbox::capabilities::CapabilityBinding;
use crate::sandbox::manifest::Manifest;
use crate::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
use crate::sandbox::workspace_paths::SandboxWorkspaceScope;

/// Where a sandbox run executes, mirroring Python's `RunConfig(sandbox=...)` options.
///
/// Either [`client`](Self::client) or [`session`](Self::session) must be set: a client creates
/// (and tears down) a fresh session for the run, while a session is reused as-is and never torn
/// down by the runner.
#[derive(Debug, Clone, Default)]
pub struct SandboxRunConfig {
    /// Sandbox client used to create a fresh session for the run.
    pub client: Option<Arc<dyn SandboxClient>>,
    /// A live sandbox session to reuse instead of creating one.
    pub session: Option<Arc<dyn SandboxSession>>,
    /// Manifest override for a fresh session, taking precedence over
    /// [`SandboxAgent::default_manifest`].
    pub manifest: Option<Manifest>,
    /// Model-facing working directory, relative to the sandbox workspace root.
    ///
    /// Relative paths used by the capability tools resolve from this directory. It changes
    /// relative-path resolution only: it does not confine the run to `cwd`, nor change
    /// [`Manifest::root`](Manifest::root).
    pub cwd: Option<String>,
}

impl SandboxRunConfig {
    /// A configuration that creates a fresh session with `client`.
    pub fn with_client(client: Arc<dyn SandboxClient>) -> Self {
        SandboxRunConfig {
            client: Some(client),
            ..Default::default()
        }
    }

    /// A configuration that reuses a live session owned by the caller.
    pub fn with_session(session: Arc<dyn SandboxSession>) -> Self {
        SandboxRunConfig {
            session: Some(session),
            ..Default::default()
        }
    }

    /// Override the manifest used for a fresh session.
    pub fn manifest(mut self, manifest: Manifest) -> Self {
        self.manifest = Some(manifest);
        self
    }

    /// Set the model-facing working directory.
    pub fn cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
}

/// Owns the teardown of a sandbox session for the duration of a run.
///
/// A session created from a [`SandboxRunConfig::client`] is shut down by
/// [`close`](Self::close); a session supplied through [`SandboxRunConfig::session`] is left
/// untouched, because the caller owns it.
#[derive(Debug, Clone)]
pub struct SandboxSessionGuard {
    session: Arc<dyn SandboxSession>,
    client: Option<Arc<dyn SandboxClient>>,
}

impl SandboxSessionGuard {
    /// The live session.
    pub fn session(&self) -> &Arc<dyn SandboxSession> {
        &self.session
    }

    /// Whether closing the guard tears the session down.
    pub fn owns_session(&self) -> bool {
        self.client.is_some()
    }

    /// Tear down the session when the guard owns it.
    pub async fn close(self) -> Result<()> {
        match self.client {
            None => Ok(()),
            Some(client) => Ok(client.delete(self.session).await?),
        }
    }
}

/// A [`SandboxAgent`] prepared against a live session, ready for [`Runner`].
pub struct PreparedSandboxRun<C> {
    /// The plain agent carrying the sandbox instructions and capability tools.
    pub agent: Agent<C>,
    /// The session teardown guard.
    pub guard: SandboxSessionGuard,
}

impl<C> fmt::Debug for PreparedSandboxRun<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSandboxRun")
            .field("agent", &self.agent.name)
            .field("session", &self.guard.session().id())
            .finish_non_exhaustive()
    }
}

impl<C> PreparedSandboxRun<C> {
    /// The live sandbox session the agent is bound to.
    pub fn session(&self) -> &Arc<dyn SandboxSession> {
        self.guard.session()
    }
}

/// Runs [`SandboxAgent`]s: creates or reuses a sandbox session, binds the capability tools to
/// it, hands the prepared agent to [`Runner`], and tears the session down afterwards.
///
/// Handoffs to other [`SandboxAgent`]s are not prepared automatically: only the starting agent is
/// bound to the session. Prepare handoff targets yourself with
/// [`SandboxAgent::prepare`] against the same session when a run needs them.
#[derive(Debug, Clone, Copy, Default)]
pub struct SandboxRunner;

impl SandboxRunner {
    /// Create or reuse the sandbox session and prepare the agent against it.
    ///
    /// The caller owns the returned [`SandboxSessionGuard`] and must `close()` it.
    pub async fn prepare<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        config: &SandboxRunConfig,
    ) -> Result<PreparedSandboxRun<C>> {
        let scope = SandboxWorkspaceScope::from_cwd(config.cwd.as_deref())?;
        let (session, client) = match (&config.session, &config.client) {
            (Some(session), _) => (session.clone(), None),
            (None, Some(client)) => {
                let manifest = agent.resolved_manifest(config.manifest.clone());
                let session = client.create(Some(manifest)).await?;
                if let Err(error) = session.start().await {
                    // Release the backend resources the failed start may already hold.
                    let _ = client.delete(session.clone()).await;
                    return Err(error.into());
                }
                (session, Some(client.clone()))
            }
            (None, None) => {
                return Err(AgentsError::user(
                    "SandboxAgent execution requires SandboxRunConfig::client or \
                     SandboxRunConfig::session",
                ));
            }
        };
        let guard = SandboxSessionGuard {
            session: session.clone(),
            client,
        };
        if let Err(error) = validate_workspace_scope(session.as_ref(), &scope, agent).await {
            let _ = guard.close().await;
            return Err(error);
        }
        let binding = CapabilityBinding::new(session)
            .with_run_as(agent.run_as.clone())
            .with_scope(scope);
        match agent.prepare(&binding).await {
            Ok(prepared) => Ok(PreparedSandboxRun {
                agent: prepared,
                guard,
            }),
            Err(error) => {
                let _ = guard.close().await;
                Err(error)
            }
        }
    }

    /// Run a sandbox agent with a default app context.
    pub async fn run<C: Default + Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
    ) -> Result<RunResult<C>> {
        Self::run_with_options(agent, input, config, RunOptions::default()).await
    }

    /// Run a sandbox agent with explicit [`RunOptions`].
    ///
    /// The session is torn down before this returns, including when the run fails. A run that
    /// ends with approval interruptions therefore cannot be resumed against the same workspace
    /// unless the session was supplied through [`SandboxRunConfig::session`], or the run was set
    /// up with [`prepare`](Self::prepare).
    pub async fn run_with_options<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
        options: RunOptions<C>,
    ) -> Result<RunResult<C>> {
        let prepared = Self::prepare(agent, config).await?;
        let input = prepare_input(agent, input.into());
        let result = Runner::run_with_options(prepared.agent, input, options).await;
        if let Err(error) = prepared.guard.close().await {
            log::warn!("Failed to tear down the sandbox session: {error}");
        }
        result
    }

    /// Run a sandbox agent in streaming mode.
    ///
    /// The returned guard must be closed once the stream is drained; tearing the session down
    /// earlier would pull the workspace out from under the running tools.
    pub async fn run_streamed_with_options<C: Send + Sync + 'static>(
        agent: &SandboxAgent<C>,
        input: impl Into<RunInput>,
        config: &SandboxRunConfig,
        options: RunOptions<C>,
    ) -> Result<(RunResultStreaming<C>, SandboxSessionGuard)> {
        let prepared = Self::prepare(agent, config).await?;
        let input = prepare_input(agent, input.into());
        let streaming = Runner::run_streamed_with_options(prepared.agent, input, options);
        Ok((streaming, prepared.guard))
    }
}

fn prepare_input<C: Send + Sync + 'static>(agent: &SandboxAgent<C>, input: RunInput) -> RunInput {
    match input {
        RunInput::Items(items) => RunInput::Items(agent.prepare_input(items)),
        other => other,
    }
}

/// Confirm the configured working directory exists and is traversable for the sandbox user.
async fn validate_workspace_scope<C: Send + Sync + 'static>(
    session: &dyn SandboxSession,
    scope: &SandboxWorkspaceScope,
    agent: &SandboxAgent<C>,
) -> Result<()> {
    let Some(cwd) = scope.cwd() else {
        return Ok(());
    };
    let resolved = session.normalize_path(cwd)?;
    for flag in ["-d", "-x"] {
        let result = session
            .exec(ExecRequest::argv(["test", flag, &resolved]).with_user(agent.run_as.clone()))
            .await?;
        if !result.ok() {
            return Err(AgentsError::user(format!(
                "Sandbox working directory `{cwd}` does not exist or is not accessible for the \
                 configured sandbox user"
            )));
        }
    }
    Ok(())
}
