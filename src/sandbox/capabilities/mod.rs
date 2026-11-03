//! Sandbox capabilities: the things a sandbox agent can do. Mirrors
//! `agents.sandbox.capabilities`.
//!
//! A capability can mutate the manifest, contribute an instruction fragment, add model request
//! parameters, transform the model input and expose tools bound to the live session.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::error::Result;
use crate::items::InputItem;
use crate::sandbox::manifest::Manifest;
use crate::sandbox::session::SandboxSession;
use crate::sandbox::types::User;
use crate::sandbox::workspace_paths::SandboxWorkspaceScope;
use crate::tool::Tool;

pub mod apply_patch_tool;
pub mod compaction;
pub mod filesystem;
pub mod memory;
pub mod pty_tools;
pub mod shell;
pub mod skills;
pub(crate) mod tool_input_errors;

pub use apply_patch_tool::SandboxApplyPatchTool;
pub use compaction::{Compaction, CompactionPolicy, context_window_for_model};
pub use filesystem::Filesystem;
pub use memory::Memory;
pub use shell::Shell;
pub use skills::{LazySkillSource, LocalDirLazySkillSource, Skill, SkillMetadata, Skills};

/// Everything a capability needs to build session-bound tools for one run.
#[derive(Debug, Clone)]
pub struct CapabilityBinding {
    /// The live sandbox session.
    pub session: Arc<dyn SandboxSession>,
    /// The sandbox user model-facing tools run as.
    pub run_as: Option<User>,
    /// The model-facing relative-path base for the run.
    pub scope: SandboxWorkspaceScope,
}

impl CapabilityBinding {
    /// A binding anchored at the workspace root, running as the sandbox default user.
    pub fn new(session: Arc<dyn SandboxSession>) -> Self {
        CapabilityBinding {
            session,
            run_as: None,
            scope: SandboxWorkspaceScope::root(),
        }
    }

    /// Set the sandbox user model-facing tools run as.
    pub fn with_run_as(mut self, run_as: Option<User>) -> Self {
        self.run_as = run_as;
        self
    }

    /// Set the model-facing relative-path base.
    pub fn with_scope(mut self, scope: SandboxWorkspaceScope) -> Self {
        self.scope = scope;
        self
    }
}

/// One sandbox capability.
///
/// Unlike the Python capability, which is bound to a session by mutation, a Rust capability is
/// stateless: the binding is passed to [`tools`](Capability::tools) for each run.
#[async_trait]
pub trait Capability<C>: Send + Sync + fmt::Debug
where
    C: Send + Sync + 'static,
{
    /// The capability type, matching the Python `type` discriminator.
    fn capability_type(&self) -> &str;

    /// Capability types that must be present alongside this capability.
    fn required_capability_types(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// Adjust the manifest used for a fresh session.
    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        manifest
    }

    /// Report a configuration problem [`process_manifest`](Self::process_manifest) could not
    /// raise, because it cannot fail. Called with the session's manifest when an agent is
    /// prepared. The default accepts everything.
    fn check_manifest(&self, manifest: &Manifest) -> Result<()> {
        let _ = manifest;
        Ok(())
    }

    /// Tools exposed to the model, bound to the live session.
    fn tools(&self, binding: &CapabilityBinding) -> Result<Vec<Tool<C>>> {
        let _ = binding;
        Ok(Vec::new())
    }

    /// A deterministic instruction fragment appended during run preparation.
    async fn instructions(&self, manifest: &Manifest) -> Result<Option<String>> {
        let _ = manifest;
        Ok(None)
    }

    /// An instruction fragment that needs the live session, appended after
    /// [`instructions`](Self::instructions).
    ///
    /// The memory capability uses this to read the stored memory summary out of the workspace.
    async fn bound_instructions(
        &self,
        binding: &CapabilityBinding,
        manifest: &Manifest,
    ) -> Result<Option<String>> {
        let _ = (binding, manifest);
        Ok(None)
    }

    /// Whether the instruction fragment is rendered once per prepared agent.
    ///
    /// The fragments are rendered again for every model call by default, as Python renders them
    /// inside the prepared agent's instructions callable, so a fragment that reads the live
    /// workspace (the memory summary) shows what an earlier turn wrote. A capability whose
    /// fragment is stable for the run returns `true` to render it on the first call only, as
    /// Python's skills capability caches its index.
    fn caches_instructions(&self) -> bool {
        false
    }

    /// The memory capability behind this value, when it is one.
    ///
    /// Lets the runner find the memory manager to flush without downcasting trait objects.
    fn as_memory(&self) -> Option<&Memory> {
        None
    }

    /// Extra model request parameters, merged into `ModelSettings::extra_args`.
    ///
    /// `sampling_params` holds the parameters accumulated so far, including a `model` key when
    /// the run resolves to a named model.
    fn sampling_params(&self, sampling_params: &Map<String, Value>) -> Map<String, Value> {
        let _ = sampling_params;
        Map::new()
    }

    /// Transform the model input before each model call.
    fn process_context(&self, context: Vec<InputItem>) -> Vec<InputItem> {
        context
    }
}

/// Replaces or wraps the tools a capability exposes. Mirrors Python's
/// `FilesystemToolConfigurator` / `configure_tools`.
///
/// The bundled capabilities expose a fixed set of tools, which is right until an application
/// needs one of them renamed, narrowed, wrapped in an approval, or dropped. Rewriting the
/// capability to change one tool loses every other thing it does, so this wraps it instead: the
/// manifest changes, the instructions and the sampling parameters all still come from the
/// capability, and only the tool list passes through the configurator.
///
/// ```no_run
/// use std::sync::Arc;
///
/// use openai_agents_sdk::sandbox::capabilities::{ConfiguredCapability, Shell};
///
/// // Expose the shell capability with every tool whose name starts with `exec_` removed.
/// let narrowed = ConfiguredCapability::<()>::new(Arc::new(Shell::new()), |tools| {
///     tools
///         .into_iter()
///         .filter(|tool| !tool.name().starts_with("exec_"))
///         .collect()
/// });
/// let _capability: Arc<dyn openai_agents_sdk::sandbox::capabilities::Capability<()>> =
///     Arc::new(narrowed);
/// ```
pub struct ConfiguredCapability<C>
where
    C: Send + Sync + 'static,
{
    inner: Arc<dyn Capability<C>>,
    configure: ToolConfigurator<C>,
}

/// Rewrites the tool list a capability produces.
pub type ToolConfigurator<C> = Arc<dyn Fn(Vec<Tool<C>>) -> Vec<Tool<C>> + Send + Sync>;

impl<C> ConfiguredCapability<C>
where
    C: Send + Sync + 'static,
{
    /// Wrap `inner`, passing its tools through `configure`.
    pub fn new<F>(inner: Arc<dyn Capability<C>>, configure: F) -> Self
    where
        F: Fn(Vec<Tool<C>>) -> Vec<Tool<C>> + Send + Sync + 'static,
    {
        ConfiguredCapability {
            inner,
            configure: Arc::new(configure),
        }
    }

    /// The capability being wrapped.
    pub fn inner(&self) -> &Arc<dyn Capability<C>> {
        &self.inner
    }
}

// Hand-written because `C` need not be `Clone` or `Debug` for the capability to be either.
impl<C> Clone for ConfiguredCapability<C>
where
    C: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        ConfiguredCapability {
            inner: self.inner.clone(),
            configure: self.configure.clone(),
        }
    }
}

impl<C> fmt::Debug for ConfiguredCapability<C>
where
    C: Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConfiguredCapability")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<C> Capability<C> for ConfiguredCapability<C>
where
    C: Send + Sync + 'static,
{
    fn capability_type(&self) -> &str {
        self.inner.capability_type()
    }

    fn required_capability_types(&self) -> Vec<&'static str> {
        self.inner.required_capability_types()
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        self.inner.process_manifest(manifest)
    }

    fn check_manifest(&self, manifest: &Manifest) -> Result<()> {
        self.inner.check_manifest(manifest)
    }

    fn tools(&self, binding: &CapabilityBinding) -> Result<Vec<Tool<C>>> {
        Ok((self.configure)(self.inner.tools(binding)?))
    }

    async fn instructions(&self, manifest: &Manifest) -> Result<Option<String>> {
        self.inner.instructions(manifest).await
    }

    async fn bound_instructions(
        &self,
        binding: &CapabilityBinding,
        manifest: &Manifest,
    ) -> Result<Option<String>> {
        self.inner.bound_instructions(binding, manifest).await
    }

    fn caches_instructions(&self) -> bool {
        self.inner.caches_instructions()
    }

    fn as_memory(&self) -> Option<&Memory> {
        self.inner.as_memory()
    }

    fn sampling_params(&self, sampling_params: &Map<String, Value>) -> Map<String, Value> {
        self.inner.sampling_params(sampling_params)
    }

    fn process_context(&self, context: Vec<InputItem>) -> Vec<InputItem> {
        self.inner.process_context(context)
    }
}

/// The capability sets the SDK ships.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities;

impl Capabilities {
    /// The default capability set: filesystem, shell and compaction.
    pub fn default_set<C: Send + Sync + 'static>() -> Vec<Arc<dyn Capability<C>>> {
        vec![
            Arc::new(Filesystem::new()),
            Arc::new(Shell::new()),
            Arc::new(Compaction::new()),
        ]
    }
}
