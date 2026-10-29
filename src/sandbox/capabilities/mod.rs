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

pub mod compaction;
pub mod filesystem;
pub mod shell;

pub use compaction::{Compaction, CompactionPolicy, context_window_for_model};
pub use filesystem::Filesystem;
pub use shell::Shell;

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

    /// Extra model request parameters, merged into `ModelSettings::extra_args`.
    ///
    /// `sampling_params` holds the parameters accumulated so far, including a `model` key when
    /// the run resolves to a named model.
    fn sampling_params(&self, sampling_params: &Map<String, Value>) -> Map<String, Value> {
        let _ = sampling_params;
        Map::new()
    }

    /// Transform the model input before the run starts.
    fn process_context(&self, context: Vec<InputItem>) -> Vec<InputItem> {
        context
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
