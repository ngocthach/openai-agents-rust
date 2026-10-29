//! Sandbox agents. Mirrors `agents.sandbox.sandbox_agent` and
//! `agents.sandbox.runtime_agent_preparation`.

use std::fmt;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::agent::{Agent, AgentModel, Instructions};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;
use crate::sandbox::capabilities::{Capabilities, Capability, CapabilityBinding};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::types::User;
use crate::sandbox::util::deep_merge;
use crate::tool::Tool;

/// The SDK sandbox base prompt, used when
/// [`base_instructions`](SandboxAgent::base_instructions) is not set.
pub const DEFAULT_SANDBOX_INSTRUCTIONS: &str = include_str!("instructions/prompt.md");

/// An [`Agent`] plus the sandbox configuration a run needs: the manifest for a fresh workspace,
/// the capabilities exposed to the model, and the sandbox user tools run as.
///
/// Transport details (which sandbox client, which live session) are provided at run time through
/// [`SandboxRunConfig`](crate::sandbox::SandboxRunConfig), not stored on the agent.
///
/// Call [`prepare`](SandboxAgent::prepare) to turn a sandbox agent plus a live session into a
/// plain [`Agent`] that [`Runner`](crate::Runner) can run, or use
/// [`SandboxRunner`](crate::sandbox::SandboxRunner) to do both plus session lifecycle management.
pub struct SandboxAgent<C> {
    /// The underlying agent: name, model, tools, guardrails, handoffs and hooks.
    pub agent: Agent<C>,
    /// Manifest materialized into workspaces created for this agent.
    pub default_manifest: Option<Manifest>,
    /// Override for the SDK sandbox base prompt. Most callers should use `agent.instructions`.
    pub base_instructions: Option<String>,
    /// Capabilities that can mutate the manifest, add instructions and expose tools.
    pub capabilities: Vec<Arc<dyn Capability<C>>>,
    /// Sandbox user identity used for model-facing tools such as shell and apply_patch.
    pub run_as: Option<User>,
}

impl<C> Clone for SandboxAgent<C> {
    fn clone(&self) -> Self {
        SandboxAgent {
            agent: self.agent.clone(),
            default_manifest: self.default_manifest.clone(),
            base_instructions: self.base_instructions.clone(),
            capabilities: self.capabilities.clone(),
            run_as: self.run_as.clone(),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for SandboxAgent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxAgent")
            .field("agent", &self.agent)
            .field("default_manifest", &self.default_manifest)
            .field(
                "capabilities",
                &self
                    .capabilities
                    .iter()
                    .map(|c| c.capability_type().to_string())
                    .collect::<Vec<_>>(),
            )
            .field("run_as", &self.run_as)
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> SandboxAgent<C> {
    /// A sandbox agent with the default capability set (filesystem, shell, compaction).
    pub fn new(name: impl Into<String>) -> Self {
        SandboxAgent {
            agent: Agent::new(name),
            default_manifest: None,
            base_instructions: None,
            capabilities: Capabilities::default_set::<C>(),
            run_as: None,
        }
    }

    /// Wrap an existing agent, keeping the default capability set.
    pub fn from_agent(agent: Agent<C>) -> Self {
        SandboxAgent {
            agent,
            default_manifest: None,
            base_instructions: None,
            capabilities: Capabilities::default_set::<C>(),
            run_as: None,
        }
    }

    /// Set the agent instructions, appended to the sandbox base prompt.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.agent = self.agent.with_instructions(instructions);
        self
    }

    /// Set the model used for the run.
    pub fn with_model(mut self, model: impl Into<AgentModel>) -> Self {
        self.agent = self.agent.with_model(model);
        self
    }

    /// Use a concrete model instance.
    pub fn with_model_instance(mut self, model: Arc<dyn crate::models::interface::Model>) -> Self {
        self.agent = self.agent.with_model_instance(model);
        self
    }

    /// Add an application tool, alongside the capability tools.
    pub fn with_tool(mut self, tool: impl Into<Tool<C>>) -> Self {
        self.agent = self.agent.with_tool(tool);
        self
    }

    /// Set the manifest materialized into fresh workspaces.
    pub fn with_default_manifest(mut self, manifest: Manifest) -> Self {
        self.default_manifest = Some(manifest);
        self
    }

    /// Replace the SDK sandbox base prompt.
    pub fn with_base_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.base_instructions = Some(instructions.into());
        self
    }

    /// Replace the capability set.
    pub fn with_capabilities(mut self, capabilities: Vec<Arc<dyn Capability<C>>>) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Add one capability to the set.
    pub fn with_capability(mut self, capability: Arc<dyn Capability<C>>) -> Self {
        self.capabilities.push(capability);
        self
    }

    /// Set the sandbox user model-facing tools run as.
    pub fn with_run_as(mut self, run_as: impl Into<User>) -> Self {
        self.run_as = Some(run_as.into());
        self
    }

    /// Modify the underlying agent in place.
    pub fn map_agent(mut self, f: impl FnOnce(Agent<C>) -> Agent<C>) -> Self {
        self.agent = f(self.agent);
        self
    }

    /// The agent name.
    pub fn name(&self) -> &str {
        &self.agent.name
    }

    /// The manifest a fresh session should materialize, after every capability has processed it.
    pub fn resolved_manifest(&self, override_manifest: Option<Manifest>) -> Manifest {
        let mut manifest = override_manifest
            .or_else(|| self.default_manifest.clone())
            .unwrap_or_default();
        for capability in &self.capabilities {
            manifest = capability.process_manifest(manifest);
        }
        manifest
    }

    /// The model name used to resolve capability sampling parameters.
    pub fn resolved_model_name(&self) -> Option<String> {
        match &self.agent.model {
            Some(AgentModel::Name(name)) => Some(name.clone()),
            Some(AgentModel::Instance(_)) | None => None,
        }
    }

    /// Apply every capability's `process_context` to the run input.
    pub fn prepare_input(&self, input: Vec<InputItem>) -> Vec<InputItem> {
        let mut items = input;
        for capability in &self.capabilities {
            items = capability.process_context(items);
        }
        items
    }

    /// Build a plain [`Agent`] bound to a live sandbox session.
    ///
    /// The prepared agent carries the sandbox instructions, the capability tools and the
    /// capability model request parameters, in addition to everything configured on
    /// [`agent`](Self::agent).
    pub async fn prepare(&self, binding: &CapabilityBinding) -> Result<Agent<C>> {
        let available: Vec<String> = self
            .capabilities
            .iter()
            .map(|c| c.capability_type().to_string())
            .collect();
        for capability in &self.capabilities {
            let missing: Vec<&str> = capability
                .required_capability_types()
                .into_iter()
                .filter(|required| !available.iter().any(|a| a == required))
                .collect();
            if !missing.is_empty() {
                return Err(AgentsError::user(format!(
                    "Sandbox capability `{}` requires missing capabilities: {}",
                    capability.capability_type(),
                    missing.join(", ")
                )));
            }
        }

        let mut capability_tools = Vec::new();
        for capability in &self.capabilities {
            capability_tools.extend(capability.tools(binding)?);
        }

        let mut extra_args: Map<String, Value> = self
            .agent
            .model_settings
            .extra_args
            .clone()
            .unwrap_or_default();
        let model_name = self.resolved_model_name();
        for capability in &self.capabilities {
            let mut params = extra_args.clone();
            if let Some(model) = &model_name {
                params.insert("model".into(), Value::String(model.clone()));
            }
            let overlay = capability.sampling_params(&params);
            deep_merge(&mut extra_args, &overlay);
        }

        let manifest = binding.session.manifest().clone();
        let mut fragments = Vec::new();
        for capability in &self.capabilities {
            if let Some(fragment) = capability.instructions(&manifest).await? {
                fragments.push(fragment);
            }
        }
        let instructions = self.build_instructions(fragments, &manifest, binding);

        let mut model_settings = self.agent.model_settings.clone();
        model_settings.extra_args = (!extra_args.is_empty()).then_some(extra_args);

        Ok(self.agent.clone_with(|agent| {
            agent.instructions = Some(instructions);
            agent.model_settings = model_settings;
            agent.tools.extend(capability_tools);
        }))
    }

    fn build_instructions(
        &self,
        fragments: Vec<String>,
        manifest: &Manifest,
        binding: &CapabilityBinding,
    ) -> Instructions<C> {
        let base = self
            .base_instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_SANDBOX_INSTRUCTIONS.trim().to_string());
        let capability_section = (!fragments.is_empty()).then(|| {
            instruction_section("Sandbox capability instructions", &fragments.join("\n\n"))
        });
        let filesystem_section = filesystem_instructions(manifest, binding);
        let original = self.agent.instructions.clone();
        Instructions::Dynamic(Arc::new(move |ctx, agent| {
            let base = base.clone();
            let capability_section = capability_section.clone();
            let filesystem_section = filesystem_section.clone();
            let original = original.clone();
            Box::pin(async move {
                let mut parts = Vec::new();
                if !base.trim().is_empty() {
                    parts.push(base);
                }
                let additional = match &original {
                    None => None,
                    Some(Instructions::Static(s)) => Some(s.clone()),
                    Some(Instructions::Dynamic(f)) => Some(f(ctx.clone(), agent.clone()).await?),
                };
                if let Some(additional) = additional.filter(|a| !a.trim().is_empty()) {
                    parts.push(instruction_section("Agent instructions", &additional));
                }
                if let Some(section) = capability_section {
                    parts.push(section);
                }
                parts.push(filesystem_section);
                Ok(parts.join("\n\n"))
            })
        }))
    }
}

fn instruction_section(title: &str, body: &str) -> String {
    format!("# {title}\n\n{body}")
}

/// The `# Filesystem` section describing the workspace layout to the model.
fn filesystem_instructions(manifest: &Manifest, binding: &CapabilityBinding) -> String {
    let mut header = vec![
        "# Filesystem".to_string(),
        "You have access to a container with a filesystem. The filesystem layout is:".to_string(),
    ];
    if let Some(cwd) = binding.scope.cwd() {
        let root = manifest.root().trim_end_matches('/');
        let working_directory = format!("{root}/{cwd}");
        header.extend([
            format!("For this run, the working directory is `{working_directory}`."),
            "Relative paths passed to the sandbox tools resolve from this directory.".to_string(),
            format!("The session workspace root remains `{root}`."),
            "The working directory changes path resolution; it does not isolate this run from \
             the rest of the session workspace."
                .to_string(),
        ]);
    }
    format!(
        "{}\n\n{}",
        header.join("\n"),
        manifest.describe().trim_end()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_context::RunContext;
    use crate::sandbox::capabilities::Shell;
    use crate::sandbox::entries::Entry;
    use crate::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
    use crate::sandbox::session::SandboxClient;
    use crate::sandbox::workspace_paths::SandboxWorkspaceScope;

    #[tokio::test]
    async fn prepares_an_agent_with_capability_tools_and_instructions() {
        let client = UnixLocalSandboxClient::new();
        let session = client
            .create(Some(
                Manifest::new().with_entry("repo", Entry::dir().with_description("the repo")),
            ))
            .await
            .unwrap();
        session.start().await.unwrap();

        let sandbox_agent = SandboxAgent::<()>::new("Sandbox engineer")
            .with_instructions("Read repo/task.md first")
            .with_model("gpt-5.5");
        let binding = CapabilityBinding::new(session.clone())
            .with_scope(SandboxWorkspaceScope::from_cwd(Some("repo")).unwrap());
        let prepared = Arc::new(sandbox_agent.prepare(&binding).await.unwrap());

        // apply_patch + view_image + shell.
        assert_eq!(prepared.tools.len(), 3);
        let extra = prepared.model_settings.extra_args.clone().unwrap();
        assert!(extra.contains_key("context_management"));

        let prompt = prepared
            .get_system_prompt(&RunContext::new(()))
            .await
            .unwrap()
            .unwrap();
        assert!(prompt.contains("# Agent instructions\n\nRead repo/task.md first"));
        assert!(prompt.contains("# Sandbox capability instructions"));
        assert!(prompt.contains("# Filesystem"));
        assert!(prompt.contains("the repo"));
        assert!(prompt.contains("the working directory is"));

        client.delete(session).await.unwrap();
    }

    #[tokio::test]
    async fn rejects_missing_required_capabilities() {
        #[derive(Debug)]
        struct NeedsMemory;

        #[async_trait::async_trait]
        impl Capability<()> for NeedsMemory {
            fn capability_type(&self) -> &str {
                "needs_memory"
            }
            fn required_capability_types(&self) -> Vec<&'static str> {
                vec!["memory"]
            }
        }

        let client = UnixLocalSandboxClient::new();
        let session = client.create(None).await.unwrap();
        session.start().await.unwrap();
        let agent = SandboxAgent::<()>::new("Agent")
            .with_capabilities(vec![Arc::new(Shell::new()), Arc::new(NeedsMemory)]);
        let binding = CapabilityBinding::new(session.clone());
        let err = agent.prepare(&binding).await.unwrap_err();
        assert!(err.to_string().contains("missing capabilities: memory"));
        client.delete(session).await.unwrap();
    }
}
