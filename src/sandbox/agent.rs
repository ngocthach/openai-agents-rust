//! Sandbox agents. Mirrors `agents.sandbox.sandbox_agent` and
//! `agents.sandbox.runtime_agent_preparation`.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::agent::{Agent, AgentModel, Instructions};
use crate::error::{AgentsError, Result};
use crate::handoffs::Handoff;
use crate::items::InputItem;
use crate::sandbox::capabilities::{Capabilities, Capability, CapabilityBinding};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::remote_mount_policy::build_remote_mount_policy_instructions;
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
    /// Other sandbox agents this agent can hand off to.
    ///
    /// Targets added here are prepared against the same sandbox session as the starting agent,
    /// each with its own capabilities and `run_as`. Plain [`Agent`] handoffs configured on
    /// [`agent`](Self::agent) still work; they simply get no capability tools.
    pub sandbox_handoffs: Vec<Arc<SandboxAgent<C>>>,
}

impl<C> Clone for SandboxAgent<C> {
    fn clone(&self) -> Self {
        SandboxAgent {
            agent: self.agent.clone(),
            default_manifest: self.default_manifest.clone(),
            base_instructions: self.base_instructions.clone(),
            capabilities: self.capabilities.clone(),
            run_as: self.run_as.clone(),
            sandbox_handoffs: self.sandbox_handoffs.clone(),
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
            .field(
                "sandbox_handoffs",
                &self
                    .sandbox_handoffs
                    .iter()
                    .map(|a| a.name().to_string())
                    .collect::<Vec<_>>(),
            )
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
            sandbox_handoffs: Vec::new(),
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
            sandbox_handoffs: Vec::new(),
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

    /// Add a sandbox agent this agent can hand off to.
    ///
    /// The target is prepared against the same session, so its capability tools act on the same
    /// workspace. Cycles are allowed: handoffs resolve lazily.
    pub fn with_sandbox_handoff(mut self, target: impl Into<Arc<SandboxAgent<C>>>) -> Self {
        self.sandbox_handoffs.push(target.into());
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

    /// The memory capability in the capability set, when one is configured.
    pub fn memory_capability(&self) -> Option<&crate::sandbox::capabilities::Memory> {
        self.capabilities
            .iter()
            .find_map(|capability| capability.as_memory())
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
        self.prepare_with_handoffs(binding, Vec::new()).await
    }

    /// Prepare the whole sandbox handoff graph against one session.
    ///
    /// Every reachable [`sandbox_handoffs`](Self::sandbox_handoffs) target is prepared with its
    /// own capabilities and `run_as`, bound to the same session, and wired in as a real handoff
    /// to the prepared target.
    ///
    /// Two constraints follow from how [`RunState`](crate::RunState) resumes a paused run: it
    /// identifies the current agent by name and finds it by walking the handoff targets of the
    /// starting agent. So agent names must be unique across the graph, and the graph must be
    /// acyclic — targets are prepared before the agents that hand off to them.
    pub async fn prepare_graph(
        &self,
        binding: &CapabilityBinding,
    ) -> Result<PreparedSandboxGraph<C>> {
        let root = Arc::new(self.clone());
        let mut by_name: HashMap<String, Arc<SandboxAgent<C>>> = HashMap::new();
        let mut order: Vec<Arc<SandboxAgent<C>>> = Vec::new();
        let mut path: Vec<String> = Vec::new();
        collect_graph(&root, &mut by_name, &mut order, &mut path)?;

        // `order` is post-order, so every handoff target is prepared before its referrer and can
        // be wired in by value.
        let mut prepared: HashMap<String, Arc<Agent<C>>> = HashMap::new();
        for node in &order {
            let handoffs: Vec<Handoff<C>> = node
                .sandbox_handoffs
                .iter()
                .map(|target| {
                    let agent = prepared.get(target.name()).cloned().ok_or_else(|| {
                        AgentsError::other(format!(
                            "sandbox handoff target `{}` was not prepared",
                            target.name()
                        ))
                    })?;
                    let mut handoff = Handoff::new(agent);
                    if let Some(description) = &target.agent.handoff_description {
                        handoff = handoff.with_tool_description(
                            crate::handoffs::default_tool_description(
                                target.name(),
                                Some(description),
                            ),
                        );
                    }
                    Ok(handoff)
                })
                .collect::<Result<Vec<_>>>()?;
            let node_binding = binding.clone().with_run_as(node.run_as.clone());
            let agent = Arc::new(node.prepare_with_handoffs(&node_binding, handoffs).await?);
            prepared.insert(node.name().to_string(), agent);
        }

        let starting_agent = prepared
            .get(self.name())
            .cloned()
            .ok_or_else(|| AgentsError::other("sandbox starting agent was not prepared"))?;
        Ok(PreparedSandboxGraph {
            starting_agent,
            agents_by_name: by_name,
        })
    }

    async fn prepare_with_handoffs(
        &self,
        binding: &CapabilityBinding,
        extra_handoffs: Vec<Handoff<C>>,
    ) -> Result<Agent<C>> {
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
            if let Some(fragment) = capability.bound_instructions(binding, &manifest).await? {
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
            agent.handoffs.extend(extra_handoffs);
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
        let mount_policy_section = build_remote_mount_policy_instructions(manifest)
            .map(|policy| instruction_section("Sandbox remote mount policy", &policy));
        let filesystem_section = filesystem_instructions(manifest, binding);
        let original = self.agent.instructions.clone();
        Instructions::Dynamic(Arc::new(move |ctx, agent| {
            let base = base.clone();
            let capability_section = capability_section.clone();
            let mount_policy_section = mount_policy_section.clone();
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
                if let Some(section) = mount_policy_section {
                    parts.push(section);
                }
                parts.push(filesystem_section);
                Ok(parts.join("\n\n"))
            })
        }))
    }
}

/// A sandbox agent graph prepared against one live session.
///
/// Holds the prepared starting agent plus the sandbox agent behind every node, so a per-turn
/// hook can look up which capability set owns the agent that is about to be called.
pub struct PreparedSandboxGraph<C> {
    starting_agent: Arc<Agent<C>>,
    agents_by_name: HashMap<String, Arc<SandboxAgent<C>>>,
}

impl<C> fmt::Debug for PreparedSandboxGraph<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names: Vec<&str> = self.agents_by_name.keys().map(String::as_str).collect();
        names.sort_unstable();
        f.debug_struct("PreparedSandboxGraph")
            .field("starting_agent", &self.starting_agent.name)
            .field("agents", &names)
            .finish()
    }
}

impl<C> Clone for PreparedSandboxGraph<C> {
    fn clone(&self) -> Self {
        PreparedSandboxGraph {
            starting_agent: self.starting_agent.clone(),
            agents_by_name: self.agents_by_name.clone(),
        }
    }
}

impl<C: Send + Sync + 'static> PreparedSandboxGraph<C> {
    /// The prepared agent the run starts from.
    pub fn starting_agent(&self) -> &Arc<Agent<C>> {
        &self.starting_agent
    }

    /// The sandbox agents in the graph, keyed by name.
    pub fn agents_by_name(&self) -> &HashMap<String, Arc<SandboxAgent<C>>> {
        &self.agents_by_name
    }

    /// Apply the capability `process_context` chain of `agent_name` to a turn's model input.
    ///
    /// Unknown agent names (plain handoff targets, nested tool agents) pass through unchanged.
    pub fn process_context(&self, agent_name: &str, input: Vec<InputItem>) -> Vec<InputItem> {
        match self.agents_by_name.get(agent_name) {
            None => input,
            Some(agent) => agent.prepare_input(input),
        }
    }
}

/// Walk the sandbox handoff graph depth-first, producing a post-order list.
///
/// `path` holds the names on the current DFS branch, so a cycle is reported with the loop it
/// closes instead of recursing forever.
fn collect_graph<C: Send + Sync + 'static>(
    node: &Arc<SandboxAgent<C>>,
    by_name: &mut HashMap<String, Arc<SandboxAgent<C>>>,
    order: &mut Vec<Arc<SandboxAgent<C>>>,
    path: &mut Vec<String>,
) -> Result<()> {
    let name = node.name().to_string();
    if let Some(index) = path.iter().position(|n| *n == name) {
        let mut loop_names = path[index..].to_vec();
        loop_names.push(name);
        return Err(AgentsError::user(format!(
            "Sandbox handoff graph contains a cycle: {}. A paused run resumes by walking handoff \
             targets, so the graph must be acyclic",
            loop_names.join(" -> ")
        )));
    }
    match by_name.get(&name) {
        // Already prepared on another branch: a diamond is fine, a name clash is not.
        Some(existing) if Arc::ptr_eq(existing, node) => return Ok(()),
        Some(_) => {
            return Err(AgentsError::user(format!(
                "Sandbox handoff graph contains two agents named `{name}`; agent names must be \
                 unique so a paused run can resume the right agent"
            )));
        }
        None => {}
    }
    path.push(name.clone());
    for target in &node.sandbox_handoffs {
        collect_graph(target, by_name, order, path)?;
    }
    path.pop();
    by_name.insert(name, node.clone());
    order.push(node.clone());
    Ok(())
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

        let tool_names: Vec<String> = prepared.tools.iter().map(|t| t.name()).collect();
        assert_eq!(
            tool_names,
            vec![
                "apply_patch",
                "view_image",
                "shell",
                "exec_command",
                "write_stdin"
            ]
        );
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
