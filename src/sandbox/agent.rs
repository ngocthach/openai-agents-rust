//! Sandbox agents. Mirrors `agents.sandbox.sandbox_agent` and
//! `agents.sandbox.runtime_agent_preparation`.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::{Arc, OnceLock, Weak};

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
    pub sandbox_handoffs: Vec<SandboxHandoffTarget<C>>,
}

/// Where one sandbox handoff points.
///
/// A by-value [`Agent`](SandboxHandoffTarget::Agent) target owns its agent, which cannot express
/// a cycle. [`Named`](SandboxHandoffTarget::Named) closes one: it is resolved against the rest of
/// the graph when it is prepared.
pub enum SandboxHandoffTarget<C> {
    /// A sandbox agent owned by this handoff.
    Agent(Arc<SandboxAgent<C>>),
    /// Another agent in the same graph, resolved by name when the graph is prepared.
    Named(String),
}

impl<C> Clone for SandboxHandoffTarget<C> {
    fn clone(&self) -> Self {
        match self {
            SandboxHandoffTarget::Agent(agent) => SandboxHandoffTarget::Agent(agent.clone()),
            SandboxHandoffTarget::Named(name) => SandboxHandoffTarget::Named(name.clone()),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for SandboxHandoffTarget<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SandboxHandoffTarget::Agent(agent) => {
                f.debug_tuple("Agent").field(&agent.name()).finish()
            }
            SandboxHandoffTarget::Named(name) => f.debug_tuple("Named").field(name).finish(),
        }
    }
}

impl<C: Send + Sync + 'static> SandboxHandoffTarget<C> {
    /// The name of the agent this handoff points at.
    pub fn name(&self) -> &str {
        match self {
            SandboxHandoffTarget::Agent(agent) => agent.name(),
            SandboxHandoffTarget::Named(name) => name,
        }
    }

    /// The owned agent, when this target carries one.
    pub fn agent(&self) -> Option<&Arc<SandboxAgent<C>>> {
        match self {
            SandboxHandoffTarget::Agent(agent) => Some(agent),
            SandboxHandoffTarget::Named(_) => None,
        }
    }
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
                    .map(|t| t.name().to_string())
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
        self.sandbox_handoffs
            .push(SandboxHandoffTarget::Agent(target.into()));
        self
    }

    /// Add a handoff to another agent in the same graph, by name.
    ///
    /// This is how a cycle is expressed: `A` owns `B` through
    /// [`with_sandbox_handoff`](Self::with_sandbox_handoff), and `B` points back at `A` by name.
    /// The name must belong to an agent reachable in the graph, or preparation fails.
    pub fn with_sandbox_handoff_to(mut self, agent_name: impl Into<String>) -> Self {
        self.sandbox_handoffs
            .push(SandboxHandoffTarget::Named(agent_name.into()));
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
    /// own capabilities and `run_as`, bound to the same session.
    ///
    /// Cyclic graphs are supported. [`RunState`](crate::RunState) resumes a paused run by
    /// identifying the current agent by name and finding it through the handoff targets of the
    /// starting agent, so every node has to be reachable by a chain of owning handoffs. The graph
    /// is therefore prepared as a spanning tree: the edge that first discovers a node owns it,
    /// and the edges that close a cycle resolve their target lazily through a `Weak` reference.
    /// That keeps every node reachable for resume without creating a reference cycle that would
    /// leak the prepared agents and the sandbox session with them.
    ///
    /// Agent names must still be unique across the graph, because a resumed run has only the
    /// name to go on.
    pub async fn prepare_graph(
        &self,
        binding: &CapabilityBinding,
    ) -> Result<PreparedSandboxGraph<C>> {
        let root = Arc::new(self.clone());
        let mut nodes: HashMap<String, GraphNode<C>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        collect_graph(&root, &mut nodes, &mut order)?;

        // Named targets are the cycle-closing edges: resolve them now that every node is known.
        for node in nodes.values() {
            for target in &node.agent.sandbox_handoffs {
                if matches!(target, SandboxHandoffTarget::Named(_))
                    && !nodes.contains_key(target.name())
                {
                    return Err(AgentsError::user(format!(
                        "Sandbox handoff target `{}` is not part of the agent graph; add it with \
                         `with_sandbox_handoff` somewhere before pointing at it by name",
                        target.name()
                    )));
                }
            }
        }

        // One cell per node, holding a `Weak` so a cycle-closing edge never keeps its target
        // alive. The cells are filled once every node has been prepared.
        let cells: HashMap<String, Arc<OnceLock<Weak<Agent<C>>>>> = order
            .iter()
            .map(|name| (name.clone(), Arc::new(OnceLock::new())))
            .collect();

        // `order` is the post-order of the spanning tree, so a node's owned targets always exist
        // by the time it is prepared.
        let mut prepared: HashMap<String, Arc<Agent<C>>> = HashMap::new();
        for name in &order {
            let node = &nodes[name];
            let mut handoffs: Vec<Handoff<C>> = Vec::new();
            for (index, target) in node.agent.sandbox_handoffs.iter().enumerate() {
                let handoff = if node.owned_edges.contains(&index) {
                    let agent = prepared.get(target.name()).cloned().ok_or_else(|| {
                        AgentsError::other(format!(
                            "sandbox handoff target `{}` was not prepared",
                            target.name()
                        ))
                    })?;
                    Handoff::new(agent)
                } else {
                    weak_sandbox_handoff(target, cells[target.name()].clone())
                };
                // A named back-edge carries no agent of its own, so the description comes from
                // the node it points at.
                let described = nodes
                    .get(target.name())
                    .and_then(|n| n.agent.agent.handoff_description.clone());
                handoffs.push(describe_handoff(handoff, target, described.as_deref()));
            }
            let node_binding = binding.clone().with_run_as(node.agent.run_as.clone());
            let agent = Arc::new(
                node.agent
                    .prepare_with_handoffs(&node_binding, handoffs)
                    .await?,
            );
            prepared.insert(name.clone(), agent);
        }

        for (name, agent) in &prepared {
            // `set` only fails when the cell is already filled, which cannot happen here.
            let _ = cells[name].set(Arc::downgrade(agent));
        }

        let starting_agent = prepared
            .get(self.name())
            .cloned()
            .ok_or_else(|| AgentsError::other("sandbox starting agent was not prepared"))?;
        Ok(PreparedSandboxGraph {
            starting_agent,
            agents_by_name: nodes
                .into_iter()
                .map(|(name, node)| (name, node.agent))
                .collect(),
            prepared_by_name: prepared,
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
    /// Strong references to every prepared agent, so a node reachable only through a
    /// cycle-closing edge stays alive for as long as the graph does.
    prepared_by_name: HashMap<String, Arc<Agent<C>>>,
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
            prepared_by_name: self.prepared_by_name.clone(),
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

    /// The prepared agents in the graph, keyed by name.
    pub fn prepared_by_name(&self) -> &HashMap<String, Arc<Agent<C>>> {
        &self.prepared_by_name
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

/// One node of the sandbox handoff graph, with its handoff edges classified.
struct GraphNode<C> {
    agent: Arc<SandboxAgent<C>>,
    /// Indices of the handoff edges that own their target, forming the spanning tree.
    owned_edges: BTreeSet<usize>,
}

/// Walk the sandbox handoff graph depth-first, classifying each edge and producing the
/// post-order of the spanning tree.
///
/// The first edge to reach a node owns it; an edge to a node already on the stack (a cycle) or
/// already discovered elsewhere resolves lazily instead.
fn collect_graph<C: Send + Sync + 'static>(
    node: &Arc<SandboxAgent<C>>,
    nodes: &mut HashMap<String, GraphNode<C>>,
    order: &mut Vec<String>,
) -> Result<()> {
    let name = node.name().to_string();
    match nodes.get(&name) {
        Some(existing) if Arc::ptr_eq(&existing.agent, node) => return Ok(()),
        Some(_) => {
            return Err(AgentsError::user(format!(
                "Sandbox handoff graph contains two agents named `{name}`; agent names must be \
                 unique so a paused run can resume the right agent"
            )));
        }
        None => {}
    }
    // Register before recursing so a cycle back to this node is seen as already discovered.
    nodes.insert(
        name.clone(),
        GraphNode {
            agent: node.clone(),
            owned_edges: BTreeSet::new(),
        },
    );
    let mut owned_edges = BTreeSet::new();
    for (index, target) in node.sandbox_handoffs.iter().enumerate() {
        let Some(agent) = target.agent() else {
            // A named target is resolved after the whole graph is known.
            continue;
        };
        let first_visit = !nodes.contains_key(agent.name());
        collect_graph(agent, nodes, order)?;
        if first_visit {
            owned_edges.insert(index);
        }
    }
    if let Some(entry) = nodes.get_mut(&name) {
        entry.owned_edges = owned_edges;
    }
    order.push(name);
    Ok(())
}

/// Apply a handoff description to a handoff.
///
/// A named back-edge carries no agent of its own, so the caller looks the description up from the
/// graph node the edge points at.
fn describe_handoff<C: Send + Sync + 'static>(
    handoff: Handoff<C>,
    target: &SandboxHandoffTarget<C>,
    description: Option<&str>,
) -> Handoff<C> {
    match description {
        None => handoff,
        Some(description) => handoff.with_tool_description(
            crate::handoffs::default_tool_description(target.name(), Some(description)),
        ),
    }
}

/// A handoff that resolves its target through a `Weak`, used for the edges that close a cycle.
fn weak_sandbox_handoff<C: Send + Sync + 'static>(
    target: &SandboxHandoffTarget<C>,
    cell: Arc<OnceLock<Weak<Agent<C>>>>,
) -> Handoff<C> {
    let name = target.name().to_string();
    let description = target
        .agent()
        .and_then(|a| a.agent.handoff_description.clone());
    let resolved_name = name.clone();
    Handoff::custom(
        crate::handoffs::default_tool_name(&name),
        crate::handoffs::default_tool_description(&name, description.as_deref()),
        name,
        serde_json::json!({
            "type": "object", "properties": {}, "required": [], "additionalProperties": false
        }),
        move |_ctx, _args| {
            let cell = cell.clone();
            let resolved_name = resolved_name.clone();
            async move {
                cell.get().and_then(Weak::upgrade).ok_or_else(|| {
                    AgentsError::other(format!(
                        "sandbox handoff target `{resolved_name}` is no longer available; the \
                             prepared agent graph was dropped"
                    ))
                })
            }
        },
    )
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
