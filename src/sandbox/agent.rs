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

    /// Expose this sandbox agent as a tool of another agent.
    ///
    /// Each call prepares the sandbox the way [`SandboxRunner`](crate::sandbox::SandboxRunner)
    /// does for a top-level run (capability tools, instructions, manifest, session from `config`)
    /// and tears it down when the nested run completes. Python reads that configuration from the
    /// parent run's `RunConfig.sandbox`; Rust's `RunConfig` has no such field, so it is given
    /// here.
    pub fn as_tool(
        &self,
        tool_name: impl Into<String>,
        tool_description: impl Into<String>,
        config: crate::sandbox::SandboxRunConfig,
    ) -> crate::tool::FunctionTool<C> {
        self.as_tool_with(
            crate::agent::AgentAsToolOptions::new()
                .tool_name(tool_name)
                .tool_description(tool_description),
            config,
        )
    }

    /// [`as_tool`](Self::as_tool) with full options. Panics on invalid options, like
    /// [`Agent::as_tool_with`].
    pub fn as_tool_with(
        &self,
        options: crate::agent::AgentAsToolOptions<C>,
        config: crate::sandbox::SandboxRunConfig,
    ) -> crate::tool::FunctionTool<C> {
        let sandbox_agent = self.clone();
        let runner: crate::agent::NestedRunFn<C> = Arc::new(move |input, run_opts| {
            let (agent, config) = (sandbox_agent.clone(), config.clone());
            Box::pin(async move {
                crate::sandbox::SandboxRunner::run_with_options(&agent, input, &config, run_opts)
                    .await
            })
        });
        self.agent
            .try_as_tool_with_runner(options, Some(runner))
            .unwrap_or_else(|error| panic!("{error}"))
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
        // Mirrors Python's `_manifest_with_run_as_user`. The account the agent's commands run as
        // has to exist before the first command runs, and `provision_accounts` only creates the
        // accounts the manifest declares.
        if let Some(run_as) = &self.run_as
            && !manifest.users().iter().any(|user| user.name == run_as.name)
            // A group member is created together with its group, so adding the account to
            // `users` as well would provision it twice.
            && !manifest
                .groups()
                .iter()
                .any(|group| group.users.iter().any(|user| user.name == run_as.name))
        {
            manifest = manifest.with_user(run_as.clone());
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
    /// Agents may share a name. The first agent of a name is keyed by it and the others as
    /// `name#2`, `name#3`, ... (skipping any key another agent's literal name already uses), so
    /// every map and per-agent session of the graph is keyed by these identity keys, which are
    /// plain names whenever names are unique.
    pub async fn prepare_graph(
        &self,
        binding: &CapabilityBinding,
    ) -> Result<PreparedSandboxGraph<C>> {
        self.prepare_graph_with_bindings(binding, &HashMap::new())
            .await
    }

    /// One agent of this agent's sandbox handoff graph, by name.
    pub fn graph_agent(&self, name: &str) -> Option<Arc<SandboxAgent<C>>> {
        let root = Arc::new(self.clone());
        let mut walk = GraphWalk::new(&root);
        collect_graph(&root, &mut walk).ok()?;
        walk.nodes.remove(name).map(|node| node.agent)
    }

    /// The names of every agent in this agent's sandbox handoff graph, in post order.
    ///
    /// A caller that wants one session per agent needs the names before it can create the
    /// sessions, which is why this is separate from preparation.
    pub fn graph_agent_names(&self) -> Result<Vec<String>> {
        let root = Arc::new(self.clone());
        let mut walk = GraphWalk::new(&root);
        collect_graph(&root, &mut walk)?;
        Ok(walk.order)
    }

    /// Prepare the graph, binding each named agent to its own session.
    ///
    /// Mirrors Python's per-agent sandbox sessions (`SandboxRuntimeSessionManager.acquire_agent`
    /// / `ensure_session`). An agent with no entry in `by_name` falls back to `binding`, so the
    /// default one-session-per-graph behaviour is `by_name` being empty.
    pub async fn prepare_graph_with_bindings(
        &self,
        binding: &CapabilityBinding,
        by_name: &HashMap<String, CapabilityBinding>,
    ) -> Result<PreparedSandboxGraph<C>> {
        let root = Arc::new(self.clone());
        let mut walk = GraphWalk::new(&root);
        collect_graph(&root, &mut walk)?;
        let GraphWalk {
            nodes, order, keys, ..
        } = walk;

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
                let target_key = target_key(target, &keys);
                let handoff = if node.owned_edges.contains(&index) {
                    let agent = prepared.get(&target_key).cloned().ok_or_else(|| {
                        AgentsError::other(format!(
                            "sandbox handoff target `{}` was not prepared",
                            target.name()
                        ))
                    })?;
                    Handoff::new(agent)
                } else {
                    weak_sandbox_handoff(target, cells[&target_key].clone())
                };
                // A named back-edge carries no agent of its own, so the description comes from
                // the node it points at.
                let described = nodes
                    .get(&target_key)
                    .and_then(|n| n.agent.agent.handoff_description.clone());
                handoffs.push(describe_handoff(handoff, target, described.as_deref()));
            }
            let node_binding = by_name
                .get(name)
                .unwrap_or(binding)
                .clone()
                .with_run_as(node.agent.run_as.clone());
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
        let keys_by_prepared_ptr = prepared
            .iter()
            .map(|(key, agent)| (Arc::as_ptr(agent) as usize, key.clone()))
            .collect();
        Ok(PreparedSandboxGraph {
            starting_agent,
            keys_by_prepared_ptr,
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
            let mut missing: Vec<&str> = capability
                .required_capability_types()
                .into_iter()
                .filter(|required| !available.iter().any(|a| a == required))
                .collect();
            if !missing.is_empty() {
                // Sorted, and named by the capability's type the way Python names its class, so
                // the message is the same however the requirements were declared.
                missing.sort_unstable();
                missing.dedup();
                return Err(AgentsError::user(format!(
                    "{} requires missing capabilities: {}",
                    pascal_case(capability.capability_type()),
                    missing.join(", ")
                )));
            }
        }

        for capability in &self.capabilities {
            if let Err(error) = capability.check_manifest(binding.session.manifest()) {
                // A capability's message can repeat the manifest it rejected, credentials of
                // its mounts included, so a protected manifest surfaces only the fixed
                // sentence.
                return Err(
                    if crate::sandbox::sandboxes::docker::manifest_has_protected_mounts(
                        binding.session.manifest(),
                    ) {
                        AgentsError::other(
                            crate::sandbox::sandboxes::docker::PROTECTED_MOUNT_ERROR_MESSAGE,
                        )
                    } else {
                        error
                    },
                );
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
        let instructions = self.build_instructions(&manifest, binding);

        let mut model_settings = self.agent.model_settings.clone();
        model_settings.extra_args = (!extra_args.is_empty()).then_some(extra_args);

        Ok(self.agent.clone_with(|agent| {
            agent.instructions = Some(instructions);
            agent.model_settings = model_settings;
            agent.tools.extend(capability_tools);
            agent.handoffs.extend(extra_handoffs);
        }))
    }

    /// The prepared agent's instructions.
    ///
    /// The capability fragments are rendered inside the callable, so every model call sees the
    /// workspace as it is then (a memory summary an earlier turn wrote, for one), as Python's
    /// `build_sandbox_instructions` does. The manifest is the one the agent was prepared
    /// against, also as in Python. A capability that
    /// [caches its instructions](Capability::caches_instructions) is rendered once.
    fn build_instructions(
        &self,
        manifest: &Manifest,
        binding: &CapabilityBinding,
    ) -> Instructions<C> {
        let base = self
            .base_instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_SANDBOX_INSTRUCTIONS.trim().to_string());
        let mount_policy_section = build_remote_mount_policy_instructions(manifest)
            .map(|policy| instruction_section("Sandbox remote mount policy", &policy));
        let filesystem_section = filesystem_instructions(manifest, binding);
        let original = self.agent.instructions.clone();
        let capabilities: Arc<Vec<CachedFragments<C>>> = Arc::new(
            self.capabilities
                .iter()
                .map(|capability| CachedFragments {
                    capability: capability.clone(),
                    cached: OnceLock::new(),
                })
                .collect(),
        );
        let (manifest, binding) = (Arc::new(manifest.clone()), Arc::new(binding.clone()));
        Instructions::Dynamic(Arc::new(move |ctx, agent| {
            let base = base.clone();
            let mount_policy_section = mount_policy_section.clone();
            let filesystem_section = filesystem_section.clone();
            let original = original.clone();
            let (capabilities, manifest, binding) =
                (capabilities.clone(), manifest.clone(), binding.clone());
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
                let mut fragments = Vec::new();
                for entry in capabilities.iter() {
                    let rendered = match entry.cached.get() {
                        Some(rendered) => rendered.clone(),
                        None => {
                            let rendered = entry.render(&binding, &manifest).await?;
                            if entry.capability.caches_instructions() {
                                let _ = entry.cached.set(rendered.clone());
                            }
                            rendered
                        }
                    };
                    fragments.extend(rendered);
                }
                let capability_section = (!fragments.is_empty()).then(|| {
                    instruction_section("Sandbox capability instructions", &fragments.join("\n\n"))
                });
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
    /// Identity key of each prepared agent, by pointer, so two agents sharing a name resolve to
    /// their own capability sets.
    keys_by_prepared_ptr: HashMap<usize, String>,
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
            keys_by_prepared_ptr: self.keys_by_prepared_ptr.clone(),
        }
    }
}

impl<C: Send + Sync + 'static> PreparedSandboxGraph<C> {
    /// The prepared agent the run starts from.
    pub fn starting_agent(&self) -> &Arc<Agent<C>> {
        &self.starting_agent
    }

    /// The sandbox agents in the graph, keyed by identity key (the name, or `name#N` for a
    /// further agent sharing it).
    pub fn agents_by_name(&self) -> &HashMap<String, Arc<SandboxAgent<C>>> {
        &self.agents_by_name
    }

    /// The prepared agents in the graph, keyed by identity key.
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

    /// [`process_context`](Self::process_context) for the prepared agent about to be called.
    /// The agent is found by identity, so two agents sharing a name each apply their own
    /// capabilities; an agent that is not one of the graph's falls back to its name.
    pub fn process_context_for_agent(
        &self,
        agent: &Arc<Agent<C>>,
        input: Vec<InputItem>,
    ) -> Vec<InputItem> {
        match self
            .keys_by_prepared_ptr
            .get(&(Arc::as_ptr(agent) as usize))
        {
            Some(key) => self.process_context(key, input),
            None => self.process_context(&agent.name, input),
        }
    }

    /// The identity key of a prepared agent of this graph.
    pub fn key_for_prepared(&self, agent: &Arc<Agent<C>>) -> Option<&str> {
        self.key_for_prepared_id(Arc::as_ptr(agent) as usize)
    }

    /// The identity key of the prepared agent whose `Arc` address is `id`, as a streamed run
    /// reports it in [`StreamCompletion::last_agent_id`](crate::StreamCompletion).
    pub fn key_for_prepared_id(&self, id: usize) -> Option<&str> {
        self.keys_by_prepared_ptr.get(&id).map(String::as_str)
    }
}

/// One node of the sandbox handoff graph, with its handoff edges classified.
struct GraphNode<C> {
    agent: Arc<SandboxAgent<C>>,
    /// Indices of the handoff edges that own their target, forming the spanning tree.
    owned_edges: BTreeSet<usize>,
}

/// Agents sharing one name, with their position in reach order.
type NameGroup<'a, C> = (&'a str, Vec<(usize, &'a Arc<SandboxAgent<C>>)>);

/// The state of one walk over the sandbox handoff graph.
struct GraphWalk<C> {
    /// Nodes by identity key.
    nodes: HashMap<String, GraphNode<C>>,
    /// Identity keys in the post-order of the spanning tree.
    order: Vec<String>,
    /// Identity key of each agent, by pointer, decided before the walk.
    keys: HashMap<usize, String>,
    /// Keys of the agents the walk has reached so far.
    visited: BTreeSet<usize>,
}

/// A deterministic description of a sandbox agent's configuration, used only to rank agents
/// that share a name. The text does not depend on where the agent sits in the graph.
fn sandbox_agent_signature<C: Send + Sync + 'static>(agent: &SandboxAgent<C>) -> String {
    let mut capabilities: Vec<&str> = agent
        .capabilities
        .iter()
        .map(|capability| capability.capability_type())
        .collect();
    capabilities.sort_unstable();
    let mut handoffs: Vec<&str> = agent
        .sandbox_handoffs
        .iter()
        .map(SandboxHandoffTarget::name)
        .collect();
    handoffs.sort_unstable();
    let instructions = match &agent.agent.instructions {
        None => Value::Null,
        Some(Instructions::Static(text)) => Value::String(text.clone()),
        Some(Instructions::Dynamic(_)) => serde_json::json!({"callable": "instructions"}),
    };
    let manifest: Vec<String> = agent
        .default_manifest
        .as_ref()
        .map(|manifest| {
            manifest
                .entries()
                .iter()
                .map(|(name, entry)| format!("{name}={entry:?}"))
                .collect()
        })
        .unwrap_or_default();
    serde_json::json!({
        "base_instructions": agent.base_instructions,
        "capabilities": capabilities,
        "handoffs": handoffs,
        "instructions": instructions,
        "manifest": manifest,
        "run_as": agent.run_as.as_ref().map(|user| user.name.clone()),
    })
    .to_string()
}

impl<C: Send + Sync + 'static> GraphWalk<C> {
    /// Decide every agent's identity key before the walk. The first agent of a name keeps it
    /// (the root always, otherwise the one whose configuration signature ranks first, then the
    /// earliest reached), and the rest get `name#2`, `name#3`, ... skipping any key another
    /// agent's literal name already uses. The keys therefore do not depend on the order the
    /// handoffs were declared in.
    fn new(root: &Arc<SandboxAgent<C>>) -> Self {
        let mut ordered: Vec<Arc<SandboxAgent<C>>> = Vec::new();
        let mut seen = BTreeSet::new();
        let mut queue = std::collections::VecDeque::from([root.clone()]);
        while let Some(agent) = queue.pop_front() {
            if !seen.insert(Arc::as_ptr(&agent) as usize) {
                continue;
            }
            queue.extend(
                agent
                    .sandbox_handoffs
                    .iter()
                    .filter_map(|target| target.agent().cloned()),
            );
            ordered.push(agent);
        }
        let literal_names: BTreeSet<&str> = ordered.iter().map(|a| a.name()).collect();
        let mut groups: Vec<NameGroup<'_, C>> = Vec::new();
        for (index, agent) in ordered.iter().enumerate() {
            match groups.iter_mut().find(|(name, _)| *name == agent.name()) {
                Some((_, group)) => group.push((index, agent)),
                None => groups.push((agent.name(), vec![(index, agent)])),
            }
        }
        let mut keys = HashMap::new();
        let mut used: BTreeSet<String> = BTreeSet::new();
        for (name, mut group) in groups {
            group.sort_by_cached_key(|(index, agent)| {
                (
                    u8::from(!Arc::ptr_eq(agent, root)),
                    sandbox_agent_signature(agent),
                    *index,
                )
            });
            let mut members = group.into_iter();
            if let Some((_, base)) = members.next() {
                used.insert(name.to_string());
                keys.insert(Arc::as_ptr(base) as usize, name.to_string());
            }
            let mut next = 2usize;
            for (_, duplicate) in members {
                let mut candidate = format!("{name}#{next}");
                while used.contains(&candidate) || literal_names.contains(candidate.as_str()) {
                    next += 1;
                    candidate = format!("{name}#{next}");
                }
                used.insert(candidate.clone());
                keys.insert(Arc::as_ptr(duplicate) as usize, candidate);
                next += 1;
            }
        }
        GraphWalk {
            nodes: HashMap::new(),
            order: Vec::new(),
            keys,
            visited: BTreeSet::new(),
        }
    }
}

/// The identity key a handoff edge points at: its agent's own key, or the name a named
/// (cycle-closing) target refers to.
fn target_key<C: Send + Sync + 'static>(
    target: &SandboxHandoffTarget<C>,
    keys: &HashMap<usize, String>,
) -> String {
    target
        .agent()
        .and_then(|agent| keys.get(&(Arc::as_ptr(agent) as usize)).cloned())
        .unwrap_or_else(|| target.name().to_string())
}

/// Walk the sandbox handoff graph depth-first, classifying each edge and producing the
/// post-order of the spanning tree.
///
/// The first edge to reach a node owns it; an edge to a node already on the stack (a cycle) or
/// already discovered elsewhere resolves lazily instead. Agents are told apart by identity, so
/// two agents may share a name.
fn collect_graph<C: Send + Sync + 'static>(
    node: &Arc<SandboxAgent<C>>,
    walk: &mut GraphWalk<C>,
) -> Result<()> {
    let ptr = Arc::as_ptr(node) as usize;
    // Register before recursing so a cycle back to this node is seen as already discovered.
    if !walk.visited.insert(ptr) {
        return Ok(());
    }
    let key = walk
        .keys
        .get(&ptr)
        .cloned()
        .ok_or_else(|| AgentsError::other("sandbox agent has no identity key"))?;
    walk.nodes.insert(
        key.clone(),
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
        let first_visit = !walk.visited.contains(&(Arc::as_ptr(agent) as usize));
        collect_graph(agent, walk)?;
        if first_visit {
            owned_edges.insert(index);
        }
    }
    if let Some(entry) = walk.nodes.get_mut(&key) {
        entry.owned_edges = owned_edges;
    }
    walk.order.push(key);
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

/// `skills` -> `Skills`, `my_capability` -> `MyCapability`.
fn pascal_case(capability_type: &str) -> String {
    capability_type
        .split(['_', '-'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// One capability of a prepared agent, with its rendered fragments once they are cached.
struct CachedFragments<C> {
    capability: Arc<dyn Capability<C>>,
    cached: OnceLock<Vec<String>>,
}

impl<C: Send + Sync + 'static> CachedFragments<C> {
    /// The capability's plain fragment followed by its session-bound one.
    async fn render(
        &self,
        binding: &CapabilityBinding,
        manifest: &Manifest,
    ) -> Result<Vec<String>> {
        let mut fragments = Vec::new();
        if let Some(fragment) = self.capability.instructions(manifest).await? {
            fragments.push(fragment);
        }
        if let Some(fragment) = self
            .capability
            .bound_instructions(binding, manifest)
            .await?
        {
            fragments.push(fragment);
        }
        Ok(fragments)
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
            "Relative paths passed to the built-in `exec_command`, `view_image`, and \
             `apply_patch` tools resolve from this directory."
                .to_string(),
            "Other sandbox tools follow their own path contract.".to_string(),
            format!("The session workspace root remains `{root}`."),
            "The working directory changes path resolution; it does not isolate this run from \
             the rest of the session workspace."
                .to_string(),
            "Files outside the working directory may be visible to or shared with other runs."
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
                "view_image",
                "apply_patch",
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
