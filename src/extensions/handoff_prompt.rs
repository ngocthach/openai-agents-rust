//! Recommended prompt prefix for agents that use handoffs. Mirrors
//! `agents.extensions.handoff_prompt`.

/// A recommended prompt prefix for agents that use handoffs.
pub const RECOMMENDED_PROMPT_PREFIX: &str = "# System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n";

/// Prefix `prompt` with [`RECOMMENDED_PROMPT_PREFIX`].
pub fn prompt_with_handoff_instructions(prompt: &str) -> String {
    format!("{RECOMMENDED_PROMPT_PREFIX}\n\n{prompt}")
}
