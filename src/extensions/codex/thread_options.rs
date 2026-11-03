//! Options applied to a Codex thread. Mirrors
//! `agents.extensions.experimental.codex.thread_options`.

use serde::{Deserialize, Serialize};

/// Approval policy for tool invocations inside Codex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApprovalMode {
    /// Never ask for approval.
    Never,
    /// Ask whenever Codex requests it.
    OnRequest,
    /// Ask only after a failure.
    OnFailure,
    /// Ask for untrusted commands.
    Untrusted,
}

impl ApprovalMode {
    /// The value passed to the Codex CLI.
    pub fn as_str(&self) -> &'static str {
        match self {
            ApprovalMode::Never => "never",
            ApprovalMode::OnRequest => "on-request",
            ApprovalMode::OnFailure => "on-failure",
            ApprovalMode::Untrusted => "untrusted",
        }
    }
}

/// Sandbox permissions for filesystem and network access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    /// Read-only access to the workspace.
    ReadOnly,
    /// Write access to the workspace.
    WorkspaceWrite,
    /// No sandbox at all.
    DangerFullAccess,
}

impl SandboxMode {
    /// The value passed to `--sandbox`.
    pub fn as_str(&self) -> &'static str {
        match self {
            SandboxMode::ReadOnly => "read-only",
            SandboxMode::WorkspaceWrite => "workspace-write",
            SandboxMode::DangerFullAccess => "danger-full-access",
        }
    }
}

/// Reasoning effort requested from the Codex model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelReasoningEffort {
    /// Minimal reasoning.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra high reasoning effort.
    Xhigh,
}

impl ModelReasoningEffort {
    /// The value passed to `--config model_reasoning_effort`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelReasoningEffort::Minimal => "minimal",
            ModelReasoningEffort::Low => "low",
            ModelReasoningEffort::Medium => "medium",
            ModelReasoningEffort::High => "high",
            ModelReasoningEffort::Xhigh => "xhigh",
        }
    }
}

/// Web search behavior configured through the Codex config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchMode {
    /// Web search is disabled.
    Disabled,
    /// Cached results only.
    Cached,
    /// Live web search.
    Live,
}

impl WebSearchMode {
    /// The value passed to `--config web_search`.
    pub fn as_str(&self) -> &'static str {
        match self {
            WebSearchMode::Disabled => "disabled",
            WebSearchMode::Cached => "cached",
            WebSearchMode::Live => "live",
        }
    }
}

/// Options applied to every turn of a Codex thread.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadOptions {
    /// Model identifier passed to the Codex CLI (`--model`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Sandbox permissions for filesystem and network access (`--sandbox`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_mode: Option<SandboxMode>,
    /// Working directory for the Codex CLI process (`--cd`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<String>,
    /// Allow running outside a Git repository (`--skip-git-repo-check`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_git_repo_check: Option<bool>,
    /// Reasoning effort for the Codex model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reasoning_effort: Option<ModelReasoningEffort>,
    /// Toggle network access for sandboxed workspace writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_access_enabled: Option<bool>,
    /// Web search mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_mode: Option<WebSearchMode>,
    /// Legacy toggle for web search behavior; ignored when `web_search_mode` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_enabled: Option<bool>,
    /// Approval policy for tool invocations within Codex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<ApprovalMode>,
    /// Additional filesystem roots available to Codex (`--add-dir`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<Vec<String>>,
}

impl ThreadOptions {
    /// Default thread options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the sandbox mode.
    pub fn with_sandbox_mode(mut self, mode: SandboxMode) -> Self {
        self.sandbox_mode = Some(mode);
        self
    }

    /// Set the working directory.
    pub fn with_working_directory(mut self, dir: impl Into<String>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    /// Allow running outside a Git repository.
    pub fn with_skip_git_repo_check(mut self, skip: bool) -> Self {
        self.skip_git_repo_check = Some(skip);
        self
    }

    /// Set the reasoning effort.
    pub fn with_model_reasoning_effort(mut self, effort: ModelReasoningEffort) -> Self {
        self.model_reasoning_effort = Some(effort);
        self
    }

    /// Toggle sandboxed network access.
    pub fn with_network_access_enabled(mut self, enabled: bool) -> Self {
        self.network_access_enabled = Some(enabled);
        self
    }

    /// Set the web search mode.
    pub fn with_web_search_mode(mut self, mode: WebSearchMode) -> Self {
        self.web_search_mode = Some(mode);
        self
    }

    /// Set the legacy web search toggle.
    pub fn with_web_search_enabled(mut self, enabled: bool) -> Self {
        self.web_search_enabled = Some(enabled);
        self
    }

    /// Set the approval policy.
    pub fn with_approval_policy(mut self, policy: ApprovalMode) -> Self {
        self.approval_policy = Some(policy);
        self
    }

    /// Set additional filesystem roots.
    pub fn with_additional_directories(mut self, dirs: Vec<String>) -> Self {
        self.additional_directories = Some(dirs);
        self
    }
}
