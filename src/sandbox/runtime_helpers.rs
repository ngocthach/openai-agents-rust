//! Shell helper scripts the SDK installs inside a sandbox. Mirrors
//! `agents.sandbox.session.runtime_helpers`.
//!
//! Two jobs need a decision made *inside* the sandbox, where the filesystem actually is:
//! resolving a model-provided path through the symlinks that exist there, and fingerprinting the
//! workspace without streaming it to the host. Both are plain POSIX `sh`, so they run on every
//! image the backends support, and both are copies of the Python scripts byte for byte — the
//! install path is derived from the script's own hash, so a drifting copy would install itself
//! somewhere else and the two SDKs would stop sharing a cache.

use std::sync::LazyLock;

use sha2::{Digest, Sha256};

/// Where helper scripts are installed inside the sandbox.
pub const HELPER_INSTALL_ROOT: &str = "/tmp/openai-agents/bin";

/// The marker Python writes into every install command, kept so the two agree.
pub const INSTALL_MARKER: &str = "INSTALL_RUNTIME_HELPER_V1";

const RESOLVE_WORKSPACE_PATH_SCRIPT: &str = include_str!("helpers/resolve-workspace-path.sh");
const WORKSPACE_FINGERPRINT_SCRIPT: &str = include_str!("helpers/workspace-fingerprint.sh");

/// A shell script the SDK installs into a sandbox before running it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeHelperScript {
    /// The script's short name, used in its install path.
    pub name: &'static str,
    /// The script body.
    pub content: &'static str,
    /// Where the script is installed inside the sandbox.
    pub install_path: String,
}

impl RuntimeHelperScript {
    /// Derive the install path from the content hash, as Python's `from_content` does.
    pub fn from_content(name: &'static str, content: &'static str) -> Self {
        let digest = Sha256::digest(content.as_bytes());
        let hash: String = digest
            .iter()
            .take(6)
            .map(|byte| format!("{byte:02x}"))
            .collect();
        RuntimeHelperScript {
            name,
            content,
            install_path: format!("{HELPER_INSTALL_ROOT}/{name}-{hash}"),
        }
    }

    /// The argv that checks whether the script is already installed and executable.
    pub fn present_command(&self) -> Vec<String> {
        vec![
            "test".to_string(),
            "-x".to_string(),
            self.install_path.clone(),
        ]
    }

    /// The argv that installs the script.
    ///
    /// The write is staged under a pid-suffixed temporary name and renamed into place, so a
    /// second session installing the same helper at the same time cannot be read half-written.
    /// An identical existing copy is left alone, which keeps the inode stable for a process that
    /// is already running it.
    pub fn install_command(&self) -> Vec<String> {
        let install_path = &self.install_path;
        let file_name = install_path
            .rsplit('/')
            .next()
            .unwrap_or(install_path.as_str());
        let heredoc = format!(
            "OPENAI_AGENTS_HELPER_{}",
            file_name.to_uppercase().replace('-', "_")
        );
        let marker = INSTALL_MARKER;
        let content = self.content;
        let script = format!(
            "# {marker}\n\
             set -eu\n\n\
             dest=\"$1\"\n\
             tmp=\"{install_path}.tmp.$$\"\n\n\
             mkdir -p -- \"$(dirname -- \"$dest\")\"\n\n\
             cleanup() {{\n    rm -f -- \"$tmp\"\n}}\n\
             trap cleanup EXIT INT TERM\n\n\
             cat > \"$tmp\" <<'{heredoc}'\n\
             {content}\n\
             {heredoc}\n\
             chmod 0555 \"$tmp\"\n\
             if [ -d \"$dest\" ]; then\n    rm -rf -- \"$dest\"\nfi\n\
             if [ -x \"$dest\" ] && command -v cmp >/dev/null 2>&1 && cmp -s \"$dest\" \"$tmp\"; \
             then\n    rm -f -- \"$tmp\"\n    trap - EXIT INT TERM\n    exit 0\nfi\n\
             rm -f -- \"$dest\"\n\
             mv -f -- \"$tmp\" \"$dest\"\n\
             trap - EXIT INT TERM"
        );
        vec![
            "sh".to_string(),
            "-c".to_string(),
            script,
            "sh".to_string(),
            install_path.clone(),
        ]
    }
}

/// Resolve a model-provided path inside the sandbox, following its symlinks.
pub static RESOLVE_WORKSPACE_PATH_HELPER: LazyLock<RuntimeHelperScript> = LazyLock::new(|| {
    RuntimeHelperScript::from_content("resolve-workspace-path", RESOLVE_WORKSPACE_PATH_SCRIPT)
});

/// Hash the workspace tree inside the sandbox and cache the record.
pub static WORKSPACE_FINGERPRINT_HELPER: LazyLock<RuntimeHelperScript> = LazyLock::new(|| {
    RuntimeHelperScript::from_content("workspace-fingerprint", WORKSPACE_FINGERPRINT_SCRIPT)
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_paths_match_the_python_helpers() {
        // The path carries the script's hash, so these also assert the scripts are byte-identical
        // to Python's.
        assert_eq!(
            WORKSPACE_FINGERPRINT_HELPER.install_path,
            "/tmp/openai-agents/bin/workspace-fingerprint-fe85e2af8427"
        );
        assert_eq!(
            RESOLVE_WORKSPACE_PATH_HELPER.install_path,
            "/tmp/openai-agents/bin/resolve-workspace-path-62a6e78539cf"
        );
    }

    #[test]
    fn the_install_command_carries_the_script_and_its_destination() {
        let command = WORKSPACE_FINGERPRINT_HELPER.install_command();
        assert_eq!(command[0], "sh");
        assert_eq!(command[1], "-c");
        assert!(command[2].contains(INSTALL_MARKER));
        assert!(command[2].contains("WORKSPACE_FINGERPRINT_V2"));
        assert!(command[2].contains("OPENAI_AGENTS_HELPER_WORKSPACE_FINGERPRINT"));
        assert_eq!(command[3], "sh");
        assert_eq!(command[4], WORKSPACE_FINGERPRINT_HELPER.install_path);
    }

    #[test]
    fn the_present_command_tests_for_an_executable() {
        assert_eq!(
            RESOLVE_WORKSPACE_PATH_HELPER.present_command(),
            vec![
                "test".to_string(),
                "-x".to_string(),
                RESOLVE_WORKSPACE_PATH_HELPER.install_path.clone(),
            ]
        );
    }
}
