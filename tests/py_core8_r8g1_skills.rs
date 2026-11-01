//! Ports of the custom lazy skill source rows of
//! `tests/sandbox/capabilities/test_skills_capability.py`: a source that returns its own
//! metadata and its own `load_skill` result, checked against the run working directory.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::capabilities::skills::{LoadedSkill, SkillMetadata};
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, LazySkillSource, Skills,
};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxResult};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::session::SandboxSession;
use openai_agents_sdk::sandbox::testing::scripted_sandbox_session;
use openai_agents_sdk::sandbox::types::User;
use openai_agents_sdk::sandbox::workspace_paths::{SandboxPathGrant, SandboxWorkspaceScope};

const SCOPED_PATH_MESSAGE: &str =
    "skill path must be non-empty and workspace-relative when sandbox.cwd is configured";

/// Python's `_StaticResultLazySkillSource`: fixed metadata and a fixed `load_skill` result.
#[derive(Debug)]
struct StaticResultLazySkillSource {
    result: LoadedSkill,
    metadata_path: Option<String>,
}

#[async_trait]
impl LazySkillSource for StaticResultLazySkillSource {
    fn list_skill_metadata(
        &self,
        _skills_path: &str,
        _source_grants: &[SandboxPathGrant],
    ) -> Vec<SkillMetadata> {
        self.metadata_path
            .iter()
            .map(|path| SkillMetadata {
                name: "dynamic-skill".to_string(),
                description: "dynamic description".to_string(),
                path: path.clone(),
            })
            .collect()
    }

    async fn load_skill(
        &self,
        _skill_name: &str,
        _session: &dyn SandboxSession,
        _skills_path: &str,
        _user: Option<&User>,
    ) -> SandboxResult<LoadedSkill> {
        Ok(self.result.clone())
    }
}

fn loaded(path: &str) -> LoadedSkill {
    LoadedSkill {
        status: "loaded".to_string(),
        skill_name: "dynamic-skill".to_string(),
        path: path.to_string(),
    }
}

fn skills_from(source: StaticResultLazySkillSource) -> Skills {
    Skills::lazy_from(Arc::new(source))
}

fn binding() -> CapabilityBinding {
    CapabilityBinding::new(Arc::new(
        scripted_sandbox_session([]).with_manifest(Manifest::new().with_root("/workspace")),
    ))
}

/// Python: `test_custom_lazy_result_is_unchanged_without_run_cwd`.
#[tokio::test]
async fn custom_lazy_result_is_unchanged_without_a_run_cwd() {
    let skills = skills_from(StaticResultLazySkillSource {
        result: loaded(".agents/dynamic-skill"),
        metadata_path: None,
    });
    let binding = binding();

    let output = skills
        .load_skill(
            binding.session.as_ref(),
            &SandboxWorkspaceScope::root(),
            None,
            "dynamic-skill",
        )
        .await
        .unwrap();

    assert_eq!(output, loaded(".agents/dynamic-skill"));
}

/// Python: `test_custom_lazy_result_requires_valid_path_with_run_cwd` (the `missing` and two
/// `invalid` parameter sets).
#[tokio::test]
async fn custom_lazy_result_requires_a_valid_path_with_a_run_cwd() {
    for (path, reason) in [
        ("", "missing"),
        ("../escape", "invalid"),
        (r".agents\dynamic-skill", "invalid"),
    ] {
        let skills = skills_from(StaticResultLazySkillSource {
            result: loaded(path),
            metadata_path: None,
        });
        let binding = binding();
        let scope = SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).unwrap();

        let error = skills
            .load_skill(binding.session.as_ref(), &scope, None, "dynamic-skill")
            .await
            .unwrap_err();

        assert_eq!(error.code, ErrorCode::SkillsConfigInvalid, "{path:?}");
        assert_eq!(error.message, SCOPED_PATH_MESSAGE, "{path:?}");
        let context = |key: &str| error.context.get(key).map(String::as_str);
        assert_eq!(context("skill_name"), Some("dynamic-skill"), "{path:?}");
        assert_eq!(context("field"), Some("path"), "{path:?}");
        assert_eq!(context("reason"), Some(reason), "{path:?}");
    }
}

/// Python: `test_custom_lazy_metadata_reports_invalid_scoped_path_as_config_error`: the index
/// rendered into the instructions refuses metadata that points outside the workspace once a run
/// working directory is configured.
#[tokio::test]
async fn custom_lazy_metadata_reports_an_invalid_scoped_path_as_a_config_error() {
    let skills = skills_from(StaticResultLazySkillSource {
        result: loaded(".agents/dynamic-skill"),
        metadata_path: Some("../outside".to_string()),
    });
    let binding =
        binding().with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).unwrap());
    let manifest = binding.session.manifest().clone();

    let error = Capability::<()>::bound_instructions(&skills, &binding, &manifest)
        .await
        .unwrap_err();

    let text = error.to_string();
    assert!(text.contains(SCOPED_PATH_MESSAGE), "{text}");
    assert!(text.contains("skills_config_invalid"), "{text}");
}
