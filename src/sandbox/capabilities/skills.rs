//! The skills capability. Mirrors `agents.sandbox.capabilities.skills`.
//!
//! A skill is a `SKILL.md` file plus optional `scripts/`, `references/` and `assets/`
//! directories, materialized under a discovery root (`.agents` by default). The capability lists
//! the available skills in the agent instructions and, for lazy sources, exposes a `load_skill`
//! tool that copies a skill into the workspace on demand.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::error::{AgentsError, Result};
use crate::sandbox::capabilities::{Capability, CapabilityBinding};
use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::session::SandboxSession;
use crate::tool::{FunctionTool, Tool};

/// The default discovery root skills are materialized into.
pub const DEFAULT_SKILLS_PATH: &str = ".agents";

/// The name of the skill-loading tool.
pub const LOAD_SKILL_TOOL_NAME: &str = "load_skill";

const SKILLS_SECTION_INTRO: &str = "A skill is a set of local instructions to follow that is \
    stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry \
    includes a name, description, and file path so you can open the source for full instructions \
    when using a specific skill.";

const SKILL_PATH_GUIDANCE: &str = "- Skill paths: Treat each listed path as the skill root. \
Resolve relative paths in `SKILL.md`, including `scripts/`, `references/`, and `assets/`, \
against that root rather than the shell working directory.
- Shared resources: Skill files belong to the sandbox session and may be visible to other runs. \
Unless the task explicitly requires editing a skill, invoke scripts through the listed skill \
root and write task inputs, outputs, caches, and temporary files in the run working directory.";

const HOW_TO_USE_SKILLS_SECTION: &str = "### How to use skills
- Discovery: The list above is the skills available in this session (name + description + file \
path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly \
matches a skill's description shown above, you must use that skill for that turn. Multiple \
mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly \
and continue with the best fallback.";

/// What the model is told about one available skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMetadata {
    /// The skill name, which is also its directory name.
    pub name: String,
    /// What the skill is for.
    pub description: String,
    /// Whether the skill body still has to be loaded with `load_skill`.
    pub deferred: bool,
    /// Compatibility note shown alongside the description.
    pub compatibility: Option<String>,
}

/// A skill: a `SKILL.md` body plus optional scripts, references and assets.
#[derive(Debug, Clone)]
pub struct Skill {
    /// The skill name, used as its directory name under the discovery root.
    pub name: String,
    /// What the skill is for.
    pub description: String,
    /// The `SKILL.md` body.
    pub content: Vec<u8>,
    /// Compatibility note shown alongside the description.
    pub compatibility: Option<String>,
    /// Files placed under `scripts/`.
    pub scripts: BTreeMap<String, Entry>,
    /// Files placed under `references/`.
    pub references: BTreeMap<String, Entry>,
    /// Files placed under `assets/`.
    pub assets: BTreeMap<String, Entry>,
}

impl Skill {
    /// A skill with the given name, description and `SKILL.md` body.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Skill {
            name: name.into(),
            description: description.into(),
            content: content.into().into_bytes(),
            compatibility: None,
            scripts: BTreeMap::new(),
            references: BTreeMap::new(),
            assets: BTreeMap::new(),
        }
    }

    /// Set the compatibility note.
    pub fn with_compatibility(mut self, compatibility: impl Into<String>) -> Self {
        self.compatibility = Some(compatibility.into());
        self
    }

    /// Add a file under `scripts/`.
    pub fn with_script(mut self, name: impl Into<String>, entry: Entry) -> Self {
        self.scripts.insert(name.into(), entry);
        self
    }

    /// Add a file under `references/`.
    pub fn with_reference(mut self, name: impl Into<String>, entry: Entry) -> Self {
        self.references.insert(name.into(), entry);
        self
    }

    /// Add a file under `assets/`.
    pub fn with_asset(mut self, name: impl Into<String>, entry: Entry) -> Self {
        self.assets.insert(name.into(), entry);
        self
    }

    /// Validate that the skill name is a safe workspace-relative directory name.
    pub fn validate(&self) -> SandboxResult<()> {
        validate_relative_path("skills[].name", &self.name)
    }

    /// The directory entry this skill materializes into.
    pub fn as_dir_entry(&self) -> Entry {
        let mut dir = Entry::dir().with_child("SKILL.md", Entry::file(self.content.clone()));
        for (group, files) in [
            ("scripts", &self.scripts),
            ("references", &self.references),
            ("assets", &self.assets),
        ] {
            if files.is_empty() {
                continue;
            }
            let mut group_dir = Entry::dir();
            for (name, entry) in files {
                group_dir = group_dir.with_child(name.clone(), entry.clone());
            }
            dir = dir.with_child(group, group_dir);
        }
        dir
    }

    /// The metadata shown to the model.
    pub fn metadata(&self) -> SkillMetadata {
        SkillMetadata {
            name: self.name.clone(),
            description: self.description.clone(),
            deferred: false,
            compatibility: self.compatibility.clone(),
        }
    }
}

fn validate_relative_path(field: &str, value: &str) -> SandboxResult<()> {
    let invalid = value.trim().is_empty()
        || value == "."
        || value.starts_with('/')
        || value.contains('\\')
        || value.split('/').any(|s| s == "..");
    if invalid {
        return Err(SandboxError::new(
            ErrorCode::SkillsConfigInvalid,
            format!("{field} must be a non-empty workspace-relative path, got `{value}`"),
        ));
    }
    Ok(())
}

/// A source that lists skills up front and materializes their bodies on demand.
#[async_trait]
pub trait LazySkillSource: Send + Sync + fmt::Debug {
    /// The skills available from this source.
    async fn list_skill_metadata(&self) -> SandboxResult<Vec<SkillMetadata>>;

    /// Load one skill, returning the directory entry to materialize.
    async fn load_skill(&self, name: &str) -> SandboxResult<Entry>;
}

/// A lazy skill source backed by a directory on the machine running the SDK.
///
/// Each immediate subdirectory holding a `SKILL.md` is one skill. The description is read from
/// the `description:` line of the skill's YAML front matter, falling back to its first
/// non-heading line.
#[derive(Debug, Clone)]
pub struct LocalDirLazySkillSource {
    source: PathBuf,
}

impl LocalDirLazySkillSource {
    /// A source reading skills from `source` on the host.
    pub fn new(source: impl Into<PathBuf>) -> Self {
        LocalDirLazySkillSource {
            source: source.into(),
        }
    }

    /// The host directory holding the skills.
    pub fn source(&self) -> &Path {
        &self.source
    }

    fn skill_dir(&self, name: &str) -> SandboxResult<PathBuf> {
        validate_relative_path("load_skill.name", name)?;
        let dir = self.source.join(name);
        // Reject anything that leaves the source directory, even through a symlink.
        let real_source =
            std::fs::canonicalize(&self.source).unwrap_or_else(|_| self.source.clone());
        let real_dir = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if !real_dir.starts_with(&real_source) {
            return Err(SandboxError::new(
                ErrorCode::SkillsConfigInvalid,
                format!("skill `{name}` is outside the configured skill source"),
            ));
        }
        Ok(dir)
    }
}

/// Extract a skill description from a `SKILL.md` body.
pub fn skill_description_from_body(body: &str) -> String {
    for line in body.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("description:") {
            let rest = rest.trim().trim_matches('"').trim_matches('\'');
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
    }
    for line in body.lines() {
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') && !line.starts_with("---") {
            return line.to_string();
        }
    }
    String::new()
}

#[async_trait]
impl LazySkillSource for LocalDirLazySkillSource {
    async fn list_skill_metadata(&self) -> SandboxResult<Vec<SkillMetadata>> {
        let mut entries = match tokio::fs::read_dir(&self.source).await {
            Ok(entries) => entries,
            Err(e) => {
                return Err(SandboxError::new(
                    ErrorCode::LocalDirReadError,
                    format!(
                        "Failed to read the skill source `{}`: {e}",
                        self.source.display()
                    ),
                ));
            }
        };
        let mut skills = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let Ok(file_type) = entry.file_type().await else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if validate_relative_path("skills[].name", &name).is_err() {
                continue;
            }
            let Ok(body) = tokio::fs::read_to_string(entry.path().join("SKILL.md")).await else {
                continue;
            };
            skills.push(SkillMetadata {
                description: skill_description_from_body(&body),
                name,
                deferred: true,
                compatibility: None,
            });
        }
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(skills)
    }

    async fn load_skill(&self, name: &str) -> SandboxResult<Entry> {
        let dir = self.skill_dir(name)?;
        if tokio::fs::metadata(dir.join("SKILL.md")).await.is_err() {
            return Err(SandboxError::new(
                ErrorCode::SkillsConfigInvalid,
                format!("skill `{name}` has no SKILL.md in the configured source"),
            ));
        }
        Ok(Entry::local_dir(dir))
    }
}

/// Arguments of the `load_skill` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct LoadSkillArgs {
    /// Name of the skill to load into the workspace.
    pub skill_name: String,
}

/// Where skills come from.
#[derive(Clone)]
enum SkillSource {
    Inline(Vec<Skill>),
    From(Box<Entry>),
    Lazy(Arc<dyn LazySkillSource>),
}

impl fmt::Debug for SkillSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkillSource::Inline(skills) => f
                .debug_tuple("Inline")
                .field(&skills.iter().map(|s| s.name.clone()).collect::<Vec<_>>())
                .finish(),
            SkillSource::From(_) => f.write_str("From(..)"),
            SkillSource::Lazy(source) => f.debug_tuple("Lazy").field(source).finish(),
        }
    }
}

/// Materializes skills into a discovery root inside the sandbox workspace.
#[derive(Debug, Clone)]
pub struct Skills {
    source: SkillSource,
    skills_path: String,
}

impl Skills {
    /// Skills defined inline.
    pub fn from_skills(skills: Vec<Skill>) -> Self {
        Skills {
            source: SkillSource::Inline(skills),
            skills_path: DEFAULT_SKILLS_PATH.to_string(),
        }
    }

    /// Skills taken from a directory entry that is already part of the manifest world.
    pub fn from_entry(entry: Entry) -> Self {
        Skills {
            source: SkillSource::From(Box::new(entry)),
            skills_path: DEFAULT_SKILLS_PATH.to_string(),
        }
    }

    /// Skills listed up front and loaded on demand with `load_skill`.
    pub fn lazy_from(source: Arc<dyn LazySkillSource>) -> Self {
        Skills {
            source: SkillSource::Lazy(source),
            skills_path: DEFAULT_SKILLS_PATH.to_string(),
        }
    }

    /// Set the discovery root skills are materialized into.
    pub fn with_skills_path(mut self, skills_path: impl Into<String>) -> Self {
        self.skills_path = skills_path.into();
        self
    }

    /// The discovery root.
    pub fn skills_path(&self) -> &str {
        &self.skills_path
    }

    /// Validate the configuration.
    pub fn validate(&self) -> SandboxResult<()> {
        validate_relative_path("skills_path", &self.skills_path)?;
        match &self.source {
            SkillSource::Inline(skills) => {
                if skills.is_empty() {
                    return Err(SandboxError::new(
                        ErrorCode::SkillsConfigInvalid,
                        "skills capability requires at least one skill",
                    ));
                }
                let mut seen = std::collections::BTreeSet::new();
                for skill in skills {
                    skill.validate()?;
                    if !seen.insert(skill.name.clone()) {
                        return Err(SandboxError::new(
                            ErrorCode::SkillsConfigInvalid,
                            format!("duplicate skill name: {}", skill.name),
                        ));
                    }
                }
                Ok(())
            }
            SkillSource::From(entry) => {
                if !entry.is_dir() {
                    return Err(SandboxError::new(
                        ErrorCode::SkillsConfigInvalid,
                        "`from_entry` must be a directory-like entry",
                    ));
                }
                Ok(())
            }
            SkillSource::Lazy(_) => Ok(()),
        }
    }

    /// The absolute sandbox path of one skill directory.
    pub fn skill_path(&self, manifest_root: &str, name: &str) -> String {
        format!(
            "{}/{}/{name}",
            manifest_root.trim_end_matches('/'),
            self.skills_path.trim_matches('/')
        )
    }

    /// Materialize one lazily sourced skill into the live workspace.
    pub async fn load_skill(
        &self,
        session: &dyn SandboxSession,
        name: &str,
    ) -> SandboxResult<String> {
        let SkillSource::Lazy(source) = &self.source else {
            return Err(SandboxError::new(
                ErrorCode::SkillsConfigInvalid,
                "load_skill is only available for lazily sourced skills",
            ));
        };
        let entry = source.load_skill(name).await?;
        let dest = format!(
            "{}/{}",
            self.skills_path.trim_matches('/'),
            name.trim_matches('/')
        );
        let resolved = session.normalize_path(&dest)?;
        session.mkdir(&resolved, true).await?;
        entry.apply(session, &resolved).await?;
        Ok(resolved)
    }

    async fn metadata(&self) -> SandboxResult<Vec<SkillMetadata>> {
        match &self.source {
            SkillSource::Inline(skills) => Ok(skills.iter().map(Skill::metadata).collect()),
            SkillSource::From(entry) => Ok(match &entry.kind {
                EntryKind::Dir { children } => children
                    .iter()
                    .map(|(name, child)| SkillMetadata {
                        name: name.clone(),
                        description: child.description.clone().unwrap_or_default(),
                        deferred: false,
                        compatibility: None,
                    })
                    .collect(),
                _ => Vec::new(),
            }),
            SkillSource::Lazy(source) => source.list_skill_metadata().await,
        }
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for Skills {
    fn capability_type(&self) -> &str {
        "skills"
    }

    fn required_capability_types(&self) -> Vec<&'static str> {
        vec!["shell"]
    }

    fn process_manifest(&self, manifest: Manifest) -> Manifest {
        let root = self.skills_path.trim_matches('/').to_string();
        match &self.source {
            SkillSource::Inline(skills) => {
                let mut dir = Entry::dir();
                for skill in skills {
                    dir = dir.with_child(skill.name.clone(), skill.as_dir_entry());
                }
                manifest.with_entry(root, dir)
            }
            SkillSource::From(entry) => manifest.with_entry(root, (**entry).clone()),
            // A lazy source reserves the namespace but materializes nothing up front.
            SkillSource::Lazy(_) => manifest.with_entry(root, Entry::dir()),
        }
    }

    fn tools(&self, binding: &CapabilityBinding) -> Result<Vec<Tool<C>>> {
        if !matches!(self.source, SkillSource::Lazy(_)) {
            return Ok(Vec::new());
        }
        let capability = self.clone();
        let session = binding.session.clone();
        let tool = FunctionTool::<C>::new(
            LOAD_SKILL_TOOL_NAME,
            "Loads a skill into the sandbox workspace and returns the path of its skill root.",
            move |_ctx, args: LoadSkillArgs| {
                let capability = capability.clone();
                let session = session.clone();
                async move {
                    match capability
                        .load_skill(session.as_ref(), &args.skill_name)
                        .await
                    {
                        Ok(path) => Ok::<_, AgentsError>(format!(
                            "Loaded skill `{}` at {path}. Read {path}/SKILL.md for its \
                             instructions.",
                            args.skill_name
                        )),
                        Err(error) => Ok(format!(
                            "Could not load skill `{}`: {}",
                            args.skill_name, error.message
                        )),
                    }
                }
            },
        )
        .with_strict_json_schema(false);
        Ok(vec![Tool::Function(Arc::new(tool))])
    }

    async fn instructions(&self, manifest: &Manifest) -> Result<Option<String>> {
        let skills = self.metadata().await?;
        if skills.is_empty() {
            return Ok(None);
        }
        let mut lines = vec![
            "## Skills".to_string(),
            SKILLS_SECTION_INTRO.to_string(),
            String::new(),
        ];
        for skill in &skills {
            let path = self.skill_path(manifest.root(), &skill.name);
            let mut line = format!("- {}: {}", skill.name, skill.description);
            if let Some(compatibility) = &skill.compatibility {
                line.push_str(&format!(" (compatibility: {compatibility})"));
            }
            line.push_str(&format!(" [{path}/SKILL.md]"));
            if skill.deferred {
                line.push_str(&format!(
                    " — run `{LOAD_SKILL_TOOL_NAME}` with skill_name=`{}` before reading it.",
                    skill.name
                ));
            }
            lines.push(line);
        }
        lines.push(String::new());
        lines.push(HOW_TO_USE_SKILLS_SECTION.to_string());
        lines.push(SKILL_PATH_GUIDANCE.to_string());
        Ok(Some(lines.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
    use crate::sandbox::session::SandboxClient;

    #[test]
    fn builds_skill_directories() {
        let skill = Skill::new("deploy", "Deploys the app", "# Deploy\n")
            .with_script("run.sh", Entry::text_file("echo hi"))
            .with_reference("notes.md", Entry::text_file("notes"));
        let entry = skill.as_dir_entry();
        let EntryKind::Dir { children } = &entry.kind else {
            panic!("expected a dir");
        };
        assert!(children.contains_key("SKILL.md"));
        assert!(children.contains_key("scripts"));
        assert!(children.contains_key("references"));
        assert!(!children.contains_key("assets"));
    }

    #[test]
    fn validates_configuration() {
        assert!(Skills::from_skills(vec![]).validate().is_err());
        assert!(
            Skills::from_skills(vec![Skill::new("a", "", "x"), Skill::new("a", "", "y")])
                .validate()
                .is_err()
        );
        assert!(
            Skills::from_skills(vec![Skill::new("../a", "", "x")])
                .validate()
                .is_err()
        );
        assert!(
            Skills::from_skills(vec![Skill::new("a", "", "x")])
                .with_skills_path("/abs")
                .validate()
                .is_err()
        );
        assert!(
            Skills::from_entry(Entry::text_file("x"))
                .validate()
                .is_err()
        );
        assert!(
            Skills::from_skills(vec![Skill::new("a", "", "x")])
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn injects_skills_into_the_manifest() {
        let skills = Skills::from_skills(vec![Skill::new("deploy", "Deploys", "# Deploy")]);
        let manifest = Capability::<()>::process_manifest(&skills, Manifest::new());
        let rendered = manifest.render(4, None);
        assert!(rendered.contains(".agents/"));
        assert!(rendered.contains("deploy/"));
        assert!(rendered.contains("SKILL.md"));
    }

    #[test]
    fn extracts_descriptions_from_skill_bodies() {
        assert_eq!(
            skill_description_from_body("---\nname: a\ndescription: Does a thing\n---\n# A"),
            "Does a thing"
        );
        assert_eq!(
            skill_description_from_body("# Heading\n\nFirst real line\n"),
            "First real line"
        );
        assert_eq!(skill_description_from_body("# Only a heading"), "");
    }

    #[tokio::test]
    async fn lists_and_loads_lazy_skills() {
        let host = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(host.path().join("deploy/scripts")).unwrap();
        std::fs::write(
            host.path().join("deploy/SKILL.md"),
            "---\ndescription: Deploys the app\n---\n# Deploy\n",
        )
        .unwrap();
        std::fs::write(host.path().join("deploy/scripts/run.sh"), "echo deployed").unwrap();
        std::fs::write(host.path().join("not-a-skill.txt"), "x").unwrap();

        let source = Arc::new(LocalDirLazySkillSource::new(host.path()));
        let listed = source.list_skill_metadata().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "deploy");
        assert_eq!(listed[0].description, "Deploys the app");
        assert!(listed[0].deferred);
        assert!(source.load_skill("../escape").await.is_err());
        assert!(source.load_skill("missing").await.is_err());

        let skills = Skills::lazy_from(source);
        let manifest = Capability::<()>::process_manifest(&skills, Manifest::new());
        let client = UnixLocalSandboxClient::new();
        let session = client.create(Some(manifest)).await.unwrap();
        session.start().await.unwrap();

        let path = skills.load_skill(session.as_ref(), "deploy").await.unwrap();
        assert!(path.ends_with("/.agents/deploy"));
        let body = session.read_file(".agents/deploy/SKILL.md").await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("# Deploy"));
        let script = session
            .read_file(".agents/deploy/scripts/run.sh")
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&script), "echo deployed");

        let instructions = Capability::<()>::instructions(&skills, session.manifest())
            .await
            .unwrap()
            .unwrap();
        assert!(instructions.contains("## Skills"));
        assert!(instructions.contains("- deploy: Deploys the app"));
        assert!(instructions.contains("load_skill"));
        assert!(instructions.contains("### How to use skills"));

        client.delete(session).await.unwrap();
    }
}
