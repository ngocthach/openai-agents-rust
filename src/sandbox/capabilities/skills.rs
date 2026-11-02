//! The skills capability. Mirrors `agents.sandbox.capabilities.skills`.
//!
//! A skill is a `SKILL.md` file plus optional `scripts/`, `references/` and `assets/`
//! directories, materialized under a discovery root (`.agents` by default). The capability lists
//! the available skills in the agent instructions and, for lazy sources, exposes a `load_skill`
//! tool that copies a skill into the workspace on demand.
//!
//! Skill metadata is read from the `SKILL.md` front matter with [`parse_frontmatter`], a small
//! subset of YAML that matches Python's hand-written reader rather than a full loader.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::error::{AgentsError, Result};
use crate::sandbox::capabilities::{Capability, CapabilityBinding};
use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::manifest::Manifest;
use crate::sandbox::session::SandboxSession;
use crate::sandbox::types::User;
use crate::sandbox::workspace_paths::{
    SandboxPathGrant, SandboxWorkspaceScope, is_windows_absolute, normpath, to_posix,
};
use crate::tool::{FunctionTool, Tool};

/// The default discovery root skills are materialized into.
pub const DEFAULT_SKILLS_PATH: &str = ".agents";

/// The name of the skill-loading tool.
pub const LOAD_SKILL_TOOL_NAME: &str = "load_skill";

/// The description given when a skill's front matter has none.
const NO_DESCRIPTION: &str = "No description provided.";

const SKILLS_SECTION_INTRO: &str = "A skill is a set of local instructions to follow that is \
    stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry \
    includes a name, description, and file path so you can open the source for full instructions \
    when using a specific skill.";

const SKILL_PATH_GUIDANCE: [&str; 2] = [
    "- Skill paths: Treat each listed path as the skill root. Resolve relative paths in \
     `SKILL.md`, including `scripts/`, `references/`, and `assets/`, against that root rather \
     than the shell working directory.",
    "- Shared resources: Skill files belong to the sandbox session and may be visible to other \
     runs. Unless the task explicitly requires editing a skill, invoke scripts through the \
     listed skill root and write task inputs, outputs, caches, and temporary files in the run \
     working directory.",
];

const SCOPED_SKILL_PATH_ERROR: &str =
    "skill path must be non-empty and workspace-relative when sandbox.cwd is configured";

/// The `### How to use skills` section, with the one step that differs for lazy sources.
fn how_to_use_section(lazy: bool) -> String {
    let (discovery, step_one) = if lazy {
        (
            "- Discovery: The list above is the skill index available in this session (name + \
             description + workspace path). In lazy mode, those paths are loaded on demand \
             instead of being present up front.",
            "  1) After deciding to use a lazy skill, call `load_skill` for that skill first, \
             then open its `SKILL.md`.",
        )
    } else {
        (
            "- Discovery: The list above is the skills available in this session (name + \
             description + file path). Skill bodies live on disk at the listed paths.",
            "  1) After deciding to use a skill, open its `SKILL.md`. Read only enough to follow \
             the workflow.",
        )
    };
    [
        "### How to use skills",
        discovery,
        "- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task \
         clearly matches a skill's description shown above, you must use that skill for that \
         turn. Multiple mentions mean use them all. Do not carry skills across turns unless \
         re-mentioned.",
        "- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so \
         briefly and continue with the best fallback.",
        "- How to use a skill (progressive disclosure):",
        step_one,
        "  2) If `SKILL.md` points to extra folders such as `references/`, load only the \
         specific files needed for the request; don't bulk-load everything.",
        "  3) If `scripts/` exist, prefer running or patching them instead of retyping large \
         code blocks.",
        "  4) If `assets/` or templates exist, reuse them instead of recreating from scratch.",
        "- Coordination and sequencing:",
        "  - If multiple skills apply, choose the minimal set that covers the request and state \
         the order you'll use them.",
        "  - Announce which skill(s) you're using and why (one short line). If you skip an \
         obvious skill, say why.",
        "- Context hygiene:",
        "  - Keep context small: summarize long sections instead of pasting them; only load \
         extra files when needed.",
        "  - Avoid deep reference-chasing: prefer opening only files directly linked from \
         `SKILL.md` unless you're blocked.",
        "  - When variants exist (frameworks, providers, domains), pick only the relevant \
         reference file(s) and note that choice.",
        "- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear \
         instructions), state the issue, pick the next-best approach, and continue.",
    ]
    .join("\n")
}

fn config_error(message: impl Into<String>) -> SandboxError {
    SandboxError::new(ErrorCode::SkillsConfigInvalid, message).with_retryable(false)
}

/// What the model is told about one available skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMetadata {
    /// The skill name shown to the model.
    pub name: String,
    /// What the skill is for.
    pub description: String,
    /// Where the skill lives, relative to the workspace root, with POSIX separators.
    pub path: String,
}

/// The result of `load_skill`. Mirrors the `dict[str, str]` Python returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedSkill {
    /// `loaded`, or `already_loaded` when the skill was in the workspace already.
    pub status: String,
    /// The skill's name.
    pub skill_name: String,
    /// Where the skill lives, as the model should address it.
    pub path: String,
}

impl LoadedSkill {
    fn to_json(&self) -> serde_json::Value {
        json!({"status": self.status, "skill_name": self.skill_name, "path": self.path})
    }
}

// ----------------------------------------------------------------------------------------------
// Front matter
// ----------------------------------------------------------------------------------------------

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// Whether `value` is a block scalar header: `>` or `|`, an optional indentation digit and
/// chomping sign in either order, and an optional trailing comment.
fn block_scalar_header(value: &str) -> Option<&str> {
    let first = value.split_whitespace().next()?;
    let mut chars = first.chars();
    let style = chars.next()?;
    if style != '>' && style != '|' {
        return None;
    }
    let indicators: Vec<char> = chars.collect();
    let valid = match indicators.as_slice() {
        [] => true,
        [a] => a.is_ascii_digit() && *a != '0' || matches!(a, '+' | '-'),
        [a, b] => {
            (a.is_ascii_digit() && *a != '0' && matches!(b, '+' | '-'))
                || (matches!(a, '+' | '-') && b.is_ascii_digit() && *b != '0')
        }
        _ => false,
    };
    if !valid {
        return None;
    }
    // Only whitespace and a `#` comment may follow the header.
    let rest = value[first.len()..].trim_start();
    (rest.is_empty() || (value[first.len()..].starts_with([' ', '\t']) && rest.starts_with('#')))
        .then_some(first)
}

/// Fold ordinary lines, keeping paragraphs and more-indented block content.
fn fold_lines(lines: &[String]) -> String {
    let mut out = String::new();
    let mut previous = String::new();
    let mut blank_lines = 0usize;
    let mut any = false;
    for line in lines {
        if line.is_empty() {
            blank_lines += 1;
            continue;
        }
        if any {
            let more_indented = previous.starts_with([' ', '\t']) || line.starts_with([' ', '\t']);
            if more_indented {
                out.push_str(&"\n".repeat(blank_lines + 1));
            } else if blank_lines > 0 {
                out.push_str(&"\n".repeat(blank_lines));
            } else {
                out.push(' ');
            }
        }
        out.push_str(line);
        previous = line.clone();
        blank_lines = 0;
        any = true;
    }
    out
}

/// Consume a field's indented body, stripping a block scalar's content indentation.
fn take_continuation_lines(
    lines: &[&str],
    start: usize,
    end: usize,
    key_indent: usize,
    block_header: Option<&str>,
) -> (Vec<String>, usize) {
    let mut content_indent: Option<usize> = block_header.and_then(|header| {
        header[1..]
            .chars()
            .find(|c| c.is_ascii_digit())
            .map(|digit| key_indent + digit.to_digit(10).unwrap_or(0) as usize)
    });
    let mut body: Vec<&str> = Vec::new();
    let mut index = start;
    while index < end {
        let line = lines[index];
        let stripped = line.trim();
        // Plain-scalar comments may have any indentation and are not scalar content.
        if block_header.is_none() && stripped.starts_with('#') {
            index += 1;
            continue;
        }
        if !stripped.is_empty() {
            let indent = indent_of(line);
            // A dedented comment ends a block just like any other nonblank line.
            if indent < content_indent.unwrap_or(key_indent + 1) {
                break;
            }
            if block_header.is_some() && content_indent.is_none() {
                content_indent = Some(indent);
            }
        }
        body.push(line);
        index += 1;
    }
    while body.last().is_some_and(|line| line.trim().is_empty()) {
        body.pop();
    }
    let body = match content_indent {
        Some(indent) => body
            .into_iter()
            .map(|line| {
                // Slice on characters: the indent counts spaces, never part of a character.
                line.chars().skip(indent).collect::<String>()
            })
            .collect(),
        None => body.into_iter().map(str::to_string).collect(),
    };
    (body, index)
}

/// Python's `str.splitlines`, which also splits on form feed, vertical tab and Unicode
/// separators, but not on a bare `\r` inside a `\r\n`.
fn split_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < text.len() {
        let rest = &text[index..];
        let ch = rest.chars().next().unwrap_or('\0');
        let is_break = matches!(
            ch,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            lines.push(&text[start..index]);
            index += ch.len_utf8();
            if ch == '\r' && bytes.get(index) == Some(&b'\n') {
                index += 1;
            }
            start = index;
        } else {
            index += ch.len_utf8();
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Read the string metadata subset used by skill indexes, without a YAML dependency.
///
/// Mirrors Python's `_parse_frontmatter`. Supports single-line values, indented wrapped plain
/// values, and folded/literal blocks with optional indentation/chomping indicators and header
/// comments. Block values have outer whitespace removed for the index; chomping indicators are
/// accepted but do not change that normalization. Internal paragraph breaks and block
/// indentation survive.
///
/// This is not a general YAML loader: tags, aliases, collections, multiline quoted values and
/// quoted escape decoding are not interpreted. Nested fields are consumed without promoting
/// their keys. Supply explicit [`Skill`] metadata or a [`LazySkillSource`] when a full YAML
/// loader is needed.
pub fn parse_frontmatter(markdown: &str) -> BTreeMap<String, String> {
    let lines = split_lines(markdown);
    let mut metadata = BTreeMap::new();
    if lines.first().is_none_or(|line| line.trim() != "---") {
        return metadata;
    }
    // Indented document markers inside scalar content do not close the front matter.
    let delimiter_indent = indent_of(lines[0]);
    let Some(end_index) = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, line)| line.trim() == "---" && indent_of(line) <= delimiter_indent)
        .map(|(index, _)| index)
    else {
        return metadata;
    };

    let mut index = 1;
    while index < end_index {
        let line = lines[index];
        index += 1;
        let stripped = line.trim();
        if stripped.is_empty() || stripped.starts_with('#') || !stripped.contains(':') {
            continue;
        }
        let (key, value) = stripped.split_once(':').unwrap_or((stripped, ""));
        let mut parsed = value.trim().to_string();
        // Header comments cannot supply an indentation or chomping indicator.
        let block_header = block_scalar_header(&parsed).map(str::to_string);
        let key_indent = indent_of(line);
        let (continuation, next) = take_continuation_lines(
            &lines,
            index,
            end_index,
            key_indent,
            block_header.as_deref(),
        );
        index = next;

        if let Some(header) = &block_header {
            parsed = if header.starts_with('|') {
                continuation.join("\n")
            } else {
                fold_lines(&continuation)
            }
            .trim()
            .to_string();
        } else if parsed.chars().count() >= 2
            && parsed.chars().next() == parsed.chars().last()
            && matches!(parsed.chars().next(), Some('\'' | '"'))
        {
            let inner: Vec<char> = parsed.chars().collect();
            parsed = inner[1..inner.len() - 1].iter().collect();
        } else if !parsed.is_empty() && !continuation.is_empty() {
            let mut parts = vec![parsed.clone()];
            parts.extend(continuation.iter().map(|item| item.trim().to_string()));
            parsed = fold_lines(&parts);
        }
        metadata.insert(key.trim().to_string(), parsed);
    }
    metadata
}

// ----------------------------------------------------------------------------------------------
// Paths and entries
// ----------------------------------------------------------------------------------------------

/// Validate a skills-relative path and return it normalized.
///
/// Mirrors Python's `_validate_relative_path`: the error names the `field`, the offending `path`
/// and a `reason` (`absolute`, `escape_root` or `empty`).
fn validate_relative_path(value: &str, field: &str) -> SandboxResult<String> {
    let fail = |message: String, path: &str, reason: &str| {
        config_error(message)
            .with_context("field", field)
            .with_context("path", path)
            .with_context("reason", reason)
    };
    let posix = to_posix(value);
    if is_windows_absolute(value) {
        return Err(fail(
            format!("{field} must be a relative path"),
            &posix,
            "absolute",
        ));
    }
    if posix.starts_with('/') {
        return Err(fail(
            format!("{field} must be a relative path"),
            &posix,
            "absolute",
        ));
    }
    if posix.split('/').any(|part| part == "..") {
        return Err(fail(
            format!("{field} must not escape the skills root"),
            &posix,
            "escape_root",
        ));
    }
    let normalized = normpath(&posix);
    if normalized.is_empty() || normalized == "." {
        return Err(fail(format!("{field} must be non-empty"), &posix, "empty"));
    }
    Ok(normalized)
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
    /// Compatibility note.
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

    /// Validate the name and the entry paths.
    ///
    /// Two spellings of one path (`run.sh` and `./run.sh`) are a duplicate after normalization,
    /// which a `BTreeMap` keyed by the spelling cannot rule out on its own.
    pub fn validate(&self) -> SandboxResult<()> {
        let context = |error: SandboxError| error.with_context("skill_name", self.name.clone());
        validate_relative_path(&self.name, "name").map_err(context)?;
        for (field, entries) in [
            ("scripts", &self.scripts),
            ("references", &self.references),
            ("assets", &self.assets),
        ] {
            let mut seen = BTreeSet::new();
            for key in entries.keys() {
                let normalized = validate_relative_path(key, field)
                    .map_err(|error| context(error).with_context("entry_path", key.clone()))?;
                if !seen.insert(normalized.clone()) {
                    return Err(
                        config_error(format!("duplicate entry path in skill {field}"))
                            .with_context("skill_name", self.name.clone())
                            .with_context("field", field)
                            .with_context("entry_path", normalized),
                    );
                }
            }
        }
        Ok(())
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
                let normalized = normpath(&to_posix(name));
                group_dir = group_dir.with_child(normalized, entry.clone());
            }
            dir = dir.with_child(group, group_dir);
        }
        dir
    }

    /// The metadata shown to the model.
    pub fn metadata(&self, skills_path: &str) -> SkillMetadata {
        SkillMetadata {
            name: self.name.clone(),
            description: self.description.clone(),
            path: format!("{skills_path}/{}", self.name),
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Lazy sources
// ----------------------------------------------------------------------------------------------

/// A source that lists skills up front and materializes their bodies on demand.
#[async_trait]
pub trait LazySkillSource: Send + Sync + fmt::Debug {
    /// The skills this source offers. `source_grants` are the manifest's extra path grants, which
    /// decide whether a host directory outside the working directory may be read at all.
    fn list_skill_metadata(
        &self,
        skills_path: &str,
        source_grants: &[SandboxPathGrant],
    ) -> Vec<SkillMetadata>;

    /// Materialize one skill into the session's workspace.
    async fn load_skill(
        &self,
        skill_name: &str,
        session: &dyn SandboxSession,
        skills_path: &str,
        user: Option<&User>,
    ) -> SandboxResult<LoadedSkill>;
}

/// A lazy skill source backed by a directory on the machine running the SDK.
///
/// Each immediate subdirectory holding a `SKILL.md` regular file is one skill; the name and
/// description come from the file's front matter.
#[derive(Debug, Clone)]
pub struct LocalDirLazySkillSource {
    source: PathBuf,
    /// The configured source entry. A loaded skill is materialized through a copy of it, so it
    /// keeps the permissions, group and copy concurrency the eager `from_` path applies.
    entry: Entry,
}

impl LocalDirLazySkillSource {
    /// A source reading skills from `source` on the host.
    pub fn new(source: impl Into<PathBuf>) -> Self {
        let source = source.into();
        LocalDirLazySkillSource {
            entry: Entry::local_dir(source.clone()),
            source,
        }
    }

    /// A source configured by a `local_dir` entry, as Python's `LocalDirLazySkillSource(source=
    /// LocalDir(...))`: the entry's permissions and group are applied to every loaded skill. An
    /// entry that is not a `local_dir` with a source path is refused.
    pub fn from_entry(entry: Entry) -> SandboxResult<Self> {
        match &entry.kind {
            EntryKind::LocalDir { src: Some(src) } => Ok(LocalDirLazySkillSource {
                source: src.clone(),
                entry,
            }),
            _ => Err(config_error(
                "a lazy local directory source needs a `local_dir` entry with a source path",
            )),
        }
    }

    /// The configured source entry.
    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    /// The host directory holding the skills.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// The source directory, when it exists and may be read: under the working directory, or
    /// under a host path an extra grant covers. Mirrors `LocalDir._resolve_local_dir_src_root`.
    fn src_root(&self, source_grants: &[SandboxPathGrant]) -> Option<PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let absolute = if self.source.is_absolute() {
            self.source.clone()
        } else {
            cwd.join(&self.source)
        };
        let input = std::fs::canonicalize(&absolute).ok()?;
        let base = std::fs::canonicalize(&cwd).unwrap_or(cwd);
        let granted = source_grants.iter().any(|grant| {
            std::fs::canonicalize(grant.host_source())
                .map(|root| input.starts_with(root))
                .unwrap_or(false)
        });
        if !input.starts_with(&base) && !granted {
            return None;
        }
        input.is_dir().then_some(input)
    }

    fn list_in(&self, src_root: &Path, skills_path: &str) -> Vec<SkillMetadata> {
        let Ok(children) = std::fs::read_dir(src_root) else {
            return Vec::new();
        };
        let mut names: Vec<String> = children
            .filter_map(|child| child.ok())
            .filter_map(|child| child.file_name().into_string().ok())
            .collect();
        names.sort();
        let mut metadata = Vec::new();
        for name in names {
            let dir = src_root.join(&name);
            let Ok(dir_meta) = std::fs::symlink_metadata(&dir) else {
                continue;
            };
            if !dir_meta.is_dir() {
                continue;
            }
            let skill_md = dir.join("SKILL.md");
            let Ok(md_meta) = std::fs::symlink_metadata(&skill_md) else {
                continue;
            };
            if !md_meta.is_file() {
                continue;
            }
            let Ok(markdown) = std::fs::read_to_string(&skill_md) else {
                continue;
            };
            let frontmatter = parse_frontmatter(&markdown);
            metadata.push(SkillMetadata {
                name: frontmatter
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| name.clone()),
                description: frontmatter
                    .get("description")
                    .cloned()
                    .unwrap_or_else(|| NO_DESCRIPTION.to_string()),
                path: format!("{skills_path}/{name}"),
            });
        }
        metadata
    }
}

#[async_trait]
impl LazySkillSource for LocalDirLazySkillSource {
    fn list_skill_metadata(
        &self,
        skills_path: &str,
        source_grants: &[SandboxPathGrant],
    ) -> Vec<SkillMetadata> {
        match self.src_root(source_grants) {
            Some(root) => self.list_in(&root, skills_path),
            None => Vec::new(),
        }
    }

    async fn load_skill(
        &self,
        skill_name: &str,
        session: &dyn SandboxSession,
        skills_path: &str,
        user: Option<&User>,
    ) -> SandboxResult<LoadedSkill> {
        let grants = session.extra_path_grants();
        let Some(src_root) = self.src_root(&grants) else {
            return Err(config_error("lazy skill source directory is unavailable")
                .with_context("skill_name", skill_name));
        };
        let matches: Vec<SkillMetadata> = self
            .list_in(&src_root, skills_path)
            .into_iter()
            .filter(|skill| {
                skill.name == skill_name || skill.path.rsplit('/').next() == Some(skill_name)
            })
            .collect();
        let metadata = match matches.as_slice() {
            [] => {
                return Err(config_error("lazy skill not found")
                    .with_context("skill_name", skill_name)
                    .with_context("skills_path", skills_path));
            }
            [only] => only.clone(),
            many => {
                return Err(config_error("lazy skill name is ambiguous")
                    .with_context("skill_name", skill_name)
                    .with_context(
                        "matching_paths",
                        many.iter()
                            .map(|skill| skill.path.clone())
                            .collect::<Vec<_>>()
                            .join(", "),
                    ));
            }
        };

        let directory = metadata
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&metadata.name)
            .to_string();
        let dest = format!(
            "{}/{}",
            session.workspace_root().trim_end_matches('/'),
            metadata.path
        );
        match session
            .read_file_expecting(
                &format!("{dest}/SKILL.md"),
                user,
                &[ErrorCode::WorkspaceReadNotFound],
            )
            .await
        {
            Ok(_) => {
                return Ok(LoadedSkill {
                    status: "already_loaded".to_string(),
                    skill_name: metadata.name,
                    path: metadata.path,
                });
            }
            // A missing SKILL.md just means the skill has not been loaded yet. Anything else is
            // the session failing to answer, which the caller has to hear about.
            Err(error) if error.code == ErrorCode::WorkspaceReadNotFound => {}
            Err(error) => return Err(error),
        }

        // Materialize through the configured source so the loaded skill keeps what the eager
        // `from_` path applies.
        let mut source = self.entry.clone();
        source.kind = EntryKind::LocalDir {
            src: Some(src_root.join(directory)),
        };
        source.apply_as(session, &dest, user).await?;
        Ok(LoadedSkill {
            status: "loaded".to_string(),
            skill_name: metadata.name,
            path: metadata.path,
        })
    }
}

/// Arguments of the `load_skill` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct LoadSkillArgs {
    /// Name of the skill to load into the workspace.
    pub skill_name: String,
}

// ----------------------------------------------------------------------------------------------
// The capability
// ----------------------------------------------------------------------------------------------

/// Where skills come from. Python lets a caller set several and rejects the combination; one
/// variant per source makes the combination unrepresentable.
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

fn manifest_key(key: &str) -> String {
    normpath(&to_posix(key))
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
        validate_relative_path(&self.skills_path, "skills_path")?;
        match &self.source {
            SkillSource::Inline(skills) => {
                if skills.is_empty() {
                    return Err(config_error(
                        "skills capability requires `skills`, `from_`, or `lazy_from`",
                    )
                    .with_context("field", "skills"));
                }
                let mut seen = BTreeSet::new();
                for skill in skills {
                    skill.validate()?;
                    let name = validate_relative_path(&skill.name, "skills[].name")
                        .map_err(|e| e.with_context("skill_name", skill.name.clone()))?;
                    if !seen.insert(name) {
                        return Err(
                            config_error(format!("duplicate skill name: {}", skill.name))
                                .with_context("field", "skills[].name")
                                .with_context("skill_name", skill.name.clone()),
                        );
                    }
                }
                Ok(())
            }
            SkillSource::From(entry) => {
                if !entry.is_dir() {
                    return Err(config_error("`from_` must be a directory-like artifact")
                        .with_context("field", "from_"));
                }
                Ok(())
            }
            SkillSource::Lazy(_) => Ok(()),
        }
    }

    /// The manifest with this capability's skills added, or why it cannot be.
    ///
    /// Mirrors Python's `process_manifest`, which raises: a lazy source reserves the whole skills
    /// namespace and refuses any manifest entry that equals, contains or sits under it; an eager
    /// `from_` directory takes the skills root unless the manifest already has a directory
    /// there; a literal skill is added unless an identical one is present, and refused when a
    /// different entry holds its path.
    pub fn try_process_manifest(&self, mut manifest: Manifest) -> SandboxResult<Manifest> {
        let root = manifest_key(&self.skills_path);
        let existing: BTreeSet<String> = manifest
            .entries()
            .keys()
            .map(|key| manifest_key(key))
            .collect();
        match &self.source {
            SkillSource::Lazy(_) => {
                let mut overlaps: Vec<String> = existing
                    .iter()
                    .filter(|path| {
                        **path == root
                            || root.starts_with(&format!("{path}/"))
                            || path.starts_with(&format!("{root}/"))
                    })
                    .cloned()
                    .collect();
                overlaps.sort();
                if !overlaps.is_empty() {
                    return Err(config_error(
                        "skills lazy_from path overlaps existing manifest entries",
                    )
                    .with_context("path", root)
                    .with_context("source", "lazy_from")
                    .with_context("overlaps", overlaps.join(", ")));
                }
                Ok(manifest)
            }
            SkillSource::From(entry) => {
                if existing.contains(&root) {
                    let current = manifest
                        .entries()
                        .iter()
                        .find(|(key, _)| manifest_key(key) == root)
                        .map(|(_, entry)| entry.clone());
                    return match current {
                        Some(current) if current.is_dir() => Ok(manifest),
                        Some(_) => Err(config_error("skills root path already exists in manifest")
                            .with_context("path", root)
                            .with_context("source", "from_")),
                        None => Err(config_error("skills root path lookup failed")
                            .with_context("path", root)
                            .with_context("source", "from_")),
                    };
                }
                Ok(manifest.with_entry(root, (**entry).clone()))
            }
            SkillSource::Inline(skills) => {
                let mut taken = existing;
                for skill in skills {
                    let relative = format!("{root}/{}", skill.name);
                    let rendered = skill.as_dir_entry();
                    if taken.contains(&relative) {
                        let current = manifest
                            .entries()
                            .iter()
                            .find(|(key, _)| manifest_key(key) == relative)
                            .map(|(_, entry)| entry.clone());
                        match current {
                            Some(current) if entries_match(&current, &rendered) => continue,
                            _ => {
                                return Err(config_error("skill path already exists in manifest")
                                    .with_context("path", relative)
                                    .with_context("skill_name", skill.name.clone()));
                            }
                        }
                    }
                    manifest = manifest.with_entry(relative.clone(), rendered);
                    taken.insert(relative);
                }
                Ok(manifest)
            }
        }
    }

    /// The path of `path` as the model should address it: unchanged without a run working
    /// directory, absolute with one, because a path relative to the workspace root stops
    /// resolving once a command changes directory.
    fn model_skill_path(
        &self,
        scope: &SandboxWorkspaceScope,
        manifest: &Manifest,
        skill_name: &str,
        path: &str,
    ) -> SandboxResult<String> {
        if scope.cwd().is_none() {
            return Ok(path.replace('\\', "/"));
        }
        scope
            .model_resource_path(manifest.root(), path)
            .map_err(|_| {
                config_error(SCOPED_SKILL_PATH_ERROR)
                    .with_context("skill_name", skill_name)
                    .with_context("field", "path")
                    .with_context("path", path)
                    .with_context("reason", "invalid")
            })
    }

    /// Materialize one lazily sourced skill into the live workspace.
    ///
    /// Returns what the model is told: `status`, `skill_name` and `path`, the path addressed
    /// the way the run's working directory requires.
    pub async fn load_skill(
        &self,
        session: &dyn SandboxSession,
        scope: &SandboxWorkspaceScope,
        user: Option<&User>,
        skill_name: &str,
    ) -> SandboxResult<LoadedSkill> {
        let SkillSource::Lazy(source) = &self.source else {
            return Err(
                config_error("load_skill is only available when lazy_from is configured")
                    .with_context("skill_name", skill_name),
            );
        };
        let mut result = source
            .load_skill(skill_name, session, &self.skills_path, user)
            .await?;
        if scope.cwd().is_some() {
            if result.path.is_empty() {
                return Err(config_error(SCOPED_SKILL_PATH_ERROR)
                    .with_context("skill_name", skill_name)
                    .with_context("field", "path")
                    .with_context("reason", "missing"));
            }
            result.path =
                self.model_skill_path(scope, session.manifest(), skill_name, &result.path)?;
        }
        Ok(result)
    }

    /// Skills found in the workspace by listing the skills root and reading each `SKILL.md`.
    async fn runtime_metadata(
        &self,
        binding: &CapabilityBinding,
        manifest: &Manifest,
    ) -> Vec<SkillMetadata> {
        let root = format!(
            "{}/{}",
            manifest.root().trim_end_matches('/'),
            self.skills_path
        );
        let Ok(entries) = binding.session.ls(&root).await else {
            return Vec::new();
        };
        let mut metadata = Vec::new();
        for entry in entries {
            if !entry.is_dir {
                continue;
            }
            let dir_name = entry
                .path
                .rsplit('/')
                .next()
                .unwrap_or(&entry.path)
                .to_string();
            let Ok(bytes) = binding
                .session
                .read_file_as(&format!("{}/SKILL.md", entry.path), binding.run_as.as_ref())
                .await
            else {
                continue;
            };
            let frontmatter = parse_frontmatter(&String::from_utf8_lossy(&bytes));
            metadata.push(SkillMetadata {
                name: frontmatter
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| dir_name.clone()),
                description: frontmatter
                    .get("description")
                    .cloned()
                    .unwrap_or_else(|| NO_DESCRIPTION.to_string()),
                path: format!("{}/{dir_name}", self.skills_path),
            });
        }
        metadata
    }

    async fn skill_metadata(
        &self,
        binding: &CapabilityBinding,
        manifest: &Manifest,
    ) -> Vec<SkillMetadata> {
        let mut metadata: Vec<SkillMetadata> = Vec::new();
        match &self.source {
            SkillSource::Inline(skills) => {
                metadata.extend(skills.iter().map(|skill| skill.metadata(&self.skills_path)));
            }
            SkillSource::Lazy(source) => {
                metadata.extend(
                    source.list_skill_metadata(&self.skills_path, manifest.extra_path_grants()),
                );
            }
            SkillSource::From(_) => {
                metadata.extend(self.runtime_metadata(binding, manifest).await);
            }
        }
        if let SkillSource::From(entry) = &self.source
            && metadata.is_empty()
            && let EntryKind::Dir { children } = &entry.kind
        {
            for (key, child) in children {
                if !matches!(child.kind, EntryKind::Dir { .. }) {
                    continue;
                }
                let name = to_posix(key);
                metadata.push(SkillMetadata {
                    description: child
                        .description
                        .clone()
                        .unwrap_or_else(|| NO_DESCRIPTION.to_string()),
                    path: format!("{}/{name}", self.skills_path),
                    name,
                });
            }
        }
        let mut deduped: BTreeMap<(String, String), SkillMetadata> = BTreeMap::new();
        for item in metadata {
            deduped.insert((item.name.clone(), item.path.clone()), item);
        }
        let mut sorted: Vec<SkillMetadata> = deduped.into_values().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        sorted
    }
}

/// Whether two entries describe the same thing, for the "identical skill already present" rule.
fn entries_match(left: &Entry, right: &Entry) -> bool {
    format!("{left:?}") == format!("{right:?}")
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
        // A configuration error cannot be returned from here; `check_manifest` reports it when
        // the agent is prepared, so the manifest is left as it was rather than half-processed.
        match self.try_process_manifest(manifest.clone()) {
            Ok(processed) => processed,
            Err(_) => manifest,
        }
    }

    fn check_manifest(&self, manifest: &Manifest) -> Result<()> {
        self.validate()?;
        self.try_process_manifest(manifest.clone())?;
        Ok(())
    }

    fn tools(&self, binding: &CapabilityBinding) -> Result<Vec<Tool<C>>> {
        if !matches!(self.source, SkillSource::Lazy(_)) {
            return Ok(Vec::new());
        }
        let capability = self.clone();
        let session = binding.session.clone();
        let scope = binding.scope.clone();
        let run_as = binding.run_as.clone();
        let tool = FunctionTool::<C>::new(
            LOAD_SKILL_TOOL_NAME,
            "Load a single lazily configured skill into the sandbox so its SKILL.md, scripts, \
             references, and assets can be read from the workspace.",
            move |_ctx, args: LoadSkillArgs| {
                let capability = capability.clone();
                let session = session.clone();
                let scope = scope.clone();
                let run_as = run_as.clone();
                async move {
                    capability
                        .load_skill(session.as_ref(), &scope, run_as.as_ref(), &args.skill_name)
                        .await
                        .map(|loaded| loaded.to_json().to_string())
                        .map_err(AgentsError::from)
                }
            },
        )
        .with_strict_json_schema(false);
        Ok(vec![Tool::Function(Arc::new(tool))])
    }

    async fn bound_instructions(
        &self,
        binding: &CapabilityBinding,
        manifest: &Manifest,
    ) -> Result<Option<String>> {
        let skills = self.skill_metadata(binding, manifest).await;
        if skills.is_empty() {
            return Ok(None);
        }
        let lazy = matches!(self.source, SkillSource::Lazy(_));
        let mut lines = vec![
            "## Skills".to_string(),
            SKILLS_SECTION_INTRO.to_string(),
            "### Available skills".to_string(),
        ];
        for skill in &skills {
            let path = self.model_skill_path(&binding.scope, manifest, &skill.name, &skill.path)?;
            lines.push(format!(
                "- {}: {} (file: {path})",
                skill.name, skill.description
            ));
        }
        if binding.scope.cwd().is_some() {
            lines.push("### Run-scoped skill paths".to_string());
            lines.extend(SKILL_PATH_GUIDANCE.iter().map(|line| line.to_string()));
        }
        if lazy {
            lines.extend([
                "### Lazy loading".to_string(),
                "- These skills are indexed for planning, but they are not materialized in the \
                 workspace yet."
                    .to_string(),
                "- Call `load_skill` with a single skill name from the list before reading its \
                 `SKILL.md` or other files from the workspace."
                    .to_string(),
                "- `load_skill` stages exactly one skill under the listed path. If you need more \
                 than one skill, call it multiple times."
                    .to_string(),
            ]);
        }
        lines.push(how_to_use_section(lazy));
        Ok(Some(lines.join("\n")))
    }
}
