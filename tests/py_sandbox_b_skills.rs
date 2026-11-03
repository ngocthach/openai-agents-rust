//! Port of `tests/sandbox/capabilities/test_skills_capability.py` (and the skill row of
//! `tests/sandbox/test_run_cwd.py`).
//!
//! Python configures a `Skills` capability with several optional sources and rejects combining
//! them; Rust has one constructor per source, so a combination cannot be written. The metadata
//! rows run both through the lazy local directory source and through a workspace that holds the
//! skill (Python's "runtime" mode).
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::RunContext;
use openai_agents_sdk::sandbox::capabilities::skills::parse_frontmatter;
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, LazySkillSource, LocalDirLazySkillSource, Skill, SkillMetadata,
    Skills,
};
use openai_agents_sdk::sandbox::entries::{Entry, EntryKind};
use openai_agents_sdk::sandbox::errors::ErrorCode;
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::session::{SandboxClient, SandboxSession};
use openai_agents_sdk::sandbox::testing::scripted_sandbox_session;
use openai_agents_sdk::sandbox::workspace_paths::{SandboxPathGrant, SandboxWorkspaceScope};
use openai_agents_sdk::tool::{Tool, ToolOutput};
use openai_agents_sdk::tool_context::ToolContext;
use serde_json::{Value, json};

fn grant(path: &std::path::Path) -> SandboxPathGrant {
    SandboxPathGrant::new(path.to_string_lossy()).unwrap()
}

fn skill(name: &str, description: &str, content: &str) -> Skill {
    Skill::new(name, description, content)
}

fn binding_for(manifest: Manifest) -> CapabilityBinding {
    CapabilityBinding::new(Arc::new(
        scripted_sandbox_session([]).with_manifest(manifest),
    ))
}

async fn instructions(skills: &Skills, binding: &CapabilityBinding) -> Option<String> {
    let manifest = binding.session.manifest().clone();
    Capability::<()>::bound_instructions(skills, binding, &manifest)
        .await
        .unwrap()
}

/// A source root holding `dynamic-skill/SKILL.md` with `markdown`, outside the working
/// directory, so reading it needs an extra path grant.
fn skill_source(markdown: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap().join("skills");
    std::fs::create_dir_all(root.join("dynamic-skill")).unwrap();
    std::fs::write(root.join("dynamic-skill/SKILL.md"), markdown).unwrap();
    (temp, root)
}

fn lazy_metadata(root: &std::path::Path) -> Vec<SkillMetadata> {
    LocalDirLazySkillSource::new(root).list_skill_metadata(".agents", &[grant(root)])
}

fn expected(name: &str, description: &str) -> Vec<SkillMetadata> {
    vec![SkillMetadata {
        name: name.to_string(),
        description: description.to_string(),
        path: ".agents/dynamic-skill".to_string(),
    }]
}

// ----------------------------------------------------------------------------------------------
// Validation
// ----------------------------------------------------------------------------------------------

/// Python: `test_rejects_duplicate_script_paths_after_normalization`.
#[test]
fn duplicate_script_paths_are_rejected_after_normalization() {
    let error = skill("my-skill", "desc", "literal")
        .with_script("run.sh", Entry::file(b"echo one".to_vec()))
        .with_script("./run.sh", Entry::file(b"echo two".to_vec()))
        .validate()
        .unwrap_err();

    assert_eq!(error.code, ErrorCode::SkillsConfigInvalid);
    assert!(
        error.message.contains("duplicate entry path"),
        "{}",
        error.message
    );
}

/// Python: `TestSkillsValidation::test_requires_at_least_one_source`.
#[test]
fn at_least_one_source_is_required() {
    let error = Skills::from_skills(vec![]).validate().unwrap_err();
    assert_eq!(error.code, ErrorCode::SkillsConfigInvalid);
}

/// Python: `test_rejects_non_directory_from_artifact`.
#[test]
fn a_non_directory_from_entry_is_rejected() {
    let error = Skills::from_entry(Entry::file(b"not-a-dir".to_vec()))
        .validate()
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::SkillsConfigInvalid);
}

/// Python: `test_rejects_duplicate_skill_names`.
#[test]
fn duplicate_skill_names_are_rejected() {
    let error = Skills::from_skills(vec![
        skill("dup", "first", "a"),
        skill("dup", "second", "b"),
    ])
    .validate()
    .unwrap_err();
    assert!(
        error.message.contains("duplicate skill name: dup"),
        "{}",
        error.message
    );
}

/// Python: `test_rejects_absolute_skills_path`, `test_rejects_windows_drive_absolute_skills_path`
/// and `test_rejects_escape_root_skills_path`.
#[test]
fn bad_skills_paths_are_rejected_with_a_reason() {
    for (path, reason, shown) in [
        ("/skills", "absolute", "/skills"),
        ("C:\\skills", "absolute", "C:/skills"),
        ("../skills", "escape_root", "../skills"),
    ] {
        let error = Skills::from_skills(vec![skill("my-skill", "desc", "literal")])
            .with_skills_path(path)
            .validate()
            .unwrap_err();

        assert_eq!(
            error.context.get("field").map(String::as_str),
            Some("skills_path")
        );
        assert_eq!(error.context.get("path").map(String::as_str), Some(shown));
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some(reason)
        );
    }
}

// ----------------------------------------------------------------------------------------------
// Manifest
// ----------------------------------------------------------------------------------------------

fn children(entry: &Entry) -> Vec<String> {
    match &entry.kind {
        EntryKind::Dir { children } => children.keys().cloned().collect(),
        other => panic!("expected a directory, got {other:?}"),
    }
}

/// Python: `test_literals_materialize_full_skill_structure`.
#[test]
fn literals_materialize_the_full_skill_structure() {
    let skills = Skills::from_skills(vec![
        skill("my-skill", "desc", "Use this skill.")
            .with_script("run.sh", Entry::file(b"echo run".to_vec()))
            .with_reference("docs/readme.md", Entry::file(b"ref".to_vec()))
            .with_asset("images/icon.txt", Entry::file(b"asset".to_vec())),
    ]);

    let processed = skills
        .try_process_manifest(Manifest::new().with_root("/workspace"))
        .unwrap();

    let entry = processed
        .entries()
        .get(".agents/my-skill")
        .expect("the skill entry");
    assert_eq!(
        children(entry),
        ["SKILL.md", "assets", "references", "scripts"]
    );
    let EntryKind::Dir { children: groups } = &entry.kind else {
        unreachable!()
    };
    assert_eq!(children(&groups["scripts"]), ["run.sh"]);
    assert_eq!(children(&groups["references"]), ["docs/readme.md"]);
    assert_eq!(children(&groups["assets"]), ["images/icon.txt"]);
}

/// Python: `test_from_source_is_mapped_to_skills_root` and
/// `test_local_dir_from_source_stays_eager_by_default`.
#[test]
fn a_from_source_is_mapped_to_the_skills_root() {
    let source = Entry::dir().with_child(
        "imported",
        Entry::dir().with_child("SKILL.md", Entry::file(b"imported".to_vec())),
    );
    let processed = Skills::from_entry(source)
        .try_process_manifest(Manifest::new().with_root("/workspace"))
        .unwrap();
    assert!(matches!(
        processed.entries()[".agents"].kind,
        EntryKind::Dir { .. }
    ));

    let eager = Skills::from_entry(Entry::local_dir("skills"))
        .try_process_manifest(Manifest::new().with_root("/workspace"))
        .unwrap();
    assert!(matches!(
        eager.entries()[".agents"].kind,
        EntryKind::LocalDir { .. }
    ));
}

/// Python: `test_lazy_local_dir_source_skips_manifest_materialization` and
/// `test_lazy_local_dir_rejects_overlapping_manifest_entries`.
#[test]
fn a_lazy_source_reserves_the_namespace_without_materializing() {
    let (_temp, root) = skill_source(b"# Skill\n");
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&root)));

    let processed = skills
        .try_process_manifest(Manifest::new().with_root("/workspace"))
        .unwrap();
    assert!(processed.entries().is_empty());

    let error = skills
        .try_process_manifest(
            Manifest::new()
                .with_root("/workspace")
                .with_entry(".agents", Entry::dir()),
        )
        .unwrap_err();
    assert_eq!(
        error.message,
        "skills lazy_from path overlaps existing manifest entries"
    );
    assert_eq!(
        error.context.get("path").map(String::as_str),
        Some(".agents")
    );
    assert_eq!(
        error.context.get("source").map(String::as_str),
        Some("lazy_from")
    );
    assert_eq!(
        error.context.get("overlaps").map(String::as_str),
        Some(".agents")
    );
}

/// Python: `test_literal_skills_allow_existing_manifest_entry_when_content_matches`.
#[test]
fn an_identical_existing_skill_entry_is_allowed() {
    let one = skill("my-skill", "desc", "Use this skill.")
        .with_script("run.sh", Entry::file(b"echo run".to_vec()));
    let rendered = one.as_dir_entry();
    let skills = Skills::from_skills(vec![one]);
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry(".agents/my-skill", rendered);

    let processed = skills.try_process_manifest(manifest.clone()).unwrap();

    assert_eq!(processed.entries().len(), manifest.entries().len());
}

/// Python: `test_process_manifest_rejects_exact_path_collision`.
#[test]
fn a_different_entry_at_a_skill_path_is_a_collision() {
    let skills = Skills::from_skills(vec![skill("my-skill", "desc", "literal")]);
    let manifest = Manifest::new()
        .with_root("/workspace")
        .with_entry(".agents/my-skill", Entry::dir());

    let error = skills.try_process_manifest(manifest.clone()).unwrap_err();

    assert_eq!(error.message, "skill path already exists in manifest");
    // The infallible hook leaves the manifest alone, and preparing the agent reports the error.
    assert!(Capability::<()>::check_manifest(&skills, &manifest).is_err());
}

/// Python: `test_custom_skills_path_is_used_for_manifest_entries`.
#[test]
fn a_custom_skills_path_is_used_for_manifest_entries() {
    let skills = Skills::from_skills(vec![skill("my-skill", "desc", "literal")])
        .with_skills_path(".sandbox/skills");

    let processed = skills
        .try_process_manifest(Manifest::new().with_root("/workspace"))
        .unwrap();

    assert!(processed.entries().contains_key(".sandbox/skills/my-skill"));
}

// ----------------------------------------------------------------------------------------------
// Instructions
// ----------------------------------------------------------------------------------------------

/// Python: `test_instructions_include_root_and_literal_index`.
#[tokio::test]
async fn instructions_include_the_literal_index_sorted_by_name() {
    let skills = Skills::from_skills(vec![
        skill("z-skill", "z description", "z"),
        skill("a-skill", "a description", "a"),
    ]);

    let text = instructions(
        &skills,
        &binding_for(Manifest::new().with_root("/workspace")),
    )
    .await
    .expect("instructions");

    assert!(text.starts_with("## Skills\n"), "{text}");
    assert!(text.contains("### Available skills"));
    assert!(text.contains("### How to use skills"));
    let a = "- a-skill: a description (file: .agents/a-skill)";
    let z = "- z-skill: z description (file: .agents/z-skill)";
    assert!(text.contains(a) && text.contains(z));
    assert!(!text.contains("### Run-scoped skill paths"));
    assert!(text.find(a).unwrap() < text.find(z).unwrap());
}

/// Python: `test_instructions_use_custom_skills_path`.
#[tokio::test]
async fn instructions_use_the_custom_skills_path() {
    let skills = Skills::from_skills(vec![skill("my-skill", "desc", "literal")])
        .with_skills_path(".sandbox/skills");

    let text = instructions(
        &skills,
        &binding_for(Manifest::new().with_root("/workspace")),
    )
    .await
    .unwrap();

    assert!(
        text.contains("- my-skill: desc (file: .sandbox/skills/my-skill)"),
        "{text}"
    );
}

/// Python: `test_instructions_render_session_owned_paths_as_absolute_with_run_cwd`.
#[tokio::test]
async fn instructions_render_absolute_paths_with_a_run_cwd() {
    let skills = Skills::from_skills(vec![skill("my-skill", "desc", "literal")]);
    let binding = binding_for(Manifest::new().with_root("/workspace"))
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).unwrap());

    let text = instructions(&skills, &binding).await.unwrap();

    assert!(
        text.contains("- my-skill: desc (file: /workspace/.agents/my-skill)"),
        "{text}"
    );
    assert!(text.contains("Treat each listed path as the skill root"));
    assert!(text.contains("write task inputs, outputs, caches, and temporary files"));
}

/// Python: `test_instructions_return_none_when_metadata_is_empty`.
#[tokio::test]
async fn instructions_are_none_when_there_is_nothing_to_list() {
    let skills = Skills::from_entry(Entry::dir());

    let text = instructions(
        &skills,
        &binding_for(Manifest::new().with_root("/workspace")),
    )
    .await;

    assert_eq!(text, None);
}

/// Python: `test_lazy_local_dir_metadata_requires_extra_path_grant` and
/// `test_instructions_resolve_opt_in_lazy_local_dir_metadata`.
#[tokio::test]
async fn lazy_metadata_needs_an_extra_path_grant() {
    let (_temp, root) = skill_source(
        b"---\nname: discovered-skill\ndescription: local dir metadata\n---\n# Skill\n",
    );
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&root)));

    let without = instructions(
        &skills,
        &binding_for(Manifest::new().with_root("/workspace")),
    )
    .await;
    assert_eq!(without, None);

    let granted = binding_for(
        Manifest::new()
            .with_root("/workspace")
            .with_extra_path_grant(grant(&root)),
    );
    let text = instructions(&skills, &granted).await.expect("instructions");
    assert!(
        text.contains("- discovered-skill: local dir metadata (file: .agents/dynamic-skill)"),
        "{text}"
    );
    assert!(text.contains("Call `load_skill` with a single skill name from the list"));
    assert!(text.contains("loaded on demand instead of being present up front"));
}

/// Python: `test_instructions_resolve_from_runtime_frontmatter`.
#[tokio::test]
async fn instructions_resolve_from_runtime_frontmatter() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = std::fs::canonicalize(temp.path())
        .unwrap()
        .join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let skills = Skills::from_entry(Entry::dir().with_child(
        "dynamic-skill",
        Entry::dir().with_child(
            "SKILL.md",
            Entry::file(
                b"---\nname: discovered-skill\ndescription: loaded from runtime frontmatter\n---\n\n# Skill\n"
                    .to_vec(),
            ),
        ),
    ));
    let manifest = skills
        .try_process_manifest(Manifest::new().with_root(workspace.to_string_lossy()))
        .unwrap();
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();
    let binding = CapabilityBinding::new(session.clone())
        .with_scope(SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).unwrap());

    let text = instructions(&skills, &binding).await.expect("instructions");

    assert!(
        text.contains(&format!(
            "- discovered-skill: loaded from runtime frontmatter (file: {}/.agents/dynamic-skill)",
            workspace.display()
        )),
        "{text}"
    );
    client.delete(session).await.ok();
}

// ----------------------------------------------------------------------------------------------
// Front matter: multi-line descriptions (Python: 2 source modes x 44 cases)
// ----------------------------------------------------------------------------------------------

fn block_header_cases() -> Vec<(String, String)> {
    let mut cases = Vec::new();
    for style in [">", "|"] {
        for indicator in ["", "-", "+", "2", "2-", "-2", "2+", "+2"] {
            cases.push((
                format!("description: {style}{indicator} # header\n  first\n  last\n\n"),
                if style == ">" {
                    "first last"
                } else {
                    "first\nlast"
                }
                .to_string(),
            ));
        }
    }
    cases
}

fn multiline_description_cases() -> Vec<(String, String)> {
    let fixed: [(&str, &str); 30] = [
        (
            "description: >\n  Use for GitHub issue triage.\n  Triggers: /triage, bug report",
            "Use for GitHub issue triage. Triggers: /triage, bug report",
        ),
        (
            "description: |\n  Use for GitHub issue triage.\n  Triggers: /triage, bug report",
            "Use for GitHub issue triage.\nTriggers: /triage, bug report",
        ),
        (
            "description: >-\n  Use for GitHub issue triage.\n  Triggers: /triage, bug report",
            "Use for GitHub issue triage. Triggers: /triage, bug report",
        ),
        (
            "description: Use for GitHub issue\n  triage, not for PR review.",
            "Use for GitHub issue triage, not for PR review.",
        ),
        (
            "description: >\n  Use for GitHub issue triage.\n\n  Not for PR review.",
            "Use for GitHub issue triage.\nNot for PR review.",
        ),
        (
            "description: >\n  first\n    indented\n  last",
            "first\n  indented\nlast",
        ),
        (
            "description: >\n  first\n\n    indented\n\n  last",
            "first\n\n  indented\n\nlast",
        ),
        (
            "description: >\n  first\n    one\n    two\n  last",
            "first\n  one\n  two\nlast",
        ),
        ("description: >\n  first\n\n\n  last", "first\n\nlast"),
        (
            "description: |\n  first\n    indented\n\n  last",
            "first\n  indented\n\nlast",
        ),
        (
            "description: > # a multi-line description + comment\n  first\n  last",
            "first last",
        ),
        ("description: >\n  first\n  ---\n  last", "first --- last"),
        (
            "description: |\n  first\n  ---\n  name: content\n  last",
            "first\n---\nname: content\nlast",
        ),
        ("description: >2\n    indented\n  last", "indented\nlast"),
        ("description: |2\n    indented\n  last", "indented\nlast"),
        (
            "description: >\n  first\n  # content\n  last",
            "first # content last",
        ),
        (
            "description: >\n    first\n    last\n  # maintainer note\n    # name: ignored",
            "first last",
        ),
        (
            "description: |\n\n    first\n    last\n\n  # maintainer note",
            "first\nlast",
        ),
        (
            "description: >4-\n      first\n    last\n  # maintainer note",
            "first\nlast",
        ),
        (
            "description: |2\n  first\n  last\n # maintainer note",
            "first\nlast",
        ),
        ("description: >2\n # maintainer note\n  # name: ignored", ""),
        ("description: |\n  # content\n  last", "# content\nlast"),
        ("description: first\n\n  last", "first\nlast"),
        (
            "description: \"  quoted content  \"\n\n  # explanation",
            "  quoted content  ",
        ),
        ("description: >\n\n  first  \n  last\n\n", "first   last"),
        ("description: >\n  first\n    \n  last", "first\n  \nlast"),
        ("description: |\n  first\n    \n  last", "first\n  \nlast"),
        (
            "description: >\n  first\n  \tindented\n  last",
            "first\n\tindented\nlast",
        ),
        (
            "description: >\n  Investigate issues.\n  Keep emoji \u{1f40d} and accented caf\u{e9} text.",
            "Investigate issues. Keep emoji \u{1f40d} and accented caf\u{e9} text.",
        ),
        ("description: >\n  first\n  last", "first last"),
    ];
    let mut cases: Vec<(String, String)> = fixed
        .iter()
        .map(|(input, output)| (input.to_string(), output.to_string()))
        .collect();
    cases.extend(block_header_cases());
    cases
}

/// Python: `test_instructions_keep_multi_line_frontmatter_descriptions` (lazy mode).
#[test]
fn lazy_metadata_keeps_multi_line_descriptions() {
    for (description, expected_description) in multiline_description_cases() {
        let markdown = format!("---\n{description}\nname: discovered-skill\n---\n# Skill\n");
        let (_temp, root) = skill_source(markdown.as_bytes());

        assert_eq!(
            lazy_metadata(&root),
            expected("discovered-skill", &expected_description),
            "{description:?}"
        );
    }
}

/// Python: `test_instructions_keep_multi_line_frontmatter_descriptions` (runtime mode): the
/// same skill in the workspace is listed with the same description.
#[tokio::test]
async fn runtime_metadata_keeps_multi_line_descriptions() {
    for (description, expected_description) in multiline_description_cases() {
        let markdown = format!("---\n{description}\nname: discovered-skill\n---\n# Skill\n");
        let temp = tempfile::tempdir().unwrap();
        let workspace = std::fs::canonicalize(temp.path())
            .unwrap()
            .join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let skills = Skills::from_entry(Entry::dir().with_child(
            "dynamic-skill",
            Entry::dir().with_child("SKILL.md", Entry::file(markdown.into_bytes())),
        ));
        let manifest = skills
            .try_process_manifest(Manifest::new().with_root(workspace.to_string_lossy()))
            .unwrap();
        let client = UnixLocalSandboxClient::new();
        let session = client.create(Some(manifest)).await.unwrap();
        session.start().await.unwrap();

        let text = instructions(&skills, &CapabilityBinding::new(session.clone()))
            .await
            .expect("instructions");

        assert!(
            text.contains(&format!(
                "- discovered-skill: {expected_description} (file: .agents/dynamic-skill)"
            )),
            "{description:?}: {text}"
        );
        client.delete(session).await.ok();
    }
}

/// Python: `test_instructions_treat_comment_lines_the_way_yaml_does` (3 sets).
#[test]
fn comment_lines_are_treated_the_way_yaml_does() {
    let cases = [
        (
            "name: discovered-skill\n  # explanation\ndescription: local dir metadata",
            "local dir metadata",
        ),
        (
            "name: discovered-skill\ndescription: \"local dir metadata\"\n  # note",
            "local dir metadata",
        ),
        (
            "name: discovered-skill\ndescription: >\n  Use for triage.\n  # kept as content",
            "Use for triage. # kept as content",
        ),
    ];
    for (frontmatter, description) in cases {
        let (_temp, root) = skill_source(format!("---\n{frontmatter}\n---\n# Skill\n").as_bytes());
        assert_eq!(
            lazy_metadata(&root),
            expected("discovered-skill", description),
            "{frontmatter:?}"
        );
    }
}

/// Python: `test_instructions_keep_skill_name_when_a_description_line_looks_like_a_key`.
#[test]
fn a_description_line_that_looks_like_a_key_does_not_rename_the_skill() {
    let (_temp, root) = skill_source(
        b"---\nname: discovered-skill\ndescription: >\n  Use for GitHub issue triage.\n  name: not-the-skill-name\n---\n# Skill\n",
    );

    assert_eq!(
        lazy_metadata(&root),
        expected(
            "discovered-skill",
            "Use for GitHub issue triage. name: not-the-skill-name"
        )
    );
}

/// Python: `test_local_metadata_preserves_scalar_boundaries` (19 cases x 2 newline styles).
#[test]
fn local_metadata_preserves_scalar_boundaries() {
    let cases: [(&str, &str, &str); 19] = [
        (
            "name: \"discovered-skill\"\n\n  # explanation\ndescription: triage",
            "discovered-skill",
            "triage",
        ),
        (
            "name: 'discovered-skill'\n\n  # explanation\ndescription: triage",
            "discovered-skill",
            "triage",
        ),
        (
            "name: discovered-skill\ndescription: \"  keep spaces  \"",
            "discovered-skill",
            "  keep spaces  ",
        ),
        (
            "name: discovered-skill\ndescription: use the #triage tag",
            "discovered-skill",
            "use the #triage tag",
        ),
        (
            "name: discovered-skill\ndescription: \"literal \\n escape\"",
            "discovered-skill",
            "literal \\n escape",
        ),
        (
            "name: discovered-skill\ndescription: false",
            "discovered-skill",
            "false",
        ),
        (
            "name: discovered-skill\ndescription: 123",
            "discovered-skill",
            "123",
        ),
        (
            "name: discovered-skill\ndescription:",
            "discovered-skill",
            "",
        ),
        (
            "description: >\nname: discovered-skill",
            "discovered-skill",
            "",
        ),
        (
            "description: |+\n\n  \nname: discovered-skill",
            "discovered-skill",
            "",
        ),
        (
            "metadata:\n  name: nested\n  description: nested\nname: discovered-skill\ndescription: triage",
            "discovered-skill",
            "triage",
        ),
        (
            "name: discovered-skill\ndescription: triage\nmetadata:\n  name: nested\n  description: nested",
            "discovered-skill",
            "triage",
        ),
        (
            "metadata:\n  name: nested\n  description: nested",
            "dynamic-skill",
            "No description provided.",
        ),
        (
            "metadata:\n  - name: nested\n    description: nested\nname: discovered-skill\ndescription: triage",
            "discovered-skill",
            "triage",
        ),
        (
            "  name: discovered-skill\n  description: >2\n    first\n    last",
            "discovered-skill",
            "first last",
        ),
        (
            "name: discovered-skill\ndescription: |\n  \"quoted content\"",
            "discovered-skill",
            "\"quoted content\"",
        ),
        (
            "name: discovered-skill\ndescription: |\n  \\n is literal",
            "discovered-skill",
            "\\n is literal",
        ),
        (
            "name: discovered-skill\ndescription: >\n  first\n# trailing comment\nmetadata:\n  name: nested",
            "discovered-skill",
            "first",
        ),
        (
            "name: discovered-skill\nmetadata: &ignored\n  name: nested\ndescription: *ignored",
            "discovered-skill",
            "*ignored",
        ),
    ];
    for newline in ["\n", "\r\n"] {
        for (frontmatter, name, description) in cases {
            let markdown = format!("---\n{frontmatter}\n---\n# Body\nname: body-only\n")
                .replace('\n', newline);
            let (_temp, root) = skill_source(markdown.as_bytes());

            assert_eq!(
                lazy_metadata(&root),
                expected(name, description),
                "{frontmatter:?} with {newline:?}"
            );
        }
    }
}

/// Python: `test_local_metadata_keeps_missing_frontmatter_fallbacks` (3 sets).
#[test]
fn missing_frontmatter_falls_back_to_the_directory_name() {
    for markdown in [
        "# No frontmatter\n",
        "---\nname: unfinished\n",
        "\n---\nname: not-frontmatter\n---",
    ] {
        let (_temp, root) = skill_source(markdown.as_bytes());
        assert_eq!(
            lazy_metadata(&root),
            expected("dynamic-skill", "No description provided."),
            "{markdown:?}"
        );
        assert!(parse_frontmatter(markdown).is_empty(), "{markdown:?}");
    }
}

/// Python: `test_lazy_local_dir_metadata_skips_symlinked_skill_directory`.
#[cfg(unix)]
#[test]
fn a_symlinked_skill_directory_is_not_listed() {
    let temp = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(temp.path()).unwrap();
    let root = base.join("skills");
    let outside = base.join("outside/linked-skill");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(
        outside.join("SKILL.md"),
        "---\nname: linked-skill\ndescription: linked metadata\n---\n# Skill\n",
    )
    .unwrap();
    std::os::unix::fs::symlink(&outside, root.join("linked-skill")).unwrap();

    assert_eq!(lazy_metadata(&root), Vec::new());
}

// ----------------------------------------------------------------------------------------------
// load_skill
// ----------------------------------------------------------------------------------------------

async fn local_session(
    root: &std::path::Path,
    source: &std::path::Path,
) -> (UnixLocalSandboxClient, Arc<dyn SandboxSession>) {
    let manifest = Manifest::new()
        .with_root(root.to_string_lossy())
        .with_extra_path_grant(grant(source));
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();
    (client, session)
}

fn load_skill_tool(
    skills: &Skills,
    binding: &CapabilityBinding,
) -> Arc<openai_agents_sdk::tool::FunctionTool<()>> {
    Capability::<()>::tools(skills, binding)
        .unwrap()
        .into_iter()
        .find_map(|tool| match tool {
            Tool::Function(function) if function.name == "load_skill" => Some(function),
            _ => None,
        })
        .expect("a load_skill tool")
}

async fn call(tool: &openai_agents_sdk::tool::FunctionTool<()>, arguments: Value) -> Value {
    let arguments = arguments.to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "load_skill", &arguments);
    match tool.invoke(ctx, arguments).await.expect("load_skill") {
        ToolOutput::Text(text) => serde_json::from_str(&text).expect("a JSON record"),
        other => panic!("expected text, got {other:?}"),
    }
}

/// Python: `test_lazy_local_dir_load_skill_tool_materializes_single_skill`.
#[tokio::test]
async fn load_skill_materializes_a_single_skill() {
    let (_temp, source) = skill_source(b"# dynamic skill\n");
    let workspace = source.parent().unwrap().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&source)));
    let (client, session) = local_session(&workspace, &source).await;
    let tool = load_skill_tool(&skills, &CapabilityBinding::new(session.clone()));
    assert!(
        session
            .read_file(".agents/dynamic-skill/SKILL.md")
            .await
            .is_err()
    );

    let output = call(&tool, json!({"skill_name": "dynamic-skill"})).await;

    assert_eq!(
        output,
        json!({"status": "loaded", "skill_name": "dynamic-skill", "path": ".agents/dynamic-skill"})
    );
    assert_eq!(
        std::fs::read_to_string(workspace.join(".agents/dynamic-skill/SKILL.md")).unwrap(),
        "# dynamic skill\n"
    );

    // Python: `test_load_skill_returns_already_loaded_for_existing_materialized_skill`.
    let again = call(&tool, json!({"skill_name": "dynamic-skill"})).await;
    assert_eq!(again["status"], "already_loaded");
    client.delete(session).await.ok();
}

/// Python: `test_lazy_load_reports_absolute_path_without_relocating_skill`.
#[tokio::test]
async fn load_skill_reports_an_absolute_path_under_a_run_cwd() {
    let (_temp, source) = skill_source(b"# dynamic skill\n");
    let workspace = source.parent().unwrap().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&source)));
    let (client, session) = local_session(&workspace, &source).await;
    let scope = SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).unwrap();

    let loaded = skills
        .load_skill(session.as_ref(), &scope, None, "dynamic-skill")
        .await
        .unwrap();

    assert_eq!(loaded.status, "loaded");
    assert_eq!(
        loaded.path,
        format!("{}/.agents/dynamic-skill", workspace.display())
    );
    assert!(workspace.join(".agents/dynamic-skill/SKILL.md").is_file());
    assert!(!workspace.join("tasks/task-a/.agents").exists());
    client.delete(session).await.ok();
}

/// Python: `test_load_skill_rejects_non_lazy_capability`, `test_tools_returns_empty_without_lazy_source`.
#[tokio::test]
async fn load_skill_needs_a_lazy_source() {
    let skills = Skills::from_skills(vec![skill("my-skill", "desc", "literal")]);
    let binding = binding_for(Manifest::new().with_root("/workspace"));
    assert!(
        Capability::<()>::tools(&skills, &binding)
            .unwrap()
            .is_empty()
    );

    let error = skills
        .load_skill(
            binding.session.as_ref(),
            &SandboxWorkspaceScope::root(),
            None,
            "my-skill",
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.message,
        "load_skill is only available when lazy_from is configured"
    );
}

/// Python: `test_load_skill_rejects_missing_lazy_source_directory` and
/// `test_load_skill_rejects_ambiguous_skill_name`.
#[tokio::test]
async fn load_skill_reports_unavailable_missing_and_ambiguous_skills() {
    let (_temp, source) = skill_source(b"---\nname: shared\n---\n");
    // A second directory whose front matter claims the same name.
    std::fs::create_dir_all(source.join("other")).unwrap();
    std::fs::write(source.join("other/SKILL.md"), "---\nname: shared\n---\n").unwrap();
    let workspace = source.parent().unwrap().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let (client, session) = local_session(&workspace, &source).await;
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&source)));

    let ambiguous = skills
        .load_skill(
            session.as_ref(),
            &SandboxWorkspaceScope::root(),
            None,
            "shared",
        )
        .await
        .unwrap_err();
    assert_eq!(ambiguous.message, "lazy skill name is ambiguous");

    let missing = skills
        .load_skill(
            session.as_ref(),
            &SandboxWorkspaceScope::root(),
            None,
            "nope",
        )
        .await
        .unwrap_err();
    assert_eq!(missing.message, "lazy skill not found");

    let unavailable = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(
        source.join("does-not-exist"),
    )))
    .load_skill(
        session.as_ref(),
        &SandboxWorkspaceScope::root(),
        None,
        "shared",
    )
    .await
    .unwrap_err();
    assert_eq!(
        unavailable.message,
        "lazy skill source directory is unavailable"
    );
    client.delete(session).await.ok();
}

/// Python: `test_lazy_local_dir_load_skill_applies_source_metadata`,
/// `..._keeps_default_permissions`, `test_load_skill_materializes_with_bound_run_as_user` and the
/// `_StaticResultLazySkillSource` rows describe Python's `LocalDir` metadata and an arbitrary
/// `dict` result; Rust's lazy source returns a typed [`LoadedSkill`], and its path is validated
/// by `model_skill_path`, whose scoped failure is checked here.
#[test]
fn a_scoped_skill_path_must_be_workspace_relative() {
    let scope = SandboxWorkspaceScope::from_cwd(Some("tasks/task-a")).unwrap();
    for bad in ["../escape", ".agents\\dynamic-skill", ""] {
        let result = scope.model_resource_path("/workspace", bad);
        assert!(result.is_err(), "{bad:?}");
    }
    assert_eq!(
        scope
            .model_resource_path("/workspace", ".agents/dynamic-skill")
            .unwrap(),
        "/workspace/.agents/dynamic-skill"
    );
}

// ----------------------------------------------------------------------------------------------
// test_run_cwd.py: a shared skill keeps task files in the run working directory
// ----------------------------------------------------------------------------------------------

/// Python: `test_python_skill_uses_absolute_root_from_nested_workdir`. The skill's script reads
/// its own assets through the absolute skill root and the task's files through the working
/// directory.
#[cfg(unix)]
#[tokio::test]
async fn a_shared_skill_script_keeps_task_files_local() {
    use openai_agents_sdk::items::OutputItem;
    use openai_agents_sdk::sandbox::capabilities::{Filesystem, Shell};
    use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};
    use openai_agents_sdk::testing::{FakeModel, assistant_message};

    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let script = "root=\"$(cd \"$(dirname \"$0\")/..\" && pwd)\"\nsuffix=\"$(cat \"$root/assets/suffix.txt\")\"\nprintf '%s%s' \"$(cat input.txt)\" \"$suffix\" > output.txt\n";
    let skills = Skills::from_skills(vec![
        skill(
            "shell-proof",
            "Proves shared skills keep task files local.",
            "# Proof\n",
        )
        .with_script("prove.sh", Entry::file(script.as_bytes().to_vec()))
        .with_asset("suffix.txt", Entry::file(b"-from-shared-skill".to_vec())),
    ]);
    let manifest = skills.try_process_manifest(Manifest::new()).unwrap();
    let client = UnixLocalSandboxClient::new();
    let session = client.create(Some(manifest)).await.unwrap();
    session.start().await.unwrap();
    session.mkdir("tasks/task-a/nested", true).await.unwrap();
    session
        .write_file("tasks/task-a/nested/input.txt", b"task-a")
        .await
        .unwrap();
    let root = session.workspace_root().to_string();
    let script_path = format!("{root}/.agents/shell-proof/scripts/prove.sh");
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![OutputItem::from_value(json!({
        "type": "function_call", "call_id": "python_skill", "name": "exec_command",
        "arguments": json!({
            "cmd": format!("sh '{script_path}'"), "workdir": "nested", "login": false
        }).to_string()
    }))]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = SandboxAgent::<()>::new("skill-task")
        .with_model_instance(model.clone())
        .with_capabilities(vec![
            Arc::new(Shell::new()),
            Arc::new(skills),
            Arc::new(Filesystem::new()),
        ]);

    let result = SandboxRunner::run(
        &agent,
        "Use the available skill.",
        &SandboxRunConfig::with_session(session.clone()).cwd("tasks/task-a"),
    )
    .await
    .unwrap();

    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert_eq!(
        session
            .read_file("tasks/task-a/nested/output.txt")
            .await
            .unwrap(),
        b"task-a-from-shared-skill"
    );
    assert_eq!(
        session
            .read_file(".agents/shell-proof/scripts/prove.sh")
            .await
            .unwrap(),
        script.as_bytes()
    );
    let prompt = model.calls()[0].system_instructions.clone().unwrap();
    assert!(
        prompt.contains(&format!("(file: {root}/.agents/shell-proof)")),
        "{prompt}"
    );
    assert!(prompt.contains("Treat each listed path as the skill root"));
    assert!(prompt.contains("Files outside the working directory may be visible to or shared"));
    client.delete(session).await.ok();
}

// ----------------------------------------------------------------------------------------------
// Entry metadata and run-as users on load
// ----------------------------------------------------------------------------------------------

use std::sync::Mutex;

use async_trait::async_trait;
use openai_agents_sdk::sandbox::errors::SandboxResult;
use openai_agents_sdk::sandbox::session::ExecRequest;
use openai_agents_sdk::sandbox::types::{ExecResult, Group, Permissions, User, file_mode};

/// A session over a real local workspace that records the commands and the users it was asked
/// to act as. Python: `_ExecRecordingSkillsSession` and `_SkillsSession`.
#[derive(Debug)]
struct RecordingSession {
    inner: Arc<dyn SandboxSession>,
    commands: Mutex<Vec<Vec<String>>>,
    read_users: Mutex<Vec<Option<String>>>,
    write_users: Mutex<Vec<Option<String>>>,
    mkdir_users: Mutex<Vec<Option<String>>>,
}

impl RecordingSession {
    fn new(inner: Arc<dyn SandboxSession>) -> Self {
        RecordingSession {
            inner,
            commands: Mutex::default(),
            read_users: Mutex::default(),
            write_users: Mutex::default(),
            mkdir_users: Mutex::default(),
        }
    }
}

fn name_of(user: Option<&User>) -> Option<String> {
    user.map(|user| user.name.clone())
}

#[async_trait]
impl SandboxSession for RecordingSession {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn manifest(&self) -> &Manifest {
        self.inner.manifest()
    }
    fn backend_id(&self) -> &str {
        "recording"
    }
    async fn start(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn stop(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn shutdown(&self) -> SandboxResult<()> {
        Ok(())
    }
    async fn running(&self) -> SandboxResult<bool> {
        Ok(true)
    }
    async fn exec(&self, request: ExecRequest) -> SandboxResult<ExecResult> {
        self.commands.lock().unwrap().push(request.command.clone());
        self.inner
            .exec(ExecRequest {
                user: None,
                ..request
            })
            .await
    }
    async fn read_file(&self, path: &str) -> SandboxResult<Vec<u8>> {
        self.inner.read_file(path).await
    }
    async fn write_file(&self, path: &str, contents: &[u8]) -> SandboxResult<()> {
        self.inner.write_file(path, contents).await
    }
    async fn read_file_as(&self, path: &str, user: Option<&User>) -> SandboxResult<Vec<u8>> {
        self.read_users.lock().unwrap().push(name_of(user));
        self.inner.read_file(path).await
    }
    async fn write_file_as(
        &self,
        path: &str,
        contents: &[u8],
        user: Option<&User>,
    ) -> SandboxResult<()> {
        self.write_users.lock().unwrap().push(name_of(user));
        self.inner.write_file(path, contents).await
    }
    async fn mkdir_as(&self, path: &str, parents: bool, user: Option<&User>) -> SandboxResult<()> {
        self.mkdir_users.lock().unwrap().push(name_of(user));
        self.inner.mkdir(path, parents).await
    }
    async fn hydrate_workspace(&self, archive: &[u8]) -> SandboxResult<()> {
        self.inner.hydrate_workspace(archive).await
    }
}

async fn recording(
    workspace: &std::path::Path,
    source: &std::path::Path,
) -> (UnixLocalSandboxClient, Arc<RecordingSession>) {
    let (client, session) = local_session(workspace, source).await;
    (client, Arc::new(RecordingSession::new(session)))
}

fn has_command(session: &RecordingSession, wanted: &[&str]) -> bool {
    session.commands.lock().unwrap().iter().any(|command| {
        command
            .iter()
            .map(String::as_str)
            .eq(wanted.iter().copied())
    })
}

/// Python: `test_lazy_local_dir_load_skill_applies_source_metadata`.
#[tokio::test]
async fn a_loaded_skill_keeps_the_source_entrys_permissions_and_group() {
    let (_temp, source) = skill_source(b"# dynamic skill\n");
    let workspace = source.parent().unwrap().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let mut entry = Entry::local_dir(&source);
    entry.permissions = Permissions::new(file_mode::ALL, 0, 0).as_directory(true);
    entry.group = Some(Group::new("staff"));
    let lazy = LocalDirLazySkillSource::from_entry(entry).unwrap();
    let skills = Skills::lazy_from(Arc::new(lazy.clone()));
    let (client, session) = recording(&workspace, &source).await;

    skills
        .load_skill(
            session.as_ref(),
            &SandboxWorkspaceScope::root(),
            None,
            "dynamic-skill",
        )
        .await
        .unwrap();

    let dest = format!("{}/.agents/dynamic-skill", workspace.display());
    assert!(
        has_command(&session, &["chmod", "0700", &dest]),
        "{:?}",
        session.commands
    );
    assert!(has_command(&session, &["chgrp", "staff", &dest]));
    // The configured source entry is not repointed at the loaded skill.
    assert!(matches!(
        &lazy.entry().kind,
        EntryKind::LocalDir { src: Some(src) } if *src == source
    ));
    client.delete(session.inner.clone()).await.ok();
}

/// Python: `test_lazy_local_dir_load_skill_keeps_default_permissions`.
#[tokio::test]
async fn a_loaded_skill_keeps_default_permissions() {
    let (_temp, source) = skill_source(b"# dynamic skill\n");
    let workspace = source.parent().unwrap().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&source)));
    let (client, session) = recording(&workspace, &source).await;

    skills
        .load_skill(
            session.as_ref(),
            &SandboxWorkspaceScope::root(),
            None,
            "dynamic-skill",
        )
        .await
        .unwrap();

    let dest = format!("{}/.agents/dynamic-skill", workspace.display());
    assert!(has_command(&session, &["chmod", "0755", &dest]));
    assert!(
        !session
            .commands
            .lock()
            .unwrap()
            .iter()
            .any(|c| c[0] == "chgrp")
    );
    client.delete(session.inner.clone()).await.ok();
}

/// Python: `test_load_skill_materializes_with_bound_run_as_user`.
#[tokio::test]
async fn load_skill_materializes_as_the_bound_user() {
    let (_temp, source) = skill_source(b"# dynamic skill\n");
    let workspace = source.parent().unwrap().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let skills = Skills::lazy_from(Arc::new(LocalDirLazySkillSource::new(&source)));
    let (client, session) = recording(&workspace, &source).await;
    let user = User::new("sandbox-user");

    let loaded = skills
        .load_skill(
            session.as_ref(),
            &SandboxWorkspaceScope::root(),
            Some(&user),
            "dynamic-skill",
        )
        .await
        .unwrap();

    assert_eq!(loaded.status, "loaded");
    let named = Some("sandbox-user".to_string());
    assert_eq!(*session.read_users.lock().unwrap(), vec![named.clone()]);
    assert_eq!(*session.write_users.lock().unwrap(), vec![named.clone()]);
    {
        let mkdirs = session.mkdir_users.lock().unwrap();
        assert!(!mkdirs.is_empty());
        assert!(mkdirs.iter().all(|user| *user == named));
    }
    client.delete(session.inner.clone()).await.ok();
}
