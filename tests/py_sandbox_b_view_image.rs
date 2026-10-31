//! Ports of `tests/sandbox/capabilities/test_view_image_tool.py` and
//! `tests/sandbox/test_view_image_content_validation.py`.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use base64::Engine as _;
use openai_agents_sdk::RunContext;
use openai_agents_sdk::sandbox::capabilities::{Capability, CapabilityBinding, Filesystem};
use openai_agents_sdk::sandbox::errors::{ErrorCode, SandboxError};
use openai_agents_sdk::sandbox::manifest::Manifest;
use openai_agents_sdk::sandbox::testing::{
    SandboxMethod, SandboxStep, ScriptedSandboxSession, scripted_sandbox_session,
};
use openai_agents_sdk::sandbox::types::User;
use openai_agents_sdk::sandbox::workspace_paths::{SandboxPathGrant, SandboxWorkspaceScope};
use openai_agents_sdk::tool::{FunctionTool, Tool, ToolOutput, ToolOutputPart};
use openai_agents_sdk::tool_context::ToolContext;
use serde_json::json;

const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a84QAAAAASUVORK5CYII=";

fn png() -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(PNG_BASE64)
        .unwrap()
}

fn read_step(bytes: impl Into<Vec<u8>>) -> SandboxStep {
    SandboxStep::new(SandboxMethod::ReadFile).returning_bytes(bytes)
}

fn session(steps: Vec<SandboxStep>) -> Arc<ScriptedSandboxSession> {
    Arc::new(scripted_sandbox_session(steps).with_manifest(Manifest::new().with_root("/workspace")))
}

fn tool_for(
    session: &Arc<ScriptedSandboxSession>,
    scope: Option<&str>,
    run_as: Option<User>,
) -> Arc<FunctionTool<()>> {
    let mut binding = CapabilityBinding::new(session.clone()).with_run_as(run_as);
    if let Some(cwd) = scope {
        binding = binding.with_scope(SandboxWorkspaceScope::from_cwd(Some(cwd)).unwrap());
    }
    let tools = Capability::<()>::tools(&Filesystem::new(), &binding).unwrap();
    tools
        .into_iter()
        .find_map(|tool| match tool {
            Tool::Function(function) if function.name == "view_image" => Some(function),
            _ => None,
        })
        .expect("view_image is exposed")
}

async fn view(tool: &FunctionTool<()>, path: &str) -> ToolOutput {
    let arguments = json!({ "path": path }).to_string();
    let ctx = ToolContext::direct(RunContext::new(()), "agent", "view_image", &arguments);
    tool.invoke(ctx, arguments).await.expect("view_image")
}

fn text(output: ToolOutput) -> String {
    match output {
        ToolOutput::Text(text) => text,
        other => panic!("expected text, got {other:?}"),
    }
}

fn image_url(output: ToolOutput) -> String {
    match output {
        ToolOutput::Parts(parts) => match parts.into_iter().next() {
            Some(ToolOutputPart::Image { image_url, .. }) => image_url.expect("an image url"),
            other => panic!("expected an image part, got {other:?}"),
        },
        other => panic!("expected an image, got {other:?}"),
    }
}

fn path_of(call: &openai_agents_sdk::sandbox::testing::SandboxCall) -> String {
    call.arguments[0].clone()
}

// ----------------------------------------------------------------------------------------------
// test_view_image_tool.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_view_image_accepts_needs_approval_setting`. Rust sets approval on the capability
/// for `apply_patch`; `view_image` has none to configure, so the tool is never held for approval.
#[test]
fn view_image_is_not_held_for_approval_by_default() {
    let tool = tool_for(&session(vec![]), None, None);
    assert!(matches!(
        tool.needs_approval,
        openai_agents_sdk::tool::NeedsApproval::Static(false)
    ));
}

/// Python: `test_view_image_returns_tool_output_image_for_png`.
#[tokio::test]
async fn view_image_returns_an_image_output_for_png() {
    let s = session(vec![read_step(png())]);

    let output = view(&tool_for(&s, None, None), "images/dot.png").await;

    assert_eq!(
        image_url(output),
        format!("data:image/png;base64,{PNG_BASE64}")
    );
    s.assert_complete();
}

/// Python: `test_view_image_reads_absolute_extra_path_grant`.
#[tokio::test]
async fn view_image_reads_an_absolute_extra_path_grant() {
    let grant = SandboxPathGrant::new("/shared").unwrap().read_only(true);
    let s = Arc::new(
        scripted_sandbox_session(vec![read_step(png())]).with_manifest(
            Manifest::new()
                .with_root("/workspace")
                .with_extra_path_grant(grant),
        ),
    );

    let output = view(&tool_for(&s, None, None), "/shared/dot.png").await;

    assert!(image_url(output).starts_with("data:image/png;base64,"));
    assert_eq!(path_of(&s.calls()[0]), "/shared/dot.png");
    s.assert_complete();
}

/// Python: `test_view_image_still_rejects_ungranted_absolute_path`. Through the tool the
/// rejection is the model-facing "Invalid path." text, and nothing reaches the backend.
#[tokio::test]
async fn view_image_still_rejects_an_ungranted_absolute_path() {
    let s = session(vec![]);

    let output = view(&tool_for(&s, None, None), "/shared/dot.png").await;

    assert!(text(output).starts_with("Invalid path."));
    assert!(s.calls().is_empty());
}

/// Python: `test_view_image_resolves_relative_path_from_workspace_scope` and
/// `test_view_image_normalizes_raw_backslashes_before_workspace_scope`.
#[tokio::test]
async fn view_image_resolves_a_relative_path_from_the_workspace_scope() {
    for requested in ["images/dot.png", r"images\dot.png"] {
        let s = session(vec![read_step(png())]);

        let output = view(&tool_for(&s, Some("tasks/a"), None), requested).await;

        assert!(image_url(output).starts_with("data:image/png;base64,"));
        assert_eq!(path_of(&s.calls()[0]), "/workspace/tasks/a/images/dot.png");
        s.assert_complete();
    }
}

/// Python: `test_view_image_keeps_absolute_path_behavior_with_workspace_scope` and
/// `test_view_image_reports_raw_posix_absolute_path_after_normalization`.
#[tokio::test]
async fn view_image_keeps_absolute_path_behavior_with_a_workspace_scope() {
    for (requested, display, resolved) in [
        ("/workspace/notes.txt", "notes.txt", "/workspace/notes.txt"),
        (r"\workspace\root.txt", "root.txt", "/workspace/root.txt"),
    ] {
        let s = session(vec![read_step(b"hello\n".to_vec())]);

        let output = view(&tool_for(&s, Some("tasks/a"), None), requested).await;

        assert_eq!(
            text(output),
            format!("image path `{display}` is not a supported image file")
        );
        assert_eq!(path_of(&s.calls()[0]), resolved);
        s.assert_complete();
    }
}

/// Python: `test_view_image_scoped_error_uses_model_relative_path`.
#[tokio::test]
async fn view_image_scoped_error_uses_the_model_relative_path() {
    let s = session(vec![SandboxStep::new(SandboxMethod::ReadFile).failing(
        SandboxError::new(
            ErrorCode::WorkspaceReadNotFound,
            "missing /provider/private/root/tasks/a/images/missing.png",
        ),
    )]);

    let output = text(view(&tool_for(&s, Some("tasks/a"), None), "images/missing.png").await);

    assert_eq!(output, "image path `images/missing.png` was not found");
    assert!(!output.contains("/provider/private/root"));
    assert_eq!(
        path_of(&s.calls()[0]),
        "/workspace/tasks/a/images/missing.png"
    );
    s.assert_complete();
}

/// Python: `test_view_image_reads_as_bound_user`. A run-as read goes through `exec` (`sudo cat`)
/// in Rust rather than a `read(user=...)` call.
#[tokio::test]
async fn view_image_reads_as_the_bound_user() {
    let s = session(vec![
        SandboxStep::new(SandboxMethod::Exec)
            .matching(|call| call.has_argument("sudo") && call.has_argument("sandbox-user"))
            .returning_exec(openai_agents_sdk::sandbox::types::ExecResult {
                exit_code: 0,
                stdout: png(),
                stderr: Vec::new(),
            }),
    ]);

    let output = view(
        &tool_for(&s, None, Some(User::new("sandbox-user"))),
        "images/dot.png",
    )
    .await;

    assert!(image_url(output).starts_with("data:image/png;base64,"));
    s.assert_complete();
}

/// Python: `test_view_image_rejects_non_image_files`.
#[tokio::test]
async fn view_image_rejects_non_image_files() {
    let s = session(vec![read_step(b"hello\n".to_vec())]);

    let output = view(&tool_for(&s, None, None), "notes.txt").await;

    assert_eq!(
        text(output),
        "image path `notes.txt` is not a supported image file"
    );
    s.assert_complete();
}

fn huge_png() -> Vec<u8> {
    let mut payload = b"\x89PNG\r\n\x1a\n".to_vec();
    payload.extend(std::iter::repeat_n(b'0', MAX_IMAGE_BYTES + 1));
    payload
}

/// Python: `test_view_image_rejects_images_larger_than_10mb`.
#[tokio::test]
async fn view_image_rejects_images_larger_than_10mb() {
    let s = session(vec![read_step(huge_png())]);

    let output = view(&tool_for(&s, None, None), "images/huge.png").await;

    assert_eq!(
        text(output),
        "image path `images/huge.png` exceeded the allowed size of 10MB; resize or compress the image and try again"
    );
    s.assert_complete();
}

/// Python: `test_view_image_rejection_text_does_not_expose_provider_path`.
#[tokio::test]
async fn view_image_rejection_text_does_not_expose_the_provider_path() {
    let s = Arc::new(
        scripted_sandbox_session(vec![
            SandboxStep::new(SandboxMethod::ReadFile).failing(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                "missing /provider/private/root/images/missing.png",
            )),
            read_step(b"hello\n".to_vec()),
            read_step(huge_png()),
        ])
        .with_manifest(Manifest::new().with_root("/provider/private/root")),
    );
    let tool = tool_for(&s, None, None);

    let outputs = [
        text(view(&tool, "images/missing.png").await),
        text(view(&tool, "notes.txt").await),
        text(view(&tool, "images/huge.png").await),
    ];

    assert_eq!(
        outputs,
        [
            "image path `images/missing.png` was not found".to_string(),
            "image path `notes.txt` is not a supported image file".to_string(),
            "image path `images/huge.png` exceeded the allowed size of 10MB; resize or compress the image and try again".to_string(),
        ]
    );
    for output in &outputs {
        assert!(!output.contains("/provider/private/root"), "{output}");
    }
}

// ----------------------------------------------------------------------------------------------
// test_view_image_content_validation.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_view_image_rejects_non_image_bytes_with_raster_extension` and
/// `test_view_image_ignores_mutated_mime_mapping_for_raster_extension`. Rust never consults a
/// process-wide MIME table (the second test mutates Python's `mimetypes`), so one check covers
/// both: a raster extension is not evidence.
#[tokio::test]
async fn view_image_rejects_non_image_bytes_with_a_raster_extension() {
    let s = session(vec![read_step(b"not an image\n".to_vec())]);

    let output = view(&tool_for(&s, None, None), "images/fake.png").await;

    assert_eq!(
        text(output),
        "image path `images/fake.png` is not a supported image file"
    );
    s.assert_complete();
}

/// Python: `test_view_image_accepts_raster_signature_without_image_extension`.
#[tokio::test]
async fn view_image_accepts_a_raster_signature_without_an_image_extension() {
    let s = session(vec![read_step(png())]);

    let output = view(&tool_for(&s, None, None), "images/payload.bin").await;

    assert!(image_url(output).starts_with("data:image/png;base64,"));
    s.assert_complete();
}

/// Python: `test_view_image_preserves_svg_filename_compatibility` (4 parameter sets).
#[tokio::test]
async fn view_image_preserves_svg_filename_compatibility() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg"></svg>"#;
    let mut utf16 = vec![0xff, 0xfe];
    utf16.extend(svg.encode_utf16().flat_map(u16::to_le_bytes));
    let payloads: Vec<Vec<u8>> = vec![
        [b"\xef\xbb\xbf".as_slice(), svg.as_bytes()].concat(),
        [b"<!-- generated -->\n".as_slice(), svg.as_bytes()].concat(),
        [
            br#"<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "svg11.dtd">"#.as_slice(),
            b"\n",
            svg.as_bytes(),
        ]
        .concat(),
        utf16,
    ];
    for payload in payloads {
        let s = session(vec![read_step(payload)]);

        let output = view(&tool_for(&s, None, None), "images/vector.svg").await;

        assert!(image_url(output).starts_with("data:image/svg+xml;base64,"));
        s.assert_complete();
    }
}

/// Python: `test_view_image_preserves_svgz_filename_compatibility`.
#[tokio::test]
async fn view_image_preserves_svgz_filename_compatibility() {
    use std::io::Write as _;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder
        .write_all(br#"<svg xmlns="http://www.w3.org/2000/svg"></svg>"#)
        .unwrap();
    let s = session(vec![read_step(encoder.finish().unwrap())]);

    let output = view(&tool_for(&s, None, None), "images/vector.svgz").await;

    assert!(image_url(output).starts_with("data:image/svg+xml;base64,"));
    s.assert_complete();
}
