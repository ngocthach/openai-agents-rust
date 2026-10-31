//! The filesystem capability. Mirrors `agents.sandbox.capabilities.filesystem`.
//!
//! Like the Python capability, this exposes exactly two tools: `apply_patch` for editing files
//! and `view_image` for loading an image out of the workspace. Reading, writing and listing are
//! done with the shell capability or the session API, not with dedicated tools.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::error::Result;
use crate::sandbox::capabilities::tool_input_errors::{FieldKind, tool_with_safe_input};
use crate::sandbox::capabilities::{Capability, CapabilityBinding, SandboxApplyPatchTool};
use crate::sandbox::editor::SandboxEditor;
use crate::sandbox::errors::ErrorCode;
use crate::sandbox::util::{
    MAX_IMAGE_BYTES, MAX_IMAGE_SIZE_LABEL, base64_encode, detect_image_media_type,
};
use crate::tool::{ApplyPatchTool, Tool, ToolOutput, ToolOutputPart};

/// The name of the image inspection tool.
pub const VIEW_IMAGE_TOOL_NAME: &str = "view_image";

const VIEW_IMAGE_DESCRIPTION: &str = "Loads an image from the sandbox workspace and returns it as \
                                      a structured image output.";

/// Arguments of the `view_image` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ViewImageArgs {
    /// Path to the image file, relative to the run working directory or absolute inside the
    /// sandbox workspace.
    pub path: String,
}

/// Exposes `view_image` and `apply_patch`, both bound to the live sandbox session.
#[derive(Debug, Clone)]
pub struct Filesystem {
    apply_patch_needs_approval: bool,
    include_view_image: bool,
    custom_apply_patch: bool,
}

impl Default for Filesystem {
    fn default() -> Self {
        Filesystem::new()
    }
}

impl Filesystem {
    /// The filesystem capability with both tools enabled.
    pub fn new() -> Self {
        Filesystem {
            apply_patch_needs_approval: false,
            include_view_image: true,
            custom_apply_patch: true,
        }
    }

    /// Choose the `apply_patch` flavour (default: the freeform custom tool, as in Python).
    ///
    /// The custom tool takes one raw `*** Begin Patch` string that can combine several file
    /// operations, validates every path before applying the first, and checks approval per
    /// operation. `false` selects the native `apply_patch_call` tool instead, which Python's
    /// capability does not offer; it stays available for a model or a recorded conversation
    /// built around native `apply_patch_call` items.
    pub fn with_custom_apply_patch(mut self, custom: bool) -> Self {
        self.custom_apply_patch = custom;
        self
    }

    /// Expose the native `apply_patch_call` tool instead of the freeform custom tool.
    pub fn with_native_apply_patch(self) -> Self {
        self.with_custom_apply_patch(false)
    }

    /// Require human approval for every `apply_patch` call.
    pub fn with_apply_patch_needs_approval(mut self, needs_approval: bool) -> Self {
        self.apply_patch_needs_approval = needs_approval;
        self
    }

    /// Include the `view_image` tool (default: true).
    pub fn with_view_image(mut self, include: bool) -> Self {
        self.include_view_image = include;
        self
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> Capability<C> for Filesystem {
    fn capability_type(&self) -> &str {
        "filesystem"
    }

    fn tools(&self, binding: &CapabilityBinding) -> Result<Vec<Tool<C>>> {
        let editor = SandboxEditor::new(binding.session.clone())
            .with_scope(binding.scope.clone())
            .with_run_as(binding.run_as.clone());
        let apply_patch_tool: Tool<C> = if self.custom_apply_patch {
            let custom = SandboxApplyPatchTool::<C>::new(editor)
                .with_needs_approval(self.apply_patch_needs_approval)
                .into_custom_tool();
            Tool::Custom(Arc::new(custom))
        } else {
            let apply_patch = ApplyPatchTool::<C>::new(editor)
                .with_needs_approval(self.apply_patch_needs_approval);
            Tool::ApplyPatch(Arc::new(apply_patch))
        };
        // Python lists `view_image` first, then `apply_patch`.
        let mut tools: Vec<Tool<C>> = Vec::new();
        if self.include_view_image {
            let session = binding.session.clone();
            let scope = binding.scope.clone();
            let run_as = binding.run_as.clone();
            let view_image = tool_with_safe_input::<C, ViewImageArgs, _, _, _>(
                VIEW_IMAGE_TOOL_NAME,
                VIEW_IMAGE_DESCRIPTION,
                &[("path", FieldKind::Str { non_empty: true })],
                move |_ctx, args: ViewImageArgs| {
                    let session = session.clone();
                    let scope = scope.clone();
                    let run_as = run_as.clone();
                    async move {
                        session.ensure_mounts_settled()?;
                        let anchored = scope.anchor(&args.path);
                        let resolved = match session.normalize_path(&anchored) {
                            Ok(resolved) => resolved,
                            Err(_) => {
                                return Ok(ToolOutput::Text(
                                    "Invalid path. Use a path within the workspace or an \
                                     explicitly granted path."
                                        .to_string(),
                                ));
                            }
                        };
                        // The model sees the path it asked about, in the shape it used: a
                        // provider-private root or the run's cwd prefix never reaches it.
                        let display = session
                            .workspace_path_policy()
                            .and_then(|policy| policy.relative_path(&anchored))
                            .ok()
                            .and_then(|relative| scope.display_path(&anchored, &relative).ok())
                            .unwrap_or_else(|| resolved.clone());
                        // Read one byte past the limit: that is enough to know the file is too
                        // big without holding or fetching all of it. A run-as read goes through
                        // `exec`, which has no bounded form, so it reads whole.
                        let read = match run_as.as_ref() {
                            None => session.read_bounded(&resolved, MAX_IMAGE_BYTES + 1).await,
                            Some(user) => session.read_file_as(&resolved, Some(user)).await,
                        };
                        let bytes = match read {
                            Ok(bytes) => bytes,
                            Err(error) if error.code == ErrorCode::WorkspaceReadNotFound => {
                                return Ok(ToolOutput::Text(format!(
                                    "image path `{display}` was not found"
                                )));
                            }
                            Err(error) => {
                                return Ok(ToolOutput::Text(format!(
                                    "unable to read image at `{display}`: {}",
                                    error.code
                                )));
                            }
                        };
                        // An image is inlined into the conversation, so an unbounded one would
                        // spend the whole context on a single tool call.
                        if bytes.len() > MAX_IMAGE_BYTES {
                            return Ok(ToolOutput::Text(format!(
                                "image path `{display}` exceeded the allowed size of {}; resize \
                                 or compress the image and try again",
                                MAX_IMAGE_SIZE_LABEL
                            )));
                        }
                        // The file's own bytes decide the media type: a name is not evidence,
                        // and labelling a text file `image/png` only confuses the model.
                        let Some(media_type) = detect_image_media_type(&resolved, &bytes) else {
                            return Ok(ToolOutput::Text(format!(
                                "image path `{display}` is not a supported image file"
                            )));
                        };
                        let data_url =
                            format!("data:{media_type};base64,{}", base64_encode(&bytes));
                        Ok::<_, crate::error::AgentsError>(ToolOutput::Parts(vec![
                            ToolOutputPart::image_url(data_url),
                        ]))
                    }
                },
            )
            .with_strict_json_schema(false);
            tools.push(Tool::Function(Arc::new(view_image)));
        }
        tools.push(apply_patch_tool);
        Ok(tools)
    }
}
