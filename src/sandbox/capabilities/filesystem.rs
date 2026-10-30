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
use crate::sandbox::capabilities::{Capability, CapabilityBinding};
use crate::sandbox::editor::SandboxEditor;
use crate::sandbox::util::{
    MAX_IMAGE_BYTES, MAX_IMAGE_SIZE_LABEL, base64_encode, detect_image_media_type,
};
use crate::tool::{ApplyPatchTool, FunctionTool, Tool, ToolOutput, ToolOutputPart};

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

/// Exposes `apply_patch` and `view_image`, both bound to the live sandbox session.
#[derive(Debug, Clone)]
pub struct Filesystem {
    apply_patch_needs_approval: bool,
    include_view_image: bool,
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
        }
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
        let apply_patch =
            ApplyPatchTool::<C>::new(editor).with_needs_approval(self.apply_patch_needs_approval);
        let mut tools = vec![Tool::ApplyPatch(Arc::new(apply_patch))];
        if self.include_view_image {
            let session = binding.session.clone();
            let scope = binding.scope.clone();
            let run_as = binding.run_as.clone();
            let view_image = FunctionTool::<C>::new(
                VIEW_IMAGE_TOOL_NAME,
                VIEW_IMAGE_DESCRIPTION,
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
                                    "Invalid path. Use a path within the sandbox workspace."
                                        .to_string(),
                                ));
                            }
                        };
                        // Check the size before reading: a multi-gigabyte file would be held
                        // in memory, and on a hosted backend pulled over the wire, only to be
                        // refused afterwards.
                        if let Ok(entries) = session.ls(&resolved).await
                            && let Some(size) = entries.first().and_then(|entry| entry.size)
                            && size > MAX_IMAGE_BYTES as u64
                        {
                            return Ok(ToolOutput::Text(format!(
                                "image path `{}` exceeded the allowed size of {}; resize or \
                                 compress the image and try again",
                                args.path, MAX_IMAGE_SIZE_LABEL
                            )));
                        }
                        let bytes = match session.read_file_as(&resolved, run_as.as_ref()).await {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                return Ok(ToolOutput::Text(format!(
                                    "Could not read `{}`: {}",
                                    args.path, error.message
                                )));
                            }
                        };
                        // An image is inlined into the conversation, so an unbounded one would
                        // spend the whole context on a single tool call.
                        if bytes.len() > MAX_IMAGE_BYTES {
                            return Ok(ToolOutput::Text(format!(
                                "image path `{}` exceeded the allowed size of {}; resize or \
                                 compress the image and try again",
                                args.path, MAX_IMAGE_SIZE_LABEL
                            )));
                        }
                        // The file's own bytes decide the media type: a name is not evidence,
                        // and labelling a text file `image/png` only confuses the model.
                        let Some(media_type) = detect_image_media_type(&resolved, &bytes) else {
                            return Ok(ToolOutput::Text(format!(
                                "image path `{}` is not a supported image file",
                                args.path
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
        Ok(tools)
    }
}
