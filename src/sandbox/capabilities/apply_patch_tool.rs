//! The freeform `apply_patch` custom tool. Mirrors
//! `agents.sandbox.capabilities.tools.apply_patch_tool`.
//!
//! The model sends one raw patch string (`*** Begin Patch` ... `*** End Patch`) constrained by a
//! Lark grammar, which can combine several file operations in one call. Every path is validated
//! before the first operation runs, so a bad path late in the patch does not leave the earlier
//! operations applied.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType};
use crate::error::{AgentsError, Result};
use crate::run_context::RunContext;
use crate::sandbox::editor::SandboxEditor;
use crate::tool::{ApprovalPolicy, CustomTool};
use crate::util::BoxFuture;

/// The Lark grammar the model's patch is constrained to.
pub const APPLY_PATCH_CUSTOM_TOOL_GRAMMAR: &str = r#"start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF"#;

/// The tool description the model reads.
pub const APPLY_PATCH_CUSTOM_TOOL_DESCRIPTION: &str = r#"Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.
Your patch language is a stripped-down, file-oriented diff format designed to be easy to
parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more hunks, each introduced by @@ (optionally followed by a hunk header).
Within a hunk, each line starts with a space, -, or +.

For context lines:
- By default, show 3 lines of code immediately above and 3 lines immediately below each
change. If a change is within 3 lines of a previous change, do NOT duplicate the first
change's post-context lines in the second change's pre-context lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the
file, use the @@ operator to indicate the class or function to which the snippet belongs.
For instance:
@@ class BaseClass
[3 lines of pre-context]
-[old_code]
+[new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function that a single @@ statement
and 3 lines of context cannot uniquely identify the snippet, use multiple @@ statements to
jump to the right context. For instance:

@@ class BaseClass
@@ def method():
[3 lines of pre-context]
-[old_code]
+[new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] Hunk { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

Important:
- You must include a header with your intended action (Add/Delete/Update).
- You must prefix new lines with + even when creating a new file.
- File references can only be relative, NEVER ABSOLUTE."#;

/// The Responses `format` object the custom tool declares.
pub fn apply_patch_custom_tool_format() -> Value {
    json!({
        "type": "grammar",
        "syntax": "lark",
        "definition": APPLY_PATCH_CUSTOM_TOOL_GRAMMAR,
    })
}

const BEGIN_PATCH: &str = "*** Begin Patch";
const END_PATCH: &str = "*** End Patch";
const ADD_FILE: &str = "*** Add File: ";
const DELETE_FILE: &str = "*** Delete File: ";
const UPDATE_FILE: &str = "*** Update File: ";
const MOVE_TO: &str = "*** Move to: ";

fn invalid(message: impl Into<String>) -> AgentsError {
    AgentsError::user(message)
}

/// Parse the raw custom-tool input into operations.
///
/// Mirrors `_parse_custom_tool_input`: input that starts with `{` or `[` is the JSON form some
/// models fall back to; everything else is the freeform patch language.
pub fn parse_custom_input(raw_input: &str) -> Result<Vec<ApplyPatchOperation>> {
    let stripped = raw_input.trim_start();
    if stripped.starts_with('{') || stripped.starts_with('[') {
        return parse_json(raw_input);
    }
    parse_patch(raw_input)
}

fn parse_json(raw_input: &str) -> Result<Vec<ApplyPatchOperation>> {
    let payload: Value =
        serde_json::from_str(raw_input).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
    match &payload {
        Value::Object(map) => {
            if let Some(Value::Array(operations)) = map.get("operations") {
                return operations.iter().map(parse_json_operation).collect();
            }
            if let Some(operation) = map.get("operation")
                && !operation.is_null()
            {
                return Ok(vec![parse_json_operation(operation)?]);
            }
            Ok(vec![parse_json_operation(&payload)?])
        }
        Value::Array(operations) => operations.iter().map(parse_json_operation).collect(),
        _ => Err(invalid("apply_patch JSON input must be an object or array")),
    }
}

fn parse_json_operation(operation: &Value) -> Result<ApplyPatchOperation> {
    let Some(map) = operation.as_object() else {
        return Err(invalid("apply_patch operation must be an object"));
    };
    let raw_type = map.get("type").and_then(Value::as_str);
    let op_type = match raw_type {
        Some("create_file") => ApplyPatchOperationType::CreateFile,
        Some("update_file") => ApplyPatchOperationType::UpdateFile,
        Some("delete_file") => ApplyPatchOperationType::DeleteFile,
        _ => {
            return Err(invalid(format!(
                "Invalid apply_patch operation type: {}",
                map.get("type")
                    .map(Value::to_string)
                    .unwrap_or("None".into())
            )));
        }
    };
    let path = match map.get("path").and_then(Value::as_str) {
        Some(path) if !path.is_empty() => path.to_string(),
        _ => return Err(invalid("apply_patch operation is missing a path")),
    };
    let diff = match (op_type, map.get("diff").and_then(Value::as_str)) {
        (ApplyPatchOperationType::DeleteFile, _) => None,
        (_, Some(diff)) => Some(diff.to_string()),
        (_, None) => {
            return Err(invalid(format!(
                "apply_patch operation {} is missing a diff",
                raw_type.unwrap_or_default()
            )));
        }
    };
    let move_to = match map.get("move_to") {
        None | Some(Value::Null) => None,
        Some(Value::String(move_to)) => Some(move_to.clone()),
        Some(_) => return Err(invalid("apply_patch operation move_to must be a string")),
    };
    Ok(ApplyPatchOperation {
        op_type,
        path,
        diff,
        move_to,
    })
}

fn parse_patch(raw_input: &str) -> Result<Vec<ApplyPatchOperation>> {
    // The grammar uses LF delimiters; other line separators are file content.
    let mut lines: Vec<&str> = raw_input.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    let lines: Vec<&str> = lines
        .into_iter()
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();
    if lines.first() != Some(&BEGIN_PATCH) {
        return Err(invalid(
            "apply_patch input must start with '*** Begin Patch'",
        ));
    }
    if lines.len() < 2 || lines.last() != Some(&END_PATCH) {
        return Err(invalid("apply_patch input must end with '*** End Patch'"));
    }

    let mut operations = Vec::new();
    let mut index = 1;
    while index < lines.len() - 1 {
        let line = lines[index];
        let (operation, next) = if line.starts_with(ADD_FILE) {
            parse_add_file(&lines, index)?
        } else if line.starts_with(DELETE_FILE) {
            parse_delete_file(&lines, index)?
        } else if line.starts_with(UPDATE_FILE) {
            parse_update_file(&lines, index)?
        } else {
            return Err(invalid(format!(
                "Invalid apply_patch file operation header: {line}"
            )));
        };
        operations.push(operation);
        index = next;
    }
    if operations.is_empty() {
        return Err(invalid(
            "apply_patch input must include at least one file operation",
        ));
    }
    Ok(operations)
}

fn is_file_operation_header(line: &str) -> bool {
    line.starts_with(ADD_FILE) || line.starts_with(DELETE_FILE) || line.starts_with(UPDATE_FILE)
}

fn parse_path_header(line: &str, prefix: &str) -> Result<String> {
    let path = line.strip_prefix(prefix).unwrap_or(line).trim();
    if path.is_empty() {
        return Err(invalid(format!(
            "Missing path in apply_patch header: {line}"
        )));
    }
    Ok(path.to_string())
}

fn join_diff(lines: &[&str]) -> String {
    format!("{}\n", lines.join("\n"))
}

fn parse_add_file(lines: &[&str], mut index: usize) -> Result<(ApplyPatchOperation, usize)> {
    let path = parse_path_header(lines[index], ADD_FILE)?;
    index += 1;
    let mut diff_lines = Vec::new();
    while index < lines.len() - 1 && !is_file_operation_header(lines[index]) {
        let line = lines[index];
        if !line.starts_with('+') {
            return Err(invalid(format!("Invalid Add File line: {line}")));
        }
        diff_lines.push(line);
        index += 1;
    }
    if diff_lines.is_empty() {
        return Err(invalid(format!(
            "Add File patch for {path} must include at least one + line"
        )));
    }
    Ok((
        ApplyPatchOperation {
            op_type: ApplyPatchOperationType::CreateFile,
            path,
            diff: Some(join_diff(&diff_lines)),
            move_to: None,
        },
        index,
    ))
}

fn parse_delete_file(lines: &[&str], mut index: usize) -> Result<(ApplyPatchOperation, usize)> {
    let path = parse_path_header(lines[index], DELETE_FILE)?;
    index += 1;
    if index < lines.len() - 1 && !is_file_operation_header(lines[index]) {
        return Err(invalid(format!(
            "Delete File patch for {path} must not include a diff"
        )));
    }
    Ok((
        ApplyPatchOperation {
            op_type: ApplyPatchOperationType::DeleteFile,
            path,
            diff: None,
            move_to: None,
        },
        index,
    ))
}

fn parse_update_file(lines: &[&str], mut index: usize) -> Result<(ApplyPatchOperation, usize)> {
    let path = parse_path_header(lines[index], UPDATE_FILE)?;
    index += 1;
    let mut move_to = None;
    if index < lines.len() - 1 && lines[index].starts_with(MOVE_TO) {
        move_to = Some(parse_path_header(lines[index], MOVE_TO)?);
        index += 1;
    }
    let mut diff_lines = Vec::new();
    while index < lines.len() - 1 && !is_file_operation_header(lines[index]) {
        diff_lines.push(lines[index]);
        index += 1;
    }
    if diff_lines.is_empty() {
        return Err(invalid(format!(
            "Update File patch for {path} must include a hunk"
        )));
    }
    Ok((
        ApplyPatchOperation {
            op_type: ApplyPatchOperationType::UpdateFile,
            path,
            diff: Some(join_diff(&diff_lines)),
            move_to,
        },
        index,
    ))
}

/// Decides whether one canonicalized operation needs human approval.
pub type OperationApprovalFn<C> = Arc<
    dyn Fn(RunContext<C>, ApplyPatchOperation, String) -> BoxFuture<'static, bool> + Send + Sync,
>;

/// How the tool decides about approval.
#[derive(Clone)]
pub enum OperationApproval<C> {
    /// Always (true) or never (false).
    Static(bool),
    /// Decided per operation, with the operation's paths in canonical workspace-relative form.
    Dynamic(OperationApprovalFn<C>),
}

impl<C> std::fmt::Debug for OperationApproval<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationApproval::Static(value) => write!(f, "OperationApproval::Static({value})"),
            OperationApproval::Dynamic(_) => f.write_str("OperationApproval::Dynamic(..)"),
        }
    }
}

/// The freeform `apply_patch` tool bound to a sandbox editor.
///
/// Build the model-facing tool with [`into_custom_tool`](Self::into_custom_tool).
pub struct SandboxApplyPatchTool<C> {
    editor: SandboxEditor,
    needs_approval: OperationApproval<C>,
}

impl<C> std::fmt::Debug for SandboxApplyPatchTool<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxApplyPatchTool")
            .field("needs_approval", &self.needs_approval)
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> SandboxApplyPatchTool<C> {
    /// The tool over `editor`, which carries the session, the run's cwd and the run-as user.
    pub fn new(editor: SandboxEditor) -> Self {
        SandboxApplyPatchTool {
            editor,
            needs_approval: OperationApproval::Static(false),
        }
    }

    /// Require approval for every call (or none).
    pub fn with_needs_approval(mut self, needs_approval: bool) -> Self {
        self.needs_approval = OperationApproval::Static(needs_approval);
        self
    }

    /// Decide approval per operation. The operation's paths are canonical: relative to the
    /// workspace root, with the run's cwd applied and separators normalized, so a rule written
    /// for `sensitive/secret.txt` also catches `sensitive\secret.txt`.
    pub fn with_needs_approval_fn<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(RunContext<C>, ApplyPatchOperation, String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = bool> + Send + 'static,
    {
        self.needs_approval =
            OperationApproval::Dynamic(Arc::new(move |c, o, id| Box::pin(f(c, o, id))));
        self
    }

    /// The editor operations are applied through.
    pub fn editor(&self) -> &SandboxEditor {
        &self.editor
    }

    /// Parse the raw model input into operations.
    pub fn parse_custom_input(&self, raw_input: &str) -> Result<Vec<ApplyPatchOperation>> {
        parse_custom_input(raw_input)
    }

    /// Parse the input and validate every path, returning canonical operations.
    pub fn parse_and_normalize_input(&self, raw_input: &str) -> Result<Vec<ApplyPatchOperation>> {
        parse_custom_input(raw_input)?
            .iter()
            .map(|operation| self.editor.normalize_operation(operation))
            .collect()
    }

    /// Run the operations of one call. Every path is validated before the first is applied.
    pub async fn apply(&self, raw_input: &str) -> Result<String> {
        apply_operations(&self.editor, raw_input).await
    }

    /// The Responses custom tool the model calls.
    pub fn into_custom_tool(self) -> CustomTool<C> {
        let editor = self.editor.clone();
        let mut tool = CustomTool::<C>::new(
            "apply_patch",
            APPLY_PATCH_CUSTOM_TOOL_DESCRIPTION,
            move |_ctx, raw_input| {
                let editor = editor.clone();
                async move { apply_operations(&editor, &raw_input).await }
            },
        )
        .with_format(apply_patch_custom_tool_format());

        let editor = self.editor;
        tool.needs_approval = match self.needs_approval {
            OperationApproval::Static(value) => ApprovalPolicy::Static(value),
            OperationApproval::Dynamic(check) => {
                ApprovalPolicy::DynamicForAgent(Arc::new(
                    move |ctx, agent, raw_input: String, call_id| {
                        let editor = editor.clone();
                        let check = check.clone();
                        Box::pin(async move {
                            let operations = match parse_custom_input(&raw_input).and_then(|ops| {
                                ops.iter()
                                    .map(|operation| editor.normalize_operation(operation))
                                    .collect::<Result<Vec<_>>>()
                            }) {
                                Ok(operations) => operations,
                                // A malformed patch or an invalid path flows through normal tool
                                // execution so the model gets a recoverable tool error instead of
                                // the run aborting during the approval pre-check.
                                Err(_) => return false,
                            };
                            // Once a decision lands for the call, the remaining operations are
                            // not checked: the decision governs the whole call.
                            let key = crate::run_context::scoped_tool_key(
                                "apply_patch",
                                Some("custom_tool_call"),
                            );
                            for operation in operations {
                                let needs = check(ctx.clone(), operation, call_id.clone()).await;
                                if needs || ctx.is_tool_approved(&agent, &key, &call_id).is_some() {
                                    return needs;
                                }
                            }
                            false
                        })
                    },
                ))
            }
        };
        tool
    }
}

async fn apply_operations(editor: &SandboxEditor, raw_input: &str) -> Result<String> {
    let operations = parse_custom_input(raw_input)?;
    // Validate every operation before executing any of them. The raw paths are kept for
    // execution so an absolute model path does not lose its identity before the scoped editor
    // applies the run cwd.
    for operation in &operations {
        editor.normalize_operation(operation)?;
    }
    let mut outputs = Vec::new();
    for operation in &operations {
        let result = match operation.op_type {
            ApplyPatchOperationType::CreateFile => editor.create_file(operation).await?,
            ApplyPatchOperationType::UpdateFile => editor.update_file(operation).await?,
            ApplyPatchOperationType::DeleteFile => editor.delete_file(operation).await?,
        };
        if let Some(output) = result.output.filter(|output| !output.is_empty()) {
            outputs.push(output);
        }
    }
    Ok(outputs.join("\n"))
}
