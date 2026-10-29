//! Apply V4A diffs (the format used by the `apply_patch` tool) to text.
//!
//! Port of `agents.apply_diff`.

use crate::error::{AgentsError, Result};

const END_PATCH: &str = "*** End Patch";
const END_FILE: &str = "*** End of File";
const SECTION_TERMINATORS: [&str; 4] = [
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
];
const END_SECTION_MARKERS: [&str; 5] = [
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
    END_FILE,
];

/// How a diff is interpreted by [`apply_diff`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApplyDiffMode {
    /// An update diff with context hunks.
    #[default]
    Default,
    /// A create-file diff (only `+` prefixed lines).
    Create,
}

#[derive(Debug, Clone)]
struct Chunk {
    orig_index: usize,
    del_lines: Vec<String>,
    ins_lines: Vec<String>,
}

struct ParserState {
    lines: Vec<String>,
    index: usize,
    fuzz: usize,
}

struct ReadSectionResult {
    next_context: Vec<String>,
    section_chunks: Vec<Chunk>,
    end_index: usize,
    eof: bool,
}

fn invalid(message: impl Into<String>) -> AgentsError {
    AgentsError::other(message.into())
}

/// Apply a V4A diff to `input`.
///
/// Understands both the create-file syntax (only `+` prefixed lines) and the update syntax
/// with `@@` anchors and context lines. The newline style of the input is preserved.
pub fn apply_diff(input: &str, diff: &str, mode: ApplyDiffMode) -> Result<String> {
    let newline = detect_newline(input, diff, mode);
    let diff_lines = normalize_diff_lines(diff);
    if mode == ApplyDiffMode::Create {
        return parse_create_diff(diff_lines, newline);
    }
    let normalized_input = input.replace("\r\n", "\n");
    let chunks = parse_update_diff(diff_lines, &normalized_input)?;
    apply_chunks(&normalized_input, &chunks, newline)
}

fn normalize_diff_lines(diff: &str) -> Vec<String> {
    let mut lines: Vec<String> = diff
        .split('\n')
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect();
    if lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

fn newline_of(text: &str) -> &'static str {
    if text.contains("\r\n") { "\r\n" } else { "\n" }
}

fn detect_newline(input: &str, diff: &str, mode: ApplyDiffMode) -> &'static str {
    if mode != ApplyDiffMode::Create && input.contains('\n') {
        return newline_of(input);
    }
    newline_of(diff)
}

fn is_done(state: &ParserState, prefixes: &[&str]) -> bool {
    match state.lines.get(state.index) {
        None => true,
        Some(line) => prefixes.iter().any(|p| line.starts_with(p)),
    }
}

fn read_str(state: &mut ParserState, prefix: &str) -> String {
    if let Some(current) = state.lines.get(state.index)
        && let Some(rest) = current.strip_prefix(prefix)
    {
        let rest = rest.to_string();
        state.index += 1;
        return rest;
    }
    String::new()
}

fn parse_create_diff(mut lines: Vec<String>, newline: &str) -> Result<String> {
    lines.push(END_PATCH.to_string());
    let mut parser = ParserState {
        lines,
        index: 0,
        fuzz: 0,
    };
    let mut output = Vec::new();
    while !is_done(&parser, &SECTION_TERMINATORS) {
        let line = parser.lines[parser.index].clone();
        parser.index += 1;
        match line.strip_prefix('+') {
            Some(rest) => output.push(rest.to_string()),
            None => return Err(invalid(format!("Invalid Add File Line: {line}"))),
        }
    }
    Ok(output.join(newline))
}

fn parse_update_diff(mut lines: Vec<String>, input: &str) -> Result<Vec<Chunk>> {
    lines.push(END_PATCH.to_string());
    let mut parser = ParserState {
        lines,
        index: 0,
        fuzz: 0,
    };
    let input_lines: Vec<&str> = input.split('\n').collect();
    let mut chunks = Vec::new();
    let mut cursor = 0usize;

    while !is_done(&parser, &END_SECTION_MARKERS) {
        let (anchors, anchor_count) = read_anchors(&mut parser);
        if !(anchor_count > 0 || cursor == 0) {
            let current = parser.lines.get(parser.index).cloned().unwrap_or_default();
            return Err(invalid(format!("Invalid Line:\n{current}")));
        }
        let require_match = anchor_count > 1;
        for (i, anchor) in anchors.iter().enumerate() {
            cursor = advance_cursor_to_anchor(
                anchor,
                &input_lines,
                cursor,
                &mut parser,
                require_match,
                i > 0,
            )?;
        }

        let section = read_section(&parser.lines, parser.index)?;
        let (new_index, fuzz) =
            find_context(&input_lines, &section.next_context, cursor, section.eof);
        let Some(new_index) = new_index else {
            let ctx = section.next_context.join("\n");
            return Err(invalid(if section.eof {
                format!("Invalid EOF Context {cursor}:\n{ctx}")
            } else {
                format!("Invalid Context {cursor}:\n{ctx}")
            }));
        };
        cursor = new_index + section.next_context.len();
        parser.fuzz += fuzz;
        parser.index = section.end_index;
        for ch in section.section_chunks {
            chunks.push(Chunk {
                orig_index: ch.orig_index + new_index,
                ..ch
            });
        }
    }
    let _ = parser.fuzz;
    Ok(chunks)
}

/// Consume the `@@` header lines introducing one hunk. Returns the non-empty anchors and the
/// number of consumed headers (including bare `@@` markers).
fn read_anchors(parser: &mut ParserState) -> (Vec<String>, usize) {
    let mut anchors = Vec::new();
    let mut count = 0;
    loop {
        let start = parser.index;
        let anchor = read_str(parser, "@@ ");
        let mut consumed = parser.index != start;
        if !consumed && parser.lines.get(parser.index).is_some_and(|l| l == "@@") {
            parser.index += 1;
            consumed = true;
        }
        if !consumed {
            break;
        }
        count += 1;
        if !anchor.trim().is_empty() {
            anchors.push(anchor);
        }
    }
    (anchors, count)
}

fn advance_cursor_to_anchor(
    anchor: &str,
    input_lines: &[&str],
    mut cursor: usize,
    parser: &mut ParserState,
    require_match: bool,
    force_forward_search: bool,
) -> Result<usize> {
    let before = &input_lines[..cursor.min(input_lines.len())];
    let mut found = false;
    if !force_forward_search && before.contains(&anchor) {
        found = true;
    } else if let Some(i) = (cursor..input_lines.len()).find(|&i| input_lines[i] == anchor) {
        cursor = i + 1;
        found = true;
    }
    if !found {
        let trimmed = anchor.trim();
        if !force_forward_search && before.iter().any(|l| l.trim() == trimmed) {
            found = true;
        } else if let Some(i) =
            (cursor..input_lines.len()).find(|&i| input_lines[i].trim() == trimmed)
        {
            cursor = i + 1;
            parser.fuzz += 1;
            found = true;
        }
    }
    if require_match && !found {
        return Err(invalid(format!("Invalid Anchor {cursor}:\n{anchor}")));
    }
    Ok(cursor)
}

fn read_section(lines: &[String], start_index: usize) -> Result<ReadSectionResult> {
    #[derive(PartialEq, Clone, Copy)]
    enum Mode {
        Keep,
        Add,
        Delete,
    }
    let mut context: Vec<String> = Vec::new();
    let mut del_lines: Vec<String> = Vec::new();
    let mut ins_lines: Vec<String> = Vec::new();
    let mut chunks = Vec::new();
    let mut mode = Mode::Keep;
    let mut index = start_index;

    while index < lines.len() {
        let raw = lines[index].as_str();
        if raw.starts_with("@@")
            || raw.starts_with(END_PATCH)
            || raw.starts_with("*** Update File:")
            || raw.starts_with("*** Delete File:")
            || raw.starts_with("*** Add File:")
            || raw.starts_with(END_FILE)
            || raw == "***"
        {
            break;
        }
        if raw.starts_with("***") {
            return Err(invalid(format!("Invalid Line: {raw}")));
        }
        index += 1;
        let last_mode = mode;
        let line = if raw.is_empty() { " " } else { raw };
        let mut chars = line.chars();
        let prefix = chars.next().unwrap_or(' ');
        mode = match prefix {
            '+' => Mode::Add,
            '-' => Mode::Delete,
            ' ' => Mode::Keep,
            _ => return Err(invalid(format!("Invalid Line: {line}"))),
        };
        let content = chars.as_str().to_string();
        if mode == Mode::Keep
            && last_mode != mode
            && (!del_lines.is_empty() || !ins_lines.is_empty())
        {
            chunks.push(Chunk {
                orig_index: context.len() - del_lines.len(),
                del_lines: std::mem::take(&mut del_lines),
                ins_lines: std::mem::take(&mut ins_lines),
            });
        }
        match mode {
            Mode::Delete => {
                del_lines.push(content.clone());
                context.push(content);
            }
            Mode::Add => ins_lines.push(content),
            Mode::Keep => context.push(content),
        }
    }
    if !del_lines.is_empty() || !ins_lines.is_empty() {
        chunks.push(Chunk {
            orig_index: context.len() - del_lines.len(),
            del_lines,
            ins_lines,
        });
    }
    if lines.get(index).is_some_and(|l| l == END_FILE) {
        return Ok(ReadSectionResult {
            next_context: context,
            section_chunks: chunks,
            end_index: index + 1,
            eof: true,
        });
    }
    if index == start_index {
        let next = lines.get(index).cloned().unwrap_or_default();
        return Err(invalid(format!(
            "Nothing in this section - index={index} {next}"
        )));
    }
    Ok(ReadSectionResult {
        next_context: context,
        section_chunks: chunks,
        end_index: index,
        eof: false,
    })
}

fn find_context(
    lines: &[&str],
    context: &[String],
    start: usize,
    eof: bool,
) -> (Option<usize>, usize) {
    if eof {
        // A trailing empty element is the file terminator, not a line.
        let search = if lines.last().is_some_and(|l| l.is_empty()) {
            &lines[..lines.len() - 1]
        } else {
            lines
        };
        let end_start = search.len().saturating_sub(context.len());
        let end_match = find_context_core(search, context, end_start);
        if end_match.0.is_some() {
            return end_match;
        }
        let fallback = find_context_core(search, context, start.min(search.len()));
        return (fallback.0, fallback.1 + 10000);
    }
    find_context_core(lines, context, start)
}

type LineMap = fn(&str) -> &str;

fn find_context_core(lines: &[&str], context: &[String], start: usize) -> (Option<usize>, usize) {
    if context.is_empty() {
        return (Some(start), 0);
    }
    let passes: [(LineMap, usize); 3] = [(|s| s, 0), (|s| s.trim_end(), 1), (|s| s.trim(), 100)];
    for (map, fuzz) in passes {
        for i in start..lines.len() {
            if i + context.len() <= lines.len()
                && context
                    .iter()
                    .enumerate()
                    .all(|(o, c)| map(lines[i + o]) == map(c))
            {
                return (Some(i), fuzz);
            }
        }
    }
    (None, 0)
}

fn apply_chunks(input: &str, chunks: &[Chunk], newline: &str) -> Result<String> {
    let orig: Vec<&str> = input.split('\n').collect();
    let mut dest: Vec<&str> = Vec::new();
    let mut cursor = 0usize;
    for chunk in chunks {
        if chunk.orig_index > orig.len() {
            return Err(invalid(format!(
                "applyDiff: chunk.origIndex {} > input length {}",
                chunk.orig_index,
                orig.len()
            )));
        }
        if cursor > chunk.orig_index {
            return Err(invalid(format!(
                "applyDiff: overlapping chunk at {} (cursor {cursor})",
                chunk.orig_index
            )));
        }
        dest.extend_from_slice(&orig[cursor..chunk.orig_index]);
        cursor = chunk.orig_index;
        dest.extend(chunk.ins_lines.iter().map(String::as_str));
        cursor += chunk.del_lines.len();
    }
    if cursor < orig.len() {
        dest.extend_from_slice(&orig[cursor..]);
    }
    Ok(dest.join(newline))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_mode() {
        let out = apply_diff("", "+hello\n+world\n", ApplyDiffMode::Create).unwrap();
        assert_eq!(out, "hello\nworld");
        assert!(apply_diff("", "oops", ApplyDiffMode::Create).is_err());
    }

    #[test]
    fn update_with_context() {
        let input = "line1\nline2\nline3\n";
        let diff = "@@\n line1\n-line2\n+changed\n line3\n";
        let out = apply_diff(input, diff, ApplyDiffMode::Default).unwrap();
        assert_eq!(out, "line1\nchanged\nline3\n");
    }

    #[test]
    fn update_with_anchor_and_crlf() {
        let input = "fn a() {\r\n    1\r\n}\r\nfn b() {\r\n    1\r\n}\r\n";
        let diff = "@@ fn b() {\n-    1\n+    2\n }\n";
        let out = apply_diff(input, diff, ApplyDiffMode::Default).unwrap();
        assert_eq!(out, "fn a() {\r\n    1\r\n}\r\nfn b() {\r\n    2\r\n}\r\n");
    }

    #[test]
    fn eof_hunk_appends() {
        let input = "a\nb\n";
        let diff = "@@\n b\n+c\n*** End of File\n";
        let out = apply_diff(input, diff, ApplyDiffMode::Default).unwrap();
        assert_eq!(out, "a\nb\nc\n");
    }

    #[test]
    fn invalid_context_errors() {
        assert!(apply_diff("a\n", "@@\n-zzz\n", ApplyDiffMode::Default).is_err());
    }
}
