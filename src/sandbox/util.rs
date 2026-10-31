//! Small helpers shared by the sandbox modules. Mirrors `agents.sandbox.util`.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Map, Value};

use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::session::{ExecRequest, SandboxSession, shell_quote};

/// Encode bytes as standard base64 with padding, for `view_image` data URLs.
pub fn base64_encode(data: &[u8]) -> String {
    BASE64.encode(data)
}

/// The largest image `view_image` will inline into the model's context.
///
/// Mirrors Python's `_MAX_IMAGE_BYTES`. Without it a single `view_image` call can base64 an
/// arbitrarily large file straight into the conversation.
pub const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;

/// The human-readable form of [`MAX_IMAGE_BYTES`], for the tool's refusal message.
pub const MAX_IMAGE_SIZE_LABEL: &str = "10MB";

/// How much of a file is examined when looking for an SVG root element.
const SVG_SNIFF_BYTES: usize = 2048;

/// Detect an image media type from the file's own bytes. Mirrors
/// `_detect_image_mime_type`.
///
/// Returns `None` for a file that is not a recognized image, so `view_image` can refuse it
/// instead of sending the model a data URL that claims to be a PNG. The extension is consulted
/// only for SVG, which has no magic number.
pub fn detect_image_media_type(path: &str, payload: &[u8]) -> Option<&'static str> {
    if payload.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if payload.starts_with(b"\xff\xd8\xff") {
        return Some("image/jpeg");
    }
    if payload.starts_with(b"GIF87a") || payload.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if payload.starts_with(b"RIFF") && payload.get(8..12) == Some(b"WEBP") {
        return Some("image/webp");
    }
    if payload.starts_with(b"BM") {
        return Some("image/bmp");
    }
    if payload.starts_with(b"II*\x00") || payload.starts_with(b"MM\x00*") {
        return Some("image/tiff");
    }

    let snippet: Vec<u8> = payload
        .iter()
        .take(SVG_SNIFF_BYTES)
        .copied()
        .collect::<Vec<u8>>()
        .iter()
        .copied()
        .skip_while(|byte| byte.is_ascii_whitespace())
        .collect();
    let lowercase = snippet.to_ascii_lowercase();
    if lowercase.starts_with(b"<svg")
        || (lowercase.starts_with(b"<?xml") && lowercase.windows(4).any(|window| window == b"<svg"))
    {
        return Some("image/svg+xml");
    }

    let path = path.to_ascii_lowercase();
    if path.ends_with(".svg") || path.ends_with(".svgz") {
        return Some("image/svg+xml");
    }
    None
}

/// Guess an image media type from a file extension, defaulting to `image/png`.
///
/// Prefer [`detect_image_media_type`], which reads the file's own bytes; this is the fallback
/// for callers that only have a name.
pub fn image_media_type(path: &str) -> &'static str {
    let lowercase = path.to_ascii_lowercase();
    match lowercase.rsplit('.').next() {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("svg") => "image/svg+xml",
        _ => "image/png",
    }
}

/// Recursively merge `overlay` into `base`, with objects merged key by key.
pub fn deep_merge(base: &mut Map<String, Value>, overlay: &Map<String, Value>) {
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(Value::Object(existing)), Value::Object(incoming)) => {
                deep_merge(existing, incoming);
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Approximate bytes per token used by the sandbox truncation helpers. Mirrors
/// `APPROX_BYTES_PER_TOKEN`.
pub const APPROX_BYTES_PER_TOKEN: usize = 4;

/// The approximate token count of `text`.
pub fn approx_token_count(text: &str) -> usize {
    text.len().div_ceil(APPROX_BYTES_PER_TOKEN)
}

/// The byte budget for a token budget.
pub fn approx_bytes_for_tokens(tokens: usize) -> usize {
    tokens.saturating_mul(APPROX_BYTES_PER_TOKEN)
}

/// The approximate token count of a byte count. Mirrors `approx_tokens_from_byte_count`.
pub fn approx_tokens_from_byte_count(byte_count: usize) -> usize {
    byte_count.div_ceil(APPROX_BYTES_PER_TOKEN)
}

/// Whether a [`TruncationPolicy`] counts bytes or approximate tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationMode {
    /// The limit is a byte count.
    Bytes,
    /// The limit is an approximate token count.
    Tokens,
}

/// How much of a tool's output is kept. Mirrors `TruncationPolicy`.
///
/// Truncation elides the *middle*: the head says what the command was doing and the tail says
/// how it ended, and a marker in between reports how much is missing, so the model is never left
/// guessing whether it saw the whole output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncationPolicy {
    /// What the limit counts.
    pub mode: TruncationMode,
    /// The limit itself.
    pub limit: usize,
}

impl TruncationPolicy {
    /// A policy limiting the output to `limit` bytes.
    pub fn bytes(limit: usize) -> Self {
        TruncationPolicy {
            mode: TruncationMode::Bytes,
            limit,
        }
    }

    /// A policy limiting the output to `limit` approximate tokens.
    pub fn tokens(limit: usize) -> Self {
        TruncationPolicy {
            mode: TruncationMode::Tokens,
            limit,
        }
    }

    /// The budget expressed in approximate tokens.
    pub fn token_budget(self) -> usize {
        match self.mode {
            TruncationMode::Bytes => approx_tokens_from_byte_count(self.limit),
            TruncationMode::Tokens => self.limit,
        }
    }

    /// The budget expressed in bytes.
    pub fn byte_budget(self) -> usize {
        match self.mode {
            TruncationMode::Bytes => self.limit,
            TruncationMode::Tokens => approx_bytes_for_tokens(self.limit),
        }
    }

    /// The marker written where the elided middle was. Mirrors `format_truncation_marker`.
    pub fn marker(self, removed: usize) -> String {
        match self.mode {
            TruncationMode::Tokens => format!("…{removed} tokens truncated…"),
            TruncationMode::Bytes => format!("…{removed} chars truncated…"),
        }
    }

    /// How much was removed, in the unit this policy reports. Mirrors `removed_units_for_source`.
    pub fn removed_units(self, removed_bytes: usize, removed_chars: usize) -> usize {
        match self.mode {
            TruncationMode::Tokens => approx_tokens_from_byte_count(removed_bytes),
            TruncationMode::Bytes => removed_chars,
        }
    }
}

/// Split a budget into a head half and a tail half. Mirrors `split_budget`.
pub fn split_budget(budget: usize) -> (usize, usize) {
    let left = budget / 2;
    (left, budget - left)
}

/// Split `text` into the head that fits `head_bytes` and the tail that fits `tail_bytes`.
///
/// Mirrors `split_string`. Both halves end on character boundaries, so the result is always
/// valid UTF-8 even when the budget falls inside a multi-byte character. Returns the number of
/// characters dropped from the middle.
pub fn split_string(text: &str, head_bytes: usize, tail_bytes: usize) -> (usize, &str, &str) {
    if text.is_empty() {
        return (0, "", "");
    }
    let length = text.len();
    let tail_start_target = length.saturating_sub(tail_bytes);
    let mut prefix_end = 0;
    let mut suffix_start = length;
    let mut removed_chars = 0;
    let mut suffix_started = false;

    for (index, character) in text.char_indices() {
        let char_end = index + character.len_utf8();
        if char_end <= head_bytes {
            prefix_end = char_end;
            continue;
        }
        if index >= tail_start_target {
            if !suffix_started {
                suffix_start = index;
                suffix_started = true;
            }
            continue;
        }
        removed_chars += 1;
    }
    if suffix_start < prefix_end {
        suffix_start = prefix_end;
    }
    (removed_chars, &text[..prefix_end], &text[suffix_start..])
}

/// Keep `max_bytes` of `text` from the front, stopping on a character boundary.
fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Elide the middle of `text` so it fits `policy`, with `prefix` kept at the front.
fn truncate_middle(text: &str, policy: TruncationPolicy, prefix: &str) -> String {
    let max_bytes = policy.byte_budget();
    if max_bytes == 0 {
        return String::new();
    }
    let marker = policy.marker(approx_token_count(text));
    // When even the header plus the marker will not fit, the marker alone has to.
    let prefix = if prefix.len() + marker.len() > max_bytes {
        ""
    } else {
        prefix
    };
    let Some(content_budget) = max_bytes.checked_sub(prefix.len() + marker.len()) else {
        return truncate_utf8(&marker, max_bytes).to_string();
    };

    let (head_budget, tail_budget) = split_budget(content_budget);
    let (removed_chars, head, tail) = split_string(text, head_budget, tail_budget);
    let removed_bytes = text.len() - head.len() - tail.len();
    let marker = policy.marker(policy.removed_units(removed_bytes, removed_chars));
    format!("{prefix}{head}{marker}{tail}")
}

/// Truncate `text` under `policy`, eliding the middle. Mirrors `truncate_text`.
///
/// The two modes do not share an implementation, because they answer different questions. A
/// token budget is an estimate the model has to live inside, so the marker is paid for out of
/// the budget; a byte budget says how much *content* to keep, so the marker is added on top.
pub fn truncate_text(text: &str, policy: TruncationPolicy) -> String {
    match policy.mode {
        TruncationMode::Bytes => truncate_with_byte_estimate(text, policy),
        TruncationMode::Tokens => truncate_with_token_budget(text, policy).0,
    }
}

/// Truncate `text` to a token budget, reporting the original token count when it was cut.
///
/// Mirrors `truncate_with_token_budget`: the marker comes out of the budget, so the result
/// stays inside it.
pub fn truncate_with_token_budget(text: &str, policy: TruncationPolicy) -> (String, Option<usize>) {
    if text.is_empty() {
        return (String::new(), None);
    }
    let max_tokens = policy.token_budget();
    if max_tokens > 0 && text.len() <= approx_bytes_for_tokens(max_tokens) {
        return (text.to_string(), None);
    }
    let approx_total = approx_token_count(text);
    let truncated = truncate_middle(text, policy, "");
    if truncated == text {
        return (truncated, None);
    }
    (truncated, Some(approx_total))
}

/// Keep `policy.byte_budget()` bytes of `text` and report the elided characters.
///
/// Mirrors `truncate_with_byte_estimate`. The head and the tail together fill the whole budget
/// and the marker is appended between them, so a zero budget still yields the marker rather
/// than an empty string that looks like a command which printed nothing.
pub fn truncate_with_byte_estimate(text: &str, policy: TruncationPolicy) -> String {
    if text.is_empty() {
        return String::new();
    }
    let total_chars = text.chars().count();
    let max_bytes = policy.byte_budget();
    if max_bytes == 0 {
        return policy.marker(policy.removed_units(text.len(), total_chars));
    }
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let (head_budget, tail_budget) = split_budget(max_bytes);
    let (removed_chars, head, tail) = split_string(text, head_budget, tail_budget);
    let marker = policy.marker(policy.removed_units(text.len() - max_bytes, removed_chars));
    format!("{head}{marker}{tail}")
}

/// Truncate `text` under `policy` and prepend the total line count. Mirrors
/// `formatted_truncate_text`.
///
/// The line count is what tells the model the output was long, not just cut: without it a
/// truncated listing looks like a complete short one.
pub fn formatted_truncate_text(text: &str, policy: TruncationPolicy) -> String {
    if text.len() <= policy.byte_budget() {
        return text.to_string();
    }
    let prefix = format!("Total output lines: {}\n\n", text.lines().count());
    match policy.mode {
        TruncationMode::Tokens => truncate_middle(text, policy, &prefix),
        TruncationMode::Bytes => format!("{prefix}{}", truncate_text(text, policy)),
    }
}

/// Truncate `text` to a token budget, reporting the original token count when it was cut.
///
/// Mirrors `formatted_truncate_text_with_token_count`, which is what the shell and PTY tools
/// call: `None` means no budget and the text is returned untouched.
pub fn formatted_truncate_text_with_token_count(
    text: &str,
    max_output_tokens: Option<usize>,
) -> (String, Option<usize>) {
    let Some(max_output_tokens) = max_output_tokens else {
        return (text.to_string(), None);
    };
    let policy = TruncationPolicy::tokens(max_output_tokens);
    if text.len() <= policy.byte_budget() {
        return (text.to_string(), None);
    }
    let prefix = format!("Total output lines: {}\n\n", text.lines().count());
    (
        truncate_middle(text, policy, &prefix),
        Some(approx_token_count(text)),
    )
}

/// Truncate `text` to a token budget, as the Python sandbox tools do.
///
/// Kept as the name the sandbox tools already call;
/// [`formatted_truncate_text_with_token_count`] is the same function under Python's name.
pub fn truncate_text_by_tokens(
    text: &str,
    max_output_tokens: Option<usize>,
) -> (String, Option<usize>) {
    formatted_truncate_text_with_token_count(text, max_output_tokens)
}

/// Fail unless `git` is callable inside the sandbox. Mirrors `agents.sandbox.util.github`.
pub async fn ensure_git_available(session: &dyn SandboxSession) -> SandboxResult<()> {
    let probe = session
        .exec(ExecRequest::shell("command -v git >/dev/null 2>&1"))
        .await?;
    if probe.ok() {
        return Ok(());
    }
    Err(SandboxError::new(
        ErrorCode::GitMissingInImage,
        "git is required to use github_repo artifacts",
    )
    .with_retryable(false))
}

/// Shallow-clone a GitHub repository at `git_ref` into `dest` inside the sandbox.
///
/// Mirrors `agents.sandbox.util.github.clone_repo`, with the clone running inside the sandbox
/// rather than on the host: the destination is a workspace path, so cloning on the host would
/// only be useful for a backend that shares the host filesystem. A tag or branch is fetched with
/// a shallow clone; a commit sha, which `--branch` does not accept, falls back to a checkout of
/// a repository cloned without one.
pub async fn clone_repo(
    session: &dyn SandboxSession,
    repo: &str,
    git_ref: &str,
    dest: &str,
) -> SandboxResult<()> {
    ensure_git_available(session).await?;
    let target = session.normalize_path_for_write(dest)?;
    let url = format!("https://github.com/{repo}.git");
    let quoted_target = shell_quote(&target);
    let quoted_url = shell_quote(&url);
    let quoted_ref = shell_quote(git_ref);

    let shallow = format!(
        "set -e\nmkdir -p \"$(dirname {target})\"\nrm -rf {target}\n\
         git clone --depth 1 --no-tags --branch {git_ref} {url} {target}",
        target = quoted_target,
        url = quoted_url,
        git_ref = quoted_ref,
    );
    if session.exec(ExecRequest::shell(shallow)).await?.ok() {
        return Ok(());
    }

    let pinned = format!(
        "set -e\nmkdir -p \"$(dirname {target})\"\nrm -rf {target}\n\
         git clone --no-checkout {url} {target}\ngit -C {target} checkout {git_ref}",
        target = quoted_target,
        url = quoted_url,
        git_ref = quoted_ref,
    );
    let result = session.exec(ExecRequest::shell(pinned)).await?;
    if !result.ok() {
        return Err(SandboxError::new(
            ErrorCode::GitCloneError,
            format!(
                "Failed to clone `{url}` at `{git_ref}`: {}",
                result.stderr_text().trim()
            ),
        )
        .with_context("repo", repo)
        .with_context("ref", git_ref));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn truncates_to_a_token_budget() {
        let text = "a".repeat(400);
        let (out, original) = truncate_text_by_tokens(&text, None);
        assert_eq!(out, text);
        assert_eq!(original, None);

        let (out, original) = truncate_text_by_tokens(&text, Some(1000));
        assert_eq!(out, text);
        assert_eq!(original, None);

        // A budget large enough for the header keeps it, and the middle is elided.
        let (out, original) = truncate_text_by_tokens(&text, Some(40));
        assert!(out.starts_with("Total output lines: 1\n\n"), "{out}");
        assert!(out.contains("tokens truncated"), "{out}");
        assert!(out.ends_with('a'), "{out}");
        assert!(out.len() <= approx_bytes_for_tokens(40), "{}", out.len());
        assert_eq!(original, Some(100));

        // A budget too small for the header drops it rather than the marker.
        let (out, _) = truncate_text_by_tokens(&text, Some(10));
        assert!(!out.starts_with("Total output lines"), "{out}");
        assert!(out.contains("truncated"), "{out}");
    }

    #[test]
    fn elides_the_middle_rather_than_the_head() {
        let text = format!("{}{}{}", "H".repeat(100), "M".repeat(200), "T".repeat(100));
        let out = truncate_text(&text, TruncationPolicy::bytes(80));
        assert!(out.starts_with('H'), "{out}");
        assert!(out.ends_with('T'), "{out}");
        assert!(!out.contains('M'), "{out}");
        assert!(out.contains("chars truncated"), "{out}");
    }

    #[test]
    fn a_policy_converts_between_bytes_and_tokens() {
        assert_eq!(TruncationPolicy::tokens(10).byte_budget(), 40);
        assert_eq!(TruncationPolicy::bytes(40).token_budget(), 10);
        // A partial token still costs a whole one.
        assert_eq!(TruncationPolicy::bytes(41).token_budget(), 11);
    }

    #[test]
    fn text_within_the_budget_is_untouched() {
        assert_eq!(
            truncate_text("short", TruncationPolicy::bytes(100)),
            "short"
        );
        assert_eq!(truncate_text("", TruncationPolicy::bytes(0)), "");
        assert_eq!(
            formatted_truncate_text("short", TruncationPolicy::tokens(100)),
            "short"
        );
    }

    #[test]
    fn keeps_character_boundaries() {
        let text = "é".repeat(200);
        let (out, _) = truncate_text_by_tokens(&text, Some(12));
        assert!(out.is_char_boundary(0));
        // The result is valid UTF-8 by construction, so this must not panic.
        assert!(!out.is_empty());
    }

    #[test]
    fn encodes_base64() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(&[0xff, 0xfe, 0xfd]), "//79");
    }

    #[test]
    fn guesses_media_types() {
        assert_eq!(image_media_type("a/b.PNG"), "image/png");
        assert_eq!(image_media_type("a/b.jpeg"), "image/jpeg");
        assert_eq!(image_media_type("a/b"), "image/png");
    }

    #[test]
    fn detects_image_types_from_their_bytes() {
        assert_eq!(
            detect_image_media_type("a.bin", b"\x89PNG\r\n\x1a\n...."),
            Some("image/png")
        );
        assert_eq!(
            detect_image_media_type("a.png", b"\xff\xd8\xffmore"),
            Some("image/jpeg")
        );
        assert_eq!(
            detect_image_media_type("a.bin", b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            Some("image/webp")
        );
        assert_eq!(
            detect_image_media_type("a.bin", b"  <SVG xmlns=\"x\">"),
            Some("image/svg+xml")
        );
        assert_eq!(
            detect_image_media_type("a.svgz", b"compressed"),
            Some("image/svg+xml")
        );
        // A file that is not an image is refused rather than mislabelled as a PNG.
        assert_eq!(detect_image_media_type("notes.txt", b"hello"), None);
    }

    #[test]
    fn merges_nested_objects() {
        let mut base = json!({"a": {"x": 1}, "b": 1}).as_object().unwrap().clone();
        let overlay = json!({"a": {"y": 2}, "b": 2}).as_object().unwrap().clone();
        deep_merge(&mut base, &overlay);
        assert_eq!(Value::Object(base), json!({"a": {"x": 1, "y": 2}, "b": 2}));
    }
}
