//! Small helpers shared by the sandbox modules. Mirrors `agents.sandbox.util`.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Map, Value};

/// Encode bytes as standard base64 with padding, for `view_image` data URLs.
pub fn base64_encode(data: &[u8]) -> String {
    BASE64.encode(data)
}

/// Guess an image media type from a file extension, defaulting to `image/png`.
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

/// Truncate `text` to a token budget, keeping the tail, as the Python sandbox tools do.
///
/// Returns the truncated text and, when truncation happened, the approximate token count of the
/// original. The tail is kept because the newest output matters most for shell and PTY reads.
pub fn truncate_text_by_tokens(
    text: &str,
    max_output_tokens: Option<usize>,
) -> (String, Option<usize>) {
    let Some(max_tokens) = max_output_tokens else {
        return (text.to_string(), None);
    };
    let budget = approx_bytes_for_tokens(max_tokens);
    if text.len() <= budget {
        return (text.to_string(), None);
    }
    let original_tokens = approx_token_count(text);
    let total_lines = text.lines().count();
    let prefix = format!("Total output lines: {total_lines}\n\n");
    let remaining = budget.saturating_sub(prefix.len());
    // Keep the tail, starting at a character boundary.
    let start = text.len().saturating_sub(remaining);
    let start = (start..=text.len())
        .find(|i| text.is_char_boundary(*i))
        .unwrap_or(text.len());
    (format!("{prefix}{}", &text[start..]), Some(original_tokens))
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

        let (out, original) = truncate_text_by_tokens(&text, Some(10));
        assert!(out.starts_with("Total output lines: 1\n\n"));
        assert!(out.len() <= approx_bytes_for_tokens(10));
        assert_eq!(original, Some(100));
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
    fn merges_nested_objects() {
        let mut base = json!({"a": {"x": 1}, "b": 1}).as_object().unwrap().clone();
        let overlay = json!({"a": {"y": 2}, "b": 2}).as_object().unwrap().clone();
        deep_merge(&mut base, &overlay);
        assert_eq!(Value::Object(base), json!({"a": {"x": 1, "y": 2}, "b": 2}));
    }
}
