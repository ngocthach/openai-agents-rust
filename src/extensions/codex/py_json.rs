//! JSON text byte-compatible with CPython's `json.dumps(..., ensure_ascii=True)`.
//!
//! The Codex extension sends its tool result to the model as a JSON string and sizes span
//! payloads against a character budget. Both are observable, so the bytes must match the Python
//! SDK: `serde_json` leaves non-ASCII characters unescaped and writes no space after `,` / `:`,
//! while `json.dumps` escapes every character outside `0x20..=0x7e` and defaults to the
//! `", "` / `": "` separators.

use serde_json::Value;

/// Separators of `json.dumps(value)` (the default).
pub(crate) const DEFAULT_SEPARATORS: (&str, &str) = (", ", ": ");
/// Separators of `json.dumps(value, separators=(",", ":"))`.
pub(crate) const COMPACT_SEPARATORS: (&str, &str) = (",", ":");

/// Append `text` as a JSON string literal, escaping like `ensure_ascii=True`.
pub(crate) fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let code = ch as u32;
                if code > 0xFFFF {
                    // Python writes astral characters as a UTF-16 surrogate pair.
                    let value = code - 0x1_0000;
                    let high = 0xD800 + (value >> 10);
                    let low = 0xDC00 + (value & 0x3FF);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                } else {
                    out.push_str(&format!("\\u{code:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Append `value` the way `json.dumps` would, with the given `(item, key)` separators.
pub(crate) fn write_value(out: &mut String, value: &Value, separators: (&str, &str)) {
    let (item_sep, key_sep) = separators;
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                write_value(out, item, separators);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                write_string(out, key);
                out.push_str(key_sep);
                write_value(out, item, separators);
            }
            out.push('}');
        }
    }
}

/// `json.dumps(value, separators=(",", ":"))`.
pub(crate) fn dumps_compact(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, COMPACT_SEPARATORS);
    out
}

/// The character count of the compact ASCII encoding. Port of `_json_char_size`.
pub(crate) fn char_size(value: &Value) -> usize {
    // `ensure_ascii` output is ASCII, so bytes and characters agree.
    dumps_compact(value).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `json.dumps(value)`.
    fn dumps(value: &Value) -> String {
        let mut out = String::new();
        write_value(&mut out, value, DEFAULT_SEPARATORS);
        out
    }

    #[test]
    fn escapes_non_ascii_like_python() {
        assert_eq!(dumps(&json!("café")), "\"caf\\u00e9\"");
        assert_eq!(dumps(&json!("日本")), "\"\\u65e5\\u672c\"");
        // Astral plane characters become a surrogate pair.
        assert_eq!(dumps(&json!("😀")), "\"\\ud83d\\ude00\"");
        // DEL is outside the printable range Python keeps verbatim.
        assert_eq!(dumps(&json!("\u{7f}")), "\"\\u007f\"");
    }

    #[test]
    fn escapes_control_characters_like_python() {
        assert_eq!(
            dumps(&json!("a\nb\tc\rd\"e\\f\u{8}g\u{c}h\u{1}")),
            "\"a\\nb\\tc\\rd\\\"e\\\\f\\bg\\fh\\u0001\""
        );
    }

    #[test]
    fn uses_pythons_default_separators() {
        assert_eq!(
            dumps(&json!({"a": 1, "b": [1, 2]})),
            r#"{"a": 1, "b": [1, 2]}"#
        );
        assert_eq!(
            dumps_compact(&json!({"a": 1, "b": [1, 2]})),
            r#"{"a":1,"b":[1,2]}"#
        );
    }

    #[test]
    fn sizes_values_in_ascii_characters() {
        assert_eq!(char_size(&json!(null)), 4);
        assert_eq!(char_size(&json!("é")), 8);
        assert_eq!(char_size(&json!({"status": "ok"})), 15);
    }
}
