//! Small helpers shared by the sandbox modules. Mirrors `agents.sandbox.util`.

use serde_json::{Map, Value};

/// Encode bytes as standard base64 with padding.
pub fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
