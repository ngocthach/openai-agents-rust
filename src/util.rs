//! Small internal helpers.

use std::future::Future;
use std::pin::Pin;

/// A boxed, sendable future.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Transform a string into a valid function-style name (`transfer_to_Billing Agent` ->
/// `transfer_to_billing_agent`). Port of `_transforms.transform_string_function_style`.
///
/// Non-alphanumeric characters become `_` and the result is lowercased, so the name matches
/// the Python SDK byte for byte on the wire.
pub fn transform_string_function_style(name: &str) -> String {
    transform_string_function_style_with(name, true)
}

/// [`transform_string_function_style`] with control over the whitespace warning.
///
/// With `warn_on_whitespace` set to `false`, a name that only differs by whitespace
/// substitution is transformed silently.
pub fn transform_string_function_style_with(name: &str, warn_on_whitespace: bool) -> String {
    let whitespace_normalized: String = name
        .chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect();
    let transformed: String = whitespace_normalized
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let final_name = transformed.to_lowercase();
    if transformed != name && (warn_on_whitespace || transformed != whitespace_normalized) {
        log::warn!(
            "Tool name {name:?} contains invalid characters for function calling and has been \
             transformed to {final_name:?}. Please use only letters, digits, and underscores to \
             avoid potential naming conflicts."
        );
    }
    final_name
}

/// Install the aws-lc-rs rustls crypto provider as the process default, unless one is already
/// installed. rustls panics when both `ring` and `aws-lc-rs` are compiled in and no default is
/// set, which happens as soon as any other dependency enables `ring`.
pub fn ensure_crypto_provider() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
}

/// A `reqwest` client built after [`ensure_crypto_provider`].
pub fn http_client() -> reqwest::Client {
    ensure_crypto_provider();
    reqwest::Client::new()
}

/// Current UTC time in RFC 3339 format with microseconds.
pub fn time_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

/// Generate a 32 hex character identifier.
pub fn gen_hex32() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Generate a 24 hex character identifier.
pub fn gen_hex24() -> String {
    gen_hex32()[..24].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_names() {
        assert_eq!(
            transform_string_function_style("transfer_to_Billing Agent"),
            "transfer_to_billing_agent"
        );
        assert_eq!(transform_string_function_style("a-b.c"), "a_b_c");
        assert_eq!(transform_string_function_style("A\tB"), "a_b");
    }
}
