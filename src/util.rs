//! Small internal helpers.

use std::future::Future;
use std::pin::Pin;

/// A boxed, sendable future.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Transform a string into a valid function-style name (`transfer_to_Billing Agent` ->
/// `transfer_to_Billing_Agent`). Port of `_transforms.transform_string_function_style`.
pub fn transform_string_function_style(name: &str) -> String {
    let replaced = name.replace(' ', "_");
    replaced
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
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
            "transfer_to_Billing_Agent"
        );
        assert_eq!(transform_string_function_style("a-b.c"), "a_b_c");
    }
}
