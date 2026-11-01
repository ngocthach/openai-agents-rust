//! Ports of the Google-docstring cases of `tests/test_function_schema.py`: an `Args:` section that
//! directly follows another section's indented body (no blank line) still yields the
//! per-parameter descriptions and never leaks into the tool description.

#![cfg(feature = "macros")]

use openai_agents_sdk::Result;
use openai_agents_sdk::function_tool;
use serde_json::json;

/// Get the weather for a city.
///
/// Note:
///     Results are cached.
/// Args:
///     city: The city to get weather for.
///     units: Temperature units to use.
#[function_tool]
fn section_body_before_args(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

/// Get the weather for a city.
///
/// Note:
///     Results are cached.
///
/// Args:
///     city: The city to get weather for.
///     units: Temperature units to use.
#[function_tool]
fn section_body_before_args_blank_line(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

/// Python: `test_google_docstring_missing_blank_line_after_section_body`.
#[test]
fn an_args_section_right_after_another_section_body_still_describes_the_parameters() {
    let tool = section_body_before_args_tool::<()>();
    let properties = &tool.params_json_schema["properties"];

    assert_eq!(
        properties["city"]["description"],
        json!("The city to get weather for.")
    );
    assert_eq!(
        properties["units"]["description"],
        json!("Temperature units to use.")
    );
    assert!(!tool.description.contains("Args:"), "{}", tool.description);
}

/// Python: `test_google_docstring_after_section_body_matches_blank_line_form`.
#[test]
fn the_missing_blank_line_form_matches_the_well_formed_one() {
    let fixed = section_body_before_args_tool::<()>();
    let control = section_body_before_args_blank_line_tool::<()>();

    assert_eq!(fixed.description, control.description);
    assert_eq!(
        fixed.params_json_schema["properties"],
        control.params_json_schema["properties"]
    );
}
