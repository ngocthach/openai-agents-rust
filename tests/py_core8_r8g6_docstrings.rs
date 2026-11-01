//! Port of `test_doc_parsing.py`: the doc comment of a `#[function_tool]` function yields the tool
//! description and the per-parameter descriptions in Google, NumPy and Sphinx styles, and the `Returns`
//! section never leaks into the description. Python's `test_instance_method` and
//! `test_classmethod` only differ by a bound receiver, which a Rust free function has none of.
//!
//! Python's `test_functions_are_ok` only calls the fixtures, which Rust's compiler already does.

#![cfg(feature = "macros")]

use openai_agents_sdk::Result;
use openai_agents_sdk::function_tool;
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

/// This is func_foo.
///
/// Args:
///     a: The first argument.
///     b: The second argument.
///
/// Returns:
///     A result
#[function_tool]
fn func_foo_google(a: i64, b: f64) -> Result<String> {
    Ok(format!("{a} {b}"))
}

/// This is func_foo.
///
/// Parameters
/// ----------
/// a: int
///     The first argument.
/// b: float
///     The second argument.
///
/// Returns
/// -------
/// str
///     A result
#[function_tool]
fn func_foo_numpy(a: i64, b: f64) -> Result<String> {
    Ok(format!("{a} {b}"))
}

/// This is func_foo.
///
/// :param a: The first argument.
/// :param b: The second argument.
/// :return: A result
#[function_tool]
fn func_foo_sphinx(a: i64, b: f64) -> Result<String> {
    Ok(format!("{a} {b}"))
}

fn assert_documented(tool: &FunctionTool<()>, name: &str, description: &str) {
    assert_eq!(tool.name, name);
    assert_eq!(tool.description, description);
    let properties = &tool.params_json_schema["properties"];
    assert_eq!(properties["a"]["description"], json!("The first argument."));
    assert_eq!(
        properties["b"]["description"],
        json!("The second argument.")
    );
}

/// Python: `test_auto_detection`.
#[test]
fn google_numpy_and_sphinx_docs_are_detected() {
    assert_documented(
        &func_foo_google_tool::<()>(),
        "func_foo_google",
        "This is func_foo.",
    );
    assert_documented(
        &func_foo_numpy_tool::<()>(),
        "func_foo_numpy",
        "This is func_foo.",
    );
    assert_documented(
        &func_foo_sphinx_tool::<()>(),
        "func_foo_sphinx",
        "This is func_foo.",
    );
}
